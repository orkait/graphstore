from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from graphstore import GraphStore

_EMBEDDER_UNSET = object()

# ---------------------------------------------------------------------------
# Native-ingest helpers
# ---------------------------------------------------------------------------

def _corpus_id_from_node_id(node_id: str) -> str:
    """Strip :chunk:N or :section:N suffix produced by INGEST to get corpus_id."""
    for marker in (":chunk:", ":section:"):
        if marker in node_id:
            return node_id.split(marker, 1)[0]
    return node_id


def _safe_filename(corpus_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", corpus_id)


def _format_session_for_ingest(item: CorpusItem) -> str:
    """Format corpus text for graphstore INGEST.

    Wraps the conversation in a minimal markdown structure so the heading
    chunker produces one logical chunk for the whole session / turn.
    Plain text (no headings) falls through to chunk_by_paragraph, which
    splits on double newlines — also fine for conversations.
    """
    return item.text


def _ingest_corpus_item(gs: GraphStore, item: CorpusItem, ingest_dir: Path) -> None:
    """Write item text to a .txt file and INGEST through graphstore's native pipeline."""
    txt_path = ingest_dir / f"{_safe_filename(item.corpus_id)}.txt"
    txt_path.write_text(_format_session_for_ingest(item), encoding="utf-8")
    gs.execute(
        f'INGEST {_dsl_quote(str(txt_path))} '
        f'AS {_dsl_quote(item.corpus_id)} '
        f'KIND "conversation"'
    )


def _normalize_ranked_rows(rows: list[dict], item_by_id: dict[str, CorpusItem]) -> list[dict]:
    """Map chunk/section node IDs back to corpus IDs and dedupe by corpus_id (first-seen wins)."""
    seen: set[str] = set()
    normalized: list[dict] = []
    for row in rows:
        corpus_id = _corpus_id_from_node_id(row["corpus_id"])
        if corpus_id in seen:
            continue
        seen.add(corpus_id)
        item = item_by_id.get(corpus_id)
        if item is None:
            continue
        normalized.append({
            **row,
            "corpus_id": corpus_id,
            "session_id": item.session_id,
            "turn_id": item.turn_id,
            "text_preview": item.text[:200],
        })
    return normalized


@dataclass(slots=True)
class CorpusItem:
    corpus_id: str
    session_id: str
    turn_id: int | None
    timestamp: str
    text: str


def load_longmemeval(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("LongMemEval file must be a JSON array")
    return data


def session_id_from_corpus_id(corpus_id: str) -> str:
    if corpus_id.startswith("turn:"):
        _, session_id, _turn = corpus_id.split(":", 2)
        return session_id
    if corpus_id.startswith("session:"):
        return corpus_id.split(":", 1)[1]
    return corpus_id


def build_corpus(entry: dict, granularity: str) -> list[CorpusItem]:
    items: list[CorpusItem] = []
    seen_corpus_ids: set[str] = set()
    sessions = entry["haystack_sessions"]
    session_ids = entry["haystack_session_ids"]
    timestamps = entry["haystack_dates"]
    for session, session_id, timestamp in zip(sessions, session_ids, timestamps):
        if granularity == "session":
            corpus_id = f"session:{session_id}"
            if corpus_id in seen_corpus_ids:
                continue
            seen_corpus_ids.add(corpus_id)
            text = "\n".join(turn["content"] for turn in session)
            items.append(
                CorpusItem(
                    corpus_id=corpus_id,
                    session_id=session_id,
                    turn_id=None,
                    timestamp=timestamp,
                    text=text,
                )
            )
            continue
        if granularity != "turn":
            raise ValueError(f"Unsupported granularity: {granularity}")
        for turn_index, turn in enumerate(session):
            corpus_id = f"turn:{session_id}:{turn_index}"
            if corpus_id in seen_corpus_ids:
                continue
            seen_corpus_ids.add(corpus_id)
            items.append(
                CorpusItem(
                    corpus_id=corpus_id,
                    session_id=session_id,
                    turn_id=turn_index,
                    timestamp=timestamp,
                    text=turn["content"],
                )
            )
    return items


def _dcg(relevances: list[int], k: int) -> float:
    import math

    score = 0.0
    for rank, rel in enumerate(relevances[:k], start=1):
        if rel:
            score += rel / math.log2(rank + 1)
    return score


def evaluate_retrieval(ranked_ids: list[str], correct_ids: set[str], k: int) -> tuple[float, float, float]:
    if not correct_ids:
        return 0.0, 0.0, 0.0
    top_k = ranked_ids[:k]
    hits = [1 if ranked_id in correct_ids else 0 for ranked_id in top_k]
    hit_count = sum(hits)
    recall_any = 1.0 if hit_count > 0 else 0.0
    recall_all = hit_count / len(correct_ids)
    ideal = [1] * min(len(correct_ids), k)
    ndcg = _dcg(hits, k) / _dcg(ideal, k) if ideal else 0.0
    return recall_any, recall_all, ndcg


def is_abstention_entry(entry: dict) -> bool:
    return entry["question_id"].endswith("_abs") or entry["question_type"] == "abstention"


def iter_scored_entries(entries: list[dict], include_abstention: bool, limit: int | None) -> list[dict]:
    selected = []
    for entry in entries:
        if not include_abstention and is_abstention_entry(entry):
            continue
        selected.append(entry)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def _dsl_quote(text: str) -> str:
    return json.dumps(text)


def _normalize_lexical_query(text: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z]+", " ", text).lower().strip()
    tokens = [token for token in normalized.split() if len(token) > 1]
    if not tokens:
        return text
    deduped = list(dict.fromkeys(tokens))
    return " OR ".join(deduped)


def _register_benchmark_kind(gs: GraphStore) -> None:
    gs.execute(
        'SYS REGISTER NODE KIND "benchmark_memory" '
        'REQUIRED text:string, session_id:string, session_date:string, question_id:string '
        'OPTIONAL turn_id:int EMBED text'
    )


def _create_benchmark_node(gs: GraphStore, item: CorpusItem, question_id: str) -> None:
    fields = [
        f'CREATE NODE {_dsl_quote(item.corpus_id)}',
        'kind = "benchmark_memory"',
        f'text = {_dsl_quote(item.text)}',
        f'session_id = {_dsl_quote(item.session_id)}',
        f'session_date = {_dsl_quote(item.timestamp)}',
        f'question_id = {_dsl_quote(question_id)}',
    ]
    if item.turn_id is not None:
        fields.append(f"turn_id = {item.turn_id}")
    fields.append(f'DOCUMENT {_dsl_quote(item.text)}')
    gs.execute(" ".join(fields))

    slot = gs._store.id_to_slot[gs._store.string_table.intern(item.corpus_id)]
    gs._document_store.put_summary(slot, item.text[:2000])


def _result_rows(result, item_by_id: dict[str, CorpusItem]) -> list[dict]:
    rows = []
    for node in result.data:
        corpus_id = node["id"]
        # Native ingest returns chunk/section node IDs — strip the suffix to get corpus_id.
        item = item_by_id.get(corpus_id) or item_by_id.get(_corpus_id_from_node_id(corpus_id))
        if item is None:
            continue
        score = (
            node.get("_remember_score")
            or node.get("_similarity")
            or node.get("_bm25_score")
            or 0.0
        )
        rows.append(
            {
                "corpus_id": corpus_id,
                "session_id": item.session_id,
                "turn_id": item.turn_id,
                "score": float(score),
                "text_preview": item.text[:200],
            }
        )
    return rows


def _fuse_rows(*row_groups: list[dict], top_k: int) -> list[dict]:
    fused: dict[str, dict] = {}
    for rows in row_groups:
        for rank, row in enumerate(rows, start=1):
            current = fused.setdefault(row["corpus_id"], dict(row))
            current["score"] = current.get("score", 0.0) + 1.0 / (rank + 60)
    ranked = sorted(fused.values(), key=lambda row: row["score"], reverse=True)
    return ranked[:top_k]


def _run_retrieval(gs: GraphStore, item_by_id: dict[str, CorpusItem], question: str, mode: str, top_k: int) -> list[dict]:
    normalized_question = _normalize_lexical_query(question)
    if mode == "remember":
        result = gs.execute(f'REMEMBER {_dsl_quote(normalized_question)} LIMIT {top_k}')
        return _result_rows(result, item_by_id)
    if mode == "similar":
        result = gs.execute(f'SIMILAR TO {_dsl_quote(question)} LIMIT {top_k}')
        return _result_rows(result, item_by_id)
    if mode == "lexical":
        result = gs.execute(f'LEXICAL SEARCH {_dsl_quote(normalized_question)} LIMIT {top_k}')
        return _result_rows(result, item_by_id)
    if mode == "hybrid":
        similar = _result_rows(
            gs.execute(f'SIMILAR TO {_dsl_quote(question)} LIMIT {top_k * 3}'),
            item_by_id,
        )
        lexical = _result_rows(
            gs.execute(f'LEXICAL SEARCH {_dsl_quote(normalized_question)} LIMIT {top_k * 3}'),
            item_by_id,
        )
        return _fuse_rows(similar, lexical, top_k=top_k)
    raise ValueError(f"Unsupported mode: {mode}")


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def run_benchmark(
    dataset_path: str | Path,
    mode: str = "remember",
    granularity: str = "session",
    top_k: int = 10,
    limit: int | None = None,
    include_abstention: bool = False,
    out_path: str | Path | None = None,
    embedder=_EMBEDDER_UNSET,
    ingest_mode: str = "flat",
) -> dict:
    all_entries = load_longmemeval(dataset_path)
    entries = iter_scored_entries(
        all_entries,
        include_abstention=include_abstention,
        limit=limit,
    )
    metrics: dict[str, defaultdict[str, list[float]]] = {
        "session": defaultdict(list),
        "turn": defaultdict(list),
    }
    per_type: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    results_log = []
    started_at = time.time()

    for entry in entries:
        items = build_corpus(entry, granularity=granularity)
        item_by_id = {item.corpus_id: item for item in items}

        with tempfile.TemporaryDirectory(prefix="graphstore-longmemeval-") as tempdir:
            tempdir_path = Path(tempdir)
            if embedder is _EMBEDDER_UNSET:
                gs = GraphStore(path=tempdir)
            else:
                gs = GraphStore(path=tempdir, embedder=embedder)
            try:
                if ingest_mode == "native":
                    ingest_dir = tempdir_path / "ingest_files"
                    ingest_dir.mkdir()
                    for item in items:
                        _ingest_corpus_item(gs, item, ingest_dir)
                else:
                    _register_benchmark_kind(gs)
                    for item in items:
                        _create_benchmark_node(gs, item, question_id=entry["question_id"])
                ranked_rows = _run_retrieval(gs, item_by_id, entry["question"], mode=mode, top_k=top_k)
                if ingest_mode == "native":
                    ranked_rows = _normalize_ranked_rows(ranked_rows, item_by_id)
            finally:
                gs.close()

        ranked_ids = [row["corpus_id"] for row in ranked_rows]
        answer_session_ids = set(entry["answer_session_ids"])
        ranked_session_ids = [session_id_from_corpus_id(corpus_id) for corpus_id in ranked_ids]
        correct_turn_ids = {
            item.corpus_id for item in items if item.session_id in answer_session_ids
        }

        session_recall_any_5, session_recall_all_5, _ = evaluate_retrieval(ranked_session_ids, answer_session_ids, 5)
        session_recall_any_10, session_recall_all_10, session_ndcg_10 = evaluate_retrieval(
            ranked_session_ids,
            answer_session_ids,
            10,
        )
        turn_recall_any_5, turn_recall_all_5, _ = evaluate_retrieval(ranked_ids, correct_turn_ids, 5)
        turn_recall_any_10, turn_recall_all_10, turn_ndcg_10 = evaluate_retrieval(ranked_ids, correct_turn_ids, 10)

        metrics["session"]["recall_any@5"].append(session_recall_any_5)
        metrics["session"]["recall_all@5"].append(session_recall_all_5)
        metrics["session"]["recall_any@10"].append(session_recall_any_10)
        metrics["session"]["recall_all@10"].append(session_recall_all_10)
        metrics["session"]["ndcg_any@10"].append(session_ndcg_10)
        metrics["turn"]["recall_any@5"].append(turn_recall_any_5)
        metrics["turn"]["recall_all@5"].append(turn_recall_all_5)
        metrics["turn"]["recall_any@10"].append(turn_recall_any_10)
        metrics["turn"]["recall_all@10"].append(turn_recall_all_10)
        metrics["turn"]["ndcg_any@10"].append(turn_ndcg_10)

        qtype = entry["question_type"]
        per_type[qtype]["session_recall_any@10"].append(session_recall_any_10)
        per_type[qtype]["session_ndcg_any@10"].append(session_ndcg_10)

        results_log.append(
            {
                "question_id": entry["question_id"],
                "question_type": qtype,
                "question": entry["question"],
                "answer": entry["answer"],
                "answer_session_ids": entry["answer_session_ids"],
                "retrieval_results": {
                    "mode": mode,
                    "granularity": granularity,
                    "ranked_items": ranked_rows,
                    "metrics": {
                        "session": {
                            "recall_any@5": session_recall_any_5,
                            "recall_all@5": session_recall_all_5,
                            "recall_any@10": session_recall_any_10,
                            "recall_all@10": session_recall_all_10,
                            "ndcg_any@10": session_ndcg_10,
                        },
                        "turn": {
                            "recall_any@5": turn_recall_any_5,
                            "recall_all@5": turn_recall_all_5,
                            "recall_any@10": turn_recall_any_10,
                            "recall_all@10": turn_recall_all_10,
                            "ndcg_any@10": turn_ndcg_10,
                        },
                    },
                },
            }
        )

    summary = {
        "dataset_path": str(dataset_path),
        "mode": mode,
        "ingest_mode": ingest_mode,
        "granularity": granularity,
        "total_questions": len(all_entries),
        "scored_questions": len(entries),
        "skipped_questions": len(all_entries) - len(entries),
        "elapsed_seconds": round(time.time() - started_at, 3),
        "metrics": {
            axis: {name: _average(values) for name, values in axis_metrics.items()}
            for axis, axis_metrics in metrics.items()
        },
        "per_type": {
            qtype: {name: _average(values) for name, values in qtype_metrics.items()}
            for qtype, qtype_metrics in per_type.items()
        },
    }

    if out_path is not None:
        output_path = Path(out_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in results_log:
                handle.write(json.dumps(row) + "\n")
        output_path.with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    return summary


def _resolve_embedder(name: str | None):
    """Resolve --embedder CLI arg to an Embedder instance or sentinel."""
    if name is None or name == "default":
        return _EMBEDDER_UNSET  # graphstore picks model2vec M2V_base_output

    if name.startswith("model2vec:"):
        model_id = name[len("model2vec:"):]
        from graphstore.embedding.model2vec_embedder import Model2VecEmbedder
        print(f"[embedder] loading model2vec: {model_id}", flush=True)
        return Model2VecEmbedder(model_name=model_id)

    if name in ("embeddinggemma", "embeddinggemma-256"):
        from graphstore.registry.installer import load_installed_embedder, install_embedder, is_installed
        if not is_installed("embeddinggemma-300m"):
            print("[embedder] embeddinggemma-300m not installed — running: graphstore install-embedder embeddinggemma-300m", flush=True)
            install_embedder("embeddinggemma-300m")
        print("[embedder] loading embeddinggemma-300m (256d Matryoshka)", flush=True)
        return load_installed_embedder("embeddinggemma-300m", dims=256)

    if name == "embeddinggemma-768":
        from graphstore.registry.installer import load_installed_embedder, install_embedder, is_installed
        if not is_installed("embeddinggemma-300m"):
            print("[embedder] embeddinggemma-300m not installed — running: graphstore install-embedder embeddinggemma-300m", flush=True)
            install_embedder("embeddinggemma-300m")
        print("[embedder] loading embeddinggemma-300m (768d full)", flush=True)
        return load_installed_embedder("embeddinggemma-300m", dims=768)

    raise ValueError(
        f"Unknown embedder: {name!r}. "
        "Valid options: default, model2vec:<hf-model-id>, embeddinggemma, embeddinggemma-768"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GraphStore × LongMemEval benchmark")
    parser.add_argument("dataset")
    parser.add_argument("--mode", choices=["remember", "similar", "lexical", "hybrid"], default="remember")
    parser.add_argument("--ingest-mode", choices=["flat", "native"], default="native",
                        help="native: INGEST pipeline with auto-chunking (default); flat: CREATE NODE per session (baseline comparison only)")
    parser.add_argument("--embedder", default=None,
                        help=(
                            "Embedder to use. Options:\n"
                            "  default                        — model2vec M2V_base_output (256d)\n"
                            "  model2vec:<hf-id>              — any model2vec HuggingFace model\n"
                            "                                   e.g. model2vec:minishlab/potion-retrieval-32M\n"
                            "  embeddinggemma                 — EmbeddingGemma-300M ONNX (256d Matryoshka)\n"
                            "  embeddinggemma-768             — EmbeddingGemma-300M ONNX (768d full)\n"
                        ))
    parser.add_argument("--granularity", choices=["session", "turn"], default="session")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--include-abstention", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    embedder = _resolve_embedder(args.embedder)

    summary = run_benchmark(
        dataset_path=args.dataset,
        mode=args.mode,
        ingest_mode=args.ingest_mode,
        granularity=args.granularity,
        top_k=args.top_k,
        limit=args.limit,
        include_abstention=args.include_abstention,
        out_path=args.out,
        embedder=embedder,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
