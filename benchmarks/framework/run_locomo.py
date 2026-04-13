"""LoCoMo benchmark runner - official protocol.

Protocol (matches snap-research/locomo):
    - Ingest ALL sessions for a conversation ONCE
    - Query ALL QAs against that ingested state
    - Score with token-level F1 (Porter stemming, Counter-based)
    - Report per-category (official order: 4,1,2,3,5) and overall
    - Use ALL 10 conversations, ALL questions (no sampling)

Usage:
    python -m benchmarks.framework.run_locomo --data-path /path/to/locomo
    python -m benchmarks.framework.run_locomo --max-conversations 1 --max-questions 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

from .adapter import QueryContext, TimedOperation
from .datasets import load_locomo
from .llm_client import generate_answer, compute_f1

# Official LoCoMo category IDs
_CAT_TO_ID = {
    "single-hop": 1, "multi-hop": 2, "temporal": 3,
    "open-domain": 4, "adversarial": 5,
}

# Official reporting order
_CAT_ORDER = ["open-domain", "single-hop", "multi-hop", "temporal", "adversarial"]


def run_locomo(
    adapter,
    dataset,
    k: int = 5,
    max_questions: int | None = None,
    reranker=None,
    llm_workers: int = 8,
) -> dict:
    """Run LoCoMo: ingest once per conversation, query all QAs.

    Official protocol: all conversations, all questions, F1 with stemming.
    """

    # Group records by conversation
    conversations: dict[str, list] = defaultdict(list)
    sessions_by_conv: dict[str, list] = {}

    for rec in dataset.records:
        conv_id = rec.question.metadata.get("sample_id", "unknown")
        conversations[conv_id].append(rec)
        if conv_id not in sessions_by_conv:
            sessions_by_conv[conv_id] = rec.sessions

    results_by_category: dict[str, list[float]] = defaultdict(list)
    all_f1: list[float] = []
    all_details: list[dict] = []
    total_ingest_ms = 0
    total_query_ms = 0
    q_count = 0

    n_convs = len(conversations)
    for conv_idx, (conv_id, records) in enumerate(conversations.items()):
        qas = records
        if max_questions is not None:
            qas = qas[:max_questions]

        sessions = sessions_by_conv[conv_id]
        print(f"\n[{conv_idx+1}/{n_convs}] [{conv_id}] Ingesting {len(sessions)} sessions, {len(qas)} questions...")

        # Ingest ONCE per conversation
        adapter.reset()
        t0 = time.perf_counter()
        has_ingest_done = hasattr(adapter, "ingest_done")
        for si, sess in enumerate(sessions):
            adapter.ingest(sess)
            if (si + 1) % 5 == 0:
                print(f"  ingest {si+1}/{len(sessions)} sessions")

        if has_ingest_done:
            adapter.ingest_done()
        ingest_ms = (time.perf_counter() - t0) * 1000
        total_ingest_ms += ingest_ms
        print(f"  Ingested in {ingest_ms:.0f}ms")

        # Phase 1: Retrieval (serial - GraphStore is single-writer)
        print(f"[{conv_id}] Retrieving {len(qas)} questions...")
        has_query_ctx = hasattr(adapter, "query_with_context")

        retrieval_results = []
        for i, rec in enumerate(qas):
            with TimedOperation() as t:
                if has_query_ctx:
                    ctx = QueryContext(
                        question=rec.question.question,
                        category=rec.question.category,
                    )
                    qres = adapter.query_with_context(ctx, k=k)
                else:
                    qres = adapter.query(rec.question.question, k=k)

                # Rerank if provided
                if reranker and len(qres.retrieved_memories) > k:
                    scores = reranker.score(rec.question.question, qres.retrieved_memories)
                    ranked = sorted(zip(scores, qres.retrieved_memories), reverse=True)
                    qres.retrieved_memories = [t for _, t in ranked[:k]]

            total_query_ms += t.elapsed_ms
            retrieval_results.append(qres)

        print(f"[{conv_id}] Retrieval done ({total_query_ms:.0f}ms). Generating {len(qas)} answers ({llm_workers} concurrent)...")

        # Phase 2: LLM answer generation (concurrent)
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _generate_one(idx):
            rec = qas[idx]
            qres = retrieval_results[idx]
            answer = generate_answer(rec.question.question, qres.retrieved_memories)
            return idx, answer

        answers = [None] * len(qas)
        with ThreadPoolExecutor(max_workers=llm_workers) as pool:
            futures = {pool.submit(_generate_one, i): i for i in range(len(qas))}
            done_count = 0
            for future in as_completed(futures):
                idx, answer = future.result()
                answers[idx] = answer
                done_count += 1
                if done_count % 20 == 0:
                    print(f"  [{conv_id}] LLM {done_count}/{len(qas)}")

        # Phase 3: Score (official F1 with category-aware handling)
        for i, rec in enumerate(qas):
            answer = answers[i] or ""
            gold = rec.question.gold_answers[0] if rec.question.gold_answers else ""
            cat_id = _CAT_TO_ID.get(rec.question.category)
            f1 = compute_f1(answer, gold, category=cat_id)

            all_f1.append(f1)
            results_by_category[rec.question.category].append(f1)
            all_details.append({
                "conversation": conv_id,
                "question": rec.question.question,
                "gold": gold,
                "answer": answer,
                "f1": round(f1, 4),
                "category": rec.question.category,
                "category_id": cat_id,
                "retrieved": retrieval_results[i].retrieved_memories[:3],
            })
            q_count += 1

        conv_f1 = sum(all_f1[-len(qas):]) / len(qas) if qas else 0
        print(f"  [{conv_id}] {len(qas)} Qs, conv_f1={conv_f1:.3f}, running_avg={sum(all_f1)/len(all_f1):.3f}")

    # Summary - official format
    overall_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0

    by_category = {}
    for cat in _CAT_ORDER:
        scores = results_by_category.get(cat, [])
        if scores:
            by_category[cat] = {
                "n": len(scores),
                "f1": round(sum(scores) / len(scores), 4),
                "category_id": _CAT_TO_ID.get(cat),
            }

    summary = {
        "benchmark": "LoCoMo",
        "n_conversations": len(conversations),
        "n_questions": len(all_f1),
        "overall_f1": round(overall_f1, 4),
        "by_category": by_category,
        "ingest_ms": round(total_ingest_ms, 1),
        "query_avg_ms": round(total_query_ms / max(q_count, 1), 1),
        "adapter": adapter.name,
    }

    return summary, all_details


def main():
    parser = argparse.ArgumentParser(prog="run_locomo")
    parser.add_argument("--data-path", default="/tmp/locomo")
    parser.add_argument("--max-conversations", type=int, default=None)
    parser.add_argument("--max-questions", type=int, default=None,
                        help="max questions PER conversation")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--embedder", default="model2vec")
    parser.add_argument("--out-dir", default="benchmarks/framework/results")
    args = parser.parse_args()

    from .adapters.graphstore_ import GraphStoreAdapter

    ds = load_locomo(args.data_path, max_conversations=args.max_conversations)
    print(f"LoCoMo: {len(ds)} total QA pairs, {len(set(r.question.metadata.get('sample_id') for r in ds.records))} conversations")

    # Use config.py defaults - no hardcoded benchmark overrides
    config = {"ceiling_mb": 512}
    if ":" in args.embedder:
        backend, model = args.embedder.split(":", 1)
        config["embedder"] = backend
        config["embedder_model"] = model
    else:
        config["embedder"] = args.embedder
    adapter = GraphStoreAdapter(config=config)

    summary, details = run_locomo(adapter, ds, k=args.k, max_questions=args.max_questions)

    print(f"\n{'='*60}")
    print(f"LOCOMO RESULTS")
    print(f"  System:      {summary['adapter']}")
    print(f"  Convs:       {summary['n_conversations']}")
    print(f"  Questions:   {summary['n_questions']}")
    print(f"  Overall F1:  {summary['overall_f1']:.4f}")
    print(f"  By category (official order):")
    for cat, v in summary["by_category"].items():
        print(f"    {cat:<20} (cat-{v['category_id']}) n={v['n']:<4} f1={v['f1']:.4f}")
    print(f"  Ingest:      {summary['ingest_ms']:.0f}ms total")
    print(f"  Query avg:   {summary['query_avg_ms']:.1f}ms")
    print(f"{'='*60}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"locomo_{summary['adapter']}.json"
    out_path.write_text(json.dumps({"summary": summary, "details": details}, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
