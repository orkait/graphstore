"""GraphStore adapter, skill-compliant.

Implements every rule from skills/graphstore-ingestion/SKILL.md:
    - Schema first (SYS REGISTER NODE KIND) with EMBED on message only
    - deferred_embeddings() per session
    - put_summary() per message so REMEMBER's BM25 leg actually works (G2)
    - No EMBED on entity nodes (G5)
    - No __updated_at__ override (G3)
    - Entity graph via regex extraction, linked via "mentions" edges
    - Per-category query dispatch that combines REMEMBER + RECALL (G1)
    - WHERE kind = "message" on REMEMBER so entities/sessions don't pollute
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from graphstore import GraphStore, __version__ as _GS_VERSION
from graphstore.core.errors import NodeExists

from ..adapter import QueryContext, QueryResult, Session, TimedOperation


_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z0-9_-]{2,}(?:\s+[A-Z][a-zA-Z0-9_-]{2,}){0,3}\b")
_SLUG_RE = re.compile(r"[^a-zA-Z0-9_]+")


def _escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", " ")
        .replace("\r", " ")
    )


def _slug(text: str) -> str:
    return _SLUG_RE.sub("_", text.lower()).strip("_")[:40]


def _extract_entities(text: str) -> list[str]:
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for raw in _ENTITY_RE.findall(text):
        norm = raw.strip()
        if len(norm) < 3:
            continue
        key = norm.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(norm)
    return out[:6]


def _build_embedder(config: dict[str, Any]):
    name = (config.get("embedder") or "model2vec").lower()
    gpu = bool(config.get("embedder_gpu"))
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else None

    if name in ("model2vec", "default"):
        return "default"
    if name == "fastembed":
        from graphstore.embedding.fastembed_embedder import FastEmbedEmbedder

        return FastEmbedEmbedder(
            model_name=config.get("embedder_model", "BAAI/bge-small-en-v1.5"),
            cache_dir=config.get("cache_dir"),
            threads=config.get("embedder_threads"),
        )
    if name == "onnx":
        from graphstore.embedding.onnx_hf_embedder import OnnxHFEmbedder

        model_dir = config.get("embedder_model_dir")
        if not model_dir:
            raise ValueError("embedder=onnx requires embedder_model_dir")
        return OnnxHFEmbedder(
            model_dir=model_dir,
            output_dims=config.get("embedder_output_dims"),
            max_length=int(config.get("embedder_max_length", 512)),
            pooling_mode=config.get("embedder_pooling", "mean"),
            providers=providers,
            gpu_mem_limit=config.get("embedder_gpu_mem_limit"),
        )
    if name == "installed":
        from graphstore.registry.installer import load_installed_embedder, set_cache_dir
        cache = config.get("embedder_cache_dir")
        if cache:
            set_cache_dir(cache)
        model_name = config.get("embedder_model")
        if not model_name:
            raise ValueError("embedder=installed requires embedder_model (e.g. 'harrier-oss-v1-0.6b')")
        return load_installed_embedder(
            model_name,
            dims=config.get("embedder_output_dims"),
            providers=providers,
        )
    if name == "gguf":
        from graphstore.embedding.llamacpp_embedder import LlamaCppEmbedder
        gguf_path = config.get("embedder_gguf_path")
        if not gguf_path:
            raise ValueError("embedder=gguf requires embedder_gguf_path")
        return LlamaCppEmbedder(
            model_path=gguf_path,
            n_ctx=int(config.get("embedder_max_length", 2048)),
            n_gpu_layers=int(config.get("embedder_gpu_layers", 0)),
            output_dims=config.get("embedder_output_dims"),
            query_prefix=config.get("embedder_query_prefix", ""),
            doc_prefix_template=config.get("embedder_doc_prefix", ""),
        )
    raise ValueError(f"unknown embedder: {name!r}")


def _build_reranker(config: dict[str, Any]):
    backend = config.get("reranker", "").lower()
    if backend == "flashrank":
        from graphstore.embedding.reranker import FlashRankReranker
        return FlashRankReranker(
            model_name=config.get("reranker_model", "rank-T5-flan"),
            max_length=int(config.get("reranker_max_length", 512)),
        )
    if backend == "onnx":
        reranker_dir = config.get("reranker_model_dir")
        if not reranker_dir:
            return None
        from graphstore.embedding.reranker import OnnxReranker
        return OnnxReranker(
            model_dir=reranker_dir,
            onnx_file=config.get("reranker_onnx_file", "onnx/model_int8.onnx"),
            max_length=int(config.get("reranker_max_length", 512)),
        )
    return None


class GraphStoreAdapter:
    name = "graphstore"
    version = _GS_VERSION

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._tmpdir: Path | None = None
        self._gs: GraphStore | None = None
        self._embed_batch_size: int = int(self.config.get("embed_batch_size", 128))
        self._entity_extraction: bool = bool(self.config.get("entities", True))
        self._populate_fts: bool = bool(self.config.get("populate_fts", True))
        self._reranker = _build_reranker(self.config)
        self._embedder = _build_embedder(self.config)
        emb_name = (self.config.get("embedder") or "model2vec").lower()
        if emb_name == "fastembed":
            short = self.config.get("embedder_model", "bge-small").split("/")[-1]
            self.name = f"graphstore-skill-fastembed-{short}"
        elif emb_name == "onnx":
            short = Path(self.config.get("embedder_model_dir", "onnx")).name
            self.name = f"graphstore-skill-onnx-{short}"
        elif emb_name == "installed":
            short = (self.config.get("embedder_model") or "installed").lower()
            self.name = f"graphstore-skill-{short}"
        else:
            self.name = "graphstore-skill-model2vec"

    def reset(self) -> None:
        self.close()
        self._tmpdir = Path(tempfile.mkdtemp(prefix="gs_bench_"))
        # Build constructor kwargs from any CLI overrides (only explicitly set values)
        gs_kwargs: dict[str, Any] = {
            "path": str(self._tmpdir),
            "ceiling_mb": self.config.get("ceiling_mb", 4096),
            "queued": False,
            "embedder": self._embedder,
        }
        for key in ("remember_weights", "search_oversample", "recall_decay",
                     "similarity_threshold", "retrieval_depth", "recall_depth",
                     "max_query_entities", "recency_boost_k"):
            val = self.config.get(key)
            if val is not None:
                gs_kwargs[key] = val
        self._gs = GraphStore(**gs_kwargs)
        self._gs.execute(
            'SYS REGISTER NODE KIND "message" '
            'REQUIRED session:string, role:string, content:string '
            'OPTIONAL position:int '
            'EMBED content'
        )
        self._gs.execute(
            'SYS REGISTER NODE KIND "session" '
            'REQUIRED session_id:string '
            'OPTIONAL position:int, msg_count:int'
        )
        if self._entity_extraction:
            self._gs.execute(
                'SYS REGISTER NODE KIND "entity" REQUIRED name:string'
            )

    def ingest(self, session: Session) -> float:
        if self._gs is None:
            raise RuntimeError("reset() must be called first")

        n = len(session.messages)
        if n == 0:
            return 0.0

        sid = _escape(session.session_id)
        store = self._gs._store
        intern = store.string_table.intern
        doc_store = self._gs._document_store
        populate_fts = self._populate_fts and doc_store is not None

        with TimedOperation() as t:
            with self._gs.deferred_embeddings(batch_size=self._embed_batch_size):
                sess_node_id = f"sess:{session.session_id}"
                self._gs.execute(
                    f'CREATE NODE "{sess_node_id}" kind = "session" '
                    f'session_id = "{sid}" '
                    f'position = {int(session.metadata.get("position", 0))} '
                    f'msg_count = {n}'
                )

                entity_seen: set[str] = set()
                for i, msg in enumerate(session.messages):
                    msg_id = f"{session.session_id}:msg{i}"
                    content_raw = msg.content
                    content = _escape(content_raw)
                    role = _escape(msg.role)
                    self._gs.execute(
                        f'CREATE NODE "{msg_id}" kind = "message" '
                        f'session = "{sid}" role = "{role}" '
                        f'content = "{content}" '
                        f'position = {i}'
                    )
                    if populate_fts:
                        try:
                            slot = store.id_to_slot[intern(msg_id)]
                            doc_store.put_summary(slot, content_raw, doc_slot=0, chunk_index=i)
                        except Exception:
                            pass
                    self._gs.execute(
                        f'CREATE EDGE "{sess_node_id}" -> "{msg_id}" kind = "has_message"'
                    )

                    if self._entity_extraction:
                        for ent_name in _extract_entities(content_raw):
                            ent_slug = _slug(ent_name)
                            if not ent_slug:
                                continue
                            ent_id = f"ent:{ent_slug}"
                            if ent_id not in entity_seen:
                                try:
                                    self._gs.execute(
                                        f'CREATE NODE "{ent_id}" kind = "entity" '
                                        f'name = "{_escape(ent_name)}"'
                                    )
                                except NodeExists:
                                    pass
                                entity_seen.add(ent_id)
                            try:
                                self._gs.execute(
                                    f'CREATE EDGE "{msg_id}" -> "{ent_id}" kind = "mentions"'
                                )
                            except Exception:
                                pass

                for i in range(n - 1):
                    a = f"{session.session_id}:msg{i}"
                    b = f"{session.session_id}:msg{i + 1}"
                    self._gs.execute(f'CREATE EDGE "{a}" -> "{b}" kind = "next"')

        return t.elapsed_ms / 1000.0

    def query_with_context(self, ctx: QueryContext, k: int = 5) -> QueryResult:
        if self._gs is None:
            raise RuntimeError("reset() must be called first")

        category = ctx.category or ""
        with TimedOperation() as t:
            retrieved, raw = self._dispatch(ctx.question, category, k)
        return QueryResult(
            retrieved_memories=retrieved,
            elapsed_ms=t.elapsed_ms,
            raw=raw,
        )

    def query(self, question: str, k: int = 5) -> QueryResult:
        return self.query_with_context(QueryContext(question=question), k=k)

    def _cfg(self, attr: str):
        """Read a config value from the GraphStore executor (single source of truth)."""
        return getattr(self._gs._executor, f"_{attr}")

    def _dispatch(self, question: str, category: str, k: int) -> tuple[list[str], Any]:
        strategy = self._cfg("retrieval_strategy")
        handler = self._STRATEGIES.get(strategy, self._strategy_full)
        return handler(self, question, k)

    def _strategy_remember(self, question: str, k: int) -> tuple[list[str], Any]:
        """Hybrid only. Fast, no graph overhead."""
        q = _escape(question)
        depth = self._cfg("retrieval_depth")
        r = self._gs.execute(
            f'REMEMBER "{q}" LIMIT {k * depth} WHERE kind = "message"'
        )
        return self._texts(r.data)[:k], r.data

    def _strategy_remember_graph(self, question: str, k: int) -> tuple[list[str], Any]:
        """Hybrid + entity graph traversal. Good for cross-session."""
        q = _escape(question)
        depth = self._cfg("retrieval_depth")
        primary = self._gs.execute(
            f'REMEMBER "{q}" LIMIT {k * depth} WHERE kind = "message"'
        )
        merged = self._texts(primary.data)

        if self._entity_extraction:
            max_ents = self._cfg("max_query_entities")
            recall_d = self._cfg("recall_depth")
            for ent_name in _extract_entities(question)[:max_ents]:
                ent_id = f"ent:{_slug(ent_name)}"
                try:
                    rec = self._gs.execute(
                        f'RECALL FROM "{ent_id}" DEPTH {recall_d} LIMIT {k}'
                    )
                    for text in self._texts(rec.data):
                        if text not in merged:
                            merged.append(text)
                except Exception:
                    pass

        return merged[:k], primary.data

    def _strategy_remember_recency(self, question: str, k: int) -> tuple[list[str], Any]:
        """Hybrid + recency boost. Good for knowledge updates."""
        q = _escape(question)
        depth = self._cfg("retrieval_depth")
        primary = self._gs.execute(
            f'REMEMBER "{q}" LIMIT {k * depth} WHERE kind = "message"'
        )
        merged = self._texts(primary.data)

        try:
            recency_k = k * self._cfg("recency_boost_k")
            recent = self._gs.execute(
                f'NODES WHERE kind = "message" '
                f'ORDER BY __updated_at__ DESC LIMIT {recency_k}'
            )
            for text in self._texts(recent.data):
                if text not in merged:
                    merged.append(text)
        except Exception:
            pass

        return merged[:k], primary.data

    def _strategy_full(self, question: str, k: int) -> tuple[list[str], Any]:
        """Hybrid + graph + recency. All signals, no category routing."""
        q = _escape(question)
        depth = self._cfg("retrieval_depth")
        primary = self._gs.execute(
            f'REMEMBER "{q}" LIMIT {k * depth} WHERE kind = "message"'
        )
        merged = self._texts(primary.data)

        if self._entity_extraction:
            max_ents = self._cfg("max_query_entities")
            recall_d = self._cfg("recall_depth")
            for ent_name in _extract_entities(question)[:max_ents]:
                ent_id = f"ent:{_slug(ent_name)}"
                try:
                    rec = self._gs.execute(
                        f'RECALL FROM "{ent_id}" DEPTH {recall_d} LIMIT {k}'
                    )
                    for text in self._texts(rec.data):
                        if text not in merged:
                            merged.append(text)
                except Exception:
                    pass

        try:
            recency_k = k * self._cfg("recency_boost_k")
            recent = self._gs.execute(
                f'NODES WHERE kind = "message" '
                f'ORDER BY __updated_at__ DESC LIMIT {recency_k}'
            )
            for text in self._texts(recent.data):
                if text not in merged:
                    merged.append(text)
        except Exception:
            pass

        return merged[:k], primary.data

    def _strategy_lexical_boost(self, question: str, k: int) -> tuple[list[str], Any]:
        """Hybrid + extra lexical search. Good for keyword-heavy queries."""
        q = _escape(question)
        depth = self._cfg("retrieval_depth")
        primary = self._gs.execute(
            f'REMEMBER "{q}" LIMIT {k * depth} WHERE kind = "message"'
        )
        merged = self._texts(primary.data)

        try:
            lexical = self._gs.execute(
                f'LEXICAL SEARCH "{q}" LIMIT {k * 2}'
            )
            for text in self._texts(lexical.data):
                if text not in merged:
                    merged.append(text)
        except Exception:
            pass

        return merged[:k], primary.data

    def _rerank(self, question: str, texts: list[str], k: int) -> list[str]:
        """Rerank texts by cross-encoder relevance. Falls back to truncation if no reranker."""
        if not self._reranker or len(texts) <= k:
            return texts[:k]
        scores = self._reranker.score(question, texts)
        ranked = sorted(zip(scores, texts), reverse=True)
        return [t for _, t in ranked[:k]]

    def _strategy_remember_rerank(self, question: str, k: int) -> tuple[list[str], Any]:
        """Hybrid retrieval + cross-encoder rerank. Research-backed best pattern."""
        q = _escape(question)
        depth = self._cfg("retrieval_depth")
        primary = self._gs.execute(
            f'REMEMBER "{q}" LIMIT {k * depth} WHERE kind = "message"'
        )
        candidates = self._texts(primary.data)
        reranked = self._rerank(question, candidates, k)
        return reranked, primary.data

    def _strategy_full_rerank(self, question: str, k: int) -> tuple[list[str], Any]:
        """All signals + cross-encoder rerank. Maximum accuracy."""
        texts, raw = self._strategy_full(question, k * 2)
        reranked = self._rerank(question, texts, k)
        return reranked, raw

    _STRATEGIES = {
        "remember": _strategy_remember,
        "remember_graph": _strategy_remember_graph,
        "remember_recency": _strategy_remember_recency,
        "remember_lexical": _strategy_lexical_boost,
        "remember_rerank": _strategy_remember_rerank,
        "full": _strategy_full,
        "full_rerank": _strategy_full_rerank,
    }

    @staticmethod
    def _texts(rows: Any) -> list[str]:
        if not rows:
            return []
        out: list[str] = []
        for node in rows:
            text = node.get("content") or node.get("name") or node.get("summary") or ""
            if text:
                out.append(text)
        return out

    def close(self) -> None:
        if self._gs is not None:
            try:
                self._gs.close()
            except Exception:
                pass
            self._gs = None
        if self._tmpdir and self._tmpdir.exists():
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._tmpdir = None
