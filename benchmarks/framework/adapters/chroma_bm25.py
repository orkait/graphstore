"""Chroma + BM25 baseline adapter.

Vanilla hybrid retrieval control. This is what every "just use a vector
store" project does. Uses chromadb for dense retrieval, rank-bm25 for
sparse, and Reciprocal Rank Fusion to combine. Same fastembed embedder
as the graphstore adapter for true apples-to-apples.

No LLM. No graph. No reranker. No per-category routing. Just
hybrid(vector, bm25) with RRF fusion.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ..adapter import QueryContext, QueryResult, Session, TimedOperation


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "") if len(t) > 1]


class ChromaBM25Adapter:
    name = "chroma-bm25"
    version = "vanilla-hybrid"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._tmpdir: Path | None = None
        self._client = None
        self._collection = None
        self._bm25 = None
        self._doc_texts: list[str] = []
        self._doc_meta: list[dict] = []
        self._emb_fn = None
        self._embed_model = self.config.get(
            "embedder_model", "BAAI/bge-small-en-v1.5"
        )
        short = self._embed_model.split("/")[-1]
        self.name = f"chroma-bm25-{short}"

    def _build_embedder(self):
        try:
            import chromadb.utils.embedding_functions as ef

            if hasattr(ef, "FastEmbedEmbeddingFunction"):
                return ef.FastEmbedEmbeddingFunction(
                    model_name=self._embed_model
                )
        except Exception:
            pass

        from fastembed import TextEmbedding

        _fe = TextEmbedding(model_name=self._embed_model)
        _model_name = self._embed_model

        class _FastEmbedFn:
            def __call__(self, input):
                if isinstance(input, str):
                    input = [input]
                return [list(v) for v in _fe.embed(list(input))]

            def embed_query(self, input):
                if isinstance(input, str):
                    input = [input]
                return [list(v) for v in _fe.query_embed(list(input))]

            def embed_documents(self, input):
                if isinstance(input, str):
                    input = [input]
                return [list(v) for v in _fe.embed(list(input))]

            def name(self) -> str:
                return f"fastembed_{_model_name.replace('/', '_')}"

        return _FastEmbedFn()

    def reset(self) -> None:
        self.close()
        import chromadb

        self._tmpdir = Path(tempfile.mkdtemp(prefix="chroma_bench_"))
        self._client = chromadb.PersistentClient(path=str(self._tmpdir))
        self._emb_fn = self._build_embedder()
        self._collection = self._client.get_or_create_collection(
            name="bench",
            embedding_function=self._emb_fn,
        )
        self._doc_texts = []
        self._doc_meta = []
        self._bm25 = None

    def ingest(self, session: Session) -> float:
        if self._collection is None:
            raise RuntimeError("reset() first")
        n = len(session.messages)
        if n == 0:
            return 0.0

        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict] = []
        for i, msg in enumerate(session.messages):
            mid = f"{session.session_id}:msg{i}"
            ids.append(mid)
            docs.append(msg.content)
            metas.append({
                "session": session.session_id,
                "role": msg.role,
                "position": i,
            })

        with TimedOperation() as t:
            self._collection.add(ids=ids, documents=docs, metadatas=metas)
            for txt, meta, mid in zip(docs, metas, ids):
                self._doc_texts.append(txt)
                self._doc_meta.append({**meta, "id": mid})
        return t.elapsed_ms / 1000.0

    def _build_bm25(self):
        from rank_bm25 import BM25Okapi

        tokenized = [_tokenize(t) for t in self._doc_texts]
        self._bm25 = BM25Okapi(tokenized)

    def ingest_done(self, record_metadata: dict | None = None) -> None:
        if self._doc_texts:
            self._build_bm25()

    def query_with_context(self, ctx: QueryContext, k: int = 5) -> QueryResult:
        return self._query(ctx.question, k)

    def query(self, question: str, k: int = 5) -> QueryResult:
        return self._query(question, k)

    def _query(self, question: str, k: int = 5) -> QueryResult:
        if self._collection is None:
            raise RuntimeError("reset() first")

        with TimedOperation() as t:
            over = k * 4
            # Dense retrieval
            dense = self._collection.query(
                query_texts=[question], n_results=over
            )
            dense_ids = (dense.get("ids") or [[]])[0]
            dense_docs = (dense.get("documents") or [[]])[0]
            dense_metas = (dense.get("metadatas") or [[]])[0]

            dense_rank: dict[str, int] = {}
            dense_map: dict[str, tuple[str, dict]] = {}
            for rank, (did, doc, meta) in enumerate(
                zip(dense_ids, dense_docs, dense_metas)
            ):
                dense_rank[did] = rank
                dense_map[did] = (doc, meta or {})

            # Sparse retrieval
            sparse_rank: dict[str, int] = {}
            sparse_map: dict[str, tuple[str, dict]] = {}
            if self._bm25 is not None:
                scores = self._bm25.get_scores(_tokenize(question))
                ranked_idx = sorted(
                    range(len(scores)), key=lambda i: scores[i], reverse=True
                )[:over]
                for rank, idx in enumerate(ranked_idx):
                    meta = self._doc_meta[idx]
                    mid = meta["id"]
                    sparse_rank[mid] = rank
                    sparse_map[mid] = (self._doc_texts[idx], meta)

            # Reciprocal Rank Fusion
            rrf: dict[str, float] = {}
            K_RRF = 60.0
            for mid, r in dense_rank.items():
                rrf[mid] = rrf.get(mid, 0.0) + 1.0 / (K_RRF + r)
            for mid, r in sparse_rank.items():
                rrf[mid] = rrf.get(mid, 0.0) + 1.0 / (K_RRF + r)

            fused = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:k]
            retrieved: list[str] = []
            raw: list[dict] = []
            for mid, _score in fused:
                if mid in dense_map:
                    doc, meta = dense_map[mid]
                else:
                    doc, meta = sparse_map[mid]
                retrieved.append(doc)
                raw.append({"id": mid, "content": doc, **(meta or {})})

        return QueryResult(
            retrieved_memories=retrieved,
            elapsed_ms=t.elapsed_ms,
            raw=raw,
        )

    def close(self) -> None:
        if self._client is not None:
            try:
                # chromadb doesn't need explicit close
                self._client = None
                self._collection = None
            except Exception:
                pass
        if self._tmpdir and self._tmpdir.exists():
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._tmpdir = None
        self._bm25 = None
        self._doc_texts = []
        self._doc_meta = []
