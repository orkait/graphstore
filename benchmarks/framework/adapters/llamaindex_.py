"""LlamaIndex VectorStoreIndex baseline adapter.

Uses llama-index-core's VectorStoreIndex with a SimpleVectorStore and a
fastembed-backed embedder. No LLM — retrieval only. Represents "just use
LlamaIndex's default memory/retrieval pattern."

Same bge-small embedder as graphstore and chroma for apples-to-apples.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from ..adapter import QueryContext, QueryResult, Session, TimedOperation


class LlamaIndexAdapter:
    name = "llamaindex"
    version = "vectorstore-default"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._tmpdir: Path | None = None
        self._index = None
        self._retriever = None
        self._embed_model_name = self.config.get(
            "embedder_model", "BAAI/bge-small-en-v1.5"
        )
        self._embed_model = None
        short = self._embed_model_name.split("/")[-1]
        self.name = f"llamaindex-{short}"

    def _build_embed_model(self):
        from llama_index.embeddings.fastembed import FastEmbedEmbedding

        return FastEmbedEmbedding(
            model_name=self._embed_model_name,
            cache_dir=self.config.get("cache_dir"),
        )

    def reset(self) -> None:
        self.close()
        from llama_index.core import VectorStoreIndex, Settings
        from llama_index.core.schema import TextNode

        self._tmpdir = Path(tempfile.mkdtemp(prefix="li_bench_"))
        if self._embed_model is None:
            self._embed_model = self._build_embed_model()
        Settings.embed_model = self._embed_model
        Settings.llm = None
        self._pending_nodes: list = []
        self._TextNode = TextNode
        self._VectorStoreIndex = VectorStoreIndex

    def ingest(self, session: Session) -> float:
        n = len(session.messages)
        if n == 0:
            return 0.0
        with TimedOperation() as t:
            for i, msg in enumerate(session.messages):
                node_id = f"{session.session_id}:msg{i}"
                self._pending_nodes.append(
                    self._TextNode(
                        id_=node_id,
                        text=msg.content,
                        metadata={
                            "session": session.session_id,
                            "role": msg.role,
                            "position": i,
                        },
                    )
                )
        return t.elapsed_ms / 1000.0

    def ingest_done(self, record_metadata: dict | None = None) -> None:
        if not self._pending_nodes:
            return
        self._index = self._VectorStoreIndex(
            self._pending_nodes,
            embed_model=self._embed_model,
        )
        self._retriever = self._index.as_retriever(similarity_top_k=5)

    def query_with_context(self, ctx: QueryContext, k: int = 5) -> QueryResult:
        return self._query(ctx.question, k)

    def query(self, question: str, k: int = 5) -> QueryResult:
        return self._query(question, k)

    def _query(self, question: str, k: int = 5) -> QueryResult:
        if self._retriever is None:
            return QueryResult(retrieved_memories=[], elapsed_ms=0.0, raw=[])
        if k != 5:
            self._retriever = self._index.as_retriever(similarity_top_k=k)

        with TimedOperation() as t:
            nodes = self._retriever.retrieve(question)

        retrieved: list[str] = []
        raw: list[dict] = []
        for n in nodes:
            txt = n.get_content() if hasattr(n, "get_content") else getattr(n, "text", "")
            meta = n.metadata if hasattr(n, "metadata") else {}
            retrieved.append(txt)
            raw.append({"id": getattr(n, "node_id", ""), "content": txt, **(meta or {})})
        return QueryResult(
            retrieved_memories=retrieved,
            elapsed_ms=t.elapsed_ms,
            raw=raw,
        )

    def close(self) -> None:
        self._index = None
        self._retriever = None
        if self._tmpdir and self._tmpdir.exists():
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._tmpdir = None
