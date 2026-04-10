"""GraphStore adapter for the benchmark framework.

Shows graphstore at its best:
    - Pre-registered schema with EMBED field so CREATE auto-embeds
    - Typed columns (int32_interned / int64 / float64) via SYS REGISTER
    - Per-session deferred_embeddings() context batches embedder calls
    - Importance scores computed as a single numpy pass per session
    - REMEMBER (5-signal hybrid fusion) as the query primitive
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from graphstore import GraphStore, __version__ as _GS_VERSION

from ..adapter import QueryResult, Session, TimedOperation


def _escape(text: str) -> str:
    """Escape a string so it is safe to embed inside a DSL string literal."""
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", " ")
        .replace("\r", " ")
    )


def _importance_scores(lengths: np.ndarray) -> np.ndarray:
    """Compute per-message importance in a single vectorized pass.

    Longer messages carry more content signal in LongMemEval-style
    benchmarks, so we scale by length and floor at 0.3 so short
    utterances still receive REMEMBER's recency boost.
    """
    if lengths.size == 0:
        return lengths.astype(np.float32)
    max_len = float(lengths.max()) or 1.0
    return np.clip(lengths.astype(np.float32) / max_len, 0.3, 1.0)


class GraphStoreAdapter:
    name = "graphstore"
    version = _GS_VERSION

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._tmpdir: Path | None = None
        self._gs: GraphStore | None = None
        self._embed_batch_size: int = int(self.config.get("embed_batch_size", 128))

    def reset(self) -> None:
        self.close()
        self._tmpdir = Path(tempfile.mkdtemp(prefix="gs_bench_"))
        self._gs = GraphStore(
            path=str(self._tmpdir),
            ceiling_mb=self.config.get("ceiling_mb", 4096),
            threaded=self.config.get("threaded", False),
        )
        # Pre-register the schema so CREATE auto-embeds and columns are
        # pre-allocated with known types.
        self._gs.execute(
            'SYS REGISTER NODE KIND "memory" '
            'REQUIRED session:string, role:string, content:string '
            'OPTIONAL importance:float, position:int '
            'EMBED content'
        )

    def ingest(self, session: Session) -> float:
        if self._gs is None:
            raise RuntimeError("GraphStoreAdapter not reset - call reset() first")

        n = len(session.messages)
        if n == 0:
            return 0.0

        # Vectorized importance score from message lengths
        lengths = np.fromiter(
            (len(m.content) for m in session.messages), dtype=np.int32, count=n
        )
        importance = _importance_scores(lengths)
        sid = _escape(session.session_id)

        with TimedOperation() as t:
            with self._gs.deferred_embeddings(batch_size=self._embed_batch_size):
                for i, msg in enumerate(session.messages):
                    node_id = f"{session.session_id}:msg{i}"
                    content = _escape(msg.content)
                    role = _escape(msg.role)
                    self._gs.execute(
                        f'CREATE NODE "{node_id}" kind = "memory" '
                        f'session = "{sid}" role = "{role}" '
                        f'content = "{content}" '
                        f'importance = {float(importance[i]):.3f} '
                        f'position = {i}'
                    )
                for i in range(n - 1):
                    a = f"{session.session_id}:msg{i}"
                    b = f"{session.session_id}:msg{i + 1}"
                    self._gs.execute(f'CREATE EDGE "{a}" -> "{b}" kind = "next"')
            # deferred_embeddings exits here -> flushes pending embeddings
        return t.elapsed_ms / 1000.0

    def query(self, question: str, k: int = 5) -> QueryResult:
        if self._gs is None:
            raise RuntimeError("GraphStoreAdapter not reset - call reset() first")

        q = _escape(question)
        with TimedOperation() as t:
            result = self._gs.execute(f'REMEMBER "{q}" LIMIT {k}')

        retrieved: list[str] = []
        if result.data:
            for node in result.data:
                text = node.get("content") or node.get("summary") or ""
                if text:
                    retrieved.append(text)

        return QueryResult(
            retrieved_memories=retrieved,
            elapsed_ms=t.elapsed_ms,
            raw=result.data,
        )

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
