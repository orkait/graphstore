"""GraphStore adapter for the benchmark framework.

Ingests each session's messages as memory nodes wired with "next" edges,
then queries via REMEMBER (hybrid vector + BM25 + recency + confidence + recall).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from graphstore import GraphStore, __version__ as _GS_VERSION

from ..adapter import QueryResult, Session, TimedOperation


def _escape(text: str) -> str:
    """Escape a string so it is safe to embed inside a DSL string literal."""
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").replace("\r", " ")


def _sanitize_query(text: str) -> str:
    """Strip everything FTS5 might interpret as an operator.

    LEXICAL SEARCH and REMEMBER pass the raw query through SQLite FTS5,
    which treats `?`, `*`, `(`, `)`, `:`, `^`, `+`, `-`, `~`, `'`, `"` as
    syntax. A natural-language question with any of these blows up as
    "fts5: syntax error near '...'". We keep only alphanumerics and
    whitespace - enough for BM25 bag-of-words matching, safe against
    every FTS5 operator.

    NOTE: this is a workaround. The real fix belongs in
    graphstore.document.store.search_text, which should quote the query
    for FTS5 bareword search.
    """
    out = []
    for ch in text:
        if ch.isalnum() or ch.isspace():
            out.append(ch)
        else:
            out.append(" ")
    return " ".join("".join(out).split()).strip()


class GraphStoreAdapter:
    name = "graphstore"
    version = _GS_VERSION

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._tmpdir: Path | None = None
        self._gs: GraphStore | None = None

    def reset(self) -> None:
        self.close()
        self._tmpdir = Path(tempfile.mkdtemp(prefix="gs_bench_"))
        self._gs = GraphStore(
            path=str(self._tmpdir),
            ceiling_mb=self.config.get("ceiling_mb", 2048),
            threaded=self.config.get("threaded", False),
        )

    def ingest(self, session: Session) -> float:
        if self._gs is None:
            raise RuntimeError("GraphStoreAdapter not reset - call reset() first")

        with TimedOperation() as t:
            for i, msg in enumerate(session.messages):
                node_id = f"{session.session_id}:msg{i}"
                content = _escape(msg.content)
                role = _escape(msg.role)
                sid = _escape(session.session_id)
                self._gs.execute(
                    f'CREATE NODE "{node_id}" kind = "memory" '
                    f'role = "{role}" summary = "{content}" '
                    f'session = "{sid}" '
                    f'DOCUMENT "{content}"'
                )
            # Wire consecutive messages so RECALL can walk the conversation
            for i in range(len(session.messages) - 1):
                a = f"{session.session_id}:msg{i}"
                b = f"{session.session_id}:msg{i + 1}"
                self._gs.execute(
                    f'CREATE EDGE "{a}" -> "{b}" kind = "next"'
                )
        return t.elapsed_ms / 1000.0

    def query(self, question: str, k: int = 5) -> QueryResult:
        if self._gs is None:
            raise RuntimeError("GraphStoreAdapter not reset - call reset() first")

        q = _escape(_sanitize_query(question))
        with TimedOperation() as t:
            result = self._gs.execute(f'REMEMBER "{q}" LIMIT {k}')

        retrieved: list[str] = []
        if result.data:
            for node in result.data:
                text = (
                    node.get("summary")
                    or node.get("content")
                    or node.get("text")
                    or ""
                )
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
