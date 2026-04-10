"""Letta (MemGPT) adapter stub for the benchmark framework.

Install:
    pip install letta

Docs:
    https://docs.letta.com

Letta uses a three-tier memory model: core memory (always in-context),
archival memory (vector store), and recall memory (conversation history).
The LLM itself drives what gets stored where.

This adapter is a stub - implement against the letta SDK once
the benchmark tooling is validated against graphstore + mem0.
"""

from __future__ import annotations

from typing import Any

from ..adapter import QueryResult, Session


class LettaAdapter:
    name = "letta"
    version = "unknown"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        try:
            import letta  # type: ignore  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "letta not installed. Run: pip install letta"
            ) from e
        self.config = config or {}

    def reset(self) -> None:
        raise NotImplementedError(
            "LettaAdapter is a stub. "
            "Implement against the letta SDK: https://docs.letta.com"
        )

    def ingest(self, session: Session) -> float:
        raise NotImplementedError

    def query(self, question: str, k: int = 5) -> QueryResult:
        raise NotImplementedError

    def close(self) -> None:
        pass
