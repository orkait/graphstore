"""Mem0 adapter for the benchmark framework.

Install:
    pip install mem0ai

Docs:
    https://docs.mem0.ai

Mem0 uses an LLM to extract salient facts during ingest and stores them as
natural-language memories. This adds quality but also adds token cost,
which is part of the apples-to-apples story we want to surface.
"""

from __future__ import annotations

from typing import Any

from ..adapter import QueryResult, Session, TimedOperation


class Mem0Adapter:
    name = "mem0"
    version = "unknown"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        try:
            from mem0 import Memory  # type: ignore
        except ImportError as e:
            raise ImportError(
                "mem0ai not installed. Run: pip install mem0ai"
            ) from e

        self._Memory = Memory
        self.config = config or {}
        self._mem: Any = None
        self._user_id: str = self.config.get("user_id", "bench_user")

        try:
            import mem0  # type: ignore

            self.version = getattr(mem0, "__version__", "unknown")
        except Exception:
            pass

    def reset(self) -> None:
        self.close()
        self._mem = self._Memory()

    def ingest(self, session: Session) -> float:
        if self._mem is None:
            raise RuntimeError("Mem0Adapter not reset - call reset() first")

        with TimedOperation() as t:
            for msg in session.messages:
                self._mem.add(
                    messages=[{"role": msg.role, "content": msg.content}],
                    user_id=self._user_id,
                )
        return t.elapsed_ms / 1000.0

    def query(self, question: str, k: int = 5) -> QueryResult:
        if self._mem is None:
            raise RuntimeError("Mem0Adapter not reset - call reset() first")

        with TimedOperation() as t:
            results = self._mem.search(
                query=question,
                user_id=self._user_id,
                limit=k,
            )

        retrieved: list[str] = []
        if isinstance(results, dict):
            results = results.get("results", [])
        for r in results or []:
            if isinstance(r, dict):
                text = r.get("memory") or r.get("text") or r.get("content") or ""
                if text:
                    retrieved.append(text)
            elif isinstance(r, str):
                retrieved.append(r)

        return QueryResult(
            retrieved_memories=retrieved,
            elapsed_ms=t.elapsed_ms,
            raw=results,
        )

    def close(self) -> None:
        self._mem = None
