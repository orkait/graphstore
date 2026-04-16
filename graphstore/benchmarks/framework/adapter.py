"""Adapter protocol for benchmarking agent memory systems.

Every system under test implements the same four-method interface:
    reset()         wipe memory, fresh state
    ingest(session) add a conversation session
    query(question) retrieve top-K memories
    close()         release resources
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str
    timestamp: float | None = None


@dataclass
class Session:
    session_id: str
    messages: list[Message]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryContext:
    question: str
    category: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    retrieved_memories: list[str]
    answer: str | None = None
    elapsed_ms: float = 0.0
    tokens_used: int = 0
    raw: Any = None  # optional: per-system native result for debugging


class MemoryAdapter(Protocol):
    """Protocol every benchmarked system must satisfy."""

    name: str
    version: str

    def reset(self) -> None: ...
    def ingest(self, session: Session) -> float: ...
    def query(self, question: str, k: int = 5) -> QueryResult: ...
    def close(self) -> None: ...


class TimedOperation:
    """Context manager that records elapsed wall-clock time in ms."""

    def __init__(self) -> None:
        self._start_ns = 0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "TimedOperation":
        self._start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed_ms = (time.perf_counter_ns() - self._start_ns) / 1_000_000
