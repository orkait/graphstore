"""Benchmark runner: orchestrates reset -> ingest -> query -> score for one system."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from .adapter import MemoryAdapter
from .datasets import BenchmarkDataset
from .metrics import RunResult


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def run_benchmark(
    adapter: MemoryAdapter,
    dataset: BenchmarkDataset,
    *,
    k: int = 5,
    max_questions: int | None = None,
    config: dict[str, Any] | None = None,
    progress_every: int = 50,
) -> RunResult:
    """Run one pass against one adapter.

    Phases:
        0. adapter.reset()
        1. memory baseline snapshot
        2. ingest every session in dataset.sessions (timed per session)
        3. query every benchmark question (timed per query), score retrieval
        4. record final memory + completion stamp
    """
    result = RunResult(
        system_name=adapter.name,
        system_version=adapter.version,
        benchmark=dataset.name,
        config=config or {},
        started_at=_now_iso(),
    )

    adapter.reset()
    result.memory.start()
    t0 = time.perf_counter()

    n_sessions = len(dataset.sessions)
    print(f"[{adapter.name}] ingesting {n_sessions} sessions...")
    for i, session in enumerate(dataset.sessions):
        elapsed_s = adapter.ingest(session)
        result.latency_ingest.add(elapsed_s * 1000)
        if (i + 1) % progress_every == 0:
            result.memory.snapshot_peak()
            print(f"  [{adapter.name}] ingested {i + 1}/{n_sessions}")

    questions = dataset.questions
    if max_questions is not None:
        questions = questions[:max_questions]

    print(f"[{adapter.name}] querying {len(questions)} questions...")
    for i, q in enumerate(questions):
        qres = adapter.query(q.question, k=k)
        result.latency_query.add(qres.elapsed_ms)
        result.quality.add(
            gold_answers=q.gold_answers,
            retrieved=qres.retrieved_memories,
            k=k,
        )
        if qres.tokens_used:
            result.cost.query_tokens += qres.tokens_used
        if (i + 1) % progress_every == 0:
            print(f"  [{adapter.name}] queried {i + 1}/{len(questions)}")

    result.memory.stop()
    result.total_elapsed_s = time.perf_counter() - t0
    result.completed_at = _now_iso()

    try:
        adapter.close()
    except Exception as e:
        print(f"  [{adapter.name}] close() raised: {type(e).__name__}: {e}")

    return result
