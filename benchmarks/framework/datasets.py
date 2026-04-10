"""Benchmark dataset loaders.

Datasets are loaded into a uniform shape so every adapter sees identical
input.

LongMemEval uses a per-question evaluation protocol: each record ships
with its own haystack of ~500 messages across ~53 sessions, and the
system is scored on that record in isolation. This is different from
"ingest one global pool and query many times", so the dataset loader
returns one BenchmarkRecord per question rather than a flat session pool.

Supported:
    longmemeval - https://github.com/xiaowu0162/LongMemEval
    locomo      - https://snap-research.github.io/locomo/   (stub)
    amb         - https://agentmemorybenchmark.ai          (stub)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .adapter import Message, Session


@dataclass
class BenchmarkQuestion:
    question: str
    gold_answers: list[str]
    category: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkRecord:
    """One evaluation unit: a question plus the haystack that contains its answer.

    LongMemEval's per-question protocol requires ingesting this haystack
    fresh for every question, running the query, scoring, then resetting
    state before moving on. The runner enforces this.
    """

    question: BenchmarkQuestion
    sessions: list[Session]


@dataclass
class BenchmarkDataset:
    name: str
    records: list[BenchmarkRecord]

    def __len__(self) -> int:
        return len(self.records)


def load_longmemeval(
    data_path: str | Path,
    variant: str = "s",
    max_records: int | None = None,
    start: int = 0,
    categories: set[str] | list[str] | None = None,
    per_category: int | None = None,
) -> BenchmarkDataset:
    """Load LongMemEval with flexible slicing.

    Args:
        data_path: directory containing longmemeval_<variant>_cleaned.json
        variant: s / m / l
        max_records: cap total returned (after filtering)
        start: skip this many records from the top of the file (pre-filter)
        categories: if set, only keep records whose question_type is in this set
                    e.g. {"multi-session", "temporal-reasoning"}
        per_category: if set, return at most this many records per category
                      (sampled in the order they appear in the file)

    Each record has:
        question_id, question_type, question, answer, question_date,
        answer_session_ids, haystack_dates, haystack_session_ids, haystack_sessions
    """
    p = Path(data_path)
    candidates = [
        p / f"longmemeval_{variant}_cleaned.json",
        p / f"longmemeval_{variant}.json",
    ]
    data_file = next((c for c in candidates if c.exists()), None)
    if data_file is None:
        raise FileNotFoundError(
            f"LongMemEval data not found. Looked in: {', '.join(str(c) for c in candidates)}. "
            f"Download from https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned"
        )

    with open(data_file) as f:
        raw_records = json.load(f)

    if start > 0:
        raw_records = raw_records[start:]

    cat_filter: set[str] | None = set(categories) if categories else None
    if cat_filter is not None:
        raw_records = [r for r in raw_records if r.get("question_type") in cat_filter]

    if per_category is not None:
        by_cat: dict[str, list] = {}
        sampled: list = []
        for r in raw_records:
            cat = r.get("question_type") or "unknown"
            bucket = by_cat.setdefault(cat, [])
            if len(bucket) < per_category:
                bucket.append(r)
                sampled.append(r)
        raw_records = sampled

    if max_records is not None:
        raw_records = raw_records[:max_records]

    records: list[BenchmarkRecord] = []
    for rec in raw_records:
        raw_answer = rec.get("answer", "")
        if isinstance(raw_answer, list):
            gold = [str(a) for a in raw_answer]
        else:
            gold = [str(raw_answer)]

        haystack_sessions = rec.get("haystack_sessions", [])
        haystack_ids = rec.get("haystack_session_ids", [])
        haystack_dates = rec.get("haystack_dates", [])
        question_date = rec.get("question_date")

        sessions: list[Session] = []
        for idx, (sid, msg_list) in enumerate(zip(haystack_ids, haystack_sessions)):
            base = sid or f"sess_{idx}"
            session_id = f"h{idx:03d}_{base}"
            sess_date = haystack_dates[idx] if idx < len(haystack_dates) else None
            msgs = [
                Message(role=m.get("role", "user"), content=m.get("content", ""))
                for m in msg_list
            ]
            sessions.append(Session(
                session_id=session_id,
                messages=msgs,
                metadata={
                    "date": sess_date,
                    "position": idx,
                    "question_date": question_date,
                },
            ))

        question = BenchmarkQuestion(
            question=rec["question"],
            gold_answers=gold,
            category=rec.get("question_type"),
            metadata={
                "question_id": rec.get("question_id"),
                "question_date": question_date,
                "answer_session_ids": rec.get("answer_session_ids", []),
            },
        )
        records.append(BenchmarkRecord(question=question, sessions=sessions))

    return BenchmarkDataset(
        name=f"LongMemEval-{variant.upper()}",
        records=records,
    )


def load_locomo(data_path: str | Path) -> BenchmarkDataset:
    raise NotImplementedError("LoCoMo loader is a stub.")


def load_amb(data_path: str | Path) -> BenchmarkDataset:
    raise NotImplementedError("AMB loader is a stub.")


DATASET_LOADERS = {
    "longmemeval": load_longmemeval,
    "locomo": load_locomo,
    "amb": load_amb,
}
