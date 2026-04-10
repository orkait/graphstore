"""Benchmark dataset loaders.

Datasets are loaded into a uniform BenchmarkDataset shape so every adapter
sees identical input. Each loader is responsible for its own format quirks.

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
    session_id: str | None = None
    category: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkDataset:
    name: str
    sessions: list[Session]
    questions: list[BenchmarkQuestion]

    def __len__(self) -> int:
        return len(self.questions)


def load_longmemeval(data_path: str | Path, variant: str = "s") -> BenchmarkDataset:
    """Load the LongMemEval dataset.

    variant: "s" (short, ~115k tokens/session), "m", or "l"

    Expected layout under data_path:
        longmemeval_s.json   (or _m, _l)

    Each record in the JSON file is a dict with at least:
        question            the user's probe question
        answer              gold answer (string or list)
        haystack_sessions   list of sessions, each a list of {role, content}
        question_type       optional category tag
    """
    p = Path(data_path)
    fname = f"longmemeval_{variant}.json"
    data_file = p / fname
    if not data_file.exists():
        raise FileNotFoundError(
            f"LongMemEval data not found at {data_file}. "
            f"Download from https://github.com/xiaowu0162/LongMemEval"
        )

    with open(data_file) as f:
        records = json.load(f)

    sessions: list[Session] = []
    questions: list[BenchmarkQuestion] = []
    seen: set[str] = set()

    for rec_idx, record in enumerate(records):
        haystack = record.get("haystack_sessions", [])
        for sess_idx, sess in enumerate(haystack):
            sid = f"rec{rec_idx}_sess{sess_idx}"
            if sid in seen:
                continue
            seen.add(sid)
            msgs = [
                Message(
                    role=m.get("role", "user"),
                    content=m.get("content", ""),
                    timestamp=m.get("timestamp"),
                )
                for m in sess
            ]
            sessions.append(Session(session_id=sid, messages=msgs))

        raw_answer = record.get("answer", "")
        gold = [raw_answer] if isinstance(raw_answer, str) else list(raw_answer)

        questions.append(
            BenchmarkQuestion(
                question=record["question"],
                gold_answers=gold,
                category=record.get("question_type"),
                metadata={"rec_index": rec_idx},
            )
        )

    return BenchmarkDataset(
        name=f"LongMemEval-{variant.upper()}",
        sessions=sessions,
        questions=questions,
    )


def load_locomo(data_path: str | Path) -> BenchmarkDataset:
    """LoCoMo loader (stub).

    https://snap-research.github.io/locomo/
    """
    raise NotImplementedError(
        "LoCoMo loader is a stub. Implement after LongMemEval is validated."
    )


def load_amb(data_path: str | Path) -> BenchmarkDataset:
    """Agent Memory Benchmark loader (stub).

    https://agentmemorybenchmark.ai
    """
    raise NotImplementedError(
        "AMB loader is a stub. See https://agentmemorybenchmark.ai for format."
    )


DATASET_LOADERS = {
    "longmemeval": load_longmemeval,
    "locomo": load_locomo,
    "amb": load_amb,
}
