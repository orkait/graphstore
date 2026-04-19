"""LLM-driven skill-guided ingest adapter.

Pattern B from the architecture discussion: instead of deterministic parse +
NER + CREATE NODE (the baseline `graphstore_.py` adapter does that), this
adapter hands each session to an LLM along with the `graphstore-dsl` skill
and asks the LLM to emit DSL statements. Each emitted line is parsed
through Lark; valid statements execute, invalid ones get counted and
dropped.

Why this shape:
- graphstore's DSL is small (~7 constructs for ingest) -> LLM can learn it
- skill text is tight (~19 KB) -> fits cleanly in context
- parser roundtrip guarantees no silent corruption
- Python-side escape helpers never run; LLM must escape its own strings
  per the skill's rules

Bench: same query side as `graphstore_.py`; only `ingest()` differs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graphstore.dsl.parser import parse as dsl_parse

from ..adapter import Session, TimedOperation
from ..llm_client import llm_call
from .graphstore_ import GraphStoreAdapter


_SKILL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "tools" / "skills" / "graphstore-dsl" / "SKILL.md"


@dataclass
class _IngestStats:
    emitted: int = 0
    executed: int = 0
    parse_failed: int = 0
    exec_failed: int = 0
    llm_empty: int = 0
    sessions: int = 0
    last_parse_errors: list[tuple[str, str]] = None  # type: ignore
    last_exec_errors: list[tuple[str, str]] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.last_parse_errors is None:
            self.last_parse_errors = []
        if self.last_exec_errors is None:
            self.last_exec_errors = []

    def as_dict(self) -> dict[str, Any]:
        return {
            "emitted": self.emitted,
            "executed": self.executed,
            "parse_failed": self.parse_failed,
            "exec_failed": self.exec_failed,
            "llm_empty": self.llm_empty,
            "sessions": self.sessions,
            "accept_rate": round(self.executed / max(self.emitted, 1), 3),
        }


def _render_session_prompt(session: Session, skill: str) -> str:
    """Build the LLM prompt: skill + session + emission instructions."""
    date = session.metadata.get("date", "unknown-date")
    turns = []
    for i, msg in enumerate(session.messages):
        turns.append(f"[{i}] [{date}] {msg.role}: {msg.content}")
    turns_block = "\n".join(turns)

    instructions = f"""
--- YOUR TASK ---

You have been given the `graphstore-dsl` skill above. You are now ingesting one
conversation session into graphstore. Your job: emit graphstore DSL statements,
one per line, that store this session as durable memory.

Session id: {session.session_id}
Date: {date}
Messages: {len(session.messages)}

Rules:
- Emit plain DSL text only. No Python. No code fences. No prose, no numbering.
- One statement per line. No blank lines inside a statement.
- Escape `"` as `\\"` inside every string literal.
- Schema is already registered (kinds: session, message, entity). Do NOT emit
  `SYS REGISTER` statements.
- Create one `session` node with EVENT_AT set to the date:
  CREATE NODE "sess:{session.session_id}" kind = "session" session_id = "{session.session_id}" EVENT_AT "{date}"
- For each message, CREATE NODE with kind="message", the `session` field set
  to "{session.session_id}", the role, and a DOCUMENT clause carrying the
  content. Include EVENT_AT. Use ids of the form `{session.session_id}:msg<i>`
  where <i> is the zero-based index.
- Add `CREATE EDGE "sess:<id>" -> "<session_id>:msg<i>" kind = "has_message"`
  for every message.
- Between adjacent messages, add `CREATE EDGE "<session_id>:msg<i>" -> "<session_id>:msg<i+1>" kind = "next"`.
- Extract prominent named entities (people, places, concrete objects) from
  each message. For each unique entity in a message:
    UPSERT NODE "ent:<slug>" kind = "entity" name = "<display-name>"
    CREATE EDGE "<session_id>:msg<i>" -> "ent:<slug>" kind = "mentions"
  where <slug> is lowercase + underscores, stripped of punctuation.
- Dedupe entity mentions per message (G8 in the skill). Do not emit two
  `mentions` edges from the same message to the same entity.
- If a message asserts a durable fact (preferences, allergies, pets,
  relationships), additionally emit:
    ASSERT "fact:<slug>" kind = "fact" value = "<value>" CONFIDENCE <0.5-1.0> SOURCE "<session_id>:msg<i>" EVENT_AT "{date}"
  with confidence reflecting how explicit the statement is.
- If a later message in this session contradicts an earlier fact asserted in
  this same session, emit `RETRACT "fact:<slug>" REASON "<why>"` after the
  superseding `ASSERT`.

--- RAW TURNS ---

{turns_block}

--- EMIT DSL NOW ---
""".strip()

    return f"{skill}\n\n{instructions}"


_STATEMENT_BATCH_LIMIT = 400
_FENCE_RE = re.compile(r"^```(?:\w+)?\s*$", re.MULTILINE)
_PROSE_PREFIX_RE = re.compile(r"^(?:#|//|--|\*|\d+[\.)]\s|- |\* )")


def _iter_dsl_lines(raw: str) -> list[str]:
    """Strip code fences + prose, return candidate DSL lines."""
    cleaned = _FENCE_RE.sub("", raw)
    out: list[str] = []
    for line in cleaned.split("\n"):
        line = line.strip()
        if not line:
            continue
        if _PROSE_PREFIX_RE.match(line):
            continue
        out.append(line)
    return out


class GraphStoreSkillAdapter(GraphStoreAdapter):
    """LLM-driven ingest. Query path inherited from GraphStoreAdapter."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.stats = _IngestStats()
        self._skill: str | None = None
        self._max_tokens: int = int(self.config.get("skill_max_tokens", 6000))
        self._retry_on_empty: int = int(self.config.get("skill_retry_on_empty", 1))
        self.name = f"{self.name}-skill-llm"

    def _load_skill(self) -> str:
        if self._skill is not None:
            return self._skill
        if not _SKILL_PATH.exists():
            raise FileNotFoundError(f"graphstore-dsl skill not found at {_SKILL_PATH}")
        self._skill = _SKILL_PATH.read_text(encoding="utf-8")
        return self._skill

    def ingest(self, session: Session) -> float:
        if self._gs is None:
            raise RuntimeError("reset() must be called first")
        if not session.messages:
            return 0.0

        skill = self._load_skill()
        prompt = _render_session_prompt(session, skill)

        with TimedOperation() as t:
            raw = llm_call(prompt, max_tokens=self._max_tokens, _retries=self._retry_on_empty)
            if not raw:
                self.stats.llm_empty += 1
                self.stats.sessions += 1
                return t.elapsed_ms

            lines = _iter_dsl_lines(raw)
            self.stats.emitted += len(lines)

            with self._gs.deferred_embeddings(batch_size=self._embed_batch_size):
                for line in lines[:_STATEMENT_BATCH_LIMIT]:
                    try:
                        dsl_parse(line)
                    except Exception as e:
                        self.stats.parse_failed += 1
                        if len(self.stats.last_parse_errors) < 5:
                            self.stats.last_parse_errors.append((line[:120], str(e)[:120]))
                        continue
                    try:
                        self._gs.execute(line)
                        self.stats.executed += 1
                    except Exception as e:
                        self.stats.exec_failed += 1
                        if len(self.stats.last_exec_errors) < 5:
                            self.stats.last_exec_errors.append((line[:120], str(e)[:120]))
                        continue

        self.stats.sessions += 1
        return t.elapsed_ms

    def ingest_done(self, record_metadata: dict[str, Any] | None = None) -> None:
        """Emit ingest stats at end of record for the benchmark runner to log."""
        if record_metadata is not None:
            record_metadata.setdefault("ingest_stats", {}).update(self.stats.as_dict())

    def reset(self) -> None:
        super().reset()
        self.stats = _IngestStats()
        # Parent adapter registers `message` with `content:string` REQUIRED +
        # `EMBED content`. That fits the content-field path. This adapter
        # teaches the LLM to use `DOCUMENT "..."` instead (G2 / PR #102), so
        # re-register `message` without `content` and without EMBED. DOCUMENT
        # alone populates blob + FTS5 + vector in one shot.
        self._gs.execute('SYS UNREGISTER NODE KIND "message"')
        self._gs.execute(
            'SYS REGISTER NODE KIND "message" '
            'REQUIRED session:string, role:string '
            'OPTIONAL position:int'
        )
        # Parent skipped `entity` kind when entity_extraction=False. Add it
        # here - the LLM is expected to emit UPSERT NODE for entities.
        try:
            self._gs.execute('SYS REGISTER NODE KIND "entity" REQUIRED name:string')
        except Exception:
            pass  # already registered
