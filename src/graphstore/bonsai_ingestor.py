"""Natural-language to DSL ingestor backed by a local llama.cpp GGUF.

Target model: Ternary-Bonsai 4B TQ1_0 (CPU + RAM only, offline).

Download the model once before first use (models/ is gitignored):

    mkdir -p models/Ternary-Bonsai-4B-TQ1_0 && curl -L -o \\
      models/Ternary-Bonsai-4B-TQ1_0/Ternary-Bonsai-4B-TQ1_0.gguf \\
      https://huggingface.co/superkaiii/Ternary-Bonsai-4B-TQ1_0-GGUF/resolve/main/Ternary-Bonsai-4B-TQ1_0.gguf

Publication pipeline: benchmarks/kaggle/pack_ternary_bonsai/ converts
prism-ml/Ternary-Bonsai-4B-unpacked (FP16) to TQ1_0 via a Kaggle kernel and
publishes the result to superkaiii/Ternary-Bonsai-4B-TQ1_0-GGUF on HF.

The public surface is `BonsaiIngestor`. Every call:
  1. Loads the skill prompt once, fingerprints it, pins that fingerprint into
     the system prompt so KV cache stays coherent across calls. When the skill
     file on disk changes, the next call naturally forces a warm re-process
     because the system-prompt prefix now differs.
  2. Serializes access to the `Llama` instance with a lock - llama-cpp-python
     is NOT thread-safe; concurrent calls corrupt the KV cache / segfault.
  3. Checks the combined prompt + output budget against `n_ctx` and resets the
     cache if the request would force a head-trim (which silently evicts the
     skill prefix and causes quality collapse).
  4. Treats empty / <think>-only outputs as ingestion errors, not as silent
     no-ops (the behaviour-analysis audit called this out - silent failures
     hid real bugs).
  5. Dedupes UPSERT NODE by id before handing the DSL to the parser, so a
     BatchRollback from a duplicated entity never takes out an otherwise
     valid batch.
  6. Emits one structured log line per ingest with counts for every
     category: statements emitted, parsed, rejected, executed, duration.
  7. Supports `dry_run=True` to generate the DSL and return it without
     touching the GraphStore - used for testing, previewing, and building
     training data.

It also tracks cross-message belief state. After every successful ingest, the
executed ASSERT / RETRACT lines are scraped into a running `fact_id -> FactState`
dict. The next ingest injects the live (non-retracted) facts into the USER
message (not the system prompt - see below), so the model sees prior fact ids
and reuses them when updating the same concept. Without this, the model coins
a new fact_id per message and the graph ends up with multiple beliefs for the
same underlying concept.

The known-facts block goes in the USER message on purpose: the system prompt
stays byte-identical across calls, which keeps llama.cpp's prefix-match KV
cache warm. If we appended facts to the system prompt, every call would be a
cold one.

Not handled here:
  - Streaming. Generation is blocking. Callers can run it in a thread.
  - Multi-user / multi-tenant. Use one BonsaiIngestor per user.
  - Model swap. Create a new BonsaiIngestor; don't mutate this one's paths.
"""
from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


# --------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------

class BonsaiError(Exception):
    """Base class for ingestor errors."""


class IngestEmpty(BonsaiError):
    """LLM returned empty or <think>-only output. No DSL to execute."""


class IngestOverflow(BonsaiError):
    """Requested prompt + output would exceed n_ctx even after a cache reset."""


# --------------------------------------------------------------------
# Result
# --------------------------------------------------------------------

@dataclass
class FactState:
    """One live belief tracked across messages within this ingestor.

    Updated by `_scrape_belief_updates` after every successful ingest and
    rendered back into the next ingest's user message by
    `_render_known_facts_block`, so the model reuses the same fact_id for the
    same concept instead of coining a new one each call.
    """

    fact_id: str
    kind: str = ""
    value: str = ""
    confidence: float = 1.0
    source: str = ""
    retracted: bool = False
    retract_reason: str = ""


@dataclass
class IngestResult:
    """Everything an ingest call produced, for inspection and tracing."""

    statements: list[str] = field(default_factory=list)
    executed: int = 0
    rejected: list[tuple[str, str]] = field(default_factory=list)
    entities_new: list[str] = field(default_factory=list)
    beliefs_changed: list[tuple[str, str]] = field(default_factory=list)
    duration_ms: int = 0
    raw_output: str = ""
    skill_fingerprint: str = ""
    dry_run: bool = False


# --------------------------------------------------------------------
# Post-processing helpers
# --------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*$|^```\s*$")
_UPSERT_RE = re.compile(r'^\s*UPSERT\s+NODE\s+"([^"\\]+(?:\\.[^"\\]*)*)"', re.IGNORECASE)
_ASSERT_LINE_RE = re.compile(
    r'^\s*ASSERT\s+"([^"\\]+(?:\\.[^"\\]*)*)"\s+(.*)$',
    re.IGNORECASE,
)
_ASSERT_RE = re.compile(r'^\s*ASSERT\s+"([^"\\]+(?:\\.[^"\\]*)*)"', re.IGNORECASE)
_RETRACT_RE = re.compile(
    r'^\s*RETRACT\s+"([^"\\]+(?:\\.[^"\\]*)*)"(?:\s+REASON\s+"([^"\\]*(?:\\.[^"\\]*)*)")?\s*$',
    re.IGNORECASE,
)
_CREATE_NODE_RE = re.compile(r'^\s*CREATE\s+NODE\s+"([^"\\]+(?:\\.[^"\\]*)*)"', re.IGNORECASE)
_ENT_FROM_ID_RE = re.compile(r'"(ent:[^"\\]+)"')
_KV_VALUE_RE = re.compile(r'value\s*=\s*"([^"\\]*(?:\\.[^"\\]*)*)"', re.IGNORECASE)
_KV_KIND_RE = re.compile(r'kind\s*=\s*"([^"\\]*(?:\\.[^"\\]*)*)"', re.IGNORECASE)
_KV_CONFIDENCE_RE = re.compile(r'CONFIDENCE\s+([0-9.]+)', re.IGNORECASE)
_KV_SOURCE_RE = re.compile(r'SOURCE\s+"([^"\\]*(?:\\.[^"\\]*)*)"', re.IGNORECASE)


def _strip_think(raw: str) -> str:
    """Remove reasoning-model `<think>...</think>` wrappers."""
    return _THINK_RE.sub("", raw).strip()


def _split_lines(cleaned: str) -> list[str]:
    """One statement per line. Drop markdown fences and blanks."""
    out: list[str] = []
    for raw_ln in cleaned.splitlines():
        ln = raw_ln.strip()
        if not ln or _FENCE_RE.match(ln):
            continue
        out.append(ln)
    return out


def _dedupe_upserts(stmts: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """Keep the first UPSERT per entity id. Drop later duplicates.

    Duplicate UPSERT NODE with the same id inside a BEGIN/COMMIT batch makes
    the rollback snapshot logic trip over itself - entire batch errors out.
    Pre-filter here so the transaction never sees the duplicate.
    """
    seen: set[str] = set()
    kept: list[str] = []
    dropped: list[tuple[str, str]] = []
    for ln in stmts:
        m = _UPSERT_RE.match(ln)
        if m:
            eid = m.group(1)
            if eid in seen:
                dropped.append((ln, f"duplicate UPSERT of {eid!r}"))
                continue
            seen.add(eid)
        kept.append(ln)
    return kept, dropped


def _scrape_belief_updates(
    executed_lines: list[str],
    facts: dict[str, FactState],
) -> None:
    """Walk successfully-executed DSL lines and update running fact state.

    Only ASSERT / RETRACT contribute. Other writes ignored. Mutates `facts`
    in place.
    """
    for line in executed_lines:
        m = _ASSERT_LINE_RE.match(line)
        if m:
            fact_id = m.group(1)
            rest = m.group(2)
            st = facts.get(fact_id) or FactState(fact_id=fact_id)
            km = _KV_KIND_RE.search(rest)
            if km:
                st.kind = km.group(1)
            vm = _KV_VALUE_RE.search(rest)
            if vm:
                st.value = vm.group(1)
            cm = _KV_CONFIDENCE_RE.search(rest)
            if cm:
                try:
                    st.confidence = float(cm.group(1))
                except ValueError:
                    pass
            sm = _KV_SOURCE_RE.search(rest)
            if sm:
                st.source = sm.group(1)
            st.retracted = False
            st.retract_reason = ""
            facts[fact_id] = st
            continue
        m = _RETRACT_RE.match(line)
        if m:
            fact_id = m.group(1)
            reason = m.group(2) or ""
            st = facts.get(fact_id) or FactState(fact_id=fact_id)
            st.retracted = True
            st.retract_reason = reason
            facts[fact_id] = st


# --------------------------------------------------------------------
# Compact output mode ("caveman v5"): LLM emits one verb per line covering
# the whole DSL surface. Python inflates to full DSL. Measured ~3-5x fewer
# output tokens than raw DSL on every path. See
# tools/skills/graphstore-bonsai-dsl-compact/SKILL.md for the contract.
#
# Verbs fall into three groups:
#   1. Fact-state (U / F / D): populate entities / beliefs / retracts slots
#      so _synthesize_dsl can auto-wire mention edges and cross-message
#      belief identity works.
#   2. Edge (E): pre-renders a CREATE EDGE line.
#   3. Retrieval (RM/SM/LX/AQ), walks (RL/TR/AN/SG), sys/vault
#      (SS/SC/SH/ST/SX/VS): each pre-renders one full DSL line directly.
#
# Groups 2 and 3 accumulate in turn.statements and get appended verbatim
# after the mention wiring and fact updates.
#
# Unknown verbs and malformed lines are silently dropped (LLM may drift;
# parser is lax so a single bad line doesn't lose the whole turn).
# --------------------------------------------------------------------


@dataclass
class CompactTurn:
    """Parsed structured output of a compact-mode LLM call (v5 option A)."""

    entities: list[tuple[str, str]] = field(default_factory=list)
    beliefs: list[tuple[str, str]] = field(default_factory=list)
    retracts: list[str] = field(default_factory=list)
    statements: list[str] = field(default_factory=list)


def _dsl_escape(s: str) -> str:
    """Escape a Python string for safe embedding inside a DSL "..." literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _h_upsert(turn: CompactTurn, ln: str) -> None:
    """U <slug> <Name ...>"""
    parts = ln.split(None, 2)
    if len(parts) < 3:
        return
    slug = parts[1].removeprefix("ent:").strip('"')
    name = parts[2].strip().strip('"')
    if slug and name:
        turn.entities.append((f"ent:{slug}", name))


def _h_fact(turn: CompactTurn, ln: str) -> None:
    """F <topic> <value ...>"""
    parts = ln.split(None, 2)
    if len(parts) < 3:
        return
    topic = parts[1].removeprefix("fact:").strip('"')
    value = parts[2].strip().strip('"')
    if topic and value:
        turn.beliefs.append((f"fact:{topic}", value))


def _h_drop(turn: CompactTurn, ln: str) -> None:
    """D <topic>"""
    parts = ln.split()
    if len(parts) < 2:
        return
    topic = parts[1].removeprefix("fact:").strip('"')
    if topic:
        turn.retracts.append(f"fact:{topic}")


def _h_edge(turn: CompactTurn, ln: str) -> None:
    """E <from_id> <to_id> <kind>  (ids include ent:/fact: prefix)"""
    parts = ln.split()
    if len(parts) < 4:
        return
    from_id = parts[1].strip('"')
    to_id = parts[2].strip('"')
    kind = parts[3].strip('"')
    if from_id and to_id and kind:
        turn.statements.append(
            f'CREATE EDGE "{_dsl_escape(from_id)}" -> "{_dsl_escape(to_id)}" '
            f'kind = "{_dsl_escape(kind)}"'
        )


def _h_query(template: str):
    """Factory for verbs whose argument is free-text (wrapped in DSL quotes)."""
    def h(turn: CompactTurn, ln: str) -> None:
        parts = ln.split(None, 1)
        if len(parts) < 2 or not parts[1].strip():
            return
        q = _dsl_escape(parts[1].strip())
        turn.statements.append(template.replace("{q}", q))
    return h


def _h_walk(template: str):
    """Factory for verbs whose argument is a single anchor id."""
    def h(turn: CompactTurn, ln: str) -> None:
        parts = ln.split()
        if len(parts) < 2:
            return
        anchor = parts[1].strip('"')
        if anchor:
            turn.statements.append(template.replace("{id}", _dsl_escape(anchor)))
    return h


def _h_plain(template: str):
    """Factory for zero-arg verbs (SS/SC/SH/ST/VS)."""
    def h(turn: CompactTurn, _ln: str) -> None:
        turn.statements.append(template)
    return h


_COMPACT_HANDLERS: dict[str, Any] = {
    "U": _h_upsert, "UP": _h_upsert, "UPSERT": _h_upsert,
    "F": _h_fact, "FACT": _h_fact, "B": _h_fact, "ASSERT": _h_fact,
    "D": _h_drop, "DROP": _h_drop, "R": _h_drop, "RETRACT": _h_drop,
    "E": _h_edge, "EDGE": _h_edge,
    "RM": _h_query('REMEMBER "{q}" LIMIT 10'),
    "REMEMBER": _h_query('REMEMBER "{q}" LIMIT 10'),
    "SM": _h_query('SIMILAR TO "{q}" LIMIT 10'),
    "SIMILAR": _h_query('SIMILAR TO "{q}" LIMIT 10'),
    "LX": _h_query('LEXICAL SEARCH "{q}" LIMIT 10'),
    "LEXICAL": _h_query('LEXICAL SEARCH "{q}" LIMIT 10'),
    "AQ": _h_query('ANSWER "{q}"'),
    "ANSWER": _h_query('ANSWER "{q}"'),
    "RL": _h_walk('RECALL FROM "{id}" DEPTH 2'),
    "RECALL": _h_walk('RECALL FROM "{id}" DEPTH 2'),
    "TR": _h_walk('TRAVERSE FROM "{id}" DEPTH 2'),
    "TRAVERSE": _h_walk('TRAVERSE FROM "{id}" DEPTH 2'),
    "AN": _h_walk('ANCESTORS OF "{id}" DEPTH 3'),
    "ANCESTORS": _h_walk('ANCESTORS OF "{id}" DEPTH 3'),
    "SG": _h_walk('SUBGRAPH FROM "{id}" DEPTH 2'),
    "SUBGRAPH": _h_walk('SUBGRAPH FROM "{id}" DEPTH 2'),
    "SS": _h_plain('SYS SNAPSHOT'),
    "SC": _h_plain('SYS COMPACT'),
    "SH": _h_plain('SYS HEALTH'),
    "ST": _h_plain('SYS STATS'),
    "SX": _h_query('SYS EXPLAIN REMEMBER "{q}"'),
    "VS": _h_plain('VAULT SYNC'),
}


def _parse_compact_output(cleaned: str) -> CompactTurn:
    """Parse v5 unified verb-positional output.

    Each non-blank, non-fence line is one verb + positional args. U/F/D
    populate the entities/beliefs/retracts slots for fact-state wiring; all
    other verbs render directly to pre-built DSL lines in turn.statements.
    Unknown verbs and malformed lines are silently dropped.
    """
    turn = CompactTurn()
    for raw_ln in cleaned.splitlines():
        ln = raw_ln.strip()
        if not ln or _FENCE_RE.match(ln):
            continue
        head = ln.split(maxsplit=1)[0]
        verb = head.upper().rstrip(":")
        handler = _COMPACT_HANDLERS.get(verb)
        if handler is None:
            continue
        handler(turn, ln)
    return turn


def _synthesize_dsl(
    turn: CompactTurn,
    *,
    msg_id: str,
    session_id: str,
    role: str,
    text: str,
) -> list[str]:
    """Build the full DSL statement list from the parsed compact output.

    Deterministic. Emits in order:
      1. CREATE NODE for the message (DOCUMENT = user text).
      2. UPSERT NODE per entity + matching CREATE EDGE kind = "mentions".
         Entities are deduped by id (first wins).
      3. RETRACT per retract (before any ASSERT).
      4. ASSERT per belief.
      5. Pre-rendered statements (edges, queries, walks, sys ops) verbatim.
    """
    out: list[str] = []
    text_esc = _dsl_escape(text)
    session_esc = _dsl_escape(session_id)
    role_esc = _dsl_escape(role)
    msg_esc = _dsl_escape(msg_id)

    out.append(
        f'CREATE NODE "{msg_esc}" kind = "message" '
        f'session = "{session_esc}" role = "{role_esc}" '
        f'DOCUMENT "{text_esc}"'
    )

    ordered_ents: list[str] = []
    seen_ents: set[str] = set()
    for ent_id, name in turn.entities:
        if ent_id in seen_ents:
            continue
        seen_ents.add(ent_id)
        ordered_ents.append(ent_id)
        out.append(
            f'UPSERT NODE "{_dsl_escape(ent_id)}" kind = "entity" name = "{_dsl_escape(name)}"'
        )
    for ent_id in ordered_ents:
        out.append(
            f'CREATE EDGE "{msg_esc}" -> "{_dsl_escape(ent_id)}" kind = "mentions"'
        )

    for fact_id in turn.retracts:
        out.append(
            f'RETRACT "{_dsl_escape(fact_id)}" REASON "superseded by {msg_esc}"'
        )

    for fact_id, value in turn.beliefs:
        out.append(
            f'ASSERT "{_dsl_escape(fact_id)}" kind = "belief" '
            f'value = "{_dsl_escape(value)}" CONFIDENCE 0.9 SOURCE "{msg_esc}"'
        )

    out.extend(turn.statements)

    return out


def _render_known_facts_block(facts: dict[str, FactState], max_facts: int = 40) -> str:
    """Format non-retracted facts into a block the LLM reads before the input.

    Placed in the USER message (not the system prompt) to keep the system
    prompt byte-identical across calls for llama.cpp KV-cache prefix reuse.

    Retracted facts are hidden so the model cannot re-assert a superseded
    belief. If the dict grows beyond `max_facts`, the most-recently-touched
    entries win (dict preserves insertion order since Python 3.7).
    """
    alive = [f for f in facts.values() if not f.retracted]
    if not alive:
        return ""
    alive = alive[-max_facts:]
    lines = [
        "### KNOWN FACTS (reuse these fact_ids; emit RETRACT + ASSERT to update)",
    ]
    for f in alive:
        bits = [f'[{f.fact_id}]']
        if f.kind:
            bits.append(f'kind="{f.kind}"')
        if f.value:
            bits.append(f'value="{f.value}"')
        bits.append(f'confidence={f.confidence:.2f}')
        if f.source:
            bits.append(f'source="{f.source}"')
        lines.append(" ".join(bits))
    return "\n".join(lines)


# --------------------------------------------------------------------
# Ingestor
# --------------------------------------------------------------------

_DEFAULT_SKILL_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "tools" / "skills" / "graphstore-bonsai-dsl" / "SKILL.md"
)

_DEFAULT_COMPACT_SKILL_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "tools" / "skills" / "graphstore-bonsai-dsl-compact" / "SKILL.md"
)


class BonsaiIngestor:
    """NL -> DSL via a local llama.cpp GGUF, with correctness guards.

    Parameters
    ----------
    model_path : str | Path
        Path to a .gguf file. The matching manifest under the same directory
        is not required; this class talks to llama.cpp directly.
    gs : GraphStore | None
        Target store. Required for non-dry-run ingests. Dry-runs don't need one.
    skill_path : str | Path | None
        Prompt file. Defaults to tools/skills/graphstore-bonsai-dsl/SKILL.md.
    n_ctx : int
        Context window. 2048 is enough for ~500-token skill + 200-token user
        message + 400-token output + headroom.
    n_threads : int | None
        Physical core count works best on memory-bandwidth-bound CPU inference.
        Defaults to os.cpu_count() // 2 via llama.cpp.
    chat_format : str
        Matches the GGUF. Qwen3-based Bonsai works with 'qwen'.
    max_output_tokens : int
        Hard cap per call. Generation stops either here or at the model's
        natural stop.
    temperature : float
        Default 0.0 for reproducible DSL.
    """

    # Headroom we leave between (prompt + output) and n_ctx. Below this we
    # force a reset so llama.cpp never auto-evicts the skill prefix.
    _CTX_HEADROOM = 128

    def __init__(
        self,
        model_path: str | Path,
        *,
        gs: Any | None = None,
        skill_path: str | Path | None = None,
        compact: bool = False,
        n_ctx: int = 2048,
        n_threads: int | None = None,
        chat_format: str = "qwen",
        max_output_tokens: int = 400,
        temperature: float = 0.0,
        kv_cache_path: str | Path | None = None,
    ) -> None:
        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(f"bonsai model not found: {self._model_path}")
        self._gs = gs
        self._compact = compact
        if skill_path:
            self._skill_path = Path(skill_path)
        else:
            self._skill_path = _DEFAULT_COMPACT_SKILL_PATH if compact else _DEFAULT_SKILL_PATH
        self._n_ctx = n_ctx
        # Compact mode emits ~30 tokens of structured output. Cap lower so
        # stray model verbosity doesn't burn decode time.
        self._max_output_tokens = max_output_tokens if not compact else min(max_output_tokens, 160)
        self._temperature = temperature
        self._chat_format = chat_format
        self._n_threads = n_threads

        self._skill_text = ""
        self._skill_fingerprint = ""
        self._system_prompt = ""
        self._reload_skill()

        self._llm: Any | None = None
        self._lock = threading.Lock()

        # Cross-message belief tracking. fact_id -> FactState. Fed into the
        # user message of the next ingest so the model reuses ids.
        self._facts: dict[str, FactState] = {}

        # Optional persistent KV cache. Eliminates the ~10s cold penalty on
        # process restarts. File holds a pickled (meta, LlamaState) tuple;
        # meta guards against loading stale state when the skill or config
        # changed since the cache was written.
        self._kv_cache_path = Path(kv_cache_path) if kv_cache_path else None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _reload_skill(self) -> None:
        """Read skill from disk, compute fingerprint, pin into system prompt.

        Pinning the fingerprint into the prompt means if the file changes on
        disk the system-prompt prefix changes too, which naturally invalidates
        the llama.cpp prefix-match KV cache without us having to call reset.
        """
        if not self._skill_path.exists():
            raise FileNotFoundError(f"bonsai skill not found: {self._skill_path}")
        body = self._skill_path.read_text()
        if body.startswith("---"):
            _, _, body = body.partition("---")
            _, _, body = body.partition("---")
            body = body.strip()
        self._skill_text = body
        self._skill_fingerprint = hashlib.sha256(body.encode()).hexdigest()[:12]
        self._system_prompt = f"# skill-sha256={self._skill_fingerprint}\n\n{body}"

    def _kv_meta(self) -> dict[str, Any]:
        """What the current config looks like. Written alongside the KV cache
        so we can refuse to load state if any of these changed."""
        return {
            "model_path": str(self._model_path),
            "model_size_bytes": self._model_path.stat().st_size,
            "skill_fingerprint": self._skill_fingerprint,
            "n_ctx": self._n_ctx,
            "chat_format": self._chat_format,
        }

    def _try_load_kv_cache(self, llm: Any) -> bool:
        """Load a persisted KV cache into `llm` if one exists and is valid.

        Returns True on successful load, False otherwise. Invalid cache is
        silently ignored - the caller warms up normally.
        """
        if not self._kv_cache_path or not self._kv_cache_path.exists():
            return False
        import pickle

        try:
            with self._kv_cache_path.open("rb") as f:
                payload = pickle.load(f)
        except Exception as err:
            _log.warning("bonsai: KV cache unreadable (%s); skipping", err)
            return False

        meta = payload.get("meta") if isinstance(payload, dict) else None
        state = payload.get("state") if isinstance(payload, dict) else None
        if not meta or state is None:
            _log.warning("bonsai: KV cache shape invalid; skipping")
            return False

        cur = self._kv_meta()
        if meta != cur:
            diff = {k: (meta.get(k), cur.get(k)) for k in cur if meta.get(k) != cur.get(k)}
            _log.info(
                "bonsai: KV cache stale (diff=%s); warming fresh",
                diff,
            )
            return False

        try:
            llm.load_state(state)
        except Exception as err:
            _log.warning("bonsai: KV cache load_state failed (%s); warming fresh", err)
            return False

        _log.info("bonsai: KV cache loaded from %s (skipped warmup)", self._kv_cache_path)
        return True

    def save_kv_cache(self) -> None:
        """Persist the current Llama instance's KV state to `kv_cache_path`.

        Call after `warmup()` (or after one real ingest) so the skill-prefix
        tokens are in the cache. The file is (meta, LlamaState) pickled.

        No-op if kv_cache_path was not configured or the Llama hasn't been
        constructed yet.
        """
        if not self._kv_cache_path or self._llm is None:
            return
        import pickle

        state = self._llm.save_state()
        self._kv_cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._kv_cache_path.with_suffix(self._kv_cache_path.suffix + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump({"meta": self._kv_meta(), "state": state}, f)
        tmp.replace(self._kv_cache_path)
        _log.info(
            "bonsai: KV cache saved to %s (%.1f MB)",
            self._kv_cache_path,
            self._kv_cache_path.stat().st_size / 1e6,
        )

    def _ensure_llm(self) -> Any:
        """Lazy-load the Llama instance on first use."""
        if self._llm is not None:
            return self._llm
        from llama_cpp import Llama
        kwargs: dict[str, Any] = {
            "model_path": str(self._model_path),
            "n_ctx": self._n_ctx,
            "chat_format": self._chat_format,
            "verbose": False,
        }
        if self._n_threads is not None:
            kwargs["n_threads"] = self._n_threads
        _log.info(
            "bonsai: loading %s n_ctx=%d threads=%s chat_format=%s",
            self._model_path.name, self._n_ctx, self._n_threads, self._chat_format,
        )
        self._llm = Llama(**kwargs)
        self._try_load_kv_cache(self._llm)
        return self._llm

    def reset(self) -> None:
        """Drop the Llama instance so the next call reloads from scratch.

        Use when the skill file changed and you want to force a cold start,
        or when KV state is suspected corrupt (e.g. after a thread crash).
        Automatic: only called from internal guards.
        """
        with self._lock:
            self._llm = None

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def skill_fingerprint(self) -> str:
        """12-hex-char sha256 prefix of the loaded skill text. Stable across
        processes for the same skill bytes. Emitted in every ingest log line."""
        return self._skill_fingerprint

    @property
    def facts(self) -> dict[str, FactState]:
        """Live fact state accumulated from successful ingests. Read-only view.

        Use reset_facts() to clear. Dry-run ingests don't update this.
        """
        return dict(self._facts)

    def reset_facts(self) -> None:
        """Clear the running fact state so the next ingest starts fresh."""
        with self._lock:
            self._facts.clear()

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def warmup(self) -> None:
        """Process the system prompt once so the skill-prefix KV is warm.

        Optional - the first real ingest pays this cost anyway. Separate
        because long-running daemons want the cost up-front, not on the
        first user-facing request.
        """
        llm = self._ensure_llm()
        with self._lock:
            llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": "ready"},
                ],
                max_tokens=1,
                temperature=0.0,
            )

    def ingest(
        self,
        text: str,
        *,
        msg_id: str | None = None,
        session_id: str = "default",
        role: str = "user",
        dry_run: bool = False,
    ) -> IngestResult:
        """Convert `text` to DSL statements and (optionally) execute them.

        In full-DSL mode (compact=False) the LLM emits DSL directly; msg_id
        and session_id come from the text the caller supplies ("Session s1,
        msg m:s1:0, user: ...") so the extra kwargs are unused.

        In compact mode (compact=True) the LLM emits ENTS/BELIEFS/RETRACTS
        and Python synthesizes the DSL. The caller must pass msg_id (and
        may override session_id / role); these become the identifiers in
        the synthesized CREATE NODE / CREATE EDGE statements.

        `dry_run=True` returns the DSL without touching the store.
        """
        if not text or not text.strip():
            raise IngestEmpty("input text is empty or whitespace-only")
        if not dry_run and self._gs is None:
            raise ValueError("ingest requires a GraphStore (pass gs=...) or dry_run=True")
        if self._compact and not msg_id:
            raise ValueError(
                "compact=True ingest requires an explicit msg_id "
                "(DSL synthesis needs the exact CREATE NODE id)"
            )

        self._reload_skill()
        with self._lock:
            return self._ingest_locked(
                text,
                msg_id=msg_id,
                session_id=session_id,
                role=role,
                dry_run=dry_run,
            )

    def _ingest_locked(
        self,
        text: str,
        *,
        msg_id: str | None,
        session_id: str,
        role: str,
        dry_run: bool,
    ) -> IngestResult:
        t0 = time.perf_counter()
        llm = self._ensure_llm()

        # Compose the user message: prior facts block + user text. Keeps the
        # system prompt byte-identical so the skill stays KV-cache warm.
        facts_block = _render_known_facts_block(self._facts)
        user_msg = f"{facts_block}\n\n{text}" if facts_block else text

        est = self._estimate_tokens(self._system_prompt) + self._estimate_tokens(user_msg)
        budget = est + self._max_output_tokens
        if budget > self._n_ctx - self._CTX_HEADROOM:
            if est + self._max_output_tokens > self._n_ctx - self._CTX_HEADROOM:
                raise IngestOverflow(
                    f"prompt+output ({budget}) exceeds n_ctx-headroom "
                    f"({self._n_ctx - self._CTX_HEADROOM}); increase n_ctx or "
                    f"shorten input"
                )
            _log.warning("bonsai: KV would overflow, resetting before ingest")
            self._llm = None
            llm = self._ensure_llm()

        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=self._max_output_tokens,
            temperature=self._temperature,
        )
        raw = response["choices"][0]["message"]["content"] or ""

        cleaned = _strip_think(raw)
        if not cleaned:
            duration_ms = int((time.perf_counter() - t0) * 1000)
            self._log_event(text, raw, [], 0, [], [], [], duration_ms, dry_run)
            raise IngestEmpty(
                "LLM returned empty or <think>-only output. "
                f"raw={raw!r}"
            )

        if self._compact:
            assert msg_id is not None  # guarded in ingest()
            turn = _parse_compact_output(cleaned)
            deduped = _synthesize_dsl(
                turn, msg_id=msg_id, session_id=session_id, role=role, text=text,
            )
            dup_dropped: list[tuple[str, str]] = []
        else:
            raw_lines = _split_lines(cleaned)
            deduped, dup_dropped = _dedupe_upserts(raw_lines)

        from graphstore.dsl.parser import parse as _dsl_parse

        valid: list[str] = []
        rejected: list[tuple[str, str]] = list(dup_dropped)
        for ln in deduped:
            try:
                _dsl_parse(ln)
                valid.append(ln)
            except Exception as err:
                rejected.append((ln, f"parse error: {err}"))

        entities_new: list[str] = []
        beliefs_changed: list[tuple[str, str]] = []
        for ln in valid:
            if _UPSERT_RE.match(ln):
                m = _ENT_FROM_ID_RE.search(ln)
                if m:
                    entities_new.append(m.group(1))
            elif _ASSERT_RE.match(ln):
                m = _ASSERT_RE.match(ln)
                if m:
                    beliefs_changed.append((m.group(1), "assert"))
            elif _RETRACT_RE.match(ln):
                m = _RETRACT_RE.match(ln)
                if m:
                    beliefs_changed.append((m.group(1), "retract"))

        executed = 0
        executed_lines: list[str] = []
        if not dry_run:
            for ln in valid:
                try:
                    self._gs.execute(ln)
                    executed += 1
                    executed_lines.append(ln)
                except Exception as err:
                    rejected.append((ln, f"execute error: {err}"))
            # Scrape belief updates so the next ingest sees the current fact
            # state. Only lines that actually executed contribute - failed
            # ones leave the running state unchanged.
            _scrape_belief_updates(executed_lines, self._facts)

        duration_ms = int((time.perf_counter() - t0) * 1000)
        self._log_event(
            text, raw, valid, executed, rejected, entities_new,
            beliefs_changed, duration_ms, dry_run,
        )
        return IngestResult(
            statements=list(valid),
            executed=executed,
            rejected=rejected,
            entities_new=entities_new,
            beliefs_changed=beliefs_changed,
            duration_ms=duration_ms,
            raw_output=raw,
            skill_fingerprint=self._skill_fingerprint,
            dry_run=dry_run,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _estimate_tokens(self, text: str) -> int:
        """Fast char/4 estimate. llama-cpp-python's tokenize() is authoritative
        but costs a forward through the vocab table; for budget guards the
        cheap estimate is fine and conservative-enough (chars/4 over-counts
        for ASCII, under-counts for dense tokens - net neutral)."""
        return len(text) // 4 + 8

    def _log_event(
        self,
        input_text: str,
        raw: str,
        valid: list[str],
        executed: int,
        rejected: list[tuple[str, str]],
        entities_new: list[str],
        beliefs_changed: list[tuple[str, str]],
        duration_ms: int,
        dry_run: bool,
    ) -> None:
        _log.info(
            "bonsai.ingest: input_chars=%d raw_chars=%d stmts=%d exec=%d "
            "rejected=%d entities=%d beliefs=%d dur_ms=%d skill=%s dry_run=%s",
            len(input_text), len(raw), len(valid), executed, len(rejected),
            len(entities_new), len(beliefs_changed), duration_ms,
            self._skill_fingerprint, dry_run,
        )
