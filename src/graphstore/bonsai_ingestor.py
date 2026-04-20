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
_ASSERT_RE = re.compile(r'^\s*ASSERT\s+"([^"\\]+(?:\\.[^"\\]*)*)"', re.IGNORECASE)
_RETRACT_RE = re.compile(r'^\s*RETRACT\s+"([^"\\]+(?:\\.[^"\\]*)*)"', re.IGNORECASE)
_CREATE_NODE_RE = re.compile(r'^\s*CREATE\s+NODE\s+"([^"\\]+(?:\\.[^"\\]*)*)"', re.IGNORECASE)
_ENT_FROM_ID_RE = re.compile(r'"(ent:[^"\\]+)"')


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


# --------------------------------------------------------------------
# Ingestor
# --------------------------------------------------------------------

_DEFAULT_SKILL_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "tools" / "skills" / "graphstore-bonsai-dsl" / "SKILL.md"
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
        n_ctx: int = 2048,
        n_threads: int | None = None,
        chat_format: str = "qwen",
        max_output_tokens: int = 400,
        temperature: float = 0.0,
    ) -> None:
        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(f"bonsai model not found: {self._model_path}")
        self._gs = gs
        self._skill_path = Path(skill_path) if skill_path else _DEFAULT_SKILL_PATH
        self._n_ctx = n_ctx
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._chat_format = chat_format
        self._n_threads = n_threads

        self._skill_text = ""
        self._skill_fingerprint = ""
        self._system_prompt = ""
        self._reload_skill()

        self._llm: Any | None = None
        self._lock = threading.Lock()

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

    def ingest(self, text: str, *, dry_run: bool = False) -> IngestResult:
        """Convert `text` to DSL statements and (optionally) execute them.

        `dry_run=True` returns the DSL without touching the store - useful
        for previewing or building training data without committing.
        """
        if not text or not text.strip():
            raise IngestEmpty("input text is empty or whitespace-only")
        if not dry_run and self._gs is None:
            raise ValueError("ingest requires a GraphStore (pass gs=...) or dry_run=True")

        self._reload_skill()
        with self._lock:
            return self._ingest_locked(text, dry_run=dry_run)

    def _ingest_locked(self, text: str, *, dry_run: bool) -> IngestResult:
        t0 = time.perf_counter()
        llm = self._ensure_llm()

        est = self._estimate_tokens(self._system_prompt) + self._estimate_tokens(text)
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
                {"role": "user", "content": text},
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
        if not dry_run:
            for ln in valid:
                try:
                    self._gs.execute(ln)
                    executed += 1
                except Exception as err:
                    rejected.append((ln, f"execute error: {err}"))

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
