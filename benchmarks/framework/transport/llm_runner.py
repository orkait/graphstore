"""Unified LLM caller for every bench runner.

Replaces:
  - `llm_batch.generate_all_answers` (ad-hoc async batch w/ fragile "no progress" quit)
  - `llm_client.llm_call` sync path (blocks, no concurrency control)
  - direct `openai.chat.completions.create` in run_beam.py

One pattern across all benches. One place to fix LLM issues.

Design:
  * Shared per-run concurrency cap (asyncio.Semaphore). Default derived from
    the provider chain: `:free` models -> 20 req/min (matches OpenRouter
    free-tier cap); local/paid -> higher.
  * Per-call retries with exponential backoff on empty/timeout/429.
  * 429 retry-after header honored when litellm surfaces it.
  * Provider fallback: on total failure for a call, try the next provider
    in the chain before giving up for that call.
  * No "no progress, quit round" heuristic. A slow round is not a broken
    run. The runner keeps going until every call has exhausted retries
    for every provider.

Usage:

    from benchmarks.framework.transport.llm_runner import LLMRunner
    from benchmarks.framework.transport.llm_client import _resolve_providers

    runner = LLMRunner(_resolve_providers())
    answers = await runner.call_many(prompts, max_tokens=1000)

Judge/helper calls:

    verdict = await runner.call_one(judge_prompt, max_tokens=500)
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass

_log = logging.getLogger(__name__)


# Defaults. Override via LLMRunner kwargs.
DEFAULT_TIMEOUT_S = 90
DEFAULT_RETRIES = 3
DEFAULT_MAX_TOKENS = 1000
FREE_TIER_RPM = 20            # OpenRouter :free cap
PAID_DEFAULT_RPM = 120        # conservative for paid hosted
LOCAL_DEFAULT_RPM = 0         # 0 = unlimited (local Ollama / self-hosted)

# Rate-limit window. We enforce RPM with a simple sliding window: if the
# last N timestamps fall within the last 60s, we wait until the oldest
# falls out. Simpler than a token bucket, matches "requests per minute".
_WINDOW_SECONDS = 60.0


@dataclass(slots=True)
class _Concurrency:
    rpm: int                                # 0 = unlimited
    timestamps: list[float]                 # monotonic ts of recent calls
    semaphore: asyncio.Semaphore            # concurrent in-flight cap

    def can_fire_now(self) -> float:
        """Return seconds to wait before next call allowed. 0 if now."""
        if self.rpm <= 0:
            return 0.0
        cutoff = time.monotonic() - _WINDOW_SECONDS
        # Drop stale timestamps.
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.pop(0)
        if len(self.timestamps) < self.rpm:
            return 0.0
        # Oldest in-window timestamp. Wait until it ages out.
        return max(0.0, self.timestamps[0] + _WINDOW_SECONDS - time.monotonic())


def _infer_rpm(providers: list[dict]) -> int:
    """Derive a concurrency cap from the provider chain.

    Pick the tightest cap: if any provider is :free, 20 rpm. If any is
    local, that provider is unlimited but the shared budget falls back
    to paid defaults since we always fallback to paid on local failure.
    """
    has_free = any(":free" in p.get("litellm_model", "") for p in providers)
    if has_free:
        return FREE_TIER_RPM
    has_local = any(
        "localhost" in p.get("api_base", "") or "127.0.0.1" in p.get("api_base", "")
        for p in providers
    )
    if has_local and len(providers) == 1:
        return LOCAL_DEFAULT_RPM
    return PAID_DEFAULT_RPM


def _parse_retry_after(err: Exception) -> float:
    """Best-effort parse of the retry-after hint on a 429 exception."""
    text = str(err)
    m = re.search(r"retry[- ]after[^0-9]*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    # Status-code hint
    if "429" in text or "RateLimit" in text or "rate_limit" in text:
        return 5.0
    return 0.0


class LLMRunner:
    """Shared LLM caller. One instance per run, used by every bench."""

    def __init__(
        self,
        providers: list[dict],
        *,
        rpm: int | None = None,
        max_concurrent: int | None = None,
        retries: int = DEFAULT_RETRIES,
        timeout_s: int = DEFAULT_TIMEOUT_S,
    ) -> None:
        if not providers:
            raise ValueError("LLMRunner needs at least one provider")
        self._providers = providers
        resolved_rpm = rpm if rpm is not None else _infer_rpm(providers)
        # Concurrent in-flight cap defaults to rpm (or 16 when unlimited).
        conc_cap = max_concurrent or (resolved_rpm if resolved_rpm > 0 else 16)
        self._c = _Concurrency(
            rpm=resolved_rpm,
            timestamps=[],
            semaphore=asyncio.Semaphore(conc_cap),
        )
        self._retries = max(1, retries)
        self._timeout_s = max(5, timeout_s)

    @property
    def rpm(self) -> int:
        """Requests-per-minute cap in effect. 0 = unlimited."""
        return self._c.rpm

    async def _wait_rate(self) -> None:
        """Block until the rate budget allows another call."""
        while True:
            wait = self._c.can_fire_now()
            if wait <= 0:
                self._c.timestamps.append(time.monotonic())
                return
            await asyncio.sleep(min(wait, 1.0))

    async def _litellm_call(
        self,
        prompt: str,
        provider: dict,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """One async call via litellm. Returns text, or raises on failure."""
        import litellm
        litellm.suppress_debug_info = True

        def _sync() -> str:
            resp = litellm.completion(
                model=provider["litellm_model"],
                messages=[{"role": "user", "content": prompt}],
                api_base=provider["api_base"],
                api_key=provider["api_key"],
                stream=False,
                timeout=self._timeout_s,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or ""
            # Strip <think> blocks from reasoning models just in case.
            return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, _sync),
            timeout=self._timeout_s + 10,
        )

    async def call_one(
        self,
        prompt: str,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ) -> str:
        """Execute a single prompt. Tries providers + retries.

        Returns empty string only after exhausting every provider and
        retry. Rate-limited by the shared semaphore + rpm window.
        """
        async with self._c.semaphore:
            for attempt in range(self._retries):
                for provider in self._providers:
                    await self._wait_rate()
                    try:
                        result = await self._litellm_call(prompt, provider, max_tokens, temperature)
                        if result:
                            return result
                        # Empty reply - treat as soft failure, try next.
                    except asyncio.TimeoutError:
                        _log.debug("timeout on %s attempt=%d", provider.get("pid"), attempt)
                        continue
                    except Exception as err:
                        wait = _parse_retry_after(err)
                        if wait > 0:
                            _log.info(
                                "rate limit on %s, sleeping %.1fs",
                                provider.get("pid"), wait,
                            )
                            await asyncio.sleep(wait)
                        else:
                            _log.debug(
                                "call failed on %s attempt=%d: %s",
                                provider.get("pid"), attempt, err,
                            )
                # Exponential backoff between retry rounds.
                backoff = 2 ** attempt
                await asyncio.sleep(backoff)
            return ""

    async def call_many(
        self,
        prompts: list[str],
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
        on_progress=None,
    ) -> list[str]:
        """Run N prompts concurrently.

        Concurrency is limited by the runner's semaphore + rpm window;
        the caller doesn't have to worry about batching. Returns a list
        aligned with the input order.
        """
        total = len(prompts)
        results: list[str] = [""] * total
        done = 0
        lock = asyncio.Lock()

        async def _worker(i: int, p: str) -> None:
            nonlocal done
            r = await self.call_one(p, max_tokens=max_tokens, temperature=temperature)
            async with lock:
                results[i] = r
                done += 1
                if on_progress:
                    on_progress(done, total)

        await asyncio.gather(*[_worker(i, p) for i, p in enumerate(prompts)])
        return results

    def call_sync(
        self,
        prompt: str,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ) -> str:
        """Sync convenience wrapper for non-async callers (e.g. test code).

        Uses ``asyncio.run``; safe only from non-async contexts. Async
        callers should use ``call_one``.
        """
        return asyncio.run(self.call_one(prompt, max_tokens=max_tokens, temperature=temperature))


# ---------------------------------------------------------------------------
# Process-wide singleton: every bench / helper uses the same runner so the
# rpm budget + concurrency cap span the whole process.
# ---------------------------------------------------------------------------

_SHARED: "LLMRunner | None" = None


def get_shared_runner() -> "LLMRunner":
    """Return (or construct) the process-wide LLMRunner.

    Provider chain comes from `llm_client._resolve_providers()` which reads
    `tools/autoresearch/config.json`. If the config changes between calls,
    call `reset_shared_runner()` to rebuild.
    """
    global _SHARED
    if _SHARED is None:
        from benchmarks.framework.transport.llm_client import _resolve_providers
        providers = _resolve_providers()
        if not providers:
            raise RuntimeError(
                "No LLM providers resolved. Check tools/autoresearch/config.json "
                "active_provider + provider_fallback_order."
            )
        _SHARED = LLMRunner(providers)
    return _SHARED


def reset_shared_runner() -> None:
    """Drop the cached shared runner so the next call rebuilds it."""
    global _SHARED
    _SHARED = None
