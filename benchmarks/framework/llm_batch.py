"""Async batch LLM caller with dual-provider rotation and fast-fail detection.

Design:
    - 8 calls per provider per batch (never more)
    - 90s hard timeout per call
    - If a provider's batch returns mostly empty within 5s -> rate-limited, skip it
    - Remaining unanswered questions loop to next batch with provider rotation
    - Keeps going until all answered or max rounds exhausted
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

_log = logging.getLogger(__name__)

BATCH_SIZE = 8          # max concurrent per provider
CALL_TIMEOUT = 90       # seconds per LLM call
FAST_FAIL_TIME = 5      # if all empties arrive within this, provider is rate-limited
FAST_FAIL_RATIO = 0.75  # if this fraction of batch is empty within FAST_FAIL_TIME, skip provider
MAX_ROUNDS = 15


async def _call_one(
    prompt: str,
    provider: dict,
    timeout: float = CALL_TIMEOUT,
) -> str:
    """Single async LLM call via litellm (run in thread since litellm is sync)."""
    import litellm
    litellm.suppress_debug_info = True

    def _sync_call():
        try:
            response = litellm.completion(
                model=provider["litellm_model"],
                messages=[{"role": "user", "content": prompt}],
                api_base=provider["api_base"],
                api_key=provider["api_key"],
                stream=True,
                timeout=int(timeout),
                temperature=0.0,
                max_tokens=1000,
            )
            chunks = []
            for chunk in response:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    chunks.append(delta)
            content = "".join(chunks)
            return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        except Exception:
            return ""

    loop = asyncio.get_event_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, _sync_call),
            timeout=timeout + 10,  # extra buffer over litellm timeout
        )
    except asyncio.TimeoutError:
        return ""


async def _call_indexed(idx: int, prompt: str, provider: dict) -> tuple[int, str]:
    """Call with index tracking."""
    result = await _call_one(prompt, provider)
    return idx, result


async def _run_batch(
    indices: list[int],
    prompts: dict[int, str],
    provider: dict,
    answers: dict[int, str],
) -> tuple[list[int], bool]:
    """Run a batch of indices on one provider.

    Returns (failed_indices, was_rate_limited).
    Rate-limited = most results came back empty within FAST_FAIL_TIME.
    """
    if not indices:
        return [], False

    start = time.monotonic()
    tasks = [_call_indexed(idx, prompts[idx], provider) for idx in indices]

    failed = []
    empty_count = 0
    answered_count = 0
    rate_limited = False

    for coro in asyncio.as_completed(tasks):
        idx, result = await coro

        if result:
            answers[idx] = result
            answered_count += 1
        else:
            empty_count += 1
            failed.append(idx)

        # Fast-fail: if most empties arrive quickly, provider is rate-limited
        elapsed = time.monotonic() - start
        total_done = empty_count + answered_count
        if (elapsed < FAST_FAIL_TIME
                and total_done >= len(indices) * FAST_FAIL_RATIO
                and empty_count > answered_count * 2):
            rate_limited = True

    return failed, rate_limited


async def generate_all_answers(
    questions: list[dict],
    providers: list[dict],
    on_progress: Any = None,
) -> list[str]:
    """Generate answers for all questions using batch rotation across providers.

    Args:
        questions: list of {"idx": int, "prompt": str}
        providers: from _resolve_providers()
        on_progress: callable(answered, total) for progress updates

    Returns list of answer strings (same order as questions).
    """
    n = len(questions)
    n_prov = len(providers)
    if n_prov == 0:
        return [""] * n

    prompts = {q["idx"]: q["prompt"] for q in questions}
    all_indices = [q["idx"] for q in questions]
    answers: dict[int, str] = {}

    for round_num in range(MAX_ROUNDS):
        pending = [i for i in all_indices if i not in answers]
        if not pending:
            break

        # Split pending into batches of BATCH_SIZE per provider
        # Interleave across providers
        provider_batches: list[tuple[list[int], dict]] = []
        for pi in range(n_prov):
            chunk = [pending[j] for j in range(pi, len(pending), n_prov)]
            # Split into sub-batches of BATCH_SIZE
            for start in range(0, len(chunk), BATCH_SIZE):
                batch = chunk[start:start + BATCH_SIZE]
                provider_batches.append((batch, providers[pi]))

        pid_names = [p.get("pid", "?") for p in providers]
        print(
            f"    Round {round_num+1}: {len(pending)} pending, "
            f"{len(provider_batches)} batches across {pid_names}",
            flush=True,
        )

        # Run all batches concurrently (each batch is 8 calls to one provider)
        batch_tasks = []
        for batch_indices, provider in provider_batches:
            batch_tasks.append(_run_batch(batch_indices, prompts, provider, answers))

        results = await asyncio.gather(*batch_tasks)

        # Collect rate-limited providers
        rl_providers = set()
        all_failed = []
        for (failed, was_rl), (_, provider) in zip(results, provider_batches):
            all_failed.extend(failed)
            if was_rl:
                rl_providers.add(provider.get("pid", ""))

        answered_count = len(answers)
        remaining = n - answered_count
        if on_progress:
            on_progress(answered_count, n)
        print(
            f"    Round {round_num+1} done: {answered_count}/{n} answered, {remaining} remaining"
            + (f" (rate-limited: {rl_providers})" if rl_providers else ""),
            flush=True,
        )

        if remaining == 0:
            break

        # Cooldown when rate-limited - let the window reset
        if rl_providers:
            wait = 10 if len(rl_providers) < n_prov else 15
            print(f"    Cooldown {wait}s (rate-limited providers: {rl_providers})...", flush=True)
            await asyncio.sleep(wait)

    # Build ordered result list
    result = []
    for q in questions:
        result.append(answers.get(q["idx"], ""))
    return result
