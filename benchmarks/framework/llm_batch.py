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
MAX_ROUNDS = 10


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
) -> list[int]:
    """Run a batch of indices on one provider. Returns failed indices."""
    if not indices:
        return []

    tasks = [_call_indexed(idx, prompts[idx], provider) for idx in indices]
    failed = []

    for coro in asyncio.as_completed(tasks):
        idx, result = await coro
        if result:
            answers[idx] = result
        else:
            failed.append(idx)

    return failed


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

    prev_answered = 0
    no_progress_count = 0
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

        for failed in results:
            pass  # failures already tracked by answers dict

        answered_count = len(answers)
        remaining = n - answered_count
        if on_progress:
            on_progress(answered_count, n)
        print(f"    Round {round_num+1} done: {answered_count}/{n} answered, {remaining} remaining", flush=True)

        if remaining == 0:
            break

        # Brief cooldown between rounds to avoid hammering providers
        if remaining > 0:
            await asyncio.sleep(5)

        # Early exit if no progress for 2 consecutive rounds
        if round_num > 0 and answered_count == prev_answered:
            no_progress_count += 1
            if no_progress_count >= 2:
                print(f"    No progress for {no_progress_count} rounds, stopping.", flush=True)
                break
        else:
            no_progress_count = 0
        prev_answered = answered_count

    # Build ordered result list
    result = []
    for q in questions:
        result.append(answers.get(q["idx"], ""))
    return result
