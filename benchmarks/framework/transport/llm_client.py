"""Unified LLM client for LoCoMo benchmark QA.

Uses litellm with autoresearch config for provider fallback.
Primary: minimax-m2.7:cloud (Ollama) or minimax/minimax-m2.7:nitro (OpenRouter)
"""

from __future__ import annotations

import json
import logging
import os
import re
import string
from pathlib import Path

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "tools" / "autoresearch" / "config.json"

# Model preference for LoCoMo QA.
# Dual-name: Ollama cloud tag comes first so local_ollama wins when present;
# OpenRouter is the paid fallback.
#
# gemma4:31b-cloud chosen over minimax-m2.7 because gemma4 is a
# non-reasoning model. Reasoning models (minimax) emit variable-length
# thinking tokens that introduce jitter even at temperature=0.0; this
# caused LLM-judge verdicts to flip on identical (pred, gold) pairs
# between smoke runs. gemma4 produces deterministic single-shot answers.
QA_MODEL = "gemma4:31b-cloud"
QA_MODEL_OR = "google/gemma-4-31b-it"  # paid fallback when Ollama unreachable


def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        return json.loads(_CONFIG_PATH.read_text())
    return {}


def llm_call(prompt: str, max_tokens: int = 1000, temperature: float = 0.0, _retries: int | None = None) -> str:
    """Sync LLM call. Delegates to the shared LLMRunner.

    Keeps the legacy signature for in-tree callers (`compute_llm_judge`,
    `llm_judge.llm_call`, existing tests). The runner handles rate limit +
    retry + provider fallback centrally. The ``_retries`` kwarg is kept
    for back-compat but ignored (runner retries internally).
    """
    from benchmarks.framework.transport.llm_runner import get_shared_runner
    return get_shared_runner().call_sync(prompt, max_tokens=max_tokens, temperature=temperature)


def _resolve_providers() -> list[dict]:
    """Resolve available LLM providers with model + litellm config.

    Returns list of provider dicts with keys: litellm_model, api_base, api_key.
    """
    import litellm
    litellm.suppress_debug_info = True

    config = _load_config()
    providers = config.get("providers", {})
    active_pid = config.get("active_provider", "")
    provider_order = [active_pid] + [
        p for p in config.get("provider_fallback_order", []) if p != active_pid
    ]
    provider_order = [p for p in dict.fromkeys(provider_order) if p in providers]

    resolved = []
    for pid in provider_order:
        p = providers.get(pid)
        if not p:
            continue
        base_url = p.get("base_url", "")
        api_key = (p.get("api_key", "")
                   or os.environ.get(p.get("api_key_env", ""), "")
                   or "ollama")
        if not base_url:
            continue
        available = p.get("models", {})
        model_order = [m for m in [QA_MODEL, QA_MODEL_OR] if m in available]
        if not model_order:
            continue
        is_local = p.get("is_local", "localhost" in base_url or "127.0.0.1" in base_url)
        prefix = p.get("litellm_prefix") or ("ollama_chat" if is_local else "")
        model = model_order[0]
        litellm_model = f"{prefix}/{model}" if prefix else model
        resolved.append({
            "pid": pid,
            "litellm_model": litellm_model,
            "api_base": base_url,
            "api_key": api_key,
        })
    return resolved


def llm_call_on_provider(prompt: str, provider: dict, max_tokens: int = 1000, temperature: float = 0.0) -> str:
    """Call LLM on a specific provider with streaming. Returns empty string on failure."""
    import litellm
    litellm.suppress_debug_info = True
    try:
        response = litellm.completion(
            model=provider["litellm_model"],
            messages=[{"role": "user", "content": prompt}],
            api_base=provider["api_base"],
            api_key=provider["api_key"],
            stream=True,
            timeout=90,
            temperature=temperature,
            max_tokens=max_tokens,
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


def health_check() -> bool:
    """Verify LLM is reachable. Call before starting a benchmark."""
    result = llm_call("Say OK", max_tokens=500)
    if not result:
        raise RuntimeError(
            "LLM health check failed: got empty response. "
            "Check that minimax-m2.7:cloud (Ollama) or minimax/minimax-m2.7:nitro (OpenRouter) is available."
        )
    return True


def generate_answer(question: str, context_texts: list[str], scored_nodes: list[dict] | None = None) -> str:
    """Generate answer from retrieved context.

    Prompt matches official LoCoMo QA_PROMPT from snap-research/locomo.
    """
    context = "\n\n".join(f"[{i+1}]: {t}" for i, t in enumerate(context_texts))
    # Matches official LoCoMo QA_PROMPT
    prompt = (
        f"Based on the below context, write an answer in the form of a short phrase "
        f"for the following question. Answer with exact words from the context whenever possible.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question} Short answer:"
    )
    return llm_call(prompt, max_tokens=1000, temperature=0.0)


def _normalize_answer(s: str) -> str:
    """Normalize answer string - matches official LoCoMo evaluation.py."""
    s = s.replace(',', '')
    s = re.sub(r'\b(a|an|the|and)\b', ' ', s.lower())
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    return ' '.join(s.split())


def _f1_score(prediction: str, gold: str) -> float:
    """Token-level F1 with Porter stemming - matches official LoCoMo."""
    from collections import Counter
    try:
        from nltk.stem import PorterStemmer
        _stemmer = PorterStemmer()
        stem = _stemmer.stem
    except ImportError:
        stem = lambda w: w

    pred_tokens = [stem(w) for w in _normalize_answer(prediction).split()]
    gold_tokens = [stem(w) for w in _normalize_answer(gold).split()]
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def compute_f1(prediction: str, gold: str, category: int | None = None) -> float:
    """Compute F1 matching official LoCoMo protocol.

    - Categories 2,3,4 (single-hop, temporal, open-domain): direct F1
    - Category 1 (multi-hop): split comma-separated sub-answers, partial F1 each
    - Category 5 (adversarial): check for "no information available" / "not mentioned"
    """
    if category == 5:
        low = prediction.lower()
        if 'no information available' in low or 'not mentioned' in low:
            return 1.0
        return 0.0

    if category == 1:
        import numpy as np
        preds = [p.strip() for p in prediction.split(',')]
        golds = [g.strip() for g in gold.split(',')]
        return float(np.mean([max([_f1_score(p, g) for p in preds]) for g in golds]))

    return _f1_score(prediction, gold)


# ---------------------------------------------------------------------------
# LLM-as-judge scoring (semantic equivalence, more lenient than token F1)
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are grading a short-answer question.

Question: {question}
Gold answer: {gold}
Predicted answer: {prediction}

Mark the predicted answer CORRECT if it is semantically equivalent to the
gold answer OR captures the same essential fact, even with different
wording, abbreviation, or paraphrase. Mark it INCORRECT only if it
contradicts the gold answer, omits the essential fact, or fabricates
unsupported content.

For adversarial questions (gold answer is "no information available" or
similar), CORRECT requires the prediction to also acknowledge that no
information is available. Any confidently-stated answer is INCORRECT.

Respond with a single word: CORRECT or INCORRECT.
"""


def compute_llm_judge(
    prediction: str,
    gold: str,
    question: str,
    category: int | None = None,
    debug: bool = False,
) -> float:
    """Semantic-equivalence judge via llm_call.

    Returns 1.0 if LLM says CORRECT, 0.0 if INCORRECT, 0.0 on parse failure.
    Costs 1 LLM call per QA. Used with --judge llm on run_locomo.

    Empty predictions short-circuit to 0.0 (nothing to judge). Adversarial
    category: if gold is None / empty / "no information available" and pred
    is similarly empty-sounding, credit without calling the LLM.

    Reasoning models (like minimax-m2.7) need a generous max_tokens so
    reasoning tokens don't starve the verdict. Parsing looks for CORRECT
    or INCORRECT anywhere in the verdict, not just the first token.
    """
    pred = (prediction or "").strip()
    if not pred:
        return 0.0
    if category == 5:
        low = pred.lower()
        if any(phrase in low for phrase in (
            "no information available", "not mentioned", "cannot answer",
            "don't know", "do not know", "not enough information",
        )):
            return 1.0
        return 0.0

    prompt = _JUDGE_PROMPT.format(
        question=(question or "").strip(),
        gold=(gold or "").strip(),
        prediction=pred,
    )
    # 2000 tokens: plenty for reasoning-model thinking + the final verdict word.
    verdict = llm_call(prompt, max_tokens=2000, temperature=0.0)
    if debug:
        print(f"  [judge] raw verdict: {verdict!r}")
    if not verdict:
        return 0.0
    # Search the whole verdict for CORRECT vs INCORRECT. Reasoning models
    # may emit thinking before the final word; we take the LAST occurrence.
    upper = verdict.upper()
    last_incorrect = upper.rfind("INCORRECT")
    last_correct = upper.rfind("CORRECT")
    if last_incorrect == -1 and last_correct == -1:
        return 0.0
    # "INCORRECT" contains "CORRECT" as a substring, so prefer whichever
    # appears later as a standalone word. If INCORRECT is found at position
    # P, then CORRECT will be found at position P+2; discount that.
    if last_incorrect != -1 and (last_correct == -1 or last_correct <= last_incorrect + 2):
        return 0.0
    return 1.0
