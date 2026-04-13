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

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "autoresearch" / "config.json"

# Model preference for LoCoMo QA (fast, no reasoning overhead)
QA_MODEL = "gemma4:31b-cloud"
# Same model family on OpenRouter as fallback
QA_MODEL_OR = "google/gemma-4-31b-it:free"


def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        return json.loads(_CONFIG_PATH.read_text())
    return {}


def llm_call(prompt: str, max_tokens: int = 1000, temperature: float = 0.0, _retries: int = 2) -> str:
    """Call LLM using litellm with autoresearch provider fallback.

    Retries on empty response (MiniMax sometimes returns empty content
    when reasoning tokens consume the budget or during transient outages).
    """
    import litellm
    litellm.suppress_debug_info = True

    config = _load_config()
    config = {**config, "active_model": QA_MODEL}

    providers = config.get("providers", {})
    active_pid = config.get("active_provider", "")
    provider_order = [active_pid] + [
        p for p in config.get("provider_fallback_order", []) if p != active_pid
    ]
    provider_order = [p for p in dict.fromkeys(provider_order) if p in providers]

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
        # Same model across providers for consistent answers
        model_order = [m for m in [QA_MODEL, QA_MODEL_OR] if m in available]
        if not model_order:
            continue

        is_local = p.get("is_local", "localhost" in base_url or "127.0.0.1" in base_url)
        prefix = p.get("litellm_prefix") or ("ollama_chat" if is_local else "")
        for model in model_order:
            litellm_model = f"{prefix}/{model}" if prefix else model
            try:
                response = litellm.completion(
                    model=litellm_model,
                    messages=[{"role": "user", "content": prompt}],
                    api_base=base_url,
                    api_key=api_key,
                    stream=False,
                    timeout=30,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content or ""
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                if content:
                    return content
            except Exception:
                continue

    # All providers/models returned empty - retry if attempts remain
    if _retries > 0:
        import time
        time.sleep(1)
        return llm_call(prompt, max_tokens=max_tokens, temperature=temperature, _retries=_retries - 1)
    return ""


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
    result = llm_call("Say OK", max_tokens=10)
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
