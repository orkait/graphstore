"""LLM-based QA evaluation for LongMemEval.

Follows the EXACT official LongMemEval evaluation protocol from:
https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py

Uses litellm with the same provider/model config as autoresearch.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "autoresearch" / "config.json"

QA_MODEL = "gemma4:31b-cloud"
QA_FALLBACK = "qwen3.5:cloud"


def _load_llm_config() -> dict:
    if _CONFIG_PATH.exists():
        return json.loads(_CONFIG_PATH.read_text())
    return {}


def llm_call(prompt: str, config: dict | None = None, temperature: float = 0.0, max_tokens: int = 512) -> str:
    """Call LLM using litellm with autoresearch-style provider fallback."""
    import litellm
    litellm.suppress_debug_info = True

    if config is None:
        config = _load_llm_config()
    config = {**config, "active_model": QA_MODEL}

    providers = config.get("providers", {})
    active_pid = config.get("active_provider", "")
    active_model = config.get("active_model", "")
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
        model_order = [active_model, QA_FALLBACK]
        model_order = [m for m in dict.fromkeys(model_order) if m and m in available]
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

    return ""


# ---------------------------------------------------------------------------
# Official LongMemEval judge prompts (verbatim from evaluate_qa.py)
# https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py
# ---------------------------------------------------------------------------

_JUDGE_PROMPTS = {
    "single-session-user": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response is equivalent to the correct answer or contains all the intermediate "
        "steps to get the correct answer, you should also answer yes. If the response only "
        "contains a subset of the information required by the answer, answer no. "
        "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}"
        "\n\nIs the model response correct? Answer yes or no only."
    ),
    "single-session-assistant": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response is equivalent to the correct answer or contains all the intermediate "
        "steps to get the correct answer, you should also answer yes. If the response only "
        "contains a subset of the information required by the answer, answer no. "
        "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}"
        "\n\nIs the model response correct? Answer yes or no only."
    ),
    "multi-session": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response is equivalent to the correct answer or contains all the intermediate "
        "steps to get the correct answer, you should also answer yes. If the response only "
        "contains a subset of the information required by the answer, answer no. "
        "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}"
        "\n\nIs the model response correct? Answer yes or no only."
    ),
    "temporal-reasoning": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response is equivalent to the correct answer or contains all the intermediate "
        "steps to get the correct answer, you should also answer yes. If the response only "
        "contains a subset of the information required by the answer, answer no. "
        "In addition, do not penalize off-by-one errors for the number of days. "
        "If the question asks for the number of days/weeks/months, etc., and the model makes "
        "off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's "
        "response is still correct. "
        "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}"
        "\n\nIs the model response correct? Answer yes or no only."
    ),
    "knowledge-update": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response contains some previous information along with an updated answer, "
        "the response should be considered as correct as long as the updated answer is the "
        "required answer."
        "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}"
        "\n\nIs the model response correct? Answer yes or no only."
    ),
    "single-session-preference": (
        "I will give you a question, a rubric for desired personalized response, and a "
        "response from a model. Please answer yes if the response satisfies the desired "
        "response. Otherwise, answer no. The model does not need to reflect all the points "
        "in the rubric. The response is correct as long as it recalls and utilizes the "
        "user's personal information correctly."
        "\n\nQuestion: {question}\n\nRubric: {answer}\n\nModel Response: {response}"
        "\n\nIs the model response correct? Answer yes or no only."
    ),
}

_DEFAULT_JUDGE_PROMPT = _JUDGE_PROMPTS["single-session-user"]


def generate_answer(question: str, retrieved_texts: list[str]) -> str:
    """Generate an answer from retrieved context using LLM."""
    context = "\n\n".join(f"[Memory {i+1}]: {t}" for i, t in enumerate(retrieved_texts))

    prompt = (
        "You are a helpful chat assistant with access to memories from past conversations. "
        "Answer the question based ONLY on the retrieved memories below. Be concise and specific.\n\n"
        f"Retrieved memories:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    return llm_call(prompt, temperature=0.1, max_tokens=256)


def judge_answer(
    question: str,
    gold_answers: list[str],
    hypothesis: str,
    category: str | None = None,
) -> bool:
    """Judge if the generated answer is correct using official LongMemEval protocol.

    Uses the exact per-category judge prompts from the official eval script.
    Scoring: "yes" in response.lower() -> correct, else incorrect.
    """
    answer_str = "; ".join(gold_answers)
    template = _JUDGE_PROMPTS.get(category or "", _DEFAULT_JUDGE_PROMPT)
    prompt = template.format(question=question, answer=answer_str, response=hypothesis)

    response = llm_call(prompt, temperature=0.0, max_tokens=10)
    return "yes" in response.lower()
