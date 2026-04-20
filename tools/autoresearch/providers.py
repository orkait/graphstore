"""Shared provider + model resolution from autoresearch config.json.

Single source of truth for reading config.json and building the ordered
(provider, model) candidate list used by both the bench transport layer
and the autoresearch run_loop.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def load_config() -> dict:
    if _CONFIG_PATH.exists():
        return json.loads(_CONFIG_PATH.read_text())
    return {}


def resolve_providers(
    config: dict,
    model_priority: list[str] | None = None,
) -> list[dict]:
    """Build an ordered list of (provider, model) candidates.

    Returns a flat list of dicts: {pid, litellm_model, api_base, api_key}.
    Ordered by provider_fallback_order (active provider first).

    model_priority: explicit ordered model names to prefer.
        For each provider only models in this list are included; first match
        per provider is returned (one entry per provider). Used by bench
        runners that target a specific eval model (e.g. gemma4:31b-cloud).

        None: use active_model + provider's model_fallback_order. All matching
        models per provider are returned in order. Used by autoresearch which
        tries every model in sequence before giving up.
    """
    providers = config.get("providers", {})
    active_pid = config.get("active_provider", "")
    provider_order = [active_pid] + [
        p for p in config.get("provider_fallback_order", []) if p != active_pid
    ]
    provider_order = [p for p in dict.fromkeys(provider_order) if p in providers]

    result: list[dict] = []
    for pid in provider_order:
        p = providers.get(pid)
        if not p:
            continue
        base_url = p.get("base_url", "")
        if not base_url:
            continue
        api_key = (
            p.get("api_key", "")
            or os.environ.get(p.get("api_key_env", ""), "")
            or "ollama"
        )
        is_local = p.get("is_local", "localhost" in base_url or "127.0.0.1" in base_url)
        prefix = p.get("litellm_prefix") or ("ollama_chat" if is_local else "")
        available = p.get("models", {})

        if model_priority is not None:
            selected = [m for m in model_priority if m in available][:1]
        else:
            active_model = config.get("active_model", "")
            fallbacks = list(p.get("model_fallback_order", []))
            order = [active_model] + fallbacks
            selected = [m for m in dict.fromkeys(order) if m and m in available]

        for model in selected:
            litellm_model = f"{prefix}/{model}" if prefix else model
            result.append({
                "pid": pid,
                "litellm_model": litellm_model,
                "api_base": base_url,
                "api_key": api_key,
            })

    return result
