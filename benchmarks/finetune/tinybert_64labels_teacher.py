from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, Sequence

from benchmarks.finetune.tinybert_64labels_prompt import build_teacher_messages
from benchmarks.finetune.tinybert_ner_64labels.labels import build_label_maps

_ROOT = Path(__file__).resolve().parent.parent.parent
_WORKSPACE_DIR = _ROOT / "benchmarks" / "finetune" / "tinybert-ner-64labels"
_CONFIG_PATHS = (
    _WORKSPACE_DIR / "config.json",
    _ROOT / "autoresearch" / "config.json",
)
_CONFIG_EXAMPLE_PATHS = (
    _WORKSPACE_DIR / "config.example.json",
    _ROOT / "autoresearch" / "config.example.json",
)

_FENCE_RE = re.compile(r"^```(?:json)?\s*$|^```\s*$", re.IGNORECASE)


def load_litellm_config(config_path: str | Path | None = None) -> dict:
    path = Path(config_path) if config_path else None
    if path and path.exists():
        return json.loads(path.read_text())
    env_path = os.environ.get("GRAPHSTORE_LITELLM_CONFIG")
    if env_path and Path(env_path).exists():
        return json.loads(Path(env_path).read_text())
    for candidate in _CONFIG_PATHS:
        if candidate.exists():
            return json.loads(candidate.read_text())
    for candidate in _CONFIG_EXAMPLE_PATHS:
        if candidate.exists():
            return json.loads(candidate.read_text())
    return {}


def resolve_teacher_provider(config: dict, teacher_model: str | None = None) -> dict:
    providers = config.get("providers", {})
    active_pid = config.get("active_provider", "")
    provider_order = [active_pid] + [
        p for p in config.get("provider_fallback_order", []) if p != active_pid
    ]
    provider_order = [p for p in dict.fromkeys(provider_order) if p in providers]
    requested = (
        teacher_model
        or config.get("active_model")
        or os.environ.get("TINYBERT_NER_TEACHER_MODEL", "")
        or "gemma4:31b-cloud"
        or "minimax-m2.7:cloud"
    )

    for pid in provider_order:
        provider = providers.get(pid) or {}
        base_url = provider.get("base_url", "")
        if not base_url:
            continue
        available = provider.get("models", {})
        model_order = [m for m in [requested, config.get("active_model"), os.environ.get("TINYBERT_NER_TEACHER_MODEL", "")] if m]
        model_order.extend(provider.get("model_fallback_order", []))
        model_order.extend(list(available.keys()))
        model_order = [m for m in dict.fromkeys(model_order) if m in available]
        if not model_order:
            continue
        is_local = provider.get("is_local", "localhost" in base_url or "127.0.0.1" in base_url)
        prefix = provider.get("litellm_prefix") or ("ollama_chat" if is_local else "")
        model = model_order[0]
        litellm_model = f"{prefix}/{model}" if prefix else model
        api_key = (
            provider.get("api_key", "")
            or os.environ.get(provider.get("api_key_env", ""), "")
            or "ollama"
        )
        return {
            "pid": pid,
            "litellm_model": litellm_model,
            "api_base": base_url,
            "api_key": api_key,
            "model_name": model,
        }
    raise RuntimeError("no usable LiteLLM provider found for teacher generation")


def call_teacher_llm(
    messages: list[dict[str, str]],
    *,
    config: dict | None = None,
    teacher_model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> str:
    import litellm

    litellm.suppress_debug_info = True
    cfg = config or load_litellm_config()
    provider = resolve_teacher_provider(cfg, teacher_model=teacher_model)
    response = litellm.completion(
        model=provider["litellm_model"],
        messages=messages,
        api_base=provider["api_base"],
        api_key=provider["api_key"],
        stream=False,
        timeout=120,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or ""
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def _iter_json_objects(text: str) -> Iterable[dict]:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or _FENCE_RE.match(line):
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def _normalize_tags(tags: Sequence[str], label2id: dict[str, int]) -> list[str]:
    out: list[str] = []
    for tag in tags:
        tag = tag.strip()
        if tag == "O":
            out.append(tag)
        elif "-" not in tag:
            out.append(f"B-{tag}")
        else:
            prefix, entity = tag.split("-", 1)
            prefix = prefix.upper()
            if prefix not in {"B", "I"}:
                prefix = "B"
            out.append(f"{prefix}-{entity}")
        if out[-1] not in label2id:
            raise ValueError(f"unknown BIO tag: {out[-1]}")
    return out


def parse_teacher_output(
    text: str,
    *,
    source_label: str,
    teacher_model: str,
    examples_per_label: int,
) -> list[dict]:
    maps = build_label_maps()
    records: list[dict] = []
    for idx, obj in enumerate(_iter_json_objects(text)):
        tokens = obj.get("tokens")
        tags = obj.get("ner_tags")
        if not isinstance(tokens, list) or not isinstance(tags, list):
            continue
        if len(tokens) != len(tags):
            continue
        normalized_tags = _normalize_tags([str(tag) for tag in tags], dict(maps.label2id))
        records.append(
            {
                "sentence_id": f"{source_label.lower()}-{idx:04d}",
                "source_label": source_label,
                "difficulty": obj.get("difficulty", "standard"),
                "teacher_model": teacher_model,
                "tokens": [str(token) for token in tokens],
                "ner_tags": normalized_tags,
            }
        )
        if len(records) >= examples_per_label:
            break
    return records


def _prompt_label_difficulty(label: str) -> str:
    hard_labels = {
        "VARIABLE_NAME",
        "FUNCTION_NAME",
        "CLASS_NAME",
        "PROG_LANG",
        "LIBRARY",
        "DOMAIN_NAME",
        "URL",
        "DATE",
        "TIME",
        "DURATION",
        "CLOUD_SERVICE",
        "CLOUD_PROVIDER",
    }
    return "hard" if label in hard_labels else "standard"


def generate_label_examples(
    label: str,
    *,
    examples_per_label: int = 100,
    teacher_model: str | None = None,
    config: dict | None = None,
    max_rounds: int = 6,
) -> list[dict]:
    records: list[dict] = []
    cfg = config or load_litellm_config()
    provider = resolve_teacher_provider(cfg, teacher_model=teacher_model)
    for round_idx in range(max_rounds):
        if len(records) >= examples_per_label:
            break
        remaining = examples_per_label - len(records)
        messages = build_teacher_messages(label, examples_per_label=remaining)
        content = call_teacher_llm(
            messages,
            config=cfg,
            teacher_model=teacher_model or provider["model_name"],
        )
        parsed = parse_teacher_output(
            content,
            source_label=label,
            teacher_model=provider["model_name"],
            examples_per_label=remaining,
        )
        for record in parsed:
            record["difficulty"] = _prompt_label_difficulty(label)
        records.extend(parsed)
    if len(records) < examples_per_label:
        raise RuntimeError(
            f"teacher failed to generate enough records for {label}: "
            f"{len(records)}/{examples_per_label}"
        )
    return records[:examples_per_label]


def generate_dataset(
    *,
    examples_per_label: int = 100,
    teacher_model: str | None = None,
    config: dict | None = None,
) -> list[dict]:
    dataset: list[dict] = []
    for label in build_label_maps().semantic_labels:
        dataset.extend(
            generate_label_examples(
                label,
                examples_per_label=examples_per_label,
                teacher_model=teacher_model,
                config=config,
            )
        )
    return dataset


def save_jsonl(records: Sequence[dict], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=True))
            fh.write("\n")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tinybert_ner_teacher")
    parser.add_argument("--out", required=True, help="JSONL output path")
    parser.add_argument("--examples-per-label", type=int, default=100)
    parser.add_argument("--teacher-model", default=None)
    parser.add_argument("--config", default=None, help="LiteLLM/Ollama config path")
    args = parser.parse_args(argv)

    cfg = load_litellm_config(args.config)
    records = generate_dataset(
        examples_per_label=args.examples_per_label,
        teacher_model=args.teacher_model,
        config=cfg,
    )
    save_jsonl(records, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
