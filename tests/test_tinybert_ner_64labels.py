from __future__ import annotations

import json

from benchmarks.finetune.tinybert_ner_64labels.labels import build_label_maps
from benchmarks.finetune.tinybert_64labels_prompt import build_teacher_prompt
from benchmarks.finetune.tinybert_64labels_teacher import load_litellm_config, resolve_teacher_provider
from benchmarks.finetune.tinybert_ner_64labels.train import align_wordpiece_labels


def test_label_vocab_is_129():
    maps = build_label_maps()

    assert len(maps.id2label) == 129
    assert maps.id2label[0] == "O"
    assert maps.label2id["B-PROG_LANG"] == 1
    assert maps.label2id["I-MISC_ENTITY"] == 128


def test_teacher_prompt_forces_balanced_bio_examples():
    prompt = build_teacher_prompt("FUNCTION_NAME", examples_per_label=100)

    assert "BIO" in prompt
    assert "100" in prompt
    assert "VARIABLE_NAME" in prompt
    assert "FUNCTION_NAME" in prompt
    assert "Do not merge" in prompt


def test_align_wordpiece_labels_relabels_subwords():
    maps = build_label_maps()

    aligned = align_wordpiece_labels(
        word_ids=[None, 0, 0, 1, None],
        word_labels=["B-PROG_LANG", "O"],
        label2id=maps.label2id,
    )

    assert aligned == [
        -100,
        maps.label2id["B-PROG_LANG"],
        maps.label2id["I-PROG_LANG"],
        maps.label2id["O"],
        -100,
    ]


def test_teacher_config_prefers_gemma_model(tmp_path):
    config = {
        "active_provider": "local_ollama",
        "active_model": "gemma4:31b-cloud",
        "provider_fallback_order": ["local_ollama"],
        "providers": {
            "local_ollama": {
                "base_url": "http://localhost:11434",
                "api_key": "",
                "is_local": True,
                "litellm_prefix": "ollama_chat",
                "models": {
                    "gemma4:31b-cloud": {"notes": "Primary teacher model"},
                    "minimax-m2.7:cloud": {"notes": "Fallback"},
                },
                "model_fallback_order": ["gemma4:31b-cloud", "minimax-m2.7:cloud"],
            }
        },
    }

    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    loaded = load_litellm_config(path)
    provider = resolve_teacher_provider(loaded)

    assert provider["pid"] == "local_ollama"
    assert provider["litellm_model"] == "ollama_chat/gemma4:31b-cloud"


def test_finetune_config_uses_same_ollama_key_literal():
    from pathlib import Path

    config = json.loads(
        Path("benchmarks/finetune/tinybert-ner-64labels/config.example.json").read_text()
    )
    assert config["providers"]["local_ollama"]["api_key"] == "REPLACE_WITH_OLLAMA_KEY_OR_LEAVE_EMPTY"
