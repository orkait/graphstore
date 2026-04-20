"""Unit tests for bonsai_ingestor correctness guards.

The live LLM path needs the 1.09 GB TQ1_0 GGUF on disk and is skipped by
default. These tests exercise the post-processing and guard logic without
touching llama.cpp.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from graphstore.bonsai_ingestor import (
    BonsaiIngestor,
    IngestEmpty,
    IngestOverflow,
    IngestResult,
    _dedupe_upserts,
    _split_lines,
    _strip_think,
)


# --------------------------------------------------------------------
# Post-processing helpers
# --------------------------------------------------------------------

def test_strip_think_removes_single_block():
    out = _strip_think("<think>reasoning</think>CREATE NODE \"x\" kind = \"k\"")
    assert "think" not in out.lower()
    assert "CREATE NODE" in out


def test_strip_think_removes_multiple_blocks():
    inp = "<think>a</think>line1<think>b</think>line2"
    assert _strip_think(inp) == "line1line2"


def test_strip_think_empty_on_only_think():
    assert _strip_think("<think>foo</think>") == ""


def test_split_lines_drops_fences_and_blanks():
    inp = "```dsl\nCREATE NODE \"a\" kind = \"k\"\n\n```\nCREATE NODE \"b\" kind = \"k\""
    lines = _split_lines(inp)
    assert lines == [
        'CREATE NODE "a" kind = "k"',
        'CREATE NODE "b" kind = "k"',
    ]


def test_dedupe_upserts_keeps_first_drops_later():
    stmts = [
        'UPSERT NODE "ent:priya" kind = "entity" name = "Priya"',
        'UPSERT NODE "ent:openai" kind = "entity" name = "OpenAI"',
        'UPSERT NODE "ent:priya" kind = "entity" name = "Priya"',
        'CREATE EDGE "m:0" -> "ent:priya" kind = "mentions"',
    ]
    kept, dropped = _dedupe_upserts(stmts)
    assert len(kept) == 3
    assert 'ent:priya' in kept[0]
    assert 'ent:openai' in kept[1]
    assert 'CREATE EDGE' in kept[2]
    assert len(dropped) == 1
    assert "duplicate" in dropped[0][1]


def test_dedupe_upserts_passes_non_upsert():
    stmts = ['CREATE NODE "m:0" kind = "message"', 'CREATE EDGE "a" -> "b" kind = "k"']
    kept, dropped = _dedupe_upserts(stmts)
    assert kept == stmts
    assert dropped == []


# --------------------------------------------------------------------
# Skill fingerprint
# --------------------------------------------------------------------

def test_skill_fingerprint_is_stable_across_instances(tmp_path: Path):
    skill = tmp_path / "skill.md"
    skill.write_text("content A")
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"ignored; no llama init yet")

    ing1 = BonsaiIngestor(model_path=model, skill_path=skill)
    ing2 = BonsaiIngestor(model_path=model, skill_path=skill)
    assert ing1.skill_fingerprint == ing2.skill_fingerprint

    expected = hashlib.sha256(b"content A").hexdigest()[:12]
    assert ing1.skill_fingerprint == expected


def test_skill_fingerprint_changes_on_skill_edit(tmp_path: Path):
    skill = tmp_path / "skill.md"
    skill.write_text("v1")
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"")

    ing = BonsaiIngestor(model_path=model, skill_path=skill)
    fp_v1 = ing.skill_fingerprint

    skill.write_text("v2")
    ing._reload_skill()
    assert ing.skill_fingerprint != fp_v1


def test_skill_fingerprint_pinned_into_system_prompt(tmp_path: Path):
    skill = tmp_path / "skill.md"
    skill.write_text("rule body")
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"")

    ing = BonsaiIngestor(model_path=model, skill_path=skill)
    assert f"skill-sha256={ing.skill_fingerprint}" in ing._system_prompt
    assert "rule body" in ing._system_prompt


# --------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------

def test_empty_input_raises_ingest_empty(tmp_path: Path):
    skill = tmp_path / "skill.md"
    skill.write_text("any")
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"")

    ing = BonsaiIngestor(model_path=model, skill_path=skill)
    with pytest.raises(IngestEmpty):
        ing.ingest("")
    with pytest.raises(IngestEmpty):
        ing.ingest("   \n\t  ")


def test_non_dry_run_without_store_raises(tmp_path: Path):
    skill = tmp_path / "skill.md"
    skill.write_text("any")
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"")

    ing = BonsaiIngestor(model_path=model, skill_path=skill)
    with pytest.raises(ValueError, match="requires a GraphStore"):
        ing.ingest("hello")


def test_missing_model_file_raises(tmp_path: Path):
    skill = tmp_path / "skill.md"
    skill.write_text("any")
    with pytest.raises(FileNotFoundError):
        BonsaiIngestor(model_path=tmp_path / "nope.gguf", skill_path=skill)


def test_missing_skill_file_raises(tmp_path: Path):
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        BonsaiIngestor(model_path=model, skill_path=tmp_path / "nope.md")


# --------------------------------------------------------------------
# Frontmatter strip
# --------------------------------------------------------------------

def test_yaml_frontmatter_stripped_from_skill(tmp_path: Path):
    skill = tmp_path / "skill.md"
    skill.write_text("---\nname: x\n---\n\nactual rules")
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"")

    ing = BonsaiIngestor(model_path=model, skill_path=skill)
    assert "name: x" not in ing._skill_text
    assert "actual rules" in ing._skill_text


# --------------------------------------------------------------------
# IngestResult shape
# --------------------------------------------------------------------

def test_ingest_result_defaults():
    r = IngestResult()
    assert r.statements == []
    assert r.executed == 0
    assert r.rejected == []
    assert r.entities_new == []
    assert r.beliefs_changed == []
    assert r.duration_ms == 0
    assert r.raw_output == ""
    assert r.skill_fingerprint == ""
    assert r.dry_run is False
