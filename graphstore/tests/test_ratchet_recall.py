import json
from pathlib import Path

import pytest

_LOCOMO_PATH = Path("/tmp/locomo/raw/locomo10.json")


@pytest.mark.skipif(not _LOCOMO_PATH.exists(), reason="LoCoMo dataset not present")
def test_build_evidence_lookup_maps_observation_ids_to_message_ids():
    from benchmarks.framework.ratchet_recall import build_evidence_lookup

    raw = json.loads(_LOCOMO_PATH.read_text())
    conv = raw[0]
    lookup = build_evidence_lookup(conv)

    assert lookup["D1:3"] == "s1:msg0"
    assert lookup["D1:7"] == "s1:msg1"
    assert lookup["D1:9"] == "s1:msg2"


def test_score_evidence_support_reports_strict_and_pragmatic_hits():
    from benchmarks.framework.ratchet_recall import score_evidence_support

    evidence_lookup = {
        "D1:3": "s1:msg0",
        "D1:7": "s1:msg1",
        "D2:1": "s2:msg0",
    }
    retrieved_ids = ["s1:msg1", "s3:msg0"]
    scores = score_evidence_support(
        evidence_ids=["D1:3", "D1:7", "D2:1"],
        evidence_lookup=evidence_lookup,
        retrieved_ids=retrieved_ids,
    )

    assert scores["strict_hit"] is True
    assert scores["strict_coverage"] == 1 / 3
    assert scores["pragmatic_hit"] is True
    assert scores["pragmatic_coverage"] == 1 / 2


def test_score_evidence_support_ignores_unknown_evidence_ids():
    from benchmarks.framework.ratchet_recall import score_evidence_support

    scores = score_evidence_support(
        evidence_ids=["UNKNOWN"],
        evidence_lookup={},
        retrieved_ids=["s1:msg0"],
    )

    assert scores["strict_hit"] is False
    assert scores["strict_coverage"] == 0.0
    assert scores["pragmatic_hit"] is False
    assert scores["pragmatic_coverage"] == 0.0


def test_build_evidence_lookups_by_sample_id():
    from benchmarks.framework.ratchet_recall import build_evidence_lookups

    raw = [
        {
            "sample_id": "conv-a",
            "observation": {
                "session_1_observation": {
                    "Alice": [["hello", "D1:1"]],
                }
            },
        },
        {
            "sample_id": "conv-b",
            "observation": {
                "session_1_observation": {
                    "Bob": [["world", "D1:1"]],
                }
            },
        },
    ]
    lookups = build_evidence_lookups(raw)
    assert lookups["conv-a"]["D1:1"] == "s1:msg0"
    assert lookups["conv-b"]["D1:1"] == "s1:msg0"
