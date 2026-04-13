import numpy as np
import pytest


def test_regex_extractor_strips_prefix_and_skips_date_junk():
    from benchmarks.framework.entity_extraction import RegexEntityExtractor

    ex = RegexEntityExtractor()
    text = "[1:56 pm on 8 May, 2023] Caroline: Caroline met Melanie on Saturday in Sweden."
    ents = ex.extract(text)
    assert "Caroline" in ents
    assert "Melanie" in ents
    assert "Sweden" in ents
    assert "May" not in ents
    assert "Saturday" not in ents


def test_build_entity_extractor_defaults_to_regex():
    from benchmarks.framework.entity_extraction import RegexEntityExtractor, build_entity_extractor

    ex = build_entity_extractor({})
    assert isinstance(ex, RegexEntityExtractor)


def test_build_entity_extractor_tinybert_requires_model_dir():
    from benchmarks.framework.entity_extraction import build_entity_extractor

    with pytest.raises(ValueError, match="entity_model_dir"):
        build_entity_extractor({"entity_extractor": "tinybert_onnx"})


def test_onnx_extractor_decodes_bio_spans():
    from benchmarks.framework.entity_extraction import OnnxTokenClassificationEntityExtractor

    ex = object.__new__(OnnxTokenClassificationEntityExtractor)
    ex._score_threshold = 0.0
    ex._allowed_labels = {"PER", "LOC"}
    ex._blocklist = frozenset({"may", "saturday", "the"})

    text = "Caroline met Melanie in Sweden."
    offsets = [(0, 8), (9, 12), (13, 20), (21, 23), (24, 30), (30, 31)]
    labels = ["B-PER", "O", "B-PER", "O", "B-LOC", "O"]
    scores = np.array([0.99, 0.0, 0.98, 0.0, 0.97, 0.0], dtype=np.float32)

    ents = ex._decode_entities(text, offsets, labels, scores)
    assert ents == ["Caroline", "Melanie", "Sweden"]


def test_onnx_extractor_maps_generic_label_ids():
    from benchmarks.framework.entity_extraction import OnnxTokenClassificationEntityExtractor

    ex = object.__new__(OnnxTokenClassificationEntityExtractor)
    ex._score_threshold = 0.0
    ex._allowed_labels = {"PER", "LOC"}
    ex._blocklist = frozenset({"may", "saturday", "the"})
    ex._label_aliases = {
        "LABEL_1": "B-PER",
        "LABEL_2": "I-PER",
        "LABEL_5": "B-LOC",
    }

    text = "Barack Obama in Paris"
    offsets = [(0, 6), (7, 12), (13, 15), (16, 21)]
    labels = ["LABEL_1", "LABEL_2", "O", "LABEL_5"]
    scores = np.array([0.99, 0.98, 0.0, 0.97], dtype=np.float32)

    ents = ex._decode_entities(text, offsets, labels, scores)
    assert ents == ["Barack Obama", "Paris"]
