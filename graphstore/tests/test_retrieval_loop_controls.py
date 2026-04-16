"""Tests for benchmark adapter routing and retrieval tuning config output."""

import pytest
from types import SimpleNamespace
from unittest.mock import Mock

from benchmarks.framework.adapters.graphstore_ import GraphStoreAdapter
from benchmarks.framework.adapter import QueryContext
from benchmarks.framework import ratchet_recall, ratchet_test

try:
    from autoresearch import tune_config
except ImportError:
    tune_config = None


def test_adapter_routes_categories_when_no_explicit_strategy():
    adapter = GraphStoreAdapter(config={})

    assert adapter._resolve_strategy("multi-session") == "full"
    assert adapter._resolve_strategy("temporal-reasoning") == "full"
    assert adapter._resolve_strategy("knowledge-update") == "full"
    assert adapter._resolve_strategy("single-session-user") == "full"
    assert adapter._resolve_strategy("single-session-assistant") == "full"
    assert adapter._resolve_strategy("single-session-preference") == "full"
    assert adapter._resolve_strategy("unknown") == "full"


def test_adapter_respects_explicit_strategy_override():
    adapter = GraphStoreAdapter(config={"retrieval_strategy": "remember_lexical"})
    assert adapter._resolve_strategy("multi-session") == "remember_lexical"
    assert adapter._resolve_strategy("temporal-reasoning") == "remember_lexical"


def test_adapter_passes_temporal_anchor_to_dispatch():
    adapter = GraphStoreAdapter(config={})
    adapter._gs = SimpleNamespace()  # truthy, no internal access needed

    seen = {}

    def fake_dispatch(question: str, category: str, k: int, anchor_ms=None):
        seen["anchor_ms"] = anchor_ms
        return [], []

    adapter._dispatch = fake_dispatch  # type: ignore[method-assign]
    ctx = QueryContext(
        question="What happened on 2023-05-29?",
        category="temporal-reasoning",
        metadata={"question_date": "2023-05-30"},
    )

    adapter.query_with_context(ctx, k=5)
    assert seen["anchor_ms"] is not None


def test_adapter_ingest_done_runs_consolidation_when_enabled():
    adapter = GraphStoreAdapter(config={"enable_consolidation": True})
    execute = Mock()
    adapter._gs = SimpleNamespace(execute=execute)

    adapter.ingest_done()
    execute.assert_called_once_with("SYS CONSOLIDATE")


def test_adapter_ingest_done_skips_consolidation_by_default():
    adapter = GraphStoreAdapter(config={})
    execute = Mock()
    adapter._gs = SimpleNamespace(execute=execute)

    adapter.ingest_done()
    execute.assert_not_called()


def test_ratchet_defaults_are_locked_to_jina_small():
    assert ratchet_test.BASE_CONFIG["embedder_model"] == "jina-v5-small-retrieval"
    assert "jina-v5-small-retrieval" in ratchet_recall.run.__code__.co_consts


@pytest.mark.skipif(tune_config is None, reason="optuna not installed")
def test_build_output_config_keeps_extended_retrieval_keys():
    params = {
        "retrieval_depth": 8,
        "recall_depth": 4,
        "max_query_entities": 1,
        "recency_boost_k": 3,
        "recall_decay": 0.43,
        "recency_half_life_days": 51.7,
        "similar_to_oversample": 8,
        "lexical_search_oversample": 4,
        "retrieval_strategy": "remember_graph",
        "fusion_method": "rrf",
        "rrf_k": 40.0,
        "recency_mode": "multiplicative",
        "nucleus_expansion": False,
        "nucleus_hops": 2,
        "nucleus_max_neighbors": 4,
        "search_oversample": 20,
    }

    config = tune_config.build_output_config(params)

    assert config["dsl"]["retrieval_strategy"] == "remember_graph"
    assert config["dsl"]["fusion_method"] == "rrf"
    assert config["dsl"]["rrf_k"] == 40.0
    assert config["dsl"]["recency_mode"] == "multiplicative"
    assert config["dsl"]["nucleus_expansion"] is False
    assert config["dsl"]["nucleus_hops"] == 2
    assert config["dsl"]["nucleus_max_neighbors"] == 4
    assert config["vector"]["search_oversample"] == 20
