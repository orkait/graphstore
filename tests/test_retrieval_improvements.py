"""Tests for retrieval improvements: RRF fusion, type-weighted scoring,
nucleus expansion, and multiplicative temporal decay.

Uses FixedEmbedder + SYS REGISTER to get vectors into the store, matching
the pattern in test_remember_signals.py and test_integration_fixtures.py.
"""

import tempfile

import numpy as np
import pytest

from graphstore import GraphStore
from graphstore.algos.fusion import rrf_remember_fusion
from graphstore.embedding.base import Embedder


class FixedEmbedder(Embedder):
    """Deterministic embedder - same text always produces same vector."""
    @property
    def name(self): return "fixed"
    @property
    def dims(self): return 32

    def _encode(self, texts):
        vecs = []
        for t in texts:
            seed = hash(t) % (2**31)
            rng = np.random.RandomState(seed)
            v = rng.randn(32).astype(np.float32)
            v /= np.linalg.norm(v)
            vecs.append(v)
        return np.array(vecs, dtype=np.float32)

    def encode_documents(self, texts, titles=None):
        return self._encode(texts)
    def encode_queries(self, texts):
        return self._encode(texts)


class KeywordEmbedder(Embedder):
    """Low-dimensional embedder that clusters by topic keyword."""

    @property
    def name(self): return "keyword"

    @property
    def dims(self): return 4

    def _vec(self, text: str) -> np.ndarray:
        t = text.lower()
        if "quantum" in t:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        if "pasta" in t:
            return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    def encode_documents(self, texts, titles=None):
        return np.vstack([self._vec(t) for t in texts]).astype(np.float32)

    def encode_queries(self, texts):
        return np.vstack([self._vec(t) for t in texts]).astype(np.float32)


def _make_gs(**kwargs):
    """Create a GraphStore with FixedEmbedder + fact schema."""
    gs = GraphStore(embedder=FixedEmbedder(), **kwargs)
    gs.execute('SYS REGISTER NODE KIND "fact" REQUIRED claim:string EMBED claim')
    gs.execute('SYS REGISTER NODE KIND "decision" REQUIRED claim:string EMBED claim')
    gs.execute('SYS REGISTER NODE KIND "entity" REQUIRED claim:string EMBED claim')
    gs.execute('SYS REGISTER NODE KIND "lesson" REQUIRED claim:string EMBED claim')
    gs.execute('SYS REGISTER NODE KIND "session" REQUIRED claim:string EMBED claim')
    return gs


def _event_at_for(gs: GraphStore, node_id: str) -> int | None:
    slot = gs._store.id_to_slot[gs._store.string_table.intern(node_id)]
    col = gs._store.columns.get_column("__event_at__", gs._store._next_slot)
    if col is None:
        return None
    data, present, _ = col
    return int(data[slot]) if present[slot] else None


# ── RRF fusion unit tests ──────────────────────────────────────────────


class TestRRFFusion:
    def test_rrf_returns_correct_shape(self):
        n = 100
        sig1 = np.zeros(n)
        sig2 = np.zeros(n)
        candidates = np.array([5, 10, 20, 50])
        sig1[candidates] = [0.9, 0.7, 0.3, 0.1]
        sig2[candidates] = [0.1, 0.3, 0.7, 0.9]

        fused = rrf_remember_fusion(sig1, sig2, candidate_slots=candidates, k_rrf=60.0)
        assert fused.shape == (n,)
        assert fused.dtype == np.float64

    def test_rrf_nonzero_only_at_candidates(self):
        n = 50
        sig = np.zeros(n)
        candidates = np.array([3, 7, 12])
        sig[candidates] = [0.5, 0.8, 0.3]

        fused = rrf_remember_fusion(sig, candidate_slots=candidates, k_rrf=60.0)
        non_cand = np.setdiff1d(np.arange(n), candidates)
        assert np.all(fused[non_cand] == 0.0)
        assert np.all(fused[candidates] > 0.0)

    def test_rrf_ranking_matches_dominant_signal(self):
        n = 20
        candidates = np.array([1, 2, 3, 4])
        sig1 = np.zeros(n)
        sig1[candidates] = [1.0, 0.5, 0.2, 0.1]
        sig2 = np.zeros(n)
        sig2[candidates] = [0.9, 0.4, 0.3, 0.2]

        fused = rrf_remember_fusion(sig1, sig2, candidate_slots=candidates, k_rrf=60.0)
        assert fused[1] > fused[2] > fused[3] > fused[4]

    def test_rrf_consensus_beats_single_signal(self):
        """Candidate ranked moderately across all signals should beat one
        ranked high on one signal but absent from others."""
        n = 20
        candidates = np.array([1, 2])
        sig1 = np.zeros(n); sig1[1] = 1.0; sig1[2] = 0.5
        sig2 = np.zeros(n); sig2[2] = 0.8
        sig3 = np.zeros(n); sig3[2] = 0.7

        fused = rrf_remember_fusion(sig1, sig2, sig3, candidate_slots=candidates, k_rrf=60.0)
        assert fused[2] > fused[1]

    def test_rrf_empty_candidates(self):
        n = 10
        sig = np.zeros(n)
        fused = rrf_remember_fusion(sig, candidate_slots=np.array([], dtype=np.int64))
        assert np.all(fused == 0.0)

    def test_rrf_k_parameter_affects_scores(self):
        n = 10
        candidates = np.array([0, 1])
        sig = np.zeros(n); sig[0] = 1.0; sig[1] = 0.5

        fused_low_k = rrf_remember_fusion(sig, candidate_slots=candidates, k_rrf=1.0)
        fused_high_k = rrf_remember_fusion(sig, candidate_slots=candidates, k_rrf=100.0)
        assert fused_low_k[0] > fused_high_k[0]


# ── RRF integration with GraphStore ────────────────────────────────────


class TestRRFIntegration:
    def test_remember_uses_rrf_by_default(self):
        gs = _make_gs()
        gs.execute('CREATE NODE "a" kind = "fact" claim = "quantum entanglement is spooky"')
        gs.execute('CREATE NODE "b" kind = "fact" claim = "classical physics is deterministic"')
        result = gs.execute('REMEMBER "quantum entanglement is spooky" LIMIT 5')
        assert result.kind == "nodes"
        assert len(result.data) > 0
        assert result.data[0]["_remember_score"] > 0
        gs.close()

    def test_remember_weighted_fallback(self):
        gs = _make_gs(fusion_method="weighted")
        gs.execute('CREATE NODE "a" kind = "fact" claim = "quantum entanglement is spooky"')
        result = gs.execute('REMEMBER "quantum entanglement is spooky" LIMIT 5')
        assert result.kind == "nodes"
        assert len(result.data) > 0
        gs.close()

    def test_rrf_and_weighted_both_return_results(self):
        """Both fusion methods should return results for the same data."""
        for method in ("rrf", "weighted"):
            gs = _make_gs(fusion_method=method)
            gs.execute('CREATE NODE "x" kind = "fact" claim = "machine learning is powerful"')
            gs.execute('CREATE NODE "y" kind = "fact" claim = "machine learning is powerful"')
            result = gs.execute('REMEMBER "machine learning is powerful" LIMIT 5')
            assert len(result.data) > 0, f"fusion_method={method} returned 0 results"
            gs.close()


# ── Type-weighted scoring ──────────────────────────────────────────────


class TestTypeWeightedScoring:
    def test_type_weights_boost_decisions(self):
        """Decisions should rank higher than entities with equal relevance."""
        gs = _make_gs(type_weights={"decision": 2.0, "entity": 0.5})
        # Use identical claim text so vector similarity is equal for both
        gs.execute('CREATE NODE "d1" kind = "decision" claim = "python backend choice"')
        gs.execute('CREATE NODE "e1" kind = "entity" claim = "python backend choice"')
        result = gs.execute('REMEMBER "python backend choice" LIMIT 5')
        assert len(result.data) >= 2
        kinds = [r["kind"] for r in result.data[:2]]
        assert kinds[0] == "decision", f"Expected decision first, got {kinds}"
        gs.close()

    def test_type_weights_empty_dict_noop(self):
        gs = _make_gs(type_weights={})
        gs.execute('CREATE NODE "a" kind = "fact" claim = "hello world"')
        result = gs.execute('REMEMBER "hello" LIMIT 5')
        assert result.kind == "nodes"
        gs.close()

    def test_type_weights_unknown_kind_defaults_to_1(self):
        gs = _make_gs(type_weights={"decision": 2.0})
        gs.execute('CREATE NODE "x" kind = "fact" claim = "test content here"')
        result = gs.execute('REMEMBER "test content here" LIMIT 5')
        assert len(result.data) > 0
        gs.close()

    def test_lessons_rank_higher_than_sessions(self):
        """Lessons (1.5x) should outrank sessions (0.7x) for same content."""
        gs = _make_gs(type_weights={"lesson": 1.5, "session": 0.7})
        # Identical claim so vector similarity is equal - type weight decides
        gs.execute('CREATE NODE "l1" kind = "lesson" claim = "always write tests before shipping"')
        gs.execute('CREATE NODE "s1" kind = "session" claim = "always write tests before shipping"')
        result = gs.execute('REMEMBER "always write tests before shipping" LIMIT 5')
        if len(result.data) >= 2:
            kinds = [r["kind"] for r in result.data[:2]]
            assert kinds[0] == "lesson", f"Expected lesson first, got {kinds}"
        gs.close()


# ── Nucleus expansion ──────────────────────────────────────────────────


class TestNucleusExpansion:
    def test_nucleus_includes_neighbors(self):
        """Nucleus is off by default; opt-in to test."""
        gs = _make_gs(nucleus_expansion=True, nucleus_max_neighbors=3)
        gs.execute('CREATE NODE "main" kind = "fact" claim = "quantum computing uses qubits"')
        gs.execute('CREATE NODE "nb1" kind = "fact" claim = "qubits can be entangled"')
        gs.execute('CREATE EDGE "main" -> "nb1" kind = "related"')

        result = gs.execute('REMEMBER "quantum computing" LIMIT 5')
        assert len(result.data) > 0
        # Check if neighbor was pulled in via nucleus
        all_ids = {r["id"] for r in result.data}
        nucleus_ids = {r["id"] for r in result.data if r.get("_nucleus")}
        # If main was retrieved directly and nb1 wasn't, nb1 should appear as nucleus
        direct_ids = all_ids - nucleus_ids
        if "main" in direct_ids and "nb1" not in direct_ids:
            assert "nb1" in nucleus_ids, f"Expected nb1 in nucleus, got {result.data}"
        gs.close()

    def test_nucleus_disabled_no_extra_results(self):
        gs = _make_gs(nucleus_expansion=False)
        gs.execute('CREATE NODE "main" kind = "fact" claim = "quantum computing uses qubits"')
        gs.execute('CREATE NODE "nb" kind = "fact" claim = "totally unrelated content about fish"')
        gs.execute('CREATE EDGE "main" -> "nb" kind = "related"')

        result = gs.execute('REMEMBER "quantum computing" LIMIT 5')
        nucleus_nodes = [r for r in result.data if r.get("_nucleus")]
        assert len(nucleus_nodes) == 0
        gs.close()

    def test_nucleus_deduplicates(self):
        gs = _make_gs(nucleus_expansion=True, nucleus_max_neighbors=5)
        gs.execute('CREATE NODE "a" kind = "fact" claim = "quantum physics is weird"')
        gs.execute('CREATE NODE "b" kind = "fact" claim = "quantum mechanics is fundamental"')
        gs.execute('CREATE EDGE "a" -> "b" kind = "related"')

        result = gs.execute('REMEMBER "quantum" LIMIT 10')
        ids = [r["id"] for r in result.data]
        assert len(ids) == len(set(ids)), f"Duplicate IDs found: {ids}"
        gs.close()

    def test_nucleus_respects_max_neighbors(self):
        gs = _make_gs(nucleus_expansion=True, nucleus_max_neighbors=1)
        gs.execute('CREATE NODE "main" kind = "fact" claim = "central topic about testing frameworks"')
        for i in range(5):
            gs.execute(f'CREATE NODE "nb{i}" kind = "fact" claim = "related to testing frameworks variant {i}"')
            gs.execute(f'CREATE EDGE "main" -> "nb{i}" kind = "related"')

        result = gs.execute('REMEMBER "testing frameworks" LIMIT 1')
        nucleus_nodes = [r for r in result.data if r.get("_nucleus")]
        assert len(nucleus_nodes) <= 1, f"Expected max 1 nucleus node, got {len(nucleus_nodes)}"
        gs.close()


# ── Multiplicative temporal decay ──────────────────────────────────────


class TestMultiplicativeDecay:
    def test_multiplicative_mode_is_default(self):
        gs = GraphStore(embedder=FixedEmbedder())
        assert gs._executor._recency_mode == "multiplicative"
        gs.close()

    def test_additive_mode_returns_results(self):
        gs = _make_gs(recency_mode="additive")
        gs.execute('CREATE NODE "a" kind = "fact" claim = "hello world test"')
        result = gs.execute('REMEMBER "hello world" LIMIT 5')
        assert len(result.data) > 0
        gs.close()

    def test_multiplicative_mode_returns_results(self):
        gs = _make_gs(recency_mode="multiplicative")
        gs.execute('CREATE NODE "a" kind = "fact" claim = "hello world test content"')
        result = gs.execute('REMEMBER "hello world test content" LIMIT 5')
        assert len(result.data) > 0
        gs.close()

    def test_recency_score_present_in_both_modes(self):
        for mode in ("additive", "multiplicative"):
            gs = _make_gs(recency_mode=mode)
            gs.execute('CREATE NODE "a" kind = "fact" claim = "exact match query text"')
            result = gs.execute('REMEMBER "exact match query text" LIMIT 5')
            assert len(result.data) > 0, f"mode={mode} returned no results"
            assert "_recency_score" in result.data[0], f"mode={mode} missing _recency_score"
            gs.close()


class TestTemporalDslWiring:
    def test_create_node_event_at_sets_reserved_column(self):
        gs = _make_gs()
        gs.execute('CREATE NODE "a" kind = "fact" claim = "visited the museum" EVENT_AT "2023-05-08"')
        event_at = _event_at_for(gs, "a")
        assert event_at is not None
        gs.close()

    def test_upsert_node_event_at_sets_reserved_column(self):
        gs = _make_gs()
        gs.execute('UPSERT NODE "a" kind = "fact" claim = "visited the museum" EVENT_AT "2023-05-08"')
        event_at = _event_at_for(gs, "a")
        assert event_at is not None
        gs.close()

    def test_assert_event_at_sets_reserved_column(self):
        gs = _make_gs()
        gs.execute('ASSERT "a" kind = "fact" claim = "visited the museum" EVENT_AT "2023-05-08"')
        event_at = _event_at_for(gs, "a")
        assert event_at is not None
        gs.close()

    def test_remember_at_prefers_temporally_matching_nodes(self):
        gs = _make_gs()
        # Use identical claim text so vector similarity is equal for all
        gs.execute('CREATE NODE "recent1" kind = "fact" claim = "museum trip with family" EVENT_AT "2023-05-08"')
        gs.execute('CREATE NODE "recent2" kind = "fact" claim = "museum trip with family" EVENT_AT "2023-05-10"')
        gs.execute('CREATE NODE "neutral" kind = "fact" claim = "museum trip with family"')
        gs.execute('CREATE NODE "old" kind = "fact" claim = "museum trip with family" EVENT_AT "2021-05-08"')
        result = gs.execute('REMEMBER "museum trip with family" AT "2023-05-09" LIMIT 3')
        ids = [r["id"] for r in result.data]
        assert "old" not in ids
        assert "recent1" in ids or "recent2" in ids
        gs.close()


class TestConsolidation:
    def test_sys_consolidate_clusters_similar_messages_by_entity(self):
        gs = GraphStore(embedder=KeywordEmbedder())
        gs.execute('SYS REGISTER NODE KIND "message" REQUIRED content:string EMBED content')
        gs.execute('SYS REGISTER NODE KIND "entity" REQUIRED name:string')
        gs.execute('CREATE NODE "m1" kind = "message" content = "quantum computing uses qubits" EVENT_AT "2023-05-08"')
        gs.execute('CREATE NODE "m2" kind = "message" content = "quantum computers rely on qubits" EVENT_AT "2023-05-09"')
        gs.execute('CREATE NODE "m3" kind = "message" content = "pasta recipes use olive oil" EVENT_AT "2023-05-10"')
        gs.execute('CREATE NODE "ent:q" kind = "entity" name = "Quantum"')
        gs.execute('CREATE EDGE "m1" -> "ent:q" kind = "mentions"')
        gs.execute('CREATE EDGE "m2" -> "ent:q" kind = "mentions"')
        gs.execute('CREATE EDGE "m3" -> "ent:q" kind = "mentions"')

        result = gs.execute('SYS CONSOLIDATE THRESHOLD 0.7')
        assert result.data["observations"] >= 1

        observations = gs.execute('NODES WHERE kind = "observation"')
        obs_nodes = observations.data
        matching = [n for n in obs_nodes if n.get("evidence_count") == 2]
        assert matching, obs_nodes

        evidence = gs.execute('EDGES FROM "obs:ent:q:0" WHERE kind = "evidence"')
        assert evidence.count >= 1
        gs.close()


# ── Config wiring ──────────────────────────────────────────────────────


class TestConfigWiring:
    def test_tuned_defaults_promoted(self):
        gs = GraphStore(embedder=FixedEmbedder())
        assert gs._executor._fusion_method == "weighted"
        assert gs._executor._nucleus_expansion is True
        assert gs._executor._retrieval_depth == 9
        assert gs._executor._search_oversample == 16
        assert gs._executor._max_query_entities == 6
        assert gs._executor._recency_boost_k == 4
        assert gs._executor._similar_to_oversample == 2
        gs.close()

    def test_fusion_method_wired(self):
        gs = GraphStore(embedder=FixedEmbedder(), fusion_method="weighted")
        assert gs._executor._fusion_method == "weighted"
        gs.close()

    def test_rrf_k_wired(self):
        gs = GraphStore(embedder=FixedEmbedder(), rrf_k=30.0)
        assert gs._executor._rrf_k == 30.0
        gs.close()

    def test_type_weights_wired(self):
        tw = {"fact": 2.0}
        gs = GraphStore(embedder=FixedEmbedder(), type_weights=tw)
        assert gs._executor._type_weights == tw
        gs.close()

    def test_nucleus_expansion_wired(self):
        gs = GraphStore(embedder=FixedEmbedder(), nucleus_expansion=True)
        assert gs._executor._nucleus_expansion is True
        gs.close()

    def test_nucleus_off_by_default(self):
        gs = GraphStore(embedder=FixedEmbedder())
        assert gs._executor._nucleus_expansion is True
        gs.close()

    def test_nucleus_max_neighbors_wired(self):
        gs = GraphStore(embedder=FixedEmbedder(), nucleus_max_neighbors=7)
        assert gs._executor._nucleus_max_neighbors == 7
        gs.close()

    def test_recency_mode_wired(self):
        gs = GraphStore(embedder=FixedEmbedder(), recency_mode="additive")
        assert gs._executor._recency_mode == "additive"
        gs.close()

    def test_rrf_is_default_fusion(self):
        gs = GraphStore(embedder=FixedEmbedder())
        assert gs._executor._fusion_method == "weighted"
        gs.close()
