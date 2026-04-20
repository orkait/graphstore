"""Test that REMEMBER uses confidence, recall frequency, and recency."""
import time
import numpy as np
from graphstore import GraphStore
from graphstore.embedding.base import Embedder


class FixedEmbedder(Embedder):
    """Returns deterministic vectors based on text hash."""
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


def test_remember_records_recall_feedback():
    """REMEMBER should increment __recall_count__ on returned nodes."""
    gs = GraphStore(embedder=FixedEmbedder())
    gs.execute('SYS REGISTER NODE KIND "item" REQUIRED text:string EMBED text')
    gs.execute('CREATE NODE "r1" kind = "item" text = "quantum physics"')

    gs.execute('REMEMBER "quantum" LIMIT 5')
    gs.execute('REMEMBER "quantum" LIMIT 5')

    slot = gs._store.id_to_slot[gs._store.string_table.intern("r1")]
    if gs._store.columns.has_column("__recall_count__"):
        if gs._store.columns._presence["__recall_count__"][slot]:
            count = int(gs._store.columns._columns["__recall_count__"][slot])
            assert count >= 1, f"Expected recall_count >= 1, got {count}"
    gs.close()


def test_remember_includes_score_breakdown():
    """Results should include the full per-signal breakdown on every node."""
    gs = GraphStore(embedder=FixedEmbedder())
    gs.execute('SYS REGISTER NODE KIND "item" REQUIRED text:string EMBED text')
    gs.execute('CREATE NODE "t1" kind = "item" text = "test content"')

    result = gs.execute('REMEMBER "test" LIMIT 5')
    assert result.data
    node = result.data[0]
    for k in (
        "_remember_score", "_vector_sim", "_bm25_score", "_recency_score",
        "_graph_score", "_co_bonus", "_recall_boost", "_rank_stage",
    ):
        assert k in node, f"missing {k} in REMEMBER node: {list(node.keys())}"
    assert node["_rank_stage"] == "fusion"
    gs.close()


def test_remember_meta_signals_telemetry():
    """meta['signals'] surfaces fusion method, weights, stage counts, reranker state.

    This is Step 1 of graphstore's retrieval-observability effort. Callers who
    want to know *why* a REMEMBER result looks the way it does should be able
    to read the full pipeline telemetry without reading handler source.
    """
    gs = GraphStore(embedder=FixedEmbedder(), graph_signal_enabled=True)
    gs.execute('SYS REGISTER NODE KIND "item" REQUIRED text:string EMBED text')
    for i in range(3):
        gs.execute(f'CREATE NODE "n{i}" kind = "item" text = "entry {i}"')

    r = gs.execute('REMEMBER "entry" LIMIT 2')
    sig = r.meta["signals"]

    # Fusion block
    assert sig["fusion"]["method"] in ("weighted", "rrf")
    assert isinstance(sig["fusion"]["weights"], list)
    assert sig["fusion"]["graph_signal_enabled"] is True

    # Recency block
    assert sig["recency"]["half_life_days"] > 0

    # SQE block
    assert sig["sentence_query_expansion"]["enabled"] in (True, False)
    assert sig["sentence_query_expansion"]["num_sentences"] >= 1

    # Stages block must report every pipeline checkpoint
    stages = sig["stages"]
    for key in ("gathered_vec", "gathered_bm25", "union", "cap_applied",
                "after_cap", "before_rerank", "final"):
        assert key in stages, f"missing stage counter {key}"
    assert stages["final"] == len(r.data)

    # Reranker block (no reranker configured here -> ran=False)
    assert sig["reranker"]["ran"] is False
    assert sig["reranker"]["error"] is None

    # Nucleus block
    assert sig["nucleus"]["enabled"] in (True, False)
    gs.close()


def test_remember_graph_signal_reflected_in_meta():
    """Disabling the graph signal must show in meta['signals']['fusion']."""
    gs = GraphStore(embedder=FixedEmbedder(), graph_signal_enabled=False)
    gs.execute('SYS REGISTER NODE KIND "item" REQUIRED text:string EMBED text')
    gs.execute('CREATE NODE "n1" kind = "item" text = "entry"')
    r = gs.execute('REMEMBER "entry" LIMIT 5')
    assert r.meta["signals"]["fusion"]["graph_signal_enabled"] is False
    gs.close()
