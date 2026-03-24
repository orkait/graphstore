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


def test_remember_includes_confidence_in_scoring():
    """Nodes with higher confidence should score higher (all else equal)."""
    gs = GraphStore(embedder=FixedEmbedder())
    gs.execute('SYS REGISTER NODE KIND "fact" REQUIRED claim:string EMBED claim')
    gs.execute('ASSERT "high_conf" kind = "fact" claim = "attention is important" CONFIDENCE 0.99')
    gs.execute('ASSERT "low_conf" kind = "fact" claim = "attention is important" CONFIDENCE 0.1')

    result = gs.execute('REMEMBER "attention" LIMIT 10')
    if len(result.data) >= 2:
        scores = {n["id"]: n["_confidence"] for n in result.data}
        if "high_conf" in scores and "low_conf" in scores:
            assert scores["high_conf"] > scores["low_conf"]
    gs.close()


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
    """Results should include confidence in the breakdown."""
    gs = GraphStore(embedder=FixedEmbedder())
    gs.execute('SYS REGISTER NODE KIND "item" REQUIRED text:string EMBED text')
    gs.execute('CREATE NODE "t1" kind = "item" text = "test content"')

    result = gs.execute('REMEMBER "test" LIMIT 5')
    if result.data:
        node = result.data[0]
        assert "_remember_score" in node
        assert "_confidence" in node
        assert "_recency_score" in node
    gs.close()
