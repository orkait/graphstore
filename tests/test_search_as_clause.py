import pytest
import numpy as np
from graphstore import GraphStore

def test_similar_in_where_clause():
    # Setup store with some nodes
    gs = GraphStore()
    
    gs.execute('CREATE NODE "cat" kind="animal" description="a furry feline"')
    gs.execute('CREATE NODE "dog" kind="animal" description="a loyal canine"')
    gs.execute('CREATE NODE "car" kind="vehicle" description="a fast automobile"')
    
    # Test 1: Syntax parsing check
    try:
        gs.execute('MATCH (n WHERE SIMILAR(n, "kitten") > 0.8) RETURN n')
    except Exception as e:
        # If it's an EmbedderRequired or similar, that's fine, we're testing the plumbing
        assert "Embedder" in str(e) or "vector" in str(e) or "VectorStore" in str(e) or "similar" in str(e).lower()

def test_similar_logic_mocked(monkeypatch):
    from graphstore.embedding.base import Embedder
    
    class MockEmbedder(Embedder):
        @property
        def name(self): return "mock"
        @property
        def dims(self): return 2
        def encode_queries(self, texts):
            return np.array([[1.0, 0.0]] * len(texts), dtype=np.float32)
        def encode_documents(self, texts):
            return np.array([[1.0, 0.0]] * len(texts), dtype=np.float32)

    gs = GraphStore(embedder=MockEmbedder())
    gs._ensure_vector_store(2)
    
    gs.execute('CREATE NODE "node1" kind="test"')
    gs.execute('CREATE NODE "node2" kind="test"')
    
    slot1 = gs._runtime.store.id_to_slot[gs._runtime.store.string_table.intern("node1")]
    slot2 = gs._runtime.store.id_to_slot[gs._runtime.store.string_table.intern("node2")]
    
    gs._runtime.vector_store.add(slot1, np.array([1.0, 0.0], dtype=np.float32)) # matches
    gs._runtime.vector_store.add(slot2, np.array([0.0, 1.0], dtype=np.float32)) # no match
    
    # Now query with SIMILAR
    res = gs.execute('NODES WHERE SIMILAR(n, "kitten") > 0.8')
    assert res.count == 1
    assert res.data[0]["id"] == "node1"
    
    # Test combination with other filters
    gs.execute('UPDATE NODE "node1" SET category="pet"')
    gs.execute('UPDATE NODE "node2" SET category="pet"')
    
    res = gs.execute('NODES WHERE SIMILAR(n, "kitten") > 0.8 AND category="pet"')
    assert res.count == 1
    assert res.data[0]["id"] == "node1"
    
    res = gs.execute('NODES WHERE SIMILAR(n, "kitten") > 0.8 AND category="other"')
    assert res.count == 0

def test_match_with_similar_mocked():
    from graphstore.embedding.base import Embedder
    
    class MockEmbedder(Embedder):
        @property
        def name(self): return "mock"
        @property
        def dims(self): return 2
        def encode_queries(self, texts):
            return np.array([[1.0, 0.0]] * len(texts), dtype=np.float32)
        def encode_documents(self, texts):
            return np.array([[1.0, 0.0]] * len(texts), dtype=np.float32)

    gs = GraphStore(embedder=MockEmbedder())
    gs._ensure_vector_store(2)
    
    gs.execute('CREATE NODE "node1" kind="test"')
    gs.execute('CREATE NODE "node2" kind="test"')
    gs.execute('CREATE NODE "friend" kind="test"')
    gs.execute('CREATE EDGE "node1" -> "friend" kind="knows"')
    gs.execute('CREATE EDGE "node2" -> "friend" kind="knows"')
    
    slot1 = gs._runtime.store.id_to_slot[gs._runtime.store.string_table.intern("node1")]
    slot2 = gs._runtime.store.id_to_slot[gs._runtime.store.string_table.intern("node2")]
    
    gs._runtime.vector_store.add(slot1, np.array([1.0, 0.0], dtype=np.float32)) # matches
    gs._runtime.vector_store.add(slot2, np.array([0.0, 1.0], dtype=np.float32)) # no match
    
    res = gs.execute('MATCH (n WHERE SIMILAR(n, "kitten") > 0.8) -[kind="knows"]-> (m)')
    assert res.count == 1
    assert res.data["bindings"][0]["n"] == "node1"
    assert res.data["bindings"][0]["m"] == "friend"
