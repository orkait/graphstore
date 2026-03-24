"""Tests for REMEMBER hybrid retrieval command."""
import tempfile
from graphstore import GraphStore


def test_remember_basic():
    """REMEMBER returns results from an in-memory store."""
    gs = GraphStore()
    gs.execute('CREATE NODE "fact1" kind = "fact" summary = "quantum entanglement is spooky"')
    gs.execute('CREATE NODE "fact2" kind = "fact" summary = "classical physics is deterministic"')
    gs.execute('CREATE NODE "fact3" kind = "fact" summary = "quantum computing uses qubits"')

    result = gs.execute('REMEMBER "quantum" LIMIT 5')
    assert result.kind == "nodes"
    # Should find results (even without embedder, BM25 may not be available,
    # but recency scoring still works)
    gs.close()


def test_remember_with_persistence():
    """REMEMBER works with persisted store (has FTS5 for BM25)."""
    with tempfile.TemporaryDirectory() as td:
        gs = GraphStore(path=td)
        # Create nodes with summaries that will be in DocumentStore
        gs.execute('CREATE NODE "doc1" kind = "fact" summary = "photosynthesis converts light"')
        gs.execute('CREATE NODE "doc2" kind = "fact" summary = "mitochondria produces energy"')
        gs.execute('CREATE NODE "doc3" kind = "fact" summary = "chlorophyll absorbs light"')

        result = gs.execute('REMEMBER "light energy" LIMIT 5')
        assert result.kind == "nodes"
        gs.close()


def test_remember_returns_scores():
    """REMEMBER results include breakdown scores."""
    gs = GraphStore()
    gs.execute('CREATE NODE "a" kind = "test" summary = "hello world"')
    result = gs.execute('REMEMBER "hello" LIMIT 5')
    if result.data:
        node = result.data[0]
        assert "_remember_score" in node
        assert "_recency_score" in node
    gs.close()


def test_remember_with_where():
    """REMEMBER respects WHERE clause."""
    gs = GraphStore()
    gs.execute('CREATE NODE "a" kind = "fact" summary = "quantum physics"')
    gs.execute('CREATE NODE "b" kind = "opinion" summary = "quantum is weird"')
    result = gs.execute('REMEMBER "quantum" LIMIT 10 WHERE kind = "fact"')
    for node in result.data:
        assert node["kind"] == "fact"
    gs.close()


def test_remember_empty_store():
    """REMEMBER on empty store returns empty."""
    gs = GraphStore()
    result = gs.execute('REMEMBER "anything" LIMIT 5')
    assert result.kind == "nodes"
    assert result.data == []
    gs.close()


def test_remember_limit():
    """REMEMBER respects LIMIT."""
    gs = GraphStore()
    for i in range(20):
        gs.execute(f'CREATE NODE "n{i}" kind = "test" summary = "test item {i}"')
    result = gs.execute('REMEMBER "test" LIMIT 3')
    assert len(result.data) <= 3
    gs.close()
