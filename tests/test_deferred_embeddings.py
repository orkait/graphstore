"""Tests for the deferred_embeddings context manager.

The deferred embedding path is the critical perf fix for transformer embedders
(EmbeddingGemma, Harrier, bge-*, etc.) where per-call inference overhead
dominates. These tests verify both correctness (same vectors as the immediate
path) and that batching actually happens.
"""
import numpy as np
from graphstore import GraphStore
from graphstore.embedding.base import Embedder


class CountingEmbedder(Embedder):
    """Embedder that records each encode_documents call's batch size.

    Useful for verifying that deferred mode actually batches calls rather than
    falling back to one-at-a-time encoding.
    """
    def __init__(self):
        self.calls: list[int] = []  # batch size of each encode_documents call

    @property
    def name(self) -> str:
        return "counting"

    @property
    def dims(self) -> int:
        return 16

    def _hash_encode(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            seed = hash(t) % (2**31)
            rng = np.random.RandomState(seed)
            vec = rng.randn(16).astype(np.float32)
            vec /= np.linalg.norm(vec)
            vecs.append(vec)
        return np.array(vecs, dtype=np.float32)

    def encode_documents(self, texts, titles=None):
        self.calls.append(len(texts))
        return self._hash_encode(texts)

    def encode_queries(self, texts):
        return self._hash_encode(texts)


def test_deferred_mode_batches_create_node():
    """Within deferred_embeddings, N CREATE NODEs should trigger 1 batched embed call,
    not N calls."""
    emb = CountingEmbedder()
    gs = GraphStore(embedder=emb)
    gs.execute('SYS REGISTER NODE KIND "doc" REQUIRED text:string EMBED text')

    with gs.deferred_embeddings(batch_size=64):
        for i in range(10):
            gs.execute(f'CREATE NODE "d{i}" kind = "doc" text = "doc number {i}"')

    # With batch_size=64 and 10 inserts, exactly one flush at context exit.
    assert emb.calls == [10], f"expected one batched call of size 10, got {emb.calls}"
    gs.close()


def test_deferred_mode_auto_flushes_when_batch_size_reached():
    """Deferred mode should auto-flush when the pending queue hits batch_size."""
    emb = CountingEmbedder()
    gs = GraphStore(embedder=emb)
    gs.execute('SYS REGISTER NODE KIND "doc" REQUIRED text:string EMBED text')

    with gs.deferred_embeddings(batch_size=4):
        for i in range(10):
            gs.execute(f'CREATE NODE "d{i}" kind = "doc" text = "doc {i}"')

    # 10 inserts with batch_size=4:
    #   - after 4th insert: auto-flush (call 1, size 4)
    #   - after 8th insert: auto-flush (call 2, size 4)
    #   - context exit: final flush for remaining 2 (call 3, size 2)
    assert emb.calls == [4, 4, 2], f"expected [4, 4, 2], got {emb.calls}"
    gs.close()


def test_deferred_mode_produces_same_vectors_as_immediate():
    """Deferred and immediate modes must produce identical vectors for the same text."""
    emb_deferred = CountingEmbedder()
    gs_d = GraphStore(embedder=emb_deferred)
    gs_d.execute('SYS REGISTER NODE KIND "doc" REQUIRED text:string EMBED text')
    texts = [f"document content {i}" for i in range(5)]
    with gs_d.deferred_embeddings(batch_size=10):
        for i, t in enumerate(texts):
            gs_d.execute(f'CREATE NODE "d{i}" kind = "doc" text = "{t}"')
    deferred_vecs = [
        gs_d._vector_store.get_vector(gs_d._store.id_to_slot[gs_d._store.string_table.intern(f"d{i}")])
        for i in range(5)
    ]
    gs_d.close()

    emb_immediate = CountingEmbedder()
    gs_i = GraphStore(embedder=emb_immediate)
    gs_i.execute('SYS REGISTER NODE KIND "doc" REQUIRED text:string EMBED text')
    for i, t in enumerate(texts):
        gs_i.execute(f'CREATE NODE "d{i}" kind = "doc" text = "{t}"')
    immediate_vecs = [
        gs_i._vector_store.get_vector(gs_i._store.id_to_slot[gs_i._store.string_table.intern(f"d{i}")])
        for i in range(5)
    ]
    gs_i.close()

    for i, (d, m) in enumerate(zip(deferred_vecs, immediate_vecs)):
        assert np.allclose(d, m), f"vector {i} differs between deferred and immediate modes"


def test_deferred_mode_retrieval_returns_correct_nodes():
    """After deferred ingestion, SIMILAR TO queries should return the right nodes."""
    gs = GraphStore(embedder=CountingEmbedder())
    gs.execute('SYS REGISTER NODE KIND "doc" REQUIRED text:string EMBED text')
    with gs.deferred_embeddings(batch_size=8):
        for i in range(6):
            gs.execute(f'CREATE NODE "d{i}" kind = "doc" text = "unique content number {i}"')
    result = gs.execute('SIMILAR TO "unique content number 3" LIMIT 3')
    ids = [n["id"] for n in result.data]
    assert "d3" in ids, f"expected d3 in top-3 after deferred ingest, got {ids}"
    gs.close()


def test_deferred_mode_with_document_clause_no_double_embed():
    """CREATE NODE with both schema EMBED and DOCUMENT clause must embed exactly once.

    This is the double-embedding bug fix: previously, _handle_vector would embed
    via the schema EMBED field, then _create_node would embed q.document again
    and overwrite the vector. Same text → 2x wasted inference per node.
    """
    emb = CountingEmbedder()
    gs = GraphStore(embedder=emb)
    gs.execute('SYS REGISTER NODE KIND "doc" REQUIRED text:string EMBED text')

    # Immediate mode (no deferred context)
    gs.execute('CREATE NODE "n1" kind = "doc" text = "hello world" DOCUMENT "hello world"')

    # Should be exactly one encode_documents call with one text, not two.
    assert emb.calls == [1], (
        f"CREATE NODE with schema EMBED + DOCUMENT should embed once, got calls={emb.calls}"
    )
    gs.close()


def test_deferred_mode_restores_prior_state_on_exception():
    """If an exception occurs inside deferred_embeddings, the defer flag must be reset."""
    gs = GraphStore(embedder=CountingEmbedder())
    gs.execute('SYS REGISTER NODE KIND "doc" REQUIRED text:string EMBED text')

    assert gs._executor._defer_embeddings is False
    try:
        with gs.deferred_embeddings(batch_size=4):
            gs.execute('CREATE NODE "ok1" kind = "doc" text = "fine"')
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    # Defer flag must be reset even after exception
    assert gs._executor._defer_embeddings is False
    # And subsequent non-deferred CREATE NODE should still work
    gs.execute('CREATE NODE "ok2" kind = "doc" text = "after exception"')
    gs.close()
