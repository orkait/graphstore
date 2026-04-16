import pytest
from graphstore import GraphStore


def test_batch_vector_rolled_back_on_failure():
    gs = GraphStore(embedder=None)
    try:
        gs.execute('CREATE NODE "seed" kind = "doc" text = "hello"')
        assert gs._vector_store is None or gs._vector_store.count() == 0

        with pytest.raises(Exception):
            gs.execute(
                'BEGIN\n'
                'CREATE NODE "x" kind = "doc" text = "a" VECTOR [0.1, 0.2, 0.3]\n'
                'CREATE NODE "x" kind = "doc" text = "b"\n'
                'COMMIT'
            )

        assert gs.execute('NODE "x"').data is None
        vs = gs._vector_store
        assert vs is None or vs.count() == 0
    finally:
        gs.close()


def test_auto_id_width_is_16_hex():
    gs = GraphStore()
    try:
        r = gs.execute('CREATE NODE AUTO kind = "k" v = 1')
        nid = r.data["id"]
        assert len(nid) == 16, f"expected 16 hex chars, got {len(nid)}: {nid!r}"
        assert all(c in "0123456789abcdef" for c in nid)
    finally:
        gs.close()
