import pytest
from graphstore import GraphStore
from graphstore.core.errors import GraphStoreError


def test_merge_self_is_rejected():
    gs = GraphStore()
    try:
        gs.execute('CREATE NODE "a" kind = "person" name = "alice"')
        gs.execute('CREATE NODE "b" kind = "person" name = "bob"')
        gs.execute('CREATE EDGE "a" -> "b" kind = "knows"')
        with pytest.raises(GraphStoreError, match="same slot"):
            gs.execute('MERGE NODE "a" INTO "a"')
        assert gs.execute('NODE "a"').data is not None
        assert gs.execute('NODE "b"').data is not None
        assert gs.execute('COUNT EDGES WHERE kind = "knows"').data == 1
    finally:
        gs.close()


def test_merge_different_nodes_still_works():
    gs = GraphStore()
    try:
        gs.execute('CREATE NODE "a" kind = "person" name = "alice"')
        gs.execute('CREATE NODE "b" kind = "person" name = "bob"')
        gs.execute('MERGE NODE "a" INTO "b"')
        assert gs.execute('NODE "a"').data is None
        assert gs.execute('NODE "b"').data is not None
    finally:
        gs.close()
