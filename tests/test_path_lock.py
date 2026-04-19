"""Cross-process single-owner lock for persistent GraphStore paths."""
import pytest

from graphstore import GraphStore, StoreInUse


def test_second_open_on_same_path_raises(tmp_path):
    a = GraphStore(path=str(tmp_path))
    try:
        with pytest.raises(StoreInUse):
            GraphStore(path=str(tmp_path))
    finally:
        a.close()


def test_reopen_after_close_works(tmp_path):
    a = GraphStore(path=str(tmp_path))
    a.close()
    b = GraphStore(path=str(tmp_path))
    b.close()


def test_in_memory_stores_do_not_lock():
    # Two in-memory stores can coexist - no shared state.
    a = GraphStore()
    b = GraphStore()
    a.close()
    b.close()


def test_lock_file_created_under_path(tmp_path):
    gs = GraphStore(path=str(tmp_path))
    try:
        lock = tmp_path / ".graphstore.lock"
        assert lock.exists()
    finally:
        gs.close()
