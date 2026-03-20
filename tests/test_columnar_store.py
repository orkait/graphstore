"""Tests for columnar source-of-truth store operations."""
import time
from graphstore.store import CoreStore


class TestMaterializeSlot:
    def test_materialize_basic_fields(self):
        store = CoreStore()
        slot = store.put_node("n1", "person", {"name": "Alice", "age": 30, "score": 4.5})
        result = store._materialize_slot(slot)
        assert result["id"] == "n1"
        assert result["kind"] == "person"
        assert result["name"] == "Alice"
        assert result["age"] == 30
        assert abs(result["score"] - 4.5) < 1e-10

    def test_materialize_tombstoned_returns_none(self):
        store = CoreStore()
        slot = store.put_node("n1", "person", {"name": "Alice"})
        store.delete_node("n1")
        result = store._materialize_slot(slot)
        assert result is None

    def test_materialize_empty_slot_returns_none(self):
        store = CoreStore()
        result = store._materialize_slot(0)
        assert result is None

    def test_materialize_excludes_reserved_columns(self):
        store = CoreStore()
        slot = store.put_node("n1", "person", {"name": "Alice"})
        store.columns.set_reserved(slot, "__created_at__", 1710000000000)
        result = store._materialize_slot(slot)
        assert "__created_at__" not in result
        assert "name" in result

    def test_materialize_boolean_as_int(self):
        store = CoreStore()
        slot = store.put_node("n1", "test", {"active": True})
        result = store._materialize_slot(slot)
        assert result["active"] == 1  # bools stored as int64
