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


class TestAutoTimestamps:
    def test_put_node_sets_created_at(self):
        store = CoreStore()
        before = int(time.time() * 1000)
        store.put_node("n1", "person", {"name": "Alice"})
        after = int(time.time() * 1000)
        col = store.columns.get_column("__created_at__", 1)
        assert col is not None
        ts = int(col[0][0])
        assert before <= ts <= after

    def test_put_node_sets_updated_at(self):
        store = CoreStore()
        store.put_node("n1", "person", {"name": "Alice"})
        col = store.columns.get_column("__updated_at__", 1)
        assert col is not None
        assert int(col[0][0]) > 0

    def test_update_node_updates_timestamp(self):
        store = CoreStore()
        store.put_node("n1", "person", {"name": "Alice"})
        created = int(store.columns._columns["__created_at__"][0])
        time.sleep(0.01)
        store.update_node("n1", {"name": "Bob"})
        updated = int(store.columns._columns["__updated_at__"][0])
        assert updated >= created

    def test_increment_updates_timestamp(self):
        store = CoreStore()
        store.put_node("n1", "test", {"score": 10})
        time.sleep(0.01)
        store.increment_field("n1", "score", 5)
        updated = int(store.columns._columns["__updated_at__"][0])
        assert updated > 0

    def test_timestamps_not_in_materialized_output(self):
        store = CoreStore()
        store.put_node("n1", "person", {"name": "Alice"})
        result = store._materialize_slot(0)
        assert "__created_at__" not in result
        assert "__updated_at__" not in result
        assert "name" in result
