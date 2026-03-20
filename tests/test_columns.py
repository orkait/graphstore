"""Unit tests for graphstore.columns.ColumnStore."""

import numpy as np
import pytest

from graphstore.columns import ColumnStore
from graphstore.strings import StringTable


class TestSetAndPresence:
    def test_set_int_creates_column(self):
        cs = ColumnStore(StringTable(), capacity=8)
        cs.set(0, {"score": 42})
        assert cs.has_column("score")
        pres = cs.get_presence("score", 1)
        assert pres is not None and pres[0]

    def test_set_float_creates_column(self):
        cs = ColumnStore(StringTable(), capacity=8)
        cs.set(0, {"weight": 3.14})
        assert cs.has_column("weight")

    def test_set_string_creates_interned_column(self):
        st = StringTable()
        cs = ColumnStore(st, capacity=8)
        cs.set(0, {"name": "alice"})
        assert cs.has_column("name")
        pres = cs.get_presence("name", 1)
        assert pres is not None and pres[0]

    def test_set_bool_stored_as_int64(self):
        cs = ColumnStore(StringTable(), capacity=8)
        cs.set(0, {"active": True})
        assert cs.has_column("active")

    def test_set_multiple_fields(self):
        cs = ColumnStore(StringTable(), capacity=8)
        cs.set(0, {"score": 42, "name": "alice", "weight": 1.5})
        assert cs.has_column("score")
        assert cs.has_column("name")
        assert cs.has_column("weight")

    def test_has_column_false_for_missing(self):
        cs = ColumnStore(StringTable(), capacity=8)
        assert not cs.has_column("nonexistent")

    def test_get_presence_none_for_missing_column(self):
        cs = ColumnStore(StringTable(), capacity=8)
        assert cs.get_presence("missing", 1) is None


class TestClear:
    def test_clear_removes_presence(self):
        cs = ColumnStore(StringTable(), capacity=8)
        cs.set(0, {"score": 42})
        cs.clear(0)
        pres = cs.get_presence("score", 1)
        assert pres is not None and not pres[0]

    def test_clear_all_fields(self):
        cs = ColumnStore(StringTable(), capacity=8)
        cs.set(0, {"score": 42, "name": "alice"})
        cs.clear(0)
        assert not cs.get_presence("score", 1)[0]
        assert not cs.get_presence("name", 1)[0]


class TestGrow:
    def test_grow_preserves_data(self):
        cs = ColumnStore(StringTable(), capacity=4)
        cs.set(0, {"score": 42})
        cs.grow(8)
        mask = cs.get_mask("score", "=", 42, 1)
        assert mask is not None and mask[0]

    def test_grow_extends_arrays(self):
        cs = ColumnStore(StringTable(), capacity=4)
        cs.set(0, {"score": 42})
        cs.grow(8)
        cs.set(5, {"score": 99})
        pres = cs.get_presence("score", 6)
        assert pres[5]


class TestTypeMismatch:
    def test_mismatch_stores_sentinel_not_present(self):
        cs = ColumnStore(StringTable(), capacity=8)
        cs.set(0, {"score": 42})
        cs.set(1, {"score": "oops"})
        pres = cs.get_presence("score", 2)
        assert pres[0] and not pres[1]

    def test_float_into_int_column_is_mismatch(self):
        cs = ColumnStore(StringTable(), capacity=8)
        cs.set(0, {"score": 42})
        cs.set(1, {"score": 3.14})
        pres = cs.get_presence("score", 2)
        assert pres[0] and not pres[1]

    def test_int_into_float_column_is_ok(self):
        cs = ColumnStore(StringTable(), capacity=8)
        cs.set(0, {"weight": 3.14})
        cs.set(1, {"weight": 5})
        pres = cs.get_presence("weight", 2)
        assert pres[0] and pres[1]


class TestGetMask:
    def test_int_equals(self):
        cs = ColumnStore(StringTable(), capacity=8)
        cs.set(0, {"score": 10})
        cs.set(1, {"score": 20})
        cs.set(2, {"score": 10})
        mask = cs.get_mask("score", "=", 10, 3)
        assert list(mask) == [True, False, True]

    def test_int_greater_than(self):
        cs = ColumnStore(StringTable(), capacity=8)
        cs.set(0, {"score": 10})
        cs.set(1, {"score": 20})
        cs.set(2, {"score": 30})
        mask = cs.get_mask("score", ">", 15, 3)
        assert list(mask) == [False, True, True]

    def test_int_less_than(self):
        cs = ColumnStore(StringTable(), capacity=8)
        cs.set(0, {"score": 10})
        cs.set(1, {"score": 20})
        mask = cs.get_mask("score", "<", 15, 2)
        assert list(mask) == [True, False]

    def test_int_not_equals(self):
        cs = ColumnStore(StringTable(), capacity=8)
        cs.set(0, {"score": 10})
        cs.set(1, {"score": 20})
        mask = cs.get_mask("score", "!=", 10, 2)
        assert list(mask) == [False, True]

    def test_int_gte(self):
        cs = ColumnStore(StringTable(), capacity=8)
        cs.set(0, {"score": 10})
        cs.set(1, {"score": 20})
        mask = cs.get_mask("score", ">=", 20, 2)
        assert list(mask) == [False, True]

    def test_int_lte(self):
        cs = ColumnStore(StringTable(), capacity=8)
        cs.set(0, {"score": 10})
        cs.set(1, {"score": 20})
        mask = cs.get_mask("score", "<=", 10, 2)
        assert list(mask) == [True, False]

    def test_float_greater_than(self):
        cs = ColumnStore(StringTable(), capacity=8)
        cs.set(0, {"weight": 1.5})
        cs.set(1, {"weight": 3.0})
        mask = cs.get_mask("weight", ">", 2.0, 2)
        assert list(mask) == [False, True]

    def test_string_equals(self):
        st = StringTable()
        cs = ColumnStore(st, capacity=8)
        cs.set(0, {"name": "alice"})
        cs.set(1, {"name": "bob"})
        mask = cs.get_mask("name", "=", "alice", 2)
        assert list(mask) == [True, False]

    def test_string_not_equals(self):
        st = StringTable()
        cs = ColumnStore(st, capacity=8)
        cs.set(0, {"name": "alice"})
        cs.set(1, {"name": "bob"})
        mask = cs.get_mask("name", "!=", "alice", 2)
        assert list(mask) == [False, True]

    def test_string_equals_not_interned_returns_zeros(self):
        st = StringTable()
        cs = ColumnStore(st, capacity=8)
        cs.set(0, {"name": "alice"})
        mask = cs.get_mask("name", "=", "unknown", 1)
        assert list(mask) == [False]

    def test_string_gt_returns_none(self):
        st = StringTable()
        cs = ColumnStore(st, capacity=8)
        cs.set(0, {"name": "alice"})
        assert cs.get_mask("name", ">", "a", 1) is None

    def test_missing_column_returns_none(self):
        cs = ColumnStore(StringTable(), capacity=8)
        assert cs.get_mask("missing", "=", 1, 1) is None

    def test_null_equals(self):
        cs = ColumnStore(StringTable(), capacity=8)
        cs.set(0, {"score": 42})
        mask = cs.get_mask("score", "=", None, 2)
        assert list(mask) == [False, True]

    def test_null_not_equals(self):
        cs = ColumnStore(StringTable(), capacity=8)
        cs.set(0, {"score": 42})
        mask = cs.get_mask("score", "!=", None, 2)
        assert list(mask) == [True, False]

    def test_bool_equals_true(self):
        cs = ColumnStore(StringTable(), capacity=8)
        cs.set(0, {"active": True})
        cs.set(1, {"active": False})
        mask = cs.get_mask("active", "=", True, 2)
        assert list(mask) == [True, False]

    def test_absent_slots_excluded(self):
        cs = ColumnStore(StringTable(), capacity=8)
        cs.set(0, {"score": 10})
        cs.set(2, {"score": 10})
        mask = cs.get_mask("score", "=", 10, 3)
        assert list(mask) == [True, False, True]


class TestGetMaskIn:
    def test_int_in(self):
        cs = ColumnStore(StringTable(), capacity=8)
        cs.set(0, {"score": 10})
        cs.set(1, {"score": 20})
        cs.set(2, {"score": 30})
        mask = cs.get_mask_in("score", [10, 30], 3)
        assert list(mask) == [True, False, True]

    def test_string_in(self):
        st = StringTable()
        cs = ColumnStore(st, capacity=8)
        cs.set(0, {"name": "alice"})
        cs.set(1, {"name": "bob"})
        cs.set(2, {"name": "carol"})
        mask = cs.get_mask_in("name", ["alice", "carol"], 3)
        assert list(mask) == [True, False, True]

    def test_string_in_with_unknown_value(self):
        st = StringTable()
        cs = ColumnStore(st, capacity=8)
        cs.set(0, {"name": "alice"})
        mask = cs.get_mask_in("name", ["unknown"], 1)
        assert list(mask) == [False]

    def test_missing_column_returns_none(self):
        cs = ColumnStore(StringTable(), capacity=8)
        assert cs.get_mask_in("missing", [1], 1) is None


class TestRebuildFrom:
    def test_rebuild_reconstructs_columns(self):
        cs = ColumnStore(StringTable(), capacity=8)
        node_data = [
            {"score": 42, "name": "alice"},
            {"score": 99, "name": "bob"},
            None,
        ]
        cs.rebuild_from(node_data, 3)
        assert cs.has_column("score")
        assert cs.has_column("name")
        mask = cs.get_mask("score", "=", 42, 3)
        assert list(mask) == [True, False, False]

    def test_rebuild_clears_old_columns(self):
        cs = ColumnStore(StringTable(), capacity=8)
        cs.set(0, {"old_field": 1})
        assert cs.has_column("old_field")
        cs.rebuild_from([{"new_field": 2}], 1)
        assert not cs.has_column("old_field")
        assert cs.has_column("new_field")


class TestDeclareColumn:
    def test_declare_creates_empty_column(self):
        cs = ColumnStore(StringTable(), capacity=8)
        cs.declare_column("score", "int64")
        assert cs.has_column("score")
        pres = cs.get_presence("score", 1)
        assert not pres[0]

    def test_declare_does_not_overwrite_existing(self):
        cs = ColumnStore(StringTable(), capacity=8)
        cs.set(0, {"score": 42})
        cs.declare_column("score", "int64")
        mask = cs.get_mask("score", "=", 42, 1)
        assert mask[0]


class TestGetColumn:
    def test_get_column_returns_data_and_presence(self):
        cs = ColumnStore(StringTable(), capacity=8)
        cs.set(0, {"score": 42})
        cs.set(1, {"score": 99})
        result = cs.get_column("score", 2)
        assert result is not None
        col, pres, dtype_str = result
        assert col[0] == 42
        assert col[1] == 99
        assert pres[0] and pres[1]
        assert dtype_str == "int64"

    def test_get_column_missing_returns_none(self):
        cs = ColumnStore(StringTable(), capacity=8)
        assert cs.get_column("missing", 1) is None


class TestMemoryBytes:
    def test_empty_columns_zero_bytes(self):
        cs = ColumnStore(StringTable(), capacity=8)
        assert cs.memory_bytes == 0

    def test_columns_report_memory(self):
        cs = ColumnStore(StringTable(), capacity=8)
        cs.set(0, {"score": 42})
        assert cs.memory_bytes > 0


# -- CoreStore integration tests --

from graphstore.store import CoreStore


class TestCoreStoreColumnIntegration:
    def test_put_node_populates_columns(self):
        s = CoreStore()
        s.put_node("n1", "thing", {"score": 42, "name": "alice"})
        assert s.columns.has_column("score")
        assert s.columns.has_column("name")
        mask = s.columns.get_mask("score", "=", 42, s._next_slot)
        assert mask[0]

    def test_update_node_updates_columns(self):
        s = CoreStore()
        s.put_node("n1", "thing", {"score": 42})
        s.update_node("n1", {"score": 99})
        mask = s.columns.get_mask("score", "=", 99, s._next_slot)
        assert mask[0]

    def test_upsert_node_updates_columns(self):
        s = CoreStore()
        s.put_node("n1", "thing", {"score": 42})
        s.upsert_node("n1", "thing", {"score": 99})
        mask = s.columns.get_mask("score", "=", 99, s._next_slot)
        assert mask[0]

    def test_delete_node_clears_columns(self):
        s = CoreStore()
        s.put_node("n1", "thing", {"score": 42})
        s.delete_node("n1")
        pres = s.columns.get_presence("score", s._next_slot)
        assert not pres[0]

    def test_increment_field_updates_column(self):
        s = CoreStore()
        s.put_node("n1", "thing", {"hits": 0})
        s.increment_field("n1", "hits", 5)
        mask = s.columns.get_mask("hits", "=", 5, s._next_slot)
        assert mask[0]

    def test_grow_extends_columns(self):
        s = CoreStore()
        for i in range(1025):
            s.put_node(f"n{i}", "thing", {"score": i})
        mask = s.columns.get_mask("score", "=", 1024, s._next_slot)
        assert np.any(mask)

    def test_live_mask_basic(self):
        s = CoreStore()
        s.put_node("n1", "fn", {"x": 1})
        s.put_node("n2", "cls", {"x": 2})
        s.put_node("n3", "fn", {"x": 3})
        mask = s._live_mask()
        assert np.count_nonzero(mask) == 3

    def test_live_mask_with_kind(self):
        s = CoreStore()
        s.put_node("n1", "fn", {"x": 1})
        s.put_node("n2", "cls", {"x": 2})
        s.put_node("n3", "fn", {"x": 3})
        mask = s._live_mask("fn")
        assert np.count_nonzero(mask) == 2

    def test_live_mask_excludes_tombstones(self):
        s = CoreStore()
        s.put_node("n1", "fn", {"x": 1})
        s.put_node("n2", "fn", {"x": 2})
        s.delete_node("n1")
        mask = s._live_mask()
        assert np.count_nonzero(mask) == 1
