# Hybrid Columnar Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a numpy-vectorized columnar acceleration layer to node storage for 100-300x faster WHERE/COUNT/ORDER BY queries, without replacing the existing dict-based storage.

**Architecture:** Hybrid dual-write - `node_data: list[dict]` remains source of truth. A new `ColumnStore` maintains typed numpy arrays as read-acceleration layer. CoreStore dual-writes on every mutation. Executor checks columns first, falls back to dict predicate for non-columnarized fields.

**Tech Stack:** numpy (already available via scipy dependency), existing StringTable for string interning

**Spec:** `docs/superpowers/specs/2026-03-20-columnar-storage-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `graphstore/columns.py` | **Create** | ColumnStore class - typed numpy arrays indexed by slot |
| `graphstore/store.py` | Modify | Integrate ColumnStore into write/delete/grow paths, add `_live_mask` |
| `graphstore/dsl/executor.py` | Modify | Column filter path in `_nodes`, `_count`, `_delete_nodes` |
| `graphstore/dsl/executor_system.py` | Modify | Column rebuild in `SYS REBUILD INDICES`, typed field pre-creation |
| `graphstore/dsl/grammar.lark` | Modify | `typed_ident` rule for `name:string` syntax |
| `graphstore/dsl/ast_nodes.py` | Modify | `SysRegisterNodeKind` typed fields |
| `graphstore/dsl/transformer.py` | Modify | Parse typed idents |
| `graphstore/schema.py` | Modify | Type annotations on field declarations, type validation |
| `graphstore/persistence/serializer.py` | Modify | Serialize column arrays to SQLite |
| `graphstore/persistence/deserializer.py` | Modify | Deserialize column arrays from SQLite |
| `tests/test_columns.py` | **Create** | ColumnStore unit tests |
| `tests/test_column_integration.py` | **Create** | End-to-end DSL + column tests |

---

### Task 1: ColumnStore Class

**Files:**
- Create: `graphstore/columns.py`
- Create: `tests/test_columns.py`

- [ ] **Step 1: Write failing tests for set/clear/grow/has_column/get_presence**

```python
# tests/test_columns.py
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
        # Should be able to set at slot 5 now
        cs.set(5, {"score": 99})
        pres = cs.get_presence("score", 6)
        assert pres[5]


class TestTypeMismatch:
    def test_mismatch_stores_sentinel_not_present(self):
        cs = ColumnStore(StringTable(), capacity=8)
        cs.set(0, {"score": 42})       # creates int64 column
        cs.set(1, {"score": "oops"})   # type mismatch
        pres = cs.get_presence("score", 2)
        assert pres[0] and not pres[1]

    def test_float_into_int_column_is_mismatch(self):
        cs = ColumnStore(StringTable(), capacity=8)
        cs.set(0, {"score": 42})
        cs.set(1, {"score": 3.14})     # float into int64 column
        pres = cs.get_presence("score", 2)
        assert pres[0] and not pres[1]

    def test_int_into_float_column_is_ok(self):
        cs = ColumnStore(StringTable(), capacity=8)
        cs.set(0, {"weight": 3.14})    # creates float64 column
        cs.set(1, {"weight": 5})       # int into float64 is fine
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
        # slot 1 has no score
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
        # slot 1 has no score
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
            None,  # tombstone
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
        assert not pres[0]  # no data yet

    def test_declare_does_not_overwrite_existing(self):
        cs = ColumnStore(StringTable(), capacity=8)
        cs.set(0, {"score": 42})
        cs.declare_column("score", "int64")  # no-op
        mask = cs.get_mask("score", "=", 42, 1)
        assert mask[0]  # data preserved


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_columns.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'graphstore.columns'`

- [ ] **Step 3: Implement ColumnStore**

```python
# graphstore/columns.py
"""Columnar acceleration layer for node properties.

Manages typed numpy arrays indexed by slot, providing vectorized
filtering as a read-acceleration layer over the dict-based node_data.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from graphstore.strings import StringTable


class ColumnStore:
    """Typed numpy arrays indexed by slot for fast vectorized filtering."""

    INT64_SENTINEL = np.iinfo(np.int64).min
    STR_SENTINEL = np.int32(-1)

    def __init__(self, string_table: StringTable, capacity: int = 1024):
        self._columns: dict[str, np.ndarray] = {}
        self._presence: dict[str, np.ndarray] = {}
        self._dtypes: dict[str, str] = {}
        self._string_table = string_table
        self._capacity = capacity

    def set(self, slot: int, data: dict) -> None:
        """Write field values to columns. Auto-infers types for new fields."""
        for field, value in data.items():
            if field not in self._dtypes:
                dtype_str = self._infer_dtype(value)
                if dtype_str is None:
                    continue
                self._create_column(field, dtype_str)

            dtype_str = self._dtypes[field]
            if dtype_str == "int64":
                if isinstance(value, int):
                    self._columns[field][slot] = int(value)
                    self._presence[field][slot] = True
                else:
                    self._columns[field][slot] = self.INT64_SENTINEL
                    self._presence[field][slot] = False
            elif dtype_str == "float64":
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self._columns[field][slot] = float(value)
                    self._presence[field][slot] = True
                else:
                    self._columns[field][slot] = np.nan
                    self._presence[field][slot] = False
            elif dtype_str == "int32_interned":
                if isinstance(value, str):
                    self._columns[field][slot] = self._string_table.intern(value)
                    self._presence[field][slot] = True
                else:
                    self._columns[field][slot] = self.STR_SENTINEL
                    self._presence[field][slot] = False

    def clear(self, slot: int) -> None:
        """Clear all column values at slot (node deletion)."""
        for field in self._columns:
            dtype_str = self._dtypes[field]
            if dtype_str == "int64":
                self._columns[field][slot] = self.INT64_SENTINEL
            elif dtype_str == "float64":
                self._columns[field][slot] = np.nan
            elif dtype_str == "int32_interned":
                self._columns[field][slot] = self.STR_SENTINEL
            self._presence[field][slot] = False

    def grow(self, new_capacity: int) -> None:
        """Extend all arrays to new_capacity."""
        for field in list(self._columns):
            old_col = self._columns[field]
            new_col = self._make_sentinel_array(self._dtypes[field], new_capacity)
            new_col[: len(old_col)] = old_col
            self._columns[field] = new_col

            old_pres = self._presence[field]
            new_pres = np.zeros(new_capacity, dtype=bool)
            new_pres[: len(old_pres)] = old_pres
            self._presence[field] = new_pres
        self._capacity = new_capacity

    def get_mask(self, field: str, op: str, value: Any, n: int) -> np.ndarray | None:
        """Return boolean mask for a comparison, or None if field not columnarized."""
        if field not in self._columns:
            return None

        dtype_str = self._dtypes[field]
        col = self._columns[field][:n]
        pres = self._presence[field][:n]

        if value is None:
            if op == "=":
                return ~pres
            elif op == "!=":
                return pres.copy()
            return None

        if dtype_str == "int32_interned":
            if not isinstance(value, str):
                return None
            if value not in self._string_table:
                if op == "=":
                    return np.zeros(n, dtype=bool)
                elif op == "!=":
                    return pres.copy()
                return None
            int_val = self._string_table.intern(value)
            if op == "=":
                return (col == int_val) & pres
            elif op == "!=":
                return (col != int_val) & pres
            return None  # >, <, >=, <= not supported on interned strings

        # Numeric columns (int64, float64)
        ops = {
            "=": lambda c, v: c == v,
            "!=": lambda c, v: c != v,
            ">": lambda c, v: c > v,
            "<": lambda c, v: c < v,
            ">=": lambda c, v: c >= v,
            "<=": lambda c, v: c <= v,
        }
        fn = ops.get(op)
        if fn is None:
            return None
        return fn(col, value) & pres

    def get_mask_in(self, field: str, values: list, n: int) -> np.ndarray | None:
        """Return mask for IN operator."""
        if field not in self._columns:
            return None
        dtype_str = self._dtypes[field]
        col = self._columns[field][:n]
        pres = self._presence[field][:n]

        if dtype_str == "int32_interned":
            int_vals = [
                self._string_table.intern(v)
                for v in values
                if isinstance(v, str) and v in self._string_table
            ]
            if not int_vals:
                return np.zeros(n, dtype=bool)
            return np.isin(col, int_vals) & pres
        return np.isin(col, values) & pres

    def get_presence(self, field: str, n: int) -> np.ndarray | None:
        """Return presence bitmask for a field, or None if not columnarized."""
        if field not in self._presence:
            return None
        return self._presence[field][:n]

    def has_column(self, field: str) -> bool:
        """Check if a column exists for this field."""
        return field in self._columns

    def get_column(self, field: str, n: int) -> tuple[np.ndarray, np.ndarray, str] | None:
        """Return (data[:n], presence[:n], dtype_str) for a column, or None."""
        if field not in self._columns:
            return None
        return self._columns[field][:n], self._presence[field][:n], self._dtypes[field]

    def declare_column(self, field: str, dtype_str: str) -> None:
        """Pre-create a typed column. No-op if column already exists."""
        if field not in self._dtypes:
            self._create_column(field, dtype_str)

    def rebuild_from(self, node_data: list[dict | None], n: int) -> None:
        """Clear all columns and re-scan node_data[:n] to repopulate."""
        self._columns.clear()
        self._presence.clear()
        self._dtypes.clear()
        for slot in range(n):
            data = node_data[slot]
            if data is not None:
                self.set(slot, data)

    @property
    def memory_bytes(self) -> int:
        """Total memory used by column arrays."""
        total = 0
        for field in self._columns:
            total += self._columns[field].nbytes
            total += self._presence[field].nbytes
        return total

    # -- internal helpers ---

    def _infer_dtype(self, value) -> str | None:
        if isinstance(value, bool):
            return "int64"
        if isinstance(value, int):
            return "int64"
        if isinstance(value, float):
            return "float64"
        if isinstance(value, str):
            return "int32_interned"
        return None

    def _create_column(self, field: str, dtype_str: str) -> None:
        self._columns[field] = self._make_sentinel_array(dtype_str, self._capacity)
        self._presence[field] = np.zeros(self._capacity, dtype=bool)
        self._dtypes[field] = dtype_str

    def _make_sentinel_array(self, dtype_str: str, size: int) -> np.ndarray:
        if dtype_str == "int64":
            return np.full(size, self.INT64_SENTINEL, dtype=np.int64)
        elif dtype_str == "float64":
            return np.full(size, np.nan, dtype=np.float64)
        elif dtype_str == "int32_interned":
            return np.full(size, self.STR_SENTINEL, dtype=np.int32)
        raise ValueError(f"Unknown column dtype: {dtype_str}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_columns.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite to verify no regressions**

Run: `pytest tests/ -v`
Expected: All existing tests PASS

- [ ] **Step 6: Commit**

```bash
git add graphstore/columns.py tests/test_columns.py
git commit -m "feat: add ColumnStore class with typed numpy arrays for columnar acceleration"
```

---

### Task 2: CoreStore Write Integration

**Files:**
- Modify: `graphstore/store.py`
- Test: `tests/test_columns.py` (add integration section)

- [ ] **Step 1: Write failing tests for CoreStore column integration**

Append to `tests/test_columns.py`:

```python
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
        # Fill to trigger grow (default capacity 1024)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_columns.py::TestCoreStoreColumnIntegration -v`
Expected: FAIL with `AttributeError: 'CoreStore' object has no attribute 'columns'`

- [ ] **Step 3: Implement CoreStore integration**

In `graphstore/store.py`:

1. Add import at top:
```python
from graphstore.columns import ColumnStore
```

2. In `__init__`, after secondary indices:
```python
        # Columnar acceleration layer
        self.columns = ColumnStore(self.string_table, self._capacity)
```

3. In `put_node`, after `self.node_data[slot] = dict(data)` (line ~109):
```python
        self.columns.set(slot, data)
```

4. In `update_node`, after `self.node_data[slot].update(data)` (line ~155):
```python
        self.columns.set(slot, data)
```

5. In `delete_node`, before tombstone (line ~191, before `self.node_tombstones.add(slot)`):
```python
        self.columns.clear(slot)
```

6. In `increment_field`, after `data[field] = current + amount` (line ~499):
```python
        self.columns.set(slot, {field: data[field]})
```

7. In `_grow`, after extending `node_data` (line ~70):
```python
        self.columns.grow(new_cap)
```

8. Add `_live_mask` method after `_live_slots`:
```python
    def _live_mask(self, kind: str | None = None) -> np.ndarray:
        """Return boolean mask of live slots, optionally filtered by kind."""
        n = self._next_slot
        if n == 0:
            return np.empty(0, dtype=bool)

        mask = self.node_ids[:n] >= 0

        if self.node_tombstones:
            tomb_arr = np.array(list(self.node_tombstones), dtype=np.int32)
            tomb_mask = np.zeros(n, dtype=bool)
            tomb_mask[tomb_arr[tomb_arr < n]] = True
            mask = mask & ~tomb_mask

        if kind is not None:
            if kind not in self.string_table:
                return np.zeros(n, dtype=bool)
            kind_id = self.string_table.intern(kind)
            mask = mask & (self.node_kinds[:n] == kind_id)

        return mask
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_columns.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: All existing tests PASS

- [ ] **Step 6: Commit**

```bash
git add graphstore/store.py tests/test_columns.py
git commit -m "feat: integrate ColumnStore into CoreStore write/delete/grow paths"
```

---

### Task 3: Executor Column Filter Path

**Files:**
- Modify: `graphstore/dsl/executor.py`
- Create: `tests/test_column_integration.py`

- [ ] **Step 1: Write failing integration tests**

```python
# tests/test_column_integration.py
"""End-to-end tests for column-accelerated DSL queries."""

import pytest
from graphstore.store import CoreStore
from graphstore.dsl.parser import parse
from graphstore.dsl.executor import Executor


@pytest.fixture
def graph():
    """Graph with columnarized numeric and string fields."""
    store = CoreStore()
    store.put_node("fn1", "function", {"name": "main", "line": 1, "score": 100})
    store.put_node("fn2", "function", {"name": "helper", "line": 10, "score": 50})
    store.put_node("fn3", "function", {"name": "parse", "line": 5, "score": 200})
    store.put_node("cls1", "class", {"name": "App", "line": 20, "score": 75})
    store.put_node("cls2", "class", {"name": "Base", "line": 1, "score": 25})
    return Executor(store)


def execute(executor, query):
    ast = parse(query)
    return executor.execute(ast)


class TestColumnFilterNodes:
    def test_where_int_gt(self, graph):
        r = execute(graph, 'NODES WHERE score > 90')
        ids = {n["id"] for n in r.data}
        assert ids == {"fn1", "fn3"}

    def test_where_int_eq(self, graph):
        r = execute(graph, 'NODES WHERE line = 1')
        ids = {n["id"] for n in r.data}
        assert ids == {"fn1", "cls2"}

    def test_where_string_eq(self, graph):
        r = execute(graph, 'NODES WHERE name = "main"')
        assert r.count == 1
        assert r.data[0]["id"] == "fn1"

    def test_where_kind_and_int(self, graph):
        r = execute(graph, 'NODES WHERE kind = "function" AND score > 90')
        ids = {n["id"] for n in r.data}
        assert ids == {"fn1", "fn3"}

    def test_where_or(self, graph):
        r = execute(graph, 'NODES WHERE score = 100 OR score = 25')
        ids = {n["id"] for n in r.data}
        assert ids == {"fn1", "cls2"}

    def test_where_not(self, graph):
        r = execute(graph, 'NODES WHERE kind = "function" AND NOT score > 90')
        ids = {n["id"] for n in r.data}
        assert ids == {"fn2"}

    def test_where_in(self, graph):
        r = execute(graph, 'NODES WHERE name IN ("main", "parse")')
        ids = {n["id"] for n in r.data}
        assert ids == {"fn1", "fn3"}

    def test_where_null_eq(self, graph):
        # Add a node without score
        graph.store.put_node("bare", "function", {"name": "bare", "line": 99})
        r = execute(graph, 'NODES WHERE score = NULL')
        ids = {n["id"] for n in r.data}
        assert "bare" in ids

    def test_where_null_neq(self, graph):
        graph.store.put_node("bare", "function", {"name": "bare", "line": 99})
        r = execute(graph, 'NODES WHERE score != NULL')
        ids = {n["id"] for n in r.data}
        assert "bare" not in ids
        assert len(ids) == 5

    def test_contains_falls_back(self, graph):
        r = execute(graph, 'NODES WHERE name CONTAINS "main"')
        assert r.count == 1
        assert r.data[0]["id"] == "fn1"

    def test_like_falls_back(self, graph):
        r = execute(graph, 'NODES WHERE name LIKE "ma%"')
        assert r.count == 1

    def test_order_by_with_limit(self, graph):
        r = execute(graph, 'NODES WHERE kind = "function" ORDER BY score DESC LIMIT 2')
        assert r.data[0]["id"] == "fn3"
        assert r.data[1]["id"] == "fn1"

    def test_order_by_asc(self, graph):
        r = execute(graph, 'NODES WHERE kind = "function" ORDER BY score ASC LIMIT 3')
        assert r.data[0]["score"] == 50
        assert r.data[1]["score"] == 100

    def test_offset(self, graph):
        r = execute(graph, 'NODES WHERE kind = "function" ORDER BY score ASC OFFSET 1 LIMIT 2')
        assert len(r.data) == 2

    def test_order_by_no_where(self, graph):
        r = execute(graph, 'NODES ORDER BY score DESC LIMIT 2')
        assert r.data[0]["score"] == 200
        assert r.data[1]["score"] == 100


class TestColumnFilterCount:
    def test_count_with_filter(self, graph):
        r = execute(graph, 'COUNT NODES WHERE score > 50')
        assert r.data == 3

    def test_count_with_kind_and_filter(self, graph):
        r = execute(graph, 'COUNT NODES WHERE kind = "function" AND score > 50')
        assert r.data == 2


class TestColumnFilterDelete:
    def test_delete_with_filter(self, graph):
        r = execute(graph, 'DELETE NODES WHERE score < 50')
        assert r.count == 1
        assert r.data[0]["id"] == "cls2"
        # Verify deleted
        r2 = execute(graph, 'COUNT NODES')
        assert r2.data == 4
```

- [ ] **Step 2: Run tests to verify they fail or are slower (pre-column path)**

Run: `pytest tests/test_column_integration.py -v`
Expected: Tests should pass even without column filter (existing fallback works). The point is to verify correctness before and after adding the column path.

- [ ] **Step 3: Implement _try_column_filter and wire into _nodes, _count, _delete_nodes**

In `graphstore/dsl/executor.py`:

1. Add import at top:
```python
import numpy as np
```

2. Add `_try_column_filter` method:
```python
    def _try_column_filter(self, expr, base_mask: np.ndarray, n: int) -> np.ndarray | None:
        """Try to evaluate expression using column store. Returns bool mask or None."""
        columns = self.store.columns

        if isinstance(expr, Condition):
            if expr.field in ("kind", "id"):
                return None
            mask = columns.get_mask(expr.field, expr.op, expr.value, n)
            if mask is None:
                return None
            return mask & base_mask

        elif isinstance(expr, InCondition):
            if expr.field in ("kind", "id"):
                return None
            mask = columns.get_mask_in(expr.field, expr.values, n)
            if mask is None:
                return None
            return mask & base_mask

        elif isinstance(expr, AndExpr):
            result = base_mask.copy()
            for op in expr.operands:
                sub = self._try_column_filter(op, result, n)
                if sub is None:
                    return None
                result = sub
            return result

        elif isinstance(expr, OrExpr):
            result = np.zeros(n, dtype=bool)
            for op in expr.operands:
                sub = self._try_column_filter(op, base_mask, n)
                if sub is None:
                    return None
                result |= sub
            return result

        elif isinstance(expr, NotExpr):
            sub = self._try_column_filter(expr.operand, base_mask, n)
            if sub is None:
                return None
            # Absent values should NOT match NOT - AND with field presence
            fields = self._column_fields(expr.operand)
            if fields is None:
                return None
            combined_pres = np.ones(n, dtype=bool)
            for f in fields:
                fp = self.store.columns.get_presence(f, n)
                if fp is None:
                    return None
                combined_pres &= fp
            return ~sub & combined_pres & base_mask

        return None

    def _column_fields(self, expr) -> set[str] | None:
        """Extract field names from expression. None if non-columnarizable."""
        if isinstance(expr, (Condition, ContainsCondition, LikeCondition, InCondition)):
            if expr.field in ("kind", "id"):
                return None
            return {expr.field}
        if isinstance(expr, AndExpr):
            fields = set()
            for op in expr.operands:
                sub = self._column_fields(op)
                if sub is None:
                    return None
                fields |= sub
            return fields
        if isinstance(expr, OrExpr):
            fields = set()
            for op in expr.operands:
                sub = self._column_fields(op)
                if sub is None:
                    return None
                fields |= sub
            return fields
        if isinstance(expr, NotExpr):
            return self._column_fields(expr.operand)
        return None
```

3. Add helper methods:
```python
    def _try_column_nodes(self, expr, kind_filter: str | None) -> list[dict] | None:
        """Try column-accelerated node query. Returns node dicts or None."""
        n = self.store._next_slot
        if n == 0:
            return []
        base_mask = self.store._live_mask(kind_filter)
        col_mask = self._try_column_filter(expr, base_mask, n)
        if col_mask is None:
            return None
        slots = np.nonzero(col_mask)[0]
        nodes = []
        for slot in slots:
            node = self.store._materialize_slot(int(slot))
            if node:
                nodes.append(node)
        return nodes

    def _try_column_count(self, expr, kind_filter: str | None) -> int | None:
        """Try column-accelerated count. Returns count or None."""
        n = self.store._next_slot
        if n == 0:
            return 0
        base_mask = self.store._live_mask(kind_filter)
        col_mask = self._try_column_filter(expr, base_mask, n)
        if col_mask is None:
            return None
        return int(np.count_nonzero(col_mask))

    def _try_column_delete_ids(self, expr, kind_filter: str | None) -> list[str] | None:
        """Try column-accelerated ID query for deletion. Returns IDs or None."""
        n = self.store._next_slot
        if n == 0:
            return []
        base_mask = self.store._live_mask(kind_filter)
        col_mask = self._try_column_filter(expr, base_mask, n)
        if col_mask is None:
            return None
        slots = np.nonzero(col_mask)[0]
        return [
            nid for slot in slots
            if (nid := self.store._slot_to_id(int(slot))) is not None
        ]

    def _try_column_order_by(self, nodes: list[dict], field: str,
                              descending: bool, limit: int | None,
                              offset: int | None) -> list[dict] | None:
        """Try column-accelerated ORDER BY using np.argpartition for top-K.

        Returns sorted+sliced node list, or None if field not columnarized.
        Only optimizes when LIMIT is set (top-K pattern). Falls back for
        full sorts since the existing Python sort is adequate.
        """
        col_info = self.store.columns.get_column(field, self.store._next_slot)
        if col_info is None:
            return None
        col_data, col_pres, dtype_str = col_info

        # String columns: can't do numeric ORDER BY on interned IDs
        if dtype_str == "int32_interned":
            return None

        # Build slot -> index mapping for the materialized nodes
        slot_to_idx: dict[int, int] = {}
        for i, node in enumerate(nodes):
            slot = self._resolve_slot(node["id"])
            if slot is not None:
                slot_to_idx[slot] = i

        if not slot_to_idx:
            return nodes

        slots = np.array(list(slot_to_idx.keys()), dtype=np.int32)
        values = col_data[slots].astype(np.float64)
        present = col_pres[slots]

        # Non-present values go to end
        if descending:
            values[~present] = -np.inf
        else:
            values[~present] = np.inf

        total = len(slots)
        eff_offset = (offset or 0)
        eff_limit = limit if limit is not None else total

        k = min(eff_offset + eff_limit, total)

        if k < total and k > 0:
            # argpartition: O(n) to find top-k, then sort the k results
            if descending:
                # Negate for descending partition
                part_idx = np.argpartition(-values, k)[:k]
                sorted_idx = part_idx[np.argsort(-values[part_idx])]
            else:
                part_idx = np.argpartition(values, k)[:k]
                sorted_idx = part_idx[np.argsort(values[part_idx])]
        else:
            # Full sort
            if descending:
                sorted_idx = np.argsort(-values)
            else:
                sorted_idx = np.argsort(values)

        sorted_idx = sorted_idx[eff_offset:eff_offset + eff_limit]

        result = []
        for idx in sorted_idx:
            slot = int(slots[idx])
            node_idx = slot_to_idx[slot]
            result.append(nodes[node_idx])
        return result
```

4. Update `_nodes` to try column filter:

Replace the block inside `if nodes is None:` / `if remaining is not None:`:
```python
            if remaining is not None:
                # Try column filter first
                col_result = self._try_column_nodes(remaining, kind_filter)
                if col_result is not None:
                    nodes = col_result
                else:
                    raw_pred = self._make_raw_predicate(remaining)
                    if raw_pred is not None:
                        nodes = self.store.get_all_nodes(kind=kind_filter, predicate=raw_pred)
                    else:
                        nodes = self.store.get_all_nodes(kind=kind_filter)
                        nodes = [n for n in nodes if self._eval_where(q.where.expr, n)]
```

5. Update `_count` (NODES branch) similarly:

Replace the `else:` branch after `if remaining is None:`:
```python
                else:
                    col_count = self._try_column_count(remaining, kind_filter)
                    if col_count is not None:
                        count = col_count
                    else:
                        raw_pred = self._make_raw_predicate(remaining)
                        if raw_pred is not None:
                            count = self.store.count_nodes(kind=kind_filter, predicate=raw_pred)
                        else:
                            nodes = self.store.get_all_nodes(kind=kind_filter)
                            count = sum(1 for n in nodes if self._eval_where(q.where.expr, n))
```

6. Update `_delete_nodes` to try column filter:

After extracting `remaining`, before the existing `if remaining is not None:`:
```python
        ids_to_delete = None

        if remaining is not None:
            col_ids = self._try_column_delete_ids(remaining, kind_filter)
            if col_ids is not None:
                ids_to_delete = col_ids

        if ids_to_delete is None:
            if remaining is not None:
                raw_pred = self._make_raw_predicate(remaining)
            else:
                raw_pred = None

            if raw_pred is not None or remaining is None:
                ids_to_delete = self.store.query_node_ids(kind=kind_filter, predicate=raw_pred)
            else:
                nodes = self.store.get_all_nodes(kind=kind_filter)
                ids_to_delete = [n["id"] for n in nodes if self._eval_where(q.where.expr, n)]
```

7. Update ORDER BY in `_nodes` to use column acceleration.

Replace the existing ORDER BY block:
```python
        if q.order:
            reverse = q.order.direction == "DESC"
            nodes.sort(key=lambda n: (n.get(q.order.field) is None, n.get(q.order.field, "")), reverse=reverse)
        if q.offset:
            nodes = nodes[q.offset.value:]
        if q.limit:
            nodes = nodes[:q.limit.value]
```

With:
```python
        if q.order:
            reverse = q.order.direction == "DESC"
            col_sorted = self._try_column_order_by(
                nodes, q.order.field, reverse,
                q.limit.value if q.limit else None,
                q.offset.value if q.offset else None,
            )
            if col_sorted is not None:
                nodes = col_sorted
            else:
                nodes.sort(
                    key=lambda n: (n.get(q.order.field) is None, n.get(q.order.field, "")),
                    reverse=reverse,
                )
                if q.offset:
                    nodes = nodes[q.offset.value:]
                if q.limit:
                    nodes = nodes[:q.limit.value]
        else:
            if q.offset:
                nodes = nodes[q.offset.value:]
            if q.limit:
                nodes = nodes[:q.limit.value]
```

- [ ] **Step 4: Run integration tests**

Run: `pytest tests/test_column_integration.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add graphstore/dsl/executor.py tests/test_column_integration.py
git commit -m "feat: column-accelerated filter, COUNT, DELETE, and ORDER BY for NODES queries"
```

---

### Task 4: Batch Rollback + SYS REBUILD

**Files:**
- Modify: `graphstore/dsl/executor.py` (batch rollback)
- Modify: `graphstore/dsl/executor_system.py` (rebuild)
- Test: `tests/test_column_integration.py` (add tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_column_integration.py`:

```python
from graphstore.schema import SchemaRegistry
from graphstore.dsl.executor_system import SystemExecutor
from graphstore.errors import BatchRollback


class TestBatchRollbackColumns:
    def test_rollback_restores_columns(self):
        store = CoreStore()
        store.put_node("n1", "fn", {"score": 10})
        executor = Executor(store)

        # Batch that fails partway through
        with pytest.raises(BatchRollback):
            execute(executor, '''BEGIN
CREATE NODE "n2" kind = "fn" score = 99
CREATE NODE "n1" kind = "fn" score = 50
COMMIT''')

        # n2 should not exist (rolled back)
        assert store.get_node("n2") is None
        # Column should reflect original state
        mask = store.columns.get_mask("score", "=", 10, store._next_slot)
        assert mask[0]


class TestSysRebuildColumns:
    def test_rebuild_restores_columns(self):
        store = CoreStore()
        store.put_node("n1", "fn", {"score": 42})
        schema = SchemaRegistry()
        sys_exec = SystemExecutor(store, schema)

        # Corrupt columns manually
        store.columns._columns.clear()
        store.columns._presence.clear()
        store.columns._dtypes.clear()
        assert not store.columns.has_column("score")

        # Rebuild should restore
        ast = parse("SYS REBUILD INDICES")
        sys_exec.execute(ast)
        assert store.columns.has_column("score")
        mask = store.columns.get_mask("score", "=", 42, store._next_slot)
        assert mask[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_column_integration.py::TestBatchRollbackColumns -v && pytest tests/test_column_integration.py::TestSysRebuildColumns -v`
Expected: TestSysRebuildColumns FAILS (rebuild doesn't restore columns yet). TestBatchRollbackColumns may also fail.

- [ ] **Step 3: Implement batch rollback + SYS REBUILD column restoration**

In `graphstore/dsl/executor.py`, in the `_batch` method's `except` block, after `self.store._rebuild_edges()`:
```python
            self.store.columns.rebuild_from(
                self.store.node_data, self.store._next_slot
            )
```

In `graphstore/dsl/executor_system.py`, in `_rebuild`, after rebuilding secondary indices:
```python
        self.store.columns.rebuild_from(
            self.store.node_data, self.store._next_slot
        )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_column_integration.py::TestBatchRollbackColumns tests/test_column_integration.py::TestSysRebuildColumns -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add graphstore/dsl/executor.py graphstore/dsl/executor_system.py tests/test_column_integration.py
git commit -m "feat: column rebuild on batch rollback and SYS REBUILD INDICES"
```

---

### Task 5: Typed Fields (Grammar + AST + Transformer + Schema)

**Files:**
- Modify: `graphstore/dsl/grammar.lark`
- Modify: `graphstore/dsl/ast_nodes.py`
- Modify: `graphstore/dsl/transformer.py`
- Modify: `graphstore/schema.py`
- Modify: `graphstore/dsl/executor_system.py`
- Test: `tests/test_dsl_parser.py` (add tests), `tests/test_dsl_system.py` (add tests), `tests/test_schema.py` (add tests)

- [ ] **Step 1: Write failing parser test**

Append to `tests/test_dsl_parser.py`:

```python
class TestTypedFieldParsing:
    def test_register_with_typed_fields(self):
        ast = parse('SYS REGISTER NODE KIND "function" REQUIRED name:string, line:int OPTIONAL score:float')
        assert isinstance(ast, SysRegisterNodeKind)
        assert ast.required == [("name", "string"), ("line", "int")]
        assert ast.optional == [("score", "float")]

    def test_register_mixed_typed_untyped(self):
        ast = parse('SYS REGISTER NODE KIND "function" REQUIRED name:string, description')
        assert ast.required == [("name", "string"), ("description", None)]

    def test_register_untyped_still_works(self):
        ast = parse('SYS REGISTER NODE KIND "thing" REQUIRED name, value')
        assert ast.required == [("name", None), ("value", None)]
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_dsl_parser.py::TestTypedFieldParsing -v`
Expected: FAIL (grammar doesn't support `name:string` syntax yet)

- [ ] **Step 3: Implement grammar changes**

In `graphstore/dsl/grammar.lark`, replace the `ident_list` rule (line 145):

```lark
ident_list: typed_ident ("," typed_ident)*
typed_ident: IDENTIFIER ":" IDENTIFIER -> typed_ident_with_type
           | IDENTIFIER -> typed_ident_bare
```

- [ ] **Step 4: Implement AST changes**

In `graphstore/dsl/ast_nodes.py`, update `SysRegisterNodeKind`:

```python
@dataclass
class SysRegisterNodeKind:
    kind: str
    required: list[tuple[str, str | None]]  # [(field_name, type_name_or_none), ...]
    optional: list[tuple[str, str | None]]
```

- [ ] **Step 5: Implement transformer changes**

In `graphstore/dsl/transformer.py`, replace the `ident_list` method and add `typed_ident` handlers:

```python
    def typed_ident_with_type(self, args):
        return (str(args[0]), str(args[1]))

    def typed_ident_bare(self, args):
        return (str(args[0]), None)

    def ident_list(self, args):
        return list(args)
```

- [ ] **Step 6: Run parser tests**

Run: `pytest tests/test_dsl_parser.py::TestTypedFieldParsing -v`
Expected: PASS

- [ ] **Step 7: Write failing schema tests**

Append to `tests/test_schema.py`:

```python
class TestTypedFields:
    def test_register_with_types(self):
        sr = SchemaRegistry()
        sr.register_node_kind("fn", [("name", "string"), ("line", "int")], [("score", "float")])
        defn = sr.get_node_kind("fn")
        assert "name" in defn.required
        assert defn.field_types == {"name": "string", "line": "int", "score": "float"}

    def test_validate_type_mismatch_raises(self):
        sr = SchemaRegistry()
        sr.register_node_kind("fn", [("name", "string"), ("line", "int")])
        with pytest.raises(SchemaError):
            sr.validate_node("fn", {"name": 42, "line": 1})  # name should be string

    def test_validate_correct_types_passes(self):
        sr = SchemaRegistry()
        sr.register_node_kind("fn", [("name", "string"), ("line", "int")])
        sr.validate_node("fn", {"name": "main", "line": 1})  # no error

    def test_untyped_fields_skip_type_check(self):
        sr = SchemaRegistry()
        sr.register_node_kind("fn", [("name", None)], [])
        sr.validate_node("fn", {"name": 42})  # no type declared, no error

    def test_backward_compat_string_list(self):
        sr = SchemaRegistry()
        sr.register_node_kind("fn", ["name", "line"])  # old format
        defn = sr.get_node_kind("fn")
        assert "name" in defn.required
        assert defn.field_types == {}

    def test_describe_includes_types(self):
        sr = SchemaRegistry()
        sr.register_node_kind("fn", [("name", "string")])
        desc = sr.describe_node_kind("fn")
        assert desc["field_types"] == {"name": "string"}

    def test_serialization_preserves_types(self):
        sr = SchemaRegistry()
        sr.register_node_kind("fn", [("name", "string"), ("line", "int")])
        data = sr.to_dict()
        sr2 = SchemaRegistry.from_dict(data)
        defn = sr2.get_node_kind("fn")
        assert defn.field_types == {"name": "string", "line": "int"}
```

- [ ] **Step 8: Implement schema changes**

In `graphstore/schema.py`:

1. Update `NodeKindDef`:
```python
from dataclasses import dataclass, field

@dataclass
class NodeKindDef:
    required: set[str]
    optional: set[str]
    field_types: dict[str, str] = field(default_factory=dict)
```

2. Update `register_node_kind` to handle both old `list[str]` and new `list[tuple]` formats:
```python
    def register_node_kind(self, kind: str, required: list, optional: list | None = None):
        req_fields = set()
        field_types = {}
        for item in required:
            if isinstance(item, tuple):
                name, type_name = item
                req_fields.add(name)
                if type_name:
                    field_types[name] = type_name
            else:
                req_fields.add(item)

        opt_fields = set()
        for item in (optional or []):
            if isinstance(item, tuple):
                name, type_name = item
                opt_fields.add(name)
                if type_name:
                    field_types[name] = type_name
            else:
                opt_fields.add(item)

        self._node_kinds[kind] = NodeKindDef(
            required=req_fields, optional=opt_fields, field_types=field_types,
        )
```

3. Add type validation to `validate_node`:
```python
    def validate_node(self, kind: str, data: dict):
        if kind not in self._node_kinds:
            return
        defn = self._node_kinds[kind]
        missing = defn.required - set(data.keys())
        if missing:
            raise SchemaError(f"kind '{kind}' requires fields: {sorted(missing)}")
        for field_name, value in data.items():
            if field_name in defn.field_types:
                expected = defn.field_types[field_name]
                if not self._type_matches(value, expected):
                    raise SchemaError(
                        f"Field '{field_name}' expects type '{expected}', "
                        f"got {type(value).__name__}"
                    )

    @staticmethod
    def _type_matches(value, type_name: str) -> bool:
        if type_name == "string":
            return isinstance(value, str)
        if type_name == "int":
            return isinstance(value, int) and not isinstance(value, bool)
        if type_name == "float":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        return True
```

4. Update `describe_node_kind`:
```python
    def describe_node_kind(self, kind: str) -> dict | None:
        defn = self._node_kinds.get(kind)
        if defn is None:
            return None
        result = {
            "kind": kind,
            "required": sorted(defn.required),
            "optional": sorted(defn.optional),
        }
        if defn.field_types:
            result["field_types"] = defn.field_types
        return result
```

5. Update `to_dict`:
```python
    def to_dict(self) -> dict:
        return {
            "node_kinds": {
                k: {
                    "required": sorted(v.required),
                    "optional": sorted(v.optional),
                    "field_types": v.field_types,
                }
                for k, v in self._node_kinds.items()
            },
            "edge_kinds": {
                k: {"from_kinds": sorted(v.from_kinds), "to_kinds": sorted(v.to_kinds)}
                for k, v in self._edge_kinds.items()
            },
        }
```

6. Update `from_dict`:
```python
    @classmethod
    def from_dict(cls, data: dict) -> SchemaRegistry:
        registry = cls()
        for kind, defn in data.get("node_kinds", {}).items():
            field_types = defn.get("field_types", {})
            required = [(f, field_types.get(f)) for f in defn["required"]]
            optional = [(f, field_types.get(f)) for f in defn.get("optional", [])]
            registry.register_node_kind(kind, required, optional)
        for kind, defn in data.get("edge_kinds", {}).items():
            registry.register_edge_kind(kind, defn["from_kinds"], defn["to_kinds"])
        return registry
```

- [ ] **Step 9: Update SystemExecutor to pre-create columns for typed fields**

In `graphstore/dsl/executor_system.py`, update `_register_node_kind`:

```python
    def _register_node_kind(self, q: SysRegisterNodeKind) -> Result:
        self.schema.register_node_kind(q.kind, q.required, q.optional)
        # Pre-create columns for typed fields
        type_map = {"string": "int32_interned", "int": "int64", "float": "float64"}
        for item in q.required + q.optional:
            if isinstance(item, tuple):
                name, type_name = item
            else:
                name, type_name = item, None
            if type_name and type_name in type_map:
                self.store.columns.declare_column(name, type_map[type_name])
        return Result(kind="ok", data=None, count=0)
```

- [ ] **Step 10: Run all tests**

Run: `pytest tests/test_dsl_parser.py tests/test_schema.py tests/test_dsl_system.py -v`
Expected: All PASS

- [ ] **Step 11: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 12: Commit**

```bash
git add graphstore/dsl/grammar.lark graphstore/dsl/ast_nodes.py graphstore/dsl/transformer.py graphstore/schema.py graphstore/dsl/executor_system.py tests/test_dsl_parser.py tests/test_schema.py
git commit -m "feat: typed field declarations in schema registration (name:string, line:int)"
```

---

### Task 6: Persistence

**Files:**
- Modify: `graphstore/persistence/serializer.py`
- Modify: `graphstore/persistence/deserializer.py`
- Test: `tests/test_persistence.py` (add tests)

- [ ] **Step 1: Write failing persistence tests**

Append to `tests/test_persistence.py`:

```python
class TestColumnPersistence:
    def test_checkpoint_and_load_preserves_columns(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = open_database(db_path)

        store = CoreStore()
        schema = SchemaRegistry()
        store.put_node("n1", "fn", {"score": 42, "name": "main"})
        store.put_node("n2", "fn", {"score": 99, "name": "helper"})

        checkpoint(store, schema, conn)
        store2, schema2 = load(conn)

        assert store2.columns.has_column("score")
        assert store2.columns.has_column("name")
        mask = store2.columns.get_mask("score", "=", 42, store2._next_slot)
        assert mask[0] and not mask[1]
        mask2 = store2.columns.get_mask("name", "=", "main", store2._next_slot)
        assert mask2[0] and not mask2[1]
        conn.close()

    def test_backward_compat_no_column_data(self, tmp_path):
        """Loading a checkpoint without column data should work (empty columns)."""
        db_path = tmp_path / "test.db"
        conn = open_database(db_path)

        store = CoreStore()
        schema = SchemaRegistry()
        store.put_node("n1", "fn", {"score": 42})

        # Checkpoint without column data (simulate old format)
        checkpoint(store, schema, conn)
        # Delete column blobs to simulate old format
        conn.execute("DELETE FROM blobs WHERE key LIKE 'columns:%'")
        conn.commit()

        store2, schema2 = load(conn)
        # Columns should be empty but not crash
        assert not store2.columns.has_column("score")
        # Auto-inference kicks in on next write
        store2.put_node("n2", "fn", {"score": 99})
        assert store2.columns.has_column("score")
        conn.close()

    def test_column_dtypes_preserved(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = open_database(db_path)

        store = CoreStore()
        schema = SchemaRegistry()
        store.put_node("n1", "fn", {"line": 42, "weight": 3.14, "name": "x"})

        checkpoint(store, schema, conn)
        store2, _ = load(conn)

        assert store2.columns._dtypes["line"] == "int64"
        assert store2.columns._dtypes["weight"] == "float64"
        assert store2.columns._dtypes["name"] == "int32_interned"
        conn.close()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_persistence.py::TestColumnPersistence -v`
Expected: FAIL (columns not serialized/deserialized yet)

- [ ] **Step 3: Implement serialization**

In `graphstore/persistence/serializer.py`, after the node_data serialization and before the edge matrices section, add:

```python
        # Column store data
        conn.execute("DELETE FROM blobs WHERE key LIKE 'columns:%'")
        for field in store.columns._columns:
            col_data = store.columns._columns[field][:store._next_slot]
            pres_data = store.columns._presence[field][:store._next_slot]
            dtype_str = store.columns._dtypes[field]
            conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                         (f"columns:{field}:data", col_data.tobytes(), str(col_data.dtype)))
            conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                         (f"columns:{field}:presence", pres_data.tobytes(), str(pres_data.dtype)))
            conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                         (f"columns:{field}:dtype", dtype_str.encode(), "text"))
```

- [ ] **Step 4: Implement deserialization**

In `graphstore/persistence/deserializer.py`, add import:
```python
from graphstore.columns import ColumnStore
```

After loading indexed fields and before loading schema, add:

```python
    # Load column store data
    col_rows = conn.execute(
        "SELECT key, data, dtype FROM blobs WHERE key LIKE 'columns:%'"
    ).fetchall()

    if col_rows:
        col_blobs: dict[str, dict] = {}
        for key, data, dtype in col_rows:
            parts = key.split(":", 2)
            if len(parts) == 3:
                field_name = parts[1]
                sub_key = parts[2]
                col_blobs.setdefault(field_name, {})[sub_key] = (data, dtype)

        for field_name, blobs in col_blobs.items():
            if "data" in blobs and "presence" in blobs and "dtype" in blobs:
                data_blob, data_np_dtype = blobs["data"]
                pres_blob, pres_np_dtype = blobs["presence"]
                col_dtype_raw = blobs["dtype"][0]
                col_dtype_str = col_dtype_raw.decode() if isinstance(col_dtype_raw, bytes) else col_dtype_raw

                col_arr = np.frombuffer(data_blob, dtype=np.dtype(data_np_dtype)).copy()
                pres_arr = np.frombuffer(pres_blob, dtype=np.dtype(pres_np_dtype)).copy()

                full_col = store.columns._make_sentinel_array(col_dtype_str, capacity)
                full_col[:len(col_arr)] = col_arr
                store.columns._columns[field_name] = full_col

                full_pres = np.zeros(capacity, dtype=bool)
                full_pres[:len(pres_arr)] = pres_arr
                store.columns._presence[field_name] = full_pres

                store.columns._dtypes[field_name] = col_dtype_str
```

- [ ] **Step 5: Run persistence tests**

Run: `pytest tests/test_persistence.py::TestColumnPersistence -v`
Expected: All PASS

- [ ] **Step 6: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add graphstore/persistence/serializer.py graphstore/persistence/deserializer.py tests/test_persistence.py
git commit -m "feat: persist column store data in SQLite checkpoints"
```

---

### Task 7: Memory Stats + Final Integration

**Files:**
- Modify: `graphstore/dsl/executor_system.py`
- Test: `tests/test_column_integration.py` (add end-to-end tests)

- [ ] **Step 1: Write test for column memory in stats**

Append to `tests/test_column_integration.py`:

```python
class TestColumnMemoryStats:
    def test_stats_includes_column_memory(self):
        store = CoreStore()
        store.put_node("n1", "fn", {"score": 42})
        schema = SchemaRegistry()
        sys_exec = SystemExecutor(store, schema)

        ast = parse("SYS STATS MEMORY")
        r = sys_exec.execute(ast)
        assert "column_memory_bytes" in r.data
        assert r.data["column_memory_bytes"] > 0
```

- [ ] **Step 2: Implement column memory in stats**

In `graphstore/dsl/executor_system.py`, in `_stats`, inside the `MEMORY` block:

```python
        if q.target is None or q.target == "MEMORY":
            data["memory_bytes"] = estimate_memory(
                self.store.node_count, self.store.edge_count
            )
            data["ceiling_bytes"] = self.store._ceiling_bytes
            data["column_memory_bytes"] = self.store.columns.memory_bytes
```

- [ ] **Step 3: Write end-to-end DSL integration tests**

Append to `tests/test_column_integration.py`:

```python
from graphstore import GraphStore


class TestEndToEnd:
    def test_graphstore_round_trip(self, tmp_path):
        """Full lifecycle: create -> query -> checkpoint -> reload -> query."""
        with GraphStore(str(tmp_path / "db"), ceiling_mb=64) as gs:
            gs.execute('CREATE NODE "fn1" kind = "function" name = "main" score = 100')
            gs.execute('CREATE NODE "fn2" kind = "function" name = "helper" score = 50')
            gs.execute('CREATE NODE "fn3" kind = "function" name = "parse" score = 200')
            gs.checkpoint()

        # Reload
        with GraphStore(str(tmp_path / "db"), ceiling_mb=64) as gs:
            r = gs.execute('NODES WHERE score > 90')
            ids = {n["id"] for n in r.data}
            assert ids == {"fn1", "fn3"}

            r = gs.execute('COUNT NODES WHERE score > 50')
            assert r.data == 2

    def test_typed_schema_end_to_end(self, tmp_path):
        with GraphStore(str(tmp_path / "db")) as gs:
            gs.execute('SYS REGISTER NODE KIND "function" REQUIRED name:string, line:int OPTIONAL score:float')
            gs.execute('CREATE NODE "fn1" kind = "function" name = "main" line = 1 score = 9.5')

            r = gs.execute('NODES WHERE name = "main"')
            assert r.count == 1

            r = gs.execute('COUNT NODES WHERE line = 1')
            assert r.data == 1

    def test_columns_survive_wal_replay(self, tmp_path):
        with GraphStore(str(tmp_path / "db")) as gs:
            gs.execute('CREATE NODE "n1" kind = "fn" score = 42')
            # Don't checkpoint - force WAL replay on next open

        with GraphStore(str(tmp_path / "db")) as gs:
            r = gs.execute('NODES WHERE score = 42')
            assert r.count == 1
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/test_column_integration.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASS (zero regressions)

- [ ] **Step 6: Commit**

```bash
git add graphstore/dsl/executor_system.py tests/test_column_integration.py
git commit -m "feat: column memory stats and end-to-end integration tests"
```
