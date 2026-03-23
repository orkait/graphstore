"""Columnar storage layer for node properties.

Manages typed numpy arrays indexed by slot, providing vectorized
filtering. This is the sole source of truth for node field data.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from graphstore.core.strings import StringTable


class ColumnStore:
    """Typed numpy arrays indexed by slot for fast vectorized filtering."""

    INT64_SENTINEL = np.iinfo(np.int64).min
    STR_SENTINEL = np.int32(-1)
    _NUM_OPS = {
        "=": np.equal,
        "!=": np.not_equal,
        ">": np.greater,
        "<": np.less,
        ">=": np.greater_equal,
        "<=": np.less_equal,
    }

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

        fn = self._NUM_OPS.get(op)
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

    def _ensure_column(self, field: str, dtype_str: str) -> None:
        """Create column if it doesn't exist. No-op if it already exists."""
        self.declare_column(field, dtype_str)

    def set_reserved(self, slot: int, field: str, value) -> None:
        """Set a system-managed column value. Auto-interns strings."""
        if isinstance(value, str):
            self._ensure_column(field, "int32_interned")
            self._columns[field][slot] = self._string_table.intern(value)
        elif isinstance(value, float):
            self._ensure_column(field, "float64")
            self._columns[field][slot] = value
        else:
            self._ensure_column(field, "int64")
            self._columns[field][slot] = int(value)
        self._presence[field][slot] = True

    def set_field(self, slot: int, field: str, value) -> None:
        """Set a single field value at a slot. Auto-infers type like set()."""
        self.set(slot, {field: value})

    def snapshot_arrays(self) -> dict[str, tuple]:
        """Return deep copies of all column arrays for snapshot/rollback.

        Returns dict mapping field -> (column_copy, presence_copy, dtype_str).
        """
        snap: dict[str, tuple] = {}
        for field in self._columns:
            snap[field] = (
                self._columns[field].copy(),
                self._presence[field].copy(),
                self._dtypes[field],
            )
        return snap

    def restore_arrays(self, snap: dict[str, tuple]) -> None:
        """Restore column arrays from a snapshot created by snapshot_arrays()."""
        self._columns.clear()
        self._presence.clear()
        self._dtypes.clear()
        for field, (col, pres, dtype_str) in snap.items():
            self._columns[field] = col
            self._presence[field] = pres
            self._dtypes[field] = dtype_str

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
