# Hybrid Columnar Storage for graphstore

## Goal

Add a thin columnar acceleration layer to graphstore's node storage that provides numpy-vectorized filtering (100-300x speedup) and reduced memory overhead at scale (50K-500K+ nodes), without replacing the existing dict-based storage or breaking any existing APIs.

## Context

graphstore stores node properties as `list[dict | None]` indexed by slot. Filtering requires iterating Python dicts one-by-one, even with the recently added late-materialization optimization. At 50K+ nodes, this becomes the dominant bottleneck for WHERE, COUNT, and ORDER BY queries.

Node IDs and kinds are already stored as numpy arrays (`node_ids`, `node_kinds`). This design extends that pattern to user-defined properties.

## Architecture

**Approach: Hybrid (Thin Column Layer)**

`node_data: list[dict]` stays as the source of truth for all fields. A new `ColumnStore` maintains typed numpy arrays as a read-acceleration layer. CoreStore dual-writes to both on every mutation. Reads check columns first for filterable fields, fall back to dict for uncolumnarized fields.

This is additive - nothing breaks if columns aren't populated. The column layer can later evolve into a full replacement (dropping dicts entirely) when scale demands it.

**Why not full column replacement?** Requires rewriting every read/write/serialization path and all tests. Risk-to-benefit ratio is wrong for now.

**Why not Polars/Arrow?** DataFrames are append-optimized. graphstore's mutable-graph model (`update_node` mutates a single field on a single node) fights Polars at every write.

## Components

### ColumnStore (`graphstore/columns.py`)

Single new file. Manages typed numpy arrays indexed by slot.

```python
class ColumnStore:
    _columns: dict[str, np.ndarray]    # field_name -> typed array
    _presence: dict[str, np.ndarray]   # field_name -> bool mask
    _dtypes: dict[str, np.dtype]       # field_name -> type
    _string_table: StringTable         # shared ref for string interning
```

**Supported column types:**

| Python type | Column dtype | Sentinel | Use case |
|-------------|-------------|----------|----------|
| `int` | `int64` | `np.iinfo(int64).min` | line, hits, score |
| `float` | `float64` | `NaN` | latency, weight |
| `str` | `int32` (interned) | `-1` | name, file, method |
| `bool` | `int64` | sentinel | stored as 0/1 in int64 (bool is subclass of int) |

**Public interface:**

```python
def set(self, slot: int, data: dict) -> None
    """Write field values to columns. Auto-infers types for new fields."""

def clear(self, slot: int) -> None
    """Clear all column values at slot (node deletion)."""

def grow(self, new_capacity: int) -> None
    """Extend all arrays to new_capacity (called by CoreStore._grow)."""

def get_mask(self, field: str, op: str, value: Any, n: int) -> np.ndarray | None
    """Return boolean mask for a comparison, or None if field not columnarized."""

def get_presence(self, field: str, n: int) -> np.ndarray | None
    """Return presence bitmask for a field, or None if not columnarized."""

def has_column(self, field: str) -> bool
    """Check if a column exists for this field."""
```

### Type inference

**Auto-inference (default):** On first `set()` call for a new field, the Python type of the value determines the column dtype. Once created, the column type is fixed.

**Type mismatch:** If a value's type doesn't match the column's dtype:
- Value goes to `node_data[slot]` only (dict)
- Column stores sentinel + `presence[slot] = False`
- No crash, no data loss, no warning

**Explicit declaration (optional):** Extend schema registration to accept type annotations:

```
SYS REGISTER NODE KIND "function" REQUIRED name:string, line:int OPTIONAL score:float
```

When declared, type mismatches raise `SchemaError` instead of silently falling back. Existing `REQUIRED name` (no type) continues to work - types are additive.

**Grammar changes for typed fields:**

The `ident_list` rule in `grammar.lark` needs a typed variant:

```lark
typed_ident: IDENTIFIER ":" IDENTIFIER    // name:string
           | IDENTIFIER                   // name (untyped, backward compat)
typed_ident_list: typed_ident ("," typed_ident)*
```

The `SysRegisterNodeKind` AST node changes from `required: list[str]` to `required: list[tuple[str, str | None]]` where each tuple is `(field_name, type_name_or_none)`. The transformer maps type names to column dtypes: `"string"` -> `int32_interned`, `"int"` -> `int64`, `"float"` -> `float64`. Untyped fields get `None` (no column pre-created, auto-inferred on first write).

### CoreStore write integration

In `put_node`, `update_node`, `upsert_node`:

```python
# Existing: self.node_data[slot] = dict(data)
# Add:      self.columns.set(slot, data)
```

In `increment_field`:

```python
# After: data[field] = current + amount
# Add:   self.columns.set(slot, {field: data[field]})
```

`increment_field` mutates `node_data[slot]` directly without going through `update_node`, so it must push the updated value to the column explicitly.

In `delete_node`:

```python
# Add: self.columns.clear(slot)
```

In `_grow`:

```python
# Add: self.columns.grow(new_capacity)
```

Overhead per write: one numpy scalar assignment per columnarized field (~50ns each). Negligible compared to dict operations.

### Batch rollback

`Executor._batch` saves and restores CoreStore state on failure. ColumnStore must also be restorable. Since `node_data` dicts remain the source of truth, the simplest approach is to rebuild columns from dicts after rollback:

```python
# In _batch rollback handler, after restoring node_data:
self.store.columns.rebuild_from(self.store.node_data, self.store._next_slot)
```

`ColumnStore.rebuild_from(node_data, n)` clears all columns and re-scans `node_data[:n]` to repopulate. This is O(n * fields) but batch rollback is an error path, not a hot path.

### SYS REBUILD integration

`SYS REBUILD INDICES` (in `executor_system.py`) must also trigger a full column rebuild:

```python
# After rebuilding secondary indices:
store.columns.rebuild_from(store.node_data, store._next_slot)
```

This ensures columns stay consistent with dicts after any manual rebuild.

### Executor read integration

**New filter path:** `_try_column_filter(expr, base_mask) -> np.ndarray | None`

For simple conditions (`score > 50`, `name = "main"`):
1. Check if column exists for the field
2. Build numpy mask via `ColumnStore.get_mask(field, op, value, n)`
3. AND with presence mask
4. AND with base_mask (live slots + kind filter)
5. Return combined mask

For compound expressions:
- `AndExpr` → compose masks with `&`
- `OrExpr` → compose with `|`
- `NotExpr` → `~mask & presence` (absent values should NOT match NOT)

Returns `None` if any sub-expression references a non-columnarized field, causing fallback to dict predicate.

**NULL handling:**
- `WHERE field = NULL` → returns `~presence[field][:n]` (slots where field is absent)
- `WHERE field != NULL` → returns `presence[field][:n]` (slots where field is present)
- All other operators with `value=None` fall back to dict predicate.

**Updated fallback chain in `_nodes()`:**
1. Secondary index → O(1) equality (indexed field)
2. Column filter → numpy vectorized mask (columnarized field)
3. Late materialization predicate → Python predicate on raw dict
4. Full dict filter → DegreeCondition and other special cases

Each level is tried in order; first non-None result wins.

**COUNT via columns:**
`COUNT NODES WHERE score > 50` becomes `np.count_nonzero(mask)`. Zero dict construction, zero Python iteration.

**DELETE NODES via columns:**
`_delete_nodes` uses `_try_column_filter` to build the deletion mask, then materializes only the IDs of matching slots (via `_slot_to_id`). Avoids building full dicts just to extract IDs for deletion.

**ORDER BY via columns:**
`NODES ORDER BY score DESC LIMIT 10` uses `np.argpartition` to find top-K slot indices in O(n), then sorts the K results for correct ordering: O(n + K log K) total.

### String interning for columns

Strings are stored as `int32` IDs via the existing `StringTable`. Filter support:

| Filter | Column-accelerated? | Mechanism |
|--------|---------------------|-----------|
| `name = "main"` | Yes | `column == intern("main")` |
| `name != "main"` | Yes | `column != intern("main")` |
| `name IN ("a", "b")` | Yes | `np.isin(column, [intern("a"), intern("b")])` |
| `name CONTAINS "main"` | No, fallback to dict | Substring needs actual string |
| `name LIKE "main%"` | No, fallback to dict | Pattern needs actual string |

### Persistence

**Serialization:** Add column data to existing SQLite checkpoint:

```
columns:{field}:data     → np.ndarray.tobytes()
columns:{field}:presence → np.ndarray.tobytes()
columns:{field}:dtype    → string ("int64", "float64", "int32_interned")
```

**Deserialization:** Reconstruct arrays from blobs using the dtype string stored alongside each blob (never assume a hardcoded dtype - there is a known dtype mismatch between CoreStore and the deserializer for `node_kinds` that should not be repeated). If checkpoint has no column data (old format), ColumnStore starts empty and auto-infers on next write. Backward compatible.

**WAL:** Column mutations are covered by the existing WAL replay mechanism since they're derived from the same `put_node`/`update_node` calls that are already logged.

## Data flow

### Write: `put_node("x", "fn", {"line": 42, "name": "main"})`

```
Executor._create_node
  -> CoreStore.put_node
       -> node_data[slot] = {"line": 42, "name": "main"}       # existing
       -> columns.set(slot, {"line": 42, "name": "main"})      # NEW
            -> columns["line"][slot] = 42                       # int64
            -> presence["line"][slot] = True
            -> columns["name"][slot] = intern("main")           # int32
            -> presence["name"][slot] = True
```

### Read: `NODES WHERE kind = "fn" AND score > 50 LIMIT 10`

```
Executor._nodes
  -> _extract_kind_from_where -> "fn"
  -> _strip_kind_from_expr -> Condition(score, >, 50)
  -> _try_column_filter(Condition(score, >, 50), live_mask)
       -> columns.get_mask("score", ">", 50, n)                # numpy: ~5us
       -> mask &= presence["score"]
       -> mask &= live_mask (kind="fn")
       -> return mask                                           # ~20us total
  -> materialize only matching slots (LIMIT 10)                 # 10 dicts built
```

### Fallback: `NODES WHERE description CONTAINS "auth"`

```
Executor._nodes
  -> _try_column_filter -> None (CONTAINS not supported on columns)
  -> _make_raw_predicate -> callable
  -> store.get_all_nodes(predicate=raw_pred)                    # late materialization
```

## File changes

| File | Change |
|------|--------|
| `graphstore/columns.py` | **NEW** - ColumnStore class |
| `graphstore/store.py` | Integrate ColumnStore into write/delete/grow paths |
| `graphstore/dsl/executor.py` | Add column filter path to `_nodes`, `_count`, `_delete_nodes` |
| `graphstore/schema.py` | Optional type annotations on field declarations |
| `graphstore/dsl/grammar.lark` | Type syntax in REGISTER (`name:string`) |
| `graphstore/dsl/transformer.py` | Parse type annotations |
| `graphstore/persistence/serializer.py` | Serialize column arrays |
| `graphstore/persistence/deserializer.py` | Deserialize column arrays |
| `graphstore/dsl/ast_nodes.py` | Update `SysRegisterNodeKind` to carry typed fields |
| `graphstore/dsl/executor_system.py` | Add column rebuild to `SYS REBUILD INDICES` |
| `graphstore/__init__.py` | Expose column stats in memory estimation; no API changes |
| `tests/test_columns.py` | **NEW** - ColumnStore unit tests |
| `tests/test_column_integration.py` | **NEW** - End-to-end DSL + column tests |

## Expected performance

At 50K nodes, 5 columnarized fields:

| Query | Current | With columns | Speedup |
|-------|---------|-------------|---------|
| `WHERE score > 4000` | ~15ms | ~50us | 300x |
| `COUNT WHERE score > 4000` | ~15ms | ~10us | 1500x |
| `WHERE kind = "fn" AND score > 4000` | ~5ms | ~20us | 250x |
| `WHERE name = "main"` (interned) | ~15ms | ~30us | 500x |
| `ORDER BY score DESC LIMIT 10` | ~15ms + sort | ~200us | 75x |
| `WHERE desc CONTAINS "auth"` | ~15ms | ~15ms (fallback) | 1x |

Baseline timings assume the current late-materialization path (Python predicate over raw dicts). Actual speedup varies based on predicate selectivity and data distribution.

Memory overhead: ~2.4MB additional (5 columns x 50K slots x ~10 bytes average). String IDs are never reclaimed from the StringTable; at target scale this is negligible, but at 500K+ nodes with high-cardinality string fields, StringTable memory should be monitored.

## Evolution path

1. **Now:** Hybrid layer - columns accelerate reads, dicts remain source of truth
2. **When dict memory becomes a problem (500K+ nodes):** Drop `node_data` dicts. Reconstruct on demand from columns + overflow dict for rare/untyped fields. The column infrastructure built now supports this transition.
3. **If analytics becomes primary use case:** Consider replacing ColumnStore internals with Arrow arrays for zero-copy serialization and cross-process sharing. The ColumnStore interface stays the same.

## Non-goals

- Columnar storage for edge properties (edges use CSR matrices, different optimization path)
- Query planning / cost-based optimizer (premature at current scale)
- Concurrent/parallel query execution
- Compression (Arrow/Parquet-style encoding)
- Column compaction (reclaiming space from tombstoned slots) - arrays remain at high-water-mark size, consistent with existing `node_ids`/`node_kinds` behavior
