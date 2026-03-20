# Agentic Brain DB Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform graphstore from a dict-based graph DB into a columnar-first agent memory substrate with aggregations, belief ops, spreading activation, and hypothesis testing.

**Architecture:** Flip source of truth from `list[dict]` to numpy ColumnStore (14x memory, 216x COUNT). Add `live_mask` for unified visibility filtering (tombstone + TTL + retracted). Extend DSL with 15+ new query forms across 4 phases.

**Tech Stack:** numpy (columnar arrays, sparse matmul), scipy (CSR matrices), lark (LALR grammar), sqlite3 (persistence)

**Spec:** `docs/superpowers/specs/2026-03-20-agentic-brain-db-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `graphstore/columns.py` | Modify | Add `set_reserved()`, `set_field()`, `_ensure_column()`, update `rebuild_from()` |
| `graphstore/store.py` | **Rewrite** | Remove `node_data`, all CRUD uses columns only, new `_materialize_slot()` |
| `graphstore/errors.py` | Modify | Add `AggregationError`, `SnapshotNotFound`, `ContextError` |
| `graphstore/dsl/grammar.lark` | Modify | All new grammar rules (AGGREGATE, ASSERT, RETRACT, RECALL, etc.) |
| `graphstore/dsl/ast_nodes.py` | Modify | All new AST dataclasses |
| `graphstore/dsl/transformer.py` | Modify | New transformer methods + time expression resolution |
| `graphstore/dsl/executor.py` | Modify | `_compute_live_mask()`, `_aggregate()`, `_recall()`, `_merge()`, `_assert()`, `_retract()`, `_update_nodes()`, `_counterfactual()`, `_bind_context()`, `_propagate()` |
| `graphstore/dsl/executor_system.py` | Modify | `_expire()`, `_contradictions()`, `_snapshot()`, `_rollback()`, `_snapshots()` |
| `graphstore/persistence/serializer.py` | Modify | Remove node_data JSON blob, columns-only serialization |
| `graphstore/persistence/deserializer.py` | Modify | Migration from old format, columns-only loading |
| `graphstore/__init__.py` | Modify | Route new query types, export new errors |
| `tests/test_columnar_store.py` | **Create** | Phase 1: columnar source-of-truth tests |
| `tests/test_aggregate.py` | **Create** | Phase 2: aggregation tests |
| `tests/test_beliefs.py` | **Create** | Phase 3: ASSERT/RETRACT/CONTRADICTIONS/TTL/MERGE tests |
| `tests/test_recall.py` | **Create** | Phase 4: RECALL/PROPAGATE/COUNTERFACTUAL/SNAPSHOT/CONTEXT tests |

---

## Phase 1: Infrastructure

### Task 1: ColumnStore Enhancements

**Files:**
- Modify: `graphstore/columns.py:16-217`
- Test: `tests/test_columns.py`

- [ ] **Step 1: Write tests for set_reserved and set_field**

```python
# Add to tests/test_columns.py
def test_set_reserved_int64(col_store):
    col_store.set_reserved(0, "__created_at__", 1710000000000)
    assert col_store._columns["__created_at__"][0] == 1710000000000
    assert col_store._presence["__created_at__"][0] == True

def test_set_reserved_auto_interns_string(col_store):
    col_store.set_reserved(0, "__source__", "web_search")
    assert col_store._dtypes["__source__"] == "int32_interned"
    assert col_store._presence["__source__"][0] == True
    str_id = col_store._columns["__source__"][0]
    assert col_store._string_table.lookup(int(str_id)) == "web_search"

def test_set_reserved_float64(col_store):
    col_store.set_reserved(0, "__confidence__", 0.95)
    assert col_store._dtypes["__confidence__"] == "float64"
    assert abs(col_store._columns["__confidence__"][0] - 0.95) < 1e-10

def test_set_field_alias(col_store):
    col_store.set_field(0, "score", 42)
    assert col_store._columns["score"][0] == 42

def test_ensure_column_creates_on_first_call(col_store):
    col_store._ensure_column("new_field", "float64")
    assert "new_field" in col_store._columns
    assert col_store._dtypes["new_field"] == "float64"

def test_ensure_column_noop_on_existing(col_store):
    col_store._ensure_column("x", "int64")
    col_store._columns["x"][0] = 42
    col_store._ensure_column("x", "int64")  # should not reset
    assert col_store._columns["x"][0] == 42
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_columns.py -v -k "set_reserved or set_field or ensure_column"`
Expected: FAIL (methods not defined)

- [ ] **Step 3: Implement set_reserved, set_field, _ensure_column**

In `graphstore/columns.py`, add after `declare_column()` (line 171):

```python
def _ensure_column(self, field: str, dtype_str: str) -> None:
    """Create column if it doesn't exist. No-op if it already exists."""
    if field not in self._dtypes:
        self._create_column(field, dtype_str)

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
    """Set a user field value. Same as set() for a single field."""
    self.set({field: value} if not callable(getattr(self, '_ensure_column', None))
             else None)  # placeholder
    # Actually: delegate to set_reserved which handles type inference
    self.set_reserved(slot, field, value)
```

Wait - `set_field` should be simpler. It's just an alias for single-field writes:

```python
def set_field(self, slot: int, field: str, value) -> None:
    """Set a single field value at a slot. Auto-infers type."""
    self.set(slot, {field: value})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_columns.py -v -k "set_reserved or set_field or ensure_column"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add graphstore/columns.py tests/test_columns.py
git commit -m "feat(columns): add set_reserved, set_field, _ensure_column methods"
```

---

### Task 2: CoreStore._materialize_slot From Columns

**Files:**
- Modify: `graphstore/store.py:433-442`
- Test: `tests/test_columnar_store.py` (create)

- [ ] **Step 1: Write test for column-based materialization**

```python
# tests/test_columnar_store.py
from graphstore.store import CoreStore

def test_materialize_from_columns():
    store = CoreStore()
    slot = store.put_node("n1", "person", {"name": "Alice", "age": 30, "score": 4.5})
    result = store._materialize_slot(slot)
    assert result["id"] == "n1"
    assert result["kind"] == "person"
    assert result["name"] == "Alice"
    assert result["age"] == 30
    assert abs(result["score"] - 4.5) < 1e-10

def test_materialize_tombstoned_returns_none():
    store = CoreStore()
    slot = store.put_node("n1", "person", {"name": "Alice"})
    store.delete_node("n1")
    result = store._materialize_slot(slot)
    assert result is None
```

- [ ] **Step 2: Run test to verify it fails or passes with current dict-based impl**

Run: `uv run pytest tests/test_columnar_store.py -v`

The current `_materialize_slot` uses `node_data[slot]` which still works. Tests may pass but we need the new column-based version.

- [ ] **Step 3: Rewrite _materialize_slot to read from columns**

In `graphstore/store.py`, replace `_materialize_slot` (lines 433-442):

```python
def _materialize_slot(self, slot: int) -> dict | None:
    """Build a full node dict from columns at a slot index."""
    if slot in self.node_tombstones:
        return None
    str_id = int(self.node_ids[slot])
    if str_id == -1:
        return None
    d = {
        "id": self.string_table.lookup(str_id),
        "kind": self.string_table.lookup(int(self.node_kinds[slot])),
    }
    for field in self.columns._columns:
        if field.startswith("__") and field.endswith("__"):
            continue  # skip reserved columns in user-facing output
        if self.columns._presence[field][slot]:
            dtype = self.columns._dtypes[field]
            raw = self.columns._columns[field][slot]
            if dtype == "int32_interned":
                d[field] = self.string_table.lookup(int(raw))
            elif dtype == "float64":
                d[field] = float(raw)
            elif dtype == "int64":
                d[field] = int(raw)
    return d
```

- [ ] **Step 4: Run tests to verify**

Run: `uv run pytest tests/test_columnar_store.py tests/test_store.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add graphstore/store.py tests/test_columnar_store.py
git commit -m "feat(store): materialize node dicts from columns instead of node_data"
```

---

### Task 3: Remove node_data - Write Path

**Files:**
- Modify: `graphstore/store.py:97-169` (put_node, update_node, upsert_node)
- Test: `tests/test_columnar_store.py`

- [ ] **Step 1: Write tests for column-only writes**

```python
def test_put_node_stores_in_columns():
    store = CoreStore()
    store.put_node("n1", "person", {"name": "Alice", "age": 30})
    assert store.columns._presence["name"][0] == True
    assert store.columns._presence["age"][0] == True
    # Verify node_data is not used
    assert not hasattr(store, 'node_data') or store.node_data[0] is None

def test_update_node_updates_columns():
    store = CoreStore()
    store.put_node("n1", "person", {"name": "Alice", "age": 30})
    store.update_node("n1", {"age": 31})
    result = store.get_node("n1")
    assert result["age"] == 31
    assert result["name"] == "Alice"

def test_increment_field_column_only():
    store = CoreStore()
    store.put_node("n1", "person", {"name": "Alice", "score": 10})
    store.increment_field("n1", "score", 5)
    result = store.get_node("n1")
    assert result["score"] == 15
```

- [ ] **Step 2: Run tests to verify they pass with current impl**

Run: `uv run pytest tests/test_columnar_store.py -v`

- [ ] **Step 3: Rewrite put_node, update_node, increment_field to be column-only**

**put_node** (replace lines 97-125):
```python
def put_node(self, id: str, kind: str, data: dict) -> int:
    """Add a node. Returns slot index. Raises NodeExists if ID exists."""
    str_id = self.string_table.intern(id)
    if str_id in self.id_to_slot:
        slot = self.id_to_slot[str_id]
        if slot not in self.node_tombstones:
            raise NodeExists(id)

    raw_edge_count = sum(len(v) for v in self._edges_by_type.values())
    check_ceiling(self._count, raw_edge_count, 1, 0, self._ceiling_bytes)

    kind_id = self.string_table.intern(kind)
    slot = self._alloc_slot()

    self.node_ids[slot] = str_id
    self.node_kinds[slot] = kind_id
    self.columns.set(slot, data)
    self.id_to_slot[str_id] = slot
    self._count += 1

    # Secondary indices
    for field in self._indexed_fields:
        if field in data:
            self.secondary_indices[field].setdefault(data[field], []).append(slot)

    return slot
```

**update_node** (replace lines 142-168):
```python
def update_node(self, id: str, data: dict):
    """Update node data. Raises NodeNotFound if missing."""
    if id not in self.string_table:
        raise NodeNotFound(id)
    str_id = self.string_table.intern(id)
    slot = self.id_to_slot.get(str_id)
    if slot is None or slot in self.node_tombstones:
        raise NodeNotFound(id)

    # Remove old values from indices
    for field in self._indexed_fields:
        if self.columns.has_column(field) and self.columns._presence[field][slot]:
            dtype = self.columns._dtypes[field]
            raw = self.columns._columns[field][slot]
            if dtype == "int32_interned":
                old_val = self.string_table.lookup(int(raw))
            elif dtype == "float64":
                old_val = float(raw)
            else:
                old_val = int(raw)
            idx_list = self.secondary_indices[field].get(old_val, [])
            if slot in idx_list:
                idx_list.remove(slot)

    # Update columns
    self.columns.set(slot, data)

    # Add new values to indices
    for field in self._indexed_fields:
        if field in data:
            self.secondary_indices[field].setdefault(data[field], []).append(slot)
```

**increment_field** (replace lines 514-530):
```python
def increment_field(self, id: str, field: str, amount: int | float):
    """Increment a numeric field. Raises NodeNotFound or TypeError."""
    if id not in self.string_table:
        raise NodeNotFound(id)
    str_id = self.string_table.intern(id)
    slot = self.id_to_slot.get(str_id)
    if slot is None or slot in self.node_tombstones:
        raise NodeNotFound(id)

    if not self.columns.has_column(field) or not self.columns._presence[field][slot]:
        current = 0
    else:
        dtype = self.columns._dtypes[field]
        if dtype not in ("int64", "float64"):
            raise TypeError(f"Field '{field}' is not numeric")
        current = float(self.columns._columns[field][slot]) if dtype == "float64" else int(self.columns._columns[field][slot])

    new_val = current + amount
    self.columns.set_field(slot, field, new_val)
```

- [ ] **Step 4: Remove node_data from _grow, delete_node, add_index, and __init__**

In `__init__` (line 33): Remove `self.node_data: list[dict | None] = [None] * self._capacity`

In `_grow` (line 74): Remove `self.node_data.extend([None] * self._capacity)`

In `delete_node` (line 200): Remove `self.node_data[slot] = None`. Keep tombstone + cascade. Update index removal to read from columns (same pattern as update_node above).

In `add_index` (lines 358-367): Rebuild from columns instead of node_data.

In `get_node` (lines 127-140): Replace `**self.node_data[slot]` with `_materialize_slot(slot)`.

In `get_all_nodes` (lines 444-471): Replace `data = self.node_data[slot]` with column-based materialization.

In `count_nodes` (lines 473-490): Replace `self.node_data[int(slot)]` with column-based predicate.

In `query_node_ids` (lines 492-510): Replace `self.node_data[slot]` with column-based predicate.

- [ ] **Step 5: Run full test suite to verify**

Run: `uv run pytest tests/ -v --tb=short`
Expected: Majority PASS. Fix any failures from node_data removal.

- [ ] **Step 6: Commit**

```bash
git add graphstore/store.py tests/test_columnar_store.py
git commit -m "refactor(store): remove node_data, columns are source of truth"
```

---

### Task 4: Auto-timestamps

**Files:**
- Modify: `graphstore/store.py` (put_node, update_node, upsert_node, increment_field)
- Test: `tests/test_columnar_store.py`

- [ ] **Step 1: Write tests**

```python
import time

def test_put_node_sets_created_at():
    store = CoreStore()
    before = int(time.time() * 1000)
    store.put_node("n1", "person", {"name": "Alice"})
    after = int(time.time() * 1000)
    result = store.columns.get_column("__created_at__", 1)
    assert result is not None
    ts = int(result[0][0])
    assert before <= ts <= after

def test_update_node_sets_updated_at():
    store = CoreStore()
    store.put_node("n1", "person", {"name": "Alice"})
    time.sleep(0.01)
    store.update_node("n1", {"name": "Bob"})
    created = int(store.columns._columns["__created_at__"][0])
    updated = int(store.columns._columns["__updated_at__"][0])
    assert updated >= created

def test_timestamps_not_in_materialized_output():
    """Reserved columns (__xx__) should not appear in user-facing dicts."""
    store = CoreStore()
    store.put_node("n1", "person", {"name": "Alice"})
    result = store._materialize_slot(0)
    assert "__created_at__" not in result
    assert "__updated_at__" not in result
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_columnar_store.py -v -k "timestamp"`
Expected: FAIL

- [ ] **Step 3: Add timestamp injection to put_node and update_node**

At the end of `put_node`, after `self._count += 1`:
```python
now_ms = int(time.time() * 1000)
self.columns.set_reserved(slot, "__created_at__", now_ms)
self.columns.set_reserved(slot, "__updated_at__", now_ms)
```

At the end of `update_node`, after index update:
```python
self.columns.set_reserved(slot, "__updated_at__", int(time.time() * 1000))
```

At the end of `increment_field`:
```python
self.columns.set_reserved(slot, "__updated_at__", int(time.time() * 1000))
```

Add `import time` to store.py imports.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_columnar_store.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add graphstore/store.py tests/test_columnar_store.py
git commit -m "feat(store): auto-inject __created_at__ and __updated_at__ timestamps"
```

---

### Task 5: Relative Time Grammar + Transformer

**Files:**
- Modify: `graphstore/dsl/grammar.lark:95-97`
- Modify: `graphstore/dsl/transformer.py`
- Test: `tests/test_dsl_parser.py`

- [ ] **Step 1: Write parser tests**

```python
def test_parse_now():
    ast = parse('NODES WHERE __created_at__ > NOW()')
    assert ast.where is not None

def test_parse_now_minus_duration():
    ast = parse('NODES WHERE __created_at__ > NOW() - 7d')
    # Should resolve to integer literal
    cond = ast.where.expr
    assert isinstance(cond.value, (int, float))

def test_parse_today():
    ast = parse('NODES WHERE __created_at__ > TODAY')
    cond = ast.where.expr
    assert isinstance(cond.value, (int, float))

def test_parse_yesterday():
    ast = parse('NODES WHERE __updated_at__ > YESTERDAY')
    cond = ast.where.expr
    assert isinstance(cond.value, (int, float))
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_dsl_parser.py -v -k "now or today or yesterday"`
Expected: FAIL (grammar doesn't recognize NOW/TODAY/YESTERDAY)

- [ ] **Step 3: Add grammar rules**

In `grammar.lark`, update the `value` rule (line 95-97):
```lark
?value: STRING -> val_string
      | NUMBER -> val_number
      | "NULL" -> val_null
      | time_expr

time_expr: "NOW" "()" "-" NUMBER TIME_UNIT -> time_offset
         | "NOW" "()"                       -> time_now
         | "TODAY"                           -> time_today
         | "YESTERDAY"                       -> time_yesterday

TIME_UNIT: /[smhd]/
```

- [ ] **Step 4: Add transformer methods**

In `transformer.py`, add:
```python
import time as _time
from datetime import datetime, timedelta

def time_now(self, _items):
    return int(_time.time() * 1000)

def time_offset(self, items):
    n = int(float(str(items[0])))
    unit = str(items[1])
    ms = {"s": 1000, "m": 60000, "h": 3600000, "d": 86400000}[unit]
    return int(_time.time() * 1000) - n * ms

def time_today(self, _items):
    midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    return int(midnight.timestamp() * 1000)

def time_yesterday(self, _items):
    midnight = (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return int(midnight.timestamp() * 1000)
```

Note: time expressions containing `NOW()` must bypass the LRU parser cache. In `parser.py`, check if query contains "NOW()" or "TODAY" or "YESTERDAY" and skip the cache.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_dsl_parser.py -v -k "now or today or yesterday"`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add graphstore/dsl/grammar.lark graphstore/dsl/transformer.py graphstore/dsl/parser.py tests/test_dsl_parser.py
git commit -m "feat(dsl): add NOW(), TODAY, YESTERDAY, NOW() - Nd relative time"
```

---

### Task 6: live_mask in Executor

**Files:**
- Modify: `graphstore/dsl/executor.py`
- Test: `tests/test_columnar_store.py`

- [ ] **Step 1: Write test for live_mask with TTL and retraction**

```python
from graphstore import GraphStore

def test_live_mask_excludes_expired():
    g = GraphStore(ceiling_mb=256)
    g.execute('CREATE NODE "n1" kind = "working" name = "temp"')
    # Manually set expires_at in the past
    g._store.columns.set_reserved(0, "__expires_at__", 1)  # 1ms = long expired
    result = g.execute('NODES')
    assert len(result.data) == 0  # expired node invisible

def test_live_mask_excludes_retracted():
    g = GraphStore(ceiling_mb=256)
    g.execute('CREATE NODE "n1" kind = "fact" name = "test"')
    g._store.columns.set_reserved(0, "__retracted__", 1)
    result = g.execute('NODES')
    assert len(result.data) == 0  # retracted node invisible

def test_live_mask_includes_non_expired():
    g = GraphStore(ceiling_mb=256)
    g.execute('CREATE NODE "n1" kind = "fact" name = "test"')
    # expires_at = far future
    g._store.columns.set_reserved(0, "__expires_at__", int(time.time() * 1000) + 86400000)
    result = g.execute('NODES')
    assert len(result.data) == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_columnar_store.py -v -k "live_mask"`
Expected: FAIL (live_mask not integrated)

- [ ] **Step 3: Implement _compute_live_mask in executor.py**

Add method to `Executor` class:
```python
def _compute_live_mask(self, n: int) -> np.ndarray:
    """Unified visibility filter: tombstones + TTL + retracted."""
    mask = self.store.node_ids[:n] >= 0
    if self.store.node_tombstones:
        tomb_arr = np.array(list(self.store.node_tombstones), dtype=np.int32)
        tomb_mask = np.zeros(n, dtype=bool)
        tomb_mask[tomb_arr[tomb_arr < n]] = True
        mask = mask & ~tomb_mask
    # TTL expiry
    expires = self.store.columns.get_column("__expires_at__", n)
    if expires is not None:
        col, pres, _ = expires
        now_ms = int(time.time() * 1000)
        mask = mask & ~(pres & (col > 0) & (col < now_ms))
    # Retracted
    retracted = self.store.columns.get_column("__retracted__", n)
    if retracted is not None:
        col, pres, _ = retracted
        mask = mask & ~(pres & (col == 1))
    return mask
```

- [ ] **Step 4: Integrate live_mask into _nodes, _count, _delete_nodes**

Replace `self.store._live_mask(kind)` calls in the executor with `self._compute_live_mask(n)` combined with kind filter. Update `_try_column_nodes`, `_try_column_count`, `_try_column_delete_ids` to accept a `base_mask` parameter that starts from `live_mask`.

This is a targeted refactor of the executor's existing filtering pipeline - the key change is that the base mask now includes TTL + retracted in addition to tombstones.

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -v --tb=short`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add graphstore/dsl/executor.py tests/test_columnar_store.py
git commit -m "feat(executor): add _compute_live_mask with TTL + retracted filtering"
```

---

### Task 7: Persistence Migration

**Files:**
- Modify: `graphstore/persistence/serializer.py`
- Modify: `graphstore/persistence/deserializer.py`
- Test: `tests/test_persistence.py`

- [ ] **Step 1: Write migration test**

```python
def test_load_old_format_with_node_data_migrates(tmp_path):
    """Databases saved with node_data JSON blob should still load."""
    # Create a database in old format manually
    import sqlite3, json
    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(str(db_path))
    # ... create tables, insert old-format blobs with node_data key ...
    conn.close()

    g = GraphStore(path=str(tmp_path))
    assert g.node_count > 0

def test_roundtrip_columnar_only(tmp_path):
    """New format should serialize/deserialize without node_data."""
    g = GraphStore(path=str(tmp_path))
    g.execute('CREATE NODE "n1" kind = "test" name = "Alice" score = 42')
    g.checkpoint()
    g.close()

    g2 = GraphStore(path=str(tmp_path))
    result = g2.execute('NODE "n1"')
    assert result.data["name"] == "Alice"
    assert result.data["score"] == 42
    g2.close()
```

- [ ] **Step 2: Run to verify state**

Run: `uv run pytest tests/test_persistence.py -v`

- [ ] **Step 3: Update serializer - remove node_data blob**

In `serializer.py`, remove the `node_data` JSON blob write (lines 33-36). Column blobs already handle all field data.

- [ ] **Step 4: Update deserializer - handle migration**

In `deserializer.py`, add migration path: if `node_data` blob exists, load it, write fields into columns, then proceed as normal. If no `node_data` blob, columns are already the source of truth.

- [ ] **Step 5: Run persistence tests**

Run: `uv run pytest tests/test_persistence.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add graphstore/persistence/serializer.py graphstore/persistence/deserializer.py tests/test_persistence.py
git commit -m "feat(persistence): columnar-only serialization with old-format migration"
```

---

### Task 8: Batch Rollback with Column Snapshots

**Files:**
- Modify: `graphstore/dsl/executor.py` (_batch method)
- Test: `tests/test_dsl_user_writes.py`

- [ ] **Step 1: Write test**

```python
def test_batch_rollback_restores_columns():
    g = GraphStore(ceiling_mb=256)
    g.execute('CREATE NODE "n1" kind = "test" score = 10')
    try:
        g.execute('''BEGIN
$x = CREATE NODE "n2" kind = "test" score = 20
CREATE EDGE $x -> "nonexistent"
COMMIT''')
    except Exception:
        pass
    # n2 should not exist (rolled back)
    result = g.execute('NODE "n2"')
    assert result.data is None
    # n1 should be unchanged
    result = g.execute('NODE "n1"')
    assert result.data["score"] == 10
```

- [ ] **Step 2: Update _batch to save/restore column arrays instead of dicts**

Replace the dict-based state save/restore in `_batch()` with numpy array copies:

```python
# Save state
saved_columns = {}
for f in self.store.columns._columns:
    saved_columns[f] = (
        self.store.columns._columns[f][:ns].copy(),
        self.store.columns._presence[f][:ns].copy(),
    )
saved_dtypes = dict(self.store.columns._dtypes)
# ... (keep existing edge/tombstone saves)

# Restore on rollback
for f, (col_snap, pres_snap) in saved_columns.items():
    self.store.columns._columns[f][:ns] = col_snap
    self.store.columns._presence[f][:ns] = pres_snap
self.store.columns._dtypes = saved_dtypes
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_dsl_user_writes.py -v -k "batch"`
Expected: PASS

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest tests/ -v --tb=short`
Expected: ALL PASS (Phase 1 complete)

- [ ] **Step 5: Commit**

```bash
git add graphstore/dsl/executor.py tests/test_dsl_user_writes.py
git commit -m "perf(batch): rollback saves numpy arrays instead of dicts (324x faster)"
```

---

## Phase 2: Aggregations

### Task 9: AGGREGATE Grammar + AST + Transformer

**Files:**
- Modify: `graphstore/dsl/grammar.lark`
- Modify: `graphstore/dsl/ast_nodes.py`
- Modify: `graphstore/dsl/transformer.py`
- Test: `tests/test_dsl_parser.py`

- [ ] **Step 1: Write parser tests**

```python
def test_parse_aggregate_group_by():
    ast = parse('AGGREGATE NODES WHERE kind = "memory" GROUP BY topic SELECT COUNT(), AVG(importance)')
    assert isinstance(ast, AggregateQuery)
    assert len(ast.group_by) == 1
    assert ast.group_by[0] == "topic"
    assert len(ast.select) == 2

def test_parse_aggregate_with_having():
    ast = parse('AGGREGATE NODES GROUP BY kind SELECT COUNT() HAVING COUNT() > 5')
    assert ast.having is not None

def test_parse_aggregate_no_group_by():
    ast = parse('AGGREGATE NODES SELECT COUNT(), SUM(score)')
    assert ast.group_by == []

def test_parse_aggregate_with_order_limit():
    ast = parse('AGGREGATE NODES GROUP BY kind SELECT COUNT() ORDER BY COUNT() DESC LIMIT 10')
    assert ast.order_by is not None
    assert ast.limit is not None
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_dsl_parser.py -v -k "aggregate"`
Expected: FAIL

- [ ] **Step 3: Add grammar rules**

In `grammar.lark`, add to `read_query` alternatives:
```lark
| aggregate_q

aggregate_q: "AGGREGATE" "NODES" where_clause? group_clause? select_clause having_clause? order_agg_clause? limit_clause?
group_clause: "GROUP" "BY" IDENTIFIER ("," IDENTIFIER)*
select_clause: "SELECT" agg_func ("," agg_func)*
having_clause: "HAVING" condition
order_agg_clause: "ORDER" "BY" agg_func order_dir?

agg_func: "COUNT" "()"                             -> agg_count
        | "COUNT" "DISTINCT" "(" IDENTIFIER ")"     -> agg_count_distinct
        | "SUM" "(" IDENTIFIER ")"                   -> agg_sum
        | "AVG" "(" IDENTIFIER ")"                   -> agg_avg
        | "MIN" "(" IDENTIFIER ")"                   -> agg_min
        | "MAX" "(" IDENTIFIER ")"                   -> agg_max
```

- [ ] **Step 4: Add AST nodes**

In `ast_nodes.py`:
```python
@dataclass
class AggFunc:
    func: str       # "COUNT", "COUNT_DISTINCT", "SUM", "AVG", "MIN", "MAX"
    field: str | None  # None for COUNT()

@dataclass
class AggregateQuery:
    where: WhereClause | None = None
    group_by: list[str] = None
    select: list[AggFunc] = None
    having: Any | None = None
    order_by: AggFunc | None = None
    order_desc: bool = False
    limit: LimitClause | None = None
```

- [ ] **Step 5: Add transformer methods**

```python
def aggregate_q(self, items):
    where = group_by = select = having = order_by = limit = None
    for item in items:
        if isinstance(item, WhereClause): where = item
        elif isinstance(item, list) and item and isinstance(item[0], str): group_by = item
        elif isinstance(item, list) and item and isinstance(item[0], AggFunc): select = item
        # ... dispatch by type
    return AggregateQuery(where=where, group_by=group_by or [], select=select or [], ...)

def group_clause(self, items):
    return [str(i) for i in items]

def select_clause(self, items):
    return list(items)

def agg_count(self, _): return AggFunc("COUNT", None)
def agg_count_distinct(self, items): return AggFunc("COUNT_DISTINCT", str(items[0]))
def agg_sum(self, items): return AggFunc("SUM", str(items[0]))
def agg_avg(self, items): return AggFunc("AVG", str(items[0]))
def agg_min(self, items): return AggFunc("MIN", str(items[0]))
def agg_max(self, items): return AggFunc("MAX", str(items[0]))
```

- [ ] **Step 6: Run parser tests**

Run: `uv run pytest tests/test_dsl_parser.py -v -k "aggregate"`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add graphstore/dsl/grammar.lark graphstore/dsl/ast_nodes.py graphstore/dsl/transformer.py tests/test_dsl_parser.py
git commit -m "feat(dsl): add AGGREGATE grammar, AST nodes, and transformer"
```

---

### Task 10: Executor _aggregate

**Files:**
- Modify: `graphstore/dsl/executor.py`
- Modify: `graphstore/errors.py`
- Modify: `graphstore/__init__.py`
- Test: `tests/test_aggregate.py` (create)

- [ ] **Step 1: Write integration tests**

```python
# tests/test_aggregate.py
from graphstore import GraphStore

def setup_graph():
    g = GraphStore(ceiling_mb=256)
    g.execute('SYS REGISTER NODE KIND "memory" REQUIRED topic:string, importance:float')
    for i in range(100):
        topic = f"topic_{i % 5}"
        g.execute(f'CREATE NODE "m{i}" kind = "memory" topic = "{topic}" importance = {(i % 10) * 0.1}')
    return g

def test_aggregate_count_by_topic():
    g = setup_graph()
    result = g.execute('AGGREGATE NODES WHERE kind = "memory" GROUP BY topic SELECT COUNT()')
    assert result.kind == "aggregate"
    assert result.count == 5  # 5 unique topics
    for row in result.data:
        assert row["COUNT()"] == 20  # 100/5

def test_aggregate_avg_importance():
    g = setup_graph()
    result = g.execute('AGGREGATE NODES WHERE kind = "memory" GROUP BY topic SELECT AVG(importance)')
    assert result.count == 5
    for row in result.data:
        assert "AVG(importance)" in row

def test_aggregate_global_no_group_by():
    g = setup_graph()
    result = g.execute('AGGREGATE NODES WHERE kind = "memory" SELECT COUNT(), SUM(importance)')
    assert result.count == 1
    assert result.data[0]["COUNT()"] == 100

def test_aggregate_having():
    g = setup_graph()
    result = g.execute('AGGREGATE NODES GROUP BY topic SELECT COUNT() HAVING COUNT() > 10')
    assert all(row["COUNT()"] > 10 for row in result.data)

def test_aggregate_non_columnarized_field_raises():
    g = GraphStore(ceiling_mb=256)
    g.execute('CREATE NODE "n1" kind = "test" name = "x"')
    # "unknown_field" is not columnarized
    try:
        g.execute('AGGREGATE NODES GROUP BY unknown_field SELECT COUNT()')
        assert False, "Should have raised"
    except Exception as e:
        assert "not columnarized" in str(e).lower() or "AggregationError" in type(e).__name__
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_aggregate.py -v`
Expected: FAIL

- [ ] **Step 3: Add AggregationError to errors.py**

```python
class AggregationError(SchemaError):
    """Aggregation requires columnarized fields."""
    pass
```

- [ ] **Step 4: Implement _aggregate in executor.py**

Follow the pseudocode in spec section 6.3. Key steps:
1. Compute `live_mask`
2. Apply WHERE filter via `_try_column_filter`
3. Validate all GROUP BY + aggregate fields are columnarized
4. Use `np.unique` + `np.bincount` + `np.add.at` for group computation
5. Build result dicts, apply HAVING, ORDER BY, LIMIT

- [ ] **Step 5: Wire into _dispatch and GraphStore.execute**

Add `AggregateQuery` to the dispatch table in executor.py. Add routing in `__init__.py`.

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_aggregate.py -v`
Expected: PASS

- [ ] **Step 7: Run full suite**

Run: `uv run pytest tests/ -v --tb=short`
Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add graphstore/dsl/executor.py graphstore/errors.py graphstore/__init__.py tests/test_aggregate.py
git commit -m "feat: AGGREGATE NODES with GROUP BY + SUM/AVG/MIN/MAX/COUNT numpy engine"
```

---

## Phase 3: Belief Operations

### Task 11: ASSERT + RETRACT

**Files:**
- Modify: `graphstore/dsl/grammar.lark`, `graphstore/dsl/ast_nodes.py`, `graphstore/dsl/transformer.py`, `graphstore/dsl/executor.py`
- Test: `tests/test_beliefs.py` (create)

- [ ] **Step 1: Write tests**

```python
def test_assert_creates_node_with_confidence():
    g = GraphStore(ceiling_mb=256)
    result = g.execute('ASSERT "fact:earth" value = 6371 kind = "fact" CONFIDENCE 0.99 SOURCE "physics"')
    node = g.execute('NODE "fact:earth"')
    assert node.data["value"] == 6371

def test_assert_upserts_existing():
    g = GraphStore(ceiling_mb=256)
    g.execute('ASSERT "fact:earth" value = 6371 kind = "fact" CONFIDENCE 0.99')
    g.execute('ASSERT "fact:earth" value = 6372 kind = "fact" CONFIDENCE 0.95')
    node = g.execute('NODE "fact:earth"')
    assert node.data["value"] == 6372

def test_retract_makes_node_invisible():
    g = GraphStore(ceiling_mb=256)
    g.execute('CREATE NODE "f1" kind = "fact" value = "old"')
    g.execute('RETRACT "f1" REASON "outdated"')
    result = g.execute('NODE "f1"')
    assert result.data is None
    result = g.execute('NODES')
    assert len(result.data) == 0
```

- [ ] **Step 2: Add grammar, AST, transformer, executor**

Grammar:
```lark
| assert_stmt | retract_stmt

assert_stmt: "ASSERT" STRING field_pairs confidence_clause? source_clause?
confidence_clause: "CONFIDENCE" NUMBER
source_clause: "SOURCE" STRING
retract_stmt: "RETRACT" STRING reason_clause?
reason_clause: "REASON" STRING
```

AST:
```python
@dataclass
class AssertStmt:
    id: str
    fields: list[FieldPair]
    confidence: float | None = None
    source: str | None = None

@dataclass
class RetractStmt:
    id: str
    reason: str | None = None
```

- [ ] **Step 3: Run tests**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: ASSERT with confidence/source, RETRACT with reason"
```

---

### Task 12: UPDATE NODES WHERE (Bulk)

**Files:**
- Modify: grammar, ast_nodes, transformer, executor
- Test: `tests/test_beliefs.py`

- [ ] **Step 1: Write tests**

```python
def test_update_nodes_where():
    g = GraphStore(ceiling_mb=256)
    g.execute('SYS REGISTER NODE KIND "fact" REQUIRED confidence:float')
    for i in range(10):
        g.execute(f'CREATE NODE "f{i}" kind = "fact" confidence = 0.9')
    result = g.execute('UPDATE NODES WHERE kind = "fact" SET confidence = 0.1')
    assert result.data["updated"] == 10
    for i in range(10):
        node = g.execute(f'NODE "f{i}"')
        assert abs(node.data["confidence"] - 0.1) < 1e-10
```

- [ ] **Step 2: Add grammar + implementation**

Grammar: `update_nodes: "UPDATE" "NODES" where_clause "SET" field_pairs`

Executor: filter via column mask, bulk-assign new values to matching slots.

- [ ] **Step 3: Run tests + commit**

```bash
git commit -m "feat: UPDATE NODES WHERE for bulk belief revision"
```

---

### Task 13: TTL (EXPIRES IN / EXPIRES AT + SYS EXPIRE)

**Files:**
- Modify: grammar, ast_nodes, transformer, executor, executor_system
- Test: `tests/test_beliefs.py`

- [ ] **Step 1: Write tests**

```python
def test_expires_in_makes_node_invisible_after_expiry():
    g = GraphStore(ceiling_mb=256)
    g.execute('CREATE NODE "tmp1" kind = "working" name = "scratch" EXPIRES IN 0s')
    import time; time.sleep(0.01)
    result = g.execute('NODES')
    assert len(result.data) == 0  # expired

def test_sys_expire_tombstones_expired():
    g = GraphStore(ceiling_mb=256)
    g.execute('CREATE NODE "tmp1" kind = "working" name = "scratch" EXPIRES IN 0s')
    import time; time.sleep(0.01)
    result = g.execute('SYS EXPIRE')
    assert result.data["expired"] == 1
    assert g.node_count == 0
```

- [ ] **Step 2: Add grammar + implementation**

Grammar additions to `create_node` and `upsert_node`:
```lark
create_node: "CREATE" "NODE" STRING field_pairs expires_clause?
           | "CREATE" "NODE" "AUTO" field_pairs expires_clause? -> create_node_auto
expires_clause: "EXPIRES" "IN" NUMBER TIME_UNIT -> expires_in
              | "EXPIRES" "AT" STRING -> expires_at
```

System grammar: `sys_expire: "EXPIRE" where_clause?`

- [ ] **Step 3: Run tests + commit**

```bash
git commit -m "feat: TTL with EXPIRES IN/AT clause and SYS EXPIRE"
```

---

### Task 14: SYS CONTRADICTIONS

**Files:**
- Modify: grammar, ast_nodes, transformer, executor_system
- Test: `tests/test_beliefs.py`

- [ ] **Step 1: Write tests**

```python
def test_contradictions_detects_conflicting_beliefs():
    g = GraphStore(ceiling_mb=256)
    g.execute('SYS REGISTER NODE KIND "belief" REQUIRED topic:string, value:string')
    g.execute('CREATE NODE "b1" kind = "belief" topic = "capital" value = "Paris"')
    g.execute('CREATE NODE "b2" kind = "belief" topic = "capital" value = "London"')
    g.execute('CREATE NODE "b3" kind = "belief" topic = "color" value = "blue"')
    result = g.execute('SYS CONTRADICTIONS WHERE kind = "belief" FIELD value GROUP BY topic')
    assert result.count == 1  # only "capital" has conflicting values
    assert result.data[0]["group"] == "capital"

def test_contradictions_returns_empty_when_consistent():
    g = GraphStore(ceiling_mb=256)
    g.execute('SYS REGISTER NODE KIND "belief" REQUIRED topic:string, value:string')
    g.execute('CREATE NODE "b1" kind = "belief" topic = "capital" value = "Paris"')
    g.execute('CREATE NODE "b2" kind = "belief" topic = "capital" value = "Paris"')
    result = g.execute('SYS CONTRADICTIONS WHERE kind = "belief" FIELD value GROUP BY topic')
    assert result.count == 0
```

- [ ] **Step 2: Implement + commit**

```bash
git commit -m "feat: SYS CONTRADICTIONS for belief consistency checking"
```

---

### Task 15: MERGE NODE

**Files:**
- Modify: grammar, ast_nodes, transformer, executor
- Test: `tests/test_beliefs.py`

- [ ] **Step 1: Write tests**

```python
def test_merge_node_copies_fields():
    g = GraphStore(ceiling_mb=256)
    g.execute('CREATE NODE "src" kind = "fact" name = "old" extra = "data"')
    g.execute('CREATE NODE "tgt" kind = "fact" name = "canonical"')
    result = g.execute('MERGE NODE "src" INTO "tgt"')
    assert result.data["fields_merged"] >= 1
    tgt = g.execute('NODE "tgt"')
    assert tgt.data["name"] == "canonical"  # target wins on conflict
    assert tgt.data["extra"] == "data"  # source field copied
    src = g.execute('NODE "src"')
    assert src.data is None  # source tombstoned

def test_merge_rewires_edges():
    g = GraphStore(ceiling_mb=256)
    g.execute('CREATE NODE "src" kind = "x"')
    g.execute('CREATE NODE "tgt" kind = "x"')
    g.execute('CREATE NODE "other" kind = "x"')
    g.execute('CREATE EDGE "src" -> "other" kind = "link"')
    g.execute('MERGE NODE "src" INTO "tgt"')
    edges = g.execute('EDGES FROM "tgt"')
    assert len(edges.data) == 1
    assert edges.data[0]["target"] == "other"
```

- [ ] **Step 2: Implement + commit**

Grammar: `merge_stmt: "MERGE" "NODE" STRING "INTO" STRING`

Follow spec section 7.7 for implementation.

```bash
git commit -m "feat: MERGE NODE for memory consolidation"
```

---

## Phase 4: Graph Intelligence

### Task 16: RECALL (Spreading Activation)

**Files:**
- Modify: grammar, ast_nodes, transformer, executor
- Test: `tests/test_recall.py` (create)

- [ ] **Step 1: Write tests**

```python
def test_recall_returns_connected_nodes():
    g = GraphStore(ceiling_mb=256)
    g.execute('CREATE NODE "cue" kind = "concept" name = "Paris"')
    g.execute('CREATE NODE "m1" kind = "memory" name = "Eiffel Tower" importance = 0.9')
    g.execute('CREATE NODE "m2" kind = "memory" name = "Louvre" importance = 0.8')
    g.execute('CREATE NODE "m3" kind = "memory" name = "Unrelated" importance = 0.5')
    g.execute('CREATE EDGE "cue" -> "m1" kind = "related"')
    g.execute('CREATE EDGE "cue" -> "m2" kind = "related"')
    result = g.execute('RECALL FROM "cue" DEPTH 1 LIMIT 10')
    ids = [n["id"] for n in result.data]
    assert "m1" in ids
    assert "m2" in ids
    assert "m3" not in ids  # not connected

def test_recall_with_where():
    g = GraphStore(ceiling_mb=256)
    g.execute('SYS REGISTER NODE KIND "memory" REQUIRED importance:float')
    g.execute('CREATE NODE "cue" kind = "concept"')
    g.execute('CREATE NODE "m1" kind = "memory" importance = 0.9')
    g.execute('CREATE NODE "m2" kind = "other" importance = 0.8')
    g.execute('CREATE EDGE "cue" -> "m1" kind = "r"')
    g.execute('CREATE EDGE "cue" -> "m2" kind = "r"')
    result = g.execute('RECALL FROM "cue" DEPTH 1 LIMIT 10 WHERE kind = "memory"')
    assert len(result.data) == 1
    assert result.data[0]["id"] == "m1"

def test_recall_scores_annotated():
    g = GraphStore(ceiling_mb=256)
    g.execute('CREATE NODE "cue" kind = "concept"')
    g.execute('CREATE NODE "m1" kind = "memory"')
    g.execute('CREATE EDGE "cue" -> "m1" kind = "r"')
    result = g.execute('RECALL FROM "cue" DEPTH 1 LIMIT 10')
    assert "_activation_score" in result.data[0]
```

- [ ] **Step 2: Add grammar**

```lark
recall_q: "RECALL" "FROM" STRING "DEPTH" NUMBER limit_clause? where_clause?
```

- [ ] **Step 3: Implement _recall using sparse matmul**

Follow spec section 8.1. Key: `csr.dot(activation)` per hop, weight by importance and recency, top-K via `np.argpartition`.

- [ ] **Step 4: Run tests + commit**

```bash
git commit -m "feat: RECALL spreading activation via sparse matrix-vector multiply"
```

---

### Task 17: PROPAGATE

**Files:**
- Modify: grammar, ast_nodes, transformer, executor
- Test: `tests/test_recall.py`

- [ ] **Step 1: Write tests**

```python
def test_propagate_updates_descendants():
    g = GraphStore(ceiling_mb=256)
    g.execute('SYS REGISTER NODE KIND "belief" REQUIRED confidence:float')
    g.execute('CREATE NODE "root" kind = "belief" confidence = 0.9')
    g.execute('CREATE NODE "child" kind = "belief" confidence = 0.5')
    g.execute('CREATE EDGE "root" -> "child" kind = "supports"')
    result = g.execute('PROPAGATE "root" FIELD confidence DEPTH 1')
    assert result.data["updated"] >= 1
```

- [ ] **Step 2: Implement + commit**

```bash
git commit -m "feat: PROPAGATE for forward belief chaining"
```

---

### Task 18: COUNTERFACTUAL (WHAT IF)

**Files:**
- Modify: grammar, ast_nodes, transformer, executor
- Test: `tests/test_recall.py`

- [ ] **Step 1: Write tests**

```python
def test_counterfactual_does_not_commit():
    g = GraphStore(ceiling_mb=256)
    g.execute('CREATE NODE "belief1" kind = "belief" value = "x"')
    g.execute('CREATE NODE "conclusion1" kind = "conclusion"')
    g.execute('CREATE EDGE "belief1" -> "conclusion1" kind = "supports"')
    result = g.execute('WHAT IF RETRACT "belief1"')
    assert result.data["affected_count"] >= 1
    # Original node should still exist
    node = g.execute('NODE "belief1"')
    assert node.data is not None
```

- [ ] **Step 2: Implement + commit**

Grammar: `counterfactual: "WHAT" "IF" "RETRACT" STRING`

Follow spec section 8.3 - full state snapshot, find descendants, restore.

```bash
git commit -m "feat: WHAT IF RETRACT counterfactual reasoning"
```

---

### Task 19: SYS SNAPSHOT / SYS ROLLBACK

**Files:**
- Modify: grammar, ast_nodes, transformer, executor_system
- Test: `tests/test_recall.py`

- [ ] **Step 1: Write tests**

```python
def test_snapshot_and_rollback():
    g = GraphStore(ceiling_mb=256)
    g.execute('CREATE NODE "n1" kind = "test" value = "original"')
    g.execute('SYS SNAPSHOT "before-change"')
    g.execute('UPDATE NODE "n1" SET value = "modified"')
    assert g.execute('NODE "n1"').data["value"] == "modified"
    g.execute('SYS ROLLBACK TO "before-change"')
    assert g.execute('NODE "n1"').data["value"] == "original"

def test_sys_snapshots_lists_all():
    g = GraphStore(ceiling_mb=256)
    g.execute('SYS SNAPSHOT "snap1"')
    g.execute('SYS SNAPSHOT "snap2"')
    result = g.execute('SYS SNAPSHOTS')
    assert len(result.data) == 2
```

- [ ] **Step 2: Implement + commit**

Grammar:
```lark
sys_snapshot: "SNAPSHOT" STRING
sys_rollback: "ROLLBACK" "TO" STRING
sys_snapshots: "SNAPSHOTS"
```

Follow spec section 8.4.

```bash
git commit -m "feat: SYS SNAPSHOT/ROLLBACK for hypothesis exploration"
```

---

### Task 20: BIND CONTEXT / DISCARD CONTEXT

**Files:**
- Modify: grammar, ast_nodes, transformer, executor
- Test: `tests/test_recall.py`

- [ ] **Step 1: Write tests**

```python
def test_bind_context_isolates_nodes():
    g = GraphStore(ceiling_mb=256)
    g.execute('CREATE NODE "global" kind = "fact" name = "always visible"')
    g.execute('BIND CONTEXT "session-1"')
    g.execute('CREATE NODE "local" kind = "hypothesis" name = "maybe"')
    # Only context nodes visible
    result = g.execute('NODES')
    assert len(result.data) == 1
    assert result.data[0]["id"] == "local"
    g.execute('DISCARD CONTEXT "session-1"')
    # Back to global view, local deleted
    result = g.execute('NODES')
    assert len(result.data) == 1
    assert result.data[0]["id"] == "global"
```

- [ ] **Step 2: Implement + commit**

Grammar:
```lark
bind_context: "BIND" "CONTEXT" STRING
discard_context: "DISCARD" "CONTEXT" STRING
```

Executor: session state `_active_context`, auto-tag on CREATE, filter in `_compute_live_mask`.

```bash
git commit -m "feat: BIND/DISCARD CONTEXT for isolated reasoning sessions"
```

---

### Task 21: Final Integration + Full Test Suite

**Files:**
- All test files
- `graphstore/__init__.py`

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v --tb=short`
Expected: ALL PASS

- [ ] **Step 2: Run benchmark to verify performance targets**

Run: `uv run python bench_flip.py`
Verify performance targets from spec section 14.

- [ ] **Step 3: Clean up bench_flip.py**

```bash
rm bench_flip.py
```

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete agentic brain DB - 4 phases implemented

Phase 1: Columnar source of truth, live_mask, auto-timestamps, relative time
Phase 2: AGGREGATE GROUP BY + SUM/AVG/MIN/MAX/COUNT DISTINCT + HAVING
Phase 3: ASSERT/RETRACT, UPDATE NODES WHERE, TTL, SYS CONTRADICTIONS, MERGE
Phase 4: RECALL, PROPAGATE, WHAT IF, SYS SNAPSHOT/ROLLBACK, BIND/DISCARD CONTEXT"
```
