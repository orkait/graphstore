# Agentic Brain DB - Design Spec

> Transform graphstore from a typed graph database into an agent memory substrate with superpowers no competitor offers.

**Date:** 2026-03-20
**Status:** Design approved, pending implementation plan

---

## 1. Problem

Graphstore has a solid graph engine (40+ DSL forms, sparse matrix traversal, schema enforcement, columnar acceleration) but lacks the primitives agents need to use it as their brain:

- No aggregation (GROUP BY, SUM, AVG) for memory summarization
- No temporal awareness (auto-timestamps, relative time, TTL)
- No belief management (assert/retract, contradiction detection)
- No spreading activation for associative recall
- No hypothesis testing (snapshot/rollback, counterfactual reasoning)
- No isolation primitives (contexts, namespaces)
- `list[dict]` source of truth wastes 14x memory vs columnar storage

## 2. Goal

Ship a complete agent memory substrate in 4 phases:

| Phase | What | Why |
|---|---|---|
| 1 - Infrastructure | Columnar source of truth, `live_mask`, auto-timestamps, relative time | Foundation everything else depends on |
| 2 - Aggregations | GROUP BY + SUM/AVG/MIN/MAX/COUNT DISTINCT + HAVING | Memory summarization at scale |
| 3 - Belief ops | ASSERT/RETRACT, UPDATE NODES WHERE, SYS CONTRADICTIONS, TTL, MERGE | Agent reasoning primitives |
| 4 - Graph intelligence | RECALL, PROPAGATE, COUNTERFACTUAL, SNAPSHOT/ROLLBACK, BIND CONTEXT | Superpowers no competitor offers |

## 3. Non-goals

- Vector embeddings / ANN search (separate future spec)
- Full-text inverted index (separate future spec)
- Multi-process concurrency (single-agent-per-process is fine)
- Distributed / clustered deployment

---

## 4. Architecture: Columnar Source of Truth

### 4.1 The Flip

Current architecture: `node_data: list[dict]` is source of truth, `ColumnStore` is acceleration cache.

New architecture: `ColumnStore` is source of truth. Dicts are materialized on demand for query results only.

**Evidence (measured benchmarks at 100k nodes):**

| Operation | dict | columnar | Delta |
|---|---|---|---|
| COUNT WHERE score > X | 7,324 μs | 34 μs | **216x faster** |
| GROUP BY + AVG | 11,698 μs | 744 μs | **16x faster** |
| Batch rollback copy | 24,300 μs | 75 μs | **324x faster** |
| Memory usage | 49.2 MB | 3.5 MB | **14x less** |
| Single write | 1 μs (dual) | 1 μs (single) | **12% faster** |
| Materialize LIMIT 10 | 2 μs | 6 μs | **+4 μs (fixed)** |
| Materialize LIMIT 1000 | 190 μs | 517 μs | **+327 μs** |
| CONTAINS full scan | 13,538 μs | 13,817 μs | **~same** |

The only regression is materialization (+4μs at LIMIT 10, invisible for agent use). Everything else is same speed or dramatically faster.

### 4.2 Storage Layout After Flip

```
CoreStore:
  node_ids:      np.ndarray[int32]       # interned string IDs by slot
  node_kinds:    np.ndarray[int32]       # interned kind IDs by slot
  node_tombstones: set[int]              # deleted slots

  columns: ColumnStore                   # ALL field data lives here
    _columns:   dict[str, np.ndarray]    # field -> typed numpy array
    _presence:  dict[str, np.ndarray]    # field -> bool mask (is field set?)
    _dtypes:    dict[str, str]           # field -> "int64" | "float64" | "int32_interned"

  # Removed: node_data: list[dict]       # no longer exists
```

### 4.3 Materialization (On Demand)

Result dicts are built from columns only when needed for query output:

```python
def _materialize_slot(self, slot: int) -> dict:
    d = {
        "id": self.string_table.lookup(int(self.node_ids[slot])),
        "kind": self.string_table.lookup(int(self.node_kinds[slot])),
    }
    for field in self.columns._columns:
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

Cost: ~0.5μs per node. At LIMIT 10 = 5μs. At LIMIT 100 = 50μs. Negligible for agent workloads.

### 4.4 Schema-Free Preservation

Fields auto-create columns on first write (existing `ColumnStore.set()` behavior). Type is inferred from the first value written:
- `int` or `bool` -> `int64`
- `float` -> `float64`
- `str` -> `int32_interned`

No schema declaration required. `SYS REGISTER NODE KIND` pre-creates columns for declared fields (existing behavior, unchanged).

### 4.5 Reserved Columns

System-managed columns prefixed with `__` (double underscore). Users cannot write to these directly.

| Column | Type | Set by |
|---|---|---|
| `__created_at__` | int64 | `put_node()` - Unix ms timestamp |
| `__updated_at__` | int64 | `put_node()`, `update_node()`, `upsert_node()` - Unix ms timestamp |
| `__expires_at__` | int64 | CREATE/UPSERT with `EXPIRES` clause. -1 = no expiry |
| `__retracted__` | int64 | RETRACT command. 0 = active, 1 = retracted |

### 4.6 `live_mask` - Unified Visibility Filter

Computed once per query, ANDed into every subsequent filter:

```python
def _compute_live_mask(self, n: int) -> np.ndarray:
    mask = np.ones(n, dtype=bool)
    # Tombstones
    for slot in self.store.node_tombstones:
        if slot < n:
            mask[slot] = False
    # TTL expiry
    expires = self.store.columns.get_column("__expires_at__", n)
    if expires is not None:
        col, pres, _ = expires
        now_ms = int(time.time() * 1000)
        mask &= ~(pres & (col > 0) & (col < now_ms))
    # Retracted beliefs
    retracted = self.store.columns.get_column("__retracted__", n)
    if retracted is not None:
        col, pres, _ = retracted
        mask &= ~(pres & (col == 1))
    return mask
```

All existing queries use `live_mask` as their base. Zero code duplication for new invisible-node concepts.

**Sentinel values:** If `__expires_at__` is not present (presence=False) or set to -1, the node never expires. If `0 < __expires_at__ < NOW()`, the node is expired and excluded. If `__retracted__` is not present or 0, the node is active.

### 4.7 `live_mask` Integration Points

Every query type integrates `live_mask` at its entry point:

| Query type | Integration |
|---|---|
| `NODES WHERE`, `COUNT`, `AGGREGATE` | `mask = live_mask` before any WHERE filter |
| `TRAVERSE`, `ANCESTORS`, `DESCENDANTS` | BFS skips slots where `live_mask[slot] == False` |
| `SUBGRAPH` | Both node collection and edge collection filter by `live_mask` |
| `MATCH` | Each pattern step filters candidates by `live_mask` |
| `RECALL` | Activation zeroed for non-live nodes at each hop |
| `PATH`, `SHORTEST`, `DISTANCE` | BFS frontier expansion skips non-live nodes |
| `EDGES FROM/TO` | Edge endpoints checked against `live_mask` |
| `NODE "id"` | Returns null if `live_mask[slot] == False` |
| Secondary indices | Index lookup results filtered by `live_mask` post-lookup |

Retracted or expired nodes in the middle of a traversal path break the path - they are invisible, same as tombstoned nodes.

### 4.8 Core Helper Methods

These methods are used throughout Phases 1-4 and must be defined in Phase 1:

```python
def _resolve_slot(self, node_id: str) -> int | None:
    """Return slot index for node ID, or None if not found or tombstoned."""
    if node_id not in self.store.string_table:
        return None
    str_id = self.store.string_table.intern(node_id)
    slot = self.store.id_to_slot.get(str_id)
    if slot is None or slot in self.store.node_tombstones:
        return None
    return slot
```

`ColumnStore` gains two new methods:

```python
def set_reserved(self, slot: int, field: str, value) -> None:
    """Set a system-managed column value. Auto-interns strings."""
    if isinstance(value, str):
        interned = self._string_table.intern(value)
        self._ensure_column(field, "int32_interned")
        self._columns[field][slot] = interned
    elif isinstance(value, float):
        self._ensure_column(field, "float64")
        self._columns[field][slot] = value
    else:
        self._ensure_column(field, "int64")
        self._columns[field][slot] = value
    self._presence[field][slot] = True

def set_field(self, slot: int, field: str, value) -> None:
    """Set a user field value. Same as set_reserved but no prefix check."""
    self.set_reserved(slot, field, value)
```

---

## 5. Phase 1: Infrastructure

### 5.1 Remove `node_data: list[dict]`

**This is a full removal in Phase 1.** The `node_data: list[dict]` field is deleted from `CoreStore`. All field storage moves to `ColumnStore._columns`. No dual-write, no gradual deprecation.

- `CoreStore.put_node()`: write to columns only (no dict store)
- `CoreStore.update_node()`: update columns only
- `CoreStore.delete_node()`: tombstone + `columns.clear(slot)`
- `CoreStore.increment_field()`: `col[slot] += amount` directly
- `CoreStore.get_node()`: call `_materialize_slot()` from columns
- `CoreStore.get_all_nodes()`: filter via masks, materialize matching slots

Impact: `store.py` rewrite of ~15 methods. All tests updated.

**Upsert behavior:** `upsert_node()` calls `put_node()` on create (sets both `__created_at__` and `__updated_at__`) or `update_node()` on update (sets `__updated_at__` only).

### 5.2 Auto-timestamps

On every `put_node()`:
```python
now_ms = int(time.time() * 1000)
columns.set_reserved(slot, "__created_at__", now_ms)
columns.set_reserved(slot, "__updated_at__", now_ms)
```

On every `update_node()` / `upsert_node()` / `increment_field()`:
```python
columns.set_reserved(slot, "__updated_at__", int(time.time() * 1000))
```

These are regular int64 columns - automatically available for WHERE, ORDER BY, GROUP BY. No special handling in the executor.

### 5.3 Relative Time in Transformer

The Lark transformer resolves time expressions to int64 literals at **parse time**. Within a single `execute()` call or batch, `NOW()` returns a constant value. This is by design - batch statements see a consistent timestamp. All timestamps are in local timezone via `datetime.now()`. For UTC, use explicit ISO-8601 strings.

Note: because the parser uses an LRU cache, queries containing `NOW()` must bypass the cache (or the cache key must include the resolved timestamp). Implementation should resolve time expressions in the transformer, after cache lookup.

Grammar additions:
```lark
time_expr: "NOW" "()"                              -> time_now
         | "NOW" "()" "-" NUMBER TIME_UNIT         -> time_offset
         | "TODAY"                                  -> time_today
         | "YESTERDAY"                              -> time_yesterday

TIME_UNIT: "s" | "m" | "h" | "d"
```

Transformer resolves to integer:
```python
def time_now(self, _): return int(time.time() * 1000)
def time_offset(self, items):
    n, unit = int(items[0]), str(items[1])
    ms = {"s": 1000, "m": 60000, "h": 3600000, "d": 86400000}[unit]
    return int(time.time() * 1000) - n * ms
def time_today(self, _):
    return int(datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
def time_yesterday(self, _):
    return int((datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
```

Usage in queries (no executor changes needed):
```sql
NODES WHERE __created_at__ > NOW() - 7d
NODES WHERE __updated_at__ > TODAY
AGGREGATE NODES WHERE __created_at__ > YESTERDAY GROUP BY kind SELECT COUNT()
```

### 5.4 Persistence Update

Serializer: write columns directly (no JSON dicts). Deserializer: read columns, rebuild secondary indices from columns.

The `node_data` JSON blob row is removed from the `blobs` table. All field data is stored as column blobs (existing column serialization pattern, already implemented).

### 5.5 Batch Rollback Update

Save/restore numpy arrays instead of dicts:
```python
# Save state
saved_columns = {f: (col[:ns].copy(), pres[:ns].copy()) for f, (col, pres) in ...}

# Restore on rollback
for f, (col_snap, pres_snap) in saved_columns.items():
    store.columns._columns[f][:ns] = col_snap
    store.columns._presence[f][:ns] = pres_snap
```

Measured: 75μs at 100k nodes vs 24,300μs with dict copy (324x faster).

---

## 6. Phase 2: Aggregations

### 6.1 DSL Grammar

```lark
aggregate_query: "AGGREGATE" "NODES" where_clause? group_clause select_clause having_clause? order_clause? limit_clause?

group_clause: "GROUP" "BY" IDENTIFIER ("," IDENTIFIER)*
select_clause: "SELECT" agg_func ("," agg_func)*
having_clause: "HAVING" condition

agg_func: "COUNT" "()"                              -> agg_count
        | "COUNT" "DISTINCT" "(" IDENTIFIER ")"      -> agg_count_distinct
        | "SUM" "(" IDENTIFIER ")"                    -> agg_sum
        | "AVG" "(" IDENTIFIER ")"                    -> agg_avg
        | "MIN" "(" IDENTIFIER ")"                    -> agg_min
        | "MAX" "(" IDENTIFIER ")"                    -> agg_max
```

### 6.2 AST Nodes

```python
@dataclass
class AggFunc:
    func: str                # "COUNT", "COUNT_DISTINCT", "SUM", "AVG", "MIN", "MAX"
    field: str | None        # None for COUNT()

@dataclass
class AggregateQuery:
    where: Expression | None
    group_by: list[str]
    select: list[AggFunc]
    having: Expression | None
    order_by: str | None     # aggregate expression to sort by
    order_desc: bool
    limit: int | None
```

### 6.3 Executor: `_aggregate()`

**Columnar-only enforcement:** all GROUP BY fields and aggregate fields must have columns. If not, raise `AggregationError`.

```python
def _aggregate(self, query: AggregateQuery) -> Result:
    n = self.store._next_slot
    mask = self._compute_live_mask(n)

    # Apply WHERE filter
    if query.where:
        where_mask = self._try_column_filter(query.where, mask, n)
        if where_mask is None:
            raise AggregationError("WHERE fields must be columnarized for AGGREGATE")
        mask = where_mask

    # Validate all referenced fields are columnarized
    for field in query.group_by:
        if not self.store.columns.has_column(field):
            raise AggregationError(f"Field '{field}' is not columnarized")
    for func in query.select:
        if func.field and not self.store.columns.has_column(func.field):
            raise AggregationError(f"Field '{func.field}' is not columnarized")

    # Build group keys
    if query.group_by:
        group_cols = [self.store.columns._columns[f][:n][mask] for f in query.group_by]
        if len(group_cols) == 1:
            keys = group_cols[0]
        else:
            keys = np.stack(group_cols, axis=1)
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    else:
        # No GROUP BY = single group
        unique_keys = np.array([0])
        inverse = np.zeros(int(np.sum(mask)), dtype=np.intp)

    num_groups = len(unique_keys)

    # Compute aggregates
    results = {}
    for func in query.select:
        if func.func == "COUNT":
            results["COUNT()"] = np.bincount(inverse, minlength=num_groups)
        elif func.func == "COUNT_DISTINCT":
            col = self.store.columns._columns[func.field][:n][mask]
            # Per-group distinct count
            counts = np.zeros(num_groups, dtype=np.int64)
            for g in range(num_groups):
                counts[g] = len(np.unique(col[inverse == g]))
            results[f"COUNT_DISTINCT({func.field})"] = counts
        elif func.func == "SUM":
            col = self.store.columns._columns[func.field][:n][mask]
            sums = np.zeros(num_groups, dtype=np.float64)
            np.add.at(sums, inverse, col.astype(np.float64))
            results[f"SUM({func.field})"] = sums
        elif func.func == "AVG":
            col = self.store.columns._columns[func.field][:n][mask]
            sums = np.zeros(num_groups, dtype=np.float64)
            np.add.at(sums, inverse, col.astype(np.float64))
            counts = np.bincount(inverse, minlength=num_groups)
            results[f"AVG({func.field})"] = sums / np.maximum(counts, 1)
        elif func.func == "MIN":
            col = self.store.columns._columns[func.field][:n][mask].astype(np.float64)
            mins = np.full(num_groups, np.inf)
            np.minimum.at(mins, inverse, col)
            results[f"MIN({func.field})"] = mins
        elif func.func == "MAX":
            col = self.store.columns._columns[func.field][:n][mask].astype(np.float64)
            maxs = np.full(num_groups, -np.inf)
            np.maximum.at(maxs, inverse, col)
            results[f"MAX({func.field})"] = maxs

    # Build result dicts
    group_dicts = []
    for i in range(num_groups):
        d = {}
        # Group-by field values
        for j, field in enumerate(query.group_by):
            raw = unique_keys[i] if len(query.group_by) == 1 else unique_keys[i][j]
            dtype = self.store.columns._dtypes[field]
            if dtype == "int32_interned":
                d[field] = self.store.string_table.lookup(int(raw))
            elif dtype == "float64":
                d[field] = float(raw)
            elif dtype == "int64":
                d[field] = int(raw)
        # Aggregate values
        for key, arr in results.items():
            d[key] = float(arr[i]) if isinstance(arr[i], (np.floating, np.integer)) else arr[i]
        group_dicts.append(d)

    # HAVING filter
    if query.having:
        group_dicts = [d for d in group_dicts if self._eval_where(query.having, d)]

    # ORDER BY
    if query.order_by:
        group_dicts.sort(key=lambda d: d.get(query.order_by, 0), reverse=query.order_desc)

    # LIMIT
    if query.limit:
        group_dicts = group_dicts[:query.limit]

    return Result(kind="aggregate", data=group_dicts, count=len(group_dicts))
```

### 6.4 Examples

```sql
-- How many memories per topic?
AGGREGATE NODES WHERE kind = "memory"
  GROUP BY topic
  SELECT COUNT(), AVG(importance)
  HAVING COUNT() > 2
  ORDER BY AVG(importance) DESC

-- Global summary (no GROUP BY)
AGGREGATE NODES WHERE kind = "fact"
  SELECT COUNT(), AVG(confidence), MAX(__updated_at__)

-- Token usage per session
AGGREGATE NODES WHERE kind = "event"
  GROUP BY session_id
  SELECT SUM(tokens), MIN(__created_at__), MAX(__created_at__)
```

**Performance:** GROUP BY + AVG at 100k nodes: ~744μs (measured).

---

## 7. Phase 3: Belief Operations + Agent Primitives

### 7.1 `ASSERT` - First-Class Belief Creation

Grammar:
```lark
assert_stmt: "ASSERT" STRING field_assign+ ("CONFIDENCE" NUMBER)? ("SOURCE" STRING)?
```

Semantics: UPSERT with reserved fields.

```python
def _assert(self, query):
    # Build fields from user assigns
    fields = dict(query.assigns)
    fields["__confidence__"] = query.confidence or 1.0
    fields["__source__"] = query.source or ""
    fields["__retracted__"] = 0
    # Delegate to upsert
    return self._upsert_node(UpsertNode(id=query.id, assigns=fields))
```

Example:
```sql
ASSERT "fact:earth-radius" value = 6371 kind = "fact" CONFIDENCE 0.99 SOURCE "physics-tool"
```

### 7.2 `RETRACT` - Belief Retraction

Grammar:
```lark
retract_stmt: "RETRACT" STRING ("REASON" STRING)?
```

Semantics: marks node as retracted (invisible to normal queries via `live_mask`), but preserves the node for audit.

```python
def _retract(self, query):
    slot = self._resolve_slot(query.id)
    if slot is None:
        raise NodeNotFound(query.id)
    self.store.columns.set_reserved(slot, "__retracted__", 1)
    self.store.columns.set_reserved(slot, "__retract_reason__", query.reason or "")
    self.store.columns.set_reserved(slot, "__retracted_at__", int(time.time() * 1000))
    return Result(kind="ok", data={"id": query.id, "retracted": True})
```

Example:
```sql
RETRACT "fact:old-user-pref" REASON "user corrected this"
```

### 7.3 `UPDATE NODES WHERE` - Bulk Mutation

Grammar:
```lark
update_nodes: "UPDATE" "NODES" where_clause "SET" field_assign ("," field_assign)*
```

Semantics: filter via column mask, bulk-assign new values.

```python
def _update_nodes(self, query):
    n = self.store._next_slot
    mask = self._compute_live_mask(n)
    col_mask = self._try_column_filter(query.where, mask, n)
    if col_mask is None:
        raise QueryError("UPDATE NODES WHERE requires columnarized filter fields")

    slots = np.where(col_mask)[0]
    for slot in slots:
        for field, value in query.assigns:
            self.store.columns.set_field(int(slot), field, value)
        self.store.columns.set_reserved(int(slot), "__updated_at__", int(time.time() * 1000))

    # Rebuild secondary indices for affected fields
    for field, _ in query.assigns:
        if field in self.store._indexed_fields:
            self.store._rebuild_index(field)

    return Result(kind="ok", data={"updated": len(slots)}, count=len(slots))
```

Example:
```sql
UPDATE NODES WHERE kind = "conclusion" AND source = "unreliable-agent"
  SET confidence = 0.1
```

### 7.4 `SYS CONTRADICTIONS` - Belief Consistency Check

Grammar:
```lark
sys_contradictions: "CONTRADICTIONS" where_clause? "FIELD" IDENTIFIER "GROUP" "BY" IDENTIFIER
```

Semantics: GROUP BY the grouping field, find groups where the target field has more than one distinct value.

```python
def _contradictions(self, query):
    n = self.store._next_slot
    mask = self._compute_live_mask(n)
    if query.where:
        mask = self._try_column_filter(query.where, mask, n)

    group_col = self.store.columns._columns[query.group_by][:n][mask]
    value_col = self.store.columns._columns[query.field][:n][mask]

    unique_groups = np.unique(group_col)
    contradictions = []
    for g in unique_groups:
        g_mask = group_col == g
        unique_vals = np.unique(value_col[g_mask])
        if len(unique_vals) > 1:
            group_name = self._deref(query.group_by, g)
            values = [self._deref(query.field, v) for v in unique_vals]
            contradictions.append({"group": group_name, "values": values, "count": len(unique_vals)})

    return Result(kind="contradictions", data=contradictions, count=len(contradictions))
```

Example:
```sql
SYS CONTRADICTIONS WHERE kind = "belief" FIELD value GROUP BY topic
-- returns: [{group: "paris_population", values: ["2M", "2.1M"], count: 2}, ...]
```

### 7.5 TTL - `EXPIRES IN` / `EXPIRES AT`

Grammar additions to `create_node` and `upsert_node`:
```lark
expires_clause: "EXPIRES" "IN" NUMBER TIME_UNIT    -> expires_in
              | "EXPIRES" "AT" STRING              -> expires_at

TIME_UNIT: "s" | "m" | "h" | "d"
```

On create:
```python
if query.expires_in:
    ms = {"s": 1000, "m": 60000, "h": 3600000, "d": 86400000}[query.expires_unit]
    expires_at = int(time.time() * 1000) + query.expires_in * ms
    self.store.columns.set_reserved(slot, "__expires_at__", expires_at)
elif query.expires_at:
    expires_at = int(datetime.fromisoformat(query.expires_at).timestamp() * 1000)
    self.store.columns.set_reserved(slot, "__expires_at__", expires_at)
```

`live_mask` automatically excludes expired nodes. No query changes needed.

### 7.6 `SYS EXPIRE` - Flush Expired Nodes

Grammar:
```lark
sys_expire: "EXPIRE" where_clause?
```

Semantics: find all nodes where `__expires_at__ > 0 AND __expires_at__ < NOW()`, tombstone them, cascade-delete their edges.

```python
def _expire(self, query):
    n = self.store._next_slot
    expires = self.store.columns._columns.get("__expires_at__")
    if expires is None:
        return Result(kind="ok", data={"expired": 0})

    now_ms = int(time.time() * 1000)
    expired_mask = (self.store.columns._presence["__expires_at__"][:n]
                    & (expires[:n] > 0)
                    & (expires[:n] < now_ms))

    if query.where:
        extra = self._try_column_filter(query.where, expired_mask, n)
        if extra is not None:
            expired_mask = extra

    slots = np.where(expired_mask)[0]
    for slot in slots:
        node_id = self.store.string_table.lookup(int(self.store.node_ids[slot]))
        self.store.delete_node(node_id)

    return Result(kind="ok", data={"expired": len(slots)}, count=len(slots))
```

Example:
```sql
SYS EXPIRE                                    -- flush all expired nodes
SYS EXPIRE WHERE kind = "working"             -- flush only expired working memories
```

### 7.7 `MERGE NODE "a" INTO "b"` - Memory Consolidation

Grammar:
```lark
merge_stmt: "MERGE" "NODE" STRING "INTO" STRING
```

Semantics:
1. Copy all fields from source to target (target wins on conflict)
2. Re-wire all edges from source to target
3. Drop duplicate edges (same src/tgt/kind)
4. Tombstone source

```python
def _merge(self, query):
    src_slot = self._resolve_slot(query.source_id)
    tgt_slot = self._resolve_slot(query.target_id)
    if src_slot is None: raise NodeNotFound(query.source_id)
    if tgt_slot is None: raise NodeNotFound(query.target_id)

    # 1. Merge fields: copy source fields to target where target has no value
    fields_merged = 0
    for field in self.store.columns._columns:
        if (self.store.columns._presence[field][src_slot]
                and not self.store.columns._presence[field][tgt_slot]):
            self.store.columns._columns[field][tgt_slot] = self.store.columns._columns[field][src_slot]
            self.store.columns._presence[field][tgt_slot] = True
            fields_merged += 1

    # 2. Re-wire edges
    edges_rewired = 0
    for kind, edge_list in self.store._edges_by_type.items():
        for i, (s, t, d) in enumerate(edge_list):
            if s == src_slot:
                edge_list[i] = (tgt_slot, t, d)
                edges_rewired += 1
            elif t == src_slot:
                edge_list[i] = (s, tgt_slot, d)
                edges_rewired += 1

    # 3. Remove duplicate edges
    for kind in self.store._edges_by_type:
        seen = set()
        deduped = []
        for s, t, d in self.store._edges_by_type[kind]:
            key = (s, t)
            if key not in seen:
                seen.add(key)
                deduped.append((s, t, d))
        self.store._edges_by_type[kind] = deduped

    # 4. Rebuild edge keys
    self.store._edge_keys = {
        (s, t, k) for k, edges in self.store._edges_by_type.items() for s, t, _d in edges
    }
    self.store._edges_dirty = True

    # 5. Tombstone source
    self.store.delete_node(query.source_id)

    # 6. Update target timestamp
    self.store.columns.set_reserved(tgt_slot, "__updated_at__", int(time.time() * 1000))

    return Result(kind="ok", data={
        "merged_into": query.target_id,
        "fields_merged": fields_merged,
        "edges_rewired": edges_rewired,
    })
```

Example:
```sql
MERGE NODE "memory:paris-1" INTO "memory:paris-canonical"
```

---

## 8. Phase 4: Graph Intelligence

### 8.1 `RECALL` - Spreading Activation

Grammar:
```lark
recall_query: "RECALL" "FROM" STRING "DEPTH" NUMBER ("LIMIT" NUMBER)? where_clause?
```

Semantics: seed activation at cue node, propagate through CSR edges via sparse matrix-vector multiply for `depth` hops. At each hop, activation is weighted by `importance * recency`. This produces exponential decay with distance - nodes 2 hops away get `(importance * recency)^2` weighting, which naturally prioritizes close, important, recent memories. Return top-K by activation score.

WHERE clause is applied as an output filter after propagation completes. Nodes excluded by WHERE still propagate activation to their neighbors (they act as relay nodes). Nodes excluded by `live_mask` (tombstoned, expired, retracted) do NOT propagate.

```python
def _recall(self, query):
    n = self.store._next_slot
    live = self._compute_live_mask(n)
    cue_slot = self._resolve_slot(query.node_id)
    if cue_slot is None:
        raise NodeNotFound(query.node_id)

    # Seed activation
    activation = np.zeros(n, dtype=np.float64)
    activation[cue_slot] = 1.0

    # Get combined CSR matrix (sum all edge types)
    self.store._maybe_rebuild_edges()
    matrices = self.store.edge_matrices
    csr = None
    for etype in matrices.edge_types:
        m = matrices.get(etype)
        if m is not None:
            if m.shape[0] < n or m.shape[1] < n:
                from scipy.sparse import csr_matrix
                m = csr_matrix(m, shape=(n, n))
            csr = m if csr is None else csr + m
    if csr is None:
        return Result(kind="nodes", data=[], count=0)

    # Importance weighting (default 1.0 if no column)
    importance = np.ones(n, dtype=np.float64)
    imp_col = self.store.columns.get_column("importance", n)
    if imp_col is not None:
        col, pres, _ = imp_col
        importance[pres] = col[pres].astype(np.float64)

    conf_col = self.store.columns.get_column("__confidence__", n)
    if conf_col is not None:
        col, pres, _ = conf_col
        importance[pres] *= col[pres].astype(np.float64)

    # Recency weighting
    now_ms = int(time.time() * 1000)
    recency = np.ones(n, dtype=np.float64)
    updated = self.store.columns.get_column("__updated_at__", n)
    if updated is not None:
        col, pres, _ = updated
        age_days = (now_ms - col[:n].astype(np.float64)) / 86_400_000
        recency = 1.0 / (1.0 + np.maximum(age_days, 0))

    # Propagate
    for hop in range(query.depth):
        activation = csr.dot(activation)
        activation *= importance
        activation *= recency
        activation *= live.astype(np.float64)

    # Exclude cue node from results
    activation[cue_slot] = 0.0

    # Apply WHERE filter if present
    if query.where:
        where_mask = self._try_column_filter(query.where, live, n)
        if where_mask is not None:
            activation *= where_mask.astype(np.float64)

    # Top-K
    k = query.limit or 20
    k = min(k, int(np.sum(activation > 0)))
    if k == 0:
        return Result(kind="nodes", data=[], count=0)

    top_slots = np.argpartition(-activation, k)[:k]
    top_slots = top_slots[np.argsort(-activation[top_slots])]

    # Materialize
    results = []
    for slot in top_slots:
        d = self._materialize_slot(int(slot))
        d["_activation_score"] = float(activation[slot])
        results.append(d)

    return Result(kind="nodes", data=results, count=len(results))
```

**Performance:** O(edges x depth) per recall. At 100k nodes, 500k edges, depth 3: ~1-3ms (scipy sparse matmul runs in C).

Example:
```sql
RECALL FROM "concept:paris" DEPTH 3 LIMIT 10
RECALL FROM "concept:paris" DEPTH 3 LIMIT 10 WHERE kind = "memory"
```

### 8.2 `PROPAGATE` - Forward Belief Chaining

Grammar:
```lark
propagate_stmt: "PROPAGATE" STRING "FIELD" IDENTIFIER "DEPTH" NUMBER
```

Semantics: starting from a source node, propagate a field value through outgoing edges. Each hop multiplies the value by edge weight (default 1.0). Descendants get `field = source_value * product(edge_weights along path)`.

```python
def _propagate(self, query):
    n = self.store._next_slot
    src_slot = self._resolve_slot(query.node_id)
    if src_slot is None:
        raise NodeNotFound(query.node_id)

    # Get source field value
    col, pres, dtype = self.store.columns.get_column(query.field, n)
    if not pres[src_slot]:
        raise QueryError(f"Source node has no value for field '{query.field}'")
    src_value = float(col[src_slot])

    # BFS propagation
    self.store._maybe_rebuild_edges()
    csr = None
    for etype in self.store.edge_matrices.edge_types:
        m = self.store.edge_matrices.get(etype)
        if m is not None:
            csr = m if csr is None else csr + m
    if csr is None:
        return Result(kind="ok", data={"updated": 0})

    # Propagation vector
    prop = np.zeros(n, dtype=np.float64)
    prop[src_slot] = src_value

    updated_slots = set()
    now_ms = int(time.time() * 1000)
    for hop in range(query.depth):
        prop = csr.T.dot(prop)  # transpose for forward propagation (follow outgoing edges)
        nonzero = np.where(prop > 0)[0]
        for slot in nonzero:
            if slot != src_slot:
                self.store.columns.set_field(int(slot), query.field, float(prop[slot]))
                self.store.columns.set_reserved(int(slot), "__updated_at__", now_ms)
                updated_slots.add(int(slot))

    return Result(kind="ok", data={"updated": len(updated_slots)}, count=len(updated_slots))
```

Example:
```sql
PROPAGATE "belief:user-is-expert" FIELD confidence DEPTH 3
-- Descendants get confidence proportional to graph distance and edge weight
```

### 8.3 `COUNTERFACTUAL WHAT IF` - Hypothesis Testing

Grammar:
```lark
counterfactual: "WHAT" "IF" "RETRACT" STRING
```

Semantics: fork the column state, apply the retraction, find all affected downstream nodes, return the diff without committing.

```python
def _counterfactual(self, query):
    n = self.store._next_slot

    # Full state snapshot (same as SYS SNAPSHOT but ephemeral)
    saved = {
        "columns": {f: (self.store.columns._columns[f][:n].copy(),
                        self.store.columns._presence[f][:n].copy())
                    for f in self.store.columns._columns},
        "tombstones": set(self.store.node_tombstones),
        "edges_by_type": {k: list(v) for k, v in self.store._edges_by_type.items()},
        "edge_keys": set(self.store._edge_keys),
    }

    # Apply hypothetical retraction
    slot = self._resolve_slot(query.node_id)
    if slot is None:
        raise NodeNotFound(query.node_id)

    # Find all descendants that reference this node
    self.store._maybe_rebuild_edges()
    desc = set()
    frontier = {slot}
    for _ in range(5):  # max propagation depth
        next_frontier = set()
        for s in frontier:
            for etype in self.store.edge_matrices.edge_types:
                m = self.store.edge_matrices.get(etype)
                if m is not None and s < m.shape[0]:
                    row = m.getrow(s)
                    next_frontier.update(row.indices.tolist())
        frontier = next_frontier - desc
        desc.update(frontier)

    # Materialize affected nodes
    affected = []
    for s in desc:
        if s < n and s not in self.store.node_tombstones:
            d = self._materialize_slot(s)
            affected.append(d)

    # Restore full state (discard hypothesis)
    for f, (col_snap, pres_snap) in saved["columns"].items():
        self.store.columns._columns[f][:n] = col_snap
        self.store.columns._presence[f][:n] = pres_snap
    self.store.node_tombstones = saved["tombstones"]
    self.store._edges_by_type = saved["edges_by_type"]
    self.store._edge_keys = saved["edge_keys"]
    self.store._edges_dirty = True

    return Result(kind="counterfactual", data={
        "retracted": query.node_id,
        "affected_nodes": affected,
        "affected_count": len(affected),
    }, count=len(affected))
```

**Performance:** np.copy of all columns at 100k = ~75μs (measured). Total: 75μs copy + traversal time + 75μs restore.

Example:
```sql
WHAT IF RETRACT "belief:user-likes-dark-mode"
-- Returns: {affected_nodes: [...], affected_count: 14}
-- Does NOT commit. Pure simulation.
```

### 8.4 `SYS SNAPSHOT` / `SYS ROLLBACK` - Reasoning Branches

Grammar:
```lark
sys_snapshot: "SNAPSHOT" STRING
sys_rollback: "ROLLBACK" "TO" STRING
sys_snapshots: "SNAPSHOTS"
```

Semantics: Named snapshots of the entire column store + edge state. Implemented via existing `snapshot.py` infrastructure, adapted for columnar storage.

```python
def _snapshot(self, query):
    name = query.name
    n = self.store._next_slot
    snap = {
        "columns": {f: (col[:n].copy(), pres[:n].copy())
                    for f, col in self.store.columns._columns.items()
                    for pres in [self.store.columns._presence[f]]},
        "dtypes": dict(self.store.columns._dtypes),
        "node_ids": self.store.node_ids[:n].copy(),
        "node_kinds": self.store.node_kinds[:n].copy(),
        "tombstones": set(self.store.node_tombstones),
        "edges_by_type": {k: list(v) for k, v in self.store._edges_by_type.items()},
        "next_slot": self.store._next_slot,
        "count": self.store._count,
    }
    self._snapshots[name] = snap
    return Result(kind="ok", data={"snapshot": name})

def _rollback(self, query):
    name = query.name
    snap = self._snapshots.get(name)
    if snap is None:
        raise QueryError(f"Snapshot '{name}' not found")

    # Restore all state
    n = snap["next_slot"]
    for f, (col_snap, pres_snap) in snap["columns"].items():
        self.store.columns._columns[f][:n] = col_snap
        self.store.columns._presence[f][:n] = pres_snap
    self.store.columns._dtypes = dict(snap["dtypes"])
    self.store.node_ids[:n] = snap["node_ids"]
    self.store.node_kinds[:n] = snap["node_kinds"]
    self.store.node_tombstones = set(snap["tombstones"])
    self.store._edges_by_type = {k: list(v) for k, v in snap["edges_by_type"].items()}
    self.store._next_slot = snap["next_slot"]
    self.store._count = snap["count"]
    self.store._edges_dirty = True

    # Rebuild derived state
    self.store._rebuild_id_to_slot()
    self.store._rebuild_edge_keys()
    for field in self.store._indexed_fields:
        self.store._rebuild_index(field)

    # Remove snapshot after rollback (one-shot)
    del self._snapshots[name]

    return Result(kind="ok", data={"rolled_back_to": name})
```

Examples:
```sql
SYS SNAPSHOT "before-hypothesis"
-- ... agent explores reasoning branch ...
SYS ROLLBACK TO "before-hypothesis"
SYS SNAPSHOTS                          -- list all named snapshots
```

### 8.5 `BIND CONTEXT` / `DISCARD CONTEXT` - Isolation

Grammar:
```lark
bind_context: "BIND" "CONTEXT" STRING
discard_context: "DISCARD" "CONTEXT" STRING
```

Semantics: sets a session-level context name. While bound:
- All CREATEs auto-tag nodes with `__context__ = context_name`
- All reads (NODES, RECALL, etc.) auto-filter by `__context__ = context_name`
- DISCARD deletes all nodes in the context

```python
# Session state in Executor
self._active_context: str | None = None

def _bind_context(self, query):
    self._active_context = query.name
    return Result(kind="ok", data={"context": query.name})

def _discard_context(self, query):
    # Delete all nodes in context
    n = self.store._next_slot
    ctx_mask = self.store.columns.get_mask("__context__", "=",
        self.store.string_table.intern(query.name), n)
    if ctx_mask is not None:
        slots = np.where(ctx_mask)[0]
        for slot in slots:
            nid = self.store.string_table.lookup(int(self.store.node_ids[slot]))
            self.store.delete_node(nid)
    self._active_context = None
    return Result(kind="ok", data={"discarded": query.name, "deleted": len(slots)})
```

When `_active_context` is set, `_compute_live_mask()` adds:
```python
if self._active_context:
    ctx_id = self.store.string_table.intern(self._active_context)
    ctx_mask = self.store.columns.get_mask("__context__", "=", ctx_id, n)
    if ctx_mask is not None:
        mask &= ctx_mask
```

Examples:
```sql
BIND CONTEXT "reasoning-session-42"
CREATE NODE "hyp:1" kind = "hypothesis" content = "..."   -- auto-tagged
RECALL FROM "hyp:1" DEPTH 3 LIMIT 10                      -- scoped to context
DISCARD CONTEXT "reasoning-session-42"                     -- cleanup
```

---

## 9. Reserved Columns Summary

| Column | Type | Managed by | Purpose |
|---|---|---|---|
| `__created_at__` | int64 | put_node | Unix ms timestamp of creation |
| `__updated_at__` | int64 | put/update/upsert/increment/propagate | Unix ms timestamp of last modification |
| `__expires_at__` | int64 | EXPIRES clause | Unix ms expiry time. -1 = no expiry. Presence=False = no expiry. |
| `__retracted__` | int64 | RETRACT | 0 = active, 1 = retracted. Presence=False = active. |
| `__retracted_at__` | int64 | RETRACT | Unix ms timestamp of retraction |
| `__retract_reason__` | int32_interned | RETRACT | Reason string (auto-interned via `set_reserved`) |
| `__confidence__` | float64 | ASSERT | Belief confidence 0.0-1.0 |
| `__source__` | int32_interned | ASSERT | Source agent/tool identifier (auto-interned via `set_reserved`) |
| `__context__` | int32_interned | BIND CONTEXT | Isolation context name (auto-interned via `set_reserved`) |

All string-typed reserved columns use `set_reserved()` which auto-interns the string value to int32 before storing. See section 4.8 for the method signature.

---

## 10. New DSL Commands Summary

### Phase 1 (Infrastructure - no new commands)
- Internal: `live_mask`, auto-timestamps, relative time, columnar source of truth
- Grammar: `NOW()`, `TODAY`, `YESTERDAY`, `NOW() - Nd/h/m/s` as value expressions

### Phase 2 (Aggregation)
```
AGGREGATE NODES [WHERE expr] [GROUP BY field, ...] SELECT func, ... [HAVING cond] [ORDER BY func ASC|DESC] [LIMIT n]
```

### Phase 3 (Belief ops)
```
ASSERT "id" field = value ... [CONFIDENCE n] [SOURCE "s"]
RETRACT "id" [REASON "r"]
UPDATE NODES WHERE expr SET field = value, ...
MERGE NODE "a" INTO "b"
CREATE NODE "id" ... EXPIRES IN n[s|m|h|d]
CREATE NODE "id" ... EXPIRES AT "iso8601"
SYS EXPIRE [WHERE expr]
SYS CONTRADICTIONS [WHERE expr] FIELD f GROUP BY g
```

### Phase 4 (Graph intelligence)
```
RECALL FROM "id" DEPTH n [LIMIT k] [WHERE expr]
PROPAGATE "id" FIELD f DEPTH n
WHAT IF RETRACT "id"
SYS SNAPSHOT "name"
SYS ROLLBACK TO "name"
SYS SNAPSHOTS
BIND CONTEXT "name"
DISCARD CONTEXT "name"
```

---

## 11. New Error Types

| Error | Raised by | Message |
|---|---|---|
| `AggregationError(SchemaError)` | AGGREGATE | Field not columnarized |
| `SnapshotNotFound(GraphStoreError)` | SYS ROLLBACK | Named snapshot doesn't exist |
| `ContextError(GraphStoreError)` | DISCARD CONTEXT | Context not bound |

---

## 12. Files Changed

| File | Phase | Changes |
|---|---|---|
| `graphstore/store.py` | 1 | Remove `node_data`, all methods use columns |
| `graphstore/columns.py` | 1 | `set_reserved()`, `set_field()`, `get_column()` enhancements |
| `graphstore/dsl/grammar.lark` | 1-4 | All new grammar rules |
| `graphstore/dsl/ast_nodes.py` | 1-4 | All new AST dataclasses |
| `graphstore/dsl/transformer.py` | 1-4 | Parse tree -> AST for new nodes |
| `graphstore/dsl/executor.py` | 1-4 | `_aggregate`, `_recall`, `_merge`, `_update_nodes`, `_assert`, `_retract`, `_counterfactual`, `_bind_context`, `_discard_context`, `_compute_live_mask`, `_materialize_slot` |
| `graphstore/dsl/executor_system.py` | 3-4 | `_expire`, `_contradictions`, `_snapshot`, `_rollback`, `_snapshots` |
| `graphstore/persistence/serializer.py` | 1 | Remove node_data JSON blob, columns only |
| `graphstore/persistence/deserializer.py` | 1 | Rebuild from columns, no dict loading |
| `graphstore/errors.py` | 2-4 | New error types |
| `graphstore/__init__.py` | 1 | Remove node_data references from GraphStore wrapper |
| `tests/` | 1-4 | All test files updated |

---

## 13. Migration & Compatibility

### 13.1 Existing Stored Graphs

The Phase 1 deserializer must handle the old format (node_data JSON blob). Migration path:

1. If `blobs` table contains a `node_data` key, load the JSON dicts
2. For each node dict, write all fields into column arrays
3. Auto-infer column types from values (same as `ColumnStore.set()`)
4. Do NOT write a `node_data` blob on next checkpoint - only column blobs

This is a one-time migration on first load. After checkpoint, the old format is gone.

### 13.2 AGGREGATE Edge Cases

**Global aggregate (no GROUP BY):** Returns a single-row result with aggregate values. If no nodes match the WHERE clause, returns `[{"COUNT()": 0, "SUM(x)": 0, "AVG(x)": null}]` (one row, zeroed aggregates). This matches SQL behavior.

**Empty GROUP BY result:** If a GROUP BY field has no non-null values, returns `[]` (empty list).

---

## 14. Performance Targets

| Operation | Scale | Target | Basis |
|---|---|---|---|
| AGGREGATE GROUP BY + AVG | 100k nodes | < 1ms | Measured: 744μs |
| COUNT WHERE | 100k nodes | < 50μs | Measured: 34μs |
| RECALL DEPTH 3 LIMIT 10 | 100k nodes, 500k edges | < 5ms | Estimated: sparse matmul |
| live_mask computation | 100k nodes | < 50μs | 3 numpy boolean ops |
| Materialize LIMIT 10 | any | < 10μs | Measured: 6μs |
| SYS EXPIRE (1k expired) | 100k nodes | < 10ms | mask + 1k delete_node calls |
| COUNTERFACTUAL | 100k nodes | < 5ms | 75μs copy + traversal + 75μs restore |
| Batch rollback | 100k nodes | < 100μs | Measured: 75μs |
| Memory per node | 5-10 fields | < 100 bytes | Measured: 35 bytes at 100k |
