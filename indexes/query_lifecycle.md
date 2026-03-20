# Query Lifecycle

The execution of a query in `graphstore` involves a streamlined pipeline driven by the bespoke Domain Specific Language (DSL).

## 1. Parsing and Grammar (`dsl/parser.py`, `dsl/grammar.lark`)
1. **LALR(1) Parsing**: Queries are evaluated through a context-free grammar using the `Lark` parser.
2. **LRU Cache**: Parsed ASTs are cached. If the exact statement hits the cache, heavy parsing is skipped entirely.
3. **Time bypass**: Queries containing `NOW()`, `TODAY`, or `YESTERDAY` bypass the cache since time expressions resolve at parse time.

## 2. AST Transformation (`dsl/transformer.py`, `dsl/ast_nodes.py`)
The transformer traverses the parse tree bottom-up and maps symbols into typed dataclasses:
- **Read queries**: `NodesQuery`, `AggregateQuery`, `RecallQuery`, `CounterfactualQuery`, `MatchQuery`, etc.
- **Write queries**: `CreateNode`, `AssertStmt`, `RetractStmt`, `UpdateNodes`, `MergeStmt`, `PropagateStmt`, `BindContext`, `DiscardContext`, etc.
- **System queries**: `SysExpire`, `SysContradictions`, `SysSnapshot`, `SysRollback`, etc.
- **Time resolution**: `NOW()`, `TODAY`, `YESTERDAY`, `NOW() - 7d` are resolved to int64 millisecond timestamps during transformation.

## 3. Cost Estimation (`dsl/cost_estimator.py`)
To protect against runaway queries:
- `TRAVERSE`, `MATCH`, and `PATH` queries undergo pre-execution frontier estimation.
- If the estimated frontier exceeds the threshold (default 100,000), `CostThresholdExceeded` is raised before any work.
- `MATCH` patterns cap result bindings at 1,000 per hop.

## 4. Visibility Mask (`executor.py:_compute_live_mask`)
Before any query executes, a boolean numpy mask is computed:
- Excludes tombstoned slots, TTL-expired nodes, retracted beliefs
- Adds context filtering when `BIND CONTEXT` is active
- This mask is ANDed into every subsequent filter operation

## 5. Execution Pipeline (`dsl/executor.py`)

### Read path (3 tiers, fastest first)
1. **Secondary index** - O(1) equality lookup for indexed fields
2. **Column filter** - numpy vectorized mask operations (=, !=, >, <, IN, NULL). 100-300x faster than dict scan
3. **Raw predicate** - Python callable fallback for CONTAINS, LIKE, DegreeCondition

### Write path
- Writes go directly to `ColumnStore.set()` (no dict intermediary)
- Auto-timestamps (`__created_at__`, `__updated_at__`) injected on every write
- Secondary indices maintained inline

### Aggregate path
- `AGGREGATE NODES GROUP BY ... SELECT SUM/AVG/MIN/MAX/COUNT DISTINCT`
- Columnar-only enforcement (raises `AggregationError` if field not columnarized)
- Uses `np.unique` + `np.bincount` + `np.add.at` for vectorized group computation

### Recall path (spreading activation)
- `RECALL FROM "cue" DEPTH n LIMIT k`
- Sparse matrix-vector multiply: `csr.T.dot(activation)` per hop
- Weighted by importance column, confidence, and recency (`1 / (1 + age_days)`)
- Live_mask applied each hop (retracted/expired nodes don't relay activation)
- Top-K via `np.argpartition` for O(n + k log k)

### Belief operations
- `ASSERT`: upsert with `__confidence__`, `__source__`, `__retracted__=0`
- `RETRACT`: sets `__retracted__=1` (node becomes invisible via live_mask, kept for audit)
- `UPDATE NODES WHERE`: bulk column mutation via numpy masks
- `MERGE NODE "a" INTO "b"`: copy fields (target wins), rewire edges, tombstone source

### Hypothesis testing
- `WHAT IF RETRACT "x"`: save full state, apply hypothetical retraction, BFS descendants, restore
- `SYS SNAPSHOT/ROLLBACK`: named state snapshots via numpy array copy (~75us at 100k nodes)
- `BIND/DISCARD CONTEXT`: session-level isolation with auto-tagging

### Batch transactions
- `BEGIN ... COMMIT` with full rollback on failure
- State saved/restored via `ColumnStore.snapshot_arrays()` / `restore_arrays()`
- 324x faster than dict-based rollback at 100k nodes

## 6. Yielding Results (`types.py:Result`)
Execution folds into `Result(kind=..., data=..., count=...)`. Node dicts are materialized on demand from column arrays only for result slots - never for intermediate filtering steps.
