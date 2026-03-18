# GraphStore — Complete Implementation Plan

## 1. What This Is

An in-memory typed graph database in Python. Directed graph with typed nodes and typed edges, backed by scipy CSR sparse matrices, persisted to sqlite, queryable through a custom DSL designed for LLM consumption over MCP. Hard memory ceiling of 256MB. If the graph doesn't fit, we don't support it.

Deliverable: a single pip-installable Python package.

---

## 2. Design Principles

1. Speed is the primary constraint. Reads must be sub-millisecond. The LLM's API latency (500ms-2s) must always be the bottleneck, never the DB.
2. The graph lives entirely in memory. Persistence is for durability across restarts, not for serving queries. No disk I/O on the read path.
3. 256MB hard ceiling. No fallback disk-backed mode. No partial indexing. No streaming queries. The graph fits or it doesn't.
4. Single process, single machine. No concurrency model beyond snapshot swap for safe reads during writes.
5. Two dependencies: scipy (CSR + graph algorithms), lark (DSL parser). Persistence uses stdlib sqlite3. Nothing else.
6. The DSL is the only interface. Both user queries and system operations go through `store.execute(query_string)`. No escape hatches.

---

## 3. Dependencies

```toml
[project]
name = "graphstore"
requires-python = ">=3.10"
dependencies = [
    "scipy>=1.10",
    "lark>=1.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-benchmark>=4.0",
]
```

numpy comes as a transitive dependency of scipy. sqlite3 is stdlib. No other runtime dependencies.

---

## 4. Package Structure

```
graphstore/
├── __init__.py                # Public API: GraphStore class
├── store.py                   # Core store: node arrays, edge matrices, indices
├── edges.py                   # Edge matrix manager: per-type CSR, combination cache
├── strings.py                 # String table: intern + lookup
├── types.py                   # Result, Edge, NodeData dataclasses
├── errors.py                  # All error types
├── memory.py                  # Memory estimator + ceiling enforcement
├── snapshot.py                # GraphSnapshot + atomic swap
├── path.py                    # Bidirectional BFS for point-to-point shortest path
├── dsl/
│   ├── __init__.py
│   ├── grammar.lark           # Complete DSL grammar (lark format)
│   ├── parser.py              # Lark parser wrapper + plan cache
│   ├── transformer.py         # Lark Transformer: parse tree → AST nodes
│   ├── ast_nodes.py           # Typed AST node dataclasses
│   ├── executor.py            # AST → store method dispatch (user queries)
│   ├── executor_system.py     # AST → store method dispatch (SYS queries)
│   └── cost_estimator.py      # EXPLAIN + pre-execution rejection
├── persistence/
│   ├── __init__.py
│   ├── database.py            # sqlite3 wrapper: blob storage, WAL table, query log, schema tables
│   ├── serializer.py          # Graph → sqlite blobs (numpy tobytes)
│   └── deserializer.py        # sqlite blobs → graph (numpy frombuffer)
└── schema.py                  # Kind registry + write validation
```

```
tests/
├── test_store.py              # Node/edge CRUD, index operations
├── test_edges.py              # CSR build, per-type matrices, combination cache
├── test_strings.py            # String intern/lookup round-trip
├── test_snapshot.py           # Swap, concurrent read during write
├── test_path.py               # Bidirectional BFS, edge cases
├── test_memory.py             # Ceiling enforcement, estimation accuracy
├── test_schema.py             # Kind registration, validation on write
├── test_dsl_parser.py         # One test per grammar production
├── test_dsl_user_reads.py     # All read query forms end-to-end
├── test_dsl_user_writes.py    # Mutations, batches, rollback, safety rules
├── test_dsl_relations.py      # PATH, MATCH, ANCESTORS, degree queries
├── test_dsl_system.py         # SYS queries: STATS, EXPLAIN, schema ops
├── test_cost_estimator.py     # Frontier estimation, rejection thresholds
├── test_persistence.py        # Save/load round-trip, corruption detection
├── test_wal.py                # WAL append, replay, crash recovery
├── test_query_log.py          # Slow/frequent/failed query retrieval
└── test_integration.py        # Full workflows: create → mutate → query → persist → reload → verify
```

```
benchmarks/
├── bench_reads.py             # All read operations at 170K/800K nodes
├── bench_writes.py            # Single + batch mutations, CSR rebuild
├── bench_match.py             # MATCH patterns at varying frontier sizes
├── bench_persistence.py       # Checkpoint + load at varying graph sizes
└── generate_graph.py          # Synthetic graph generator (configurable size/density)
```

```
docs/
├── USER_DSL.md                # Complete user DSL reference with examples
├── SYSTEM_DSL.md              # Complete system DSL reference with examples
├── QUICK_REFERENCE.md         # Single-page cheat sheet (both DSLs)
├── ARCHITECTURE.md            # Internals: CSR layout, snapshot swap, WAL, memory model
└── LLM_PROMPT_SNIPPET.md      # Copy-paste system prompt block for LLM consumers
```

---

## 5. Complete DSL Specification

### 5.1 User DSL — Graph Data Operations

#### 5.1.1 Reads

```
# Single node by ID
NODE "id"

# Filter nodes
NODES WHERE kind = "function" LIMIT 10
NODES WHERE (kind = "function" OR kind = "method") AND NOT name = "internal"
NODES WHERE INDEGREE > 10
NODES WHERE OUTDEGREE kind = "calls" > 5
NODES WHERE docstring = NULL

# Edges
EDGES FROM "id" WHERE kind = "calls"
EDGES TO "id" WHERE kind = "imports"

# Traversal
TRAVERSE FROM "id" DEPTH 3 WHERE kind = "calls"
SUBGRAPH FROM "id" DEPTH 2
```

#### 5.1.2 Relations

```
# Path finding
PATH FROM "a" TO "b" MAX_DEPTH 5 WHERE kind = "calls"
PATHS FROM "a" TO "b" MAX_DEPTH 4
SHORTEST PATH FROM "a" TO "b" WHERE kind = "calls"
DISTANCE FROM "a" TO "b" MAX_DEPTH 10

# Ancestry
ANCESTORS OF "a" DEPTH 3 WHERE kind = "calls"
DESCENDANTS OF "a" DEPTH 3

# Intersection
COMMON NEIGHBORS OF "a" AND "b" WHERE kind = "calls"

# Pattern matching
MATCH ("a") -[kind = "calls"]-> (b) -[kind = "imports"]-> (c)
MATCH (x WHERE kind = "class") -[kind = "extends"]-> ("base_id")
MATCH ("a") -[kind = "calls"]-> (b WHERE kind = "function") -[kind = "calls"]-> (c) LIMIT 50
```

#### 5.1.3 Writes

```
# Nodes
CREATE NODE "id" kind = "function" name = "foo" file = "src/main.py"
UPDATE NODE "id" SET name = "bar" docstring = "Parses input"
UPSERT NODE "id" kind = "function" name = "foo"
DELETE NODE "id"
DELETE NODES WHERE kind = "test" AND file = "old.py"

# Edges
CREATE EDGE "src" -> "tgt" kind = "calls" confidence = 0.9
DELETE EDGE "src" -> "tgt" kind = "calls"
DELETE EDGES FROM "src" WHERE kind = "calls"
DELETE EDGES TO "tgt" WHERE kind = "imports"

# Counters
INCREMENT NODE "id" hits BY 1
```

#### 5.1.4 Batch

```
BEGIN
  DELETE NODES WHERE file = "src/main.py"
  CREATE NODE "a1" kind = "function" name = "new_func" file = "src/main.py"
  CREATE NODE "a2" kind = "function" name = "helper" file = "src/main.py"
  CREATE EDGE "a1" -> "a2" kind = "calls"
  CREATE EDGE "a1" -> "b2" kind = "imports"
COMMIT
```

#### 5.1.5 Filter Syntax

```
# Operators
=  !=  >  <  >=  <=

# Logic (OR requires parentheses when mixed with AND)
AND  OR  NOT

# Null
field = NULL
NOT field = NULL

# Degree pseudo-fields
INDEGREE > 10
OUTDEGREE > 5
INDEGREE kind = "calls" > 3
OUTDEGREE kind = "imports" >= 2
```

#### 5.1.6 Write Safety Rules

- DELETE NODES without WHERE is a syntax error. Prevents accidental graph wipe.
- CREATE NODE with existing ID fails. Use UPSERT for create-or-update.
- DELETE NODE always cascades edges. All edges touching deleted node are removed.
- CREATE EDGE with identical source + target + kind + data is rejected as duplicate.
- INCREMENT on non-numeric field is an error.
- BEGIN/COMMIT is atomic. If any statement in the block fails, the entire block rolls back.
- Bare mutations outside BEGIN/COMMIT are each their own implicit transaction.

### 5.2 System DSL — Introspection, Diagnostics, Schema, Maintenance

All system queries are prefixed with `SYS`.

#### 5.2.1 Introspection

```
SYS STATS
SYS STATS NODES
SYS STATS EDGES
SYS STATS MEMORY
SYS STATS WAL
```

STATS returns: node count, edge count, edge counts by type, memory usage in bytes, ceiling in bytes, WAL entry count, last checkpoint timestamp, uptime.

```
SYS KINDS
SYS EDGE KINDS
SYS DESCRIBE NODE "function"
SYS DESCRIBE EDGE "calls"
```

KINDS lists all registered node kinds. DESCRIBE returns the kind's required/optional fields and endpoint constraints.

#### 5.2.2 Diagnostics

```
SYS SLOW QUERIES LIMIT 10
SYS SLOW QUERIES SINCE "2025-03-01T00:00:00" LIMIT 20
SYS FREQUENT QUERIES LIMIT 10
SYS FAILED QUERIES LIMIT 10
```

Queries the internal query log (sqlite table). SLOW returns queries ordered by elapsed time descending. FREQUENT returns queries ordered by occurrence count. FAILED returns queries that raised errors.

```
SYS EXPLAIN MATCH ("abc") -[kind = "calls"]-> (b) -[kind = "imports"]-> (c)
SYS EXPLAIN TRAVERSE FROM "abc" DEPTH 5 WHERE kind = "calls"
SYS EXPLAIN NODES WHERE kind = "function" AND file = "src/main.py"
```

EXPLAIN runs the cost estimator without executing. Returns: estimated frontier size per hop (for MATCH/TRAVERSE), whether index or full scan will be used (for NODES WHERE), estimated latency, and ACCEPT/REJECT verdict.

#### 5.2.3 Schema Management

```
SYS REGISTER NODE KIND "function" REQUIRED name OPTIONAL docstring, file, line
SYS REGISTER NODE KIND "class" REQUIRED name OPTIONAL bases, file, line
SYS REGISTER NODE KIND "method" REQUIRED name, parent OPTIONAL docstring, file, line

SYS REGISTER EDGE KIND "calls" FROM "function", "method" TO "function", "method"
SYS REGISTER EDGE KIND "imports" FROM "file" TO "file", "module"
SYS REGISTER EDGE KIND "extends" FROM "class" TO "class"

SYS UNREGISTER NODE KIND "function"
SYS UNREGISTER EDGE KIND "calls"
```

Schema is optional. If no kinds are registered, the DB is schema-free. When kinds are registered, writes are validated (CREATE NODE checks required fields, CREATE EDGE checks endpoint kind constraints). Reads are never affected by schema.

#### 5.2.4 Maintenance

```
SYS CHECKPOINT
SYS REBUILD INDICES
SYS CLEAR LOG
SYS CLEAR CACHE
SYS WAL STATUS
SYS WAL REPLAY
```

CHECKPOINT forces a full persist to sqlite. REBUILD INDICES forces CSR and secondary index rebuild. CLEAR LOG deletes the query log. CLEAR CACHE invalidates the plan cache and edge matrix combination cache. WAL STATUS returns entry count and size. WAL REPLAY manually replays WAL (normally automatic on startup).

---

## 6. Grammar (Lark Format)

```lark
start: statement
statement: system_query | user_query

// =============================================
// USER DSL
// =============================================

user_query: read_query | write_query | batch

// --- Reads ---
read_query: node_q | nodes_q | edges_q | traverse_q | subgraph_q
          | path_q | paths_q | shortest_q | distance_q
          | ancestors_q | descendants_q | common_q | match_q

node_q:        "NODE" STRING
nodes_q:       "NODES" [where_clause] [limit_clause]
edges_q:       "EDGES" direction STRING [where_clause]
traverse_q:    "TRAVERSE" "FROM" STRING "DEPTH" NUMBER [where_clause]
subgraph_q:    "SUBGRAPH" "FROM" STRING "DEPTH" NUMBER
path_q:        "PATH" "FROM" STRING "TO" STRING "MAX_DEPTH" NUMBER [where_clause]
paths_q:       "PATHS" "FROM" STRING "TO" STRING "MAX_DEPTH" NUMBER [where_clause]
shortest_q:    "SHORTEST" "PATH" "FROM" STRING "TO" STRING [where_clause]
distance_q:    "DISTANCE" "FROM" STRING "TO" STRING "MAX_DEPTH" NUMBER
ancestors_q:   "ANCESTORS" "OF" STRING "DEPTH" NUMBER [where_clause]
descendants_q: "DESCENDANTS" "OF" STRING "DEPTH" NUMBER [where_clause]
common_q:      "COMMON" "NEIGHBORS" "OF" STRING "AND" STRING [where_clause]
match_q:       "MATCH" pattern [limit_clause]

direction: "FROM" | "TO"

// --- Pattern matching ---
pattern: step (arrow step)+
step: STRING | "(" IDENTIFIER [step_where] ")"
step_where: "WHERE" expr
arrow: "-[" expr "]->"

// --- Writes ---
write_query: create_node | update_node | upsert_node
           | delete_node | delete_nodes
           | create_edge | delete_edge | delete_edges
           | increment

create_node:  "CREATE" "NODE" STRING field_pairs
update_node:  "UPDATE" "NODE" STRING "SET" field_pairs
upsert_node:  "UPSERT" "NODE" STRING field_pairs
delete_node:  "DELETE" "NODE" STRING
delete_nodes: "DELETE" "NODES" where_clause

create_edge:  "CREATE" "EDGE" STRING "->" STRING field_pairs
delete_edge:  "DELETE" "EDGE" STRING "->" STRING [where_clause]
delete_edges: "DELETE" "EDGES" direction STRING [where_clause]

increment:    "INCREMENT" "NODE" STRING IDENTIFIER "BY" NUMBER

batch: "BEGIN" NEWLINE (write_query NEWLINE)* "COMMIT"

field_pairs: (IDENTIFIER "=" value)*

// --- Filters ---
where_clause: "WHERE" expr
expr: and_expr ("OR" and_expr)*
and_expr: term ("AND" term)*
term: "NOT" atom | atom
atom: condition | "(" expr ")" | degree_condition
condition: IDENTIFIER OP value
degree_condition: degree_kw [IDENTIFIER] OP NUMBER
degree_kw: "INDEGREE" | "OUTDEGREE"

OP: "=" | "!=" | ">" | "<" | ">=" | "<="
value: STRING | NUMBER | "NULL"
limit_clause: "LIMIT" NUMBER

// =============================================
// SYSTEM DSL
// =============================================

system_query: "SYS" sys_command

sys_command: sys_stats | sys_kinds | sys_edge_kinds
           | sys_describe | sys_slow | sys_frequent | sys_failed
           | sys_explain | sys_register_node_kind | sys_register_edge_kind
           | sys_unregister | sys_checkpoint | sys_rebuild
           | sys_clear | sys_wal

sys_stats:      "STATS" [STATS_TARGET]
STATS_TARGET:   "NODES" | "EDGES" | "MEMORY" | "WAL"

sys_kinds:      "KINDS"
sys_edge_kinds: "EDGE" "KINDS"
sys_describe:   "DESCRIBE" ("NODE" | "EDGE") STRING

sys_slow:       "SLOW" "QUERIES" [since_clause] [limit_clause]
sys_frequent:   "FREQUENT" "QUERIES" [limit_clause]
sys_failed:     "FAILED" "QUERIES" [limit_clause]
since_clause:   "SINCE" STRING

sys_explain:    "EXPLAIN" read_query

sys_register_node_kind: "REGISTER" "NODE" "KIND" STRING "REQUIRED" ident_list ["OPTIONAL" ident_list]
sys_register_edge_kind: "REGISTER" "EDGE" "KIND" STRING "FROM" string_list "TO" string_list
sys_unregister:         "UNREGISTER" ("NODE" | "EDGE") "KIND" STRING

sys_checkpoint: "CHECKPOINT"
sys_rebuild:    "REBUILD" "INDICES"
sys_clear:      "CLEAR" ("LOG" | "CACHE")
sys_wal:        "WAL" ("STATUS" | "REPLAY")

ident_list: IDENTIFIER ("," IDENTIFIER)*
string_list: STRING ("," STRING)*

// =============================================
// TERMINALS
// =============================================

STRING: "\"" /[^"\\]*(\\.[^"\\]*)*/ "\""
NUMBER: /\-?[0-9]+(\.[0-9]+)?/
IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/
NEWLINE: /\n/

%import common.WS
%ignore WS
```

---

## 7. Core Data Model

### 7.1 String Table

File: `strings.py`

Bidirectional mapping between strings and int32 IDs. Every string that enters the store (node IDs, field names, field values, edge types, file paths) is interned.

- `intern(s: str) → int` — returns existing ID or assigns next sequential.
- `lookup(i: int) → str` — reverse lookup.
- Internal: `list[str]` for int→str, `dict[str, int]` for str→int.
- String IDs are not stable across serialization. The table is rebuilt on load.
- Used by: every other component. This is the foundation.

### 7.2 Node Storage

File: `store.py`

Hybrid storage: numpy arrays for filterable fixed fields, Python list for flexible data.

Per node:
- `node_id: int32` — internal ID from string table, mapped from caller's string ID.
- `node_kind: uint8` — enum for node type. Kinds are registered at runtime, mapped to uint8 by the string table.
- `node_data: list[dict]` — arbitrary key-value data, indexed by internal node ID.

`node_kind` is a numpy array (vectorized filtering at C speed). `node_data` is a Python list (flexible schema, no numpy dtype constraints).

Operations:
- `put_node(id: str, kind: str, data: dict)` — intern ID, assign slot, populate arrays.
- `get_node(id: str) → dict | None` — intern ID, lookup.
- `delete_node(id: str)` — tombstone slot, cascade-delete edges.
- Tombstoned slots are tracked in a freelist. Reused on next insert. Compact on checkpoint if fragmentation exceeds 20%.

Secondary indices:
- `add_index(field: str)` — builds `dict[value → list[int]]` over node_data.
- Maintained incrementally on put/delete.
- Used by executor to short-circuit full scans when WHERE matches an indexed field.

### 7.3 Edge Storage — Per-Type CSR Matrices

File: `edges.py`

Each edge type (calls, imports, extends, uses_type) is stored as a separate `scipy.sparse.csr_matrix`. This gives us:

- Type-filtered neighbor lookup: one CSR slice, no filtering needed.
- Combined-type queries: sparse matrix addition.
- Traversal on specific types: pass the right matrix to scipy.sparse.csgraph.
- MATCH patterns: sparse matrix-vector multiply, one matrix per hop.

```python
class EdgeMatrices:
    def __init__(self):
        self._typed: dict[str, csr_matrix] = {}     # per-type CSR
        self._combined_all: csr_matrix = None        # precomputed union
        self._cache: dict[frozenset, csr_matrix] = {} # combination cache
        self._edge_data: dict[str, list[dict]] = {}  # per-type edge data lists
        self._transpose_cache: dict[str, csr_matrix] = {} # CSC for incoming edges

    def get(self, edge_types: set[str] | None) -> csr_matrix:
        """Get CSR matrix for query. None = all types."""
        if edge_types is None:
            return self._combined_all
        if len(edge_types) == 1:
            return self._typed[next(iter(edge_types))]
        key = frozenset(edge_types)
        if key not in self._cache:
            self._cache[key] = sum(self._typed[t] for t in edge_types)
        return self._cache[key]

    def get_transpose(self, edge_type: str) -> csr_matrix:
        """Get transposed CSR (CSC equivalent) for incoming edge queries."""
        if edge_type not in self._transpose_cache:
            self._transpose_cache[edge_type] = self._typed[edge_type].T.tocsr()
        return self._transpose_cache[edge_type]

    def rebuild(self, edges_by_type: dict[str, list[tuple]], num_nodes: int):
        """Full rebuild from edge lists. Called after COMMIT."""
        self._typed.clear()
        self._cache.clear()
        self._transpose_cache.clear()
        for etype, edge_list in edges_by_type.items():
            if edge_list:
                sources, targets = zip(*[(s, t) for s, t, _ in edge_list])
                data = [d for _, _, d in edge_list]
                self._typed[etype] = csr_matrix(
                    (np.ones(len(sources), dtype=np.int8),
                     (np.array(sources, dtype=np.int32),
                      np.array(targets, dtype=np.int32))),
                    shape=(num_nodes, num_nodes)
                )
                self._edge_data[etype] = data
        self._combined_all = sum(self._typed.values()) if self._typed else None
```

Degree arrays precomputed on rebuild:
```python
self._out_degree = {etype: np.diff(m.indptr) for etype, m in self._typed.items()}
self._out_degree_all = np.diff(self._combined_all.indptr) if self._combined_all is not None else None
# Incoming degree from transpose
self._in_degree = {etype: np.diff(self.get_transpose(etype).indptr) for etype in self._typed}
```

Cache invalidation: `_cache` and `_transpose_cache` are cleared on every `rebuild()`. Between rebuilds, they accumulate lazily.

### 7.4 Snapshot Swap

File: `snapshot.py`

```python
@dataclass
class GraphSnapshot:
    string_table: StringTable
    node_ids: np.ndarray          # int32, caller string IDs in slot order
    node_kinds: np.ndarray        # uint8
    node_data: list[dict]
    node_tombstones: set[int]     # tombstoned slot indices
    edge_matrices: EdgeMatrices
    secondary_indices: dict[str, dict]
    id_to_slot: dict[int, int]    # interned string ID → array slot

class SnapshotManager:
    def __init__(self):
        self._current: GraphSnapshot = None

    @property
    def current(self) -> GraphSnapshot:
        return self._current

    def swap(self, new_snapshot: GraphSnapshot):
        """Atomic swap. Python GIL makes pointer assignment atomic."""
        old = self._current
        self._current = new_snapshot
        del old  # explicit cleanup of heavy arrays
```

All read operations go through `manager.current`. Mutations build a working copy. On COMMIT, the working copy's CSR matrices are rebuilt, then `manager.swap()` is called. Readers in progress see the old snapshot. New readers see the new one. No locks.

### 7.5 Memory Ceiling

File: `memory.py`

```python
BYTES_PER_NODE = 330      # numpy arrays + Python dict + string table entry
BYTES_PER_EDGE = 20       # CSR entries across typed matrices

def estimate(node_count: int, edge_count: int) -> int:
    return (node_count * BYTES_PER_NODE) + (edge_count * BYTES_PER_EDGE)

def check_ceiling(current_nodes, current_edges, added_nodes, added_edges, ceiling_bytes):
    projected = estimate(current_nodes + added_nodes, current_edges + added_edges)
    if projected > ceiling_bytes:
        raise CeilingExceeded(
            current_mb=estimate(current_nodes, current_edges) // 1_000_000,
            ceiling_mb=ceiling_bytes // 1_000_000,
            operation=f"add {added_nodes} nodes, {added_edges} edges"
        )
```

Checked on every `put_node` and `put_edge`. Also checked at startup during bulk load — check every 1000 insertions, fail fast.

`memory_usage() → int` computes actual usage by summing array `.nbytes` and `sys.getsizeof` on Python structures. Expensive, called only on SYS STATS MEMORY or checkpoint, not per-query.

---

## 8. Traversal and Path Finding

### 8.1 BFS / DFS — scipy.sparse.csgraph

TRAVERSE, ANCESTORS, DESCENDANTS all delegate to scipy:

```python
from scipy.sparse.csgraph import breadth_first_order, depth_first_order

def traverse_bfs(matrix: csr_matrix, start: int, max_depth: int) -> np.ndarray:
    order, predecessors = breadth_first_order(matrix, start, directed=True, return_predecessors=True)
    # Filter by depth using predecessor chain
    depths = compute_depths(predecessors, start)
    return order[depths <= max_depth]
```

For ANCESTORS, use the transposed matrix (incoming edges):
```python
ancestors = traverse_bfs(edge_matrices.get_transpose("calls"), start, max_depth)
```

### 8.2 Point-to-Point Shortest Path — Custom Bidirectional BFS

File: `path.py`

scipy.sparse.csgraph.shortest_path computes distances to ALL nodes. For point-to-point, we use bidirectional BFS with early termination.

Core algorithm:
1. Maintain two frontiers: forward (from source) and backward (from target, on transposed matrix).
2. Each iteration, expand the smaller frontier (key optimization — keeps total work O(b^(d/2))).
3. After expansion, check intersection of the two visited sets.
4. If intersection found, reconstruct path from predecessor dicts.
5. If max_depth exhausted with no intersection, return None.

Performance:
- 800K nodes, path found at depth 3: ~20-50μs
- 800K nodes, path found at depth 6: ~100-300μs
- 800K nodes, no path, depth 10 exhausted: ~500μs-1ms

Versus scipy full Dijkstra: 10-100× faster for the common case.

### 8.3 MATCH — Sparse Matrix-Vector Multiplication

Multi-hop pattern queries use repeated sparse matrix-vector multiplication, entirely within scipy/numpy (no Python per-element loops).

```python
def execute_match(start_id, hops, matrices, node_arrays):
    # One-hot vector for bound start
    vec = csr_matrix(([1], ([0], [start_id])), shape=(1, num_nodes))
    bindings = []

    for edge_type, node_filter in hops:
        # Sparse mat-vec multiply — Cython speed
        vec = vec @ matrices.get({edge_type})

        # Apply node filter as element-wise mask
        if node_filter:
            mask = evaluate_node_filter(node_filter, node_arrays)
            vec = vec.multiply(mask)

        bindings.append(vec.nonzero()[1])

    return bindings
```

For variable binding reconstruction (which `b` led to which `c`), iterate over nonzero entries of intermediate vectors — Python loop but over the frontier (small), not the graph (large).

### 8.4 Cost Estimator

File: `dsl/cost_estimator.py`

Before executing MATCH or TRAVERSE, estimate frontier growth:

```python
def estimate_match_cost(pattern, matrices):
    frontier_size = 1 if pattern.start_is_bound else estimate_filter_selectivity(pattern.start_filter)

    for edge_type, node_filter in pattern.hops:
        avg_degree = matrices.get({edge_type}).nnz / matrices.get({edge_type}).shape[0]
        frontier_size *= avg_degree
        if node_filter:
            frontier_size *= estimate_selectivity(node_filter)
        if frontier_size > 100_000:
            return CostEstimate(rejected=True, reason="frontier exceeds 100K at this hop")

    return CostEstimate(rejected=False, estimated_frontier=frontier_size)
```

SYS EXPLAIN calls this and returns the plan without executing. The executor calls this before execution and rejects queries that exceed the threshold. The threshold is configurable (default 100K frontier size). The 100ms hard timeout is the backstop for estimates that are wrong.

---

## 9. DSL Parser and Executor

### 9.1 Parser

File: `dsl/parser.py`

Lark in LALR(1) mode. The grammar from Section 6 is loaded once at import time. Lark auto-generates the parse tree.

```python
from lark import Lark, Transformer

_grammar = open(Path(__file__).parent / "grammar.lark").read()
_parser = Lark(_grammar, parser="lalr", start="start")

def parse(query: str) -> ASTNode:
    tree = _parser.parse(query)
    return DSLTransformer().transform(tree)
```

### 9.2 Transformer

File: `dsl/transformer.py`

Lark Transformer converts the parse tree into typed AST nodes:

```python
class DSLTransformer(Transformer):
    def node_q(self, args):
        return NodeQuery(id=args[0])

    def nodes_q(self, args):
        where = next((a for a in args if isinstance(a, WhereClause)), None)
        limit = next((a for a in args if isinstance(a, LimitClause)), None)
        return NodesQuery(where=where, limit=limit)

    def match_q(self, args):
        pattern = args[0]
        limit = next((a for a in args[1:] if isinstance(a, LimitClause)), None)
        return MatchQuery(pattern=pattern, limit=limit)

    def expr(self, args):
        if len(args) == 1:
            return args[0]
        return OrExpr(operands=args)

    # ... one method per grammar production
```

### 9.3 Plan Cache

LRU dict keyed by whitespace-normalized query string:

```python
class PlanCache:
    def __init__(self, maxsize=256):
        self._cache = OrderedDict()
        self._maxsize = maxsize

    def get_or_parse(self, query: str) -> ASTNode:
        key = " ".join(query.split())  # normalize whitespace
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        ast = parse(query)
        self._cache[key] = ast
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
        return ast
```

Invalidated on schema change (SYS REGISTER/UNREGISTER). Not invalidated on data change — plans don't embed data.

### 9.4 Executor (User)

File: `dsl/executor.py`

Maps each AST node type to store operations. No optimization pass.

**NODE:** dict lookup → return.

**NODES WHERE:** Compile WHERE into a numpy boolean mask builder. If WHERE references an indexed field with `=`, start from the index. Otherwise, vectorized scan over numpy arrays. Apply LIMIT after mask.

**EDGES FROM/TO:** Resolve node ID. Get CSR matrix (per-type or combined). Slice `indptr`/`indices`. Return edges.

**TRAVERSE:** Get matrix, call `breadth_first_order`, filter by depth.

**SHORTEST PATH:** Get matrix, call custom bidirectional BFS from `path.py`.

**PATH:** BFS from source, stop when target found, return first path.

**PATHS:** BFS collecting all paths up to MAX_DEPTH. Cap at 100 results.

**DISTANCE:** Bidirectional BFS, return hop count or -1.

**ANCESTORS/DESCENDANTS:** BFS on transposed/normal matrix respectively.

**COMMON NEIGHBORS:** CSR slice for both nodes, `np.intersect1d` on neighbor arrays.

**MATCH:** Cost estimator first. If accepted, sparse matrix-vector multiply chain. Cap at 5 hops. Cap results at 1000.

**CREATE NODE:** Check ceiling. Check schema (if registered). Intern ID. Append to arrays. Log to WAL.

**UPDATE NODE:** Verify node exists. Update data dict. Update secondary indices. Log to WAL.

**UPSERT NODE:** CREATE or UPDATE depending on existence.

**DELETE NODE:** Tombstone node. Remove from all edge lists. Mark for CSR rebuild. Log to WAL.

**DELETE NODES WHERE:** Build mask, apply to all matching nodes. WHERE required (syntax error without).

**CREATE EDGE:** Check ceiling. Check schema endpoint constraints (if registered). Check duplicate. Add to pending edge list. Log to WAL.

**DELETE EDGE/EDGES:** Mark edges for removal. Mark for CSR rebuild. Log to WAL.

**INCREMENT:** Verify node exists. Verify field is numeric. Increment in data dict. Log to WAL.

**BEGIN/COMMIT:** Clone current snapshot to working copy. Apply all mutations to working copy. If any fail, discard (rollback). If all succeed, rebuild CSR on working copy, swap snapshot. Log entire block as one WAL entry.

### 9.5 Executor (System)

File: `dsl/executor_system.py`

**SYS STATS:** Query store for counts, memory, WAL size, last checkpoint time. Return as structured Result.

**SYS DESCRIBE:** Query schema registry. Return kind fields and constraints.

**SYS KINDS / EDGE KINDS:** List registered kinds from schema registry.

**SYS SLOW/FREQUENT/FAILED QUERIES:** Query sqlite query_log table. Return list of entries.

**SYS EXPLAIN:** Run cost estimator on the given read query. Return plan without executing.

**SYS REGISTER NODE/EDGE KIND:** Add to schema registry. Persist to sqlite schema table.

**SYS UNREGISTER:** Remove from schema registry. Remove from sqlite.

**SYS CHECKPOINT:** Force full persist.

**SYS REBUILD INDICES:** Force CSR rebuild + secondary index rebuild.

**SYS CLEAR LOG:** DELETE FROM query_log.

**SYS CLEAR CACHE:** Clear plan cache + edge matrix combination cache.

**SYS WAL STATUS:** Return WAL entry count and size from sqlite.

**SYS WAL REPLAY:** Read WAL entries, execute in order.

---

## 10. Persistence

### 10.1 sqlite Database Layout

File: `persistence/database.py`

Single sqlite file at `{store_path}/graphstore.db`.

```sql
-- Graph data blobs
CREATE TABLE IF NOT EXISTS blobs (
    key TEXT PRIMARY KEY,
    data BLOB,
    dtype TEXT           -- numpy dtype string for array blobs
);

-- WAL: DSL mutations since last checkpoint
CREATE TABLE IF NOT EXISTS wal (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    statement TEXT NOT NULL
);

-- Query log: performance observability
CREATE TABLE IF NOT EXISTS query_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    query TEXT NOT NULL,
    elapsed_us INTEGER NOT NULL,
    result_count INTEGER NOT NULL,
    error TEXT
);

-- Schema: kind definitions
CREATE TABLE IF NOT EXISTS schema_node_kinds (
    kind TEXT PRIMARY KEY,
    required TEXT NOT NULL,     -- JSON array of field names
    optional TEXT NOT NULL      -- JSON array of field names
);

CREATE TABLE IF NOT EXISTS schema_edge_kinds (
    kind TEXT PRIMARY KEY,
    from_kinds TEXT NOT NULL,   -- JSON array of kind names
    to_kinds TEXT NOT NULL      -- JSON array of kind names
);

-- Metadata
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
-- Stores: schema_version, last_checkpoint_time, store_creation_time
```

sqlite configured with:
```python
conn.execute("PRAGMA journal_mode=WAL")       # WAL mode for concurrent reads during write
conn.execute("PRAGMA synchronous=NORMAL")     # fsync on checkpoint, not every commit
conn.execute("PRAGMA foreign_keys=OFF")       # not using foreign keys
conn.execute("PRAGMA busy_timeout=5000")      # wait up to 5s if locked
```

### 10.2 Serializer

File: `persistence/serializer.py`

Checkpoint writes the full graph state to sqlite blob table:

```python
def checkpoint(snapshot: GraphSnapshot, conn: sqlite3.Connection):
    with conn:
        # String table
        conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                     ("strings", json.dumps(snapshot.string_table.strings).encode(), "json"))

        # Node arrays
        conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                     ("node_kinds", snapshot.node_kinds.tobytes(), str(snapshot.node_kinds.dtype)))

        # Node data
        conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                     ("node_data", msgpack_not_needed_json_dumps(snapshot.node_data).encode(), "json"))
        # Actually, use pickle protocol 5 for node_data since it's a list of dicts.
        # Or just json. json is safer and inspectable.

        # Edge matrices: one blob per type
        for etype, matrix in snapshot.edge_matrices._typed.items():
            conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                         (f"edges:{etype}:indptr", matrix.indptr.tobytes(), str(matrix.indptr.dtype)))
            conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                         (f"edges:{etype}:indices", matrix.indices.tobytes(), str(matrix.indices.dtype)))
            conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                         (f"edges:{etype}:data", matrix.data.tobytes(), str(matrix.data.dtype)))
            conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                         (f"edges:{etype}:shape", json.dumps(list(matrix.shape)).encode(), "json"))

        # Edge data (per-type, json)
        for etype, data_list in snapshot.edge_matrices._edge_data.items():
            conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                         (f"edge_data:{etype}", json.dumps(data_list).encode(), "json"))

        # Tombstones
        conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                     ("tombstones", json.dumps(list(snapshot.node_tombstones)).encode(), "json"))

        # Secondary index definitions (which fields, not the data)
        conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                     ("indexed_fields", json.dumps(list(snapshot.secondary_indices.keys())).encode(), "json"))

        # Version
        conn.execute("INSERT OR REPLACE INTO metadata VALUES (?, ?)",
                     ("schema_version", str(SCHEMA_VERSION)))
        conn.execute("INSERT OR REPLACE INTO metadata VALUES (?, ?)",
                     ("last_checkpoint_time", str(time.time())))

        # Clear WAL after successful checkpoint
        conn.execute("DELETE FROM wal")
```

All within a single sqlite transaction. Either everything commits or nothing does. sqlite handles atomicity.

### 10.3 Deserializer

File: `persistence/deserializer.py`

Startup loads from sqlite:

```python
def load(conn: sqlite3.Connection) -> GraphSnapshot:
    # Check version
    row = conn.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
    if row is None or int(row[0]) != SCHEMA_VERSION:
        raise VersionMismatch(found=row[0] if row else None, expected=SCHEMA_VERSION)

    # Rebuild string table
    strings_json = conn.execute("SELECT data FROM blobs WHERE key='strings'").fetchone()
    string_table = StringTable.from_list(json.loads(strings_json[0]))

    # Rebuild node arrays
    kinds_row = conn.execute("SELECT data, dtype FROM blobs WHERE key='node_kinds'").fetchone()
    node_kinds = np.frombuffer(kinds_row[0], dtype=np.dtype(kinds_row[1]))

    # Rebuild node data
    data_row = conn.execute("SELECT data FROM blobs WHERE key='node_data'").fetchone()
    node_data = json.loads(data_row[0])

    # Rebuild edge matrices
    edge_matrices = EdgeMatrices()
    edge_type_rows = conn.execute(
        "SELECT DISTINCT SUBSTR(key, 7, INSTR(SUBSTR(key, 7), ':') - 1) "
        "FROM blobs WHERE key LIKE 'edges:%'"
    ).fetchall()

    for (etype,) in edge_type_rows:
        indptr = np.frombuffer(fetch_blob(conn, f"edges:{etype}:indptr"), dtype=np.int32)
        indices = np.frombuffer(fetch_blob(conn, f"edges:{etype}:indices"), dtype=np.int32)
        data = np.frombuffer(fetch_blob(conn, f"edges:{etype}:data"), dtype=np.int8)
        shape = tuple(json.loads(fetch_blob(conn, f"edges:{etype}:shape")))
        edge_matrices._typed[etype] = csr_matrix((data, indices, indptr), shape=shape)

    edge_matrices._combined_all = sum(edge_matrices._typed.values()) if edge_matrices._typed else None

    # Rebuild secondary indices
    indexed_fields = json.loads(fetch_blob(conn, "indexed_fields"))
    secondary_indices = rebuild_indices(indexed_fields, node_data)

    # Rebuild tombstones
    tombstones = set(json.loads(fetch_blob(conn, "tombstones")))

    return GraphSnapshot(
        string_table=string_table,
        node_kinds=node_kinds,
        node_data=node_data,
        node_tombstones=tombstones,
        edge_matrices=edge_matrices,
        secondary_indices=secondary_indices,
        id_to_slot=rebuild_id_to_slot(string_table, tombstones),
    )
```

### 10.4 WAL

Mutations are logged to the sqlite `wal` table before being applied in-memory.

Write flow:
```
1. Parse DSL statement.
2. Validate (dry-run checks).
3. INSERT INTO wal (timestamp, statement) — within sqlite transaction.
4. Apply to in-memory graph.
5. Return success.
```

BEGIN/COMMIT blocks are stored as a single WAL entry (all statements joined by newline).

Replay on startup:
```python
def replay_wal(conn, executor):
    rows = conn.execute("SELECT statement FROM wal ORDER BY seq").fetchall()
    for (statement,) in rows:
        executor.execute(statement, replay_mode=True)
    if rows:
        checkpoint(executor.snapshot, conn)  # clean state
```

Replay mode: CREATE is treated as UPSERT (idempotent replay). DELETE of nonexistent node is a no-op.

Checkpoint trigger: when WAL exceeds 10MB or 50K entries. Also on clean shutdown.

WAL hard limit: if WAL exceeds 50MB and checkpoint keeps failing (disk full), refuse new writes.

### 10.5 Query Log

Every `execute()` call is logged:

```python
def execute(self, query: str) -> Result:
    start = time.perf_counter_ns()
    try:
        result = self._execute_internal(query)
        elapsed = (time.perf_counter_ns() - start) // 1000
        self._conn.execute(
            "INSERT INTO query_log (timestamp, query, elapsed_us, result_count, error) VALUES (?,?,?,?,NULL)",
            [time.time(), query, elapsed, result.count]
        )
        return result
    except Exception as e:
        elapsed = (time.perf_counter_ns() - start) // 1000
        self._conn.execute(
            "INSERT INTO query_log (timestamp, query, elapsed_us, result_count, error) VALUES (?,?,?,?,?)",
            [time.time(), query, elapsed, 0, str(e)]
        )
        raise
```

Log rotation: delete entries older than 7 days on checkpoint. Or on SYS CLEAR LOG.

### 10.6 Startup Flow

```
1. Open sqlite database at {store_path}/graphstore.db.
2. Create tables if not exist.
3. Check metadata for schema_version.
   - Missing → empty store, fresh start.
   - Mismatch → raise VersionMismatch. Caller decides (rebuild or error).
4. Load graph from blobs table.
5. Load schema from schema tables.
6. Replay WAL entries.
   - If any entries replayed, trigger immediate checkpoint.
7. Build plan cache (empty).
8. Ready.
```

### 10.7 File Layout on Disk

```
{store_path}/
└── graphstore.db         # single sqlite file (contains blobs, WAL, query log, schema, metadata)
```

No lock file needed — sqlite handles file locking natively. No backup rotation needed — sqlite WAL mode provides crash recovery natively. No custom binary format — sqlite IS the format.

---

## 11. Schema System

File: `schema.py`

Optional type system. When kinds are registered, writes are validated.

```python
class SchemaRegistry:
    def __init__(self):
        self._node_kinds: dict[str, NodeKindDef] = {}
        self._edge_kinds: dict[str, EdgeKindDef] = {}

    def register_node_kind(self, kind: str, required: list[str], optional: list[str]):
        self._node_kinds[kind] = NodeKindDef(required=set(required), optional=set(optional))

    def register_edge_kind(self, kind: str, from_kinds: list[str], to_kinds: list[str]):
        self._edge_kinds[kind] = EdgeKindDef(from_kinds=set(from_kinds), to_kinds=set(to_kinds))

    def validate_node(self, kind: str, data: dict):
        if kind not in self._node_kinds:
            return  # unregistered kind, no validation
        defn = self._node_kinds[kind]
        missing = defn.required - set(data.keys())
        if missing:
            raise SchemaError(f"kind '{kind}' requires fields: {missing}")

    def validate_edge(self, kind: str, source_kind: str, target_kind: str):
        if kind not in self._edge_kinds:
            return
        defn = self._edge_kinds[kind]
        if source_kind not in defn.from_kinds:
            raise SchemaError(f"edge '{kind}' cannot originate from kind '{source_kind}'")
        if target_kind not in defn.to_kinds:
            raise SchemaError(f"edge '{kind}' cannot target kind '{target_kind}'")
```

Schema persists to sqlite schema tables. Loaded on startup. Modified via SYS REGISTER/UNREGISTER.

---

## 12. Public API

File: `__init__.py`

```python
class GraphStore:
    def __init__(self, path: str = None, ceiling_mb: int = 256,
                 allow_system_queries: bool = True):
        """
        path: directory for graphstore.db.
              If None, in-memory only (no persistence, no WAL, no query log).
        ceiling_mb: hard memory limit in MB. Raises CeilingExceeded on breach.
        allow_system_queries: if False, SYS queries raise PermissionError.
        """

    # Primary interface
    def execute(self, query: str) -> Result:
        """Execute a single DSL query (user or system). Returns Result."""

    def execute_batch(self, queries: list[str]) -> list[Result]:
        """Execute multiple queries. Each is independent (not transactional).
           For transactional writes, use BEGIN/COMMIT in a single execute() call."""

    # Persistence
    def checkpoint(self) -> None:
        """Force persist to disk. No-op if path is None."""

    def close(self) -> None:
        """Checkpoint + close sqlite connection. Always call on shutdown."""

    # Introspection (shortcuts, equivalent to SYS STATS)
    @property
    def node_count(self) -> int: ...
    @property
    def edge_count(self) -> int: ...
    @property
    def memory_usage(self) -> int: ...

    # Context manager
    def __enter__(self) -> "GraphStore": ...
    def __exit__(self, *args) -> None: ...  # calls close()
```

Both `execute("NODE \"abc\"")` and `execute("SYS STATS")` go through the same entry point. The router splits on the first token.

### 12.1 Result Format

```python
@dataclass
class Result:
    kind: str              # "node", "nodes", "edges", "path", "paths", "match",
                           # "subgraph", "distance", "stats", "plan", "schema",
                           # "log_entries", "ok", "error"
    data: Any              # varies by kind (see below)
    count: int             # number of items in data
    elapsed_us: int        # execution time in microseconds

    def to_dict(self) -> dict:
        """JSON-serializable representation."""

    def to_json(self) -> str:
        """Compact JSON string."""
```

Result.data by kind:
- `"node"` → dict (id, kind, + all data fields) or None
- `"nodes"` → list[dict]
- `"edges"` → list[dict] (source, target, kind, + edge data)
- `"path"` → list[str] (ordered node IDs) or None
- `"paths"` → list[list[str]]
- `"match"` → list[dict[str, str]] (variable bindings)
- `"subgraph"` → dict with "nodes" and "edges" lists
- `"distance"` → int or -1
- `"stats"` → dict (node_count, edge_count, memory_bytes, etc.)
- `"plan"` → dict (hops, estimated_frontier, accept/reject, reason)
- `"schema"` → dict (kind name, required, optional, constraints)
- `"log_entries"` → list[dict] (timestamp, query, elapsed_us, error)
- `"ok"` → None (mutation succeeded)
- `"error"` → str (error message)

### 12.2 Error Types

All subclass `GraphStoreError`:

- `QueryError(message, position, query)` — DSL parse or validation failure. Position is character offset. Message includes caret.
- `NodeNotFound(id)` — UPDATE/INCREMENT on missing node.
- `NodeExists(id)` — CREATE on existing node.
- `CeilingExceeded(current_mb, ceiling_mb, operation)` — memory limit hit.
- `VersionMismatch(found, expected)` — store version doesn't match.
- `SchemaError(message)` — write violates registered schema.
- `CostThresholdExceeded(estimated_frontier, threshold)` — MATCH/TRAVERSE too expensive.
- `BatchRollback(failed_statement, error)` — a statement within BEGIN/COMMIT failed, entire block rolled back.

---

## 13. Performance Targets

Measured at 800K nodes, 3.2M edges, near ceiling.

| Operation | Target p99 | Implementation |
|---|---|---|
| NODE by ID | <1μs | dict lookup |
| NODES WHERE (indexed field) | <100μs | dict lookup + gather |
| NODES WHERE (full scan) | <2ms | numpy vectorized mask |
| EDGES FROM/TO | <5μs | CSR indptr slice |
| TRAVERSE depth 3 | <50μs | scipy BFS (Cython) |
| TRAVERSE depth 5 | <500μs | scipy BFS |
| SHORTEST PATH (found at depth 3) | <50μs | custom bidir BFS |
| SHORTEST PATH (found at depth 6) | <300μs | custom bidir BFS |
| SHORTEST PATH (no path, depth 10) | <1ms | custom bidir BFS |
| MATCH 3-hop bound start | <50μs | sparse mat-vec multiply |
| MATCH 2-hop unbound+filter (1K start) | <2ms | sparse mat-vec |
| ANCESTORS/DESCENDANTS depth 3 | <50μs | scipy BFS on transpose |
| COMMON NEIGHBORS | <50μs | CSR slice + np.intersect1d |
| Degree filter | <200μs | precomputed np.diff + compare |
| DSL parse (cold) | <100μs | lark LALR |
| DSL parse (cached) | <2μs | dict lookup |
| CREATE NODE (single) | <100μs | array append + WAL write |
| BEGIN/COMMIT (100 mutations) | <80ms | mutations + CSR rebuild |
| WAL append | <100μs | sqlite INSERT |
| Checkpoint (full) | <500ms | sqlite blob writes |
| Load from disk | <300ms | sqlite reads + frombuffer |
| SYS STATS | <100μs | counters + memory read |
| SYS EXPLAIN | <50μs | cost estimator only |

---

## 14. Risks and Mitigations

**Risk: numpy array resizing during incremental inserts.**
Mitigation: pre-allocate arrays with 2× headroom on creation. Track fill level. Reallocate only when full. Amortized O(1) insert.

**Risk: CSR rebuild cost at ceiling (3.2M edges).**
Mitigation: measured at ~50-100ms. Acceptable for batch writes. If it exceeds 200ms, batch CSR rebuilds — buffer commits for 100ms before rebuilding. But measure first.

**Risk: MATCH with large unbound frontiers.**
Mitigation: cost estimator rejects before execution. 100ms hard timeout as backstop. LIMIT clause caps result count.

**Risk: sqlite WAL replay slow after long run without checkpoint.**
Mitigation: auto-checkpoint when WAL exceeds 10MB / 50K entries. Typical replay: <100ms. Hard limit: reject writes if WAL exceeds 50MB and checkpoint failing.

**Risk: scipy.sparse matrix addition for combining edge types is slow.**
Mitigation: precompute all-types combined matrix on CSR rebuild. LRU cache for other combinations. First miss: ~5ms. Subsequent: free.

**Risk: lark grammar changes break existing queries.**
Mitigation: comprehensive test suite with one test per grammar production. Grammar file is versioned. Breaking changes bump the grammar version.

**Risk: Python GIL doesn't protect against logical races in async contexts.**
Mitigation: snapshot swap. All reads go through immutable reference. Writes build new snapshot and swap atomically. No locks needed.

**Risk: json serialization of node_data is slow for large dicts.**
Mitigation: node_data is list[dict] where most dicts are small (5-10 keys). json.dumps of 800K small dicts: ~2-3 seconds. This is the checkpoint bottleneck. If it matters, switch to msgpack for node_data blobs only (one additional dependency, but scipy already pulls in numpy). Or use pickle protocol 5 (stdlib, faster, but opaque).

---

## 15. Implementation Phases

### Phase 1: Core Store (no DSL, no persistence)

Build the in-memory graph engine with Python API only.

```
1.1  strings.py — string table with intern/lookup
     Tests: round-trip, dedup, sequential IDs
     ~0.5 day

1.2  types.py + errors.py — Result, dataclasses, error hierarchy
     ~0.5 day

1.3  memory.py — estimator + ceiling check
     Tests: estimation accuracy, ceiling enforcement, CeilingExceeded raised
     ~0.5 day

1.4  edges.py — EdgeMatrices: per-type CSR, combination cache, transpose cache, degree arrays
     Tests: build from edge list, per-type lookup, combined lookup, cache invalidation
     ~1.5 days

1.5  store.py — node arrays, secondary indices, put/get/delete node, put/delete edge
     Tests: node CRUD, edge CRUD, cascade delete, secondary index maintenance
     ~2 days

1.6  path.py — bidirectional BFS
     Tests: path found, path not found, single node, max depth, edge type filtering
     ~1 day

1.7  snapshot.py — GraphSnapshot + SnapshotManager + swap
     Tests: swap is atomic, readers see consistent state, old snapshot cleanup
     ~0.5 day
```

**Phase 1 total: ~6.5 days**
**Exit criteria:** can create/query/delete nodes and edges, traverse, find shortest paths, all through Python methods. All tests pass. Benchmark confirms read targets.

### Phase 2: DSL Parser

Build the parser. No execution yet — just parse queries into AST nodes.

```
2.1  dsl/ast_nodes.py — typed dataclasses for every query form
     ~0.5 day

2.2  dsl/grammar.lark — complete grammar file
     ~1 day

2.3  dsl/transformer.py — Lark Transformer: parse tree → AST nodes
     Tests: one test per grammar production, error messages with position
     ~1.5 days

2.4  dsl/parser.py — parser wrapper + plan cache
     Tests: parse + cache hit, cache eviction, whitespace normalization
     ~0.5 day
```

**Phase 2 total: ~3.5 days**
**Exit criteria:** every valid DSL query (user + system) parses into the correct AST. Every invalid query raises QueryError with position. Plan cache works.

### Phase 3: DSL Executor — User Reads

Wire AST nodes to store operations for all read queries.

```
3.1  dsl/executor.py — NODE, NODES WHERE, EDGES FROM/TO
     Tests: end-to-end DSL string → Result for each form
     ~1 day

3.2  executor: TRAVERSE, SUBGRAPH, ANCESTORS, DESCENDANTS, COMMON NEIGHBORS
     Tests: depth limits, edge type filtering, empty results
     ~1 day

3.3  executor: PATH, PATHS, SHORTEST PATH, DISTANCE
     Tests: path found/not found, multiple paths, edge type filtering
     ~1 day

3.4  executor: MATCH
     Tests: bound start, unbound with filter, multi-hop, variable bindings, LIMIT
     ~1.5 days

3.5  dsl/cost_estimator.py — EXPLAIN + pre-execution rejection
     Tests: cost estimation accuracy, rejection threshold, EXPLAIN output
     ~1 day

3.6  executor: degree pseudo-fields (INDEGREE, OUTDEGREE, filtered degree)
     Tests: degree conditions in WHERE clauses
     ~0.5 day
```

**Phase 3 total: ~6 days**
**Exit criteria:** every read DSL form executes correctly. Benchmark confirms performance targets. EXPLAIN works for MATCH and TRAVERSE.

### Phase 4: DSL Executor — User Writes

Wire AST nodes to store mutations.

```
4.1  executor: CREATE/UPDATE/UPSERT/DELETE NODE, INCREMENT
     Tests: each operation + safety rules (CREATE fails on existing, UPDATE fails on missing, etc.)
     ~1 day

4.2  executor: CREATE/DELETE EDGE, DELETE EDGES
     Tests: cascade delete, duplicate detection, edge type filtering
     ~1 day

4.3  executor: BEGIN/COMMIT — batch atomicity
     Tests: partial failure rolls back, snapshot swap on commit, concurrent reads see old state
     ~1.5 days

4.4  executor: CSR rebuild after COMMIT
     Tests: rebuilt matrices match expected state, combination cache cleared
     ~0.5 day
```

**Phase 4 total: ~4 days**
**Exit criteria:** all write DSL forms work. Batch atomicity verified. Concurrent reads during writes see consistent snapshots. Safety rules enforced.

### Phase 5: Persistence

Wire sqlite for durability.

```
5.1  persistence/database.py — sqlite wrapper, table creation, PRAGMA config
     ~0.5 day

5.2  persistence/serializer.py — graph → sqlite blobs
     Tests: all data types round-trip correctly
     ~1 day

5.3  persistence/deserializer.py — sqlite blobs → graph
     Tests: loaded graph matches saved graph exactly
     ~1 day

5.4  WAL: write mutations to sqlite wal table, replay on startup
     Tests: WAL append, replay, idempotent replay (CREATE → UPSERT), checkpoint clears WAL
     ~1 day

5.5  Query log: INSERT on every execute(), SYS SLOW/FREQUENT/FAILED retrieval
     Tests: log entries created, slow query ordering, failed query filtering
     ~0.5 day

5.6  Startup flow: load → version check → WAL replay → ready
     Tests: fresh start, normal load, version mismatch, WAL replay after crash
     ~0.5 day

5.7  Checkpoint triggering: auto on WAL size, on close(), on SYS CHECKPOINT
     Tests: auto-trigger at threshold, close() always checkpoints
     ~0.5 day
```

**Phase 5 total: ~5 days**
**Exit criteria:** full round-trip: create graph → checkpoint → kill process → restart → load → verify all data intact. WAL replay recovers mutations after simulated crash. Version mismatch detected.

### Phase 6: System DSL + Schema

Wire system query executor and schema registry.

```
6.1  dsl/executor_system.py — SYS STATS, KINDS, DESCRIBE
     Tests: each returns correct structured data
     ~0.5 day

6.2  executor_system: SYS EXPLAIN (already built in 3.5, just wire to SYS prefix)
     ~0.25 day

6.3  executor_system: SYS SLOW/FREQUENT/FAILED QUERIES
     Tests: queries retrieve correct log entries
     ~0.5 day

6.4  executor_system: SYS CHECKPOINT, REBUILD INDICES, CLEAR LOG, CLEAR CACHE, WAL STATUS/REPLAY
     Tests: each maintenance operation works
     ~0.5 day

6.5  schema.py — SchemaRegistry + validation
     Tests: register kind, validate on write, reject invalid writes, persist to sqlite
     ~1 day

6.6  executor_system: SYS REGISTER/UNREGISTER NODE/EDGE KIND
     Tests: end-to-end schema registration through DSL
     ~0.5 day
```

**Phase 6 total: ~3.25 days**
**Exit criteria:** all SYS queries work. Schema validation catches invalid writes. Schema persists across restarts.

### Phase 7: Integration, Benchmarks, Documentation

```
7.1  __init__.py — GraphStore public class, context manager, entry point routing
     ~0.5 day

7.2  test_integration.py — full workflow tests
     - Create schema → bulk load → query → mutate → checkpoint → reload → verify
     - Concurrent read during write
     - WAL recovery after crash
     - Ceiling enforcement end-to-end
     ~1.5 days

7.3  Benchmark suite
     - generate_graph.py: synthetic graph generator
     - bench_reads.py, bench_writes.py, bench_match.py, bench_persistence.py
     - Markdown output table
     ~1.5 days

7.4  Documentation
     - USER_DSL.md: complete reference with examples
     - SYSTEM_DSL.md: complete reference with examples
     - QUICK_REFERENCE.md: single-page cheat sheet
     - ARCHITECTURE.md: internals overview
     - LLM_PROMPT_SNIPPET.md: copy-paste system prompt block
     ~2 days

7.5  Package setup
     - pyproject.toml, README.md, LICENSE
     - CI configuration (pytest + benchmarks)
     ~0.5 day
```

**Phase 7 total: ~6 days**
**Exit criteria:** all integration tests pass. Benchmarks meet targets. Documentation complete. Package pip-installable.

---

## 16. Total Estimate

| Phase | Description | Days |
|---|---|---|
| 1 | Core Store | 6.5 |
| 2 | DSL Parser | 3.5 |
| 3 | User Reads | 6 |
| 4 | User Writes | 4 |
| 5 | Persistence | 5 |
| 6 | System DSL + Schema | 3.25 |
| 7 | Integration + Polish | 6 |
| **Total** | | **~34.25 days** |

Add 30% buffer for debugging, edge cases, and cross-platform testing: **~45 days realistic.**

Each phase produces a working, testable artifact. Phase 1 is usable on its own (Python API). Phase 1+2+3 gives read-only DSL queries. Phase 1-4 gives full read/write DSL. Phase 1-5 gives durability. Phase 1-6 gives the complete feature set. Phase 7 makes it shippable.

---

## 17. What's Out of Scope for v1

- Disk-backed fallback for graphs exceeding 256MB. Unsupported, hard error.
- Multi-root workspaces. One root, one graph, one sqlite file.
- Schema migration. Version mismatch = caller rebuilds. No migration logic.
- Graph algorithms beyond BFS/DFS/shortest path (PageRank, community detection, etc.). These can be built on top using the CSR matrices the store exposes, but they're not part of the DSL.
- Distributed or multi-process access. Single process only.
- History/versioning. No "show me the graph as of 5 minutes ago."
- Undo/redo. Mutations are permanent (within the WAL recovery window).
- Full-text search on node data. Equality and comparison only.
- Regex/LIKE/CONTAINS in WHERE. Add in v2 if needed.
- Streaming/subscription queries. No "notify me when a node changes."
