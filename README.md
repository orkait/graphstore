<p align="center">
  <h1 align="center">graphstore</h1>
  <p align="center">In-memory typed graph database with a human-readable DSL</p>
</p>

<p align="center">
  <a href="https://github.com/orkait/graphstore/actions/workflows/ci.yml"><img src="https://github.com/orkait/graphstore/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue" alt="Python versions">
  <a href="https://github.com/orkait/graphstore/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <img src="https://img.shields.io/badge/version-0.1.0-orange" alt="Version">
</p>

---

**graphstore** is a lightweight, in-memory typed graph database built on scipy sparse matrices and numpy arrays. It ships with a custom DSL designed for readability — no Cypher, no SPARQL, just plain English-like queries.

## Features

- **Typed nodes and edges** — every node has a `kind`, edges connect typed endpoints
- **Sparse matrix storage** — per-type CSR matrices via scipy for fast neighbor lookups
- **Custom DSL** — 40+ query forms parsed by a Lark LALR(1) grammar
- **Pattern matching** — `MATCH` queries backed by sparse matrix-vector multiply
- **Path finding** — bidirectional BFS for shortest path, multi-path enumeration
- **Persistence** — sqlite-backed with WAL for crash recovery
- **Memory ceiling** — configurable hard limit with pre-operation checks
- **Schema validation** — optional kind registration with required/optional fields
- **Zero config** — works in-memory with no setup, add a path for persistence

## Install

```bash
pip install graphstore
```

## Quick Start

```python
from graphstore import GraphStore

# In-memory (no persistence)
g = GraphStore()

# Or with persistence
g = GraphStore(path="./my_graph")

# Create nodes
g.execute('CREATE NODE "fn_main" kind = "function" name = "main" file = "app.py"')
g.execute('CREATE NODE "fn_helper" kind = "function" name = "helper" file = "utils.py"')
g.execute('CREATE NODE "cls_app" kind = "class" name = "App" file = "app.py"')

# Create edges
g.execute('CREATE EDGE "fn_main" -> "fn_helper" kind = "calls"')
g.execute('CREATE EDGE "fn_main" -> "cls_app" kind = "uses"')

# Query
result = g.execute('NODE "fn_main"')
print(result.data)
# {'id': 'fn_main', 'kind': 'function', 'name': 'main', 'file': 'app.py'}

result = g.execute('NODES WHERE kind = "function"')
print(result.count)  # 2

result = g.execute('EDGES FROM "fn_main" WHERE kind = "calls"')
print(result.data)
# [{'source': 'fn_main', 'target': 'fn_helper', 'kind': 'calls'}]
```

## DSL Reference

### Reads

```sql
-- Single node
NODE "node_id"

-- Filter nodes
NODES WHERE kind = "function" AND file = "app.py" LIMIT 10

-- Edges
EDGES FROM "node_id" WHERE kind = "calls"
EDGES TO "node_id"

-- Traversal (BFS)
TRAVERSE FROM "node_id" DEPTH 3 WHERE kind = "calls"
SUBGRAPH FROM "node_id" DEPTH 2

-- Path finding
PATH FROM "a" TO "b" MAX_DEPTH 5 WHERE kind = "calls"
PATHS FROM "a" TO "b" MAX_DEPTH 5
SHORTEST PATH FROM "a" TO "b"
DISTANCE FROM "a" TO "b" MAX_DEPTH 10

-- Ancestry
ANCESTORS OF "node_id" DEPTH 3
DESCENDANTS OF "node_id" DEPTH 3
COMMON NEIGHBORS OF "a" AND "b"

-- Pattern matching
MATCH ("fn_main") -[kind = "calls"]-> (callee) -[kind = "calls"]-> (transitive)

-- Degree filters
NODES WHERE OUTDEGREE calls > 5
NODES WHERE INDEGREE > 10
```

### Writes

```sql
-- Nodes
CREATE NODE "id" kind = "function" name = "foo" file = "bar.py"
UPDATE NODE "id" SET name = "new_name"
UPSERT NODE "id" kind = "function" name = "foo"
DELETE NODE "id"
DELETE NODES WHERE kind = "test"

-- Edges
CREATE EDGE "source" -> "target" kind = "calls"
DELETE EDGE "source" -> "target" WHERE kind = "calls"
DELETE EDGES FROM "node_id" WHERE kind = "calls"

-- Counters
INCREMENT NODE "id" hits BY 1

-- Batch (atomic with rollback)
BEGIN
CREATE NODE "a" kind = "x" name = "alpha"
CREATE NODE "b" kind = "x" name = "beta"
CREATE EDGE "a" -> "b" kind = "link"
COMMIT
```

### System

```sql
-- Statistics
SYS STATS
SYS STATS NODES
SYS STATS MEMORY

-- Schema
SYS REGISTER NODE KIND "function" REQUIRED name OPTIONAL file, line
SYS REGISTER EDGE KIND "calls" FROM "function" TO "function"
SYS UNREGISTER NODE KIND "function"
SYS KINDS
SYS EDGE KINDS
SYS DESCRIBE NODE "function"

-- Query analysis
SYS EXPLAIN TRAVERSE FROM "a" DEPTH 5 WHERE kind = "calls"
SYS SLOW QUERIES LIMIT 10
SYS FREQUENT QUERIES LIMIT 5
SYS FAILED QUERIES LIMIT 10

-- Maintenance
SYS CHECKPOINT
SYS REBUILD INDICES
SYS CLEAR CACHE
SYS CLEAR LOG
SYS WAL STATUS
```

## Persistence

```python
# Data is persisted to sqlite when a path is provided
with GraphStore(path="./data") as g:
    g.execute('CREATE NODE "a" kind = "x" name = "alpha"')
    # Auto-checkpoints on close

# Reopen — data survives
with GraphStore(path="./data") as g:
    r = g.execute('NODE "a"')
    assert r.data["name"] == "alpha"

# Manual checkpoint
g.checkpoint()
```

Writes are logged to a WAL (write-ahead log) for crash recovery. On restart, uncommitted WAL entries are replayed automatically.

## Schema Validation

```python
g = GraphStore()

# Register node kinds with required/optional fields
g.execute('SYS REGISTER NODE KIND "function" REQUIRED name OPTIONAL file, line')
g.execute('SYS REGISTER EDGE KIND "calls" FROM "function" TO "function"')

# List registered kinds
r = g.execute('SYS KINDS')
print(r.data)  # ['function']

# Describe a kind
r = g.execute('SYS DESCRIBE NODE "function"')
print(r.data)  # {'required': ['name'], 'optional': ['file', 'line']}
```

## Memory Management

```python
# Set a memory ceiling (default: 256 MB)
g = GraphStore(ceiling_mb=512)

# Check usage
print(g.memory_usage)  # bytes

r = g.execute('SYS STATS MEMORY')
print(r.data)
# {'memory_bytes': 680, 'ceiling_bytes': 512000000}
```

Operations that would exceed the ceiling raise `CeilingExceeded` before modifying state.

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `path` | `None` | Directory for sqlite persistence. `None` = in-memory only |
| `ceiling_mb` | `256` | Memory ceiling in MB |
| `allow_system_queries` | `True` | Enable/disable `SYS` queries |

## Architecture

```
graphstore/
├── __init__.py          # GraphStore public API
├── store.py             # CoreStore: numpy arrays + node/edge CRUD
├── edges.py             # EdgeMatrices: per-type scipy CSR matrices
├── strings.py           # StringTable: interned string ↔ int mapping
├── types.py             # Result, Edge, NodeData dataclasses
├── errors.py            # Error hierarchy
├── memory.py            # Memory estimation + ceiling enforcement
├── path.py              # Bidirectional BFS, multi-path, common neighbors
├── snapshot.py          # GraphSnapshot + SnapshotManager
├── schema.py            # SchemaRegistry: kind validation
├── dsl/
│   ├── grammar.lark     # Lark LALR(1) grammar (40+ productions)
│   ├── parser.py        # Parser wrapper + LRU plan cache
│   ├── transformer.py   # Parse tree → typed AST nodes
│   ├── ast_nodes.py     # 38 AST dataclasses
│   ├── executor.py      # User query executor (reads + writes)
│   ├── executor_system.py  # System query executor
│   └── cost_estimator.py   # Frontier-based cost rejection
└── persistence/
    ├── database.py      # sqlite setup + table creation
    ├── serializer.py    # Graph → sqlite blobs
    └── deserializer.py  # sqlite blobs → graph
```

## Playground

An interactive browser-based workbench for exploring the DSL:

```bash
pip install graphstore[playground]
graphstore playground
```

Three-panel UI with a CodeMirror editor (DSL syntax highlighting), React Flow graph visualization, and stacked query results. Includes 4 pre-loaded example scripts. See [`playground/README.md`](playground/README.md) for details.

## Development

```bash
git clone https://github.com/orkait/graphstore.git
cd graphstore
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## License

MIT
