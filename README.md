<div align="center">

# graphstore

**Agentic brain DB - in-memory graph database built for AI agent memory**

[![CI](https://github.com/orkait/graphstore/actions/workflows/ci.yml/badge.svg)](https://github.com/orkait/graphstore/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-22c55e?logo=opensourceinitiative&logoColor=white)](https://github.com/orkait/graphstore/blob/main/LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.0-f59e0b?logo=semver&logoColor=white)](https://github.com/orkait/graphstore)
[![NumPy](https://img.shields.io/badge/numpy-%23013243?logo=numpy&logoColor=white)](https://numpy.org)
[![SciPy](https://img.shields.io/badge/scipy-%238CAAE6?logo=scipy&logoColor=white)](https://scipy.org)
[![FastAPI](https://img.shields.io/badge/fastapi-%23009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)

</div>

---

A graph database designed as a memory substrate for AI agents. Columnar numpy storage, sparse matrix traversal, spreading activation recall, belief management, and a human-readable DSL - no Cypher, no SPARQL, just queries that read like sentences.

## 🧩 Why graphstore?

| What agents need | How graphstore solves it |
|---|---|
| Fast recall by association | `RECALL FROM "concept:paris" DEPTH 3 LIMIT 10` - spreading activation via sparse matmul |
| Memory summarization | `AGGREGATE NODES GROUP BY topic SELECT COUNT(), AVG(importance)` - numpy vectorized |
| Belief tracking | `ASSERT "fact:x" CONFIDENCE 0.9 SOURCE "tool"` / `RETRACT "fact:x" REASON "outdated"` |
| Working memory cleanup | `CREATE NODE "tmp" ... EXPIRES IN 30m` + `SYS EXPIRE` |
| Contradiction detection | `SYS CONTRADICTIONS WHERE kind = "belief" FIELD value GROUP BY topic` |
| Hypothesis testing | `SYS SNAPSHOT "before"` ... explore ... `SYS ROLLBACK TO "before"` |
| Temporal reasoning | `NODES WHERE __created_at__ > NOW() - 7d` |
| Isolated reasoning | `BIND CONTEXT "session-1"` ... `DISCARD CONTEXT "session-1"` |

**Performance at 100k nodes:** COUNT in 34us, GROUP BY in 744us, 14x less memory than dict storage. Everything runs on numpy arrays - no Python loops in the hot path.

## 📦 Install

```bash
pip install graphstore
```

## 🚀 Quick Start

```python
from graphstore import GraphStore

g = GraphStore()  # in-memory, or GraphStore(path="./brain") for persistence

# Store memories
g.execute('CREATE NODE "memory:paris" kind = "memory" topic = "travel" importance = 0.9')
g.execute('CREATE NODE "memory:eiffel" kind = "memory" topic = "travel" importance = 0.8')
g.execute('CREATE NODE "fact:capital" kind = "fact" topic = "geography" importance = 1.0')

# Connect them
g.execute('CREATE EDGE "memory:paris" -> "memory:eiffel" kind = "associated"')
g.execute('CREATE EDGE "fact:capital" -> "memory:paris" kind = "supports"')

# Recall by association (spreading activation)
result = g.execute('RECALL FROM "fact:capital" DEPTH 2 LIMIT 5')
for node in result.data:
    print(f"{node['id']} (score: {node['_activation_score']:.3f})")

# Summarize what the agent knows
result = g.execute('AGGREGATE NODES GROUP BY topic SELECT COUNT(), AVG(importance)')
for row in result.data:
    print(f"{row['topic']}: {row['COUNT()']} memories, avg importance {row['AVG(importance)']:.2f}")
```

## 🧠 Agent Memory Features

<details>
<summary><strong>Belief Management</strong> - assert facts, retract outdated beliefs, detect contradictions</summary>

```sql
-- Assert a fact with confidence and source
ASSERT "fact:earth-radius" value = 6371 kind = "fact" CONFIDENCE 0.99 SOURCE "physics-tool"

-- Retract when outdated (node becomes invisible but kept for audit)
RETRACT "fact:old-preference" REASON "user corrected this"

-- Bulk update confidence when a source is discredited
UPDATE NODES WHERE kind = "fact" AND source = "unreliable" SET confidence = 0.1

-- Find contradicting beliefs automatically
SYS CONTRADICTIONS WHERE kind = "belief" FIELD value GROUP BY topic
```

</details>

<details>
<summary><strong>Associative Recall</strong> - spreading activation, importance/recency weighted</summary>

```sql
-- Recall memories associated with a concept (sparse matrix-vector multiply)
RECALL FROM "concept:paris" DEPTH 3 LIMIT 10
RECALL FROM "concept:paris" DEPTH 3 LIMIT 10 WHERE kind = "memory"

-- Results include activation scores
-- [{"id": "memory:eiffel", "_activation_score": 0.72, ...}, ...]

-- Propagate confidence through the knowledge graph
PROPAGATE "belief:user-is-expert" FIELD confidence DEPTH 3
```

</details>

<details>
<summary><strong>Working Memory + TTL</strong> - auto-expiring scratch space</summary>

```sql
-- Create with expiry
CREATE NODE "scratch:123" kind = "working" content = "..." EXPIRES IN 30m
CREATE NODE "ctx:456" kind = "context" data = "..." EXPIRES AT "2026-04-01T00:00:00"

-- Expired nodes are automatically invisible to all queries
-- Flush to reclaim memory
SYS EXPIRE
SYS EXPIRE WHERE kind = "working"
```

</details>

<details>
<summary><strong>Hypothesis Testing</strong> - snapshot, explore, rollback</summary>

```sql
-- Save state before exploring a reasoning branch
SYS SNAPSHOT "before-hypothesis"

-- ... agent explores, creates nodes, draws conclusions ...

-- Didn't work out? Undo everything
SYS ROLLBACK TO "before-hypothesis"

-- Or just check what would happen without committing
WHAT IF RETRACT "belief:earth-is-flat"
-- Returns: {affected_nodes: [...], affected_count: 14}
```

</details>

<details>
<summary><strong>Context Isolation</strong> - isolated reasoning sessions</summary>

```sql
-- Start an isolated context
BIND CONTEXT "reasoning-session-42"

-- All creates are auto-tagged, all reads are scoped
CREATE NODE "hyp:1" kind = "hypothesis" content = "maybe X"
RECALL FROM "hyp:1" DEPTH 3 LIMIT 10  -- only searches within context

-- Done? Cleanup everything in the context
DISCARD CONTEXT "reasoning-session-42"
```

</details>

<details>
<summary><strong>Aggregations</strong> - GROUP BY with numpy-vectorized SUM/AVG/MIN/MAX/COUNT</summary>

```sql
-- How many memories per topic?
AGGREGATE NODES WHERE kind = "memory"
  GROUP BY topic
  SELECT COUNT(), AVG(importance)
  HAVING COUNT() > 2
  ORDER BY AVG(importance) DESC

-- Global summary (no GROUP BY)
AGGREGATE NODES SELECT COUNT(), SUM(importance), MAX(__updated_at__)

-- Columnar-only: all fields must be declared via SYS REGISTER for aggregation
SYS REGISTER NODE KIND "memory" REQUIRED topic:string, importance:float
```

</details>

<details>
<summary><strong>Temporal Queries</strong> - auto-timestamps + relative time</summary>

```sql
-- Every node gets __created_at__ and __updated_at__ automatically (Unix ms)
NODES WHERE __created_at__ > NOW() - 7d
NODES WHERE __updated_at__ > TODAY
NODES WHERE __created_at__ > YESTERDAY ORDER BY __created_at__ DESC LIMIT 20

-- Consolidate old memories
MERGE NODE "memory:old-paris" INTO "memory:paris-canonical"
```

</details>

## 📖 DSL Reference

<details>
<summary><strong>Reads</strong> - queries, traversals, path finding, pattern matching</summary>

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

-- Weighted paths (Dijkstra)
WEIGHTED SHORTEST PATH FROM "a" TO "b"
WEIGHTED DISTANCE FROM "a" TO "b"

-- Ancestry
ANCESTORS OF "node_id" DEPTH 3
DESCENDANTS OF "node_id" DEPTH 3
COMMON NEIGHBORS OF "a" AND "b"

-- Pattern matching (sparse matrix-vector multiply)
MATCH ("fn_main") -[kind = "calls"]-> (callee) -[kind = "calls"]-> (transitive)

-- Counting
COUNT NODES WHERE kind = "function"
COUNT EDGES WHERE kind = "calls"

-- Degree filters
NODES WHERE OUTDEGREE calls > 5
NODES WHERE INDEGREE > 10
```

</details>

<details>
<summary><strong>Writes</strong> - create, update, delete, beliefs, batch operations</summary>

```sql
-- Nodes
CREATE NODE "id" kind = "function" name = "foo" file = "bar.py"
CREATE NODE "id" kind = "working" name = "tmp" EXPIRES IN 1h
UPDATE NODE "id" SET name = "new_name"
UPSERT NODE "id" kind = "function" name = "foo"
DELETE NODE "id"
DELETE NODES WHERE kind = "test"
UPDATE NODES WHERE kind = "fact" SET confidence = 0.5

-- Beliefs
ASSERT "fact:x" kind = "fact" value = 42 CONFIDENCE 0.9 SOURCE "tool"
RETRACT "fact:x" REASON "outdated"

-- Memory consolidation
MERGE NODE "memory:old" INTO "memory:canonical"

-- Edges
CREATE EDGE "source" -> "target" kind = "calls"
DELETE EDGE "source" -> "target" WHERE kind = "calls"
DELETE EDGES FROM "node_id" WHERE kind = "calls"

-- Counters
INCREMENT NODE "id" hits BY 1

-- Belief propagation
PROPAGATE "belief:x" FIELD confidence DEPTH 3

-- Context isolation
BIND CONTEXT "session-1"
DISCARD CONTEXT "session-1"

-- Batch (atomic with rollback)
BEGIN
CREATE NODE "a" kind = "x" name = "alpha"
CREATE NODE "b" kind = "x" name = "beta"
CREATE EDGE "a" -> "b" kind = "link"
COMMIT
```

</details>

<details>
<summary><strong>System</strong> - stats, schema, query analysis, maintenance, snapshots</summary>

```sql
-- Statistics
SYS STATS
SYS STATS NODES
SYS STATS MEMORY

-- Schema (with typed fields for columnar acceleration)
SYS REGISTER NODE KIND "memory" REQUIRED topic:string, importance:float OPTIONAL tag:string
SYS REGISTER EDGE KIND "calls" FROM "function" TO "function"
SYS UNREGISTER NODE KIND "function"
SYS KINDS
SYS EDGE KINDS
SYS DESCRIBE NODE "memory"

-- Belief consistency
SYS CONTRADICTIONS WHERE kind = "belief" FIELD value GROUP BY topic

-- TTL management
SYS EXPIRE
SYS EXPIRE WHERE kind = "working"

-- Snapshots (hypothesis testing)
SYS SNAPSHOT "before-experiment"
SYS ROLLBACK TO "before-experiment"
SYS SNAPSHOTS

-- Counterfactual analysis
WHAT IF RETRACT "belief:x"

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

</details>

## 💾 Persistence

```python
# Data is persisted to sqlite when a path is provided
with GraphStore(path="./data") as g:
    g.execute('CREATE NODE "a" kind = "x" name = "alpha"')
    # Auto-checkpoints on close

# Reopen - data survives
with GraphStore(path="./data") as g:
    r = g.execute('NODE "a"')
    assert r.data["name"] == "alpha"

# Manual checkpoint
g.checkpoint()
```

Writes are logged to a WAL (write-ahead log) for crash recovery. On restart, uncommitted WAL entries are replayed automatically.

## 🛡️ Schema Validation

```python
g = GraphStore()

# Register node kinds with required/optional typed fields
g.execute('SYS REGISTER NODE KIND "memory" REQUIRED topic:string, importance:float OPTIONAL tag:string')
g.execute('SYS REGISTER EDGE KIND "associated" FROM "memory" TO "memory"')

# Typed fields are automatically columnarized for fast filtering + aggregation
result = g.execute('AGGREGATE NODES WHERE kind = "memory" GROUP BY topic SELECT AVG(importance)')
```

## 🧠 Memory Management

```python
# Set a memory ceiling (default: 256 MB)
g = GraphStore(ceiling_mb=512)

# Columnar storage: ~35 bytes/node (vs ~1.2KB with dicts)
# 8GB budget = ~200M nodes
print(g.memory_usage)  # bytes

r = g.execute('SYS STATS MEMORY')
print(r.data)
```

Operations that would exceed the ceiling raise `CeilingExceeded` before modifying state.

## ⚡ Performance

Measured at 100k nodes, 5 columnarized fields:

| Operation | Time |
|---|---|
| COUNT WHERE score > X | 34 us |
| GROUP BY + AVG | 744 us |
| WHERE field = value | 50 us |
| ORDER BY LIMIT 10 | 200 us |
| Batch rollback (100k) | 75 us |
| RECALL DEPTH 3 | ~1-3 ms |
| Memory per node | ~35 bytes |

## ⚙️ Configuration

| Parameter | Default | Description |
|---|---|---|
| `path` | `None` | Directory for sqlite persistence. `None` = in-memory only |
| `ceiling_mb` | `256` | Memory ceiling in MB |
| `allow_system_queries` | `True` | Enable/disable `SYS` queries |

## 🏗️ Architecture

<details>
<summary>Project structure</summary>

```
graphstore/
├── __init__.py             # GraphStore public API
├── store.py                # CoreStore: numpy columnar arrays + node/edge CRUD
├── columns.py              # ColumnStore: typed numpy arrays (source of truth)
├── edges.py                # EdgeMatrices: per-type scipy CSR matrices
├── strings.py              # StringTable: interned string <-> int mapping
├── types.py                # Result, Edge, NodeData dataclasses
├── errors.py               # Error hierarchy
├── memory.py               # Memory estimation + ceiling enforcement
├── path.py                 # Bidirectional BFS, multi-path, Dijkstra
├── snapshot.py             # GraphSnapshot + SnapshotManager
├── schema.py               # SchemaRegistry: kind validation
├── dsl/
│   ├── grammar.lark        # Lark LALR(1) grammar (60+ productions)
│   ├── parser.py           # Parser wrapper + LRU plan cache
│   ├── transformer.py      # Parse tree -> typed AST nodes
│   ├── ast_nodes.py        # 50+ AST dataclasses
│   ├── executor.py         # User query executor (reads + writes + recall)
│   ├── executor_system.py  # System query executor
│   └── cost_estimator.py   # Frontier-based cost rejection
└── persistence/
    ├── database.py         # sqlite setup + table creation
    ├── serializer.py       # Columns -> sqlite blobs
    └── deserializer.py     # sqlite blobs -> columns (with legacy migration)
```

</details>

## 🎮 Playground

An interactive browser-based workbench for exploring the DSL:

```bash
pip install graphstore[playground]
graphstore playground
```

Three-panel UI with a CodeMirror editor (DSL syntax highlighting), React Flow graph visualization, and stacked query results. Includes 4 pre-loaded example scripts, two layout engines (dagre hierarchy + force-directed cluster), configurable graph settings, and light/dark theme support.

See [`playground/README.md`](playground/README.md) for details.

## 🛠️ Development

```bash
git clone https://github.com/orkait/graphstore.git
cd graphstore
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## 📄 License

MIT
