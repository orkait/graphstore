# graphstore - Complete Project Summary

Everything a new session needs to understand this codebase.

## What is graphstore

An agent-first memory substrate. Not a user-facing app. It provides all the memory tools an AI agent needs: graph recall, semantic recall, lexical recall, belief tracking, document ingestion, and structured note management.

One agent per DB instance. Single-threaded. No locks.

## Core architecture

Six engines, one DSL:

| Engine | Backing | Purpose |
|--------|---------|---------|
| Graph | numpy arrays + scipy CSR matrices | Columnar node storage, sparse edge traversal |
| Vector | usearch HNSW index | Semantic similarity search |
| Document | SQLite (separate `documents.db`) | Blob storage, summaries, FTS5 full-text search |
| DSL | Lark LALR(1) parser + LRU cache | Human-readable query language |
| Vault | Markdown files on disk | Structured notes synced to graph |
| Persistence | SQLite WAL + blob serialization | Checkpoint/restore, query logging |

### Key internals

**Node storage**: Typed numpy arrays (`ColumnStore`) indexed by slot. Fields stored as `int64`, `float64`, or `int32_interned` (string table IDs). Presence tracked via boolean masks. Reserved columns prefixed with `__` (e.g., `__created_at__`, `__expires_at__`, `__retracted__`, `__confidence__`, `__blob_state__`).

**Edge storage**: Per-type scipy `csr_matrix`. Combined-all transpose cached in `EdgeMatrices._combined_transpose` (invalidated on rebuild). Enables O(1) neighbor lookup and matrix-vector multiply for spreading activation (RECALL). Deferred CSR rebuild via `_edges_dirty` flag.

**String interning**: All strings (node IDs, kinds, field values) mapped to int32 via `StringTable`. Shrinks memory, accelerates numpy comparisons.

**Visibility system**: `VisibilityMixin` provides a single boolean mask per query combining tombstones + TTL expiry + retracted beliefs + context isolation. All query types respect this mask.

**Slot lifecycle**: Nodes occupy slots in pre-allocated arrays. Deleted nodes become tombstones (slot reused on next insert). Tombstone compaction via `SYS OPTIMIZE COMPACT` renumbers all slots. DocumentStore slot remapping uses temp-table batch SQL (O(1) statements, not O(N) per-slot loops).

**WAL**: `WALManager` (in `wal.py`) handles append, replay, checkpoint, auto-checkpoint, and query log rotation. Wired into `GraphStore` — no inline WAL logic in `graphstore.py`.

### Data flow

```
agent calls gs.execute("CREATE NODE ...")
  -> dsl/parser.py: parse(query) -> AST node (cached in LRU)
  -> dsl/executor.py: _dispatch(ast) -> registry lookup -> handler method
  -> handler modifies CoreStore (numpy arrays, edge lists)
  -> WALManager.append() (if write + persistence enabled)
  -> Result(kind, data, count, elapsed_us) returned
```

### File structure

```
graphstore/
├── graphstore.py             # GraphStore facade: init, execute(), checkpoint, public API
├── config.py                 # Typed config (msgspec Structs, graphstore.json)
├── wal.py                    # WALManager: append, replay, checkpoint, log rotation
├── core/
│   ├── store.py              # CoreStore: slot CRUD, materialization, live mask
│   ├── columns.py            # ColumnStore: typed numpy arrays, vectorized filtering
│   ├── edges.py              # EdgeMatrices: scipy CSR per edge type + cached transpose
│   ├── strings.py            # StringTable: bidirectional string <-> int32
│   ├── schema.py             # SchemaRegistry: optional type validation + EMBED field
│   ├── path.py               # BFS, Dijkstra, bidirectional BFS, common neighbors
│   ├── memory.py             # Memory ceiling enforcement
│   ├── optimizer.py          # Self-balancing: compact (batch SQL), string GC, defrag, cleanup
│   ├── types.py              # Result, Edge dataclasses
│   └── errors.py             # Exception hierarchy (13 error types)
├── dsl/
│   ├── grammar.lark          # LALR(1) grammar (~280 rules)
│   ├── parser.py             # Lark parser + LRU plan cache (256 entries)
│   ├── transformer.py        # Parse tree -> AST dataclasses
│   ├── ast_nodes.py          # 65+ AST dataclasses (flat file)
│   ├── executor.py           # Auto-dispatch via handler registry + mixin inheritance
│   ├── executor_base.py      # ExecutorBase: __init__, execute(), mixin wiring (~30 lines)
│   ├── visibility.py         # VisibilityMixin: live-mask, slot visibility, TTL
│   ├── filtering.py          # FilteringMixin: WHERE eval, column accel, index helpers
│   ├── executor_system.py    # SYS * command handlers
│   ├── cost_estimator.py     # Frontier-based cost rejection for MATCH/TRAVERSE
│   └── handlers/             # Domain-specific handler mixins
│       ├── _registry.py      # @handles decorator, DISPATCH dict, WRITE_OPS set
│       ├── nodes.py          # NODE, NODES, COUNT
│       ├── edges.py          # EDGES, CREATE/UPDATE/DELETE EDGE
│       ├── traversal.py      # TRAVERSE, PATH, SHORTEST, ANCESTORS, DESCENDANTS
│       ├── pattern.py        # MATCH
│       ├── aggregation.py    # AGGREGATE
│       ├── intelligence.py   # RECALL, SIMILAR TO, LEXICAL SEARCH, COUNTERFACTUAL
│       ├── beliefs.py        # ASSERT, RETRACT, PROPAGATE
│       ├── mutations.py      # CREATE/UPDATE/DELETE NODE, MERGE, BATCH, FORGET
│       ├── context.py        # BIND/DISCARD CONTEXT
│       └── ingest.py         # INGEST, CONNECT NODE
├── embedding/                # model2vec (default, 30MB), ONNX HF (opt-in)
├── vector/                   # VectorStore: usearch HNSW wrapper
├── document/                 # DocumentStore: SQLite multi-table + FTS5
├── ingest/                   # File parsing, chunking, routing, vision, connector
├── vault/                    # Markdown notes: parser, manager, sync, executor
├── persistence/              # SQLite checkpoint/restore
├── registry/                 # Model management (install-embedder, install-vision)
├── voice/                    # Moonshine STT + Piper TTS (opt-in)
├── server.py                 # FastAPI playground wrapper (uses public GraphStore API)
└── cli.py                    # CLI entry point
```

### Public API surface

`GraphStore` exposes these public methods/properties so `server.py` and external code don't need private access:

| Method / Property | Purpose |
|-------------------|---------|
| `execute(query)` | Execute a DSL query, returns `Result` |
| `execute_batch(queries)` | Execute multiple queries |
| `checkpoint()` | Force persist to disk |
| `close()` | Checkpoint + close connection |
| `get_all_nodes()` | All live nodes (used by server `/api/graph`) |
| `get_all_edges()` | All live edges (used by server `/api/graph`) |
| `node_count` | Number of live nodes |
| `edge_count` | Number of edges |
| `memory_usage` | Estimated memory in bytes |
| `cost_threshold` | DSL query cost threshold (get/set) |
| `ceiling_mb` | Memory ceiling in MB (get/set) |
| `set_script(s)` / `get_script()` | Playground script storage |
| `speak(text)` / `listen(cb)` | Voice TTS/STT |

## Dependencies

Core: `numpy`, `scipy`, `lark`, `usearch`, `model2vec`, `pyyaml`, `markitdown`, `pymupdf4llm`, `pymupdf`, `msgspec`

Optional: `fastapi`/`uvicorn` (playground), `docling`/`openai` (ingest-pro)

## Config system

One JSON file (`graphstore.json`), sectioned by engine. All sections have typed defaults via frozen msgspec Structs. Missing keys auto-fill.

```json
{
  "core": {"ceiling_mb": 256, "initial_capacity": 1024},
  "vector": {"embedder": "default", "similarity_threshold": 0.85},
  "document": {"fts_tokenizer": "porter unicode61"},
  "dsl": {"cost_threshold": 100000, "plan_cache_size": 256, "auto_optimize": false, "optimize_interval": 500},
  "vault": {"enabled": false, "path": null, "auto_sync": true},
  "persistence": {"wal_hard_limit": 100000, "auto_checkpoint_threshold": 50000, "log_retention_days": 7},
  "retention": {"blob_warm_days": 30, "blob_archive_days": 90, "blob_delete_days": 365},
  "server": {"cors_origins": ["*"], "ingest_root": null}
}
```

Load order: explicit config object > `config_path` kwarg > `GRAPHSTORE_CONFIG` env var > `{path}/graphstore.json` > defaults.

## DSL command reference

### Reads
```
NODE "id" [WITH DOCUMENT]
NODES [WHERE ...] [ORDER BY field ASC|DESC] [LIMIT N] [OFFSET N]
EDGES FROM|TO "id" [WHERE ...] [LIMIT N]
TRAVERSE FROM "id" DEPTH N [WHERE ...] [LIMIT N]
SUBGRAPH FROM "id" DEPTH N
PATH FROM "a" TO "b" MAX_DEPTH N
SHORTEST PATH FROM "a" TO "b"
WEIGHTED SHORTEST PATH FROM "a" TO "b"
DISTANCE FROM "a" TO "b" MAX_DEPTH N
ANCESTORS OF "id" DEPTH N
DESCENDANTS OF "id" DEPTH N
COMMON NEIGHBORS OF "a" AND "b"
MATCH ("id") -[kind = "x"]-> (var WHERE ...)
COUNT NODES|EDGES [WHERE ...]
AGGREGATE NODES [WHERE ...] GROUP BY field SELECT COUNT(), AVG(f) [HAVING ...] [ORDER BY ...] [LIMIT N]
RECALL FROM "id" DEPTH N [LIMIT N] [WHERE ...]
SIMILAR TO "text"|[vector]|NODE "id" [LIMIT N] [WHERE ...]
LEXICAL SEARCH "query" [LIMIT N] [WHERE ...]
WHAT IF RETRACT "id"
```

### Writes
```
CREATE NODE "id" kind = "x" field = value [VECTOR [...]] [EXPIRES IN 30m] [DOCUMENT "text"]
CREATE NODE AUTO kind = "x" field = value
UPDATE NODE "id" SET field = value
UPSERT NODE "id" kind = "x" field = value
DELETE NODE "id"
DELETE NODES WHERE ...
UPDATE NODES WHERE ... SET field = value
CREATE EDGE "src" -> "tgt" kind = "x"
UPDATE EDGE "src" -> "tgt" SET field = value
DELETE EDGE "src" -> "tgt" [WHERE ...]
DELETE EDGES FROM|TO "id" [WHERE ...]
INCREMENT NODE "id" field BY N
ASSERT "id" field = value [CONFIDENCE 0.9] [SOURCE "tool"]
RETRACT "id" [REASON "why"]
MERGE NODE "src" INTO "tgt"
PROPAGATE "id" FIELD f DEPTH N
INGEST "file" [AS "id"] [KIND "type"] [USING markitdown|docling|VISION "model"]
CONNECT NODE "id" [THRESHOLD 0.8]
FORGET NODE "id"
BIND CONTEXT "name"
DISCARD CONTEXT "name"
BEGIN ... COMMIT (batch with rollback)
```

### System
```
SYS STATS [NODES|EDGES|MEMORY|WAL]
SYS STATUS
SYS HEALTH
SYS OPTIMIZE [COMPACT|STRINGS|EDGES|VECTORS|BLOBS|CACHE]
SYS RETAIN
SYS KINDS / SYS EDGE KINDS / SYS DESCRIBE NODE|EDGE "kind"
SYS REGISTER NODE KIND "k" REQUIRED f1:type, f2:type [OPTIONAL ...] [EMBED field]
SYS REGISTER EDGE KIND "k" FROM "kind1" TO "kind2"
SYS UNREGISTER NODE|EDGE KIND "k"
SYS CONTRADICTIONS WHERE ... FIELD f GROUP BY g
SYS CONNECT [WHERE ...] [THRESHOLD 0.85]
SYS DUPLICATES [WHERE ...] [THRESHOLD 0.95]
SYS EXPIRE [WHERE ...]
SYS SNAPSHOT "name" / SYS ROLLBACK TO "name" / SYS SNAPSHOTS
SYS EMBEDDERS / SYS REEMBED
SYS CHECKPOINT / SYS REBUILD INDICES / SYS CLEAR LOG|CACHE
SYS EXPLAIN <read_query>
SYS SLOW QUERIES [SINCE "iso"] [LIMIT N] / SYS FAILED QUERIES [LIMIT N]
SYS WAL STATUS|REPLAY
```

### Vault
```
VAULT NEW "title" [KIND "type"] [TAGS "a,b"]
VAULT READ "title"
VAULT WRITE "title" SECTION "name" CONTENT "text"
VAULT APPEND "title" SECTION "name" CONTENT "text"
VAULT SEARCH "query" [LIMIT N] [WHERE ...]
VAULT BACKLINKS "title"
VAULT LIST [WHERE ...] [ORDER BY ...] [LIMIT N]
VAULT SYNC / VAULT DAILY / VAULT ARCHIVE "title"
```

## What shipped

### Session 1: Core features + handler registry
1. **Lexical recall** - FTS5 on DocumentStore, `LEXICAL SEARCH` command
2. **Vision ingest** - VisionHandler wired into pipeline, standalone image ingest
3. **Blob lifecycle** - `__blob_state__`, `FORGET NODE`, `SYS RETAIN`, retention config
4. **Section hierarchy** - `doc -> section -> chunk` with `__confidence__=0.6`
5. **Typed config** - msgspec Structs, `graphstore.json`, `GRAPHSTORE_CONFIG` env var
6. **Self-balancing optimizer** - `SYS HEALTH`, `SYS OPTIMIZE` (6 ops), auto-trigger
7. **Handler registry** - `@handles` decorator, auto-dispatch, 11 domain handler files

### Session 2: Quality improvements (PR #17)
1. **WALManager wired** - `wal.py` connected to `graphstore.py`, 5 inline WAL methods deleted, vector-store checkpoint bug fixed
2. **RECALL transpose cache** - `EdgeMatrices._combined_transpose` cached (O(1) after first call), invalidated on rebuild
3. **Batch DocumentStore remap** - `compact_tombstones` uses temp-table batch SQL instead of per-slot UPDATE loop; also fixes pre-existing UNIQUE constraint bug on tombstoned slots
4. **executor_base split** - 641-line monolith split into `visibility.py` (VisibilityMixin) + `filtering.py` (FilteringMixin); `executor_base.py` reduced to ~30 lines
5. **Public API surface** - `get_all_nodes()`, `get_all_edges()`, `cost_threshold`, `ceiling_mb` on GraphStore; `server.py` uses public API only

### Performance
| What | Speedup |
|------|---------|
| Model2Vec cache | 40x (test suite 102s -> 2.5s) |
| Plaintext ingest fast-path | 273x |
| msgspec.json serialization | 6x |
| Batch SQLite commits | 48x |
| Tombstone mask cache | 22x |
| Vectorized optimizer | 17x |
| Batch embedding | N -> 1 model calls |
| RECALL transpose cache | O(nnz) -> O(1) per repeated query |
| DocumentStore slot remap | O(N) round-trips -> O(1) batch SQL |

### Codebase
- Deleted `executor_reads.py` (1150 lines) + `executor_writes.py` (958 lines)
- Created 11 handler files under `dsl/handlers/` (41-476 lines each)
- Split `executor_base.py` into `visibility.py` + `filtering.py`
- Wired `WALManager`, deleted 5 inline methods from `graphstore.py`
- Removed tracked artifacts (`zen-graph-db/`, `logs/`, `indexes/`)
- 843 tests, ~2.4s runtime

## Prior art

Concepts from two repos influenced graphstore's retrieval architecture (see `mcp_refs/prior-art-recall-pageindex.md`):

| Repo | What graphstore took |
|------|---------------------|
| [arniesaha/recall](https://github.com/arniesaha/recall) | BM25/FTS5 pattern for LEXICAL SEARCH |
| [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) | `doc -> section -> chunk` hierarchy; graph-traversal retrieval via RECALL as analogue to LLM-guided tree descent |

## Test fixtures

`scripts/download_fixtures.py` downloads ~47MB of test fixtures (gitignored). Run once per dev environment.

| Category | Count | Size | Source |
|----------|-------|------|--------|
| Text | 20 books | 15MB | Project Gutenberg (fiction, non-fiction, short stories) |
| PDF | 6 papers | 18MB | arXiv (attention, BERT, RAG, toolformer, node2vec, word2vec) |
| HTML | 5 articles | 2MB | Wikipedia (transformer, LLM, knowledge graph, RAG, vector DB) |
| Markdown | 5 READMEs | 140KB | GitHub vector DB repos (faiss, chroma, qdrant, milvus, pgvector) |
| CSV | 5 datasets | 64KB | UCI/seaborn (iris, tips, countries, penguins, mpg) |
| Images | 17 photos | 520KB | Wikimedia CC0 (dog, cat, elephant, pizza, guitar, tower, etc.) |
| Voice | 25 clips | 12MB | OpenSLR: Hindi, English, Tamil, Telugu, Marathi (real human speakers) |

## Known gaps

| Issue | Severity | Status |
|-------|----------|--------|
| ~~WAL not wired~~ | ~~Medium~~ | Fixed (PR #17) |
| ~~executor_base.py bloat~~ | ~~Medium~~ | Fixed (PR #17) |
| ~~Optimizer SQL perf~~ | ~~Low~~ | Fixed (PR #17) |
| executor_system.py untouched | Low | 841 lines, not moved to handler registry pattern |
| No E2E optimize test | Medium | Missing: ingest -> delete -> optimize -> query -> checkpoint -> reload |
| ast_nodes.py flat | Low | 534 lines, could split into subpackage by domain |
| Mock test layer not wired | Medium | Fixtures downloaded, pytest integration pending |

## Architecture decisions

- **Single agent per DB** - no locking, optimizer safe under exclusive access
- **Config: one file, sectioned by engine** - not per-engine config files
- **Inferred structure: durable but not sacred** - section nodes carry `__confidence__=0.6`
- **Blob lifecycle != memory lifecycle** - `SYS RETAIN` deletes bytes, `FORGET NODE` deletes everything
- **Auto-optimize disabled by default** - agent controls when via `SYS HEALTH` + `SYS OPTIMIZE`
- **Adding a DSL command = 2 files** - grammar rule + handler file with `@handles`
- **Public API over private access** - `server.py` uses `get_all_nodes()` / `cost_threshold` / `ceiling_mb` instead of `_store._ceiling_bytes`
- **Mixin-based executor** - `ExecutorBase` inherits `VisibilityMixin` + `FilteringMixin`, each <100 lines

## Speed Demon evaluation

| Tool | Verdict | Why |
|------|---------|-----|
| msgspec | **USING** | 6x over json, already a dependency |
| orjson | Skip | 1.37x over msgspec encode, 0.89x decode. Marginal, extra dep. |
| polars | Wrong fit | Slot-indexed numpy arrays != dataframes. +80MB bundle. |
| uvloop | N/A | No event loop (single-threaded sync DB) |
| robyn | N/A | Server is 200 LOC wrapper, not bottleneck |

## What to pick up next

1. Move `executor_system.py` handlers into registry pattern
2. Add E2E integration test for optimize cycle
3. Wire mock test layer (`tests/fixtures/`) into pytest with `@pytest.mark.fixture` tests
4. MCP server implementation (see `mcp_refs/` for design decisions)
