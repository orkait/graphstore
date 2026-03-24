<div align="center">

# graphstore

**Agentic brain DB - the cognitive layer for AI agents**

[![CI](https://github.com/orkait/graphstore/actions/workflows/ci.yml/badge.svg)](https://github.com/orkait/graphstore/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-22c55e?logo=opensourceinitiative&logoColor=white)](https://github.com/orkait/graphstore/blob/main/LICENSE)
[![Version](https://img.shields.io/badge/version-0.3.0-f59e0b?logo=semver&logoColor=white)](https://github.com/orkait/graphstore)
[![NumPy](https://img.shields.io/badge/numpy-%23013243?logo=numpy&logoColor=white)](https://numpy.org)
[![SciPy](https://img.shields.io/badge/scipy-%238CAAE6?logo=scipy&logoColor=white)](https://scipy.org)
[![usearch](https://img.shields.io/badge/usearch-%23FF6B35?logo=data:image/svg+xml;base64,&logoColor=white)](https://github.com/unum-cloud/usearch)

</div>

---

Six engines, one DSL. Columnar numpy storage, sparse matrix traversal, HNSW vector search, document ingestion, markdown vault, and a human-readable query language - everything an AI agent needs to remember, recall, reason, and speak. Thread-safe command queue, persistent cron scheduler, intelligent logging, and memory-safe eviction for 24/7 operation.

## 🧩 What agents get

| Need | graphstore solves it | Speed (100k nodes) |
|---|---|---|
| **Hybrid recall** | `REMEMBER "Paris travel" TOKENS 4000` (5-signal fusion with token budget) | ~200 μs |
| **Recall by meaning** | `SIMILAR TO "Paris travel" LIMIT 10` | 127 μs |
| **Recall by association** | `RECALL FROM "concept:paris" DEPTH 3` | 983 μs |
| **Memory summarization** | `AGGREGATE NODES GROUP BY topic SELECT COUNT(), AVG(importance)` | 788 μs |
| **Belief tracking** | `ASSERT "fact:x" CONFIDENCE 0.9` / `RETRACT "fact:x"` | 9 μs |
| **Contradiction detection** | `SYS CONTRADICTIONS WHERE kind = "belief" FIELD value GROUP BY topic` | 981 μs |
| **Hypothesis testing** | `SYS SNAPSHOT "before"` ... `SYS ROLLBACK TO "before"` | 1.8 ms |
| **Lexical recall** | `LEXICAL SEARCH "exact phrase" LIMIT 10` | 52 μs |
| **Document ingestion** | `INGEST "report.pdf"` (auto-parse, section hierarchy, embed, wire) | < 2 sec |
| **Visual memory** | `INGEST "photo.png" USING VISION "smolvlm2"` (VLM description) | < 3 sec |
| **Cross-doc connections** | `SYS CONNECT` (auto-wire similar chunks across documents) | < 5 sec |
| **Blob lifecycle** | `SYS RETAIN` (warm -> archived -> deleted by retention policy) | 89 μs |
| **Hard forget** | `FORGET NODE "mem1"` (blob + vector + memory, irreversible) | 12 μs |
| **Working memory** | `CREATE NODE ... EXPIRES IN 30m` + `SYS EXPIRE` | 9 μs |
| **Temporal queries** | `NODES WHERE __created_at__ > NOW() - 7d` | 102 μs |
| **Isolated reasoning** | `BIND CONTEXT "session"` ... `DISCARD CONTEXT "session"` | 72 μs |
| **Scheduled maintenance** | `SYS CRON ADD "expire" SCHEDULE "0 * * * *" QUERY "SYS EXPIRE"` | persistent |
| **Activity log** | `SYS LOG LIMIT 20` / `SYS LOG TRACE "session-42"` | queryable |
| **Memory safety** | `SYS EVICT` (emergency eviction when approaching ceiling) | auto |
| **Thread-safe access** | `GraphStore(threaded=True)` + `submit_background()` | zero-lock |
| **Point lookup** | `NODE "memory:42"` | 4 μs |

## 🏗️ Six Engines

<div align="center">
<img src="assets/engines.svg" alt="graphstore six engines architecture" width="100%">
</div>

## 📦 Install

```bash
pip install graphstore
```

Core includes: numpy, scipy, lark, usearch, model2vec, croniter, msgspec (~90 MB)

<details>
<summary><strong>Opt-in upgrades</strong></summary>

```bash
# Higher quality embeddings (EmbeddingGemma-300M via ONNX)
graphstore install-embedder embeddinggemma

# Image understanding in documents (SmolVLM2 / Qwen3-VL via Ollama)
graphstore install-vision smolvlm2
graphstore install-vision qwen3-vl

# Voice: speech-to-text + text-to-speech (Moonshine + Piper)
graphstore install-voice
```

</details>

## 🚀 Quick Start

```python
from graphstore import GraphStore

g = GraphStore(path="./brain", threaded=True)

# Store memories
g.execute('CREATE NODE "memory:paris" kind = "memory" topic = "travel" importance = 0.9')
g.execute('CREATE NODE "memory:eiffel" kind = "memory" topic = "travel" importance = 0.8')
g.execute('CREATE EDGE "memory:paris" -> "memory:eiffel" kind = "associated"')

# Hybrid recall - best 4000 tokens of context (vector + BM25 + recency + confidence)
result = g.execute('REMEMBER "European architecture" TOKENS 4000')

# Recall by association (spreading activation with decay)
result = g.execute('RECALL FROM "memory:paris" DEPTH 2 LIMIT 10')

# Ingest a document (auto-parse, chunk, embed full text, wire)
g.execute('INGEST "report.pdf" AS "doc:q3" KIND "report"')
g.execute('SYS CONNECT')  # auto-wire similar chunks across documents

# Search by meaning or keywords
result = g.execute('SIMILAR TO "Q3 revenue growth" LIMIT 5')
result = g.execute('LEXICAL SEARCH "quarterly revenue" LIMIT 5')
doc = g.execute('NODE "doc:q3:chunk:3" WITH DOCUMENT')  # fetch full text

# Schedule maintenance (runs in background, doesn't block queries)
g.execute('SYS CRON ADD "cleanup" SCHEDULE "@hourly" QUERY "SYS EXPIRE"')
g.execute('SYS CRON ADD "optimize" SCHEDULE "0 3 * * *" QUERY "SYS OPTIMIZE"')
```

## 🧠 Agent Memory Features

<details>
<summary><strong>Belief Management</strong> - assert, retract, detect contradictions</summary>

```sql
ASSERT "fact:earth-radius" value = 6371 kind = "fact" CONFIDENCE 0.99 SOURCE "physics-tool"
RETRACT "fact:old-preference" REASON "user corrected this"
UPDATE NODES WHERE kind = "fact" AND source = "unreliable" SET confidence = 0.1
SYS CONTRADICTIONS WHERE kind = "belief" FIELD value GROUP BY topic
```

</details>

<details>
<summary><strong>Semantic Search</strong> - find memories by meaning</summary>

```sql
SIMILAR TO "European travel" LIMIT 10
SIMILAR TO "revenue growth" LIMIT 10 WHERE kind = "chunk" AND confidence > 0.5
SIMILAR TO NODE "concept:paris" LIMIT 10
SIMILAR TO [0.12, -0.34, 0.56, ...] LIMIT 10
LEXICAL SEARCH "exact phrase" LIMIT 10         -- BM25 full-text search
LEXICAL SEARCH "error log" WHERE kind = "chunk" -- with filters
SYS DUPLICATES THRESHOLD 0.95
SYS CONNECT THRESHOLD 0.85
```

</details>

<details>
<summary><strong>Document Ingestion</strong> - PDF, Word, text, images, audio with section hierarchy</summary>

```sql
-- Tiered parsing: MarkItDown → PyMuPDF4LLM → Docling → VLM
INGEST "report.pdf" AS "doc:q3" KIND "report"
INGEST "notes.docx" USING markitdown
INGEST "financials.pdf" USING docling

-- Standalone images with VLM description
INGEST "diagram.png" USING VISION "smolvlm2"

-- Documents auto-build: doc → section → chunk hierarchy
-- Section nodes carry __confidence__ = 0.6 (inferred structure)

-- Fetch full document text (from disk, on demand)
NODE "doc:q3:chunk:7" WITH DOCUMENT

-- Auto-wire similar chunks across all documents
SYS CONNECT
```

</details>

<details>
<summary><strong>Vault</strong> - structured markdown notes for agent memory</summary>

```python
g = GraphStore(path="./brain", vault="./notes")

# Agent writes notes as structured markdown files
g.execute('VAULT NEW "Project Requirements" KIND "context" TAGS "project,specs"')
g.execute('VAULT WRITE "Project Requirements" SECTION "body" CONTENT "The app must support..."')

# Vault files are human-readable markdown, auto-indexed in graphstore
g.execute('VAULT SEARCH "deployment requirements" LIMIT 5')

# Load standing instructions at session start
instructions = g.execute('VAULT LIST WHERE note_kind = "instruction" AND status = "active"')

# Daily working notes
g.execute('VAULT DAILY')
g.execute('VAULT APPEND "2026-03-21" SECTION "body" CONTENT "Completed task X"')

# Facts auto-assert with confidence
# (note with kind: fact in frontmatter auto-sets __confidence__)
g.execute('VAULT SYNC')  # re-index after external edits

# Wikilinks become graph edges
# [[related-note]] in Links section → EDGES FROM "note:x"
g.execute('VAULT BACKLINKS "my-note"')
```

Note kinds: `instruction`, `goal`, `context`, `plan`, `memory`, `artifact`, `log`, `daily`, `entity`, `fact`, `scratch`

</details>

<details>
<summary><strong>Blob Lifecycle + Forgetting</strong> - retention policy, hard delete</summary>

```sql
-- Hard forget: removes blob + vector + graph node (irreversible)
FORGET NODE "memory:old"          -- cascades to children for documents

-- Retention policy scan (warm → archived → blob deleted by age)
SYS RETAIN
-- Configurable: GraphStore(retention={"blob_warm_days": 30, "blob_archive_days": 90, "blob_delete_days": 365})

-- Blob deletion != memory forgetting:
-- SYS RETAIN deletes bytes but keeps the graph node + summary
-- FORGET NODE removes everything
```

</details>

<details>
<summary><strong>Hypothesis Testing</strong> - snapshot, explore, rollback</summary>

```sql
SYS SNAPSHOT "before-hypothesis"
-- ... explore reasoning branch ...
SYS ROLLBACK TO "before-hypothesis"

WHAT IF RETRACT "belief:earth-is-flat"
-- Returns affected nodes without committing
```

</details>

<details>
<summary><strong>Temporal + Working Memory</strong></summary>

```sql
CREATE NODE "scratch:123" kind = "working" data = "..." EXPIRES IN 30m
NODES WHERE __created_at__ > NOW() - 7d
NODES WHERE __updated_at__ > TODAY ORDER BY __created_at__ DESC
SYS EXPIRE WHERE kind = "working"
MERGE NODE "memory:old" INTO "memory:canonical"
```

</details>

<details>
<summary><strong>Aggregations</strong> - GROUP BY with numpy-vectorized ops</summary>

```sql
AGGREGATE NODES WHERE kind = "memory"
  GROUP BY topic
  SELECT COUNT(), AVG(importance)
  HAVING COUNT() > 2
  ORDER BY AVG(importance) DESC
```

</details>

<details>
<summary><strong>Hybrid Retrieval</strong> - REMEMBER fuses 5 signals with token budget</summary>

```sql
-- Single command for agent context retrieval
REMEMBER "quantum entanglement" TOKENS 4000
REMEMBER "deployment architecture" LIMIT 5 WHERE kind = "fact"
REMEMBER "project status" TOKENS 8000 WHERE kind = "memory"

-- Results include full score breakdown
-- _remember_score, _vector_sim, _bm25_score, _recency_score, _confidence
```

5-signal scoring (configurable via `graphstore.json`):
```
0.30 * vector_similarity    -- cosine distance from embedder
0.20 * bm25_normalized      -- full-text BM25 on chunk content
0.15 * recency              -- exp(-age_days / 30)
0.20 * confidence           -- from __confidence__ column (beliefs)
0.15 * recall_frequency     -- how often this memory was retrieved before
```

`TOKENS N` fills up to N tokens (estimated as `len(text) // 4`) instead of returning a fixed count. Agent gets exactly the right context budget.

Retrieval feedback: REMEMBER auto-increments `__recall_count__` and `__last_recalled_at__` on returned nodes. Frequently useful memories become easier to find.

</details>

<details>
<summary><strong>CRON Scheduler</strong> - persistent scheduled jobs with full cron syntax</summary>

```sql
SYS CRON ADD "expire-ttl" SCHEDULE "0 * * * *" QUERY "SYS EXPIRE"
SYS CRON ADD "nightly-optimize" SCHEDULE "0 3 * * *" QUERY "SYS OPTIMIZE"
SYS CRON ADD "reembed-weekly" SCHEDULE "0 2 * * 0" QUERY "SYS REEMBED"
SYS CRON ADD "vault-sync" SCHEDULE "*/5 * * * *" QUERY "VAULT SYNC"
SYS CRON ADD "health-check" SCHEDULE "@hourly" QUERY "SYS HEALTH"

SYS CRON LIST            -- show all jobs with next_run, run_count, errors
SYS CRON DISABLE "job"   -- pause without deleting
SYS CRON RUN "job"       -- manual trigger for testing
SYS CRON DELETE "job"
```

Full cron expressions: `*/15 9-17 * * MON-FRI`, `@hourly`, `@daily`, `@weekly`. Jobs persist in SQLite and survive restarts. Requires `threaded=True`.

</details>

<details>
<summary><strong>Intelligent Logging</strong> - auto-tagged, traceable, queryable</summary>

```sql
-- Every query is auto-tagged: read, write, intelligence, belief, ingest, vault, system
SYS LOG LIMIT 20
SYS LOG TRACE "research-session-42"    -- find all queries in a trace
SYS LOG SINCE "2026-03-24" LIMIT 50
SYS LOG WHERE tag = "intelligence"     -- filter by semantic tag
```

```python
# Trace binding for causality tracking
gs.bind_trace("research-42")
gs.execute('RECALL FROM "quantum" DEPTH 3')
gs.execute('CREATE NODE "insight" kind = "fact" ...')
gs.discard_trace()
# Both queries tagged with trace_id = "research-42"

# Structured events via Python logging (pipe anywhere)
import logging
logging.getLogger("graphstore.events").addHandler(your_handler)
```

REST API: `GET /api/logs?tag=intelligence&limit=20`

</details>

<details>
<summary><strong>Thread Safety</strong> - multi-agent access via command queue</summary>

```python
gs = GraphStore(path="./brain", threaded=True)

# Multiple agents can call execute() from different threads
# All serialized through a priority queue (interactive > background)
result = gs.execute('RECALL FROM "cue" DEPTH 3')  # blocks, returns Result

# Background maintenance doesn't block interactive queries
future = gs.submit_background('SYS OPTIMIZE')
future = gs.submit_background('SYS CONNECT')
```

</details>

<details>
<summary><strong>Memory Safety</strong> - accurate accounting + emergency eviction for Docker</summary>

```sql
SYS STATUS     -- includes memory_measured breakdown (real component sizes)
SYS HEALTH     -- memory_utilization ratio (actual/ceiling)
SYS EVICT      -- emergency eviction of oldest non-protected nodes
```

```python
# For Docker containers with fixed RAM
gs = GraphStore(path="./brain", ceiling_mb=256, threaded=True)
# Auto-eviction triggers at 90% ceiling via health checks
# Protected kinds (schema, config, system) are never evicted
```

</details>

<details>
<summary><strong>Context Isolation</strong></summary>

```sql
BIND CONTEXT "reasoning-session-42"
CREATE NODE "hyp:1" kind = "hypothesis" content = "maybe X"
RECALL FROM "hyp:1" DEPTH 3 LIMIT 10
DISCARD CONTEXT "reasoning-session-42"
```

</details>

<details>
<summary><strong>Voice</strong> (opt-in: graphstore install-voice)</summary>

```python
g = GraphStore(path="./brain", voice=True)

# Agent speaks
g.speak("The Q3 revenue grew 15%")

# Agent listens (real-time streaming)
g.listen(on_text=lambda text: agent.process(text))
g.stop_listening()

# Ingest audio files
g.execute('INGEST "meeting.wav"')  # Moonshine transcribes → chunks → embeds
```

</details>

## 📖 DSL Reference

<details>
<summary><strong>Reads</strong> - queries, traversals, path finding, pattern matching, semantic search</summary>

```sql
NODE "node_id"
NODE "node_id" WITH DOCUMENT
NODES WHERE kind = "function" AND file = "app.py" LIMIT 10
EDGES FROM "node_id" WHERE kind = "calls"
TRAVERSE FROM "node_id" DEPTH 3 WHERE kind = "calls"
SUBGRAPH FROM "node_id" DEPTH 2
PATH FROM "a" TO "b" MAX_DEPTH 5
SHORTEST PATH FROM "a" TO "b"
WEIGHTED SHORTEST PATH FROM "a" TO "b"
ANCESTORS OF "node_id" DEPTH 3
DESCENDANTS OF "node_id" DEPTH 3
COMMON NEIGHBORS OF "a" AND "b"
MATCH ("fn_main") -[kind = "calls"]-> (callee)
COUNT NODES WHERE kind = "function"
AGGREGATE NODES GROUP BY kind SELECT COUNT()
RECALL FROM "concept:x" DEPTH 3 LIMIT 10
SIMILAR TO "search text" LIMIT 10
SIMILAR TO [0.1, 0.2, ...] LIMIT 10 WHERE kind = "memory"
LEXICAL SEARCH "exact phrase" LIMIT 10
REMEMBER "hybrid query" TOKENS 4000 WHERE kind = "fact"
REMEMBER "query" LIMIT 10
WHAT IF RETRACT "belief:x"
```

</details>

<details>
<summary><strong>Writes</strong> - create, update, delete, beliefs, documents, voice</summary>

```sql
CREATE NODE "id" kind = "x" name = "foo"
CREATE NODE "id" kind = "x" DOCUMENT "full text..." EXPIRES IN 1h
UPDATE NODE "id" SET name = "new"
UPSERT NODE "id" kind = "x" name = "foo"
DELETE NODE "id"
DELETE NODES WHERE kind = "test"
UPDATE NODES WHERE kind = "fact" SET confidence = 0.5
CREATE EDGE "source" -> "target" kind = "calls"
INCREMENT NODE "id" hits BY 1
ASSERT "fact:x" kind = "fact" value = 42 CONFIDENCE 0.9 SOURCE "tool"
RETRACT "fact:x" REASON "outdated"
MERGE NODE "old" INTO "canonical"
PROPAGATE "belief:x" FIELD confidence DEPTH 3
INGEST "report.pdf" AS "doc:q3" KIND "report"
CONNECT NODE "chunk:7" THRESHOLD 0.8
FORGET NODE "old-memory"
BIND CONTEXT "session-1"
DISCARD CONTEXT "session-1"
BEGIN ... COMMIT
```

</details>

<details>
<summary><strong>System</strong> - stats, schema, maintenance, model management</summary>

```sql
SYS STATUS
SYS STATS / SYS STATS NODES / SYS STATS MEMORY / SYS STATS DOCUMENTS
SYS REGISTER NODE KIND "memory" REQUIRED topic:string, importance:float EMBED content
SYS KINDS / SYS EDGE KINDS / SYS DESCRIBE NODE "memory"
SYS CONTRADICTIONS WHERE kind = "belief" FIELD value GROUP BY topic
SYS CONNECT / SYS CONNECT THRESHOLD 0.9
SYS DUPLICATES THRESHOLD 0.95
SYS EXPIRE / SYS EXPIRE WHERE kind = "working"
SYS SNAPSHOT "name" / SYS ROLLBACK TO "name" / SYS SNAPSHOTS
SYS EMBEDDERS / SYS SET EMBEDDER "embeddinggemma-300m" / SYS REEMBED
SYS INGESTORS
SYS RETAIN
SYS CHECKPOINT / SYS REBUILD INDICES / SYS CLEAR CACHE
SYS EXPLAIN TRAVERSE FROM "a" DEPTH 5
SYS SLOW QUERIES LIMIT 10 / SYS FAILED QUERIES LIMIT 10
SYS LOG LIMIT 20 / SYS LOG TRACE "id" / SYS LOG SINCE "ISO-8601"
SYS CRON ADD "name" SCHEDULE "0 * * * *" QUERY "SYS EXPIRE"
SYS CRON DELETE / ENABLE / DISABLE / LIST / RUN "name"
SYS EVICT
SYS HEALTH / SYS OPTIMIZE
```

</details>

<details>
<summary><strong>Vault</strong> - markdown note operations</summary>

```sql
VAULT NEW "title" KIND "memory" TAGS "tag1,tag2"
VAULT READ "title"
VAULT WRITE "title" SECTION "body" CONTENT "..."
VAULT APPEND "title" SECTION "body" CONTENT "..."
VAULT SEARCH "query" LIMIT 10
VAULT BACKLINKS "title"
VAULT LIST [WHERE ...] [ORDER BY ...] [LIMIT n]
VAULT SYNC
VAULT DAILY
VAULT ARCHIVE "title"
```

</details>

## ⚡ Performance (100k nodes)

| Operation | Time |
|---|---|
| Point lookup | 4 μs |
| Filtered scan LIMIT 10 | 68 μs |
| COUNT | 41 μs |
| ORDER BY LIMIT 10 | 168 μs |
| GROUP BY + AVG | 788 μs |
| SIMILAR TO LIMIT 10 | 127 μs |
| RECALL DEPTH 3 | 983 μs |
| ASSERT / RETRACT | 4-9 μs |
| UPDATE NODES 50k bulk | 171 μs |
| SYS SNAPSHOT | 1.8 ms |
| Memory per node | 66 bytes (columns) + 1 KB (vector) |

## ⚙️ Configuration

```python
g = GraphStore(
    path="./brain",          # persistence directory (None = in-memory)
    ceiling_mb=256,          # graph memory ceiling
    embedder="default",      # "default" (model2vec) or custom Embedder
    threaded=True,           # thread-safe command queue + cron scheduler
    voice=False,             # True to enable STT/TTS (opt-in)
    vault="./notes",         # markdown vault directory (None = disabled)
    retention={              # blob lifecycle policy
        "blob_warm_days": 30,
        "blob_archive_days": 90,
        "blob_delete_days": 365,
    },
)
```

<details>
<summary><strong>Full graphstore.json</strong> - 54 fields, all with defaults</summary>

```json
{
  "core": {
    "ceiling_mb": 256,
    "initial_capacity": 1024,
    "compact_threshold": 0.2,
    "string_gc_threshold": 3.0,
    "protected_kinds": ["schema", "config", "system"]
  },
  "vector": {
    "embedder": "default",
    "similarity_threshold": 0.85,
    "search_oversample": 5,
    "model2vec_model": "minishlab/M2V_base_output"
  },
  "document": {
    "chunk_max_size": 2000,
    "chunk_overlap": 50,
    "summary_max_length": 200,
    "fts_full_text": true,
    "vision_model": "smolvlm2:2.2b"
  },
  "dsl": {
    "cost_threshold": 100000,
    "plan_cache_size": 256,
    "recall_decay": 0.7,
    "remember_weights": [0.30, 0.20, 0.15, 0.20, 0.15],
    "auto_optimize": false,
    "optimize_interval": 500
  },
  "persistence": {
    "wal_hard_limit": 100000,
    "auto_checkpoint_threshold": 50000,
    "log_retention_days": 7
  },
  "retention": {
    "blob_warm_days": 30,
    "blob_archive_days": 90,
    "blob_delete_days": 365
  },
  "server": {
    "auth_token": null,
    "rate_limit_rpm": 120,
    "max_query_length": 10000,
    "max_batch_size": 1000
  }
}
```

</details>
```

## 🏗️ Architecture

<details>
<summary>Project structure</summary>

```
graphstore/
├── __init__.py               # Thin re-exports
├── graphstore.py             # GraphStore facade + routing
├── wal.py                    # WALManager: append, replay, checkpoint, query log
├── cron.py                   # CronScheduler: persistent jobs, daemon timer
├── config.py                 # Typed config via msgspec Structs
├── core/                     # Graph engine
│   ├── store.py              # CoreStore: columnar node CRUD
│   ├── columns.py            # ColumnStore: typed numpy arrays
│   ├── edges.py              # EdgeMatrices: scipy CSR
│   ├── strings.py            # StringTable: string interning
│   ├── schema.py             # SchemaRegistry + EMBED field
│   ├── path.py               # BFS, Dijkstra, common neighbors
│   ├── memory.py             # Accurate memory measurement + ceiling
│   ├── optimizer.py           # Self-balancing: compact, GC, evict
│   ├── scheduler.py          # OptimizerScheduler: health + auto-optimize
│   ├── queue.py              # CommandQueue: thread-safe priority queue
│   ├── types.py              # Result, Edge dataclasses
│   └── errors.py             # Error hierarchy
├── dsl/                      # Query language
│   ├── grammar.lark          # Lark LALR(1) grammar
│   ├── parser.py             # Parser + LRU cache
│   ├── transformer.py        # Parse tree → AST
│   ├── ast_nodes.py          # 70+ AST dataclasses
│   ├── tagger.py             # Auto-tag inference for log layer
│   ├── executor_base.py      # Shared: live_mask, eval_where, column filters
│   ├── handlers/             # Auto-dispatch handler mixins
│   │   ├── nodes.py          # NODE, NODES, COUNT
│   │   ├── edges.py          # EDGES, CREATE/DELETE EDGE
│   │   ├── traversal.py      # TRAVERSE, PATH, ANCESTORS, DESCENDANTS
│   │   ├── pattern.py        # MATCH pattern queries
│   │   ├── aggregation.py    # AGGREGATE GROUP BY
│   │   ├── intelligence.py   # RECALL, SIMILAR, LEXICAL, REMEMBER
│   │   ├── beliefs.py        # ASSERT, RETRACT, PROPAGATE
│   │   ├── mutations.py      # CREATE/UPDATE/DELETE/MERGE/BATCH
│   │   ├── context.py        # BIND/DISCARD CONTEXT
│   │   └── ingest.py         # INGEST, CONNECT NODE
│   ├── executor.py           # Unified dispatcher (MRO from handlers)
│   ├── executor_system.py    # SYS + CRON + LOG commands
│   └── cost_estimator.py     # Frontier-based cost rejection
├── embedding/                # Text → vectors
│   ├── base.py               # Embedder protocol
│   ├── model2vec_embedder.py # Default (30MB, CPU)
│   ├── onnx_hf_embedder.py   # EmbeddingGemma ONNX (opt-in)
│   └── postprocess.py        # L2 normalize, Matryoshka truncation
├── vector/                   # Semantic search
│   ├── index.py              # usearch HNSW wrapper
│   └── store.py              # VectorStore: slot ↔ vector
├── document/                 # Document storage
│   └── store.py              # DocumentStore: SQLite multi-table
├── ingest/                   # File → graph
│   ├── base.py               # Ingestor protocol
│   ├── markitdown_ingestor.py  # Tier 1: general files
│   ├── pymupdf4llm_ingestor.py # Tier 2: PDF structure
│   ├── docling_ingestor.py   # Tier 3: hard PDFs (lazy)
│   ├── chunker.py            # Text splitting + summaries
│   ├── vision.py             # SmolVLM2/Qwen3-VL via Ollama
│   ├── router.py             # Tiered routing
│   └── connector.py          # SYS CONNECT cross-doc wiring
├── voice/                    # Speech (opt-in)
│   ├── stt.py                # Moonshine STT
│   └── tts.py                # Piper TTS
├── vault/                    # Markdown notes
│   ├── parser.py             # Frontmatter, sections, wikilinks
│   ├── manager.py            # Note CRUD (new/read/write/append)
│   ├── sync.py               # Vault dir → graphstore sync
│   └── executor.py           # VAULT DSL command handler
├── registry/                 # Model management
│   ├── models.py             # Supported models config
│   ├── installer.py          # Download + verify + smoke test
│   └── manifest.py           # Model manifest schema
├── persistence/              # SQLite checkpoints
│   ├── database.py
│   ├── serializer.py
│   └── deserializer.py
├── server.py                 # FastAPI playground
└── cli.py                    # CLI commands
```

</details>

## 🎮 Playground

```bash
pip install graphstore[playground]
graphstore playground
```

Three-panel UI: CodeMirror editor, React Flow graph visualization, stacked query results.

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
