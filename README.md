<div align="center">

# graphstore

**Memory infrastructure for AI agents**

[![CI](https://github.com/orkait/graphstore/actions/workflows/ci.yml/badge.svg)](https://github.com/orkait/graphstore/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/graphstore?color=f59e0b&logo=pypi&logoColor=white)](https://pypi.org/project/graphstore/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/graphstore?color=f59e0b&logo=pypi&logoColor=white)](https://pypi.org/project/graphstore/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-22c55e?logo=opensourceinitiative&logoColor=white)](LICENSE)
[![SQLite](https://img.shields.io/badge/storage-SQLite-003B57?logo=sqlite&logoColor=white)](https://sqlite.org)
[![usearch](https://img.shields.io/badge/vector-HNSW%20%2F%20usearch-FF6B35?logoColor=white)](https://github.com/unum-cloud/usearch)

</div>

---

graphstore is a Python library that gives AI agents a persistent, queryable memory. You store nodes and edges with a simple DSL, and retrieve them by meaning, by association, by text search, or by any combination - all from one call.

It is designed for agent frameworks, LLM applications, and research tools that need more than a vector database but less than a full graph database. Everything lives in-process and persists to SQLite. No server to run, no infrastructure to manage.

---

## 🏗️ Architecture

Three storage engines, one typed DSL, a layered feature set on top, and a pair of optional subsystems.

```text
              ┌─────────────────────────────────────────┐
              │        DSL — Lark LALR(1) grammar       │
              │    one query language for everything    │
              └────────────────────┬────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
        ┌──────────┐         ┌──────────┐         ┌──────────┐
        │  Graph   │         │  Vector  │         │ Document │
        ├──────────┤         ├──────────┤         ├──────────┤
        │ numpy    │         │ usearch  │         │ SQLite   │
        │ columns  │         │  HNSW    │         │ FTS5 BM25│
        │ scipy CSR│         │  cosine  │         │ blobs    │
        │ + indices│         │          │         │ + summary│
        └──────────┘         └──────────┘         └──────────┘
              ▲                    ▲                    ▲
              └────────────────────┼────────────────────┘
                                   │
                         ┌─────────┴──────────┐
                         │  Ingest pipeline   │
                         │  markitdown · pdf  │
                         │  docling · vision  │
                         └────────────────────┘
```

**Layered features** (reach into the three engines via the DSL):

| Feature | What it does | Where it lives |
|---|---|---|
| **Beliefs** | `ASSERT` / `RETRACT` / `PROPAGATE` with confidence, source, retraction | reserved columns on Graph |
| **Evolution** | `SYS EVOLVE` WHEN/THEN rules — self-tuning on live signals (opt-in) | `EvolutionEngine` + SQLite history |
| **Snapshots** | `SYS SNAPSHOT` / `SYS ROLLBACK` — full-state rewind | in-memory store snapshots |
| **Cron** | `SYS CRON ADD/LIST/RUN` — scheduled DSL queries | persistent cron_jobs table |
| **WAL** | append → replay on open, auto-checkpoint at 50k | SQLite WAL + blobs |
| **Ceiling** | pre-flight RAM guard, self-calibrating per-node estimate | `check_ceiling` on every write |

**Optional subsystems** (off by default):

| Subsystem | What it adds | Toggle |
|---|---|---|
| **Vault** | markdown notebook synced into Graph | `GraphStore(vault="./notes")` |
| **Voice** | Moonshine STT + Piper TTS | `GraphStore(voice=True)` |

---

## 📦 Installation

**Requires Python 3.10 or higher.**

```bash
pip install graphstore
```

This is the lightweight core: numpy, scipy, usearch, lark, msgspec. Five runtime deps, no torch, no PDF parser, no HTTP server. The graph + vector + doc engines are fully functional — you can create nodes/edges, run BFS/Dijkstra paths, query by kind/property, and persist via WAL+snapshots out of the box.

For anything beyond the core engine, opt into an extra:

```bash
# zero-config default embedder (30 MB, CPU-only)
pip install 'graphstore[embed-default]'

# 30+ ONNX models (BGE, mxbai, e5, nomic, jina-v2)
pip install 'graphstore[embed-fastembed]'

# PDF / DOCX / HTML → markdown ingestion
pip install 'graphstore[ingest]'

# Obsidian-style vault sync
pip install 'graphstore[vault]'

# cron-scheduled DSL jobs
pip install 'graphstore[scheduler]'

# local web playground (FastAPI + React UI)
pip install 'graphstore[playground]'

# GPU acceleration for the ONNX embedder path (onnxruntime-gpu + cu12 wheels)
pip install 'graphstore[gpu]'
```

Extras compose — `pip install 'graphstore[embed-default,ingest,vault,scheduler]'` gets you a full agent-memory stack. When a feature is used without its extra installed, graphstore raises a targeted `ImportError` pointing at the right `pip install` recipe, not a cryptic `ModuleNotFoundError`.

<details>
<summary><strong>Full extras reference</strong></summary>

| Extra | What it adds |
|---|---|
| `embed-default` | model2vec — zero-config CPU embedder |
| `embed-fastembed` | fastembed — ~30 pre-exported ONNX encoder models |
| `ingest` | markitdown + pymupdf + pymupdf4llm (~80 MB, PDF/DOCX/HTML → markdown) |
| `ingest-pro` | docling + openai (heavier PDF + vision via LLM) |
| `scheduler` | croniter (cron-expression parsing for `SYS CRON ADD`) |
| `vault` | pyyaml (Obsidian-style markdown vault sync) |
| `playground` | fastapi + uvicorn + pydantic (local web UI) |
| `gpu` | onnxruntime-gpu + nvidia cu12 runtime wheels; activate via `GRAPHSTORE_GPU=1` |
| `voice` | sounddevice + moonshine-voice + piper-tts |
| `dev` | pytest + pytest-benchmark + pytest-cov |

For vision-powered ingestion (images, diagrams), you also need [Ollama](https://ollama.com) running:

```bash
ollama pull smolvlm2:2.2b
```

</details>

---

## 🚀 Quickstart

```python
from graphstore import GraphStore

g = GraphStore(path="./brain")

# Store two memories
g.execute('CREATE NODE "mem:paris" kind = "memory" topic = "travel" importance = 0.9')
g.execute('CREATE NODE "mem:eiffel" kind = "memory" topic = "travel" importance = 0.8')
g.execute('CREATE EDGE "mem:paris" -> "mem:eiffel" kind = "associated"')

# Recall by meaning - returns best matches within a 4000-token budget
result = g.execute('REMEMBER "European architecture" TOKENS 4000')

# Recall by association - follows edges from a starting node
result = g.execute('RECALL FROM "mem:paris" DEPTH 2 LIMIT 10')

# Exact text search
result = g.execute('LEXICAL SEARCH "Eiffel Tower" LIMIT 5')
```

The graph persists to `./brain/` as SQLite. Next time you create a `GraphStore` with the same path, all your memories are back.

---

## 🧠 What you can do

### Store and recall memories

The core use case: store structured memories, retrieve the right ones for your agent's context window.

```sql
-- Create memories with any fields you need
CREATE NODE "mem:123" kind = "memory" topic = "finance" importance = 0.8

-- Recall by meaning (vector search + BM25 + recency + confidence)
REMEMBER "quarterly revenue trends" TOKENS 4000

-- Recall by association (spreading activation through edges)
RECALL FROM "concept:finance" DEPTH 3 LIMIT 10

-- Recall by exact text
LEXICAL SEARCH "Q3 revenue" LIMIT 10

-- Recall by vector similarity
SIMILAR TO "budget forecasting" LIMIT 10
```

`REMEMBER` fuses five signals and fills your token budget automatically. You get the most relevant memories that fit, not a fixed count.

<details>
<summary>How REMEMBER scores memories</summary>

```
0.30 × vector_similarity    (cosine distance from embedder)
0.20 × bm25_score           (full-text BM25 on content)
0.15 × recency              (exp(-age_days / 30))
0.20 × confidence           (__confidence__ column)
0.15 × recall_frequency     (how often this was retrieved before)
```

Weights are configurable in `graphstore.json`. `REMEMBER` also increments `__recall_count__` on returned nodes, so frequently-useful memories naturally bubble up over time.

</details>

---

### Ingest documents

Drop a file in and graphstore parses it, chunks it, embeds it, and wires it into the graph.

```sql
-- Ingest a PDF (auto-parse → section hierarchy → embed → store)
INGEST "report.pdf" AS "doc:q3" KIND "report"

-- Auto-connect similar chunks across all ingested documents
SYS CONNECT

-- Search across documents
SIMILAR TO "Q3 revenue growth" LIMIT 5
LEXICAL SEARCH "quarterly revenue" LIMIT 5

-- Fetch the full text of a chunk
NODE "doc:q3:chunk:3" WITH DOCUMENT
```

Supported out of the box: PDF, Word, Markdown, text, images (via VLM), audio (via speech-to-text). The ingestor chain tries fast parsers first (MarkItDown) and falls back to heavier ones (PyMuPDF, Docling) for hard files.

<details>
<summary>Custom ingestors and chunkers</summary>

```python
from graphstore.ingest.base import Ingestor, IngestResult, Chunk

class MyIngestor(Ingestor):
    name = "myformat"
    supported_extensions = ["myext"]

    def convert(self, file_path: str, **kwargs) -> IngestResult:
        text = my_parser.parse(file_path)
        return IngestResult(markdown=text, parser_used="myformat")

g = GraphStore(
    path="./brain",
    ingestors={"pdf": MyPDFIngestor(), "myext": MyIngestor()},
    chunker=MySentenceChunker(),
)
```

Custom ingestors override the built-in chain per extension. Anything not overridden falls through to the default tiered router.

</details>

---

### Track beliefs and facts

Agents deal with uncertain, contradictory information. graphstore has first-class support for confidence-weighted facts and belief retraction.

```sql
-- Assert a fact with a source and confidence
ASSERT "fact:earth-radius" value = 6371 kind = "fact" CONFIDENCE 0.99 SOURCE "physics-tool"

-- Retract when it becomes stale
RETRACT "fact:old-preference" REASON "user corrected this"

-- Find contradictions (same field, different values, same topic)
SYS CONTRADICTIONS WHERE kind = "belief" FIELD value GROUP BY topic

-- Explore a hypothesis without committing
WHAT IF RETRACT "belief:earth-is-flat"
```

---

### Manage memory lifecycle

Memories age. graphstore gives you tools to expire temporary ones, archive old blobs, and surgically delete anything.

```sql
-- Set a TTL on creation
CREATE NODE "scratch:temp" kind = "working" data = "..." EXPIRES IN 30m

-- Run TTL sweep (also schedulable via cron)
SYS EXPIRE WHERE kind = "working"

-- Hard delete - removes graph node, vector, and blob permanently
FORGET NODE "mem:old"

-- Archive blob bytes by age (node stays in graph, blob is deleted)
SYS RETAIN
```

---

### Snapshot and rollback

Before exploring a reasoning branch, snapshot the current state. If the hypothesis doesn't pan out, roll back.

```sql
SYS SNAPSHOT "before-hypothesis"
-- ... reasoning branch ...
SYS ROLLBACK TO "before-hypothesis"
```

---

### Schedule maintenance automatically

```python
g = GraphStore(path="./brain", queued=True)

# Persist cron jobs in SQLite - they survive restarts
g.execute('SYS CRON ADD "expire-ttl"  SCHEDULE "@hourly"      QUERY "SYS EXPIRE"')
g.execute('SYS CRON ADD "nightly-opt" SCHEDULE "0 3 * * *"    QUERY "SYS OPTIMIZE"')
g.execute('SYS CRON ADD "vault-sync"  SCHEDULE "*/5 * * * *"  QUERY "VAULT SYNC"')
```

Full cron expressions, `@hourly`, `@daily`, `@weekly` all work. Requires `queued=True`.

---

### Write self-healing rules

The evolution engine watches live signals and fires rules automatically when thresholds are crossed.

```sql
-- Re-embed when retrieval quality drops
SYS EVOLVE RULE "reindex-on-drift"
  WHEN recall_hit_rate <= 0.4
  THEN RUN SYS REEMBED
  COOLDOWN 86400

-- Auto-compact when tombstone bloat is high
SYS EVOLVE RULE "compact-on-bloat"
  WHEN tombstone_ratio >= 0.3
  THEN RUN SYS OPTIMIZE COMPACT
  COOLDOWN 3600

-- List, inspect, and manage rules
SYS EVOLVE LIST
SYS EVOLVE SHOW "reindex-on-drift"
SYS EVOLVE HISTORY LIMIT 10
```

<details>
<summary>All available signals and tunable parameters</summary>

**Signals (13):** `memory_pct`, `memory_mb`, `node_count`, `tombstone_ratio`, `string_bloat`, `recall_hit_rate`, `recall_misses`, `avg_similarity`, `eviction_count`, `query_rate`, `write_rate`, `edge_density`, `wal_pending`

**Tunable params (10):** `ceiling_mb`, `eviction_target_ratio`, `recall_decay`, `chunk_max_size`, `cost_threshold`, `optimize_interval`, `similarity_threshold`, `duplicate_threshold`, `remember_weights`, `protected_kinds`

**THEN actions:** `SET param = value`, `ADJUST param BY delta UNTIL target`, `ADD param "element"`, `REMOVE param "element"`, `RUN <any DSL query>`

</details>

---

### More features

<details>
<summary><strong>Vault</strong> - structured markdown notes</summary>

Point graphstore at a folder of markdown files and they become queryable nodes with edges for wikilinks.

```python
g = GraphStore(path="./brain", vault="./notes")
```

```sql
VAULT NEW "Project Requirements" KIND "context" TAGS "project,specs"
VAULT WRITE "Project Requirements" SECTION "body" CONTENT "The app must support..."
VAULT SEARCH "deployment requirements" LIMIT 5
VAULT BACKLINKS "my-note"
VAULT DAILY   -- create/open today's daily note
VAULT SYNC    -- re-index after external edits
```

Note kinds: `instruction`, `goal`, `context`, `plan`, `memory`, `artifact`, `log`, `daily`, `entity`, `fact`, `scratch`

</details>

<details>
<summary><strong>Voice</strong> - speech in, speech out (opt-in)</summary>

```python
g = GraphStore(path="./brain", voice=True)

g.speak("The Q3 revenue grew 15%")
g.listen(on_text=lambda text: agent.process(text))
g.execute('INGEST "meeting.wav"')  # transcribe → chunk → embed
```

Ships with Moonshine STT and Piper TTS. Swap either with any duck-typed adapter via `stt=` and `tts=` constructor args.

</details>

<details>
<summary><strong>Context isolation</strong> - sandboxed reasoning sessions</summary>

```sql
BIND CONTEXT "reasoning-session-42"
CREATE NODE "hyp:1" kind = "hypothesis" content = "maybe X"
RECALL FROM "hyp:1" DEPTH 3 LIMIT 10
DISCARD CONTEXT "reasoning-session-42"
```

</details>

<details>
<summary><strong>Aggregations</strong></summary>

```sql
AGGREGATE NODES WHERE kind = "memory"
  GROUP BY topic
  SELECT COUNT(), AVG(importance)
  HAVING COUNT() > 2
  ORDER BY AVG(importance) DESC
```

</details>

<details>
<summary><strong>Temporal queries</strong></summary>

```sql
NODES WHERE __created_at__ > NOW() - 7d
NODES WHERE __updated_at__ > TODAY ORDER BY __created_at__ DESC
MERGE NODE "memory:old" INTO "memory:canonical"
```

</details>

<details>
<summary><strong>Activity log</strong></summary>

Every query is auto-tagged. You can trace a full session or filter by semantic type.

```sql
SYS LOG LIMIT 20
SYS LOG TRACE "research-session-42"
SYS LOG SINCE "2026-03-24" LIMIT 50
SYS LOG WHERE tag = "intelligence"
```

```python
gs.bind_trace("research-42")
gs.execute('RECALL FROM "quantum" DEPTH 3')
gs.discard_trace()
```

</details>

<details>
<summary><strong>Thread safety</strong></summary>

graphstore is single-writer by design. The storage engine assumes one writer at a time - no MVCC, no row-level locking.

`queued=True` installs a single-worker submission queue in front of the write path. It is a thread-safety guarantee, not parallelism. Multiple caller threads can share one instance; every `execute()` call is serialized through one daemon worker. It also starts the cron scheduler.

```python
gs = GraphStore(path="./brain", queued=True)

result = gs.execute('RECALL FROM "cue" DEPTH 3')   # blocks caller, returns Result
future = gs.submit_background('SYS OPTIMIZE')       # queues behind interactive work
future = gs.submit_background('SYS CONNECT')
```

With `queued=False` (the default), you are responsible for not calling `execute()` from multiple threads simultaneously.

</details>

---

## 📖 DSL Reference

<details>
<summary><strong>Reads</strong></summary>

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
<summary><strong>Writes</strong></summary>

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
<summary><strong>System</strong></summary>

```sql
SYS STATUS
SYS STATS / SYS STATS NODES / SYS STATS MEMORY / SYS STATS DOCUMENTS
SYS HEALTH
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
SYS OPTIMIZE / SYS OPTIMIZE COMPACT
SYS EVOLVE RULE "name" WHEN signal OP value [AND ...] THEN action [COOLDOWN n] [PRIORITY n]
SYS EVOLVE LIST / SHOW "name" / ENABLE "name" / DISABLE "name" / DELETE "name"
SYS EVOLVE HISTORY [LIMIT n] / RESET
```

</details>

<details>
<summary><strong>Vault</strong></summary>

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

---

## ⚡ Performance

All numbers measured at 100k nodes on a mid-range laptop (no GPU).

| Operation | Latency |
|---|---|
| Point lookup (`NODE "id"`) | 4 μs |
| Filtered scan (`NODES WHERE ... LIMIT 10`) | 68 μs |
| Count | 41 μs |
| Order by, limit 10 | 168 μs |
| Group by + avg | 788 μs |
| Semantic search (`SIMILAR TO` LIMIT 10) | 127 μs |
| Graph traversal (`RECALL` DEPTH 3) | 983 μs |
| Assert / retract | 4-9 μs |
| Bulk update 50k nodes | 171 μs |
| Snapshot | 1.8 ms |
| Memory per node | 66 bytes (columns) + ~1 KB (vector) |

**Retrieval quality** (LongMemEval-S, 500 records, bge-small-en-v1.5, CPU-only):

| System | Accuracy / R@5 |
|---|---|
| **graphstore (skill adapter + REMEMBER)** | **97.6%** |
| MemPalace | 96.6% |
| OMEGA | 95.4% |
| MemMachine | 93.0% |

Full methodology, category breakdown, comparison to the public leaderboard, and reproduction instructions: see [**BENCHMARKS.md**](BENCHMARKS.md).

---

## ⚙️ Configuration

The constructor exposes the most common options:

```python
g = GraphStore(
    path="./brain",       # where to persist (None = in-memory only)
    ceiling_mb=256,       # memory ceiling for the graph
    embedder="default",   # "default" (model2vec) or a custom Embedder instance
    queued=True,          # single-worker submission queue + cron scheduler
    voice=False,          # True to enable built-in STT/TTS
    vault="./notes",      # path to markdown vault (None = disabled)
    retention={           # blob lifecycle - when to archive/delete file bytes
        "blob_warm_days": 30,
        "blob_archive_days": 90,
        "blob_delete_days": 365,
    },
)
```

For deeper tuning, drop a `graphstore.json` file next to your persistence directory.

<details>
<summary><strong>Full graphstore.json reference</strong> (48 fields)</summary>

```json
{
  "core": {
    "ceiling_mb": 256,
    "initial_capacity": 1024,
    "compact_threshold": 0.2,
    "string_gc_threshold": 3.0,
    "eviction_target_ratio": 0.8,
    "protected_kinds": ["schema", "config", "system"]
  },
  "vector": {
    "embedder": "default",
    "similarity_threshold": 0.85,
    "duplicate_threshold": 0.95,
    "search_oversample": 5,
    "model2vec_model": "minishlab/M2V_base_output",
    "model_cache_dir": null
  },
  "document": {
    "fts_tokenizer": "porter unicode61",
    "chunk_max_size": 2000,
    "chunk_overlap": 50,
    "summary_max_length": 200,
    "fts_full_text": true,
    "vision_model": "smolvlm2:2.2b",
    "vision_base_url": "http://localhost:11434/v1",
    "vision_max_tokens": 300
  },
  "dsl": {
    "cost_threshold": 100000,
    "plan_cache_size": 256,
    "auto_optimize": false,
    "optimize_interval": 500,
    "recall_decay": 0.7,
    "remember_weights": [0.30, 0.20, 0.15, 0.20, 0.15],
    "cache_gc_threshold": 200
  },
  "vault": {
    "enabled": false,
    "path": null,
    "auto_sync": true
  },
  "persistence": {
    "wal_hard_limit": 100000,
    "auto_checkpoint_threshold": 50000,
    "log_retention_days": 7,
    "busy_timeout_ms": 5000
  },
  "retention": {
    "blob_warm_days": 30,
    "blob_archive_days": 90,
    "blob_delete_days": 365
  },
  "server": {
    "cors_origins": ["*"],
    "ingest_root": null,
    "auth_token": null,
    "rate_limit_rpm": 120,
    "rate_limit_window": 60,
    "max_query_length": 10000,
    "max_batch_size": 1000
  },
  "evolution": {
    "similarity_buffer_size": 100,
    "max_rules": 50,
    "min_cooldown": 10,
    "history_retention": 1000
  }
}
```

</details>

---

## 🎮 Playground (experimental)

A local web UI for exploring your graph interactively - CodeMirror editor, React Flow visualization, stacked query results.

```bash
pip install "graphstore[playground]"
graphstore playground
```

The playground currently supports the core DSL (reads, writes, traversals, semantic search). Newer features (evolution rules, cron, vault, voice, document ingest) are not yet surfaced in the UI - use the Python API for those.

---

## 🏗️ Architecture

<details>
<summary>Project structure</summary>

```
graphstore/
├── __init__.py               # Public API re-exports
├── graphstore.py             # GraphStore facade
├── wal.py                    # WAL: append, replay, checkpoint, query log
├── cron.py                   # CronScheduler: persistent jobs, daemon timer
├── config.py                 # Typed config via msgspec Structs
├── evolve.py                 # EvolutionEngine: WHEN/THEN signal-driven rules
├── evolve_defaults.py        # Starter rules (disabled by default)
├── core/                     # Graph engine
│   ├── store.py              # CoreStore: columnar node CRUD
│   ├── columns.py            # ColumnStore: typed numpy arrays
│   ├── edges.py              # EdgeMatrices: scipy CSR
│   ├── strings.py            # StringTable: string interning
│   ├── schema.py             # SchemaRegistry + EMBED field
│   ├── path.py               # BFS, Dijkstra, common neighbors
│   ├── memory.py             # Memory measurement + ceiling
│   ├── optimizer.py          # Compact, GC, evict
│   ├── scheduler.py          # OptimizerScheduler: health + auto-optimize
│   ├── queue.py              # CommandQueue: thread-safe priority queue
│   ├── runtime.py            # RuntimeState: shared component refs
│   ├── types.py              # Result, Edge dataclasses
│   └── errors.py             # Error hierarchy
├── dsl/                      # Query language
│   ├── grammar.lark          # Lark LALR(1) grammar
│   ├── parser.py             # Parser + LRU cache
│   ├── transformer.py        # Parse tree → AST
│   ├── ast_nodes.py          # 70+ AST node types
│   ├── tagger.py             # Auto-tag AST for the log layer (read/write/intelligence/...)
│   ├── visibility.py         # VisibilityMixin: tombstone / TTL / retracted / context filter
│   ├── filtering.py          # FilteringMixin: eval_where + numpy column acceleration
│   ├── executor_base.py      # Shared init, properties, and execute() entry point
│   ├── handlers/             # Auto-dispatch handler mixins
│   │   ├── nodes.py          # NODE, NODES, COUNT
│   │   ├── edges.py          # EDGES, CREATE/DELETE EDGE
│   │   ├── traversal.py      # TRAVERSE, PATH, ANCESTORS, DESCENDANTS
│   │   ├── pattern.py        # MATCH
│   │   ├── aggregation.py    # AGGREGATE GROUP BY
│   │   ├── intelligence.py   # RECALL, SIMILAR, LEXICAL, REMEMBER
│   │   ├── beliefs.py        # ASSERT, RETRACT, PROPAGATE
│   │   ├── mutations.py      # CREATE/UPDATE/DELETE/MERGE/BATCH
│   │   ├── context.py        # BIND/DISCARD CONTEXT
│   │   └── ingest.py         # INGEST, CONNECT NODE
│   ├── executor.py           # Unified dispatcher
│   ├── executor_system.py    # SYS + CRON + LOG commands
│   └── cost_estimator.py     # Frontier-based cost rejection
├── embedding/                # Text → vectors
│   ├── base.py               # Embedder protocol
│   ├── model2vec_embedder.py # Default (30MB, CPU-only)
│   ├── onnx_hf_embedder.py   # EmbeddingGemma ONNX (opt-in)
│   └── postprocess.py        # L2 normalize, Matryoshka truncation
├── vector/                   # Semantic search
│   ├── index.py              # usearch HNSW wrapper
│   └── store.py              # VectorStore: slot ↔ vector mapping
├── document/                 # Document storage
│   └── store.py              # DocumentStore: SQLite FTS5
├── ingest/                   # File → graph pipeline
│   ├── base.py               # Ingestor protocol + Chunk dataclass
│   ├── registry.py           # Per-extension routing
│   ├── markitdown_ingestor.py  # Tier 1: general files
│   ├── pymupdf4llm_ingestor.py # Tier 2: PDF structure
│   ├── docling_ingestor.py   # Tier 3: hard PDFs (lazy load)
│   ├── chunker.py            # HeadingChunker
│   ├── vision.py             # SmolVLM2/Qwen3-VL via Ollama
│   ├── router.py             # Tiered routing
│   └── connector.py          # SYS CONNECT cross-doc wiring
├── voice/                    # Speech I/O (opt-in)
│   ├── protocol.py           # STTProtocol, TTSProtocol
│   ├── stt.py                # MoonshineSTT
│   └── tts.py                # PiperTTS
├── vault/                    # Markdown notes
│   ├── parser.py             # Frontmatter, sections, wikilinks
│   ├── manager.py            # Note CRUD
│   ├── sync.py               # Vault dir → graphstore sync
│   └── executor.py           # VAULT DSL handler
├── registry/                 # Model management
│   ├── models.py             # Supported models config
│   ├── installer.py          # Download + verify + smoke test
│   └── manifest.py           # Model manifest schema
├── persistence/              # SQLite checkpoints
│   ├── database.py
│   ├── serializer.py
│   └── deserializer.py
├── server.py                 # FastAPI playground server
└── cli.py                    # CLI entry point
```

</details>

---

## 🛠️ Development

```bash
git clone https://github.com/orkait/graphstore.git
cd graphstore
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

---

## 📄 License

MIT
