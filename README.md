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

graphstore gives AI agents persistent, queryable memory. Store nodes and edges with a simple DSL, retrieve them by meaning, by association, by text search, or any combination - all from one call.

It is designed for agent frameworks, LLM applications, and research tools that need more than a vector database but less than a full graph database. Everything runs in-process, persists to SQLite. No server, no infrastructure.

---

## Why graphstore?

Most agent memory systems are wrappers around a vector database. That works for simple retrieval but breaks down when you need:

- **Multi-signal retrieval** - vector similarity alone misses keyword matches. BM25 alone misses semantic matches. You need both, plus graph structure, plus recency, fused intelligently.
- **Graph-native operations** - spreading activation, subgraph extraction, path queries, counterfactual reasoning. These aren't afterthoughts - they're first-class DSL commands.
- **Temporal awareness** - knowing WHEN something happened matters as much as WHAT happened. `__event_at__` is a reserved column, not a hack.
- **Belief tracking** - agents deal with uncertain, contradictory facts. ASSERT with confidence, RETRACT when wrong, find CONTRADICTIONS automatically.
- **Zero infrastructure** - everything is SQLite + numpy + usearch. No Docker, no server, no cloud dependency.

---

## 🏗️ How it works

Three storage engines, one typed DSL, and a hybrid retrieval pipeline that fuses all of them.

```text
              ┌─────────────────────────────────────────┐
              │        DSL - Lark LALR(1) grammar       │
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
```

### REMEMBER - the retrieval engine

`REMEMBER` is the core command. It fuses five signals into a single ranked result set:

```
REMEMBER "European architecture" LIMIT 10

  ┌─────────────────────────────────────────────────────────┐
  │  vec_signal     cosine similarity (usearch ANN)         │  0.50
  │  bm25_signal    keyword match (SQLite FTS5)             │  0.20
  │  recency        decay from event time                   │  0.10
  │  graph_degree   how connected is this node              │  0.15
  │  recall_freq    how often retrieved before              │  0.05
  │                                                         │
  │  + co-occurrence boost (found by both vec AND bm25)     │
  │  + HybridRAG graph expansion (spreading activation)     │
  │  + temporal-first filtering (when AT anchor present)    │
  │  + type-weighted scoring (observations > messages)      │
  └─────────────────────────────────────────────────────────┘
```

Weights are configurable. The pipeline is 10 stages - see [summary.md](summary.md) for the full breakdown.

---

## 📦 Installation

```bash
pip install graphstore
```

Lightweight core: numpy, scipy, usearch, lark, msgspec. No torch, no PDF parser, no HTTP server.

```bash
# Embedder (30 MB, CPU-only, zero-config)
pip install 'graphstore[embed-default]'

# PDF / DOCX / HTML ingestion
pip install 'graphstore[ingest]'

# GPU acceleration (Linux x86_64, CUDA 12)
pip install 'graphstore[gpu]'

# Everything
pip install 'graphstore[embed-default,ingest,vault,scheduler]'
```

<details>
<summary><strong>All extras</strong></summary>

| Extra | What it adds |
|---|---|
| `embed-default` | model2vec - zero-config CPU embedder |
| `embed-fastembed` | fastembed - ~30 ONNX encoder models |
| `ingest` | markitdown + pymupdf (PDF/DOCX/HTML -> markdown) |
| `ingest-pro` | docling + openai (heavier PDF + vision via LLM) |
| `scheduler` | croniter (cron-expression parsing) |
| `vault` | pyyaml (markdown vault sync) |
| `playground` | fastapi + uvicorn (local web UI) |
| `gpu` | onnxruntime-gpu + bundled nvidia-cu12 (Linux x86_64) |
| `gpu-ort` | onnxruntime-gpu only (bring your own CUDA 12) |
| `voice` | moonshine STT + piper TTS |
| `dev` | pytest + pytest-benchmark |

</details>

---

## 🚀 Quickstart

```python
from graphstore import GraphStore

g = GraphStore(path="./brain")

# Store memories
g.execute('CREATE NODE "mem:paris" kind = "memory" topic = "travel" importance = 0.9')
g.execute('CREATE NODE "mem:eiffel" kind = "memory" topic = "travel" importance = 0.8')
g.execute('CREATE EDGE "mem:paris" -> "mem:eiffel" kind = "associated"')

# Retrieve by meaning (fuses vector + BM25 + recency + graph)
result = g.execute('REMEMBER "European architecture" TOKENS 4000')

# Retrieve by association (spreading activation through edges)
result = g.execute('RECALL FROM "mem:paris" DEPTH 2 LIMIT 10')

# Retrieve by keywords
result = g.execute('LEXICAL SEARCH "Eiffel Tower" LIMIT 5')

# Retrieve with temporal anchor
result = g.execute('REMEMBER "trip plans" AT "2024-03" LIMIT 10')
```

Everything persists to `./brain/` as SQLite. Reopen with the same path and all memories are back.

---

## 🧠 What you can do

### Store and recall memories

```sql
CREATE NODE "mem:123" kind = "memory" topic = "finance" importance = 0.8

-- Hybrid retrieval (5-signal fusion)
REMEMBER "quarterly revenue trends" TOKENS 4000

-- Graph traversal (spreading activation)
RECALL FROM "concept:finance" DEPTH 3 LIMIT 10

-- Keyword search (BM25)
LEXICAL SEARCH "Q3 revenue" LIMIT 10

-- Vector similarity
SIMILAR TO "budget forecasting" LIMIT 10

-- Temporal retrieval
REMEMBER "what happened in May" AT "2024-05" LIMIT 10
```

### Ingest documents

```sql
INGEST "report.pdf" AS "doc:q3" KIND "report"
SYS CONNECT    -- auto-wire similar chunks across documents
```

Supports PDF, Word, Markdown, text, images (via VLM), audio (via STT). Parses, chunks, embeds, and wires into the graph automatically.

### Track beliefs

```sql
ASSERT "fact:earth-radius" value = 6371 kind = "fact" CONFIDENCE 0.99 SOURCE "physics-tool"
RETRACT "fact:old-preference" REASON "user corrected this"
SYS CONTRADICTIONS WHERE kind = "belief" FIELD value GROUP BY topic
WHAT IF RETRACT "belief:earth-is-flat"
```

### Temporal awareness

```sql
-- Store when events happened (not just when ingested)
CREATE NODE "event:trip" kind = "event" content = "visited Paris" EVENT_AT "2024-03-15"

-- Query with temporal anchoring
REMEMBER "trip plans" AT "2024-03" LIMIT 10

-- Range queries on event time
NODES WHERE __event_at__ > 1710000000000 LIMIT 10
```

### Memory consolidation

```sql
-- Cluster episodic memories into observations (no LLM needed)
SYS CONSOLIDATE THRESHOLD 0.7

-- Observations get higher retrieval priority via type_weights
```

<details>
<summary><strong>More features</strong></summary>

**Memory lifecycle**
```sql
CREATE NODE "scratch:temp" kind = "working" data = "..." EXPIRES IN 30m
SYS EXPIRE WHERE kind = "working"
FORGET NODE "mem:old"    -- hard delete (graph + vector + blob)
```

**Snapshots**
```sql
SYS SNAPSHOT "before-hypothesis"
-- ... reasoning branch ...
SYS ROLLBACK TO "before-hypothesis"
```

**Scheduled maintenance**
```python
g = GraphStore(path="./brain", queued=True)
g.execute('SYS CRON ADD "expire-ttl" SCHEDULE "@hourly" QUERY "SYS EXPIRE"')
g.execute('SYS CRON ADD "nightly-opt" SCHEDULE "0 3 * * *" QUERY "SYS OPTIMIZE"')
```

**Self-healing rules**
```sql
SYS EVOLVE RULE "reindex-on-drift"
  WHEN recall_hit_rate <= 0.4
  THEN RUN SYS REEMBED
  COOLDOWN 86400
```

**Vault (markdown notes)**
```python
g = GraphStore(path="./brain", vault="./notes")
g.execute('VAULT NEW "Project Requirements" KIND "context"')
g.execute('VAULT SEARCH "deployment requirements" LIMIT 5')
```

**Voice (speech I/O)**
```python
g = GraphStore(path="./brain", voice=True)
g.speak("The Q3 revenue grew 15%")
g.listen(on_text=lambda text: agent.process(text))
```

**Context isolation**
```sql
BIND CONTEXT "reasoning-session-42"
CREATE NODE "hyp:1" kind = "hypothesis" content = "maybe X"
DISCARD CONTEXT "reasoning-session-42"
```

**Graph queries**
```sql
TRAVERSE FROM "node_id" DEPTH 3
SUBGRAPH FROM "node_id" DEPTH 2
PATH FROM "a" TO "b" MAX_DEPTH 5
SHORTEST PATH FROM "a" TO "b"
ANCESTORS OF "node_id" DEPTH 3
COMMON NEIGHBORS OF "a" AND "b"
AGGREGATE NODES WHERE kind = "memory" GROUP BY topic SELECT COUNT(), AVG(importance)
```

</details>

---

## ⚡ Performance

All numbers at 100k nodes, mid-range laptop, no GPU.

| Operation | Latency |
|---|---|
| Point lookup (`NODE "id"`) | 4 us |
| Filtered scan (`NODES WHERE ... LIMIT 10`) | 68 us |
| Semantic search (`SIMILAR TO` LIMIT 10) | 127 us |
| Graph traversal (`RECALL` DEPTH 3) | 983 us |
| Hybrid retrieval (`REMEMBER` LIMIT 10) | ~2 ms |
| Assert / retract | 4-9 us |
| Memory per node | 66 bytes (columns) + ~1 KB (vector) |

### Benchmark results

**LongMemEval-S** (500 records, retrieval-only, Jina v5 Nano 768d, Kaggle T4 GPU):

| Category | Accuracy |
|---|---|
| knowledge-update | 100.0% |
| multi-session | 98.5% |
| single-session-assistant | 100.0% |
| single-session-user | 98.6% |
| temporal-reasoning | 91.7% |
| single-session-preference | 86.7% |
| **Overall** | **96.4%** |

**LoCoMo** (50Q, token-level F1, MiniMax M2.7 reader, Jina v5 Small 1024d):

| Category | F1 |
|---|---|
| open-domain | 0.452 |
| multi-hop | 0.418 |
| adversarial | 0.500 |
| single-hop | 0.224 |
| temporal | 0.189 |
| **Overall** | **0.357** |

For context: GPT-3.5-turbo with full conversation context scores 0.378 on LoCoMo. graphstore achieves comparable quality using only retrieved passages (no full context), with a smaller LLM reader.

Retrieval recall (keyword in top-K passages, no LLM):

| K | Recall |
|---|---|
| top-5 | 60% |
| top-10 | 80% |
| top-20 | 84% |
| top-50 | 96% |

Full methodology, reproduction instructions, and comparison details: see [summary.md](summary.md).

---

## ⚙️ Configuration

```python
g = GraphStore(
    path="./brain",
    ceiling_mb=256,
    embedder="default",       # or a custom Embedder instance, or "none"
    queued=True,              # thread-safe + cron scheduler
    vault="./notes",          # markdown vault (None = disabled)
    voice=False,              # STT/TTS (False = disabled)
    fusion_method="weighted", # "weighted" or "rrf"
    recency_half_life_days=7300,  # ~20 years - agent memory decays slowly
)
```

Config is loaded in layers: `config.py` defaults -> `graphstore.json` overrides -> `GRAPHSTORE_*` env vars -> constructor kwargs.

<details>
<summary><strong>graphstore.json reference</strong></summary>

Only include fields you want to override. Missing fields use defaults from `config.py`.

```json
{
  "core": {
    "ceiling_mb": 512
  },
  "vector": {
    "search_oversample": 16,
    "similarity_threshold": 0.85
  },
  "dsl": {
    "fusion_method": "weighted",
    "remember_weights": [0.50, 0.20, 0.10, 0.15, 0.05],
    "recency_half_life_days": 7300,
    "hybridrag_weight": 0.15,
    "nucleus_expansion": true
  }
}
```

Environment variables: `GRAPHSTORE_CORE_CEILING_MB=512`, `GRAPHSTORE_DSL_FUSION_METHOD=weighted`, etc.

</details>

---

## 📖 DSL Reference

<details>
<summary><strong>Reads (25+ commands)</strong></summary>

```sql
NODE "id"
NODE "id" WITH DOCUMENT
NODES WHERE kind = "memory" AND importance > 0.5 LIMIT 10
EDGES FROM "id" WHERE kind = "calls"
TRAVERSE FROM "id" DEPTH 3
SUBGRAPH FROM "id" DEPTH 2
PATH FROM "a" TO "b" MAX_DEPTH 5
SHORTEST PATH FROM "a" TO "b"
ANCESTORS OF "id" DEPTH 3
DESCENDANTS OF "id" DEPTH 3
COMMON NEIGHBORS OF "a" AND "b"
MATCH ("fn_main") -[kind = "calls"]-> (callee)
COUNT NODES WHERE kind = "memory"
AGGREGATE NODES GROUP BY kind SELECT COUNT()
RECALL FROM "id" DEPTH 3 LIMIT 10
SIMILAR TO "text" LIMIT 10
SIMILAR TO NODE "id" LIMIT 10
SIMILAR TO [0.1, 0.2, ...] LIMIT 10
LEXICAL SEARCH "phrase" LIMIT 10
REMEMBER "query" LIMIT 10
REMEMBER "query" AT "2024-03" TOKENS 4000
WHAT IF RETRACT "id"
```

</details>

<details>
<summary><strong>Writes (15+ commands)</strong></summary>

```sql
CREATE NODE "id" kind = "x" name = "foo"
CREATE NODE "id" kind = "x" EVENT_AT "2024-03-15"
CREATE NODE "id" kind = "x" DOCUMENT "full text..." EXPIRES IN 1h
UPDATE NODE "id" SET name = "new"
UPSERT NODE "id" kind = "x" name = "foo"
DELETE NODE "id"
DELETE NODES WHERE kind = "test"
UPDATE NODES WHERE kind = "fact" SET confidence = 0.5
CREATE EDGE "src" -> "tgt" kind = "calls"
INCREMENT NODE "id" hits BY 1
ASSERT "id" kind = "fact" value = 42 CONFIDENCE 0.9 SOURCE "tool" EVENT_AT "2024-01"
RETRACT "id" REASON "outdated"
MERGE NODE "old" INTO "canonical"
PROPAGATE "id" FIELD confidence DEPTH 3
INGEST "file.pdf" AS "doc:q3" KIND "report"
FORGET NODE "id"
BIND CONTEXT "session-1"
DISCARD CONTEXT "session-1"
BEGIN ... COMMIT
```

</details>

<details>
<summary><strong>System (30+ commands)</strong></summary>

```sql
SYS STATUS / SYS STATS / SYS HEALTH
SYS KINDS / SYS EDGE KINDS / SYS DESCRIBE NODE "memory"
SYS REGISTER NODE KIND "memory" REQUIRED topic:string EMBED content
SYS CONNECT / SYS CONNECT THRESHOLD 0.9
SYS CONSOLIDATE THRESHOLD 0.7
SYS DUPLICATES THRESHOLD 0.95
SYS CONTRADICTIONS WHERE kind = "belief" FIELD value GROUP BY topic
SYS EXPIRE WHERE kind = "working"
SYS SNAPSHOT "name" / SYS ROLLBACK TO "name"
SYS EMBEDDERS / SYS REEMBED
SYS RETAIN / SYS EVICT
SYS CHECKPOINT / SYS REBUILD INDICES / SYS CLEAR CACHE
SYS OPTIMIZE / SYS OPTIMIZE COMPACT
SYS LOG LIMIT 20 / SYS LOG TRACE "id"
SYS CRON ADD "name" SCHEDULE "0 * * * *" QUERY "SYS EXPIRE"
SYS EVOLVE RULE "name" WHEN signal OP value THEN action COOLDOWN n
SYS EVOLVE LIST / SHOW / ENABLE / DISABLE / DELETE / HISTORY
```

</details>

---

## 🏗️ Project structure

<details>
<summary>Expand</summary>

```
graphstore/
  graphstore.py           # Main entry point
  config.py               # Typed config (msgspec Structs)
  wal.py                  # Write-ahead log
  cron.py                 # Scheduled jobs
  evolve.py               # Self-tuning rules
  core/                   # Graph engine (numpy + CSR + columns)
  dsl/                    # Query language (grammar + parser + handlers)
    handlers/
      intelligence.py     # REMEMBER, RECALL, SIMILAR, LEXICAL SEARCH
      mutations.py        # CREATE, UPDATE, DELETE, MERGE
      traversal.py        # TRAVERSE, PATH, SUBGRAPH
      beliefs.py          # ASSERT, RETRACT, PROPAGATE
      ...
  algos/                  # Pure algorithms (fusion, spreading, consolidation)
  embedding/              # Embedder backends (model2vec, ONNX, GGUF, fastembed)
  vector/                 # usearch ANN index
  document/               # SQLite FTS5 + blob storage
  ingest/                 # File -> graph pipeline
  retrieval/              # Query-aware retrieval planner
  vault/                  # Markdown note system
  voice/                  # STT + TTS
  registry/               # Model management
  persistence/            # SQLite serialization
  server.py               # Playground web UI
  cli.py                  # CLI
```

</details>

---

## 🛠️ Development

```bash
git clone https://github.com/orkait/graphstore.git
cd graphstore
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest     # 1166 tests
```

---

## 📄 License

MIT
</div>
