<div align="center">

# graphstore

**Memory infrastructure for AI agents**

[![CI](https://github.com/orkait/graphstore/actions/workflows/ci.yml/badge.svg)](https://github.com/orkait/graphstore/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/graphstore?color=f59e0b&logo=pypi&logoColor=white)](https://pypi.org/project/graphstore/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/graphstore?color=f59e0b&logo=pypi&logoColor=white)](https://pypi.org/project/graphstore/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-ea580c?logo=gnu&logoColor=white)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-graphstore--docs.orkait.com-f59e0b?logo=readthedocs&logoColor=white)](https://graphstore-docs.orkait.com)
[![SQLite](https://img.shields.io/badge/storage-SQLite-003B57?logo=sqlite&logoColor=white)](https://sqlite.org)
[![usearch](https://img.shields.io/badge/vector-HNSW%20%2F%20usearch-FF6B35?logoColor=white)](https://github.com/unum-cloud/usearch)

</div>

---

graphstore gives AI agents persistent, queryable memory. Store nodes and edges with a typed DSL; retrieve by meaning, by association, by text, or any combination - one call. Runs in-process, persists to SQLite. No server, no infrastructure.

Full docs: **[graphstore-docs.orkait.com](https://graphstore-docs.orkait.com)**

## ⚡ 60-second start

```bash
pip install graphstore
```

```python
from graphstore import GraphStore

g = GraphStore(path="./brain")

# Store: DOCUMENT populates vector + BM25 + blob in one shot
g.execute('CREATE NODE "mem:paris" kind = "memory" '
          'DOCUMENT "Paris is the capital of France, famous for the Eiffel Tower."')
g.execute('CREATE NODE "mem:rome" kind = "memory" '
          'DOCUMENT "Rome is the capital of Italy, home to the Colosseum."')
g.execute('CREATE EDGE "mem:paris" -> "mem:rome" kind = "both_european_capitals"')

# Retrieve - four primitives, all backed by the three storage engines
g.execute('REMEMBER "European history" LIMIT 5')          # hybrid fusion
g.execute('RECALL FROM "mem:paris" DEPTH 2 LIMIT 10')     # graph walk
g.execute('LEXICAL SEARCH "Eiffel Tower" LIMIT 5')        # BM25
g.execute('SIMILAR TO "capital city" LIMIT 5')            # cosine only

g.close()
```

Core install covers REMEMBER / RECALL / LEXICAL / SIMILAR / SYS CRON / VAULT SYNC. Extras for PDF, image, audio, GPU, playground UI are opt-in. See [Installation](#-installation) or [full docs](https://graphstore-docs.orkait.com/installation).

---

## Why graphstore?

Most agent memory systems are wrappers around a vector database. That works for simple retrieval but breaks down when you need:

- **Multi-signal retrieval** - vector similarity alone misses keyword matches. BM25 alone misses semantic matches. You need both, plus graph structure, plus recency, fused intelligently.
- **Graph-native operations** - spreading activation, subgraph extraction, path queries, counterfactual reasoning. These aren't afterthoughts, they're first-class DSL commands.
- **Temporal awareness** - knowing WHEN something happened matters as much as WHAT happened. `__event_at__` is a reserved column, not a hack.
- **Belief tracking** - agents deal with uncertain, contradictory facts. ASSERT with confidence, RETRACT when wrong, find CONTRADICTIONS automatically.
- **Zero infrastructure** - everything is SQLite + numpy + usearch. No Docker, no server, no cloud dependency.

---

## 🏗️ How it works

Three storage engines, one typed DSL, a tiered ingest pipeline, and a hybrid retrieval engine that fuses all of them.

<p align="center">
  <img src="website/static/img/architecture.svg" alt="graphstore architecture: DSL + three storage engines + ingest pipeline + retrieval" width="780">
</p>

**What flows where:**

- The **DSL** (Lark LALR(1), ~70 verbs) is the only way in. Every `CREATE`, `UPDATE`, `DELETE`, `ASSERT`, `RETRACT`, `INGEST`, `SYS *` goes through it.
- **Direct writes** land straight in the three engines:
  - **Graph** - typed numpy columns + scipy CSR edges. Reserved columns like `__event_at__`, `__confidence__`, `__retracted__` live here.
  - **Vector** - usearch HNSW with cosine. Auto-populated via schema `EMBED content` or the `DOCUMENT "..."` clause.
  - **Document** - SQLite + FTS5 virtual table. BM25 + blob storage + single-owner path lock.
- **`INGEST "file.ext"`** is itself a DSL verb. Tiered and modality-aware: `txt/md` (direct), `html/docx/xlsx` (markitdown), `pdf` (pymupdf4llm / docling), `png/jpg` (vision sidecar), `wav/mp3/flac/m4a` (whisper). Output flows into the same three engines.
- **Retrieval** (REMEMBER / RECALL / SIMILAR TO / LEXICAL SEARCH / TRAVERSE) reads from all three engines and fuses the signals.

Deep dive: [Architecture](https://graphstore-docs.orkait.com/concepts/architecture) | [Edge matrix internals](https://graphstore-docs.orkait.com/concepts/edge-matrix).

### REMEMBER, the retrieval engine

`REMEMBER` is the core command. Five-stage pipeline, four weighted signals, optional rerank + nucleus walk.

<p align="center">
  <img src="website/static/img/remember.svg" alt="REMEMBER 5-stage retrieval pipeline: gather -> fuse -> temporal -> rerank -> nucleus" width="620">
</p>

**Signals fused at stage 2** (defaults; weights configurable):

| Signal | Weight | Source |
|---|---|---|
| `vec_signal` | 0.52 | max sentence cosine over usearch ANN |
| `bm25_signal` | 0.25 | SQLite FTS5 over `doc_fts` |
| `recency` | 0.15 | `exp(-age / half_life)` from `__event_at__` or `__updated_at__` |
| `graph_signal` | 0.08 | entity-degree sum over mentioned entities (opt-in) |
| + co-occurrence | bonus | `min(vec, bm25) * 0.10` when a candidate is found by both |
| + recall-frequency | nudge | `log1p(recall_count) * 0.05` |

Every weight is configurable via `graphstore.json`, `GRAPHSTORE_DSL_*` env vars, or constructor kwargs. Full pipeline walkthrough: [REMEMBER docs](https://graphstore-docs.orkait.com/concepts/remember-pipeline).

---

## 🐍 Typed query builder

Every DSL verb is a typed Python function. Escape-safe, autocomplete-friendly, composable.

```python
from graphstore import q, F, Time

# The same three queries as above, via the builder
q.create_node("mem:paris", kind="memory",
              document="Paris is the capital of France.").execute(g)
q.remember("European history", limit=5).execute(g)
q.nodes(where=F.eq("kind", "memory") & F.gt("importance", 0.5), limit=10).execute(g)

# Predicate algebra (Django-Q style, operators &, |, ~)
recent = F.gte("__event_at__", Time.now_minus(7, "d"))
q.nodes(where=F.eq("kind", "memory") & recent & ~F.eq("__retracted__", True))

# Batch compose with variable assignment
q.batch(
    q.var("x", q.create_node("n1", kind="memory", document="a")),
    q.var("y", q.create_node("n2", kind="memory", document="b")),
    q.create_edge("$x", "$y", kind="next"),
).execute(g)
```

**100% DSL coverage** (87 typed verbs + 4 typed sub-DSLs) · **100% line coverage** on the builder (1880 / 1880) · **injection-proof** (every user string through a single `dsl_literal` helper) · **immutable** (modifiers return new Query, never mutate) · **parser-roundtrip-verified**.

Full reference: [Query builder docs](https://graphstore-docs.orkait.com/query-builder).

---

## 🧠 What you can do

### Store and recall

```sql
-- Store a retrievable memory: DOCUMENT populates vector + BM25 + blob in one shot
CREATE NODE "mem:123" kind = "memory" topic = "finance"
  DOCUMENT "Q3 revenue beat expectations driven by enterprise renewals."

-- Hybrid retrieval (5-signal fusion)
REMEMBER "quarterly revenue trends" TOKENS 4000

-- Graph traversal (spreading activation)
RECALL FROM "concept:finance" DEPTH 3 LIMIT 10

-- Keyword search (BM25 over FTS5)
LEXICAL SEARCH "Q3 revenue" LIMIT 10

-- Vector similarity (pure cosine, no fusion)
SIMILAR TO "budget forecasting" LIMIT 10

-- Temporal retrieval (recency-weighted + hard filter)
REMEMBER "what happened in May" AT "2024-05" LIMIT 10
```

### Beliefs, time, consolidation

```sql
ASSERT "fact:earth-radius" value = 6371 kind = "fact" CONFIDENCE 0.99 SOURCE "physics-tool"
RETRACT "fact:old-preference" REASON "user corrected this"
SYS CONTRADICTIONS WHERE kind = "belief" FIELD value GROUP BY topic

-- When-it-happened, not just when-ingested
CREATE NODE "event:trip" kind = "event" content = "visited Paris" EVENT_AT "2024-03-15"
REMEMBER "trip plans" AT "2024-03" LIMIT 10

-- Cluster episodic memories, no LLM needed
SYS CONSOLIDATE THRESHOLD 0.7
```

### Ingest documents

```sql
INGEST "report.pdf" AS "doc:q3" KIND "report"
INGEST "chart.png" USING VISION "smolvlm2-2.2b"
INGEST "interview.mp3"                                -- needs [audio]
SYS CONNECT    -- auto-wire similar chunks across documents
```

<details>
<summary><strong>More: TTL, snapshots, cron, evolution, vault, contexts, graph walks</strong></summary>

```sql
-- TTL + hard delete
CREATE NODE "scratch:temp" kind = "working" data = "..." EXPIRES IN 30m
SYS EXPIRE WHERE kind = "working"
FORGET NODE "mem:old"

-- Snapshot reasoning branches
SYS SNAPSHOT "before-hypothesis"
SYS ROLLBACK TO "before-hypothesis"

-- Scheduled maintenance (needs queued=True)
SYS CRON ADD "expire-ttl" SCHEDULE "@hourly" QUERY "SYS EXPIRE"

-- Self-tuning rules
SYS EVOLVE RULE "reindex-on-drift"
  WHEN recall_hit_rate <= 0.4
  THEN RUN SYS REEMBED
  COOLDOWN 86400

-- Markdown vault (python: GraphStore(path="./brain", vault="./notes"))
VAULT NEW "Project Requirements" KIND "context"
VAULT SEARCH "deployment requirements" LIMIT 5

-- Context isolation
BIND CONTEXT "reasoning-session-42"
CREATE NODE "hyp:1" kind = "hypothesis" content = "maybe X"
DISCARD CONTEXT "reasoning-session-42"

-- Graph walks
TRAVERSE FROM "id" DEPTH 3
PATH FROM "a" TO "b" MAX_DEPTH 5
ANCESTORS OF "id" DEPTH 3
COMMON NEIGHBORS OF "a" AND "b"
AGGREGATE NODES WHERE kind = "memory" GROUP BY topic SELECT COUNT(), AVG(importance)
```

</details>

Full DSL reference (every verb, every clause): [DSL reference](https://graphstore-docs.orkait.com/dsl/reference).

---

## ⚡ Performance

### Micro-latency (single operation)

Median over 30 iters, model2vec 256d, 16-core CPU @ 2-thread BLAS cap (graphstore's default). Reproduce: `python benchmarks/micro_latency.py`. Last measured 2026-04-19.

| Operation | In-memory | On-disk | Notes |
|---|---|---|---|
| Point lookup `NODE "id"` | **5 us** | 11 us | hash to slot |
| Filtered scan `NODES WHERE ... LIMIT 10` | **14 us** | 51 us | typed column filter |
| Semantic search `SIMILAR TO "..." LIMIT 10` | **87 us** | 175 us | usearch HNSW ANN |
| Graph traversal `RECALL DEPTH 3` | ~1 ms | ~1 ms | spreading activation |
| Hybrid retrieval `REMEMBER LIMIT 10` | ~6 ms | ~50 ms | 4-signal fusion |
| `ASSERT` | 11 us | 4 ms | disk pays WAL sync per call |
| Memory per node | ~1.6 KB | ~1.6 KB | ~80 B typed columns + ~1 KB vector |

### LongMemEval-S (retrieval accuracy)

500 records, Jina v5 Small 1024d, Kaggle T4 GPU, 2026-04-19. Public kernel (full logs, in-browser reproducible): [kaggle.com/code/superkaiii/graphstore-jina-v5-small](https://www.kaggle.com/code/superkaiii/graphstore-jina-v5-small).

| Category | n | Retrieval accuracy |
|---|---|---|
| knowledge-update | 78 | 100.0% |
| single-session-assistant | 56 | 100.0% |
| single-session-user | 70 | 98.6% |
| multi-session | 133 | 98.5% |
| temporal-reasoning | 133 | 94.7% |
| single-session-preference | 30 | 83.3% |
| **Overall** | **500** | **97.0%** |

Query p50 46 ms / p95 76 ms, ingest p50 1035 ms / p95 1070 ms. Memory delta +283 MB across 23,867 ingest ops. 5 h 20 m wall, no LLM judge, zero API calls.

### LoCoMo (end-to-end F1)

50Q random sample, MiniMax M2.7 reader, Jina v5 Small 1024d:

| Category | F1 |
|---|---|
| open-domain | 0.452 |
| multi-hop | 0.418 |
| adversarial | 0.500 |
| single-hop | 0.224 |
| temporal | 0.189 |
| **Overall** | **0.357** |

For context: GPT-3.5-turbo with full conversation context scores 0.378 on LoCoMo. graphstore hits comparable quality using only retrieved passages (no full context), with a smaller reader LLM.

Retrieval recall at K (no LLM): top-5 60%, top-10 80%, top-20 84%, top-50 96%.

Full methodology + BEAM + comparison band (Mem0, MemGPT, Zep): [benchmark docs](https://graphstore-docs.orkait.com/benchmarks/overview).

---

## 📦 Installation

```bash
pip install graphstore                        # core (always enough to start)
pip install 'graphstore[ingest]'              # PDF / DOCX / HTML parsing
pip install 'graphstore[vision]'              # local VLM sidecar for scanned PDFs + images
pip install 'graphstore[audio]'               # faster-whisper speech-to-text
pip install 'graphstore[playground]'          # FastAPI web UI
pip install 'graphstore[gpu]'                 # onnxruntime-gpu (Linux x86_64, CUDA 12)
pip install 'graphstore[ingest,vision,playground]'   # everything heavy
```

Core covers the agentic DB contract out of the box: REMEMBER / RECALL (model2vec embedder), SYS CRON (croniter), VAULT SYNC (pyyaml), plus the numpy / scipy / usearch / lark / msgspec / psutil / threadpoolctl foundation. No torch, no PDF parser, no HTTP server.

Full extras table + config layering + single-owner lock semantics: [Installation](https://graphstore-docs.orkait.com/installation) and [Configuration](https://graphstore-docs.orkait.com/configuration).

> **Heads up.** `CREATE NODE "id" kind = "X" topic = "..."` without a `DOCUMENT` clause stores typed columns only. REMEMBER and LEXICAL return zero for that node. Use `DOCUMENT "text"` whenever the node's *content* is what you want to retrieve on.

Everything persists to `./brain/` as SQLite. Reopen with the same path and all memories are back.

---

## 🛠️ Development

```bash
git clone https://github.com/orkait/graphstore.git
cd graphstore
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ingest,vision,embedders-extra,playground]"
pytest     # ~17s on 8-core CPU with -n 4
```

Docs site lives under `website/` (Docusaurus, Cloudflare Pages). Run locally:

```bash
cd website && bun install && bun run start
```

---

## 📄 License

AGPL-3.0, see [LICENSE](LICENSE).
