<div align="center">

# graphstore

**Agentic memory for AI agents. Not a database.**

[![CI](https://github.com/orkait/graphstore/actions/workflows/ci.yml/badge.svg)](https://github.com/orkait/graphstore/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/graphstore?color=f59e0b&logo=pypi&logoColor=white)](https://pypi.org/project/graphstore/)
[![Downloads](https://img.shields.io/pypi/dm/graphstore?color=f59e0b&logo=pypi&logoColor=white)](https://pypi.org/project/graphstore/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-ea580c?logo=gnu&logoColor=white)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-graphstore--docs.orkait.com-f59e0b?logo=readthedocs&logoColor=white)](https://graphstore-docs.orkait.com)

</div>

---

```python
from graphstore import GraphStore

g = GraphStore(path="./brain")

g.execute('CREATE NODE "mem:paris" kind = "memory" '
          'DOCUMENT "Paris is the capital of France, famous for the Eiffel Tower."')
g.execute('CREATE NODE "mem:rome" kind = "memory" '
          'DOCUMENT "Rome is the capital of Italy, home to the Colosseum."')
g.execute('CREATE EDGE "mem:paris" -> "mem:rome" kind = "both_european_capitals"')

g.execute('REMEMBER "European history" LIMIT 5')          # hybrid fusion
g.execute('RECALL FROM "mem:paris" DEPTH 2 LIMIT 10')     # graph walk
g.execute('LEXICAL SEARCH "Eiffel Tower" LIMIT 5')        # BM25
```

`pip install graphstore`. Runs in-process, persists to SQLite. No Docker, no server, no service account.

## It is not a database

graphstore is not a vector DB, not a graph DB, not a relational DB. Call it any of those and you will be disappointed. There is no ACID, no query planner, no SQL. What there is: a memory store that behaves the way an agent actually uses memory. Facts get written with a confidence score. They expire. They get contradicted. They decay by recency. They get retrieved by meaning, by association, by keyword, by structure, or by all four fused together in one call.

The insight is simple. Most agent memory is a thin wrapper over a vector store. That wrapper survives demos and falls apart the moment an agent asks "what happened last May" and cosine similarity cheerfully returns something from 2022 that happens to be semantically close. Or the moment three tools write three contradictory facts about the same entity and nobody notices. Or the moment the agent needs to walk from a conversation to the entity it mentioned to every other conversation that entity appeared in. Vector similarity does not answer that. Neither does keyword search. And neither does a graph on its own.

graphstore pretends none of these primitives exist alone. Vector, BM25, graph, recency, and confidence all live in the same store and get fused at retrieval time. That is the whole thesis. Everything else in this repo is the plumbing.

## 60-second start

```bash
pip install graphstore
```

Core ships with everything you need to start, meaning REMEMBER, RECALL, LEXICAL, SIMILAR, SYS CRON, and VAULT SYNC. The default embedder is [model2vec](https://github.com/MinishLab/model2vec) at 256 dimensions, which is fast enough that you will not notice it. Swap it for Jina v5, bge-*, EmbeddingGemma, or anything else via `graphstore install-embedder`. PDFs, images, audio, GPU, web UI are opt-in extras. See [Installation](#installation).

## How it stores things

Three engines, one DSL, one lock file.

<p align="center">
  <img src="website/static/img/architecture.svg" alt="graphstore architecture: DSL + three storage engines + ingest pipeline + retrieval" width="780">
</p>

The **graph engine** is columnar numpy arrays plus scipy CSR edge matrices. Every field you set becomes a typed column. Every edge you create lands in a sparse matrix that supports row-level lookups in O(degree). Reserved columns (`__event_at__`, `__confidence__`, `__retracted__`, `__source__`) are first-class, not metadata bags.

The **vector engine** is usearch HNSW with cosine distance. `DOCUMENT "text"` or a schema with `EMBED content` triggers auto-embedding. You never call `.embed()` yourself.

The **document engine** is SQLite plus FTS5. BM25 scoring, blob storage, and a single-owner advisory lock live here. If a second process tries to open the same path, it gets `StoreInUse`. WAL replay + compact + checkpoint are not safe across processes, and pretending otherwise is how data gets corrupted.

The **DSL** (Lark LALR(1), around 70 verbs) is the only way in. Every write, every read, every `INGEST`, every `SYS *` goes through the same grammar. There is no alternate escape hatch that bypasses type checks or the confidence model.

Full walkthrough of the architecture: [Architecture](https://graphstore-docs.orkait.com/concepts/architecture). Edge matrix internals (LSM-style dynamic buffer on top of frozen CSR, bidirectional spread matrix, cache invalidation rules): [Edge matrix](https://graphstore-docs.orkait.com/concepts/edge-matrix).

## How it retrieves things: REMEMBER

`REMEMBER` is the one verb that matters. Everything else (`SIMILAR`, `LEXICAL`, `RECALL`) is a leg of the same pipeline exposed on its own when you want to skip the fusion.

<p align="center">
  <img src="website/static/img/remember.svg" alt="REMEMBER 5-stage retrieval pipeline: gather -> fuse -> temporal -> rerank -> nucleus" width="620">
</p>

Five stages. At stage two, these signals combine with configurable weights:

| Signal | Weight | Where it comes from |
|---|---|---|
| `vec_signal` | 0.52 | max sentence cosine over the usearch ANN |
| `bm25_signal` | 0.25 | SQLite FTS5 over `doc_fts` |
| `recency` | 0.15 | `exp(-age / half_life)` from `__event_at__` or `__updated_at__` |
| `graph_signal` | 0.08 | sum of entity degrees for entities mentioned in the candidate |
| co-occurrence bonus | +0.10 | when a candidate is found by both vec and bm25 |
| recall-frequency nudge | +0.05 | `log1p(recall_count)`, tiny boost for frequently-accessed items |

Every weight is configurable through `graphstore.json`, `GRAPHSTORE_DSL_*` env vars, or constructor kwargs. There is no hidden logic tuning itself behind your back. If LongMemEval is giving you 97% and LoCoMo is giving you 0.36 F1, it is because of these numbers, and you can change them.

Every `REMEMBER` result includes the per-signal breakdown so you can see why a candidate ranked where it did:

```python
r = g.execute('REMEMBER "Caroline counseling" LIMIT 1 WHERE kind = "message"')
n = r.data[0]
print(n["_remember_score"], n["_vector_sim"], n["_bm25_score"],
      n["_recency_score"], n["_graph_score"], n["_recall_score"])
```

Deep dive: [REMEMBER pipeline](https://graphstore-docs.orkait.com/concepts/remember-pipeline).

## A real example

Say you are building a support agent that handles one user across many sessions. The user told you two months ago they had a pet cat named Luna. Yesterday they complained their dog is sick. What does the agent actually need from memory?

First, ingest sessions as they happen:

```python
for session in transcripts:
    g.execute(f'CREATE NODE "sess:{session.id}" kind = "session" '
              f'EVENT_AT "{session.date}"')
    for i, msg in enumerate(session.messages):
        g.execute(f'CREATE NODE "msg:{session.id}:{i}" kind = "message" '
                  f'speaker = "{msg.speaker}" '
                  f'EVENT_AT "{session.date}" '
                  f'DOCUMENT "{msg.content}"')
        g.execute(f'CREATE EDGE "sess:{session.id}" -> "msg:{session.id}:{i}" '
                  f'kind = "has_message"')
```

Then ask questions the way an agent would:

```python
# "What pets does this user have?" - vector alone gets "cat", "dog", "Luna" mixed up.
# REMEMBER fuses vector + BM25 + recency and ranks yesterday's dog message above
# the older cat message because recency matters for state.
g.execute('REMEMBER "user pets" LIMIT 5')

# "What did they say in May?" - pure vector will pick semantically-close May-like
# content from any time. REMEMBER with AT adds a hard temporal filter.
g.execute('REMEMBER "recent concerns" AT "2024-05" LIMIT 10')

# "Tell me everything about Luna." - not a search. This is a graph walk from an
# entity you already know exists, through every message that mentions it.
g.execute('RECALL FROM "ent:luna" DEPTH 2 LIMIT 20')
```

And when the agent learns something contradicts an earlier belief:

```python
g.execute('ASSERT "fact:pet" value = "cat" CONFIDENCE 0.9 SOURCE "session-1"')
g.execute('ASSERT "fact:pet" value = "dog" CONFIDENCE 0.95 SOURCE "session-42"')
g.execute('SYS CONTRADICTIONS WHERE kind = "fact" FIELD value')
# returns the pair; your agent decides whether both are true (multiple pets) or
# one supersedes the other.
```

That is the shape of it. No glue code to write, no vector search to bolt onto a SQL database, no FTS5 index to manage, no recency filter to compute in Python.

## Typed query builder

If you prefer typed Python over DSL strings, every verb has a function. Same injection protection, same grammar, IDE autocomplete.

```python
from graphstore import q, F, Time

q.create_node("mem:paris", kind="memory",
              document="Paris is the capital of France.").execute(g)

# Predicate algebra with & | ~
recent = F.gte("__event_at__", Time.now_minus(7, "d"))
q.nodes(where=F.eq("kind", "memory") & recent & ~F.eq("__retracted__", True))

# Batches compose with variable assignment
q.batch(
    q.var("x", q.create_node("n1", kind="memory", document="a")),
    q.var("y", q.create_node("n2", kind="memory", document="b")),
    q.create_edge("$x", "$y", kind="next"),
).execute(g)
```

87 typed verbs, full DSL coverage, 100% line coverage on the builder (1880 of 1880 statements), every user string routed through a single escape helper, every emission round-tripped through the real DSL parser in tests. If the builder emits it, the parser accepts it. Full reference: [Query builder](https://graphstore-docs.orkait.com/query-builder).

## Benchmarks

### LongMemEval-S, retrieval accuracy

500 records, Jina v5 Small 1024d, Kaggle T4 GPU, run on 2026-04-19. The kernel is public, the logs are public, the kernel runs in-browser if you want to reproduce: [kaggle.com/code/superkaiii/graphstore-jina-v5-small](https://www.kaggle.com/code/superkaiii/graphstore-jina-v5-small).

| Category | n | Retrieval accuracy |
|---|---|---|
| knowledge-update | 78 | 100.0% |
| single-session-assistant | 56 | 100.0% |
| single-session-user | 70 | 98.6% |
| multi-session | 133 | 98.5% |
| temporal-reasoning | 133 | 94.7% |
| single-session-preference | 30 | 83.3% |
| **Overall** | **500** | **97.0%** |

Query p50 46 ms, p95 76 ms. Ingest p50 1035 ms, p95 1070 ms. Memory delta +283 MB across 23,867 ingest ops. 5 h 20 m wall time, no LLM judge, zero API calls. 97% on this benchmark is where Mem0 and MemGPT sit with large readers in the loop. graphstore does it retrieval-only, on a T4.

### LoCoMo, end-to-end F1

50Q random sample, MiniMax M2.7 reader, same embedder.

| Category | F1 |
|---|---|
| open-domain | 0.452 |
| multi-hop | 0.418 |
| adversarial | 0.500 |
| single-hop | 0.224 |
| temporal | 0.189 |
| **Overall** | **0.357** |

For context, GPT-3.5-turbo with the entire conversation in context scores 0.378 on LoCoMo. graphstore gets within a point of that using retrieved passages only, with a smaller reader. This is not a flex. It is the number. Single-hop and temporal are where retrieval alone hurts most, and improving them is an active area.

<details>
<summary><strong>Micro-latency (one op at a time)</strong></summary>

Median over 30 iters, model2vec 256d, 16-core CPU at 2-thread BLAS cap (graphstore's default). Reproduce: `python benchmarks/micro_latency.py`.

| Operation | In-memory | On-disk | Notes |
|---|---|---|---|
| Point lookup `NODE "id"` | 5 us | 11 us | hash to slot |
| Filtered scan | 14 us | 51 us | typed column filter |
| Semantic search | 87 us | 175 us | usearch HNSW ANN |
| Graph traversal `RECALL DEPTH 3` | ~1 ms | ~1 ms | spreading activation |
| Hybrid retrieval `REMEMBER LIMIT 10` | ~6 ms | ~50 ms | 4-signal fusion |
| `ASSERT` | 11 us | 4 ms | disk pays WAL sync |
| Memory per node | ~1.6 KB | ~1.6 KB | columns + vector + overhead |

Disk numbers at 100k nodes, in-memory at 10k. WAL sync dominates at small N; ANN tree depth dominates at large N. REMEMBER scales with the number of candidates the ANN and FTS legs return, which for realistic workloads is far under 100.

</details>

Full methodology, BEAM support, and the comparison band against Mem0 / MemGPT / Zep: [Benchmarks](https://graphstore-docs.orkait.com/benchmarks/overview).

## Installation

```bash
pip install graphstore                        # core
pip install 'graphstore[ingest]'              # PDF / DOCX / HTML
pip install 'graphstore[vision]'              # local VLM sidecar for images + scanned PDFs
pip install 'graphstore[audio]'               # faster-whisper speech-to-text
pip install 'graphstore[playground]'          # FastAPI web UI
pip install 'graphstore[gpu]'                 # onnxruntime-gpu, Linux x86_64, CUDA 12
```

Core is 8 deps: numpy, scipy, usearch, lark, msgspec, psutil, threadpoolctl, model2vec. No torch, no PDF parser, no HTTP server until you ask for them. Full extras matrix and install quirks (docling pulling torch, VLM weights on first use, apparmor on Ubuntu 24.04): [Installation docs](https://graphstore-docs.orkait.com/installation).

> Without `DOCUMENT "text"`, a node is structured data only. `REMEMBER` and `LEXICAL` will not see it. Add a `DOCUMENT` clause whenever the node's content is what you want to retrieve on.

## What graphstore is not

It is not a relational DB. There is no SQL, no joins, no foreign keys, no transactions in the ACID sense. Use Postgres or SQLite directly if that is what you need.

It is not a graph DB. There is no Cypher, no openCypher, no gremlin. Graph ops are there because an agent's memory is a graph, not because graphstore wants to compete with Neo4j.

It is not a vector DB. Vector search is a leg of retrieval, not the product. If you only need ANN, use usearch or faiss directly and skip the rest.

It is not a service. It is a library. It runs in your process. One Python process holds the lock; a second `GraphStore(path=...)` against the same path raises `StoreInUse` rather than pretending it is safe. If you want multi-tenant, put it behind your own service layer.

It is not finished. The benchmarks are decent, not state of the art in every category. Single-hop and temporal F1 on LoCoMo have room. Fusion weights are tuned by hand. Reranking is opt-in and off by default. Contributions, especially on retrieval quality, are welcome.

## Development

```bash
git clone https://github.com/orkait/graphstore.git
cd graphstore
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ingest,vision,embedders-extra,playground]"
pytest    # ~17 s on an 8-core CPU with -n 4
```

Docs site lives under `website/` (Docusaurus, deployed to Cloudflare Pages). Run locally:

```bash
cd website && bun install && bun run start
```

## License

AGPL-3.0. See [LICENSE](LICENSE).
