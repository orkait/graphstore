<div align="center">

# graphstore

**A memory database for AI agents**

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

`pip install graphstore`. Runs in-process, persists to SQLite. Status: v0.3.0, alpha. For agent builders.

## What it is

An embedded database purpose-built for one workload: how an agent reaches for memory. Facts get written with confidence scores, expire, get contradicted, decay by recency. Retrieval fuses vector similarity, BM25, graph structure, and recency in one call. Everything goes through a typed DSL.

Most agent memory is a thin wrapper over a vector store. That works until the agent asks "what happened last May" and cosine similarity returns a 2022 message that happens to be semantically close. graphstore keeps vector, BM25, graph, recency, and confidence in the same store so retrieval can use all of them.

## 60-second start

```bash
pip install graphstore
```

Core ships with the default [model2vec](https://github.com/MinishLab/model2vec) embedder at 256d. Swap for Jina v5, bge-*, EmbeddingGemma, or any ONNX / GGUF model via `graphstore install-embedder`. PDFs, images, audio, GPU, web UI are opt-in extras. See [Installation](#installation).

## Architecture

<p align="center">
  <img src="website/static/img/architecture.svg" alt="graphstore architecture: DSL + three storage engines + ingest pipeline + retrieval" width="780">
</p>

Three engines behind one DSL:

- **Graph**: columnar numpy arrays plus scipy CSR edge matrices. Typed columns per field. Reserved columns `__event_at__`, `__confidence__`, `__retracted__`, `__source__` are first-class.
- **Vector**: usearch HNSW, cosine. `DOCUMENT "text"` or `EMBED content` schema triggers auto-embedding.
- **Document**: SQLite + FTS5 for BM25 and blobs. Single-owner advisory lock on the path.

The **DSL** (Lark LALR(1), 70+ verbs) is the only way in. Every write, read, `INGEST`, and `SYS *` goes through it.

Deep dive: [Architecture](https://graphstore-docs.orkait.com/concepts/architecture), [Edge matrix internals](https://graphstore-docs.orkait.com/concepts/edge-matrix).

## REMEMBER

`REMEMBER` is the hybrid retrieval verb. `SIMILAR`, `LEXICAL`, and `RECALL` each expose a single leg when you want to skip fusion.

<p align="center">
  <img src="website/static/img/remember.svg" alt="REMEMBER 5-stage retrieval pipeline: gather -> fuse -> temporal -> rerank -> nucleus" width="620">
</p>

Default fusion weights:

| Signal | Weight | Source |
|---|---|---|
| `vec_signal` | 0.52 | max sentence cosine over usearch ANN |
| `bm25_signal` | 0.25 | SQLite FTS5 over `doc_fts` |
| `recency` | 0.15 | `exp(-age / half_life)` from `__event_at__` or `__updated_at__` |
| `graph_signal` | 0.08 | sum of entity degrees for entities in the candidate |
| co-occurrence bonus | +0.10 | when a candidate is found by vec and bm25 |
| recall-frequency | +0.05 | `log1p(recall_count)` |

Weights are configurable via `graphstore.json`, `GRAPHSTORE_DSL_*` env vars, or constructor kwargs. Every `REMEMBER` result returns per-signal scores:

```python
r = g.execute('REMEMBER "Caroline counseling" LIMIT 1 WHERE kind = "message"')
n = r.data[0]
print(n["_remember_score"], n["_vector_sim"], n["_bm25_score"],
      n["_recency_score"], n["_graph_score"], n["_recall_score"])
```

Deep dive: [REMEMBER pipeline](https://graphstore-docs.orkait.com/concepts/remember-pipeline).

## A real example

A support agent handling one user across many sessions. User mentioned a cat named Luna two months ago; yesterday they said their dog is sick.

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

Three queries the agent might run:

```python
# Hybrid: vector + BM25 + recency. Yesterday's dog outranks older cat on state queries.
g.execute('REMEMBER "user pets" LIMIT 5')

# Temporal filter beyond recency bias.
g.execute('REMEMBER "recent concerns" AT "2024-05" LIMIT 10')

# Graph walk from a known entity through every message that mentions it.
g.execute('RECALL FROM "ent:luna" DEPTH 2 LIMIT 20')
```

Contradicting facts are first-class:

```python
g.execute('ASSERT "fact:pet" value = "cat" CONFIDENCE 0.9 SOURCE "session-1"')
g.execute('ASSERT "fact:pet" value = "dog" CONFIDENCE 0.95 SOURCE "session-42"')
g.execute('SYS CONTRADICTIONS WHERE kind = "fact" FIELD value')
```

## Typed query builder

Every DSL verb has a typed function. Same grammar, same escape protection, IDE autocomplete.

```python
from graphstore import q, F, Time

q.create_node("mem:paris", kind="memory",
              document="Paris is the capital of France.").execute(g)

recent = F.gte("__event_at__", Time.now_minus(7, "d"))
q.nodes(where=F.eq("kind", "memory") & recent & ~F.eq("__retracted__", True))

q.batch(
    q.var("x", q.create_node("n1", kind="memory", document="a")),
    q.var("y", q.create_node("n2", kind="memory", document="b")),
    q.create_edge("$x", "$y", kind="next"),
).execute(g)
```

87 typed verbs, 100% DSL coverage, 100% line coverage on the builder (1880 / 1880 statements), parser-roundtrip-verified. Every user string routes through one escape helper. Full reference: [Query builder](https://graphstore-docs.orkait.com/query-builder).

## Benchmarks

### LongMemEval-S, retrieval accuracy

500 records, Jina v5 Small 1024d, Kaggle T4 GPU, 2026-04-19. Public kernel with logs, reproducible in-browser: [kaggle.com/code/superkaiii/graphstore-jina-v5-small](https://www.kaggle.com/code/superkaiii/graphstore-jina-v5-small).

| Category | n | Accuracy |
|---|---|---|
| knowledge-update | 78 | 100.0% |
| single-session-assistant | 56 | 100.0% |
| single-session-user | 70 | 98.6% |
| multi-session | 133 | 98.5% |
| temporal-reasoning | 133 | 94.7% |
| single-session-preference | 30 | 83.3% |
| **Overall** | **500** | **97.0%** |

Query p50 46 ms / p95 76 ms. Ingest p50 1035 ms / p95 1070 ms. Memory delta +283 MB over 23,867 ingest ops. Retrieval-only, no LLM judge, zero API calls.

### LoCoMo, token-level F1

50Q random sample, MiniMax M2.7 reader, same embedder.

| Category | F1 |
|---|---|
| open-domain | 0.452 |
| multi-hop | 0.418 |
| adversarial | 0.500 |
| single-hop | 0.224 |
| temporal | 0.189 |
| **Overall** | **0.357** |

Retrieval recall at K (no LLM): top-5 60%, top-10 80%, top-20 84%, top-50 96%.

<details>
<summary><strong>Micro-latency (single operation)</strong></summary>

Median over 30 iters, model2vec 256d, 16-core CPU at 2-thread BLAS cap. Reproduce: `python benchmarks/micro_latency.py`.

| Operation | In-memory | On-disk | Notes |
|---|---|---|---|
| Point lookup `NODE "id"` | 5 us | 11 us | hash to slot |
| Filtered scan | 14 us | 51 us | typed column filter |
| Semantic search | 87 us | 175 us | usearch HNSW ANN |
| `RECALL DEPTH 3` | ~1 ms | ~1 ms | spreading activation |
| `REMEMBER LIMIT 10` | ~6 ms | ~50 ms | 4-signal fusion |
| `ASSERT` | 11 us | 4 ms | disk pays WAL sync |
| Memory per node | ~1.6 KB | ~1.6 KB | columns + vector + overhead |

Disk numbers at 100k nodes, in-memory at 10k.

</details>

Full methodology + BEAM: [Benchmarks](https://graphstore-docs.orkait.com/benchmarks/overview).

## Installation

```bash
pip install graphstore                        # core
pip install 'graphstore[ingest]'              # PDF / DOCX / HTML
pip install 'graphstore[vision]'              # local VLM sidecar for images + scanned PDFs
pip install 'graphstore[audio]'               # faster-whisper speech-to-text
pip install 'graphstore[playground]'          # FastAPI web UI
pip install 'graphstore[gpu]'                 # onnxruntime-gpu, Linux x86_64, CUDA 12
```

Core is 10 deps: numpy, scipy, usearch, lark, msgspec, psutil, threadpoolctl, model2vec, croniter, pyyaml. Full extras matrix: [Installation docs](https://graphstore-docs.orkait.com/installation).

> Without `DOCUMENT "text"`, a node is structured data only. `REMEMBER` and `LEXICAL` will not see it.

## Scope

- Embedded, in-process. One writer per path (second opener hits `StoreInUse`). For multi-tenant, put it behind your own service.
- No SQL, no Cypher, no distributed cluster. Graph ops exist because agent memory is a graph.
- Status: v0.3.0, alpha. Fusion weights are hand-tuned. Reranking is opt-in, off by default.

## Development

```bash
git clone https://github.com/orkait/graphstore.git
cd graphstore
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ingest,vision,embedders-extra,playground]"
pytest    # ~17 s on an 8-core CPU with -n 4
```

Docs site under `website/` (Docusaurus, Cloudflare Pages). Run locally:

```bash
cd website && bun install && bun run start
```

## License

AGPL-3.0. See [LICENSE](LICENSE).
