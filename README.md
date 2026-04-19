<div align="center">

# graphstore

**A memory database for AI agents**

[![CI](https://github.com/orkait/graphstore/actions/workflows/ci.yml/badge.svg)](https://github.com/orkait/graphstore/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/graphstore?color=f59e0b&logo=pypi&logoColor=white)](https://pypi.org/project/graphstore/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-ea580c?logo=gnu&logoColor=white)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-graphstore--docs.orkait.com-f59e0b?logo=readthedocs&logoColor=white)](https://graphstore-docs.orkait.com)

</div>

---

An embedded memory database for AI agents. Facts get written with confidence scores, expire, get contradicted, decay by recency. Retrieval fuses vector similarity, BM25, graph structure, and recency in one call. Everything goes through a typed DSL. Runs in-process, persists to SQLite.

Status: v0.3.0, alpha.

## Install

```bash
pip install graphstore
```

Core ships with [model2vec](https://github.com/MinishLab/model2vec) as the default embedder. Swap for Jina v5, bge-*, EmbeddingGemma, or any ONNX / GGUF model via `graphstore install-embedder`. PDFs, images, audio, GPU, and the web UI are opt-in extras.

```bash
pip install 'graphstore[ingest]'       # PDF / DOCX / HTML
pip install 'graphstore[vision]'       # local VLM for images + scanned PDFs
pip install 'graphstore[audio]'        # faster-whisper speech-to-text
pip install 'graphstore[playground]'   # FastAPI web UI
pip install 'graphstore[gpu]'          # onnxruntime-gpu, Linux x86_64, CUDA 12
```

Full extras matrix: [Installation](https://graphstore-docs.orkait.com/installation).

## Quickstart

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
g.execute('SIMILAR TO "capital city" LIMIT 5')            # vector only
```

`DOCUMENT "text"` populates the vector index, FTS5 index, and blob storage in one shot. Without it, a node is structured data only.

## Architecture

<p align="center">
  <img src="website/static/img/architecture.svg" alt="graphstore architecture: DSL + three storage engines + ingest pipeline + retrieval" width="760">
</p>

Three engines behind one DSL.

- **Graph**: columnar numpy arrays + scipy CSR edge matrices. Reserved columns `__event_at__`, `__confidence__`, `__retracted__`, `__source__` are first-class.
- **Vector**: usearch HNSW, cosine. Auto-embedding on `DOCUMENT` or `EMBED content` schemas.
- **Document**: SQLite + FTS5 for BM25 and blobs. Single-owner advisory lock on the path.

The **DSL** is Lark LALR(1). Every write, read, `INGEST`, and `SYS *` goes through it.

Deep dive: [Architecture](https://graphstore-docs.orkait.com/concepts/architecture) · [Edge matrix](https://graphstore-docs.orkait.com/concepts/edge-matrix).

## REMEMBER

`REMEMBER` fuses four signals at retrieval time. `SIMILAR`, `LEXICAL`, `RECALL` each expose a single leg.

<p align="center">
  <img src="website/static/img/remember.svg" alt="REMEMBER 5-stage retrieval pipeline" width="620">
</p>

| Signal | Default weight | Source |
|---|---|---|
| `vec_signal` | 0.52 | max sentence cosine over usearch ANN |
| `bm25_signal` | 0.25 | SQLite FTS5 over `doc_fts` |
| `recency` | 0.15 | `exp(-age / half_life)` from `__event_at__` |
| `graph_signal` | 0.08 | sum of entity degrees |

Weights are configurable via `graphstore.json`, `GRAPHSTORE_DSL_*` env vars, or constructor kwargs.

Every result returns per-signal scores:

```python
r = g.execute('REMEMBER "Caroline counseling" LIMIT 1 WHERE kind = "message"')
n = r.data[0]
print(n["_remember_score"], n["_vector_sim"], n["_bm25_score"],
      n["_recency_score"], n["_graph_score"])
```

Deep dive: [REMEMBER pipeline](https://graphstore-docs.orkait.com/concepts/remember-pipeline).

## Typed query builder

Every DSL verb has a typed function. Same grammar, IDE autocomplete, injection-safe.

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

Full reference: [Query builder](https://graphstore-docs.orkait.com/query-builder).

## Benchmarks

**LongMemEval-S**, 500 records, Jina v5 Small 1024d, Kaggle T4 GPU, 2026-04-19. Public kernel: [kaggle.com/code/superkaiii/graphstore-jina-v5-small](https://www.kaggle.com/code/superkaiii/graphstore-jina-v5-small).

| Overall | knowledge-update | single-session-assistant | single-session-user | multi-session | temporal | preference |
|---|---|---|---|---|---|---|
| **97.0%** | 100.0% | 100.0% | 98.6% | 98.5% | 94.7% | 83.3% |

Query p50 46 ms / p95 76 ms. Retrieval-only, no LLM judge.

**LoCoMo**, 50Q sample, MiniMax M2.7 reader. Overall F1 **0.357**. Retrieval recall: top-10 80%, top-50 96%.

Full methodology: [Benchmarks](https://graphstore-docs.orkait.com/benchmarks/overview).

## Scope

- Embedded, one writer per path. For multi-tenant, wrap in your own service.
- No SQL, no Cypher, no distributed cluster. Graph ops exist because agent memory is a graph.
- Fusion weights are hand-tuned. Reranking is opt-in, off by default.

## Development

```bash
git clone https://github.com/orkait/graphstore.git
cd graphstore
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ingest,vision,embedders-extra,playground]"
pytest
```

Docs site under `website/` (Docusaurus). Run locally:

```bash
cd website && bun install && bun run start
```

## License

AGPL-3.0. See [LICENSE](LICENSE).
