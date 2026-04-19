<div align="center">

# graphstore

**Memory infrastructure for AI agents**

[![CI](https://github.com/orkait/graphstore/actions/workflows/ci.yml/badge.svg)](https://github.com/orkait/graphstore/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/graphstore?color=f59e0b&logo=pypi&logoColor=white)](https://pypi.org/project/graphstore/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/graphstore?color=f59e0b&logo=pypi&logoColor=white)](https://pypi.org/project/graphstore/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-ea580c?logo=gnu&logoColor=white)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-graphstore--docs.orkait.com-f59e0b?logo=readthedocs&logoColor=white)](https://graphstore-docs.orkait.com)

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

g.execute('CREATE NODE "mem:paris" kind = "memory" '
          'DOCUMENT "Paris is the capital of France, famous for the Eiffel Tower."')
g.execute('CREATE NODE "mem:rome" kind = "memory" '
          'DOCUMENT "Rome is the capital of Italy, home to the Colosseum."')
g.execute('CREATE EDGE "mem:paris" -> "mem:rome" kind = "both_european_capitals"')

g.execute('REMEMBER "European history" LIMIT 5')          # hybrid fusion
g.execute('RECALL FROM "mem:paris" DEPTH 2 LIMIT 10')     # graph walk
g.execute('LEXICAL SEARCH "Eiffel Tower" LIMIT 5')        # BM25
g.execute('SIMILAR TO "capital city" LIMIT 5')            # cosine only

g.close()
```

Core install covers REMEMBER / RECALL / LEXICAL / SIMILAR / SYS CRON / VAULT SYNC. Extras for PDF, image, audio, GPU, playground UI are opt-in. See [Installation](https://graphstore-docs.orkait.com/installation).

## 🐍 Typed query builder

Every DSL verb is a typed Python function. Escape-safe, autocomplete-friendly, composable.

```python
from graphstore import q, F, Time

q.create_node("mem:paris", kind="memory",
              document="Paris is the capital of France.").execute(g)

q.nodes(
    where=F.eq("kind", "memory")
          & F.gt("importance", 0.5)
          & F.gte("__event_at__", Time.now_minus(7, "d")),
    limit=10,
).execute(g)

q.batch(
    q.var("x", q.create_node("n1", kind="memory", document="a")),
    q.var("y", q.create_node("n2", kind="memory", document="b")),
    q.create_edge("$x", "$y", kind="next"),
).execute(g)
```

87 typed verbs. 100% DSL coverage. 100% line coverage on the builder. Injection-proof via a single escape helper. Parser-roundtrip-verified. Full reference: [Query builder](https://graphstore-docs.orkait.com/query-builder).

## ⚡ Performance snapshot

LongMemEval-S, 500 records, Jina v5 Small 1024d, Kaggle T4 (2026-04-19): **97.0% retrieval accuracy**, query p50 46 ms. Full methodology + LoCoMo + micro-latency: [benchmarks](https://graphstore-docs.orkait.com/benchmarks/overview).

## 🛠️ Development

```bash
git clone https://github.com/orkait/graphstore.git
cd graphstore
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ingest,vision,embedders-extra,playground]"
pytest     # ~17s on 8-core CPU with -n 4
```

Docs site lives under `website/` (Docusaurus, deployed to Cloudflare Pages). Run locally:

```bash
cd website && bun install && bun run start
```

## 📄 License

AGPL-3.0 - see [LICENSE](LICENSE).
