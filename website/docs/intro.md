---
slug: /
title: Introduction
sidebar_position: 1
---

# graphstore

Memory infrastructure for AI agents. Nodes and edges with a typed DSL. Retrieve by meaning, association, text, or any mix - one call. Runs in-process, persists to SQLite. No server.

## 60-second start

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

g.execute('REMEMBER "European history" LIMIT 5')
g.execute('RECALL FROM "mem:paris" DEPTH 2 LIMIT 10')
g.execute('LEXICAL SEARCH "Eiffel Tower" LIMIT 5')
g.execute('SIMILAR TO "capital city" LIMIT 5')

g.close()
```

## Typed query builder

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
```

87 typed verbs. 100% DSL coverage. 100% line coverage. Injection-proof via a single escape helper.

## What to read next

Docs are under active migration. For now, the source of truth remains the [README on GitHub](https://github.com/orkait/graphstore#readme) until migration lands in PR 2.
