# How to Run the Caroline/LoCoMo Example

This walks through populating GraphStore with a real conversation (LoCoMo conv-26: Caroline & Melanie, 19 sessions, 184 messages) and querying it.

## Prerequisites

```bash
# Install graphstore
uv sync

# Install Jina v5 Small embedder (1024d, 677M params)
GRAPHSTORE_MODEL_CACHE_DIR=/tmp/gs_models uv run python3 -c "
from graphstore.registry.installer import install_embedder, set_cache_dir
set_cache_dir('/tmp/gs_models')
install_embedder('jina-v5-small-retrieval')
"

# Download LoCoMo dataset
# Get locomo10.json from https://huggingface.co/datasets/Percena/locomo-mc10
# Place at /tmp/locomo/raw/locomo10.json
```

## Populate GraphStore

```python
import os
os.environ['GRAPHSTORE_MODEL_CACHE_DIR'] = '/tmp/gs_models'

from benchmarks.framework.datasets import load_locomo
from benchmarks.framework.adapters.graphstore_ import GraphStoreAdapter

# Load 1 conversation (19 sessions, 184 messages)
ds = load_locomo('/tmp/locomo', max_conversations=1)

# Create adapter with Jina v5 Small embedder
config = {
    'embedder': 'installed',
    'embedder_model': 'jina-v5-small-retrieval',
    'embedder_cache_dir': '/tmp/gs_models',
    'embedder_gpu': True,    # set False if no GPU
    'ceiling_mb': 512,
}
adapter = GraphStoreAdapter(config=config)
adapter.reset()

# Ingest all 19 sessions
for sess in ds.records[0].sessions:
    adapter.ingest(sess)

gs = adapter._gs
print(f'Nodes: {gs.node_count}, Edges: {gs.edge_count}')
# Expected: 240 nodes (184 messages + 19 sessions + 37 entities), 784 edges
```

## Query Examples

### REMEMBER (hybrid retrieval - vector + BM25 + graph + recency)

```python
# Single-hop factual
r = gs.execute('REMEMBER "What did Caroline research" LIMIT 10 WHERE kind = "message"')
for n in r.data[:3]:
    print(f'  score={n["_remember_score"]:.3f} {n.get("content","")[:70]}')
# Expected: "Caroline is researching adoption agencies" near top

# Multi-hop
r = gs.execute('REMEMBER "When did Caroline go to the LGBTQ support group" LIMIT 10 WHERE kind = "message"')

# Open-domain
r = gs.execute('REMEMBER "What pet does Caroline have" LIMIT 10 WHERE kind = "message"')

# With temporal anchor
r = gs.execute('REMEMBER "what happened" AT "2023-05-08" LIMIT 10 WHERE kind = "message"')
```

### RECALL (graph spreading activation)

```python
# From an entity - finds all messages connected via graph edges
r = gs.execute('RECALL FROM "ent:caroline" DEPTH 2 LIMIT 10')
for n in r.data[:5]:
    print(f'  [{n["kind"]}] {n.get("content", n.get("name", ""))[:50]}')

# From a specific message - finds neighbors
r = gs.execute('RECALL FROM "s1:msg0" DEPTH 2 LIMIT 10')
```

### SIMILAR TO (pure vector search)

```python
r = gs.execute('SIMILAR TO "counseling career psychology" LIMIT 5 WHERE kind = "message"')
for n in r.data:
    print(f'  sim={n["_similarity_score"]:.3f} {n.get("content","")[:60]}')
```

### LEXICAL SEARCH (BM25 keyword search)

```python
r = gs.execute('LEXICAL SEARCH "adoption agency" LIMIT 5')
for n in r.data:
    print(f'  bm25={n["_bm25_score"]:.3f} {n.get("content","")[:60]}')
```

### TRAVERSE / SUBGRAPH (graph exploration)

```python
# See a message's neighborhood
r = gs.execute('SUBGRAPH FROM "s1:msg0" DEPTH 2')
print(f'Nodes: {len(r.data["nodes"])}, Edges: {len(r.data["edges"])}')

# What entities exist?
ents = gs.execute('NODES WHERE kind = "entity"')
for e in ents.data[:10]:
    print(f'  {e["id"]}: {e.get("name","")}')
```

### Score breakdown

Every REMEMBER result includes signal breakdown:

```python
r = gs.execute('REMEMBER "Caroline counseling" LIMIT 1 WHERE kind = "message"')
n = r.data[0]
print(f'_remember_score: {n["_remember_score"]}')
print(f'_vector_sim:     {n["_vector_sim"]}')
print(f'_bm25_score:     {n["_bm25_score"]}')
print(f'_recency_score:  {n["_recency_score"]}')
print(f'_graph_score:    {n["_graph_score"]}')
print(f'_recall_score:   {n["_recall_score"]}')
```

## Run Retrieval Recall Test (no LLM needed)

Tests if gold answer keywords appear in top-10 retrieved passages for 50 validated questions:

```bash
uv run python3 -m benchmarks.framework.ratchet50
# Expected: ~40/50 (80%) with jina-v5-small
```

## Run LoCoMo Benchmark with LLM

Requires an LLM for answer generation. Set the model in `benchmarks/framework/llm_client.py`:

```python
QA_MODEL = "minimax/minimax-m2.7:nitro"   # OpenRouter paid
QA_MODEL_OR = "minimax/minimax-m2.7:nitro"
```

Then run:

```bash
# 50Q random sample (~$0.07 on MiniMax nitro)
uv run python3 -c "
import os
os.environ['GRAPHSTORE_MODEL_CACHE_DIR'] = '/tmp/gs_models'
from benchmarks.framework.run_locomo import run_locomo
from benchmarks.framework.datasets import load_locomo
from benchmarks.framework.adapters.graphstore_ import GraphStoreAdapter

ds = load_locomo('/tmp/locomo', max_conversations=1)
config = {
    'embedder': 'installed',
    'embedder_model': 'jina-v5-small-retrieval',
    'embedder_cache_dir': '/tmp/gs_models',
    'embedder_gpu': True,
    'ceiling_mb': 512,
}
adapter = GraphStoreAdapter(config=config)
summary, details = run_locomo(adapter, ds, k=10)
print(f'Overall F1: {summary[\"overall_f1\"]:.4f}')
"

# Full 1986Q (~$0.40 on MiniMax nitro)
uv run python3 -m benchmarks.framework.run_locomo \
  --data-path /tmp/locomo \
  --embedder installed:jina-v5-small-retrieval \
  --k 10
```

## Run LongMemEval on Kaggle

```bash
# Update benchmarks/kaggle/graphstore_jina_500.py with your HF token
# Then push to Kaggle:
kaggle kernels push -p benchmarks/kaggle
```

## Graph Structure After Ingestion

```
240 nodes:
  184 messages  - conversation observations with [date] speaker: content
  19 sessions   - one per conversation session
  37 entities   - extracted via regex (Caroline, Melanie, LGBTQ, months, etc.)

784 edges:
  has_message   - session -> message
  mentions      - message -> entity
  next          - message -> next message (sequential order)
```

Each message has:
- Vector embedding (Jina v5 Small, 1024d)
- FTS5 index entry (BM25 searchable)
- `__event_at__` timestamp (from session date)
- Entity edges (mentions)
- Sequential edges (next)
