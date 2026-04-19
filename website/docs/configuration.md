---
title: Configuration
sidebar_position: 5
---

# Configuration

```python
from graphstore import GraphStore

g = GraphStore(
    path="./brain",
    ceiling_mb=256,
    embedder="default",           # "default" (model2vec), "none", "installed:<name>", or an Embedder
    queued=True,                  # worker-thread write queue + cron scheduler
    vault="./notes",              # markdown vault (None = disabled)
    fusion_method="weighted",     # "weighted" or "rrf"
    recency_half_life_days=7300,  # ~20 years - agent memory decays slowly
    graph_signal_enabled=True,
    nucleus_expansion=False,
    # Vision (only used when INGEST ... USING VISION fires)
    # vision_base_url=None,       # None -> auto-resolve sidecar -> env -> error
    # vision_model="SmolVLM2-2.2B-Instruct-Q4_K_M.gguf",
    # vision_max_tokens=512,
)
```

## Layered config

Config is resolved in layers, later layers win:

1. `config.py` defaults
2. `graphstore.json` in cwd
3. `GRAPHSTORE_*` env vars
4. Constructor kwargs

## graphstore.json reference

Include only the fields you want to override. Missing fields use defaults from `config.py`.

```json
{
  "core": {
    "ceiling_mb": 512,
    "embed_batch_size": 64
  },
  "vector": {
    "search_oversample": 16,
    "similarity_threshold": 0.85,
    "embedder": "default",
    "model2vec_model": "minishlab/M2V_base_output"
  },
  "document": {
    "chunk_max_size": 2000,
    "chunk_overlap": 50,
    "vision_model": "SmolVLM2-2.2B-Instruct-Q4_K_M.gguf",
    "vision_base_url": null,
    "vision_max_tokens": 512
  },
  "dsl": {
    "fusion_method": "weighted",
    "remember_weights": [0.52, 0.25, 0.15, 0.08],
    "recency_half_life_days": 7300,
    "graph_signal_enabled": true,
    "sentence_query_expansion": true,
    "nucleus_expansion": false,
    "reranker": null
  }
}
```

## Env overrides

Flattened by section:

```bash
GRAPHSTORE_CORE_CEILING_MB=512
GRAPHSTORE_DSL_FUSION_METHOD=rrf
GRAPHSTORE_VISION_URL=http://localhost:8080
GRAPHSTORE_VISION_MODEL=smolvlm-500m
GRAPHSTORE_VLM_CACHE_DIR=/mnt/cache/vlm
```

## CLI

```bash
graphstore config --defaults    # dump current defaults as JSON
graphstore config --schema      # JSON Schema for graphstore.json
graphstore config --path graphstore.json   # show resolved values
```

## Single-owner per path

Persistent stores take an advisory lock on `<path>/.graphstore.lock`. A second `GraphStore(path=...)` against the same path raises `StoreInUse` - WAL replay + compact + checkpoint are not safe across processes. In-memory stores are unlocked. OS reclaims the lock on process exit.

## Thread safety

Default is single-threaded. `queued=True` installs a worker thread that serialises writes from multiple callers. BLAS thread count is capped at import time (`OMP_NUM_THREADS=2` by default; override with `GRAPHSTORE_BLAS_CAP=N`) so importing graphstore doesn't saturate all cores.
