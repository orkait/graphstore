# Vector Store - Design Spec

> Add semantic similarity search to graphstore via embedded vector index, completing the three-pillar memory architecture: columnar (facts) + graph (associations) + vectors (meaning).

**Date:** 2026-03-20
**Status:** Design approved, pending implementation plan
**Depends on:** Agentic Brain DB (Phase 1-4, implemented)

---

## 1. Problem

Agents store 1M+ memories. Finding relevant ones requires EXACT field matches (`NODES WHERE topic = "paris"`) or pre-wired graph edges (`RECALL FROM "concept:paris"`). Both require the agent to perfectly pre-organize memories at write time.

A memory about "Eiffel Tower at sunset" tagged `topic = "travel"` with no edge to "paris" is invisible to any paris-related query. The agent can't recall what it didn't anticipate needing.

## 2. Goal

Add vector similarity search so agents can find memories by MEANING without pre-organization. One `SIMILAR TO "Paris travel" LIMIT 10` finds "Eiffel Tower at sunset" because the meanings are close in embedding space - no tags, no edges needed.

## 3. Architecture

```
Three-pillar memory:

┌──────────┐  ┌──────────┐  ┌──────────┐
│ Columns  │  │  Graph   │  │ Vectors  │
│ (facts)  │  │ (assoc)  │  │ (meaning)│
│          │  │          │  │          │
│ WHERE    │  │ RECALL   │  │ SIMILAR  │
│ COUNT    │  │ TRAVERSE │  │ DUPLICATES│
│ GROUP BY │  │ PATH     │  │          │
└──────────┘  └──────────┘  └──────────┘
  numpy         scipy CSR     usearch HNSW
```

Every node optionally has a vector. The VectorStore manages a usearch HNSW index alongside the existing ColumnStore. Queries can combine all three: `SIMILAR TO "text" LIMIT 10 WHERE kind = "memory" AND confidence > 0.5`.

## 4. Dependencies

```toml
[project.dependencies]
# Existing: scipy, lark, numpy
usearch = ">=2.0"          # HNSW index, 5MB, MIT, incremental insert
model2vec = ">=0.4"        # default embedder, 30MB, numpy-only, CPU

[project.optional-dependencies]
onnx = [
    "onnxruntime>=1.24",       # ~17MB CPU, runs ONNX models without torch
    "transformers>=4.57",      # tokenizer + model loading (no torch needed for tokenizer)
    "huggingface_hub>=0.34",   # model download
]
onnx-gpu = [
    "onnxruntime-gpu>=1.24",   # ~250MB, CUDA-accelerated ONNX inference
    "transformers>=4.57",
    "huggingface_hub>=0.34",
]
```

- `usearch` + `model2vec` are core dependencies - always installed, always available
- `onnxruntime` is optional - for EmbeddingGemma-300M (~17MB CPU / ~250MB GPU, no torch)
- `sentence-transformers` is NOT a dependency - torch is too heavy (5.5GB)

Note: `transformers` is needed for `AutoTokenizer` only (tokenization + prompt formatting), not for model inference. ONNX Runtime handles inference. The `transformers` package without torch is ~500MB but does not pull in torch unless explicitly requested.

## 5. Embedder Interface

The embedder contract distinguishes between **queries** and **documents**. This matters because EmbeddingGemma (and other asymmetric models) use different prompt prefixes for queries vs stored documents. Symmetric models (like model2vec) can use the same method for both.

```python
# graphstore/embedder.py

import numpy as np


class Embedder:
    """Base interface for text embedding models.

    Subclasses must implement encode_documents (for storage) and
    encode_queries (for search). Models with asymmetric prefixes
    (like EmbeddingGemma) override both. Symmetric models can
    implement encode() and have both delegate to it.
    """

    @property
    def dims(self) -> int:
        """Output embedding dimensionality."""
        raise NotImplementedError

    def encode_documents(self, texts: list[str], titles: list[str | None] | None = None) -> np.ndarray:
        """Encode texts for storage. Shape: (len(texts), dims).

        Args:
            texts: document texts to embed
            titles: optional per-document titles (used by EmbeddingGemma prefix)
        """
        raise NotImplementedError

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """Encode texts for search/retrieval. Shape: (len(texts), dims)."""
        raise NotImplementedError


class Model2VecEmbedder(Embedder):
    """Default embedder. 30MB, numpy-only, 50k texts/sec on CPU.

    Symmetric model - queries and documents use the same encoding.
    """

    def __init__(self, model_name: str = "minishlab/M2V_base_output"):
        from model2vec import StaticModel
        self._model = StaticModel.from_pretrained(model_name)

    @property
    def dims(self) -> int:
        return self._model.dim

    def encode_documents(self, texts: list[str], titles: list[str | None] | None = None) -> np.ndarray:
        return self._model.encode(texts).astype(np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts).astype(np.float32)


class EmbeddingGemmaONNX(Embedder):
    """EmbeddingGemma-300M via ONNX Runtime. No torch required.

    Supports Matryoshka truncation: 768 (full), 512, 256, 128 dims.
    Uses asymmetric prefixes for queries vs documents per Google's spec.
    ONNX model from: onnx-community/embeddinggemma-300m-ONNX
    Precision options: fp32, q8, q4 (not fp16).
    Max input: 2048 tokens.
    """

    QUERY_PREFIX = "task: search result | query: "
    DOC_PREFIX_TEMPLATE = "title: {title} | text: "

    def __init__(
        self,
        model_id: str = "onnx-community/embeddinggemma-300m-ONNX",
        output_dims: int = 256,
        cache_dir: str | None = None,
    ):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        from huggingface_hub import hf_hub_download

        self._output_dims = output_dims
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

        # Download ONNX model file
        model_path = hf_hub_download(model_id, "model.onnx", cache_dir=cache_dir)
        self._session = ort.InferenceSession(model_path)

    @property
    def dims(self) -> int:
        return self._output_dims

    def encode_documents(self, texts: list[str], titles: list[str | None] | None = None) -> np.ndarray:
        prefixed = []
        for i, text in enumerate(texts):
            title = titles[i] if titles and i < len(titles) and titles[i] else "none"
            prefixed.append(f"{self.DOC_PREFIX_TEMPLATE.format(title=title)}{text}")
        return self._encode(prefixed)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        prefixed = [f"{self.QUERY_PREFIX}{t}" for t in texts]
        return self._encode(prefixed)

    def _encode(self, texts: list[str]) -> np.ndarray:
        inputs = self._tokenizer(texts, padding=True, truncation=True,
                                 max_length=2048, return_tensors="np")
        outputs = self._session.run(None, {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        })
        embeddings = outputs[0]  # (batch, seq_len, hidden_dim)
        # Mean pooling
        mask = inputs["attention_mask"][:, :, np.newaxis]
        pooled = (embeddings * mask).sum(axis=1) / mask.sum(axis=1)
        # Matryoshka truncation
        truncated = pooled[:, :self._output_dims]
        # L2 renormalize after truncation
        norms = np.linalg.norm(truncated, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return (truncated / norms).astype(np.float32)


# --- Model registry ---

EMBEDDER_REGISTRY = {
    "model2vec": {
        "class": "Model2VecEmbedder",
        "deps": [],
        "description": "Default. 30MB, numpy-only, CPU, 50k/sec.",
    },
    "embeddinggemma-300m": {
        "class": "EmbeddingGemmaONNX",
        "deps": ["onnxruntime", "transformers", "huggingface_hub"],
        "description": "Google EmbeddingGemma via ONNX. 768d Matryoshka, 2048 tokens. Requires: pip install graphstore[onnx]",
    },
}


def load_embedder(name: str = "model2vec", **kwargs) -> Embedder:
    """Load an embedder by registry name."""
    if name == "model2vec":
        return Model2VecEmbedder(**kwargs)
    elif name == "embeddinggemma-300m":
        return EmbeddingGemmaONNX(**kwargs)
    else:
        raise ValueError(f"Unknown embedder: {name!r}. Available: {list(EMBEDDER_REGISTRY.keys())}")
```

### 5.1 Embedder Responsibilities

The embedder wrapper owns four responsibilities:

1. **Tokenizer loading** - `AutoTokenizer.from_pretrained()` for ONNX models
2. **Query/document prefixing** - EmbeddingGemma requires `"task: search result | query: "` for queries and `"title: {title} | text: "` for documents
3. **Truncation/padding** - max 2048 tokens for EmbeddingGemma, tokenizer handles this
4. **Post-processing** - Matryoshka dimension truncation (768/512/256/128) with L2 renormalization after truncation

## 6. VectorStore

```python
# graphstore/vectors.py

import numpy as np
from usearch.index import Index


class VectorStore:
    """HNSW vector index for semantic similarity search."""

    def __init__(self, dims: int, capacity: int = 1024):
        self._dims = dims
        self._index = Index(ndim=dims, metric="cos", dtype="f32")
        self._has_vector = np.zeros(capacity, dtype=bool)
        self._capacity = capacity

    @property
    def dims(self) -> int:
        return self._dims

    def add(self, slot: int, vector: np.ndarray) -> None:
        """Add or replace vector for a slot."""
        if slot >= self._capacity:
            self.grow(self._capacity * 2)
        vec = np.asarray(vector, dtype=np.float32)
        if len(vec) != self._dims:
            raise ValueError(f"Expected {self._dims} dims, got {len(vec)}")
        self._index.add(slot, vec)
        self._has_vector[slot] = True

    def remove(self, slot: int) -> None:
        """Remove vector for a slot."""
        if slot < self._capacity and self._has_vector[slot]:
            self._index.remove(slot)
            self._has_vector[slot] = False

    def search(self, query: np.ndarray, k: int, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors. Returns (slot_indices, distances).

        If mask is provided, only slots where mask[slot] == True are considered.
        Uses usearch's built-in filtering for efficiency.
        """
        query = np.asarray(query, dtype=np.float32)
        if mask is not None:
            # usearch supports predicate-based filtering
            results = self._index.search(query, k * 3)  # oversample
            valid = [(key, dist) for key, dist in zip(results.keys, results.distances)
                     if key < len(mask) and mask[key]]
            valid = valid[:k]
            if not valid:
                return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
            slots = np.array([v[0] for v in valid], dtype=np.int64)
            dists = np.array([v[1] for v in valid], dtype=np.float32)
            return slots, dists
        else:
            results = self._index.search(query, k)
            return np.array(results.keys, dtype=np.int64), np.array(results.distances, dtype=np.float32)

    def has_vector(self, slot: int) -> bool:
        return slot < self._capacity and self._has_vector[slot]

    def get_vector(self, slot: int) -> np.ndarray | None:
        """Get stored vector for a slot."""
        if not self.has_vector(slot):
            return None
        return self._index.get(slot)

    def grow(self, new_capacity: int) -> None:
        old_pres = self._has_vector
        self._has_vector = np.zeros(new_capacity, dtype=bool)
        self._has_vector[:len(old_pres)] = old_pres
        self._capacity = new_capacity

    def count(self) -> int:
        return int(np.sum(self._has_vector))

    @property
    def memory_bytes(self) -> int:
        """Approximate memory usage of the HNSW index + presence mask."""
        # usearch memory: ~(dims * 4 + 64) bytes per vector (approximate)
        n = self.count()
        return n * (self._dims * 4 + 64) + self._has_vector.nbytes

    def save(self) -> bytes:
        """Serialize the index to bytes for persistence."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        self._index.save(path)
        with open(path, "rb") as f:
            data = f.read()
        os.unlink(path)
        return data

    def load(self, data: bytes) -> None:
        """Deserialize the index from bytes."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(data)
            path = f.name
        self._index = Index.restore(path, ndim=self._dims, metric="cos", dtype="f32")
        os.unlink(path)
```

## 7. CoreStore Integration

### 7.1 New fields on CoreStore

```python
class CoreStore:
    def __init__(self, ceiling_bytes, vector_ceiling_bytes):
        # ... existing ...
        self._vector_ceiling_bytes = vector_ceiling_bytes
        self.vectors: VectorStore | None = None  # lazy init on first vector

    def _ensure_vector_store(self, dims: int) -> VectorStore:
        if self.vectors is None:
            self.vectors = VectorStore(dims=dims, capacity=self._capacity)
        return self.vectors
```

### 7.2 Integration with write operations

**put_node:** If an embedder is configured and the node's kind has an EMBED field declared, auto-embed and store vector.

**update_node:** If the EMBED field is modified, re-embed and replace vector.

**delete_node:** Remove vector from index via `vectors.remove(slot)`.

**_grow:** Grow vector store's presence array alongside columns.

**increment_field:** No vector impact (numeric fields aren't embedded).

### 7.3 Schema extension: EMBED clause

```sql
SYS REGISTER NODE KIND "memory" REQUIRED content:string, topic:string EMBED content
```

The `EMBED content` clause tells graphstore which field to auto-embed when a node of this kind is created or updated. Stored in `SchemaRegistry` alongside required/optional fields.

If no embedder is installed, `EMBED` declarations are stored but inactive - vectors are only computed when an embedder is available.

## 8. DSL Commands

### 8.1 VECTOR clause on CREATE/UPSERT

```lark
create_node: "CREATE" "NODE" STRING field_pairs vector_clause? expires_clause?
           | "CREATE" "NODE" "AUTO" field_pairs vector_clause? expires_clause? -> create_node_auto
upsert_node: "UPSERT" "NODE" STRING field_pairs vector_clause? expires_clause?

vector_clause: "VECTOR" "[" NUMBER ("," NUMBER)* "]"
```

AST: Add `vector: list[float] | None` to `CreateNode` and `UpsertNode`.

Priority: explicit `VECTOR [...]` always wins over auto-embed. If both are present, use the provided vector.

### 8.2 SIMILAR TO

```lark
similar_q: "SIMILAR" "TO" similar_target limit_clause? where_clause?
similar_target: vector_literal                    -> similar_vector
              | STRING                            -> similar_text
              | "NODE" STRING                     -> similar_node

vector_literal: "[" NUMBER ("," NUMBER)* "]"
```

Add `similar_q` to `read_query` alternatives.

AST:
```python
@dataclass
class SimilarQuery:
    target_vector: list[float] | None = None   # SIMILAR TO [...]
    target_text: str | None = None             # SIMILAR TO "text"
    target_node_id: str | None = None          # SIMILAR TO NODE "id"
    limit: LimitClause | None = None
    where: WhereClause | None = None
```

Executor:
1. Resolve query vector:
   - `target_vector`: use directly
   - `target_text`: embed via configured embedder (error if no embedder)
   - `target_node_id`: get stored vector from VectorStore (error if node has no vector)
2. Compute `live_mask` (tombstone + TTL + retracted + context)
3. Call `vectors.search(query_vec, k, mask=live_mask)`
4. If WHERE clause present, post-filter results via column mask
5. Materialize matching slots
6. Annotate each result with `_similarity_score`

Result: `kind="nodes"`, data includes `_similarity_score` per node.

### 8.3 SYS DUPLICATES

```lark
sys_duplicates: "DUPLICATES" where_clause? threshold_clause?
threshold_clause: "THRESHOLD" NUMBER
```

AST:
```python
@dataclass
class SysDuplicates:
    where: WhereClause | None = None
    threshold: float = 0.95
```

Executor:
1. Apply `live_mask` + optional WHERE filter
2. For each node with a vector, find its nearest neighbor
3. If similarity > threshold, add to duplicates list
4. Return pairs: `[{node_a: "id1", node_b: "id2", similarity: 0.97}, ...]`

## 9. Memory Ceiling Split

```python
g = GraphStore(
    ceiling_mb=256,          # graph: columns + edges + CSR matrices
    vector_ceiling_mb=2048,  # vectors: HNSW index + presence array
)
```

Enforcement:
- Graph ceiling checked in `put_node`, `put_edge` (existing)
- Vector ceiling checked before `vectors.add()`:
  ```python
  if self.vectors and self.vectors.memory_bytes > self._vector_ceiling_bytes:
      raise CeilingExceeded(current_mb, ceiling_mb, "vector_add")
  ```

Default: `vector_ceiling_mb=2048` (2GB). Sufficient for ~2M vectors at 256 dims.

## 10. Persistence

### Serializer additions

```python
# In checkpoint():
if store.vectors is not None and store.vectors.count() > 0:
    # Save HNSW index as blob
    index_bytes = store.vectors.save()
    conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                 ("vector_index", index_bytes, "usearch"))
    # Save presence mask
    pres = store.vectors._has_vector[:store._next_slot]
    conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                 ("vector_presence", pres.tobytes(), str(pres.dtype)))
    # Save dims
    conn.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)",
                 ("vector_dims", str(store.vectors.dims).encode(), "text"))
```

### Deserializer additions

```python
# In load():
dims_row = conn.execute("SELECT data FROM blobs WHERE key='vector_dims'").fetchone()
if dims_row:
    dims = int(dims_row[0])
    store._ensure_vector_store(dims)

    index_row = conn.execute("SELECT data FROM blobs WHERE key='vector_index'").fetchone()
    if index_row:
        store.vectors.load(index_row[0])

    pres_row = conn.execute("SELECT data, dtype FROM blobs WHERE key='vector_presence'").fetchone()
    if pres_row:
        loaded = np.frombuffer(pres_row[0], dtype=np.dtype(pres_row[1])).copy()
        store.vectors._has_vector[:len(loaded)] = loaded
```

## 11. Snapshot / Rollback Integration

### SYS SNAPSHOT

```python
# Add to snapshot dict:
snap["vector_index"] = store.vectors.save() if store.vectors and store.vectors.count() > 0 else None
snap["vector_presence"] = store.vectors._has_vector[:ns].copy() if store.vectors else None
snap["vector_dims"] = store.vectors.dims if store.vectors else None
```

### SYS ROLLBACK

```python
if snap.get("vector_index") is not None:
    store._ensure_vector_store(snap["vector_dims"])
    store.vectors.load(snap["vector_index"])
    store.vectors._has_vector[:ns] = snap["vector_presence"]
```

### Batch rollback

Same pattern as snapshot - save/restore vector index bytes around batch execution.

Note: usearch save/load at 1M vectors takes ~1-2 seconds. This is acceptable for explicit SYS ROLLBACK but adds overhead to batch rollback. For batches, consider only rolling back vector additions (remove added slots) rather than full index serialization.

### WHAT IF (counterfactual)

Counterfactual doesn't modify vectors (RETRACT just marks `__retracted__=1`, doesn't delete the vector). No vector state save/restore needed for WHAT IF.

## 12. live_mask Integration

`SIMILAR TO` respects the same visibility rules as every other query:

- Tombstoned nodes: excluded (not in index after delete)
- Expired nodes: excluded via mask filter on search results
- Retracted nodes: excluded via mask filter
- Context-bound: only context-tagged nodes returned

The mask is passed to `vectors.search(query, k, mask=live_mask)`. usearch oversamples (k*3) and post-filters, since HNSW doesn't natively support mask-based exclusion.

## 13. Interaction with Existing Features

| Feature | Vector Interaction |
|---|---|
| CREATE NODE | Auto-embed if kind has EMBED field and embedder is available |
| CREATE NODE ... VECTOR [...] | Store explicit vector, skip auto-embed |
| UPDATE NODE | Re-embed if EMBED field modified |
| DELETE NODE | Remove from vector index |
| MERGE NODE "a" INTO "b" | Keep target's vector, discard source's |
| RETRACT "x" | Vector stays, node excluded from search via live_mask |
| SYS EXPIRE | Remove expired vectors from index |
| ASSERT "x" ... | Auto-embed if kind has EMBED field |
| BATCH rollback | Restore vector state |
| SYS SNAPSHOT/ROLLBACK | Full vector index save/restore |
| BIND CONTEXT | SIMILAR TO filtered to context nodes only |
| RECALL FROM "x" | Independent - graph activation, not vector search |

## 14. Auto-embed Configuration

### Schema registration with EMBED

```sql
SYS REGISTER NODE KIND "memory" REQUIRED content:string, topic:string EMBED content
```

The EMBED clause is stored in `SchemaRegistry`:

```python
@dataclass
class NodeKindDef:
    required: set[str]
    optional: set[str]
    field_types: dict[str, str]
    embed_field: str | None = None  # NEW: field name to auto-embed
```

### Auto-embed flow

On `put_node` / `update_node`:
1. Look up kind in schema
2. If `embed_field` is set and embedder is available:
   - Get field value from the node data
   - If value is a non-empty string: `vector = embedder.encode([value])[0]`
   - Store: `vectors.add(slot, vector)`
3. If explicit VECTOR clause provided: use that, skip auto-embed

### No embedder installed

If `embed_field` is declared but no embedder is available:
- No error on CREATE (vectors are optional)
- `SIMILAR TO "text"` raises: `"Text similarity requires an embedder. Install: pip install graphstore[embeddings]"`
- `SIMILAR TO [0.12, ...]` works (raw vector search, no embedder needed)

## 15. GraphStore Constructor + Model Management

### 15.1 Constructor

```python
class GraphStore:
    def __init__(
        self,
        path: str | None = None,
        ceiling_mb: int = 256,
        vector_ceiling_mb: int = 2048,
        embedder: Embedder | str | None = "default",
        allow_system_queries: bool = True,
    ):
        # ...
        if embedder == "default":
            self._embedder = load_embedder("model2vec")
        elif isinstance(embedder, str):
            self._embedder = load_embedder(embedder)
        elif isinstance(embedder, Embedder):
            self._embedder = embedder
        elif embedder is None:
            self._embedder = None  # no auto-embedding, VECTOR [...] only
```

Default is `"default"` which loads model2vec automatically. model2vec is a core dependency so this always works.

### 15.2 Model Management DSL

Agents and developers can manage embedders via DSL commands:

```sql
-- List available embedders and their status
SYS EMBEDDERS
-- Returns: [{name: "model2vec", status: "active", dims: 256},
--           {name: "embeddinggemma-300m", status: "available", deps: "pip install graphstore[onnx]"}]

-- Download and install a model (downloads ONNX weights to ~/.graphstore/models/)
SYS INSTALL EMBEDDER "embeddinggemma-300m"

-- Switch active embedder for this graph (persisted in sqlite metadata)
SYS SET EMBEDDER "embeddinggemma-300m"
-- Optional: set output dims for Matryoshka models
SYS SET EMBEDDER "embeddinggemma-300m" DIMS 256

-- Switch back to default
SYS SET EMBEDDER "model2vec"

-- Check current embedder
SYS STATS EMBEDDER
-- Returns: {name: "model2vec", dims: 256, vectors_stored: 12345}
```

The active embedder name is persisted in the sqlite `metadata` table. On next `GraphStore(path=...)` open, the saved embedder is loaded automatically.

**Important:** Changing embedders invalidates existing vectors (different models produce incompatible embedding spaces). `SYS SET EMBEDDER` warns and requires re-embedding:

```sql
SYS SET EMBEDDER "embeddinggemma-300m"
-- Warning: changing embedder invalidates 12345 existing vectors.
-- Run SYS REEMBED to re-encode all documents with the new model.

SYS REEMBED
-- Re-embeds all nodes that have an EMBED field declared in their schema.
-- Progress: 12345/12345 nodes re-embedded.
```

### 15.3 Model Cache

Downloaded models are stored in `~/.graphstore/models/`:

```
~/.graphstore/models/
  embeddinggemma-300m-onnx/
    model.onnx        # ONNX weights
    tokenizer.json    # HuggingFace tokenizer
    config.json       # Model config
```

This directory is managed by `huggingface_hub` download utilities. Models are downloaded once and reused across all GraphStore instances.

## 16. New Error Types

```python
class VectorError(GraphStoreError):
    """Vector operation failure."""
    pass

class EmbedderRequired(VectorError):
    """Text-based SIMILAR TO requires an embedder."""
    pass

class VectorNotFound(VectorError):
    """Node has no stored vector."""
    pass
```

## 17. Files Changed

| File | Action | Responsibility |
|---|---|---|
| `graphstore/embedder.py` | **Create** | Embedder interface + Model2VecEmbedder |
| `graphstore/vectors.py` | **Create** | VectorStore class (usearch HNSW) |
| `graphstore/store.py` | Modify | Add vectors field, _ensure_vector_store, ceiling split |
| `graphstore/schema.py` | Modify | Add embed_field to NodeKindDef |
| `graphstore/errors.py` | Modify | Add VectorError, EmbedderRequired, VectorNotFound |
| `graphstore/dsl/grammar.lark` | Modify | VECTOR clause, SIMILAR TO, SYS DUPLICATES, EMBED |
| `graphstore/dsl/ast_nodes.py` | Modify | SimilarQuery, SysDuplicates, vector fields on CreateNode |
| `graphstore/dsl/transformer.py` | Modify | Transform new grammar rules |
| `graphstore/dsl/executor.py` | Modify | _similar, auto-embed on create/update |
| `graphstore/dsl/executor_system.py` | Modify | _duplicates, snapshot/rollback vectors |
| `graphstore/persistence/serializer.py` | Modify | Persist vector index |
| `graphstore/persistence/deserializer.py` | Modify | Load vector index |
| `graphstore/__init__.py` | Modify | embedder param, vector_ceiling_mb, new error exports |
| `tests/test_vectors.py` | **Create** | VectorStore unit tests |
| `tests/test_similar.py` | **Create** | SIMILAR TO integration tests |

## 18. Performance Targets

Targets based on usearch benchmarks and numpy baselines. Embedding throughput must be benchmarked on actual hardware before publishing claims.

| Operation | Scale | Target | Notes |
|---|---|---|---|
| Vector add (single) | any | < 100 μs | usearch incremental insert |
| SIMILAR TO LIMIT 10 | 100k vectors | < 500 μs | HNSW search + materialize |
| SIMILAR TO LIMIT 10 | 1M vectors | < 1 ms | HNSW logarithmic scaling |
| SIMILAR TO + WHERE | 100k vectors | < 1 ms | HNSW + column mask filter |
| Auto-embed (model2vec) | per text | benchmark needed | model2vec claims ~50k/sec on CPU |
| Auto-embed (EmbeddingGemma ONNX) | per text | benchmark needed | depends on CPU/GPU, batch size, precision |
| Vector persistence save | 1M vectors | < 2 sec | usearch serialization |
| Vector persistence load | 1M vectors | < 2 sec | usearch deserialization |
| SYS DUPLICATES | 100k vectors | < 5 sec | O(n) nearest-neighbor scan |
| Memory per vector (256d) | per node | ~1.1 KB | 256*4 + 64 bytes HNSW overhead |
| 1M vectors storage | 256 dims | ~1.1 GB | Fits in 2GB default ceiling |

### 18.1 ONNX Model Provenance

The ONNX export of EmbeddingGemma-300M is hosted at `onnx-community/embeddinggemma-300m-ONNX` on HuggingFace. The original model is by Google DeepMind (`google/embeddinggemma-300m`). The ONNX conversion is maintained by the onnx-community, not Google directly. The ONNX model card documents fp32, q8, and q4 precision options (not fp16).

### 18.2 EmbeddingGemma Specifications

- **Parameters:** 308M
- **Max input tokens:** 2048
- **Full embedding dims:** 768
- **Matryoshka truncation:** 512, 256, 128 (with L2 renormalization after truncation)
- **Asymmetric prefixes:** queries use `"task: search result | query: "`, documents use `"title: {title} | text: "`
- **ONNX precision:** fp32, q8, q4

## 19. DSL Summary

```sql
-- Schema: declare auto-embed field
SYS REGISTER NODE KIND "memory" REQUIRED content:string, topic:string EMBED content

-- Create with auto-embed (requires embedder)
CREATE NODE "m1" kind = "memory" content = "Eiffel Tower at sunset" topic = "travel"

-- Create with explicit vector (no embedder needed)
CREATE NODE "m2" kind = "raw" VECTOR [0.12, -0.34, ...] name = "pre-computed"

-- Search by text (embeds query automatically)
SIMILAR TO "European travel" LIMIT 10
SIMILAR TO "European travel" LIMIT 10 WHERE kind = "memory" AND topic = "travel"

-- Search by vector
SIMILAR TO [0.12, -0.34, ...] LIMIT 10

-- Search using another node's vector
SIMILAR TO NODE "concept:paris" LIMIT 10

-- Find near-duplicates
SYS DUPLICATES WHERE kind = "memory" THRESHOLD 0.95
```
