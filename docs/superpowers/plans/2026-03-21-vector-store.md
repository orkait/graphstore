# Vector Store + Codebase Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure graphstore into its final package layout, then add semantic similarity search via usearch HNSW + model2vec embeddings.

**Architecture:** Move existing flat files into `core/` package, split executor.py (2257 lines) into reads/writes/base, extract GraphStore class from `__init__.py`. Then add three new packages: `vector/` (usearch HNSW index), `embedding/` (Embedder protocol + model2vec), `registry/` (model management + installer for EmbeddingGemma ONNX opt-in).

**Tech Stack:** usearch (HNSW), model2vec (default embedder), onnxruntime + tokenizers (opt-in EmbeddingGemma), numpy

**Spec:** `docs/superpowers/specs/2026-03-20-vector-store-design.md`

---

## File Structure

### Phase A: Restructure (move existing code)

| From | To | Notes |
|------|------|-------|
| `graphstore/store.py` | `graphstore/core/store.py` | CoreStore, untouched logic |
| `graphstore/columns.py` | `graphstore/core/columns.py` | ColumnStore |
| `graphstore/edges.py` | `graphstore/core/edges.py` | EdgeMatrices |
| `graphstore/strings.py` | `graphstore/core/strings.py` | StringTable |
| `graphstore/memory.py` | `graphstore/core/memory.py` | Ceiling enforcement |
| `graphstore/schema.py` | `graphstore/core/schema.py` | SchemaRegistry |
| `graphstore/path.py` | `graphstore/core/path.py` | BFS, Dijkstra |
| `graphstore/types.py` | `graphstore/core/types.py` | Result, Edge |
| `graphstore/errors.py` | `graphstore/core/errors.py` | Error hierarchy |
| `graphstore/__init__.py` | `graphstore/graphstore.py` + thin `__init__.py` | Extract GraphStore class |
| `graphstore/dsl/executor.py` | Split into `executor_base.py` + `executor_reads.py` + `executor_writes.py` | 2257 lines → ~800 each |

### Phase B: Vector + Embedding Layer (new code)

| File | Action | Responsibility |
|------|--------|----------------|
| `graphstore/vector/__init__.py` | Create | Package init |
| `graphstore/vector/index.py` | Create | usearch HNSW wrapper |
| `graphstore/vector/store.py` | Create | VectorStore: slot ↔ vector + ceiling |
| `graphstore/embedding/__init__.py` | Create | Package init |
| `graphstore/embedding/base.py` | Create | Embedder protocol |
| `graphstore/embedding/model2vec_embedder.py` | Create | Default embedder |
| `graphstore/embedding/onnx_hf_embedder.py` | Create | EmbeddingGemma ONNX adapter |
| `graphstore/embedding/postprocess.py` | Create | L2 norm, Matryoshka truncation |
| `graphstore/registry/__init__.py` | Create | Package init |
| `graphstore/registry/models.py` | Create | SUPPORTED_MODELS config dict |
| `graphstore/registry/installer.py` | Create | Download + verify + smoke test |
| `graphstore/dsl/grammar.lark` | Modify | Add SIMILAR TO, VECTOR clause |
| `graphstore/dsl/ast_nodes.py` | Modify | Add SimilarQuery, vector field |
| `graphstore/dsl/transformer.py` | Modify | Transform new rules |
| `graphstore/dsl/executor_reads.py` | Modify | Add _similar |
| `graphstore/dsl/executor_writes.py` | Modify | Auto-embed on create/update |
| `graphstore/dsl/executor_system.py` | Modify | Add _duplicates, _embedders |
| `graphstore/core/errors.py` | Modify | Add VectorError, EmbedderRequired |
| `graphstore/core/schema.py` | Modify | Add embed_field |
| `graphstore/core/memory.py` | Modify | Split graph/vector ceiling |
| `graphstore/persistence/serializer.py` | Modify | Persist vector index |
| `graphstore/persistence/deserializer.py` | Modify | Load vector index |
| `graphstore/graphstore.py` | Modify | Add embedder param, vector routing |
| `graphstore/cli.py` | Modify | Add install-embedder command |
| `pyproject.toml` | Modify | Add usearch, model2vec to deps |

---

## Phase A: Restructure

### Task 1: Create core/ package and move files

**Files:**
- Create: `graphstore/core/__init__.py`
- Move: 9 files from `graphstore/` to `graphstore/core/`

- [ ] **Step 1: Create core/ directory with __init__.py**

```bash
mkdir -p graphstore/core
```

Create `graphstore/core/__init__.py`:
```python
"""Core graph engine: columnar storage, sparse matrices, node/edge CRUD."""
from graphstore.core.store import CoreStore
from graphstore.core.columns import ColumnStore
from graphstore.core.edges import EdgeMatrices
from graphstore.core.strings import StringTable
from graphstore.core.types import Result, Edge
from graphstore.core.errors import (
    GraphStoreError, QueryError, NodeNotFound, NodeExists,
    CeilingExceeded, VersionMismatch, SchemaError,
    CostThresholdExceeded, BatchRollback, AggregationError,
)
from graphstore.core.schema import SchemaRegistry
```

- [ ] **Step 2: Move files with git mv**

```bash
git mv graphstore/store.py graphstore/core/store.py
git mv graphstore/columns.py graphstore/core/columns.py
git mv graphstore/edges.py graphstore/core/edges.py
git mv graphstore/strings.py graphstore/core/strings.py
git mv graphstore/memory.py graphstore/core/memory.py
git mv graphstore/schema.py graphstore/core/schema.py
git mv graphstore/path.py graphstore/core/path.py
git mv graphstore/types.py graphstore/core/types.py
git mv graphstore/errors.py graphstore/core/errors.py
```

- [ ] **Step 3: Update ALL internal imports**

Every `from graphstore.store import` becomes `from graphstore.core.store import`. Every `from graphstore.errors import` becomes `from graphstore.core.errors import`. Etc.

Files to update (find all with grep):
- `graphstore/__init__.py`
- `graphstore/core/store.py` (imports from edges, errors, memory, columns, strings)
- `graphstore/core/columns.py` (imports from strings)
- `graphstore/core/schema.py` (imports from errors)
- `graphstore/dsl/executor.py` (imports from store, errors, ast_nodes, types, path, cost_estimator)
- `graphstore/dsl/executor_system.py` (imports from store, errors, types, schema)
- `graphstore/dsl/transformer.py` (imports from ast_nodes)
- `graphstore/dsl/parser.py`
- `graphstore/dsl/cost_estimator.py`
- `graphstore/persistence/serializer.py`
- `graphstore/persistence/deserializer.py`
- `graphstore/server.py`
- `graphstore/cli.py`
- ALL test files (35 files with graphstore imports)

Use this pattern for each file:
```
from graphstore.store    → from graphstore.core.store
from graphstore.errors   → from graphstore.core.errors
from graphstore.types    → from graphstore.core.types
from graphstore.schema   → from graphstore.core.schema
from graphstore.strings  → from graphstore.core.strings
from graphstore.columns  → from graphstore.core.columns
from graphstore.edges    → from graphstore.core.edges
from graphstore.memory   → from graphstore.core.memory
from graphstore.path     → from graphstore.core.path
```

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest tests/ -v --tb=short`
Expected: ALL 634 pass

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: move core engine files to graphstore/core/ package"
```

---

### Task 2: Extract GraphStore from __init__.py

**Files:**
- Create: `graphstore/graphstore.py`
- Modify: `graphstore/__init__.py`

- [ ] **Step 1: Move GraphStore class to graphstore.py**

Cut the `GraphStore` class and all its methods from `__init__.py` into a new `graphstore/graphstore.py`. Keep imports for its dependencies.

- [ ] **Step 2: Make __init__.py a thin re-export**

```python
"""graphstore - Agentic brain DB with semantic search."""

__version__ = "0.2.0"

from graphstore.graphstore import GraphStore
from graphstore.core.store import CoreStore
from graphstore.core.schema import SchemaRegistry
from graphstore.core.types import Result, Edge
from graphstore.core.errors import (
    GraphStoreError, QueryError, NodeNotFound, NodeExists,
    CeilingExceeded, VersionMismatch, SchemaError,
    CostThresholdExceeded, BatchRollback, AggregationError,
)

__all__ = [
    "GraphStore", "CoreStore", "SchemaRegistry",
    "Result", "Edge",
    "GraphStoreError", "QueryError", "NodeNotFound", "NodeExists",
    "CeilingExceeded", "VersionMismatch", "SchemaError",
    "CostThresholdExceeded", "BatchRollback", "AggregationError",
]
```

- [ ] **Step 3: Update any imports that reference GraphStore from __init__**

Check: `from graphstore import GraphStore` should still work (re-exported).
Check: `from graphstore.__init__ import` patterns in tests.

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest tests/ -v --tb=short`
Expected: ALL 634 pass

- [ ] **Step 5: Commit**

```bash
git add graphstore/graphstore.py graphstore/__init__.py
git commit -m "refactor: extract GraphStore class to graphstore/graphstore.py"
```

---

### Task 3: Split executor.py into base/reads/writes

**Files:**
- Create: `graphstore/dsl/executor_base.py`
- Create: `graphstore/dsl/executor_reads.py`
- Create: `graphstore/dsl/executor_writes.py`
- Delete: `graphstore/dsl/executor.py`

- [ ] **Step 1: Create executor_base.py with shared methods**

Extract from executor.py into executor_base.py:
- `class ExecutorBase` with `__init__(self, store, schema, embedder, vector_store)`
- `_compute_live_mask()`
- `_materialize_slot()`
- `_resolve_slot()`
- `_is_visible_by_id()`
- `_filter_visible()`
- `_try_column_filter()`
- `_try_index_lookup()`
- `_make_raw_predicate()`
- `_eval_where()`, `_eval_condition()`, `_eval_degree_condition()`
- `_compare()`
- `_extract_kind_from_where()`, `_strip_kind_from_expr()`, `_is_simple_kind_filter()`
- `_extract_edge_type_from_expr()`

- [ ] **Step 2: Create executor_reads.py**

```python
from graphstore.dsl.executor_base import ExecutorBase

class ReadExecutor(ExecutorBase):
    """Handles all read queries."""
```

Move these methods:
- `_node`, `_nodes`, `_edges`, `_count`
- `_traverse`, `_subgraph`
- `_path`, `_paths`, `_shortest_path`, `_distance`
- `_weighted_shortest_path`, `_weighted_distance`
- `_ancestors`, `_descendants`, `_common_neighbors`
- `_match` and all match helpers
- `_aggregate`
- `_recall`
- `_counterfactual`

- [ ] **Step 3: Create executor_writes.py**

```python
from graphstore.dsl.executor_base import ExecutorBase

class WriteExecutor(ExecutorBase):
    """Handles all write queries."""
```

Move these methods:
- `_create_node`, `_update_node`, `_upsert_node`
- `_delete_node`, `_delete_nodes`, `_update_nodes`
- `_create_edge`, `_update_edge`, `_delete_edge`, `_delete_edges`
- `_increment`
- `_assert`, `_retract`
- `_merge`
- `_propagate`
- `_bind_context`, `_discard_context`
- `_batch`

- [ ] **Step 4: Create unified Executor that composes both**

In `executor_reads.py` or a new `executor.py` (thin):
```python
class Executor(ReadExecutor, WriteExecutor):
    """Full executor combining reads and writes."""

    def _dispatch(self, ast):
        """Route AST to appropriate handler."""
        # ... existing dispatch logic using isinstance checks
```

Or use multiple inheritance: `class Executor(ReadExecutor, WriteExecutor)` since both inherit ExecutorBase.

- [ ] **Step 5: Update imports everywhere**

`graphstore/graphstore.py` imports `Executor` - update path.
`graphstore/dsl/__init__.py` - update exports if any.

- [ ] **Step 6: Delete old executor.py**

```bash
git rm graphstore/dsl/executor.py
```

- [ ] **Step 7: Run full test suite**

Run: `uv run pytest tests/ -v --tb=short`
Expected: ALL 634 pass

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: split executor.py (2257 lines) into base/reads/writes"
```

---

## Phase B: Vector + Embedding Layer

### Task 4: VectorStore (usearch HNSW wrapper)

**Files:**
- Create: `graphstore/vector/__init__.py`
- Create: `graphstore/vector/index.py`
- Create: `graphstore/vector/store.py`
- Test: `tests/test_vector_store.py`

- [ ] **Step 1: Add usearch to pyproject.toml dependencies**

- [ ] **Step 2: Write VectorStore tests**

```python
import numpy as np
from graphstore.vector.store import VectorStore

class TestVectorStore:
    def test_add_and_search(self):
        vs = VectorStore(dims=4, capacity=100)
        vs.add(0, np.array([1.0, 0.0, 0.0, 0.0]))
        vs.add(1, np.array([0.9, 0.1, 0.0, 0.0]))
        vs.add(2, np.array([0.0, 0.0, 1.0, 0.0]))
        slots, dists = vs.search(np.array([1.0, 0.0, 0.0, 0.0]), k=2)
        assert 0 in slots
        assert 1 in slots
        assert 2 not in slots

    def test_remove(self):
        vs = VectorStore(dims=4, capacity=100)
        vs.add(0, np.array([1.0, 0.0, 0.0, 0.0]))
        vs.remove(0)
        assert not vs.has_vector(0)

    def test_search_with_mask(self):
        vs = VectorStore(dims=4, capacity=100)
        vs.add(0, np.array([1.0, 0.0, 0.0, 0.0]))
        vs.add(1, np.array([0.9, 0.1, 0.0, 0.0]))
        mask = np.array([False, True] + [False] * 98)
        slots, _ = vs.search(np.array([1.0, 0.0, 0.0, 0.0]), k=5, mask=mask)
        assert 0 not in slots  # masked out
        assert 1 in slots

    def test_save_and_load(self):
        vs = VectorStore(dims=4, capacity=100)
        vs.add(0, np.array([1.0, 0.0, 0.0, 0.0]))
        data = vs.save()
        vs2 = VectorStore(dims=4, capacity=100)
        vs2.load(data)
        assert vs2.has_vector(0)

    def test_memory_bytes(self):
        vs = VectorStore(dims=256, capacity=1000)
        for i in range(100):
            vs.add(i, np.random.randn(256).astype(np.float32))
        assert vs.memory_bytes > 0
        assert vs.count() == 100
```

- [ ] **Step 3: Implement VectorStore**

Follow spec section 6. Key: usearch Index with cosine metric, incremental add/remove, search with mask via oversampling.

- [ ] **Step 4: Run tests, commit**

```bash
git commit -m "feat: VectorStore with usearch HNSW index"
```

---

### Task 5: Embedder interface + model2vec

**Files:**
- Create: `graphstore/embedding/__init__.py`
- Create: `graphstore/embedding/base.py`
- Create: `graphstore/embedding/model2vec_embedder.py`
- Create: `graphstore/embedding/postprocess.py`
- Modify: `pyproject.toml` (add model2vec dep)
- Test: `tests/test_embedding.py`

- [ ] **Step 1: Add model2vec to pyproject.toml**

- [ ] **Step 2: Write tests**

```python
import numpy as np
from graphstore.embedding.model2vec_embedder import Model2VecEmbedder

class TestModel2Vec:
    def test_encode_queries(self):
        emb = Model2VecEmbedder()
        vecs = emb.encode_queries(["hello world"])
        assert vecs.shape == (1, emb.dims)
        assert vecs.dtype == np.float32

    def test_encode_documents(self):
        emb = Model2VecEmbedder()
        vecs = emb.encode_documents(["hello world", "foo bar"])
        assert vecs.shape == (2, emb.dims)

    def test_similar_texts_have_similar_vectors(self):
        emb = Model2VecEmbedder()
        v1 = emb.encode_queries(["Paris France"])[0]
        v2 = emb.encode_queries(["France Paris"])[0]
        v3 = emb.encode_queries(["quantum physics"])[0]
        sim_close = np.dot(v1, v2)
        sim_far = np.dot(v1, v3)
        assert sim_close > sim_far

    def test_dims_property(self):
        emb = Model2VecEmbedder()
        assert isinstance(emb.dims, int)
        assert emb.dims > 0
```

- [ ] **Step 3: Implement base.py and model2vec_embedder.py**

Follow spec section 5. Embedder protocol with encode_queries/encode_documents. Model2VecEmbedder wraps StaticModel.

- [ ] **Step 4: Implement postprocess.py**

```python
def l2_normalize(vectors: np.ndarray) -> np.ndarray
def truncate_dims(vectors: np.ndarray, target_dims: int) -> np.ndarray
```

- [ ] **Step 5: Run tests, commit**

```bash
git commit -m "feat: Embedder interface + model2vec default embedder"
```

---

### Task 6: Schema EMBED field + memory ceiling split

**Files:**
- Modify: `graphstore/core/schema.py`
- Modify: `graphstore/core/memory.py`
- Modify: `graphstore/core/errors.py`
- Test: `tests/test_schema.py`, `tests/test_memory.py`

- [ ] **Step 1: Add embed_field to NodeKindDef**

In schema.py, add `embed_field: str | None = None` to NodeKindDef.
Update `register_node_kind` to accept `embed_field` param.
Update `to_dict`/`from_dict` for persistence.

- [ ] **Step 2: Update grammar for EMBED clause**

In grammar.lark, update `sys_register_node_kind`:
```lark
sys_register_node_kind: "REGISTER" "NODE" "KIND" STRING "REQUIRED" ident_list optional_clause? embed_clause?
embed_clause: "EMBED" IDENTIFIER
```

Update transformer + AST to pass embed_field.

- [ ] **Step 3: Add vector ceiling to memory.py**

- [ ] **Step 4: Add VectorError, EmbedderRequired to errors.py**

- [ ] **Step 5: Run tests, commit**

```bash
git commit -m "feat: EMBED clause on schema, split graph/vector ceiling, vector errors"
```

---

### Task 7: SIMILAR TO DSL + executor integration

**Files:**
- Modify: `graphstore/dsl/grammar.lark`
- Modify: `graphstore/dsl/ast_nodes.py`
- Modify: `graphstore/dsl/transformer.py`
- Modify: `graphstore/dsl/executor_reads.py`
- Modify: `graphstore/dsl/executor_writes.py`
- Modify: `graphstore/graphstore.py`
- Test: `tests/test_similar.py`

- [ ] **Step 1: Add grammar rules**

```lark
similar_q: "SIMILAR" "TO" similar_target limit_clause? where_clause?
similar_target: vector_literal      -> similar_vector
              | STRING              -> similar_text
              | "NODE" STRING       -> similar_node

vector_literal: "[" NUMBER ("," NUMBER)* "]"
vector_clause: "VECTOR" vector_literal
```

Add `similar_q` to `read_query`, `vector_clause` to `create_node`/`upsert_node`.

- [ ] **Step 2: Add AST nodes**

```python
@dataclass
class SimilarQuery:
    target_vector: list[float] | None = None
    target_text: str | None = None
    target_node_id: str | None = None
    limit: LimitClause | None = None
    where: WhereClause | None = None
```

Add `vector: list[float] | None` to CreateNode and UpsertNode.

- [ ] **Step 3: Write integration tests**

```python
from graphstore import GraphStore

class TestSimilarTo:
    def test_similar_by_text(self):
        g = GraphStore()
        g.execute('SYS REGISTER NODE KIND "memory" REQUIRED content:string EMBED content')
        g.execute('CREATE NODE "m1" kind = "memory" content = "Eiffel Tower at sunset"')
        g.execute('CREATE NODE "m2" kind = "memory" content = "Louvre museum in Paris"')
        g.execute('CREATE NODE "m3" kind = "memory" content = "quantum physics lecture"')
        result = g.execute('SIMILAR TO "Paris travel" LIMIT 2')
        ids = [n["id"] for n in result.data]
        assert "m3" not in ids  # unrelated
        assert len(result.data) <= 2

    def test_similar_by_vector(self):
        g = GraphStore()
        g.execute('CREATE NODE "m1" kind = "test" VECTOR [1.0, 0.0, 0.0, 0.0]')
        g.execute('CREATE NODE "m2" kind = "test" VECTOR [0.9, 0.1, 0.0, 0.0]')
        g.execute('CREATE NODE "m3" kind = "test" VECTOR [0.0, 0.0, 1.0, 0.0]')
        result = g.execute('SIMILAR TO [1.0, 0.0, 0.0, 0.0] LIMIT 2')
        ids = [n["id"] for n in result.data]
        assert "m1" in ids
        assert "m2" in ids

    def test_similar_with_where(self):
        g = GraphStore()
        g.execute('SYS REGISTER NODE KIND "fact" REQUIRED content:string, confidence:float EMBED content')
        g.execute('CREATE NODE "f1" kind = "fact" content = "Paris is capital" confidence = 0.9')
        g.execute('CREATE NODE "f2" kind = "fact" content = "Paris has Eiffel Tower" confidence = 0.3')
        result = g.execute('SIMILAR TO "Paris" LIMIT 10 WHERE confidence > 0.5')
        assert all(n.get("confidence", 0) > 0.5 for n in result.data)

    def test_similar_to_node(self):
        g = GraphStore()
        g.execute('CREATE NODE "q1" kind = "query" VECTOR [1.0, 0.0, 0.0, 0.0]')
        g.execute('CREATE NODE "m1" kind = "test" VECTOR [0.9, 0.1, 0.0, 0.0]')
        result = g.execute('SIMILAR TO NODE "q1" LIMIT 5')
        assert len(result.data) >= 1

    def test_similar_has_score(self):
        g = GraphStore()
        g.execute('CREATE NODE "m1" kind = "test" VECTOR [1.0, 0.0, 0.0, 0.0]')
        result = g.execute('SIMILAR TO [1.0, 0.0, 0.0, 0.0] LIMIT 5')
        assert "_similarity_score" in result.data[0]

    def test_similar_respects_retracted(self):
        g = GraphStore()
        g.execute('CREATE NODE "m1" kind = "test" VECTOR [1.0, 0.0, 0.0, 0.0]')
        g.execute('RETRACT "m1"')
        result = g.execute('SIMILAR TO [1.0, 0.0, 0.0, 0.0] LIMIT 5')
        assert len(result.data) == 0
```

- [ ] **Step 4: Implement _similar in executor_reads.py**

1. Resolve query vector (from literal, text via embedder, or node's stored vector)
2. Compute live_mask
3. Call vector_store.search(query_vec, k, mask=live_mask)
4. Post-filter with WHERE via column mask
5. Materialize, annotate with _similarity_score

- [ ] **Step 5: Implement auto-embed on create/update in executor_writes.py**

In _create_node: if schema has embed_field and embedder is available, embed and store vector.
In _update_node: if embed_field was modified, re-embed.

- [ ] **Step 6: Wire into GraphStore**

Update graphstore.py constructor to accept `embedder` param, initialize VectorStore and Embedder, pass to executor.

- [ ] **Step 7: Run tests, commit**

```bash
git commit -m "feat: SIMILAR TO query, VECTOR clause, auto-embed on create/update"
```

---

### Task 8: SYS DUPLICATES + SYS EMBEDDERS

**Files:**
- Modify: `graphstore/dsl/executor_system.py`
- Modify: `graphstore/dsl/grammar.lark`
- Modify: `graphstore/dsl/ast_nodes.py`
- Test: `tests/test_similar.py`

- [ ] **Step 1: Add grammar + AST**

```lark
sys_duplicates: "DUPLICATES" where_clause? threshold_clause?
threshold_clause: "THRESHOLD" NUMBER
sys_embedders: "EMBEDDERS"
```

- [ ] **Step 2: Implement _duplicates**

For each node with vector, find nearest neighbor. If similarity > threshold, add pair.

- [ ] **Step 3: Implement _embedders**

Return list of registered embedders with status.

- [ ] **Step 4: Run tests, commit**

```bash
git commit -m "feat: SYS DUPLICATES and SYS EMBEDDERS"
```

---

### Task 9: Vector persistence (save/load/snapshot)

**Files:**
- Modify: `graphstore/persistence/serializer.py`
- Modify: `graphstore/persistence/deserializer.py`
- Modify: `graphstore/dsl/executor_system.py` (snapshot/rollback)
- Test: `tests/test_persistence.py`

- [ ] **Step 1: Add vector index persistence to serializer**

Save vector index as blob, presence mask, dims metadata.

- [ ] **Step 2: Add vector index loading to deserializer**

Load and restore vector index from blobs.

- [ ] **Step 3: Update snapshot/rollback to include vectors**

- [ ] **Step 4: Test roundtrip**

```python
def test_vector_persistence_roundtrip(tmp_path):
    g = GraphStore(path=str(tmp_path))
    g.execute('CREATE NODE "m1" kind = "test" VECTOR [1.0, 0.0, 0.0, 0.0]')
    g.checkpoint()
    g.close()
    g2 = GraphStore(path=str(tmp_path))
    result = g2.execute('SIMILAR TO [1.0, 0.0, 0.0, 0.0] LIMIT 5')
    assert len(result.data) >= 1
```

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: vector index persistence + snapshot/rollback"
```

---

### Task 10: EmbeddingGemma ONNX adapter + CLI installer

**Files:**
- Create: `graphstore/embedding/onnx_hf_embedder.py`
- Modify: `graphstore/registry/models.py`
- Modify: `graphstore/registry/installer.py`
- Modify: `graphstore/cli.py`
- Test: `tests/test_onnx_embedder.py` (skip if deps not installed)

- [ ] **Step 1: Implement SUPPORTED_MODELS registry**

- [ ] **Step 2: Implement OnnxHFEmbedder**

Uses `tokenizers` lib for tokenization, `onnxruntime` for inference.
Handles query/document prefixes, Matryoshka truncation, L2 renorm.

- [ ] **Step 3: Implement installer.py**

```python
def install_embedder(name: str):
    # 1. Check GPU availability
    # 2. pip install onnxruntime or onnxruntime-gpu
    # 3. pip install tokenizers huggingface_hub
    # 4. Download ONNX model to ~/.graphstore/models/
    # 5. Write manifest.json
    # 6. Smoke test
```

- [ ] **Step 4: Extend cli.py**

```bash
graphstore install-embedder embeddinggemma
graphstore list-embedders
graphstore info-embedder embeddinggemma
```

- [ ] **Step 5: Write conditional tests**

```python
pytest.importorskip("onnxruntime")

class TestOnnxEmbedder:
    def test_encode(self):
        ...
```

- [ ] **Step 6: Commit**

```bash
git commit -m "feat: EmbeddingGemma ONNX adapter + CLI installer"
```

---

### Task 11: Performance benchmark + final integration

**Files:**
- Create: `bench_vectors.py`
- All test files

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest tests/ -v --tb=short
```

- [ ] **Step 2: Run vector benchmark**

Benchmark at 10k/100k/1M scales: SIMILAR TO, auto-embed, vector add, persistence.

- [ ] **Step 3: Verify performance targets from spec**

- [ ] **Step 4: Cleanup and final commit**

```bash
git commit -m "feat: vector store complete - semantic search, auto-embed, HNSW index"
```
