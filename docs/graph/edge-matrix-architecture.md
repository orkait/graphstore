# Edge Matrix Architecture

How `EdgeMatrices` stores, queries, and updates typed graph edges using scipy CSR matrices.

## Data Model

Edges are directed, typed, and weighted:

```
source_id -> target_id  kind = "mentions"  weight = 1.0
```

Every node has a slot index (`int`). The slot-to-ID mapping lives in `CoreStore.string_table`. Edge matrices operate purely on slot indices — no string lookups during traversal.

## Three-Tier Storage (LSM-Inspired)

```
┌─────────────────────────────────────────────────┐
│  Frozen CSR (L1)                                │
│  Per-type scipy CSR matrices, immutable          │
│  Built by rebuild() from edge_list dicts         │
│  ┌───────┐ ┌───────┐ ┌───────┐                  │
│  │ calls │ │ uses  │ │mentions│                  │
│  │  CSR  │ │  CSR  │ │  CSR   │                  │
│  └───────┘ └───────┘ └───────┘                  │
│  + combined_all (union of all types)             │
│  + combined_spread (A + A.T)                     │
└─────────────────────────────────────────────────┘
                      ▲
                      │ merged on query
┌─────────────────────────────────────────────────┐
│  Dynamic Edge Buffer (L0)                       │
│  Dict-based, per-type:                          │
│    _dynamic_out[kind][src] = [tgt, ...]         │
│    _dynamic_in[kind][tgt]  = [src, ...]         │
│    _dynamic_weights[kind][(src,tgt)] = weight   │
│  O(1) writes. Merged into CSR at query time.    │
└─────────────────────────────────────────────────┘
```

**Write path**: `add_dynamic()` appends to L0 dicts. No CSR rebuild.

**Read path**: `get()` returns frozen CSR + merges delta CSR from L0 on first call, then caches the result.

**Rebuild path**: `rebuild()` flushes L0 into L1, clears all caches. Triggered by node deletion, bulk mutation, or when `_edges_dirty` is set.

## Matrix Variants

Each edge type gets a CSR matrix where `M[src, tgt] = weight`.

| Method | Returns | Use Case |
|---|---|---|
| `get()` | Forward CSR `A` | Outgoing neighbors, graph degree, match traversal |
| `get_transpose(type)` | Transpose CSR `A.T` | Incoming neighbors, ANCESTORS, in-degree |
| `get_combined()` | Union of all types `Σ A_k` | Cross-type traversal, global degree |
| `get_combined_transpose()` | Union of transposes `Σ A_k.T` | Cross-type incoming queries |
| `get_combined_spread()` | Symmetric `Σ (A_k + A_k.T)` | Spreading activation, RECALL, HybridRAG |

Each has a `*_split()` variant that returns `(frozen, delta)` separately, avoiding O(N) merge when the caller just needs both parts for parallel computation.

## Forward vs Transpose Convention

**Forward CSR** (`get()`):
```
M[src, tgt] > 0    means edge FROM src TO tgt
row_access M[src]  gives all outgoing neighbors
```

**Transpose CSR** (`get_transpose()`):
```
M.T[tgt, src] > 0  means edge FROM src TO tgt (same edge, viewed from target)
row_access M.T[tgt] gives all incoming neighbors
```

Row access in CSR is O(degree) — the `indptr` array gives the slice of `indices` in constant time. Column access would be O(nnz). This is why both orientations are stored.

## Spread Matrix (Bidirectional)

For spreading activation, flow must work regardless of edge write direction. An entity with only incoming "mentions" edges should still propagate activation back to the messages that mention it.

The spread matrix is the symmetric closure:

```
spread = A + A.T
```

For edge `msg -> ent` (weight 1.0):
```
spread[msg, ent] = 1.0   # forward
spread[ent, msg] = 1.0   # reverse (from A.T)
```

Dynamic edges follow the same pattern — each edge is written in both directions into the delta:

```python
srcs.extend((src, tgt))   # forward entry
tgts.extend((tgt, src))   # reverse entry
data_list.extend((w, w))  # same weight both ways
```

RECALL and HybridRAG both use `get_combined_spread_split()`, never raw `get()` or `get_transpose()`.

## Cache Hierarchy

8 cache dicts manage the 3^2 = 9 possible combinations (3 base matrices x 3 subset types: single, frozenset, all):

| Cache | Key | Value |
|---|---|---|
| `_typed` | edge type string | Frozen per-type CSR |
| `_cache` | frozenset(types) | Frozen subset union CSR |
| `_combined_all` | N/A (single) | Frozen union of all types |
| `_transpose_cache` | edge type string | Frozen per-type transpose CSR |
| `_combined_transpose` | N/A (single) | Frozen union of all transposes |
| `_combined_spread` | N/A (single) | Frozen symmetric closure |
| `_dynamic_cache` | None / string / frozenset | Frozen + Delta merged |
| `_dynamic_transpose_cache` | edge type string | Frozen transpose + Delta transpose merged |
| `_dynamic_combined_transpose` | N/A | All transposes + all deltas merged |
| `_dynamic_combined_spread` | N/A | All spreads + all deltas merged |

**Invalidation**:

| Event | Cleared |
|---|---|
| `add_dynamic()` | `_dynamic_cache`, `_dynamic_combined_transpose`, `_dynamic_combined_spread`, `_dynamic_transpose_cache` |
| `rebuild()` | All caches, all frozen matrices, all L0 dicts |
| Node mutation in CoreStore | Sets `_edges_dirty` in CoreStore, triggers rebuild on next `edge_matrices` access |

Frozen caches are **never** cleared by L0 writes — the frozen matrices stay valid. Only the delta/merged caches are invalidated.

## Degree Arrays

```python
def out_degree(type=None):       # np.diff(M.indptr) — O(1) after CSR exists
def in_degree(type):             # np.diff(M.T.indptr) — O(1) after transpose exists
```

Used by the graph signal in REMEMBER fusion (signal 4: normalized degree).

## Neighbor Queries

`neighbors_out(node, type)` and `neighbors_in(node, type)` bypass the CSR and read directly from L0 dicts when dynamic edges exist. This avoids building a delta CSR just to look up one row.

```
neighbors_out(42, "calls"):
  1. Read frozen CSR row: indices[indptr[42]:indptr[43]]
  2. Read L0 _dynamic_out["calls"][42]
  3. Concatenate
```

## Rebuild Pipeline

```
CoreStore._edges_by_type: {"calls": [(0,3,{}), (1,4,{}), ...], ...}
        │
        ▼
build_typed_csrs(edges_by_type, num_nodes)
        │
        ├── sources[]  = [0, 1, ...]
        ├── targets[]  = [3, 4, ...]
        └── weights[]  = [1.0, 1.0, ...]
        │
        ▼
csr_matrix((weights, (sources, targets)), shape=(n, n))
        │
        ├── _typed["calls"] = CSR
        ├── _combined_all   = sum(_typed.values())
        ├── _out_degree     = np.diff(indptr)
        └── _in_degree      = np.diff(T.indptr)  [computed on demand]
```

The `build_typed_csrs()` function lives in `algos/edges_ops.py` — a pure function that takes a dict and returns CSRs. `EdgeMatrices.rebuild()` calls it and populates all caches.

## Memory Footprint

For `E` edges across `N` nodes:

| Component | Size |
|---|---|
| CSR (indices + data) | `E * 4 + E * 4` bytes (int32 indices + float32 data) |
| CSR (indptr) | `N * 8` bytes (int64, one per row + 1) |
| Per-type overhead | `3 * (E * 8 + E * 4 + N * 8)` (indices + data + indptr) |
| L0 dicts | `E * (3 * 8 + dict_overhead)` (src, tgt, weight entries) |

Typical: 100k edges, 50k nodes, 5 edge types ≈ 3-5 MB total.
