"""Typed edge matrices backed by scipy sparse CSR."""

import numpy as np
from scipy.sparse import csr_matrix


def resize_csr(mat: csr_matrix, n: int) -> csr_matrix:
    """Safely resize a CSR matrix to (n, n) by padding indptr.

    When nodes are added after the last edge rebuild, the CSR matrix
    has fewer rows than _next_slot. Padding indptr with the last value
    (meaning "no edges for these rows") fixes the shape mismatch.
    """
    if mat.shape[0] >= n:
        return mat
    old_size = len(mat.indptr)
    needed = n + 1
    if old_size < needed:
        pad = np.full(needed - old_size, mat.indptr[-1], dtype=mat.indptr.dtype)
        new_indptr = np.concatenate([mat.indptr, pad])
    else:
        new_indptr = mat.indptr
    return csr_matrix((mat.data, mat.indices, new_indptr), shape=(n, n))


class EdgeMatrices:
    def __init__(self):
        self._typed: dict[str, csr_matrix] = {}     # per-type CSR
        self._combined_all: csr_matrix | None = None  # precomputed union
        self._cache: dict[frozenset, csr_matrix] = {} # combination cache
        self._edge_data: dict[str, list[dict]] = {}  # per-type edge data lists
        self._transpose_cache: dict[str, csr_matrix] = {} # CSC for incoming edges
        self._combined_transpose: csr_matrix | None = None  # cached combined-all transpose

        # Degree arrays - precomputed on rebuild
        self._out_degree: dict[str, np.ndarray] = {}
        self._out_degree_all: np.ndarray | None = None
        self._in_degree: dict[str, np.ndarray] = {}

    @property
    def edge_types(self) -> list[str]:
        """Return list of edge type names."""
        return list(self._typed.keys())

    @property
    def total_edges(self) -> int:
        """Total number of edges across all types."""
        return sum(m.nnz for m in self._typed.values())

    def get(self, edge_types: set[str] | None = None) -> csr_matrix | None:
        """Get CSR matrix for query. None = all types."""
        if edge_types is None:
            return self._combined_all
        if len(edge_types) == 1:
            etype = next(iter(edge_types))
            return self._typed.get(etype)
        key = frozenset(edge_types)
        if key not in self._cache:
            matrices = [self._typed[t] for t in edge_types if t in self._typed]
            if not matrices:
                return None
            self._cache[key] = sum(matrices)
        return self._cache[key]

    def get_transpose(self, edge_type: str) -> csr_matrix | None:
        """Get transposed CSR for incoming edge queries."""
        if edge_type not in self._typed:
            return None
        if edge_type not in self._transpose_cache:
            self._transpose_cache[edge_type] = self._typed[edge_type].T.tocsr()
        return self._transpose_cache[edge_type]

    def get_combined_transpose(self) -> csr_matrix | None:
        """Cached transpose of combined-all matrix. Used by RECALL spreading activation."""
        if self._combined_all is None:
            return None
        if self._combined_transpose is None:
            self._combined_transpose = self._combined_all.T.tocsr()
        return self._combined_transpose

    def get_edge_data(self, edge_type: str) -> list[dict]:
        """Get edge data for a given type."""
        return self._edge_data.get(edge_type, [])

    def out_degree(self, edge_type: str | None = None) -> np.ndarray | None:
        """Get out-degree array. None = all types combined."""
        if edge_type is None:
            return self._out_degree_all
        return self._out_degree.get(edge_type)

    def in_degree(self, edge_type: str) -> np.ndarray | None:
        """Get in-degree array for given type."""
        return self._in_degree.get(edge_type)

    def neighbors_out(self, node_idx: int, edge_type: str | None = None) -> np.ndarray:
        """Get outgoing neighbor indices for a node."""
        matrix = self.get({edge_type} if edge_type else None)
        if matrix is None:
            return np.array([], dtype=np.int32)
        start = matrix.indptr[node_idx]
        end = matrix.indptr[node_idx + 1]
        return matrix.indices[start:end].copy()

    def neighbors_in(self, node_idx: int, edge_type: str) -> np.ndarray:
        """Get incoming neighbor indices for a node."""
        t = self.get_transpose(edge_type)
        if t is None:
            return np.array([], dtype=np.int32)
        start = t.indptr[node_idx]
        end = t.indptr[node_idx + 1]
        return t.indices[start:end].copy()

    def rebuild(self, edges_by_type: dict[str, list[tuple]], num_nodes: int):
        """Full rebuild from edge lists. Called after COMMIT.

        edges_by_type: {edge_type: [(source_idx, target_idx, data_dict), ...]}
        num_nodes: total number of node slots (matrix dimension)
        """
        self._typed.clear()
        self._cache.clear()
        self._transpose_cache.clear()
        self._combined_transpose = None  # invalidate on rebuild
        self._edge_data.clear()
        self._out_degree.clear()
        self._in_degree.clear()

        for etype, edge_list in edges_by_type.items():
            if edge_list:
                sources = np.array([s for s, t, d in edge_list], dtype=np.int32)
                targets = np.array([t for s, t, d in edge_list], dtype=np.int32)
                data = [d for s, t, d in edge_list]

                # Use edge weight if available, else 1.0
                weights = np.array(
                    [d.get("weight", 1.0) if d else 1.0 for d in data],
                    dtype=np.float32
                )
                self._typed[etype] = csr_matrix(
                    (weights, (sources, targets)),
                    shape=(num_nodes, num_nodes)
                )
                self._edge_data[etype] = data

        # Precompute combined matrix
        if self._typed:
            self._combined_all = sum(self._typed.values())
        else:
            self._combined_all = None

        # Precompute degree arrays
        for etype, m in self._typed.items():
            self._out_degree[etype] = np.diff(m.indptr)
        if self._combined_all is not None:
            self._out_degree_all = np.diff(self._combined_all.indptr)
        else:
            self._out_degree_all = None

        # Precompute in-degree from transposed matrices
        for etype in self._typed:
            t = self.get_transpose(etype)
            self._in_degree[etype] = np.diff(t.indptr)
