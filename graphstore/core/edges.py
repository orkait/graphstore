"""Typed edge matrices backed by scipy sparse CSR."""

import numpy as np
from scipy.sparse import csr_matrix

from graphstore.algos.edges_ops import (
    resize_csr,  # noqa: F401 - re-export
    build_typed_csrs,
)


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
        """Full rebuild from edge lists. Delegates CSR construction to algos."""
        self._typed.clear()
        self._cache.clear()
        self._transpose_cache.clear()
        self._combined_transpose = None
        self._edge_data.clear()
        self._out_degree.clear()
        self._in_degree.clear()

        typed, data_lists = build_typed_csrs(edges_by_type, num_nodes)
        self._typed = typed
        self._edge_data = data_lists

        if self._typed:
            self._combined_all = sum(self._typed.values())
        else:
            self._combined_all = None

        for etype, m in self._typed.items():
            self._out_degree[etype] = np.diff(m.indptr)
        if self._combined_all is not None:
            self._out_degree_all = np.diff(self._combined_all.indptr)
        else:
            self._out_degree_all = None

        for etype in self._typed:
            t = self.get_transpose(etype)
            self._in_degree[etype] = np.diff(t.indptr)
