"""Memory accounting: real component measurement + ceiling enforcement."""

import sys
from graphstore.core.errors import CeilingExceeded

DEFAULT_CEILING_BYTES = 256 * 1_000_000  # 256MB

# Rough per-item overhead for estimation when real measurement is unavailable
BYTES_PER_NODE_ESTIMATE = 330
BYTES_PER_EDGE_ESTIMATE = 20

# Backwards-compatible aliases
BYTES_PER_NODE = BYTES_PER_NODE_ESTIMATE
BYTES_PER_EDGE = BYTES_PER_EDGE_ESTIMATE


def estimate(node_count: int, edge_count: int) -> int:
    """Quick estimate for backwards compat. Prefer measure() for accuracy."""
    return (node_count * BYTES_PER_NODE_ESTIMATE) + (edge_count * BYTES_PER_EDGE_ESTIMATE)


def measure(store, vector_store=None, document_store=None) -> dict:
    """Measure actual memory usage of all components. Returns detailed breakdown."""
    report = {}

    # Node arrays
    node_arrays = store.node_ids.nbytes + store.node_kinds.nbytes
    report["node_arrays"] = node_arrays

    # Column store
    col_bytes = store.columns.memory_bytes
    report["columns"] = col_bytes

    # String table (both the list and the dict)
    st = store.string_table
    str_list_size = sys.getsizeof(st._id_to_str)
    str_dict_size = sys.getsizeof(st._str_to_id)
    # Add estimated string content (average ~20 bytes per string)
    str_content = len(st) * 20
    report["string_table"] = str_list_size + str_dict_size + str_content

    # Edge lists (Python objects)
    edge_list_bytes = 0
    for etype, edges in store._edges_by_type.items():
        edge_list_bytes += sys.getsizeof(edges)
        # Each tuple (int, int, dict) ~100 bytes
        edge_list_bytes += len(edges) * 100
    report["edge_lists"] = edge_list_bytes

    # Edge data index
    edge_idx_bytes = 0
    for etype, idx in store._edge_data_idx.items():
        edge_idx_bytes += sys.getsizeof(idx)
        edge_idx_bytes += len(idx) * 80  # dict entry overhead
    report["edge_data_idx"] = edge_idx_bytes

    # CSR matrices
    csr_bytes = 0
    if hasattr(store, '_edge_matrices'):
        store._ensure_edges_built()
        for etype, m in store._edge_matrices._typed.items():
            csr_bytes += m.data.nbytes + m.indices.nbytes + m.indptr.nbytes
        if store._edge_matrices._combined_all is not None:
            m = store._edge_matrices._combined_all
            csr_bytes += m.data.nbytes + m.indices.nbytes + m.indptr.nbytes
    report["edge_matrices"] = csr_bytes

    # Edge keys set
    report["edge_keys"] = sys.getsizeof(store._edge_keys) + len(store._edge_keys) * 80

    # Tombstone set
    report["tombstones"] = sys.getsizeof(store.node_tombstones) + len(store.node_tombstones) * 28

    # id_to_slot dict
    report["id_to_slot"] = sys.getsizeof(store.id_to_slot) + len(store.id_to_slot) * 40

    # Vector store
    if vector_store is not None:
        report["vector_store"] = vector_store.memory_bytes
    else:
        report["vector_store"] = 0

    # Total core
    core_total = sum(report.values())
    report["core_total"] = core_total

    # Document store is on disk (SQLite), doesn't count toward RAM
    report["document_store_disk"] = 0
    if document_store is not None:
        try:
            stats = document_store.stats()
            report["document_store_disk"] = stats.get("total_bytes", 0)
        except Exception:
            pass

    report["total"] = core_total

    return report


def check_ceiling(
    current_nodes: int,
    current_edges: int,
    added_nodes: int,
    added_edges: int,
    ceiling_bytes: int = DEFAULT_CEILING_BYTES,
) -> None:
    """Check if adding nodes/edges would exceed memory ceiling.

    Uses quick estimate for hot-path performance. Accurate measurement
    happens in SYS HEALTH and auto-optimize health checks.
    """
    projected = estimate(current_nodes + added_nodes, current_edges + added_edges)
    if projected > ceiling_bytes:
        raise CeilingExceeded(
            current_mb=estimate(current_nodes, current_edges) // 1_000_000,
            ceiling_mb=ceiling_bytes // 1_000_000,
            operation=f"add {added_nodes} nodes, {added_edges} edges",
        )


def check_ceiling_accurate(store, vector_store, ceiling_bytes: int) -> bool:
    """Accurate ceiling check. Returns True if over ceiling."""
    report = measure(store, vector_store)
    return report["total"] > ceiling_bytes
