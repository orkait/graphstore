"""Memory estimator and ceiling enforcement."""

from graphstore.errors import CeilingExceeded

BYTES_PER_NODE = 330  # numpy arrays + Python dict + string table entry
BYTES_PER_EDGE = 20  # CSR entries across typed matrices
DEFAULT_CEILING_BYTES = 256 * 1_000_000  # 256MB


def estimate(node_count: int, edge_count: int) -> int:
    """Estimate memory usage in bytes."""
    return (node_count * BYTES_PER_NODE) + (edge_count * BYTES_PER_EDGE)


def check_ceiling(
    current_nodes: int,
    current_edges: int,
    added_nodes: int,
    added_edges: int,
    ceiling_bytes: int = DEFAULT_CEILING_BYTES,
) -> None:
    """Check if adding nodes/edges would exceed memory ceiling.

    Raises CeilingExceeded if projected usage exceeds ceiling.
    """
    projected = estimate(current_nodes + added_nodes, current_edges + added_edges)
    if projected > ceiling_bytes:
        raise CeilingExceeded(
            current_mb=estimate(current_nodes, current_edges) // 1_000_000,
            ceiling_mb=ceiling_bytes // 1_000_000,
            operation=f"add {added_nodes} nodes, {added_edges} edges",
        )
