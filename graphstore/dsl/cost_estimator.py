"""Cost estimation for MATCH and TRAVERSE queries.

Estimates frontier growth to reject queries that would explore too many
nodes before they are executed.
"""

from graphstore.dsl.ast_nodes import Condition
from graphstore.errors import CostThresholdExceeded

DEFAULT_FRONTIER_THRESHOLD = 100_000


class CostEstimate:
    def __init__(
        self,
        rejected: bool = False,
        estimated_frontier: float = 0,
        reason: str = "",
        hops: list | None = None,
    ):
        self.rejected = rejected
        self.estimated_frontier = estimated_frontier
        self.reason = reason
        self.hops = hops or []

    def to_dict(self):
        return {
            "rejected": self.rejected,
            "estimated_frontier": self.estimated_frontier,
            "reason": self.reason,
            "hops": self.hops,
        }


def estimate_match_cost(pattern, edge_matrices, threshold=DEFAULT_FRONTIER_THRESHOLD):
    """Estimate frontier growth for a MATCH pattern."""
    frontier_size = 1.0  # bound start
    hops = []

    for arrow in pattern.arrows:
        edge_type = _extract_edge_type(arrow.expr)
        matrix = edge_matrices.get({edge_type} if edge_type else None)
        if matrix is None:
            return CostEstimate(
                rejected=False,
                estimated_frontier=0,
                reason="no edges of this type",
                hops=hops,
            )

        avg_degree = matrix.nnz / max(matrix.shape[0], 1)
        frontier_size *= avg_degree
        hops.append({"edge_type": edge_type, "frontier_after": frontier_size})

        if frontier_size > threshold:
            return CostEstimate(
                rejected=True,
                estimated_frontier=frontier_size,
                reason=f"frontier exceeds {threshold} at hop {len(hops)}",
                hops=hops,
            )

    return CostEstimate(rejected=False, estimated_frontier=frontier_size, hops=hops)


def estimate_traverse_cost(
    depth, edge_matrices, edge_type=None, threshold=DEFAULT_FRONTIER_THRESHOLD
):
    """Estimate frontier growth for TRAVERSE."""
    matrix = edge_matrices.get({edge_type} if edge_type else None)
    if matrix is None:
        return CostEstimate(rejected=False, estimated_frontier=0)

    avg_degree = matrix.nnz / max(matrix.shape[0], 1)
    frontier_size = 1.0
    hops = []

    for d in range(depth):
        frontier_size *= avg_degree
        hops.append({"depth": d + 1, "frontier_after": frontier_size})
        if frontier_size > threshold:
            return CostEstimate(
                rejected=True,
                estimated_frontier=frontier_size,
                reason=f"frontier exceeds {threshold} at depth {d + 1}",
                hops=hops,
            )

    return CostEstimate(rejected=False, estimated_frontier=frontier_size, hops=hops)


def _extract_edge_type(expr):
    """Extract edge type from a filter expression like kind = 'calls'."""
    if isinstance(expr, Condition) and expr.field == "kind" and expr.op == "=":
        return expr.value
    return None
