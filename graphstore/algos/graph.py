"""Graph algorithms on scipy CSR matrices.

All functions operate on immutable matrix inputs. No graphstore imports.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as _csgraph_dijkstra

__all__ = [
    "bfs_traverse",
    "bidirectional_bfs",
    "find_all_paths",
    "dijkstra",
    "common_neighbors",
]


def bidirectional_bfs(
    matrix: csr_matrix,
    matrix_t: csr_matrix,
    source: int,
    target: int,
    max_depth: int = 10,
) -> list[int] | None:
    """Shortest unweighted path via bidirectional BFS.

    Returns list of node indices including source and target, or None.
    """
    if source == target:
        return [source]

    fwd_visited = {source: None}
    fwd_frontier = [source]
    bwd_visited = {target: None}
    bwd_frontier = [target]

    for _ in range(max_depth):
        if not fwd_frontier and not bwd_frontier:
            return None

        if len(fwd_frontier) <= len(bwd_frontier):
            if fwd_frontier:
                fwd_frontier, fwd_visited = _expand_frontier(
                    matrix, fwd_frontier, fwd_visited
                )
        else:
            if bwd_frontier:
                bwd_frontier, bwd_visited = _expand_frontier(
                    matrix_t, bwd_frontier, bwd_visited
                )

        meeting = set(fwd_visited) & set(bwd_visited)
        if meeting:
            mid = min(meeting)
            return _reconstruct_path(fwd_visited, bwd_visited, mid)

    return None


def _expand_frontier(matrix, frontier, visited):
    new_frontier = []
    for node in frontier:
        start = matrix.indptr[node]
        end = matrix.indptr[node + 1]
        neighbors = matrix.indices[start:end]
        for nb in neighbors:
            nb = int(nb)
            if nb not in visited:
                visited[nb] = node
                new_frontier.append(nb)
    return new_frontier, visited


def _reconstruct_path(fwd_visited, bwd_visited, meeting_point):
    fwd_path = []
    node = meeting_point
    while node is not None:
        fwd_path.append(node)
        node = fwd_visited[node]
    fwd_path.reverse()

    bwd_path = []
    node = bwd_visited[meeting_point]
    while node is not None:
        bwd_path.append(node)
        node = bwd_visited[node]

    return fwd_path + bwd_path


def bfs_traverse(
    matrix: csr_matrix, start: int, max_depth: int
) -> list[tuple[int, int]]:
    """BFS from start up to max_depth via sparse matrix-power.

    One CSR.T @ bool matvec per hop — O(nnz) per level, vectorized.
    Returns (node_index, depth) pairs including (start, 0).
    """
    n = matrix.shape[0]
    result = [(start, 0)]
    if n == 0 or start >= n or max_depth <= 0:
        return result

    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    frontier = np.zeros(n, dtype=np.float32)
    frontier[start] = 1.0
    matT = matrix.T.tocsr()

    for depth in range(1, max_depth + 1):
        reached = matT.dot(frontier) > 0
        next_mask = reached & ~visited
        if not next_mask.any():
            break
        visited |= next_mask
        new_slots = np.nonzero(next_mask)[0]
        d_tuple = int(depth)
        result.extend((int(s), d_tuple) for s in new_slots)
        frontier = next_mask.astype(np.float32)

    return result


def find_all_paths(
    matrix: csr_matrix,
    source: int,
    target: int,
    max_depth: int,
    max_results: int = 100,
) -> list[list[int]]:
    """Enumerate all simple paths from source to target up to max_depth.

    Caps at max_results paths. Pure recursive DFS.
    """
    paths: list[list[int]] = []

    def dfs(node, path, visited_set, depth):
        if len(paths) >= max_results:
            return
        if node == target:
            paths.append(list(path))
            return
        if depth >= max_depth:
            return

        start_idx = matrix.indptr[node]
        end_idx = matrix.indptr[node + 1]
        neighbors = matrix.indices[start_idx:end_idx]

        for nb in neighbors:
            nb = int(nb)
            if nb not in visited_set:
                path.append(nb)
                visited_set.add(nb)
                dfs(nb, path, visited_set, depth + 1)
                path.pop()
                visited_set.discard(nb)

    dfs(source, [source], {source}, 0)
    return paths


def dijkstra(
    matrix: csr_matrix,
    source: int,
    target: int,
    max_cost: float = float('inf'),
) -> tuple[list[int] | None, float]:
    """Weighted shortest path via scipy.sparse.csgraph.dijkstra (native C)."""
    if source == target:
        return [source], 0.0

    limit = max_cost if max_cost != float('inf') else np.inf
    dist_arr, predecessors = _csgraph_dijkstra(
        matrix,
        indices=source,
        return_predecessors=True,
        limit=limit,
        directed=True,
    )
    total = float(dist_arr[target])
    if not np.isfinite(total):
        return None, float('inf')

    path: list[int] = []
    node = int(target)
    while node != source:
        path.append(node)
        node = int(predecessors[node])
        if node == -9999:
            return None, float('inf')
    path.append(source)
    path.reverse()
    return path, total


def common_neighbors(matrix: csr_matrix, node_a: int, node_b: int) -> np.ndarray:
    """Outgoing neighbors shared between two nodes."""
    start_a, end_a = matrix.indptr[node_a], matrix.indptr[node_a + 1]
    start_b, end_b = matrix.indptr[node_b], matrix.indptr[node_b + 1]
    neighbors_a = matrix.indices[start_a:end_a]
    neighbors_b = matrix.indices[start_b:end_b]
    return np.intersect1d(neighbors_a, neighbors_b)
