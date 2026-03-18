import numpy as np
from scipy.sparse import csr_matrix


def bidirectional_bfs(
    matrix: csr_matrix,
    matrix_t: csr_matrix,
    source: int,
    target: int,
    max_depth: int = 10,
) -> list[int] | None:
    """Find shortest path from source to target using bidirectional BFS.

    Returns list of node indices forming the path (inclusive of source and target),
    or None if no path exists within max_depth.
    """
    if source == target:
        return [source]

    # Forward: source -> target using matrix
    fwd_visited = {source: None}  # node -> predecessor
    fwd_frontier = [source]

    # Backward: target -> source using matrix_t
    bwd_visited = {target: None}
    bwd_frontier = [target]

    for depth in range(max_depth):
        if not fwd_frontier and not bwd_frontier:
            return None

        # Expand smaller frontier
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

        # Check intersection
        meeting = set(fwd_visited) & set(bwd_visited)
        if meeting:
            mid = min(meeting)  # deterministic choice
            return _reconstruct_path(fwd_visited, bwd_visited, mid)

    return None


def _expand_frontier(matrix, frontier, visited):
    """Expand frontier by one hop. Returns (new_frontier, updated_visited)."""
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
    """Reconstruct path from source through meeting point to target."""
    # Forward path: source -> meeting
    fwd_path = []
    node = meeting_point
    while node is not None:
        fwd_path.append(node)
        node = fwd_visited[node]
    fwd_path.reverse()

    # Backward path: meeting -> target
    bwd_path = []
    node = bwd_visited[meeting_point]
    while node is not None:
        bwd_path.append(node)
        node = bwd_visited[node]

    return fwd_path + bwd_path


def bfs_traverse(
    matrix: csr_matrix, start: int, max_depth: int
) -> list[tuple[int, int]]:
    """BFS from start up to max_depth hops.

    Returns list of (node_index, depth) pairs, including the start node at depth 0.
    """
    visited = {start: 0}
    frontier = [start]
    result = [(start, 0)]

    for depth in range(1, max_depth + 1):
        next_frontier = []
        for node in frontier:
            start_idx = matrix.indptr[node]
            end_idx = matrix.indptr[node + 1]
            neighbors = matrix.indices[start_idx:end_idx]
            for nb in neighbors:
                nb = int(nb)
                if nb not in visited:
                    visited[nb] = depth
                    next_frontier.append(nb)
                    result.append((nb, depth))
        frontier = next_frontier
        if not frontier:
            break

    return result


def find_all_paths(
    matrix: csr_matrix,
    source: int,
    target: int,
    max_depth: int,
    max_results: int = 100,
) -> list[list[int]]:
    """Find all paths from source to target up to max_depth.

    Returns list of paths. Each path is a list of node indices.
    Caps at max_results paths.
    """
    paths = []

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


def common_neighbors(matrix: csr_matrix, node_a: int, node_b: int) -> np.ndarray:
    """Find common outgoing neighbors of two nodes."""
    start_a, end_a = matrix.indptr[node_a], matrix.indptr[node_a + 1]
    start_b, end_b = matrix.indptr[node_b], matrix.indptr[node_b + 1]
    neighbors_a = matrix.indices[start_a:end_a]
    neighbors_b = matrix.indices[start_b:end_b]
    return np.intersect1d(neighbors_a, neighbors_b)
