"""Read query handlers for the DSL executor."""

import time
from collections import deque

import numpy as np
from scipy.sparse import csr_matrix

from graphstore.dsl.ast_nodes import (
    AggFunc,
    AggregateQuery,
    AncestorsQuery,
    AndExpr,
    CommonNeighborsQuery,
    Condition,
    CountQuery,
    CounterfactualQuery,
    DescendantsQuery,
    DistanceQuery,
    EdgesQuery,
    MatchPattern,
    MatchQuery,
    NodeQuery,
    NodesQuery,
    PathQuery,
    PathsQuery,
    RecallQuery,
    ShortestPathQuery,
    SimilarQuery,
    SubgraphQuery,
    TraverseQuery,
    WeightedDistanceQuery,
    WeightedShortestPathQuery,
)
from graphstore.core.errors import (
    AggregationError,
    CostThresholdExceeded,
    EmbedderRequired,
    NodeNotFound,
    VectorError,
    VectorNotFound,
)
from graphstore.core.path import (
    bfs_traverse,
    bidirectional_bfs,
    common_neighbors,
    dijkstra,
    find_all_paths,
)
from graphstore.core.types import Result
from graphstore.dsl.cost_estimator import estimate_match_cost, estimate_traverse_cost
from graphstore.dsl.executor_base import ExecutorBase


class ReadExecutor(ExecutorBase):

    def _node(self, q: NodeQuery) -> Result:
        data = self.store.get_node(q.id)
        if data is not None:
            # Check if retracted or expired
            slot = self._resolve_slot(q.id)
            if slot is not None:
                if not self._is_slot_visible(slot):
                    data = None
        return Result(kind="node", data=data, count=1 if data else 0)

    def _nodes(self, q: NodesQuery) -> Result:
        kind_filter = self._extract_kind_from_where(q.where) if q.where else None

        # Try secondary index fast path for simple equality on indexed field
        nodes = self._try_index_lookup(q.where, kind_filter) if q.where else None

        if nodes is not None:
            # Index path returned materialized nodes - apply visibility + order + limit
            nodes = self._filter_visible(nodes)
            if q.order:
                reverse = q.order.direction == "DESC"
                col_sorted = self._try_column_order_by(
                    nodes, q.order.field, reverse,
                    q.limit.value if q.limit else None,
                    q.offset.value if q.offset else None,
                )
                if col_sorted is not None:
                    nodes = col_sorted
                else:
                    nodes.sort(
                        key=lambda n: (n.get(q.order.field) is None, n.get(q.order.field, "")),
                        reverse=reverse,
                    )
                    if q.offset:
                        nodes = nodes[q.offset.value:]
                    if q.limit:
                        nodes = nodes[:q.limit.value]
            else:
                if q.offset:
                    nodes = nodes[q.offset.value:]
                if q.limit:
                    nodes = nodes[:q.limit.value]
            return Result(kind="nodes", data=nodes, count=len(nodes))

        # === Slot-mask path: compute mask, apply LIMIT before materialization ===
        n = self.store._next_slot
        if n == 0:
            return Result(kind="nodes", data=[], count=0)

        # Start with full live mask (tombstones + TTL + retracted + context)
        final_mask = self._compute_live_mask(n)

        # Apply kind filter via numpy
        if kind_filter:
            kind_mask = self.store._live_mask(kind_filter)
            # Intersect with live mask (kind_mask only filters tombstones+kind,
            # final_mask also filters TTL/retracted/context)
            final_mask = final_mask & kind_mask

        # Apply WHERE column/predicate filter
        fallback_predicate = None
        if q.where and not self._is_simple_kind_filter(q.where):
            remaining = self._strip_kind_from_expr(q.where.expr)
            if remaining is not None:
                col_mask = self._try_column_filter(remaining, final_mask, n)
                if col_mask is not None:
                    final_mask = col_mask
                else:
                    # Try raw predicate for slot-level filtering
                    raw_pred = self._make_raw_predicate(remaining)
                    if raw_pred is not None:
                        # Apply predicate as a post-materialization filter
                        fallback_predicate = lambda node, _expr=remaining: self._eval_where(_expr, node)
                    else:
                        # Full expression eval needed
                        fallback_predicate = lambda node, _expr=q.where.expr: self._eval_where(_expr, node)

        # Get matching slots
        slots = np.where(final_mask)[0]

        # Apply ORDER BY on column data (not materialized dicts)
        if q.order:
            reverse = q.order.direction == "DESC"
            slots = self._order_slots_by_column(
                slots, q.order.field, reverse,
                q.limit.value if q.limit else None,
                q.offset.value if q.offset else None,
                fallback_predicate,
            )
            if slots is not None:
                # Slots already ordered and sliced; materialize
                result = []
                for s in slots:
                    node = self.store._materialize_slot(int(s))
                    if node is not None:
                        if fallback_predicate and not fallback_predicate(node):
                            continue
                        result.append(node)
                return Result(kind="nodes", data=result, count=len(result))
            else:
                # Column sort not possible - materialize, filter, sort in Python
                nodes = self._materialize_slots_filtered(slots, fallback_predicate)
                nodes.sort(
                    key=lambda nd: (nd.get(q.order.field) is None, nd.get(q.order.field, "")),
                    reverse=reverse,
                )
                if q.offset:
                    nodes = nodes[q.offset.value:]
                if q.limit:
                    nodes = nodes[:q.limit.value]
                return Result(kind="nodes", data=nodes, count=len(nodes))

        # No ORDER BY: apply OFFSET + LIMIT to slot array BEFORE materializing
        if fallback_predicate is None:
            if q.offset:
                slots = slots[q.offset.value:]
            if q.limit:
                slots = slots[:q.limit.value]
            result = []
            for s in slots:
                node = self.store._materialize_slot(int(s))
                if node is not None:
                    result.append(node)
        else:
            # With fallback predicate, must materialize+filter then slice
            result = self._materialize_slots_filtered(slots, fallback_predicate)
            if q.offset:
                result = result[q.offset.value:]
            if q.limit:
                result = result[:q.limit.value]

        return Result(kind="nodes", data=result, count=len(result))

    def _edges(self, q: EdgesQuery) -> Result:
        kind = self._extract_kind_from_where(q.where)
        if q.direction == "FROM":
            edges = self.store.get_edges_from(q.node_id, kind=kind)
        else:
            edges = self.store.get_edges_to(q.node_id, kind=kind)
        # Apply remaining filters if where has more than just kind
        if q.where and not self._is_simple_kind_filter(q.where):
            edges = [e for e in edges if self._eval_where(q.where.expr, e)]
        # Filter edges touching non-visible nodes
        edges = [
            e for e in edges
            if self._is_visible_by_id(e["source"]) and self._is_visible_by_id(e["target"])
        ]
        if q.limit:
            edges = edges[:q.limit.value]
        return Result(kind="edges", data=edges, count=len(edges))

    def _traverse(self, q: TraverseQuery) -> Result:
        slot = self._resolve_slot(q.start_id)
        if slot is None:
            return Result(kind="nodes", data=[], count=0)

        edge_type = self._extract_kind_from_where(q.where)
        matrix = self.store.edge_matrices.get({edge_type} if edge_type else None)
        if matrix is None:
            node_data = self.store.get_node(q.start_id)
            return Result(
                kind="nodes",
                data=[node_data] if node_data else [],
                count=1 if node_data else 0,
            )

        # Cost check
        cost = estimate_traverse_cost(q.depth, self.store.edge_matrices, edge_type, threshold=self.cost_threshold)
        if cost.rejected:
            raise CostThresholdExceeded(cost.estimated_frontier, self.cost_threshold)

        visited = bfs_traverse(matrix, slot, q.depth)
        nodes = []
        for node_idx, depth in visited:
            nid = self.store._slot_to_id(node_idx)
            if nid:
                node = self.store.get_node(nid)
                if node:
                    node["_depth"] = depth
                    nodes.append(node)
        nodes = self._filter_visible(nodes)
        if q.limit:
            nodes = nodes[:q.limit.value]
        return Result(kind="nodes", data=nodes, count=len(nodes))

    def _subgraph(self, q: SubgraphQuery) -> Result:
        slot = self._resolve_slot(q.start_id)
        if slot is None:
            return Result(
                kind="subgraph", data={"nodes": [], "edges": []}, count=0
            )

        matrix = self.store.edge_matrices.get(None)
        if matrix is None:
            node_data = self.store.get_node(q.start_id)
            return Result(
                kind="subgraph",
                data={
                    "nodes": [node_data] if node_data else [],
                    "edges": [],
                },
                count=1 if node_data else 0,
            )

        visited = bfs_traverse(matrix, slot, q.depth)
        visited_slots = {node_idx for node_idx, _ in visited}

        nodes = []
        for node_idx, depth in visited:
            nid = self.store._slot_to_id(node_idx)
            if nid:
                node = self.store.get_node(nid)
                if node:
                    nodes.append(node)

        # Filter out retracted/expired nodes
        nodes = self._filter_visible(nodes)
        visible_ids = {n["id"] for n in nodes}

        # Collect edges between visible nodes
        edges = []
        for node_idx in visited_slots:
            nid = self.store._slot_to_id(node_idx)
            if nid and nid in visible_ids:
                for e in self.store.get_edges_from(nid):
                    tgt_id = e["target"]
                    if tgt_id in visible_ids:
                        edges.append(e)

        return Result(
            kind="subgraph",
            data={"nodes": nodes, "edges": edges},
            count=len(nodes),
        )

    def _path(self, q: PathQuery) -> Result:
        src_slot = self._resolve_slot(q.from_id)
        tgt_slot = self._resolve_slot(q.to_id)
        if src_slot is None or tgt_slot is None:
            return Result(kind="path", data=None, count=0)

        edge_type = self._extract_kind_from_where(q.where)
        matrix = self.store.edge_matrices.get({edge_type} if edge_type else None)
        if matrix is None:
            return Result(kind="path", data=None, count=0)
        matrix_t = matrix.T.tocsr()

        path = bidirectional_bfs(matrix, matrix_t, src_slot, tgt_slot, q.max_depth)
        if path is None:
            return Result(kind="path", data=None, count=0)

        path_ids = [self.store._slot_to_id(s) for s in path]
        # Validate all nodes in path are visible
        if any(pid is None or not self._is_visible_by_id(pid) for pid in path_ids):
            return Result(kind="path", data=None, count=0)
        return Result(kind="path", data=path_ids, count=len(path_ids))

    def _paths(self, q: PathsQuery) -> Result:
        src_slot = self._resolve_slot(q.from_id)
        tgt_slot = self._resolve_slot(q.to_id)
        if src_slot is None or tgt_slot is None:
            return Result(kind="paths", data=[], count=0)

        edge_type = self._extract_kind_from_where(q.where)
        matrix = self.store.edge_matrices.get({edge_type} if edge_type else None)
        if matrix is None:
            return Result(kind="paths", data=[], count=0)

        paths = find_all_paths(matrix, src_slot, tgt_slot, q.max_depth)
        result = []
        for path in paths:
            path_ids = [self.store._slot_to_id(s) for s in path]
            # Only include paths where all nodes are visible
            if all(pid is not None and self._is_visible_by_id(pid) for pid in path_ids):
                result.append(path_ids)
        return Result(kind="paths", data=result, count=len(result))

    def _shortest_path(self, q: ShortestPathQuery) -> Result:
        src_slot = self._resolve_slot(q.from_id)
        tgt_slot = self._resolve_slot(q.to_id)
        if src_slot is None or tgt_slot is None:
            return Result(kind="path", data=None, count=0)

        edge_type = self._extract_kind_from_where(q.where)
        matrix = self.store.edge_matrices.get({edge_type} if edge_type else None)
        if matrix is None:
            return Result(kind="path", data=None, count=0)
        matrix_t = matrix.T.tocsr()

        path = bidirectional_bfs(matrix, matrix_t, src_slot, tgt_slot)
        if path is None:
            return Result(kind="path", data=None, count=0)

        path_ids = [self.store._slot_to_id(s) for s in path]
        # Validate all nodes in path are visible
        if any(pid is None or not self._is_visible_by_id(pid) for pid in path_ids):
            return Result(kind="path", data=None, count=0)
        return Result(kind="path", data=path_ids, count=len(path_ids))

    def _distance(self, q: DistanceQuery) -> Result:
        src_slot = self._resolve_slot(q.from_id)
        tgt_slot = self._resolve_slot(q.to_id)
        if src_slot is None or tgt_slot is None:
            return Result(kind="distance", data=-1, count=1)

        matrix = self.store.edge_matrices.get(None)
        if matrix is None:
            if src_slot == tgt_slot:
                return Result(kind="distance", data=0, count=1)
            return Result(kind="distance", data=-1, count=1)
        matrix_t = matrix.T.tocsr()

        path = bidirectional_bfs(matrix, matrix_t, src_slot, tgt_slot, q.max_depth)
        if path is not None:
            # Validate all nodes in path are visible
            path_ids = [self.store._slot_to_id(s) for s in path]
            if any(pid is None or not self._is_visible_by_id(pid) for pid in path_ids):
                path = None
        dist = len(path) - 1 if path else -1
        return Result(kind="distance", data=dist, count=1)

    def _weighted_shortest_path(self, q: WeightedShortestPathQuery) -> Result:
        src_slot = self._resolve_slot(q.from_id)
        tgt_slot = self._resolve_slot(q.to_id)
        if src_slot is None or tgt_slot is None:
            return Result(kind="path", data=None, count=0)

        edge_type = self._extract_kind_from_where(q.where)
        matrix = self.store.edge_matrices.get({edge_type} if edge_type else None)
        if matrix is None:
            return Result(kind="path", data=None, count=0)

        path, cost = dijkstra(matrix, src_slot, tgt_slot)
        if path is None:
            return Result(kind="path", data=None, count=0)

        path_ids = [self.store._slot_to_id(s) for s in path]
        # Validate all nodes in path are visible
        if any(pid is None or not self._is_visible_by_id(pid) for pid in path_ids):
            return Result(kind="path", data=None, count=0)
        return Result(kind="path", data=path_ids, count=len(path_ids))

    def _weighted_distance(self, q: WeightedDistanceQuery) -> Result:
        src_slot = self._resolve_slot(q.from_id)
        tgt_slot = self._resolve_slot(q.to_id)
        if src_slot is None or tgt_slot is None:
            return Result(kind="distance", data=-1, count=1)

        matrix = self.store.edge_matrices.get(None)
        if matrix is None:
            if src_slot == tgt_slot:
                return Result(kind="distance", data=0.0, count=1)
            return Result(kind="distance", data=-1, count=1)

        path, cost = dijkstra(matrix, src_slot, tgt_slot)
        if path is not None:
            # Validate all nodes in path are visible
            path_ids = [self.store._slot_to_id(s) for s in path]
            if any(pid is None or not self._is_visible_by_id(pid) for pid in path_ids):
                path = None
        if path is None:
            return Result(kind="distance", data=-1, count=1)
        return Result(kind="distance", data=cost, count=1)

    def _ancestors(self, q: AncestorsQuery) -> Result:
        slot = self._resolve_slot(q.node_id)
        if slot is None:
            return Result(kind="subgraph", data={"nodes": [], "edges": []}, count=0)

        edge_type = self._extract_kind_from_where(q.where)
        # Ancestors = BFS on transposed matrix (follow incoming edges)
        if edge_type:
            matrix = self.store.edge_matrices.get_transpose(edge_type)
        else:
            matrix = self.store.edge_matrices.get(None)
            if matrix is not None:
                matrix = matrix.T.tocsr()

        if matrix is None:
            node_data = self.store.get_node(q.node_id)
            return Result(
                kind="subgraph",
                data={"nodes": [{**node_data, "_query_anchor": True}] if node_data else [], "edges": []},
                count=0,
            )

        visited = bfs_traverse(matrix, slot, q.depth)
        visited_slots = {node_idx for node_idx, _ in visited}

        nodes = []
        for node_idx, depth in visited:
            nid = self.store._slot_to_id(node_idx)
            if nid:
                node = self.store.get_node(nid)
                if node:
                    node["_depth"] = depth
                    if node_idx == slot:
                        node["_query_anchor"] = True
                    nodes.append(node)

        # Filter out retracted/expired nodes
        nodes = self._filter_visible(nodes)
        visible_ids = {n["id"] for n in nodes}

        # Collect forward edges between visible nodes (incl. anchor)
        edges = []
        for node_idx in visited_slots:
            nid = self.store._slot_to_id(node_idx)
            if nid and nid in visible_ids:
                for e in self.store.get_edges_from(nid, kind=edge_type):
                    if e["target"] in visible_ids:
                        edges.append(e)

        ancestor_count = sum(1 for n in nodes if not n.get("_query_anchor"))
        return Result(kind="subgraph", data={"nodes": nodes, "edges": edges}, count=ancestor_count)

    def _descendants(self, q: DescendantsQuery) -> Result:
        slot = self._resolve_slot(q.node_id)
        if slot is None:
            return Result(kind="subgraph", data={"nodes": [], "edges": []}, count=0)

        edge_type = self._extract_kind_from_where(q.where)
        # Descendants = BFS on forward matrix (follow outgoing edges)
        matrix = self.store.edge_matrices.get({edge_type} if edge_type else None)
        if matrix is None:
            node_data = self.store.get_node(q.node_id)
            return Result(
                kind="subgraph",
                data={"nodes": [{**node_data, "_query_anchor": True}] if node_data else [], "edges": []},
                count=0,
            )

        visited = bfs_traverse(matrix, slot, q.depth)
        visited_slots = {node_idx for node_idx, _ in visited}

        nodes = []
        for node_idx, depth in visited:
            nid = self.store._slot_to_id(node_idx)
            if nid:
                node = self.store.get_node(nid)
                if node:
                    node["_depth"] = depth
                    if node_idx == slot:
                        node["_query_anchor"] = True
                    nodes.append(node)

        # Filter out retracted/expired nodes
        nodes = self._filter_visible(nodes)
        visible_ids = {n["id"] for n in nodes}

        edges = []
        for node_idx in visited_slots:
            nid = self.store._slot_to_id(node_idx)
            if nid and nid in visible_ids:
                for e in self.store.get_edges_from(nid, kind=edge_type):
                    if e["target"] in visible_ids:
                        edges.append(e)

        descendant_count = sum(1 for n in nodes if not n.get("_query_anchor"))
        return Result(kind="subgraph", data={"nodes": nodes, "edges": edges}, count=descendant_count)

    def _common_neighbors(self, q: CommonNeighborsQuery) -> Result:
        slot_a = self._resolve_slot(q.node_a)
        slot_b = self._resolve_slot(q.node_b)
        if slot_a is None or slot_b is None:
            return Result(kind="nodes", data=[], count=0)

        edge_type = self._extract_kind_from_where(q.where)
        matrix = self.store.edge_matrices.get({edge_type} if edge_type else None)
        if matrix is None:
            return Result(kind="nodes", data=[], count=0)

        shared = common_neighbors(matrix, slot_a, slot_b)
        nodes = []
        for idx in shared:
            nid = self.store._slot_to_id(int(idx))
            if nid:
                node = self.store.get_node(nid)
                if node:
                    nodes.append(node)
        nodes = self._filter_visible(nodes)
        return Result(kind="nodes", data=nodes, count=len(nodes))

    def _match(self, q: MatchQuery) -> Result:
        pattern = q.pattern

        # Cost check
        cost = estimate_match_cost(pattern, self.store.edge_matrices, threshold=self.cost_threshold)
        if cost.rejected:
            raise CostThresholdExceeded(cost.estimated_frontier, self.cost_threshold)

        # Execute match pattern using sparse matrix traversal
        bindings, edges = self._execute_match_pattern(pattern)

        # Filter out bindings that reference non-visible nodes
        bindings = [
            b for b in bindings
            if all(self._is_visible_by_id(nid) for nid in b.values())
        ]
        # Filter edges that reference non-visible nodes
        edges = [
            e for e in edges
            if self._is_visible_by_id(e["source"]) and self._is_visible_by_id(e["target"])
        ]

        if q.limit:
            bindings = bindings[: q.limit.value]

        return Result(
            kind="match",
            data={"bindings": bindings, "edges": edges},
            count=len(bindings),
        )

    def _execute_match_pattern(self, pattern: MatchPattern) -> tuple[list[dict], list[dict]]:
        """Execute a MATCH pattern. Returns (bindings, edges)."""
        steps = pattern.steps
        arrows = pattern.arrows

        # Get starting nodes
        first_step = steps[0]
        if first_step.bound_id:
            start_slot = self._resolve_slot(first_step.bound_id)
            if start_slot is None:
                return [], []
            current_slots = [start_slot]
        else:
            # Unbound start - get all nodes matching filter
            all_nodes = self.store.get_all_nodes()
            if first_step.where:
                all_nodes = [
                    n for n in all_nodes if self._eval_where(first_step.where, n)
                ]
            current_slots = []
            for n in all_nodes:
                s = self._resolve_slot(n["id"])
                if s is not None:
                    current_slots.append(s)

        if not current_slots:
            return [], []

        # Track bindings and edges for each path
        paths = [[] for _ in current_slots]
        edge_trails = [[] for _ in current_slots]

        # Add first step bindings
        for i, slot in enumerate(current_slots):
            if first_step.variable:
                nid = self.store._slot_to_id(slot)
                paths[i].append((first_step.variable, nid))
            elif first_step.bound_id:
                paths[i].append(("_start", first_step.bound_id))

        # Process each hop
        for arrow, next_step in zip(arrows, steps[1:]):
            edge_type = self._extract_edge_type_from_expr(arrow.expr)
            new_paths = []
            new_slots = []
            new_edge_trails = []

            for i, slot in enumerate(current_slots):
                source_nid = self.store._slot_to_id(slot)
                neighbors = self.store.edge_matrices.neighbors_out(slot, edge_type)

                for nb in neighbors:
                    nb = int(nb)
                    nid = self.store._slot_to_id(nb)
                    if nid is None:
                        continue

                    if next_step.bound_id:
                        if nid != next_step.bound_id:
                            continue

                    if next_step.where:
                        node_data = self.store.get_node(nid)
                        if not node_data or not self._eval_where(
                            next_step.where, node_data
                        ):
                            continue

                    new_path = list(paths[i])
                    if next_step.variable:
                        new_path.append((next_step.variable, nid))
                    elif next_step.bound_id:
                        new_path.append(("_bound", nid))

                    new_edge_trail = list(edge_trails[i]) + [
                        {"source": source_nid, "target": nid, "kind": edge_type or ""}
                    ]

                    new_paths.append(new_path)
                    new_slots.append(nb)
                    new_edge_trails.append(new_edge_trail)

            current_slots = new_slots
            paths = new_paths
            edge_trails = new_edge_trails

            if not current_slots:
                return [], []

            # Cap at 1000 results
            if len(current_slots) > 1000:
                current_slots = current_slots[:1000]
                paths = paths[:1000]
                edge_trails = edge_trails[:1000]

        # Convert paths to binding dicts
        bindings = []
        for path in paths:
            binding = {}
            for var_name, node_id in path:
                if not var_name.startswith("_"):
                    binding[var_name] = node_id
            bindings.append(binding)

        # Deduplicate edges by (source, target, kind)
        seen_edges: dict[str, dict] = {}
        for trail in edge_trails:
            for e in trail:
                key = f"{e['source']}->{e['target']}:{e['kind']}"
                seen_edges[key] = e

        return bindings, list(seen_edges.values())

    def _count(self, q: CountQuery) -> Result:
        if q.target == "NODES":
            if q.where:
                kind_filter = self._extract_kind_from_where(q.where)
                remaining = self._strip_kind_from_expr(q.where.expr)

                if remaining is None:
                    # Pure kind filter - numpy count, zero dict construction
                    count = self.store.count_nodes(kind=kind_filter)
                else:
                    col_count = self._try_column_count(remaining, kind_filter)
                    if col_count is not None:
                        count = col_count
                    else:
                        raw_pred = self._make_raw_predicate(remaining)
                        if raw_pred is not None:
                            count = self.store.count_nodes(kind=kind_filter, predicate=raw_pred)
                        else:
                            nodes = self.store.get_all_nodes(kind=kind_filter)
                            count = sum(1 for n in nodes if self._eval_where(q.where.expr, n))
            else:
                count = self.store.node_count
        else:  # EDGES
            if q.where:
                kind_filter = self._extract_kind_from_where(q.where)
                if self._is_simple_kind_filter(q.where) and kind_filter:
                    count = len(self.store._edges_by_type.get(kind_filter, []))
                else:
                    edges = self.store.get_all_edges()
                    count = sum(1 for e in edges if self._eval_where(q.where.expr, e))
            else:
                count = self.store.edge_count
        return Result(kind="count", data=count, count=count)

    def _aggregate(self, query: AggregateQuery) -> Result:
        n = self.store._next_slot
        if n == 0:
            group_by = query.group_by or []
            select = query.select or []
            if group_by:
                return Result(kind="aggregate", data=[], count=0)
            else:
                row = {}
                for func in select:
                    if func.func == "COUNT" or func.func == "COUNT_DISTINCT":
                        row[func.label()] = 0
                    elif func.func == "AVG":
                        row[func.label()] = None
                    else:
                        row[func.label()] = 0
                return Result(kind="aggregate", data=[row], count=1)

        mask = self._compute_live_mask(n)

        # Apply WHERE
        if query.where:
            kind_filter = self._extract_kind_from_where(query.where)
            if kind_filter:
                kind_mask = self.store._live_mask(kind_filter)
                mask = mask & kind_mask
                remaining = self._strip_kind_from_expr(query.where.expr)
                if remaining is not None:
                    where_mask = self._try_column_filter(remaining, mask, n)
                    if where_mask is None:
                        raise AggregationError("AGGREGATE WHERE fields must be columnarized")
                    mask = where_mask
            else:
                where_mask = self._try_column_filter(query.where.expr, mask, n)
                if where_mask is None:
                    raise AggregationError("AGGREGATE WHERE fields must be columnarized")
                mask = where_mask

        # Validate fields are columnarized
        group_by = query.group_by or []
        select = query.select or []
        for field in group_by:
            if not self.store.columns.has_column(field):
                raise AggregationError(f"GROUP BY field '{field}' is not columnarized")
        for func in select:
            if func.field and not self.store.columns.has_column(func.field):
                raise AggregationError(f"Aggregate field '{func.field}' is not columnarized")

        # Get filtered count
        filtered_count = int(np.sum(mask))
        if filtered_count == 0:
            if group_by:
                return Result(kind="aggregate", data=[], count=0)
            else:
                # Global aggregate returns one row with zeros
                row = {}
                for func in select:
                    if func.func == "COUNT" or func.func == "COUNT_DISTINCT":
                        row[func.label()] = 0
                    elif func.func == "AVG":
                        row[func.label()] = None
                    else:
                        row[func.label()] = 0
                return Result(kind="aggregate", data=[row], count=1)

        # Build group keys
        if group_by:
            group_cols = [self.store.columns._columns[f][:n][mask] for f in group_by]
            if len(group_cols) == 1:
                keys = group_cols[0]
                unique_keys, inverse = np.unique(keys, return_inverse=True)
            else:
                keys = np.column_stack(group_cols)
                unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
        else:
            unique_keys = np.array([0])
            inverse = np.zeros(filtered_count, dtype=np.intp)

        num_groups = len(unique_keys)

        # Compute aggregates
        results = {}
        for func in select:
            label = func.label()
            if func.func == "COUNT":
                results[label] = np.bincount(inverse, minlength=num_groups).astype(np.float64)
            elif func.func == "COUNT_DISTINCT":
                col = self.store.columns._columns[func.field][:n][mask]
                counts = np.zeros(num_groups, dtype=np.float64)
                for g in range(num_groups):
                    counts[g] = len(np.unique(col[inverse == g]))
                results[label] = counts
            elif func.func == "SUM":
                col = self.store.columns._columns[func.field][:n][mask].astype(np.float64)
                sums = np.zeros(num_groups, dtype=np.float64)
                np.add.at(sums, inverse, col)
                results[label] = sums
            elif func.func == "AVG":
                col = self.store.columns._columns[func.field][:n][mask].astype(np.float64)
                sums = np.zeros(num_groups, dtype=np.float64)
                np.add.at(sums, inverse, col)
                counts = np.bincount(inverse, minlength=num_groups).astype(np.float64)
                results[label] = sums / np.maximum(counts, 1)
            elif func.func == "MIN":
                col = self.store.columns._columns[func.field][:n][mask].astype(np.float64)
                mins = np.full(num_groups, np.inf)
                np.minimum.at(mins, inverse, col)
                results[label] = mins
            elif func.func == "MAX":
                col = self.store.columns._columns[func.field][:n][mask].astype(np.float64)
                maxs = np.full(num_groups, -np.inf)
                np.maximum.at(maxs, inverse, col)
                results[label] = maxs

        # Build result dicts
        group_dicts = []
        for i in range(num_groups):
            d = {}
            if group_by:
                for j, field in enumerate(group_by):
                    if len(group_by) == 1:
                        raw = unique_keys[i]
                    else:
                        raw = unique_keys[i][j]
                    dtype = self.store.columns._dtypes[field]
                    if dtype == "int32_interned":
                        d[field] = self.store.string_table.lookup(int(raw))
                    elif dtype == "float64":
                        d[field] = float(raw)
                    elif dtype == "int64":
                        d[field] = int(raw)
            for label, arr in results.items():
                val = arr[i]
                if np.isnan(val) or np.isinf(val):
                    d[label] = None if np.isnan(val) else float(val)
                else:
                    d[label] = int(val) if float(val) == int(float(val)) else float(val)
            group_dicts.append(d)

        # HAVING filter
        if query.having:
            group_dicts = [d for d in group_dicts if self._eval_where(query.having, d)]

        # ORDER BY
        if query.order_by:
            sort_key = query.order_by.label()
            group_dicts.sort(key=lambda d: d.get(sort_key, 0), reverse=query.order_desc)

        # LIMIT
        if query.limit:
            group_dicts = group_dicts[:query.limit.value]

        return Result(kind="aggregate", data=group_dicts, count=len(group_dicts))

    # --- Intelligence handlers ---

    def _recall(self, q: RecallQuery) -> Result:
        """RECALL: spreading activation from a cue node."""
        cue_slot = self._resolve_slot(q.node_id)
        if cue_slot is None:
            raise NodeNotFound(q.node_id)

        n = self.store._next_slot
        if n == 0:
            return Result(kind="nodes", data=[], count=0)

        # Get combined CSR matrix (sum of all edge types)
        combined = self.store.edge_matrices.get(None)
        if combined is None:
            return Result(kind="nodes", data=[], count=0)

        # Resize CSR matrix if needed (may be smaller than n)
        mat = combined
        if mat.shape[0] < n:
            mat = csr_matrix((mat.data, mat.indices, mat.indptr), shape=(n, n))

        # Seed activation
        activation = np.zeros(n, dtype=np.float64)
        activation[cue_slot] = 1.0

        # Build importance weights
        importance = np.ones(n, dtype=np.float64)
        imp_col = self.store.columns.get_column("importance", n)
        if imp_col is not None:
            col_data, col_pres, _ = imp_col
            importance[col_pres] = col_data[col_pres].astype(np.float64)
        conf_col = self.store.columns.get_column("__confidence__", n)
        if conf_col is not None:
            col_data, col_pres, _ = conf_col
            mask = col_pres
            importance[mask] *= col_data[mask].astype(np.float64)

        # Recency weighting: 1.0 / (1.0 + age_in_days)
        recency = np.ones(n, dtype=np.float64)
        updated_col = self.store.columns.get_column("__updated_at__", n)
        if updated_col is not None:
            col_data, col_pres, _ = updated_col
            now_ms = int(time.time() * 1000)
            age_days = np.where(col_pres, (now_ms - col_data.astype(np.float64)) / 86400000.0, 0.0)
            recency = np.where(col_pres, 1.0 / (1.0 + age_days), 1.0)

        # Compute live mask
        live_mask = self._compute_live_mask(n)

        # Spreading activation over hops
        # mat[i,j] = edge from i to j
        # mat.T.dot(v) propagates activation from sources to targets
        mat_t = mat.T.tocsr()
        for _ in range(q.depth):
            activation = mat_t.dot(activation)
            # Weight by importance and recency
            activation *= importance[:len(activation)]
            activation *= recency[:len(activation)]
            # Zero out non-live nodes EACH hop
            activation[:n] *= live_mask.astype(np.float64)

        # Zero out the cue node itself
        activation[cue_slot] = 0.0

        # Apply WHERE filter (kind filter)
        if q.where:
            kind_filter = self._extract_kind_from_where(q.where)
            if kind_filter:
                kind_mask = self.store._live_mask(kind_filter)
                activation[:n] *= kind_mask

        # Find activated nodes
        active_indices = np.nonzero(activation > 0)[0]
        if len(active_indices) == 0:
            return Result(kind="nodes", data=[], count=0)

        # Top-K via argpartition
        limit = q.limit.value if q.limit else len(active_indices)
        k = min(limit, len(active_indices))

        if k < len(active_indices):
            top_k_idx = np.argpartition(-activation[active_indices], k)[:k]
        else:
            top_k_idx = np.arange(len(active_indices))

        # Sort by activation descending
        top_k_slots = active_indices[top_k_idx]
        sorted_order = np.argsort(-activation[top_k_slots])
        top_k_slots = top_k_slots[sorted_order]

        # Materialize results
        results = []
        for slot in top_k_slots:
            slot = int(slot)
            node = self.store._materialize_slot(slot)
            if node is not None:
                # Apply WHERE if it's more than a kind filter
                if q.where and not self._is_simple_kind_filter(q.where):
                    remaining = self._strip_kind_from_expr(q.where.expr)
                    if remaining is not None:
                        if not self._eval_where(remaining, node):
                            continue
                node["_activation_score"] = float(activation[slot])
                results.append(node)

        return Result(kind="nodes", data=results, count=len(results))

    def _similar(self, q: SimilarQuery) -> Result:
        """SIMILAR TO: vector similarity search."""
        # 1. Resolve query vector
        if q.target_vector is not None:
            query_vec = np.array(q.target_vector, dtype=np.float32)
        elif q.target_text is not None:
            if not self._embedder:
                raise EmbedderRequired("Text similarity requires an embedder")
            query_vec = self._embedder.encode_queries([q.target_text])[0]
        elif q.target_node_id is not None:
            slot = self._resolve_slot(q.target_node_id)
            if slot is None:
                raise NodeNotFound(q.target_node_id)
            if not self._vector_store or not self._vector_store.has_vector(slot):
                raise VectorNotFound(f"Node '{q.target_node_id}' has no vector")
            query_vec = self._vector_store.get_vector(slot)
        else:
            raise VectorError("SIMILAR TO requires a vector, text, or node target")

        if not self._vector_store or self._vector_store.count() == 0:
            return Result(kind="nodes", data=[], count=0)

        # 2. Build combined mask: live + has_vector
        n = self.store._next_slot
        if n == 0:
            return Result(kind="nodes", data=[], count=0)

        mask = self._compute_live_mask(n)
        vs_cap = self._vector_store._capacity
        if n <= vs_cap:
            vs_mask = self._vector_store._has_vector[:n]
        else:
            vs_mask = np.zeros(n, dtype=bool)
            vs_mask[:vs_cap] = self._vector_store._has_vector[:vs_cap]
        combined_mask = mask & vs_mask

        # 3. Search
        k = q.limit.value if q.limit else 10
        search_k = k * 3 if q.where else k
        slots, dists = self._vector_store.search(query_vec, k=search_k, mask=combined_mask)

        # 4. Post-filter with WHERE
        results = []
        target_k = q.limit.value if q.limit else 10
        for slot_idx, dist in zip(slots, dists):
            slot = int(slot_idx)
            node = self.store._materialize_slot(slot)
            if node is None:
                continue
            if q.where and not self._eval_where(q.where.expr, node):
                continue
            node["_similarity_score"] = round(1.0 - float(dist), 4)
            results.append(node)
            if len(results) >= target_k:
                break

        return Result(kind="nodes", data=results, count=len(results))

    def _counterfactual(self, q: CounterfactualQuery) -> Result:
        """WHAT IF RETRACT: simulate retraction without committing."""
        src_slot = self._resolve_slot(q.node_id)
        if src_slot is None:
            raise NodeNotFound(q.node_id)

        # Save full state
        saved_columns = self.store.columns.snapshot_arrays()
        saved_tombstones = set(self.store.node_tombstones)
        saved_edges = {k: list(v) for k, v in self.store._edges_by_type.items()}
        saved_edge_keys = set(self.store._edge_keys)
        saved_id_to_slot = dict(self.store.id_to_slot)
        saved_count = self.store._count
        saved_next_slot = self.store._next_slot
        saved_node_ids = self.store.node_ids[:self.store._next_slot].copy()
        saved_node_kinds = self.store.node_kinds[:self.store._next_slot].copy()

        try:
            # Apply hypothetical retraction
            self.store.columns.set_reserved(src_slot, "__retracted__", 1)

            # Find all descendants via BFS on edge matrices
            combined = self.store.edge_matrices.get(None)
            n = self.store._next_slot
            affected_slots = set()
            affected_slots.add(src_slot)

            if combined is not None:
                mat = combined
                if mat.shape[0] < n:
                    mat = csr_matrix((mat.data, mat.indices, mat.indptr), shape=(n, n))

                frontier = deque([src_slot])
                visited = {src_slot}
                while frontier:
                    current = frontier.popleft()
                    if current < mat.shape[0]:
                        start = mat.indptr[current]
                        end = mat.indptr[current + 1]
                        for nb in mat.indices[start:end]:
                            nb = int(nb)
                            if nb not in visited:
                                visited.add(nb)
                                affected_slots.add(nb)
                                frontier.append(nb)

            # Materialize affected nodes before restore
            affected_nodes = []
            for slot in affected_slots:
                node = self.store._materialize_slot(int(slot))
                if node is not None:
                    affected_nodes.append(node)

            return Result(
                kind="counterfactual",
                data={
                    "retracted": q.node_id,
                    "affected_nodes": affected_nodes,
                    "affected_count": len(affected_nodes),
                },
                count=len(affected_nodes),
            )
        finally:
            # Restore state completely
            self.store.columns.restore_arrays(saved_columns)
            self.store.node_tombstones = saved_tombstones
            self.store._edges_by_type = saved_edges
            self.store._edge_keys = saved_edge_keys
            self.store.id_to_slot = saved_id_to_slot
            self.store._count = saved_count
            self.store._next_slot = saved_next_slot
            self.store.node_ids[:saved_next_slot] = saved_node_ids
            self.store.node_kinds[:saved_next_slot] = saved_node_kinds
            self.store._rebuild_edges()
