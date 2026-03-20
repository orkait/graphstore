"""DSL executor: maps parsed AST nodes to CoreStore operations.

Handles all user read and write queries. System queries are handled
separately by the server layer.
"""

import time
from collections import deque

import numpy as np
from scipy.sparse import csr_matrix

from graphstore.dsl.ast_nodes import (
    AggFunc,
    AggregateQuery,
    AncestorsQuery,
    AndExpr,
    AssertStmt,
    Batch,
    BindContext,
    CommonNeighborsQuery,
    Condition,
    ContainsCondition,
    CounterfactualQuery,
    DiscardContext,
    LikeCondition,
    InCondition,
    CreateEdge,
    CreateNode,
    DegreeCondition,
    DeleteEdge,
    MergeStmt,
    PropagateStmt,
    RecallQuery,
    RetractStmt,
    UpdateNodes,
    VarAssign,
    WeightedShortestPathQuery,
    WeightedDistanceQuery,
    CountQuery,
    UpdateEdge,
    OffsetClause,
    OrderClause,
    DeleteEdges,
    DeleteNode,
    DeleteNodes,
    DescendantsQuery,
    DistanceQuery,
    EdgesQuery,
    Increment,
    MatchPattern,
    MatchQuery,
    NodeQuery,
    NodesQuery,
    NotExpr,
    OrExpr,
    PathQuery,
    PathsQuery,
    ShortestPathQuery,
    SubgraphQuery,
    TraverseQuery,
    UpdateNode,
    UpsertNode,
)
from graphstore.dsl.cost_estimator import estimate_match_cost, estimate_traverse_cost
from graphstore.errors import (
    AggregationError,
    BatchRollback,
    CostThresholdExceeded,
    GraphStoreError,
    NodeNotFound,
)
from graphstore.path import (
    bfs_traverse,
    bidirectional_bfs,
    common_neighbors,
    dijkstra,
    find_all_paths,
)
from graphstore.schema import SchemaRegistry
from graphstore.store import CoreStore
from graphstore.types import Result


class Executor:
    def __init__(self, store: CoreStore, schema: SchemaRegistry | None = None):
        self.store = store
        self.schema = schema or SchemaRegistry()
        self.cost_threshold = 100_000

    def execute(self, ast) -> Result:
        """Execute a parsed AST node and return a Result."""
        start = time.perf_counter_ns()
        result = self._dispatch(ast)
        elapsed = (time.perf_counter_ns() - start) // 1000
        result.elapsed_us = elapsed
        return result

    def _dispatch(self, ast) -> Result:
        """Route AST node to handler."""
        handlers = {
            NodeQuery: self._node,
            NodesQuery: self._nodes,
            EdgesQuery: self._edges,
            TraverseQuery: self._traverse,
            SubgraphQuery: self._subgraph,
            PathQuery: self._path,
            PathsQuery: self._paths,
            ShortestPathQuery: self._shortest_path,
            DistanceQuery: self._distance,
            WeightedShortestPathQuery: self._weighted_shortest_path,
            WeightedDistanceQuery: self._weighted_distance,
            AncestorsQuery: self._ancestors,
            DescendantsQuery: self._descendants,
            CommonNeighborsQuery: self._common_neighbors,
            MatchQuery: self._match,
            CountQuery: self._count,
            AggregateQuery: self._aggregate,
            CreateNode: self._create_node,
            UpdateNode: self._update_node,
            UpsertNode: self._upsert_node,
            DeleteNode: self._delete_node,
            DeleteNodes: self._delete_nodes,
            CreateEdge: self._create_edge,
            UpdateEdge: self._update_edge,
            DeleteEdge: self._delete_edge,
            DeleteEdges: self._delete_edges,
            Increment: self._increment,
            Batch: self._batch,
            AssertStmt: self._assert,
            RetractStmt: self._retract,
            UpdateNodes: self._update_nodes,
            MergeStmt: self._merge,
            RecallQuery: self._recall,
            CounterfactualQuery: self._counterfactual,
            PropagateStmt: self._propagate,
            BindContext: self._bind_context,
            DiscardContext: self._discard_context,
        }
        handler = handlers.get(type(ast))
        if handler is None:
            raise GraphStoreError(f"Unknown AST node type: {type(ast).__name__}")
        return handler(ast)

    # --- Read handlers ---

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

    # --- Write handlers ---

    def _generate_auto_id(self, kind: str, data: dict) -> str:
        """Generate a deterministic content-hash ID from kind + sorted fields."""
        import hashlib
        parts = [f"kind={kind}"]
        for k in sorted(data.keys()):
            parts.append(f"{k}={data[k]}")
        content = "|".join(parts)
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def _create_node(self, q: CreateNode) -> Result:
        data = {fp.name: fp.value for fp in q.fields}
        kind = data.pop("kind", "default")
        self.schema.validate_node(kind, data)
        if q.auto_id:
            node_id = self._generate_auto_id(kind, data)
        else:
            node_id = q.id
        self.store.put_node(node_id, kind, data)
        # Handle TTL
        self._apply_ttl(node_id, q.expires_in, q.expires_at)
        # Auto-tag context if bound
        if self.store._active_context:
            str_id = self.store.string_table.intern(node_id)
            slot = self.store.id_to_slot[str_id]
            self.store.columns.set_reserved(slot, "__context__", self.store._active_context)
        node = self.store.get_node(node_id)
        return Result(kind="node", data=node, count=1)

    def _update_node(self, q: UpdateNode) -> Result:
        data = {fp.name: fp.value for fp in q.fields}
        self.store.update_node(q.id, data)
        node = self.store.get_node(q.id)
        return Result(kind="node", data=node, count=1)

    def _upsert_node(self, q: UpsertNode) -> Result:
        data = {fp.name: fp.value for fp in q.fields}
        kind = data.pop("kind", "default")
        self.schema.validate_node(kind, data)
        self.store.upsert_node(q.id, kind, data)
        # Handle TTL
        self._apply_ttl(q.id, q.expires_in, q.expires_at)
        node = self.store.get_node(q.id)
        return Result(kind="node", data=node, count=1)

    def _delete_node(self, q: DeleteNode) -> Result:
        self.store.delete_node(q.id)
        return Result(kind="ok", data={"id": q.id}, count=1)

    def _delete_nodes(self, q: DeleteNodes) -> Result:
        kind_filter = self._extract_kind_from_where(q.where)
        remaining = self._strip_kind_from_expr(q.where.expr)

        ids_to_delete = None

        if remaining is not None:
            col_ids = self._try_column_delete_ids(remaining, kind_filter)
            if col_ids is not None:
                ids_to_delete = col_ids

        if ids_to_delete is None:
            if remaining is not None:
                raw_pred = self._make_raw_predicate(remaining)
            else:
                raw_pred = None

            if raw_pred is not None or remaining is None:
                ids_to_delete = self.store.query_node_ids(kind=kind_filter, predicate=raw_pred)
            else:
                nodes = self.store.get_all_nodes(kind=kind_filter)
                ids_to_delete = [n["id"] for n in nodes if self._eval_where(q.where.expr, n)]

        deleted_ids = []
        for nid in ids_to_delete:
            try:
                self.store.delete_node(nid)
                deleted_ids.append(nid)
            except NodeNotFound:
                pass
        return Result(kind="nodes", data=[{"id": i} for i in deleted_ids], count=len(deleted_ids))

    def _create_edge(self, q: CreateEdge) -> Result:
        data = {fp.name: fp.value for fp in q.fields}
        kind = data.pop("kind", "default")
        # Validate edge endpoint kinds if schema is registered
        src_node = self.store.get_node(q.source)
        tgt_node = self.store.get_node(q.target)
        if src_node and tgt_node:
            self.schema.validate_edge(kind, src_node["kind"], tgt_node["kind"])
        self.store.put_edge(q.source, q.target, kind, data if data else None)
        return Result(kind="edges", data=[{"source": q.source, "target": q.target, "kind": kind}], count=1)

    def _update_edge(self, q: UpdateEdge) -> Result:
        kind = self._extract_kind_from_where(q.where)
        update_data = {fp.name: fp.value for fp in q.fields}
        # Find and update matching edges
        updated = 0
        for etype in ([kind] if kind else list(self.store._edges_by_type.keys())):
            if etype not in self.store._edges_by_type:
                continue
            src_str_id = self.store.string_table.intern(q.source)
            tgt_str_id = self.store.string_table.intern(q.target)
            src_slot = self.store.id_to_slot.get(src_str_id)
            tgt_slot = self.store.id_to_slot.get(tgt_str_id)
            if src_slot is None or tgt_slot is None:
                continue
            for i, (s, t, d) in enumerate(self.store._edges_by_type[etype]):
                if s == src_slot and t == tgt_slot:
                    self.store._edges_by_type[etype][i] = (s, t, {**d, **update_data})
                    updated += 1
        if updated > 0:
            self.store._edges_dirty = True
            self.store._ensure_edges_built()
        return Result(kind="ok", data={"source": q.source, "target": q.target, "updated": updated}, count=updated)

    def _count(self, q: CountQuery) -> Result:
        if q.target == "NODES":
            if q.where:
                kind_filter = self._extract_kind_from_where(q.where)
                remaining = self._strip_kind_from_expr(q.where.expr)

                if remaining is None:
                    # Pure kind filter — numpy count, zero dict construction
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

    def _delete_edge(self, q: DeleteEdge) -> Result:
        kind = self._extract_kind_from_where(q.where)
        if kind:
            self.store.delete_edge(q.source, q.target, kind)
        else:
            for etype in list(self.store._edges_by_type.keys()):
                self.store.delete_edge(q.source, q.target, etype)
        return Result(kind="ok", data=None, count=1)

    def _delete_edges(self, q: DeleteEdges) -> Result:
        kind = self._extract_kind_from_where(q.where)
        if q.direction == "FROM":
            edges = self.store.get_edges_from(q.node_id, kind=kind)
        else:
            edges = self.store.get_edges_to(q.node_id, kind=kind)

        for e in edges:
            self.store.delete_edge(e["source"], e["target"], e["kind"])
        return Result(kind="edges", data=edges, count=len(edges))

    def _increment(self, q: Increment) -> Result:
        self.store.increment_field(q.node_id, q.field, q.amount)
        return Result(kind="ok", data=None, count=0)

    def _assert(self, q: AssertStmt) -> Result:
        """ASSERT: upsert with reserved __confidence__, __source__, __retracted__=0."""
        data = {fp.name: fp.value for fp in q.fields}
        kind = data.pop("kind", "default")
        self.schema.validate_node(kind, data)
        self.store.upsert_node(q.id, kind, data)
        # Set reserved belief fields
        str_id = self.store.string_table.intern(q.id)
        slot = self.store.id_to_slot[str_id]
        if q.confidence is not None:
            self.store.columns.set_reserved(slot, "__confidence__", q.confidence)
        if q.source is not None:
            self.store.columns.set_reserved(slot, "__source__", q.source)
        self.store.columns.set_reserved(slot, "__retracted__", 0)
        node = self.store.get_node(q.id)
        return Result(kind="node", data=node, count=1)

    def _retract(self, q: RetractStmt) -> Result:
        """RETRACT: mark node as retracted (invisible via live_mask)."""
        if q.id not in self.store.string_table:
            raise NodeNotFound(q.id)
        str_id = self.store.string_table.intern(q.id)
        slot = self.store.id_to_slot.get(str_id)
        if slot is None or slot in self.store.node_tombstones:
            raise NodeNotFound(q.id)
        now_ms = int(time.time() * 1000)
        self.store.columns.set_reserved(slot, "__retracted__", 1)
        self.store.columns.set_reserved(slot, "__retracted_at__", now_ms)
        if q.reason:
            self.store.columns.set_reserved(slot, "__retract_reason__", q.reason)
        return Result(kind="ok", data={"id": q.id}, count=1)

    def _update_nodes(self, q: UpdateNodes) -> Result:
        """UPDATE NODES WHERE ... SET ...: bulk column update."""
        n = self.store._next_slot
        if n == 0:
            return Result(kind="ok", data={"updated": 0}, count=0)

        mask = self._compute_live_mask(n)

        # Apply WHERE filter
        kind_filter = self._extract_kind_from_where(q.where)
        if kind_filter:
            kind_mask = self.store._live_mask(kind_filter)
            mask = mask & kind_mask
            remaining = self._strip_kind_from_expr(q.where.expr)
            if remaining is not None:
                col_mask = self._try_column_filter(remaining, mask, n)
                if col_mask is not None:
                    mask = col_mask
                else:
                    # Fallback: materialize and filter
                    fallback_mask = np.zeros(n, dtype=bool)
                    for slot_idx in np.nonzero(mask)[0]:
                        node = self.store._materialize_slot(int(slot_idx))
                        if node and self._eval_where(q.where.expr, node):
                            fallback_mask[int(slot_idx)] = True
                    mask = fallback_mask
        else:
            col_mask = self._try_column_filter(q.where.expr, mask, n)
            if col_mask is not None:
                mask = col_mask
            else:
                fallback_mask = np.zeros(n, dtype=bool)
                for slot_idx in np.nonzero(mask)[0]:
                    node = self.store._materialize_slot(int(slot_idx))
                    if node and self._eval_where(q.where.expr, node):
                        fallback_mask[int(slot_idx)] = True
                mask = fallback_mask

        update_data = {fp.name: fp.value for fp in q.fields}
        matching_slots = np.nonzero(mask)[0]
        now_ms = int(time.time() * 1000)

        if len(matching_slots) > 0:
            store = self.store
            # Bulk numpy assignment for columnar fields
            for fp in q.fields:
                field = fp.name
                value = fp.value
                if store.columns.has_column(field):
                    dtype_str = store.columns._dtypes[field]
                    if dtype_str == "int32_interned" and isinstance(value, str):
                        raw_val = store.string_table.intern(value)
                        store.columns._columns[field][matching_slots] = raw_val
                        store.columns._presence[field][matching_slots] = True
                    elif dtype_str == "int64" and isinstance(value, int):
                        store.columns._columns[field][matching_slots] = int(value)
                        store.columns._presence[field][matching_slots] = True
                    elif dtype_str == "float64" and isinstance(value, (int, float)) and not isinstance(value, bool):
                        store.columns._columns[field][matching_slots] = float(value)
                        store.columns._presence[field][matching_slots] = True
                    else:
                        # Type mismatch - fall back to per-slot set
                        for slot_idx in matching_slots:
                            store.columns.set(int(slot_idx), {field: value})
                else:
                    # Column doesn't exist yet - use set() which auto-creates
                    for slot_idx in matching_slots:
                        store.columns.set(int(slot_idx), {field: value})

            # Bulk set __updated_at__ via numpy
            if store.columns.has_column("__updated_at__"):
                store.columns._columns["__updated_at__"][matching_slots] = now_ms
                store.columns._presence["__updated_at__"][matching_slots] = True
            else:
                for slot_idx in matching_slots:
                    store.columns.set_reserved(int(slot_idx), "__updated_at__", now_ms)

        # Rebuild secondary indices for updated fields
        for fp in q.fields:
            if fp.name in self.store._indexed_fields:
                self.store.add_index(fp.name)  # rebuild index for this field

        updated = len(matching_slots)
        return Result(kind="ok", data={"updated": updated}, count=updated)

    def _merge(self, q: MergeStmt) -> Result:
        """MERGE NODE src INTO tgt: copy fields, rewire edges, tombstone source."""
        # Validate source and target exist
        if q.source_id not in self.store.string_table:
            raise NodeNotFound(q.source_id)
        src_str = self.store.string_table.intern(q.source_id)
        src_slot = self.store.id_to_slot.get(src_str)
        if src_slot is None or src_slot in self.store.node_tombstones:
            raise NodeNotFound(q.source_id)

        if q.target_id not in self.store.string_table:
            raise NodeNotFound(q.target_id)
        tgt_str = self.store.string_table.intern(q.target_id)
        tgt_slot = self.store.id_to_slot.get(tgt_str)
        if tgt_slot is None or tgt_slot in self.store.node_tombstones:
            raise NodeNotFound(q.target_id)

        # 1. Copy source column values to target where target has no value
        fields_merged = 0
        for field in list(self.store.columns._columns.keys()):
            if field.startswith("__") and field.endswith("__"):
                continue  # skip reserved columns
            if not self.store.columns._presence[field][src_slot]:
                continue  # source has no value
            if self.store.columns._presence[field][tgt_slot]:
                continue  # target wins on conflict
            # Copy source value to target
            dtype = self.store.columns._dtypes[field]
            raw = self.store.columns._columns[field][src_slot]
            self.store.columns._columns[field][tgt_slot] = raw
            self.store.columns._presence[field][tgt_slot] = True
            fields_merged += 1

        # 2. Re-wire all edges from source to target
        edges_rewired = 0
        for etype in list(self.store._edges_by_type.keys()):
            new_edges = []
            for s, t, d in self.store._edges_by_type[etype]:
                if s == src_slot:
                    new_edges.append((tgt_slot, t, d))
                    edges_rewired += 1
                elif t == src_slot:
                    new_edges.append((s, tgt_slot, d))
                    edges_rewired += 1
                else:
                    new_edges.append((s, t, d))
            self.store._edges_by_type[etype] = new_edges

        # 3. Drop duplicate edges after re-wiring
        for etype in list(self.store._edges_by_type.keys()):
            seen = set()
            deduped = []
            for s, t, d in self.store._edges_by_type[etype]:
                key = (s, t)
                if key not in seen:
                    seen.add(key)
                    deduped.append((s, t, d))
            self.store._edges_by_type[etype] = deduped
            if not deduped:
                del self.store._edges_by_type[etype]

        # 4. Rebuild edge keys
        self.store._edge_keys = {
            (s, t, k)
            for k, edges in self.store._edges_by_type.items()
            for s, t, _d in edges
        }
        self.store._edges_dirty = True
        self.store._ensure_edges_built()

        # 5. Tombstone source
        self.store.delete_node(q.source_id)

        # 6. Update target's __updated_at__
        now_ms = int(time.time() * 1000)
        self.store.columns.set_reserved(tgt_slot, "__updated_at__", now_ms)

        return Result(
            kind="ok",
            data={
                "merged_into": q.target_id,
                "fields_merged": fields_merged,
                "edges_rewired": edges_rewired,
            },
            count=1,
        )

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

    def _propagate(self, q: PropagateStmt) -> Result:
        """PROPAGATE: BFS forward belief chaining."""
        src_slot = self._resolve_slot(q.node_id)
        if src_slot is None:
            raise NodeNotFound(q.node_id)

        n = self.store._next_slot
        field = q.field

        # Get source field value
        if not self.store.columns.has_column(field):
            return Result(kind="ok", data={"updated": 0}, count=0)
        if not self.store.columns._presence[field][src_slot]:
            return Result(kind="ok", data={"updated": 0}, count=0)

        # BFS from source
        combined = self.store.edge_matrices.get(None)
        if combined is None:
            return Result(kind="ok", data={"updated": 0}, count=0)

        mat = combined
        if mat.shape[0] < n:
            mat = csr_matrix((mat.data, mat.indices, mat.indptr), shape=(n, n))

        visited = set()
        visited.add(src_slot)
        frontier = deque()

        # Get source value
        dtype = self.store.columns._dtypes[field]
        raw = self.store.columns._columns[field][src_slot]
        if dtype == "float64":
            source_value = float(raw)
        elif dtype == "int64":
            source_value = int(raw)
        else:
            return Result(kind="ok", data={"updated": 0}, count=0)

        # Push initial neighbors with (slot, parent_value)
        frontier.append((src_slot, source_value, 0))
        updated_count = 0
        now_ms = int(time.time() * 1000)

        while frontier:
            current_slot, parent_value, depth = frontier.popleft()
            if depth >= q.depth:
                continue

            # Get outgoing neighbors from CSR
            start = mat.indptr[current_slot]
            end = mat.indptr[current_slot + 1]
            neighbors = mat.indices[start:end]
            weights = mat.data[start:end]

            for i, nb in enumerate(neighbors):
                nb = int(nb)
                if nb in visited:
                    continue
                visited.add(nb)

                if nb in self.store.node_tombstones:
                    continue

                # Get current value or default
                if self.store.columns._presence[field][nb]:
                    if dtype == "float64":
                        current_val = float(self.store.columns._columns[field][nb])
                    else:
                        current_val = int(self.store.columns._columns[field][nb])
                else:
                    current_val = 0.0

                # Propagated value = parent * edge_weight
                edge_weight = float(weights[i]) if i < len(weights) else 1.0
                propagated = parent_value * edge_weight

                # Update the field
                self.store.columns.set(nb, {field: propagated})
                self.store.columns.set_reserved(nb, "__updated_at__", now_ms)
                updated_count += 1

                frontier.append((nb, propagated, depth + 1))

        return Result(kind="ok", data={"updated": updated_count}, count=updated_count)

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

    def _bind_context(self, q: BindContext) -> Result:
        """BIND CONTEXT: set active context on store."""
        self.store._active_context = q.name
        return Result(kind="ok", data={"context": q.name}, count=0)

    def _discard_context(self, q: DiscardContext) -> Result:
        """DISCARD CONTEXT: delete all nodes with matching __context__ and unbind."""
        deleted_count = 0
        n = self.store._next_slot
        if n > 0 and self.store.columns.has_column("__context__"):
            ctx_col = self.store.columns.get_column("__context__", n)
            if ctx_col is not None:
                col_data, col_pres, _ = ctx_col
                ctx_id = self.store.string_table.intern(q.name)
                ctx_mask = col_pres & (col_data == ctx_id)
                slots_to_delete = np.nonzero(ctx_mask)[0]
                for slot in slots_to_delete:
                    nid = self.store._slot_to_id(int(slot))
                    if nid:
                        try:
                            self.store.delete_node(nid)
                            deleted_count += 1
                        except NodeNotFound:
                            pass

        # Unbind context
        self.store._active_context = None
        return Result(kind="ok", data={"discarded": q.name, "deleted": deleted_count}, count=deleted_count)

    def _apply_ttl(self, node_id: str, expires_in: tuple | None, expires_at: str | None):
        """Set __expires_at__ on a node based on TTL clauses."""
        if expires_in is None and expires_at is None:
            return
        str_id = self.store.string_table.intern(node_id)
        slot = self.store.id_to_slot[str_id]
        if expires_in is not None:
            amount, unit = expires_in
            unit_ms = {"s": 1000, "m": 60000, "h": 3600000, "d": 86400000}[unit]
            expire_ms = int(time.time() * 1000) + amount * unit_ms
        else:
            from datetime import datetime
            dt = datetime.fromisoformat(expires_at)
            expire_ms = int(dt.timestamp() * 1000)
        self.store.columns.set_reserved(slot, "__expires_at__", expire_ms)

    def _batch(self, q: Batch) -> Result:
        """Execute batch with rollback on failure.

        Saves a copy of the store state before executing. If any statement
        fails, restores from the copy.
        """
        saved_edges = {k: list(v) for k, v in self.store._edges_by_type.items()}
        saved_edge_keys = set(self.store._edge_keys)
        saved_columns = self.store.columns.snapshot_arrays()
        saved_tombstones = set(self.store.node_tombstones)
        saved_id_to_slot = dict(self.store.id_to_slot)
        saved_count = self.store._count
        saved_next_slot = self.store._next_slot
        saved_node_ids = self.store.node_ids[: self.store._next_slot].copy()
        saved_node_kinds = self.store.node_kinds[: self.store._next_slot].copy()

        try:
            variables: dict[str, str] = {}  # $var -> resolved node ID
            for stmt in q.statements:
                if isinstance(stmt, VarAssign):
                    result = self._dispatch(stmt.statement)
                    # Extract the generated ID from the result
                    if result.data and isinstance(result.data, dict) and "id" in result.data:
                        variables[stmt.variable] = result.data["id"]
                    else:
                        raise GraphStoreError(
                            f"Variable {stmt.variable}: statement did not return an ID"
                        )
                else:
                    # Resolve $variables in CreateEdge source/target
                    if isinstance(stmt, CreateEdge):
                        src = variables.get(stmt.source, stmt.source) if stmt.source.startswith("$") else stmt.source
                        tgt = variables.get(stmt.target, stmt.target) if stmt.target.startswith("$") else stmt.target
                        if src.startswith("$"):
                            raise GraphStoreError(f"Unresolved variable: {src}")
                        if tgt.startswith("$"):
                            raise GraphStoreError(f"Unresolved variable: {tgt}")
                        resolved = CreateEdge(source=src, target=tgt, fields=stmt.fields)
                        self._dispatch(resolved)
                    else:
                        self._dispatch(stmt)
            # Flush any deferred edge rebuilds after successful batch
            self.store._ensure_edges_built()
            return Result(kind="ok", data=None, count=0)
        except Exception as e:
            # Rollback
            self.store._edges_by_type = saved_edges
            self.store._edge_keys = saved_edge_keys
            self.store.node_tombstones = saved_tombstones
            self.store.id_to_slot = saved_id_to_slot
            self.store._count = saved_count
            self.store._next_slot = saved_next_slot
            self.store.node_ids[:saved_next_slot] = saved_node_ids
            self.store.node_kinds[:saved_next_slot] = saved_node_kinds
            self.store._rebuild_edges()
            self.store.columns.restore_arrays(saved_columns)
            raise BatchRollback(
                failed_statement=str(type(e).__name__), error=str(e)
            )

    # --- Helpers ---

    def _compute_live_mask(self, n: int) -> np.ndarray:
        """Unified visibility filter: tombstones + TTL + retracted + context."""
        mask = self.store.compute_live_mask(n)

        # Context filtering: when bound, only show nodes tagged with active context
        if hasattr(self.store, '_active_context') and self.store._active_context:
            ctx_name = self.store._active_context
            ctx_mask = self.store.columns.get_mask("__context__", "=", ctx_name, n)
            if ctx_mask is not None:
                mask = mask & ctx_mask
            else:
                # No __context__ column at all - nothing has context
                mask = np.zeros(n, dtype=bool)

        return mask

    def _resolve_slot(self, node_id: str) -> int | None:
        """Resolve a string node ID to its slot index."""
        if node_id not in self.store.string_table:
            return None
        str_id = self.store.string_table.intern(node_id)
        slot = self.store.id_to_slot.get(str_id)
        if slot is None or slot in self.store.node_tombstones:
            return None
        return slot

    def _is_slot_visible(self, slot: int) -> bool:
        """Check if a slot passes TTL, retraction, and context checks."""
        # Check retracted
        if self.store.columns.has_column("__retracted__"):
            if self.store.columns._presence["__retracted__"][slot]:
                if int(self.store.columns._columns["__retracted__"][slot]) == 1:
                    return False
        # Check TTL expiry
        if self.store.columns.has_column("__expires_at__"):
            if self.store.columns._presence["__expires_at__"][slot]:
                expire_ms = int(self.store.columns._columns["__expires_at__"][slot])
                if expire_ms > 0 and expire_ms < int(time.time() * 1000):
                    return False
        # Check context
        if self.store._active_context is not None:
            if self.store.columns.has_column("__context__"):
                if self.store.columns._presence["__context__"][slot]:
                    ctx_id = self.store.string_table.intern(self.store._active_context)
                    if int(self.store.columns._columns["__context__"][slot]) != ctx_id:
                        return False
                else:
                    # Node has no context tag but context is active - invisible
                    return False
            else:
                # No context column at all - nothing has context
                return False
        return True

    def _is_visible_by_id(self, node_id: str) -> bool:
        """Check if a node ID is visible (not tombstoned, expired, or retracted)."""
        slot = self._resolve_slot(node_id)
        if slot is None:
            return False
        return self._is_slot_visible(slot)

    def _filter_visible(self, nodes: list[dict]) -> list[dict]:
        """Filter out retracted, expired, and out-of-context nodes."""
        has_retracted = self.store.columns.has_column("__retracted__")
        has_expires = self.store.columns.has_column("__expires_at__")
        has_context = self.store._active_context is not None
        if not has_retracted and not has_expires and not has_context:
            return nodes
        result = []
        for node in nodes:
            slot = self._resolve_slot(node["id"])
            if slot is not None and self._is_slot_visible(slot):
                result.append(node)
        return result

    def _eval_where(self, expr, data: dict) -> bool:
        """Evaluate a WHERE expression against a data dict."""
        if isinstance(expr, Condition):
            return self._eval_condition(expr, data)
        elif isinstance(expr, ContainsCondition):
            actual = data.get(expr.field)
            if actual is None:
                return False
            return expr.value in str(actual)
        elif isinstance(expr, LikeCondition):
            actual = data.get(expr.field)
            if actual is None:
                return False
            # Use cached compiled regex
            if not hasattr(expr, '_compiled_re'):
                import re
                parts = []
                for ch in expr.pattern:
                    if ch == '%':
                        parts.append('.*')
                    elif ch == '_':
                        parts.append('.')
                    else:
                        parts.append(re.escape(ch))
                expr._compiled_re = re.compile(''.join(parts))  # type: ignore[attr-defined]
            return bool(expr._compiled_re.fullmatch(str(actual)))
        elif isinstance(expr, InCondition):
            actual = data.get(expr.field)
            return actual in expr.values
        elif isinstance(expr, DegreeCondition):
            return self._eval_degree_condition(expr, data)
        elif isinstance(expr, AndExpr):
            return all(self._eval_where(op, data) for op in expr.operands)
        elif isinstance(expr, OrExpr):
            return any(self._eval_where(op, data) for op in expr.operands)
        elif isinstance(expr, NotExpr):
            return not self._eval_where(expr.operand, data)
        return True

    def _eval_condition(self, cond: Condition, data: dict) -> bool:
        """Evaluate a single condition."""
        actual = data.get(cond.field)
        expected = cond.value

        if expected is None:  # NULL check
            if cond.op == "=":
                return actual is None
            elif cond.op == "!=":
                return actual is not None

        if actual is None:
            return False  # NULL doesn't match non-NULL comparisons

        try:
            if cond.op == "=":
                return actual == expected
            elif cond.op == "!=":
                return actual != expected
            elif cond.op == ">":
                return actual > expected
            elif cond.op == "<":
                return actual < expected
            elif cond.op == ">=":
                return actual >= expected
            elif cond.op == "<=":
                return actual <= expected
        except TypeError:
            return False
        return False

    def _eval_degree_condition(self, cond: DegreeCondition, data: dict) -> bool:
        """Evaluate a degree condition. Requires node ID in data dict."""
        node_id = data.get("id")
        if node_id is None:
            return False

        slot = self._resolve_slot(node_id)
        if slot is None:
            return False

        if cond.degree_type == "INDEGREE":
            if cond.edge_kind:
                degree_arr = self.store.edge_matrices.in_degree(cond.edge_kind)
            else:
                # Sum in-degrees across all types
                total = 0
                for etype in self.store.edge_matrices.edge_types:
                    arr = self.store.edge_matrices.in_degree(etype)
                    if arr is not None and slot < len(arr):
                        total += int(arr[slot])
                return self._compare(total, cond.op, cond.value)
        else:  # OUTDEGREE
            if cond.edge_kind:
                degree_arr = self.store.edge_matrices.out_degree(cond.edge_kind)
            else:
                degree_arr = self.store.edge_matrices.out_degree(None)

        if degree_arr is None or slot >= len(degree_arr):
            return self._compare(0, cond.op, cond.value)

        return self._compare(int(degree_arr[slot]), cond.op, cond.value)

    def _compare(self, actual, op, expected) -> bool:
        if op == "=":
            return actual == expected
        if op == "!=":
            return actual != expected
        if op == ">":
            return actual > expected
        if op == "<":
            return actual < expected
        if op == ">=":
            return actual >= expected
        if op == "<=":
            return actual <= expected
        return False

    def _try_index_lookup(self, where, kind_filter: str | None) -> list[dict] | None:
        """Try to use secondary indices for O(1) equality lookups.

        Returns list of matching node dicts if index hit, None otherwise.
        """
        if where is None:
            return None
        expr = where.expr

        # Handle simple: WHERE field = value
        if isinstance(expr, Condition) and expr.op == "=" and expr.field != "kind":
            if expr.field in self.store._indexed_fields:
                slots = self.store.query_by_index(expr.field, expr.value)
                nodes = []
                for slot in slots:
                    node = self.store._materialize_slot(slot)
                    if node is None:
                        continue
                    if kind_filter and node["kind"] != kind_filter:
                        continue
                    nodes.append(node)
                return nodes

        # Handle AND with an indexed equality: WHERE kind = "X" AND name = "Y"
        if isinstance(expr, AndExpr):
            for op in expr.operands:
                if isinstance(op, Condition) and op.op == "=" and op.field != "kind":
                    if op.field in self.store._indexed_fields:
                        slots = self.store.query_by_index(op.field, op.value)
                        # Build remaining expression excluding the indexed condition
                        remaining_ops = [o for o in expr.operands if o is not op]
                        # Also strip kind (handled separately)
                        remaining_ops = [
                            o for o in remaining_ops
                            if not (isinstance(o, Condition) and o.field == "kind" and o.op == "=")
                        ]
                        nodes = []
                        for slot in slots:
                            node = self.store._materialize_slot(slot)
                            if node is None:
                                continue
                            if kind_filter and node["kind"] != kind_filter:
                                continue
                            # Apply remaining filters
                            if remaining_ops:
                                remaining_expr = remaining_ops[0] if len(remaining_ops) == 1 else AndExpr(operands=remaining_ops)
                                if not self._eval_where(remaining_expr, node):
                                    continue
                            nodes.append(node)
                        return nodes

        return None

    def _extract_kind_from_where(self, where) -> str | None:
        """Extract kind value from WHERE clause, including AND expressions."""
        if where is None:
            return None
        expr = where.expr
        if isinstance(expr, Condition) and expr.field == "kind" and expr.op == "=":
            return expr.value
        if isinstance(expr, AndExpr):
            for op in expr.operands:
                if isinstance(op, Condition) and op.field == "kind" and op.op == "=":
                    return op.value
        return None

    def _is_simple_kind_filter(self, where) -> bool:
        """Check if WHERE clause is just kind = 'x' with nothing else."""
        if where is None:
            return False
        expr = where.expr
        return isinstance(expr, Condition) and expr.field == "kind" and expr.op == "="

    def _strip_kind_from_expr(self, expr):
        """Remove kind='X' from expression, return remaining or None."""
        if isinstance(expr, Condition) and expr.field == "kind" and expr.op == "=":
            return None
        if isinstance(expr, AndExpr):
            remaining = [
                op for op in expr.operands
                if not (isinstance(op, Condition) and op.field == "kind" and op.op == "=")
            ]
            if len(remaining) == 0:
                return None
            if len(remaining) == 1:
                return remaining[0]
            return AndExpr(operands=remaining)
        return expr

    def _contains_degree_condition(self, expr) -> bool:
        """Check if expression tree contains any DegreeCondition."""
        if isinstance(expr, DegreeCondition):
            return True
        if isinstance(expr, AndExpr):
            return any(self._contains_degree_condition(op) for op in expr.operands)
        if isinstance(expr, OrExpr):
            return any(self._contains_degree_condition(op) for op in expr.operands)
        if isinstance(expr, NotExpr):
            return self._contains_degree_condition(expr.operand)
        return False

    def _references_synthetic_fields(self, expr) -> bool:
        """Check if expression references 'kind' or 'id' (stored in separate arrays)."""
        if isinstance(expr, (Condition, ContainsCondition, LikeCondition, InCondition)):
            return expr.field in ("kind", "id")
        if isinstance(expr, AndExpr):
            return any(self._references_synthetic_fields(op) for op in expr.operands)
        if isinstance(expr, OrExpr):
            return any(self._references_synthetic_fields(op) for op in expr.operands)
        if isinstance(expr, NotExpr):
            return self._references_synthetic_fields(expr.operand)
        return False

    def _make_raw_predicate(self, expr):
        """Build a callable(raw_data_dict) -> bool for slot-level filtering.

        Works against materialized data fields (no id/kind).
        Returns None if expression contains DegreeCondition or references
        synthetic fields (kind, id) that aren't in column data.
        """
        if self._contains_degree_condition(expr):
            return None
        if self._references_synthetic_fields(expr):
            return None
        return lambda data, _expr=expr: self._eval_where(_expr, data)

    # --- Column-accelerated query helpers ---

    def _try_column_filter(self, expr, base_mask: np.ndarray, n: int) -> np.ndarray | None:
        """Try to evaluate expression using column store. Returns bool mask or None."""
        columns = self.store.columns

        if isinstance(expr, Condition):
            if expr.field in ("kind", "id"):
                return None
            mask = columns.get_mask(expr.field, expr.op, expr.value, n)
            if mask is None:
                return None
            return mask & base_mask

        elif isinstance(expr, InCondition):
            if expr.field in ("kind", "id"):
                return None
            mask = columns.get_mask_in(expr.field, expr.values, n)
            if mask is None:
                return None
            return mask & base_mask

        elif isinstance(expr, AndExpr):
            result = base_mask.copy()
            for op in expr.operands:
                sub = self._try_column_filter(op, result, n)
                if sub is None:
                    return None
                result = sub
            return result

        elif isinstance(expr, OrExpr):
            result = np.zeros(n, dtype=bool)
            for op in expr.operands:
                sub = self._try_column_filter(op, base_mask, n)
                if sub is None:
                    return None
                result |= sub
            return result

        elif isinstance(expr, NotExpr):
            sub = self._try_column_filter(expr.operand, base_mask, n)
            if sub is None:
                return None
            fields = self._column_fields(expr.operand)
            if fields is None:
                return None
            combined_pres = np.ones(n, dtype=bool)
            for f in fields:
                fp = self.store.columns.get_presence(f, n)
                if fp is None:
                    return None
                combined_pres &= fp
            return ~sub & combined_pres & base_mask

        return None

    def _column_fields(self, expr) -> set[str] | None:
        """Extract field names from expression. None if non-columnarizable."""
        if isinstance(expr, (Condition, ContainsCondition, LikeCondition, InCondition)):
            if expr.field in ("kind", "id"):
                return None
            return {expr.field}
        if isinstance(expr, AndExpr):
            fields = set()
            for op in expr.operands:
                sub = self._column_fields(op)
                if sub is None:
                    return None
                fields |= sub
            return fields
        if isinstance(expr, OrExpr):
            fields = set()
            for op in expr.operands:
                sub = self._column_fields(op)
                if sub is None:
                    return None
                fields |= sub
            return fields
        if isinstance(expr, NotExpr):
            return self._column_fields(expr.operand)
        return None

    def _try_column_nodes(self, expr, kind_filter: str | None) -> list[dict] | None:
        """Try column-accelerated node query. Returns node dicts or None."""
        n = self.store._next_slot
        if n == 0:
            return []
        base_mask = self.store._live_mask(kind_filter)
        col_mask = self._try_column_filter(expr, base_mask, n)
        if col_mask is None:
            return None
        slots = np.nonzero(col_mask)[0]
        nodes = []
        for slot in slots:
            node = self.store._materialize_slot(int(slot))
            if node:
                nodes.append(node)
        return nodes

    def _try_column_count(self, expr, kind_filter: str | None) -> int | None:
        """Try column-accelerated count. Returns count or None."""
        n = self.store._next_slot
        if n == 0:
            return 0
        base_mask = self.store._live_mask(kind_filter)
        col_mask = self._try_column_filter(expr, base_mask, n)
        if col_mask is None:
            return None
        return int(np.count_nonzero(col_mask))

    def _try_column_delete_ids(self, expr, kind_filter: str | None) -> list[str] | None:
        """Try column-accelerated ID query for deletion. Returns IDs or None."""
        n = self.store._next_slot
        if n == 0:
            return []
        base_mask = self.store._live_mask(kind_filter)
        col_mask = self._try_column_filter(expr, base_mask, n)
        if col_mask is None:
            return None
        slots = np.nonzero(col_mask)[0]
        return [
            nid for slot in slots
            if (nid := self.store._slot_to_id(int(slot))) is not None
        ]

    def _try_column_order_by(self, nodes: list[dict], field: str,
                              descending: bool, limit: int | None,
                              offset: int | None) -> list[dict] | None:
        """Try column-accelerated ORDER BY using np.argpartition for top-K."""
        col_info = self.store.columns.get_column(field, self.store._next_slot)
        if col_info is None:
            return None
        col_data, col_pres, dtype_str = col_info

        if dtype_str == "int32_interned":
            return None

        slot_to_idx: dict[int, int] = {}
        for i, node in enumerate(nodes):
            slot = self._resolve_slot(node["id"])
            if slot is not None:
                slot_to_idx[slot] = i

        if not slot_to_idx:
            return nodes

        slots = np.array(list(slot_to_idx.keys()), dtype=np.int32)
        values = col_data[slots].astype(np.float64)
        present = col_pres[slots]

        if descending:
            values[~present] = -np.inf
        else:
            values[~present] = np.inf

        total = len(slots)
        eff_offset = (offset or 0)
        eff_limit = limit if limit is not None else total

        k = min(eff_offset + eff_limit, total)

        if k < total and k > 0:
            if descending:
                part_idx = np.argpartition(-values, k)[:k]
                sorted_idx = part_idx[np.argsort(-values[part_idx])]
            else:
                part_idx = np.argpartition(values, k)[:k]
                sorted_idx = part_idx[np.argsort(values[part_idx])]
        else:
            if descending:
                sorted_idx = np.argsort(-values)
            else:
                sorted_idx = np.argsort(values)

        sorted_idx = sorted_idx[eff_offset:eff_offset + eff_limit]

        result = []
        for idx in sorted_idx:
            slot = int(slots[idx])
            node_idx = slot_to_idx[slot]
            result.append(nodes[node_idx])
        return result

    def _order_slots_by_column(self, slots: np.ndarray, field: str,
                               descending: bool, limit: int | None,
                               offset: int | None,
                               fallback_predicate=None) -> np.ndarray | None:
        """Sort slot indices by column values using numpy, apply offset+limit.

        Returns sorted+sliced slot array, or None if column not available.
        When fallback_predicate is set, we can't use argpartition (need full sort)
        but still sort by column values.
        """
        col_info = self.store.columns.get_column(field, self.store._next_slot)
        if col_info is None:
            return None
        col_data, col_pres, dtype_str = col_info

        if dtype_str == "int32_interned":
            return None

        if len(slots) == 0:
            return slots

        values = col_data[slots].astype(np.float64)
        present = col_pres[slots]

        if descending:
            values[~present] = -np.inf
        else:
            values[~present] = np.inf

        total = len(slots)
        eff_offset = offset or 0
        eff_limit = limit if limit is not None else total

        if fallback_predicate is not None:
            # Can't partition when we need post-filter; do full sort
            if descending:
                sorted_idx = np.argsort(-values)
            else:
                sorted_idx = np.argsort(values)
            return slots[sorted_idx]

        k = min(eff_offset + eff_limit, total)

        if k < total and k > 0:
            if descending:
                part_idx = np.argpartition(-values, k)[:k]
                sorted_idx = part_idx[np.argsort(-values[part_idx])]
            else:
                part_idx = np.argpartition(values, k)[:k]
                sorted_idx = part_idx[np.argsort(values[part_idx])]
        else:
            if descending:
                sorted_idx = np.argsort(-values)
            else:
                sorted_idx = np.argsort(values)

        sorted_idx = sorted_idx[eff_offset:eff_offset + eff_limit]
        return slots[sorted_idx]

    def _materialize_slots_filtered(self, slots: np.ndarray, predicate=None) -> list[dict]:
        """Materialize slot array into node dicts, optionally filtering by predicate."""
        result = []
        for s in slots:
            node = self.store._materialize_slot(int(s))
            if node is not None:
                if predicate and not predicate(node):
                    continue
                result.append(node)
        return result

    def _extract_edge_type_from_expr(self, expr) -> str | None:
        """Extract edge type from an arrow expression."""
        if isinstance(expr, Condition) and expr.field == "kind" and expr.op == "=":
            return expr.value
        return None
