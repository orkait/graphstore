"""DSL executor: maps parsed AST nodes to CoreStore operations.

Handles all user read and write queries. System queries are handled
separately by the server layer.
"""

import time


from graphstore.dsl.ast_nodes import (
    AncestorsQuery,
    AndExpr,
    Batch,
    CommonNeighborsQuery,
    Condition,
    ContainsCondition,
    LikeCondition,
    InCondition,
    CreateEdge,
    CreateNode,
    DegreeCondition,
    DeleteEdge,
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
        self.store._ensure_edges_built()
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
        }
        handler = handlers.get(type(ast))
        if handler is None:
            raise GraphStoreError(f"Unknown AST node type: {type(ast).__name__}")
        return handler(ast)

    # --- Read handlers ---

    def _node(self, q: NodeQuery) -> Result:
        data = self.store.get_node(q.id)
        return Result(kind="node", data=data, count=1 if data else 0)

    def _nodes(self, q: NodesQuery) -> Result:
        nodes = self.store.get_all_nodes()
        if q.where:
            nodes = [n for n in nodes if self._eval_where(q.where.expr, n)]
        if q.order:
            reverse = q.order.direction == "DESC"
            nodes.sort(key=lambda n: (n.get(q.order.field) is None, n.get(q.order.field, "")), reverse=reverse)
        if q.offset:
            nodes = nodes[q.offset.value:]
        if q.limit:
            nodes = nodes[:q.limit.value]
        return Result(kind="nodes", data=nodes, count=len(nodes))

    def _edges(self, q: EdgesQuery) -> Result:
        kind = self._extract_kind_from_where(q.where)
        if q.direction == "FROM":
            edges = self.store.get_edges_from(q.node_id, kind=kind)
        else:
            edges = self.store.get_edges_to(q.node_id, kind=kind)
        # Apply remaining filters if where has more than just kind
        if q.where and not self._is_simple_kind_filter(q.where):
            edges = [e for e in edges if self._eval_where(q.where.expr, e)]
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

        # Collect edges between visited nodes
        edges = []
        for node_idx in visited_slots:
            nid = self.store._slot_to_id(node_idx)
            if nid:
                for e in self.store.get_edges_from(nid):
                    tgt_id = e["target"]
                    tgt_slot = self._resolve_slot(tgt_id)
                    if tgt_slot in visited_slots:
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

        # Collect forward edges between all visited nodes (incl. anchor)
        edges = []
        for node_idx in visited_slots:
            nid = self.store._slot_to_id(node_idx)
            if nid:
                for e in self.store.get_edges_from(nid, kind=edge_type):
                    tgt_slot = self._resolve_slot(e["target"])
                    if tgt_slot in visited_slots:
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

        edges = []
        for node_idx in visited_slots:
            nid = self.store._slot_to_id(node_idx)
            if nid:
                for e in self.store.get_edges_from(nid, kind=edge_type):
                    tgt_slot = self._resolve_slot(e["target"])
                    if tgt_slot in visited_slots:
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
        return Result(kind="nodes", data=nodes, count=len(nodes))

    def _match(self, q: MatchQuery) -> Result:
        pattern = q.pattern

        # Cost check
        cost = estimate_match_cost(pattern, self.store.edge_matrices, threshold=self.cost_threshold)
        if cost.rejected:
            raise CostThresholdExceeded(cost.estimated_frontier, self.cost_threshold)

        # Execute match pattern using sparse matrix traversal
        bindings, edges = self._execute_match_pattern(pattern)

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
        node = self.store.get_node(q.id)
        return Result(kind="node", data=node, count=1)

    def _delete_node(self, q: DeleteNode) -> Result:
        self.store.delete_node(q.id)
        return Result(kind="ok", data={"id": q.id}, count=1)

    def _delete_nodes(self, q: DeleteNodes) -> Result:
        nodes = self.store.get_all_nodes()
        to_delete = [n for n in nodes if self._eval_where(q.where.expr, n)]
        deleted_ids = []
        for n in to_delete:
            try:
                self.store.delete_node(n["id"])
                deleted_ids.append(n["id"])
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
                nodes = self.store.get_all_nodes()
                count = sum(1 for n in nodes if self._eval_where(q.where.expr, n))
            else:
                count = self.store.node_count
        else:  # EDGES
            if q.where:
                edges = self.store.get_all_edges()
                count = sum(1 for e in edges if self._eval_where(q.where.expr, e))
            else:
                count = self.store.edge_count
        return Result(kind="count", data=count, count=count)

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

    def _batch(self, q: Batch) -> Result:
        """Execute batch with rollback on failure.

        Saves a copy of the store state before executing. If any statement
        fails, restores from the copy.
        """
        saved_edges = {k: list(v) for k, v in self.store._edges_by_type.items()}
        saved_edge_keys = set(self.store._edge_keys)
        saved_node_data = [
            dict(d) if d is not None else None
            for d in self.store.node_data[: self.store._next_slot]
        ]
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
            for i in range(saved_next_slot):
                self.store.node_data[i] = saved_node_data[i]
            self.store.node_tombstones = saved_tombstones
            self.store.id_to_slot = saved_id_to_slot
            self.store._count = saved_count
            self.store._next_slot = saved_next_slot
            self.store.node_ids[:saved_next_slot] = saved_node_ids
            self.store.node_kinds[:saved_next_slot] = saved_node_kinds
            self.store._rebuild_edges()
            raise BatchRollback(
                failed_statement=str(type(e).__name__), error=str(e)
            )

    # --- Helpers ---

    def _resolve_slot(self, node_id: str) -> int | None:
        """Resolve a string node ID to its slot index."""
        if node_id not in self.store.string_table:
            return None
        str_id = self.store.string_table.intern(node_id)
        slot = self.store.id_to_slot.get(str_id)
        if slot is None or slot in self.store.node_tombstones:
            return None
        return slot

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
            import re
            # Convert SQL LIKE pattern to regex: % -> .*, _ -> .
            # Replace wildcards first, escape the rest, then restore wildcards
            parts = []
            i = 0
            p = expr.pattern
            while i < len(p):
                if p[i] == '%':
                    parts.append('.*')
                elif p[i] == '_':
                    parts.append('.')
                else:
                    parts.append(re.escape(p[i]))
                i += 1
            pattern = ''.join(parts)
            return bool(re.fullmatch(pattern, str(actual)))
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

    def _extract_kind_from_where(self, where) -> str | None:
        """Extract kind value from a simple WHERE kind = 'x' clause."""
        if where is None:
            return None
        expr = where.expr
        if isinstance(expr, Condition) and expr.field == "kind" and expr.op == "=":
            return expr.value
        return None

    def _is_simple_kind_filter(self, where) -> bool:
        """Check if WHERE clause is just kind = 'x'."""
        if where is None:
            return False
        expr = where.expr
        return isinstance(expr, Condition) and expr.field == "kind" and expr.op == "="

    def _extract_edge_type_from_expr(self, expr) -> str | None:
        """Extract edge type from an arrow expression."""
        if isinstance(expr, Condition) and expr.field == "kind" and expr.op == "=":
            return expr.value
        return None
