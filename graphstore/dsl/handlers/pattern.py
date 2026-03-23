"""MATCH pattern handlers."""

from graphstore.dsl.handlers._registry import handles
from graphstore.dsl.ast_nodes import MatchQuery, MatchPattern
from graphstore.core.types import Result
from graphstore.core.errors import CostThresholdExceeded
from graphstore.dsl.cost_estimator import estimate_match_cost


class PatternHandlers:

    @handles(MatchQuery)
    def _match(self, q: MatchQuery) -> Result:
        pattern = q.pattern

        cost = estimate_match_cost(pattern, self.store.edge_matrices, threshold=self.cost_threshold)
        if cost.rejected:
            raise CostThresholdExceeded(cost.estimated_frontier, self.cost_threshold)

        bindings, edges = self._execute_match_pattern(pattern)

        bindings = [
            b for b in bindings
            if all(self._is_visible_by_id(nid) for nid in b.values())
        ]
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

        first_step = steps[0]
        if first_step.bound_id:
            start_slot = self._resolve_slot(first_step.bound_id)
            if start_slot is None:
                return [], []
            current_slots = [start_slot]
        else:
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

        paths = [[] for _ in current_slots]
        edge_trails = [[] for _ in current_slots]

        for i, slot in enumerate(current_slots):
            if first_step.variable:
                nid = self.store._slot_to_id(slot)
                paths[i].append((first_step.variable, nid))
            elif first_step.bound_id:
                paths[i].append(("_start", first_step.bound_id))

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

            if len(current_slots) > 1000:
                current_slots = current_slots[:1000]
                paths = paths[:1000]
                edge_trails = edge_trails[:1000]

        bindings = []
        for path in paths:
            binding = {}
            for var_name, node_id in path:
                if not var_name.startswith("_"):
                    binding[var_name] = node_id
            bindings.append(binding)

        seen_edges: dict[str, dict] = {}
        for trail in edge_trails:
            for e in trail:
                key = f"{e['source']}->{e['target']}:{e['kind']}"
                seen_edges[key] = e

        return bindings, list(seen_edges.values())
