"""WHERE evaluation, column acceleration, and index helpers."""

import numpy as np

from graphstore.dsl.ast_nodes import (
    AndExpr,
    Condition,
    ContainsCondition,
    DegreeCondition,
    InCondition,
    LikeCondition,
    NotExpr,
    OrExpr,
)


class FilteringMixin:

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

    def _extract_edge_type_from_expr(self, expr) -> str | None:
        """Extract edge type from an arrow expression."""
        if isinstance(expr, Condition) and expr.field == "kind" and expr.op == "=":
            return expr.value
        return None

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
