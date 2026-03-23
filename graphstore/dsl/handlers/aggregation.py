"""AGGREGATE query handlers."""

import numpy as np

from graphstore.dsl.handlers._registry import handles
from graphstore.dsl.ast_nodes import AggFunc, AggregateQuery
from graphstore.core.types import Result
from graphstore.core.errors import AggregationError


class AggregationHandlers:

    @handles(AggregateQuery)
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

        group_by = query.group_by or []
        select = query.select or []
        for field in group_by:
            if not self.store.columns.has_column(field):
                raise AggregationError(f"GROUP BY field '{field}' is not columnarized")
        for func in select:
            if func.field and not self.store.columns.has_column(func.field):
                raise AggregationError(f"Aggregate field '{func.field}' is not columnarized")

        filtered_count = int(np.sum(mask))
        if filtered_count == 0:
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

        if query.having:
            group_dicts = [d for d in group_dicts if self._eval_where(query.having, d)]

        if query.order_by:
            sort_key = query.order_by.label()
            group_dicts.sort(key=lambda d: d.get(sort_key, 0), reverse=query.order_desc)

        if query.limit:
            group_dicts = group_dicts[:query.limit.value]

        return Result(kind="aggregate", data=group_dicts, count=len(group_dicts))
