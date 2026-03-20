import time as _time
from datetime import datetime, timedelta

from lark import Transformer, Token
from graphstore.dsl.ast_nodes import (
    Condition,
    ContainsCondition,
    LikeCondition,
    InCondition,
    DegreeCondition,
    NotExpr,
    AndExpr,
    OrExpr,
    WhereClause,
    LimitClause,
    NodeQuery,
    NodesQuery,
    EdgesQuery,
    TraverseQuery,
    SubgraphQuery,
    PathQuery,
    PathsQuery,
    ShortestPathQuery,
    DistanceQuery,
    WeightedShortestPathQuery,
    WeightedDistanceQuery,
    AncestorsQuery,
    DescendantsQuery,
    CommonNeighborsQuery,
    PatternStep,
    PatternArrow,
    MatchPattern,
    MatchQuery,
    AggFunc,
    AggregateQuery,
    FieldPair,
    CreateNode,
    VarAssign,
    UpdateNode,
    UpsertNode,
    DeleteNode,
    DeleteNodes,
    CreateEdge,
    UpdateEdge,
    DeleteEdge,
    DeleteEdges,
    OffsetClause,
    OrderClause,
    CountQuery,
    Increment,
    Batch,
    AssertStmt,
    RetractStmt,
    UpdateNodes,
    MergeStmt,
    SysStats,
    SysKinds,
    SysEdgeKinds,
    SysDescribe,
    SysSlowQueries,
    SysFrequentQueries,
    SysFailedQueries,
    SysExplain,
    SysRegisterNodeKind,
    SysRegisterEdgeKind,
    SysUnregister,
    SysCheckpoint,
    SysRebuild,
    SysClear,
    SysWal,
    SysExpire,
    SysContradictions,
)



class DSLTransformer(Transformer):
    def start(self, args):
        return args[0]

    def statement(self, args):
        return args[0]

    def user_query(self, args):
        return args[0]

    def read_query(self, args):
        return args[0]

    def write_query(self, args):
        return args[0]

    def system_query(self, args):
        return args[0]

    def sys_command(self, args):
        return args[0]

    # --- Values ---
    def val_string(self, args):
        s = str(args[0])
        if s.startswith('"') and s.endswith('"'):
            s = s[1:-1]
        return s.replace('\\"', '"').replace('\\\\', '\\')

    def val_number(self, args):
        s = str(args[0])
        return float(s) if '.' in s else int(s)

    def val_null(self, args):
        return None

    # --- Time expressions ---
    def time_now(self, _items):
        return int(_time.time() * 1000)

    def time_offset(self, items):
        n = int(float(str(items[0])))
        unit = str(items[1])
        ms = {"s": 1000, "m": 60000, "h": 3600000, "d": 86400000}[unit]
        return int(_time.time() * 1000) - n * ms

    def time_today(self, _items):
        midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        return int(midnight.timestamp() * 1000)

    def time_yesterday(self, _items):
        midnight = (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return int(midnight.timestamp() * 1000)

    def time_expr(self, items):
        return items[0]  # unwrap - the inner rule already returns the int value

    def STRING(self, token):
        return token

    def NUMBER(self, token):
        return token

    def IDENTIFIER(self, token):
        return str(token)

    # --- Read queries ---
    def node_q(self, args):
        return NodeQuery(id=self._str(args[0]))

    def nodes_q(self, args):
        where = self._find(args, WhereClause)
        order = self._find(args, OrderClause)
        limit = self._find(args, LimitClause)
        offset = self._find(args, OffsetClause)
        return NodesQuery(where=where, order=order, limit=limit, offset=offset)

    def edges_q(self, args):
        direction = args[0]  # "FROM" or "TO"
        node_id = self._str(args[1])
        where = self._find(args[2:], WhereClause)
        limit = self._find(args[2:], LimitClause)
        return EdgesQuery(direction=direction, node_id=node_id, where=where, limit=limit)

    def traverse_q(self, args):
        return TraverseQuery(
            start_id=self._str(args[0]),
            depth=self._num(args[1]),
            where=self._find(args[2:], WhereClause),
            limit=self._find(args[2:], LimitClause),
        )

    def subgraph_q(self, args):
        return SubgraphQuery(start_id=self._str(args[0]), depth=self._num(args[1]))

    def path_q(self, args):
        return PathQuery(
            from_id=self._str(args[0]),
            to_id=self._str(args[1]),
            max_depth=self._num(args[2]),
            where=self._find(args[3:], WhereClause),
        )

    def paths_q(self, args):
        return PathsQuery(
            from_id=self._str(args[0]),
            to_id=self._str(args[1]),
            max_depth=self._num(args[2]),
            where=self._find(args[3:], WhereClause),
        )

    def shortest_q(self, args):
        return ShortestPathQuery(
            from_id=self._str(args[0]),
            to_id=self._str(args[1]),
            where=self._find(args[2:], WhereClause),
        )

    def distance_q(self, args):
        return DistanceQuery(
            from_id=self._str(args[0]),
            to_id=self._str(args[1]),
            max_depth=self._num(args[2]),
        )

    def weighted_sp_q(self, args):
        return WeightedShortestPathQuery(
            from_id=self._str(args[0]),
            to_id=self._str(args[1]),
            where=self._find(args[2:], WhereClause),
        )

    def weighted_dist_q(self, args):
        return WeightedDistanceQuery(
            from_id=self._str(args[0]),
            to_id=self._str(args[1]),
        )

    def ancestors_q(self, args):
        return AncestorsQuery(
            node_id=self._str(args[0]),
            depth=self._num(args[1]),
            where=self._find(args[2:], WhereClause),
        )

    def descendants_q(self, args):
        return DescendantsQuery(
            node_id=self._str(args[0]),
            depth=self._num(args[1]),
            where=self._find(args[2:], WhereClause),
        )

    def common_q(self, args):
        return CommonNeighborsQuery(
            node_a=self._str(args[0]),
            node_b=self._str(args[1]),
            where=self._find(args[2:], WhereClause),
        )

    # --- Direction ---
    def dir_from(self, args):
        return "FROM"

    def dir_to(self, args):
        return "TO"

    # --- Pattern matching ---
    def match_q(self, args):
        pattern = args[0]
        limit = self._find(args[1:], LimitClause)
        return MatchQuery(pattern=pattern, limit=limit)

    def pattern(self, args):
        steps = []
        arrows = []
        for a in args:
            if isinstance(a, PatternStep):
                steps.append(a)
            elif isinstance(a, PatternArrow):
                arrows.append(a)
        return MatchPattern(steps=steps, arrows=arrows)

    def bound_step(self, args):
        return PatternStep(bound_id=self._str(args[0]))

    def var_step(self, args):
        var_name = str(args[0])
        where = args[1] if len(args) > 1 else None
        return PatternStep(variable=var_name, where=where)

    def step_where(self, args):
        return args[0]

    def arrow(self, args):
        return PatternArrow(expr=args[0])

    # --- Writes ---
    def create_node(self, args):
        fields = args[1] if isinstance(args[1], list) else []
        exp_in, exp_at = self._find_expires(args[2:])
        return CreateNode(
            id=self._str(args[0]),
            fields=fields,
            expires_in=exp_in,
            expires_at=exp_at,
        )

    def create_node_auto(self, args):
        fields = args[0] if isinstance(args[0], list) else []
        exp_in, exp_at = self._find_expires(args[1:])
        return CreateNode(
            id=None,
            fields=fields,
            auto_id=True,
            expires_in=exp_in,
            expires_at=exp_at,
        )

    def var_assign(self, args):
        var_name = str(args[0])
        stmt = args[1]
        return VarAssign(variable=var_name, statement=stmt)

    def node_ref(self, args):
        """Return either a literal string or a $variable reference."""
        token = args[0]
        s = str(token)
        if s.startswith('$'):
            return s  # variable reference, kept as-is
        # Strip quotes from string literal
        if s.startswith('"') and s.endswith('"'):
            return s[1:-1].replace('\\"', '"').replace('\\\\', '\\')
        return s

    def update_node(self, args):
        return UpdateNode(
            id=self._str(args[0]),
            fields=args[1] if isinstance(args[1], list) else [],
        )

    def upsert_node(self, args):
        fields = args[1] if isinstance(args[1], list) else []
        exp_in, exp_at = self._find_expires(args[2:])
        return UpsertNode(
            id=self._str(args[0]),
            fields=fields,
            expires_in=exp_in,
            expires_at=exp_at,
        )

    def expires_in(self, args):
        return ("expires_in", self._num(args[0]), str(args[1]))

    def expires_at(self, args):
        return ("expires_at", self._str(args[0]))

    def assert_stmt(self, args):
        node_id = self._str(args[0])
        fields = args[1] if isinstance(args[1], list) else []
        confidence = None
        source = None
        for a in args[2:]:
            if isinstance(a, tuple) and a[0] == "confidence":
                confidence = a[1]
            elif isinstance(a, tuple) and a[0] == "source":
                source = a[1]
        return AssertStmt(id=node_id, fields=fields, confidence=confidence, source=source)

    def confidence_clause(self, args):
        return ("confidence", self._num(args[0]))

    def source_clause(self, args):
        return ("source", self._str(args[0]))

    def retract_stmt(self, args):
        node_id = self._str(args[0])
        reason = None
        for a in args[1:]:
            if isinstance(a, tuple) and a[0] == "reason":
                reason = a[1]
        return RetractStmt(id=node_id, reason=reason)

    def reason_clause(self, args):
        return ("reason", self._str(args[0]))

    def update_nodes(self, args):
        where = args[0]
        fields = args[1] if isinstance(args[1], list) else []
        return UpdateNodes(where=where, fields=fields)

    def merge_stmt(self, args):
        return MergeStmt(source_id=self._str(args[0]), target_id=self._str(args[1]))

    def delete_node(self, args):
        return DeleteNode(id=self._str(args[0]))

    def delete_nodes(self, args):
        return DeleteNodes(where=args[0])

    def create_edge(self, args):
        return CreateEdge(
            source=args[0],  # node_ref returns str directly
            target=args[1],
            fields=args[2] if len(args) > 2 and isinstance(args[2], list) else [],
        )

    def delete_edge(self, args):
        return DeleteEdge(
            source=self._str(args[0]),
            target=self._str(args[1]),
            where=self._find(args[2:], WhereClause),
        )

    def delete_edges(self, args):
        direction = args[0]
        node_id = self._str(args[1])
        where = self._find(args[2:], WhereClause)
        return DeleteEdges(direction=direction, node_id=node_id, where=where)

    def increment(self, args):
        return Increment(
            node_id=self._str(args[0]),
            field=str(args[1]),
            amount=self._num(args[2]),
        )

    def batch(self, args):
        stmts = [a for a in args if not isinstance(a, Token) or a.type != "NEWLINE"]
        return Batch(statements=stmts)

    def field_pairs(self, args):
        pairs = []
        i = 0
        while i < len(args):
            name = str(args[i])
            value = args[i + 1]
            pairs.append(FieldPair(name=name, value=value))
            i += 2
        return pairs

    # --- Filters ---
    def where_clause(self, args):
        return WhereClause(expr=args[0])

    def expr(self, args):
        return args[0]

    def or_expr(self, args):
        if len(args) == 1:
            return args[0]
        # Right-recursive: left_operand, right (which may itself be OrExpr)
        left, right = args[0], args[1]
        # Flatten nested OrExpr on the right
        if isinstance(right, OrExpr):
            return OrExpr(operands=[left] + right.operands)
        return OrExpr(operands=[left, right])

    def and_expr(self, args):
        if len(args) == 1:
            return args[0]
        left, right = args[0], args[1]
        if isinstance(right, AndExpr):
            return AndExpr(operands=[left] + right.operands)
        return AndExpr(operands=[left, right])

    def not_term(self, args):
        return NotExpr(operand=args[0])

    def not_expr(self, args):
        return args[0]

    def condition(self, args):
        return Condition(field=str(args[0]), op=str(args[1]), value=args[2])

    def contains_cond(self, args):
        return ContainsCondition(field=str(args[0]), value=self._str(args[1]))

    def like_cond(self, args):
        return LikeCondition(field=str(args[0]), pattern=self._str(args[1]))

    def in_cond(self, args):
        field = str(args[0])
        values = list(args[1:])
        return InCondition(field=field, values=values)

    def degree_condition(self, args):
        degree_type = args[0]
        rest = args[1:]
        if len(rest) == 3:
            return DegreeCondition(
                degree_type=degree_type,
                edge_kind=str(rest[0]),
                op=str(rest[1]),
                value=self._num(rest[2]),
            )
        else:
            return DegreeCondition(
                degree_type=degree_type,
                edge_kind=None,
                op=str(rest[0]),
                value=self._num(rest[1]),
            )

    def indegree(self, args):
        return "INDEGREE"

    def outdegree(self, args):
        return "OUTDEGREE"

    def limit_clause(self, args):
        return LimitClause(value=self._num(args[0]))

    def offset_clause(self, args):
        return OffsetClause(value=self._num(args[0]))

    def order_clause(self, args):
        field = str(args[0])
        direction = args[1] if len(args) > 1 else "ASC"
        return OrderClause(field=field, direction=direction)

    def order_asc(self, args):
        return "ASC"

    def order_desc(self, args):
        return "DESC"

    def count_q(self, args):
        target = args[0]
        where = self._find(args[1:], WhereClause)
        return CountQuery(target=target, where=where)

    def count_nodes(self, args):
        return "NODES"

    def count_edges(self, args):
        return "EDGES"

    # --- Aggregate queries ---
    def aggregate_q(self, items):
        where = None
        group_by = []
        select = []
        having = None
        order_by = None
        order_desc = False
        limit = None
        for item in items:
            if isinstance(item, WhereClause):
                where = item
            elif isinstance(item, list) and item and isinstance(item[0], str):
                group_by = item
            elif isinstance(item, list) and item and isinstance(item[0], AggFunc):
                select = item
            elif isinstance(item, LimitClause):
                limit = item
            elif isinstance(item, AggFunc):
                order_by = item
            elif isinstance(item, tuple) and len(item) == 2:
                order_by, order_desc = item
            # having is an expression (Condition, AndExpr, etc.)
            elif item is not None and not isinstance(item, (WhereClause, LimitClause, AggFunc, list)):
                having = item
        return AggregateQuery(where=where, group_by=group_by, select=select,
                              having=having, order_by=order_by, order_desc=order_desc, limit=limit)

    def group_clause(self, items):
        return [str(i) for i in items]

    def select_clause(self, items):
        return list(items)

    def having_clause(self, items):
        return items[0]  # the having_expr (a Condition)

    def having_expr(self, items):
        agg = items[0]  # AggFunc
        op = str(items[1])
        val = items[2]
        return Condition(field=agg.label(), op=op, value=val)

    def order_agg_clause(self, items):
        agg = items[0]
        desc = len(items) > 1 and items[1] == "DESC"
        return (agg, desc)

    def agg_func_ref(self, items):
        return items[0]

    def agg_count(self, _items):
        return AggFunc("COUNT", None)

    def agg_count_distinct(self, items):
        return AggFunc("COUNT_DISTINCT", str(items[0]))

    def agg_sum(self, items):
        return AggFunc("SUM", str(items[0]))

    def agg_avg(self, items):
        return AggFunc("AVG", str(items[0]))

    def agg_min(self, items):
        return AggFunc("MIN", str(items[0]))

    def agg_max(self, items):
        return AggFunc("MAX", str(items[0]))

    def update_edge(self, args):
        return UpdateEdge(
            source=self._str(args[0]),
            target=self._str(args[1]),
            fields=args[2] if len(args) > 2 and isinstance(args[2], list) else [],
            where=self._find(args[2:], WhereClause),
        )

    # --- System queries ---
    def sys_stats(self, args):
        target = str(args[0]) if args else None
        return SysStats(target=target)

    def sys_kinds(self, args):
        return SysKinds()

    def sys_edge_kinds(self, args):
        return SysEdgeKinds()

    def sys_describe(self, args):
        entity_type = str(args[0])
        name = self._str(args[1])
        return SysDescribe(entity_type=entity_type, name=name)

    def sys_slow(self, args):
        since = None
        limit = None
        for a in args:
            if isinstance(a, str):
                since = a
            elif isinstance(a, LimitClause):
                limit = a
        return SysSlowQueries(since=since, limit=limit)

    def sys_frequent(self, args):
        return SysFrequentQueries(limit=self._find(args, LimitClause))

    def sys_failed(self, args):
        return SysFailedQueries(limit=self._find(args, LimitClause))

    def since_clause(self, args):
        return self._str(args[0])

    def sys_explain(self, args):
        return SysExplain(query=args[0])

    def sys_register_node_kind(self, args):
        kind = self._str(args[0])
        required = args[1]  # ident_list
        optional = args[2] if len(args) > 2 else []
        return SysRegisterNodeKind(kind=kind, required=required, optional=optional)

    def sys_register_edge_kind(self, args):
        kind = self._str(args[0])
        from_kinds = args[1]
        to_kinds = args[2]
        return SysRegisterEdgeKind(kind=kind, from_kinds=from_kinds, to_kinds=to_kinds)

    def optional_clause(self, args):
        return args[0]

    def sys_unregister(self, args):
        entity_type = str(args[0])
        kind = self._str(args[1])
        return SysUnregister(entity_type=entity_type, kind=kind)

    def sys_checkpoint(self, args):
        return SysCheckpoint()

    def sys_rebuild(self, args):
        return SysRebuild()

    def sys_clear(self, args):
        target = str(args[0])
        return SysClear(target=target)

    def sys_wal(self, args):
        action = str(args[0])
        return SysWal(action=action)

    def sys_expire(self, args):
        where = self._find(args, WhereClause)
        return SysExpire(where=where)

    def sys_contradictions(self, args):
        where = self._find(args, WhereClause)
        # The last two args are IDENTIFIER tokens: field, group_by
        identifiers = [str(a) for a in args if isinstance(a, str) and not isinstance(a, Token)]
        # Actually, IDENTIFIER tokens are strings after transformation
        # Filter out WhereClause and collect remaining string identifiers
        idents = []
        for a in args:
            if isinstance(a, WhereClause):
                continue
            if isinstance(a, (str, Token)) and not isinstance(a, WhereClause):
                idents.append(str(a))
        # idents should be [field, group_by_field]
        field = idents[0] if len(idents) >= 1 else ""
        group_by = idents[1] if len(idents) >= 2 else ""
        return SysContradictions(where=where, field=field, group_by=group_by)

    def typed_ident_with_type(self, args):
        return (str(args[0]), str(args[1]))

    def typed_ident_bare(self, args):
        return (str(args[0]), None)

    def ident_list(self, args):
        return list(args)

    def string_list(self, args):
        return [self._str(a) for a in args]

    # --- Helpers ---
    def _str(self, token) -> str:
        """Extract string value from token, stripping quotes."""
        s = str(token)
        if s.startswith('"') and s.endswith('"'):
            return s[1:-1].replace('\\"', '"').replace('\\\\', '\\')
        return s

    def _num(self, token) -> int | float:
        s = str(token)
        return float(s) if '.' in s else int(s)

    def _find(self, args, cls):
        """Find first instance of cls in args."""
        for a in args:
            if isinstance(a, cls):
                return a
        return None

    def _find_expires(self, args):
        """Extract expires_in or expires_at from args. Returns (exp_in, exp_at)."""
        exp_in = None
        exp_at = None
        for a in args:
            if isinstance(a, tuple):
                if a[0] == "expires_in":
                    exp_in = (int(a[1]), a[2])
                elif a[0] == "expires_at":
                    exp_at = a[1]
        return exp_in, exp_at
