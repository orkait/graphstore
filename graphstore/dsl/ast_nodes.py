from __future__ import annotations
from dataclasses import dataclass
from typing import Any

# --- Filter expressions ---
@dataclass
class Condition:
    field: str
    op: str      # "=", "!=", ">", "<", ">=", "<="
    value: Any   # str, int, float, or None (for NULL)

@dataclass
class ContainsCondition:
    field: str
    value: str

@dataclass
class LikeCondition:
    field: str
    pattern: str  # SQL-like: % = any, _ = single char

@dataclass
class InCondition:
    field: str
    values: list

@dataclass
class DegreeCondition:
    degree_type: str   # "INDEGREE" or "OUTDEGREE"
    edge_kind: str | None  # optional edge kind filter
    op: str
    value: int | float

@dataclass
class NotExpr:
    operand: Any

@dataclass
class AndExpr:
    operands: list

@dataclass
class OrExpr:
    operands: list

@dataclass
class WhereClause:
    expr: Any

@dataclass
class LimitClause:
    value: int

@dataclass
class OffsetClause:
    value: int

@dataclass
class OrderClause:
    field: str
    direction: str = "ASC"  # "ASC" or "DESC"

@dataclass
class AggFunc:
    func: str       # "COUNT", "COUNT_DISTINCT", "SUM", "AVG", "MIN", "MAX"
    field: str | None  # None for COUNT()

    def label(self) -> str:
        if self.field is None:
            return f"{self.func}()"
        return f"{self.func}({self.field})"

@dataclass
class AggregateQuery:
    where: WhereClause | None = None
    group_by: list[str] | None = None
    select: list[AggFunc] | None = None
    having: Any | None = None
    order_by: AggFunc | None = None
    order_desc: bool = False
    limit: LimitClause | None = None

# --- Read queries ---
@dataclass
class NodeQuery:
    id: str
    with_document: bool = False

@dataclass
class NodesQuery:
    where: WhereClause | None = None
    order: OrderClause | None = None
    limit: LimitClause | None = None
    offset: OffsetClause | None = None

@dataclass
class EdgesQuery:
    direction: str   # "FROM" or "TO"
    node_id: str
    where: WhereClause | None = None
    limit: LimitClause | None = None

@dataclass
class CountQuery:
    target: str  # "NODES" or "EDGES"
    where: WhereClause | None = None

@dataclass
class TraverseQuery:
    start_id: str
    depth: int
    where: WhereClause | None = None
    limit: LimitClause | None = None

@dataclass
class SubgraphQuery:
    start_id: str
    depth: int

@dataclass
class PathQuery:
    from_id: str
    to_id: str
    max_depth: int
    where: WhereClause | None = None

@dataclass
class PathsQuery:
    from_id: str
    to_id: str
    max_depth: int
    where: WhereClause | None = None

@dataclass
class ShortestPathQuery:
    from_id: str
    to_id: str
    where: WhereClause | None = None

@dataclass
class DistanceQuery:
    from_id: str
    to_id: str
    max_depth: int

@dataclass
class WeightedShortestPathQuery:
    from_id: str
    to_id: str
    where: WhereClause | None = None

@dataclass
class WeightedDistanceQuery:
    from_id: str
    to_id: str

@dataclass
class AncestorsQuery:
    node_id: str
    depth: int
    where: WhereClause | None = None

@dataclass
class DescendantsQuery:
    node_id: str
    depth: int
    where: WhereClause | None = None

@dataclass
class CommonNeighborsQuery:
    node_a: str
    node_b: str
    where: WhereClause | None = None

# --- Pattern matching ---
@dataclass
class PatternStep:
    """A step in a MATCH pattern - either a bound ID or a variable with optional filter."""
    bound_id: str | None = None      # if this is a literal string ID
    variable: str | None = None      # if this is a variable binding
    where: Any | None = None         # optional filter for variable steps

@dataclass
class PatternArrow:
    """An edge constraint in a MATCH pattern."""
    expr: Any   # filter expression (typically kind = "something")

@dataclass
class MatchPattern:
    """A full MATCH pattern: step (arrow step)+"""
    steps: list[PatternStep]
    arrows: list[PatternArrow]

@dataclass
class MatchQuery:
    pattern: MatchPattern
    limit: LimitClause | None = None

# --- Write queries ---
@dataclass
class FieldPair:
    name: str
    value: Any

@dataclass
class CreateNode:
    id: str | None  # None for AUTO ID
    fields: list[FieldPair]
    auto_id: bool = False
    expires_in: tuple[int, str] | None = None   # (amount, unit) e.g. (30, "m")
    expires_at: str | None = None                # ISO-8601 string
    vector: list[float] | None = None
    document: str | None = None

@dataclass
class VarAssign:
    variable: str  # e.g. "$fn1"
    statement: Any  # the write query that produces an ID

@dataclass
class UpdateNode:
    id: str
    fields: list[FieldPair]

@dataclass
class UpsertNode:
    id: str
    fields: list[FieldPair]
    expires_in: tuple[int, str] | None = None
    expires_at: str | None = None
    vector: list[float] | None = None

@dataclass
class DeleteNode:
    id: str

@dataclass
class DeleteNodes:
    where: WhereClause

@dataclass
class CreateEdge:
    source: str  # literal ID or "$variable"
    target: str  # literal ID or "$variable"
    fields: list[FieldPair]

@dataclass
class UpdateEdge:
    source: str
    target: str
    fields: list[FieldPair]
    where: WhereClause | None = None

@dataclass
class DeleteEdge:
    source: str
    target: str
    where: WhereClause | None = None

@dataclass
class DeleteEdges:
    direction: str  # "FROM" or "TO"
    node_id: str
    where: WhereClause | None = None

@dataclass
class Increment:
    node_id: str
    field: str
    amount: int | float

@dataclass
class Batch:
    statements: list  # list of write queries

@dataclass
class AssertStmt:
    id: str
    fields: list[FieldPair]
    confidence: float | None = None
    source: str | None = None

@dataclass
class RetractStmt:
    id: str
    reason: str | None = None

@dataclass
class UpdateNodes:
    where: WhereClause
    fields: list[FieldPair]

@dataclass
class MergeStmt:
    source_id: str
    target_id: str

@dataclass
class PropagateStmt:
    node_id: str
    field: str
    depth: int

@dataclass
class BindContext:
    name: str

@dataclass
class DiscardContext:
    name: str

@dataclass
class IngestStmt:
    file_path: str
    node_id: str | None = None
    kind: str | None = None
    using: str | None = None
    vision_model: str | None = None

# --- Read queries (intelligence) ---

@dataclass
class RecallQuery:
    node_id: str
    depth: int
    limit: LimitClause | None = None
    where: WhereClause | None = None

@dataclass
class CounterfactualQuery:
    node_id: str

@dataclass
class SimilarQuery:
    target_vector: list[float] | None = None
    target_text: str | None = None
    target_node_id: str | None = None
    limit: LimitClause | None = None
    where: WhereClause | None = None

# --- System queries ---
@dataclass
class SysStats:
    target: str | None = None  # "NODES", "EDGES", "MEMORY", "WAL", or None for all

@dataclass
class SysKinds:
    pass

@dataclass
class SysEdgeKinds:
    pass

@dataclass
class SysDescribe:
    entity_type: str  # "NODE" or "EDGE"
    name: str

@dataclass
class SysSlowQueries:
    since: str | None = None
    limit: LimitClause | None = None

@dataclass
class SysFrequentQueries:
    limit: LimitClause | None = None

@dataclass
class SysFailedQueries:
    limit: LimitClause | None = None

@dataclass
class SysExplain:
    query: Any  # the read query to explain

@dataclass
class SysRegisterNodeKind:
    kind: str
    required: list[tuple[str, str | None]]  # [(field_name, type_name_or_none), ...]
    optional: list[tuple[str, str | None]]
    embed_field: str | None = None

@dataclass
class SysRegisterEdgeKind:
    kind: str
    from_kinds: list[str]
    to_kinds: list[str]

@dataclass
class SysUnregister:
    entity_type: str  # "NODE" or "EDGE"
    kind: str

@dataclass
class SysCheckpoint:
    pass

@dataclass
class SysRebuild:
    pass

@dataclass
class SysClear:
    target: str  # "LOG" or "CACHE"

@dataclass
class SysWal:
    action: str  # "STATUS" or "REPLAY"

@dataclass
class SysExpire:
    where: WhereClause | None = None

@dataclass
class SysContradictions:
    where: WhereClause | None = None
    field: str = ""
    group_by: str = ""

@dataclass
class SysSnapshot:
    name: str

@dataclass
class SysRollback:
    name: str

@dataclass
class SysSnapshots:
    pass

@dataclass
class SysDuplicates:
    where: WhereClause | None = None
    threshold: float = 0.95

@dataclass
class SysEmbedders:
    pass

@dataclass
class SysConnect:
    where: WhereClause | None = None
    threshold: float = 0.85

@dataclass
class ConnectNode:
    node_id: str
    threshold: float = 0.8

@dataclass
class SysReembed:
    pass

@dataclass
class SysStatus:
    pass
