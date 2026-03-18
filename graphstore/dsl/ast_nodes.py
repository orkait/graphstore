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

# --- Read queries ---
@dataclass
class NodeQuery:
    id: str

@dataclass
class NodesQuery:
    where: WhereClause | None = None
    limit: LimitClause | None = None

@dataclass
class EdgesQuery:
    direction: str   # "FROM" or "TO"
    node_id: str
    where: WhereClause | None = None

@dataclass
class TraverseQuery:
    start_id: str
    depth: int
    where: WhereClause | None = None

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
    id: str
    fields: list[FieldPair]

@dataclass
class UpdateNode:
    id: str
    fields: list[FieldPair]

@dataclass
class UpsertNode:
    id: str
    fields: list[FieldPair]

@dataclass
class DeleteNode:
    id: str

@dataclass
class DeleteNodes:
    where: WhereClause

@dataclass
class CreateEdge:
    source: str
    target: str
    fields: list[FieldPair]

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
    required: list[str]
    optional: list[str]

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
