"""Handler registry and domain-specific handler mixins.

Importing this package triggers @handles registration for all handlers.
"""

from graphstore.dsl.handlers._registry import DISPATCH, WRITE_OPS, is_write_op
from graphstore.dsl.handlers.nodes import NodeHandlers
from graphstore.dsl.handlers.edges import EdgeHandlers
from graphstore.dsl.handlers.traversal import TraversalHandlers
from graphstore.dsl.handlers.pattern import PatternHandlers
from graphstore.dsl.handlers.aggregation import AggregationHandlers
from graphstore.dsl.handlers.intelligence import IntelligenceHandlers
from graphstore.dsl.handlers.beliefs import BeliefHandlers
from graphstore.dsl.handlers.mutations import MutationHandlers
from graphstore.dsl.handlers.context import ContextHandlers
from graphstore.dsl.handlers.ingest import IngestHandlers

__all__ = [
    "DISPATCH", "WRITE_OPS", "is_write_op",
    "NodeHandlers", "EdgeHandlers", "TraversalHandlers",
    "PatternHandlers", "AggregationHandlers", "IntelligenceHandlers",
    "BeliefHandlers", "MutationHandlers", "ContextHandlers", "IngestHandlers",
]
