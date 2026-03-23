"""DSL executor: maps parsed AST nodes to CoreStore operations.

Handles all user read and write queries. System queries are handled
separately by the server layer.
"""

from graphstore.dsl.ast_nodes import (
    AggregateQuery,
    AncestorsQuery,
    AssertStmt,
    Batch,
    BindContext,
    CommonNeighborsQuery,
    ConnectNode,
    CountQuery,
    CounterfactualQuery,
    CreateEdge,
    CreateNode,
    DeleteEdge,
    DeleteEdges,
    DeleteNode,
    DeleteNodes,
    DescendantsQuery,
    DiscardContext,
    DistanceQuery,
    EdgesQuery,
    ForgetNode,
    Increment,
    IngestStmt,
    LexicalSearchQuery,
    MatchQuery,
    MergeStmt,
    NodeQuery,
    NodesQuery,
    PathQuery,
    PathsQuery,
    PropagateStmt,
    RecallQuery,
    RetractStmt,
    ShortestPathQuery,
    SimilarQuery,
    SubgraphQuery,
    TraverseQuery,
    UpdateEdge,
    UpdateNode,
    UpdateNodes,
    UpsertNode,
    VaultNew,
    VaultRead,
    VaultWrite,
    VaultAppend,
    VaultSearch,
    VaultBacklinks,
    VaultList,
    VaultSync,
    VaultDaily,
    VaultArchive,
    WeightedDistanceQuery,
    WeightedShortestPathQuery,
)
from graphstore.core.errors import GraphStoreError
from graphstore.core.types import Result
from graphstore.dsl.executor_reads import ReadExecutor
from graphstore.dsl.executor_writes import WriteExecutor


_VAULT_TYPES = (VaultNew, VaultRead, VaultWrite, VaultAppend,
                VaultSearch, VaultBacklinks, VaultList,
                VaultSync, VaultDaily, VaultArchive)


class Executor(ReadExecutor, WriteExecutor):
    """Full executor combining read and write handlers."""

    _vault_executor = None  # set by GraphStore when vault is configured

    def _dispatch(self, ast) -> Result:
        """Route AST node to handler."""
        # Vault commands
        if isinstance(ast, _VAULT_TYPES):
            if not self._vault_executor:
                raise GraphStoreError("Vault not configured. Use GraphStore(vault='./notes')")
            return self._vault_executor.dispatch(ast)

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
            SimilarQuery: self._similar,
            LexicalSearchQuery: self._lexical_search,
            PropagateStmt: self._propagate,
            BindContext: self._bind_context,
            DiscardContext: self._discard_context,
            IngestStmt: self._ingest,
            ConnectNode: self._connect_node,
            ForgetNode: self._forget,
        }
        handler = handlers.get(type(ast))
        if handler is None:
            raise GraphStoreError(f"Unknown AST node type: {type(ast).__name__}")
        return handler(ast)
