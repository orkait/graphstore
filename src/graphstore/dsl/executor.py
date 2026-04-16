"""DSL executor: auto-dispatch via handler registry.

Combines all domain handler mixins via multiple inheritance.
New commands self-register via @handles decorator - no manual dispatch dict.
"""

from graphstore.dsl.ast_nodes import (
    VaultNew, VaultRead, VaultWrite, VaultAppend,
    VaultSearch, VaultBacklinks, VaultList,
    VaultSync, VaultDaily, VaultArchive,
)
from graphstore.core.errors import GraphStoreError
from graphstore.core.types import Result
from graphstore.dsl.executor_base import ExecutorBase

from graphstore.dsl.handlers import (
    DISPATCH,
    NodeHandlers,
    EdgeHandlers,
    TraversalHandlers,
    PatternHandlers,
    AggregationHandlers,
    IntelligenceHandlers,
    BeliefHandlers,
    MutationHandlers,
    ContextHandlers,
    IngestHandlers,
)


_VAULT_TYPES = (VaultNew, VaultRead, VaultWrite, VaultAppend,
                VaultSearch, VaultBacklinks, VaultList,
                VaultSync, VaultDaily, VaultArchive)


class Executor(
    NodeHandlers,
    EdgeHandlers,
    TraversalHandlers,
    PatternHandlers,
    AggregationHandlers,
    IntelligenceHandlers,
    BeliefHandlers,
    MutationHandlers,
    ContextHandlers,
    IngestHandlers,
    ExecutorBase,
):
    """Full executor combining all domain handlers via auto-dispatch registry."""

    _vault_executor = None

    def _dispatch(self, ast) -> Result:
        if isinstance(ast, _VAULT_TYPES):
            if not self._vault_executor:
                raise GraphStoreError("Vault not configured. Use GraphStore(vault='./notes')")
            return self._vault_executor.dispatch(ast)

        handler = DISPATCH.get(type(ast))
        if handler is None:
            raise GraphStoreError(f"Unknown AST node type: {type(ast).__name__}")
        return handler(self, ast)
