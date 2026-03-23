"""Context binding handlers for the DSL executor."""

import numpy as np

from graphstore.dsl.handlers._registry import handles
from graphstore.dsl.ast_nodes import BindContext, DiscardContext
from graphstore.core.types import Result
from graphstore.core.errors import NodeNotFound


class ContextHandlers:

    @handles(BindContext, write=True)
    def _bind_context(self, q: BindContext) -> Result:
        """BIND CONTEXT: set active context on store."""
        self.store._active_context = q.name
        return Result(kind="ok", data={"context": q.name}, count=0)

    @handles(DiscardContext, write=True)
    def _discard_context(self, q: DiscardContext) -> Result:
        """DISCARD CONTEXT: delete all nodes with matching __context__ and unbind."""
        deleted_count = 0
        n = self.store._next_slot
        if n > 0 and self.store.columns.has_column("__context__"):
            ctx_col = self.store.columns.get_column("__context__", n)
            if ctx_col is not None:
                col_data, col_pres, _ = ctx_col
                ctx_id = self.store.string_table.intern(q.name)
                ctx_mask = col_pres & (col_data == ctx_id)
                slots_to_delete = np.nonzero(ctx_mask)[0]
                for slot in slots_to_delete:
                    nid = self.store._slot_to_id(int(slot))
                    if nid:
                        try:
                            self.store.delete_node(nid)
                            deleted_count += 1
                        except NodeNotFound:
                            pass

        self.store._active_context = None
        return Result(kind="ok", data={"discarded": q.name, "deleted": deleted_count}, count=deleted_count)
