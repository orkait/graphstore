"""Visibility helpers: live-mask computation, slot/ID visibility, TTL."""

import time

import numpy as np


class VisibilityMixin:

    def _compute_live_mask(self, n: int) -> np.ndarray:
        """Unified visibility filter: tombstones + TTL + retracted + context."""
        mask = self.store.compute_live_mask(n)

        # Context filtering: when bound, only show nodes tagged with active context
        if hasattr(self.store, '_active_context') and self.store._active_context:
            ctx_name = self.store._active_context
            ctx_mask = self.store.columns.get_mask("__context__", "=", ctx_name, n)
            if ctx_mask is not None:
                mask = mask & ctx_mask
            else:
                # No __context__ column at all - nothing has context
                mask = np.zeros(n, dtype=bool)

        return mask

    def _resolve_slot(self, node_id: str) -> int | None:
        """Resolve a string node ID to its slot index."""
        if node_id not in self.store.string_table:
            return None
        str_id = self.store.string_table.intern(node_id)
        slot = self.store.id_to_slot.get(str_id)
        if slot is None or slot in self.store.node_tombstones:
            return None
        return slot

    def _is_slot_visible(self, slot: int) -> bool:
        """Check if a slot passes TTL, retraction, and context checks."""
        # Check retracted
        if self.store.columns.has_column("__retracted__"):
            if self.store.columns._presence["__retracted__"][slot]:
                if int(self.store.columns._columns["__retracted__"][slot]) == 1:
                    return False
        # Check TTL expiry
        if self.store.columns.has_column("__expires_at__"):
            if self.store.columns._presence["__expires_at__"][slot]:
                expire_ms = int(self.store.columns._columns["__expires_at__"][slot])
                if expire_ms > 0 and expire_ms < int(time.time() * 1000):
                    return False
        # Check context
        if self.store._active_context is not None:
            if self.store.columns.has_column("__context__"):
                if self.store.columns._presence["__context__"][slot]:
                    ctx_id = self.store.string_table.intern(self.store._active_context)
                    if int(self.store.columns._columns["__context__"][slot]) != ctx_id:
                        return False
                else:
                    # Node has no context tag but context is active - invisible
                    return False
            else:
                # No context column at all - nothing has context
                return False
        return True

    def _is_visible_by_id(self, node_id: str) -> bool:
        """Check if a node ID is visible (not tombstoned, expired, or retracted)."""
        slot = self._resolve_slot(node_id)
        if slot is None:
            return False
        return self._is_slot_visible(slot)

    def _filter_visible(self, nodes: list[dict]) -> list[dict]:
        """Filter out retracted, expired, and out-of-context nodes."""
        has_retracted = self.store.columns.has_column("__retracted__")
        has_expires = self.store.columns.has_column("__expires_at__")
        has_context = self.store._active_context is not None
        if not has_retracted and not has_expires and not has_context:
            return nodes
        result = []
        for node in nodes:
            slot = self._resolve_slot(node["id"])
            if slot is not None and self._is_slot_visible(slot):
                result.append(node)
        return result

    def _apply_ttl(self, node_id: str, expires_in: tuple | None, expires_at: str | None):
        """Set __expires_at__ on a node based on TTL clauses."""
        if expires_in is None and expires_at is None:
            return
        str_id = self.store.string_table.intern(node_id)
        slot = self.store.id_to_slot[str_id]
        if expires_in is not None:
            amount, unit = expires_in
            unit_ms = {"s": 1000, "m": 60000, "h": 3600000, "d": 86400000}[unit]
            expire_ms = int(time.time() * 1000) + amount * unit_ms
        else:
            from datetime import datetime
            dt = datetime.fromisoformat(expires_at)
            expire_ms = int(dt.timestamp() * 1000)
        self.store.columns.set_reserved(slot, "__expires_at__", expire_ms)
