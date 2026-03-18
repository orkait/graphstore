"""Immutable graph snapshot and atomic swap manager.

GraphSnapshot is an immutable (by convention) container for the entire graph
state.  SnapshotManager handles atomic swap of the current snapshot reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from graphstore.strings import StringTable
from graphstore.edges import EdgeMatrices


@dataclass
class GraphSnapshot:
    """Immutable snapshot of the graph state."""

    string_table: StringTable
    node_ids: np.ndarray  # int32, interned string IDs in slot order
    node_kinds: np.ndarray  # uint8
    node_data: list[dict]
    node_tombstones: set[int]  # tombstoned slot indices
    edge_matrices: EdgeMatrices
    secondary_indices: dict[str, dict]
    id_to_slot: dict[int, int]  # interned string ID -> array slot
    next_slot: int = 0  # next slot to fill
    node_count: int = 0  # number of live nodes
    edges_by_type: dict = field(default_factory=dict)  # raw edge lists for rebuild
    indexed_fields: set = field(default_factory=set)

    @staticmethod
    def empty(capacity: int = 1024) -> GraphSnapshot:
        """Create an empty snapshot with pre-allocated arrays."""
        return GraphSnapshot(
            string_table=StringTable(),
            node_ids=np.full(capacity, -1, dtype=np.int32),
            node_kinds=np.zeros(capacity, dtype=np.uint8),
            node_data=[None] * capacity,
            node_tombstones=set(),
            edge_matrices=EdgeMatrices(),
            secondary_indices={},
            id_to_slot={},
            next_slot=0,
            node_count=0,
            edges_by_type={},
            indexed_fields=set(),
        )

    def copy(self) -> GraphSnapshot:
        """Deep copy for creating a working copy before mutations."""
        return GraphSnapshot(
            string_table=StringTable.from_list(self.string_table.to_list()),
            node_ids=self.node_ids.copy(),
            node_kinds=self.node_kinds.copy(),
            node_data=[dict(d) if d is not None else None for d in self.node_data],
            node_tombstones=set(self.node_tombstones),
            edge_matrices=EdgeMatrices(),  # will be rebuilt after mutations
            secondary_indices={
                f: {v: list(slots) for v, slots in idx.items()}
                for f, idx in self.secondary_indices.items()
            },
            id_to_slot=dict(self.id_to_slot),
            next_slot=self.next_slot,
            node_count=self.node_count,
            edges_by_type={
                etype: list(edges) for etype, edges in self.edges_by_type.items()
            },
            indexed_fields=set(self.indexed_fields),
        )


class SnapshotManager:
    """Manages the current graph snapshot with atomic swap."""

    def __init__(self):
        self._current: GraphSnapshot | None = None

    @property
    def current(self) -> GraphSnapshot | None:
        return self._current

    def initialize(self, snapshot: GraphSnapshot):
        """Set the initial snapshot."""
        self._current = snapshot

    def swap(self, new_snapshot: GraphSnapshot) -> GraphSnapshot | None:
        """Atomic swap. Python GIL makes pointer assignment atomic.

        Returns the old snapshot for reference.
        """
        old = self._current
        self._current = new_snapshot
        return old
