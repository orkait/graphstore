"""Core in-memory graph engine.

Manages nodes (numpy arrays + dicts) and edges (EdgeMatrices with CSR)
with secondary indices, tombstone-based deletion, and memory ceiling
enforcement.
"""

from __future__ import annotations

import numpy as np

from graphstore.edges import EdgeMatrices
from graphstore.errors import GraphStoreError, NodeExists, NodeNotFound
from graphstore.memory import DEFAULT_CEILING_BYTES, check_ceiling
from graphstore.strings import StringTable


class CoreStore:
    """In-memory graph store backed by numpy arrays and sparse matrices."""

    def __init__(self, ceiling_bytes: int = DEFAULT_CEILING_BYTES):
        self.string_table = StringTable()
        self._edge_matrices = EdgeMatrices()
        self._ceiling_bytes = ceiling_bytes

        # Node storage - pre-allocate
        self._capacity = 1024
        self._count = 0  # number of live nodes (not counting tombstones)
        self._next_slot = 0  # next slot to fill
        self.node_ids = np.full(self._capacity, -1, dtype=np.int32)
        self.node_kinds = np.zeros(self._capacity, dtype=np.int32)
        self.node_data: list[dict | None] = [None] * self._capacity
        self.node_tombstones: set[int] = set()
        self.id_to_slot: dict[int, int] = {}

        # Edge storage
        self._edges_by_type: dict[str, list[tuple]] = {}
        self._edge_keys: set[tuple[int, int, str]] = set()  # (src_slot, tgt_slot, kind) for O(1) duplicate check
        self._edges_dirty = False  # deferred CSR rebuild flag

        # Secondary indices
        self.secondary_indices: dict[str, dict] = {}
        self._indexed_fields: set[str] = set()

    # -- slot management -----------------------------------------------------

    def _alloc_slot(self) -> int:
        """Get next available slot, reusing tombstones or expanding."""
        if self.node_tombstones:
            slot = self.node_tombstones.pop()
            return slot
        if self._next_slot >= self._capacity:
            self._grow()
        slot = self._next_slot
        self._next_slot += 1
        return slot

    def _grow(self):
        """Double array capacity."""
        new_cap = self._capacity * 2

        new_ids = np.full(new_cap, -1, dtype=np.int32)
        new_ids[: self._capacity] = self.node_ids
        self.node_ids = new_ids

        new_kinds = np.zeros(new_cap, dtype=np.int32)
        new_kinds[: self._capacity] = self.node_kinds
        self.node_kinds = new_kinds

        self.node_data.extend([None] * self._capacity)
        self._capacity = new_cap

    # -- properties ----------------------------------------------------------

    @property
    def node_count(self) -> int:
        return self._count

    @property
    def edge_matrices(self) -> EdgeMatrices:
        """Auto-rebuilds CSR matrices if dirty."""
        self._ensure_edges_built()
        return self._edge_matrices

    @property
    def edge_count(self) -> int:
        # Use raw count to avoid CSR rebuild for simple counting
        return sum(len(v) for v in self._edges_by_type.values())

    # -- node CRUD -----------------------------------------------------------

    def put_node(self, id: str, kind: str, data: dict) -> int:
        """Add a node. Returns slot index. Raises NodeExists if ID exists."""
        str_id = self.string_table.intern(id)
        if str_id in self.id_to_slot:
            slot = self.id_to_slot[str_id]
            if slot not in self.node_tombstones:
                raise NodeExists(id)

        # Check ceiling (use raw count to avoid triggering CSR rebuild)
        raw_edge_count = sum(len(v) for v in self._edges_by_type.values())
        check_ceiling(self._count, raw_edge_count, 1, 0, self._ceiling_bytes)

        kind_id = self.string_table.intern(kind)
        slot = self._alloc_slot()

        self.node_ids[slot] = str_id
        self.node_kinds[slot] = kind_id
        self.node_data[slot] = dict(data)  # copy
        self.id_to_slot[str_id] = slot
        self._count += 1

        # Update secondary indices
        for field in self._indexed_fields:
            if field in data:
                val = data[field]
                self.secondary_indices[field].setdefault(val, []).append(slot)

        return slot

    def get_node(self, id: str) -> dict | None:
        """Get node data by ID. Returns None if not found."""
        if id not in self.string_table:
            return None
        str_id = self.string_table.intern(id)
        slot = self.id_to_slot.get(str_id)
        if slot is None or slot in self.node_tombstones:
            return None
        kind_id = self.node_kinds[slot]
        return {
            "id": id,
            "kind": self.string_table.lookup(kind_id),
            **self.node_data[slot],
        }

    def update_node(self, id: str, data: dict):
        """Update node data. Raises NodeNotFound if missing."""
        if id not in self.string_table:
            raise NodeNotFound(id)
        str_id = self.string_table.intern(id)
        slot = self.id_to_slot.get(str_id)
        if slot is None or slot in self.node_tombstones:
            raise NodeNotFound(id)

        old_data = self.node_data[slot]
        # Remove old values from indices
        for field in self._indexed_fields:
            if field in old_data:
                old_val = old_data[field]
                idx_list = self.secondary_indices[field].get(old_val, [])
                if slot in idx_list:
                    idx_list.remove(slot)

        # Update data
        self.node_data[slot].update(data)

        # Add new values to indices
        for field in self._indexed_fields:
            if field in data:
                val = self.node_data[slot][field]
                self.secondary_indices[field].setdefault(val, []).append(slot)

    def upsert_node(self, id: str, kind: str, data: dict) -> int:
        """Create or update node."""
        if id in self.string_table:
            str_id = self.string_table.intern(id)
            slot = self.id_to_slot.get(str_id)
            if slot is not None and slot not in self.node_tombstones:
                self.update_node(id, data)
                return slot
        return self.put_node(id, kind, data)

    def delete_node(self, id: str):
        """Delete node and cascade-delete all edges. Raises NodeNotFound."""
        if id not in self.string_table:
            raise NodeNotFound(id)
        str_id = self.string_table.intern(id)
        slot = self.id_to_slot.get(str_id)
        if slot is None or slot in self.node_tombstones:
            raise NodeNotFound(id)

        # Remove from indices
        for field in self._indexed_fields:
            if field in self.node_data[slot]:
                val = self.node_data[slot][field]
                idx_list = self.secondary_indices[field].get(val, [])
                if slot in idx_list:
                    idx_list.remove(slot)

        # Tombstone
        self.node_tombstones.add(slot)
        self.node_data[slot] = None
        del self.id_to_slot[str_id]
        self._count -= 1

        # Cascade-delete edges touching this node
        self._cascade_delete_edges(slot)

    # -- edge CRUD -----------------------------------------------------------

    def put_edge(
        self, source_id: str, target_id: str, kind: str, data: dict | None = None
    ):
        """Add an edge. Both nodes must exist."""
        if source_id not in self.string_table:
            raise NodeNotFound(source_id)
        if target_id not in self.string_table:
            raise NodeNotFound(target_id)

        src_str_id = self.string_table.intern(source_id)
        tgt_str_id = self.string_table.intern(target_id)

        src_slot = self.id_to_slot.get(src_str_id)
        tgt_slot = self.id_to_slot.get(tgt_str_id)

        if src_slot is None or src_slot in self.node_tombstones:
            raise NodeNotFound(source_id)
        if tgt_slot is None or tgt_slot in self.node_tombstones:
            raise NodeNotFound(target_id)

        # Check ceiling (use raw count to avoid triggering CSR rebuild)
        raw_edge_count = sum(len(v) for v in self._edges_by_type.values())
        check_ceiling(self._count, raw_edge_count, 0, 1, self._ceiling_bytes)

        # Check duplicate - O(1) set lookup
        edge_key = (src_slot, tgt_slot, kind)
        edge_data = data or {}
        if edge_key in self._edge_keys:
            # Only raise if data also matches (rare: same src/tgt/kind but different data)
            for s, t, d in self._edges_by_type.get(kind, []):
                if s == src_slot and t == tgt_slot and d == edge_data:
                    raise GraphStoreError(
                        f"Duplicate edge: {source_id} -> {target_id} kind={kind}"
                    )

        self._edge_keys.add(edge_key)
        self._edges_by_type.setdefault(kind, []).append(
            (src_slot, tgt_slot, edge_data)
        )
        self._edges_dirty = True

    def delete_edge(self, source_id: str, target_id: str, kind: str):
        """Delete a specific edge."""
        src_str_id = self.string_table.intern(source_id)
        tgt_str_id = self.string_table.intern(target_id)
        src_slot = self.id_to_slot.get(src_str_id)
        tgt_slot = self.id_to_slot.get(tgt_str_id)

        if kind in self._edges_by_type:
            self._edges_by_type[kind] = [
                (s, t, d)
                for s, t, d in self._edges_by_type[kind]
                if not (s == src_slot and t == tgt_slot)
            ]
            if not self._edges_by_type[kind]:
                del self._edges_by_type[kind]
        self._edge_keys.discard((src_slot, tgt_slot, kind))
        self._rebuild_edges()

    def get_edges_from(self, id: str, kind: str | None = None) -> list[dict]:
        """Get outgoing edges from a node."""
        self._ensure_edges_built()
        if id not in self.string_table:
            return []
        str_id = self.string_table.intern(id)
        slot = self.id_to_slot.get(str_id)
        if slot is None or slot in self.node_tombstones:
            return []

        result = []
        types_to_check = [kind] if kind else self.edge_matrices.edge_types
        for etype in types_to_check:
            for s, t, d in self._edges_by_type.get(etype, []):
                if s == slot:
                    tgt_id = self._slot_to_id(t)
                    if tgt_id is not None:
                        result.append({"source": id, "target": tgt_id, "kind": etype, **d})
        return result

    def get_edges_to(self, id: str, kind: str | None = None) -> list[dict]:
        """Get incoming edges to a node."""
        self._ensure_edges_built()
        if id not in self.string_table:
            return []
        str_id = self.string_table.intern(id)
        slot = self.id_to_slot.get(str_id)
        if slot is None or slot in self.node_tombstones:
            return []

        result = []
        types_to_check = [kind] if kind else self.edge_matrices.edge_types
        for etype in types_to_check:
            for s, t, d in self._edges_by_type.get(etype, []):
                if t == slot:
                    src_id = self._slot_to_id(s)
                    if src_id is not None:
                        result.append({"source": src_id, "target": id, "kind": etype, **d})
        return result

    # -- cascade / rebuild ---------------------------------------------------

    def _cascade_delete_edges(self, slot: int):
        """Remove all edges involving this slot from pending edges."""
        for etype in list(self._edges_by_type.keys()):
            old_len = len(self._edges_by_type[etype])
            self._edges_by_type[etype] = [
                (s, t, d)
                for s, t, d in self._edges_by_type[etype]
                if s != slot and t != slot
            ]
            if not self._edges_by_type[etype]:
                del self._edges_by_type[etype]
            if len(self._edges_by_type.get(etype, [])) != old_len:
                # Rebuild edge_keys set from scratch (simpler than tracking removals)
                self._edge_keys = {
                    (s, t, k)
                    for k, edges in self._edges_by_type.items()
                    for s, t, _d in edges
                }
        self._edges_dirty = True
        self._ensure_edges_built()

    def _ensure_edges_built(self):
        """Lazily rebuild CSR matrices only when dirty."""
        if not self._edges_dirty:
            return
        self._edges_dirty = False
        num_nodes = max(self._next_slot, 1)
        self._edge_matrices.rebuild(self._edges_by_type, num_nodes)

    def _rebuild_edges(self):
        """Force rebuild EdgeMatrices from pending edge lists."""
        self._edges_dirty = False
        num_nodes = max(self._next_slot, 1)
        self._edge_matrices.rebuild(self._edges_by_type, num_nodes)

    # -- helpers -------------------------------------------------------------

    def _slot_to_id(self, slot: int) -> str | None:
        """Convert slot index back to string ID."""
        if slot >= self._next_slot or slot in self.node_tombstones:
            return None
        str_id = int(self.node_ids[slot])
        if str_id == -1:
            return None
        return self.string_table.lookup(str_id)

    # -- secondary indices ---------------------------------------------------

    def add_index(self, field: str):
        """Build secondary index on a field."""
        self._indexed_fields.add(field)
        index: dict = {}
        for slot in range(self._next_slot):
            if slot not in self.node_tombstones and self.node_data[slot] is not None:
                val = self.node_data[slot].get(field)
                if val is not None:
                    index.setdefault(val, []).append(slot)
        self.secondary_indices[field] = index

    def query_by_index(self, field: str, value) -> list[int]:
        """Query secondary index. Returns slot indices."""
        if field not in self.secondary_indices:
            return []
        return self.secondary_indices[field].get(value, [])

    # -- bulk queries --------------------------------------------------------

    def get_all_edges(self) -> list[dict]:
        """Get all edges across all types."""
        result = []
        for etype, edge_list in self._edges_by_type.items():
            for src_slot, tgt_slot, data in edge_list:
                src_id = self._slot_to_id(src_slot)
                tgt_id = self._slot_to_id(tgt_slot)
                if src_id and tgt_id:
                    result.append({"source": src_id, "target": tgt_id, "kind": etype, **data})
        return result

    def get_all_nodes(self, kind: str | None = None) -> list[dict]:
        """Get all live nodes, optionally filtered by kind."""
        if kind and kind not in self.string_table:
            return []

        n = self._next_slot
        if n == 0:
            return []

        # Build live-slot mask using numpy
        ids = self.node_ids[:n]
        mask = ids >= 0  # not empty slots

        # Exclude tombstones via numpy
        if self.node_tombstones:
            tomb_arr = np.array(list(self.node_tombstones), dtype=np.int32)
            tomb_mask = np.zeros(n, dtype=bool)
            tomb_mask[tomb_arr[tomb_arr < n]] = True
            mask &= ~tomb_mask

        # Kind filter via numpy
        if kind is not None:
            kind_id = self.string_table.intern(kind)
            mask &= self.node_kinds[:n] == kind_id

        # Extract only matching slots
        slots = np.nonzero(mask)[0]

        result = []
        for slot in slots:
            slot = int(slot)
            if self.node_data[slot] is None:
                continue
            result.append({
                "id": self.string_table.lookup(int(ids[slot])),
                "kind": self.string_table.lookup(int(self.node_kinds[slot])),
                **self.node_data[slot],
            })
        return result

    def count_nodes(self, kind: str | None = None) -> int:
        """Count live nodes without building dicts. Uses numpy."""
        if kind and kind not in self.string_table:
            return 0
        n = self._next_slot
        if n == 0:
            return 0
        mask = self.node_ids[:n] >= 0
        if self.node_tombstones:
            tomb_arr = np.array(list(self.node_tombstones), dtype=np.int32)
            tomb_mask = np.zeros(n, dtype=bool)
            tomb_mask[tomb_arr[tomb_arr < n]] = True
            mask &= ~tomb_mask
        if kind is not None:
            kind_id = self.string_table.intern(kind)
            mask &= self.node_kinds[:n] == kind_id
        return int(np.count_nonzero(mask))

    # -- field operations ----------------------------------------------------

    def increment_field(self, id: str, field: str, amount: int | float):
        """Increment a numeric field. Raises NodeNotFound or TypeError."""
        if id not in self.string_table:
            raise NodeNotFound(id)
        str_id = self.string_table.intern(id)
        slot = self.id_to_slot.get(str_id)
        if slot is None or slot in self.node_tombstones:
            raise NodeNotFound(id)

        data = self.node_data[slot]
        current = data.get(field, 0)
        if not isinstance(current, (int, float)):
            raise TypeError(
                f"Field '{field}' is not numeric: {type(current).__name__}"
            )
        data[field] = current + amount
