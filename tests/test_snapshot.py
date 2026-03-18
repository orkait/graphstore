"""Tests for graphstore.snapshot — GraphSnapshot and SnapshotManager."""

import numpy as np
import pytest

from graphstore.snapshot import GraphSnapshot, SnapshotManager


# ---------------------------------------------------------------------------
# GraphSnapshot.empty()
# ---------------------------------------------------------------------------


class TestGraphSnapshotEmpty:
    def test_default_capacity(self):
        snap = GraphSnapshot.empty()
        assert snap.node_ids.shape == (1024,)
        assert snap.node_kinds.shape == (1024,)
        assert len(snap.node_data) == 1024

    def test_custom_capacity(self):
        snap = GraphSnapshot.empty(capacity=64)
        assert snap.node_ids.shape == (64,)
        assert snap.node_kinds.shape == (64,)
        assert len(snap.node_data) == 64

    def test_array_dtypes(self):
        snap = GraphSnapshot.empty()
        assert snap.node_ids.dtype == np.int32
        assert snap.node_kinds.dtype == np.uint8

    def test_node_ids_initialized_to_minus_one(self):
        snap = GraphSnapshot.empty(capacity=8)
        assert np.all(snap.node_ids == -1)

    def test_node_kinds_initialized_to_zero(self):
        snap = GraphSnapshot.empty(capacity=8)
        assert np.all(snap.node_kinds == 0)

    def test_node_data_all_none(self):
        snap = GraphSnapshot.empty(capacity=8)
        assert all(d is None for d in snap.node_data)

    def test_counters_zero(self):
        snap = GraphSnapshot.empty()
        assert snap.node_count == 0
        assert snap.next_slot == 0

    def test_tombstones_empty(self):
        snap = GraphSnapshot.empty()
        assert snap.node_tombstones == set()
        assert len(snap.node_tombstones) == 0

    def test_indices_empty(self):
        snap = GraphSnapshot.empty()
        assert snap.secondary_indices == {}
        assert snap.id_to_slot == {}
        assert snap.edges_by_type == {}
        assert snap.indexed_fields == set()

    def test_string_table_empty(self):
        snap = GraphSnapshot.empty()
        assert len(snap.string_table) == 0

    def test_edge_matrices_empty(self):
        snap = GraphSnapshot.empty()
        assert snap.edge_matrices.total_edges == 0
        assert snap.edge_matrices.edge_types == []


# ---------------------------------------------------------------------------
# GraphSnapshot.copy()
# ---------------------------------------------------------------------------


class TestGraphSnapshotCopy:
    @pytest.fixture()
    def populated_snapshot(self):
        """Build a snapshot with some real data for copy tests."""
        snap = GraphSnapshot.empty(capacity=8)

        # Populate string table
        sid_alice = snap.string_table.intern("alice")
        sid_bob = snap.string_table.intern("bob")

        # Populate slots
        snap.node_ids[0] = sid_alice
        snap.node_ids[1] = sid_bob
        snap.node_kinds[0] = 1
        snap.node_kinds[1] = 2
        snap.node_data[0] = {"name": "Alice", "age": 30}
        snap.node_data[1] = {"name": "Bob", "age": 25}
        snap.id_to_slot[sid_alice] = 0
        snap.id_to_slot[sid_bob] = 1
        snap.next_slot = 2
        snap.node_count = 2

        # Tombstone
        snap.node_tombstones.add(5)

        # Secondary index
        snap.secondary_indices["name"] = {"Alice": [0], "Bob": [1]}
        snap.indexed_fields.add("name")

        # Edge data
        snap.edges_by_type["knows"] = [(0, 1, {"since": 2020})]

        return snap

    def test_copy_returns_new_object(self, populated_snapshot):
        cp = populated_snapshot.copy()
        assert cp is not populated_snapshot

    def test_numpy_arrays_are_independent(self, populated_snapshot):
        cp = populated_snapshot.copy()

        # Modify copy's arrays
        cp.node_ids[0] = 999
        cp.node_kinds[0] = 255

        # Original unaffected
        assert populated_snapshot.node_ids[0] != 999
        assert populated_snapshot.node_kinds[0] != 255

    def test_node_data_dicts_are_independent(self, populated_snapshot):
        cp = populated_snapshot.copy()

        cp.node_data[0]["name"] = "CHANGED"
        assert populated_snapshot.node_data[0]["name"] == "Alice"

    def test_node_data_list_is_independent(self, populated_snapshot):
        cp = populated_snapshot.copy()

        cp.node_data[2] = {"new": True}
        assert populated_snapshot.node_data[2] is None

    def test_string_table_is_independent(self, populated_snapshot):
        cp = populated_snapshot.copy()

        # Intern a new string in the copy
        cp.string_table.intern("charlie")
        assert "charlie" in cp.string_table
        assert "charlie" not in populated_snapshot.string_table

    def test_string_table_preserves_existing(self, populated_snapshot):
        cp = populated_snapshot.copy()

        assert len(cp.string_table) == len(populated_snapshot.string_table)
        assert cp.string_table.lookup(0) == populated_snapshot.string_table.lookup(0)
        assert cp.string_table.lookup(1) == populated_snapshot.string_table.lookup(1)

    def test_tombstones_are_independent(self, populated_snapshot):
        cp = populated_snapshot.copy()

        cp.node_tombstones.add(6)
        assert 6 not in populated_snapshot.node_tombstones
        assert 5 in populated_snapshot.node_tombstones

    def test_id_to_slot_is_independent(self, populated_snapshot):
        cp = populated_snapshot.copy()

        cp.id_to_slot[999] = 7
        assert 999 not in populated_snapshot.id_to_slot

    def test_secondary_indices_are_independent(self, populated_snapshot):
        cp = populated_snapshot.copy()

        # Mutate the copied index
        cp.secondary_indices["name"]["Alice"].append(99)
        assert 99 not in populated_snapshot.secondary_indices["name"]["Alice"]

    def test_secondary_indices_new_key_independent(self, populated_snapshot):
        cp = populated_snapshot.copy()

        cp.secondary_indices["age"] = {30: [0]}
        assert "age" not in populated_snapshot.secondary_indices

    def test_edges_by_type_are_independent(self, populated_snapshot):
        cp = populated_snapshot.copy()

        cp.edges_by_type["knows"].append((1, 0, {}))
        assert len(populated_snapshot.edges_by_type["knows"]) == 1

    def test_edges_by_type_new_type_independent(self, populated_snapshot):
        cp = populated_snapshot.copy()

        cp.edges_by_type["likes"] = [(0, 1, {})]
        assert "likes" not in populated_snapshot.edges_by_type

    def test_indexed_fields_are_independent(self, populated_snapshot):
        cp = populated_snapshot.copy()

        cp.indexed_fields.add("age")
        assert "age" not in populated_snapshot.indexed_fields

    def test_edge_matrices_is_fresh_instance(self, populated_snapshot):
        cp = populated_snapshot.copy()
        assert cp.edge_matrices is not populated_snapshot.edge_matrices

    def test_scalar_fields_copied(self, populated_snapshot):
        cp = populated_snapshot.copy()

        assert cp.next_slot == populated_snapshot.next_slot
        assert cp.node_count == populated_snapshot.node_count

    def test_scalar_mutation_independent(self, populated_snapshot):
        cp = populated_snapshot.copy()

        cp.next_slot = 100
        cp.node_count = 50
        assert populated_snapshot.next_slot == 2
        assert populated_snapshot.node_count == 2


# ---------------------------------------------------------------------------
# SnapshotManager
# ---------------------------------------------------------------------------


class TestSnapshotManager:
    def test_initial_current_is_none(self):
        mgr = SnapshotManager()
        assert mgr.current is None

    def test_initialize_sets_current(self):
        mgr = SnapshotManager()
        snap = GraphSnapshot.empty()
        mgr.initialize(snap)
        assert mgr.current is snap

    def test_swap_replaces_current(self):
        mgr = SnapshotManager()
        snap1 = GraphSnapshot.empty()
        snap2 = GraphSnapshot.empty()

        mgr.initialize(snap1)
        mgr.swap(snap2)
        assert mgr.current is snap2

    def test_swap_returns_old_snapshot(self):
        mgr = SnapshotManager()
        snap1 = GraphSnapshot.empty()
        snap2 = GraphSnapshot.empty()

        mgr.initialize(snap1)
        old = mgr.swap(snap2)
        assert old is snap1

    def test_swap_from_none_returns_none(self):
        mgr = SnapshotManager()
        snap = GraphSnapshot.empty()

        old = mgr.swap(snap)
        assert old is None
        assert mgr.current is snap

    def test_multiple_swaps(self):
        mgr = SnapshotManager()
        snaps = [GraphSnapshot.empty() for _ in range(5)]

        mgr.initialize(snaps[0])
        for i in range(1, 5):
            old = mgr.swap(snaps[i])
            assert old is snaps[i - 1]

        assert mgr.current is snaps[4]

    def test_current_always_returns_latest(self):
        mgr = SnapshotManager()
        snap1 = GraphSnapshot.empty(capacity=16)
        snap2 = GraphSnapshot.empty(capacity=32)

        mgr.initialize(snap1)
        assert mgr.current.node_ids.shape == (16,)

        mgr.swap(snap2)
        assert mgr.current.node_ids.shape == (32,)


# ---------------------------------------------------------------------------
# Integration: snapshot with real StringTable / EdgeMatrices data
# ---------------------------------------------------------------------------


class TestSnapshotIntegration:
    def _make_populated_snapshot(self):
        """Helper: build a snapshot with nodes, edges, and indices."""
        snap = GraphSnapshot.empty(capacity=16)

        # Add nodes via string table
        for node_id, kind, data in [
            ("server-1", 1, {"role": "web", "cpu": 4}),
            ("server-2", 1, {"role": "db", "cpu": 8}),
            ("client-1", 2, {"role": "browser"}),
        ]:
            sid = snap.string_table.intern(node_id)
            slot = snap.next_slot
            snap.node_ids[slot] = sid
            snap.node_kinds[slot] = kind
            snap.node_data[slot] = data
            snap.id_to_slot[sid] = slot
            snap.next_slot += 1
            snap.node_count += 1

        # Edges
        snap.edges_by_type["connects_to"] = [
            (0, 1, {"latency_ms": 5}),
            (2, 0, {"latency_ms": 50}),
        ]

        # Build edge matrices
        snap.edge_matrices.rebuild(snap.edges_by_type, snap.next_slot)

        # Secondary index
        snap.secondary_indices["role"] = {
            "web": [0],
            "db": [1],
            "browser": [2],
        }
        snap.indexed_fields.add("role")

        return snap

    def test_copy_preserves_node_data(self):
        snap = self._make_populated_snapshot()
        cp = snap.copy()

        assert cp.node_count == 3
        assert cp.next_slot == 3
        assert cp.string_table.lookup(cp.node_ids[0]) == "server-1"
        assert cp.string_table.lookup(cp.node_ids[1]) == "server-2"
        assert cp.string_table.lookup(cp.node_ids[2]) == "client-1"

    def test_copy_independence_with_real_data(self):
        snap = self._make_populated_snapshot()
        cp = snap.copy()

        # Add a node to the copy
        sid = cp.string_table.intern("server-3")
        slot = cp.next_slot
        cp.node_ids[slot] = sid
        cp.node_kinds[slot] = 1
        cp.node_data[slot] = {"role": "cache", "cpu": 2}
        cp.id_to_slot[sid] = slot
        cp.next_slot += 1
        cp.node_count += 1

        # Original unchanged
        assert snap.node_count == 3
        assert snap.next_slot == 3
        assert "server-3" not in snap.string_table

    def test_copy_edge_data_preserved(self):
        snap = self._make_populated_snapshot()
        cp = snap.copy()

        assert len(cp.edges_by_type["connects_to"]) == 2
        assert cp.edges_by_type["connects_to"][0] == (0, 1, {"latency_ms": 5})

    def test_copy_edge_data_independent(self):
        snap = self._make_populated_snapshot()
        cp = snap.copy()

        cp.edges_by_type["connects_to"].append((1, 2, {}))
        assert len(snap.edges_by_type["connects_to"]) == 2

    def test_copy_edge_matrices_can_be_rebuilt(self):
        snap = self._make_populated_snapshot()
        cp = snap.copy()

        # Rebuild on the copy with modified edges
        cp.edges_by_type["connects_to"].append((1, 2, {"latency_ms": 10}))
        cp.edge_matrices.rebuild(cp.edges_by_type, cp.next_slot)

        assert cp.edge_matrices.total_edges == 3
        # Original matrices untouched
        assert snap.edge_matrices.total_edges == 2

    def test_manager_swap_with_populated_snapshots(self):
        mgr = SnapshotManager()

        snap1 = self._make_populated_snapshot()
        mgr.initialize(snap1)
        assert mgr.current.node_count == 3

        snap2 = snap1.copy()
        sid = snap2.string_table.intern("server-4")
        slot = snap2.next_slot
        snap2.node_ids[slot] = sid
        snap2.node_data[slot] = {"role": "queue"}
        snap2.id_to_slot[sid] = slot
        snap2.next_slot += 1
        snap2.node_count += 1

        old = mgr.swap(snap2)
        assert old.node_count == 3
        assert mgr.current.node_count == 4

    def test_secondary_index_independence_integration(self):
        snap = self._make_populated_snapshot()
        cp = snap.copy()

        cp.secondary_indices["role"]["cache"] = [3]
        assert "cache" not in snap.secondary_indices["role"]

        cp.secondary_indices["role"]["web"].append(99)
        assert snap.secondary_indices["role"]["web"] == [0]
