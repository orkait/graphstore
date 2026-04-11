"""Tests for the compaction sentinel crash-recovery mechanism.

``compact_tombstones_safe`` wraps ``compact_tombstones`` in a pre/post
checkpoint sandwich with a metadata sentinel. On crash mid-compaction,
the next ``deserializer.load()`` runs in-memory recovery from the
pre-compaction blob state; ``GraphStore.__init__`` then finishes the
DocumentStore orphan cleanup and clears the sentinel.
"""

import json
import time

from graphstore import GraphStore


def _write_sentinel(conn, pre_next_slot: int, pre_live_count: int) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO metadata VALUES ('compaction_sentinel', ?)",
        (json.dumps({
            "ts": time.time(),
            "pre_next_slot": pre_next_slot,
            "pre_live_count": pre_live_count,
        }),),
    )
    conn.commit()


def _sentinel_present(conn) -> bool:
    row = conn.execute(
        "SELECT value FROM metadata WHERE key='compaction_sentinel'"
    ).fetchone()
    return row is not None


def test_compaction_sentinel_happy_path(tmp_path):
    """Clean compaction leaves no sentinel row and compacts tombstones."""
    gs = GraphStore(path=str(tmp_path))
    gs.execute('SYS REGISTER NODE KIND "item" REQUIRED name')
    for i in range(10):
        gs.execute(f'CREATE NODE "n{i}" name = "item_{i}" kind = "item"')
    for i in range(5):
        gs.execute(f'DELETE NODE "n{i}"')

    assert len(gs._store.node_tombstones) == 5
    gs.execute("SYS OPTIMIZE COMPACT")

    assert not _sentinel_present(gs._conn)
    assert len(gs._store.node_tombstones) == 0
    assert gs._store._count == 5

    # All remaining nodes must still be retrievable by their IDs.
    for i in range(5, 10):
        result = gs.execute(f'NODE "n{i}"')
        assert result.data is not None
        assert result.data["name"] == f"item_{i}"

    gs.close()


def test_compaction_sentinel_recovery_from_mid_run_crash(tmp_path):
    """Simulate crash after sentinel write but before compaction runs.

    We manually checkpoint the pre-compaction state, write a sentinel,
    then close without running compact. On reopen, ``load()`` should
    detect the sentinel and run ``compact_tombstones`` in-memory.
    """
    gs = GraphStore(path=str(tmp_path))
    gs.execute('SYS REGISTER NODE KIND "item" REQUIRED name')
    for i in range(10):
        gs.execute(f'CREATE NODE "n{i}" name = "item_{i}" kind = "item"')
    for i in range(5):
        gs.execute(f'DELETE NODE "n{i}"')

    # Force a checkpoint so blobs reflect current pre-compaction state.
    gs.checkpoint()
    _write_sentinel(gs._conn, pre_next_slot=10, pre_live_count=5)
    gs.close()

    # Reopen - sentinel recovery should fire automatically.
    gs2 = GraphStore(path=str(tmp_path))

    # In-memory recovery should have compacted the store.
    assert len(gs2._store.node_tombstones) == 0
    assert gs2._store._count == 5

    # Sentinel cleared after Phase 2 (DocStore orphan cleanup).
    assert not _sentinel_present(gs2._conn)

    # Survived nodes still retrievable.
    for i in range(5, 10):
        result = gs2.execute(f'NODE "n{i}"')
        assert result.data is not None
        assert result.data["name"] == f"item_{i}"

    gs2.close()


def test_compaction_sentinel_no_persistence_falls_through():
    """In-memory mode (no path) bypasses the sentinel sandwich entirely."""
    gs = GraphStore()
    gs.execute('SYS REGISTER NODE KIND "item" REQUIRED name')
    for i in range(10):
        gs.execute(f'CREATE NODE "n{i}" name = "item_{i}" kind = "item"')
    for i in range(5):
        gs.execute(f'DELETE NODE "n{i}"')

    result = gs.execute("SYS OPTIMIZE COMPACT")
    assert result.kind == "ok"
    assert len(gs._store.node_tombstones) == 0
    assert gs._store._count == 5

    gs.close()


def test_compaction_sentinel_no_tombstones_is_noop(tmp_path):
    """A sentinel over an already-clean store (no tombstones) is a no-op.

    Models the crash-between-post-checkpoint-and-sentinel-delete case:
    the blobs already reflect post-compaction state (no tombstones),
    so recovery just clears the sentinel.
    """
    gs = GraphStore(path=str(tmp_path))
    gs.execute('SYS REGISTER NODE KIND "item" REQUIRED name')
    for i in range(10):
        gs.execute(f'CREATE NODE "n{i}" name = "item_{i}" kind = "item"')
    # No deletes → no tombstones.
    gs.checkpoint()
    _write_sentinel(gs._conn, pre_next_slot=10, pre_live_count=10)
    gs.close()

    # Reopen - recovery should succeed as a no-op.
    gs2 = GraphStore(path=str(tmp_path))
    assert len(gs2._store.node_tombstones) == 0
    assert gs2._store._count == 10
    assert not _sentinel_present(gs2._conn)
    gs2.close()


def test_compact_tombstones_safe_fast_paths(tmp_path):
    """Direct unit test of compact_tombstones_safe's early-exit branches."""
    from graphstore.core.optimizer import compact_tombstones_safe

    gs = GraphStore(path=str(tmp_path))
    gs.execute('SYS REGISTER NODE KIND "item" REQUIRED name')
    for i in range(5):
        gs.execute(f'CREATE NODE "n{i}" name = "item_{i}" kind = "item"')

    # No tombstones → fast-path returns {"compacted": 0} without writing
    # a sentinel.
    result = compact_tombstones_safe(
        gs._store, gs._schema, gs._conn,
        gs._vector_store, gs._document_store,
    )
    assert result == {"compacted": 0}
    assert not _sentinel_present(gs._conn)

    # Tombstone some nodes then run through the full sandwich.
    gs.execute('DELETE NODE "n0"')
    gs.execute('DELETE NODE "n1"')
    assert len(gs._store.node_tombstones) == 2

    result = compact_tombstones_safe(
        gs._store, gs._schema, gs._conn,
        gs._vector_store, gs._document_store,
    )
    assert result["compacted"] == 2
    assert len(gs._store.node_tombstones) == 0
    assert not _sentinel_present(gs._conn)

    gs.close()


def test_compact_tombstones_safe_inmemory_conn_none(tmp_path):
    """conn=None triggers the unsafe fall-through path."""
    from graphstore.core.optimizer import compact_tombstones_safe

    gs = GraphStore()  # no path → no conn
    gs.execute('SYS REGISTER NODE KIND "item" REQUIRED name')
    for i in range(5):
        gs.execute(f'CREATE NODE "n{i}" name = "item_{i}" kind = "item"')
    gs.execute('DELETE NODE "n0"')

    # schema is present but conn is None - should fall through cleanly.
    result = compact_tombstones_safe(
        gs._store, gs._schema, None,
        gs._vector_store, gs._document_store,
    )
    assert "compacted" in result
    assert result["compacted"] == 1

    gs.close()
