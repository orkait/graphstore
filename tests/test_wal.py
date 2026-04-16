"""Tests for WAL replay and query log rotation."""
import pytest
from graphstore import GraphStore


def test_wal_replay_tolerates_duplicate_create(tmp_path):
    path = tmp_path / "gs"
    gs = GraphStore(path=str(path))
    try:
        gs.execute('CREATE NODE "dup" kind = "doc" text = "x"')
        gs.checkpoint()
        gs._conn.execute(
            "INSERT INTO wal (timestamp, statement) VALUES (?, ?)",
            (0.0, 'CREATE NODE "dup" kind = "doc" text = "x"'),
        )
        gs._conn.commit()
    finally:
        gs._wal = None
        if gs._conn is not None:
            gs._conn.close()
            gs._runtime.conn = None

    gs2 = GraphStore(path=str(path))
    try:
        assert gs2.execute('NODE "dup"').data is not None
    finally:
        gs2.close()


def test_query_log_row_cap(tmp_path):
    path = tmp_path / "gs"
    gs = GraphStore(path=str(path))
    try:
        gs._wal._query_log_max_rows = 50
        for i in range(120):
            gs.execute('COUNT NODES')
        gs._wal.maybe_auto_checkpoint()
        count = gs._conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
        assert count <= 50
    finally:
        gs.close()
