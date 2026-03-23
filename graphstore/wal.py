"""Write-Ahead Log management for graphstore persistence."""

import time
import sqlite3

from graphstore.core.errors import GraphStoreError
from graphstore.dsl.parser import parse
from graphstore.dsl import ast_nodes
from graphstore.persistence.serializer import checkpoint as _checkpoint_fn


class WALManager:
    """Manages WAL append, replay, auto-checkpoint, and query log rotation."""

    def __init__(self, conn: sqlite3.Connection | None, store, schema,
                 executor, wal_hard_limit: int, auto_checkpoint_threshold: int,
                 log_retention_days: int):
        self._conn = conn
        self._store = store
        self._schema = schema
        self._executor = executor
        self._wal_hard_limit = wal_hard_limit
        self._auto_checkpoint_threshold = auto_checkpoint_threshold
        self._log_retention_days = log_retention_days

    def append(self, statement: str) -> None:
        conn = self._conn
        if conn is None:
            return
        row = conn.execute("SELECT COUNT(*) FROM wal").fetchone()
        if row and row[0] >= self._wal_hard_limit:
            try:
                self.checkpoint()
            except Exception:
                pass
            row = conn.execute("SELECT COUNT(*) FROM wal").fetchone()
            if row and row[0] >= self._wal_hard_limit:
                raise GraphStoreError(
                    f"WAL exceeds hard limit ({self._wal_hard_limit} entries). "
                    "Checkpoint failed - check disk space."
                )
        conn.execute(
            "INSERT INTO wal (timestamp, statement) VALUES (?, ?)",
            (time.time(), statement)
        )
        conn.commit()

    def replay(self) -> None:
        conn = self._conn
        if conn is None:
            return
        rows = conn.execute("SELECT statement FROM wal ORDER BY seq").fetchall()
        if not rows:
            return
        for (statement,) in rows:
            try:
                ast = parse(statement)
                if isinstance(ast, ast_nodes.CreateNode):
                    ast = ast_nodes.UpsertNode(id=ast.id, fields=ast.fields)
                self._executor.execute(ast)
            except Exception:
                pass
        if rows:
            _checkpoint_fn(self._store, self._schema, conn)

    def checkpoint(self) -> None:
        conn = self._conn
        if conn is None:
            return
        self._store.vectors = getattr(self, '_vector_store', None)
        _checkpoint_fn(self._store, self._schema, conn)

    def maybe_auto_checkpoint(self) -> None:
        conn = self._conn
        if conn is None:
            return
        row = conn.execute("SELECT COUNT(*) FROM wal").fetchone()
        if row and row[0] > self._auto_checkpoint_threshold:
            self.checkpoint()
        self._rotate_query_log()

    def log_query(self, query: str, elapsed_us: int, result_count: int, error: str | None) -> None:
        conn = self._conn
        if conn is None:
            return
        try:
            conn.execute(
                "INSERT INTO query_log (timestamp, query, elapsed_us, result_count, error) VALUES (?,?,?,?,?)",
                (time.time(), query, elapsed_us, result_count, error)
            )
            conn.commit()
        except Exception:
            pass

    def _rotate_query_log(self) -> None:
        conn = self._conn
        if conn is None:
            return
        try:
            cutoff = time.time() - self._log_retention_days * 86400
            conn.execute("DELETE FROM query_log WHERE timestamp < ?", (cutoff,))
            conn.commit()
        except Exception:
            pass
