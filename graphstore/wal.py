"""Write-Ahead Log management for graphstore persistence."""

import time
import logging

logger = logging.getLogger(__name__)
_event_logger = logging.getLogger("graphstore.events")

from graphstore.core.errors import GraphStoreError
from graphstore.core.runtime import RuntimeState
from graphstore.dsl.parser import parse
from graphstore.dsl import ast_nodes
from graphstore.persistence.serializer import checkpoint as _checkpoint_fn


class WALManager:
    """Manages WAL append, replay, auto-checkpoint, and query log rotation."""

    def __init__(self, runtime: RuntimeState, executor,
                 wal_hard_limit: int, auto_checkpoint_threshold: int,
                 log_retention_days: int):
        self._runtime = runtime
        self._executor = executor
        self._wal_hard_limit = wal_hard_limit
        self._auto_checkpoint_threshold = auto_checkpoint_threshold
        self._log_retention_days = log_retention_days

    @property
    def _conn(self):
        return self._runtime.conn

    @property
    def _store(self):
        return self._runtime.store

    @property
    def _schema(self):
        return self._runtime.schema

    @property
    def _vector_store(self):
        return self._runtime.vector_store

    @property
    def pending_count(self) -> int:
        """Number of WAL entries not yet checkpointed."""
        if self._conn is None:
            return 0
        row = self._conn.execute("SELECT COUNT(*) FROM wal").fetchone()
        return row[0] if row else 0

    def append(self, statement: str) -> None:
        conn = self._conn
        if conn is None:
            return
        row = conn.execute("SELECT COUNT(*) FROM wal").fetchone()
        if row and row[0] >= self._wal_hard_limit:
            try:
                self.checkpoint()
            except Exception as e:
                logger.debug("WAL checkpoint before hard-limit failed: %s", e, exc_info=True)
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
            except Exception as e:
                logger.debug("WAL replay statement skipped: %s", e, exc_info=True)
        if rows:
            _checkpoint_fn(self._store, self._schema, conn, force=True)

    def checkpoint(self, vector_store=None) -> None:
        conn = self._conn
        if conn is None:
            return
        if self._vector_store is not None:
            self._store.vectors = self._vector_store
        _checkpoint_fn(self._store, self._schema, conn)
        self._store.reset_dirty_flags()

    def maybe_auto_checkpoint(self) -> None:
        conn = self._conn
        if conn is None:
            return
        row = conn.execute("SELECT COUNT(*) FROM wal").fetchone()
        if row and row[0] > self._auto_checkpoint_threshold:
            self.checkpoint()
        self._rotate_query_log()

    def log_query(self, query: str, elapsed_us: int, result_count: int, error: str | None,
                  tag: str | None = None, trace_id: str | None = None,
                  source: str = "user", phase: str | None = None) -> None:
        conn = self._conn
        if conn is None:
            return
        try:
            conn.execute(
                "INSERT INTO query_log (timestamp, query, elapsed_us, result_count, error, tag, trace_id, source, phase) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (time.time(), query, elapsed_us, result_count, error, tag, trace_id, source, phase)
            )
            conn.commit()
        except Exception as e:
            logger.debug("query log insert failed: %s", e, exc_info=True)

    def emit_event(self, query: str, elapsed_us: int, result_count: int, error: str | None,
                   tag: str | None = None, trace_id: str | None = None,
                   source: str = "user", phase: str | None = None) -> None:
        """Emit structured log event via graphstore.events logger."""
        _event_logger.info(
            "%s [%s] %dus %d results",
            tag or "unknown", source, elapsed_us, result_count,
            extra={
                "gs_query": query,
                "gs_tag": tag,
                "gs_source": source,
                "gs_trace_id": trace_id,
                "gs_phase": phase,
                "gs_elapsed_us": elapsed_us,
                "gs_result_count": result_count,
                "gs_error": error,
            },
        )

    def _rotate_query_log(self) -> None:
        conn = self._conn
        if conn is None:
            return
        try:
            cutoff = time.time() - self._log_retention_days * 86400
            conn.execute("DELETE FROM query_log WHERE timestamp < ?", (cutoff,))
            conn.commit()
        except Exception as e:
            logger.debug("query log rotation failed: %s", e, exc_info=True)
