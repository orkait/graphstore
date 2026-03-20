"""graphstore — In-memory typed graph database with DSL interface.

Public API:
    GraphStore   — main entry point (create, query, persist)
    Result       — query result dataclass
    Edge         — edge dataclass
    GraphStoreError, QueryError, NodeNotFound, NodeExists,
    CeilingExceeded, VersionMismatch, SchemaError,
    CostThresholdExceeded, BatchRollback — error hierarchy
"""

__version__ = "0.1.0"

import time
import sqlite3
from pathlib import Path

from graphstore.store import CoreStore
from graphstore.schema import SchemaRegistry
from graphstore.types import Result, Edge
from graphstore.dsl.parser import parse, clear_cache
from graphstore.dsl.executor import Executor
from graphstore.dsl.executor_system import SystemExecutor
from graphstore.dsl import ast_nodes
from graphstore.persistence.database import open_database
from graphstore.persistence.serializer import checkpoint as _checkpoint_fn
from graphstore.persistence.deserializer import load as _load_fn
from graphstore.errors import (
    GraphStoreError,
    QueryError,
    NodeNotFound,
    NodeExists,
    CeilingExceeded,
    VersionMismatch,
    SchemaError,
    CostThresholdExceeded,
    BatchRollback,
)
from graphstore.memory import estimate as _estimate_memory, DEFAULT_CEILING_BYTES

__all__ = [
    "GraphStore",
    "CoreStore",
    "SchemaRegistry",
    "Result",
    "Edge",
    "parse",
    "clear_cache",
    "Executor",
    "SystemExecutor",
    "GraphStoreError",
    "QueryError",
    "NodeNotFound",
    "NodeExists",
    "CeilingExceeded",
    "VersionMismatch",
    "SchemaError",
    "CostThresholdExceeded",
    "BatchRollback",
    "DEFAULT_CEILING_BYTES",
]

# All system AST types
_SYS_TYPES = tuple(
    getattr(ast_nodes, name)
    for name in dir(ast_nodes)
    if name.startswith("Sys")
)


class GraphStore:
    """In-memory typed graph database with DSL interface.

    Args:
        path: Directory for graphstore.db persistence.
              If None, in-memory only (no persistence, no WAL, no query log).
        ceiling_mb: Hard memory limit in MB. Raises CeilingExceeded on breach.
        allow_system_queries: If False, SYS queries raise PermissionError.
    """

    def __init__(self, path: str | None = None, ceiling_mb: int = 256,
                 allow_system_queries: bool = True):
        self._path = Path(path) if path else None
        self._ceiling_bytes = ceiling_mb * 1_000_000
        self._allow_system = allow_system_queries
        self._conn: sqlite3.Connection | None = None

        # Initialize store
        p = self._path
        if p is not None:
            p.mkdir(parents=True, exist_ok=True)
            db_file = p / "graphstore.db"
            self._conn = open_database(db_file)

            # Try to load existing data
            self._store, self._schema = _load_fn(self._conn)
            self._store._ceiling_bytes = self._ceiling_bytes
        else:
            self._store = CoreStore(ceiling_bytes=self._ceiling_bytes)
            self._schema = SchemaRegistry()

        # Create executors before WAL replay so _replay_wal can use them
        self._executor = Executor(self._store, self._schema)
        self._sys_executor = SystemExecutor(self._store, self._schema, self._conn)

        # Replay WAL (must happen after executor is created)
        if self._path and self._conn:
            self._replay_wal()

    def execute(self, query: str) -> Result:
        """Execute a single DSL query (user or system). Returns Result."""
        start = time.perf_counter_ns()

        try:
            ast = parse(query)

            # Route to correct executor
            if isinstance(ast, _SYS_TYPES):
                if not self._allow_system:
                    raise PermissionError("System queries are disabled")

                # Special handling for checkpoint
                if isinstance(ast, ast_nodes.SysCheckpoint):
                    self.checkpoint()
                    result = Result(kind="ok", data=None, count=0)
                elif isinstance(ast, ast_nodes.SysWal) and ast.action == "REPLAY":  # type: ignore[attr-defined]
                    self._replay_wal()
                    result = Result(kind="ok", data=None, count=0)
                else:
                    result = self._sys_executor.execute(ast)
            else:
                # Check if it's a write - log to WAL before executing
                is_write = isinstance(ast, (
                    ast_nodes.CreateNode, ast_nodes.UpdateNode, ast_nodes.UpsertNode,
                    ast_nodes.DeleteNode, ast_nodes.DeleteNodes,
                    ast_nodes.CreateEdge, ast_nodes.UpdateEdge, ast_nodes.DeleteEdge, ast_nodes.DeleteEdges,
                    ast_nodes.Increment, ast_nodes.Batch,
                ))

                if is_write and self._conn:
                    self._wal_append(query)

                result = self._executor.execute(ast)

                # Auto-checkpoint check
                if is_write and self._conn:
                    self._maybe_auto_checkpoint()

            elapsed = (time.perf_counter_ns() - start) // 1000
            result.elapsed_us = elapsed

            # Log query
            self._log_query(query, elapsed, result.count, None)

            return result

        except Exception as e:
            elapsed = (time.perf_counter_ns() - start) // 1000
            self._log_query(query, elapsed, 0, str(e))
            raise

    def execute_batch(self, queries: list[str]) -> list[Result]:
        """Execute multiple queries. Each is independent (not transactional)."""
        return [self.execute(q) for q in queries]

    def checkpoint(self) -> None:
        """Force persist to disk. No-op if path is None."""
        if self._conn is None:
            return
        _checkpoint_fn(self._store, self._schema, self._conn)

    def set_script(self, script: str) -> None:
        """Store a DSL script in metadata so the playground can load it."""
        conn = self._conn
        if conn is None:
            return
        from graphstore.persistence.database import set_metadata
        set_metadata(conn, "playground_script", script)

    def get_script(self) -> str | None:
        """Retrieve the stored playground script, if any."""
        conn = self._conn
        if conn is None:
            return None
        from graphstore.persistence.database import get_metadata
        return get_metadata(conn, "playground_script")

    def close(self) -> None:
        """Checkpoint + close sqlite connection."""
        conn = self._conn
        if conn is not None:
            self.checkpoint()
            conn.close()
            self._conn = None

    @property
    def node_count(self) -> int:
        return self._store.node_count

    @property
    def edge_count(self) -> int:
        return self._store.edge_count

    @property
    def memory_usage(self) -> int:
        return _estimate_memory(self._store.node_count, self._store.edge_count)

    def __enter__(self) -> "GraphStore":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # --- Internal ---

    _WAL_HARD_LIMIT = 100_000  # max WAL entries before rejecting writes

    def _wal_append(self, statement: str):
        """Append a mutation to the WAL table. Rejects if WAL exceeds hard limit."""
        conn = self._conn
        if conn is not None:
            row = conn.execute("SELECT COUNT(*) FROM wal").fetchone()
            if row and row[0] >= self._WAL_HARD_LIMIT:
                # Try checkpoint first to clear WAL
                try:
                    self.checkpoint()
                except Exception:
                    pass
                # Re-check after checkpoint
                row = conn.execute("SELECT COUNT(*) FROM wal").fetchone()
                if row and row[0] >= self._WAL_HARD_LIMIT:
                    raise GraphStoreError(
                        f"WAL exceeds hard limit ({self._WAL_HARD_LIMIT} entries). "
                        "Checkpoint failed - check disk space."
                    )
            conn.execute(
                "INSERT INTO wal (timestamp, statement) VALUES (?, ?)",
                (time.time(), statement)
            )
            conn.commit()

    def _replay_wal(self):
        """Replay WAL entries from sqlite."""
        conn = self._conn
        if conn is None:
            return
        rows = conn.execute("SELECT statement FROM wal ORDER BY seq").fetchall()
        if not rows:
            return

        for (statement,) in rows:
            try:
                ast = parse(statement)
                # During replay, treat CREATE as UPSERT for idempotency
                if isinstance(ast, ast_nodes.CreateNode):
                    ast = ast_nodes.UpsertNode(id=ast.id, fields=ast.fields)
                self._executor.execute(ast)
            except Exception:
                pass  # Skip failed statements during replay

        # Checkpoint to clean state
        if rows:
            _checkpoint_fn(self._store, self._schema, conn)

    def _maybe_auto_checkpoint(self):
        """Auto-checkpoint if WAL exceeds thresholds."""
        conn = self._conn
        if conn is None:
            return
        row = conn.execute("SELECT COUNT(*) FROM wal").fetchone()
        if row and row[0] > 50_000:
            self.checkpoint()
        # Log rotation: delete entries older than 7 days
        self._rotate_query_log()

    def _rotate_query_log(self):
        """Delete query_log entries older than 7 days."""
        conn = self._conn
        if conn is None:
            return
        try:
            cutoff = time.time() - 7 * 86400
            conn.execute("DELETE FROM query_log WHERE timestamp < ?", (cutoff,))
            conn.commit()
        except Exception:
            pass

    def _log_query(self, query: str, elapsed_us: int, result_count: int, error: str | None):
        """Log query to sqlite query_log table."""
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
            pass  # Don't fail on log errors
