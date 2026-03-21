"""GraphStore - main entry point for the graphstore package."""

import os
import time
import sqlite3
from pathlib import Path

from graphstore.core.store import CoreStore
from graphstore.core.schema import SchemaRegistry
from graphstore.core.types import Result
from graphstore.dsl.parser import parse
from graphstore.dsl.executor import Executor
from graphstore.dsl.executor_system import SystemExecutor
from graphstore.dsl import ast_nodes
from graphstore.persistence.database import open_database
from graphstore.persistence.serializer import checkpoint as _checkpoint_fn
from graphstore.persistence.deserializer import load as _load_fn
from graphstore.core.errors import GraphStoreError
from graphstore.core.memory import estimate as _estimate_memory

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
        embedder: Embedder instance, "default" for Model2Vec, or None to disable.
        allow_system_queries: If False, SYS queries raise PermissionError.
    """

    def __init__(self, path: str | None = None, ceiling_mb: int = 256,
                 embedder="default", allow_system_queries: bool = True,
                 voice: bool = False, ingest_root: str | None = None):
        self._path = Path(path) if path else None
        self._ceiling_bytes = ceiling_mb * 1_000_000
        self._allow_system = allow_system_queries
        self._ingest_root = ingest_root
        self._conn: sqlite3.Connection | None = None
        self._stt = None
        self._tts = None

        if voice:
            try:
                from graphstore.voice.stt import MoonshineSTT
                from graphstore.voice.tts import PiperTTS
                self._stt = MoonshineSTT()
                self._tts = PiperTTS()
            except ImportError:
                pass  # Voice not installed; speak/listen will raise on use

        # Initialize embedder
        if embedder == "default":
            try:
                from graphstore.embedding.model2vec_embedder import Model2VecEmbedder
                self._embedder = Model2VecEmbedder()
            except Exception:
                self._embedder = None
        elif embedder is None:
            self._embedder = None
        else:
            self._embedder = embedder  # custom Embedder instance

        # Vector store (lazy init on first vector operation)
        self._vector_store = None

        # Initialize store
        p = self._path
        if p is not None:
            p.mkdir(parents=True, exist_ok=True)
            db_file = p / "graphstore.db"
            self._conn = open_database(db_file)

            # Try to load existing data
            self._store, self._schema = _load_fn(self._conn)
            self._store._ceiling_bytes = self._ceiling_bytes
            # Restore vector store from persisted state
            if hasattr(self._store, 'vectors') and self._store.vectors is not None:
                self._vector_store = self._store.vectors
        else:
            self._store = CoreStore(ceiling_bytes=self._ceiling_bytes)
            self._schema = SchemaRegistry()

        # DocumentStore: separate SQLite, always on disk
        from graphstore.document.store import DocumentStore
        if p is not None:
            doc_db = os.path.join(str(p), "documents.db")
        else:
            doc_db = None  # DocumentStore uses temp file
        self._document_store = DocumentStore(doc_db)

        # Embedder dirty flag (set when SYS SET EMBEDDER changes the embedder)
        self._embedder_dirty = False

        # Create executors before WAL replay so _replay_wal can use them
        self._executor = Executor(self._store, self._schema,
                                  embedder=self._embedder,
                                  vector_store=self._vector_store,
                                  document_store=self._document_store,
                                  ingest_root=self._ingest_root)
        self._executor._ensure_vector_store_cb = self._ensure_vector_store
        self._sys_executor = SystemExecutor(self._store, self._schema, self._conn,
                                               embedder=self._embedder,
                                               vector_store=self._vector_store,
                                               document_store=self._document_store)

        # Replay WAL (must happen after executor is created)
        if self._path and self._conn:
            self._replay_wal()

    def execute(self, query: str) -> Result:
        """Execute a single DSL query (user or system). Returns Result."""
        start = time.perf_counter_ns()

        # Sync vector store reference (may have been lazily created)
        if self._vector_store is not None:
            if self._executor._vector_store is not self._vector_store:
                self._executor._vector_store = self._vector_store
            if self._sys_executor._vector_store is not self._vector_store:
                self._sys_executor._vector_store = self._vector_store

        # Sync embedder dirty flag
        self._executor._embedder_dirty = self._embedder_dirty

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
                    # Sync vector store back from sys executor (rollback may change it)
                    if self._sys_executor._vector_store is not self._vector_store:
                        self._vector_store = self._sys_executor._vector_store
                        self._executor._vector_store = self._vector_store
                    # Clear dirty flag after successful SYS REEMBED
                    if isinstance(ast, ast_nodes.SysReembed):
                        self._embedder_dirty = False
                        self._executor._embedder_dirty = False
            else:
                # Check if it's a write - log to WAL before executing
                is_write = isinstance(ast, (
                    ast_nodes.CreateNode, ast_nodes.UpdateNode, ast_nodes.UpsertNode,
                    ast_nodes.DeleteNode, ast_nodes.DeleteNodes,
                    ast_nodes.CreateEdge, ast_nodes.UpdateEdge, ast_nodes.DeleteEdge, ast_nodes.DeleteEdges,
                    ast_nodes.Increment, ast_nodes.Batch,
                    ast_nodes.AssertStmt, ast_nodes.RetractStmt,
                    ast_nodes.UpdateNodes, ast_nodes.MergeStmt,
                    ast_nodes.PropagateStmt, ast_nodes.DiscardContext,
                    ast_nodes.IngestStmt, ast_nodes.ConnectNode,
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
        self._store.vectors = self._vector_store  # expose to serializer
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

    def speak(self, text: str) -> None:
        """Text-to-speech via Piper."""
        if not self._tts:
            raise ImportError("Voice not installed. Run: graphstore install-voice")
        self._tts.speak(text)

    def listen(self, on_text=None) -> None:
        """Start real-time STT via Moonshine."""
        if not self._stt:
            raise ImportError("Voice not installed. Run: graphstore install-voice")
        if on_text is None:
            raise ValueError("on_text callback is required")
        self._stt.start_listening(on_text)

    def stop_listening(self) -> None:
        """Stop real-time STT."""
        if self._stt:
            self._stt.stop_listening()

    def close(self) -> None:
        """Checkpoint + close sqlite connection."""
        conn = self._conn
        if conn is not None:
            self.checkpoint()
            conn.close()
            self._conn = None
        if self._document_store is not None:
            self._document_store.close()
            self._document_store = None

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

    def _ensure_vector_store(self, dims: int):
        """Lazily initialize vector store with given dimensionality."""
        if self._vector_store is None:
            from graphstore.vector.store import VectorStore
            self._vector_store = VectorStore(dims=dims, capacity=self._store._capacity)
            # Sync to executor
            self._executor._vector_store = self._vector_store
        return self._vector_store

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
