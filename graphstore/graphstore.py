"""GraphStore - main entry point for the graphstore package."""

import os
import time
import sqlite3
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from concurrent.futures import Future
from graphstore.core.store import CoreStore
from graphstore.core.schema import SchemaRegistry
from graphstore.core.types import Result
from graphstore.dsl.parser import parse
from graphstore.dsl.tagger import infer_tag, infer_phase
from graphstore.dsl.executor import Executor
from graphstore.dsl.executor_system import SystemExecutor
from graphstore.dsl import ast_nodes
from graphstore.persistence.database import open_database
from graphstore.persistence.deserializer import load as _load_fn
from graphstore.wal import WALManager
from graphstore.core.scheduler import OptimizerScheduler
from graphstore.core.errors import OptimizationInProgress
from graphstore.dsl.handlers import is_write_op
from graphstore.core.memory import estimate as _estimate_memory
from graphstore.config import GraphStoreConfig, load_config, merge_kwargs

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

    _UNSET = object()

    def __init__(self, path: str | None = None, ceiling_mb=_UNSET,
                 embedder=_UNSET, allow_system_queries: bool = True,
                 voice: bool = False, ingest_root=_UNSET,
                 vault=_UNSET, retention=_UNSET,
                 config: GraphStoreConfig | None = None,
                 config_path: str | None = None,
                 threaded: bool = False):
        # Load config: explicit object > explicit path > env var > db dir > defaults
        if config is not None:
            self._config = config
        elif config_path is not None:
            self._config = load_config(config_path)
        elif os.environ.get("GRAPHSTORE_CONFIG"):
            self._config = load_config(os.environ["GRAPHSTORE_CONFIG"])
        elif path is not None:
            self._config = load_config(Path(path) / "graphstore.json")
        else:
            self._config = GraphStoreConfig()

        # Only merge kwargs that were explicitly passed (not sentinel)
        overrides = {}
        if ceiling_mb is not self._UNSET:
            overrides["ceiling_mb"] = ceiling_mb
        if embedder is not self._UNSET:
            overrides["embedder"] = embedder
        if ingest_root is not self._UNSET:
            overrides["ingest_root"] = ingest_root
        if vault is not self._UNSET:
            overrides["vault"] = vault
        if retention is not self._UNSET:
            overrides["retention"] = retention
        if overrides:
            self._config = merge_kwargs(self._config, **overrides)
        cfg = self._config

        self._path = Path(path) if path else None
        self._ceiling_bytes = cfg.core.ceiling_mb * 1_000_000
        self._allow_system = allow_system_queries
        self._ingest_root = cfg.server.ingest_root
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
        emb_cfg = cfg.vector.embedder
        if embedder is not self._UNSET and embedder is not None and not isinstance(embedder, str):
            self._embedder = embedder  # custom Embedder instance
        elif emb_cfg == "none":
            self._embedder = None
        elif emb_cfg in ("default", "model2vec"):
            try:
                from graphstore.embedding.model2vec_embedder import Model2VecEmbedder
                self._embedder = Model2VecEmbedder()
            except Exception:
                self._embedder = None
        else:
            self._embedder = None

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
            self._store = CoreStore(ceiling_bytes=self._ceiling_bytes,
                                    capacity=cfg.core.initial_capacity)
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
        self._executor.cost_threshold = cfg.dsl.cost_threshold
        retention_dict = {
            "blob_warm_days": cfg.retention.blob_warm_days,
            "blob_archive_days": cfg.retention.blob_archive_days,
            "blob_delete_days": cfg.retention.blob_delete_days,
        }
        self._sys_executor = SystemExecutor(self._store, self._schema, self._conn,
                                               embedder=self._embedder,
                                               vector_store=self._vector_store,
                                               document_store=self._document_store,
                                               retention=retention_dict)

        # WAL manager
        self._wal = WALManager(
            self._conn, self._store, self._schema, self._executor,
            wal_hard_limit=cfg.persistence.wal_hard_limit,
            auto_checkpoint_threshold=cfg.persistence.auto_checkpoint_threshold,
            log_retention_days=cfg.persistence.log_retention_days,
        )

        # Replay WAL (must happen after executor is created)
        if self._path and self._conn:
            self._wal.replay()

        # Optimizer scheduler
        self._optimizer = OptimizerScheduler(
            self._store, self._vector_store, self._document_store,
            auto_optimize=cfg.dsl.auto_optimize,
            optimize_interval=cfg.dsl.optimize_interval,
        )

        # Vault: markdown note system
        vault_path = cfg.vault.path if cfg.vault.enabled else None
        if vault_path:
            from graphstore.vault.manager import VaultManager
            from graphstore.vault.sync import VaultSync
            from graphstore.vault.executor import VaultExecutor
            self._vault_manager = VaultManager(vault_path)
            self._vault_sync = VaultSync(
                self._vault_manager, self._store, self._schema,
                self._embedder, self._vector_store, self._document_store
            )
            # Initial sync
            self._vault_sync.sync_all()
            # Wire vault executor into DSL executor
            self._vault_executor = VaultExecutor(
                self._vault_manager, self._vault_sync, self._store,
                self._embedder, self._vector_store
            )
            self._executor._vault_executor = self._vault_executor
        else:
            self._vault_manager = None
            self._vault_sync = None
            self._vault_executor = None

        # Trace binding for log layer
        self._active_trace: str | None = None

        # Command queue for thread-safe access
        self._threaded = threaded
        self._queue = None
        if threaded:
            from graphstore.core.queue import CommandQueue
            self._queue = CommandQueue(self._execute_internal)

    def execute(self, query: str) -> Result:
        """Execute a DSL query. Thread-safe if threaded=True."""
        if self._queue is not None:
            return self._queue.submit(query)
        return self._execute_internal(query)

    def _execute_internal(self, query: str) -> Result:
        """Execute a DSL query directly (no queue). Internal use only."""
        if self._optimizer.optimizing:
            raise OptimizationInProgress()

        self._optimizer.maybe_optimize()

        tag, phase, source, trace_id = "system", "system", getattr(self, '_current_source', 'user'), self._active_trace

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
            tag = infer_tag(ast)
            phase = infer_phase(tag)

            # Route to correct executor
            if isinstance(ast, _SYS_TYPES):
                if not self._allow_system:
                    raise PermissionError("System queries are disabled")

                # Special handling for checkpoint
                if isinstance(ast, ast_nodes.SysCheckpoint):
                    self.checkpoint()
                    result = Result(kind="ok", data=None, count=0)
                elif isinstance(ast, ast_nodes.SysWal) and ast.action == "REPLAY":  # type: ignore[attr-defined]
                    self._wal.replay()
                    result = Result(kind="ok", data=None, count=0)
                elif isinstance(ast, ast_nodes.SysOptimize):
                    self._optimizer._optimizing = True
                    try:
                        result = self._sys_executor.execute(ast)
                    finally:
                        self._optimizer._optimizing = False
                        self._optimizer._needs_optimize = False
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
                is_write = is_write_op(ast)

                if is_write and self._conn:
                    self._wal.append(query)

                result = self._executor.execute(ast)

                if is_write:
                    if self._conn:
                        self._wal.maybe_auto_checkpoint()
                    self._optimizer.on_write()

            elapsed = (time.perf_counter_ns() - start) // 1000
            result.elapsed_us = elapsed

            self._wal.log_query(query, elapsed, result.count, None,
                                tag=tag, trace_id=trace_id, source=source, phase=phase)
            self._wal.emit_event(query, elapsed, result.count, None,
                                 tag=tag, trace_id=trace_id, source=source, phase=phase)

            return result

        except Exception as e:
            elapsed = (time.perf_counter_ns() - start) // 1000
            self._wal.log_query(query, elapsed, 0, str(e),
                                tag=tag, trace_id=trace_id, source=source, phase=phase)
            self._wal.emit_event(query, elapsed, 0, str(e),
                                 tag=tag, trace_id=trace_id, source=source, phase=phase)
            raise

    def execute_batch(self, queries: list[str]) -> list[Result]:
        """Execute multiple queries. Each is independent (not transactional)."""
        return [self.execute(q) for q in queries]

    def submit_background(self, query: str) -> "Future":
        """Submit a background query (low priority). Only available in threaded mode.
        Returns a Future that resolves to a Result."""
        if self._queue is None:
            raise RuntimeError("submit_background requires GraphStore(threaded=True)")
        return self._queue.submit_background(query)

    def checkpoint(self) -> None:
        """Force persist to disk. No-op if path is None."""
        if self._conn is None:
            return
        self._wal.checkpoint(vector_store=self._vector_store)

    def bind_trace(self, trace_id: str) -> None:
        """Set active trace ID for log correlation."""
        self._active_trace = trace_id

    def discard_trace(self) -> None:
        """Clear active trace ID."""
        self._active_trace = None

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
        if self._queue is not None:
            self._queue.shutdown()
            self._queue = None
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
            self._executor._vector_store = self._vector_store
            if hasattr(self, '_optimizer'):
                self._optimizer.sync_vector_store(self._vector_store)
        return self._vector_store

