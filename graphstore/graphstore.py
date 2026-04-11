"""GraphStore - main entry point for the graphstore package."""

import os
import time
import sqlite3
import logging
from contextlib import contextmanager
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
from graphstore.cron import CronScheduler
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
        queued: Install a single-worker submission queue in front of the write
                path. Makes the instance safe to share across caller threads -
                each execute() is serialized through one daemon worker (NOT
                concurrent execution, the storage engine is single-writer).
                Also starts the cron scheduler when a persistent path is set.
                Default False = caller is responsible for single-threaded use.
    """

    _UNSET = object()

    def __init__(self, path: str | None = None, ceiling_mb=_UNSET,
                 embedder=_UNSET, allow_system_queries: bool = True,
                 voice: bool = False, ingest_root=_UNSET,
                 vault=_UNSET, retention=_UNSET,
                 config: GraphStoreConfig | None = None,
                 config_path: str | None = None,
                 queued: bool = False,
                 ingestors: dict | None = None,
                 chunker=None,
                 stt=None,
                 tts=None):
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
        self._stt = None
        self._tts = None

        # Custom STT/TTS override built-ins; voice=True enables built-ins when not overridden
        if stt is not None:
            self._stt = stt
        elif voice:
            try:
                from graphstore.voice.stt import MoonshineSTT
                self._stt = MoonshineSTT()
            except ImportError:
                pass
        if tts is not None:
            self._tts = tts
        elif voice:
            try:
                from graphstore.voice.tts import PiperTTS
                self._tts = PiperTTS()
            except ImportError:
                pass

        # Initialize embedder (local; placed into RuntimeState below)
        emb_cfg = cfg.vector.embedder
        if embedder is not self._UNSET and embedder is not None and not isinstance(embedder, str):
            _embedder = embedder
        elif emb_cfg == "none":
            _embedder = None
        elif emb_cfg in ("default", "model2vec"):
            try:
                from graphstore.embedding.model2vec_embedder import Model2VecEmbedder
                _embedder = Model2VecEmbedder(model_name=cfg.vector.model2vec_model)
            except Exception:
                _embedder = None
        else:
            _embedder = None

        # Wire model cache directory from config
        if cfg.vector.model_cache_dir:
            from graphstore.registry.installer import set_cache_dir
            set_cache_dir(cfg.vector.model_cache_dir)

        # Ingestor registry - only constructed when custom ingestors passed.
        # When None, the handler falls through to router.ingest_file() (preserves "direct" fast-path).
        self._ingestor_registry = None
        if ingestors:
            from graphstore.ingest.registry import IngestorRegistry
            self._ingestor_registry = IngestorRegistry()
            for ext, inst in ingestors.items():
                if not hasattr(inst, 'supported_extensions') or not inst.supported_extensions:
                    inst.supported_extensions = [ext]
                if not hasattr(inst, 'name') or not inst.name:
                    inst.name = f"custom_{ext}"
                # Directly map the dict key extension - preserves user intent
                self._ingestor_registry._instances[inst.name] = inst
                self._ingestor_registry._ext_map[ext] = inst.name

        # Custom chunker (None → handler uses chunk_by_heading from chunker.py)
        self._chunker = chunker

        # Store / schema / conn / vector_store - local vars folded into RuntimeState below
        p = self._path
        _vector_store = None
        if p is not None:
            p.mkdir(parents=True, exist_ok=True)
            db_file = p / "graphstore.db"
            _conn = open_database(db_file, busy_timeout_ms=cfg.persistence.busy_timeout_ms)
            _store, _schema = _load_fn(_conn)
            _store._ceiling_bytes = self._ceiling_bytes
            if hasattr(_store, 'vectors') and _store.vectors is not None:
                _vector_store = _store.vectors
        else:
            _conn = None
            _store = CoreStore(ceiling_bytes=self._ceiling_bytes,
                               capacity=cfg.core.initial_capacity)
            _schema = SchemaRegistry()

        # DocumentStore: separate SQLite, always on disk
        from graphstore.document.store import DocumentStore
        if p is not None:
            doc_db = os.path.join(str(p), "documents.db")
        else:
            doc_db = None
        _document_store = DocumentStore(doc_db, fts_tokenizer=cfg.document.fts_tokenizer)

        import collections
        self._similarity_buffer = collections.deque(
            maxlen=cfg.evolution.similarity_buffer_size
        )

        from graphstore.core.runtime import RuntimeState
        self._runtime = RuntimeState(
            store=_store, schema=_schema,
            vector_store=_vector_store, document_store=_document_store,
            embedder=_embedder, conn=_conn,
            similarity_buffer=self._similarity_buffer,
        )

        # Compaction sentinel recovery Phase 2: DocStore orphan cleanup.
        if self._runtime.conn is not None:
            _sentinel_row = self._runtime.conn.execute(
                "SELECT value FROM metadata WHERE key='compaction_sentinel'"
            ).fetchone()
            if _sentinel_row is not None:
                import numpy as _np
                _n = self._runtime.store._next_slot
                _live = self._runtime.store.compute_live_mask(_n)
                _live_slots = set(int(s) for s in _np.nonzero(_live)[0])
                try:
                    self._runtime.document_store.orphan_cleanup(_live_slots)
                except Exception as _e:
                    logger.warning("compaction recovery orphan cleanup failed: %s", _e)
                self._runtime.conn.execute("DELETE FROM metadata WHERE key='compaction_sentinel'")
                self._runtime.conn.commit()
                logger.warning("compaction sentinel recovery complete")

        self._embedder_dirty = False

        # Create executors before WAL replay so _replay_wal can use them
        self._executor = Executor(self._runtime,
                                  ingest_root=self._ingest_root,
                                  ingestor_registry=self._ingestor_registry,
                                  chunker=self._chunker)
        self._executor._ensure_vector_store_cb = self._ensure_vector_store
        from graphstore.dsl.parser import set_cache_size
        set_cache_size(cfg.dsl.plan_cache_size)
        self._executor.cost_threshold = cfg.dsl.cost_threshold
        self._executor._fts_full_text = cfg.document.fts_full_text
        self._executor._recall_decay = cfg.dsl.recall_decay
        self._executor._remember_weights = cfg.dsl.remember_weights
        self._executor._chunk_max_size = cfg.document.chunk_max_size
        self._executor._summary_max_length = cfg.document.summary_max_length
        self._executor._chunk_overlap = cfg.document.chunk_overlap
        self._executor._search_oversample = cfg.vector.search_oversample
        self._executor._vision_model = cfg.document.vision_model
        self._executor._vision_base_url = cfg.document.vision_base_url
        self._executor._vision_max_tokens = cfg.document.vision_max_tokens
        retention_dict = {
            "blob_warm_days": cfg.retention.blob_warm_days,
            "blob_archive_days": cfg.retention.blob_archive_days,
            "blob_delete_days": cfg.retention.blob_delete_days,
        }
        self._sys_executor = SystemExecutor(self._runtime, retention=retention_dict)
        self._sys_executor._eviction_target_ratio = cfg.core.eviction_target_ratio
        # Evolution engine wired after engine is created (below)

        # Cron scheduler (requires queued mode for background execution)
        self._cron: CronScheduler | None = None

        # WAL manager
        self._wal = WALManager(
            self._runtime, self._executor,
            wal_hard_limit=cfg.persistence.wal_hard_limit,
            auto_checkpoint_threshold=cfg.persistence.auto_checkpoint_threshold,
            log_retention_days=cfg.persistence.log_retention_days,
        )
        self._sys_executor._wal_manager = self._wal

        # Replay WAL (must happen after executor is created)
        if self._path and self._runtime.conn:
            self._wal.replay()

        # Optimizer scheduler (evolution_engine wired after engine init below)
        self._optimizer = OptimizerScheduler(
            self._runtime,
            auto_optimize=cfg.dsl.auto_optimize,
            optimize_interval=cfg.dsl.optimize_interval,
            compact_threshold=cfg.core.compact_threshold,
            string_gc_threshold=cfg.core.string_gc_threshold,
            cache_gc_threshold=cfg.dsl.cache_gc_threshold,
        )

        # Vault: markdown note system
        vault_path = cfg.vault.path if cfg.vault.enabled else None
        if vault_path:
            from graphstore.vault.manager import VaultManager
            from graphstore.vault.sync import VaultSync
            from graphstore.vault.executor import VaultExecutor
            self._vault_manager = VaultManager(vault_path)
            self._vault_sync = VaultSync(
                self._vault_manager, self._runtime,
                summary_max_length=cfg.document.summary_max_length,
            )
            self._vault_sync.sync_all()
            self._vault_executor = VaultExecutor(
                self._vault_manager, self._vault_sync, self._runtime,
            )
            self._executor._vault_executor = self._vault_executor
        else:
            self._vault_manager = None
            self._vault_sync = None
            self._vault_executor = None

        # Trace binding for log layer
        self._active_trace: str | None = None

        self._counters: dict = {
            "execute_ok": 0,
            "execute_err": 0,
            "recall_hits": 0,
            "recall_misses": 0,
            "eviction_total": 0,
        }
        self._start_time: float = time.time()
        self._last_evolution_events: list = []

        # Evolution engine (Layer 5: metacognitive memory)
        from graphstore.evolve import EvolutionEngine
        self._evolution_engine = EvolutionEngine(self, self._runtime.conn, cfg.evolution)
        # Wire into scheduler and sys_executor
        self._optimizer._evolution_engine = self._evolution_engine
        self._sys_executor._evolution_engine = self._evolution_engine

        # Single-worker submission queue (not parallelism) - gives
        # thread-safe caller side even though writes stay serialized.
        self._queued = queued
        self._queue = None
        if queued:
            from graphstore.core.queue import CommandQueue
            self._queue = CommandQueue(self._execute_internal)

        # Start cron scheduler if queued and persistent
        if queued and self._conn is not None:
            self._cron = CronScheduler(self._conn, self.submit_background)
            self._cron.start()
            self._sys_executor._cron = self._cron

    def execute(self, query: str) -> Result:
        """Execute a DSL query. Thread-safe if queued=True."""
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

            self._counters["execute_ok"] += 1
            # Attach pending evolution events to result.meta and clear
            if self._last_evolution_events:
                result.meta["evolution"] = list(self._last_evolution_events)
                self._last_evolution_events.clear()

            return result

        except Exception as e:
            elapsed = (time.perf_counter_ns() - start) // 1000
            self._wal.log_query(query, elapsed, 0, str(e),
                                tag=tag, trace_id=trace_id, source=source, phase=phase)
            self._wal.emit_event(query, elapsed, 0, str(e),
                                 tag=tag, trace_id=trace_id, source=source, phase=phase)
            self._counters["execute_err"] += 1
            raise

    def execute_batch(self, queries: list[str]) -> list[Result]:
        """Execute multiple queries. Each is independent (not transactional)."""
        return [self.execute(q) for q in queries]

    @contextmanager
    def deferred_embeddings(self, batch_size: int = 64):
        """Defer vector embeddings during CREATE NODE, flushing in batches.

        While inside this context, write queries that would trigger embedding
        (CREATE NODE with schema EMBED field, or with DOCUMENT clause) append
        their (slot, text) pairs to a pending queue instead of calling the
        embedder per-node. The queue is auto-flushed when `batch_size` is
        reached, and any remaining pending pairs are flushed on context exit.

        This is a ~4-10x speedup on transformer embedders (EmbeddingGemma,
        Harrier, bge-*, etc.) where per-call overhead dominates. For static
        embedders (model2vec) it's roughly neutral.

        Example::

            with gs.deferred_embeddings(batch_size=128):
                for item in items:
                    gs.execute(f'CREATE NODE "{item.id}" text = "{item.text}" DOCUMENT "{item.text}"')
            # All embeddings flushed here.
        """
        executor = self._executor
        prev_defer = executor._defer_embeddings
        prev_batch_size = executor._embed_batch_size
        executor._defer_embeddings = True
        executor._embed_batch_size = batch_size
        try:
            yield
            executor.flush_pending_embeddings()
        finally:
            executor._defer_embeddings = prev_defer
            executor._embed_batch_size = prev_batch_size
            executor._pending_embeddings.clear()

    def submit_background(self, query: str) -> "Future":
        """Submit a background query (low priority). Only available in queued mode.
        Returns a Future that resolves to a Result."""
        if self._queue is None:
            raise RuntimeError("submit_background requires GraphStore(queued=True)")
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
        if self._cron is not None:
            self._cron.stop()
            self._cron = None
        if self._queue is not None:
            self._queue.shutdown()
            self._queue = None
        conn = self._runtime.conn
        if conn is not None:
            self.checkpoint()
            conn.close()
            self._runtime.conn = None
        if self._runtime.document_store is not None:
            self._runtime.document_store.close()
            self._runtime.document_store = None

    def reset_memory(self) -> Result:
        """Reset the in-memory graph (nodes, edges) to empty. SQLite is left alone."""
        if self._optimizer.optimizing:
            raise OptimizationInProgress()

        self._runtime.store = CoreStore(
            ceiling_bytes=self._ceiling_bytes,
            capacity=self._config.core.initial_capacity,
        )
        self._runtime.schema = SchemaRegistry()
        if self._runtime.vector_store is not None:
            from graphstore.vector.store import VectorStore
            self._runtime.vector_store = VectorStore(
                dims=self._runtime.vector_store.dims,
                capacity=self._config.core.initial_capacity,
            )

        import collections
        self._counters = {
            "execute_ok": 0,
            "execute_err": 0,
            "recall_hits": 0,
            "recall_misses": 0,
            "eviction_total": 0,
        }
        self._similarity_buffer.clear()
        self._last_evolution_events.clear()
        self._start_time = time.time()

        self.discard_trace()
        self._embedder_dirty = False
        return Result(kind="ok", data={"reset": "memory"}, count=0)

    def reset_store(self, preserve_config: bool = True) -> Result:
        """Gracefully drop and recreate SQLite internal tables, clear memory, restart store."""
        if self._conn:
            self.checkpoint()
            old_script = self.get_script()
            
            self._conn.execute("DELETE FROM blobs")
            self._conn.execute("DELETE FROM wal")
            self._conn.execute("DELETE FROM query_log")
            self._conn.execute("DELETE FROM metadata")
            self._conn.commit()
            
            if old_script:
                self.set_script(old_script)
                
            if self._document_store and self._document_store._conn:
                doc_conn = self._document_store._conn
                doc_conn.execute("DELETE FROM documents")
                doc_conn.execute("DELETE FROM summaries")
                doc_conn.execute("DELETE FROM doc_fts")
                doc_conn.execute("DELETE FROM images")
                doc_conn.execute("DELETE FROM doc_metadata")
                doc_conn.commit()

        self.reset_memory()
        # reset_memory creates a fresh CoreStore whose dirty flags start True;
        # since we just wiped SQLite there is nothing to flush, so clear them
        # now to avoid a no-op checkpoint write on the next close().
        self._store.reset_dirty_flags()
        return Result(kind="ok", data={"reset": "store"}, count=0)
        
    def reset_session(self) -> Result:
        """Clears all bindings, active context, trace IDs, and terminates active Voice sessions."""
        if self._store:
            self._store._active_context = None
        self.discard_trace()
        self._embedder_dirty = False
        self._current_source = 'user'
        if self._stt:
            self._stt.stop_listening()
        if self._conn:
            self._wal.maybe_auto_checkpoint()
        return Result(kind="ok", data={"reset": "session"}, count=0)

    def get_runtime_config(self) -> Result:
        """Return the running configuration as a dict."""
        import msgspec
        data = msgspec.json.decode(msgspec.json.encode(self._config))
        return Result(kind="config", data=data, count=1)
        
    def get_persisted_config(self) -> Result:
        """Read the graphstore.json from disk."""
        if self._path is None:
            return Result(kind="config", data=None, count=0)
        from graphstore.config import load_config
        cfg = load_config(self._path / "graphstore.json")
        import msgspec
        data = msgspec.json.decode(msgspec.json.encode(cfg))
        return Result(kind="config", data=data, count=1)
        
    def update_runtime_config(self, changes: dict) -> Result:
        """Update live features."""
        from graphstore.config import merge_kwargs
        self._config = merge_kwargs(self._config, **changes)
        
        if "ceiling_mb" in changes:
            self.ceiling_mb = changes["ceiling_mb"]
        if "cost_threshold" in changes:
            self.cost_threshold = changes["cost_threshold"]
        if "eviction_target_ratio" in changes:
            self._sys_executor._eviction_target_ratio = changes["eviction_target_ratio"]
            
        return Result(kind="ok", data={"updated": "runtime"}, count=1)
        
    def update_persisted_config(self, changes: dict) -> Result:
        """Update graphstore.json on disk."""
        if self._path is None:
            return Result(kind="error", data={"message": "No persistence path set"}, count=0)
            
        from graphstore.config import load_config, save_config, merge_kwargs
        cfg_path = self._path / "graphstore.json"
        
        if cfg_path.exists():
            cfg = load_config(cfg_path)
        else:
            cfg = self._config
            
        new_cfg = merge_kwargs(cfg, **changes)
        save_config(new_cfg, cfg_path)
        return Result(kind="ok", data={"updated": "persisted"}, count=1)

    @property
    def node_count(self) -> int:
        return self._store.node_count

    @property
    def edge_count(self) -> int:
        return self._store.edge_count

    @property
    def memory_usage(self) -> int:
        return _estimate_memory(self._store.node_count, self._store.edge_count)

    def get_all_nodes(self) -> list[dict]:
        """Return all live nodes. Used by server /api/graph endpoint."""
        return self._store.get_all_nodes()

    def get_all_edges(self) -> list[dict]:
        """Return all live edges. Used by server /api/graph endpoint."""
        return self._store.get_all_edges()

    @property
    def cost_threshold(self) -> int:
        """DSL query cost threshold. Queries exceeding this are rejected."""
        return self._executor.cost_threshold

    @cost_threshold.setter
    def cost_threshold(self, value: int) -> None:
        self._executor.cost_threshold = value

    @property
    def ceiling_mb(self) -> int:
        """Memory ceiling in MB. Writes exceeding this raise CeilingExceeded."""
        return self._store._ceiling_bytes // 1_000_000

    @ceiling_mb.setter
    def ceiling_mb(self, value: int) -> None:
        self._store._ceiling_bytes = value * 1_000_000

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
    def _document_store(self):
        return self._runtime.document_store

    @property
    def _embedder(self):
        return self._runtime.embedder

    @property
    def _conn(self):
        return self._runtime.conn

    def __enter__(self) -> "GraphStore":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _ensure_vector_store(self, dims: int):
        """Lazily initialize vector store with given dimensionality."""
        if self._runtime.vector_store is None:
            from graphstore.vector.store import VectorStore
            self._runtime.vector_store = VectorStore(
                dims=dims, capacity=self._runtime.store._capacity,
            )
        return self._runtime.vector_store


