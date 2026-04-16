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
from graphstore.config import GraphStoreConfig, load_config, merge_kwargs, apply_env_overrides

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
                 ingest_root=_UNSET,
                 vault=_UNSET, retention=_UNSET,
                 config: GraphStoreConfig | None = None,
                 config_path: str | None = None,
                 queued: bool = False,
                 auto_optimize=_UNSET,
                 enable_wal=_UNSET,
                 strict_recovery: bool = False,
                 ingestors: dict | None = None,
                 chunker=None,
                 remember_weights=_UNSET,
                 recall_decay=_UNSET,
                 search_oversample=_UNSET,
                 similarity_threshold=_UNSET,
                 duplicate_threshold=_UNSET,
                 recency_half_life_days=_UNSET,
                 similar_to_oversample=_UNSET,
                 lexical_search_oversample=_UNSET,
                 fusion_method=_UNSET,
                 rrf_k=_UNSET,
                 nucleus_expansion=_UNSET,
                 nucleus_hops=_UNSET,
                 nucleus_neighbors_per_hop=_UNSET,
                 nucleus_min_text_length=_UNSET,
                 nucleus_allowed_kinds=_UNSET,
                 sentence_query_expansion=_UNSET,
                 enable_sentence_nodes=_UNSET,
                 enable_rollback=_UNSET,
                 graph_signal_enabled=_UNSET,
                 entity_extractor=_UNSET,
                 entity_model_dir=_UNSET,
                 entity_score_threshold=_UNSET,
                 entity_max_length=_UNSET,
                 reranker=_UNSET,
                 reranker_model_dir=_UNSET,
                 reranker_projector_path=_UNSET,
                 reranker_max_length=_UNSET,
                 reranker_gpu_layers=_UNSET,
                 quantize_binary=_UNSET,
                 use_compression=_UNSET,
                 initial_capacity=_UNSET,
                 ):
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

        # Layer 3: env var overrides (GRAPHSTORE_SECTION_FIELD)
        self._config = apply_env_overrides(self._config)

        from graphstore.core.compute_profile import configure as _configure_profile, describe_profile
        cc = self._config.compute
        _configure_profile(
            profile=cc.profile,
            ner_threads=cc.ner_threads,
            embed_threads=cc.embed_threads,
            rerank_threads=cc.rerank_threads,
            embed_batch_size=cc.embed_batch_size,
            disable_load_scaling=cc.disable_load_scaling,
            disable_battery_scaling=cc.disable_battery_scaling,
        )
        logger.info("graphstore compute: %s", describe_profile())

        # Layer 4: constructor kwargs (highest priority). Every kwarg listed
        # here must also appear in config._KWARG_SHORTCUTS or be one of the
        # legacy keys (embedder, ingest_root, vault, retention, auto_optimize,
        # enable_wal) that merge_kwargs handles separately.
        _kwarg_names = (
            "ceiling_mb", "embedder", "ingest_root", "vault", "retention",
            "remember_weights", "recall_decay", "search_oversample",
            "similarity_threshold", "duplicate_threshold",
            "recency_half_life_days", "similar_to_oversample",
            "lexical_search_oversample", "fusion_method", "rrf_k",
            "nucleus_expansion", "nucleus_hops", "nucleus_neighbors_per_hop",
            "nucleus_min_text_length", "nucleus_allowed_kinds",
            "sentence_query_expansion", "enable_sentence_nodes",
            "enable_rollback", "graph_signal_enabled", "entity_extractor",
            "entity_model_dir", "auto_optimize", "enable_wal",
            "entity_score_threshold", "entity_max_length", "reranker",
            "reranker_model_dir", "reranker_projector_path",
            "reranker_max_length", "reranker_gpu_layers", "quantize_binary",
            "use_compression", "initial_capacity",
        )
        _loc = locals()
        overrides = {k: _loc[k] for k in _kwarg_names if _loc[k] is not self._UNSET}
        if overrides:
            self._config = merge_kwargs(self._config, **overrides)
        cfg = self._config

        self._path = Path(path) if path else None
        self._ceiling_bytes = cfg.core.ceiling_mb * 1_000_000
        self._allow_system = allow_system_queries
        self._ingest_root = cfg.server.ingest_root

        # Initialize embedder (local; placed into RuntimeState below)
        _embedder = self._build_embedder(cfg, embedder)

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
            _store, _schema = _load_fn(_conn, use_compression=cfg.core.use_compression, db_path=p)
            _store._ceiling_bytes = self._ceiling_bytes
            if hasattr(_store, 'vectors') and _store.vectors is not None:
                _vector_store = _store.vectors
                _vector_store._path = str(p / "vectors.usearch")
        else:
            _conn = None
            _store = CoreStore(ceiling_bytes=self._ceiling_bytes,
                               capacity=cfg.core.initial_capacity,
                               use_compression=cfg.core.use_compression)
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

        self._embedder_dirty = False
        self._check_embedder_identity(_conn, _embedder)

        # Create executors before WAL replay so _replay_wal can use them
        self._executor = Executor(self._runtime,
                                  ingest_root=self._ingest_root,
                                  ingestor_registry=self._ingestor_registry,
                                  chunker=self._chunker)
        self._executor._ensure_vector_store_cb = self._ensure_vector_store
        from graphstore.dsl.parser import set_cache_size
        set_cache_size(cfg.dsl.plan_cache_size)
        self._wire_executor_from_cfg(self._executor, cfg)
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
            strict_recovery=strict_recovery,
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

    def _build_embedder(self, cfg, embedder_arg):
        """Resolve the embedder instance from kwarg + cfg. Explicit instance
        wins over cfg.vector.embedder name. Returns None when embedder is
        disabled or when an optional dependency is missing.
        """
        if embedder_arg is not self._UNSET and embedder_arg is not None and not isinstance(embedder_arg, str):
            return embedder_arg
        emb_cfg = cfg.vector.embedder
        if emb_cfg == "none":
            return None
        if emb_cfg in ("default", "model2vec"):
            try:
                from graphstore.embedding.model2vec_embedder import Model2VecEmbedder
                return Model2VecEmbedder(
                    model_name=cfg.vector.model2vec_model,
                    cache_dir=cfg.vector.model_cache_dir,
                )
            except ImportError:
                return None
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("embedder init failed: %s", e)
                return None
        if emb_cfg == "installed" or emb_cfg.startswith("installed:"):
            model_name = cfg.vector.embedder_model or (emb_cfg.split(":", 1)[-1] if ":" in emb_cfg else None)
            if not model_name:
                raise ValueError(
                    "embedder='installed' requires vector.embedder_model in config "
                    "or use 'installed:model-name' format"
                )
            from graphstore.registry.installer import load_installed_embedder
            return load_installed_embedder(
                model_name,
                dims=cfg.vector.embedder_dims,
                n_gpu_layers=cfg.vector.gpu_layers,
                max_length=cfg.vector.embedder_max_length,
            )
        return None

    def _wire_executor_from_cfg(self, executor, cfg) -> None:
        """Push every DSL / document / vector tuning knob from cfg onto the
        executor instance. Keeps __init__ readable by factoring out the ~40
        attribute assignments.
        """
        executor.cost_threshold = cfg.dsl.cost_threshold
        d = cfg.dsl
        doc = cfg.document
        vec = cfg.vector
        executor._fts_full_text = doc.fts_full_text
        executor._recall_decay = d.recall_decay
        executor._remember_weights = d.remember_weights
        executor._fusion_method = d.fusion_method
        executor._rrf_k = d.rrf_k
        executor._recency_half_life_days = d.recency_half_life_days
        executor._similar_to_oversample = d.similar_to_oversample
        executor._lexical_search_oversample = d.lexical_search_oversample
        executor._nucleus_expansion = d.nucleus_expansion
        executor._nucleus_hops = d.nucleus_hops
        executor._nucleus_neighbors_per_hop = d.nucleus_neighbors_per_hop
        executor._nucleus_min_text_length = d.nucleus_min_text_length
        executor._nucleus_allowed_kinds = d.nucleus_allowed_kinds
        executor._sentence_query_expansion = d.sentence_query_expansion
        executor._enable_sentence_nodes = d.enable_sentence_nodes
        executor._enable_rollback = d.enable_rollback
        executor._graph_signal_enabled = d.graph_signal_enabled
        executor._entity_model_dir = d.entity_model_dir
        executor._entity_score_threshold = d.entity_score_threshold
        executor._entity_max_length = d.entity_max_length
        executor._reranker = self._build_reranker(cfg)
        executor._chunk_max_size = doc.chunk_max_size
        executor._summary_max_length = doc.summary_max_length
        executor._chunk_overlap = doc.chunk_overlap
        executor._search_oversample = vec.search_oversample
        executor._vision_model = doc.vision_model
        executor._vision_base_url = doc.vision_base_url
        executor._vision_max_tokens = doc.vision_max_tokens

    def execute(self, query: str) -> Result:
        """Execute a DSL query. Thread-safe if queued=True."""
        if self._queue is not None:
            return self._queue.submit(query)
        return self._execute_internal(query)

    def _build_reranker(self, cfg):
        """Build reranker from config. Returns None if not configured."""
        backend = (cfg.dsl.reranker or "").lower()
        if not backend:
            return None
        if backend == "gguf":
            from graphstore.embedding.reranker import GGUFReranker
            return GGUFReranker(
                model_path=cfg.dsl.reranker_model_dir or "",
                projector_path=cfg.dsl.reranker_projector_path,
                n_gpu_layers=cfg.dsl.reranker_gpu_layers,
            )
        if backend == "flashrank":
            from graphstore.embedding.reranker import FlashRankReranker
            return FlashRankReranker(max_length=cfg.dsl.reranker_max_length)
        if backend == "onnx":
            from graphstore.embedding.reranker import OnnxReranker
            return OnnxReranker(
                model_dir=cfg.dsl.reranker_model_dir or "",
                max_length=cfg.dsl.reranker_max_length,
            )
        return None

    def _execute_internal(self, query: str) -> Result:
        """Execute a DSL query directly (no queue). Internal use only."""
        if self._optimizer.optimizing:
            raise OptimizationInProgress()

        self._optimizer.maybe_optimize()

        tag, phase, source, trace_id = "system", "system", "user", self._active_trace

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
                        self._update_embedder_identity()
            else:
                is_write = is_write_op(ast)

                if is_write and self._conn and self._config.persistence.enable_wal:
                    self._wal.append(query)

                result = self._executor.execute(ast)

                if is_write:
                    if self._conn and self._config.persistence.enable_wal:
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
                    # Use DSL escaping to prevent injection from special chars
                    safe_id = item.id.replace('"', '\\"')
                    safe_text = item.text.replace('"', '\\"').replace("\\\\", "\\\\")
                    gs.execute(f'CREATE NODE "{safe_id}" text = "{safe_text}" DOCUMENT "{safe_text}"')
            # All embeddings flushed here.
        """
        executor = self._executor
        prev_defer = executor._defer_embeddings
        prev_batch_size = executor._embed_batch_size
        executor._defer_embeddings = True
        executor._embed_batch_size = batch_size
        try:
            yield
        except BaseException:
            try:
                executor.flush_pending_embeddings()
            except Exception as flush_err:
                pending = list(getattr(executor, "_pending_embeddings", []))
                slots = [s for s, _ in pending]
                import logging as _logging
                _logging.getLogger(__name__).error(
                    "deferred_embeddings: flush failed during exception unwind; "
                    "%d slot(s) have no vector: %s (flush error: %s)",
                    len(slots), slots[:20], flush_err,
                )
                executor._pending_embeddings.clear()
            raise
        else:
            executor.flush_pending_embeddings()
        finally:
            executor._defer_embeddings = prev_defer
            executor._embed_batch_size = prev_batch_size

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
            vector_path = os.path.join(str(self._path), "vectors.usearch") if self._path else None
            self._runtime.vector_store = VectorStore(
                dims=self._runtime.vector_store.dims,
                capacity=self._config.core.initial_capacity,
                quantize_binary=self._config.vector.quantize_binary,
                path=vector_path
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
        """Clears active context, trace IDs, and embedder-dirty flag."""
        if self._store:
            self._store._active_context = None
        self.discard_trace()
        self._embedder_dirty = False
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
        vs = self._runtime.vector_store
        if vs is None:
            from graphstore.vector.store import VectorStore
            vector_path = os.path.join(str(self._path), "vectors.usearch") if self._path else None
            vs = VectorStore(
                dims=dims, 
                capacity=self._runtime.store._capacity,
                quantize_binary=self._config.vector.quantize_binary,
                path=vector_path
            )
            self._runtime.vector_store = vs
            self._record_embedder_identity(dims)
        elif vs.dims != dims:
            from graphstore.core.errors import EmbedderRequired
            raise EmbedderRequired(
                f"Vector index has {vs.dims} dims but embedder produces {dims}. "
                f"Either use the same embedder or run SYS REEMBED."
            )
        return vs

    def _check_embedder_identity(self, conn, embedder):
        """On open, verify current embedder matches what the database was built
        with. Sets self._embedder_dirty on mismatch; SIMILAR/REMEMBER will refuse
        until SYS REEMBED is run."""
        if conn is None or embedder is None:
            return
        from graphstore.persistence.database import get_metadata
        stored = get_metadata(conn, "embedder_name")
        if stored is None:
            return
        current = embedder.name
        if stored != current:
            stored_dims = get_metadata(conn, "embedder_dims") or "?"
            import logging
            logging.getLogger(__name__).warning(
                "embedder mismatch: database was built with '%s' (%s dims), "
                "current embedder is '%s'. Run SYS REEMBED to re-encode; "
                "until then SIMILAR and REMEMBER will raise.",
                stored, stored_dims, current,
            )
            self._embedder_dirty = True

    def _record_embedder_identity(self, dims: int):
        """Store embedder name + dims in database metadata on first vector write."""
        conn = self._runtime.conn
        if conn is None:
            return
        from graphstore.persistence.database import get_metadata, set_metadata
        if get_metadata(conn, "embedder_name") is not None:
            return
        emb = self._runtime.embedder
        name = emb.name if emb else "unknown"
        set_metadata(conn, "embedder_name", name)
        set_metadata(conn, "embedder_dims", str(dims))

    def _update_embedder_identity(self):
        """Update stored embedder identity after SYS REEMBED."""
        conn = self._runtime.conn
        emb = self._runtime.embedder
        if conn is None or emb is None:
            return
        from graphstore.persistence.database import set_metadata
        set_metadata(conn, "embedder_name", emb.name)
        set_metadata(conn, "embedder_dims", str(emb.dims))




