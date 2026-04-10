"""Optimizer scheduling: health checks and auto-optimize between execute calls."""

import logging

logger = logging.getLogger(__name__)


class OptimizerScheduler:
    """Tracks write pressure and triggers optimization at safe points."""

    def __init__(self, store, vector_store, document_store,
                 auto_optimize: bool = False, optimize_interval: int = 500,
                 compact_threshold: float = 0.2, string_gc_threshold: float = 3.0,
                 cache_gc_threshold: int = 200, evolution_engine=None,
                 schema=None, conn=None):
        self._store = store
        self._vector_store = vector_store
        self._document_store = document_store
        self._auto_optimize = auto_optimize
        self._optimize_interval = optimize_interval
        self._compact_threshold = compact_threshold
        self._string_gc_threshold = string_gc_threshold
        self._cache_gc_threshold = cache_gc_threshold
        self._optimizing = False
        self._needs_optimize = False
        self._write_counter = 0
        self._evolution_engine = evolution_engine
        self._schema = schema
        self._conn = conn

    @property
    def optimizing(self) -> bool:
        return self._optimizing

    def on_write(self) -> None:
        """Called after each write operation. Increments counter and checks health."""
        self._write_counter += 1
        if self._auto_optimize and self._write_counter % self._optimize_interval == 0:
            self._check_health()

    def maybe_optimize(self) -> None:
        """Run optimization if pressure detected. Call at safe points (between execute calls)."""
        if not self._needs_optimize:
            return
        self._optimizing = True
        try:
            from graphstore.core.optimizer import optimize_all
            optimize_all(
                self._store, self._vector_store, self._document_store,
                schema=self._schema, conn=self._conn,
            )
        except Exception as e:
            logger.debug("auto-optimize failed: %s", e)
        finally:
            self._optimizing = False
            self._needs_optimize = False

    def sync_vector_store(self, vector_store) -> None:
        """Update vector store reference (may change after lazy init or rollback)."""
        self._vector_store = vector_store

    def _check_health(self) -> None:
        """Lightweight health check - sets _needs_optimize if pressure detected."""
        try:
            from graphstore.core.optimizer import health_check, needs_optimization
            health = health_check(self._store, self._vector_store, self._document_store)
            if needs_optimization(health,
                                  compact_threshold=self._compact_threshold,
                                  string_gc_threshold=self._string_gc_threshold,
                                  cache_gc_threshold=self._cache_gc_threshold):
                self._needs_optimize = True
            # Emergency eviction if memory > 90% ceiling
            from graphstore.core.memory import check_ceiling_accurate
            if check_ceiling_accurate(self._store, self._vector_store, self._store._ceiling_bytes):
                from graphstore.core.optimizer import evict_oldest
                target = int(self._store._ceiling_bytes * 0.8)
                evict_oldest(self._store, target, self._vector_store, self._document_store)
        except Exception as e:
            logger.debug("health check failed: %s", e)

        # Evolution tick: evaluate rules if engine present and not re-entrant
        engine = self._evolution_engine
        if engine is not None and not engine._evaluating:
            try:
                signals = engine.compute_signals()
                engine.evaluate(signals)
            except Exception as e:
                logger.warning("evolution tick failed: %s", e)
