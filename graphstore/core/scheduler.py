"""Optimizer scheduling: health checks and auto-optimize between execute calls."""

import logging

logger = logging.getLogger(__name__)


class OptimizerScheduler:
    """Tracks write pressure and triggers optimization at safe points."""

    def __init__(self, store, vector_store, document_store,
                 auto_optimize: bool = False, optimize_interval: int = 500):
        self._store = store
        self._vector_store = vector_store
        self._document_store = document_store
        self._auto_optimize = auto_optimize
        self._optimize_interval = optimize_interval
        self._optimizing = False
        self._needs_optimize = False
        self._write_counter = 0

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
            optimize_all(self._store, self._vector_store, self._document_store)
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
            if needs_optimization(health):
                self._needs_optimize = True
        except Exception as e:
            logger.debug("health check failed: %s", e)
