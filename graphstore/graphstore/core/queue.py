"""Single-writer command queue for thread-safe GraphStore access.

All execute() calls are serialized through a PriorityQueue drained by
a dedicated daemon worker thread. Two priority levels ensure interactive
queries complete before background refinement jobs.
"""

from __future__ import annotations

import logging
import threading
import queue
from concurrent.futures import Future
from typing import Callable, Any

logger = logging.getLogger(__name__)

INTERACTIVE = 0
BACKGROUND = 1

_SHUTDOWN = object()


class CommandQueue:
    """Priority queue with dedicated worker thread for serialized execution."""

    def __init__(self, execute_fn: Callable[[str], Any]):
        self._execute_fn = execute_fn
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, daemon=True, name="graphstore-worker")
        self._running = True
        self._worker.start()

    def _next_seq(self) -> int:
        with self._seq_lock:
            seq = self._seq
            self._seq += 1
            return seq

    def submit(self, query: str) -> Any:
        """Submit interactive query, block until result."""
        if not self._running:
            raise RuntimeError("CommandQueue is shut down")
        future: Future = Future()
        self._queue.put((INTERACTIVE, self._next_seq(), query, future))
        return future.result()

    def submit_background(self, query: str) -> Future:
        """Submit background query, return Future immediately.

        Failed background jobs are logged at WARNING level even if
        the caller never calls .result() on the returned Future.
        """
        if not self._running:
            raise RuntimeError("CommandQueue is shut down")
        future: Future = Future()
        future.add_done_callback(lambda f: self._on_background_done(f, query))
        self._queue.put((BACKGROUND, self._next_seq(), query, future))
        return future

    @staticmethod
    def _on_background_done(future: Future, query: str) -> None:
        """Log failed background jobs."""
        exc = future.exception()
        if exc is not None:
            logger.warning("background job failed: %s - %s: %s", query, type(exc).__name__, exc)

    def shutdown(self) -> None:
        """Stop the worker thread. Idempotent."""
        if not self._running:
            return
        self._running = False
        self._queue.put((999, 0, _SHUTDOWN, None))
        self._worker.join(timeout=5)

    def _run(self) -> None:
        """Worker loop: drain queue, execute, set results."""
        while True:
            item = self._queue.get()
            priority, seq, query, future = item
            if query is _SHUTDOWN:
                break
            try:
                result = self._execute_fn(query)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

    @property
    def pending(self) -> int:
        """Approximate number of pending items."""
        return self._queue.qsize()
