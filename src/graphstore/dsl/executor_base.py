"""ExecutorBase: wires store/schema/embedder into the handler mixin chain."""

import re
import time
from functools import lru_cache


@lru_cache(maxsize=256)
def _compile_like_pattern(pattern: str):
    parts = []
    for ch in pattern:
        if ch == '%':
            parts.append('.*')
        elif ch == '_':
            parts.append('.')
        else:
            parts.append(re.escape(ch))
    return re.compile(''.join(parts))

from graphstore.core.runtime import RuntimeState
from graphstore.core.types import Result
from graphstore.dsl.visibility import VisibilityMixin
from graphstore.dsl.filtering import FilteringMixin


class ExecutorBase(VisibilityMixin, FilteringMixin):
    def __init__(self, runtime: RuntimeState,
                 ingest_root: str | None = None,
                 ingestor_registry=None, chunker=None):
        self._runtime = runtime
        self._ingest_root = ingest_root
        self._ingestor_registry = ingestor_registry
        self._chunker = chunker
        self.cost_threshold = 100_000
        self._embedder_dirty = False
        self._fts_full_text = True
        self._vision_model = "smolvlm2:2.2b"
        self._vision_base_url = "http://localhost:11434/v1"
        self._vision_max_tokens = 300
        self._defer_embeddings: bool = False
        self._pending_embeddings: list[tuple[int, str]] = []
        self._embed_batch_size: int = 64
        self._similarity_threshold: float | None = None

    @property
    def store(self):
        return self._runtime.store

    @property
    def schema(self):
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

    def execute(self, ast) -> Result:
        """Execute a parsed AST node and return a Result."""
        start = time.perf_counter_ns()
        result = self._dispatch(ast)
        elapsed = (time.perf_counter_ns() - start) // 1000
        result.elapsed_us = elapsed
        return result
