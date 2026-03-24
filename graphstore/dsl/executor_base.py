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

from graphstore.core.schema import SchemaRegistry
from graphstore.core.store import CoreStore
from graphstore.core.types import Result
from graphstore.dsl.visibility import VisibilityMixin
from graphstore.dsl.filtering import FilteringMixin


class ExecutorBase(VisibilityMixin, FilteringMixin):
    def __init__(self, store: CoreStore, schema: SchemaRegistry | None = None,
                 embedder=None, vector_store=None, document_store=None,
                 ingest_root: str | None = None):
        self.store = store
        self.schema = schema or SchemaRegistry()
        self._embedder = embedder
        self._vector_store = vector_store
        self._document_store = document_store
        self._ingest_root = ingest_root
        self.cost_threshold = 100_000
        self._embedder_dirty = False
        self._fts_full_text = True  # default, overridden by config

    def execute(self, ast) -> Result:
        """Execute a parsed AST node and return a Result."""
        start = time.perf_counter_ns()
        result = self._dispatch(ast)
        elapsed = (time.perf_counter_ns() - start) // 1000
        result.elapsed_us = elapsed
        return result
