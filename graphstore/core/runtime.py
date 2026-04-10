"""RuntimeState: single mutable container for shared component refs.

GraphStore owns one RuntimeState. All components that need references
to store / schema / vector_store / document_store / embedder / conn
take the same RuntimeState instance in their constructor and read
those fields as properties. reset_memory() and lazy vector-store init
mutate the container directly; every component sees the new ref
through its shared reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3

    from graphstore.core.store import CoreStore
    from graphstore.core.schema import SchemaRegistry
    from graphstore.vector.store import VectorStore
    from graphstore.document.store import DocumentStore
    from graphstore.embedding.base import Embedder


@dataclass
class RuntimeState:
    store: "CoreStore"
    schema: "SchemaRegistry"
    vector_store: "VectorStore | None" = None
    document_store: "DocumentStore | None" = None
    embedder: "Embedder | None" = None
    conn: "sqlite3.Connection | None" = None
