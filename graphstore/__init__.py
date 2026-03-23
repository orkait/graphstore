"""graphstore - Agentic brain DB with semantic search."""

__version__ = "0.2.0"

from graphstore.graphstore import GraphStore
from graphstore.core.store import CoreStore
from graphstore.core.schema import SchemaRegistry
from graphstore.core.types import Result, Edge
from graphstore.dsl.parser import parse, clear_cache
from graphstore.dsl.executor import Executor
from graphstore.dsl.executor_system import SystemExecutor
from graphstore.core.errors import (
    GraphStoreError, QueryError, NodeNotFound, NodeExists,
    CeilingExceeded, VersionMismatch, SchemaError,
    CostThresholdExceeded, BatchRollback, AggregationError,
    VectorError, EmbedderRequired, VectorNotFound,
)
from graphstore.core.memory import DEFAULT_CEILING_BYTES
from graphstore.config import (
    GraphStoreConfig, load_config, save_config,
    CoreConfig, VectorConfig, DocumentConfig, DslConfig,
    VaultConfig, PersistenceConfig, RetentionConfig, ServerConfig,
)

__all__ = [
    "GraphStore", "CoreStore", "SchemaRegistry",
    "Result", "Edge",
    "parse", "clear_cache", "Executor", "SystemExecutor",
    "GraphStoreError", "QueryError", "NodeNotFound", "NodeExists",
    "CeilingExceeded", "VersionMismatch", "SchemaError",
    "CostThresholdExceeded", "BatchRollback", "AggregationError",
    "VectorError", "EmbedderRequired", "VectorNotFound",
    "DEFAULT_CEILING_BYTES",
    "GraphStoreConfig", "load_config", "save_config",
    "CoreConfig", "VectorConfig", "DocumentConfig", "DslConfig",
    "VaultConfig", "PersistenceConfig", "RetentionConfig", "ServerConfig",
]
