"""graphstore - Agentic brain DB with semantic search."""

__version__ = "0.3.0"

# Import compute_profile first so its module-level _apply_blas_env_cap()
# runs before numpy / scipy initialise their BLAS thread pools. Protects
# the host from a graphstore install accidentally saturating all cores.
from .core import compute_profile as _compute_profile_init  # noqa: F401

from .store import GraphStore
from .core.store import CoreStore
from .core.schema import SchemaRegistry
from .core.types import Result, Edge
from .dsl.parser import parse, clear_cache
from .dsl.executor import Executor
from .dsl.executor_system import SystemExecutor
from .core.errors import (
    GraphStoreError, QueryError, NodeNotFound, NodeExists,
    CeilingExceeded, VersionMismatch, SchemaError,
    CostThresholdExceeded, BatchRollback, AggregationError,
    VectorError, EmbedderRequired, VectorNotFound,
    OptimizationInProgress, StoreInUse,
)
from .core.memory import DEFAULT_CEILING_BYTES
from .config import (
    GraphStoreConfig, load_config, save_config,
    CoreConfig, VectorConfig, DocumentConfig, DslConfig,
    VaultConfig, PersistenceConfig, RetentionConfig, ServerConfig,
)
from .query import q, F, Query, register_verb

__all__ = [
    "GraphStore", "CoreStore", "SchemaRegistry",
    "Result", "Edge",
    "parse", "clear_cache", "Executor", "SystemExecutor",
    "GraphStoreError", "QueryError", "NodeNotFound", "NodeExists",
    "CeilingExceeded", "VersionMismatch", "SchemaError",
    "CostThresholdExceeded", "BatchRollback", "AggregationError",
    "VectorError", "EmbedderRequired", "VectorNotFound",
    "StoreInUse",
    "DEFAULT_CEILING_BYTES",
    "GraphStoreConfig", "load_config", "save_config",
    "CoreConfig", "VectorConfig", "DocumentConfig", "DslConfig",
    "VaultConfig", "PersistenceConfig", "RetentionConfig", "ServerConfig",
    "q", "F", "Query", "register_verb",
]
