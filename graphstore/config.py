"""Typed configuration for graphstore via msgspec Structs.

One file, sectioned by engine. Missing keys auto-fill from defaults.
Load order: defaults -> graphstore.json -> constructor kwargs.
"""

from __future__ import annotations

import json
from pathlib import Path

import msgspec


class CoreConfig(msgspec.Struct, frozen=True):
    ceiling_mb: int = 256
    initial_capacity: int = 1024
    compact_threshold: float = 0.2
    string_gc_threshold: float = 3.0
    eviction_target_ratio: float = 0.8
    protected_kinds: list[str] = msgspec.field(default_factory=lambda: ["schema", "config", "system"])


class VectorConfig(msgspec.Struct, frozen=True):
    embedder: str = "default"
    similarity_threshold: float = 0.85
    duplicate_threshold: float = 0.95
    search_oversample: int = 5
    model2vec_model: str = "minishlab/M2V_base_output"
    model_cache_dir: str | None = None


class DocumentConfig(msgspec.Struct, frozen=True):
    fts_tokenizer: str = "porter unicode61"
    chunk_max_size: int = 2000
    chunk_overlap: int = 50
    summary_max_length: int = 200
    fts_full_text: bool = True
    vision_model: str = "smolvlm2:2.2b"
    vision_base_url: str = "http://localhost:11434/v1"
    vision_max_tokens: int = 300


class DslConfig(msgspec.Struct, frozen=True):
    cost_threshold: int = 100_000
    plan_cache_size: int = 256
    auto_optimize: bool = False
    optimize_interval: int = 500
    recall_decay: float = 0.7
    remember_weights: list[float] = msgspec.field(default_factory=lambda: [0.30, 0.20, 0.15, 0.20, 0.15])
    cache_gc_threshold: int = 200


class VaultConfig(msgspec.Struct, frozen=True):
    enabled: bool = False
    path: str | None = None
    auto_sync: bool = True


class PersistenceConfig(msgspec.Struct, frozen=True):
    wal_hard_limit: int = 100_000
    auto_checkpoint_threshold: int = 50_000
    log_retention_days: int = 7
    busy_timeout_ms: int = 5000


class RetentionConfig(msgspec.Struct, frozen=True):
    blob_warm_days: int = 30
    blob_archive_days: int = 90
    blob_delete_days: int = 365


class ServerConfig(msgspec.Struct, frozen=True):
    cors_origins: list[str] = msgspec.field(default_factory=lambda: ["*"])
    ingest_root: str | None = None
    auth_token: str | None = None
    rate_limit_rpm: int = 120
    rate_limit_window: int = 60
    max_query_length: int = 10_000
    max_batch_size: int = 1000


class GraphStoreConfig(msgspec.Struct, frozen=True):
    core: CoreConfig = msgspec.field(default_factory=CoreConfig)
    vector: VectorConfig = msgspec.field(default_factory=VectorConfig)
    document: DocumentConfig = msgspec.field(default_factory=DocumentConfig)
    dsl: DslConfig = msgspec.field(default_factory=DslConfig)
    vault: VaultConfig = msgspec.field(default_factory=VaultConfig)
    persistence: PersistenceConfig = msgspec.field(default_factory=PersistenceConfig)
    retention: RetentionConfig = msgspec.field(default_factory=RetentionConfig)
    server: ServerConfig = msgspec.field(default_factory=ServerConfig)


_decoder = msgspec.json.Decoder(GraphStoreConfig)
_encoder = msgspec.json.Encoder()


def load_config(path: str | Path | None = None) -> GraphStoreConfig:
    """Load config from a JSON file. Returns defaults if file doesn't exist."""
    if path is None:
        return GraphStoreConfig()
    p = Path(path)
    if not p.exists():
        return GraphStoreConfig()
    raw = p.read_bytes()
    return _decoder.decode(raw)


def save_config(config: GraphStoreConfig, path: str | Path) -> None:
    """Save config to a JSON file (pretty-printed)."""
    p = Path(path)
    data = msgspec.json.decode(msgspec.json.encode(config))
    p.write_text(json.dumps(data, indent=2) + "\n")


def merge_kwargs(config: GraphStoreConfig, **kwargs) -> GraphStoreConfig:
    """Override config fields from constructor kwargs for backwards compat.

    Supports: ceiling_mb, embedder, ingest_root, vault, retention (dict).
    """
    updates: dict = {}

    if "ceiling_mb" in kwargs or "eviction_target_ratio" in kwargs:
        updates["core"] = CoreConfig(
            ceiling_mb=kwargs.get("ceiling_mb", config.core.ceiling_mb),
            initial_capacity=config.core.initial_capacity,
            compact_threshold=config.core.compact_threshold,
            string_gc_threshold=config.core.string_gc_threshold,
            eviction_target_ratio=kwargs.get("eviction_target_ratio", config.core.eviction_target_ratio),
            protected_kinds=config.core.protected_kinds,
        )

    if "embedder" in kwargs:
        emb = kwargs["embedder"]
        emb_name = emb if isinstance(emb, str) else "custom"
        if emb is None:
            emb_name = "none"
        if emb_name != config.vector.embedder:
            updates["vector"] = VectorConfig(
                embedder=emb_name,
                similarity_threshold=config.vector.similarity_threshold,
                duplicate_threshold=config.vector.duplicate_threshold,
                search_oversample=config.vector.search_oversample,
                model2vec_model=config.vector.model2vec_model,
                model_cache_dir=config.vector.model_cache_dir,
            )

    if "ingest_root" in kwargs and kwargs["ingest_root"] is not None:
        updates["server"] = ServerConfig(
            cors_origins=config.server.cors_origins,
            ingest_root=kwargs["ingest_root"],
            auth_token=config.server.auth_token,
            rate_limit_rpm=config.server.rate_limit_rpm,
            rate_limit_window=config.server.rate_limit_window,
            max_query_length=config.server.max_query_length,
            max_batch_size=config.server.max_batch_size,
        )

    if "vault" in kwargs and kwargs["vault"] is not None:
        updates["vault"] = VaultConfig(
            enabled=True,
            path=kwargs["vault"],
            auto_sync=config.vault.auto_sync,
        )

    if "retention" in kwargs and kwargs["retention"] is not None:
        r = kwargs["retention"]
        updates["retention"] = RetentionConfig(
            blob_warm_days=r.get("blob_warm_days", config.retention.blob_warm_days),
            blob_archive_days=r.get("blob_archive_days", config.retention.blob_archive_days),
            blob_delete_days=r.get("blob_delete_days", config.retention.blob_delete_days),
        )

    if not updates:
        return config

    return GraphStoreConfig(
        core=updates.get("core", config.core),
        vector=updates.get("vector", config.vector),
        document=updates.get("document", config.document),
        dsl=updates.get("dsl", config.dsl),
        vault=updates.get("vault", config.vault),
        persistence=updates.get("persistence", config.persistence),
        retention=updates.get("retention", config.retention),
        server=updates.get("server", config.server),
    )
