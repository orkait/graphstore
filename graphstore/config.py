"""Typed configuration for graphstore via msgspec Structs.

One file, sectioned by engine. Missing keys auto-fill from defaults.

Load order (each layer overrides the previous):
    1. config.py dataclass defaults (source of truth for types + default values)
    2. graphstore.json file (per-deployment overrides)
    3. GRAPHSTORE_* environment variables (Docker/k8s friendly)
    4. Constructor kwargs (code-level, highest priority)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import msgspec


_log = logging.getLogger(__name__)


class CoreConfig(msgspec.Struct, frozen=True):
    ceiling_mb: int = 256
    initial_capacity: int = 1024
    compact_threshold: float = 0.2
    string_gc_threshold: float = 3.0
    eviction_target_ratio: float = 0.8
    protected_kinds: list[str] = msgspec.field(default_factory=lambda: ["schema", "config", "system"])


class VectorConfig(msgspec.Struct, frozen=True):
    embedder: str = "default"
    embedder_model: str | None = None
    embedder_dims: int | None = None
    gpu_layers: int = 0
    similarity_threshold: float = 0.85
    duplicate_threshold: float = 0.95
    search_oversample: int = 16
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
    recall_decay: float = 0.5912428069710964
    remember_weights: list[float] = msgspec.field(default_factory=lambda: [0.50, 0.20, 0.10, 0.15, 0.05])
    fusion_method: str = "weighted"  # "rrf" or "weighted"
    rrf_k: float = 60.0
    retrieval_strategy: str = "full"
    retrieval_depth: int = 9
    recall_depth: int = 2
    max_query_entities: int = 6
    recency_mode: str = "multiplicative"  # "additive" or "multiplicative"
    recency_boost_k: int = 4
    recency_half_life_days: float = 7300.0  # ~20 years
    similar_to_oversample: int = 2
    lexical_search_oversample: int = 3
    hybridrag_weight: float = 0.15
    hybridrag_min_seeds: int = 5
    type_weights: dict = msgspec.field(default_factory=lambda: {
        "observation": 1.8, "fact": 1.3, "event": 1.2, "preference": 1.3,
        "claim": 1.2, "decision": 1.5, "lesson": 1.5,
        "memory": 1.0, "entity": 0.8, "session": 0.7,
    })
    temporal_weight: float = 0.15
    temporal_decay_days: float = 365.0
    nucleus_expansion: bool = True  # benchmark-tuned default; may append neighbors after top-k
    nucleus_hops: int = 2
    nucleus_max_neighbors: int = 3
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


class EvolutionConfig(msgspec.Struct, frozen=True):
    similarity_buffer_size: int = 100
    max_rules: int = 50
    min_cooldown: int = 10
    history_retention: int = 1000


class GraphStoreConfig(msgspec.Struct, frozen=True):
    core: CoreConfig = msgspec.field(default_factory=CoreConfig)
    vector: VectorConfig = msgspec.field(default_factory=VectorConfig)
    document: DocumentConfig = msgspec.field(default_factory=DocumentConfig)
    dsl: DslConfig = msgspec.field(default_factory=DslConfig)
    vault: VaultConfig = msgspec.field(default_factory=VaultConfig)
    persistence: PersistenceConfig = msgspec.field(default_factory=PersistenceConfig)
    retention: RetentionConfig = msgspec.field(default_factory=RetentionConfig)
    server: ServerConfig = msgspec.field(default_factory=ServerConfig)
    evolution: EvolutionConfig = msgspec.field(default_factory=EvolutionConfig)


_decoder = msgspec.json.Decoder(GraphStoreConfig)
_encoder = msgspec.json.Encoder()

# Section name -> (config class, field name -> type)
_SECTION_MAP: dict[str, tuple[type, dict[str, type]]] = {
    "core": (CoreConfig, {f: type(getattr(CoreConfig(), f)) for f in CoreConfig.__struct_fields__}),
    "vector": (VectorConfig, {f: type(getattr(VectorConfig(), f)) for f in VectorConfig.__struct_fields__}),
    "document": (DocumentConfig, {f: type(getattr(DocumentConfig(), f)) for f in DocumentConfig.__struct_fields__}),
    "dsl": (DslConfig, {f: type(getattr(DslConfig(), f)) for f in DslConfig.__struct_fields__}),
    "vault": (VaultConfig, {f: type(getattr(VaultConfig(), f)) for f in VaultConfig.__struct_fields__}),
    "persistence": (PersistenceConfig, {f: type(getattr(PersistenceConfig(), f)) for f in PersistenceConfig.__struct_fields__}),
    "retention": (RetentionConfig, {f: type(getattr(RetentionConfig(), f)) for f in RetentionConfig.__struct_fields__}),
    "server": (ServerConfig, {f: type(getattr(ServerConfig(), f)) for f in ServerConfig.__struct_fields__}),
    "evolution": (EvolutionConfig, {f: type(getattr(EvolutionConfig(), f)) for f in EvolutionConfig.__struct_fields__}),
}

# Flat kwarg name -> (section, field) for constructor shortcuts
_KWARG_SHORTCUTS: dict[str, tuple[str, str]] = {
    "ceiling_mb":           ("core", "ceiling_mb"),
    "eviction_target_ratio":("core", "eviction_target_ratio"),
    "remember_weights":     ("dsl", "remember_weights"),
    "fusion_method":        ("dsl", "fusion_method"),
    "rrf_k":                ("dsl", "rrf_k"),
    "recall_decay":         ("dsl", "recall_decay"),
    "retrieval_strategy":   ("dsl", "retrieval_strategy"),
    "retrieval_depth":      ("dsl", "retrieval_depth"),
    "recall_depth":         ("dsl", "recall_depth"),
    "max_query_entities":   ("dsl", "max_query_entities"),
    "recency_mode":         ("dsl", "recency_mode"),
    "recency_boost_k":      ("dsl", "recency_boost_k"),
    "recency_half_life_days": ("dsl", "recency_half_life_days"),
    "similar_to_oversample": ("dsl", "similar_to_oversample"),
    "lexical_search_oversample": ("dsl", "lexical_search_oversample"),
    "hybridrag_weight":     ("dsl", "hybridrag_weight"),
    "hybridrag_min_seeds":  ("dsl", "hybridrag_min_seeds"),
    "type_weights":         ("dsl", "type_weights"),
    "temporal_weight":      ("dsl", "temporal_weight"),
    "temporal_decay_days":  ("dsl", "temporal_decay_days"),
    "nucleus_expansion":    ("dsl", "nucleus_expansion"),
    "nucleus_hops":         ("dsl", "nucleus_hops"),
    "nucleus_max_neighbors":("dsl", "nucleus_max_neighbors"),
    "search_oversample":    ("vector", "search_oversample"),
    "similarity_threshold": ("vector", "similarity_threshold"),
    "duplicate_threshold":  ("vector", "duplicate_threshold"),
    "fts_tokenizer":        ("document", "fts_tokenizer"),
}


def _coerce(value: str, target_type: type):
    """Coerce a string env var value to the target type."""
    if target_type is bool or target_type is type(True):
        return value.lower() in ("1", "true", "yes")
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    if target_type is list:
        return [v.strip() for v in value.split(",") if v.strip()]
    if target_type is type(None):
        return value if value else None
    return value


def load_config(path: str | Path | None = None) -> GraphStoreConfig:
    """Load config overrides from a JSON file and merge onto defaults.

    The file is expected to be a partial dict - only sections/fields the user
    changed. Missing sections and fields fall back to config.py defaults.
    Returns full defaults if the file doesn't exist or is empty.
    """
    if path is None:
        return GraphStoreConfig()
    p = Path(path)
    if not p.exists():
        return GraphStoreConfig()
    raw = p.read_bytes().strip()
    if not raw:
        return GraphStoreConfig()
    try:
        overrides = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as e:
        _log.warning("config parse error in %s: %s - using defaults", p, e)
        return GraphStoreConfig()
    if not isinstance(overrides, dict) or not overrides:
        return GraphStoreConfig()
    return _rebuild_config(GraphStoreConfig(), overrides)


def save_config(config: GraphStoreConfig, path: str | Path) -> None:
    """Save config overrides to a JSON file.

    Only writes values that differ from defaults. This keeps graphstore.json
    minimal and ensures users benefit from future default improvements.
    """
    defaults = GraphStoreConfig()
    current = msgspec.json.decode(msgspec.json.encode(config))
    default_data = msgspec.json.decode(msgspec.json.encode(defaults))

    diff: dict = {}
    for section, fields in current.items():
        if not isinstance(fields, dict):
            continue
        section_diff = {}
        for field, value in fields.items():
            default_val = default_data.get(section, {}).get(field)
            if value != default_val:
                section_diff[field] = value
        if section_diff:
            diff[section] = section_diff

    p = Path(path)
    if diff:
        p.write_text(json.dumps(diff, indent=2) + "\n")
    elif p.exists():
        p.unlink()


def apply_env_overrides(config: GraphStoreConfig) -> GraphStoreConfig:
    """Override config fields from GRAPHSTORE_* environment variables.

    Convention: GRAPHSTORE_{SECTION}_{FIELD} in uppercase.
    Examples:
        GRAPHSTORE_CORE_CEILING_MB=512
        GRAPHSTORE_DSL_RECALL_DECAY=0.5
        GRAPHSTORE_DSL_REMEMBER_WEIGHTS=0.40,0.30,0.15,0.10,0.05
        GRAPHSTORE_VECTOR_SEARCH_OVERSAMPLE=10
        GRAPHSTORE_VECTOR_SIMILARITY_THRESHOLD=0.80
        GRAPHSTORE_SERVER_CORS_ORIGINS=http://localhost:3000,http://localhost:8080
        GRAPHSTORE_SERVER_AUTH_TOKEN=secret123
    """
    updates: dict[str, dict[str, object]] = {}

    for env_key, env_val in os.environ.items():
        if not env_key.startswith("GRAPHSTORE_") or not env_val:
            continue

        parts = env_key[len("GRAPHSTORE_"):].lower().split("_", 1)
        if len(parts) != 2:
            continue
        section, field = parts[0], parts[1]

        if section not in _SECTION_MAP:
            continue
        _, field_types = _SECTION_MAP[section]
        if field not in field_types:
            continue

        target_type = field_types[field]
        try:
            if field == "remember_weights":
                parsed = [float(w) for w in env_val.split(",")]
                if len(parsed) != 5:
                    _log.warning("GRAPHSTORE_DSL_REMEMBER_WEIGHTS needs 5 values, got %d", len(parsed))
                    continue
                updates.setdefault(section, {})[field] = parsed
            elif field == "protected_kinds" or field == "cors_origins":
                updates.setdefault(section, {})[field] = [v.strip() for v in env_val.split(",")]
            elif field == "type_weights":
                # Format: "fact:1.3,decision:1.5,entity:0.8"
                try:
                    parsed = {}
                    for pair in env_val.split(","):
                        k, v = pair.strip().split(":")
                        parsed[k.strip()] = float(v.strip())
                    updates.setdefault(section, {})[field] = parsed
                except (ValueError, IndexError):
                    _log.warning("GRAPHSTORE_DSL_TYPE_WEIGHTS format: 'kind:weight,kind:weight'")
                    continue
            else:
                updates.setdefault(section, {})[field] = _coerce(env_val, target_type)
        except (ValueError, TypeError) as e:
            _log.warning("invalid env var %s=%s: %s", env_key, env_val, e)

    if not updates:
        return config

    return _rebuild_config(config, updates)


def merge_kwargs(config: GraphStoreConfig, **kwargs) -> GraphStoreConfig:
    """Override config fields from constructor kwargs.

    Supports flat shortcuts for common tuning knobs:
        ceiling_mb, eviction_target_ratio, remember_weights, recall_decay,
        search_oversample, similarity_threshold, duplicate_threshold, fts_tokenizer

    Plus legacy kwargs: embedder, ingest_root, vault, retention (dict).
    """
    updates: dict[str, dict[str, object]] = {}

    # Flat shortcuts -> section overrides
    for kwarg_name, (section, field) in _KWARG_SHORTCUTS.items():
        if kwarg_name in kwargs:
            val = kwargs[kwarg_name]
            current_val = getattr(getattr(config, section), field)
            if val != current_val:
                updates.setdefault(section, {})[field] = val

    # Legacy: embedder (string or object -> vector.embedder name)
    if "embedder" in kwargs:
        emb = kwargs["embedder"]
        emb_name = emb if isinstance(emb, str) else "custom"
        if emb is None:
            emb_name = "none"
        updates.setdefault("vector", {})["embedder"] = emb_name

    # Legacy: ingest_root -> server.ingest_root
    if "ingest_root" in kwargs and kwargs["ingest_root"] is not None:
        updates.setdefault("server", {})["ingest_root"] = kwargs["ingest_root"]

    # Legacy: vault -> vault.enabled + vault.path
    if "vault" in kwargs and kwargs["vault"] is not None:
        updates.setdefault("vault", {})["enabled"] = True
        updates["vault"]["path"] = kwargs["vault"]

    # Legacy: retention (dict)
    if "retention" in kwargs and kwargs["retention"] is not None:
        r = kwargs["retention"]
        for key in ("blob_warm_days", "blob_archive_days", "blob_delete_days"):
            if key in r:
                updates.setdefault("retention", {})[key] = r[key]

    if not updates:
        return config

    return _rebuild_config(config, updates)


def _rebuild_config(
    config: GraphStoreConfig,
    updates: dict[str, dict[str, object]],
) -> GraphStoreConfig:
    """Rebuild a frozen config with section-level field overrides."""
    sections = {}
    for section_name in GraphStoreConfig.__struct_fields__:
        current = getattr(config, section_name)
        if section_name not in updates:
            sections[section_name] = current
            continue
        field_overrides = updates[section_name]
        current_dict = {f: getattr(current, f) for f in current.__struct_fields__}
        current_dict.update(field_overrides)
        section_cls = type(current)
        sections[section_name] = section_cls(**current_dict)

    return GraphStoreConfig(**sections)
