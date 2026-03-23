"""Tests for the typed config system."""

import json
import msgspec
import pytest

from graphstore.config import (
    GraphStoreConfig, CoreConfig, VectorConfig, RetentionConfig,
    PersistenceConfig, DslConfig, VaultConfig, ServerConfig,
    load_config, save_config, merge_kwargs,
)
from graphstore import GraphStore


class TestConfigDefaults:
    def test_all_defaults(self):
        cfg = GraphStoreConfig()
        assert cfg.core.ceiling_mb == 256
        assert cfg.core.initial_capacity == 1024
        assert cfg.vector.embedder == "default"
        assert cfg.vector.similarity_threshold == 0.85
        assert cfg.dsl.cost_threshold == 100_000
        assert cfg.dsl.plan_cache_size == 256
        assert cfg.persistence.wal_hard_limit == 100_000
        assert cfg.persistence.auto_checkpoint_threshold == 50_000
        assert cfg.persistence.log_retention_days == 7
        assert cfg.retention.blob_warm_days == 30
        assert cfg.retention.blob_archive_days == 90
        assert cfg.retention.blob_delete_days == 365
        assert cfg.vault.enabled is False
        assert cfg.server.cors_origins == ["*"]

    def test_frozen(self):
        cfg = GraphStoreConfig()
        with pytest.raises(AttributeError):
            cfg.core = CoreConfig(ceiling_mb=512)


class TestConfigSerialize:
    def test_json_roundtrip(self):
        cfg = GraphStoreConfig()
        raw = msgspec.json.encode(cfg)
        cfg2 = msgspec.json.decode(raw, type=GraphStoreConfig)
        assert cfg == cfg2

    def test_partial_json_fills_defaults(self):
        raw = b'{"core": {"ceiling_mb": 512}}'
        cfg = msgspec.json.decode(raw, type=GraphStoreConfig)
        assert cfg.core.ceiling_mb == 512
        assert cfg.core.initial_capacity == 1024
        assert cfg.persistence.wal_hard_limit == 100_000

    def test_empty_json(self):
        cfg = msgspec.json.decode(b'{}', type=GraphStoreConfig)
        assert cfg == GraphStoreConfig()


class TestConfigFileIO:
    def test_save_and_load(self, tmp_path):
        cfg = GraphStoreConfig(core=CoreConfig(ceiling_mb=512))
        path = tmp_path / "graphstore.json"
        save_config(cfg, path)
        loaded = load_config(path)
        assert loaded.core.ceiling_mb == 512

    def test_load_missing_file(self, tmp_path):
        cfg = load_config(tmp_path / "nonexistent.json")
        assert cfg == GraphStoreConfig()

    def test_saved_file_is_pretty_json(self, tmp_path):
        cfg = GraphStoreConfig()
        path = tmp_path / "graphstore.json"
        save_config(cfg, path)
        content = path.read_text()
        parsed = json.loads(content)
        assert "core" in parsed
        assert parsed["core"]["ceiling_mb"] == 256


class TestMergeKwargs:
    def test_ceiling_override(self):
        cfg = merge_kwargs(GraphStoreConfig(), ceiling_mb=512)
        assert cfg.core.ceiling_mb == 512
        assert cfg.core.initial_capacity == 1024

    def test_embedder_none(self):
        cfg = merge_kwargs(GraphStoreConfig(), embedder=None)
        assert cfg.vector.embedder == "none"

    def test_vault_override(self):
        cfg = merge_kwargs(GraphStoreConfig(), vault="./notes")
        assert cfg.vault.enabled is True
        assert cfg.vault.path == "./notes"

    def test_retention_override(self):
        cfg = merge_kwargs(GraphStoreConfig(), retention={"blob_warm_days": 7})
        assert cfg.retention.blob_warm_days == 7
        assert cfg.retention.blob_archive_days == 90  # default preserved

    def test_no_change_returns_same(self):
        base = GraphStoreConfig()
        cfg = merge_kwargs(base, ceiling_mb=256)  # same as default
        assert cfg is base


class TestGraphStoreConfig:
    def test_graphstore_uses_config(self, tmp_path):
        gs = GraphStore(path=str(tmp_path / "db"), embedder=None, ceiling_mb=128)
        assert gs._config.core.ceiling_mb == 128
        assert gs._ceiling_bytes == 128_000_000
        gs.close()

    def test_graphstore_loads_config_file(self, tmp_path):
        db_path = tmp_path / "db"
        db_path.mkdir()
        cfg_path = db_path / "graphstore.json"
        cfg_path.write_text('{"core": {"ceiling_mb": 64}}')
        gs = GraphStore(path=str(db_path), embedder=None)
        assert gs._config.core.ceiling_mb == 64
        assert gs._ceiling_bytes == 64_000_000
        gs.close()

    def test_kwargs_override_config_file(self, tmp_path):
        db_path = tmp_path / "db"
        db_path.mkdir()
        cfg_path = db_path / "graphstore.json"
        cfg_path.write_text('{"core": {"ceiling_mb": 64}}')
        gs = GraphStore(path=str(db_path), embedder=None, ceiling_mb=512)
        assert gs._config.core.ceiling_mb == 512
        gs.close()

    def test_explicit_config_object(self, tmp_path):
        cfg = GraphStoreConfig(core=CoreConfig(ceiling_mb=32))
        gs = GraphStore(path=str(tmp_path / "db"), embedder=None, config=cfg)
        assert gs._ceiling_bytes == 32_000_000
        gs.close()
