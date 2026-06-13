from graphstore.config import GraphStoreConfig, IngestConfig, apply_env_overrides


def test_ingest_defaults():
    cfg = GraphStoreConfig()
    assert cfg.ingest.nl_backend is None
    assert cfg.ingest.free_first is True
    assert cfg.ingest.nl_max_tokens == 1000


def test_ingest_env_override(monkeypatch):
    monkeypatch.setenv("GRAPHSTORE_INGEST_NL_BACKEND", "cloud")
    monkeypatch.setenv("GRAPHSTORE_INGEST_NL_MAX_TOKENS", "2000")
    cfg = apply_env_overrides(GraphStoreConfig())
    assert cfg.ingest.nl_backend == "cloud"
    assert cfg.ingest.nl_max_tokens == 2000


def test_ingest_config_is_frozen():
    import msgspec
    c = IngestConfig()
    assert isinstance(c, msgspec.Struct)
