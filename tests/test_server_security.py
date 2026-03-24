"""Test server authentication and rate limiting."""
import os
import pytest


@pytest.fixture
def auth_client():
    """Create a test client with auth enabled."""
    os.environ["GRAPHSTORE_AUTH_TOKEN"] = "test-secret-token"
    os.environ.pop("GRAPHSTORE_DB_PATH", None)

    import importlib
    import graphstore.server as srv
    srv._store = None
    importlib.reload(srv)

    from fastapi.testclient import TestClient
    client = TestClient(srv.app)
    yield client

    os.environ.pop("GRAPHSTORE_AUTH_TOKEN", None)
    srv._store = None


@pytest.fixture
def open_client():
    """Create a test client without auth."""
    os.environ.pop("GRAPHSTORE_AUTH_TOKEN", None)
    os.environ.pop("GRAPHSTORE_DB_PATH", None)

    import importlib
    import graphstore.server as srv
    srv._store = None
    importlib.reload(srv)

    from fastapi.testclient import TestClient
    client = TestClient(srv.app)
    yield client
    srv._store = None


def test_no_auth_allows_access(open_client):
    """Without auth configured, all requests pass."""
    resp = open_client.post("/api/execute", json={"query": "SYS STATS"})
    assert resp.status_code == 200


def test_auth_rejects_missing_token(auth_client):
    """With auth configured, missing Authorization header is 401."""
    resp = auth_client.post("/api/execute", json={"query": "SYS STATS"})
    assert resp.status_code == 401


def test_auth_rejects_wrong_token(auth_client):
    """With auth configured, wrong token is 401."""
    resp = auth_client.post(
        "/api/execute",
        json={"query": "SYS STATS"},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp.status_code == 401


def test_auth_accepts_correct_token(auth_client):
    """With auth configured, correct token passes."""
    resp = auth_client.post(
        "/api/execute",
        json={"query": "SYS STATS"},
        headers={"Authorization": "Bearer test-secret-token"},
    )
    assert resp.status_code == 200


def test_rate_limit_returns_429(open_client):
    """Exceeding rate limit returns 429."""
    import graphstore.server as srv
    original = srv._RATE_LIMIT_RPM
    srv._RATE_LIMIT_RPM = 3
    srv._rate_buckets.clear()
    try:
        for i in range(3):
            resp = open_client.post("/api/execute", json={"query": "SYS STATS"})
            assert resp.status_code == 200
        resp = open_client.post("/api/execute", json={"query": "SYS STATS"})
        assert resp.status_code == 429
        assert "Rate limit" in resp.json()["error"]
    finally:
        srv._RATE_LIMIT_RPM = original
        srv._rate_buckets.clear()
