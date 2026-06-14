"""graphstore owns multimodal understanding: media bytes -> text -> stored."""
import base64

import pytest

import graphstore.ingest.media as media


def test_image_builds_an_image_part():
    msgs = media._messages(b"\xff\xd8\xff", "image/jpeg", None)
    parts = msgs[0]["content"]
    assert any(p["type"] == "image_url" and p["image_url"]["url"].startswith("data:image/jpeg;base64,") for p in parts)


def test_audio_builds_an_input_audio_part():
    msgs = media._messages(b"ID3", "audio/mp3", None)
    parts = msgs[0]["content"]
    assert any(p["type"] == "input_audio" and p["input_audio"]["format"] == "mp3" for p in parts)


def test_unsupported_mime_raises():
    with pytest.raises(media.MediaUnsupported):
        media._messages(b"%PDF", "application/pdf", None)


def test_ingest_media_endpoint_understands_and_stores(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import graphstore.server as server

    # mock the cloud understanding so the test needs no provider/network
    monkeypatch.setattr(media, "understand_media",
                        lambda data, mime, **kw: "a red bicycle leaning on a brick wall")
    server._store = None  # fresh in-memory store
    with TestClient(server.app) as client:
        img_b64 = base64.b64encode(b"\xff\xd8\xff\xe0fake").decode()
        r = client.post("/api/ingest-media", json={
            "id": "m:1", "mime": "image/jpeg", "data_b64": img_b64, "namespace": "ns",
        })
        body = r.json()
        assert body["kind"] == "media"
        assert body["data"]["text"] == "a red bicycle leaning on a brick wall"
        # stored + recallable in the namespace
        rec = client.post("/api/execute", json={"query": 'REMEMBER "bicycle" LIMIT 3', "namespace": "ns"}).json()
        assert "m:1" in [row.get("id") for row in (rec.get("data") or [])]


def test_ingest_media_endpoint_rejects_bad_base64(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import graphstore.server as server
    server._store = None
    with TestClient(server.app) as client:
        r = client.post("/api/ingest-media", json={"id": "m:2", "mime": "image/jpeg", "data_b64": "!!!notb64"})
        assert r.json()["kind"] == "error"
