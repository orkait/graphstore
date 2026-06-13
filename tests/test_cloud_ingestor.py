import types
import pytest
from graphstore.llm_runner import LLMRunner


def _fake_completion_response(content: str):
    msg = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=msg)
    return types.SimpleNamespace(choices=[choice])


def _fake_stream_chunks(deltas):
    for d in deltas:
        delta = types.SimpleNamespace(content=d)
        choice = types.SimpleNamespace(delta=delta)
        yield types.SimpleNamespace(choices=[choice])


PROVIDERS = [{"pid": "groq/x", "litellm_model": "groq/x",
              "api_base": None, "api_key": "k"}]


def test_complete_messages_returns_first_nonempty(monkeypatch):
    import litellm
    calls = {}

    def fake_completion(**kw):
        calls["kw"] = kw
        return _fake_completion_response("UPSERT")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    runner = LLMRunner(PROVIDERS)
    out = runner.complete_messages(
        [{"role": "user", "content": "hi"}], max_tokens=10, temperature=0.0,
    )
    assert out == "UPSERT"
    assert calls["kw"]["model"] == "groq/x"
    assert calls["kw"]["stream"] is False


def test_complete_messages_falls_back_on_error(monkeypatch):
    import litellm
    seq = iter([RuntimeError("boom"), _fake_completion_response("ok2")])

    def fake_completion(**kw):
        nxt = next(seq)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    monkeypatch.setattr(litellm, "completion", fake_completion)
    runner = LLMRunner([
        {"pid": "a", "litellm_model": "groq/a", "api_base": None, "api_key": "k"},
        {"pid": "b", "litellm_model": "groq/b", "api_base": None, "api_key": "k"},
    ], retries=1)
    assert runner.complete_messages([{"role": "user", "content": "x"}]) == "ok2"


def test_stream_messages_yields_deltas(monkeypatch):
    import litellm

    def fake_completion(**kw):
        assert kw["stream"] is True
        return _fake_stream_chunks(["@U ", "alice ", "Alice"])

    monkeypatch.setattr(litellm, "completion", fake_completion)
    runner = LLMRunner(PROVIDERS)
    got = list(runner.stream_messages([{"role": "user", "content": "x"}]))
    assert "".join(got) == "@U alice Alice"
