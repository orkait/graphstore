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


def test_synthesis_shim_reexports_bonsai_pipeline():
    from graphstore.ingest.llm import synthesis as S
    # @-verb parse -> ParsedTurn (v6 verbs are full words: UPSERT / FACT / ...)
    turn = S.parse_verb_output('@UPSERT alice Alice\n@FACT fav_color blue')
    assert ("alice", "Alice") in turn.entities
    assert ("fact:fav_color", "blue") in turn.beliefs
    # whole-turn synthesis (gs=None => dry, mints entities) yields DSL lines
    dsl = S.synthesize_dsl(
        turn, msg_id="msg:1", session_id="s", role="user",
        text="alice likes blue", gs=None,
    )
    assert any(line.startswith('CREATE NODE "msg:1"') for line in dsl)
    assert any(line.startswith("ASSERT ") for line in dsl)
    # types + errors are re-exported
    assert S.IngestError is S.BonsaiError
    assert issubclass(S.IngestEmpty, S.IngestError)
    r = S.IngestResult(statements=["x"], executed=1, rejected=[],
                       entities_new=[], beliefs_changed=[], duration_ms=0)
    assert r.executed == 1
