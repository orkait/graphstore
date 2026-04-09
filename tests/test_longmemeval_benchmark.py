import json
from pathlib import Path

import numpy as np

from graphstore.embedding.base import Embedder

from benchmarks.longmemeval import (
    build_corpus,
    evaluate_retrieval,
    load_longmemeval,
    main,
    session_id_from_corpus_id,
)


FIXTURE = Path(__file__).parent / "fixtures" / "benchmarks" / "longmemeval_sample.json"


class MockEmbedder(Embedder):
    @property
    def name(self) -> str:
        return "mock"

    @property
    def dims(self) -> int:
        return 64

    def encode_documents(self, texts, titles=None):
        return self._encode(texts)

    def encode_queries(self, texts):
        return self._encode(texts)

    def _encode(self, texts):
        vectors = []
        for text in texts:
            seed = hash(text) % (2**31)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self.dims).astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


def test_load_longmemeval_reads_entries():
    data = load_longmemeval(FIXTURE)
    assert len(data) == 2
    assert data[0]["question_id"] == "q1"


def test_build_corpus_session_uses_all_turns():
    entry = load_longmemeval(FIXTURE)[0]
    items = build_corpus(entry, granularity="session")
    assert len(items) == 2
    assert items[0].session_id == "sess_1"
    assert "Seattle sounds like a great move." in items[0].text


def test_build_corpus_turn_keeps_session_mapping():
    entry = load_longmemeval(FIXTURE)[0]
    items = build_corpus(entry, granularity="turn")
    assert len(items) == 4
    assert items[1].turn_id == 1
    assert session_id_from_corpus_id(items[1].corpus_id) == "sess_1"


def test_evaluate_retrieval_scores_known_ranking():
    ranked_ids = ["sess_2", "sess_1", "sess_3"]
    recall_any, recall_all, ndcg = evaluate_retrieval(
        ranked_ids=ranked_ids,
        correct_ids={"sess_1"},
        k=2,
    )
    assert recall_any == 1.0
    assert recall_all == 1.0
    assert 0.0 < ndcg <= 1.0


def test_cli_smoke_runs_lexical_mode(tmp_path):
    out_path = tmp_path / "results.jsonl"
    exit_code = main([
        str(FIXTURE),
        "--mode", "lexical",
        "--limit", "1",
        "--out", str(out_path),
    ])
    assert exit_code == 0
    assert out_path.exists()
    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == 1
