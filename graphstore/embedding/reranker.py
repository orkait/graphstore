"""Pluggable reranker interface + implementations.

Two backends:
- FlashRankReranker: 4MB default model, no torch, CPU. Best for most users.
- OnnxReranker: Any ONNX cross-encoder (GTE, BGE, Jina). For power users.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class Reranker(Protocol):
    """Any reranker must implement score()."""

    def score(self, query: str, documents: list[str]) -> np.ndarray:
        """Score query-document pairs. Returns array of relevance scores (higher = better)."""
        ...


class FlashRankReranker:
    """Tiny cross-encoder reranker via FlashRank.

    Default model is 4MB (ms-marco-TinyBERT-L-2-v2). No torch required.
    Install: pip install flashrank
    """

    def __init__(self, model_name: str = "rank-T5-flan", max_length: int = 512):
        try:
            from flashrank import Ranker
        except ImportError as e:
            raise ImportError(
                "FlashRankReranker requires flashrank. "
                "Install with: pip install flashrank"
            ) from e

        self._ranker = Ranker(model_name=model_name, max_length=max_length)
        self._model_name = model_name

    def score(self, query: str, documents: list[str]) -> np.ndarray:
        if not documents:
            return np.empty(0, dtype=np.float64)

        from flashrank import RerankRequest

        passages = [{"id": i, "text": doc} for i, doc in enumerate(documents)]
        request = RerankRequest(query=query, passages=passages)
        results = self._ranker.rerank(request)
        scores = np.zeros(len(documents), dtype=np.float64)
        for r in results:
            scores[r["id"]] = r["score"]
        return scores


class OnnxReranker:
    """Cross-encoder reranker via ONNX Runtime.

    Works with any cross-encoder model (GTE, BGE, Jina, etc.)
    that takes (query, document) pairs and outputs relevance scores.
    """

    def __init__(self, model_dir: str | Path, onnx_file: str = "onnx/model_int8.onnx", max_length: int = 512):
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:
            raise ImportError(
                "OnnxReranker requires onnxruntime and tokenizers."
            ) from e

        model_dir = Path(model_dir)
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=max_length)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        onnx_path = model_dir / onnx_file
        self._session = ort.InferenceSession(str(onnx_path), sess_options=sess_options)
        self._input_names = {i.name for i in self._session.get_inputs()}

    def score(self, query: str, documents: list[str]) -> np.ndarray:
        if not documents:
            return np.empty(0, dtype=np.float64)

        encoded = [self._tokenizer.encode(query, doc) for doc in documents]

        max_len = max(len(e.ids) for e in encoded)
        input_ids = np.zeros((len(encoded), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(encoded), max_len), dtype=np.int64)

        for i, enc in enumerate(encoded):
            length = len(enc.ids)
            input_ids[i, :length] = enc.ids
            attention_mask[i, :length] = enc.attention_mask

        feed = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids)

        logits = self._session.run(None, feed)[0]
        if logits.ndim == 2:
            logits = logits[:, 0]
        return logits.astype(np.float64)
