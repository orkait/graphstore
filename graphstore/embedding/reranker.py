"""Pluggable reranker interface + implementations.

Three backends:
- FlashRankReranker: 22MB default model, no torch, CPU. Best for most users.
- OnnxReranker: Any ONNX cross-encoder (GTE, BGE). For power users.
- GGUFReranker: Jina Reranker v3 via llama-cpp-python. CUDA/Metal native.
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


class GGUFReranker:
    """Late-interaction reranker via llama-cpp-python + projector MLP.

    Designed for Jina Reranker v3: embed query and docs via GGUF model,
    project through MLP, score by cosine similarity. Native CUDA/Metal.
    """

    def __init__(self, model_path: str, projector_path: str | None = None,
                 n_ctx: int = 2048, n_gpu_layers: int = -1):
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "GGUFReranker requires llama-cpp-python. "
                "Install with: pip install llama-cpp-python"
            ) from e

        self._model = Llama(
            model_path=model_path,
            embedding=True,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

        self._projector = None
        if projector_path:
            from safetensors import safe_open
            with safe_open(projector_path, framework="numpy") as f:
                self._proj_w1 = f.get_tensor("projector.0.weight")
                self._proj_w2 = f.get_tensor("projector.2.weight")

    def _embed_and_project(self, text: str) -> np.ndarray:
        emb = np.array(self._model.embed(text), dtype=np.float32)
        if emb.ndim == 2:
            emb = emb.mean(axis=0)
        if self._proj_w1 is not None:
            emb = emb @ self._proj_w1.T
            emb = np.maximum(emb, 0)  # ReLU
            emb = emb @ self._proj_w2.T
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def score(self, query: str, documents: list[str]) -> np.ndarray:
        if not documents:
            return np.empty(0, dtype=np.float64)

        q_emb = self._embed_and_project(query)
        scores = np.zeros(len(documents), dtype=np.float64)
        for i, doc in enumerate(documents):
            d_emb = self._embed_and_project(doc)
            scores[i] = float(np.dot(q_emb, d_emb))
        return scores
