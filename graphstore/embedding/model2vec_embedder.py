"""Model2Vec embedder: lightweight, numpy-only, 50k texts/sec on CPU."""

import numpy as np
from graphstore.embedding.base import Embedder


class Model2VecEmbedder(Embedder):
    """Default embedder. 30MB, numpy-only, zero-config.

    Symmetric model - queries and documents use the same encoding.
    """

    def __init__(self, model_name: str = "minishlab/M2V_base_output"):
        from model2vec import StaticModel
        self._model = StaticModel.from_pretrained(model_name)
        self._name = "model2vec"

    @property
    def name(self) -> str:
        return self._name

    @property
    def dims(self) -> int:
        return self._model.dim

    def encode_documents(self, texts: list[str], titles: list[str | None] | None = None) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dims), dtype=np.float32)
        return self._model.encode(texts).astype(np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dims), dtype=np.float32)
        return self._model.encode(texts).astype(np.float32)
