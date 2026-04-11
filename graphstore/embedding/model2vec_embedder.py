"""Model2Vec embedder: lightweight, numpy-only, 50k texts/sec on CPU."""

import numpy as np
from graphstore.embedding.base import Embedder

_model_cache: dict[str, object] = {}


class Model2VecEmbedder(Embedder):
    """Default embedder. 30MB, numpy-only, zero-config.

    Symmetric model - queries and documents use the same encoding.
    Model instance is cached at module level so repeated construction is free.
    """

    def __init__(self, model_name: str = "minishlab/M2V_base_output"):
        if model_name not in _model_cache:
            try:
                from model2vec import StaticModel
            except ImportError as e:
                raise ImportError(
                    "Model2VecEmbedder requires the `embed-default` extra. "
                    "Install with: pip install 'graphstore[embed-default]'"
                ) from e
            _model_cache[model_name] = StaticModel.from_pretrained(model_name)
        self._model = _model_cache[model_name]
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
