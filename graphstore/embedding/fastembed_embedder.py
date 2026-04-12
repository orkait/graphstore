"""FastEmbed wrapper - Qdrant's lightweight ONNX embedder library."""

import numpy as np

from graphstore.embedding.base import Embedder


class FastEmbedEmbedder(Embedder):
    """Wraps fastembed.TextEmbedding with graphstore's Embedder API.

    Maps encode_queries -> query_embed and encode_documents -> passage_embed
    so asymmetric models (e5, bge-*-en-v1.5) get the correct prefixes.
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        threads: int | None = None,
    ):
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise ImportError(
                "FastEmbedEmbedder requires the `embed-fastembed` extra. "
                "Install with: pip install 'graphstore[embed-fastembed]'"
            ) from e

        self._model_name = model_name
        self._model = TextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=threads,
        )
        self._dims = int(self._model.embedding_size)

    @property
    def name(self) -> str:
        return f"fastembed:{self._model_name}"

    @property
    def dims(self) -> int:
        return self._dims

    def encode_documents(
        self,
        texts: list[str],
        titles: list[str | None] | None = None,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dims), dtype=np.float32)
        return np.array(list(self._model.passage_embed(texts)), dtype=np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dims), dtype=np.float32)
        return np.array(list(self._model.query_embed(texts)), dtype=np.float32)
