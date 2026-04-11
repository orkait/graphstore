"""FastEmbed wrapper - Qdrant's lightweight embedder library.

FastEmbed ships pre-exported ONNX weights for ~30 popular embedding models
(BGE, mxbai, e5, snowflake, jina-v2, MiniLM, nomic) and runs them through
onnxruntime with built-in batching. We already depend on onnxruntime and
tokenizers, so adding fastembed is a small delta (~20 MB of Python glue
and small deps) that unlocks strong encoder embedders without torch.

Use this instead of the custom OnnxHFEmbedder path for any model FastEmbed
already knows about - FastEmbed handles model downloading, tokenizer setup,
asymmetric query/passage prefixes (for e5 and bge), and batched inference.
"""

import numpy as np

from graphstore.embedding.base import Embedder


class FastEmbedEmbedder(Embedder):
    """Wraps :class:`fastembed.TextEmbedding` with graphstore's Embedder API.

    FastEmbed exposes three encode methods:

    - ``embed(texts)`` - generic; no prefix applied
    - ``query_embed(texts)`` - adds the query prefix for asymmetric models
    - ``passage_embed(texts)`` - adds the passage/document prefix

    For symmetric models (MiniLM, bge-small-zh) all three produce the same
    output. For asymmetric retrieval models (e5, bge-*-en-v1.5) using
    ``query_embed``/``passage_embed`` is what actually wires up the
    train-time prompt instructions. graphstore always maps
    ``encode_queries`` → ``query_embed`` and ``encode_documents`` →
    ``passage_embed`` so the right prefixes are applied regardless of the
    underlying model.
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
                "FastEmbedEmbedder requires fastembed. "
                "Install with: pip install fastembed"
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
        # fastembed.passage_embed returns an iterator of numpy arrays; consume
        # it in one shot since the whole batch is already in memory upstream.
        vecs = list(self._model.passage_embed(list(texts)))
        return np.stack(vecs).astype(np.float32, copy=False)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dims), dtype=np.float32)
        vecs = list(self._model.query_embed(list(texts)))
        return np.stack(vecs).astype(np.float32, copy=False)
