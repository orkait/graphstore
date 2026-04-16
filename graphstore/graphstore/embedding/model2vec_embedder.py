"""Model2Vec embedder: lightweight, numpy-only, 50k texts/sec on CPU."""

from pathlib import Path
import numpy as np
from graphstore.embedding.base import Embedder

_model_cache: dict = {}


class Model2VecEmbedder(Embedder):
    """Default embedder. 30MB, numpy-only, zero-config.

    Symmetric model - queries and documents use the same encoding.
    Model instance is cached at module level so repeated construction is free.
    """

    def __init__(self, model_name: str = "minishlab/M2V_base_output", cache_dir: str | None = None):
        cache_key = (model_name, cache_dir)
        if cache_key not in _model_cache:
            try:
                from model2vec import StaticModel
            except ImportError as e:
                raise ImportError(
                    "Model2VecEmbedder requires the `embed-default` extra. "
                    "Install with: pip install 'graphstore[embed-default]'"
                ) from e
            
            # If cache_dir is provided and model exists there, try loading from local
            if cache_dir:
                import os
                local_path = Path(cache_dir) / model_name.split("/")[-1]
                if local_path.exists():
                    _model_cache[cache_key] = StaticModel.from_pretrained(str(local_path))
                else:
                    # Set HF_HOME to cache_dir temporarily
                    old_hf_home = os.environ.get("HF_HOME")
                    os.environ["HF_HOME"] = str(cache_dir)
                    try:
                        _model_cache[cache_key] = StaticModel.from_pretrained(model_name)
                    finally:
                        if old_hf_home:
                            os.environ["HF_HOME"] = old_hf_home
                        else:
                            del os.environ["HF_HOME"]
            else:
                _model_cache[cache_key] = StaticModel.from_pretrained(model_name)
                
        self._model = _model_cache[cache_key]
        self._name = "model2vec"

    @property
    def name(self) -> str:
        return self._name

    @property
    def dims(self) -> int:
        return self._model.dim

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dims), dtype=np.float32)
        return self._model.encode(texts).astype(np.float32)

    def encode_documents(self, texts: list[str], titles: list[str | None] | None = None) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)
