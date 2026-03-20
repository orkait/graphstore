"""Generic HuggingFace ONNX embedder using tokenizers + onnxruntime."""

import numpy as np
from pathlib import Path
from graphstore.embedding.base import Embedder
from graphstore.embedding.postprocess import l2_normalize, truncate_dims


class OnnxHFEmbedder(Embedder):
    """HuggingFace ONNX model embedder. No torch required.

    Uses `tokenizers` for tokenization and `onnxruntime` for inference.
    Supports asymmetric query/document prefixes and Matryoshka truncation.
    """

    def __init__(
        self,
        model_dir: str | Path,
        output_dims: int | None = None,
        query_prefix: str = "",
        doc_prefix_template: str = "",
        max_length: int = 512,
    ):
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:
            raise ImportError(
                "OnnxHFEmbedder requires onnxruntime and tokenizers. "
                "Install with: graphstore install-embedder embeddinggemma"
            ) from e

        model_dir = Path(model_dir)
        self._output_dims = output_dims
        self._query_prefix = query_prefix
        self._doc_prefix_template = doc_prefix_template
        self._max_length = max_length

        # Load tokenizer
        tok_path = model_dir / "tokenizer.json"
        if not tok_path.exists():
            raise FileNotFoundError(f"tokenizer.json not found in {model_dir}")
        self._tokenizer = Tokenizer.from_file(str(tok_path))
        self._tokenizer.enable_padding(pad_id=0)
        self._tokenizer.enable_truncation(max_length=max_length)

        # Load ONNX model
        onnx_files = list(model_dir.glob("*.onnx"))
        if not onnx_files:
            onnx_dir = model_dir / "onnx"
            if onnx_dir.exists():
                onnx_files = list(onnx_dir.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"No .onnx file found in {model_dir}")

        self._session = ort.InferenceSession(str(onnx_files[0]))

        # Detect output dims from a test run
        test_enc = self._tokenizer.encode("test")
        test_ids = np.array([[test_enc.ids[0]]], dtype=np.int64)
        test_mask = np.array([[1]], dtype=np.int64)
        test_out = self._session.run(None, {"input_ids": test_ids, "attention_mask": test_mask})
        self._base_dims = test_out[0].shape[-1]
        if self._output_dims is None:
            self._output_dims = self._base_dims

    @property
    def name(self) -> str:
        return "onnx-hf"

    @property
    def dims(self) -> int:
        return self._output_dims

    def encode_documents(self, texts: list[str], titles: list[str | None] | None = None) -> np.ndarray:
        if not texts:
            return np.empty((0, self._output_dims), dtype=np.float32)
        prefixed = []
        for i, text in enumerate(texts):
            title = titles[i] if titles and i < len(titles) and titles[i] else "none"
            prefix = self._doc_prefix_template.format(title=title) if self._doc_prefix_template else ""
            prefixed.append(f"{prefix}{text}")
        return self._encode(prefixed)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._output_dims), dtype=np.float32)
        prefixed = [f"{self._query_prefix}{t}" for t in texts]
        return self._encode(prefixed)

    def _encode(self, texts: list[str]) -> np.ndarray:
        encoded = self._tokenizer.encode_batch(texts)
        max_len = max(len(e.ids) for e in encoded)
        batch = len(encoded)

        input_ids = np.zeros((batch, max_len), dtype=np.int64)
        attention_mask = np.zeros((batch, max_len), dtype=np.int64)
        for i, e in enumerate(encoded):
            input_ids[i, :len(e.ids)] = e.ids
            attention_mask[i, :len(e.attention_mask)] = e.attention_mask

        outputs = self._session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })
        embeddings = outputs[0]  # (batch, seq_len, hidden_dim)

        # Mean pooling
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        pooled = (embeddings * mask_expanded).sum(axis=1) / mask_expanded.sum(axis=1)

        # Matryoshka truncation + renormalize
        if self._output_dims and self._output_dims < pooled.shape[1]:
            pooled = truncate_dims(pooled, self._output_dims)
        else:
            pooled = l2_normalize(pooled)

        return pooled.astype(np.float32)
