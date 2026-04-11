"""Generic HuggingFace ONNX embedder using tokenizers + onnxruntime."""

import numpy as np
from pathlib import Path
from graphstore.embedding.base import Embedder
from graphstore.embedding.postprocess import l2_normalize, truncate_dims


class OnnxHFEmbedder(Embedder):
    """HuggingFace ONNX model embedder. No torch required.

    Uses `tokenizers` for tokenization and `onnxruntime` for inference.
    Supports asymmetric query/document prefixes, Matryoshka truncation,
    and both mean and last-token pooling (for encoder vs decoder models).
    """

    def __init__(
        self,
        model_dir: str | Path,
        output_dims: int | None = None,
        query_prefix: str = "",
        doc_prefix_template: str = "",
        max_length: int = 512,
        pooling_mode: str = "mean",
        onnx_file: str | None = None,
    ):
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:
            raise ImportError(
                "OnnxHFEmbedder requires onnxruntime and tokenizers. "
                "Install with: graphstore install-embedder embeddinggemma"
            ) from e

        if pooling_mode not in ("mean", "last_token"):
            raise ValueError(
                f"pooling_mode must be 'mean' or 'last_token', got {pooling_mode!r}"
            )

        model_dir = Path(model_dir)
        self._output_dims = output_dims
        self._query_prefix = query_prefix
        self._doc_prefix_template = doc_prefix_template
        self._max_length = max_length
        self._pooling_mode = pooling_mode

        # Load tokenizer
        tok_path = model_dir / "tokenizer.json"
        if not tok_path.exists():
            raise FileNotFoundError(f"tokenizer.json not found in {model_dir}")
        self._tokenizer = Tokenizer.from_file(str(tok_path))
        self._tokenizer.enable_padding(pad_id=0)
        self._tokenizer.enable_truncation(max_length=max_length)

        # Load ONNX model — prefer explicit file from manifest if provided.
        if onnx_file:
            candidate = model_dir / onnx_file
            if not candidate.exists():
                raise FileNotFoundError(f"ONNX file not found: {candidate}")
            onnx_path = candidate
        else:
            onnx_files = list(model_dir.glob("*.onnx"))
            if not onnx_files:
                onnx_dir = model_dir / "onnx"
                if onnx_dir.exists():
                    onnx_files = list(onnx_dir.glob("*.onnx"))
            if not onnx_files:
                raise FileNotFoundError(f"No .onnx file found in {model_dir}")
            onnx_path = onnx_files[0]

        self._session = ort.InferenceSession(str(onnx_path))
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._needs_token_type_ids = "token_type_ids" in self._input_names
        self._needs_position_ids = "position_ids" in self._input_names

        # Decoder-style exports (Qwen3, Llama, Mistral) expose past_key_values
        # inputs and require a prefill pass with empty KV cache. Cache the
        # per-layer shape spec so _encode can build zero tensors on demand.
        self._kv_cache_specs: list = []
        self._hidden_output_idx: int = 0
        for inp in self._session.get_inputs():
            if inp.name.startswith("past_key_values."):
                shape = inp.shape
                num_heads = shape[1] if isinstance(shape[1], int) else None
                head_dim = shape[3] if isinstance(shape[3], int) else None
                dtype = np.float16 if inp.type == "tensor(float16)" else np.float32
                self._kv_cache_specs.append((inp.name, num_heads, head_dim, dtype))
        if self._kv_cache_specs:
            for i, out in enumerate(self._session.get_outputs()):
                if out.name == "last_hidden_state" or "hidden" in out.name:
                    self._hidden_output_idx = i
                    break

        test_enc = self._tokenizer.encode("test")
        test_ids = np.array([[test_enc.ids[0]]], dtype=np.int64)
        test_mask = np.array([[1]], dtype=np.int64)
        test_feed = {"input_ids": test_ids, "attention_mask": test_mask}
        if self._needs_token_type_ids:
            test_feed["token_type_ids"] = np.zeros((1, 1), dtype=np.int64)
        if self._needs_position_ids:
            test_feed["position_ids"] = np.zeros((1, 1), dtype=np.int64)
        for name, num_heads, head_dim, dtype in self._kv_cache_specs:
            test_feed[name] = np.zeros((1, num_heads, 0, head_dim), dtype=dtype)
        test_out = self._session.run(None, test_feed)
        self._base_dims = test_out[self._hidden_output_idx].shape[-1]
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

        feed = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if self._needs_token_type_ids:
            feed["token_type_ids"] = np.zeros_like(input_ids)
        if self._needs_position_ids:
            pos = attention_mask.cumsum(axis=1) - 1
            pos = np.maximum(pos, 0)
            feed["position_ids"] = pos.astype(np.int64)
        if self._kv_cache_specs:
            for name, num_heads, head_dim, dtype in self._kv_cache_specs:
                feed[name] = np.zeros((batch, num_heads, 0, head_dim), dtype=dtype)

        outputs = self._session.run(None, feed)
        embeddings = outputs[self._hidden_output_idx]

        # Some ONNX exports (e.g. Harrier fp16) bake the SentenceTransformer
        # pooling head into the graph and return (batch, hidden_dim) directly.
        # Others return raw (batch, seq_len, hidden_dim) and expect the caller
        # to pool. Handle both.
        if embeddings.ndim == 2:
            pooled = embeddings
        elif self._pooling_mode == "last_token":
            last_idx = attention_mask.sum(axis=1) - 1
            batch_idx = np.arange(embeddings.shape[0])
            pooled = embeddings[batch_idx, last_idx]
        else:
            mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
            pooled = (embeddings * mask_expanded).sum(axis=1) / mask_expanded.sum(axis=1)

        # Matryoshka truncation + renormalize
        if self._output_dims and self._output_dims < pooled.shape[1]:
            pooled = truncate_dims(pooled, self._output_dims)
        else:
            pooled = l2_normalize(pooled)

        return pooled.astype(np.float32)
