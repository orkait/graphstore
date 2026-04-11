"""Generic HuggingFace ONNX embedder using tokenizers + onnxruntime."""

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

from graphstore.embedding.base import Embedder
from graphstore.embedding.postprocess import l2_normalize, truncate_dims


_CU12_PRELOADED = False

# cu12-family sonames ORT's CUDA provider .so looks up at session creation.
# Ordered so dependents load after their dependencies (cudart first).
_CU12_SONAMES = (
    "libcudart.so.12",
    "libnvrtc.so.12",
    "libnvJitLink.so.12",
    "libcublas.so.12",
    "libcublasLt.so.12",
    "libcufft.so.11",
    "libcurand.so.10",
    "libcudnn.so.9",
)

# Map soname → (nvidia wheel subdir, wheel lib filename).
# We look these up inside $site-packages/nvidia/<sub>/lib when the
# filesystem probe can't find the libs in a system path.
_WHEEL_LAYOUT: dict[str, str] = {
    "libcudart.so.12": "cuda_runtime/lib",
    "libnvrtc.so.12": "cuda_nvrtc/lib",
    "libnvJitLink.so.12": "nvjitlink/lib",
    "libcublas.so.12": "cublas/lib",
    "libcublasLt.so.12": "cublas/lib",
    "libcufft.so.11": "cufft/lib",
    "libcurand.so.10": "curand/lib",
    "libcudnn.so.9": "cudnn/lib",
}


def _apparmor_userns_blocks_cuda() -> bool:
    """True if Ubuntu 24.04+ apparmor userns restriction is active.

    When set, cudaGetDeviceCount() fails with error 304 from any
    venv-isolated process - the exact reason every first-time GPU run
    on a fresh 24.04 box mysteriously breaks.
    """
    path = "/proc/sys/kernel/apparmor_restrict_unprivileged_userns"
    try:
        with open(path) as f:
            return f.read().strip() == "1"
    except (FileNotFoundError, PermissionError, OSError):
        return False


def _system_cuda_libs_present() -> bool:
    """Filesystem-only probe: are all required cu12 sonames in a path
    the dynamic linker will search?

    No dlopen - dlopen caches by soname and would pollute later fallback
    attempts with partial loads if one lib is missing. We only check
    existence on disk.
    """
    search: list[str] = []
    for env in ("CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(env)
        if value:
            search.append(os.path.join(value, "lib64"))
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    search.extend(p for p in ld.split(":") if p)
    search.extend([
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-12/lib64",
        "/usr/lib/x86_64-linux-gnu",
    ])
    for soname in _CU12_SONAMES:
        if not any(os.path.exists(os.path.join(d, soname)) for d in search):
            return False
    return True


def _preload_cu12_libs() -> None:
    """Make CUDA 12 runtime libs findable by onnxruntime-gpu.

    Strategy:
        1. Non-Linux: no-op. cu12 wheels are Linux x86_64 only.
        2. Ubuntu 24.04 apparmor userns restriction active: raise a
           targeted error with the sysctl fix - otherwise the user
           would hit an opaque "CUDA failure 304" at session creation.
        3. System already has the libs in a standard search path:
           skip preload. ORT's own dlopen will find them.
        4. nvidia-*-cu12 pip wheels installed alongside onnxruntime-gpu:
           dlopen each lib from the wheel location so subsequent
           ORT provider loads resolve the same soname to our wheel.
        5. None of the above: do nothing - ORT will raise a clear
           "libcublasLt.so.12 not found" at session creation which
           points users at ``pip install graphstore[gpu]``.

    Idempotent.
    """
    global _CU12_PRELOADED
    if _CU12_PRELOADED or sys.platform != "linux":
        return

    if _apparmor_userns_blocks_cuda():
        raise RuntimeError(
            "CUDA cannot initialize: Ubuntu 24.04+ apparmor is blocking\n"
            "unprivileged user namespaces (error 304 from cudaGetDeviceCount).\n"
            "\n"
            "Temporary fix for this session:\n"
            "    sudo sysctl kernel.apparmor_restrict_unprivileged_userns=0\n"
            "\n"
            "Restore the safeguard after the run with:\n"
            "    sudo sysctl kernel.apparmor_restrict_unprivileged_userns=1"
        )

    if _system_cuda_libs_present():
        _CU12_PRELOADED = True
        return

    for sp in (p for p in sys.path if p.endswith("site-packages")):
        nvidia_root = Path(sp) / "nvidia"
        if not nvidia_root.exists():
            continue
        for soname in _CU12_SONAMES:
            lib_path = nvidia_root / _WHEEL_LAYOUT[soname] / soname
            if lib_path.exists():
                try:
                    ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
    _CU12_PRELOADED = True


def _resolve_providers(providers: list[str] | str | None) -> list[str]:
    """Pick onnxruntime execution providers with sensible fallbacks.

    Resolution order:
        1. Explicit ``providers`` argument (list or comma string)
        2. ``GRAPHSTORE_ORT_PROVIDERS`` env var (comma string)
        3. ``GRAPHSTORE_GPU=1`` env var → try CUDA/Tensorrt first
        4. Default → CPU only

    Unavailable providers are silently dropped - if CUDA is requested
    but the installed onnxruntime wheel is CPU-only, we fall back to
    CPU rather than erroring.
    """
    wanted: list[str] = []
    if providers is not None:
        if isinstance(providers, str):
            wanted = [p.strip() for p in providers.split(",") if p.strip()]
        else:
            wanted = list(providers)
    else:
        env_providers = os.environ.get("GRAPHSTORE_ORT_PROVIDERS")
        if env_providers:
            wanted = [p.strip() for p in env_providers.split(",") if p.strip()]
        elif os.environ.get("GRAPHSTORE_GPU") == "1":
            wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            wanted = ["CPUExecutionProvider"]

    # Preload CUDA 12 shared libs from nvidia-*-cu12 pip wheels before
    # onnxruntime probes its provider plugins - otherwise get_available_providers
    # may silently drop CUDAExecutionProvider because the cuda provider .so
    # fails its dlopen of libcublasLt.so.12 / libcudnn.so.9 / libcudart.so.12.
    if any(p in ("CUDAExecutionProvider", "TensorrtExecutionProvider") for p in wanted):
        _preload_cu12_libs()

    import onnxruntime as ort
    available = set(ort.get_available_providers())

    resolved = [p for p in wanted if p in available]
    if "CPUExecutionProvider" not in resolved:
        resolved.append("CPUExecutionProvider")
    return resolved


class OnnxHFEmbedder(Embedder):
    """HuggingFace ONNX model embedder. No torch required.

    Uses `tokenizers` for tokenization and `onnxruntime` for inference.
    Supports asymmetric query/document prefixes, Matryoshka truncation,
    and both mean and last-token pooling (for encoder vs decoder models).

    GPU (Linux x86_64 only): set ``GRAPHSTORE_GPU=1`` or pass
    ``providers=["CUDAExecutionProvider", "CPUExecutionProvider"]``.

    Two install paths:
        - ``pip install 'graphstore[gpu]'``     - zero-config, bundles
          onnxruntime-gpu plus the nvidia-*-cu12 runtime wheels.
        - ``pip install 'graphstore[gpu-ort]'`` - leaner, assumes the
          host already has CUDA 12 / cuDNN 9 in a standard search path
          (system install, conda env, or LD_LIBRARY_PATH).

    Falls back to CPU silently if CUDA is unavailable. Raises a targeted
    error on Ubuntu 24.04 if apparmor's unprivileged-userns restriction
    is blocking cudaGetDeviceCount().
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
        providers: list[str] | str | None = None,
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

        # Load ONNX model - prefer explicit file from manifest if provided.
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

        self._providers = _resolve_providers(providers)
        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=self._providers,
        )
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._needs_token_type_ids = "token_type_ids" in self._input_names
        self._needs_position_ids = "position_ids" in self._input_names

        # Decoder-style exports (Qwen3, Llama, Mistral) expose past_key_values
        # inputs and require a prefill pass with empty KV cache. Cache the
        # per-layer shape spec so _encode can build zero tensors on demand.
        self._kv_cache_specs: list = []
        for inp in self._session.get_inputs():
            if inp.name.startswith("past_key_values."):
                shape = inp.shape
                num_heads = shape[1] if isinstance(shape[1], int) else None
                head_dim = shape[3] if isinstance(shape[3], int) else None
                dtype = np.float16 if inp.type == "tensor(float16)" else np.float32
                self._kv_cache_specs.append((inp.name, num_heads, head_dim, dtype))

        # Output selection priority:
        #   1. sentence_embedding - SBERT-style exports with pooling baked in
        #   2. last_hidden_state / anything containing "hidden" - raw 3D hidden states
        #   3. first output - fallback
        outputs = self._session.get_outputs()
        self._hidden_output_idx = 0
        for i, out in enumerate(outputs):
            if out.name == "sentence_embedding":
                self._hidden_output_idx = i
                break
        else:
            for i, out in enumerate(outputs):
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
