"""Model installer: download, verify, activate."""

import subprocess
import sys
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
from graphstore.registry.models import get_model_info, SUPPORTED_MODELS


DEFAULT_CACHE_DIR = Path.home() / ".graphstore" / "models"

_cache_dir_override: Path | None = None


def set_cache_dir(path: str | Path | None) -> None:
    """Override the model cache directory (from config)."""
    global _cache_dir_override
    _cache_dir_override = Path(path) if path else None


def get_model_dir(name: str) -> Path:
    base = _cache_dir_override if _cache_dir_override is not None else DEFAULT_CACHE_DIR
    return base / name


def is_installed(name: str) -> bool:
    model_dir = get_model_dir(name)
    return model_dir.exists() and any(model_dir.rglob("*.onnx"))


def install_embedder(name: str, variant: str | None = None) -> Path:
    """Download and install an embedder model.

    1. Check deps are installed, install if not
    2. Download model files from HuggingFace
    3. Write manifest
    4. Return model directory
    """
    info = get_model_info(name)
    if info is None:
        available = list(SUPPORTED_MODELS.keys())
        raise ValueError(f"Unknown model: {name!r}. Available: {available}")

    variant = variant or info["default_variant"]
    model_dir = get_model_dir(name)
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Install missing deps
    for dep in info["deps"]:
        try:
            __import__(dep.replace("-", "_"))
        except ImportError:
            print(f"Installing {dep}...")
            # Check for GPU
            if dep == "onnxruntime":
                dep_name = _detect_onnx_package()
            else:
                dep_name = dep
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep_name, "-q"])

    # 2. Download model files
    from huggingface_hub import hf_hub_download

    repo_id = info["repo_id"]
    variant_info = info["variants"][variant]

    # Download tokenizer files
    for tok_file in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        try:
            hf_hub_download(repo_id, tok_file, local_dir=str(model_dir))
        except Exception as e:
            logger.debug("tokenizer file %r download skipped: %s", tok_file, e, exc_info=True)

    # Download ONNX model files
    for f in variant_info["files"]:
        print(f"Downloading {f}...")
        hf_hub_download(repo_id, f, local_dir=str(model_dir))

    # 3. Write manifest
    import json
    manifest = {
        "name": name,
        "variant": variant,
        "dims": info["base_dims"],
        "default_dims": info["default_dims"],
        "max_length": info["max_length"],
        "query_prefix": info["query_prefix"],
        "doc_prefix_template": info["doc_prefix_template"],
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Installed {name} ({variant}) to {model_dir}")
    return model_dir


def uninstall_embedder(name: str) -> None:
    model_dir = get_model_dir(name)
    if model_dir.exists():
        shutil.rmtree(model_dir)
        print(f"Uninstalled {name}")
    else:
        print(f"{name} is not installed")


def load_installed_embedder(name: str, dims: int | None = None):
    """Load an installed ONNX embedder."""
    model_dir = get_model_dir(name)
    if not is_installed(name):
        raise FileNotFoundError(
            f"Model {name!r} not installed. Run: graphstore install-embedder {name}"
        )

    import json
    manifest = json.loads((model_dir / "manifest.json").read_text())

    from graphstore.embedding.onnx_hf_embedder import OnnxHFEmbedder
    return OnnxHFEmbedder(
        model_dir=model_dir,
        output_dims=dims or manifest["default_dims"],
        query_prefix=manifest.get("query_prefix", ""),
        doc_prefix_template=manifest.get("doc_prefix_template", ""),
        max_length=manifest.get("max_length", 512),
    )


def _detect_onnx_package() -> str:
    """Detect GPU and return appropriate onnxruntime package."""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True)
        if result.returncode == 0:
            return "onnxruntime-gpu"
    except FileNotFoundError:
        pass
    return "onnxruntime"
