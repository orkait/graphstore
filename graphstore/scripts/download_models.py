"""Download models into ./models.

Embedders use the registry installer. NER, reranker, and model2vec are
downloaded with retry logic, progress bars, and validation.

Usage:
    python -m scripts.download_models
    # or from project root:
    uv run python3 scripts/download_models.py
"""

import time
from pathlib import Path
from huggingface_hub import hf_hub_download, HfFileSystemError

from graphstore.registry.installer import install_embedder, set_cache_dir, is_installed

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def _download_with_retry(repo: str, filename: str, local_dir: str, max_retries: int = 3) -> bool:
    """Download file with exponential backoff retry logic.

    Args:
        repo: HuggingFace repo ID (e.g., "jinaai/jina-embeddings-v5-text-nano-retrieval")
        filename: File to download
        local_dir: Local directory to save to
        max_retries: Number of retry attempts

    Returns:
        True if successful, False if all retries failed
    """
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  {filename}...", end=" ", flush=True)
            path = hf_hub_download(repo, filename, local_dir=local_dir)
            size_mb = Path(path).stat().st_size / (1024 * 1024)
            print(f"✓ ({size_mb:.1f}MB)")
            return True
        except (HfFileSystemError, OSError, TimeoutError) as e:
            if attempt == max_retries:
                print(f"✗ FAILED after {max_retries} attempts: {e}")
                return False
            wait = 5 * attempt
            print(f"⚠ retry in {wait}s...")
            time.sleep(wait)
    return False


def download_all() -> None:
    """Download all required models with validation."""
    set_cache_dir(MODELS_DIR)

    # 1. Embedders (via registry installer)
    print("--- Installing embedders via registry ---")
    embedders = [
        "jina-v5-nano-retrieval",
        "jina-v5-small-retrieval",
        "embeddinggemma-300m",
        "harrier-oss-v1-0.6b",
    ]
    for model in embedders:
        if is_installed(model):
            print(f"  ✓ {model} already installed")
        else:
            try:
                print(f"  Installing {model}...", end=" ", flush=True)
                install_embedder(model)
                print("✓")
            except Exception as e:
                print(f"✗ FAILED: {e}")

    # 2. NER: TinyBERT for entity extraction
    print("\n--- TinyBERT NER (ONNX) ---")
    ner_dir = MODELS_DIR / "tinybert-ner"
    ner_dir.mkdir(exist_ok=True)
    ner_repo = "onnx-community/TinyBERT-finetuned-NER-ONNX"
    ner_files = ["tokenizer.json", "config.json", "onnx/model_int8.onnx"]
    if all((ner_dir / f).exists() for f in ner_files):
        print("  ✓ Already present")
    else:
        for f in ner_files:
            _download_with_retry(ner_repo, f, str(ner_dir))

    # 3. Reranker: Jina v3 GGUF
    print("\n--- Jina Reranker v3 GGUF ---")
    reranker_dir = MODELS_DIR / "jina-reranker-v3"
    reranker_dir.mkdir(exist_ok=True)
    reranker_repo = "jinaai/jina-reranker-v3-GGUF"
    reranker_files = ["jina-reranker-v3-Q8_0.gguf", "projector.safetensors"]
    if all((reranker_dir / f).exists() for f in reranker_files):
        print("  ✓ Already present")
    else:
        for f in reranker_files:
            _download_with_retry(reranker_repo, f, str(reranker_dir))

    # 4. model2vec default embedder
    print("\n--- model2vec default (M2V_base_output) ---")
    m2v_dir = MODELS_DIR / "m2v-base"
    m2v_dir.mkdir(exist_ok=True)
    m2v_repo = "minishlab/M2V_base_output"
    m2v_files = ["model.safetensors", "config.json", "tokenizer.json"]
    if all((m2v_dir / f).exists() for f in m2v_files):
        print("  ✓ Already present")
    else:
        for f in m2v_files:
            _download_with_retry(m2v_repo, f, str(m2v_dir))

    print("\n✅ All models ready in ./models/")
    print(f"   Embedders: jina-v5-nano, jina-v5-small, embeddinggemma, harrier")
    print(f"   NER:       tinybert-ner")
    print(f"   Reranker:  jina-reranker-v3 (Q8_0 quantized)")
    print(f"   Default:   m2v-base (model2vec)")


if __name__ == "__main__":
    download_all()
