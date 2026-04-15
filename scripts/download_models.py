"""Download models for benchmarking into ./models."""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from graphstore.registry.installer import install_embedder, set_cache_dir, is_installed

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

def download_all():
    # 1. Embedders (via registry)
    set_cache_dir(MODELS_DIR)

    print("--- Checking embedders from registry ---")
    for model in ["jina-v5-nano-retrieval", "jina-v5-small-retrieval", "embeddinggemma-300m"]:
        if not is_installed(model):
            print(f"Installing {model}...")
            install_embedder(model)
        else:
            print(f"{model} is already installed.")

    # 2. NER Model (Manual)
    print("\n--- Checking TinyBERT NER model ---")
    ner_dir = MODELS_DIR / "tinybert-ner"
    ner_dir.mkdir(exist_ok=True)
    ner_repo = "onnx-community/TinyBERT-finetuned-NER-ONNX"
    if not (ner_dir / "onnx" / "model_int8.onnx").exists():
        for f in ["tokenizer.json", "config.json", "onnx/model_int8.onnx"]:
            print(f"Downloading {f}...")
            hf_hub_download(ner_repo, f, local_dir=str(ner_dir))
    else:
        print("TinyBERT NER model already exists.")

    # 3. Reranker (Manual)
    print("\n--- Checking Jina Reranker v3 GGUF ---")
    reranker_dir = MODELS_DIR / "jina-reranker-v3"
    reranker_dir.mkdir(exist_ok=True)
    reranker_repo = "jinaai/jina-reranker-v3-GGUF"
    if not (reranker_dir / "jina-reranker-v3-Q8_0.gguf").exists():
        for f in ["jina-reranker-v3-Q8_0.gguf", "projector.safetensors"]:
            print(f"Downloading {f}...")
            hf_hub_download(reranker_repo, f, local_dir=str(reranker_dir))
    else:
        print("Jina Reranker v3 GGUF already exists.")

    # 4. Model2Vec Default
    print("\n--- Checking model2vec default model ---")
    m2v_dir = MODELS_DIR / "m2v-base"
    m2v_dir.mkdir(exist_ok=True)
    m2v_repo = "minishlab/M2V_base_output"
    if not (m2v_dir / "model.safetensors").exists():
        for f in ["model.safetensors", "config.json", "tokenizer.json"]:
            try:
                print(f"Downloading {f}...")
                hf_hub_download(m2v_repo, f, local_dir=str(m2v_dir))
            except Exception:
                pass
    else:
        print("Model2Vec default model already exists.")

    print("\nAll models ready in ./models")

if __name__ == "__main__":
    download_all()
