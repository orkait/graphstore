"""Reference config for Kaggle benchmark environment.

NOT imported by Kaggle scripts (Kaggle copies scripts to /kaggle/src/,
breaking sibling imports). Values are inlined in each kernel script.

GraphStore tuning config lives in: benchmarks/graphstore.json
(loaded via GRAPHSTORE_CONFIG env var - single source of truth).

This file exists as documentation for what the Kaggle scripts use.
"""

# Embedder (Jina v5 Nano FP16 ONNX - 768d, 239M params)
EMBEDDER = "onnx"
EMBEDDER_MODEL_REPO = "jinaai/jina-embeddings-v5-text-nano-retrieval"
EMBEDDER_MODEL_PATTERNS = ["onnx/model_fp16.onnx*", "tokenizer*", "config*"]
EMBEDDER_POOLING = "mean"
EMBEDDER_MAX_LENGTH = 2048
EMBEDDER_OUTPUT_DIMS = 768

# Dataset
DATASET_REPO = "xiaowu0162/longmemeval-cleaned"
DATASET_VARIANT = "s"

# Hardware (Kaggle T4 GPU)
GPU = True
GPU_MEM_LIMIT_GB = 12
EMBED_BATCH_SIZE = 256

# Paths
MODEL_DIR = "/kaggle/working/jina-nano"
DATA_DIR = "/kaggle/working/longmemeval-data"
RESULTS_DIR = "/kaggle/working/results"
GRAPHSTORE_DIR = "/kaggle/working/graphstore"
GRAPHSTORE_REPO = "https://github.com/orkait/graphstore.git"
GRAPHSTORE_CONFIG = f"{GRAPHSTORE_DIR}/benchmarks/graphstore.json"

# Runtime deps (graphstore is cloned from source)
PIP_DEPS = [
    "numpy>=1.24", "scipy>=1.10", "lark>=1.1", "usearch>=2.0",
    "model2vec>=0.4", "msgspec>=0.18", "croniter>=6.0", "orjson>=3.11.8",
    "psutil>=5.9",
    "tokenizers>=0.20", "onnxruntime-gpu>=1.23", "onnx>=1.14",
    "huggingface_hub",
]
