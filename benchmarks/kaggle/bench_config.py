"""Shared benchmark config for Kaggle notebooks."""

# Embedder
EMBEDDER = "onnx"
EMBEDDER_MODEL_REPO = "jinaai/jina-embeddings-v5-text-nano-retrieval"
EMBEDDER_MODEL_PATTERNS = ["onnx/model_fp16.onnx*", "tokenizer*", "config*"]
EMBEDDER_POOLING = "mean"
EMBEDDER_MAX_LENGTH = 2048
EMBEDDER_OUTPUT_DIMS = 768

# Dataset
DATASET_REPO = "xiaowu0162/longmemeval-cleaned"
DATASET_VARIANT = "s"

# Hardware
GPU = True
GPU_MEM_LIMIT_GB = 12
EMBED_BATCH_SIZE = 256

# Auth
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Paths (Kaggle)
MODEL_DIR = "/kaggle/working/jina-nano"
DATA_DIR = "/kaggle/working/longmemeval-data"
RESULTS_DIR = "/kaggle/working/results"
GRAPHSTORE_DIR = "/kaggle/working/graphstore"

# Deps
PIP_DEPS = [
    "numpy>=1.24", "scipy>=1.10", "lark>=1.1", "usearch>=2.0",
    "model2vec>=0.4", "msgspec>=0.18", "croniter>=6.0", "orjson>=3.11.8",
    "fastembed>=0.8", "psutil>=5.9",
    "chromadb>=0.5", "rank-bm25>=0.2.2",
    "tokenizers>=0.20", "onnxruntime-gpu>=1.23", "onnx>=1.14",
    "huggingface_hub",
]
