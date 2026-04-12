"""Kaggle: GraphStore + Jina v5 Nano on LongMemEval-S (500 records)"""
import subprocess, sys, os

HF_TOKEN = os.environ.get("HF_TOKEN", "")
EMBED_BATCH_SIZE = 128
os.environ["HF_TOKEN"] = HF_TOKEN

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "numpy>=1.24", "scipy>=1.10", "lark>=1.1", "usearch>=2.0",
    "model2vec>=0.4", "msgspec>=0.18", "croniter>=6.0", "orjson>=3.11.8",
    "fastembed>=0.8", "psutil>=5.9",
    "chromadb>=0.5", "rank-bm25>=0.2.2",
    "tokenizers>=0.20", "onnxruntime-gpu>=1.23", "onnx>=1.14",
    "huggingface_hub",
])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "--no-deps", "--force-reinstall", "onnxruntime-gpu>=1.23"])

from huggingface_hub import snapshot_download
print("Downloading Jina v5 Nano FP16...")
snapshot_download("jinaai/jina-embeddings-v5-text-nano-retrieval",
    local_dir="/kaggle/working/jina-nano",
    allow_patterns=["onnx/model_fp16.onnx*", "tokenizer*", "config*"])
print("Downloading LongMemEval-S...")
snapshot_download("xiaowu0162/longmemeval-cleaned",
    repo_type="dataset", local_dir="/kaggle/working/longmemeval-data")
print("Cloning graphstore...")
subprocess.check_call(["git", "clone", "--depth", "1",
    "https://github.com/orkait/graphstore.git", "/kaggle/working/graphstore"])

sys.path.insert(0, "/kaggle/working/graphstore")
sys.argv = ["bench",
    "--system", "graphstore",
    "--dataset", "longmemeval",
    "--data-path", "/kaggle/working/longmemeval-data",
    "--variant", "s",
    "--embedder", "onnx",
    "--embedder-model-dir", "/kaggle/working/jina-nano",
    "--embedder-pooling", "mean",
    "--embedder-max-length", "2048",
    "--embedder-output-dims", "768",
    "--gpu",
    "--gpu-mem-limit-gb", "12",
    "--embed-batch-size", str(EMBED_BATCH_SIZE),
    "--out-dir", "/kaggle/working/results",
    "--run-tag", "graphstore-jina-500",
]
from benchmarks.framework.docker_runner import main
sys.exit(main())
