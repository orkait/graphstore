"""Kaggle: GraphStore + Jina v5 Small on LongMemEval-S (500 records)

Tuning config: benchmarks/graphstore.json (loaded via GRAPHSTORE_CONFIG env var)
Environment config: inlined below (Kaggle-specific paths, deps, GPU settings)
"""
import subprocess, sys, os

HF_TOKEN = os.environ.get("HF_TOKEN", "")
os.environ["HF_TOKEN"] = HF_TOKEN

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "numpy>=1.24", "scipy>=1.10", "lark>=1.1", "usearch>=2.0",
    "model2vec>=0.4", "msgspec>=0.18", "croniter>=6.0", "orjson>=3.11.8",
    "psutil>=5.9",
    "tokenizers>=0.20", "onnxruntime-gpu>=1.23", "onnx>=1.14",
    "huggingface_hub",
])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "--no-deps", "--force-reinstall", "onnxruntime-gpu>=1.23"])

from huggingface_hub import snapshot_download
print("Downloading Jina v5 Small ONNX...")
snapshot_download("jinaai/jina-embeddings-v5-text-small-retrieval",
    local_dir="/kaggle/working/jina-small",
    allow_patterns=["onnx/*.onnx*", "tokenizer*", "config*"])
print("Downloading LongMemEval-S...")
snapshot_download("xiaowu0162/longmemeval-cleaned",
    repo_type="dataset", local_dir="/kaggle/working/longmemeval-data")
print("Cloning graphstore (refactor/simplify-retrieval-pipeline)...")
subprocess.check_call(["git", "clone", "--depth", "1",
    "--branch", "refactor/simplify-retrieval-pipeline",
    "https://github.com/orkait/graphstore.git", "/kaggle/working/graphstore"])

# Point GraphStore at the benchmark config (single source of truth for tuning)
os.environ["GRAPHSTORE_CONFIG"] = "/kaggle/working/graphstore/benchmarks/graphstore.json"

sys.path.insert(0, "/kaggle/working/graphstore")
sys.argv = ["bench",
    "--system", "graphstore",
    "--dataset", "longmemeval",
    "--data-path", "/kaggle/working/longmemeval-data",
    "--variant", "s",
    "--embedder", "onnx",
    "--embedder-model-dir", "/kaggle/working/jina-small",
    "--embedder-pooling", "last_token",
    "--embedder-max-length", "2048",
    "--embedder-output-dims", "1024",
    "--gpu",
    "--gpu-mem-limit-gb", "12",
    "--embed-batch-size", "256",
    "--out-dir", "/kaggle/working/results",
    "--run-tag", "graphstore-jina-v5-small",
]
from benchmarks.framework.docker_runner import main
sys.exit(main())
