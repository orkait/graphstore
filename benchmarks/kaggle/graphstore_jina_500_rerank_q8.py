"""Kaggle: GraphStore + Jina v5 Nano + Jina v3 GGUF Reranker Q8 on LongMemEval-S (500 records)

Tuning config: benchmarks/graphstore.json (loaded via GRAPHSTORE_CONFIG env var)
Environment config: inlined below (Kaggle-specific paths, deps, GPU settings)
"""
import subprocess, sys, os

HF_TOKEN = os.environ.get("HF_TOKEN", "")
os.environ["HF_TOKEN"] = HF_TOKEN

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "numpy>=1.24", "scipy>=1.10", "lark>=1.1", "usearch>=2.0",
    "model2vec>=0.4", "msgspec>=0.18", "croniter>=6.0", "orjson>=3.11.8",
    "psutil>=5.9", "safetensors",
    "tokenizers>=0.20", "onnxruntime-gpu>=1.23", "onnx>=1.14",
    "huggingface_hub",
])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "--no-deps", "--force-reinstall", "onnxruntime-gpu>=1.23"])

print("Installing llama-cpp-python with CUDA...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "llama-cpp-python", "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu121"])

from huggingface_hub import snapshot_download, hf_hub_download
print("Downloading Jina v5 Nano FP16...")
snapshot_download("jinaai/jina-embeddings-v5-text-nano-retrieval",
    local_dir="/kaggle/working/jina-nano",
    allow_patterns=["onnx/model_fp16.onnx*", "tokenizer*", "config*"])

print("Downloading Jina v3 GGUF Reranker (Q8)...")
gguf_path = hf_hub_download(repo_id="jinaai/jina-reranker-v3-GGUF", filename="jina-reranker-v3-Q8_0.gguf", local_dir="/kaggle/working/jina-reranker")
projector_path = hf_hub_download(repo_id="jinaai/jina-reranker-v3-GGUF", filename="projector.safetensors", local_dir="/kaggle/working/jina-reranker")

print("Downloading LongMemEval-S...")
snapshot_download("xiaowu0162/longmemeval-cleaned",
    repo_type="dataset", local_dir="/kaggle/working/longmemeval-data")
print("Cloning graphstore (feat/sentence-level-embeddings)...")
subprocess.check_call(["git", "clone", "--depth", "1", "--branch", "feat/sentence-level-embeddings",
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
    "--embedder-model-dir", "/kaggle/working/jina-nano",
    "--embedder-pooling", "mean",
    "--embedder-max-length", "2048",
    "--embedder-output-dims", "768",
    "--reranker", "gguf",
    "--reranker-model-dir", gguf_path,
    "--reranker-projector-path", projector_path,
    "--gpu",
    "--gpu-mem-limit-gb", "12",
    "--embed-batch-size", "256",
    "--out-dir", "/kaggle/working/results",
    "--run-tag", "graphstore-jina-500-rerank-q8",
]
from benchmarks.framework.docker_runner import main
sys.exit(main())
