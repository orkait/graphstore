"""Kaggle script: Chroma-BM25 + Jina v5 Nano on LongMemEval-S (500 records)

Push to Kaggle with:
    kaggle kernels push -p benchmarks/kaggle/
"""

import subprocess, sys, os

# Install deps
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "graphstore", "chromadb>=0.5", "rank-bm25>=0.2.2", "onnxruntime-gpu>=1.23",
    "onnx>=1.14", "tokenizers>=0.20", "psutil>=5.9", "huggingface_hub",
])

# Download model + dataset
from huggingface_hub import snapshot_download

print("Downloading Jina v5 Nano FP16...")
model_dir = snapshot_download(
    "onnx-community/harrier-oss-v1-0.6b-ONNX",
    local_dir="/kaggle/working/jina-nano",
    allow_patterns=["onnx/model_fp16.onnx*", "tokenizer*", "config*"],
)

print("Downloading LongMemEval-S...")
data_dir = snapshot_download(
    "xiaowu0162/longmemeval-cleaned",
    local_dir="/kaggle/working/longmemeval-data",
)

# Clone graphstore for benchmark framework
print("Cloning graphstore...")
subprocess.check_call(["git", "clone", "--depth", "1",
    "https://github.com/orkait/graphstore.git", "/kaggle/working/graphstore"])

# Run benchmark
sys.path.insert(0, "/kaggle/working/graphstore")
os.environ["PYTHONPATH"] = "/kaggle/working/graphstore"

sys.argv = ["bench",
    "--system", "chroma-bm25",
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
    "--out-dir", "/kaggle/working/results",
    "--run-tag", "chroma-jina-500-kaggle",
]

from benchmarks.framework.docker_runner import main
sys.exit(main())
