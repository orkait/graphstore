"""Kaggle: GraphStore Pipeline Refactored - LongMemEval-S (500 records)

Branch: refactor/simplify-retrieval-pipeline
Features:
- Entity extraction on full chunk text (TinyBERT NER)
- Entity-aware graph signal (4-signal fusion)
- Recall count boost (0.05*log1p(count))
- Sentence query expansion
- jina-v5-nano-retrieval via GPU
"""
import subprocess, sys, os

# HF_TOKEN should be provided via Kaggle Secrets or environment variable
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
        if HF_TOKEN:
            os.environ["HF_TOKEN"] = HF_TOKEN
    except:
        pass

# Install deps
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "numpy>=1.24", "scipy>=1.10", "lark>=1.1", "usearch>=2.0",
    "model2vec>=0.4", "msgspec>=0.18", "croniter>=6.0", "orjson>=3.11.8",
    "psutil>=5.9",
    "tokenizers>=0.20", "onnxruntime-gpu>=1.23", "onnx>=1.14",
    "huggingface_hub",
])

# Download models
from huggingface_hub import snapshot_download, login
login(token=HF_TOKEN)

print("Downloading Jina v5 Nano FP16...")
snapshot_download("jinaai/jina-embeddings-v5-text-nano-retrieval",
    local_dir="/kaggle/working/jina-nano",
    allow_patterns=["onnx/model_fp16.onnx*", "tokenizer*", "config*"])

print("Downloading TinyBERT NER...")
snapshot_download("onnx-community/TinyBERT-finetuned-NER-ONNX",
    local_dir="/kaggle/working/tinybert-ner",
    allow_patterns=["onnx/model_int8.onnx", "tokenizer*", "config*"])

print("Downloading LongMemEval-S...")
snapshot_download("xiaowu0162/longmemeval-cleaned",
    repo_type="dataset", local_dir="/kaggle/working/longmemeval-data")

print("Cloning graphstore (refactor branch)...")
subprocess.check_call(["git", "clone", "--depth", "1", "--branch",
    "refactor/simplify-retrieval-pipeline",
    "https://github.com/orkait/graphstore.git", "/kaggle/working/graphstore"])

# GraphStore config - use defaults from config.py (no entity extraction for speed)
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
    "--gpu",
    "--gpu-mem-limit-gb", "12",
    "--embed-batch-size", "256",
    "--out-dir", "/kaggle/working/results",
    "--run-tag", "graphstore-pipeline-refactored",
]
from benchmarks.framework.docker_runner import main
sys.exit(main())
