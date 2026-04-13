"""Kaggle: GraphStore + Jina v5 Nano on LongMemEval-S (500 records)

All config lives in bench_config.py - this script only orchestrates.
"""
import subprocess, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_config import (
    PIP_DEPS, EMBEDDER, EMBEDDER_MODEL_REPO, EMBEDDER_MODEL_PATTERNS,
    EMBEDDER_POOLING, EMBEDDER_MAX_LENGTH, EMBEDDER_OUTPUT_DIMS,
    DATASET_REPO, DATASET_VARIANT,
    GPU, GPU_MEM_LIMIT_GB, EMBED_BATCH_SIZE, HF_TOKEN,
    MODEL_DIR, DATA_DIR, RESULTS_DIR, GRAPHSTORE_DIR, GRAPHSTORE_REPO,
)

os.environ["HF_TOKEN"] = HF_TOKEN

# 1. Install dependencies
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + PIP_DEPS)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "--no-deps", "--force-reinstall", "onnxruntime-gpu>=1.23"])

# 2. Download model + dataset + graphstore source
from huggingface_hub import snapshot_download

print("Downloading embedder model...")
snapshot_download(EMBEDDER_MODEL_REPO, local_dir=MODEL_DIR,
    allow_patterns=EMBEDDER_MODEL_PATTERNS)

print("Downloading LongMemEval-S...")
snapshot_download(DATASET_REPO, repo_type="dataset", local_dir=DATA_DIR)

print("Cloning graphstore (latest main)...")
subprocess.check_call(["git", "clone", "--depth", "1", GRAPHSTORE_REPO, GRAPHSTORE_DIR])

# 3. Run benchmark from source
sys.path.insert(0, GRAPHSTORE_DIR)
sys.argv = ["bench",
    "--system", "graphstore",
    "--dataset", "longmemeval",
    "--data-path", DATA_DIR,
    "--variant", DATASET_VARIANT,
    "--embedder", EMBEDDER,
    "--embedder-model-dir", MODEL_DIR,
    "--embedder-pooling", EMBEDDER_POOLING,
    "--embedder-max-length", str(EMBEDDER_MAX_LENGTH),
    "--embedder-output-dims", str(EMBEDDER_OUTPUT_DIMS),
    *(["--gpu", "--gpu-mem-limit-gb", str(GPU_MEM_LIMIT_GB)] if GPU else []),
    "--embed-batch-size", str(EMBED_BATCH_SIZE),
    "--out-dir", RESULTS_DIR,
    "--run-tag", "graphstore-jina-500",
]
from benchmarks.framework.docker_runner import main
sys.exit(main())
