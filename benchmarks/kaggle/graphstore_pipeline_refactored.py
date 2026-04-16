"""Kaggle: GraphStore Pipeline Refactored - LongMemEval-S (500 records)

Branch: refactor/simplify-retrieval-pipeline
Features:
- Entity extraction on full chunk text (TinyBERT NER)
- Entity-aware graph signal (4-signal fusion)
- Recall count boost (0.05*log1p(count))
- Sentence query expansion
- jina-v5-nano-retrieval via GPU
"""
import subprocess, sys, os, time
from pathlib import Path

def validate_file_exists(path, name):
    """Validate a file/directory exists, exit with helpful message if not."""
    if not Path(path).exists():
        print(f"ERROR: {name} not found at {path}")
        sys.exit(1)
    print(f"✓ {name} found")

def validate_dir_contents(path, required_files, name):
    """Validate directory contains expected files."""
    path = Path(path)
    missing = [f for f in required_files if not (path / f).exists()]
    if missing:
        print(f"ERROR: {name} incomplete. Missing: {', '.join(missing)}")
        print(f"       Path: {path}")
        print(f"       Contents: {list(path.glob('**/*'))[:5]}...")
        sys.exit(1)
    print(f"✓ {name} valid")

def run_with_retry(cmd, description, max_retries=3):
    """Run command with retry logic for transient failures."""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"{description}... (attempt {attempt}/{max_retries})")
            subprocess.check_call(cmd, timeout=300)
            return True
        except subprocess.TimeoutExpired:
            if attempt == max_retries:
                print(f"ERROR: {description} timed out after {max_retries} attempts")
                sys.exit(1)
            wait = 5 * attempt
            print(f"  Timeout, retrying in {wait}s...")
            time.sleep(wait)
        except subprocess.CalledProcessError as e:
            if attempt == max_retries:
                print(f"ERROR: {description} failed: {e}")
                sys.exit(1)
            wait = 5 * attempt
            print(f"  Failed, retrying in {wait}s...")
            time.sleep(wait)

# Step 1: Get and validate HF_TOKEN
print("\n=== Step 1: Validating HuggingFace Token ===")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
        if HF_TOKEN:
            os.environ["HF_TOKEN"] = HF_TOKEN
            print(f"✓ HF_TOKEN found via Kaggle Secrets: {HF_TOKEN[:8]}...")
    except Exception as e:
        print(f"Note: Could not access Kaggle Secrets: {e}")

if not HF_TOKEN:
    print("ERROR: HF_TOKEN not found in environment or Kaggle Secrets")
    print("       Set it via Kaggle Secrets (recommended) or HF_TOKEN env var")
    sys.exit(1)

# Step 2: Install dependencies
print("\n=== Step 2: Installing Dependencies ===")
run_with_retry(
    [sys.executable, "-m", "pip", "install", "-q",
     "numpy>=1.24", "scipy>=1.10", "lark>=1.1", "usearch>=2.0",
     "model2vec>=0.4", "msgspec>=0.18", "croniter>=6.0", "orjson>=3.11.8",
     "psutil>=5.9", "tokenizers>=0.20", "onnxruntime-gpu>=1.23", "onnx>=1.14",
     "huggingface_hub"],
    "Installing pip dependencies"
)

# Step 3: Login to HuggingFace
print("\n=== Step 3: Authenticating with HuggingFace ===")
from huggingface_hub import snapshot_download, login
try:
    login(token=HF_TOKEN)
    print("✓ HuggingFace authentication successful")
except Exception as e:
    print(f"ERROR: HuggingFace login failed: {e}")
    sys.exit(1)

# Step 4: Download Jina embedder
print("\n=== Step 4: Downloading Jina v5 Nano FP16 (1.2 GB) ===")
try:
    snapshot_download("jinaai/jina-embeddings-v5-text-nano-retrieval",
        local_dir="/kaggle/working/jina-nano",
        allow_patterns=["onnx/model_fp16.onnx*", "tokenizer*", "config*"])
    validate_dir_contents("/kaggle/working/jina-nano",
        ["onnx/model_fp16.onnx", "tokenizer.json", "config.json"],
        "Jina model")
except Exception as e:
    print(f"ERROR: Jina download failed: {e}")
    sys.exit(1)

# Step 5: Download TinyBERT NER model
print("\n=== Step 5: Downloading TinyBERT NER (with validation) ===")
try:
    # Download the full model to ensure we get tokenizer
    snapshot_download("onnx-community/TinyBERT-finetuned-NER-ONNX",
        local_dir="/kaggle/working/tinybert-ner")

    # Validate structure - tokenizer can be in root or onnx/ folder
    tinybert_path = Path("/kaggle/working/tinybert-ner")
    tokenizer_locations = [
        tinybert_path / "tokenizer.json",
        tinybert_path / "onnx" / "tokenizer.json"
    ]
    has_tokenizer = any(p.exists() for p in tokenizer_locations)

    if not has_tokenizer:
        print("ERROR: TinyBERT tokenizer.json not found in expected locations:")
        for loc in tokenizer_locations:
            print(f"       - {loc}")
        print(f"       Actual contents: {list(tinybert_path.glob('**/*'))[:10]}...")
        sys.exit(1)

    # Validate ONNX model exists
    onnx_locations = [
        tinybert_path / "onnx" / "model_int8.onnx",
        tinybert_path / "model_int8.onnx"
    ]
    has_onnx = any(p.exists() for p in onnx_locations)

    if not has_onnx:
        print("ERROR: TinyBERT ONNX model not found in expected locations:")
        for loc in onnx_locations:
            print(f"       - {loc}")
        sys.exit(1)

    print("✓ TinyBERT NER model downloaded and validated")
except Exception as e:
    print(f"ERROR: TinyBERT download failed: {e}")
    sys.exit(1)

# Step 6: Download dataset
print("\n=== Step 6: Downloading LongMemEval-S Dataset (500 records) ===")
try:
    snapshot_download("xiaowu0162/longmemeval-cleaned",
        repo_type="dataset", local_dir="/kaggle/working/longmemeval-data")
    validate_file_exists("/kaggle/working/longmemeval-data/longmemeval_s_cleaned.json",
        "LongMemEval dataset")
except Exception as e:
    print(f"ERROR: Dataset download failed: {e}")
    sys.exit(1)

# Step 7: Clone GraphStore repository
print("\n=== Step 7: Cloning GraphStore Repository ===")
run_with_retry(
    ["git", "clone", "--depth", "1", "--branch",
     "refactor/simplify-retrieval-pipeline",
     "https://github.com/orkait/graphstore.git", "/kaggle/working/graphstore"],
    "Cloning graphstore refactor branch",
    max_retries=2
)
validate_file_exists("/kaggle/working/graphstore/benchmarks/graphstore.json",
    "GraphStore config")

# Step 8: Validate TinyBERT exists at download location
print("\n=== Step 8: Validating TinyBERT NER Model ===")
tinybert_path = Path("/kaggle/working/tinybert-ner")
tokenizer_locations = [
    tinybert_path / "tokenizer.json",
    tinybert_path / "onnx" / "tokenizer.json"
]
has_tokenizer = any(p.exists() for p in tokenizer_locations)
if not has_tokenizer:
    print("ERROR: TinyBERT tokenizer.json missing after download")
    print(f"       Checked: {[str(p) for p in tokenizer_locations]}")
    print(f"       Contents: {list(tinybert_path.glob('**/*'))[:10]}...")
    sys.exit(1)
print("✓ TinyBERT model verified")

# Step 9: Validate benchmark config
print("\n=== Step 9: Validating Benchmark Configuration ===")
config_path = "/kaggle/working/graphstore/benchmarks/graphstore.json"
validate_file_exists(config_path, "Benchmark configuration")
os.environ["GRAPHSTORE_CONFIG"] = config_path

# Step 10: Prepare benchmark environment
print("\n=== Step 10: Preparing Benchmark Environment ===")
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
    "--entity-model-dir", "/kaggle/working/tinybert-ner",
    "--gpu",
    "--gpu-mem-limit-gb", "12",
    "--embed-batch-size", "256",
    "--out-dir", "/kaggle/working/results",
    "--run-tag", "graphstore-pipeline-refactored",
]

print(f"✓ All validations passed")
print(f"✓ Environment ready - starting benchmark (this may take 8-10 hours)")
print()

# Step 10: Run benchmark
try:
    from benchmarks.framework.docker_runner import main
    sys.exit(main())
except MemoryError as e:
    print(f"\nERROR: Out of GPU memory: {e}")
    print("       Try reducing --embed-batch-size or --gpu-mem-limit-gb")
    sys.exit(1)
except Exception as e:
    print(f"\nERROR: Benchmark failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
