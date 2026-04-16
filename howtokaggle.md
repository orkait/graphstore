# 🧩 How to Kaggle

Everything you need to push, run, monitor, and debug GraphStore benchmarks on Kaggle.

---

## 🔑 Authentication

Two auth methods. **KGAT is preferred** (new bearer token format).

### KGAT Token (new)
```bash
echo "KGAT_xxxxxxxxxxxx" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```
Used automatically by `kagglesdk` (`kernel_ctl.py`).

### Legacy API Key (fallback)
```bash
# ~/.kaggle/kaggle.json
{"username": "superkaiii", "key": "YOUR_API_KEY"}
chmod 600 ~/.kaggle/kaggle.json
```
Used by `kaggle` CLI (`kaggle kernels push`, `kaggle kernels status`).

> Both can coexist. KGAT is used by `kernel_ctl.py`, legacy key is used by `kaggle` CLI.

---

## ✅ Before Every Push - Validate Locally

Catches import errors, circular imports, missing models, and broken config before wasting Kaggle GPU time.

```bash
.venv/bin/python3 benchmarks/kaggle/validate_before_push.py
```

With mini 1-record benchmark run (slower, catches runtime errors too):
```bash
.venv/bin/python3 benchmarks/kaggle/validate_before_push.py
# (mini run is on by default, skip with --skip-run)
```

**What it checks:**
| Check | What |
|---|---|
| imports | graphstore, onnxruntime, huggingface_hub, docker_runner |
| circular | no circular import in graphstore package |
| models | jina-small + tinybert-ner structure valid |
| argv | all required flags present in sys.argv config |
| mini-run | 1 record through full pipeline (CPU, local models) |

---

## 🚀 Push a Kernel

```bash
# Push default kernel (graphstore-jina-v5-small)
kaggle kernels push -p benchmarks/kaggle/

# Pushed version is confirmed in output:
# Kernel version 16 successfully pushed.
```

Kernel metadata files in `benchmarks/kaggle/`:
| File | Kernel |
|---|---|
| `kernel-metadata.json` | `graphstore-jina-v5-small` (default) |
| `pipeline-kernel-metadata.json` | `graphstore-pipeline-refactored` |
| `rrf-kernel-metadata.json` | `graphstore-jina-500-rrf` |
| `q4-kernel-metadata.json` | `graphstore-jina-500-rerank-q4` |
| `q8-kernel-metadata.json` | `graphstore-jina-500-rerank-q8` |
| `gs-kernel-metadata.json` | `graphstore-jina-500` |

---

## 🎛 Kernel Control (`kernel_ctl.py`)

Full programmatic control via `kagglesdk` (KGAT auth, no CLI needed).

```bash
# Status
python3 benchmarks/kaggle/kernel_ctl.py status

# Logs (stdout + stderr)
python3 benchmarks/kaggle/kernel_ctl.py logs

# Cancel running kernel
python3 benchmarks/kaggle/kernel_ctl.py cancel

# Start kernel session
python3 benchmarks/kaggle/kernel_ctl.py run

# Different kernel
python3 benchmarks/kaggle/kernel_ctl.py logs --kernel graphstore-pipeline-refactored
```

---

## 📋 Monitor a Run

```bash
# Check status
python3 benchmarks/kaggle/kernel_ctl.py status
# KernelWorkerStatus.RUNNING / ERROR / COMPLETE / CANCEL_REQUESTED

# Get logs (available after ~30-60s into run)
python3 benchmarks/kaggle/kernel_ctl.py logs 2>&1 | tail -50

# Filter for errors only
python3 benchmarks/kaggle/kernel_ctl.py logs 2>&1 | grep "\[ERR\]"

# Watch for key milestones
python3 benchmarks/kaggle/kernel_ctl.py logs 2>&1 | grep -E "system:|config:|evaluating|interrupted|PASS|FAIL|ERROR"
```

---

## 🐛 Common Errors & Fixes

<details>
<summary><strong>ModuleNotFoundError: No module named 'graphstore'</strong></summary>

Subprocess doesn't inherit `sys.path`. Fix: pass `PYTHONPATH` explicitly.

```python
env = os.environ.copy()
env["PYTHONPATH"] = "/kaggle/working/graphstore"
subprocess.check_call([sys.executable, "scripts/some_script.py"],
    cwd="/kaggle/working/graphstore", env=env)
```

</details>

<details>
<summary><strong>ImportError: cannot import name 'GraphStore' from partially initialized module (circular import)</strong></summary>

Caused by absolute imports in `graphstore/graphstore/__init__.py`. Fixed by switching to relative imports.

`graphstore/graphstore/__init__.py` must use:
```python
from .graphstore import GraphStore   # relative - correct
from .core.store import CoreStore    # relative - correct
```

NOT:
```python
from graphstore.graphstore import GraphStore  # absolute - circular when on PYTHONPATH
```

</details>

<details>
<summary><strong>FileNotFoundError: tokenizer.json not found in models/tinybert-ner</strong></summary>

Two causes:

1. TinyBERT not downloaded yet - add to script:
```python
snapshot_download("onnx-community/TinyBERT-finetuned-NER-ONNX",
    local_dir="/kaggle/working/models/tinybert-ner", token=HF_TOKEN)
```

2. Path mismatch - `--entity-model-dir` must point to actual download location:
```python
sys.argv = [..., "--entity-model-dir", "/kaggle/working/models/tinybert-ner", ...]
```

The extractor checks both `{model_dir}/tokenizer.json` and `{model_dir}/onnx/tokenizer.json`.

</details>

<details>
<summary><strong>KernelWorkerStatus.ERROR with empty failure message</strong></summary>

Kaggle doesn't always populate `failure_message`. Get the real error from logs:

```bash
python3 benchmarks/kaggle/kernel_ctl.py logs 2>&1 | grep "\[ERR\]" | tail -20
```

</details>

<details>
<summary><strong>Push rejected / HF_TOKEN secrets issue</strong></summary>

Never hardcode tokens in scripts. Read from Kaggle Secrets or private dataset:

```python
# Option 1: Private dataset (preferred)
token_file = "/kaggle/input/hf-token-private/hf_token.txt"
if os.path.exists(token_file):
    with open(token_file) as f:
        HF_TOKEN = f.read().strip()

# Option 2: Kaggle Secrets (UI)
from kaggle_secrets import UserSecretsClient
HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
```

</details>

---

## 🏗 Typical Workflow

```bash
# 1. Make code changes

# 2. Validate locally
.venv/bin/python3 benchmarks/kaggle/validate_before_push.py

# 3. Commit + push branch
git add . && git commit -m "fix: ..."
git push origin refactor/simplify-retrieval-pipeline

# 4. Push kernel (clones latest branch at runtime)
kaggle kernels push -p benchmarks/kaggle/

# 5. Monitor
python3 benchmarks/kaggle/kernel_ctl.py status
python3 benchmarks/kaggle/kernel_ctl.py logs 2>&1 | tail -30

# 6. If failed - check logs, fix, repeat from step 1
python3 benchmarks/kaggle/kernel_ctl.py logs 2>&1 | grep "\[ERR\]"
```

---

## 📁 Key Files

| File | Purpose |
|---|---|
| `benchmarks/kaggle/graphstore_jina_500.py` | Main benchmark script (jina-small, 500 records) |
| `benchmarks/kaggle/graphstore_pipeline_refactored.py` | Pipeline refactor benchmark |
| `benchmarks/kaggle/kernel_ctl.py` | kagglesdk kernel control |
| `benchmarks/kaggle/validate_before_push.py` | Pre-push local validator |
| `benchmarks/kaggle/kernel-metadata.json` | Kaggle kernel config (maps script to kernel) |
| `benchmarks/graphstore.json` | DSL tuning config (loaded via `GRAPHSTORE_CONFIG`) |
| `~/.kaggle/access_token` | KGAT bearer token for kagglesdk |
| `~/.kaggle/kaggle.json` | Legacy API key for kaggle CLI |
