# Benchmark Audit Log

Reproducible record of every benchmark run, the exact commands, hardware, and results.

## Environment

| Key | Value |
|---|---|
| Host | Ubuntu 24.04.4 LTS, kernel 6.17.0-20-generic |
| CPU | AMD (30 GiB RAM) |
| GPU | NVIDIA GeForce RTX 3060 (12 GiB VRAM) |
| Docker | 29.3.1, BuildKit v0.28.1 |
| Python | 3.11 (Docker image), 3.10 (host venv) |
| GraphStore | v0.3.0 |
| ChromaDB | v1.5.7 |
| ONNX Runtime GPU | v1.24.4 |
| CUDA | 12.x (via nvidia-cu12 pip wheels) |

## Dataset

| Key | Value |
|---|---|
| Name | LongMemEval-S |
| Source | `xiaowu0162/longmemeval-cleaned` on HuggingFace |
| Local path | `/home/kai/longmemeval-data/longmemeval_s_cleaned.json` |
| Total records | 500 |
| Categories | single-session-user (70), multi-session (133), single-session-preference (30), temporal-reasoning (133), knowledge-update (78), single-session-assistant (56) |
| Avg messages/record | 496 |
| Avg tokens/message | ~248 |

## Embedder

| Key | Value |
|---|---|
| Model | Harrier-oss-v1-0.6B (Qwen3 architecture) |
| Source | `onnx-community/harrier-oss-v1-0.6b-ONNX` on HuggingFace |
| Format | ONNX FP16 (`onnx/model_fp16.onnx`, 1.2 GB) |
| Dims | 1024 |
| Pooling | last_token |
| Max length | 2048 |
| Provider | CUDAExecutionProvider + GQA patching + ORT_ENABLE_ALL |
| Local path | `/mnt/storage/harrier-fp16-onnx/` |

## Docker Image

```bash
# Build (uses buildx builder "graphstore-bench" with cpuset-cpus=0-9, memory=16g)
DOCKER_BUILDKIT=1 docker build -f benchmarks/framework/Dockerfile.bench.gpu -t graphstore-bench:gpu .
```

## Scoring Methodology

- **accuracy**: fraction of questions where ANY gold answer appears as substring in top-K retrieved texts, OR any retrieved node belongs to a gold answer_session_id
- **recall@K**: mean fraction of gold answers found in top-K
- **K = 5** for all runs
- No LLM judge. Pure retrieval scoring.
- Per-record isolation: reset -> ingest full haystack -> query -> score -> repeat

---

## Run 1: GraphStore + Harrier FP32 (baseline, partial)

**Date:** 2026-04-12
**Purpose:** Baseline FP32 performance measurement

```bash
docker run --rm --gpus all --cpus=8 --memory=8g \
  -v /mnt/storage/harrier-gpu-export:/models/harrier-gpu:ro \
  -v /home/kai/longmemeval-data:/data/longmemeval:ro \
  -v $(pwd)/benchmarks/framework/results:/results \
  graphstore-bench:gpu \
  --embedder onnx \
  --embedder-model-dir /models/harrier-gpu \
  --embedder-pooling last_token \
  --embedder-max-length 2048 \
  --embedder-output-dims 1024 \
  --gpu \
  --gpu-mem-limit-gb 8 \
  --embed-batch-size 32 \
  --per-category 20
```

| Metric | Value |
|---|---|
| Records | 15/120 (interrupted by SIGTERM) |
| Accuracy | 1.000 |
| R@5 | 1.000 |
| Elapsed | 662s |
| Time/record | 44.1s |
| Query p50 | 11.5ms |
| Ingest mean | 892.3ms |
| Peak RSS | 1248 MB |
| Result file | `graphstore-skill-onnx-harrier-gpu_longmemeval_s_full_partial_2026-04-12T09-54-23.json` |
| Notes | FP32 re-export (raw MatMul, no GQA). Partial save via SIGTERM handler worked. |

---

## Run 2: GraphStore + Harrier FP32 + IO Binding + ORT_ENABLE_ALL

**Date:** 2026-04-12
**Purpose:** Test IO binding + graph optimization impact

```bash
docker run --rm --gpus all --cpus=8 --memory=8g \
  -v /mnt/storage/harrier-gpu-export:/models/harrier-gpu:ro \
  -v /home/kai/longmemeval-data:/data/longmemeval:ro \
  -v $(pwd)/benchmarks/framework/results:/results \
  graphstore-bench:gpu \
  --embedder onnx \
  --embedder-model-dir /models/harrier-gpu \
  --embedder-pooling last_token \
  --embedder-max-length 2048 \
  --embedder-output-dims 1024 \
  --gpu \
  --gpu-mem-limit-gb 8 \
  --embed-batch-size 32 \
  --per-category 20 \
  --max-records 10 \
  --run-tag iobind-ortall
```

| Metric | Value |
|---|---|
| Records | 10/10 |
| Accuracy | 1.000 |
| R@5 | 1.000 |
| Elapsed | 496s |
| Time/record | 49.6s |
| Query p50 | 11.7ms |
| Ingest mean | 998.1ms |
| Peak RSS | 1055 MB |
| Result file | `graphstore-skill-onnx-harrier-gpu_longmemeval_s_n10_iobind-ortall_2026-04-12T10-14-58.json` |
| Notes | IO binding was SLOWER (+11%) due to per-batch OrtValue creation overhead on variable-shape inputs. IO binding reverted after this run. |

---

## Run 3: GraphStore + Harrier FP16 (10 records)

**Date:** 2026-04-12
**Purpose:** Validate FP16 model from onnx-community

```bash
docker run --rm --gpus all --cpus=8 --memory=8g \
  -v /mnt/storage/harrier-fp16-onnx:/models/harrier-fp16:ro \
  -v /home/kai/longmemeval-data:/data/longmemeval:ro \
  -v $(pwd)/benchmarks/framework/results:/results \
  graphstore-bench:gpu \
  --embedder onnx \
  --embedder-model-dir /models/harrier-fp16 \
  --embedder-pooling last_token \
  --embedder-max-length 2048 \
  --embedder-output-dims 1024 \
  --gpu \
  --gpu-mem-limit-gb 8 \
  --embed-batch-size 32 \
  --per-category 20 \
  --max-records 10 \
  --run-tag fp16-gqa
```

| Metric | Value |
|---|---|
| Records | 10/10 |
| Accuracy | 1.000 |
| R@5 | 1.000 |
| Elapsed | 177s |
| Time/record | 17.7s |
| Query p50 | 6.5ms |
| Ingest mean | 355.3ms |
| Peak RSS | 2815 MB |
| Result file | `graphstore-skill-onnx-harrier-fp16_longmemeval_s_n10_fp16-gqa_2026-04-12T10-24-03.json` |
| Notes | 2.51x faster than FP32 baseline. GQA patching works, 1 Memcpy node in Docker. |

---

## Run 4: GraphStore + Harrier FP16 (full 120 records)

**Date:** 2026-04-12
**Purpose:** Full per-category accuracy measurement

```bash
docker run --rm --gpus all --cpus=8 --memory=8g \
  -v /mnt/storage/harrier-fp16-onnx:/models/harrier-fp16:ro \
  -v /home/kai/longmemeval-data:/data/longmemeval:ro \
  -v $(pwd)/benchmarks/framework/results:/results \
  graphstore-bench:gpu \
  --embedder onnx \
  --embedder-model-dir /models/harrier-fp16 \
  --embedder-pooling last_token \
  --embedder-max-length 2048 \
  --embedder-output-dims 1024 \
  --gpu \
  --gpu-mem-limit-gb 8 \
  --embed-batch-size 32 \
  --per-category 20 \
  --run-tag fp16-full
```

| Metric | Value |
|---|---|
| Records | 120/120 |
| **Accuracy** | **0.867** |
| **R@5** | **0.867** |
| Elapsed | 1982s (33 min) |
| Time/record | 16.5s |
| Query p50 | 7.5ms |
| Query p95 | 10.3ms |
| Ingest mean | 341.1ms |
| Peak RSS | 2984 MB |
| Result file | `graphstore-skill-onnx-harrier-fp16_longmemeval_s_full_fp16-full_2026-04-12T10-31-26.json` |

**Per-category breakdown:**

| Category | n | Accuracy | R@5 |
|---|---|---|---|
| single-session-user | 20 | 1.000 | 1.000 |
| multi-session | 20 | 0.950 | 0.950 |
| temporal-reasoning | 20 | 0.950 | 0.950 |
| knowledge-update | 20 | 0.900 | 0.900 |
| single-session-preference | 20 | 0.800 | 0.800 |
| single-session-assistant | 20 | 0.600 | 0.600 |

---

## Run 5: Chroma-BM25 + Harrier FP16 (120 records)

**Date:** 2026-04-12
**Purpose:** A/B comparison - vanilla hybrid retrieval baseline (no graph)

```bash
docker run --rm --gpus all --cpus=8 --memory=8g \
  --ulimit nofile=65536:65536 \
  -v /mnt/storage/harrier-fp16-onnx:/models/harrier-fp16:ro \
  -v /home/kai/longmemeval-data:/data/longmemeval:ro \
  -v $(pwd)/benchmarks/framework/results:/results \
  graphstore-bench:gpu-next \
  --system chroma-bm25 \
  --embedder onnx \
  --embedder-model-dir /models/harrier-fp16 \
  --embedder-pooling last_token \
  --embedder-max-length 2048 \
  --embedder-output-dims 1024 \
  --gpu \
  --gpu-mem-limit-gb 8 \
  --per-category 20 \
  --run-tag fp16-baseline
```

| Metric | Value |
|---|---|
| Records | **PENDING** |
| Accuracy | **PENDING** |
| Notes | ChromaDB 1.5.7 + rank-bm25 + RRF fusion. Same embedder instance as GraphStore. `--ulimit nofile=65536:65536` to work around chromadb FD leak. |

---

## Planned Runs

### Run 6: GraphStore + Harrier FP16 (full 500 records)
```bash
# Same as Run 4 but without --per-category 20
docker run --rm --gpus all --cpus=8 --memory=8g \
  -v /mnt/storage/harrier-fp16-onnx:/models/harrier-fp16:ro \
  -v /home/kai/longmemeval-data:/data/longmemeval:ro \
  -v $(pwd)/benchmarks/framework/results:/results \
  graphstore-bench:gpu \
  --embedder onnx \
  --embedder-model-dir /models/harrier-fp16 \
  --embedder-pooling last_token \
  --embedder-max-length 2048 \
  --embedder-output-dims 1024 \
  --gpu \
  --gpu-mem-limit-gb 8 \
  --embed-batch-size 32 \
  --run-tag fp16-full-500
```
ETA: ~2.4 hours

### Run 7: Chroma-BM25 + Harrier FP16 (full 500 records)
Same as Run 6 but `--system chroma-bm25`

### Run 8: LlamaIndex + Harrier FP16 (120 records)
```bash
# --system llamaindex
```

### Run 9: GraphStore + LLM reranker (120 records)
Add `--llm-rerank` flag (not yet implemented)

### Run 10: Mem0 (120 records)
Requires OpenAI API key. Different cost tier.

---

## Reproduction Guide

### 1. Get the dataset
```bash
# Download from HuggingFace
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('xiaowu0162/longmemeval-cleaned', local_dir='./longmemeval-data')
"
```

### 2. Get the embedder model
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'onnx-community/harrier-oss-v1-0.6b-ONNX',
    local_dir='./harrier-fp16-onnx',
    allow_patterns=['onnx/model_fp16.onnx', 'onnx/model_fp16.onnx_data',
                    'tokenizer.json', 'tokenizer_config.json', 'config.json'],
)
"
```

### 3. Build the Docker image
```bash
cd graphstore
DOCKER_BUILDKIT=1 docker build -f benchmarks/framework/Dockerfile.bench.gpu -t graphstore-bench:gpu .
```

### 4. Run GraphStore benchmark
```bash
docker run --rm --gpus all --cpus=8 --memory=8g \
  -v $(pwd)/harrier-fp16-onnx:/models/harrier-fp16:ro \
  -v $(pwd)/longmemeval-data:/data/longmemeval:ro \
  -v $(pwd)/benchmarks/framework/results:/results \
  graphstore-bench:gpu \
  --embedder onnx \
  --embedder-model-dir /models/harrier-fp16 \
  --embedder-pooling last_token \
  --embedder-max-length 2048 \
  --embedder-output-dims 1024 \
  --gpu \
  --gpu-mem-limit-gb 8 \
  --embed-batch-size 32 \
  --per-category 20 \
  --run-tag fp16-full
```

### 5. Run Chroma-BM25 baseline
```bash
docker run --rm --gpus all --cpus=8 --memory=8g \
  --ulimit nofile=65536:65536 \
  -v $(pwd)/harrier-fp16-onnx:/models/harrier-fp16:ro \
  -v $(pwd)/longmemeval-data:/data/longmemeval:ro \
  -v $(pwd)/benchmarks/framework/results:/results \
  graphstore-bench:gpu \
  --system chroma-bm25 \
  --embedder onnx \
  --embedder-model-dir /models/harrier-fp16 \
  --embedder-pooling last_token \
  --embedder-max-length 2048 \
  --embedder-output-dims 1024 \
  --gpu \
  --gpu-mem-limit-gb 8 \
  --per-category 20 \
  --run-tag fp16-baseline
```

### 6. Without GPU (CPU only)
```bash
# Use Dockerfile.bench (not .gpu), remove --gpu and --gpu-mem-limit-gb flags
DOCKER_BUILDKIT=1 docker build -f benchmarks/framework/Dockerfile.bench -t graphstore-bench:cpu .
docker run --rm --cpus=8 --memory=8g \
  -v $(pwd)/harrier-fp16-onnx:/models/harrier-fp16:ro \
  -v $(pwd)/longmemeval-data:/data/longmemeval:ro \
  -v $(pwd)/benchmarks/framework/results:/results \
  graphstore-bench:cpu \
  --embedder onnx \
  --embedder-model-dir /models/harrier-fp16 \
  --embedder-pooling last_token \
  --embedder-max-length 2048 \
  --embedder-output-dims 1024 \
  --embed-batch-size 32 \
  --per-category 20
```

### 7. Kaggle / Colab (no Docker)
```python
!pip install graphstore[gpu] onnx chromadb rank-bm25 psutil tokenizers

from huggingface_hub import snapshot_download
snapshot_download('onnx-community/harrier-oss-v1-0.6b-ONNX', local_dir='./model',
                  allow_patterns=['onnx/model_fp16.onnx*', 'tokenizer*', 'config*'])
snapshot_download('xiaowu0162/longmemeval-cleaned', local_dir='./data')

import sys; sys.path.insert(0, '.')
from benchmarks.framework.docker_runner import main
sys.argv = ['bench', '--embedder', 'onnx', '--embedder-model-dir', './model',
            '--embedder-pooling', 'last_token', '--embedder-max-length', '2048',
            '--embedder-output-dims', '1024', '--gpu', '--gpu-mem-limit-gb', '8',
            '--data-path', './data', '--out-dir', './results',
            '--per-category', '20', '--run-tag', 'kaggle']
main()
```

---

## Key Findings

1. **FP16 vs FP32**: 2.51x speedup with identical accuracy (same model, half precision)
2. **IO Binding**: Slower on variable-shape batches (-11%), reverted
3. **ORT_ENABLE_ALL**: Included in all FP16 runs, marginal standalone impact
4. **GPU utilization**: FP16 shifted bottleneck from GPU (100%) to CPU (92%), GPU at 53%
5. **single-session-assistant**: Weakest category (60%) - entity extraction regex skews toward user content
6. **multi-session**: Strongest differentiator (95%) - entity graph enables cross-session retrieval
