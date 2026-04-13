# GraphStore Benchmarks

Benchmark results, methodology, and reproduction instructions.

---

## LongMemEval-S

Retrieval-only benchmark. 500 records, each with ~53 sessions (~500 messages). Per-record isolated evaluation: reset -> ingest haystack -> query -> check if answer session ID in retrieved results.

### Results

**Docker (CPU, bge-small-en-v1.5, 384d):**

| Metric | Value |
|---|---|
| **Accuracy** | **97.6%** |
| Query p50 | 7.6 ms |
| Peak RSS | 2,352 MB |
| Hardware | Docker `--cpus=12 --memory=16g`, CPU only |
| Run date | 2026-04-10 |

**Kaggle (GPU, Jina v5 Nano, 768d):**

| Metric | Value |
|---|---|
| **Accuracy** | **96.4%** |
| Query p50 | 20 ms |
| Ingest p50 | 322 ms |
| Hardware | Kaggle T4 GPU |
| Run date | 2026-04-12 |

<details>
<summary>Per-category (Kaggle, Jina v5 Nano)</summary>

| Category | n | Accuracy |
|---|---|---|
| knowledge-update | 78 | 100.0% |
| single-session-assistant | 56 | 100.0% |
| single-session-user | 70 | 98.6% |
| multi-session | 133 | 98.5% |
| temporal-reasoning | 133 | 91.7% |
| single-session-preference | 30 | 86.7% |

</details>

### How to reproduce

**Docker:**
```bash
python -m benchmarks.framework.docker_runner \
  --embedder onnx \
  --embedder-model-dir /path/to/jina-nano \
  --data /path/to/longmemeval \
  --gpu
```

**Kaggle:**
```bash
# Update benchmarks/kaggle/graphstore_jina_500.py with your HF token
kaggle kernels push -p benchmarks/kaggle
```

---

## LoCoMo

End-to-end QA benchmark. 10 conversations, ~200 QAs each, 1986 total. Ingest all sessions for a conversation once, query all QAs against same state. Token-level F1 with Porter stemming (official protocol from snap-research/locomo).

### Results

**50Q random sample (conv-26, MiniMax M2.7 nitro, Jina v5 Small 1024d):**

| Category | n | F1 |
|---|---|---|
| open-domain | 10 | 0.452 |
| multi-hop | 10 | 0.418 |
| adversarial | 10 | 0.500 |
| single-hop | 10 | 0.224 |
| temporal | 10 | 0.189 |
| **Overall** | **50** | **0.357** |

For context: GPT-3.5-turbo-16k with full conversation context scores 0.378 on LoCoMo.

### Retrieval recall (no LLM)

How often the gold answer keyword appears in retrieved passages:

| K | Recall |
|---|---|
| top-5 | 60% |
| top-10 | 80% |
| top-20 | 84% |
| top-50 | 96% |

Measured on 50 validated questions where the keyword exists in the ingested data.

### How to reproduce

```bash
# Download locomo10.json from https://huggingface.co/datasets/Percena/locomo-mc10
# Place at /tmp/locomo/raw/locomo10.json

# Install jina-v5-small
GRAPHSTORE_MODEL_CACHE_DIR=/tmp/gs_models python -c "
from graphstore.registry.installer import install_embedder, set_cache_dir
set_cache_dir('/tmp/gs_models')
install_embedder('jina-v5-small-retrieval')
"

# Run full benchmark (requires LLM - set QA_MODEL in benchmarks/framework/llm_client.py)
python -m benchmarks.framework.run_locomo \
  --data-path /tmp/locomo \
  --embedder installed:jina-v5-small-retrieval \
  --k 10

# Run retrieval recall test (no LLM needed)
python -m benchmarks.framework.ratchet50
```

---

## Methodology notes

- LongMemEval scoring: substring match of any gold answer in retrieved text, OR answer session ID match in retrieved node metadata
- LoCoMo scoring: official token-level F1 with Porter stemming, Counter-based multiset overlap, per-category handling (multi-hop splits sub-answers, adversarial checks "no information available")
- The LLM reader (MiniMax M2.7) generates a short answer from retrieved context. F1 is computed between this answer and the gold answer. No LLM judge involved.
- MemMachine and other SOTA systems report LLM-judge accuracy (binary correct/wrong), which inflates ~1.5-2x compared to token F1. Numbers are not directly comparable.
- Our retrieval recall test uses keyword matching (does the longest distinctive word from the gold answer appear in the retrieved text?) which is conservative - the LLM might extract the answer even without an exact keyword match.
