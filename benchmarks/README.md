# GraphStore Benchmarks

## LongMemEval

LongMemEval is a retrieval benchmark for long-term conversational memory. This harness loads one LongMemEval dataset file, ingests each haystack into a fresh temporary `GraphStore`, retrieves candidate memories, and reports session and turn retrieval metrics.

### Download

```bash
mkdir -p /tmp/longmemeval-data
curl -fsSL -o /tmp/longmemeval-data/longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
```

### Quick Smoke

```bash
cd /mnt/storage/codespace/code/orkait/graphstore/graphstore
uv run python benchmarks/longmemeval.py /tmp/longmemeval-data/longmemeval_s_cleaned.json --mode lexical --limit 20
```

### Full Run

```bash
cd /mnt/storage/codespace/code/orkait/graphstore/graphstore
uv run python benchmarks/longmemeval.py /tmp/longmemeval-data/longmemeval_s_cleaned.json --mode remember --out benchmarks/results_longmemeval_remember.jsonl
```

### Ingest Modes

Two strategies for loading corpus items into graphstore:

- `flat` **(default)**: `CREATE NODE` with the full session text as one blob — one vector per session.
- `native`: `INGEST` pipeline — auto-chunks each session by paragraph, embeds each chunk individually, populates FTS properly.

**Full-run results (500 questions, 470 scored, REMEMBER mode):**

| Ingest Mode | R@5 | R@10 | nDCG@10 |
|-------------|-----|------|---------|
| flat        | **89.1%** | **93.2%** | **78.4%** |
| native      | 85.5% | 91.3% | 73.7% |

**Per-type R@10 breakdown:**

| Question Type | flat | native | Winner |
|---------------|------|--------|--------|
| single-session-user | 75.0% | **90.6%** | native (+15.6pp) |
| multi-session | **98.3%** | 95.9% | flat |
| single-session-preference | **80.0%** | 50.0% | flat (+30pp) |
| temporal-reasoning | **97.6%** | 92.9% | flat |
| knowledge-update | 94.4% | **100.0%** | native |
| single-session-assistant | **98.2%** | 89.3% | flat |

**Why flat is the default:** flat wins overall (93.2% vs 91.3% R@10) and on 4/6 question types. native excels only on `single-session-user` (+15.6pp) — questions about buried facts inside long mixed-topic conversations — but at the cost of a severe regression on `single-session-preference` (-30pp) and a -1.9pp overall loss.

Use native only if your workload is dominated by single-session long conversations where the answer is a specific buried fact.

```bash
# native — paragraph-chunked (better only for single-session-user type)
uv run python benchmarks/longmemeval.py data/longmemeval_s_cleaned.json --ingest-mode native --mode remember
```

### Retrieval Modes

- `remember`: 5-signal fusion (vector + BM25 + recency + confidence + recall_freq)
- `similar`: dense-only retrieval
- `lexical`: BM25-only retrieval
- `hybrid`: simple RRF fusion of similar + lexical

**Signal contribution (flat ingest, full run):**

| Mode | R@5 |
|------|-----|
| remember (5-signal fusion) | **89.1%** |
| similar (pure vector) | ~70.0% |

BM25 fusion in REMEMBER contributes approximately +19pp over pure vector retrieval with the default model2vec embedder.

### Embedder Options

GraphStore's benchmark harness supports multiple embedders via `--embedder`:

| Flag | Model | Dims | Kind | Status |
|------|-------|------|------|--------|
| `default` | model2vec `M2V_base_output` | 256 | static distilled | fast, low memory, recommended default |
| `embeddinggemma` | Google EmbeddingGemma-300M | 256 (Matryoshka) | encoder, mean pooling | slow on CPU (~67 sec/question, long sessions) |
| `embeddinggemma-768` | Google EmbeddingGemma-300M | 768 | encoder, mean pooling | slower, better quality |
| `harrier` | Microsoft Harrier-OSS-v1 0.6B | 1024 | decoder, last-token pooling | registered but **blocked** — ONNX export uses `com.microsoft.GroupQueryAttention` with 11 inputs, requires `onnxruntime-genai` |
| `model2vec:<hf-id>` | any model2vec HF model | varies | static distilled | e.g. `model2vec:minishlab/potion-retrieval-32M` |

**Why the default is model2vec:** it's instant (no transformer inference, just table lookup), matches MemPalace's MiniLM quality when combined with REMEMBER's BM25 fusion (89.1% vs 96.6% R@5), and doesn't need GPUs or large model downloads.

### Ingest Performance (batched embedding)

The `flat` ingest path uses `GraphStore.deferred_embeddings()` to batch vector
computations. Within the context, CREATE NODE queues `(slot, text)` pairs in a
pending buffer and flushes them in one `encode_documents()` call when either
the batch size is reached or the context exits. This is a ~1.5-2x speedup for
transformer embedders and also fixes a double-embedding bug where `CREATE NODE`
with both schema `EMBED` and `DOCUMENT` clauses previously embedded the same
text twice.

**Measured speedup (model2vec default, full 500-question run):**

| Version | elapsed | R@5 | R@10 |
|---------|---------|-----|------|
| pre-fix (double embed, unbatched) | 397 sec | 89.1% | 93.2% |
| post-fix (single embed, batched)  | **180 sec** | **89.1%** | **93.2%** |

Same scores, **2.2x faster**. The double-embedding fix alone accounts for most
of the model2vec speedup since model2vec has negligible per-call overhead.
Transformer embedders get an additional win from batching.

```python
with gs.deferred_embeddings(batch_size=64):
    for item in items:
        gs.execute(f'CREATE NODE "{item.id}" text = "{item.text}" DOCUMENT "{item.text}"')
# All embeddings flushed here.
```

For static embedders (model2vec) it's roughly neutral. For transformer
embedders (EmbeddingGemma, bge-*, e5-*, Harrier once runtime support lands)
it's the difference between viable and infeasible on CPU.

### Notes

- v1 is retrieval-only. It does not run answer generation.
- BM25 indexing is backed by `DocumentStore` summaries populated during benchmark ingestion.
- Default scoring skips abstention questions. Use `--include-abstention` to include them.
- Each question is evaluated in a fresh temporary `GraphStore` to avoid cross-question leakage.
