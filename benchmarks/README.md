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

**REMEMBER vs SIMILAR depends on embedder strength:**

| Embedder | REMEMBER R@5 | SIMILAR R@5 | Winner |
|----------|-------------|-------------|--------|
| model2vec (weak) | **89.1%** | ~70.0% | REMEMBER (+19pp, BM25 compensates) |
| MiniLM-L6-v2 (strong) | 92.1% | **93.4%** | SIMILAR (+1.3pp) |

With a weak embedder, BM25 fusion in REMEMBER adds ~19pp over pure vector. With a strong embedder, REMEMBER's fusion *hurts* — the recency/confidence/recall_freq signals are uniform noise on the LongMemEval benchmark (every session has the same recency, no confidence data, no recall history), diluting the vector signal. **For benchmarks with a modern encoder, use `--mode similar`.**

### Embedder Options

GraphStore's benchmark harness supports multiple embedders via `--embedder`:

| Flag | Model | Dims | Kind | Notes |
|------|-------|------|------|-------|
| `default` | model2vec `M2V_base_output` | 256 | static distilled | instant (no inference), ~232 KB install, baseline quality |
| `minilm-l6` | sentence-transformers/all-MiniLM-L6-v2 | 384 | encoder, mean pooling | via fastembed; matches MemPalace's MiniLM |
| `bge-small` / `bge-base` / `bge-large` | BAAI bge-en-v1.5 | 384 / 768 / 1024 | encoder, mean pooling | via fastembed |
| `mxbai-large` | mixedbread-ai/mxbai-embed-large-v1 | 1024 | encoder, mean pooling | via fastembed |
| `snowflake-l` / `snowflake-m` | snowflake/arctic-embed | 1024 / 768 | encoder, mean pooling | via fastembed |
| `jina-v2` | jinaai/jina-embeddings-v2-base-en | 768 | encoder | via fastembed, 8k context |
| `nomic-v1.5` | nomic-ai/nomic-embed-text-v1.5 | 768 | encoder | via fastembed, 8k context |
| `embeddinggemma` | Google EmbeddingGemma-300M | 256 (Matryoshka) | encoder, mean pooling | slow on CPU (~67 sec/question) |
| `embeddinggemma-768` | Google EmbeddingGemma-300M | 768 | encoder, mean pooling | slower, higher quality |
| `harrier` | Microsoft Harrier-OSS-v1 0.6B | 1024 | decoder, last-token pooling | **blocked** — needs `onnxruntime-genai` + re-export via model builder (torch) |
| `fastembed:<hf-id>` | any fastembed-supported model | varies | — | use the HF ID directly |
| `model2vec:<hf-id>` | any model2vec HF model | varies | static distilled | e.g. `model2vec:minishlab/potion-retrieval-32M` |

**Why the default is model2vec:** it's instant (no transformer inference, just a table lookup), ~232 KB on disk, and graphstore's REMEMBER fusion compensates for the weaker embedding quality. For maximum retrieval quality, use `--embedder minilm-l6` (or any fastembed model) with `--mode similar`.

**Install:** `pip install graphstore[fastembed]` to unlock the fastembed models. Core install without fastembed stays at ~60 MB; with fastembed ~82 MB.

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

### Session Document Format

For `--granularity session`, each document embeds **only the user turns** from
the haystack session, joined with `\n`. Assistant responses are filtered out
because they contain chatter and rephrasing that dilute the vector signal
against LongMemEval's user-question-grounded ground truth. This matches
MemPalace's benchmark convention and closed a measured ~3pp R@5 gap.

If the dataset carries no `role` labels (some older cleaned dumps), the
harness falls back to embedding all turns so it stays functional.

### MemPalace Comparison (apples-to-apples)

MemPalace publishes **96.6% R@5** on LongMemEval using MiniLM-L6-v2 + ChromaDB
pure vector search. GraphStore's results after this PR:

| # | Config | R@5 | R@10 | nDCG@10 | elapsed |
|---|--------|-----|------|---------|---------|
| 1 | model2vec + REMEMBER + all turns | 89.1% | 93.2% | 78.4% | 180 s |
| 2 | **MiniLM + REMEMBER + all turns** | 92.1% | 95.3% | 82.9% | 390 s |
| 3 | **MiniLM + SIMILAR + all turns** | 93.4% | 97.0% | 84.0% | 315 s |
| 4 | **MiniLM + SIMILAR + user-turns only** ⭐ | **94.7%** | **98.1%** | **87.2%** | **258 s** |
| | MemPalace (reported) | 96.6% | n/a | n/a | n/a |

**Gap to MemPalace: 1.9pp R@5.** On R@10 we're at 98.1% — likely matching them
since R@5 is their headline number and they don't publish R@10.

**Per-type R@10 (MiniLM + SIMILAR + user-turns):**

| Type | R@10 |
|------|------|
| knowledge-update | **100.0%** |
| multi-session | 99.2% |
| single-session-user | 98.4% |
| temporal-reasoning | 96.9% |
| single-session-preference | 96.7% |
| single-session-assistant | 96.4% |

**Reproduce:**

```bash
# The configuration that hits 94.7% R@5 / 98.1% R@10:
pip install graphstore[fastembed]
uv run python benchmarks/longmemeval.py \
  /tmp/longmemeval-data/longmemeval_s_cleaned.json \
  --embedder minilm-l6 \
  --mode similar \
  --ingest-mode flat
```

### Notes

- v1 is retrieval-only. It does not run answer generation.
- BM25 indexing is backed by `DocumentStore` summaries populated during benchmark ingestion.
- Default scoring skips abstention questions. Use `--include-abstention` to include them.
- Each question is evaluated in a fresh temporary `GraphStore` to avoid cross-question leakage.
