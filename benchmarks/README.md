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

### Notes

- v1 is retrieval-only. It does not run answer generation.
- BM25 indexing is backed by `DocumentStore` summaries populated during benchmark ingestion.
- Default scoring skips abstention questions. Use `--include-abstention` to include them.
- Each question is evaluated in a fresh temporary `GraphStore` to avoid cross-question leakage.
