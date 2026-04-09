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

- `native` **(default)**: `INGEST` pipeline — auto-chunks each session by paragraph, embeds each chunk individually, populates FTS properly. This is how graphstore is designed to be used and reflects real production performance.
- `flat`: `CREATE NODE` with the full session text as one blob — one vector per session. Use this as a **baseline comparison** to measure the chunking delta, or when content is already atomic (short facts, structured records).

**Why native is the default:** flat mode stores entire conversations as a single vector (the session centroid). Facts buried inside long mixed-topic sessions get swamped by the dominant topic. On the LongMemEval `single-session-user` category, flat scores 59.4% vs native's 96.9% — a 37.5pp difference — because the answer is a buried fact inside an otherwise unrelated conversation. For topically coherent sessions, flat is marginally better (multi-session: 95.9% flat vs 94.2% native).

```bash
# flat — session-level baseline (for comparison only)
uv run python benchmarks/longmemeval.py data/longmemeval_s_cleaned.json --ingest-mode flat --mode remember --limit 20
```

### Retrieval Modes

- `remember`: 5-signal fusion (vector + BM25 + recency + confidence + recall_freq)
- `similar`: dense-only retrieval
- `lexical`: BM25-only retrieval
- `hybrid`: simple RRF fusion of similar + lexical

### Notes

- v1 is retrieval-only. It does not run answer generation.
- BM25 indexing is backed by `DocumentStore` summaries populated during benchmark ingestion.
- Default scoring skips abstention questions. Use `--include-abstention` to include them.
- Each question is evaluated in a fresh temporary `GraphStore` to avoid cross-question leakage.
