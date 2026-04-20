# Benchmark Framework

Benchmark GraphStore retrieval quality on three standardized datasets.

## Supported Benchmarks

| Benchmark | Protocol | Scoring | Runner |
|---|---|---|---|
| **LongMemEval** | Per-record: reset - ingest haystack - query - score | Accuracy, R@K, LLM judge | `runner.py` / `run_longmemeval.py` |
| **LoCoMo** | Per-conversation: ingest once - query all QAs | Token-level F1 (Porter stemming) | `run_locomo.py` |
| **BEAM** | Per-chat: ingest chunks - answer probing questions | External BEAM evaluator | `run_beam.py` |

## Quickstart

```bash
# List available benchmarks
python -m benchmarks.framework.cli list

# LongMemEval-S (500 records, per-record isolation)
python -m benchmarks.framework.cli run \
    --dataset longmemeval \
    --data-path ./data/longmemeval \
    --variant s \
    --k 5

# LoCoMo (10 conversations, 2000 QAs, F1 scoring)
python -m benchmarks.framework.cli run \
    --dataset locomo \
    --data-path ./data/locomo

# BEAM (chat-based probing questions, LLM answer generation)
python -m benchmarks.framework.cli run \
    --dataset beam \
    --data-path /tmp/BEAM \
    --variant 16k \
    --end-index 10 \
    --reader-model gpt-4.1-mini \
    --reader-url https://api.openai.com/v1
```

## Specialized Runners

Each benchmark also has a standalone runner for finer control:

```bash
# LongMemEval with native INGEST pipeline, hybrid retrieval, per-type NDCG
python -m benchmarks.framework.runners.longmemeval \
    data/longmemeval_s_cleaned.json \
    --mode remember --granularity session --top-k 10

# LoCoMo with async LLM batch scoring
python -m benchmarks.framework.runners.locomo \
    --data-path ./data/locomo \
    --embedder installed:jina-v5-small-retrieval \
    --k 10

# BEAM answer generation
python -m benchmarks.framework.runners.beam \
    --beam-root /tmp/BEAM \
    --chat-size 100K \
    --start-index 1 --end-index 3 \
    --embedder installed:jina-v5-small-retrieval \
    --reader-model-name gpt-4.1-mini
```

## What Gets Measured

```
quality         accuracy, recall@K, F1, NDCG (per benchmark)
latency_ingest  p50, p95, p99, mean, stddev
latency_query   p50, p95, p99, mean, stddev
memory          rss_before, rss_after, rss_peak, delta (MB)
cost            ingest_tokens, query_tokens
```

## File Layout

```
framework/
  cli.py                        # Unified CLI (all 3 benchmarks)
  runner.py                     # Generic per-record runner (LongMemEval)
  run_locomo.py                 # LoCoMo protocol (ingest-once, F1 + LLM judge)
  run_beam.py                   # BEAM protocol (chunk + answer generation)
  run_longmemeval.py            # LongMemEval native runner (NDCG, per-type)
  adapter.py                    # MemoryAdapter protocol
  adapters/graphstore_.py       # Native-DSL adapter (5-signal REMEMBER)
  adapters/graphstore_skill.py  # Skill-based ingest adapter (LLM-planned DSL)
  datasets.py                   # Dataset loaders (longmemeval, locomo)
  metrics.py                    # Quality, latency, memory metrics
  report.py                     # JSON, CSV, Markdown output
  entity_extraction.py          # NER for graph enrichment (used by graphstore_.py)
  ratchet_recall.py             # LoCoMo evidence-recall metrics
  ratchet_test.py               # Ratchet test harness (50Q random 10/cat)
  llm_runner.py                 # Shared LLM transport: rate-limit + retry + fallback
  llm_client.py                 # LoCoMo reader/judge wrappers (delegates to llm_runner)
  llm_judge.py                  # LongMemEval per-category judge prompts
  docker_runner.py              # Docker entry point
  Dockerfile.bench              # CPU container
  Dockerfile.bench.gpu          # GPU container
```

## Docker

```bash
# Build
docker build -f benchmarks/framework/Dockerfile.bench.gpu -t graphstore-bench:gpu .

# Run LongMemEval-S
docker run --cpus=8 --memory=16g --gpus all \
    -v ./data:/data:ro -v ./results:/results \
    graphstore-bench:gpu \
    --dataset longmemeval --variant s \
    --embedder installed --embedder-model jina-v5-small-retrieval \
    --gpu --k 5
```

## References

- [LongMemEval](https://github.com/xiaowu0162/LongMemEval)
- [LoCoMo](https://snap-research.github.io/locomo/)
- [BEAM](https://github.com/stanford-crfm/BEAM)
