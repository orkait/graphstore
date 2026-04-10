# Benchmarking graphstore

How we measure graphstore against agent memory rivals, what we've actually run, and how to reproduce it.

## Goal

Prove graphstore works as an **agent memory system** by running it on the standard 2026 benchmarks under conditions every competitor has to match. No vendor cherry-picking, no mystery hardware, no shifting baselines.

## What we benchmark against

LongMemEval is the headline benchmark for agent memory in 2026. Every serious system publishes a number against it. We start there.

| Benchmark | Status | Records | Notes |
|---|---|---|---|
| LongMemEval-S | live | 500 | 5 categories, ~115k tokens / record |
| LoCoMo | planned | - | Very long conversations, 35+ sessions |
| Agent Memory Benchmark (AMB) | planned | - | Vectorize's apples-to-apples framework |
| BEIR subset | planned | - | Pure retrieval control |

LongMemEval-S splits its 500 questions into six categories that probe different memory abilities:

| Category | n | What it tests |
|---|---|---|
| `single-session-user` | 70 | Pulling a fact from one conversation |
| `single-session-preference` | 30 | Pulling a stated preference |
| `single-session-assistant` | 56 | Reasoning about an assistant turn |
| `multi-session` | 133 | Stitching facts across multiple sessions |
| `temporal-reasoning` | 133 | Ordering, recency, before/after |
| `knowledge-update` | 78 | A fact was revised mid-conversation, use the new one |

Quality on multi-session and temporal-reasoning is where most systems collapse. Those are the ones to watch.

## Hardware: 4 CPU / 4 GB Docker, no GPU

Every benchmark run lives in a Docker container with `--cpus=4 --memory=4g`. This matches CLAUDE.md Rule 12 and gives a reproducible footprint anyone can match.

GPU is not used. We tested it - the embedder is the only GPU-friendly path and even then bge-small on 4 CPU clears the entire 500-record benchmark in well under an hour. The reproducibility win is worth more than the speedup, and Mem0 / Letta / cloud systems run CPU-default in apples-to-apples mode anyway.

## Dataset location

LongMemEval-S lives at `~/graphstore-models/longmemeval/longmemeval_s_cleaned.json` (~265 MB). The repo has a `models/` symlink pointing at `~/graphstore-models` so anywhere in the codebase you can refer to `models/longmemeval/` and it works on any contributor's machine. The `models/` symlink target is in `.gitignore` and never committed.

To download it from scratch:

```bash
mkdir -p ~/graphstore-models/longmemeval
cd ~/graphstore-models/longmemeval
curl -L -o longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
```

The cleaned variant is the one published in 2025/09 after an answer-leakage cleanup. Use it, not the older one.

Other benchmarks should land in the same `~/graphstore-models/<dataset>/` layout so the docker container only needs one bind mount.

## Running a benchmark in Docker

Build the image once:

```bash
docker build -f benchmarks/framework/Dockerfile.bench -t graphstore-bench:latest .
```

Then run any benchmark with the same shape:

```bash
docker run --rm \
  --cpus=4 --memory=4g \
  -v ~/graphstore-models:/data:ro \
  -v ~/graphstore-models/cache:/cache \
  -v "$(pwd)/benchmarks/framework/results:/results" \
  graphstore-bench:latest \
  --embedder fastembed \
  --embedder-model BAAI/bge-small-en-v1.5 \
  --max-records 500 \
  --run-tag full
```

Mounts:

- `/data` - read-only dataset (LongMemEval, LoCoMo, etc.)
- `/cache` - persistent embedder weights cache (fastembed downloads bge-small once, reuses forever)
- `/results` - JSON / CSV / Markdown output written here

CLI flags worth knowing:

| Flag | Default | Purpose |
|---|---|---|
| `--system` | `graphstore` | Which adapter to run |
| `--dataset` | `longmemeval` | Which loader to use |
| `--variant` | `s` | s / m / l for LongMemEval |
| `--embedder` | `fastembed` | `model2vec` (default fast) or `fastembed` (BGE family) |
| `--embedder-model` | `BAAI/bge-small-en-v1.5` | Any fastembed-supported model id |
| `--max-records` | all | Cap for fast iteration |
| `--ceiling-mb` | `3072` | Graphstore memory ceiling, leaves headroom inside the 4 GB container |
| `--run-tag` | `""` | Suffix on the result filename for clarity |

## How the framework runs LongMemEval

LongMemEval uses **per-record isolated evaluation**. Each of the 500 records carries its own ~500-message haystack across ~53 sessions, and the answer-bearing session is buried inside that haystack. The framework follows this protocol literally:

```
for each record:
    adapter.reset()                 # full db flush
    for session in record.haystack: # 53 sessions
        adapter.ingest(session)
    answer = adapter.query(record.question, k=5)
    score(answer, record.gold)
```

Critical: `reset()` removes the entire tmpdir so each record starts with an empty graphstore. No state leaks between records.

## Why graphstore should win

Every system in this race is a memory layer for agents. Most do one thing well. Graphstore does several:

1. **Hybrid retrieval out of the box.** `REMEMBER` fuses 5 signals: vector similarity, BM25, recency, confidence, and recall frequency. No competitor ships this as the default query primitive.
2. **Schema-first ingestion.** `SYS REGISTER NODE KIND ... EMBED content` tells graphstore which field to auto-embed and pre-allocates typed columns. The benchmark adapter uses this so CREATE doesn't infer column types lazily on first write.
3. **Deferred embeddings.** The adapter wraps every session ingest in `gs.deferred_embeddings(batch_size=128)`, which batches the embedder call across all messages in a session. On transformer embedders this is a 4-10x speedup.
4. **Numpy importance scoring.** Per-message importance is computed in a single vectorized pass over content lengths and fed into REMEMBER's confidence signal.

Run mode used for the leaderboard: schema + fastembed + deferred + REMEMBER.

## Results

All result files live in `benchmarks/framework/results/`. Each run produces three artifacts with the same prefix:

- `*.json` - structured result, full config snapshot
- `*.csv` - flat row for spreadsheets
- `*.md` - human-readable leaderboard for that run

Result files are gitignored. Re-run anything to regenerate.

### Run log

| Date | System | Embedder | n | Acc | R@5 | p50 ms | p95 ms | Peak MB | Elapsed |
|---|---|---|---|---|---|---|---|---|---|
| _populated as runs complete_ | | | | | | | | | |

## Reproduction checklist

- [ ] Docker installed (no GPU runtime needed)
- [ ] LongMemEval-S downloaded to `~/graphstore-models/longmemeval/longmemeval_s_cleaned.json`
- [ ] `models/` symlink at repo root points to `~/graphstore-models`
- [ ] `docker build -f benchmarks/framework/Dockerfile.bench -t graphstore-bench:latest .`
- [ ] First run: `--max-records 5 --run-tag smoke` to verify the stack
- [ ] Real run: drop `--max-records` for the full 500 record evaluation
- [ ] Result files appear in `benchmarks/framework/results/` on the host

## Known issues

- The fastembed embedder downloads ~100 MB of weights on the first run and caches them in `/cache/fastembed`. Mount this volume on every run so subsequent runs are instant.
- Docker Desktop on Linux only shares paths under `/home/$USER` by default. Keep dataset files there, not under `/mnt` or `/var`.
- Peak RSS is measured by the runner process via `psutil`. The container's actual peak RSS (visible via `docker stats`) is slightly higher because of the Python interpreter overhead.

## Roadmap

- [ ] Mem0 adapter, run side-by-side
- [ ] Letta adapter, run side-by-side
- [ ] Cognee, LightRAG, Graphiti adapters
- [ ] LoCoMo loader
- [ ] AMB loader
- [ ] LLM-as-judge scorer for second-opinion quality signal
- [ ] Multi-run driver (5x each system, median + stddev)
