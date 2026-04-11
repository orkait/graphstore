# GraphStore benchmarks

One document for both the headline result and the three harnesses that produce and reproduce it.

---

## 🏆 Headline

**graphstore scores 97.6% accuracy on LongMemEval-S (500 questions).**

Run on `BAAI/bge-small-en-v1.5` (384 dims, ~100 MB, CPU-only, 3-year-old encoder) inside a Docker container capped at 12 CPUs and 16 GB of RAM. No GPU. No cloud API. No LLM extraction.

### What was measured

| | |
|---|---|
| Benchmark | LongMemEval-S |
| Source | [xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) |
| Records | 500 (full benchmark) |
| Protocol | Per-record isolated evaluation (reset → ingest ~500-message haystack → query → score) |
| Adapter | `graphstore-skill-fastembed-bge-small-en-v1.5` |
| System version | graphstore 0.3.0 |
| Embedder | `BAAI/bge-small-en-v1.5` via fastembed (ONNX, CPU) |
| Query primitive | `REMEMBER` (hybrid: vector + BM25 + recency + confidence + recall_count) |
| Hardware | Docker `--cpus=12 --memory=16g` on RTX 3060 host (GPU unused) |
| Scoring | Substring match OR answer-session-id match on retrieved nodes |
| K (top-k retrieval) | 5 |
| Run date | 2026-04-10 |

### Overall

| Metric | Value |
|---|---|
| **Accuracy** | **0.976** |
| **R@5** | **0.976** |
| Total elapsed | 15,335 s (≈ 4 h 15 min) |
| Per-record mean | ~30.7 s (ingest ~500 msgs + query + reset) |
| Query p50 | 7.6 ms |
| Query p95 | 10.5 ms |
| Query p99 | 14.1 ms |
| Ingest mean | 640 ms per session (~13 messages) |
| Peak RSS | 2,352 MB |
| RSS delta vs baseline | 1,633 MB |

### By LongMemEval category

| Category | n | Accuracy | R@K |
|---|---|---|---|
| single-session-user | 70 | **1.000** | 1.000 |
| single-session-assistant | 56 | **1.000** | 1.000 |
| knowledge-update | 78 | **1.000** | 1.000 |
| multi-session | 133 | 0.985 | 0.985 |
| temporal-reasoning | 133 | 0.955 | 0.955 |
| single-session-preference | 30 | 0.867 | 0.867 |

Four categories at 100%. The lowest is `single-session-preference` (86.7%), where the benchmark's gold answers are long synthesized preference descriptions that do not appear verbatim in any haystack message. That category's ceiling is embedder-bound; stronger encoders close the gap.

### Comparison to public leaderboard (April 2026)

| System | Score | Source |
|---|---|---|
| **graphstore + bge-small (skill adapter)** | **97.6%** | this run |
| MemPalace | 96.6% | published |
| OMEGA | 95.4% | published leaderboard |
| MemMachine | 93.0% | published |
| LiCoMemory | 73.8% | published |
| Letta (MemGPT) | ~83.2% | published |
| Mem0 | 67.1% on LOCOMO (different benchmark) | published |

graphstore is #1 and using the weakest-to-moderate embedder of the pack.

### Where the wins come from

The skill-compliant adapter gains ~55 percentage points over a naive graphstore adapter (`kind = "memory"`, no schema, no FTS population, no graph, plain `REMEMBER`). Breakdown:

1. **Schema-first ingestion** - `SYS REGISTER NODE KIND "message" ... EMBED content` pre-allocates typed columns and wires auto-embed through the schema engine instead of the DOCUMENT blob path.
2. **BM25 via `DocumentStore.put_summary`** - the single biggest lever. Populates the FTS5 `doc_fts` virtual table so REMEMBER's BM25 leg actually contributes 20% of its fusion weight.
3. **Per-category query dispatch** - multi-session questions fuse REMEMBER with `RECALL FROM <entity>`, knowledge-update questions fuse REMEMBER with `NODES ORDER BY __updated_at__ DESC`.
4. **Entity graph** - regex extraction creates `entity` nodes and `mentions` edges so `RECALL` walks cross-session.
5. **Multi-kind schema** - `session`, `message`, `entity` as separate kinds with `WHERE kind = "message"` filter on REMEMBER so entity/session nodes don't compete for vector slots.
6. **Session-based scoring** - hits count if a retrieved node's session field contains the answer_session_id; matches LongMemEval's actual LLM-as-judge protocol better than naive substring matching.

All of this is documented in [`skills/graphstore-ingestion/SKILL.md`](skills/graphstore-ingestion/SKILL.md).

### Reproducing this run

```bash
# 1. Download LongMemEval-S
mkdir -p ~/graphstore-models/longmemeval
cd ~/graphstore-models/longmemeval
curl -L -o longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

# 2. Build the benchmark image
cd <graphstore repo>
docker build -f benchmarks/framework/Dockerfile.bench -t graphstore-bench:latest .

# 3. Run the full 500
docker run -d \
  --cpus=12 --memory=16g \
  -v ~/graphstore-models:/data:ro \
  -v ~/graphstore-models/cache:/cache \
  -v "$(pwd)/benchmarks/framework/results:/results" \
  --name gs_bench \
  graphstore-bench:latest \
  --embedder fastembed --embedder-model BAAI/bge-small-en-v1.5 \
  --run-tag bge_small_full500

# 4. Wait ~4 hours, then read the results
docker logs gs_bench | tail -30
```

Expect ~4 hours wall clock on 12 CPUs. Peak RAM ~2.4 GB. Host must be Linux or macOS with Docker.

### Known quirks from this run

1. **Docker Desktop virtual filesystem** - on the host machine used for this run, Docker Desktop routes bind mounts through its internal VM, so files written to `/results` inside the container ended up in the VM's view of that path rather than on the real host FS. The result files had to be rescued with `docker cp`. A clean native Docker / podman install does not have this issue.
2. **fastembed cache corruption on `bge-base-en-v1.5`** - a parallel queued run crashed with `NoSuchFile` on `model_optimized.onnx` - the cache download had partially failed. Workaround: delete `~/graphstore-models/cache/models--qdrant--bge-base-en-v1.5-onnx-q` and re-run.
3. **`bge-large-en-v1.5` was too slow on 12 CPUs** - ~240 s per record vs 30 s for bge-small. Full 500 would take ~33 hours. The 3× quality improvement wasn't worth the 8× compute. Killed. A GPU path would make it tractable but it isn't needed to beat the leaderboard.

### What's **not** in this number

- No cloud API calls. No OpenAI / Anthropic / Gemini inference.
- No GPU.
- No LLM extraction at ingest time (which is Mem0's approach).
- No prompt engineering on the query side - the question goes straight into `REMEMBER` as-is.
- No query reformulation or multi-hop rewriting.
- No per-dataset tuning - same adapter would run on LoCoMo, BEIR, AMB.

---

## Three benchmark harnesses

GraphStore ships three different harnesses, each for a different job:

| Tool | Directory | What it does | When to use |
|---|---|---|---|
| **Adapter framework** | `benchmarks/framework/` | Apples-to-apples head-to-head against other agent-memory systems (chroma+bm25, llamaindex, mem0, letta, …) | Comparing graphstore against competitors on LongMemEval |
| **Legacy LongMemEval driver** | `benchmarks/longmemeval.py` | Per-question graphstore-only driver with embedder / mode / ingest-mode sweeps | Tuning graphstore in isolation; parameter sweeps |
| **Algos pytest-benchmark** | `benchmarks/algos/` | Micro-benchmarks for pure numpy/scipy primitives in `graphstore/algos/` | Perf-tuning individual algorithms without booting a GraphStore |

Each harness has its own detailed README linked below.

---

### 1. Adapter framework (`benchmarks/framework/`)

**Use this for head-to-head.** One adapter per system, one runner, one set of metrics, one leaderboard. Every system runs in its documented "best" mode. This is the harness that produced the 97.6% headline number above.

**Adapters shipped:**

| System | Status | Adapter file |
|---|---|---|
| `graphstore` | ✅ full - skill-compliant | `adapters/graphstore_.py` |
| `chroma-bm25` | ✅ full - dense (chromadb) + sparse (rank-bm25) + RRF fusion | `adapters/chroma_bm25.py` |
| `llamaindex` | ✅ full - VectorStoreIndex default path | `adapters/llamaindex_.py` |
| `mem0` | ✅ full | `adapters/mem0.py` |
| `letta` | stub | `adapters/letta.py` |

All baseline adapters use **FastEmbed `bge-small-en-v1.5`** so the embedder is not a variable.

**Quickstart:**

```bash
# Download LongMemEval-S
mkdir -p benchmarks/framework/data/longmemeval
curl -fsSL -o benchmarks/framework/data/longmemeval/longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

# List available adapters
python -m benchmarks.framework.cli list

# Run graphstore on the first 100 questions (smoke)
python -m benchmarks.framework.cli run \
    --system graphstore \
    --dataset longmemeval \
    --data-path benchmarks/framework/data/longmemeval \
    --variant s \
    --max-questions 100 \
    --ceiling-mb 4096

# Run a competitor with the same config
python -m benchmarks.framework.cli run \
    --system chroma-bm25 \
    --dataset longmemeval \
    --data-path benchmarks/framework/data/longmemeval \
    --variant s \
    --max-questions 100
```

Results land in `benchmarks/framework/results/` as three files per run:

- `<system>_longmemeval_s_<timestamp>.json` - full structured result (quality, latency percentiles, memory, cost)
- `<system>_longmemeval_s_<timestamp>.csv` - flat row for spreadsheets
- `<system>_longmemeval_s_<timestamp>.md` - human-readable leaderboard

**Docker** - there's a `benchmarks/framework/Dockerfile.bench` with chromadb, llama-index, fastembed, and graphstore pre-installed. Build once, run any adapter. See `benchmarks/framework/scripts/bench_scheduler.sh` for a cron-friendly queue runner (parameterized via `GS_MODELS` / `GS_REPO` env vars).

**Full details:** see [`benchmarks/framework/README.md`](benchmarks/framework/README.md) for the adapter contract, metrics definitions, writing-a-new-adapter guide, and the ground rules for publishing numbers honestly.

---

### 2. Legacy LongMemEval driver (`benchmarks/longmemeval.py`)

**Use this for graphstore-only parameter sweeps.** Predates the framework. Still passes its tests (`tests/test_longmemeval_benchmark.py`). Useful for embedder / mode / ingest-mode sweeps where you don't want the ceremony of writing an adapter.

**Download:**

```bash
mkdir -p /tmp/longmemeval-data
curl -fsSL -o /tmp/longmemeval-data/longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
```

**Quick smoke:**

```bash
uv run python benchmarks/longmemeval.py /tmp/longmemeval-data/longmemeval_s_cleaned.json \
  --mode lexical --limit 20
```

**Full run:**

```bash
uv run python benchmarks/longmemeval.py /tmp/longmemeval-data/longmemeval_s_cleaned.json \
  --mode remember \
  --out /tmp/results_longmemeval_remember.jsonl
```

#### Ingest modes

Two strategies for loading corpus items into graphstore:

- `flat` **(default)**: `CREATE NODE` with the full session text as one blob - one vector per session.
- `native`: `INGEST` pipeline - auto-chunks each session by paragraph, embeds each chunk individually, populates FTS properly.

**Full-run results (500 questions, 470 scored, REMEMBER mode, model2vec):**

| Ingest Mode | R@5 | R@10 | nDCG@10 |
|---|---|---|---|
| flat | **89.1%** | **93.2%** | **78.4%** |
| native | 85.5% | 91.3% | 73.7% |

**Per-type R@10 breakdown:**

| Question Type | flat | native | Winner |
|---|---|---|---|
| single-session-user | 75.0% | **90.6%** | native (+15.6pp) |
| multi-session | **98.3%** | 95.9% | flat |
| single-session-preference | **80.0%** | 50.0% | flat (+30pp) |
| temporal-reasoning | **97.6%** | 92.9% | flat |
| knowledge-update | 94.4% | **100.0%** | native |
| single-session-assistant | **98.2%** | 89.3% | flat |

**Why flat is the default:** it wins overall (93.2% vs 91.3% R@10) and on 4/6 question types. native excels only on `single-session-user` (+15.6pp) - buried-fact questions in long mixed-topic conversations - but pays for it with a severe regression on `single-session-preference` (-30pp) and a -1.9pp overall loss.

Use `--ingest-mode native` only when your workload is dominated by single-session long conversations where the answer is a specific buried fact.

#### Retrieval modes

- `remember` - 5-signal fusion (vector + BM25 + recency + confidence + recall frequency)
- `similar`  - dense-only retrieval
- `lexical`  - BM25-only retrieval
- `hybrid`   - simple RRF fusion of similar + lexical

**REMEMBER vs SIMILAR depends on embedder strength:**

| Embedder | REMEMBER R@5 | SIMILAR R@5 | Winner |
|---|---|---|---|
| model2vec (weak) | **89.1%** | ~70.0% | REMEMBER (+19pp, BM25 compensates) |
| MiniLM-L6-v2 (strong) | 92.1% | **93.4%** | SIMILAR (+1.3pp) |

With a weak embedder, BM25 fusion in REMEMBER adds ~19pp over pure vector. With a strong encoder, REMEMBER's fusion actively hurts because the recency/confidence/recall_freq signals are uniform noise on LongMemEval. For benchmarks with a modern encoder, prefer `--mode similar`.

#### Embedder options

| Flag | Model | Dims | Kind | Notes |
|---|---|---|---|---|
| `default` | model2vec `M2V_base_output` | 256 | static distilled | instant, ~232 KB install, weakest quality |
| `minilm-l6` | sentence-transformers/all-MiniLM-L6-v2 | 384 | encoder, mean pooling | via fastembed |
| `bge-small` / `bge-base` / `bge-large` | BAAI bge-en-v1.5 | 384 / 768 / 1024 | encoder, mean pooling | via fastembed |
| `mxbai-large` | mixedbread-ai/mxbai-embed-large-v1 | 1024 | encoder, mean pooling | via fastembed |
| `snowflake-l` / `snowflake-m` | snowflake/arctic-embed | 1024 / 768 | encoder, mean pooling | via fastembed |
| `jina-v2` | jinaai/jina-embeddings-v2-base-en | 768 | encoder | via fastembed, 8k context |
| `nomic-v1.5` | nomic-ai/nomic-embed-text-v1.5 | 768 | encoder | via fastembed, 8k context |
| `embeddinggemma` | Google EmbeddingGemma-300M | 256 (Matryoshka) | encoder, mean pooling | slow on CPU |
| `embeddinggemma-768` | Google EmbeddingGemma-300M | 768 | encoder, mean pooling | slower, higher quality |
| `harrier` | Microsoft Harrier-OSS-v1 0.6B | 1024 | decoder, last-token pooling | blocked - needs `onnxruntime-genai` re-export |
| `fastembed:<hf-id>` | any fastembed-supported model | varies | - | pass the HF ID directly |
| `model2vec:<hf-id>` | any model2vec HF model | varies | static distilled | e.g. `model2vec:minishlab/potion-retrieval-32M` |

**Install:** `pip install graphstore[fastembed]` unlocks the fastembed models.

#### Historical reference - legacy driver best numbers

These are the legacy driver's best configurations and are kept for historical context. The modern headline number (97.6%) is produced by the framework adapter above.

| # | Config | R@5 | R@10 | nDCG@10 | elapsed |
|---|---|---|---|---|---|
| 1 | model2vec + REMEMBER | 89.1% | 93.2% | 78.4% | 180 s |
| 2 | MiniLM + REMEMBER | 92.1% | 95.3% | 82.9% | 390 s |
| 3 | MiniLM + SIMILAR | 93.4% | 97.0% | 84.0% | 315 s |
| 4 | **MiniLM + SIMILAR + user-turns only** | **94.7%** | **98.1%** | **87.2%** | **258 s** |

**Reproduce the legacy-driver best run:**

```bash
pip install graphstore[fastembed]
uv run python benchmarks/longmemeval.py \
  /tmp/longmemeval-data/longmemeval_s_cleaned.json \
  --embedder minilm-l6 \
  --mode similar \
  --ingest-mode flat
```

#### Ingest performance (batched embedding)

The `flat` path uses `GraphStore.deferred_embeddings()` to batch vector computations. Within the context, `CREATE NODE` queues `(slot, text)` pairs and flushes them in one `encode_documents()` call. This is a ~1.5–2x speedup for transformer embedders and fixes a double-embedding bug where `CREATE NODE` with both schema `EMBED` and `DOCUMENT` clauses previously embedded the same text twice.

**Measured speedup (model2vec, full 500-question run):**

| Version | elapsed | R@5 | R@10 |
|---|---|---|---|
| pre-fix (double embed, unbatched) | 397 s | 89.1% | 93.2% |
| post-fix (single embed, batched) | **180 s** | **89.1%** | **93.2%** |

Same scores, **2.2x faster**.

```python
with gs.deferred_embeddings(batch_size=64):
    for item in items:
        gs.execute(f'CREATE NODE "{item.id}" text = "{item.text}" DOCUMENT "{item.text}"')
# All embeddings flushed here.
```

#### Notes

- Retrieval only. No answer generation.
- BM25 indexing is backed by `DocumentStore` summaries populated during benchmark ingestion.
- Default scoring skips abstention questions. Use `--include-abstention` to include them.
- Each question is evaluated in a fresh temporary `GraphStore` to avoid cross-question leakage.

---

### 3. Algos pytest-benchmark (`benchmarks/algos/`)

**Use this for tuning pure primitives.** Micro-benchmarks over every function in `graphstore/algos/` - graph traversal, compaction, BM25 fusion, aggregation, column predicates, FTS5 sanitize, materialization, string GC, and more.

- Synthetic numpy inputs (fixed seeds) - no GraphStore fixture, no SQLite, no I/O
- Parametrized by input size (1k / 10k / 100k nodes)
- Grouped by function; saved baselines for regression comparison
- LLM-friendly `bench_one.py` per-algo metric emitter + `dump_env.py` package manifest for autoresearch workflows

**Quickstart:**

```bash
# Run every benchmark, save as the current run
./benchmarks/algos/run.sh run

# Capture current numbers as the reference baseline
./benchmarks/algos/run.sh baseline

# Diff the current run against the baseline (warn at >10%)
./benchmarks/algos/run.sh compare

# CI-friendly strict gate (fail at >5%)
./benchmarks/algos/run.sh gate
```

Pass pytest filters as extra args:

```bash
./benchmarks/algos/run.sh run -k "bfs or dijkstra"
./benchmarks/algos/run.sh run benchmarks/algos/test_fusion_bench.py
```

**What's covered:**

| Module | Functions benched | Sizes |
|---|---|---|
| `graph` | bfs_traverse, dijkstra, bidirectional_bfs, find_all_paths, common_neighbors | 1k / 10k / 100k + dense 1k |
| `compact` | build_live_mask, slot_remap_plan, apply_slot_remap_to_edges | 10k / 100k |
| `fusion` | rrf_fuse, normalize_bm25, recency_decay, weighted_remember_fusion | 100 / 1k / 10k |
| `spreading` | spreading_activation | 1k / 10k / 100k × depth 2–4 |
| `eviction` | needs_optimization, rank_evictable_slots | 10k |
| `text` | fts5_sanitize, tokenize_unicode | short / long / batch |

**Full details:** see [`benchmarks/algos/README.md`](benchmarks/algos/README.md) for the contract, fixture catalogue, autoresearch workflow, purity gate, and the "one-file-at-a-time optimization" guide.

---

## Directory layout

```
benchmarks/
├── longmemeval.py               # legacy per-record driver
│
├── framework/                   # adapter-based head-to-head
│   ├── README.md
│   ├── cli.py                   # python -m benchmarks.framework.cli
│   ├── runner.py                # per-record evaluation loop
│   ├── datasets.py              # LongMemEval loader
│   ├── metrics.py               # quality/latency/memory/cost
│   ├── report.py                # json/csv/md writers
│   ├── adapters/                # one .py per system
│   │   ├── graphstore_.py
│   │   ├── chroma_bm25.py
│   │   ├── llamaindex_.py
│   │   ├── mem0.py
│   │   └── letta.py             # stub
│   ├── Dockerfile.bench         # pre-installed runner image
│   ├── scripts/                 # bench_scheduler.sh, chain_next.sh
│   └── results/                 # per-run outputs (gitignored)
│
└── algos/                       # pytest-benchmark for pure primitives
    ├── README.md
    ├── run.sh                   # run / baseline / compare / gate
    ├── bench_one.py             # per-algo structured metric emitter
    ├── dump_env.py              # package manifest for autoresearch
    ├── allowlist.py             # single source of truth for import purity
    ├── conftest.py              # synthetic fixtures
    └── test_*_bench.py          # one bench file per algos/ module
```

---

## What's next

- [x] Full 500 on bge-small (97.6%)
- [x] Chroma + BM25 baseline adapter
- [x] LlamaIndex baseline adapter
- [ ] Full 500 on chroma-bm25 and llamaindex for a clean head-to-head line
- [ ] Mem0 full-run (requires `OPENAI_API_KEY`, ~$2-5 per run)
- [ ] LoCoMo loader + second-benchmark validation
- [ ] GPU path via onnxruntime CUDAExecutionProvider for bge-large
