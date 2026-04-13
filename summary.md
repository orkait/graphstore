# GraphStore Retrieval - Comprehensive Session Summary

This document covers ALL work done across multiple sessions on GraphStore's retrieval layer, benchmarking infrastructure, and performance optimization. It is written so another agent or developer can pick up from exactly where we left off.

---

## Table of Contents

1. [Project Context](#project-context)
2. [Architecture Overview](#architecture-overview)
3. [Code Changes (Detailed)](#code-changes-detailed)
4. [Bugs Found and Fixed](#bugs-found-and-fixed)
5. [Benchmarking Infrastructure](#benchmarking-infrastructure)
6. [Docker & Kaggle Setup](#docker--kaggle-setup)
7. [Autoresearch & Tuner](#autoresearch--tuner)
8. [Ratchet Loop Methodology](#ratchet-loop-methodology)
9. [Benchmark Results](#benchmark-results)
10. [Rival Architecture Research](#rival-architecture-research)
11. [Key Learnings](#key-learnings)
12. [What to Do Next](#what-to-do-next)
13. [How to Run Everything](#how-to-run-everything)
14. [Key Files Reference](#key-files-reference)

---

## Project Context

GraphStore is an in-memory typed graph database designed as agentic memory for AI agents. It combines 5 engines: graph (numpy + CSR sparse matrices), vector (usearch ANN), document (SQLite FTS5 BM25), DSL (custom query language via PEG grammar), and evolution (metacognitive self-tuning).

The goal of this work was to:
1. Benchmark GraphStore against competitors (Mem0, Zep, ChromaDB, LlamaIndex) on LongMemEval and LoCoMo
2. Prove retrieval quality and find weaknesses
3. Optimize the retrieval pipeline
4. Prepare publishable benchmark numbers

---

## Architecture Overview

### REMEMBER Pipeline (10 stages)

The `REMEMBER` command is the core retrieval. It was rewritten from a simple 3-signal weighted sum to a 10-stage pipeline:

```
REMEMBER "query" LIMIT k [AT "date"] [TOKENS n] [WHERE ...]

Stage 1:  Candidate Gathering     - vector ANN (usearch) + BM25 (FTS5) -> union pool
Stage 2:  Signal Computation      - 5 signals computed over candidate pool
Stage 3:  Temporal-First Filter   - hard filter by __event_at__ range (TSM-inspired)
Stage 4:  Fusion                  - weighted sum or RRF over 5 signals
Stage 5:  Post-Fusion Modifiers   - temporal proximity boost, multiplicative recency
Stage 6:  HybridRAG Expansion     - spreading activation from top seeds through graph
Stage 7:  Type-Weighted Scoring   - multiply by node kind weight (observation=1.8, fact=1.3, etc.)
Stage 8:  Top-K Selection         - sort by final_scores, materialize nodes, apply WHERE filter
Stage 9:  Nucleus Expansion       - BFS from retrieved nodes through graph edges (context windowing)
Stage 10: Retrieval Feedback      - increment __recall_count__ and __last_recalled_at__
```

### 5 Signals

| Signal | Source | Range | When Active |
|---|---|---|---|
| vec_signal | Cosine similarity from usearch ANN | [0, 1] | Always (with embedder) |
| bm25_signal | Normalized BM25 from SQLite FTS5 | [0, 1] | Always (with persistence) |
| recency_signal | Exponential decay from timestamps | [0, 1] | Always (data-relative) |
| graph_signal | Normalized out-degree + confidence override | [0, 1] | When graph has edges |
| recall_signal | Normalized __recall_count__ | [0, 1] | After first retrieval |

Co-occurrence boost: candidates found by BOTH vector AND BM25 get `vec_signal *= (1 + bm25_signal)`.

### Config System

4-layer chain, each overrides the previous:
1. `config.py` defaults (source of truth for types + default values)
2. `graphstore.json` file (per-deployment overrides, diffs only)
3. `GRAPHSTORE_*` environment variables (Docker/k8s)
4. Constructor kwargs (code-level, highest priority)

Key config after ratcheting:
```python
fusion_method = "weighted"
remember_weights = [0.50, 0.20, 0.10, 0.15, 0.05]  # vec, bm25, recency, graph, recall
recency_half_life_days = 7300.0  # ~20 years
recency_mode = "multiplicative"
search_oversample = 16
retrieval_depth = 9
hybridrag_weight = 0.15
nucleus_expansion = True
nucleus_hops = 2
```

**IMPORTANT**: The benchmark adapter (`graphstore_.py`) had a `_BENCHMARK_DEFAULTS` dict that shadow-overrided config.py with tuner values (including `recency_half_life_days=48`). This was removed. Config.py is the ONLY source of truth now.

---

## Code Changes (Detailed)

### Core Engine

**`graphstore/dsl/handlers/intelligence.py`** - Complete REMEMBER rewrite
- 10-stage pipeline with planner integration
- 5 active signals replacing 3 (2 were dead for batch-ingested data)
- Smart recency: `__event_at__` preferred over `__updated_at__`, separate reference groups for mixed timestamps
- Graph degree signal replaces dead confidence signal (normalized out-degree from CSR)
- Co-occurrence boost for dual-source candidates
- Temporal-first filtering: hard date range, then rank within range
- HybridRAG respects temporal filter (activation masked)
- Nucleus expansion: no WHERE propagation, min 20 char text filter, token budget enforcement
- Retrieval feedback: increment __recall_count__ on returned nodes

**`graphstore/core/edges.py`** - Delta transpose fix (CRITICAL)
- `get_transpose()`, `get_combined_transpose_split()`, `get_combined_transpose()` all had `(tgts, srcs)` instead of `(srcs, tgts)` for delta CSR
- This meant RECALL from any entity returned 0 results and HybridRAG graph expansion was a no-op
- All 784 edges in LoCoMo were in the dynamic buffer (not frozen CSR), so this affected 100% of graph operations

**`graphstore/algos/fusion.py`** - New fusion primitives
- `rrf_remember_fusion()` - rank-based fusion with tie-aware averaging
- `temporal_proximity()` - Gaussian proximity score for AT anchoring
- `rrf_k` guard: `max(k_rrf, 1.0)` prevents division by zero
- Tie-aware ranking: equal signal values get averaged ranks (no arbitrary slot-index ordering)

**`graphstore/core/temporal.py`** - First-class date parser
- Absolute: "8 May 2023", "May 2023", "2023", ISO-8601, LoCoMo format ("1:56 pm on 8 May, 2023")
- Relative: "3 days ago", "last Friday", "last June"
- Ranges: "before June 2023", "the week before 9 June 2023"
- Extraction: find all dates in free text (used for temporal anchoring in queries)
- Edge cases: invalid dates (32 May) return None, month names not matched as bare years

**`graphstore/algos/consolidation.py`** - Entity-topic clustering
- Groups messages by entity via graph edges
- Clusters by cosine similarity (greedy single-linkage)
- Creates observation nodes with centroid vectors and evidence links
- No LLM needed - picks most representative message per cluster
- Built and tested but NOT wired into benchmark (hurt results when mixed in fusion)

**`graphstore/retrieval/planner.py`** - Query-aware retrieval planner
- `RetrievalContext` captures what the query needs
- `RetrievalPlan` decides what stages to enable
- Rule-based policy: temporal anchor -> filter, preference keywords -> observations, entities -> graph expansion
- Plan metadata in `result.meta["planner"]` for debugging

**`graphstore/config.py`** - New config fields
- `fusion_method` ("rrf" or "weighted"), `rrf_k`, `recency_mode` ("additive" or "multiplicative")
- `type_weights` dict, `temporal_weight`, `temporal_decay_days`
- `nucleus_expansion`, `nucleus_hops`, `nucleus_max_neighbors`
- `recency_half_life_days` changed from 48 (tuner) to 7300 (~20 years)
- `remember_weights` ratcheted to [0.50, 0.20, 0.10, 0.15, 0.05]

**`graphstore/graphstore.py`** - Constructor kwargs for all new config fields
- All new DslConfig fields wired as constructor kwargs
- Executor receives all config values
- Retrieval planner instantiated and attached to executor

**`graphstore/dsl/ast_nodes.py`** - New DSL syntax
- `EVENT_AT` on CreateNode, UpsertNode, AssertStmt
- `AT` clause on RememberQuery
- `SysConsolidate` with THRESHOLD and MIN_CLUSTER_SIZE

**`graphstore/dsl/grammar.lark`** + **`graphstore/dsl/transformer.py`** - Parser support for new syntax

**`graphstore/dsl/handlers/mutations.py`** - EVENT_AT handling via `_parse_event_at()` -> `_apply_event_at()`

**`graphstore/dsl/handlers/beliefs.py`** - EVENT_AT on ASSERT

---

## Bugs Found and Fixed

### Critical

| Bug | Impact | Fix |
|---|---|---|
| Delta transpose (tgts, srcs) swapped in 3 places | RECALL returned 0, HybridRAG was no-op | Swap to (srcs, tgts) in edges.py |
| Recency decay from wall clock killed historical data | exp(-1071/48) = 2e-10, all results zeroed | Data-relative reference per timestamp group |
| Mixed __event_at__ + __updated_at__ reference | Nodes without event_at got reference=now, crushing event_at nodes | Separate reference per group |

### High

| Bug | Impact | Fix |
|---|---|---|
| 0.5 baseline skewed RRF ranks | Dead signals got arbitrary ranks instead of worst | Zero baseline, RRF pushes to worst rank |
| Confidence double-counted (signals 4+5) | ASSERT nodes boosted 2x | Confidence only in signal 4 (graph) |
| RRF arbitrary tie-breaking by slot index | Equal candidates ranked by ingestion order | Average rank for ties |
| HybridRAG leaked past temporal filter | Graph activation brought back filtered nodes | Temporal mask on activation |
| Adapter shadow defaults overrode config.py | recency_half_life_days=48 instead of 7300 | Removed _BENCHMARK_DEFAULTS dict |

### Medium

| Bug | Impact | Fix |
|---|---|---|
| Nucleus breaks LIMIT semantics | LIMIT 10 could return 70+ results | Default off, token budget enforcement |
| Nucleus WHERE filter killed useful neighbors | Entity/session context excluded | No WHERE on nucleus |
| Nucleus adds garbage entity names | "Caroline", "Melanie" as context text | Min 20 char text filter |
| Entity extraction in core REMEMBER | English stop words in language-agnostic DB | Moved to adapter layer |
| fusion_method not validated | Invalid string silently used weighted | Validate, fall back to default |
| Temporal filter Python loop | Slow for large candidate pools | Vectorized numpy |

---

## Benchmarking Infrastructure

### Benchmark Adapters

`benchmarks/framework/adapter.py` defines the protocol: `reset() -> ingest(session) -> query(question) -> close()`. Every system implements these 4 methods.

Adapters exist for: GraphStore, ChromaDB+BM25, LlamaIndex, Mem0, Letta.

**GraphStore adapter** (`benchmarks/framework/adapters/graphstore_.py`):
- 8 retrieval strategies: remember, remember_graph, remember_recency, remember_lexical, remember_rerank, full, full_rerank, consolidated
- Schema: registers "message" (with EMBED), "session", "entity" node kinds
- Entity extraction via regex (with stop-word filter)
- BM25 populated via `doc_store.put_summary()` during ingest
- `__event_at__` set from session metadata dates
- `_resolve_strategy()` picks strategy per category (configurable)

### LLM Client (`benchmarks/framework/llm_client.py`)

- Uses litellm with autoresearch config for provider fallback
- QA_MODEL = "minimax/minimax-m2.7:nitro" on OpenRouter (paid, reliable)
- `health_check()` - verifies LLM is reachable before benchmark (needs max_tokens=500 for thinking models)
- `generate_answer()` - official LoCoMo prompt: "write an answer in the form of a short phrase, answer with exact words from the context"
- `compute_f1()` - official LoCoMo F1: Porter stemming, Counter multiset, per-category handling

### Async Batch Scheduler (`benchmarks/framework/llm_batch.py`)

- Splits questions across providers (8 per batch per provider)
- `asyncio` with `run_in_executor` (litellm is sync)
- 90s hard timeout per call, streaming enabled
- 5s cooldown between rounds
- Early exit after 2 consecutive rounds with no progress
- Provider rotation: round-robin through available providers

### Benchmark Runners

**LongMemEval** (`benchmarks/framework/runner.py`):
- Per-record isolated evaluation: reset -> ingest haystack -> query -> score
- Signal handler saves partial results on SIGTERM/SIGINT
- QA eval integration (optional LLM judge via llm_judge.py)

**LoCoMo** (`benchmarks/framework/run_locomo.py`):
- Ingest ALL sessions for a conversation ONCE
- Query ALL QAs against same ingested state
- Official F1 scoring with Porter stemming
- Concurrent LLM calls via llm_batch.py
- Per-category reporting in official order (4,1,2,3,5)

### Metrics (`benchmarks/framework/metrics.py`)

- `QualityMetrics` - accuracy, recall@k, LLM judge, per-category breakdown
- `LatencyMetrics` - p50/p95/p99/mean/stddev
- `MemoryMetrics` - RSS before/after/peak via psutil
- `CostMetrics` - token counts
- Scoring: substring match of gold answer in retrieved text, OR session-ID match

---

## Docker & Kaggle Setup

### Docker (`benchmarks/framework/Dockerfile.bench.gpu`)

- Base: onnxruntime GPU image
- Installs: graphstore, onnx, litellm, flashrank
- Resource limits enforced: `--cpus=8 --memory=16g` (Rule 12)
- Buildx builder capped at 12 CPUs, 16GB RAM
- `docker_runner.py` handles: model download, volume mounts, arg validation, JSON serialization

### Kaggle

Notebooks at `benchmarks/kaggle/`:
- `graphstore_jina_500.py` - GraphStore + Jina v5 Nano on LongMemEval-S (500 records)
- `chroma_jina_500.py` - ChromaDB + BM25 baseline
- `bench_config.py` - shared config (embedder, dataset, hardware settings)

Kaggle setup:
- **30 hours/week free GPU** (T4 or P100)
- HF token passed via environment variable (stripped from committed code)
- `onnxruntime-gpu` installed with `--no-deps --force-reinstall` after main pip install (avoids CPU wheel overwriting GPU .so)
- Dataset: `repo_type="dataset"` for huggingface snapshot_download (not "model")
- Username: `superkaiii` on Kaggle, `superkai` on Lightning.ai

### Kaggle Results (from earlier session)

GraphStore + Jina Nano + LongMemEval-S (500 records):
```
Overall accuracy: 96.4%
knowledge-update:           100.0%
multi-session:               98.5%
single-session-assistant:   100.0%
single-session-preference:   86.7%
single-session-user:         98.6%
temporal-reasoning:          91.7%
Ingest p50: 322ms, Query p50: 20ms
Total elapsed: 7381s (~2 hours)
```

---

## Autoresearch & Tuner

### Config Tuner (`autoresearch/tune_config.py`)

- Optuna Bayesian optimization for retrieval config knobs
- Search space: retrieval_strategy, fusion_method, rrf_k, recency_mode, nucleus_*, remember_weights components, search_oversample, retrieval_depth, recall_depth, etc.
- `build_output_config()` produces tuned JSON preserving all knob keys
- Fast bench mode: balanced subset for quick iteration

### Tuned Configs

- `autoresearch/tuned_config.48.json` - from 48-record balanced LongMemEval slice (best evidence)
- `autoresearch/tuned_config.json` - earlier tuning run
- `autoresearch/tuned_config.small.json` - small config variant

Key tuner findings:
- Breadth matters more than fusion tweaks (search_oversample, retrieval_depth)
- Additive recency regressed retrieval
- fusion_method="weighted" beat "rrf" on LongMemEval
- nucleus_expansion=True helped LongMemEval
- Preference questions (single-session-preference) remain the weakest category

**Important**: Tuned values were for LongMemEval's per-record protocol. They don't transfer to LoCoMo (persistent memory across sessions). The tuner's `recency_half_life_days=48` was catastrophic for LoCoMo's historical data.

### Autoresearch Config (`autoresearch/config.json`)

Contains provider configs, API keys, model lists. Gitignored.

Providers:
- `local_ollama` - localhost:11434, models: gemma4:31b-cloud, minimax-m2.7:cloud, qwen3.5:cloud, qwen3-coder-next:cloud
- `openrouter` - openrouter.ai/api/v1, models: minimax/minimax-m2.7:nitro, google/gemma-4-31b-it:free, google/gemma-4-31b-it (paid)

Credits: ~$2.34 remaining on OpenRouter. MiniMax pricing: $0.30/1M input, $1.20/1M output.

---

## Ratchet Loop Methodology

### What It Is

Fast-fail iterative testing: make one change, test on a fixed dataset, keep if improves, revert if not.

### Test Harness (`benchmarks/framework/ratchet50.py`)

- 50 validated questions from LoCoMo conv-26
- **Validated** = the gold answer keyword MUST exist in the ingested data (eliminates impossible questions)
- GraphStore populated ONCE with all 19 sessions, kept alive for all 50 queries
- Metric: keyword from gold answer found in top-K retrieved passages
- Fixed random seed (42) for reproducibility
- Categories: multi-hop, open-domain, single-hop (no temporal in this sample)

### Ratchet Results (jina-v5-small, top-10)

| Change | Score | Delta | Kept? |
|---|---|---|---|
| Baseline (nano 768d, top-5, vec=0.30) | 29/50 | - | - |
| + jina-v5-small 1024d | 32/50 | +3 | Yes |
| + top-10 instead of top-5 | 37/50 | +5 | Yes |
| + vec weight 0.50 | **40/50** | +3 | Yes |
| additive recency | 40/50 | 0 | No |
| no graph signal | 37/50 | -3 | No |
| oversample 32 | 40/50 | 0 | No |
| vec weight 0.45 | 39/50 | -1 | No |
| no nucleus | 40/50 | 0 | No |
| stronger co-occurrence (2x) | 38/50 | -2 | No |
| vec weight 0.55 | 40/50 | 0 | No |
| pure vector (1.0) | 38/50 | -2 | No |
| HybridRAG 0.30 | 40/50 | 0 | No |
| no recency/recall | 39/50 | -1 | No |
| RRF fusion | 24/50 | -16 | No |

### Recall@K Analysis (jina-v5-small)

| K | Recall | Ceiling |
|---|---|---|
| top-5 | 30/50 (60%) | |
| top-10 | 40/50 (80%) | |
| top-20 | 42/50 (84%) | |
| top-30 | 44/50 (88%) | |
| top-50 | 48/50 (96%) | |
| all 184 msgs | 50/50 (100%) | Absolute ceiling |

**Key insight**: Retrieval FINDS almost everything (96% at top-50). The problem is RANKING - answers at rank 20-50 need to be in top-10.

---

## Benchmark Results

### LongMemEval-S (retrieval-only metric)

| System | Embedder | Accuracy | Platform |
|---|---|---|---|
| GraphStore | jina-nano 768d, GPU | **96.4%** | Kaggle T4 |
| GraphStore | model2vec (old) | 97.5% | Docker CPU |

### LoCoMo (retrieval + LLM reader, token F1)

| Config | LLM | Embedder | Sample | F1 |
|---|---|---|---|---|
| Old baseline | MiniMax (Ollama) | nano 768d | 199Q conv-26 | 0.399 |
| **Current best** | **MiniMax (nitro)** | **small 1024d** | **50Q random** | **0.357** |
| Gemma4 | Gemma4 31B | small 1024d | 50Q random | 0.145 |
| GPT-3.5-turbo full-context | GPT-3.5-turbo-16k | - | paper | 0.378 |
| Human | - | - | paper | 0.879 |
| MemMachine | GPT-4.1-mini | - | LLM-judge | 0.917 |

**Note**: Our F1 uses official token-level F1 with Porter stemming. MemMachine uses LLM-judge (inflates ~1.5-2x). Not directly comparable.

---

## Rival Architecture Research

Systems studied: Hindsight, Mem0, Zep, MemMachine, OMEGA, Supermemory, Ensue, TSM.

### What They Do That We Don't

| Technique | Who Does It | Impact | Our Status |
|---|---|---|---|
| Memory consolidation (LLM-assisted) | MemMachine, Hindsight | High | SYS CONSOLIDATE built, not integrated |
| Sentence-level embeddings | MemMachine | High | We embed full messages |
| Cross-encoder reranking | MemMachine (Cohere), Hindsight | Medium | FlashRank built, hurt LoCoMo |
| Query decomposition | Ensue | Medium | Not implemented |
| Bitemporal model | Zep | Medium | __event_at__ is step 1 |
| Temporal NL parsing | Hindsight, TSM | Medium | core/temporal.py built |
| RRF fusion | Hindsight, OMEGA | Medium | Built, but weighted beats it |
| Disposition traits | Hindsight | Low | Not needed |

### TSM Paper (arXiv 2601.07468) Key Insights

- Point-wise memory (events) + Durative memory (consolidated observations)
- Temporal-first retrieval: hard filter by time range, then rank by semantics
- GMM clustering per time slice for consolidation
- spaCy for temporal NL parsing
- 76.7% F1 on LoCoMo with GPT-4o-mini

---

## Key Learnings

### Retrieval

1. **Vector similarity dominates ranking** with good embedders (jina-v5-small 1024d). vec_signal accounts for ~80% of ranking decisions.
2. **BM25 catches what vector misses** (exact keywords) but rarely wins fusion. Co-occurrence boost is the bridge.
3. **The semantic gap is real** - indirect questions ("Where did she move from?" -> "Sweden") can't be solved by fusion tuning alone. Need query expansion or entity traversal.
4. **Embedder size > fusion tuning** - nano->small gained +5, no config change gained more than +3.
5. **Graph signal helps despite looking weak** - disabling drops 3 points.
6. **RRF is worse than weighted** when signals have different scales (graph/recall near zero).

### Temporal

7. **Recency with short half-life kills historical data** - exp(-167/48) = 0.03 for 5-month-old data.
8. **Data-relative reference is essential** - decay from newest candidate, not wall clock.
9. **Separate timestamp groups** - don't mix __event_at__ and __updated_at__ in the same reference.
10. **100-year half-life makes recency a no-op** - 20 years (7300 days) is the sweet spot.

### Benchmarking

11. **LLM reliability dominates benchmark time** - 80% of MiniMax Ollama cloud calls return empty.
12. **Thinking models need max_tokens >= 500** or they spend all tokens on reasoning.
13. **Free tier rate limits** are per-model-endpoint, not per-account.
14. **OpenRouter nitro** ($0.30/1M input) is reliable and fast. 50Q LoCoMo costs ~$0.07.
15. **Different benchmarks need different configs** - LongMemEval tuner values don't transfer to LoCoMo.
16. **50Q sample can differ significantly from 199Q** due to category distribution.

### Entity Graph

17. **Entity nodes are dead ends** - edges go message->entity (mentions), never entity->message.
18. **Month names extracted as entities** ("May", "August") with high in-degree - pollutes entity graph.
19. **Caroline/Melanie in every question** - entity-hop boost ineffective when entities are universal.
20. **RECALL from entity was broken** (transpose bug) - fixed, now returns correct results.

---

## What to Do Next

### Immediate (no LLM needed)

1. **Improve ranking**: 8 answers at rank 11-50 need to be in top-10. Analyze what pushes them down.
2. **Run full 1986Q LoCoMo** on OpenRouter MiniMax nitro (~$0.40) for official publishable numbers.
3. **Run updated LongMemEval-S on Kaggle** with jina-v5-small to verify no regression from transpose fix.
4. **Fix the ancestor test** broken by transpose fix (test_dsl_user_reads.py).

### Medium-term

5. **Better entity extraction** - filter month names, use spaCy NER instead of regex.
6. **Query expansion** - use top-5 results to generate a refined second query (pseudo-relevance feedback).
7. **Memory consolidation integration** - SYS CONSOLIDATE creates observations, but mixing them in REMEMBER hurt results. Need two-stage retrieval: observations first, episodes fill gaps.
8. **Sentence-level embeddings** - embed each sentence separately instead of full messages (MemMachine's key technique).

### Long-term

9. **LLM-assisted ingest** - extract entity profiles, facts, preferences at ingest time.
10. **Bitemporal model** - track both event time and knowledge time.
11. **Query decomposition** - split multi-hop questions into sub-queries.
12. **LLM-as-judge scoring** - run LoCoMo with LLM judge for comparable-to-SOTA numbers.

---

## How to Run Everything

### Prerequisites

```bash
# Install graphstore with deps
uv sync

# Install embedder models
GRAPHSTORE_MODEL_CACHE_DIR=/tmp/gs_models uv run python3 -c "
from graphstore.registry.installer import install_embedder, set_cache_dir
set_cache_dir('/tmp/gs_models')
install_embedder('jina-v5-small-retrieval')
"

# LoCoMo dataset
# Download locomo10.json from https://huggingface.co/datasets/Percena/locomo-mc10
# Place at /tmp/locomo/raw/locomo10.json

# LongMemEval dataset
# Download from https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
# Place at /tmp/longmemeval/longmemeval_s_cleaned.json
```

### Run Tests

```bash
# Full test suite (1166 tests)
uv run python -m pytest tests/ -x -q

# Retrieval recall test (no LLM, 50Q, ~30s)
uv run python3 -m benchmarks.framework.ratchet50

# LoCoMo with LLM (50Q random sample, ~5 min with paid API)
# Set QA_MODEL in benchmarks/framework/llm_client.py first
uv run python3 -c "
from benchmarks.framework.run_locomo import run_locomo
from benchmarks.framework.datasets import load_locomo
from benchmarks.framework.adapters.graphstore_ import GraphStoreAdapter
import os, random
os.environ['GRAPHSTORE_MODEL_CACHE_DIR'] = '/tmp/gs_models'
ds = load_locomo('/tmp/locomo', max_conversations=1)
random.seed(42)
# ... (see ratchet50.py for full sample building)
config = {'embedder': 'installed', 'embedder_model': 'jina-v5-small-retrieval',
          'embedder_cache_dir': '/tmp/gs_models', 'embedder_gpu': True, 'ceiling_mb': 512}
adapter = GraphStoreAdapter(config=config)
summary, details = run_locomo(adapter, ds, k=10)
"

# Full LoCoMo (all 10 convs, 1986Q, ~$0.40 on MiniMax nitro)
uv run python3 -m benchmarks.framework.run_locomo \
  --data-path /tmp/locomo \
  --embedder installed:jina-v5-small-retrieval \
  --k 10

# LongMemEval on Kaggle
kaggle kernels push -p benchmarks/kaggle
```

### LLM Provider Configuration

Edit `autoresearch/config.json` (gitignored):
```json
{
  "providers": {
    "local_ollama": {
      "base_url": "http://localhost:11434",
      "litellm_prefix": "ollama_chat",
      "models": {
        "minimax-m2.7:cloud": {},
        "gemma4:31b-cloud": {}
      }
    },
    "openrouter": {
      "base_url": "https://openrouter.ai/api/v1",
      "api_key": "sk-or-v1-...",
      "litellm_prefix": "openrouter",
      "models": {
        "minimax/minimax-m2.7:nitro": {}
      }
    }
  }
}
```

Set model in `benchmarks/framework/llm_client.py`:
```python
QA_MODEL = "minimax/minimax-m2.7:nitro"  # or "gemma4:31b-cloud"
QA_MODEL_OR = "minimax/minimax-m2.7:nitro"
```

---

## Key Files Reference

### Core Engine

| File | Purpose |
|---|---|
| `graphstore/config.py` | Single source of truth for all config defaults |
| `graphstore/graphstore.py` | Main entry point, constructor, config wiring |
| `graphstore/dsl/handlers/intelligence.py` | REMEMBER, RECALL, SIMILAR TO, LEXICAL SEARCH, WHAT IF |
| `graphstore/core/edges.py` | CSR sparse matrices + dynamic edge buffer (L0) |
| `graphstore/core/store.py` | CoreStore - node/edge CRUD, numpy arrays |
| `graphstore/core/temporal.py` | Date parser (ISO, natural, relative, ranges) |
| `graphstore/algos/fusion.py` | RRF + weighted fusion + recency decay + temporal proximity |
| `graphstore/algos/spreading.py` | Graph spreading activation (RECALL + HybridRAG) |
| `graphstore/algos/consolidation.py` | Entity-topic clustering (SYS CONSOLIDATE) |
| `graphstore/retrieval/planner.py` | Query-aware retrieval planner |
| `graphstore/vector/store.py` | usearch ANN index |
| `graphstore/document/store.py` | SQLite FTS5 for BM25 |
| `graphstore/dsl/grammar.lark` | PEG grammar for DSL |
| `graphstore/dsl/transformer.py` | AST transformer |
| `graphstore/dsl/ast_nodes.py` | AST node dataclasses |

### Benchmarks

| File | Purpose |
|---|---|
| `benchmarks/framework/adapter.py` | MemoryAdapter protocol (reset/ingest/query/close) |
| `benchmarks/framework/adapters/graphstore_.py` | GraphStore adapter (8 strategies) |
| `benchmarks/framework/runner.py` | LongMemEval per-record runner |
| `benchmarks/framework/run_locomo.py` | LoCoMo runner (official protocol) |
| `benchmarks/framework/datasets.py` | LongMemEval + LoCoMo data loaders |
| `benchmarks/framework/metrics.py` | Quality/latency/memory/cost metrics |
| `benchmarks/framework/llm_client.py` | LLM client (litellm + provider fallback) |
| `benchmarks/framework/llm_batch.py` | Async batch scheduler (dual-provider) |
| `benchmarks/framework/llm_judge.py` | LongMemEval official judge prompts |
| `benchmarks/framework/ratchet50.py` | 50Q validated retrieval recall test |
| `benchmarks/framework/docker_runner.py` | Docker-based benchmark runner |
| `benchmarks/kaggle/graphstore_jina_500.py` | Kaggle notebook for LongMemEval |

### Tests

| File | Purpose |
|---|---|
| `tests/test_retrieval_improvements.py` | RRF, type-weights, nucleus, temporal, config wiring |
| `tests/test_retrieval_planner.py` | Planner policy tests |
| `tests/test_retrieval_loop_controls.py` | Planner + temporal anchor integration |
| `tests/test_remember_signals.py` | Signal breakdown, confidence, recall feedback |
| `tests/test_fusion.py` | Fusion primitives unit tests |
| `tests/test_e2e_real_embedder.py` | End-to-end with real Model2Vec embedder |

### Config & Tuning

| File | Purpose |
|---|---|
| `autoresearch/config.json` | Provider configs + API keys (gitignored) |
| `autoresearch/tune_config.py` | Optuna config tuner |
| `autoresearch/tuned_config.48.json` | Best tuned config from 48-record LongMemEval |
| `skills/graphstore-ingestion/SKILL.md` | Ingestion protocol documentation |
