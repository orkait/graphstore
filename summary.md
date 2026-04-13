# GraphStore Retrieval - Session Summary

## Part 1: Code-Level Changes

### What We Built

**1. REMEMBER Pipeline Rewrite (`graphstore/dsl/handlers/intelligence.py`)**

The REMEMBER command is the core retrieval engine. We rewrote it from a 3-signal weighted sum to a 10-stage pipeline:

```
Stage 1: Candidate Gathering (vector ANN + BM25 FTS5)
Stage 2: Signal Computation (5 signals)
Stage 3: Temporal-First Filtering (hard date range filter)
Stage 4: Fusion (weighted sum or RRF)
Stage 5: Post-Fusion Modifiers (temporal boost, multiplicative recency)
Stage 6: HybridRAG Graph Expansion (spreading activation)
Stage 7: Type-Weighted Scoring (kind-based multipliers)
Stage 8: Top-K Selection + Materialization
Stage 9: Nucleus Expansion (graph neighbor context)
Stage 10: Retrieval Feedback (recall count increment)
```

**5 Signals (all actively differentiate):**
- `vec_signal` - cosine similarity from usearch ANN
- `bm25_signal` - normalized BM25 from SQLite FTS5
- `recency_signal` - decay from `__event_at__` (preferred) or `__updated_at__` (fallback), per-slot, data-relative reference
- `graph_signal` - normalized out-degree from CSR matrix + confidence override
- `recall_signal` - retrieval frequency feedback loop

**Co-occurrence Boost:** Candidates found by BOTH vector AND BM25 get `vec_signal *= (1 + bm25_signal)`. This amplifies mutual agreement between the two search engines.

**2. Delta Transpose Bug Fix (`graphstore/core/edges.py`) - CRITICAL**

Found and fixed a bug where `get_combined_transpose_split()`, `get_combined_transpose()`, and `get_transpose()` all built the delta CSR matrix with `(tgts, srcs)` instead of `(srcs, tgts)`. This meant:
- RECALL FROM any entity returned 0 results
- HybridRAG graph expansion in REMEMBER was a complete no-op
- Spreading activation never flowed through dynamic (non-rebuilt) edges

Fix: swap to `csr_matrix((data, (srcs, tgts)), shape=(n, n))` in all 3 methods.

**3. Temporal Awareness**

- `__event_at__` reserved column on every node (event time vs ingestion time)
- `EVENT_AT` clause on CREATE NODE, UPSERT NODE, ASSERT DSL commands
- `AT` clause on REMEMBER for temporal anchoring
- `core/temporal.py` - first-class date parser supporting ISO-8601, natural dates ("8 May 2023"), relative ("3 days ago", "last Friday"), ranges ("before June 2023")
- Temporal-first retrieval: hard filter by `__event_at__` range when AT anchor present, then rank by semantics within range
- HybridRAG respects temporal filter (activation masked to filtered slots)

**4. Recency Signal Fixes**

- **Data-relative reference**: Recency decays from the NEWEST candidate in its timestamp group, not from wall clock. Prevents `exp(-1071/48) = 0` killing all historical data.
- **Separate groups for `__event_at__` and `__updated_at__`**: Mixed data doesn't corrupt. Nodes with event_at decay relative to newest event; nodes without event_at decay relative to newest updated_at.
- **`recency_half_life_days = 7300` (~20 years)**: Agent memory decays very slowly. The tuner's 48 days was tuned for LongMemEval's per-record fresh ingest, not real-world usage.

**5. RRF Fusion (`graphstore/algos/fusion.py`)**

- `rrf_remember_fusion()` - rank-based fusion over N signal arrays
- Tie-aware ranking: equal signal values get averaged ranks (no arbitrary slot-index ordering)
- `rrf_k` guard: `max(k_rrf, 1.0)` prevents division by zero
- Zero-signal candidates pushed to worst rank (`ranks[zero_mask] = n_cand`)

**6. SYS CONSOLIDATE (`graphstore/algos/consolidation.py`, `graphstore/dsl/executor_system.py`)**

- Entity-topic clustering: groups messages by entity via graph edges, clusters by cosine similarity
- Creates `observation` nodes with evidence links (no LLM needed)
- Grammar: `SYS CONSOLIDATE THRESHOLD 0.6 MIN_CLUSTER_SIZE 2`
- Tested but NOT wired into benchmark adapter (hurt results when mixed with messages)

**7. Retrieval Planner (`graphstore/retrieval/planner.py`)**

- `RetrievalContext` - captures query needs (temporal? entities? observations?)
- `RetrievalPlan` - decides what stages to enable per query
- Rule-based policy: temporal anchor -> enable temporal filter, preference keywords -> use observations, entities in query -> graph expansion
- Explicit overrides can override any decision
- Plan metadata attached to `result.meta["planner"]`

**8. Benchmark Infrastructure**

- `benchmarks/framework/llm_client.py` - litellm-based with provider fallback
- `benchmarks/framework/llm_batch.py` - async batch scheduler: splits questions across providers, 8 per batch, cooldown between rounds, early exit on no progress
- `benchmarks/framework/run_locomo.py` - official LoCoMo protocol (all convs, all Qs, per-category F1)
- `benchmarks/framework/ratchet50.py` - validated 50Q semantic test (keyword must exist in data)
- Official F1 scoring: Porter stemming, Counter (multiset), per-category handling (multi-hop splits sub-answers, adversarial checks "no information available")
- Health check before benchmark start (catches dead LLMs early)

**9. Other Bug Fixes**

- Nucleus expansion: min 20 chars text filter (skip entity/session nodes with short names), token budget enforcement, no WHERE propagation to neighbors
- `fusion_method` validation (invalid falls back to default)
- Entity extraction stop-word filter in adapter
- `type_weights` env var parsing (`GRAPHSTORE_DSL_TYPE_WEIGHTS="kind:weight,..."`)
- Adapter shadow defaults removed - config.py is single source of truth

### Config (Single Source of Truth: `graphstore/config.py`)

Key values after ratcheting:
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

---

## Part 2: General Learnings

### How Benchmarks Work

**LongMemEval-S (retrieval-only)**
- 500 records, each with ~53 sessions (~500 messages)
- Protocol: reset -> ingest ALL sessions -> query -> check if answer session ID in retrieved results
- Metric: `recall_any@k` (did we find the right session?)
- Per-record isolated evaluation (fresh GraphStore per question)
- Our result: **96.4%** with jina-nano on Kaggle GPU
- Runner: `benchmarks/framework/runner.py`

**LoCoMo (retrieval + LLM reader)**
- 10 conversations, ~200 QAs each, 1986 total
- Protocol: ingest ALL sessions for a conversation ONCE, then query ALL QAs against same state
- Metric: token-level F1 with Porter stemming (official from snap-research/locomo)
- Categories: single-hop (1), multi-hop (2), temporal (3), open-domain (4), adversarial (5)
- LLM generates answer from retrieved context, F1 computed against gold answer
- Official prompt: "write an answer in the form of a short phrase, answer with exact words from the context"
- Category 5 (adversarial): check for "no information available" in response
- Multi-hop (category 1): split comma-separated sub-answers, partial F1 for each
- Runner: `benchmarks/framework/run_locomo.py`

**Key difference**: LongMemEval tests "can you find it?" LoCoMo tests "can you find it AND extract the answer?"

### What We Learned About Retrieval

**1. Vector similarity dominates ranking.** With jina-v5-small (1024d), vec_signal accounts for ~80% of ranking decisions. BM25 and graph signals are secondary. This is correct - semantic search is the primary retrieval mechanism.

**2. BM25 catches what vector misses, but rarely wins fusion.** BM25 finds exact keyword matches perfectly ("adoption", "sweden", "counseling"). But the vec_signal for generic related messages outscores the BM25 signal for specific keyword matches. The co-occurrence boost partially fixes this.

**3. The semantic gap is real.** "Where did Caroline move from?" -> answer "Sweden" but the content says "received a necklace from Sweden." Neither vector similarity ("move from" != "necklace from") nor BM25 ("move" != "necklace") bridges this gap. Only entity traversal (Caroline -> messages mentioning Sweden) or query expansion (add "origin country") can solve this.

**4. Recency with short half-life kills historical data.** The tuner found `half_life=48 days` optimal for LongMemEval (per-record fresh ingest). But for LoCoMo (data from May-Oct 2023, queried in 2026), this makes `exp(-1071/48) = 0`. Data-relative reference + 20-year half-life fixed this.

**5. Graph signal helps despite looking weak.** Disabling graph_signal drops recall by 3 points (40->37). The normalized out-degree gives a mild but consistent boost to well-connected messages (those mentioning multiple entities or having many conversation links).

**6. RRF is worse than weighted for this data.** RRF equalizes all signals which hurts when graph and recall signals are near-zero for most candidates. Weighted fusion lets vec_signal dominate appropriately.

**7. Embedder size matters more than fusion tuning.** jina-nano (768d) -> jina-small (1024d) gained +5 at top-10. No fusion config change gained more than +3.

**8. Top-k to LLM matters a lot.** top-5: 60%, top-10: 80%, top-20: 84%. Diminishing returns after 10.

### Retrieval Recall Ratchet Results

50 validated questions (keyword exists in ingested data), jina-v5-small 1024d:

| K | Recall |
|---|---|
| top-5 | 30/50 (60%) |
| top-10 | 40/50 (80%) |
| top-20 | 42/50 (84%) |
| top-50 | 48/50 (96%) |

**Ceiling: 48/50 (96%).** The retrieval layer CAN find almost everything. The problem is ranking - answers at rank 20-50 need to be pushed into top-10.

### LoCoMo F1 Results

| Config | LLM | F1 |
|---|---|---|
| MiniMax + jina-nano (old baseline, 199Q) | MiniMax M2.7 Ollama | 0.399 |
| MiniMax nitro + jina-small (50Q) | MiniMax M2.7 OpenRouter | 0.357 |
| Gemma4 + jina-small (50Q) | Gemma4 31B | 0.145 |
| GPT-3.5-turbo full-context (paper) | GPT-3.5-turbo-16k | 0.378 |
| Human baseline (paper) | - | 0.879 |

### What SOTA Systems Do Differently

| System | Key Technique | LoCoMo Score |
|---|---|---|
| MemMachine | Sentence-level embeddings, neighbor expansion, reranking | 0.917 (LLM-judge) |
| Synapse | Spreading activation (like our RECALL) | +7.2 F1 over baseline |
| TSM | Temporal Knowledge Graph, durative memory | 0.767 F1 (GPT-4o-mini) |
| Hindsight | TEMPR (4 parallel search), disposition traits | - |

Key techniques we DON'T have:
- **Memory consolidation with LLM** - extract facts from episodes, merge into profiles
- **Query decomposition** - split complex queries into sub-queries
- **Cross-encoder reranking** - we have FlashRank but it hurt LoCoMo (observation-style text confuses it)

### What We Should Aim For Next

**Immediate (no LLM needed):**
1. Improve recall@10 from 80% to 90%+ by better ranking of found results
2. The 8 answers that ARE in top-50 but NOT in top-10 need targeted ranking fixes
3. Run full 1986Q LoCoMo on OpenRouter MiniMax nitro (~$0.40) for official numbers
4. Run updated LongMemEval-S on Kaggle with jina-v5-small to verify no regression

**Medium-term (may need LLM at ingest time):**
1. Memory consolidation: extract entity profiles from episodes (SYS CONSOLIDATE is built, needs better integration)
2. Query expansion: rewrite ambiguous queries using retrieved context
3. Temporal NL parsing in core DSL (not just benchmark adapter)

**Long-term:**
1. Bitemporal model (event time + knowledge time, like Zep)
2. Sentence-level embeddings (like MemMachine) instead of message-level
3. LLM-as-judge scoring for LoCoMo (comparable to published SOTA numbers)

### How to Run Benchmarks

**LoCoMo (local, 50Q fast test):**
```bash
# Set QA_MODEL in benchmarks/framework/llm_client.py
# Then:
uv run python3 -m benchmarks.framework.ratchet50
```

**LoCoMo (local, full 1986Q):**
```bash
uv run python3 -m benchmarks.framework.run_locomo \
  --data-path /tmp/locomo \
  --embedder installed:jina-v5-small-retrieval \
  --k 10
```

**LongMemEval-S (Kaggle):**
```bash
# Update benchmarks/kaggle/graphstore_jina_500.py with jina-v5-small config
kaggle kernels push -p benchmarks/kaggle
```

**Retrieval recall test (no LLM needed):**
```bash
uv run python3 -m benchmarks.framework.ratchet50
```

### LLM Provider Setup

Config: `autoresearch/config.json` (gitignored, contains API keys)

Providers:
- **Ollama cloud** (localhost:11434): gemma4:31b-cloud, minimax-m2.7:cloud, qwen3.5:cloud
- **OpenRouter** (openrouter.ai): minimax/minimax-m2.7:nitro (paid, reliable), google/gemma-4-31b-it:free (rate-limited)

The async batch scheduler (`benchmarks/framework/llm_batch.py`) splits questions across providers, 8 per batch, with 5s cooldown between rounds and early exit after 2 rounds of no progress.

**Important: thinking models (MiniMax, Qwen3.5) need `max_tokens >= 500` or they spend all tokens on reasoning and return empty content.**

### Key Files

| File | Purpose |
|---|---|
| `graphstore/config.py` | Single source of truth for all config |
| `graphstore/dsl/handlers/intelligence.py` | REMEMBER pipeline (the heart of retrieval) |
| `graphstore/core/edges.py` | CSR matrices + dynamic edge buffer |
| `graphstore/algos/fusion.py` | RRF + weighted fusion + recency decay |
| `graphstore/algos/spreading.py` | Graph spreading activation |
| `graphstore/core/temporal.py` | Date parser |
| `graphstore/algos/consolidation.py` | Entity-topic clustering |
| `graphstore/retrieval/planner.py` | Query-aware retrieval planner |
| `benchmarks/framework/run_locomo.py` | LoCoMo benchmark runner |
| `benchmarks/framework/llm_batch.py` | Async dual-provider LLM scheduler |
| `benchmarks/framework/llm_client.py` | LLM client with litellm |
| `benchmarks/framework/ratchet50.py` | 50Q retrieval recall test |
| `benchmarks/framework/adapters/graphstore_.py` | Benchmark adapter (bridges GraphStore to benchmark protocol) |
| `benchmarks/framework/datasets.py` | LongMemEval + LoCoMo data loaders |
