# Agent memory benchmark framework

Apples-to-apples benchmarking for agent memory systems. One adapter per system, one runner, one set of metrics, one leaderboard. No cherry-picking, no shifting baselines.

## Why this exists

Every agent memory vendor publishes benchmark numbers, but they use different LLMs, different embedders, different benchmark versions, different evaluation protocols, and different hardware. The result is a pile of scores nobody can actually compare.

This framework fixes the variables:

- **One dataset per run.** LongMemEval-S today, LoCoMo and AMB next.
- **One metric contract.** Accuracy, R@K, p50/p95/p99 latency, peak RSS, tokens consumed.
- **One LLM at a time.** `gpt-4o-mini` by default, pinned in the config snapshot.
- **Reproducible.** Every run writes a config snapshot, a raw JSON result, a CSV, and a Markdown leaderboard.
- **Run five times, report median.** The framework supports single runs today; a `run_all.py` multi-run driver is on the roadmap.

## Every system runs in its "best" mode

Agent memory systems are not just key-value stores. Each one ships with a recommended ingestion pipeline, query primitive, and tuning knobs that make a real difference to quality and speed. Benchmarking a system in its dumbest mode is as misleading as benchmarking it in someone else's.

This framework's ground rule is: **every adapter uses the ingestion pattern the system's docs recommend for agent memory.** No minimal pass-through, no lowest-common-denominator schemas.

For graphstore specifically, the adapter uses:

- **Schema pre-registration** — `SYS REGISTER NODE KIND "memory" REQUIRED session:string, role:string, content:string OPTIONAL importance:float, position:int EMBED content`. This pre-allocates typed columns (int32_interned / int64 / float64), validates writes, and tells graphstore which field to auto-embed. Without this, CREATE would infer column types lazily on first write — a real throughput hit at scale.
- **Deferred embeddings** — ingestion is wrapped in `gs.deferred_embeddings(batch_size=128)` so the embedder is called in batches instead of per-node. This is a 4-10x speedup on transformer embedders.
- **Hybrid retrieval via REMEMBER** — graphstore's 5-signal fusion (vector + BM25 + recency + confidence + recall frequency) is the query primitive, not bare `SIMILAR TO`.
- **Vectorized importance scoring** — per-message importance is computed in a single numpy pass over content lengths, feeding REMEMBER's confidence weight.
- **CSR lazy rebuild** — edges are accumulated raw during ingestion; the CSR sparse matrix is only built on the first read, avoiding a rebuild per-edge.

When you write a new adapter, apply the same principle: use the features the system's maintainers tell you to use.

## Supported systems

| System | Status | Notes |
|---|---|---|
| `graphstore` | Full implementation | Hybrid retrieval via `REMEMBER` |
| `mem0` | Full implementation | Requires `pip install mem0ai` |
| `letta` | Stub | Install `pip install letta`, then fill in the client calls |
| `cognee` | Planned | Graph-first, our closest cousin |
| `lightrag` | Planned | Graph RAG baseline |
| `graphiti` | Planned | Zep's OSS temporal KG |
| `mempalace` | Planned | Our prior baseline |
| `chroma-bm25` | Planned | Pure-vector + BM25 control |

## Quickstart

```bash
# Download the LongMemEval-S dataset
# (see https://github.com/xiaowu0162/LongMemEval for the data files)
mkdir -p benchmarks/framework/data/longmemeval
cp longmemeval_s.json benchmarks/framework/data/longmemeval/

# List what's installed
python -m benchmarks.framework.cli list

# Run graphstore on LongMemEval-S (first 100 questions for a smoke test)
python -m benchmarks.framework.cli run \
    --system graphstore \
    --dataset longmemeval \
    --data-path benchmarks/framework/data/longmemeval \
    --variant s \
    --max-questions 100 \
    --ceiling-mb 4096

# Run again without --max-questions for the full 500
python -m benchmarks.framework.cli run \
    --system graphstore \
    --dataset longmemeval \
    --data-path benchmarks/framework/data/longmemeval \
    --variant s \
    --ceiling-mb 4096
```

Results land in `benchmarks/framework/results/` as three files per run:

- `graphstore_longmemeval_s_<timestamp>.json` — full structured result
- `graphstore_longmemeval_s_<timestamp>.csv` — flat row for spreadsheets
- `graphstore_longmemeval_s_<timestamp>.md` — human-readable leaderboard

## What the runner measures

Every run records:

```
quality         accuracy, recall@k, (optional) llm_judge
latency_ingest  p50, p95, p99, mean, stddev of per-session ingest time
latency_query   p50, p95, p99, mean, stddev of per-query retrieval time
memory          rss_before_mb, rss_after_mb, rss_peak_mb, delta_mb
cost            ingest_tokens, query_tokens, llm_api_calls
total_elapsed_s end-to-end wall clock
```

Anything a system does not surface (e.g. token accounting for local embedders) is recorded as zero, not estimated.

## Writing a new adapter

Any system that can ingest conversation sessions and retrieve top-K memories can plug in. Implement the four-method contract in `adapter.py`:

```python
from benchmarks.framework.adapter import MemoryAdapter, Session, QueryResult, TimedOperation

class MySystemAdapter:
    name = "my-system"
    version = "1.0.0"

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        ...

    def reset(self) -> None:
        """Wipe memory. Called before every run."""
        ...

    def ingest(self, session: Session) -> float:
        """Add a session. Return elapsed seconds."""
        with TimedOperation() as t:
            for msg in session.messages:
                ... # your ingest call
        return t.elapsed_ms / 1000

    def query(self, question: str, k: int = 5) -> QueryResult:
        """Retrieve top-K. Return QueryResult with elapsed_ms populated."""
        with TimedOperation() as t:
            results = ... # your query call
        return QueryResult(retrieved_memories=[...], elapsed_ms=t.elapsed_ms)

    def close(self) -> None:
        """Release resources."""
        ...
```

Then register it in `adapters/__init__.py`:

```python
try:
    from .my_system import MySystemAdapter
    AVAILABLE["my-system"] = MySystemAdapter
except ImportError:
    pass
```

The registration is guarded by a try/except so a missing dependency on one system never blocks the others.

## Ground rules

Read these before publishing any numbers.

1. **Publish every config file.** Readers must be able to reproduce your run and any competitor's run.
2. **Publish failure cases.** Which questions did each system get wrong, and why.
3. **Publish raw latency histograms**, not just the p95.
4. **Separate quality, speed, and cost.** Nobody is SOTA on all three.
5. **Call out apples-to-oranges explicitly.** If Mem0 uses an LLM to extract during ingest and graphstore uses a static embedder, say so in the results.
6. **Run each system five times and report the median** with standard deviation. High-variance systems need the warning label.
7. **Version-pin every dependency** in the config snapshot so runs can be re-executed six months from now.

## Roadmap

- LoCoMo loader (Very Long-Term Conversational Memory)
- Agent Memory Benchmark (AMB) loader
- BEIR subset loader (pure retrieval control)
- LLM-as-judge scorer (gpt-4o-mini rubric)
- Multi-run driver with median + stddev across N passes
- Docker images per adapter for reproducibility audits
- Additional adapters: cognee, lightrag, graphiti, mempalace, chroma+bm25

## References

- [LongMemEval](https://github.com/xiaowu0162/LongMemEval) — the benchmark we start with
- [LoCoMo](https://snap-research.github.io/locomo/) — very long-term conversations
- [Agent Memory Benchmark](https://agentmemorybenchmark.ai) — the 2026 manifesto for apples-to-apples
- [GraphRAG-Bench (ICLR 2026)](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark) — for graph-based systems
