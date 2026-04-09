# LongMemEval Benchmark Harness - Design Spec

## Problem

`graphstore` has internal microbenchmarks for synthetic latency and memory usage, but it does not yet have a reproducible external benchmark that answers the important public question:

> Can `graphstore` retrieve the right long-term memory from a realistic multi-session conversation benchmark?

The benchmark we need to match is `LongMemEval`, which provides:

- 500 evaluation questions
- per-question haystack sessions with timestamps
- ground-truth evidence session IDs
- question-type labels for breakdown reporting

MemPal’s public benchmark script uses LongMemEval as a retrieval harness and publishes `Recall@k` and `NDCG@k`. `graphstore` needs the same class of benchmark so results are externally comparable and auditable.

## Goals

1. Add a reproducible `LongMemEval` retrieval benchmark under the `graphstore` repo.
2. Exercise `graphstore`’s real retrieval surfaces, not a benchmark-only side index.
3. Report auditable retrieval metrics at both session and turn level.
4. Support quick local smoke runs and full-dataset runs.
5. Produce logs that are easy to inspect, diff, and publish.

## Non-Goals

1. End-to-end answer generation for v1.
2. Additional benchmark suites such as LoCoMo or ConvoMem in the first implementation.
3. Benchmark-specific retrieval heuristics that only exist inside the harness and cannot be mapped back to `graphstore` capabilities.
4. Large-scale ingestion features such as graph wiring, edge creation, or document parsing pipelines during benchmark loading.

## Recommended Solution

Add a dedicated benchmark package at repo root:

- `benchmarks/longmemeval.py` - CLI runner and retrieval harness
- `benchmarks/README.md` - reproduction instructions
- `tests/test_longmemeval_benchmark.py` - dataset parsing, corpus shaping, metric, and CLI smoke tests

The runner will ingest each LongMemEval question’s haystack into a fresh temporary `GraphStore`, retrieve top-k memories using one of several retrieval modes, score those rankings against LongMemEval evidence session IDs, and emit JSONL + JSON summary outputs.

## Why This Shape

This is the smallest design that produces an external benchmark with public value.

- It uses `graphstore` itself as the retrieval engine.
- It avoids contaminating results with reader-LLM variability.
- It matches the style of rival benchmarks closely enough for comparison.
- It keeps the implementation bounded enough to ship quickly and test properly.

## Benchmark Flow

For each LongMemEval entry:

1. Load one dataset row from `longmemeval_s_cleaned.json`.
2. Create a fresh temporary `GraphStore(path=<tempdir>)`.
3. Register a benchmark node kind with one embedding field.
4. Materialize the haystack as either session documents or turn documents.
5. Populate `DocumentStore` summaries so BM25-backed retrieval paths are real.
6. Run the selected retrieval mode against the benchmark question.
7. Convert retrieved nodes to ranked corpus IDs.
8. Score the ranking with retrieval metrics.
9. Append one JSONL audit record.
10. Close and delete the temporary store before the next question.

Per-question isolation is intentional. It mirrors the benchmark’s structure, avoids cross-question leakage, and keeps each run auditable.

## Corpus Mapping

### Shared Metadata

Every indexed benchmark item should preserve:

- `question_id`
- `question_type`
- `session_id`
- `turn_id` when relevant
- `session_date`
- corpus text used for retrieval

### Session Granularity

One node per LongMemEval session.

- `node_id`: `session:<session_id>`
- `kind`: `benchmark_memory`
- `text`: concatenated session text
- `summary`: same retrieval text or a trimmed version
- `document`: full retrieval text

Default session text should include **all turns**, not just user turns. LongMemEval evidence can live in assistant turns, and `graphstore` should benchmark retrieval over the memory actually available to the agent.

### Turn Granularity

One node per turn.

- `node_id`: `turn:<session_id>:<turn_index>`
- retains `session_id` for session-level remapping

Default turn text should also include all turns individually. This keeps the turn benchmark aligned with the actual evidence-bearing units.

## Retrieval Modes

V1 should support four modes:

1. `remember`
Uses `REMEMBER "<question>" LIMIT <k>`.
This is the default and the primary benchmark mode because it is `graphstore`’s intended hybrid retrieval path.

2. `similar`
Uses `SIMILAR TO "<question>" LIMIT <k>`.
This is the dense-only baseline.

3. `lexical`
Uses `LEXICAL SEARCH "<question>" LIMIT <k>`.
This is the BM25-only baseline.

4. `hybrid`
Runs both `SIMILAR TO` and `LEXICAL SEARCH`, then performs simple reciprocal-rank or score fusion in the benchmark harness.
This is a sanity baseline against `REMEMBER`, not a replacement for it.

## Important Implementation Constraint

`graphstore`’s BM25 path currently searches `DocumentStore` summaries, and plain `CREATE NODE ... summary = "..."` does not automatically populate `DocumentStore`.

Therefore the harness must explicitly populate `DocumentStore.put_summary(...)` after creating each benchmark node. This is acceptable because:

- it uses `graphstore`’s real document index
- it avoids fake BM25 numbers from an unindexed field
- it keeps the harness transparent about what storage layer is being exercised

The harness should also use the DSL or public object methods for the actual retrieval query execution.

## Metrics

Report:

- session `Recall@5`
- session `Recall@10`
- session `NDCG@10`
- turn `Recall@5`
- turn `Recall@10`
- turn `NDCG@10`

Internally compute:

- `recall_any@k`
- `recall_all@k`
- `ndcg_any@k`

Public console output can stay compact, but JSON outputs should retain the fuller metric set.

## Abstention Handling

LongMemEval’s retrieval guidance treats abstention items as special because they often have no real evidence location.

V1 should support both:

- default: skip abstention items for retrieval scoring
- flag: `--include-abstention`

The summary output should always report how many questions were scored vs skipped.

## CLI

Initial CLI shape:

```bash
python benchmarks/longmemeval.py path/to/longmemeval_s_cleaned.json
python benchmarks/longmemeval.py path/to/longmemeval_s_cleaned.json --mode remember
python benchmarks/longmemeval.py path/to/longmemeval_s_cleaned.json --mode lexical
python benchmarks/longmemeval.py path/to/longmemeval_s_cleaned.json --mode hybrid
python benchmarks/longmemeval.py path/to/longmemeval_s_cleaned.json --granularity turn
python benchmarks/longmemeval.py path/to/longmemeval_s_cleaned.json --limit 20
python benchmarks/longmemeval.py path/to/longmemeval_s_cleaned.json --top-k 10
python benchmarks/longmemeval.py path/to/longmemeval_s_cleaned.json --include-abstention
```

Arguments:

- positional dataset path
- `--mode {remember,similar,lexical,hybrid}`
- `--granularity {session,turn}`
- `--limit N`
- `--top-k N`
- `--include-abstention`
- `--out <jsonl-path>`

## Output Format

### JSONL Audit Log

One record per scored question:

```json
{
  "question_id": "abc123",
  "question_type": "multi-session",
  "question": "What city did I say I moved to last year?",
  "answer": "Seattle",
  "answer_session_ids": ["sess_18"],
  "retrieval_results": {
    "mode": "remember",
    "granularity": "session",
    "ranked_items": [
      {
        "corpus_id": "session:sess_18",
        "session_id": "sess_18",
        "turn_id": null,
        "score": 0.91,
        "text_preview": "..."
      }
    ],
    "metrics": {
      "session": {
        "recall_any@5": 1.0,
        "recall_any@10": 1.0,
        "ndcg_any@10": 0.93
      },
      "turn": {
        "recall_any@5": 1.0,
        "recall_any@10": 1.0,
        "ndcg_any@10": 0.88
      }
    }
  }
}
```

### Summary JSON

One aggregate file with:

- dataset path
- mode
- granularity
- question count
- scored count
- skipped count
- aggregate metrics
- per-question-type breakdown
- run timestamp
- elapsed time

## Testing Strategy

Add focused tests with a tiny synthetic LongMemEval fixture:

1. Dataset loader parses the expected schema.
2. Session corpus builder preserves `session_id` and all-turn text.
3. Turn corpus builder preserves `session_id` and `turn_id`.
4. Session-level remapping from turn IDs works.
5. Metric helpers return expected values for known rankings.
6. Abstention filtering behaves correctly.
7. CLI smoke test runs a tiny fixture and writes output files.

Tests should use the existing deterministic mock embedder pattern from `tests/test_integration_fixtures.py` so benchmark tests stay offline and stable.

## Documentation

`benchmarks/README.md` should include:

- what the benchmark measures
- dataset download commands
- quick smoke command
- full benchmark command
- explanation of modes and granularity
- description of output files
- note that QA generation is intentionally out of scope for v1

## Risks and Mitigations

### Risk: benchmark numbers are weak because BM25 is not actually indexed

Mitigation:
Populate `DocumentStore` summaries explicitly during corpus loading and test that lexical mode returns indexed rows on the sample fixture.

### Risk: benchmark becomes a benchmark-only code path

Mitigation:
All retrieval execution should still go through `GraphStore.execute(...)` for the public query surface. Only corpus preparation may touch `DocumentStore` summary loading directly.

### Risk: default embedder download or initialization makes smoke runs slow

Mitigation:
Use a small test fixture with a deterministic mock embedder in tests. Document that the real benchmark uses the normal runtime embedder path.

### Risk: comparisons with rival results differ because of abstention treatment or turn shaping

Mitigation:
Document defaults clearly, keep flags explicit, and preserve raw JSONL so any scoring choice can be audited later.

## Acceptance Criteria

The design is complete when:

1. A user can run a single command against `longmemeval_s_cleaned.json`.
2. The command emits aggregate session and turn retrieval metrics.
3. The run writes auditable JSONL results.
4. The harness supports at least `remember`, `similar`, `lexical`, and `hybrid`.
5. The implementation is covered by deterministic tests using a tiny fixture.
