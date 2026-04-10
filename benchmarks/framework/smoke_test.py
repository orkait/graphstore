"""Smoke test the framework end-to-end on a tiny in-memory dataset.

Run:
    python -m benchmarks.framework.smoke_test

Not a real benchmark. Validates the adapter protocol, per-record
isolated runner, metrics collection, and reporter end-to-end without
needing an external dataset download.
"""

from __future__ import annotations

from pathlib import Path

from .adapter import Message, Session
from .adapters import get_adapter
from .datasets import BenchmarkDataset, BenchmarkQuestion, BenchmarkRecord
from .report import write_csv, write_json, write_markdown
from .runner import run_benchmark


def _make_record(qid: str, question: str, gold: list[str], category: str,
                 sessions: list[Session]) -> BenchmarkRecord:
    return BenchmarkRecord(
        question=BenchmarkQuestion(
            question=question,
            gold_answers=gold,
            category=category,
            metadata={"question_id": qid},
        ),
        sessions=sessions,
    )


def _tiny_dataset() -> BenchmarkDataset:
    records = [
        _make_record(
            "q1", "Where does the user live?", ["Tokyo"], "single-session-user",
            [Session("s1", [
                Message("user", "I live in Tokyo and work as a data scientist."),
                Message("assistant", "Got it, Tokyo-based data scientist."),
                Message("user", "I take the Marunouchi line every morning."),
                Message("assistant", "Marunouchi line commute noted."),
            ])],
        ),
        _make_record(
            "q2", "What is the user's favorite food?", ["tonkotsu", "ramen"],
            "single-session-preference",
            [Session("s2", [
                Message("user", "My favorite food is tonkotsu ramen."),
                Message("assistant", "Tonkotsu ramen is a great choice."),
                Message("user", "I eat it at least once a week."),
                Message("assistant", "Weekly ramen tradition noted."),
            ])],
        ),
        _make_record(
            "q3", "What is the name of the user's cat?", ["Miso", "Natto"],
            "multi-session",
            [
                Session("s3a", [
                    Message("user", "I adopted a cat named Miso last year."),
                    Message("assistant", "Miso is a lovely name."),
                ]),
                Session("s3b", [
                    Message("user", "I just adopted a second cat named Natto."),
                    Message("assistant", "Natto joins Miso."),
                ]),
            ],
        ),
        _make_record(
            "q4", "Where is the user planning to travel?", ["Hokkaido"],
            "single-session-user",
            [Session("s4", [
                Message("user", "I'm planning a trip to Hokkaido in December."),
                Message("assistant", "December Hokkaido trip noted."),
                Message("user", "I want to try fresh uni there."),
                Message("assistant", "Uni in Hokkaido, confirmed."),
            ])],
        ),
        _make_record(
            "q5", "How old is the user's cat?", ["seven", "7"], "temporal-reasoning",
            [Session("s5", [
                Message("user", "My cat Miso is seven years old."),
                Message("assistant", "Miso, age seven, logged."),
            ])],
        ),
    ]
    return BenchmarkDataset(name="smoke-test", records=records)


def main() -> int:
    print("=" * 60)
    print("Benchmark framework smoke test")
    print("=" * 60)

    dataset = _tiny_dataset()
    print(f"Dataset: {dataset.name} ({len(dataset)} records)")

    adapter_cls = get_adapter("graphstore")
    adapter = adapter_cls(config={"ceiling_mb": 256, "embedder": "model2vec"})
    print(f"System:  {adapter.name} v{adapter.version}")
    print()

    result = run_benchmark(adapter, dataset, k=5, progress_every=2)

    out_dir = Path(__file__).parent / "results"
    prefix = out_dir / "smoke_test"
    write_json([result], prefix.with_suffix(".json"))
    write_csv([result], prefix.with_suffix(".csv"))
    write_markdown([result], prefix.with_suffix(".md"))

    print()
    print("-" * 60)
    print("Summary")
    print("-" * 60)
    print(f"  accuracy      {result.quality.accuracy:.3f}")
    print(f"  recall@5      {result.quality.recall_at_k:.3f}")
    print(f"  query p50     {result.latency_query.p50:.2f} ms")
    print(f"  query p95     {result.latency_query.p95:.2f} ms")
    print(f"  peak memory   {result.memory.rss_peak_mb:.1f} MB")
    print(f"  elapsed       {result.total_elapsed_s:.2f} s")
    if result.quality._categories:
        print("  by category:")
        for cat, b in sorted(result.quality._categories.items()):
            print(f"    {cat:<30} n={b.n} acc={b.accuracy:.3f} r@k={b.recall_at_k:.3f}")
    print()
    print(f"Results written to {prefix}.{{json,csv,md}}")

    if result.quality.accuracy < 0.5:
        print()
        print("accuracy below 0.5 - check the adapter wiring")
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
