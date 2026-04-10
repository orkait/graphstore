"""Smoke test the framework end-to-end against a tiny in-memory dataset.

Run:
    python -m benchmarks.framework.smoke_test

This is NOT a real benchmark. It validates that the adapter protocol,
runner, metrics collection, and reporter all wire up correctly, without
needing an external dataset download. Use this as the CI sanity check
and as a template when writing new adapters.
"""

from __future__ import annotations

from pathlib import Path

from .adapter import Message, Session
from .adapters import get_adapter
from .datasets import BenchmarkDataset, BenchmarkQuestion
from .report import write_csv, write_json, write_markdown
from .runner import run_benchmark


def _tiny_dataset() -> BenchmarkDataset:
    sessions = [
        Session(
            session_id="s1",
            messages=[
                Message(role="user", content="I live in Tokyo and work as a data scientist."),
                Message(role="assistant", content="Got it, Tokyo-based data scientist."),
                Message(role="user", content="My favorite food is ramen, specifically tonkotsu."),
                Message(role="assistant", content="Tonkotsu ramen noted."),
            ],
        ),
        Session(
            session_id="s2",
            messages=[
                Message(role="user", content="I have a pet cat named Miso. She is seven years old."),
                Message(role="assistant", content="Miso, age seven, logged."),
                Message(role="user", content="I just adopted a second cat named Natto."),
                Message(role="assistant", content="Welcome Natto."),
            ],
        ),
        Session(
            session_id="s3",
            messages=[
                Message(role="user", content="I'm planning a trip to Hokkaido in December."),
                Message(role="assistant", content="December Hokkaido trip noted."),
                Message(role="user", content="I really want to try uni (sea urchin) there."),
                Message(role="assistant", content="Uni in Hokkaido, got it."),
            ],
        ),
    ]
    questions = [
        BenchmarkQuestion(
            question="Where does the user live?",
            gold_answers=["Tokyo"],
            category="single-hop",
        ),
        BenchmarkQuestion(
            question="What is the user's favorite food?",
            gold_answers=["ramen", "tonkotsu"],
            category="single-hop",
        ),
        BenchmarkQuestion(
            question="What is the name of the user's cat?",
            gold_answers=["Miso", "Natto"],
            category="single-hop",
        ),
        BenchmarkQuestion(
            question="Where is the user planning to travel?",
            gold_answers=["Hokkaido"],
            category="single-hop",
        ),
        BenchmarkQuestion(
            question="How old is the cat?",
            gold_answers=["seven", "7"],
            category="temporal",
        ),
    ]
    return BenchmarkDataset(
        name="smoke-test",
        sessions=sessions,
        questions=questions,
    )


def main() -> int:
    print("=" * 60)
    print("Benchmark framework smoke test")
    print("=" * 60)

    dataset = _tiny_dataset()
    print(f"Dataset: {dataset.name} ({len(dataset.sessions)} sessions, "
          f"{len(dataset.questions)} questions)")

    adapter_cls = get_adapter("graphstore")
    adapter = adapter_cls(config={"ceiling_mb": 256})
    print(f"System:  {adapter.name} v{adapter.version}")
    print()

    result = run_benchmark(adapter, dataset, k=5, progress_every=10)

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
    print()
    print(f"Results written to {prefix}.{{json,csv,md}}")

    if result.quality.accuracy < 0.5:
        print()
        print("⚠️  accuracy below 0.5 — check the adapter wiring")
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
