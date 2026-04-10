"""Benchmark CLI entry point.

Usage:
    python -m benchmarks.framework.cli list
    python -m benchmarks.framework.cli run \\
        --system graphstore \\
        --dataset longmemeval \\
        --data-path ./data/longmemeval \\
        --variant s \\
        --max-questions 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .adapters import AVAILABLE, get_adapter
from .datasets import DATASET_LOADERS
from .report import write_csv, write_json, write_markdown
from .runner import run_benchmark


def cmd_list(args: argparse.Namespace) -> int:
    print("Available adapters:")
    for name in sorted(AVAILABLE):
        cls = AVAILABLE[name]
        version = getattr(cls, "version", "unknown")
        print(f"  {name:<15} v{version}")
    print()
    print("Available datasets:")
    for name in sorted(DATASET_LOADERS):
        print(f"  {name}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    adapter_cls = get_adapter(args.system)
    adapter = adapter_cls(
        config={
            "ceiling_mb": args.ceiling_mb,
            "threaded": args.threaded,
        }
    )

    if args.dataset not in DATASET_LOADERS:
        print(f"Unknown dataset: {args.dataset}", file=sys.stderr)
        return 2

    loader = DATASET_LOADERS[args.dataset]
    if args.dataset == "longmemeval":
        dataset = loader(args.data_path, variant=args.variant)
    else:
        dataset = loader(args.data_path)

    result = run_benchmark(
        adapter,
        dataset,
        k=args.k,
        max_questions=args.max_questions,
        config={
            "ceiling_mb": args.ceiling_mb,
            "threaded": args.threaded,
            "k": args.k,
            "variant": args.variant,
            "max_questions": args.max_questions,
        },
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = result.started_at.replace(":", "-")[:19]
    prefix = out_dir / f"{args.system}_{args.dataset}_{args.variant}_{stamp}"
    write_json([result], prefix.with_suffix(".json"))
    write_csv([result], prefix.with_suffix(".csv"))
    write_markdown([result], prefix.with_suffix(".md"))

    print()
    print(f"Results: {prefix}.{{json,csv,md}}")
    print(f"  accuracy    {result.quality.accuracy:.3f}")
    print(f"  recall@{args.k}    {result.quality.recall_at_k:.3f}")
    print(
        f"  latency     p50={result.latency_query.p50:.1f}ms "
        f"p95={result.latency_query.p95:.1f}ms "
        f"p99={result.latency_query.p99:.1f}ms"
    )
    print(
        f"  memory      peak={result.memory.rss_peak_mb:.1f}MB "
        f"delta={result.memory.delta_mb:.1f}MB"
    )
    print(f"  elapsed     {result.total_elapsed_s:.1f}s")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="benchmarks.framework.cli",
        description="Apples-to-apples benchmark for agent memory systems",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List available adapters and datasets")
    p_list.set_defaults(func=cmd_list)

    p_run = sub.add_parser("run", help="Run one benchmark against one system")
    p_run.add_argument("--system", required=True, choices=sorted(AVAILABLE.keys()))
    p_run.add_argument(
        "--dataset",
        default="longmemeval",
        choices=sorted(DATASET_LOADERS.keys()),
    )
    p_run.add_argument("--data-path", required=True, type=str)
    p_run.add_argument("--variant", default="s", choices=["s", "m", "l"])
    p_run.add_argument("--k", type=int, default=5)
    p_run.add_argument("--max-questions", type=int, default=None)
    p_run.add_argument("--ceiling-mb", type=int, default=2048)
    p_run.add_argument("--threaded", action="store_true")
    p_run.add_argument(
        "--out-dir",
        default="benchmarks/framework/results",
    )
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
