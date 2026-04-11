"""Docker entrypoint for the benchmark framework.

Reads benchmark config from CLI args, runs against the mounted dataset
at /data, and writes results to /results inside the container.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .adapters import get_adapter
from .datasets import load_longmemeval
from .report import write_csv, write_json, write_markdown
from .runner import run_benchmark


def main() -> int:
    p = argparse.ArgumentParser(prog="docker_runner")
    p.add_argument("--system", default="graphstore")
    p.add_argument("--dataset", default="longmemeval", choices=["longmemeval"])
    p.add_argument("--data-path", default="/data/longmemeval")
    p.add_argument("--variant", default="s", choices=["s", "m", "l"])
    p.add_argument("--out-dir", default="/results")
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--start", type=int, default=0,
                   help="skip this many records from the top of the file")
    p.add_argument("--categories", default="",
                   help="comma-separated question_type filter, e.g. 'multi-session,temporal-reasoning'")
    p.add_argument("--per-category", type=int, default=None,
                   help="at most this many records per category")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--embedder", default="fastembed",
                   choices=["model2vec", "fastembed", "onnx", "installed"])
    p.add_argument("--embedder-model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--embedder-model-dir", default=None,
                   help="local dir for onnx embedder (tokenizer.json + onnx/*.onnx)")
    p.add_argument("--embedder-cache-dir", default=None,
                   help="graphstore registry cache root (for --embedder installed)")
    p.add_argument("--embedder-output-dims", type=int, default=None)
    p.add_argument("--embedder-max-length", type=int, default=512)
    p.add_argument("--embedder-pooling", default="mean",
                   choices=["mean", "last_token"])
    p.add_argument("--embedder-threads", type=int, default=None)
    p.add_argument("--gpu", action="store_true",
                   help="enable onnxruntime CUDA provider for ONNX embedders "
                        "(requires graphstore[gpu] install)")
    p.add_argument("--ceiling-mb", type=int, default=3072)
    p.add_argument("--cache-dir", default="/cache/fastembed")
    p.add_argument("--run-tag", default="")
    args = p.parse_args()

    if args.dataset != "longmemeval":
        print(f"unknown dataset {args.dataset}", file=sys.stderr)
        return 2

    cats = {c.strip() for c in args.categories.split(",") if c.strip()} or None
    dataset = load_longmemeval(
        args.data_path,
        variant=args.variant,
        max_records=args.max_records,
        start=args.start,
        categories=cats,
        per_category=args.per_category,
    )
    print(f"loaded {len(dataset)} records from {dataset.name}")
    if cats:
        print(f"categories filter: {sorted(cats)}")
    if args.per_category:
        print(f"per-category cap: {args.per_category}")
    if args.start:
        print(f"start offset: {args.start}")

    adapter_cls = get_adapter(args.system)
    adapter_config = {
        "embedder": args.embedder,
        "embedder_model": args.embedder_model,
        "embedder_model_dir": args.embedder_model_dir,
        "embedder_cache_dir": args.embedder_cache_dir,
        "embedder_output_dims": args.embedder_output_dims,
        "embedder_max_length": args.embedder_max_length,
        "embedder_pooling": args.embedder_pooling,
        "embedder_threads": args.embedder_threads,
        "embedder_gpu": args.gpu,
        "ceiling_mb": args.ceiling_mb,
        "cache_dir": args.cache_dir,
    }
    adapter = adapter_cls(config=adapter_config)
    print(f"system: {adapter.name} v{adapter.version}")
    print(f"config: {adapter_config}")

    result = run_benchmark(
        adapter, dataset,
        k=args.k,
        max_records=args.max_records,
        config=adapter_config,
        progress_every=10,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = result.started_at.replace(":", "-")[:19]
    tag = f"_{args.run_tag}" if args.run_tag else ""
    n_part = f"_n{args.max_records}" if args.max_records else "_full"
    prefix = out_dir / (
        f"{adapter.name}_{args.dataset}_{args.variant}{n_part}{tag}_{stamp}"
    )
    write_json([result], prefix.with_suffix(".json"))
    write_csv([result], prefix.with_suffix(".csv"))
    write_markdown([result], prefix.with_suffix(".md"))

    print()
    print("RESULTS")
    print(f"  overall acc    {result.quality.accuracy:.3f}")
    print(f"  overall r@5    {result.quality.recall_at_k:.3f}")
    print(f"  query p50      {result.latency_query.p50:.1f} ms")
    print(f"  query p95      {result.latency_query.p95:.1f} ms")
    print(f"  query p99      {result.latency_query.p99:.1f} ms")
    print(f"  ingest mean    {result.latency_ingest.mean:.1f} ms")
    print(f"  peak RSS       {result.memory.rss_peak_mb:.0f} MB")
    print(f"  elapsed        {result.total_elapsed_s:.0f} s")
    if result.quality._categories:
        print("  by category:")
        for cat, b in sorted(result.quality._categories.items()):
            print(f"    {cat:<30} n={b.n:<3} acc={b.accuracy:.3f} r@k={b.recall_at_k:.3f}")
    print()
    print(f"results: {prefix}.{{json,csv,md}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
