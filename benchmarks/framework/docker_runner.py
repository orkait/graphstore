"""Docker entrypoint for the benchmark framework.

Reads benchmark config from CLI args, runs against the mounted dataset
at /data, and writes results to /results inside the container.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path

from .adapters import get_adapter
from .datasets import load_longmemeval
from .report import write_csv, write_json, write_markdown
from .runner import run_benchmark


def _is_mount(path: str) -> bool:
    """Check if a path is a mount point (volume-backed in Docker)."""
    try:
        with open("/proc/mounts") as f:
            mounts = f.read()
        return any(f" {path} " in line for line in mounts.splitlines())
    except (FileNotFoundError, PermissionError):
        return os.path.ismount(path)


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
                   choices=["model2vec", "fastembed", "onnx", "installed", "gguf"])
    p.add_argument("--embedder-model", default=None)
    p.add_argument("--embedder-model-dir", default=None,
                   help="local dir for onnx embedder (tokenizer.json + onnx/*.onnx)")
    p.add_argument("--embedder-cache-dir", default=None,
                   help="graphstore registry cache root (for --embedder installed)")
    p.add_argument("--embedder-output-dims", type=int, default=None)
    p.add_argument("--embedder-max-length", type=int, default=512)
    p.add_argument("--embedder-pooling", default="mean",
                   choices=["mean", "last_token"])
    p.add_argument("--embedder-threads", type=int, default=None)
    p.add_argument("--embedder-gguf-path", default=None,
                   help="path to .gguf file for gguf embedder")
    p.add_argument("--embedder-gpu-layers", type=int, default=0,
                   help="n_gpu_layers for gguf embedder (-1 = all)")
    p.add_argument("--embedder-query-prefix", default="",
                   help="query prefix for gguf embedder")
    p.add_argument("--gpu", action="store_true",
                   help="enable onnxruntime CUDA provider for ONNX embedders")
    p.add_argument("--gpu-mem-limit-gb", type=float, default=None,
                   help="GPU VRAM limit in GB for ONNX CUDA provider")
    p.add_argument("--embed-batch-size", type=int, default=128,
                   help="batch size for deferred embeddings (lower = less VRAM)")
    p.add_argument("--ceiling-mb", type=int, default=3072)
    p.add_argument("--cache-dir", default="/cache/fastembed")
    p.add_argument("--run-tag", default="")

    # GraphStore retrieval tuning
    p.add_argument("--retrieval-depth", type=int, default=4,
                   help="REMEMBER candidate multiplier (LIMIT k*depth, default 4)")
    p.add_argument("--recall-depth", type=int, default=2,
                   help="graph traversal depth for RECALL in multi-session queries")
    p.add_argument("--max-query-entities", type=int, default=3,
                   help="max entities extracted from query for RECALL")
    p.add_argument("--recency-boost-k", type=int, default=1,
                   help="multiplier for recency-sorted results in knowledge-update (k * this)")
    p.add_argument("--remember-weights", default=None,
                   help="5 comma-separated fusion weights: vector,bm25,recency,confidence,recall_freq")
    p.add_argument("--search-oversample", type=int, default=None,
                   help="usearch ANN oversample factor (default 5, higher = more candidates)")
    p.add_argument("--recall-decay", type=float, default=None,
                   help="spreading activation decay per hop (default 0.7)")
    p.add_argument("--similarity-threshold", type=float, default=None,
                   help="minimum cosine similarity floor (default 0.85)")
    args = p.parse_args()

    # --- Validate args early ---

    if not args.embedder_model:
        if args.embedder_model_dir:
            args.embedder_model = Path(args.embedder_model_dir).name
        elif args.embedder_gguf_path:
            args.embedder_model = Path(args.embedder_gguf_path).stem
        elif args.embedder == "installed":
            print("error: --embedder installed requires --embedder-model", file=sys.stderr)
            return 2
        else:
            args.embedder_model = args.embedder

    if args.embedder == "onnx" and args.embedder_model_dir:
        model_dir = Path(args.embedder_model_dir)
        if not model_dir.exists():
            print(f"error: embedder model dir not found: {model_dir}", file=sys.stderr)
            print("hint: did you mount the model volume? (-v /host/path:/container/path:ro)", file=sys.stderr)
            return 2

    if args.embedder == "gguf" and args.embedder_gguf_path:
        gguf_path = Path(args.embedder_gguf_path)
        if not gguf_path.exists():
            print(f"error: gguf model not found: {gguf_path}", file=sys.stderr)
            print("hint: did you mount the model volume?", file=sys.stderr)
            return 2

    if args.gpu_mem_limit_gb and not args.gpu:
        print("warning: --gpu-mem-limit-gb has no effect without --gpu", file=sys.stderr)

    out_dir = Path(args.out_dir)
    if os.path.exists("/.dockerenv") and not _is_mount(args.out_dir):
        print(f"warning: {args.out_dir} is not a mounted volume - results will be lost when container exits", file=sys.stderr)
        print(f"hint: add -v $(pwd)/results:{args.out_dir} to your docker run command", file=sys.stderr)

    # --- Load dataset ---

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

    if len(dataset) == 0:
        print("warning: 0 records after filtering - nothing to benchmark", file=sys.stderr)
        return 0

    # --- Build adapter ---

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
        "embedder_gpu_mem_limit": int(args.gpu_mem_limit_gb * 1024**3) if args.gpu_mem_limit_gb else None,
        "embedder_gguf_path": args.embedder_gguf_path,
        "embedder_gpu_layers": args.embedder_gpu_layers,
        "embedder_query_prefix": args.embedder_query_prefix,
        "embed_batch_size": args.embed_batch_size,
        "ceiling_mb": args.ceiling_mb,
        "retrieval_depth": args.retrieval_depth,
        "recall_depth": args.recall_depth,
        "max_query_entities": args.max_query_entities,
        "recency_boost_k": args.recency_boost_k,
        "remember_weights": [float(w) for w in args.remember_weights.split(",")] if args.remember_weights else None,
        "search_oversample": args.search_oversample,
        "recall_decay": args.recall_decay,
        "similarity_threshold": args.similarity_threshold,
        "cache_dir": args.cache_dir,
    }
    if args.system != "graphstore" and args.embedder in ("gguf", "onnx"):
        if args.embedder == "gguf":
            from graphstore.embedding.llamacpp_embedder import LlamaCppEmbedder
            adapter_config["_embedder_instance"] = LlamaCppEmbedder(
                model_path=args.embedder_gguf_path,
                n_ctx=args.embedder_max_length,
                n_gpu_layers=args.embedder_gpu_layers,
                output_dims=args.embedder_output_dims,
                query_prefix=args.embedder_query_prefix,
            )
        elif args.embedder == "onnx" and args.embedder_model_dir:
            from graphstore.embedding.onnx_hf_embedder import OnnxHFEmbedder
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.gpu else None
            adapter_config["_embedder_instance"] = OnnxHFEmbedder(
                model_dir=args.embedder_model_dir,
                output_dims=args.embedder_output_dims,
                max_length=args.embedder_max_length,
                pooling_mode=args.embedder_pooling,
                providers=providers,
                gpu_mem_limit=adapter_config.get("embedder_gpu_mem_limit"),
            )
    adapter = adapter_cls(config=adapter_config)
    print(f"system: {adapter.name} v{adapter.version}")
    print(f"config: {adapter_config}")

    # Strip non-serializable objects from config before passing to runner
    serializable_config = {k: v for k, v in adapter_config.items() if not k.startswith("_")}

    # --- Run benchmark with partial-result safety ---

    def _save_results(result, tag_suffix=""):
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = result.started_at.replace(":", "-")[:19]
        tag = f"_{args.run_tag}" if args.run_tag else ""
        n_part = f"_n{args.max_records}" if args.max_records else "_full"
        prefix = out_dir / (
            f"{adapter.name}_{args.dataset}_{args.variant}{n_part}{tag}{tag_suffix}_{stamp}"
        )
        write_json([result], prefix.with_suffix(".json"))
        write_csv([result], prefix.with_suffix(".csv"))
        write_markdown([result], prefix.with_suffix(".md"))
        return prefix

    result = run_benchmark(
        adapter, dataset,
        k=args.k,
        max_records=args.max_records,
        config=serializable_config,
        progress_every=10,
        on_interrupt=lambda partial: _save_results(partial, "_partial"),
    )

    prefix = _save_results(result)

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
