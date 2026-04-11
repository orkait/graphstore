#!/usr/bin/env bash
set -u
LOG=${GS_BENCH_CHAIN_LOG:-/tmp/gs_bench_chain.log}
REPO=${GS_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
MODELS=${GS_MODELS:?set GS_MODELS to the host path containing fastembed model cache}
WAIT_FOR=${GS_BENCH_WAIT_FOR:-gs_bench_overnight}
NEXT_NAME=${GS_BENCH_NEXT_NAME:-gs_bench_bge_base}
NEXT_MODEL=${GS_BENCH_NEXT_MODEL:-BAAI/bge-base-en-v1.5}
NEXT_TAG=${GS_BENCH_NEXT_TAG:-overnight_bge_base_full500}
{
  echo "[chain] $(date) waiting for $WAIT_FOR"
  docker wait "$WAIT_FOR"
  echo "[chain] $(date) $WAIT_FOR finished, firing $NEXT_NAME"
  docker run -d \
    --cpus=12 --memory=16g \
    -v "$MODELS:/data:ro" \
    -v "$MODELS/cache:/cache" \
    -v "$REPO/benchmarks/framework/results:/results" \
    --name "$NEXT_NAME" \
    graphstore-bench:latest \
    --embedder fastembed --embedder-model "$NEXT_MODEL" \
    --run-tag "$NEXT_TAG"
  echo "[chain] $(date) $NEXT_NAME started: $(docker ps --format '{{.Names}}' | grep "$NEXT_NAME")"
} >> "$LOG" 2>&1
