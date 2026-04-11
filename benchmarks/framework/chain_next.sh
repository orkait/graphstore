#!/usr/bin/env bash
set -u
LOG=/tmp/gs_bench_chain.log
{
  echo "[chain] $(date) waiting for gs_bench_overnight"
  docker wait gs_bench_overnight
  echo "[chain] $(date) bge-small finished, firing bge-base"
  docker run -d \
    --cpus=12 --memory=16g \
    -v /home/kai/graphstore-models:/data:ro \
    -v /home/kai/graphstore-models/cache:/cache \
    -v /home/kai/orkait/graphstore/graphstore/benchmarks/framework/results:/results \
    --name gs_bench_bge_base \
    graphstore-bench:latest \
    --embedder fastembed --embedder-model BAAI/bge-base-en-v1.5 \
    --run-tag overnight_bge_base_full500
  echo "[chain] $(date) bge-base started: $(docker ps --format '{{.Names}}' | grep bge_base)"
} >> "$LOG" 2>&1
