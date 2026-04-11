#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

MODE="${1:-run}"
shift || true

case "$MODE" in
  run)
    .venv/bin/python -m pytest benchmarks/algos/ \
      --benchmark-only \
      --benchmark-autosave \
      --benchmark-columns=mean,min,max,stddev,rounds \
      --benchmark-sort=mean \
      --benchmark-group-by=func \
      "$@"
    ;;
  baseline)
    .venv/bin/python -m pytest benchmarks/algos/ \
      --benchmark-only \
      --benchmark-save=baseline \
      --benchmark-columns=mean,min,max,stddev,rounds \
      --benchmark-group-by=func \
      "$@"
    ;;
  compare)
    .venv/bin/python -m pytest benchmarks/algos/ \
      --benchmark-only \
      --benchmark-compare=baseline \
      --benchmark-compare-fail=mean:10% \
      --benchmark-columns=mean,min,max,stddev,rounds \
      --benchmark-group-by=func \
      "$@"
    ;;
  gate)
    .venv/bin/python -m pytest benchmarks/algos/ \
      --benchmark-only \
      --benchmark-compare=baseline \
      --benchmark-compare-fail=mean:5% \
      --benchmark-columns=mean,min,max \
      --benchmark-group-by=func \
      "$@"
    ;;
  list)
    ls -1 .benchmarks/ 2>/dev/null || echo "(no saved benchmarks yet)"
    ;;
  solo)
    ALGO="${1:?usage: $0 solo <algo> [--fast|--json|--quiet]}"
    shift
    .venv/bin/python -m benchmarks.algos.bench_one "$ALGO" "$@"
    ;;
  *)
    cat <<USAGE
Usage: $0 <mode> [extra pytest args]

Modes:
  run              Run benchmarks, autosave result under .benchmarks/<machine>/
  baseline         Save as 'baseline' - the reference to compare against
  compare          Compare current against saved baseline, fail if any algo >10% slower
  gate             Strict gate: fail if any algo >5% slower (for CI)
  list             List saved benchmark runs
  solo <algo>      Run ONE algo's bench file, print METRIC lines (for autoresearch)
                   algos: graph compact fusion materialization spreading eviction sort text
                   extra flags: --fast --quiet --json

Typical workflow:
  1. $0 baseline           # capture reference
  2. <edit algos/*.py>
  3. $0 compare            # see relative perf
  4. $0 baseline           # if change is good, lock new baseline
USAGE
    exit 1
    ;;
esac
