#!/usr/bin/env bash
# Idempotent benchmark queue scheduler.
# Cron fires this every N minutes. It inspects state file + Docker
# container state, and fires the next queued bench run if the
# previous one has exited. flock prevents double-fire if two cron
# ticks overlap.
#
# State file: ~/.gs_bench_state (contains current queue index)
# Log file:   ~/.gs_bench_scheduler.log
# Lock file:  /tmp/gs_bench_scheduler.lock

set -u

STATE=$HOME/.gs_bench_state
LOG=$HOME/.gs_bench_scheduler.log
LOCK=/tmp/gs_bench_scheduler.lock
REPO=/home/kai/orkait/graphstore/graphstore
MODELS=/home/kai/graphstore-models
PER_CATEGORY=20

exec 9>"$LOCK" 2>/dev/null || exit 0
flock -n 9 || exit 0

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

# Ordered queue: container|system|embedder|model|tag
QUEUE=(
  "gs_h2h_chroma|chroma-bm25|fastembed|BAAI/bge-small-en-v1.5|h2h_chroma_bge_small"
  "gs_h2h_llamaindex|llamaindex|fastembed|BAAI/bge-small-en-v1.5|h2h_llamaindex_bge_small"
  "gs_h2h_graphstore|graphstore|fastembed|BAAI/bge-small-en-v1.5|h2h_graphstore_bge_small"
)

[ ! -f "$STATE" ] && echo "0" > "$STATE"
idx=$(cat "$STATE")

if [ "$idx" -ge "${#QUEUE[@]}" ]; then
  log "tick idx=$idx state=queue_complete (all ${#QUEUE[@]} runs done)"
  exit 0
fi

IFS='|' read -r name system embedder model tag <<< "${QUEUE[$idx]}"

# Step currently running? Log pulse + pull container progress tail.
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${name}$"; then
  last_progress=$(docker logs --tail 1 "$name" 2>&1 | grep -oE '[0-9]+/[0-9]+' | tail -1)
  log "tick idx=$idx state=running name=$name progress=${last_progress:-starting}"
  exit 0
fi

# Step finished (exists but not running)? Advance and log outcome.
existing=$(docker ps -a --format '{{.Names}}' 2>/dev/null | grep "^${name}$" || true)
if [ -n "$existing" ]; then
  status=$(docker inspect -f '{{.State.Status}}' "$name" 2>/dev/null || echo unknown)
  exit_code=$(docker inspect -f '{{.State.ExitCode}}' "$name" 2>/dev/null || echo -1)
  summary=$(docker logs --tail 20 "$name" 2>&1 | grep -E "overall|by category|elapsed|accuracy" | head -10 | tr '\n' ' | ')
  log "tick idx=$idx state=finished name=$name status=$status exit=$exit_code summary=$summary"
  echo $((idx + 1)) > "$STATE"
  exit 0
fi

# Step not yet started. Fire it.
log "tick idx=$idx state=firing name=$name system=$system embedder=$embedder model=$model tag=$tag per_cat=$PER_CATEGORY"
docker run -d \
  --cpus=12 --memory=16g \
  -v "$MODELS:/data:ro" \
  -v "$MODELS/cache:/cache" \
  -v "$REPO/benchmarks/framework/results:/results" \
  --name "$name" \
  graphstore-bench:latest \
  --system "$system" \
  --embedder "$embedder" --embedder-model "$model" \
  --per-category "$PER_CATEGORY" \
  --run-tag "$tag" >> "$LOG" 2>&1 \
  && log "fired $name successfully" \
  || log "docker run FAILED for $name"
