# Self-Balancing: Agent-Driven DB Optimization

## Context

graphstore is the DB layer. The agent layer sits above it. The DB should not manage its own cleanup silently - the agent needs to know when optimization is happening because it must pause operations during it.

## Two Modes

### Manual Mode (agent-triggered)

Agent reads pressure, decides when to optimize, calls the command.

```
SYS HEALTH        -> {"tombstone_ratio": 0.35, "dead_vectors": 48, ...}
SYS OPTIMIZE      -> lock, run all ops, unlock, return summary
```

During `SYS OPTIMIZE`, all other `execute()` calls are rejected with `OptimizationInProgress`. Agent knows to wait.

### Auto Mode (DB-triggered at safe points)

Health check runs every N writes. If pressure detected, optimization runs at the **top of the next `execute()` call** - the natural safe point between operations where nothing is in flight.

Caller sees slightly higher latency on one query. No rejected calls, no retry needed.

Disabled by default in config (`auto_optimize: false`).

## MCP Commands to Expose

| Command | Purpose | Agent uses it when |
|---|---|---|
| `SYS HEALTH` | Read pressure metrics without acting | Periodic health polling, before deciding to optimize |
| `SYS OPTIMIZE` | Run all safe optimizations | Agent decides pressure is high enough |
| `SYS OPTIMIZE [target]` | Run specific optimization (EDGES, VECTORS, BLOBS, CACHE, COMPACT, STRINGS) | Agent wants targeted cleanup |

## 6 Optimization Operations

| Operation | What it does | Safe standalone? |
|---|---|---|
| Tombstone Compact | Shift live slots down, rebuild id_to_slot, remap edges/vectors/documents | Yes - only under lock (no concurrent calls) |
| String GC | Rebuild string table with only referenced strings, remap all int32 columns | Yes - only under lock |
| Edge Defrag | Rebuild CSR from clean edge lists, purge stale edge_keys | Yes always |
| Vector Cleanup | Remove HNSW entries for tombstoned/retracted slots | Yes always |
| Orphan Sweep | Delete DocumentStore rows not in live mask | Yes always |
| Cache Clear | Clear plan cache + edge combination cache | Yes always |

## Key Constraint

Tombstone compaction and string GC renumber slot indices and string IDs. Every data structure uses these as keys. They are only safe because the agent guarantees no concurrent calls during `SYS OPTIMIZE`. The lock in `execute()` enforces this.

## Health Metrics

| Metric | Threshold | Triggers |
|---|---|---|
| `tombstone_ratio` | > 0.2 | COMPACT |
| `string_bloat` | string_table / live_nodes > 3 | STRINGS |
| `dead_vectors` | dead / total > 0.2 | VECTORS |
| `orphan_blobs` | checked via DocumentStore scan | BLOBS |
| `stale_edges` | edge_keys size vs actual edges | EDGES |
| `cache_size` | plan cache entries > threshold | CACHE |
