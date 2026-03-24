# Concurrency Model

## Current Design

Single-writer command queue with opt-in threading.

```
GraphStore(threaded=False)  # default: single-threaded, zero overhead
GraphStore(threaded=True)   # command queue serializes all access
```

## In-Process Concurrency (supported)

Multiple threads sharing one `GraphStore(threaded=True)` instance:

```
Agent thread 1 ──┐
Agent thread 2 ──┤──► PriorityQueue ──► Worker Thread ──► CoreStore
Background job ──┘
```

- All `execute()` calls go through a single-writer queue
- Interactive queries (priority 0) complete before background jobs (priority 1)
- `submit_background(query)` returns a Future for fire-and-forget refinement
- Worker thread survives individual query failures
- Zero locks in the core engine - serialization happens at the queue level

**Use cases:**
- Multi-agent orchestrator with shared knowledge graph
- Background refinement (SYS OPTIMIZE, SYS CONNECT, SYS REEMBED) while agents query
- Vault filesystem watcher triggering sync during agent operation

## Cross-Process Sharing (NOT supported)

Two separate processes opening `GraphStore(path="./brain")` simultaneously will corrupt data:

- Each process has separate numpy arrays in separate memory
- Checkpoint writes to the same SQLite blobs table - last writer wins, data lost
- WAL entries interleave with conflicting slot assignments
- String tables diverge - same string gets different IDs in each process

**Do not share a path between processes.**

### Multi-process architecture

If multiple processes need access to the same graph, use the server as coordination point:

```
Process A ──► HTTP ──┐
Process B ──► HTTP ──┤──► FastAPI server ──► GraphStore(threaded=True)
Process C ──► HTTP ──┘
```

One process owns the GraphStore instance. Others connect via `/api/execute`.

## Why the Core Engine Stays Single-Threaded

- numpy arrays are not thread-safe for mutation (`_grow()` reallocates during writes)
- scipy CSR matrices are rebuilt lazily via `_edges_dirty` flag - concurrent writers would race
- `StringTable.intern()` assigns monotonic IDs - concurrent interns of same string race
- Python GIL serializes CPU-bound work anyway - threading buys nothing for numpy loops
- No locks = no deadlocks = easy to reason about = fast

## Implications for MCP

- MCP server wraps one GraphStore instance with `threaded=True`
- Multiple MCP clients submit through the same queue
- Background maintenance (optimize, connect, reembed) uses `submit_background()`
- The `_optimizing` flag remains as a re-entrancy guard within the worker thread
