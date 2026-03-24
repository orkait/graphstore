# Command Queue Design Spec

## Problem

GraphStore's core engine is single-threaded by design (numpy arrays, CSR matrices, string interning - none are thread-safe for mutation). But agents need to run background refinement jobs (SYS OPTIMIZE, SYS CONNECT, SYS REEMBED, retention sweeps) concurrently with interactive queries. Currently, calling `execute()` from multiple threads corrupts state.

## Solution

Opt-in command queue with a single-writer worker thread. `GraphStore(threaded=True)` serializes all access through a priority queue. Default behavior is unchanged.

## Components

### CommandQueue (`graphstore/core/queue.py`)

New file. ~80 lines.

```
Callers ──submit()──────────► PriorityQueue ──► Worker Thread ──► _execute_internal()
         submit_background()►                   (daemon)
```

- `queue.PriorityQueue` with items: `(priority, sequence_number, query, Future)`
- Two priority levels: `INTERACTIVE = 0`, `BACKGROUND = 1`
- Monotonic sequence number for FIFO within same priority
- Worker thread: infinite loop draining queue, calling callback, setting future result
- `submit(query) -> Result`: enqueues at INTERACTIVE priority, blocks via `future.result()`
- `submit_background(query) -> Future[Result]`: enqueues at BACKGROUND priority, returns immediately
- `shutdown()`: sends sentinel, joins worker thread (5s timeout)
- Worker catches exceptions per-query and sets them on the future (thread survives)

### GraphStore changes (`graphstore/graphstore.py`)

- Add `threaded: bool = False` to constructor
- If `threaded=True`: instantiate `CommandQueue` with `self._execute_internal` as callback
- Rename current `execute()` to `_execute_internal()`
- New `execute()`: if threaded, delegates to `self._queue.submit(query)`. If not, calls `_execute_internal(query)` directly.
- New `submit_background(query) -> Future[Result]`: raises RuntimeError if not threaded. Otherwise delegates to `self._queue.submit_background(query)`.
- `close()`: calls `self._queue.shutdown()` before existing cleanup

### Server changes (`graphstore/server.py`)

None required. `store.execute()` is transparent - it blocks and returns `Result` regardless of threading mode. The `_get_store()` factory can optionally set `threaded=True` via env var but this is not required for correctness since FastAPI+uvicorn already serializes sync endpoints.

## What does NOT change

- `CoreStore` - no locks, no threading awareness
- `ColumnStore`, `EdgeMatrices`, `StringTable` - untouched
- All executors - untouched
- Existing single-threaded callers - zero overhead (no queue created)
- Test suite - all existing tests run with `threaded=False`

## API

```python
# Current usage (unchanged)
gs = GraphStore(path="./brain")
result = gs.execute('NODE "alice"')

# Threaded usage
gs = GraphStore(path="./brain", threaded=True)
result = gs.execute('NODE "alice"')  # blocks, same API

# Background job
future = gs.submit_background('SYS OPTIMIZE')
# ... agent continues working ...
result = future.result()  # optional: check result later

# Fire-and-forget
gs.submit_background('SYS CONNECT')  # don't care about result
```

## Error handling

- Query exception in worker thread -> set on Future -> re-raised in caller's thread when `.result()` is called
- Worker thread survives individual query failures
- If worker thread dies unexpectedly, subsequent `submit()` calls raise `RuntimeError`
- `shutdown()` is idempotent

## Testing

- `test_command_queue.py`:
  - Single-threaded submit returns correct result
  - Background submit returns future that resolves
  - Priority ordering: interactive queries complete before background
  - Error propagation: query error raised in caller thread
  - Shutdown is clean (no hanging threads)
  - Multiple threads submitting concurrently all get correct results
  - `threaded=False` has no queue overhead
