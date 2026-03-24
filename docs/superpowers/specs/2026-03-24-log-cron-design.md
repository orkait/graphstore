# Intelligent Log Layer + CRON Scheduler - Design Spec

## Problem

1. **Logging**: Agents can't introspect their own activity. `query_log` captures raw DSL + timing but has no semantic tags, no causality traces, no source tracking. Humans can't monitor agent behaviour without reading SQLite directly.

2. **CRON**: Agents need scheduled maintenance (expire, retain, optimize, reembed, vault sync) but all system commands are manual. No way to say "run SYS EXPIRE every hour."

## Solution

### A. Intelligent Log Layer

Enrich the existing `query_log` table with semantic metadata. Expose via DSL (`SYS LOG`), Python `logging` module (structured events), and REST API (`/api/logs`).

### B. CRON Scheduler

Persistent cron jobs in SQLite, full cron expression support via `croniter`, daemon timer thread submitting jobs through the existing command queue. Requires `threaded=True`.

---

## A. Intelligent Log Layer

### Schema Change

Extend `query_log` table (backwards compatible - new columns are nullable):

```sql
ALTER TABLE query_log ADD COLUMN tag TEXT;
ALTER TABLE query_log ADD COLUMN trace_id TEXT;
ALTER TABLE query_log ADD COLUMN source TEXT DEFAULT 'user';
ALTER TABLE query_log ADD COLUMN phase TEXT;
```

| Column | Values | Auto-inferred |
|--------|--------|---------------|
| `tag` | "read", "write", "intelligence", "system", "ingest", "vault", "belief" | Yes, from AST type |
| `trace_id` | arbitrary string or null | Set via `BIND TRACE` |
| `source` | "user", "cron:{name}", "background", "wal_replay" | Yes, from execution context |
| `phase` | "query", "mutation", "system", "batch" | Yes, from AST type |

### Auto-Tagging Rules

Inferred from AST node type at log time (zero manual effort):

```python
TAG_MAP = {
    # Reads
    NodeQuery: "read", NodesQuery: "read", EdgesQuery: "read",
    CountQuery: "read",
    # Intelligence
    RecallQuery: "intelligence", SimilarQuery: "intelligence",
    LexicalSearchQuery: "intelligence", CounterfactualQuery: "intelligence",
    # Traversal
    TraverseQuery: "read", PathQuery: "read", ShortestPathQuery: "read",
    SubgraphQuery: "read", AncestorsQuery: "read", DescendantsQuery: "read",
    MatchQuery: "read",
    # Writes
    CreateNode: "write", UpdateNode: "write", UpsertNode: "write",
    DeleteNode: "write", DeleteNodes: "write", Increment: "write",
    CreateEdge: "write", UpdateEdge: "write", DeleteEdge: "write",
    # Beliefs
    AssertStmt: "belief", RetractStmt: "belief", PropagateStmt: "belief",
    # Ingest
    IngestStmt: "ingest",
    # Vault
    VaultNew: "vault", VaultRead: "vault", VaultWrite: "vault",
    VaultSync: "vault", VaultSearch: "vault",
    # Batch
    Batch: "write",
    # System (all Sys* types)
    # Default: "system"
}

PHASE_MAP = {
    "read": "query", "intelligence": "query",
    "write": "mutation", "belief": "mutation", "ingest": "mutation",
    "vault": "mutation", "system": "system",
}
```

### Trace ID

Optional causality tracking via existing context binding pattern:

```
BIND TRACE "research-session-42"
RECALL FROM "quantum" DEPTH 3
SIMILAR TO "entanglement" LIMIT 5
CREATE NODE "insight_1" kind = "fact" ...
DISCARD TRACE "research-session-42"
```

All queries between BIND/DISCARD get `trace_id = "research-session-42"` in the log. Nesting is NOT supported (last BIND wins).

Implementation: store `_active_trace: str | None` on GraphStore, pass to WALManager.log_query().

### DSL: SYS LOG

```
SYS LOG limit_clause?
SYS LOG WHERE expr limit_clause?
SYS LOG SINCE "ISO-8601" limit_clause?
SYS LOG TRACE "trace-id"
```

Grammar additions:
```lark
sys_log: "LOG" log_filter? limit_clause?
log_filter: "WHERE" expr
          | "SINCE" STRING
          | "TRACE" STRING
```

Returns: `Result(kind="log_entries", data=[...], count=N)`

Each entry:
```json
{
    "id": 42,
    "timestamp": 1711234567.89,
    "query": "RECALL FROM \"cue\" DEPTH 3",
    "elapsed_us": 1234,
    "result_count": 5,
    "error": null,
    "tag": "intelligence",
    "trace_id": "research-42",
    "source": "user",
    "phase": "query"
}
```

### Python Logging Integration

Emit structured events via standard `logging` module:

```python
import logging
_event_logger = logging.getLogger("graphstore.events")

def _emit_event(query, elapsed_us, result_count, error, tag, source, trace_id, phase):
    _event_logger.info(
        "%s [%s] %dus %d results",
        tag, source, elapsed_us, result_count,
        extra={
            "gs_query": query,
            "gs_tag": tag,
            "gs_source": source,
            "gs_trace_id": trace_id,
            "gs_phase": phase,
            "gs_elapsed_us": elapsed_us,
            "gs_result_count": result_count,
            "gs_error": error,
        },
    )
```

Operators configure handlers on `graphstore.events` logger. If no handler configured, events go nowhere (Python default). Prefix `gs_` avoids collision with standard LogRecord fields.

### REST API: /api/logs

```
GET  /api/logs?limit=50
GET  /api/logs?tag=intelligence&limit=20
GET  /api/logs?source=cron&limit=20
GET  /api/logs?trace_id=research-42
GET  /api/logs?since=2025-03-24T00:00:00&limit=100
```

Returns JSON array of log entries. Uses same SQLite query as `SYS LOG`.

---

## B. CRON Scheduler

### Dependency

`croniter>=6.0` - lightweight (single file, no deps), well-maintained, full cron expression support including:
- Standard 5-field: `* * * * *` (minute hour day month weekday)
- Extended 6-field: `* * * * * *` (second minute hour day month weekday)
- Ranges: `1-5`, lists: `1,3,5`, steps: `*/15`
- Named days/months: `MON-FRI`, `JAN-DEC`
- Special strings: `@hourly`, `@daily`, `@weekly`, `@monthly`, `@yearly`

### SQLite Table

```sql
CREATE TABLE IF NOT EXISTS cron_jobs (
    name TEXT PRIMARY KEY,
    schedule TEXT NOT NULL,
    query TEXT NOT NULL,
    enabled INTEGER DEFAULT 1,
    created_at REAL NOT NULL,
    last_run REAL,
    next_run REAL NOT NULL,
    run_count INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    last_error TEXT
);
```

### DSL Commands

```
SYS CRON ADD "name" SCHEDULE "cron-expr" QUERY "dsl-query"
SYS CRON DELETE "name"
SYS CRON ENABLE "name"
SYS CRON DISABLE "name"
SYS CRON LIST
SYS CRON RUN "name"
```

Grammar additions:
```lark
sys_cron: "CRON" cron_command
cron_command: cron_add | cron_delete | cron_enable | cron_disable | cron_list | cron_run
cron_add: "ADD" STRING "SCHEDULE" STRING "QUERY" STRING
cron_delete: "DELETE" STRING
cron_enable: "ENABLE" STRING
cron_disable: "DISABLE" STRING
cron_list: "LIST"
cron_run: "RUN" STRING
```

### Examples

```
SYS CRON ADD "expire-ttl" SCHEDULE "0 * * * *" QUERY "SYS EXPIRE"
SYS CRON ADD "nightly-optimize" SCHEDULE "0 3 * * *" QUERY "SYS OPTIMIZE"
SYS CRON ADD "reembed-weekly" SCHEDULE "0 2 * * 0" QUERY "SYS REEMBED"
SYS CRON ADD "retain-daily" SCHEDULE "@daily" QUERY "SYS RETAIN"
SYS CRON ADD "vault-sync" SCHEDULE "*/5 * * * *" QUERY "VAULT SYNC"
SYS CRON ADD "health-check" SCHEDULE "*/10 * * * *" QUERY "SYS HEALTH"

SYS CRON LIST
SYS CRON DISABLE "reembed-weekly"
SYS CRON RUN "expire-ttl"
SYS CRON DELETE "health-check"
```

### CronScheduler Class (`graphstore/cron.py`)

```python
class CronScheduler:
    """Persistent cron scheduler with daemon timer thread."""

    def __init__(self, conn, submit_background_fn):
        self._conn = conn
        self._submit = submit_background_fn  # GraphStore.submit_background
        self._thread = None
        self._running = False

    def start(self):
        """Start the timer thread. Called after GraphStore init."""
        self._ensure_table()
        self._running = True
        self._thread = threading.Thread(target=self._tick_loop, daemon=True, name="graphstore-cron")
        self._thread.start()

    def stop(self):
        """Stop the timer thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _tick_loop(self):
        """Check every 60s for jobs due to fire."""
        while self._running:
            self._tick()
            # Sleep in 1s increments so shutdown is responsive
            for _ in range(60):
                if not self._running:
                    return
                time.sleep(1)

    def _tick(self):
        """Find and execute all due jobs."""
        now = time.time()
        rows = self._conn.execute(
            "SELECT name, query, schedule FROM cron_jobs WHERE enabled = 1 AND next_run <= ?",
            (now,)
        ).fetchall()
        for name, query, schedule in rows:
            try:
                future = self._submit(query)
                future.add_done_callback(lambda f, n=name: self._on_done(f, n))
                # Compute next run BEFORE waiting for result
                from croniter import croniter
                next_run = croniter(schedule, now).get_next(float)
                self._conn.execute(
                    "UPDATE cron_jobs SET last_run = ?, next_run = ?, run_count = run_count + 1 WHERE name = ?",
                    (now, next_run, name)
                )
                self._conn.commit()
            except Exception as e:
                self._conn.execute(
                    "UPDATE cron_jobs SET error_count = error_count + 1, last_error = ? WHERE name = ?",
                    (str(e), name)
                )
                self._conn.commit()

    def _on_done(self, future, job_name):
        """Callback after background job completes."""
        exc = future.exception()
        if exc:
            try:
                self._conn.execute(
                    "UPDATE cron_jobs SET error_count = error_count + 1, last_error = ? WHERE name = ?",
                    (str(exc), job_name)
                )
                self._conn.commit()
            except Exception:
                pass

    # CRUD methods: add, delete, enable, disable, list, run_now
    # Each wraps a simple SQL operation + croniter validation
```

### Lifecycle

```
GraphStore.__init__(threaded=True)
  → CronScheduler(conn, self.submit_background)
  → cron.start()  # starts daemon timer thread

GraphStore.close()
  → cron.stop()   # stops timer, joins thread
  → queue.shutdown()
  → checkpoint + close
```

### Requires threaded=True

`CronScheduler.__init__` validates:
```python
if submit_background_fn is None:
    raise RuntimeError("CRON requires GraphStore(threaded=True)")
```

Without the command queue, cron jobs would corrupt state by running on the timer thread while the main thread is mid-query.

### Cron + Log Integration

When the cron tick submits a job:
- `source` in query_log = `"cron:{job_name}"` (e.g., `"cron:expire-ttl"`)
- The `_execute_internal` method needs to accept a `source` parameter
- CronScheduler passes source metadata through a thread-local or via the command queue item

Implementation: extend `CommandQueue` items to carry optional metadata `(priority, seq, query, future, metadata)`. The metadata dict flows through to `_execute_internal` and into the log.

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `graphstore/persistence/database.py` | Modify | Add log columns + cron_jobs table to schema |
| `graphstore/wal.py` | Modify | Extend log_query() with tag/source/trace_id/phase |
| `graphstore/graphstore.py` | Modify | Wire trace binding, cron scheduler, source propagation |
| `graphstore/cron.py` | Create | CronScheduler class |
| `graphstore/core/queue.py` | Modify | Add metadata to queue items |
| `graphstore/dsl/grammar.lark` | Modify | Add SYS LOG and SYS CRON commands |
| `graphstore/dsl/ast_nodes.py` | Modify | Add SysLog, SysCron* AST nodes |
| `graphstore/dsl/transformer.py` | Modify | Parse new commands |
| `graphstore/dsl/executor_system.py` | Modify | Handle SYS LOG and SYS CRON dispatch |
| `graphstore/server.py` | Modify | Add /api/logs endpoint |
| `graphstore/config.py` | Modify | Add CronConfig section |
| `pyproject.toml` | Modify | Add croniter dependency |
| `tests/test_log_layer.py` | Create | Log enrichment tests |
| `tests/test_cron.py` | Create | Cron scheduler tests |

---

## What Does NOT Change

- CoreStore, ColumnStore, EdgeMatrices - untouched
- All existing DSL commands - unchanged
- Existing `SYS SLOW/FREQUENT/FAILED` - still work (read same table)
- Non-threaded mode - works exactly as before (no cron, no background)
- Existing tests - all pass (new columns are nullable)

---

## Testing Strategy

### Log Layer Tests
- Auto-tag inference: CREATE -> "write", RECALL -> "intelligence", SYS STATS -> "system"
- Trace binding: BIND TRACE -> queries get trace_id -> DISCARD TRACE -> null
- Source tracking: normal = "user", cron = "cron:name"
- SYS LOG query with filters
- /api/logs endpoint

### Cron Tests
- Add/delete/enable/disable/list CRUD
- Schedule validation (invalid cron expr rejected)
- Tick fires due jobs
- Priority ordering (cron jobs are BACKGROUND)
- Error handling (job failure doesn't kill scheduler)
- Persistence (jobs survive restart)
- SYS CRON RUN manual trigger
- Full cron expressions: `*/5 * * * *`, `0 3 * * MON-FRI`, `@hourly`
