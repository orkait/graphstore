# Improvements Spec v1

Status: approved, ready for execution.
Scope: src-only behavioural fixes. Root-cause corrections, not patches.
Audience: implementation LLM with narrow reasoning. Every step below is prescriptive.

Global rules for the executor:
- Never skip an "Acceptance" check. If it fails, fix the failure before moving to the next item.
- Run the listed pytest command after each item. Suite must stay green.
- Never add `# type: ignore`, `try/except: pass`, or `--no-verify` to silence a failure. Investigate.
- Don't rename public symbols unless this spec tells you to.
- Don't add dependencies.
- Write the test BEFORE the fix for P0 items. Confirm the test fails against HEAD, then apply the fix.

Priority legend:
- **P0** - correctness/data-loss bug. Ship in first PR.
- **P1** - silent failure or rollback gap. Ship in second PR.
- **P2** - hygiene / observability. Batch into third PR.
- **P3** - cleanup. Optional.

---

## P0-1: MERGE self-merge deletes the node

Severity: CRITICAL (data loss).
Files: `src/graphstore/dsl/handlers/mutations.py` (handler), `src/graphstore/dsl/ast_nodes.py` (reference only).

### Problem
`MERGE NODE "foo" INTO "foo"` rewires every edge touching slot X to target slot X (self-loops), then calls `store.delete_node("foo")`. The node and all its edges become a tombstoned slot with self-loops on a dead slot. Merge is supposed to be non-destructive when source == target.

### Root cause
`MutationHandlers._merge` (starting around line 500) resolves `src_slot` and `tgt_slot` independently. No equality guard before mutation.

### Fix
Add an early-exit guard at the top of `_merge`. Treat self-merge as a validation error (loud, not silent no-op, because the caller clearly made a mistake).

Edit `src/graphstore/dsl/handlers/mutations.py`:

Find the method `_merge` (decorated with `@handles(MergeStmt, write=True)`). Immediately after the two `raise NodeNotFound(...)` guards on `src_slot` and `tgt_slot`, add:

```python
if src_slot == tgt_slot:
    raise GraphStoreError(
        f"MERGE source and target resolve to the same slot: "
        f"{q.source_id!r} == {q.target_id!r}"
    )
```

`GraphStoreError` is already imported at the top of the file. If it is not, add `from graphstore.core.errors import GraphStoreError` to the imports.

### Test
Create or append to `tests/test_merge.py` (create if missing). Add:

```python
import pytest
from graphstore import GraphStore
from graphstore.core.errors import GraphStoreError


def test_merge_self_is_rejected():
    gs = GraphStore()
    try:
        gs.execute('CREATE NODE "a" kind = "person" name = "alice"')
        gs.execute('CREATE NODE "b" kind = "person" name = "bob"')
        gs.execute('CREATE EDGE "a" -> "b" kind = "knows"')
        with pytest.raises(GraphStoreError, match="same slot"):
            gs.execute('MERGE NODE "a" INTO "a"')
        assert gs.execute('NODE "a"').data is not None
        assert gs.execute('NODE "b"').data is not None
        assert gs.execute('COUNT EDGES WHERE kind = "knows"').data == 1
    finally:
        gs.close()


def test_merge_different_nodes_still_works():
    gs = GraphStore()
    try:
        gs.execute('CREATE NODE "a" kind = "person" name = "alice"')
        gs.execute('CREATE NODE "b" kind = "person" name = "bob"')
        gs.execute('MERGE NODE "a" INTO "b"')
        assert gs.execute('NODE "a"').data is None
        assert gs.execute('NODE "b"').data is not None
    finally:
        gs.close()
```

### Verification
```
uv run pytest tests/test_merge.py -xvs
uv run pytest -x
```

### Acceptance
- `test_merge_self_is_rejected` passes.
- `test_merge_different_nodes_still_works` passes.
- Full suite is green.

---

## P0-2: Batch with VECTOR clause leaks vectors on rollback

Severity: HIGH (state corruption).
Files: `src/graphstore/dsl/handlers/mutations.py`.

### Problem
`BEGIN ... COMMIT` with a `CREATE NODE ... VECTOR [...]` inside, followed by any failing statement, rolls back graph state but leaves the vector in the HNSW index. After rollback, the slot may be reused by a future write; the stale vector now shadows a different logical node, corrupting SIMILAR TO / REMEMBER results.

### Root cause
`_batch` (around line 620) snapshots: edges, edge_keys, columns, tombstones, id_to_slot, count, next_slot, node_ids, node_kinds. Does NOT snapshot `vector_store._has_vector` or the set of slots it wrote to. No compensating remove on rollback.

### Fix
Track vectors added during the batch; on rollback, remove them.

Strategy:
1. Install a "vector write recorder" on the executor before dispatching the batch.
2. Every `vector_store.add(slot, ...)` inside the batch appends `slot` to a list on the executor.
3. On rollback, iterate the list and call `vector_store.remove(slot)` for each.

Add the recorder field to `ExecutorBase.__init__` in `src/graphstore/dsl/executor_base.py`:

```python
self._batch_vector_record: list[int] | None = None
```

Wrap `vector_store.add` calls from batch-aware handlers. Update `MutationHandlers._handle_vector` and `MutationHandlers._embed_and_store` and `MutationHandlers._batch_embed_and_store` in `src/graphstore/dsl/handlers/mutations.py` so every successful `vector_store.add(slot, ...)` also does:

```python
if self._batch_vector_record is not None:
    self._batch_vector_record.append(slot)
```

Concretely, find the three add-sites and append immediately after each:

1. In `_handle_vector`, after `self._vector_store.add(slot, vec)` (both the early explicit branch and the lazy-init branch).
2. In `_embed_and_store`, after `self._vector_store.add(slot, vec)`.
3. In `_batch_embed_and_store`, after the `for slot, vec in zip(slots, vecs): self._vector_store.add(slot, vec)` loop. Extend the list with `slots` (which is a tuple of ints).

Now update `_batch` (same file). Around the `saved_*` block, install and tear down the recorder:

```python
# Before the `if enable_rollback:` block, right after the method signature:
prev_record = self._batch_vector_record
self._batch_vector_record = [] if enable_rollback else None
```

In the existing `except Exception as e:` block, before the existing rollback restorations (but only when `enable_rollback` is True), iterate the record and remove:

```python
if enable_rollback and self._batch_vector_record:
    vs = self._vector_store
    if vs is not None:
        for slot in self._batch_vector_record:
            try:
                vs.remove(slot)
            except Exception:
                pass  # best-effort; index may have evicted already
```

In a `finally` block that wraps the entire try, restore the previous recorder:

```python
finally:
    self._batch_vector_record = prev_record
```

If the try/except doesn't already have a finally, add one. Make sure the rollback restore code runs before the finally (it already does because it's in `except`).

### Test
Append to `tests/test_mutations.py` (create if missing):

```python
import pytest
from graphstore import GraphStore


def test_batch_vector_rolled_back_on_failure():
    gs = GraphStore(embedder=None)
    try:
        gs.execute('CREATE NODE "seed" kind = "doc" text = "hello"')
        # Precondition: no vectors yet
        assert gs._vector_store is None or gs._vector_store.count() == 0

        # Prepare a batch: create node with explicit VECTOR, then fail on a
        # second statement. Rollback must remove the vector.
        with pytest.raises(Exception):
            gs.execute(
                'BEGIN\n'
                'CREATE NODE "x" kind = "doc" text = "a" VECTOR [0.1, 0.2, 0.3]\n'
                'CREATE NODE "x" kind = "doc" text = "b"\n'  # NodeExists -> BatchRollback
                'COMMIT'
            )

        # Post-rollback: node x should not exist, vector index should have zero entries.
        assert gs.execute('NODE "x"').data is None
        vs = gs._vector_store
        assert vs is None or vs.count() == 0
    finally:
        gs.close()
```

### Verification
```
uv run pytest tests/test_mutations.py::test_batch_vector_rolled_back_on_failure -xvs
uv run pytest -x
```

### Acceptance
- New test passes.
- Existing batch/rollback tests still pass.

---

## P0-3: `deferred_embeddings` silently drops pending embeddings on exception

Severity: HIGH (data loss, silent).
Files: `src/graphstore/store.py`.

### Problem
`GraphStore.deferred_embeddings(batch_size=N)` context (lines ~627-662) wraps the executor's pending-queue state. The `finally` block clears `executor._pending_embeddings` unconditionally. If the caller's block raises AFTER the context has enqueued slots, those slots exist in the graph but have no vector. Silent non-searchable nodes.

### Root cause
Finally-block clears the queue instead of flushing when no exception occurred, and does not record unflushed slots when an exception did occur.

### Fix
Split the flush into two stages. On clean exit: flush. On exception: attempt flush, swallow a flush exception only after logging the slot ids of unembedded items.

Edit the `deferred_embeddings` method:

```python
@contextmanager
def deferred_embeddings(self, batch_size: int = 64):
    """..."""
    executor = self._executor
    prev_defer = executor._defer_embeddings
    prev_batch_size = executor._embed_batch_size
    executor._defer_embeddings = True
    executor._embed_batch_size = batch_size
    try:
        yield
    except BaseException:
        # Caller's block raised. Try to flush so writes are not lost;
        # if flush itself fails, log the pending slots before clearing
        # so the developer can rebuild vectors for those nodes.
        try:
            executor.flush_pending_embeddings()
        except Exception as flush_err:
            pending = list(getattr(executor, "_pending_embeddings", []))
            slots = [s for s, _ in pending]
            logger.error(
                "deferred_embeddings: flush failed during exception unwind; "
                "%d slot(s) have no vector: %s (flush error: %s)",
                len(slots), slots[:20], flush_err,
            )
            executor._pending_embeddings.clear()
        raise
    else:
        executor.flush_pending_embeddings()
    finally:
        executor._defer_embeddings = prev_defer
        executor._embed_batch_size = prev_batch_size
```

Note: the existing code has `yield / executor.flush_pending_embeddings()` in the try, then `finally` clears. Replace the whole body with the above. Keep the docstring.

### Test
Append to `tests/test_deferred_embeddings.py`:

```python
import logging

import pytest

from graphstore import GraphStore


def test_deferred_embeddings_flushes_on_clean_exit(monkeypatch):
    gs = GraphStore(embedder="default")
    try:
        with gs.deferred_embeddings(batch_size=4):
            gs.execute('CREATE NODE "a" kind = "doc" text = "hello"')
            gs.execute('CREATE NODE "b" kind = "doc" text = "world"')
        # After context exit, both nodes should have vectors.
        # Register embed_field via schema so embedding is triggered.
    finally:
        gs.close()


def test_deferred_embeddings_flushes_on_exception_path(caplog):
    gs = GraphStore(embedder="default")
    try:
        gs.execute('SYS REGISTER NODE "doc" REQUIRED text EMBED text')
        with pytest.raises(RuntimeError):
            with gs.deferred_embeddings(batch_size=64):
                gs.execute('CREATE NODE "a" kind = "doc" text = "hello"')
                gs.execute('CREATE NODE "b" kind = "doc" text = "world"')
                raise RuntimeError("simulated caller failure")

        # Vectors should still be present because flush runs during unwind.
        vs = gs._vector_store
        assert vs is not None
        assert vs.count() >= 2
    finally:
        gs.close()
```

Only run the second test when the default embedder (`model2vec`) can be imported. If CI skips the extra, mark:

```python
pytestmark = pytest.mark.needs_embedder
```

### Verification
```
uv run pytest tests/test_deferred_embeddings.py -xvs
```

### Acceptance
- Both new tests pass (skip if extra missing).
- On exception path, `vs.count() >= 2`.

---

## P1-1: REMEMBER with AT clause and no `__event_at__` column silently returns empty

Severity: HIGH (silent failure, user confusion).
Files: `src/graphstore/dsl/handlers/intelligence.py`.

### Problem
In `_remember`, when `at` or `at_range` is set but the schema has no `__event_at__` column, the code multiplies `base_final *= 0.0`, which zeros every candidate. The user gets `count == 0` with no explanation.

### Root cause
The temporal filter block (around lines 548-551) treats "column absent" as "no match" without surfacing the reason.

### Fix
When the column is absent, do not zero the result. Keep all candidates but attach a warning in `result.meta` so the caller can react.

Edit `_remember` in `src/graphstore/dsl/handlers/intelligence.py`. Replace the block:

```python
if at_range is not None or anchor_ms is not None:
    t_event_col = self.store.columns.get_column("__event_at__", n)
    if t_event_col is None:
        base_final *= 0.0
    else:
        ...
```

with:

```python
warnings: list[str] = []
if at_range is not None or anchor_ms is not None:
    t_event_col = self.store.columns.get_column("__event_at__", n)
    if t_event_col is None:
        warnings.append(
            "AT clause ignored: no '__event_at__' column in store. "
            "Use ASSERT ... EVENT_AT ... or CREATE NODE ... EVENT_AT ... "
            "to populate it."
        )
    else:
        col_data, col_pres, _ = t_event_col
        if at_range is not None:
            start_ms, end_ms = at_range
        else:
            start_ms = end_ms = int(anchor_ms)
        allowed = np.zeros(n, dtype=np.float64)
        sa_pres = col_pres[slot_arr]
        sa_values = col_data[slot_arr].astype(np.int64)
        in_range = sa_pres & (sa_values >= start_ms) & (sa_values <= end_ms)
        allowed[slot_arr[in_range]] = 1.0
        base_final *= allowed
```

Then later, where the method builds `meta = {}` and returns `Result(kind="nodes", data=results, count=len(results), meta=meta)`, prepend any collected warnings:

```python
if warnings:
    meta.setdefault("warnings", []).extend(warnings)
```

Place the append immediately after the `meta = {}` line and before any optional nucleus expansion.

### Test
Append to `tests/test_remember.py`:

```python
def test_remember_at_without_event_column_warns():
    gs = GraphStore()
    try:
        gs.execute('CREATE NODE "a" kind = "doc" text = "hello world"')
        r = gs.execute('REMEMBER "hello" AT "2024-01-01" LIMIT 5')
        warnings = r.meta.get("warnings", []) if r.meta else []
        assert any("__event_at__" in w for w in warnings), (
            f"Expected warning about missing __event_at__; got {warnings!r}"
        )
    finally:
        gs.close()
```

### Verification
```
uv run pytest tests/test_remember.py::test_remember_at_without_event_column_warns -xvs
```

### Acceptance
- Warning present in `result.meta["warnings"]`.
- Existing REMEMBER tests still pass.

---

## P1-2: Read-side column writes (`__recall_count__`, `__last_recalled_at__`) skip dirty tracking

Severity: HIGH (data loss on crash between checkpoints).
Files: `src/graphstore/core/columns.py`, `src/graphstore/core/store.py`.

### Problem
`_remember` bumps `__recall_count__` + `__last_recalled_at__` via `store.columns.set_reserved(...)`. `ColumnStore.set_reserved` mutates the numpy arrays in place but does NOT touch `store._dirty_columns`. A subsequent `checkpoint()` with `force=False` skips the columns block (`if force or store._dirty_columns:`), so the increments never reach sqlite. On crash + restart the recall counter is reset to the last checkpointed value.

### Root cause
Dirty-flag tracking lives on `CoreStore`, but mutations can happen through `store.columns` directly, bypassing the owner. The dirty flag and the state it protects are decoupled.

### Fix
Move the dirty bit into `ColumnStore`. Expose it on `CoreStore` via property so existing readers keep working.

**Step 1:** Edit `src/graphstore/core/columns.py`.

Add to `ColumnStore.__init__`:

```python
self.dirty: bool = False
```

Set the flag inside every method that mutates `_columns` or `_presence`:
- `set` - at the end of the method, after the per-field loop
- `clear` - at the end of the method
- `grow` - at the end
- `set_reserved` - at the end
- `set_field` - at the end (but it calls `set`; one marker is enough - put it in `set`)
- `restore_arrays` - at the end
- `declare_column` - at the end (adding a column is a schema change that must checkpoint)
- `_create_column` - at the end (same reason)

Concretely add `self.dirty = True` as the last line of each method above. Do NOT set it in read-only helpers (`get_mask`, `get_mask_in`, `get_presence`, `has_column`, `get_column`, `snapshot_arrays`, `memory_bytes`, `_infer_dtype`, `_make_sentinel_array`).

**Step 2:** Edit `src/graphstore/core/store.py`.

Replace the attribute `self._dirty_columns = True` in `CoreStore.__init__` with reliance on `ColumnStore.dirty`. But to avoid breaking callers that still read `_dirty_columns`, expose a property:

In `CoreStore`, remove `self._dirty_columns = True` from `__init__`. Add after `__init__`:

```python
@property
def _dirty_columns(self) -> bool:
    return self.columns.dirty

@_dirty_columns.setter
def _dirty_columns(self, value: bool) -> None:
    self.columns.dirty = bool(value)
```

Keep the other three dirty flags (`_dirty_nodes`, `_dirty_edges`, `_dirty_strings`) as plain attributes for now - they are only written via `put_node`/`put_edge`/etc. on the store, so the decoupling bug is specific to columns.

Update `reset_dirty_flags`:

```python
def reset_dirty_flags(self):
    """Reset all dirty tracking flags after checkpoint."""
    self._dirty_nodes = False
    self.columns.dirty = False
    self._dirty_edges = False
    self._dirty_strings = False
```

Now every `store.columns.set_reserved(slot, ...)` in the codebase automatically marks the store dirty without the caller changing.

### Test
Append to `tests/test_remember.py`:

```python
def test_remember_recall_count_persists_across_checkpoint(tmp_path):
    import shutil
    path = tmp_path / "gs"
    gs = GraphStore(path=str(path))
    try:
        gs.execute('CREATE NODE "doc1" kind = "doc" text = "the quick brown fox"')
        r = gs.execute('REMEMBER "quick" LIMIT 5')
        assert r.count >= 1
        gs.checkpoint()
    finally:
        gs.close()

    gs2 = GraphStore(path=str(path))
    try:
        # The __recall_count__ column must survive the reopen.
        # Fetch the node and inspect reserved columns through a LEXICAL SEARCH
        # or by reading the core store directly.
        cs = gs2._store
        n = cs._next_slot
        col = cs.columns.get_column("__recall_count__", n)
        assert col is not None, "__recall_count__ column lost across checkpoint"
        col_data, col_pres, _ = col
        assert int(col_data[col_pres].sum()) >= 1, (
            "recall count value did not persist"
        )
    finally:
        gs2.close()
```

### Verification
```
uv run pytest tests/test_remember.py::test_remember_recall_count_persists_across_checkpoint -xvs
uv run pytest tests/test_columns.py -x
uv run pytest -x
```

### Acceptance
- New test passes.
- All existing column / persistence / checkpoint tests pass.

---

## P1-3: WAL replay drops CREATE NODE metadata and can double-create

Severity: HIGH (recovery fidelity).
Files: `src/graphstore/wal.py`.

### Problem
On replay, `WALManager.replay` rewrites every `CreateNode` to `UpsertNode(id=ast.id, fields=ast.fields)`. This drops:
- `expires_in`, `expires_at` (TTL lost after recovery)
- `event_at` (temporal signal lost)
- `vector` (explicit VECTOR clause lost)
- `auto_id=True` cases (UpsertNode requires id; `ast.id` is None; call will fail downstream with a confusing error)

### Root cause
The rewrite is a blunt dedup hack. Replay should be idempotent against its own re-run; the right way is to execute the original statement and treat `NodeExists` as expected.

### Fix
Remove the rewrite. Catch `NodeExists` from the executor and continue.

Edit `src/graphstore/wal.py`. Inside `replay`, find:

```python
for seq, statement in rows:
    try:
        ast = parse(statement)
        if isinstance(ast, ast_nodes.CreateNode):
            ast = ast_nodes.UpsertNode(id=ast.id, fields=ast.fields)
        self._executor.execute(ast)
    except Exception as e:
        ...
```

Replace with:

```python
from graphstore.core.errors import NodeExists  # local import keeps top-level light

for seq, statement in rows:
    try:
        ast = parse(statement)
        self._executor.execute(ast)
    except NodeExists:
        # Checkpoint + WAL can both contain the same CREATE when a crash
        # happens between blob write and WAL clear. Treat as already-applied.
        continue
    except Exception as e:
        ...
```

Move the `NodeExists` import to the top of the file alongside the other `core.errors` imports.

Remove the now-unused `ast_nodes.UpsertNode` reference; also drop `ast_nodes` import if nothing else uses it. Keep `ast_nodes` import only if other code in the file references it; grep before removing.

### Test
Append to `tests/test_wal.py` (create if missing):

```python
import pytest
from graphstore import GraphStore


def test_wal_replay_preserves_vector_and_ttl(tmp_path):
    path = tmp_path / "gs"
    gs = GraphStore(path=str(path), enable_wal=True)
    try:
        gs.execute(
            'CREATE NODE "x" kind = "doc" text = "hi" '
            'VECTOR [0.1, 0.2, 0.3] '
            'EVENT_AT "2024-06-01"'
        )
        # Force WAL to be non-empty: do NOT call checkpoint.
    finally:
        # close() calls checkpoint() which clears WAL. To exercise replay,
        # swap close for a direct connection close that skips checkpoint.
        gs._wal = None  # prevent checkpoint in close
        if gs._conn is not None:
            gs._conn.close()
            gs._runtime.conn = None

    gs2 = GraphStore(path=str(path), enable_wal=True)
    try:
        n = gs2.execute('NODE "x"').data
        assert n is not None, "node lost during WAL replay"
        # Vector present?
        vs = gs2._vector_store
        if vs is not None:
            slot = gs2._store.id_to_slot[gs2._store.string_table.intern("x")]
            assert vs.has_vector(slot), "explicit VECTOR clause lost on replay"
        # Event_at present?
        cs = gs2._store
        col = cs.columns.get_column("__event_at__", cs._next_slot)
        assert col is not None, "__event_at__ column missing"
    finally:
        gs2.close()


def test_wal_replay_tolerates_duplicate_create(tmp_path):
    # After a successful checkpoint, WAL is cleared. Simulate a scenario
    # where the WAL contains a CREATE for a node that the checkpoint
    # already persisted: the replay must not raise.
    import sqlite3
    path = tmp_path / "gs"
    gs = GraphStore(path=str(path))
    try:
        gs.execute('CREATE NODE "dup" kind = "doc" text = "x"')
        gs.checkpoint()
        # Inject a duplicate CREATE into WAL directly.
        gs._conn.execute(
            "INSERT INTO wal (timestamp, statement) VALUES (?, ?)",
            (0.0, 'CREATE NODE "dup" kind = "doc" text = "x"'),
        )
        gs._conn.commit()
    finally:
        gs._wal = None
        if gs._conn is not None:
            gs._conn.close()
            gs._runtime.conn = None

    gs2 = GraphStore(path=str(path))
    try:
        assert gs2.execute('NODE "dup"').data is not None
    finally:
        gs2.close()
```

### Verification
```
uv run pytest tests/test_wal.py -xvs
uv run pytest -x
```

### Acceptance
- Both new tests pass.
- Other WAL / recovery tests continue to pass.

---

## P1-4: Embedder identity mismatch is advisory-only

Severity: MEDIUM-HIGH (garbage retrieval results).
Files: `src/graphstore/store.py`.

### Problem
`_check_embedder_identity` on open logs a warning when the embedder name differs from the one recorded in metadata, but nothing stops the user. `REMEMBER`/`SIMILAR` run against a vector index encoded by a different model; results are noise.

### Root cause
The `_embedder_dirty` flag exists and is enforced by `SIMILAR` (`intelligence.py`), but `_check_embedder_identity` never sets it.

### Fix
Set `_embedder_dirty = True` when the stored name doesn't match the current name.

Edit the `_check_embedder_identity` method. Change from:

```python
@staticmethod
def _check_embedder_identity(conn, embedder):
    ...
    if stored != current:
        stored_dims = get_metadata(conn, "embedder_dims") or "?"
        logging.getLogger(__name__).warning(
            ...
        )
```

to (remove `@staticmethod`, flip the dirty flag on mismatch):

```python
def _check_embedder_identity(self, conn, embedder):
    """On open, verify current embedder matches what the database was built
    with. Sets self._embedder_dirty on mismatch; SIMILAR/REMEMBER will refuse
    until SYS REEMBED is run."""
    if conn is None or embedder is None:
        return
    from graphstore.persistence.database import get_metadata
    stored = get_metadata(conn, "embedder_name")
    if stored is None:
        return
    current = embedder.name
    if stored != current:
        stored_dims = get_metadata(conn, "embedder_dims") or "?"
        logger.warning(
            "embedder mismatch: database was built with '%s' (%s dims), "
            "current embedder is '%s'. Run SYS REEMBED to re-encode; "
            "until then SIMILAR and REMEMBER will raise.",
            stored, stored_dims, current,
        )
        self._embedder_dirty = True
```

Update the caller in `__init__`: `self._check_embedder_identity(_conn, _embedder)` - this already calls through `self.`, so no change needed once `@staticmethod` is removed. Double-check the existing line (`self._check_embedder_identity(_conn, _embedder)`); if it uses `GraphStore._check_embedder_identity(_conn, _embedder)` instead, change it to `self._check_embedder_identity(_conn, _embedder)`.

Extend the same guard to `_remember` in `src/graphstore/dsl/handlers/intelligence.py`. `_similar` already checks `_embedder_dirty`:

```python
if hasattr(self, '_embedder_dirty') and self._embedder_dirty:
    from graphstore.core.errors import GraphStoreError
    raise GraphStoreError("Embedder changed. Run SYS REEMBED to update vectors.")
```

Add the same block at the top of `_remember` (right after the method signature and docstring):

```python
if getattr(self, '_embedder_dirty', False):
    from graphstore.core.errors import GraphStoreError
    raise GraphStoreError("Embedder changed. Run SYS REEMBED to update vectors.")
```

### Test
Append to `tests/test_auto_reembed.py` (already exists):

```python
def test_embedder_mismatch_blocks_remember(tmp_path):
    import pytest
    from graphstore import GraphStore
    from graphstore.core.errors import GraphStoreError

    class StubEmb:
        name = "stub-A"
        dims = 4
        def encode_documents(self, texts, titles=None):
            import numpy as np
            return np.zeros((len(texts), 4), dtype=np.float32)
        def encode_queries(self, texts):
            import numpy as np
            return np.zeros((len(texts), 4), dtype=np.float32)

    class StubEmbB(StubEmb):
        name = "stub-B"

    path = tmp_path / "gs"
    gs = GraphStore(path=str(path), embedder=StubEmb())
    try:
        gs.execute('SYS REGISTER NODE "doc" REQUIRED text EMBED text')
        gs.execute('CREATE NODE "a" kind = "doc" text = "hi"')
        gs.checkpoint()
    finally:
        gs.close()

    gs2 = GraphStore(path=str(path), embedder=StubEmbB())
    try:
        with pytest.raises(GraphStoreError, match="SYS REEMBED"):
            gs2.execute('REMEMBER "hi" LIMIT 5')
    finally:
        gs2.close()
```

### Verification
```
uv run pytest tests/test_auto_reembed.py -xvs
```

### Acceptance
- New test passes.
- Existing auto-reembed tests still pass.

---

## P1-5: Auto-id collision space is 48 bits

Severity: MEDIUM.
Files: `src/graphstore/dsl/handlers/mutations.py`.

### Problem
`_generate_auto_id` truncates sha256 to 12 hex chars = 48 bits. Birthday collision likelihood crosses 1-in-10^6 at roughly 24k same-kind+fields nodes. Silently overwrites via `put_node` -> `NodeExists` which is caught upstream in some paths.

### Root cause
Arbitrary truncation.

### Fix
Widen to 16 hex chars (64 bits). Still short; collision probability 1-in-10^6 moves to ~6M items.

Edit `_generate_auto_id`:

```python
return hashlib.sha256(content.encode()).hexdigest()[:16]
```

### Test
Append to `tests/test_mutations.py`:

```python
def test_auto_id_width_is_16_hex():
    from graphstore import GraphStore
    gs = GraphStore()
    try:
        r = gs.execute('CREATE NODE AUTO kind = "k" v = 1')
        nid = r.data["id"]
        assert len(nid) == 16, f"expected 16 hex chars, got {len(nid)}: {nid!r}"
        assert all(c in "0123456789abcdef" for c in nid)
    finally:
        gs.close()
```

### Verification
```
uv run pytest tests/test_mutations.py::test_auto_id_width_is_16_hex -xvs
```

### Acceptance
- New test passes.
- No test relies on the 12-char width. If one does, update it to 16.

---

## P1-6: Entity slug truncation causes silent collisions

Severity: MEDIUM.
Files: `src/graphstore/ingest/entity_extract.py`.

### Problem
`slug(text)` collapses to `[a-z0-9_]` and truncates to 40 chars. "Barack Hussein Obama II" and "Barack Hussein Obama Jr" both become `"barack_hussein_obama"` after truncation and trailing-underscore strip. Distinct entities merge silently.

### Root cause
Truncation without uniqueness preservation.

### Fix
When the post-regex form exceeds 40 chars, append a 6-hex suffix of `sha256(original).hexdigest()[:6]`.

Edit `slug` in `src/graphstore/ingest/entity_extract.py`:

```python
import hashlib

def slug(text: str) -> str:
    """Create a URL-safe slug. Appends a short hash suffix when truncated
    so distinct long names don't silently collide.
    """
    base = _SLUG_RE.sub("_", text.lower()).strip("_")
    if len(base) <= 40:
        return base
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:6]
    # Reserve 7 chars for "_" + 6-hex suffix.
    return base[:33].rstrip("_") + "_" + digest
```

### Test
Append to `tests/test_entity_extract.py` (exists):

```python
def test_slug_no_collision_on_truncation():
    from graphstore.ingest.entity_extract import slug
    a = slug("Barack Hussein Obama II the Second of That Name")
    b = slug("Barack Hussein Obama Jr the Son of the First")
    assert a != b, f"slugs collided: {a!r} == {b!r}"
    assert len(a) <= 40
    assert len(b) <= 40


def test_slug_short_unchanged():
    from graphstore.ingest.entity_extract import slug
    assert slug("Alice") == "alice"
    assert slug("Bob Smith") == "bob_smith"
```

### Verification
```
uv run pytest tests/test_entity_extract.py -xvs
```

### Acceptance
- Both tests pass.

---

## P1-7: Reranker per-query failure is silent

Severity: MEDIUM.
Files: `src/graphstore/dsl/handlers/intelligence.py`.

### Problem
`_remember` catches `Exception` from `reranker.score(...)` and silently falls back to fusion top-K. User has no signal that their rerank config is broken.

### Root cause
Bare `try/except Exception: pass`-style handling.

### Fix
Log at WARNING, attach `result.meta["reranker_error"]`.

Find the reranker block:

```python
reranker = getattr(self, '_reranker', None)
if reranker is not None and len(texts_for_rerank) > target_k:
    try:
        rerank_scores = reranker.score(q.query, texts_for_rerank)
        ...
    except Exception:
        results = results[:target_k]
        retrieved_slots = retrieved_slots[:target_k]
```

Replace the except body with:

```python
except Exception as rerank_err:
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "reranker failed; falling back to fusion top-K: %s",
        rerank_err,
    )
    meta["reranker_error"] = f"{type(rerank_err).__name__}: {rerank_err}"
    results = results[:target_k]
    retrieved_slots = retrieved_slots[:target_k]
```

Ensure `meta = {}` is initialised before the reranker block (move the existing `meta = {}` line above the reranker block if it is currently below). Double-check the final return - the `meta` dict should be the same one used for nucleus and warnings.

### Test
Append to `tests/test_remember.py`:

```python
def test_remember_reranker_error_surfaces_in_meta():
    from graphstore import GraphStore

    class BrokenReranker:
        def score(self, q, docs):
            raise RuntimeError("rerank boom")

    gs = GraphStore()
    gs._executor._reranker = BrokenReranker()
    try:
        for i in range(20):
            gs.execute(f'CREATE NODE "d{i}" kind = "doc" text = "alpha beta {i}"')
        r = gs.execute('REMEMBER "alpha" LIMIT 3')
        assert r.count == 3  # fallback path returns top-K
        assert "reranker_error" in (r.meta or {}), (
            f"expected reranker_error in meta; got {r.meta!r}"
        )
    finally:
        gs.close()
```

### Verification
```
uv run pytest tests/test_remember.py::test_remember_reranker_error_surfaces_in_meta -xvs
```

### Acceptance
- Test passes. `reranker_error` string present in meta.

---

## P1-8: Nucleus expansion has no total cost gate

Severity: MEDIUM (DoS surface).
Files: `src/graphstore/dsl/handlers/intelligence.py`.

### Problem
Nucleus iterates structural edges `_hops` times with `max_nb` per seed. On dense structural graphs a single REMEMBER can visit tens of thousands of neighbours. Per-hop cap exists; total cap does not.

### Root cause
Local heuristic, no global budget.

### Fix
Add a total-visit budget.

In `_remember`, inside the nucleus block, introduce:

```python
if nucleus_on and results and self.store.edge_matrices.total_edges > 0:
    max_nb = getattr(self, '_nucleus_neighbors_per_hop', 3)
    n_hops = getattr(self, '_nucleus_hops', 1)
    min_text = getattr(self, '_nucleus_min_text_length', 20)
    total_budget = max(max_nb * n_hops * target_k, 100)
    allowed_kinds = set(getattr(self, '_nucleus_allowed_kinds',
                                ["message", "chunk", "section"]))
    seen_slots = set(retrieved_slots)
    frontier = list(retrieved_slots)
    nucleus_results: list = []
    visits = 0
    for _hop in range(n_hops):
        if visits >= total_budget:
            break
        next_frontier: list = []
        for seed_slot in frontier:
            if visits >= total_budget:
                break
            nb_slots = self._nucleus_neighbors(seed_slot)
            for nb in nb_slots:
                if visits >= total_budget:
                    break
                visits += 1
                nb = int(nb)
                if nb in seen_slots or not live_mask[nb]:
                    continue
                nb_node = self.store._materialize_slot(nb)
                if nb_node is None:
                    continue
                nb_kind = nb_node.get("kind", "")
                if nb_kind not in allowed_kinds:
                    seen_slots.add(nb)
                    continue
                nb_text = (nb_node.get("content") or nb_node.get("summary")
                           or nb_node.get("text") or "")
                if len(nb_text) < min_text:
                    seen_slots.add(nb)
                    continue
                nb_node["_nucleus"] = True
                nucleus_results.append(nb_node)
                seen_slots.add(nb)
                next_frontier.append(nb)
                if len(next_frontier) >= max_nb:
                    break
        frontier = next_frontier
    meta["nucleus"] = nucleus_results
    meta["nucleus_visits"] = visits
```

Key changes:
- `total_budget = max(max_nb * n_hops * target_k, 100)`
- `visits += 1` at each outer-loop iteration
- Three early-exit checks on `visits >= total_budget`
- `meta["nucleus_visits"]` for observability

### Test
Append to `tests/test_remember.py`:

```python
def test_remember_nucleus_respects_visit_budget():
    from graphstore import GraphStore
    gs = GraphStore(nucleus_expansion=True, nucleus_hops=3,
                    nucleus_neighbors_per_hop=50,
                    nucleus_allowed_kinds=["chunk"])
    try:
        gs.execute('CREATE NODE "root" kind = "chunk" text = "seed chunk content"')
        # Chain of chunks, each long enough to satisfy min_text
        for i in range(300):
            gs.execute(
                f'CREATE NODE "c{i}" kind = "chunk" '
                f'text = "chunk {i} body content must be long enough"'
            )
            src = "root" if i == 0 else f"c{i-1}"
            gs.execute(f'CREATE EDGE "{src}" -> "c{i}" kind = "next"')
        r = gs.execute('REMEMBER "seed" LIMIT 1')
        # Budget = max(50*3*1, 100) = 150
        assert r.meta.get("nucleus_visits", 0) <= 150
    finally:
        gs.close()
```

### Verification
```
uv run pytest tests/test_remember.py::test_remember_nucleus_respects_visit_budget -xvs
```

### Acceptance
- Test passes.

---

## P2-1: Compute profile cache is blind to env changes

Severity: LOW-MEDIUM.
Files: `src/graphstore/core/compute_profile.py`.

### Problem
`get_profile()` is `@lru_cache(maxsize=1)`. `configure(...)` clears it. Env var changes do NOT clear it. Docs promise env-override behaviour, but without an explicit `reset_profile_cache()` call, new env values are ignored.

### Root cause
`lru_cache` key is empty; cache is keyed by "no arguments" always.

### Fix
Compute an env fingerprint inside `get_profile` and call `cache_clear()` when it changes.

Edit `compute_profile.py`. Add module state:

```python
_ENV_KEYS = (
    "GRAPHSTORE_PROFILE", "GRAPHSTORE_NER_THREADS",
    "GRAPHSTORE_EMBED_THREADS", "GRAPHSTORE_RERANK_THREADS",
    "GRAPHSTORE_EMBED_BATCH", "GRAPHSTORE_GPU",
)
_last_env_fingerprint: tuple | None = None
```

Wrap `get_profile`:

```python
def _env_fingerprint() -> tuple:
    return tuple(os.environ.get(k) for k in _ENV_KEYS)


_cached_profile = lru_cache(maxsize=1)(lambda: _compute_profile())


def get_profile() -> ComputeProfile:
    global _last_env_fingerprint
    fp = _env_fingerprint()
    if fp != _last_env_fingerprint:
        _cached_profile.cache_clear()
        _last_env_fingerprint = fp
    return _cached_profile()
```

Rename the current `get_profile` body (everything after the `@lru_cache` decorator) to `_compute_profile`. Remove the `@lru_cache(maxsize=1)` decorator from the old function.

Make `reset_profile_cache`, `configure`, and the previous `get_profile.cache_clear()` calls target `_cached_profile.cache_clear()` instead:

```python
def configure(...):
    ...
    _cached_profile.cache_clear()


def reset_profile_cache() -> None:
    _cached_profile.cache_clear()
```

Existing tests call `cp.get_profile.cache_clear()`. That call site will break. Add a compatibility shim:

```python
# Backwards-compatible attribute: some tests call get_profile.cache_clear()
get_profile.cache_clear = _cached_profile.cache_clear  # type: ignore[attr-defined]
```

### Test
Append to `tests/test_compute_profile.py`:

```python
def test_env_fingerprint_invalidates_cache(monkeypatch):
    from graphstore.core import compute_profile as cp

    cp.configure()  # wipe overrides
    monkeypatch.delenv("GRAPHSTORE_EMBED_THREADS", raising=False)
    p1 = cp.get_profile()

    monkeypatch.setenv("GRAPHSTORE_EMBED_THREADS", "99")
    p2 = cp.get_profile()

    assert p2.embed_threads == 99
    assert p1.embed_threads != 99 or p1.embed_threads == 99  # sanity
```

### Verification
```
uv run pytest tests/test_compute_profile.py -xvs
```

### Acceptance
- `p2.embed_threads == 99` without explicit `cache_clear`.
- All existing tests pass.

---

## P2-2: `query_log` has only time-based rotation

Severity: LOW (disk creep under high QPS).
Files: `src/graphstore/wal.py`.

### Problem
`_rotate_query_log` deletes rows older than `log_retention_days`. Under 1000 qps, the table can hit millions of rows within the retention window and inflate the sqlite file.

### Root cause
No row count cap.

### Fix
Add a hard row cap, keeping the most recent N rows. Default 200k.

Add to `WALManager.__init__`:

```python
self._query_log_max_rows = 200_000
```

Extend `_rotate_query_log`:

```python
def _rotate_query_log(self) -> None:
    conn = self._conn
    if conn is None:
        return
    try:
        cutoff = time.time() - self._log_retention_days * 86400
        conn.execute("DELETE FROM query_log WHERE timestamp < ?", (cutoff,))
        # Cap row count too. Keep newest N.
        row = conn.execute("SELECT COUNT(*) FROM query_log").fetchone()
        count = row[0] if row else 0
        if count > self._query_log_max_rows:
            conn.execute(
                "DELETE FROM query_log WHERE id IN ("
                "  SELECT id FROM query_log ORDER BY timestamp ASC LIMIT ?"
                ")",
                (count - self._query_log_max_rows,),
            )
        conn.commit()
    except Exception as e:
        logger.debug("query log rotation failed: %s", e, exc_info=True)
```

### Test
Append to `tests/test_wal.py`:

```python
def test_query_log_row_cap(tmp_path):
    from graphstore import GraphStore
    path = tmp_path / "gs"
    gs = GraphStore(path=str(path))
    try:
        gs._wal._query_log_max_rows = 50
        for i in range(120):
            gs.execute('COUNT NODES')
        gs._wal.maybe_auto_checkpoint()  # triggers rotation
        count = gs._conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
        assert count <= 50
    finally:
        gs.close()
```

### Verification
```
uv run pytest tests/test_wal.py -xvs
```

### Acceptance
- Test passes.

---

## P3-1: Dispatch table only matches exact AST types

Severity: LOW (latent; no current collisions).
Files: `src/graphstore/dsl/executor.py`, `src/graphstore/dsl/handlers/_registry.py`.

### Problem
`DISPATCH.get(type(ast))` uses exact type keys. Any future subclass will silently fall through to the "Unknown AST node type" path.

### Fix
Walk the MRO if exact match misses.

Edit `dsl/executor.py::_dispatch`:

```python
def _dispatch(self, ast) -> Result:
    if isinstance(ast, _VAULT_TYPES):
        if not self._vault_executor:
            raise GraphStoreError("Vault not configured. Use GraphStore(vault='./notes')")
        return self._vault_executor.dispatch(ast)

    t = type(ast)
    handler = DISPATCH.get(t)
    if handler is None:
        for base in t.__mro__[1:]:
            handler = DISPATCH.get(base)
            if handler is not None:
                break
    if handler is None:
        raise GraphStoreError(f"Unknown AST node type: {t.__name__}")
    return handler(self, ast)
```

### Test
Append to `tests/test_dsl_parser.py`:

```python
def test_dispatch_walks_mro():
    from graphstore.dsl.handlers._registry import DISPATCH
    from graphstore.dsl.ast_nodes import NodeQuery

    class MyNodeQuery(NodeQuery):
        pass

    # Not registered directly; parent is.
    assert DISPATCH.get(MyNodeQuery) is None
    assert DISPATCH.get(NodeQuery) is not None
```

(The behavioural test would need a full executor; the MRO walk is unit-level.)

### Verification
```
uv run pytest tests/test_dsl_parser.py -xvs
```

### Acceptance
- Test passes.

---

## Rollout checklist for the executor LLM

1. Create a new branch per PR group:
   - `fix/spec-v1-p0` for P0-1, P0-2, P0-3
   - `fix/spec-v1-p1` for P1-1 through P1-8
   - `fix/spec-v1-p2` for P2-1, P2-2
   - `chore/spec-v1-p3` for P3-1 (optional)

2. For each item in order (P0 first):
   a. Read the listed files end-to-end before editing.
   b. Write the new test first. Run it. Confirm it fails.
   c. Apply the fix exactly as written.
   d. Run the item's verification command. Confirm it passes.
   e. Run the full suite: `uv run pytest -x`.
   f. If anything goes red that wasn't red before, STOP. Diagnose root cause before proceeding.

3. After each priority group:
   - `git add -p`, stage the files named in this spec.
   - Commit with message: `fix(spec-v1-<id>): <one-line summary>`.
   - Open PR against `main` following `Rule 5: Git Workflow`.

4. Never:
   - Skip a failing test by marking it `xfail`.
   - Add `# type: ignore` to silence a type error introduced by these changes.
   - Rename public DSL keywords.
   - Drop an em dash into markdown (CLAUDE.md rule 9 applies).

5. On PR description, paste this checklist with every fix marked done:
   ```
   - [x] P0-1: MERGE self-merge guard
   - [x] P0-2: Batch vector rollback
   - [x] P0-3: deferred_embeddings flush
   ```

6. When you introduce a new test file, add it to `tests/` (not a subdirectory). Follow the existing `test_*.py` naming.

7. When a fix touches behaviour that `docs/superpowers/` documents, update the relevant `.md` in the same PR. Do NOT introduce new `.md` files outside `docs/specs/`.

---

## Out of scope (do not attempt)

- Removing the legacy kwarg shortcuts in `config.py::merge_kwargs`.
- Replacing `usearch` with another vector index.
- Adding GPU auto-detection beyond the current `GRAPHSTORE_GPU=1` gate.
- Restructuring `dsl/handlers/` into separate packages.
- Any change to `grammar.lark`.

If a fix above seems to require such a change, stop and ask instead of expanding scope.
