"""Self-balancing optimizer: 6 operations that clean up accumulated pressure.

All operations assume exclusive access (no concurrent reads/writes).
The caller (GraphStore.execute) enforces this via the _optimizing lock.
"""

from __future__ import annotations

import numpy as np

from graphstore.core.store import CoreStore
from graphstore.core.strings import StringTable


def health_check(store: CoreStore, vector_store=None, document_store=None) -> dict:
    """Compute pressure metrics. Lightweight, no mutations."""
    n = store._next_slot
    tombstone_count = len(store.node_tombstones)
    tombstone_ratio = tombstone_count / max(n, 1)

    string_count = len(store.string_table)
    live_nodes = store.node_count
    string_bloat = string_count / max(live_nodes, 1)

    dead_vectors = 0
    total_vectors = 0
    if vector_store is not None:
        total_vectors = vector_store.count()
        for slot in range(min(n, vector_store._capacity)):
            if vector_store._has_vector[slot] and slot in store.node_tombstones:
                dead_vectors += 1

    edge_keys_count = len(store._edge_keys)
    actual_edges = sum(len(v) for v in store._edges_by_type.values())
    stale_edge_keys = edge_keys_count - actual_edges

    from graphstore.dsl.parser import _plan_cache
    cache_size = len(_plan_cache)

    return {
        "tombstone_ratio": round(tombstone_ratio, 3),
        "tombstone_count": tombstone_count,
        "total_slots": n,
        "live_nodes": live_nodes,
        "string_count": string_count,
        "string_bloat": round(string_bloat, 2),
        "dead_vectors": dead_vectors,
        "total_vectors": total_vectors,
        "stale_edge_keys": max(stale_edge_keys, 0),
        "cache_size": cache_size,
    }


def needs_optimization(health: dict, compact_threshold: float = 0.2,
                       string_gc_threshold: float = 3.0,
                       cache_gc_threshold: int = 200) -> list[str]:
    """Determine which operations are needed based on health metrics."""
    ops = []
    if health["tombstone_ratio"] > compact_threshold:
        ops.append("COMPACT")
    if health["string_bloat"] > string_gc_threshold and health["live_nodes"] > 0:
        ops.append("STRINGS")
    if health["dead_vectors"] > 0:
        ops.append("VECTORS")
    if health["stale_edge_keys"] > 0:
        ops.append("EDGES")
    if health["cache_size"] > cache_gc_threshold:
        ops.append("CACHE")
    return ops


def compact_tombstones(store: CoreStore, vector_store=None, document_store=None) -> dict:
    """Shift live slots down to eliminate tombstone gaps.

    Renumbers all slot references: node_ids, node_kinds, columns,
    edges, id_to_slot, vectors, DocumentStore, FTS5.
    Uses numpy fancy indexing for bulk array operations.
    """
    if not store.node_tombstones:
        return {"compacted": 0}

    n = store._next_slot

    # Vectorized live slot detection
    live_mask = store.node_ids[:n] >= 0
    if store.node_tombstones:
        tomb_arr = np.array(list(store.node_tombstones), dtype=np.int32)
        valid = tomb_arr[tomb_arr < n]
        if len(valid) > 0:
            live_mask[valid] = False

    live_slots_arr = np.nonzero(live_mask)[0]
    new_count = len(live_slots_arr)

    if new_count == n:
        return {"compacted": 0}

    # Build remapping via numpy
    old_to_new = np.full(n, -1, dtype=np.int32)
    old_to_new[live_slots_arr] = np.arange(new_count, dtype=np.int32)
    old_to_new_dict = {int(old): int(old_to_new[old]) for old in live_slots_arr}

    new_capacity = max(new_count * 2, store._capacity)

    # Remap node arrays via fancy indexing
    new_ids = np.full(new_capacity, -1, dtype=np.int32)
    new_ids[:new_count] = store.node_ids[live_slots_arr]
    new_kinds = np.zeros(new_capacity, dtype=np.int32)
    new_kinds[:new_count] = store.node_kinds[live_slots_arr]

    # Remap columns via fancy indexing (no Python loop over slots)
    for field in list(store.columns._columns.keys()):
        old_col = store.columns._columns[field]
        old_pres = store.columns._presence[field]
        dtype_str = store.columns._dtypes[field]
        new_col = store.columns._make_sentinel_array(dtype_str, new_capacity)
        new_pres = np.zeros(new_capacity, dtype=bool)
        new_col[:new_count] = old_col[live_slots_arr]
        new_pres[:new_count] = old_pres[live_slots_arr]
        store.columns._columns[field] = new_col
        store.columns._presence[field] = new_pres
    store.columns._capacity = new_capacity

    # Remap edges using numpy lookup array
    for etype in list(store._edges_by_type.keys()):
        remapped = []
        for s, t, d in store._edges_by_type[etype]:
            ns = old_to_new[s] if s < n else -1
            nt = old_to_new[t] if t < n else -1
            if ns >= 0 and nt >= 0:
                remapped.append((int(ns), int(nt), d))
        if remapped:
            store._edges_by_type[etype] = remapped
        else:
            del store._edges_by_type[etype]

    store._edge_keys = {
        (s, t, k) for k, edges in store._edges_by_type.items() for s, t, _d in edges
    }

    # Remap vectors
    if vector_store is not None:
        old_vectors: dict[int, np.ndarray] = {}
        for old_slot in live_slots_arr:
            old_slot = int(old_slot)
            if vector_store.has_vector(old_slot):
                old_vectors[old_slot] = vector_store.get_vector(old_slot)

        # Clear and re-add with new slots
        vector_store._has_vector = np.zeros(new_capacity, dtype=bool)
        vector_store._capacity = new_capacity
        from usearch.index import Index
        vector_store._index = Index(ndim=vector_store._dims, metric="cos", dtype="f32")
        for old_slot, vec in old_vectors.items():
            new_slot = old_to_new_dict[old_slot]
            vector_store._index.add(new_slot, vec)
            vector_store._has_vector[new_slot] = True

    # Remap DocumentStore slot keys via temp-table batch (O(1) statements vs O(N))
    if document_store is not None:
        conn = document_store._conn

        # Build live-slot temp table — used for both orphan deletion and remap.
        conn.execute(
            "CREATE TEMP TABLE _live_slots (slot INT PRIMARY KEY)"
        )
        conn.executemany(
            "INSERT INTO _live_slots VALUES (?)",
            [(s,) for s in old_to_new_dict.keys()]
        )

        # Delete tombstoned slot rows. Tombstoned slots are not in _live_slots
        # but may still occupy rows in document tables; leaving them causes
        # UNIQUE constraint violations when a live slot remaps onto that number.
        _ORPHAN_COLS = [
            ("documents", "slot"),
            ("summaries", "slot"),
            ("images", "slot"),
            ("doc_metadata", "doc_slot"),
        ]
        for tbl, col in _ORPHAN_COLS:
            conn.execute(
                f"DELETE FROM {tbl} WHERE {col} >= 0 AND {col} < ?"  # noqa: S608
                f" AND {col} NOT IN (SELECT slot FROM _live_slots)",
                (n,),
            )

        # Remap live slots that change position (old_slot != new_slot)
        remapped = [(old, new) for old, new in old_to_new_dict.items() if old != new]
        if remapped:
            conn.execute(
                "CREATE TEMP TABLE _slot_remap (old_slot INT PRIMARY KEY, new_slot INT)"
            )
            conn.executemany("INSERT INTO _slot_remap VALUES (?,?)", remapped)

            _REMAP_COLS = [
                ("documents", "slot"),
                ("summaries", "slot"),
                ("summaries", "doc_slot"),
                ("images", "slot"),
                ("doc_metadata", "doc_slot"),
            ]
            # Pass 1: old slot → negative sentinel (avoids collision with valid new slots)
            for tbl, col in _REMAP_COLS:
                conn.execute(
                    f"UPDATE {tbl} SET {col} = -(({col}) + 1)"  # noqa: S608
                    f" WHERE {col} IN (SELECT old_slot FROM _slot_remap)"
                )
            # Pass 2: negative sentinel → new slot
            for tbl, col in _REMAP_COLS:
                conn.execute(
                    f"UPDATE {tbl} SET {col} = ("  # noqa: S608
                    f"  SELECT new_slot FROM _slot_remap"
                    f"  WHERE old_slot = -(({tbl}.{col}) + 1)"
                    f") WHERE {col} IN (SELECT -(old_slot + 1) FROM _slot_remap)"
                )
            conn.execute("DROP TABLE _slot_remap")

        conn.execute("DROP TABLE _live_slots")

        # Rebuild FTS from remapped summaries
        conn.execute("DELETE FROM doc_fts")
        rows = conn.execute("SELECT slot, summary FROM summaries").fetchall()
        for slot, summary in rows:
            conn.execute("INSERT INTO doc_fts (rowid, summary) VALUES (?, ?)", (slot, summary))
        conn.commit()

    # Apply to store
    store.node_ids = new_ids
    store.node_kinds = new_kinds
    store._capacity = new_capacity
    store.node_tombstones.clear()
    store.id_to_slot = {}
    for slot in range(new_count):
        str_id = int(new_ids[slot])
        if str_id >= 0:
            store.id_to_slot[str_id] = slot
    store._next_slot = new_count
    store._edges_dirty = True
    store._ensure_edges_built()

    # Rebuild secondary indices
    for field in list(store._indexed_fields):
        store.add_index(field)

    # Invalidate snapshots (slot numbering changed)
    store._snapshots.clear()

    return {"compacted": n - new_count}


def gc_strings(store: CoreStore) -> dict:
    """Rebuild string table with only referenced strings.

    Remaps all int32 values: node_ids, node_kinds, all int32_interned columns.
    Uses numpy for bulk ID collection instead of Python loops.
    """
    n = store._next_slot

    # Vectorized: collect all referenced string IDs
    live_mask = store.node_ids[:n] >= 0
    if store.node_tombstones:
        tomb_arr = np.array(list(store.node_tombstones), dtype=np.int32)
        valid = tomb_arr[tomb_arr < n]
        if len(valid) > 0:
            live_mask[valid] = False

    live_ids = store.node_ids[:n][live_mask]
    live_kinds = store.node_kinds[:n][live_mask]
    referenced: set[int] = set(live_ids[live_ids >= 0].tolist())
    referenced.update(live_kinds[live_kinds >= 0].tolist())

    for field in store.columns._columns:
        if store.columns._dtypes[field] == "int32_interned":
            col = store.columns._columns[field][:n]
            pres = store.columns._presence[field][:n]
            referenced.update(col[pres].tolist())

    # Also reference edge type names
    for etype in store._edges_by_type:
        if etype in store.string_table:
            referenced.add(store.string_table.intern(etype))

    old_table = store.string_table
    old_count = len(old_table)

    if len(referenced) == old_count:
        return {"strings_freed": 0}

    # Build new table from referenced strings only
    old_to_new: dict[int, int] = {}
    new_strings: list[str] = []
    for old_id in sorted(referenced):
        old_to_new[old_id] = len(new_strings)
        new_strings.append(old_table.lookup(old_id))

    new_table = StringTable.from_list(new_strings)

    # Vectorized remap: build lookup array old_id -> new_id
    max_old = len(old_table)
    remap_arr = np.arange(max_old, dtype=np.int32)  # identity by default
    for old_id, new_id in old_to_new.items():
        remap_arr[old_id] = new_id

    # Remap node_ids and node_kinds via numpy fancy indexing
    ids = store.node_ids[:n]
    valid_ids = ids >= 0
    ids[valid_ids] = remap_arr[ids[valid_ids]]

    kinds = store.node_kinds[:n]
    valid_kinds = live_mask
    kinds[valid_kinds] = remap_arr[kinds[valid_kinds]]

    # Remap int32_interned columns via numpy fancy indexing
    for field in store.columns._columns:
        if store.columns._dtypes[field] == "int32_interned":
            col = store.columns._columns[field][:n]
            pres = store.columns._presence[field][:n]
            col[pres] = remap_arr[col[pres]]

    # Remap id_to_slot keys
    store.id_to_slot = {
        old_to_new[old_str_id]: slot
        for old_str_id, slot in store.id_to_slot.items()
    }

    # Remap edge type names in _edges_by_type
    new_edges: dict[str, list] = {}
    for etype, edges in store._edges_by_type.items():
        new_edges[etype] = edges  # edge type is a string key, not an int
    store._edges_by_type = new_edges

    store.string_table = new_table
    store.columns._string_table = new_table

    # Clear plan cache (cached ASTs may hold stale interned values)
    from graphstore.dsl.parser import clear_cache
    clear_cache()

    # Rebuild secondary indices
    for field in list(store._indexed_fields):
        store.add_index(field)

    return {"strings_freed": old_count - len(new_strings)}


def defrag_edges(store: CoreStore) -> dict:
    """Rebuild edge keys set and CSR matrices from clean edge lists."""
    store._edge_keys = {
        (s, t, k) for k, edges in store._edges_by_type.items() for s, t, _d in edges
    }
    store._edges_dirty = True
    store._ensure_edges_built()
    return {"edge_types": len(store._edges_by_type)}


def cleanup_vectors(store: CoreStore, vector_store) -> dict:
    """Remove HNSW entries for tombstoned/retracted slots."""
    if vector_store is None:
        return {"removed": 0}

    n = store._next_slot
    live = store.compute_live_mask(n)
    removed = 0

    for slot in range(min(n, vector_store._capacity)):
        if vector_store._has_vector[slot] and not live[slot]:
            vector_store.remove(slot)
            removed += 1

    return {"removed": removed}


def sweep_orphans(store: CoreStore, document_store) -> dict:
    """Delete DocumentStore rows not referenced by live slots."""
    if document_store is None:
        return {"cleaned": 0}

    live = store.compute_live_mask(store._next_slot)
    live_slots = set(int(s) for s in np.nonzero(live)[0])
    cleaned = document_store.orphan_cleanup(live_slots)
    return {"cleaned": cleaned}


def clear_caches(store: CoreStore) -> dict:
    """Clear plan cache and edge combination/transpose caches."""
    from graphstore.dsl.parser import clear_cache
    clear_cache()
    store.edge_matrices._cache.clear()
    store.edge_matrices._transpose_cache.clear()
    return {"cleared": True}


def optimize_all(store: CoreStore, vector_store=None, document_store=None) -> dict:
    """Run all 6 optimization operations in safe order.

    Short-circuits on the first failure: once compact_tombstones or
    gc_strings breaks mid-run, the store may be in a half-remapped state
    and running subsequent passes on it just compounds the damage.
    Returns the results collected so far plus {"error": "..."} so the
    caller knows which step died.
    """
    results: dict = {}
    steps: list[tuple[str, callable]] = [
        ("compact", lambda: compact_tombstones(store, vector_store, document_store)),
        ("strings", lambda: gc_strings(store)),
        ("edges",   lambda: defrag_edges(store)),
        ("vectors", lambda: cleanup_vectors(store, vector_store)),
        ("blobs",   lambda: sweep_orphans(store, document_store)),
        ("cache",   lambda: clear_caches(store)),
    ]
    for name, op in steps:
        try:
            results[name] = op()
        except Exception as e:
            results["error"] = f"{name}: {type(e).__name__}: {e}"
            break
    return results


def _get_evictable_slots_sorted(store: CoreStore, protected_kinds: set) -> list[int]:
    """Helper: get ascending sorted list of (slot) by __updated_at__ for oldest-first eviction."""
    n = store._next_slot
    if n == 0:
        return []

    live = store.compute_live_mask(n)
    live_slots = np.nonzero(live)[0]

    if len(live_slots) == 0:
        return []

    # Get updated_at for sorting
    updated_col = store.columns.get_column("__updated_at__", n)
    if updated_col is not None:
        col_data, col_pres, _ = updated_col
        # Slots without updated_at get timestamp 0 (oldest, evict first)
        timestamps = np.where(col_pres[:n], col_data[:n].astype(np.float64), 0.0)
    else:
        timestamps = np.zeros(n, dtype=np.float64)

    # Filter protected kinds before sorting
    candidates = []
    for s in live_slots:
        kind_id = int(store.node_kinds[s])
        if kind_id >= 0:
            try:
                kind_name = store.string_table.lookup(kind_id)
                if kind_name in protected_kinds:
                    continue
            except KeyError:
                pass
        candidates.append((int(s), timestamps[s]))

    # Sort live slots by timestamp ascending (oldest first)
    candidates.sort(key=lambda x: x[1])
    return [s[0] for s in candidates]


def _evict_nodes(store: CoreStore, slots_to_evict: list[int], vector_store=None, document_store=None) -> int:
    """Helper: actually evict the given slots."""
    if not slots_to_evict:
        return 0

    evicted = 0
    for slot in slots_to_evict:
        str_id = int(store.node_ids[slot])
        if str_id < 0:
            continue
        try:
            _ = store.string_table.lookup(str_id)
        except KeyError:
            continue

        if vector_store is not None:
            vector_store.remove(slot)
        if document_store is not None:
            try:
                document_store.delete_document(slot)
            except Exception:
                pass

        # Tombstone
        store.columns.clear(slot)
        store.node_tombstones.add(slot)
        if str_id in store.id_to_slot:
            del store.id_to_slot[str_id]
        store._count -= 1
        evicted += 1

    # Cascade edge cleanup for all evicted slots
    if evicted > 0:
        store._edge_keys = {
            (s, t, k)
            for k, edges in store._edges_by_type.items()
            for s, t, _d in edges
            if s not in store.node_tombstones and t not in store.node_tombstones
        }
        for etype in list(store._edges_by_type.keys()):
            store._edges_by_type[etype] = [
                (s, t, d) for s, t, d in store._edges_by_type[etype]
                if s not in store.node_tombstones and t not in store.node_tombstones
            ]
            if not store._edges_by_type[etype]:
                del store._edges_by_type[etype]
        store._edges_dirty = True
        store._ensure_edges_built()

    return evicted


def evict_oldest(store: CoreStore, target_bytes: int, vector_store=None, document_store=None, protected_kinds: set | None = None) -> dict:
    """Evict oldest non-protected nodes until memory usage is at or below target_bytes."""
    from graphstore.core.memory import measure
    
    PROTECTED_KINDS = protected_kinds if protected_kinds is not None else {"schema", "config", "system"}
    
    before = measure(store, vector_store)
    if before["total"] <= target_bytes:
        return {"evicted": 0, "bytes_before": before["total"], "bytes_after": before["total"]}

    candidates = _get_evictable_slots_sorted(store, PROTECTED_KINDS)
    evicted = 0
    
    to_evict = []
    for slot in candidates:
        to_evict.append(slot)
        # Check if we're under target periodically to avoid measuring every node
        if len(to_evict) % 100 == 0:
            _evict_nodes(store, to_evict, vector_store, document_store)
            evicted += len(to_evict)
            to_evict.clear()
            
            current = measure(store, vector_store)
            if current["total"] <= target_bytes:
                break
                
    if to_evict:
        _evict_nodes(store, to_evict, vector_store, document_store)
        evicted += len(to_evict)

    after = measure(store, vector_store)
    return {"evicted": evicted, "bytes_before": before["total"], "bytes_after": after["total"]}


def evict_by_count(store: CoreStore, limit: int, vector_store=None, document_store=None, protected_kinds: set | None = None) -> dict:
    """Evict exactly limit number of oldest non-protected nodes."""
    from graphstore.core.memory import measure
    
    if limit <= 0:
        before = measure(store, vector_store)
        return {"evicted": 0, "bytes_before": before["total"], "bytes_after": before["total"]}
        
    PROTECTED_KINDS = protected_kinds if protected_kinds is not None else {"schema", "config", "system"}
    
    before = measure(store, vector_store)
    
    candidates = _get_evictable_slots_sorted(store, PROTECTED_KINDS)
    to_evict = candidates[:limit]
    
    evicted = _evict_nodes(store, to_evict, vector_store, document_store)
    
    after = measure(store, vector_store)
    return {"evicted": evicted, "bytes_before": before["total"], "bytes_after": after["total"]}
