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


def needs_optimization(health: dict) -> list[str]:
    """Determine which operations are needed based on health metrics."""
    ops = []
    if health["tombstone_ratio"] > 0.2:
        ops.append("COMPACT")
    if health["string_bloat"] > 3.0 and health["live_nodes"] > 0:
        ops.append("STRINGS")
    if health["dead_vectors"] > 0:
        ops.append("VECTORS")
    if health["stale_edge_keys"] > 0:
        ops.append("EDGES")
    if health["cache_size"] > 200:
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

    # Remap DocumentStore slot keys
    if document_store is not None:
        conn = document_store._conn
        for old_slot, new_slot in old_to_new_dict.items():
            if old_slot == new_slot:
                continue
            conn.execute("UPDATE documents SET slot = ? WHERE slot = ?", (-old_slot - 1, old_slot))
            conn.execute("UPDATE summaries SET slot = ? WHERE slot = ?", (-old_slot - 1, old_slot))
            conn.execute("UPDATE summaries SET doc_slot = ? WHERE doc_slot = ?", (-old_slot - 1, old_slot))
            conn.execute("UPDATE images SET slot = ? WHERE slot = ?", (-old_slot - 1, old_slot))
            conn.execute("UPDATE doc_metadata SET doc_slot = ? WHERE doc_slot = ?", (-old_slot - 1, old_slot))
            conn.execute("DELETE FROM doc_fts WHERE rowid = ?", (old_slot,))
        for old_slot, new_slot in old_to_new_dict.items():
            if old_slot == new_slot:
                continue
            temp = -old_slot - 1
            conn.execute("UPDATE documents SET slot = ? WHERE slot = ?", (new_slot, temp))
            conn.execute("UPDATE summaries SET slot = ? WHERE slot = ?", (new_slot, temp))
            conn.execute("UPDATE summaries SET doc_slot = ? WHERE doc_slot = ?", (new_slot, temp))
            conn.execute("UPDATE images SET slot = ? WHERE slot = ?", (new_slot, temp))
            conn.execute("UPDATE doc_metadata SET doc_slot = ? WHERE doc_slot = ?", (new_slot, temp))
        # Rebuild FTS from summaries
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
    """Run all 6 optimization operations in safe order."""
    results = {}
    results["compact"] = compact_tombstones(store, vector_store, document_store)
    results["strings"] = gc_strings(store)
    results["edges"] = defrag_edges(store)
    results["vectors"] = cleanup_vectors(store, vector_store)
    results["blobs"] = sweep_orphans(store, document_store)
    results["cache"] = clear_caches(store)
    return results


def evict_oldest(store: CoreStore, vector_store=None, document_store=None, target_bytes: int = 0) -> dict:
    """Emergency eviction: delete oldest non-essential nodes until under target.

    Evicts nodes in order of __updated_at__ (oldest first).
    Skips nodes with kind in PROTECTED_KINDS.
    Returns {"evicted": count, "bytes_before": N, "bytes_after": N}.
    """
    from graphstore.core.memory import measure

    PROTECTED_KINDS = {"schema", "config", "system"}

    before = measure(store, vector_store)
    if target_bytes <= 0 or before["total"] <= target_bytes:
        return {"evicted": 0, "bytes_before": before["total"], "bytes_after": before["total"]}

    n = store._next_slot
    if n == 0:
        return {"evicted": 0, "bytes_before": before["total"], "bytes_after": before["total"]}

    live = store.compute_live_mask(n)
    live_slots = np.nonzero(live)[0]

    if len(live_slots) == 0:
        return {"evicted": 0, "bytes_before": before["total"], "bytes_after": before["total"]}

    # Get updated_at for sorting
    updated_col = store.columns.get_column("__updated_at__", n)
    if updated_col is not None:
        col_data, col_pres, _ = updated_col
        # Slots without updated_at get timestamp 0 (oldest, evict first)
        timestamps = np.where(col_pres[:n], col_data[:n].astype(np.float64), 0.0)
    else:
        timestamps = np.zeros(n, dtype=np.float64)

    # Sort live slots by timestamp ascending (oldest first)
    slot_times = [(int(s), timestamps[s]) for s in live_slots]
    slot_times.sort(key=lambda x: x[1])

    evicted = 0
    for slot, ts in slot_times:
        # Check if we're under target
        if evicted > 0 and evicted % 100 == 0:
            current = measure(store, vector_store)
            if current["total"] <= target_bytes:
                break

        # Skip protected kinds
        kind_id = int(store.node_kinds[slot])
        if kind_id >= 0:
            try:
                kind_name = store.string_table.lookup(kind_id)
                if kind_name in PROTECTED_KINDS:
                    continue
            except KeyError:
                pass

        # Evict: tombstone the node
        str_id = int(store.node_ids[slot])
        if str_id < 0:
            continue
        try:
            node_id = store.string_table.lookup(str_id)
        except KeyError:
            continue

        # Remove from vector store
        if vector_store is not None:
            vector_store.remove(slot)

        # Remove from document store
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

    after = measure(store, vector_store)
    return {"evicted": evicted, "bytes_before": before["total"], "bytes_after": after["total"]}
