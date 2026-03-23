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
    """
    if not store.node_tombstones:
        return {"compacted": 0}

    n = store._next_slot
    live_slots = []
    for slot in range(n):
        if slot not in store.node_tombstones and int(store.node_ids[slot]) >= 0:
            live_slots.append(slot)

    if len(live_slots) == n:
        return {"compacted": 0}

    old_to_new: dict[int, int] = {}
    for new_slot, old_slot in enumerate(live_slots):
        old_to_new[old_slot] = new_slot

    new_count = len(live_slots)
    new_capacity = max(new_count * 2, store._capacity)

    # Remap node arrays
    new_ids = np.full(new_capacity, -1, dtype=np.int32)
    new_kinds = np.zeros(new_capacity, dtype=np.int32)
    for old_slot, new_slot in old_to_new.items():
        new_ids[new_slot] = store.node_ids[old_slot]
        new_kinds[new_slot] = store.node_kinds[old_slot]

    # Remap columns
    for field in list(store.columns._columns.keys()):
        old_col = store.columns._columns[field]
        old_pres = store.columns._presence[field]
        dtype_str = store.columns._dtypes[field]
        new_col = store.columns._make_sentinel_array(dtype_str, new_capacity)
        new_pres = np.zeros(new_capacity, dtype=bool)
        for old_slot, new_slot in old_to_new.items():
            if old_slot < len(old_col):
                new_col[new_slot] = old_col[old_slot]
                new_pres[new_slot] = old_pres[old_slot]
        store.columns._columns[field] = new_col
        store.columns._presence[field] = new_pres
    store.columns._capacity = new_capacity

    # Remap edges
    for etype in list(store._edges_by_type.keys()):
        remapped = []
        for s, t, d in store._edges_by_type[etype]:
            if s in old_to_new and t in old_to_new:
                remapped.append((old_to_new[s], old_to_new[t], d))
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
        for old_slot in live_slots:
            if vector_store.has_vector(old_slot):
                old_vectors[old_slot] = vector_store.get_vector(old_slot)

        # Clear and re-add with new slots
        vector_store._has_vector = np.zeros(new_capacity, dtype=bool)
        vector_store._capacity = new_capacity
        from usearch.index import Index
        vector_store._index = Index(ndim=vector_store._dims, metric="cos", dtype="f32")
        for old_slot, vec in old_vectors.items():
            new_slot = old_to_new[old_slot]
            vector_store._index.add(new_slot, vec)
            vector_store._has_vector[new_slot] = True

    # Remap DocumentStore slot keys
    if document_store is not None:
        conn = document_store._conn
        for old_slot, new_slot in old_to_new.items():
            if old_slot == new_slot:
                continue
            conn.execute("UPDATE documents SET slot = ? WHERE slot = ?", (-old_slot - 1, old_slot))
            conn.execute("UPDATE summaries SET slot = ? WHERE slot = ?", (-old_slot - 1, old_slot))
            conn.execute("UPDATE summaries SET doc_slot = ? WHERE doc_slot = ?", (-old_slot - 1, old_slot))
            conn.execute("UPDATE images SET slot = ? WHERE slot = ?", (-old_slot - 1, old_slot))
            conn.execute("UPDATE doc_metadata SET doc_slot = ? WHERE doc_slot = ?", (-old_slot - 1, old_slot))
            conn.execute("DELETE FROM doc_fts WHERE rowid = ?", (old_slot,))
        for old_slot, new_slot in old_to_new.items():
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
    """
    n = store._next_slot

    # Collect all referenced string IDs
    referenced: set[int] = set()
    for slot in range(n):
        if slot in store.node_tombstones:
            continue
        str_id = int(store.node_ids[slot])
        if str_id >= 0:
            referenced.add(str_id)
        kind_id = int(store.node_kinds[slot])
        if kind_id >= 0:
            referenced.add(kind_id)

    for field in store.columns._columns:
        if store.columns._dtypes[field] == "int32_interned":
            col = store.columns._columns[field]
            pres = store.columns._presence[field]
            for slot in range(n):
                if pres[slot]:
                    referenced.add(int(col[slot]))

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

    # Remap node_ids and node_kinds
    for slot in range(n):
        if slot in store.node_tombstones:
            continue
        old_id = int(store.node_ids[slot])
        if old_id >= 0:
            store.node_ids[slot] = old_to_new[old_id]
        old_kind = int(store.node_kinds[slot])
        if old_kind >= 0:
            store.node_kinds[slot] = old_to_new[old_kind]

    # Remap int32_interned columns
    for field in store.columns._columns:
        if store.columns._dtypes[field] == "int32_interned":
            col = store.columns._columns[field]
            pres = store.columns._presence[field]
            for slot in range(n):
                if pres[slot]:
                    col[slot] = old_to_new[int(col[slot])]

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
