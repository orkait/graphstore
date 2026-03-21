"""Auto-wire cross-document relationships via vector similarity."""
import numpy as np
from graphstore.core.types import Result


def connect_all(store, vector_store, threshold=0.85, where_expr=None, executor=None):
    """Find and wire similar chunks across documents."""
    if vector_store is None or vector_store.count() == 0:
        return Result(kind="ok", data={"edges_created": 0}, count=0)

    n = store._next_slot
    live = store.compute_live_mask(n)

    edges_created = 0
    checked = set()

    for slot in range(n):
        if not live[slot] or not vector_store.has_vector(slot):
            continue

        vec = vector_store.get_vector(slot)
        # Find top-10 similar (oversample to filter self + same-doc)
        results_slots, dists = vector_store.search(vec, k=10, mask=live)

        for other_slot, dist in zip(results_slots, dists):
            other_slot = int(other_slot)
            if other_slot == slot:
                continue

            similarity = 1.0 - float(dist)
            if similarity < threshold:
                continue

            pair = (min(slot, other_slot), max(slot, other_slot))
            if pair in checked:
                continue
            checked.add(pair)

            # Check if edge already exists
            src_id = store._slot_to_id(slot)
            tgt_id = store._slot_to_id(other_slot)
            if src_id and tgt_id:
                edge_key = (slot, other_slot, "similar_to")
                if edge_key not in store._edge_keys:
                    try:
                        store.put_edge(src_id, tgt_id, "similar_to", {"similarity": round(similarity, 4)})
                        edges_created += 1
                    except Exception:
                        pass

    return Result(kind="ok", data={"edges_created": edges_created}, count=edges_created)


def connect_node(store, vector_store, node_id, threshold=0.8):
    """Wire one node to its nearest similar neighbors."""
    if vector_store is None:
        return Result(kind="ok", data={"edges_created": 0}, count=0)

    str_id = store.string_table.intern(node_id) if node_id in store.string_table else None
    slot = store.id_to_slot.get(str_id) if str_id is not None else None
    if slot is None:
        from graphstore.core.errors import NodeNotFound
        raise NodeNotFound(node_id)

    if not vector_store.has_vector(slot):
        from graphstore.core.errors import VectorError
        raise VectorError(f"Node '{node_id}' has no vector")

    vec = vector_store.get_vector(slot)
    n = store._next_slot
    live = store.compute_live_mask(n)

    results_slots, dists = vector_store.search(vec, k=10, mask=live)

    edges_created = 0
    for other_slot, dist in zip(results_slots, dists):
        other_slot = int(other_slot)
        if other_slot == slot:
            continue
        similarity = 1.0 - float(dist)
        if similarity < threshold:
            continue
        tgt_id = store._slot_to_id(other_slot)
        if tgt_id:
            edge_key = (slot, other_slot, "similar_to")
            if edge_key not in store._edge_keys:
                try:
                    store.put_edge(node_id, tgt_id, "similar_to", {"similarity": round(similarity, 4)})
                    edges_created += 1
                except Exception:
                    pass

    return Result(kind="ok", data={"edges_created": edges_created}, count=edges_created)
