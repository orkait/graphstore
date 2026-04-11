"""Pure eviction + pressure-metric helpers."""

import numpy as np

__all__ = ["needs_optimization", "rank_evictable_slots"]


def needs_optimization(
    health: dict,
    compact_threshold: float = 0.2,
    string_gc_threshold: float = 3.0,
    cache_gc_threshold: int = 200,
) -> list[str]:
    """Decide which optimizer ops are required from a health snapshot."""
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


def rank_evictable_slots(
    live_mask: np.ndarray,
    kind_ids: np.ndarray,
    kind_lookup,
    updated_at: np.ndarray | None,
    updated_at_present: np.ndarray | None,
    protected_kinds: set[str],
) -> list[int]:
    # Identify live slots
    live_slots = np.flatnonzero(live_mask)
    if live_slots.size == 0:
        return []

    # Compute timestamps for all slots
    timestamps = np.zeros(live_mask.shape[0], dtype=np.float64)
    if updated_at is not None and updated_at_present is not None:
        timestamps = np.where(updated_at_present, updated_at.astype(np.float64), 0.0)

    # Extract kind IDs for live slots
    kind_ids_live = kind_ids[live_slots]

    # Unique kind IDs present among live slots and mapping back to each slot
    unique_kind_ids, first_idx = np.unique(kind_ids_live, return_inverse=True)

    # Helper to test protection per kind ID (handles negative IDs and missing keys)
    def _is_protected(k):
        if k < 0:
            return False
        try:
            return kind_lookup(k) in protected_kinds
        except KeyError:
            return False

    # Vectorized evaluation of protection for each unique kind ID
    protected_flags = np.frompyfunc(_is_protected, 1, 1)(unique_kind_ids)
    protected_flags = protected_flags.astype(bool)

    # Map protection back to live slots
    protected_mask = protected_flags[first_idx]

    # Filter out protected slots
    evictable_mask = ~protected_mask
    evictable_slots = live_slots[evictable_mask]

    if evictable_slots.size == 0:
        return []

    # Sort by timestamp (oldest first); stable sort preserves original order for ties
    evictable_timestamps = timestamps[evictable_slots]
    sorted_indices = np.argsort(evictable_timestamps, kind="stable")
    sorted_slots = evictable_slots[sorted_indices]

    return sorted_slots.tolist()