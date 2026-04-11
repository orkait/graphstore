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
    """Return live slot indices sorted oldest-first, skipping protected kinds.

    Args:
        live_mask: bool array, True for live slots
        kind_ids: int array of interned kind IDs per slot
        kind_lookup: callable(int) -> str, resolves kind_id to kind name
        updated_at: int array of __updated_at__ ms timestamps (or None)
        updated_at_present: bool presence mask for updated_at (or None)
        protected_kinds: kind names that must not be evicted
    """
    live_slots = np.nonzero(live_mask)[0]
    if live_slots.size == 0:
        return []

    if updated_at is not None and updated_at_present is not None:
        timestamps = np.where(
            updated_at_present, updated_at.astype(np.float64), 0.0
        )
    else:
        timestamps = np.zeros(len(live_mask), dtype=np.float64)

    candidates: list[tuple[int, float]] = []
    for s in live_slots:
        s_int = int(s)
        kind_id = int(kind_ids[s_int])
        if kind_id >= 0:
            try:
                if kind_lookup(kind_id) in protected_kinds:
                    continue
            except KeyError:
                pass
        candidates.append((s_int, float(timestamps[s_int])))

    candidates.sort(key=lambda x: x[1])
    return [c[0] for c in candidates]
