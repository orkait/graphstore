"""Slot compaction primitives.

Pure numpy helpers that compute slot remap plans without touching stores.
The caller (core/optimizer.py) applies the plan to mutable state.
"""

import numpy as np
from scipy.sparse import csr_matrix

__all__ = [
    "build_live_mask",
    "slot_remap_plan",
    "apply_slot_remap_to_edges",
]


def build_live_mask(node_ids: np.ndarray, tombstones: set[int], n: int) -> np.ndarray:
    """Bool mask of slots that are live (valid id, not tombstoned)."""
    mask = node_ids[:n] >= 0
    if tombstones:
        tomb_arr = np.fromiter(
            (t for t in tombstones if t < n), dtype=np.int32,
        )
        if tomb_arr.size:
            mask[tomb_arr] = False
    return mask


def slot_remap_plan(live_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Build old_to_new lookup array and new live count.

    Returns:
        old_to_new: int32 array of length len(live_mask); old_to_new[i] = new
                    slot index if live, -1 if tombstoned/invalid.
        new_count: number of live slots (== nonzero entries in live_mask)
    """
    n = len(live_mask)
    live_slots_arr = np.nonzero(live_mask)[0]
    new_count = int(live_slots_arr.size)
    old_to_new = np.full(n, -1, dtype=np.int32)
    old_to_new[live_slots_arr] = np.arange(new_count, dtype=np.int32)
    return old_to_new, new_count


def apply_slot_remap_to_edges(
    edge_list: list[tuple[int, int, dict]],
    old_to_new: np.ndarray,
    n: int,
) -> list[tuple[int, int, dict]]:
    """Remap edge endpoints via old_to_new. Drops edges touching dead slots."""
    m = len(edge_list)
    if m == 0:
        return []
    srcs = np.fromiter((e[0] for e in edge_list), dtype=np.int64, count=m)
    tgts = np.fromiter((e[1] for e in edge_list), dtype=np.int64, count=m)
    in_range = (srcs < n) & (tgts < n)
    ns = np.where(in_range, old_to_new[np.clip(srcs, 0, n - 1)], -1)
    nt = np.where(in_range, old_to_new[np.clip(tgts, 0, n - 1)], -1)
    keep = (ns >= 0) & (nt >= 0)
    if not keep.any():
        return []
    keep_idx = np.nonzero(keep)[0]
    new_srcs = ns[keep_idx].tolist()
    new_tgts = nt[keep_idx].tolist()
    return [
        (new_srcs[i], new_tgts[i], edge_list[int(keep_idx[i])][2])
        for i in range(len(keep_idx))
    ]
