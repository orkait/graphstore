"""Spreading activation over a directed CSR graph."""

import numpy as np
from scipy.sparse import csr_matrix

__all__ = ["spreading_activation"]


def spreading_activation(
    matrix_t: csr_matrix,
    cue_slot: int,
    depth: int,
    decay: float,
    live_mask: np.ndarray,
    importance: np.ndarray | None = None,
    recency: np.ndarray | None = None,
) -> np.ndarray:
    """Iterative spreading activation over pre-transposed adjacency.

    The transposed matrix is passed in so callers can cache it. After
    `depth` hops the activation is multiplied by the importance and
    recency modulations (if provided). The cue slot's activation is
    zeroed before return so callers don't rank it.

    Args:
        matrix_t: CSR transpose of the adjacency matrix
        cue_slot: starting slot index
        depth: number of spreading hops
        decay: per-hop decay factor
        live_mask: bool mask zeroing activation on tombstoned/retracted slots
        importance: optional per-slot scalar multiplier
        recency: optional per-slot scalar multiplier

    Returns a float64 activation array of the same length as live_mask.
    """
    n = len(live_mask)
    activation = np.zeros(n, dtype=np.float64)
    if cue_slot < 0 or cue_slot >= n:
        return activation
    activation[cue_slot] = 1.0

    live_f = live_mask.astype(np.float64)
    for _ in range(depth):
        spread = matrix_t.dot(activation) * decay
        activation = activation + spread
        activation[:n] *= live_f

    if importance is not None:
        activation *= importance[:len(activation)]
    if recency is not None:
        activation *= recency[:len(activation)]

    activation[cue_slot] = 0.0
    return activation
