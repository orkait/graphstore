"""Score fusion primitives for hybrid retrieval."""

import numpy as np

__all__ = [
    "rrf_fuse",
    "normalize_bm25",
    "recency_decay",
    "weighted_remember_fusion",
]


def rrf_fuse(
    rank_groups: list[dict[str, int]],
    k_rrf: float = 60.0,
) -> dict[str, float]:
    """Reciprocal Rank Fusion over N ranked lists keyed by id.

    Each entry in rank_groups is {item_id: rank_from_0}. Returns fused
    {item_id: score} summing 1 / (k_rrf + rank) across groups.
    """
    fused: dict[str, float] = {}
    for group in rank_groups:
        for item_id, rank in group.items():
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k_rrf + rank)
    return fused


def normalize_bm25(scores: np.ndarray) -> np.ndarray:
    """Scale BM25 scores by their max. Returns all-zero if max == 0."""
    if scores.size == 0:
        return scores
    m = float(scores.max()) if scores.size else 0.0
    if m <= 0.0:
        return np.zeros_like(scores, dtype=np.float64)
    return scores.astype(np.float64) / m


def recency_decay(
    updated_at_ms: np.ndarray,
    present: np.ndarray,
    now_ms: int,
    half_life_days: float = 30.0,
) -> np.ndarray:
    """Exponential decay score from timestamp column.

    Missing timestamps return 1.0 (treat as most recent).
    """
    if updated_at_ms.size == 0:
        return np.ones(0, dtype=np.float64)
    age_ms = (now_ms - updated_at_ms.astype(np.float64))
    age_days = np.maximum(age_ms / 86400000.0, 0.0)
    decay = np.exp(-age_days / half_life_days)
    return np.where(present, decay, 1.0)


def weighted_remember_fusion(
    vec_signal: np.ndarray,
    bm25_signal: np.ndarray,
    recency_signal: np.ndarray,
    confidence_signal: np.ndarray,
    recall_signal: np.ndarray,
    weights: list[float],
) -> np.ndarray:
    """5-signal weighted sum over aligned candidate arrays.

    Returns a float64 array of fused scores indexed the same as the inputs.
    Missing weights fall back to defaults [0.30, 0.20, 0.15, 0.20, 0.15].
    """
    defaults = (0.30, 0.20, 0.15, 0.20, 0.15)
    w = [
        weights[i] if i < len(weights) else defaults[i]
        for i in range(5)
    ]
    return (
        w[0] * vec_signal
        + w[1] * bm25_signal
        + w[2] * recency_signal
        + w[3] * confidence_signal
        + w[4] * recall_signal
    )
