"""Group-by aggregation primitives on numpy arrays.

Pure numpy ops for SQL-like GROUP BY: unique-based bucket assignment,
vectorized COUNT / SUM / AVG / MIN / MAX, and per-group distinct count.
No graphstore imports. The caller supplies column arrays already filtered
by its WHERE mask.
"""

import numpy as np

__all__ = [
    "group_assign_single",
    "group_assign_multi",
    "group_count",
    "group_sum",
    "group_avg",
    "group_min",
    "group_max",
    "group_count_distinct",
]


def group_assign_single(keys: np.ndarray) -> tuple:
    """Bucket rows by a single-column key.

    Returns (unique_keys, inverse) where inverse[i] is the group index
    of row i. Same shape semantics as np.unique(return_inverse=True).
    """
    return np.unique(keys, return_inverse=True)


def group_assign_multi(key_cols: list) -> tuple:
    """Bucket rows by a list of column arrays (multi-key GROUP BY).

    Returns (unique_rows, inverse). unique_rows has shape (num_groups, k).
    """
    stacked = np.column_stack(key_cols)
    return np.unique(stacked, axis=0, return_inverse=True)


def group_count(inverse: np.ndarray, num_groups: int) -> np.ndarray:
    """Count rows per group via np.bincount."""
    return np.bincount(inverse, minlength=num_groups).astype(np.float64)


def group_sum(
    values: np.ndarray,
    inverse: np.ndarray,
    num_groups: int,
) -> np.ndarray:
    """Sum values per group via np.add.at."""
    sums = np.zeros(num_groups, dtype=np.float64)
    np.add.at(sums, inverse, values.astype(np.float64))
    return sums


def group_avg(
    values: np.ndarray,
    inverse: np.ndarray,
    num_groups: int,
) -> np.ndarray:
    """Mean per group. Empty groups return 0 via max(count, 1) floor."""
    sums = group_sum(values, inverse, num_groups)
    counts = group_count(inverse, num_groups)
    return sums / np.maximum(counts, 1)


def group_min(
    values: np.ndarray,
    inverse: np.ndarray,
    num_groups: int,
) -> np.ndarray:
    """Per-group minimum via np.minimum.at. Empty groups return +inf."""
    mins = np.full(num_groups, np.inf, dtype=np.float64)
    np.minimum.at(mins, inverse, values.astype(np.float64))
    return mins


def group_max(
    values: np.ndarray,
    inverse: np.ndarray,
    num_groups: int,
) -> np.ndarray:
    """Per-group maximum via np.maximum.at. Empty groups return -inf."""
    maxs = np.full(num_groups, -np.inf, dtype=np.float64)
    np.maximum.at(maxs, inverse, values.astype(np.float64))
    return maxs


def group_count_distinct(
    values: np.ndarray,
    inverse: np.ndarray,
    num_groups: int,
) -> np.ndarray:
    """Per-group distinct count.

    Unlike COUNT/SUM/AVG/MIN/MAX this does NOT have a single-shot numpy
    primitive — we walk each group with a mask, unique, and len. For
    small group counts this is fine; for very wide GROUP BYs consider a
    sort-then-adjacent-diff approach.
    """
    counts = np.zeros(num_groups, dtype=np.float64)
    for g in range(num_groups):
        counts[g] = len(np.unique(values[inverse == g]))
    return counts
