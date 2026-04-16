"""Adapter registry.

Only the graphstore adapter is registered. The adapter protocol
(adapter.py) remains generic so external adapters can still be
plugged in programmatically.
"""

from .graphstore_ import GraphStoreAdapter

AVAILABLE: dict[str, type] = {
    "graphstore": GraphStoreAdapter,
}


def get_adapter(name: str) -> type:
    if name not in AVAILABLE:
        raise ValueError(
            f"Unknown adapter: {name!r}. Available: {sorted(AVAILABLE.keys())}"
        )
    return AVAILABLE[name]
