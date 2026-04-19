"""Metacognitive evolution layer.

Moved from ``graphstore.evolve`` into ``graphstore.core.evolve`` so the
self-tuning runtime lives next to the subsystems it tunes. Public surface
unchanged: every previously-importable symbol is re-exported here and
through ``graphstore.evolve`` for backwards compat.
"""

from graphstore.core.evolve._impl import (
    Action,
    Condition,
    EvolutionEngine,
    EvolutionRule,
    KNOWN_SIGNALS,
    TUNABLE_PARAMS,
)
from graphstore.core.evolve._defaults import STARTER_RULES

__all__ = [
    "Action",
    "Condition",
    "EvolutionEngine",
    "EvolutionRule",
    "KNOWN_SIGNALS",
    "TUNABLE_PARAMS",
    "STARTER_RULES",
]
