"""Adapter registry.

Each adapter is imported optionally so a missing dependency on one system
does not block the others. If your runtime has mem0ai and letta installed,
all three adapters become available. If not, only graphstore is listed.
"""

from .graphstore_ import GraphStoreAdapter

AVAILABLE: dict[str, type] = {
    "graphstore": GraphStoreAdapter,
}

try:  # pragma: no cover
    from .mem0 import Mem0Adapter

    AVAILABLE["mem0"] = Mem0Adapter
except ImportError:
    pass

try:  # pragma: no cover
    from .letta import LettaAdapter

    AVAILABLE["letta"] = LettaAdapter
except ImportError:
    pass

try:  # pragma: no cover
    from .chroma_bm25 import ChromaBM25Adapter

    AVAILABLE["chroma-bm25"] = ChromaBM25Adapter
except ImportError:
    pass

try:  # pragma: no cover
    from .llamaindex_ import LlamaIndexAdapter

    AVAILABLE["llamaindex"] = LlamaIndexAdapter
except ImportError:
    pass


def get_adapter(name: str) -> type:
    if name not in AVAILABLE:
        raise ValueError(
            f"Unknown adapter: {name!r}. Available: {sorted(AVAILABLE.keys())}"
        )
    return AVAILABLE[name]
