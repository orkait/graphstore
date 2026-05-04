"""MCP server exposing graphstore as agent-callable tools.

Default mode: in-process GraphStore() (fast, no HTTP, no playground server).
Remote mode: set GRAPHSTORE_URL=http://host:7200 to forward calls to a
running `graphstore playground` instance.

Launch via the console script (installed by the [mcp] extra):

    graphstore-mcp

or the equivalent module form:

    python -m graphstore.mcp.server

Environment:
    GRAPHSTORE_DB_PATH    path for in-process db (default ./graphstore-mcp.db)
    GRAPHSTORE_PROFILE    "pro" to enable Pro mode (Bonsai + Jina + NER)
    GRAPHSTORE_URL        optional remote playground URL; if set, in-process
                          store is bypassed and DSL is forwarded over HTTP
    GRAPHSTORE_AUTH_TOKEN optional bearer token for the remote URL
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    sys.stderr.write(
        "missing dep 'mcp'. install with: uv pip install 'graphstore[mcp]'\n"
    )
    raise

logger = logging.getLogger("graphstore.mcp")

mcp = FastMCP("graphstore")

_REMOTE_URL = os.environ.get("GRAPHSTORE_URL")
_AUTH_TOKEN = os.environ.get("GRAPHSTORE_AUTH_TOKEN")

_store = None


def _get_store():
    """Lazy in-process GraphStore singleton."""
    global _store
    if _store is None:
        from graphstore import GraphStore

        kwargs: dict[str, Any] = {
            "path": os.environ.get("GRAPHSTORE_DB_PATH", "./graphstore-mcp.db"),
        }
        profile = os.environ.get("GRAPHSTORE_PROFILE")
        if profile:
            kwargs["profile"] = profile
        _store = GraphStore(**kwargs)
    return _store


def _execute_remote(dsl: str) -> dict:
    """Forward DSL to a running playground HTTP API."""
    import urllib.request

    headers = {"Content-Type": "application/json"}
    if _AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {_AUTH_TOKEN}"
    req = urllib.request.Request(
        f"{_REMOTE_URL.rstrip('/')}/api/execute",
        data=json.dumps({"query": dsl}).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _execute(dsl: str) -> dict:
    """Run DSL via in-process or remote, returning a structured dict.

    Errors are caught and returned as {"error": "...", "dsl": dsl} so the
    MCP client sees a clean response instead of a transport-level crash.
    """
    try:
        if _REMOTE_URL:
            return _execute_remote(dsl)
        r = _get_store().execute(dsl)
        return {
            "kind": r.kind,
            "count": r.count,
            "data": r.data,
            "elapsed_us": r.elapsed_us,
            "meta": getattr(r, "meta", None),
        }
    except Exception as e:
        logger.exception("graphstore execute failed")
        return {
            "error": f"{type(e).__name__}: {e}",
            "dsl": dsl,
        }


def _esc(s: str) -> str:
    """Escape a string for embedding in a DSL double-quoted literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


@mcp.tool()
def gs_remember(text: str) -> dict:
    """Store a fact in graphstore. Use for any agent observation, decision,
    or piece of context worth recalling later.

    Internally emits CREATE NODE AUTO content="..." DOCUMENT "..." which
    persists the text into the column store, splits sentences, indexes
    BM25, and computes the embedding so subsequent gs_lexical / gs_similar
    / gs_recall calls can find it. (The graphstore DSL verb REMEMBER is a
    fusion *search*, not a write; this MCP tool name reflects agent
    semantics, not the DSL verb.)

    Example: gs_remember("user prefers dark mode in the IDE")
    """
    esc = _esc(text)
    return _execute(f'CREATE NODE AUTO content="{esc}" DOCUMENT "{esc}"')


@mcp.tool()
def gs_remember_batch(texts: list[str]) -> dict:
    """Store many facts in one call. Saves N round-trips when ingesting
    a list of observations at once. Each text becomes one node via
    CREATE NODE AUTO. Returns per-fact results plus success/error count.
    """
    results = []
    ok = 0
    err = 0
    for t in texts:
        esc = _esc(t)
        r = _execute(f'CREATE NODE AUTO content="{esc}" DOCUMENT "{esc}"')
        results.append({"text": t, "result": r})
        if "error" in r:
            err += 1
        else:
            ok += 1
    return {"ok": ok, "errors": err, "results": results}


@mcp.tool()
def gs_recall(node_id: str, depth: int = 2, limit: int = 10) -> dict:
    """Pull a node + its DEPTH-hop neighbours from the graph by node id.

    Use when you already know the id (returned by gs_lexical/gs_similar).
    """
    return _execute(
        f'RECALL FROM "{_esc(node_id)}" DEPTH {int(depth)} LIMIT {int(limit)}'
    )


@mcp.tool()
def gs_search(query: str, limit: int = 10) -> dict:
    """Best general-purpose retrieval. Emits the DSL `REMEMBER "..." LIMIT N`
    which runs graphstore's 3-signal fusion: vector similarity + BM25 +
    recency + graph proximity, with optional reranker. Use this as the
    default search tool unless you specifically need pure lexical or pure
    vector ranking.
    """
    return _execute(f'REMEMBER "{_esc(query)}" LIMIT {int(limit)}')


@mcp.tool()
def gs_lexical(query: str, limit: int = 10) -> dict:
    """Pure full-text (BM25) search across stored content. Returns
    matching nodes by keyword overlap only - no embeddings.

    Use when the user gave exact keywords or for fast retrieval without
    embedding cost.
    """
    return _execute(f'LEXICAL SEARCH "{_esc(query)}" LIMIT {int(limit)}')


@mcp.tool()
def gs_similar(text: str, limit: int = 5) -> dict:
    """Vector similarity search. Returns nodes whose embeddings are
    nearest to the embedding of the given text.

    Use for fuzzy/semantic queries ("anything about X?").
    """
    return _execute(f'SIMILAR TO "{_esc(text)}" LIMIT {int(limit)}')


@mcp.tool()
def gs_traverse(from_node_id: str, depth: int = 2, limit: int = 50) -> dict:
    """Walk the graph outward from a node id, returning subgraph rows."""
    return _execute(
        f'TRAVERSE FROM "{_esc(from_node_id)}" DEPTH {int(depth)} LIMIT {int(limit)}'
    )


@mcp.tool()
def gs_answer(query: str, max_tokens: int = 200) -> dict:
    """Ask a natural-language question. ANSWER retrieves relevant nodes,
    then runs a local LLM (Bonsai TQ1_0 in Pro mode) to synthesise a reply.

    Pro mode (GRAPHSTORE_PROFILE=pro) gives best results.
    """
    return _execute(f'ANSWER "{_esc(query)}" TOKENS {int(max_tokens)}')


@mcp.tool()
def gs_count_nodes() -> dict:
    """Return the number of nodes currently stored. Edges and other
    objects are NOT counted.
    """
    return _execute("COUNT NODES")


@mcp.tool()
def gs_execute(dsl: str) -> dict:
    """Escape hatch: execute raw DSL. Reach for this only when the typed
    tools above don't cover the verb you need. See the graphstore-dsl
    skill for the full grammar (~70 verbs).
    """
    return _execute(dsl)


@mcp.resource("graph://stats")
def stats_resource() -> str:
    """Live node count + connection mode."""
    n = _execute("COUNT NODES")
    return json.dumps(
        {
            "nodes": n.get("count", 0) if "error" not in n else None,
            "error": n.get("error"),
            "mode": "remote" if _REMOTE_URL else "in-process",
            "remote_url": _REMOTE_URL,
            "db_path": os.environ.get("GRAPHSTORE_DB_PATH", "./graphstore-mcp.db"),
            "profile": os.environ.get("GRAPHSTORE_PROFILE"),
        }
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
