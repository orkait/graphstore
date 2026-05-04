"""MCP server package for graphstore.

Exposes graphstore as agent-callable tools over the Model Context Protocol.
The console script `graphstore-mcp` (installed by the [mcp] extra) launches
the stdio server. See `graphstore.mcp.server` for the implementation.
"""
from graphstore.mcp.server import main

__all__ = ["main"]
