# graphstore MCP server

Exposes graphstore as agent-callable tools over the Model Context Protocol
(MCP). Use from Claude Desktop, Cursor, or any MCP-aware client to give an
agent a persistent memory + retrieval substrate without writing DSL by hand.

## Why this exists

graphstore already speaks DSL and has a Python API. An agent that wants to
"remember" or "recall" things has three options:

1. Call `gs.execute("REMEMBER \"...\"")` directly in a Python tool.
2. POST to `graphstore playground`'s `/api/execute` over HTTP.
3. Speak MCP and let its host (Claude Desktop, Cursor, etc.) wire the tools
   in for you.

Option 3 needs no playground server, no HTTP, no manual DSL escaping. The
MCP server holds an in-process `GraphStore()` and translates typed tool
calls into DSL.

## Install

```bash
uv pip install 'graphstore[mcp]'
```

This installs the `mcp` Python SDK and registers a `graphstore-mcp`
console script. The server lives inside the wheel at
`graphstore.mcp.server:main` - no need to clone the repo.

## Run

```bash
graphstore-mcp                       # console script (preferred)
python -m graphstore.mcp.server      # equivalent module form
```

Both speak stdio - the transport Claude Desktop / Cursor expect.

## Modes

| Mode | When to use | How |
|---|---|---|
| **In-process** (default) | Single agent, local laptop, no other consumers of the db | nothing - just run the server |
| **Remote** | Multiple agents share one graphstore over HTTP | set `GRAPHSTORE_URL=http://host:7200` and start `graphstore playground` separately |

The in-process default means **the playground HTTP server is NOT required**
for MCP. Keep it disabled unless you want the web UI or multi-agent shared
state.

## Claude Desktop config

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) or `%APPDATA%/Claude/claude_desktop_config.json` (Windows). See
`claude_desktop_config.example.json` in this dir for a copy-paste version.

```json
{
  "mcpServers": {
    "graphstore": {
      "command": "graphstore-mcp",
      "env": {
        "GRAPHSTORE_DB_PATH": "/path/to/graphstore-agent.db"
      }
    }
  }
}
```

For Pro mode (Bonsai LLM + Jina + NER), add `"GRAPHSTORE_PROFILE": "pro"`.

## Tools exposed

| Tool | DSL emitted | Use for |
|---|---|---|
| `gs_remember(text)` | `CREATE NODE AUTO content="..." DOCUMENT "..."` | store one observation (writes + indexes BM25 + embeds) |
| `gs_remember_batch(texts)` | N x `CREATE NODE AUTO ...` | bulk ingest in one round-trip |
| `gs_search(query, limit)` | `REMEMBER "..." LIMIT K` | best general retrieval - 3-signal fusion (vector + BM25 + recency + graph) |
| `gs_recall(node_id, depth, limit)` | `RECALL FROM "..." DEPTH N LIMIT K` | pull a node + neighbours by id |
| `gs_lexical(query, limit)` | `LEXICAL SEARCH "..." LIMIT K` | pure BM25 keyword search |
| `gs_similar(text, limit)` | `SIMILAR TO "..." LIMIT K` | pure vector / fuzzy search |
| `gs_traverse(from_id, depth, limit)` | `TRAVERSE FROM "..." DEPTH N LIMIT K` | walk the graph |
| `gs_answer(query, max_tokens)` | `ANSWER "..." TOKENS K` | RAG over stored content (Pro mode for best results) |
| `gs_count_nodes()` | `COUNT NODES` | check store size |
| `gs_execute(dsl)` | raw DSL | escape hatch for verbs not covered above |

**Note on naming**: the MCP tool `gs_remember` writes; the DSL verb
`REMEMBER` searches (3-signal fusion). The MCP name reflects agent
semantics ("remember this fact" = store it); the DSL verb name reflects
recall semantics ("remember about X" = pull related nodes). The
`gs_search` tool is what wraps the DSL `REMEMBER` verb.

Resource `graph://stats` returns live node count + mode.

All tools catch DSL errors and return `{"error": "..."}` instead of
crashing the transport - the agent can recover and retry.

## Environment

| Var | Default | Meaning |
|---|---|---|
| `GRAPHSTORE_DB_PATH` | `./graphstore-mcp.db` | in-process db location |
| `GRAPHSTORE_PROFILE` | unset | set to `pro` for Bonsai-backed `gs_answer` |
| `GRAPHSTORE_URL` | unset | when set, forwards DSL to a running playground at that URL |
| `GRAPHSTORE_AUTH_TOKEN` | unset | bearer token for the remote URL |

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `missing dep 'mcp'` on stderr | install ran without `[mcp]` extra | `uv pip install 'graphstore[mcp]'` |
| Tool calls hang on first use | embedder model downloading from HF (~30 MB) | wait for first call to finish; subsequent calls are fast |
| `gs_count_nodes` always returns 0 after `gs_remember` | WAL not yet flushed to nodes table | run `gs_execute("SYS COMMIT")` or wait for periodic flush |
| `gs_answer` returns garbage / errors | not in Pro mode, no Bonsai loaded | set `GRAPHSTORE_PROFILE=pro` and ensure Bonsai gguf is reachable |
| Remote mode 401 | playground requires auth | set `GRAPHSTORE_AUTH_TOKEN` to the value from playground startup logs |

## Related skills

- `tools/skills/graphstore-agent/SKILL.md` - when/why to call these tools
- `tools/skills/graphstore-dsl/SKILL.md` - full DSL grammar (use with `gs_execute`)
- `tools/skills/graphstore-bonsai-dsl/SKILL.md` - LLM-friendly DSL subset
