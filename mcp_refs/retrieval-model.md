# Retrieval Model: How Agents Get Context from graphstore

## The Problem

An agent has a context window (~100K tokens). It needs to fill ~20-50K tokens with relevant context from memory. If graphstore returns the WRONG context, the agent makes bad decisions.

## Four Retrieval Modes

| Command | What it searches | Use case |
|---------|-----------------|----------|
| `SIMILAR TO "text" LIMIT 10` | Vector cosine similarity (HNSW) | Pure semantic search - "find things like this" |
| `LEXICAL SEARCH "query" LIMIT 10` | BM25 full-text on chunk content | Keyword/phrase search - "find exact terms" |
| `RECALL FROM "node" DEPTH N` | Spreading activation on graph edges | Associative recall - "what's connected to this?" |
| `REMEMBER "query" TOKENS 4000` | 5-signal fusion with token budget | Agent context retrieval - "give me the best context" |

## REMEMBER: The Primary Retrieval Command

REMEMBER is what agents should use for context retrieval. It fuses all available signals:

```
score = 0.30 * vector_sim       -- cosine similarity from embedder
      + 0.20 * bm25_norm        -- BM25 full-text, normalized to 0-1
      + 0.15 * recency          -- exp(-age_days / 30)
      + 0.20 * confidence       -- from __confidence__ column
      + 0.15 * recall_frequency -- how often this was retrieved before
```

Weights are configurable via `dsl.remember_weights` in graphstore.json.

### Token Budget

`REMEMBER "query" TOKENS 4000` fills up to 4000 tokens (estimated as `len(text) // 4`), stopping when the budget is exhausted. This replaces `LIMIT N` which returns N nodes of unknown total size.

### Retrieval Feedback Loop

Every REMEMBER call auto-increments `__recall_count__` and sets `__last_recalled_at__` on returned nodes. Frequently retrieved memories score higher in future queries - the more useful a memory proves to be, the easier it becomes to find.

## RECALL: Cognitive Association

RECALL uses accumulative spreading activation on the graph:

```python
for each hop in depth:
    spread = adjacency_matrix_transpose @ activation * decay
    activation = activation + spread    # accumulate, don't overwrite
activation *= importance * confidence * recency
```

With `decay=0.7` (configurable via `dsl.recall_decay`), closer nodes retain higher activation:
- Depth 1 neighbors: ~0.7 activation
- Depth 2 neighbors: ~0.49 activation
- Depth 3 neighbors: ~0.34 activation

## What Gets Embedded

Documents are embedded using **full chunk text with heading context**, not truncated summaries:

```python
embed_text = f"{heading}: {chunk.text}" if heading else chunk.text
```

Model2Vec (default embedder) has no length limit - 50k chars in <1ms. The full chunk content is what vector search operates on, so semantic search actually finds content by meaning.

FTS5 also indexes the full chunk text, so LEXICAL SEARCH finds keywords anywhere in the chunk.

## Engine Synchronization

All mutations propagate to all relevant search indices:
- UPDATE NODE (embed field) -> auto re-embed vector
- UPDATE NODES (bulk, embed field) -> auto re-embed all
- DELETE / RETRACT -> immediately remove vector (no ghost vectors)
- FORGET -> remove vector + document blob + FTS entry

## What REMEMBER Doesn't Do (yet)

- **Query-type routing** - treats "What is X?" and "X vs Y?" the same way
- **Multi-hop reasoning** - can't combine graph traversal + vector search in one query
- **Memory consolidation** - can't summarize old memories (needs LLM callback)

For multi-hop: run RECALL first (graph), then REMEMBER on the results (semantic). The agent orchestrates the composition.

## MCP Tool Mapping

For the MCP server, expose REMEMBER as the primary retrieval tool:

```json
{
  "name": "remember",
  "description": "Retrieve relevant context from agent memory",
  "parameters": {
    "query": "natural language query",
    "tokens": 4000,
    "filter": "optional WHERE clause"
  }
}
```

The other modes (SIMILAR TO, LEXICAL SEARCH, RECALL) can be separate tools for power users, but REMEMBER should be the default.
