# Graphstore Semantic Overview

This directory contains the semantic understanding of the `graphstore` codebase. Unlike a raw API dump or literal file-by-file AST traversal, this index provides a high-level conceptual mapping of how the database functions, the design decisions, and how data flows through the system.

## Key Concepts documentation
- **[System Architecture](architecture.md)**: Details the columnar storage model, sparse matrix engine, and persistence layer.
- **[Query Lifecycle](query_lifecycle.md)**: Explains the journey of a DSL query from parsing, transformation, execution, to results yielding.

## What is `graphstore`?
`graphstore` is a lightweight, strictly typed in-memory graph database designed as an **agentic brain** - a memory substrate for AI agents. Backed by NumPy columnar arrays and SciPy CSR sparse matrices, it provides microsecond-level filtering, aggregation, and associative recall.

It uses a human-readable DSL (no Cypher, no SPARQL), and persists data to SQLite over a Write-Ahead Log (WAL).

### Core capabilities
- **Columnar storage** - all node fields stored as typed numpy arrays (int64, float64, int32-interned strings). 14x less memory than dict-based storage, 216x faster COUNT queries.
- **Belief management** - ASSERT facts with confidence/source, RETRACT outdated beliefs, detect contradictions automatically.
- **Associative recall** - RECALL via spreading activation (sparse matrix-vector multiply) weighted by importance and recency.
- **Temporal awareness** - auto-timestamps, TTL with expiry, relative time queries (NOW(), TODAY, YESTERDAY).
- **Hypothesis testing** - SYS SNAPSHOT/ROLLBACK for reasoning branches, WHAT IF for counterfactual analysis.
- **Aggregation** - GROUP BY with SUM/AVG/MIN/MAX/COUNT DISTINCT via numpy vectorized ops.
- **Context isolation** - BIND/DISCARD CONTEXT for isolated reasoning sessions.
