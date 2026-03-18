# Graphstore Semantic Overview

This directory contains the semantic understanding of the `graphstore` codebase. Unlike a raw API dump or literal file-by-file AST traversal, this index provides a high-level conceptual mapping of how the database functions, the design decisions, and how data flows through the system.

## Key Concepts documentation
- **[System Architecture](architecture.md)**: Details the in-memory storage model, the sparse matrix engine, and the persistence layer.
- **[Query Lifecycle](query_lifecycle.md)**: Explains the journey of a DSL query from parsing, transformation, execution, to results yielding.

## What is `graphstore`?
`graphstore` is a lightweight, strictly typed in-memory graph database backed by NumPy and SciPy. It allows for extremely fast neighbor lookups (out-degree, in-degree) via SciPy CSR (Compressed Sparse Row) matrices, while metadata (node dictionaries) are stored in pre-allocated normal Python lists and NumPy arrays.
It uses a human-readable Domain Specific Language (DSL) without the overhead of Cypher or SPARQL, and persists data securely to SQLite over a Write-Ahead Log (WAL).
