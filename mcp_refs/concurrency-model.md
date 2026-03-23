# Concurrency Model: One Agent, One DB

## Decision

Single agent per graphstore instance. Not a shared database.

## Why

- No locking, no contention, no deadlocks
- Optimizer can compact/remap slots without coordination
- Single-threaded Python (GIL) means no concurrent mutations
- Agent controls its own lifecycle - it decides when to read, write, optimize

## Implications for MCP

- MCP server wraps one GraphStore instance
- If multiple agents need shared state, they use separate DBs + a coordination layer above graphstore
- The `_optimizing` flag is a re-entrancy guard, not a mutex
- No need for MVCC, WAL readers, or read snapshots
