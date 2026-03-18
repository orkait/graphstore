# Query Lifecycle

The execution of a query in `graphstore` involves a streamlined pipeline driven by the bespoke Domain Specific Language (DSL).

## 1. Parsing and Grammar (`dsl/parser.py`, `dsl/grammar.lark`)
1. **LALR(1) Parsing**: Queries are evaluated through a customized context-free grammar using the `Lark` parser. 
2. **LRU Cache**: A parsed request caches the abstract syntax tree (AST). If the exact statement structure hits the `parser.py` cache, it skips the heavy parsing entirely.

## 2. AST Transformation (`dsl/transformer.py`, `dsl/ast_nodes.py`)
Once a raw parse tree is generated, the `transformer.py` script traverses it bottom-up and maps the parsed symbols into rigorous, static data classes defined in `ast_nodes.py`.
- **Examples**: `NodesQuery`, `MatchQuery`, `CreateEdge`, `Batch`.
- This ensures that execution receives a strictly typed query representation (e.g., conditions, limit limits, traversal depths).

## 3. Cost Estimation (`dsl/cost_estimator.py`)
To protect against potentially infinite or extraordinarily memory-intensive operations:
- Queries like `TRAVERSE`, `MATCH`, or `PATH` undergo pre-execution evaluation. 
- The system checks degree bounds recursively up to the configured `depth`. If the "frontier" of visited nodes scales past a secure threshold (e.g., 100,000 nodes), it proactively raises `CostThresholdExceeded`.

## 4. Execution Pipeline (`dsl/executor.py`)
The `executor.py` dispatches the specific AST Object into the `Result` yielding workflow.
- **Reads**: Functions like `_traverse` heavily utilize `path.py` utility methods like `bfs_traverse` over the scipy `EdgeMatrices`.
  - For standard pattern matching (`MATCH ... -> () -> ...`), sparse matrices rapidly collect neighbor pointers until the match completes.
- **Writes/Mutates**: Write-queries (like `UPDATE NODE`) go through the `CoreStore` and directly manipulate numpy properties or dictionaries.
- **Transactions (`Batch`)**: Executed linearly. If a sub-statement violates memory limits, fails logic constraints, or causes an error, it manually triggers a deep python rollback (`try...except BatchRollback`) restoring pre-transaction `CoreStore` array snapshots.

## 5. Yielding Results (`types.py:Result`)
Execution ultimately folds into `Result(kind=..., data=..., count=...)`. The top-level Server or Library call consumes this standard object without exposing sparse matrix objects to the user.
