"""Exception hierarchy for graphstore."""


class GraphStoreError(Exception):
    """Base exception for all graphstore errors."""


class QueryError(GraphStoreError):
    """DSL parse or validation failure."""

    def __init__(self, message: str, position: int = -1, query: str = "") -> None:
        self.message = message
        self.position = position
        self.query = query
        super().__init__(message)

    def __str__(self) -> str:
        parts = [self.message]
        if self.position >= 0:
            parts.append(f"at position {self.position}")
        if self.query:
            parts.append(f"in query: {self.query!r}")
        return " ".join(parts)


class NodeNotFound(GraphStoreError):
    """UPDATE/INCREMENT on a node that does not exist."""

    def __init__(self, id: str) -> None:
        self.id = id
        super().__init__(f"Node not found: {id!r}")


class NodeExists(GraphStoreError):
    """CREATE on a node that already exists."""

    def __init__(self, id: str) -> None:
        self.id = id
        super().__init__(f"Node already exists: {id!r}")


class CeilingExceeded(GraphStoreError):
    """Memory ceiling has been exceeded."""

    def __init__(self, current_mb: int, ceiling_mb: int, operation: str) -> None:
        self.current_mb = current_mb
        self.ceiling_mb = ceiling_mb
        self.operation = operation
        super().__init__(
            f"Memory ceiling exceeded: {current_mb} MB / {ceiling_mb} MB "
            f"during {operation!r}"
        )


class VersionMismatch(GraphStoreError):
    """Store version does not match expected version."""

    def __init__(self, found: str | None, expected: int) -> None:
        self.found = found
        self.expected = expected
        super().__init__(
            f"Version mismatch: found {found!r}, expected {expected}"
        )


class SchemaError(GraphStoreError):
    """Write violates a registered schema."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class AggregationError(SchemaError):
    """Aggregation requires columnarized fields."""
    pass


class CostThresholdExceeded(GraphStoreError):
    """MATCH/TRAVERSE estimated cost exceeds threshold."""

    def __init__(self, estimated_frontier: float, threshold: float) -> None:
        self.estimated_frontier = estimated_frontier
        self.threshold = threshold
        super().__init__(
            f"Cost threshold exceeded: estimated frontier {estimated_frontier} "
            f"> threshold {threshold}"
        )


class BatchRollback(GraphStoreError):
    """A statement within BEGIN/COMMIT failed, triggering rollback."""

    def __init__(self, failed_statement: str, error: str) -> None:
        self.failed_statement = failed_statement
        self.error = error
        super().__init__(
            f"Batch rollback on statement {failed_statement!r}: {error}"
        )
