"""graphstore query builder: typed, composable, 100% DSL coverage.

Public surface:

  from graphstore.query import q, F, Query, register_verb

  q.remember("European history", limit=10)
  q.nodes(where=F.eq("kind", "memory") & F.gt("importance", 0.5))

Spec: docs/specs/query-builder.md
"""
from __future__ import annotations

from typing import Any

from graphstore.query import plugins
from graphstore.query.filters import F
from graphstore.query.runtime import Query
from graphstore.query.verbs import reads as _reads, writes as _writes, traversal as _traversal

register_verb = plugins.register_verb


class _QNamespace:
    """Attribute-access facade for verb functions + plugin fallthrough.

    ``q.remember(...)`` and ``q.create_node(...)`` look up built-in verbs;
    anything else falls through to the plugin registry so third-party
    packages can register custom verbs without modifying core.
    """

    # --- Reads ---
    node           = staticmethod(_reads.node)
    nodes          = staticmethod(_reads.nodes)
    remember       = staticmethod(_reads.remember)
    recall         = staticmethod(_reads.recall)
    similar        = staticmethod(_reads.similar)
    lexical        = staticmethod(_reads.lexical)
    edges          = staticmethod(_reads.edges)
    count_nodes    = staticmethod(_reads.count_nodes)

    # --- Traversal ---
    traverse       = staticmethod(_traversal.traverse)
    subgraph       = staticmethod(_traversal.subgraph)
    path           = staticmethod(_traversal.path)
    paths          = staticmethod(_traversal.paths)
    shortest_path  = staticmethod(_traversal.shortest_path)
    distance       = staticmethod(_traversal.distance)
    weighted_shortest_path = staticmethod(_traversal.weighted_shortest_path)
    weighted_distance      = staticmethod(_traversal.weighted_distance)
    ancestors      = staticmethod(_traversal.ancestors)
    descendants    = staticmethod(_traversal.descendants)
    common_neighbors = staticmethod(_traversal.common_neighbors)
    match          = staticmethod(_traversal.match)
    what_if_retract = staticmethod(_traversal.what_if_retract)
    aggregate_nodes = staticmethod(_traversal.aggregate_nodes)
    count_edges    = staticmethod(_traversal.count_edges)

    # --- Writes ---
    create_node    = staticmethod(_writes.create_node)
    create_edge    = staticmethod(_writes.create_edge)
    delete_node    = staticmethod(_writes.delete_node)

    # --- Escape hatch ---
    @staticmethod
    def raw(dsl: str, **params: Any) -> Query:
        """Pass-through DSL with optional ``:name`` parameter substitution.

        Every referenced ``:name`` in ``dsl`` must be provided in ``params``;
        values are escaped through the normal DSL literal path.
        """
        from graphstore.query.escape import dsl_literal
        if not isinstance(dsl, str) or not dsl:
            raise ValueError("q.raw() requires a non-empty DSL string")
        import re
        # Match :name but not node-id colons inside quoted strings.
        # Strategy: only recognise :name when preceded by start/space/= /comma/open-paren.
        placeholders = set(re.findall(r"(?<=[\s=,(])\:(\w+)|(?<=^)\:(\w+)", dsl))
        # Also allow plain :name at any position as legacy pattern
        placeholders = set(re.findall(r":(\w+)", dsl)) & set(re.findall(r":([a-zA-Z_]\w*)", dsl))
        # Simpler: treat any :word as placeholder (callers should quote node ids themselves).
        placeholders = set(re.findall(r"(?<!\\):([a-zA-Z_]\w*)", dsl))
        missing = placeholders - set(params)
        if missing:
            raise ValueError(f"q.raw() missing params: {sorted(missing)}")
        unused = set(params) - placeholders
        if unused:
            raise ValueError(f"q.raw() unused params: {sorted(unused)}")
        out = dsl
        for name, value in params.items():
            out = out.replace(f":{name}", dsl_literal(value))
        return Query(_verb="raw", _params={"text": out}, _kind="raw")

    def __getattr__(self, name: str) -> Any:
        fn = plugins._lookup(name)
        if fn is not None:
            return fn
        known = sorted(n for n in dir(self) if not n.startswith("_"))
        registered = plugins._registered_names()
        raise AttributeError(
            f"q has no verb {name!r}. "
            f"Built-in: {known}. Registered: {registered}"
        )


q = _QNamespace()


__all__ = ["q", "F", "Query", "register_verb"]
