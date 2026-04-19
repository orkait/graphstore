"""Write-verb builders. Clause ordering matches grammar.lark exactly."""
from __future__ import annotations

import re
from typing import Any

from graphstore.query.escape import dsl_identifier, dsl_literal, dsl_node_id
from graphstore.query.runtime import Query, register_compiler


# Reserved clause keywords handled separately from arbitrary field kwargs.
_CREATE_NODE_RESERVED = {"kind", "event_at", "expires_in", "document", "vector"}


def _format_field(name: str, value: Any) -> str:
    return f"{dsl_identifier(name)} = {dsl_literal(value)}"


def _parse_expires_in(spec: str) -> tuple[int, str]:
    """Parse "1h" / "30s" / "7d" -> (1, "h"). Grammar: NUMBER TIME_UNIT."""
    if not isinstance(spec, str):
        raise ValueError(f"expires_in must be a str like '1h', got {spec!r}")
    m = re.match(r"^(\d+)\s*([smhd])$", spec.strip())
    if not m:
        raise ValueError(f"expires_in must match <NUMBER><smhd>, got {spec!r}")
    return int(m.group(1)), m.group(2)


# ---------- CREATE NODE ---------------------------------------------------
# create_node: "CREATE" "NODE" STRING field_pairs vector_clause? expires_clause? event_clause? document_clause?

def create_node(
    id: str,
    *,
    kind: str,
    event_at: Any = None,
    expires_in: str | None = None,
    document: str | None = None,
    vector: list[float] | None = None,
    **fields: Any,
) -> Query:
    if not isinstance(kind, str) or not kind:
        raise ValueError("create_node() requires kind= as a non-empty str")
    overlap = _CREATE_NODE_RESERVED & set(fields)
    if overlap:
        raise TypeError(
            f"create_node() got reserved kwarg(s) via **fields: {sorted(overlap)}. "
            f"Use the dedicated parameter instead."
        )
    params: dict = {
        "id": id,
        "kind": kind,
        "fields": dict(fields),
    }
    if event_at is not None: params["event_at"] = event_at
    if expires_in is not None: params["expires_in"] = expires_in
    if document is not None: params["document"] = document
    if vector is not None: params["vector"] = list(vector)
    return Query(_verb="create_node", _params=params, _kind="write")


def _compile_create_node(p: dict) -> str:
    parts = [f"CREATE NODE {dsl_node_id(p['id'])}"]
    parts.append(_format_field("kind", p["kind"]))
    for name, value in p["fields"].items():
        parts.append(_format_field(name, value))
    if "vector" in p:
        parts.append("VECTOR [" + ", ".join(dsl_literal(v) for v in p["vector"]) + "]")
    if "expires_in" in p:
        n, u = _parse_expires_in(p["expires_in"])
        parts.append(f"EXPIRES IN {n}{u}")
    if "event_at" in p:
        parts.append(f"EVENT_AT {dsl_literal(p['event_at'])}")
    if "document" in p:
        parts.append(f"DOCUMENT {dsl_literal(p['document'])}")
    return " ".join(parts)


register_compiler("create_node", _compile_create_node)


# ---------- CREATE EDGE ---------------------------------------------------
# create_edge: "CREATE" "EDGE" node_ref "->" node_ref field_pairs

def create_edge(
    src: str,
    tgt: str,
    *,
    kind: str,
    **fields: Any,
) -> Query:
    if not isinstance(kind, str) or not kind:
        raise ValueError("create_edge() requires kind= as a non-empty str")
    return Query(
        _verb="create_edge",
        _params={"src": src, "tgt": tgt, "kind": kind, "fields": dict(fields)},
        _kind="write",
    )


def _compile_create_edge(p: dict) -> str:
    parts = [f"CREATE EDGE {dsl_node_id(p['src'])} -> {dsl_node_id(p['tgt'])}"]
    parts.append(_format_field("kind", p["kind"]))
    for name, value in p["fields"].items():
        parts.append(_format_field(name, value))
    return " ".join(parts)


register_compiler("create_edge", _compile_create_edge)


# ---------- DELETE NODE ---------------------------------------------------
# delete_node: "DELETE" "NODE" STRING

def delete_node(id: str) -> Query:
    return Query(_verb="delete_node", _params={"id": id}, _kind="write")


def _compile_delete_node(p: dict) -> str:
    return f"DELETE NODE {dsl_node_id(p['id'])}"


register_compiler("delete_node", _compile_delete_node)
