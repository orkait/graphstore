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


# ---------- UPDATE NODE ---------------------------------------------------
# update_node: "UPDATE" "NODE" STRING "SET" field_pairs

def update_node(id: str, **set_fields: Any) -> Query:
    if not set_fields:
        raise ValueError("update_node() requires at least one field=value to set")
    return Query(
        _verb="update_node",
        _params={"id": id, "fields": dict(set_fields)},
        _kind="write",
    )


def _compile_update_node(p: dict) -> str:
    parts = [f"UPDATE NODE {dsl_node_id(p['id'])}", "SET"]
    parts.extend(_format_field(n, v) for n, v in p["fields"].items())
    return " ".join(parts)


register_compiler("update_node", _compile_update_node)


# ---------- UPSERT NODE ---------------------------------------------------
# upsert_node: "UPSERT" "NODE" STRING field_pairs vector? expires? event_at?

_UPSERT_RESERVED = {"vector", "expires_in", "event_at"}


def upsert_node(
    id: str,
    *,
    kind: str | None = None,
    vector: list[float] | None = None,
    expires_in: str | None = None,
    event_at: Any = None,
    **fields: Any,
) -> Query:
    overlap = _UPSERT_RESERVED & set(fields)
    if overlap:
        raise TypeError(
            f"upsert_node() got reserved kwarg(s) via **fields: {sorted(overlap)}"
        )
    fp: dict[str, Any] = {}
    if kind is not None:
        fp["kind"] = kind
    fp.update(fields)
    params: dict = {"id": id, "fields": fp}
    if vector is not None: params["vector"] = list(vector)
    if expires_in is not None: params["expires_in"] = expires_in
    if event_at is not None: params["event_at"] = event_at
    return Query(_verb="upsert_node", _params=params, _kind="write")


def _compile_upsert_node(p: dict) -> str:
    parts = [f"UPSERT NODE {dsl_node_id(p['id'])}"]
    for name, value in p["fields"].items():
        parts.append(_format_field(name, value))
    if "vector" in p:
        parts.append("VECTOR [" + ", ".join(dsl_literal(v) for v in p["vector"]) + "]")
    if "expires_in" in p:
        n, u = _parse_expires_in(p["expires_in"])
        parts.append(f"EXPIRES IN {n}{u}")
    if "event_at" in p:
        parts.append(f"EVENT_AT {dsl_literal(p['event_at'])}")
    return " ".join(parts)


register_compiler("upsert_node", _compile_upsert_node)


# ---------- DELETE NODES --------------------------------------------------
# delete_nodes: "DELETE" "NODES" where_clause

from graphstore.query.filters import F, compile_where


def delete_nodes(*, where) -> Query:
    if where is None:
        raise ValueError("delete_nodes() requires where= to prevent accidental mass deletion")
    return Query(_verb="delete_nodes", _params={"where": where}, _kind="write")


def _compile_delete_nodes(p: dict) -> str:
    w = compile_where(p["where"])
    if not w:
        raise ValueError("delete_nodes compiled to empty WHERE; use delete_node(id) for unconditional deletion of one node")
    return f"DELETE NODES WHERE {w}"


register_compiler("delete_nodes", _compile_delete_nodes)


# ---------- UPDATE NODES --------------------------------------------------
# update_nodes: "UPDATE" "NODES" where "SET" field_pairs

def update_nodes(*, where, set: dict) -> Query:
    if where is None:
        raise ValueError("update_nodes() requires where=")
    if not isinstance(set, dict) or not set:
        raise ValueError("update_nodes() requires set={field: value, ...} non-empty dict")
    return Query(
        _verb="update_nodes",
        _params={"where": where, "set": dict(set)},
        _kind="write",
    )


def _compile_update_nodes(p: dict) -> str:
    w = compile_where(p["where"])
    parts = [f"UPDATE NODES WHERE {w}", "SET"]
    parts.extend(_format_field(n, v) for n, v in p["set"].items())
    return " ".join(parts)


register_compiler("update_nodes", _compile_update_nodes)


# ---------- UPDATE EDGE ---------------------------------------------------
# update_edge: "UPDATE" "EDGE" STRING "->" STRING "SET" field_pairs where?

def update_edge(src: str, tgt: str, *, set: dict, where=None) -> Query:
    if not isinstance(set, dict) or not set:
        raise ValueError("update_edge() requires set={field: value, ...} non-empty dict")
    params: dict = {"src": src, "tgt": tgt, "set": dict(set)}
    if where is not None: params["where"] = where
    return Query(_verb="update_edge", _params=params, _kind="write")


def _compile_update_edge(p: dict) -> str:
    parts = [f"UPDATE EDGE {dsl_node_id(p['src'])} -> {dsl_node_id(p['tgt'])}", "SET"]
    parts.extend(_format_field(n, v) for n, v in p["set"].items())
    w = compile_where(p.get("where"))
    if w:
        parts.append(f"WHERE {w}")
    return " ".join(parts)


register_compiler("update_edge", _compile_update_edge)


# ---------- DELETE EDGE ---------------------------------------------------
# delete_edge: "DELETE" "EDGE" STRING "->" STRING where?

def delete_edge(src: str, tgt: str, *, where=None) -> Query:
    params: dict = {"src": src, "tgt": tgt}
    if where is not None: params["where"] = where
    return Query(_verb="delete_edge", _params=params, _kind="write")


def _compile_delete_edge(p: dict) -> str:
    out = f"DELETE EDGE {dsl_node_id(p['src'])} -> {dsl_node_id(p['tgt'])}"
    w = compile_where(p.get("where"))
    if w:
        out += f" WHERE {w}"
    return out


register_compiler("delete_edge", _compile_delete_edge)


# ---------- DELETE EDGES (bulk) -------------------------------------------
# delete_edges: "DELETE" "EDGES" direction STRING where?

def delete_edges(node: str, *, direction: str = "FROM", where=None) -> Query:
    if direction not in ("FROM", "TO"):
        raise ValueError(f'delete_edges() direction must be "FROM" or "TO", got {direction!r}')
    params: dict = {"node": node, "direction": direction}
    if where is not None: params["where"] = where
    return Query(_verb="delete_edges", _params=params, _kind="write")


def _compile_delete_edges(p: dict) -> str:
    out = f"DELETE EDGES {p['direction']} {dsl_node_id(p['node'])}"
    w = compile_where(p.get("where"))
    if w:
        out += f" WHERE {w}"
    return out


register_compiler("delete_edges", _compile_delete_edges)


# ---------- INCREMENT -----------------------------------------------------
# increment: "INCREMENT" "NODE" STRING IDENTIFIER "BY" NUMBER

def increment(id: str, field_name: str, *, by: int | float) -> Query:
    dsl_identifier(field_name)
    if not isinstance(by, (int, float)) or isinstance(by, bool):
        raise ValueError(f"increment() by= must be int or float, got {by!r}")
    return Query(
        _verb="increment",
        _params={"id": id, "field": field_name, "by": by},
        _kind="write",
    )


def _compile_increment(p: dict) -> str:
    return f"INCREMENT NODE {dsl_node_id(p['id'])} {p['field']} BY {p['by']}"


register_compiler("increment", _compile_increment)


# ---------- ASSERT --------------------------------------------------------
# assert_stmt: "ASSERT" STRING field_pairs confidence? source? event_at?

_ASSERT_RESERVED = {"confidence", "source", "event_at"}


def assert_(
    id: str,
    *,
    kind: str,
    value: Any = None,
    confidence: float | None = None,
    source: str | None = None,
    event_at: Any = None,
    **fields: Any,
) -> Query:
    if not isinstance(kind, str) or not kind:
        raise ValueError("assert_() requires kind= as a non-empty str")
    overlap = _ASSERT_RESERVED & set(fields)
    if overlap:
        raise TypeError(f"assert_() got reserved kwarg(s) via **fields: {sorted(overlap)}")
    fp: dict[str, Any] = {"kind": kind}
    if value is not None:
        fp["value"] = value
    fp.update(fields)
    params: dict = {"id": id, "fields": fp}
    if confidence is not None: params["confidence"] = confidence
    if source is not None: params["source"] = source
    if event_at is not None: params["event_at"] = event_at
    return Query(_verb="assert_", _params=params, _kind="write")


def _compile_assert(p: dict) -> str:
    parts = [f"ASSERT {dsl_node_id(p['id'])}"]
    for name, value in p["fields"].items():
        parts.append(_format_field(name, value))
    if "confidence" in p:
        parts.append(f"CONFIDENCE {p['confidence']}")
    if "source" in p:
        parts.append(f"SOURCE {dsl_literal(p['source'])}")
    if "event_at" in p:
        parts.append(f"EVENT_AT {dsl_literal(p['event_at'])}")
    return " ".join(parts)


register_compiler("assert_", _compile_assert)


# ---------- RETRACT -------------------------------------------------------
# retract_stmt: "RETRACT" STRING reason?

def retract(id: str, *, reason: str | None = None) -> Query:
    params: dict = {"id": id}
    if reason is not None: params["reason"] = reason
    return Query(_verb="retract", _params=params, _kind="write")


def _compile_retract(p: dict) -> str:
    out = f"RETRACT {dsl_node_id(p['id'])}"
    if "reason" in p:
        out += f" REASON {dsl_literal(p['reason'])}"
    return out


register_compiler("retract", _compile_retract)


# ---------- MERGE ---------------------------------------------------------
# merge_stmt: "MERGE" "NODE" STRING "INTO" STRING

def merge(old: str, into: str) -> Query:
    return Query(_verb="merge", _params={"old": old, "into": into}, _kind="write")


def _compile_merge(p: dict) -> str:
    return f"MERGE NODE {dsl_node_id(p['old'])} INTO {dsl_node_id(p['into'])}"


register_compiler("merge", _compile_merge)


# ---------- PROPAGATE -----------------------------------------------------
# propagate_stmt: "PROPAGATE" STRING "FIELD" IDENTIFIER "DEPTH" NUMBER

def propagate(id: str, *, field: str, depth: int) -> Query:
    dsl_identifier(field)
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 0:
        raise ValueError(f"propagate() depth must be a non-negative int, got {depth!r}")
    return Query(
        _verb="propagate",
        _params={"id": id, "field": field, "depth": depth},
        _kind="write",
    )


def _compile_propagate(p: dict) -> str:
    return f"PROPAGATE {dsl_node_id(p['id'])} FIELD {p['field']} DEPTH {p['depth']}"


register_compiler("propagate", _compile_propagate)


# ---------- BIND / DISCARD CONTEXT ----------------------------------------
# bind_context / discard_context

def bind_context(name: str) -> Query:
    return Query(_verb="bind_context", _params={"name": name}, _kind="control")


def _compile_bind_context(p: dict) -> str:
    return f"BIND CONTEXT {dsl_literal(p['name'])}"


register_compiler("bind_context", _compile_bind_context)


def discard_context(name: str) -> Query:
    return Query(_verb="discard_context", _params={"name": name}, _kind="control")


def _compile_discard_context(p: dict) -> str:
    return f"DISCARD CONTEXT {dsl_literal(p['name'])}"


register_compiler("discard_context", _compile_discard_context)


# ---------- FORGET --------------------------------------------------------
# forget_node: "FORGET" "NODE" STRING

def forget(id: str) -> Query:
    return Query(_verb="forget", _params={"id": id}, _kind="write")


def _compile_forget(p: dict) -> str:
    return f"FORGET NODE {dsl_node_id(p['id'])}"


register_compiler("forget", _compile_forget)


# ---------- CONNECT NODE --------------------------------------------------
# connect_node: "CONNECT" "NODE" STRING threshold?

def connect_node(id: str, *, threshold: float | None = None) -> Query:
    params: dict = {"id": id}
    if threshold is not None:
        params["threshold"] = threshold
    return Query(_verb="connect_node", _params=params, _kind="write")


def _compile_connect_node(p: dict) -> str:
    out = f"CONNECT NODE {dsl_node_id(p['id'])}"
    if "threshold" in p:
        out += f" THRESHOLD {p['threshold']}"
    return out


register_compiler("connect_node", _compile_connect_node)


# ---------- INGEST --------------------------------------------------------
# ingest_stmt: "INGEST" STRING ingest_as? ingest_kind? ingest_using?
# ingest_using: vision_clause | using_clause
# vision_clause: "USING" "VISION" STRING
# using_clause:  "USING" IDENTIFIER

_KNOWN_USING = {"markitdown", "pymupdf4llm", "docling", "direct", "whisper", "vision"}


def ingest(
    file: str,
    *,
    as_id: str | None = None,
    kind: str | None = None,
    using: str | None = None,
    vision_model: str | None = None,
) -> Query:
    if not isinstance(file, str) or not file:
        raise ValueError("ingest() requires a non-empty file path")
    if vision_model is not None and using not in (None, "vision"):
        raise ValueError("ingest() vision_model= is only valid with using='vision'")
    if using is not None and using != "vision" and using not in _KNOWN_USING:
        raise ValueError(f"ingest() using={using!r} not in known list {sorted(_KNOWN_USING)}")
    params: dict = {"file": file}
    if as_id is not None: params["as_id"] = as_id
    if kind is not None: params["kind"] = kind
    if using is not None: params["using"] = using
    if vision_model is not None: params["vision_model"] = vision_model
    return Query(_verb="ingest", _params=params, _kind="write")


def _compile_ingest(p: dict) -> str:
    parts = [f"INGEST {dsl_literal(p['file'])}"]
    if "as_id" in p:
        parts.append(f"AS {dsl_literal(p['as_id'])}")
    if "kind" in p:
        parts.append(f"KIND {dsl_literal(p['kind'])}")
    if p.get("using") == "vision":
        model = p.get("vision_model") or ""
        parts.append(f"USING VISION {dsl_literal(model)}")
    elif "using" in p:
        parts.append(f"USING {p['using']}")
    return " ".join(parts)


register_compiler("ingest", _compile_ingest)


# ---------- BATCH WRAPPERS ------------------------------------------------
# batch: "BEGIN" _NL (_batch_stmt _NL)* "COMMIT"
#
# User constructs a batch by composing statements with ``|`` and wrapping
# with q.begin() ... q.commit(). The runtime Query.__or__ already joins
# DSL fragments with newlines.

def begin() -> Query:
    """Open a BEGIN...COMMIT block. Compose with ``|``."""
    return Query(_verb="raw", _params={"text": "BEGIN"}, _kind="control")


def commit() -> Query:
    """Close a BEGIN...COMMIT block."""
    return Query(_verb="raw", _params={"text": "COMMIT"}, _kind="control")


def batch(*statements: Query) -> Query:
    """Wrap a sequence of statements in BEGIN...COMMIT.

    Shorthand for ``q.begin() | stmt1 | stmt2 | ... | q.commit()``.
    """
    if not statements:
        raise ValueError("batch() requires at least one statement")
    body = "\n".join(s.dsl() for s in statements)
    return Query(
        _verb="batch",
        _params={"text": f"BEGIN\n{body}\nCOMMIT"},
        _kind="batch",
    )
