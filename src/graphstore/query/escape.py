"""DSL value escaping. One function per Python type. No state.

Every user-supplied value that lands in a DSL query goes through
``dsl_literal``. The function is the single point where injection is
prevented. Anything that cannot be safely coerced raises ``TypeError``.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any


def _escape_string(s: str) -> str:
    """Escape a string for DSL: backslash and double-quote escapes."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def dsl_literal(value: Any) -> str:
    """Coerce a Python value to a DSL literal string.

    Type handling:
      - str          -> quoted, backslash+quote escaped
      - bool         -> ``true`` / ``false`` (bool is checked before int because
                        ``isinstance(True, int)`` is True)
      - int / float  -> decimal literal
      - None         -> ``null``
      - list / tuple -> parenthesised comma-separated literals (for IN clauses)
      - datetime     -> ISO-8601 quoted (``"YYYY-MM-DDTHH:MM:SS"``)
      - date         -> ``"YYYY-MM-DD"`` quoted
      - anything else -> ``TypeError``
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return _escape_string(value)
    if isinstance(value, datetime):
        return _escape_string(value.isoformat())
    if isinstance(value, date):
        return _escape_string(value.isoformat())
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("empty list is not a valid DSL literal (use a WHERE op like __in with a non-empty list)")
        return "(" + ", ".join(dsl_literal(v) for v in value) + ")"
    raise TypeError(
        f"unsupported DSL value type: {type(value).__name__}. "
        f"supported: str, int, float, bool, None, list, tuple, datetime, date"
    )


def dsl_identifier(name: str) -> str:
    """Emit a field / column name. Currently unquoted per grammar.

    Only accepts identifiers matching ``[a-zA-Z_][a-zA-Z0-9_]*``. Reserved
    double-underscore names (``__event_at__`` etc.) are allowed.
    """
    if not name or not isinstance(name, str):
        raise ValueError(f"identifier must be a non-empty str, got {name!r}")
    first = name[0]
    if not (first.isalpha() or first == "_"):
        raise ValueError(f"identifier must start with letter or underscore: {name!r}")
    for ch in name[1:]:
        if not (ch.isalnum() or ch == "_"):
            raise ValueError(f"identifier contains invalid character {ch!r}: {name!r}")
    return name


def dsl_node_id(node_id: str) -> str:
    """Emit a node identifier in its DSL form (quoted string)."""
    if not isinstance(node_id, str) or not node_id:
        raise ValueError(f"node id must be a non-empty str, got {node_id!r}")
    return _escape_string(node_id)
