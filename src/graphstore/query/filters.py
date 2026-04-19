"""Predicate algebra for WHERE clauses (Django-Q-inspired, immutable).

Terminology:
  Leaf         a single comparison: field op value  (e.g. ``kind = "memory"``)
  Not          negation of another F
  And / Or     n-ary conjunction / disjunction (flattened for associativity)

The algebra supports:
  - ``&`` AND            (commutative, associative, identity: true-leaf)
  - ``|`` OR             (commutative, associative, identity: false-leaf)
  - ``~`` NOT            (involution: ``~~f == f``)
  - ``F.raw(expr)``      unescaped escape hatch
  - ``F.from_dict({})``  dict shorthand -> F tree

Compiled to DSL via ``.to_dsl()``. Compilation is pure; the F tree is
never mutated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from graphstore.query.escape import dsl_identifier, dsl_literal


# -------------------------------------------------------------------------
# Operator registry
# -------------------------------------------------------------------------

_COMPARISON_OPS: dict[str, str] = {
    "eq":  "=",
    "ne":  "!=",
    "gt":  ">",
    "gte": ">=",
    "lt":  "<",
    "lte": "<=",
}

_MEMBERSHIP_OPS: dict[str, str] = {
    "in":     "IN",
    "not_in": "NOT IN",
}

_STRING_OPS: dict[str, str] = {
    "startswith": "STARTSWITH",
    "contains":   "CONTAINS",
}

_UNARY_OPS: dict[str, str] = {
    "is_null":     "IS NULL",
    "is_not_null": "IS NOT NULL",
}

ALL_OPS = frozenset(_COMPARISON_OPS) | frozenset(_MEMBERSHIP_OPS) | frozenset(_STRING_OPS) | frozenset(_UNARY_OPS)


# -------------------------------------------------------------------------
# F nodes
# -------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class F:
    """Base predicate. Subclasses are Leaf / Not / And / Or / Raw / Const."""

    def __and__(self, other: "F") -> "F":
        if not isinstance(other, F):
            return NotImplemented
        # true & x == x  / false & x == false
        if isinstance(self, _Const):
            return other if self.value else self
        if isinstance(other, _Const):
            return self if other.value else other
        # flatten nested And
        left = self.operands if isinstance(self, _And) else (self,)
        right = other.operands if isinstance(other, _And) else (other,)
        return _And(left + right)

    def __or__(self, other: "F") -> "F":
        if not isinstance(other, F):
            return NotImplemented
        # false | x == x  / true | x == true
        if isinstance(self, _Const):
            return self if self.value else other
        if isinstance(other, _Const):
            return other if other.value else self
        left = self.operands if isinstance(self, _Or) else (self,)
        right = other.operands if isinstance(other, _Or) else (other,)
        return _Or(left + right)

    def __invert__(self) -> "F":
        # double negation
        if isinstance(self, _Not):
            return self.inner
        return _Not(self)

    def to_dsl(self) -> str:
        raise NotImplementedError

    # -- Builder class methods ------------------------------------------

    @classmethod
    def eq(cls, field_name: str, value: Any) -> "F":
        return _Leaf(field_name, "eq", value)

    @classmethod
    def ne(cls, field_name: str, value: Any) -> "F":
        return _Leaf(field_name, "ne", value)

    @classmethod
    def gt(cls, field_name: str, value: Any) -> "F":
        return _Leaf(field_name, "gt", value)

    @classmethod
    def gte(cls, field_name: str, value: Any) -> "F":
        return _Leaf(field_name, "gte", value)

    @classmethod
    def lt(cls, field_name: str, value: Any) -> "F":
        return _Leaf(field_name, "lt", value)

    @classmethod
    def lte(cls, field_name: str, value: Any) -> "F":
        return _Leaf(field_name, "lte", value)

    @classmethod
    def in_(cls, field_name: str, values: list | tuple) -> "F":
        if not isinstance(values, (list, tuple)):
            raise TypeError(f"F.in_ requires list or tuple, got {type(values).__name__}")
        if not values:
            raise ValueError("F.in_ requires a non-empty sequence")
        return _Leaf(field_name, "in", tuple(values))

    @classmethod
    def not_in(cls, field_name: str, values: list | tuple) -> "F":
        if not isinstance(values, (list, tuple)):
            raise TypeError(f"F.not_in requires list or tuple, got {type(values).__name__}")
        if not values:
            raise ValueError("F.not_in requires a non-empty sequence")
        return _Leaf(field_name, "not_in", tuple(values))

    @classmethod
    def startswith(cls, field_name: str, prefix: str) -> "F":
        return _Leaf(field_name, "startswith", prefix)

    @classmethod
    def contains(cls, field_name: str, substr: str) -> "F":
        return _Leaf(field_name, "contains", substr)

    @classmethod
    def is_null(cls, field_name: str) -> "F":
        return _Leaf(field_name, "is_null", None)

    @classmethod
    def is_not_null(cls, field_name: str) -> "F":
        return _Leaf(field_name, "is_not_null", None)

    @classmethod
    def raw(cls, dsl_expr: str) -> "F":
        if not isinstance(dsl_expr, str) or not dsl_expr.strip():
            raise ValueError("F.raw requires a non-empty expression string")
        return _Raw(dsl_expr)

    @classmethod
    def true(cls) -> "F":
        return _Const(True)

    @classmethod
    def false(cls) -> "F":
        return _Const(False)

    @classmethod
    def from_dict(cls, d: dict) -> "F":
        """Compile a dict shorthand into an F tree.

        Plain keys compile to ``field = value``. Keys ending in ``__op`` use
        the named operator. ``__or__`` / ``__and__`` / ``__not__`` keys
        introduce explicit groupings.
        """
        if not d:
            return _Const(True)
        leaves: list[F] = []
        for key, value in d.items():
            if key == "__and__":
                if not isinstance(value, list):
                    raise ValueError("__and__ expects list of dict predicates")
                sub = cls.true()
                for child in value:
                    sub = sub & cls.from_dict(child)
                leaves.append(sub)
                continue
            if key == "__or__":
                if not isinstance(value, list):
                    raise ValueError("__or__ expects list of dict predicates")
                sub = cls.false()
                for child in value:
                    sub = sub | cls.from_dict(child)
                leaves.append(sub)
                continue
            if key == "__not__":
                if not isinstance(value, dict):
                    raise ValueError("__not__ expects dict predicate")
                leaves.append(~cls.from_dict(value))
                continue
            # Split field__op or treat as plain eq
            if "__" in key and key.rsplit("__", 1)[1] in ALL_OPS:
                field_name, op = key.rsplit("__", 1)
                leaves.append(_Leaf(field_name, op, value))
            else:
                leaves.append(_Leaf(key, "eq", value))
        if len(leaves) == 1:
            return leaves[0]
        result = leaves[0]
        for extra in leaves[1:]:
            result = result & extra
        return result


@dataclass(frozen=True, slots=True)
class _Leaf(F):
    field_name: str
    op: str
    value: Any

    def to_dsl(self) -> str:
        name = dsl_identifier(self.field_name)
        if self.op in _COMPARISON_OPS:
            return f"{name} {_COMPARISON_OPS[self.op]} {dsl_literal(self.value)}"
        if self.op in _MEMBERSHIP_OPS:
            return f"{name} {_MEMBERSHIP_OPS[self.op]} {dsl_literal(list(self.value))}"
        if self.op in _STRING_OPS:
            return f"{name} {_STRING_OPS[self.op]} {dsl_literal(self.value)}"
        if self.op in _UNARY_OPS:
            return f"{name} {_UNARY_OPS[self.op]}"
        raise ValueError(f"unknown op {self.op!r}. Valid: {sorted(ALL_OPS)}")


@dataclass(frozen=True, slots=True)
class _Not(F):
    inner: F

    def to_dsl(self) -> str:
        return f"NOT ({self.inner.to_dsl()})"


@dataclass(frozen=True, slots=True)
class _And(F):
    operands: tuple[F, ...]

    def to_dsl(self) -> str:
        return " AND ".join(f"({op.to_dsl()})" if isinstance(op, _Or) else op.to_dsl() for op in self.operands)


@dataclass(frozen=True, slots=True)
class _Or(F):
    operands: tuple[F, ...]

    def to_dsl(self) -> str:
        return "(" + " OR ".join(op.to_dsl() for op in self.operands) + ")"


@dataclass(frozen=True, slots=True)
class _Raw(F):
    expr: str

    def to_dsl(self) -> str:
        return self.expr


@dataclass(frozen=True, slots=True)
class _Const(F):
    value: bool

    def to_dsl(self) -> str:
        return "true" if self.value else "false"


def compile_where(where: F | dict | None) -> str | None:
    """Normalise a ``where`` argument to a DSL string or ``None``."""
    if where is None:
        return None
    if isinstance(where, dict):
        if not where:
            return None
        where = F.from_dict(where)
    if isinstance(where, F):
        if isinstance(where, _Const) and where.value:
            return None
        return where.to_dsl()
    raise TypeError(f"where must be F, dict, or None; got {type(where).__name__}")
