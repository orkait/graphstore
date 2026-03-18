"""Optional type system for graphstore.

When kinds are registered, writes are validated against them.
Unregistered kinds pass through without validation (schema-free mode).
"""

from __future__ import annotations

from dataclasses import dataclass

from graphstore.errors import SchemaError


@dataclass
class NodeKindDef:
    required: set[str]
    optional: set[str]


@dataclass
class EdgeKindDef:
    from_kinds: set[str]
    to_kinds: set[str]


class SchemaRegistry:
    def __init__(self):
        self._node_kinds: dict[str, NodeKindDef] = {}
        self._edge_kinds: dict[str, EdgeKindDef] = {}

    @property
    def has_node_kinds(self) -> bool:
        return bool(self._node_kinds)

    @property
    def has_edge_kinds(self) -> bool:
        return bool(self._edge_kinds)

    def register_node_kind(self, kind: str, required: list[str], optional: list[str] | None = None):
        self._node_kinds[kind] = NodeKindDef(
            required=set(required),
            optional=set(optional or []),
        )

    def register_edge_kind(self, kind: str, from_kinds: list[str], to_kinds: list[str]):
        self._edge_kinds[kind] = EdgeKindDef(
            from_kinds=set(from_kinds),
            to_kinds=set(to_kinds),
        )

    def unregister_node_kind(self, kind: str):
        if kind in self._node_kinds:
            del self._node_kinds[kind]

    def unregister_edge_kind(self, kind: str):
        if kind in self._edge_kinds:
            del self._edge_kinds[kind]

    def get_node_kind(self, kind: str) -> NodeKindDef | None:
        return self._node_kinds.get(kind)

    def get_edge_kind(self, kind: str) -> EdgeKindDef | None:
        return self._edge_kinds.get(kind)

    def list_node_kinds(self) -> list[str]:
        return list(self._node_kinds.keys())

    def list_edge_kinds(self) -> list[str]:
        return list(self._edge_kinds.keys())

    def describe_node_kind(self, kind: str) -> dict | None:
        defn = self._node_kinds.get(kind)
        if defn is None:
            return None
        return {
            "kind": kind,
            "required": sorted(defn.required),
            "optional": sorted(defn.optional),
        }

    def describe_edge_kind(self, kind: str) -> dict | None:
        defn = self._edge_kinds.get(kind)
        if defn is None:
            return None
        return {
            "kind": kind,
            "from_kinds": sorted(defn.from_kinds),
            "to_kinds": sorted(defn.to_kinds),
        }

    def validate_node(self, kind: str, data: dict):
        """Validate node data against registered kind schema.

        If kind is not registered, no validation (schema-free mode).
        Raises SchemaError if required fields are missing.
        """
        if kind not in self._node_kinds:
            return  # unregistered kind, no validation
        defn = self._node_kinds[kind]
        missing = defn.required - set(data.keys())
        if missing:
            raise SchemaError(f"kind '{kind}' requires fields: {sorted(missing)}")

    def validate_edge(self, kind: str, source_kind: str, target_kind: str):
        """Validate edge endpoint kinds against registered schema.

        If kind is not registered, no validation.
        Raises SchemaError if endpoint kinds don't match.
        """
        if kind not in self._edge_kinds:
            return
        defn = self._edge_kinds[kind]
        if source_kind not in defn.from_kinds:
            raise SchemaError(
                f"edge '{kind}' cannot originate from kind '{source_kind}', "
                f"allowed: {sorted(defn.from_kinds)}"
            )
        if target_kind not in defn.to_kinds:
            raise SchemaError(
                f"edge '{kind}' cannot target kind '{target_kind}', "
                f"allowed: {sorted(defn.to_kinds)}"
            )

    def to_dict(self) -> dict:
        """Export schema for serialization."""
        return {
            "node_kinds": {
                k: {"required": sorted(v.required), "optional": sorted(v.optional)}
                for k, v in self._node_kinds.items()
            },
            "edge_kinds": {
                k: {"from_kinds": sorted(v.from_kinds), "to_kinds": sorted(v.to_kinds)}
                for k, v in self._edge_kinds.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> SchemaRegistry:
        """Rebuild from serialized form."""
        registry = cls()
        for kind, defn in data.get("node_kinds", {}).items():
            registry.register_node_kind(kind, defn["required"], defn.get("optional", []))
        for kind, defn in data.get("edge_kinds", {}).items():
            registry.register_edge_kind(kind, defn["from_kinds"], defn["to_kinds"])
        return registry
