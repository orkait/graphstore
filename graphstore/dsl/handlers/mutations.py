"""Mutation handlers for the DSL executor (create, update, delete, merge, batch)."""

import time
from collections import deque

import numpy as np
from scipy.sparse import csr_matrix

from graphstore.dsl.handlers._registry import handles
from graphstore.dsl.ast_nodes import (
    Batch,
    ConnectNode,
    CreateEdge,
    CreateNode,
    DeleteNode,
    DeleteNodes,
    ForgetNode,
    Increment,
    MergeStmt,
    UpdateNode,
    UpdateNodes,
    UpsertNode,
    VarAssign,
)
from graphstore.core.errors import BatchRollback, GraphStoreError, NodeNotFound
from graphstore.core.types import Result


class MutationHandlers:

    def _generate_auto_id(self, kind: str, data: dict) -> str:
        """Generate a deterministic content-hash ID from kind + sorted fields."""
        import hashlib
        parts = [f"kind={kind}"]
        for k in sorted(data.keys()):
            parts.append(f"{k}={data[k]}")
        content = "|".join(parts)
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def _handle_vector(self, slot: int, kind: str, data: dict, explicit_vector: list[float] | None):
        """Handle explicit VECTOR clause or auto-embed from schema EMBED field."""
        if explicit_vector is not None and self._vector_store is not None:
            vec = np.array(explicit_vector, dtype=np.float32)
            self._vector_store.add(slot, vec)
        elif explicit_vector is not None and self._vector_store is None:
            if hasattr(self, '_ensure_vector_store_cb') and self._ensure_vector_store_cb:
                vec = np.array(explicit_vector, dtype=np.float32)
                self._ensure_vector_store_cb(len(vec))
                self._vector_store.add(slot, vec)
        elif self._embedder and self._vector_store is not None:
            self._try_auto_embed(slot, kind, data)
        elif self._embedder and self._vector_store is None:
            kind_def = self.schema.describe_node_kind(kind)
            if kind_def and kind_def.get("embed_field"):
                embed_field = kind_def["embed_field"]
                text = data.get(embed_field)
                if text and isinstance(text, str):
                    if hasattr(self, '_ensure_vector_store_cb') and self._ensure_vector_store_cb:
                        self._ensure_vector_store_cb(self._embedder.dims)
                        vec = self._embedder.encode_documents([text])[0]
                        self._vector_store.add(slot, vec)

    def _try_auto_embed(self, slot: int, kind: str, data: dict):
        """Auto-embed if schema has EMBED field defined for this kind."""
        kind_def = self.schema.describe_node_kind(kind)
        if kind_def and kind_def.get("embed_field"):
            embed_field = kind_def["embed_field"]
            text = data.get(embed_field)
            if text and isinstance(text, str):
                vec = self._embedder.encode_documents([text])[0]
                self._vector_store.add(slot, vec)

    def _embed_and_store(self, slot: int, text: str) -> None:
        """Embed text and store vector at slot. Handles lazy vector store init."""
        if not self._embedder:
            return
        if self._vector_store is not None:
            vec = self._embedder.encode_documents([text])[0]
            self._vector_store.add(slot, vec)
        elif hasattr(self, '_ensure_vector_store_cb') and self._ensure_vector_store_cb:
            vec = self._embedder.encode_documents([text])[0]
            self._ensure_vector_store_cb(len(vec))
            self._vector_store.add(slot, vec)

    def _batch_embed_and_store(self, items: list[tuple[int, str]]) -> None:
        """Batch-embed multiple (slot, text) pairs in one model call."""
        if not self._embedder or not items:
            return
        slots, texts = zip(*items)
        if self._vector_store is None:
            if hasattr(self, '_ensure_vector_store_cb') and self._ensure_vector_store_cb:
                self._ensure_vector_store_cb(self._embedder.dims)
            else:
                return
        vecs = self._embedder.encode_documents(list(texts))
        for slot, vec in zip(slots, vecs):
            self._vector_store.add(slot, vec)

    def _collect_doc_children(self, doc_id: str) -> list[tuple[str, int]]:
        """Collect all (node_id, slot) pairs for a document's children.
        Works at slot level via raw edge lists to avoid materializing dicts."""
        doc_slot = self._resolve_slot(doc_id)
        if doc_slot is None:
            return []
        children = []
        child_kinds = {"has_chunk", "has_image", "has_section"}
        slot_to_id = self.store._slot_to_id
        edges_by_type = self.store._edges_by_type
        tombstones = self.store.node_tombstones

        for etype in child_kinds:
            for s, t, _d in edges_by_type.get(etype, []):
                if s == doc_slot and t not in tombstones:
                    tgt_id = slot_to_id(t)
                    if tgt_id is None:
                        continue
                    if etype == "has_section":
                        for se_etype in ("has_chunk",):
                            for ss, st, _sd in edges_by_type.get(se_etype, []):
                                if ss == t and st not in tombstones:
                                    st_id = slot_to_id(st)
                                    if st_id is not None:
                                        children.append((st_id, st))
                    children.append((tgt_id, t))
        return children

    @handles(CreateNode, write=True)
    def _create_node(self, q: CreateNode) -> Result:
        data = {fp.name: fp.value for fp in q.fields}
        kind = data.pop("kind", "default")
        self.schema.validate_node(kind, data)
        if q.auto_id:
            node_id = self._generate_auto_id(kind, data)
        else:
            node_id = q.id
        self.store.put_node(node_id, kind, data)
        self._apply_ttl(node_id, q.expires_in, q.expires_at)
        if self.store._active_context:
            str_id = self.store.string_table.intern(node_id)
            slot = self.store.id_to_slot[str_id]
            self.store.columns.set_reserved(slot, "__context__", self.store._active_context)
        str_id = self.store.string_table.intern(node_id)
        slot = self.store.id_to_slot[str_id]
        self._handle_vector(slot, kind, data, q.vector)
        if q.document and self._document_store:
            self._document_store.put_document(slot, q.document.encode("utf-8"), "text/plain")
        node = self.store.get_node(node_id)
        return Result(kind="node", data=node, count=1)

    @handles(UpdateNode, write=True)
    def _update_node(self, q: UpdateNode) -> Result:
        data = {fp.name: fp.value for fp in q.fields}
        self.store.update_node(q.id, data)

        # Auto re-embed if an embed field was updated
        if self._embedder and self._vector_store is not None:
            str_id = self.store.string_table.intern(q.id)
            slot = self.store.id_to_slot.get(str_id)
            if slot is not None:
                node_data = self.store.get_node(q.id)
                if node_data:
                    kind = node_data.get("kind", "default")
                    kind_def = self.schema.describe_node_kind(kind)
                    if kind_def and kind_def.get("embed_field"):
                        embed_field = kind_def["embed_field"]
                        if embed_field in data:
                            text = data[embed_field]
                            if text and isinstance(text, str):
                                vec = self._embedder.encode_documents([text])[0]
                                self._vector_store.add(slot, vec)

        node = self.store.get_node(q.id)
        return Result(kind="node", data=node, count=1)

    @handles(UpsertNode, write=True)
    def _upsert_node(self, q: UpsertNode) -> Result:
        data = {fp.name: fp.value for fp in q.fields}
        kind = data.pop("kind", "default")
        self.schema.validate_node(kind, data)
        self.store.upsert_node(q.id, kind, data)
        self._apply_ttl(q.id, q.expires_in, q.expires_at)
        str_id = self.store.string_table.intern(q.id)
        slot = self.store.id_to_slot[str_id]
        self._handle_vector(slot, kind, data, q.vector)
        node = self.store.get_node(q.id)
        return Result(kind="node", data=node, count=1)

    @handles(DeleteNode, write=True)
    def _delete_node(self, q: DeleteNode) -> Result:
        slot = self._resolve_slot(q.id)

        if slot is not None:
            kind_str_id = int(self.store.node_kinds[slot])
            kind_name = self.store.string_table.lookup(kind_str_id)
            if kind_name == "document":
                for child_id, child_slot in self._collect_doc_children(q.id):
                    if self._vector_store:
                        self._vector_store.remove(child_slot)
                    if self._document_store:
                        self._document_store.delete_document(child_slot)
                    try:
                        self.store.delete_node(child_id)
                    except NodeNotFound:
                        pass
                if self._document_store:
                    self._document_store.delete_all_for_doc(slot)

        if self._vector_store and slot is not None:
            self._vector_store.remove(slot)
        self.store.delete_node(q.id)
        return Result(kind="ok", data={"id": q.id}, count=1)

    @handles(DeleteNodes, write=True)
    def _delete_nodes(self, q: DeleteNodes) -> Result:
        kind_filter = self._extract_kind_from_where(q.where)
        remaining = self._strip_kind_from_expr(q.where.expr)

        ids_to_delete = None

        if remaining is not None:
            col_ids = self._try_column_delete_ids(remaining, kind_filter)
            if col_ids is not None:
                ids_to_delete = col_ids

        if ids_to_delete is None:
            if remaining is not None:
                raw_pred = self._make_raw_predicate(remaining)
            else:
                raw_pred = None

            if raw_pred is not None or remaining is None:
                ids_to_delete = self.store.query_node_ids(kind=kind_filter, predicate=raw_pred)
            else:
                nodes = self.store.get_all_nodes(kind=kind_filter)
                ids_to_delete = [n["id"] for n in nodes if self._eval_where(q.where.expr, n)]

        deleted_ids = []
        for nid in ids_to_delete:
            try:
                self.store.delete_node(nid)
                deleted_ids.append(nid)
            except NodeNotFound:
                pass
        return Result(kind="nodes", data=[{"id": i} for i in deleted_ids], count=len(deleted_ids))

    @handles(UpdateNodes, write=True)
    def _update_nodes(self, q: UpdateNodes) -> Result:
        """UPDATE NODES WHERE ... SET ...: bulk column update."""
        n = self.store._next_slot
        if n == 0:
            return Result(kind="ok", data={"updated": 0}, count=0)

        mask = self._compute_live_mask(n)

        kind_filter = self._extract_kind_from_where(q.where)
        if kind_filter:
            kind_mask = self.store._live_mask(kind_filter)
            mask = mask & kind_mask
            remaining = self._strip_kind_from_expr(q.where.expr)
            if remaining is not None:
                col_mask = self._try_column_filter(remaining, mask, n)
                if col_mask is not None:
                    mask = col_mask
                else:
                    fallback_mask = np.zeros(n, dtype=bool)
                    for slot_idx in np.nonzero(mask)[0]:
                        node = self.store._materialize_slot(int(slot_idx))
                        if node and self._eval_where(q.where.expr, node):
                            fallback_mask[int(slot_idx)] = True
                    mask = fallback_mask
        else:
            col_mask = self._try_column_filter(q.where.expr, mask, n)
            if col_mask is not None:
                mask = col_mask
            else:
                fallback_mask = np.zeros(n, dtype=bool)
                for slot_idx in np.nonzero(mask)[0]:
                    node = self.store._materialize_slot(int(slot_idx))
                    if node and self._eval_where(q.where.expr, node):
                        fallback_mask[int(slot_idx)] = True
                mask = fallback_mask

        update_data = {fp.name: fp.value for fp in q.fields}
        matching_slots = np.nonzero(mask)[0]
        now_ms = int(time.time() * 1000)

        if len(matching_slots) > 0:
            store = self.store
            for fp in q.fields:
                field = fp.name
                value = fp.value
                if store.columns.has_column(field):
                    dtype_str = store.columns._dtypes[field]
                    if dtype_str == "int32_interned" and isinstance(value, str):
                        raw_val = store.string_table.intern(value)
                        store.columns._columns[field][matching_slots] = raw_val
                        store.columns._presence[field][matching_slots] = True
                    elif dtype_str == "int64" and isinstance(value, int):
                        store.columns._columns[field][matching_slots] = int(value)
                        store.columns._presence[field][matching_slots] = True
                    elif dtype_str == "float64" and isinstance(value, (int, float)) and not isinstance(value, bool):
                        store.columns._columns[field][matching_slots] = float(value)
                        store.columns._presence[field][matching_slots] = True
                    else:
                        for slot_idx in matching_slots:
                            store.columns.set(int(slot_idx), {field: value})
                else:
                    for slot_idx in matching_slots:
                        store.columns.set(int(slot_idx), {field: value})

            if store.columns.has_column("__updated_at__"):
                store.columns._columns["__updated_at__"][matching_slots] = now_ms
                store.columns._presence["__updated_at__"][matching_slots] = True
            else:
                for slot_idx in matching_slots:
                    store.columns.set_reserved(int(slot_idx), "__updated_at__", now_ms)

        for fp in q.fields:
            if fp.name in self.store._indexed_fields:
                self.store.add_index(fp.name)

        updated = len(matching_slots)
        return Result(kind="ok", data={"updated": updated}, count=updated)

    @handles(Increment, write=True)
    def _increment(self, q: Increment) -> Result:
        self.store.increment_field(q.node_id, q.field, q.amount)
        return Result(kind="ok", data=None, count=0)

    @handles(MergeStmt, write=True)
    def _merge(self, q: MergeStmt) -> Result:
        """MERGE NODE src INTO tgt: copy fields, rewire edges, tombstone source."""
        if q.source_id not in self.store.string_table:
            raise NodeNotFound(q.source_id)
        src_str = self.store.string_table.intern(q.source_id)
        src_slot = self.store.id_to_slot.get(src_str)
        if src_slot is None or src_slot in self.store.node_tombstones:
            raise NodeNotFound(q.source_id)

        if q.target_id not in self.store.string_table:
            raise NodeNotFound(q.target_id)
        tgt_str = self.store.string_table.intern(q.target_id)
        tgt_slot = self.store.id_to_slot.get(tgt_str)
        if tgt_slot is None or tgt_slot in self.store.node_tombstones:
            raise NodeNotFound(q.target_id)

        fields_merged = 0
        for field in list(self.store.columns._columns.keys()):
            if field.startswith("__") and field.endswith("__"):
                continue
            if not self.store.columns._presence[field][src_slot]:
                continue
            if self.store.columns._presence[field][tgt_slot]:
                continue
            dtype = self.store.columns._dtypes[field]
            raw = self.store.columns._columns[field][src_slot]
            self.store.columns._columns[field][tgt_slot] = raw
            self.store.columns._presence[field][tgt_slot] = True
            fields_merged += 1

        edges_rewired = 0
        for etype in list(self.store._edges_by_type.keys()):
            new_edges = []
            for s, t, d in self.store._edges_by_type[etype]:
                if s == src_slot:
                    new_edges.append((tgt_slot, t, d))
                    edges_rewired += 1
                elif t == src_slot:
                    new_edges.append((s, tgt_slot, d))
                    edges_rewired += 1
                else:
                    new_edges.append((s, t, d))
            self.store._edges_by_type[etype] = new_edges

        for etype in list(self.store._edges_by_type.keys()):
            seen = set()
            deduped = []
            for s, t, d in self.store._edges_by_type[etype]:
                key = (s, t)
                if key not in seen:
                    seen.add(key)
                    deduped.append((s, t, d))
            self.store._edges_by_type[etype] = deduped
            if not deduped:
                del self.store._edges_by_type[etype]

        self.store._edge_keys = {
            (s, t, k)
            for k, edges in self.store._edges_by_type.items()
            for s, t, _d in edges
        }
        self.store._edges_dirty = True
        self.store._ensure_edges_built()

        self.store.delete_node(q.source_id)

        now_ms = int(time.time() * 1000)
        self.store.columns.set_reserved(tgt_slot, "__updated_at__", now_ms)

        return Result(
            kind="ok",
            data={
                "merged_into": q.target_id,
                "fields_merged": fields_merged,
                "edges_rewired": edges_rewired,
            },
            count=1,
        )

    @handles(ForgetNode, write=True)
    def _forget(self, q: ForgetNode) -> Result:
        """FORGET NODE: hard delete blob + vector + memory (irreversible)."""
        slot = self._resolve_slot(q.id)
        if slot is None:
            raise NodeNotFound(q.id)

        kind_str_id = int(self.store.node_kinds[slot])
        kind_name = self.store.string_table.lookup(kind_str_id)
        if kind_name == "document":
            for child_id, child_slot in self._collect_doc_children(q.id):
                if self._vector_store:
                    self._vector_store.remove(child_slot)
                if self._document_store:
                    self._document_store.delete_document(child_slot)
                try:
                    self.store.delete_node(child_id)
                except NodeNotFound:
                    pass
            if self._document_store:
                self._document_store.delete_all_for_doc(slot)

        if self._vector_store:
            self._vector_store.remove(slot)
        if self._document_store:
            self._document_store.delete_document(slot)
        self.store.delete_node(q.id)
        return Result(kind="ok", data={"forgotten": q.id}, count=1)

    @handles(Batch, write=True)
    def _batch(self, q: Batch) -> Result:
        """Execute batch with rollback on failure.

        Saves a copy of the store state before executing. If any statement
        fails, restores from the copy.
        """
        saved_edges = {k: list(v) for k, v in self.store._edges_by_type.items()}
        saved_edge_keys = set(self.store._edge_keys)
        saved_columns = self.store.columns.snapshot_arrays()
        saved_tombstones = set(self.store.node_tombstones)
        saved_id_to_slot = dict(self.store.id_to_slot)
        saved_count = self.store._count
        saved_next_slot = self.store._next_slot
        saved_node_ids = self.store.node_ids[: self.store._next_slot].copy()
        saved_node_kinds = self.store.node_kinds[: self.store._next_slot].copy()

        try:
            variables: dict[str, str] = {}
            for stmt in q.statements:
                if isinstance(stmt, VarAssign):
                    result = self._dispatch(stmt.statement)
                    if result.data and isinstance(result.data, dict) and "id" in result.data:
                        variables[stmt.variable] = result.data["id"]
                    else:
                        raise GraphStoreError(
                            f"Variable {stmt.variable}: statement did not return an ID"
                        )
                else:
                    if isinstance(stmt, CreateEdge):
                        src = variables.get(stmt.source, stmt.source) if stmt.source.startswith("$") else stmt.source
                        tgt = variables.get(stmt.target, stmt.target) if stmt.target.startswith("$") else stmt.target
                        if src.startswith("$"):
                            raise GraphStoreError(f"Unresolved variable: {src}")
                        if tgt.startswith("$"):
                            raise GraphStoreError(f"Unresolved variable: {tgt}")
                        resolved = CreateEdge(source=src, target=tgt, fields=stmt.fields)
                        self._dispatch(resolved)
                    else:
                        self._dispatch(stmt)
            self.store._ensure_edges_built()
            return Result(kind="ok", data=None, count=0)
        except Exception as e:
            self.store._edges_by_type = saved_edges
            self.store._edge_keys = saved_edge_keys
            self.store.node_tombstones = saved_tombstones
            self.store.id_to_slot = saved_id_to_slot
            self.store._count = saved_count
            self.store._next_slot = saved_next_slot
            self.store.node_ids[:saved_next_slot] = saved_node_ids
            self.store.node_kinds[:saved_next_slot] = saved_node_kinds
            self.store._rebuild_edges()
            self.store.columns.restore_arrays(saved_columns)
            raise BatchRollback(
                failed_statement=str(type(e).__name__), error=str(e)
            )
