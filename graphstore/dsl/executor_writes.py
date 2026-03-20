"""Write query handlers for the DSL executor."""

import time
from collections import deque

import numpy as np
from scipy.sparse import csr_matrix

from graphstore.dsl.ast_nodes import (
    AssertStmt,
    Batch,
    BindContext,
    CreateEdge,
    CreateNode,
    DeleteEdge,
    DeleteEdges,
    DeleteNode,
    DeleteNodes,
    DiscardContext,
    Increment,
    IngestStmt,
    MergeStmt,
    PropagateStmt,
    RetractStmt,
    UpdateEdge,
    UpdateNode,
    UpdateNodes,
    UpsertNode,
    VarAssign,
)
from graphstore.core.errors import (
    BatchRollback,
    GraphStoreError,
    NodeNotFound,
)
from graphstore.core.types import Result
from graphstore.dsl.executor_base import ExecutorBase


class WriteExecutor(ExecutorBase):

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
            # Need to lazily init vector store via GraphStore callback
            if hasattr(self, '_ensure_vector_store_cb') and self._ensure_vector_store_cb:
                vec = np.array(explicit_vector, dtype=np.float32)
                self._ensure_vector_store_cb(len(vec))
                self._vector_store.add(slot, vec)
        elif self._embedder and self._vector_store is not None:
            self._try_auto_embed(slot, kind, data)
        elif self._embedder and self._vector_store is None:
            # Check if auto-embed needed, lazily init vector store
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

    def _create_node(self, q: CreateNode) -> Result:
        data = {fp.name: fp.value for fp in q.fields}
        kind = data.pop("kind", "default")
        self.schema.validate_node(kind, data)
        if q.auto_id:
            node_id = self._generate_auto_id(kind, data)
        else:
            node_id = q.id
        self.store.put_node(node_id, kind, data)
        # Handle TTL
        self._apply_ttl(node_id, q.expires_in, q.expires_at)
        # Auto-tag context if bound
        if self.store._active_context:
            str_id = self.store.string_table.intern(node_id)
            slot = self.store.id_to_slot[str_id]
            self.store.columns.set_reserved(slot, "__context__", self.store._active_context)
        # Handle vector: explicit VECTOR clause or auto-embed
        str_id = self.store.string_table.intern(node_id)
        slot = self.store.id_to_slot[str_id]
        self._handle_vector(slot, kind, data, q.vector)
        # Handle DOCUMENT clause
        if q.document and self._document_store:
            self._document_store.put_document(slot, q.document.encode("utf-8"), "text/plain")
        node = self.store.get_node(node_id)
        return Result(kind="node", data=node, count=1)

    def _update_node(self, q: UpdateNode) -> Result:
        data = {fp.name: fp.value for fp in q.fields}
        self.store.update_node(q.id, data)
        node = self.store.get_node(q.id)
        return Result(kind="node", data=node, count=1)

    def _upsert_node(self, q: UpsertNode) -> Result:
        data = {fp.name: fp.value for fp in q.fields}
        kind = data.pop("kind", "default")
        self.schema.validate_node(kind, data)
        self.store.upsert_node(q.id, kind, data)
        # Handle TTL
        self._apply_ttl(q.id, q.expires_in, q.expires_at)
        # Handle vector: explicit VECTOR clause or auto-embed
        str_id = self.store.string_table.intern(q.id)
        slot = self.store.id_to_slot[str_id]
        self._handle_vector(slot, kind, data, q.vector)
        node = self.store.get_node(q.id)
        return Result(kind="node", data=node, count=1)

    def _delete_node(self, q: DeleteNode) -> Result:
        # Remove vector before deleting the node (need slot while node still exists)
        if self._vector_store:
            slot = self._resolve_slot(q.id)
            if slot is not None:
                self._vector_store.remove(slot)
        self.store.delete_node(q.id)
        return Result(kind="ok", data={"id": q.id}, count=1)

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

    def _create_edge(self, q: CreateEdge) -> Result:
        data = {fp.name: fp.value for fp in q.fields}
        kind = data.pop("kind", "default")
        # Validate edge endpoint kinds if schema is registered
        src_node = self.store.get_node(q.source)
        tgt_node = self.store.get_node(q.target)
        if src_node and tgt_node:
            self.schema.validate_edge(kind, src_node["kind"], tgt_node["kind"])
        self.store.put_edge(q.source, q.target, kind, data if data else None)
        return Result(kind="edges", data=[{"source": q.source, "target": q.target, "kind": kind}], count=1)

    def _update_edge(self, q: UpdateEdge) -> Result:
        kind = self._extract_kind_from_where(q.where)
        update_data = {fp.name: fp.value for fp in q.fields}
        # Find and update matching edges
        updated = 0
        for etype in ([kind] if kind else list(self.store._edges_by_type.keys())):
            if etype not in self.store._edges_by_type:
                continue
            src_str_id = self.store.string_table.intern(q.source)
            tgt_str_id = self.store.string_table.intern(q.target)
            src_slot = self.store.id_to_slot.get(src_str_id)
            tgt_slot = self.store.id_to_slot.get(tgt_str_id)
            if src_slot is None or tgt_slot is None:
                continue
            for i, (s, t, d) in enumerate(self.store._edges_by_type[etype]):
                if s == src_slot and t == tgt_slot:
                    self.store._edges_by_type[etype][i] = (s, t, {**d, **update_data})
                    updated += 1
        if updated > 0:
            self.store._edges_dirty = True
            self.store._ensure_edges_built()
        return Result(kind="ok", data={"source": q.source, "target": q.target, "updated": updated}, count=updated)

    def _delete_edge(self, q: DeleteEdge) -> Result:
        kind = self._extract_kind_from_where(q.where)
        if kind:
            self.store.delete_edge(q.source, q.target, kind)
        else:
            for etype in list(self.store._edges_by_type.keys()):
                self.store.delete_edge(q.source, q.target, etype)
        return Result(kind="ok", data=None, count=1)

    def _delete_edges(self, q: DeleteEdges) -> Result:
        kind = self._extract_kind_from_where(q.where)
        if q.direction == "FROM":
            edges = self.store.get_edges_from(q.node_id, kind=kind)
        else:
            edges = self.store.get_edges_to(q.node_id, kind=kind)

        for e in edges:
            self.store.delete_edge(e["source"], e["target"], e["kind"])
        return Result(kind="edges", data=edges, count=len(edges))

    def _increment(self, q: Increment) -> Result:
        self.store.increment_field(q.node_id, q.field, q.amount)
        return Result(kind="ok", data=None, count=0)

    def _assert(self, q: AssertStmt) -> Result:
        """ASSERT: upsert with reserved __confidence__, __source__, __retracted__=0."""
        data = {fp.name: fp.value for fp in q.fields}
        kind = data.pop("kind", "default")
        self.schema.validate_node(kind, data)
        self.store.upsert_node(q.id, kind, data)
        # Set reserved belief fields
        str_id = self.store.string_table.intern(q.id)
        slot = self.store.id_to_slot[str_id]
        if q.confidence is not None:
            self.store.columns.set_reserved(slot, "__confidence__", q.confidence)
        if q.source is not None:
            self.store.columns.set_reserved(slot, "__source__", q.source)
        self.store.columns.set_reserved(slot, "__retracted__", 0)
        node = self.store.get_node(q.id)
        return Result(kind="node", data=node, count=1)

    def _retract(self, q: RetractStmt) -> Result:
        """RETRACT: mark node as retracted (invisible via live_mask)."""
        if q.id not in self.store.string_table:
            raise NodeNotFound(q.id)
        str_id = self.store.string_table.intern(q.id)
        slot = self.store.id_to_slot.get(str_id)
        if slot is None or slot in self.store.node_tombstones:
            raise NodeNotFound(q.id)
        now_ms = int(time.time() * 1000)
        self.store.columns.set_reserved(slot, "__retracted__", 1)
        self.store.columns.set_reserved(slot, "__retracted_at__", now_ms)
        if q.reason:
            self.store.columns.set_reserved(slot, "__retract_reason__", q.reason)
        return Result(kind="ok", data={"id": q.id}, count=1)

    def _update_nodes(self, q: UpdateNodes) -> Result:
        """UPDATE NODES WHERE ... SET ...: bulk column update."""
        n = self.store._next_slot
        if n == 0:
            return Result(kind="ok", data={"updated": 0}, count=0)

        mask = self._compute_live_mask(n)

        # Apply WHERE filter
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
                    # Fallback: materialize and filter
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
            # Bulk numpy assignment for columnar fields
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
                        # Type mismatch - fall back to per-slot set
                        for slot_idx in matching_slots:
                            store.columns.set(int(slot_idx), {field: value})
                else:
                    # Column doesn't exist yet - use set() which auto-creates
                    for slot_idx in matching_slots:
                        store.columns.set(int(slot_idx), {field: value})

            # Bulk set __updated_at__ via numpy
            if store.columns.has_column("__updated_at__"):
                store.columns._columns["__updated_at__"][matching_slots] = now_ms
                store.columns._presence["__updated_at__"][matching_slots] = True
            else:
                for slot_idx in matching_slots:
                    store.columns.set_reserved(int(slot_idx), "__updated_at__", now_ms)

        # Rebuild secondary indices for updated fields
        for fp in q.fields:
            if fp.name in self.store._indexed_fields:
                self.store.add_index(fp.name)  # rebuild index for this field

        updated = len(matching_slots)
        return Result(kind="ok", data={"updated": updated}, count=updated)

    def _merge(self, q: MergeStmt) -> Result:
        """MERGE NODE src INTO tgt: copy fields, rewire edges, tombstone source."""
        # Validate source and target exist
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

        # 1. Copy source column values to target where target has no value
        fields_merged = 0
        for field in list(self.store.columns._columns.keys()):
            if field.startswith("__") and field.endswith("__"):
                continue  # skip reserved columns
            if not self.store.columns._presence[field][src_slot]:
                continue  # source has no value
            if self.store.columns._presence[field][tgt_slot]:
                continue  # target wins on conflict
            # Copy source value to target
            dtype = self.store.columns._dtypes[field]
            raw = self.store.columns._columns[field][src_slot]
            self.store.columns._columns[field][tgt_slot] = raw
            self.store.columns._presence[field][tgt_slot] = True
            fields_merged += 1

        # 2. Re-wire all edges from source to target
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

        # 3. Drop duplicate edges after re-wiring
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

        # 4. Rebuild edge keys
        self.store._edge_keys = {
            (s, t, k)
            for k, edges in self.store._edges_by_type.items()
            for s, t, _d in edges
        }
        self.store._edges_dirty = True
        self.store._ensure_edges_built()

        # 5. Tombstone source
        self.store.delete_node(q.source_id)

        # 6. Update target's __updated_at__
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

    def _propagate(self, q: PropagateStmt) -> Result:
        """PROPAGATE: BFS forward belief chaining."""
        src_slot = self._resolve_slot(q.node_id)
        if src_slot is None:
            raise NodeNotFound(q.node_id)

        n = self.store._next_slot
        field = q.field

        # Get source field value
        if not self.store.columns.has_column(field):
            return Result(kind="ok", data={"updated": 0}, count=0)
        if not self.store.columns._presence[field][src_slot]:
            return Result(kind="ok", data={"updated": 0}, count=0)

        # BFS from source
        combined = self.store.edge_matrices.get(None)
        if combined is None:
            return Result(kind="ok", data={"updated": 0}, count=0)

        mat = combined
        if mat.shape[0] < n:
            mat = csr_matrix((mat.data, mat.indices, mat.indptr), shape=(n, n))

        visited = set()
        visited.add(src_slot)
        frontier = deque()

        # Get source value
        dtype = self.store.columns._dtypes[field]
        raw = self.store.columns._columns[field][src_slot]
        if dtype == "float64":
            source_value = float(raw)
        elif dtype == "int64":
            source_value = int(raw)
        else:
            return Result(kind="ok", data={"updated": 0}, count=0)

        # Push initial neighbors with (slot, parent_value)
        frontier.append((src_slot, source_value, 0))
        updated_count = 0
        now_ms = int(time.time() * 1000)

        while frontier:
            current_slot, parent_value, depth = frontier.popleft()
            if depth >= q.depth:
                continue

            # Get outgoing neighbors from CSR
            start = mat.indptr[current_slot]
            end = mat.indptr[current_slot + 1]
            neighbors = mat.indices[start:end]
            weights = mat.data[start:end]

            for i, nb in enumerate(neighbors):
                nb = int(nb)
                if nb in visited:
                    continue
                visited.add(nb)

                if nb in self.store.node_tombstones:
                    continue

                # Get current value or default
                if self.store.columns._presence[field][nb]:
                    if dtype == "float64":
                        current_val = float(self.store.columns._columns[field][nb])
                    else:
                        current_val = int(self.store.columns._columns[field][nb])
                else:
                    current_val = 0.0

                # Propagated value = parent * edge_weight
                edge_weight = float(weights[i]) if i < len(weights) else 1.0
                propagated = parent_value * edge_weight

                # Update the field
                self.store.columns.set(nb, {field: propagated})
                self.store.columns.set_reserved(nb, "__updated_at__", now_ms)
                updated_count += 1

                frontier.append((nb, propagated, depth + 1))

        return Result(kind="ok", data={"updated": updated_count}, count=updated_count)

    def _bind_context(self, q: BindContext) -> Result:
        """BIND CONTEXT: set active context on store."""
        self.store._active_context = q.name
        return Result(kind="ok", data={"context": q.name}, count=0)

    def _discard_context(self, q: DiscardContext) -> Result:
        """DISCARD CONTEXT: delete all nodes with matching __context__ and unbind."""
        deleted_count = 0
        n = self.store._next_slot
        if n > 0 and self.store.columns.has_column("__context__"):
            ctx_col = self.store.columns.get_column("__context__", n)
            if ctx_col is not None:
                col_data, col_pres, _ = ctx_col
                ctx_id = self.store.string_table.intern(q.name)
                ctx_mask = col_pres & (col_data == ctx_id)
                slots_to_delete = np.nonzero(ctx_mask)[0]
                for slot in slots_to_delete:
                    nid = self.store._slot_to_id(int(slot))
                    if nid:
                        try:
                            self.store.delete_node(nid)
                            deleted_count += 1
                        except NodeNotFound:
                            pass

        # Unbind context
        self.store._active_context = None
        return Result(kind="ok", data={"discarded": q.name, "deleted": deleted_count}, count=deleted_count)

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
            variables: dict[str, str] = {}  # $var -> resolved node ID
            for stmt in q.statements:
                if isinstance(stmt, VarAssign):
                    result = self._dispatch(stmt.statement)
                    # Extract the generated ID from the result
                    if result.data and isinstance(result.data, dict) and "id" in result.data:
                        variables[stmt.variable] = result.data["id"]
                    else:
                        raise GraphStoreError(
                            f"Variable {stmt.variable}: statement did not return an ID"
                        )
                else:
                    # Resolve $variables in CreateEdge source/target
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
            # Flush any deferred edge rebuilds after successful batch
            self.store._ensure_edges_built()
            return Result(kind="ok", data=None, count=0)
        except Exception as e:
            # Rollback
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

    def _ingest(self, q: IngestStmt) -> Result:
        """INGEST: parse file, chunk, create graph nodes + edges, store documents."""
        import hashlib
        import os as _os

        from graphstore.ingest.router import ingest_file
        from graphstore.ingest.chunker import chunk_by_heading

        if not _os.path.exists(q.file_path):
            raise GraphStoreError(f"File not found: {q.file_path}")

        # 1. Parse file
        result = ingest_file(q.file_path, using=q.using)

        # 2. Chunk
        chunks = chunk_by_heading(result.markdown)

        # 3. Parent node ID
        parent_id = q.node_id
        if not parent_id:
            h = hashlib.sha256(q.file_path.encode()).hexdigest()[:12]
            parent_id = f"doc:{h}"

        parent_kind = q.kind or "document"

        # Check duplicate: if parent_id already exists, raise
        existing = self.store.get_node(parent_id)
        if existing is not None:
            raise GraphStoreError(f"Node already exists: {parent_id}")

        # 4. Create parent node with metadata
        metadata_fields = {
            "source": q.file_path,
            "parser": result.parser_used,
            "confidence": result.confidence,
        }
        metadata_fields.update({
            k: v for k, v in result.metadata.items()
            if isinstance(v, (str, int, float)) and k not in ("source",)
        })

        # Use store.put_node directly to avoid recursion
        parent_slot = self.store.put_node(parent_id, parent_kind, metadata_fields)

        # Store full markdown in DocumentStore
        if self._document_store:
            self._document_store.put_document(
                parent_slot, result.markdown.encode("utf-8"), "text/markdown")
            self._document_store.put_metadata(parent_slot, {
                "source_path": q.file_path,
                "pages": result.metadata.get("pages"),
                "author": result.metadata.get("author"),
                "title": result.metadata.get("title"),
                "parser_used": result.parser_used,
                "confidence": result.confidence,
                "ingested_at": int(time.time() * 1000),
            })

        # 5. Create chunk nodes
        chunk_ids = []
        for chunk in chunks:
            chunk_id = f"{parent_id}:chunk:{chunk.index}"
            chunk_fields = {"summary": chunk.summary}
            if chunk.heading:
                chunk_fields["heading"] = chunk.heading
            if chunk.page is not None:
                chunk_fields["page"] = chunk.page

            chunk_slot = self.store.put_node(chunk_id, "chunk", chunk_fields)
            chunk_ids.append(chunk_id)

            # Store full chunk text in DocumentStore
            if self._document_store:
                self._document_store.put_document(
                    chunk_slot, chunk.text.encode("utf-8"), "text/markdown")
                self._document_store.put_summary(
                    chunk_slot, chunk.summary, chunk.heading,
                    chunk.page, chunk.index, parent_slot)

            # Auto-embed summary
            if self._embedder and self._vector_store is not None:
                vec = self._embedder.encode_documents([chunk.summary])[0]
                self._vector_store.add(chunk_slot, vec)
            elif self._embedder and self._vector_store is None:
                if hasattr(self, '_ensure_vector_store_cb') and self._ensure_vector_store_cb:
                    vec = self._embedder.encode_documents([chunk.summary])[0]
                    self._ensure_vector_store_cb(len(vec))
                    self._vector_store.add(chunk_slot, vec)

            # Edge: parent -> chunk
            self.store.put_edge(parent_id, chunk_id, "has_chunk")

        # 6. Handle images
        image_count = 0
        for i, img in enumerate(result.images):
            img_id = f"{parent_id}:image:{i}"
            img_fields = {}
            if img.page is not None:
                img_fields["page"] = img.page
            if img.description:
                img_fields["description"] = img.description

            img_slot = self.store.put_node(img_id, "image", img_fields)

            if self._document_store:
                self._document_store.put_image(
                    img_slot, img.data, img.mime_type,
                    img.page, img.description)

            self.store.put_edge(parent_id, img_id, "has_image")
            image_count += 1

        return Result(kind="ok", data={
            "doc_id": parent_id,
            "chunks": len(chunks),
            "images": image_count,
            "parser": result.parser_used,
            "confidence": result.confidence,
        }, count=len(chunks))
