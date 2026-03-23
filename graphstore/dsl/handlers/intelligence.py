"""Intelligence handlers: RECALL, SIMILAR, LEXICAL SEARCH, WHAT IF."""

import time
from collections import deque

import numpy as np
from scipy.sparse import csr_matrix

from graphstore.dsl.handlers._registry import handles
from graphstore.dsl.ast_nodes import (
    RecallQuery, SimilarQuery, LexicalSearchQuery, CounterfactualQuery,
)
from graphstore.core.types import Result
from graphstore.core.errors import EmbedderRequired, NodeNotFound, VectorError, VectorNotFound


class IntelligenceHandlers:

    @handles(RecallQuery)
    def _recall(self, q: RecallQuery) -> Result:
        """RECALL: spreading activation from a cue node."""
        cue_slot = self._resolve_slot(q.node_id)
        if cue_slot is None:
            raise NodeNotFound(q.node_id)

        n = self.store._next_slot
        if n == 0:
            return Result(kind="nodes", data=[], count=0)

        combined = self.store.edge_matrices.get(None)
        if combined is None:
            return Result(kind="nodes", data=[], count=0)

        mat = combined
        if mat.shape[0] < n:
            mat = csr_matrix((mat.data, mat.indices, mat.indptr), shape=(n, n))

        activation = np.zeros(n, dtype=np.float64)
        activation[cue_slot] = 1.0

        importance = np.ones(n, dtype=np.float64)
        imp_col = self.store.columns.get_column("importance", n)
        if imp_col is not None:
            col_data, col_pres, _ = imp_col
            importance[col_pres] = col_data[col_pres].astype(np.float64)
        conf_col = self.store.columns.get_column("__confidence__", n)
        if conf_col is not None:
            col_data, col_pres, _ = conf_col
            mask = col_pres
            importance[mask] *= col_data[mask].astype(np.float64)

        recency = np.ones(n, dtype=np.float64)
        updated_col = self.store.columns.get_column("__updated_at__", n)
        if updated_col is not None:
            col_data, col_pres, _ = updated_col
            now_ms = int(time.time() * 1000)
            age_days = np.where(col_pres, (now_ms - col_data.astype(np.float64)) / 86400000.0, 0.0)
            recency = np.where(col_pres, 1.0 / (1.0 + age_days), 1.0)

        live_mask = self._compute_live_mask(n)

        mat_t = mat.T.tocsr()
        for _ in range(q.depth):
            activation = mat_t.dot(activation)
            activation *= importance[:len(activation)]
            activation *= recency[:len(activation)]
            activation[:n] *= live_mask.astype(np.float64)

        activation[cue_slot] = 0.0

        if q.where:
            kind_filter = self._extract_kind_from_where(q.where)
            if kind_filter:
                kind_mask = self.store._live_mask(kind_filter)
                activation[:n] *= kind_mask

        active_indices = np.nonzero(activation > 0)[0]
        if len(active_indices) == 0:
            return Result(kind="nodes", data=[], count=0)

        limit = q.limit.value if q.limit else len(active_indices)
        k = min(limit, len(active_indices))

        if k < len(active_indices):
            top_k_idx = np.argpartition(-activation[active_indices], k)[:k]
        else:
            top_k_idx = np.arange(len(active_indices))

        top_k_slots = active_indices[top_k_idx]
        sorted_order = np.argsort(-activation[top_k_slots])
        top_k_slots = top_k_slots[sorted_order]

        results = []
        for slot in top_k_slots:
            slot = int(slot)
            node = self.store._materialize_slot(slot)
            if node is not None:
                if q.where and not self._is_simple_kind_filter(q.where):
                    remaining = self._strip_kind_from_expr(q.where.expr)
                    if remaining is not None:
                        if not self._eval_where(remaining, node):
                            continue
                node["_activation_score"] = float(activation[slot])
                results.append(node)

        return Result(kind="nodes", data=results, count=len(results))

    @handles(SimilarQuery)
    def _similar(self, q: SimilarQuery) -> Result:
        """SIMILAR TO: vector similarity search."""
        if hasattr(self, '_embedder_dirty') and self._embedder_dirty:
            from graphstore.core.errors import GraphStoreError
            raise GraphStoreError("Embedder changed. Run SYS REEMBED to update vectors.")

        if q.target_vector is not None:
            query_vec = np.array(q.target_vector, dtype=np.float32)
        elif q.target_text is not None:
            if not self._embedder:
                raise EmbedderRequired("Text similarity requires an embedder")
            query_vec = self._embedder.encode_queries([q.target_text])[0]
        elif q.target_node_id is not None:
            slot = self._resolve_slot(q.target_node_id)
            if slot is None:
                raise NodeNotFound(q.target_node_id)
            if not self._vector_store or not self._vector_store.has_vector(slot):
                raise VectorNotFound(f"Node '{q.target_node_id}' has no vector")
            query_vec = self._vector_store.get_vector(slot)
        else:
            raise VectorError("SIMILAR TO requires a vector, text, or node target")

        if not self._vector_store or self._vector_store.count() == 0:
            return Result(kind="nodes", data=[], count=0)

        n = self.store._next_slot
        if n == 0:
            return Result(kind="nodes", data=[], count=0)

        mask = self._compute_live_mask(n)
        vs_cap = self._vector_store._capacity
        if n <= vs_cap:
            vs_mask = self._vector_store._has_vector[:n]
        else:
            vs_mask = np.zeros(n, dtype=bool)
            vs_mask[:vs_cap] = self._vector_store._has_vector[:vs_cap]
        combined_mask = mask & vs_mask

        k = q.limit.value if q.limit else 10
        search_k = k * 3 if q.where else k
        slots, dists = self._vector_store.search(query_vec, k=search_k, mask=combined_mask)

        results = []
        target_k = q.limit.value if q.limit else 10
        for slot_idx, dist in zip(slots, dists):
            slot = int(slot_idx)
            node = self.store._materialize_slot(slot)
            if node is None:
                continue
            if q.where and not self._eval_where(q.where.expr, node):
                continue
            node["_similarity_score"] = round(1.0 - float(dist), 4)
            results.append(node)
            if len(results) >= target_k:
                break

        return Result(kind="nodes", data=results, count=len(results))

    @handles(LexicalSearchQuery)
    def _lexical_search(self, q: LexicalSearchQuery) -> Result:
        """LEXICAL SEARCH: BM25 full-text search over document summaries."""
        if not self._document_store:
            return Result(kind="nodes", data=[], count=0)

        target_k = q.limit.value if q.limit else 10
        hits = self._document_store.search_text(q.query, limit=target_k * 3)

        results = []
        for slot, score in hits:
            if not self._is_slot_visible(slot):
                continue
            node = self.store._materialize_slot(slot)
            if node is None:
                continue
            if q.where and not self._eval_where(q.where.expr, node):
                continue
            node["_bm25_score"] = round(score, 4)
            results.append(node)
            if len(results) >= target_k:
                break

        return Result(kind="nodes", data=results, count=len(results))

    @handles(CounterfactualQuery)
    def _counterfactual(self, q: CounterfactualQuery) -> Result:
        """WHAT IF RETRACT: simulate retraction without committing."""
        src_slot = self._resolve_slot(q.node_id)
        if src_slot is None:
            raise NodeNotFound(q.node_id)

        saved_columns = self.store.columns.snapshot_arrays()
        saved_tombstones = set(self.store.node_tombstones)
        saved_edges = {k: list(v) for k, v in self.store._edges_by_type.items()}
        saved_edge_keys = set(self.store._edge_keys)
        saved_id_to_slot = dict(self.store.id_to_slot)
        saved_count = self.store._count
        saved_next_slot = self.store._next_slot
        saved_node_ids = self.store.node_ids[:self.store._next_slot].copy()
        saved_node_kinds = self.store.node_kinds[:self.store._next_slot].copy()

        try:
            self.store.columns.set_reserved(src_slot, "__retracted__", 1)

            combined = self.store.edge_matrices.get(None)
            n = self.store._next_slot
            affected_slots = set()
            affected_slots.add(src_slot)

            if combined is not None:
                mat = combined
                if mat.shape[0] < n:
                    mat = csr_matrix((mat.data, mat.indices, mat.indptr), shape=(n, n))

                frontier = deque([src_slot])
                visited = {src_slot}
                while frontier:
                    current = frontier.popleft()
                    if current < mat.shape[0]:
                        start = mat.indptr[current]
                        end = mat.indptr[current + 1]
                        for nb in mat.indices[start:end]:
                            nb = int(nb)
                            if nb not in visited:
                                visited.add(nb)
                                affected_slots.add(nb)
                                frontier.append(nb)

            affected_nodes = []
            for slot in affected_slots:
                node = self.store._materialize_slot(int(slot))
                if node is not None:
                    affected_nodes.append(node)

            return Result(
                kind="counterfactual",
                data={
                    "retracted": q.node_id,
                    "affected_nodes": affected_nodes,
                    "affected_count": len(affected_nodes),
                },
                count=len(affected_nodes),
            )
        finally:
            self.store.columns.restore_arrays(saved_columns)
            self.store.node_tombstones = saved_tombstones
            self.store._edges_by_type = saved_edges
            self.store._edge_keys = saved_edge_keys
            self.store.id_to_slot = saved_id_to_slot
            self.store._count = saved_count
            self.store._next_slot = saved_next_slot
            self.store.node_ids[:saved_next_slot] = saved_node_ids
            self.store.node_kinds[:saved_next_slot] = saved_node_kinds
            self.store._rebuild_edges()
