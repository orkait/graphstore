"""Intelligence handlers: RECALL, SIMILAR, LEXICAL SEARCH, WHAT IF."""

import time
from collections import deque

import numpy as np
from scipy.sparse import csr_matrix

from graphstore.core.edges import resize_csr

from graphstore.dsl.handlers._registry import handles
from graphstore.dsl.ast_nodes import (
    RecallQuery, SimilarQuery, LexicalSearchQuery, CounterfactualQuery, RememberQuery,
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

        if combined.shape[0] < n:
            # Rare path: matrix predates latest node additions, resize then transpose.
            # Intentionally NOT cached — resized matrix differs from _combined_all.
            mat_t = resize_csr(combined, n).T.tocsr()
        else:
            mat_t = self.store.edge_matrices.get_combined_transpose()
        decay = getattr(self, '_recall_decay', 0.7)
        for _ in range(q.depth):
            spread = mat_t.dot(activation) * decay
            activation = activation + spread
            activation[:n] *= live_mask.astype(np.float64)
        activation *= importance[:len(activation)]
        activation *= recency[:len(activation)]

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

        conf_col = self.store.columns.get_column("__confidence__", n)
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
            if conf_col is not None:
                col_data, col_pres, _ = conf_col
                if col_pres[slot]:
                    node["_confidence"] = round(float(col_data[slot]), 4)
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
                    mat = resize_csr(mat, n)

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
            self.store._invalidate_live_cache()
            self.store._tombstone_mask_cache = None

    @handles(RememberQuery)
    def _remember(self, q: RememberQuery) -> Result:
        """REMEMBER: hybrid retrieval fusing vector + BM25 + recency + confidence."""
        import math

        target_k = q.limit.value if q.limit else 10
        oversample_factor = getattr(self, '_search_oversample', 5)
        oversample = target_k * oversample_factor

        candidates: dict[int, dict] = {}

        n = self.store._next_slot
        if n == 0:
            return Result(kind="nodes", data=[], count=0)

        live_mask = self._compute_live_mask(n)
        now_ms = int(time.time() * 1000)

        # --- Vector candidates ---
        if self._embedder and self._vector_store and self._vector_store.count() > 0:
            query_vec = self._embedder.encode_queries([q.query])[0]
            vs_cap = self._vector_store._capacity
            vs_mask = np.zeros(n, dtype=bool)
            cap = min(n, vs_cap)
            vs_mask[:cap] = self._vector_store._has_vector[:cap]
            combined = live_mask & vs_mask

            slots, dists = self._vector_store.search(query_vec, k=oversample, mask=combined, oversample_factor=oversample_factor)
            for slot_idx, dist in zip(slots, dists):
                slot = int(slot_idx)
                sim = max(0.0, 1.0 - float(dist))
                candidates[slot] = {"vector_sim": sim, "bm25": 0.0, "recency": 0.0,
                                    "confidence": 1.0, "recall_boost": 0.0}

        # --- BM25 candidates ---
        if self._document_store:
            hits = self._document_store.search_text(q.query, limit=oversample)
            for slot, score in hits:
                if slot < n and live_mask[slot]:
                    if slot not in candidates:
                        candidates[slot] = {"vector_sim": 0.0, "bm25": 0.0, "recency": 0.0,
                                            "confidence": 1.0, "recall_boost": 0.0}
                    candidates[slot]["bm25"] = abs(float(score))

        if not candidates:
            return Result(kind="nodes", data=[], count=0)

        # Normalize BM25 scores to 0-1
        max_bm25 = max((c["bm25"] for c in candidates.values()), default=0.0)
        if max_bm25 > 0:
            for c in candidates.values():
                c["bm25"] = c["bm25"] / max_bm25

        # --- Recency scoring ---
        updated_col = self.store.columns.get_column("__updated_at__", n)
        for slot, scores in candidates.items():
            age_days = 0.0
            if updated_col is not None:
                col_data, col_pres, _ = updated_col
                if col_pres[slot]:
                    age_ms = now_ms - int(col_data[slot])
                    age_days = max(0.0, age_ms / 86400000.0)
            scores["recency"] = math.exp(-age_days / 30.0)

        # --- Confidence scoring ---
        conf_col = self.store.columns.get_column("__confidence__", n)
        if conf_col is not None:
            col_data, col_pres, _ = conf_col
            for slot, scores in candidates.items():
                if col_pres[slot]:
                    scores["confidence"] = max(0.0, min(1.0, float(col_data[slot])))

        # --- Recall frequency boost ---
        recall_col = self.store.columns.get_column("__recall_count__", n)
        if recall_col is not None:
            col_data, col_pres, _ = recall_col
            max_recalls = 1.0
            for slot in candidates:
                if col_pres[slot]:
                    max_recalls = max(max_recalls, float(col_data[slot]))
            for slot, scores in candidates.items():
                if col_pres[slot] and max_recalls > 0:
                    scores["recall_boost"] = float(col_data[slot]) / max_recalls

        # --- Weighted fusion ---
        weights = getattr(self, '_remember_weights', [0.30, 0.20, 0.15, 0.20, 0.15])
        W_VECTOR = weights[0] if len(weights) > 0 else 0.30
        W_BM25 = weights[1] if len(weights) > 1 else 0.20
        W_RECENCY = weights[2] if len(weights) > 2 else 0.15
        W_CONFIDENCE = weights[3] if len(weights) > 3 else 0.20
        W_RECALL = weights[4] if len(weights) > 4 else 0.15

        scored = []
        for slot, scores in candidates.items():
            final = (W_VECTOR * scores["vector_sim"]
                     + W_BM25 * scores["bm25"]
                     + W_RECENCY * scores["recency"]
                     + W_CONFIDENCE * scores["confidence"]
                     + W_RECALL * scores["recall_boost"])
            scored.append((slot, final, scores))

        scored.sort(key=lambda x: -x[1])

        # --- Materialize, filter, and record retrieval feedback ---
        results = []
        retrieved_slots = []
        for slot, final_score, scores in scored:
            node = self.store._materialize_slot(slot)
            if node is None:
                continue
            if q.where and not self._eval_where(q.where.expr, node):
                continue
            node["_remember_score"] = round(final_score, 4)
            node["_vector_sim"] = round(scores["vector_sim"], 4)
            node["_bm25_score"] = round(scores["bm25"], 4)
            node["_recency_score"] = round(scores["recency"], 4)
            node["_confidence"] = round(scores["confidence"], 4)
            results.append(node)
            retrieved_slots.append(slot)
            if q.tokens is not None:
                # Token budget mode: estimate tokens from text length
                total_tokens = sum(
                    len(r.get("summary", "") + r.get("claim", "") + r.get("text", "")) // 4
                    for r in results
                )
                if total_tokens >= q.tokens:
                    break
            elif len(results) >= target_k:
                break

        # --- Retrieval feedback: increment recall count for returned nodes ---
        for slot in retrieved_slots:
            try:
                if self.store.columns.has_column("__recall_count__"):
                    if self.store.columns._presence["__recall_count__"][slot]:
                        current = int(self.store.columns._columns["__recall_count__"][slot])
                        self.store.columns.set_reserved(slot, "__recall_count__", current + 1)
                    else:
                        self.store.columns.set_reserved(slot, "__recall_count__", 1)
                else:
                    self.store.columns.set_reserved(slot, "__recall_count__", 1)
                self.store.columns.set_reserved(slot, "__last_recalled_at__", int(time.time() * 1000))
            except Exception:
                pass  # Don't fail retrieval on feedback errors

        return Result(kind="nodes", data=results, count=len(results))
