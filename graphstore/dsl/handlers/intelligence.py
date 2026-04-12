"""Intelligence handlers: RECALL, SIMILAR, LEXICAL SEARCH, WHAT IF."""

import time
from collections import deque
import concurrent.futures

import numpy as np
from scipy.sparse import csr_matrix

from graphstore.algos.fusion import (
    normalize_bm25 as _algo_normalize_bm25,
    recency_decay as _algo_recency_decay,
    weighted_remember_fusion as _algo_weighted_fusion,
)
from graphstore.algos.spreading import (
    spreading_activation as _algo_spreading_activation,
)
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

        if self.store.edge_matrices.total_edges == 0:
            return Result(kind="nodes", data=[], count=0)

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

        mat_t, mat_t_delta = self.store.edge_matrices.get_combined_transpose_split()
        if mat_t is None and mat_t_delta is None:
            return Result(kind="nodes", data=[], count=0)
            
        if mat_t is not None and mat_t.shape[0] < n:
            mat_t = resize_csr(mat_t, n)
        if mat_t_delta is not None and mat_t_delta.shape[0] < n:
            mat_t_delta = resize_csr(mat_t_delta, n)

        decay = getattr(self, '_recall_decay', 0.7)
        activation = _algo_spreading_activation(
            matrix_t=mat_t,
            cue_slot=cue_slot,
            depth=q.depth,
            decay=decay,
            live_mask=live_mask,
            importance=importance,
            recency=recency,
            matrix_t_delta=mat_t_delta,
        )

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
        sim_oversample = getattr(self, '_similar_to_oversample', 3)
        search_k = k * sim_oversample if q.where else k
        slots, dists = self._vector_store.search(query_vec, k=search_k, mask=combined_mask)

        conf_col = self.store.columns.get_column("__confidence__", n)
        results = []
        target_k = q.limit.value if q.limit else 10
        sim_floor = getattr(self, "_similarity_threshold", None)
        for slot_idx, dist in zip(slots, dists):
            slot = int(slot_idx)
            node = self.store._materialize_slot(slot)
            if node is None:
                continue
            if q.where and not self._eval_where(q.where.expr, node):
                continue
            similarity = 1.0 - float(dist)
            if sim_floor is not None and similarity < sim_floor:
                continue
            node["_similarity_score"] = round(similarity, 4)
            if conf_col is not None:
                col_data, col_pres, _ = conf_col
                if col_pres[slot]:
                    node["_confidence"] = round(float(col_data[slot]), 4)
            results.append(node)
            if len(results) >= target_k:
                break

        buf = getattr(self._runtime, "similarity_buffer", None)
        if buf is not None and results:
            buf.append(results[0]["_similarity_score"])

        return Result(kind="nodes", data=results, count=len(results))

    @handles(LexicalSearchQuery)
    def _lexical_search(self, q: LexicalSearchQuery) -> Result:
        """LEXICAL SEARCH: BM25 full-text search over document summaries."""
        if not self._document_store:
            return Result(kind="nodes", data=[], count=0)

        target_k = q.limit.value if q.limit else 10
        lex_oversample = getattr(self, '_lexical_search_oversample', 3)
        hits = self._document_store.search_text(q.query, limit=target_k * lex_oversample)

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
            from graphstore.algos.graph import bfs_reach

            self.store.columns.set_reserved(src_slot, "__retracted__", 1)

            combined = self.store.edge_matrices.get(None)
            n = self.store._next_slot
            affected_slots: set = {src_slot}

            if combined is not None:
                mat = combined
                if mat.shape[0] < n:
                    mat = resize_csr(mat, n)
                affected_slots |= bfs_reach(mat, src_slot, max_depth=None)

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
        target_k = q.limit.value if q.limit else 10
        oversample_factor = getattr(self, '_search_oversample', 5)
        oversample = target_k * oversample_factor

        n = self.store._next_slot
        if n == 0:
            return Result(kind="nodes", data=[], count=0)

        live_mask = self._compute_live_mask(n)
        now_ms = int(time.time() * 1000)

        vec_slots_np = np.empty(0, dtype=np.int64)
        vec_sims_np = np.empty(0, dtype=np.float64)
        if self._embedder and self._vector_store and self._vector_store.count() > 0:
            query_vec = self._embedder.encode_queries([q.query])[0]
            vs_cap = self._vector_store._capacity
            vs_mask = np.zeros(n, dtype=bool)
            cap = min(n, vs_cap)
            vs_mask[:cap] = self._vector_store._has_vector[:cap]
            combined = live_mask & vs_mask
            slots, dists = self._vector_store.search(
                query_vec, k=oversample, mask=combined, oversample_factor=oversample_factor,
            )
            if len(slots) > 0:
                vec_slots_np = np.asarray(slots, dtype=np.int64)
                vec_sims_np = np.maximum(
                    1.0 - np.asarray(dists, dtype=np.float64), 0.0
                )

        bm25_slots_np = np.empty(0, dtype=np.int64)
        bm25_scores_np = np.empty(0, dtype=np.float64)
        if self._document_store:
            hits = self._document_store.search_text(q.query, limit=oversample)
            if hits:
                raw_slots = np.fromiter(
                    (s for s, _ in hits), dtype=np.int64, count=len(hits),
                )
                raw_scores = np.abs(
                    np.fromiter(
                        (sc for _, sc in hits), dtype=np.float64, count=len(hits),
                    )
                )
                in_range = raw_slots < n
                if in_range.any():
                    rs = raw_slots[in_range]
                    rsc = raw_scores[in_range]
                    live_pick = live_mask[rs]
                    bm25_slots_np = rs[live_pick]
                    bm25_scores_np = rsc[live_pick]

        if len(vec_slots_np) == 0 and len(bm25_slots_np) == 0:
            return Result(kind="nodes", data=[], count=0)

        slot_arr = np.union1d(vec_slots_np, bm25_slots_np)
        
        vec_signal = np.zeros(n, dtype=np.float64)
        if len(vec_slots_np) > 0:
            vec_signal[vec_slots_np] = vec_sims_np

        bm25_signal = np.zeros(n, dtype=np.float64)
        if len(bm25_slots_np) > 0:
            scattered = np.zeros(n, dtype=np.float64)
            scattered[bm25_slots_np] = bm25_scores_np
            bm25_signal = _algo_normalize_bm25(scattered)

        recency_signal = np.zeros(n, dtype=np.float64)
        recency_signal[slot_arr] = 1.0
        updated_col = self.store.columns.get_column("__updated_at__", n)
        if updated_col is not None:
            col_data, col_pres, _ = updated_col
            pres_at = col_pres[slot_arr]
            if pres_at.any():
                half_life = getattr(self, '_recency_half_life_days', 30.0)
                r_scores = _algo_recency_decay(
                    col_data[slot_arr], pres_at, now_ms, half_life_days=half_life,
                )
                recency_signal[slot_arr] = r_scores

        confidence_signal = np.zeros(n, dtype=np.float64)
        confidence_signal[slot_arr] = 1.0
        conf_col = self.store.columns.get_column("__confidence__", n)
        if conf_col is not None:
            col_data, col_pres, _ = conf_col
            pres_at = col_pres[slot_arr]
            if pres_at.any():
                clamped = np.clip(col_data[slot_arr].astype(np.float64), 0.0, 1.0)
                confidence_signal[slot_arr] = np.where(pres_at, clamped, 1.0)

        recall_signal = np.zeros(n, dtype=np.float64)
        recall_col = self.store.columns.get_column("__recall_count__", n)
        if recall_col is not None:
            col_data, col_pres, _ = recall_col
            pres_at = col_pres[slot_arr]
            if pres_at.any():
                counts = col_data[slot_arr].astype(np.float64)
                max_recalls = max(1.0, float(counts[pres_at].max()))
                boost = counts / max_recalls
                recall_signal[slot_arr] = np.where(pres_at, boost, 0.0)

        weights = getattr(self, '_remember_weights', [0.30, 0.20, 0.15, 0.20, 0.15])
        base_final = _algo_weighted_fusion(
            vec_signal, bm25_signal, recency_signal,
            confidence_signal, recall_signal, list(weights),
        )
        base_final *= live_mask

        # --- HybridRAG Expansion (Vector-First k-Hop) ---
        # Use top fusion results as seeds, spread through graph, blend back
        # with normalized weight so graph signal doesn't overwhelm fusion.
        graph_weight = getattr(self, '_hybridrag_weight', 0.15)
        min_seeds = getattr(self, '_hybridrag_min_seeds', 5)
        seed_count = min(len(slot_arr), max(min_seeds, target_k // 2))
        final_scores = base_final.copy()

        if seed_count > 0 and self.store.edge_matrices.total_edges > 0:
            base_order = slot_arr[np.argsort(-base_final[slot_arr])]
            seed_slots = base_order[:seed_count].astype(np.int64)
            seed_scores = base_final[seed_slots].astype(np.float32)

            mat_t, mat_t_delta = self.store.edge_matrices.get_combined_transpose_split()
            if mat_t is not None or mat_t_delta is not None:
                if mat_t is not None and mat_t.shape[0] < n:
                    mat_t = resize_csr(mat_t, n)
                if mat_t_delta is not None and mat_t_delta.shape[0] < n:
                    mat_t_delta = resize_csr(mat_t_delta, n)

                activation = _algo_spreading_activation(
                    matrix_t=mat_t,
                    cue_slot=seed_slots,
                    depth=getattr(self, '_recall_depth', 2),
                    decay=getattr(self, '_recall_decay', 0.7),
                    live_mask=live_mask,
                    matrix_t_delta=mat_t_delta,
                    cue_scores=seed_scores,
                )

                # Normalize activation to 0-1 before blending
                act_max = activation.max()
                if act_max > 0:
                    activation /= act_max
                    final_scores = (1 - graph_weight) * base_final + graph_weight * activation

        valid_slots = np.where(final_scores > 0)[0]
        if len(valid_slots) == 0:
            return Result(kind="nodes", data=[], count=0)

        order = valid_slots[np.argsort(-final_scores[valid_slots])]

        results = []
        retrieved_slots = []
        for slot in order:
            slot = int(slot)
            node = self.store._materialize_slot(slot)
            if node is None:
                continue
            if q.where and not self._eval_where(q.where.expr, node):
                continue
            
            node["_remember_score"] = round(float(final_scores[slot]), 4)
            node["_vector_sim"] = round(float(vec_signal[slot]), 4)
            node["_bm25_score"] = round(float(bm25_signal[slot]), 4)
            node["_recency_score"] = round(float(recency_signal[slot]), 4)
            node["_confidence"] = round(float(confidence_signal[slot]), 4)
            
            results.append(node)
            retrieved_slots.append(slot)
            
            if q.tokens is not None:
                total_tokens = sum(
                    len(r.get("summary", "") + r.get("claim", "") + r.get("text", "")) // 4
                    for r in results
                )
                if total_tokens >= q.tokens:
                    break
            elif len(results) >= target_k:
                break

        buf = getattr(self._runtime, "similarity_buffer", None)
        if buf is not None and results:
            buf.append(results[0].get("_vector_sim", 0.0))

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
