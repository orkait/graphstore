"""Intelligence handlers: RECALL, SIMILAR, LEXICAL SEARCH, WHAT IF."""

import re
import time
from collections import deque

import numpy as np
from scipy.sparse import csr_matrix

from graphstore.algos.fusion import (
    normalize_bm25 as _algo_normalize_bm25,
    recency_decay as _algo_recency_decay,
    rrf_remember_fusion as _algo_rrf_fusion,
    temporal_proximity as _algo_temporal_proximity,
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

    _NUCLEUS_STRUCTURAL_EDGES = ("next", "prev", "has_section", "has_chunk")

    def _resolve_remember_parent_slot(self, slot: int, n: int) -> int:
        """Collapse sub-nodes (sentence, chunk, section) back to their logical parent."""
        if slot < 0 or slot >= n:
            return slot

        # Check for common parent fields across any kind
        for field in ("parent_chunk", "parent_node", "parent_id"):
            col_info = self.store.columns.get_column(field, n)
            if col_info is None:
                continue
            col_data, col_pres, _ = col_info
            if col_pres[slot]:
                parent_id_val = col_data[slot]
                if parent_id_val is not None:
                    parent_slot = self.store.id_to_slot.get(int(parent_id_val))
                    if parent_slot is not None:
                        return int(parent_slot)
        return slot

    def _nucleus_neighbors(self, slot: int) -> np.ndarray:
        """Traverse only structural edges for nucleus expansion."""
        seen: set[int] = set()
        neighbors: list[int] = []
        for edge_type in self._NUCLEUS_STRUCTURAL_EDGES:
            for arr in (
                self.store.edge_matrices.neighbors_out(slot, edge_type=edge_type),
                self.store.edge_matrices.neighbors_in(slot, edge_type=edge_type),
            ):
                for nb in arr:
                    nb_int = int(nb)
                    if nb_int in seen:
                        continue
                    seen.add(nb_int)
                    neighbors.append(nb_int)
        return np.asarray(neighbors, dtype=np.int32)

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

        mat, mat_delta = self.store.edge_matrices.get_combined_spread_split()
        if mat is None and mat_delta is None:
            return Result(kind="nodes", data=[], count=0)
            
        if mat is not None and mat.shape[0] < n:
            mat = resize_csr(mat, n)
        if mat_delta is not None and mat_delta.shape[0] < n:
            mat_delta = resize_csr(mat_delta, n)

        decay = getattr(self, '_recall_decay', 0.7)
        activation = _algo_spreading_activation(
            matrix_t=mat,
            cue_slot=cue_slot,
            depth=q.depth,
            decay=decay,
            live_mask=live_mask,
            importance=importance,
            recency=recency,
            matrix_t_delta=mat_delta,
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

        # Centralized snapshot — captures the fields the old hand-rolled
        # version missed (string_table, secondary_indices, _indexed_fields,
        # _edge_data_idx). Matters here because WHAT IF RETRACT runs
        # bfs_reach over the edge graph which indirectly pulls slots via
        # materialize_slot → column interned lookups (bug #49).
        snap = self.store.make_snapshot()

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
            # Unconditional restore: WHAT IF RETRACT is explicitly a
            # simulation, so we roll back even on the success path.
            self.store.restore_snapshot(snap)
            self.store._rebuild_edges()
            self.store._invalidate_live_cache()
            self.store._tombstone_mask_cache = None

    @handles(RememberQuery, write=True)
    def _remember(self, q: RememberQuery) -> Result:
        """REMEMBER: 3-signal fusion with optional reranker and nucleus.

        Pipeline: gather -> 3-signal fusion -> top-N -> reranker -> results

        Marked ``write=True`` because the handler mutates per-slot columns
        (``__recall_count__``, ``__last_recalled_at__``) as a side effect
        of every call. Pre-fix, the query was read-only from the dispatch
        layer's perspective so those column writes never reached the WAL
        — a crash between the mutation and the next checkpoint lost the
        recall-count increment (bug #50).
        """
        if getattr(self, '_embedder_dirty', False):
            from graphstore.core.errors import GraphStoreError
            raise GraphStoreError("Embedder changed. Run SYS REEMBED to update vectors.")

        target_k = q.limit.value if q.limit else 10
        oversample_factor = getattr(self, '_search_oversample', 5)

        n = self.store._next_slot
        if n == 0:
            return Result(
                kind="nodes", data=[], count=0,
                meta={"debug": {"empty_result_reasons": ["no nodes in store"]}},
            )

        live_mask = self._compute_live_mask(n)
        now_ms = int(time.time() * 1000)
        anchor_ms = getattr(q, 'at', None)
        at_range = getattr(q, 'at_range', None)

        # Sentence query expansion
        use_sqe = getattr(self, '_sentence_query_expansion', True)
        if use_sqe:
            from graphstore.algos.sentence_split import split_sentences
            sentences = split_sentences(q.query)
        else:
            sentences = [q.query]
        num_sentences = max(len(sentences), 1)

        # ── Stage 1: Candidate Gathering ──────────────────────────────
        vs_mask = None
        vec_slots_np = np.empty(0, dtype=np.int64)
        vec_sims_np = np.empty(0, dtype=np.float64)

        if self._embedder and self._vector_store and self._vector_store.count() > 0:
            vs_cap = self._vector_store._capacity
            vs_mask = np.zeros(n, dtype=bool)
            cap = min(n, vs_cap)
            vs_mask[:cap] = self._vector_store._has_vector[:cap]
            combined = live_mask & vs_mask

            _vec_max_sim: dict[int, float] = {}
            sent_vecs = self._embedder.encode_queries(sentences)
            per_sent_oversample = max(oversample_factor, 2)
            for svec in sent_vecs:
                slots, dists = self._vector_store.search(
                    svec, k=target_k * per_sent_oversample,
                    mask=combined, oversample_factor=per_sent_oversample,
                )
                if len(slots) > 0:
                    s_sims = np.maximum(1.0 - np.asarray(dists, dtype=np.float64), 0.0)
                    for j, slot in enumerate(slots):
                        original_slot = int(slot)
                        slot_int = self._resolve_remember_parent_slot(original_slot, n)
                        sim = float(s_sims[j])
                        if slot_int not in _vec_max_sim or sim > _vec_max_sim[slot_int]:
                            _vec_max_sim[slot_int] = sim

            if _vec_max_sim:
                vec_slots_np = np.array(list(_vec_max_sim.keys()), dtype=np.int64)
                vec_sims_np = np.array(list(_vec_max_sim.values()), dtype=np.float64)

        bm25_slots_np = np.empty(0, dtype=np.int64)
        bm25_scores_np = np.empty(0, dtype=np.float64)
        if self._document_store:
            hits = self._document_store.search_text(q.query, limit=target_k * oversample_factor)
            if hits:
                raw_slots = np.fromiter((s for s, _ in hits), dtype=np.int64, count=len(hits))
                raw_scores = np.abs(np.fromiter((sc for _, sc in hits), dtype=np.float64, count=len(hits)))
                in_range = raw_slots < n
                if in_range.any():
                    rs = raw_slots[in_range]
                    rsc = raw_scores[in_range]
                    live_pick = live_mask[rs]
                    bm25_slots_np = np.array(
                        [self._resolve_remember_parent_slot(int(slot), n) for slot in rs[live_pick]],
                        dtype=np.int64,
                    )
                    bm25_scores_np = rsc[live_pick]
                    if len(bm25_slots_np) > 0:
                        bm25_max: dict[int, float] = {}
                        for slot, score in zip(bm25_slots_np, bm25_scores_np):
                            slot_int = int(slot)
                            score_f = float(score)
                            prev = bm25_max.get(slot_int)
                            if prev is None or score_f > prev:
                                bm25_max[slot_int] = score_f
                        bm25_slots_np = np.array(list(bm25_max.keys()), dtype=np.int64)
                        bm25_scores_np = np.array(list(bm25_max.values()), dtype=np.float64)

        if len(vec_slots_np) == 0 and len(bm25_slots_np) == 0:
            early_meta: dict = {}
            anchor_ms_e = getattr(q, 'at', None)
            at_range_e = getattr(q, 'at_range', None)
            warnings: list[str] = []
            if at_range_e is not None or anchor_ms_e is not None:
                if self.store.columns.get_column("__event_at__", n) is None:
                    warnings.append(
                        "AT clause ignored: no '__event_at__' column in store. "
                        "Use ASSERT ... EVENT_AT ... or CREATE NODE ... EVENT_AT ... "
                        "to populate it."
                    )
            # Diagnose why both channels returned nothing so users do not have
            # to guess whether they need a schema, an embedder, or content.
            vec_ready = bool(self._embedder and self._vector_store and self._vector_store.count() > 0)
            fts_ready = False
            if self._document_store is not None:
                try:
                    row = self._document_store._conn.execute(
                        "SELECT 1 FROM doc_fts LIMIT 1"
                    ).fetchone()
                    fts_ready = row is not None
                except Exception:
                    fts_ready = False
            reasons: list[str] = []
            if not vec_ready:
                if self._embedder is None:
                    reasons.append("no embedder configured")
                elif self._vector_store is None or self._vector_store.count() == 0:
                    reasons.append(
                        "no vectors indexed - register schema with "
                        "'SYS REGISTER NODE KIND <kind> ... EMBED <field>' or "
                        "pass explicit VECTOR [...] on CREATE NODE"
                    )
            if not fts_ready:
                reasons.append(
                    "no BM25 content - use 'CREATE NODE ... DOCUMENT \"text\"' or "
                    "put_summary() so REMEMBER's keyword channel has data"
                )
            if not vec_ready and not fts_ready:
                reasons.append(
                    "neither vec nor BM25 is populated; REMEMBER has no channel to search"
                )
            if reasons:
                early_meta["debug"] = {"empty_result_reasons": reasons}
            if warnings:
                early_meta["warnings"] = warnings
            return Result(kind="nodes", data=[], count=0, meta=early_meta)

        # Adaptive oversample cap
        max_candidates = min(num_sentences * oversample_factor * target_k, 200)
        slot_arr = np.union1d(vec_slots_np, bm25_slots_np)
        if len(slot_arr) > max_candidates:
            combined_scores = np.zeros(n, dtype=np.float64)
            if len(vec_slots_np) > 0:
                combined_scores[vec_slots_np] = vec_sims_np
            if len(bm25_slots_np) > 0:
                bm25_norm = np.zeros(n, dtype=np.float64)
                bm25_norm[bm25_slots_np] = bm25_scores_np
                m = bm25_norm.max()
                if m > 0:
                    bm25_norm /= m
                combined_scores += bm25_norm
            top_indices = np.argsort(-combined_scores[slot_arr])[:max_candidates]
            slot_arr = slot_arr[top_indices]

        # ── Stage 2: Signal Fusion ───────────────────────────────────
        weights = getattr(self, '_remember_weights', [0.55, 0.25, 0.20])

        vec_signal = np.zeros(n, dtype=np.float64)
        if len(vec_slots_np) > 0:
            vec_signal[vec_slots_np] = vec_sims_np

        bm25_signal = np.zeros(n, dtype=np.float64)
        if len(bm25_slots_np) > 0:
            scattered = np.zeros(n, dtype=np.float64)
            scattered[bm25_slots_np] = bm25_scores_np
            m = scattered.max()
            if m > 0:
                bm25_signal = scattered / m

        # Co-occurrence bonus (counted once, not amplified by fusion weights)
        co_bonus = np.zeros(n, dtype=np.float64)
        if len(vec_slots_np) > 0 and len(bm25_slots_np) > 0:
            vec_set = set(vec_slots_np.tolist())
            bm25_set = set(bm25_slots_np.tolist())
            both = np.array(list(vec_set & bm25_set), dtype=np.int64)
            if len(both) > 0:
                co_bonus[both] = np.minimum(vec_signal[both], bm25_signal[both]) * 0.10

        recency_signal = np.zeros(n, dtype=np.float64)
        half_life = getattr(self, '_recency_half_life_days', 7300.0)
        event_col = self.store.columns.get_column("__event_at__", n)
        updated_col = self.store.columns.get_column("__updated_at__", n)

        has_event_mask = np.zeros(n, dtype=bool)
        if event_col is not None:
            _, e_pres, _ = event_col
            has_event_mask[:n] = e_pres[:n]

        if event_col is not None:
            e_data, e_pres, _ = event_col
            event_cands = slot_arr[has_event_mask[slot_arr]]
            if len(event_cands) > 0:
                e_ts = e_data[event_cands].astype(np.int64)
                e_ref = int(e_ts.max())
                e_pres_at = e_pres[event_cands]
                e_scores = _algo_recency_decay(e_ts, e_pres_at, e_ref, half_life_days=half_life)
                recency_signal[event_cands] = e_scores

        if updated_col is not None:
            u_data, u_pres, _ = updated_col
            no_event_cands = slot_arr[~has_event_mask[slot_arr]]
            if len(no_event_cands) > 0:
                u_ts = u_data[no_event_cands].astype(np.int64)
                u_pres_at = u_pres[no_event_cands]
                valid_u = u_ts[u_pres_at]
                u_ref = int(valid_u.max()) if len(valid_u) > 0 else now_ms
                u_scores = _algo_recency_decay(u_ts, u_pres_at, u_ref, half_life_days=half_life)
                recency_signal[no_event_cands] = u_scores

        # Graph signal: entity-aware scoring.
        # Chunks connected to high-degree entities (mentioned across many sessions)
        # get boosted. This rewards topical relevance over structural noise.
        graph_enabled = getattr(self, '_graph_signal_enabled', False)
        graph_signal = np.zeros(n, dtype=np.float64)
        if graph_enabled:
            # Cache entity degrees: entity_slot -> number of chunks mentioning it
            entity_degrees: dict[int, int] = {}

            def _entity_degree(ent_slot: int) -> int:
                if ent_slot not in entity_degrees:
                    ent_in = self.store.edge_matrices.neighbors_in(ent_slot, "mentions")
                    entity_degrees[ent_slot] = len(ent_in)
                return entity_degrees[ent_slot]

            for slot in slot_arr:
                slot_int = int(slot)
                # Get entities this chunk mentions
                ent_slots = self.store.edge_matrices.neighbors_out(slot_int, "mentions")
                if len(ent_slots) == 0:
                    continue
                # Score = sum of log1p(entity_degree)
                score = 0.0
                for es in ent_slots:
                    es_int = int(es)
                    if es_int < n:  # only valid slots
                        score += np.log1p(_entity_degree(es_int))
                graph_signal[slot_int] = score

            # Normalize against max in candidate set
            cand_scores = graph_signal[slot_arr]
            max_gs = float(cand_scores.max())
            if max_gs > 0:
                graph_signal /= max_gs

        # Fusion. Graph-signal semantics, parity across fusion methods:
        #   - 4 weights: graph is additive 4th channel       (w3 * graph)
        #   - 3 weights: graph replaces recency channel      (w2 * graph)  per spec
        #   - graph_enabled=False: recency stays, no graph
        fusion_method = getattr(self, '_fusion_method', 'weighted')
        if fusion_method == 'weighted':
            w = weights if len(weights) >= 3 else [0.55, 0.25, 0.20]
            if graph_enabled and len(w) >= 4:
                base_final = w[0] * vec_signal + w[1] * bm25_signal + w[2] * recency_signal + w[3] * graph_signal
            elif graph_enabled:
                base_final = w[0] * vec_signal + w[1] * bm25_signal + w[2] * graph_signal
            else:
                base_final = w[0] * vec_signal + w[1] * bm25_signal + w[2] * recency_signal
        else:
            signals = [vec_signal, bm25_signal]
            signals.append(graph_signal if graph_enabled else recency_signal)
            if graph_enabled and len(weights) >= 4:
                signals.append(recency_signal)
            base_final = _algo_rrf_fusion(*signals, candidate_slots=slot_arr,
                                           k_rrf=max(getattr(self, '_rrf_k', 60.0), 1.0))

        # Recall frequency boost: frequently recalled memories are implicitly important.
        # Log-scaled with w=0.05 to nudge without dominating: count 1→0.03, 10→0.12, 100→0.23
        recall_count_col = self.store.columns.get_column("__recall_count__", n)
        if recall_count_col is not None:
            rc_data, rc_pres, _ = recall_count_col
            rc_counts = rc_data[slot_arr].astype(np.float64)
            rc_valid = rc_pres[slot_arr]
            recall_boost = 0.05 * np.log1p(np.where(rc_valid, rc_counts, 0.0))
            base_final[slot_arr] += recall_boost

        base_final += co_bonus

        # ── Stage 3: Temporal Filter ─────────────────────────────────
        warnings: list[str] = []
        if at_range is not None or anchor_ms is not None:
            t_event_col = self.store.columns.get_column("__event_at__", n)
            if t_event_col is None:
                warnings.append(
                    "AT clause ignored: no '__event_at__' column in store. "
                    "Use ASSERT ... EVENT_AT ... or CREATE NODE ... EVENT_AT ... "
                    "to populate it."
                )
            else:
                col_data, col_pres, _ = t_event_col
                if at_range is not None:
                    start_ms, end_ms = at_range
                else:
                    start_ms = end_ms = int(anchor_ms)
                allowed = np.zeros(n, dtype=np.float64)
                sa_pres = col_pres[slot_arr]
                sa_values = col_data[slot_arr].astype(np.int64)
                in_range = sa_pres & (sa_values >= start_ms) & (sa_values <= end_ms)
                allowed[slot_arr[in_range]] = 1.0
                base_final *= allowed

        base_final *= live_mask

        # ── Stage 4: Top-N Selection + Materialization ───────────────
        if len(slot_arr) == 0:
            return Result(kind="nodes", data=[], count=0)

        # Sort all candidates by their final fused score
        order = slot_arr[np.argsort(-base_final[slot_arr])]

        # Materialize candidates for reranker
        results = []
        retrieved_slots = []
        running_tokens = 0
        texts_for_rerank = []

        for slot in order:
            slot = int(slot)
            node = self.store._materialize_slot(slot)
            if node is None:
                continue
            if q.where and not self._eval_where(q.where.expr, node):
                continue

            node["_remember_score"] = round(float(base_final[slot]), 4)
            node["_vector_sim"] = round(float(vec_signal[slot]), 4)
            node["_bm25_score"] = round(float(bm25_signal[slot]), 4)
            node["_recency_score"] = round(float(recency_signal[slot]), 4)

            results.append(node)
            retrieved_slots.append(slot)

            text = node.get("content") or node.get("summary") or node.get("text") or ""
            texts_for_rerank.append(text)

            if q.tokens is not None:
                running_tokens += len(
                    node.get("summary", "") + node.get("claim", "") + node.get("text", "")
                ) // 4
                if running_tokens >= q.tokens:
                    break

        # Reranker stage
        meta: dict = {}
        if warnings:
            meta.setdefault("warnings", []).extend(warnings)
        reranker = getattr(self, '_reranker', None)
        if reranker is not None and len(texts_for_rerank) > target_k:
            try:
                rerank_scores = reranker.score(q.query, texts_for_rerank)
                ranked = sorted(zip(rerank_scores, results, retrieved_slots),
                               key=lambda x: x[0], reverse=True)
                results = [r for _, r, _ in ranked[:target_k]]
                retrieved_slots = [s for _, _, s in ranked[:target_k]]
                # Update scores with reranker scores
                for score, node, slot in ranked[:target_k]:
                    node["_remember_score"] = round(float(score), 4)
            except Exception as rerank_err:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "reranker failed; falling back to fusion top-K: %s",
                    rerank_err,
                )
                meta["reranker_error"] = f"{type(rerank_err).__name__}: {rerank_err}"
                results = results[:target_k]
                retrieved_slots = retrieved_slots[:target_k]
        else:
            results = results[:target_k]
            retrieved_slots = retrieved_slots[:target_k]

        # ── Optional: Nucleus Expansion ──────────────────────────────
        nucleus_on = getattr(self, '_nucleus_expansion', False)
        if nucleus_on and results and self.store.edge_matrices.total_edges > 0:
            max_nb = getattr(self, '_nucleus_neighbors_per_hop', 3)
            n_hops = getattr(self, '_nucleus_hops', 1)
            min_text = getattr(self, '_nucleus_min_text_length', 20)
            total_budget = max(max_nb * n_hops * target_k, 100)
            allowed_kinds = set(getattr(self, '_nucleus_allowed_kinds',
                                        ["message", "chunk", "section"]))
            seen_slots = set(retrieved_slots)
            frontier = list(retrieved_slots)
            nucleus_results: list = []
            visits = 0
            for _hop in range(n_hops):
                if visits >= total_budget:
                    break
                next_frontier: list = []
                for seed_slot in frontier:
                    if visits >= total_budget:
                        break
                    nb_slots = self._nucleus_neighbors(seed_slot)
                    for nb in nb_slots:
                        if visits >= total_budget:
                            break
                        visits += 1
                        nb = int(nb)
                        if nb in seen_slots or not live_mask[nb]:
                            continue
                        nb_node = self.store._materialize_slot(nb)
                        if nb_node is None:
                            continue
                        nb_kind = nb_node.get("kind", "")
                        if nb_kind not in allowed_kinds:
                            seen_slots.add(nb)
                            continue
                        nb_text = (nb_node.get("content") or nb_node.get("summary")
                                   or nb_node.get("text") or "")
                        if len(nb_text) < min_text:
                            seen_slots.add(nb)
                            continue
                        nb_node["_nucleus"] = True
                        nucleus_results.append(nb_node)
                        seen_slots.add(nb)
                        next_frontier.append(nb)
                        if len(next_frontier) >= max_nb:
                            break
                frontier = next_frontier
            meta["nucleus"] = nucleus_results
            meta["nucleus_visits"] = visits

        # ── Retrieval Feedback ───────────────────────────────────────
        # Recall-count bumps are best-effort - if ColumnStore rejects a write
        # (schema lock, concurrent compact), retrieval still succeeds.
        import logging as _logging
        _rc_log = _logging.getLogger(__name__)
        for slot in retrieved_slots:
            if slot is None:
                continue
            try:
                if not self.store.columns.has_column("__recall_count__"):
                    self.store.columns._ensure_column("__recall_count__", "int64")
                pres_col = self.store.columns._presence.get("__recall_count__")
                if pres_col is not None and pres_col[slot]:
                    current = int(self.store.columns._columns["__recall_count__"][slot])
                    self.store.columns.set_reserved(slot, "__recall_count__", current + 1)
                else:
                    self.store.columns.set_reserved(slot, "__recall_count__", 1)
                self.store.columns.set_reserved(slot, "__last_recalled_at__", int(time.time() * 1000))
            except Exception as _rc_err:
                _rc_log.debug("recall count bump failed for slot=%s: %s", slot, _rc_err)

        buf = getattr(self._runtime, "similarity_buffer", None)
        if buf is not None and results:
            buf.append(results[0].get("_vector_sim", 0.0))

        return Result(kind="nodes", data=results, count=len(results), meta=meta)
