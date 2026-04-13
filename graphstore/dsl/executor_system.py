"""System DSL executor: handles SYS-prefixed queries.

Maps system AST nodes to CoreStore, SchemaRegistry, and sqlite operations
for stats, schema management, query logs, WAL, and diagnostics.
"""

import time
import logging

logger = logging.getLogger(__name__)
from graphstore.core.runtime import RuntimeState
from graphstore.core.types import Result
import numpy as np

from graphstore.dsl.ast_nodes import (
    Condition,
    MatchQuery,
    NodesQuery,
    SysCheckpoint,
    SysClear,
    SysConnect,
    SysConsolidate,
    SysContradictions,
    SysDescribe,
    SysDuplicates,
    SysEdgeKinds,
    SysEmbedders,
    SysExpire,
    SysExplain,
    SysFailedQueries,
    SysFrequentQueries,
    SysKinds,
    SysRebuild,
    SysReembed,
    SysRegisterEdgeKind,
    SysRegisterNodeKind,
    SysHealth,
    SysOptimize,
    SysEvict,
    SysRetain,
    SysRollback,
    SysSlowQueries,
    SysSnapshot,
    SysSnapshots,
    SysStats,
    SysStatus,
    SysUnregister,
    SysWal,
    SysLog,
    SysCronAdd,
    SysCronDelete,
    SysCronEnable,
    SysCronDisable,
    SysCronList,
    SysCronRun,
    SysEvolveRule,
    SysEvolveList,
    SysEvolveShow,
    SysEvolveEnable,
    SysEvolveDisable,
    SysEvolveDelete,
    SysEvolveHistory,
    SysEvolveReset,
    TraverseQuery,
)
from graphstore.dsl.cost_estimator import estimate_match_cost, estimate_traverse_cost
from graphstore.core.memory import estimate as estimate_memory
from graphstore.core.errors import GraphStoreError, NodeNotFound


class SystemExecutor:
    def __init__(
        self,
        runtime: RuntimeState,
        retention: dict | None = None,
        cron=None,
        evolution_engine=None,
    ):
        self._runtime = runtime
        self._retention = retention or {}
        self._cron = cron
        self._evolution_engine = evolution_engine
        self._eviction_target_ratio = 0.8
        self._start_time = time.time()
        self._duplicate_threshold_override: float | None = None
        self._protected_kinds: set[str] | None = None
        self._wal_manager = None

    @property
    def store(self):
        return self._runtime.store

    @property
    def schema(self):
        return self._runtime.schema

    @property
    def conn(self):
        return self._runtime.conn

    @property
    def _vector_store(self):
        return self._runtime.vector_store

    @property
    def _document_store(self):
        return self._runtime.document_store

    @property
    def _embedder(self):
        return self._runtime.embedder

    def execute(self, ast) -> Result:
        start = time.perf_counter_ns()
        result = self._dispatch(ast)
        result.elapsed_us = (time.perf_counter_ns() - start) // 1000
        return result

    def _dispatch(self, ast) -> Result:
        handlers = {
            SysStats: self._stats,
            SysKinds: self._kinds,
            SysEdgeKinds: self._edge_kinds,
            SysDescribe: self._describe,
            SysSlowQueries: self._slow_queries,
            SysFrequentQueries: self._frequent_queries,
            SysFailedQueries: self._failed_queries,
            SysExplain: self._explain,
            SysRegisterNodeKind: self._register_node_kind,
            SysRegisterEdgeKind: self._register_edge_kind,
            SysUnregister: self._unregister,
            SysCheckpoint: self._checkpoint,
            SysRebuild: self._rebuild,
            SysClear: self._clear,
            SysWal: self._wal,
            SysExpire: self._expire,
            SysContradictions: self._contradictions,
            SysSnapshot: self._snapshot,
            SysRollback: self._rollback,
            SysSnapshots: self._snapshots,
            SysDuplicates: self._duplicates,
            SysEmbedders: self._embedders,
            SysConnect: self._connect,
            SysConsolidate: self._consolidate,
            SysReembed: self._reembed,
            SysStatus: self._status,
            SysRetain: self._retain,
            SysHealth: self._health,
            SysOptimize: self._optimize,
            SysEvict: self._evict,
            SysLog: self._log,
            SysCronAdd: self._cron_add,
            SysCronDelete: self._cron_delete,
            SysCronEnable: self._cron_enable,
            SysCronDisable: self._cron_disable,
            SysCronList: self._cron_list,
            SysCronRun: self._cron_run,
            SysEvolveRule: self._evolve_rule,
            SysEvolveList: self._evolve_list,
            SysEvolveShow: self._evolve_show,
            SysEvolveEnable: self._evolve_enable,
            SysEvolveDisable: self._evolve_disable,
            SysEvolveDelete: self._evolve_delete,
            SysEvolveHistory: self._evolve_history,
            SysEvolveReset: self._evolve_reset,
        }
        handler = handlers.get(type(ast))
        if handler is None:
            raise GraphStoreError(f"Unknown system command: {type(ast).__name__}")
        return handler(ast)

    def _stats(self, q: SysStats) -> Result:
        data = {}
        if q.target is None or q.target == "NODES":
            data["node_count"] = self.store.node_count
        if q.target is None or q.target == "EDGES":
            data["edge_count"] = self.store.edge_count
            if q.target is None:
                data["edge_counts_by_type"] = {
                    etype: len(edges)
                    for etype, edges in self.store._edges_by_type.items()
                }
        if q.target is None or q.target == "MEMORY":
            data["memory_bytes"] = estimate_memory(
                self.store.node_count, self.store.edge_count
            )
            data["ceiling_bytes"] = self.store._ceiling_bytes
            data["column_memory_bytes"] = self.store.columns.memory_bytes
        if q.target is None or q.target == "WAL":
            if self.conn:
                row = self.conn.execute("SELECT COUNT(*) FROM wal").fetchone()
                data["wal_entries"] = row[0]
            else:
                data["wal_entries"] = 0
        if q.target is None:
            data["uptime_seconds"] = time.time() - self._start_time
        return Result(kind="stats", data=data, count=1)

    def _kinds(self, q: SysKinds) -> Result:
        kinds = self.schema.list_node_kinds()
        return Result(kind="schema", data=kinds, count=len(kinds))

    def _edge_kinds(self, q: SysEdgeKinds) -> Result:
        kinds = self.schema.list_edge_kinds()
        return Result(kind="schema", data=kinds, count=len(kinds))

    def _describe(self, q: SysDescribe) -> Result:
        if q.entity_type == "NODE":
            data = self.schema.describe_node_kind(q.name)
        else:
            data = self.schema.describe_edge_kind(q.name)
        return Result(kind="schema", data=data, count=1 if data else 0)

    def _slow_queries(self, q: SysSlowQueries) -> Result:
        if not self.conn:
            return Result(kind="log_entries", data=[], count=0)
        sql = "SELECT timestamp, query, elapsed_us, result_count, error FROM query_log"
        params: list = []
        if q.since:
            sql += " WHERE timestamp >= ?"
            import datetime

            dt = datetime.datetime.fromisoformat(q.since)
            params.append(dt.timestamp())
        sql += " ORDER BY elapsed_us DESC"
        if q.limit:
            sql += " LIMIT ?"
            params.append(q.limit.value)
        rows = self.conn.execute(sql, params).fetchall()
        entries = [
            {
                "timestamp": r[0],
                "query": r[1],
                "elapsed_us": r[2],
                "result_count": r[3],
                "error": r[4],
            }
            for r in rows
        ]
        return Result(kind="log_entries", data=entries, count=len(entries))

    def _frequent_queries(self, q: SysFrequentQueries) -> Result:
        if not self.conn:
            return Result(kind="log_entries", data=[], count=0)
        sql = "SELECT query, COUNT(*) as cnt FROM query_log GROUP BY query ORDER BY cnt DESC"
        params: list = []
        if q.limit:
            sql += " LIMIT ?"
            params.append(q.limit.value)
        rows = self.conn.execute(sql, params).fetchall()
        entries = [{"query": r[0], "count": r[1]} for r in rows]
        return Result(kind="log_entries", data=entries, count=len(entries))

    def _failed_queries(self, q: SysFailedQueries) -> Result:
        if not self.conn:
            return Result(kind="log_entries", data=[], count=0)
            
        entries = []
        params: list = []
        if q.limit:
            params.append(q.limit.value)
            
        # 1. Fetch runtime query failures
        sql_log = (
            "SELECT timestamp, query, elapsed_us, error FROM query_log "
            "WHERE error IS NOT NULL ORDER BY timestamp DESC"
        )
        if q.limit:
            sql_log += " LIMIT ?"
        rows_log = self.conn.execute(sql_log, params).fetchall()
        for r in rows_log:
            entries.append({
                "source": "runtime",
                "timestamp": r[0],
                "query": r[1],
                "elapsed_us": r[2],
                "error": r[3],
            })
            
        # 2. Fetch recovery (WAL replay) failures
        sql_wal = "SELECT timestamp, statement, error_msg FROM failed_wal_entries ORDER BY timestamp DESC"
        if q.limit:
            sql_wal += " LIMIT ?"
        rows_wal = self.conn.execute(sql_wal, params).fetchall()
        for r in rows_wal:
            entries.append({
                "source": "recovery",
                "timestamp": r[0],
                "query": r[1],
                "elapsed_us": 0,
                "error": r[2],
            })
            
        entries.sort(key=lambda x: x["timestamp"], reverse=True)
        if q.limit:
            entries = entries[:q.limit.value]
            
        return Result(kind="log_entries", data=entries, count=len(entries))

    def _explain(self, q: SysExplain) -> Result:
        """Run cost estimator on the inner query without executing."""
        inner = q.query
        if isinstance(inner, MatchQuery):
            cost = estimate_match_cost(inner.pattern, self.store.edge_matrices)
        elif isinstance(inner, TraverseQuery):
            edge_type = None
            if inner.where:
                expr = inner.where.expr
                if (
                    isinstance(expr, Condition)
                    and expr.field == "kind"
                    and expr.op == "="
                ):
                    edge_type = expr.value
            cost = estimate_traverse_cost(
                inner.depth, self.store.edge_matrices, edge_type
            )
        elif isinstance(inner, NodesQuery):
            data = {"type": "scan", "estimated_nodes": self.store.node_count}
            if inner.where:
                expr = inner.where.expr
                if (
                    isinstance(expr, Condition)
                    and expr.field in self.store._indexed_fields
                ):
                    data["type"] = "index_lookup"
            return Result(kind="plan", data=data, count=1)
        else:
            return Result(
                kind="plan",
                data={
                    "type": "direct",
                    "note": "no cost estimation for this query type",
                },
                count=1,
            )

        return Result(kind="plan", data=cost.to_dict(), count=1)

    def _register_node_kind(self, q: SysRegisterNodeKind) -> Result:
        self.schema.register_node_kind(q.kind, q.required, q.optional,
                                       embed_field=q.embed_field)
        # Pre-create columns for typed fields
        type_map = {"string": "int32_interned", "int": "int64", "float": "float64"}
        for item in q.required + q.optional:
            if isinstance(item, tuple):
                name, type_name = item
            else:
                name, type_name = item, None
            if type_name and type_name in type_map:
                self.store.columns.declare_column(name, type_map[type_name])
        return Result(kind="ok", data=None, count=0)

    def _register_edge_kind(self, q: SysRegisterEdgeKind) -> Result:
        self.schema.register_edge_kind(q.kind, q.from_kinds, q.to_kinds)
        return Result(kind="ok", data=None, count=0)

    def _unregister(self, q: SysUnregister) -> Result:
        if q.entity_type == "NODE":
            self.schema.unregister_node_kind(q.kind)
        else:
            self.schema.unregister_edge_kind(q.kind)
        return Result(kind="ok", data=None, count=0)

    def _checkpoint(self, q: SysCheckpoint) -> Result:
        # Will be wired up by GraphStore
        return Result(kind="ok", data=None, count=0)

    def _rebuild(self, q: SysRebuild) -> Result:
        self.store._rebuild_edges()
        # Rebuild secondary indices from columns (sole source of truth)
        for field in list(self.store._indexed_fields):
            self.store.add_index(field)
        return Result(kind="ok", data=None, count=0)

    def _clear(self, q: SysClear) -> Result:
        if q.target == "LOG":
            if self.conn:
                self.conn.execute("DELETE FROM query_log")
                self.conn.commit()
        elif q.target == "CACHE":
            from graphstore.dsl.parser import clear_cache

            clear_cache()
            self.store.edge_matrices._cache.clear()
            self.store.edge_matrices._transpose_cache.clear()
        return Result(kind="ok", data=None, count=0)

    def _wal(self, q: SysWal) -> Result:
        if q.action == "STATUS":
            if self.conn:
                row = self.conn.execute("SELECT COUNT(*) FROM wal").fetchone()
                count = row[0]
                size_row = self.conn.execute(
                    "SELECT COALESCE(SUM(LENGTH(statement)), 0) FROM wal"
                ).fetchone()
                size = size_row[0]
                return Result(
                    kind="stats",
                    data={"wal_entries": count, "wal_bytes": size},
                    count=1,
                )
            return Result(
                kind="stats", data={"wal_entries": 0, "wal_bytes": 0}, count=1
            )
        elif q.action == "REPLAY":
            # Will be wired up by GraphStore
            return Result(kind="ok", data=None, count=0)
        return Result(kind="ok", data=None, count=0)

    def _expire(self, q: SysExpire) -> Result:
        """SYS EXPIRE: tombstone nodes whose __expires_at__ has passed."""
        n = self.store._next_slot
        if n == 0:
            return Result(kind="ok", data={"expired": 0}, count=0)

        # Build live mask (non-tombstoned only, don't filter by expiry since we want to find expired)
        mask = self.store.node_ids[:n] >= 0
        if self.store.node_tombstones:
            tomb_arr = np.array(list(self.store.node_tombstones), dtype=np.int32)
            tomb_mask = np.zeros(n, dtype=bool)
            valid = tomb_arr[tomb_arr < n]
            if len(valid) > 0:
                tomb_mask[valid] = True
            mask = mask & ~tomb_mask

        # Apply optional WHERE filter (kind filter)
        if q.where:
            from graphstore.dsl.executor import Executor
            kind_filter = None
            expr = q.where.expr
            if isinstance(expr, Condition) and expr.field == "kind" and expr.op == "=":
                kind_filter = expr.value
            if kind_filter:
                kind_mask = self.store._live_mask(kind_filter)
                mask = mask & kind_mask

        # Find expired nodes
        expires = self.store.columns.get_column("__expires_at__", n)
        if expires is None:
            return Result(kind="ok", data={"expired": 0}, count=0)

        col, pres, _ = expires
        now_ms = int(time.time() * 1000)
        expired_mask = mask & pres & (col > 0) & (col < now_ms)

        expired_slots = np.nonzero(expired_mask)[0]
        if len(expired_slots) == 0:
            return Result(kind="ok", data={"expired": 0}, count=0)

        # Batch tombstone: clear columns, remove from id_to_slot, add to tombstones
        expired_count = 0
        expired_slot_set = set()
        for slot_idx in expired_slots:
            slot = int(slot_idx)
            nid_str_id = int(self.store.node_ids[slot])
            if nid_str_id == -1:
                continue

            # Remove from secondary indices
            for field in self.store._indexed_fields:
                if self.store.columns.has_column(field) and self.store.columns._presence[field][slot]:
                    dtype = self.store.columns._dtypes[field]
                    raw = self.store.columns._columns[field][slot]
                    if dtype == "int32_interned":
                        val = self.store.string_table.lookup(int(raw))
                    elif dtype == "float64":
                        val = float(raw)
                    else:
                        val = int(raw)
                    idx_list = self.store.secondary_indices.get(field, {}).get(val, [])
                    if slot in idx_list:
                        idx_list.remove(slot)

            self.store.columns.clear(slot)
            self.store.node_tombstones.add(slot)
            if nid_str_id in self.store.id_to_slot:
                del self.store.id_to_slot[nid_str_id]
            self.store._count -= 1
            expired_slot_set.add(slot)
            expired_count += 1

        # Clean vectors for expired nodes
        if self._vector_store:
            for slot_idx in expired_slots:
                slot = int(slot_idx)
                self._vector_store.remove(slot)

        # Single pass: remove all expired slots from all edge lists at once
        if expired_slot_set:
            any_removed = False
            for etype in list(self.store._edges_by_type.keys()):
                old_len = len(self.store._edges_by_type[etype])
                self.store._edges_by_type[etype] = [
                    (s, t, d) for s, t, d in self.store._edges_by_type[etype]
                    if s not in expired_slot_set and t not in expired_slot_set
                ]
                if not self.store._edges_by_type[etype]:
                    del self.store._edges_by_type[etype]
                if len(self.store._edges_by_type.get(etype, [])) != old_len:
                    any_removed = True

            if any_removed:
                # Rebuild edge_keys from scratch once
                self.store._edge_keys = {
                    (s, t, k)
                    for k, edges in self.store._edges_by_type.items()
                    for s, t, _d in edges
                }
            self.store._edges_dirty = True
            self.store._ensure_edges_built()

        return Result(kind="ok", data={"expired": expired_count}, count=expired_count)

    def _contradictions(self, q: SysContradictions) -> Result:
        """SYS CONTRADICTIONS: find groups with >1 distinct value for a field."""
        n = self.store._next_slot
        if n == 0:
            return Result(kind="contradictions", data=[], count=0)

        # Build live mask (tombstones + TTL + retracted)
        mask = self.store.compute_live_mask(n)

        # Apply WHERE filter (kind filter)
        if q.where:
            expr = q.where.expr
            if isinstance(expr, Condition) and expr.field == "kind" and expr.op == "=":
                kind_mask = self.store._live_mask(expr.value)
                mask = mask & kind_mask

        filtered_count = int(np.sum(mask))
        if filtered_count == 0:
            return Result(kind="contradictions", data=[], count=0)

        # Get group_by and field columns
        group_field = q.group_by
        value_field = q.field

        if not self.store.columns.has_column(group_field):
            return Result(kind="contradictions", data=[], count=0)
        if not self.store.columns.has_column(value_field):
            return Result(kind="contradictions", data=[], count=0)

        group_col = self.store.columns._columns[group_field][:n][mask]
        value_col = self.store.columns._columns[value_field][:n][mask]
        group_dtype = self.store.columns._dtypes[group_field]
        value_dtype = self.store.columns._dtypes[value_field]

        # Group by group_col, find groups with >1 distinct value
        unique_groups = np.unique(group_col)
        contradictions = []
        for gkey in unique_groups:
            group_mask = group_col == gkey
            values_in_group = value_col[group_mask]
            unique_vals = np.unique(values_in_group)
            if len(unique_vals) > 1:
                # Resolve group key
                if group_dtype == "int32_interned":
                    group_name = self.store.string_table.lookup(int(gkey))
                elif group_dtype == "float64":
                    group_name = float(gkey)
                else:
                    group_name = int(gkey)
                # Resolve values
                resolved_vals = []
                for v in unique_vals:
                    if value_dtype == "int32_interned":
                        resolved_vals.append(self.store.string_table.lookup(int(v)))
                    elif value_dtype == "float64":
                        resolved_vals.append(float(v))
                    else:
                        resolved_vals.append(int(v))
                contradictions.append({
                    "group": group_name,
                    "values": resolved_vals,
                    "count": len(unique_vals),
                })

        return Result(kind="contradictions", data=contradictions, count=len(contradictions))

    def _snapshot(self, q: SysSnapshot) -> Result:
        """SYS SNAPSHOT: save full graph state to a named snapshot."""
        store = self.store
        ns = store._next_slot
        snap = {
            "columns": store.columns.snapshot_arrays(),
            "node_ids": store.node_ids[:ns].copy(),
            "node_kinds": store.node_kinds[:ns].copy(),
            "tombstones": set(store.node_tombstones),
            "edges_by_type": {k: list(v) for k, v in store._edges_by_type.items()},
            "edge_keys": set(store._edge_keys),
            "id_to_slot": dict(store.id_to_slot),
            "next_slot": ns,
            "count": store._count,
            "capacity": store._capacity,
            "active_context": store._active_context,
        }
        # Snapshot vector state
        vs = self._vector_store
        if vs is not None and vs.count() > 0:
            snap["vector_index"] = vs.save()
            snap["vector_presence"] = vs._has_vector[:ns].copy()
            snap["vector_dims"] = vs.dims
        else:
            snap["vector_index"] = None
        store._snapshots[q.name] = snap
        return Result(kind="ok", data={"snapshot": q.name}, count=1)

    def _rollback(self, q: SysRollback) -> Result:
        """SYS ROLLBACK TO: restore full graph state from a named snapshot."""
        store = self.store
        if q.name not in store._snapshots:
            raise GraphStoreError(f"Snapshot not found: {q.name!r}")

        snap = store._snapshots[q.name]

        saved_next_slot = snap["next_slot"]
        saved_capacity = snap.get("capacity", saved_next_slot)
        while store._capacity < saved_capacity:
            store._grow()

        store.columns.restore_arrays(snap["columns"])
        store.columns.grow(store._capacity)

        # Restore node arrays
        store.node_ids[:saved_next_slot] = snap["node_ids"]
        store.node_kinds[:saved_next_slot] = snap["node_kinds"]

        # Clear any slots beyond the saved state
        if saved_next_slot < store._next_slot:
            store.node_ids[saved_next_slot:store._next_slot] = -1
            store.node_kinds[saved_next_slot:store._next_slot] = 0

        store.node_tombstones = set(snap["tombstones"])
        store._edges_by_type = {k: list(v) for k, v in snap["edges_by_type"].items()}
        store._edge_keys = set(snap["edge_keys"])
        store.id_to_slot = dict(snap["id_to_slot"])
        store._next_slot = saved_next_slot
        store._count = snap["count"]
        store._active_context = snap.get("active_context")

        # Restore vector state - mutate the runtime container so every
        # component sees the new VectorStore through their shared ref.
        if snap.get("vector_index") is not None:
            from graphstore.vector.store import VectorStore
            vs = VectorStore(dims=snap["vector_dims"], capacity=store._capacity)
            vs.load(snap["vector_index"])
            vs._has_vector[:len(snap["vector_presence"])] = snap["vector_presence"]
            self._runtime.vector_store = vs
        else:
            self._runtime.vector_store = None

        # Rebuild derived state
        store._rebuild_edges()

        # Orphan cleanup in DocumentStore
        if self._document_store:
            live = store.compute_live_mask(store._next_slot)
            live_slots = set(int(s) for s in np.nonzero(live)[0])
            try:
                self._document_store.orphan_cleanup(live_slots)
            except Exception as e:
                logger.debug("orphan cleanup after rollback failed: %s", e, exc_info=True)

        return Result(kind="ok", data={"rollback": q.name}, count=1)

    def _snapshots(self, q: SysSnapshots) -> Result:
        """SYS SNAPSHOTS: list all named snapshots."""
        names = list(self.store._snapshots.keys())
        return Result(kind="snapshots", data=names, count=len(names))

    def _duplicates(self, q: SysDuplicates) -> Result:
        """SYS DUPLICATES: find near-duplicate nodes by vector similarity."""
        vs = self._vector_store
        if vs is None or vs.count() == 0:
            return Result(kind="duplicates", data=[], count=0)

        store = self.store
        n = store._next_slot

        effective_threshold = q.threshold
        if self._duplicate_threshold_override is not None and q.threshold == 0.95:
            effective_threshold = self._duplicate_threshold_override

        mask = store.compute_live_mask(n)

        # Apply optional WHERE filter
        if q.where:
            expr = q.where.expr
            if isinstance(expr, Condition) and expr.field == "kind" and expr.op == "=":
                kind_mask = store._live_mask(expr.value)
                mask = mask & kind_mask

        # Collect slots that are both live and have vectors
        candidate_slots = []
        for slot in range(n):
            if mask[slot] and vs.has_vector(slot):
                candidate_slots.append(slot)

        if len(candidate_slots) < 2:
            return Result(kind="duplicates", data=[], count=0)

        # For each candidate, find its nearest neighbor among candidates
        seen = set()
        pairs = []
        candidate_mask = np.zeros(max(n, vs._capacity), dtype=bool)
        for s in candidate_slots:
            candidate_mask[s] = True

        for slot in candidate_slots:
            vec = vs.get_vector(slot)
            if vec is None:
                continue
            # Search for top-2 (first result may be self)
            slots_found, dists = vs.search(vec, k=2, mask=candidate_mask)
            for found_slot, dist in zip(slots_found, dists):
                found_slot = int(found_slot)
                if found_slot == slot:
                    continue
                # cosine similarity = 1 - cosine distance
                similarity = 1.0 - float(dist)
                if similarity >= effective_threshold:
                    pair_key = (min(slot, found_slot), max(slot, found_slot))
                    if pair_key not in seen:
                        seen.add(pair_key)
                        # Resolve IDs
                        id_a = store.string_table.lookup(int(store.node_ids[pair_key[0]]))
                        id_b = store.string_table.lookup(int(store.node_ids[pair_key[1]]))
                        pairs.append({
                            "node_a": id_a,
                            "node_b": id_b,
                            "similarity": round(similarity, 6),
                        })

        return Result(kind="duplicates", data=pairs, count=len(pairs))

    def _embedders(self, q: SysEmbedders) -> Result:
        """SYS EMBEDDERS: list active embedder info."""
        info = []
        if self._embedder:
            info.append({
                "name": self._embedder.name,
                "dims": self._embedder.dims,
                "status": "active",
            })
        else:
            info.append({"name": "none", "status": "no embedder configured"})
        return Result(kind="embedders", data=info, count=len(info))

    def _connect(self, q: SysConnect) -> Result:
        """SYS CONNECT: auto-wire cross-document relationships via vector similarity."""
        from graphstore.ingest.connector import connect_all
        return connect_all(
            self.store, self._vector_store,
            threshold=q.threshold, where_expr=q.where,
        )

    def _consolidate(self, q: SysConsolidate) -> Result:
        """SYS CONSOLIDATE: cluster episodic memories into observations.

        Groups messages by entity (via 'mentions' edges), clusters by
        cosine similarity, creates 'observation' nodes with evidence links.
        No LLM required - picks the most representative message per cluster.
        """
        from graphstore.algos.consolidation import cluster_by_entity

        store = self.store
        n = store._next_slot
        live = store.compute_live_mask(n)

        # Build entity -> message slot mapping from graph edges
        entity_to_slots: dict[str, list[int]] = {}
        for etype, edge_list in store._edges_by_type.items():
            if etype != "mentions":
                continue
            for src_slot, tgt_slot, _data in edge_list:
                if not live[src_slot] or not live[tgt_slot]:
                    continue
                # src=message, tgt=entity
                entity_id = store._slot_to_id(tgt_slot)
                if entity_id:
                    entity_to_slots.setdefault(entity_id, []).append(src_slot)

        if not entity_to_slots:
            return Result(kind="ok", data={"observations": 0, "reason": "no entity edges found"}, count=0)

        # Gather texts
        texts: dict[int, str] = {}
        lookup = store.string_table.lookup
        for entity_slots in entity_to_slots.values():
            for s in entity_slots:
                node = store._materialize_slot(s)
                if node:
                    texts[s] = node.get("content") or node.get("summary") or node.get("claim") or ""

        # Gather event times
        event_times: dict[int, int] = {}
        event_col = store.columns.get_column("__event_at__", n)
        if event_col is not None:
            col_data, col_pres, _ = event_col
            for slots in entity_to_slots.values():
                for s in slots:
                    if col_pres[s]:
                        event_times[s] = int(col_data[s])

        # Gather vectors
        vs = self._vector_store
        vectors = None
        has_vector = None
        if vs is not None:
            max_slot = max((s for slots in entity_to_slots.values() for s in slots), default=-1)
            if max_slot >= 0:
                cap = max_slot + 1
                vectors = np.zeros((cap, vs.dims), dtype=np.float32)
                has_vector = np.zeros(cap, dtype=bool)
                for slots in entity_to_slots.values():
                    for s in slots:
                        if s >= cap or not vs.has_vector(s):
                            continue
                        vec = vs.get_vector(s)
                        if vec is None:
                            continue
                        vectors[s] = vec
                        has_vector[s] = True

        observations = cluster_by_entity(
            entity_to_slots=entity_to_slots,
            vectors=vectors,
            has_vector=has_vector,
            texts=texts,
            event_times=event_times,
            similarity_threshold=q.similarity_threshold,
            min_cluster_size=q.min_cluster_size,
        )

        # Create observation nodes + evidence edges
        created = 0
        for i, obs in enumerate(observations):
            obs_id = f"obs:{obs.entity}:{i}"
            try:
                store.put_node(obs_id, "observation", {
                    "entity": obs.entity,
                    "content": obs.text,
                    "evidence_count": obs.evidence_count,
                })
                obs_slot = store.id_to_slot[store.string_table.intern(obs_id)]

                # Set confidence and event_at
                store.columns.set_reserved(obs_slot, "__confidence__", obs.confidence)
                if obs.event_at_ms is not None:
                    store.columns.set_reserved(obs_slot, "__event_at__", obs.event_at_ms)

                # Embed observation with cluster centroid
                if obs.centroid is not None and vs is not None:
                    vs.add(obs_slot, obs.centroid)

                # FTS index the observation text
                if self._document_store and obs.text:
                    self._document_store.put_summary(obs_slot, obs.text, doc_slot=0, chunk_index=0)

                # Link to evidence
                for ev_slot in obs.evidence_slots:
                    ev_id = store._slot_to_id(ev_slot)
                    if ev_id:
                        try:
                            store.put_edge(obs_id, ev_id, "evidence")
                        except Exception:
                            pass

                created += 1
            except Exception:
                continue

        return Result(kind="ok", data={
            "observations": created,
            "entities": len(entity_to_slots),
            "clusters": len(observations),
        }, count=created)

    def _reembed(self, q: SysReembed) -> Result:
        """SYS REEMBED: re-embed all summaries with current embedder."""
        if not self._embedder:
            raise GraphStoreError("No embedder configured")

        vs = self._vector_store
        if vs is None:
            raise GraphStoreError("No vector store initialized")

        store = self.store
        n = store._next_slot
        live = store.compute_live_mask(n)

        pairs: list[tuple[int, str]] = []
        if store.columns.has_column("summary"):
            col_info = store.columns.get_column("summary", n)
            if col_info is not None:
                col_data, col_pres, col_dtype = col_info
                if col_dtype == "int32_interned":
                    lookup = store.string_table.lookup
                    for slot in range(n):
                        if not live[slot] or not col_pres[slot]:
                            continue
                        text = lookup(int(col_data[slot]))
                        if text:
                            pairs.append((slot, text))

        if pairs:
            texts = [t for _, t in pairs]
            vecs = self._embedder.encode_documents(texts)
            for (slot, _), vec in zip(pairs, vecs):
                vs.add(slot, vec)

        # Clear dirty flag on the GraphStore (set by caller after return)
        return Result(kind="ok", data={"reembedded": len(pairs)}, count=len(pairs))

    def _status(self, q: SysStatus) -> Result:
        """SYS STATUS: comprehensive system state."""
        store = self.store
        data = {
            "nodes": store.node_count,
            "edges": store.edge_count,
            "memory_bytes": estimate_memory(store.node_count, store.edge_count),
            "ceiling_bytes": store._ceiling_bytes,
            "column_memory_bytes": store.columns.memory_bytes,
            "memory_measured": None,
        }

        # Vector store info
        vs = self._vector_store
        if vs is not None:
            data["vectors"] = {
                "count": vs.count(),
                "dims": vs.dims,
                "memory_bytes": vs.memory_bytes,
            }
        else:
            data["vectors"] = {"count": 0, "dims": 0, "memory_bytes": 0}

        # Embedder info
        if self._embedder:
            data["embedder"] = {
                "name": getattr(self._embedder, "name", "unknown"),
                "dims": getattr(self._embedder, "dims", 0),
                "status": "active",
            }
        else:
            data["embedder"] = {"name": "none", "status": "inactive"}

        # Document store info
        if self._document_store:
            try:
                data["documents"] = self._document_store.stats()
            except Exception as e:
                logger.debug("document store stats failed: %s", e, exc_info=True)
                data["documents"] = {}
        else:
            data["documents"] = {}

        # Edge types
        data["edge_types"] = {
            etype: len(edges)
            for etype, edges in store._edges_by_type.items()
        }

        # Schema info
        data["registered_kinds"] = self.schema.list_node_kinds()
        data["registered_edge_kinds"] = self.schema.list_edge_kinds()

        data["uptime_seconds"] = time.time() - self._start_time

        if self._wal_manager is not None:
            data["wal_replay_errors"] = self._wal_manager.replay_error_count
            if self._wal_manager.replay_error_count > 0:
                data["wal_replay_error_details"] = self._wal_manager.replay_errors

        # Accurate measurement
        try:
            from graphstore.core.memory import measure as measure_memory
            data["memory_measured"] = measure_memory(store, self._vector_store, self._document_store)
        except Exception:
            pass

        return Result(kind="status", data=data, count=1)

    def _retain(self, q: SysRetain) -> Result:
        """SYS RETAIN: apply blob retention policy based on node age. Vectorized."""
        store = self.store
        n = store._next_slot
        if n == 0:
            return Result(kind="ok", data={"archived": 0, "blob_deleted": 0}, count=0)

        archive_cutoff_ms = int(time.time() * 1000) - self._retention.get("blob_archive_days", 90) * 86400000
        delete_cutoff_ms = int(time.time() * 1000) - self._retention.get("blob_delete_days", 365) * 86400000

        live = store.compute_live_mask(n)
        created_col = store.columns.get_column("__created_at__", n)
        blob_col = store.columns.get_column("__blob_state__", n)

        if created_col is None or blob_col is None:
            return Result(kind="ok", data={"archived": 0, "blob_deleted": 0}, count=0)

        created_data, created_pres, _ = created_col
        blob_data, blob_pres, _ = blob_col

        warm_id = store.string_table.intern("warm") if "warm" in store.string_table else None
        if warm_id is None:
            return Result(kind="ok", data={"archived": 0, "blob_deleted": 0}, count=0)

        archived_id = store.string_table.intern("archived") if "archived" in store.string_table else None
        deleted_str_id = store.string_table.intern("deleted")

        # Vectorized: find warm nodes older than archive_cutoff
        eligible = live & created_pres & blob_pres
        warm_mask = eligible & (blob_data == warm_id) & (created_data < archive_cutoff_ms)
        to_archive = np.nonzero(warm_mask)[0]

        if len(to_archive) > 0:
            archived_str_id = store.string_table.intern("archived")
            store.columns._columns["__blob_state__"][to_archive] = archived_str_id
        archived_count = len(to_archive)

        # Vectorized: find archived nodes older than delete_cutoff
        deleted_count = 0
        if archived_id is not None:
            archive_mask = eligible & (blob_data == archived_id) & (created_data < delete_cutoff_ms)
            to_delete = np.nonzero(archive_mask)[0]
            if len(to_delete) > 0:
                if self._document_store:
                    for slot in to_delete:
                        self._document_store.delete_document(int(slot))
                store.columns._columns["__blob_state__"][to_delete] = deleted_str_id
            deleted_count = len(to_delete)

        return Result(kind="ok", data={
            "archived": archived_count,
            "blob_deleted": deleted_count,
        }, count=archived_count + deleted_count)

    def _health(self, q: SysHealth) -> Result:
        """SYS HEALTH: pressure metrics for self-balancing decisions."""
        from graphstore.core.optimizer import health_check, needs_optimization
        health = health_check(self.store, self._vector_store, self._document_store)
        try:
            from graphstore.core.memory import measure as measure_memory
            mem = measure_memory(self.store, self._vector_store, self._document_store)
            health["memory_total"] = mem["total"]
            health["memory_ceiling"] = self.store._ceiling_bytes
            health["memory_utilization"] = round(mem["total"] / max(self.store._ceiling_bytes, 1), 3)
        except Exception:
            pass
        health["recommended"] = needs_optimization(health)
        return Result(kind="health", data=health, count=1)

    def _optimize(self, q: SysOptimize) -> Result:
        """SYS OPTIMIZE: run optimization operations under exclusive lock."""
        from graphstore.core.optimizer import (
            optimize_all, compact_tombstones, compact_tombstones_safe,
            gc_strings, defrag_edges, cleanup_vectors, sweep_orphans, clear_caches,
        )
        target = q.target
        if target is None:
            data = optimize_all(
                self.store, self._vector_store, self._document_store,
                schema=self.schema, conn=self.conn,
            )
        elif target == "COMPACT":
            data = compact_tombstones_safe(
                self.store, self.schema, self.conn,
                self._vector_store, self._document_store,
            )
        elif target == "STRINGS":
            data = gc_strings(self.store)
        elif target == "EDGES":
            data = defrag_edges(self.store)
        elif target == "VECTORS":
            data = cleanup_vectors(self.store, self._vector_store)
        elif target == "BLOBS":
            data = sweep_orphans(self.store, self._document_store)
        elif target == "CACHE":
            data = clear_caches(self.store)
        else:
            raise GraphStoreError(f"Unknown optimize target: {target}")
        return Result(kind="ok", data=data, count=1)

    def _evict(self, q: SysEvict) -> Result:
        """SYS EVICT: emergency eviction of oldest nodes to free memory."""
        from graphstore.core.optimizer import evict_oldest, evict_by_count

        protected = self._protected_kinds

        if q.limit:
            data = evict_by_count(
                self.store, q.limit.value, self._vector_store, self._document_store,
                protected_kinds=protected,
            )
        else:
            target = int(self.store._ceiling_bytes * self._eviction_target_ratio)
            data = evict_oldest(
                self.store, target, self._vector_store, self._document_store,
                protected_kinds=protected,
            )

        return Result(kind="ok", data=data, count=data["evicted"])

    def _log(self, q: SysLog) -> Result:
        """SYS LOG: query the enriched query log."""
        if not self.conn:
            return Result(kind="log_entries", data=[], count=0)

        sql = "SELECT id, timestamp, query, elapsed_us, result_count, error, tag, trace_id, source, phase FROM query_log"
        params: list = []
        conditions = []

        if q.trace_id:
            conditions.append("trace_id = ?")
            params.append(q.trace_id)
        if q.since:
            import datetime
            dt = datetime.datetime.fromisoformat(q.since)
            conditions.append("timestamp >= ?")
            params.append(dt.timestamp())

        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        sql += " ORDER BY timestamp DESC"

        if q.limit:
            sql += " LIMIT ?"
            params.append(q.limit.value)
        else:
            sql += " LIMIT 50"

        rows = self.conn.execute(sql, params).fetchall()
        entries = [
            {
                "id": r[0], "timestamp": r[1], "query": r[2],
                "elapsed_us": r[3], "result_count": r[4], "error": r[5],
                "tag": r[6], "trace_id": r[7], "source": r[8], "phase": r[9],
            }
            for r in rows
        ]
        return Result(kind="log_entries", data=entries, count=len(entries))

    def _cron_add(self, q: SysCronAdd) -> Result:
        if not self._cron:
            raise GraphStoreError("CRON not configured. Use GraphStore(queued=True)")
        data = self._cron.add(q.name, q.schedule, q.query)
        return Result(kind="ok", data=data, count=1)

    def _cron_delete(self, q: SysCronDelete) -> Result:
        if not self._cron:
            raise GraphStoreError("CRON not configured")
        self._cron.delete(q.name)
        return Result(kind="ok", data={"deleted": q.name}, count=1)

    def _cron_enable(self, q: SysCronEnable) -> Result:
        if not self._cron:
            raise GraphStoreError("CRON not configured")
        self._cron.enable(q.name)
        return Result(kind="ok", data={"enabled": q.name}, count=1)

    def _cron_disable(self, q: SysCronDisable) -> Result:
        if not self._cron:
            raise GraphStoreError("CRON not configured")
        self._cron.disable(q.name)
        return Result(kind="ok", data={"disabled": q.name}, count=1)

    def _cron_list(self, q: SysCronList) -> Result:
        if not self._cron:
            raise GraphStoreError("CRON not configured")
        jobs = self._cron.list_jobs()
        return Result(kind="cron_jobs", data=jobs, count=len(jobs))

    def _cron_run(self, q: SysCronRun) -> Result:
        if not self._cron:
            raise GraphStoreError("CRON not configured")
        self._cron.run_now(q.name)
        return Result(kind="ok", data={"triggered": q.name}, count=1)

    # --- Evolution handlers ---

    def _get_engine(self):
        if self._evolution_engine is None:
            raise GraphStoreError("Evolution engine not initialized")
        return self._evolution_engine

    def _evolve_rule(self, q: SysEvolveRule) -> Result:
        from graphstore.evolve import EvolutionRule, Condition, Action
        engine = self._get_engine()
        rule = EvolutionRule(
            name=q.name,
            cooldown=q.cooldown,
            priority=q.priority,
        )
        for c in q.conditions:
            rule.conditions.append(Condition(
                signal=c["signal"],
                operator=c["operator"],
                value=c["value"],
            ))
        for a in q.actions:
            rule.actions.append(Action(
                kind=a["kind"],
                param=a["param"],
                value=a.get("value"),
                delta=a.get("delta", 0.0),
                until=a.get("until"),
            ))
        err = engine.add_rule(rule)
        if err:
            return Result(kind="error", data=err, count=0)
        return Result(kind="ok", data={"created": q.name}, count=1)

    def _evolve_list(self, q: SysEvolveList) -> Result:
        engine = self._get_engine()
        rules = engine.list_rules()
        return Result(kind="rules", data=rules, count=len(rules))

    def _evolve_show(self, q: SysEvolveShow) -> Result:
        engine = self._get_engine()
        rule = engine.get_rule(q.name)
        if rule is None:
            return Result(kind="error", data=f"rule not found: '{q.name}'", count=0)
        return Result(kind="rule", data=rule, count=1)

    def _evolve_enable(self, q: SysEvolveEnable) -> Result:
        engine = self._get_engine()
        if not engine.enable_rule(q.name):
            return Result(kind="error", data=f"rule not found: '{q.name}'", count=0)
        return Result(kind="ok", data={"enabled": q.name}, count=1)

    def _evolve_disable(self, q: SysEvolveDisable) -> Result:
        engine = self._get_engine()
        if not engine.disable_rule(q.name):
            return Result(kind="error", data=f"rule not found: '{q.name}'", count=0)
        return Result(kind="ok", data={"disabled": q.name}, count=1)

    def _evolve_delete(self, q: SysEvolveDelete) -> Result:
        engine = self._get_engine()
        if not engine.delete_rule(q.name):
            return Result(kind="error", data=f"rule not found: '{q.name}'", count=0)
        return Result(kind="ok", data={"deleted": q.name}, count=1)

    def _evolve_history(self, q: SysEvolveHistory) -> Result:
        engine = self._get_engine()
        entries = engine.history(limit=q.limit)
        return Result(kind="history", data=entries, count=len(entries))

    def _evolve_reset(self, q: SysEvolveReset) -> Result:
        engine = self._get_engine()
        engine.reset()
        return Result(kind="ok", data={"reset": True}, count=0)
