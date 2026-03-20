"""Evidence-backed benchmark: dict-based vs columnar-only storage.

Measures actual costs of flipping source of truth from list[dict] to numpy columns.
Tests at 10k, 50k, and 100k node scales.
"""

import time
import sys
import json
import numpy as np
from graphstore import GraphStore

SCALES = [10_000, 50_000, 100_000]
ITERS = 20


def timed(fn, iters=ITERS):
    """Return median time in microseconds."""
    times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        result = fn()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000)  # ns -> us
    times.sort()
    return times[len(times) // 2], result


def populate(n):
    """Create a graph with n nodes, registered schema, columnarized fields."""
    g = GraphStore(ceiling_mb=2048)
    g.execute('SYS REGISTER NODE KIND "item" REQUIRED name:string, score:float, importance:float OPTIONAL tag:string')

    # Batch insert for speed
    for i in range(n):
        g.execute(f'CREATE NODE "n{i}" kind = "item" name = "node_{i}" score = {i * 0.1} importance = {(i % 100) * 0.01} tag = "tag_{i % 50}"')

    return g


def bench_materialization(g, n):
    """Current: return node_data[slot]. Columnar: build dict from columns."""
    store = g._store

    # Current: dict return
    def dict_materialize():
        results = []
        for slot in range(min(10, n)):
            d = store.node_data[slot]
            if d is not None:
                results.append({
                    "id": store.string_table.lookup(int(store.node_ids[slot])),
                    "kind": store.string_table.lookup(int(store.node_kinds[slot])),
                    **d
                })
        return results

    # Columnar: build dict from columns
    def col_materialize():
        results = []
        cols = store.columns
        fields = list(cols._columns.keys())
        for slot in range(min(10, n)):
            d = {"id": store.string_table.lookup(int(store.node_ids[slot])),
                 "kind": store.string_table.lookup(int(store.node_kinds[slot]))}
            for f in fields:
                if cols._presence[f][slot]:
                    dtype = cols._dtypes[f]
                    raw = cols._columns[f][slot]
                    if dtype == "int32_interned":
                        d[f] = store.string_table.lookup(int(raw))
                    elif dtype == "float64":
                        d[f] = float(raw)
                    elif dtype == "int64":
                        d[f] = int(raw)
            results.append(d)
        return results

    t_dict, _ = timed(dict_materialize)
    t_col, _ = timed(col_materialize)
    return t_dict, t_col


def bench_materialize_bulk(g, n, count=1000):
    """Materialization at LIMIT 1000."""
    store = g._store
    limit = min(count, n)
    fields = list(store.columns._columns.keys())

    def dict_materialize():
        results = []
        for slot in range(limit):
            d = store.node_data[slot]
            if d is not None:
                results.append({
                    "id": store.string_table.lookup(int(store.node_ids[slot])),
                    "kind": store.string_table.lookup(int(store.node_kinds[slot])),
                    **d
                })
        return len(results)

    def col_materialize():
        results = []
        cols = store.columns
        for slot in range(limit):
            d = {"id": store.string_table.lookup(int(store.node_ids[slot])),
                 "kind": store.string_table.lookup(int(store.node_kinds[slot]))}
            for f in fields:
                if cols._presence[f][slot]:
                    dtype = cols._dtypes[f]
                    raw = cols._columns[f][slot]
                    if dtype == "int32_interned":
                        d[f] = store.string_table.lookup(int(raw))
                    elif dtype == "float64":
                        d[f] = float(raw)
                    elif dtype == "int64":
                        d[f] = int(raw)
            results.append(d)
        return len(results)

    t_dict, _ = timed(dict_materialize, iters=5)
    t_col, _ = timed(col_materialize, iters=5)
    return t_dict, t_col


def bench_contains(g, n):
    """CONTAINS: dict field access vs de-intern + check."""
    store = g._store
    target = "node_42"
    limit = min(n, store._next_slot)

    # Dict path
    def dict_contains():
        hits = 0
        for slot in range(limit):
            d = store.node_data[slot]
            if d is not None:
                val = d.get("name", "")
                if target in str(val):
                    hits += 1
        return hits

    # Column de-intern path
    def col_contains():
        hits = 0
        name_col = store.columns._columns["name"]
        pres = store.columns._presence["name"]
        for slot in range(limit):
            if pres[slot]:
                str_val = store.string_table.lookup(int(name_col[slot]))
                if target in str_val:
                    hits += 1
        return hits

    t_dict, r1 = timed(dict_contains, iters=5)
    t_col, r2 = timed(col_contains, iters=5)
    assert r1 == r2, f"Mismatch: {r1} vs {r2}"
    return t_dict, t_col


def bench_write(g, n):
    """Single node write: dual-write (current) vs column-only."""
    store = g._store
    slot = 0
    data = {"name": "updated", "score": 99.9, "importance": 0.5, "tag": "new"}

    # Current: dict + column dual-write
    def dict_write():
        store.node_data[slot].update(data)
        store.columns.set(slot, data)

    # Column-only write
    def col_write():
        store.columns.set(slot, data)

    t_dict, _ = timed(dict_write)
    t_col, _ = timed(col_write)
    return t_dict, t_col


def bench_rollback_copy(g, n):
    """Batch rollback state copy: dict list vs numpy arrays."""
    store = g._store
    ns = store._next_slot

    # Current: deep copy dicts
    def dict_copy():
        saved = [dict(d) if d is not None else None for d in store.node_data[:ns]]
        return len(saved)

    # Column: copy numpy arrays
    def col_copy():
        saved = {}
        for f in store.columns._columns:
            saved[f] = (store.columns._columns[f][:ns].copy(),
                       store.columns._presence[f][:ns].copy())
        return len(saved)

    t_dict, _ = timed(dict_copy, iters=5)
    t_col, _ = timed(col_copy, iters=5)
    return t_dict, t_col


def bench_memory(g, n):
    """Memory usage: dict vs columns."""
    store = g._store
    ns = store._next_slot

    # Dict memory (approximate)
    dict_bytes = 0
    for slot in range(ns):
        d = store.node_data[slot]
        if d is not None:
            dict_bytes += sys.getsizeof(d)
            for k, v in d.items():
                dict_bytes += sys.getsizeof(k) + sys.getsizeof(v)

    # Column memory
    col_bytes = store.columns.memory_bytes

    return dict_bytes, col_bytes


def bench_count_columnar_vs_dict(g, n):
    """COUNT WHERE score > X: column mask vs dict predicate."""
    # Column path (via DSL)
    def col_count():
        return g.execute(f'COUNT NODES WHERE kind = "item" AND score > {n * 0.05}')

    # Force dict path: use CONTAINS which falls back to dict
    # Actually let's just measure the raw operations
    store = g._store
    threshold = n * 0.05
    ns = store._next_slot

    def dict_count():
        count = 0
        for slot in range(ns):
            d = store.node_data[slot]
            if d is not None and d.get("score", 0) > threshold:
                count += 1
        return count

    def col_count_raw():
        mask = store.columns.get_mask("score", ">", threshold, ns)
        if mask is not None:
            return int(np.sum(mask))
        return 0

    t_dict, r1 = timed(dict_count)
    t_col, r2 = timed(col_count_raw)
    return t_dict, t_col


def bench_group_by_simulation(g, n):
    """Simulated GROUP BY tag SELECT COUNT(), AVG(score): dict vs numpy."""
    store = g._store
    ns = store._next_slot

    # Dict path
    def dict_groupby():
        groups = {}
        for slot in range(ns):
            d = store.node_data[slot]
            if d is not None:
                tag = d.get("tag")
                if tag:
                    if tag not in groups:
                        groups[tag] = [0, 0.0]
                    groups[tag][0] += 1
                    groups[tag][1] += d.get("score", 0)
        return {k: (v[0], v[1] / v[0]) for k, v in groups.items()}

    # Column path
    def col_groupby():
        tag_col = store.columns._columns["tag"][:ns]
        tag_pres = store.columns._presence["tag"][:ns]
        score_col = store.columns._columns["score"][:ns]

        mask = tag_pres
        filtered_tags = tag_col[mask]
        filtered_scores = score_col[mask]

        unique_tags, inverse = np.unique(filtered_tags, return_inverse=True)
        counts = np.bincount(inverse)
        sums = np.zeros(len(unique_tags), dtype=np.float64)
        np.add.at(sums, inverse, filtered_scores)
        avgs = sums / np.maximum(counts, 1)

        return {int(t): (int(c), float(a)) for t, c, a in zip(unique_tags, counts, avgs)}

    t_dict, r1 = timed(dict_groupby, iters=5)
    t_col, r2 = timed(col_groupby, iters=5)
    return t_dict, t_col


def main():
    for n in SCALES:
        print(f"\n{'='*70}")
        print(f"  SCALE: {n:,} nodes")
        print(f"{'='*70}")

        print(f"  Populating {n:,} nodes...", end=" ", flush=True)
        t0 = time.time()
        g = populate(n)
        print(f"done in {time.time() - t0:.1f}s")

        # Materialization LIMIT 10
        t_dict, t_col = bench_materialization(g, n)
        pct = ((t_col - t_dict) / t_dict) * 100
        print(f"\n  Materialize LIMIT 10:")
        print(f"    dict:   {t_dict:>10.0f} μs")
        print(f"    column: {t_col:>10.0f} μs  ({pct:+.0f}%)")

        # Materialization LIMIT 1000
        t_dict, t_col = bench_materialize_bulk(g, n)
        pct = ((t_col - t_dict) / t_dict) * 100
        print(f"\n  Materialize LIMIT 1000:")
        print(f"    dict:   {t_dict:>10.0f} μs")
        print(f"    column: {t_col:>10.0f} μs  ({pct:+.0f}%)")

        # CONTAINS
        t_dict, t_col = bench_contains(g, n)
        pct = ((t_col - t_dict) / t_dict) * 100
        print(f"\n  CONTAINS scan (full):")
        print(f"    dict:   {t_dict:>10.0f} μs")
        print(f"    column: {t_col:>10.0f} μs  ({pct:+.0f}%)")

        # Write
        t_dict, t_col = bench_write(g, n)
        pct = ((t_col - t_dict) / t_dict) * 100
        print(f"\n  Single node write:")
        print(f"    dual:   {t_dict:>10.0f} μs")
        print(f"    col:    {t_col:>10.0f} μs  ({pct:+.0f}%)")

        # Rollback copy
        t_dict, t_col = bench_rollback_copy(g, n)
        pct = ((t_col - t_dict) / t_dict) * 100
        print(f"\n  Batch rollback state copy:")
        print(f"    dict:   {t_dict:>10.0f} μs")
        print(f"    column: {t_col:>10.0f} μs  ({pct:+.0f}%)")

        # COUNT with filter
        t_dict, t_col = bench_count_columnar_vs_dict(g, n)
        speedup = t_dict / max(t_col, 1)
        print(f"\n  COUNT WHERE score > threshold:")
        print(f"    dict:   {t_dict:>10.0f} μs")
        print(f"    column: {t_col:>10.0f} μs  ({speedup:.0f}x faster)")

        # GROUP BY simulation
        t_dict, t_col = bench_group_by_simulation(g, n)
        speedup = t_dict / max(t_col, 1)
        print(f"\n  GROUP BY tag SELECT COUNT(), AVG(score):")
        print(f"    dict:   {t_dict:>10.0f} μs")
        print(f"    column: {t_col:>10.0f} μs  ({speedup:.0f}x faster)")

        # Memory
        dict_bytes, col_bytes = bench_memory(g, n)
        ratio = dict_bytes / max(col_bytes, 1)
        print(f"\n  Memory usage:")
        print(f"    dict:   {dict_bytes / 1024 / 1024:>10.1f} MB")
        print(f"    column: {col_bytes / 1024 / 1024:>10.1f} MB  ({ratio:.0f}x less)")

        g.close()


if __name__ == "__main__":
    main()
