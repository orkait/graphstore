"""Performance analysis: Agentic Brain DB features.

Benchmarks all new capabilities at 10k, 50k, and 100k node scales.
Measures: columnar ops, aggregation, recall, beliefs, TTL, contradictions,
snapshot/rollback, counterfactual, context isolation.
"""

import time
import numpy as np
from graphstore import GraphStore


SCALES = [10_000, 50_000, 100_000]
ITERS = 20


def timed(fn, iters=ITERS):
    """Return median time in microseconds."""
    times = []
    result = None
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        result = fn()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000)
    times.sort()
    return times[len(times) // 2], result


def populate(n):
    """Create a graph with n nodes, edges, registered schema."""
    g = GraphStore(ceiling_mb=4096)
    g.execute('SYS REGISTER NODE KIND "memory" REQUIRED topic:string, importance:float, score:int')
    g.execute('SYS REGISTER NODE KIND "fact" REQUIRED topic:string, confidence:float')

    for i in range(n):
        kind = "memory" if i % 2 == 0 else "fact"
        topic = f"topic_{i % 50}"
        if kind == "memory":
            g.execute(f'CREATE NODE "n{i}" kind = "memory" topic = "{topic}" importance = {(i % 100) * 0.01} score = {i}')
        else:
            g.execute(f'CREATE NODE "n{i}" kind = "fact" topic = "{topic}" confidence = {(i % 100) * 0.01}')

    # Create edges (10% of nodes connected)
    edge_count = n // 10
    for i in range(edge_count):
        src = f"n{i}"
        tgt = f"n{(i * 7 + 3) % n}"
        try:
            g.execute(f'CREATE EDGE "{src}" -> "{tgt}" kind = "related"')
        except Exception:
            pass

    return g


def section(title):
    print(f"\n  {'─' * 50}")
    print(f"  {title}")
    print(f"  {'─' * 50}")


def bench_phase1(g, n):
    """Phase 1: Infrastructure - columnar reads, timestamps, live_mask."""
    section("Phase 1: Infrastructure")

    # Point lookup
    t, _ = timed(lambda: g.execute('NODE "n0"'))
    print(f"  NODE point lookup:              {t:>10.0f} μs")

    # WHERE filter
    t, r = timed(lambda: g.execute('NODES WHERE kind = "memory" AND importance > 0.5 LIMIT 10'))
    print(f"  WHERE + LIMIT 10:               {t:>10.0f} μs  ({r.count} results)")

    # COUNT
    t, r = timed(lambda: g.execute('COUNT NODES WHERE kind = "memory"'))
    print(f"  COUNT WHERE kind=memory:        {t:>10.0f} μs  ({r.data} nodes)")

    # COUNT with compound filter
    t, r = timed(lambda: g.execute('COUNT NODES WHERE kind = "memory" AND importance > 0.5'))
    print(f"  COUNT compound filter:          {t:>10.0f} μs  ({r.data} nodes)")

    # ORDER BY LIMIT
    t, _ = timed(lambda: g.execute('NODES WHERE kind = "memory" ORDER BY score DESC LIMIT 10'))
    print(f"  ORDER BY LIMIT 10:              {t:>10.0f} μs")

    # Timestamp filter
    t, r = timed(lambda: g.execute('NODES WHERE __created_at__ > NOW() - 1d LIMIT 10'))
    print(f"  WHERE __created_at__ > NOW()-1d: {t:>10.0f} μs")

    # Memory usage
    col_bytes = g._store.columns.memory_bytes
    node_count = g._store._count
    per_node = col_bytes / max(node_count, 1)
    print(f"  Memory: {col_bytes / 1024 / 1024:.1f} MB total, {per_node:.0f} bytes/node")


def bench_phase2(g, n):
    """Phase 2: Aggregations."""
    section("Phase 2: Aggregations")

    # GROUP BY COUNT
    t, r = timed(lambda: g.execute('AGGREGATE NODES WHERE kind = "memory" GROUP BY topic SELECT COUNT()'))
    print(f"  GROUP BY topic COUNT():         {t:>10.0f} μs  ({r.count} groups)")

    # GROUP BY AVG
    t, r = timed(lambda: g.execute('AGGREGATE NODES WHERE kind = "memory" GROUP BY topic SELECT AVG(importance)'))
    print(f"  GROUP BY topic AVG(importance): {t:>10.0f} μs")

    # GROUP BY multiple funcs
    t, _ = timed(lambda: g.execute(
        'AGGREGATE NODES WHERE kind = "memory" GROUP BY topic SELECT COUNT(), SUM(score), AVG(importance), MIN(score), MAX(score)'))
    print(f"  GROUP BY 5 agg funcs:           {t:>10.0f} μs")

    # Global aggregate (no GROUP BY)
    t, r = timed(lambda: g.execute('AGGREGATE NODES WHERE kind = "memory" SELECT COUNT(), SUM(score)'))
    print(f"  Global SUM + COUNT:             {t:>10.0f} μs")

    # HAVING
    t, r = timed(lambda: g.execute(
        'AGGREGATE NODES WHERE kind = "memory" GROUP BY topic SELECT COUNT() HAVING COUNT() > 50'))
    print(f"  GROUP BY + HAVING:              {t:>10.0f} μs  ({r.count} groups)")


def bench_phase3(g, n):
    """Phase 3: Belief operations."""
    section("Phase 3: Belief Operations")

    # ASSERT
    t, _ = timed(lambda: g.execute('ASSERT "bench_fact" kind = "fact" topic = "bench" confidence = 0.95 CONFIDENCE 0.95 SOURCE "bench"'), iters=5)
    print(f"  ASSERT (upsert):                {t:>10.0f} μs")

    # RETRACT + un-retract for repeated measurement
    def retract_cycle():
        g.execute('RETRACT "bench_fact"')
        g._store.columns.set_reserved(
            g._store.id_to_slot[g._store.string_table.intern("bench_fact")],
            "__retracted__", 0)
    t, _ = timed(retract_cycle, iters=5)
    print(f"  RETRACT + restore:              {t:>10.0f} μs")

    # UPDATE NODES WHERE (bulk)
    t, r = timed(lambda: g.execute('UPDATE NODES WHERE kind = "fact" SET confidence = 0.5'), iters=5)
    print(f"  UPDATE NODES WHERE (bulk):      {t:>10.0f} μs  ({r.data['updated']} updated)")

    # SYS CONTRADICTIONS
    # First create some contradictions
    g.execute('CREATE NODE "contra_a" kind = "fact" topic = "test_contra" confidence = 0.9')
    g.execute('CREATE NODE "contra_b" kind = "fact" topic = "test_contra" confidence = 0.1')
    t, r = timed(lambda: g.execute('SYS CONTRADICTIONS WHERE kind = "fact" FIELD confidence GROUP BY topic'))
    print(f"  SYS CONTRADICTIONS:             {t:>10.0f} μs  ({r.count} contradictions)")
    # Cleanup
    g.execute('DELETE NODE "contra_a"')
    g.execute('DELETE NODE "contra_b"')

    # MERGE
    g.execute('CREATE NODE "merge_src" kind = "memory" topic = "merge" importance = 0.5 score = 1')
    g.execute('CREATE NODE "merge_tgt" kind = "memory" topic = "merge" importance = 0.9 score = 2')
    t, r = timed(lambda: None)  # Can't repeat merge, measure single
    t0 = time.perf_counter_ns()
    r = g.execute('MERGE NODE "merge_src" INTO "merge_tgt"')
    t1 = time.perf_counter_ns()
    t = (t1 - t0) / 1000
    print(f"  MERGE NODE:                     {t:>10.0f} μs  ({r.data['fields_merged']} fields, {r.data['edges_rewired']} edges)")
    g.execute('DELETE NODE "merge_tgt"')


def bench_phase3_ttl(g, n):
    """Phase 3: TTL."""
    section("Phase 3: TTL")

    # CREATE with EXPIRES
    t, _ = timed(lambda: g.execute('UPSERT NODE "ttl_bench" kind = "memory" topic = "ttl" importance = 0.1 score = 0 EXPIRES IN 3600s'), iters=5)
    print(f"  CREATE with EXPIRES IN:         {t:>10.0f} μs")

    # SYS EXPIRE (nothing to expire since TTL is future)
    t, r = timed(lambda: g.execute('SYS EXPIRE'))
    print(f"  SYS EXPIRE (0 expired):         {t:>10.0f} μs")

    # Force some expirations
    for i in range(100):
        g.execute(f'CREATE NODE "exp_{i}" kind = "memory" topic = "expire" importance = 0.1 score = 0 EXPIRES IN 0s')
    time.sleep(0.01)
    t0 = time.perf_counter_ns()
    r = g.execute('SYS EXPIRE')
    t1 = time.perf_counter_ns()
    t = (t1 - t0) / 1000
    print(f"  SYS EXPIRE (100 expired):       {t:>10.0f} μs")


def bench_phase4(g, n):
    """Phase 4: Graph intelligence."""
    section("Phase 4: Graph Intelligence")

    # RECALL
    t, r = timed(lambda: g.execute('RECALL FROM "n0" DEPTH 1 LIMIT 10'), iters=10)
    print(f"  RECALL DEPTH 1 LIMIT 10:        {t:>10.0f} μs  ({r.count} results)")

    t, r = timed(lambda: g.execute('RECALL FROM "n0" DEPTH 2 LIMIT 10'), iters=10)
    print(f"  RECALL DEPTH 2 LIMIT 10:        {t:>10.0f} μs  ({r.count} results)")

    t, r = timed(lambda: g.execute('RECALL FROM "n0" DEPTH 3 LIMIT 10'), iters=10)
    print(f"  RECALL DEPTH 3 LIMIT 10:        {t:>10.0f} μs  ({r.count} results)")

    # RECALL with WHERE
    t, r = timed(lambda: g.execute('RECALL FROM "n0" DEPTH 2 LIMIT 10 WHERE kind = "memory"'), iters=10)
    print(f"  RECALL DEPTH 2 + WHERE:         {t:>10.0f} μs  ({r.count} results)")

    # PROPAGATE
    t0 = time.perf_counter_ns()
    r = g.execute('PROPAGATE "n0" FIELD importance DEPTH 2')
    t1 = time.perf_counter_ns()
    t = (t1 - t0) / 1000
    print(f"  PROPAGATE DEPTH 2:              {t:>10.0f} μs  ({r.data['updated']} updated)")

    # COUNTERFACTUAL
    t, r = timed(lambda: g.execute('WHAT IF RETRACT "n0"'), iters=10)
    print(f"  WHAT IF RETRACT:                {t:>10.0f} μs  ({r.data['affected_count']} affected)")

    # SYS SNAPSHOT
    t0 = time.perf_counter_ns()
    g.execute('SYS SNAPSHOT "perf_bench"')
    t1 = time.perf_counter_ns()
    t = (t1 - t0) / 1000
    print(f"  SYS SNAPSHOT:                   {t:>10.0f} μs")

    # SYS ROLLBACK
    g.execute('UPDATE NODE "n0" SET importance = 0.999')
    t0 = time.perf_counter_ns()
    g.execute('SYS ROLLBACK TO "perf_bench"')
    t1 = time.perf_counter_ns()
    t = (t1 - t0) / 1000
    print(f"  SYS ROLLBACK:                   {t:>10.0f} μs")

    # BIND CONTEXT
    t0 = time.perf_counter_ns()
    g.execute('BIND CONTEXT "bench_ctx"')
    t1 = time.perf_counter_ns()
    t = (t1 - t0) / 1000
    print(f"  BIND CONTEXT:                   {t:>10.0f} μs")

    # Create in context + NODES (scoped)
    g.execute('CREATE NODE "ctx_node" kind = "memory" topic = "ctx" importance = 0.5 score = 0')
    t, r = timed(lambda: g.execute('NODES LIMIT 10'))
    print(f"  NODES in context:               {t:>10.0f} μs  ({r.count} results)")

    # DISCARD CONTEXT
    t0 = time.perf_counter_ns()
    g.execute('DISCARD CONTEXT "bench_ctx"')
    t1 = time.perf_counter_ns()
    t = (t1 - t0) / 1000
    print(f"  DISCARD CONTEXT:                {t:>10.0f} μs")


def main():
    for n in SCALES:
        print(f"\n{'=' * 60}")
        print(f"  SCALE: {n:,} nodes, {n // 10:,} edges")
        print(f"{'=' * 60}")

        print(f"  Populating...", end=" ", flush=True)
        t0 = time.time()
        g = populate(n)
        print(f"done in {time.time() - t0:.1f}s")

        bench_phase1(g, n)
        bench_phase2(g, n)
        bench_phase3(g, n)
        bench_phase3_ttl(g, n)
        bench_phase4(g, n)

        g.close()


if __name__ == "__main__":
    main()
