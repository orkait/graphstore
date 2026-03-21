"""Performance benchmark: vector store + semantic search."""
import time
import numpy as np
from graphstore import GraphStore

SCALES = [1_000, 10_000, 100_000]
ITERS = 20
DIMS = 256


def timed(fn, iters=ITERS):
    times = []
    result = None
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        result = fn()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000)
    times.sort()
    return times[len(times) // 2], result


def section(title):
    print(f"\n  {'─' * 50}")
    print(f"  {title}")
    print(f"  {'─' * 50}")


def populate(n):
    g = GraphStore(embedder=None, ceiling_mb=4096)
    g._ensure_vector_store(DIMS)
    g.execute('SYS REGISTER NODE KIND "memory" REQUIRED score:int')
    g.execute('SYS REGISTER NODE KIND "fact" REQUIRED score:int')

    for i in range(n):
        kind = "memory" if i % 2 == 0 else "fact"
        g.execute(f'CREATE NODE "n{i}" kind = "{kind}" score = {i % 100}')

    # Add vectors (random, normalized)
    for i in range(n):
        vec = np.random.randn(DIMS).astype(np.float32)
        vec /= np.linalg.norm(vec)
        g._vector_store.add(i, vec)

    # Create some edges
    for i in range(n // 10):
        try:
            g.execute(f'CREATE EDGE "n{i}" -> "n{(i * 7 + 3) % n}" kind = "related"')
        except Exception:
            pass

    return g


def main():
    for n in SCALES:
        print(f"\n{'=' * 60}")
        print(f"  SCALE: {n:,} nodes, {DIMS}d vectors")
        print(f"{'=' * 60}")

        print(f"  Populating...", end=" ", flush=True)
        t0 = time.time()
        g = populate(n)
        print(f"done in {time.time() - t0:.1f}s")

        query_vec = np.random.randn(DIMS).astype(np.float32)
        query_vec /= np.linalg.norm(query_vec)

        section("Vector Operations")

        # Vector add (single)
        vec = np.random.randn(DIMS).astype(np.float32)
        vec /= np.linalg.norm(vec)
        t, _ = timed(lambda: g._vector_store.add(n + 999, vec))
        print(f"  Vector add (single):            {t:>10.0f} μs")

        # SIMILAR TO by vector (direct API to avoid DSL scientific notation issue)
        def search_10():
            n_ = g._store._next_slot
            mask = g._store.compute_live_mask(n_)
            slots, dists = g._vector_store.search(query_vec, k=10, mask=mask)
            return len(slots)
        t, cnt = timed(search_10)
        print(f"  SIMILAR TO [...] LIMIT 10:      {t:>10.0f} μs  ({cnt} results)")

        def search_5():
            n_ = g._store._next_slot
            mask = g._store.compute_live_mask(n_)
            slots, dists = g._vector_store.search(query_vec, k=5, mask=mask)
            return len(slots)
        t, cnt = timed(search_5)
        print(f"  SIMILAR TO [...] LIMIT 5:       {t:>10.0f} μs  ({cnt} results)")

        # SIMILAR TO with WHERE (mask + kind filter)
        def search_where():
            n_ = g._store._next_slot
            mask = g._store.compute_live_mask(n_)
            kind_mask = g._store.columns.get_mask("kind", "=", "memory", n_)
            if kind_mask is not None:
                mask = mask & kind_mask
            slots, dists = g._vector_store.search(query_vec, k=10, mask=mask)
            return len(slots)
        t, cnt = timed(search_where)
        print(f"  SIMILAR TO + WHERE:             {t:>10.0f} μs  ({cnt} results)")

        # SYS DUPLICATES
        if n <= 10_000:
            t, r = timed(lambda: g.execute('SYS DUPLICATES THRESHOLD 0.99'), iters=3)
            print(f"  SYS DUPLICATES (t=0.99):        {t:>10.0f} μs  ({r.count} pairs)")

        section("Memory")
        vs_mem = g._vector_store.memory_bytes
        col_mem = g._store.columns.memory_bytes
        print(f"  Vector index:                   {vs_mem / 1024 / 1024:>10.1f} MB")
        print(f"  Columnar store:                 {col_mem / 1024 / 1024:>10.1f} MB")
        print(f"  Per node (vector):              {vs_mem / n:>10.0f} bytes")
        print(f"  Per node (columns):             {col_mem / n:>10.0f} bytes")
        print(f"  Per node (total):               {(vs_mem + col_mem) / n:>10.0f} bytes")

        section("Comparison: Graph vs Vector Recall")

        # Graph RECALL
        t, r = timed(lambda: g.execute('RECALL FROM "n0" DEPTH 1 LIMIT 10'), iters=10)
        print(f"  RECALL DEPTH 1 LIMIT 10:        {t:>10.0f} μs  ({r.count} results)")

        # Vector SIMILAR
        t, cnt = timed(search_10)
        print(f"  SIMILAR TO LIMIT 10:            {t:>10.0f} μs  ({cnt} results)")

        # Columnar COUNT
        t, r = timed(lambda: g.execute('COUNT NODES WHERE kind = "memory"'))
        print(f"  COUNT WHERE:                    {t:>10.0f} μs  ({r.data} nodes)")

        # AGGREGATE
        t, r = timed(lambda: g.execute('AGGREGATE NODES SELECT COUNT(), SUM(score)'))
        print(f"  AGGREGATE SUM+COUNT:            {t:>10.0f} μs")

        if n <= 10_000:
            section("Persistence")
            import tempfile, os
            with tempfile.TemporaryDirectory() as tmp:
                g2 = GraphStore(path=tmp, embedder=None, ceiling_mb=4096)
                g2._vector_store = g._vector_store
                g2._store.vectors = g._vector_store
                # Measure checkpoint
                t0 = time.perf_counter_ns()
                g2.checkpoint()
                t1 = time.perf_counter_ns()
                print(f"  Checkpoint ({n:,} vectors):     {(t1-t0)/1000:>10.0f} μs")
                g2.close()

        g.close()


if __name__ == "__main__":
    main()
