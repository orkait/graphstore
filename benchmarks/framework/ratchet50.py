"""50-question validated semantic ratchet test.

Only includes questions where the gold answer keyword EXISTS in the ingested data.
GraphStore is populated ONCE, kept alive, all 50 questions tested against it.

Usage: uv run python3 -m benchmarks.framework.ratchet50
"""
import os, logging, random
logging.getLogger('graphstore.events').setLevel(logging.WARNING)
os.environ.setdefault('GRAPHSTORE_MODEL_CACHE_DIR', '/tmp/gs_models')

from .datasets import load_locomo
from .adapters.graphstore_ import GraphStoreAdapter
from collections import Counter

SKIP_WORDS = frozenset({
    'likely', 'since', 'because', 'though', 'about', 'their',
    'would', 'there', 'being', 'could', 'should', 'which',
    'where', 'still', 'after', 'before', 'other', 'these',
    'something', 'years', 'week', 'some', 'that', 'this',
    'than', 'them', 'from', 'with', 'have', 'more', 'also',
})


def run(label="test"):
    ds = load_locomo('/tmp/locomo', max_conversations=1)

    # Step 1: Populate GraphStore
    config = {
        'embedder': 'installed', 'embedder_model': 'jina-v5-nano-retrieval',
        'embedder_cache_dir': '/tmp/gs_models', 'embedder_gpu': True,
        'ceiling_mb': 512,
    }
    adapter = GraphStoreAdapter(config=config)
    adapter.reset()
    for sess in ds.records[0].sessions:
        adapter.ingest(sess)
    gs = adapter._gs

    # Step 2: Get all content for validation
    all_content = ' '.join(
        n.get('content', '').lower()
        for n in gs.execute('NODES WHERE kind = "message"').data
    )

    # Step 3: Build validated test cases (keyword MUST exist in data)
    candidates = []
    for rec in ds.records:
        gold = rec.question.gold_answers[0] if rec.question.gold_answers else ''
        if not gold or len(gold) < 3:
            continue
        words = [w.strip('.,;:!?()"') for w in gold.split() if len(w.strip('.,;:!?()"')) >= 4]
        words = [w for w in words if w.lower() not in SKIP_WORDS]
        if not words:
            continue
        keyword = max(words, key=len).lower()
        if keyword in all_content:
            candidates.append((rec.question.question, keyword, rec.question.category))

    random.seed(42)
    random.shuffle(candidates)
    tests = candidates[:50]

    # Step 4: Run retrieval against the SAME populated GraphStore
    score = 0
    by_cat = {}
    for question, keyword, cat in tests:
        q = question.replace('"', '\\"')
        r = gs.execute(f'REMEMBER "{q}" LIMIT 10 WHERE kind = "message"')
        top10 = ' '.join(n.get('content', '') for n in r.data[:10]).lower()
        found = keyword in top10
        if found:
            score += 1
        by_cat.setdefault(cat, [0, 0])
        by_cat[cat][0] += 1 if found else 0
        by_cat[cat][1] += 1

    print(f'{label}: {score}/{len(tests)} ({100*score//len(tests)}%)')
    for cat in sorted(by_cat):
        h, total = by_cat[cat]
        print(f'  {cat:<20} {h}/{total}')
    adapter.close()
    return score


if __name__ == '__main__':
    import sys
    label = sys.argv[1] if len(sys.argv) > 1 else "test"
    run(label)
