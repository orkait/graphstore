---
name: graphstore-ingestion
description: How to inject data into graphstore correctly. Covers the three storage engines (graph, vector, document), schema-first ingestion, REMEMBER vs RECALL vs SIMILAR TO, timestamp pitfalls, BM25 FTS gotchas, benchmark-friendly patterns, and the anti-patterns that make graphstore underperform vector stores. Use whenever you are writing a graphstore adapter, ingesting a new dataset, debugging retrieval quality, or about to write CREATE NODE in a loop.
compatibility: Requires a local graphstore install. Works against any graphstore >= 0.3.0.
metadata:
  author: orkait
  version: "1.0"
---

# Graphstore ingestion

How to inject data into graphstore **correctly** so retrieval actually works. This skill exists because the naive "one CREATE NODE per item" approach throws away 70% of what graphstore can do, and you will under-perform a plain vector database if you write ingestion that way.

Read this before you touch an ingestion loop.

## When to use this skill

- You are writing a new adapter (benchmark, production, or agent integration)
- You are wondering why your REMEMBER / RECALL / SIMILAR TO queries return garbage
- You are about to write `CREATE NODE` in a for-loop
- You want to ingest a conversation dataset (LongMemEval, LoCoMo, etc.)
- You need to choose between REMEMBER, RECALL, SIMILAR TO, LEXICAL SEARCH
- You are debugging "my benchmark accuracy is lower than a plain vector store"

## Three storage engines unified by one DSL

graphstore is not one thing. It is three storage engines stitched together behind one query language, plus a pile of feature layers that reach into them. Treating it as "just a vector store" leaves most of the value on the table.

**The three storage engines** — each is an independent class with its own state and lifecycle:

1. **Graph engine** (`graphstore/core/`) — `CoreStore` holds columnar node arrays (numpy), sparse CSR edge matrices (scipy), a string intern table, and tombstone-based deletion. This is the data plane for structured fields + the relationship graph. Every field you set lives in a typed numpy column. `node_ids`, `node_kinds`, `id_to_slot`, `_edges_by_type` are the load-bearing structures.
2. **Vector engine** (`graphstore/vector/`) — `VectorStore` wraps a usearch HNSW index over `(slot, vector)` pairs with cosine metric. `VectorStore.search(query, k, mask)` returns top-k slot ids filtered by a boolean live mask.
3. **Document engine** (`graphstore/document/`) — `DocumentStore` is SQLite multi-table storage (`documents`, `summaries`, `doc_metadata`, `images`) with an FTS5 virtual table (`doc_fts`) over summaries. This is where BM25 lives. `put_summary(slot, text, ...)` writes both the row AND the FTS index.

A single node can live in **all three at once**: row in numpy columns + vector in usearch + entry in FTS5.

**Plus feature layers on top:**

- **DSL** (`graphstore/dsl/`) — Lark LALR(1) grammar compiled to 70+ AST dataclasses, handler-registry dispatch. The unified interface to all three engines.
- **Embedding** (`graphstore/embedding/`) — pluggable embedder protocol (model2vec default, FastEmbed for BGE/e5/mxbai/nomic, OnnxHF for EmbeddingGemma/Harrier).
- **Beliefs** — `ASSERT` / `RETRACT` / `PROPAGATE` write reserved columns (`__confidence__`, `__retracted__`, `__source__`) on the Graph engine.
- **Evolution** (`graphstore/evolve.py`) — `EvolutionEngine`, opt-in. WHEN/THEN rules that self-tune graphstore's own runtime parameters based on live signals.
- **Ingest pipeline** (`graphstore/ingest/`) — file-to-graph routing (MarkItDown → PyMuPDF4LLM → Docling), chunker, vision. Used by the `INGEST` DSL verb.
- **Algos** (`graphstore/algos/`) — 17 pure numpy/scipy primitives under a strict purity gate. Tunable in isolation.

**Optional subsystems** (off by default): **Vault** (markdown notebook, `graphstore/vault/`) and **Voice** (Moonshine STT + Piper TTS).

## The mental model

graphstore stores three independent things:

```
Nodes  → numpy columns (Core engine)
Edges  → scipy CSR matrices (Core engine)
Vectors → usearch HNSW (Vector engine)
Text   → SQLite FTS5 (Document engine)
```

A single node can live in all four at once. When you CREATE a node with `EMBED content`, it gets a row in the numpy columns AND a vector in usearch AND (if you go through INGEST or `put_summary`) a row in doc_fts. Different retrieval primitives hit different layers:

| Primitive | Hits | Use when |
|---|---|---|
| `NODE "id"` | columns | You know the ID |
| `NODES WHERE ...` | columns | Structured filter |
| `SIMILAR TO "text"` | vectors | You have a natural-language cue but no anchor |
| `SIMILAR TO NODE "id"` | vectors (from anchor) | Find things like a known node |
| `LEXICAL SEARCH "text"` | FTS5 | Exact token BM25 |
| `REMEMBER "text"` | vectors + FTS5 + columns | Hybrid natural-language retrieval |
| `RECALL FROM "id" DEPTH k` | edges | Spreading activation from a known node |
| `TRAVERSE`, `PATH`, `MATCH` | edges | Graph walks |

**Critical:** `REMEMBER` does NOT use graph edges. Adding edges helps `RECALL`, `TRAVERSE`, `PATH` — NOT `REMEMBER`. If you build a beautiful entity graph and then only call REMEMBER, the graph is invisible. This is the #1 trap.

## The golden ingestion pattern

This is the order of operations every ingestion should follow. Skipping steps costs accuracy, speed, or both.

### 1. Register your schema FIRST

Always. Before the first CREATE NODE.

```python
gs.execute(
    'SYS REGISTER NODE KIND "message" '
    'REQUIRED session:string, role:string, content:string '
    'OPTIONAL importance:float, position:int '
    'EMBED content'
)
```

Why:

- `REQUIRED` + typed fields pre-allocate the numpy column with the correct dtype. Without this, graphstore infers the dtype on the first write — which works but locks you in and costs a branch on every subsequent write.
- `EMBED content` tells the engine "when this kind is created, auto-embed the `content` field". You do not need to pass a `DOCUMENT` clause or call `_embed_and_store` manually. One source of truth.
- Typed `string` fields become `int32_interned` columns which make `WHERE session = "..."` vectorized-fast via the string table.

Register every edge kind too if you plan to validate endpoints:

```python
gs.execute('SYS REGISTER EDGE KIND "mentions" FROM message TO entity')
```

This is validated on `CREATE EDGE`. Unregistered edge kinds are allowed but unvalidated.

### 2. Wrap ingestion in `deferred_embeddings`

Always. This is a 4-10x speedup on transformer embedders (bge-*, e5, EmbeddingGemma) and neutral on model2vec.

```python
with gs.deferred_embeddings(batch_size=128):
    for item in batch:
        gs.execute(f'CREATE NODE "{item.id}" kind = "message" content = "..." ...')
# Pending embeddings flush on context exit
```

Without this, every CREATE triggers a single-row embedder call. With it, CREATE appends (slot, text) to a pending list and the embedder is called in batches of `batch_size`.

Do NOT wrap ingestion in `BEGIN ... COMMIT` (BATCH) for bulk loads. BATCH does a full column snapshot at entry for rollback support, which is O(n_columns × n_nodes) in memory and slow. BATCH is for small atomic groups (e.g. create a parent + wire N children), not bulk.

### 3. Use the right field names

| You want... | Use this field | Why |
|---|---|---|
| The text to embed | `content` (or whatever you put in EMBED) | Schema EMBED directive picks this up |
| A human-readable label | `summary` (common convention) | REMEMBER and NODE results surface this by habit |
| A stable ordering hint | `position` | For "what came next" |
| Source attribution | `source` | For "where did this come from" |
| Confidence scoring | **`__confidence__`** | Reserved. REMEMBER reads this for its confidence signal |
| Custom importance | `importance` | Just a column. REMEMBER does NOT read this. |

**Do not confuse `importance` with `__confidence__`.** REMEMBER's hybrid fusion reads `__confidence__` (reserved, set via `ASSERT ... CONFIDENCE` or direct `set_reserved`). Setting `importance` does nothing for REMEMBER. See the gotchas section.

### 4. Time is a special column

`__created_at__` and `__updated_at__` are set to wall clock time on every `CREATE NODE` and `UPDATE NODE`. They are reserved. **There is no DSL way to override them** — no `CREATE NODE ... AT "2023-05-30"` syntax.

REMEMBER's recency signal is `exp(-age_days / 30)` where `age_days = (now - __updated_at__) / day_ms`. If all your nodes get `now_ms` at ingest, every node has `recency = 1.0` → the recency signal contributes no differential ranking → effectively 4-signal fusion.

**If you want real recency**, override the columns directly after CREATE:

```python
slot = gs._store.id_to_slot[gs._store.string_table.intern(node_id)]
gs._store.columns.set_reserved(slot, "__created_at__", real_ms)
gs._store.columns.set_reserved(slot, "__updated_at__", real_ms)
```

This is a private API. It's the right tool when you have true timestamps (document creation dates, log entries, backfilled history). But see the gotcha below about benchmarks.

### 5. Edges are cheap to create, expensive to rebuild

Every `CREATE EDGE` appends to `_edges_by_type` and sets `_edges_dirty = True`. On the first read that needs the CSR matrix, `_rebuild_edges` fires and reconstructs the whole thing. This is O(total_edges).

Bulk-create all edges before the first read. Do not interleave CREATE EDGE with NODE / TRAVERSE queries — you'll trigger a rebuild per interleaving point.

### 6. Chain messages with `next` edges

For conversational data, create a `next` edge between consecutive messages in a session. This enables RECALL walks, ANCESTORS / DESCENDANTS queries, and subgraph retrieval.

```python
for i in range(n - 1):
    gs.execute(f'CREATE EDGE "{s_id}:msg{i}" -> "{s_id}:msg{i+1}" kind = "next"')
```

### 7. If you want cross-session reasoning, build an entity graph

This is the step most ingestion loops skip. Graphstore can do things a vector store cannot — but only if the graph is actually built.

```python
# Regex-based capitalized-phrase extraction is fine for a first pass.
import re
ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z0-9_-]{2,}(?:\s+[A-Z][a-zA-Z0-9_-]{2,}){0,3}\b")

for message in messages:
    entities = set(ENTITY_RE.findall(message.content))
    for ent in entities:
        ent_id = f"ent:{ent.lower().replace(' ', '_')}"
        try:
            gs.execute(f'CREATE NODE "{ent_id}" kind = "entity" name = "{ent}"')
        except NodeExists:
            pass
        gs.execute(f'CREATE EDGE "{message.id}" -> "{ent_id}" kind = "mentions"')
```

Then at query time:

```python
# "Tell me everything we discussed about Max the dog"
recall = gs.execute('RECALL FROM "ent:max" DEPTH 2 LIMIT 20')
```

Now RECALL surfaces every message that mentions Max across every session. No vector store can do this without ALSO running cross-session similarity — which is noisier and slower.

**Do NOT `EMBED name` on entity nodes.** Short entity names make bad vectors (one or two tokens). Let entities live in the graph layer only. If you need entity → vector routes, go through the messages that mention them.

### 8. For BM25 to work, text must be in `doc_fts`

`LEXICAL SEARCH` and the BM25 leg of `REMEMBER` both query `doc_fts` (the FTS5 virtual table). This table is only populated by:

- `INGEST "file.pdf" ...` (via the ingest engine)
- `DocumentStore.put_summary(slot, text, ...)` (direct Python API)
- `CREATE NODE ... DOCUMENT "text"` does NOT add to `doc_fts` — it only writes to the `documents` blob table

If you use plain `CREATE NODE` (no INGEST, no put_summary), **you have no BM25 signal**. REMEMBER degrades to vector + recency + confidence + recall_count.

Fixes:

- Use the INGEST DSL command if you have files
- Call `put_summary(slot, content)` directly from Python after CREATE for bulk non-file content
- Accept that BM25 is off and tune the REMEMBER weights accordingly

## Query primitives: which one to reach for

This is where the benchmark you are running should dictate your choice. Do not reflexively default to REMEMBER.

### `REMEMBER "query" [LIMIT k] [WHERE ...]`

The default natural-language retrieval primitive. Fuses 5 signals with default weights `[0.30, 0.20, 0.15, 0.20, 0.15]`:

```
0.30 × vector_similarity     (cosine from embedder)
0.20 × bm25_normalized       (FTS5 - needs doc_fts populated!)
0.15 × recency               (exp(-age_days/30) from __updated_at__)
0.20 × confidence            (from __confidence__ column)
0.15 × recall_frequency      (from __recall_count__ column)
```

Use when: the question is natural language and you want the best single-shot retrieval. The sweet spot.

Do NOT use when: your only data is via CREATE NODE (no FTS). You'll get a 20%-weighted null signal.

### `SIMILAR TO "text" LIMIT k [WHERE ...]`

Pure vector similarity. No recency, no BM25, no confidence.

Use when: you need deterministic, fast nearest-neighbor search without fusion. E.g. duplicate detection, clustering, nearest-neighbor retrieval for a single anchor vector.

### `SIMILAR TO NODE "id" LIMIT k`

Same but anchored to an existing node's vector. Useful for "find things like this one".

### `LEXICAL SEARCH "text" [WHERE ...]`

Pure BM25. Only works if `doc_fts` is populated. Case-insensitive, stemmed (porter + unicode61).

Use when: the query has distinctive keywords (technical terms, proper nouns) that vector search blurs.

### `RECALL FROM "node_id" DEPTH k LIMIT n`

Spreading activation from a cue node. Walks the edge graph for `k` hops with a decay factor, multiplies by node importance, filters by live mask, returns top-n by activation score.

Use when: you have an anchor concept (entity, topic, person) and want everything connected to it. The signature use case for a graph DB in a memory stack.

### `TRAVERSE`, `PATH`, `ANCESTORS`, `DESCENDANTS`, `MATCH`

Deterministic graph walks without activation scoring. Use for structured queries over the graph — "what called this function", "path from A to B", "all children of this parent".

## Gotchas we learned the hard way

### G1. REMEMBER does not read graph edges

`SYS CONNECT` creates `similar_to` edges between vector-similar nodes. If you then call `REMEMBER`, the edges are invisible. `REMEMBER` only fuses vector + BM25 + recency + confidence + recall_count. You have to call `RECALL FROM <anchor>` (or combine REMEMBER + RECALL results in the adapter) to get any graph-based retrieval.

### G2. The BM25 signal is off unless you populate `doc_fts`

`CREATE NODE ... DOCUMENT "text"` writes to the `documents` blob table, not to `doc_fts`. REMEMBER's BM25 leg returns empty. You have to go through INGEST or call `put_summary` directly.

### G3. Real timestamps can hurt as much as they help

For temporal-reasoning questions, real `__updated_at__` values let `exp(-age_days/30)` discriminate. But if the data is years old (say, a 2023 conversation dataset being evaluated in 2026), every message's recency collapses to `exp(-1100/30) ≈ 0`. The signal is dead AND the ranking is subtly disturbed if some sessions happen to be closer to the question date than the answer-bearing session.

For static benchmarks, leaving the default wall-clock `now_ms` on every node produces a uniform `recency = 1.0` which contributes zero differential signal — BUT it does not hurt. For temporal benchmarks, you need real timestamps AND you need to be thoughtful about which values land where.

**Rule of thumb:** override timestamps only when you KNOW the question-time vs data-time relationship. Otherwise leave them at wall clock.

### G4. `importance` and `__confidence__` are different columns

REMEMBER reads `__confidence__`. If you stuff your scores into `importance`, REMEMBER ignores them. Write to `__confidence__` via `set_reserved` or via `ASSERT ... CONFIDENCE 0.9`. Or set REMEMBER's weights to zero out the confidence signal.

### G5. Entity nodes should not be embedded

Entities are short strings (1-3 tokens). Their embeddings are noisy and they compete in vector search against much richer message content. Register entities WITHOUT `EMBED`:

```python
gs.execute('SYS REGISTER NODE KIND "entity" REQUIRED name:string')  # no EMBED
```

Use entities only as graph anchors for RECALL, not as vector search targets.

### G6. `WHERE kind = "X"` in REMEMBER works, but only after candidates are gathered

REMEMBER collects vector and BM25 candidates FIRST, then applies WHERE. If all your entity / session / other-kind nodes have no vectors, they never appear as candidates anyway and the kind filter is redundant. But if ALL your nodes have vectors (e.g. you EMBED entities), you are wasting candidate slots on nodes you're going to filter out.

### G7. Schema `EMBED field` and `DOCUMENT` clause are two different embedding paths

- `EMBED content` in the schema + `CREATE NODE ... content = "..."` → embeds the content field
- `CREATE NODE ... DOCUMENT "text"` → stores the blob + embeds the blob IF no EMBED field is set

If you set BOTH (`EMBED content` schema + `DOCUMENT "..."` clause), only the EMBED path fires. The DOCUMENT text is stored as a blob but not embedded. You probably do not want this.

### G8. `deferred_embeddings` does not batch across ingest calls

The context manager is per-call. If you have 500 records and call `ingest(record)` 500 times, each call gets its own deferred context. The embedder is called 500 times with ~500 messages each, not once with 250k. To get a single mega-batch, you'd need to restructure the runner.

### G9. Direct column writes bypass dirty tracking

`store.columns.set_reserved(slot, field, value)` writes straight to numpy. The `_dirty_columns` flag does not get set. This means the next checkpoint might not persist your write. For benchmark runs (where we close without persisting) this is fine. For production, either use `UPDATE NODE` (which does set the flag) or manually set `store._dirty_columns = True` after your direct writes.

### G10. Single-writer is a hard rule

There is no concurrency in the write path. `queued=True` installs a submission queue with a single worker (the flag name is honest — it's a queue, not parallelism). If you try to call `execute` from two threads on a `queued=False` GraphStore, you will get silent corruption of `id_to_slot` and `_edges_by_type`. This is architectural and will not be fixed — see `skills/.../docs/single-writer.md` (TODO) or the README thread safety section.

## Patterns by use case

### Pattern A: conversational memory benchmark (LongMemEval, LoCoMo)

```python
# Per record: reset, ingest haystack, query, score
gs = GraphStore(path=tmpdir, embedder=my_embedder)

gs.execute('SYS REGISTER NODE KIND "session" REQUIRED session_id:string')
gs.execute(
    'SYS REGISTER NODE KIND "message" '
    'REQUIRED session:string, role:string, content:string '
    'OPTIONAL position:int '
    'EMBED content'
)
gs.execute('SYS REGISTER NODE KIND "entity" REQUIRED name:string')  # no EMBED

for session in record.haystack:
    with gs.deferred_embeddings(batch_size=128):
        # Session node
        gs.execute(f'CREATE NODE "sess:{session.id}" kind = "session" session_id = "{session.id}"')

        # Messages
        for i, msg in enumerate(session.messages):
            gs.execute(
                f'CREATE NODE "{session.id}:msg{i}" kind = "message" '
                f'session = "{session.id}" role = "{msg.role}" '
                f'content = "{escape(msg.content)}" position = {i}'
            )
            gs.execute(f'CREATE EDGE "sess:{session.id}" -> "{session.id}:msg{i}" kind = "has_message"')

            # Entity extraction
            for ent in extract_entities(msg.content):
                ent_id = f"ent:{slug(ent)}"
                try:
                    gs.execute(f'CREATE NODE "{ent_id}" kind = "entity" name = "{ent}"')
                except NodeExists:
                    pass
                gs.execute(f'CREATE EDGE "{session.id}:msg{i}" -> "{ent_id}" kind = "mentions"')

        # Next edges
        for i in range(len(session.messages) - 1):
            gs.execute(f'CREATE EDGE "{session.id}:msg{i}" -> "{session.id}:msg{i+1}" kind = "next"')
```

Query time routing by question category:

```python
def query(question, category):
    if category in ("single-session-user", "single-session-preference", "single-session-assistant"):
        return gs.execute(f'REMEMBER "{question}" LIMIT 5 WHERE kind = "message"')

    if category == "multi-session":
        primary = gs.execute(f'REMEMBER "{question}" LIMIT 10 WHERE kind = "message"')
        # Extract entities from the QUESTION and RECALL from each
        for ent in extract_entities(question):
            recall = gs.execute(f'RECALL FROM "ent:{slug(ent)}" DEPTH 2 LIMIT 5')
            # fuse with primary
        return fused

    if category == "knowledge-update":
        # Most recent mentions outrank older ones
        primary = gs.execute(f'REMEMBER "{question}" LIMIT 10 WHERE kind = "message"')
        recent = gs.execute(f'NODES WHERE kind = "message" ORDER BY __updated_at__ DESC LIMIT 5')
        return dedupe(primary + recent)[:5]

    return gs.execute(f'REMEMBER "{question}" LIMIT 5 WHERE kind = "message"')
```

Key insight: **different categories want different retrieval strategies**. Do not serve one query primitive to all of them.

### Pattern B: document ingestion (PDFs, long text)

Use the built-in INGEST DSL command. It handles chunking, FTS5 population, vector indexing, and cross-doc wiring.

```python
gs.execute('INGEST "report.pdf" AS "doc:q3" KIND "report"')
gs.execute('SYS CONNECT')  # wire similar chunks across documents
```

Do not reinvent this. The tiered router (MarkItDown → PyMuPDF4LLM → Docling → VLM) handles most real-world files.

### Pattern C: fact / belief tracking

Use `ASSERT` + `RETRACT` with confidence scores and sources. These set `__confidence__` which REMEMBER reads.

```python
gs.execute('ASSERT "fact:earth-radius" kind = "fact" value = 6371 CONFIDENCE 0.99 SOURCE "physics-tool"')
gs.execute('RETRACT "fact:old-preference" REASON "user corrected"')
gs.execute('SYS CONTRADICTIONS WHERE kind = "fact" FIELD value GROUP BY topic')
```

### Pattern D: temporal data (logs, events, time series)

Override `__updated_at__` on ingest with real timestamps, use REMEMBER for recency-weighted retrieval, use `NODES WHERE __created_at__ > NOW() - 7d` for time-range filtering. See G3 for the caveats.

## Debug checklist: "my retrieval is bad"

Run through this top to bottom when REMEMBER results look worse than expected.

1. **Is `doc_fts` actually populated?** Run `SELECT COUNT(*) FROM doc_fts`. If zero, your BM25 leg is dead. Either switch to INGEST or call `put_summary` per node. Expected: ~1 row per message/chunk.
2. **Do your nodes have vectors?** Run `gs._vector_store.count()`. Should match your live message count. If zero, your embedder silently failed or the schema EMBED field isn't set correctly.
3. **Is `__confidence__` set anywhere?** If not, the confidence signal contributes a flat 1.0 (default) which is fine but uninformative. Set it via ASSERT or `set_reserved` if you have ground-truth confidence scores.
4. **Is `__updated_at__` all equal to wall-clock ingest time?** Then recency contributes a flat 1.0. That is actually fine for static corpora — see G3. Only fix this if you actually need temporal discrimination.
5. **Are you calling REMEMBER for a graph-shaped question?** Multi-session and multi-entity questions want RECALL, not REMEMBER. The fix is to combine both in your adapter.
6. **Did you filter out the answer with `WHERE`?** A WHERE clause applied post-candidates can shrink results to zero. Remove the filter and see if results come back. A common mistake is `WHERE role = "user"` when the answer lives in an assistant turn.
7. **Are your edges actually reaching the CSR matrix?** Run a single `TRAVERSE FROM "any_node" DEPTH 1` and verify you see neighbors. If not, the edge builder hasn't flushed yet — do a no-op query to trigger `_ensure_edges_built`.
8. **Is your embedder dimension mismatch?** If you swap embedders mid-run, the vector index gets poisoned with mixed-dim vectors. `SYS REEMBED` fixes it.
9. **Are you competing for vector slots with non-content nodes?** If you EMBED entity / session / metadata nodes, they contaminate vector search results. Remove EMBED from any kind that isn't the primary content.
10. **Are you using `importance` expecting REMEMBER to read it?** It doesn't. Use `__confidence__`.

## Quick reference: the dos and don'ts

**DO:**
- Register schema before first CREATE
- Wrap bulk ingestion in `deferred_embeddings`
- Use typed fields in REQUIRED/OPTIONAL
- Build an entity graph for cross-session / multi-hop questions
- Use RECALL for graph queries, REMEMBER for language queries
- Set `__confidence__` (not `importance`) for confidence-weighted scoring
- Populate `doc_fts` via INGEST or `put_summary` if you want BM25

**DO NOT:**
- Call CREATE NODE 500k times without deferred_embeddings
- Use BATCH for bulk loads
- EMBED short entity / tag nodes
- Expect `CREATE NODE ... DOCUMENT "text"` to populate BM25
- Override `__updated_at__` unless you actually have real timestamps
- Use REMEMBER and then wonder why your graph edges are ignored
- Stuff arbitrary scores into `importance` expecting REMEMBER to read them
- Run two writer threads on `queued=False`

## TL;DR

graphstore is three storage engines (graph, vector, document) unified by a DSL, plus feature layers on top. You have to feed data into the right ones. Schema first, deferred embeddings, entity graph for multi-hop, `put_summary` for BM25, REMEMBER + RECALL fusion for non-trivial questions, `__confidence__` not `importance`, and timestamps are a trap unless you know what you're doing.

When in doubt, reach for Pattern A above and adapt.
