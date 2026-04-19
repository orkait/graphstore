---
name: graphstore-ingestion
description: How to inject data into graphstore correctly. Covers the three storage engines (graph, vector, document), schema-first ingestion, REMEMBER vs RECALL vs SIMILAR TO, timestamp pitfalls, BM25 FTS gotchas, benchmark-friendly patterns, and the anti-patterns that make graphstore underperform vector stores. Use whenever you are writing a graphstore adapter, ingesting a new dataset, debugging retrieval quality, or about to write CREATE NODE in a loop.
compatibility: Requires a local graphstore install. Works against any graphstore >= 0.3.0.
metadata:
  author: orkait
  version: "2.3"
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

**The three storage engines** - each is an independent class with its own state and lifecycle:

1. **Graph engine** (`graphstore/core/`) - `CoreStore` holds columnar node arrays (numpy), sparse CSR edge matrices (scipy), a string intern table, and tombstone-based deletion. This is the data plane for structured fields + the relationship graph. Every field you set lives in a typed numpy column. `node_ids`, `node_kinds`, `id_to_slot`, `_edges_by_type` are the load-bearing structures.
2. **Vector engine** (`graphstore/vector/`) - `VectorStore` wraps a usearch HNSW index over `(slot, vector)` pairs with cosine metric. `VectorStore.search(query, k, mask)` returns top-k slot ids filtered by a boolean live mask.
3. **Document engine** (`graphstore/document/`) - `DocumentStore` is SQLite multi-table storage (`documents`, `summaries`, `doc_metadata`, `images`) with an FTS5 virtual table (`doc_fts`) over summaries. This is where BM25 lives. `put_summary(slot, text, ...)` writes both the row AND the FTS index.

A single node can live in **all three at once**: row in numpy columns + vector in usearch + entry in FTS5.

**Plus feature layers on top:**

- **DSL** (`graphstore/dsl/`) - Lark LALR(1) grammar compiled to 70+ AST dataclasses, handler-registry dispatch. The unified interface to all three engines.
- **Embedding** (`graphstore/embedding/`) - pluggable embedder protocol. Default is **model2vec** (core dep since the extras consolidation - no extra install needed). Alternatives under `[embedders-extra]`: FastEmbed and GGUF via llama-cpp-python. OnnxHF for Jina v5 / Harrier / EmbeddingGemma ships separately via `graphstore install-embedder`. Registry at `graphstore/registry/models.py`.
- **Reranking** (`graphstore/embedding/reranker.py`) - pluggable cross-encoder reranker (FlashRank, ONNX, GGUF backends). Used by `remember_rerank` and `full_rerank` retrieval strategies.
- **Beliefs** - `ASSERT` / `RETRACT` / `PROPAGATE` write reserved columns (`__confidence__`, `__retracted__`, `__source__`) on the Graph engine.
- **Evolution** (`graphstore/core/evolve/`) - `EvolutionEngine`, opt-in. WHEN/THEN rules that self-tune graphstore's own runtime parameters based on live signals.
- **Ingest pipeline** (`graphstore/ingest/`) - file-to-graph routing (MarkItDown → PyMuPDF4LLM → Docling), chunker, vision. Used by the `INGEST` DSL verb.
- **Algos** (`graphstore/algos/`) - 17 pure numpy/scipy primitives under a strict purity gate. Tunable in isolation.

**Optional subsystems** (via extras): **Vault** (markdown notebook, `graphstore/vault/`, core since pyyaml moved in), **Vision** (`[vision]` extra: local llama.cpp sidecar + SmolVLM2-2.2B by default for image captioning), **Audio** (`[audio]` extra: in-process faster-whisper STT). No voice / TTS subsystem - graphstore is a DB, not an engine (see PR #104 rationale).

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

**Updated:** `REMEMBER` can optionally include an entity-graph signal in fusion when `dsl.graph_signal_enabled=true` - chunks connected to high-degree entities (mentioned across many sessions) get boosted. Heavy multi-hop traversal still belongs to `RECALL`. Building an entity graph helps both.

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

- `REQUIRED` + typed fields pre-allocate the numpy column with the correct dtype. Without this, graphstore infers the dtype on the first write - which works but locks you in and costs a branch on every subsequent write.
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

`__created_at__` and `__updated_at__` are set to wall clock time on every `CREATE NODE` and `UPDATE NODE`. They are reserved. **There is no DSL way to override them** - no `CREATE NODE ... AT "2023-05-30"` syntax.

REMEMBER's recency signal is `exp(-age_days / half_life)` where `half_life` is configurable via `dsl.recency_half_life_days` (default 30.0). If all your nodes get `now_ms` at ingest, every node has `recency = 1.0` → the recency signal contributes no differential ranking → effectively 4-signal fusion.

**If you want real recency**, override the columns directly after CREATE:

```python
slot = gs._store.id_to_slot[gs._store.string_table.intern(node_id)]
gs._store.columns.set_reserved(slot, "__created_at__", real_ms)
gs._store.columns.set_reserved(slot, "__updated_at__", real_ms)
```

This is a private API. It's the right tool when you have true timestamps (document creation dates, log entries, backfilled history). But see the gotcha below about benchmarks.

### 5. Edges are cheap to create, expensive to rebuild

Every `CREATE EDGE` appends to `_edges_by_type` and sets `_edges_dirty = True`. On the first read that needs the CSR matrix, `_rebuild_edges` fires and reconstructs the whole thing. This is O(total_edges).

Bulk-create all edges before the first read. Do not interleave CREATE EDGE with NODE / TRAVERSE queries - you'll trigger a rebuild per interleaving point.

### 6. Chain messages with `next` edges

For conversational data, create a `next` edge between consecutive messages in a session. This enables RECALL walks, ANCESTORS / DESCENDANTS queries, and subgraph retrieval.

```python
for i in range(n - 1):
    gs.execute(f'CREATE EDGE "{s_id}:msg{i}" -> "{s_id}:msg{i+1}" kind = "next"')
```

### 7. If you want cross-session reasoning, build an entity graph

This is the step most ingestion loops skip. Graphstore can do things a vector store cannot - but only if the graph is actually built.

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

Now RECALL surfaces every message that mentions Max across every session. No vector store can do this without ALSO running cross-session similarity - which is noisier and slower.

**Do NOT `EMBED name` on entity nodes.** Short entity names make bad vectors (one or two tokens). Let entities live in the graph layer only. If you need entity → vector routes, go through the messages that mention them.

### 8. For BM25 to work, text must be in `doc_fts`

`LEXICAL SEARCH` and the BM25 leg of `REMEMBER` both query `doc_fts` (the FTS5 virtual table). As of PR #102, this table is populated by **any** of:

- `INGEST "file.pdf" ...` (via the ingest engine)
- `DocumentStore.put_summary(slot, text, ...)` (direct Python API)
- **`CREATE NODE ... DOCUMENT "text"`** - auto-populates BM25 + stores the blob in one shot (this is the simplest path for bulk conversational data)
- `CREATE NODE ... kind = "K"` with a schema that has `EMBED content` - still only embeds; FTS is populated only when the text is passed via the `DOCUMENT` clause.

If you call plain `CREATE NODE kind = "X" content = "..."` without `DOCUMENT`, you still get a vector (via `EMBED content` schema directive) but no BM25 row. REMEMBER degrades to vector + recency + confidence + recall_count.

**Fastest BM25-ready path for bulk conversational data:**

```python
gs.execute(
    f'CREATE NODE "{msg_id}" kind = "message" '
    f'session = "{s}" role = "{role}" '
    f'DOCUMENT "{escape(content)}"'
)
# One line: graph row + vector + doc_fts entry all populated.
```

## Query primitives: which one to reach for

This is where the benchmark you are running should dictate your choice. Do not reflexively default to REMEMBER.

### `REMEMBER "query" [LIMIT k] [WHERE ...]`

The default natural-language retrieval primitive. 5-stage pipeline:
1. **Gather** - sentence-level vector search + chunk-level BM25, union capped adaptively
2. **Fuse** - weighted (or RRF) blend of signals below, plus co-occurrence bonus + recall-frequency nudge
3. **Temporal filter** - hard zero out of range when `AT` clause present
4. **Rerank** - optional (GGUF / ONNX) when configured; otherwise top-K from fusion
5. **Nucleus** (optional, off by default) - walk structural edges only; attached to `meta["nucleus"]`

```
weighted fusion (default 4-weight config, graph_signal_enabled=true):
  0.52 x vec_signal       (max over sentence cosine similarities per message)
  0.25 x bm25_signal      (FTS5 normalized - needs doc_fts populated!)
  0.15 x recency_signal   (exp(-age/half_life) from __event_at__ or __updated_at__)
  0.08 x graph_signal     (entity-degree sum over mentioned entities, log-scaled)
  + co-occurrence bonus   (min(vec, bm25) * 0.10 when a candidate is found by both)
  + recall-frequency nudge (log1p(recall_count) * 0.05)
```

All weights and parameters are configurable via `graphstore.json` or constructor kwargs.

Key config knobs for REMEMBER:
- `dsl.remember_weights`: fusion weights (default `[0.52, 0.25, 0.15, 0.08]`; 3 weights drops the graph channel)
- `dsl.fusion_method`: `"weighted"` (default) or `"rrf"`
- `dsl.graph_signal_enabled`: include entity-degree channel (default `true`)
- `dsl.recency_half_life_days`: recency decay half-life (default 7300.0)
- `dsl.sentence_query_expansion`: split query into sentences for multi-vector search (default `true`)
- `dsl.nucleus_expansion`: enable structural-edge context walk (default `false`)
- `vector.search_oversample`: ANN candidate multiplier (default 16)

Use when: the question is natural language and you want the best single-shot retrieval.

Do NOT use when: your only data is via CREATE NODE (no FTS). The BM25 signal will be dead.

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

Deterministic graph walks without activation scoring. Use for structured queries over the graph - "what called this function", "path from A to B", "all children of this parent".

## Gotchas we learned the hard way

### G1. REMEMBER can use entity graph as a fusion channel

Set `dsl.graph_signal_enabled=true` (default) to include an entity-degree signal in REMEMBER fusion: chunks mentioning entities that many other chunks also mention get boosted. Not a multi-hop expansion - use `RECALL` for that. The heavy-lift HybridRAG spreading-activation blend was removed in the pipeline refactor; the 5-stage pipeline now does rerank instead. Building an entity graph still directly improves REMEMBER results via this channel.

### G2. BM25 requires `doc_fts` to be populated (PR #102 updated path)

As of PR #102, `CREATE NODE ... DOCUMENT "text"` DOES populate `doc_fts` in addition to the blob table. So BM25 works out of the box for nodes you write with a `DOCUMENT` clause. Plain `CREATE NODE kind = "X" content = "..."` (no DOCUMENT) still only writes columns + vector - BM25 is empty for those. If you want BM25 and do not want the full INGEST pipeline, use `DOCUMENT "..."` on your CREATE NODE statements or call `DocumentStore.put_summary` directly.

### G3. Real timestamps can hurt as much as they help

For temporal-reasoning questions, real `__updated_at__` values let `exp(-age_days/30)` discriminate. But if the data is years old (say, a 2023 conversation dataset being evaluated in 2026), every message's recency collapses to `exp(-1100/30) ≈ 0`. The signal is dead AND the ranking is subtly disturbed if some sessions happen to be closer to the question date than the answer-bearing session.

For static benchmarks, leaving the default wall-clock `now_ms` on every node produces a uniform `recency = 1.0` which contributes zero differential signal - BUT it does not hurt. For temporal benchmarks, you need real timestamps AND you need to be thoughtful about which values land where.

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

### G11. NER emits duplicate entity names per message (PR #128)

The batched NER call returns a list of entity strings per message. Multi-span matches can produce the SAME entity name multiple times for one message. A naive `CREATE EDGE msg -> ent kind = "mentions"` loop over that list raises `BatchRollback: Duplicate edge`, killing the entire BEGIN...COMMIT block.

Fix: dedupe per message before emitting the edge. See Pattern A above. This is load-bearing for any benchmark that runs for more than ~100 records - the failure mode is "benchmark progresses fine for an hour, then crashes at record 92".

### G12. `USING VISION` auto-starts the sidecar only if `[vision]` is installed

`INGEST "scan.pdf" USING VISION "<model>"` calls `VisionHandler(base_url=None)` which resolves: env `GRAPHSTORE_VISION_URL` → PID-file live check → probe default port 8418 → auto-spawn sidecar (iff `[vision]` extra is importable). If none of those yield a URL, you get a `RuntimeError` pointing at `pip install 'graphstore[vision]'`. Do not catch-and-swallow that error - it means vision is genuinely unavailable and the caption will be empty.

For production, pre-start via CLI so the first ingest doesn't pay the weight-download cost:

```bash
graphstore vision serve --pull-only    # download weights only (~1.5 GB for SmolVLM2-2.2B default)
graphstore vision serve                # spawn sidecar, block until ready
graphstore vision status               # verify
```

### G13. Audio ingest is in-process, not sidecar

Unlike vision which uses a sidecar (VLM inference is long, benefits from server-side batching), whisper transcription is short enough that IPC overhead would dominate. `[audio]` extra installs `faster-whisper` and the `WhisperIngestor` caches one `WhisperModel` per `(size, device, compute_type)` tuple. First call downloads the model from HF Hub (~40 MB for `tiny`, ~150 MB for `base`). Default is `base`. Segments are fused into chunks with `[mm:ss-mm:ss]` headings so retrieval can cite timestamps.

### G10. Single-writer is a hard rule

There is no concurrency in the write path. `queued=True` installs a submission queue with a single worker (the flag name is honest - it's a queue, not parallelism). If you try to call `execute` from two threads on a `queued=False` GraphStore, you will get silent corruption of `id_to_slot` and `_edges_by_type`. This is architectural and will not be fixed - see `skills/.../docs/single-writer.md` (TODO) or the README thread safety section.

## Patterns by use case

### Pattern A: conversational memory benchmark (LongMemEval, LoCoMo)

**Use the `DOCUMENT` clause on messages.** This is the current best practice - it hits all three engines in a single CREATE and populates BM25 automatically (PR #102). Dedupe entity mentions per message (PR #128) to avoid the "Duplicate edge" BatchRollback when NER emits the same entity name from multiple spans.

```python
# Per record: reset, ingest haystack, query, score
gs = GraphStore(path=tmpdir, embedder=my_embedder)

gs.execute('SYS REGISTER NODE KIND "session" REQUIRED session_id:string')
gs.execute(
    'SYS REGISTER NODE KIND "message" '
    'REQUIRED session:string, role:string '
    'OPTIONAL position:int'
)
gs.execute('SYS REGISTER NODE KIND "entity" REQUIRED name:string')  # no EMBED

# Use a real NER (TinyBERT ONNX is in the box; see G11 below).
from graphstore.ingest.entity_extract import extract_batch

for session in record.haystack:
    # One big batched transaction per session: NER batch + single DSL blob
    # executed under deferred_embeddings so the embedder runs once.
    msg_contents = [m.content for m in session.messages]
    per_msg_entities = extract_batch(msg_contents)   # list[list[str]] aligned to msg_contents

    dsl = ["BEGIN"]
    dsl.append(f'CREATE NODE "sess:{session.id}" kind = "session" session_id = "{session.id}"')
    for i, msg in enumerate(session.messages):
        msg_id = f"{session.id}:msg{i}"
        dsl.append(
            f'CREATE NODE "{msg_id}" kind = "message" '
            f'session = "{session.id}" role = "{msg.role}" '
            f'position = {i} '
            f'DOCUMENT "{escape(msg.content)}"'      # <-- populates BM25 + blob + vector in one shot
        )
        dsl.append(f'CREATE EDGE "sess:{session.id}" -> "{msg_id}" kind = "has_message"')

        # PR #128: dedupe per message. NER can emit the same entity multiple
        # times from multi-span hits; re-creating the same "mentions" edge
        # raises Duplicate edge and rolls the whole session back.
        msg_ent_seen: set[str] = set()
        for ent in per_msg_entities[i]:
            ent_id = f"ent:{slug(ent)}"
            if ent_id in msg_ent_seen:
                continue
            msg_ent_seen.add(ent_id)
            dsl.append(f'UPSERT NODE "{ent_id}" kind = "entity" name = "{ent}"')
            dsl.append(f'CREATE EDGE "{msg_id}" -> "{ent_id}" kind = "mentions"')

    for i in range(len(session.messages) - 1):
        dsl.append(f'CREATE EDGE "{session.id}:msg{i}" -> "{session.id}:msg{i+1}" kind = "next"')
    dsl.append("COMMIT")

    with gs.deferred_embeddings(batch_size=128):
        gs.execute("\n".join(dsl))
```

Query time - REMEMBER handles fusion + optional rerank internally:

```python
def query(question, k=5):
    depth = 8  # adapter-side over-fetch multiplier

    # Hybrid retrieval (vector + BM25 + entity-graph channel when enabled +
    # optional reranker)
    primary = gs.execute(f'REMEMBER "{question}" LIMIT {k * depth} WHERE kind = "message"')
    merged = [node["content"] for node in primary.data if node.get("content")]

    # Optional: entity graph traversal for cross-session reasoning
    for ent in extract_entities(question)[:3]:
        try:
            rec = gs.execute(f'RECALL FROM "ent:{slug(ent)}" DEPTH 2 LIMIT {k}')
            for node in rec.data:
                text = node.get("content", "")
                if text and text not in merged:
                    merged.append(text)
        except Exception:
            pass

    # Optional: recency boost for knowledge-update questions
    recent = gs.execute(f'NODES WHERE kind = "message" ORDER BY __updated_at__ DESC LIMIT {k * 2}')
    for node in recent.data:
        text = node.get("content", "")
        if text and text not in merged:
            merged.append(text)

    return merged[:k]
```

Key insight: REMEMBER's 5-stage pipeline (gather → fuse → temporal → rerank → optional nucleus) runs uniformly for every query. No category routing. Adding explicit RECALL and recency on top gives extra coverage for specialised question types. Configure a reranker (`dsl.reranker="gguf"`) when ranking accuracy matters more than latency.

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

### Pattern E: image + scanned PDF ingest (vision)

Requires `[vision]` extra. The sidecar auto-starts on first call, but pre-pulling weights avoids a mid-ingest surprise.

```bash
pip install 'graphstore[vision]'
graphstore vision serve --pull-only    # cache weights; SmolVLM2-2.2B Q4_K_M default (~1.5 GB)
```

```python
# Scanned-PDF fallback: tier-4 route when earlier tiers produce empty text
gs.execute('INGEST "scan.pdf" USING VISION "SmolVLM2-2.2B-Instruct-Q4_K_M.gguf"')

# Standalone image with VLM caption stored on the node + embedded
gs.execute('INGEST "chart.png" USING VISION "SmolVLM2-2.2B-Instruct-Q4_K_M.gguf" AS "img:q3-chart"')
```

Switch preset / runtime:

```bash
export GRAPHSTORE_VISION_MODEL=smolvlm-500m      # faster, lower quality (400 MB)
export GRAPHSTORE_VISION_URL=http://my-vllm/v1   # bring-your-own endpoint (Ollama, vLLM, OpenAI cloud)
```

Cap generation length via config:

```json
{ "document": { "vision_max_tokens": 512 } }
```

Captions flow into the image node's `summary` field and get embedded so REMEMBER / LEXICAL SEARCH hit them like any other chunk.

### Pattern F: audio ingest (speech-to-text)

Requires `[audio]` extra. In-process, no sidecar.

```bash
pip install 'graphstore[audio]'
```

```python
# Interview / voicememo -> transcript chunks with timestamp headings
gs.execute('INGEST "interview.mp3"')
gs.execute('INGEST "standup.m4a" AS "mem:standup-2026-04-15" KIND "standup"')
```

Chunks surface as:

```
heading: [00:12-00:34]
text: "... actual segment transcript ..."
```

so REMEMBER citations land inside the audio timeline. Format support: wav, mp3, ogg, flac, m4a, opus, webm. Default model is `base` (multilingual, ~150 MB on first use). Override:

```python
from graphstore.ingest.whisper_ingestor import WhisperIngestor
WhisperIngestor().convert("clip.wav", model="small", language="en", beam_size=5)
```

## Debug checklist: "my retrieval is bad"

Run through this top to bottom when REMEMBER results look worse than expected.

1. **Is `doc_fts` actually populated?** Run `SELECT COUNT(*) FROM doc_fts`. If zero, your BM25 leg is dead. Either switch to INGEST or call `put_summary` per node. Expected: ~1 row per message/chunk.
2. **Do your nodes have vectors?** Run `gs._vector_store.count()`. Should match your live message count. If zero, your embedder silently failed or the schema EMBED field isn't set correctly.
3. **Is `__confidence__` set anywhere?** If not, the confidence signal contributes a flat 1.0 (default) which is fine but uninformative. Set it via ASSERT or `set_reserved` if you have ground-truth confidence scores.
4. **Is `__updated_at__` all equal to wall-clock ingest time?** Then recency contributes a flat 1.0. That is actually fine for static corpora - see G3. Only fix this if you actually need temporal discrimination.
5. **Are you calling REMEMBER for a graph-shaped question?** Multi-session and multi-entity questions want RECALL, not REMEMBER. The fix is to combine both in your adapter.
6. **Did you filter out the answer with `WHERE`?** A WHERE clause applied post-candidates can shrink results to zero. Remove the filter and see if results come back. A common mistake is `WHERE role = "user"` when the answer lives in an assistant turn.
7. **Are your edges actually reaching the CSR matrix?** Run a single `TRAVERSE FROM "any_node" DEPTH 1` and verify you see neighbors. If not, the edge builder hasn't flushed yet - do a no-op query to trigger `_ensure_edges_built`.
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

## Configuration

All retrieval knobs are configurable via `graphstore.json` (overrides only), env vars (`GRAPHSTORE_DSL_*`), or constructor kwargs. The config chain:

```
config.py defaults -> graphstore.json (diffs only) -> GRAPHSTORE_* env vars -> constructor kwargs
```

Key retrieval config (in `graphstore.json`):

```json
{
  "dsl": {
    "remember_weights": [0.52, 0.25, 0.15, 0.08],
    "fusion_method": "weighted",
    "recency_half_life_days": 7300.0,
    "graph_signal_enabled": true,
    "nucleus_expansion": false,
    "sentence_query_expansion": true,
    "reranker": null
  },
  "vector": {
    "search_oversample": 16,
    "similarity_threshold": 0.85
  }
}
```

Or as constructor kwargs:

```python
gs = GraphStore(
    path="./db",
    embedder=my_embedder,
    search_oversample=16,
    remember_weights=[0.52, 0.25, 0.15, 0.08],
    graph_signal_enabled=True,
)
```

CLI to inspect config:

```bash
graphstore config --defaults    # all defaults
graphstore config --schema      # JSON Schema for graphstore.json
```

## TL;DR

graphstore = three storage engines (graph + vector + document) behind one DSL. REMEMBER fuses all three via a 5-stage pipeline (gather → fuse → temporal → rerank → optional nucleus). Ingestion rules:

1. Register schema first.
2. Wrap bulk writes in `deferred_embeddings`.
3. Use `CREATE NODE ... DOCUMENT "text"` for conversational data - one line, populates BM25 + blob + vector (PR #102).
4. Dedupe entity mentions per message before emitting `mentions` edges (PR #128).
5. `graph_signal_enabled=true` folds the entity graph into REMEMBER fusion; use `RECALL` for actual multi-hop.
6. Image ingest = `INGEST ... USING VISION "<model>"` under `[vision]`; sidecar auto-starts. Default SmolVLM2-2.2B.
7. Audio ingest = just `INGEST "clip.mp3"` under `[audio]`; in-process faster-whisper.

When in doubt: Pattern A for conversations, Pattern B for documents, Pattern E for images, Pattern F for audio.

## Query builder (v0.3.0+)

**Use the typed builder for every adapter, ingest loop, and retrieval entry point.** String DSL is still supported for one-off experiments; for production code the builder eliminates entire classes of bugs.

100% DSL coverage (87 verbs + 4 typed sub-DSLs), 100% line coverage, parser-roundtrip-verified in tests, injection-proof by construction.

```python
from graphstore import q, F, P, agg, Time, EvolveWhen as W, EvolveThen as A

# Ingestion adapter using the builder end-to-end
q.sys.register_node_kind("message",
                         required={"session": "string", "role": "string"},
                         optional={"position": "int"}).execute(gs)
q.sys.register_node_kind("entity", required={"name": "string"}).execute(gs)

# Batch-compose an entire session in one transaction. Variable refs
# (``$msg0``) let later statements point at earlier auto-ids without
# re-deriving the id on the Python side.
stmts = []
for i, msg in enumerate(session.messages):
    msg_id = f"{session_id}:msg{i}"
    stmts.append(q.create_node(msg_id, kind="message",
                               session=session_id, role=msg.role,
                               position=i, document=msg.content))
    stmts.append(q.create_edge(f"sess:{session_id}", msg_id, kind="has_message"))
    msg_ent_seen: set[str] = set()
    for ent in extract_entities(msg.content):
        ent_id = f"ent:{slug(ent)}"
        if ent_id in msg_ent_seen:
            continue
        msg_ent_seen.add(ent_id)
        stmts.append(q.upsert_node(ent_id, kind="entity", name=ent))
        stmts.append(q.create_edge(msg_id, ent_id, kind="mentions"))

q.batch(*stmts).execute(gs)

# Retrieval
q.remember(question, limit=20, where=F.eq("kind", "message")).execute(gs)

# Cross-session walk
q.recall(f"ent:{slug(entity)}", depth=2, limit=20).execute(gs)

# Temporal queries via Time namespace
q.nodes(where=F.eq("kind", "message")
              & F.gte("__event_at__", Time.now_minus(7, "d")),
        limit=100).execute(gs)

# Aggregates with typed HAVING (comparison operators on agg builders
# produce HavingExpr objects)
q.aggregate_nodes(
    select=[agg.count(), agg.avg("importance")],
    where=F.eq("kind", "memory"),
    group_by=["topic"],
    having=agg.count() >= 10,
).execute(gs)

# Graph-shaped queries via typed pattern builder
pattern = P.node("ent:paris").to(P.var("msg"), edge=F.eq("kind", "mentions"))
q.match(pattern, limit=20).execute(gs)

# Self-tuning rules
q.sys.evolve.rule("r1",
    when=[W.cond("recall_hit_rate", "<=", 0.4)],
    then=[A.run("SYS", "REEMBED")],
    cooldown=86400,
).execute(gs)
```

Full reference: [Query builder docs](https://graphstore-docs.orkait.com/query-builder).

**Why prefer the builder for adapters:**
- **Escape-safe by construction.** Every user string flows through the single ``dsl_literal`` helper. ``q.create_node(id, kind="memory", document=untrusted_text)`` is injection-proof regardless of what's in ``untrusted_text``.
- **Clause ordering grammar-correct by construction.** Grammar requires EXPIRES before DOCUMENT in CREATE NODE; the builder emits them in the right order automatically. No more "works in one test, fails in prod because kwarg order changed."
- **Typo / refactor safety.** ``q.create_node(id, kind="memory", contnt="...")`` is a mypy error. String DSL silently stores an extra column.
- **Predicate algebra (`F`)** gives reusable filter libraries. Define ``recent = F.gte("__event_at__", Time.now_minus(30, "d"))`` once, reuse in every query.
- **Immutable chains.** ``base = q.nodes(kind="memory"); top = base.limit(10); hot = base.where(F.gt(...))``. Base never mutates. No "did I accidentally share state?" bugs.
- **Composable via ``|`` (BATCH) and ``.pipe(fn)`` (reusable transformations).** Functional, testable.
- **PEP 561 `py.typed` marker.** Downstream mypy / pyright get full type checking on your adapter.

**Adapter anti-patterns the builder eliminates:**

| Old pattern | Builder equivalent |
|---|---|
| ``f'CREATE NODE "{id}" kind = "memory" DOCUMENT "{text}"'`` | ``q.create_node(id, kind="memory", document=text)`` |
| Manual quote-escaping of user text | Automatic via ``dsl_literal`` |
| Hand-emitting WHERE strings for bulk queries | ``F.eq(...) & F.gt(...) | F.eq(...)`` |
| Duplicate ``CREATE EDGE mentions`` for same (msg, ent) pair | Python ``set`` of seen ent_ids + builder in batch |
