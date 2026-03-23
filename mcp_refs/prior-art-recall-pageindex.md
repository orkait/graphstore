# Prior Art: arniesaha/recall + VectifyAI/PageIndex

Two repos that influenced graphstore's retrieval architecture.

---

## arniesaha/recall

**Repo:** https://github.com/arniesaha/recall

Personal knowledge retrieval over Obsidian vaults (markdown, PDFs, meeting transcripts).
No embeddings, no GPU — purely BM25/FTS5 over SQLite.

**Key techniques:**
- Documents chunked and indexed into SQLite FTS5 (built-in BM25 ranking)
- Temporal expression parsing resolves "last week" / "in January" to date ranges before search
- Name-aware boosting: capitalized proper nouns trigger a two-pass BM25 search (name-only + full query), merged with 3x score boost for name hits and 1.5x for notes titled with that name

**What graphstore took from it:**
- The LEXICAL SEARCH implementation uses the same pattern: FTS5 on the `summaries` table, BM25 ranking, slot-based rowid mapping
- Temporal filtering concept maps to `__expires_at__` / `__created_at__` column queries

---

## VectifyAI/PageIndex

**Repo:** https://github.com/VectifyAI/PageIndex

Vectorless, reasoning-based RAG that replaces embedding similarity with LLM-driven tree traversal.
Core insight: **similarity != relevance** — retrieval should involve reasoning, not nearest-neighbor lookup.

**Key techniques:**
- Indexes a document as a JSON tree: TOC → section → page range, each node with an LLM-generated summary
- Retrieval is tree traversal guided by LLM reasoning (the LLM decides which branches to descend into, like a human scanning a table of contents)
- No embeddings needed; achieves 98.7% on FinanceBench vs lower scores from vector RAG

**What graphstore took from it:**
- The `doc → section → chunk` hierarchy with `__confidence__ = 0.6` on inferred section nodes (see `ingest/chunker.py`)
- The idea that retrieval follows graph structure rather than pure nearest-neighbor: `RECALL FROM "doc:x" DEPTH 2` is the spreading-activation analogue of PageIndex's LLM-guided tree descent
- Node summary + page-range metadata pattern directly maps to the `summaries` table schema (`slot`, `heading`, `page`, `chunk_index`, `doc_slot`)

---

## How they connect in graphstore

```
PageIndex tree traversal          graphstore equivalent
--------------------------------  --------------------------------
TOC node (title + summary)        doc node (kind="document")
Section node (page range)         section node (__confidence__=0.6)
Leaf chunk (raw text)             chunk node (DOCUMENT content)
LLM-guided tree descent           RECALL FROM "doc:x" DEPTH N
BM25 keyword fallback             LEXICAL SEARCH "query"
Vector similarity (optional)      SIMILAR TO "text" / NODE "id"
```

All three retrieval modes (graph, lexical, vector) are available simultaneously in graphstore.
The agent picks the right one per query, or combines them with multiple execute() calls.
