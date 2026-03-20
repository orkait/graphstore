# Document Layer - Design Spec

> Add document ingestion, storage, and cross-document intelligence to graphstore. SQLite as persistent staging area, graph brain as intelligence layer.

**Date:** 2026-03-21
**Status:** Design approved, pending implementation plan
**Depends on:** Vector Store (implemented)

---

## 1. Problem

Agents process documents (PDFs, Word docs, text files) that contain:
- Text too large for column fields (meeting transcripts, reports, articles)
- Images with charts, diagrams, screenshots that need understanding
- Structured metadata (author, pages, date) mixed with unstructured content
- Cross-document relationships the agent can't pre-wire manually

Currently agents must pre-process documents externally and manually create nodes. There's no way to say "ingest this PDF and make it searchable."

## 2. Goal

Three-phase pipeline: INGEST (file to SQLite), EMBED (SQLite to graph brain), CONNECT (auto-wire cross-document relationships).

```sql
INGEST "report.pdf" AS "doc:q3"     -- parse, chunk, store in SQLite + graph
SYS CONNECT                          -- auto-wire cross-document relationships
SIMILAR TO "Q3 revenue growth" LIMIT 5   -- search summaries by meaning
NODE "chunk:3" WITH DOCUMENT         -- fetch full text from disk on demand
```

## 3. Architecture

```
┌──────────────────────────────────────────────────────────┐
│  SQLite (disk) - persistent document storage              │
│                                                           │
│  documents:    full markdown text (cold, read on demand)  │
│  summaries:    chunk summaries (loaded into embedder)     │
│  doc_metadata: pages, author, parser, confidence          │
│  images:       extracted image blobs + descriptions       │
│                                                           │
│  ↓ Phase 2: summaries embedded, metadata columnarized ↓   │
├──────────────────────────────────────────────────────────┤
│  Graph Brain (memory) - intelligence layer                │
│                                                           │
│  ColumnStore: heading, page, summary, confidence, source  │
│  VectorStore: embedded summaries (not full text)          │
│  Graph: doc→chunks, chunk→chunk, cross-doc edges          │
│  live_mask: TTL, retracted, context                       │
└──────────────────────────────────────────────────────────┘
```

**Key principle:** Embed summaries, not full text. Full text stays on disk. Summaries (~200 chars) get embedded and live in the graph. Agent finds relevant summaries via SIMILAR TO, fetches full text from SQLite only when needed.

**Documents always on disk.** Even with `path=None`, DocumentStore uses a temp SQLite file. Full document text never sits in RAM. The graph brain (columns + vectors + edges) is in memory for speed.

## 4. Three Phases

### Phase 1: INGEST (file → SQLite + graph nodes)

```
report.pdf
  → Tiered parser (MarkItDown / PyMuPDF4LLM / Docling)
  → Full markdown + extracted images + metadata
  → Chunker splits into sections
  → Per chunk:
      SQLite: full text (documents table)
      SQLite: summary (summaries table)
      Graph: CREATE NODE kind="chunk" heading="Revenue" summary="Q3 grew 15%..." page=7
      Graph: CREATE EDGE doc→chunk
      Vector: embed summary → VectorStore
  → Per image:
      SQLite: image blob (images table)
      Graph: CREATE NODE kind="image" page=3 description="Bar chart..."
      Graph: CREATE EDGE doc→image, nearby_chunk→image
```

### Phase 2: EMBED (automatic, part of INGEST)

Summaries are embedded by the configured embedder (model2vec default) during INGEST. No separate step needed. The embedder reads summaries, produces vectors, stores in VectorStore. Other models (EmbeddingGemma) can re-embed later via `SYS REEMBED`.

### Phase 3: CONNECT (auto-wire cross-document relationships)

```sql
SYS CONNECT
-- For each chunk with a vector:
--   1. Find top-3 similar chunks from OTHER documents via vector search
--   2. If similarity > 0.85, create edge: chunk:a -[similar_to]-> chunk:b
--   3. Deduplicate (no A→B and B→A)

SYS CONNECT WHERE kind = "report" THRESHOLD 0.9
-- Only connect chunks within "report" kind, higher threshold

CONNECT NODE "chunk:7" THRESHOLD 0.8
-- Wire one specific node to its nearest neighbors
```

This gives agents cross-document associations without manual edge creation.

## 5. SQLite Schema (DocumentStore)

```sql
-- Full document content (cold storage, read on demand)
CREATE TABLE IF NOT EXISTS documents (
    slot INTEGER PRIMARY KEY,
    content BLOB NOT NULL,
    content_type TEXT NOT NULL,
    size INTEGER NOT NULL
);

-- Chunk summaries (loaded into embedder, small enough to scan)
CREATE TABLE IF NOT EXISTS summaries (
    slot INTEGER PRIMARY KEY,
    summary TEXT NOT NULL,
    heading TEXT,
    page INTEGER,
    chunk_index INTEGER,
    doc_slot INTEGER
);

-- Document-level metadata
CREATE TABLE IF NOT EXISTS doc_metadata (
    doc_slot INTEGER PRIMARY KEY,
    source_path TEXT,
    pages INTEGER,
    author TEXT,
    title TEXT,
    parser_used TEXT,
    confidence REAL,
    ingested_at INTEGER
);

-- Extracted images (cold storage)
CREATE TABLE IF NOT EXISTS images (
    slot INTEGER PRIMARY KEY,
    image_data BLOB NOT NULL,
    mime_type TEXT NOT NULL,
    page INTEGER,
    description TEXT
);
```

Python interface:

```python
# graphstore/document/store.py
class DocumentStore:
    def __init__(self, db_path: str | None = None):
        """Initialize with a file path. If None, uses a temp file (cleaned on close)."""
        if db_path:
            self._conn = sqlite3.connect(db_path)
            self._temp = False
        else:
            import tempfile
            self._path = tempfile.mktemp(suffix=".graphstore-docs.db")
            self._conn = sqlite3.connect(self._path)
            self._temp = True
        self._ensure_tables()

    def put_document(self, slot: int, content: bytes, content_type: str) -> None
    def get_document(self, slot: int) -> tuple[bytes, str] | None
    def delete_document(self, slot: int) -> None
    def has_document(self, slot: int) -> bool

    def put_summary(self, slot: int, summary: str, heading: str | None,
                    page: int | None, chunk_index: int, doc_slot: int) -> None
    def get_summary(self, slot: int) -> dict | None
    def get_summaries_for_doc(self, doc_slot: int) -> list[dict]

    def put_metadata(self, doc_slot: int, metadata: dict) -> None
    def get_metadata(self, doc_slot: int) -> dict | None

    def put_image(self, slot: int, image_data: bytes, mime_type: str,
                  page: int | None, description: str | None) -> None
    def get_image(self, slot: int) -> dict | None

    def delete_all_for_doc(self, doc_slot: int) -> None
    """Delete document + all its summaries + images. For cascade delete."""

    def orphan_cleanup(self, live_slots: set[int]) -> int
    """Delete documents/summaries/images whose slots are not in live_slots. For rollback cleanup."""

    def close(self) -> None
    """Close connection. If temp file, delete it."""
```

## 6. Tiered Ingestor Routing

```
Ingest router (deterministic parsers first, VLM as fallback):
  ├── Office / HTML / misc files → MarkItDown     (Tier 1, fastest)
  ├── Normal PDFs                → PyMuPDF4LLM    (Tier 2, good structure)
  ├── Hard PDFs / tables / OCR   → Docling         (Tier 3, best quality)
  └── Image fallback             → SmolVLM2 / Qwen3-VL (Tier 4, via Ollama)
```

| Backend | What it does | Quality | Speed | Install |
|---|---|---|---|---|
| `markitdown` | PDF/Word/Excel/HTML/images → markdown | Good, multi-format | Fastest | ~5 MB (core dep) |
| `pymupdf4llm` | PDF → LLM-optimized markdown + image extraction | Better structure | Fast | ~30 MB (core dep) |
| `docling` | PDF/Word → markdown with table extraction | Best tables | 4 sec/page | ~200 MB (core dep, lazy import) |

**Docling lazy import:** Core dependency in pyproject.toml but never imported at startup. Only loaded when `USING docling` is specified.

**Scanned PDF detection:** After pymupdf4llm extraction, if result averages < 50 chars/page, set `confidence < 0.5` and include warning in metadata. Agent decides whether to re-ingest with Docling.

## 7. Vision Handler

Image understanding via Ollama (local GPU models). VLMs are Tier 4 - used for image description during ingestion, NOT as document parsers.

| Model | VRAM | Quality | Use case |
|---|---|---|---|
| `smolvlm2:2.2b` | ~2 GB | Good captions | Default - photos, screenshots, simple diagrams |
| `qwen3-vl:8b-q4` | ~8-10 GB | Best - charts, tables, OCR | Opt-in - financial reports, complex documents |

Integration via MarkItDown's native LLM plugin:

```python
from markitdown import MarkItDown
from openai import OpenAI

# MarkItDown talks to Ollama via OpenAI-compatible API
md = MarkItDown(
    enable_plugins=True,
    llm_client=OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
    llm_model="smolvlm2:2.2b",
)
result = md.convert("report.pdf")
# Images auto-described inline in markdown output
```

## 8. Chunker

Splits markdown into searchable sections. Generates summaries for each chunk.

```python
@dataclass
class Chunk:
    text: str                    # full chunk text (goes to SQLite)
    summary: str                 # truncated or LLM-summarized (goes to columns + embedder)
    index: int
    heading: str | None = None
    page: int | None = None
    start_char: int = 0
```

Three strategies:
- `chunk_by_heading` - splits on markdown headings (default)
- `chunk_by_paragraph` - splits on double newlines
- `chunk_fixed` - fixed size with overlap

**Summary generation:** Default is first 200 characters of chunk text (free, instant). If LLM is available, can generate proper summaries (future enhancement).

## 9. DSL Commands

### 9.1 INGEST

```lark
ingest_stmt: "INGEST" STRING ingest_as? ingest_kind? using_clause? vision_clause?
ingest_as: "AS" STRING
ingest_kind: "KIND" STRING
using_clause: "USING" IDENTIFIER
vision_clause: "USING" "VISION" STRING
```

```sql
INGEST "report.pdf"
INGEST "report.pdf" AS "doc:q3" KIND "report"
INGEST "report.pdf" USING docling
INGEST "report.pdf" USING VISION "qwen3-vl"
```

What INGEST does:
1. Parse file via tiered router → full markdown + images + metadata
2. Chunk markdown → list of Chunk objects with summaries
3. Store in SQLite: full text (documents), summaries, metadata, images
4. Create parent node: `kind="document"` with metadata in columns
5. Per chunk: create node `kind="chunk"` with summary + heading + page in columns
6. Embed each summary → VectorStore
7. Create edges: parent → each chunk, chunk → nearby images
8. If BIND CONTEXT active, all nodes inherit context
9. Return `{doc_id, chunks: N, images: N, parser: "pymupdf4llm", confidence: 0.95}`

### 9.2 CONNECT / SYS CONNECT

```lark
connect_stmt: "CONNECT" "NODE" STRING threshold_clause?
sys_connect: "CONNECT" where_clause? threshold_clause?
threshold_clause: "THRESHOLD" NUMBER
```

```sql
SYS CONNECT                              -- all chunks, threshold 0.85
SYS CONNECT WHERE kind = "report"        -- only report chunks
SYS CONNECT THRESHOLD 0.9               -- higher threshold
CONNECT NODE "chunk:7" THRESHOLD 0.8    -- wire one node
```

### 9.3 DOCUMENT clause on CREATE

```lark
document_clause: "DOCUMENT" STRING
```

```sql
CREATE NODE "note:1" kind = "note" title = "Meeting" DOCUMENT "Full meeting notes..."
```

Stores the text in DocumentStore (SQLite disk), not in ColumnStore (RAM).

### 9.4 WITH DOCUMENT on NODE

```lark
node_q: "NODE" STRING ("WITH" "DOCUMENT")?
```

```sql
NODE "chunk:3"                -- fast, memory only: {heading, summary, page}
NODE "chunk:3" WITH DOCUMENT  -- includes disk read: {... _document: "full text..."}
```

### 9.5 SYS INGESTORS / SYS STATS DOCUMENTS

```sql
SYS INGESTORS
-- [{name: "markitdown", formats: ["pdf","docx","html",...], status: "active"},
--  {name: "pymupdf4llm", formats: ["pdf"], status: "active"},
--  {name: "docling", formats: ["pdf","docx"], status: "available"}]

SYS STATS DOCUMENTS
-- {document_count: 42, total_bytes: 5242880, chunk_count: 315, image_count: 23}
```

## 10. Interaction with Existing Features

| Feature | Document Interaction |
|---|---|
| CREATE NODE ... DOCUMENT "text" | Store text in DocumentStore (SQLite disk) |
| DELETE NODE (kind="document") | Cascade-delete all chunk + image nodes and their DocumentStore entries |
| DELETE NODE (kind="chunk") | Delete document + summary from SQLite |
| MERGE NODE "a" INTO "b" | If target has no document, copy source's. Otherwise keep target's. |
| RETRACT | Document stays in SQLite (audit trail), node invisible via live_mask |
| SYS EXPIRE | Delete expired documents from SQLite |
| SYS SNAPSHOT/ROLLBACK | Documents not snapshotted. On rollback, run orphan_cleanup on DocumentStore. |
| BATCH rollback | Delete documents created in batch |
| BIND CONTEXT + INGEST | Chunks inherit active context |
| SIMILAR TO | Searches embedded summaries (memory). Agent fetches full text with WITH DOCUMENT (disk). |
| RECALL | Follows graph edges to related chunks. Agent fetches full text on demand. |
| AGGREGATE | Can GROUP BY heading, page, confidence - all in columns |

### 10.1 Behavioural Rules

**Documents always on disk.** Even with `path=None`, DocumentStore uses a temp SQLite file. Full text never in RAM. Temp file cleaned on `close()`.

**Duplicate ingestion.** `INGEST "file.pdf" AS "doc:report"` raises `NodeExists` if "doc:report" exists. Auto-generated IDs (content-hash) are unique.

**Cascade delete.** Deleting a node with `kind="document"` cascade-deletes outgoing edges to `kind="chunk"` and `kind="image"` nodes, plus their DocumentStore entries. Regular nodes don't cascade.

**Scanned PDF detection.** PyMuPDF4LLM result with < 50 chars/page average → `confidence < 0.5`, warning in metadata.

**Docling lazy import.** Never imported at startup. Only when `USING docling`.

## 11. File Structure

```
graphstore/
  document/                        # NEW package
    __init__.py
    store.py                       # DocumentStore: SQLite tables + CRUD

  ingest/                          # NEW package
    __init__.py
    base.py                        # Ingestor protocol, IngestResult, Chunk, ExtractedImage
    markitdown_ingestor.py         # Tier 1: general files
    pymupdf4llm_ingestor.py        # Tier 2: PDF with structure + image extraction
    docling_ingestor.py            # Tier 3: hard PDFs (lazy import)
    chunker.py                     # Split markdown → chunks with summaries
    vision.py                      # VisionHandler (Ollama client)
    router.py                      # Tiered routing + ingest_file()
    connector.py                   # SYS CONNECT: auto-wire cross-doc relationships
```

## 12. Dependencies

```toml
[project.dependencies]
# ... existing (numpy, scipy, lark, usearch, model2vec) ...
markitdown = ">=0.1"       # 5MB, Microsoft, Tier 1
pymupdf4llm = ">=0.1"     # 30MB, PDF → markdown, Tier 2
docling = ">=2.0"          # 200MB, IBM, Tier 3 (lazy import)
```

Vision models managed via Ollama CLI:
```bash
graphstore install-vision smolvlm2
graphstore install-vision qwen3-vl
```

## 13. GraphStore Constructor

```python
class GraphStore:
    def __init__(
        self,
        path: str | None = None,
        ceiling_mb: int = 256,
        embedder: str | Embedder = "default",
        vision_model: str | None = None,
        vision_base_url: str = "http://localhost:11434/v1",
        allow_system_queries: bool = True,
    ):
        # Document store: always on disk
        if path:
            doc_db = os.path.join(path, "documents.db")
        else:
            doc_db = None  # DocumentStore creates temp file
        self._document_store = DocumentStore(doc_db)
```

## 14. GPU Budget (RTX 3060 12GB)

```
SmolVLM2 (default vision):    2 GB   ← image understanding during INGEST
EmbeddingGemma (opt-in):      1-2 GB ← text embeddings (if upgraded from model2vec)
Free:                          8-9 GB ← room for Qwen3-VL or text LLM

Ollama manages model loading/unloading:
- Vision model loaded during INGEST, unloaded after
- Models don't compete for VRAM
```

## 15. Performance Targets

| Operation | Target | Notes |
|---|---|---|
| INGEST 10-page PDF (text only) | < 2 sec | pymupdf4llm, no vision |
| INGEST 10-page PDF (with vision) | < 10 sec | + SmolVLM2 image descriptions |
| NODE "x" WITH DOCUMENT | < 1 ms | SQLite point lookup by slot |
| SIMILAR TO (searches summaries) | < 500 μs | Same as vector search (summaries embedded) |
| SYS CONNECT (1000 chunks) | < 5 sec | O(n) vector searches |
| CONNECT NODE (single) | < 50 ms | One vector search + edge creation |
| Summary generation (truncation) | < 1 ms | First 200 chars |
| Document storage overhead | ~0 RAM | All on disk |

## 16. The Full Agent Flow

```python
g = GraphStore(path="./brain", vision_model="smolvlm2")

# Phase 1+2: Ingest (parse → SQLite → graph + vectors)
g.execute('INGEST "/docs/q3_report.pdf" AS "doc:q3" KIND "report"')
g.execute('INGEST "/docs/q4_targets.pdf" AS "doc:q4" KIND "report"')

# Phase 3: Connect (auto-wire cross-document relationships)
g.execute('SYS CONNECT WHERE kind = "report"')

# Agent query: find relevant chunks by meaning
results = g.execute('SIMILAR TO "revenue growth targets" LIMIT 5')
# Returns chunks with summaries from BOTH documents

# Agent reads full text of best match
doc = g.execute('NODE "chunk:q3:7" WITH DOCUMENT')
# {heading: "Revenue", summary: "Q3 revenue grew 15%...", _document: "## Revenue\n\nQ3..."}

# Agent follows cross-document connections
related = g.execute('RECALL FROM "chunk:q3:7" DEPTH 2 LIMIT 10')
# Finds chunk:q4:3 (Q4 targets) because SYS CONNECT wired them
```

## 17. CLI Commands

```bash
graphstore install-vision smolvlm2       # ollama pull smolvlm2:2.2b
graphstore install-vision qwen3-vl       # ollama pull qwen3-vl:8b-q4
graphstore list-vision                   # list installed vision models
graphstore list-ingestors                # show available backends
```
