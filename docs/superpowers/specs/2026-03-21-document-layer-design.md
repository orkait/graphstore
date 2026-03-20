# Document Layer - Design Spec

> Add document ingestion, storage, and image understanding to graphstore - completing the four-engine architecture: columns + graph + vectors + documents.

**Date:** 2026-03-21
**Status:** Design approved, pending implementation plan
**Depends on:** Vector Store (implemented)

---

## 1. Problem

Agents process documents (PDFs, Word docs, text files) that contain:
- Text too large for column fields (meeting transcripts, reports, articles)
- Images with charts, diagrams, screenshots that need understanding
- Structured metadata (author, pages, date) mixed with unstructured content

Currently agents must pre-process documents externally and manually create nodes. There's no way to say "ingest this PDF and make it searchable."

## 2. Goal

One command ingests a document, extracts text + images, chunks it, stores it, embeds it, and makes it searchable:

```sql
INGEST "report.pdf"
SIMILAR TO "Q3 revenue growth" LIMIT 5
NODE "chunk:3" WITH DOCUMENT
```

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                          DSL                                 │
└──────┬──────────┬──────────┬──────────┬─────────────────────┘
       ▼          ▼          ▼          ▼          ▼
 ColumnStore    Graph    VectorStore  DocStore   Ingestor
  (numpy)    (scipy CSR) (usearch)   (SQLite)    (NEW)

 structured  relationships meaning   raw payload  PDF/Word/text
 fields      edges        vectors    blobs, text  → markdown
 WHERE       RECALL       SIMILAR    WITH DOC     → chunks
 GROUP BY    TRAVERSE     TO                      → images
```

## 4. Four New Components

### 4.1 DocumentStore (SQLite)

Stores raw payloads (full text, JSON, binary blobs) on disk. Keyed by node slot. Not in memory - read on demand.

```sql
-- SQLite table (alongside existing blobs/metadata/wal tables)
CREATE TABLE IF NOT EXISTS documents (
    slot INTEGER PRIMARY KEY,
    content BLOB NOT NULL,
    content_type TEXT NOT NULL,   -- "text/plain", "text/markdown", "application/json", "image/png"
    size INTEGER NOT NULL
);
```

Python interface:

```python
# graphstore/document/store.py
class DocumentStore:
    def __init__(self, conn):
        self._conn = conn
        self._ensure_table()

    def put(self, slot: int, content: bytes, content_type: str) -> None
    def get(self, slot: int) -> tuple[bytes, str] | None  # (content, content_type)
    def delete(self, slot: int) -> None
    def has(self, slot: int) -> bool
    def size(self, slot: int) -> int | None
```

### 4.2 Ingestor

Converts documents (PDF, Word, text, HTML) to markdown text. Three built-in backends, agent picks per document.

| Backend | What it does | Quality | Speed | Install |
|---|---|---|---|---|
| `markitdown` | PDF/Word/Excel/HTML/images → markdown | Good, multi-format | Fastest | ~5 MB (core dep) |
| `pymupdf4llm` | PDF → LLM-optimized markdown | Better structure | Fast | ~30 MB (core dep) |
| `docling` | PDF/Word → markdown with table extraction | Best tables | 4 sec/page | ~200 MB (core dep) |

```python
# graphstore/ingest/base.py
class Ingestor:
    def convert(self, file_path: str, **kwargs) -> IngestResult

@dataclass
class IngestResult:
    markdown: str                          # full document as markdown
    images: list[ExtractedImage]           # extracted images with positions
    metadata: dict                         # pages, author, title, etc.

@dataclass
class ExtractedImage:
    data: bytes                            # raw image bytes
    mime_type: str                         # "image/png", "image/jpeg"
    page: int | None                       # source page number
    caption: str | None                    # from document, if any
    description: str | None                # from vision model, if configured
```

### 4.3 Chunker

Splits markdown text into searchable sections. Three strategies:

```python
# graphstore/ingest/chunker.py
def chunk_by_heading(text: str, max_chunk_size: int = 2000) -> list[Chunk]
def chunk_by_paragraph(text: str, max_chunk_size: int = 1000) -> list[Chunk]
def chunk_fixed(text: str, chunk_size: int = 500, overlap: int = 50) -> list[Chunk]

@dataclass
class Chunk:
    text: str
    index: int               # chunk number
    heading: str | None      # parent heading if chunked by heading
    start_char: int          # position in original document
```

Default: `chunk_by_heading` - splits on markdown headings, falls back to paragraph if no headings.

### 4.4 Vision Handler

Image understanding via Ollama (local GPU models). Two models supported:

| Model | VRAM | Quality | Speed | Use case |
|---|---|---|---|---|
| `smolvlm2:2.2b` | ~2 GB | Good for captions | ~50 tok/sec | Default - photos, screenshots, simple diagrams |
| `qwen3-vl:8b-q4` | ~8-10 GB | Best - charts, tables, OCR | ~12 tok/sec | Opt-in - financial reports, complex documents |

**Integration:** MarkItDown natively supports LLM-based image description via any OpenAI-compatible client. SmolVLM2 and Qwen3-VL run on Ollama which exposes an OpenAI-compatible API.

```python
# graphstore/ingest/vision.py
from openai import OpenAI

class VisionHandler:
    def __init__(self, model: str = "smolvlm2:2.2b", base_url: str = "http://localhost:11434/v1"):
        self._client = OpenAI(base_url=base_url, api_key="ollama")
        self._model = model

    def describe(self, image_bytes: bytes) -> str:
        # Uses OpenAI chat completions API with image
        ...
```

MarkItDown integration:
```python
from markitdown import MarkItDown

md = MarkItDown(
    enable_plugins=True,
    llm_client=vision_handler._client,
    llm_model=vision_handler._model,
)
result = md.convert("report.pdf")
# Images auto-described inline in the markdown output
```

## 5. DSL Commands

### 5.1 INGEST

```lark
ingest_stmt: "INGEST" STRING ("AS" STRING)? ("KIND" STRING)? using_clause? vision_clause?
using_clause: "USING" IDENTIFIER          -- "markitdown", "pymupdf4llm", "docling"
vision_clause: "USING" "VISION" STRING    -- "smolvlm2", "qwen3-vl"
```

Examples:
```sql
-- Default: markitdown, auto-generated ID
INGEST "/path/to/report.pdf"

-- Specify parent node ID and kind
INGEST "/path/to/report.pdf" AS "doc:q3-report" KIND "document"

-- Choose ingestor backend
INGEST "/path/to/report.pdf" USING pymupdf4llm

-- Choose vision model for image-heavy documents
INGEST "/path/to/financials.pdf" USING VISION "qwen3-vl"

-- Combine
INGEST "/path/to/report.pdf" AS "doc:report" KIND "document" USING docling USING VISION "qwen3-vl"
```

**What INGEST does internally:**

1. Read file from disk
2. Convert to markdown via chosen ingestor (images described if vision configured)
3. Extract metadata (pages, author, title)
4. Chunk markdown into sections
5. Create parent node: `CREATE NODE "doc:report" kind = "document" pages = 42 title = "Q3 Report"`
6. For each chunk: `CREATE NODE "chunk:N" kind = "chunk" heading = "Revenue" DOCUMENT "Full chunk text..."`
7. Auto-embed each chunk (via configured embedder)
8. Create edges: `doc:report -> chunk:1`, `doc:report -> chunk:2`, ...
9. For extracted images: `CREATE NODE "image:N" kind = "image" page = 3 DOCUMENT <image_bytes>`
10. Create edges: `doc:report -> image:1`, `chunk:3 -> image:1` (image near chunk 3)

### 5.2 DOCUMENT clause on CREATE

```lark
create_node: "CREATE" "NODE" STRING field_pairs vector_clause? document_clause? expires_clause?
document_clause: "DOCUMENT" STRING           -- text/markdown content
               | "DOCUMENT" "BINARY" STRING  -- base64-encoded binary
```

### 5.3 WITH DOCUMENT on reads

```lark
node_q: "NODE" STRING ("WITH" "DOCUMENT")?
```

Default `NODE "x"` returns structured fields only (from ColumnStore, no disk IO).
`NODE "x" WITH DOCUMENT` also fetches the raw document from SQLite.

```python
# Without DOCUMENT (default, fast, memory only)
result = g.execute('NODE "chunk:3"')
# {id: "chunk:3", kind: "chunk", heading: "Revenue"}

# With DOCUMENT (includes SQLite read)
result = g.execute('NODE "chunk:3" WITH DOCUMENT')
# {id: "chunk:3", kind: "chunk", heading: "Revenue", _document: "## Revenue\n\nQ3 revenue grew 15%..."}
```

### 5.4 SYS INGESTORS

```sql
SYS INGESTORS
-- Returns: [{name: "markitdown", status: "active", formats: ["pdf","docx","html",...]},
--           {name: "pymupdf4llm", status: "active", formats: ["pdf"]},
--           {name: "docling", status: "active", formats: ["pdf","docx"]}]
```

## 6. Dependencies

```toml
[project.dependencies]
# ... existing ...
markitdown = ">=0.1"       # 5MB, Microsoft, PDF/Word/Excel/HTML → markdown
pymupdf4llm = ">=0.1"     # 30MB, PDF → LLM-optimized markdown
docling = ">=2.0"          # 200MB, IBM, best tables + structure
```

All three ship as core deps. No opt-in tiers for ingestors.

Vision models are managed via Ollama (external binary, not a Python dep):
```bash
graphstore install-vision smolvlm2        # ollama pull smolvlm2:2.2b
graphstore install-vision qwen3-vl        # ollama pull qwen3-vl:8b-q4
```

## 7. GraphStore Constructor Update

```python
class GraphStore:
    def __init__(
        self,
        path: str | None = None,
        ceiling_mb: int = 256,
        embedder: str | Embedder = "default",
        vision_model: str | None = None,       # NEW: "smolvlm2", "qwen3-vl", or None
        vision_base_url: str = "http://localhost:11434/v1",  # NEW: Ollama endpoint
        allow_system_queries: bool = True,
    ):
```

If `vision_model` is set, MarkItDown is initialized with the Ollama client for image descriptions. If None, images are stored as blobs without descriptions.

## 8. File Structure

```
graphstore/
  # ... existing ...

  document/                      # NEW package
    __init__.py
    store.py                     # DocumentStore: SQLite blob storage

  ingest/                        # NEW package
    __init__.py
    base.py                      # Ingestor protocol + IngestResult
    markitdown_ingestor.py       # MarkItDown wrapper
    pymupdf4llm_ingestor.py      # PyMuPDF4LLM wrapper
    docling_ingestor.py          # Docling wrapper
    chunker.py                   # Text chunking strategies
    vision.py                    # VisionHandler (Ollama client)
    registry.py                  # Ingestor registry + file extension mapping
```

## 9. Interaction with Existing Features

| Feature | Document Interaction |
|---|---|
| CREATE NODE ... DOCUMENT "text" | Store text in DocumentStore (SQLite) |
| DELETE NODE | Remove document from DocumentStore. If kind="document", cascade-delete all chunk nodes connected via outgoing edges. |
| MERGE NODE "a" INTO "b" | If target has no document, copy source's document to target. If target has document, keep target's. |
| RETRACT | Document stays (audit trail), node invisible |
| SYS EXPIRE | Remove expired documents from DocumentStore |
| SYS SNAPSHOT/ROLLBACK | Documents not snapshotted (too large). On ROLLBACK, scan DocumentStore for orphaned slots (slots with documents but no live graph node) and delete them. |
| SIMILAR TO | Finds chunks by embedding. Agent fetches document with WITH DOCUMENT. |
| RECALL | Graph traversal finds related chunks. Agent fetches document with WITH DOCUMENT. |
| BATCH rollback | Documents created in batch are deleted on rollback. |
| BIND CONTEXT + INGEST | Chunks created by INGEST inherit the active context. |
| Persistence | DocumentStore IS SQLite - already persistent. |
| path=None (in-memory mode) | DocumentStore uses `sqlite3.connect(":memory:")`. Documents exist in memory only, lost on close. Consistent with graph behavior. |

### 9.1 Behavioural Rules

**In-memory mode:** When `path=None`, DocumentStore uses an in-memory SQLite connection. INGEST and DOCUMENT work normally but data is lost on close. This is consistent - without a path, everything is ephemeral.

**Duplicate ingestion:** If `INGEST "file.pdf" AS "doc:report"` and "doc:report" already exists, raise `NodeExists`. Agent must delete the existing document first or use a different ID. If no AS clause, auto-generated IDs (content-hash) are always unique.

**Cascade delete:** When a node with `kind = "document"` is deleted, all outgoing edge targets that have `kind = "chunk"` or `kind = "image"` are also deleted (cascade). This prevents orphaned chunks. Regular node deletion (non-document kind) does NOT cascade.

**Scanned PDF detection:** After pymupdf4llm extraction, if the result averages less than 50 characters per page, the IngestResult.confidence is set to < 0.5 and a warning is included in metadata: `{"warning": "Low text extraction. Consider: USING docling"}`. The agent can decide whether to re-ingest with a different backend.

**Docling lazy import:** Docling is a core dependency (in pyproject.toml) but is never imported at startup. The `DoclingIngestor` class imports `docling` only when `USING docling` is specified. This keeps startup time and memory usage low for agents that don't need Docling.

## 10. The Full Agent Flow

```python
g = GraphStore(
    path="./agent_brain",
    vision_model="smolvlm2",    # local GPU, 2GB VRAM
)

# Agent receives a PDF from the user
g.execute('INGEST "/tmp/q3_report.pdf" AS "doc:q3" KIND "report"')
# → markitdown converts PDF to markdown (images described by SmolVLM2)
# → chunked into 15 sections
# → each chunk embedded by model2vec
# → all stored: columns (metadata) + vectors (embeddings) + documents (full text)

# Agent needs to answer "What was Q3 revenue?"
results = g.execute('SIMILAR TO "Q3 revenue numbers" LIMIT 3')
# → vector search finds chunk:7 (heading: "Financial Results")

doc = g.execute('NODE "chunk:7" WITH DOCUMENT')
# → returns full markdown text of that section
# → agent reads it and answers the user

# Agent also checks graph associations
context = g.execute('RECALL FROM "chunk:7" DEPTH 2 LIMIT 10')
# → finds related chunks and images via edges

# Agent finds all charts in the document
charts = g.execute('NODES WHERE kind = "image" AND page > 0')
for chart in charts.data:
    detail = g.execute(f'NODE "{chart["id"]}" WITH DOCUMENT')
    # → returns image description + raw bytes
```

## 11. GPU Budget (RTX 3060 12GB)

```
SmolVLM2 (default vision):    2 GB   ← image understanding
EmbeddingGemma (opt-in):      1-2 GB ← text embeddings (if upgraded from model2vec)
Free:                          8-9 GB ← room for Qwen3-VL when needed, or text LLM

Ollama manages model loading/unloading:
- SmolVLM2 loaded during INGEST, unloaded after
- Qwen3-VL loaded on demand for complex documents
- Models don't compete for VRAM - Ollama handles scheduling
```

## 12. Performance Targets

| Operation | Target | Notes |
|---|---|---|
| INGEST 10-page PDF (text only) | < 2 sec | markitdown, no vision |
| INGEST 10-page PDF (with images) | < 10 sec | markitdown + SmolVLM2 |
| INGEST 10-page PDF (docling) | ~40 sec | 4 sec/page |
| NODE "x" WITH DOCUMENT | < 1 ms | SQLite point lookup |
| DOCUMENT store/retrieve | < 500 μs | SQLite blob I/O |
| Chunk 10-page document | < 50 ms | Pure Python string split |
| Memory overhead | ~0 | Documents on disk, not in RAM |

## 13. CLI Commands

```bash
# Vision model management
graphstore install-vision smolvlm2       # ollama pull smolvlm2:2.2b
graphstore install-vision qwen3-vl       # ollama pull qwen3-vl:8b-q4
graphstore list-vision                   # list installed vision models

# Ingestor info
graphstore list-ingestors                # show available backends
```
