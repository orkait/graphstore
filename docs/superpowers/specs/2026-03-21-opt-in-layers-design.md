# Opt-in Layers - Consolidated Design Spec

> Complete product spec for graphstore's opt-in capabilities: document ingestion, vision, voice, and embedder upgrades.

**Date:** 2026-03-21
**Status:** Design approved
**Depends on:** Agentic Brain DB (implemented), Vector Store (implemented)

---

## 1. Product Overview

```
pip install graphstore                              ~85 MB
├── ColumnStore (numpy)                             structured fields
├── Graph (scipy CSR)                               relationships
├── VectorStore (usearch + model2vec)               semantic search
├── DocumentStore (SQLite, always on disk)           raw documents
├── Ingestors (markitdown + pymupdf4llm + docling)  file parsing
└── DSL (lark)                                      one query language

graphstore install-embedder embeddinggemma          +~300 MB
└── EmbeddingGemma-300M via ONNX (higher quality vectors)

graphstore install-vision smolvlm2                  +~2 GB
└── SmolVLM2-2.2B via Ollama (image understanding)

graphstore install-vision qwen3-vl                  +~8 GB
└── Qwen3-VL-8B via Ollama (premium vision)

graphstore install-voice                            +~100 MB
├── Moonshine (STT, real-time streaming, CPU)
└── Piper (TTS, clear speech, CPU)
```

## 2. Document Layer

### 2.1 Architecture

SQLite is the persistent staging area. Graph brain is the intelligence layer.

```
File → Tiered Parser → Markdown + Images → SQLite (disk, cold)
                                          → Summaries → ColumnStore + VectorStore (memory, hot)
                                          → Edges → Graph (memory)
                                          → SYS CONNECT → cross-doc edges (memory)
```

**Embed summaries, not full text.** Summaries (~200 chars) get embedded and live in the graph. Full text stays on disk. Agent finds relevant summaries via SIMILAR TO, fetches full text with WITH DOCUMENT.

**Documents always on disk.** Even with `path=None`, DocumentStore uses a temp SQLite file cleaned on close(). Full text never in RAM.

### 2.2 SQLite Schema

```sql
CREATE TABLE IF NOT EXISTS documents (
    slot INTEGER PRIMARY KEY,
    content BLOB NOT NULL,
    content_type TEXT NOT NULL,
    size INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS summaries (
    slot INTEGER PRIMARY KEY,
    summary TEXT NOT NULL,
    heading TEXT,
    page INTEGER,
    chunk_index INTEGER,
    doc_slot INTEGER
);

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

CREATE TABLE IF NOT EXISTS images (
    slot INTEGER PRIMARY KEY,
    image_data BLOB NOT NULL,
    mime_type TEXT NOT NULL,
    page INTEGER,
    description TEXT
);
```

Separate file from main graphstore.db:
```
./brain/
  graphstore.db    # hot: columns, edges, vectors, WAL, metadata
  documents.db     # cold: full text, images, summaries
```

### 2.3 DocumentStore API

```python
class DocumentStore:
    def __init__(self, db_path: str | None = None):
        """File path or temp file if None."""

    def put_document(self, slot, content, content_type) -> None
    def get_document(self, slot) -> tuple[bytes, str] | None
    def delete_document(self, slot) -> None
    def has_document(self, slot) -> bool

    def put_summary(self, slot, summary, heading, page, chunk_index, doc_slot) -> None
    def get_summary(self, slot) -> dict | None
    def get_summaries_for_doc(self, doc_slot) -> list[dict]

    def put_metadata(self, doc_slot, metadata) -> None
    def get_metadata(self, doc_slot) -> dict | None

    def put_image(self, slot, image_data, mime_type, page, description) -> None
    def get_image(self, slot) -> dict | None

    def delete_all_for_doc(self, doc_slot) -> None
    def orphan_cleanup(self, live_slots: set[int]) -> int
    def stats(self) -> dict  # {document_count, total_bytes, chunk_count, image_count}
    def close(self) -> None
```

### 2.4 Tiered Ingestor Routing

Deterministic parsers first. VLM as fallback/enrichment only.

```
Tier 1: MarkItDown     → Office, HTML, CSV, JSON, XML, text, misc
Tier 2: PyMuPDF4LLM    → Normal PDFs (structure + image extraction)
Tier 3: Docling         → Hard PDFs (tables, OCR, formulas) - lazy import
Tier 4: VLM             → Image description (SmolVLM2/Qwen3-VL via Ollama)
```

Extension mapping:
```
.txt .md .html .htm .csv .json .xml .docx .pptx .xlsx → markitdown
.pdf → pymupdf4llm
.png .jpg .jpeg .gif .webp → VLM (if installed) or markitdown
.wav .mp3 .ogg .flac → moonshine (if voice installed)
```

Scanned PDF detection: if pymupdf4llm result < 50 chars/page average, set confidence < 0.5, include warning.

Docling: core dep in pyproject.toml, lazy import (only loaded when `USING docling`).

### 2.5 Chunker

```python
@dataclass
class Chunk:
    text: str           # full text → SQLite
    summary: str        # first 200 chars → columns + embedder
    index: int
    heading: str | None
    page: int | None
    start_char: int

def chunk_by_heading(text, max_chunk_size=2000) -> list[Chunk]   # default
def chunk_by_paragraph(text, max_chunk_size=1000) -> list[Chunk]
def chunk_fixed(text, chunk_size=500, overlap=50) -> list[Chunk]
```

Summary: first 200 characters of chunk text (free, instant).

### 2.6 INGEST Pipeline

```sql
INGEST "report.pdf" AS "doc:q3" KIND "report" USING pymupdf4llm USING VISION "smolvlm2"
```

Steps (atomic - rollback SQLite on graph failure):
1. Parse file via tiered router → markdown + images + metadata
2. If vision installed + images found → describe images via VLM
3. Chunk markdown → list of Chunk objects with summaries
4. **SQLite writes:** full text, summaries, metadata, images
5. **Graph writes:** parent node (kind="document"), chunk nodes, image nodes
6. **Vector writes:** embed each summary → VectorStore
7. **Edge writes:** parent→chunks, chunks→nearby images
8. If BIND CONTEXT active, all nodes inherit context
9. Return `{doc_id, chunks: N, images: N, parser, confidence}`

If step 5-7 fails (e.g. ceiling exceeded), delete SQLite entries from step 4.

### 2.7 SYS CONNECT (cross-document wiring)

```sql
SYS CONNECT                              -- all chunks, threshold 0.85
SYS CONNECT WHERE kind = "report"        -- scoped
SYS CONNECT THRESHOLD 0.9               -- higher threshold
CONNECT NODE "chunk:7" THRESHOLD 0.8    -- single node
```

For each chunk with a vector, find top-3 similar chunks from OTHER documents. If similarity > threshold, create edge `kind="similar_to"`. Skip already-connected pairs. Respects live_mask.

### 2.8 Supported Formats

```
Text:     .txt, .md, .csv, .json, .xml, .html, .htm
Office:   .docx, .pptx, .xlsx
PDF:      .pdf (text or scanned)
Images:   .png, .jpg, .jpeg, .gif, .webp (needs vision opt-in for descriptions)
Audio:    .wav, .mp3, .ogg, .flac (needs voice opt-in for transcription)
```

Unsupported formats: clear error listing supported extensions.

## 3. Vision Layer (opt-in)

```bash
graphstore install-vision smolvlm2    # ollama pull smolvlm2:2.2b (~2GB)
graphstore install-vision qwen3-vl    # ollama pull qwen3-vl:8b-q4 (~8GB)
```

Integrated into MarkItDown via its native LLM plugin:

```python
from markitdown import MarkItDown
from openai import OpenAI

md = MarkItDown(
    enable_plugins=True,
    llm_client=OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
    llm_model="smolvlm2:2.2b",
)
```

If Ollama not running or model not pulled: degrade gracefully. Images stored without descriptions. Warning in INGEST result.

If agent requests `USING VISION "qwen3-vl"` but only smolvlm2 installed: error with install instructions.

## 4. Voice Layer (opt-in)

```bash
graphstore install-voice
# pip install piper-tts moonshine
# Downloads default Piper voice (~50MB)
```

### 4.1 STT: Moonshine

Real-time streaming transcription. CPU only, < 1GB memory, < 200ms latency.

```python
g.listen(on_text=callback)    # start streaming STT
g.stop_listening()            # stop
```

Also used for audio file ingestion:
```sql
INGEST "meeting.wav"
-- Moonshine transcribes → text → chunked → embedded → searchable
```

### 4.2 TTS: Piper

Clear, functional speech. CPU only, no GPU needed, < 100ms latency. Not emotional, not natural - just communicates thought clearly.

```python
g.speak("The Q3 revenue grew 15% based on the March report")
```

### 4.3 Python API (not DSL)

Voice is streaming - doesn't fit request/response DSL pattern.

```python
g = GraphStore(path="./brain", voice=True)

g.listen(on_text=lambda text: agent.process(text))
g.speak("The answer is 42")
g.stop_listening()
```

Safeguards:
- `g.listen()` when already listening → raises "Already listening"
- `g.speak()` without voice installed → raises with install instructions
- `g.listen()` without `on_text` callback → raises "on_text callback required"
- `g.stop_listening()` when not listening → no-op

## 5. Embedder Upgrade (opt-in)

```bash
graphstore install-embedder embeddinggemma
# pip install onnxruntime tokenizers huggingface_hub
# Downloads ONNX model (~200MB q4) to ~/.graphstore/models/
```

### 5.1 Embedder Switch Safety

Switching embedders invalidates all existing vectors (different embedding spaces produce incompatible vectors).

```sql
SYS SET EMBEDDER "embeddinggemma-300m"
-- Warning: 48000 existing vectors will be invalidated.
-- SIMILAR TO queries blocked until SYS REEMBED runs.

SYS REEMBED
-- Re-reads all summaries from DocumentStore SQLite
-- Re-embeds with new model
-- Replaces all vectors in VectorStore
-- Unblocks SIMILAR TO queries
```

`_embedder_dirty` flag on GraphStore:
- Set to True by SYS SET EMBEDDER
- SIMILAR TO raises "Embedder changed. Run SYS REEMBED." while dirty
- Set to False after SYS REEMBED completes

### 5.2 SYS REEMBED Flow

1. Read all summaries from DocumentStore SQLite (summaries table)
2. Batch-encode with new embedder
3. Replace vectors in VectorStore (remove old, add new)
4. Clear `_embedder_dirty` flag

## 6. Unified SYS STATUS

```sql
SYS STATUS
```

Returns complete system state:

```json
{
    "nodes": 50000,
    "edges": 120000,
    "memory_mb": 45,
    "vectors": 48000,
    "vector_memory_mb": 52,
    "documents": 150,
    "document_size_mb": 23,
    "embedder": {"name": "model2vec", "dims": 256, "dirty": false},
    "vision": "smolvlm2",
    "voice": {"stt": "moonshine", "tts": "piper"},
    "ollama": "running"
}
```

Fields are null/omitted when opt-in not installed.

## 7. Behavioural Rules

### 7.1 INGEST Atomicity

Wrap entire INGEST in try/except. If graph writes fail, delete SQLite entries.

### 7.2 Cascade Delete

Deleting a node with `kind="document"`: cascade-delete all outgoing chunk + image nodes and their DocumentStore entries.

### 7.3 RETRACT Cascade

RETRACT on a document node: also retract all chunk + image nodes.

### 7.4 MERGE Documents

If target has no document and source has one, copy source's document to target.

### 7.5 Snapshot/Rollback + Documents

Documents not snapshotted (too large, on disk). On rollback, run `orphan_cleanup` on DocumentStore to remove entries whose graph nodes no longer exist.

### 7.6 Duplicate Ingestion

`INGEST "file" AS "doc:x"` raises NodeExists if "doc:x" exists. Auto-generated IDs (content-hash) are unique.

### 7.7 SYS CONNECT Idempotency

Second SYS CONNECT skips pairs that already have a "similar_to" edge.

### 7.8 Context + INGEST

Chunks created by INGEST inherit the active BIND CONTEXT.

## 8. DSL Commands Summary

### Document commands
```sql
INGEST "file" [AS "id"] [KIND "kind"] [USING backend] [USING VISION "model"]
CREATE NODE "x" ... DOCUMENT "text"
NODE "x" WITH DOCUMENT
SYS CONNECT [WHERE expr] [THRESHOLD n]
CONNECT NODE "x" [THRESHOLD n]
SYS INGESTORS
SYS STATS DOCUMENTS
```

### Embedder commands
```sql
SYS SET EMBEDDER "name" [DIMS n]
SYS REEMBED
SYS EMBEDDERS
```

### Status
```sql
SYS STATUS    -- unified: all engines + opt-ins + health
```

### Voice (Python API, not DSL)
```python
g.speak("text")
g.listen(on_text=callback)
g.stop_listening()
```

## 9. File Structure

```
graphstore/
  document/
    __init__.py
    store.py                     # DocumentStore: SQLite multi-table CRUD

  ingest/
    __init__.py
    base.py                      # Ingestor protocol, IngestResult, Chunk, ExtractedImage
    markitdown_ingestor.py       # Tier 1
    pymupdf4llm_ingestor.py      # Tier 2
    docling_ingestor.py          # Tier 3 (lazy import)
    chunker.py                   # Heading/paragraph/fixed splitting + summary generation
    vision.py                    # VisionHandler (Ollama client for Tier 4)
    router.py                    # Tiered routing + ingest_file()
    connector.py                 # SYS CONNECT: auto-wire cross-doc relationships

  voice/
    __init__.py
    stt.py                       # Moonshine STT wrapper
    tts.py                       # Piper TTS wrapper
```

## 10. Dependencies

```toml
[project.dependencies]
numpy = ">=1.24"
scipy = ">=1.10"
lark = ">=1.1"
usearch = ">=2.0"
model2vec = ">=0.4"
markitdown = ">=0.1"
pymupdf4llm = ">=0.1"
docling = ">=2.0"              # lazy import, only loaded on USING docling

# Voice, vision, embedder-upgrade installed via CLI, not listed here
```

## 11. GPU Budget (RTX 3060 12GB)

```
SmolVLM2 (vision, opt-in):     2 GB     loaded during INGEST only
EmbeddingGemma (opt-in):       1-2 GB   loaded during embed/search
Moonshine (voice, opt-in):     CPU      no GPU
Piper (voice, opt-in):         CPU      no GPU
Free for LLM:                  8-9 GB
```

## 12. Performance Targets

| Operation | Target |
|---|---|
| INGEST 10-page PDF (text) | < 2 sec |
| INGEST 10-page PDF (+ vision) | < 10 sec |
| NODE "x" WITH DOCUMENT | < 1 ms |
| SIMILAR TO (summaries) | < 500 μs |
| SYS CONNECT (1k chunks) | < 5 sec |
| SYS REEMBED (100k summaries) | < 30 sec |
| g.speak() latency | < 100 ms |
| g.listen() latency | < 200 ms |
| Moonshine transcription | real-time streaming |
| Document storage RAM overhead | ~0 (all on disk) |
