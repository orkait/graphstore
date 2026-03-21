# Document Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add document ingestion, persistent storage, and cross-document intelligence. SQLite as disk staging area, graph brain as intelligence layer.

**Architecture:** Three-phase pipeline: INGEST (file → SQLite + graph nodes with summaries), EMBED (summaries → VectorStore, automatic), CONNECT (auto-wire cross-document relationships via vector similarity). DocumentStore is always on disk (SQLite). Summaries (~200 chars) live in columns + vectors for fast search. Full text fetched on demand via WITH DOCUMENT.

**Tech Stack:** markitdown (Microsoft), pymupdf4llm, docling (IBM), SQLite (blobs), Ollama (vision models)

**Spec:** `docs/superpowers/specs/2026-03-21-document-layer-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `graphstore/document/__init__.py` | Create | Package init |
| `graphstore/document/store.py` | Create | DocumentStore: SQLite blob CRUD |
| `graphstore/ingest/__init__.py` | Create | Package init |
| `graphstore/ingest/base.py` | Create | Ingestor protocol, IngestResult, ExtractedImage |
| `graphstore/ingest/markitdown_ingestor.py` | Create | MarkItDown wrapper (Tier 1) |
| `graphstore/ingest/pymupdf4llm_ingestor.py` | Create | PyMuPDF4LLM wrapper (Tier 2) |
| `graphstore/ingest/docling_ingestor.py` | Create | Docling wrapper (Tier 3) |
| `graphstore/ingest/chunker.py` | Create | Text chunking (heading/paragraph/fixed) |
| `graphstore/ingest/vision.py` | Create | VisionHandler (Ollama client) |
| `graphstore/ingest/router.py` | Create | Tiered routing logic |
| `graphstore/ingest/connector.py` | Create | SYS CONNECT: auto-wire cross-doc relationships |
| `graphstore/dsl/grammar.lark` | Modify | INGEST, DOCUMENT, WITH DOCUMENT, CONNECT |
| `graphstore/dsl/ast_nodes.py` | Modify | IngestStmt, document fields |
| `graphstore/dsl/transformer.py` | Modify | Transform new rules |
| `graphstore/dsl/executor_writes.py` | Modify | _ingest, DOCUMENT on create |
| `graphstore/dsl/executor_reads.py` | Modify | WITH DOCUMENT on NODE |
| `graphstore/dsl/executor_system.py` | Modify | SYS INGESTORS |
| `graphstore/graphstore.py` | Modify | vision_model param, DocumentStore init |
| `graphstore/persistence/database.py` | Modify | Add documents table creation |
| `graphstore/cli.py` | Modify | install-vision, list-vision commands |
| `pyproject.toml` | Modify | Add markitdown, pymupdf4llm, docling deps |
| `tests/test_document_store.py` | Create | DocumentStore unit tests |
| `tests/test_ingest.py` | Create | Ingestor + chunker + integration tests |

---

## Task 1: DocumentStore (SQLite blob CRUD)

**Files:**
- Create: `graphstore/document/__init__.py`
- Create: `graphstore/document/store.py`
- Modify: `graphstore/persistence/database.py`
- Test: `tests/test_document_store.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_document_store.py
import sqlite3
import pytest
from graphstore.document.store import DocumentStore


@pytest.fixture
def doc_store():
    conn = sqlite3.connect(":memory:")
    ds = DocumentStore(conn)
    return ds


class TestDocumentStore:
    def test_put_and_get(self, doc_store):
        doc_store.put(0, b"Hello world", "text/plain")
        content, ctype = doc_store.get(0)
        assert content == b"Hello world"
        assert ctype == "text/plain"

    def test_get_missing_returns_none(self, doc_store):
        assert doc_store.get(999) is None

    def test_delete(self, doc_store):
        doc_store.put(0, b"data", "text/plain")
        doc_store.delete(0)
        assert doc_store.get(0) is None

    def test_has(self, doc_store):
        assert not doc_store.has(0)
        doc_store.put(0, b"data", "text/plain")
        assert doc_store.has(0)

    def test_size(self, doc_store):
        doc_store.put(0, b"12345", "text/plain")
        assert doc_store.size(0) == 5

    def test_put_replaces(self, doc_store):
        doc_store.put(0, b"old", "text/plain")
        doc_store.put(0, b"new", "text/markdown")
        content, ctype = doc_store.get(0)
        assert content == b"new"
        assert ctype == "text/markdown"

    def test_store_binary(self, doc_store):
        blob = bytes(range(256))
        doc_store.put(0, blob, "image/png")
        content, ctype = doc_store.get(0)
        assert content == blob
        assert ctype == "image/png"

    def test_store_large_text(self, doc_store):
        text = ("x" * 10000).encode()
        doc_store.put(0, text, "text/markdown")
        content, _ = doc_store.get(0)
        assert len(content) == 10000
```

- [ ] **Step 2: Implement DocumentStore**

```python
# graphstore/document/store.py
"""DocumentStore: SQLite-backed blob storage for raw documents."""
import sqlite3


class DocumentStore:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._ensure_table()

    def _ensure_table(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                slot INTEGER PRIMARY KEY,
                content BLOB NOT NULL,
                content_type TEXT NOT NULL,
                size INTEGER NOT NULL
            )
        """)

    def put(self, slot: int, content: bytes, content_type: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO documents (slot, content, content_type, size) VALUES (?, ?, ?, ?)",
            (slot, content, content_type, len(content)),
        )

    def get(self, slot: int) -> tuple[bytes, str] | None:
        row = self._conn.execute(
            "SELECT content, content_type FROM documents WHERE slot = ?", (slot,)
        ).fetchone()
        return (row[0], row[1]) if row else None

    def delete(self, slot: int) -> None:
        self._conn.execute("DELETE FROM documents WHERE slot = ?", (slot,))

    def has(self, slot: int) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM documents WHERE slot = ?", (slot,)
        ).fetchone()
        return row is not None

    def size(self, slot: int) -> int | None:
        row = self._conn.execute(
            "SELECT size FROM documents WHERE slot = ?", (slot,)
        ).fetchone()
        return row[0] if row else None
```

- [ ] **Step 3: Update database.py to create documents table**

In `graphstore/persistence/database.py`, add to the table creation:
```python
conn.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        slot INTEGER PRIMARY KEY,
        content BLOB NOT NULL,
        content_type TEXT NOT NULL,
        size INTEGER NOT NULL
    )
""")
```

- [ ] **Step 4: Run tests, commit**

```bash
uv run pytest tests/test_document_store.py -v
uv run pytest tests/ --ignore=tests/test_server.py -q
git add graphstore/document/ graphstore/persistence/database.py tests/test_document_store.py
git commit -m "feat: DocumentStore - SQLite blob storage for raw documents"
```

---

## Task 2: Ingestor Protocol + Chunker

**Files:**
- Create: `graphstore/ingest/__init__.py`
- Create: `graphstore/ingest/base.py`
- Create: `graphstore/ingest/chunker.py`
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Write chunker tests**

```python
# tests/test_ingest.py
from graphstore.ingest.chunker import chunk_by_heading, chunk_by_paragraph, chunk_fixed


class TestChunkByHeading:
    def test_splits_on_headings(self):
        text = "# Intro\nHello world\n## Details\nMore stuff\n## End\nBye"
        chunks = chunk_by_heading(text)
        assert len(chunks) == 3
        assert "Hello world" in chunks[0].text
        assert "More stuff" in chunks[1].text

    def test_no_headings_returns_one_chunk(self):
        text = "Just plain text without any headings."
        chunks = chunk_by_heading(text)
        assert len(chunks) == 1

    def test_respects_max_size(self):
        text = "# Heading\n" + "word " * 1000
        chunks = chunk_by_heading(text, max_chunk_size=100)
        assert all(len(c.text) <= 200 for c in chunks)  # some tolerance

    def test_chunk_has_heading(self):
        text = "# Overview\nContent here"
        chunks = chunk_by_heading(text)
        assert chunks[0].heading == "Overview"


class TestChunkFixed:
    def test_fixed_size(self):
        text = "a" * 500
        chunks = chunk_fixed(text, chunk_size=100, overlap=20)
        assert len(chunks) >= 5
        assert all(len(c.text) <= 120 for c in chunks)

    def test_overlap(self):
        text = "abcdefghij" * 50
        chunks = chunk_fixed(text, chunk_size=100, overlap=20)
        if len(chunks) > 1:
            end_of_first = chunks[0].text[-20:]
            start_of_second = chunks[1].text[:20]
            assert end_of_first == start_of_second
```

- [ ] **Step 2: Implement base.py and chunker.py**

```python
# graphstore/ingest/base.py
"""Ingestor protocol and data types."""
from dataclasses import dataclass, field


@dataclass
class ExtractedImage:
    data: bytes
    mime_type: str
    page: int | None = None
    caption: str | None = None
    description: str | None = None


@dataclass
class IngestResult:
    markdown: str
    images: list[ExtractedImage] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    parser_used: str = ""
    confidence: float = 1.0


class Ingestor:
    """Base interface for document ingestors."""
    name: str = "base"
    supported_extensions: list[str] = []

    def convert(self, file_path: str, **kwargs) -> IngestResult:
        raise NotImplementedError
```

```python
# graphstore/ingest/chunker.py
"""Text chunking strategies for document ingestion."""
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    index: int
    heading: str | None = None
    start_char: int = 0


def chunk_by_heading(text: str, max_chunk_size: int = 2000) -> list[Chunk]:
    """Split on markdown headings. Falls back to paragraph if no headings."""
    heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    matches = list(heading_pattern.finditer(text))

    if not matches:
        return chunk_by_paragraph(text, max_chunk_size)

    chunks = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        heading = match.group(2).strip()

        if len(section) > max_chunk_size:
            sub_chunks = chunk_fixed(section, max_chunk_size, overlap=50)
            for j, sc in enumerate(sub_chunks):
                sc.heading = heading
                sc.index = len(chunks)
                sc.start_char = start + sc.start_char
                chunks.append(sc)
        else:
            chunks.append(Chunk(text=section, index=len(chunks), heading=heading, start_char=start))

    # Text before first heading
    if matches and matches[0].start() > 0:
        preamble = text[:matches[0].start()].strip()
        if preamble:
            chunks.insert(0, Chunk(text=preamble, index=0, heading=None, start_char=0))
            for i, c in enumerate(chunks):
                c.index = i

    return chunks


def chunk_by_paragraph(text: str, max_chunk_size: int = 1000) -> list[Chunk]:
    """Split on double newlines."""
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current = ""
    start = 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 > max_chunk_size and current:
            chunks.append(Chunk(text=current.strip(), index=len(chunks), start_char=start))
            start += len(current)
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), index=len(chunks), start_char=start))
    return chunks if chunks else [Chunk(text=text.strip(), index=0, start_char=0)]


def chunk_fixed(text: str, chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Fixed-size chunks with overlap."""
    chunks = []
    pos = 0
    while pos < len(text):
        end = min(pos + chunk_size, len(text))
        chunks.append(Chunk(text=text[pos:end], index=len(chunks), start_char=pos))
        pos += chunk_size - overlap
        if pos >= len(text):
            break
    return chunks
```

- [ ] **Step 3: Run tests, commit**

```bash
uv run pytest tests/test_ingest.py -v
git add graphstore/ingest/ tests/test_ingest.py
git commit -m "feat: Ingestor protocol + chunker (heading/paragraph/fixed)"
```

---

## Task 3: MarkItDown Ingestor (Tier 1)

**Files:**
- Create: `graphstore/ingest/markitdown_ingestor.py`
- Modify: `pyproject.toml`
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Add markitdown to pyproject.toml**

- [ ] **Step 2: Write tests**

```python
class TestMarkItDownIngestor:
    def test_convert_text_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello world\nThis is a test")
        from graphstore.ingest.markitdown_ingestor import MarkItDownIngestor
        ingestor = MarkItDownIngestor()
        result = ingestor.convert(str(f))
        assert "Hello world" in result.markdown
        assert result.parser_used == "markitdown"

    def test_supported_extensions(self):
        from graphstore.ingest.markitdown_ingestor import MarkItDownIngestor
        i = MarkItDownIngestor()
        assert "txt" in i.supported_extensions
        assert "html" in i.supported_extensions
```

- [ ] **Step 3: Implement**

```python
# graphstore/ingest/markitdown_ingestor.py
from graphstore.ingest.base import Ingestor, IngestResult

class MarkItDownIngestor(Ingestor):
    name = "markitdown"
    supported_extensions = ["txt", "html", "htm", "csv", "json", "xml",
                           "pdf", "docx", "pptx", "xlsx", "zip", "md"]

    def __init__(self, llm_client=None, llm_model=None):
        from markitdown import MarkItDown
        kwargs = {}
        if llm_client:
            kwargs["llm_client"] = llm_client
            kwargs["llm_model"] = llm_model
            kwargs["enable_plugins"] = True
        self._md = MarkItDown(**kwargs)

    def convert(self, file_path: str, **kwargs) -> IngestResult:
        result = self._md.convert(file_path)
        return IngestResult(
            markdown=result.text_content,
            metadata={"source": file_path},
            parser_used=self.name,
        )
```

- [ ] **Step 4: Run tests, commit**

```bash
git commit -m "feat: MarkItDown ingestor (Tier 1 - general files)"
```

---

## Task 4: PyMuPDF4LLM Ingestor (Tier 2) + Docling Ingestor (Tier 3)

**Files:**
- Create: `graphstore/ingest/pymupdf4llm_ingestor.py`
- Create: `graphstore/ingest/docling_ingestor.py`
- Modify: `pyproject.toml`
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Implement PyMuPDF4LLM ingestor**

```python
# graphstore/ingest/pymupdf4llm_ingestor.py
from graphstore.ingest.base import Ingestor, IngestResult, ExtractedImage

class PyMuPDF4LLMIngestor(Ingestor):
    name = "pymupdf4llm"
    supported_extensions = ["pdf"]

    def convert(self, file_path: str, **kwargs) -> IngestResult:
        import pymupdf4llm
        import pymupdf
        md_text = pymupdf4llm.to_markdown(file_path)
        # Extract images
        images = []
        doc = pymupdf.open(file_path)
        metadata = {"pages": len(doc), "source": file_path}
        if doc.metadata:
            metadata.update({k: v for k, v in doc.metadata.items() if v})
        for page_num, page in enumerate(doc):
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                if base_image:
                    images.append(ExtractedImage(
                        data=base_image["image"],
                        mime_type=f"image/{base_image['ext']}",
                        page=page_num,
                    ))
        doc.close()
        return IngestResult(markdown=md_text, images=images, metadata=metadata, parser_used=self.name)
```

- [ ] **Step 2: Implement Docling ingestor**

```python
# graphstore/ingest/docling_ingestor.py
from graphstore.ingest.base import Ingestor, IngestResult

class DoclingIngestor(Ingestor):
    name = "docling"
    supported_extensions = ["pdf", "docx"]

    def convert(self, file_path: str, **kwargs) -> IngestResult:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(file_path)
        md_text = result.document.export_to_markdown()
        metadata = {"source": file_path, "parser_used": self.name}
        return IngestResult(markdown=md_text, metadata=metadata, parser_used=self.name)
```

- [ ] **Step 3: Add deps to pyproject.toml, run tests, commit**

```bash
git commit -m "feat: PyMuPDF4LLM (Tier 2) + Docling (Tier 3) ingestors"
```

---

## Task 5: Ingest Router (tiered selection)

**Files:**
- Create: `graphstore/ingest/router.py`
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Write tests**

```python
class TestRouter:
    def test_routes_txt_to_markitdown(self):
        from graphstore.ingest.router import select_ingestor
        name = select_ingestor("notes.txt")
        assert name == "markitdown"

    def test_routes_html_to_markitdown(self):
        from graphstore.ingest.router import select_ingestor
        assert select_ingestor("page.html") == "markitdown"

    def test_routes_pdf_to_pymupdf4llm(self):
        from graphstore.ingest.router import select_ingestor
        assert select_ingestor("report.pdf") == "pymupdf4llm"

    def test_routes_docx_to_markitdown(self):
        from graphstore.ingest.router import select_ingestor
        assert select_ingestor("doc.docx") == "markitdown"

    def test_explicit_override(self):
        from graphstore.ingest.router import select_ingestor
        assert select_ingestor("report.pdf", using="docling") == "docling"
```

- [ ] **Step 2: Implement router**

```python
# graphstore/ingest/router.py
"""Tiered ingestor routing: deterministic parsers first, VLM as fallback."""
from pathlib import Path

# Tier 1: MarkItDown handles general files
# Tier 2: PyMuPDF4LLM handles normal PDFs (better structure)
# Tier 3: Docling handles hard PDFs (tables, OCR, formulas)
# Tier 4: VLM fallback for image-heavy/scanned pages

EXTENSION_MAP = {
    # Tier 1: MarkItDown
    "txt": "markitdown", "md": "markitdown", "html": "markitdown", "htm": "markitdown",
    "csv": "markitdown", "json": "markitdown", "xml": "markitdown",
    "docx": "markitdown", "pptx": "markitdown", "xlsx": "markitdown",
    "zip": "markitdown",
    # Tier 2: PyMuPDF4LLM for PDFs
    "pdf": "pymupdf4llm",
}

INGESTORS = {}

def _get_ingestor(name: str):
    if name not in INGESTORS:
        if name == "markitdown":
            from graphstore.ingest.markitdown_ingestor import MarkItDownIngestor
            INGESTORS[name] = MarkItDownIngestor()
        elif name == "pymupdf4llm":
            from graphstore.ingest.pymupdf4llm_ingestor import PyMuPDF4LLMIngestor
            INGESTORS[name] = PyMuPDF4LLMIngestor()
        elif name == "docling":
            from graphstore.ingest.docling_ingestor import DoclingIngestor
            INGESTORS[name] = DoclingIngestor()
    return INGESTORS[name]

def select_ingestor(file_path: str, using: str | None = None) -> str:
    if using:
        return using
    ext = Path(file_path).suffix.lstrip(".").lower()
    return EXTENSION_MAP.get(ext, "markitdown")

def ingest_file(file_path: str, using: str | None = None, **kwargs):
    name = select_ingestor(file_path, using)
    ingestor = _get_ingestor(name)
    return ingestor.convert(file_path, **kwargs)
```

- [ ] **Step 3: Run tests, commit**

```bash
git commit -m "feat: tiered ingest router - deterministic parsers first"
```

---

## Task 6: VisionHandler (Ollama integration)

**Files:**
- Create: `graphstore/ingest/vision.py`
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Implement VisionHandler**

```python
# graphstore/ingest/vision.py
"""Vision model handler via Ollama for image understanding."""


class VisionHandler:
    """Connects to Ollama for image description. Used as Tier 4 fallback."""

    def __init__(self, model: str = "smolvlm2:2.2b", base_url: str = "http://localhost:11434/v1"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("VisionHandler requires openai package. pip install openai")
        self._client = OpenAI(base_url=base_url, api_key="ollama")
        self._model = model

    @property
    def client(self):
        return self._client

    @property
    def model(self):
        return self._model

    def describe(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        """Describe an image using the vision model."""
        import base64
        b64 = base64.b64encode(image_bytes).decode()
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail. Focus on data, text, and key visual elements."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                ],
            }],
            max_tokens=500,
        )
        return response.choices[0].message.content

    def is_available(self) -> bool:
        """Check if Ollama is running and model is loaded."""
        try:
            self._client.models.list()
            return True
        except Exception:
            return False
```

- [ ] **Step 2: Update MarkItDown ingestor to accept vision handler**

Update `MarkItDownIngestor.__init__` to accept `vision_handler: VisionHandler | None` and pass its client/model to MarkItDown's `llm_client`/`llm_model` params.

- [ ] **Step 3: Run tests, commit**

```bash
git commit -m "feat: VisionHandler for image understanding via Ollama"
```

---

## Task 7: INGEST DSL command + GraphStore integration

**Files:**
- Modify: `graphstore/dsl/grammar.lark`
- Modify: `graphstore/dsl/ast_nodes.py`
- Modify: `graphstore/dsl/transformer.py`
- Modify: `graphstore/dsl/executor_writes.py`
- Modify: `graphstore/graphstore.py`
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Add grammar rules**

```lark
# Add to write_query alternatives
| ingest_stmt

ingest_stmt: "INGEST" STRING ingest_as? ingest_kind? using_clause? vision_clause?
ingest_as: "AS" STRING
ingest_kind: "KIND" STRING
using_clause: "USING" IDENTIFIER
vision_clause: "USING" "VISION" STRING
```

- [ ] **Step 2: Add AST node**

```python
@dataclass
class IngestStmt:
    file_path: str
    node_id: str | None = None      # AS clause
    kind: str | None = None         # KIND clause
    using: str | None = None        # USING ingestor
    vision_model: str | None = None # USING VISION
```

- [ ] **Step 3: Add DOCUMENT clause to CREATE and WITH DOCUMENT to NODE**

Grammar:
```lark
create_node: "CREATE" "NODE" STRING field_pairs vector_clause? document_clause? expires_clause?
document_clause: "DOCUMENT" STRING

node_q: "NODE" STRING with_document?
with_document: "WITH" "DOCUMENT"
```

Add `document: str | None` to CreateNode AST.
Add `with_document: bool = False` to NodeQuery AST.

- [ ] **Step 4: Implement _ingest in executor_writes.py**

```python
def _ingest(self, q: IngestStmt) -> Result:
    from graphstore.ingest.router import ingest_file
    # 1. Convert file to markdown
    result = ingest_file(q.file_path, using=q.using)
    # 2. Chunk
    from graphstore.ingest.chunker import chunk_by_heading
    chunks = chunk_by_heading(result.markdown)
    # 3. Create parent node
    parent_id = q.node_id or f"doc:{hash(q.file_path) & 0xFFFFFFFF:08x}"
    kind = q.kind or "document"
    parent_fields = {"source": q.file_path, **result.metadata}
    # ... create parent node with fields
    # 4. Create chunk nodes with DOCUMENT + auto-embed
    # 5. Create image nodes if any
    # 6. Create edges parent -> chunks, chunks -> images
    # 7. Return summary
```

- [ ] **Step 5: Implement WITH DOCUMENT in executor_reads.py**

In `_node`, if `q.with_document` and DocumentStore has the slot:
```python
if q.with_document and self._document_store:
    doc = self._document_store.get(slot)
    if doc:
        node["_document"] = doc[0].decode("utf-8") if doc[1].startswith("text") else doc[0]
```

- [ ] **Step 6: Wire DocumentStore into GraphStore**

In `graphstore.py` `__init__`, after opening SQLite connection:
```python
from graphstore.document.store import DocumentStore
self._document_store = DocumentStore(self._conn) if self._conn else None
```

Add `vision_model` and `vision_base_url` params. If set, create VisionHandler and pass to MarkItDown ingestor.

- [ ] **Step 7: Write integration tests**

```python
class TestIngestDSL:
    def test_ingest_text_file(self, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("# Meeting Notes\n\nDiscussed Q3 results.\n\n## Action Items\n\nFollow up with Alice.")
        g = GraphStore(path=str(tmp_path / "db"), embedder=None)
        result = g.execute(f'INGEST "{f}"')
        assert result.data["chunks"] >= 2
        # Parent node exists
        nodes = g.execute('NODES WHERE kind = "document"')
        assert len(nodes.data) >= 1

    def test_node_with_document(self, tmp_path):
        g = GraphStore(path=str(tmp_path / "db"), embedder=None)
        g.execute('CREATE NODE "n1" kind = "note" name = "test" DOCUMENT "Full text here"')
        # Without DOCUMENT
        node = g.execute('NODE "n1"')
        assert "_document" not in node.data
        # With DOCUMENT
        node = g.execute('NODE "n1" WITH DOCUMENT')
        assert node.data["_document"] == "Full text here"

    def test_ingest_creates_edges(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("# Part 1\nContent\n\n# Part 2\nMore content")
        g = GraphStore(path=str(tmp_path / "db"), embedder=None)
        g.execute(f'INGEST "{f}" AS "doc:test" KIND "document"')
        edges = g.execute('EDGES FROM "doc:test"')
        assert len(edges.data) >= 2  # edges to chunks
```

- [ ] **Step 8: Run full suite, commit**

```bash
uv run pytest tests/ --ignore=tests/test_server.py -q
git add graphstore/ tests/ pyproject.toml
git commit -m "feat: INGEST command, DOCUMENT clause, WITH DOCUMENT, full pipeline"
```

---

## Task 8: SYS INGESTORS + CLI vision commands

**Files:**
- Modify: `graphstore/dsl/executor_system.py`
- Modify: `graphstore/cli.py`
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Add SYS INGESTORS**

Grammar: `sys_ingestors: "INGESTORS"`
Returns list of available ingestors with supported formats.

- [ ] **Step 2: Add CLI commands**

```bash
graphstore install-vision smolvlm2       # ollama pull smolvlm2:2.2b
graphstore install-vision qwen3-vl       # ollama pull qwen3-vl:8b-q4
graphstore list-vision                   # list available vision models
```

- [ ] **Step 3: Run tests, commit**

```bash
git commit -m "feat: SYS INGESTORS, CLI install-vision/list-vision"
```

---

## Task 9: Final integration + tests

**Files:**
- All test files

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest tests/ --ignore=tests/test_server.py -v --tb=short
```

- [ ] **Step 2: Test the full agent flow**

```python
g = GraphStore(path="./test_brain")
g.execute('INGEST "test.txt" AS "doc:test" KIND "document"')
result = g.execute('SIMILAR TO "meeting notes" LIMIT 5')
doc = g.execute('NODE "chunk:0" WITH DOCUMENT')
edges = g.execute('EDGES FROM "doc:test"')
g.execute('SYS INGESTORS')
```

- [ ] **Step 3: Commit**

```bash
git commit -m "feat: document layer complete - INGEST, DocumentStore, tiered routing, vision"
```
