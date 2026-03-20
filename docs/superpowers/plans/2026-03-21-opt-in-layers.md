# Opt-in Layers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add document ingestion (3 parsers + chunker + SQLite storage + cross-doc wiring), voice (Moonshine STT + Piper TTS), embedder safety (dirty flag + SYS REEMBED), and unified SYS STATUS.

**Architecture:** DocumentStore is a separate SQLite file (documents.db) for cold storage. Summaries (~200 chars) are embedded and stored in-memory for fast search. Full text fetched on demand. Voice is Python API (not DSL). All opt-ins managed via graphstore CLI.

**Tech Stack:** markitdown, pymupdf4llm, docling (lazy), moonshine, piper-tts, SQLite, Ollama

**Spec:** `docs/superpowers/specs/2026-03-21-opt-in-layers-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `graphstore/document/__init__.py` | Create | Package init |
| `graphstore/document/store.py` | Create | DocumentStore: SQLite multi-table CRUD |
| `graphstore/ingest/__init__.py` | Create | Package init |
| `graphstore/ingest/base.py` | Create | Ingestor protocol, IngestResult, Chunk, ExtractedImage |
| `graphstore/ingest/markitdown_ingestor.py` | Create | Tier 1: general files |
| `graphstore/ingest/pymupdf4llm_ingestor.py` | Create | Tier 2: PDF structure + images |
| `graphstore/ingest/docling_ingestor.py` | Create | Tier 3: hard PDFs (lazy import) |
| `graphstore/ingest/chunker.py` | Create | Heading/paragraph/fixed + summary |
| `graphstore/ingest/vision.py` | Create | VisionHandler (Ollama) |
| `graphstore/ingest/router.py` | Create | Tiered routing |
| `graphstore/ingest/connector.py` | Create | SYS CONNECT cross-doc wiring |
| `graphstore/voice/__init__.py` | Create | Package init |
| `graphstore/voice/stt.py` | Create | Moonshine STT wrapper |
| `graphstore/voice/tts.py` | Create | Piper TTS wrapper |
| `graphstore/dsl/grammar.lark` | Modify | INGEST, DOCUMENT, WITH DOCUMENT, CONNECT |
| `graphstore/dsl/ast_nodes.py` | Modify | New AST nodes |
| `graphstore/dsl/transformer.py` | Modify | New transformer methods |
| `graphstore/dsl/executor_writes.py` | Modify | _ingest, DOCUMENT on create, RETRACT cascade |
| `graphstore/dsl/executor_reads.py` | Modify | WITH DOCUMENT |
| `graphstore/dsl/executor_system.py` | Modify | SYS CONNECT, INGESTORS, STATUS, REEMBED |
| `graphstore/graphstore.py` | Modify | DocumentStore init, voice, embedder dirty flag |
| `graphstore/cli.py` | Modify | install-vision, install-voice, list commands |
| `pyproject.toml` | Modify | markitdown, pymupdf4llm, docling deps |
| `tests/test_document_store.py` | Create | DocumentStore tests |
| `tests/test_ingest.py` | Create | Ingestor + chunker + pipeline tests |
| `tests/test_voice.py` | Create | Voice tests (conditional) |
| `tests/test_sys_status.py` | Create | Unified status + embedder safety tests |

---

## Task 1: DocumentStore (SQLite multi-table)

**Files:**
- Create: `graphstore/document/__init__.py`, `graphstore/document/store.py`
- Test: `tests/test_document_store.py`

- [ ] **Step 1: Write tests for all 4 tables (documents, summaries, doc_metadata, images)**

Tests: put/get/delete/has for documents, put/get summaries, put/get metadata, put/get images, delete_all_for_doc (cascade), orphan_cleanup, stats, temp file mode.

- [ ] **Step 2: Implement DocumentStore**

Multi-table SQLite with temp file fallback when no path provided. See spec section 2.3 for full API.

- [ ] **Step 3: Run tests, commit**

```bash
git commit -m "feat: DocumentStore - SQLite multi-table blob storage (documents, summaries, metadata, images)"
```

---

## Task 2: Chunker + Ingestor Protocol

**Files:**
- Create: `graphstore/ingest/__init__.py`, `graphstore/ingest/base.py`, `graphstore/ingest/chunker.py`
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Write chunker tests**

Tests: chunk_by_heading splits on `#` headings, no headings falls back to paragraph, respects max size, chunk has heading/summary, chunk_fixed with overlap, summary is first 200 chars.

- [ ] **Step 2: Implement base.py (Ingestor protocol, IngestResult, Chunk, ExtractedImage)**

- [ ] **Step 3: Implement chunker.py (heading, paragraph, fixed strategies + summary generation)**

Summary = first 200 characters of chunk text.

- [ ] **Step 4: Run tests, commit**

```bash
git commit -m "feat: Ingestor protocol + chunker (heading/paragraph/fixed + summaries)"
```

---

## Task 3: MarkItDown + PyMuPDF4LLM + Docling Ingestors

**Files:**
- Create: `graphstore/ingest/markitdown_ingestor.py`, `graphstore/ingest/pymupdf4llm_ingestor.py`, `graphstore/ingest/docling_ingestor.py`
- Modify: `pyproject.toml`
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Add deps to pyproject.toml**

```toml
markitdown = ">=0.1"
pymupdf4llm = ">=0.1"
docling = ">=2.0"
```

- [ ] **Step 2: Write tests for each ingestor**

Tests: MarkItDown converts .txt file, PyMuPDF4LLM converts .pdf (create a small test PDF or skip if no test PDF), Docling lazy import (import only triggers when used), supported_extensions correct.

- [ ] **Step 3: Implement all three ingestors**

MarkItDown: wraps `markitdown.MarkItDown`, accepts optional llm_client for vision.
PyMuPDF4LLM: wraps `pymupdf4llm.to_markdown`, extracts images via `pymupdf`.
Docling: wraps `docling.DocumentConverter`, lazy import only.

Scanned PDF detection in PyMuPDF4LLM: if < 50 chars/page, set confidence < 0.5.

- [ ] **Step 4: Run tests, commit**

```bash
git commit -m "feat: MarkItDown (Tier 1) + PyMuPDF4LLM (Tier 2) + Docling (Tier 3) ingestors"
```

---

## Task 4: Router + VisionHandler

**Files:**
- Create: `graphstore/ingest/router.py`, `graphstore/ingest/vision.py`
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Write router tests**

Tests: .txt routes to markitdown, .pdf routes to pymupdf4llm, .docx routes to markitdown, explicit USING override, .png routes to markitdown (no vision) or vision (if available).

- [ ] **Step 2: Implement router with extension map + ingest_file()**

- [ ] **Step 3: Implement VisionHandler (Ollama client)**

Uses OpenAI-compatible client. `is_available()` checks Ollama health. `describe()` sends image for description. Graceful degradation if Ollama not running.

- [ ] **Step 4: Run tests, commit**

```bash
git commit -m "feat: tiered ingest router + VisionHandler (Ollama)"
```

---

## Task 5: INGEST DSL + GraphStore Integration

**Files:**
- Modify: `graphstore/dsl/grammar.lark`, `graphstore/dsl/ast_nodes.py`, `graphstore/dsl/transformer.py`
- Modify: `graphstore/dsl/executor_writes.py`
- Modify: `graphstore/graphstore.py`
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Add grammar**

```lark
ingest_stmt: "INGEST" STRING ingest_as? ingest_kind? using_clause? vision_clause?
ingest_as: "AS" STRING
ingest_kind: "KIND" STRING
using_clause: "USING" IDENTIFIER
vision_clause: "USING" "VISION" STRING

document_clause: "DOCUMENT" STRING
node_q: "NODE" STRING with_doc?
with_doc: "WITH" "DOCUMENT"
```

- [ ] **Step 2: Add AST nodes (IngestStmt, document field on CreateNode, with_document on NodeQuery)**

- [ ] **Step 3: Implement _ingest in executor_writes.py**

Atomic: parse → SQLite → graph. Rollback SQLite on graph failure.

- [ ] **Step 4: Implement WITH DOCUMENT in executor_reads.py**

- [ ] **Step 5: Wire DocumentStore into GraphStore constructor**

Separate documents.db file. Pass to executors.

- [ ] **Step 6: Write integration tests**

Tests: INGEST text file creates parent + chunks + edges, NODE WITH DOCUMENT returns full text, INGEST with AS id, duplicate INGEST raises NodeExists, CREATE with DOCUMENT clause.

- [ ] **Step 7: Run full suite, commit**

```bash
git commit -m "feat: INGEST command, DOCUMENT clause, WITH DOCUMENT - full pipeline"
```

---

## Task 6: SYS CONNECT (cross-document wiring)

**Files:**
- Create: `graphstore/ingest/connector.py`
- Modify: `graphstore/dsl/grammar.lark`, `graphstore/dsl/ast_nodes.py`, `graphstore/dsl/transformer.py`
- Modify: `graphstore/dsl/executor_system.py`
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Add grammar**

```lark
sys_connect: "CONNECT" where_clause? threshold_clause?
connect_node: "CONNECT" "NODE" STRING threshold_clause?
```

- [ ] **Step 2: Implement connector.py**

For each chunk with vector, find top-3 similar from OTHER docs. If similarity > threshold, create edge kind="similar_to". Skip existing pairs. Respects live_mask.

- [ ] **Step 3: Write tests**

Tests: CONNECT creates cross-doc edges, idempotent (second run adds 0 edges), respects threshold, CONNECT NODE wires one node.

- [ ] **Step 4: Run tests, commit**

```bash
git commit -m "feat: SYS CONNECT + CONNECT NODE - auto-wire cross-document relationships"
```

---

## Task 7: Embedder Safety (dirty flag + SYS REEMBED)

**Files:**
- Modify: `graphstore/graphstore.py`
- Modify: `graphstore/dsl/executor_system.py`
- Modify: `graphstore/dsl/executor_reads.py`
- Test: `tests/test_sys_status.py`

- [ ] **Step 1: Add _embedder_dirty flag to GraphStore**

Set True by SYS SET EMBEDDER. SIMILAR TO raises error while dirty. Cleared by SYS REEMBED.

- [ ] **Step 2: Implement SYS REEMBED**

Reads summaries from DocumentStore SQLite, re-embeds with current embedder, replaces vectors.

- [ ] **Step 3: Block SIMILAR TO when dirty**

In executor_reads._similar: if `self._embedder_dirty`, raise "Embedder changed. Run SYS REEMBED."

- [ ] **Step 4: Write tests**

Tests: SYS SET EMBEDDER sets dirty, SIMILAR TO blocked while dirty, SYS REEMBED clears dirty and updates vectors.

- [ ] **Step 5: Run tests, commit**

```bash
git commit -m "feat: embedder dirty flag + SYS REEMBED safety"
```

---

## Task 8: RETRACT Cascade + DELETE Cascade

**Files:**
- Modify: `graphstore/dsl/executor_writes.py`
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Implement RETRACT cascade**

When RETRACT targets a node with kind="document", also retract all outgoing chunk + image nodes.

- [ ] **Step 2: Implement DELETE cascade for documents**

When DELETE targets kind="document", cascade-delete chunks + images + their DocumentStore entries.

- [ ] **Step 3: Implement orphan cleanup for rollback**

After SYS ROLLBACK, call `document_store.orphan_cleanup(live_slots)`.

- [ ] **Step 4: Write tests**

Tests: RETRACT doc retracts chunks, DELETE doc deletes chunks + SQLite entries, rollback cleans orphans.

- [ ] **Step 5: Run tests, commit**

```bash
git commit -m "feat: RETRACT/DELETE cascade for documents, orphan cleanup on rollback"
```

---

## Task 9: Unified SYS STATUS + SYS INGESTORS + SYS STATS DOCUMENTS

**Files:**
- Modify: `graphstore/dsl/grammar.lark`, `graphstore/dsl/ast_nodes.py`, `graphstore/dsl/transformer.py`
- Modify: `graphstore/dsl/executor_system.py`
- Test: `tests/test_sys_status.py`

- [ ] **Step 1: Implement SYS STATUS**

Returns: nodes, edges, memory, vectors, documents, embedder info, vision status, voice status, ollama health.

- [ ] **Step 2: Implement SYS INGESTORS**

Returns list of available ingestors with formats and status.

- [ ] **Step 3: Implement SYS STATS DOCUMENTS**

Returns document_count, total_bytes, chunk_count, image_count from DocumentStore.

- [ ] **Step 4: Write tests, commit**

```bash
git commit -m "feat: unified SYS STATUS, SYS INGESTORS, SYS STATS DOCUMENTS"
```

---

## Task 10: Voice Layer (Moonshine STT + Piper TTS)

**Files:**
- Create: `graphstore/voice/__init__.py`, `graphstore/voice/stt.py`, `graphstore/voice/tts.py`
- Modify: `graphstore/graphstore.py`
- Modify: `graphstore/cli.py`
- Test: `tests/test_voice.py`

- [ ] **Step 1: Implement stt.py (Moonshine wrapper)**

Lazy import. `start_listening(on_text)`, `stop_listening()`, `transcribe_file(path) -> str`.

- [ ] **Step 2: Implement tts.py (Piper wrapper)**

Lazy import. `speak(text)`, `speak_to_file(text, path)`.

- [ ] **Step 3: Add voice=True to GraphStore constructor**

```python
if voice:
    from graphstore.voice.stt import MoonshineSTT
    from graphstore.voice.tts import PiperTTS
    self._stt = MoonshineSTT()
    self._tts = PiperTTS()
```

Add `g.speak()`, `g.listen()`, `g.stop_listening()` methods.

- [ ] **Step 4: Add audio INGEST support**

In router.py: .wav/.mp3/.ogg/.flac → moonshine transcription if voice installed, else error.

- [ ] **Step 5: Add CLI commands**

```bash
graphstore install-voice     # pip install piper-tts moonshine
graphstore list-voice        # show installed voice models
```

- [ ] **Step 6: Write conditional tests**

```python
pytest.importorskip("moonshine")
pytest.importorskip("piper")
```

- [ ] **Step 7: Run tests, commit**

```bash
git commit -m "feat: voice layer - Moonshine STT + Piper TTS + audio INGEST"
```

---

## Task 11: Final Integration + Full Test Suite

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest tests/ --ignore=tests/test_server.py -v --tb=short
```

- [ ] **Step 2: Test the full agent flow**

```python
g = GraphStore(path="./test_brain")
g.execute('INGEST "test.txt" AS "doc:test"')
g.execute('SYS CONNECT')
g.execute('SIMILAR TO "meeting notes" LIMIT 5')
g.execute('NODE "chunk:0" WITH DOCUMENT')
g.execute('SYS STATUS')
g.execute('SYS STATS DOCUMENTS')
g.execute('SYS INGESTORS')
```

- [ ] **Step 3: Final commit**

```bash
git commit -m "feat: opt-in layers complete - documents, vision, voice, embedder safety"
```
