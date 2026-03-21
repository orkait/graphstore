<div align="center">

# graphstore

**Agentic brain DB - the cognitive layer for AI agents**

[![CI](https://github.com/orkait/graphstore/actions/workflows/ci.yml/badge.svg)](https://github.com/orkait/graphstore/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-22c55e?logo=opensourceinitiative&logoColor=white)](https://github.com/orkait/graphstore/blob/main/LICENSE)
[![Version](https://img.shields.io/badge/version-0.3.0-f59e0b?logo=semver&logoColor=white)](https://github.com/orkait/graphstore)
[![NumPy](https://img.shields.io/badge/numpy-%23013243?logo=numpy&logoColor=white)](https://numpy.org)
[![SciPy](https://img.shields.io/badge/scipy-%238CAAE6?logo=scipy&logoColor=white)](https://scipy.org)
[![usearch](https://img.shields.io/badge/usearch-%23FF6B35?logo=data:image/svg+xml;base64,&logoColor=white)](https://github.com/unum-cloud/usearch)

</div>

---

Six engines, one DSL. Columnar numpy storage, sparse matrix traversal, HNSW vector search, document ingestion, markdown vault, and a human-readable query language - everything an AI agent needs to remember, recall, reason, and speak.

## 🧩 What agents get

| Need | graphstore solves it | Speed (100k nodes) |
|---|---|---|
| **Recall by meaning** | `SIMILAR TO "Paris travel" LIMIT 10` | 127 μs |
| **Recall by association** | `RECALL FROM "concept:paris" DEPTH 3` | 983 μs |
| **Memory summarization** | `AGGREGATE NODES GROUP BY topic SELECT COUNT(), AVG(importance)` | 788 μs |
| **Belief tracking** | `ASSERT "fact:x" CONFIDENCE 0.9` / `RETRACT "fact:x"` | 9 μs |
| **Contradiction detection** | `SYS CONTRADICTIONS WHERE kind = "belief" FIELD value GROUP BY topic` | 981 μs |
| **Hypothesis testing** | `SYS SNAPSHOT "before"` ... `SYS ROLLBACK TO "before"` | 1.8 ms |
| **Document ingestion** | `INGEST "report.pdf"` (auto-parse, chunk, embed, wire) | < 2 sec |
| **Cross-doc connections** | `SYS CONNECT` (auto-wire similar chunks across documents) | < 5 sec |
| **Working memory** | `CREATE NODE ... EXPIRES IN 30m` + `SYS EXPIRE` | 9 μs |
| **Temporal queries** | `NODES WHERE __created_at__ > NOW() - 7d` | 102 μs |
| **Isolated reasoning** | `BIND CONTEXT "session"` ... `DISCARD CONTEXT "session"` | 72 μs |
| **Point lookup** | `NODE "memory:42"` | 4 μs |

## 🏗️ Six Engines

```
┌──────────────────────────────────────────────────────────────────────┐
│                         DSL (one language)                            │
└──┬──────────┬──────────┬──────────┬──────────┬──────────┬────────────┘
   ▼          ▼          ▼          ▼          ▼          ▼
ColumnStore  Graph    VectorStore DocStore   Ingestor    Vault
 (numpy)  (scipy CSR) (usearch)  (SQLite)              (markdown)
structured relations  meaning    raw docs   PDF/Word   agent notes
WHERE      RECALL     SIMILAR    WITH DOC   → chunks   NEW/READ
GROUP BY   TRAVERSE   TO                    → embed    WRITE/SYNC
ORDER BY   PATH                             → images   DAILY/LIST
```

## 📦 Install

```bash
pip install graphstore
```

Core includes: numpy, scipy, lark, usearch, model2vec (~85 MB)

<details>
<summary><strong>Opt-in upgrades</strong></summary>

```bash
# Higher quality embeddings (EmbeddingGemma-300M via ONNX)
graphstore install-embedder embeddinggemma

# Image understanding in documents (SmolVLM2 / Qwen3-VL via Ollama)
graphstore install-vision smolvlm2
graphstore install-vision qwen3-vl

# Voice: speech-to-text + text-to-speech (Moonshine + Piper)
graphstore install-voice
```

</details>

## 🚀 Quick Start

```python
from graphstore import GraphStore

g = GraphStore(path="./brain")

# Store memories
g.execute('CREATE NODE "memory:paris" kind = "memory" topic = "travel" importance = 0.9')
g.execute('CREATE NODE "memory:eiffel" kind = "memory" topic = "travel" importance = 0.8')
g.execute('CREATE EDGE "memory:paris" -> "memory:eiffel" kind = "associated"')

# Recall by meaning (vector similarity)
result = g.execute('SIMILAR TO "European architecture" LIMIT 5')

# Recall by association (graph activation)
result = g.execute('RECALL FROM "memory:paris" DEPTH 2 LIMIT 10')

# Ingest a document (auto-parse, chunk, embed, wire)
g.execute('INGEST "report.pdf" AS "doc:q3" KIND "report"')
g.execute('SYS CONNECT')  # auto-wire similar chunks across documents

# Search document contents by meaning
result = g.execute('SIMILAR TO "Q3 revenue growth" LIMIT 5')
doc = g.execute('NODE "doc:q3:chunk:3" WITH DOCUMENT')  # fetch full text
```

## 🧠 Agent Memory Features

<details>
<summary><strong>Belief Management</strong> - assert, retract, detect contradictions</summary>

```sql
ASSERT "fact:earth-radius" value = 6371 kind = "fact" CONFIDENCE 0.99 SOURCE "physics-tool"
RETRACT "fact:old-preference" REASON "user corrected this"
UPDATE NODES WHERE kind = "fact" AND source = "unreliable" SET confidence = 0.1
SYS CONTRADICTIONS WHERE kind = "belief" FIELD value GROUP BY topic
```

</details>

<details>
<summary><strong>Semantic Search</strong> - find memories by meaning</summary>

```sql
SIMILAR TO "European travel" LIMIT 10
SIMILAR TO "revenue growth" LIMIT 10 WHERE kind = "chunk" AND confidence > 0.5
SIMILAR TO NODE "concept:paris" LIMIT 10
SIMILAR TO [0.12, -0.34, 0.56, ...] LIMIT 10
SYS DUPLICATES THRESHOLD 0.95
SYS CONNECT THRESHOLD 0.85
```

</details>

<details>
<summary><strong>Document Ingestion</strong> - PDF, Word, text, images, audio</summary>

```sql
-- Tiered parsing: MarkItDown → PyMuPDF4LLM → Docling → VLM
INGEST "report.pdf" AS "doc:q3" KIND "report"
INGEST "notes.docx" USING markitdown
INGEST "financials.pdf" USING docling
INGEST "slides.pptx" USING VISION "smolvlm2"

-- Fetch full document text (from disk, on demand)
NODE "doc:q3:chunk:7" WITH DOCUMENT

-- Auto-wire similar chunks across all documents
SYS CONNECT

-- System info
SYS INGESTORS
SYS STATS DOCUMENTS
```

</details>

<details>
<summary><strong>Vault</strong> - structured markdown notes for agent memory</summary>

```python
g = GraphStore(path="./brain", vault="./notes")

# Agent writes notes as structured markdown files
g.execute('VAULT NEW "Project Requirements" KIND "context" TAGS "project,specs"')
g.execute('VAULT WRITE "Project Requirements" SECTION "body" CONTENT "The app must support..."')

# Vault files are human-readable markdown, auto-indexed in graphstore
g.execute('VAULT SEARCH "deployment requirements" LIMIT 5')

# Load standing instructions at session start
instructions = g.execute('VAULT LIST WHERE note_kind = "instruction" AND status = "active"')

# Daily working notes
g.execute('VAULT DAILY')
g.execute('VAULT APPEND "2026-03-21" SECTION "body" CONTENT "Completed task X"')

# Facts auto-assert with confidence
# (note with kind: fact in frontmatter auto-sets __confidence__)
g.execute('VAULT SYNC')  # re-index after external edits

# Wikilinks become graph edges
# [[related-note]] in Links section → EDGES FROM "note:x"
g.execute('VAULT BACKLINKS "my-note"')
```

Note kinds: `instruction`, `goal`, `context`, `plan`, `memory`, `artifact`, `log`, `daily`, `entity`, `fact`, `scratch`

</details>

<details>
<summary><strong>Hypothesis Testing</strong> - snapshot, explore, rollback</summary>

```sql
SYS SNAPSHOT "before-hypothesis"
-- ... explore reasoning branch ...
SYS ROLLBACK TO "before-hypothesis"

WHAT IF RETRACT "belief:earth-is-flat"
-- Returns affected nodes without committing
```

</details>

<details>
<summary><strong>Temporal + Working Memory</strong></summary>

```sql
CREATE NODE "scratch:123" kind = "working" data = "..." EXPIRES IN 30m
NODES WHERE __created_at__ > NOW() - 7d
NODES WHERE __updated_at__ > TODAY ORDER BY __created_at__ DESC
SYS EXPIRE WHERE kind = "working"
MERGE NODE "memory:old" INTO "memory:canonical"
```

</details>

<details>
<summary><strong>Aggregations</strong> - GROUP BY with numpy-vectorized ops</summary>

```sql
AGGREGATE NODES WHERE kind = "memory"
  GROUP BY topic
  SELECT COUNT(), AVG(importance)
  HAVING COUNT() > 2
  ORDER BY AVG(importance) DESC
```

</details>

<details>
<summary><strong>Context Isolation</strong></summary>

```sql
BIND CONTEXT "reasoning-session-42"
CREATE NODE "hyp:1" kind = "hypothesis" content = "maybe X"
RECALL FROM "hyp:1" DEPTH 3 LIMIT 10
DISCARD CONTEXT "reasoning-session-42"
```

</details>

<details>
<summary><strong>Voice</strong> (opt-in: graphstore install-voice)</summary>

```python
g = GraphStore(path="./brain", voice=True)

# Agent speaks
g.speak("The Q3 revenue grew 15%")

# Agent listens (real-time streaming)
g.listen(on_text=lambda text: agent.process(text))
g.stop_listening()

# Ingest audio files
g.execute('INGEST "meeting.wav"')  # Moonshine transcribes → chunks → embeds
```

</details>

## 📖 DSL Reference

<details>
<summary><strong>Reads</strong> - queries, traversals, path finding, pattern matching, semantic search</summary>

```sql
NODE "node_id"
NODE "node_id" WITH DOCUMENT
NODES WHERE kind = "function" AND file = "app.py" LIMIT 10
EDGES FROM "node_id" WHERE kind = "calls"
TRAVERSE FROM "node_id" DEPTH 3 WHERE kind = "calls"
SUBGRAPH FROM "node_id" DEPTH 2
PATH FROM "a" TO "b" MAX_DEPTH 5
SHORTEST PATH FROM "a" TO "b"
WEIGHTED SHORTEST PATH FROM "a" TO "b"
ANCESTORS OF "node_id" DEPTH 3
DESCENDANTS OF "node_id" DEPTH 3
COMMON NEIGHBORS OF "a" AND "b"
MATCH ("fn_main") -[kind = "calls"]-> (callee)
COUNT NODES WHERE kind = "function"
AGGREGATE NODES GROUP BY kind SELECT COUNT()
RECALL FROM "concept:x" DEPTH 3 LIMIT 10
SIMILAR TO "search text" LIMIT 10
SIMILAR TO [0.1, 0.2, ...] LIMIT 10 WHERE kind = "memory"
WHAT IF RETRACT "belief:x"
```

</details>

<details>
<summary><strong>Writes</strong> - create, update, delete, beliefs, documents, voice</summary>

```sql
CREATE NODE "id" kind = "x" name = "foo"
CREATE NODE "id" kind = "x" DOCUMENT "full text..." EXPIRES IN 1h
UPDATE NODE "id" SET name = "new"
UPSERT NODE "id" kind = "x" name = "foo"
DELETE NODE "id"
DELETE NODES WHERE kind = "test"
UPDATE NODES WHERE kind = "fact" SET confidence = 0.5
CREATE EDGE "source" -> "target" kind = "calls"
INCREMENT NODE "id" hits BY 1
ASSERT "fact:x" kind = "fact" value = 42 CONFIDENCE 0.9 SOURCE "tool"
RETRACT "fact:x" REASON "outdated"
MERGE NODE "old" INTO "canonical"
PROPAGATE "belief:x" FIELD confidence DEPTH 3
INGEST "report.pdf" AS "doc:q3" KIND "report"
CONNECT NODE "chunk:7" THRESHOLD 0.8
BIND CONTEXT "session-1"
DISCARD CONTEXT "session-1"
BEGIN ... COMMIT
```

</details>

<details>
<summary><strong>System</strong> - stats, schema, maintenance, model management</summary>

```sql
SYS STATUS
SYS STATS / SYS STATS NODES / SYS STATS MEMORY / SYS STATS DOCUMENTS
SYS REGISTER NODE KIND "memory" REQUIRED topic:string, importance:float EMBED content
SYS KINDS / SYS EDGE KINDS / SYS DESCRIBE NODE "memory"
SYS CONTRADICTIONS WHERE kind = "belief" FIELD value GROUP BY topic
SYS CONNECT / SYS CONNECT THRESHOLD 0.9
SYS DUPLICATES THRESHOLD 0.95
SYS EXPIRE / SYS EXPIRE WHERE kind = "working"
SYS SNAPSHOT "name" / SYS ROLLBACK TO "name" / SYS SNAPSHOTS
SYS EMBEDDERS / SYS SET EMBEDDER "embeddinggemma-300m" / SYS REEMBED
SYS INGESTORS
SYS CHECKPOINT / SYS REBUILD INDICES / SYS CLEAR CACHE
SYS EXPLAIN TRAVERSE FROM "a" DEPTH 5
SYS SLOW QUERIES LIMIT 10 / SYS FAILED QUERIES LIMIT 10
```

</details>

<details>
<summary><strong>Vault</strong> - markdown note operations</summary>

```sql
VAULT NEW "title" KIND "memory" TAGS "tag1,tag2"
VAULT READ "title"
VAULT WRITE "title" SECTION "body" CONTENT "..."
VAULT APPEND "title" SECTION "body" CONTENT "..."
VAULT SEARCH "query" LIMIT 10
VAULT BACKLINKS "title"
VAULT LIST [WHERE ...] [ORDER BY ...] [LIMIT n]
VAULT SYNC
VAULT DAILY
VAULT ARCHIVE "title"
```

</details>

## ⚡ Performance (100k nodes)

| Operation | Time |
|---|---|
| Point lookup | 4 μs |
| Filtered scan LIMIT 10 | 68 μs |
| COUNT | 41 μs |
| ORDER BY LIMIT 10 | 168 μs |
| GROUP BY + AVG | 788 μs |
| SIMILAR TO LIMIT 10 | 127 μs |
| RECALL DEPTH 3 | 983 μs |
| ASSERT / RETRACT | 4-9 μs |
| UPDATE NODES 50k bulk | 171 μs |
| SYS SNAPSHOT | 1.8 ms |
| Memory per node | 66 bytes (columns) + 1 KB (vector) |

## ⚙️ Configuration

```python
g = GraphStore(
    path="./brain",          # persistence directory (None = in-memory)
    ceiling_mb=256,          # graph memory ceiling
    embedder="default",      # "default" (model2vec) or custom Embedder
    vision_model=None,       # "smolvlm2" or "qwen3-vl" (opt-in)
    voice=False,             # True to enable STT/TTS (opt-in)
)
```

## 🏗️ Architecture

<details>
<summary>Project structure</summary>

```
graphstore/
├── __init__.py               # Thin re-exports
├── graphstore.py             # GraphStore facade + WAL + routing
├── core/                     # Graph engine
│   ├── store.py              # CoreStore: columnar node CRUD
│   ├── columns.py            # ColumnStore: typed numpy arrays
│   ├── edges.py              # EdgeMatrices: scipy CSR
│   ├── strings.py            # StringTable: string interning
│   ├── schema.py             # SchemaRegistry + EMBED field
│   ├── path.py               # BFS, Dijkstra, common neighbors
│   ├── memory.py             # Ceiling enforcement
│   ├── types.py              # Result, Edge dataclasses
│   └── errors.py             # Error hierarchy
├── dsl/                      # Query language
│   ├── grammar.lark          # Lark LALR(1) grammar
│   ├── parser.py             # Parser + LRU cache
│   ├── transformer.py        # Parse tree → AST
│   ├── ast_nodes.py          # 60+ AST dataclasses
│   ├── executor_base.py      # Shared: live_mask, eval_where, column filters
│   ├── executor_reads.py     # NODES, RECALL, SIMILAR TO, AGGREGATE, MATCH
│   ├── executor_writes.py    # CREATE, ASSERT, RETRACT, INGEST, MERGE, BATCH
│   ├── executor_system.py    # SYS commands
│   ├── executor.py           # Unified dispatcher
│   └── cost_estimator.py     # Frontier-based cost rejection
├── embedding/                # Text → vectors
│   ├── base.py               # Embedder protocol
│   ├── model2vec_embedder.py # Default (30MB, CPU)
│   ├── onnx_hf_embedder.py   # EmbeddingGemma ONNX (opt-in)
│   └── postprocess.py        # L2 normalize, Matryoshka truncation
├── vector/                   # Semantic search
│   ├── index.py              # usearch HNSW wrapper
│   └── store.py              # VectorStore: slot ↔ vector
├── document/                 # Document storage
│   └── store.py              # DocumentStore: SQLite multi-table
├── ingest/                   # File → graph
│   ├── base.py               # Ingestor protocol
│   ├── markitdown_ingestor.py  # Tier 1: general files
│   ├── pymupdf4llm_ingestor.py # Tier 2: PDF structure
│   ├── docling_ingestor.py   # Tier 3: hard PDFs (lazy)
│   ├── chunker.py            # Text splitting + summaries
│   ├── vision.py             # SmolVLM2/Qwen3-VL via Ollama
│   ├── router.py             # Tiered routing
│   └── connector.py          # SYS CONNECT cross-doc wiring
├── voice/                    # Speech (opt-in)
│   ├── stt.py                # Moonshine STT
│   └── tts.py                # Piper TTS
├── vault/                    # Markdown notes
│   ├── parser.py             # Frontmatter, sections, wikilinks
│   ├── manager.py            # Note CRUD (new/read/write/append)
│   ├── sync.py               # Vault dir → graphstore sync
│   └── executor.py           # VAULT DSL command handler
├── registry/                 # Model management
│   ├── models.py             # Supported models config
│   ├── installer.py          # Download + verify + smoke test
│   └── manifest.py           # Model manifest schema
├── persistence/              # SQLite checkpoints
│   ├── database.py
│   ├── serializer.py
│   └── deserializer.py
├── server.py                 # FastAPI playground
└── cli.py                    # CLI commands
```

</details>

## 🎮 Playground

```bash
pip install graphstore[playground]
graphstore playground
```

Three-panel UI: CodeMirror editor, React Flow graph visualization, stacked query results.

## 🛠️ Development

```bash
git clone https://github.com/orkait/graphstore.git
cd graphstore
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## 📄 License

MIT
