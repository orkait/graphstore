# graphstore Comprehensive Summary

Updated: 2026-03-25

This file captures both:

- a broader repository summary for future sessions
- the current session state: scoped codebase understanding, corrected findings, behaviour analysis, and the design direction agreed so far for improving graphstore

It is still not a line-by-line full-codebase monograph, but it is substantially broader than a single-session note.

## Package metadata

- package name: `graphstore`
- package version: `0.3.0`
- Python: `>=3.10`
- license: `MIT`
- maturity classifier: `Development Status :: 3 - Alpha`
- console entry point: `graphstore = graphstore.cli:main`

Core dependencies from `pyproject.toml`:

- `numpy`
- `scipy`
- `lark`
- `usearch`
- `model2vec`
- `pyyaml`
- `markitdown`
- `pymupdf4llm`
- `pymupdf`
- `msgspec`
- `croniter`

Optional extras:

- `dev`
  - pytest, benchmark, coverage, httpx
- `playground`
  - fastapi, uvicorn
- `ingest-pro`
  - docling, openai

## What graphstore is

The repo aims to be an agent memory substrate, not only a graph library and not only a visual app.

The intended value proposition is a combined memory system:

- graph relations for association and traversal
- vector similarity for meaning-based recall
- lexical search for exact phrase retrieval
- columnar structured storage for fast filtering and aggregation
- document storage and ingestion for large cold content
- vault integration for human-readable markdown memory
- optional voice for speech I/O

The codebase is therefore best understood as a small platform with a graph engine at the center and several opt-in capability layers around it.

## Repository layout

Top-level repository structure:

- `graphstore/`
  - main Python package
- `tests/`
  - 49 pytest files covering engine, DSL, persistence, server, vector, ingest, vault, and voice
- `playground/`
  - Vite/React frontend for the optional browser playground
- `docs/superpowers/specs/`
  - historical design specs for major features
- `docs/superpowers/plans/`
  - historical implementation plans tied to those specs
- `mcp_refs/`
  - reference notes on retrieval, concurrency, self-balancing, datasets, and prior art
- `assets/`
  - diagrams and static project assets
- top-level working docs
  - `README.md`
  - `summary.md`
  - `VAULT_PLAN.md`
  - `VAULT_REVISION.md`
- helper/benchmark scripts
  - `populate_zen.py`
  - `bench_brain.py`
  - `bench_flip.py`
  - `bench_vectors.py`
  - `scripts/download_fixtures.py`

## Scope of analysis

- The deep ingest was run in a scoped codemode over `pyproject.toml` and `graphstore/**/*.py`.
- The main design/spec work focused on three areas:
  - runtime/admin hardening
  - `SYS EVICT` semantics
  - production-grade voice support
- The playground was explicitly de-scoped as an authoritative target. It may drift. Core runtime and server contracts are the priority.

## Codemode snapshot

- Scoped files:
  - `graphstore/**/*.py`: 78 files, 9,921 lines
  - `pyproject.toml` + `graphstore/**/*.py`: 12,298 total scoped lines
- Main runtime spine:
  - `graphstore/graphstore.py`
  - `graphstore/core/store.py`
  - `graphstore/dsl/executor.py`
  - `graphstore/dsl/executor_system.py`
  - `graphstore/document/store.py`
  - `graphstore/vector/store.py`
  - `graphstore/server.py`
  - `graphstore/vault/sync.py`

## Public package surface

The package exports a reasonably broad top-level Python API through `graphstore/__init__.py`.

Main exports:

- runtime objects
  - `GraphStore`
  - `CoreStore`
  - `SchemaRegistry`
- types
  - `Result`
  - `Edge`
- parsing/execution helpers
  - `parse`
  - `clear_cache`
  - `Executor`
  - `SystemExecutor`
- config types and helpers
  - `GraphStoreConfig`
  - `load_config`
  - `save_config`
  - section structs like `CoreConfig`, `VectorConfig`, `DocumentConfig`, `DslConfig`, `VaultConfig`, `PersistenceConfig`, `RetentionConfig`, `ServerConfig`
- error surface
  - `GraphStoreError`
  - `QueryError`
  - `NodeNotFound`
  - `NodeExists`
  - `CeilingExceeded`
  - `VersionMismatch`
  - `SchemaError`
  - `CostThresholdExceeded`
  - `BatchRollback`
  - `AggregationError`
  - `VectorError`
  - `EmbedderRequired`
  - `VectorNotFound`
  - `OptimizationInProgress`

This means the project exposes both the high-level application service (`GraphStore`) and a fair amount of lower-level implementation surface. That is useful for power users, but it also means public API boundaries are less strict than they could be.

## Architecture snapshot

The codebase is a strong systems-oriented prototype with a coherent core:

- `core/`
  - columnar node storage in numpy
  - sparse edge traversal via CSR matrices
  - string interning
  - tombstones first, compaction later
- `dsl/`
  - parser -> AST -> executor pipeline is clear
  - separation between normal DSL and `SYS` execution is good
- `persistence/`
  - sqlite is used as persistence substrate, not the live query engine
- optional subsystems
  - vector, document, vault, voice, registry

The main architectural pattern is:

- in-memory authoritative runtime
- sqlite-backed durability and metadata
- DSL as the primary interaction surface
- python methods as secondary integration surface

This is not a “database server first” architecture. It is an embedded runtime with optional adapter layers.

## Core subsystem map

### `graphstore/graphstore.py`

Main facade and runtime bootstrap:

- loads config
- initializes store/schema/executors
- wires persistence, WAL, optimizer, document store, vector store, vault, and optional voice
- exposes `execute()` and the main public service methods

### `graphstore/core/`

Main engine internals:

- `store.py`
  - authoritative graph state, node/edge CRUD, materialization
- `columns.py`
  - typed columnar arrays for fields and reserved fields
- `edges.py`
  - sparse matrix edge storage and caches
- `strings.py`
  - string interning and lookup
- `schema.py`
  - registered node/edge kinds and EMBED field declarations
- `path.py`
  - path/traversal algorithms
- `memory.py`
  - measurement and ceiling enforcement
- `optimizer.py`
  - compaction, GC, cleanup, eviction
- `scheduler.py`
  - self-balancing hooks and auto-optimize scheduling
- `queue.py`
  - single-writer command queue for threaded mode
- `types.py`
  - result and edge datatypes
- `errors.py`
  - domain-specific error hierarchy

### `graphstore/dsl/`

Query system:

- `grammar.lark`
  - DSL grammar
- `parser.py`
  - parser + cache
- `transformer.py`
  - parse tree -> AST
- `ast_nodes.py`
  - AST dataclasses
- `executor.py`
  - user query dispatch
- `executor_system.py`
  - `SYS` command dispatch
- `visibility.py`
  - unified visibility semantics
- `filtering.py`
  - WHERE evaluation and filtering acceleration
- `handlers/`
  - domain mixins by command category

### `graphstore/vector/` and `graphstore/embedding/`

Semantic retrieval layer:

- vector index and slot mapping
- embedder abstraction
- default model2vec path
- optional ONNX/HF path

### `graphstore/document/` and `graphstore/ingest/`

Cold document and file-ingestion layer:

- sqlite document storage
- summaries and FTS
- parser routing
- chunking
- vision support
- similarity-based connector logic

### `graphstore/vault/`

Markdown note system:

- parse frontmatter/sections/wikilinks
- manage note files
- sync vault contents into graphstore
- expose `VAULT *` operations

### `graphstore/server.py`

Optional FastAPI adapter:

- `/api/execute`
- `/api/execute-batch`
- `/api/graph`
- `/api/reset`
- `/api/script`
- `/api/config`
- `/api/logs`

### `graphstore/cli.py`

CLI adapter:

- `graphstore playground`
- `graphstore install-embedder`
- `graphstore list-embedders`
- `graphstore uninstall-embedder`
- `graphstore install-voice`
- `graphstore list-voice`

High-level rating from this session:

- overall: `7/10`
- core engine/library: `8.5/10`
- product/integration polish: `5.5-6/10`

## Important correction

One earlier concern was wrong and should not be carried forward:

- `GraphStore` does already expose:
  - `get_all_nodes()`
  - `get_all_edges()`
  - `cost_threshold`
  - `ceiling_mb`
- So the earlier “server drift because these passthroughs are missing” finding is retracted.

Relevant locations:

- `graphstore/graphstore.py:433-457`
- `graphstore/server.py:212-274`

## CLI surface

The CLI is small but strategically important because it is the operational entry point for optional features.

Current commands:

- `graphstore playground`
  - launches the FastAPI server and serves static frontend files if available
- `graphstore install-embedder <name> [--variant ...]`
  - fetches and installs optional embedding models
- `graphstore list-embedders`
  - shows supported and installed models
- `graphstore uninstall-embedder <name>`
  - removes installed embedder artifacts
- `graphstore install-voice`
  - installs `moonshine` + `piper-tts`
- `graphstore list-voice`
  - reports availability of STT/TTS dependencies

Operational note:

- The CLI is capability-oriented rather than admin-oriented. It does not currently expose a richer runtime control plane for status, reset modes, or voice session management.

## Behaviour analysis summary

The strongest behavioural issues are about state truthfulness and incomplete contracts, not basic CRUD.

### State inventory

- backend store lifecycle
  - module-level singleton `_store`
  - lazily created
  - replaced on reset
- runtime config
  - live `ceiling_mb` and `cost_threshold` mutate the in-memory store/executor
  - no durable read/write contract at the server layer
- persistent metadata
  - script and logs rely on sqlite-backed mode
- voice state
  - `_stt`, `_tts`, `is_listening`, callback state exist
  - realtime streaming contract is not fully implemented

### Key behavioural findings

1. `SYS EVICT LIMIT N` is effectively broken.
- In `graphstore/dsl/executor_system.py`, `_evict()` sets `target = 0` when `q.limit` exists.
- In `graphstore/core/optimizer.py`, `evict_oldest()` returns immediately when `target_bytes <= 0`.
- Result: the count-based path does not perform count-based eviction.

2. Voice realtime STT is a contract stub, not a production feature.
- `graphstore/voice/stt.py` stores callback state and flips `_listening = True`
- It does not actually start a streaming transcription loop or invoke the callback path.

3. Voice TTS can fail silently.
- `graphstore/voice/tts.py` swallows synthesis exceptions and returns `b""`
- playback does not provide strong success/failure signaling

4. Logs input validation is weak.
- `since` parsing uses `datetime.fromisoformat()` directly in both:
  - `graphstore/server.py`
  - `graphstore/dsl/executor_system.py`
- Invalid timestamps are likely to surface as uncaught runtime errors instead of typed validation failures.

5. Reset logic performs partial teardown.
- `graphstore/server.py` manually manipulates sqlite state and replaces `_store`
- it does not route teardown through `GraphStore.close()`
- that leaves cleanup semantics weaker than the public runtime contract should be

### Behaviour verdict

- Core graph CRUD and graph read surfaces are mostly fine.
- The current weaknesses are:
  - admin semantics are underspecified
  - some “success” paths do not mean “durable/complete/safe”
  - voice exposes interfaces without full lifecycle/runtime backing

## DSL capability summary

The DSL is broad enough that graphstore functions more like a small language runtime than a simple query parser.

Main command families visible from source and docs:

- point and set reads
  - `NODE`, `NODES`, `EDGES`, `COUNT`
- traversal and graph algorithms
  - `TRAVERSE`, `PATH`, `SHORTEST PATH`, `ANCESTORS`, `DESCENDANTS`, `COMMON NEIGHBORS`
- pattern matching
  - `MATCH`
- aggregations
  - `AGGREGATE ... GROUP BY ...`
- mutations
  - `CREATE`, `UPDATE`, `DELETE`, `UPSERT`, `MERGE`, `INCREMENT`
- belief and contradiction management
  - `ASSERT`, `RETRACT`, `PROPAGATE`, `SYS CONTRADICTIONS`
- context and isolation
  - `BIND CONTEXT`, `DISCARD CONTEXT`
- semantic and lexical retrieval
  - `SIMILAR TO`, `LEXICAL SEARCH`, `RECALL`, `REMEMBER`
- ingest and document intelligence
  - `INGEST`, `CONNECT NODE`, `SYS CONNECT`, `SYS DUPLICATES`
- system/runtime control
  - `SYS STATUS`, `SYS HEALTH`, `SYS OPTIMIZE`, `SYS CHECKPOINT`, `SYS SNAPSHOT`, `SYS ROLLBACK`, `SYS LOG`, `SYS CRON`, `SYS REEMBED`, `SYS RETAIN`, `SYS EXPIRE`
- vault commands
  - `VAULT *`

This breadth is one of the project’s strengths. It also increases the burden on:

- parser stability
- AST coverage
- executor separation
- test completeness

## Test landscape

The repository has 49 pytest files.

Coverage clusters:

- core engine and data structures
  - `test_store.py`
  - `test_columns.py`
  - `test_columnar_store.py`
  - `test_column_integration.py`
  - `test_edges.py`
  - `test_strings.py`
  - `test_path.py`
  - `test_types.py`
- memory and optimization
  - `test_memory.py`
  - `test_memory_accounting.py`
  - `test_optimizer.py`
  - `test_auto_reembed.py`
- DSL parsing and behavior
  - `test_dsl_parser.py`
  - `test_dsl_user_reads.py`
  - `test_dsl_user_writes.py`
  - `test_dsl_system.py`
  - `test_untested_commands.py`
  - `test_order_by_string.py`
  - `test_aggregate.py`
  - `test_beliefs.py`
  - `test_recall.py`
  - `test_remember.py`
  - `test_remember_signals.py`
  - `test_similar.py`
- persistence and WAL/restore
  - `test_persistence.py`
  - `test_incremental_checkpoint.py`
  - `test_deserializer_kinds.py`
- config/schema
  - `test_config.py`
  - `test_schema.py`
- vector and embedding
  - `test_vector_store.py`
  - `test_vector_persist.py`
  - `test_embedding.py`
  - `test_onnx_embedder.py`
  - `test_e2e_real_embedder.py`
- ingest/document/vault
  - `test_document_store.py`
  - `test_ingest.py`
  - `test_vault.py`
- server and security
  - `test_server.py`
  - `test_server_endpoints.py`
  - `test_server_security.py`
  - `test_log_layer.py`
- concurrency and scheduling
  - `test_command_queue.py`
  - `test_cron.py`
- broader integration and gaps
  - `test_integration.py`
  - `test_gaps.py`
  - `test_edges_to_perf.py`
  - `test_cost_estimator.py`
  - `test_sys_status.py`

Test observations from this session:

- the suite is broad
- several historically risky surfaces do have tests
- but some current spec-critical gaps still exist, especially:
  - no strong test for `SYS EVICT LIMIT N` semantics
  - no real verification of streaming voice behavior
  - limited evidence of robust validation/error-contract testing for admin endpoints

## Design and implementation history in-repo

The repository keeps a useful internal history of how major subsystems were introduced.

### Historical specs under `docs/superpowers/specs/`

- `2026-03-18-playground-design.md`
  - initial playground product shape
- `2026-03-18-light-mode-design-tokens.md`
  - playground visual system work
- `2026-03-19-high-fanout-visualization-design.md`
  - advanced graph visualization modes
- `2026-03-20-agentic-brain-db-design.md`
  - shift from typed graph DB toward agent memory substrate
- `2026-03-20-columnar-storage-design.md`
  - columnar acceleration/source-of-truth evolution
- `2026-03-20-vector-store-design.md`
  - semantic retrieval layer
- `2026-03-21-document-layer-design.md`
  - document ingestion and storage
- `2026-03-21-opt-in-layers-design.md`
  - document, vision, voice, embedder opt-ins
- `2026-03-24-command-queue-design.md`
  - threaded single-writer queue
- `2026-03-24-log-cron-design.md`
  - intelligent log layer and cron scheduler

### Historical plans under `docs/superpowers/plans/`

- implementation plans exist for nearly every major feature above
- later plans also include:
  - `2026-03-24-implementation-improvements.md`
  - `2026-03-24-quality-improvements.md`
- together they show that the repo has been developed through a design-first workflow rather than only ad hoc coding

## Top-level reference documents

### `VAULT_PLAN.md`

The original vault plan frames the vault as a first-class human/agent markdown knowledge system with graph indexing and sync.

### `VAULT_REVISION.md`

This file narrows and corrects the vault goal:

- the vault is a human-agent interface layer
- markdown files are the source of truth
- vault is not intended to become a second query engine
- graphstore DSL remains the main query/retrieval mechanism over indexed vault data

This revision is important because it clarifies architectural intent and prevents the vault from sprawling into a duplicate runtime.

### `mcp_refs/`

These files provide design rationale and future-session context:

- `concurrency-model.md`
  - explains the single-writer threaded model
- `retrieval-model.md`
  - explains why `REMEMBER` is the primary fused retrieval command
- `self-balancing.md`
  - explains optimization as agent-visible behavior, not silent background magic
- `prior-art-recall-pageindex.md`
  - notes external influences on lexical and retrieval design
- `mock-testing-datasets.md`
  - sketches dataset choices for ingest/vision/voice testing

## Playground status

The repository contains a substantial optional frontend:

- React + TypeScript + Vite
- Zustand for state
- React Flow for graph rendering
- CodeMirror for DSL editing
- Tailwind/shadcn UI shell

It has its own docs in `playground/README.md`, UI components, examples, and graph layout logic.

For this session’s design work, the playground was explicitly treated as non-authoritative. That means:

- its current behavior still matters as evidence
- but the runtime/server redesign should not be constrained to preserve playground assumptions
- adapter drift is acceptable if the core runtime becomes cleaner

## Design direction chosen in-session

User choices made during brainstorming:

- scope: combined improvement spec across runtime/admin hardening and voice completion
- voice target: `production-grade voice`
- breaking changes: `clean break if it materially improves the architecture`
- optimization priority: `feature completeness`
- design approach selected: boundary-first hardening, with a strong voice subsystem

## Agreed design direction so far

## 1. Architecture

The target architecture is three layers:

- `Layer 1: Engine`
  - graph state
  - parser/executors
  - persistence
  - vector/document subsystems
- `Layer 2: Runtime Surface`
  - `GraphStore` as the canonical public application service
  - the only supported surface for adapters and integrations
- `Layer 3: Adapters`
  - server
  - CLI
  - voice I/O adapters
  - future integrations

Intent:

- `GraphStore` becomes the explicit public boundary
- internal stores/executors remain internal
- adapters must not depend on private fields casually

## 2. Runtime surface

`GraphStore` should be a complete application service, not just a DSL wrapper.

Public surface groups proposed:

- query and mutation
  - `execute(query)`
  - `execute_batch(queries)`
  - `submit_background(query)` where threaded mode applies
- structured graph reads
  - explicit node/edge/graph access methods
- admin/runtime operations
  - checkpoint
  - replay
  - status/health
  - optimize
  - byte-based eviction
  - count-based eviction
- document/ingest operations
  - explicit access instead of private reach-through
- voice operations
  - session-oriented APIs rather than thin `speak/listen` wrappers

Important rule:

- if an adapter needs something and it is not on `GraphStore`, that is a public-surface design gap

## 3. Admin and state contracts

The current problem is that admin actions mix:

- session mutation
- runtime mutation
- persistent mutation
- destructive reset

under APIs that often only report `{ok: true}`.

Proposed direction:

- reset becomes explicit and scoped
  - `reset_memory()`
  - `reset_store(...)`
  - `reset_session()`
- config gets real read/write semantics
  - `get_runtime_config()`
  - `get_persisted_config()`
  - `update_runtime_config(...)`
  - `update_persisted_config(...)`
- eviction gets split
  - `evict_by_bytes(...)`
  - `evict_by_count(...)`
- admin/log/reset/config operations return structured result objects, not only boolean success

Note:

- Earlier discussion included some playground-facing reasoning, but the final direction should be interpreted as a core runtime/server contract change, not a playground-driven redesign.

## 4. Voice subsystem

The voice redesign should be session-based and production-grade.

### Core position

`graphstore` is a good home for:

- voice orchestration
- session registry/state
- persistence of transcripts and session metadata
- correlation with graph memory, logs, and traces

`graphstore` is not the right place to be the actual low-level audio engine.

### Voice ownership split

`GraphStore` / runtime owns:

- voice session registry
- provider selection
- capability checks
- transcript persistence
- synthesis/transcription job state
- session metadata and logs

Voice backends/providers own:

- realtime STT stream handling
- realtime TTS playback/synthesis streaming
- device I/O
- buffering/backpressure
- provider-specific recovery

### Public voice API direction

Replace thin wrappers with session APIs such as:

- `create_voice_session(config)`
- `get_voice_session(session_id)`
- `close_voice_session(session_id)`
- `transcribe_file(path, options)`
- `start_transcription(session_id)`
- `stop_transcription(session_id)`
- `synthesize(text, options)`
- `speak(text, options)`
- `interrupt_playback(...)`

### Voice state model

Proposed observable states:

- `created`
- `ready`
- `listening`
- `processing`
- `speaking`
- `paused`
- `stopped`
- `error`

### Internal module split proposed

- `voice/service.py`
  - public orchestration layer
- `voice/session.py`
  - session state machine and typed events
- `voice/providers/stt/*.py`
  - STT provider adapters
- `voice/providers/tts/*.py`
  - TTS provider adapters
- `voice/runtime.py`
  - workers, queues, buffering, cancellation, health

### Why graphstore still makes sense as the home for voice

The conclusion from this session was:

- `graphstore` is a good place for voice orchestration, session state, persistence, transcript integration, and operational contracts
- it is not the right place to make the core graph execution loop responsible for realtime audio transport

So the right shape is:

- `GraphStore` owns the control plane
- provider runtimes own media streaming
- only durable or explicitly persisted artifacts flow back into the graph engine

## Critical voice rule agreed in-session

This rule is important and should be preserved:

- transcript/audio events should not go through the normal DSL write queue one chunk at a time
- they should update voice session state through a dedicated voice runtime
- only finalized or explicitly persisted artifacts should cross into the graph engine

Reason:

- voice events are high-frequency, ephemeral, and latency-sensitive
- graph writes are lower-frequency, durable, and consistency-sensitive
- combining them in the same queue creates head-of-line blocking and failure coupling

What should cross into the graph engine:

- final transcript segments
- explicit memory writes
- durable checkpoints or summaries
- session metrics/audit records

What should not cross chunk-by-chunk:

- raw audio frames
- every partial transcript token
- playback chunk events
- device heartbeat or VAD noise

## Voice Data Model and Event Contracts

The voice subsystem operates entirely via a Session lifecycle.

### State Model (Observable States)
A session (`VoiceSession`) transitions through the following exact states:
- `created`: Session initialized but media devices not yet claimed. Let adapters configure routing.
- `ready`: STT/TTS providers warmed up, locks acquired, ready for I/O.
- `listening`: VAD/STT active. Capturing audio chunks from the microphone.
- `processing`: VAD detected silence; STT is processing the final transcript buffer. 
- `speaking`: TTS is actively streaming response audio to the output device.
- `paused`: Media routing temporarily suspended (e.g., user muted microphone without killing session).
- `stopped`: Session finalized, artifacts written to persistent memory, locks released.
- `error`: Unrecoverable exception (e.g., microphone disconnected). Artifacts partially saved.

### Event Contracts
The `VoiceSession` object emits typed `dataclass` events to consumers (e.g. adapters, clients, or GraphStore memory loggers):
- `SessionStateTransition(old_state, new_state, timestamp)`
- `PartialTranscript(text, is_final=False)`: Emitted by the STT worker during `listening`. *CRITICAL RULE:* These do NOT cross into the graph memory engine. They only flow to connected frontend adapters for realtime UX.
- `FinalTranscript(text, id, duration_ms, is_final=True)`: Emitted when `processing` completes. Crosses into graph memory.
- `SynthesisStarted(text, duration_ms)`
- `PlaybackInterrupted(reason)`

## Persistence Model for Voice Sessions and Transcripts

`GraphStore` (specifically the overarching `CoreStore` or a dedicated Sqlite table) persists definitive markers of the session, not the transient frames.

### Persistent Schema
If enabled via `config.voice.persistence`, sessions are backed by SQLite:
- `voice_sessions`: `id`, `created_at`, `ended_at`, `total_duration_ms`, `end_reason`.
- `voice_transcripts`: `id`, `session_id`, `role` ("user" | "agent"), `text`, `timestamp`, `audio_blob_id` (optional).

### Integration with Graph Memory
When a `FinalTranscript` is emitted by the STT worker containing user inputs, the session orchestrator performs an asynchronous query to GraphStore:
```
CREATE NODE auto kind "transcript" text "<text>" role "user" session "<session_id>"
```
This guarantees transcripts become nodes that `REMEMBER` or `SIMILAR TO` semantic searches can immediately hit for context reasoning.

## Migration Phases and Deprecation Policy

Transitioning from the current stub implementation to the orchestrated session API:

**Phase 1: Stub Deprecation & Orchestrator Intro (Next Minor)**
- Introduce `graphstore/voice/session.py` and empty state machine.
- Introduce `GraphStore.create_voice_session(...)` and log warnings on direct calls to `GraphStore.speak()` or `GraphStore.listen()`.
- Expose the new `update_runtime_config` paths for `VoiceConfig`.

**Phase 2: Rework STT & TTS Providers (Next Minor)**
- Move `stt.py` and `tts.py` to `voice/providers/`.
- Rewrite the `MoonshineSTT` class to spawn a background daemon thread for audio I/O that emits `PartialTranscript` and `FinalTranscript` using `queue.Queue`.
- Prevent TTS silent swallows (raise explicit exceptions which transition the session to `error`).

**Phase 3: Deep Persistence (Target 1.0)**
- Tie `FinalTranscript` into the main graph engine automatically if `auto_persist` is enabled.
- Remove old wrappers entirely.
- Establish strong test verification gates: mock audio streams proving session lifecycle transitions correctly handle interruptions.

## Rollout Plan & Test Verification Gates

Before shipping Phase 2, the following verification gates must pass:
1. **Mock End-to-End Simulation**: A mocked `STTProvider` must push pseudo-frames triggering full lifecycle transitions `created -> ready -> listening -> processing -> speaking -> stopped`.
2. **Interrupt Consistency Test**: Sending a `stop_listening()` while `speaking` or `processing` must gracefully cleanly terminate the worker threads without corrupting the socket.
3. **Graph Engine Independence**: A 5,000 requests/sec graph load test must show zero latency impact when a background voice session is furiously emitting `PartialTranscript` events (proving the queue boundaries act correctly).

## Files most relevant for next session

- `graphstore/graphstore.py`
- `graphstore/server.py`
- `graphstore/dsl/executor_system.py`
- `graphstore/core/optimizer.py`
- `graphstore/voice/stt.py`
- `graphstore/voice/tts.py`
- `graphstore/config.py`
- `graphstore/document/store.py`
- `tests/test_server.py`
- `tests/test_server_endpoints.py`
- `tests/test_server_security.py`
- `tests/test_memory_accounting.py`
- `tests/test_voice.py`

Additional useful files for broader orientation:

- `pyproject.toml`
- `graphstore/__init__.py`
- `graphstore/cli.py`
- `README.md`
- `playground/README.md`
- `VAULT_PLAN.md`
- `VAULT_REVISION.md`
- `docs/superpowers/specs/*.md`
- `docs/superpowers/plans/*.md`
- `mcp_refs/*.md`

## Verification limits in this session

No live runtime/browser validation was completed in this shell because required Python deps were missing here, including:

- `fastapi`
- `numpy`

So the conclusions in this summary are based on:

- direct source reading
- test reading
- behavioural reasoning from implementation paths

not on executed end-to-end runtime behavior.

## Practical next steps for a future session

If a later session wants to continue from this summary, the highest-value order is:

1. confirm the admin/runtime redesign scope
2. write the unfinished spec sections
3. codify precise test cases for:
   - `SYS EVICT LIMIT`
   - config persistence/runtime split
   - reset semantics
   - voice session states and event contracts
4. only then move to implementation work

If the goal instead is a truly exhaustive repo summary, the next missing layer would be:

- a per-file purpose matrix for every package and top-level file
- a dependency graph by subsystem
- a DSL feature matrix with parser/executor/test references
- a detailed test coverage gap analysis
