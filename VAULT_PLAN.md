# Vault Feature - Implementation Plan

## What we're building

A `vault` module inside graphstore that turns a directory of markdown files into a first-class agent memory + knowledge system. Agents write structured notes, graphstore indexes them (graph + vector), and the notes remain human-readable in any Obsidian-compatible viewer.

---

## Integration

### GraphStore init

Add a `vault` parameter to `GraphStore.__init__`:

```python
g = GraphStore(path="./brain", vault="./notes")
```

- `vault=None` (default) - vault feature disabled, no behaviour change
- `vault="./notes"` - instantiates `VaultManager`, registers `VAULT *` DSL commands, runs initial sync

---

## Standard Note Format

Every note is a `.md` file with YAML frontmatter + fixed sections.

```markdown
---
kind: memory
tags: [research, ai]
created: 2026-03-21T10:00:00
updated: 2026-03-21T10:00:00
status: active
agent: claude-sonnet-4-6
---

## Summary
One paragraph. Indexed for vector search. Required.

## Body
Free-form markdown content.

## Instructions
(Optional) Agent-facing directives that persist across sessions.

## Links
- [[related-note-title]]
- [[another-note]]
```

### Note kinds

| Kind | Purpose |
|------|---------|
| `memory` | Agent learned something and wants to retain it |
| `instruction` | Standing rule or directive (agent loads at session start) |
| `context` | Descriptive world knowledge (project, system, preference) |
| `entity` | A named thing - person, project, codebase, system |
| `fact` | Discrete verifiable belief with confidence. Maps to `ASSERT`/`RETRACT` |
| `artifact` | Output produced by agent - plan, report, summary |
| `log` | Append-only record of agent actions and decisions |
| `daily` | Date-scoped working note (`YYYY-MM-DD.md`) |
| `scratch` | Ephemeral, disposable. Can be given an expiry. |

### File naming

- Title → kebab-case slug → `<slug>.md`
- Daily: `2026-03-21.md`
- Stored flat in vault root (no subdirectory structure for simplicity)

---

## Graph Mapping

When a note is created or updated, it becomes a node + edges in graphstore.

### Node

```sql
CREATE NODE "note:<slug>"
  kind = "note"
  note_kind = "<memory|instruction|...>"
  tags = "tag1,tag2"
  status = "active"
  agent = "<agent-name>"
  title = "<title>"
  file = "<slug>.md"
  DOCUMENT "<full markdown content>"
  -- summary section is EMBED field (vector indexed)
```

### Edges (wikilinks)

Every `[[note-title]]` in the Links section becomes:

```sql
CREATE EDGE "note:<source-slug>" -> "note:<target-slug>" kind = "links"
```

### Backlinks query

```sql
-- notes that link TO "note:x"
EDGES FROM "note:x" WHERE kind = "links"
-- or full traversal
ANCESTORS OF "note:x" DEPTH 1 WHERE kind = "note"
```

---

## New DSL Commands

All commands are prefixed `VAULT`. They route through a new `VaultExecutor`.

```sql
-- Create a new note file + index it
VAULT NEW "title" KIND "memory" TAGS "research,ai"

-- Read note content from file
VAULT READ "title"

-- Overwrite a section (body | instructions | summary)
VAULT WRITE "title" SECTION "body" CONTENT "new content here"

-- Append to a section
VAULT APPEND "title" SECTION "body" CONTENT "additional content"

-- Semantic search across notes
VAULT SEARCH "agent memory systems" LIMIT 10
VAULT SEARCH "deployment steps" LIMIT 5 WHERE note_kind = "instruction"

-- Find notes that link to this note
VAULT BACKLINKS "title"

-- List/filter notes (uses existing columnar WHERE engine)
VAULT LIST
VAULT LIST WHERE note_kind = "instruction" AND status = "active"
VAULT LIST WHERE tags CONTAINS "research" ORDER BY __created_at__ DESC

-- Re-index vault directory (handles externally edited files)
VAULT SYNC

-- Create or open today's daily note
VAULT DAILY

-- Archive a note (sets status = "archived", does not delete file)
VAULT ARCHIVE "title"
```

---

## New Module: `graphstore/vault/`

```
graphstore/vault/
├── __init__.py          # exports VaultManager
├── manager.py           # VaultManager: file I/O, create/read/write/append notes
├── parser.py            # parse frontmatter, extract sections, resolve [[wikilinks]]
├── sync.py              # VaultSync: mtime-based re-indexing of vault dir → graphstore
└── executor.py          # VaultExecutor: handles VAULT * AST nodes
```

### manager.py - VaultManager

Responsibilities:
- `new(title, kind, tags, agent)` → write `.md` file with frontmatter + empty sections
- `read(title)` → return full file content
- `write_section(title, section, content)` → overwrite a section in the file
- `append_section(title, section, content)` → append to a section in the file
- `daily()` → create or return today's `YYYY-MM-DD.md`
- `archive(title)` → update frontmatter `status: archived`
- After every write: update `updated` timestamp in frontmatter, call `_sync_note_to_graph(slug)`

### parser.py - NoteParser

Responsibilities:
- `parse_frontmatter(content)` → returns dict of YAML fields
- `parse_sections(content)` → returns dict of section name → content
- `extract_wikilinks(content)` → returns list of `[[target]]` slugs from Links section
- `write_frontmatter(content, updates)` → returns updated file content with new frontmatter values
- `write_section(content, section, new_content)` → returns updated file content with replaced section

### sync.py - VaultSync

Responsibilities:
- `sync_all()` → walk vault dir, compare file mtimes against graphstore node `__updated_at__`, re-index stale files
- `sync_file(path)` → parse + upsert node + recreate edges for one file
- Called on `VAULT SYNC` and on `GraphStore.__init__` when vault is set

### executor.py - VaultExecutor

Responsibilities:
- Handles all `Vault*` AST nodes dispatched from the main Executor
- Delegates file ops to `VaultManager`, graph ops to `CoreStore`
- Returns `Result` objects consistent with rest of DSL

---

## DSL Integration Points

### 1. Grammar (`graphstore/dsl/grammar.lark`)

Add `VAULT` statement rules for each command.

### 2. AST nodes (`graphstore/dsl/ast_nodes.py`)

Add dataclasses:
- `VaultNew`, `VaultRead`, `VaultWrite`, `VaultAppend`
- `VaultSearch`, `VaultBacklinks`, `VaultList`
- `VaultSync`, `VaultDaily`, `VaultArchive`

### 3. Executor routing (`graphstore/dsl/executor.py`)

Add `Vault*` types to `_dispatch` handler map, routing to `VaultExecutor`.

### 4. GraphStore facade (`graphstore/graphstore.py`)

- Add `vault: str | Path | None = None` to `__init__`
- Instantiate `VaultManager` and `VaultSync` if vault is set
- Pass vault executor into the DSL executor chain

---

## Toolkit Plugin (separate repo - thin wrapper)

After graphstore vault is implemented, add `plugins/notes_plugin.py` to the toolkit:

```python
class NotesPlugin(RookPlugin):
    name = "notes"
    description = "Agent-facing markdown vault powered by graphstore"

    def register(self, app: typer.Typer) -> None:
        vault = os.environ.get("ROOK_VAULT")
        brain = os.environ.get("ROOK_BRAIN")

        # commands: new, read, write, append, search, backlinks, list, sync, daily, archive
        # each command: GraphStore(path=brain, vault=vault).execute("VAULT ...")
        # --vault and --brain flags override env vars
        # --json flag for structured output
```

---

## Implementation Order

1. `graphstore/vault/parser.py` - note parsing utils (no dependencies)
2. `graphstore/vault/manager.py` - file I/O using parser
3. `graphstore/vault/sync.py` - vault → graph sync using manager + CoreStore
4. DSL: grammar + AST nodes + VaultExecutor
5. Wire into `GraphStore.__init__` and `Executor._dispatch`
6. `graphstore/vault/__init__.py` - clean exports
7. Tests
8. Toolkit plugin (`plugins/notes_plugin.py`) - after graphstore is done
