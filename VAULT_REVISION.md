# Vault Revision - Scope Correction

> This document amends VAULT_PLAN.md. An implementing agent should read both,
> with this file taking precedence where they conflict.

---

## Core Insight (from OpenClaw)

The vault is a **human-agent interface layer**, not a query engine.

- Humans write instructions, goals, plans, context in markdown files
- Agents read those files to get standing orders and context
- Agents write memories, logs, artifacts back as markdown for humans to inspect
- Plain markdown files are the source of truth - transparent and editable by anyone

The vault is NOT another way to query graphstore. For search, recall, backlinks,
filtering - agents use the existing graphstore DSL directly against indexed vault nodes.

---

## What to Remove from VAULT_PLAN.md

Remove these DSL commands - they duplicate graphstore and belong there:

```
VAULT SEARCH    →  use  SIMILAR TO "query" WHERE kind = "note"
VAULT BACKLINKS →  use  EDGES FROM "note:x" WHERE kind = "links"
VAULT LIST      →  use  NODES WHERE kind = "note" AND note_kind = "instruction"
```

---

## What Vault Actually Does

Vault handles file operations only. Agents use it to read/write structured markdown.
Graphstore handles all query/recall/search operations against the indexed content.

### Vault DSL (file operations only)

```sql
-- Create a new note file with standard structure
VAULT NEW "title" KIND "instruction" TAGS "tag1,tag2"

-- Read full file content
VAULT READ "title"

-- Overwrite a section (summary | body | instructions | links)
VAULT WRITE "title" SECTION "body" CONTENT "..."

-- Append to a section
VAULT APPEND "title" SECTION "body" CONTENT "..."

-- Create or open today's daily note (YYYY-MM-DD.md)
VAULT DAILY

-- Archive a note (sets status = "archived" in frontmatter, keeps file)
VAULT ARCHIVE "title"

-- Re-index vault dir into graphstore (handles external edits)
VAULT SYNC
```

That's the full surface area. 7 commands.

---

## Note Kinds - Revised Purpose

| Kind | Who writes it | Purpose |
|------|--------------|---------|
| `instruction` | Human | Standing rules agents load at session start |
| `goal` | Human | Current objectives and success criteria |
| `context` | Human | World knowledge - project details, preferences, constraints |
| `plan` | Human or Agent | Structured steps for a task |
| `memory` | Agent | Something the agent learned and wants to retain |
| `artifact` | Agent | Output produced - report, summary, code analysis |
| `log` | Agent | Append-only record of actions and decisions |
| `daily` | Agent | Date-scoped working note (`YYYY-MM-DD.md`) |
| `entity` | Human or Agent | A named thing - person, system, codebase |
| `fact` | Human or Agent | Discrete belief. Synced to graphstore `ASSERT` on write. |
| `scratch` | Agent | Ephemeral, disposable. Can have expiry. |

Removed: `artifact` is kept. Removed duplicate kinds that were just memory variants.

---

## Agent Session Start Pattern

An agent using the vault should open a session like this:

```python
g = GraphStore(path="./brain", vault="./notes")

# Load standing instructions from vault
instructions = g.execute('NODES WHERE kind = "note" AND note_kind = "instruction" AND status = "active"')

# Load current goals
goals = g.execute('NODES WHERE kind = "note" AND note_kind = "goal" AND status = "active"')

# Load relevant context (semantic)
context = g.execute('SIMILAR TO "current task description" LIMIT 5 WHERE note_kind = "context"')
```

The vault writes the files. Graphstore answers the queries. Clear separation.

---

## Fact Kind - Special Sync Behaviour

When a note of kind `fact` is written/updated, the vault manager should:

1. Write the markdown file as normal
2. Also issue a graphstore `ASSERT` for the fact, pulling `confidence` from frontmatter

```markdown
---
kind: fact
confidence: 0.95
source: human
---
## Summary
Production database is PostgreSQL 15.3
```

→ automatically becomes:
```sql
ASSERT "fact:<slug>" kind = "fact" confidence = 0.95 source = "human"
  DOCUMENT "Production database is PostgreSQL 15.3"
```

---

## Sync Behaviour (unchanged from VAULT_PLAN.md)

On `VAULT SYNC` and on `GraphStore.__init__` when vault is set:
- Walk vault dir
- Compare file mtimes to graphstore node `__updated_at__`
- Re-index stale files: upsert node + recreate edges
- `fact` kind files also re-assert in graphstore belief store

---

## Implementation Order (revised)

1. `graphstore/vault/parser.py` - frontmatter + section parsing + wikilink extraction
2. `graphstore/vault/manager.py` - file I/O: new, read, write_section, append_section, daily, archive
3. `graphstore/vault/sync.py` - mtime-based vault dir → graphstore sync (including fact assertions)
4. DSL: grammar rules + AST nodes for the 7 VAULT commands only
5. `graphstore/vault/executor.py` - VaultExecutor routing to manager + sync
6. Wire into `GraphStore.__init__` (vault param) and `Executor._dispatch`
7. Tests: file creation, section writes, sync, fact assertion, wikilink edge creation
8. Toolkit plugin (separate repo) - thin CLI wrapper, after graphstore is done
