# Graphstore Playground — Design Spec

## Overview

A browser-based interactive playground for the graphstore DSL. Users write and execute DSL queries in a code editor, see results in a table/JSON view, and visualize the graph in real-time using React Flow.

Launched via a single CLI command:

```bash
graphstore playground [--port 7200]
```

## Architecture

### Backend: FastAPI server

A FastAPI process that:
- Holds a single in-memory `GraphStore` instance per session
- Exposes a REST API at `/api/` for DSL execution
- Serves the built Vite frontend as static files from `playground/dist/`

### API Endpoints

| Endpoint | Method | Body / Params | Response |
|---|---|---|---|
| `/api/execute` | POST | `{ "query": "..." }` | `Result` JSON (see Response Types) |
| `/api/execute-batch` | POST | `{ "queries": ["..."] }` | `[Result]` JSON |
| `/api/graph` | GET | — | `{ nodes: [...], edges: [...] }` |
| `/api/reset` | POST | — | `{ "ok": true }` |
| `/api/config` | POST | `{ ceiling_mb?, cost_threshold? }` | `{ "ok": true }` |

**Error handling:** Successful queries return HTTP 200. Query errors (`QueryError`, `NodeNotFound`, `NodeExists`) return HTTP 400 with `{ "error": "message", "type": "ErrorClassName" }`. Server errors return HTTP 500.

**`/api/graph` implementation:** Iterates `CoreStore._edges_by_type` to collect all edges, calls `get_all_nodes()` for nodes. Adds `get_all_edges()` method to `CoreStore` for this.

### Response Types (TypeScript)

```typescript
interface Result {
  kind: "node" | "nodes" | "edges" | "path" | "paths" | "match"
      | "subgraph" | "distance" | "stats" | "plan" | "schema"
      | "log_entries" | "ok";
  data: any;     // shape depends on kind (see below)
  count: number;
  elapsed_us: number;
}

// kind="node"     → data: { id, kind, ...fields } | null
// kind="nodes"    → data: [{ id, kind, ...fields }]
// kind="edges"    → data: [{ source, target, kind }]
// kind="path"     → data: ["id1", "id2", ...] | null
// kind="paths"    → data: [["id1", "id2"], ...]
// kind="match"    → data: [{ varName: "nodeId", ... }]
// kind="subgraph" → data: { nodes: [...], edges: [...] }
// kind="distance" → data: number (-1 if no path)
// kind="stats"    → data: { node_count, edge_count, ... }
// kind="ok"       → data: null
```

### Frontend Stack

- Vite + React 18 + TypeScript
- Tailwind CSS
- shadcn/ui (ResizablePanelGroup, Button, DropdownMenu, Dialog, Tabs, Badge, etc.)
- Lucide icons
- React Flow v12
- `@uiw/react-codemirror` with custom DSL language mode
- dagre (for hierarchical graph layout)

## Layout

Three-panel resizable layout using shadcn `ResizablePanelGroup`:

```
┌─────────────────────────────────────────────────────────────────┐
│ Toolbar                                                         │
│ [Examples ▾] [Run Selected ▶] [Run All ▶▶] [Reset 🗑]          │
│ [View Mode ▾] [Layout ▾] [Settings ⚙]                          │
├───────────────────────┬─────────────────────────────────────────┤
│                       │                                         │
│   EDITOR              │   GRAPH (React Flow)                    │
│   ~40% width          │   ~60% width                            │
│                       │                                         │
│   CodeMirror with     │   Custom nodes (id + kind badge)        │
│   DSL highlighting    │   Labeled edges by kind                 │
│   Line numbers        │   Pan / zoom / fit controls             │
│   Ctrl+Enter: run     │   Minimap (toggleable)                  │
│                       │                                         │
├───────────────────────┤   Stats bar:                            │
│                       │   Nodes: N | Edges: N | Mem | Time      │
│   RESULTS (~30% h)    │                                         │
│   Collapsible         ├─────────────────────────────────────────┤
│   Stacked results     │                                         │
│   [Table] [JSON] tabs │                                         │
│   Error display       │                                         │
│                       │                                         │
└───────────────────────┴─────────────────────────────────────────┘
```

All panel splits are draggable. Dark theme by default.

## Editor Panel

**Component:** `@uiw/react-codemirror` with custom DSL language mode.

**Syntax highlighting:**
- Keywords: `CREATE`, `NODE`, `EDGES`, `WHERE`, `FROM`, `TO`, `TRAVERSE`, `MATCH`, `SYS`, `BEGIN`, `COMMIT`, `DELETE`, `UPDATE`, `UPSERT`, `INCREMENT`, `PATH`, `SHORTEST`, `DISTANCE`, `ANCESTORS`, `DESCENDANTS`, `COMMON`, `NEIGHBORS`, `SUBGRAPH`, `DEPTH`, `MAX_DEPTH`, `LIMIT`, `SET`, `BY`, `OF`, `AND`, `KIND`, `REGISTER`, `UNREGISTER`, `DESCRIBE`, `STATS`, `EXPLAIN`, `CHECKPOINT`, `REBUILD`, `INDICES`, `CLEAR`, `WAL`
- Operators: `=`, `!=`, `>`, `<`, `>=`, `<=`, `AND`, `OR`, `NOT`, `->`, `-[`, `]->`
- Strings: `"..."`
- Numbers: integers and floats
- System prefix: `SYS` in distinct color

**Keyboard shortcuts:**
- `Ctrl+Enter` — Run selected text (or current line if no selection)
- `Ctrl+Shift+Enter` — Run all queries in editor

**Behavior:**
- Multiple queries separated by newlines
- "Run Selected" executes only highlighted text
- "Run All" splits editor into queries: scan line-by-line; when a line starts with `BEGIN`, accumulate until `COMMIT` as one query; otherwise each non-blank line is a separate query
- BEGIN...COMMIT blocks treated as a single unit

## Graph Panel (React Flow v12)

**Custom node component:**
- Shows node `id` (bold) and `kind` as a colored badge
- Colors grouped by kind (deterministic color mapping)
- Click node → popover/tooltip with all node fields

**Edge rendering:**
- Label showing edge `kind`
- Stroke style varies per type (solid, dashed, dotted — cycled)
- Animated edges for highlighting query results

**Layout algorithms (toggleable):**
- Dagre — hierarchical/directed, good for call graphs and trees
- d3-force — force-directed, good for dense interconnected graphs

**View modes (configurable via toolbar dropdown):**
1. **Live** — Full graph always visible, updates after every mutation
2. **Query Result** — Graph is blank until a read query returns nodes/edges, then renders only those results
3. **Highlight** — Full graph always visible; query results are highlighted (pulsing border, bright edges) while non-result nodes/edges dim to 30% opacity

Manual node dragging is supported. Positions persist until graph reset.

## Results Panel

**Stacked query results (most recent on top):**

Each result is a collapsible card showing:
- Query text (truncated, expandable)
- Status badge: `ok` (green) or `error` (red)
- Elapsed time and result count
- Two view tabs:
  - **Table** — renders node/edge arrays as a table
  - **JSON** — raw Result object
- Errors display exception message in red

**Controls:**
- "Clear Results" button to wipe result history

## Configuration (Settings Panel)

Accessible via gear icon in toolbar. Opens as a slide-over/dialog with tabs.

### Tab 1: Graph Settings
- View mode selector (Live / Query Result / Highlight)
- Layout algorithm (Dagre / Force-directed)
- Show edge labels toggle
- Show minimap toggle
- Node color scheme (by kind / monochrome)

### Tab 2: Store Configuration
- Memory ceiling slider (64 MB – 1024 MB, default 256 MB)
- Current memory usage progress bar (live)
- Node count / Edge count display
- Auto-checkpoint toggle

### Tab 3: Query Analysis
- Cost threshold slider (frontier limit, default 100,000)
- "Explain before execute" toggle — auto-runs `SYS EXPLAIN` before every read query and shows cost estimate
- Show elapsed time per query toggle

### Stats Bar (always visible, bottom of graph panel)
```
Nodes: 12 | Edges: 8 | Memory: 4.2 KB / 256 MB | Last query: 0.3ms
```

## Example Scripts

Dropdown in toolbar. Loading an example replaces editor content and auto-executes all statements.

### 1. Function Call Graph (basic)
~5 functions with `calls` edges forming a diamond + chain pattern.
Demonstrates: NODE, EDGES FROM, SHORTEST PATH, COMMON NEIGHBORS, TRAVERSE.

### 2. Class Hierarchy (basic)
~8 classes with `extends`, `implements`, `uses` edges.
Demonstrates: ANCESTORS, DESCENDANTS, MATCH multi-hop, INDEGREE filters.

### 3. Code Graph (complex)
~20 nodes: functions, classes, modules with `calls`, `extends`, `imports`, `contains` edges.
Demonstrates: TRAVERSE, SUBGRAPH, MATCH patterns, schema registration, batch operations, SYS EXPLAIN.

### 4. Microservices Map (complex)
Services, databases, queues with `calls_api`, `reads_from`, `publishes_to`, `subscribes_to` edges.
Demonstrates: PATH, DISTANCE, batch, DELETE cascade, cost thresholds.

Each example stored as a TypeScript constant in `playground/src/examples/`. No backend endpoints needed — examples are bundled with the frontend.

## State Management

Use **Zustand** for global state. Single store with these slices:

- `graph`: `{ nodes, edges }` — fetched from `/api/graph` after mutations
- `results`: `ResultEntry[]` — stacked query results
- `config`: `{ viewMode, layout, ceiling, costThreshold, ... }`
- `editor`: `{ content }` — current editor text
- Actions: `executeQuery`, `executeAll`, `resetGraph`, `updateConfig`, `clearResults`

## Development Setup

During development, Vite dev server proxies `/api/*` to the FastAPI backend:

```typescript
// vite.config.ts
server: {
  proxy: {
    '/api': 'http://localhost:7200'
  }
}
```

## Project Structure

```
playground/
├── package.json
├── vite.config.ts
├── tsconfig.json
├── tailwind.config.ts
├── index.html
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── api/
│   │   └── client.ts              # fetch wrappers for /api/*
│   ├── components/
│   │   ├── Toolbar.tsx
│   │   ├── EditorPanel.tsx         # CodeMirror + DSL mode
│   │   ├── GraphPanel.tsx          # React Flow canvas
│   │   ├── ResultsPanel.tsx        # Stacked result cards
│   │   ├── SettingsDialog.tsx      # Config tabs
│   │   ├── StatsBar.tsx
│   │   ├── graph/
│   │   │   ├── CustomNode.tsx      # React Flow custom node
│   │   │   ├── CustomEdge.tsx      # React Flow custom edge
│   │   │   └── layout.ts          # dagre + force layout helpers
│   │   └── ui/                    # shadcn components (auto-generated)
│   ├── lang/
│   │   └── graphstore.ts          # CodeMirror language mode
│   ├── examples/
│   │   ├── index.ts
│   │   ├── function-call-graph.ts
│   │   ├── class-hierarchy.ts
│   │   ├── code-graph.ts
│   │   └── microservices-map.ts
│   ├── hooks/
│   │   ├── useGraphStore.ts       # state: nodes, edges, results, config
│   │   └── useAutoLayout.ts       # layout computation
│   └── lib/
│       └── utils.ts               # shadcn cn() helper
├── components.json                # shadcn config
└── public/

graphstore/
├── cli.py                         # CLI entry point: `graphstore playground`
└── server.py                      # FastAPI app + static file serving
```

## Dependencies

### Frontend (playground/package.json)
- vite, react, react-dom, typescript
- tailwindcss, @tailwindcss/vite
- shadcn/ui components
- lucide-react
- @xyflow/react (React Flow v12)
- @uiw/react-codemirror + @codemirror/language
- dagre

### Backend (added to graphstore pyproject.toml)
- fastapi
- uvicorn

## Out of Scope (for v1)
- Authentication / multi-user
- Persistence across playground sessions (in-memory only)
- Export/import graph data
- Custom themes beyond dark
- Collaborative editing
