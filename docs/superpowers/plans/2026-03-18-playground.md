# Graphstore Playground Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a browser-based interactive playground where users write graphstore DSL queries, execute them, see results in table/JSON format, and visualize the graph with React Flow.

**Architecture:** FastAPI backend wraps GraphStore and exposes REST API. Vite+React frontend provides three-panel UI (editor, graph, results). Single `graphstore playground` CLI command starts both. Zustand manages frontend state.

**Tech Stack:** Python (FastAPI, uvicorn), React 18, TypeScript, Vite, Tailwind CSS, shadcn/ui, Lucide icons, React Flow v12, CodeMirror 6, dagre, Zustand

**Spec:** `docs/superpowers/specs/2026-03-18-playground-design.md`

---

## File Map

### Backend (Python)

| File | Responsibility |
|---|---|
| `graphstore/store.py` | Add `get_all_edges()` method |
| `graphstore/server.py` | FastAPI app: API endpoints + static file serving |
| `graphstore/cli.py` | CLI entry point: `graphstore playground [--port]` |
| `pyproject.toml` | Add fastapi, uvicorn deps + console_scripts entry |
| `tests/test_server.py` | API endpoint tests |

### Frontend (playground/)

| File | Responsibility |
|---|---|
| `playground/package.json` | Dependencies |
| `playground/vite.config.ts` | Vite config with API proxy |
| `playground/tailwind.config.ts` | Tailwind config |
| `playground/tsconfig.json` | TypeScript config |
| `playground/index.html` | HTML entry point |
| `playground/src/main.tsx` | React root mount |
| `playground/src/App.tsx` | Three-panel layout shell |
| `playground/src/api/client.ts` | Fetch wrappers for `/api/*` |
| `playground/src/hooks/useGraphStore.ts` | Zustand store (graph, results, config, editor) |
| `playground/src/hooks/useAutoLayout.ts` | Dagre layout computation |
| `playground/src/lang/graphstore.ts` | CodeMirror DSL language mode |
| `playground/src/components/Toolbar.tsx` | Top toolbar with all actions |
| `playground/src/components/EditorPanel.tsx` | CodeMirror editor |
| `playground/src/components/GraphPanel.tsx` | React Flow canvas |
| `playground/src/components/ResultsPanel.tsx` | Stacked result cards |
| `playground/src/components/SettingsDialog.tsx` | Config dialog with tabs |
| `playground/src/components/StatsBar.tsx` | Stats bar |
| `playground/src/components/graph/CustomNode.tsx` | React Flow custom node |
| `playground/src/components/graph/CustomEdge.tsx` | React Flow custom edge |
| `playground/src/components/graph/layout.ts` | Dagre layout helpers |
| `playground/src/examples/index.ts` | Example registry |
| `playground/src/examples/function-call-graph.ts` | Example 1 |
| `playground/src/examples/class-hierarchy.ts` | Example 2 |
| `playground/src/examples/code-graph.ts` | Example 3 |
| `playground/src/examples/microservices-map.ts` | Example 4 |
| `playground/src/lib/utils.ts` | shadcn `cn()` helper |

---

### Task 1: Backend — CoreStore.get_all_edges() + FastAPI server

**Files:**
- Modify: `graphstore/store.py` — add `get_all_edges()` method
- Create: `graphstore/server.py` — FastAPI app
- Create: `graphstore/cli.py` — CLI entry point
- Modify: `pyproject.toml` — add deps + console_scripts
- Create: `tests/test_server.py` — API tests

- [ ] **Step 1: Add `get_all_edges()` to CoreStore**

In `graphstore/store.py`, add after `get_all_nodes()`:

```python
def get_all_edges(self) -> list[dict]:
    """Get all edges across all types."""
    result = []
    for etype, edge_list in self._edges_by_type.items():
        for src_slot, tgt_slot, data in edge_list:
            src_id = self._slot_to_id(src_slot)
            tgt_id = self._slot_to_id(tgt_slot)
            if src_id and tgt_id:
                result.append({"source": src_id, "target": tgt_id, "kind": etype, **data})
    return result
```

- [ ] **Step 2: Create `graphstore/server.py`**

```python
"""FastAPI server for graphstore playground."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from graphstore import GraphStore
from graphstore.errors import GraphStoreError, NodeNotFound, NodeExists, QueryError
from graphstore.memory import estimate as estimate_memory

app = FastAPI(title="Graphstore Playground")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global store instance — created at startup
_store: GraphStore | None = None
_cost_threshold: int = 100_000


def get_store() -> GraphStore:
    global _store
    if _store is None:
        _store = GraphStore()
    return _store


class ExecuteRequest(BaseModel):
    query: str


class BatchRequest(BaseModel):
    queries: list[str]


class ConfigRequest(BaseModel):
    ceiling_mb: int | None = None
    cost_threshold: int | None = None


def _result_to_dict(result) -> dict:
    """Convert a Result to a JSON-safe dict."""
    data = result.data
    # numpy arrays and other non-serializable types
    if hasattr(data, 'tolist'):
        data = data.tolist()
    return {
        "kind": result.kind,
        "data": data,
        "count": result.count,
        "elapsed_us": result.elapsed_us,
    }


@app.post("/api/execute")
async def execute(req: ExecuteRequest):
    store = get_store()
    try:
        result = store.execute(req.query)
        return _result_to_dict(result)
    except (QueryError, NodeNotFound, NodeExists) as e:
        raise HTTPException(status_code=400, detail={
            "error": str(e),
            "type": type(e).__name__,
        })
    except GraphStoreError as e:
        raise HTTPException(status_code=400, detail={
            "error": str(e),
            "type": type(e).__name__,
        })


@app.post("/api/execute-batch")
async def execute_batch(req: BatchRequest):
    store = get_store()
    results = []
    for query in req.queries:
        try:
            result = store.execute(query)
            results.append(_result_to_dict(result))
        except GraphStoreError as e:
            results.append({
                "kind": "error",
                "data": str(e),
                "count": 0,
                "elapsed_us": 0,
            })
    return results


@app.get("/api/graph")
async def get_graph():
    store = get_store()
    nodes = store._store.get_all_nodes()
    edges = store._store.get_all_edges()
    return {"nodes": nodes, "edges": edges}


@app.post("/api/reset")
async def reset():
    global _store
    _store = GraphStore()
    return {"ok": True}


@app.post("/api/config")
async def update_config(req: ConfigRequest):
    store = get_store()
    global _cost_threshold
    if req.ceiling_mb is not None:
        store._store._ceiling_bytes = req.ceiling_mb * 1_000_000
    if req.cost_threshold is not None:
        _cost_threshold = req.cost_threshold
    return {"ok": True}


def mount_static(app: FastAPI, static_dir: Path):
    """Mount built frontend if it exists."""
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
```

- [ ] **Step 3: Create `graphstore/cli.py`**

```python
"""CLI entry point for graphstore playground."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Graphstore CLI")
    subparsers = parser.add_subparsers(dest="command")

    pg = subparsers.add_parser("playground", help="Launch interactive playground")
    pg.add_argument("--port", type=int, default=7200, help="Server port (default: 7200)")
    pg.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    pg.add_argument("--no-browser", action="store_true", help="Don't open browser")

    args = parser.parse_args()

    if args.command == "playground":
        run_playground(args)
    else:
        parser.print_help()


def run_playground(args):
    import uvicorn
    from graphstore.server import app, mount_static

    # Mount static files from playground/dist if built
    static_dir = Path(__file__).parent.parent / "playground" / "dist"
    if not static_dir.exists():
        # Try installed package location
        static_dir = Path(__file__).parent / "playground_dist"
    mount_static(app, static_dir)

    if not args.no_browser:
        import webbrowser
        import threading
        def open_browser():
            import time
            time.sleep(1)
            webbrowser.open(f"http://{args.host}:{args.port}")
        threading.Thread(target=open_browser, daemon=True).start()

    print(f"Graphstore Playground running at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Update `pyproject.toml`**

Add to `[project]` dependencies:
```toml
[project.optional-dependencies]
playground = [
    "fastapi>=0.100",
    "uvicorn>=0.20",
]
```

Add console script entry:
```toml
[project.scripts]
graphstore = "graphstore.cli:main"
```

- [ ] **Step 5: Write API tests**

Create `tests/test_server.py`:

```python
import pytest
from fastapi.testclient import TestClient
from graphstore.server import app, get_store

@pytest.fixture(autouse=True)
def reset_store():
    """Reset store before each test."""
    import graphstore.server as srv
    srv._store = None
    yield
    srv._store = None

client = TestClient(app)

def test_execute_create_node():
    r = client.post("/api/execute", json={"query": 'CREATE NODE "a" kind = "x" name = "alpha"'})
    assert r.status_code == 200
    data = r.json()
    assert data["kind"] == "ok"

def test_execute_query_node():
    client.post("/api/execute", json={"query": 'CREATE NODE "a" kind = "x" name = "alpha"'})
    r = client.post("/api/execute", json={"query": 'NODE "a"'})
    assert r.status_code == 200
    assert r.json()["data"]["name"] == "alpha"

def test_execute_invalid_query():
    r = client.post("/api/execute", json={"query": "INVALID QUERY"})
    assert r.status_code == 400

def test_execute_batch():
    r = client.post("/api/execute-batch", json={
        "queries": [
            'CREATE NODE "a" kind = "x" name = "alpha"',
            'CREATE NODE "b" kind = "x" name = "beta"',
        ]
    })
    assert r.status_code == 200
    results = r.json()
    assert len(results) == 2
    assert all(res["kind"] == "ok" for res in results)

def test_get_graph():
    client.post("/api/execute", json={"query": 'CREATE NODE "a" kind = "x" name = "alpha"'})
    client.post("/api/execute", json={"query": 'CREATE NODE "b" kind = "x" name = "beta"'})
    client.post("/api/execute", json={"query": 'CREATE EDGE "a" -> "b" kind = "link"'})
    r = client.get("/api/graph")
    assert r.status_code == 200
    data = r.json()
    assert len(data["nodes"]) == 2
    assert len(data["edges"]) == 1

def test_reset():
    client.post("/api/execute", json={"query": 'CREATE NODE "a" kind = "x" name = "alpha"'})
    client.post("/api/reset")
    r = client.get("/api/graph")
    assert r.json()["nodes"] == []

def test_config():
    r = client.post("/api/config", json={"ceiling_mb": 512})
    assert r.status_code == 200
    assert r.json()["ok"]

def test_get_all_edges():
    """Test CoreStore.get_all_edges()."""
    from graphstore.store import CoreStore
    store = CoreStore()
    store.put_node("a", "x", {"name": "alpha"})
    store.put_node("b", "x", {"name": "beta"})
    store.put_edge("a", "b", "link")
    edges = store.get_all_edges()
    assert len(edges) == 1
    assert edges[0]["source"] == "a"
    assert edges[0]["target"] == "b"
    assert edges[0]["kind"] == "link"
```

- [ ] **Step 6: Install and verify**

```bash
source .venv/bin/activate
pip install -e ".[dev,playground]"
python -m pytest tests/test_server.py -v
```

- [ ] **Step 7: Commit**

```bash
git add graphstore/store.py graphstore/server.py graphstore/cli.py pyproject.toml tests/test_server.py
git commit -m "feat: add FastAPI server and CLI for playground"
```

---

### Task 2: Frontend scaffold — Vite + React + Tailwind + shadcn

**Files:**
- Create: `playground/` directory with full Vite scaffold
- Install all dependencies

- [ ] **Step 1: Scaffold Vite project**

```bash
cd /home/kai/code/orkait/orkait-tinygraph
npm create vite@latest playground -- --template react-ts
cd playground
```

- [ ] **Step 2: Install dependencies**

```bash
cd playground
npm install @xyflow/react @uiw/react-codemirror @codemirror/language @codemirror/lang-sql @lezer/highlight @lezer/lr zustand dagre lucide-react clsx tailwind-merge
npm install -D tailwindcss @tailwindcss/vite @types/dagre
```

- [ ] **Step 3: Configure Tailwind**

Replace `playground/src/index.css`:
```css
@import "tailwindcss";
```

Update `playground/vite.config.ts`:
```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    proxy: {
      '/api': 'http://localhost:7200',
    },
  },
})
```

- [ ] **Step 4: Initialize shadcn**

```bash
cd playground
npx shadcn@latest init -d
```

Then install needed components:
```bash
npx shadcn@latest add button badge tabs dialog dropdown-menu separator resizable slider switch label select tooltip scroll-area
```

- [ ] **Step 5: Create `playground/src/lib/utils.ts`**

This should already exist from shadcn init. Verify it has:
```typescript
import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
```

- [ ] **Step 6: Create minimal `App.tsx` to verify setup**

```typescript
export default function App() {
  return (
    <div className="h-screen bg-zinc-950 text-zinc-100 flex items-center justify-center">
      <h1 className="text-2xl font-bold">Graphstore Playground</h1>
    </div>
  )
}
```

- [ ] **Step 7: Verify dev server starts**

```bash
cd playground && npm run dev
```

Open browser, verify dark page with "Graphstore Playground" text.

- [ ] **Step 8: Commit**

```bash
git add playground/
git commit -m "feat: scaffold playground frontend with Vite, Tailwind, shadcn"
```

---

### Task 3: API client + Zustand store

**Files:**
- Create: `playground/src/api/client.ts`
- Create: `playground/src/hooks/useGraphStore.ts`

- [ ] **Step 1: Create API client**

`playground/src/api/client.ts`:
```typescript
const BASE = '/api'

export interface Result {
  kind: string
  data: any
  count: number
  elapsed_us: number
}

export interface GraphData {
  nodes: Record<string, any>[]
  edges: Record<string, any>[]
}

export interface ApiError {
  error: string
  type: string
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: { error: res.statusText, type: 'Unknown' } }))
    throw err.detail || { error: res.statusText, type: 'HttpError' }
  }
  return res.json()
}

export const api = {
  execute: (query: string) =>
    request<Result>('/execute', {
      method: 'POST',
      body: JSON.stringify({ query }),
    }),

  executeBatch: (queries: string[]) =>
    request<Result[]>('/execute-batch', {
      method: 'POST',
      body: JSON.stringify({ queries }),
    }),

  getGraph: () => request<GraphData>('/graph'),

  reset: () =>
    request<{ ok: boolean }>('/reset', { method: 'POST' }),

  updateConfig: (config: { ceiling_mb?: number; cost_threshold?: number }) =>
    request<{ ok: boolean }>('/config', {
      method: 'POST',
      body: JSON.stringify(config),
    }),
}
```

- [ ] **Step 2: Create Zustand store**

`playground/src/hooks/useGraphStore.ts`:
```typescript
import { create } from 'zustand'
import { api, type Result, type GraphData } from '@/api/client'

export type ViewMode = 'live' | 'query-result' | 'highlight'
export type LayoutMode = 'dagre' | 'force'

export interface ResultEntry {
  id: string
  query: string
  result: Result | null
  error: string | null
  timestamp: number
}

interface Config {
  viewMode: ViewMode
  layoutMode: LayoutMode
  showEdgeLabels: boolean
  showMinimap: boolean
  colorByKind: boolean
  ceilingMb: number
  costThreshold: number
  explainBeforeExecute: boolean
  showElapsed: boolean
}

interface GraphStoreState {
  // Graph data
  graph: GraphData
  // Results
  results: ResultEntry[]
  // Config
  config: Config
  // Editor
  editorContent: string
  // Highlighted nodes/edges from last query
  highlightedNodeIds: Set<string>
  highlightedEdges: Set<string>
  // Loading
  loading: boolean

  // Actions
  setEditorContent: (content: string) => void
  executeQuery: (query: string) => Promise<void>
  executeAll: () => Promise<void>
  refreshGraph: () => Promise<void>
  resetGraph: () => Promise<void>
  clearResults: () => void
  updateConfig: (partial: Partial<Config>) => void
}

let resultCounter = 0

function splitQueries(text: string): string[] {
  const lines = text.split('\n')
  const queries: string[] = []
  let batch: string[] = []
  let inBatch = false

  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('//')) continue

    if (trimmed === 'BEGIN') {
      inBatch = true
      batch = [line]
    } else if (trimmed === 'COMMIT' && inBatch) {
      batch.push(line)
      queries.push(batch.join('\n'))
      batch = []
      inBatch = false
    } else if (inBatch) {
      batch.push(line)
    } else {
      queries.push(trimmed)
    }
  }
  return queries
}

function extractHighlights(result: Result): { nodeIds: Set<string>; edgeKeys: Set<string> } {
  const nodeIds = new Set<string>()
  const edgeKeys = new Set<string>()

  if (!result.data) return { nodeIds, edgeKeys }

  switch (result.kind) {
    case 'node':
      if (result.data?.id) nodeIds.add(result.data.id)
      break
    case 'nodes':
      if (Array.isArray(result.data)) {
        result.data.forEach((n: any) => n?.id && nodeIds.add(n.id))
      }
      break
    case 'edges':
      if (Array.isArray(result.data)) {
        result.data.forEach((e: any) => {
          if (e.source) nodeIds.add(e.source)
          if (e.target) nodeIds.add(e.target)
          edgeKeys.add(`${e.source}->${e.target}`)
        })
      }
      break
    case 'path':
      if (Array.isArray(result.data)) {
        result.data.forEach((id: string) => nodeIds.add(id))
        for (let i = 0; i < result.data.length - 1; i++) {
          edgeKeys.add(`${result.data[i]}->${result.data[i + 1]}`)
        }
      }
      break
    case 'paths':
      if (Array.isArray(result.data)) {
        result.data.forEach((path: string[]) => {
          path.forEach((id: string) => nodeIds.add(id))
          for (let i = 0; i < path.length - 1; i++) {
            edgeKeys.add(`${path[i]}->${path[i + 1]}`)
          }
        })
      }
      break
    case 'subgraph':
      if (result.data?.nodes) {
        result.data.nodes.forEach((n: any) => n?.id && nodeIds.add(n.id))
      }
      if (result.data?.edges) {
        result.data.edges.forEach((e: any) => edgeKeys.add(`${e.source}->${e.target}`))
      }
      break
    case 'match':
      if (Array.isArray(result.data)) {
        result.data.forEach((binding: Record<string, string>) => {
          Object.values(binding).forEach(id => nodeIds.add(id))
        })
      }
      break
  }
  return { nodeIds, edgeKeys }
}

export const useGraphStore = create<GraphStoreState>((set, get) => ({
  graph: { nodes: [], edges: [] },
  results: [],
  config: {
    viewMode: 'live',
    layoutMode: 'dagre',
    showEdgeLabels: true,
    showMinimap: true,
    colorByKind: true,
    ceilingMb: 256,
    costThreshold: 100_000,
    explainBeforeExecute: false,
    showElapsed: true,
  },
  editorContent: '',
  highlightedNodeIds: new Set(),
  highlightedEdges: new Set(),
  loading: false,

  setEditorContent: (content) => set({ editorContent: content }),

  executeQuery: async (query) => {
    set({ loading: true })
    const id = `r-${++resultCounter}`
    try {
      const result = await api.execute(query)
      const { nodeIds, edgeKeys } = extractHighlights(result)
      set((s) => ({
        results: [{ id, query, result, error: null, timestamp: Date.now() }, ...s.results],
        highlightedNodeIds: nodeIds,
        highlightedEdges: edgeKeys,
        loading: false,
      }))
      // Refresh graph if it was a write
      const writeKinds = ['ok']
      if (writeKinds.includes(result.kind)) {
        await get().refreshGraph()
      }
    } catch (err: any) {
      set((s) => ({
        results: [{ id, query, result: null, error: err?.error || String(err), timestamp: Date.now() }, ...s.results],
        loading: false,
      }))
    }
  },

  executeAll: async () => {
    const queries = splitQueries(get().editorContent)
    for (const q of queries) {
      await get().executeQuery(q)
    }
  },

  refreshGraph: async () => {
    try {
      const graph = await api.getGraph()
      set({ graph })
    } catch {
      // ignore
    }
  },

  resetGraph: async () => {
    await api.reset()
    set({
      graph: { nodes: [], edges: [] },
      results: [],
      highlightedNodeIds: new Set(),
      highlightedEdges: new Set(),
    })
  },

  clearResults: () => set({ results: [], highlightedNodeIds: new Set(), highlightedEdges: new Set() }),

  updateConfig: (partial) => set((s) => ({
    config: { ...s.config, ...partial },
  })),
}))
```

- [ ] **Step 3: Commit**

```bash
git add playground/src/api/ playground/src/hooks/
git commit -m "feat: add API client and Zustand store"
```

---

### Task 4: DSL syntax highlighting (CodeMirror language mode)

**Files:**
- Create: `playground/src/lang/graphstore.ts`

- [ ] **Step 1: Create DSL language mode**

`playground/src/lang/graphstore.ts`:
```typescript
import { StreamLanguage } from '@codemirror/language'

const KEYWORDS = new Set([
  'CREATE', 'NODE', 'NODES', 'EDGE', 'EDGES', 'WHERE', 'FROM', 'TO',
  'TRAVERSE', 'MATCH', 'DELETE', 'UPDATE', 'UPSERT', 'INCREMENT',
  'PATH', 'PATHS', 'SHORTEST', 'DISTANCE', 'ANCESTORS', 'DESCENDANTS',
  'COMMON', 'NEIGHBORS', 'SUBGRAPH', 'DEPTH', 'MAX_DEPTH', 'LIMIT',
  'SET', 'BY', 'OF', 'AND', 'OR', 'NOT', 'BEGIN', 'COMMIT',
  'KIND', 'REGISTER', 'UNREGISTER', 'DESCRIBE', 'STATS', 'EXPLAIN',
  'CHECKPOINT', 'REBUILD', 'INDICES', 'CLEAR', 'WAL', 'REQUIRED',
  'OPTIONAL', 'SINCE', 'SLOW', 'FREQUENT', 'FAILED', 'KINDS',
  'STATUS', 'REPLAY', 'LOG', 'CACHE', 'NULL', 'INDEGREE', 'OUTDEGREE',
])

const SYS_KEYWORD = 'SYS'

export const graphstoreLang = StreamLanguage.define({
  token(stream) {
    // Skip whitespace
    if (stream.eatSpace()) return null

    // Comments
    if (stream.match('//')) {
      stream.skipToEnd()
      return 'comment'
    }

    // Strings
    if (stream.match('"')) {
      while (!stream.eol()) {
        const ch = stream.next()
        if (ch === '\\') { stream.next(); continue }
        if (ch === '"') break
      }
      return 'string'
    }

    // Numbers
    if (stream.match(/-?[0-9]+(\.[0-9]+)?/)) return 'number'

    // Arrow operators
    if (stream.match('->')) return 'operator'
    if (stream.match('-[')) return 'operator'
    if (stream.match(']->')) return 'operator'

    // Comparison operators
    if (stream.match('!=') || stream.match('>=') || stream.match('<=')) return 'operator'
    if (stream.match(/[=><]/)) return 'operator'

    // Words
    if (stream.match(/[a-zA-Z_][a-zA-Z0-9_]*/)) {
      const word = stream.current()
      if (word === SYS_KEYWORD) return 'keyword2'
      if (KEYWORDS.has(word)) return 'keyword'
      return 'variableName'
    }

    stream.next()
    return null
  },
})
```

- [ ] **Step 2: Commit**

```bash
git add playground/src/lang/
git commit -m "feat: add CodeMirror DSL syntax highlighting"
```

---

### Task 5: Three-panel layout + Editor + Results

**Files:**
- Create: `playground/src/components/EditorPanel.tsx`
- Create: `playground/src/components/ResultsPanel.tsx`
- Update: `playground/src/App.tsx`

- [ ] **Step 1: Create EditorPanel**

`playground/src/components/EditorPanel.tsx`:
```typescript
import CodeMirror from '@uiw/react-codemirror'
import { graphstoreLang } from '@/lang/graphstore'
import { useGraphStore } from '@/hooks/useGraphStore'
import { keymap } from '@codemirror/view'
import { useCallback, useMemo } from 'react'

export function EditorPanel() {
  const { editorContent, setEditorContent, executeQuery, executeAll } = useGraphStore()

  const handleRunSelected = useCallback((view: any) => {
    const selection = view.state.sliceDoc(
      view.state.selection.main.from,
      view.state.selection.main.to
    )
    const text = selection || view.state.doc.lineAt(view.state.selection.main.head).text
    if (text.trim()) executeQuery(text.trim())
    return true
  }, [executeQuery])

  const extensions = useMemo(() => [
    graphstoreLang,
    keymap.of([
      { key: 'Ctrl-Enter', run: handleRunSelected },
      { key: 'Ctrl-Shift-Enter', run: () => { executeAll(); return true } },
    ]),
  ], [handleRunSelected, executeAll])

  return (
    <div className="h-full flex flex-col bg-zinc-950">
      <CodeMirror
        value={editorContent}
        onChange={setEditorContent}
        extensions={extensions}
        theme="dark"
        className="flex-1 overflow-auto text-sm"
        basicSetup={{
          lineNumbers: true,
          foldGutter: false,
          highlightActiveLine: true,
          autocompletion: false,
        }}
      />
    </div>
  )
}
```

- [ ] **Step 2: Create ResultsPanel**

`playground/src/components/ResultsPanel.tsx`:
```typescript
import { useGraphStore } from '@/hooks/useGraphStore'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Trash2, ChevronDown, ChevronRight } from 'lucide-react'
import { useState } from 'react'

function ResultCard({ entry }: { entry: any }) {
  const [expanded, setExpanded] = useState(false)
  const isError = entry.error != null
  const result = entry.result

  const renderTable = () => {
    if (!result?.data) return <span className="text-zinc-500">No data</span>
    if (Array.isArray(result.data)) {
      if (result.data.length === 0) return <span className="text-zinc-500">Empty result</span>
      const keys = Object.keys(result.data[0])
      return (
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-zinc-800">
              {keys.map(k => <th key={k} className="text-left p-1 text-zinc-400">{k}</th>)}
            </tr>
          </thead>
          <tbody>
            {result.data.map((row: any, i: number) => (
              <tr key={i} className="border-b border-zinc-800/50">
                {keys.map(k => <td key={k} className="p-1">{JSON.stringify(row[k])}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      )
    }
    if (typeof result.data === 'object') {
      const keys = Object.keys(result.data)
      return (
        <table className="w-full text-xs">
          <tbody>
            {keys.map(k => (
              <tr key={k} className="border-b border-zinc-800/50">
                <td className="p-1 text-zinc-400 w-32">{k}</td>
                <td className="p-1">{JSON.stringify(result.data[k])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )
    }
    return <span className="text-sm">{JSON.stringify(result.data)}</span>
  }

  return (
    <div className="border border-zinc-800 rounded-md mb-2 bg-zinc-900/50">
      <div
        className="flex items-center gap-2 p-2 cursor-pointer hover:bg-zinc-800/50"
        onClick={() => setExpanded(!expanded)}
      >
        {expanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
        <code className="text-xs text-zinc-400 truncate flex-1">{entry.query}</code>
        <Badge variant={isError ? 'destructive' : 'default'} className="text-[10px]">
          {isError ? 'error' : result?.kind || 'ok'}
        </Badge>
        {result && (
          <span className="text-[10px] text-zinc-500">
            {result.count} results &middot; {(result.elapsed_us / 1000).toFixed(1)}ms
          </span>
        )}
      </div>
      {expanded && (
        <div className="p-2 border-t border-zinc-800">
          {isError ? (
            <div className="text-red-400 text-xs font-mono">{entry.error}</div>
          ) : (
            <Tabs defaultValue="table" className="w-full">
              <TabsList className="h-7">
                <TabsTrigger value="table" className="text-xs h-6">Table</TabsTrigger>
                <TabsTrigger value="json" className="text-xs h-6">JSON</TabsTrigger>
              </TabsList>
              <TabsContent value="table" className="mt-2">{renderTable()}</TabsContent>
              <TabsContent value="json" className="mt-2">
                <pre className="text-xs text-zinc-300 whitespace-pre-wrap overflow-auto max-h-48">
                  {JSON.stringify(result, null, 2)}
                </pre>
              </TabsContent>
            </Tabs>
          )}
        </div>
      )}
    </div>
  )
}

export function ResultsPanel() {
  const { results, clearResults } = useGraphStore()

  return (
    <div className="h-full flex flex-col bg-zinc-950">
      <div className="flex items-center justify-between px-2 py-1 border-b border-zinc-800">
        <span className="text-xs text-zinc-400">Results ({results.length})</span>
        <Button variant="ghost" size="sm" onClick={clearResults} className="h-6 px-2">
          <Trash2 className="w-3 h-3" />
        </Button>
      </div>
      <ScrollArea className="flex-1 p-2">
        {results.map(entry => <ResultCard key={entry.id} entry={entry} />)}
        {results.length === 0 && (
          <div className="text-zinc-600 text-xs text-center mt-8">
            Run a query to see results
          </div>
        )}
      </ScrollArea>
    </div>
  )
}
```

- [ ] **Step 3: Update App.tsx with three-panel layout**

`playground/src/App.tsx`:
```typescript
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from '@/components/ui/resizable'
import { EditorPanel } from '@/components/EditorPanel'
import { ResultsPanel } from '@/components/ResultsPanel'
import { GraphPanel } from '@/components/GraphPanel'
import { Toolbar } from '@/components/Toolbar'
import { StatsBar } from '@/components/StatsBar'

export default function App() {
  return (
    <div className="h-screen flex flex-col bg-zinc-950 text-zinc-100">
      <Toolbar />
      <ResizablePanelGroup direction="horizontal" className="flex-1">
        <ResizablePanel defaultSize={40} minSize={20}>
          <ResizablePanelGroup direction="vertical">
            <ResizablePanel defaultSize={65} minSize={20}>
              <EditorPanel />
            </ResizablePanel>
            <ResizableHandle withHandle />
            <ResizablePanel defaultSize={35} minSize={10}>
              <ResultsPanel />
            </ResizablePanel>
          </ResizablePanelGroup>
        </ResizablePanel>
        <ResizableHandle withHandle />
        <ResizablePanel defaultSize={60} minSize={30}>
          <div className="h-full flex flex-col">
            <div className="flex-1">
              <GraphPanel />
            </div>
            <StatsBar />
          </div>
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  )
}
```

- [ ] **Step 4: Create placeholder components**

Create stubs for `GraphPanel.tsx`, `Toolbar.tsx`, `StatsBar.tsx` so App compiles:

`playground/src/components/Toolbar.tsx`:
```typescript
export function Toolbar() {
  return <div className="h-10 border-b border-zinc-800 px-2 flex items-center text-xs text-zinc-400">Toolbar (placeholder)</div>
}
```

`playground/src/components/GraphPanel.tsx`:
```typescript
export function GraphPanel() {
  return <div className="h-full flex items-center justify-center text-zinc-600 text-sm">Graph (placeholder)</div>
}
```

`playground/src/components/StatsBar.tsx`:
```typescript
export function StatsBar() {
  return <div className="h-7 border-t border-zinc-800 px-3 flex items-center text-[10px] text-zinc-500">Stats (placeholder)</div>
}
```

- [ ] **Step 5: Verify three-panel layout renders**

```bash
cd playground && npm run dev
```

Verify: dark page with toolbar top, editor top-left, results bottom-left, graph right, stats bar bottom-right. Panels resizable.

- [ ] **Step 6: Commit**

```bash
git add playground/src/
git commit -m "feat: three-panel layout with editor and results panels"
```

---

### Task 6: Graph panel — React Flow + custom nodes/edges + dagre layout

**Files:**
- Create: `playground/src/components/graph/CustomNode.tsx`
- Create: `playground/src/components/graph/CustomEdge.tsx`
- Create: `playground/src/components/graph/layout.ts`
- Create: `playground/src/hooks/useAutoLayout.ts`
- Update: `playground/src/components/GraphPanel.tsx`

**Skill reference:** @reactflow for React Flow v12 patterns

- [ ] **Step 1: Create dagre layout helper**

`playground/src/components/graph/layout.ts`:
```typescript
import dagre from 'dagre'
import type { Node, Edge } from '@xyflow/react'

const NODE_WIDTH = 180
const NODE_HEIGHT = 60

export function applyDagreLayout(nodes: Node[], edges: Edge[], direction: 'TB' | 'LR' = 'TB'): Node[] {
  const g = new dagre.graphlib.Graph()
  g.setDefaultEdgeLabel(() => ({}))
  g.setGraph({ rankdir: direction, nodesep: 50, ranksep: 80 })

  nodes.forEach((node) => {
    g.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT })
  })

  edges.forEach((edge) => {
    g.setEdge(edge.source, edge.target)
  })

  dagre.layout(g)

  return nodes.map((node) => {
    const pos = g.node(node.id)
    return {
      ...node,
      position: { x: pos.x - NODE_WIDTH / 2, y: pos.y - NODE_HEIGHT / 2 },
    }
  })
}
```

- [ ] **Step 2: Create CustomNode**

`playground/src/components/graph/CustomNode.tsx`:
```typescript
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { Badge } from '@/components/ui/badge'

const KIND_COLORS: Record<string, string> = {
  function: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
  class: 'bg-green-500/20 text-green-300 border-green-500/30',
  module: 'bg-purple-500/20 text-purple-300 border-purple-500/30',
  service: 'bg-orange-500/20 text-orange-300 border-orange-500/30',
  database: 'bg-red-500/20 text-red-300 border-red-500/30',
  queue: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30',
}

function getKindColor(kind: string): string {
  if (KIND_COLORS[kind]) return KIND_COLORS[kind]
  // Deterministic color from hash
  const colors = Object.values(KIND_COLORS)
  let hash = 0
  for (let i = 0; i < kind.length; i++) hash = kind.charCodeAt(i) + ((hash << 5) - hash)
  return colors[Math.abs(hash) % colors.length]
}

export function CustomNode({ data }: NodeProps) {
  const highlighted = data.highlighted as boolean
  const dimmed = data.dimmed as boolean

  return (
    <div
      className={`
        px-3 py-2 rounded-lg border bg-zinc-900 min-w-[140px]
        ${highlighted ? 'border-blue-400 ring-2 ring-blue-400/50' : 'border-zinc-700'}
        ${dimmed ? 'opacity-30' : 'opacity-100'}
        transition-all duration-300
      `}
    >
      <Handle type="target" position={Position.Top} className="!w-2 !h-2 !bg-zinc-600" />
      <div className="text-xs font-bold text-zinc-100 truncate">{data.label as string}</div>
      <Badge className={`text-[9px] mt-1 ${getKindColor(data.kind as string)}`}>
        {data.kind as string}
      </Badge>
      <Handle type="source" position={Position.Bottom} className="!w-2 !h-2 !bg-zinc-600" />
    </div>
  )
}
```

- [ ] **Step 3: Create CustomEdge**

`playground/src/components/graph/CustomEdge.tsx`:
```typescript
import { BaseEdge, getStraightPath, type EdgeProps } from '@xyflow/react'

const STROKE_STYLES = ['', '5,5', '2,2']

export function CustomEdge(props: EdgeProps) {
  const { sourceX, sourceY, targetX, targetY, data, style } = props
  const [edgePath] = getStraightPath({ sourceX, sourceY, targetX, targetY })
  const highlighted = data?.highlighted as boolean
  const dimmed = data?.dimmed as boolean

  return (
    <>
      <BaseEdge
        path={edgePath}
        style={{
          ...style,
          stroke: highlighted ? '#60a5fa' : dimmed ? '#27272a' : '#52525b',
          strokeWidth: highlighted ? 2 : 1,
          strokeDasharray: STROKE_STYLES[(data?.styleIndex as number) || 0],
          opacity: dimmed ? 0.3 : 1,
          transition: 'all 0.3s',
        }}
      />
      {data?.showLabel && (
        <text
          x={(sourceX + targetX) / 2}
          y={(sourceY + targetY) / 2 - 8}
          className="fill-zinc-500 text-[9px]"
          textAnchor="middle"
        >
          {data.kind as string}
        </text>
      )}
    </>
  )
}
```

- [ ] **Step 4: Create useAutoLayout hook**

`playground/src/hooks/useAutoLayout.ts`:
```typescript
import { useMemo } from 'react'
import type { Node, Edge } from '@xyflow/react'
import { applyDagreLayout } from '@/components/graph/layout'
import { useGraphStore, type ViewMode } from '@/hooks/useGraphStore'
import type { GraphData } from '@/api/client'

const EDGE_STYLE_MAP: Record<string, number> = {}
let styleCounter = 0

function getEdgeStyleIndex(kind: string): number {
  if (!(kind in EDGE_STYLE_MAP)) {
    EDGE_STYLE_MAP[kind] = styleCounter++ % 3
  }
  return EDGE_STYLE_MAP[kind]
}

export function useAutoLayout(): { nodes: Node[]; edges: Edge[] } {
  const { graph, config, highlightedNodeIds, highlightedEdges } = useGraphStore()
  const { viewMode, showEdgeLabels } = config

  return useMemo(() => {
    if (!graph.nodes.length) return { nodes: [], edges: [] }

    const rfNodes: Node[] = graph.nodes.map((n) => {
      const isHighlighted = highlightedNodeIds.has(n.id)
      const isDimmed = viewMode === 'highlight' && highlightedNodeIds.size > 0 && !isHighlighted

      return {
        id: n.id,
        type: 'custom',
        position: { x: 0, y: 0 },
        data: {
          label: n.id,
          kind: n.kind,
          highlighted: isHighlighted,
          dimmed: isDimmed,
          fields: n,
        },
      }
    })

    const rfEdges: Edge[] = graph.edges.map((e, i) => {
      const key = `${e.source}->${e.target}`
      const isHighlighted = highlightedEdges.has(key)
      const isDimmed = viewMode === 'highlight' && highlightedEdges.size > 0 && !isHighlighted

      return {
        id: `e-${i}-${e.source}-${e.target}-${e.kind}`,
        source: e.source,
        target: e.target,
        type: 'custom',
        data: {
          kind: e.kind,
          highlighted: isHighlighted,
          dimmed: isDimmed,
          showLabel: showEdgeLabels,
          styleIndex: getEdgeStyleIndex(e.kind),
        },
      }
    })

    // Filter for query-result mode
    if (viewMode === 'query-result' && highlightedNodeIds.size > 0) {
      const filteredNodes = rfNodes.filter(n => highlightedNodeIds.has(n.id))
      const filteredEdges = rfEdges.filter(e => highlightedEdges.has(`${e.source}->${e.target}`))
      const laid = applyDagreLayout(filteredNodes, filteredEdges)
      return { nodes: laid, edges: filteredEdges }
    }

    const laid = applyDagreLayout(rfNodes, rfEdges)
    return { nodes: laid, edges: rfEdges }
  }, [graph, highlightedNodeIds, highlightedEdges, viewMode, showEdgeLabels])
}
```

- [ ] **Step 5: Update GraphPanel**

`playground/src/components/GraphPanel.tsx`:
```typescript
import { ReactFlow, Controls, MiniMap, Background, BackgroundVariant } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { CustomNode } from '@/components/graph/CustomNode'
import { CustomEdge } from '@/components/graph/CustomEdge'
import { useAutoLayout } from '@/hooks/useAutoLayout'
import { useGraphStore } from '@/hooks/useGraphStore'
import { useMemo } from 'react'

const nodeTypes = { custom: CustomNode }
const edgeTypes = { custom: CustomEdge }

export function GraphPanel() {
  const { nodes, edges } = useAutoLayout()
  const { config } = useGraphStore()

  const proOptions = useMemo(() => ({ hideAttribution: true }), [])

  return (
    <div className="h-full w-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        proOptions={proOptions}
        fitView
        className="bg-zinc-950"
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="#27272a" />
        <Controls className="!bg-zinc-800 !border-zinc-700 !text-zinc-300 [&>button]:!bg-zinc-800 [&>button]:!border-zinc-700 [&>button]:!text-zinc-300 [&>button:hover]:!bg-zinc-700" />
        {config.showMinimap && (
          <MiniMap
            className="!bg-zinc-900 !border-zinc-800"
            nodeColor="#3f3f46"
            maskColor="rgba(0,0,0,0.7)"
          />
        )}
      </ReactFlow>
    </div>
  )
}
```

- [ ] **Step 6: Verify graph renders**

Start backend: `source .venv/bin/activate && python -c "from graphstore.server import app; import uvicorn; uvicorn.run(app, port=7200)"`

Start frontend: `cd playground && npm run dev`

In the editor, type and run:
```
CREATE NODE "a" kind = "function" name = "main"
CREATE NODE "b" kind = "function" name = "helper"
CREATE EDGE "a" -> "b" kind = "calls"
```

Verify: nodes appear in graph panel with kind badges, edge connects them.

- [ ] **Step 7: Commit**

```bash
git add playground/src/components/graph/ playground/src/hooks/useAutoLayout.ts playground/src/components/GraphPanel.tsx
git commit -m "feat: React Flow graph panel with custom nodes, edges, dagre layout"
```

---

### Task 7: Toolbar with examples and settings

**Files:**
- Update: `playground/src/components/Toolbar.tsx`
- Create: `playground/src/components/SettingsDialog.tsx`
- Create: `playground/src/examples/index.ts`
- Create: `playground/src/examples/function-call-graph.ts`
- Create: `playground/src/examples/class-hierarchy.ts`
- Create: `playground/src/examples/code-graph.ts`
- Create: `playground/src/examples/microservices-map.ts`

- [ ] **Step 1: Create example scripts**

`playground/src/examples/function-call-graph.ts`:
```typescript
export const functionCallGraph = {
  name: 'Function Call Graph',
  description: '5 functions with calls edges — diamond + chain pattern',
  script: `CREATE NODE "main" kind = "function" name = "main" file = "app.py"
CREATE NODE "parse_args" kind = "function" name = "parse_args" file = "cli.py"
CREATE NODE "validate" kind = "function" name = "validate" file = "utils.py"
CREATE NODE "process" kind = "function" name = "process" file = "core.py"
CREATE NODE "output" kind = "function" name = "output" file = "io.py"
CREATE EDGE "main" -> "parse_args" kind = "calls"
CREATE EDGE "main" -> "validate" kind = "calls"
CREATE EDGE "parse_args" -> "validate" kind = "calls"
CREATE EDGE "validate" -> "process" kind = "calls"
CREATE EDGE "process" -> "output" kind = "calls"

// Try these queries:
// EDGES FROM "main" WHERE kind = "calls"
// SHORTEST PATH FROM "main" TO "output" WHERE kind = "calls"
// COMMON NEIGHBORS OF "main" AND "parse_args" WHERE kind = "calls"
// TRAVERSE FROM "main" DEPTH 3 WHERE kind = "calls"`,
}
```

`playground/src/examples/class-hierarchy.ts`:
```typescript
export const classHierarchy = {
  name: 'Class Hierarchy',
  description: '8 classes with extends, implements, uses edges',
  script: `CREATE NODE "Animal" kind = "class" name = "Animal" abstract = "true"
CREATE NODE "Dog" kind = "class" name = "Dog"
CREATE NODE "Cat" kind = "class" name = "Cat"
CREATE NODE "Serializable" kind = "class" name = "Serializable" abstract = "true"
CREATE NODE "Comparable" kind = "class" name = "Comparable" abstract = "true"
CREATE NODE "PetDog" kind = "class" name = "PetDog"
CREATE NODE "DogFood" kind = "class" name = "DogFood"
CREATE NODE "PetShop" kind = "class" name = "PetShop"
CREATE EDGE "Dog" -> "Animal" kind = "extends"
CREATE EDGE "Cat" -> "Animal" kind = "extends"
CREATE EDGE "PetDog" -> "Dog" kind = "extends"
CREATE EDGE "Dog" -> "Serializable" kind = "implements"
CREATE EDGE "Cat" -> "Serializable" kind = "implements"
CREATE EDGE "Dog" -> "Comparable" kind = "implements"
CREATE EDGE "PetShop" -> "PetDog" kind = "uses"
CREATE EDGE "PetShop" -> "DogFood" kind = "uses"

// Try these queries:
// ANCESTORS OF "PetDog" DEPTH 3
// DESCENDANTS OF "Animal" DEPTH 2
// MATCH ("PetDog") -[kind = "extends"]-> (parent) -[kind = "extends"]-> (grandparent)
// NODES WHERE INDEGREE > 1`,
}
```

`playground/src/examples/code-graph.ts`:
```typescript
export const codeGraph = {
  name: 'Code Graph',
  description: '~20 nodes: functions, classes, modules with multiple edge types',
  script: `BEGIN
CREATE NODE "mod_auth" kind = "module" name = "auth" file = "src/auth/"
CREATE NODE "mod_api" kind = "module" name = "api" file = "src/api/"
CREATE NODE "mod_db" kind = "module" name = "db" file = "src/db/"
CREATE NODE "cls_User" kind = "class" name = "User" file = "src/auth/models.py"
CREATE NODE "cls_Session" kind = "class" name = "Session" file = "src/auth/session.py"
CREATE NODE "cls_Token" kind = "class" name = "Token" file = "src/auth/token.py"
CREATE NODE "cls_BaseModel" kind = "class" name = "BaseModel" file = "src/db/base.py"
CREATE NODE "cls_Router" kind = "class" name = "Router" file = "src/api/router.py"
CREATE NODE "fn_login" kind = "function" name = "login" file = "src/auth/handlers.py"
CREATE NODE "fn_logout" kind = "function" name = "logout" file = "src/auth/handlers.py"
CREATE NODE "fn_verify" kind = "function" name = "verify_token" file = "src/auth/token.py"
CREATE NODE "fn_hash" kind = "function" name = "hash_password" file = "src/auth/crypto.py"
CREATE NODE "fn_query" kind = "function" name = "query" file = "src/db/engine.py"
CREATE NODE "fn_connect" kind = "function" name = "connect" file = "src/db/engine.py"
CREATE NODE "fn_handle_req" kind = "function" name = "handle_request" file = "src/api/handler.py"
CREATE NODE "fn_serialize" kind = "function" name = "serialize" file = "src/api/serial.py"
COMMIT

CREATE EDGE "mod_auth" -> "cls_User" kind = "contains"
CREATE EDGE "mod_auth" -> "cls_Session" kind = "contains"
CREATE EDGE "mod_auth" -> "cls_Token" kind = "contains"
CREATE EDGE "mod_db" -> "cls_BaseModel" kind = "contains"
CREATE EDGE "mod_api" -> "cls_Router" kind = "contains"
CREATE EDGE "cls_User" -> "cls_BaseModel" kind = "extends"
CREATE EDGE "cls_Session" -> "cls_BaseModel" kind = "extends"
CREATE EDGE "fn_login" -> "fn_verify" kind = "calls"
CREATE EDGE "fn_login" -> "fn_hash" kind = "calls"
CREATE EDGE "fn_login" -> "fn_query" kind = "calls"
CREATE EDGE "fn_logout" -> "fn_query" kind = "calls"
CREATE EDGE "fn_handle_req" -> "fn_login" kind = "calls"
CREATE EDGE "fn_handle_req" -> "fn_verify" kind = "calls"
CREATE EDGE "fn_handle_req" -> "fn_serialize" kind = "calls"
CREATE EDGE "fn_query" -> "fn_connect" kind = "calls"
CREATE EDGE "mod_api" -> "mod_auth" kind = "imports"
CREATE EDGE "mod_auth" -> "mod_db" kind = "imports"

// Try these queries:
// SUBGRAPH FROM "fn_login" DEPTH 2
// TRAVERSE FROM "fn_handle_req" DEPTH 3 WHERE kind = "calls"
// MATCH ("fn_handle_req") -[kind = "calls"]-> (a) -[kind = "calls"]-> (b)
// SYS STATS
// SYS EXPLAIN TRAVERSE FROM "fn_handle_req" DEPTH 5 WHERE kind = "calls"`,
}
```

`playground/src/examples/microservices-map.ts`:
```typescript
export const microservicesMap = {
  name: 'Microservices Map',
  description: 'Services, databases, queues with API calls and messaging',
  script: `BEGIN
CREATE NODE "api_gateway" kind = "service" name = "API Gateway" port = 8080
CREATE NODE "auth_svc" kind = "service" name = "Auth Service" port = 8081
CREATE NODE "user_svc" kind = "service" name = "User Service" port = 8082
CREATE NODE "order_svc" kind = "service" name = "Order Service" port = 8083
CREATE NODE "payment_svc" kind = "service" name = "Payment Service" port = 8084
CREATE NODE "notify_svc" kind = "service" name = "Notification Service" port = 8085
CREATE NODE "users_db" kind = "database" name = "Users DB" engine = "postgres"
CREATE NODE "orders_db" kind = "database" name = "Orders DB" engine = "postgres"
CREATE NODE "payments_db" kind = "database" name = "Payments DB" engine = "postgres"
CREATE NODE "event_bus" kind = "queue" name = "Event Bus" engine = "kafka"
CREATE NODE "email_queue" kind = "queue" name = "Email Queue" engine = "rabbitmq"
COMMIT

CREATE EDGE "api_gateway" -> "auth_svc" kind = "calls_api"
CREATE EDGE "api_gateway" -> "user_svc" kind = "calls_api"
CREATE EDGE "api_gateway" -> "order_svc" kind = "calls_api"
CREATE EDGE "auth_svc" -> "users_db" kind = "reads_from"
CREATE EDGE "user_svc" -> "users_db" kind = "reads_from"
CREATE EDGE "order_svc" -> "orders_db" kind = "reads_from"
CREATE EDGE "payment_svc" -> "payments_db" kind = "reads_from"
CREATE EDGE "order_svc" -> "payment_svc" kind = "calls_api"
CREATE EDGE "order_svc" -> "event_bus" kind = "publishes_to"
CREATE EDGE "payment_svc" -> "event_bus" kind = "publishes_to"
CREATE EDGE "notify_svc" -> "event_bus" kind = "subscribes_to"
CREATE EDGE "notify_svc" -> "email_queue" kind = "publishes_to"

// Try these queries:
// PATH FROM "api_gateway" TO "payments_db" MAX_DEPTH 5
// DISTANCE FROM "api_gateway" TO "notify_svc" MAX_DEPTH 5
// NODES WHERE kind = "service"
// EDGES FROM "api_gateway" WHERE kind = "calls_api"
// DESCENDANTS OF "api_gateway" DEPTH 3`,
}
```

`playground/src/examples/index.ts`:
```typescript
import { functionCallGraph } from './function-call-graph'
import { classHierarchy } from './class-hierarchy'
import { codeGraph } from './code-graph'
import { microservicesMap } from './microservices-map'

export const examples = [
  functionCallGraph,
  classHierarchy,
  codeGraph,
  microservicesMap,
]
```

- [ ] **Step 2: Create SettingsDialog**

`playground/src/components/SettingsDialog.tsx`:
```typescript
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Slider } from '@/components/ui/slider'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useGraphStore, type ViewMode, type LayoutMode } from '@/hooks/useGraphStore'

interface Props {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function SettingsDialog({ open, onOpenChange }: Props) {
  const { config, updateConfig } = useGraphStore()

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-zinc-900 border-zinc-800 text-zinc-100 max-w-md">
        <DialogHeader>
          <DialogTitle>Settings</DialogTitle>
        </DialogHeader>
        <Tabs defaultValue="graph">
          <TabsList className="w-full">
            <TabsTrigger value="graph" className="flex-1">Graph</TabsTrigger>
            <TabsTrigger value="store" className="flex-1">Store</TabsTrigger>
            <TabsTrigger value="query" className="flex-1">Query</TabsTrigger>
          </TabsList>

          <TabsContent value="graph" className="space-y-4 mt-4">
            <div className="space-y-2">
              <Label>View Mode</Label>
              <Select value={config.viewMode} onValueChange={(v) => updateConfig({ viewMode: v as ViewMode })}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="live">Live</SelectItem>
                  <SelectItem value="query-result">Query Result</SelectItem>
                  <SelectItem value="highlight">Highlight</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Layout</Label>
              <Select value={config.layoutMode} onValueChange={(v) => updateConfig({ layoutMode: v as LayoutMode })}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="dagre">Dagre (hierarchical)</SelectItem>
                  <SelectItem value="force">Force-directed</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-center justify-between">
              <Label>Edge Labels</Label>
              <Switch checked={config.showEdgeLabels} onCheckedChange={(v) => updateConfig({ showEdgeLabels: v })} />
            </div>
            <div className="flex items-center justify-between">
              <Label>Minimap</Label>
              <Switch checked={config.showMinimap} onCheckedChange={(v) => updateConfig({ showMinimap: v })} />
            </div>
          </TabsContent>

          <TabsContent value="store" className="space-y-4 mt-4">
            <div className="space-y-2">
              <Label>Memory Ceiling: {config.ceilingMb} MB</Label>
              <Slider
                value={[config.ceilingMb]}
                min={64} max={1024} step={64}
                onValueChange={([v]) => updateConfig({ ceilingMb: v })}
              />
            </div>
          </TabsContent>

          <TabsContent value="query" className="space-y-4 mt-4">
            <div className="space-y-2">
              <Label>Cost Threshold: {config.costThreshold.toLocaleString()}</Label>
              <Slider
                value={[config.costThreshold]}
                min={1000} max={1_000_000} step={1000}
                onValueChange={([v]) => updateConfig({ costThreshold: v })}
              />
            </div>
            <div className="flex items-center justify-between">
              <Label>Explain before execute</Label>
              <Switch checked={config.explainBeforeExecute} onCheckedChange={(v) => updateConfig({ explainBeforeExecute: v })} />
            </div>
            <div className="flex items-center justify-between">
              <Label>Show elapsed time</Label>
              <Switch checked={config.showElapsed} onCheckedChange={(v) => updateConfig({ showElapsed: v })} />
            </div>
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  )
}
```

- [ ] **Step 3: Update Toolbar**

`playground/src/components/Toolbar.tsx`:
```typescript
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Separator } from '@/components/ui/separator'
import { Play, PlayCircle, RotateCcw, Settings, BookOpen } from 'lucide-react'
import { useGraphStore } from '@/hooks/useGraphStore'
import { SettingsDialog } from '@/components/SettingsDialog'
import { examples } from '@/examples'
import { useState } from 'react'

export function Toolbar() {
  const { executeQuery, executeAll, resetGraph, setEditorContent, editorContent } = useGraphStore()
  const [settingsOpen, setSettingsOpen] = useState(false)

  const handleRunSelected = () => {
    // Run all if nothing selected (we can't access CodeMirror selection from here easily)
    // Keyboard shortcuts handle run-selected within the editor
    const text = editorContent.trim()
    if (text) executeQuery(text)
  }

  const loadExample = async (example: typeof examples[0]) => {
    await resetGraph()
    setEditorContent(example.script)
    // Auto-execute the create statements
    const lines = example.script.split('\n')
    const queries: string[] = []
    let batch: string[] = []
    let inBatch = false

    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed || trimmed.startsWith('//')) continue
      if (trimmed === 'BEGIN') { inBatch = true; batch = [line]; continue }
      if (trimmed === 'COMMIT' && inBatch) { batch.push(line); queries.push(batch.join('\n')); batch = []; inBatch = false; continue }
      if (inBatch) { batch.push(line); continue }
      // Only auto-execute CREATE/EDGE statements, not queries
      if (trimmed.startsWith('CREATE') || trimmed.startsWith('UPSERT')) queries.push(trimmed)
    }

    for (const q of queries) {
      await useGraphStore.getState().executeQuery(q)
    }
  }

  return (
    <>
      <div className="h-10 border-b border-zinc-800 px-3 flex items-center gap-1.5 bg-zinc-900/50">
        <span className="text-sm font-semibold text-zinc-300 mr-2">graphstore</span>
        <Separator orientation="vertical" className="h-5" />

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="sm" className="h-7 text-xs gap-1.5">
              <BookOpen className="w-3.5 h-3.5" /> Examples
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent>
            {examples.map((ex) => (
              <DropdownMenuItem key={ex.name} onClick={() => loadExample(ex)}>
                <div>
                  <div className="text-sm">{ex.name}</div>
                  <div className="text-xs text-zinc-500">{ex.description}</div>
                </div>
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>

        <Separator orientation="vertical" className="h-5" />

        <Button variant="ghost" size="sm" className="h-7 text-xs gap-1.5" onClick={handleRunSelected}>
          <Play className="w-3.5 h-3.5" /> Run
        </Button>
        <Button variant="ghost" size="sm" className="h-7 text-xs gap-1.5" onClick={executeAll}>
          <PlayCircle className="w-3.5 h-3.5" /> Run All
        </Button>

        <Separator orientation="vertical" className="h-5" />

        <Button variant="ghost" size="sm" className="h-7 text-xs gap-1.5" onClick={resetGraph}>
          <RotateCcw className="w-3.5 h-3.5" /> Reset
        </Button>

        <div className="flex-1" />

        <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={() => setSettingsOpen(true)}>
          <Settings className="w-3.5 h-3.5" />
        </Button>
      </div>

      <SettingsDialog open={settingsOpen} onOpenChange={setSettingsOpen} />
    </>
  )
}
```

- [ ] **Step 4: Commit**

```bash
git add playground/src/examples/ playground/src/components/Toolbar.tsx playground/src/components/SettingsDialog.tsx
git commit -m "feat: toolbar with examples dropdown and settings dialog"
```

---

### Task 8: Stats bar + final integration

**Files:**
- Update: `playground/src/components/StatsBar.tsx`

- [ ] **Step 1: Implement StatsBar**

`playground/src/components/StatsBar.tsx`:
```typescript
import { useGraphStore } from '@/hooks/useGraphStore'

export function StatsBar() {
  const { graph, results } = useGraphStore()
  const lastResult = results[0]
  const elapsed = lastResult?.result?.elapsed_us

  return (
    <div className="h-7 border-t border-zinc-800 px-3 flex items-center gap-4 text-[10px] text-zinc-500 bg-zinc-900/30">
      <span>Nodes: {graph.nodes.length}</span>
      <span>Edges: {graph.edges.length}</span>
      {elapsed != null && <span>Last query: {(elapsed / 1000).toFixed(1)}ms</span>}
    </div>
  )
}
```

- [ ] **Step 2: Verify full integration**

1. Start backend: `source .venv/bin/activate && python -m graphstore.cli playground --no-browser`
2. Start frontend: `cd playground && npm run dev`
3. Load "Function Call Graph" example from dropdown
4. Verify: nodes appear in graph, edges connect them, results show in panel
5. Run queries from the example comments
6. Open settings, change view mode, verify graph updates

- [ ] **Step 3: Commit**

```bash
git add playground/src/components/StatsBar.tsx
git commit -m "feat: stats bar and final integration"
```

---

### Task 9: Build + serve static + cleanup

**Files:**
- Update: `playground/.gitignore`

- [ ] **Step 1: Add playground .gitignore**

`playground/.gitignore`:
```
node_modules/
dist/
```

- [ ] **Step 2: Build frontend**

```bash
cd playground && npm run build
```

Verify `playground/dist/` contains `index.html` and assets.

- [ ] **Step 3: Test production mode**

```bash
source .venv/bin/activate
python -m graphstore.cli playground --no-browser --port 7200
```

Open `http://localhost:7200` — verify the playground works fully from the single server (no separate Vite dev server).

- [ ] **Step 4: Final commit**

```bash
git add playground/ graphstore/ pyproject.toml tests/
git commit -m "feat: graphstore playground — interactive DSL workbench"
```
