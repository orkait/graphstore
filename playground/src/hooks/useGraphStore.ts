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
  graph: GraphData
  results: ResultEntry[]
  config: Config
  editorContent: string
  highlightedNodeIds: Set<string>
  highlightedEdges: Set<string>
  loading: boolean

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
      if (Array.isArray(result.data))
        result.data.forEach((n: any) => n?.id && nodeIds.add(n.id))
      break
    case 'edges':
      if (Array.isArray(result.data))
        result.data.forEach((e: any) => {
          if (e.source) nodeIds.add(e.source)
          if (e.target) nodeIds.add(e.target)
          edgeKeys.add(`${e.source}->${e.target}`)
        })
      break
    case 'path':
      if (Array.isArray(result.data)) {
        result.data.forEach((id: string) => nodeIds.add(id))
        for (let i = 0; i < result.data.length - 1; i++)
          edgeKeys.add(`${result.data[i]}->${result.data[i + 1]}`)
      }
      break
    case 'paths':
      if (Array.isArray(result.data))
        result.data.forEach((path: string[]) => {
          path.forEach((id: string) => nodeIds.add(id))
          for (let i = 0; i < path.length - 1; i++)
            edgeKeys.add(`${path[i]}->${path[i + 1]}`)
        })
      break
    case 'subgraph':
      if (result.data?.nodes)
        result.data.nodes.forEach((n: any) => n?.id && nodeIds.add(n.id))
      if (result.data?.edges)
        result.data.edges.forEach((e: any) => edgeKeys.add(`${e.source}->${e.target}`))
      break
    case 'match':
      if (Array.isArray(result.data))
        result.data.forEach((binding: Record<string, string>) =>
          Object.values(binding).forEach(id => nodeIds.add(id)))
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
      if (result.kind === 'ok') await get().refreshGraph()
    } catch (err: any) {
      set((s) => ({
        results: [{ id, query, result: null, error: err?.error || String(err), timestamp: Date.now() }, ...s.results],
        loading: false,
      }))
    }
  },

  executeAll: async () => {
    const queries = splitQueries(get().editorContent)
    for (const q of queries) await get().executeQuery(q)
  },

  refreshGraph: async () => {
    try {
      const graph = await api.getGraph()
      set({ graph })
    } catch { /* ignore */ }
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

  updateConfig: (partial) => set((s) => ({ config: { ...s.config, ...partial } })),
}))
