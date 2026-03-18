import { create } from 'zustand'
import { toast } from 'sonner'
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
  editorSelection: string
  highlightedNodeIds: Set<string>
  highlightedEdges: Set<string>
  loading: boolean

  setEditorContent: (content: string) => void
  setEditorSelection: (selection: string) => void
  executeQuery: (query: string) => Promise<void>
  executeSelected: () => Promise<void>
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
  editorSelection: '',
  highlightedNodeIds: new Set(),
  highlightedEdges: new Set(),
  loading: false,

  setEditorContent: (content) => set({ editorContent: content }),
  setEditorSelection: (selection) => set({ editorSelection: selection }),

  executeSelected: async () => {
    const sel = get().editorSelection
    if (sel.trim()) {
      const queries = splitQueries(sel)
      for (const q of queries) await get().executeQuery(q)
    }
  },

  executeQuery: async (query) => {
    set({ loading: true })
    const id = `r-${++resultCounter}`
    try {
      const result = await api.execute(query)
      if (result.kind === 'error') {
        // Server returned a soft error (e.g. NodeExists, duplicate edge)
        set((s) => ({
          results: [{ id, query, result: null, error: result.data as string, timestamp: Date.now() }, ...s.results],
          loading: false,
        }))
      } else {
        const { nodeIds, edgeKeys } = extractHighlights(result)
        set((s) => ({
          results: [{ id, query, result, error: null, timestamp: Date.now() }, ...s.results],
          highlightedNodeIds: nodeIds,
          highlightedEdges: edgeKeys,
          loading: false,
        }))
        if (result.kind === 'ok') await get().refreshGraph()
      }
    } catch (err: any) {
      const msg = err?.error || String(err)
      toast.error('Query failed', { description: msg })
      set((s) => ({
        results: [{ id, query, result: null, error: msg, timestamp: Date.now() }, ...s.results],
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
    } catch {
      toast.error('Server unreachable', { description: 'Could not fetch graph data' })
    }
  },

  resetGraph: async () => {
    try {
      await api.reset()
    } catch {
      toast.error('Server unreachable', { description: 'Could not reset graph — is the server running?' })
    }
    set({
      graph: { nodes: [], edges: [] },
      results: [],
      highlightedNodeIds: new Set(),
      highlightedEdges: new Set(),
    })
  },

  clearResults: () => set({ results: [], highlightedNodeIds: new Set(), highlightedEdges: new Set() }),

  updateConfig: (partial) => {
    set((s) => ({ config: { ...s.config, ...partial } }))
    // Sync server-side settings
    const serverUpdate: Record<string, number> = {}
    if (partial.ceilingMb !== undefined) serverUpdate.ceiling_mb = partial.ceilingMb
    if (partial.costThreshold !== undefined) serverUpdate.cost_threshold = partial.costThreshold
    if (Object.keys(serverUpdate).length > 0) {
      api.updateConfig(serverUpdate).catch(() => {
        toast.error('Server unreachable', { description: 'Could not update server config' })
      })
    }
  },
}))
