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
