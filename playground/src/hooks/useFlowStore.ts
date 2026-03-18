import { create } from 'zustand'
import {
  applyNodeChanges,
  applyEdgeChanges,
  type Node,
  type Edge,
  type OnNodesChange,
  type OnEdgesChange,
} from '@xyflow/react'

interface FlowStoreState {
  nodes: Node[]
  edges: Edge[]
  onNodesChange: OnNodesChange
  onEdgesChange: OnEdgesChange
  setNodes: (nodes: Node[]) => void
  setEdges: (edges: Edge[]) => void
  updateNodeData: (id: string, data: Partial<Record<string, unknown>>) => void
  updateAllNodeData: (updater: (node: Node) => Partial<Record<string, unknown>>) => void
  updateAllEdgeData: (updater: (edge: Edge) => Partial<Record<string, unknown>> & { animated?: boolean }) => void
}

export const useFlowStore = create<FlowStoreState>()((set) => ({
  nodes: [],
  edges: [],
  onNodesChange: (changes) =>
    set((s) => ({ nodes: applyNodeChanges(changes, s.nodes) })),
  onEdgesChange: (changes) =>
    set((s) => ({ edges: applyEdgeChanges(changes, s.edges) })),
  setNodes: (nodes) => set({ nodes }),
  setEdges: (edges) => set({ edges }),
  updateNodeData: (id, data) =>
    set((s) => ({
      nodes: s.nodes.map((n) =>
        n.id === id ? { ...n, data: { ...n.data, ...data } } : n
      ),
    })),
  updateAllNodeData: (updater) =>
    set((s) => ({
      nodes: s.nodes.map((n) => {
        const patch = updater(n)
        return { ...n, data: { ...n.data, ...patch } }
      }),
    })),
  updateAllEdgeData: (updater) =>
    set((s) => ({
      edges: s.edges.map((e) => {
        const { animated, ...dataPatch } = updater(e)
        return {
          ...e,
          ...(animated !== undefined ? { animated } : {}),
          data: { ...e.data, ...dataPatch },
        }
      }),
    })),
}))
