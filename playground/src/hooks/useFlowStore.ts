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
  hoveredNodeId: string | null
  onNodesChange: OnNodesChange
  onEdgesChange: OnEdgesChange
  setNodes: (nodes: Node[]) => void
  setEdges: (edges: Edge[]) => void
  setHoveredNodeId: (id: string | null) => void
}

export const useFlowStore = create<FlowStoreState>()((set) => ({
  nodes: [],
  edges: [],
  hoveredNodeId: null,
  onNodesChange: (changes) =>
    set((s) => ({ nodes: applyNodeChanges(changes, s.nodes) })),
  onEdgesChange: (changes) =>
    set((s) => ({ edges: applyEdgeChanges(changes, s.edges) })),
  setNodes: (nodes) => set({ nodes }),
  setEdges: (edges) => set({ edges }),
  setHoveredNodeId: (id) => set({ hoveredNodeId: id }),
}))
