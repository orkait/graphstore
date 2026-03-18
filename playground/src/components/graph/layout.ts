import dagre from 'dagre'
import type { Node, Edge } from '@xyflow/react'

const NODE_WIDTH = 180
const NODE_HEIGHT = 60

export function applyDagreLayout(nodes: Node[], edges: Edge[], direction: 'TB' | 'LR' = 'TB'): Node[] {
  if (nodes.length === 0) return nodes
  const g = new dagre.graphlib.Graph()
  g.setDefaultEdgeLabel(() => ({}))
  g.setGraph({ rankdir: direction, nodesep: 50, ranksep: 80 })
  nodes.forEach((node) => g.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT }))
  edges.forEach((edge) => g.setEdge(edge.source, edge.target))
  dagre.layout(g)
  return nodes.map((node) => {
    const pos = g.node(node.id)
    return { ...node, position: { x: pos.x - NODE_WIDTH / 2, y: pos.y - NODE_HEIGHT / 2 } }
  })
}
