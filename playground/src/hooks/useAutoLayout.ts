import { useMemo } from 'react'
import type { Node, Edge } from '@xyflow/react'
import { applyDagreLayout } from '@/components/graph/layout'
import { useGraphStore } from '@/hooks/useGraphStore'

const EDGE_STYLE_MAP: Record<string, number> = {}
let styleCounter = 0
function getEdgeStyleIndex(kind: string): number {
  if (!(kind in EDGE_STYLE_MAP)) EDGE_STYLE_MAP[kind] = styleCounter++ % 3
  return EDGE_STYLE_MAP[kind]
}

export function useAutoLayout(): { nodes: Node[]; edges: Edge[] } {
  const { graph, config, highlightedNodeIds, highlightedEdges } = useGraphStore()
  const { viewMode, showEdgeLabels } = config

  return useMemo(() => {
    if (!graph.nodes.length) return { nodes: [], edges: [] }

    const rfNodes: Node[] = graph.nodes.map((n) => ({
      id: n.id,
      type: 'custom',
      position: { x: 0, y: 0 },
      data: {
        label: n.id,
        kind: n.kind,
        highlighted: highlightedNodeIds.has(n.id),
        dimmed:
          viewMode === 'highlight' &&
          highlightedNodeIds.size > 0 &&
          !highlightedNodeIds.has(n.id),
        fields: n,
      },
    }))

    const rfEdges: Edge[] = graph.edges.map((e, i) => {
      const key = `${e.source}->${e.target}`
      return {
        id: `e-${i}-${e.source}-${e.target}-${e.kind}`,
        source: e.source,
        target: e.target,
        type: 'custom',
        data: {
          kind: e.kind,
          highlighted: highlightedEdges.has(key),
          dimmed:
            viewMode === 'highlight' &&
            highlightedEdges.size > 0 &&
            !highlightedEdges.has(key),
          showLabel: showEdgeLabels,
          styleIndex: getEdgeStyleIndex(e.kind),
        },
      }
    })

    if (viewMode === 'query-result' && highlightedNodeIds.size > 0) {
      const filteredNodes = rfNodes.filter((n) => highlightedNodeIds.has(n.id))
      const filteredEdges = rfEdges.filter((e) =>
        highlightedEdges.has(`${e.source}->${e.target}`),
      )
      return {
        nodes: applyDagreLayout(filteredNodes, filteredEdges),
        edges: filteredEdges,
      }
    }

    return { nodes: applyDagreLayout(rfNodes, rfEdges), edges: rfEdges }
  }, [graph, highlightedNodeIds, highlightedEdges, viewMode, showEdgeLabels])
}
