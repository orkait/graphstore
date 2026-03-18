import {
  ReactFlow,
  Controls,
  MiniMap,
  Background,
  BackgroundVariant,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  type NodeMouseHandler,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { CustomNode } from '@/components/graph/CustomNode'
import { CustomEdge } from '@/components/graph/CustomEdge'
import { useGraphStore } from '@/hooks/useGraphStore'
import { applyDagreLayout } from '@/components/graph/layout'
import { useMemo, useEffect, useRef, useState, useCallback } from 'react'
import { Search } from 'lucide-react'

const nodeTypes = { custom: CustomNode }
const edgeTypes = { custom: CustomEdge }

// Build adjacency map for hover highlighting
function buildAdjacency(edges: { source: string; target: string }[]) {
  const adj: Record<string, Set<string>> = {}
  for (const e of edges) {
    if (!adj[e.source]) adj[e.source] = new Set()
    if (!adj[e.target]) adj[e.target] = new Set()
    adj[e.source].add(e.target)
    adj[e.target].add(e.source)
  }
  return adj
}

// Compute degree for each node
function computeDegrees(edges: { source: string; target: string }[]) {
  const deg: Record<string, number> = {}
  for (const e of edges) {
    deg[e.source] = (deg[e.source] || 0) + 1
    deg[e.target] = (deg[e.target] || 0) + 1
  }
  return deg
}

const EDGE_COLORS: Record<string, string> = {
  calls: '#3b82f6', extends: '#22c55e', implements: '#a855f7',
  imports: '#f59e0b', contains: '#06b6d4', uses: '#ec4899',
  calls_api: '#3b82f6', reads_from: '#ef4444',
  publishes_to: '#f59e0b', subscribes_to: '#8b5cf6',
}

function getEdgeColor(kind: string): string {
  if (EDGE_COLORS[kind]) return EDGE_COLORS[kind]
  const colors = Object.values(EDGE_COLORS)
  let hash = 0
  for (let i = 0; i < kind.length; i++) hash = kind.charCodeAt(i) + ((hash << 5) - hash)
  return colors[Math.abs(hash) % colors.length]
}

export function GraphPanel() {
  const { graph, config, highlightedNodeIds, highlightedEdges } = useGraphStore()
  const { viewMode, showEdgeLabels, showMinimap } = config
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([])
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([])
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null)
  const [searchFilter, setSearchFilter] = useState('')
  const prevStructureRef = useRef('')
  const proOptions = useMemo(() => ({ hideAttribution: true }), [])

  const adjacency = useMemo(() => buildAdjacency(graph.edges), [graph.edges])
  const degrees = useMemo(() => computeDegrees(graph.edges), [graph.edges])

  // Only re-layout when graph structure changes, not on every highlight/hover
  useEffect(() => {
    const structureKey = JSON.stringify({
      nodes: graph.nodes.map(n => n.id).sort(),
      edges: graph.edges.map(e => `${e.source}->${e.target}`).sort(),
    })

    const needsLayout = structureKey !== prevStructureRef.current
    prevStructureRef.current = structureKey

    // Filter by search
    const filteredGraphNodes = searchFilter
      ? graph.nodes.filter(n =>
          n.id.toLowerCase().includes(searchFilter.toLowerCase()) ||
          (n.name || '').toLowerCase().includes(searchFilter.toLowerCase()) ||
          (n.kind || '').toLowerCase().includes(searchFilter.toLowerCase())
        )
      : graph.nodes
    const filteredIds = new Set(filteredGraphNodes.map(n => n.id))

    const rfNodes: Node[] = filteredGraphNodes.map((n) => {
      const deg = degrees[n.id] || 0
      // Hover: highlight this node + its neighbors
      const isHoverTarget = hoveredNodeId === n.id
      const isHoverNeighbor = hoveredNodeId != null && adjacency[hoveredNodeId]?.has(n.id)
      const isHoverDimmed = hoveredNodeId != null && !isHoverTarget && !isHoverNeighbor
      // Query highlights
      const isQueryHighlighted = highlightedNodeIds.has(n.id)
      const isQueryDimmed = viewMode === 'highlight' && highlightedNodeIds.size > 0 && !isQueryHighlighted

      return {
        id: n.id,
        type: 'custom',
        position: { x: 0, y: 0 },
        data: {
          label: n.id,
          kind: n.kind,
          degree: deg,
          highlighted: isHoverTarget || isQueryHighlighted,
          dimmed: isHoverDimmed || isQueryDimmed,
          fields: n,
        },
      }
    })

    const rfEdges: Edge[] = graph.edges
      .filter(e => filteredIds.has(e.source) && filteredIds.has(e.target))
      .map((e, i) => {
        const key = `${e.source}->${e.target}`
        const isHoverEdge = hoveredNodeId != null && (e.source === hoveredNodeId || e.target === hoveredNodeId)
        const isHoverDimmed = hoveredNodeId != null && !isHoverEdge
        const isQueryHighlighted = highlightedEdges.has(key)
        const isQueryDimmed = viewMode === 'highlight' && highlightedEdges.size > 0 && !isQueryHighlighted

        return {
          id: `e-${i}-${e.source}-${e.target}-${e.kind}`,
          source: e.source,
          target: e.target,
          type: 'custom',
          animated: isHoverEdge,
          data: {
            kind: e.kind,
            highlighted: isHoverEdge || isQueryHighlighted,
            dimmed: isHoverDimmed || isQueryDimmed,
            showLabel: showEdgeLabels,
          },
        }
      })

    if (viewMode === 'query-result' && highlightedNodeIds.size > 0) {
      const filtered = rfNodes.filter(n => highlightedNodeIds.has(n.id))
      const filtEdges = rfEdges.filter(e => highlightedEdges.has(`${e.source}->${e.target}`))
      setNodes(applyDagreLayout(filtered, filtEdges))
      setEdges(filtEdges)
      return
    }

    if (needsLayout) {
      setNodes(applyDagreLayout(rfNodes, rfEdges))
    } else {
      // Preserve positions, only update data
      setNodes(prev => prev.map(pn => {
        const updated = rfNodes.find(n => n.id === pn.id)
        if (updated) return { ...pn, data: updated.data }
        return pn
      }).concat(rfNodes.filter(n => !prev.find(pn => pn.id === n.id))))
    }
    setEdges(rfEdges)
  }, [graph, hoveredNodeId, highlightedNodeIds, highlightedEdges, viewMode, showEdgeLabels, searchFilter, degrees, adjacency])

  const onNodeMouseEnter: NodeMouseHandler = useCallback((_event, node) => {
    setHoveredNodeId(node.id)
  }, [])

  const onNodeMouseLeave: NodeMouseHandler = useCallback(() => {
    setHoveredNodeId(null)
  }, [])

  return (
    <div className="h-full w-full flex flex-col">
      {/* Search bar */}
      <div className="h-8 border-b border-border flex items-center px-2 gap-2 bg-card/30">
        <Search className="w-3 h-3 text-muted-foreground" />
        <input
          type="text"
          placeholder="Filter nodes..."
          value={searchFilter}
          onChange={(e) => setSearchFilter(e.target.value)}
          className="flex-1 bg-transparent text-xs text-foreground placeholder:text-muted-foreground outline-none"
        />
        {searchFilter && (
          <button onClick={() => setSearchFilter('')} className="text-[10px] text-muted-foreground hover:text-foreground">
            Clear
          </button>
        )}
      </div>
      <div className="flex-1">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeMouseEnter={onNodeMouseEnter}
          onNodeMouseLeave={onNodeMouseLeave}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          proOptions={proOptions}
          fitView
          className="bg-background"
          nodesDraggable
        >
          <Background variant={BackgroundVariant.Dots} gap={20} size={1} className="!text-border" />
          <Controls className="!bg-card !border-border !text-foreground [&>button]:!bg-card [&>button]:!border-border [&>button]:!text-foreground [&>button:hover]:!bg-accent" />
          {showMinimap && (
            <MiniMap
              className="!bg-card !border-border"
              nodeColor={(n) => {
                const kind = n.data?.kind as string
                return getEdgeColor(kind) || '#666'
              }}
              maskColor="rgba(0,0,0,0.7)"
            />
          )}
        </ReactFlow>
      </div>
    </div>
  )
}
