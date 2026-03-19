import {
  ReactFlow,
  Controls,
  MiniMap,
  Background,
  BackgroundVariant,
  type Node,
  type Edge,
  type NodeMouseHandler,
} from '@xyflow/react'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import '@xyflow/react/dist/style.css'
import { CustomNode } from '@/components/graph/CustomNode'
import { CustomEdge } from '@/components/graph/CustomEdge'
import { CanvasControls } from '@/components/graph/CanvasControls'
import { useGraphStore, type ViewMode } from '@/hooks/useGraphStore'
import { useFlowStore } from '@/hooks/useFlowStore'
import { applyDagreLayout } from '@/components/graph/layout'
import { collapseTransform, type GroupNodeData } from '@/components/graph/collapseTransform'
import { applyForceLayout, createForceSimulation } from '@/components/graph/forceLayout'
import type { Simulation } from 'd3-force'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { useMemo, useEffect, useRef, useState, useCallback } from 'react'
import { Search } from 'lucide-react'

const nodeTypes = { custom: CustomNode }
const edgeTypes = { custom: CustomEdge }

function computeDegrees(edges: { source: string; target: string }[]) {
  const deg: Record<string, number> = {}
  for (const e of edges) {
    deg[e.source] = (deg[e.source] || 0) + 1
    deg[e.target] = (deg[e.target] || 0) + 1
  }
  return deg
}

function getEdgeColor(kind: string): string {
  const map: Record<string, string> = {
    calls: '#3b82f6', extends: '#22c55e', implements: '#a855f7',
    imports: '#f59e0b', contains: '#06b6d4', uses: '#ec4899',
    calls_api: '#3b82f6', reads_from: '#ef4444',
    publishes_to: '#f59e0b', subscribes_to: '#8b5cf6',
  }
  if (map[kind]) return map[kind]
  const colors = Object.values(map)
  let hash = 0
  for (let i = 0; i < kind.length; i++) hash = kind.charCodeAt(i) + ((hash << 5) - hash)
  return colors[Math.abs(hash) % colors.length]
}

export function GraphPanel() {
  const graph = useGraphStore((s) => s.graph)
  const config = useGraphStore((s) => s.config)
  const { viewMode, showMinimap, isDark, nodesep, ranksep, layoutDirection } = config
  const { collapseThreshold } = config
  const expandedGroups = useGraphStore((s) => s.expandedGroups)
  const highlightedNodeIds = useGraphStore((s) => s.highlightedNodeIds)
  const highlightedEdges = useGraphStore((s) => s.highlightedEdges)
  const updateConfig = useGraphStore((s) => s.updateConfig)

  const nodes = useFlowStore((s) => s.nodes)
  const edges = useFlowStore((s) => s.edges)
  const onNodesChange = useFlowStore((s) => s.onNodesChange)
  const onEdgesChange = useFlowStore((s) => s.onEdgesChange)
  const setFlowNodes = useFlowStore((s) => s.setNodes)
  const setFlowEdges = useFlowStore((s) => s.setEdges)
  const setHoveredNodeId = useFlowStore((s) => s.setHoveredNodeId)

  const [searchFilter, setSearchFilter] = useState('')
  const prevStructureRef = useRef('')
  const proOptions = useMemo(() => ({ hideAttribution: true }), [])
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const simulationRef = useRef<{ simulation: Simulation<any, any>; nodeMap: Map<string, any> } | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  const degrees = useMemo(() => computeDegrees(graph.edges), [graph.edges])

  // Single effect: layout ONLY when graph structure or search changes
  // Highlight/dim is computed inside CustomNode/CustomEdge directly
  useEffect(() => {
    const structureKey = JSON.stringify({
      n: graph.nodes.map(n => n.id).sort(),
      e: graph.edges.map(e => `${e.source}->${e.target}`).sort(),
      f: searchFilter,
      v: viewMode,
      h: [...highlightedNodeIds].sort(),
      he: [...highlightedEdges].sort(),
      ns: nodesep, rs: ranksep, ld: layoutDirection,
      ct: collapseThreshold,
      eg: expandedGroups,
      cs: config.clusterStrength, rs2: config.repelStrength,
    })

    if (structureKey === prevStructureRef.current) return
    prevStructureRef.current = structureKey

    const filteredGraphNodes = searchFilter
      ? graph.nodes.filter(n =>
          n.id.toLowerCase().includes(searchFilter.toLowerCase()) ||
          (n.name || '').toLowerCase().includes(searchFilter.toLowerCase()) ||
          (n.kind || '').toLowerCase().includes(searchFilter.toLowerCase())
        )
      : graph.nodes
    const filteredIds = new Set(filteredGraphNodes.map(n => n.id))

    // Apply collapse transform for dagre mode
    const { nodes: collapsedNodes, edges: collapsedEdges, groupMeta } =
      config.layoutMode === 'dagre'
        ? collapseTransform(
            filteredGraphNodes,
            graph.edges.filter(e => filteredIds.has(e.source) && filteredIds.has(e.target)),
            collapseThreshold,
            expandedGroups
          )
        : { nodes: filteredGraphNodes, edges: graph.edges.filter(e => filteredIds.has(e.source) && filteredIds.has(e.target)), groupMeta: new Map<string, GroupNodeData>() }
    const collapsedIds = new Set(collapsedNodes.map(n => n.id))

    const rfNodes: Node[] = collapsedNodes.map((n) => {
      const meta = groupMeta.get(n.id)
      return {
        id: n.id,
        type: 'custom',
        position: { x: 0, y: 0 },
        data: meta
          ? {
              label: n.id,
              kind: meta.kind,
              degree: meta.childCount,
              fields: n,
              isGroup: true,
              childCount: meta.childCount,
              groupParentId: meta.parentId,
              groupKind: meta.kind,
            }
          : {
              label: n.id,
              kind: n.kind,
              degree: degrees[n.id] || 0,
              fields: n,
            },
      }
    })

    const rfEdges: Edge[] = collapsedEdges
      .filter(e => collapsedIds.has(e.source) && collapsedIds.has(e.target))
      .map((e, i) => ({
        id: `e-${i}-${e.source}-${e.target}-${e.kind}`,
        source: e.source,
        target: e.target,
        type: 'custom',
        data: {
          kind: e.kind,
          crossGroupCount: e.crossGroupCount,
          groupChildCount: e.groupChildCount,
        },
      }))

    if (config.layoutMode === 'cluster') {
      // Stop any previous simulation
      simulationRef.current?.simulation.stop()

      const container = containerRef.current
      const width = container?.clientWidth || 800
      const height = container?.clientHeight || 600

      // Initial synchronous layout for instant positioning
      const positioned = applyForceLayout(rfNodes, rfEdges, {
        clusterStrength: config.clusterStrength,
        repelStrength: config.repelStrength,
        width,
        height,
      })
      setFlowNodes(positioned)
      setFlowEdges(rfEdges)

      // Create a live simulation for drag interactions (starts stopped/cooled)
      const live = createForceSimulation(positioned, rfEdges, {
        clusterStrength: config.clusterStrength,
        repelStrength: config.repelStrength,
        width,
        height,
      }, (updatedNodes) => {
        setFlowNodes(updatedNodes)
      })
      // Stop immediately — we only restart on drag
      live.simulation.stop()
      simulationRef.current = live
    } else {
      // Dagre layout (existing code)
      const layoutOpts = { direction: layoutDirection, nodesep, ranksep }
      if (viewMode === 'query-result' && highlightedNodeIds.size > 0) {
        const filtered = rfNodes.filter(n => highlightedNodeIds.has(n.id))
        const filtEdges = rfEdges.filter(e => highlightedEdges.has(`${e.source}->${e.target}`))
        setFlowNodes(applyDagreLayout(filtered, filtEdges, layoutOpts))
        setFlowEdges(filtEdges)
      } else {
        setFlowNodes(applyDagreLayout(rfNodes, rfEdges, layoutOpts))
        setFlowEdges(rfEdges)
      }
    }
  }, [graph, searchFilter, degrees, viewMode, highlightedNodeIds, highlightedEdges, nodesep, ranksep, layoutDirection, expandedGroups, collapseThreshold, config.layoutMode, config.clusterStrength, config.repelStrength, setFlowNodes, setFlowEdges])

  // Cleanup simulation when layout mode changes
  useEffect(() => {
    return () => {
      simulationRef.current?.simulation.stop()
      simulationRef.current = null
    }
  }, [config.layoutMode])

  const onNodeDragStart = useCallback((_event: React.MouseEvent, node: Node) => {
    if (config.layoutMode !== 'cluster' || !simulationRef.current) return
    const { simulation, nodeMap } = simulationRef.current
    const fn = nodeMap.get(node.id)
    if (fn) {
      fn.fx = fn.x
      fn.fy = fn.y
      simulation.alphaTarget(0.3).restart()
    }
  }, [config.layoutMode])

  const onNodeDragStop = useCallback((_event: React.MouseEvent, node: Node) => {
    if (config.layoutMode !== 'cluster' || !simulationRef.current) return
    const { simulation, nodeMap } = simulationRef.current
    const fn = nodeMap.get(node.id)
    if (fn) {
      fn.fx = null
      fn.fy = null
      simulation.alphaTarget(0)
    }
  }, [config.layoutMode])

  const onNodeMouseEnter: NodeMouseHandler = useCallback((_event, node) => {
    setHoveredNodeId(node.id)
  }, [setHoveredNodeId])

  const onNodeMouseLeave: NodeMouseHandler = useCallback(() => {
    setHoveredNodeId(null)
  }, [setHoveredNodeId])

  return (
    <Card className="h-full gap-0 py-0 rounded-lg flex flex-col">
      <CardHeader className="px-3 py-2 border-b flex flex-row items-center justify-between space-y-0 h-[40px] flex-shrink-0">
        <div className="text-xs text-muted-foreground flex items-center gap-2 flex-1">
          <Search className="w-3 h-3" />
          <input
            type="text"
            placeholder="Filter nodes..."
            value={searchFilter}
            onChange={(e) => setSearchFilter(e.target.value)}
            className="flex-1 bg-transparent text-xs text-foreground placeholder:text-muted-foreground outline-none font-normal"
          />
          {searchFilter && (
            <button onClick={() => setSearchFilter('')} className="text-[10px] text-muted-foreground hover:text-foreground">
              Clear
            </button>
          )}
        </div>
        <div className="flex items-center ml-4">
          <Tabs value={viewMode} onValueChange={(v) => updateConfig({ viewMode: v as ViewMode })} className="h-7 w-[240px]">
            <TabsList className="grid w-full grid-cols-3 h-7 bg-muted/50 p-1 rounded-md">
              <TabsTrigger value="live" className="text-[10px] h-5 rounded-sm px-2">Live</TabsTrigger>
              <TabsTrigger value="query-result" className="text-[10px] h-5 rounded-sm px-2">Results</TabsTrigger>
              <TabsTrigger value="highlight" className="text-[10px] h-5 rounded-sm px-2">Highlight</TabsTrigger>
            </TabsList>
          </Tabs>
        </div>
      </CardHeader>
      <CardContent ref={containerRef} className="flex-1 min-h-0 overflow-hidden p-0 relative">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeMouseEnter={onNodeMouseEnter}
          onNodeMouseLeave={onNodeMouseLeave}
          onNodeDragStart={onNodeDragStart}
          onNodeDragStop={onNodeDragStop}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          proOptions={proOptions}
          fitView
          nodesDraggable
          className="bg-background"
        >
          <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--graph-bg-dots)" />
          <Controls className="!bg-card !border-border !text-foreground [&>button]:!bg-card [&>button]:!border-border [&>button]:!text-foreground [&>button:hover]:!bg-accent" />
          <CanvasControls />
          {showMinimap && (
            <MiniMap
              className="!bg-card !border-border"
              nodeColor={(n) => getEdgeColor((n.data?.kind as string) || '')}
              maskColor={isDark ? 'rgba(0,0,0,0.7)' : 'rgba(255,255,255,0.7)'}
            />
          )}
        </ReactFlow>
      </CardContent>
    </Card>
  )
}
