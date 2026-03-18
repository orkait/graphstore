import {
  ReactFlow,
  Controls,
  MiniMap,
  Background,
  BackgroundVariant,
} from '@xyflow/react'
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
  const showMinimap = useGraphStore((s) => s.config.showMinimap)
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
        className="bg-background"
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1} className="!text-border" />
        <Controls className="!bg-card !border-border !text-foreground [&>button]:!bg-card [&>button]:!border-border [&>button]:!text-foreground [&>button:hover]:!bg-accent" />
        {showMinimap && (
          <MiniMap className="!bg-card !border-border" nodeColor="var(--muted)" maskColor="rgba(0,0,0,0.7)" />
        )}
      </ReactFlow>
    </div>
  )
}
