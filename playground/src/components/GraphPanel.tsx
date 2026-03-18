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
        className="bg-zinc-950"
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={20}
          size={1}
          color="#27272a"
        />
        <Controls className="!bg-zinc-800 !border-zinc-700 !text-zinc-300 [&>button]:!bg-zinc-800 [&>button]:!border-zinc-700 [&>button]:!text-zinc-300 [&>button:hover]:!bg-zinc-700" />
        {showMinimap && (
          <MiniMap
            className="!bg-zinc-900 !border-zinc-800"
            nodeColor="#3f3f46"
            maskColor="rgba(0,0,0,0.7)"
          />
        )}
      </ReactFlow>
    </div>
  )
}
