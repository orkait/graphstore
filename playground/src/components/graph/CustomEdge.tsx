import { memo } from 'react'
import { BaseEdge, getBezierPath, EdgeLabelRenderer, type EdgeProps } from '@xyflow/react'
import { useFlowStore } from '@/hooks/useFlowStore'
import { useGraphStore } from '@/hooks/useGraphStore'

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

export const CustomEdge = memo(function CustomEdge(props: EdgeProps) {
  const { sourceX, sourceY, targetX, targetY, source, target, data, style, id } = props
  const kind = (data?.kind as string) || ''
  const color = getEdgeColor(kind)

  // Primitive selectors — stable, no unnecessary re-renders
  const hoveredNodeId = useFlowStore((st) => st.hoveredNodeId)
  const highlightedEdges = useGraphStore((st) => st.highlightedEdges)
  const showEdgeLabels = useGraphStore((st) => st.config.showEdgeLabels)
  const layoutMode = useGraphStore((st) => st.config.layoutMode)

  // Cluster mode: straight lines (freeform). Dagre mode: bezier curves.
  const [edgePath, labelX, labelY] = layoutMode === 'cluster'
    ? [`M ${sourceX},${sourceY} L ${targetX},${targetY}`, (sourceX + targetX) / 2, (sourceY + targetY) / 2] as [string, number, number]
    : getBezierPath({ sourceX, sourceY, targetX, targetY })

  const key = `${source}->${target}`
  const isHoverEdge = hoveredNodeId != null && (source === hoveredNodeId || target === hoveredNodeId)
  const isHoverDimmed = hoveredNodeId != null && !isHoverEdge
  const isQueryHighlighted = highlightedEdges.has(key)
  const isQueryDimmed = highlightedEdges.size > 0 && !isQueryHighlighted

  const highlighted = isHoverEdge || isQueryHighlighted
  const dimmed = isHoverDimmed || isQueryDimmed

  const isCrossGroup = kind === 'cross-group'
  const crossGroupCount = (data?.crossGroupCount as number) || 0
  const displayLabel = isCrossGroup
    ? `${crossGroupCount} edges`
    : kind

  const groupChildCount = (data?.groupChildCount as number) || 0
  const isGroupEdge = groupChildCount > 0
  const groupStrokeWidth = isGroupEdge ? Math.min(2 + groupChildCount * 0.05, 6) : undefined

  return (
    <>
      <BaseEdge
        path={edgePath}
        style={{
          ...style,
          stroke: highlighted ? 'var(--graph-highlight-border)' : dimmed ? 'var(--graph-edge-dimmed)' : color,
          strokeWidth: highlighted ? 3 : (groupStrokeWidth || 2),
          strokeDasharray: isCrossGroup ? '6 3' : undefined,
          opacity: dimmed ? 0.15 : 0.8,
          transition: 'opacity 0.2s, stroke 0.2s',
        }}
        id={id}
      />
      {showEdgeLabels && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: 'absolute',
              transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
              pointerEvents: 'none',
              fontSize: '10px',
              color: dimmed ? 'var(--graph-node-dimmed-text)' : 'var(--graph-edge-label-text)',
              backgroundColor: 'var(--graph-edge-label-bg)',
              padding: '1px 4px',
              borderRadius: '3px',
              opacity: dimmed ? 0.15 : 1,
              transition: 'opacity 0.2s',
            }}
          >
            {displayLabel}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  )
})
