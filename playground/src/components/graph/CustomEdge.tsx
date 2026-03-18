import { BaseEdge, getBezierPath, EdgeLabelRenderer, type EdgeProps } from '@xyflow/react'

const EDGE_COLORS: Record<string, string> = {
  calls: '#3b82f6',
  extends: '#22c55e',
  implements: '#a855f7',
  imports: '#f59e0b',
  contains: '#06b6d4',
  uses: '#ec4899',
  calls_api: '#3b82f6',
  reads_from: '#ef4444',
  publishes_to: '#f59e0b',
  subscribes_to: '#8b5cf6',
}

function getEdgeColor(kind: string): string {
  if (EDGE_COLORS[kind]) return EDGE_COLORS[kind]
  const colors = Object.values(EDGE_COLORS)
  let hash = 0
  for (let i = 0; i < kind.length; i++) hash = kind.charCodeAt(i) + ((hash << 5) - hash)
  return colors[Math.abs(hash) % colors.length]
}

export function CustomEdge(props: EdgeProps) {
  const { sourceX, sourceY, targetX, targetY, data, style, id } = props
  const [edgePath, labelX, labelY] = getBezierPath({ sourceX, sourceY, targetX, targetY })
  const highlighted = data?.highlighted as boolean
  const dimmed = data?.dimmed as boolean
  const kind = (data?.kind as string) || ''
  const color = getEdgeColor(kind)

  return (
    <>
      <BaseEdge
        path={edgePath}
        style={{
          ...style,
          stroke: highlighted ? '#60a5fa' : dimmed ? '#333' : color,
          strokeWidth: highlighted ? 3 : 2,
          opacity: dimmed ? 0.15 : 0.8,
          transition: 'all 0.3s',
        }}
        id={id}
      />
      {data?.showLabel && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: 'absolute',
              transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
              pointerEvents: 'none',
              fontSize: '10px',
              color: dimmed ? '#444' : '#aaa',
              backgroundColor: 'rgba(0,0,0,0.6)',
              padding: '1px 4px',
              borderRadius: '3px',
            }}
          >
            {kind}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  )
}
