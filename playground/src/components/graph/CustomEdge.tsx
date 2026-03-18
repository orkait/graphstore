import { BaseEdge, getStraightPath, type EdgeProps } from '@xyflow/react'

export function CustomEdge(props: EdgeProps) {
  const { sourceX, sourceY, targetX, targetY, data, style } = props
  const [edgePath] = getStraightPath({ sourceX, sourceY, targetX, targetY })
  const highlighted = data?.highlighted as boolean
  const dimmed = data?.dimmed as boolean

  return (
    <>
      <BaseEdge
        path={edgePath}
        style={{
          ...style,
          stroke: highlighted
            ? 'oklch(var(--ring))'
            : dimmed
              ? 'oklch(var(--border))'
              : 'oklch(var(--muted-foreground))',
          strokeWidth: highlighted ? 2 : 1,
          opacity: dimmed ? 0.3 : 1,
          transition: 'all 0.3s',
        }}
      />
      {data?.showLabel && (
        <text
          x={(sourceX + targetX) / 2}
          y={(sourceY + targetY) / 2 - 8}
          className="fill-muted-foreground text-[9px]"
          textAnchor="middle"
        >
          {data.kind as string}
        </text>
      )}
    </>
  )
}
