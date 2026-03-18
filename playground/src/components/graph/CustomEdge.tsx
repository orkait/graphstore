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
          stroke: highlighted ? '#60a5fa' : dimmed ? '#27272a' : '#52525b',
          strokeWidth: highlighted ? 2 : 1,
          opacity: dimmed ? 0.3 : 1,
          transition: 'all 0.3s',
        }}
      />
      {data?.showLabel && (
        <text
          x={(sourceX + targetX) / 2}
          y={(sourceY + targetY) / 2 - 8}
          className="fill-zinc-500 text-[9px]"
          textAnchor="middle"
        >
          {data.kind as string}
        </text>
      )}
    </>
  )
}
