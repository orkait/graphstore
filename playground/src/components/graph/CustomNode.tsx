import { Handle, Position, type NodeProps } from '@xyflow/react'
import { Badge } from '@/components/ui/badge'

const KIND_COLORS: Record<string, string> = {
  function: 'bg-chart-1/20 text-chart-1 border-chart-1/30',
  class: 'bg-chart-2/20 text-chart-2 border-chart-2/30',
  module: 'bg-chart-3/20 text-chart-3 border-chart-3/30',
  service: 'bg-chart-4/20 text-chart-4 border-chart-4/30',
  database: 'bg-destructive/20 text-destructive border-destructive/30',
  queue: 'bg-chart-5/20 text-chart-5 border-chart-5/30',
}

function getKindColor(kind: string): string {
  if (KIND_COLORS[kind]) return KIND_COLORS[kind]
  const colors = Object.values(KIND_COLORS)
  let hash = 0
  for (let i = 0; i < kind.length; i++) hash = kind.charCodeAt(i) + ((hash << 5) - hash)
  return colors[Math.abs(hash) % colors.length]
}

export function CustomNode({ data }: NodeProps) {
  const highlighted = data.highlighted as boolean
  const dimmed = data.dimmed as boolean
  return (
    <div
      className={`px-3 py-2 rounded-lg border bg-card min-w-[140px] transition-all duration-300 ${
        highlighted ? 'border-ring ring-2 ring-ring/50' : 'border-border'
      } ${dimmed ? 'opacity-30' : 'opacity-100'}`}
    >
      <Handle type="target" position={Position.Top} className="!w-2 !h-2 !bg-muted-foreground" />
      <div className="text-xs font-bold text-foreground truncate">{data.label as string}</div>
      <Badge className={`text-[9px] mt-1 border ${getKindColor(data.kind as string)}`}>
        {data.kind as string}
      </Badge>
      <Handle type="source" position={Position.Bottom} className="!w-2 !h-2 !bg-muted-foreground" />
    </div>
  )
}
