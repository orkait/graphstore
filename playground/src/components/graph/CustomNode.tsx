import { Handle, Position, type NodeProps } from '@xyflow/react'
import { Badge } from '@/components/ui/badge'

const KIND_COLORS: Record<string, string> = {
  function: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
  class: 'bg-green-500/20 text-green-300 border-green-500/30',
  module: 'bg-purple-500/20 text-purple-300 border-purple-500/30',
  service: 'bg-orange-500/20 text-orange-300 border-orange-500/30',
  database: 'bg-red-500/20 text-red-300 border-red-500/30',
  queue: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30',
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
      className={`px-3 py-2 rounded-lg border bg-zinc-900 min-w-[140px] transition-all duration-300 ${
        highlighted ? 'border-blue-400 ring-2 ring-blue-400/50' : 'border-zinc-700'
      } ${dimmed ? 'opacity-30' : 'opacity-100'}`}
    >
      <Handle type="target" position={Position.Top} className="!w-2 !h-2 !bg-zinc-600" />
      <div className="text-xs font-bold text-zinc-100 truncate">{data.label as string}</div>
      <Badge className={`text-[9px] mt-1 border ${getKindColor(data.kind as string)}`}>
        {data.kind as string}
      </Badge>
      <Handle type="source" position={Position.Bottom} className="!w-2 !h-2 !bg-zinc-600" />
    </div>
  )
}
