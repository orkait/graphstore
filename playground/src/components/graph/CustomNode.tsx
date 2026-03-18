import { Handle, Position, type NodeProps } from '@xyflow/react'

const KIND_STYLES: Record<string, { bg: string; border: string; text: string; badge: string }> = {
  function: { bg: '#1e3a5f', border: '#3b82f6', text: '#93c5fd', badge: '#3b82f6' },
  class:    { bg: '#14532d', border: '#22c55e', text: '#86efac', badge: '#22c55e' },
  module:   { bg: '#3b0764', border: '#a855f7', text: '#d8b4fe', badge: '#a855f7' },
  service:  { bg: '#431407', border: '#f97316', text: '#fdba74', badge: '#f97316' },
  database: { bg: '#450a0a', border: '#ef4444', text: '#fca5a5', badge: '#ef4444' },
  queue:    { bg: '#422006', border: '#eab308', text: '#fde047', badge: '#eab308' },
}

const FALLBACK_STYLES = [
  { bg: '#164e63', border: '#06b6d4', text: '#67e8f9', badge: '#06b6d4' },
  { bg: '#4a044e', border: '#d946ef', text: '#f0abfc', badge: '#d946ef' },
  { bg: '#1e1b4b', border: '#6366f1', text: '#a5b4fc', badge: '#6366f1' },
]

function getKindStyle(kind: string) {
  if (KIND_STYLES[kind]) return KIND_STYLES[kind]
  let hash = 0
  for (let i = 0; i < kind.length; i++) hash = kind.charCodeAt(i) + ((hash << 5) - hash)
  return FALLBACK_STYLES[Math.abs(hash) % FALLBACK_STYLES.length]
}

export function CustomNode({ data }: NodeProps) {
  const highlighted = data.highlighted as boolean
  const dimmed = data.dimmed as boolean
  const kind = (data.kind as string) || 'default'
  const s = getKindStyle(kind)

  return (
    <div
      style={{
        backgroundColor: dimmed ? '#1a1a1a' : s.bg,
        borderColor: highlighted ? '#60a5fa' : dimmed ? '#333' : s.border,
        borderWidth: highlighted ? '2px' : '1px',
        boxShadow: highlighted ? '0 0 12px rgba(96,165,250,0.4)' : `0 0 8px ${s.border}33`,
        opacity: dimmed ? 0.3 : 1,
        transition: 'all 0.3s',
      }}
      className="px-3 py-2 rounded-lg border min-w-[150px]"
    >
      <Handle type="target" position={Position.Top} className="!w-2 !h-2" style={{ background: s.border }} />
      <div className="text-xs font-semibold truncate" style={{ color: dimmed ? '#666' : s.text }}>
        {data.label as string}
      </div>
      <div
        className="text-[9px] mt-1 inline-block px-1.5 py-0.5 rounded"
        style={{
          backgroundColor: `${s.badge}22`,
          color: dimmed ? '#555' : s.badge,
          border: `1px solid ${s.badge}44`,
        }}
      >
        {kind}
      </div>
      <Handle type="source" position={Position.Bottom} className="!w-2 !h-2" style={{ background: s.border }} />
    </div>
  )
}
