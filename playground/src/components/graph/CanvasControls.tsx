import { Panel } from '@xyflow/react'
import { useGraphStore, type LayoutMode } from '@/hooks/useGraphStore'
import { Slider } from '@/components/ui/slider'

export function CanvasControls() {
  const config = useGraphStore((s) => s.config)
  const updateConfig = useGraphStore((s) => s.updateConfig)
  const { layoutMode, collapseThreshold, clusterStrength, repelStrength } = config

  return (
    <Panel position="top-right">
      <div className="bg-card/80 backdrop-blur-sm border border-border rounded-lg p-3 space-y-3 min-w-[180px]">
        {/* Mode toggle */}
        <div className="flex rounded-md overflow-hidden border border-border text-[11px]">
          <button
            onClick={() => updateConfig({ layoutMode: 'dagre' as LayoutMode })}
            className={`flex-1 px-3 py-1.5 transition-colors ${
              layoutMode === 'dagre'
                ? 'bg-primary text-primary-foreground'
                : 'bg-card text-muted-foreground hover:bg-accent'
            }`}
          >
            Dagre
          </button>
          <button
            onClick={() => updateConfig({ layoutMode: 'cluster' as LayoutMode })}
            className={`flex-1 px-3 py-1.5 transition-colors ${
              layoutMode === 'cluster'
                ? 'bg-primary text-primary-foreground'
                : 'bg-card text-muted-foreground hover:bg-accent'
            }`}
          >
            Cluster
          </button>
        </div>

        {/* Dagre-specific: collapse threshold */}
        {layoutMode === 'dagre' && (
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-muted-foreground">Collapse</span>
              <span className="text-[10px] font-mono text-muted-foreground">{collapseThreshold}</span>
            </div>
            <Slider
              value={[collapseThreshold]}
              min={5} max={100} step={5}
              onValueChange={(v) => updateConfig({ collapseThreshold: Array.isArray(v) ? v[0] : v })}
            />
          </div>
        )}

        {/* Cluster-specific: strength sliders */}
        {layoutMode === 'cluster' && (
          <>
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-muted-foreground">Cluster</span>
                <span className="text-[10px] font-mono text-muted-foreground">{clusterStrength.toFixed(1)}</span>
              </div>
              <Slider
                value={[clusterStrength * 100]}
                min={10} max={100} step={5}
                onValueChange={(v) => updateConfig({ clusterStrength: (Array.isArray(v) ? v[0] : v) / 100 })}
              />
            </div>
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-muted-foreground">Repel</span>
                <span className="text-[10px] font-mono text-muted-foreground">{repelStrength}</span>
              </div>
              <Slider
                value={[repelStrength]}
                min={50} max={500} step={25}
                onValueChange={(v) => updateConfig({ repelStrength: Array.isArray(v) ? v[0] : v })}
              />
            </div>
          </>
        )}
      </div>
    </Panel>
  )
}
