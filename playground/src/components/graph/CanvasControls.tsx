import { Panel } from '@xyflow/react'
import { useGraphStore, type LayoutMode } from '@/hooks/useGraphStore'
import { Slider } from '@/components/ui/slider'

export function CanvasControls() {
  const config = useGraphStore((s) => s.config)
  const updateConfig = useGraphStore((s) => s.updateConfig)
  const { layoutMode, clusterStrength, repelStrength, centerForce, linkForce, linkDistance } = config

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

        {/* Cluster-specific: all sliders 0-100 */}
        {layoutMode === 'cluster' && (
          <>
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-muted-foreground">Center force</span>
                <span className="text-[10px] font-mono text-muted-foreground">{centerForce}</span>
              </div>
              <Slider
                value={[centerForce]}
                min={0} max={100} step={1}
                onValueChange={(v) => updateConfig({ centerForce: Array.isArray(v) ? v[0] : v })}
              />
            </div>
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-muted-foreground">Repel force</span>
                <span className="text-[10px] font-mono text-muted-foreground">{repelStrength}</span>
              </div>
              <Slider
                value={[repelStrength]}
                min={0} max={100} step={1}
                onValueChange={(v) => updateConfig({ repelStrength: Array.isArray(v) ? v[0] : v })}
              />
            </div>
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-muted-foreground">Link force</span>
                <span className="text-[10px] font-mono text-muted-foreground">{linkForce}</span>
              </div>
              <Slider
                value={[linkForce]}
                min={0} max={100} step={1}
                onValueChange={(v) => updateConfig({ linkForce: Array.isArray(v) ? v[0] : v })}
              />
            </div>
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-muted-foreground">Link distance</span>
                <span className="text-[10px] font-mono text-muted-foreground">{linkDistance}</span>
              </div>
              <Slider
                value={[linkDistance]}
                min={0} max={100} step={1}
                onValueChange={(v) => updateConfig({ linkDistance: Array.isArray(v) ? v[0] : v })}
              />
            </div>
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-muted-foreground">Cluster force</span>
                <span className="text-[10px] font-mono text-muted-foreground">{clusterStrength}</span>
              </div>
              <Slider
                value={[clusterStrength]}
                min={0} max={100} step={1}
                onValueChange={(v) => updateConfig({ clusterStrength: Array.isArray(v) ? v[0] : v })}
              />
            </div>
          </>
        )}
      </div>
    </Panel>
  )
}
