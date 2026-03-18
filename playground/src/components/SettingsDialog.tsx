import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Slider } from '@/components/ui/slider'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { useGraphStore, type ViewMode, type LayoutMode } from '@/hooks/useGraphStore'

interface Props {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function SettingsDialog({ open, onOpenChange }: Props) {
  const config = useGraphStore((s) => s.config)
  const updateConfig = useGraphStore((s) => s.updateConfig)
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-zinc-900 border-zinc-800 text-zinc-100 max-w-md">
        <DialogHeader>
          <DialogTitle>Settings</DialogTitle>
        </DialogHeader>
        <Tabs defaultValue="graph">
          <TabsList className="w-full">
            <TabsTrigger value="graph" className="flex-1">
              Graph
            </TabsTrigger>
            <TabsTrigger value="store" className="flex-1">
              Store
            </TabsTrigger>
            <TabsTrigger value="query" className="flex-1">
              Query
            </TabsTrigger>
          </TabsList>
          <TabsContent value="graph" className="space-y-4 mt-4">
            <div className="space-y-2">
              <Label>View Mode</Label>
              <Select
                value={config.viewMode}
                onValueChange={(v) =>
                  updateConfig({ viewMode: v as ViewMode })
                }
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="live">Live</SelectItem>
                  <SelectItem value="query-result">Query Result</SelectItem>
                  <SelectItem value="highlight">Highlight</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Layout</Label>
              <Select
                value={config.layoutMode}
                onValueChange={(v) =>
                  updateConfig({ layoutMode: v as LayoutMode })
                }
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="dagre">Dagre (hierarchical)</SelectItem>
                  <SelectItem value="force">Force-directed</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-center justify-between">
              <Label>Edge Labels</Label>
              <Switch
                checked={config.showEdgeLabels}
                onCheckedChange={(v) => updateConfig({ showEdgeLabels: v })}
              />
            </div>
            <div className="flex items-center justify-between">
              <Label>Minimap</Label>
              <Switch
                checked={config.showMinimap}
                onCheckedChange={(v) => updateConfig({ showMinimap: v })}
              />
            </div>
          </TabsContent>
          <TabsContent value="store" className="space-y-4 mt-4">
            <div className="space-y-2">
              <Label>Memory Ceiling: {config.ceilingMb} MB</Label>
              <Slider
                value={[config.ceilingMb]}
                min={64}
                max={1024}
                step={64}
                onValueChange={([v]) => updateConfig({ ceilingMb: v })}
              />
            </div>
          </TabsContent>
          <TabsContent value="query" className="space-y-4 mt-4">
            <div className="space-y-2">
              <Label>
                Cost Threshold: {config.costThreshold.toLocaleString()}
              </Label>
              <Slider
                value={[config.costThreshold]}
                min={1000}
                max={1_000_000}
                step={1000}
                onValueChange={([v]) => updateConfig({ costThreshold: v })}
              />
            </div>
            <div className="flex items-center justify-between">
              <Label>Explain before execute</Label>
              <Switch
                checked={config.explainBeforeExecute}
                onCheckedChange={(v) =>
                  updateConfig({ explainBeforeExecute: v })
                }
              />
            </div>
            <div className="flex items-center justify-between">
              <Label>Show elapsed time</Label>
              <Switch
                checked={config.showElapsed}
                onCheckedChange={(v) => updateConfig({ showElapsed: v })}
              />
            </div>
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  )
}
