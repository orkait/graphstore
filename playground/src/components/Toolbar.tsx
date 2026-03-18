import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Separator } from '@/components/ui/separator'
import { RotateCcw, Settings, BookOpen, Sun, Moon } from 'lucide-react'
import { useGraphStore } from '@/hooks/useGraphStore'
import { SettingsDialog } from '@/components/SettingsDialog'
import { examples } from '@/examples'
import { useState } from 'react'

export function Toolbar() {
  const resetGraph = useGraphStore((s) => s.resetGraph)
  const setEditorContent = useGraphStore((s) => s.setEditorContent)
  const isDark = useGraphStore((s) => s.config.isDark)
  const updateConfig = useGraphStore((s) => s.updateConfig)
  const [settingsOpen, setSettingsOpen] = useState(false)

  const loadExample = async (example: (typeof examples)[0]) => {
    await resetGraph()
    setEditorContent(example.script)
    const lines = example.script.split('\n')
    const queries: string[] = []
    let batch: string[] = []
    let inBatch = false
    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed || trimmed.startsWith('//')) continue
      if (trimmed === 'BEGIN') { inBatch = true; batch = [line]; continue }
      if (trimmed === 'COMMIT' && inBatch) { batch.push(line); queries.push(batch.join('\n')); batch = []; inBatch = false; continue }
      if (inBatch) { batch.push(line); continue }
      if (trimmed.startsWith('CREATE') || trimmed.startsWith('UPSERT')) queries.push(trimmed)
    }
    for (const q of queries) await useGraphStore.getState().executeQuery(q)
  }

  return (
    <>
      <div className="h-10 border-b border-border px-3 flex items-center gap-1.5 bg-card/50">
        <span className="text-sm font-semibold text-foreground mr-2">graphstore</span>
        <Separator orientation="vertical" className="h-5" />
        <DropdownMenu>
          <DropdownMenuTrigger className="inline-flex items-center gap-1.5 h-7 px-3 text-xs rounded-md hover:bg-accent text-foreground">
            <BookOpen className="w-3.5 h-3.5" /> Examples
          </DropdownMenuTrigger>
          <DropdownMenuContent className="w-72 p-2">
            {examples.map((ex) => (
              <DropdownMenuItem
                key={ex.name}
                onClick={() => loadExample(ex)}
                className="flex flex-col items-start gap-1 px-3 py-2.5 cursor-pointer"
              >
                <span className="text-sm font-medium">{ex.name}</span>
                <span className="text-xs text-muted-foreground">{ex.description}</span>
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
        <Separator orientation="vertical" className="h-5" />
        <Button variant="ghost" size="sm" className="h-7 text-xs gap-1.5" onClick={resetGraph}>
          <RotateCcw className="w-3.5 h-3.5" /> Reset
        </Button>
        <div className="flex-1" />
        <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={() => updateConfig({ isDark: !isDark })}>
          {isDark ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
        </Button>
        <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={() => setSettingsOpen(true)}>
          <Settings className="w-3.5 h-3.5" />
        </Button>
      </div>
      <SettingsDialog open={settingsOpen} onOpenChange={setSettingsOpen} />
    </>
  )
}
