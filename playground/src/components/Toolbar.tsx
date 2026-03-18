import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Separator } from '@/components/ui/separator'
import { Play, RotateCcw, Settings, BookOpen } from 'lucide-react'
import { useGraphStore } from '@/hooks/useGraphStore'
import { SettingsDialog } from '@/components/SettingsDialog'
import { examples } from '@/examples'
import { useState } from 'react'

export function Toolbar() {
  const executeAll = useGraphStore((s) => s.executeAll)
  const resetGraph = useGraphStore((s) => s.resetGraph)
  const setEditorContent = useGraphStore((s) => s.setEditorContent)
  const [settingsOpen, setSettingsOpen] = useState(false)

  const handleRunAll = () => {
    executeAll()
  }

  const loadExample = async (example: (typeof examples)[0]) => {
    await resetGraph()
    setEditorContent(example.script)
    // Auto-execute CREATE/UPSERT/BEGIN statements only
    const lines = example.script.split('\n')
    const queries: string[] = []
    let batch: string[] = []
    let inBatch = false
    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed || trimmed.startsWith('//')) continue
      if (trimmed === 'BEGIN') {
        inBatch = true
        batch = [line]
        continue
      }
      if (trimmed === 'COMMIT' && inBatch) {
        batch.push(line)
        queries.push(batch.join('\n'))
        batch = []
        inBatch = false
        continue
      }
      if (inBatch) {
        batch.push(line)
        continue
      }
      if (trimmed.startsWith('CREATE') || trimmed.startsWith('UPSERT'))
        queries.push(trimmed)
    }
    for (const q of queries) await useGraphStore.getState().executeQuery(q)
  }

  return (
    <>
      <div className="h-10 border-b border-zinc-800 px-3 flex items-center gap-1.5 bg-zinc-900/50">
        <span className="text-sm font-semibold text-zinc-300 mr-2">
          graphstore
        </span>
        <Separator orientation="vertical" className="h-5" />
        <DropdownMenu>
          <DropdownMenuTrigger
            className="inline-flex items-center gap-1.5 h-7 px-3 text-xs rounded-md hover:bg-zinc-800 text-zinc-300"
          >
            <BookOpen className="w-3.5 h-3.5" /> Examples
          </DropdownMenuTrigger>
          <DropdownMenuContent>
            {examples.map((ex) => (
              <DropdownMenuItem key={ex.name} onClick={() => loadExample(ex)}>
                <div>
                  <div className="text-sm">{ex.name}</div>
                  <div className="text-xs text-zinc-500">{ex.description}</div>
                </div>
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
        <Separator orientation="vertical" className="h-5" />
        <Button
          variant="ghost"
          size="sm"
          className="h-7 text-xs gap-1.5"
          onClick={handleRunAll}
        >
          <Play className="w-3.5 h-3.5" /> Run All
        </Button>
        <Separator orientation="vertical" className="h-5" />
        <Button
          variant="ghost"
          size="sm"
          className="h-7 text-xs gap-1.5"
          onClick={resetGraph}
        >
          <RotateCcw className="w-3.5 h-3.5" /> Reset
        </Button>
        <div className="flex-1" />
        <Button
          variant="ghost"
          size="sm"
          className="h-7 text-xs"
          onClick={() => setSettingsOpen(true)}
        >
          <Settings className="w-3.5 h-3.5" />
        </Button>
      </div>
      <SettingsDialog open={settingsOpen} onOpenChange={setSettingsOpen} />
    </>
  )
}
