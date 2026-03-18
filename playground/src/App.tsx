import {
  ResizablePanelGroup,
  ResizablePanel,
  ResizableHandle,
} from '@/components/ui/resizable'
import { EditorPanel } from '@/components/EditorPanel'
import { ResultsPanel } from '@/components/ResultsPanel'
import { GraphPanel } from '@/components/GraphPanel'
import { Toolbar } from '@/components/Toolbar'
import { StatsBar } from '@/components/StatsBar'
import { Toaster } from '@/components/ui/sonner'
import { useGraphStore } from '@/hooks/useGraphStore'
import { useEffect } from 'react'
import { Card } from '@/components/ui/card'
import { api } from '@/api/client'

export default function App() {
  const refreshGraph = useGraphStore((s) => s.refreshGraph)
  const executeAll = useGraphStore((s) => s.executeAll)
  const isDark = useGraphStore((s) => s.config.isDark)

  useEffect(() => {
    // 1. Sync the persisted Zustand config (Memory Ceiling, etc) to the backend server
    const { ceilingMb, costThreshold } = useGraphStore.getState().config
    api.updateConfig({ ceiling_mb: ceilingMb, cost_threshold: costThreshold }).catch(console.error)

    // 2. Fetch the graph and conditionally populate the editor example
    refreshGraph().then(() => {
      const g = useGraphStore.getState().graph
      if (g.nodes.length === 0) executeAll()
    })
  }, [refreshGraph, executeAll])
  useEffect(() => { document.documentElement.classList.toggle('dark', isDark) }, [isDark])

  return (
    <div className="h-screen flex flex-col bg-background text-foreground">
      <Toaster position="top-right" />
      <Toolbar />
      <ResizablePanelGroup orientation="horizontal" className="flex-1">
        <ResizablePanel defaultSize={40} minSize={20}>
          <div className="h-full p-2">
            <Card className="h-full rounded-lg overflow-hidden flex flex-col border border-border bg-card">
              <ResizablePanelGroup orientation="vertical" className="h-full">
                <ResizablePanel defaultSize={65} minSize={20}>
                  <EditorPanel />
                </ResizablePanel>
                <ResizableHandle withHandle className="bg-border" />
                <ResizablePanel defaultSize={35} minSize={10}>
                  <ResultsPanel />
                </ResizablePanel>
              </ResizablePanelGroup>
            </Card>
          </div>
        </ResizablePanel>
        <ResizableHandle withHandle />
        <ResizablePanel defaultSize={60} minSize={30}>
          <div className="h-full flex flex-col p-2">
            <div className="flex-1 min-h-0">
              <GraphPanel />
            </div>
            <StatsBar />
          </div>
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  )
}
