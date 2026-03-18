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

export default function App() {
  return (
    <div className="h-screen flex flex-col bg-background text-foreground">
      <Toolbar />
      <ResizablePanelGroup orientation="horizontal" className="flex-1">
        <ResizablePanel defaultSize={40} minSize={20}>
          <ResizablePanelGroup orientation="vertical">
            <ResizablePanel defaultSize={65} minSize={20}>
              <EditorPanel />
            </ResizablePanel>
            <ResizableHandle withHandle />
            <ResizablePanel defaultSize={35} minSize={10}>
              <ResultsPanel />
            </ResizablePanel>
          </ResizablePanelGroup>
        </ResizablePanel>
        <ResizableHandle withHandle />
        <ResizablePanel defaultSize={60} minSize={30}>
          <div className="h-full flex flex-col">
            <div className="flex-1">
              <GraphPanel />
            </div>
            <StatsBar />
          </div>
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  )
}
