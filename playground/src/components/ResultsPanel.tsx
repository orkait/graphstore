import { useGraphStore, type ResultEntry } from '@/hooks/useGraphStore'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Trash2, ChevronDown, ChevronRight } from 'lucide-react'
import { useState } from 'react'

function ResultCard({ entry }: { entry: ResultEntry }) {
  const [expanded, setExpanded] = useState(false)
  const isError = entry.error != null
  const result = entry.result

  const renderTable = () => {
    if (!result?.data) return <span className="text-muted-foreground">No data</span>

    // Path: array of string IDs
    if (result.kind === 'path' && Array.isArray(result.data)) {
      return (
        <div className="text-xs flex items-center gap-1 flex-wrap">
          {result.data.map((id: string, i: number) => (
            <span key={i} className="flex items-center gap-1">
              <span className="px-1.5 py-0.5 rounded bg-primary/10 text-primary">{id}</span>
              {i < result.data.length - 1 && <span className="text-muted-foreground">→</span>}
            </span>
          ))}
        </div>
      )
    }

    // Paths: array of arrays of string IDs
    if (result.kind === 'paths' && Array.isArray(result.data)) {
      return (
        <div className="space-y-1">
          {result.data.map((path: string[], pi: number) => (
            <div key={pi} className="text-xs flex items-center gap-1 flex-wrap">
              <span className="text-muted-foreground mr-1">#{pi + 1}</span>
              {path.map((id: string, i: number) => (
                <span key={i} className="flex items-center gap-1">
                  <span className="px-1.5 py-0.5 rounded bg-primary/10 text-primary">{id}</span>
                  {i < path.length - 1 && <span className="text-muted-foreground">→</span>}
                </span>
              ))}
            </div>
          ))}
        </div>
      )
    }

    // Distance: single number
    if (result.kind === 'distance') {
      const d = result.data as number
      return <span className="text-sm">{d === -1 ? 'No path found' : `Distance: ${d}`}</span>
    }

    // Match: array of binding objects
    if (result.kind === 'match' && Array.isArray(result.data)) {
      if (result.data.length === 0)
        return <span className="text-muted-foreground">No matches</span>
      const allKeys = new Set<string>()
      result.data.forEach((b: Record<string, string>) => Object.keys(b).forEach(k => allKeys.add(k)))
      const keys = [...allKeys]
      return (
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border">
              {keys.map(k => <th key={k} className="text-left p-1 text-muted-foreground">{k}</th>)}
            </tr>
          </thead>
          <tbody>
            {result.data.map((row: Record<string, string>, i: number) => (
              <tr key={i} className="border-b border-border/50">
                {keys.map(k => <td key={k} className="p-1">{row[k] || ''}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      )
    }

    // Array of objects (nodes, edges)
    if (Array.isArray(result.data)) {
      if (result.data.length === 0)
        return <span className="text-muted-foreground">Empty result</span>
      // Check first item is an object (not a string/number)
      if (typeof result.data[0] === 'object' && result.data[0] !== null) {
        const keys = Object.keys(result.data[0])
        return (
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-border">
                {keys.map((k) => (
                  <th key={k} className="text-left p-1 text-muted-foreground">{k}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.data.map((row: Record<string, unknown>, i: number) => (
                <tr key={i} className="border-b border-border/50">
                  {keys.map((k) => (
                    <td key={k} className="p-1">{JSON.stringify(row[k])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )
      }
      // Array of primitives
      return <span className="text-sm">{JSON.stringify(result.data)}</span>
    }

    // Single object (stats, node, plan, etc.)
    if (typeof result.data === 'object') {
      const obj = result.data as Record<string, unknown>
      const keys = Object.keys(obj)
      return (
        <table className="w-full text-xs">
          <tbody>
            {keys.map((k) => (
              <tr key={k} className="border-b border-border/50">
                <td className="p-1 text-muted-foreground w-32">{k}</td>
                <td className="p-1">{JSON.stringify(obj[k])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )
    }

    return <span className="text-sm">{JSON.stringify(result.data)}</span>
  }

  return (
    <div className="border border-border rounded-md mb-2 bg-card/50">
      <div
        className="flex items-center gap-2 p-2 cursor-pointer hover:bg-accent/50"
        onClick={() => setExpanded(!expanded)}
      >
        {expanded ? (
          <ChevronDown className="w-3 h-3" />
        ) : (
          <ChevronRight className="w-3 h-3" />
        )}
        <code className="text-xs text-muted-foreground truncate flex-1">
          {entry.query}
        </code>
        <Badge
          variant={isError ? 'destructive' : 'default'}
          className="text-[10px]"
        >
          {isError ? 'error' : result?.kind || 'ok'}
        </Badge>
        {result && (
          <span className="text-[10px] text-muted-foreground">
            {result.count} &middot; {(result.elapsed_us / 1000).toFixed(1)}ms
          </span>
        )}
      </div>
      {expanded && (
        <div className="p-2 border-t border-border">
          {isError ? (
            <div className="text-destructive text-xs font-mono">{entry.error}</div>
          ) : (
            <Tabs defaultValue="table" className="w-full">
              <TabsList className="h-7">
                <TabsTrigger value="table" className="text-xs h-6">Table</TabsTrigger>
                <TabsTrigger value="json" className="text-xs h-6">JSON</TabsTrigger>
              </TabsList>
              <TabsContent value="table" className="mt-2">{renderTable()}</TabsContent>
              <TabsContent value="json" className="mt-2">
                <pre className="text-xs text-foreground/80 whitespace-pre-wrap overflow-auto max-h-48">
                  {JSON.stringify(result, null, 2)}
                </pre>
              </TabsContent>
            </Tabs>
          )}
        </div>
      )}
    </div>
  )
}

export function ResultsPanel() {
  const results = useGraphStore((s) => s.results)
  const clearResults = useGraphStore((s) => s.clearResults)
  return (
    <div className="h-full flex flex-col bg-background">
      <div className="flex items-center justify-between px-2 py-1 border-b border-border">
        <span className="text-xs text-muted-foreground">Results ({results.length})</span>
        <Button variant="ghost" size="sm" onClick={clearResults} className="h-6 px-2">
          <Trash2 className="w-3 h-3" />
        </Button>
      </div>
      <ScrollArea className="flex-1 p-2">
        {results.map((entry) => (
          <ResultCard key={entry.id} entry={entry} />
        ))}
        {results.length === 0 && (
          <div className="text-muted-foreground/50 text-xs text-center mt-8">
            Run a query to see results
          </div>
        )}
      </ScrollArea>
    </div>
  )
}
