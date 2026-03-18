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
    if (!result?.data) return <span className="text-zinc-500">No data</span>
    if (Array.isArray(result.data)) {
      if (result.data.length === 0)
        return <span className="text-zinc-500">Empty result</span>
      const keys = Object.keys(result.data[0])
      return (
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-zinc-800">
              {keys.map((k) => (
                <th key={k} className="text-left p-1 text-zinc-400">
                  {k}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {result.data.map((row: Record<string, unknown>, i: number) => (
              <tr key={i} className="border-b border-zinc-800/50">
                {keys.map((k) => (
                  <td key={k} className="p-1">
                    {JSON.stringify(row[k])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      )
    }
    if (typeof result.data === 'object') {
      const obj = result.data as Record<string, unknown>
      const keys = Object.keys(obj)
      return (
        <table className="w-full text-xs">
          <tbody>
            {keys.map((k) => (
              <tr key={k} className="border-b border-zinc-800/50">
                <td className="p-1 text-zinc-400 w-32">{k}</td>
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
    <div className="border border-zinc-800 rounded-md mb-2 bg-zinc-900/50">
      <div
        className="flex items-center gap-2 p-2 cursor-pointer hover:bg-zinc-800/50"
        onClick={() => setExpanded(!expanded)}
      >
        {expanded ? (
          <ChevronDown className="w-3 h-3" />
        ) : (
          <ChevronRight className="w-3 h-3" />
        )}
        <code className="text-xs text-zinc-400 truncate flex-1">
          {entry.query}
        </code>
        <Badge
          variant={isError ? 'destructive' : 'default'}
          className="text-[10px]"
        >
          {isError ? 'error' : result?.kind || 'ok'}
        </Badge>
        {result && (
          <span className="text-[10px] text-zinc-500">
            {result.count} &middot; {(result.elapsed_us / 1000).toFixed(1)}ms
          </span>
        )}
      </div>
      {expanded && (
        <div className="p-2 border-t border-zinc-800">
          {isError ? (
            <div className="text-red-400 text-xs font-mono">{entry.error}</div>
          ) : (
            <Tabs defaultValue="table" className="w-full">
              <TabsList className="h-7">
                <TabsTrigger value="table" className="text-xs h-6">
                  Table
                </TabsTrigger>
                <TabsTrigger value="json" className="text-xs h-6">
                  JSON
                </TabsTrigger>
              </TabsList>
              <TabsContent value="table" className="mt-2">
                {renderTable()}
              </TabsContent>
              <TabsContent value="json" className="mt-2">
                <pre className="text-xs text-zinc-300 whitespace-pre-wrap overflow-auto max-h-48">
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
    <div className="h-full flex flex-col bg-zinc-950">
      <div className="flex items-center justify-between px-2 py-1 border-b border-zinc-800">
        <span className="text-xs text-zinc-400">
          Results ({results.length})
        </span>
        <Button variant="ghost" size="sm" onClick={clearResults} className="h-6 px-2">
          <Trash2 className="w-3 h-3" />
        </Button>
      </div>
      <ScrollArea className="flex-1 p-2">
        {results.map((entry) => (
          <ResultCard key={entry.id} entry={entry} />
        ))}
        {results.length === 0 && (
          <div className="text-zinc-600 text-xs text-center mt-8">
            Run a query to see results
          </div>
        )}
      </ScrollArea>
    </div>
  )
}
