import { useGraphStore } from '@/hooks/useGraphStore'

export function StatsBar() {
  const graph = useGraphStore((s) => s.graph)
  const results = useGraphStore((s) => s.results)
  const lastResult = results[0]
  const elapsed = lastResult?.result?.elapsed_us
  return (
    <div className="h-7 border-t border-border px-3 flex items-center gap-4 text-[10px] text-muted-foreground bg-card/30">
      <span>Nodes: {graph.nodes.length}</span>
      <span>Edges: {graph.edges.length}</span>
      {elapsed != null && (
        <span>Last query: {(elapsed / 1000).toFixed(1)}ms</span>
      )}
    </div>
  )
}
