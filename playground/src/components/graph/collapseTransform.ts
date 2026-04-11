import type { GraphNode, GraphEdge } from '@/api/client'

export interface GroupNodeData {
  isGroup: true
  parentId: string
  kind: string
  childCount: number
  childIds: string[]
  matchCount: number
}

export interface CollapseResult {
  nodes: GraphNode[]
  edges: GraphEdge[]
  /** Maps group node ID -> GroupNodeData for rendering */
  groupMeta: Map<string, GroupNodeData>
}

/**
 * Build a parent->children map from edges.
 * A "child" of parentId is any node that is a target of an edge from parentId.
 */
function buildChildrenMap(
  nodes: GraphNode[],
  edges: GraphEdge[]
): Map<string, GraphNode[]> {
  const nodeMap = new Map(nodes.map(n => [n.id, n]))
  const children = new Map<string, GraphNode[]>()
  for (const e of edges) {
    const child = nodeMap.get(e.target)
    if (!child) continue
    const list = children.get(e.source) || []
    list.push(child)
    children.set(e.source, list)
  }
  return children
}

export function groupNodeId(parentId: string, kind: string): string {
  return `group:${parentId}:${kind}`
}

/**
 * Transform the graph by collapsing high-fanout children into kind-groups.
 *
 * For each parent with children > threshold:
 *   - Children whose kind-group is NOT in expandedGroups are replaced
 *     by a single group node per kind.
 *   - Edges are rewritten accordingly.
 */
export function collapseTransform(
  nodes: GraphNode[],
  edges: GraphEdge[],
  threshold: number,
  expandedGroups: Record<string, string[]>,
  highlightedNodeIds?: Set<string>
): CollapseResult {
  if (threshold <= 0) {
    return { nodes, edges, groupMeta: new Map() }
  }

  const childrenMap = buildChildrenMap(nodes, edges)
  const groupMeta = new Map<string, GroupNodeData>()

  // Set of node IDs that get collapsed (hidden)
  const collapsedNodeIds = new Set<string>()
  // Group nodes to add
  const groupNodes: GraphNode[] = []
  // Reverse lookup: nodeId -> groupId (built during collapse, used for edge rewriting)
  const nodeToGroup = new Map<string, string>()

  for (const [parentId, children] of childrenMap) {
    if (children.length <= threshold) continue

    const expanded = expandedGroups[parentId] || []

    // Group children by kind
    const byKind = new Map<string, GraphNode[]>()
    for (const child of children) {
      const kind = child.kind || 'default'
      const list = byKind.get(kind) || []
      list.push(child)
      byKind.set(kind, list)
    }

    for (const [kind, kindChildren] of byKind) {
      if (expanded.includes(kind)) continue // this kind-group is expanded, leave nodes alone

      // Hybrid auto-expand: if <= 5 children are highlighted, leave expanded
      if (highlightedNodeIds && highlightedNodeIds.size > 0) {
        const matchCount = kindChildren.filter(c => highlightedNodeIds.has(c.id)).length
        if (matchCount > 0 && matchCount <= 5) continue // auto-expand for small match sets
      }

      const gid = groupNodeId(parentId, kind)
      const childIds = kindChildren.map(c => c.id)
      childIds.forEach(id => {
        collapsedNodeIds.add(id)
        nodeToGroup.set(id, gid)
      })

      const matchCount = highlightedNodeIds
        ? kindChildren.filter(c => highlightedNodeIds.has(c.id)).length
        : 0

      const meta: GroupNodeData = {
        isGroup: true,
        parentId,
        kind,
        childCount: kindChildren.length,
        childIds,
        matchCount,
      }
      groupMeta.set(gid, meta)
      groupNodes.push({ id: gid, kind })
    }
  }

  // Build output nodes: keep non-collapsed + add group nodes
  const outNodes = nodes.filter(n => !collapsedNodeIds.has(n.id)).concat(groupNodes)

  // Rewrite edges
  const outEdges: GraphEdge[] = []
  // Track cross-group edge counts: "gidA->gidB" -> count
  const crossGroupCounts = new Map<string, { count: number; kind: string }>()

  for (const e of edges) {
    const srcCollapsed = collapsedNodeIds.has(e.source)
    const tgtCollapsed = collapsedNodeIds.has(e.target)

    if (!srcCollapsed && !tgtCollapsed) {
      outEdges.push(e)
    } else if (!srcCollapsed && tgtCollapsed) {
      const gid = nodeToGroup.get(e.target)
      if (gid) {
        outEdges.push({ ...e, target: gid, kind: e.kind, groupChildCount: groupMeta.get(gid)?.childCount })
      }
    } else if (srcCollapsed && !tgtCollapsed) {
      const gid = nodeToGroup.get(e.source)
      if (gid) {
        outEdges.push({ ...e, source: gid, kind: e.kind })
      }
    } else {
      // Both collapsed - aggregate as cross-group edge
      const srcGid = nodeToGroup.get(e.source)
      const tgtGid = nodeToGroup.get(e.target)
      if (srcGid && tgtGid && srcGid !== tgtGid) {
        const key = `${srcGid}->${tgtGid}`
        const existing = crossGroupCounts.get(key)
        if (existing) {
          existing.count++
        } else {
          crossGroupCounts.set(key, { count: 1, kind: e.kind })
        }
      }
    }
  }

  // Add cross-group edges
  for (const [key, { count, kind }] of crossGroupCounts) {
    const [source, target] = key.split('->')
    outEdges.push({
      source,
      target,
      kind: `cross-group`,
      crossGroupCount: count,
      originalKind: kind,
    } as GraphEdge)
  }

  // Deduplicate edges (parent->group may appear multiple times)
  const seen = new Set<string>()
  const dedupedEdges = outEdges.filter(e => {
    const key = `${e.source}->${e.target}:${e.kind}`
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })

  return { nodes: outNodes, edges: dedupedEdges, groupMeta }
}
