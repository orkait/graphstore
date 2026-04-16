import type { SimulationNodeDatum } from 'd3-force'

export interface ClusterNode extends SimulationNodeDatum {
  id: string
  kind: string
}

/**
 * Custom d3-force that attracts nodes toward their kind's centroid.
 * Strength controls how strongly nodes pull toward their kind-group center.
 */
export function forceCluster(strength: number = 0.5) {
  let nodes: ClusterNode[] = []

  function force(alpha: number) {
    // Compute centroids per kind
    const centroids = new Map<string, { x: number; y: number; count: number }>()

    for (const node of nodes) {
      const kind = node.kind || 'default'
      const c = centroids.get(kind) || { x: 0, y: 0, count: 0 }
      c.x += node.x || 0
      c.y += node.y || 0
      c.count++
      centroids.set(kind, c)
    }

    for (const c of centroids.values()) {
      c.x /= c.count
      c.y /= c.count
    }

    // Apply force toward centroid
    for (const node of nodes) {
      const kind = node.kind || 'default'
      const c = centroids.get(kind)
      if (!c || c.count <= 1) continue
      const dx = (c.x - (node.x || 0)) * strength * 0.04 * alpha
      const dy = (c.y - (node.y || 0)) * strength * 0.04 * alpha
      node.vx = (node.vx || 0) + dx
      node.vy = (node.vy || 0) + dy
    }
  }

  force.initialize = (n: ClusterNode[]) => {
    nodes = n
  }

  force.strength = (s: number) => {
    strength = s
    return force
  }

  return force
}
