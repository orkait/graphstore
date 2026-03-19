import {
  forceSimulation,
  forceLink,
  forceManyBody,
  forceCenter,
  forceCollide,
  type Simulation,
  type SimulationLinkDatum,
} from 'd3-force'
import { forceCluster, type ClusterNode } from './clusterForce'
import type { Node, Edge } from '@xyflow/react'

// All force values are 0-100 normalized. This maps them to physics values.
export interface ForceLayoutOptions {
  clusterStrength: number  // 0-100
  repelStrength: number    // 0-100
  centerForce: number      // 0-100
  linkForce: number        // 0-100
  linkDistance: number      // 0-100
  width: number
  height: number
}

// Map 0-100 slider → actual physics value (linear interpolation)
function lerp(min: number, max: number, t: number): number {
  return min + (max - min) * (t / 100)
}

function mapForces(opts: ForceLayoutOptions) {
  return {
    center: lerp(0.01, 0.5, opts.centerForce),      // 0→0.01, 100→0.5
    repel: lerp(50, 500, opts.repelStrength),         // 0→50, 100→500
    link: lerp(0.05, 1.0, opts.linkForce),            // 0→0.05, 100→1.0
    distance: lerp(50, 400, opts.linkDistance),        // 0→50, 100→400
    cluster: lerp(0, 0.5, opts.clusterStrength),      // 0→0, 100→0.5
  }
}

export interface ForceNode extends ClusterNode {
  rfNode: Node
}

interface ForceLink extends SimulationLinkDatum<ForceNode> {
  source: ForceNode | string
  target: ForceNode | string
}

// ---------------------------------------------------------------------------
// Live simulation — Obsidian-style elastic physics.
//
// Performance:
//  - Tick callback reuses node array, only updates position objects
//  - Throttled: skips ticks when positions haven't changed enough
//  - Barnes-Hut approximation via d3 manyBody (default theta 0.9)
// ---------------------------------------------------------------------------
export function createLiveSimulation(
  rfNodes: Node[],
  rfEdges: Edge[],
  opts: ForceLayoutOptions,
  onTick: (nodes: Node[]) => void,
  existingPositions?: Map<string, { x: number; y: number }>
): { simulation: Simulation<ForceNode, ForceLink>; nodeMap: Map<string, ForceNode> } {
  const { width, height } = opts
  const f = mapForces(opts)
  const cx = width / 2
  const cy = height / 2

  // Use existing positions if available (highlight changes), scatter otherwise
  const startRadius = Math.min(width, height) * 0.35
  const forceNodes: ForceNode[] = rfNodes.map((n, i) => {
    const existing = existingPositions?.get(n.id)
    const angle = (i / rfNodes.length) * 2 * Math.PI
    return {
      id: n.id,
      kind: (n.data?.kind as string) || 'default',
      x: existing?.x ?? (cx + Math.cos(angle) * startRadius + (Math.random() - 0.5) * 10),
      y: existing?.y ?? (cy + Math.sin(angle) * startRadius + (Math.random() - 0.5) * 10),
      rfNode: n,
    }
  })

  const nodeMap = new Map(forceNodes.map(n => [n.id, n]))

  const forceLinks: ForceLink[] = rfEdges
    .filter(e => nodeMap.has(e.source) && nodeMap.has(e.target))
    .map(e => ({ source: e.source, target: e.target }))

  // Pre-allocate output array — reuse on every tick instead of .map()
  const outputNodes: Node[] = forceNodes.map(fn => ({
    ...fn.rfNode,
    position: { x: fn.x || 0, y: fn.y || 0 },
  }))

  // Throttle: only push updates when positions change meaningfully
  let tickCount = 0

  const simulation: Simulation<ForceNode, ForceLink> = forceSimulation(forceNodes)
    .force('center', forceCenter(cx, cy).strength(f.center))
    .force('charge', forceManyBody<ForceNode>().strength(-f.repel).theta(0.9))
    .force('link', forceLink<ForceNode, ForceLink>(forceLinks)
      .id(d => d.id).distance(f.distance).strength(f.link))
    .force('collide', forceCollide<ForceNode>(55).strength(0.7).iterations(2))
    .force('cluster', forceCluster(f.cluster))
    .alpha(existingPositions ? 0.1 : 1)
    .alphaMin(0.0001)
    .alphaDecay(existingPositions ? 0.02 : 0.005)
    .alphaTarget(0.02)
    .velocityDecay(0.15)
    .on('tick', () => {
      tickCount++
      // During initial settling (high alpha), update every tick.
      // Once cooled, update every 3rd tick — positions barely change.
      if (simulation.alpha() < 0.05 && tickCount % 3 !== 0) return

      // Mutate pre-allocated output — no new objects
      for (let i = 0; i < forceNodes.length; i++) {
        outputNodes[i].position.x = forceNodes[i].x || 0
        outputNodes[i].position.y = forceNodes[i].y || 0
      }
      onTick([...outputNodes])  // shallow copy so React detects change
    })

  return { simulation, nodeMap }
}

// ---------------------------------------------------------------------------
// Update forces on a running simulation (sliders changed).
// ---------------------------------------------------------------------------
export function updateSimulationForces(
  simulation: Simulation<ForceNode, ForceLink>,
  opts: ForceLayoutOptions
) {
  const f = mapForces(opts)

  const center = simulation.force('center') as ReturnType<typeof forceCenter> | undefined
  if (center) center.x(opts.width / 2).y(opts.height / 2).strength(f.center)

  const charge = simulation.force('charge') as ReturnType<typeof forceManyBody> | undefined
  if (charge) charge.strength(-f.repel)

  const link = simulation.force('link') as ReturnType<typeof forceLink> | undefined
  if (link) link.distance(f.distance).strength(f.link)

  simulation.force('cluster', forceCluster(f.cluster))

  simulation.alpha(0.6).alphaTarget(0.02).restart()
}
