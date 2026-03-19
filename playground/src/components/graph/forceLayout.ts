import {
  forceSimulation,
  forceLink,
  forceManyBody,
  forceCenter,
  forceCollide,
  type Simulation,
  type SimulationNodeDatum,
  type SimulationLinkDatum,
} from 'd3-force'
import { forceCluster, type ClusterNode } from './clusterForce'
import type { Node, Edge } from '@xyflow/react'

export interface ForceLayoutOptions {
  clusterStrength: number
  repelStrength: number
  width: number
  height: number
}

interface ForceNode extends ClusterNode {
  rfNode: Node  // reference to original React Flow node
}

interface ForceLink extends SimulationLinkDatum<ForceNode> {
  source: ForceNode | string
  target: ForceNode | string
}

/**
 * Run a d3-force simulation and return positioned React Flow nodes.
 * Simulation runs synchronously until convergence (alpha < alphaMin).
 */
export function applyForceLayout(
  rfNodes: Node[],
  rfEdges: Edge[],
  opts: ForceLayoutOptions
): Node[] {
  if (rfNodes.length === 0) return rfNodes

  const { clusterStrength, repelStrength, width, height } = opts

  // Create simulation nodes
  const forceNodes: ForceNode[] = rfNodes.map((n) => ({
    id: n.id,
    kind: (n.data?.kind as string) || 'default',
    x: n.position.x || Math.random() * width,
    y: n.position.y || Math.random() * height,
    rfNode: n,
  }))

  const nodeMap = new Map(forceNodes.map(n => [n.id, n]))

  // Create simulation links
  const forceLinks: ForceLink[] = rfEdges
    .filter(e => nodeMap.has(e.source) && nodeMap.has(e.target))
    .map(e => ({
      source: e.source,
      target: e.target,
    }))

  // Build simulation
  const simulation: Simulation<ForceNode, ForceLink> = forceSimulation(forceNodes)
    .force('center', forceCenter(width / 2, height / 2).strength(0.05))
    .force('charge', forceManyBody<ForceNode>().strength(-repelStrength))
    .force('link', forceLink<ForceNode, ForceLink>(forceLinks).id(d => d.id).distance(100).strength(0.3))
    .force('collide', forceCollide<ForceNode>(45))
    .force('cluster', forceCluster(clusterStrength))
    .stop()

  // Run until convergence (standard d3-force synchronous pattern)
  const alphaMin = simulation.alphaMin()
  const alphaDecay = simulation.alphaDecay()
  const numIterations = Math.ceil(Math.log(alphaMin) / Math.log(1 - alphaDecay))
  for (let i = 0; i < numIterations; i++) {
    simulation.tick()
  }

  // Map positions back to React Flow nodes
  return forceNodes.map(fn => ({
    ...fn.rfNode,
    position: { x: fn.x || 0, y: fn.y || 0 },
  }))
}

/**
 * Create a live d3-force simulation for drag interactions.
 * Returns the simulation instance so the caller can reheat on drag.
 */
export function createForceSimulation(
  rfNodes: Node[],
  rfEdges: Edge[],
  opts: ForceLayoutOptions,
  onTick: (nodes: Node[]) => void
): { simulation: Simulation<ForceNode, ForceLink>; nodeMap: Map<string, ForceNode> } {
  const { clusterStrength, repelStrength, width, height } = opts

  const forceNodes: ForceNode[] = rfNodes.map((n) => ({
    id: n.id,
    kind: (n.data?.kind as string) || 'default',
    x: n.position.x || Math.random() * width,
    y: n.position.y || Math.random() * height,
    rfNode: n,
  }))

  const nodeMap = new Map(forceNodes.map(n => [n.id, n]))

  const forceLinks: ForceLink[] = rfEdges
    .filter(e => nodeMap.has(e.source) && nodeMap.has(e.target))
    .map(e => ({
      source: e.source,
      target: e.target,
    }))

  const simulation = forceSimulation(forceNodes)
    .force('center', forceCenter(width / 2, height / 2).strength(0.05))
    .force('charge', forceManyBody<ForceNode>().strength(-repelStrength))
    .force('link', forceLink<ForceNode, ForceLink>(forceLinks).id(d => d.id).distance(100).strength(0.3))
    .force('collide', forceCollide<ForceNode>(45))
    .force('cluster', forceCluster(clusterStrength))
    .on('tick', () => {
      const positioned = forceNodes.map(fn => ({
        ...fn.rfNode,
        position: { x: fn.x || 0, y: fn.y || 0 },
      }))
      onTick(positioned)
    })

  return { simulation, nodeMap }
}
