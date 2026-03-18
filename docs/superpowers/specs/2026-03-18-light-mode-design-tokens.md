# Design Spec: Dual-Mode Design Token System (Warm Pastel Light + Dark)

**Date:** 2026-03-18
**Status:** Approved

---

## Problem

The playground has several color system issues:

1. `CustomNode.tsx` hardcodes dark-mode colors in JS objects (`KIND_STYLES`, `FALLBACK_STYLES`). These render incorrectly in light mode.
2. `CustomEdge.tsx` hardcodes dark-mode colors (`#60a5fa`, `#333`, `#444`, `rgba(0,0,0,0.6)`) for highlight, dimmed, and label states.
3. `EditorPanel.tsx` hardcodes `theme="dark"` for CodeMirror.
4. `GraphPanel.tsx` hardcodes `maskColor="rgba(0,0,0,0.7)"` for MiniMap.
5. Light mode UI shell (`:root` in `index.css`) is pure achromatic white — harsh, no warmth.
6. `index.css` already defines `--kind-*` and `--graph-*` tokens for light mode but they are **not used** by any component (all components read from JS constants instead). These existing values will be **replaced** by the new values in this spec.

---

## Goal

A complete, locked dual-mode design token system:

- **Light mode**: warm pastel palette — subtle amber undertone on all surfaces, desaturated hue-coded graph nodes
- **Dark mode**: existing dark colors stabilized by moving them from JS into CSS tokens
- All color values live in CSS custom properties in `index.css`; no hardcoded hex/rgb in component files
- Exception: `EDGE_COLORS` in `CustomEdge.tsx` and `getEdgeColor` in `GraphPanel.tsx` (used for MiniMap `nodeColor` callback) remain as JS constants — reading CSS vars inside a render callback via `getComputedStyle` has performance implications, and MiniMap coloring is cosmetic only. These will be updated to light-mode-appropriate values but remain in JS.

---

## Token Architecture

### Naming Convention

```
--background, --card, --border, ...            UI shell (shadcn standard)
--kind-{name}-{bg|border|text|badge}           Node kind colors
--graph-{property}                              Graph interaction colors
```

Node kinds: `function`, `class`, `module`, `service`, `database`, `queue`, `default`
Fallback kinds (for unknown node kinds, selected by hash mod 3): `fallback-0`, `fallback-1`, `fallback-2`

`--kind-{}-badge` is a **net-new sub-token** — it does not exist in the current `index.css`. It must be added to both `:root` and `.dark {}`.

`--graph-highlight-border` and `--graph-highlight-shadow` are also **entirely absent from `index.css` currently**. They must be added to both `:root` and `.dark {}`.

---

## Token Values

### Light Mode `:root` — UI Shell (replaces current achromatic values)

| Token | Value | Notes |
|---|---|---|
| `--background` | `oklch(0.987 0.008 80)` | warm off-white |
| `--foreground` | `oklch(0.220 0.015 55)` | warm near-black |
| `--card` | `oklch(0.972 0.012 75)` | slightly warmer than bg |
| `--card-foreground` | `oklch(0.220 0.015 55)` | |
| `--popover` | `oklch(0.972 0.012 75)` | same as card |
| `--popover-foreground` | `oklch(0.220 0.015 55)` | |
| `--primary` | `oklch(0.220 0.015 55)` | warm near-black |
| `--primary-foreground` | `oklch(0.987 0.008 80)` | |
| `--secondary` | `oklch(0.945 0.018 72)` | visible warm tint |
| `--secondary-foreground` | `oklch(0.220 0.015 55)` | |
| `--muted` | `oklch(0.945 0.018 72)` | same as secondary |
| `--muted-foreground` | `oklch(0.520 0.018 65)` | warm medium gray |
| `--accent` | `oklch(0.945 0.018 72)` | same as secondary |
| `--accent-foreground` | `oklch(0.220 0.015 55)` | |
| `--destructive` | `oklch(0.577 0.245 27.325)` | unchanged |
| `--border` | `oklch(0.878 0.016 70)` | warm soft divider |
| `--input` | `oklch(0.878 0.016 70)` | same as border |
| `--ring` | `oklch(0.650 0.020 65)` | warm focus ring |
| `--sidebar` | `oklch(0.958 0.016 70)` | slightly deeper than card |
| `--sidebar-foreground` | `oklch(0.220 0.015 55)` | |
| `--sidebar-primary` | `oklch(0.220 0.015 55)` | |
| `--sidebar-primary-foreground` | `oklch(0.987 0.008 80)` | |
| `--sidebar-accent` | `oklch(0.945 0.018 72)` | |
| `--sidebar-accent-foreground` | `oklch(0.220 0.015 55)` | |
| `--sidebar-border` | `oklch(0.878 0.016 70)` | |
| `--sidebar-ring` | `oklch(0.650 0.020 65)` | |

### Light Mode `:root` — Node Kind Tokens (replaces existing `--kind-*` values; adds `--kind-*-badge`)

| Kind | `--kind-{}-bg` | `--kind-{}-border` | `--kind-{}-text` | `--kind-{}-badge` |
|---|---|---|---|---|
| function | `#dce8f5` | `#7ba3cc` | `#2a5580` | `#7ba3cc` |
| class | `#daeee3` | `#6aad84` | `#256640` | `#6aad84` |
| module | `#ece0f5` | `#9b78c8` | `#4d2880` | `#9b78c8` |
| service | `#f5e8d8` | `#c88255` | `#7a3a12` | `#c88255` |
| database | `#f5dede` | `#c87575` | `#7a2020` | `#c87575` |
| queue | `#f5efd8` | `#c8a840` | `#7a5500` | `#c8a840` |
| default | `#d8eff5` | `#5aacc0` | `#1a5f6e` | `#5aacc0` |
| fallback-0 (rose) | `#f5d8e4` | `#c56b8a` | `#6a1d35` | `#c56b8a` |
| fallback-1 (indigo) | `#dde0f5` | `#7577c8` | `#2a2d80` | `#7577c8` |
| fallback-2 (amber) | `#f5ebd8` | `#c89650` | `#7a4d00` | `#c89650` |

### Light Mode `:root` — Graph Interaction Tokens (replaces existing `--graph-*` values; adds highlight tokens)

| Token | Value |
|---|---|
| `--graph-highlight-border` | `#6495d4` *(net-new)* |
| `--graph-highlight-shadow` | `rgba(100,149,212,0.4)` *(net-new)* |
| `--graph-node-dimmed-bg` | `#ede8e0` |
| `--graph-node-dimmed-border` | `#c8c0b5` |
| `--graph-node-dimmed-text` | `#a09888` |
| `--graph-edge-dimmed` | `#c8c0b5` *(same value in dark mode — a light/medium gray reads correctly as "faded" in both themes)* |
| `--graph-edge-label-bg` | `rgba(254,252,248,0.88)` |
| `--graph-edge-label-text` | `#8a7a68` |
| `--graph-degree-bg` | `rgba(0,0,0,0.04)` |
| `--graph-degree-text` | `#8a7a68` |
| `--graph-bg-dots` | `#d4c8b8` |

---

### Dark Mode `.dark {}` — Node Kind Tokens (moved from JS `KIND_STYLES`; adds `--kind-*-badge`)

Note: the current `.dark {}` block in `index.css` has **zero** `--kind-*` or `--graph-*` entries — all of these are net-new additions to `.dark {}`, not replacements.

| Kind | `--kind-{}-bg` | `--kind-{}-border` | `--kind-{}-text` | `--kind-{}-badge` |
|---|---|---|---|---|
| function | `#1e3a5f` | `#3b82f6` | `#93c5fd` | `#3b82f6` |
| class | `#14532d` | `#22c55e` | `#86efac` | `#22c55e` |
| module | `#3b0764` | `#a855f7` | `#d8b4fe` | `#a855f7` |
| service | `#431407` | `#f97316` | `#fdba74` | `#f97316` |
| database | `#450a0a` | `#ef4444` | `#fca5a5` | `#ef4444` |
| queue | `#422006` | `#eab308` | `#fde047` | `#eab308` |
| default | `#164e63` | `#06b6d4` | `#67e8f9` | `#06b6d4` |
| fallback-0 (rose) | `#4a0a1a` | `#f43f5e` | `#fda4af` | `#f43f5e` |
| fallback-1 (fuchsia) | `#4a044e` | `#d946ef` | `#f0abfc` | `#d946ef` |
| fallback-2 (indigo) | `#1e1b4b` | `#6366f1` | `#a5b4fc` | `#6366f1` |

Note: dark fallback-0 is changed from teal (previously a duplicate of `default`) to rose, making all 3 fallbacks visually distinct from each other and from the 7 named kinds.

### Dark Mode `.dark {}` — Graph Interaction Tokens (moved from JS; adds highlight tokens)

| Token | Value |
|---|---|
| `--graph-highlight-border` | `#60a5fa` *(net-new in CSS)* |
| `--graph-highlight-shadow` | `rgba(96,165,250,0.5)` *(net-new in CSS)* |
| `--graph-node-dimmed-bg` | `#1a1a1a` |
| `--graph-node-dimmed-border` | `#333333` |
| `--graph-node-dimmed-text` | `#555555` |
| `--graph-edge-dimmed` | `#d4d4d8` |
| `--graph-edge-label-bg` | `rgba(14,14,14,0.88)` |
| `--graph-edge-label-text` | `#a1a1aa` |
| `--graph-degree-bg` | `rgba(255,255,255,0.07)` |
| `--graph-degree-text` | `#888888` |
| `--graph-bg-dots` | `#3f3f46` |

---

## Component Changes

### `useGraphStore.ts`
Add `isDark: boolean` to the `Config` interface (default `true`). Use the existing `updateConfig(partial: Partial<Config>)` pattern — no separate `setIsDark` action needed. All callsites use `updateConfig({ isDark: ... })`.

### `Toolbar.tsx`
- Remove local `isDark` state
- Read `config.isDark` from `useGraphStore`; the toggle button `onClick` changes from `setIsDark(!isDark)` to `updateConfig({ isDark: !config.isDark })`
- Keep the `useEffect(() => { document.documentElement.classList.toggle('dark', isDark) }, [isDark])` in `Toolbar.tsx` — it now reads `isDark` from the store rather than local state

### `EditorPanel.tsx`
Read `isDark` from `useGraphStore`. Replace hardcoded `theme="dark"` with `theme={isDark ? 'dark' : 'light'}`.

### `CustomNode.tsx`
Remove `KIND_STYLES`, `FALLBACK_STYLES`, and `getKindStyle()` entirely. Replace with:

- A minimal JS hash-to-name map: `const FALLBACK_NAMES = ['fallback-0', 'fallback-1', 'fallback-2']`
- A helper `getKindTokenName(kind)` that returns the kind string if it's one of the 7 named kinds (`function`, `class`, `module`, `service`, `database`, `queue`, `default`), or hashes it to a `fallback-{n}` name. Note: `default` is currently absent from `KIND_STYLES` in the source and falls to `FALLBACK_STYLES[0]`. In the new code, `default` must be explicitly included in the named-kind guard set so it resolves to `--kind-default-*` tokens rather than the fallback path.
- All inline style color values replaced with CSS var references:

```tsx
const kindName = getKindTokenName(kind)
// opacity — retain existing behavior: dimmed ? 0.2 : 1
// (the dimmed-color tokens adjust the base colors before this opacity is applied; both work together)
dimmed ? 0.2 : 1
// backgroundColor:
dimmed ? 'var(--graph-node-dimmed-bg)' : `var(--kind-${kindName}-bg)`
// borderColor:
highlighted ? 'var(--graph-highlight-border)' : dimmed ? 'var(--graph-node-dimmed-border)' : `var(--kind-${kindName}-border)`
// boxShadow — CSS var cannot have a hex alpha suffix appended (e.g. `var(--foo)44` is invalid CSS).
// Use color-mix() to apply alpha to the border token:
highlighted
  ? `0 0 16px var(--graph-highlight-shadow)`
  : dimmed ? 'none'
  : `0 0 ${6 + degree * 2}px color-mix(in srgb, var(--kind-${kindName}-border) 27%, transparent)`
// text color:
dimmed ? 'var(--graph-node-dimmed-text)' : `var(--kind-${kindName}-text)`
// badge backgroundColor — use color-mix() since var() cannot take a hex alpha suffix:
`color-mix(in srgb, var(--kind-${kindName}-badge) 13%, transparent)`
// badge color (intentional normalization: current code uses '#444' for badge and '#555' for label when dimmed — both collapse to --graph-node-dimmed-text):
dimmed ? 'var(--graph-node-dimmed-text)' : `var(--kind-${kindName}-badge)`
// badge border — dimmed uses dimmed-text token; active uses kind badge with alpha:
dimmed
  ? `1px solid var(--graph-node-dimmed-text)`
  : `1px solid color-mix(in srgb, var(--kind-${kindName}-badge) 27%, transparent)`
// degree counter color:
'var(--graph-degree-text)'
// degree counter backgroundColor:
'var(--graph-degree-bg)'
// Handle background (applies to both the top target Handle and bottom source Handle):
`var(--kind-${kindName}-border)`
// transform: scale(...) — retain as-is, no change
// borderWidth: highlighted ? '2px' : '1px' — retain as-is, no change
```

### `CustomEdge.tsx`
Replace hardcoded highlight/dimmed/label colors with CSS var references:

```tsx
// BaseEdge stroke:
highlighted ? 'var(--graph-highlight-border)' : dimmed ? 'var(--graph-edge-dimmed)' : color
// BaseEdge opacity — retain existing behavior: dimmed ? 0.15 : 0.8
// Label opacity — retain existing behavior: dimmed ? 0.15 : 1
// label color:
dimmed ? 'var(--graph-node-dimmed-text)' : 'var(--graph-edge-label-text)'
// label backgroundColor:
'var(--graph-edge-label-bg)'
```

Note: the current dimmed stroke hardcode of `#333` is **incorrect in dark mode** (it renders too dark against an already-dark background). `--graph-edge-dimmed` in dark mode is `#d4d4d8` (light gray), which correctly represents a faded-but-visible edge. This is an intentional bug fix, not just a refactor.

`EDGE_COLORS` and `getEdgeColor` remain as JS constants. Update their values to work well in both modes — these are used for the non-dimmed/non-highlighted edge stroke color. The current vivid hex values are acceptable as-is; they are visually correct in dark mode and for light mode will show as somewhat saturated lines (a minor visual inconsistency, accepted as a pragmatic trade-off to avoid `getComputedStyle` in a render callback).

### `GraphPanel.tsx`
- Read `isDark` from `useGraphStore`
- Set `maskColor={isDark ? 'rgba(0,0,0,0.7)' : 'rgba(255,255,255,0.7)'}` on MiniMap
- The `getEdgeColor` function (used for MiniMap `nodeColor` callback) is a duplicate of `CustomEdge.tsx`'s `getEdgeColor`. It remains as-is for the same performance reason (no `getComputedStyle` in callbacks). No change needed beyond the maskColor fix.
- Replace `<Background ... className="!text-border" />` with `<Background ... color="var(--graph-bg-dots)" />`. The current `!text-border` Tailwind class wires dot color through `--border`, which doesn't respond to the `--graph-bg-dots` token. The ReactFlow `Background` component's `color` prop accepts a CSS var string and applies it directly to the SVG pattern fill.

---

## Out of Scope

- Chart tokens (`--chart-1` through `--chart-5`) — not rendered in visible UI
- Animations, typography, spacing — no changes
- Dark mode UI shell values in `.dark {}` — already correct, no changes
- Edge stroke colors for non-dimmed, non-highlighted edges (`EDGE_COLORS` values) — remain as JS constants
