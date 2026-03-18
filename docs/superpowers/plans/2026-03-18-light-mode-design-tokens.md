# Light Mode Design Tokens Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all hardcoded component colors with CSS custom property tokens, delivering a warm pastel light mode and a stabilized dark mode.

**Architecture:** All color values live in `index.css` (`:root` for light, `.dark` for dark). Components read tokens via CSS `var()` references. A shared `isDark` config field in `useGraphStore` lets theme-dependent components (editor, minimap) respond to the toggle.

**Tech Stack:** React, Tailwind v4, shadcn/ui, CSS custom properties (`color-mix()`), CodeMirror (`@uiw/react-codemirror`), ReactFlow (`@xyflow/react`), Zustand

**Spec:** `docs/superpowers/specs/2026-03-18-light-mode-design-tokens.md`

---

## File Map

| File | Change |
|---|---|
| `playground/src/index.css` | Replace `:root` UI shell + kind/graph tokens; add full dark-mode kind/graph token block to `.dark {}` |
| `playground/src/hooks/useGraphStore.ts` | Add `isDark: boolean` to `Config` interface; default `true` |
| `playground/src/components/Toolbar.tsx` | Remove local `isDark` state; read/write via `updateConfig({ isDark })` |
| `playground/src/components/EditorPanel.tsx` | Read `isDark` from store; pass to CodeMirror `theme` prop |
| `playground/src/components/GraphPanel.tsx` | Read `isDark` from store; dynamic `maskColor`; swap Background `className` to `color` prop |
| `playground/src/components/graph/CustomNode.tsx` | Remove `KIND_STYLES`/`FALLBACK_STYLES`; replace all inline colors with `var()` and `color-mix()` |
| `playground/src/components/graph/CustomEdge.tsx` | Replace hardcoded highlight/dimmed/label colors with `var()` references |

---

## Task 1: CSS Tokens — Light Mode UI Shell

**Files:**
- Modify: `playground/src/index.css` (`:root` block, lines 8–74)

- [ ] **Step 1: Replace the `:root` UI shell tokens**

Open `playground/src/index.css`. Replace the entire `:root` block (from `--background` through `--sidebar-ring`) with the warm pastel values. Keep `--radius` and the `--chart-*` values unchanged.

```css
:root {
    --background: oklch(0.987 0.008 80);
    --foreground: oklch(0.220 0.015 55);
    --card: oklch(0.972 0.012 75);
    --card-foreground: oklch(0.220 0.015 55);
    --popover: oklch(0.972 0.012 75);
    --popover-foreground: oklch(0.220 0.015 55);
    --primary: oklch(0.220 0.015 55);
    --primary-foreground: oklch(0.987 0.008 80);
    --secondary: oklch(0.945 0.018 72);
    --secondary-foreground: oklch(0.220 0.015 55);
    --muted: oklch(0.945 0.018 72);
    --muted-foreground: oklch(0.520 0.018 65);
    --accent: oklch(0.945 0.018 72);
    --accent-foreground: oklch(0.220 0.015 55);
    --destructive: oklch(0.577 0.245 27.325);
    --border: oklch(0.878 0.016 70);
    --input: oklch(0.878 0.016 70);
    --ring: oklch(0.650 0.020 65);
    --chart-1: oklch(0.809 0.105 251.813);
    --chart-2: oklch(0.623 0.214 259.815);
    --chart-3: oklch(0.546 0.245 262.881);
    --chart-4: oklch(0.488 0.243 264.376);
    --chart-5: oklch(0.424 0.199 265.638);
    --radius: 0.625rem;

    /* Graph node kind colors — light (warm pastel) */
    --kind-function-bg: #dce8f5;
    --kind-function-border: #7ba3cc;
    --kind-function-text: #2a5580;
    --kind-function-badge: #7ba3cc;
    --kind-class-bg: #daeee3;
    --kind-class-border: #6aad84;
    --kind-class-text: #256640;
    --kind-class-badge: #6aad84;
    --kind-module-bg: #ece0f5;
    --kind-module-border: #9b78c8;
    --kind-module-text: #4d2880;
    --kind-module-badge: #9b78c8;
    --kind-service-bg: #f5e8d8;
    --kind-service-border: #c88255;
    --kind-service-text: #7a3a12;
    --kind-service-badge: #c88255;
    --kind-database-bg: #f5dede;
    --kind-database-border: #c87575;
    --kind-database-text: #7a2020;
    --kind-database-badge: #c87575;
    --kind-queue-bg: #f5efd8;
    --kind-queue-border: #c8a840;
    --kind-queue-text: #7a5500;
    --kind-queue-badge: #c8a840;
    --kind-default-bg: #d8eff5;
    --kind-default-border: #5aacc0;
    --kind-default-text: #1a5f6e;
    --kind-default-badge: #5aacc0;
    --kind-fallback-0-bg: #f5d8e4;
    --kind-fallback-0-border: #c56b8a;
    --kind-fallback-0-text: #6a1d35;
    --kind-fallback-0-badge: #c56b8a;
    --kind-fallback-1-bg: #dde0f5;
    --kind-fallback-1-border: #7577c8;
    --kind-fallback-1-text: #2a2d80;
    --kind-fallback-1-badge: #7577c8;
    --kind-fallback-2-bg: #f5ebd8;
    --kind-fallback-2-border: #c89650;
    --kind-fallback-2-text: #7a4d00;
    --kind-fallback-2-badge: #c89650;

    /* Graph interaction tokens — light */
    --graph-highlight-border: #6495d4;
    --graph-highlight-shadow: rgba(100,149,212,0.4);
    --graph-node-dimmed-bg: #ede8e0;
    --graph-node-dimmed-border: #c8c0b5;
    --graph-node-dimmed-text: #a09888;
    --graph-edge-dimmed: #c8c0b5;
    --graph-edge-label-bg: rgba(254,252,248,0.88);
    --graph-edge-label-text: #8a7a68;
    --graph-degree-bg: rgba(0,0,0,0.04);
    --graph-degree-text: #8a7a68;
    --graph-bg-dots: #d4c8b8;

    --sidebar: oklch(0.958 0.016 70);
    --sidebar-foreground: oklch(0.220 0.015 55);
    --sidebar-primary: oklch(0.220 0.015 55);
    --sidebar-primary-foreground: oklch(0.987 0.008 80);
    --sidebar-accent: oklch(0.945 0.018 72);
    --sidebar-accent-foreground: oklch(0.220 0.015 55);
    --sidebar-border: oklch(0.878 0.016 70);
    --sidebar-ring: oklch(0.650 0.020 65);
}
```

- [ ] **Step 2: Add dark mode kind/graph tokens to `.dark {}`**

The current `.dark {}` block has zero `--kind-*` or `--graph-*` entries. Append these after the existing shadcn dark tokens (before the closing `}`):

```css
    /* Graph node kind colors — dark (moved from JS KIND_STYLES) */
    --kind-function-bg: #1e3a5f;
    --kind-function-border: #3b82f6;
    --kind-function-text: #93c5fd;
    --kind-function-badge: #3b82f6;
    --kind-class-bg: #14532d;
    --kind-class-border: #22c55e;
    --kind-class-text: #86efac;
    --kind-class-badge: #22c55e;
    --kind-module-bg: #3b0764;
    --kind-module-border: #a855f7;
    --kind-module-text: #d8b4fe;
    --kind-module-badge: #a855f7;
    --kind-service-bg: #431407;
    --kind-service-border: #f97316;
    --kind-service-text: #fdba74;
    --kind-service-badge: #f97316;
    --kind-database-bg: #450a0a;
    --kind-database-border: #ef4444;
    --kind-database-text: #fca5a5;
    --kind-database-badge: #ef4444;
    --kind-queue-bg: #422006;
    --kind-queue-border: #eab308;
    --kind-queue-text: #fde047;
    --kind-queue-badge: #eab308;
    --kind-default-bg: #164e63;
    --kind-default-border: #06b6d4;
    --kind-default-text: #67e8f9;
    --kind-default-badge: #06b6d4;
    --kind-fallback-0-bg: #4a0a1a;
    --kind-fallback-0-border: #f43f5e;
    --kind-fallback-0-text: #fda4af;
    --kind-fallback-0-badge: #f43f5e;
    --kind-fallback-1-bg: #4a044e;
    --kind-fallback-1-border: #d946ef;
    --kind-fallback-1-text: #f0abfc;
    --kind-fallback-1-badge: #d946ef;
    --kind-fallback-2-bg: #1e1b4b;
    --kind-fallback-2-border: #6366f1;
    --kind-fallback-2-text: #a5b4fc;
    --kind-fallback-2-badge: #6366f1;

    /* Graph interaction tokens — dark */
    --graph-highlight-border: #60a5fa;
    --graph-highlight-shadow: rgba(96,165,250,0.5);
    --graph-node-dimmed-bg: #1a1a1a;
    --graph-node-dimmed-border: #333333;
    --graph-node-dimmed-text: #555555;
    --graph-edge-dimmed: #d4d4d8;
    --graph-edge-label-bg: rgba(14,14,14,0.88);
    --graph-edge-label-text: #a1a1aa;
    --graph-degree-bg: rgba(255,255,255,0.07);
    --graph-degree-text: #888888;
    --graph-bg-dots: #3f3f46;
}
```

- [ ] **Step 3: Verify TypeScript build passes**

```bash
cd playground && npm run build
```

Expected: build succeeds with no errors (CSS-only change, no TS involved).

- [ ] **Step 4: Visual check — toggle to light mode**

```bash
cd playground && npm run dev
```

Open browser at `http://localhost:5173`. Click the Sun/Moon toggle. Light mode should show a warm cream background (not stark white) and the graph nodes should use the existing CSS vars (they will still show old colors until Task 4 since `CustomNode.tsx` hasn't changed yet — that's expected).

- [ ] **Step 5: Commit**

```bash
git add playground/src/index.css
git commit -m "feat: warm pastel light mode tokens + dark mode kind/graph tokens in CSS"
```

---

## Task 2: Store — Add `isDark` to Config

**Files:**
- Modify: `playground/src/hooks/useGraphStore.ts`

- [ ] **Step 1: Add `isDark` to the `Config` interface**

In `useGraphStore.ts`, find the `Config` interface (line 16). Add `isDark: boolean` as the last field:

```ts
interface Config {
  viewMode: ViewMode
  layoutMode: LayoutMode
  showEdgeLabels: boolean
  showMinimap: boolean
  colorByKind: boolean
  ceilingMb: number
  costThreshold: number
  explainBeforeExecute: boolean
  showElapsed: boolean
  isDark: boolean
}
```

- [ ] **Step 2: Add `isDark: true` to the initial config**

Find the initial `config:` object in the store (around line 132). Add `isDark: true` as the last entry:

```ts
  config: {
    viewMode: 'live',
    layoutMode: 'dagre',
    showEdgeLabels: true,
    showMinimap: true,
    colorByKind: true,
    ceilingMb: 256,
    costThreshold: 100_000,
    explainBeforeExecute: false,
    showElapsed: true,
    isDark: true,
  },
```

- [ ] **Step 3: Verify TypeScript build passes**

```bash
cd playground && npm run build
```

Expected: build succeeds. TypeScript will now type-check all callers of `updateConfig` — no existing caller passes `isDark` so no errors expected.

- [ ] **Step 4: Commit**

```bash
git add playground/src/hooks/useGraphStore.ts
git commit -m "feat: add isDark to graph store config"
```

---

## Task 3: Toolbar — Use Store isDark

**Files:**
- Modify: `playground/src/components/Toolbar.tsx`

- [ ] **Step 1: Replace local state with store**

Current `Toolbar.tsx` has `const [isDark, setIsDark] = useState(true)`. Replace with:

```tsx
const isDark = useGraphStore((s) => s.config.isDark)
const updateConfig = useGraphStore((s) => s.updateConfig)
```

Keep the `useState` import — it is still used for `settingsOpen`. Keep the existing `useEffect` that toggles the `dark` class — it now reads `isDark` from the store:

```tsx
useEffect(() => {
  document.documentElement.classList.toggle('dark', isDark)
}, [isDark])
```

Change the toggle button's `onClick` from `setIsDark(!isDark)` to `updateConfig({ isDark: !isDark })`:

```tsx
<Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={() => updateConfig({ isDark: !isDark })}>
  {isDark ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
</Button>
```

- [ ] **Step 2: Verify TypeScript build passes**

```bash
cd playground && npm run build
```

Expected: build succeeds.

- [ ] **Step 3: Visual check**

```bash
cd playground && npm run dev
```

Toggle the theme. The dark class should still be applied to `<html>`. Light mode warm pastels and dark mode should both render correctly.

- [ ] **Step 4: Commit**

```bash
git add playground/src/components/Toolbar.tsx
git commit -m "feat: toolbar reads isDark from store"
```

---

## Task 4: EditorPanel + GraphPanel — Dynamic Theme

**Files:**
- Modify: `playground/src/components/EditorPanel.tsx`
- Modify: `playground/src/components/GraphPanel.tsx`

### EditorPanel

- [ ] **Step 1: Read isDark from store**

In `EditorPanel.tsx`, add a store selector at the top of the component:

```tsx
const isDark = useGraphStore((s) => s.config.isDark)
```

- [ ] **Step 2: Replace hardcoded CodeMirror theme**

Change `theme="dark"` to:

```tsx
theme={isDark ? 'dark' : 'light'}
```

### GraphPanel

- [ ] **Step 3: Read isDark from store**

In `GraphPanel.tsx`, add:

```tsx
const isDark = useGraphStore((s) => s.config.isDark)
```

- [ ] **Step 4: Dynamic MiniMap maskColor**

Find the `MiniMap` component. Replace the hardcoded `maskColor="rgba(0,0,0,0.7)"` with:

```tsx
maskColor={isDark ? 'rgba(0,0,0,0.7)' : 'rgba(255,255,255,0.7)'}
```

- [ ] **Step 5: Wire Background to `--graph-bg-dots` token**

Find the `<Background>` component. It currently uses `className="!text-border"` which routes through `--border`. Replace with the `color` prop pointing to the token:

```tsx
<Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--graph-bg-dots)" />
```

(Remove the `className` prop from Background entirely.)

- [ ] **Step 6: Verify TypeScript build passes**

```bash
cd playground && npm run build
```

Expected: build succeeds.

- [ ] **Step 7: Visual check**

```bash
cd playground && npm run dev
```

Toggle between light and dark mode. Verify:
- Editor background switches between light (white/cream) and dark
- MiniMap mask is dark in dark mode, light in light mode
- Graph background dots use the warm taupe color in light mode and dark zinc in dark mode

- [ ] **Step 8: Commit**

```bash
git add playground/src/components/EditorPanel.tsx playground/src/components/GraphPanel.tsx
git commit -m "feat: dynamic editor theme and minimap mask from store isDark"
```

---

## Task 5: CustomNode — CSS Var Migration

**Files:**
- Modify: `playground/src/components/graph/CustomNode.tsx`

- [ ] **Step 1: Replace KIND_STYLES, FALLBACK_STYLES, and getKindStyle with token helper**

Remove the entire `KIND_STYLES` object, `FALLBACK_STYLES` array, and `getKindStyle` function. Replace with:

```tsx
const NAMED_KINDS = new Set(['function', 'class', 'module', 'service', 'database', 'queue', 'default'])
const FALLBACK_NAMES = ['fallback-0', 'fallback-1', 'fallback-2']

function getKindTokenName(kind: string): string {
  if (NAMED_KINDS.has(kind)) return kind
  let hash = 0
  for (let i = 0; i < kind.length; i++) hash = kind.charCodeAt(i) + ((hash << 5) - hash)
  return FALLBACK_NAMES[Math.abs(hash) % FALLBACK_NAMES.length]
}
```

Note: `default` is explicitly in `NAMED_KINDS` — it resolves to `--kind-default-*` tokens, not a fallback.

- [ ] **Step 2: Rewrite the node's inline styles**

Replace the `const s = getKindStyle(kind)` line and all uses of `s.bg`, `s.border`, `s.text`, `s.badge` with CSS var references. The full component `<div>` style object:

```tsx
const kindName = getKindTokenName(kind)

// in the outer <div> style:
style={{
  backgroundColor: dimmed ? 'var(--graph-node-dimmed-bg)' : `var(--kind-${kindName}-bg)`,
  borderColor: highlighted
    ? 'var(--graph-highlight-border)'
    : dimmed ? 'var(--graph-node-dimmed-border)'
    : `var(--kind-${kindName}-border)`,
  borderWidth: highlighted ? '2px' : '1px',
  boxShadow: highlighted
    ? '0 0 16px var(--graph-highlight-shadow)'
    : dimmed ? 'none'
    : `0 0 ${6 + degree * 2}px color-mix(in srgb, var(--kind-${kindName}-border) 27%, transparent)`,
  opacity: dimmed ? 0.2 : 1,
  transition: 'opacity 0.2s, background-color 0.2s, border-color 0.2s, box-shadow 0.2s',
  transform: `scale(${dimmed ? 0.95 : scale})`,
  minWidth: `${minWidth}px`,
}}
```

Label text color:
```tsx
style={{ color: dimmed ? 'var(--graph-node-dimmed-text)' : `var(--kind-${kindName}-text)`, fontSize: `${11 + Math.min(degree, 3)}px` }}
```

Badge styles:
```tsx
style={{
  backgroundColor: `color-mix(in srgb, var(--kind-${kindName}-badge) 13%, transparent)`,
  color: dimmed ? 'var(--graph-node-dimmed-text)' : `var(--kind-${kindName}-badge)`,
  border: dimmed
    ? '1px solid var(--graph-node-dimmed-text)'
    : `1px solid color-mix(in srgb, var(--kind-${kindName}-badge) 27%, transparent)`,
}}
```

Degree counter:
```tsx
style={{ color: 'var(--graph-degree-text)', backgroundColor: 'var(--graph-degree-bg)' }}
```

Both Handle elements (top target and bottom source):
```tsx
style={{ background: `var(--kind-${kindName}-border)` }}
```

- [ ] **Step 3: Verify TypeScript build passes**

```bash
cd playground && npm run build
```

Expected: build succeeds. The `KIND_STYLES` type was inferred, so removing it should be clean.

- [ ] **Step 4: Visual check — both modes**

```bash
cd playground && npm run dev
```

Load an example (e.g. "Function Call Graph" from the Examples menu). Toggle between light and dark mode. Verify:
- Dark mode: nodes still show dark bg with vivid colored borders (same as before)
- Light mode: nodes show soft pastel bg with muted borders — not the dark navy/forest/plum they showed before
- Hover a node: neighbors highlight, others dim with warm gray in light mode
- `default`-kinded nodes (teal) render correctly in both modes (not as fallback-0 rose)

- [ ] **Step 5: Commit**

```bash
git add playground/src/components/graph/CustomNode.tsx
git commit -m "feat: CustomNode uses CSS design tokens for all colors"
```

---

## Task 6: CustomEdge — CSS Var Migration

**Files:**
- Modify: `playground/src/components/graph/CustomEdge.tsx`

- [ ] **Step 1: Replace hardcoded edge colors with token references**

In `CustomEdge.tsx`, find the `BaseEdge` `style` prop. Replace the `stroke` value:

```tsx
stroke: highlighted ? 'var(--graph-highlight-border)' : dimmed ? 'var(--graph-edge-dimmed)' : color,
```

Keep `strokeWidth`, `opacity`, and `transition` unchanged.

- [ ] **Step 2: Replace hardcoded label colors**

Find the edge label `<div>` style. Replace `color` and `backgroundColor`:

```tsx
color: dimmed ? 'var(--graph-node-dimmed-text)' : 'var(--graph-edge-label-text)',
backgroundColor: 'var(--graph-edge-label-bg)',
```

Keep `position`, `transform`, `pointerEvents`, `fontSize`, `padding`, `borderRadius`, `opacity`, and `transition` unchanged.

- [ ] **Step 3: Verify TypeScript build passes**

```bash
cd playground && npm run build
```

Expected: build succeeds with no errors.

- [ ] **Step 4: Visual check — edge behavior**

```bash
cd playground && npm run dev
```

Load an example. Toggle to light mode. Verify:
- Non-dimmed edges: still show vivid colored strokes (from `EDGE_COLORS` JS constants) with 0.8 opacity
- Hover a node: non-adjacent edges dim — stroke switches to `#c8c0b5` (warm gray) in light mode
- Edge labels (if enabled in Settings): show warm cream background and warm gray text in light mode
- Dark mode: highlighted edges use `#60a5fa`, dimmed edges use `#d4d4d8` (lighter gray than the old `#333`)

- [ ] **Step 5: Final commit**

```bash
git add playground/src/components/graph/CustomEdge.tsx
git commit -m "feat: CustomEdge uses CSS design tokens for highlight/dim/label colors"
```

---

## Final Verification

- [ ] **Full build**

```bash
cd playground && npm run build && npm run lint
```

Expected: zero TypeScript errors, zero lint errors.

- [ ] **Manual smoke test — all 6 node kinds**

Load "Class Hierarchy" or "Microservices Map" example which uses multiple node kinds. In light mode:
- Each kind has a distinct pastel bg/border
- `default` renders teal (not rose)
- Unknown kinds hash to one of 3 fallback palettes (rose / indigo / amber)
- Hover, dim, highlight all work as expected

- [ ] **Manual smoke test — dark mode unchanged**

Toggle to dark mode. Verify dark mode looks identical to before this change was made (vivid colored node backgrounds, bright borders).
