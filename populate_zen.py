#!/usr/bin/env python3
"""Populate graphstore with orkait-zen behavior map."""

import sys
sys.path.insert(0, "/home/kai/code/orkait/graphstore")

from graphstore import GraphStore

DB_PATH = "/home/kai/code/orkait/graphstore/zen-graph-db"

g = GraphStore(path=DB_PATH)

_all_queries: list[str] = []

def q(query: str):
    """Execute query, ignore duplicate errors. Collect for script storage."""
    _all_queries.append(query)
    try:
        g.execute(query)
    except Exception as e:
        if "exists" not in str(e).lower() and "duplicate" not in str(e).lower():
            print(f"ERROR: {query[:80]}... → {e}")

# ═══════════════════════════════════════════════════════════════════
# SCHEMA REGISTRATION
# ═══════════════════════════════════════════════════════════════════

q('SYS REGISTER NODE KIND "page" REQUIRED name OPTIONAL file, description')
q('SYS REGISTER NODE KIND "component" REQUIRED name OPTIONAL file, description')
q('SYS REGISTER NODE KIND "hook" REQUIRED name OPTIONAL file, description')
q('SYS REGISTER NODE KIND "store" REQUIRED name OPTIONAL file, description')
q('SYS REGISTER NODE KIND "yjs_map" REQUIRED name OPTIONAL description')
q('SYS REGISTER NODE KIND "api_route" REQUIRED name OPTIONAL file, method, path, description')
q('SYS REGISTER NODE KIND "pipeline_fn" REQUIRED name OPTIONAL file, description')
q('SYS REGISTER NODE KIND "util" REQUIRED name OPTIONAL file, description')
q('SYS REGISTER NODE KIND "type" REQUIRED name OPTIONAL file, description')
q('SYS REGISTER NODE KIND "canvas_node" REQUIRED name OPTIONAL file, description')
q('SYS REGISTER NODE KIND "canvas_edge" REQUIRED name OPTIONAL file, description')
q('SYS REGISTER NODE KIND "action" REQUIRED name OPTIONAL description, store')
q('SYS REGISTER NODE KIND "yjs_action" REQUIRED name OPTIONAL file, description')
q('SYS REGISTER NODE KIND "module" REQUIRED name OPTIONAL file, description')

# ═══════════════════════════════════════════════════════════════════
# PAGES
# ═══════════════════════════════════════════════════════════════════

q('CREATE NODE "page:home" kind = "page" name = "HomePage" file = "pages/HomePage.tsx" description = "Landing page with project list and prompt"')
q('CREATE NODE "page:editor" kind = "page" name = "EditorPage" file = "pages/EditorPage.tsx" description = "Main editor — canvas, sidebar, prompt bar, resizable panels"')
q('CREATE NODE "page:projects" kind = "page" name = "ProjectsPage" file = "pages/ProjectsPage.tsx" description = "Project list/management"')
q('CREATE NODE "page:login" kind = "page" name = "LoginPage" file = "pages/LoginPage.tsx" description = "Auth login/register"')

# ═══════════════════════════════════════════════════════════════════
# COMPONENTS — Main Blocks
# ═══════════════════════════════════════════════════════════════════

q('CREATE NODE "comp:canvas" kind = "component" name = "Canvas" file = "components/blocks/Canvas.tsx" description = "ReactFlow wrapper — renders screen/transition nodes, edges, handles connections"')
q('CREATE NODE "comp:sidebar" kind = "component" name = "Sidebar" file = "components/blocks/Sidebar.tsx" description = "Project/screen tree sidebar"')
q('CREATE NODE "comp:inspector" kind = "component" name = "Inspector" file = "components/blocks/Inspector.tsx" description = "Screen/node property panel"')
q('CREATE NODE "comp:prompt_bar" kind = "component" name = "PromptBar" file = "components/blocks/PromptBar.tsx" description = "Generation input bar — triggers generate/retouch/remake"')
q('CREATE NODE "comp:prompt_homepage" kind = "component" name = "PromptHomepage" file = "components/blocks/PromptHomepage.tsx" description = "Homepage prompt — creates project then navigates with pendingPrompt"')
q('CREATE NODE "comp:preview_modal" kind = "component" name = "PreviewModal" file = "components/blocks/PreviewModal.tsx" description = "Full-screen HTML preview of generated screen"')
q('CREATE NODE "comp:theme_palette" kind = "component" name = "ThemePalette" file = "components/blocks/ThemePalette.tsx" description = "Theme color/typography editor"')
q('CREATE NODE "comp:settings_modal" kind = "component" name = "SettingsModal" file = "components/blocks/SettingsModal.tsx" description = "Model routing configuration"')
q('CREATE NODE "comp:autoflow_modal" kind = "component" name = "AutoflowModal" file = "components/blocks/AutoflowModal.tsx" description = "Autoflow check/run confirmation dialog"')
q('CREATE NODE "comp:app_sidebar" kind = "component" name = "AppSidebar" file = "components/blocks/AppSidebar.tsx" description = "Top-level navigation shell"')
q('CREATE NODE "comp:templates" kind = "component" name = "TemplatesComponent" file = "components/blocks/TemplatesComponent.tsx" description = "Template library browser"')
q('CREATE NODE "comp:recent" kind = "component" name = "RecentComponent" file = "components/blocks/RecentComponent.tsx" description = "Recent projects list"')
q('CREATE NODE "comp:project_card" kind = "component" name = "ProjectCard" file = "components/blocks/ProjectCard.tsx" description = "Project card in list"')

# Canvas sub-components
q('CREATE NODE "comp:canvas_toolbar" kind = "component" name = "CanvasToolbar" file = "components/blocks/canvas/CanvasToolbar.tsx" description = "Canvas zoom/pan/layout controls"')
q('CREATE NODE "comp:layers_panel" kind = "component" name = "LayersPanel" file = "components/blocks/canvas/LayersPanel.tsx" description = "Layer ordering and group management"')
q('CREATE NODE "comp:node_palette" kind = "component" name = "NodePalette" file = "components/blocks/canvas/NodePalette.tsx" description = "Node creation palette — add transition nodes"')
q('CREATE NODE "comp:screen_context_menu" kind = "component" name = "ScreenContextMenu" file = "components/blocks/canvas/ScreenContextMenu.tsx" description = "Right-click menu for screen nodes"')
q('CREATE NODE "comp:tx_context_menu" kind = "component" name = "TxNodeContextMenu" file = "components/blocks/canvas/TxNodeContextMenu.tsx" description = "Right-click menu for transition nodes"')
q('CREATE NODE "comp:add_screen_modal" kind = "component" name = "AddScreenModal" file = "components/blocks/canvas/AddScreenModal.tsx" description = "Screen creation dialog"')
q('CREATE NODE "comp:group_frames" kind = "component" name = "GroupFrames" file = "components/blocks/canvas/components/GroupFrames.tsx" description = "Visual group containers on canvas"')
q('CREATE NODE "comp:presence_cursors" kind = "component" name = "PresenceCursors" file = "components/blocks/canvas/PresenceCursors.tsx" description = "Collaborative cursor presence overlay"')
q('CREATE NODE "comp:edge_label_editor" kind = "component" name = "EdgeLabelEditor" file = "components/blocks/canvas/components/EdgeLabelEditor.tsx" description = "Inline edge label editing"')

# Canvas node types
q('CREATE NODE "canvas_node:screen" kind = "canvas_node" name = "ScreenNode" file = "canvas/nodes/ScreenNode.tsx" description = "ReactFlow node — renders screen preview with HTML iframe"')
q('CREATE NODE "canvas_node:action" kind = "canvas_node" name = "ActionNode" file = "canvas/nodes/ActionNode.tsx" description = "Transition node — user action (tap, swipe, click)"')
q('CREATE NODE "canvas_node:branch" kind = "canvas_node" name = "BranchNode" file = "canvas/nodes/BranchNode.tsx" description = "Conditional branch node"')
q('CREATE NODE "canvas_node:context" kind = "canvas_node" name = "ContextNode" file = "canvas/nodes/ContextNode.tsx" description = "Context/data reference node"')
q('CREATE NODE "canvas_node:state" kind = "canvas_node" name = "StateNode" file = "canvas/nodes/StateNode.tsx" description = "UI state/variant node"')
q('CREATE NODE "canvas_node:variant" kind = "canvas_node" name = "VariantNode" file = "canvas/nodes/VariantNode.tsx" description = "Variant/theme variation node"')
q('CREATE NODE "canvas_node:style" kind = "canvas_node" name = "StyleNode" file = "canvas/nodes/StyleNode.tsx" description = "Design token override node"')
q('CREATE NODE "canvas_node:sitemap" kind = "canvas_node" name = "SitemapCard" file = "canvas/nodes/SitemapCard.tsx" description = "Sitemap view card"')

# Canvas edge types
q('CREATE NODE "canvas_edge:zen" kind = "canvas_edge" name = "ZenEdge" file = "canvas/edges/ZenEdge.tsx" description = "Custom edge renderer with labels and markers"')

# Inspector sub-components
q('CREATE NODE "comp:screen_inspector" kind = "component" name = "ScreenInspector" file = "components/blocks/inspector/ScreenInspector.tsx" description = "Screen property editor"')
q('CREATE NODE "comp:transition_inspector" kind = "component" name = "TransitionInspector" file = "components/blocks/inspector/TransitionInspector.tsx" description = "Transition node property editor"')

# Panels
q('CREATE NODE "comp:history_panel" kind = "component" name = "HistoryPanel" file = "components/panels/HistoryPanel.tsx" description = "Restore points history UI"')
q('CREATE NODE "comp:debug_panel" kind = "component" name = "GenerationDebugPanel" file = "components/panels/GenerationDebugPanel.tsx" description = "Generation debug output"')

# ═══════════════════════════════════════════════════════════════════
# HOOKS
# ═══════════════════════════════════════════════════════════════════

q('CREATE NODE "hook:useScreens" kind = "hook" name = "useScreens" file = "hooks/useScreens.ts" description = "Yjs-reactive screens — CRUD, layout, locking, themes"')
q('CREATE NODE "hook:useTxNodes" kind = "hook" name = "useTxNodes" file = "hooks/useTxNodes.ts" description = "Yjs-reactive transition nodes — CRUD, positioning"')
q('CREATE NODE "hook:useYjsEdges" kind = "hook" name = "useYjsEdges" file = "hooks/useYjsEdges.ts" description = "Yjs-reactive edges — add/remove/bulk operations"')
q('CREATE NODE "hook:useGroups" kind = "hook" name = "useGroups" file = "hooks/useGroups.ts" description = "Yjs-reactive groups — create/delete/toggle/membership"')
q('CREATE NODE "hook:useGeneration" kind = "hook" name = "useGeneration" file = "hooks/useGeneration.ts" description = "Generation API client — generate, autoflow check/run"')
q('CREATE NODE "hook:useUndoRedo" kind = "hook" name = "useUndoRedo" file = "hooks/useUndoRedo.ts" description = "Undo/redo via Yjs UndoManager"')
q('CREATE NODE "hook:useSyncFlow" kind = "hook" name = "useSyncFlow" file = "hooks/useSyncFlow.ts" description = "Yjs WebSocket provider — connects doc to Hocuspocus server"')
q('CREATE NODE "hook:useToast" kind = "hook" name = "useToast" file = "hooks/useToast.ts" description = "Toast notification hook"')
q('CREATE NODE "hook:useProjects" kind = "hook" name = "useProjects" file = "hooks/use-projects.ts" description = "Project list fetching"')
q('CREATE NODE "hook:useAwareness" kind = "hook" name = "useAwarenessUsers" file = "hooks/useAwarenessUsers.ts" description = "Collaborative presence — cursor positions via Yjs awareness"')
q('CREATE NODE "hook:useAutoLayout" kind = "hook" name = "useAutoLayout" file = "canvas/hooks/useAutoLayout.ts" description = "Dagre-based automatic canvas layout"')
q('CREATE NODE "hook:useSitemapDagre" kind = "hook" name = "useSitemapDagre" file = "canvas/hooks/useSitemapDagre.ts" description = "Sitemap tab dagre layout"')
q('CREATE NODE "hook:useCanvasShortcuts" kind = "hook" name = "useCanvasShortcuts" file = "canvas/useCanvasShortcuts.ts" description = "Canvas keyboard shortcuts (delete, undo, redo)"')
q('CREATE NODE "hook:useGroupDragSync" kind = "hook" name = "useGroupDragSync" file = "canvas/hooks/useGroupDragSync.ts" description = "Group drag synchronization"')
q('CREATE NODE "hook:useGroupSelection" kind = "hook" name = "useGroupSelection" file = "canvas/hooks/useGroupSelection.ts" description = "Multi-select and group selection"')
q('CREATE NODE "hook:useNodeBuilder" kind = "hook" name = "useNodeBuilder" file = "canvas/hooks/useNodeBuilder.ts" description = "Build ReactFlow nodes from Yjs data"')

# ═══════════════════════════════════════════════════════════════════
# STORES
# ═══════════════════════════════════════════════════════════════════

q('CREATE NODE "store:editor" kind = "store" name = "useEditorStore" file = "store/editor.ts" description = "Main editor state — project, activeScreen, theme, canvas settings, generating IDs"')
q('CREATE NODE "store:auth" kind = "store" name = "useAuthStore" file = "store/auth.ts" description = "Auth state — token, user"')
q('CREATE NODE "store:settings" kind = "store" name = "useSettingsStore" file = "store/settings.ts" description = "Model routing settings — model, modelRouting, debugMode"')

# ═══════════════════════════════════════════════════════════════════
# YJS MAPS (the 7 Yjs document maps)
# ═══════════════════════════════════════════════════════════════════

q('CREATE NODE "yjs:screens" kind = "yjs_map" name = "screens" description = "Y.Map<screenId, ScreenNode> — source of truth for all screens"')
q('CREATE NODE "yjs:txNodes" kind = "yjs_map" name = "txNodes" description = "Y.Map<txNodeId, TransitionNode> — transition nodes (action, branch, context, state, variant, style)"')
q('CREATE NODE "yjs:edges" kind = "yjs_map" name = "edges" description = "Y.Map<edgeId, EdgeNode> — connections between screens and transition nodes"')
q('CREATE NODE "yjs:layout" kind = "yjs_map" name = "layout" description = "Y.Map<nodeId, LayoutEntry> — x,y positions of all nodes"')
q('CREATE NODE "yjs:layers" kind = "yjs_map" name = "layers" description = "Y.Array<LayerItem> — ordered layer stack with groups"')
q('CREATE NODE "yjs:meta" kind = "yjs_map" name = "meta" description = "Y.Map — theme, project metadata, shared settings"')
q('CREATE NODE "yjs:dict" kind = "yjs_map" name = "dict" description = "Y.Map<shortId, string> — GP7 dictionary aliasing for compression"')

# Yjs infrastructure
q('CREATE NODE "module:yjs_doc" kind = "module" name = "yjs-doc" file = "store/yjs-doc.ts" description = "Y.Doc singleton + HocuspocusProvider + UndoManager"')
q('CREATE NODE "module:yjs_actions" kind = "module" name = "yjs-actions" file = "store/yjs-actions.ts" description = "All Yjs mutation functions"')
q('CREATE NODE "module:yjs_observer" kind = "module" name = "yjs-observer" file = "store/yjs-observer.ts" description = "Yjs → Zustand observer bridge"')

# ═══════════════════════════════════════════════════════════════════
# YJS ACTIONS (mutation functions)
# ═══════════════════════════════════════════════════════════════════

q('CREATE NODE "yjs_act:addScreenYjs" kind = "yjs_action" name = "addScreenYjs" description = "Add screen to Yjs screens map + layout + layers"')
q('CREATE NODE "yjs_act:removeScreenYjs" kind = "yjs_action" name = "removeScreenYjs" description = "Mark screen deleted, remove edges, remove from layers"')
q('CREATE NODE "yjs_act:updateScreenYjs" kind = "yjs_action" name = "updateScreenYjs" description = "Update screen properties in Yjs"')
q('CREATE NODE "yjs_act:duplicateScreenYjs" kind = "yjs_action" name = "duplicateScreenYjs" description = "Clone screen with new ID"')
q('CREATE NODE "yjs_act:bulkSetScreensYjs" kind = "yjs_action" name = "bulkSetScreensYjs" description = "Bulk-replace all screens (project load seeding)"')
q('CREATE NODE "yjs_act:addTransitionNode" kind = "yjs_action" name = "addTransitionNode" description = "Add transition node to txNodes map + layout + layers"')
q('CREATE NODE "yjs_act:updateTransitionNode" kind = "yjs_action" name = "updateTransitionNode" description = "Update transition node properties"')
q('CREATE NODE "yjs_act:deleteTransitionNode" kind = "yjs_action" name = "deleteTransitionNode" description = "Mark transition node deleted, remove edges"')
q('CREATE NODE "yjs_act:addEdgeYjs" kind = "yjs_action" name = "addEdgeYjs" description = "Add ReactFlow edge to Yjs edges map"')
q('CREATE NODE "yjs_act:removeEdgeYjs" kind = "yjs_action" name = "removeEdgeYjs" description = "Remove edge from Yjs edges map"')
q('CREATE NODE "yjs_act:bulkSetEdgesYjs" kind = "yjs_action" name = "bulkSetEdgesYjs" description = "Bulk-replace all edges"')
q('CREATE NODE "yjs_act:addGroupYjs" kind = "yjs_action" name = "addGroupYjs" description = "Create group in layers array"')
q('CREATE NODE "yjs_act:removeGroupYjs" kind = "yjs_action" name = "removeGroupYjs" description = "Remove group, reparent children to root"')
q('CREATE NODE "yjs_act:detachAtBoundary" kind = "yjs_action" name = "detachAtBoundary" description = "Remove all edges touching a node"')
q('CREATE NODE "yjs_act:reconnectFragments" kind = "yjs_action" name = "reconnectFragments" description = "Reconnect graph fragments after node deletion"')
q('CREATE NODE "yjs_act:updateTheme" kind = "yjs_action" name = "updateThemeMeta" description = "Update theme in meta map"')

# ═══════════════════════════════════════════════════════════════════
# EDITOR STORE ACTIONS
# ═══════════════════════════════════════════════════════════════════

q('CREATE NODE "action:setActiveScreen" kind = "action" name = "setActiveScreen" description = "Set the currently active screen ID" store = "editor"')
q('CREATE NODE "action:setActiveNode" kind = "action" name = "setActiveNode" description = "Set the currently active node ID" store = "editor"')
q('CREATE NODE "action:addGeneratingId" kind = "action" name = "addGeneratingId" description = "Mark a screen as currently generating" store = "editor"')
q('CREATE NODE "action:removeGeneratingId" kind = "action" name = "removeGeneratingId" description = "Unmark screen generating state" store = "editor"')
q('CREATE NODE "action:updateTheme" kind = "action" name = "updateTheme" description = "Update generation theme (colors, typography)" store = "editor"')
q('CREATE NODE "action:resetEditor" kind = "action" name = "resetEditor" description = "Reset all editor state (on project switch)" store = "editor"')
q('CREATE NODE "action:setYjsSynced" kind = "action" name = "setYjsSynced" description = "Mark Yjs WebSocket sync complete" store = "editor"')
q('CREATE NODE "action:setEditorMode" kind = "action" name = "setEditorMode" description = "Switch canvas vs sitemap mode" store = "editor"')
q('CREATE NODE "action:toggleCanvasDark" kind = "action" name = "toggleCanvasDark" description = "Toggle dark/light canvas background" store = "editor"')

# ═══════════════════════════════════════════════════════════════════
# API ROUTES (Backend)
# ═══════════════════════════════════════════════════════════════════

q('CREATE NODE "api:generate" kind = "api_route" name = "POST /api/v2/generate" file = "routes/generate.ts" method = "POST" path = "/api/v2/generate" description = "Run generation pipeline — intent→spec→HTML→persist→Yjs"')
q('CREATE NODE "api:autoflow_check" kind = "api_route" name = "POST /api/v2/autoflow/check" file = "routes/autoflow.ts" method = "POST" path = "/api/v2/autoflow/check" description = "Pre-flight check for autoflow — alignment, cost estimate"')
q('CREATE NODE "api:autoflow_run" kind = "api_route" name = "POST /api/v2/autoflow/run" file = "routes/autoflow.ts" method = "POST" path = "/api/v2/autoflow/run" description = "Execute multi-screen autoflow generation"')
q('CREATE NODE "api:projects_list" kind = "api_route" name = "GET /api/projects" file = "routes/projects.ts" method = "GET" path = "/api/projects" description = "List user projects"')
q('CREATE NODE "api:projects_create" kind = "api_route" name = "POST /api/projects" file = "routes/projects.ts" method = "POST" path = "/api/projects" description = "Create new project"')
q('CREATE NODE "api:project_get" kind = "api_route" name = "GET /api/projects/:id" file = "routes/projects.ts" method = "GET" path = "/api/projects/:id" description = "Get project with latestGenerations"')
q('CREATE NODE "api:login" kind = "api_route" name = "POST /api/login" file = "routes/auth.ts" method = "POST" path = "/api/login" description = "User login"')
q('CREATE NODE "api:register" kind = "api_route" name = "POST /api/register" file = "routes/auth.ts" method = "POST" path = "/api/register" description = "User registration"')
q('CREATE NODE "api:restore_points" kind = "api_route" name = "GET /api/restore-points" file = "routes/history.ts" method = "GET" path = "/api/restore-points" description = "List restore points for project"')
q('CREATE NODE "api:restore" kind = "api_route" name = "POST /api/restore" file = "routes/history.ts" method = "POST" path = "/api/restore" description = "Restore project to restore point"')

# ═══════════════════════════════════════════════════════════════════
# PIPELINE FUNCTIONS (zen-core)
# ═══════════════════════════════════════════════════════════════════

q('CREATE NODE "pipe:runMockupPipeline" kind = "pipeline_fn" name = "runMockupPipeline" file = "mockup/pipeline.ts" description = "Main 6-step pipeline: Intent→Seeds→FlexSpec→Tokens→HTML→FlowGraph"')
q('CREATE NODE "pipe:runIntentLayer" kind = "pipeline_fn" name = "runIntentLayer" file = "intent/index.ts" description = "4-candidate fan-out intent extraction — domain, platform, personas, goals"')
q('CREATE NODE "pipe:buildFallbackIntent" kind = "pipeline_fn" name = "buildFallbackIntent" file = "intent/index.ts" description = "Deterministic fallback when LLM intent fails"')
q('CREATE NODE "pipe:buildScreenSeeds" kind = "pipeline_fn" name = "buildScreenSeeds" file = "mockup/screen-spec-flex.ts" description = "Select screen seeds from intent — single or multi-screen"')
q('CREATE NODE "pipe:buildFlexibleScreenSpec" kind = "pipeline_fn" name = "buildFlexibleScreenSpec" file = "mockup/screen-spec-flex.ts" description = "LLM generates FlexibleScreenSpec per screen"')
q('CREATE NODE "pipe:buildFallbackSpec" kind = "pipeline_fn" name = "buildFallbackSpec" file = "mockup/screen-spec-flex.ts" description = "Deterministic 3-section fallback spec"')
q('CREATE NODE "pipe:buildDesignTokens" kind = "pipeline_fn" name = "buildDesignTokens" file = "mockup/design-tokens.ts" description = "Deterministic design token generation from prompt seed"')
q('CREATE NODE "pipe:generateScreenHtml" kind = "pipeline_fn" name = "generateScreenHtml" file = "mockup/render-html.ts" description = "LLM generates Tailwind HTML from spec+tokens"')
q('CREATE NODE "pipe:buildFallbackBody" kind = "pipeline_fn" name = "buildFallbackBody" file = "mockup/render-html.ts" description = "Deterministic HTML fallback when LLM fails"')
q('CREATE NODE "pipe:callLLM" kind = "pipeline_fn" name = "callLLM" file = "ai/openrouter.ts" description = "Base LLM call via OpenRouter — cache check, API call, cache store"')
q('CREATE NODE "pipe:buildStackForScreen" kind = "pipeline_fn" name = "buildStackForScreen" file = "transitionStack/builder.ts" description = "Walk graph backward to build execution stack for screen"')
q('CREATE NODE "pipe:executeStack" kind = "pipeline_fn" name = "executeStack" file = "transitionStack/executor.ts" description = "Evaluate transition stack — produces sharedContext, actionNarrative"')

# Backend-specific functions
q('CREATE NODE "pipe:applyGenerationToDoc" kind = "pipeline_fn" name = "applyGenerationToDoc" file = "routes/generate.ts" description = "Write generation result to Yjs doc (screens, layout, layers, edges)"')
q('CREATE NODE "pipe:persistGeneration" kind = "pipeline_fn" name = "persistGeneration" file = "routes/generate.ts" description = "Save generation to database (compressed)"')
q('CREATE NODE "pipe:chargeCredits" kind = "pipeline_fn" name = "chargeCredits" file = "routes/generate.ts" description = "Billing — idempotent credit charge with replay detection"')
q('CREATE NODE "pipe:acquireLock" kind = "pipeline_fn" name = "acquireLock" file = "routes/generate.ts" description = "Acquire generation lock for screen"')
q('CREATE NODE "pipe:appendLineage" kind = "pipeline_fn" name = "appendGenerationLineage" file = "routes/generate.ts" description = "Record generation lineage event"')
q('CREATE NODE "pipe:createRestorePoint" kind = "pipeline_fn" name = "createRestorePoint" file = "routes/generate.ts" description = "Snapshot project state for undo"')

# Cache
q('CREATE NODE "pipe:cache" kind = "pipeline_fn" name = "ExaiCache" file = "ai/cache.ts" description = "Unified LLM cache — SHA256 keys, 7-day TTL, namespace files"')

# ═══════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════

q('CREATE NODE "util:apiFetch" kind = "util" name = "apiFetch" file = "lib/api.ts" description = "Generic HTTP fetch wrapper with auth"')
q('CREATE NODE "util:api_client" kind = "util" name = "api-client" file = "lib/api-client.ts" description = "Typed API endpoints — generateScreen, autoflowCheck, autoflowRun"')
q('CREATE NODE "util:cn" kind = "util" name = "cn" file = "lib/utils.ts" description = "Tailwind class merger"')
q('CREATE NODE "util:edgeValidation" kind = "util" name = "isConnectionValid" file = "canvas/edgeValidation.ts" description = "Edge connection validation rules"')
q('CREATE NODE "util:syncHash" kind = "util" name = "sync-hash" file = "canvas/sync-hash.ts" description = "Fingerprinting — computePromptFingerprint, isOutOfSync"')
q('CREATE NODE "util:groupUtils" kind = "util" name = "groupUtils" file = "store/groupUtils.ts" description = "buildGroupsFromLayers helper"')

# ═══════════════════════════════════════════════════════════════════
# TYPES
# ═══════════════════════════════════════════════════════════════════

q('CREATE NODE "type:Screen" kind = "type" name = "Screen" file = "store/editor.ts" description = "id, title, spec, htmlOutput, device, locked, hidden, isDirty, themeMode"')
q('CREATE NODE "type:ScreenNode" kind = "type" name = "ScreenNode" file = "core/canvas/types.ts" description = "Yjs screen node — id, name, device, contentJson, isDeleted, status"')
q('CREATE NODE "type:TransitionNode" kind = "type" name = "TransitionNode" file = "core/canvas/types.ts" description = "Yjs transition — id, type, label, payload, mergePolicy"')
q('CREATE NODE "type:EdgeNode" kind = "type" name = "EdgeNode" file = "core/canvas/types.ts" description = "Yjs edge — id, from, to, label, handles, attachments"')
q('CREATE NODE "type:LayoutEntry" kind = "type" name = "LayoutEntry" file = "core/canvas/types.ts" description = "Node position — x, y, w, h"')
q('CREATE NODE "type:LayerItem" kind = "type" name = "LayerItem" file = "core/canvas/types.ts" description = "Layer stack item — kind, id, orderIndex, parentGroupId"')
q('CREATE NODE "type:GenerateMode" kind = "type" name = "GenerateMode" file = "core/api/types.ts" description = "generate | variant | retouch | remake | sync"')
q('CREATE NODE "type:FlexibleScreenSpec" kind = "type" name = "FlexibleScreenSpec" file = "core/mockup/types.ts" description = "Screen spec — id, title, description, sections[], navItems[]"')
q('CREATE NODE "type:IntentSpec" kind = "type" name = "IntentSpec" file = "core/intent/types.ts" description = "Intent — domain, platform, personas, primaryGoal, screensCore/Optional"')
q('CREATE NODE "type:MockupTokens" kind = "type" name = "MockupTokens" file = "core/mockup/types.ts" description = "Design tokens — colors, typography, spacing, radius"')
q('CREATE NODE "type:GenerationTheme" kind = "type" name = "GenerationTheme" file = "store/editor.ts" description = "Theme palette — background, primary, secondary, accent, text"')
q('CREATE NODE "type:Group" kind = "type" name = "Group" file = "store/editor.ts" description = "Layer group — id, name, screenIds[], txNodeIds[], color, collapsed"')

# ═══════════════════════════════════════════════════════════════════
# EDGES — Page renders components
# ═══════════════════════════════════════════════════════════════════

# HomePage renders
q('CREATE EDGE "page:home" -> "comp:prompt_homepage" kind = "renders"')
q('CREATE EDGE "page:home" -> "comp:templates" kind = "renders"')
q('CREATE EDGE "page:home" -> "comp:recent" kind = "renders"')
q('CREATE EDGE "page:home" -> "comp:app_sidebar" kind = "renders"')

# EditorPage renders
q('CREATE EDGE "page:editor" -> "comp:canvas" kind = "renders"')
q('CREATE EDGE "page:editor" -> "comp:sidebar" kind = "renders"')
q('CREATE EDGE "page:editor" -> "comp:prompt_bar" kind = "renders"')
q('CREATE EDGE "page:editor" -> "comp:theme_palette" kind = "renders"')
q('CREATE EDGE "page:editor" -> "comp:preview_modal" kind = "renders"')
q('CREATE EDGE "page:editor" -> "comp:autoflow_modal" kind = "renders"')
q('CREATE EDGE "page:editor" -> "comp:debug_panel" kind = "renders"')

# Canvas renders sub-components
q('CREATE EDGE "comp:canvas" -> "canvas_node:screen" kind = "renders"')
q('CREATE EDGE "comp:canvas" -> "canvas_node:action" kind = "renders"')
q('CREATE EDGE "comp:canvas" -> "canvas_node:branch" kind = "renders"')
q('CREATE EDGE "comp:canvas" -> "canvas_node:context" kind = "renders"')
q('CREATE EDGE "comp:canvas" -> "canvas_node:state" kind = "renders"')
q('CREATE EDGE "comp:canvas" -> "canvas_node:variant" kind = "renders"')
q('CREATE EDGE "comp:canvas" -> "canvas_node:style" kind = "renders"')
q('CREATE EDGE "comp:canvas" -> "canvas_edge:zen" kind = "renders"')
q('CREATE EDGE "comp:canvas" -> "comp:canvas_toolbar" kind = "renders"')
q('CREATE EDGE "comp:canvas" -> "comp:layers_panel" kind = "renders"')
q('CREATE EDGE "comp:canvas" -> "comp:node_palette" kind = "renders"')
q('CREATE EDGE "comp:canvas" -> "comp:screen_context_menu" kind = "renders"')
q('CREATE EDGE "comp:canvas" -> "comp:tx_context_menu" kind = "renders"')
q('CREATE EDGE "comp:canvas" -> "comp:group_frames" kind = "renders"')
q('CREATE EDGE "comp:canvas" -> "comp:presence_cursors" kind = "renders"')
q('CREATE EDGE "comp:canvas" -> "comp:edge_label_editor" kind = "renders"')

# Inspector renders
q('CREATE EDGE "comp:inspector" -> "comp:screen_inspector" kind = "renders"')
q('CREATE EDGE "comp:inspector" -> "comp:transition_inspector" kind = "renders"')

# ═══════════════════════════════════════════════════════════════════
# EDGES — Components use hooks
# ═══════════════════════════════════════════════════════════════════

# Canvas hooks
q('CREATE EDGE "comp:canvas" -> "hook:useScreens" kind = "uses_hook"')
q('CREATE EDGE "comp:canvas" -> "hook:useTxNodes" kind = "uses_hook"')
q('CREATE EDGE "comp:canvas" -> "hook:useYjsEdges" kind = "uses_hook"')
q('CREATE EDGE "comp:canvas" -> "hook:useGroups" kind = "uses_hook"')
q('CREATE EDGE "comp:canvas" -> "hook:useUndoRedo" kind = "uses_hook"')
q('CREATE EDGE "comp:canvas" -> "hook:useCanvasShortcuts" kind = "uses_hook"')
q('CREATE EDGE "comp:canvas" -> "hook:useAutoLayout" kind = "uses_hook"')
q('CREATE EDGE "comp:canvas" -> "hook:useGroupDragSync" kind = "uses_hook"')
q('CREATE EDGE "comp:canvas" -> "hook:useGroupSelection" kind = "uses_hook"')
q('CREATE EDGE "comp:canvas" -> "hook:useNodeBuilder" kind = "uses_hook"')

# EditorPage hooks
q('CREATE EDGE "page:editor" -> "hook:useScreens" kind = "uses_hook"')
q('CREATE EDGE "page:editor" -> "hook:useSyncFlow" kind = "uses_hook"')
q('CREATE EDGE "page:editor" -> "hook:useToast" kind = "uses_hook"')

# Sidebar hooks
q('CREATE EDGE "comp:sidebar" -> "hook:useScreens" kind = "uses_hook"')
q('CREATE EDGE "comp:sidebar" -> "hook:useGroups" kind = "uses_hook"')

# PromptBar hooks
q('CREATE EDGE "comp:prompt_bar" -> "hook:useGeneration" kind = "uses_hook"')

# Inspector hooks
q('CREATE EDGE "comp:inspector" -> "hook:useScreens" kind = "uses_hook"')

# Layers panel hooks
q('CREATE EDGE "comp:layers_panel" -> "hook:useGroups" kind = "uses_hook"')
q('CREATE EDGE "comp:layers_panel" -> "hook:useScreens" kind = "uses_hook"')
q('CREATE EDGE "comp:layers_panel" -> "hook:useTxNodes" kind = "uses_hook"')

# Presence
q('CREATE EDGE "comp:presence_cursors" -> "hook:useAwareness" kind = "uses_hook"')

# ProjectsPage
q('CREATE EDGE "page:projects" -> "hook:useProjects" kind = "uses_hook"')

# ═══════════════════════════════════════════════════════════════════
# EDGES — Components/hooks read stores
# ═══════════════════════════════════════════════════════════════════

q('CREATE EDGE "page:editor" -> "store:editor" kind = "reads_store"')
q('CREATE EDGE "comp:canvas" -> "store:editor" kind = "reads_store"')
q('CREATE EDGE "comp:sidebar" -> "store:editor" kind = "reads_store"')
q('CREATE EDGE "comp:prompt_bar" -> "store:editor" kind = "reads_store"')
q('CREATE EDGE "comp:theme_palette" -> "store:editor" kind = "reads_store"')
q('CREATE EDGE "comp:settings_modal" -> "store:settings" kind = "reads_store"')
q('CREATE EDGE "hook:useScreens" -> "store:editor" kind = "reads_store"')
q('CREATE EDGE "hook:useTxNodes" -> "store:editor" kind = "reads_store"')
q('CREATE EDGE "hook:useSyncFlow" -> "store:auth" kind = "reads_store"')
q('CREATE EDGE "hook:useSyncFlow" -> "store:editor" kind = "reads_store"')
q('CREATE EDGE "page:login" -> "store:auth" kind = "reads_store"')

# ═══════════════════════════════════════════════════════════════════
# EDGES — Hooks observe Yjs maps
# ═══════════════════════════════════════════════════════════════════

q('CREATE EDGE "hook:useScreens" -> "yjs:screens" kind = "observes"')
q('CREATE EDGE "hook:useScreens" -> "yjs:layout" kind = "observes"')
q('CREATE EDGE "hook:useScreens" -> "yjs:layers" kind = "observes"')
q('CREATE EDGE "hook:useTxNodes" -> "yjs:txNodes" kind = "observes"')
q('CREATE EDGE "hook:useTxNodes" -> "yjs:layout" kind = "observes"')
q('CREATE EDGE "hook:useYjsEdges" -> "yjs:edges" kind = "observes"')
q('CREATE EDGE "hook:useGroups" -> "yjs:layers" kind = "observes"')
q('CREATE EDGE "hook:useUndoRedo" -> "module:yjs_doc" kind = "observes"')

# ═══════════════════════════════════════════════════════════════════
# EDGES — Hooks call Yjs actions (write path)
# ═══════════════════════════════════════════════════════════════════

q('CREATE EDGE "hook:useScreens" -> "yjs_act:addScreenYjs" kind = "calls"')
q('CREATE EDGE "hook:useScreens" -> "yjs_act:removeScreenYjs" kind = "calls"')
q('CREATE EDGE "hook:useScreens" -> "yjs_act:updateScreenYjs" kind = "calls"')
q('CREATE EDGE "hook:useScreens" -> "yjs_act:duplicateScreenYjs" kind = "calls"')
q('CREATE EDGE "hook:useScreens" -> "yjs_act:bulkSetScreensYjs" kind = "calls"')
q('CREATE EDGE "hook:useTxNodes" -> "yjs_act:addTransitionNode" kind = "calls"')
q('CREATE EDGE "hook:useTxNodes" -> "yjs_act:updateTransitionNode" kind = "calls"')
q('CREATE EDGE "hook:useTxNodes" -> "yjs_act:deleteTransitionNode" kind = "calls"')
q('CREATE EDGE "hook:useYjsEdges" -> "yjs_act:addEdgeYjs" kind = "calls"')
q('CREATE EDGE "hook:useYjsEdges" -> "yjs_act:removeEdgeYjs" kind = "calls"')
q('CREATE EDGE "hook:useYjsEdges" -> "yjs_act:bulkSetEdgesYjs" kind = "calls"')
q('CREATE EDGE "hook:useGroups" -> "yjs_act:addGroupYjs" kind = "calls"')
q('CREATE EDGE "hook:useGroups" -> "yjs_act:removeGroupYjs" kind = "calls"')

# Yjs actions mutate Yjs maps
q('CREATE EDGE "yjs_act:addScreenYjs" -> "yjs:screens" kind = "mutates"')
q('CREATE EDGE "yjs_act:addScreenYjs" -> "yjs:layout" kind = "mutates"')
q('CREATE EDGE "yjs_act:addScreenYjs" -> "yjs:layers" kind = "mutates"')
q('CREATE EDGE "yjs_act:removeScreenYjs" -> "yjs:screens" kind = "mutates"')
q('CREATE EDGE "yjs_act:removeScreenYjs" -> "yjs:edges" kind = "mutates"')
q('CREATE EDGE "yjs_act:updateScreenYjs" -> "yjs:screens" kind = "mutates"')
q('CREATE EDGE "yjs_act:updateScreenYjs" -> "yjs:layout" kind = "mutates"')
q('CREATE EDGE "yjs_act:duplicateScreenYjs" -> "yjs:screens" kind = "mutates"')
q('CREATE EDGE "yjs_act:duplicateScreenYjs" -> "yjs:layout" kind = "mutates"')
q('CREATE EDGE "yjs_act:duplicateScreenYjs" -> "yjs:layers" kind = "mutates"')
q('CREATE EDGE "yjs_act:bulkSetScreensYjs" -> "yjs:screens" kind = "mutates"')
q('CREATE EDGE "yjs_act:bulkSetScreensYjs" -> "yjs:layout" kind = "mutates"')
q('CREATE EDGE "yjs_act:bulkSetScreensYjs" -> "yjs:layers" kind = "mutates"')
q('CREATE EDGE "yjs_act:addTransitionNode" -> "yjs:txNodes" kind = "mutates"')
q('CREATE EDGE "yjs_act:addTransitionNode" -> "yjs:layout" kind = "mutates"')
q('CREATE EDGE "yjs_act:addTransitionNode" -> "yjs:layers" kind = "mutates"')
q('CREATE EDGE "yjs_act:updateTransitionNode" -> "yjs:txNodes" kind = "mutates"')
q('CREATE EDGE "yjs_act:deleteTransitionNode" -> "yjs:txNodes" kind = "mutates"')
q('CREATE EDGE "yjs_act:deleteTransitionNode" -> "yjs:edges" kind = "mutates"')
q('CREATE EDGE "yjs_act:addEdgeYjs" -> "yjs:edges" kind = "mutates"')
q('CREATE EDGE "yjs_act:removeEdgeYjs" -> "yjs:edges" kind = "mutates"')
q('CREATE EDGE "yjs_act:bulkSetEdgesYjs" -> "yjs:edges" kind = "mutates"')
q('CREATE EDGE "yjs_act:addGroupYjs" -> "yjs:layers" kind = "mutates"')
q('CREATE EDGE "yjs_act:removeGroupYjs" -> "yjs:layers" kind = "mutates"')
q('CREATE EDGE "yjs_act:detachAtBoundary" -> "yjs:edges" kind = "mutates"')
q('CREATE EDGE "yjs_act:reconnectFragments" -> "yjs:edges" kind = "mutates"')
q('CREATE EDGE "yjs_act:updateTheme" -> "yjs:meta" kind = "mutates"')

# ═══════════════════════════════════════════════════════════════════
# EDGES — Frontend API calls
# ═══════════════════════════════════════════════════════════════════

q('CREATE EDGE "comp:prompt_homepage" -> "api:projects_create" kind = "api_call"')
q('CREATE EDGE "page:editor" -> "api:project_get" kind = "api_call"')
q('CREATE EDGE "page:editor" -> "api:generate" kind = "api_call"')
q('CREATE EDGE "comp:prompt_bar" -> "api:generate" kind = "api_call"')
q('CREATE EDGE "comp:autoflow_modal" -> "api:autoflow_check" kind = "api_call"')
q('CREATE EDGE "comp:autoflow_modal" -> "api:autoflow_run" kind = "api_call"')
q('CREATE EDGE "page:projects" -> "api:projects_list" kind = "api_call"')
q('CREATE EDGE "page:login" -> "api:login" kind = "api_call"')
q('CREATE EDGE "comp:history_panel" -> "api:restore_points" kind = "api_call"')
q('CREATE EDGE "comp:history_panel" -> "api:restore" kind = "api_call"')

# ═══════════════════════════════════════════════════════════════════
# EDGES — Backend pipeline flow
# ═══════════════════════════════════════════════════════════════════

# generate route pipeline
q('CREATE EDGE "api:generate" -> "pipe:chargeCredits" kind = "calls"')
q('CREATE EDGE "api:generate" -> "pipe:acquireLock" kind = "calls"')
q('CREATE EDGE "api:generate" -> "pipe:buildStackForScreen" kind = "calls"')
q('CREATE EDGE "api:generate" -> "pipe:executeStack" kind = "calls"')
q('CREATE EDGE "api:generate" -> "pipe:runMockupPipeline" kind = "calls"')
q('CREATE EDGE "api:generate" -> "pipe:persistGeneration" kind = "calls"')
q('CREATE EDGE "api:generate" -> "pipe:appendLineage" kind = "calls"')
q('CREATE EDGE "api:generate" -> "pipe:applyGenerationToDoc" kind = "calls"')
q('CREATE EDGE "api:generate" -> "pipe:createRestorePoint" kind = "calls"')

# autoflow route pipeline
q('CREATE EDGE "api:autoflow_run" -> "pipe:runMockupPipeline" kind = "calls"')
q('CREATE EDGE "api:autoflow_run" -> "pipe:applyGenerationToDoc" kind = "calls"')
q('CREATE EDGE "api:autoflow_run" -> "pipe:chargeCredits" kind = "calls"')
q('CREATE EDGE "api:autoflow_run" -> "pipe:createRestorePoint" kind = "calls"')

# Pipeline internal flow
q('CREATE EDGE "pipe:runMockupPipeline" -> "pipe:runIntentLayer" kind = "calls"')
q('CREATE EDGE "pipe:runMockupPipeline" -> "pipe:buildScreenSeeds" kind = "calls"')
q('CREATE EDGE "pipe:runMockupPipeline" -> "pipe:buildFlexibleScreenSpec" kind = "calls"')
q('CREATE EDGE "pipe:runMockupPipeline" -> "pipe:buildDesignTokens" kind = "calls"')
q('CREATE EDGE "pipe:runMockupPipeline" -> "pipe:generateScreenHtml" kind = "calls"')

# LLM calls
q('CREATE EDGE "pipe:runIntentLayer" -> "pipe:callLLM" kind = "calls"')
q('CREATE EDGE "pipe:runIntentLayer" -> "pipe:buildFallbackIntent" kind = "fallback"')
q('CREATE EDGE "pipe:buildFlexibleScreenSpec" -> "pipe:callLLM" kind = "calls"')
q('CREATE EDGE "pipe:buildFlexibleScreenSpec" -> "pipe:buildFallbackSpec" kind = "fallback"')
q('CREATE EDGE "pipe:generateScreenHtml" -> "pipe:callLLM" kind = "calls"')
q('CREATE EDGE "pipe:generateScreenHtml" -> "pipe:buildFallbackBody" kind = "fallback"')
q('CREATE EDGE "pipe:callLLM" -> "pipe:cache" kind = "calls"')

# applyGenerationToDoc writes to Yjs
q('CREATE EDGE "pipe:applyGenerationToDoc" -> "yjs:screens" kind = "mutates"')
q('CREATE EDGE "pipe:applyGenerationToDoc" -> "yjs:layout" kind = "mutates"')
q('CREATE EDGE "pipe:applyGenerationToDoc" -> "yjs:layers" kind = "mutates"')
q('CREATE EDGE "pipe:applyGenerationToDoc" -> "yjs:edges" kind = "mutates"')
q('CREATE EDGE "pipe:applyGenerationToDoc" -> "yjs:txNodes" kind = "mutates"')

# ═══════════════════════════════════════════════════════════════════
# EDGES — Yjs sync flow (WebSocket)
# ═══════════════════════════════════════════════════════════════════

q('CREATE EDGE "hook:useSyncFlow" -> "module:yjs_doc" kind = "manages"')
q('CREATE EDGE "module:yjs_doc" -> "yjs:screens" kind = "owns"')
q('CREATE EDGE "module:yjs_doc" -> "yjs:txNodes" kind = "owns"')
q('CREATE EDGE "module:yjs_doc" -> "yjs:edges" kind = "owns"')
q('CREATE EDGE "module:yjs_doc" -> "yjs:layout" kind = "owns"')
q('CREATE EDGE "module:yjs_doc" -> "yjs:layers" kind = "owns"')
q('CREATE EDGE "module:yjs_doc" -> "yjs:meta" kind = "owns"')
q('CREATE EDGE "module:yjs_doc" -> "yjs:dict" kind = "owns"')

# ═══════════════════════════════════════════════════════════════════
# EDGES — Navigation flow
# ═══════════════════════════════════════════════════════════════════

q('CREATE EDGE "comp:prompt_homepage" -> "page:editor" kind = "navigates" label = "pendingPrompt in state"')
q('CREATE EDGE "page:projects" -> "page:editor" kind = "navigates"')
q('CREATE EDGE "page:login" -> "page:home" kind = "navigates"')
q('CREATE EDGE "comp:project_card" -> "page:editor" kind = "navigates"')

# ═══════════════════════════════════════════════════════════════════
# EDGES — EditorPage generation trigger flow (the bug area)
# ═══════════════════════════════════════════════════════════════════

q('CREATE EDGE "page:editor" -> "action:setActiveScreen" kind = "calls"')
q('CREATE EDGE "page:editor" -> "action:setYjsSynced" kind = "calls"')
q('CREATE EDGE "page:editor" -> "action:addGeneratingId" kind = "calls"')
q('CREATE EDGE "page:editor" -> "action:removeGeneratingId" kind = "calls"')
q('CREATE EDGE "page:editor" -> "action:resetEditor" kind = "calls"')

# Store actions to store
q('CREATE EDGE "action:setActiveScreen" -> "store:editor" kind = "writes_store"')
q('CREATE EDGE "action:setActiveNode" -> "store:editor" kind = "writes_store"')
q('CREATE EDGE "action:addGeneratingId" -> "store:editor" kind = "writes_store"')
q('CREATE EDGE "action:removeGeneratingId" -> "store:editor" kind = "writes_store"')
q('CREATE EDGE "action:updateTheme" -> "store:editor" kind = "writes_store"')
q('CREATE EDGE "action:resetEditor" -> "store:editor" kind = "writes_store"')
q('CREATE EDGE "action:setYjsSynced" -> "store:editor" kind = "writes_store"')
q('CREATE EDGE "action:setEditorMode" -> "store:editor" kind = "writes_store"')
q('CREATE EDGE "action:toggleCanvasDark" -> "store:editor" kind = "writes_store"')

# Canvas toolbar actions
q('CREATE EDGE "comp:canvas_toolbar" -> "hook:useAutoLayout" kind = "uses_hook"')
q('CREATE EDGE "comp:canvas_toolbar" -> "action:toggleCanvasDark" kind = "calls"')
q('CREATE EDGE "comp:canvas_toolbar" -> "action:setEditorMode" kind = "calls"')

# Context menu actions
q('CREATE EDGE "comp:screen_context_menu" -> "hook:useScreens" kind = "uses_hook"')
q('CREATE EDGE "comp:tx_context_menu" -> "hook:useTxNodes" kind = "uses_hook"')

# Store the script so the playground editor shows it
g.set_script("\n".join(_all_queries))

# Checkpoint and save
g.checkpoint()
print("=== Graph populated successfully ===")

stats = g.execute("SYS STATS")
print(f"Stats: {stats.data}")
stats_nodes = g.execute("SYS STATS NODES")
print(f"Node stats: {stats_nodes.data}")
