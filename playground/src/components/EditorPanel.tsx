import CodeMirror from '@uiw/react-codemirror'
import { graphstoreLang } from '@/lang/graphstore'
import { useGraphStore } from '@/hooks/useGraphStore'
import { keymap, type ViewUpdate } from '@codemirror/view'
import { useCallback, useMemo } from 'react'
import type { EditorView } from '@codemirror/view'

export function EditorPanel() {
  const editorContent = useGraphStore((s) => s.editorContent)
  const setEditorContent = useGraphStore((s) => s.setEditorContent)
  const setEditorSelection = useGraphStore((s) => s.setEditorSelection)
  const executeQuery = useGraphStore((s) => s.executeQuery)
  const executeAll = useGraphStore((s) => s.executeAll)

  const handleRunSelected = useCallback(
    (view: EditorView) => {
      const sel = view.state.sliceDoc(
        view.state.selection.main.from,
        view.state.selection.main.to,
      )
      const text = sel || view.state.doc.lineAt(view.state.selection.main.head).text
      if (text.trim()) executeQuery(text.trim())
      return true
    },
    [executeQuery],
  )

  const onUpdate = useCallback(
    (update: ViewUpdate) => {
      const sel = update.state.sliceDoc(
        update.state.selection.main.from,
        update.state.selection.main.to,
      )
      setEditorSelection(sel)
    },
    [setEditorSelection],
  )

  const extensions = useMemo(
    () => [
      graphstoreLang,
      keymap.of([
        { key: 'Ctrl-Enter', run: handleRunSelected },
        { key: 'Ctrl-Shift-Enter', run: () => { executeAll(); return true } },
      ]),
    ],
    [handleRunSelected, executeAll],
  )

  return (
    <div className="h-full flex flex-col bg-background">
      <CodeMirror
        value={editorContent}
        onChange={setEditorContent}
        onUpdate={onUpdate}
        extensions={extensions}
        theme="dark"
        height="100%"
        className="flex-1 overflow-auto text-sm [&_.cm-editor]:!h-full [&_.cm-scroller]:!overflow-auto"
        basicSetup={{
          lineNumbers: true,
          foldGutter: false,
          highlightActiveLine: true,
          autocompletion: false,
        }}
      />
    </div>
  )
}
