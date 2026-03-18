import CodeMirror from '@uiw/react-codemirror'
import { graphstoreLang } from '@/lang/graphstore'
import { useGraphStore } from '@/hooks/useGraphStore'
import { keymap, type ViewUpdate } from '@codemirror/view'
import type { EditorView } from '@codemirror/view'
import { useCallback, useMemo } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { Play, PlayCircle } from 'lucide-react'

function toggleComment(view: EditorView): boolean {
  const { state } = view
  const { from, to } = state.selection.main
  const fromLine = state.doc.lineAt(from).number
  const toLine = state.doc.lineAt(to).number

  const lines: { line: number; text: string }[] = []
  for (let i = fromLine; i <= toLine; i++) {
    const line = state.doc.line(i)
    lines.push({ line: i, text: line.text })
  }

  const allCommented = lines.every(l => l.text.trimStart().startsWith('//'))

  const changes = lines.map(l => {
    const line = state.doc.line(l.line)
    if (allCommented) {
      // Uncomment: remove first occurrence of "// " or "//"
      const idx = line.text.indexOf('//')
      const len = line.text[idx + 2] === ' ' ? 3 : 2
      return { from: line.from + idx, to: line.from + idx + len, insert: '' }
    } else {
      // Comment: prepend "// "
      return { from: line.from, to: line.from, insert: '// ' }
    }
  })

  view.dispatch({ changes })
  return true
}

export function EditorPanel() {
  const editorContent = useGraphStore((s) => s.editorContent)
  const setEditorContent = useGraphStore((s) => s.setEditorContent)
  const setEditorSelection = useGraphStore((s) => s.setEditorSelection)
  const executeQuery = useGraphStore((s) => s.executeQuery)
  const executeAll = useGraphStore((s) => s.executeAll)
  const executeSelected = useGraphStore((s) => s.executeSelected)
  const editorSelection = useGraphStore((s) => s.editorSelection)
  const isDark = useGraphStore((s) => s.config.isDark)
  const hasSelection = editorSelection.trim().length > 0

  const getFullLines = useCallback((view: EditorView) => {
    const { from, to } = view.state.selection.main
    if (from === to) return ''
    const firstLine = view.state.doc.lineAt(from)
    const lastLine = view.state.doc.lineAt(to)
    return view.state.sliceDoc(firstLine.from, lastLine.to)
  }, [])

  const handleRunSelected = useCallback(
    (view: EditorView) => {
      const text = getFullLines(view) || view.state.doc.lineAt(view.state.selection.main.head).text
      if (text.trim()) executeQuery(text.trim())
      return true
    },
    [executeQuery, getFullLines],
  )

  const onUpdate = useCallback(
    (update: ViewUpdate) => {
      if (!update.selectionSet) return
      const { from, to } = update.state.selection.main
      if (from === to) {
        setEditorSelection('')
        return
      }
      const firstLine = update.state.doc.lineAt(from)
      const lastLine = update.state.doc.lineAt(to)
      setEditorSelection(update.state.sliceDoc(firstLine.from, lastLine.to))
    },
    [setEditorSelection],
  )

  const extensions = useMemo(
    () => [
      graphstoreLang,
      keymap.of([
        { key: 'Ctrl-Enter', run: handleRunSelected },
        { key: 'Ctrl-Shift-Enter', run: () => { executeAll(); return true } },
        { key: 'Ctrl-/', run: toggleComment },
      ]),
    ],
    [handleRunSelected, executeAll],
  )

  return (
    <Card className="h-full gap-0 py-0 rounded-lg">
      <CardHeader className="px-3 py-2 border-b">
        <CardTitle className="text-xs text-muted-foreground">Editor</CardTitle>
      </CardHeader>
      <CardContent className="flex-1 overflow-hidden p-0">
        <CodeMirror
          value={editorContent}
          onChange={setEditorContent}
          onUpdate={onUpdate}
          extensions={extensions}
          theme={isDark ? 'dark' : 'light'}
          height="100%"
          className="h-full text-sm [&_.cm-editor]:!h-full [&_.cm-scroller]:!overflow-auto"
          basicSetup={{
            lineNumbers: true,
            foldGutter: false,
            highlightActiveLine: true,
            autocompletion: false,
          }}
        />
      </CardContent>
      <CardFooter className="justify-end px-3 py-1.5">
        <Button
          variant="ghost"
          size="sm"
          className="h-7 text-xs gap-1.5"
          onMouseDown={(e) => { e.preventDefault(); hasSelection ? executeSelected() : executeAll() }}
        >
          {hasSelection
            ? <><Play className="w-3.5 h-3.5" /> Run Selected</>
            : <><PlayCircle className="w-3.5 h-3.5" /> Run All</>
          }
        </Button>
      </CardFooter>
    </Card>
  )
}
