import { useState } from "react";
import Editor, { type OnMount } from "@monaco-editor/react";
import {
  definePaperTheme,
  editorLook,
  editorThemeName,
  fontHasLigatures,
  fontStack,
  useEditorPrefs,
  useMonacoPrefs,
} from "../lib/editorPrefs";
import { DARK_THEME, LIGHT_THEME, monaco as bundledMonaco } from "../lib/monaco";
import { useTheme } from "../lib/theme";

type Props = {
  value: string;
  onChange: (value: string) => void;
  /** Called on Ctrl+Enter or Cmd+Enter inside the editor, with or without Shift. */
  onRun?: () => void;
  /** Path shown to Monaco, so it picks the language and keeps undo history per lab. */
  path: string;
  /**
   * Fill the parent instead of growing with the file. The split view gives the
   * editor a pane of its own, so there the height is the pane's to decide.
   */
  fill?: boolean;
};

// The editor's own colour themes, one per page theme, plus the shared light
// paper theme that the editor settings offer. Defined here, in the lazy
// editor chunk, so Monaco stays out of the main bundle.
const THEMES = { light: LIGHT_THEME, dark: DARK_THEME };
definePaperTheme(bundledMonaco);

type MountedEditor = Parameters<OnMount>[0];
type MountedMonaco = Parameters<OnMount>[1];

const LINE_HEIGHT = 22;
const PADDING = 18;
const MIN_HEIGHT = 18 * LINE_HEIGHT + PADDING * 2;
const MAX_HEIGHT = 34 * LINE_HEIGHT + PADDING * 2;

function clampHeight(contentHeight: number): number {
  return Math.min(MAX_HEIGHT, Math.max(MIN_HEIGHT, contentHeight));
}

/**
 * The lab editor. In a pane it fills what it is given; laid out in a page it
 * grows with the file up to a cap, so short starters do not sit in an empty box
 * and long ones do not trap the page scroll. The theme, font, size, and Vim
 * mode come from the shared editor prefs.
 */
export default function CodeEditor({ value, onChange, onRun, path, fill = false }: Props) {
  const [height, setHeight] = useState(MIN_HEIGHT);
  const site = useTheme();
  const [prefs] = useEditorPrefs();
  const theme = editorThemeName(editorLook(prefs.theme, site), THEMES);
  const [editor, setEditor] = useState<MountedEditor | null>(null);
  const [instance, setInstance] = useState<MountedMonaco | null>(null);
  const [vimStatus, setVimStatus] = useState<HTMLDivElement | null>(null);

  useMonacoPrefs({ editor, monaco: instance, site, themes: THEMES, vimStatus, onRun, onCheck: onRun });

  const handleMount: OnMount = (mounted, loaded) => {
    setEditor(mounted);
    setInstance(loaded);
    const editor = mounted;
    if (!fill) {
      setHeight(clampHeight(editor.getContentHeight()));
      editor.onDidContentSizeChange((event) => {
        setHeight(clampHeight(event.contentHeight));
      });
    }
    // Monaco measures glyphs once, at mount. If the mono font arrives after
    // that, the cursor drifts from the text until the metrics are taken again.
    document.fonts?.ready.then(() => loaded.editor.remeasureFonts());
  };

  return (
    <div
      className={`flex flex-col bg-code ${fill ? "h-full min-h-0" : ""}`}
      style={fill ? undefined : { height }}
    >
      <div className="min-h-0 flex-1">
      <Editor
        height="100%"
        path={path}
        defaultLanguage="python"
        theme={theme === "dark" ? DARK_THEME : LIGHT_THEME}
        value={value}
        onChange={(next) => onChange(next ?? "")}
        onMount={handleMount}
        loading={
          <div className="flex h-full w-full items-center justify-center bg-code text-[12.5px] text-code-muted">
            Loading the editor…
          </div>
        }
        options={{
          // @monaco-editor/react reapplies these on every render, so they
          // must come from the prefs too, or a re-render resets the font.
          fontSize: prefs.fontSize,
          lineHeight: Math.round(prefs.fontSize * 1.55),
          fontFamily: fontStack(prefs.fontFamily),
          fontLigatures: fontHasLigatures(prefs.fontFamily),
          padding: { top: PADDING, bottom: PADDING },
          tabSize: 4,
          insertSpaces: true,
          detectIndentation: false,
          wordWrap: "on",
          wrappingIndent: "indent",
          minimap: { enabled: false },
          lineNumbersMinChars: 3,
          lineDecorationsWidth: 14,
          glyphMargin: false,
          folding: true,
          showFoldingControls: "mouseover",
          renderLineHighlight: "line",
          renderLineHighlightOnlyWhenFocus: true,
          renderWhitespace: "selection",
          guides: { indentation: true, bracketPairs: false },
          // The slab theme colours brackets as punctuation, not in rainbow pairs.
          bracketPairColorization: { enabled: false },
          matchBrackets: "always",
          rulers: [],
          overviewRulerLanes: 0,
          overviewRulerBorder: false,
          hideCursorInOverviewRuler: true,
          scrollbar: {
            vertical: "auto",
            horizontal: "hidden",
            verticalScrollbarSize: 10,
            useShadows: false,
            alwaysConsumeMouseWheel: false,
          },
          scrollBeyondLastLine: false,
          smoothScrolling: true,
          cursorBlinking: "smooth",
          cursorSmoothCaretAnimation: "on",
          stickyScroll: { enabled: false },
          // There is no language server behind this editor, so the only
          // completions are words from the file. Offer them on demand
          // (Ctrl+Space) rather than after every keystroke.
          quickSuggestions: false,
          suggestOnTriggerCharacters: false,
          wordBasedSuggestions: "currentDocument",
          parameterHints: { enabled: false },
          acceptSuggestionOnEnter: "off",
          // The starters use em dashes in docstrings. Without this Monaco
          // boxes every one of them as an ambiguous character.
          unicodeHighlight: { ambiguousCharacters: false, invisibleCharacters: false },
          fixedOverflowWidgets: true,
          automaticLayout: true,
        }}
      />
      </div>
      {prefs.vim && (
        <div ref={setVimStatus} className="sh-vim-status" aria-live="polite" aria-label="Vim status" />
      )}
    </div>
  );
}
