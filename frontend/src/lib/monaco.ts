/**
 * Monaco, bundled with the app instead of fetched from a CDN at runtime.
 *
 * `@monaco-editor/react` loads the editor from jsdelivr unless it is handed an
 * instance, which made the first paint of every lab wait on a third-party
 * download. This module builds a lean instance with the features and the one
 * language the labs need, gives it the app's colours, and registers it with the
 * loader. Import it from the editor component only, so the bundle stays in its
 * own chunk and pages without a lab never download it.
 */
import { loader } from "@monaco-editor/react";
import * as monaco from "monaco-editor/editor/editor.api";
import EditorWorker from "monaco-editor/editor/editor.worker?worker";

// Editor features. The full set adds a diff editor, a colour picker, code
// lenses, and other machinery that has no provider here.
import "monaco-editor/features/codicon/register";
import "monaco-editor/features/anchorSelect/register";
import "monaco-editor/features/bracketMatching/register";
import "monaco-editor/features/caretOperations/register";
import "monaco-editor/features/clipboard/register";
import "monaco-editor/features/comment/register";
import "monaco-editor/features/contextmenu/register";
import "monaco-editor/features/cursorUndo/register";
import "monaco-editor/features/dnd/register";
import "monaco-editor/features/find/register";
import "monaco-editor/features/folding/register";
import "monaco-editor/features/fontZoom/register";
import "monaco-editor/features/indentation/register";
import "monaco-editor/features/lineSelection/register";
import "monaco-editor/features/linesOperations/register";
import "monaco-editor/features/links/register";
import "monaco-editor/features/longLinesHelper/register";
import "monaco-editor/features/multicursor/register";
import "monaco-editor/features/placeholderText/register";
import "monaco-editor/features/quickCommand/register";
import "monaco-editor/features/readOnlyMessage/register";
import "monaco-editor/features/smartSelect/register";
import "monaco-editor/features/snippet/register";
import "monaco-editor/features/suggest/register";
import "monaco-editor/features/tokenization/register";
import "monaco-editor/features/unicodeHighlighter/register";
import "monaco-editor/features/wordHighlighter/register";
import "monaco-editor/features/wordOperations/register";
import "monaco-editor/features/wordPartOperations/register";

// The one language the labs are written in.
import "monaco-editor/languages/definitions/python/register";

self.MonacoEnvironment = {
  getWorker: () => new EditorWorker(),
};

/** The editor theme. Values mirror the `ink`, `flame`, and `mint` tokens in index.css. */
export const THEME = "ink";

monaco.editor.defineTheme(THEME, {
  base: "vs-dark",
  inherit: true,
  rules: [
    { token: "", foreground: "d7e2eb" },
    { token: "keyword", foreground: "87b7ff" },
    { token: "string", foreground: "91f4ce" },
    { token: "string.escape", foreground: "c0ffe7" },
    { token: "comment", foreground: "6f88a3", fontStyle: "italic" },
    { token: "number", foreground: "ffc582" },
    { token: "tag", foreground: "c9a8ff" },
    { token: "type", foreground: "f4f8fb" },
    { token: "delimiter", foreground: "92a8bc" },
    { token: "delimiter.parenthesis", foreground: "92a8bc" },
    { token: "identifier", foreground: "d7e2eb" },
  ],
  colors: {
    "editor.background": "#07111d",
    "editor.foreground": "#d7e2eb",
    "editorCursor.foreground": "#91f4ce",
    "editor.lineHighlightBackground": "#0b1726",
    "editor.lineHighlightBorder": "#0b172600",
    "editor.selectionBackground": "#65e6b03a",
    "editor.inactiveSelectionBackground": "#65e6b01f",
    "editor.selectionHighlightBackground": "#65e6b01a",
    "editor.wordHighlightBackground": "#87b7ff1f",
    "editor.wordHighlightStrongBackground": "#87b7ff2e",
    "editor.findMatchBackground": "#ffc58255",
    "editor.findMatchHighlightBackground": "#ffc58226",
    "editorLineNumber.foreground": "#46617f",
    "editorLineNumber.activeForeground": "#b7c7d5",
    "editorIndentGuide.background1": "#172a40",
    "editorIndentGuide.activeBackground1": "#27415e",
    "editorWhitespace.foreground": "#27415e",
    "editorBracketMatch.background": "#65e6b01f",
    "editorBracketMatch.border": "#65e6b080",
    "editorBracketHighlight.foreground1": "#92a8bc",
    "editorBracketHighlight.foreground2": "#87b7ff",
    "editorBracketHighlight.foreground3": "#91f4ce",
    "editorGutter.background": "#07111d",
    "editorWidget.background": "#0b1726",
    "editorWidget.border": "#27415e",
    "editorSuggestWidget.background": "#0b1726",
    "editorSuggestWidget.border": "#27415e",
    "editorSuggestWidget.selectedBackground": "#172a40",
    "editorSuggestWidget.highlightForeground": "#91f4ce",
    "editorHoverWidget.background": "#0b1726",
    "editorHoverWidget.border": "#27415e",
    "input.background": "#07111d",
    "input.border": "#27415e",
    "input.foreground": "#d7e2eb",
    "focusBorder": "#65e6b0",
    "list.hoverBackground": "#101f32",
    "list.focusBackground": "#172a40",
    "list.highlightForeground": "#91f4ce",
    "menu.background": "#0b1726",
    "menu.foreground": "#d7e2eb",
    "menu.selectionBackground": "#172a40",
    "menu.separatorBackground": "#27415e",
    "scrollbar.shadow": "#00000000",
    "scrollbarSlider.background": "#27415e80",
    "scrollbarSlider.hoverBackground": "#46617f99",
    "scrollbarSlider.activeBackground": "#46617fcc",
    "minimap.background": "#07111d",
    "editorOverviewRuler.border": "#00000000",
  },
});

loader.config({ monaco });

export { monaco };
