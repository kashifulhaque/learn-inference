import { useRef, useState, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import { Link } from "react-router-dom";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import macros from "../lib/katexMacros.json";
import { rehypeSectionize, remarkCallouts, remarkMark, remarkViz } from "../lib/markdownPlugins";
import Viz from "./viz/Viz";

/** A link that stays inside the app, as opposed to one that leaves it. */
function isInternal(href: string): boolean {
  return href.startsWith("/") && !href.startsWith("//");
}

/** The language rehype-highlight tagged a fenced block with, if any. */
function languageOf(children: ReactNode): string {
  const child = Array.isArray(children) ? children[0] : children;
  const className =
    child && typeof child === "object" && "props" in child
      ? String((child.props as { className?: string }).className ?? "")
      : "";
  const match = /language-([\w+-]+)/.exec(className);
  return match ? match[1] : "";
}

function CodeBlock({ children }: { children: ReactNode }) {
  const pre = useRef<HTMLPreElement>(null);
  const [copied, setCopied] = useState(false);
  const language = languageOf(children);

  function copy() {
    const text = pre.current?.innerText ?? "";
    navigator.clipboard
      ?.writeText(text)
      .then(() => {
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1400);
      })
      .catch(() => undefined);
  }

  return (
    <div className="code-block">
      <div className="code-block-bar">
        <span>{language || "text"}</span>
        <button type="button" onClick={copy} aria-label="Copy the code">
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre ref={pre}>{children}</pre>
    </div>
  );
}

export default function Markdown({ children }: { children: string }) {
  return (
    <div className="prose-chapter">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath, remarkViz, remarkCallouts, remarkMark]}
        rehypePlugins={[
          // Sections first, so each one wraps the source headings before KaTeX
          // and the highlighter add their own markup.
          rehypeSectionize,
          // KaTeX mutates the macro table it is given, so hand it a copy. The
          // colour macros expand to \htmlClass, the one HTML extension allowed,
          // so the theme decides their colour. scripts/check_math.mjs matches.
          [
            rehypeKatex,
            {
              macros: { ...macros },
              strict: (code: string) => (code === "htmlExtension" ? "ignore" : "warn"),
              trust: (context: { command: string }) => context.command === "\\htmlClass",
            },
          ],
          [rehypeHighlight, { detect: true, ignoreMissing: true }],
        ]}
        components={{
          a: ({ href, children }) =>
            // A cross-reference to another chapter routes in place. Sending it
            // to a new tab would lose the reader's scroll position and their
            // lab draft.
            href && isInternal(href) ? (
              <Link to={href}>{children}</Link>
            ) : (
              <a href={href} target="_blank" rel="noreferrer noopener">
                {children}
              </a>
            ),
          // remarkViz leaves a placeholder where a ```viz fence was.
          div: ({ node: _node, ...props }) => {
            const name = (props as Record<string, unknown>)["data-viz"];
            return typeof name === "string" && String(props.className).includes("viz-embed") ? (
              <Viz name={name} />
            ) : (
              <div {...props} />
            );
          },
          pre: ({ children }) => <CodeBlock>{children}</CodeBlock>,
          table: ({ children }) => (
            <div className="table-wrap">
              <table>{children}</table>
            </div>
          ),
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
