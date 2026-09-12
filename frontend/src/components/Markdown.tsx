import ReactMarkdown from "react-markdown";
import { Link } from "react-router-dom";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";

/** A link that stays inside the app, as opposed to one that leaves it. */
function isInternal(href: string): boolean {
  return href.startsWith("/") && !href.startsWith("//");
}

export default function Markdown({ children }: { children: string }) {
  return (
    <div className="prose-chapter">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, [rehypeHighlight, { detect: true, ignoreMissing: true }]]}
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
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
