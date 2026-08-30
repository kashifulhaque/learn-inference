import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, type Chapter } from "../lib/api";
import Markdown from "../components/Markdown";
import LabPane from "../components/LabPane";

type Props = {
  progress: Record<string, string>;
  onProgress: (slug: string, status: "in_progress" | "done") => void;
};

export default function ChapterPage({ progress, onProgress }: Props) {
  const { slug = "" } = useParams();
  const [chapter, setChapter] = useState<Chapter | null>(null);
  const [note, setNote] = useState("");
  const [error, setError] = useState("");
  const noteTimer = useRef<number | null>(null);
  const top = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setChapter(null);
    setError("");
    api
      .chapter(slug)
      .then((result) => {
        setChapter(result);
        setNote(result.note);
        if (!progress[slug]) onProgress(slug, "in_progress");
      })
      .catch((e) => setError(String(e)));
    top.current?.scrollIntoView();
    window.scrollTo(0, 0);
    // Progress is intentionally excluded: marking a chapter read must not refetch.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [slug]);

  function editNote(value: string) {
    setNote(value);
    if (noteTimer.current) window.clearTimeout(noteTimer.current);
    noteTimer.current = window.setTimeout(() => {
      api.saveNote(slug, value).catch(() => undefined);
    }, 800);
  }

  if (error) {
    return <p className="p-10 text-sm text-rose-450">{error}</p>;
  }
  if (!chapter) {
    return <p className="p-10 text-sm text-ink-500">Loading…</p>;
  }

  const done = progress[slug] === "done";

  return (
    <div ref={top} className="mx-auto max-w-5xl px-5 py-8 sm:px-8 lg:px-12 lg:py-11">
      <div className="mb-7 flex flex-wrap items-center gap-2.5 text-xs">
        <span className="rounded-lg border border-ink-700 bg-ink-900 px-2.5 py-1.5 font-medium text-ink-300">
          {chapter.part}
        </span>
        {chapter.minutes && (
          <span className="rounded-lg border border-ink-800 bg-ink-900/60 px-2.5 py-1.5 text-ink-500">
            {chapter.minutes} min read
          </span>
        )}
        {chapter.gpu && (
          <span className="rounded-lg border border-flame-500/30 bg-flame-500/10 px-2.5 py-1.5 font-semibold text-flame-400">
            GPU lab included
          </span>
        )}
        <span className="ml-auto hidden items-center gap-2 text-[11px] text-ink-500 sm:flex">
          <span className={`h-2 w-2 rounded-full ${done ? "bg-mint-400" : "bg-flame-500"}`} />
          {done ? "Completed" : "In progress"}
        </span>
      </div>

      {chapter.objectives.length > 0 && (
        <section className="panel-glow mb-9 overflow-hidden rounded-2xl border border-ink-800 bg-ink-900/70">
          <div className="flex items-center gap-3 border-b border-ink-800 bg-ink-850/60 px-5 py-3">
            <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-flame-500/15 text-xs font-bold text-flame-400">
              ✓
            </span>
            <div className="text-[10px] font-bold uppercase tracking-[0.18em] text-ink-400">
              Learning objectives
            </div>
          </div>
          <ul className="grid gap-x-8 gap-y-3 px-5 py-5 text-sm leading-6 text-ink-300 sm:grid-cols-2">
            {chapter.objectives.map((objective) => (
              <li key={objective} className="flex gap-3">
                <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-flame-500" />
                {objective}
              </li>
            ))}
          </ul>
        </section>
      )}

      <article className="prose-chapter max-w-3xl">
        <Markdown>{chapter.body}</Markdown>
      </article>

      {chapter.lab_detail && (
        <section className="mt-12">
          <div className="mb-4 flex items-center gap-3">
            <span className="h-px w-8 bg-flame-500" />
            <span className="text-[10px] font-bold uppercase tracking-[0.18em] text-flame-400">
              Hands-on lab
            </span>
          </div>
          <LabPane
            lab={chapter.lab_detail}
            onPassed={() => onProgress(slug, "done")}
          />
        </section>
      )}

      <section className="panel-glow mt-12 rounded-2xl border border-ink-800 bg-ink-900/70 p-5 sm:p-6">
        <label
          htmlFor="note"
          className="mb-2 flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.18em] text-ink-500"
        >
          <span className="h-1.5 w-1.5 rounded-full bg-ink-600" />
          Chapter notes
        </label>
        <textarea
          id="note"
          value={note}
          onChange={(event) => editNote(event.target.value)}
          rows={4}
          placeholder="Capture the idea you want to revisit."
          className="mt-3 w-full resize-y rounded-xl border border-ink-700 bg-ink-950/45 px-3.5 py-3 text-sm leading-6 text-ink-200 outline-none transition placeholder:text-ink-600 focus:border-flame-500 focus:bg-ink-950"
        />
        <p className="mt-2 text-xs text-ink-600">Notes save automatically after you stop typing.</p>
      </section>

      <div className="mt-8 flex flex-wrap items-center justify-between gap-4 border-t border-ink-800 pt-6">
        <button
          onClick={() => onProgress(slug, done ? "in_progress" : "done")}
          className={`rounded-xl px-4 py-2.5 text-xs font-bold transition ${
            done
              ? "border border-mint-400/35 bg-mint-400/10 text-mint-400 hover:bg-mint-400/15"
              : "border border-ink-700 bg-ink-900 text-ink-300 hover:border-flame-500/60 hover:text-flame-300"
          }`}
        >
          {done ? "✓ Completed" : "Mark chapter complete"}
        </button>

        <div className="flex gap-2.5">
          {chapter.prev && (
            <Link
              to={`/c/${chapter.prev}`}
              className="rounded-xl border border-ink-700 bg-ink-900 px-4 py-2.5 text-xs font-semibold text-ink-300 transition hover:border-ink-600 hover:text-ink-100"
            >
              ← Previous
            </Link>
          )}
          {chapter.next && (
            <Link
              to={`/c/${chapter.next}`}
              className="rounded-xl bg-flame-500 px-4 py-2.5 text-xs font-bold text-ink-950 transition hover:bg-flame-400"
            >
              Next chapter →
            </Link>
          )}
        </div>
      </div>
    </div>
  );
}
