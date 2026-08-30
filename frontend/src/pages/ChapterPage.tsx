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
    <div ref={top} className="mx-auto max-w-4xl px-6 py-10 lg:px-10">
      <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
        <span className="rounded-md bg-ink-800 px-2 py-1 text-ink-400">
          {chapter.part}
        </span>
        {chapter.minutes && (
          <span className="text-ink-600">{chapter.minutes} min</span>
        )}
        {chapter.gpu && (
          <span className="rounded-md border border-flame-500/40 px-2 py-0.5 text-flame-400">
            GPU
          </span>
        )}
      </div>

      {chapter.objectives.length > 0 && (
        <div className="mb-8 rounded-xl border border-ink-800 bg-ink-900 px-5 py-4">
          <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-ink-500">
            By the end you can
          </div>
          <ul className="space-y-1 text-sm text-ink-300">
            {chapter.objectives.map((objective) => (
              <li key={objective} className="flex gap-2">
                <span className="text-flame-500">→</span>
                {objective}
              </li>
            ))}
          </ul>
        </div>
      )}

      <Markdown>{chapter.body}</Markdown>

      {chapter.lab_detail && (
        <div className="mt-10">
          <LabPane
            lab={chapter.lab_detail}
            onPassed={() => onProgress(slug, "done")}
          />
        </div>
      )}

      <div className="mt-10 rounded-xl border border-ink-800 bg-ink-900 p-5">
        <label
          htmlFor="note"
          className="mb-2 block text-[11px] font-semibold uppercase tracking-wider text-ink-500"
        >
          Your notes on this chapter
        </label>
        <textarea
          id="note"
          value={note}
          onChange={(event) => editNote(event.target.value)}
          rows={4}
          placeholder="Saved automatically."
          className="w-full resize-y rounded-lg border border-ink-700 bg-ink-850 px-3 py-2.5 text-sm text-ink-200 outline-none transition placeholder:text-ink-600 focus:border-flame-500"
        />
      </div>

      <div className="mt-8 flex flex-wrap items-center justify-between gap-4 border-t border-ink-800 pt-6">
        <button
          onClick={() => onProgress(slug, done ? "in_progress" : "done")}
          className={`rounded-lg px-4 py-2 text-xs font-semibold transition ${
            done
              ? "border border-mint-400/40 bg-mint-400/10 text-mint-400"
              : "border border-ink-700 text-ink-300 hover:border-ink-600 hover:text-ink-100"
          }`}
        >
          {done ? "✓ Marked done" : "Mark as done"}
        </button>

        <div className="flex gap-3">
          {chapter.prev && (
            <Link
              to={`/c/${chapter.prev}`}
              className="rounded-lg border border-ink-700 px-4 py-2 text-xs text-ink-300 transition hover:border-ink-600 hover:text-ink-100"
            >
              ← Previous
            </Link>
          )}
          {chapter.next && (
            <Link
              to={`/c/${chapter.next}`}
              className="rounded-lg bg-ink-800 px-4 py-2 text-xs font-medium text-ink-100 transition hover:bg-ink-700"
            >
              Next →
            </Link>
          )}
        </div>
      </div>
    </div>
  );
}
