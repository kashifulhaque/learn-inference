import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, type Chapter } from "../lib/api";
import Markdown from "../components/Markdown";
import LabPane from "../components/LabPane";
import SplitPane from "../components/SplitPane";
import { useIsWide } from "../lib/useMediaQuery";

type Props = {
  progress: Record<string, string>;
  onProgress: (slug: string, status: "in_progress" | "done") => void;
};

export default function ChapterPage({ progress, onProgress }: Props) {
  const { slug = "" } = useParams();
  const [chapter, setChapter] = useState<Chapter | null>(null);
  const [note, setNote] = useState("");
  const [error, setError] = useState("");
  const [view, setView] = useState<"read" | "lab">("read");
  const noteTimer = useRef<number | null>(null);
  const reading = useRef<HTMLDivElement>(null);
  const wide = useIsWide();

  useEffect(() => {
    setChapter(null);
    setError("");
    setView("read");
    api
      .chapter(slug)
      .then((result) => {
        setChapter(result);
        setNote(result.note);
        if (!progress[slug]) onProgress(slug, "in_progress");
      })
      .catch((e) => setError(String(e)));
    reading.current?.scrollTo({ top: 0 });
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
  const hasLab = Boolean(chapter.lab_detail);

  const readingPane = (
    <div className="flex h-full min-h-0 flex-col bg-ink-950/20">
      <header className="flex h-14 shrink-0 items-center gap-2.5 border-b border-ink-800 bg-ink-900/70 px-4">
        <span className="truncate rounded-lg border border-ink-800 bg-ink-900 px-2 py-1 text-[11px] font-medium text-ink-300">
          {chapter.part}
        </span>
        {chapter.minutes && (
          <span className="hidden whitespace-nowrap text-[11px] text-ink-500 sm:inline">
            {chapter.minutes} min read
          </span>
        )}
        <span className="ml-auto flex items-center gap-2">
          <button
            onClick={() => onProgress(slug, done ? "in_progress" : "done")}
            className={`rounded-lg border px-2.5 py-1.5 text-[11px] font-semibold transition ${
              done
                ? "border-mint-400/35 bg-mint-400/10 text-mint-400 hover:bg-mint-400/20"
                : "border-ink-700 bg-ink-900 text-ink-300 hover:border-flame-500/60 hover:text-flame-300"
            }`}
          >
            {done ? "✓ Done" : "Mark done"}
          </button>
          <span className="flex items-center gap-1">
            {chapter.prev ? (
              <Link
                to={`/c/${chapter.prev}`}
                title="Previous chapter"
                className="flex h-7 w-7 items-center justify-center rounded-lg border border-ink-800 bg-ink-900 text-ink-400 transition hover:border-ink-600 hover:text-ink-100"
              >
                ←
              </Link>
            ) : (
              <span className="flex h-7 w-7 items-center justify-center rounded-lg border border-ink-800/60 text-ink-700">
                ←
              </span>
            )}
            {chapter.next ? (
              <Link
                to={`/c/${chapter.next}`}
                title="Next chapter"
                className="flex h-7 w-7 items-center justify-center rounded-lg border border-ink-800 bg-ink-900 text-ink-400 transition hover:border-ink-600 hover:text-ink-100"
              >
                →
              </Link>
            ) : (
              <span className="flex h-7 w-7 items-center justify-center rounded-lg border border-ink-800/60 text-ink-700">
                →
              </span>
            )}
          </span>
        </span>
      </header>

      <div ref={reading} className="min-h-0 flex-1 overflow-y-auto">
        <div className={`px-5 py-7 sm:px-8 ${hasLab ? "max-w-3xl" : "mx-auto max-w-3xl"}`}>
          {chapter.objectives.length > 0 && (
            <section className="mb-8 overflow-hidden rounded-xl border border-ink-800 bg-ink-900/60">
              <div className="border-b border-ink-800 bg-ink-850/50 px-4 py-2 text-[10px] font-bold uppercase tracking-[0.16em] text-ink-400">
                What you will be able to do
              </div>
              <ul className="space-y-2 px-4 py-3.5 text-[13px] leading-6 text-ink-300">
                {chapter.objectives.map((objective) => (
                  <li key={objective} className="flex gap-2.5">
                    <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-flame-500" />
                    {objective}
                  </li>
                ))}
              </ul>
            </section>
          )}

          <article className="prose-chapter">
            <Markdown>{chapter.body}</Markdown>
          </article>

          <section className="mt-10 rounded-xl border border-ink-800 bg-ink-900/60 p-4">
            <label
              htmlFor="note"
              className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.16em] text-ink-500"
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
              className="mt-3 w-full resize-y rounded-lg border border-ink-800 bg-ink-950/50 px-3 py-2.5 text-[13px] leading-6 text-ink-200 outline-none transition placeholder:text-ink-600 focus:border-flame-500 focus:bg-ink-950"
            />
            <p className="mt-2 text-[11px] text-ink-600">
              Notes save automatically after you stop typing.
            </p>
          </section>

          <div className="mt-8 flex flex-wrap items-center justify-between gap-3 border-t border-ink-800 pt-5">
            <button
              onClick={() => onProgress(slug, done ? "in_progress" : "done")}
              className={`rounded-lg px-3.5 py-2 text-xs font-bold transition ${
                done
                  ? "border border-mint-400/35 bg-mint-400/10 text-mint-400 hover:bg-mint-400/20"
                  : "border border-ink-700 bg-ink-900 text-ink-300 hover:border-flame-500/60 hover:text-flame-300"
              }`}
            >
              {done ? "✓ Chapter complete" : "Mark chapter complete"}
            </button>
            {chapter.next && (
              <Link
                to={`/c/${chapter.next}`}
                className="rounded-lg bg-flame-500 px-3.5 py-2 text-xs font-bold text-ink-950 transition hover:bg-flame-400"
              >
                Next chapter →
              </Link>
            )}
          </div>
        </div>
      </div>
    </div>
  );

  if (!hasLab) {
    return <div className="h-full min-h-0">{readingPane}</div>;
  }

  const labPane = (
    <LabPane lab={chapter.lab_detail!} onPassed={() => onProgress(slug, "done")} />
  );

  // Wide screens read on the left and build on the right. Narrow ones get the
  // same two panes as a pair of tabs, because a 380px column cannot hold both.
  if (wide) {
    return (
      <SplitPane
        direction="row"
        storageKey="li.chapter.split"
        initial={50}
        min={25}
        max={75}
        className="h-full"
        label="Resize the chapter and the lab"
        first={readingPane}
        second={labPane}
      />
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex shrink-0 gap-1 border-b border-ink-800 bg-ink-900 p-1.5">
        {(["read", "lab"] as const).map((option) => (
          <button
            key={option}
            onClick={() => setView(option)}
            className={`flex-1 rounded-lg px-3 py-1.5 text-xs font-semibold transition ${
              view === option
                ? "bg-ink-850 text-ink-100"
                : "text-ink-500 hover:text-ink-200"
            }`}
          >
            {option === "read" ? "Chapter" : "Lab"}
          </button>
        ))}
      </div>
      {/* Both panes stay mounted: switching tabs must not throw away a draft
          or a run in flight. */}
      <div className="min-h-0 flex-1">
        <div className={`h-full ${view === "read" ? "" : "hidden"}`}>{readingPane}</div>
        <div className={`h-full ${view === "lab" ? "" : "hidden"}`}>{labPane}</div>
      </div>
    </div>
  );
}
