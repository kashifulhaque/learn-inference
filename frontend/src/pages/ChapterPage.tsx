import { memo, useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
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

type Section = { index: number; id: string; title: string };

// Scrolling updates this page many times a second. The chapter body is the
// expensive part to render and never changes while you read it.
const ChapterBody = memo(Markdown);

const FOCUS_KEY = "li.reader.focus";

function readFocus(): boolean {
  try {
    return window.localStorage.getItem(FOCUS_KEY) === "1";
  } catch {
    return false;
  }
}

function pad(value: number): string {
  return String(value).padStart(2, "0");
}

/** A heading's visible text, without the MathML copy KaTeX adds for screen readers. */
function headingText(heading: HTMLElement): string {
  const clone = heading.cloneNode(true) as HTMLElement;
  clone.querySelectorAll(".katex-mathml").forEach((node) => node.remove());
  return (clone.textContent ?? "").replace(/\s+/g, " ").trim();
}

export default function ChapterPage({ progress, onProgress }: Props) {
  const { slug = "" } = useParams();
  const [chapter, setChapter] = useState<Chapter | null>(null);
  const [note, setNote] = useState("");
  const [error, setError] = useState("");
  const [view, setView] = useState<"read" | "lab">("read");
  const [sections, setSections] = useState<Section[]>([]);
  const [active, setActive] = useState(0);
  const [furthest, setFurthest] = useState(0);
  const [minutesLeft, setMinutesLeft] = useState<number | null>(null);
  const [focus, setFocus] = useState(readFocus);
  const [mapOpen, setMapOpen] = useState(false);
  const noteTimer = useRef<number | null>(null);
  const reading = useRef<HTMLDivElement>(null);
  const article = useRef<HTMLElement>(null);
  const bar = useRef<HTMLDivElement>(null);
  const frame = useRef<number | null>(null);
  const tracker = useRef<HTMLDivElement>(null);
  const wide = useIsWide();

  useEffect(() => {
    setChapter(null);
    setError("");
    setView("read");
    setSections([]);
    setActive(0);
    setFurthest(0);
    setMapOpen(false);
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

  // The body renders synchronously, so its sections exist by the time this runs.
  useLayoutEffect(() => {
    const root = article.current;
    if (!chapter || !root) return;
    const found: Section[] = [];
    root.querySelectorAll<HTMLElement>("section[data-section]").forEach((section) => {
      const heading = section.querySelector<HTMLElement>(":scope > h2");
      if (!heading) return;
      found.push({
        index: Number(section.dataset.section),
        id: heading.id,
        title: headingText(heading),
      });
    });
    setSections(found);
  }, [chapter]);

  const track = useCallback(() => {
    frame.current = null;
    const box = reading.current;
    const root = article.current;
    if (!box || !root) return;

    const max = box.scrollHeight - box.clientHeight;
    const done = max > 0 ? Math.min(1, box.scrollTop / max) : 1;
    // The bar moves on every frame, so it is styled directly rather than
    // through state.
    if (bar.current) bar.current.style.transform = `scaleX(${done})`;
    if (chapter?.minutes) setMinutesLeft(Math.ceil(chapter.minutes * (1 - done)));

    // The section you are reading is the last one whose heading has passed a
    // line 30% of the way down the pane.
    const line = box.getBoundingClientRect().top + box.clientHeight * 0.3;
    let current = 0;
    root.querySelectorAll<HTMLElement>("section[data-section]").forEach((section) => {
      if (section.getBoundingClientRect().top <= line) current = Number(section.dataset.section);
    });
    // A short last section never reaches the line, so the end of the page
    // counts as reaching it.
    if (max > 0 && box.scrollTop >= max - 4) {
      const all = root.querySelectorAll<HTMLElement>("section[data-section]");
      current = Number(all[all.length - 1]?.dataset.section ?? current);
    }
    setActive(current);
    setFurthest((previous) => Math.max(previous, current));
  }, [chapter]);

  const onScroll = useCallback(() => {
    if (frame.current === null) frame.current = window.requestAnimationFrame(track);
  }, [track]);

  useEffect(() => {
    track();
    return () => {
      if (frame.current !== null) window.cancelAnimationFrame(frame.current);
    };
  }, [track, sections]);

  // Focus mode dims every section except the one you are reading. The
  // sections belong to the memoized body, so their classes are set here
  // rather than re-rendering it.
  useEffect(() => {
    const root = article.current;
    if (!root) return;
    root.classList.toggle("focus-mode", focus);
    root.querySelectorAll<HTMLElement>("section[data-section]").forEach((section) => {
      section.classList.toggle("is-active", Number(section.dataset.section) === active);
    });
  }, [active, focus, sections]);

  useEffect(() => {
    try {
      window.localStorage.setItem(FOCUS_KEY, focus ? "1" : "0");
    } catch {
      // Focus mode still works, it just forgets between visits.
    }
  }, [focus]);

  const goTo = useCallback((id: string) => {
    const box = reading.current;
    const target = document.getElementById(id);
    if (!box || !target) return;
    const top =
      target.getBoundingClientRect().top - box.getBoundingClientRect().top + box.scrollTop - 16;
    box.scrollTo({ top, behavior: "smooth" });
    setMapOpen(false);
  }, []);

  // The section list closes on a click anywhere else, or on Escape.
  useEffect(() => {
    if (!mapOpen) return;
    const onPointer = (event: PointerEvent) => {
      if (!tracker.current?.contains(event.target as Node)) setMapOpen(false);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMapOpen(false);
    };
    window.addEventListener("pointerdown", onPointer);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("pointerdown", onPointer);
      window.removeEventListener("keydown", onKey);
    };
  }, [mapOpen]);

  // `[` and `]` step between sections, unless you are typing somewhere.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (event.key !== "[" && event.key !== "]") return;
      const target = event.target as HTMLElement | null;
      if (target?.closest("input, textarea, select, [contenteditable], .monaco-editor")) return;
      const position = sections.findIndex((section) => section.index === active);
      const next = event.key === "]" ? position + 1 : position - 1;
      if (next >= 0 && next < sections.length) {
        event.preventDefault();
        goTo(sections[next].id);
      } else if (next < 0) {
        event.preventDefault();
        reading.current?.scrollTo({ top: 0, behavior: "smooth" });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [sections, active, goTo]);

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
  const position = sections.findIndex((section) => section.index === active);
  const current = position >= 0 ? sections[position] : null;

  const sectionList = (compact: boolean) => (
    <ol className={compact ? "space-y-0.5" : "grid gap-1 sm:grid-cols-2"}>
      {sections.map((section) => {
        const isCurrent = section.index === active;
        const seen = section.index <= furthest && !isCurrent;
        return (
          <li key={section.id}>
            <button
              type="button"
              onClick={() => goTo(section.id)}
              aria-current={isCurrent ? "location" : undefined}
              className={`group flex w-full items-start gap-2.5 rounded-lg px-2.5 py-1.5 text-left text-[12.5px] leading-5 transition ${
                isCurrent
                  ? "bg-flame-500/12 text-flame-300"
                  : "text-ink-300 hover:bg-ink-800/70 hover:text-ink-100"
              }`}
            >
              <span
                className={`mt-px flex h-5 w-5 shrink-0 items-center justify-center rounded-md font-mono text-[10px] font-bold ${
                  isCurrent
                    ? "bg-flame-500 text-ink-950"
                    : seen
                      ? "bg-mint-400/15 text-mint-400"
                      : "bg-ink-800 text-ink-500 group-hover:text-ink-300"
                }`}
              >
                {seen ? "✓" : pad(section.index)}
              </span>
              <span className="min-w-0">{section.title}</span>
            </button>
          </li>
        );
      })}
    </ol>
  );

  const readingPane = (
    <div className="flex h-full min-h-0 flex-col bg-ink-950/20">
      <header className="flex h-14 shrink-0 items-center gap-2 border-b border-ink-800 bg-ink-900/70 px-3 sm:gap-2.5 sm:px-4">
        <span className="min-w-0 truncate rounded-lg border border-ink-800 bg-ink-900 px-2 py-1 text-[11px] font-medium text-ink-300">
          {chapter.part}
        </span>
        {minutesLeft !== null && (
          <span className="hidden whitespace-nowrap text-[11px] text-ink-500 sm:inline">
            {minutesLeft > 0 ? `About ${minutesLeft} min left` : "Finished reading"}
          </span>
        )}
        <span className="ml-auto flex shrink-0 items-center gap-1.5 sm:gap-2">
          <button
            onClick={() => onProgress(slug, done ? "in_progress" : "done")}
            aria-label={done ? "Mark this chapter unread" : "Mark this chapter done"}
            className={`flex h-9 items-center gap-1.5 whitespace-nowrap rounded-lg border px-2.5 text-[11px] font-semibold transition ${
              done
                ? "border-mint-400/35 bg-mint-400/10 text-mint-400 hover:bg-mint-400/20"
                : "border-ink-700 bg-ink-900 text-ink-300 hover:border-flame-500/60 hover:text-flame-300"
            }`}
          >
            {/* "Mark done" breaks over two lines inside its own pill on a
                narrow phone, so there the mark stands in for the words. */}
            <span className={done ? "" : "min-[400px]:hidden"}>{done ? "✓" : "○"}</span>
            <span className="hidden min-[400px]:inline">
              {done ? "Done" : "Mark done"}
            </span>
          </button>
          <span className="flex items-center gap-1">
            {chapter.prev ? (
              <Link
                to={`/c/${chapter.prev}`}
                title="Previous chapter"
                aria-label="Previous chapter"
                className="flex h-9 w-9 items-center justify-center rounded-lg border border-ink-800 bg-ink-900 text-ink-400 transition hover:border-ink-600 hover:text-ink-100 lg:h-7 lg:w-7"
              >
                ←
              </Link>
            ) : (
              <span className="flex h-9 w-9 items-center justify-center rounded-lg border border-ink-800/60 text-ink-700 lg:h-7 lg:w-7">
                ←
              </span>
            )}
            {chapter.next ? (
              <Link
                to={`/c/${chapter.next}`}
                title="Next chapter"
                aria-label="Next chapter"
                className="flex h-9 w-9 items-center justify-center rounded-lg border border-ink-800 bg-ink-900 text-ink-400 transition hover:border-ink-600 hover:text-ink-100 lg:h-7 lg:w-7"
              >
                →
              </Link>
            ) : (
              <span className="flex h-9 w-9 items-center justify-center rounded-lg border border-ink-800/60 text-ink-700 lg:h-7 lg:w-7">
                →
              </span>
            )}
          </span>
        </span>
      </header>

      {/* The section tracker: where you are, how far you have come, and a
          way to jump anywhere without scrolling back to the top. */}
      <div ref={tracker} className="relative shrink-0 border-b border-ink-800 bg-ink-900/40">
        <div className="flex h-10 items-center gap-2 px-3 sm:px-4">
          <button
            type="button"
            onClick={() => setMapOpen((open) => !open)}
            aria-expanded={mapOpen}
            aria-label="Show the sections of this chapter"
            className="flex min-w-0 flex-1 items-center gap-2 rounded-lg px-1.5 py-1 text-left transition hover:bg-ink-800/60"
          >
            <span className="shrink-0 rounded-md bg-ink-800 px-1.5 py-0.5 font-mono text-[10px] font-bold text-flame-400">
              {current ? `${pad(current.index)}/${pad(sections.length)}` : "Intro"}
            </span>
            <span className="min-w-0 truncate text-[12px] font-medium text-ink-200">
              {current ? current.title : chapter.title}
            </span>
            <span className={`shrink-0 text-[10px] text-ink-500 transition ${mapOpen ? "rotate-180" : ""}`}>
              ▾
            </span>
          </button>
          <button
            type="button"
            onClick={() => setFocus((on) => !on)}
            aria-pressed={focus}
            title="Dim every section except the one you are reading"
            className={`flex h-7 shrink-0 items-center gap-1.5 rounded-lg border px-2 text-[11px] font-semibold transition ${
              focus
                ? "border-flame-500/50 bg-flame-500/12 text-flame-300"
                : "border-ink-800 text-ink-400 hover:border-ink-600 hover:text-ink-100"
            }`}
          >
            <span className={`h-1.5 w-1.5 rounded-full ${focus ? "bg-flame-400" : "bg-ink-600"}`} />
            Focus
          </button>
        </div>
        <div className="h-[3px] w-full overflow-hidden bg-ink-800/60">
          <div
            ref={bar}
            className="h-full origin-left bg-gradient-to-r from-flame-500 via-mint-400 to-amber-400 transition-transform duration-150"
            style={{ transform: "scaleX(0)" }}
          />
        </div>
        {mapOpen && sections.length > 0 && (
          <div className="absolute inset-x-2 top-full z-20 mt-1 max-h-[60vh] overflow-y-auto rounded-xl border border-ink-700 bg-ink-900 p-2 shadow-2xl shadow-black/40 sm:inset-x-4">
            {sectionList(true)}
            <p className="mt-2 border-t border-ink-800 px-2.5 pt-2 text-[10.5px] text-ink-500">
              Press <kbd className="rounded bg-ink-800 px-1 font-mono">[</kbd> and{" "}
              <kbd className="rounded bg-ink-800 px-1 font-mono">]</kbd> to step between sections.
            </p>
          </div>
        )}
      </div>

      <div ref={reading} onScroll={onScroll} className="min-h-0 flex-1 overflow-y-auto">
        <div className={`px-4 py-6 sm:px-8 sm:py-8 ${hasLab ? "max-w-3xl" : "mx-auto max-w-3xl"}`}>
          <div className="chapter-hero mb-7">
            <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-flame-400">
              {chapter.part}
            </p>
            <h1 className="mt-2 text-[1.9rem] font-bold leading-tight tracking-[-0.02em] text-ink-100 sm:text-[2.2rem]">
              {chapter.title}
            </h1>
            {chapter.summary && (
              <p className="mt-3 text-[15px] leading-7 text-ink-300">{chapter.summary}</p>
            )}
            <div className="mt-4 flex flex-wrap gap-2 text-[11px] font-medium text-ink-400">
              {chapter.minutes && (
                <span className="rounded-full border border-ink-800 bg-ink-900/70 px-2.5 py-1">
                  {chapter.minutes} min
                </span>
              )}
              {sections.length > 0 && (
                <span className="rounded-full border border-ink-800 bg-ink-900/70 px-2.5 py-1">
                  {sections.length} sections
                </span>
              )}
              {hasLab && (
                <span className="rounded-full border border-flame-500/30 bg-flame-500/10 px-2.5 py-1 text-flame-300">
                  Hands-on lab
                </span>
              )}
            </div>
          </div>

          {(chapter.objectives.length > 0 || sections.length > 0) && (
            <section className="mb-9 overflow-hidden rounded-2xl border border-ink-800 bg-ink-900/60">
              {chapter.objectives.length > 0 && (
                <div className="border-b border-ink-800 px-4 py-4">
                  <div className="text-[10px] font-bold uppercase tracking-[0.16em] text-ink-400">
                    By the end, you can
                  </div>
                  <ul className="mt-3 space-y-2 text-[13px] leading-6 text-ink-300">
                    {chapter.objectives.map((objective) => (
                      <li key={objective} className="flex gap-2.5">
                        <span className="mt-[9px] h-1.5 w-1.5 shrink-0 rounded-full bg-flame-500" />
                        {objective}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {sections.length > 0 && (
                <div className="px-2 py-3">
                  <div className="px-2.5 pb-2 text-[10px] font-bold uppercase tracking-[0.16em] text-ink-400">
                    Your route through this chapter
                  </div>
                  {sectionList(false)}
                </div>
              )}
            </section>
          )}

          <article ref={article} className="chapter-article">
            <ChapterBody>{chapter.body}</ChapterBody>
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
            className={`flex-1 rounded-lg px-3 py-2.5 text-xs font-semibold transition ${
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
