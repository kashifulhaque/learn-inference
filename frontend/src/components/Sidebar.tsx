import { useEffect, useMemo, useState } from "react";
import { NavLink } from "react-router-dom";
import type { ChapterMeta } from "../lib/api";

type Props = {
  chapters: ChapterMeta[];
  progress: Record<string, string>;
  /** Drawer state on a narrow screen. */
  open: boolean;
  /** Rail state on a wide screen. */
  collapsed: boolean;
  onNavigate: () => void;
  onToggleCollapsed: () => void;
};

const SHUT_PARTS_KEY = "li.sidebar.shutParts";

function readShutParts(): string[] {
  try {
    const raw = window.localStorage.getItem(SHUT_PARTS_KEY);
    const value = raw ? JSON.parse(raw) : [];
    return Array.isArray(value) ? value.filter((p) => typeof p === "string") : [];
  } catch {
    return [];
  }
}

function Chevron({ open }: { open: boolean }) {
  return (
    <svg
      viewBox="0 0 12 12"
      className={`h-3 w-3 shrink-0 transition-transform duration-200 ${
        open ? "" : "-rotate-90"
      }`}
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M2.5 4.5 6 8l3.5-3.5" />
    </svg>
  );
}

function HomeIcon() {
  return (
    <svg viewBox="0 0 16 16" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" aria-hidden>
      <path d="M2.5 6.8 8 2.5l5.5 4.3v6a.7.7 0 0 1-.7.7H3.2a.7.7 0 0 1-.7-.7Z" />
      <path d="M6.3 13.5v-4h3.4v4" />
    </svg>
  );
}

function ChipIcon() {
  return (
    <svg viewBox="0 0 16 16" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" aria-hidden>
      <rect x="4.5" y="4.5" width="7" height="7" rx="1.2" />
      <path d="M6.5 2v2.5M9.5 2v2.5M6.5 11.5V14M9.5 11.5V14M2 6.5h2.5M2 9.5h2.5M11.5 6.5H14M11.5 9.5H14" />
    </svg>
  );
}

function PanelIcon({ collapsed }: { collapsed: boolean }) {
  return (
    <svg viewBox="0 0 16 16" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" aria-hidden>
      <rect x="2" y="3" width="12" height="10" rx="1.6" />
      <path d="M6.5 3v10" />
      <path d={collapsed ? "M9 6.4 10.6 8 9 9.6" : "M11.4 6.4 9.8 8l1.6 1.6"} strokeLinecap="round" />
    </svg>
  );
}

export default function Sidebar({
  chapters,
  progress,
  open,
  collapsed,
  onNavigate,
  onToggleCollapsed,
}: Props) {
  const [filter, setFilter] = useState("");
  const [shutParts, setShutParts] = useState<string[]>(readShutParts);

  useEffect(() => {
    try {
      window.localStorage.setItem(SHUT_PARTS_KEY, JSON.stringify(shutParts));
    } catch {
      // Losing the open/shut state is harmless.
    }
  }, [shutParts]);

  const numbers = useMemo(
    () => new Map(chapters.map((chapter, index) => [chapter.slug, index + 1])),
    [chapters],
  );

  const needle = filter.trim().toLowerCase();
  const matches = needle
    ? chapters.filter(
        (chapter) =>
          chapter.title.toLowerCase().includes(needle) ||
          chapter.part.toLowerCase().includes(needle) ||
          chapter.summary.toLowerCase().includes(needle),
      )
    : chapters;

  const parts = matches.reduce<Record<string, ChapterMeta[]>>((groups, chapter) => {
    (groups[chapter.part] ||= []).push(chapter);
    return groups;
  }, {});

  const done = chapters.filter((c) => progress[c.slug] === "done").length;
  const completion = chapters.length ? Math.round((done / chapters.length) * 100) : 0;

  function togglePart(part: string) {
    setShutParts((current) =>
      current.includes(part) ? current.filter((p) => p !== part) : [...current, part],
    );
  }

  function statusRing(slug: string): string {
    if (progress[slug] === "done") return "border-mint-400 bg-mint-400/15 text-mint-400";
    if (progress[slug]) return "border-flame-500 bg-flame-500/12 text-flame-400";
    return "border-ink-700 text-ink-500";
  }

  // The rail: chapter numbers only, enough to keep your place while the screen
  // belongs to the chapter and its lab.
  if (collapsed) {
    return (
      <aside className="hidden w-14 shrink-0 flex-col border-r border-ink-800 bg-ink-900/80 lg:flex">
        <div className="flex h-14 shrink-0 items-center justify-center border-b border-ink-800">
          <button
            onClick={onToggleCollapsed}
            title="Expand the course list"
            aria-label="Expand the course list"
            className="flex h-9 w-9 items-center justify-center rounded-lg text-ink-400 transition hover:bg-ink-850 hover:text-flame-300"
          >
            <PanelIcon collapsed />
          </button>
        </div>

        <div className="flex min-h-0 flex-1 flex-col items-center gap-1 overflow-y-auto py-3">
          <NavLink
            to="/"
            end
            title="Course overview"
            className={({ isActive }) =>
              `flex h-9 w-9 items-center justify-center rounded-lg transition ${
                isActive
                  ? "bg-flame-500/12 text-flame-300"
                  : "text-ink-400 hover:bg-ink-850 hover:text-ink-100"
              }`
            }
          >
            <HomeIcon />
          </NavLink>
          <NavLink
            to="/compute"
            title="Compute"
            className={({ isActive }) =>
              `flex h-9 w-9 items-center justify-center rounded-lg transition ${
                isActive
                  ? "bg-flame-500/12 text-flame-300"
                  : "text-ink-400 hover:bg-ink-850 hover:text-ink-100"
              }`
            }
          >
            <ChipIcon />
          </NavLink>

          <div className="my-2 h-px w-6 bg-ink-800" />

          {chapters.map((chapter) => (
            <NavLink
              key={chapter.slug}
              to={`/c/${chapter.slug}`}
              title={`${numbers.get(chapter.slug)}. ${chapter.title}`}
              className={({ isActive }) =>
                `flex h-8 w-8 items-center justify-center rounded-lg border font-mono text-[11px] transition ${
                  isActive
                    ? "border-flame-400 bg-flame-500/15 text-flame-300"
                    : `${statusRing(chapter.slug)} hover:border-ink-600 hover:text-ink-100`
                }`
              }
            >
              {String(numbers.get(chapter.slug)).padStart(2, "0")}
            </NavLink>
          ))}
        </div>

        <div
          className="shrink-0 border-t border-ink-800 py-3 text-center"
          title={`${done} of ${chapters.length} chapters complete`}
        >
          <div className="text-[10px] font-bold text-flame-400">{completion}%</div>
        </div>
      </aside>
    );
  }

  return (
    <aside
      className={`${
        open ? "translate-x-0" : "-translate-x-full"
      } fixed inset-y-0 left-0 z-30 flex w-72 shrink-0 flex-col border-r border-ink-800 bg-ink-900/95 shadow-2xl shadow-ink-950/40 backdrop-blur transition-transform duration-200 lg:static lg:translate-x-0 lg:shadow-none`}
    >
      <div className="flex h-14 shrink-0 items-center gap-2.5 border-b border-ink-800 px-3">
        <NavLink to="/" onClick={onNavigate} className="group flex min-w-0 items-center gap-2.5">
          <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-flame-500 text-sm font-black text-ink-950 transition group-hover:scale-105">
            λ
          </span>
          <span className="min-w-0">
            <span className="block truncate text-[13px] font-semibold tracking-tight text-ink-100">
              learn-inference
            </span>
            <span className="block truncate text-[10px] uppercase tracking-[0.16em] text-ink-500">
              Engineering course
            </span>
          </span>
        </NavLink>
        <button
          onClick={onToggleCollapsed}
          title="Collapse the course list"
          aria-label="Collapse the course list"
          className="ml-auto hidden h-8 w-8 items-center justify-center rounded-lg text-ink-500 transition hover:bg-ink-850 hover:text-flame-300 lg:flex"
        >
          <PanelIcon collapsed={false} />
        </button>
      </div>

      <div className="shrink-0 border-b border-ink-800 px-3 py-3">
        <div className="mb-1.5 flex items-center justify-between text-[10px] font-semibold uppercase tracking-[0.14em] text-ink-500">
          <span>Progress</span>
          <span className="text-flame-400">
            {done}/{chapters.length}
          </span>
        </div>
        <div className="h-1 overflow-hidden rounded-full bg-ink-800">
          <div
            className="h-full rounded-full bg-flame-500 transition-all duration-500"
            style={{ width: `${completion}%` }}
          />
        </div>
      </div>

      <div className="shrink-0 space-y-1 px-2 py-2">
        <NavLink
          to="/"
          end
          onClick={onNavigate}
          className={({ isActive }) =>
            `flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-[13px] font-medium transition ${
              isActive
                ? "bg-flame-500/12 text-flame-300"
                : "text-ink-300 hover:bg-ink-850 hover:text-ink-100"
            }`
          }
        >
          <HomeIcon />
          Course overview
        </NavLink>
        <NavLink
          to="/compute"
          onClick={onNavigate}
          className={({ isActive }) =>
            `flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-[13px] font-medium transition ${
              isActive
                ? "bg-flame-500/12 text-flame-300"
                : "text-ink-300 hover:bg-ink-850 hover:text-ink-100"
            }`
          }
        >
          <ChipIcon />
          Compute
        </NavLink>
      </div>

      <div className="shrink-0 px-3 pb-3">
        <input
          value={filter}
          onChange={(event) => setFilter(event.target.value)}
          placeholder="Filter chapters"
          aria-label="Filter chapters"
          className="w-full rounded-lg border border-ink-800 bg-ink-950/60 px-2.5 py-1.5 text-xs text-ink-200 outline-none transition placeholder:text-ink-600 focus:border-flame-500"
        />
      </div>

      <nav className="min-h-0 flex-1 overflow-y-auto px-2 pb-5">
        {Object.entries(parts).map(([part, items]) => {
          const shut = shutParts.includes(part) && !needle;
          return (
            <div key={part} className="mb-1">
              <button
                onClick={() => togglePart(part)}
                className="flex w-full items-center gap-1.5 rounded-lg px-2.5 py-2 text-[10px] font-bold uppercase tracking-[0.14em] text-ink-500 transition hover:text-ink-300"
                aria-expanded={!shut}
              >
                <Chevron open={!shut} />
                <span className="truncate">{part}</span>
                <span className="ml-auto font-mono text-[10px] tracking-normal text-ink-600">
                  {items.filter((c) => progress[c.slug] === "done").length}/{items.length}
                </span>
              </button>

              {!shut &&
                items.map((chapter) => (
                  <NavLink
                    key={chapter.slug}
                    to={`/c/${chapter.slug}`}
                    onClick={onNavigate}
                    className={({ isActive }) =>
                      `group flex items-center gap-2.5 rounded-lg py-1.5 pl-3 pr-2.5 text-[13px] transition ${
                        isActive
                          ? "bg-ink-850 text-ink-100"
                          : "text-ink-400 hover:bg-ink-850/60 hover:text-ink-100"
                      }`
                    }
                  >
                    <span
                      className={`flex h-5 w-6 shrink-0 items-center justify-center rounded border font-mono text-[10px] ${statusRing(
                        chapter.slug,
                      )}`}
                      aria-hidden
                    >
                      {String(numbers.get(chapter.slug)).padStart(2, "0")}
                    </span>
                    <span className="min-w-0 flex-1 truncate leading-snug" title={chapter.title}>
                      {chapter.title}
                    </span>
                    {chapter.gpu && (
                      <span
                        className="shrink-0 rounded bg-flame-500/12 px-1 py-0.5 text-[9px] font-bold uppercase tracking-wide text-flame-400"
                        title="This chapter has a GPU lab"
                      >
                        lab
                      </span>
                    )}
                  </NavLink>
                ))}
            </div>
          );
        })}

        {matches.length === 0 && (
          <p className="px-3 py-6 text-xs text-ink-500">No chapter matches “{filter}”.</p>
        )}
      </nav>
    </aside>
  );
}
