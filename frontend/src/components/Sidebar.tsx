import { NavLink } from "react-router-dom";
import type { ChapterMeta } from "../lib/api";

type Props = {
  chapters: ChapterMeta[];
  progress: Record<string, string>;
  open: boolean;
  onNavigate: () => void;
};

export default function Sidebar({ chapters, progress, open, onNavigate }: Props) {
  const parts = chapters.reduce<Record<string, ChapterMeta[]>>((groups, chapter) => {
    (groups[chapter.part] ||= []).push(chapter);
    return groups;
  }, {});

  const done = chapters.filter((c) => progress[c.slug] === "done").length;
  const completion = chapters.length ? Math.round((done / chapters.length) * 100) : 0;

  return (
    <aside
      className={`${
        open ? "translate-x-0" : "-translate-x-full"
      } fixed inset-y-0 left-0 z-30 flex w-78 shrink-0 flex-col border-r border-ink-800/80 bg-ink-900/95 shadow-2xl shadow-ink-950/30 backdrop-blur transition-transform lg:static lg:translate-x-0 lg:shadow-none`}
    >
      <div className="border-b border-ink-800 px-5 py-5">
        <NavLink to="/" onClick={onNavigate} className="group flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-flame-500 text-base font-black tracking-tighter text-ink-950 shadow-lg shadow-flame-500/15 transition group-hover:scale-105">
            λ
          </div>
          <div>
            <div className="text-[11px] font-bold uppercase tracking-[0.2em] text-flame-400">
              Learn inference
            </div>
            <div className="mt-0.5 text-xs text-ink-500">Engineering course</div>
          </div>
        </NavLink>

        <div className="mt-6">
          <div className="mb-2 flex items-center justify-between text-[11px] font-semibold uppercase tracking-wider text-ink-500">
            <span>Course progress</span>
            <span className="text-flame-400">{completion}%</span>
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-ink-800">
            <div
              className="h-full rounded-full bg-flame-500 transition-all duration-500"
              style={{ width: `${completion}%` }}
            />
          </div>
          <div className="mt-2 text-xs text-ink-400">
            <span className="font-semibold text-ink-200">{done}</span> of {chapters.length} chapters complete
          </div>
        </div>
      </div>

      <nav className="min-h-0 flex-1 overflow-y-auto px-3 py-5">
        <NavLink
          to="/"
          end
          onClick={onNavigate}
          className={({ isActive }) =>
            `mb-5 flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition ${
              isActive
                ? "bg-flame-500/12 text-flame-300"
                : "text-ink-300 hover:bg-ink-850 hover:text-ink-100"
            }`
          }
        >
          <span className="flex h-5 w-5 items-center justify-center rounded-md border border-current/30 text-[11px]">⌂</span>
          Course overview
        </NavLink>

        <NavLink
          to="/compute"
          onClick={onNavigate}
          className={({ isActive }) =>
            `mb-5 flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition ${
              isActive
                ? "bg-flame-500/12 text-flame-300"
                : "text-ink-300 hover:bg-ink-850 hover:text-ink-100"
            }`
          }
        >
          <span className="flex h-5 w-5 items-center justify-center rounded-md border border-current/30 text-[11px]">
            ◎
          </span>
          Compute
        </NavLink>

        {Object.entries(parts).map(([part, items]) => (
          <div key={part} className="mb-6">
            <div className="mb-2 flex items-center gap-2 px-3 text-[10px] font-bold uppercase tracking-[0.16em] text-ink-600">
              <span className="h-px flex-1 bg-ink-800" />
              {part}
              <span className="h-px w-3 bg-ink-800" />
            </div>
            {items.map((chapter) => (
              <NavLink
                key={chapter.slug}
                to={`/c/${chapter.slug}`}
                onClick={onNavigate}
                className={({ isActive }) =>
                  `group flex items-start gap-3 rounded-xl px-3 py-2.5 text-sm transition ${
                    isActive
                      ? "bg-ink-850 text-ink-100 shadow-sm shadow-ink-950/20"
                      : "text-ink-400 hover:bg-ink-850/70 hover:text-ink-100"
                  }`
                }
              >
                <span
                  className={`mt-1.5 flex h-2.5 w-2.5 shrink-0 rounded-full border ${
                    progress[chapter.slug] === "done"
                      ? "border-mint-400 bg-mint-400"
                      : progress[chapter.slug]
                        ? "border-flame-500 bg-flame-500"
                        : "border-ink-600 bg-transparent"
                  }`}
                  aria-hidden
                />
                <span className="leading-snug">{chapter.title}</span>
              </NavLink>
            ))}
          </div>
        ))}
      </nav>
    </aside>
  );
}
