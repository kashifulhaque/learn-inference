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

  return (
    <aside
      className={`${
        open ? "translate-x-0" : "-translate-x-full"
      } fixed inset-y-0 left-0 z-30 w-72 shrink-0 overflow-y-auto border-r border-ink-800 bg-ink-900 transition-transform lg:static lg:translate-x-0`}
    >
      <div className="sticky top-0 z-10 border-b border-ink-800 bg-ink-900 px-5 py-4">
        <NavLink to="/" onClick={onNavigate} className="block">
          <div className="text-xs font-medium uppercase tracking-[0.18em] text-flame-500">
            learn-inference
          </div>
          <div className="mt-0.5 text-sm text-ink-400">
            {done} of {chapters.length} chapters done
          </div>
        </NavLink>
        <div className="mt-3 h-1 overflow-hidden rounded-full bg-ink-800">
          <div
            className="h-full rounded-full bg-flame-500 transition-all"
            style={{ width: `${chapters.length ? (done / chapters.length) * 100 : 0}%` }}
          />
        </div>
      </div>

      <nav className="px-3 py-4">
        {Object.entries(parts).map(([part, items]) => (
          <div key={part} className="mb-5">
            <div className="mb-1.5 px-2 text-[11px] font-semibold uppercase tracking-wider text-ink-600">
              {part}
            </div>
            {items.map((chapter) => (
              <NavLink
                key={chapter.slug}
                to={`/c/${chapter.slug}`}
                onClick={onNavigate}
                className={({ isActive }) =>
                  `group flex items-start gap-2.5 rounded-lg px-2 py-1.5 text-sm transition ${
                    isActive
                      ? "bg-ink-800 text-ink-100"
                      : "text-ink-300 hover:bg-ink-850 hover:text-ink-100"
                  }`
                }
              >
                <span
                  className={`mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full ${
                    progress[chapter.slug] === "done"
                      ? "bg-mint-400"
                      : progress[chapter.slug]
                        ? "bg-flame-500"
                        : "bg-ink-700"
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
