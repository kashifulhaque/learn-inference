import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, type ChapterMeta, type Run } from "../lib/api";

type Props = {
  chapters: ChapterMeta[];
  progress: Record<string, string>;
  model: string;
  gpu: string;
};

export default function Dashboard({ chapters, progress, model, gpu }: Props) {
  const [runs, setRuns] = useState<Run[]>([]);

  useEffect(() => {
    api.runs().then((result) => setRuns(result.runs));
  }, []);

  const done = chapters.filter((c) => progress[c.slug] === "done");
  const next = chapters.find((c) => progress[c.slug] !== "done") ?? chapters[0];
  const passed = runs.filter((r) => r.passed).length;
  const completion = chapters.length ? Math.round((done.length / chapters.length) * 100) : 0;
  const attempts = Object.entries(
    runs.reduce<Record<string, { total: number; passed: number }>>((acc, run) => {
      const entry = (acc[run.lab] ||= { total: 0, passed: 0 });
      entry.total += 1;
      if (run.passed) entry.passed += 1;
      return acc;
    }, {}),
  )
    .map(([lab, counts]) => ({ lab: lab.slice(0, 2), ...counts }))
    .sort((a, b) => a.lab.localeCompare(b.lab));

  return (
    <div className="mx-auto max-w-6xl px-5 py-8 sm:px-8 lg:px-12 lg:py-11">
      <section className="surface-grid panel-glow relative overflow-hidden rounded-[2rem] border border-ink-700/70 bg-ink-900/70 px-6 py-8 sm:px-9 sm:py-10">
        <div className="pointer-events-none absolute -right-20 -top-24 h-80 w-80 rounded-full bg-flame-500/10 blur-3xl" />
        <div className="relative max-w-3xl">
          <div className="mb-5 flex items-center gap-3">
            <span className="h-2 w-2 rounded-full bg-flame-500 shadow-sm shadow-flame-500/90" />
            <span className="text-[11px] font-bold uppercase tracking-[0.2em] text-flame-400">
              Course dashboard
            </span>
          </div>
          <h1 className="text-4xl font-semibold tracking-[-0.045em] text-ink-100 sm:text-5xl">
            Build an inference engine.
          </h1>
          <p className="mt-5 max-w-2xl text-sm leading-7 text-ink-400 sm:text-base">
            Trace the path from a safetensors file to a serving engine—CUDA kernels,
            paged cache, continuous batching, and hybrid attention for{" "}
            <span className="rounded bg-ink-850 px-1.5 py-0.5 font-mono text-[0.88em] text-ink-200">
              {model || "your model"}
            </span>
            .
          </p>
        </div>
      </section>

      <section className="relative -mt-1 grid gap-3 border-x border-b border-ink-800/80 bg-ink-900/35 p-4 sm:grid-cols-2 lg:grid-cols-4">
        {[
          { label: "Course complete", value: `${completion}%`, note: `${done.length} chapters` },
          { label: "Labs passed", value: String(passed), note: "verified runs" },
          { label: "Total runs", value: String(runs.length), note: "across all labs" },
          { label: "Compute target", value: gpu || "—", note: "active accelerator" },
        ].map((tile) => (
          <div
            key={tile.label}
            className="rounded-xl border border-ink-800 bg-ink-900/70 px-4 py-4 transition hover:border-ink-700"
          >
            <div className="text-[10px] font-bold uppercase tracking-[0.16em] text-ink-500">
              {tile.label}
            </div>
            <div className="mt-2 truncate text-2xl font-semibold tracking-[-0.03em] text-ink-100">
              {tile.value}
            </div>
            <div className="mt-1 text-xs text-ink-500">{tile.note}</div>
          </div>
        ))}
      </section>

      <div className="mt-9 grid gap-6 lg:grid-cols-[1.25fr_0.75fr]">
        {next && (
          <Link
            to={`/c/${next.slug}`}
            className="group panel-glow relative overflow-hidden rounded-2xl border border-flame-500/35 bg-linear-to-br from-flame-500/15 to-ink-900 px-6 py-6 transition hover:border-flame-400 sm:px-7"
          >
            <div className="absolute right-5 top-5 text-3xl text-flame-400/80 transition-transform group-hover:translate-x-1">
              →
            </div>
            <div className="text-[10px] font-bold uppercase tracking-[0.18em] text-flame-400">
              {progress[next.slug] === "in_progress" ? "Continue building" : "Next chapter"}
            </div>
            <div className="mt-3 max-w-md text-xl font-semibold tracking-[-0.025em] text-ink-100">
              {next.title}
            </div>
            <div className="mt-2 max-w-lg text-sm leading-6 text-ink-400">{next.summary}</div>
            <div className="mt-6 inline-flex items-center gap-2 text-sm font-semibold text-flame-300">
              Open chapter <span aria-hidden>→</span>
            </div>
          </Link>
        )}

        <div className="panel-glow rounded-2xl border border-ink-800 bg-ink-900/70 px-6 py-6">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-[10px] font-bold uppercase tracking-[0.18em] text-ink-500">
                Your trajectory
              </div>
              <div className="mt-2 text-sm font-medium text-ink-200">Keep your momentum.</div>
            </div>
            <div className="flex h-16 w-16 items-center justify-center rounded-full border-4 border-ink-800 bg-ink-850 text-sm font-bold text-flame-400">
              {completion}%
            </div>
          </div>
          <div className="mt-6 h-2 overflow-hidden rounded-full bg-ink-800">
            <div
              className="h-full rounded-full bg-linear-to-r from-flame-500 to-mint-400 transition-all duration-500"
              style={{ width: `${completion}%` }}
            />
          </div>
          <p className="mt-4 text-xs leading-5 text-ink-500">
            Each completed chapter marks an implementation milestone in the serving stack.
          </p>
        </div>
      </div>

      {attempts.length > 0 && (
        <section className="panel-glow mt-9 rounded-2xl border border-ink-800 bg-ink-900/70 p-5 sm:p-6">
          <div className="mb-5 flex items-start justify-between gap-4">
            <div>
              <div className="text-[10px] font-bold uppercase tracking-[0.18em] text-ink-500">
                Lab activity
              </div>
              <h2 className="mt-1 text-lg font-semibold tracking-tight text-ink-100">
                Attempts and passes
              </h2>
            </div>
            <div className="rounded-lg border border-ink-700 bg-ink-850 px-2.5 py-1 text-[11px] text-ink-400">
              Run history
            </div>
          </div>
          <div className="h-60">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={attempts}>
                <CartesianGrid stroke="#27415e" vertical={false} />
                <XAxis dataKey="lab" stroke="#6f88a3" fontSize={11} tickLine={false} axisLine={false} />
                <YAxis stroke="#6f88a3" fontSize={11} tickLine={false} axisLine={false} allowDecimals={false} />
                <Tooltip
                  contentStyle={{
                    background: "#0b1726",
                    border: "1px solid #27415e",
                    borderRadius: 10,
                    fontSize: 12,
                  }}
                  cursor={{ fill: "rgba(101, 230, 176, 0.06)" }}
                />
                <Bar dataKey="total" fill="#46617f" radius={[5, 5, 0, 0]} name="runs" />
                <Bar dataKey="passed" fill="#65e6b0" radius={[5, 5, 0, 0]} name="passed" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      )}

      <section className="mt-10">
        <div className="mb-5 flex items-end justify-between gap-4">
          <div>
            <div className="text-[10px] font-bold uppercase tracking-[0.18em] text-ink-500">
              Curriculum
            </div>
            <h2 className="mt-1 text-xl font-semibold tracking-tight text-ink-100">All chapters</h2>
          </div>
          <span className="text-xs text-ink-500">{chapters.length} modules</span>
        </div>
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {chapters.map((chapter, index) => (
            <Link
              key={chapter.slug}
              to={`/c/${chapter.slug}`}
              className="group panel-glow rounded-2xl border border-ink-800 bg-ink-900/65 p-5 transition hover:-translate-y-0.5 hover:border-ink-700 hover:bg-ink-850"
            >
              <div className="mb-6 flex items-center justify-between">
                <span className="font-mono text-[11px] text-ink-600">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <span
                  className={`h-2.5 w-2.5 rounded-full ${
                    progress[chapter.slug] === "done"
                      ? "bg-mint-400 shadow-[0_0_12px_rgba(135,183,255,0.75)]"
                      : progress[chapter.slug]
                        ? "bg-flame-500 shadow-[0_0_12px_rgba(101,230,176,0.75)]"
                        : "bg-ink-700"
                  }`}
                  aria-label={
                    progress[chapter.slug] === "done"
                      ? "Completed"
                      : progress[chapter.slug]
                        ? "In progress"
                        : "Not started"
                  }
                />
              </div>
              <div className="text-sm font-semibold text-ink-100 transition group-hover:text-flame-300">
                {chapter.title}
              </div>
              <div className="mt-2 line-clamp-2 text-xs leading-5 text-ink-500">{chapter.summary}</div>
              <div className="mt-5 flex items-center justify-between text-[11px] text-ink-500">
                <span>{chapter.part}</span>
                <span className="text-ink-600 transition group-hover:text-flame-400">Open →</span>
              </div>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
