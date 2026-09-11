import { useEffect, useMemo, useState } from "react";
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

  const attempts = useMemo(
    () =>
      Object.entries(
        runs.reduce<Record<string, { total: number; passed: number }>>((acc, run) => {
          const entry = (acc[run.lab] ||= { total: 0, passed: 0 });
          entry.total += 1;
          if (run.passed) entry.passed += 1;
          return acc;
        }, {}),
      )
        .map(([lab, counts]) => ({ lab: lab.slice(0, 2), ...counts }))
        .sort((a, b) => a.lab.localeCompare(b.lab)),
    [runs],
  );

  const parts = chapters.reduce<Record<string, ChapterMeta[]>>((groups, chapter) => {
    (groups[chapter.part] ||= []).push(chapter);
    return groups;
  }, {});
  const numbers = new Map(chapters.map((chapter, index) => [chapter.slug, index + 1]));

  const tiles = [
    { label: "Course complete", value: `${completion}%`, note: `${done.length} of ${chapters.length} chapters` },
    { label: "Labs passed", value: String(passed), note: "verified runs" },
    { label: "Total runs", value: String(runs.length), note: "across all labs" },
    { label: "Compute target", value: gpu || "—", note: "active accelerator" },
  ];

  return (
    <div className="mx-auto max-w-6xl px-5 py-7 sm:px-8">
      <section className="surface-grid relative overflow-hidden rounded-2xl border border-ink-800 bg-ink-900/70 px-6 py-8 sm:px-8">
        <div className="pointer-events-none absolute -right-24 -top-28 h-72 w-72 rounded-full bg-flame-500/10 blur-3xl" />
        <div className="relative max-w-2xl">
          <div className="mb-4 flex items-center gap-2.5">
            <span className="h-1.5 w-1.5 rounded-full bg-flame-500" />
            <span className="text-[10px] font-bold uppercase tracking-[0.18em] text-flame-400">
              Course dashboard
            </span>
          </div>
          <h1 className="text-3xl font-semibold tracking-[-0.035em] text-ink-100 sm:text-4xl">
            Build an inference engine.
          </h1>
          <p className="mt-4 text-sm leading-6 text-ink-400">
            Trace the path from a safetensors file to a serving engine — CUDA kernels, paged
            cache, continuous batching, and hybrid attention for{" "}
            <span className="rounded bg-ink-850 px-1.5 py-0.5 font-mono text-[0.88em] text-ink-200">
              {model || "your model"}
            </span>
            .
          </p>
          {next && (
            <Link
              to={`/c/${next.slug}`}
              className="mt-6 inline-flex items-center gap-2 rounded-lg bg-flame-500 px-4 py-2 text-xs font-bold text-ink-950 transition hover:bg-flame-400"
            >
              {progress[next.slug] === "in_progress" ? "Continue" : "Start"}: {next.title}
              <span aria-hidden>→</span>
            </Link>
          )}
        </div>
      </section>

      <section className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {tiles.map((tile) => (
          <div
            key={tile.label}
            className="rounded-xl border border-ink-800 bg-ink-900/60 px-4 py-3.5 transition hover:border-ink-700"
          >
            <div className="text-[10px] font-bold uppercase tracking-[0.14em] text-ink-500">
              {tile.label}
            </div>
            <div className="mt-1.5 truncate text-2xl font-semibold tracking-[-0.03em] text-ink-100">
              {tile.value}
            </div>
            <div className="mt-0.5 truncate text-[11px] text-ink-500">{tile.note}</div>
          </div>
        ))}
      </section>

      {attempts.length > 0 && (
        <section className="mt-4 rounded-xl border border-ink-800 bg-ink-900/60 p-5">
          <div className="mb-4 flex items-start justify-between gap-4">
            <div>
              <div className="text-[10px] font-bold uppercase tracking-[0.14em] text-ink-500">
                Lab activity
              </div>
              <h2 className="mt-1 text-base font-semibold tracking-tight text-ink-100">
                Attempts and passes
              </h2>
            </div>
            <span className="rounded-lg border border-ink-800 bg-ink-850 px-2.5 py-1 text-[11px] text-ink-400">
              Run history
            </span>
          </div>
          <div className="h-52">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={attempts}>
                <CartesianGrid stroke="#172a40" vertical={false} />
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
                <Bar dataKey="total" fill="#46617f" radius={[4, 4, 0, 0]} name="runs" />
                <Bar dataKey="passed" fill="#65e6b0" radius={[4, 4, 0, 0]} name="passed" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      )}

      <section className="mt-8">
        <div className="mb-4 flex items-end justify-between gap-4">
          <div>
            <div className="text-[10px] font-bold uppercase tracking-[0.14em] text-ink-500">
              Curriculum
            </div>
            <h2 className="mt-1 text-lg font-semibold tracking-tight text-ink-100">All chapters</h2>
          </div>
          <span className="text-[11px] text-ink-500">{chapters.length} chapters</span>
        </div>

        <div className="space-y-6">
          {Object.entries(parts).map(([part, items]) => (
            <div key={part}>
              <div className="mb-2 flex items-center gap-3">
                <span className="text-[10px] font-bold uppercase tracking-[0.14em] text-ink-500">
                  {part}
                </span>
                <span className="h-px flex-1 bg-ink-800" />
                <span className="font-mono text-[10px] text-ink-600">
                  {items.filter((c) => progress[c.slug] === "done").length}/{items.length}
                </span>
              </div>
              <div className="overflow-hidden rounded-xl border border-ink-800">
                {items.map((chapter) => (
                  <Link
                    key={chapter.slug}
                    to={`/c/${chapter.slug}`}
                    className="group flex items-center gap-3.5 border-b border-ink-800 bg-ink-900/50 px-4 py-3 transition last:border-b-0 hover:bg-ink-850"
                  >
                    <span
                      className={`flex h-7 w-8 shrink-0 items-center justify-center rounded-lg border font-mono text-[11px] ${
                        progress[chapter.slug] === "done"
                          ? "border-mint-400 bg-mint-400/12 text-mint-400"
                          : progress[chapter.slug]
                            ? "border-flame-500 bg-flame-500/12 text-flame-400"
                            : "border-ink-700 text-ink-500"
                      }`}
                    >
                      {String(numbers.get(chapter.slug)).padStart(2, "0")}
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-[13px] font-semibold text-ink-100 transition group-hover:text-flame-300">
                        {chapter.title}
                      </span>
                      <span className="mt-0.5 block truncate text-[11px] leading-5 text-ink-500">
                        {chapter.summary}
                      </span>
                    </span>
                    {chapter.gpu && (
                      <span className="hidden shrink-0 rounded bg-flame-500/12 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wide text-flame-400 sm:inline">
                        lab
                      </span>
                    )}
                    {chapter.minutes && (
                      <span className="hidden w-14 shrink-0 text-right text-[11px] text-ink-600 sm:inline">
                        {chapter.minutes} min
                      </span>
                    )}
                    <span className="shrink-0 text-ink-700 transition group-hover:text-flame-400">
                      →
                    </span>
                  </Link>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
