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

  // One bar per lab, showing how many attempts it took to get a pass.
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
    <div className="mx-auto max-w-5xl px-6 py-12 lg:px-10">
      <div className="mb-1 text-xs font-medium uppercase tracking-[0.2em] text-flame-500">
        learn-inference
      </div>
      <h1 className="mb-2 text-3xl font-semibold tracking-tight text-ink-100">
        Build an inference engine
      </h1>
      <p className="mb-10 max-w-2xl text-ink-400">
        Twenty-one chapters that take you from a safetensors file to a serving
        engine: CUDA kernels, a paged cache, continuous batching, and the hybrid
        attention stack behind{" "}
        <span className="font-mono text-sm text-ink-300">{model}</span>. The labs
        run on an {gpu}.
      </p>

      <div className="mb-10 grid grid-cols-2 gap-px overflow-hidden rounded-2xl border border-ink-800 bg-ink-800 md:grid-cols-4">
        {[
          { label: "Chapters done", value: `${done.length} / ${chapters.length}` },
          { label: "Labs passed", value: String(passed) },
          { label: "Total runs", value: String(runs.length) },
          { label: "GPU", value: gpu },
        ].map((tile) => (
          <div key={tile.label} className="bg-ink-900 px-5 py-4">
            <div className="text-[11px] uppercase tracking-wider text-ink-500">
              {tile.label}
            </div>
            <div className="mt-1 truncate text-xl font-semibold text-ink-100">
              {tile.value}
            </div>
          </div>
        ))}
      </div>

      {next && (
        <Link
          to={`/c/${next.slug}`}
          className="mb-10 block rounded-2xl border border-flame-500/30 bg-flame-500/5 px-6 py-5 transition hover:border-flame-500/60"
        >
          <div className="text-[11px] font-semibold uppercase tracking-wider text-flame-500">
            Pick up here
          </div>
          <div className="mt-1 text-lg font-semibold text-ink-100">{next.title}</div>
          <div className="mt-1 text-sm text-ink-400">{next.summary}</div>
        </Link>
      )}

      {attempts.length > 0 && (
        <section className="mb-10 rounded-2xl border border-ink-800 bg-ink-900 p-6">
          <h2 className="mb-4 text-sm font-semibold text-ink-100">Attempts per lab</h2>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={attempts}>
                <CartesianGrid stroke="#222836" vertical={false} />
                <XAxis dataKey="lab" stroke="#6b7688" fontSize={11} tickLine={false} />
                <YAxis stroke="#6b7688" fontSize={11} tickLine={false} allowDecimals={false} />
                <Tooltip
                  contentStyle={{
                    background: "#12151c",
                    border: "1px solid #222836",
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                />
                <Bar dataKey="total" fill="#333c4f" radius={[3, 3, 0, 0]} name="runs" />
                <Bar dataKey="passed" fill="#3ddc97" radius={[3, 3, 0, 0]} name="passed" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      )}

      <section>
        <h2 className="mb-4 text-sm font-semibold text-ink-100">All chapters</h2>
        <div className="grid gap-3 sm:grid-cols-2">
          {chapters.map((chapter) => (
            <Link
              key={chapter.slug}
              to={`/c/${chapter.slug}`}
              className="rounded-xl border border-ink-800 bg-ink-900 px-5 py-4 transition hover:border-ink-700"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="truncate text-sm font-medium text-ink-100">
                    {chapter.title}
                  </div>
                  <div className="mt-1 line-clamp-2 text-xs text-ink-500">
                    {chapter.summary}
                  </div>
                </div>
                <span
                  className={`mt-1 h-2 w-2 shrink-0 rounded-full ${
                    progress[chapter.slug] === "done"
                      ? "bg-mint-400"
                      : progress[chapter.slug]
                        ? "bg-flame-500"
                        : "bg-ink-700"
                  }`}
                  aria-hidden
                />
              </div>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
