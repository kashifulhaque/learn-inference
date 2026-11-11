import { useState } from "react";
import { Figure, Segmented, Slider, Stat, color, useWidth } from "../kit";

// Values come from scripts/figures/17c-observability-and-cold-start.py, which
// runs the lab solution on its seeded synthetic event log (400 requests, 15,603
// inter-token gaps). The exact values use chapter 17's percentile. Nothing here
// is measured on a GPU.
const SERIES = {
  ttft: {
    label: "TTFT p99",
    unit: "s",
    exact: 1.87076,
    max: 5,
    layouts: {
      coarse: { bounds: [0.05, 0.1, 0.5, 1, 5], estimate: 4.15789 },
      lab: { bounds: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5], estimate: 2.18421 },
      fine: {
        bounds: [0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5],
        estimate: 1.875,
      },
    },
  },
  itl: {
    label: "ITL p50",
    unit: "s",
    exact: 0.0278,
    max: 0.06,
    layouts: {
      coarse: { bounds: [0.05], estimate: 0.02518 },
      lab: { bounds: [0.005, 0.01, 0.025, 0.05], estimate: 0.03759 },
      fine: { bounds: [0.025, 0.03, 0.035, 0.04, 0.045, 0.05], estimate: 0.02854 },
    },
  },
} as const;

type SeriesKey = keyof typeof SERIES;
type Layout = "coarse" | "lab" | "fine";
const LAYOUTS: readonly Layout[] = ["coarse", "lab", "fine"];

export default function BucketQuantile() {
  const [key, setKey] = useState<SeriesKey>("ttft");
  const [layoutIndex, setLayoutIndex] = useState(0);
  const layout = LAYOUTS[layoutIndex];
  const [ref, width] = useWidth();
  const series = SERIES[key];
  const { bounds, estimate } = series.layouts[layout];
  const edges = [0, ...bounds.filter((b) => b <= series.max)];
  const hi = edges.find((b) => b >= series.exact) ?? series.max;
  const lo = [...edges].reverse().find((b) => b < series.exact) ?? 0;
  const left = 16;
  const right = 16;
  const height = 120;
  const innerW = Math.max(width - left - right, 40);
  const xOf = (value: number) => left + (Math.min(value, series.max) / series.max) * innerW;
  const err = estimate - series.exact;
  const fmt = (v: number) => (v >= 1 ? v.toFixed(3) : v.toFixed(4));

  return (
    <Figure
      title="A quantile from buckets is a guess inside one bucket"
      controls={
        <>
          <Segmented
            label="Series"
            value={key}
            options={[
              { value: "ttft", label: "TTFT p99" },
              { value: "itl", label: "ITL p50" },
            ]}
            onChange={setKey}
          />
          <Slider
            label="Buckets"
            values={[0, 1, 2]}
            value={layoutIndex}
            onChange={setLayoutIndex}
            format={(v) => LAYOUTS[v]}
          />
        </>
      }
      readout={
        <>
          <Stat label="Exact, from samples" value={`${fmt(series.exact)} s`} tone={color.ok} />
          <Stat label="From buckets" value={`${fmt(estimate)} s`} tone={color.accent} />
          <Stat label="Error" value={`${err >= 0 ? "+" : ""}${fmt(err)} s`} />
          <Stat label="Bucket width" value={`${fmt(hi - lo)} s`} />
        </>
      }
      caption="Seeded synthetic event log, 400 requests. The shaded band is the bucket that holds the exact quantile. The estimate always lands inside that band, so the error is at most its width. Values are computed by the lab solution, not measured."
    >
      <div ref={ref}>
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={`${series.label}: exact ${fmt(series.exact)} seconds, bucket estimate ${fmt(estimate)} seconds`}
        >
          <rect x={xOf(lo)} y={28} width={Math.max(xOf(hi) - xOf(lo), 1)} height={44} fill={color.tint} />
          <line x1={left} x2={left + innerW} y1={50} y2={50} stroke={color.lineStrong} />
          {edges.map((b) => (
            <line key={b} x1={xOf(b)} x2={xOf(b)} y1={40} y2={60} stroke={color.muted} />
          ))}
          <circle cx={xOf(series.exact)} cy={50} r={6} fill={color.ok} />
          <rect x={xOf(estimate) - 5} y={45} width={10} height={10} fill={color.accent} />
          <text x={left} y={92} fill={color.muted} fontSize={11}>
            0
          </text>
          <text x={left + innerW} y={92} textAnchor="end" fill={color.muted} fontSize={11}>
            {series.max} s
          </text>
          <text x={left + innerW / 2} y={108} textAnchor="middle" fill={color.muted} fontSize={11}>
            Circle: exact. Square: from buckets. Ticks: bucket bounds.
          </text>
        </svg>
      </div>
    </Figure>
  );
}
