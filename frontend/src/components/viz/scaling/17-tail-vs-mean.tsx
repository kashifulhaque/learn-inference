import { useState } from "react";
import { Figure, Legend, Slider, Stat, color, useWidth, wash } from "../kit";
import { YTicks, spreadLabels } from "./svg";

// Constants from chapter 17, "Why p50 and p99 diverge under batching": a
// decode-only step takes t_d = 30 ms, and a step that also carries a 1024-token
// prefill chunk takes t_m = 90 ms. The worked example has rho = 0.05, one mixed
// step in twenty. The run of 100 steps is an illustrative example, with the
// mixed steps spread evenly through it.
const T_D = 30;
const T_M_DEFAULT = 90;
const STEPS = 100;
const RHO_DEFAULT = 5; // mixed steps per 100
const Y_MAX = 200;

/** The chapter's percentile: linear interpolation between order statistics. */
function percentile(sorted: readonly number[], q: number): number {
  const r = (q / 100) * (sorted.length - 1);
  const i = Math.floor(r);
  if (i >= sorted.length - 1) return sorted[sorted.length - 1];
  return sorted[i] + (r - i) * (sorted[i + 1] - sorted[i]);
}

/** Which steps are mixed: `mixed` of them, spread evenly over the run. */
function run(mixed: number, tm: number): number[] {
  return Array.from({ length: STEPS }, (_, j) =>
    Math.floor(((j + 1) * mixed) / STEPS) > Math.floor((j * mixed) / STEPS) ? tm : T_D,
  );
}

const fmt = (ms: number) => (Number.isInteger(ms) ? `${ms}` : ms.toFixed(1));
const change = (ms: number) => `${ms >= T_D ? "+" : ""}${Math.round((100 * (ms - T_D)) / T_D)}%`;

const HEIGHT = 210;

export default function TailVsMean() {
  const [mixed, setMixed] = useState(RHO_DEFAULT);
  const [tm, setTm] = useState(T_M_DEFAULT);
  const [ref, width] = useWidth();

  const steps = run(mixed, tm);
  const sorted = [...steps].sort((a, b) => a - b);
  const mean = steps.reduce((a, b) => a + b, 0) / STEPS;
  const p50 = percentile(sorted, 50);
  const p99 = percentile(sorted, 99);

  const left = 30;
  const gutter = 92;
  const top = 8;
  const bottom = 22;
  const pw = width - left - gutter;
  const ph = HEIGHT - top - bottom;
  const y = (ms: number) => top + ph - (ms / Y_MAX) * ph;
  const barW = pw / STEPS;

  const lines = [
    { label: `mean ${fmt(mean)}`, ms: mean, tone: color.fg, dash: "5 3" },
    { label: `p50 ${fmt(p50)}`, ms: p50, tone: color.a, dash: "2 2" },
    { label: `p99 ${fmt(p99)}`, ms: p99, tone: color.bad, dash: "" },
  ];
  const labelY = spreadLabels(
    lines.map((l) => y(l.ms) + 4),
    13,
    top + 8,
    top + ph + 4,
  );

  return (
    <Figure
      title="One slow step in twenty moves the tail, not the mean"
      controls={
        <>
          <Slider
            label="Mixed steps per 100"
            min={0}
            max={20}
            value={mixed}
            onChange={setMixed}
            format={(v) => `${v} (ρ = ${(v / STEPS).toFixed(2)})`}
          />
          <Slider label="Mixed step time" min={40} max={180} step={10} value={tm} onChange={setTm} format={(v) => `${v} ms`} />
        </>
      }
      readout={
        <>
          <Stat label="ρ" value={(mixed / STEPS).toFixed(2)} tone={color.c} />
          <Stat label="Decode, mixed step" value={`${T_D} ms, ${tm} ms`} />
          <Stat label="Mean" value={`${fmt(mean)} ms (${change(mean)})`} />
          <Stat label="p50" value={`${fmt(p50)} ms (${change(p50)})`} tone={color.a} />
          <Stat label="p99" value={`${fmt(p99)} ms (${change(p99)})`} tone={color.bad} />
        </>
      }
      caption={
        <>
          Each bar is one step of an example run. Once more than 1 step in 100 is mixed, p99 jumps to the mixed step
          time, while the mean rises only in proportion to ρ.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={HEIGHT}
          viewBox={`0 0 ${width} ${HEIGHT}`}
          role="img"
          aria-label={`${STEPS} decode steps, ${mixed} of them mixed at ${tm} ms and the rest at ${T_D} ms. The mean is ${fmt(mean)} ms, p50 is ${fmt(p50)} ms, and p99 is ${fmt(p99)} ms.`}
        >
          <YTicks ticks={[0, 50, 100, 150, 200]} y={y} x0={left} x1={left + pw} />
          <text x={left - 5} y={HEIGHT - 4} textAnchor="end">
            ms
          </text>
          <text x={left} y={HEIGHT - 4}>
            Step 1
          </text>
          <text x={left + pw} y={HEIGHT - 4} textAnchor="end">
            Step {STEPS}
          </text>
          {steps.map((ms, j) => {
            const isMixed = ms !== T_D;
            const tone = isMixed ? color.b : color.a;
            return (
              <rect
                key={j}
                x={left + j * barW + (barW > 3 ? 0.4 : 0)}
                y={y(ms)}
                width={Math.max(0.6, barW - (barW > 3 ? 0.8 : 0))}
                height={y(0) - y(ms)}
                style={{ fill: isMixed ? tone : wash(tone, 45) }}
              />
            );
          })}
          {lines.map((l, k) => (
            <g key={l.label}>
              <line
                x1={left}
                x2={left + pw + 4}
                y1={y(l.ms)}
                y2={y(l.ms)}
                style={{ stroke: l.tone, strokeWidth: 1.5, strokeDasharray: l.dash || undefined }}
              />
              <line
                x1={left + pw + 4}
                x2={left + pw + 12}
                y1={y(l.ms)}
                y2={labelY[k] - 4}
                style={{ stroke: l.tone, strokeWidth: 1 }}
              />
              <text x={left + pw + 15} y={labelY[k]} style={{ fill: l.tone, fontWeight: 600 }}>
                {l.label} ms
              </text>
            </g>
          ))}
        </svg>
      </div>
      <Legend
        items={[
          { label: `Decode-only step, ${T_D} ms`, tone: wash(color.a, 45) },
          { label: `Step with a prefill chunk, ${tm} ms`, tone: color.b },
        ]}
      />
    </Figure>
  );
}
