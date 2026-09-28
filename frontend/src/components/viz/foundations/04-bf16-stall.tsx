import { useState } from "react";
import { Figure, Legend, Slider, Stat, color, formatCount, useWidth } from "../kit";
import { linear, polyline, toBfloat16 } from "./helpers";

// Chapter 4, "What that does to a 5120-element reduction": a left-to-right sum
// over d squared activations, idealized so that every x_j^2 is the same value
// s. Here s = 1. The float32 sum is exact at these sizes (every integer up to
// 2^24 is representable), and the bfloat16 sum rounds after every addition.
const TERM = 1;
const LENGTHS = [64, 128, 256, 512, 1024, 2048, 4096, 5120] as const;

/** The running bfloat16 sum after each of `d` additions, starting from 0. */
function bf16Sums(d: number): number[] {
  const sums = [0];
  let s = 0;
  for (let k = 0; k < d; k += 1) {
    s = toBfloat16(s + TERM);
    sums.push(s);
  }
  return sums;
}

const HEIGHT = 220;
const M = { left: 44, right: 12, top: 12, bottom: 36 };

export default function Bf16Stall() {
  const [ref, width] = useWidth();
  const [d, setD] = useState<number>(5120);

  const sums = bf16Sums(d);
  const bf16 = sums[d];
  const exact = d * TERM;
  const stallAt = sums.findIndex((s, k) => k > 0 && s === sums[k - 1]); // first addend that vanished
  const x = linear(0, d, M.left, width - M.right);
  const y = linear(0, d, HEIGHT - M.bottom, M.top);
  const bottom = HEIGHT - M.bottom;
  const right = width - M.right;
  const ticks = [0, d / 2, d];

  // Keep the first point, every point where the slope changes, and the last.
  const points: [number, number][] = [];
  sums.forEach((s, k) => {
    const bend = k > 0 && k < d && s - sums[k - 1] !== sums[k + 1] - s;
    if (k === 0 || k === d || bend) points.push([x(k), y(s)]);
  });
  const bf16Path = polyline(points);
  const scale = Math.sqrt(exact / bf16);

  return (
    <Figure
      title="A bfloat16 running sum stalls at 256"
      controls={
        <Slider label="Reduction length d" values={LENGTHS} value={d} onChange={setD} format={(v) => formatCount(v)} />
      }
      readout={
        <>
          <Stat label="Terms d" value={formatCount(d)} />
          <Stat label="float32 sum" value={formatCount(exact)} tone={color.info} />
          <Stat label="bfloat16 sum" value={formatCount(bf16)} tone={color.bad} />
          <Stat label="Computed ms(x)" value={bf16 === exact ? "s, exact" : `s / ${+(exact / bf16).toFixed(2)}`} />
          <Stat label="Output too large by" value={`${scale.toFixed(2)}x`} />
        </>
      }
      caption={
        <>
          Every squared activation is idealized as 1, as in the stall derivation. Up to 256 terms the bfloat16 sum is
          exact; after that each new term is half an ulp or less and rounds away, so at <em>d</em> = 5120 the sum is a
          twentieth of the truth. A GPU's tree-shaped reduction avoids most of this.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={HEIGHT}
          role="img"
          aria-label={`Running sum against terms added, for d = ${d}. The float32 sum rises to ${exact}; the bfloat16 sum ${stallAt > 0 ? `stops at ${bf16} after ${stallAt - 1} terms` : `tracks it exactly to ${bf16}`}.`}
        >
          {ticks.map((t) => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={M.top} y2={bottom} style={{ stroke: color.line }} />
              <text x={x(t)} y={bottom + 14} textAnchor={t === 0 ? "start" : t === d ? "end" : "middle"}>
                {formatCount(t)}
              </text>
              <line x1={M.left} x2={right} y1={y(t)} y2={y(t)} style={{ stroke: color.line }} />
              <text x={M.left - 6} y={y(t) + 4} textAnchor="end">
                {formatCount(t)}
              </text>
            </g>
          ))}
          <text x={(M.left + right) / 2} y={HEIGHT - 4} textAnchor="middle">
            Terms added, k
          </text>
          <text x={M.left + 6} y={M.top + 12} style={{ fill: color.muted }}>
            Running sum
          </text>

          <line x1={x(0)} y1={y(0)} x2={x(d)} y2={y(exact)} style={{ stroke: color.info, strokeWidth: 2 }} />
          <path d={bf16Path} style={{ fill: "none", stroke: color.bad, strokeWidth: 2 }} />

          {stallAt > 0 && (
            <>
              <circle cx={x(stallAt - 1)} cy={y(bf16)} r={4} style={{ fill: color.bad, stroke: color.card, strokeWidth: 2 }} />
              <text x={Math.min(x(stallAt - 1) + 8, right - 120)} y={y(bf16) - 8} style={{ fill: color.bad }}>
                {`stalls at ${formatCount(bf16)}`}
              </text>
            </>
          )}
        </svg>
        <div style={{ marginTop: "0.5rem" }}>
          <Legend
            items={[
              { label: "float32", tone: color.info },
              { label: "bfloat16", tone: color.bad },
            ]}
          />
        </div>
      </div>
    </Figure>
  );
}
