// Chapter 9, "The memory arithmetic": cache memory against context length for
// the hybrid model and for an all-full-attention model of the same geometry,
// on log-log axes, with chapter 2's cache budget as a ceiling.

import { useState } from "react";
import { Figure, Legend, Slider, Stat, color, formatBytes, formatCount, useWidth } from "../kit";
import { log10Scale } from "./scales";

// From chapter 9, "The memory arithmetic" and "Break-even against an
// all-full-attention model".
const KV_BYTES_PER_TOKEN = 65_536; // 16 full-attention layers × 4 KiB.
const STATE_BYTES = 154_927_104; // 48 linear layers' recurrent state and conv window, 147.8 MiB.
const FULL_BYTES_PER_TOKEN = 262_144; // All 64 layers with full attention, 256 KiB.
const BUDGET_BYTES = 19 * 2 ** 30; // Chapter 2's cache budget, "roughly 19 GiB".
const BREAK_EVEN = STATE_BYTES / (FULL_BYTES_PER_TOKEN - KV_BYTES_PER_TOKEN); // 788 tokens.

const GIB = 2 ** 30;
const CONTEXTS = [256, 512, 788, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144] as const;
const BATCHES = [1, 2, 4, 8, 16, 32, 64] as const;
const L_MIN = 256;
const L_MAX = 262144;
const Y_MIN = 1 / 64; // GiB
const Y_MAX = 8192; // GiB
const HEIGHT = 250;
const M = { top: 12, right: 12, bottom: 34, left: 46 };

const kv = (b: number, l: number) => b * l * KV_BYTES_PER_TOKEN;
const state = (b: number) => b * STATE_BYTES;
const full = (b: number, l: number) => b * l * FULL_BYTES_PER_TOKEN;

function gib(bytes: number): string {
  const g = bytes / GIB;
  if (g >= 100) return `${g.toFixed(0)} GiB`;
  if (g >= 10) return `${g.toFixed(1)} GiB`;
  return `${g.toFixed(2)} GiB`;
}

export default function CacheMemory() {
  const [ref, width] = useWidth();
  const [context, setContext] = useState<number>(32768);
  const [batch, setBatch] = useState<number>(8);

  const x = log10Scale(L_MIN, L_MAX, M.left, width - M.right);
  const y = log10Scale(Y_MIN, Y_MAX, HEIGHT - M.bottom, M.top);
  const yb = (bytes: number) => y(Math.min(Y_MAX, Math.max(Y_MIN, bytes / GIB)));

  // Sample each curve along the context axis; the hybrid total bends, so it
  // needs more than two points.
  const samples = Array.from({ length: 81 }, (_, i) => L_MIN * (L_MAX / L_MIN) ** (i / 80));
  const path = (f: (l: number) => number) =>
    samples.map((l, i) => `${i === 0 ? "M" : "L"}${x(l).toFixed(1)},${yb(f(l)).toFixed(1)}`).join("");

  const hybridTotal = kv(batch, context) + state(batch);
  const fullTotal = full(batch, context);
  const fits = hybridTotal <= BUDGET_BYTES;
  const xTicks = [256, 1024, 4096, 16384, 65536, 262144];
  const yTicks = [0.1, 1, 10, 100, 1000];
  const cx = x(context);
  const narrow = width < 420;

  return (
    <Figure
      title="Cache memory against context"
      controls={
        <>
          <Slider label="Context, L" value={context} values={CONTEXTS} onChange={setContext} format={(v) => formatCount(v)} />
          <Slider label="Batch, B" value={batch} values={BATCHES} onChange={setBatch} />
        </>
      }
      readout={
        <>
          <Stat label="Batch × context" value={`${batch} × ${formatCount(context)}`} />
          <Stat label="KV cache" value={gib(kv(batch, context))} tone={color.c} />
          <Stat label="Recurrent state" value={gib(state(batch))} tone={color.d} />
          <Stat label="Hybrid total" value={gib(hybridTotal)} tone={color.accent} />
          <Stat label="All full attention" value={gib(fullTotal)} />
          <Stat label="Hybrid fits in 19 GiB?" value={fits ? "Yes" : "No"} tone={fits ? color.ok : color.bad} />
        </>
      }
      caption={
        <>
          The recurrent state is a flat {formatBytes(STATE_BYTES)} per sequence, so the hybrid costs more than full
          attention below {Math.round(BREAK_EVEN)} tokens and less above it. Batch multiplies both terms and lifts
          every curve toward the 19 GiB budget.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={HEIGHT}
          role="img"
          aria-label={`Log-log plot of cache memory against context length for batch ${batch}. The hybrid model's recurrent state is flat at ${gib(state(batch))}, its KV cache grows with context, and it crosses the all-full-attention line at ${Math.round(BREAK_EVEN)} tokens. At ${formatCount(context)} tokens the hybrid needs ${gib(hybridTotal)} against ${gib(fullTotal)} for full attention, which ${fits ? "fits within" : "exceeds"} the 19 GiB budget.`}
        >
          {yTicks.map((t) => (
            <g key={t}>
              <line x1={M.left} x2={width - M.right} y1={y(t)} y2={y(t)} style={{ stroke: color.line }} />
              <text x={M.left - 5} y={y(t) + 4} textAnchor="end">
                {t >= 1 ? formatCount(t) : t}
              </text>
            </g>
          ))}
          <text x={4} y={M.top + 4} style={{ fill: color.muted }}>
            GiB
          </text>
          {xTicks.map((t) => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={HEIGHT - M.bottom} y2={HEIGHT - M.bottom + 4} style={{ stroke: color.lineStrong }} />
              <text x={x(t)} y={HEIGHT - M.bottom + 15} textAnchor="middle">
                {formatCount(t, true)}
              </text>
            </g>
          ))}
          <text x={width - M.right} y={HEIGHT - 3} textAnchor="end" style={{ fill: color.muted }}>
            context, tokens (log scale)
          </text>
          <line
            x1={M.left}
            x2={width - M.right}
            y1={HEIGHT - M.bottom}
            y2={HEIGHT - M.bottom}
            style={{ stroke: color.lineStrong }}
          />

          {/* The budget. */}
          <line
            x1={M.left}
            x2={width - M.right}
            y1={yb(BUDGET_BYTES)}
            y2={yb(BUDGET_BYTES)}
            style={{ stroke: color.bad, strokeDasharray: "5 3", strokeWidth: 1.25 }}
          />
          <text x={M.left + 4} y={yb(BUDGET_BYTES) - 5} style={{ fill: color.bad }}>
            19 GiB budget
          </text>

          {/* The break-even context. */}
          <line
            x1={x(BREAK_EVEN)}
            x2={x(BREAK_EVEN)}
            y1={M.top}
            y2={HEIGHT - M.bottom}
            style={{ stroke: color.faint, strokeDasharray: "2 3" }}
          />
          <text x={x(BREAK_EVEN) + 4} y={HEIGHT - M.bottom - 6} style={{ fill: color.muted }}>
            {narrow ? "788" : "break-even, 788"}
          </text>

          {/* The two terms of the hybrid, then the two totals. */}
          <path d={path((l) => kv(batch, l))} style={{ fill: "none", stroke: color.c, strokeWidth: 1.5, strokeDasharray: "4 3" }} />
          <path d={path(() => state(batch))} style={{ fill: "none", stroke: color.d, strokeWidth: 1.5, strokeDasharray: "4 3" }} />
          <path d={path((l) => full(batch, l))} style={{ fill: "none", stroke: color.muted, strokeWidth: 2 }} />
          <path d={path((l) => kv(batch, l) + state(batch))} style={{ fill: "none", stroke: color.accent, strokeWidth: 2.5 }} />

          {/* The chosen context. */}
          <line x1={cx} x2={cx} y1={M.top} y2={HEIGHT - M.bottom} style={{ stroke: color.lineStrong }} />
          <circle cx={cx} cy={yb(fullTotal)} r={4} style={{ fill: color.card, stroke: color.muted, strokeWidth: 2 }} />
          <circle cx={cx} cy={yb(hybridTotal)} r={4.5} style={{ fill: color.accent, stroke: color.card, strokeWidth: 1.5 }} />
        </svg>
        <Legend
          items={[
            { label: "Hybrid total", tone: color.accent },
            { label: "All full attention", tone: color.muted },
            { label: "Hybrid KV cache", tone: color.c, dashed: true },
            { label: "Hybrid recurrent state", tone: color.d, dashed: true },
          ]}
        />
      </div>
    </Figure>
  );
}
