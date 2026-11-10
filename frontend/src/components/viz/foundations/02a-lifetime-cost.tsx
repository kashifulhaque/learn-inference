import { useState } from "react";
import { Figure, Legend, Slider, Stat, color, formatCount, useWidth } from "../kit";
import { logarithmic, polyline } from "./helpers";

// Chapter 2a, "Training plus serving": the fitted loss of Hoffmann et al. with
// the digits Besiroglu et al. report, and the lab's search over N.
const E = 1.6934;
const A = 406.4;
const B = 410.7;
const ALPHA = 0.3392;
const BETA = 0.2849;
const N_BASE = 26_895_979_520; // chapter 2's parameter count

const G = ((ALPHA * A) / (BETA * B)) ** (1 / (ALPHA + BETA));
const A_EXP = BETA / (ALPHA + BETA);
const D_BASE = (N_BASE / G) ** (1 / A_EXP); // tokens at which N_BASE is compute-optimal
const TARGET = E + A / N_BASE ** ALPHA + B / D_BASE ** BETA;
const N_FLOOR = (A / (TARGET - E)) ** (1 / ALPHA);

const tokensFor = (n: number) => (B / (TARGET - E - A / n ** ALPHA)) ** (1 / BETA);
const lifetime = (n: number, dInf: number) => 6 * n * tokensFor(n) + 2 * n * dInf;

function optimum(dInf: number): number {
  let lo = Math.log(N_FLOOR * (1 + 1e-6));
  let hi = Math.log(N_FLOOR * 1e3);
  const phi = (Math.sqrt(5) - 1) / 2;
  let x1 = hi - phi * (hi - lo);
  let x2 = lo + phi * (hi - lo);
  let f1 = lifetime(Math.exp(x1), dInf);
  let f2 = lifetime(Math.exp(x2), dInf);
  for (let i = 0; i < 200; i++) {
    if (f1 < f2) {
      hi = x2;
      x2 = x1;
      f2 = f1;
      x1 = hi - phi * (hi - lo);
      f1 = lifetime(Math.exp(x1), dInf);
    } else {
      lo = x1;
      x1 = x2;
      f1 = f2;
      x2 = lo + phi * (hi - lo);
      f2 = lifetime(Math.exp(x2), dInf);
    }
  }
  return Math.exp((lo + hi) / 2);
}

const VOLUMES = [0, 1e11, 1e12, 1e13, 1e14, 1e15] as const;
const HEIGHT = 230;
const M = { left: 46, right: 12, top: 10, bottom: 38 };
const N_MIN = N_FLOOR * 1.02;
const N_MAX = 60e9;
const RATIO_MAX = 4;
const X_TICKS = [2e9, 5e9, 10e9, 20e9, 50e9];

const billions = (n: number) => `${(n / 1e9).toFixed(n < 1e10 ? 2 : 1)}B`;

export default function LifetimeCost() {
  const [ref, width] = useWidth();
  const [dInf, setDInf] = useState<number>(1e13);

  const nStar = optimum(dInf);
  const dStar = tokensFor(nStar);
  const best = lifetime(nStar, dInf);
  const atBase = lifetime(N_BASE, dInf);
  const saving = (1 - best / atBase) * 100;

  const x = logarithmic(N_MIN, N_MAX, M.left, width - M.right);
  const y = (r: number) => M.top + (1 - (r - 1) / (RATIO_MAX - 1)) * (HEIGHT - M.top - M.bottom);
  const samples = Array.from({ length: 120 }, (_, k) => N_MIN * (N_MAX / N_MIN) ** (k / 119));
  const curve = polyline(samples.map((n) => [x(n), y(Math.min(RATIO_MAX, lifetime(n, dInf) / best))] as const));
  const bottom = HEIGHT - M.bottom;

  return (
    <Figure
      title="Lifetime FLOPs against model size at a fixed loss"
      controls={
        <Slider
          label="Served tokens"
          values={VOLUMES}
          value={dInf}
          onChange={setDInf}
          format={(v) => (v === 0 ? "none" : formatCount(v, true))}
        />
      }
      readout={
        <>
          <Stat label="Best N" value={billions(nStar)} tone={color.accent} />
          <Stat label="Tokens trained" value={formatCount(dStar, true)} />
          <Stat label="Tokens per parameter" value={Math.round(dStar / nStar).toLocaleString("en-US")} />
          <Stat label="Saved against 26.9B" value={`${saving.toFixed(1)}%`} />
        </>
      }
      caption={
        <>
          Every point on the curve reaches the same loss, the one a compute-optimal 26.9B model reaches under the fit.
          A smaller N needs more training tokens. The cost is 6ND of training plus 2N per served token, relative to the
          best point. As served tokens grow, the valley moves left.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={HEIGHT}
          role="img"
          aria-label={`Lifetime FLOPs relative to the best model size, against parameters. With ${dInf === 0 ? "no" : formatCount(dInf, true)} served tokens the best size is ${billions(nStar)} on ${formatCount(dStar, true)} training tokens, which saves ${saving.toFixed(1)} percent against 26.9 billion parameters.`}
        >
          {X_TICKS.map((t) => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={M.top} y2={bottom} style={{ stroke: color.line }} />
              <text x={x(t)} y={bottom + 14} textAnchor="middle">
                {billions(t)}
              </text>
            </g>
          ))}
          {[1, 2, 3, 4].map((r) => (
            <g key={r}>
              <line x1={M.left} x2={width - M.right} y1={y(r)} y2={y(r)} style={{ stroke: color.line }} />
              <text x={M.left - 6} y={y(r) + 4} textAnchor="end">
                {r}x
              </text>
            </g>
          ))}
          <text x={(M.left + width - M.right) / 2} y={HEIGHT - 4} textAnchor="middle">
            Parameters N
          </text>
          <line
            x1={x(N_BASE)}
            x2={x(N_BASE)}
            y1={M.top}
            y2={bottom}
            strokeDasharray="5 4"
            style={{ stroke: color.muted }}
          />
          <text x={x(N_BASE) + 4} y={M.top + 10} style={{ fill: color.muted }}>
            26.9B
          </text>
          <path d={curve} style={{ fill: "none", stroke: color.accent, strokeWidth: 2 }} />
          <circle cx={x(nStar)} cy={y(1)} r={4.5} style={{ fill: color.accent, stroke: color.card, strokeWidth: 2 }} />
        </svg>
        <div style={{ marginTop: "0.5rem" }}>
          <Legend
            items={[
              { label: "Lifetime cost, relative to the best N", tone: color.accent },
              { label: "Chapter 2's N", tone: color.muted, dashed: true },
            ]}
          />
        </div>
      </div>
    </Figure>
  );
}
