import { useState } from "react";
import { Figure, Legend, Slider, Stat, color, useWidth } from "../kit";
import { YTicks } from "./svg";

// Formulas from chapter 20, "Expected tokens per round" and "Net speedup, and
// the optimal draft length": E[N + 1] = (1 - a^(g+1)) / (1 - a), a round costs
// 1 + g c target passes, and speedup = E[N + 1] / (1 + g c). The default
// alpha = 0.8 and c = 0.05 is the chapter's optimum table (gamma 8, 3.09x).
// The c stops include the chapter's multi-token prediction head (about 0.016),
// the small same-family draft (0.022), and the lab's expensive draft (0.5).
const ALPHAS = [0.5, 0.6, 0.7, 0.8, 0.9] as const;
const COSTS = [0, 0.016, 0.022, 0.05, 0.1, 0.2, 0.5] as const;
const G_MAX = 16;

const tokens = (a: number, g: number) => (1 - a ** (g + 1)) / (1 - a);
const cost = (c: number, g: number) => 1 + g * c;
const speedup = (a: number, g: number, c: number) => tokens(a, g) / cost(c, g);

/** The best integer draft length, reporting a tie when two agree to three decimals. */
function optimum(a: number, c: number) {
  const values = Array.from({ length: G_MAX }, (_, i) => ({ g: i + 1, s: speedup(a, i + 1, c) }));
  const best = values.reduce((m, v) => (v.s > m.s ? v : m));
  const ties = values.filter((v) => v.s.toFixed(3) === best.s.toFixed(3)).map((v) => v.g);
  return { best, ties, atEdge: best.g === G_MAX };
}

const HEIGHT = 230;

export default function SpeedupCurve() {
  const [alpha, setAlpha] = useState(0.8);
  const [c, setC] = useState(0.05);
  const [ref, width] = useWidth();

  const ceiling = 1 / (1 - alpha);
  const opt = optimum(alpha, c);
  const yMax = Math.ceil(Math.max(ceiling, cost(c, G_MAX) > ceiling * 1.6 ? ceiling * 1.6 : cost(c, G_MAX)) + 0.5);
  const step = yMax > 12 ? 4 : yMax > 6 ? 2 : 1;
  const ticks = Array.from({ length: Math.floor(yMax / step) + 1 }, (_, i) => i * step);

  const left = 28;
  const right = 8;
  const top = 10;
  const bottom = 32;
  const pw = width - left - right;
  const ph = HEIGHT - top - bottom;
  const x = (g: number) => left + ((g - 1) / (G_MAX - 1)) * pw;
  const y = (v: number) => top + ph - (Math.min(v, yMax) / yMax) * ph;
  const gammas = Array.from({ length: G_MAX }, (_, i) => i + 1);
  const path = (f: (g: number) => number) => gammas.map((g) => `${x(g)},${y(f(g))}`).join(" ");
  // The cost line can run off the top; stop it there.
  const costEnd = Math.min(G_MAX, c > 0 ? (yMax - 1) / c : G_MAX);

  const gLabel = opt.atEdge ? `${G_MAX} or more` : opt.ties.length > 1 ? `${opt.ties[0]} to ${opt.ties[opt.ties.length - 1]}` : `${opt.best.g}`;

  return (
    <Figure
      title="Speedup against draft length"
      controls={
        <>
          <Slider label="Acceptance α" values={ALPHAS} value={alpha} onChange={setAlpha} format={(v) => v.toFixed(1)} />
          <Slider label="Draft cost c" values={COSTS} value={c} onChange={setC} format={(v) => String(v)} />
        </>
      }
      readout={
        <>
          <Stat label="α, c" value={`${alpha.toFixed(1)}, ${c}`} />
          <Stat label="Best draft length γ" value={gLabel} />
          <Stat label="Speedup there" value={`${opt.best.s.toFixed(2)}×`} />
          <Stat label="Tokens per round there" value={tokens(alpha, opt.best.g).toFixed(2)} tone={color.a} />
          <Stat label="Ceiling 1/(1 − α)" value={`${ceiling.toFixed(1)}×`} />
        </>
      }
      caption={
        <>
          Tokens per round saturate toward 1/(1 − α) while a round's cost keeps rising with γ, so their ratio, the
          speedup, peaks and then falls. Raise α, and both the peak and the best γ move up steeply.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={HEIGHT}
          viewBox={`0 0 ${width} ${HEIGHT}`}
          role="img"
          aria-label={`At acceptance ${alpha.toFixed(1)} and draft cost ${c}, the speedup peaks at draft length ${gLabel} with ${opt.best.s.toFixed(2)} times, below the ceiling of ${ceiling.toFixed(1)}.`}
        >
          <YTicks ticks={ticks} y={y} x0={left} x1={left + pw} />
          {[1, 4, 8, 12, 16].map((g) => (
            <text key={g} x={x(g)} y={top + ph + 14} textAnchor="middle">
              {g}
            </text>
          ))}
          <text x={left + pw / 2} y={HEIGHT - 3} textAnchor="middle">
            Draft length γ
          </text>

          <line
            x1={left}
            x2={left + pw}
            y1={y(ceiling)}
            y2={y(ceiling)}
            style={{ stroke: color.subtle, strokeDasharray: "2 3" }}
          />
          <text x={left + 4} y={y(ceiling) - 5} style={{ fill: color.subtle }}>
            ceiling {ceiling.toFixed(1)}
          </text>

          <polyline points={path((g) => tokens(alpha, g))} style={{ fill: "none", stroke: color.a, strokeWidth: 2 }} />
          <line
            x1={x(1)}
            y1={y(cost(c, 1))}
            x2={x(costEnd)}
            y2={y(cost(c, costEnd))}
            style={{ stroke: color.b, strokeWidth: 2, strokeDasharray: "5 3" }}
          />
          <polyline points={path((g) => speedup(alpha, g, c))} style={{ fill: "none", stroke: color.fg, strokeWidth: 2.5 }} />
          {gammas.map((g) => (
            <circle key={g} cx={x(g)} cy={y(speedup(alpha, g, c))} r={2} style={{ fill: color.fg }} />
          ))}
          <circle
            cx={x(opt.best.g)}
            cy={y(opt.best.s)}
            r={5.5}
            style={{ fill: color.accent, stroke: color.card, strokeWidth: 1.5 }}
          />
          <text
            x={x(opt.best.g) + (opt.best.g > G_MAX * 0.7 ? -9 : 9)}
            y={y(opt.best.s) + 16}
            textAnchor={opt.best.g > G_MAX * 0.7 ? "end" : "start"}
            style={{ fill: color.fg, fontWeight: 600 }}
          >
            {opt.best.s.toFixed(2)}× at γ = {opt.best.g}
          </text>
        </svg>
      </div>
      <Legend
        items={[
          { label: "Tokens per round, E[N + 1]", tone: color.a },
          { label: "Cost per round, 1 + γc", tone: color.b, dashed: true },
          { label: "Speedup, their ratio", tone: color.fg },
        ]}
      />
    </Figure>
  );
}
