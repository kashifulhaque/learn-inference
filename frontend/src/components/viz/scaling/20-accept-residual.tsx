import { useState } from "react";
import { Figure, Legend, Segmented, Stat, color, useWidth, wash } from "../kit";
import { YTicks } from "./svg";

// The target p = [0.5, 0.3, 0.2] and the lab's draft q = [0.25, 0.6, 0.15] come
// from chapter 20's worked example, "The lab's three-token vocabulary". The
// uniform and poor drafts are illustrative examples. The accept path delivers
// min(p, q), and the reject path delivers (p - q)+, which add up to p.
const TARGET = [0.5, 0.3, 0.2] as const;
const DRAFTS = {
  lab: { label: "The lab's draft", q: [0.25, 0.6, 0.15] },
  uniform: { label: "Uniform", q: [1 / 3, 1 / 3, 1 / 3] },
  poor: { label: "A poor draft", q: [0.05, 0.15, 0.8] },
  same: { label: "Same as target", q: [...TARGET] },
} as const;
type Draft = keyof typeof DRAFTS;
const Y_MAX = 0.8;

const f2 = (v: number) => v.toFixed(2);

const HEIGHT = 214;

export default function AcceptResidual() {
  const [draft, setDraft] = useState<Draft>("lab");
  const [ref, width] = useWidth();
  const q = DRAFTS[draft].q;
  const p = TARGET;

  const accepted = p.map((pi, i) => Math.min(pi, q[i]));
  const residual = p.map((pi, i) => Math.max(0, pi - q[i]));
  const alpha = accepted.reduce((a, b) => a + b, 0);
  const reject = 1 - alpha;
  const tv = 0.5 * p.reduce((a, pi, i) => a + Math.abs(pi - q[i]), 0);
  const residualSum = residual.reduce((a, b) => a + b, 0);
  // The output distribution: accept path plus reject probability times the normalized residual.
  const output = accepted.map((a, i) => a + (residualSum > 0 ? (reject * residual[i]) / residualSum : 0));

  const left = 30;
  const top = 10;
  const bottom = 34;
  const pw = width - left - 4;
  const ph = HEIGHT - top - bottom;
  const y = (v: number) => top + ph - (v / Y_MAX) * ph;
  const colW = pw / p.length;
  const barW = Math.min(colW * 0.46, 70);

  return (
    <Figure
      title="The accept path and the residual add up to the target"
      controls={
        <Segmented
          label="Draft q"
          value={draft}
          options={(Object.keys(DRAFTS) as Draft[]).map((k) => ({ value: k, label: DRAFTS[k].label }))}
          onChange={setDraft}
        />
      }
      readout={
        <>
          <Stat label="Draft q" value={`[${q.map(f2).join(", ")}]`} />
          <Stat label="Acceptance α = Σ min(p, q)" value={f2(alpha)} tone={color.c} />
          <Stat label="Rejection 1 − α" value={f2(reject)} tone={color.d} />
          <Stat label="1 − total variation" value={f2(1 - tv)} />
          <Stat label="Output distribution" value={`[${output.map(f2).join(", ")}]`} />
        </>
      }
      caption={
        <>
          Where the draft over-proposes a token, the accept step trims the excess and keeps min(<i>p</i>, <i>q</i>);
          where it under-proposes, the residual supplies the shortfall (<i>p</i> − <i>q</i>)
          <sub>+</sub>. Pick any draft: the output always equals the target <i>p</i>, and only the acceptance rate
          changes.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={HEIGHT}
          viewBox={`0 0 ${width} ${HEIGHT}`}
          role="img"
          aria-label={`Three tokens with target probabilities 0.5, 0.3, and 0.2. The draft proposes ${q.map(f2).join(", ")}. The accepted parts sum to ${f2(alpha)}, and the residual fills each bar back up to the target.`}
        >
          <YTicks ticks={[0, 0.2, 0.4, 0.6, 0.8]} y={y} x0={left} x1={left + pw} format={(v) => v.toFixed(1)} />
          {p.map((pi, i) => {
            const cx = left + (i + 0.5) * colW;
            const x = cx - barW / 2;
            const qi = q[i];
            return (
              <g key={i}>
                {/* The accept path, then the residual on top of it. */}
                <rect
                  x={x}
                  y={y(accepted[i])}
                  width={barW}
                  height={y(0) - y(accepted[i])}
                  style={{ fill: wash(color.c, 55), stroke: color.c }}
                />
                {residual[i] > 0 && (
                  <rect
                    x={x}
                    y={y(pi)}
                    width={barW}
                    height={y(accepted[i]) - y(pi)}
                    style={{ fill: wash(color.d, 55), stroke: color.d }}
                  />
                )}
                {/* The draft's excess, which the accept step trims. */}
                {qi > pi && (
                  <rect
                    x={x}
                    y={y(qi)}
                    width={barW}
                    height={y(pi) - y(qi)}
                    style={{ fill: "none", stroke: color.subtle, strokeDasharray: "4 3" }}
                  />
                )}
                {/* The target p. */}
                <rect x={x} y={y(pi)} width={barW} height={y(0) - y(pi)} style={{ fill: "none", stroke: color.fg, strokeWidth: 1.5 }} />
                <line x1={x - 5} x2={x + barW + 5} y1={y(qi)} y2={y(qi)} style={{ stroke: color.fg, strokeWidth: 1.5, strokeDasharray: "2 2" }} />
                <text x={x + barW + 7} y={y(qi) + 4} style={{ fill: color.fg }}>
                  q
                </text>
                <text x={cx} y={top + ph + 14} textAnchor="middle" style={{ fill: color.fg, fontWeight: 600 }}>
                  Token {i}
                </text>
                <text x={cx} y={top + ph + 27} textAnchor="middle">
                  p {pi} · q {f2(qi)}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
      <Legend
        items={[
          { label: "Accepted, min(p, q)", tone: color.c },
          { label: "From the residual, (p − q)+", tone: color.d },
          { label: "Draft excess, trimmed", tone: color.subtle, dashed: true },
          { label: "Draft q", tone: color.fg, dashed: true },
        ]}
      />
    </Figure>
  );
}
