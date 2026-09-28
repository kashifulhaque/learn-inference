// Chapter 6: a head's forget gate as a time constant. With the rate held at
// r = e^{A_log}, whatever the head wrote n tokens ago survives as αⁿ = e^{−rn}.

import { useState } from "react";
import { Figure, Slider, Stat, color, useWidth } from "../kit";

// Chapter 6, "Reading a head's time constant": r = softplus(a_t + b_Δ) · e^{A_log}
// and α = e^{−r}. Its table holds softplus at 1, so r = e^{A_log}; the faint
// curves are the table's rows.
const TABLE_A_LOG = [2, 0, -2, -4, -6, -8];
const A_LOG_MIN = -8;
const A_LOG_MAX = 2;
// The plotted range of n, in tokens since the write.
const DECADES = 4;

const HEIGHT = 210;
const MARGIN = { left: 34, right: 12, top: 18, bottom: 34 };
const SAMPLES = 160;

function minus(text: string): string {
  return text.replace("-", "−");
}

/** A horizon in tokens, as chapter 6's table rounds it: 0.09, 5.1, 38, 2,066. */
function formatTokens(n: number): string {
  if (n >= 10) return Math.round(n).toLocaleString("en-US");
  if (n >= 1) return n.toFixed(1);
  return n.toFixed(2);
}

function formatGate(alpha: number): string {
  return alpha > 0.99 ? alpha.toFixed(5) : alpha.toPrecision(4);
}

export default function ForgetHorizon() {
  const [ref, width] = useWidth();
  const [aLog, setALog] = useState(-4);

  const rate = Math.exp(aLog);
  const alpha = Math.exp(-rate);
  const half = Math.LN2 / rate;
  const ninety = Math.LN10 / rate;

  const plotW = width - MARGIN.left - MARGIN.right;
  const plotH = HEIGHT - MARGIN.top - MARGIN.bottom;
  const x = (n: number) => MARGIN.left + (Math.log10(n) / DECADES) * plotW;
  const y = (share: number) => MARGIN.top + (1 - share) * plotH;
  const curve = (r: number) =>
    Array.from({ length: SAMPLES + 1 }, (_, i) => {
      const n = 10 ** ((i / SAMPLES) * DECADES);
      return `${i === 0 ? "M" : "L"} ${x(n).toFixed(1)} ${y(Math.exp(-r * n)).toFixed(1)}`;
    }).join(" ");
  const inRange = (n: number) => n >= 1 && n <= 10 ** DECADES;

  return (
    <Figure
      title="How long a head remembers"
      controls={
        <Slider
          label="A_log"
          min={A_LOG_MIN}
          max={A_LOG_MAX}
          step={0.5}
          value={aLog}
          onChange={setALog}
          format={(v) => minus(v.toFixed(1))}
        />
      }
      readout={
        <>
          <Stat label="A_log" value={minus(aLog.toFixed(1))} />
          <Stat label="Rate r = e^A_log" value={rate.toPrecision(4)} />
          <Stat label="Gate α = e^−r" value={formatGate(alpha)} tone={color.a} />
          <Stat label="Half left, n½" value={`${formatTokens(half)} tokens`} />
          <Stat label="90% forgotten, n₉₀" value={`${formatTokens(ninety)} tokens`} />
        </>
      }
      caption={
        <>
          Each curve is the share of one write still in the state n tokens later, with softplus held at 1 as in
          the table; the faint curves are the table's rows. Two units of A_log move the curve by a factor of
          e² ≈ 7.4 in tokens, so one layer's heads can span a few tokens to several thousand.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={HEIGHT}
          role="img"
          aria-label={`Decay curves of αⁿ against tokens on a log scale from 1 to 10,000. At A_log ${minus(aLog.toFixed(1))}, α is ${formatGate(alpha)}, half the write is left after ${formatTokens(half)} tokens and 90% is forgotten after ${formatTokens(ninety)}.`}
        >
          {[0, 0.1, 0.5, 1].map((share) => (
            <g key={share}>
              <line
                x1={MARGIN.left}
                x2={MARGIN.left + plotW}
                y1={y(share)}
                y2={y(share)}
                style={{
                  stroke: share === 0 ? color.lineStrong : color.line,
                  strokeDasharray: share === 0.1 || share === 0.5 ? "3 3" : undefined,
                }}
              />
              <text x={MARGIN.left - 5} y={y(share) + 4} textAnchor="end">
                {share}
              </text>
            </g>
          ))}
          {Array.from({ length: DECADES + 1 }, (_, d) => 10 ** d).map((n) => (
            <g key={n}>
              <line x1={x(n)} x2={x(n)} y1={y(0)} y2={y(0) + 4} style={{ stroke: color.lineStrong }} />
              <text x={x(n)} y={y(0) + 16} textAnchor={n === 1 ? "start" : n === 10 ** DECADES ? "end" : "middle"}>
                {n.toLocaleString("en-US")}
              </text>
            </g>
          ))}
          <text x={MARGIN.left + plotW} y={HEIGHT - 2} textAnchor="end">
            tokens since the write, n
          </text>
          <text x={MARGIN.left} y={10}>
            share of the write left, αⁿ
          </text>

          {TABLE_A_LOG.map((value) => {
            const r = Math.exp(value);
            const n = Math.LN2 / r;
            return (
              <g key={value}>
                <path d={curve(r)} style={{ fill: "none", stroke: color.faint, strokeWidth: 1 }} />
                {inRange(n) && value !== aLog && (
                  <text x={x(n) + 4} y={y(0.5) - 5} style={{ fill: color.faint }}>
                    {minus(String(value))}
                  </text>
                )}
              </g>
            );
          })}

          <path d={curve(rate)} style={{ fill: "none", stroke: color.a, strokeWidth: 2.5 }} />
          {[
            { n: half, share: 0.5 },
            { n: ninety, share: 0.1 },
          ]
            .filter((marker) => inRange(marker.n))
            .map((marker) => (
              <g key={marker.share}>
                <line
                  x1={x(marker.n)}
                  x2={x(marker.n)}
                  y1={y(marker.share)}
                  y2={y(0)}
                  style={{ stroke: color.a, strokeDasharray: "2 3" }}
                />
                <circle cx={x(marker.n)} cy={y(marker.share)} r={3.5} style={{ fill: color.a }} />
              </g>
            ))}
        </svg>
      </div>
    </Figure>
  );
}
