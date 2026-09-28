// Chapter 5: each rotary pair is a clock hand that turns at its own speed. A
// query at m and a key at n sit at angles mθ_i and nθ_i, and only the angle
// between them, (m − n)θ_i, reaches the score.

import { useState } from "react";
import { Figure, Legend, Segmented, Slider, Stat, color, formatCount, useWidth, wash } from "../kit";

// head_dim 256 × partial_rotary_factor 0.25 = rotary_dim 64, so 32 pairs with
// θ_i = Θ^(−2i/d) = Θ^(−i/32) (chapter 5, "The frequencies").
const ROTARY_DIM = 64;
// The two values of rope_theta that chapter 5's wavelength table compares.
const THETAS = [1e4, 1e7] as const;
// Key positions and gaps to try. A key at 10 and a query at 15 is the
// chapter's clock-hand example, 1010 and 1015 its shifted copy, and a gap of
// 47,000 is its "position 1,000 and position 48,000" case at Θ = 10^4.
const KEY_POSITIONS = [0, 1, 2, 5, 10, 20, 50, 100, 1000, 1010, 10000, 100000];
const GAPS = [1, 5, 50, 1000, 47000] as const;
// Which of the 32 pairs to draw: every eighth, plus the slowest.
const WIDE_PAIRS = [0, 8, 16, 24, 31];
const NARROW_PAIRS = [0, 16, 31];

const HEIGHT = 136;
const MAX_RADIUS = 30;
const TAU = 2 * Math.PI;

function frequency(theta: number, pair: number): number {
  return theta ** (-(2 * pair) / ROTARY_DIM);
}

/** A wavelength in positions: 6.3, 628, 47.1k, 38.0M. */
function formatWavelength(value: number): string {
  if (value < 100) return value.toFixed(1);
  if (value < 1000) return value.toFixed(0);
  if (value < 1e6) return `${(value / 1e3).toFixed(1)}k`;
  return `${(value / 1e6).toFixed(1)}M`;
}

function formatSmall(value: number, bigDigits: number): string {
  if (value >= 10) return value.toFixed(bigDigits);
  if (value >= 1) return value.toFixed(1);
  if (value >= 0.01) return value.toFixed(2);
  return "<0.01";
}

function formatTurns(turns: number): string {
  if (turns >= 1000) return formatCount(Math.round(turns));
  return formatSmall(turns, 1);
}

function thetaLabel(theta: number): string {
  return theta === 1e4 ? "10⁴" : "10⁷";
}

function Dial({ cx, cy, r, pair, theta, m, n }: {
  cx: number; cy: number; r: number; pair: number; theta: number; m: number; n: number;
}) {
  const f = frequency(theta, pair);
  const keyAngle = (n * f) % TAU;
  const queryAngle = (m * f) % TAU;
  const gapAngle = ((m - n) * f) % TAU;
  const turns = ((m - n) * f) / TAU;
  // SVG's y axis points down, so a positive angle is drawn counterclockwise.
  const point = (radius: number, angle: number) =>
    [cx + radius * Math.cos(angle), cy - radius * Math.sin(angle)] as const;
  const [kx, ky] = point(r - 3, keyAngle);
  const [qx, qy] = point(r - 3, queryAngle);
  const wedge = r * 0.55;
  const [sx, sy] = point(wedge, keyAngle);
  const [ex, ey] = point(wedge, keyAngle + gapAngle);
  const large = gapAngle > Math.PI ? 1 : 0;
  const degrees = (gapAngle * 180) / Math.PI;
  const [zx, zy] = point(r, 0);
  const [zx2, zy2] = point(r - 5, 0);

  return (
    <g>
      <circle cx={cx} cy={cy} r={r} style={{ fill: color.well, stroke: color.lineStrong }} />
      <line x1={zx2} y1={zy2} x2={zx} y2={zy} style={{ stroke: color.faint, strokeWidth: 1.5 }} />
      {gapAngle > 1e-3 && (
        <path
          d={`M ${cx} ${cy} L ${sx} ${sy} A ${wedge} ${wedge} 0 ${large} 0 ${ex} ${ey} Z`}
          style={{ fill: wash(color.c, 30), stroke: color.c, strokeWidth: 1 }}
        />
      )}
      <line x1={cx} y1={cy} x2={kx} y2={ky} style={{ stroke: color.b, strokeWidth: 2.5, strokeLinecap: "round" }} />
      <line x1={cx} y1={cy} x2={qx} y2={qy} style={{ stroke: color.a, strokeWidth: 2.5, strokeLinecap: "round" }} />
      <circle cx={cx} cy={cy} r={2.5} style={{ fill: color.fg }} />
      <text x={cx} y={cy + MAX_RADIUS + 18} textAnchor="middle" style={{ fill: color.fg }}>
        pair {pair}
      </text>
      <text x={cx} y={cy + MAX_RADIUS + 32} textAnchor="middle">
        λ {formatWavelength(TAU / f)}
      </text>
      <text x={cx} y={cy + MAX_RADIUS + 46} textAnchor="middle" style={{ fill: color.c }}>
        gap {formatSmall(degrees, 0)}°
      </text>
      <text x={cx} y={cy + MAX_RADIUS + 60} textAnchor="middle">
        {formatTurns(turns)} turns
      </text>
    </g>
  );
}

export default function RotaryDials() {
  const [ref, width] = useWidth();
  const [theta, setTheta] = useState<number>(1e7);
  const [n, setN] = useState(10);
  const [gap, setGap] = useState<number>(5);
  const m = n + gap;

  const pairs = width >= 440 ? WIDE_PAIRS : NARROW_PAIRS;
  const column = width / pairs.length;
  const radius = Math.max(18, Math.min(MAX_RADIUS, column / 2 - 8));
  const cy = 6 + MAX_RADIUS;
  const slowest = TAU / frequency(theta, ROTARY_DIM / 2 - 1);

  return (
    <Figure
      title="Rotary pairs turn at different speeds"
      controls={
        <>
          <Slider label="Key position n" values={KEY_POSITIONS} value={n} onChange={setN} />
          <Segmented
            label="Gap m − n"
            value={gap}
            options={GAPS.map((value) => ({ value, label: formatCount(value) }))}
            onChange={setGap}
          />
          <Segmented
            label="rope_theta Θ"
            value={theta}
            options={THETAS.map((value) => ({ value, label: thetaLabel(value) }))}
            onChange={setTheta}
          />
        </>
      }
      readout={
        <>
          <Stat label="Query position m" value={formatCount(m)} tone={color.a} />
          <Stat label="Key position n" value={formatCount(n)} tone={color.b} />
          <Stat label="Gap m − n" value={formatCount(gap)} tone={color.c} />
          <Stat label="Θ" value={thetaLabel(theta)} />
          <Stat label="Slowest wavelength λ₃₁" value={`${formatWavelength(slowest)} positions`} />
        </>
      }
      caption={
        <>
          Move the key position and both hands turn together, but the angle between them, which is all the
          score sees, stays put. Set the gap to 47,000 at Θ = 10⁴ and the slowest pair comes back almost to
          where it started.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={HEIGHT}
          role="img"
          aria-label={`Clock dials for rotary pairs ${pairs.join(", ")} at rope_theta ${thetaLabel(theta)}. A key at position ${n} and a query at position ${m} point at their angles, and a wedge shows the angle the gap of ${gap} adds in each pair.`}
        >
          {pairs.map((pair, index) => (
            <Dial
              key={pair}
              cx={column * (index + 0.5)}
              cy={cy}
              r={radius}
              pair={pair}
              theta={theta}
              m={m}
              n={n}
            />
          ))}
        </svg>
        <Legend
          items={[
            { label: "Query at m", tone: color.a },
            { label: "Key at n", tone: color.b },
            { label: "Angle the gap adds", tone: color.c },
          ]}
        />
      </div>
    </Figure>
  );
}
