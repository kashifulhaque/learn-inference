import { useState } from "react";
import { Figure, Slider, Stat, color, useWidth, wash } from "../kit";
import { linear } from "./helpers";

// Chapter 4, "The operation" and "LayerNorm versus RMSNorm": the chapter's
// two-element row x = (3, 4), epsilon = rms_norm_eps = 1e-6, and the gains at
// their initial value of 1 (w = gamma = 1, beta = 0).
const BASE = [3, 4] as const;
const EPS = 1e-6;
const SCALES = [0.25, 0.5, 1, 2, 4] as const;

function rmsNorm(x: readonly number[]) {
  const ms = x.reduce((a, v) => a + v * v, 0) / x.length;
  return { ms, y: x.map((v) => v / Math.sqrt(ms + EPS)) };
}

function layerNorm(x: readonly number[]) {
  const mu = x.reduce((a, v) => a + v, 0) / x.length;
  const variance = x.reduce((a, v) => a + (v - mu) ** 2, 0) / x.length;
  return x.map((v) => (v - mu) / Math.sqrt(variance + EPS));
}

// Input axis: the largest |x| any control setting reaches is 4 x 4 + 4 = 20.
// Both output axes run to 1.8, past the largest |y| of sqrt(2) that a
// two-element row with unit root mean square can have, with room for the
// bar's label under a full negative bar.
const IN_MAX = 20;
const OUT_MAX = 1.8;
const HEIGHT = 190;
const TOP = 30;
const BOTTOM = 26;
const PANEL_GAP = 14;

/** A number to `digits` places, with a true minus sign and no negative zero. */
const fix = (v: number, digits = 2) => (Math.abs(v) < 0.5 * 10 ** -digits ? 0 : v).toFixed(digits).replace("-", "−");
const vec = (v: readonly number[], digits = 3) => `(${v.map((n) => fix(n, digits)).join(", ")})`;

function Panel({
  x0,
  w,
  title,
  values,
  max,
  tone,
}: {
  x0: number;
  w: number;
  title: string;
  values: readonly number[];
  max: number;
  tone: string;
}) {
  const y = linear(-max, max, HEIGHT - BOTTOM, TOP);
  const barW = Math.min(34, (w - 18) / 2);
  const zero = y(0);
  return (
    <g>
      <text x={x0 + w / 2} y={12} textAnchor="middle" style={{ fill: color.fg, fontWeight: 600 }}>
        {title}
      </text>
      <rect x={x0} y={TOP} width={w} height={HEIGHT - TOP - BOTTOM} style={{ fill: color.well }} rx={4} />
      <line x1={x0} x2={x0 + w} y1={zero} y2={zero} style={{ stroke: color.lineStrong }} />
      {values.map((v, i) => {
        const cx = x0 + (w / 2) * (i + 0.5);
        const top = Math.min(y(v), zero);
        const h = Math.abs(y(v) - zero);
        return (
          <g key={i}>
            <rect
              x={cx - barW / 2}
              y={top}
              width={barW}
              height={Math.max(h, 1)}
              style={{ fill: wash(tone, 35), stroke: tone }}
            />
            <text x={cx} y={v >= 0 ? top - 4 : top + h + 12} textAnchor="middle" style={{ fill: color.fg }}>
              {fix(v)}
            </text>
            <text x={cx} y={HEIGHT - 8} textAnchor="middle">
              {`x${i === 0 ? "₁" : "₂"}`}
            </text>
          </g>
        );
      })}
    </g>
  );
}

export default function RmsNormScaleShift() {
  const [ref, width] = useWidth();
  const [scale, setScale] = useState<number>(1);
  const [shift, setShift] = useState(0);

  const x = BASE.map((v) => scale * v + shift);
  const { ms, y } = rmsNorm(x);
  const ln = layerNorm(x);
  const panelW = (width - 2 * PANEL_GAP) / 3;

  return (
    <Figure
      title="RMSNorm ignores scale but not shift"
      controls={
        <>
          <Slider label="Scale c" values={SCALES} value={scale} onChange={setScale} format={(v) => `${v}`} />
          <Slider
            label="Shift"
            min={-4}
            max={4}
            step={0.5}
            value={shift}
            onChange={setShift}
            format={(v) => `${v >= 0 ? "+" : "−"}${Math.abs(v)}`}
          />
        </>
      }
      readout={
        <>
          <Stat label="x = c · (3, 4) + shift" value={vec(x, 2)} />
          <Stat label="ms(x)" value={fix(ms, 3)} tone={color.c} />
          <Stat label="RMSNorm y" value={vec(y)} tone={color.accent} />
          <Stat label="LayerNorm y" value={vec(ln)} tone={color.violet} />
        </>
      }
      caption={
        <>
          Scaling <em>x</em> leaves both outputs where they are. Shifting every channel by the same constant moves the
          RMSNorm output, because nothing subtracts the mean; LayerNorm erases the shift. With the gain at 1, the
          RMSNorm output always has a root mean square of 1.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={HEIGHT}
          role="img"
          aria-label={`Bar charts of the input ${vec(x)}, its RMSNorm output ${vec(y)}, and its LayerNorm output ${vec(ln)}.`}
        >
          <Panel x0={0} w={panelW} title="Input x" values={x} max={IN_MAX} tone={color.muted} />
          <Panel x0={panelW + PANEL_GAP} w={panelW} title="RMSNorm" values={y} max={OUT_MAX} tone={color.accent} />
          <Panel x0={2 * (panelW + PANEL_GAP)} w={panelW} title="LayerNorm" values={ln} max={OUT_MAX} tone={color.violet} />
        </svg>
      </div>
    </Figure>
  );
}
