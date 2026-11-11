import { useState } from "react";
import { Figure, Legend, Slider, Stat, color, useWidth } from "../kit";

// Relative output error of one synthetic layer, 32 outputs, 64 inputs, 256 tokens,
// four input channels scaled by 12. Seed 18. Each row of the layer drops the same
// share of its weights. Printed by scripts/figures/18b-sparsity-and-pruning.py.
// Not an A100 measurement.
const POINTS = [
  { sparsity: 0.1, magnitude: 0.018724, wanda: 0.007531 },
  { sparsity: 0.2, magnitude: 0.063394, wanda: 0.022159 },
  { sparsity: 0.3, magnitude: 0.129722, wanda: 0.040618 },
  { sparsity: 0.4, magnitude: 0.180384, wanda: 0.065807 },
  { sparsity: 0.5, magnitude: 0.248337, wanda: 0.091117 },
  { sparsity: 0.6, magnitude: 0.316175, wanda: 0.118872 },
  { sparsity: 0.7, magnitude: 0.471724, wanda: 0.159103 },
  { sparsity: 0.8, magnitude: 0.607412, wanda: 0.200234 },
  { sparsity: 0.9, magnitude: 0.765702, wanda: 0.265522 },
] as const;

const SPARSITIES = POINTS.map((point) => point.sparsity);

export default function WandaError() {
  const [sparsity, setSparsity] = useState(0.5);
  const [ref, width] = useWidth();
  const point = POINTS.find((item) => item.sparsity === sparsity) ?? POINTS[4];
  const left = 48;
  const right = 12;
  const top = 16;
  const bottom = 28;
  const height = 220;
  const innerW = Math.max(width - left - right, 40);
  const innerH = height - top - bottom;
  const yMax = 0.8;
  const xOf = (value: number) => left + ((value - 0.1) / 0.8) * innerW;
  const yOf = (value: number) => top + innerH - (value / yMax) * innerH;
  const line = (key: "magnitude" | "wanda") =>
    POINTS.map((item) => `${xOf(item.sparsity).toFixed(1)},${yOf(item[key]).toFixed(1)}`).join(" ");

  return (
    <Figure
      title="Wanda keeps the layer output closer at every sparsity"
      controls={
        <Slider
          label="Sparsity"
          values={SPARSITIES}
          value={sparsity}
          onChange={setSparsity}
          format={(value) => `${Math.round(value * 100)}%`}
        />
      }
      readout={
        <>
          <Stat label="Magnitude" value={point.magnitude.toFixed(6)} tone={color.c} />
          <Stat label="Wanda" value={point.wanda.toFixed(6)} tone={color.b} />
          <Stat label="Ratio" value={(point.wanda / point.magnitude).toFixed(3)} />
        </>
      }
      caption="Relative output error of one layer with 32 outputs, 64 inputs, and 256 tokens. Four input channels are scaled by 12. Each output row drops the same share of its weights. The points are printed by the figure script. They are not an A100 run."
    >
      <div ref={ref}>
        <svg width={width} height={height} role="img" aria-label="Magnitude and Wanda output error against sparsity">
          {[0, 0.2, 0.4, 0.6, 0.8].map((tick) => (
            <g key={tick}>
              <line x1={left} x2={left + innerW} y1={yOf(tick)} y2={yOf(tick)} style={{ stroke: color.line }} />
              <text x={left - 6} y={yOf(tick) + 3} textAnchor="end" fontSize={11} style={{ fill: color.muted }}>
                {tick.toFixed(1)}
              </text>
            </g>
          ))}
          <polyline fill="none" strokeWidth={2} points={line("magnitude")} style={{ stroke: color.c }} />
          <polyline fill="none" strokeWidth={2} points={line("wanda")} style={{ stroke: color.b }} />
          {POINTS.map((item) => (
            <circle
              key={`m-${item.sparsity}`}
              cx={xOf(item.sparsity)}
              cy={yOf(item.magnitude)}
              r={item.sparsity === sparsity ? 4 : 2}
              style={{ fill: color.c }}
            />
          ))}
          {POINTS.map((item) => (
            <circle
              key={`w-${item.sparsity}`}
              cx={xOf(item.sparsity)}
              cy={yOf(item.wanda)}
              r={item.sparsity === sparsity ? 4 : 2}
              style={{ fill: color.b }}
            />
          ))}
          <text x={left + innerW} y={height - 6} textAnchor="end" fontSize={11} style={{ fill: color.muted }}>
            Sparsity
          </text>
        </svg>
        <Legend
          items={[
            { label: "Magnitude", tone: color.c },
            { label: "Wanda", tone: color.b },
          ]}
        />
      </div>
    </Figure>
  );
}
