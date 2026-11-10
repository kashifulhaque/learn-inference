import { useState } from "react";
import { Figure, Legend, Slider, Stat, color, useWidth } from "../kit";

// Relative output error of one synthetic layer, 128 rows, 32 inputs, 4 outputs,
// group size 8. Seed 18. Printed by scripts/figures/18a-int4-and-kv-quantization.py.
// Not an A100 measurement.
const POINTS = [
  { rho: 0, rtn: 0.081093, gptq: 0.075025 },
  { rho: 0.25, rtn: 0.07102, gptq: 0.062835 },
  { rho: 0.5, rtn: 0.060018, gptq: 0.047501 },
  { rho: 0.75, rtn: 0.049697, gptq: 0.032426 },
  { rho: 0.9, rtn: 0.043692, gptq: 0.02028 },
  { rho: 0.95, rtn: 0.041674, gptq: 0.015605 },
  { rho: 0.99, rtn: 0.039994, gptq: 0.011045 },
] as const;

const RHOS = POINTS.map((point) => point.rho);

export default function GptqMargin() {
  const [rho, setRho] = useState(0.95);
  const [ref, width] = useWidth();
  const point = POINTS.find((item) => item.rho === rho) ?? POINTS[5];
  const left = 48;
  const right = 12;
  const top = 16;
  const bottom = 28;
  const height = 220;
  const innerW = Math.max(width - left - right, 40);
  const innerH = height - top - bottom;
  const yMax = 0.1;
  const xOf = (value: number) => left + value * innerW;
  const yOf = (value: number) => top + innerH - (value / yMax) * innerH;
  const line = (key: "rtn" | "gptq") =>
    POINTS.map((item) => `${xOf(item.rho).toFixed(1)},${yOf(item[key]).toFixed(1)}`).join(" ");
  const margin = point.rtn - point.gptq;

  return (
    <Figure
      title="The GPTQ margin grows with correlation"
      controls={
        <Slider
          label="Input correlation"
          values={RHOS}
          value={rho}
          onChange={setRho}
          format={(value) => value.toFixed(2)}
        />
      }
      readout={
        <>
          <Stat label="Round-to-nearest" value={point.rtn.toFixed(6)} tone={color.c} />
          <Stat label="GPTQ" value={point.gptq.toFixed(6)} tone={color.b} />
          <Stat label="Margin" value={margin.toFixed(6)} />
        </>
      }
      caption="Relative output error on a layer of 128 rows, 32 inputs, and 4 outputs, group size 8, seed 18. Correlation mixes one shared column into the inputs. The points are printed by the figure script. They are not an A100 run."
    >
      <div ref={ref}>
        <svg width={width} height={height} role="img" aria-label="GPTQ and round-to-nearest output error against input correlation">
          {[0, 0.02, 0.04, 0.06, 0.08, 0.1].map((tick) => (
            <g key={tick}>
              <line x1={left} x2={left + innerW} y1={yOf(tick)} y2={yOf(tick)} style={{ stroke: color.line }} />
              <text x={left - 6} y={yOf(tick) + 3} textAnchor="end" fontSize={11} style={{ fill: color.muted }}>
                {tick.toFixed(2)}
              </text>
            </g>
          ))}
          <polyline fill="none" strokeWidth={2} points={line("rtn")} style={{ stroke: color.c }} />
          <polyline fill="none" strokeWidth={2} points={line("gptq")} style={{ stroke: color.b }} />
          {POINTS.map((item) => (
            <circle
              key={`rtn-${item.rho}`}
              cx={xOf(item.rho)}
              cy={yOf(item.rtn)}
              r={item.rho === rho ? 4 : 2}
              style={{ fill: color.c }}
            />
          ))}
          {POINTS.map((item) => (
            <circle
              key={`gptq-${item.rho}`}
              cx={xOf(item.rho)}
              cy={yOf(item.gptq)}
              r={item.rho === rho ? 4 : 2}
              style={{ fill: color.b }}
            />
          ))}
          <text x={left + innerW} y={height - 6} textAnchor="end" fontSize={11} style={{ fill: color.muted }}>
            Correlation
          </text>
        </svg>
        <Legend
          items={[
            { label: "Round-to-nearest", tone: color.c },
            { label: "GPTQ", tone: color.b },
          ]}
        />
      </div>
    </Figure>
  );
}
