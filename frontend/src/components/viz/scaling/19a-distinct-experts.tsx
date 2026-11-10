import { useState } from "react";
import { Figure, Legend, Slider, Stat, color, useWidth } from "../kit";

// Qwen3-30B-A3B decode, uniform routing, expert MLPs only for the intensity.
// 128 experts, top-8. Weight bytes add the 3.08 GB of non-expert weights.
// Step time reads those bytes at the measured 1275 GB/s copy rate.
// Printed by scripts/figures/19a-moe-and-expert-parallelism.py from the lab
// solution. Computed, not an A100 measurement.
const POINTS = [
  { batch: 1, distinct: 8.0, intensity: 1.0, weightGB: 6.71, stepMs: 5.26 },
  { batch: 2, distinct: 15.5, intensity: 1.032, weightGB: 10.1, stepMs: 7.92 },
  { batch: 4, distinct: 29.12, intensity: 1.099, weightGB: 16.27, stepMs: 12.76 },
  { batch: 8, distinct: 51.62, intensity: 1.24, weightGB: 26.47, stepMs: 20.76 },
  { batch: 16, distinct: 82.42, intensity: 1.553, weightGB: 40.42, stepMs: 31.7 },
  { batch: 32, distinct: 111.77, intensity: 2.29, weightGB: 53.71, stepMs: 42.13 },
  { batch: 64, distinct: 125.94, intensity: 4.065, weightGB: 60.13, stepMs: 47.16 },
  { batch: 128, distinct: 127.97, intensity: 8.002, weightGB: 61.05, stepMs: 47.88 },
  { batch: 256, distinct: 128.0, intensity: 16.0, weightGB: 61.06, stepMs: 47.89 },
  { batch: 512, distinct: 128.0, intensity: 32.0, weightGB: 61.06, stepMs: 47.89 },
  { batch: 1024, distinct: 128.0, intensity: 64.0, weightGB: 61.06, stepMs: 47.89 },
  { batch: 2580, distinct: 128.0, intensity: 161.25, weightGB: 61.06, stepMs: 47.89 },
  { batch: 4096, distinct: 128.0, intensity: 256.0, weightGB: 61.06, stepMs: 47.89 },
] as const;

const RIDGE = 161.24; // 312 TFLOP/s over 1935 GB/s
const BATCHES = POINTS.map((p) => p.batch);

export default function DistinctExperts() {
  const [batch, setBatch] = useState(32);
  const [ref, width] = useWidth();
  const point = POINTS.find((p) => p.batch === batch) ?? POINTS[5];
  const left = 44;
  const right = 12;
  const top = 14;
  const bottom = 30;
  const height = 230;
  const innerW = Math.max(width - left - right, 40);
  const innerH = height - top - bottom;
  const xMin = Math.log10(1);
  const xMax = Math.log10(4096);
  const yMin = Math.log10(0.5);
  const yMax = Math.log10(400);
  const xOf = (v: number) => left + ((Math.log10(v) - xMin) / (xMax - xMin)) * innerW;
  const yOf = (v: number) => top + innerH - ((Math.log10(v) - yMin) / (yMax - yMin)) * innerH;
  const moe = POINTS.map((p) => `${xOf(p.batch).toFixed(1)},${yOf(p.intensity).toFixed(1)}`).join(" ");
  const dense = `${xOf(1).toFixed(1)},${yOf(1).toFixed(1)} ${xOf(4096).toFixed(1)},${yOf(4096).toFixed(1)}`;
  const yTicks = [1, 10, 100];
  const xTicks = [1, 8, 64, 512, 4096];

  return (
    <Figure
      title="MoE decode reaches the ridge 16 times later"
      controls={
        <Slider label="Decode batch" values={BATCHES} value={batch} onChange={setBatch} />
      }
      readout={
        <>
          <Stat label="Experts touched" value={`${point.distinct.toFixed(1)} of 128`} tone={color.b} />
          <Stat label="Weights read" value={`${point.weightGB.toFixed(1)} GB`} tone={color.c} />
          <Stat label="Step floor" value={`${point.stepMs.toFixed(1)} ms`} />
          <Stat label="Expert FLOPs per byte" value={point.intensity.toFixed(2)} tone={color.a} />
        </>
      }
      caption="Qwen3-30B-A3B, 128 experts, top-8, uniform routing. The MoE line is the expert MLPs' FLOPs per byte against the batch. The dense line has the same active width, so its intensity equals the batch. The ridge is 161.2 FLOPs per byte. Computed from the lab solution, not measured on an A100."
    >
      <div ref={ref}>
        <svg width={width} height={height} role="img" aria-label="Expert MLP intensity against decode batch for a mixture-of-experts layer and a dense layer, with the roofline ridge">
          {yTicks.map((t) => (
            <g key={t}>
              <line x1={left} x2={left + innerW} y1={yOf(t)} y2={yOf(t)} style={{ stroke: color.line }} />
              <text x={left - 6} y={yOf(t) + 3} textAnchor="end" fontSize={11} style={{ fill: color.muted }}>
                {t}
              </text>
            </g>
          ))}
          {xTicks.map((t) => (
            <text key={t} x={xOf(t)} y={height - 12} textAnchor="middle" fontSize={11} style={{ fill: color.muted }}>
              {t}
            </text>
          ))}
          <line x1={left} x2={left + innerW} y1={yOf(RIDGE)} y2={yOf(RIDGE)} strokeDasharray="4 3" style={{ stroke: color.bad }} />
          <text x={left + innerW} y={yOf(RIDGE) - 4} textAnchor="end" fontSize={11} style={{ fill: color.bad }}>
            ridge 161.2
          </text>
          <polyline fill="none" strokeWidth={2} points={dense} style={{ stroke: color.b }} />
          <polyline fill="none" strokeWidth={2} points={moe} style={{ stroke: color.c }} />
          <line x1={xOf(batch)} x2={xOf(batch)} y1={top} y2={top + innerH} style={{ stroke: color.lineStrong }} />
          <circle cx={xOf(batch)} cy={yOf(point.intensity)} r={4} style={{ fill: color.c }} />
          <circle cx={xOf(batch)} cy={yOf(batch)} r={4} style={{ fill: color.b }} />
          <text x={left + innerW} y={height - 1} textAnchor="end" fontSize={11} style={{ fill: color.muted }}>
            Decode batch
          </text>
        </svg>
        <Legend
          items={[
            { label: "MoE, 128 experts, top-8", tone: color.c },
            { label: "Dense, same active width", tone: color.b },
          ]}
        />
      </div>
    </Figure>
  );
}
