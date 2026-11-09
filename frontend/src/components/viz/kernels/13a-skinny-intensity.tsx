import { useState } from "react";
import { Figure, Legend, Slider, Stat, color, useWidth } from "../kit";

// Up projection from chapter 10: K = 5120, N = 17408.
// Rated peaks, also from chapter 10: 312 TFLOP/s and 1935 GB/s.
// The same arithmetic is printed by scripts/figures/13a-skinny-decode-gemm.py.
// A time here is bytes / 1935e9. It is not a measured kernel.
const K = 5120;
const N = 17408;
const GROUP = 128;
const RIDGE = (312 * 1000) / 1935;
const BATCHES = [1, 2, 4, 8, 16, 32, 43, 44, 64, 128, 168, 169, 256, 512, 1024, 2048, 4096] as const;

function weightBytes(w4: boolean): number {
  if (!w4) return 2 * K * N;
  return Math.floor((K * N) / 2) + Math.ceil(K / GROUP) * N * 2;
}

function totalBytes(m: number, w4: boolean): number {
  return 2 * m * K + weightBytes(w4) + 2 * m * N;
}

function flops(m: number): number {
  return 2 * m * K * N;
}

function intensity(m: number, w4: boolean): number {
  return flops(m) / totalBytes(m, w4);
}

function boundOf(m: number, w4: boolean): "memory" | "compute" {
  // flops / bytes ? 312e12 / 1935e9, with the 1e3 folded into 312_000.
  const left = flops(m) * 1935;
  const right = totalBytes(m, w4) * 312_000;
  return left < right ? "memory" : "compute";
}

function formatIntensity(value: number): string {
  if (value < 10) return value.toFixed(3);
  if (value < 100) return value.toFixed(2);
  return value.toFixed(1);
}

export default function SkinnyIntensity() {
  const [batch, setBatch] = useState(1);
  const [ref, width] = useWidth();
  const left = 44;
  const right = 12;
  const top = 16;
  const bottom = 28;
  const height = 240;
  const innerW = Math.max(width - left - right, 40);
  const innerH = height - top - bottom;
  const yMin = Math.log10(0.5);
  const yMax = Math.log10(4000);
  const xOf = (m: number) => left + (Math.log2(m) / 12) * innerW;
  const yOf = (value: number) => top + innerH * (1 - (Math.log10(value) - yMin) / (yMax - yMin));
  const line = (w4: boolean) =>
    BATCHES.map((m) => `${xOf(m).toFixed(1)},${yOf(intensity(m, w4)).toFixed(1)}`).join(" ");
  const bf16 = intensity(batch, false);
  const w4 = intensity(batch, true);
  const memoryUs = (w4Weight: boolean) => (totalBytes(batch, w4Weight) / 1935e9) * 1e6;

  return (
    <Figure
      title="A short batch stays under the ridge"
      controls={
        <Slider
          label="Batch M"
          values={BATCHES}
          value={batch}
          onChange={setBatch}
          format={(value) => `${value.toLocaleString("en-US")} rows`}
        />
      }
      readout={
        <>
          <Stat label="BF16 intensity" value={formatIntensity(bf16)} tone={color.c} />
          <Stat label="BF16 bound" value={boundOf(batch, false)} />
          <Stat label="Int4 intensity" value={formatIntensity(w4)} tone={color.b} />
          <Stat label="Int4 bound" value={boundOf(batch, true)} />
          <Stat label="BF16 floor, µs" value={memoryUs(false).toFixed(2)} />
          <Stat label="Int4 floor, µs" value={memoryUs(true).toFixed(2)} />
        </>
      }
      caption="Up projection, K = 5120 and N = 17408. Intensity is 2MNK divided by the bytes of A, the weight, and C. The dashed line is the rated ridge, 161.2 FLOPs per byte. The microsecond figures are those bytes divided by 1935 GB/s. They are not a measured kernel."
    >
      <div ref={ref}>
        <svg width={width} height={height} role="img" aria-label="Arithmetic intensity of one up projection against batch size">
          {[0.5, 1, 10, 100, RIDGE, 1000].map((tick) => (
            <g key={tick}>
              <line
                x1={left}
                x2={left + innerW}
                y1={yOf(tick)}
                y2={yOf(tick)}
                style={{ stroke: tick === RIDGE ? color.a : color.line }}
                strokeDasharray={tick === RIDGE ? "4 3" : undefined}
              />
              <text x={left - 6} y={yOf(tick) + 3} textAnchor="end" fontSize={11} style={{ fill: color.muted }}>
                {tick === RIDGE ? "161" : tick}
              </text>
            </g>
          ))}
          <polyline fill="none" strokeWidth={2} points={line(false)} style={{ stroke: color.c }} />
          <polyline fill="none" strokeWidth={2} points={line(true)} style={{ stroke: color.b }} />
          {BATCHES.map((m) => (
            <circle
              key={m}
              cx={xOf(m)}
              cy={yOf(intensity(m, false))}
              r={m === batch ? 4 : 2}
              style={{ fill: color.c }}
            />
          ))}
          <circle cx={xOf(batch)} cy={yOf(w4)} r={4} style={{ fill: color.b }} />
          <text x={left + innerW} y={height - 6} textAnchor="end" fontSize={11} style={{ fill: color.muted }}>
            Batch M, log scale
          </text>
        </svg>
        <Legend
          items={[
            { label: "Bfloat16", tone: color.c },
            { label: "Int4 weight", tone: color.b },
            { label: "Ridge 161.2", tone: color.a, dashed: true },
          ]}
        />
      </div>
    </Figure>
  );
}
