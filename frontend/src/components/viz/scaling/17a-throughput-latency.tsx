import { useState } from "react";
import { Figure, Slider, Stat, color, useWidth } from "../kit";

// Decode throughput at context 1024, one A100, degree 1. Each point is
// batch / decode_step_time from scripts/figures/17a-performance-model.py.
// These are model floors, not a device measurement. The last batch is the
// largest that fits in 80 GiB beside the weights and the recurrent state.
const POINTS = [
  { batch: 1, stepMs: 29.9963, tok: 33.34 },
  { batch: 2, stepMs: 30.1932, tok: 66.24 },
  { batch: 4, stepMs: 30.5869, tok: 130.78 },
  { batch: 8, stepMs: 31.3743, tok: 254.99 },
  { batch: 16, stepMs: 32.9492, tok: 485.6 },
  { batch: 32, stepMs: 36.0988, tok: 886.45 },
  { batch: 48, stepMs: 39.2485, tok: 1222.98 },
  { batch: 64, stepMs: 42.3982, tok: 1509.5 },
  { batch: 96, stepMs: 48.6976, tok: 1971.35 },
  { batch: 128, stepMs: 54.997, tok: 2327.4 },
  { batch: 144, stepMs: 58.1467, tok: 2476.5 },
] as const;

const BATCHES = POINTS.map((point) => point.batch);

export default function ThroughputLatency() {
  const [batch, setBatch] = useState(144);
  const [ref, width] = useWidth();
  const point = POINTS.find((item) => item.batch === batch) ?? POINTS[POINTS.length - 1];
  const left = 44;
  const right = 12;
  const top = 12;
  const bottom = 28;
  const height = 220;
  const innerW = Math.max(width - left - right, 40);
  const innerH = height - top - bottom;
  const maxBatch = POINTS[POINTS.length - 1].batch;
  const maxTok = 2600;
  const xOf = (value: number) => left + (value / maxBatch) * innerW;
  const yOf = (value: number) => top + innerH - (value / maxTok) * innerH;
  const line = POINTS.map((item) => `${xOf(item.batch).toFixed(1)},${yOf(item.tok).toFixed(1)}`).join(" ");
  const yTicks = [0, 500, 1000, 1500, 2000, 2500];

  return (
    <Figure
      title="Throughput rises until memory runs out"
      controls={
        <Slider
          label="Batch"
          values={BATCHES}
          value={batch}
          onChange={setBatch}
          format={(value) => `${value} sequences`}
        />
      }
      readout={
        <>
          <Stat label="Step" value={`${point.stepMs.toFixed(1)} ms`} />
          <Stat label="Tokens/s" value={point.tok.toFixed(0)} tone={color.b} />
          <Stat label="Bound" value="Memory" />
        </>
      }
      caption="Context 1,024, one GPU, degree 1. Each point is the model's floor: the batch divided by the step time. The last point is the largest batch that fits. This is not a measured curve."
    >
      <div ref={ref}>
        <svg
          width={width}
          height={height}
          role="img"
          aria-label="Decode throughput against batch at context 1024, from the step-time model"
        >
          {yTicks.map((tick) => (
            <g key={tick}>
              <line
                x1={left}
                x2={left + innerW}
                y1={yOf(tick)}
                y2={yOf(tick)}
                stroke={color.line}
              />
              <text x={left - 6} y={yOf(tick) + 4} textAnchor="end" fill={color.muted} fontSize={11}>
                {tick === 0 ? "0" : `${tick / 1000}k`}
              </text>
            </g>
          ))}
          <polyline fill="none" stroke={color.b} strokeWidth={2} points={line} />
          {POINTS.map((item) => (
            <circle
              key={item.batch}
              cx={xOf(item.batch)}
              cy={yOf(item.tok)}
              r={item.batch === point.batch ? 4.5 : 2.5}
              fill={item.batch === point.batch ? color.a : color.b}
            />
          ))}
          <text x={left + innerW} y={height - 6} textAnchor="end" fill={color.muted} fontSize={11}>
            Batch
          </text>
        </svg>
      </div>
    </Figure>
  );
}
