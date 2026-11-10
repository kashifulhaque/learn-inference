import { useState } from "react";
import { Figure, Slider, Stat, color, useWidth } from "../kit";

// The planted trace from scripts/figures/17b-disaggregated-serving.py: eight
// decoders and one 8192-token prompt at t = 2 s, one colocated GPU. Each point
// is a prefill chunk size. Values come from the lab solution's simulator on a
// simulated clock, with chapter 17a's step-time floor. Not a measurement.
const POINTS = [
  { chunk: 128, gapMs: 32.02, ttft: 2.063 },
  { chunk: 160, gapMs: 32.79, ttft: 1.699 },
  { chunk: 192, gapMs: 38.64, ttft: 1.631 },
  { chunk: 512, gapMs: 97.37, ttft: 1.538 },
  { chunk: 2048, gapMs: 377.34, ttft: 1.495 },
  { chunk: 16384, gapMs: 1467.48, ttft: 1.485 },
] as const;

// One prefill GPU and one decode GPU, a 300 GB/s link, the same trace.
const DISAGG = { gapMs: 31.6, ttft: 1.468 };

const CHUNKS = POINTS.map((point) => point.chunk);

export default function ChunkVsDisagg() {
  const [chunk, setChunk] = useState(512);
  const [ref, width] = useWidth();
  const point = POINTS.find((item) => item.chunk === chunk) ?? POINTS[3];
  const left = 52;
  const right = 14;
  const top = 12;
  const bottom = 34;
  const height = 240;
  const innerW = Math.max(width - left - right, 40);
  const innerH = height - top - bottom;
  const tMin = 1.4;
  const tMax = 2.15;
  const gMin = 25;
  const gMax = 2000;
  const xOf = (value: number) => left + ((value - tMin) / (tMax - tMin)) * innerW;
  const yOf = (value: number) =>
    top + innerH - ((Math.log(value) - Math.log(gMin)) / (Math.log(gMax) - Math.log(gMin))) * innerH;
  const line = POINTS.map((item) => `${xOf(item.ttft).toFixed(1)},${yOf(item.gapMs).toFixed(1)}`).join(" ");
  const yTicks = [30, 100, 300, 1000];
  const xTicks = [1.5, 1.75, 2.0];

  return (
    <Figure
      title="A chunk moves the stall; it does not remove it"
      controls={
        <Slider
          label="Chunk"
          values={CHUNKS}
          value={chunk}
          onChange={setChunk}
          format={(value) => (value === 16384 ? "whole prompt" : `${value} tokens`)}
        />
      }
      readout={
        <>
          <Stat label="Decoders' worst gap" value={`${point.gapMs.toFixed(1)} ms`} tone={color.a} />
          <Stat label="Long prompt TTFT" value={`${point.ttft.toFixed(3)} s`} tone={color.b} />
          <Stat label="Disaggregated" value={`${DISAGG.gapMs.toFixed(1)} ms, ${DISAGG.ttft.toFixed(3)} s`} />
        </>
      }
      caption="One colocated GPU, eight decoders, one 8,192-token prompt arriving at 2 s. Each point is a chunk size. The square is one prefill GPU plus one decode GPU on a 300 GB/s link. All values are simulated, not measured."
    >
      <div ref={ref}>
        <svg
          width={width}
          height={height}
          role="img"
          aria-label="Decoders' worst gap against the long prompt's time to first token, for six chunk sizes and for disaggregation"
        >
          {yTicks.map((tick) => (
            <g key={tick}>
              <line x1={left} x2={left + innerW} y1={yOf(tick)} y2={yOf(tick)} stroke={color.line} />
              <text x={left - 6} y={yOf(tick) + 4} textAnchor="end" fill={color.muted} fontSize={11}>
                {tick} ms
              </text>
            </g>
          ))}
          {xTicks.map((tick) => (
            <text key={tick} x={xOf(tick)} y={height - 14} textAnchor="middle" fill={color.muted} fontSize={11}>
              {tick.toFixed(2)} s
            </text>
          ))}
          <polyline fill="none" stroke={color.b} strokeWidth={2} points={line} />
          {POINTS.map((item) => (
            <circle
              key={item.chunk}
              cx={xOf(item.ttft)}
              cy={yOf(item.gapMs)}
              r={item.chunk === point.chunk ? 5 : 3}
              fill={item.chunk === point.chunk ? color.a : color.b}
            />
          ))}
          <rect
            x={xOf(DISAGG.ttft) - 5}
            y={yOf(DISAGG.gapMs) - 5}
            width={10}
            height={10}
            fill={color.ok}
          />
          <text x={left + innerW} y={height - 2} textAnchor="end" fill={color.muted} fontSize={11}>
            Long prompt TTFT
          </text>
        </svg>
      </div>
    </Figure>
  );
}
