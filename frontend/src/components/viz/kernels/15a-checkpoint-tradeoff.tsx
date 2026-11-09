import { useState } from "react";
import { Figure, Slider, Stat, color, formatBytes, useWidth } from "../kit";

// One hundred prompts: 40 × 256, 30 × 512, 20 × 1024, 10 × 2048.
// Each checkpoint is one recurrent state, 154,927,104 bytes (chapter 2).
// Positions and saved-token totals are the search in
// scripts/figures/15a-prefix-caching-and-radix-trees.py.
// This is not a measured hit rate.
const STATE_BYTES = 154_927_104;
const POINTS = [
  { k: 0, positions: "none", saved: 0 },
  { k: 1, positions: "512", saved: 30720 },
  { k: 2, positions: "256, 1024", saved: 48640 },
  { k: 3, positions: "256, 1024, 2048", saved: 58880 },
  { k: 4, positions: "256, 512, 1024, 2048", saved: 66560 },
] as const;

const KS = POINTS.map((point) => point.k);

export default function CheckpointTradeoff() {
  const [k, setK] = useState(2);
  const [ref, width] = useWidth();
  const point = POINTS.find((item) => item.k === k) ?? POINTS[2];
  const left = 52;
  const right = 12;
  const top = 12;
  const bottom = 28;
  const height = 220;
  const innerW = Math.max(width - left - right, 40);
  const innerH = height - top - bottom;
  const maxSaved = 70000;
  const xOf = (slots: number) => left + (slots / 4) * innerW;
  const yOf = (saved: number) => top + innerH - (saved / maxSaved) * innerH;
  const line = POINTS.map((item) => `${xOf(item.k).toFixed(1)},${yOf(item.saved).toFixed(1)}`).join(" ");
  const yTicks = [0, 20000, 40000, 60000];

  return (
    <Figure
      title="More checkpoints save fewer extra tokens"
      controls={
        <Slider
          label="Checkpoints"
          values={KS}
          value={k}
          onChange={setK}
          format={(value) => `${value} × ${formatBytes(STATE_BYTES, 2)}`}
        />
      }
      readout={
        <>
          <Stat label="Positions" value={point.positions} />
          <Stat label="Tokens saved" value={point.saved.toLocaleString("en-US")} tone={color.b} />
          <Stat label="State bytes" value={formatBytes(point.k * STATE_BYTES, 2)} tone={color.c} />
        </>
      }
      caption="One hundred prompts, forty of length 256, thirty of 512, twenty of 1,024, and ten of 2,048. Each point is the placement that saves the most prefill tokens at that checkpoint count. A prompt uses the rightmost checkpoint at or before its length."
    >
      <div ref={ref}>
        <svg
          width={width}
          height={height}
          role="img"
          aria-label="Saved prefill tokens against the number of recurrent-state checkpoints"
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
              key={item.k}
              cx={xOf(item.k)}
              cy={yOf(item.saved)}
              r={item.k === point.k ? 4.5 : 2.5}
              fill={item.k === point.k ? color.c : color.b}
            />
          ))}
          <text x={left + innerW} y={height - 6} textAnchor="end" fill={color.muted} fontSize={11}>
            Checkpoints
          </text>
        </svg>
      </div>
    </Figure>
  );
}
