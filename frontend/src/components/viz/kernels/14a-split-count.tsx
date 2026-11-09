import { useState } from "react";
import { Figure, Slider, Stat, color, useWidth } from "../kit";

// Chapter 10: 108 SMs, and 1275 GB/s from a device-to-device copy.
// The split counts are the table printed by
// scripts/figures/14a-split-kv-decode-attention.py
// for batch 1, 24 query heads, and a 64-token KV block.
// Implied GB/s is (24 * splits / 108) * 1275. It is not a measured kernel.
const SMS = 108;
const COPY_GBPS = 1275;
const HEADS = 24;
const POINTS = [
  { kv: 1, splits: 1 },
  { kv: 64, splits: 1 },
  { kv: 65, splits: 2 },
  { kv: 128, splits: 2 },
  { kv: 192, splits: 3 },
  { kv: 256, splits: 4 },
  { kv: 4096, splits: 4 },
  { kv: 32768, splits: 4 },
  { kv: 65536, splits: 4 },
] as const;

const LENGTHS = POINTS.map((point) => point.kv);

function impliedGbps(splits: number): number {
  return (HEADS * splits * COPY_GBPS) / SMS;
}

export default function SplitCount() {
  const [kv, setKv] = useState(32768);
  const [ref, width] = useWidth();
  const point = POINTS.find((item) => item.kv === kv) ?? POINTS[7];
  const left = 36;
  const right = 12;
  const top = 16;
  const bottom = 28;
  const height = 220;
  const innerW = Math.max(width - left - right, 40);
  const innerH = height - top - bottom;
  const maxSplits = 5;
  const xOf = (length: number) => left + (Math.log2(length) / 16) * innerW;
  const yOf = (splits: number) => top + innerH - (splits / maxSplits) * innerH;
  const line = POINTS.map((item) => `${xOf(item.kv).toFixed(1)},${yOf(item.splits).toFixed(1)}`).join(" ");
  const gbs = impliedGbps(point.splits);
  const unsplit = impliedGbps(1);

  return (
    <Figure
      title="Split count rises with the cache, then stops"
      controls={
        <Slider
          label="KV length"
          values={LENGTHS}
          value={kv}
          onChange={setKv}
          format={(value) => `${value.toLocaleString("en-US")} tokens`}
        />
      }
      readout={
        <>
          <Stat label="Splits" value={String(point.splits)} tone={color.b} />
          <Stat label="Programs" value={String(HEADS * point.splits)} />
          <Stat label="Implied GB/s" value={gbs.toFixed(1)} tone={color.c} />
          <Stat label="Unsplit GB/s" value={unsplit.toFixed(1)} />
        </>
      }
      caption="Batch 1 and 24 query heads on 108 SMs. Each step is one 64-token KV block. Implied GB/s assumes occupied SMs share the 1275 GB/s copy ceiling and idle SMs contribute none. This is not a measured kernel."
    >
      <div ref={ref}>
        <svg width={width} height={height} role="img" aria-label="Split count against KV length">
          {[0, 1, 2, 3, 4].map((tick) => (
            <g key={tick}>
              <line
                x1={left}
                x2={left + innerW}
                y1={yOf(tick)}
                y2={yOf(tick)}
                style={{ stroke: color.line }}
              />
              <text x={left - 8} y={yOf(tick) + 4} textAnchor="end" fontSize={11} style={{ fill: color.muted }}>
                {tick}
              </text>
            </g>
          ))}
          <polyline fill="none" strokeWidth={2} points={line} style={{ stroke: color.b }} />
          {POINTS.map((item) => (
            <circle
              key={item.kv}
              cx={xOf(item.kv)}
              cy={yOf(item.splits)}
              r={item.kv === point.kv ? 4.5 : 2.5}
              style={{ fill: item.kv === point.kv ? color.c : color.b }}
            />
          ))}
          <text x={left + innerW} y={height - 6} textAnchor="end" fontSize={11} style={{ fill: color.muted }}>
            KV length, log scale
          </text>
        </svg>
      </div>
    </Figure>
  );
}
