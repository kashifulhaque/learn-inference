import { useState } from "react";
import { Figure, Legend, Slider, Stat, color, useWidth } from "../kit";

// PCIe time to load adapters over a seeded trace of 300 batches, 48 adapters
// of rank 8, 16, and 64, four adapters per batch. Printed by
// scripts/figures/18c-multi-lora-serving.py from content/labs/18c-multi-lora.
// Link: 32 GB/s one way. Computed on a simulator. Not an A100 measurement.
const POINTS = [
  { gib: 2, fifo: 4169.7, lru: 4115.6, freq: 3581.4, misses: [667, 653, 565] },
  { gib: 3, fifo: 3620.3, lru: 3605.2, freq: 2958.6, misses: [579, 575, 417] },
  { gib: 4, fifo: 3016.9, lru: 3118.6, freq: 2428.7, misses: [484, 493, 340] },
  { gib: 6, fifo: 2638.5, lru: 2517.4, freq: 1792.9, misses: [410, 384, 249] },
  { gib: 8, fifo: 1373.3, lru: 1946.4, freq: 1148.4, misses: [213, 277, 162] },
] as const;

const GIBS = POINTS.map((point) => point.gib);
const POLICIES = [
  { key: "fifo", label: "FIFO", tone: color.c },
  { key: "lru", label: "LRU", tone: color.d },
  { key: "freq", label: "Frequency, pinned", tone: color.b },
] as const;

export default function AdapterPool() {
  const [gib, setGib] = useState(4);
  const [ref, width] = useWidth();
  const point = POINTS.find((item) => item.gib === gib) ?? POINTS[2];
  const left = 130;
  const right = 56;
  const rowH = 34;
  const height = rowH * 3 + 16;
  const innerW = Math.max(width - left - right, 40);
  const xMax = 4500;
  const best = Math.min(point.fifo, point.lru);

  return (
    <Figure
      title="Adapter pool: PCIe time by eviction policy"
      controls={<Slider label="Pool size (GiB)" values={GIBS} value={gib} onChange={setGib} format={(v) => String(v)} />}
      readout={
        <>
          <Stat label="FIFO" value={`${point.fifo.toFixed(1)} ms`} tone={color.c} />
          <Stat label="LRU" value={`${point.lru.toFixed(1)} ms`} tone={color.d} />
          <Stat label="Frequency" value={`${point.freq.toFixed(1)} ms`} tone={color.b} />
          <Stat label="Frequency over best plain" value={(point.freq / best).toFixed(3)} />
        </>
      }
      caption="Total host-to-device copy time over 300 batches at 32 GB/s one way, from the lab's simulator with seed 17. The number after a bar is its count of loads. A scan of cold adapters every 30 batches flushes recency. The numbers are computed, not measured on an A100."
    >
      <div ref={ref}>
        <svg width={width} height={height} role="img" aria-label="Bars of adapter load time for FIFO, LRU, and the frequency policy">
          {POLICIES.map((policy, i) => {
            const value = point[policy.key];
            const w = (value / xMax) * innerW;
            const y = 8 + i * rowH;
            return (
              <g key={policy.key}>
                <text x={left - 8} y={y + 17} textAnchor="end" fontSize={12} style={{ fill: color.muted }}>
                  {policy.label}
                </text>
                <rect x={left} y={y} width={w} height={22} rx={3} style={{ fill: policy.tone }} />
                <text x={left + w + 6} y={y + 16} fontSize={11} style={{ fill: color.muted }}>
                  {point.misses[i]}
                </text>
              </g>
            );
          })}
        </svg>
        <Legend items={POLICIES.map((policy) => ({ label: policy.label, tone: policy.tone }))} />
      </div>
    </Figure>
  );
}
