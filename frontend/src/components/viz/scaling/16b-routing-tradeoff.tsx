import { useState } from "react";
import { Figure, Slider, Stat, color, useWidth, wash } from "../kit";

// Seeded trace (seed 18): 600 requests at 40 per second over 4 replicas, each
// with a 96-block cache. The values are printed by
// scripts/figures/16b-scheduling-policy-and-routing.py from the lab solution.
// They come from a simulated clock, not an A100.
const POLICIES = [
  { name: "Round-robin", hit: 0.4397, p99: 175.57, busiest: 0.25 },
  { name: "Least-loaded", hit: 0.431, p99: 165.89, busiest: 0.275 },
  { name: "Affinity", hit: 0.7577, p99: 273.28, busiest: 0.3267 },
  { name: "Bounded 0.1", hit: 0.666, p99: 118.88, busiest: 0.27 },
  { name: "Bounded 0.25", hit: 0.663, p99: 135.9, busiest: 0.275 },
  { name: "Bounded 0.5", hit: 0.709, p99: 118.78, busiest: 0.2917 },
  { name: "Bounded 1.0", hit: 0.7177, p99: 177.21, busiest: 0.2767 },
];

export default function RoutingTradeoff() {
  const [index, setIndex] = useState(2);
  const [ref, width] = useWidth();
  const left = 104;
  const right = 64;
  const rowH = 20;
  const gap = 8;
  const top = 6;
  const height = top + POLICIES.length * (rowH + gap) + 18;
  const inner = Math.max(width - left - right, 40);
  const maxP99 = Math.max(...POLICIES.map((p) => p.p99));
  const pick = POLICIES[index];

  return (
    <Figure
      title="Affinity raises hits and the tail, a bounded load keeps both"
      controls={
        <Slider
          label="Policy"
          min={0}
          max={POLICIES.length - 1}
          value={index}
          onChange={setIndex}
          format={(value) => POLICIES[value].name}
        />
      }
      readout={
        <>
          <Stat label="Prefix hit rate" value={pick.hit.toFixed(3)} tone={color.a} />
          <Stat label="p99 TTFT" value={`${pick.p99.toFixed(1)} ms`} tone={color.b} />
          <Stat label="Busiest replica" value={`${(pick.busiest * 100).toFixed(1)}%`} tone={color.c} />
        </>
      }
      caption="Bar length is p99 TTFT. The label at the end is the prefix hit rate. Seeded trace on a simulated clock, one seed, so the bounded rows are not a smooth curve in epsilon."
    >
      <div ref={ref}>
        <svg width={width} height={height} role="img" aria-label="p99 time to first token and prefix hit rate for seven routing policies">
          {POLICIES.map((p, i) => {
            const y = top + i * (rowH + gap);
            const w = (p.p99 / maxP99) * inner;
            const active = i === index;
            return (
              <g key={p.name} onClick={() => setIndex(i)} style={{ cursor: "pointer" }}>
                <text x={0} y={y + 14} fill={active ? color.fg : color.muted} fontSize={12}>
                  {p.name}
                </text>
                <rect
                  x={left}
                  y={y}
                  width={w}
                  height={rowH}
                  style={{ fill: active ? color.b : wash(color.b, 45) }}
                />
                <text x={left + w + 6} y={y + 14} fill={color.muted} fontSize={12}>
                  {`hit ${p.hit.toFixed(2)}`}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </Figure>
  );
}
