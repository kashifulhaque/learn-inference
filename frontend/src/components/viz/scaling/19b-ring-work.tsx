import { useState } from "react";
import { Figure, Segmented, Stat, color, useWidth, wash } from "../kit";

// Chapter 19b: causal ring attention over 4 simulated ranks and 64 tokens.
// Each number is the count of (query, key) pairs a rank computes at a ring
// step, taken from `ring_attention` in content/labs/19b-pipeline-and-ring/
// solution.py. scripts/figures/19b-pipeline-and-context-parallelism.py
// prints the same matrices. Row = ring step, column = rank. A full 16 x 16
// block is 256 pairs.
type Kind = "contiguous" | "striped" | "zigzag";

const FULL_BLOCK = 256;
const WORK: Record<Kind, number[][]> = {
  contiguous: [
    [136, 136, 136, 136],
    [0, 256, 256, 256],
    [0, 0, 256, 256],
    [0, 0, 0, 256],
  ],
  striped: [
    [136, 136, 136, 136],
    [120, 136, 136, 136],
    [120, 120, 136, 136],
    [120, 120, 120, 136],
  ],
  zigzag: [
    [136, 136, 136, 136],
    [128, 128, 128, 128],
    [128, 128, 128, 128],
    [128, 128, 128, 128],
  ],
};

function stats(work: number[][]) {
  const totals = [0, 1, 2, 3].map((r) => work.reduce((sum, row) => sum + row[r], 0));
  const mean = totals.reduce((a, b) => a + b, 0) / totals.length;
  const critical = work.reduce((sum, row) => sum + Math.max(...row), 0);
  return { totals, imbalance: Math.max(...totals) / mean, critical, efficiency: mean / critical };
}

export default function RingWork() {
  const [kind, setKind] = useState<Kind>("contiguous");
  const [ref, width] = useWidth();
  const work = WORK[kind];
  const s = stats(work);
  const left = 64;
  const top = 22;
  const cell = Math.min(64, (width - left - 8) / 4);
  const height = top + 4 * cell + 6;

  return (
    <Figure
      title="Who waits in a causal ring"
      controls={
        <Segmented
          label="Token assignment"
          value={kind}
          onChange={setKind}
          options={[
            { value: "contiguous", label: "Contiguous" },
            { value: "striped", label: "Striped" },
            { value: "zigzag", label: "Zigzag" },
          ]}
        />
      }
      readout={
        <>
          <Stat label="Busiest rank over mean" value={s.imbalance.toFixed(3)} tone={color.c} />
          <Stat label="Critical path, pairs" value={String(s.critical)} tone={color.b} />
          <Stat label="Ideal, pairs" value={String(s.totals.reduce((a, b) => a + b, 0) / 4)} />
          <Stat label="Efficiency" value={s.efficiency.toFixed(3)} />
        </>
      }
      caption="Each cell is the work one rank does at one ring step, shaded by its share of a full 16 by 16 block. A step lasts as long as its busiest rank, so contiguous shards leave most ranks idle while the last rank does a full block at every step. Computed on 64 tokens and 4 ranks, not measured on a GPU."
    >
      <div ref={ref}>
        <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img"
          aria-label={`Work per rank and ring step for ${kind} assignment. Busiest rank over mean is ${s.imbalance.toFixed(3)}.`}>
          {[0, 1, 2, 3].map((r) => (
            <text key={r} x={left + r * cell + cell / 2} y={14} textAnchor="middle" style={{ fill: color.muted }}>
              rank {r}
            </text>
          ))}
          {work.map((row, step) => (
            <g key={step}>
              <text x={left - 8} y={top + step * cell + cell / 2 + 4} textAnchor="end" style={{ fill: color.muted }}>
                step {step}
              </text>
              {row.map((pairs, r) => (
                <g key={r}>
                  <rect
                    x={left + r * cell + 2}
                    y={top + step * cell + 2}
                    width={cell - 4}
                    height={cell - 4}
                    rx={4}
                    style={{
                      fill: pairs === 0 ? "transparent" : wash(color.b, 12 + (pairs / FULL_BLOCK) * 70),
                      stroke: pairs === Math.max(...row) ? color.c : color.line,
                      strokeWidth: pairs === Math.max(...row) ? 2 : 1,
                    }}
                  />
                  <text x={left + r * cell + cell / 2} y={top + step * cell + cell / 2 + 4} textAnchor="middle" style={{ fill: color.fg }}>
                    {pairs === 0 ? "skip" : pairs}
                  </text>
                </g>
              ))}
            </g>
          ))}
        </svg>
      </div>
    </Figure>
  );
}
