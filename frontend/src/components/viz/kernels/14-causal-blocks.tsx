import { useState } from "react";
import { Figure, Slider, Stat, Legend, color, useWidth, wash, formatCount } from "../kit";

// Constants from chapter 14, "Causal masking and the blocks you can skip":
// BLOCK_M = 128 query rows per tile and BLOCK_N = 64 keys per block, on a
// square causal pass (q_len == kv_len, so offset = 0). At L = 8192 the chapter
// counts 4160 of 8192 block pairs, 50.8%.
const BLOCK_M = 128;
const BLOCK_N = 64;
const LENGTHS = [1024, 2048, 4096, 8192] as const;

type Kind = "skip" | "diagonal" | "full";

/** Which key blocks query tile i runs, from the kernel's causal bound `hi`. */
function classify(tile: number, block: number): Kind {
  const hi = (tile + 1) * BLOCK_M; // offset = 0 for a square pass
  const firstKey = block * BLOCK_N;
  const lastKey = firstKey + BLOCK_N - 1;
  if (firstKey >= hi) return "skip"; // never launched
  // A block needs the elementwise mask if any key lies past the tile's first query.
  return lastKey > tile * BLOCK_M ? "diagonal" : "full";
}

const HEIGHT = 236;

export default function CausalBlocks() {
  const [length, setLength] = useState<number>(8192);
  const [ref, width] = useWidth();

  const tiles = length / BLOCK_M;
  const blocks = length / BLOCK_N;
  const total = tiles * blocks;
  let run = 0;
  let diagonal = 0;
  for (let i = 0; i < tiles; i += 1) {
    for (let j = 0; j < blocks; j += 1) {
      const kind = classify(i, j);
      if (kind !== "skip") run += 1;
      if (kind === "diagonal") diagonal += 1;
    }
  }

  // The score matrix drawn as a square in token space: rows are queries,
  // columns are keys.
  const side = Math.min(width - 60, HEIGHT - 40);
  const left = Math.max(40, (width - side) / 2);
  const top = 20;
  const tileH = side / tiles;
  const blockW = side / blocks;
  const fill: Record<Kind, string> = {
    skip: wash(color.faint, 10),
    diagonal: color.warn,
    full: wash(color.accent, 55),
  };

  return (
    <Figure
      title="Causal attention runs only the blocks on or below the diagonal"
      controls={<Slider label="Sequence length L" values={LENGTHS} value={length} onChange={setLength} format={(v) => formatCount(v)} />}
      readout={
        <>
          <Stat label="L" value={formatCount(length)} />
          <Stat label="Query tiles × key blocks" value={`${tiles} × ${blocks} = ${formatCount(total)}`} />
          <Stat label="Blocks run" value={`${formatCount(run)} (${((run / total) * 100).toFixed(1)}%)`} tone={color.accent} />
          <Stat label="Of those, masked" value={formatCount(diagonal)} tone={color.warn} />
          <Stat label="Never launched" value={formatCount(total - run)} />
        </>
      }
      caption={
        <>
          The loop bound <code>hi</code> stops each query tile at its own last position, so the blocks above the
          diagonal never run. Only the blocks that straddle the diagonal pay for the elementwise{" "}
          <code>tl.where</code>, and the fraction run falls toward half as L grows.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={HEIGHT}
          viewBox={`0 0 ${width} ${HEIGHT}`}
          role="img"
          aria-label={`At L = ${length}, ${tiles} query tiles by ${blocks} key blocks is ${total} block pairs. The kernel runs ${run}, ${((run / total) * 100).toFixed(1)} percent, of which ${diagonal} straddle the diagonal and need the mask.`}
        >
          <text x={left} y={12}>
            keys →
          </text>
          <text x={left - 6} y={top + 10} textAnchor="end">
            queries
          </text>
          <text x={left - 6} y={top + 24} textAnchor="end">
            ↓
          </text>
          {/* One rect per run of same-kind blocks in a tile row keeps L = 8192 light. */}
          {Array.from({ length: tiles }, (_, i) => {
            const segments: { kind: Kind; from: number; to: number }[] = [];
            for (let j = 0; j < blocks; j += 1) {
              const kind = classify(i, j);
              const last = segments[segments.length - 1];
              if (last && last.kind === kind) last.to = j + 1;
              else segments.push({ kind, from: j, to: j + 1 });
            }
            return segments.map((seg) => (
              <rect
                key={`${i}-${seg.from}`}
                x={left + seg.from * blockW}
                y={top + i * tileH}
                width={(seg.to - seg.from) * blockW + 0.3}
                height={tileH + 0.3}
                style={{ fill: fill[seg.kind] }}
              />
            ));
          })}
          {/* Block boundaries, when they're wide enough to see. */}
          {tiles <= 16 && (
            <g style={{ stroke: color.card, strokeWidth: 0.75 }}>
              {Array.from({ length: blocks - 1 }, (_, j) => (
                <line key={`v${j}`} x1={left + (j + 1) * blockW} x2={left + (j + 1) * blockW} y1={top} y2={top + side} />
              ))}
              {Array.from({ length: tiles - 1 }, (_, i) => (
                <line key={`h${i}`} x1={left} x2={left + side} y1={top + (i + 1) * tileH} y2={top + (i + 1) * tileH} />
              ))}
            </g>
          )}
          <rect x={left} y={top} width={side} height={side} style={{ fill: "none", stroke: color.lineStrong }} />
          <line x1={left} y1={top} x2={left + side} y2={top + side} style={{ stroke: color.fg, strokeWidth: 1, strokeDasharray: "3 3" }} />
          <text x={left + side} y={top + side + 14} textAnchor="end">
            {formatCount(length)} × {formatCount(length)} scores
          </text>
        </svg>
        <Legend
          items={[
            { label: "Runs, no mask needed", tone: wash(color.accent, 55) },
            { label: "Runs, straddles the diagonal", tone: color.warn },
            { label: "Never launched", tone: wash(color.faint, 25) },
          ]}
        />
      </div>
    </Figure>
  );
}
