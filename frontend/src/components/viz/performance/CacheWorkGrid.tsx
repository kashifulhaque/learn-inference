// Chapter 9, "From quadratic to linear": every token each generation step
// pushes through the model, drawn as one cell per token-forward-pass. The
// uncached panel's cells add up to np + n(n-1)/2 and the cached panel's to
// p + n - 1, the two formulas the chapter derives. The grid is kept small so
// that each cell stays visible; the caption gives the chapter's p = 512,
// n = 128 example.

import { useState } from "react";
import { Figure, Legend, Slider, Stat, color, formatCount, wash, useWidth } from "../kit";

// The largest grid the sliders allow. The cell size is fixed from these, so the
// figure's height doesn't change as the sliders move.
const MAX_PROMPT = 8;
const MAX_GENERATED = 12;
const HEAD = 22; // Space for the panel titles.
const FOOT = 26; // Space for the per-panel totals.

// The chapter's formulas, in token-forward-passes.
const workUncached = (p: number, n: number) => n * p + (n * (n - 1)) / 2;
const workCached = (p: number, n: number) => p + n - 1;

type Cell = { x: number; kind: "prompt" | "generated" | "cached" | "next" };

/** The cells of row j (the j-th generated token, from 0) for one panel. */
function rowCells(p: number, j: number, cached: boolean): Cell[] {
  const cells: Cell[] = [];
  const processed = p + j; // Tokens in the sequence when step j runs.
  for (let x = 0; x < processed; x += 1) {
    const computed = !cached || j === 0 || x === processed - 1;
    if (computed) cells.push({ x, kind: x < p ? "prompt" : "generated" });
    else cells.push({ x, kind: "cached" });
  }
  cells.push({ x: processed, kind: "next" });
  return cells;
}

function Panel({
  x0,
  cell,
  p,
  n,
  cached,
}: {
  x0: number;
  cell: number;
  p: number;
  n: number;
  cached: boolean;
}) {
  const total = cached ? workCached(p, n) : workUncached(p, n);
  const gap = cell >= 8 ? 1.5 : 1;
  const fill: Record<Cell["kind"], { fill: string; stroke: string }> = {
    prompt: { fill: color.a, stroke: "none" },
    generated: { fill: color.b, stroke: "none" },
    cached: { fill: wash(color.muted, 16), stroke: "none" },
    next: { fill: "none", stroke: color.faint },
  };
  return (
    <g transform={`translate(${x0},0)`}>
      <text x={0} y={12} style={{ fill: color.fg, fontWeight: 600 }}>
        {cached ? "With a cache" : "Without a cache"}
      </text>
      {Array.from({ length: n }, (_, j) =>
        rowCells(p, j, cached).map((c) => (
          <rect
            key={`${j}-${c.x}`}
            x={c.x * cell + gap / 2}
            y={HEAD + j * cell + gap / 2}
            width={cell - gap}
            height={cell - gap}
            rx={Math.min(2, cell / 5)}
            style={{ ...fill[c.kind], strokeWidth: 1, strokeDasharray: c.kind === "next" ? "2 2" : undefined }}
          />
        )),
      )}
      <text x={0} y={HEAD + MAX_GENERATED * cell + 17} style={{ fill: color.fg }}>
        {formatCount(total)} passes
      </text>
    </g>
  );
}

export default function CacheWorkGrid() {
  const [ref, width] = useWidth();
  const [p, setP] = useState(6);
  const [n, setN] = useState(10);

  const gutter = width < 420 ? 16 : 28;
  const panelWidth = (width - gutter) / 2;
  const cell = Math.max(4, Math.min(16, Math.floor((panelWidth / (MAX_PROMPT + MAX_GENERATED + 1)) * 10) / 10));
  const height = HEAD + MAX_GENERATED * cell + FOOT;
  const none = workUncached(p, n);
  const cache = workCached(p, n);

  return (
    <Figure
      title="Work per step, with and without a cache"
      controls={
        <>
          <Slider label="Prompt tokens, p" value={p} min={1} max={MAX_PROMPT} onChange={setP} />
          <Slider label="Generated tokens, n" value={n} min={1} max={MAX_GENERATED} onChange={setN} />
        </>
      }
      readout={
        <>
          <Stat label="Prompt, p" value={p} />
          <Stat label="Generated, n" value={n} />
          <Stat label="Without a cache, np + n(n−1)/2" value={formatCount(none)} />
          <Stat label="With a cache, p + n − 1" value={formatCount(cache)} />
          <Stat label="Ratio" value={`${(none / cache).toFixed(1)}×`} />
        </>
      }
      caption={
        <>
          Each row is one generation step and each filled cell one token-forward-pass: without a cache the
          staircase's area grows as n²/2, and with a cache each step after prefill computes one cell and reads the
          rest back. At the chapter's p = 512 and n = 128, the counts are 73,664 and 639.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={`Two grids of token-forward-passes for a ${p}-token prompt and ${n} generated tokens. Without a cache, each step recomputes every earlier token, ${formatCount(none)} passes in total. With a cache, prefill computes the prompt once and each later step computes one token, ${formatCount(cache)} passes in total.`}
        >
          <Panel x0={0} cell={cell} p={p} n={n} cached={false} />
          <Panel x0={panelWidth + gutter} cell={cell} p={p} n={n} cached />
        </svg>
        <Legend
          items={[
            { label: "Prompt token computed", tone: color.a },
            { label: "Generated token computed", tone: color.b },
            { label: "Read from the cache", tone: wash(color.muted, 16) },
            { label: "Token the step produces", tone: color.faint, dashed: true },
          ]}
        />
      </div>
    </Figure>
  );
}
