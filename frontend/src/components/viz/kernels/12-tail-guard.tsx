import { useState } from "react";
import { Figure, Slider, Stat, Legend, color, useWidth, wash, formatCount } from "../kit";

// Constants from chapter 12, "Grid geometry and the tail": 256 threads per
// block, 32 lanes per warp, so 8 warps per block. The sizes are the ones the
// lab tests (1, 1024, and 1,000,003) and benchmarks (2^24).
const BLOCK = 256;
const WARP = 32;
const WARPS_PER_BLOCK = BLOCK / WARP;
const SIZES = [1, 1024, 1_000_003, 2 ** 24] as const;

function geometry(n: number) {
  const blocks = Math.ceil(n / BLOCK);
  const threads = blocks * BLOCK;
  const firstIndex = (blocks - 1) * BLOCK; // global index of the last block's lane 0
  const live = n - firstIndex; // live lanes in the last block
  const divergingWarps = live % WARP === 0 ? 0 : 1;
  return { blocks, threads, idle: threads - n, firstIndex, live, divergingWarps, warps: blocks * WARPS_PER_BLOCK };
}

const HEIGHT = 200;

export default function TailGuard() {
  const [n, setN] = useState<number>(1_000_003);
  const [ref, width] = useWidth();
  const g = geometry(n);

  const labelW = 46;
  const top = 26;
  const rowH = 19;
  const cellW = (width - labelW) / WARP;
  const divergingRow = g.divergingWarps ? Math.floor(g.live / WARP) : -1;

  return (
    <Figure
      title="The tail guard splits at most one warp"
      controls={<Slider label="Elements n" values={SIZES} value={n} onChange={setN} format={(v) => formatCount(v)} />}
      readout={
        <>
          <Stat label="n" value={formatCount(n)} />
          <Stat label="Blocks of 256" value={formatCount(g.blocks)} />
          <Stat label="Threads launched" value={formatCount(g.threads)} />
          <Stat label="Idle threads" value={formatCount(g.idle)} />
          <Stat
            label="Diverging warps"
            value={`${g.divergingWarps} of ${formatCount(g.warps)}`}
            tone={g.divergingWarps ? color.warn : color.ok}
          />
        </>
      }
      caption={
        <>
          Every idle thread sits in the last block. Warps that are wholly live or wholly idle agree on{" "}
          <code>if (i &lt; n)</code>, so only the warp that holds the boundary diverges.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={HEIGHT}
          viewBox={`0 0 ${width} ${HEIGHT}`}
          role="img"
          aria-label={`The last block covers indices ${formatCount(g.firstIndex)} to ${formatCount(g.firstIndex + BLOCK - 1)}. ${g.live} of its 256 lanes are live, and ${g.divergingWarps} of its 8 warps diverges.`}
        >
          <text x={0} y={12}>
            Last block: {formatCount(g.firstIndex)} to {formatCount(g.firstIndex + BLOCK - 1)}
          </text>
          <text x={width} y={12} textAnchor="end" style={{ fill: color.ok }}>
            {g.live} live
          </text>
          {Array.from({ length: WARPS_PER_BLOCK }, (_, warp) => {
            const y = top + warp * rowH;
            const diverges = warp === divergingRow;
            return (
              <g key={warp}>
                <text x={0} y={y + 12} style={diverges ? { fill: color.warn, fontWeight: 600 } : undefined}>
                  warp {warp}
                </text>
                {Array.from({ length: WARP }, (_, lane) => {
                  const live = warp * WARP + lane < g.live;
                  return (
                    <rect
                      key={lane}
                      x={labelW + lane * cellW + 0.5}
                      y={y + 2}
                      width={Math.max(1, cellW - 1)}
                      height={rowH - 5}
                      rx={1}
                      style={{
                        fill: live ? color.ok : wash(color.faint, 18),
                        stroke: live ? "none" : color.line,
                        strokeWidth: 0.5,
                      }}
                    />
                  );
                })}
                {diverges && (
                  <rect
                    x={labelW - 2}
                    y={y}
                    width={width - labelW + 2}
                    height={rowH - 1}
                    rx={3}
                    style={{ fill: "none", stroke: color.warn, strokeWidth: 1.5 }}
                  />
                )}
              </g>
            );
          })}
          <text x={labelW} y={top + WARPS_PER_BLOCK * rowH + 14}>
            lane 0
          </text>
          <text x={width} y={top + WARPS_PER_BLOCK * rowH + 14} textAnchor="end">
            lane 31
          </text>
        </svg>
        <Legend
          items={[
            { label: "Live lane, i < n", tone: color.ok },
            { label: "Idle lane", tone: wash(color.faint, 40) },
            { label: "Warp that diverges", tone: color.warn },
          ]}
        />
      </div>
    </Figure>
  );
}
