import { useState } from "react";
import { Figure, Slider, Stat, Legend, color, useWidth, wash } from "../kit";

// From chapter 15, "Mapping positions to slots": B = 16 tokens per block, and
// slot(p) = table[floor(p / B)] * B + (p mod B). The chapter's example is
// position 37 -> logical block 2, offset 5 -> physical block 9 -> slot 149.
const B = 16;
const START = 37;

// Illustrative, not measured: a 50-token sequence holding four blocks. The
// entries descend and skip around, as the LIFO free list leaves them, and
// logical block 2 maps to physical block 9 to match the chapter's example.
const LENGTH = 50;
const TABLE = [14, 11, 9, 4] as const;
const POOL = 16;
// Blocks other sequences hold in this illustrative pool; the rest are free.
const OTHERS = new Set([0, 1, 2, 5, 7, 8, 12, 13, 15]);

const HEIGHT = 262;

export default function BlockTable() {
  const [p, setP] = useState(START);
  const [ref, width] = useWidth();

  const logical = Math.floor(p / B);
  const offset = p % B;
  const physical = TABLE[logical];
  const slot = physical * B + offset;
  const tailWaste = TABLE.length * B - LENGTH;

  // Logical view: the sequence's four blocks side by side.
  const gap = 8;
  const lw = (width - gap * (TABLE.length - 1)) / TABLE.length;
  const lx = (b: number) => b * (lw + gap);
  const logicalY = 24;
  const cellH = 16;
  const tableY = 76;
  const tableH = 20;

  // Physical pool: every block in one row.
  const pgap = 2;
  const pw = (width - pgap * (POOL - 1)) / POOL;
  const px = (id: number) => id * (pw + pgap);
  const poolY = 136;
  const poolH = 20;

  // Zoomed physical block.
  const zoomY = 196;
  const zw = width / B;

  const poolFill = (id: number) => {
    if (id === physical) return wash(color.a, 45);
    if ((TABLE as readonly number[]).includes(id)) return wash(color.accent, 30);
    if (OTHERS.has(id)) return wash(color.faint, 30);
    return "none";
  };

  return (
    <Figure
      title="A block table maps logical blocks onto scattered physical blocks"
      controls={<Slider label="Position p" min={0} max={LENGTH - 1} value={p} onChange={setP} />}
      readout={
        <>
          <Stat label="Position p" value={p} />
          <Stat label="Logical block ⌊p / 16⌋" value={logical} />
          <Stat label="Offset p mod 16" value={offset} tone={color.b} />
          <Stat label="Physical block" value={physical} tone={color.a} />
          <Stat label="Slot" value={`${physical} × 16 + ${offset} = ${slot}`} />
        </>
      }
      caption={
        <>
          The sequence looks contiguous through its block table, while its blocks sit anywhere in the pool. The only
          waste is the {tailWaste} empty slots of the last block, under the <code>block_size - 1</code> bound.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={HEIGHT}
          viewBox={`0 0 ${width} ${HEIGHT}`}
          role="img"
          aria-label={`A ${LENGTH}-token sequence in blocks of 16 has block table ${TABLE.join(", ")}. Position ${p} is logical block ${logical}, offset ${offset}, which maps to physical block ${physical}, slot ${slot}.`}
        >
          <text x={0} y={12}>
            Sequence, {LENGTH} tokens in logical blocks
          </text>
          {TABLE.map((_, b) => (
            <g key={b}>
              {Array.from({ length: B }, (_, t) => {
                const pos = b * B + t;
                const current = pos === p;
                return (
                  <rect
                    key={t}
                    x={lx(b) + (t * lw) / B}
                    y={logicalY}
                    width={lw / B}
                    height={cellH}
                    style={{
                      fill: current ? color.b : pos < LENGTH ? wash(color.accent, 30) : "none",
                      stroke: color.card,
                      strokeWidth: 0.5,
                    }}
                  />
                );
              })}
              <rect
                x={lx(b) + 0.5}
                y={logicalY + 0.5}
                width={lw - 1}
                height={cellH - 1}
                style={{ fill: "none", stroke: b === logical ? color.a : color.lineStrong, strokeWidth: b === logical ? 1.75 : 1 }}
              />
              <text x={lx(b) + lw / 2} y={logicalY + cellH + 13} textAnchor="middle" style={b === logical ? { fill: color.a } : undefined}>
                logical {b}
              </text>
              {/* The block table entry under its logical block. */}
              <rect
                x={lx(b) + lw / 2 - 18}
                y={tableY}
                width={36}
                height={tableH}
                rx={3}
                style={{
                  fill: b === logical ? wash(color.a, 18) : color.card,
                  stroke: b === logical ? color.a : color.lineStrong,
                  strokeWidth: b === logical ? 1.75 : 1,
                }}
              />
              <text x={lx(b) + lw / 2} y={tableY + 14} textAnchor="middle" style={{ fill: b === logical ? color.a : color.fg, fontWeight: 600 }}>
                {TABLE[b]}
              </text>
              <line
                x1={lx(b) + lw / 2}
                y1={tableY + tableH}
                x2={px(TABLE[b]) + pw / 2}
                y2={poolY}
                style={{ stroke: b === logical ? color.a : color.lineStrong, strokeWidth: b === logical ? 1.75 : 0.75 }}
              />
            </g>
          ))}
          <text x={width} y={tableY - 6} textAnchor="end">
            block table
          </text>

          <text x={0} y={poolY - 6}>
            Physical pool
          </text>
          {Array.from({ length: POOL }, (_, id) => (
            <g key={id}>
              <rect
                x={px(id) + 0.5}
                y={poolY + 0.5}
                width={pw - 1}
                height={poolH - 1}
                rx={2}
                style={{
                  fill: poolFill(id),
                  stroke: id === physical ? color.a : color.lineStrong,
                  strokeWidth: id === physical ? 1.75 : 0.75,
                  strokeDasharray: poolFill(id) === "none" ? "2 2" : undefined,
                }}
              />
              {pw >= 16 && (
                <text x={px(id) + pw / 2} y={poolY + 14} textAnchor="middle" style={{ fill: id === physical ? color.a : color.muted }}>
                  {id}
                </text>
              )}
            </g>
          ))}

          {/* Zoom into the physical block that holds position p. */}
          <path
            d={`M${px(physical)},${poolY + poolH} L0,${zoomY} M${px(physical) + pw},${poolY + poolH} L${width},${zoomY}`}
            style={{ fill: "none", stroke: color.line, strokeWidth: 1 }}
          />
          {Array.from({ length: B }, (_, t) => {
            const pos = logical * B + t;
            const current = t === offset;
            return (
              <rect
                key={t}
                x={t * zw + 0.5}
                y={zoomY}
                width={zw - 1}
                height={18}
                rx={2}
                style={{
                  fill: current ? color.b : pos < LENGTH ? wash(color.accent, 30) : "none",
                  stroke: current ? color.b : color.line,
                }}
              />
            );
          })}
          <text x={0} y={zoomY + 32}>
            slot {physical * B}
          </text>
          <text x={width} y={zoomY + 32} textAnchor="end">
            slot {physical * B + B - 1}
          </text>
          <text x={width / 2} y={zoomY + 32} textAnchor="middle" style={{ fill: color.fg }}>
            <tspan style={{ fill: color.a, fontWeight: 600 }}>{physical}</tspan>
            <tspan> × 16 + </tspan>
            <tspan style={{ fill: color.b, fontWeight: 600 }}>{offset}</tspan>
            <tspan> = {slot}</tspan>
          </text>
        </svg>
        <Legend
          items={[
            { label: "This sequence", tone: wash(color.accent, 45) },
            { label: "Other sequences", tone: wash(color.faint, 45) },
            { label: "Free", tone: color.line },
            { label: "Position p", tone: color.b },
          ]}
        />
      </div>
    </Figure>
  );
}
