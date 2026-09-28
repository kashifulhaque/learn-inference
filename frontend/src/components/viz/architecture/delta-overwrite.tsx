// Chapter 6: the two-slot memory, written three times, under plain accumulation
// (S + k vᵀ) and under the delta rule (S + β k (v − Sᵀk)ᵀ). The third write
// reuses key e₁, which is the overwrite plain accumulation can't express.

import { useState, type ReactNode } from "react";
import { Figure, Slider, Stat, color, useWidth, wash } from "../kit";

type Vec = readonly number[];
type Mat = number[][];

// Chapter 6's two-slot memory: d_k = 2 with perpendicular keys e₁ and e₂. The
// chapter keeps the values symbolic; these 2-vectors are illustrative.
const KEYS: { label: string; vector: Vec }[] = [
  { label: "e₁", vector: [1, 0] },
  { label: "e₂", vector: [0, 1] },
];
const WRITES: { key: number; value: Vec }[] = [
  { key: 0, value: [1, 2] },
  { key: 1, value: [2, -1] },
  { key: 0, value: [-1, 1] },
];
// β comes from a sigmoid, so it lies in (0, 1); the chapter's example uses 1.
const BETAS = [0.25, 0.5, 0.75, 1];

const HEIGHT = 226;
const CELL_H = 22;

/** Sᵀk: what the state returns for key k. */
function read(state: Mat, key: Vec): number[] {
  return state[0].map((_, b) => state.reduce((sum, row, a) => sum + row[b] * key[a], 0));
}

/** S + k addedᵀ, a rank-1 write along the key. */
function write(state: Mat, key: Vec, added: Vec): Mat {
  return state.map((row, a) => row.map((entry, b) => entry + key[a] * added[b]));
}

type Frame = { state: Mat; added: number[] };

function run(rule: "plain" | "delta", beta: number): Frame[] {
  let state: Mat = [
    [0, 0],
    [0, 0],
  ];
  return WRITES.map(({ key, value }) => {
    const k = KEYS[key].vector;
    const current = read(state, k);
    const added = rule === "plain" ? [...value] : value.map((v, b) => beta * (v - current[b]));
    state = write(state, k, added);
    return { state, added };
  });
}

function num(value: number): string {
  const rounded = Math.round(value * 100) / 100;
  const text = Object.is(rounded, -0) ? "0" : String(rounded);
  return text.replace("-", "−");
}

function vec(values: Vec): string {
  return `(${values.map(num).join(", ")})`;
}

function same(a: Vec, b: Vec): boolean {
  return a.every((value, index) => Math.abs(value - b[index]) < 1e-9);
}

function Matrix({ x, y, cellW, frame, highlight }: {
  x: number; y: number; cellW: number; frame: Frame; highlight: number;
}) {
  return (
    <g>
      {frame.state.map((row, a) => (
        <g key={a}>
          <text x={x - 5} y={y + a * (CELL_H + 2) + CELL_H / 2 + 4} textAnchor="end">
            {KEYS[a].label}
          </text>
          {row.map((entry, b) => (
            <g key={b}>
              <rect
                x={x + b * (cellW + 2)}
                y={y + a * (CELL_H + 2)}
                width={cellW}
                height={CELL_H}
                rx={3}
                style={{
                  fill: a === highlight ? wash(color.accent, 14) : color.well,
                  stroke: a === highlight ? color.accent : color.line,
                }}
              />
              <text
                x={x + b * (cellW + 2) + cellW / 2}
                y={y + a * (CELL_H + 2) + CELL_H / 2 + 4}
                textAnchor="middle"
                style={{ fill: color.fg }}
              >
                {num(entry)}
              </text>
            </g>
          ))}
        </g>
      ))}
    </g>
  );
}

export default function DeltaOverwrite() {
  const [ref, width] = useWidth();
  const [beta, setBeta] = useState(1);
  const plain = run("plain", beta);
  const delta = run("delta", beta);

  const column = width / WRITES.length;
  const cellW = Math.max(30, Math.min(44, (column - 26) / 2));
  const matrixW = 2 * cellW + 2;
  const target = WRITES[WRITES.length - 1].value;
  const plainRead = read(plain[plain.length - 1].state, KEYS[0].vector);
  const deltaRead = read(delta[delta.length - 1].state, KEYS[0].vector);

  const rows: { title: ReactNode; frames: Frame[]; top: number }[] = [
    { title: <>Plain sum</>, frames: plain, top: 40 },
    {
      title: (
        <>
          Delta rule, <tspan style={{ fill: color.b }}>β = {num(beta)}</tspan>
        </>
      ),
      frames: delta,
      top: 136,
    },
  ];

  return (
    <Figure
      title="Overwriting a key: plain sum against the delta rule"
      controls={<Slider label="Step size β" values={BETAS} value={beta} onChange={setBeta} />}
      readout={
        <>
          <Stat label="Plain sum reads e₁ as" value={vec(plainRead)} tone={same(plainRead, target) ? color.ok : color.bad} />
          <Stat label="Delta rule reads e₁ as" value={vec(deltaRead)} tone={same(deltaRead, target) ? color.ok : color.bad} />
          <Stat label="Latest value written to e₁" value={vec(target)} />
          <Stat label="β" value={num(beta)} tone={color.b} />
        </>
      }
      caption={
        <>
          Each matrix is the state after one write, one row per key, so a read with e₁ returns row e₁. On the
          third write the plain sum adds the new value to the old one, while the delta rule adds only the
          difference, replacing row e₁ at β = 1 and leaving row e₂ untouched.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={HEIGHT}
          role="img"
          aria-label={`Three writes to a two-slot memory: e₁ gets ${vec(WRITES[0].value)}, e₂ gets ${vec(WRITES[1].value)}, then e₁ gets ${vec(target)}. Afterwards the plain sum reads e₁ as ${vec(plainRead)} and the delta rule at β = ${num(beta)} reads it as ${vec(deltaRead)}.`}
        >
          {WRITES.map((w, index) => {
            const cx = column * (index + 0.5);
            return (
              <g key={index}>
                <text x={cx} y={12} textAnchor="middle" style={{ fill: color.fg }}>
                  Write {index + 1}
                </text>
                <text x={cx} y={26} textAnchor="middle">
                  {KEYS[w.key].label} ← {vec(w.value)}
                </text>
              </g>
            );
          })}
          {rows.map((row, r) => (
            <g key={r}>
              <line x1={0} x2={width} y1={row.top - 2} y2={row.top - 2} style={{ stroke: color.line }} />
              <text x={0} y={row.top + 12} style={{ fill: color.fg }}>
                {row.title}
              </text>
              {row.frames.map((frame, index) => {
                const cx = column * (index + 0.5);
                const x = cx - matrixW / 2 + 8;
                return (
                  <g key={index}>
                    <Matrix x={x} y={row.top + 20} cellW={cellW} frame={frame} highlight={WRITES[index].key} />
                    <text x={cx} y={row.top + 20 + 2 * CELL_H + 2 + 16} textAnchor="middle">
                      +{vec(frame.added)}
                    </text>
                  </g>
                );
              })}
            </g>
          ))}
        </svg>
      </div>
    </Figure>
  );
}
