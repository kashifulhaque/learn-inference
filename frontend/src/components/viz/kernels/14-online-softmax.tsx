import { useState } from "react";
import { Figure, Segmented, StepControls, Stat, color, useStepper, useWidth, wash } from "../kit";

// The four-score row from chapter 14, "The online softmax on four scores".
// At two keys per block, the trace reproduces the chapter's numbers:
// m = 2 then 4, alpha = 0.135, l = 1.553, u = 47.92, and 47.92 / 1.553 = 30.86.
const SCORES = [1, 2, 4, 3] as const;
const VALUES = [10, 20, 30, 40] as const;
const BLOCK_SIZES = [1, 2, 4] as const;

type Row = { keys: number[]; m: number; alpha: number | null; l: number; u: number };

/** The running maximum, correction, sum, and output after each key block. */
function trace(blockSize: number): Row[] {
  const rows: Row[] = [{ keys: [], m: -Infinity, alpha: null, l: 0, u: 0 }];
  let m = -Infinity;
  let l = 0;
  let u = 0;
  for (let start = 0; start < SCORES.length; start += blockSize) {
    const keys = Array.from({ length: Math.min(blockSize, SCORES.length - start) }, (_, k) => start + k);
    const mNew = Math.max(m, ...keys.map((j) => SCORES[j]));
    const alpha = Math.exp(m - mNew); // exp(-inf) = 0 on the first block, as in the kernel
    l = alpha * l + keys.reduce((sum, j) => sum + Math.exp(SCORES[j] - mNew), 0);
    u = alpha * u + keys.reduce((sum, j) => sum + Math.exp(SCORES[j] - mNew) * VALUES[j], 0);
    m = mNew;
    rows.push({ keys, m, alpha, l, u });
  }
  return rows;
}

const HEIGHT = 292;
const fmtM = (m: number) => (m === -Infinity ? "−∞" : String(m));
const fmtAlpha = (a: number | null) => (a === null ? "—" : a === 0 ? "0" : a.toFixed(3));

export default function OnlineSoftmax() {
  const [blockSize, setBlockSize] = useState<number>(2);
  const rows = trace(blockSize);
  const blocks = rows.length - 1;
  const count = blocks + 2; // start, one step per block, then the division
  const stepper = useStepper(count, 1100);
  const step = Math.min(stepper.step, count - 1);
  const [ref, width] = useWidth();

  const finished = step === count - 1;
  const state = rows[Math.min(step, blocks)];
  const final = rows[blocks];
  const result = final.u / final.l;
  const seen = new Set(rows.slice(1, Math.min(step, blocks) + 1).flatMap((r) => r.keys));
  const currentKeys = step >= 1 && step <= blocks ? rows[step].keys : [];

  // Key columns.
  const colW = width / SCORES.length;
  const barBase = 122;
  // Short enough that a full bar's label clears the "v =" line above it.
  const barMax = 48;
  const barW = Math.min(34, colW * 0.45);

  // Trace table.
  const tableTop = 160;
  const rowH = 19;
  const labelW = Math.min(88, width * 0.24);
  const cellW = (width - labelW) / 4;
  const colX = (c: number) => labelW + c * cellW + cellW / 2;
  const heads = [
    { label: "m", tone: color.a },
    { label: "α", tone: color.d },
    { label: "ℓ", tone: color.b },
    { label: "u", tone: color.c },
  ];

  const describe = (s: number) => {
    if (s === 0) return "nothing seen yet";
    if (s === count - 1) return "divide once";
    const { keys } = rows[s];
    return keys.length === 1 ? `key ${keys[0] + 1}` : `keys ${keys[0] + 1} to ${keys[keys.length - 1] + 1}`;
  };

  return (
    <Figure
      title="The online softmax, one key block at a time"
      controls={
        <>
          <Segmented
            label="Keys per block"
            value={blockSize}
            options={BLOCK_SIZES.map((b) => ({ value: b, label: String(b) }))}
            onChange={(b) => {
              setBlockSize(b);
              stepper.reset();
            }}
          />
          <StepControls stepper={stepper} count={count} describe={describe} />
        </>
      }
      readout={
        <>
          <Stat label="Keys per block" value={blockSize} />
          <Stat label="Running maximum m" value={fmtM(state.m)} tone={color.a} />
          <Stat label="Streaming result u / ℓ" value={finished ? result.toFixed(2) : "not yet"} />
          <Stat label="One-pass softmax" value={result.toFixed(2)} />
        </>
      }
      caption={
        <>
          Each bar is <code>e^(s − m)</code> against the current maximum. When the maximum rises, the bars already
          seen shrink by α, and the sum ℓ and the output u shrink with them; any block size ends at the same 30.86.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={HEIGHT}
          viewBox={`0 0 ${width} ${HEIGHT}`}
          role="img"
          aria-label={`Four scores 1, 2, 4, 3 with values 10, 20, 30, 40, in blocks of ${blockSize}. After ${describe(step)}, the running maximum is ${fmtM(state.m)}, the sum is ${state.l.toFixed(3)}, and the output is ${state.u.toFixed(2)}. The final result is ${result.toFixed(2)}.`}
        >
          {/* Block brackets and the block being processed. */}
          {rows.slice(1).map((row, b) => {
            const x0 = row.keys[0] * colW;
            const x1 = (row.keys[row.keys.length - 1] + 1) * colW;
            const active = b + 1 === step;
            return (
              <g key={b}>
                {active && (
                  <rect x={x0 + 2} y={2} width={x1 - x0 - 4} height={barBase + 4} rx={4} style={{ fill: wash(color.accent, 10) }} />
                )}
                <path
                  d={`M${x0 + 6},${22} v-5 H${x1 - 6} v5`}
                  style={{ fill: "none", stroke: active ? color.accent : color.lineStrong, strokeWidth: 1 }}
                />
                <text x={(x0 + x1) / 2} y={13} textAnchor="middle" style={active ? { fill: color.accent, fontWeight: 600 } : undefined}>
                  block {b + 1}
                </text>
              </g>
            );
          })}
          {SCORES.map((s, j) => {
            const cx = j * colW + colW / 2;
            const isSeen = seen.has(j) || finished;
            const weight = Math.exp(s - (finished ? final.m : state.m));
            const h = isSeen ? weight * barMax : 0;
            return (
              <g key={j}>
                <text x={cx} y={38} textAnchor="middle" style={{ fill: color.fg }}>
                  s = {s}
                </text>
                <text x={cx} y={52} textAnchor="middle">
                  v = {VALUES[j]}
                </text>
                {isSeen ? (
                  <>
                    <rect
                      x={cx - barW / 2}
                      y={barBase - h}
                      width={barW}
                      height={h}
                      style={{
                        fill: wash(color.b, currentKeys.includes(j) ? 70 : 40),
                        stroke: color.b,
                        strokeWidth: 1,
                      }}
                    />
                    <text x={cx} y={barBase - h - 4} textAnchor="middle" style={{ fill: color.b }}>
                      {weight.toFixed(3)}
                    </text>
                  </>
                ) : (
                  <>
                    <rect
                      x={cx - barW / 2}
                      y={barBase - barMax}
                      width={barW}
                      height={barMax}
                      style={{ fill: "none", stroke: color.line, strokeDasharray: "3 3" }}
                    />
                    <text x={cx} y={barBase - barMax / 2 + 4} textAnchor="middle" style={{ fill: color.faint }}>
                      unseen
                    </text>
                  </>
                )}
              </g>
            );
          })}
          <line x1={0} x2={width} y1={barBase} y2={barBase} style={{ stroke: color.lineStrong }} />
          <text x={0} y={barBase + 16}>
            Bar height: e^(s − m) with m = {fmtM(finished ? final.m : state.m)}
          </text>

          {/* The trace: one row per step, future rows faint. */}
          {heads.map((head, c) => (
            <text key={head.label} x={colX(c)} y={tableTop} textAnchor="middle" style={{ fill: head.tone, fontWeight: 600 }}>
              {head.label}
            </text>
          ))}
          <line x1={0} x2={width} y1={tableTop + 5} y2={tableTop + 5} style={{ stroke: color.line }} />
          {Array.from({ length: count }, (_, r) => {
            const y = tableTop + 5 + r * rowH;
            const reached = r <= step;
            const tone = (t: string) => (reached ? t : color.faint);
            const label = r === 0 ? "Start" : r === count - 1 ? "Finish" : `Block ${r}`;
            return (
              <g key={r}>
                {r === step && <rect x={0} y={y + 1} width={width} height={rowH - 2} rx={3} style={{ fill: color.tint }} />}
                <text x={4} y={y + 13} style={{ fill: tone(color.fg), fontWeight: r === step ? 600 : 400 }}>
                  {label}
                </text>
                {r < count - 1 ? (
                  [fmtM(rows[r].m), fmtAlpha(rows[r].alpha), rows[r].l === 0 ? "0" : rows[r].l.toFixed(3), rows[r].u === 0 ? "0" : rows[r].u.toFixed(2)].map(
                    (value, c) => (
                      <text key={c} x={colX(c)} y={y + 13} textAnchor="middle" style={{ fill: tone(heads[c].tone) }}>
                        {value}
                      </text>
                    ),
                  )
                ) : (
                  <text x={labelW + 8} y={y + 13} style={{ fill: tone(color.fg) }}>
                    u / ℓ = {final.u.toFixed(2)} / {final.l.toFixed(3)} = {result.toFixed(2)}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>
    </Figure>
  );
}
