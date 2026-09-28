import { useState } from "react";
import { Figure, Slider, Stat, Legend, color, useWidth, wash, formatCount } from "../kit";

// Constants from chapter 13. Rows are 5120 wide in bfloat16, so one
// activation tensor is S = N * 5120 * 2 bytes. The A100's L2 is 40 MB.
const HIDDEN = 5120;
const BYTES = 2;
const L2_MB = 40;

// "The measurement: fusion loses on small tensors": the finer sweep, from the
// same A100, with buffers reused. Speedup is the chapter's column, not a
// recomputation from the rounded times.
const SWEEP = [
  { rows: 256, fused: 0.072, unfused: 0.074, speedup: 1.02 },
  { rows: 1024, fused: 0.081, unfused: 0.074, speedup: 0.91 },
  { rows: 2048, fused: 0.107, unfused: 0.09, speedup: 0.84 },
  { rows: 4096, fused: 0.155, unfused: 0.148, speedup: 0.95 },
  { rows: 8192, fused: 0.252, unfused: 0.263, speedup: 1.05 },
  { rows: 16384, fused: 0.444, unfused: 0.495, speedup: 1.12 },
  { rows: 32768, fused: 0.823, unfused: 0.958, speedup: 1.16 },
] as const;

// The byte count, 5S / 4S, and the chapter's fitted model
// speedup = 0.90 * (4 + phi) / 4 at its two ends, phi = 0 and phi = 1.
const PREDICTED = 5 / 4;
const EFFICIENCY = 0.9;
const MODEL_ABSORBED = EFFICIENCY * (4 + 0) / 4; // 0.90
const MODEL_HBM = EFFICIENCY * (4 + 1) / 4; // 1.125

// Break-even sits between 4096 and 8192 rows: 41.9 MB to 83.9 MB.
const CROSSOVER = [4096, 8192] as const;

const sizeMB = (rows: number) => (rows * HIDDEN * BYTES) / 1e6;

const X_MIN = 2;
const X_MAX = 450;
const Y_MIN = 0.8;
const Y_MAX = 1.3;
const HEIGHT = 250;
const X_TICKS = [3, 10, 30, 100, 300];
const Y_TICKS = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3];

export default function FusionVsL2() {
  const [rows, setRows] = useState(4096);
  const [ref, width] = useWidth();
  const point = SWEEP.find((p) => p.rows === rows) ?? SWEEP[3];
  const size = sizeMB(point.rows);

  const left = 34;
  const right = width - 6;
  const top = 18;
  const bottom = HEIGHT - 40;
  const x = (mb: number) => left + ((Math.log10(mb) - Math.log10(X_MIN)) / (Math.log10(X_MAX) - Math.log10(X_MIN))) * (right - left);
  const y = (s: number) => bottom - ((s - Y_MIN) / (Y_MAX - Y_MIN)) * (bottom - top);

  const hLine = (value: number, tone: string, dash: string, label: string, above = true) => (
    <g>
      <line x1={left} x2={right} y1={y(value)} y2={y(value)} style={{ stroke: tone, strokeDasharray: dash, strokeWidth: 1.25 }} />
      <text x={left + 4} y={y(value) + (above ? -4 : 12)} style={{ fill: tone }}>
        {label}
      </text>
    </g>
  );

  return (
    <Figure
      title="Fused RMSNorm speedup against tensor size"
      controls={
        <Slider
          label="Rows"
          values={SWEEP.map((p) => p.rows)}
          value={rows}
          onChange={setRows}
          format={(v) => formatCount(v)}
        />
      }
      readout={
        <>
          <Stat label="Rows" value={formatCount(point.rows)} />
          <Stat label="Tensor size S" value={`${size.toFixed(1)} MB`} />
          <Stat label="S against L2" value={`${(size / L2_MB).toFixed(2)}x`} />
          <Stat label="Fused, unfused" value={`${point.fused.toFixed(3)}, ${point.unfused.toFixed(3)} ms`} />
          <Stat label="Measured speedup" value={`${point.speedup.toFixed(2)}x`} tone={point.speedup < 1 ? color.bad : color.ok} />
          <Stat label="Byte count predicts" value={`${PREDICTED.toFixed(2)}x`} />
        </>
      }
      caption={
        <>
          The byte count promises 1.25x at every size, but the measured speedup crosses 1.0 only between 41.9 MB and
          83.9 MB, once the intermediate outgrows the 40 MB L2. The dotted lines are the model{" "}
          <code>0.90 × (4 + φ) / 4</code> with the cache absorbing all of the round trip (φ = 0) or none of it (φ = 1).
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={HEIGHT}
          viewBox={`0 0 ${width} ${HEIGHT}`}
          role="img"
          aria-label={`Measured fusion speedup dips to 0.84x at 21 MB, then rises to 1.16x at 335.5 MB. It stays below the predicted 1.25x everywhere and crosses 1.0 past the 40 MB L2. Selected: ${formatCount(point.rows)} rows, ${size.toFixed(1)} MB, ${point.speedup.toFixed(2)}x.`}
        >
          {/* Crossover band and L2 line. */}
          <rect
            x={x(sizeMB(CROSSOVER[0]))}
            y={top}
            width={x(sizeMB(CROSSOVER[1])) - x(sizeMB(CROSSOVER[0]))}
            height={bottom - top}
            style={{ fill: wash(color.warn, 14) }}
          />
          <line x1={x(L2_MB)} x2={x(L2_MB)} y1={top - 6} y2={bottom} style={{ stroke: color.warn, strokeWidth: 1.25 }} />
          <text x={x(L2_MB) - 4} y={top - 6} textAnchor="end" style={{ fill: color.warn }}>
            L2, 40 MB
          </text>

          {/* Axes. */}
          {Y_TICKS.map((t) => (
            <g key={t}>
              <line x1={left} x2={right} y1={y(t)} y2={y(t)} style={{ stroke: color.line, strokeWidth: t === 1 ? 1.25 : 0.5 }} />
              <text x={left - 5} y={y(t) + 4} textAnchor="end">
                {t.toFixed(1)}x
              </text>
            </g>
          ))}
          {X_TICKS.map((t) => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={bottom} y2={bottom + 4} style={{ stroke: color.lineStrong }} />
              <text x={x(t)} y={bottom + 16} textAnchor="middle">
                {t}
              </text>
            </g>
          ))}
          <text x={(left + right) / 2} y={bottom + 32} textAnchor="middle">
            Tensor size S in MB, log scale
          </text>

          {hLine(PREDICTED, color.muted, "5 4", "byte count, 1.25x")}
          {hLine(MODEL_HBM, color.a, "2 3", "model, φ = 1")}
          {hLine(MODEL_ABSORBED, color.a, "2 3", "model, φ = 0", false)}

          {/* Measured sweep. */}
          <polyline
            points={SWEEP.map((p) => `${x(sizeMB(p.rows))},${y(p.speedup)}`).join(" ")}
            style={{ fill: "none", stroke: color.accent, strokeWidth: 1.75 }}
          />
          {SWEEP.map((p) => {
            const selected = p.rows === point.rows;
            return (
              <circle
                key={p.rows}
                cx={x(sizeMB(p.rows))}
                cy={y(p.speedup)}
                r={selected ? 5.5 : 3.5}
                style={{
                  fill: selected ? color.paper : color.accent,
                  stroke: color.accent,
                  strokeWidth: selected ? 2.5 : 1,
                }}
              />
            );
          })}
        </svg>
        <Legend
          items={[
            { label: "Measured on the A100", tone: color.accent },
            { label: "Break-even range", tone: wash(color.warn, 40) },
          ]}
        />
      </div>
    </Figure>
  );
}
