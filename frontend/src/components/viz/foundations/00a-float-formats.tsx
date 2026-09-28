import { useState } from "react";
import { Figure, Segmented, Stat, color, useWidth, wash } from "../kit";
import { BFLOAT16, FLOAT16, FLOAT32, bitsOf, encode, type FloatFormat } from "./helpers";

// Notation chapter, "Floating-point formats": the three formats' field widths.
const FORMATS = {
  float32: FLOAT32,
  float16: FLOAT16,
  bfloat16: BFLOAT16,
} as const satisfies Record<string, FloatFormat>;
type FormatName = keyof typeof FORMATS;

// Values the chapter uses: 257 (the 256 + 1 example), 65504 (float16's largest
// finite value), and a few ordinary numbers to show the precision gap.
const VALUES = [
  { value: "0.1", label: "0.1", x: 0.1 },
  { value: "pi", label: "π", x: Math.PI },
  { value: "257", label: "257", x: 257 },
  { value: "65504", label: "65504", x: 65504 },
  { value: "100000", label: "100000", x: 100000 },
] as const;
type ValueName = (typeof VALUES)[number]["value"];

const FIELD_TONE = { sign: color.d, exponent: color.c, mantissa: color.b } as const;
type Field = keyof typeof FIELD_TONE;

function show(v: number): string {
  if (!Number.isFinite(v)) return "infinity";
  if (Number.isInteger(v)) return v.toLocaleString("en-US");
  return String(Number(v.toPrecision(10)));
}

function showSmall(v: number): string {
  if (v >= 0.01 && v < 1e6) return show(v);
  return v.toExponential(2);
}

export default function FloatFormats() {
  const [ref, width] = useWidth();
  const [formatName, setFormatName] = useState<FormatName>("bfloat16");
  const [valueName, setValueName] = useState<ValueName>("257");
  const format = FORMATS[formatName];
  const typed = VALUES.find((v) => v.value === valueName)!;
  const encoded = encode(typed.x, format);
  const bits = bitsOf(encoded, format);
  const total = bits.length;
  const bias = 2 ** (format.exponentBits - 1) - 1;
  const p = format.mantissaBits + 1;

  // Wrap 32 bits onto two rows when a cell would be narrower than 15 px.
  const perRow = width / total < 15 ? 16 : total;
  const rows = Math.ceil(total / perRow);
  const cell = Math.min(24, Math.floor(width / perRow));
  const rowHeight = 50;
  const textTop = rows * rowHeight + 14;
  const height = textTop + 2 * 17 + 6;

  const fieldOf = (i: number): Field => (i === 0 ? "sign" : i <= format.exponentBits ? "exponent" : "mantissa");

  // Contiguous runs of one field within each row, for the labels above the cells.
  const runs: { field: Field; row: number; start: number; end: number }[] = [];
  bits.forEach((_, i) => {
    const row = Math.floor(i / perRow);
    const col = i % perRow;
    const field = fieldOf(i);
    const last = runs[runs.length - 1];
    if (last && last.field === field && last.row === row) last.end = col;
    else runs.push({ field, row, start: col, end: col });
  });
  const fieldLabel = (field: Field, span: number) => {
    const n = field === "sign" ? 1 : field === "exponent" ? format.exponentBits : format.mantissaBits;
    if (field === "sign") return span * cell >= 30 ? "sign" : "s";
    const name = span * cell >= 90 ? field : field === "exponent" ? "exp" : "mant";
    return `${name} (${n})`;
  };

  const relError = encoded.kind === "infinite" ? NaN : Math.abs(encoded.value - typed.x) / Math.abs(typed.x);
  const significand = show(1 + encoded.mantissaField / 2 ** format.mantissaBits);
  const lines =
    encoded.kind === "infinite"
      ? ["exponent field all ones, mantissa 0", `stores infinity: ${show(typed.x)} > ${show(65504)}`]
      : [
          `exponent ${encoded.exponentField} − ${bias} = ${encoded.exponent}`,
          `significand ${significand}`,
          `${significand} × 2^${encoded.exponent} = ${show(encoded.value)}`,
        ];

  return (
    <Figure
      title="What each format stores"
      controls={
        <>
          <Segmented
            label="Format"
            value={formatName}
            onChange={setFormatName}
            options={[
              { value: "float32", label: "float32" },
              { value: "float16", label: "float16" },
              { value: "bfloat16", label: "bfloat16" },
            ]}
          />
          <Segmented
            label="Value"
            value={valueName}
            onChange={setValueName}
            options={VALUES.map(({ value, label }) => ({ value, label }))}
          />
        </>
      }
      readout={
        <>
          <Stat label="Format" value={`${formatName}, ${p} significand bits`} />
          <Stat label="Typed" value={typed.value === "pi" ? `π ≈ ${show(typed.x)}` : show(typed.x)} />
          <Stat label="Stored" value={show(encoded.value)} tone={color.accent} />
          <Stat
            label="Relative error"
            value={Number.isNaN(relError) ? "overflow" : relError === 0 ? "0, exact" : relError.toExponential(1)}
          />
          <Stat label={`Unit roundoff 2^−${p}`} value={(2 ** -p).toExponential(1)} />
          <Stat label="ULP here" value={Number.isNaN(encoded.ulp) ? "—" : showSmall(encoded.ulp)} />
        </>
      }
      caption={
        <>
          bfloat16 keeps float32's 8 exponent bits and drops 16 mantissa bits, so 257 rounds to 256 but 100000 stays
          finite. float16 keeps 10 mantissa bits and stores 257 exactly, but anything past 65504 overflows.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={`The ${total} bits of ${show(typed.x)} in ${formatName}: 1 sign bit, ${format.exponentBits} exponent bits, and ${format.mantissaBits} mantissa bits. It stores ${show(encoded.value)}.`}
        >
          {runs.map((run) => {
            const x0 = run.start * cell;
            const x1 = (run.end + 1) * cell;
            const top = run.row * rowHeight;
            const tone = FIELD_TONE[run.field];
            return (
              <g key={`${run.field}-${run.row}`}>
                {runs.find((r) => r.field === run.field) === run && (
                  <text x={(x0 + x1) / 2} y={top + 11} textAnchor="middle" style={{ fill: tone }}>
                    {fieldLabel(run.field, run.end - run.start + 1)}
                  </text>
                )}
                <line x1={x0 + 1} x2={x1 - 1} y1={top + 16} y2={top + 16} style={{ stroke: tone }} />
              </g>
            );
          })}
          {bits.map((bit, i) => {
            const row = Math.floor(i / perRow);
            const col = i % perRow;
            const tone = FIELD_TONE[fieldOf(i)];
            const cx = col * cell;
            const cy = row * rowHeight + 20;
            return (
              <g key={i}>
                <rect
                  x={cx + 1}
                  y={cy}
                  width={cell - 2}
                  height={24}
                  rx={2}
                  style={{ fill: wash(tone, bit ? 34 : 8), stroke: tone, strokeWidth: 1 }}
                />
                <text
                  x={cx + cell / 2}
                  y={cy + 16}
                  textAnchor="middle"
                  style={{ fill: color.fg, fontFamily: "var(--font-mono)", fontSize: 11 }}
                >
                  {bit}
                </text>
              </g>
            );
          })}
          {lines.map((line, k) => (
            <text key={k} x={0} y={textTop + k * 17} style={{ fill: k === lines.length - 1 ? color.fg : color.muted }}>
              {line}
            </text>
          ))}
        </svg>
      </div>
    </Figure>
  );
}
