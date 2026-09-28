import { useState } from "react";
import { Figure, Segmented, Stat, color, useWidth, wash } from "../kit";

// Chapter 1, "The safetensors format" and "Reading the architecture from the
// names": layer 3's four full-attention projections, all BF16 (2 bytes per
// element). The figure lays them out as a one-layer safetensors file, tensors
// in name order, which puts k_proj at offset 0 as in the chapter's example.
const PREFIX = "model.language_model.layers.3.self_attn.";
const BYTES_PER_ELEMENT = 2;
const TENSORS = [
  { key: "k_proj", shape: [1024, 5120] },
  { key: "o_proj", shape: [5120, 6144] },
  { key: "q_proj", shape: [12288, 5120] },
  { key: "v_proj", shape: [1024, 5120] },
] as const;
type Key = (typeof TENSORS)[number]["key"];

const LAYOUT = (() => {
  let begin = 0;
  const entries = TENSORS.map((t) => {
    const length = t.shape[0] * t.shape[1] * BYTES_PER_ELEMENT;
    const entry = { ...t, begin, end: begin + length, length };
    begin += length;
    return entry;
  });
  const header: Record<string, unknown> = { __metadata__: { format: "pt" } };
  for (const e of entries) {
    header[PREFIX + e.key + ".weight"] = { dtype: "BF16", shape: [...e.shape], data_offsets: [e.begin, e.end] };
  }
  // N is the UTF-8 length of the header written as compact JSON.
  const headerBytes = new TextEncoder().encode(JSON.stringify(header)).length;
  return { entries, headerBytes, dataBytes: begin };
})();

const HEIGHT = 128;
const BAR_Y = 42;
const BAR_H = 34;
// The prefix and header are a few hundred bytes against 210 MB of tensors, so
// they're drawn at a fixed width instead of to scale.
const PREFIX_W = 30;
const HEADER_W = 58;

const fmt = (n: number) => n.toLocaleString("en-US");

export default function SafetensorsOffsets() {
  const [ref, width] = useWidth();
  const [key, setKey] = useState<Key>("k_proj");
  const { entries, headerBytes, dataBytes } = LAYOUT;
  const chosen = entries.find((e) => e.key === key)!;
  const absolute = 8 + headerBytes + chosen.begin;

  const dataX = PREFIX_W + HEADER_W;
  const dataW = width - dataX - 1;
  const xOf = (offset: number) => dataX + (offset / dataBytes) * dataW;
  const cx0 = xOf(chosen.begin);
  const cx1 = xOf(chosen.end);
  const labelX = Math.min(Math.max((cx0 + cx1) / 2, 40), width - 40);

  return (
    <Figure
      title="Where a tensor's bytes sit in a safetensors file"
      controls={
        <Segmented
          label="Tensor"
          value={key}
          onChange={setKey}
          options={TENSORS.map((t) => ({ value: t.key, label: t.key }))}
        />
      }
      readout={
        <>
          <Stat label="Length prefix" value="8 bytes" tone={color.a} />
          <Stat label="Header N" value={`${fmt(headerBytes)} bytes`} tone={color.b} />
          <Stat label={`${key} begin`} value={fmt(chosen.begin)} tone={color.c} />
          <Stat label="Absolute offset, 8 + N + begin" value={fmt(absolute)} />
          <Stat
            label="end − begin"
            value={`${fmt(chosen.length)} = ${chosen.shape[0]} × ${chosen.shape[1]} × 2`}
          />
        </>
      }
      caption={
        <>
          A file holding only layer 3's four attention projections, with its header written as compact JSON.{" "}
          <code>data_offsets</code> count from the end of the header, so a tensor's first byte is at 8 + N + begin.
          The prefix and header are drawn wider than their real share.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={HEIGHT}
          role="img"
          aria-label={`A safetensors file: an 8-byte length prefix, a ${headerBytes}-byte JSON header, then ${fmt(dataBytes)} bytes of tensor data. ${key} occupies bytes ${fmt(chosen.begin)} to ${fmt(chosen.end)} of the data, so its first byte is at file offset ${fmt(absolute)}.`}
        >
          {/* Prefix and header. */}
          <rect x={0.5} y={BAR_Y} width={PREFIX_W - 1} height={BAR_H} style={{ fill: wash(color.a, 22), stroke: color.a }} />
          <text x={PREFIX_W / 2} y={BAR_Y + 21} textAnchor="middle" style={{ fill: color.a }}>
            8
          </text>
          <rect
            x={PREFIX_W + 0.5}
            y={BAR_Y}
            width={HEADER_W - 1}
            height={BAR_H}
            style={{ fill: wash(color.b, 18), stroke: color.b }}
          />
          <text x={PREFIX_W + HEADER_W / 2} y={BAR_Y + 21} textAnchor="middle" style={{ fill: color.b }}>
            N
          </text>

          {/* Tensor data, to scale. */}
          {entries.map((e) => {
            const on = e.key === key;
            const x0 = xOf(e.begin);
            const w = xOf(e.end) - x0;
            return (
              <g key={e.key}>
                <rect
                  x={x0 + 0.5}
                  y={BAR_Y}
                  width={Math.max(1, w - 1)}
                  height={BAR_H}
                  style={{ fill: on ? wash(color.c, 30) : color.well, stroke: on ? color.c : color.lineStrong }}
                />
                {w >= 44 && (
                  <text x={x0 + w / 2} y={BAR_Y + 21} textAnchor="middle" style={{ fill: on ? color.c : color.subtle }}>
                    {e.key.replace("_proj", "")}
                  </text>
                )}
              </g>
            );
          })}

          {/* The chosen tensor's name above it. */}
          <text x={labelX} y={BAR_Y - 8} textAnchor="middle" style={{ fill: color.c, fontWeight: 600 }}>
            {key}
          </text>

          {/* Brace from the file's first byte to the tensor's first byte. */}
          <line x1={0.5} x2={0.5} y1={BAR_Y + BAR_H + 4} y2={BAR_Y + BAR_H + 16} style={{ stroke: color.fg }} />
          <line x1={cx0} x2={cx0} y1={BAR_Y + BAR_H + 4} y2={BAR_Y + BAR_H + 16} style={{ stroke: color.fg }} />
          <line x1={0.5} x2={cx0} y1={BAR_Y + BAR_H + 10} y2={BAR_Y + BAR_H + 10} style={{ stroke: color.fg }} />
          <text x={Math.max(cx0, 150)} y={BAR_Y + BAR_H + 30} textAnchor="end" style={{ fill: color.fg }}>
            {`byte ${fmt(absolute)} of the file`}
          </text>
        </svg>
      </div>
    </Figure>
  );
}
