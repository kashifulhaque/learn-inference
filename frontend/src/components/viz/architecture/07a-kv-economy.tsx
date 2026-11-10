// Chapter 7a: KV cache bytes for one sequence under five attention designs on
// this model's geometry. The numbers are config arithmetic from
// scripts/figures/07a-kv-economy-attention-variants.py, not A100 measurements.
// The MLA bar is hypothetical: the model has no MLA layers.

import { useState } from "react";
import { Figure, Slider, Stat, color, formatBytes, useWidth, wash } from "../kit";

// Chapter 7's geometry: 24 query heads, head_dim 256, bfloat16 (2 bytes), and
// 16 of 64 layers with full attention.
const LAYERS = 16;
const HEAD_DIM = 256;
const BYTES = 2;
// Per-layer-per-token GQA bytes: 2 × 4 KV heads × 256 × 2 = 4096.
const GQA_LAYER_TOKEN = 2 * 4 * HEAD_DIM * BYTES;
// MHA with 24 KV heads.
const MHA_LAYER_TOKEN = 2 * 24 * HEAD_DIM * BYTES;
// MLA at DeepSeek-V2's proportions on this head size: d_c = 4 d_h, d_rope = d_h / 2.
const MLA_LAYER_TOKEN = (4 * HEAD_DIM + HEAD_DIM / 2) * BYTES;
// The rolling window and its sinks.
const WINDOW = 4096;
const SINKS = 4;

const CONTEXTS = [4096, 8192, 32768, 131072] as const;

type Row = { key: string; label: string; bytes: number; tone: string };

function rows(context: number): Row[] {
  const global = (LAYERS / 2) * GQA_LAYER_TOKEN * context;
  const local = (LAYERS / 2) * GQA_LAYER_TOKEN * Math.min(context, WINDOW);
  return [
    { key: "mha", label: "MHA, 24 KV heads", bytes: LAYERS * MHA_LAYER_TOKEN * context, tone: color.muted },
    { key: "gqa", label: "GQA, 4 KV heads (this model)", bytes: LAYERS * GQA_LAYER_TOKEN * context, tone: color.b },
    { key: "alt", label: "Half local, half global", bytes: global + local, tone: color.c },
    { key: "mla", label: "MLA, V2 proportions", bytes: LAYERS * MLA_LAYER_TOKEN * context, tone: color.a },
    {
      key: "win",
      label: "Window 4096 plus 4 sinks",
      bytes: LAYERS * GQA_LAYER_TOKEN * Math.min(context, WINDOW + SINKS),
      tone: color.d,
    },
  ];
}

function bytesText(value: number): string {
  return formatBytes(value, 1).replace(".0 ", " ");
}

export default function KvEconomy() {
  const [context, setContext] = useState(32768);
  const [ref, width] = useWidth();
  const data = rows(context);
  const biggest = Math.max(...data.map((row) => row.bytes));
  const labelW = Math.min(190, Math.max(120, width * 0.38));
  const valueW = 70;
  const barMax = Math.max(width - labelW - valueW, 40);
  const rowH = 30;
  const height = rowH * data.length + 8;
  const gqa = data[1].bytes;

  return (
    <Figure
      title="KV cache bytes for one sequence"
      controls={
        <Slider
          label="Context length"
          values={CONTEXTS}
          value={context}
          onChange={setContext}
          format={(value) => `${value / 1024}k`}
        />
      }
      readout={
        <>
          <Stat label="GQA (this model)" value={bytesText(gqa)} tone={color.b} />
          <Stat label="MLA, V2 proportions" value={bytesText(data[3].bytes)} tone={color.a} />
          <Stat label="Window plus sinks" value={bytesText(data[4].bytes)} tone={color.d} />
          <Stat label="MLA over GQA" value={`${(data[3].bytes / gqa).toFixed(3)}`} />
          <Stat label="Window over GQA" value={`${(data[4].bytes / gqa).toFixed(3)}`} />
        </>
      }
      caption="Bars share one scale, so the window's bar stops growing once the context passes 4,100 tokens while the others keep climbing. MLA is a hypothetical for this model, with the latent at four head widths and a RoPE key of half a head width. All values are arithmetic over the 16 full-attention layers."
    >
      <div ref={ref}>
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={`KV cache bytes at ${context} tokens: ${data.map((row) => `${row.label} ${bytesText(row.bytes)}`).join("; ")}.`}
        >
          {data.map((row, index) => {
            const y = index * rowH + 4;
            const w = Math.max((row.bytes / biggest) * barMax, 2);
            return (
              <g key={row.key}>
                <text x={0} y={y + 17} fontSize={12} style={{ fill: color.fg }}>
                  {row.label}
                </text>
                <rect
                  x={labelW}
                  y={y + 4}
                  width={w}
                  height={rowH - 12}
                  rx={2}
                  style={{ fill: wash(row.tone, 55), stroke: row.tone }}
                />
                <text x={labelW + w + 6} y={y + 17} fontSize={12} style={{ fill: color.muted }}>
                  {bytesText(row.bytes)}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </Figure>
  );
}
