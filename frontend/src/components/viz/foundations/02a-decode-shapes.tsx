import { useState } from "react";
import { Figure, Legend, Segmented, Slider, Stat, color, formatBytes, formatCount, useWidth } from "../kit";

// Chapter 2a, "Pricing a shape": the lab's inference_cost_per_token on
// candidate shapes, from chapter 2's formulas.
type Shape = { name: string; kv: number; interval: number; vocab: number };
const BASE = { d: 5120, layers: 64, kv: 4, interval: 4, vocab: 248_320 };

const D = 5120;
const FF = 17408;
const Q_WIDTH = 24 * 256;
const attnFull = (kv: number) => D * 2 * Q_WIDTH + 2 * D * kv * 256 + Q_WIDTH * D;
const ATTN_LINEAR = D * (2 * 16 * 128 + 2 * 48 * 128) + (2 * 16 * 128 + 48 * 128) * 4 + D * 96 + 48 * 128 * D;
const STATE_PER_LINEAR_LAYER = 48 * 128 * 128 * 4 + (2 * 16 * 128 + 48 * 128) * 4 * 2;

const fullLayers = (interval: number) => Math.floor(BASE.layers / interval);

function cost(s: Shape, batch: number, context: number) {
  const full = fullLayers(s.interval);
  const linear = BASE.layers - full;
  const params =
    full * attnFull(s.kv) + linear * ATTN_LINEAR + BASE.layers * 3 * D * FF + 2 * s.vocab * D + (BASE.layers * 2 + 1) * D;
  const kvPerToken = 2 * s.kv * 256 * 2 * full;
  return { params, weights: (params * 2) / batch, kv: kvPerToken * context, state: 2 * STATE_PER_LINEAR_LAYER * linear };
}

const SHAPES: readonly Shape[] = [
  { name: "3:1 hybrid, 4 KV heads (this model)", kv: 4, interval: 4, vocab: BASE.vocab },
  { name: "3:1 hybrid, 24 KV heads", kv: 24, interval: 4, vocab: BASE.vocab },
  { name: "3:1 hybrid, 8 KV heads", kv: 8, interval: 4, vocab: BASE.vocab },
  { name: "3:1 hybrid, 2 KV heads", kv: 2, interval: 4, vocab: BASE.vocab },
  { name: "1:1 hybrid, 4 KV heads", kv: 4, interval: 2, vocab: BASE.vocab },
  { name: "15:1 hybrid, 4 KV heads", kv: 4, interval: 16, vocab: BASE.vocab },
  { name: "All full attention, 4 KV heads", kv: 4, interval: 1, vocab: BASE.vocab },
  { name: "All full attention, 24 KV heads", kv: 24, interval: 1, vocab: BASE.vocab },
  { name: "3:1 hybrid, 128k vocabulary", kv: 4, interval: 4, vocab: 128_000 },
];

const BATCHES = [1, 8, 32, 128] as const;
const CONTEXTS = [1024, 4096, 32768, 131072] as const;
const ROW = 26;
const LABEL = 214;

export default function DecodeShapes() {
  const [ref, width] = useWidth();
  const [batch, setBatch] = useState<number>(32);
  const [context, setContext] = useState<number>(32768);

  const rows = SHAPES.map((s) => ({ s, c: cost(s, batch, context) }))
    .map((r) => ({ ...r, total: r.c.weights + r.c.kv + r.c.state }))
    .sort((a, b) => a.total - b.total);
  const max = Math.max(...rows.map((r) => r.total));
  const barSpan = Math.max(60, width - LABEL - 70);
  const px = (v: number) => (v / max) * barSpan;
  const height = rows.length * ROW + 8;
  const base = rows.find((r) => r.s.kv === 4 && r.s.interval === 4 && r.s.vocab === BASE.vocab)!;

  return (
    <Figure
      title="Bytes moved per decoded token, by shape"
      controls={
        <>
          <Slider label="Batch" values={BATCHES} value={batch} onChange={setBatch} />
          <Segmented
            label="Context"
            value={context}
            options={CONTEXTS.map((c) => ({ value: c, label: formatCount(c, true) }))}
            onChange={setContext}
          />
        </>
      }
      readout={
        <>
          <Stat label="This model" value={formatBytes(base.total, 2)} tone={color.accent} />
          <Stat label="Cheapest" value={rows[0].s.name} />
          <Stat label="Costliest" value={rows[rows.length - 1].s.name} />
        </>
      }
      caption={
        <>
          Bytes per token are the weights divided by the batch, plus one sequence&apos;s KV read, plus its recurrent state
          read and written. At batch 1 the weights are almost everything. At 32 sequences of 32k tokens the KV read takes
          over, and the all-attention shapes fall to the bottom.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={`Bar chart of decode bytes per token for nine shapes at batch ${batch} and context ${formatCount(context)}. Cheapest first: ${rows
            .map((r) => `${r.s.name}, ${formatBytes(r.total, 2)}`)
            .join("; ")}.`}
        >
          {rows.map((r, i) => {
            const yy = 4 + i * ROW;
            const w1 = px(r.c.weights);
            const w2 = px(r.c.kv);
            const w3 = px(r.c.state);
            const isBase = r === base;
            return (
              <g key={r.s.name}>
                <text x={LABEL - 6} y={yy + 15} textAnchor="end" style={{ fill: isBase ? color.fg : color.muted }}>
                  {r.s.name}
                </text>
                <rect x={LABEL} y={yy + 3} width={w1} height={16} style={{ fill: color.muted }} />
                <rect x={LABEL + w1} y={yy + 3} width={w2} height={16} style={{ fill: color.accent }} />
                <rect x={LABEL + w1 + w2} y={yy + 3} width={w3} height={16} style={{ fill: color.c }} />
                <text x={LABEL + w1 + w2 + w3 + 5} y={yy + 15} style={{ fill: color.fg }}>
                  {formatBytes(r.total, 2)}
                </text>
              </g>
            );
          })}
        </svg>
        <div style={{ marginTop: "0.5rem" }}>
          <Legend
            items={[
              { label: "Weights, shared over the batch", tone: color.muted },
              { label: "KV read", tone: color.accent },
              { label: "State read and written", tone: color.c },
            ]}
          />
        </div>
      </div>
    </Figure>
  );
}
