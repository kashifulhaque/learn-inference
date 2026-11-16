import { useState } from "react";
import { Figure, Legend, Slider, Stat, color, formatBytes, useWidth } from "../kit";

// Bytes one attention layer reads for one decode token, dense against a
// block-sparse step. Arithmetic from content/labs/07b-qwen4-blocks/solution.py,
// printed by scripts/figures/07b-qwen4-sparse-attention-and-gated-residuals.py.
//
// Geometry is the Qwen3.8-2.4T-A95B card: 4 KV heads, head width 256, bfloat16.
// The index width (64), the kept-token budget (4096), and the block sizes are
// this chapter's illustrative choices. Qwen hasn't published Qwen4's values.
// Not a measurement.
const KV_BYTES_PER_TOKEN = 2 * 4 * 256 * 2; // 4096 B per layer per token
const D_IDX = 64;
const IDX_BYTES = 2;
const KEPT_TOKENS = 4096;
const ATTENTION_LAYERS = 23;
const CONTEXTS = [32768, 131072, 262144, 1000000] as const;
const BLOCKS = [16, 32, 64, 128, 256, 512] as const;

const DECADES = [5, 6, 7, 8, 9, 10] as const; // 1e5 to 1e10 bytes

export default function SparseDecodeBytes() {
  const [context, setContext] = useState<number>(1000000);
  const [block, setBlock] = useState<number>(64);
  const [ref, width] = useWidth();

  const nBlocks = Math.ceil(context / block);
  const dense = context * KV_BYTES_PER_TOKEN;
  const indexer = nBlocks * D_IDX * IDX_BYTES;
  const kv = Math.min(KEPT_TOKENS, context) * KV_BYTES_PER_TOKEN;
  const sparse = indexer + kv;

  const left = 16;
  const right = 16;
  const top = 8;
  const innerW = Math.max(width - left - right, 80);
  const xOf = (bytes: number) => left + ((Math.log10(bytes) - 5) / 5) * innerW;
  const barH = 22;
  const rows = [
    { y: top, label: "Dense", segments: [{ bytes: dense, tone: color.c }] },
    {
      y: top + 62,
      label: "Sparse",
      segments: [
        { bytes: indexer, tone: color.d },
        { bytes: kv, tone: color.b },
      ],
    },
  ];
  const height = top + 62 + barH + 34;

  return (
    <Figure
      title="Sparse decode reads kept blocks plus an index scan"
      controls={
        <>
          <Slider label="Context" values={CONTEXTS} value={context} onChange={setContext} format={(v) => v.toLocaleString("en-US")} />
          <Slider label="Block size" values={BLOCKS} value={block} onChange={setBlock} format={(v) => `${v} tokens`} />
        </>
      }
      readout={
        <>
          <Stat label="Dense" value={formatBytes(dense, 2)} tone={color.c} />
          <Stat label="Index scan" value={formatBytes(indexer, 2)} tone={color.d} />
          <Stat label="Kept KV" value={formatBytes(kv, 2)} tone={color.b} />
          <Stat label="Dense / sparse" value={`${(dense / sparse).toFixed(1)}x`} />
          <Stat label="Over 23 layers" value={formatBytes(sparse * ATTENTION_LAYERS, 1)} />
        </>
      }
      caption="Bytes read by one full-attention layer for one decode token, on a log scale, with the Qwen3.8 card's 4 KV heads of width 256 in bfloat16. The kept budget (4096 tokens), the index width (64), and the block sizes are illustrative choices, not Qwen4's. The sparse bar splits its length in proportion to the two parts. The cache itself isn't smaller: every key is still stored."
    >
      <div ref={ref}>
        <svg width={width} height={height} role="img" aria-label="Dense and block-sparse bytes read per decode token, log scale">
          {DECADES.map((d) => (
            <g key={d}>
              <line x1={xOf(10 ** d)} x2={xOf(10 ** d)} y1={top} y2={height - 24} style={{ stroke: color.line }} />
              <text x={xOf(10 ** d)} y={height - 8} textAnchor="middle" fontSize={11} style={{ fill: color.muted }}>
                {`1e${d} B`}
              </text>
            </g>
          ))}
          {rows.map((row) => {
            return (
              <g key={row.label}>
                <text x={left} y={row.y + 14 - 4} fontSize={11} style={{ fill: color.muted }}>
                  {row.label}
                </text>
                {row.segments.map((segment, index) => {
                  const total = row.segments.reduce((sum, s) => sum + s.bytes, 0);
                  // Segments share one bar whose end is the total on the log axis.
                  const endX = xOf(total);
                  const startX = left;
                  const span = endX - startX;
                  const before = row.segments.slice(0, index).reduce((sum, s) => sum + s.bytes, 0);
                  const x0 = startX + (before / total) * span;
                  const w = Math.max((segment.bytes / total) * span, 1);
                  return (
                    <rect key={index} x={x0} y={row.y + 18} width={w} height={barH} style={{ fill: segment.tone }} />
                  );
                })}
              </g>
            );
          })}
        </svg>
        <Legend
          items={[
            { label: "Dense KV read", tone: color.c },
            { label: "Index scan", tone: color.d },
            { label: "Kept blocks", tone: color.b },
          ]}
        />
      </div>
    </Figure>
  );
}
