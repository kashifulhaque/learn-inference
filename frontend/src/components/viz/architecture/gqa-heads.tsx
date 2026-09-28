// Chapter 7: 24 query heads reading H_kv shared KV heads. Query head h reads
// KV head ⌊h/g⌋, and the cache stores only the KV heads.

import { useState } from "react";
import { Figure, Slider, Stat, color, formatBytes, useWidth, wash } from "../kit";

// Chapter 7's geometry table: num_attention_heads 24, head_dim 256, bfloat16
// (2 bytes), and 16 of 64 layers with full attention.
const QUERY_HEADS = 24;
const HEAD_DIM = 256;
const BYTES_PER_ELEMENT = 2;
const FULL_ATTENTION_LAYERS = 16;
// The context length of the chapter's cache table.
const CONTEXT = 32768;
// Every KV head count that divides 24: 1 is MQA, 24 is MHA, and this model
// uses 4.
const KV_HEAD_OPTIONS = [1, 2, 3, 4, 6, 8, 12, 24];

const HEIGHT = 178;
const QUERY_Y = 22;
const BOX_H = 16;
const KV_Y = 86;
const BAR_Y = 146;

/** Cache bytes per token over the full-attention layers: 2 × H_kv × d_h × b × 16. */
function cacheBytesPerToken(kvHeads: number): number {
  return 2 * kvHeads * HEAD_DIM * BYTES_PER_ELEMENT * FULL_ATTENTION_LAYERS;
}

/** 64 KiB, 1.5 GiB, 512 MiB: no trailing ".0". */
function bytes(value: number): string {
  return formatBytes(value, 1).replace(".0 ", " ");
}

function scheme(kvHeads: number): string {
  if (kvHeads === QUERY_HEADS) return "MHA";
  if (kvHeads === 1) return "MQA";
  return "GQA";
}

export default function GqaHeads() {
  const [ref, width] = useWidth();
  const [kvHeads, setKvHeads] = useState(4);
  const [head, setHead] = useState(13);

  const group = QUERY_HEADS / kvHeads;
  const kvHead = Math.floor(head / group);
  const slot = width / QUERY_HEADS;
  const boxW = Math.max(4, slot - 2);
  const queryX = (h: number) => h * slot + (slot - boxW) / 2;
  const perToken = cacheBytesPerToken(kvHeads);
  const mhaPerToken = cacheBytesPerToken(QUERY_HEADS);

  return (
    <Figure
      title="Query heads sharing KV heads"
      controls={
        <>
          <Slider label="KV heads" values={KV_HEAD_OPTIONS} value={kvHeads} onChange={setKvHeads} />
          <Slider label="Query head" min={0} max={QUERY_HEADS - 1} value={head} onChange={setHead} />
        </>
      }
      readout={
        <>
          <Stat label="Scheme" value={scheme(kvHeads)} />
          <Stat label="KV heads, H_kv" value={kvHeads} tone={color.b} />
          <Stat label="Group size g" value={group} />
          <Stat label={`Query head ${head} reads`} value={`KV head ${kvHead}`} tone={color.accent} />
          <Stat label="KV cache per token" value={bytes(perToken)} />
          <Stat label="At 32k context" value={bytes(perToken * CONTEXT)} />
        </>
      }
      caption={
        <>
          Each line is a query head reading its KV head, ⌊h/g⌋. The cache stores only the KV heads, so it shrinks
          with H_kv while all 24 query heads keep their own queries; the bar measures it against MHA, over the 16
          full-attention layers.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={HEIGHT}
          role="img"
          aria-label={`${QUERY_HEADS} query heads above ${kvHeads} KV heads, ${group} query heads to each. Query head ${head} reads KV head ${kvHead}. The KV cache is ${bytes(perToken)} per token against ${bytes(mhaPerToken)} for MHA.`}
        >
          <text x={0} y={12}>
            {QUERY_HEADS} query heads
          </text>
          <text x={width} y={12} textAnchor="end" style={{ fill: color.accent }}>
            head {head}
          </text>

          {Array.from({ length: QUERY_HEADS }, (_, h) => {
            const target = Math.floor(h / group);
            const on = h === head;
            const kvCenter = (target + 0.5) * group * slot;
            return (
              <g key={h}>
                <line
                  x1={queryX(h) + boxW / 2}
                  y1={QUERY_Y + BOX_H}
                  x2={kvCenter}
                  y2={KV_Y}
                  style={{ stroke: on ? color.accent : color.lineStrong, strokeWidth: on ? 2 : 1 }}
                />
                <rect
                  x={queryX(h)}
                  y={QUERY_Y}
                  width={boxW}
                  height={BOX_H}
                  rx={2}
                  style={{
                    fill: on ? color.accent : wash(color.muted, 18),
                    stroke: on ? color.accent : color.lineStrong,
                  }}
                />
              </g>
            );
          })}

          {Array.from({ length: kvHeads }, (_, j) => {
            const x = j * group * slot + 1;
            const w = group * slot - 2;
            const on = j === kvHead;
            return (
              <g key={j}>
                <rect
                  x={x}
                  y={KV_Y}
                  width={w}
                  height={BOX_H + 4}
                  rx={3}
                  style={{
                    fill: wash(color.b, on ? 34 : 18),
                    stroke: on ? color.accent : color.b,
                    strokeWidth: on ? 2 : 1,
                  }}
                />
                {w >= 34 && (
                  <text x={x + w / 2} y={KV_Y + BOX_H - 1} textAnchor="middle" style={{ fill: color.fg }}>
                    {w >= 56 ? `KV ${j}` : j}
                  </text>
                )}
              </g>
            );
          })}
          <text x={0} y={KV_Y + BOX_H + 20}>
            {kvHeads} KV {kvHeads === 1 ? "head" : "heads"}, {group} query {group === 1 ? "head" : "heads"} each
          </text>

          <rect x={0} y={BAR_Y} width={width} height={10} rx={2} style={{ fill: color.well, stroke: color.line }} />
          <rect
            x={0}
            y={BAR_Y}
            width={(width * perToken) / mhaPerToken}
            height={10}
            rx={2}
            style={{ fill: color.b }}
          />
          <text x={0} y={BAR_Y + 26}>
            KV cache: {bytes(perToken)} per token, of MHA's {bytes(mhaPerToken)}
          </text>
        </svg>
      </div>
    </Figure>
  );
}
