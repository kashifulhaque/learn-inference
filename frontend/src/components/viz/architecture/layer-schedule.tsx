// Chapter 8: the 64-layer hybrid stack. A layer is full attention when
// (ℓ + 1) mod 4 = 0, and runs the gated delta rule otherwise.

import { useState } from "react";
import { Figure, Legend, Slider, Stat, color, useWidth, wash } from "../kit";

// num_hidden_layers 64 and full_attention_interval 4, from chapter 8.
const LAYERS = 64;
const INTERVAL = 4;

const CELL_H = 24;
const GAP = 3;
const LABEL_W = 24;

function isFullAttention(layer: number): boolean {
  return (layer + 1) % INTERVAL === 0;
}

const FULL_COUNT = Array.from({ length: LAYERS }, (_, l) => l).filter(isFullAttention).length;
const LINEAR_COUNT = LAYERS - FULL_COUNT;

export default function LayerSchedule() {
  const [ref, width] = useWidth();
  const [layer, setLayer] = useState(3);

  // Four rows of 16 on a wide column, eight rows of 8 on a phone. Both keep
  // every fourth column full attention.
  const columns = width >= 480 ? 16 : 8;
  const rows = LAYERS / columns;
  const cellW = (width - LABEL_W - GAP * (columns - 1)) / columns;
  const height = rows * CELL_H + (rows - 1) * GAP + 2;
  const full = isFullAttention(layer);

  return (
    <Figure
      title="The hybrid layer schedule"
      controls={<Slider label="Layer ℓ" min={0} max={LAYERS - 1} value={layer} onChange={setLayer} />}
      readout={
        <>
          <Stat label="Layer ℓ" value={layer} />
          <Stat label="(ℓ + 1) mod 4" value={(layer + 1) % INTERVAL} />
          <Stat
            label="Mixer"
            value={full ? "Grouped-query attention" : "Gated delta rule"}
            tone={full ? color.info : undefined}
          />
          <Stat label="Keeps" value={full ? "A KV cache" : "A fixed-size state"} />
        </>
      }
      caption={
        <>
          Every fourth layer, where (ℓ + 1) mod 4 = 0, is full attention, and the rest run the gated delta
          rule. The last layer, 63, is full attention, and layers 0 to 2 aren't, so a bug in the full-attention
          path can't show up before layer 3.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={`A grid of ${LAYERS} layers. Layers 3, 7, 11, and every fourth layer up to 63 are grouped-query attention, ${FULL_COUNT} in all; the other ${LINEAR_COUNT} are gated delta rule layers. Layer ${layer} is highlighted: ${full ? "grouped-query attention" : "gated delta rule"}.`}
        >
          {Array.from({ length: rows }, (_, r) => (
            <text key={r} x={LABEL_W - 6} y={r * (CELL_H + GAP) + CELL_H / 2 + 5} textAnchor="end">
              {r * columns}
            </text>
          ))}
          {Array.from({ length: LAYERS }, (_, l) => {
            const r = Math.floor(l / columns);
            const c = l % columns;
            const x = LABEL_W + c * (cellW + GAP);
            const y = 1 + r * (CELL_H + GAP);
            const attention = isFullAttention(l);
            const on = l === layer;
            return (
              <g key={l}>
                <rect
                  x={x}
                  y={y}
                  width={cellW}
                  height={CELL_H}
                  rx={3}
                  style={{
                    fill: attention ? wash(color.info, on ? 45 : 22) : on ? wash(color.muted, 26) : color.well,
                    stroke: on ? color.fg : attention ? color.info : color.line,
                    strokeWidth: on ? 2 : 1,
                  }}
                />
                <text
                  x={x + cellW / 2}
                  y={y + CELL_H / 2 + 4}
                  textAnchor="middle"
                  style={{ fill: on || attention ? color.fg : color.subtle }}
                >
                  {l}
                </text>
              </g>
            );
          })}
        </svg>
        <Legend
          items={[
            { label: `Gated delta rule, ${LINEAR_COUNT} layers`, tone: color.lineStrong },
            { label: `Grouped-query attention, ${FULL_COUNT} layers`, tone: color.info },
          ]}
        />
      </div>
    </Figure>
  );
}
