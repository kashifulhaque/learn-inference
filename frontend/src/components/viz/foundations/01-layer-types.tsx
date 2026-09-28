import { useState } from "react";
import { Figure, Legend, Segmented, Stat, color, useWidth, wash } from "../kit";

// Chapter 1, "The config that matters": num_hidden_layers is 64 and
// full_attention_interval is 4. The checkpoint's self_attn tensors sit on
// layers 3, 7, 11, and so on, which is what (i + 1) % 4 == 0 produces.
const LAYERS = 64;
const INTERVAL = 4;
const checkpointIsFull = (i: number) => (i + 1) % INTERVAL === 0;

const RULES = {
  right: { label: "(i + 1) % 4 == 0", isFull: checkpointIsFull },
  offByOne: { label: "i % 4 == 0", isFull: (i: number) => i % INTERVAL === 0 },
} as const;
type Rule = keyof typeof RULES;

const GAP = 3;
const ROW_GAP = 6;

function listLayers(layers: number[]): string {
  if (layers.length <= 4) return layers.join(", ");
  return `${layers.slice(0, 3).join(", ")}, …, ${layers[layers.length - 1]}`;
}

export default function LayerTypes() {
  const [ref, width] = useWidth();
  const [rule, setRule] = useState<Rule>("right");
  const isFull = RULES[rule].isFull;

  const columns = width >= 700 ? 32 : width >= 400 ? 16 : 8;
  const rows = LAYERS / columns;
  const cell = (width - GAP * (columns - 1)) / columns;
  const cellH = 26;
  const height = rows * cellH + (rows - 1) * ROW_GAP + 2;

  const layers = Array.from({ length: LAYERS }, (_, i) => i);
  const full = layers.filter(isFull);
  const wrong = layers.filter((i) => isFull(i) !== checkpointIsFull(i));

  return (
    <Figure
      title="Which layers are full attention"
      controls={
        <Segmented
          label="layer_types rule"
          value={rule}
          onChange={setRule}
          options={[
            { value: "right", label: RULES.right.label },
            { value: "offByOne", label: RULES.offByOne.label },
          ]}
        />
      }
      readout={
        <>
          <Stat label="Rule" value={<code>{RULES[rule].label}</code>} />
          <Stat label="Full-attention layers" value={listLayers(full)} tone={color.b} />
          <Stat label="Full / linear" value={`${full.length} / ${LAYERS - full.length}`} />
          <Stat
            label="Disagree with the tensor names"
            value={`${wrong.length} of ${LAYERS}`}
            tone={wrong.length ? color.bad : color.ok}
          />
        </>
      }
      caption={
        <>
          Each cell is one layer, numbered from 0. With <code>(i + 1) % 4</code>, full attention lands on the last
          layer of each group of four, where the checkpoint has its <code>self_attn</code> tensors. The off-by-one
          rule still counts 16 full layers, but every one of them is in the wrong place.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={`A grid of the 64 layers. The rule ${RULES[rule].label} marks layers ${listLayers(full)} as full attention; ${wrong.length} layers disagree with the checkpoint.`}
        >
          {layers.map((i) => {
            const row = Math.floor(i / columns);
            const col = i % columns;
            const x = col * (cell + GAP);
            const y = row * (cellH + ROW_GAP) + 1;
            const on = isFull(i);
            const bad = on !== checkpointIsFull(i);
            return (
              <g key={i}>
                <rect
                  x={x}
                  y={y}
                  width={cell}
                  height={cellH}
                  rx={3}
                  strokeDasharray={bad ? "3 2" : undefined}
                  style={{
                    fill: on ? wash(color.b, 30) : color.well,
                    stroke: bad ? color.bad : on ? color.b : color.line,
                    strokeWidth: bad ? 1.5 : 1,
                  }}
                />
                <text
                  x={x + cell / 2}
                  y={y + 17}
                  textAnchor="middle"
                  style={{ fill: on ? color.fg : color.subtle }}
                >
                  {i}
                </text>
              </g>
            );
          })}
        </svg>
        <div style={{ marginTop: "0.6rem" }}>
          <Legend
            items={[
              { label: "Full attention (self_attn)", tone: color.b },
              { label: "Linear attention (linear_attn)", tone: color.lineStrong },
              { label: "Disagrees with the checkpoint", tone: color.bad, dashed: true },
            ]}
          />
        </div>
      </div>
    </Figure>
  );
}
