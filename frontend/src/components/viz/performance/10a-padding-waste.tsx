import { useState } from "react";
import { Figure, Legend, Slider, Stat, color, useWidth } from "../kit";

// 200 decode steps. Counts, pads, and idle totals are the sums in
// scripts/figures/10a-cuda-graphs-and-launch-overhead.py.
// Launch cost is not drawn here. It is chapter 10's 5 µs, not a new measurement.
const ROWS = [
  { n: 1, count: 40, pow2: 1, chosen: 1 },
  { n: 2, count: 20, pow2: 2, chosen: 2 },
  { n: 3, count: 20, pow2: 4, chosen: 4 },
  { n: 5, count: 30, pow2: 8, chosen: 6 },
  { n: 6, count: 24, pow2: 8, chosen: 6 },
  { n: 7, count: 12, pow2: 8, chosen: 8 },
  { n: 8, count: 10, pow2: 8, chosen: 8 },
  { n: 9, count: 16, pow2: 16, chosen: 12 },
  { n: 10, count: 12, pow2: 16, chosen: 12 },
  { n: 12, count: 8, pow2: 16, chosen: 12 },
  { n: 13, count: 4, pow2: 16, chosen: 16 },
  { n: 15, count: 4, pow2: 16, chosen: 16 },
] as const;

const SIZES = ROWS.map((row) => row.n);

function idle(key: "pow2" | "chosen") {
  let slots = 0;
  let padded = 0;
  for (const row of ROWS) {
    slots += (row[key] - row.n) * row.count;
    padded += row[key] * row.count;
  }
  return { slots, padded };
}

const POW2_IDLE = idle("pow2");
const CHOSEN_IDLE = idle("chosen");

export default function PaddingWaste() {
  const [n, setN] = useState(5);
  const [ref, width] = useWidth();
  const row = ROWS.find((item) => item.n === n) ?? ROWS[3];
  const left = 36;
  const right = 12;
  const top = 12;
  const bottom = 28;
  const height = 220;
  const innerW = Math.max(width - left - right, 40);
  const innerH = height - top - bottom;
  const maxIdle = 8;
  const group = innerW / ROWS.length;
  const bar = Math.min(14, group * 0.28);
  const yOf = (slots: number) => top + innerH - (slots / maxIdle) * innerH;
  const yTicks = [0, 2, 4, 6, 8];

  return (
    <Figure
      title="Tighter buckets leave fewer idle slots"
      controls={
        <Slider
          label="Batch size"
          values={SIZES}
          value={n}
          onChange={setN}
          format={(value) => `${value}`}
        />
      }
      readout={
        <>
          <Stat label="Power-of-two pad" value={row.pow2} tone={color.b} />
          <Stat label="Captured pad" value={row.chosen} tone={color.c} />
          <Stat label="Idle, powers of two" value={`${POW2_IDLE.slots} / ${POW2_IDLE.padded}`} tone={color.b} />
          <Stat label="Idle, captured sizes" value={`${CHOSEN_IDLE.slots} / ${CHOSEN_IDLE.padded}`} tone={color.c} />
        </>
      }
      caption="Each bar is the idle slots in one step, the pad minus the real batch. The totals count every step in the 200-step trace. Powers of two leave 402 idle slots out of 1,472. The captured sizes leave 150 out of 1,220."
    >
      <div ref={ref}>
        <svg
          width={width}
          height={height}
          role="img"
          aria-label="Idle padded slots per step for powers of two and for the captured batch sizes"
        >
          {yTicks.map((tick) => (
            <g key={tick}>
              <line
                x1={left}
                x2={left + innerW}
                y1={yOf(tick)}
                y2={yOf(tick)}
                style={{ stroke: color.line }}
              />
              <text x={left - 6} y={yOf(tick) + 4} textAnchor="end" style={{ fill: color.muted, fontSize: 11 }}>
                {tick}
              </text>
            </g>
          ))}
          {ROWS.map((item, index) => {
            const cx = left + group * index + group / 2;
            const on = item.n === row.n;
            return (
              <g key={item.n}>
                <rect
                  x={cx - bar - 1}
                  y={yOf(item.pow2 - item.n)}
                  width={bar}
                  height={Math.max(0, yOf(0) - yOf(item.pow2 - item.n))}
                  style={{ fill: color.b, opacity: on ? 1 : 0.45 }}
                />
                <rect
                  x={cx + 1}
                  y={yOf(item.chosen - item.n)}
                  width={bar}
                  height={Math.max(0, yOf(0) - yOf(item.chosen - item.n))}
                  style={{ fill: color.c, opacity: on ? 1 : 0.7 }}
                />
                <text x={cx} y={height - 8} textAnchor="middle" style={{ fill: color.muted, fontSize: 11 }}>
                  {item.n}
                </text>
              </g>
            );
          })}
        </svg>
        <Legend
          items={[
            { label: "Powers of two", tone: color.b },
            { label: "Captured sizes", tone: color.c },
          ]}
        />
      </div>
    </Figure>
  );
}
