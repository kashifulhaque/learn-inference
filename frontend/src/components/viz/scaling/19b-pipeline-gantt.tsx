import { useState } from "react";
import { Figure, Slider, Stat, color, useWidth, wash } from "../kit";

// Chapter 19b: a fill-and-drain pipeline schedule. Stage s runs micro-batch
// t - s in slot t, as in `pipeline_schedule` in
// content/labs/19b-pipeline-and-ring/solution.py. The bubble is
// (p - 1) / (m + p - 1), checked against the table by the lab.
const STAGES = 4;
const COUNTS = [1, 2, 4, 8, 16] as const;
const TONES = [color.info, color.violet, color.warn, color.ok, color.accent] as const;

export default function PipelineGantt() {
  const [m, setM] = useState<number>(4);
  const [ref, width] = useWidth();
  const slots = m + STAGES - 1;
  const left = 56;
  const top = 8;
  const rowH = 30;
  const cellW = Math.max(6, Math.min(40, (width - left - 8) / slots));
  const height = top + STAGES * rowH + 8;
  const bubble = (STAGES - 1) / slots;

  return (
    <Figure
      title="Pipeline bubble against micro-batches"
      controls={<Slider label="Micro-batches" values={COUNTS} value={m} onChange={setM} />}
      readout={
        <>
          <Stat label="Slots" value={String(slots)} />
          <Stat label="Idle slots" value={`${STAGES * slots - STAGES * m} of ${STAGES * slots}`} />
          <Stat label="Bubble" value={`${STAGES - 1}/${slots} = ${bubble.toFixed(4)}`} tone={color.c} />
        </>
      }
      caption="Four stages, one time unit per stage per micro-batch. Colored cells are work and empty cells are the bubble. Computed from the schedule, not measured on a GPU."
    >
      <div ref={ref}>
        <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img"
          aria-label={`Gantt table for four stages and ${m} micro-batches. The bubble fraction is ${bubble.toFixed(4)}.`}>
          {Array.from({ length: STAGES }, (_, s) => (
            <g key={s}>
              <text x={left - 8} y={top + s * rowH + rowH / 2 + 4} textAnchor="end" style={{ fill: color.muted }}>
                stage {s}
              </text>
              {Array.from({ length: slots }, (_, t) => {
                const mb = t - s;
                const work = mb >= 0 && mb < m;
                return (
                  <rect
                    key={t}
                    x={left + t * cellW + 1}
                    y={top + s * rowH + 1}
                    width={cellW - 2}
                    height={rowH - 2}
                    rx={3}
                    style={{
                      fill: work ? wash(TONES[mb % TONES.length], 55) : "transparent",
                      stroke: work ? TONES[mb % TONES.length] : color.line,
                    }}
                  />
                );
              })}
            </g>
          ))}
        </svg>
      </div>
    </Figure>
  );
}
