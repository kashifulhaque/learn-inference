import { useState } from "react";
import { Figure, Slider, Stat, color, useWidth, wash } from "../kit";

// Simulated tick counts from chapter 16a. The script
// scripts/figures/16a-the-serving-loop-and-api.py prints the same ratios.
// This is not an A100 measurement.
const STEPS = 8;
const FORWARD = 10;
const POST = 2;

function clock(schedule: number, overlap: boolean) {
  const busy = STEPS * FORWARD;
  const leftover = Math.max(0, schedule - FORWARD);
  const idle = overlap
    ? schedule + (STEPS - 1) * leftover + STEPS * POST
    : STEPS * (schedule + POST);
  const wall = idle + busy;
  return { idle, wall, busy, fraction: wall === 0 ? 0 : idle / wall };
}

const fmt = (value: number) => value.toFixed(3);

export default function ScheduleOverlap() {
  const [schedule, setSchedule] = useState(6);
  const [ref, width] = useWidth();
  const serial = clock(schedule, false);
  const overlap = clock(schedule, true);
  const left = 72;
  const right = 16;
  const top = 8;
  const rowH = 28;
  const gap = 18;
  const height = top + rowH * 2 + gap + 28;
  const inner = Math.max(width - left - right, 40);
  const scale = inner / Math.max(serial.wall, overlap.wall, 1);

  const rows = [
    { label: "Serial", tone: color.c, ...serial },
    { label: "Overlap", tone: color.b, ...overlap },
  ];

  return (
    <Figure
      title="Overlap hides the schedule, not the sampling"
      controls={
        <Slider
          label="Schedule ticks"
          min={0}
          max={20}
          value={schedule}
          onChange={setSchedule}
          format={(value) => `${value} (forward ${FORWARD}, post ${POST}, ${STEPS} steps)`}
        />
      }
      readout={
        <>
          <Stat label="Serial idle" value={fmt(serial.fraction)} tone={color.c} />
          <Stat label="Overlap idle" value={fmt(overlap.fraction)} tone={color.b} />
          <Stat label="GPU busy ticks" value={String(serial.busy)} />
        </>
      }
      caption="Each bar is wall time. The solid part is the forward. The pale part is CPU time the GPU waits on. Schedule, forward, and post are simulated ticks."
    >
      <div ref={ref}>
        <svg width={width} height={height} role="img" aria-label="Serial and overlapped idle fractions on a simulated clock">
          {rows.map((row, index) => {
            const y = top + index * (rowH + gap);
            const busyW = row.busy * scale;
            const idleW = row.idle * scale;
            return (
              <g key={row.label}>
                <text x={0} y={y + 18} fill={color.fg} fontSize={13}>
                  {row.label}
                </text>
                <rect x={left} y={y} width={busyW} height={rowH} style={{ fill: color.b }} />
                <rect
                  x={left + busyW}
                  y={y}
                  width={idleW}
                  height={rowH}
                  style={{ fill: wash(row.tone, 55) }}
                />
                <text x={left + busyW + idleW + 6} y={y + 18} fill={color.muted} fontSize={12}>
                  {fmt(row.fraction)}
                </text>
              </g>
            );
          })}
          <text x={left} y={height - 6} fill={color.muted} fontSize={12}>
            {`S = ${schedule}, F = ${FORWARD}, P = ${POST}, n = ${STEPS}`}
          </text>
        </svg>
      </div>
    </Figure>
  );
}
