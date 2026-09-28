import { Figure, Legend, Stat, StepControls, color, useStepper, useWidth, wash } from "../kit";

// The workload from chapter 16, "Four requests, two ways": two batch slots and
// four requests queued A, B, C, D. A, C, and D each need 2 output tokens and B
// needs 6; each step produces one token per sequence, and prefill is ignored.
const SLOTS = 2;
const QUEUE = [
  { id: "A", tokens: 2 },
  { id: "B", tokens: 6 },
  { id: "C", tokens: 2 },
  { id: "D", tokens: 2 },
] as const;
const TONE: Record<string, string> = { A: color.a, B: color.b, C: color.c, D: color.d };

type Cell = { id: string; finishes: boolean } | null;
/** One schedule: grid[slot][step]. */
type Grid = Cell[][];

/** Static batching: fill every slot, run until all members finish, repeat. */
function staticSchedule(): Grid {
  const grid: Grid = Array.from({ length: SLOTS }, () => []);
  for (let start = 0; start < QUEUE.length; start += SLOTS) {
    const batch = QUEUE.slice(start, start + SLOTS);
    const length = Math.max(...batch.map((r) => r.tokens));
    for (let slot = 0; slot < SLOTS; slot++) {
      const r = batch[slot];
      for (let t = 0; t < length; t++) {
        grid[slot].push(r && t < r.tokens ? { id: r.id, finishes: t === r.tokens - 1 } : null);
      }
    }
  }
  return grid;
}

/** Continuous batching: after every step, a free slot takes the next waiting request. */
function continuousSchedule(): Grid {
  const grid: Grid = Array.from({ length: SLOTS }, () => []);
  const waiting = [...QUEUE];
  const slots: ({ id: string; left: number } | undefined)[] = [];
  while (waiting.length || slots.some(Boolean)) {
    for (let s = 0; s < SLOTS; s++) {
      if (!slots[s] && waiting.length) {
        const r = waiting.shift()!;
        slots[s] = { id: r.id, left: r.tokens };
      }
    }
    for (let s = 0; s < SLOTS; s++) {
      const r = slots[s];
      if (!r) {
        grid[s].push(null);
        continue;
      }
      r.left -= 1;
      grid[s].push({ id: r.id, finishes: r.left === 0 });
      if (r.left === 0) slots[s] = undefined;
    }
  }
  return grid;
}

const STATIC = staticSchedule();
const CONTINUOUS = continuousSchedule();
const STEPS = Math.max(STATIC[0].length, CONTINUOUS[0].length);

function totals(grid: Grid) {
  const steps = grid[0].length;
  const useful = grid.flat().filter(Boolean).length;
  const startOf = (id: string) => 1 + Math.min(...grid.map((row) => row.findIndex((c) => c?.id === id)).filter((i) => i >= 0));
  return { steps, paid: steps * SLOTS, useful, startOf };
}
const S = totals(STATIC);
const C = totals(CONTINUOUS);

/** What happens at each step, in words. */
function describe(step: number): string {
  const events = (grid: Grid) => {
    if (step >= grid[0].length) return "done";
    const cells = grid.map((row) => row[step]);
    const starts = cells.filter((c, s) => c && (step === 0 || grid[s][step - 1]?.id !== c.id)).map((c) => c!.id);
    const ends = cells.filter((c) => c?.finishes).map((c) => c!.id);
    const empty = cells.filter((c) => !c).length;
    const parts = [];
    if (starts.length) parts.push(`${starts.join(", ")} start${starts.length === 1 ? "s" : ""}`);
    if (ends.length) parts.push(`${ends.join(", ")} finish${ends.length === 1 ? "es" : ""}`);
    if (empty) parts.push(`${empty} slot empty`);
    return parts.join(", ") || "decoding";
  };
  return `static: ${events(STATIC)}; continuous: ${events(CONTINUOUS)}`;
}

const HEIGHT = 196;

export default function BatchTimeline() {
  const stepper = useStepper(STEPS, 900);
  const [ref, width] = useWidth();
  const current = stepper.step;

  const labelW = width < 420 ? 44 : 60;
  const cellW = (width - labelW) / STEPS;
  const rowH = 26;
  const headerY = 12;

  const lanes = [
    { title: `Static batching, ${S.steps} steps`, grid: STATIC, top: 26 },
    { title: `Continuous batching, ${C.steps} steps`, grid: CONTINUOUS, top: 26 + 18 + SLOTS * rowH + 14 },
  ];

  return (
    <Figure
      title="Static and continuous batching, step by step"
      controls={<StepControls stepper={stepper} count={STEPS} describe={(s) => describe(s)} />}
      readout={
        <>
          <Stat label="Static: useful slot-steps" value={`${S.useful} of ${S.paid} (${Math.round((100 * S.useful) / S.paid)}%)`} />
          <Stat label="Continuous: useful slot-steps" value={`${C.useful} of ${C.paid} (${Math.round((100 * C.useful) / C.paid)}%)`} />
          <Stat label="C starts at step" value={`${S.startOf("C")} static, ${C.startOf("C")} continuous`} />
        </>
      }
      caption={
        <>
          In the static schedule, slot 1 sits empty from step 3 to step 6 because B holds the batch open. The
          continuous schedule gives each freed slot to the next request on the very next step.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={HEIGHT}
          viewBox={`0 0 ${width} ${HEIGHT}`}
          role="img"
          aria-label={`Two slot timelines for requests A, B, C, and D. Static batching takes ${S.steps} steps and leaves slot 1 empty for steps 3 to 6. Continuous batching takes ${C.steps} steps with no empty slot.`}
        >
          {Array.from({ length: STEPS }, (_, t) => (
            <text
              key={t}
              x={labelW + (t + 0.5) * cellW}
              y={headerY}
              textAnchor="middle"
              style={t === current ? { fill: color.accent, fontWeight: 600 } : undefined}
            >
              {t + 1}
            </text>
          ))}
          <text x={0} y={headerY}>
            Step
          </text>
          <rect
            x={labelW + current * cellW + 1}
            y={headerY + 4}
            width={cellW - 2}
            height={HEIGHT - headerY - 6}
            rx={4}
            style={{ fill: wash(color.accent, 8), stroke: wash(color.accent, 45) }}
          />
          {lanes.map((lane) => (
            <g key={lane.title}>
              <text x={labelW} y={lane.top + 11} style={{ fill: color.fg, fontWeight: 600 }}>
                {lane.title}
              </text>
              {lane.grid.map((row, slot) => {
                const y = lane.top + 18 + slot * rowH;
                return (
                  <g key={slot}>
                    <text x={0} y={y + rowH / 2 + 4}>
                      Slot {slot + 1}
                    </text>
                    {Array.from({ length: STEPS }, (_, t) => {
                      const cell = t < row.length ? row[t] : undefined;
                      const x = labelW + t * cellW + 3;
                      const w = cellW - 6;
                      const h = rowH - 6;
                      if (cell === undefined) return null;
                      if (cell === null) {
                        return (
                          <rect
                            key={t}
                            x={x}
                            y={y + 3}
                            width={w}
                            height={h}
                            rx={3}
                            style={{ fill: "none", stroke: color.bad, strokeDasharray: "3 3" }}
                          />
                        );
                      }
                      const tone = TONE[cell.id];
                      const done = t <= current;
                      return (
                        <g key={t}>
                          <rect
                            x={x}
                            y={y + 3}
                            width={w}
                            height={h}
                            rx={3}
                            style={{ fill: done ? wash(tone, 30) : "none", stroke: tone }}
                          />
                          {cell.finishes && (
                            <line x1={x + w} x2={x + w} y1={y + 3} y2={y + 3 + h} style={{ stroke: color.fg, strokeWidth: 3 }} />
                          )}
                          <text
                            x={x + w / 2}
                            y={y + rowH / 2 + 4}
                            textAnchor="middle"
                            style={{ fill: done ? color.fg : color.muted, fontWeight: 600 }}
                          >
                            {cell.id}
                          </text>
                        </g>
                      );
                    })}
                  </g>
                );
              })}
            </g>
          ))}
        </svg>
      </div>
      <Legend
        items={[
          { label: "Empty slot, still paid for", tone: color.bad, dashed: true },
          { label: "Request finishes", tone: color.fg },
        ]}
      />
    </Figure>
  );
}
