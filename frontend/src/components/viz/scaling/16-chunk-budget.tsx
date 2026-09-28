import { useState } from "react";
import { Figure, Slider, Stat, color, useWidth } from "../kit";
import { YTicks } from "./svg";

// Constants from chapter 16, "Sizing the chunk". A decode step at B = 16 and
// L = 2048 has a memory floor of 45.8 ms (the throughput table: 58.4 GB at
// 1275 GB/s), and one token's arithmetic at the A100's 312 TFLOP/s peak is
// 53.8 GFLOP / 312 TFLOP/s = 0.172 ms. The step time is the larger of the two.
const BATCH = 16;
const FLOOR_MS = 45.8;
const MS_PER_TOKEN = 0.172;
const LAB_CHUNK = 256;
const CHUNKS = [0, 64, 128, 192, 256, 384, 512, 768, 1024] as const;
const X_MAX = 1024;
const Y_MAX = 200;

const compute = (prefill: number) => MS_PER_TOKEN * (prefill + BATCH);
const stepTime = (prefill: number) => Math.max(FLOOR_MS, compute(prefill));
/** Prefill tokens at which compute reaches the floor: T_p + B = 266. */
const FREE_UP_TO = FLOOR_MS / MS_PER_TOKEN - BATCH;

const HEIGHT = 236;

export default function ChunkBudget() {
  const [prefill, setPrefill] = useState<number>(LAB_CHUNK);
  const [ref, width] = useWidth();

  const left = 34;
  const right = 12;
  // The axis unit gets its own line above the top tick.
  const top = 26;
  const bottom = 34;
  const pw = width - left - right;
  const ph = HEIGHT - top - bottom;
  const x = (v: number) => left + (v / X_MAX) * pw;
  const y = (ms: number) => top + ph - (Math.min(ms, Y_MAX) / Y_MAX) * ph;

  const t = stepTime(prefill);
  const added = t - FLOOR_MS;
  // The compute line leaves the chart at Y_MAX.
  const computeEnd = Math.min(X_MAX, Y_MAX / MS_PER_TOKEN - BATCH);
  const narrow = width < 420;

  return (
    <Figure
      title="Prefill tokens hide under the decode step's memory floor"
      controls={
        <Slider
          label="Prefill tokens in the step"
          values={CHUNKS}
          value={prefill}
          onChange={setPrefill}
          format={(v) => v.toLocaleString("en-US")}
        />
      }
      readout={
        <>
          <Stat label="Prefill tokens" value={prefill.toLocaleString("en-US")} />
          <Stat label={`Step time at B = ${BATCH}`} value={`${t.toFixed(1)} ms`} />
          <Stat
            label="Added to every user's ITL"
            value={added < 0.05 ? "0 ms, free" : `${added.toFixed(1)} ms`}
            tone={added < 0.05 ? color.ok : color.warn}
          />
          <Stat label="Free up to" value={`about ${Math.round(FREE_UP_TO)} tokens`} />
        </>
      }
      caption={
        <>
          The step costs the larger of the memory floor and the compute time, so prefill tokens are free until the
          two lines cross. Past that point, every prefill token adds 0.172 ms to all 16 users' inter-token latency.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={HEIGHT}
          viewBox={`0 0 ${width} ${HEIGHT}`}
          role="img"
          aria-label={`Step time against prefill tokens at batch 16. The memory floor is flat at 45.8 ms, and compute at 0.172 ms per token crosses it near ${Math.round(FREE_UP_TO)} prefill tokens. With ${prefill} prefill tokens, the step takes ${t.toFixed(1)} ms.`}
        >
          <YTicks ticks={[0, 50, 100, 150, 200]} y={y} x0={left} x1={left + pw} />
          {[0, 256, 512, 768, 1024].map((v) => (
            <text key={v} x={x(v)} y={top + ph + 14} textAnchor="middle">
              {v}
            </text>
          ))}
          <text x={left + pw / 2} y={HEIGHT - 3} textAnchor="middle">
            Prefill tokens in the step
          </text>
          <text x={left - 5} y={top - 12} textAnchor="end">
            ms
          </text>

          {/* The free region, where prefill hides under the floor. */}
          <rect
            x={x(0)}
            y={top}
            width={x(FREE_UP_TO) - x(0)}
            height={ph}
            style={{ fill: color.ok, opacity: 0.07 }}
          />

          {/* Memory floor and compute time, the two terms of the max. */}
          <line
            x1={x(0)}
            x2={x(X_MAX)}
            y1={y(FLOOR_MS)}
            y2={y(FLOOR_MS)}
            style={{ stroke: color.a, strokeWidth: 1.5, strokeDasharray: "5 4" }}
          />
          <line
            x1={x(0)}
            x2={x(computeEnd)}
            y1={y(compute(0))}
            y2={y(compute(computeEnd))}
            style={{ stroke: color.b, strokeWidth: 1.5, strokeDasharray: "5 4" }}
          />
          {/* The step time: the larger of the two. */}
          <polyline
            points={[0, FREE_UP_TO, computeEnd].map((v) => `${x(v)},${y(stepTime(v))}`).join(" ")}
            style={{ fill: "none", stroke: color.fg, strokeWidth: 2.5 }}
          />

          <text x={x(X_MAX)} y={y(FLOOR_MS) + 14} textAnchor="end" style={{ fill: color.a }}>
            {narrow ? "floor" : "memory floor"} {FLOOR_MS} ms
          </text>
          <text x={x(600) + 6} y={y(compute(600)) + 14} style={{ fill: color.b }}>
            {narrow ? "compute" : `compute, ${MS_PER_TOKEN} ms per token`}
          </text>
          <text x={x(FREE_UP_TO / 2)} y={y(FLOOR_MS) - 8} textAnchor="middle" style={{ fill: color.ok }}>
            free
          </text>

          {/* The chosen chunk. */}
          <line
            x1={x(prefill)}
            x2={x(prefill)}
            y1={top}
            y2={top + ph}
            style={{ stroke: color.accent, strokeWidth: 1, strokeDasharray: "2 3" }}
          />
          <circle cx={x(prefill)} cy={y(t)} r={4.5} style={{ fill: color.accent, stroke: color.card, strokeWidth: 1.5 }} />
        </svg>
      </div>
    </Figure>
  );
}
