import { useState } from "react";
import { Figure, Legend, Slider, Stat, color, useWidth } from "../kit";

// One row per engine step of the seeded 64-request trace in chapter 21:
// [step ms, decode sequences, prefill tokens, preemptions, clock ms].
// scripts/figures/21-capstone-the-engine.py --ts prints these rows. The
// step times are chapter 17a's floors for the 27B model applied to the batches
// the tiny engine scheduled. They are not A100 measurements.
const ROWS: readonly (readonly [number, number, number, number, number])[] = [
  [29.978,0,9,0,30.287],
  [29.962,1,0,0,60.249],
  [30.328,1,22,0,90.577],
  [30.857,3,43,0,121.434],
  [30.777,5,2,0,152.211],
  [30.967,6,15,0,183.178],
  [31.302,7,21,0,214.48],
  [31.101,8,0,0,245.581],
  [31.124,7,12,0,276.705],
  [30.939,7,0,0,307.644],
  [31.326,7,32,0,338.969],
  [31.162,5,33,0,370.132],
  [30.949,6,6,0,401.081],
  [31.184,5,43,0,432.265],
  [31.171,4,38,0,463.436],
  [31.137,6,19,0,494.573],
  [31.135,6,18,0,525.708],
  [30.79,5,7,1,556.497],
  [31.15,6,25,0,587.647],
  [31.133,7,16,0,618.781],
  [31.141,6,21,0,649.922],
  [30.94,7,0,0,680.862],
  [30.978,5,21,1,711.84],
  [30.954,6,8,0,742.794],
  [31.184,5,43,0,773.978],
  [31.141,6,21,0,805.119],
  [31.119,7,9,0,836.238],
  [31.134,7,16,0,867.372],
  [30.97,6,16,1,898.342],
  [30.978,5,21,1,929.32],
  [30.97,6,16,1,960.29],
  [31.183,4,44,0,991.473],
  [31.139,6,20,0,1022.612],
  [30.971,6,16,1,1053.583],
  [30.971,6,16,1,1084.554],
  [30.978,5,21,1,1115.532],
  [30.94,7,0,0,1146.472],
  [30.94,7,0,0,1177.412],
  [30.964,6,13,0,1208.376],
  [30.971,6,16,0,1239.348],
  [30.941,7,0,0,1270.288],
  [30.971,6,16,0,1301.26],
  [30.971,5,17,0,1332.23],
  [31.183,4,44,0,1363.414],
  [31.127,7,13,0,1394.541],
  [31.129,7,14,0,1425.67],
  [31.133,7,16,0,1456.803],
  [31.338,6,39,0,1488.142],
  [30.787,5,6,0,1518.928],
  [30.96,6,11,0,1549.889],
  [30.941,7,0,0,1580.83],
  [30.957,6,9,0,1611.786],
  [30.941,7,0,0,1642.727],
  [31.344,5,43,0,1674.071],
  [31.108,6,5,0,1705.179],
  [31.344,6,42,0,1736.523],
  [30.947,6,5,0,1767.471],
  [30.797,5,11,0,1798.268],
  [30.957,6,9,0,1829.224],
  [31.184,5,43,0,1860.409],
  [30.94,7,0,0,1891.349],
  [31.344,6,42,0,1922.693],
  [30.97,5,17,0,1953.663],
  [30.941,7,0,0,1984.604],
  [30.798,5,11,0,2015.402],
  [31.001,5,32,1,2046.403],
  [30.818,4,22,1,2077.221],
  [31.184,5,43,0,2108.405],
  [31.183,4,44,0,2139.588],
  [30.801,4,14,1,2170.389],
  [30.808,5,16,1,2201.197],
  [30.993,4,29,1,2232.19],
  [30.833,4,30,1,2263.023],
  [30.797,5,11,1,2293.821],
  [31.152,6,26,0,2324.973],
  [30.791,5,8,1,2355.764],
  [31.184,5,43,0,2386.948],
  [31.344,5,43,0,2418.291],
  [30.618,4,3,1,2448.91],
  [30.941,7,0,0,2479.85],
  [30.808,5,16,1,2510.658],
  [31.022,3,45,0,2541.68],
  [30.942,6,2,0,2572.622],
  [30.94,7,0,0,2603.563],
  [30.967,6,14,0,2634.529],
  [30.808,5,16,0,2665.337],
  [30.797,5,11,0,2696.135],
  [30.615,5,0,0,2726.749],
  [30.289,3,0,0,2757.038],
  [30.289,3,0,0,2787.327],
  [30.289,3,0,0,2817.616],
  [30.126,2,0,0,2847.742],
  [29.963,1,0,0,2877.705],
  [29.799,0,0,0,2907.504],
];

const STEP_IDS = ROWS.map((_, index) => index);

export default function TraceClock() {
  const [step, setStep] = useState(0);
  const [ref, width] = useWidth();
  const [ms, decode, prefill, preempted, clock] = ROWS[step];
  const left = 36;
  const right = 10;
  const top = 10;
  const bottom = 26;
  const height = 210;
  const innerW = Math.max(width - left - right, 40);
  const innerH = height - top - bottom;
  const maxTokens = 64;
  const barW = Math.max(innerW / ROWS.length - 1, 1.5);
  const xOf = (index: number) => left + (index / ROWS.length) * innerW;
  const yOf = (value: number) => top + innerH - (Math.min(value, maxTokens) / maxTokens) * innerH;
  const yTicks = [0, 16, 32, 48, 64];

  return (
    <Figure
      title="What each step carries, and how long it takes"
      controls={
        <Slider
          label="Step"
          values={STEP_IDS}
          value={step}
          onChange={setStep}
          format={(value) => `${value + 1} of ${ROWS.length}`}
        />
      }
      readout={
        <>
          <Stat label="Decodes" value={String(decode)} tone={color.b} />
          <Stat label="Prefill tokens" value={String(prefill)} tone={color.a} />
          <Stat label="Step" value={`${ms.toFixed(1)} ms`} />
          <Stat label="Preempted" value={String(preempted)} tone={preempted ? color.bad : undefined} />
          <Stat label="Clock" value={`${clock.toFixed(0)} ms`} />
        </>
      }
      caption="One bar per step of the seeded 64-request trace: decode sequences (blue) under prefill tokens (green), with a red cap where the step preempted a request. Every step costs 29.8 to 31.3 ms, because the weight read dominates. These are the model's floors, not a measurement."
    >
      <div ref={ref}>
        <svg
          width={width}
          height={height}
          role="img"
          aria-label="Decode sequences and prefill tokens per engine step in the capstone trace"
        >
          {yTicks.map((tick) => (
            <g key={tick}>
              <line x1={left} x2={left + innerW} y1={yOf(tick)} y2={yOf(tick)} stroke={color.line} />
              <text x={left - 6} y={yOf(tick) + 4} textAnchor="end" fill={color.muted} fontSize={11}>
                {tick}
              </text>
            </g>
          ))}
          {ROWS.map(([, d, p, x], index) => (
            <g key={index} opacity={index === step ? 1 : 0.55}>
              <rect x={xOf(index)} y={yOf(d)} width={barW} height={yOf(0) - yOf(d)} style={{ fill: color.b }} />
              <rect
                x={xOf(index)}
                y={yOf(d + p)}
                width={barW}
                height={yOf(d) - yOf(d + p)}
                style={{ fill: color.a }}
              />
              {x > 0 && (
                <rect x={xOf(index)} y={top} width={barW} height={4} style={{ fill: color.bad }} />
              )}
            </g>
          ))}
          <rect
            x={xOf(step) - 1}
            y={top}
            width={barW + 2}
            height={innerH}
            fill="none"
            style={{ stroke: color.fg }}
            strokeWidth={1}
          />
          <text x={left + innerW} y={height - 6} textAnchor="end" fill={color.muted} fontSize={11}>
            Step
          </text>
        </svg>
      </div>
      <Legend
        items={[
          { label: "Decode sequences", tone: color.b },
          { label: "Prefill tokens", tone: color.a },
          { label: "Preemption", tone: color.bad },
        ]}
      />
    </Figure>
  );
}
