// Chapter 10: the A100's roofline on log-log axes, with the model's operations
// placed on it. The MLP's intensity follows the number of tokens in the pass,
// so it slides toward and past the ridge point; decode attention and RMSNorm
// don't move.

import { useState } from "react";
import { Figure, Segmented, Slider, Stat, color, formatCount, useWidth } from "../kit";
import { decadeLabel, decades, log10Scale } from "./scales";

// From chapter 10, "The two rates and the ridge point".
const PEAK_TFLOPS = 312; // bfloat16 tensor cores, rated.
const BANDWIDTHS = { rated: 1935, measured: 1275 } as const; // GB/s: rated, and a measured plain copy.

// From chapter 10, "The MLP": I(T) = T / (1 + T / 26112), where 26112 = 3i/2
// for intermediate size i = 17408 and bfloat16 (e = 2).
const MLP_ASYMPTOTE = (3 * 17408) / 2;
const mlpIntensity = (tokens: number) => tokens / (1 + tokens / MLP_ASYMPTOTE);

// From chapter 10's table of operations: RMSNorm is 2/e = 1, and decode
// attention is H / H_kv = 24 / 4 = 6.
const RMSNORM_INTENSITY = 1;
const ATTENTION_INTENSITY = 24 / 4;

const TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 161, 256, 512, 1024, 2048, 4096, 8192] as const;
const X_MIN = 0.5;
const X_MAX = 20000;
const Y_MIN = 0.3; // TFLOP/s
const Y_MAX = 1000;
const HEIGHT = 260;
const M = { top: 14, right: 12, bottom: 34, left: 40 };

type Bandwidth = keyof typeof BANDWIDTHS;

// A card-coloured outline behind a label, so it stays legible where it crosses a line.
const halo = { stroke: color.card, strokeWidth: 3, paintOrder: "stroke", strokeLinejoin: "round" } as const;

export default function RooflineExplorer() {
  const [ref, width] = useWidth();
  const [tokens, setTokens] = useState<number>(1);
  const [bw, setBw] = useState<Bandwidth>("rated");

  const beta = BANDWIDTHS[bw] / 1000; // TB/s, so beta × I is in TFLOP/s.
  const ridge = PEAK_TFLOPS / beta;
  const attainable = (i: number) => Math.min(PEAK_TFLOPS, beta * i);

  const x = log10Scale(X_MIN, X_MAX, M.left, width - M.right);
  const y = log10Scale(Y_MIN, Y_MAX, HEIGHT - M.bottom, M.top);
  const floor = HEIGHT - M.bottom;

  const mlp = mlpIntensity(tokens);
  const mlpP = attainable(mlp);
  const ratio = mlp / ridge;
  const verdict =
    ratio >= 0.95 && ratio <= 1.05
      ? "At the ridge point"
      : ratio < 1
        ? `Memory, by ${(1 / ratio).toFixed(0)}×`
        : `Compute, ${ratio.toFixed(1)}× past the ridge`;

  const slantStart = Y_MIN / beta; // Where beta × I meets the bottom of the plot.
  const roof = `M${x(Math.max(X_MIN, slantStart))},${y(attainable(Math.max(X_MIN, slantStart)))} L${x(ridge)},${y(PEAK_TFLOPS)} L${x(X_MAX)},${y(PEAK_TFLOPS)}`;
  const narrow = width < 420;

  const fixed = [
    { name: "RMSNorm, 1", intensity: RMSNORM_INTENSITY },
    { name: narrow ? "attention, 6" : "decode attention, 6", intensity: ATTENTION_INTENSITY },
  ];

  return (
    <Figure
      title="The roofline, with the model's operations on it"
      controls={
        <>
          <Slider label="Tokens in the pass, T" value={tokens} values={TOKENS} onChange={setTokens} format={(v) => formatCount(v)} />
          <Segmented<Bandwidth>
            label="Bandwidth"
            value={bw}
            onChange={setBw}
            options={[
              { value: "rated", label: "1935 GB/s rated" },
              { value: "measured", label: "1275 GB/s measured" },
            ]}
          />
        </>
      }
      readout={
        <>
          <Stat label="Tokens, T" value={formatCount(tokens)} />
          <Stat label="MLP intensity" value={`${mlp < 10 ? mlp.toFixed(1) : mlp.toFixed(0)} FLOPs/B`} tone={color.accent} />
          <Stat label="Attainable" value={`${mlpP < 10 ? mlpP.toFixed(2) : mlpP.toFixed(0)} TFLOP/s`} />
          <Stat label="Share of peak" value={`${((mlpP / PEAK_TFLOPS) * 100).toFixed(mlpP / PEAK_TFLOPS < 0.1 ? 2 : 0)}%`} />
          <Stat label={`Ridge point, ${BANDWIDTHS[bw]} GB/s`} value={`${ridge.toFixed(1)} FLOPs/B`} />
          <Stat label="MLP bound by" value={verdict} />
        </>
      }
      caption={
        <>
          More tokens in one pass share one read of the MLP's weights, so the MLP climbs the slanted roof and
          crosses the ridge near T = 161. Decode attention stays at 6 and RMSNorm at 1 whatever the batch, because
          nothing in them is shared.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={HEIGHT}
          role="img"
          aria-label={`Log-log roofline for the A100 at ${BANDWIDTHS[bw]} GB/s: a slanted memory roof meets the flat 312 TFLOP/s compute roof at a ridge point of ${ridge.toFixed(1)} FLOPs per byte. RMSNorm sits at intensity 1 and decode attention at 6, both on the memory roof. The MLP at ${formatCount(tokens)} tokens sits at ${mlp.toFixed(1)} FLOPs per byte, ${verdict.toLowerCase()}.`}
        >
          {decades(Y_MIN, Y_MAX).map((t) => (
            <g key={`y${t}`}>
              <line x1={M.left} x2={width - M.right} y1={y(t)} y2={y(t)} style={{ stroke: color.line }} />
              <text x={M.left - 5} y={y(t) + 4} textAnchor="end">
                {decadeLabel(t)}
              </text>
            </g>
          ))}
          <text x={M.left + 4} y={M.top + 11} style={{ fill: color.muted }}>
            TFLOP/s
          </text>
          {decades(X_MIN, X_MAX).map((t) => (
            <g key={`x${t}`}>
              <line x1={x(t)} x2={x(t)} y1={M.top} y2={floor} style={{ stroke: color.line }} />
              <text x={x(t)} y={floor + 15} textAnchor="middle">
                {decadeLabel(t)}
              </text>
            </g>
          ))}
          <text x={width - M.right} y={HEIGHT - 3} textAnchor="end" style={{ fill: color.muted }}>
            FLOPs per byte (log scale)
          </text>

          {/* The ridge point. */}
          <line x1={x(ridge)} x2={x(ridge)} y1={y(PEAK_TFLOPS)} y2={floor} style={{ stroke: color.faint, strokeDasharray: "3 3" }} />
          <text x={x(ridge) + 4} y={floor - 6} style={{ fill: color.muted }}>
            ridge {ridge.toFixed(0)}
          </text>

          {/* The roof: slanted bandwidth × intensity, then flat at peak. */}
          <path d={roof} style={{ fill: "none", stroke: color.fg, strokeWidth: 2 }} />
          <text x={width - M.right - 2} y={y(PEAK_TFLOPS) + 15} textAnchor="end" style={{ fill: color.muted }}>
            312 TFLOP/s peak
          </text>

          {/* Every stop of the slider, so the MLP's path shows in print. */}
          {TOKENS.map((t) => (
            <circle key={t} cx={x(mlpIntensity(t))} cy={y(attainable(mlpIntensity(t)))} r={2} style={{ fill: color.faint }} />
          ))}

          {fixed.map((op) => (
            <g key={op.name}>
              <circle cx={x(op.intensity)} cy={y(attainable(op.intensity))} r={4.5} style={{ fill: color.b, stroke: color.card, strokeWidth: 1.5 }} />
              <text x={x(op.intensity) + 7} y={y(attainable(op.intensity)) + 14} style={{ ...halo, fill: color.b }}>
                {op.name}
              </text>
            </g>
          ))}

          <circle cx={x(mlp)} cy={y(mlpP)} r={6} style={{ fill: color.accent, stroke: color.card, strokeWidth: 1.5 }} />
          <text
            x={Math.min(width - M.right - 44, Math.max(M.left + 44, x(mlp)))}
            y={y(mlpP) - 11}
            textAnchor="middle"
            style={{ ...halo, fill: color.accent, fontWeight: 600 }}
          >
            MLP, T = {formatCount(tokens)}
          </text>
        </svg>
      </div>
    </Figure>
  );
}
