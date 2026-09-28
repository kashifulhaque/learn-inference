import { useState } from "react";
import { Figure, Slider, Stat, color, formatCount, useWidth, wash } from "../kit";
import { logarithmic, polyline } from "./helpers";

// Chapter 0, "Arithmetic intensity and the ridge point" and "Prefill".
const PEAK_FLOPS = 312e12; // A100 bfloat16 tensor-core peak, FLOP/s
const BANDWIDTH = 1935e9; // A100 80GB rated HBM bandwidth, byte/s
const RIDGE = PEAK_FLOPS / BANDWIDTH; // about 161 FLOP/byte
const N = 5120; // the matrix dimension n in the chapter's table

/** I = 2nmT / (2nm + 2mT) = nT / (n + T), the chapter's exact form. */
const intensity = (tokens: number) => (N * tokens) / (N + tokens);

// The table's rows (1, 16, 161, 2048), the running 2000-token prompt, and powers of two.
const TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 161, 256, 512, 1024, 2000, 2048, 4096, 8192] as const;
const T_MAX = 8192;
const I_MIN = 0.6;
const I_MAX = 8192;
const TICKS = [1, 10, 100, 1000];
const HEIGHT = 230;
const M = { left: 40, right: 12, top: 10, bottom: 36 };

function boundBy(i: number): string {
  const ratio = RIDGE / i;
  if (Math.abs(ratio - 1) <= 0.1) return "About the ridge point";
  if (ratio < 1) return "Compute";
  return `Memory, by ${ratio >= 10 ? Math.round(ratio) : ratio.toFixed(1)}x`;
}

export default function IntensityRidge() {
  const [ref, width] = useWidth();
  const [tokens, setTokens] = useState<number>(1);
  const i = intensity(tokens);

  const x = logarithmic(1, T_MAX, M.left, width - M.right);
  const y = logarithmic(I_MIN, I_MAX, HEIGHT - M.bottom, M.top);
  const curve = Array.from({ length: 121 }, (_, k) => {
    const t = T_MAX ** (k / 120);
    return [x(t), y(intensity(t))] as const;
  });
  const ridgeY = y(RIDGE);
  const plotRight = width - M.right;
  const bottom = HEIGHT - M.bottom;

  return (
    <Figure
      title="Arithmetic intensity against tokens per weight read"
      controls={
        <Slider
          label="Tokens T"
          values={TOKENS}
          value={tokens}
          onChange={setTokens}
          format={(t) => formatCount(t)}
        />
      }
      readout={
        <>
          <Stat label="Tokens T" value={formatCount(tokens)} />
          <Stat label="Intensity" value={`${i < 100 ? i.toFixed(1) : Math.round(i)} FLOP/byte`} tone={color.accent} />
          <Stat label="Bound by" value={boundBy(i)} />
          <Stat
            label="Share of peak FLOP/s, at most"
            value={`${(Math.min(1, i / RIDGE) * 100).toFixed(i / RIDGE < 0.1 ? 1 : 0)}%`}
          />
        </>
      }
      caption={
        <>
          For an <em>n</em> = 5120 matrix, intensity follows <em>T</em> almost one for one until the activations
          start to count, and crosses the ridge point of 161 just past <em>T</em> = 161. Decode at batch 1 sits at
          the far left, 161 times short of it.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={HEIGHT}
          role="img"
          aria-label={`Log-log plot of arithmetic intensity against tokens per weight read, for a 5120-wide matrix. The curve rises from 1 FLOP per byte at one token and crosses the ridge point of 161 just past 161 tokens. The marker is at ${tokens} token${tokens === 1 ? "" : "s"}, ${Math.round(i)} FLOP per byte.`}
        >
          {/* Memory-bound region under the ridge line. */}
          <rect
            x={M.left}
            y={ridgeY}
            width={plotRight - M.left}
            height={bottom - ridgeY}
            style={{ fill: wash(color.warn, 9) }}
          />
          {TICKS.map((t) => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={M.top} y2={bottom} style={{ stroke: color.line }} />
              <text x={x(t)} y={bottom + 14} textAnchor="middle">
                {formatCount(t)}
              </text>
              <line x1={M.left} x2={plotRight} y1={y(t)} y2={y(t)} style={{ stroke: color.line }} />
              <text x={M.left - 6} y={y(t) + 4} textAnchor="end">
                {formatCount(t)}
              </text>
            </g>
          ))}
          <text x={(M.left + plotRight) / 2} y={HEIGHT - 4} textAnchor="middle">
            Tokens sharing one read of the weights, T
          </text>
          <text x={M.left + 6} y={M.top + 12} style={{ fill: color.muted }}>
            FLOP/byte
          </text>

          {/* The ridge point. */}
          <line
            x1={M.left}
            x2={plotRight}
            y1={ridgeY}
            y2={ridgeY}
            strokeDasharray="5 4"
            style={{ stroke: color.warn, strokeWidth: 1.5 }}
          />
          <text x={plotRight - 4} y={ridgeY - 5} textAnchor="end" style={{ fill: color.warn }}>
            ridge point, 161
          </text>
          <text x={plotRight - 4} y={bottom - 6} textAnchor="end" style={{ fill: color.muted }}>
            memory bound
          </text>

          <path d={polyline(curve)} style={{ fill: "none", stroke: color.accent, strokeWidth: 2 }} />

          {/* The chosen T. */}
          <line
            x1={x(tokens)}
            x2={x(tokens)}
            y1={y(i)}
            y2={bottom}
            strokeDasharray="2 3"
            style={{ stroke: color.subtle }}
          />
          <circle cx={x(tokens)} cy={y(i)} r={5} style={{ fill: color.accent, stroke: color.card, strokeWidth: 2 }} />
          <text
            x={x(tokens) + (x(tokens) > width * 0.65 ? -9 : 9)}
            y={y(i) - 8}
            textAnchor={x(tokens) > width * 0.65 ? "end" : "start"}
            style={{ fill: color.fg, fontWeight: 600 }}
          >
            {`I = ${i < 100 ? i.toFixed(1) : Math.round(i)}`}
          </text>
        </svg>
      </div>
    </Figure>
  );
}
