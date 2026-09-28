import { useState } from "react";
import { Figure, Legend, Slider, Stat, color, formatBytes, formatCount, useWidth } from "../kit";
import { logarithmic, polyline } from "./helpers";

// Chapter 2, "KV cache", "Recurrent state", and "The break-even point", all
// from the config's geometry table.
const KV_BYTES_PER_LAYER_TOKEN = 2 * 4 * 256 * 2; // K and V, h_kv = 4, d_h = 256, bfloat16: 4 KiB
const HYBRID_PER_TOKEN = KV_BYTES_PER_LAYER_TOKEN * 16; // 16 full-attention layers: 65,536
const DENSE_PER_TOKEN = KV_BYTES_PER_LAYER_TOKEN * 64; // all 64 layers: 262,144
const DELTA_STATE = 48 * 128 * 128 * 4; // 48 value heads, 128 x 128, float32
const CONV_WINDOW = (2 * 16 * 128 + 48 * 128) * 4 * 2; // 10,240 channels, 4 steps, bfloat16
const FIXED_STATE = (DELTA_STATE + CONV_WINDOW) * 48; // 154,927,104 bytes, 147.8 MiB
const BREAK_EVEN = FIXED_STATE / (DENSE_PER_TOKEN - HYBRID_PER_TOKEN); // 788 tokens

const dense = (L: number) => DENSE_PER_TOKEN * L;
const hybrid = (L: number) => HYBRID_PER_TOKEN * L + FIXED_STATE;

// The chapter's table rows (100, 788, 32k, 128k) and powers of two between.
const CONTEXTS = [100, 256, 512, 788, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072] as const;
const L_MIN = 64;
const L_MAX = 131072;
const B_MIN = 8 * 2 ** 20;
const B_MAX = 64 * 2 ** 30;
const X_TICKS = [128, 1024, 8192, 65536];
const Y_TICKS = [16 * 2 ** 20, 256 * 2 ** 20, 4 * 2 ** 30, 64 * 2 ** 30];
const HEIGHT = 240;
const M = { left: 58, right: 12, top: 10, bottom: 36 };

const bytes = (b: number) => formatBytes(b, b >= 2 ** 30 ? 2 : 1);

export default function Breakeven() {
  const [ref, width] = useWidth();
  const [context, setContext] = useState<number>(32768);

  const x = logarithmic(L_MIN, L_MAX, M.left, width - M.right);
  const y = logarithmic(B_MIN, B_MAX, HEIGHT - M.bottom, M.top);
  const samples = Array.from({ length: 97 }, (_, k) => L_MIN * (L_MAX / L_MIN) ** (k / 96));
  const densePath = polyline(samples.map((L) => [x(L), y(dense(L))] as const));
  const hybridPath = polyline(samples.map((L) => [x(L), y(hybrid(L))] as const));
  const bottom = HEIGHT - M.bottom;
  const right = width - M.right;

  const d = dense(context);
  const h = hybrid(context);
  const cheaper = Math.abs(d - h) < 1 ? "Equal" : d < h ? `Dense, by ${(h / d).toFixed(1)}x` : `Hybrid, by ${(d / h).toFixed(1)}x`;

  return (
    <Figure
      title="Per-sequence cache: hybrid against dense"
      controls={
        <Slider
          label="Context L"
          values={CONTEXTS}
          value={context}
          onChange={setContext}
          format={(L) => formatCount(L, true)}
        />
      }
      readout={
        <>
          <Stat label="Context" value={`${formatCount(context)} tokens`} />
          <Stat label="Dense, 64 layers" value={bytes(d)} tone={color.muted} />
          <Stat label="Hybrid" value={bytes(h)} tone={color.accent} />
          <Stat label="Cheaper" value={cheaper} />
        </>
      }
      caption={
        <>
          Both axes are logarithmic. The hybrid starts from a 147.8 MiB floor of recurrent state and crosses the dense
          line at exactly 788 tokens. Past that, the gap settles at 4x, the ratio of 64 layers to 16.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={HEIGHT}
          role="img"
          aria-label={`Log-log plot of cache bytes per sequence against context length. The dense line rises 256 KiB per token; the hybrid line starts at 147.8 MiB and rises 64 KiB per token. They cross at 788 tokens. At ${context} tokens, dense needs ${bytes(d)} and the hybrid ${bytes(h)}.`}
        >
          {X_TICKS.map((t) => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={M.top} y2={bottom} style={{ stroke: color.line }} />
              <text x={x(t)} y={bottom + 14} textAnchor="middle">
                {formatCount(t, true)}
              </text>
            </g>
          ))}
          {Y_TICKS.map((t) => (
            <g key={t}>
              <line x1={M.left} x2={right} y1={y(t)} y2={y(t)} style={{ stroke: color.line }} />
              <text x={M.left - 6} y={y(t) + 4} textAnchor="end">
                {formatBytes(t, 0)}
              </text>
            </g>
          ))}
          <text x={(M.left + right) / 2} y={HEIGHT - 4} textAnchor="middle">
            Context length L, tokens
          </text>

          {/* The fixed recurrent state, the hybrid's floor. */}
          <line
            x1={M.left}
            x2={right}
            y1={y(FIXED_STATE)}
            y2={y(FIXED_STATE)}
            strokeDasharray="5 4"
            style={{ stroke: color.c }}
          />
          <text x={right - 4} y={y(FIXED_STATE) + 14} textAnchor="end" style={{ fill: color.c }}>
            fixed state, 147.8 MiB
          </text>

          <path d={densePath} style={{ fill: "none", stroke: color.muted, strokeWidth: 2 }} />
          <path d={hybridPath} style={{ fill: "none", stroke: color.accent, strokeWidth: 2 }} />

          {/* Break-even. */}
          <line
            x1={x(BREAK_EVEN)}
            x2={x(BREAK_EVEN)}
            y1={y(dense(BREAK_EVEN))}
            y2={bottom}
            strokeDasharray="2 3"
            style={{ stroke: color.subtle }}
          />
          <circle cx={x(BREAK_EVEN)} cy={y(dense(BREAK_EVEN))} r={3.5} style={{ fill: color.fg }} />
          <text x={x(BREAK_EVEN) - 6} y={y(dense(BREAK_EVEN)) - 8} textAnchor="end" style={{ fill: color.fg }}>
            788
          </text>

          {/* The chosen context. */}
          <line
            x1={x(context)}
            x2={x(context)}
            y1={Math.min(y(d), y(h))}
            y2={Math.max(y(d), y(h))}
            style={{ stroke: color.subtle, strokeWidth: 1 }}
          />
          <circle cx={x(context)} cy={y(d)} r={4.5} style={{ fill: color.muted, stroke: color.card, strokeWidth: 2 }} />
          <circle cx={x(context)} cy={y(h)} r={4.5} style={{ fill: color.accent, stroke: color.card, strokeWidth: 2 }} />
        </svg>
        <div style={{ marginTop: "0.5rem" }}>
          <Legend
            items={[
              { label: "Dense: 256 KiB per token", tone: color.muted },
              { label: "Hybrid: 64 KiB per token + fixed state", tone: color.accent },
              { label: "Recurrent state", tone: color.c, dashed: true },
            ]}
          />
        </div>
      </div>
    </Figure>
  );
}
