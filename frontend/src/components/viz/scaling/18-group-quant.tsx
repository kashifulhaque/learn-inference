import { useState } from "react";
import { Figure, Legend, Segmented, Stat, color, useWidth, wash } from "../kit";
import { roundHalfEven } from "./svg";

// The first four weights are chapter 18's worked example, "Four weights in
// int8": (0.834, -0.301, 0.027, -1.27), and "Why per-tensor scaling collapses"
// makes the last one ten times larger, -12.7. The other twelve weights are an
// illustrative example. The quantizer is the chapter's symmetric one: s = max|w|
// / 127 for int8 and max|w| / 7 for int4, round to nearest, then clamp. Bytes
// per weight follow b(g) = (bits / 8) + 2 / g, for 2-byte scales.
const HEAD = [0.834, -0.301, 0.027] as const;
const LAST = { normal: -1.27, outlier: -12.7 } as const;
const REST = [0.412, -0.655, 0.118, 0.973, -0.244, 0.561, -1.05, 0.089, 0.731, -0.387, -0.019, 0.296] as const;

const FORMATS = {
  int8: { bits: 8, qmin: -128, qmax: 127 },
  int4: { bits: 4, qmin: -8, qmax: 7 },
} as const;
type Format = keyof typeof FORMATS;

/** The y range of the weight panel. The outlier runs off it and is labelled. */
const CLIP = 1.45;

function quantize(weights: readonly number[], format: Format, group: number) {
  const { qmin, qmax } = FORMATS[format];
  const scales: number[] = [];
  const q: number[] = [];
  for (let start = 0; start < weights.length; start += group) {
    const block = weights.slice(start, start + group);
    const s = Math.max(...block.map(Math.abs)) / qmax;
    scales.push(s);
    for (const w of block) q.push(Math.min(qmax, Math.max(qmin, roundHalfEven(w / s))));
  }
  const deq = q.map((v, i) => v * scales[Math.floor(i / group)]);
  const err = deq.map((v, i) => Math.abs(v - weights[i]));
  const norm = Math.sqrt(weights.reduce((a, w) => a + w * w, 0));
  const errNorm = Math.sqrt(deq.reduce((a, v, i) => a + (v - weights[i]) ** 2, 0));
  const zeroed = q.filter((v, i) => v === 0 && weights[i] !== 0).length;
  return { scales, q, deq, err, relative: errNorm / norm, zeroed };
}

const fmtScale = (s: number) => s.toPrecision(2);
const fmtWeight = (w: number) => (w < 0 ? `−${Math.abs(w)}` : `${w}`);

// Room below the error panel for its "0" label.
const HEIGHT = 258;

export default function GroupQuant() {
  const [format, setFormat] = useState<Format>("int8");
  const [group, setGroup] = useState(4);
  const [outlier, setOutlier] = useState(true);
  const [ref, width] = useWidth();

  const weights: number[] = [...HEAD, outlier ? LAST.outlier : LAST.normal, ...REST];
  const n = weights.length;
  const r = quantize(weights, format, group);
  const bytes = FORMATS[format].bits / 8 + 2 / group;

  // Layout: scale labels, the weight panel, the outlier label, then the error panel.
  const left = 34;
  const pw = width - left - 4;
  const cw = pw / n;
  const labelTop = 12;
  const wTop = 20;
  const wH = 132;
  const zero = wTop + wH / 2;
  const yW = (w: number) => zero - (Math.max(-CLIP, Math.min(CLIP, w)) / CLIP) * (wH / 2);
  const eLabel = wTop + wH + 30;
  const eTop = eLabel + 6;
  const eH = 58;
  const eMax = Math.max(...r.scales) / 2;
  const yE = (e: number) => eTop + eH - (e / eMax) * eH;
  const groups = r.scales.length;
  const gx = (k: number) => left + k * group * cw;

  return (
    <Figure
      title="One scale per group contains an outlier"
      controls={
        <>
          <Segmented
            label="Format"
            value={format}
            options={[
              { value: "int8", label: "int8" },
              { value: "int4", label: "int4" },
            ]}
            onChange={setFormat}
          />
          <Segmented
            label="Group size g"
            value={group}
            options={[
              { value: 4, label: "4" },
              { value: 8, label: "8" },
              { value: 16, label: "16, one scale" },
            ]}
            onChange={setGroup}
          />
          <Segmented
            label="Fourth weight"
            value={outlier ? "outlier" : "normal"}
            options={[
              { value: "normal", label: "−1.27" },
              { value: "outlier", label: "−12.7, an outlier" },
            ]}
            onChange={(v) => setOutlier(v === "outlier")}
          />
        </>
      }
      readout={
        <>
          <Stat label="Format, group size" value={`${format}, g = ${group}`} />
          <Stat label="Scales" value={`${groups} for ${n} weights`} />
          <Stat label="Relative error ε" value={`${(100 * r.relative).toFixed(2)}%`} tone={color.bad} />
          <Stat label="Rounded to zero" value={`${r.zeroed} of ${n}`} tone={r.zeroed ? color.bad : undefined} />
          <Stat label="Bytes per weight, b(g)" value={bytes.toFixed(3).replace(/0+$/, "").replace(/\.$/, "")} />
        </>
      }
      caption={
        <>
          The largest magnitude in a group sets that group's step, so an outlier pushes its neighbours onto a coarse
          grid and rounds 0.027 to zero. Shrink the group, and the damage stays inside the group that holds the
          outlier, at the price of more scales: these toy groups cost far more than the 1.56% that g = 128 costs.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={HEIGHT}
          viewBox={`0 0 ${width} ${HEIGHT}`}
          role="img"
          aria-label={`Sixteen example weights quantized to ${format} in groups of ${group}. The group scales are ${r.scales.map(fmtScale).join(", ")}. ${r.zeroed} weights round to zero, and the relative error is ${(100 * r.relative).toFixed(2)}%.`}
        >
          {/* Group bands and their scales. */}
          {r.scales.map((s, k) => (
            <g key={k}>
              <rect
                x={gx(k)}
                y={wTop}
                width={group * cw}
                height={eTop + eH - wTop}
                style={{ fill: k % 2 === 0 ? wash(color.c, 9) : "transparent" }}
              />
              <text x={gx(k) + (group * cw) / 2} y={labelTop} textAnchor="middle" style={{ fill: color.fg }}>
                {width < 420 ? `s=${fmtScale(s)}` : `s = ${fmtScale(s)}`}
              </text>
              {/* The quantization levels, where they are far enough apart to see. */}
              {(s / CLIP) * (wH / 2) >= 4 &&
                Array.from({ length: 2 * Math.floor(CLIP / s) + 1 }, (_, j) => j - Math.floor(CLIP / s)).map((level) => (
                  <line
                    key={level}
                    x1={gx(k) + 1}
                    x2={gx(k + 1) - 1}
                    y1={yW(level * s)}
                    y2={yW(level * s)}
                    style={{ stroke: wash(color.a, 35), strokeWidth: 1 }}
                  />
                ))}
            </g>
          ))}

          {/* Axes. */}
          {[-1, 0, 1].map((t) => (
            <text key={t} x={left - 5} y={yW(t) + 4} textAnchor="end">
              {t < 0 ? "−1" : t}
            </text>
          ))}
          <line x1={left} x2={left + pw} y1={zero} y2={zero} style={{ stroke: color.lineStrong }} />

          {/* Weights and their dequantized values. */}
          {weights.map((w, i) => {
            const cx = left + (i + 0.5) * cw;
            const clipped = Math.abs(w) > CLIP;
            const d = r.deq[i];
            return (
              <g key={i}>
                <rect
                  x={cx - cw * 0.28}
                  y={Math.min(yW(w), zero)}
                  width={cw * 0.56}
                  height={Math.abs(yW(w) - zero)}
                  style={{ fill: wash(color.fg, 14), stroke: color.subtle, strokeWidth: 0.75 }}
                />
                {clipped && (
                  <>
                    <path
                      d={`M${cx - cw * 0.4},${yW(w) - 10} l${cw * 0.8},-5 M${cx - cw * 0.4},${yW(w) - 5} l${cw * 0.8},-5`}
                      style={{ stroke: color.card, strokeWidth: 3 }}
                    />
                    <path
                      d={`M${cx - cw * 0.4},${yW(w) - 10} l${cw * 0.8},-5 M${cx - cw * 0.4},${yW(w) - 5} l${cw * 0.8},-5`}
                      style={{ stroke: color.subtle, strokeWidth: 1 }}
                    />
                    <text x={cx} y={wTop + wH + 13} textAnchor="middle" style={{ fill: color.fg, fontWeight: 600 }}>
                      {fmtWeight(w)}
                    </text>
                  </>
                )}
                {r.q[i] === 0 && w !== 0 ? (
                  <circle cx={cx} cy={zero} r={Math.min(5, cw * 0.3)} style={{ fill: color.card, stroke: color.bad, strokeWidth: 2 }} />
                ) : (
                  <line
                    x1={cx - cw * 0.4}
                    x2={cx + cw * 0.4}
                    y1={yW(d)}
                    y2={yW(d)}
                    style={{ stroke: color.a, strokeWidth: 2.5, strokeLinecap: "round" }}
                  />
                )}
              </g>
            );
          })}

          {/* The rounding error, against the half-step bound of each group. */}
          <text x={left} y={eLabel}>
            Rounding error |ŵ − w| against s/2
          </text>
          <line x1={left} x2={left + pw} y1={eTop + eH} y2={eTop + eH} style={{ stroke: color.lineStrong }} />
          <text x={left - 5} y={eTop + 4} textAnchor="end">
            {Number(eMax.toPrecision(2))}
          </text>
          <text x={left - 5} y={eTop + eH + 4} textAnchor="end">
            0
          </text>
          {r.scales.map((s, k) => (
            <line
              key={k}
              x1={gx(k) + 1}
              x2={gx(k + 1) - 1}
              y1={yE(s / 2)}
              y2={yE(s / 2)}
              style={{ stroke: color.a, strokeWidth: 1.25, strokeDasharray: "4 3" }}
            />
          ))}
          {r.err.map((e, i) => (
            <rect
              key={i}
              x={left + (i + 0.5) * cw - cw * 0.22}
              y={yE(e)}
              width={cw * 0.44}
              height={Math.max(0, eTop + eH - yE(e))}
              style={{ fill: color.bad }}
            />
          ))}
        </svg>
      </div>
      <Legend
        items={[
          { label: "Weight w, an example", tone: wash(color.fg, 25) },
          { label: "Dequantized s·q", tone: color.a },
          { label: "Rounded to zero", tone: color.bad },
          { label: "Half a step, s/2", tone: color.a, dashed: true },
        ]}
      />
    </Figure>
  );
}
