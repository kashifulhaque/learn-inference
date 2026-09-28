// Chapter 11, "Truncation: one operation, three rules" through "Min-p": one
// example row of twelve logits, tempered, then cut by top-k, top-p, or min-p
// and renormalized. The two rows are made-up examples, sorted by rank, not
// model output; one is peaked and one is flat, so you can compare how each
// rule's keep-set responds to the model's confidence.

import { useState } from "react";
import { Figure, Legend, Segmented, Slider, Stat, color, wash, useWidth } from "../kit";
import { linear } from "./scales";
import { greedyMask, keptMass, minPMask, softmax, temper, topKMask, topPMask } from "./sampling";

// Example logits, sorted from rank 1 down. Illustrative only.
const ROWS = {
  peaked: [5.0, 3.2, 2.6, 2.1, 1.7, 1.2, 0.8, 0.4, 0.0, -0.5, -1.0, -1.6],
  flat: [2.0, 1.85, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8],
} as const;

type Row = keyof typeof ROWS;
type Rule = "top-k" | "top-p" | "min-p";

const HEIGHT = 220;
const M = { top: 16, right: 8, bottom: 28, left: 34 };
const RULE_TONE: Record<Rule, string> = { "top-k": color.accent, "top-p": color.b, "min-p": color.c };
const round2 = (v: number) => Math.round(v * 100) / 100;

// A card-coloured outline behind a label, so it stays legible over a bar.
const halo = { stroke: color.card, strokeWidth: 3, paintOrder: "stroke", strokeLinejoin: "round" } as const;

export default function TruncationRules() {
  const [ref, width] = useWidth();
  const [row, setRow] = useState<Row>("peaked");
  const [rule, setRule] = useState<Rule>("top-p");
  const [temperature, setTemperature] = useState(1);
  const [k, setK] = useState(4);
  const [p, setP] = useState(0.9);
  const [minP, setMinP] = useState(0.1);

  const logits = ROWS[row];
  const greedy = temperature === 0;
  // Temperature first, then the truncation, as the engine orders them.
  const tempered = greedy ? [...logits] : temper(logits, temperature);
  const before = greedy ? greedyMask(logits).map((keep) => (keep ? 1 : 0)) : softmax(tempered);
  const keep = greedy
    ? greedyMask(logits)
    : rule === "top-k"
      ? topKMask(tempered, k)
      : rule === "top-p"
        ? topPMask(tempered, p)
        : minPMask(tempered, minP);
  const after = softmax(tempered, keep);
  const keptCount = keep.filter(Boolean).length;
  const mass = keptMass(before, keep);

  const band = (width - M.left - M.right) / logits.length;
  const barWidth = Math.max(3, band * 0.68);
  const y = linear(0, 1, HEIGHT - M.bottom, M.top);
  const bx = (i: number) => M.left + i * band + (band - barWidth) / 2;
  const boundary = M.left + keptCount * band;
  const tone = greedy ? color.accent : RULE_TONE[rule];

  const setting = greedy
    ? "Greedy (temperature 0)"
    : rule === "top-k"
      ? `Top-k, k = ${k}`
      : rule === "top-p"
        ? `Top-p, p = ${p.toFixed(2)}`
        : `Min-p, min_p = ${minP.toFixed(2)}`;
  const minPThreshold = Math.max(...before) * minP;

  return (
    <Figure
      title="Temperature and truncation on an example row"
      controls={
        <>
          <Segmented<Row>
            label="Example row"
            value={row}
            onChange={setRow}
            options={[
              { value: "peaked", label: "Peaked" },
              { value: "flat", label: "Flat" },
            ]}
          />
          <Segmented<Rule>
            label="Rule"
            value={rule}
            onChange={setRule}
            options={[
              { value: "top-k", label: "Top-k" },
              { value: "top-p", label: "Top-p" },
              { value: "min-p", label: "Min-p" },
            ]}
          />
          {rule === "top-k" && <Slider label="k" value={k} min={1} max={logits.length} onChange={setK} />}
          {rule === "top-p" && (
            <Slider label="p" value={p} min={0.05} max={1} step={0.05} onChange={(v) => setP(round2(v))} format={(v) => v.toFixed(2)} />
          )}
          {rule === "min-p" && (
            <Slider
              label="min_p"
              value={minP}
              min={0}
              max={1}
              step={0.05}
              onChange={(v) => setMinP(round2(v))}
              format={(v) => v.toFixed(2)}
            />
          )}
          <Slider
            label="Temperature, T"
            value={temperature}
            min={0}
            max={3}
            step={0.1}
            onChange={(v) => setTemperature(Math.round(v * 10) / 10)}
            format={(v) => (v === 0 ? "0, greedy" : v.toFixed(1))}
          />
        </>
      }
      readout={
        <>
          <Stat label="Row" value={row === "peaked" ? "Peaked example" : "Flat example"} />
          <Stat label="Temperature" value={greedy ? "0, greedy" : temperature.toFixed(1)} tone={color.a} />
          <Stat label="Rule" value={setting} tone={tone} />
          <Stat label="Tokens kept" value={`${keptCount} of ${logits.length}`} />
          <Stat label="Mass kept, before renormalizing" value={mass.toFixed(4)} />
          <Stat label="Top token, after" value={after[0].toFixed(4)} />
        </>
      }
      caption={
        <>
          Top-k keeps the same count on the peaked and the flat row, while top-p and min-p keep fewer tokens when
          the model is confident and more when it isn't. Raise the temperature and watch top-p's nucleus grow.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={HEIGHT}
          role="img"
          aria-label={`Bar chart of twelve example tokens by rank on the ${row} row. ${setting} at temperature ${greedy ? "0" : temperature.toFixed(1)} keeps ${keptCount} tokens, holding ${mass.toFixed(2)} of the tempered probability, and renormalizes them so the top token has probability ${after[0].toFixed(2)}.`}
        >
          {[0, 0.5, 1].map((t) => (
            <g key={t}>
              <line x1={M.left} x2={width - M.right} y1={y(t)} y2={y(t)} style={{ stroke: t === 0 ? color.lineStrong : color.line }} />
              <text x={M.left - 5} y={y(t) + 4} textAnchor="end">
                {t}
              </text>
            </g>
          ))}
          {logits.map((_, i) => (
            <g key={i}>
              <rect
                x={bx(i)}
                y={y(before[i])}
                width={barWidth}
                height={y(0) - y(before[i])}
                style={{ fill: wash(color.muted, 14), stroke: color.faint, strokeWidth: 1 }}
              />
              {keep[i] && (
                <rect
                  x={bx(i) + barWidth * 0.2}
                  y={y(after[i])}
                  width={barWidth * 0.6}
                  height={y(0) - y(after[i])}
                  style={{ fill: tone }}
                />
              )}
              <text x={bx(i) + barWidth / 2} y={HEIGHT - M.bottom + 14} textAnchor="middle" style={{ fill: keep[i] ? color.fg : color.faint }}>
                {i + 1}
              </text>
            </g>
          ))}
          <text x={width - M.right} y={HEIGHT - 2} textAnchor="end" style={{ fill: color.muted }}>
            rank
          </text>

          {/* Where the rule cuts. */}
          {(greedy || rule !== "min-p") && keptCount < logits.length && (
            <>
              <line x1={boundary} x2={boundary} y1={M.top - 4} y2={y(0)} style={{ stroke: tone, strokeDasharray: "4 3", strokeWidth: 1.5 }} />
              <text
                x={boundary > width - 120 ? boundary - 5 : boundary + 5}
                y={M.top + 8}
                textAnchor={boundary > width - 120 ? "end" : "start"}
                style={{ ...halo, fill: tone }}
              >
                {greedy ? "argmax only" : rule === "top-k" ? `cut after rank ${k}` : `mass ${mass.toFixed(2)} ≥ p`}
              </text>
            </>
          )}
          {!greedy && rule === "min-p" && minP > 0 && (
            <>
              <line x1={M.left} x2={width - M.right} y1={y(minPThreshold)} y2={y(minPThreshold)} style={{ stroke: tone, strokeDasharray: "4 3", strokeWidth: 1.5 }} />
              <text x={width - M.right} y={y(minPThreshold) - 5} textAnchor="end" style={{ ...halo, fill: tone }}>
                min_p × top = {minPThreshold.toFixed(3)}
              </text>
            </>
          )}
        </svg>
        <Legend
          items={[
            { label: "Tempered probability, before the cut", tone: wash(color.muted, 30) },
            { label: "Kept and renormalized", tone },
          ]}
        />
      </div>
    </Figure>
  );
}
