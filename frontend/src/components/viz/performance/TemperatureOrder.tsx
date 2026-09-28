// Chapter 11, "Temperature before truncation": the lab's five logits through
// temperature and top-p in both orders. At the initial temperature 2.0 and
// top_p 0.7, the numbers match the chapter's worked example: three tokens at
// (0.5065, 0.3072, 0.1863) in the engine's order, two at (0.6225, 0.3775)
// the other way round.

import { useState } from "react";
import { Figure, Slider, Stat, color, wash, useWidth } from "../kit";
import { linear } from "./scales";
import { greedyMask, keptMass, softmax, temper, topPMask } from "./sampling";

// The lab's five-logit row, from chapter 11.
const LOGITS = [3, 2, 1, 0, -1] as const;

const HEIGHT = 200;
const HEAD = 30;
const FOOT = 34;
const round2 = (v: number) => Math.round(v * 100) / 100;

type Result = { keep: boolean[]; probs: number[] };

/** Temperature, then top-p: the engine's order. */
function temperatureFirst(temperature: number, p: number): Result {
  if (temperature === 0) {
    const keep = greedyMask([...LOGITS]);
    return { keep, probs: softmax([...LOGITS], keep) };
  }
  const tempered = temper(LOGITS, temperature);
  const keep = topPMask(tempered, p);
  return { keep, probs: softmax(tempered, keep) };
}

/** Top-p on the raw logits, then temperature over the survivors. */
function topPFirst(temperature: number, p: number): Result {
  if (temperature === 0) return temperatureFirst(0, p);
  const keep = topPMask([...LOGITS], p);
  return { keep, probs: softmax(temper(LOGITS, temperature), keep) };
}

function Panel({
  x0,
  width,
  title,
  result,
  tone,
}: {
  x0: number;
  width: number;
  title: string;
  result: Result;
  tone: string;
}) {
  const y = linear(0, 1, HEIGHT - FOOT, HEAD);
  const band = width / LOGITS.length;
  const barWidth = band * 0.62;
  const kept = result.keep.filter(Boolean).length;
  return (
    <g transform={`translate(${x0},0)`}>
      <text x={0} y={12} style={{ fill: color.fg, fontWeight: 600 }}>
        {title}
      </text>
      <line x1={0} x2={width} y1={y(0)} y2={y(0)} style={{ stroke: color.lineStrong }} />
      {LOGITS.map((z, i) => {
        const bx = i * band + (band - barWidth) / 2;
        const q = result.probs[i];
        return (
          <g key={z}>
            {result.keep[i] ? (
              <>
                <rect x={bx} y={y(q)} width={barWidth} height={y(0) - y(q)} style={{ fill: tone }} />
                <text x={bx + barWidth / 2} y={y(q) - 4} textAnchor="middle" style={{ fill: color.fg }}>
                  {q.toFixed(2)}
                </text>
              </>
            ) : (
              <rect
                x={bx}
                y={y(0) - 3}
                width={barWidth}
                height={3}
                style={{ fill: wash(color.muted, 20) }}
              />
            )}
            <text x={bx + barWidth / 2} y={y(0) + 14} textAnchor="middle" style={{ fill: result.keep[i] ? color.fg : color.faint }}>
              {z < 0 ? `−${-z}` : z}
            </text>
          </g>
        );
      })}
      <text x={0} y={HEIGHT - 3} style={{ fill: color.muted }}>
        {kept} {kept === 1 ? "token" : "tokens"} kept
      </text>
    </g>
  );
}

export default function TemperatureOrder() {
  const [ref, width] = useWidth();
  const [temperature, setTemperature] = useState(2);
  const [p, setP] = useState(0.7);

  const gutter = width < 420 ? 16 : 32;
  const panelWidth = (width - gutter) / 2;
  const right = temperatureFirst(temperature, p);
  const wrong = topPFirst(temperature, p);
  const tempered = temperature === 0 ? null : softmax(temper(LOGITS, temperature));
  const wrongMass = tempered ? keptMass(tempered, wrong.keep) : 1;
  const count = (r: Result) => r.keep.filter(Boolean).length;
  const narrow = width < 420;

  return (
    <Figure
      title="Temperature before or after top-p"
      controls={
        <>
          <Slider
            label="Temperature, T"
            value={temperature}
            min={0}
            max={3}
            step={0.1}
            onChange={(v) => setTemperature(Math.round(v * 10) / 10)}
            format={(v) => (v === 0 ? "0, greedy" : v.toFixed(1))}
          />
          <Slider label="top_p" value={p} min={0.05} max={1} step={0.05} onChange={(v) => setP(round2(v))} format={(v) => v.toFixed(2)} />
        </>
      }
      readout={
        <>
          <Stat label="Temperature, top_p" value={`${temperature === 0 ? "0, greedy" : temperature.toFixed(1)}, ${p.toFixed(2)}`} tone={color.a} />
          <Stat label="Engine order keeps" value={`${count(right)} tokens, top ${right.probs[0].toFixed(4)}`} tone={color.b} />
          <Stat label="Reversed order keeps" value={`${count(wrong)} tokens, top ${wrong.probs[0].toFixed(4)}`} tone={color.bad} />
          <Stat label="Tempered mass the reversed order keeps" value={wrongMass.toFixed(4)} />
        </>
      }
      caption={
        <>
          Top-p on the untempered row picks its nucleus before temperature flattens the distribution, so at T = 2.0
          and top_p = 0.7 it keeps two tokens holding 0.6886 of the tempered mass, under the 0.7 you asked for.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={HEIGHT}
          role="img"
          aria-label={`Two bar charts of the logits 3, 2, 1, 0, and −1 at temperature ${temperature.toFixed(1)} and top_p ${p.toFixed(2)}. Temperature then top-p keeps ${count(right)} tokens; top-p then temperature keeps ${count(wrong)}.`}
        >
          <Panel
            x0={0}
            width={panelWidth}
            title={narrow ? "T, then top-p" : "Temperature, then top-p"}
            result={right}
            tone={color.b}
          />
          <Panel
            x0={panelWidth + gutter}
            width={panelWidth}
            title={narrow ? "Top-p, then T" : "Top-p, then temperature"}
            result={wrong}
            tone={color.bad}
          />
        </svg>
      </div>
    </Figure>
  );
}
