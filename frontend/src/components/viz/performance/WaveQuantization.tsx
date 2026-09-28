// Chapter 10, "Idle SMs and occupancy": the fraction of the A100's 108 SMs a
// kernel keeps busy, u(b) = b / (108 · ceil(b / 108)), plotted against the
// number of blocks it launches, with the last wave's SMs drawn underneath.

import { useState } from "react";
import { Figure, Legend, Slider, Stat, color, formatCount, wash, useWidth } from "../kit";
import { decadeLabel, linear, log10Scale } from "./scales";

// From chapter 10: the A100 has 108 streaming multiprocessors.
const SMS = 108;
const waves = (blocks: number) => Math.ceil(blocks / SMS);
const utilization = (blocks: number) => blocks / (SMS * waves(blocks));

const BLOCKS = [1, 8, 27, 54, 107, 108, 109, 162, 216, 217, 324, 540, 1000, 2000, 4095, 4096] as const;
const B_MAX = 4096;
const HEIGHT = 238;
const CHART_BOTTOM = 150;
const M = { top: 12, right: 12, left: 40 };
const STRIP_Y = 204;
const STRIP_H = 14;

export default function WaveQuantization() {
  const [ref, width] = useWidth();
  const [blocks, setBlocks] = useState<number>(109);

  const x = log10Scale(1, B_MAX, M.left, width - M.right);
  const y = linear(0, 1, CHART_BOTTOM, M.top);

  // u(b) is piecewise linear in b; on a log axis, sample every integer so the
  // teeth keep their shape.
  let d = "";
  for (let b = 1; b <= B_MAX; b += 1) d += `${b === 1 ? "M" : "L"}${x(b).toFixed(1)},${y(utilization(b)).toFixed(1)}`;

  const w = waves(blocks);
  const u = utilization(blocks);
  const busyInLast = blocks - SMS * (w - 1);
  const plotWidth = width - M.left - M.right;
  const slot = plotWidth / SMS;

  return (
    <Figure
      title="Wave quantization on 108 SMs"
      controls={<Slider label="Blocks launched, b" value={blocks} values={BLOCKS} onChange={setBlocks} format={(v) => formatCount(v)} />}
      readout={
        <>
          <Stat label="Blocks, b" value={formatCount(blocks)} />
          <Stat label="Waves" value={w} />
          <Stat label="Utilization" value={`${(u * 100).toFixed(1)}%`} tone={color.accent} />
          <Stat label="Busy SMs in the last wave" value={`${busyInLast} of ${SMS}`} />
        </>
      }
      caption={
        <>
          One block past a full wave starts a second wave with one busy SM, so 109 blocks keep the GPU only 50.5%
          busy. With many waves, the partial last one stops mattering.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={HEIGHT}
          role="img"
          aria-label={`Utilization against blocks launched, a sawtooth that drops after every multiple of 108 and flattens toward 100% as the waves multiply. At ${formatCount(blocks)} blocks the kernel runs ${w} waves at ${(u * 100).toFixed(1)}% utilization, and the last wave uses ${busyInLast} of 108 SMs.`}
        >
          {[0, 0.5, 1].map((t) => (
            <g key={t}>
              <line x1={M.left} x2={width - M.right} y1={y(t)} y2={y(t)} style={{ stroke: t === 0 ? color.lineStrong : color.line }} />
              <text x={M.left - 5} y={y(t) + 4} textAnchor="end">
                {t * 100}%
              </text>
            </g>
          ))}
          {[1, 10, 100, 1000].map((t) => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={CHART_BOTTOM} y2={CHART_BOTTOM + 4} style={{ stroke: color.lineStrong }} />
              <text x={x(t)} y={CHART_BOTTOM + 15} textAnchor="middle">
                {decadeLabel(t)}
              </text>
            </g>
          ))}
          <text x={width - M.right} y={CHART_BOTTOM + 28} textAnchor="end" style={{ fill: color.muted }}>
            blocks launched (log scale)
          </text>
          <line x1={x(SMS)} x2={x(SMS)} y1={M.top} y2={CHART_BOTTOM} style={{ stroke: color.faint, strokeDasharray: "2 3" }} />
          <text x={x(SMS) - 4} y={y(0.08)} textAnchor="end" style={{ fill: color.muted }}>
            108
          </text>

          <path d={d} style={{ fill: "none", stroke: color.info, strokeWidth: 1.5 }} />
          <line x1={x(blocks)} x2={x(blocks)} y1={M.top} y2={CHART_BOTTOM} style={{ stroke: color.lineStrong }} />
          <circle cx={x(blocks)} cy={y(u)} r={5} style={{ fill: color.accent, stroke: color.card, strokeWidth: 1.5 }} />

          {/* The last wave: one slot per SM. */}
          <text x={M.left} y={STRIP_Y - 6} style={{ fill: color.fg }}>
            {w === 1 ? "The only wave" : `Wave ${w} of ${w}`}: {busyInLast} of {SMS} SMs busy
          </text>
          {Array.from({ length: SMS }, (_, i) => (
            <rect
              key={i}
              x={M.left + i * slot + (slot > 3 ? 0.5 : 0)}
              y={STRIP_Y}
              width={Math.max(0.5, slot - (slot > 3 ? 1 : 0))}
              height={STRIP_H}
              style={{ fill: i < busyInLast ? color.accent : wash(color.muted, 14) }}
            />
          ))}
          {w > 1 && (
            <text x={M.left} y={STRIP_Y + STRIP_H + 13}>
              after {w - 1} full {w - 1 === 1 ? "wave" : "waves"}
            </text>
          )}
        </svg>
        <Legend
          items={[
            { label: "Utilization, u(b)", tone: color.info },
            { label: "Busy SM", tone: color.accent },
            { label: "Idle SM", tone: wash(color.muted, 14) },
          ]}
        />
      </div>
    </Figure>
  );
}
