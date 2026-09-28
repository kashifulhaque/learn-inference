import { useState } from "react";
import { Figure, Legend, Segmented, Slider, Stat, color, formatBytes, formatCount, useWidth, wash } from "../kit";

// Chapter 2, "The largest batch that fits": the lab's values for C, W, and O,
// and the per-sequence cost from "KV cache" and "Recurrent state".
const GiB = 2 ** 30;
const WEIGHTS = 53.8e9; // W, 53.8 GB of bfloat16 weights, in decimal bytes as the lab uses
const OVERHEAD = 6 * GiB; // O: CUDA context, driver, activation workspace, fragmentation headroom
const KV_PER_TOKEN = 65_536; // 64 KiB, the 16 full-attention layers
const FIXED_STATE = 154_927_104; // 147.8 MiB of recurrent state, the 48 linear layers
const CARDS = { 80: 80 * GiB, 40: 40 * GiB } as const;
type Card = keyof typeof CARDS;

const CONTEXTS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072] as const;

const HEIGHT = 96;
const BAR_Y = 26;
const BAR_H = 34;

const gib = (b: number) => `${(b / GiB).toFixed(1)} GiB`;

export default function BatchBudget() {
  const [ref, width] = useWidth();
  const [context, setContext] = useState<number>(32768);
  const [card, setCard] = useState<Card>(80);

  const capacity = CARDS[card];
  const perSequence = FIXED_STATE + KV_PER_TOKEN * context;
  const numerator = capacity - WEIGHTS - OVERHEAD;
  const batch = numerator > 0 ? Math.floor(numerator / perSequence) : 0;
  const unused = Math.max(0, numerator - batch * perSequence);

  // The drawing's scale covers the card, or the weights and overhead when they overflow it.
  const span = Math.max(capacity, WEIGHTS + OVERHEAD);
  const px = (b: number) => (b / span) * (width - 1);
  const wX = px(WEIGHTS);
  const oX = px(WEIGHTS + OVERHEAD);
  const cardX = px(capacity);
  const slotW = px(perSequence);
  const stateW = px(FIXED_STATE);
  const overflow = numerator < 0;

  const slots = Array.from({ length: batch }, (_, k) => oX + k * slotW);
  const aggregate = slotW < 4; // too thin to draw one by one: show all states, then all KV

  return (
    <Figure
      title="How many sequences fit on the card"
      controls={
        <>
          <Slider
            label="Context L"
            values={CONTEXTS}
            value={context}
            onChange={setContext}
            format={(L) => formatCount(L, true)}
          />
          <Segmented
            label="Card"
            value={card}
            onChange={setCard}
            options={[
              { value: 80, label: "80 GiB" },
              { value: 40, label: "40 GiB" },
            ]}
          />
        </>
      }
      readout={
        <>
          <Stat label="Card C" value={`${card} GiB`} />
          <Stat label="C − W − O" value={overflow ? `−${gib(-numerator)}` : gib(numerator)} />
          <Stat label={`One sequence at ${formatCount(context, true)}`} value={formatBytes(perSequence, 2)} />
          <Stat label="State share" value={`${Math.round((FIXED_STATE / perSequence) * 100)}%`} tone={color.c} />
          <Stat label="B max" value={String(batch)} tone={color.accent} />
          <Stat label="Unused" value={gib(unused)} />
        </>
      }
      caption={
        <>
          The weights and the overhead come off the top, and what's left divides into whole sequences of recurrent
          state plus KV cache. The batch falls from 60 at 4k context to 11 at 32k; on a 40 GiB card the weights alone
          overflow.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={HEIGHT}
          role="img"
          aria-label={`A bar for the ${card} GiB card: ${gib(WEIGHTS)} of weights, 6 GiB of overhead, then ${batch} sequences at ${context} tokens, each ${formatBytes(perSequence, 2)}.`}
        >
          {/* Weights and overhead. */}
          <rect x={0.5} y={BAR_Y} width={wX} height={BAR_H} style={{ fill: wash(color.d, 26), stroke: color.d }} />
          <text x={6} y={BAR_Y + 21} style={{ fill: color.d }}>
            {wX > 110 ? `Weights ${gib(WEIGHTS)}` : "W"}
          </text>
          <rect
            x={wX + 0.5}
            y={BAR_Y}
            width={oX - wX}
            height={BAR_H}
            style={{ fill: wash(color.muted, 16), stroke: color.muted }}
          />
          <text x={(wX + oX) / 2} y={BAR_Y + 21} textAnchor="middle" style={{ fill: color.muted }}>
            O
          </text>

          {/* Sequences. */}
          {aggregate && batch > 0 ? (
            <>
              <rect x={oX} y={BAR_Y} width={batch * stateW} height={BAR_H} style={{ fill: wash(color.c, 40) }} />
              <rect
                x={oX + batch * stateW}
                y={BAR_Y}
                width={batch * (slotW - stateW)}
                height={BAR_H}
                style={{ fill: wash(color.b, 30) }}
              />
            </>
          ) : (
            slots.map((x0, k) => (
              <g key={k}>
                <rect x={x0} y={BAR_Y} width={stateW} height={BAR_H} style={{ fill: wash(color.c, 40) }} />
                <rect
                  x={x0 + stateW}
                  y={BAR_Y}
                  width={slotW - stateW}
                  height={BAR_H}
                  style={{ fill: wash(color.b, 30), stroke: color.b, strokeWidth: 0.75 }}
                />
              </g>
            ))
          )}
          {batch > 0 && (
            <text
              x={Math.min(oX + (batch * slotW) / 2, cardX - 30)}
              y={BAR_Y - 6}
              textAnchor="middle"
              style={{ fill: color.accent, fontWeight: 600 }}
            >
              {`${batch} sequence${batch === 1 ? "" : "s"}`}
            </text>
          )}

          {/* The card's edge. */}
          <line
            x1={cardX}
            x2={cardX}
            y1={BAR_Y - 16}
            y2={BAR_Y + BAR_H + 8}
            style={{ stroke: overflow ? color.bad : color.fg, strokeWidth: 1.5 }}
          />
          {!overflow && (
            <rect
              x={0.5}
              y={BAR_Y}
              width={cardX - 0.5}
              height={BAR_H}
              style={{ fill: "none", stroke: color.lineStrong }}
            />
          )}
          <text
            x={overflow ? 0 : cardX - 4}
            y={BAR_Y + BAR_H + 22}
            textAnchor={overflow ? "start" : "end"}
            style={{ fill: overflow ? color.bad : color.fg }}
          >
            {overflow ? `The weights overflow the ${card} GiB card` : `${card} GiB card`}
          </text>
        </svg>
        <div style={{ marginTop: "0.4rem" }}>
          <Legend
            items={[
              { label: "Weights W", tone: color.d },
              { label: "Overhead O", tone: color.muted },
              { label: "Recurrent state", tone: color.c },
              { label: "KV cache", tone: color.b },
            ]}
          />
        </div>
      </div>
    </Figure>
  );
}
