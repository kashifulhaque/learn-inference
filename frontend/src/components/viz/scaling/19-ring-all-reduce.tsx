import { Figure, Legend, Stat, StepControls, color, formatBytes, useStepper, useWidth, wash } from "../kit";
import { ArrowMarker, useSvgId } from "./svg";

// The ring all-reduce from chapter 19, "What one all-reduce costs", at P = 4,
// this model's largest degree. The tensor is cut into P pieces; P - 1
// reduce-scatter hops and P - 1 all-gather hops each send one piece, S / P
// bytes, per rank, for 2(P - 1)/P x S in total. S for one token's hidden state
// is 5120 x 2 = 10240 bytes.
const P = 4;
const HIDDEN_BYTES = 5120 * 2;
const FRAMES = 2 * (P - 1) + 1;
const RANK_TONES = [color.info, color.violet, color.warn, color.ok] as const;
const FULL = (1 << P) - 1;

type Frame = {
  /** held[rank][piece]: a bitmask of the ranks whose contribution is summed in. */
  held: number[][];
  /** The piece each rank sent to its clockwise neighbour to reach this frame. */
  sent: number[] | null;
};

function simulate(): Frame[] {
  let held = Array.from({ length: P }, (_, r) => Array.from({ length: P }, () => 1 << r));
  const frames: Frame[] = [{ held, sent: null }];
  for (let s = 1; s < FRAMES; s++) {
    const gather = s > P - 1;
    const hop = gather ? s - (P - 1) : s;
    // Reduce-scatter: rank r sends piece (r - hop + 1), and the receiver adds it.
    // All-gather: rank r sends its finished piece (r - hop + 2), and the receiver keeps it.
    const sent = Array.from({ length: P }, (_, r) => (((r - hop + (gather ? 2 : 1)) % P) + P) % P);
    const next = held.map((row) => [...row]);
    for (let r = 0; r < P; r++) {
      const to = (r + 1) % P;
      const c = sent[r];
      next[to][c] = gather ? held[r][c] : next[to][c] | held[r][c];
    }
    held = next;
    frames.push({ held, sent });
  }
  return frames;
}

const FRAMES_DATA = simulate();

function phase(step: number): string {
  if (step === 0) return "Start: each GPU holds only its own partial sum";
  if (step < P) return `Reduce-scatter, hop ${step} of ${P - 1}`;
  if (step < FRAMES - 1) return `All-gather, hop ${step - (P - 1)} of ${P - 1}`;
  return `All-gather, hop ${P - 1} of ${P - 1}: every GPU holds the full sum`;
}

const HEIGHT = 206;

export default function RingAllReduce() {
  const stepper = useStepper(FRAMES, 1100);
  const [ref, width] = useWidth();
  const arrow = useSvgId("ring-arrow");
  const frame = FRAMES_DATA[stepper.step];

  const gapX = 60;
  const boxW = Math.min(170, (width - gapX) / 2);
  const boxH = 72;
  const gapY = 46;
  const top = 4;
  const pad = 8;
  const cellW = (boxW - 2 * pad) / P;
  const cellH = 28;
  const origin = [
    { x: 0, y: top },
    { x: width - boxW, y: top },
    { x: width - boxW, y: top + boxH + gapY },
    { x: 0, y: top + boxH + gapY },
  ];

  // The ring's four edges, clockwise, with where each label sits.
  const edges = [
    { x1: boxW + 4, y1: top + boxH / 2, x2: width - boxW - 4, y2: top + boxH / 2, lx: width / 2, ly: top + boxH / 2 - 7, anchor: "middle" },
    { x1: width - boxW / 2, y1: top + boxH + 4, x2: width - boxW / 2, y2: top + boxH + gapY - 4, lx: width - boxW / 2 - 8, ly: top + boxH + gapY / 2 + 4, anchor: "end" },
    { x1: width - boxW - 4, y1: top + boxH * 1.5 + gapY, x2: boxW + 4, y2: top + boxH * 1.5 + gapY, lx: width / 2, ly: top + boxH * 1.5 + gapY - 7, anchor: "middle" },
    { x1: boxW / 2, y1: top + boxH + gapY - 4, x2: boxW / 2, y2: top + boxH + 4, lx: boxW / 2 + 8, ly: top + boxH + gapY / 2 + 4, anchor: "start" },
  ] as const;

  const sentSoFar = stepper.step;
  const total = 2 * (P - 1);

  return (
    <Figure
      title="A ring all-reduce on four GPUs"
      controls={<StepControls stepper={stepper} count={FRAMES} describe={phase} />}
      readout={
        <>
          <Stat label="Sent per rank so far" value={`${sentSoFar} × S/${P} = ${sentSoFar / P} S`} />
          <Stat label="Whole all-reduce, per rank" value={`2(P − 1)/P · S = ${total / P} S`} tone={color.c} />
          <Stat label="Per rank, one token's hidden state" value={`${formatBytes((total / P) * HIDDEN_BYTES)}, with S = ${formatBytes(HIDDEN_BYTES)}`} />
        </>
      }
      caption={
        <>
          Each hop, every GPU sends one quarter of the tensor to its neighbour. After three reduce-scatter hops each GPU
          holds one fully summed piece, and three all-gather hops copy those pieces around the ring.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={HEIGHT}
          viewBox={`0 0 ${width} ${HEIGHT}`}
          role="img"
          aria-label={`Four GPUs in a ring, each holding a tensor cut into four pieces. ${phase(stepper.step)}. Over the whole all-reduce, each GPU sends 1.5 times the tensor size.`}
        >
          <defs>
            <ArrowMarker id={arrow} tone={color.muted} />
          </defs>
          {edges.map((e, r) => (
            <g key={r}>
              <line
                x1={e.x1}
                y1={e.y1}
                x2={e.x2}
                y2={e.y2}
                markerEnd={`url(#${arrow})`}
                style={{ stroke: frame.sent ? RANK_TONES[r] : color.lineStrong, strokeWidth: frame.sent ? 2 : 1 }}
              />
              {frame.sent && (
                <text x={e.lx} y={e.ly} textAnchor={e.anchor} style={{ fill: RANK_TONES[r], fontWeight: 600 }}>
                  piece {frame.sent[r]}
                </text>
              )}
            </g>
          ))}
          {origin.map((o, r) => {
            const received = frame.sent ? frame.sent[(r + P - 1) % P] : -1;
            return (
              <g key={r}>
                <rect x={o.x} y={o.y} width={boxW} height={boxH} rx={6} style={{ fill: wash(RANK_TONES[r], 7), stroke: color.line }} />
                <text x={o.x + pad} y={o.y + 15} style={{ fill: RANK_TONES[r], fontWeight: 600 }}>
                  GPU {r}
                </text>
                {frame.held[r].map((mask, c) => {
                  const x = o.x + pad + c * cellW;
                  const y = o.y + 22;
                  const complete = mask === FULL;
                  return (
                    <g key={c}>
                      {Array.from({ length: P }, (_, contributor) => (
                        <rect
                          key={contributor}
                          x={x + 2 + (contributor * (cellW - 4)) / P}
                          y={y}
                          width={(cellW - 4) / P}
                          height={cellH}
                          style={{ fill: mask & (1 << contributor) ? RANK_TONES[contributor] : wash(color.fg, 5) }}
                        />
                      ))}
                      <rect
                        x={x + 2}
                        y={y}
                        width={cellW - 4}
                        height={cellH}
                        style={{
                          fill: "none",
                          stroke: c === received ? color.fg : complete ? color.fg : color.line,
                          strokeWidth: c === received ? 2.5 : complete ? 1.5 : 1,
                          strokeDasharray: c === received ? "4 2" : undefined,
                        }}
                      />
                      <text x={x + cellW / 2} y={y + cellH + 12} textAnchor="middle" style={{ fontSize: 10 }}>
                        {c}
                      </text>
                    </g>
                  );
                })}
              </g>
            );
          })}
        </svg>
      </div>
      <Legend items={RANK_TONES.map((tone, r) => ({ label: `GPU ${r}'s contribution`, tone }))} />
    </Figure>
  );
}
