import { useState, type ReactNode } from "react";
import { Figure, Segmented, Stat, color, formatBytes, useWidth, wash } from "../kit";
import { ArrowMarker, Sym, useSvgId } from "./svg";

// Constants from chapter 19: the MLP's hidden size 5120 and intermediate size
// 17408, a bfloat16 hidden state of 5120 x 2 = 10240 bytes per token, and the
// ring all-reduce's 2(P - 1)/P factor from "What one all-reduce costs". This
// model allows P = 1, 2, or 4.
const HIDDEN = 5120;
const INTERMEDIATE = 17408;
const HIDDEN_BYTES = HIDDEN * 2;
const RANK_TONES = [color.info, color.violet, color.warn, color.ok] as const;

type Mode = "column" | "row" | "pair";
type Kind = "full" | "colStrip" | "rowStrip" | "partial" | "result" | "optional" | "allReduce" | "allGather";
type Column = { kind: Kind; head: { base: string; sub?: string; sup?: string }; op?: string };

const LAYOUTS: Record<Mode, Column[]> = {
  column: [
    { kind: "full", head: { base: "X" } },
    { kind: "colStrip", head: { base: "W", sub: "p" }, op: "×" },
    { kind: "colStrip", head: { base: "Y", sub: "p" }, op: "=" },
    { kind: "allGather", head: { base: "" } },
    { kind: "optional", head: { base: "Y" } },
  ],
  row: [
    { kind: "colStrip", head: { base: "X", sub: "p" } },
    { kind: "rowStrip", head: { base: "W", sub: "p" }, op: "×" },
    { kind: "partial", head: { base: "Y", sup: "(p)" }, op: "=" },
    { kind: "allReduce", head: { base: "" } },
    { kind: "result", head: { base: "Y" } },
  ],
  pair: [
    { kind: "full", head: { base: "X" } },
    { kind: "colStrip", head: { base: "W", sub: "g,p" }, op: "×" },
    { kind: "colStrip", head: { base: "H", sub: "p" }, op: "→" },
    { kind: "rowStrip", head: { base: "W", sub: "d,p" }, op: "×" },
    { kind: "partial", head: { base: "Y", sup: "(p)" }, op: "=" },
    { kind: "allReduce", head: { base: "" } },
    { kind: "result", head: { base: "Y" } },
  ],
};

function Block({ kind, x, y, w, h, rank, P }: { kind: Kind; x: number; y: number; w: number; h: number; rank: number; P: number }) {
  const tone = RANK_TONES[rank];
  switch (kind) {
    case "full":
      return <rect x={x} y={y} width={w} height={h} rx={2} style={{ fill: wash(color.fg, 10), stroke: color.lineStrong }} />;
    case "result":
      return <rect x={x} y={y} width={w} height={h} rx={2} style={{ fill: wash(color.fg, 14), stroke: color.fg, strokeWidth: 1.5 }} />;
    case "optional":
      return (
        <rect x={x} y={y} width={w} height={h} rx={2} style={{ fill: "none", stroke: color.subtle, strokeDasharray: "3 3" }} />
      );
    case "partial":
      return (
        <rect
          x={x}
          y={y}
          width={w}
          height={h}
          rx={2}
          style={{ fill: wash(tone, 22), stroke: tone, strokeWidth: 1.25, strokeDasharray: "3 2" }}
        />
      );
    case "colStrip":
    case "rowStrip": {
      const cols = kind === "colStrip";
      return (
        <g>
          <rect x={x} y={y} width={w} height={h} rx={2} style={{ fill: wash(color.fg, 4), stroke: color.line }} />
          <rect
            x={cols ? x + (rank * w) / P : x}
            y={cols ? y : y + (rank * h) / P}
            width={cols ? w / P : w}
            height={cols ? h : h / P}
            style={{ fill: wash(tone, 60), stroke: tone }}
          />
        </g>
      );
    }
    default:
      return null;
  }
}

const HEIGHT = 214;

export default function ColumnRowSplit() {
  const [mode, setMode] = useState<Mode>("pair");
  const [P, setP] = useState(2);
  const [ref, width] = useWidth();
  const arrow = useSvgId("tp-arrow");

  const columns = LAYOUTS[mode];
  const narrow = width < 440;
  const rankW = narrow ? 16 : 42;
  const slot = (width - rankW) / columns.length;
  const boxW = Math.min(slot * 0.66, 58);
  const top = 30;
  const laneArea = HEIGHT - top - 4;
  const laneH = laneArea / P;
  const boxH = Math.min(laneH * 0.7, 36);
  const cx = (c: number) => rankW + (c + 0.5) * slot;
  const laneY = (p: number) => top + p * laneH + (laneH - boxH) / 2;

  const ring = (2 * (P - 1)) / P;
  let readout: ReactNode;
  if (mode === "column") {
    readout = (
      <>
        <Stat label="Degree P" value={P} />
        <Stat label="Each rank computes" value={`(T, d_out / ${P}), finished`} />
        <Stat label="Communication" value="None" tone={color.ok} />
      </>
    );
  } else if (mode === "row") {
    readout = (
      <>
        <Stat label="Degree P" value={P} />
        <Stat label="Each rank computes" value="(T, d_out), a partial sum" />
        <Stat label="Communication" value="One all-reduce" tone={color.accent} />
        <Stat label="Sent per rank" value={`2(P − 1)/P · S = ${ring} S`} />
      </>
    );
  } else {
    readout = (
      <>
        <Stat label="Degree P" value={P} />
        <Stat label="Intermediate slice per rank" value={`(T, ${INTERMEDIATE / P}), never gathered`} />
        <Stat label="Communication" value={`One all-reduce of (T, ${HIDDEN})`} tone={color.accent} />
        <Stat label="Sent per rank per token" value={formatBytes(ring * HIDDEN_BYTES)} />
      </>
    );
  }

  const collective = columns.findIndex((c) => c.kind === "allReduce" || c.kind === "allGather");
  const isReduce = columns[collective].kind === "allReduce";
  const cw = Math.min(boxW * 0.62, 26);

  return (
    <Figure
      title="Column and row splits, and where the all-reduce goes"
      controls={
        <>
          <Segmented
            label="Split"
            value={mode}
            options={[
              { value: "column", label: "Column" },
              { value: "row", label: "Row" },
              { value: "pair", label: "Column then row (MLP)" },
            ]}
            onChange={setMode}
          />
          <Segmented
            label="GPUs P"
            value={P}
            options={[
              { value: 2, label: "2" },
              { value: 4, label: "4" },
            ]}
            onChange={setP}
          />
        </>
      }
      readout={readout}
      caption={
        <>
          A column split gives each GPU finished columns of the output, and a row split gives each GPU a full-size
          partial sum that an all-reduce adds up. Put the column split first, and its sharded output <i>H</i>
          <sub>p</sub> is already the input the row split wants, so the pair needs one all-reduce.
        </>
      }
    >
      <div ref={ref}>
        <svg
          width={width}
          height={HEIGHT}
          viewBox={`0 0 ${width} ${HEIGHT}`}
          role="img"
          aria-label={
            mode === "column"
              ? `A column-parallel layer on ${P} GPUs: each GPU multiplies the full input by its column block of W and produces its own columns of Y, with no communication.`
              : mode === "row"
                ? `A row-parallel layer on ${P} GPUs: each GPU multiplies its slice of the input features by its row block of W, producing a full-size partial sum, and an all-reduce adds them.`
                : `A column-parallel gate projection followed by a row-parallel down projection on ${P} GPUs. Each GPU keeps its slice of the intermediate H, and one all-reduce after the down projection produces Y on every GPU.`
          }
        >
          <defs>
            <ArrowMarker id={arrow} />
          </defs>
          {columns.map((c, k) =>
            c.head.base ? (
              <Sym key={k} x={cx(k)} y={16} base={c.head.base} sub={c.head.sub} sup={c.head.sup} style={{ fill: color.fg }} />
            ) : null,
          )}
          {Array.from({ length: P }, (_, p) => {
            const y = laneY(p);
            return (
              <g key={p}>
                <text x={0} y={y + boxH / 2 + 4} style={{ fill: RANK_TONES[p], fontWeight: 600 }}>
                  {narrow ? p : `GPU ${p}`}
                </text>
                {columns.map((c, k) => {
                  if (k === collective) return null;
                  const x = cx(k) - boxW / 2;
                  return (
                    <g key={k}>
                      {c.op && (
                        <text x={cx(k) - slot / 2} y={y + boxH / 2 + 4} textAnchor="middle" style={{ fill: color.muted }}>
                          {c.op}
                        </text>
                      )}
                      <Block kind={c.kind} x={x} y={y} w={boxW} h={boxH} rank={p} P={P} />
                    </g>
                  );
                })}
                {/* Into and out of the collective. */}
                <line
                  x1={cx(collective - 1) + boxW / 2 + 2}
                  x2={cx(collective) - cw / 2 - 2}
                  y1={y + boxH / 2}
                  y2={y + boxH / 2}
                  markerEnd={`url(#${arrow})`}
                  style={{ stroke: color.subtle, strokeDasharray: isReduce ? undefined : "3 3" }}
                />
                <line
                  x1={cx(collective) + cw / 2 + 2}
                  x2={cx(collective + 1) - boxW / 2 - 2}
                  y1={y + boxH / 2}
                  y2={y + boxH / 2}
                  markerEnd={`url(#${arrow})`}
                  style={{ stroke: color.subtle, strokeDasharray: isReduce ? undefined : "3 3" }}
                />
              </g>
            );
          })}
          {/* The collective spans every GPU. */}
          <rect
            x={cx(collective) - cw / 2}
            y={laneY(0) - 4}
            width={cw}
            height={laneY(P - 1) + boxH + 8 - laneY(0)}
            rx={4}
            style={
              isReduce
                ? { fill: wash(color.accent, 14), stroke: color.accent, strokeWidth: 1.5 }
                : { fill: "none", stroke: color.subtle, strokeDasharray: "4 3" }
            }
          />
          <text
            transform={`translate(${cx(collective) + 4}, ${(laneY(0) + laneY(P - 1) + boxH) / 2}) rotate(-90)`}
            textAnchor="middle"
            style={{ fill: isReduce ? color.accent : color.subtle, fontWeight: 600 }}
          >
            {isReduce ? "all-reduce: add" : "all-gather, if needed"}
          </text>
        </svg>
      </div>
    </Figure>
  );
}
