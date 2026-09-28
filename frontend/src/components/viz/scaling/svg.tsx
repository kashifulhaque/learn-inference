// Small drawing helpers shared by the scaling chapters' figures (16 to 20).

import { useId, type CSSProperties, type ReactNode } from "react";
import { color } from "../kit";

/** A symbol with a subscript, such as W_p, as SVG text. */
export function Sym({
  x,
  y,
  base,
  sub,
  sup,
  anchor = "middle",
  style,
}: {
  x: number;
  y: number;
  base: string;
  sub?: string;
  sup?: string;
  anchor?: "start" | "middle" | "end";
  style?: CSSProperties;
}) {
  return (
    <text x={x} y={y} textAnchor={anchor} style={{ fontStyle: "italic", ...style }}>
      {base}
      {sub && (
        <tspan dy={3} style={{ fontSize: 9 }}>
          {sub}
        </tspan>
      )}
      {sup && (
        <tspan dy={sub ? -8 : -5} style={{ fontSize: 9 }}>
          {sup}
        </tspan>
      )}
    </text>
  );
}

/**
 * Pushes label positions apart so that none sits closer than `gap` pixels to
 * its neighbour, keeping them inside [lo, hi]. Returns the new positions in the
 * input order.
 */
export function spreadLabels(ys: readonly number[], gap: number, lo: number, hi: number): number[] {
  const order = ys.map((y, i) => ({ y, i })).sort((a, b) => a.y - b.y);
  for (let k = 0; k < order.length; k++) {
    const min = k === 0 ? lo : order[k - 1].y + gap;
    order[k].y = Math.max(order[k].y, min);
  }
  for (let k = order.length - 1; k >= 0; k--) {
    const max = k === order.length - 1 ? hi : order[k + 1].y - gap;
    order[k].y = Math.min(order[k].y, max);
  }
  const out = new Array<number>(ys.length);
  for (const { y, i } of order) out[i] = y;
  return out;
}

/** Round half to even, as `torch.round` does. */
export function roundHalfEven(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** A document-unique id that is safe inside url(#...). */
export function useSvgId(prefix: string): string {
  return `${prefix}-${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;
}

/** An arrowhead marker, coloured with a theme token. Reference it as url(#id). */
export function ArrowMarker({ id, tone = color.subtle }: { id: string; tone?: string }) {
  return (
    <marker id={id} viewBox="0 0 8 8" refX={7} refY={4} markerWidth={7} markerHeight={7} orient="auto-start-reverse">
      <path d="M0,0 L8,4 L0,8 z" style={{ fill: tone }} />
    </marker>
  );
}

/** Horizontal gridline ticks for a linear y axis. */
export function YTicks({
  ticks,
  y,
  x0,
  x1,
  format = (v: number) => String(v),
}: {
  ticks: readonly number[];
  y: (v: number) => number;
  x0: number;
  x1: number;
  format?: (v: number) => ReactNode;
}) {
  return (
    <g>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x0} x2={x1} y1={y(t)} y2={y(t)} style={{ stroke: color.line }} />
          <text x={x0 - 5} y={y(t) + 4} textAnchor="end">
            {format(t)}
          </text>
        </g>
      ))}
    </g>
  );
}
