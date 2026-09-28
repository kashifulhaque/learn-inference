// Small plotting helpers shared by this group's figures. Kept here rather than
// in kit.tsx so that the shared kit stays unchanged.

/** A linear map from [d0, d1] onto [r0, r1]. */
export function linear(d0: number, d1: number, r0: number, r1: number) {
  return (value: number) => r0 + ((value - d0) / (d1 - d0)) * (r1 - r0);
}

/** A base-10 logarithmic map from [d0, d1] onto [r0, r1]. Both ends must be positive. */
export function log10Scale(d0: number, d1: number, r0: number, r1: number) {
  const l0 = Math.log10(d0);
  const l1 = Math.log10(d1);
  return (value: number) => r0 + ((Math.log10(value) - l0) / (l1 - l0)) * (r1 - r0);
}

/** The powers of ten from `lo` to `hi` inclusive, for log-axis ticks. */
export function decades(lo: number, hi: number): number[] {
  const ticks: number[] = [];
  for (let e = Math.ceil(Math.log10(lo) - 1e-9); 10 ** e <= hi * (1 + 1e-9); e += 1) ticks.push(10 ** e);
  return ticks;
}

/** A short label for a power of ten: 1, 10, 100, 1k, 10k. */
export function decadeLabel(value: number): string {
  if (value >= 1e6) return `${value / 1e6}M`;
  if (value >= 1e3) return `${value / 1e3}k`;
  return String(value);
}

/** Probabilities with four decimals, as the chapters write them. */
export function prob(value: number): string {
  return value.toFixed(4);
}
