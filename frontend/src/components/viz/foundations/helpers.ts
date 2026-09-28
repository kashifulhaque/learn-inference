// Helpers shared by the foundations figures (chapters 0 to 4): axis maps and an
// exact floating-point rounding model. Kept here so the shared kit stays
// unchanged.

/** A linear map from [d0, d1] onto [r0, r1]. */
export function linear(d0: number, d1: number, r0: number, r1: number) {
  return (value: number) => r0 + ((value - d0) / (d1 - d0)) * (r1 - r0);
}

/** A logarithmic map from [d0, d1] onto [r0, r1]. Both ends must be positive. */
export function logarithmic(d0: number, d1: number, r0: number, r1: number) {
  const l0 = Math.log(d0);
  const l1 = Math.log(d1);
  return (value: number) => r0 + ((Math.log(value) - l0) / (l1 - l0)) * (r1 - r0);
}

/** An SVG path through the given points. */
export function polyline(points: readonly (readonly [number, number])[]): string {
  return points.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`).join("");
}

/** A binary floating-point format: 1 sign bit, then exponent and mantissa fields. */
export type FloatFormat = { exponentBits: number; mantissaBits: number };

// The field widths from the notation chapter's format table.
export const FLOAT32: FloatFormat = { exponentBits: 8, mantissaBits: 23 };
export const FLOAT16: FloatFormat = { exponentBits: 5, mantissaBits: 10 };
export const BFLOAT16: FloatFormat = { exponentBits: 8, mantissaBits: 7 };

export type Encoded = {
  kind: "zero" | "subnormal" | "normal" | "infinite";
  sign: 0 | 1;
  exponentField: number;
  mantissaField: number;
  /** The unbiased exponent e of a normal number, 2^e <= |value| < 2^(e+1). */
  exponent: number;
  /** The value the format actually stores. */
  value: number;
  /** The gap from this value to the next representable one, or NaN at infinity. */
  ulp: number;
};

function roundHalfEven(v: number): number {
  const floor = Math.floor(v);
  const rest = v - floor;
  if (rest > 0.5) return floor + 1;
  if (rest < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

/**
 * Rounds `x` to the nearest value of `format`, ties to even, as IEEE 754
 * round-to-nearest does. Exact for any float64 input, because every step is a
 * multiplication by a power of two.
 */
export function encode(x: number, format: FloatFormat): Encoded {
  const { exponentBits: eb, mantissaBits: mb } = format;
  const bias = 2 ** (eb - 1) - 1;
  const eMin = 1 - bias;
  const eMax = bias;
  const sign: 0 | 1 = x < 0 ? 1 : 0;
  const s = sign ? -1 : 1;
  const a = Math.abs(x);
  const infinite: Encoded = {
    kind: "infinite",
    sign,
    exponentField: 2 ** eb - 1,
    mantissaField: 0,
    exponent: eMax + 1,
    value: s * Infinity,
    ulp: NaN,
  };
  if (!Number.isFinite(a)) return infinite;
  if (a === 0) {
    return { kind: "zero", sign, exponentField: 0, mantissaField: 0, exponent: eMin, value: 0, ulp: 2 ** (eMin - mb) };
  }

  let e = Math.floor(Math.log2(a));
  if (2 ** e > a) e -= 1;
  else if (2 ** (e + 1) <= a) e += 1;

  if (e < eMin) {
    const m = roundHalfEven(a / 2 ** (eMin - mb));
    if (m === 0) return encode(0, format);
    if (m < 2 ** mb) {
      return {
        kind: "subnormal",
        sign,
        exponentField: 0,
        mantissaField: m,
        exponent: eMin,
        value: s * m * 2 ** (eMin - mb),
        ulp: 2 ** (eMin - mb),
      };
    }
    e = eMin; // rounded up into the smallest normal number
  }

  let m = roundHalfEven((a / 2 ** e - 1) * 2 ** mb);
  if (m === 2 ** mb) {
    m = 0;
    e += 1;
  }
  if (e > eMax) return infinite;
  return {
    kind: "normal",
    sign,
    exponentField: e + bias,
    mantissaField: m,
    exponent: e,
    value: s * (1 + m / 2 ** mb) * 2 ** e,
    ulp: 2 ** (e - mb),
  };
}

/** The bits of an encoded value, most significant first: sign, exponent, mantissa. */
export function bitsOf(encoded: Encoded, format: FloatFormat): number[] {
  const bits = [encoded.sign as number];
  for (let i = format.exponentBits - 1; i >= 0; i -= 1) bits.push(Math.floor(encoded.exponentField / 2 ** i) % 2);
  for (let i = format.mantissaBits - 1; i >= 0; i -= 1) bits.push(Math.floor(encoded.mantissaField / 2 ** i) % 2);
  return bits;
}

/** `x` rounded to bfloat16. */
export function toBfloat16(x: number): number {
  return encode(x, BFLOAT16).value;
}
