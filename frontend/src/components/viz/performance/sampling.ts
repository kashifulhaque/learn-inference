// The sampling transforms from chapter 11, on a plain array of logits, for the
// chapter's figures. Each filter returns a keep mask, and `distribution`
// renormalizes over it, which is what masking to -inf and taking a softmax does.

/** Softmax with the row maximum subtracted, over the kept entries only. */
export function softmax(logits: readonly number[], keep?: readonly boolean[]): number[] {
  const kept = logits.map((_, i) => keep?.[i] ?? true);
  const max = Math.max(...logits.filter((_, i) => kept[i]));
  const exps = logits.map((z, i) => (kept[i] ? Math.exp(z - max) : 0));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

/** Divides by the temperature. Temperature 0 is greedy, handled by the caller. */
export function temper(logits: readonly number[], temperature: number): number[] {
  return logits.map((z) => z / temperature);
}

/** Keeps only the argmax, as the engine does for temperature 0. */
export function greedyMask(logits: readonly number[]): boolean[] {
  const max = Math.max(...logits);
  const first = logits.indexOf(max);
  return logits.map((_, i) => i === first);
}

/** Top-k: keeps every logit at or above the k-th largest, so ties stay in. */
export function topKMask(logits: readonly number[], k: number): boolean[] {
  if (k <= 0 || k >= logits.length) return logits.map(() => true);
  const threshold = [...logits].sort((a, b) => b - a)[k - 1];
  return logits.map((z) => z >= threshold);
}

/** Top-p: keeps rank r while the mass strictly before it, C_{r-1}, is below p. */
export function topPMask(logits: readonly number[], p: number): boolean[] {
  const probs = softmax(logits);
  const order = probs.map((_, i) => i).sort((a, b) => probs[b] - probs[a]);
  const keep = logits.map(() => false);
  let before = 0;
  order.forEach((i, rank) => {
    if (rank === 0 || before < p) keep[i] = true;
    before += probs[i];
  });
  return keep;
}

/** Min-p: keeps every token at least min_p times as likely as the top one. */
export function minPMask(logits: readonly number[], minP: number): boolean[] {
  if (minP <= 0) return logits.map(() => true);
  const probs = softmax(logits);
  const threshold = Math.max(...probs) * minP;
  return probs.map((q) => q >= threshold);
}

/** The sum of `probs` over the kept entries. */
export function keptMass(probs: readonly number[], keep: readonly boolean[]): number {
  return probs.reduce((sum, q, i) => sum + (keep[i] ? q : 0), 0);
}
