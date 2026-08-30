"""Turning logits into tokens.

The order of operations matters and is easy to get wrong. Penalties act on raw
logits, before temperature. Temperature scales logits, not probabilities.
Truncation (top-k, top-p, min-p) happens after the softmax, on the sorted
distribution. Swap any two and you get a sampler that works but does not do
what its knobs claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int | None = None

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0


def apply_repetition_penalty(
    logits: Tensor, previous: Tensor, penalty: float
) -> Tensor:
    """Divide positive logits of seen tokens, multiply negative ones.

    The asymmetry is the point: dividing a negative logit would make it larger
    and encourage the token you meant to discourage.
    """
    if penalty == 1.0:
        return logits
    scores = torch.gather(logits, 1, previous)
    scores = torch.where(scores > 0, scores / penalty, scores * penalty)
    return logits.scatter(1, previous, scores)


def apply_presence_frequency_penalty(
    logits: Tensor, counts: Tensor, presence: float, frequency: float
) -> Tensor:
    """Subtract a flat amount for having appeared, plus a per-occurrence amount.

    Args:
        counts: (batch, vocab) occurrence counts so far.
    """
    if presence == 0.0 and frequency == 0.0:
        return logits
    return logits - presence * (counts > 0).to(logits.dtype) - frequency * counts.to(
        logits.dtype
    )


def top_k_filter(logits: Tensor, k: int) -> Tensor:
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    threshold = logits.topk(k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: Tensor, p: float) -> Tensor:
    """Keep the smallest set of tokens whose probabilities sum to at least p."""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
    cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # Shift so the token that crosses the threshold is itself kept.
    remove = cumulative - sorted_logits.softmax(dim=-1) >= p
    remove[..., 0] = False
    return logits.masked_fill(
        remove.scatter(-1, sorted_idx, remove), float("-inf")
    )


def min_p_filter(logits: Tensor, min_p: float) -> Tensor:
    """Drop tokens less likely than `min_p` times the most likely one.

    Unlike top-p this adapts to how sharp the distribution is: when the model is
    confident it keeps almost nothing, when it is unsure it keeps a lot.
    """
    if min_p <= 0.0:
        return logits
    probs = logits.softmax(dim=-1)
    threshold = probs.max(dim=-1, keepdim=True).values * min_p
    return logits.masked_fill(probs < threshold, float("-inf"))


def sample(
    logits: Tensor,
    params: SamplingParams,
    previous: Tensor | None = None,
    counts: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample one token per row of `logits`, shaped (batch, vocab)."""
    logits = logits.float()

    if previous is not None:
        logits = apply_repetition_penalty(logits, previous, params.repetition_penalty)
    if counts is not None:
        logits = apply_presence_frequency_penalty(
            logits, counts, params.presence_penalty, params.frequency_penalty
        )

    if params.greedy:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / params.temperature
    logits = top_k_filter(logits, params.top_k)
    logits = min_p_filter(logits, params.min_p)
    logits = top_p_filter(logits, params.top_p)

    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


def stable_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """softmax with the max subtracted first.

    exp(x) overflows float32 above about 88. Real logits reach into the tens,
    and attention scores at long context reach further. Subtracting the row max
    leaves every exponent at most 0 and changes nothing mathematically, since
    the constant cancels between numerator and denominator. Every fast softmax
    and every attention kernel does this; chapter 11 does it incrementally.
    """
    shifted = x - x.max(dim=dim, keepdim=True).values
    exp = shifted.exp()
    return exp / exp.sum(dim=dim, keepdim=True)
