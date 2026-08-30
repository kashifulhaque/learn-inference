"""Lab 20 — speculative decoding.

The rejection rule must preserve the target's distribution exactly. An
approximation here changes what the model produces, and no ordinary test
catches it — which is why this lab checks the distribution statistically.
"""

import torch


def accept_probability(p: torch.Tensor, q: torch.Tensor, token: int) -> float:
    """Probability of accepting a draft token.

    Args:
        p: Target distribution over the vocabulary.
        q: Draft distribution over the same vocabulary.
        token: The token the draft proposed.
    """
    # TODO
    raise NotImplementedError


def residual_distribution(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """The distribution to sample from after a rejection.

    Take the positive part of p - q and normalize it. When the positive part
    sums to zero, fall back to p.
    """
    # TODO
    raise NotImplementedError


def verify(draft_tokens: list, draft_probs: list, target_probs: list,
           generator: torch.Generator | None = None) -> tuple[list, int]:
    """Accept a prefix of the draft, then correct or extend.

    Args:
        draft_tokens: The k tokens the draft proposed.
        draft_probs: k distributions, one per draft position.
        target_probs: k + 1 distributions from the target model. Entry i is the
            distribution at the position where draft_tokens[i] was proposed, and
            the last entry is the distribution after the whole draft.

    Returns:
        (accepted_tokens, num_draft_accepted). On a rejection at position i,
        return the first i accepted tokens plus one corrected token. When every
        draft token is accepted, append a bonus token drawn from the final
        target distribution, so the round yields k + 1 tokens.
    """
    # TODO
    raise NotImplementedError


def expected_tokens_per_round(alpha: float, k: int) -> float:
    """Expected tokens from one round at acceptance rate alpha with k drafts.

    The sum of alpha^i for i from 0 to k, which is
    (1 - alpha^(k+1)) / (1 - alpha) when alpha is not 1.
    """
    # TODO
    raise NotImplementedError


def net_speedup(alpha: float, k: int, draft_cost_ratio: float) -> float:
    """Speedup against plain decoding.

    One round costs one target pass plus k draft passes, each costing
    `draft_cost_ratio` of a target pass. Plain decoding produces one token per
    target pass.
    """
    # TODO
    raise NotImplementedError
