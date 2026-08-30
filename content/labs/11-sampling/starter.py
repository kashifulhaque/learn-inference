"""Lab 11 — sampling.

Order: penalties, then temperature, then top-k, min-p, and top-p, then draw.
"""

import torch


def apply_repetition_penalty(logits: torch.Tensor, previous: torch.Tensor,
                             penalty: float) -> torch.Tensor:
    """Discourage tokens that have already appeared.

    Args:
        logits: (batch, vocab)
        previous: (batch, n) token ids seen so far.

    Divide a positive logit by `penalty`; multiply a negative one by it. Return a
    new tensor; do not modify `logits` in place.
    """
    # TODO
    raise NotImplementedError


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k highest logits per row and set the rest to -inf.

    A k of 0, or one at least as large as the vocabulary, changes nothing.
    """
    # TODO
    raise NotImplementedError


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Keep the smallest set of tokens whose probabilities reach p.

    The token that crosses the threshold is kept. At least one token always
    survives.
    """
    # TODO
    raise NotImplementedError


def min_p_filter(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    """Drop tokens less probable than min_p times the most probable one."""
    # TODO
    raise NotImplementedError


def sample(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0,
           top_p: float = 1.0, min_p: float = 0.0,
           previous: torch.Tensor | None = None,
           repetition_penalty: float = 1.0,
           generator: torch.Generator | None = None) -> torch.Tensor:
    """Draw one token per row, returning shape (batch, 1).

    A temperature of 0 or less means greedy: return the argmax.
    """
    # TODO
    raise NotImplementedError
