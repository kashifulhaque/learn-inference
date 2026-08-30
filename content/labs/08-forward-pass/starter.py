"""Lab 08 — validating a forward pass.

You have a model. Now prove it is right. These are the three tools you reach
for whenever an implementation disagrees with a reference.
"""

import torch


def logit_metrics(mine: torch.Tensor, reference: torch.Tensor) -> dict:
    """Compare two logit tensors of the same shape.

    Returns a dict with:
        max_abs_diff:   largest absolute difference anywhere.
        correlation:    Pearson correlation over all flattened values.
        top1_agreement: fraction of positions whose argmax matches.
        mean_kl:        mean KL divergence from reference to mine, in nats,
                        over the last dimension.

    Compute in float32 whatever the inputs are.
    """
    # TODO
    raise NotImplementedError


def first_divergent_layer(mine: list[torch.Tensor], reference: list[torch.Tensor],
                          tol: float = 1e-2) -> int | None:
    """Return the index of the first layer whose hidden state exceeds `tol`.

    Args:
        mine, reference: One tensor per layer, in order.

    Returns:
        The index, or None when every layer agrees.
    """
    # TODO
    raise NotImplementedError


def prefill_decode_equivalence(model, input_ids, prefill_len: int,
                               make_cache) -> float:
    """Check that prefill plus single steps equals one full forward pass.

    Run `model(input_ids)` with no cache to get reference logits. Then create a
    cache with `make_cache()`, prefill `input_ids[:, :prefill_len]`, and feed the
    remaining tokens one at a time. Return the largest absolute difference
    between the stepped logits and the reference logits at the same positions.

    The model signature is model(ids, cache=..., last_token_only=...), and it
    returns logits of shape (batch, positions, vocab).
    """
    # TODO
    raise NotImplementedError
