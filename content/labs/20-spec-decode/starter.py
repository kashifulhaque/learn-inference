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


# ---------------------------------------------------------------------------
# Rolling back a hybrid cache
# ---------------------------------------------------------------------------


class ToyHybridCache:
    """A tiny hybrid layer: one attention head over a KV list, plus a
    gated-delta recurrent state.

    The KV list grows by one entry per token, so a rejected token can be
    dropped by truncating it. The state is a fixed (dim, dim) matrix that
    `step` overwrites in place, so a rejected token cannot be un-applied.
    This class mirrors the two kinds of per-sequence memory in lab 09's
    `Cache`: `length` and the KV tensors for the attention layers, and the
    recurrent state for the linear layers.
    """

    def __init__(self, dim: int = 8, seed: int = 0, decay: float = 0.9,
                 beta: float = 0.5) -> None:
        g = torch.Generator().manual_seed(seed)
        scale = dim ** -0.5
        self.dim = dim
        self.decay = decay
        self.beta = beta
        self.wq = torch.randn(dim, dim, generator=g) * scale
        self.wk = torch.randn(dim, dim, generator=g) * scale
        self.wv = torch.randn(dim, dim, generator=g) * scale
        self.length = 0
        self.keys: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.state = torch.zeros(dim, dim)

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """Process one token embedding `x` of shape (dim,) and return its output."""
        q = self.wq @ x
        k = self.wk @ x
        k = k / (k.norm() + 1e-6)
        v = self.wv @ x
        self.keys.append(k)
        self.values.append(v)
        self.length += 1
        # Gated delta update, in place: S <- decay * (S - beta k (k^T S)) + beta k v^T.
        # The old state is gone after this line.
        self.state.copy_(
            self.decay * (self.state - self.beta * torch.outer(k, k @ self.state))
            + self.beta * torch.outer(k, v)
        )
        scores = torch.stack(self.keys) @ q * (self.dim ** -0.5)
        attention = torch.softmax(scores, dim=0) @ torch.stack(self.values)
        return attention + q @ self.state

    def snapshot(self):
        """Capture what a rollback needs: the KV length and a copy of the state."""
        # TODO
        raise NotImplementedError

    def rollback(self, snapshot, replay: list) -> None:
        """Return to `snapshot`, then re-run only the tokens in `replay`.

        Truncate the KV list back to the snapshot's length, restore the state,
        then call `step` on each embedding in `replay`.
        """
        # TODO
        raise NotImplementedError


def speculative_round(cache: ToyHybridCache, inputs: list,
                      num_accepted: int) -> list:
    """Verify `inputs` in one pass, then commit only the accepted prefix.

    `inputs` holds the last committed token followed by the k draft tokens,
    k + 1 embeddings in all. The verify pass processes all of them and returns
    their k + 1 outputs. Then `num_accepted` draft tokens survive, so the
    cache keeps the first num_accepted + 1 inputs and nothing else.
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Cost model: when does speculation stop paying?
# ---------------------------------------------------------------------------

# A100 80GB PCIe, the rated pair from chapter 10 and chapter 17a.
WEIGHT_BYTES = 53.8e9
ACTIVE_PARAMS = 26.9e9
BANDWIDTH = 1935e9
PEAK_FLOPS = 312e12
# Recurrent state per sequence, from chapter 15's swap arithmetic.
STATE_BYTES = 154_927_104


def step_time(batch: int, tokens_per_sequence: int) -> float:
    """Seconds for one target forward pass over batch * tokens_per_sequence tokens.

    The pass reads the weights once and does 2 * ACTIVE_PARAMS FLOPs per token.
    It takes the longer of the memory time and the compute time.
    """
    # TODO
    raise NotImplementedError


def speculative_speedup(batch: int, alpha: float, k: int,
                        draft_cost_ratio: float,
                        snapshot_bytes: int = 0) -> float:
    """Net speedup of speculation at a batch size, against plain decoding.

    A round costs k draft passes, one verify pass over k + 1 tokens per
    sequence, and a snapshot copy of `snapshot_bytes` per sequence (read and
    written, so 2 * snapshot_bytes / BANDWIDTH each). Each draft pass costs
    `draft_cost_ratio` of a plain decode step at the same batch size. A round
    yields expected_tokens_per_round tokens per sequence.
    """
    # TODO
    raise NotImplementedError


def break_even_batch(alpha: float, k: int, draft_cost_ratio: float,
                     snapshot_bytes: int = 0, max_batch: int = 4096) -> int:
    """The largest batch size at which speculation still has speedup >= 1.

    Return 0 when it loses even at batch 1, and `max_batch` when it never loses
    below that.
    """
    # TODO
    raise NotImplementedError
