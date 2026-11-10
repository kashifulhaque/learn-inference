import torch


def accept_probability(p: torch.Tensor, q: torch.Tensor, token: int) -> float:
    q_x = q[token].item()
    if q_x <= 0.0:
        return 1.0
    return min(1.0, p[token].item() / q_x)


def residual_distribution(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    residual = torch.clamp(p - q, min=0.0)
    total = residual.sum()
    if total <= 0:
        return p / p.sum()
    return residual / total


def verify(draft_tokens: list, draft_probs: list, target_probs: list,
           generator: torch.Generator | None = None) -> tuple[list, int]:
    accepted: list[int] = []
    for index, token in enumerate(draft_tokens):
        p = target_probs[index]
        q = draft_probs[index]
        threshold = accept_probability(p, q, token)
        draw = torch.rand((), generator=generator, device=p.device).item()
        if draw < threshold:
            accepted.append(int(token))
            continue
        # Rejected: correct this position and discard the rest of the draft.
        residual = residual_distribution(p, q)
        corrected = torch.multinomial(residual, 1, generator=generator).item()
        accepted.append(int(corrected))
        return accepted, index

    # Every draft token survived, so the target's final distribution is free.
    bonus = torch.multinomial(target_probs[-1], 1, generator=generator).item()
    accepted.append(int(bonus))
    return accepted, len(draft_tokens)


def expected_tokens_per_round(alpha: float, k: int) -> float:
    if abs(alpha - 1.0) < 1e-12:
        return float(k + 1)
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)


def net_speedup(alpha: float, k: int, draft_cost_ratio: float) -> float:
    return expected_tokens_per_round(alpha, k) / (1.0 + k * draft_cost_ratio)


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

    def snapshot(self) -> tuple[int, torch.Tensor]:
        """Capture what a rollback needs: the KV length and a copy of the state."""
        return self.length, self.state.clone()

    def rollback(self, snapshot: tuple[int, torch.Tensor],
                 replay: list[torch.Tensor]) -> None:
        """Return to `snapshot`, then re-run only the tokens in `replay`."""
        length, state = snapshot
        del self.keys[length:]
        del self.values[length:]
        self.length = length
        self.state.copy_(state)
        for x in replay:
            self.step(x)


def speculative_round(cache: ToyHybridCache, inputs: list[torch.Tensor],
                      num_accepted: int) -> list[torch.Tensor]:
    """Verify `inputs` in one pass, then commit only the accepted prefix.

    `inputs` holds the last committed token followed by the k draft tokens,
    k + 1 embeddings in all. The verify pass processes all of them and returns
    their k + 1 outputs. Then `num_accepted` draft tokens survive, so the
    cache keeps the first num_accepted + 1 inputs and nothing else.
    """
    snap = cache.snapshot()
    outputs = [cache.step(x) for x in inputs]
    cache.rollback(snap, inputs[:num_accepted + 1])
    return outputs


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

    The pass reads the weights once and does 2 * params FLOPs per token. It
    takes the longer of the two.
    """
    memory = WEIGHT_BYTES / BANDWIDTH
    compute = 2 * ACTIVE_PARAMS * batch * tokens_per_sequence / PEAK_FLOPS
    return max(memory, compute)


def speculative_speedup(batch: int, alpha: float, k: int,
                        draft_cost_ratio: float,
                        snapshot_bytes: int = 0) -> float:
    """Net speedup of speculation at a batch size, against plain decoding.

    A round costs k draft passes, one verify pass over k + 1 tokens per
    sequence, and a snapshot copy of `snapshot_bytes` per sequence (read and
    written). Each draft pass costs `draft_cost_ratio` of a plain decode step
    at the same batch size. A round yields expected_tokens_per_round tokens.
    """
    plain = step_time(batch, 1)
    draft = k * draft_cost_ratio * plain
    verify = step_time(batch, k + 1)
    snapshot = batch * 2 * snapshot_bytes / BANDWIDTH
    return expected_tokens_per_round(alpha, k) * plain / (draft + verify + snapshot)


def break_even_batch(alpha: float, k: int, draft_cost_ratio: float,
                     snapshot_bytes: int = 0, max_batch: int = 4096) -> int:
    """The largest batch size at which speculation still has speedup >= 1.

    Return 0 when it loses even at batch 1, and `max_batch` when it never loses
    below that.
    """
    best = 0
    for batch in range(1, max_batch + 1):
        if speculative_speedup(batch, alpha, k, draft_cost_ratio,
                               snapshot_bytes) < 1.0:
            return best
        best = batch
    return best
