import torch


def apply_repetition_penalty(logits: torch.Tensor, previous: torch.Tensor,
                             penalty: float) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    scores = torch.gather(logits, 1, previous)
    scores = torch.where(scores > 0, scores / penalty, scores * penalty)
    return logits.scatter(1, previous, scores)


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    threshold = logits.topk(k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
    probs = sorted_logits.softmax(dim=-1)
    cumulative = probs.cumsum(dim=-1)
    remove = cumulative - probs >= p
    remove[..., 0] = False
    return logits.masked_fill(remove.scatter(-1, sorted_idx, remove), float("-inf"))


def min_p_filter(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    if min_p <= 0.0:
        return logits
    probs = logits.softmax(dim=-1)
    threshold = probs.max(dim=-1, keepdim=True).values * min_p
    return logits.masked_fill(probs < threshold, float("-inf"))


def sample(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0,
           top_p: float = 1.0, min_p: float = 0.0,
           previous: torch.Tensor | None = None,
           repetition_penalty: float = 1.0,
           generator: torch.Generator | None = None) -> torch.Tensor:
    logits = logits.float()
    if previous is not None:
        logits = apply_repetition_penalty(logits, previous, repetition_penalty)
    if temperature <= 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature
    logits = top_k_filter(logits, top_k)
    logits = min_p_filter(logits, min_p)
    logits = top_p_filter(logits, top_p)
    return torch.multinomial(logits.softmax(dim=-1), num_samples=1,
                             generator=generator)


# --- Batched sampling, sort-free top-p, and the logits-processor hook --------


def apply_logits_processors(input_ids: torch.Tensor, logits: torch.Tensor,
                            processors) -> torch.Tensor:
    """Run each `(input_ids, logits) -> logits` callable in list order."""
    for processor in processors or ():
        logits = processor(input_ids, logits)
    return logits


def top_p_filter_sortfree(logits: torch.Tensor, p, iters: int = 40) -> torch.Tensor:
    """Top-p without a sort: bisect on a probability threshold.

    `p` is a float or a (batch,) tensor. Find the largest threshold t whose
    kept mass, the sum of probabilities at or above t, still reaches p. The
    masses are summed in float64, because a float32 sum over 248,320 entries
    drifts by about 6e-5, which moves the p = 0.99 cutoff by hundreds of tokens. The
    keep-set is every token with probability >= t. Ties at the boundary are
    all kept, which a sort would break arbitrarily.
    """
    probs = logits.double().softmax(dim=-1)   # float64: see the chapter
    p = torch.as_tensor(p, dtype=probs.dtype, device=probs.device)
    p = p.expand(probs.shape[0]).unsqueeze(-1)
    lo = torch.zeros_like(p)                      # mass(lo) = 1 >= p always
    hi = probs.max(dim=-1, keepdim=True).values   # mass(hi) may be < p
    for _ in range(iters):
        mid = (lo + hi) / 2
        mass = torch.where(probs >= mid, probs, torch.zeros_like(probs)).sum(-1, keepdim=True)
        ok = mass >= p
        lo = torch.where(ok, mid, lo)
        hi = torch.where(ok, hi, mid)
    keep = probs >= lo
    keep = keep | (probs >= probs.max(dim=-1, keepdim=True).values)  # never empty
    keep = torch.where(p >= 1.0, torch.ones_like(keep), keep)
    return logits.masked_fill(~keep, float("-inf"))


def filter_batched(logits: torch.Tensor, temperature: torch.Tensor,
                   top_k: torch.Tensor, top_p: torch.Tensor,
                   sortfree: bool = False) -> torch.Tensor:
    """Temperature, then top-k, then top-p, with a separate setting per row.

    temperature, top_k, top_p are (batch,) tensors. A top_k of 0 or less
    disables top-k for that row. Rows with temperature <= 0 are scaled by 1;
    the caller handles greedy rows.
    """
    logits = logits.float()
    vocab = logits.shape[-1]
    temp = torch.where(temperature > 0, temperature, torch.ones_like(temperature))
    logits = logits / temp.unsqueeze(-1)

    k = torch.where((top_k <= 0) | (top_k >= vocab),
                    torch.full_like(top_k, vocab), top_k)
    kth = logits.sort(dim=-1, descending=True).values.gather(1, (k - 1).unsqueeze(-1))
    logits = logits.masked_fill(logits < kth, float("-inf"))

    if sortfree:
        return top_p_filter_sortfree(logits, top_p)
    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
    probs = sorted_logits.softmax(dim=-1)
    remove = (probs.cumsum(dim=-1) - probs >= top_p.unsqueeze(-1)) & (top_p < 1.0).unsqueeze(-1)
    remove[..., 0] = False
    return logits.masked_fill(remove.scatter(-1, sorted_idx, remove), float("-inf"))


def sample_batched(logits: torch.Tensor, temperature: torch.Tensor,
                   top_k: torch.Tensor, top_p: torch.Tensor,
                   input_ids: torch.Tensor | None = None,
                   processors=None, sortfree: bool = False,
                   generator: torch.Generator | None = None) -> torch.Tensor:
    """Draw one token per row with per-row parameters; returns (batch, 1).

    Processors run first, on the raw logits, before temperature.
    """
    logits = apply_logits_processors(input_ids, logits.float(), processors)
    greedy = logits.argmax(dim=-1, keepdim=True)
    filtered = filter_batched(logits, temperature, top_k, top_p, sortfree)
    drawn = torch.multinomial(filtered.softmax(dim=-1), num_samples=1,
                              generator=generator)
    return torch.where((temperature <= 0).unsqueeze(-1), greedy, drawn)
