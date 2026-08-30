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
