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
