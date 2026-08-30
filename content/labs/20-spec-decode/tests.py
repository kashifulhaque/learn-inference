"""Checks lab 20, including a statistical test that the output distribution is
unchanged. That property is the whole reason speculative decoding is safe.
"""

import torch
from lab_common import Checks, pick_device

VOCAB = 16


def run(submission):
    c = Checks()
    needed = ("accept_probability", "residual_distribution", "verify",
              "expected_tokens_per_round", "net_speedup")
    if not c.require(submission, *needed):
        return c.finish()

    device = pick_device()
    torch.manual_seed(0)

    p = torch.tensor([0.5, 0.3, 0.2], device=device)
    q = torch.tensor([0.25, 0.6, 0.15], device=device)

    c.check("a token the target likes more is always accepted",
            lambda: abs(submission.accept_probability(p, q, 0) - 1.0) < 1e-9,
            f"got {submission.accept_probability(p, q, 0)}")
    c.check("a token the target likes less is accepted with ratio p/q",
            lambda: abs(submission.accept_probability(p, q, 1) - 0.5) < 1e-9,
            f"got {submission.accept_probability(p, q, 1)}")
    c.check("the acceptance probability never exceeds 1",
            lambda: submission.accept_probability(p, q, 2) <= 1.0)

    residual = submission.residual_distribution(p, q)
    c.check("the residual is a distribution",
            lambda: abs(residual.sum().item() - 1.0) < 1e-6,
            f"sums to {residual.sum().item():.6f}")
    c.check("the residual is non-negative",
            lambda: (residual >= -1e-9).all().item())
    c.check("the residual favours what the draft under-weights",
            lambda: residual[0] > residual[1],
            f"got {residual.tolist()}")
    c.check("identical distributions fall back rather than dividing by zero",
            lambda: torch.isfinite(submission.residual_distribution(p, p)).all().item()
            and abs(submission.residual_distribution(p, p).sum().item() - 1.0) < 1e-6)

    # The statistical test. Speculating from a deliberately poor draft must
    # still reproduce the target's distribution.
    torch.manual_seed(1)
    target = torch.rand(VOCAB, device=device) + 0.05
    target = target / target.sum()
    draft = torch.rand(VOCAB, device=device) + 0.05
    draft = draft / draft.sum()

    k = 4
    rounds = 40000
    counts = torch.zeros(VOCAB, device=device)
    accepted_total = 0
    tokens_total = 0
    generator = torch.Generator(device=device).manual_seed(42)

    for _ in range(rounds):
        draft_tokens = torch.multinomial(
            draft, k, replacement=True, generator=generator).tolist()
        tokens, num_accepted = submission.verify(
            draft_tokens, [draft] * k, [target] * (k + 1), generator=generator)
        # Only the first token of a round is compared: it is the one whose
        # distribution the rejection rule is supposed to make exact.
        counts[tokens[0]] += 1
        accepted_total += num_accepted
        tokens_total += len(tokens)

    empirical = counts / counts.sum()
    total_variation = 0.5 * (empirical - target).abs().sum().item()
    acceptance = accepted_total / (rounds * k)
    mean_tokens = tokens_total / rounds

    c.check(
        "the accepted token follows the target distribution, not the draft's",
        lambda: total_variation < 0.02,
        f"total variation distance {total_variation:.4f}",
    )
    c.check(
        "the output is closer to the target than to the draft",
        lambda: total_variation < 0.5 * (empirical - draft).abs().sum().item(),
        f"target {total_variation:.4f} vs draft "
        f"{0.5 * (empirical - draft).abs().sum().item():.4f}",
    )
    c.check("a round yields between 1 and k + 1 tokens",
            lambda: 1.0 <= mean_tokens <= k + 1,
            f"mean {mean_tokens:.2f}")

    # A perfect draft accepts everything and yields k + 1 every round.
    generator = torch.Generator(device=device).manual_seed(7)
    perfect_rounds = 500
    perfect_total = 0
    for _ in range(perfect_rounds):
        tokens = torch.multinomial(
            target, k, replacement=True, generator=generator).tolist()
        out, num_accepted = submission.verify(
            tokens, [target] * k, [target] * (k + 1), generator=generator)
        perfect_total += num_accepted
    perfect_rate = perfect_total / (perfect_rounds * k)
    c.check("a draft identical to the target is always accepted",
            lambda: perfect_rate > 0.99, f"acceptance {perfect_rate:.3f}")

    c.check("expected tokens at alpha=0 is 1",
            lambda: abs(submission.expected_tokens_per_round(0.0, 5) - 1.0) < 1e-9,
            f"got {submission.expected_tokens_per_round(0.0, 5)}")
    c.check("expected tokens at alpha=1 is k + 1",
            lambda: abs(submission.expected_tokens_per_round(1.0, 5) - 6.0) < 1e-6,
            f"got {submission.expected_tokens_per_round(1.0, 5)}")
    c.check("alpha=0.9 with k=5 yields about 4.69 tokens",
            lambda: abs(submission.expected_tokens_per_round(0.9, 5) - 4.6856) < 1e-3,
            f"got {submission.expected_tokens_per_round(0.9, 5):.4f}")

    c.check("a cheap draft with high acceptance beats plain decoding",
            lambda: submission.net_speedup(0.8, 4, 0.05) > 2.0,
            f"got {submission.net_speedup(0.8, 4, 0.05):.2f}x")
    c.check("an expensive draft is not worth it",
            lambda: submission.net_speedup(0.8, 4, 0.5) < 1.2,
            f"got {submission.net_speedup(0.8, 4, 0.5):.2f}x")
    c.check("longer drafts saturate at low acceptance",
            lambda: submission.expected_tokens_per_round(0.5, 8)
            - submission.expected_tokens_per_round(0.5, 3) < 0.2,
            f"k=3 {submission.expected_tokens_per_round(0.5, 3):.3f}, "
            f"k=8 {submission.expected_tokens_per_round(0.5, 8):.3f}")

    c.metric("acceptance_rate", round(acceptance, 4))
    c.metric("distribution_error", round(total_variation, 5))
    c.metric("expected_tokens", round(mean_tokens, 3))
    return c.finish()
