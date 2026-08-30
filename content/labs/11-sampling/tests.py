import torch
from lab_common import Checks

VOCAB = 512


def kept(logits: torch.Tensor) -> int:
    return int(torch.isfinite(logits).sum().item())


def run(submission):
    c = Checks()
    needed = ("apply_repetition_penalty", "top_k_filter", "top_p_filter",
              "min_p_filter", "sample")
    if not c.require(submission, *needed):
        return c.finish()

    torch.manual_seed(0)

    # Probabilities 0.636, 0.234, 0.086, 0.032, 0.012.
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0, -1.0]])

    filtered = submission.top_k_filter(logits, 2)
    c.check("top-k keeps exactly k tokens", lambda: kept(filtered) == 2,
            f"kept {kept(filtered)}")
    c.check("top-k keeps the highest ones",
            lambda: torch.isfinite(filtered[0, :2]).all().item())
    c.check("top-k with k=0 changes nothing",
            lambda: torch.equal(submission.top_k_filter(logits, 0), logits))

    # 0.636 alone falls short of 0.7, so the second token must survive.
    p70 = submission.top_p_filter(logits, 0.7)
    c.check("top-p keeps the token that crosses the threshold",
            lambda: kept(p70) == 2, f"kept {kept(p70)}")
    # 0.636 already exceeds 0.5, so only the first token is needed.
    p50 = submission.top_p_filter(logits, 0.5)
    c.check("top-p keeps one token when the first already exceeds p",
            lambda: kept(p50) == 1, f"kept {kept(p50)}")
    # Even a tiny p must leave something to sample from.
    c.check("top-p never removes everything",
            lambda: kept(submission.top_p_filter(logits, 0.001)) >= 1)
    c.check("top-p with p=1 changes nothing",
            lambda: torch.equal(submission.top_p_filter(logits, 1.0), logits))

    # min-p 0.3 of 0.636 is 0.19, so only the first two survive.
    m30 = submission.min_p_filter(logits, 0.3)
    c.check("min-p thresholds relative to the peak", lambda: kept(m30) == 2,
            f"kept {kept(m30)}")
    flat = torch.zeros(1, 10)
    c.check("min-p keeps everything in a flat distribution",
            lambda: kept(submission.min_p_filter(flat, 0.5)) == 10)

    # Repetition penalty: sign handling is where implementations go wrong.
    raw = torch.tensor([[2.0, -2.0, 0.5]])
    previous = torch.tensor([[0, 1]])
    penalised = submission.apply_repetition_penalty(raw, previous, 2.0)
    c.check("a positive logit is divided", lambda: abs(penalised[0, 0] - 1.0) < 1e-6,
            f"got {penalised[0, 0].item()}")
    c.check("a negative logit is multiplied, not divided",
            lambda: abs(penalised[0, 1] - (-4.0)) < 1e-6,
            f"got {penalised[0, 1].item()}")
    c.check("an unseen token is untouched", lambda: abs(penalised[0, 2] - 0.5) < 1e-6)
    c.check("a penalty of 1.0 changes nothing",
            lambda: torch.equal(
                submission.apply_repetition_penalty(raw, previous, 1.0), raw))
    c.check("the input is not modified in place",
            lambda: abs(raw[0, 0] - 2.0) < 1e-9)

    # Greedy.
    batch = torch.randn(4, VOCAB)
    greedy = submission.sample(batch, temperature=0.0)
    c.check("temperature 0 returns the argmax",
            lambda: torch.equal(greedy, batch.argmax(-1, keepdim=True)))
    c.check("sample returns shape (batch, 1)", lambda: tuple(greedy.shape) == (4, 1),
            f"got {tuple(greedy.shape)}")

    # A seed must reproduce exactly.
    def draw(seed):
        gen = torch.Generator().manual_seed(seed)
        return submission.sample(batch, temperature=0.8, top_p=0.9, top_k=50,
                                 generator=gen)

    reproducible = torch.equal(draw(1234), draw(1234))
    c.check("the same seed reproduces the same tokens", lambda: reproducible)
    c.check("a different seed usually differs",
            lambda: not torch.equal(draw(1234), draw(9999)))

    # Truncation must actually bind: with top_k=1 every draw is the argmax.
    gen = torch.Generator().manual_seed(0)
    forced = submission.sample(batch, temperature=1.0, top_k=1, generator=gen)
    c.check("top_k=1 forces the argmax",
            lambda: torch.equal(forced, batch.argmax(-1, keepdim=True)))

    # Sampled tokens must lie inside the nucleus.
    peaked = torch.full((1, VOCAB), -10.0)
    peaked[0, :5] = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    gen = torch.Generator().manual_seed(3)
    draws = {int(submission.sample(peaked, temperature=1.0, top_p=0.9,
                                   generator=gen)) for _ in range(200)}
    c.check("every draw comes from the nucleus", lambda: draws <= {0, 1, 2, 3, 4},
            f"drew {sorted(draws)}")

    c.metric("nucleus_size", len(draws))
    c.metric("seed_reproducible", bool(reproducible))
    return c.finish()
