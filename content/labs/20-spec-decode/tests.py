"""Checks lab 20, including a statistical test that the output distribution is
unchanged. That property is the whole reason speculative decoding is safe.
"""

import torch
from lab_common import Checks, close, pick_device

VOCAB = 16
DIM = 8
PREFIX = 6
DRAFT = 5


def _embeddings(count, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(DIM, generator=g) for _ in range(count)]


def _fresh(submission, tokens):
    """A cache that ran only `tokens`, from scratch."""
    cache = submission.ToyHybridCache(dim=DIM, seed=3)
    outputs = [cache.step(x) for x in tokens]
    return cache, outputs


def _compare(cache, reference):
    """Length, KV contents, and recurrent state against a reference cache."""
    if cache.length != reference.length:
        return False, f"KV length {cache.length}, expected {reference.length}"
    if len(cache.keys) != reference.length or len(cache.values) != reference.length:
        return False, (f"KV lists hold {len(cache.keys)} keys and "
                       f"{len(cache.values)} values, expected {reference.length}")
    ok, msg = close(torch.stack(cache.keys), torch.stack(reference.keys), 1e-6)
    if not ok:
        return False, f"keys differ: {msg}"
    ok, msg = close(torch.stack(cache.values), torch.stack(reference.values), 1e-6)
    if not ok:
        return False, f"values differ: {msg}"
    ok, msg = close(cache.state, reference.state, 1e-6)
    if not ok:
        return False, f"recurrent state differs: {msg}"
    return True, f"length {cache.length}, state {msg}"


def _rollback_case(submission, accepted):
    """Draft DRAFT tokens, accept `accepted`, roll back, and compare the cache
    with one that only ever ran the surviving tokens."""
    prefix = _embeddings(PREFIX, 11)
    inputs = _embeddings(DRAFT + 1, 12)
    cache, _ = _fresh(submission, prefix)
    verify_outputs = submission.speculative_round(cache, inputs, accepted)
    reference, _ = _fresh(submission, prefix + inputs[:accepted + 1])
    ok, msg = _compare(cache, reference)
    if not ok:
        return False, f"after accepting {accepted} of {DRAFT}: {msg}"
    # The verify pass must have seen every drafted position.
    _, expected_outputs = _fresh(submission, prefix + inputs)
    expected_outputs = expected_outputs[PREFIX:]
    if len(verify_outputs) != DRAFT + 1:
        return False, f"verify returned {len(verify_outputs)} outputs, expected {DRAFT + 1}"
    ok, msg = close(torch.stack(verify_outputs), torch.stack(expected_outputs), 1e-6)
    if not ok:
        return False, f"verify outputs differ: {msg}"
    # The next token behaves as if the rejected ones never existed.
    nxt = _embeddings(1, 13)[0]
    ok, msg = close(cache.step(nxt), reference.step(nxt), 1e-6)
    return ok, f"after accepting {accepted} of {DRAFT}: next-token output {msg}"


def _check(c, name, fn, detail=""):
    """Like `c.check`, but a detail given as a function is evaluated safely, so
    a stub that raises still produces a failed check instead of a crash."""
    if callable(detail):
        try:
            detail = detail()
        except Exception as exc:  # noqa: BLE001
            detail = f"{type(exc).__name__}: {exc}"
    return c.check(name, fn, detail)


def _safe(fn):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return None


def run(submission):
    c = Checks()
    needed = ("accept_probability", "residual_distribution", "verify",
              "expected_tokens_per_round", "net_speedup",
              "ToyHybridCache", "speculative_round", "step_time",
              "speculative_speedup", "break_even_batch")
    if not c.require(submission, *needed):
        return c.finish()

    device = pick_device()
    torch.manual_seed(0)

    p = torch.tensor([0.5, 0.3, 0.2], device=device)
    q = torch.tensor([0.25, 0.6, 0.15], device=device)

    _check(c, "a token the target likes more is always accepted",
            lambda: abs(submission.accept_probability(p, q, 0) - 1.0) < 1e-9,
            lambda: (f"got {submission.accept_probability(p, q, 0)}"))
    _check(c, "a token the target likes less is accepted with ratio p/q",
            lambda: abs(submission.accept_probability(p, q, 1) - 0.5) < 1e-9,
            lambda: (f"got {submission.accept_probability(p, q, 1)}"))
    _check(c, "the acceptance probability never exceeds 1",
            lambda: submission.accept_probability(p, q, 2) <= 1.0)

    try:
        residual = submission.residual_distribution(p, q)
    except Exception:  # noqa: BLE001
        residual = torch.full((3,), float("nan"), device=device)
    _check(c, "the residual is a distribution",
            lambda: abs(residual.sum().item() - 1.0) < 1e-6,
            f"sums to {residual.sum().item():.6f}")
    _check(c, "the residual is non-negative",
            lambda: (residual >= -1e-9).all().item())
    _check(c, "the residual favours what the draft under-weights",
            lambda: residual[0] > residual[1],
            f"got {residual.tolist()}")
    _check(c, "identical distributions fall back rather than dividing by zero",
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

    try:
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
    except Exception:  # noqa: BLE001
        # A stub that raises still gets a result: every check below fails.
        counts = torch.zeros(VOCAB, device=device)
        counts[0] = 1
        accepted_total = tokens_total = 0

    empirical = counts / counts.sum()
    total_variation = 0.5 * (empirical - target).abs().sum().item()
    acceptance = accepted_total / (rounds * k)
    mean_tokens = tokens_total / rounds

    _check(
        c, "the accepted token follows the target distribution, not the draft's",
        lambda: total_variation < 0.02,
        f"total variation distance {total_variation:.4f}",
    )
    _check(
        c, "the output is closer to the target than to the draft",
        lambda: total_variation < 0.5 * (empirical - draft).abs().sum().item(),
        f"target {total_variation:.4f} vs draft "
        f"{0.5 * (empirical - draft).abs().sum().item():.4f}",
    )
    _check(c, "a round yields between 1 and k + 1 tokens",
            lambda: 1.0 <= mean_tokens <= k + 1,
            f"mean {mean_tokens:.2f}")

    # A perfect draft accepts everything and yields k + 1 every round.
    generator = torch.Generator(device=device).manual_seed(7)
    perfect_rounds = 500
    perfect_total = 0
    try:
        for _ in range(perfect_rounds):
            tokens = torch.multinomial(
                target, k, replacement=True, generator=generator).tolist()
            out, num_accepted = submission.verify(
                tokens, [target] * k, [target] * (k + 1), generator=generator)
            perfect_total += num_accepted
    except Exception:  # noqa: BLE001
        perfect_total = 0
    perfect_rate = perfect_total / (perfect_rounds * k)
    _check(c, "a draft identical to the target is always accepted",
            lambda: perfect_rate > 0.99, f"acceptance {perfect_rate:.3f}")

    _check(c, "expected tokens at alpha=0 is 1",
            lambda: abs(submission.expected_tokens_per_round(0.0, 5) - 1.0) < 1e-9,
            lambda: (f"got {submission.expected_tokens_per_round(0.0, 5)}"))
    _check(c, "expected tokens at alpha=1 is k + 1",
            lambda: abs(submission.expected_tokens_per_round(1.0, 5) - 6.0) < 1e-6,
            lambda: (f"got {submission.expected_tokens_per_round(1.0, 5)}"))
    _check(c, "alpha=0.9 with k=5 yields about 4.69 tokens",
            lambda: abs(submission.expected_tokens_per_round(0.9, 5) - 4.6856) < 1e-3,
            lambda: (f"got {submission.expected_tokens_per_round(0.9, 5):.4f}"))

    _check(c, "a cheap draft with high acceptance beats plain decoding",
            lambda: submission.net_speedup(0.8, 4, 0.05) > 2.0,
            lambda: (f"got {submission.net_speedup(0.8, 4, 0.05):.2f}x"))
    _check(c, "an expensive draft is not worth it",
            lambda: submission.net_speedup(0.8, 4, 0.5) < 1.2,
            lambda: (f"got {submission.net_speedup(0.8, 4, 0.5):.2f}x"))
    _check(c, "longer drafts saturate at low acceptance",
            lambda: submission.expected_tokens_per_round(0.5, 8)
            - submission.expected_tokens_per_round(0.5, 3) < 0.2,
            lambda: (f"k=3 {submission.expected_tokens_per_round(0.5, 3):.3f}, "
            f"k=8 {submission.expected_tokens_per_round(0.5, 8):.3f}"))

    # Rolling back a hybrid cache. The wrong implementation this section
    # catches: truncate the KV list back to the prefix but leave the recurrent
    # state as it was after all k draft tokens. KV length and contents then look
    # right and only the state check fails. A second wrong implementation, a
    # snapshot that aliases the state instead of copying it, restores the
    # already-advanced state and fails the same check.
    for accepted in (0, 2, DRAFT):
        _check(
        c,     f"rollback after accepting {accepted} of {DRAFT} equals never running the rejected tokens",
            lambda accepted=accepted: _rollback_case(submission, accepted),
        )

    def naive_error():
        prefix = _embeddings(PREFIX, 11)
        inputs = _embeddings(DRAFT + 1, 12)
        cache, _ = _fresh(submission, prefix)
        for x in inputs:
            cache.step(x)
        # The naive rollback: shrink the KV list, forget the state.
        del cache.keys[PREFIX + 1:]
        del cache.values[PREFIX + 1:]
        cache.length = PREFIX + 1
        reference, _ = _fresh(submission, prefix + inputs[:1])
        return (cache.state - reference.state).abs().max().item()

    _check(c, "the truncate-only rollback leaves a state error large enough to see",
            lambda: naive_error() > 1e-2,
            lambda: (f"state error of a truncate-only rollback {naive_error():.3e}, tolerance 1e-6"))

    # Cost model. Wrong implementations these checks catch: treating the verify
    # pass as always free (no break-even at all), and charging every verify
    # pass (k + 1) times a plain step (speedup 0.65 at batch 1 instead of 2.80).
    _check(c, "step time is the weight read below the ridge",
            lambda: abs(submission.step_time(1, 1) * 1e3 - 27.8036) < 1e-3
            and abs(submission.step_time(32, 5) - submission.step_time(1, 1)) < 1e-9,
            lambda: (f"batch 1 {submission.step_time(1, 1) * 1e3:.4f} ms (expected 27.8036), "
            f"batch 32 x 5 tokens {submission.step_time(32, 5) * 1e3:.4f} ms (expected 27.8036)"))
    _check(c, "above the ridge a step costs its FLOPs",
            lambda: abs(submission.step_time(64, 5) * 1e3 - 55.1795) < 1e-3,
            lambda: (f"batch 64 x 5 tokens {submission.step_time(64, 5) * 1e3:.4f} ms, expected 55.1795"))
    _check(c, "at batch 1 the speedup is chapter 20's net speedup",
            lambda: abs(submission.speculative_speedup(1, 0.8, 4, 0.05)
                        - submission.net_speedup(0.8, 4, 0.05)) < 1e-9,
            lambda: (f"got {submission.speculative_speedup(1, 0.8, 4, 0.05):.4f}, "
            f"expected {submission.net_speedup(0.8, 4, 0.05):.4f}"))
    _check(c, "verification is still free at batch 32 with k = 4 (160 tokens, ridge 161)",
            lambda: abs(submission.speculative_speedup(32, 0.8, 4, 0.05) - 2.8013) < 1e-3,
            lambda: (f"got {submission.speculative_speedup(32, 0.8, 4, 0.05):.4f}, expected 2.8013"))
    _check(c, "at batch 128 speculation loses (0.81x)",
            lambda: abs(submission.speculative_speedup(128, 0.8, 4, 0.05) - 0.8063) < 1e-3,
            lambda: (f"got {submission.speculative_speedup(128, 0.8, 4, 0.05):.4f}, expected 0.8063"))
    _check(c, "speedup never rises with batch size",
            lambda: all(submission.speculative_speedup(b + 1, 0.8, 4, 0.05)
                        <= submission.speculative_speedup(b, 0.8, 4, 0.05) + 1e-12
                        for b in range(1, 400)))
    _check(c, "break-even batch for alpha 0.8, k 4, c 0.05 is 101",
            lambda: submission.break_even_batch(0.8, 4, 0.05) == 101,
            lambda: (f"got {submission.break_even_batch(0.8, 4, 0.05)}, expected 101"))
    _check(c, "a better draft keeps speculation profitable to a larger batch (alpha 0.9 gives 125)",
            lambda: submission.break_even_batch(0.9, 4, 0.05) == 125,
            lambda: (f"got {submission.break_even_batch(0.9, 4, 0.05)}, expected 125"))
    _check(c, "a longer draft stops paying sooner (k 8 gives 70)",
            lambda: submission.break_even_batch(0.8, 8, 0.05) == 70,
            lambda: (f"got {submission.break_even_batch(0.8, 8, 0.05)}, expected 70"))
    _check(c, "snapshotting the recurrent state pulls the break-even down (85)",
            lambda: submission.break_even_batch(0.8, 4, 0.05, submission.STATE_BYTES) == 85
            if hasattr(submission, "STATE_BYTES") else False,
            "alpha 0.8, k 4, c 0.05 with 154,927,104 bytes of state per sequence")
    _check(c, "a draft that never agrees loses at batch 1",
            lambda: submission.break_even_batch(0.0, 4, 0.5) == 0,
            lambda: (f"got {submission.break_even_batch(0.0, 4, 0.5)}, expected 0"))


    c.metric("acceptance_rate", round(acceptance, 4))
    c.metric("distribution_error", round(total_variation, 5))
    c.metric("expected_tokens", round(mean_tokens, 3))
    c.metric("break_even_batch", _safe(lambda: submission.break_even_batch(0.8, 4, 0.05)))
    return c.finish()
