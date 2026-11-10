import time

import torch
from lab_common import Checks

VOCAB = 512


def kept(logits: torch.Tensor) -> int:
    return int(torch.isfinite(logits).sum().item())


BIG_VOCAB = 248_320


def batched_checks(c, submission):
    """Batched sampling, the sort-free top-p, and the processor hook."""
    # Hook. A processor is (input_ids, logits) -> logits, applied in list order.
    ids = torch.tensor([[1, 2], [3, 4]])
    base = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    seen = {}

    def add_one(input_ids, logits):
        seen["ids"] = input_ids
        return logits + 1.0

    def double(input_ids, logits):
        return logits * 2.0

    out = submission.apply_logits_processors(ids, base, [add_one, double])
    c.check("processors run in list order", lambda: torch.allclose(out, (base + 1) * 2),
            f"expected (x + 1) * 2 = {((base + 1) * 2)[0].tolist()}, "
            f"got {out[0].tolist()}")
    c.check("processors receive input_ids", lambda: torch.equal(seen["ids"], ids))
    c.check("no processors changes nothing",
            lambda: torch.equal(submission.apply_logits_processors(ids, base, None), base))

    def ban(token):
        def processor(input_ids, logits):
            return logits.index_fill(-1, torch.tensor([token]), float("-inf"))
        return processor

    probe = {}

    def spy(input_ids, logits):
        probe["logits"] = logits.clone()
        return logits

    peak = torch.tensor([[0.0, 9.0, 1.0, 2.0]]).repeat(3, 1)
    t3 = torch.tensor([2.0, 2.0, 2.0])
    k3 = torch.zeros(3, dtype=torch.long)
    p3 = torch.ones(3)
    gen = torch.Generator().manual_seed(0)
    got = submission.sample_batched(peak, t3, k3, p3, input_ids=ids[:1].repeat(3, 1),
                                    processors=[ban(1), spy], generator=gen)
    c.check("a ban-token processor removes the top token",
            lambda: bool((got != 1).all()), f"drew {got.flatten().tolist()}, token 1 is banned")
    c.check("processors see logits before temperature",
            lambda: torch.equal(probe["logits"][0, 2:], peak[0, 2:]),
            f"expected raw {peak[0, 2:].tolist()}, got {probe['logits'][0, 2:].tolist()}")

    # Per-row parameters. Plausible wrong implementation caught here: applying
    # row 0's temperature, top-k, or top-p to every row (a wrong broadcast axis).
    torch.manual_seed(1)
    batch = torch.randn(4, VOCAB) * 2
    temps = torch.tensor([0.5, 1.0, 1.5, 2.0])
    ks = torch.tensor([0, 5, 50, 10])
    ps = torch.tensor([1.0, 0.9, 0.5, 0.95])
    got = submission.filter_batched(batch, temps, ks, ps)
    ref_rows = [
        top_p_filter_ref(top_k_ref(batch[i:i + 1] / temps[i], int(ks[i])), float(ps[i]))
        for i in range(4)
    ]
    ref = torch.cat(ref_rows)
    got_kept = [kept(got[i]) for i in range(4)]
    ref_kept = [kept(ref[i]) for i in range(4)]
    c.check("per-row kept counts match looping the single-row filters",
            lambda: got_kept == ref_kept, f"expected kept {ref_kept}, got {got_kept}")
    c.check("per-row values match looping the single-row filters",
            lambda: torch.equal(torch.isfinite(got), torch.isfinite(ref))
            and torch.allclose(got[torch.isfinite(got)], ref[torch.isfinite(ref)]),
            "row i must be scaled by temperature[i]")

    # Greedy rows, forced rows, and draws inside the kept set.
    temps2 = torch.tensor([0.0, 1.0, 1.0, 0.7])
    ks2 = torch.tensor([0, 1, 3, 0])
    ps2 = torch.tensor([1.0, 1.0, 1.0, 0.8])
    gen = torch.Generator().manual_seed(7)
    draws = torch.cat([submission.sample_batched(batch, temps2, ks2, ps2, generator=gen)
                       for _ in range(30)], dim=1)
    argmax = batch.argmax(-1)
    top3 = batch.topk(3, dim=-1).indices[2]
    c.check("a temperature-0 row is greedy inside a mixed batch",
            lambda: bool((draws[0] == argmax[0]).all()))
    c.check("a top_k=1 row always draws its own argmax",
            lambda: bool((draws[1] == argmax[1]).all()),
            f"expected {argmax[1].item()}, drew {sorted(set(draws[1].tolist()))}")
    c.check("a top_k=3 row stays in its own top three",
            lambda: set(draws[2].tolist()) <= set(top3.tolist()),
            f"expected subset of {sorted(top3.tolist())}, drew {sorted(set(draws[2].tolist()))}")
    gen_a = torch.Generator().manual_seed(5)
    gen_b = torch.Generator().manual_seed(5)
    c.check("batched sampling reproduces with a seed",
            lambda: torch.equal(
                submission.sample_batched(batch, temps, ks, ps, generator=gen_a),
                submission.sample_batched(batch, temps, ks, ps, generator=gen_b)))

    # Sort-free top-p. The wrong implementation named here keeps the set strictly
    # below p (drops the token that crosses it): p=0.7 would keep 1, not 2.
    row = torch.tensor([[3.0, 2.0, 1.0, 0.0, -1.0]])
    two = torch.cat([row, row])
    sf = submission.top_p_filter_sortfree(two, torch.tensor([0.7, 0.5]))
    sf_kept = [kept(sf[0]), kept(sf[1])]
    c.check("sort-free top-p keeps the token that crosses p", lambda: sf_kept == [2, 1],
            f"expected kept [2, 1] at p = [0.7, 0.5], got {sf_kept}")
    c.check("sort-free top-p with p=1 changes nothing",
            lambda: torch.equal(submission.top_p_filter_sortfree(row, 1.0), row))

    torch.manual_seed(2)
    big = torch.randn(4, BIG_VOCAB) * 3
    big_p = torch.tensor([0.5, 0.9, 0.95, 0.99])
    start = time.perf_counter()
    got_big = submission.top_p_filter_sortfree(big, big_p)
    elapsed = time.perf_counter() - start
    # The reference runs in float64: a float32 cumsum over 248,320 entries drifts.
    ref_big = torch.cat([top_p_filter_ref(big[i:i + 1].double(), float(big_p[i])) for i in range(4)])
    same = torch.equal(torch.isfinite(got_big), torch.isfinite(ref_big))
    c.check(f"sort-free top-p keeps the same set as the sort on {BIG_VOCAB} entries",
            lambda: same,
            f"expected kept {[kept(r) for r in ref_big]}, got {[kept(r) for r in got_big]}")
    small_sf = submission.filter_batched(batch, temps, ks, ps, sortfree=True)
    c.check("filter_batched with sortfree=True keeps the same sets per row",
            lambda: torch.equal(torch.isfinite(small_sf), torch.isfinite(ref)),
            f"expected kept {ref_kept}, got {[kept(r) for r in small_sf]}")
    c.metric("sortfree_seconds", round(elapsed, 3))


def top_k_ref(logits, k):
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    threshold = logits.topk(k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter_ref(logits, p):
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
    probs = sorted_logits.softmax(dim=-1)
    remove = probs.cumsum(dim=-1) - probs >= p
    remove[..., 0] = False
    return logits.masked_fill(remove.scatter(-1, sorted_idx, remove), float("-inf"))


def run(submission):
    c = Checks()
    try:
        return _run(c, submission)
    except Exception as exc:  # noqa: BLE001
        # An unfinished stub raises before the later checks run. Report it as a
        # failed check instead of crashing the harness.
        c.add("the lab ran to the end", False, f"{type(exc).__name__}: {exc}")
        return c.finish()


def _run(c, submission):
    needed = ("apply_repetition_penalty", "top_k_filter", "top_p_filter",
              "min_p_filter", "sample", "apply_logits_processors",
              "top_p_filter_sortfree", "filter_batched", "sample_batched")
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

    batched_checks(c, submission)

    c.metric("nucleus_size", len(draws))
    c.metric("seed_reproducible", bool(reproducible))
    return c.finish()
