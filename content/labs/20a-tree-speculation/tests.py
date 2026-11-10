"""Checks lab 20a: tree attention, exact tree verification, the best tree under a
budget, adaptive speculation under batching, and per-node recurrent state.

Wrong implementations these checks catch, each run as a mutant while the tests
were written:

- A lower-triangular mask in place of the tree mask (check 3 fails: siblings see
  each other).
- Position ids equal to the node's index instead of its depth (check 3 fails: the
  rotary phases differ from the path-by-path run).
- Testing every sibling against the original target distribution, with no
  residual update between siblings (checks 4 and 5 fail, the second on total variation).
- Ranking candidate nodes by their edge acceptance alone instead of their path
  probability, and a chain-only tree (check 8 fails against exhaustive search).
- One shared recurrent state advanced over the nodes in index order instead of
  a state per node (check 10 fails).
"""

import math

import torch
from lab_common import Checks, close

PARENTS = [-1, 0, 0, 1, 1, 2, 3]
DEPTH0 = 7


def _check(c, name, fn, detail=""):
    """Like `c.check`, but a detail given as a function is evaluated safely."""
    if callable(detail):
        try:
            detail = detail()
        except Exception as exc:  # noqa: BLE001
            detail = f"{type(exc).__name__}: {exc}"
    return c.check(name, fn, detail)


# -- check helpers -----------------------------------------------------------


def _mask_case(submission):
    mask = submission.tree_mask(PARENTS)
    expected = torch.zeros(7, 7, dtype=torch.bool)
    for row, cols in enumerate([[0], [0, 1], [0, 2], [0, 1, 3], [0, 1, 4],
                                [0, 2, 5], [0, 1, 3, 6]]):
        expected[row, cols] = True
    if mask.shape != expected.shape:
        return False, f"shape {tuple(mask.shape)}, expected {tuple(expected.shape)}"
    if not torch.equal(mask.bool(), expected):
        bad = (mask.bool() != expected).nonzero().tolist()
        return False, f"mask differs from the hand-built tree at (row, col) {bad[:4]}"
    positions = submission.tree_positions(PARENTS, DEPTH0).tolist()
    want = [7, 8, 8, 9, 9, 9, 10]
    return positions == want, f"positions {positions}, expected {want}"


def _attention_case(submission, parents, seed):
    n = len(parents)
    d = 8
    g = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(n, d, generator=g) for _ in range(3))
    ctx_k_raw = torch.randn(DEPTH0, d, generator=g)
    ctx_v = torch.randn(DEPTH0, d, generator=g)
    ctx_k = submission.rope(ctx_k_raw, torch.arange(DEPTH0))
    out = submission.tree_attention(q, k, v, ctx_k, ctx_v, parents, DEPTH0)
    if out.shape != (n, d):
        return False, f"output shape {tuple(out.shape)}, expected {(n, d)}"
    worst = 0.0
    for path in submission.leaf_paths(parents):
        m = len(path)
        pos = torch.arange(DEPTH0, DEPTH0 + m)
        q_p = submission.rope(q[path], pos)
        k_p = submission.rope(k[path], pos)
        keys = torch.cat([ctx_k, k_p])
        values = torch.cat([ctx_v, v[path]])
        scores = q_p @ keys.T / d ** 0.5
        causal = torch.ones(m, DEPTH0 + m, dtype=torch.bool)
        causal[:, DEPTH0:] = torch.tril(torch.ones(m, m, dtype=torch.bool))
        ref = torch.softmax(scores.masked_fill(~causal, float("-inf")), -1) @ values
        worst = max(worst, (out[path] - ref).abs().max().item())
    return worst < 1e-5, f"largest error against the path-by-path run {worst:.2e} (tolerance 1e-5)"


V = 5


def _markov(seed):
    g = torch.Generator().manual_seed(seed)
    target = torch.rand(V, V, generator=g) + 0.05
    draft = torch.rand(V, V, generator=g) + 0.05
    return target / target.sum(1, keepdim=True), draft / draft.sum(1, keepdim=True)


def _stream(submission, target, draft, generator, width=3, fanout=2):
    """Generate two tokens by repeated tree rounds. The root of each round is the
    last committed token, and children are drawn iid from the draft."""
    stream = [0]
    while len(stream) < 3:
        tokens, parents = [stream[-1]], [-1]
        frontier = [0]
        for count in (width, fanout):
            nxt = []
            for node in frontier:
                drawn = torch.multinomial(draft[tokens[node]], count,
                                          replacement=True, generator=generator).tolist()
                for t in drawn:
                    tokens.append(t)
                    parents.append(node)
                    nxt.append(len(tokens) - 1)
            frontier = nxt
        emitted, _ = submission.verify_tree(
            tokens, parents, [draft[t] for t in tokens], [target[t] for t in tokens],
            generator)
        stream.extend(emitted)
    return stream[1], stream[2]


def _joint_tv(submission, rounds=30000, seed=21):
    target, draft = _markov(3)
    generator = torch.Generator().manual_seed(seed)
    counts = torch.zeros(V, V)
    for _ in range(rounds):
        a, b = _stream(submission, target, draft, generator)
        counts[a, b] += 1
    empirical = counts / counts.sum()
    exact = target[0][:, None] * target
    wrong = draft[0][:, None] * draft
    return (0.5 * (empirical - exact).abs().sum().item(),
            0.5 * (empirical - wrong).abs().sum().item())


def _noisy_draft_table():
    """Per-depth acceptance of 4 iid candidates, from a Zipf target and a draft
    whose logits carry more noise at every depth. Closed form, no sampling."""
    g = torch.Generator().manual_seed(20)
    logp = -0.9 * torch.log(torch.arange(1, 33).float())
    p = torch.softmax(logp, 0)
    table = []
    for sigma in (0.8, 1.0, 1.2, 1.4, 1.6, 1.8):
        q = torch.softmax(logp + torch.randn(32, generator=g) * sigma, 0)
        row, left, residual = [], 1.0, p
        for _ in range(4):
            beta = torch.minimum(residual, q).sum().item()
            row.append(left * beta)
            left *= 1.0 - beta
            r = torch.clamp(residual - q, min=0.0)
            residual = r / r.sum()
        table.append(row)
    return table


def _exhaustive(accept, budget):
    """Best expected accepted length over every ordered tree with `budget` draft nodes."""

    def shapes(n):  # ordered forests with n nodes in total, as lists of child shapes
        if n == 0:
            yield []
            return
        for first in range(1, n + 1):
            for head in shapes(first - 1):
                for rest in shapes(n - first):
                    yield [head] + rest

    def value(children, depth, parent_value):
        total = 0.0
        for rank, child in enumerate(children):
            if depth > len(accept) or rank >= len(accept[depth - 1]):
                return float("-inf")
            v = parent_value * accept[depth - 1][rank]
            total += v + value(child, depth + 1, v)
        return total

    return max(value(forest, 1, 1.0) for forest in shapes(budget))


TABLES = [
    [[0.6, 0.3, 0.1], [0.5, 0.2, 0.05], [0.4, 0.1, 0.02]],
    [[0.5, 0.45, 0.04], [0.2, 0.1, 0.05], [0.2, 0.1, 0.05]],
    [[0.7, 0.2, 0.05, 0.02], [0.7, 0.15, 0.05, 0.01], [0.6, 0.1, 0.02, 0.01],
     [0.5, 0.1, 0.02, 0.01]],
]


def _best_tree_case(submission):
    worst = 0.0
    for t, accept in enumerate(TABLES):
        for budget in range(1, 9):
            parents = submission.best_tree(accept, budget)
            if len(parents) != budget + 1 or parents[0] != -1 or any(
                    not 0 <= parents[i] < i for i in range(1, len(parents))):
                return False, f"table {t}, budget {budget}: not a valid tree, parents {parents}"
            got = submission.expected_accepted_length(parents, accept)
            want = _exhaustive(accept, budget)
            worst = max(worst, abs(got - want))
            if abs(got - want) > 1e-9:
                return False, (f"table {t}, budget {budget}: expected length {got:.6f}, "
                               f"exhaustive search {want:.6f}")
    return True, f"matches exhaustive search on 3 tables x budgets 1..8 (worst gap {worst:.1e})"


def _states_case(submission):
    g = torch.Generator().manual_seed(8)
    xs = [torch.randn(8, generator=g) for _ in range(len(PARENTS))]
    state0 = torch.randn(8, 8, generator=g) * 0.3
    states = submission.tree_states(state0, xs, PARENTS)
    if len(states) != len(PARENTS):
        return False, f"{len(states)} states, expected {len(PARENTS)}"
    worst = 0.0
    for node in range(len(PARENTS)):
        path = submission.ancestors(PARENTS, node)
        worst = max(worst, (states[node] - submission.commit_by_replay(state0, xs, path)
                            ).abs().max().item())
    shared = state0
    for x in xs:
        shared = submission.delta_step(shared, x)
    gap = (shared - states[2]).abs().max().item()
    cost = submission.snapshot_bytes(16)
    return worst < 1e-6 and gap > 1e-2 and cost == 16 * 154_927_104, (
        f"snapshot bytes for 16 nodes {cost} (expected {16 * 154_927_104}); "
        f"per-node state vs replaying its path, error {worst:.2e} (tolerance 1e-6); "
        f"one shared state in index order is off by {gap:.2e} at node 2 (should exceed 1e-2)")


def run(submission):
    c = Checks()
    needed = ("tree_mask", "tree_positions", "tree_attention", "verify_node",
              "verify_tree", "branch_acceptance", "path_probabilities",
              "expected_accepted_length", "best_tree", "chain_speedup",
              "adaptive_gamma", "tree_speedup", "adaptive_node_budget",
              "tree_states", "commit_by_replay", "snapshot_bytes")
    if not c.require(submission, *needed):
        return c.finish()
    torch.manual_seed(0)

    # 1. The smallest piece: the mask and positions of a hand-built tree.
    _check(c, "1. mask and position ids of a hand-built tree", lambda: _mask_case(submission))

    # 2. Leaf paths feed the reference run.
    _check(c, "2. root-to-leaf paths",
           lambda: submission.leaf_paths(PARENTS) == [[0, 1, 4], [0, 2, 5], [0, 1, 3, 6]],
           lambda: f"got {submission.leaf_paths(PARENTS)}")

    # 3. The tree pass equals running each path alone.
    _check(c, "3. masked tree attention equals running every root-to-leaf path alone",
           lambda: (lambda a, b: (a[0] and b[0], a[1] + " | " + b[1]))(
               _attention_case(submission, PARENTS, 1),
               _attention_case(submission, [-1, 0, 0, 0, 1, 1, 2, 3, 4, 4], 2)))

    # 4. Closed-form branch acceptance, then its agreement with verify_node.
    def branch_case():
        p = torch.tensor([0.5, 0.3, 0.2])
        q = torch.tensor([0.2, 0.3, 0.5])
        a = submission.branch_acceptance(p, q, 3)
        # beta1 = 0.2 + 0.3 + 0.2 = 0.7; residual (0.3, 0, 0) -> (1, 0, 0);
        # beta2 = 0.2, then the residual is empty and falls back to (1, 0, 0).
        want = [0.7, 0.3 * 0.2, 0.3 * 0.8 * 0.2]
        if max(abs(x - y) for x, y in zip(a, want)) > 1e-6:
            return False, f"branch_acceptance {[round(x, 4) for x in a]}, expected {want}"
        g = torch.Generator().manual_seed(4)
        hits = [0, 0, 0, 0]
        n = 20000
        for _ in range(n):
            cands = torch.multinomial(q, 3, replacement=True, generator=g).tolist()
            index, _ = submission.verify_node(cands, q, p, g)
            hits[index] += 1
        freq = [h / n for h in hits[:3]]
        err = max(abs(x - y) for x, y in zip(freq, a))
        return err < 0.015, (f"frequency of the accepted index {[round(x, 3) for x in freq]} "
                             f"vs closed form {[round(x, 3) for x in a]}, gap {err:.4f} (tolerance 0.015)")
    _check(c, "4. branch acceptance matches its closed form and verify_node", branch_case)

    # 5. The statistical test: the stream a tree speculator emits follows the target.
    tv, tv_draft = (1.0, 1.0)
    try:
        tv, tv_draft = _joint_tv(submission)
    except Exception as exc:  # noqa: BLE001
        c.add("5. the first two emitted tokens follow the target's joint distribution",
              False, f"{type(exc).__name__}: {exc}")
    else:
        _check(c, "5. the first two emitted tokens follow the target's joint distribution",
               lambda: tv < 0.025 and tv < 0.5 * tv_draft,
               f"total variation distance {tv:.4f} (tolerance 0.025), draft's joint {tv_draft:.4f}")

    # 6. Structure of one verified round.
    def structure_case():
        target, draft = _markov(3)
        g = torch.Generator().manual_seed(6)
        parents = [-1, 0, 0, 1, 1, 2, 2]
        for _ in range(300):
            tokens = [0] + torch.randint(0, V, (6,), generator=g).tolist()
            emitted, path = submission.verify_tree(
                tokens, parents, [draft[t] for t in tokens], [target[t] for t in tokens], g)
            ok = (path[0] == 0 and all(parents[b] == a for a, b in zip(path, path[1:]))
                  and len(emitted) == len(path)
                  and emitted[:-1] == [tokens[i] for i in path[1:]])
            if not ok:
                return False, f"emitted {emitted} for accepted path {path} in tree {tokens}"
        return True, "every round emits its accepted path plus one token"
    _check(c, "6. a round emits the accepted path plus one token", structure_case)

    # 7. Expected accepted length by hand.
    def expected_case():
        accept = [[0.6, 0.3], [0.5, 0.2]]
        parents = [-1, 0, 0, 1, 1]
        probs = submission.path_probabilities(parents, accept)
        want = [1.0, 0.6, 0.3, 0.3, 0.12]
        total = submission.expected_accepted_length(parents, accept)
        ok = max(abs(x - y) for x, y in zip(probs, want)) < 1e-9 and abs(total - 1.32) < 1e-9
        return ok, f"path probabilities {[round(x, 4) for x in probs]}, expected {want}; sum {total:.4f}, expected 1.32"
    _check(c, "7. expected accepted length from per-depth acceptance", expected_case)

    # 8. Best tree under a node budget, against every tree.
    _check(c, "8. best_tree matches exhaustive search for budgets 1 to 8",
           lambda: _best_tree_case(submission))

    # 9. Adaptive speculation length, against chapter 20.
    def adaptive_case():
        def optimum(alpha):
            return max(range(0, 17), key=lambda g: (
                submission.chain_speedup(1, alpha, g, 0.05), -g))
        for alpha, want in ((0.5, 3), (0.7, 6), (0.8, 8), (0.9, 13)):
            got = submission.adaptive_gamma(1, alpha, 0.05)
            root = _root(alpha, 0.05)
            if got != want or got != optimum(alpha) or got not in (math.floor(root), math.ceil(root)):
                return False, (f"alpha {alpha} at batch 1: adaptive_gamma {got}, expected {want} "
                               f"(continuous optimum {root:.3f})")
        row = [submission.adaptive_gamma(b, 0.8, 0.05) for b in (1, 8, 16, 32, 64, 96, 128, 160)]
        want_row = [8, 8, 8, 4, 2, 1, 1, 0]
        if row != want_row:
            return False, f"alpha 0.8 at batches 1, 8, 16, 32, 64, 96, 128, 160: {row}, expected {want_row}"
        if any(b > a for a, b in zip(row, row[1:])):
            return False, f"draft length rose with batch: {row}"
        table = _noisy_draft_table()
        batches = (1, 4, 8, 16, 32, 64, 128)
        free = [submission.adaptive_node_budget(b, table, 0.05) for b in batches]
        held = [submission.adaptive_node_budget(b, table, 0.05, 32, 154_927_104) for b in batches]
        if free != [32, 32, 19, 9, 4, 2, 1] or held != [32, 10, 3, 3, 3, 1, 0]:
            return False, (f"node budgets at batches {batches}: {free} without state writes "
                           f"(expected [32, 32, 19, 9, 4, 2, 1]), {held} with one 154,927,104-byte "
                           f"state per node (expected [32, 10, 3, 3, 3, 1, 0])")
        return True, (f"alpha 0.8 gives gamma {row} across batches 1, 8, 16, 32, 64, 96, 128, 160; "
                      f"node budgets {free}, with per-node state writes {held}")
    _check(c, "9. adaptive draft length and node budget match chapter 20 at batch 1 and shrink with load",
           adaptive_case)

    # 10. Per-node recurrent state on a tree.
    _check(c, "10. a recurrent state per node equals replaying that node's path",
           lambda: _states_case(submission))

    accept = [[0.795, 0.048, 0.028], [0.658, 0.08, 0.032], [0.628, 0.088, 0.056]]
    c.metric("joint_total_variation", round(tv, 5))
    c.metric("best_tree_expected_length_8", _safe(lambda: round(
        submission.expected_accepted_length(submission.best_tree(accept, 8), accept), 4)))
    c.metric("adaptive_gamma_batch_32", _safe(lambda: submission.adaptive_gamma(32, 0.8, 0.05)))
    return c.finish()


def _safe(fn):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return None


def _root(alpha, c):
    L = math.log(1 / alpha)
    lo, hi = 0.0, 400.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if alpha ** (mid + 1) * (L + c + L * c * mid) - c > 0:
            lo = mid
        else:
            hi = mid
    return lo
