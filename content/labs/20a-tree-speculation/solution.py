"""Lab 20a - tree speculation (reference solution)."""

import heapq
import math

import torch

# ---------------------------------------------------------------------------
# Given helpers
# ---------------------------------------------------------------------------


def rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Rotate the (n, d) rows of `x` by their integer `positions`. Pairs (2i, 2i+1)."""
    n, d = x.shape
    inv = base ** (-torch.arange(0, d, 2, dtype=torch.float32) / d)
    angle = positions.to(torch.float32)[:, None] * inv[None, :]
    cos, sin = angle.cos(), angle.sin()
    out = torch.empty_like(x)
    out[:, 0::2] = x[:, 0::2] * cos - x[:, 1::2] * sin
    out[:, 1::2] = x[:, 0::2] * sin + x[:, 1::2] * cos
    return out


def _delta_weights():
    g = torch.Generator().manual_seed(5)
    return (torch.randn(8, 8, generator=g) * 8 ** -0.5,
            torch.randn(8, 8, generator=g) * 8 ** -0.5)


_WK, _WV = _delta_weights()


def delta_step(state: torch.Tensor, x: torch.Tensor,
               decay: float = 0.9, beta: float = 0.5) -> torch.Tensor:
    """One gated-delta update. Returns a new (8, 8) state and leaves `state` alone."""
    k = _WK @ x
    k = k / (k.norm() + 1e-6)
    v = _WV @ x
    return decay * (state - beta * torch.outer(k, k @ state)) + beta * torch.outer(k, v)


def residual_distribution(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Chapter 20's residual: normalized positive part of p - q, or p when empty."""
    residual = torch.clamp(p - q, min=0.0)
    total = residual.sum()
    if total <= 0:
        return p / p.sum()
    return residual / total


# ---------------------------------------------------------------------------
# 1. Tree attention
# ---------------------------------------------------------------------------


def tree_depths(parents: list[int]) -> list[int]:
    """Depth of each node. The root, parents[0] == -1, has depth 0."""
    depths: list[int] = []
    for i, parent in enumerate(parents):
        depths.append(0 if parent < 0 else depths[parent] + 1)
    return depths


def ancestors(parents: list[int], node: int) -> list[int]:
    """Nodes on the path from the root to `node`, root first, `node` last."""
    path = [node]
    while parents[path[-1]] >= 0:
        path.append(parents[path[-1]])
    return path[::-1]


def leaf_paths(parents: list[int]) -> list[list[int]]:
    """Every root-to-leaf path, as node indices, ordered by leaf index."""
    has_child = {p for p in parents if p >= 0}
    return [ancestors(parents, i) for i in range(len(parents)) if i not in has_child]


def tree_mask(parents: list[int]) -> torch.Tensor:
    """Boolean (n, n) mask. mask[i, j] is True when node j is node i or an ancestor of i."""
    n = len(parents)
    mask = torch.zeros(n, n, dtype=torch.bool)
    for i in range(n):
        for j in ancestors(parents, i):
            mask[i, j] = True
    return mask


def tree_positions(parents: list[int], depth0: int) -> torch.Tensor:
    """Position id of each node: depth0 plus its depth."""
    return torch.tensor([depth0 + d for d in tree_depths(parents)], dtype=torch.long)


def tree_attention(q, k, v, ctx_k, ctx_v, parents, depth0):
    """One attention head over a committed context plus a draft tree.

    q, k, v: (n, d), one row per tree node, not yet rotated.
    ctx_k, ctx_v: (depth0, d), the committed context. ctx_k is already rotated.
    Every node attends to the whole context and to its own ancestors.
    """
    d = q.shape[1]
    pos = tree_positions(parents, depth0)
    q_r = rope(q, pos)
    k_r = rope(k, pos)
    keys = torch.cat([ctx_k, k_r], dim=0)
    values = torch.cat([ctx_v, v], dim=0)
    scores = q_r @ keys.T / d ** 0.5
    allowed = torch.cat(
        [torch.ones(len(parents), ctx_k.shape[0], dtype=torch.bool), tree_mask(parents)],
        dim=1)
    scores = scores.masked_fill(~allowed, float("-inf"))
    return torch.softmax(scores, dim=-1) @ values


# ---------------------------------------------------------------------------
# 2. Exact multi-candidate verification
# ---------------------------------------------------------------------------


def verify_node(candidates: list[int], q: torch.Tensor, p: torch.Tensor,
                generator: torch.Generator | None = None) -> tuple[int, int]:
    """Recursive rejection sampling over candidates drawn iid from q.

    Returns (index of the accepted candidate, its token), or (-1, token) when
    every candidate is rejected and the token comes from the final residual.
    """
    for index, token in enumerate(candidates):
        q_x = q[token].item()
        ratio = 1.0 if q_x <= 0.0 else min(1.0, p[token].item() / q_x)
        if torch.rand((), generator=generator).item() < ratio:
            return index, int(token)
        p = residual_distribution(p, q)
    return -1, int(torch.multinomial(p, 1, generator=generator).item())


def verify_tree(tokens, parents, draft_probs, target_probs, generator=None):
    """Walk the tree from the root with verify_node at every step.

    tokens[i] is the token at node i (the root's is the committed token).
    draft_probs[i] is the distribution node i's children were drawn from, and
    target_probs[i] the target's distribution after node i.
    Returns (emitted tokens, accepted node indices including the root).
    """
    current = 0
    path = [0]
    emitted: list[int] = []
    while True:
        children = [j for j, parent in enumerate(parents) if parent == current]
        if not children:
            emitted.append(int(torch.multinomial(target_probs[current], 1,
                                                 generator=generator).item()))
            return emitted, path
        index, token = verify_node([tokens[j] for j in children], draft_probs[current],
                                   target_probs[current], generator)
        emitted.append(token)
        if index < 0:
            return emitted, path
        current = children[index]
        path.append(current)


def branch_acceptance(p: torch.Tensor, q: torch.Tensor, m: int) -> list[float]:
    """P(the k-th of m iid candidates is the accepted one), k = 1..m, in closed form."""
    out = []
    survive = 1.0
    for _ in range(m):
        beta = torch.minimum(p, q).sum().item()
        out.append(survive * beta)
        survive *= 1.0 - beta
        p = residual_distribution(p, q)
    return out


# ---------------------------------------------------------------------------
# 3. Expected accepted length and the best tree
# ---------------------------------------------------------------------------


def path_probabilities(parents: list[int], accept: list[list[float]]) -> list[float]:
    """P(node is on the accepted path). accept[d - 1][k] is the chance that the
    k-th child (0-based, in index order) at depth d is the accepted one, given
    its parent was accepted. The root is 1."""
    depths = tree_depths(parents)
    rank = [0] * len(parents)
    seen: dict[int, int] = {}
    for i, parent in enumerate(parents):
        if parent >= 0:
            rank[i] = seen.get(parent, 0)
            seen[parent] = rank[i] + 1
    probs = [1.0] * len(parents)
    for i in range(1, len(parents)):
        probs[i] = probs[parents[i]] * accept[depths[i] - 1][rank[i]]
    return probs


def expected_accepted_length(parents: list[int], accept: list[list[float]]) -> float:
    """Expected accepted draft tokens in one round: the sum of the non-root path probabilities."""
    return sum(path_probabilities(parents, accept)[1:])


def best_tree(accept: list[list[float]], node_budget: int) -> list[int]:
    """The parents array of the tree with `node_budget` draft nodes (plus the root)
    that maximizes expected_accepted_length. Greedy on path probability."""
    parents = [-1]
    values = [1.0]
    slot: list[tuple[int, int]] = [(0, 0)]  # (depth, rank) of each chosen node
    heap: list[tuple[float, int, int, int, int]] = []
    counter = 0

    def push(parent: int, depth: int, rank: int) -> None:
        nonlocal counter
        if depth <= len(accept) and rank < len(accept[depth - 1]):
            value = values[parent] * accept[depth - 1][rank]
            heapq.heappush(heap, (-value, counter, parent, depth, rank))
            counter += 1

    push(0, 1, 0)
    while heap and len(parents) - 1 < node_budget:
        neg, _, parent, depth, rank = heapq.heappop(heap)
        parents.append(parent)
        values.append(-neg)
        slot.append((depth, rank))
        me = len(parents) - 1
        push(me, depth + 1, 0)
        push(parent, depth, rank + 1)
    return parents


# ---------------------------------------------------------------------------
# 4. Speculation under batching
# ---------------------------------------------------------------------------

WEIGHT_BYTES = 53.8e9
ACTIVE_PARAMS = 26.9e9
BANDWIDTH = 1935e9
PEAK_FLOPS = 312e12
STATE_BYTES = 154_927_104


def step_time(batch: int, tokens_per_sequence: int) -> float:
    """Chapter 20's pass time: the longer of the weight read and the arithmetic."""
    memory = WEIGHT_BYTES / BANDWIDTH
    compute = 2 * ACTIVE_PARAMS * batch * tokens_per_sequence / PEAK_FLOPS
    return max(memory, compute)


def expected_tokens_per_round(alpha: float, k: int) -> float:
    if abs(alpha - 1.0) < 1e-12:
        return float(k + 1)
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)


def chain_speedup(batch: int, alpha: float, gamma: int, c: float) -> float:
    """Speedup of a length-gamma chain at a batch size. gamma = 0 is plain decoding."""
    plain = step_time(batch, 1)
    return (expected_tokens_per_round(alpha, gamma) * plain
            / (gamma * c * plain + step_time(batch, gamma + 1)))


def optimal_gamma_continuous(alpha: float, c: float) -> float:
    """Root of alpha^(g+1) (L + c + L c g) = c, chapter 20's batch-1 optimum."""
    L = math.log(1.0 / alpha)

    def f(g: float) -> float:
        return alpha ** (g + 1) * (L + c + L * c * g) - c

    lo, hi = 0.0, 400.0
    if f(lo) <= 0:
        return 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def adaptive_gamma(batch: int, alpha: float, c: float, max_gamma: int = 16) -> int:
    """The draft length in 0..max_gamma with the best speedup at this batch.
    Ties go to the shorter draft. 0 means turn speculation off."""
    best_gamma, best = 0, 1.0
    for gamma in range(1, max_gamma + 1):
        value = chain_speedup(batch, alpha, gamma, c)
        if value > best + 1e-12:
            best_gamma, best = gamma, value
    return best_gamma


def tree_speedup(batch: int, accept: list[list[float]], budget: int, c: float,
                 state_bytes_per_node: int = 0) -> float:
    """Speedup of the best tree with `budget` draft nodes. The draft runs one pass
    per tree level, each costing c plain steps, and the verify pass carries
    budget + 1 tokens per sequence. When `state_bytes_per_node` is not 0, the
    verify pass also writes one recurrent state per node and per sequence,
    batch * (budget + 1) * state_bytes_per_node bytes at the rated bandwidth."""
    parents = best_tree(accept, budget)
    levels = max(tree_depths(parents))
    tokens = 1.0 + expected_accepted_length(parents, accept)
    plain = step_time(batch, 1)
    snapshots = batch * len(parents) * state_bytes_per_node / BANDWIDTH
    return tokens * plain / (levels * c * plain + step_time(batch, len(parents)) + snapshots)


def adaptive_node_budget(batch: int, accept: list[list[float]], c: float,
                         max_budget: int = 32, state_bytes_per_node: int = 0) -> int:
    """The node budget in 0..max_budget with the best tree speedup. 0 turns speculation off."""
    best_budget, best = 0, 1.0
    for budget in range(1, max_budget + 1):
        value = tree_speedup(batch, accept, budget, c, state_bytes_per_node)
        if value > best + 1e-12:
            best_budget, best = budget, value
    return best_budget


# ---------------------------------------------------------------------------
# 5. Recurrent state on a tree
# ---------------------------------------------------------------------------


def tree_states(state0: torch.Tensor, xs: list[torch.Tensor],
                parents: list[int]) -> list[torch.Tensor]:
    """The recurrent state after each node: node i starts from its parent's state
    (state0 for the root). One state per node."""
    states: list[torch.Tensor] = []
    for i, parent in enumerate(parents):
        before = state0 if parent < 0 else states[parent]
        states.append(delta_step(before, xs[i]))
    return states


def commit_by_replay(state0: torch.Tensor, xs: list[torch.Tensor],
                     path: list[int]) -> torch.Tensor:
    """The state after the accepted path, rebuilt from the saved pre-tree state."""
    state = state0
    for node in path:
        state = delta_step(state, xs[node])
    return state


def snapshot_bytes(num_nodes: int, state_bytes: int = STATE_BYTES) -> int:
    """Bytes of per-node recurrent state for a tree of `num_nodes` nodes (root included)."""
    return num_nodes * state_bytes
