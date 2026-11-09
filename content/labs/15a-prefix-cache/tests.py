"""Checks for the radix prefix cache, the hash chain, and state checkpoints.

The first check calls only `block_hash`. Later checks build a PrefixCache,
place recurrent-state checkpoints, and charge a prefill budget. Random
prompt lengths use seed 15.

A plausible wrong implementation hashes a block's own token ids and leaves
out the parent hash. The two blocks of token ids (9, 9, 9, 9) below then
compare equal, and a later request reuses a block that sat under a different
prefix. Evicting an interior node fails the leaf-eviction check: the shared
prefix match drops to 0 cached tokens while a child is still in the tree.
"""

import hashlib
import random
from itertools import combinations

from lab_common import Checks

# Chapter 2: 3,227,648 bytes in one linear layer, times 48 layers.
STATE_BYTES = 154_927_104


def _ran(fn, expected):
    """Run fn and always report expected and actual, including on an exception."""
    try:
        actual = fn()
    except Exception as exc:  # noqa: BLE001
        return False, f"expected {expected}, got {type(exc).__name__}: {exc}"
    if actual == expected:
        return True, f"expected {expected}, got {actual}"
    return False, f"expected {expected}, got {actual}"


def _reference_hash(parent, tokens) -> int:
    head = "none" if parent is None else str(int(parent))
    body = head + "|" + ",".join(str(int(token)) for token in tokens)
    digest = hashlib.sha256(body.encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def _chain(hash_fn, tokens, block_size):
    parent = None
    out = []
    for start in range(0, len(tokens) - block_size + 1, block_size):
        block = tuple(tokens[start:start + block_size])
        parent = hash_fn(parent, block)
        out.append(parent)
    return out


def _saved(positions, lengths) -> int:
    total = 0
    for length in lengths:
        best = 0
        for position in positions:
            if 0 < position <= length and position > best:
                best = position
        total += best
    return total


def _brute(lengths, budget) -> list[int]:
    slots = budget // STATE_BYTES
    candidates = sorted({length for length in lengths if length > 0})
    if slots <= 0 or not candidates:
        return []
    best: list[int] = []
    best_score = -1
    for size in range(0, min(slots, len(candidates)) + 1):
        for combo in combinations(candidates, size):
            score = _saved(combo, lengths)
            chosen = list(combo)
            if score > best_score or (score == best_score and chosen < best):
                best_score = score
                best = chosen
    return best


def _checkpoint_cases():
    """Hand cases first, then six draws from seed 15."""
    rng = random.Random(15)
    cases = [
        ([10] * 9 + [100], STATE_BYTES),
        ([5, 10, 20], STATE_BYTES),
        ([5, 10, 20], 2 * STATE_BYTES),
        ([5, 10, 20], STATE_BYTES - 1),
    ]
    for _ in range(6):
        lengths = [rng.randint(1, 12) for _ in range(8)]
        budget = rng.randint(0, 3) * STATE_BYTES
        cases.append((lengths, budget))
    return cases


# Checked by hand so a bug copied into the brute-force search still fails.
_HAND_PLANS = {
    ((10,) * 9 + (100,), STATE_BYTES): [10],
    ((5, 10, 20), STATE_BYTES): [10],
    ((5, 10, 20), 2 * STATE_BYTES): [5, 20],
    ((5, 10, 20), STATE_BYTES - 1): [],
}


def _expected_plans():
    plans = []
    for lengths, budget in _checkpoint_cases():
        brute = _brute(lengths, budget)
        hand = _HAND_PLANS.get((tuple(lengths), budget))
        if hand is not None and hand != brute:
            raise AssertionError(f"brute {brute} disagrees with hand {hand} for {lengths}")
        plans.append(brute)
    return plans


def run(submission):
    c = Checks()

    def first_block():
        return submission.block_hash(None, (1, 2, 3, 4))

    c.check(
        "block_hash names the first block from its tokens",
        lambda: _ran(first_block, _reference_hash(None, (1, 2, 3, 4))),
    )

    def parent_changes_the_name():
        tokens = (9, 9, 9, 9)
        left = submission.block_hash(1, tokens)
        right = submission.block_hash(2, tokens)
        return (left, right)

    c.check(
        "block_hash mixes in the parent, so the same tokens diverge",
        lambda: _ran(parent_changes_the_name, (
            _reference_hash(1, (9, 9, 9, 9)),
            _reference_hash(2, (9, 9, 9, 9)),
        )),
    )

    def page_aligned_match():
        cache = submission.PrefixCache(4, 64)
        empty = cache.match([1, 2, 3, 4, 5])
        cache.insert([1, 2, 3, 4, 5, 6, 7, 8])
        _ids, cached = cache.match([1, 2, 3, 4, 9, 9, 9, 9])
        one = submission.PrefixCache(1, 16)
        one.insert([4, 5, 6])
        _one_ids, one_cached = one.match([4, 5, 6, 7])
        chain_a = _chain(submission.block_hash, [1, 2, 3, 4, 5, 6, 7, 8], 4)
        chain_b = _chain(submission.block_hash, [1, 2, 3, 4, 9, 9, 9, 9], 4)
        common = 0
        for left, right in zip(chain_a, chain_b):
            if left != right:
                break
            common += 1
        return (empty, cached, one_cached, common * 4)

    c.check(
        "match stops on a block boundary and agrees with the hash chain",
        lambda: _ran(page_aligned_match, (([], 0), 4, 3, 4)),
    )

    def shared_prompt_ratio():
        cache = submission.PrefixCache(16, 10_000)
        shared = list(range(32))
        saved = 0
        total = 0
        for i in range(100):
            prompt = shared + [1_000 + i] * 16
            _ids, n_cached = cache.match(prompt)
            saved += n_cached
            total += len(prompt)
            cache.insert(prompt)
        c.metric("hit_ratio", round(saved / total, 4) if total else 0)
        return saved, total

    c.check(
        "a shared system prompt across 100 requests hits 3168 of 4800 tokens",
        lambda: _ran(shared_prompt_ratio, (3168, 4800)),
    )

    def leaf_eviction():
        cache = submission.PrefixCache(4, 64)
        prefix = [1, 1, 1, 1]
        seq_a = cache.insert(prefix + [2, 2, 2, 2])
        seq_b = cache.insert(prefix + [3, 3, 3, 3])
        held = cache.insert([8, 8, 8, 8])
        cache.release(seq_a)
        cache.release(seq_b)
        removed = cache.evict(1)
        prefix_hit = cache.match(prefix)[1]
        survivor = cache.match(prefix + [3, 3, 3, 3])[1]
        victim = cache.match(prefix + [2, 2, 2, 2])[1]
        held_removed = cache.evict(5)
        held_hit = cache.match([8, 8, 8, 8])[1]
        cache.release(held)
        return (removed, prefix_hit, survivor, victim, held_removed, held_hit)

    c.check(
        "evict removes one unreferenced leaf and keeps the interior and the held block",
        lambda: _ran(leaf_eviction, (1, 4, 8, 4, 2, 4)),
    )

    def copy_on_write():
        cache = submission.PrefixCache(4, 64)
        first = [1, 1, 1, 1, 2, 2]
        second = [1, 1, 1, 1, 2, 2, 3, 3]
        third = [1, 1, 1, 1, 5, 5, 5, 5]
        other = [1, 1, 1, 1, 9, 9, 9, 9]
        id_first = cache.insert(first)
        id_third = cache.insert(third)
        cache.insert(second)
        before = cache.match(other)[1]
        cache.insert(other)
        shared_third = cache.match(third)[0]
        shared_other = cache.match(other)[0]
        same_prefix = shared_third[:1] == shared_other[:1] and len(shared_third) >= 1
        return (
            cache.materialize(id_first),
            cache.materialize(id_third),
            same_prefix,
            cache.match(third)[1],
            before,
        )

    c.check(
        "copy-on-write leaves the earlier sequences' tokens unchanged",
        lambda: _ran(copy_on_write, (
            [1, 1, 1, 1, 2, 2],
            [1, 1, 1, 1, 5, 5, 5, 5],
            True,
            8,
            4,
        )),
    )

    def checkpoints():
        return [
            submission.plan_state_checkpoints(lengths, budget)
            for lengths, budget in _checkpoint_cases()
        ]

    c.check(
        "checkpoint positions match the brute-force optimum",
        lambda: _ran(checkpoints, _expected_plans()),
    )

    def saved_prefill():
        cache = submission.PrefixCache(16, 64)
        shared = [7] * 32
        prompts = [shared + [100 + i] * 16 for i in range(3)]
        report = submission.schedule_prefill(prompts, 80, cache)
        c.metric("prefill_tokens_saved", report["prefill_tokens_saved"])
        return (report["admitted"], report["charged"], report["prefill_tokens_saved"])

    c.check(
        "cached prefix tokens are not charged against the prefill budget",
        lambda: _ran(saved_prefill, (3, 80, 64)),
    )

    return c.finish()
