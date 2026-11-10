"""Checks for lab 11a, constrained decoding.

Every random draw is seeded. Plausible wrong implementations that these checks
are written to catch:

* `build_token_mask` that tests only a token's first byte (naive character
  masking). It admits `ab` after `a` in a pattern where only `ac` can follow.
  Caught by "the mask tests every byte of a token".
* `sample_constrained` that applies top-k before the mask. When the k highest
  logits are all illegal, nothing survives. Caught by "mask first, then
  top-k".
* `jump_forward` that keeps going through an accepting state. The grammar can
  stop there, so the next byte is not forced. Caught by "jump-forward stops
  at an accepting state".
* `ConstraintProcessor` that masks every row with row 0's state. Caught by
  "each row uses its own state".
"""

import json
import re
import time
import random

import torch
from lab_common import Checks


FORCED = b'{"name": "'


def _ran(fn, expected):
    """Run fn and report expected and actual, including when fn raises."""
    try:
        actual = fn()
    except Exception as exc:  # noqa: BLE001
        return False, f"expected {expected}, got {type(exc).__name__}: {exc}"
    return actual == expected, f"expected {expected}, got {actual}"


def _legal(sub, pattern, text_before, vocab=None):
    """The set of token byte strings legal after `text_before`."""
    vocab = vocab or sub.toy_vocab()
    dfa = sub.compile_regex(pattern)
    state = sub.walk(dfa, dfa.start, text_before)
    mask = sub.build_token_mask(dfa, state, vocab, sub.EOS_ID)
    return {vocab[i] for i in range(len(vocab)) if bool(mask[i])}, mask


class SeededModel:
    """A fake model: logits depend only on the seed and the prefix length."""

    def __init__(self, seed, vocab_size, eos_id, eos_boost=0.0):
        self.seed, self.vocab_size = seed, vocab_size
        self.eos_id, self.eos_boost = eos_id, eos_boost

    def __call__(self, prefix):
        gen = torch.Generator().manual_seed(self.seed * 1000 + len(prefix))
        logits = torch.randn(self.vocab_size, generator=gen) * 2.0
        logits[self.eos_id] += self.eos_boost
        return logits


def _decode(sub, ids, vocab):
    return b"".join(vocab[i] for i in ids if i != sub.EOS_ID)


def run(sub):
    c = Checks()
    if not c.require(sub, "walk", "build_token_mask", "sample_constrained", "TokenFSM",
                     "generate", "greedy_encode", "jump_forward", "nested_array_regex",
                     "ConstraintProcessor", "EmptyMaskError"):
        return c.finish()
    vocab = sub.toy_vocab()
    V = len(vocab)

    # 1. The smallest piece: walk.
    dfa_ab = sub.compile_regex("ab+c")

    def check_walk():
        s1 = sub.walk(dfa_ab, dfa_ab.start, b"abb")
        ok = [
            sub.walk(dfa_ab, dfa_ab.start, b"") == dfa_ab.start,
            s1 != sub.DEAD and not dfa_ab.accepting(s1),
            dfa_ab.accepting(sub.walk(dfa_ab, s1, b"bc")),
            sub.walk(dfa_ab, dfa_ab.start, b"ax") == sub.DEAD,
            sub.walk(dfa_ab, sub.DEAD, b"a") == sub.DEAD,
        ]
        return all(ok), f"expected [True]*5, got {ok}"

    c.check("walk follows bytes and returns DEAD on a missing edge", check_walk)

    # 2. The mask: multi-character tokens, byte fallback, end and empty tokens.
    def check_mask():
        # After "a" in "ac|abd", the bytes b and c are legal. The two-byte
        # token "bc" is not: b is fine but c cannot follow "ab".
        legal, mask = _legal(sub, "ac|abd", b"a")
        expected_extra = {b"b", b"c"}
        # The vocabulary also holds "bc", which is illegal after "a" here.
        problems = []
        if legal != expected_extra:
            problems.append(f"after 'a' expected {sorted(expected_extra)}, got {sorted(legal)}")
        # Multi-character token legal across a boundary: after '{"name": "'.
        pat = r'\{"name": "[a-z]{1,8}", "age": (0|[1-9][0-9]{0,2})\}'
        legal, _ = _legal(sub, pat, b"")
        if legal != {b"{", b'{"'}:
            problems.append(f"at start expected the tokens for {{ and {{\", got {sorted(legal)}")
        # Byte fallback: 'caf' then either 'e' or the two bytes of 'e acute'.
        legal, _ = _legal(sub, "caf(é|e)", b"caf")
        if legal != {b"e", b"\xc3"}:
            problems.append(f"after caf expected e and \\xc3, got {sorted(legal)}")
        legal, _ = _legal(sub, "caf(é|e)", b"caf\xc3")
        if legal != {b"\xa9"}:
            problems.append(f"after caf\\xc3 expected \\xa9, got {sorted(legal)}")
        # End token only at an accepting state. Empty padding token never.
        _, mask = _legal(sub, "yes|no", b"")
        _, mask_done = _legal(sub, "yes|no", b"no")
        if bool(mask[sub.EOS_ID]) or not bool(mask_done[sub.EOS_ID]):
            problems.append("end token must be legal only at an accepting state")
        if bool(mask[sub.PAD_ID]) or bool(mask_done[sub.PAD_ID]):
            problems.append("an empty non-end token must never be legal")
        if mask.dtype != torch.bool or tuple(mask.shape) != (V,):
            problems.append(f"expected a bool mask of shape ({V},), got {mask.dtype} {tuple(mask.shape)}")
        return (not problems), "; ".join(problems) or "expected exact legal sets, got them"

    c.check("the mask tests every byte of a token", check_mask)

    # 3. Sampling order and the empty mask.
    def check_sampling():
        problems = []
        n = 12
        logits = torch.zeros(n)
        logits[0], logits[1] = 10.0, 9.0          # the two highest logits
        mask = torch.ones(n, dtype=torch.bool)
        mask[0] = mask[1] = False                 # ...are both illegal
        gen = torch.Generator().manual_seed(11)
        drawn = [sub.sample_constrained(logits, mask, 1.0, 2, gen) for _ in range(200)]
        if any(d in (0, 1) for d in drawn):
            problems.append("a masked token was drawn")
        gen = torch.Generator().manual_seed(11)
        try:
            for _ in range(5):
                sub.sample_constrained(logits, mask, 1.0, 2, gen)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"top-k with an illegal top-2 raised {type(exc).__name__}; expected a legal draw")
        g = sub.sample_constrained(logits, mask, 0.0)
        if g != 2:
            problems.append(f"greedy under the mask expected 2, got {g}")
        try:
            sub.sample_constrained(logits, torch.zeros(n, dtype=torch.bool))
            problems.append("expected EmptyMaskError for an all-False mask, got a token")
        except sub.EmptyMaskError:
            pass
        return (not problems), "; ".join(problems) or "expected legal draws and EmptyMaskError, got them"

    c.check("mask first, then top-k; an empty mask raises", check_sampling)

    # 4. TokenFSM.advance.
    def check_advance():
        fsm = sub.TokenFSM(sub.compile_regex("yes|no"), vocab, sub.EOS_ID)
        id_ye, id_s = vocab.index(b"ye"), vocab.index(b"s")
        s = fsm.advance(fsm.start, id_ye)
        s = fsm.advance(s, id_s)
        problems = []
        if not fsm.is_accepting(s):
            problems.append("after ye + s expected an accepting state")
        if fsm.advance(s, sub.EOS_ID) != s:
            problems.append("the end token must leave the state unchanged")
        for bad_state, bad_token, why in [
            (fsm.start, vocab.index(b"maybe"), "maybe"),
            (fsm.start, sub.EOS_ID, "end token at a non-accepting state"),
            (fsm.start, sub.PAD_ID, "empty padding token"),
        ]:
            try:
                fsm.advance(bad_state, bad_token)
                problems.append(f"expected ValueError for {why}, got a state")
            except ValueError:
                pass
        return (not problems), "; ".join(problems) or "expected ValueError on bad tokens, got it"

    c.check("advance accepts legal tokens and rejects illegal ones", check_advance)

    # 5. The cache.
    def check_cache():
        rng = random.Random(5)
        alphabet = b'abc01{}":, '
        big = [b""] + [bytes([i]) for i in range(256)]
        big += [bytes(rng.choice(alphabet) for _ in range(rng.randint(2, 6)))
                for _ in range(30000)]
        fsm = sub.TokenFSM(sub.compile_regex(r'\{"a": [0-9]{1,3}\}'), big, 0)
        t0 = time.perf_counter()
        first = fsm.mask(fsm.start)
        build_s = time.perf_counter() - t0
        reps = 200
        t0 = time.perf_counter()
        for _ in range(reps):
            again = fsm.mask(fsm.start)
        cached_s = (time.perf_counter() - t0) / reps
        c.metric("mask_build_ms", round(build_s * 1000, 2))
        c.metric("cached_mask_us", round(cached_s * 1e6, 2))
        problems = []
        if fsm.builds != 1:
            problems.append(f"expected 1 build for {reps + 1} calls, got {fsm.builds}")
        if not torch.equal(first, again):
            problems.append("the cached mask differs from the first one")
        if not cached_s * 5 < build_s:
            problems.append(f"expected cached call at least 5x faster than build, got "
                            f"{cached_s * 1e6:.1f} us vs {build_s * 1e6:.0f} us")
        ids = [i for i in range(len(big)) if bool(first[i])]
        return (not problems), "; ".join(problems) or (
            f"expected one build and a faster cached call, got 1 build, "
            f"{build_s * 1000:.1f} ms then {cached_s * 1e6:.1f} us ({len(ids)} legal tokens)")

    c.check("the cached mask is built once and is faster", check_cache)

    # 6. The property check: every sampled completion matches its pattern.
    patterns = [
        (r'-?[0-9]+(\.[0-9]+)?', None),
        (r'(yes|no|maybe)', None),
        (r'\{"name": "[a-z]{1,8}", "age": (0|[1-9][0-9]{0,2})\}', "json"),
        (r'\[(0|[1-9][0-9]?)(,(0|[1-9][0-9]?)){0,3}\]', "json"),
    ]

    def check_property():
        total = good = finished = mismatched = 0
        bad = None
        for p_idx, (pattern, kind) in enumerate(patterns):
            fsm = sub.TokenFSM(sub.compile_regex(pattern), vocab, sub.EOS_ID)
            for seed in range(250):
                model = SeededModel(seed + 1000 * p_idx, V, sub.EOS_ID, eos_boost=4.0)
                ids = sub.generate(fsm, model, seed, max_tokens=64, temperature=1.0)
                total += 1
                if ids and ids[-1] == sub.EOS_ID:
                    finished += 1
                text = _decode(sub, ids, vocab).decode("utf-8", errors="replace")
                ok = re.fullmatch(pattern, text, re.ASCII) is not None
                if ok and kind == "json":
                    try:
                        json.loads(text)
                    except ValueError:
                        ok = False
                good += ok
                if ok and sub.greedy_encode(text.encode(), vocab, sub.EOS_ID) != \
                        [i for i in ids if i != sub.EOS_ID]:
                    mismatched += 1
                if not ok and bad is None:
                    bad = (pattern, text)
        c.metric("completions_matching", f"{good}/{total}")
        c.metric("retokenize_mismatch_rate", round(mismatched / total, 3))
        if good != total or finished != total:
            return False, (f"expected {total}/{total} finished completions that fully match, "
                           f"got {good} matching, {finished} finished; first bad: {bad}")
        return True, f"expected {total}/{total} to match, got {good}/{total} (all finished)"

    c.check("1000 seeded completions all match their pattern and json.loads", check_property)

    # 7. greedy_encode.
    def check_encode():
        v = [b"", b"a", b"b", b"ab", b"abc", b"c", b"ab"]
        problems = []
        got = sub.greedy_encode(b"abcab", v, 0)
        if got != [4, 3]:
            problems.append(f"abcab expected [4, 3], got {got}")
        got = sub.greedy_encode(b"abb", v, 0)
        if got != [3, 2]:
            problems.append(f"abb expected [3, 2], got {got}")
        # The retokenization problem in one line: this path is legal for the
        # mask, yet the tokenizer would never produce it.
        if sub.greedy_encode(b"abc", v, 0) == [1, 2, 5]:
            problems.append("abc expected [4], got the split path [1, 2, 5]")
        try:
            sub.greedy_encode(b"az", v, 0)
            problems.append("expected ValueError for a byte no token starts with")
        except ValueError:
            pass
        return (not problems), "; ".join(problems) or "expected longest-match ids, got them"

    c.check("greedy_encode takes the longest match", check_encode)

    # 8. Jump-forward.
    def check_jump():
        pat = r'\{"name": "[a-z]{1,8}", "age": (0|[1-9][0-9]{0,2})\}'
        fsm = sub.TokenFSM(sub.compile_regex(pat), vocab, sub.EOS_ID)
        ids, nxt = sub.jump_forward(fsm, fsm.start)
        # The start state has a single outgoing byte, so the forced run is
        # '{"name": "' (10 bytes) and stops where [a-z] can branch.
        forced = _decode(sub, ids, vocab)
        problems = []
        if forced != FORCED:
            problems.append(f"expected forced text {FORCED!r}, got {forced!r}")
        state = fsm.start
        for t in ids:
            state = fsm.advance(state, t)
        if state != nxt:
            problems.append(f"expected returned state {state}, got {nxt}")
        # Equal to what constrained decoding emits, for any model.
        for seed in range(30):
            out = sub.generate(fsm, SeededModel(seed, V, sub.EOS_ID, 4.0), seed)
            text = _decode(sub, out, vocab)
            if not text.startswith(forced):
                problems.append(f"seed {seed}: decoded {text[:12]!r} does not start with the forced text")
                break
        # Not forced where two bytes are possible, and not past an accepting state.
        fsm_alt = sub.TokenFSM(sub.compile_regex("(yes|no)"), vocab, sub.EOS_ID)
        if sub.jump_forward(fsm_alt, fsm_alt.start) != ([], fsm_alt.start):
            problems.append("expected nothing forced at a branch")
        fsm_opt = sub.TokenFSM(sub.compile_regex("yes(!)?"), vocab, sub.EOS_ID)
        ids2, _ = sub.jump_forward(fsm_opt, fsm_opt.start)
        got2 = _decode(sub, ids2, vocab)
        if got2 != b"yes":
            problems.append(f"jump-forward stops at an accepting state: expected b'yes', got {got2!r}")
        c.metric("jump_tokens_saved", len(ids))
        return (not problems), "; ".join(problems) or (
            f"expected forced {FORCED!r} in {len(ids)} tokens, got that")

    c.check("jump-forward equals the constrained decode and stops at an accepting state", check_jump)

    # 9. Nested structure needs a counter.
    def check_nested():
        problems = []
        sizes = []
        for d in range(1, 5):
            pat = sub.nested_array_regex(d)
            dfa = sub.compile_regex(pat)
            sizes.append(dfa.num_states)
            inside = "[" * d + "1" + "]" * d
            too_deep = "[" * (d + 1) + "1" + "]" * (d + 1)
            for text, want in [(inside, True), (too_deep, False), ("[1,[2],3]" if d >= 2 else "[1,2]", True)]:
                state = sub.walk(dfa, dfa.start, text.encode())
                got = dfa.accepting(state)
                ref = re.fullmatch(pat, text) is not None
                if got != want or ref != want:
                    problems.append(f"depth {d} on {text!r}: expected {want}, got dfa {got}, re {ref}")
        if any(b <= a for a, b in zip(sizes, sizes[1:])):
            problems.append(f"expected strictly growing state counts, got {sizes}")
        return (not problems), "; ".join(problems) or f"expected growing states, got {sizes}"

    c.check("a DFA for nesting depth d grows with d and rejects depth d + 1", check_nested)

    # 10. Per-row state in a batch.
    def check_batch():
        fsm = sub.TokenFSM(sub.compile_regex("(yes|no)"), vocab, sub.EOS_ID)
        proc = sub.ConstraintProcessor(fsm, 2)
        id_y = vocab.index(b"y")
        id_n = vocab.index(b"n")
        problems = []
        logits = torch.zeros(2, V)
        out0 = proc(None, logits)
        if not torch.isfinite(out0[0, id_y]) or torch.isfinite(out0[0, vocab.index(b"z")]):
            problems.append("row 0 at the start must allow y and forbid z")
        proc.advance([id_y, id_n])       # row 0 is after "y", row 1 after "n"
        out = proc(None, logits)
        legal0 = {vocab[i] for i in range(V) if torch.isfinite(out[0, i])}
        legal1 = {vocab[i] for i in range(V) if torch.isfinite(out[1, i])}
        if legal0 != {b"e", b"es"}:
            problems.append(f"row 0 expected {{e, es}}, got {sorted(legal0)}")
        if legal1 != {b"o"}:
            problems.append(f"row 1 expected {{o}}, got {sorted(legal1)}")
        if out.shape != logits.shape or not torch.equal(logits, torch.zeros(2, V)):
            problems.append("the processor must return a new (batch, vocab) tensor, not edit its input")
        # An empty row must raise rather than return NaN-bound logits.
        tiny = sub.TokenFSM(sub.compile_regex(r"\{a\}"), [b"", b"{", b"a"], 0)
        p2 = sub.ConstraintProcessor(tiny, 1)
        p2.advance([1])
        p2.advance([2])
        try:
            p2(None, torch.zeros(1, 3))
            problems.append("expected EmptyMaskError when the vocabulary lacks '}'")
        except sub.EmptyMaskError:
            pass
        return (not problems), "; ".join(problems) or "expected per-row masks, got them"

    c.check("each row uses its own state", check_batch)
    return c.finish()
