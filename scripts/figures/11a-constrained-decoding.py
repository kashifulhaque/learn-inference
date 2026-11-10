"""Numbers for chapter 11a.

Everything printed here comes from content/labs/11a-constrained-decoding/solution.py.
Nothing is an A100 measurement. The timing rows are pure-Python CPU timings on
whatever machine runs this script. The copy-time rows are arithmetic on the
course's PCIe figure.

Run from the repository root:

    python3 scripts/figures/11a-constrained-decoding.py          # the numbers
    python3 scripts/figures/11a-constrained-decoding.py --viz    # the data in 11a-token-mask.tsx
"""

from __future__ import annotations

import importlib.util
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "content" / "labs" / "11a-constrained-decoding"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sol = load("lab11a_solution", LAB / "solution.py")
VOCAB = sol.toy_vocab()
JSON_PATTERN = r'\{"name": "[a-z]{1,8}", "age": (0|[1-9][0-9]{0,2})\}'

# Tokens the figure shows. The first five are multi-character or byte tokens.
SHOW = [b"y", b"n", b"ye", b"es", b"yes", b"no", b"maybe", b"e", b"o", b"s"]


def legal_set(pattern: str, prefix: bytes) -> list[bytes]:
    dfa = sol.compile_regex(pattern)
    state = sol.walk(dfa, dfa.start, prefix)
    mask = sol.build_token_mask(dfa, state, VOCAB, sol.EOS_ID)
    return [VOCAB[i] for i in range(len(VOCAB)) if bool(mask[i])]


def viz_data() -> dict:
    pattern = "(yes|no)"
    dfa = sol.compile_regex(pattern)
    rows = []
    for prefix in [b"", b"y", b"ye", b"yes", b"n", b"no"]:
        state = sol.walk(dfa, dfa.start, prefix)
        mask = sol.build_token_mask(dfa, state, VOCAB, sol.EOS_ID)
        shown = {t.decode(): bool(mask[VOCAB.index(t)]) for t in SHOW}
        shown["END"] = bool(mask[sol.EOS_ID])
        rows.append({"prefix": prefix.decode(), "legal": int(mask.sum()), "tokens": shown})
    return {"pattern": pattern, "vocab": len(VOCAB), "rows": rows}


def main() -> None:
    if "--viz" in sys.argv:
        print(json.dumps(viz_data(), indent=2))
        return

    V = len(VOCAB)
    print(f"toy vocabulary: {V} tokens (end, pad, 256 single bytes, {V - 258} multi-character)")

    print("\n== Worked mask, pattern (yes|no) ==")
    for row in viz_data()["rows"]:
        print(f"prefix {row['prefix']!r:6} legal {row['legal']:3d} of {V}")
    print("after 'y':", legal_set("(yes|no)", b"y"))

    print("\n== Worked draw, T=1: tokens maybe, y, yes, n, no ==")
    names = ["maybe", "y", "yes", "n", "no"]
    logits = torch.tensor([4.0, 2.0, 1.0, 3.0, 0.0])
    mask = torch.tensor([False, True, True, True, True])
    free = logits.softmax(-1)
    masked = logits.masked_fill(~mask, float("-inf")).softmax(-1)
    for n, a, b in zip(names, free.tolist(), masked.tolist()):
        print(f"{n:6} unconstrained {a:.4f}  masked {b:.4f}")
    print(f"mass on illegal 'maybe' before the mask: {free[0].item():.4f}")
    k1 = logits.topk(1).indices.tolist()
    print(f"top-k=1 keeps {names[k1[0]]!r}, which the mask then removes: empty set")

    print(f"\n== JSON pattern: {JSON_PATTERN} ==")
    dfa = sol.compile_regex(JSON_PATTERN)
    fsm = sol.TokenFSM(dfa, VOCAB, sol.EOS_ID)
    print(f"DFA states {dfa.num_states}")
    start_legal = legal_set(JSON_PATTERN, b"")
    print(f"legal tokens at start {len(start_legal)}: {start_legal}")
    ids, nxt = sol.jump_forward(fsm, fsm.start)
    print(f"jump-forward from start: ids {ids} = {[VOCAB[i] for i in ids]}, "
          f"{len(b''.join(VOCAB[i] for i in ids))} bytes in {len(ids)} tokens, state {nxt}")
    state = dfa.start
    n_forced = 0
    path = []
    # The whole pattern, walking one example, counting forced steps.
    sample = b'{"name": "bo", "age": 7}'
    for b in sample:
        forced = len(dfa.trans[state]) == 1 and not dfa.accepting(state)
        n_forced += forced
        state = dfa.step(state, b)
    print(f"example {sample.decode()}: {n_forced} of {len(sample)} bytes are forced")

    print("\n== Mask size ==")
    full = 248_320
    row_bool, row_packed = full, math.ceil(full / 8)
    print(f"one row, bool: {row_bool:,} B; bit-packed: {row_packed:,} B")
    for batch in (1, 128):
        b_bool, b_packed = batch * row_bool, batch * row_packed
        t_bool = b_bool / 32e9 * 1e3
        t_packed = b_packed / 32e9 * 1e3
        print(f"batch {batch:3d}: bool {b_bool:,} B ({t_bool:.3f} ms at 32 GB/s), "
              f"packed {b_packed:,} B ({t_packed:.3f} ms)")
    step_ms = 53.8e9 / 1275e9 * 1e3
    print(f"decode weight read at 1275 GB/s: {step_ms:.1f} ms; "
          f"bool mask copy at batch 128 is {128 * row_bool / 32e9 * 1e3 / step_ms * 100:.1f}% of it")
    print(f"mask cache for {dfa.num_states} states, bool: {dfa.num_states * row_bool:,} B")

    print("\n== Mask build time, pure Python on this CPU ==")
    rng = random.Random(5)
    alphabet = b'abc01{}":, '
    big = [b""] + [bytes([i]) for i in range(256)]
    big += [bytes(rng.choice(alphabet) for _ in range(rng.randint(2, 6)))
            for _ in range(full - len(big))]
    big_fsm = sol.TokenFSM(dfa, big, 0)
    t0 = time.perf_counter()
    big_fsm.mask(big_fsm.start)
    build = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(1000):
        big_fsm.mask(big_fsm.start)
    cached = (time.perf_counter() - t0) / 1000
    print(f"vocab {len(big):,}: build {build * 1e3:.1f} ms, cached {cached * 1e6:.2f} us")

    print("\n== Nesting depth (unminimized subset DFA states) ==")
    for d in range(1, 6):
        pat = sol.nested_array_regex(d)
        print(f"depth {d}: {sol.compile_regex(pat).num_states} states, regex length {len(pat)}")

    print("\n== Property run: 250 seeds, JSON pattern, EOS boost 4 ==")

    class Model:
        def __init__(self, seed):
            self.seed = seed

        def __call__(self, prefix):
            g = torch.Generator().manual_seed(self.seed * 1000 + len(prefix))
            logits = torch.randn(V, generator=g) * 2.0
            logits[sol.EOS_ID] += 4.0
            return logits

    good = mismatch = 0
    example = None
    for seed in range(250):
        ids = sol.generate(fsm, Model(seed), seed)
        text = b"".join(VOCAB[i] for i in ids if i != sol.EOS_ID).decode()
        ok = re.fullmatch(JSON_PATTERN, text, re.ASCII) is not None
        if ok:
            json.loads(text)
        good += ok
        body = [i for i in ids if i != sol.EOS_ID]
        canon = sol.greedy_encode(text.encode(), VOCAB, sol.EOS_ID)
        if canon != body:
            mismatch += 1
            if example is None and len(body) < 14:
                example = (text, [VOCAB[i] for i in body], [VOCAB[i] for i in canon])
    print(f"fullmatch and json.loads: {good}/250; path differs from greedy encoding: {mismatch}/250")
    if example:
        print("example text:", example[0])
        print("sampled path:", example[1])
        print("greedy path :", example[2])

    print("\n== Why the leading-zero pattern matters ==")
    loose = r'\{"name": "[a-z]{1,8}", "age": [0-9]{1,3}\}'
    text = '{"name": "ab", "age": 007}'
    print(f"{text} fullmatches the loose pattern: {re.fullmatch(loose, text) is not None}")
    try:
        json.loads(text)
    except ValueError as exc:
        print("json.loads rejects it:", exc)


if __name__ == "__main__":
    main()
