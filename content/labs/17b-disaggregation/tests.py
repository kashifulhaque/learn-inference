"""Checks for the disaggregation lab.

Order: the transfer bytes, the link rate, the layer-wise streaming wait, a
single colocated request, a single disaggregated request and a queue, the
planted interference trace, the planted prefill-heavy and decode-heavy
traces, and the cost of a slow link on short prompts.

The cost model (`step_time`, `prefill_seconds`) is given. The first check
pins it to chapter 17a's two numbers so an edited cost model fails loudly.
Expected values elsewhere are computed from the submission's own cost model
and `transfer_seconds`, which are given and checked once here. Tolerance on a
time is 1 ns unless a check says otherwise.

Plausible wrong implementations, each of which a check below catches:

- Charging the recurrent state per prompt token, or leaving it out, in
  `kv_transfer_bytes` (check 1).
- Answering the streaming question with the bulk-copy bound, which ignores the
  overlap with the prefill (check 3).
- A disaggregated run that leaves the KV copy out of TTFT (check 5).
- A colocated step that runs the decodes and the chunk as two separate steps
  (check 4 and check 7).
"""

import math

from lab_common import Checks

TOL = 1e-9
MIB = 1024 * 1024

HYBRID_STATE = 154_927_104
KV_PER_TOKEN = 65_536


def _call(fn, expected, tol):
    try:
        got = fn()
    except Exception as exc:  # noqa: BLE001
        return False, f"expected {expected}, got {type(exc).__name__}: {exc}"
    if got is None or abs(got - expected) > tol:
        return False, f"expected {expected}, got {got}"
    return True, f"expected {expected}, got {got}"


def _brute_force_split(sub, total, trace, config):
    best = None
    for p in range(1, total):
        cfg = dict(config, mode="disagg", prefill_gpus=p, decode_gpus=total - p)
        g = sub.simulate(trace, cfg)["goodput"]
        if best is None or g > best[0]:
            best = (g, p)
    return best


def run(sub):
    c = Checks()
    if not c.require(sub, "kv_transfer_bytes", "layerwise_exposed_seconds", "min_link_gbs",
                     "simulate_colocated_gpu", "simulate_disagg_prefill", "simulate",
                     "best_split", "step_time", "prefill_seconds", "transfer_seconds"):
        return c.finish()

    # --- cost model pinned to chapter 17a ---------------------------------
    step1 = sub.step_time(1, 1)
    pre2048 = sub.prefill_seconds(2048)
    c.check(
        "the given cost model matches chapter 17a (29.962 ms decode, 360.094 ms prefill)",
        lambda: (abs(step1 * 1e3 - 29.961670) < 1e-5 and abs(pre2048 * 1e3 - 360.094256) < 1e-5,
                 f"expected 29.961670 ms and 360.094256 ms, got {step1 * 1e3:.6f} and "
                 f"{pre2048 * 1e3:.6f}"),
    )

    # --- 1: bytes ----------------------------------------------------------
    def check_bytes():
        cases = [
            ("hybrid", 0, HYBRID_STATE),
            ("hybrid", 1024, HYBRID_STATE + KV_PER_TOKEN * 1024),
            ("hybrid", 8192, HYBRID_STATE + KV_PER_TOKEN * 8192),
            ("dense", 1024, 262_144 * 1024),
        ]
        for kind, tokens, expected in cases:
            got = sub.kv_transfer_bytes(kind, tokens)
            if got != expected:
                return False, f"expected {expected} for {kind} at {tokens} tokens, got {got}"
        got8k = sub.kv_transfer_bytes("hybrid", 8192)
        return True, f"expected {HYBRID_STATE + KV_PER_TOKEN * 8192} at 8192 tokens, got {got8k}"

    c.check("KV transfer bytes: 65,536 per token plus a fixed 154,927,104 for the state", check_bytes)

    # --- 2: min link, bulk -------------------------------------------------
    def check_bulk_link():
        tokens, budget = 8192, 0.05
        expected = (HYBRID_STATE + KV_PER_TOKEN * tokens) / budget / 1e9
        ok, msg = _call(lambda: sub.min_link_gbs("hybrid", tokens, budget), expected, 1e-9)
        if not ok:
            return ok, msg
        lat = 0.002
        expected2 = (HYBRID_STATE + KV_PER_TOKEN * tokens) / (budget - lat) / 1e9
        return _call(lambda: sub.min_link_gbs("hybrid", tokens, budget, None, lat),
                     expected2, 1e-9)

    c.check("bulk copy: the least link rate is bytes over the budget, less the setup latency",
            check_bulk_link)

    # --- 3: layer-wise streaming ------------------------------------------
    def check_layerwise():
        tokens, prefill_s = 8192, 0.64
        # Fast link: every layer drains before the next finishes, so only the
        # last layer (a full-attention layer, 4096 bytes per token) is exposed.
        last = 4096 * tokens
        fast = 300.0
        expected_fast = last / (fast * 1e9)
        got = sub.layerwise_exposed_seconds("hybrid", tokens, prefill_s, fast)
        if abs(got - expected_fast) > TOL:
            return False, f"expected {expected_fast} on a fast link, got {got}"
        # Slow link: the link is busy from layer 0's finish, so it ends at
        # prefill_s / 64 + total / link, and the wait is that minus prefill_s.
        slow = 0.05  # every layer's copy outlasts a layer's compute, so the link never idles
        total = sub.kv_transfer_bytes("hybrid", tokens)
        expected_slow = prefill_s / 64 + total / (slow * 1e9) - prefill_s
        got = sub.layerwise_exposed_seconds("hybrid", tokens, prefill_s, slow)
        if abs(got - expected_slow) > TOL:
            return False, f"expected {expected_slow} on a slow link, got {got}"
        # Streaming never exposes more than a bulk copy does.
        bulk = total / (slow * 1e9)
        if got > bulk + TOL:
            return False, f"expected at most the bulk copy {bulk}, got {got}"
        # The least rate for a 5 ms wait: it meets the budget, and 1% less does not.
        budget = 0.005
        r = sub.min_link_gbs("hybrid", tokens, budget, prefill_s)
        ok_at = sub.layerwise_exposed_seconds("hybrid", tokens, prefill_s, r)
        ok_below = sub.layerwise_exposed_seconds("hybrid", tokens, prefill_s, r * 0.99)
        bulk_rate = sub.min_link_gbs("hybrid", tokens, budget)
        if ok_at > budget + 1e-9 or ok_below <= budget:
            return False, (f"expected the wait at {r:.4f} GB/s to be <= {budget} and at 99% of it "
                           f"to be > {budget}, got {ok_at} and {ok_below}")
        if r >= bulk_rate:
            return False, f"expected a layer-wise rate below the bulk rate {bulk_rate}, got {r}"
        return True, (f"expected fast {expected_fast}, slow {expected_slow}; got both. "
                      f"Least rate {r:.4f} GB/s against bulk {bulk_rate:.4f} GB/s")

    c.check("layer-wise streaming hides the copy under the prefill", check_layerwise)

    # --- 4: one colocated request -----------------------------------------
    def check_colocated_one():
        trace = [{"arrival": 0.5, "prompt": 2048, "output": 3}]
        steps = [sub.step_time(0, 0, [(512 * k, 512)]) for k in range(4)]
        d1 = sub.step_time(1, 2049)
        d2 = sub.step_time(1, 2050)
        exp_ttft = sum(steps)
        exp_tpot = (d1 + d2) / 2
        r = sub.simulate(trace, {"mode": "colocated", "gpus": 1, "chunk": 512, "max_batch": 8})
        if abs(r["p99_ttft"] - exp_ttft) > TOL:
            return False, f"expected TTFT {exp_ttft} (four 512-token chunks), got {r['p99_ttft']}"
        if abs(r["p99_tpot"] - exp_tpot) > TOL:
            return False, f"expected TPOT {exp_tpot}, got {r['p99_tpot']}"
        if abs(r["span"] - (exp_ttft + d1 + d2)) > TOL:
            return False, f"expected span {exp_ttft + d1 + d2}, got {r['span']}"
        # A request that decodes while another prompt is chunked shares its step.
        trace2 = [{"arrival": 0.0, "prompt": 512, "output": 4},
                  {"arrival": 0.0, "prompt": 1024, "output": 2}]
        r2 = sub.simulate(trace2, {"mode": "colocated", "gpus": 1, "chunk": 512, "max_batch": 8})
        s1 = sub.step_time(0, 0, [(0, 512)])               # request 0's prompt
        s2 = sub.step_time(1, 513, [(0, 512)])            # req 0 decodes, req 1 chunk 1
        s3 = sub.step_time(1, 514, [(512, 512)])          # req 0 decodes, req 1 chunk 2
        exp_ttft0, exp_ttft1 = s1, s1 + s2 + s3
        if abs(r2["p50_ttft"] - min(exp_ttft0, exp_ttft1)) > TOL:
            return False, f"expected the first TTFT {exp_ttft0}, got {r2['p50_ttft']}"
        if abs(r2["p99_ttft"] - exp_ttft1) > TOL:
            return False, f"expected the second TTFT {exp_ttft1}, got {r2['p99_ttft']}"
        return True, f"expected TTFT {exp_ttft}, got {r['p99_ttft']}; shared-step TTFT {exp_ttft1}"

    c.check("a colocated GPU chunks the prompt and shares each step with the decodes",
            check_colocated_one)

    # --- 5: disaggregated, one request and a queue -------------------------
    def check_disagg():
        link = 300.0
        P = sub.prefill_seconds(2048)
        nbytes = HYBRID_STATE + KV_PER_TOKEN * 2048
        X = sub.transfer_seconds(nbytes, link)
        base = {"mode": "disagg", "prefill_gpus": 1, "decode_gpus": 1, "link_gbs": link,
                "max_batch": 8}
        one = [{"arrival": 0.0, "prompt": 2048, "output": 3}]
        r = sub.simulate(one, base)
        if abs(r["p99_ttft"] - (P + X)) > TOL:
            return False, f"expected TTFT {P + X} (prefill plus copy), got {r['p99_ttft']}"
        d1, d2 = sub.step_time(1, 2049), sub.step_time(1, 2050)
        if abs(r["p99_tpot"] - (d1 + d2) / 2) > TOL:
            return False, f"expected TPOT {(d1 + d2) / 2}, got {r['p99_tpot']}"
        two = [{"arrival": 0.0, "prompt": 2048, "output": 2},
               {"arrival": 0.0, "prompt": 2048, "output": 2}]
        r = sub.simulate(two, base)
        if abs(r["p99_ttft"] - (2 * P + X)) > TOL:
            return False, f"expected the queued TTFT {2 * P + X}, got {r['p99_ttft']}"
        r2 = sub.simulate(two, dict(base, prefill_gpus=2))
        if abs(r2["p99_ttft"] - (P + X)) > TOL:
            return False, f"expected {P + X} with two prefill GPUs, got {r2['p99_ttft']}"
        lw = sub.simulate(one, dict(base, layerwise=True))
        exp_lw = P + sub.layerwise_exposed_seconds("hybrid", 2048, P, link)
        if abs(lw["p99_ttft"] - exp_lw) > TOL:
            return False, f"expected the streamed TTFT {exp_lw}, got {lw['p99_ttft']}"
        return True, f"expected TTFT {P + X}, queued {2 * P + X}, streamed {exp_lw}; got all"

    c.check("a disaggregated request waits for its prefill and its KV copy", check_disagg)

    # --- 6: planted interference trace ------------------------------------
    decoders = [{"arrival": 0.0, "prompt": 128, "output": 300} for _ in range(8)]
    trace_i = decoders + [{"arrival": 2.0, "prompt": 8192, "output": 2}]
    pre8k = sub.prefill_seconds(8192)

    def check_interference():
        def colo(chunk):
            return sub.simulate(trace_i, {"mode": "colocated", "gpus": 1, "chunk": chunk,
                                          "max_batch": 64})
        whole = colo(1_000_000)
        mid = colo(512)
        small = colo(160)
        dis = sub.simulate(trace_i, {"mode": "disagg", "prefill_gpus": 1, "decode_gpus": 1,
                                     "link_gbs": 300, "max_batch": 64})
        # 8 decoders stall for the whole prompt, one step of at least pre8k.
        if not (pre8k <= whole["p99_worst_gap"] <= pre8k * 1.02):
            return False, (f"expected the unchunked worst gap in [{pre8k}, {pre8k * 1.02}], "
                           f"got {whole['p99_worst_gap']}")
        # A 512-token chunk is above the ridge: the step is compute-bound.
        if not (0.090 <= mid["p99_worst_gap"] <= 0.100):
            return False, f"expected the 512-chunk worst gap in [0.090, 0.100], got {mid['p99_worst_gap']}"
        if not small["p99_worst_gap"] < 0.034:
            return False, f"expected the 160-chunk worst gap under 0.034, got {small['p99_worst_gap']}"
        if not dis["p99_worst_gap"] < 0.032:
            return False, f"expected the disaggregated worst gap under 0.032, got {dis['p99_worst_gap']}"
        # The mean hides the stall.
        if not whole["p99_tpot"] < 0.045:
            return False, f"expected mean TPOT under 0.045 even unchunked, got {whole['p99_tpot']}"
        # The 8192-token prompt's own TTFT: chunking costs it, disaggregation does not.
        if not (small["p99_ttft"] > dis["p99_ttft"] * 1.1):
            return False, (f"expected the 160-chunk TTFT {small['p99_ttft']} to exceed "
                           f"1.1 x {dis['p99_ttft']}")
        return True, (f"expected worst gaps {pre8k:.3f}, ~0.097, <0.034, <0.032; got "
                      f"{whole['p99_worst_gap']:.3f}, {mid['p99_worst_gap']:.3f}, "
                      f"{small['p99_worst_gap']:.3f}, {dis['p99_worst_gap']:.3f}")

    c.check("colocation: a chunk trades the decoders' stall against the prompt's TTFT",
            check_interference)

    # --- 7: planted prefill-heavy trace -----------------------------------
    slo = {"slo_ttft": 1.6, "slo_tpot": 0.036, "tpot_stat": "max", "max_batch": 32,
           "link_gbs": 300.0}
    heavy = sub.poisson_trace(100, 3.0, 8192, 32, 7)

    def check_prefill_heavy():
        exp_g, exp_p = _brute_force_split(sub, 8, heavy, slo)
        got = sub.best_split(8, heavy, slo)
        if got["prefill_gpus"] != exp_p or got["prefill_gpus"] + got["decode_gpus"] != 8:
            return False, f"expected {exp_p} prefill GPUs (brute force), got {got}"
        if abs(got["goodput"] - exp_g) > 1e-9:
            return False, f"expected goodput {exp_g}, got {got['goodput']}"
        if not got["prefill_gpus"] > got["decode_gpus"]:
            return False, f"expected more prefill than decode GPUs on a prefill-heavy trace, got {got}"
        best_colo = max(
            sub.simulate(heavy, dict(slo, mode="colocated", gpus=8, chunk=ch))["goodput"]
            for ch in (128, 160, 192, 256, 512, 4096)
        )
        if not exp_g > best_colo * 1.2:
            return False, (f"expected disaggregated goodput {exp_g} to beat the best colocated "
                           f"{best_colo} by 20%")
        return True, (f"expected split {exp_p}:{8 - exp_p}; got {got['prefill_gpus']}:"
                      f"{got['decode_gpus']}, goodput {exp_g:.3f} against colocated {best_colo:.3f}")

    c.check("the split search matches brute force, and a prefill-heavy trace favors disaggregation",
            check_prefill_heavy)

    # --- 8: planted decode-heavy trace ------------------------------------
    light = sub.poisson_trace(100, 8.0, 128, 512, 11)
    slo2 = {"slo_ttft": 1.0, "slo_tpot": 0.04, "tpot_stat": "max", "max_batch": 32,
            "link_gbs": 300.0}

    def check_decode_heavy():
        exp_g, exp_p = _brute_force_split(sub, 8, light, slo2)
        got = sub.best_split(8, light, slo2)
        if got["prefill_gpus"] != exp_p or abs(got["goodput"] - exp_g) > 1e-9:
            return False, f"expected {exp_p} prefill GPUs and goodput {exp_g}, got {got}"
        if not got["prefill_gpus"] < got["decode_gpus"]:
            return False, f"expected fewer prefill than decode GPUs on a decode-heavy trace, got {got}"
        return True, f"expected {exp_p} prefill GPUs, got {got['prefill_gpus']}"

    c.check("a decode-heavy trace puts most GPUs on decode", check_decode_heavy)

    # --- 9: short prompts over a slow link --------------------------------
    def check_slow_link():
        link = 12.5  # 100 Gbit/s Ethernet, one way
        trace = sub.poisson_trace(40, 2.0, 128, 16, 5)
        dis = sub.simulate(trace, {"mode": "disagg", "prefill_gpus": 1, "decode_gpus": 1,
                                   "link_gbs": link, "max_batch": 32})
        fast = sub.simulate(trace, {"mode": "disagg", "prefill_gpus": 1, "decode_gpus": 1,
                                    "link_gbs": 1e6, "max_batch": 32})
        X = (HYBRID_STATE + KV_PER_TOKEN * 128) / (link * 1e9)
        P = sub.prefill_seconds(128)
        diff = dis["p50_ttft"] - fast["p50_ttft"]
        if abs(diff - X) > 1e-6:
            return False, f"expected the slow link to add {X} to the median TTFT, got {diff}"
        if not X > 0.4 * P:
            return False, f"expected the copy {X} to exceed 40% of the prefill {P}"
        colo = sub.simulate(trace, {"mode": "colocated", "gpus": 2, "chunk": 160, "max_batch": 32})
        if not colo["p50_ttft"] < dis["p50_ttft"]:
            return False, (f"expected colocated median TTFT {colo['p50_ttft']} below the "
                           f"disaggregated {dis['p50_ttft']}")
        return True, (f"expected the copy to add {X * 1e3:.3f} ms ({X / P:.0%} of a "
                      f"{P * 1e3:.1f} ms prefill); got {diff * 1e3:.3f} ms")

    c.check("short prompts over a 12.5 GB/s link: the copy is a large share of TTFT",
            check_slow_link)

    try:
        gbs8k = sub.min_link_gbs("hybrid", 8192, 0.1 * pre8k)
        c.metric("kv_bytes_8192", sub.kv_transfer_bytes("hybrid", 8192))
        c.metric("min_link_gbs_8192", round(gbs8k, 4))
        best = sub.best_split(8, heavy, slo)
        c.metric("best_prefill_gpus", best["prefill_gpus"])
        c.metric("disagg_goodput_per_s", round(best["goodput"], 4))
    except Exception:  # noqa: BLE001
        pass  # a starter that is not finished has no metrics yet
    return c.finish()
