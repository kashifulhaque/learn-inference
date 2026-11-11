"""Checks for the observability and cold-start lab.

Order: parse the log, per-request metrics, the cumulative histogram, the
quantile read back from buckets, merging replicas, a counter's rate, the alert
state machine, the planted storm in a gauge scenario, the probes, and the cold
start.

Expected values are hand-computable on small inputs. The seeded log and the
seeded gauges are given, and the checks on them state a property or compute
the expected value from the raw samples.

Plausible wrong implementations, each of which a check below catches:

- Non-cumulative buckets, where `cum[i]` counts only the samples between
  `le[i-1]` and `le[i]` (check 3).
- A quantile with no interpolation, which returns the bucket's upper bound
  (check 4).
- Averaging per-replica p99s instead of merging histograms (check 5).
- A counter rate that reads a restart as a negative increase (check 6).
- An alert that fires on one sample, which ignores `for_s` (checks 7 and 8).
- An alert with no `keep_firing_for_s`, which pages twice for one incident
  (checks 7 and 8).
- Readiness that goes true when the weights load, before warmup, or liveness
  that waits for readiness and gets the pod killed during the load (check 9).
- A streaming loader time that adds the read and the copy, which is the mmap
  time (check 10).
"""

import math

from lab_common import Checks

TOL = 1e-9


def _close(a, b, tol=TOL):
    return a is not None and abs(a - b) <= tol


def run(sub):
    c = Checks()
    if not c.require(
        sub, "parse_events", "request_metrics", "build_histogram", "histogram_quantile",
        "merge_histograms", "counter_rate", "evaluate_alert", "probe_states",
        "min_failure_threshold", "weight_load_seconds", "time_to_ready", "warm_pool_size",
        "synth_event_log", "latency_samples", "exact_quantile", "synth_gauges", "rate_series",
        "LAT_BOUNDS",
    ):
        return c.finish()

    # --- 1: parsing --------------------------------------------------------
    HAND = [
        "0.500 req=1 preempted",
        "0.000 req=1 arrived",
        "0.020 req=1 scheduled",
        "0.100 req=1 token",
        "# a comment",
        "0.130 req=1 token",
        "",
        "0.900 req=1 scheduled",
        "0.950 req=1 token",
        "1.000 req=1 finished",
        "0.050 req=2 arrived",
        "0.7 req= token",
        "0.350 req=2 scheduled",
        "garbage",
        "0.600 req=2 token",
    ]

    def check_parse():
        events, bad = sub.parse_events(HAND)
        if bad != 2:
            return False, f"expected 2 bad lines (the truncated one and the garbage), got {bad}"
        if len(events) != 11:
            return False, f"expected 11 events, got {len(events)}"
        times = [e["t"] for e in events]
        if times != sorted(times):
            return False, f"expected events sorted by time, got {times}"
        first = events[0]
        if (first["t"], first["req"], first["event"]) != (0.0, 1, "arrived"):
            return False, f"expected (0.0, 1, 'arrived') first, got {first}"
        last = events[-1]
        if (last["t"], last["req"], last["event"]) != (1.0, 1, "finished"):
            return False, f"expected (1.0, 1, 'finished') last, got {last}"
        return True, "expected 11 events sorted by time and 2 bad lines; got the same"

    c.check("parse_events: sorts by time, skips comments, counts bad lines", check_parse)

    # --- 2: per-request metrics -------------------------------------------
    def check_metrics():
        events, _ = sub.parse_events(HAND)
        m = sub.request_metrics(events)
        r1, r2 = m[1], m[2]
        want = {
            "r1 queue": (r1["queue"], 0.02),
            "r1 ttft": (r1["ttft"], 0.1),
            "r2 queue": (r2["queue"], 0.3),
            "r2 ttft": (r2["ttft"], 0.55),
        }
        for name, (got, exp) in want.items():
            if not _close(got, exp):
                return False, f"expected {name} = {exp}, got {got}"
        if len(r1["itl"]) != 2 or not (_close(r1["itl"][0], 0.03) and _close(r1["itl"][1], 0.82)):
            return False, f"expected r1 itl [0.03, 0.82] (the stall stays in the gap), got {r1['itl']}"
        if r2["itl"] != []:
            return False, f"expected r2 itl [], got {r2['itl']}"
        if r1["preemptions"] != 1 or r2["preemptions"] != 0:
            return False, f"expected preemptions 1 and 0, got {r1['preemptions']} and {r2['preemptions']}"
        # The queue wait ends at the FIRST schedule, not the one after the preemption.
        lines = sub.synth_event_log()
        ev, bad = sub.parse_events(lines)
        if bad != 2:
            return False, f"expected 2 bad lines in the synthetic log, got {bad}"
        mm = sub.request_metrics(ev)
        if len(mm) != 400:
            return False, f"expected 400 requests, got {len(mm)}"
        return True, "expected queue 0.02 and 0.3, ttft 0.1 and 0.55, itl [0.03, 0.82]; got the same"

    c.check("request_metrics: queue time, TTFT, and ITL gaps on a hand-made log", check_metrics)

    # --- 3: the cumulative histogram --------------------------------------
    def check_histogram():
        h = sub.build_histogram([0.05, 0.1, 0.1, 0.3, 9.0], [0.1, 0.5, 1.0])
        if list(h["le"][:3]) != [0.1, 0.5, 1.0] or h["le"][3] != math.inf:
            return False, f"expected le [0.1, 0.5, 1.0, inf], got {h['le']}"
        # A sample equal to a bound belongs in that bound's bucket (le is "less than or equal").
        if list(h["cum"]) != [3, 4, 4, 5]:
            return False, f"expected cumulative counts [3, 4, 4, 5], got {list(h['cum'])}"
        if h["count"] != 5 or not _close(h["sum"], 9.55):
            return False, f"expected count 5 and sum 9.55, got {h['count']} and {h['sum']}"
        e = sub.build_histogram([], [0.1, 0.5])
        if list(e["cum"]) != [0, 0, 0] or e["count"] != 0:
            return False, f"expected an empty histogram [0, 0, 0], got {list(e['cum'])}"
        # Every histogram on the real log is cumulative and ends at the count.
        ev, _ = sub.parse_events(sub.synth_event_log())
        s = sub.latency_samples(sub.request_metrics(ev))
        for name, xs in s.items():
            hh = sub.build_histogram(xs, sub.LAT_BOUNDS)
            if list(hh["cum"]) != sorted(hh["cum"]) or hh["cum"][-1] != len(xs):
                return False, f"expected a non-decreasing {name} histogram ending at {len(xs)}, got {list(hh['cum'])}"
        return True, "expected cum [3, 4, 4, 5], count 5, sum 9.55; got the same"

    c.check("build_histogram: cumulative le buckets with +Inf", check_histogram)

    # --- 4: the quantile read back from buckets ---------------------------
    def check_quantile():
        bounds = [0.1, 0.2, 0.3, 0.5]
        h = sub.build_histogram([0.22] * 100, bounds)
        got95 = sub.histogram_quantile(0.95, h)
        got50 = sub.histogram_quantile(0.5, h)
        # All 100 samples are 0.22 and fall in (0.2, 0.3]. Interpolation assumes they are
        # spread evenly there: 0.2 + 0.1 * 95/100 = 0.295, 0.2 + 0.1 * 50/100 = 0.25.
        if not (_close(got95, 0.295) and _close(got50, 0.25)):
            return False, f"expected p95 0.295 and p50 0.25 (exact is 0.22), got {got95} and {got50}"
        # More than one bucket: 60 at 0.05 and 40 at 0.15, p80 sits in (0.1, 0.2].
        h2 = sub.build_histogram([0.05] * 60 + [0.15] * 40, bounds)
        got = sub.histogram_quantile(0.8, h2)
        exp = 0.1 + 0.1 * (80 - 60) / 40
        if not _close(got, exp):
            return False, f"expected p80 {exp}, got {got}"
        # A quantile in the +Inf bucket cannot be recovered: the answer is the top finite bound.
        h3 = sub.build_histogram([0.05] * 50 + [7.0] * 50, [0.1, 1.0])
        got = sub.histogram_quantile(0.99, h3)
        if not _close(got, 1.0):
            return False, f"expected p99 1.0 (the highest finite bound; the truth is 7.0), got {got}"
        if not math.isnan(sub.histogram_quantile(0.5, sub.build_histogram([], bounds))):
            return False, "expected nan for an empty histogram"
        # On the seeded log, the estimate lies in the bucket that holds the exact
        # nearest-rank sample, so its error is at most that bucket's width.
        ev, _ = sub.parse_events(sub.synth_event_log())
        s = sub.latency_samples(sub.request_metrics(ev))
        B = [0.0] + list(sub.LAT_BOUNDS)
        for name in ("queue", "ttft", "itl"):
            xs = sorted(s[name])
            hh = sub.build_histogram(xs, sub.LAT_BOUNDS)
            for q in (0.5, 0.9, 0.99):
                est = sub.histogram_quantile(q, hh)
                nr = xs[max(math.ceil(q * len(xs)), 1) - 1]
                lo = max(b for b in B if b < nr) if nr > B[1] else 0.0
                hi = min((b for b in sub.LAT_BOUNDS if b >= nr), default=sub.LAT_BOUNDS[-1])
                if not (lo - 1e-12 <= est <= hi + 1e-12):
                    return False, (f"expected the {name} p{q * 100:g} estimate in [{lo}, {hi}] "
                                   f"(the bucket of the exact sample {nr}), got {est}")
        return True, "expected p95 0.295 and p50 0.25 on 100 samples at 0.22; got the same"

    c.check("histogram_quantile: linear interpolation, error bounded by the bucket", check_quantile)

    # --- 5: merging replicas ----------------------------------------------
    def check_merge():
        a = [0.05] * 1000
        b = [2.0] * 10
        ha = sub.build_histogram(a, sub.LAT_BOUNDS)
        hb = sub.build_histogram(b, sub.LAT_BOUNDS)
        merged = sub.merge_histograms(ha, hb)
        whole = sub.build_histogram(a + b, sub.LAT_BOUNDS)
        if list(merged["cum"]) != list(whole["cum"]) or merged["count"] != 1010 or not _close(merged["sum"], whole["sum"]):
            return False, f"expected the merge to equal the histogram of all samples, got {list(merged['cum'])}"
        p_merged = sub.histogram_quantile(0.99, merged)
        exp = 0.025 + 0.025 * 999.9 / 1000
        if not _close(p_merged, exp):
            return False, f"expected merged p99 {exp}, got {p_merged}"
        # The mean of the replicas' p99s is 1.267 s and describes no request.
        avg = (sub.histogram_quantile(0.99, ha) + sub.histogram_quantile(0.99, hb)) / 2
        if abs(avg - p_merged) < 1.0:
            return False, f"expected the averaged p99 {avg} to be far from the merged p99 {p_merged}"
        try:
            sub.merge_histograms(ha, sub.build_histogram(a, [0.1, 1.0]))
        except ValueError:
            return True, f"expected merged p99 {exp}, got {p_merged}; the average {avg:.3f} is wrong"
        return False, "expected ValueError when the bucket bounds differ"

    c.check("merge_histograms: add replicas, never average their percentiles", check_merge)

    # --- 6: counter rate ---------------------------------------------------
    def check_rate():
        s = [(0.0, 0.0), (10.0, 50.0), (20.0, 100.0), (30.0, 4.0)]
        got = sub.counter_rate(s, 30.0, 30.0)
        exp = (50 + 50 + 4) / 30
        if not _close(got, exp):
            return False, f"expected {exp} with a reset to 0 before the last sample, got {got}"
        got = sub.counter_rate(s, 20.0, 10.0)
        if not _close(got, 5.0):
            return False, f"expected 5.0 over the window [10, 20], got {got}"
        got = sub.counter_rate(s, 5.0, 3.0)
        if got is not None:
            return False, f"expected None with fewer than two samples in the window, got {got}"
        return True, f"expected {exp}, 5.0, None; got the same"

    c.check("counter_rate: windowed increase per second, with counter resets", check_rate)

    # --- 7: the alert state machine ---------------------------------------
    def check_alert():
        vals = [0, 1, 1, 1, 0, 1, 1, 1, 1, 1]
        s = [(float(i), float(v)) for i, v in enumerate(vals)]
        got = sub.evaluate_alert(s, 1.0, 2.0, 0.0)
        if got != [(3.0, 4.0), (7.0, None)]:
            return False, f"expected [(3.0, 4.0), (7.0, None)] with for 2 and keep 0, got {got}"
        got = sub.evaluate_alert(s, 1.0, 2.0, 2.0)
        if got != [(3.0, None)]:
            return False, f"expected one interval [(3.0, None)] with keep 2, got {got}"
        spike = [(float(i), float(v)) for i, v in enumerate([0, 0, 1, 0, 0, 0])]
        got = sub.evaluate_alert(spike, 1.0, 2.0, 0.0)
        if got != []:
            return False, f"expected [] for a one-sample spike, got {got}"
        got = sub.evaluate_alert(spike, 1.0, 0.0, 0.0)
        if got != [(2.0, 3.0)]:
            return False, f"expected [(2.0, 3.0)] with for 0, got {got}"
        # The condition is >=.
        got = sub.evaluate_alert([(0.0, 5.0), (1.0, 5.0)], 5.0, 1.0, 0.0)
        if got != [(1.0, None)]:
            return False, f"expected [(1.0, None)] when the value equals the threshold, got {got}"
        return True, "expected [(3.0, 4.0), (7.0, None)], [(3.0, None)], [], [(2.0, 3.0)]; got the same"

    c.check("evaluate_alert: for and keep_firing_for on hand-made series", check_alert)

    # --- 8: a planted storm ------------------------------------------------
    def check_storm():
        g = sub.synth_gauges()
        kv = g["kv"]
        got = sub.evaluate_alert(kv, 0.95, 30.0, 15.0)
        if got != [(230.0, 335.0)]:
            return False, (f"expected one KV-full alert [(230.0, 335.0)]: the spike at t=100 and the "
                           f"flapping at t=400 to 439 never hold 30 s, and the dip at t=260 to 262 "
                           f"is shorter than 15 s; got {got}")
        naive = sub.evaluate_alert(kv, 0.95, 30.0, 0.0)
        if naive != [(230.0, 260.0), (293.0, 320.0)]:
            return False, f"expected the dip to split the alert in two without keep_firing_for, got {naive}"
        one = sub.evaluate_alert(kv, 0.95, 0.0, 0.0)
        if len(one) != 23:
            return False, f"expected 23 pages from an alert with no for (1 spike, 2 storm halves, 20 flaps), got {len(one)}"
        rates = sub.rate_series(g["preemptions"], 30.0)
        storm = sub.evaluate_alert(rates, 1.0, 30.0, 15.0)
        if len(storm) != 1:
            return False, f"expected one preemption-storm alert, got {storm}"
        start, end = storm[0]
        if not (200.0 <= start <= 270.0 and end is not None and 320.0 <= end <= 400.0):
            return False, (f"expected a start in [200, 270] and an end in [320, 400], got "
                           f"({start}, {end}). The reset at t=500 must not matter")
        c.metric("kv_alert_start_s", got[0][0])
        c.metric("preemption_alert_start_s", start)
        return True, (f"expected KV alert [(230.0, 335.0)], got {got}; preemption alert starts at "
                      f"{start} and ends at {end}")

    c.check("a planted storm: one alert for the incident, none for a spike or a flap", check_storm)

    # --- 9: probes ---------------------------------------------------------
    def check_probes():
        image, load, warm = 8.0, 26.9, 18.0
        want = [
            (0.0, "pulling", False, False),
            (7.9, "pulling", False, False),
            (8.0, "loading", True, False),
            (20.0, "loading", True, False),
            (34.9, "warmup", True, False),
            (52.8, "warmup", True, False),
            (52.9, "ready", True, True),
            (100.0, "ready", True, True),
        ]
        for t, phase, live, ready in want:
            got = sub.probe_states(t, image, load, warm)
            if (got["phase"], got["live"], got["ready"]) != (phase, live, ready):
                return False, (f"expected {(phase, live, ready)} at t={t}, got "
                               f"{(got['phase'], got['live'], got['ready'])}")
        cases = [(44.915625, 10.0, 5), (50.0, 10.0, 5), (50.0001, 10.0, 6), (0.5, 10.0, 1), (300.0, 10.0, 30)]
        for startup, period, exp in cases:
            got = sub.min_failure_threshold(startup, period)
            if got != exp:
                return False, f"expected failureThreshold {exp} for {startup} s at {period} s, got {got}"
        return True, "expected (pulling, loading, warmup, ready) phases and thresholds 5, 5, 6, 1, 30; got the same"

    c.check("probe_states and min_failure_threshold: live while loading, ready after warmup", check_probes)

    # --- 10: the cold start ------------------------------------------------
    def check_cold_start():
        w = 53.8
        cases = [
            # (source GB/s, loader, expected seconds)
            (2.0, "mmap", 53.8 / 2.0 + 53.8 / 32.0),                       # 28.58125
            (2.0, "stream", 53.8 / 2.0 + min(0.5 / 2.0, 0.5 / 32.0)),  # 26.915625
            (64.0, "mmap", 53.8 / 64.0 + 53.8 / 32.0),                     # 2.521875
            (64.0, "stream", 53.8 / 32.0 + min(0.5 / 64.0, 0.5 / 32.0)),   # 1.6890625
        ]
        for src, loader, exp in cases:
            got = sub.weight_load_seconds(w, src, 32.0, loader, 0.5)
            if not _close(got, exp, 1e-9):
                return False, f"expected {exp} s for {loader} at {src} GB/s, got {got}"
        try:
            sub.weight_load_seconds(w, 2.0, 32.0, "carrier pigeon")
        except ValueError:
            pass
        else:
            return False, "expected ValueError for an unknown loader"
        t = sub.time_to_ready(8.0, 1.0, 2.0, "stream", 9, 2.0)
        exp = {"image_pull_s": 8.0, "weight_load_s": 26.915625, "warmup_s": 18.0, "total_s": 52.915625}
        for k, v in exp.items():
            if not _close(t[k], v, 1e-9):
                return False, f"expected {k} = {v}, got {t[k]}"
        tm = sub.time_to_ready(8.0, 1.0, 2.0, "mmap", 9, 2.0)
        if not _close(tm["total_s"], 8.0 + 28.58125 + 18.0, 1e-9):
            return False, f"expected mmap total {8.0 + 28.58125 + 18.0}, got {tm['total_s']}"
        sizes = [(0.1, 52.915625, 6), (0.5, 60.0, 30), (0.0, 60.0, 0), (2.0, 1.0, 2)]
        for rate, ttr, exp_n in sizes:
            got = sub.warm_pool_size(rate, ttr)
            if got != exp_n:
                return False, f"expected a warm pool of {exp_n} for {rate}/s over {ttr} s, got {got}"
        c.metric("time_to_ready_stream_s", round(t["total_s"], 6))
        c.metric("time_to_ready_mmap_s", round(tm["total_s"], 6))
        return True, f"expected total 52.915625 s (stream) and 54.58125 s (mmap); got {t['total_s']} and {tm['total_s']}"

    c.check("cold start: pull, weight load (mmap vs streaming), warmup, and a warm pool", check_cold_start)

    try:
        ev, _ = sub.parse_events(sub.synth_event_log())
        s = sub.latency_samples(sub.request_metrics(ev))
        h = sub.build_histogram(s["ttft"], sub.LAT_BOUNDS)
        c.metric("ttft_p99_bucket_estimate", round(sub.histogram_quantile(0.99, h), 5))
        c.metric("ttft_p99_exact", round(sub.exact_quantile(s["ttft"], 0.99), 5))
    except Exception:  # noqa: BLE001
        pass  # a starter that is not finished has no metrics yet
    return c.finish()
