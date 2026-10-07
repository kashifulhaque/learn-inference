"""Chapter 17, benchmarks that mean something.

Teaches content/chapters/17-benchmarks.md in its order. Every number below
was recomputed in Python (numpy for percentiles) and matches the chapter.

The lab's first record:
- a = 10.00 s, f = 10.25 s, 11 tokens 30 ms apart: c = 10.25 + 10 x 0.03
  = 10.55 s.
- TTFT = 10.25 - 10.00 = 0.25 s. ITL = (10.55 - 10.25) / 10 = 0.030 s = 30 ms.
  E2E = 0.25 + 10 x 0.03 = 0.55 s.

What latency hides:
- 800 ms TTFT = 780 + 20 or 50 + 750 ms.
- Stall: (127 x 0.02 + 1.2) / 128 = 0.02922 s, "about 29 ms".

Throughput, the ten-request run (a_i = 0..9, TTFT 0.2 s, 11 tokens, ITL
0.05 s):
- c_9 = 9 + 0.2 + 10 x 0.05 = 9.7 s; span 9.7 s; tokens 10 x 11 = 110.
- X_tok = 110 / 9.7 = 11.34 tokens/s; X_req = 10 / 9.7 = 1.031, "1.03".

Goodput, plus one request with a 5 s TTFT: 10 / 11 = 0.909 at (1 s,
100 ms); 0 of 11 at (1 s, 1 ms).

Percentiles (linear interpolation, numpy.percentile):
- x = [0, 10], q = 25: r = 0.25, P = 2.5.
- Ten 0.2 s and one 5.0 s: P50 rank 5 -> 0.2 s; P99 rank 9.9 -> 0.2 + 0.9 x
  4.8 = 4.52 s; 4.52 / 0.2 = 22.6, "a factor of 23"; mean 7.0 / 11 = 0.636 s.
- Samples: 10 x 100 / (100 - q): 1000 for P99, 100 for P90.

Mixture: 0.95 x 30 + 0.05 x 90 = 33 ms; P50 30, P99 90; mean +10%
(3 / 30), tail +200% (60 / 30).

Little's law: 10 per minute x 3 minutes = 30. 5.3 x 6.0 = 31.8 against a
max batch of 32. Tokens: 32 / 0.030 = 1066.7, "1067"; 32 / 400 = 0.080 s,
"80 ms". Roofline check: 53.8 GB / 2 ms = 26.9 TB/s, "27 TB/s".

Repeats (c = 0.03): 1.96 c / sqrt(n) = 2.63% at 5, 1.31% at 20, 0.59% at 100
("2.6, 1.3, 0.6"). n >= 2 (1.96 c / d)^2: 17.29 at d = 2% (18 runs, and
benchmark does 20); 69.15 at d = 1% (chapter: "n >= 69").

GPU variants: 2039 / 1935 = 1.0537, "1.054", 5.4%. Sustained clocks 10 to
15% below boost (chapter figure).

Load: exponential CV = 1; uniform CV = 1 / sqrt(3) = 0.577. Log-normal,
median 128, sigma 1: mean 128 e^0.5 = 211.0; P99 128 e^2.326 = 1310.3
(the chapter prints 1311, which uses z = 2.3263); 1311 / 128 = 10.2, "ten
times the median". The drawn arrival lanes use kit.rng(1701) at 2 requests
per second, as an illustration only; the drawn density peaks at its mode,
128 / e = 47.1 tokens, where it is 0.00514 per token.

What goes wrong: n / (n - 1) - 1 = 0.79% at 128 ("0.8%"), 100% at 2.
Production engines: 2 to 5 times faster at first (chapter figure).
"""

from manim import *

import kit
from kit import theme as th

# The chapter's maths colours: arrival, first token, completion.
A_C = th.HL_A
F_C = th.HL_B
C_C = th.HL_C
QUEUE = th.INK_4

BENCH = """\
def benchmark(fn, label="fn", warmup=5, runs=20):
    for _ in range(warmup):
        fn()
    sync()

    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - start) * 1000)
"""


class Track(VGroup):
    """A horizontal time axis from `t0` to `t1`, `width` units long.

    `p(t)` reads the axis's current position, so it stays right after the
    track is placed or scaled.
    """

    def __init__(self, t0, t1, width, ticks=(), fmt="{:.1f}", size=th.SIZE_LABEL):
        super().__init__()
        self.t0, self.t1 = t0, t1
        self.axis = Line(ORIGIN, RIGHT * width, color=th.INK_3, stroke_width=th.STROKE_AXIS)
        self.add(self.axis)
        self.ticks = VGroup()
        for t in ticks:
            p = self.p(t)
            mark = Line(p + UP * 0.07, p + DOWN * 0.07, color=th.INK_3, stroke_width=th.STROKE_AXIS)
            lab = kit.label(fmt.format(t), size=size, color=th.INK_3).next_to(mark, DOWN, buff=0.1)
            self.ticks.add(VGroup(mark, lab))
        self.add(self.ticks)

    def p(self, t):
        a, b = self.axis.get_start(), self.axis.get_end()
        return a + (t - self.t0) / (self.t1 - self.t0) * (b - a)

    def span(self, t0, t1):
        return Line(self.p(t0), self.p(t1))


def seg_bar(x0, x1, y, h, color, tint=0.45):
    """A filled bar from x0 to x1 (scene units) centred at height y."""
    r = Rectangle(width=max(x1 - x0, 0.01), height=h, stroke_color=color, stroke_width=1.5,
                  fill_color=th.tint(color, tint), fill_opacity=1)
    r.move_to([(x0 + x1) / 2, y, 0])
    return r


def swatch(text, color, size=th.SIZE_LABEL):
    sq = Square(0.26, stroke_color=color, stroke_width=1.5, fill_color=th.tint(color, 0.45), fill_opacity=1)
    return VGroup(sq, kit.label(text, size=size, color=th.INK_2)).arrange(RIGHT, buff=0.15)


def marker(point, text, color, height=1.0, size=th.SIZE_LABEL, dashed=True):
    """A vertical marker rising from `point`, with a label on top."""
    cls = DashedLine if dashed else Line
    line = cls(point, point + UP * height, color=color, stroke_width=th.STROKE_AXIS + 1)
    lab = kit.math_label(text, size=size, color=color) if "\\" in text or "_" in text else kit.label(
        text, size=size, color=color, weight=MEDIUM)
    lab.next_to(line, UP, buff=0.08)
    return VGroup(line, lab)


class Lecture(kit.Lecture):
    def construct(self):
        self.title_slide(
            narration=(
                "Benchmarks that mean something. "
                "A thousand tokens per second isn't a number: under what load, at what batch size, and at what latency? "
                "In this lecture, you define each serving metric from timestamps, see what each one hides, and learn to measure without fooling yourself."
            )
        )
        self.five_timestamps()
        self.latency_metrics()
        self.latency_hides()
        self.throughput_span()
        self.goodput()
        self.percentile_rule()
        self.one_slow_request()
        self.mixture_tail()
        self.littles_law()
        self.little_check()
        self.warmup_code()
        self.sync_trap()
        self.repeats()
        self.gpu_variants()
        self.poisson_arrivals()
        self.lognormal_lengths()
        self.fixed_length()
        self.knee()
        self.what_goes_wrong()
        self.closing_slide(
            [
                r"Five values per request; ITL divides by $n - 1$ gaps.",
                r"Report throughput with latency, goodput with a tail.",
                r"$L = \lambda W$ checks any benchmark.",
                r"Warm up, synchronize, stay on one card, repeat.",
            ],
            narration=(
                "Here's what to keep. "
                "Derive every metric from five values per request, and divide by gaps, not tokens. "
                "Report throughput with its latency, and goodput with a tail percentile. "
                "Little's law catches numbers that can't all be true. "
                "Time GPU code after warmup, with a synchronize, on one card, with enough repeats. "
                "Next, quantization shrinks the weights, and these benchmarks tell you whether it helped."
            ),
            book_section="s-what-goes-wrong",
        )

    # The five timestamps -------------------------------------------------------

    def five_timestamps(self):
        self.slide("five-timestamps", "Record five values per request", book_section="s-the-five-timestamps")
        tr = Track(9.95, 10.62, 11.0, ticks=[10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6])
        unit = kit.label("time (s)", size=th.SIZE_LABEL, color=th.INK_3).next_to(tr.axis, RIGHT, buff=0.2)
        dots = VGroup(*[Dot(tr.p(10.25 + 0.03 * k), radius=0.09, color=F_C if k == 0 else C_C if k == 10
                            else th.INK_3) for k in range(11)])
        marks = VGroup()
        for t, tex, col in ((10.0, r"a = 10.00\ \mathrm{s}", A_C), (10.25, r"f = 10.25\ \mathrm{s}", F_C),
                            (10.55, r"c = 10.55\ \mathrm{s}", C_C)):
            marks.add(marker(tr.p(t), tex, col, height=0.9, dashed=False))
        tok = kit.label("11 tokens, 30 ms apart", size=th.SIZE_LABEL, color=th.INK_3)
        tok.next_to(dots, DOWN, buff=0.55).set_x(dots.get_x())
        fig = VGroup(tr, unit, dots, marks, tok)
        tab = kit.table(
            [
                [r"$a_i$", "arrival: entered the system", "timestamp", "10.00 s"],
                [r"$f_i$", "first token streamed out", "timestamp", "10.25 s"],
                [r"$c_i$", "last token streamed out", "timestamp", "10.55 s"],
                [r"$m_i$", "prompt tokens", "count", "100"],
                [r"$n_i$", "generated tokens", "count", "11"],
            ],
            header=["", "Meaning", "Kind", "Lab record"],
            align=["l", "l", "l", "r"],
            size=th.SIZE_SMALL,
            row_gap=0.18,
        )
        for r, col in ((1, A_C), (2, F_C), (3, C_C)):
            tab.cells[r][0].set_color(col)
        col = VGroup(fig, tab).arrange(DOWN, buff=0.5)
        self.place(col)
        rows = [VGroup(*tab.cells[r]) for r in range(6)]
        with self.narrate(
            "Every metric comes from five values per request. "
            "Three are timestamps: arrival, first token out, and last token out. "
            "Two are counts: prompt tokens and generated tokens. "
            "Here's the lab's first record. "
            "It arrives at 10.00 seconds, with a prompt of 100 tokens. "
            "Its first token leaves at 10.25. "
            "It generates 11 tokens, 30 ms apart, so it completes at 10.55. "
            "Keep these raw records, because no summary gives them back."
        ) as t:
            self.play(FadeIn(tab.rules), FadeIn(rows[0]), run_time=0.6)
            t.wait_until("Three are timestamps")
            self.play(LaggedStart(*[FadeIn(VGroup(*tab.cells[r][:3])) for r in (1, 2, 3)], lag_ratio=0.3),
                      run_time=1.2)
            t.wait_until("Two are counts")
            self.play(FadeIn(VGroup(*tab.cells[4][:3])), FadeIn(VGroup(*tab.cells[5][:3])), run_time=0.7)
            t.wait_until("Here's the lab's")
            self.play(Create(tr.axis), FadeIn(tr.ticks), FadeIn(unit), run_time=0.9)
            t.wait_until("It arrives at")
            self.play(FadeIn(marks[0], shift=DOWN * 0.1), FadeIn(tab.cells[1][3]), FadeIn(tab.cells[4][3]),
                      run_time=0.7)
            t.wait_until("Its first token")
            self.play(FadeIn(marks[1], shift=DOWN * 0.1), FadeIn(dots[0], scale=0.5), FadeIn(tab.cells[2][3]),
                      run_time=0.7)
            t.wait_until("It generates 11")
            self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots[1:]], lag_ratio=0.25), FadeIn(tok),
                      run_time=1.6)
            self.play(FadeIn(marks[2], shift=DOWN * 0.1), FadeIn(tab.cells[3][3]), FadeIn(tab.cells[5][3]),
                      run_time=0.7)
            t.wait_until("Keep these raw")
            self.play(Indicate(tab.rules, color=self.color, scale_factor=1.0), run_time=0.8)

    # Latency ------------------------------------------------------------------------

    def latency_metrics(self):
        self.slide("latency-metrics", "Three latencies split a request at its first token",
                   book_section="s-latency-ttft-itl-and-end-to-end")
        tr = Track(9.95, 10.62, 11.0)
        dots = VGroup(*[Dot(tr.p(10.25 + 0.03 * k), radius=0.09, color=th.INK_3) for k in range(11)])
        ends = VGroup()
        for t, tex, col in ((10.0, "a", A_C), (10.25, "f", F_C), (10.55, "c", C_C)):
            tick = Line(tr.p(t) + UP * 0.16, tr.p(t) + DOWN * 0.16, color=col, stroke_width=th.STROKE_AXIS + 2)
            lab = kit.math_label(tex, size=th.SIZE_SMALL, color=col).next_to(tick, DOWN, buff=0.12)
            ends.add(VGroup(tick, lab))
        gaps = VGroup(*[Line(tr.p(10.25 + 0.03 * k) + UP * 0.22 + RIGHT * 0.1,
                             tr.p(10.25 + 0.03 * (k + 1)) + UP * 0.22 + LEFT * 0.1,
                             color=C_C, stroke_width=5) for k in range(10)])
        b_ttft = Brace(tr.span(10.0, 10.25), UP, buff=0.45, color=F_C)
        l_ttft = VGroup(kit.math_label(r"\mathrm{TTFT} = f - a", size=th.SIZE_SMALL, color=F_C),
                        kit.math_label(r"= 0.25\ \mathrm{s}", size=th.SIZE_SMALL, color=F_C)).arrange(DOWN, buff=0.12)
        l_ttft.next_to(b_ttft, UP, buff=0.1)
        b_itl = Brace(tr.span(10.25, 10.55), UP, buff=0.45, color=C_C)
        l_itl = kit.math_label(r"\mathrm{ITL} = \frac{c - f}{n - 1} = \frac{0.30\ \mathrm{s}}{10} = 30\ \mathrm{ms}",
                               size=th.SIZE_SMALL, color=C_C)
        l_itl.next_to(b_itl, UP, buff=0.1)
        b_e2e = Brace(tr.span(10.0, 10.55), DOWN, buff=1.0, color=th.INK_2)
        l_e2e = kit.math_label(r"\mathrm{E2E} = c - a = 0.55\ \mathrm{s}", size=th.SIZE_SMALL, color=th.INK_2)
        l_e2e.next_to(b_e2e, DOWN, buff=0.1)
        fence = kit.label("11 tokens, 10 gaps", size=th.SIZE_LABEL, color=C_C)
        fence.next_to(dots, DOWN, buff=0.3).align_to(dots, RIGHT).shift(LEFT * 0.4)
        fig = VGroup(tr, dots, ends, gaps, b_ttft, l_ttft, b_itl, l_itl, b_e2e, l_e2e, fence)
        ident = kit.formula(r"\mathrm{E2E} = \mathrm{TTFT} + (n-1)\,\mathrm{ITL}", r"= 0.25 + 10 \times 0.03 = 0.55",
                            size=th.SIZE_MATH_SMALL)
        col = VGroup(fig, ident).arrange(DOWN, buff=0.6)
        self.place(col)
        with self.narrate(
                        "Time to first token is the silence before anything appears: first token minus arrival. "
            "Here it's 0.25 seconds, queueing plus prefill. "
            "Inter-token latency is the gap between tokens once the stream starts. "
            "Count the gaps like fence posts: 11 tokens leave 10 gaps. "
            "So divide the streaming time by 10, which gives 30 ms. "
            "A one-token reply has no gap, so return nothing, not zero. "
            "End-to-end latency is the whole span, 0.55 seconds. "
            "It equals the first-token time plus every gap, and your harness should assert that."
        ) as t:
            self.play(Create(tr.axis), FadeIn(dots), FadeIn(ends), run_time=0.9)
            t.wait_until("Time to first token")
            self.play(GrowFromCenter(b_ttft), run_time=0.6)
            t.wait_until("Here it's")
            self.play(FadeIn(l_ttft, shift=UP * 0.1), run_time=0.6)
            t.wait_until("Inter-token latency")
            self.play(GrowFromCenter(b_itl), run_time=0.6)
            t.wait_until("Count the gaps")
            self.play(LaggedStart(*[Create(g) for g in gaps], lag_ratio=0.2), FadeIn(fence), run_time=1.6)
            t.wait_until("So divide")
            self.play(FadeIn(l_itl, shift=UP * 0.1), run_time=0.7)
            t.wait_until("End-to-end latency")
            self.play(GrowFromCenter(b_e2e), FadeIn(l_e2e), run_time=0.8)
            t.wait_until("It equals")
            self.play(FadeIn(ident[0]), run_time=0.7)
            self.play(FadeIn(ident[1]), run_time=0.6)

    def latency_hides(self):
        self.slide("latency-hides", "Each latency metric hides something",
                   book_section="s-latency-ttft-itl-and-end-to-end")
        w, h = 6.0, 0.55
        s = w / 800

        def split(queue, prefill):
            q = seg_bar(0, queue * s, 0, h, QUEUE, tint=0.5)
            p = seg_bar(queue * s, 800 * s, 0, h, self.color, tint=0.55)
            return VGroup(q, p)

        b1, b2 = split(780, 20), split(50, 750)
        q1 = kit.label("queue 780 ms", size=th.SIZE_LABEL, color=th.INK).move_to(b1[0])
        p1 = kit.label("prefill 20 ms", size=th.SIZE_LABEL, color=self.color).next_to(b1[1], UP, buff=0.1)
        p1.align_to(b1, RIGHT)
        p2 = kit.label("prefill 750 ms", size=th.SIZE_LABEL, color=th.INK).move_to(b2[1])
        q2 = kit.label("queue 50 ms", size=th.SIZE_LABEL, color=th.INK_3).next_to(b2[0], UP, buff=0.1)
        q2.align_to(b2, LEFT)
        v1 = kit.label("a capacity problem", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        v2 = kit.label("a kernel problem", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        r1 = VGroup(VGroup(b1, q1, p1), v1).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        r2 = VGroup(VGroup(b2, q2, p2), v2).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        h_left = kit.label("TTFT of 800 ms, two ways", size=th.SIZE_SMALL, color=F_C, weight=SEMIBOLD)
        left = VGroup(h_left, r1, r2).arrange(DOWN, buff=0.45, aligned_edge=LEFT)

        gw, gh = 5.4, 2.6
        bw = gw / 128
        gaps = VGroup()
        for i in range(128):
            v = 1.2 if i == 127 else 0.02
            hh = max(v / 1.2 * gh, 0.02)
            r = Rectangle(width=bw * 0.8, height=hh, stroke_width=0,
                          fill_color=th.BAD if i == 127 else C_C, fill_opacity=0.9)
            r.move_to([i * bw + bw / 2, hh / 2, 0])
            gaps.add(r)
        base = Line([0, 0, 0], [gw, 0, 0], color=th.INK_3, stroke_width=th.STROKE_AXIS)
        mean_y = 0.02922 / 1.2 * gh
        mean = DashedLine([0, mean_y, 0], [gw, mean_y, 0], color=th.INK, stroke_width=th.STROKE_AXIS)
        mean_t = kit.label("mean ITL ≈ 29 ms", size=th.SIZE_LABEL, color=th.INK, weight=MEDIUM)
        mean_t.next_to(mean, UP, buff=0.12).align_to(mean, LEFT)
        stall = kit.label("1.2 s stall", size=th.SIZE_LABEL, color=th.BAD, weight=MEDIUM)
        stall.next_to(gaps[-1], LEFT, buff=0.15).align_to(gaps[-1], UP)
        axis_t = kit.label("127 gaps of 20 ms, then one stall", size=th.SIZE_LABEL, color=th.INK_3)
        axis_t.next_to(base, DOWN, buff=0.15).align_to(base, LEFT)
        h_right = kit.label("ITL is a mean over gaps", size=th.SIZE_SMALL, color=C_C, weight=SEMIBOLD)
        plot = VGroup(base, gaps, mean, mean_t, stall, axis_t)
        right = VGroup(h_right, plot).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        top = VGroup(left, right).arrange(RIGHT, buff=1.0, aligned_edge=UP)
        e2e = kit.label("E2E mixes speed with length: 2000 tokens take longer than 20 on the same engine.",
                        size=th.SIZE_SMALL, color=th.INK_2)
        col = VGroup(top, e2e).arrange(DOWN, buff=0.7)
        self.place(col)
        with self.narrate(
            "Each metric hides something. "
            "Time to first token adds two causes and reports neither. "
            "800 ms could be 780 of queueing and 20 of prefill: a capacity problem. "
            "Or it could be 50 of queueing and 750 of prefill: a kernel problem. "
            "Record admission time to split them. "
            "Inter-token latency is a mean, so it hides jitter. "
            "Picture 127 tokens 20 ms apart, then a stall of 1.2 seconds. "
            "The mean is about 29 ms, which looks healthy and isn't. "
            "End-to-end latency mixes speed with length, so compare the other two instead."
        ) as t:
            self.play(FadeIn(h_left), run_time=0.5)
            t.wait_until("800 ms could be")
            self.play(GrowFromEdge(b1, LEFT), FadeIn(q1), FadeIn(p1), run_time=0.8)
            self.play(FadeIn(v1), run_time=0.4)
            t.wait_until("Or it could be")
            self.play(GrowFromEdge(b2, LEFT), FadeIn(q2), FadeIn(p2), run_time=0.8)
            self.play(FadeIn(v2), run_time=0.4)
            t.wait_until("Inter-token latency is")
            self.play(FadeIn(h_right), Create(base), FadeIn(axis_t), run_time=0.6)
            t.wait_until("Picture 127")
            self.play(LaggedStart(*[GrowFromEdge(g, DOWN) for g in gaps[:-1]], lag_ratio=0.01), run_time=1.4)
            self.play(GrowFromEdge(gaps[-1], DOWN), FadeIn(stall), run_time=0.8)
            t.wait_until("The mean is about")
            self.play(Create(mean), FadeIn(mean_t), run_time=0.8)
            t.wait_until("End-to-end latency")
            self.play(FadeIn(e2e, shift=UP * 0.1), run_time=0.7)

    # Throughput and goodput -----------------------------------------------------------

    def throughput_span(self):
        self.slide("throughput-span", "Throughput divides by the wall-clock span",
                   book_section="s-throughput-and-goodput")
        tr = Track(0, 10, 7.5, ticks=range(0, 11), fmt="{:.0f}")
        rh, gap = 0.26, 0.09
        rows = VGroup()
        for i in range(10):
            y = 0.35 + (9 - i) * (rh + gap) + rh / 2
            wait = seg_bar(tr.p(i)[0], tr.p(i + 0.2)[0], y, rh, F_C)
            stream = seg_bar(tr.p(i + 0.2)[0], tr.p(i + 0.7)[0], y, rh, C_C)
            rows.add(VGroup(wait, stream))
        unit = kit.label("time (s)", size=th.SIZE_LABEL, color=th.INK_3).next_to(tr.axis, RIGHT, buff=0.2)
        brace = Brace(tr.span(0, 9.7), DOWN, buff=0.55, color=self.color)
        span_t = kit.math_label(r"\text{span} = \max_i c_i - \min_i a_i = 9.7\ \mathrm{s}", size=th.SIZE_SMALL,
                                color=self.color).next_to(brace, DOWN, buff=0.1)
        key = VGroup(swatch("waiting, 0.2 s", F_C), swatch("streaming 11 tokens", C_C)).arrange(DOWN, buff=0.15,
                                                                                                 aligned_edge=LEFT)
        fig = VGroup(tr, rows, unit, brace, span_t)
        f1 = kit.formula(r"X_{\text{tok}} = \frac{110}{9.7} = 11.34", size=th.SIZE_MATH_SMALL)
        f1u = kit.label("tokens per second", size=th.SIZE_LABEL, color=th.INK_3)
        f2 = kit.formula(r"X_{\text{req}} = \frac{10}{9.7} = 1.03", size=th.SIZE_MATH_SMALL)
        f2u = kit.label("requests per second", size=th.SIZE_LABEL, color=th.INK_3)
        hides = kit.label("Hides: all of the latency", size=th.SIZE_SMALL, color=th.BAD, weight=MEDIUM)
        side = VGroup(key, VGroup(f1, f1u).arrange(DOWN, buff=0.12, aligned_edge=LEFT),
                      VGroup(f2, f2u).arrange(DOWN, buff=0.12, aligned_edge=LEFT), hides).arrange(
            DOWN, buff=0.45, aligned_edge=LEFT)
        layout = VGroup(fig, side).arrange(RIGHT, buff=0.7)
        self.place(layout)
        with self.narrate(
            "Throughput describes the whole service. "
            "Here's the lab's ten-request run. "
            "Requests arrive one second apart. "
            "Each waits 0.2 seconds for its first token, then streams 11 tokens, 50 ms apart. "
            "Divide by that wall-clock span, not by the sum of request durations, which counts overlap twice. "
            "So 110 tokens over 9.7 seconds is 11.34 tokens per second, or 1.03 requests per second. "
            "Throughput hides all of the latency, so never report it alone."
        ) as t:
            self.play(Create(tr.axis), FadeIn(tr.ticks), FadeIn(unit), run_time=0.8)
            t.wait_until("Requests arrive")
            self.play(LaggedStart(*[GrowFromEdge(r, LEFT) for r in rows], lag_ratio=0.25), run_time=2.4)
            t.wait_until("Each waits")
            self.play(FadeIn(key), Indicate(rows[0], color=self.color, scale_factor=1.15), run_time=0.9)
            t.wait_until("Divide by that")
            self.play(GrowFromCenter(brace), FadeIn(span_t), run_time=0.8)
            t.wait_until("So 110 tokens")
            self.play(FadeIn(f1), FadeIn(f1u), run_time=0.7)
            self.play(FadeIn(f2), FadeIn(f2u), run_time=0.7)
            t.wait_until("Throughput hides")
            self.play(FadeIn(hides, shift=UP * 0.1), run_time=0.6)

    def goodput(self):
        self.slide("goodput", "Goodput counts only requests that met their targets",
                   book_section="s-throughput-and-goodput")
        f = kit.formula(r"G = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}\!\left[\mathrm{TTFT}_i \le T_{\text{ttft}}"
                        r"\;\wedge\; \mathrm{ITL}_i \le T_{\text{itl}}\right]", size=th.SIZE_MATH_SMALL)
        n = 11
        cw = 0.78
        chips = kit.grid(1, n, cell=(cw, 0.55), gap=0.06, values=[["0.2"] * 10 + ["5.0"]], size=th.SIZE_LABEL)
        chips.paint(chips.cell(0, 10), th.BAD, tint=0.25)
        chips_t = kit.label("TTFT of each request (s)", size=th.SIZE_LABEL, color=th.INK_3)
        chips_t.next_to(chips, UP, buff=0.12).align_to(chips, LEFT)

        def row(caption):
            g = kit.grid(1, n, cell=(cw, 0.4), gap=0.06)
            cap = kit.math_label(caption, size=th.SIZE_LABEL, color=th.INK_2)
            return VGroup(cap, g).arrange(DOWN, buff=0.12, aligned_edge=LEFT)

        ra = row(r"T_{\text{ttft}} = 1\ \mathrm{s},\quad T_{\text{itl}} = 100\ \mathrm{ms}")
        rb = row(r"T_{\text{ttft}} = 1\ \mathrm{s},\quad T_{\text{itl}} = 1\ \mathrm{ms}")
        ga = kit.formula(r"G = \tfrac{10}{11} = 0.909", size=th.SIZE_MATH_SMALL)
        gb = kit.formula(r"G = 0", size=th.SIZE_MATH_SMALL)
        left = VGroup(VGroup(chips_t, chips), ra, rb).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        ga.next_to(ra[1], RIGHT, buff=0.5)
        gb.next_to(rb[1], RIGHT, buff=0.5)
        gb.align_to(ga, LEFT)
        body = VGroup(left, ga, gb)
        key = VGroup(swatch("met both targets", th.GOOD), swatch("missed a target", th.BAD)).arrange(RIGHT, buff=0.6)
        col = VGroup(f, body, key).arrange(DOWN, buff=0.5)
        self.place(col)
        ga_cells = [ra[1].cell(0, c) for c in range(n)]
        gb_cells = [rb[1].cell(0, c) for c in range(n)]
        with self.narrate(
            "Goodput asks a sharper question: how many requests were served well enough? "
            "A request counts only if it meets both of its latency targets. "
            "Add one slow request that waits 5 seconds for its first token. "
            "With targets of 1 second and 100 ms, 10 of 11 pass. "
            "Goodput is 0.909. "
            "Tighten the inter-token target to 1 millisecond, and none pass. "
            "Throughput didn't change at all. "
            "Goodput can't tell a near miss from a disaster, so report it with a tail percentile."
        ) as t:
            self.play(FadeIn(chips_t), FadeIn(VGroup(*[chips.cell(0, c) for c in range(10)])), run_time=0.8)
            t.wait_until("A request counts")
            self.play(FadeIn(f), run_time=0.8)
            t.wait_until("Add one slow")
            self.play(FadeIn(chips.cell(0, 10), shift=LEFT * 0.2), run_time=0.6)
            t.wait_until("With targets of 1")
            self.play(FadeIn(ra), FadeIn(key), run_time=0.5)
            self.play(LaggedStart(*[ra[1].fill(ga_cells[c], th.GOOD if c < 10 else th.BAD, tint=0.5)
                                    for c in range(n)], lag_ratio=0.12), run_time=1.4)
            t.wait_until("Goodput is 0.909")
            self.play(FadeIn(ga), run_time=0.6)
            t.wait_until("Tighten the")
            self.play(FadeIn(rb), run_time=0.5)
            self.play(LaggedStart(*[rb[1].fill(gb_cells[c], th.BAD, tint=0.5) for c in range(n)], lag_ratio=0.1),
                      run_time=1.2)
            self.play(FadeIn(gb), run_time=0.5)

    # Percentiles ------------------------------------------------------------------------

    def percentile_rule(self):
        self.slide("percentile-rule", "A percentile interpolates between sorted samples",
                   book_section="s-percentiles")
        d = kit.formula(r"r = \frac{q}{100}(n-1),", r"\quad i = \lfloor r \rfloor,", r"\quad \phi = r - i",
                        size=th.SIZE_MATH_SMALL)
        p = kit.formula(r"P_q = x_i + \phi\,(x_{i+1} - x_i)", size=th.SIZE_MATH)
        box = SurroundingRectangle(p, color=self.color, buff=0.18, corner_radius=0.08, stroke_width=3)
        tr = Track(0, 10, 9.0, ticks=[0, 2.5, 5, 7.5, 10], fmt="{:g}")
        x0 = Dot(tr.p(0), radius=0.12, color=th.INK_2)
        x1 = Dot(tr.p(10), radius=0.12, color=th.INK_2)
        x0l = kit.math_label(r"x_0 = 0", size=th.SIZE_SMALL, color=th.INK_2).next_to(x0, UP, buff=0.2)
        x1l = kit.math_label(r"x_1 = 10", size=th.SIZE_SMALL, color=th.INK_2).next_to(x1, UP, buff=0.2)
        lo = kit.math_label(r"P_0 = \min", size=th.SIZE_LABEL, color=th.INK_3).next_to(x0l, UP, buff=0.15)
        hi = kit.math_label(r"P_{100} = \max", size=th.SIZE_LABEL, color=th.INK_3).next_to(x1l, UP, buff=0.15)
        line = VGroup(tr, x0, x1, x0l, x1l, lo, hi)
        ex = kit.formula(r"q = 25:\quad r = 0.25,\quad P_{25} = 0 + 0.25 \times 10 = 2.5", size=th.SIZE_MATH_SMALL)
        col = VGroup(d, VGroup(p, box), line, ex).arrange(DOWN, buff=0.5)
        self.place(col)
        q = ValueTracker(0)
        tip = always_redraw(lambda: Triangle(color=self.color, fill_color=self.color, fill_opacity=1, stroke_width=0)
                            .scale(0.14).rotate(PI).next_to(tr.p(q.get_value() / 10), UP, buff=0.05))
        with self.narrate(
                        "A percentile asks how slow the slow end is: 99 percent of requests come in at or under the 99th. "
            "Interpolate linearly between sorted samples. "
            "Compute a fractional rank: q over 100, times n − 1. "
            "Split it into a whole index and a fraction, then interpolate between neighbors. "
            "Using n − 1 makes the 0th percentile the minimum, and the 100th the maximum. "
            "For the samples 0 and 10, the 25th percentile is 2.5."
        ) as t:
            t.wait_until("Interpolate linearly")
            self.play(Create(tr.axis), FadeIn(tr.ticks), FadeIn(x0), FadeIn(x1), FadeIn(x0l), FadeIn(x1l),
                      run_time=0.8)
            t.wait_until("Compute a fractional")
            self.play(FadeIn(d[0]), run_time=0.7)
            t.wait_until("Split it")
            self.play(FadeIn(d[1:]), run_time=0.6)
            self.play(FadeIn(p), Create(box), run_time=0.8)
            t.wait_until("Using n")
            self.add(tip)
            self.play(FadeIn(lo), run_time=0.4)
            self.play(q.animate.set_value(100), run_time=2.0, rate_func=smooth)
            self.play(FadeIn(hi), run_time=0.4)
            t.wait_until("For the samples")
            self.play(q.animate.set_value(25), run_time=1.0)
            self.play(FadeIn(ex), run_time=0.7)

    def one_slow_request(self):
        self.slide("one-slow-request", "One slow request moves P99 23 times, and P50 not at all",
                   book_section="s-percentiles")
        vals = ["0.2"] * 10 + ["5.0"]
        g = kit.grid(1, 11, cell=(0.82, 0.55), gap=0.06, values=[vals], size=th.SIZE_LABEL)
        g.paint(g.cell(0, 10), th.BAD, tint=0.25)
        idx = VGroup(*[kit.math_label(f"x_{{{i}}}", size=th.SIZE_LABEL, color=th.INK_3).next_to(g.cell(0, i), DOWN,
                                                                                              buff=0.1)
                       for i in range(11)])
        sorted_t = kit.label("sorted TTFTs (s)", size=th.SIZE_LABEL, color=th.INK_3)
        sorted_t.next_to(g, UP, buff=0.12).align_to(g, LEFT)
        top = VGroup(sorted_t, g, idx)
        tr = Track(0, 5.5, 9.4, ticks=[0, 1, 2, 3, 4, 5], fmt="{:.0f}")
        stack = VGroup(*[Dot(tr.p(0.2) + UP * (0.1 + 0.17 * k), radius=0.075, color=th.INK_3) for k in range(10)])
        slow = Dot(tr.p(5.0) + UP * 0.1, radius=0.075, color=th.BAD)
        m50 = marker(tr.p(0.2), r"P_{50} = 0.2", F_C, height=2.1)
        m99 = marker(tr.p(4.52), r"P_{99} = 4.52", self.color, height=2.1)
        mmean = marker(tr.p(0.636), r"\text{mean } 0.636", th.INK_2, height=1.2)
        mmean[1].next_to(mmean[0], RIGHT, buff=0.1).align_to(mmean[0], UP)
        m50[1].align_to(m50[0], LEFT).shift(LEFT * 0.3)
        unit = kit.label("seconds", size=th.SIZE_LABEL, color=th.INK_3).next_to(tr.axis, RIGHT, buff=0.2)
        bottom = VGroup(tr, unit, stack, slow, m50, m99, mmean)
        need = kit.prose(r"About 10 samples beyond it: 1000 requests for P99, 100 for P90.",
                         size=th.SIZE_SMALL, color=th.INK_2, width=None)
        col = VGroup(top, bottom, need).arrange(DOWN, buff=0.5)
        self.place(col)
        p50_box = g.outline(0, 1, 5, 6, color=F_C)
        x9, x10 = g.cell(0, 9).get_center(), g.cell(0, 10).get_center()
        p99_pt = x9 + 0.9 * (x10 - x9)
        p99_tip = Triangle(color=self.color, fill_color=self.color, fill_opacity=1, stroke_width=0).scale(0.13)
        p99_tip.rotate(PI).next_to(p99_pt + UP * g.cell(0, 9).height / 2, UP, buff=0.04)
        r99 = kit.math_label(r"r = 9.9", size=th.SIZE_LABEL, color=self.color).next_to(p99_tip, RIGHT, buff=0.1)
        r50 = kit.math_label(r"r = 5", size=th.SIZE_LABEL, color=F_C).next_to(p50_box, UP, buff=0.1)
        with self.narrate(
            "Take the eleven-request run: ten waits of 0.2 seconds, and one of 5.0. "
            "Sort them. "
            "The median has rank 5, exactly on 0.2 seconds. "
            "The 99th percentile has rank 9.9: 90 percent of the way from 0.2 to 5.0, or 4.52 seconds. "
            "One bad request moved the tail by a factor of 23, and left the median alone. "
            "The mean, 0.636 seconds, describes no request at all. "
            "Tails also need samples: about 1000 requests for the 99th percentile, and 100 for the 90th."
        ) as t:
            self.play(Create(tr.axis), FadeIn(tr.ticks), FadeIn(unit),
                      LaggedStart(*[FadeIn(s, shift=DOWN * 0.2) for s in stack], lag_ratio=0.1),
                      FadeIn(slow, shift=DOWN * 0.2), run_time=1.4)
            t.wait_until("Sort them")
            self.play(FadeIn(sorted_t), FadeIn(g), FadeIn(idx), run_time=0.8)
            t.wait_until("The median has")
            self.play(Create(p50_box), FadeIn(r50), run_time=0.6)
            self.play(Create(m50[0]), FadeIn(m50[1]), run_time=0.7)
            t.wait_until("The 99th percentile has")
            self.play(FadeIn(p99_tip, shift=DOWN * 0.1), FadeIn(r99), run_time=0.6)
            self.play(Create(m99[0]), FadeIn(m99[1]), run_time=0.8)
            t.wait_until("The mean, 0.636")
            self.play(Create(mmean[0]), FadeIn(mmean[1]), run_time=0.7)
            t.wait_until("Tails also")
            self.play(FadeIn(need, shift=UP * 0.1), run_time=0.7)

    # Why p50 and p99 diverge ---------------------------------------------------------------

    def mixture_tail(self):
        self.slide("mixture-tail", "One slow step in twenty triples P99",
                   book_section="s-why-p50-and-p99-diverge-under-batching")
        ax = kit.axes([0, 120, 30], [0, 100, 25], width=6.6, height=3.9, x_numbers=[0, 30, 60, 90, 120],
                      y_numbers=[0, 25, 50, 75, 100], x_label=r"\text{step time (ms)}",
                      y_label=r"\text{share of steps (\%)}")
        rho = ValueTracker(0.0)

        def bar(x, h, color):
            p0, p1 = ax.c2p(x - 5, 0), ax.c2p(x + 5, 100 * max(h, 0.002))
            r = Rectangle(width=p1[0] - p0[0], height=p1[1] - p0[1], stroke_color=color, stroke_width=1.5,
                          fill_color=th.tint(color, 0.5), fill_opacity=1)
            return r.move_to([(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, 0])

        def mean_line():
            r = rho.get_value()
            m = (1 - r) * 30 + r * 90
            return DashedLine(ax.c2p(m, 0), ax.c2p(m, 104), color=th.INK, stroke_width=th.STROKE_AXIS)

        plot = VGroup(ax)
        rho_t = VGroup(kit.math_label(r"\rho =", size=th.SIZE_BODY, color=C_C),
                       DecimalNumber(0, num_decimal_places=3, font_size=th.SIZE_BODY * th.TEX_SCALE, color=C_C))
        rho_t.arrange(RIGHT, buff=0.15)
        e = kit.formula(r"\mathbb{E}[t] = (1-\rho)\,t_d + \rho\,t_m", size=th.SIZE_MATH_SMALL)
        ex = kit.formula(r"= 0.95 \times 30 + 0.05 \times 90 = 33\ \mathrm{ms}", size=th.SIZE_MATH_SMALL)
        tab = kit.table(
            [["mean", "30 ms", "33 ms", r"+10\%"],
             ["P50", "30 ms", "30 ms", "0"],
             ["P99", "30 ms", "90 ms", r"+200\%"]],
            header=["", r"$\rho = 0$", r"$\rho = 0.05$", "change"],
            align=["l", "r", "r", "r"], size=th.SIZE_SMALL, row_gap=0.2,
        )
        tab.cells[3][3].set_color(th.BAD)
        side = VGroup(rho_t, VGroup(e, ex).arrange(DOWN, buff=0.15, aligned_edge=LEFT), tab).arrange(
            DOWN, buff=0.5, aligned_edge=LEFT)
        layout = VGroup(plot, side).arrange(RIGHT, buff=0.8)
        self.place(layout)
        dec = rho_t[1]
        dec.add_updater(lambda m: m.set_value(rho.get_value()))
        decode = always_redraw(lambda: bar(30, 1 - rho.get_value(), F_C))
        mixed = always_redraw(lambda: bar(90, rho.get_value(), th.BAD))
        mean = always_redraw(mean_line)
        mean_t = kit.label("mean", size=th.SIZE_LABEL, color=th.INK, weight=MEDIUM)
        mean_t.add_updater(lambda m: m.next_to(mean.get_end(), UR, buff=0.05))
        lab_d = kit.math_label(r"t_d = 30", size=th.SIZE_SMALL, color=F_C).next_to(ax.c2p(35, 80), RIGHT, buff=0.15)
        lab_m = kit.math_label(r"t_m = 90", size=th.SIZE_SMALL, color=th.BAD).next_to(ax.c2p(85, 8), LEFT,
                                                                                      buff=0.15)
        p99_lo = Line(ax.c2p(30, 0), ax.c2p(30, 65), color=self.color, stroke_width=6)
        p99_hi = Line(ax.c2p(90, 0), ax.c2p(90, 65), color=self.color, stroke_width=6)
        p99_t = kit.math_label(r"P_{99}", size=th.SIZE_SMALL, color=self.color)
        p99_t.add_updater(lambda m: m.next_to(p99_lo.get_end(), UP, buff=0.1))
        with self.narrate(
            "Why does batching hurt the tail so much more than the mean? "
            "Most steps carry only decodes, and take 30 ms. "
            "Now and then, a step also carries a prefill chunk of 1024 tokens, and takes 90 ms. "
            "Let rho be the fraction of mixed steps. "
            "The mean is a weighted average, so it moves smoothly. "
            "The 99th percentile doesn't. "
            "Below 1 percent mixed steps, it stays at 30 ms. "
            "Past 1 percent, it jumps to 90. "
            "At one mixed step in twenty, the mean is 33 ms, up 10 percent. "
            "The tail is up 200 percent: a visible hitch every twentieth token."
        ) as t:
            self.play(Create(ax), run_time=0.8)
            t.wait_until("Most steps carry")
            self.play(FadeIn(decode), FadeIn(lab_d), run_time=0.6)
            t.wait_until("Now and then")
            self.add(mixed)
            self.play(FadeIn(lab_m), run_time=0.5)
            t.wait_until("Let rho be")
            self.play(FadeIn(rho_t), FadeIn(e), run_time=0.7)
            t.wait_until("The mean is a weighted")
            self.add(mean, mean_t)
            self.play(rho.animate.set_value(0.03), run_time=1.5)
            self.play(rho.animate.set_value(0.0), run_time=1.0)
            t.wait_until("The 99th percentile doesn't")
            self.play(Create(p99_lo), FadeIn(p99_t), run_time=0.6)
            t.wait_until("Below 1 percent")
            self.play(rho.animate.set_value(0.009), run_time=1.5)
            t.wait_until("Past 1 percent")
            self.play(rho.animate.set_value(0.011), p99_lo.animate.put_start_and_end_on(p99_hi.get_start(),
                                                                                       p99_hi.get_end()),
                      run_time=0.6)
            t.wait_until("At one mixed step")
            self.play(rho.animate.set_value(0.05), run_time=1.2)
            self.play(FadeIn(ex), FadeIn(tab.rules), FadeIn(VGroup(*tab.cells[0])), FadeIn(VGroup(*tab.cells[1])),
                      FadeIn(VGroup(*tab.cells[2])), run_time=0.8)
            t.wait_until("The tail is up")
            self.play(FadeIn(VGroup(*tab.cells[3])), run_time=0.6)

    # Little's law ------------------------------------------------------------------------

    def littles_law(self):
        self.slide("littles-law", "Little's law: L = λW, for any stable system", book_section="s-little-s-law")
        r = kit.rng(1717)
        n = 7
        arr = np.sort(r.uniform(0.4, 7.5, n))
        dur = r.uniform(1.6, 3.6, n)
        T = 12.0
        w = 7.6
        sx = w / T
        rh, gap = 0.2, 0.07
        top_y = 2.6
        gantt = VGroup()
        for i in range(n):
            y = top_y + (n - 1 - i) * (rh + gap)
            gantt.add(seg_bar(arr[i] * sx, (arr[i] + dur[i]) * sx, y, rh, self.color, tint=0.4))
        # n(t): requests in the system, as a step curve under the Gantt chart.
        events = sorted([(a, 1) for a in arr] + [(a + d, -1) for a, d in zip(arr, dur)])
        hy = 0.42
        pts = [np.array([0, 0, 0])]
        level = 0
        for tt, dlt in events:
            pts.append(np.array([tt * sx, level * hy, 0]))
            level += dlt
            pts.append(np.array([tt * sx, level * hy, 0]))
        pts.append(np.array([T * sx, 0, 0]))
        area = Polygon(*pts, stroke_color=th.INFO, stroke_width=3, fill_color=th.tint(th.INFO, 0.4),
                       fill_opacity=1)
        base = Line([0, 0, 0], [w, 0, 0], color=th.INK_3, stroke_width=th.STROKE_AXIS)
        nt = kit.math_label(r"n(t)", size=th.SIZE_SMALL, color=th.INFO).next_to(base, LEFT, buff=0.15).shift(UP * 0.4)
        win = kit.math_label(r"0 \le t \le T,\ \text{empty at both ends}", size=th.SIZE_LABEL, color=th.INK_3)
        win.next_to(base, DOWN, buff=0.15).align_to(base, LEFT)
        req_t = kit.label("each bar: one request, in the system for W", size=th.SIZE_LABEL, color=th.INK_3)
        req_t.next_to(gantt, UP, buff=0.12).align_to(base, LEFT)
        fig = VGroup(gantt, req_t, area, base, nt, win)
        law = kit.formula(r"L = \lambda\, W", size=th.SIZE_MATH + 8)
        law_box = SurroundingRectangle(law, color=self.color, buff=0.2, corner_radius=0.08, stroke_width=3)
        cafe = kit.prose(r"10 customers a minute, 3 minutes each: 30 inside.", size=th.SIZE_SMALL, width=4.6,
                         color=th.INK_2)
        proof = kit.Derivation(
            [r"\int_0^T n(t)\,dt &= \sum_{i=1}^{N} W_i",
             r"\tfrac{1}{T}\textstyle\int_0^T n\,dt &= \tfrac{N}{T}\cdot\tfrac{1}{N}\sum_i W_i",
             r"L &= \lambda\, W"],
            size=th.SIZE_MATH_SMALL,
        )
        side = VGroup(cafe, VGroup(law, law_box), proof).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        layout = VGroup(fig, side).arrange(RIGHT, buff=0.7)
        self.place(layout)
        with self.narrate(
            "Little's law gives you a free consistency check. "
            "If 10 customers enter a coffee shop each minute, and each stays 3 minutes, about 30 are inside. "
                        "The mean number in the system, L, equals the arrival rate, lambda, times the mean time in the system, W. "
            "The proof is an area. "
            "Count the requests in flight at each moment, and add up that curve over the window. "
            "Each request contributes exactly its own time. "
            "Divide by the window, and you get L equals lambda W, for any arrivals and any service."
        ) as t:
            t.wait_until("If 10 customers")
            self.play(FadeIn(cafe), run_time=0.7)
            t.wait_until("The mean number")
            self.play(FadeIn(law), Create(law_box), run_time=0.8)
            t.wait_until("The proof is")
            self.play(FadeIn(req_t), LaggedStart(*[GrowFromEdge(b, LEFT) for b in gantt], lag_ratio=0.2),
                      Create(base), run_time=1.6)
            t.wait_until("Count the requests")
            self.play(DrawBorderThenFill(area), FadeIn(nt), FadeIn(win), run_time=1.6)
            t.wait_until("Each request contributes")
            self.play(proof.show(0), Indicate(gantt, color=th.INFO, scale_factor=1.03), run_time=1.0)
            t.wait_until("Divide by the window")
            self.play(proof.show(1), run_time=0.9)
            self.play(proof.show(2), run_time=0.7)

    def little_check(self):
        self.slide("little-check", "Two of the three numbers fix the third", book_section="s-little-s-law")
        f1 = kit.formula(r"L = \lambda W = 5.3 \times 6.0 = 31.8", size=th.SIZE_MATH_SMALL)
        b1 = kit.bar_compare([("requests resident", 31.8, self.color, "31.8")], limit=32,
                             limit_label="max batch 32", max_value=36, width=6.0, bar_height=0.5)
        v1 = kit.label("At the batch limit: a larger batch or a second GPU, not a faster kernel", size=th.SIZE_SMALL,
                       color=th.INK_2)
        part1 = VGroup(f1, b1, v1).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        f2 = kit.formula(r"X_{\text{tok}} = \frac{L}{\mathrm{ITL}} = \frac{32}{0.030\ \mathrm{s}} = 1067",
                         size=th.SIZE_MATH_SMALL)
        b2 = kit.bar_compare([("predicted", 1067, th.INFO, "1067 tok/s"), ("measured", 400, self.color, "400 tok/s")],
                             max_value=1300, width=6.0, bar_height=0.45, gap=0.25)
        f3 = kit.formula(r"\mathrm{ITL} = \frac{32}{400} = 80\ \mathrm{ms}", size=th.SIZE_MATH_SMALL,
                         color=self.color)
        part2 = VGroup(f2, b2, f3).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        col = VGroup(part1, part2).arrange(DOWN, buff=0.6, aligned_edge=LEFT)
        self.place(col)
        with self.narrate(
            "Two of the three quantities fix the third, so you can check any report. "
            "Suppose your engine sustains 5.3 requests per second at a mean latency of 6.0 seconds. "
            "Then 31.8 requests are resident on average. "
            "The scheduler's default maximum batch is 32, so you're at the limit. "
            "The fix is a larger batch or a second GPU, not a faster kernel. "
            "For tokens, each running sequence emits one token per gap. "
            "At batch 32 and 30 ms, that's 1067 tokens per second. "
            "Measure 400, and your gap must be 80 ms."
        ) as t:
            t.wait_until("Suppose your engine")
            self.play(FadeIn(f1), run_time=0.8)
            t.wait_until("Then 31.8")
            self.play(FadeIn(b1.labels), *b1.grow(run_time=0.9), FadeIn(b1.values), run_time=0.9)
            t.wait_until("The scheduler's")
            self.play(FadeIn(b1.limit), run_time=0.6)
            t.wait_until("The fix is")
            self.play(FadeIn(v1), run_time=0.6)
            t.wait_until("At batch 32")
            self.play(FadeIn(f2), FadeIn(b2.labels[0]), GrowFromEdge(b2.bars[0], LEFT), FadeIn(b2.values[0]),
                      run_time=1.0)
            t.wait_until("Measure 400")
            self.play(FadeIn(b2.labels[1]), GrowFromEdge(b2.bars[1], LEFT), FadeIn(b2.values[1]), run_time=0.8)
            self.play(FadeIn(f3), run_time=0.6)

    # Timing GPU code -----------------------------------------------------------------------

    def warmup_code(self):
        self.slide("warmup-code", "Throw away five runs, then time twenty", book_section="s-timing-gpu-code")
        panel = kit.code(BENCH, "python", caption="engine/bench.py")
        self.place(panel)
        warm = panel.lines(2, 3)
        timed = panel.lines(7, 8, 9, 10)
        b_warm = Brace(warm, RIGHT, buff=0.3, color=th.INK_3)
        t_warm = VGroup(
            kit.label("discarded: first calls", size=th.SIZE_LABEL, color=th.INK_2, weight=MEDIUM),
            kit.label("compile, autotune,", size=th.SIZE_LABEL, color=th.INK_3),
            kit.label("allocate, warm caches", size=th.SIZE_LABEL, color=th.INK_3),
        ).arrange(DOWN, buff=0.08, aligned_edge=LEFT).next_to(b_warm, RIGHT, buff=0.15)
        t_warm.align_to(panel.line(2), UP)
        b_time = Brace(panel.lines(7, 8, 9), RIGHT, buff=0.3, color=self.color)
        t_time = kit.label("timed: 20 samples", size=th.SIZE_LABEL, color=self.color, weight=MEDIUM)
        t_time.next_to(b_time, RIGHT, buff=0.15)
        with self.narrate(
            "Now the clock. "
            "Four things happen on the first call and never again. "
            "Kernels compile, Triton and cuBLAS autotune, the allocator asks the driver for memory, and the caches are cold. "
            "A first call can take hundreds of milliseconds, against a steady state under one. "
            "So the benchmark discards five runs of the exact call, then times twenty. "
            "Each timed run ends with a synchronize, so the clock includes everything a user waits for."
        ) as t:
            self.play(FadeIn(panel.frame), *panel.reveal(1), run_time=0.7)
            t.wait_until("Four things")
            self.play(*panel.reveal(2, 3, 4), panel.highlight(2, 3), run_time=0.8)
            t.wait_until("Kernels compile")
            self.play(GrowFromCenter(b_warm), FadeIn(t_warm), run_time=0.8)
            t.wait_until("So the benchmark")
            self.play(*panel.reveal(6, 7, 8, 9, 10, 11), run_time=0.9)
            self.play(GrowFromCenter(b_time), FadeIn(t_time), run_time=0.7)
            t.wait_until("Each timed run")
            self.play(panel.highlight(4, 10, color=th.tint(th.GOOD, 0.18)), run_time=0.7)

    def sync_trap(self):
        self.slide("sync-trap", "Without a synchronize, you time the queue", book_section="s-timing-gpu-code")
        w = 10.0
        lane_h = 0.6
        host_l = kit.label("host", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        gpu_l = kit.label("GPU stream", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        host = Line([0, 0, 0], [w, 0, 0], color=th.RULE_STRONG, stroke_width=th.STROKE_AXIS)
        gpu = Line([0, -1.4, 0], [w, -1.4, 0], color=th.RULE_STRONG, stroke_width=th.STROKE_AXIS)
        host_l.next_to(host, LEFT, buff=0.3)
        gpu_l.next_to(gpu, LEFT, buff=0.3)
        launches = VGroup(*[seg_bar(0.1 + 0.18 * k, 0.22 + 0.18 * k, 0.32, lane_h * 0.8, th.INK_3, tint=0.3)
                            for k in range(6)])
        widths = [1.2, 1.9, 0.9, 1.6, 1.3, 1.5]
        kernels = VGroup()
        x = 0.4
        for k, kw in enumerate(widths):
            kernels.add(seg_bar(x, x + kw - 0.06, -1.4 + 0.32, lane_h * 0.8, self.colors[1], tint=0.35))
            x += kw
        gpu_end = x - 0.06
        k_t = kit.label("kernels run later, in order", size=th.SIZE_LABEL, color=th.INK_3)
        k_t.next_to(kernels, DOWN, buff=0.12).align_to(kernels, LEFT)
        l_t = kit.label("launches return at once", size=th.SIZE_LABEL, color=th.INK_3)
        l_t.next_to(launches, RIGHT, buff=0.25)
        no_sync = VGroup(Line([0.1, 1.05, 0], [1.12, 1.05, 0], color=th.BAD, stroke_width=6))
        no_sync.add(kit.label("no synchronize: a few microseconds", size=th.SIZE_LABEL, color=th.BAD,
                              weight=MEDIUM).next_to(no_sync[0], UP, buff=0.1).align_to(no_sync[0], LEFT))
        sync_bar = Line([0.1, -2.55, 0], [gpu_end, -2.55, 0], color=th.GOOD, stroke_width=6)
        sync_t = kit.label("synchronize: the clock stops when the GPU is done", size=th.SIZE_LABEL, color=th.GOOD,
                           weight=MEDIUM).next_to(sync_bar, DOWN, buff=0.12).align_to(sync_bar, LEFT)
        ev = VGroup()
        for xx in (0.4, gpu_end):
            flag = VGroup(Line([xx, -1.4, 0], [xx, -0.7, 0], color=th.VIOLET, stroke_width=3),
                          Triangle(color=th.VIOLET, fill_color=th.VIOLET, fill_opacity=1, stroke_width=0)
                          .scale(0.1).rotate(-PI / 2).move_to([xx + 0.1, -0.76, 0]))
            ev.add(flag)
        ev_t = kit.label("CUDA events: device timestamps", size=th.SIZE_LABEL, color=th.VIOLET)
        ev_t.move_to([0, -0.5, 0]).align_to(ev[1], RIGHT)
        fig = VGroup(host, gpu, host_l, gpu_l, launches, kernels, k_t, l_t, no_sync, sync_bar, sync_t, ev, ev_t)
        check = kit.callout("warning", r"53.8 GB moved in 2 ms would be 27 TB/s, far above the roofline floor. "
                                       r"The timing is wrong, not the kernel fast.", title="Sanity check",
                            width=11.5, size=th.SIZE_SMALL)
        col = VGroup(fig, check).arrange(DOWN, buff=0.5)
        self.place(col)
        with self.narrate(
            "A kernel launch returns to the host almost at once. "
            "The work runs later, on the GPU's stream. "
            "Without a synchronize, the clock measures only the queueing: a few microseconds, and a bug. "
            "With a synchronize, the clock stops when the GPU is done. "
            "CUDA events time the device itself, but you still wait for the end event. "
            "Check every result against the roofline: a kernel that moves 53.8 GB in 2 ms would need 27 TB/s, so the timing is wrong."
        ) as t:
            self.play(Create(host), Create(gpu), FadeIn(host_l), FadeIn(gpu_l), run_time=0.7)
            self.play(LaggedStart(*[FadeIn(lc, shift=DOWN * 0.1) for lc in launches], lag_ratio=0.2), FadeIn(l_t),
                      run_time=1.0)
            t.wait_until("The work runs")
            self.play(LaggedStart(*[GrowFromEdge(k, LEFT) for k in kernels], lag_ratio=0.5), FadeIn(k_t),
                      run_time=2.0)
            t.wait_until("Without a synchronize")
            self.play(Create(no_sync[0]), FadeIn(no_sync[1]), run_time=0.8)
            t.wait_until("With a synchronize")
            self.play(Create(sync_bar), FadeIn(sync_t), run_time=1.2)
            t.wait_until("CUDA events")
            self.play(FadeIn(ev), FadeIn(ev_t), run_time=0.8)
            t.wait_until("Check every result")
            self.play(FadeIn(check, shift=UP * 0.1), run_time=0.8)

    # Repeats --------------------------------------------------------------------------------

    def repeats(self):
        self.slide("repeats", "Twenty runs resolve about a 2% difference",
                   book_section="s-how-many-repeats-a-claim-needs")
        ax = kit.axes([0, 100, 10], [0, 3, 0.5], width=6.4, height=3.8, x_numbers=[0, 20, 40, 60, 80, 100],
                      y_numbers=[0, 1, 2, 3], x_label=r"\text{runs } n", y_label=r"\text{half-width (\%)}")
        curve = ax.plot(lambda n: 1.96 * 3 / np.sqrt(n), x_range=[4, 100], color=self.color,
                        stroke_width=th.STROKE_CURVE)
        pts = VGroup()
        for n, txt in ((5, "2.6\\%"), (20, "1.3\\%"), (100, "0.6\\%")):
            y = 1.96 * 3 / np.sqrt(n)
            d = Dot(ax.c2p(n, y), radius=0.09, color=self.color)
            lab = kit.math_label(txt, size=th.SIZE_LABEL, color=th.INK)
            lab.next_to(d, UR if n < 100 else UP, buff=0.1)
            pts.add(VGroup(d, lab))
        plot = VGroup(ax, curve, pts)
        hw = kit.formula(r"\frac{1.96\,c}{\sqrt{n}}", r",\quad c = 0.03", size=th.SIZE_MATH_SMALL)
        two = kit.formula(r"n \ge 2\left(\frac{1.96\,c}{d}\right)^{2}", size=th.SIZE_MATH)
        box = SurroundingRectangle(two, color=self.color, buff=0.18, corner_radius=0.08, stroke_width=3)
        cases = VGroup(
            kit.prose(r"$d = 2\%$: $n \ge 17.3$, so 18 runs", size=th.SIZE_SMALL, width=None),
            kit.prose(r"$d = 1\%$: $n \ge 69$", size=th.SIZE_SMALL, width=None),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        caveat = kit.prose(r"Within one process. Across processes, no repeat count helps.", size=th.SIZE_SMALL,
                           color=th.BAD, width=5.4)
        side = VGroup(hw, VGroup(two, box), cases, caveat).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        layout = VGroup(plot, side).arrange(RIGHT, buff=0.7)
        self.place(layout)
        with self.narrate(
            "How many runs does a claim need? "
            "Steady-state GPU timings spread by about 3 percent of the mean. "
            "The 95 percent confidence interval shrinks with the square root of the run count. "
            "At 5 runs, it's 2.6 percent on each side. "
            "At 20, it's 1.3 percent, and at 100, 0.6. "
            "Comparing two configurations is harder, because both carry error. "
            "A 2 percent difference needs 18 runs, and the benchmark does 20. "
            "A 1 percent difference needs about 69. "
            "And this only covers noise within one process."
        ) as t:
            self.play(Create(ax), run_time=0.8)
            t.wait_until("Steady-state")
            self.play(FadeIn(hw), run_time=0.7)
            t.wait_until("The 95 percent")
            self.play(Create(curve), run_time=1.4)
            for i, cue in enumerate(["At 5 runs", "At 20, it's"]):
                t.wait_until(cue)
                self.play(FadeIn(pts[i], scale=0.6), run_time=0.5)
            self.play(FadeIn(pts[2], scale=0.6), run_time=0.5)
            t.wait_until("Comparing two")
            self.play(FadeIn(two), Create(box), run_time=0.8)
            t.wait_until("A 2 percent")
            self.play(FadeIn(cases[0]), run_time=0.6)
            t.wait_until("A 1 percent")
            self.play(FadeIn(cases[1]), run_time=0.6)
            t.wait_until("And this only")
            self.play(FadeIn(caveat), run_time=0.6)

    def gpu_variants(self):
        self.slide("gpu-variants", "Two A100s differ by more than your speedup",
                   book_section="s-comparing-across-gpu-variants-invalidates-the-comparison")
        bw = kit.bar_compare([("A100 80GB SXM4", 2039, self.colors[1], "2039 GB/s"),
                              ("A100 80GB PCIe", 1935, self.colors[1], "1935 GB/s")],
                             width=6.5, bar_height=0.5, gap=0.3)
        ratio = kit.formula(r"\frac{t_{\text{PCIe}}}{t_{\text{SXM4}}} = \frac{2039}{1935} = 1.054",
                            size=th.SIZE_MATH_SMALL)
        top = VGroup(bw, ratio).arrange(RIGHT, buff=0.8)
        gap = kit.bar_compare([("silicon alone", 5.4, th.BAD, "5.4%"),
                               ("what 20 runs resolve", 2.0, self.color, "2%")],
                              width=6.5, bar_height=0.5, gap=0.3)
        clocks = kit.table(
            [["10-second run", "boost clocks", "one request on an idle server"],
             ["10-minute run", r"10 to 15\% lower", "a loaded server all day"]],
            header=["Benchmark", "Clock", "Answers"], size=th.SIZE_SMALL, row_gap=0.2,
        )
        col = VGroup(top, gap, clocks).arrange(DOWN, buff=0.6)
        self.place(col)
        with self.narrate(
            "A perfect measurement still fails across different hardware. "
            "The 80 GB A100 ships in two forms. "
            "The SXM4 module is rated at 2039 GB/s, and the PCIe card at 1935. "
            "Decode kernels are memory bound, so their time scales inversely with bandwidth. "
            "That's a 5.4 percent gap from silicon alone, against the 2 percent that 20 runs resolve. "
            "So record the GPU with every result. "
            "Clocks matter too: a short benchmark runs at boost clocks, and a run of 10 minutes runs 10 to 15 percent lower."
        ) as t:
            t.wait_until("The SXM4")
            self.play(FadeIn(bw.labels), *bw.grow(run_time=1.0), FadeIn(bw.values), run_time=1.0)
            t.wait_until("Decode kernels")
            self.play(FadeIn(ratio), run_time=0.8)
            t.wait_until("That's a 5.4")
            self.play(FadeIn(gap.labels[0]), GrowFromEdge(gap.bars[0], LEFT), FadeIn(gap.values[0]), run_time=0.8)
            self.play(FadeIn(gap.labels[1]), GrowFromEdge(gap.bars[1], LEFT), FadeIn(gap.values[1]), run_time=0.8)
            t.wait_until("Clocks matter")
            self.play(FadeIn(clocks.rules), FadeIn(VGroup(*clocks.cells[0])), FadeIn(VGroup(*clocks.cells[1])),
                      run_time=0.7)
            self.play(FadeIn(VGroup(*clocks.cells[2])), run_time=0.6)

    # Load generation -----------------------------------------------------------------------

    def poisson_arrivals(self):
        self.slide("poisson-arrivals", "Real users arrive in random clumps",
                   book_section="s-what-a-load-generator-must-model")
        r = kit.rng(1701)
        lam, T, w = 2.0, 10.0, 8.6
        sx = w / T

        def lane(times, color, name):
            base = Line([0, 0, 0], [w, 0, 0], color=th.RULE_STRONG, stroke_width=th.STROKE_AXIS)
            ticks = VGroup(*[Line([tt * sx, 0, 0], [tt * sx, 0.5, 0], color=color, stroke_width=3)
                             for tt in times])
            lab = kit.label(name, size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
            lab.next_to(base, UP, buff=0.62).align_to(base, LEFT)
            return VGroup(base, ticks, lab)

        def arrivals(gap_fn):
            out, now = [], 0.0
            while now < T:
                out.append(now)
                now += gap_fn()
            return out

        burst = lane([0.0] * 20, th.INK_3, "Everything at once: a burst, not a service")
        burst[1].set_stroke(width=7)
        burst_n = kit.label("every request at t = 0", size=th.SIZE_LABEL, color=th.INK_3)
        burst_n.next_to(burst[1], RIGHT, buff=0.2)
        burst.add(burst_n)
        uni = lane(arrivals(lambda: r.uniform(0, 2 / lam)), th.INFO, "Uniform gaps")
        poi = lane(arrivals(lambda: -np.log(r.random()) / lam), self.color, "Poisson: exponential gaps")
        lanes = VGroup(burst, uni, poi).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        cv_u = kit.math_label(r"\mathrm{CV} = 1/\sqrt{3} = 0.577", size=th.SIZE_SMALL, color=th.INFO)
        cv_p = kit.math_label(r"\mathrm{CV} = 1", size=th.SIZE_SMALL, color=self.color)
        cv_u.next_to(uni[0], RIGHT, buff=0.4)
        cv_p.next_to(poi[0], RIGHT, buff=0.4)
        cv_p.align_to(cv_u, LEFT)
        fig = VGroup(lanes, cv_u, cv_p)
        f1 = kit.formula(r"F(t) = 1 - e^{-\lambda t}", size=th.SIZE_MATH_SMALL)
        f2 = kit.formula(r"\Delta = -\frac{\ln U}{\lambda}", size=th.SIZE_MATH)
        box = SurroundingRectangle(f2, color=self.color, buff=0.18, corner_radius=0.08, stroke_width=3)
        maths = VGroup(f1, VGroup(f2, box)).arrange(RIGHT, buff=1.2)
        col = VGroup(fig, maths).arrange(DOWN, buff=0.6)
        self.place(col)
        with self.narrate(
            "Sending a hundred requests at once measures a burst, not a service. "
            "Queueing only appears when arrivals spread out over time. "
            "Real users don't coordinate, so arrivals come in random clumps: mostly short gaps, and the occasional long one. "
            "That's a Poisson process, with exponential gaps. "
            "Sample each gap as minus the log of a uniform number, over the rate. "
                        "As a self-check, exponential gaps have a coefficient of variation of exactly 1. "
            "Uniform gaps give 0.577, and the lab's test separates them on that number."
        ) as t:
            self.play(FadeIn(burst), run_time=0.7)
            t.wait_until("Real users")
            self.play(FadeIn(poi[0]), FadeIn(poi[2]), run_time=0.4)
            self.play(LaggedStart(*[Create(tk) for tk in poi[1]], lag_ratio=0.15), run_time=2.0)
            t.wait_until("That's a Poisson")
            self.play(FadeIn(f1), run_time=0.7)
            t.wait_until("Sample each gap")
            self.play(FadeIn(f2), Create(box), run_time=0.8)
            t.wait_until("As a self-check")
            self.play(FadeIn(cv_p), run_time=0.5)
            t.wait_until("Uniform gaps give")
            self.play(FadeIn(uni[0]), FadeIn(uni[2]), LaggedStart(*[Create(tk) for tk in uni[1]], lag_ratio=0.1),
                      FadeIn(cv_u), run_time=1.5)

    def lognormal_lengths(self):
        self.slide("lognormal-lengths", "Output lengths: mostly short, a few very long",
                   book_section="s-what-a-load-generator-must-model")
        mu, sigma = np.log(128), 1.0

        def pdf(x):
            if x <= 0:
                return 0.0
            return np.exp(-(np.log(x) - mu) ** 2 / (2 * sigma ** 2)) / (x * sigma * np.sqrt(2 * np.pi))

        ax = kit.axes([0, 1500, 250], [0, 0.006, 0.002], width=8.6, height=3.6,
                      x_numbers=[0, 250, 500, 750, 1000, 1250, 1500], x_label=r"\text{generated tokens } n",
                      show_y_axis=False)
        d = kit.density(ax, pdf, x_range=[1, 1500], color=self.color)
        ms = VGroup(
            marker(ax.c2p(128, 0), r"\text{median } 128", th.INFO, height=3.4),
            marker(ax.c2p(211, 0), r"\text{mean } 211", th.INK_2, height=2.2),
            marker(ax.c2p(1311, 0), r"P_{99} = 1311", th.BAD, height=1.4),
        )
        ms[1][1].next_to(ms[1][0], RIGHT, buff=0.1).align_to(ms[1][0], UP)
        plot = VGroup(ax, d, ms)
        f = kit.formula(r"\text{median} = e^{\mu}, \quad \mathbb{E}[n] = e^{\mu + \sigma^2/2}, \quad"
                        r" P_{99} = e^{\mu + 2.326\,\sigma}", size=th.SIZE_MATH_SMALL)
        note = kit.prose(r"Median 128, $\sigma = 1$: one request in a hundred is ten times the median.",
                         size=th.SIZE_SMALL, width=None, color=th.INK_2)
        col = VGroup(f, plot, note).arrange(DOWN, buff=0.5)
        self.place(col)
        with self.narrate(
            "Lengths matter too. "
            "Output length decides how long a sequence holds its batch slot. "
            "A log-normal is a reasonable default. "
            "Take a median of 128 tokens and a sigma of 1. "
            "The mean is 211, pulled up by the long tail. "
            "The 99th percentile is 1311 tokens. "
            "One request in a hundred runs ten times the median, and that's why continuous batching exists."
        ) as t:
            t.wait_until("A log-normal")
            self.play(Create(ax), FadeIn(f), run_time=0.9)
            t.wait_until("Take a median")
            self.play(Create(d.curve), FadeIn(d.area), run_time=1.4)
            self.play(Create(ms[0][0]), FadeIn(ms[0][1]), run_time=0.6)
            t.wait_until("The mean is 211")
            self.play(Create(ms[1][0]), FadeIn(ms[1][1]), run_time=0.6)
            t.wait_until("The 99th percentile")
            self.play(Create(ms[2][0]), FadeIn(ms[2][1]), run_time=0.7)
            t.wait_until("One request in")
            self.play(FadeIn(note), run_time=0.6)

    def fixed_length(self):
        self.slide("fixed-length", "Fixed lengths hide four hard problems",
                   book_section="s-why-a-fixed-length-benchmark-flatters-an-engine")
        lens = [5, 12, 7, 9]
        steps = 12
        cell = (0.34, 0.36)
        g1 = kit.grid(4, steps, cell=cell, gap=0.04)
        g2 = kit.grid(4, 8, cell=cell, gap=0.04)
        t1 = kit.label("Varied lengths: static batch waits for the longest", size=th.SIZE_LABEL, color=th.INK_2,
                       weight=MEDIUM)
        t2 = kit.label("Fixed lengths: nothing to wait for", size=th.SIZE_LABEL, color=th.INK_2, weight=MEDIUM)
        p1 = VGroup(t1, g1).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        p2 = VGroup(t2, g2).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        key = VGroup(swatch("working", self.color), swatch("idle slot", th.INK_4)).arrange(DOWN, buff=0.15,
                                                                                       aligned_edge=LEFT)
        top = VGroup(p1, p2, key).arrange(RIGHT, buff=0.6, aligned_edge=DOWN)
        items = kit.bullets(
            [r"\textbf{Static batching stops looking bad}: every member finishes on the same step.",
             r"\textbf{Cache fragmentation disappears}: blocks are claimed and freed together.",
             r"\textbf{The scheduler never preempts}: cache pressure is constant.",
             r"\textbf{The tail collapses}: P99 converges on P50."],
            size=th.SIZE_SMALL, width=12.4, buff=0.2,
        )
        col = VGroup(top, items).arrange(DOWN, buff=0.55)
        self.place(col)
        work1 = VGroup(*[g1.cell(r, c) for r in range(4) for c in range(lens[r])])
        idle1 = VGroup(*[g1.cell(r, c) for r in range(4) for c in range(lens[r], steps)])
        with self.narrate(
            "Now see what you lose by skipping realistic load. "
            "Fix every prompt at 512 tokens, and every output at 128. "
            "A static batch runs until its longest member finishes, so varied lengths leave idle slots. "
            "With identical lengths, there's nothing to wait for, and static batching stops looking bad. "
            "Cache fragmentation disappears. "
            "The scheduler never preempts. "
            "And the tail collapses onto the median. "
            "That's a fine kernel benchmark, but it isn't a serving benchmark."
        ) as t:
            self.play(FadeIn(t1), FadeIn(g1), FadeIn(key), run_time=0.7)
            t.wait_until("A static batch")
            self.play(g1.fill(work1, self.color, tint=0.5), run_time=0.8)
            self.play(g1.fill(idle1, th.INK_4, tint=0.15), run_time=0.8)
            t.wait_until("With identical")
            self.play(FadeIn(t2), FadeIn(g2), run_time=0.5)
            self.play(g2.fill(VGroup(*g2.squares()), self.color, tint=0.5), FadeIn(items[0]), run_time=0.9)
            for i, cue in enumerate(["Cache fragmentation", "The scheduler never", "And the tail"]):
                t.wait_until(cue)
                self.play(FadeIn(items[i + 1], shift=RIGHT * 0.1), run_time=0.6)

    # The latency-throughput curve ----------------------------------------------------------

    def knee(self):
        self.slide("knee", "Admit requests below the knee", book_section="s-the-latency-throughput-curve")
        ax1 = kit.axes([0, 1.1, 0.25], [0, 1, 0.25], width=5.6, height=3.6, x_label=r"\text{throughput}",
                       y_label=r"\text{P99 TTFT}")
        ax2 = kit.axes([0, 1.4, 0.25], [0, 1, 0.25], width=5.0, height=3.6, x_label=r"\text{request rate } \lambda",
                       y_label=r"\text{goodput}")

        def thr(lam):  # throughput ceiling at 1
            return lam if lam < 0.85 else 0.85 + 0.15 * (1 - np.exp(-(lam - 0.85) / 0.15))

        def lat(lam):
            return min(0.12 + 0.012 / max(1.0 - lam, 0.012), 1.0)

        def path(upto):
            ls = np.linspace(0, max(upto, 0.01), 160)
            v = VMobject(color=self.color, stroke_width=th.STROKE_CURVE)
            v.set_points_smoothly([ax1.c2p(thr(l), lat(l)) for l in ls])
            return v

        def good(lam):
            ok = 1 / (1 + np.exp((lam - 0.95) / 0.05))
            return lam * ok

        gl = np.linspace(0, 1.35, 200)
        gcurve = VMobject(color=th.GOOD, stroke_width=th.STROKE_CURVE)
        gcurve.set_points_smoothly([ax2.c2p(l, good(l) / 0.95) for l in gl])
        peak = gl[int(np.argmax([good(l) for l in gl]))]
        gdot = Dot(ax2.c2p(peak, good(peak) / 0.95), radius=0.1, color=th.GOOD)
        gpeak = kit.label("admit here", size=th.SIZE_LABEL, color=th.GOOD, weight=MEDIUM).next_to(gdot, UP, buff=0.15)
        regions = VGroup(
            kit.label("below the knee: flat", size=th.SIZE_LABEL, color=th.INK_3).move_to(ax1.c2p(0.38, 0.42)),
            kit.label("the knee", size=th.SIZE_LABEL, color=self.color, weight=MEDIUM).next_to(
                ax1.c2p(thr(0.93), lat(0.93)), RIGHT, buff=0.25),
            kit.label("past it: unbounded", size=th.SIZE_LABEL, color=th.BAD).next_to(ax1.c2p(thr(0.99), 0.9), LEFT,
                                                                                       buff=0.25),
        )
        left = VGroup(ax1, regions)
        right = VGroup(ax2, gcurve, gdot, gpeak)
        layout = VGroup(left, right).arrange(RIGHT, buff=1.1)
        cap = kit.label("Schematic: sweep the rate, record throughput and P99 time to first token at each",
                        size=th.SIZE_LABEL, color=th.INK_3)
        col = VGroup(layout, cap).arrange(DOWN, buff=0.45)
        self.place(col)
        lam = ValueTracker(0.0)
        dot = always_redraw(lambda: Dot(ax1.c2p(thr(lam.get_value()), lat(lam.get_value())), radius=0.1,
                                        color=self.color))
        curve = always_redraw(lambda: path(lam.get_value()))
        with self.narrate(
                        "Sweep the request rate, and record throughput and tail time to first token at each rate. "
            "Below the knee, the GPU is nearly idle, so throughput rises and latency stays flat. "
            "At the knee, bursts form a queue that drains between them. "
            "That's the operating point you want. "
            "Past the knee, the queue never drains, throughput plateaus, and latency grows without bound. "
            "Goodput finds the knee cleanly. "
            "Plotted against the rate, it peaks, and that peak is the rate to admit at."
        ) as t:
            self.play(Create(ax1), FadeIn(cap), run_time=0.8)
            t.wait_until("Below the knee")
            self.add(curve, dot)
            self.play(lam.animate.set_value(0.8), FadeIn(regions[0]), run_time=2.2)
            t.wait_until("At the knee")
            self.play(lam.animate.set_value(0.93), FadeIn(regions[1]), run_time=1.2)
            t.wait_until("Past the knee")
            self.play(lam.animate.set_value(0.995), FadeIn(regions[2]), run_time=1.6)
            t.wait_until("Goodput finds")
            self.play(Create(ax2), Create(gcurve), run_time=1.4)
            t.wait_until("Plotted against")
            self.play(FadeIn(gdot, scale=0.5), FadeIn(gpeak), run_time=0.7)

    # What goes wrong ---------------------------------------------------------------------------

    def what_goes_wrong(self):
        self.slide("what-goes-wrong", "Five mistakes that look reasonable", book_section="s-what-goes-wrong")
        tab = kit.table(
            [
                ["Timing without a synchronize", "faster than the roofline floor"],
                [r"Counting $n_i$ gaps, not $n_i - 1$", r"ITL understated: 0.8\% at 128 tokens, 100\% at 2"],
                ["Summing durations for throughput", "throughput falls as batch size rises"],
                ["Reporting the mean of a skewed tail", "lands between P50 and P90, describes neither"],
                ["Comparing different cards or drivers", "nothing looks anomalous"],
            ],
            header=["Mistake", "Symptom"],
            size=th.SIZE_SMALL, row_gap=0.32,
        )
        prod = kit.callout("note", r"Against a production engine, match every condition and expect to be "
                                   r"2 to 5 times slower at first. Split the gap into prefill and decode.",
                           title="Comparing against vLLM", width=11.5, size=th.SIZE_SMALL)
        col = VGroup(tab, prod).arrange(DOWN, buff=0.55)
        self.place(col)
        with self.narrate(
            "Each of these mistakes produces a number that looks reasonable. "
            "Time without a synchronize, and you beat the roofline floor, which is impossible. "
            "Count every token as a gap, and inter-token latency reads low, by half at 2 tokens. "
            "Sum request durations, and throughput falls as batch size rises. "
            "Report the mean of a skewed tail, and it lands between the median and the 90th percentile, describing neither. "
            "Compare different cards, and nothing looks wrong. "
            "Against a production engine, match every condition, and expect to be 2 to 5 times slower at first."
        ) as t:
            self.play(FadeIn(tab.rules), FadeIn(VGroup(*tab.cells[0])), run_time=0.6)
            for r, cue in enumerate(["Time without", "Count every token", "Sum request", "Report the mean",
                                     "Compare different"]):
                t.wait_until(cue)
                self.play(FadeIn(VGroup(*tab.cells[r + 1]), shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("Against a production")
            self.play(FadeIn(prod, shift=UP * 0.1), run_time=0.7)
