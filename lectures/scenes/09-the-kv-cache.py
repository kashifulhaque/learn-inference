"""Chapter 09, the KV cache.

Teaches content/chapters/09-the-kv-cache.md in its order. Every number below
was recomputed in Python (exact arithmetic, then rounded as the chapter does).

Work, p = 512 prompt tokens, n = 128 generated tokens:
- W_none = n*p + n(n-1)/2 = 65,536 + 8,128 = 73,664 token-forward-passes.
- W_cache = p + n - 1 = 639. Ratio 73,664 / 639 = 115.28, "115".
- 2N = 53.8 GFLOP per token-forward-pass: 73,664 x 53.8e9 = 3.963e15
  = 3.96 PFLOP; 639 x 53.8e9 = 34.38e12 = 34.4 TFLOP.
- Average uncached pass: 73,664 / 128 = 575.5 tokens ("hundreds").

Roofline floors (A100: 312 TFLOP/s, 1935 GB/s):
- Uncached: 3.963e15 / 312e12 = 12.70 s.
- Prefill: 512 x 53.8e9 / 312e12 = 88.3 ms.
- One decode step: 53.8e9 / 1935e9 = 27.80 ms; x 127 = 3.531 s.
- Cached total: 0.0883 + 3.531 = 3.62 s, "about 3.6 s"; 12.70 / 3.62 = 3.51,
  "about 3.5 times".
- One token-forward-pass at peak: 53.8e9 / 312e12 = 0.1724 ms.
- Long-output speedup: (n/2)(0.1724 / 27.80) = 0.0031009 n; 20 / 0.0031009
  = 6450, "roughly 6500".

Per-token and per-sequence state:
- KV per token per layer: 2 x 4 x 256 x 2 = 4096 B = 4 KiB; x 16 layers
  = 65,536 B = 64 KiB.
- Recurrent state: 154,927,104 B = 147.75 MiB ("147.8"); / 48 layers
  = 3.078 MiB ("3.08 MiB per layer").

Layout, 32k context, one decode step, one layer:
- Read 2 x 4 x 32768 x 256 x 2 = 134,217,728 B = 128 MiB.
- Write 2 x 4 x 1 x 256 x 2 = 4096 B = 4 KiB. Ratio 32,768 : 1.

torch.cat over the running example:
- 73,664 tokens x 65,536 B = 4.496 GiB, "4.5 GiB" read and again written.
- 2 x 4.828e9 B / 1275e9 B/s = 7.57 ms, "about 8 ms".
- Reservation: 32 x 32768 x 65,536 B = 68.72e9 B = 68.7 GB = 64 GiB.

Memory per batch (GiB = KV + state; state 0.1443 GiB per sequence):
- Batch 8, 32k: 16.00 + 1.15 = 17.15, "17.1".
- Batch 32, 4k: 8.00 + 4.62 = 12.62, "12.6".
- Batch 64, 4k: 16.00 + 9.23 = 25.23, "25.2". State alone 64 x 147.75 MiB
  = 9.23 GiB, "9.2 GiB". Budget 19 GiB (chapter 2).

Break-even against all-full-attention (256 KiB = 262,144 B per token):
- 262,144 L = 65,536 L + 154,927,104, so L = 154,927,104 / 196,608 = 788.0.
  Plot: full = 0.25 L MiB, hybrid = 147.75 + 0.0625 L MiB; both 197 MiB at 788.
- Margin 196,608 B = 192 KiB per further token.
- 32k: hybrid 2.144 GiB vs 32768 x 262,144 B = 8.0 GiB.

Decode traffic per sequence per step:
- State read + write: 2 x 154,927,104 = 309,854,208 B = 295.5 MiB.
- KV read 65,536 L B = L / 16 MiB; equal at L = 309,854,208 / 65,536 = 4728.
- At L = 512: KV 32 MiB; 295.5 / 32 = 9.23, "about 9 times".
"""

from manim import *

import kit
from kit import theme as th

APPEND = """\
def append(self, layer_idx, k, v):
    seq = k.shape[2]
    start, stop = self.length, self.length + seq
    if stop > self.max_seq_len:
        raise RuntimeError("Cache overflow ...")
    self.k_cache[layer_idx][:, :, start:stop] = k   # in-place write
    self.v_cache[layer_idx][:, :, start:stop] = v
    return (self.k_cache[layer_idx][:, :, :stop],   # views
            self.v_cache[layer_idx][:, :, :stop])
"""

ADVANCE = """\
for layer in self.layers:
    x = layer(x, cos, sin, cache=cache)
if cache is not None:
    cache.advance(seq)
"""

KV_COL = th.HL_C  # \hlc in the chapter: the KV cache term
STATE_COL = th.HL_D  # \hld in the chapter: the recurrent-state term


def cell(w, h, color=None, tint=0.35, stroke=th.RULE_STRONG):
    """One memory cell: a small rectangle, tinted when `color` is given."""
    fill = th.RAISE if color is None else th.tint(color, tint)
    return Rectangle(width=w, height=h, stroke_color=stroke, stroke_width=th.STROKE_HAIRLINE,
                     fill_color=fill, fill_opacity=1)


def staircase(p, n, w, h, gap=0.05):
    """Rows of cells: row j (0-based) holds p + j cells, left-aligned.

    Returns a VGroup of rows; row j is a VGroup of its cells.
    """
    rows = VGroup()
    for j in range(n):
        row = VGroup(*[cell(w, h) for _ in range(p + j)]).arrange(RIGHT, buff=gap)
        rows.add(row)
    rows.arrange(DOWN, buff=gap, aligned_edge=LEFT)
    return rows


def swatch(text, color, size=th.SIZE_LABEL):
    sq = Square(0.26, stroke_color=th.RULE_STRONG, stroke_width=th.STROKE_HAIRLINE,
                fill_color=th.tint(color, 0.35), fill_opacity=1)
    return VGroup(sq, kit.label(text, size=size, color=th.INK_2)).arrange(RIGHT, buff=0.16)


def line_key(text, color, size=th.SIZE_LABEL):
    rule = Line(LEFT * 0.25, RIGHT * 0.25, color=color, stroke_width=th.STROKE_CURVE)
    return VGroup(rule, kit.label(text, size=size, color=th.INK_2)).arrange(RIGHT, buff=0.16)


class Lecture(kit.Lecture):
    def construct(self):
        self.title_slide(
            narration=(
                "The KV cache. In this lecture, you stop recomputing the past. "
                "You derive why the work drops from quadratic to linear. "
                "Then you pay for it, in two very different data structures."
            )
        )
        self.same_work_again()
        self.uncached_sum()
        self.cached_sum()
        self.worked_example()
        self.roofline_floors()
        self.gain_grows()
        self.keys_fixed()
        self.append_and_read()
        self.post_rope()
        self.linear_layers()
        self.two_structures()
        self.no_rewind()
        self.layout()
        self.no_cat()
        self.append_code()
        self.memory_budget()
        self.break_even()
        self.decode_traffic()
        self.state_carry()
        self.what_goes_wrong()
        self.closing_slide(
            [
                r"Work: $np + n(n-1)/2$ uncached, $p + n - 1$ cached.",
                r"Decode is memory bound: $3.5\times$ faster, not $115\times$.",
                r"Cache post-RoPE K and V; the state is fixed and can't rewind.",
                r"BHSD, preallocated, advanced once per pass.",
            ],
            narration=(
                "Here's what to keep. "
                "Without a cache, work grows with the square of the output, and with one, it grows linearly. "
                "Decode waits on memory, so the time saved is much smaller than the work saved. "
                "Cache rotated keys and values for the full-attention layers, and a fixed float32 state for the "
                "linear ones. "
                "Lay the cache out as BHSD, allocate it once, and advance it once per forward pass. "
                "Next, the roofline shows exactly where that memory-bound ceiling comes from."
            ),
            book_section="s-the-memory-arithmetic",
        )

    # From quadratic to linear ---------------------------------------------------

    def same_work_again(self):
        self.slide("same-work-again", "Each step redoes every step before it",
                   book_section="s-from-quadratic-to-linear")
        p, n = 4, 5
        rows = staircase(p, n, 0.78, 0.5, gap=0.06)
        heads = VGroup(*[kit.label(f"t{c + 1}", size=th.SIZE_LABEL, color=th.INK_3, font=th.MONO)
                         for c in range(p + n - 1)])
        for c, h in enumerate(heads):
            h.next_to(rows[-1][c], UP, buff=0).set_y(rows[0].get_top()[1] + 0.25)
        names = VGroup(*[kit.label(f"pass {j + 1}", size=th.SIZE_LABEL, color=th.INK_3)
                         for j in range(n)])
        for j, nm in enumerate(names):
            nm.next_to(rows[j], LEFT, buff=0.3)
        brace = Brace(VGroup(*rows[0]), UP, buff=0.5, color=th.INK_4)
        brace_t = kit.label("prompt", size=th.SIZE_LABEL, color=th.INK_3).next_to(brace, UP, buff=0.08)
        legend = VGroup(swatch("new work", self.color), swatch("recomputed, identical", th.INK_4)).arrange(
            RIGHT, buff=0.6)
        figure = VGroup(rows, heads, names, brace, brace_t)
        top = kit.prose(r"Token 128 of a reply to a 512-token prompt: 639 tokens through 64 layers.",
                        size=th.SIZE_SMALL, width=None)
        col = VGroup(top, figure, legend).arrange(DOWN, buff=0.4)
        self.place(col)
        # Colour the cells now that they are placed.
        for j, row in enumerate(rows):
            for c, sq in enumerate(row):
                new = (j == 0) or (c == p + j - 1)
                sq.set_fill(th.tint(self.color if new else th.INK_4, 0.35), opacity=1)
        with self.narrate(
            "Picture the model writing the 128th token of a reply to a 512-token prompt. "
            "Run the transformer's definition literally, and it pushes all 639 tokens so far through all 64 layers, "
            "to get one token. "
            "Here's a small version of the same run, with a four-token prompt. "
            "Each row is one forward pass. "
            "The first pass processes the prompt. "
            "Every later pass reprocesses everything before it, and adds one new token at the end. "
            "The gray cells compute keys, values, and states identical to the ones from a step ago. "
            "The KV cache stores them once, and reads them back."
        ) as t:
            self.play(FadeIn(top), run_time=0.8)
            t.wait_until("Here's a small version")
            self.play(FadeIn(heads), FadeIn(brace), FadeIn(brace_t), run_time=0.8)
            t.wait_until("Each row is one")
            self.play(FadeIn(names), run_time=0.6)
            t.wait_until("The first pass")
            self.play(LaggedStart(*[FadeIn(s, scale=0.8) for s in rows[0]], lag_ratio=0.15), run_time=0.9)
            t.wait_until("Every later pass")
            for j in range(1, n):
                self.play(LaggedStart(*[FadeIn(s, scale=0.8) for s in rows[j]], lag_ratio=0.06), run_time=0.7)
            t.wait_until("The gray cells")
            self.play(FadeIn(legend), run_time=0.6)
            gray = VGroup(*[sq for j in range(1, n) for sq in rows[j][: p + j - 1]])
            self.play(Indicate(gray, color=th.INK_3, scale_factor=1.04), run_time=1.2)

    def uncached_sum(self):
        self.slide("uncached-sum", "Without a cache, work grows as n²", book_section="s-from-quadratic-to-linear")
        p, n = 5, 8
        rows = staircase(p, n, 0.46, 0.4, gap=0.05)
        for j, row in enumerate(rows):
            for c, sq in enumerate(row):
                sq.set_fill(th.tint(th.HL_A if c < p else th.HL_B, 0.35), opacity=1)
        prompt = VGroup(*[sq for row in rows for sq in row[:p]])
        growth = VGroup(*[sq for row in rows for sq in row[p:]])
        np_lab = kit.formula(r"np", size=th.SIZE_MATH, color=th.HL_A).move_to(prompt)
        tri_lab = kit.formula(r"\frac{n(n-1)}{2}", size=th.SIZE_MATH_SMALL, color=th.HL_B)
        tri_lab.next_to(rows[1][-1], RIGHT, buff=0.7).set_y(rows[0].get_center()[1] - 0.1)
        picture = VGroup(rows, np_lab, tri_lab)
        unit = kit.prose(r"Unit of work: one token through all 64 layers, once.", size=th.SIZE_SMALL, width=5.0)
        l1 = kit.formula(r"W_{\text{none}}", r"=", r"\textstyle\sum_{j=0}^{n-1} (p + j)")
        l2 = kit.formula(r"W_{\text{none}}", r"=", r"\textstyle\sum_{j=0}^{n-1} p + \sum_{j=0}^{n-1} j")
        l3 = kit.formula(r"W_{\text{none}}", r"=", r"np", r"+", r"\tfrac{n(n-1)}{2}")
        l3[2].set_color(th.HL_A)
        l3[4].set_color(th.HL_B)
        for l in (l2, l3):
            l[0].set_opacity(0)
        maths = VGroup(l1, l2, l3).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        for l in (l2, l3):
            l.shift(RIGHT * (l1[1].get_left()[0] - l[1].get_left()[0]))
        left = VGroup(unit, maths).arrange(DOWN, buff=0.55, aligned_edge=LEFT)
        layout = VGroup(left, picture).arrange(RIGHT, buff=0.9)
        self.place(layout)
        box = SurroundingRectangle(l3[2:], color=th.INK_3, buff=0.14, stroke_width=th.STROKE_AXIS)
        with self.narrate(
            "To count the waste, measure work in token-forward-passes: one token through all 64 layers, once. "
            "Without a cache, the first new token needs a pass over the p prompt tokens. "
            "The second needs p plus one, and the j-th needs p plus j minus one. "
            "Add up the passes, and split the sum in two. "
            "The green block is the prompt, re-read once for every generated token: n times p. "
            "The blue triangle is the generated tokens re-reading each other: n times n minus one, over two. "
            "That triangle grows with the square of n, so uncached generation is quadratic."
        ) as t:
            self.play(FadeIn(unit), run_time=0.7)
            t.wait_until("Without a cache, the first")
            self.play(FadeIn(rows[0]), run_time=0.6)
            t.wait_until("The second needs")
            self.play(LaggedStart(*[FadeIn(r, shift=DOWN * 0.05) for r in rows[1:]], lag_ratio=0.2),
                      FadeIn(l1), run_time=2.0)
            t.wait_until("Add up the passes")
            self.play(FadeIn(l2, shift=DOWN * 0.1), l1.animate.set_color(th.INK_3), run_time=0.9)
            t.wait_until("The green block")
            self.play(FadeIn(l3[:3], shift=DOWN * 0.1), Indicate(prompt, color=th.HL_A, scale_factor=1.02),
                      FadeIn(np_lab), run_time=1.0)
            t.wait_until("The blue triangle")
            self.play(FadeIn(l3[3:], shift=DOWN * 0.1), Indicate(growth, color=th.HL_B, scale_factor=1.02),
                      FadeIn(tri_lab), run_time=1.0)
            t.wait_until("That triangle grows")
            self.play(l2.animate.set_color(th.INK_3), Create(box), run_time=0.8)

    def cached_sum(self):
        self.slide("cached-sum", "With a cache, each token is processed once",
                   book_section="s-from-quadratic-to-linear")
        p, n = 5, 8
        rows = staircase(p, n, 0.46, 0.4, gap=0.05)
        for row in rows:
            for sq in row:
                sq.set_fill(th.tint(th.INK_4, 0.35), opacity=1)
        keep = VGroup(*rows[0], *[rows[j][p + j - 1] for j in range(1, n)])
        drop = VGroup(*[sq for j in range(1, n) for sq in rows[j][: p + j - 1]])
        pre_lab = kit.label("prefill", size=th.SIZE_LABEL, color=self.color)
        step_lab = kit.label("one token per step", size=th.SIZE_LABEL, color=self.color)
        picture = VGroup(rows)
        l1 = kit.formula(r"W_{\text{cache}}", r"=", r"p + (n - 1)", r"= O(p + n)")
        l2 = kit.formula(r"\frac{W_{\text{none}}}{W_{\text{cache}}}", r"=",
                         r"\frac{np + \tfrac{n(n-1)}{2}}{p + n - 1}")
        l3 = kit.formula(r"\xrightarrow[\;n \gg p\;]{}", r"\frac{n}{2}")
        maths = VGroup(l1, l2, l3).arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        l3.shift(RIGHT * (l2[1].get_left()[0] - l3.get_left()[0]))
        layout = VGroup(picture, maths).arrange(RIGHT, buff=1.0)
        self.place(layout)
        pre_lab.next_to(rows[0], UP, buff=0.15).align_to(rows[0], LEFT)
        step_lab.next_to(rows[-1][-1], DOWN, buff=0.18).align_to(rows[-1][-1], RIGHT)
        box = SurroundingRectangle(l3[1], color=self.color, buff=0.12, stroke_width=th.STROKE_AXIS)
        with self.narrate(
            "Now add the cache, and no token is ever processed twice. "
            "The prefill pass handles all p prompt tokens at once, and produces the first new token. "
            "Each later step processes exactly one token, and there are n minus one of them. "
            "So the cached work is p plus n minus one, which is linear. "
            "Divide the two, and for long outputs, the ratio tends to n over two. "
            "That's the whole result. "
            "The rest of this lecture is about what it costs."
        ) as t:
            self.play(FadeIn(rows), run_time=0.8)
            self.play(drop.animate.set_fill(th.CANVAS, opacity=1).set_stroke(th.RULE, opacity=0.6), run_time=1.2)
            t.wait_until("The prefill pass")
            self.play(VGroup(*rows[0]).animate.set_fill(th.tint(self.color, 0.35)), FadeIn(pre_lab), run_time=0.8)
            t.wait_until("Each later step")
            self.play(LaggedStart(*[rows[j][p + j - 1].animate.set_fill(th.tint(self.color, 0.35))
                                    for j in range(1, n)], lag_ratio=0.15), FadeIn(step_lab), run_time=1.4)
            t.wait_until("So the cached work")
            self.play(FadeIn(l1, shift=DOWN * 0.1), run_time=0.8)
            t.wait_until("Divide the two")
            self.play(FadeIn(l2, shift=DOWN * 0.1), run_time=0.8)
            self.play(FadeIn(l3, shift=RIGHT * 0.1), run_time=0.7)
            t.wait_until("That's the whole result")
            self.play(Create(box), run_time=0.6)

    def worked_example(self):
        self.slide("worked-example", "For 128 new tokens: 73,664 units of work become 639",
                   book_section="s-from-quadratic-to-linear")
        given = kit.formula(r"p = 512, \quad n = 128", size=th.SIZE_MATH)
        bc = kit.bar_compare(
            [("Without a cache", 73664, th.INFO, "73,664"), ("With a cache", 639, self.color, "639")],
            width=8.0, bar_height=0.6, gap=0.45,
        )
        flops = VGroup(
            kit.label("3.96 PFLOP", size=th.SIZE_SMALL, color=th.INK_3),
            kit.label("34.4 TFLOP", size=th.SIZE_SMALL, color=th.INK_3),
        )
        for f, v in zip(flops, bc.values):
            f.next_to(v, RIGHT, buff=0.35)
        chart = VGroup(bc, flops)
        per = kit.prose(r"One token-forward-pass $\approx 2N = 53.8$ GFLOP.", size=th.SIZE_SMALL, width=None,
                        color=th.INK_3)
        ratio = kit.pill("115 × less work", self.color, size=th.SIZE_SMALL)
        col = VGroup(given, chart, VGroup(per, ratio).arrange(RIGHT, buff=0.8)).arrange(DOWN, buff=0.7)
        self.place(col)
        with self.narrate(
            "Plug in the running example: a 512-token prompt, and 128 new tokens. "
            "Without a cache, that's 73,664 token-forward-passes. "
            "With a cache, it's 639. "
            "The cache cuts the work by a factor of 115. "
            "One token-forward-pass costs about 53.8 GFLOP. "
            "So the uncached run needs 3.96 PFLOP. "
            "The cached run needs only 34.4 TFLOP."
        ) as t:
            self.play(FadeIn(given), FadeIn(bc.labels), run_time=0.8)
            t.wait_until("Without a cache, that's")
            self.play(GrowFromEdge(bc.bars[0], LEFT), FadeIn(bc.values[0]), run_time=1.0)
            t.wait_until("With a cache, it's")
            self.play(GrowFromEdge(bc.bars[1], LEFT), FadeIn(bc.values[1]), run_time=0.6)
            t.wait_until("The cache cuts")
            self.play(FadeIn(ratio, scale=0.9), run_time=0.6)
            t.wait_until("One token-forward-pass costs")
            self.play(FadeIn(per), run_time=0.6)
            t.wait_until("So the uncached run")
            self.play(FadeIn(flops[0]), run_time=0.5)
            t.wait_until("The cached run needs")
            self.play(FadeIn(flops[1]), run_time=0.5)

    # Why the wall-clock gain is smaller -----------------------------------------

    def roofline_floors(self):
        self.slide("roofline-floors", "Decode waits on memory, not arithmetic",
                   book_section="s-why-the-wall-clock-gain-is-smaller")
        hw = kit.label("A100:  312 TFLOP/s tensor-core peak  ·  1935 GB/s HBM", size=th.SIZE_SMALL,
                       color=th.INK_2)
        bc = kit.bar_compare(
            [
                ("Uncached, 128 passes", 12.70, th.INFO, "12.7 s  ·  compute bound"),
                ("Prefill, 512 tokens", 0.0883, self.color, "88 ms  ·  compute bound"),
                ("127 decode steps", 3.531, self.color, "3.53 s  ·  memory bound"),
            ],
            width=6.6, bar_height=0.55, gap=0.42,
        )
        step = kit.prose(r"One decode step reads all 53.8 GB of weights: 27.8 ms.", size=th.SIZE_SMALL,
                         width=None, color=th.INK_3)
        result = kit.callout("key", r"Cached total $\approx 3.6$ s: about $3.5\times$ faster, not $115\times$.",
                             title="With the cache", width=8.6)
        col = VGroup(hw, bc, step, result).arrange(DOWN, buff=0.5)
        self.place(col)
        with self.narrate(
            "Work isn't time, though. "
            "The A100's tensor cores peak at 312 TFLOP/s. "
            "Its HBM delivers 1935 GB/s. "
            "Uncached passes cover hundreds of tokens each, so they run near peak arithmetic. "
            "Their floor is 12.7 seconds. "
            "Prefill is also compute bound, at 88 ms. "
            "A decode step covers one token, but it still reads all 53.8 GB of weights. "
            "That takes 27.8 ms, so 127 steps take 3.53 seconds. "
            "With the cache, the whole run takes about 3.6 seconds. "
            "That's about 3.5 times faster, not 115, because each cached step waits on memory."
        ) as t:
            t.wait_until("The A100's tensor")
            self.play(FadeIn(hw), FadeIn(bc.labels), run_time=0.8)
            t.wait_until("Their floor is")
            self.play(GrowFromEdge(bc.bars[0], LEFT), FadeIn(bc.values[0]), run_time=1.0)
            t.wait_until("Prefill is also")
            self.play(GrowFromEdge(bc.bars[1], LEFT), FadeIn(bc.values[1]), run_time=0.6)
            t.wait_until("A decode step covers")
            self.play(FadeIn(step), run_time=0.6)
            t.wait_until("That takes 27.8")
            self.play(GrowFromEdge(bc.bars[2], LEFT), FadeIn(bc.values[2]), run_time=1.0)
            t.wait_until("With the cache, the whole")
            self.play(FadeIn(result, shift=UP * 0.1), run_time=0.8)

    def gain_grows(self):
        self.slide("gain-grows", "The speedup grows with the output length",
                   book_section="s-why-the-wall-clock-gain-is-smaller")
        slope = 0.5 * (53.8e9 / 312e12) / (53.8e9 / 1935e9)  # 0.0031009 per token
        ax = kit.axes([0, 8000, 1000], [0, 25, 5], width=7.6, height=4.3,
                      x_label=r"\text{generated tokens } n", y_label=r"\text{speedup}",
                      x_numbers=[0, 2000, 4000, 6000, 8000], y_numbers=[0, 5, 10, 15, 20, 25])
        line = ax.plot(lambda x: slope * x, x_range=[0, 8000], color=self.color, stroke_width=th.STROKE_CURVE)
        n20 = 20 / slope
        h = DashedLine(ax.c2p(0, 20), ax.c2p(n20, 20), color=th.INK_4, stroke_width=th.STROKE_AXIS)
        v = DashedLine(ax.c2p(n20, 0), ax.c2p(n20, 20), color=th.INK_4, stroke_width=th.STROKE_AXIS)
        dot = Dot(ax.c2p(n20, 20), radius=0.09, color=self.color)
        note = kit.prose(r"$20\times$ needs roughly 6500 tokens.", size=th.SIZE_SMALL, width=4.0)
        plot = VGroup(ax, line, h, v, dot)
        rule = kit.formula(r"\frac{n}{2}\cdot\frac{0.172\ \text{ms}}{27.8\ \text{ms}}", r"\approx 0.0031\,n",
                           size=th.SIZE_MATH_SMALL)
        cap = kit.prose(r"Long outputs, $n \gg p$. Roofline floors, not measurements.", size=th.SIZE_SMALL,
                        width=4.0, color=th.INK_3)
        side = VGroup(rule, note, cap).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        layout = VGroup(plot, side).arrange(RIGHT, buff=0.6)
        self.place(layout)
        with self.narrate(
            "The gap closes as the output grows. "
            "For long outputs, uncached time is about n² / 2 passes, at 0.172 ms of arithmetic each. "
            "Cached time is about n decode steps, at 27.8 ms each. "
            "Divide the two, and the speedup grows linearly, at about 0.0031 times n. "
            "A 20× speedup needs roughly 6500 generated tokens. "
            "These are roofline floors, not measurements. "
            "The work ratio is the part that doesn't depend on how good your kernels are."
        ) as t:
            self.play(Create(ax), run_time=1.0)
            t.wait_until("For long outputs")
            self.play(FadeIn(rule[0]), run_time=0.8)
            t.wait_until("Divide the two")
            self.play(FadeIn(rule[1]), Create(line), run_time=1.6)
            t.wait_until("A 20× speedup")
            self.play(Create(h), Create(v), FadeIn(dot, scale=0.5), FadeIn(note), run_time=1.0)
            t.wait_until("These are roofline")
            self.play(FadeIn(cap), run_time=0.6)

    # What to cache ----------------------------------------------------------------

    def keys_fixed(self):
        self.slide("keys-fixed", "A key never changes after it's written", book_section="s-what-to-cache")
        score = kit.formula(r"s_{mn} = \frac{q_m^{\top} k_n}{\sqrt{d}}", r"\qquad d = 256")
        score[1].set_color(th.INK_3)
        vals = [str(i) for i in range(6)] + ["6", "7", "…", "5999"]
        strip = kit.grid(1, 10, cell=(0.85, 0.6), values=[vals], gap=0.06)
        later = VGroup(*[strip.cell(0, c) for c in range(6, 10)])
        for c in range(6):
            strip.paint(strip.cell(0, c), self.color)
        tok = kit.label("tokens", size=th.SIZE_LABEL, color=th.INK_3).next_to(strip, LEFT, buff=0.3)
        brace = Brace(strip.block(0, 1, 0, 6), UP, buff=0.1, color=th.INK_4)
        brace_t = kit.label("causal: tokens 0 to 5 only", size=th.SIZE_LABEL, color=th.INK_3)
        brace_t.next_to(brace, UP, buff=0.08)
        key = kit.box(r"$k_5,\ v_5$", color=self.color, width=1.8, height=0.8)
        key.next_to(strip.cell(0, 5), DOWN, buff=0.8)
        fixed = kit.prose(r"The same $k_5$ at length 6 or 6000.", size=th.SIZE_SMALL, width=None)
        fixed.next_to(key, DOWN, buff=0.3)
        fig = VGroup(strip, tok, brace, brace_t, key, fixed)
        col = VGroup(score, fig).arrange(DOWN, buff=0.55)
        self.place(col)
        link = kit.arrow(strip.cell(0, 5), key, color=self.color)
        with self.narrate(
            "The saving only holds if the stored tensors can't change. "
            "In a full-attention layer, the score between the query at position m and the key at position n is "
            "their dot product, scaled by the square root of the head width. "
            "Here, the head width is 256. "
            "The key at position n depends only on the residual stream at position n. "
            "Causal masking means that stream depends only on tokens zero through n. "
            "So once token n is processed, its key and value are fixed forever. "
            "The key for token 5 is the same whether the sequence is 6 tokens long, or 6000."
        ) as t:
            t.wait_until("In a full-attention layer")
            self.play(FadeIn(score[0]), run_time=0.8)
            t.wait_until("Here, the head width")
            self.play(FadeIn(score[1]), run_time=0.5)
            t.wait_until("The key at position n")
            self.play(FadeIn(strip.block(0, 1, 0, 6)), FadeIn(tok), run_time=0.8)
            self.play(GrowArrow(link.line), FadeIn(key), run_time=0.8)
            t.wait_until("Causal masking")
            self.play(FadeIn(brace), FadeIn(brace_t), run_time=0.7)
            t.wait_until("The key for token 5")
            self.play(LaggedStart(*[FadeIn(c, shift=RIGHT * 0.1) for c in later], lag_ratio=0.2), run_time=1.2)
            self.play(Indicate(key, color=self.color, scale_factor=1.08), FadeIn(fixed), run_time=1.0)

    def append_and_read(self):
        self.slide("append-and-read", "Each step writes one column and reads them all",
                   book_section="s-what-to-cache")
        cols = 10
        g = kit.grid(2, cols, cell=(0.66, 0.7), gap=0.05, row_labels=["keys", "values"])
        for c in range(6):
            g.paint(g.col(c), self.color)
        notes = kit.bullets(
            [
                r"Keys and values: read by every later step.",
                r"Queries: used once, by their own step.",
                r"Attention outputs: never asked for again.",
            ],
            size=th.SIZE_SMALL, width=4.6,
        )
        layout = VGroup(g, notes).arrange(RIGHT, buff=0.7)
        self.place(layout)

        def step(m):
            q = kit.pill("q", th.VIOLET, size=th.SIZE_SMALL).next_to(g.cell(0, m), UP, buff=0.25)
            self.play(FadeIn(q, shift=DOWN * 0.1), g.fill(g.col(m), th.HL_C), run_time=0.7)
            read = g.outline(0, 2, 0, m + 1, color=th.VIOLET)
            self.play(Create(read), run_time=0.7)
            self.play(g.fill(g.col(m), self.color), FadeOut(q), FadeOut(read), run_time=0.6)

        with self.narrate(
            "Here's one decode step. "
            "The new token's key and value are written into the next free column. "
            "Then its query reads every key and value in the cache, old and new. "
            "The query is used once, by this step, and never read again, so you don't cache it. "
            "The attention output is final too, but no later step asks for it. "
            "The next step does the same thing: write one column, then read them all. "
            "Keys and values are the only things every later step needs."
        ) as t:
            self.play(FadeIn(g), run_time=0.8)
            t.wait_until("The new token's key")
            q = kit.pill("q", th.VIOLET, size=th.SIZE_SMALL).next_to(g.cell(0, 6), UP, buff=0.25)
            self.play(FadeIn(q, shift=DOWN * 0.1), g.fill(g.col(6), th.HL_C), run_time=0.8)
            t.wait_until("Then its query reads")
            read = g.outline(0, 2, 0, 7, color=th.VIOLET)
            self.play(Create(read), run_time=1.0)
            t.wait_until("The query is used once")
            self.play(FadeOut(q), FadeOut(read), g.fill(g.col(6), self.color), FadeIn(notes[1]), run_time=0.8)
            t.wait_until("The attention output")
            self.play(FadeIn(notes[2]), run_time=0.6)
            t.wait_until("The next step")
            step(7)
            # The last step keeps its query and read outline on screen.
            q = kit.pill("q", th.VIOLET, size=th.SIZE_SMALL).next_to(g.cell(0, 8), UP, buff=0.25)
            self.play(FadeIn(q, shift=DOWN * 0.1), g.fill(g.col(8), th.HL_C), run_time=0.7)
            self.play(Create(g.outline(0, 2, 0, 9, color=th.VIOLET)), g.fill(g.col(8), self.color), run_time=0.8)
            t.wait_until("Keys and values are the only")
            self.play(FadeIn(notes[0]), run_time=0.6)

    def post_rope(self):
        self.slide("post-rope", "The cache stores keys after RoPE", book_section="s-what-to-cache")
        ident = kit.formula(r"R_m^{\top} R_n = R_{n-m}", size=th.SIZE_MATH)
        main = kit.formula(r"(R_m q_m)^{\top}", r"(R_n k_n)", r"= q_m^{\top} R_m^{\top} R_n k_n =",
                           r"q_m^{\top}", r"R_{n-m}", r"k_n", size=th.SIZE_MATH)
        main[0].set_color(th.HL_A)
        main[1].set_color(th.HL_B)
        main[4].set_color(th.HL_C)
        q_note = kit.label("query: rotated fresh each step", size=th.SIZE_LABEL, color=th.HL_A)
        k_note = kit.label("stored once, never changes", size=th.SIZE_LABEL, color=th.HL_B)
        d_note = kit.label("only the distance n − m", size=th.SIZE_LABEL, color=th.HL_C)
        pre = kit.callout("warning", r"Store pre-RoPE keys, and every step rotates the whole prefix again: "
                                     r"$O(L)$ extra work, for nothing.", title="The alternative", width=9.6,
                          size=th.SIZE_SMALL)
        col = VGroup(ident, main, pre).arrange(DOWN, buff=1.2)
        self.place(col)
        q_note.next_to(main[0], DOWN, buff=0.3)
        k_note.next_to(main[1], UP, buff=0.3)
        d_note.next_to(main[4], DOWN, buff=0.3)
        with self.narrate(
            "RoPE marks each key's position by rotating it, by angles that depend on the absolute position. "
            "The engine rotates the key before it writes it to the cache. "
            "That's safe because of one identity: R m transposed, times R n, equals R of n minus m. "
            "So the rotated query and the stored key combine into a rotation by the difference in positions. "
            "Position n is fixed the moment the token is written, so the stored, rotated key never changes. "
            "Store keys before RoPE instead, and every decode step must rotate the whole prefix again, for nothing. "
            "Values are never rotated, so the question doesn't arise for them."
        ) as t:
            t.wait_until("The engine rotates")
            self.play(FadeIn(main[1]), FadeIn(k_note), run_time=0.8)
            t.wait_until("That's safe because")
            self.play(FadeIn(ident, shift=DOWN * 0.1), run_time=0.8)
            t.wait_until("So the rotated query")
            self.play(FadeIn(main[0]), FadeIn(q_note), run_time=0.7)
            self.play(FadeIn(main[2:]), run_time=1.0)
            self.play(FadeIn(d_note), Indicate(main[4], color=th.HL_C), run_time=0.9)
            t.wait_until("Position n is fixed")
            self.play(Circumscribe(main[1], color=th.HL_B, buff=0.1), run_time=1.2)
            t.wait_until("Store keys before")
            self.play(FadeIn(pre, shift=UP * 0.1), run_time=0.8)

    def linear_layers(self):
        self.slide("linear-layers", "The 48 linear layers keep a fixed state", book_section="s-what-to-cache")
        g = kit.grid(1, 64, cell=(0.16, 0.6), gap=0.03)
        full = VGroup(*[g.cell(0, i) for i in range(64) if i % 4 == 3])
        lin = VGroup(*[g.cell(0, i) for i in range(64) if i % 4 != 3])
        cap = kit.label("64 layers", size=th.SIZE_LABEL, color=th.INK_3).next_to(g, UP, buff=0.15)
        cap.align_to(g, LEFT)
        key = VGroup(swatch("16 full attention: keys and values", self.color),
                     swatch("48 linear attention: state and conv window", th.INFO)).arrange(RIGHT, buff=0.7)
        state = kit.box("Delta-rule state S", sublabel="(batch, 48, 128, 128)  ·  float32", color=th.INFO,
                        width=5.4)
        conv = kit.box("Convolution window", sublabel="(batch, 10240, 3)  ·  bf16", color=th.INFO, width=5.4)
        boxes = VGroup(state, conv).arrange(RIGHT, buff=0.6)
        note = kit.prose(r"Neither grows. Both are \textbf{overwritten in full} every step.", size=th.SIZE_SMALL,
                         width=None)
        col = VGroup(VGroup(cap, g), key, boxes, note).arrange(DOWN, buff=0.5)
        self.place(col)
        with self.narrate(
            "Only 16 of the 64 layers use full attention: every fourth one. "
            "The other 48 are linear-attention layers, and they don't keep keys and values. "
            "Each one keeps a delta-rule state: 48 heads of 128 by 128 values, in float32. "
            "Each also keeps a convolution window: the previous 3 steps of a four-tap kernel, in bf16. "
            "Neither grows with the sequence, and neither is appended to. "
            "Both are overwritten in full on every step."
        ) as t:
            self.play(FadeIn(g), FadeIn(cap), run_time=0.7)
            self.play(LaggedStart(*[c.square.animate.set_fill(th.tint(self.color, 0.35)) for c in full],
                                  lag_ratio=0.08), FadeIn(key[0]), run_time=1.4)
            t.wait_until("The other 48")
            self.play(LaggedStart(*[c.square.animate.set_fill(th.tint(th.INFO, 0.35)) for c in lin],
                                  lag_ratio=0.02), FadeIn(key[1]), run_time=1.4)
            t.wait_until("Each one keeps")
            self.play(FadeIn(state, shift=UP * 0.1), run_time=0.8)
            t.wait_until("Each also keeps")
            self.play(FadeIn(conv, shift=UP * 0.1), run_time=0.8)
            t.wait_until("Both are overwritten")
            self.play(FadeIn(note), Indicate(state.rect, color=th.INFO, scale_factor=1.04),
                      Indicate(conv.rect, color=th.INFO, scale_factor=1.04), run_time=1.0)

    # Two data structures --------------------------------------------------------

    def two_structures(self):
        self.slide("two-structures", "Two data structures that behave nothing alike",
                   book_section="s-two-data-structures-not-one")
        tab = kit.table(
            [
                ["Layers", "16", "48"],
                ["Size", "+4 KiB per token, per layer", "fixed 3.08 MiB per layer"],
                ["Written per step", "one token's slice", "the whole tensor"],
                ["Read per step", "the whole prefix", "the whole tensor"],
                ["Dtype", "bf16", "float32"],
            ],
            header=["", "KV cache", "Recurrent state"],
            size=th.SIZE_SMALL, col_gap=0.8, row_gap=0.32,
        )
        self.place(tab)
        band = kit.mark(VGroup(*tab.cells[5][1:]), buff=0.1)
        with self.narrate(
            "So the engine's hybrid cache holds two data structures, keyed by layer index. "
            "The KV half grows by 4 KiB per token, in each of its 16 layers. "
            "The state half is fixed at 3.08 MiB in each of its 48 layers. "
            "Each step, the KV half writes one token's slice, and the state half rewrites the whole tensor. "
            "Both read everything they hold, every step. "
            "The KV half stays in bf16. "
            "The state stays in float32, because the recurrence multiplies it by a decay thousands of times. "
            "In bf16, the rounding from that repeated product builds into visible drift."
        ) as t:
            self.play(FadeIn(tab.rules), *[FadeIn(c) for c in tab.cells[0]], *[FadeIn(c) for c in tab.cells[1]],
                      run_time=0.9)
            t.wait_until("The KV half grows")
            self.play(FadeIn(tab.cells[2][0]), FadeIn(tab.cells[2][1]), run_time=0.6)
            t.wait_until("The state half is fixed")
            self.play(FadeIn(tab.cells[2][2]), run_time=0.6)
            t.wait_until("Each step, the KV half")
            self.play(*[FadeIn(c) for c in tab.cells[3]], run_time=0.7)
            t.wait_until("Both read everything")
            self.play(*[FadeIn(c) for c in tab.cells[4]], run_time=0.7)
            t.wait_until("The KV half stays")
            self.play(FadeIn(tab.cells[5][0]), FadeIn(tab.cells[5][1]), run_time=0.6)
            t.wait_until("The state stays in float32")
            self.play(FadeIn(tab.cells[5][2]), FadeIn(band), run_time=0.7)

    def no_rewind(self):
        self.slide("no-rewind", "Keys and values can rewind; the state can't",
                   book_section="s-two-data-structures-not-one")
        g = kit.grid(1, 10, cell=(0.5, 0.6), gap=0.05)
        for c in range(9):
            g.paint(g.cell(0, c), self.color)
        kv_t = kit.label("KV cache", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        cur_x = lambda c: (g.cell(0, c - 1).get_right()[0] + g.cell(0, c).get_left()[0]) / 2  # noqa: E731
        st = kit.box("S", width=2.0, height=1.5, color=th.INFO, size=th.SIZE_HEADING)
        st_t = kit.label("Recurrent state", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        left = VGroup(kv_t, g).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        cursor = VGroup(
            Triangle(color=th.INK, fill_color=th.INK, fill_opacity=1, stroke_width=0).scale(0.11),
            kit.label("cursor", size=th.SIZE_LABEL, color=th.INK_3),
        ).arrange(DOWN, buff=0.08)
        cursor.next_to(g, DOWN, buff=0.12).set_x(cur_x(9))
        right = VGroup(st_t, st).arrange(DOWN, buff=0.35)
        toks = VGroup(*[kit.label(f"token {j}", size=th.SIZE_LABEL, color=th.INK_3) for j in (1, 2, 3)])
        toks.arrange(DOWN, buff=0.35).next_to(st, LEFT, buff=0.9)
        ins = VGroup(*[Arrow(tk.get_right(), st.rect.get_left() + UP * (0.4 - 0.4 * i), buff=0.1,
                             color=th.INK_4, stroke_width=3, tip_length=0.16) for i, tk in enumerate(toks)])
        folded = kit.label("added in; can't be undone", size=th.SIZE_LABEL, color=th.BAD).next_to(st, DOWN, buff=0.2)
        top = VGroup(VGroup(left, cursor), VGroup(right, toks, ins, folded)).arrange(RIGHT, buff=1.0, aligned_edge=UP)
        costs = kit.bullets(
            [
                r"Prefix sharing (chapter 15) works for KV, not for state.",
                r"Preemption by recompute replays the linear layers from the start.",
                r"Speculative decoding (chapter 20) must snapshot the state first.",
            ],
            size=th.SIZE_SMALL, width=12.0,
        )
        col = VGroup(top, costs).arrange(DOWN, buff=0.7)
        self.place(col)
        with self.narrate(
            "The other difference is rewinding. "
            "To drop the last few KV entries, you move the cursor back, and the earlier entries stay untouched. "
            "You can't do that to the state. "
            "Each position's contribution was added into it, and in floating point, that addition can't be undone. "
            "That costs you three times later in the course. "
            "Prefix sharing works for KV, but not for state. "
            "Preemption by recompute must replay the linear layers from the start. "
            "And speculative decoding must snapshot the state before it proposes tokens."
        ) as t:
            self.play(FadeIn(left), FadeIn(cursor), run_time=0.8)
            t.wait_until("To drop the last")
            self.play(cursor.animate.set_x(cur_x(6)), g.fill(g.block(0, 1, 6, 9), th.INK_4, tint=0.12),
                      run_time=1.4)
            t.wait_until("You can't do that")
            self.play(FadeIn(right), run_time=0.7)
            t.wait_until("Each position's contribution")
            self.play(FadeIn(toks), LaggedStart(*[GrowArrow(a) for a in ins], lag_ratio=0.3), run_time=1.2)
            self.play(FadeIn(folded), run_time=0.6)
            for row, cue in zip(costs, ["Prefix sharing works", "Preemption by", "And speculative"]):
                t.wait_until(cue)
                self.play(FadeIn(row, shift=RIGHT * 0.1), run_time=0.6)

    # Layout -------------------------------------------------------------------------

    def layout(self):
        self.slide("layout", "BHSD streams each head as one block", book_section="s-layout")
        heads, toks = 4, 4
        cw = 0.62
        bhsd = kit.grid(1, 16, cell=(cw, 0.5), values=[[f"t{i % toks}" for i in range(16)]], gap=0.0)
        bshd = kit.grid(1, 16, cell=(cw, 0.5), values=[[f"t{i // heads}" for i in range(16)]], gap=0.0)
        for i in range(16):
            bhsd.paint(bhsd.cell(0, i), self.colors[i // toks])
            bshd.paint(bshd.cell(0, i), self.colors[i % heads])
        n1 = kit.label("BHSD", size=th.SIZE_SMALL, color=th.INK, font=th.MONO)
        n2 = kit.label("BSHD", size=th.SIZE_SMALL, color=th.INK, font=th.MONO)
        r1 = VGroup(n1, bhsd).arrange(RIGHT, buff=0.4)
        r2 = VGroup(n2, bshd).arrange(RIGHT, buff=0.4)
        key = VGroup(*[swatch(f"head {h}", self.colors[h]) for h in range(heads)]).arrange(RIGHT, buff=0.5)
        a1 = kit.label("read one head: one block   ·   write one token: 4 pieces", size=th.SIZE_LABEL,
                       color=th.INK_3)
        a2 = kit.label("read one head: 4 runs with gaps   ·   write one token: one run", size=th.SIZE_LABEL,
                       color=th.INK_3)
        ratio = kit.prose(r"At 32k, one step in one layer reads 128 MiB and writes 4 KiB: \textbf{32,768 to 1}.",
                          size=th.SIZE_SMALL, width=None)
        col = VGroup(key, VGroup(r1, a1).arrange(DOWN, buff=0.22, aligned_edge=LEFT),
                     VGroup(r2, a2).arrange(DOWN, buff=0.22, aligned_edge=LEFT), ratio).arrange(DOWN, buff=0.55)
        for a, g in ((a1, bhsd), (a2, bshd)):
            a.align_to(g, LEFT)
        self.place(col)
        read1 = bhsd.outline(0, 1, 0, 4, color=th.INK)
        read2 = VGroup(*[bshd.outline(0, 1, c, c + 1, color=th.INK) for c in (0, 4, 8, 12)])
        write1 = VGroup(*[bhsd.outline(0, 1, c, c + 1, color=th.HL_D) for c in (3, 7, 11, 15)])
        write2 = bshd.outline(0, 1, 12, 16, color=th.HL_D)
        with self.narrate(
            "Every decode step reads the whole KV cache, so the order of its axes decides how that read streams "
            "from memory. "
            "The engine uses batch, heads, sequence, dim: BHSD. "
            "Here, each color is one of the 4 KV heads. "
            "In BHSD, reading one head across the sequence is a single contiguous block. "
            "In BSHD, the same read is broken into separate runs. "
            "Writing one token is the reverse: BSHD writes one run, and BHSD writes four small pieces. "
            "At this head size, BSHD doesn't waste bytes. "
            "It loses the long sequential run that DRAM and the L2 prefetcher reward. "
            "At 32k context, one decode step in one layer reads 128 MiB of cache. "
            "It writes only 4 KiB. "
            "That's a ratio of 32,768 to 1, so you optimize the read. "
            "BHSD it is."
        ) as t:
            t.wait_until("The engine uses")
            self.play(FadeIn(r1), run_time=0.7)
            t.wait_until("Here, each color")
            self.play(FadeIn(key), FadeIn(r2), run_time=0.8)
            t.wait_until("In BHSD, reading")
            self.play(Create(read1), run_time=0.8)
            t.wait_until("In BSHD, the same read")
            self.play(Create(read2), run_time=0.8)
            t.wait_until("Writing one token")
            self.play(FadeOut(read1), FadeOut(read2), Create(write2), run_time=0.8)
            self.play(Create(write1), run_time=0.8)
            t.wait_until("At this head size")
            self.play(FadeOut(write1), FadeOut(write2), FadeIn(a1), FadeIn(a2), run_time=0.8)
            t.wait_until("At 32k context")
            self.play(FadeIn(ratio), run_time=0.8)
            t.wait_until("BHSD it is")
            self.play(Create(SurroundingRectangle(r1, color=self.color, buff=0.12, stroke_width=3)), run_time=0.7)

    # Preallocate, don't concatenate ---------------------------------------------

    def no_cat(self):
        self.slide("no-cat", "torch.cat brings the quadratic back",
                   book_section="s-preallocate-don-t-concatenate")
        w, h, steps, p = 0.36, 0.34, 6, 4
        bars = VGroup()
        for j in range(steps):
            bars.add(VGroup(*[cell(w, h, th.BAD, tint=0.3) for _ in range(p + j)]).arrange(RIGHT, buff=0.03))
        bars.arrange(DOWN, buff=0.08, aligned_edge=LEFT)
        cat_t = kit.label("torch.cat: copy it all, every step", size=th.SIZE_SMALL, color=th.INK_2,
                          weight=MEDIUM)
        left = VGroup(cat_t, bars).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        buf = VGroup(*[cell(w, h) for _ in range(p + steps - 1)]).arrange(RIGHT, buff=0.03)
        for sq in buf[:p]:
            sq.set_fill(th.tint(self.color, 0.35))
        pre_t = kit.label("Preallocate: write one slice", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        right = VGroup(pre_t, buf).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        top = VGroup(left, right).arrange(RIGHT, buff=1.2, aligned_edge=UP)
        nums = kit.bullets(
            [
                r"73,664 tokens copied, at 64 KiB each.",
                r"4.5 GiB read and 4.5 GiB written.",
                r"About 8 ms at 1275 GB/s, to accomplish nothing.",
            ],
            size=th.SIZE_SMALL, width=10.0,
        )
        col = VGroup(top, nums).arrange(DOWN, buff=0.7)
        self.place(col)
        with self.narrate(
            "Next, how the cache grows. "
            "The obvious way appends with torch.cat. "
            "Every cat allocates a new tensor and copies the whole cache into it. "
            "So the copies form the same staircase as before: the quadratic, now in data movement. "
            "For the running example, that's 73,664 tokens copied, at 64 KiB each. "
            "That's 4.5 GiB read, and another 4.5 GiB written. "
            "At the measured 1275 GB/s, it's about 8 ms of copying that accomplishes nothing. "
            "Worse, the allocator fills up with blocks it can't reuse, and you run out of memory early. "
            "Instead, allocate once, at the maximum length, and write each step into a slice."
        ) as t:
            t.wait_until("The obvious way")
            self.play(FadeIn(cat_t), run_time=0.6)
            t.wait_until("Every cat allocates")
            for j in range(steps):
                self.play(FadeIn(bars[j], shift=DOWN * 0.08), run_time=0.45)
            t.wait_until("For the running example")
            self.play(FadeIn(nums[0]), run_time=0.6)
            t.wait_until("That's 4.5 GiB")
            self.play(FadeIn(nums[1]), run_time=0.6)
            t.wait_until("At the measured")
            self.play(FadeIn(nums[2]), run_time=0.6)
            t.wait_until("Instead, allocate once")
            self.play(FadeIn(right), run_time=0.7)
            for sq in buf[p:]:
                self.play(sq.animate.set_fill(th.tint(self.color, 0.35)), run_time=0.3)

    def append_code(self):
        self.slide("append-code", "Allocate once, write into a slice",
                   book_section="s-preallocate-don-t-concatenate")
        panel = kit.code(APPEND, "python", caption="HybridCache.append")
        cost = kit.prose(r"Reserved: 32 sequences $\times$ 32k tokens $\times$ 64 KiB $=$ 64 GiB.",
                         size=th.SIZE_SMALL, width=None, color=th.INK_2)
        col = VGroup(panel, cost).arrange(DOWN, buff=0.45)
        self.place(col)
        with self.narrate(
            "Here's the append method. "
            "It computes the slice this call fills, from the current length. "
            "If the slice runs past the maximum length, it raises an error, rather than growing. "
            "The scheduler decides what to do when capacity runs out, and it can only do that if the cache tells it. "
            "The writes go in place, into the preallocated buffers. "
            "And the return values are views of the filled prefix, so nothing allocates. "
            "The cost is reservation. "
            "Picture a batch of 32 sequences that might each reach 32k tokens. "
            "It reserves 64 GiB up front, even if every sequence stops at 200 tokens. "
            "Chapter 15 fixes that with paging."
        ) as t:
            self.play(FadeIn(panel.frame), *panel.reveal(1), run_time=0.7)
            t.wait_until("It computes the slice")
            self.play(*panel.reveal(2, 3), run_time=0.8)
            t.wait_until("If the slice runs")
            self.play(*panel.reveal(4, 5), run_time=0.8)
            t.wait_until("The writes go in place")
            self.play(*panel.reveal(6, 7), panel.highlight(6, 7), run_time=0.8)
            t.wait_until("And the return values")
            self.play(*panel.reveal(8, 9), panel.highlight(8, 9), run_time=0.8)
            t.wait_until("It reserves 64 GiB")
            self.play(FadeIn(cost, shift=UP * 0.1), run_time=0.8)

    # The memory arithmetic ----------------------------------------------------------

    def memory_budget(self):
        self.slide("memory-budget", "The fixed state is paid per sequence", book_section="s-the-memory-arithmetic")
        eq = kit.formula(r"\text{bytes} =", r"B \cdot L \cdot 65{,}536", r"+", r"B \cdot 154{,}927{,}104",
                         size=th.SIZE_MATH_SMALL)
        eq[1].set_color(KV_COL)
        eq[3].set_color(STATE_COL)
        rows = [("Batch 8, 32k", 16.0, 1.154, "17.1 GiB"), ("Batch 32, 4k", 8.0, 4.617, "12.6 GiB"),
                ("Batch 64, 4k", 16.0, 9.234, "25.2 GiB")]
        scale = 7.0 / 25.234
        names = VGroup(*[kit.label(r[0], size=th.SIZE_SMALL, color=th.INK_2) for r in rows])
        name_w = max(nm.width for nm in names)
        chart = VGroup()
        kvs, sts, vals = VGroup(), VGroup(), VGroup()
        for i, (_, kv, s, txt) in enumerate(rows):
            nm = names[i]
            y = -i * 0.95
            nm.move_to([name_w - nm.width / 2, y, 0])
            x0 = name_w + 0.3
            a = Rectangle(width=kv * scale, height=0.55, fill_color=KV_COL, fill_opacity=0.85, stroke_width=0)
            a.move_to([x0 + kv * scale / 2, y, 0])
            b = Rectangle(width=s * scale, height=0.55, fill_color=STATE_COL, fill_opacity=0.85, stroke_width=0)
            b.next_to(a, RIGHT, buff=0)
            v = kit.label(txt, size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
            v.move_to([x0 + 25.234 * scale + 0.3 + v.width / 2, y, 0])
            kvs.add(a)
            sts.add(b)
            vals.add(v)
        x19 = name_w + 0.3 + 19 * scale
        lim = DashedLine([x19, -2 * 0.95 - 0.3, 0], [x19, 0.55, 0], color=th.INK_3, stroke_width=th.STROKE_AXIS,
                         dash_length=0.1)
        lim_t = kit.label("19 GiB budget", size=th.SIZE_LABEL, color=th.INK_3).next_to(lim, UP, buff=0.1)
        nine = kit.label("9.2 GiB of state", size=th.SIZE_LABEL, color=STATE_COL)
        nine.next_to(sts[2], DOWN, buff=0.15)
        chart.add(names, kvs, sts, vals, lim, lim_t, nine)
        key = VGroup(swatch("KV: 64 KiB per token", KV_COL), swatch("state: 147.8 MiB per sequence", STATE_COL))
        key.arrange(RIGHT, buff=0.7)
        col = VGroup(eq, key, chart).arrange(DOWN, buff=0.55)
        self.place(col)
        with self.narrate(
            "How much memory does the cache take? "
            "Both terms are per sequence. "
            "The KV cache costs 64 KiB per token of context. "
            "The recurrent state costs a fixed 147.8 MiB, whatever the context. "
            "Chapter 2's budget leaves roughly 19 GiB for the cache. "
            "Batch 8 at 32k context needs 17.1 GiB, so it fits, with nothing to spare. "
            "Batch 32 at 4k needs 12.6 GiB, and it fits comfortably. "
            "Batch 64 at 4k needs 25.2 GiB, and it doesn't fit. "
            "Look at the pink part of that bar. "
            "At batch 64, the fixed state alone is 9.2 GiB. "
            "That's half the budget, spent before a single token of context exists."
        ) as t:
            t.wait_until("Both terms are")
            self.play(FadeIn(eq[0]), run_time=0.5)
            t.wait_until("The KV cache costs")
            self.play(FadeIn(eq[1]), FadeIn(key[0]), run_time=0.7)
            t.wait_until("The recurrent state costs")
            self.play(FadeIn(eq[2:]), FadeIn(key[1]), run_time=0.7)
            t.wait_until("Chapter 2's budget")
            self.play(FadeIn(names), Create(lim), FadeIn(lim_t), run_time=0.8)
            for i, cue in enumerate(["Batch 8 at", "Batch 32 at", "Batch 64 at"]):
                t.wait_until(cue)
                self.play(GrowFromEdge(kvs[i], LEFT), run_time=0.6)
                self.play(GrowFromEdge(sts[i], LEFT), FadeIn(vals[i]), run_time=0.5)
            t.wait_until("Look at the pink")
            self.play(Indicate(sts[2], color=STATE_COL, scale_factor=1.1), run_time=0.9)
            t.wait_until("At batch 64, the fixed")
            self.play(FadeIn(nine, shift=UP * 0.08), run_time=0.6)

    def break_even(self):
        self.slide("break-even", "The hybrid beats full attention above 788 tokens",
                   book_section="s-the-memory-arithmetic")
        st = 154927104 / 2**20  # 147.75 MiB
        ax = kit.axes([0, 2000, 250], [0, 500, 100], width=7.0, height=4.2,
                      x_label=r"\text{context } L \text{ (tokens)}", y_label=r"\text{MiB per sequence}",
                      x_numbers=[0, 500, 1000, 1500, 2000], y_numbers=[0, 100, 200, 300, 400, 500])
        full = ax.plot(lambda x: 0.25 * x, x_range=[0, 2000], color=th.INFO, stroke_width=th.STROKE_CURVE)
        hyb = ax.plot(lambda x: st + 0.0625 * x, x_range=[0, 2000], color=self.color, stroke_width=th.STROKE_CURVE)
        dot = Dot(ax.c2p(788, 197), radius=0.09, color=th.INK)
        v = DashedLine(ax.c2p(788, 0), ax.c2p(788, 197), color=th.INK_4, stroke_width=th.STROKE_AXIS)
        lab = kit.label("788 tokens", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
        lab.next_to(dot, DR, buff=0.15)
        plot = VGroup(ax, full, hyb, v, dot, lab)
        k1 = line_key("full attention: 256 KiB/token", th.INFO)
        k2 = line_key("hybrid: 64 KiB/token + 147.8 MiB", self.color)
        eq = kit.formula(r"262{,}144\,L &= 65{,}536\,L \\ &\quad + 154{,}927{,}104", size=th.SIZE_MATH_SMALL)
        at32 = kit.prose(r"At 32k: 2.14 GiB against 8.0 GiB per sequence.", size=th.SIZE_SMALL, width=4.6)
        side = VGroup(k1, k2, eq, at32).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        layout = VGroup(plot, side).arrange(RIGHT, buff=0.5)
        self.place(layout)
        with self.narrate(
            "Is the hybrid design worth that fixed state? "
            "Compare it with an all-full-attention version of this geometry. "
            "That model pays 256 KiB per token, across all 64 layers, and carries no fixed state. "
            "The hybrid pays 64 KiB per token, plus the fixed 147.8 MiB. "
            "Set the two equal, and the lines cross at 788 tokens. "
            "Below that, the fixed state costs more than the KV entries it replaces. "
            "Above it, the hybrid wins, by 192 KiB for every further token. "
            "At 32k context, the hybrid needs 2.14 GiB per sequence, against 8 GiB."
        ) as t:
            self.play(Create(ax), run_time=1.0)
            t.wait_until("That model pays")
            self.play(Create(full), FadeIn(k1), run_time=1.2)
            t.wait_until("The hybrid pays")
            self.play(Create(hyb), FadeIn(k2), run_time=1.2)
            t.wait_until("Set the two equal")
            self.play(FadeIn(eq), Create(v), FadeIn(dot, scale=0.5), FadeIn(lab), run_time=1.0)
            t.wait_until("At 32k context")
            self.play(FadeIn(at32), run_time=0.7)

    def decode_traffic(self):
        self.slide("decode-traffic", "Below 4.7k tokens, the state moves more bytes",
                   book_section="s-the-memory-arithmetic")
        state = 2 * 154927104 / 2**20  # 295.5 MiB
        ax = kit.axes([0, 10000, 1000], [0, 700, 100], width=8.0, height=4.2,
                      x_label=r"\text{context } L \text{ (tokens)}", y_label=r"\text{MiB moved per step}",
                      x_numbers=[0, 2000, 4000, 6000, 8000, 10000], y_numbers=[0, 200, 400, 600])
        kv = ax.plot(lambda x: x / 16, x_range=[0, 10000], color=KV_COL, stroke_width=th.STROKE_CURVE)
        st = ax.plot(lambda x: state, x_range=[0, 10000], color=STATE_COL, stroke_width=th.STROKE_CURVE)
        cross = Dot(ax.c2p(4728, state), radius=0.09, color=th.INK)
        v = DashedLine(ax.c2p(4728, 0), ax.c2p(4728, state), color=th.INK_4, stroke_width=th.STROKE_AXIS)
        lab = kit.label("about 4728 tokens", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
        lab.next_to(cross, DR, buff=0.15)
        p512 = VGroup(Dot(ax.c2p(512, 32), radius=0.07, color=KV_COL), Dot(ax.c2p(512, state), radius=0.07,
                                                                           color=STATE_COL))
        plot = VGroup(ax, kv, st, v, cross, lab, p512)
        k1 = line_key("KV read: 64 KiB per token", KV_COL)
        k2 = line_key("state read + write: 295.5 MiB", STATE_COL)
        n512 = kit.prose(r"At 512 tokens: 32 MiB against 295.5 MiB, about $9\times$.", size=th.SIZE_SMALL,
                         width=4.2)
        wts = kit.prose(r"Weights: 53.8 GB per step, shared by the batch.", size=th.SIZE_SMALL, width=4.2,
                        color=th.INK_3)
        side = VGroup(k1, k2, n512, wts).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        layout = VGroup(plot, side).arrange(RIGHT, buff=0.5)
        self.place(layout)
        with self.narrate(
            "Footprint isn't traffic. "
            "Per decode step, each sequence reads its whole KV cache: 64 KiB per token of context. "
            "The recurrent state is read and written in full: about 295.5 MiB per step. "
            "The two lines cross at about 4728 tokens. "
            "Below about 4.7k context, the 48 linear layers move more bytes per step than the 16 full-attention "
            "layers do. "
            "At 512 tokens, the state moves about 9 times more. "
            "The weights add 53.8 GB per step, shared across the whole batch. "
            "The hybrid design shifts memory pressure. "
            "It doesn't remove it."
        ) as t:
            self.play(Create(ax), run_time=1.0)
            t.wait_until("Per decode step")
            self.play(Create(kv), FadeIn(k1), run_time=1.2)
            t.wait_until("The recurrent state is read")
            self.play(Create(st), FadeIn(k2), run_time=1.2)
            t.wait_until("The two lines cross")
            self.play(Create(v), FadeIn(cross, scale=0.5), FadeIn(lab), run_time=0.9)
            t.wait_until("At 512 tokens")
            self.play(FadeIn(p512, scale=0.5), FadeIn(n512), run_time=0.8)
            t.wait_until("The weights add")
            self.play(FadeIn(wts), run_time=0.6)

    # The state carry -------------------------------------------------------------

    def state_carry(self):
        self.slide("state-carry", "Read the state, then write it back", book_section="s-the-state-carry")
        pre = kit.box("Prefill", sublabel="starts from no state", width=3.0)
        cache = kit.box("Cache", sublabel="state S, conv window", color=th.INFO, width=3.0)
        dec = kit.box("Decode step", sublabel="one delta-rule step", color=self.color, width=3.0)
        row = VGroup(pre, cache, dec).arrange(RIGHT, buff=1.5)
        note = kit.prose(r"Store the conv window with the convolution's output: both, or neither.",
                         size=th.SIZE_SMALL, width=None)
        col = VGroup(row, note).arrange(DOWN, buff=1.6)
        self.place(col)
        store = kit.arrow(pre, cache, text="store")
        rd = Arrow(cache.rect.get_right() + UP * 0.3, dec.rect.get_left() + UP * 0.3, buff=0.1, color=th.INK_3,
                   stroke_width=th.STROKE_BOX + 1, tip_length=0.2)
        wr = Arrow(dec.rect.get_left() + DOWN * 0.3, cache.rect.get_right() + DOWN * 0.3, buff=0.1,
                   color=th.INK_3, stroke_width=th.STROKE_BOX + 1, tip_length=0.2)
        rd_t = kit.label("read", size=th.SIZE_LABEL, color=th.INK_3).next_to(rd, UP, buff=0.1)
        wr_t = kit.label("write", size=th.SIZE_LABEL, color=th.INK_3).next_to(wr, DOWN, buff=0.1)
        steps = kit.label("every step", size=th.SIZE_LABEL, color=self.color).next_to(dec, DOWN, buff=0.25)
        with self.narrate(
            "The last piece is carrying the linear layers' state from step to step. "
            "Prefill starts from no state, runs the whole chunked recurrence, and stores the result. "
            "Each decode step reads the stored state, runs one step of the recurrence, and writes the new state "
            "back. "
            "The convolution window works the same way, and it shifts forward every step. "
            "Its update comes back with the convolution's output, so store both, or neither. "
            "The KV path is append-only, and hard to get subtly wrong. "
            "The state path is the opposite."
        ) as t:
            self.play(FadeIn(cache), run_time=0.7)
            t.wait_until("Prefill starts")
            self.play(FadeIn(pre), run_time=0.6)
            self.play(GrowArrow(store.line), FadeIn(store.label), Indicate(cache.rect, color=th.INFO), run_time=1.0)
            t.wait_until("Each decode step")
            self.play(FadeIn(dec), run_time=0.6)
            for _ in range(2):
                self.play(GrowArrow(rd), FadeIn(rd_t), run_time=0.6)
                self.play(Indicate(dec.rect, color=self.color, scale_factor=1.05), run_time=0.6)
                self.play(GrowArrow(wr), FadeIn(wr_t), run_time=0.6)
            self.play(FadeIn(steps), run_time=0.5)
            t.wait_until("Its update comes back")
            self.play(FadeIn(note, shift=UP * 0.1), run_time=0.8)

    def what_goes_wrong(self):
        self.slide("what-goes-wrong", "Advance the cursor once per forward pass", book_section="s-what-goes-wrong")
        bugs = kit.bullets(
            [
                r"Skip reading the state: the prompt is forgotten.",
                r"Skip writing it: the model repeats itself.",
                r"Skip the conv window: easy to overlook.",
                r"Advance per layer: wrong from the second decode step.",
            ],
            size=th.SIZE_SMALL, width=6.4,
        )
        panel = kit.code(ADVANCE, "python")
        layout = VGroup(bugs, panel).arrange(RIGHT, buff=0.5)
        self.place(layout)
        with self.narrate(
            "The cache's bugs produce plausible text. "
            "So the prefill-plus-decode equivalence test from chapter 8 is the one that catches them. "
            "Forget to read the state, and the model forgets the prompt after about a sentence. "
            "Forget to write it, and every step starts from the prefill state, so the model repeats itself. "
            "The convolution window has the same structure as the state, and it's easier to overlook. "
            "The subtle one is the cursor. "
            "Every layer in a pass writes at the same position, so advance the cache once, after all 64 layers. "
            "Advance it inside each layer, and the model goes wrong from the second decode step onward."
        ) as t:
            for row, cue in zip(bugs[:3], ["Forget to read", "Forget to write", "The convolution window"]):
                t.wait_until(cue)
                self.play(FadeIn(row, shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("Every layer in a pass")
            self.play(FadeIn(panel.frame), *panel.reveal(1, 2), run_time=0.8)
            self.play(*panel.reveal(3, 4), panel.highlight(3, 4), run_time=0.8)
            t.wait_until("Advance it inside")
            self.play(FadeIn(bugs[3], shift=RIGHT * 0.1), run_time=0.6)
