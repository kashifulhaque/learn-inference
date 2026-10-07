"""Chapter 20, speculative decoding.

Teaches content/chapters/20-speculative-decoding.md in its order. Every number
below was recomputed in Python (float64), then rounded as the chapter rounds
it.

Why verifying many tokens costs one weight read (A100, batch 1):
- t_memory = 53.8e9 B / 1275e9 B/s = 42.196 ms, "42.2 ms".
- Arithmetic for T = 5 tokens: 2 x 26.9e9 x 5 = 2.69e11 FLOPs;
  / 312e12 FLOP/s = 0.862 ms, "0.86 ms".
- Intensity 2.69e11 / 53.8e9 = 5.0 FLOPs per byte; ridge 161; 161 / 5 = 32.2,
  "memory bound by a factor of 32" (the chapter's figure).

The algorithm (gamma = 3, "the cat sat" -> "the cat is"): 3 tokens from one
target pass; a round yields between 1 and gamma + 1 = 4 tokens.

The three-token example, p = [0.5, 0.3, 0.2], q = [0.25, 0.6, 0.15]:
- Accept probability min(1, p/q) = [1, 0.5, 1].
- min(p, q) = [0.25, 0.3, 0.15], sum alpha = 0.7; rejection 0.3.
- (p - q)+ = [0.25, 0, 0.05], sum 0.3 = 1 - alpha; residual [0.833, 0, 0.167].
- Output: 0.25 + 0.25 = 0.5, 0.3 + 0 = 0.3, 0.15 + 0.05 = 0.2. Exactly p.
- |p - q| = [0.25, 0.3, 0.05]; D_TV = 0.6 / 2 = 0.3; alpha = 1 - 0.3 = 0.7.

Expected tokens per round, E[N+1] = (1 - a^(g+1)) / (1 - a):
- a = 0.8, g = 4: bars a^i = 1, 0.8, 0.64, 0.512, 0.4096; sum 3.3616, "3.36".
- a = 0.5: g = 3 -> 1.875 (1.88), g = 8 -> 1.996 (2.00); ratio 1.0646, "6%".
- a = 0.7: g = 3, 5, 8 -> 2.533, 2.941, 3.199; ceiling 1 / 0.3 = 3.33.
- a = 0.9: g = 3, 5, 8 -> 3.439, 4.686, 6.126 (6.13); ceiling 10.
- Plot at g = 12: a = 0.9 gives 7.46, under the y-axis top of 8.

Net speedup E[N+1] / (1 + g c):
- a = 0.8, g = 4, c = 0.05: 1 - 0.8^5 = 0.67232; / (0.2 x 1.2) = 2.8013, "2.80".
- c = 0.5: / (0.2 x 3) = 1.1205, "1.12".
- Optimum at c = 0.05 (integer search, g = 1..29): a = 0.5 -> g = 3, 1.6304
  (1.63); a = 0.7 -> g = 5 and 6 tie at 2.3529 (2.35); a = 0.8 -> g = 8,
  3.0921 (3.09); a = 0.9 -> g = 13, 4.6741 (4.67).
- Bound 1 / (1 - a) = 5 at a = 0.8.

Choosing a draft: 1.2 GB / 1275 GB/s = 0.941 ms, "0.94 ms"; 0.94 / 42.2 =
0.0223, "0.022"; 1 / 0.022 = 45.5 and 42.2 / 0.94 = 44.9, "45 times
cheaper". MTP head: one layer of 64, c about 0.016 (chapter figure).

Batching: 32 x (4 + 1) = 160 tokens against the ridge point 161.
Efficiency 3.3616 / 5 = 0.672, "67%". The ragged grid draws N + 1 for six
sequences from the truncated geometric law at a = 0.8, g = 4 (seed 20); it
illustrates, and the slide's number is the expectation.

Rolling back:
- KV: prefix 6 tokens, 4 speculative slots, 1 accepted, so the length goes
  back to 6 + 1 + 1 = 8; blocks of 4 slots, so block 2 (slots 8 to 11) held
  only rejected tokens and is freed.
- Inversion over gamma = 8: 0.9^-8 = 2.323 ("2.3"); 2^8 = 256.
- Snapshot: 147.75 MiB = 154,927,104 B, about 155e6; 2 x 155e6 / 1275e9 =
  0.2431 ms ("0.24"); / 42.2 = 0.58%, "0.6%". Batch 32: 32 x 147.75 MiB =
  4.617 GiB ("4.6"); 32 x 0.243 = 7.776 ms ("7.8"); 7.776 / 42.2 = 18.4%,
  which the chapter rounds to "roughly 19%" (kept as the chapter states it).
"""

from manim import *

import kit
from kit import blocks, course
from kit import theme as th

ACC_C = th.HL_C  # min(p, q), the accept path (\hlc in the chapter)
RES_C = th.HL_D  # (p - q)+, the residual (\hld)
TOK_C = th.HL_A  # E[N + 1] (\hla)
COST_C = th.HL_B  # 1 + gamma c (\hlb)

P = [0.5, 0.3, 0.2]
Q = [0.25, 0.6, 0.15]

CODE = """\
def accept_probability(p, q, token):
    return min(1.0, (p[token] / q[token]).item())

def residual_distribution(p, q):
    residual = torch.clamp(p - q, min=0.0)
    total = residual.sum()
    if total <= 0:          # p == q
        return p.clone()
    return residual / total
"""


def expected(a, g):
    """E[N + 1], the truncated geometric sum."""
    return (1 - a ** (g + 1)) / (1 - a)


def speedup(a, g, c):
    return expected(a, g) / (1 + g * c)


def token(text, color=None, width=1.35, height=0.62):
    """A token as a small box in the code font."""
    return kit.box(text, width=width, height=height, color=color, font=th.MONO, size=th.SIZE_SMALL)


def swatch(text, color, size=th.SIZE_LABEL):
    sq = Square(0.26, stroke_color=th.RULE_STRONG, stroke_width=th.STROKE_HAIRLINE,
                fill_color=th.tint(color, 0.45), fill_opacity=1)
    return VGroup(sq, kit.label(text, size=size, color=th.INK_2)).arrange(RIGHT, buff=0.16)


def dash_key(text, color, size=th.SIZE_LABEL):
    sq = DashedVMobject(Square(0.26, stroke_color=color, stroke_width=3), num_dashes=8)
    return VGroup(sq, kit.label(text, size=size, color=th.INK_2)).arrange(RIGHT, buff=0.16)


def line_key(text, color, size=th.SIZE_LABEL):
    rule = Line(LEFT * 0.25, RIGHT * 0.25, color=color, stroke_width=th.STROKE_CURVE)
    return VGroup(rule, kit.label(text, size=size, color=th.INK_2)).arrange(RIGHT, buff=0.16)


# The proof chart: the draft's bars against the target's dashed outlines. It
# stays on screen for three slides, so it's built once at a fixed position.
H = 5.0  # Manim units per unit of probability
BW = 1.1
BGAP = 0.7
BASE_Y = -2.75
CHART_X0 = th.SAFE_LEFT + 1.0


def proof_chart():
    xs = [CHART_X0 + i * (BW + BGAP) for i in range(3)]

    def rect(x, y0, y1, color):
        r = Rectangle(width=BW, height=(y1 - y0) * H, stroke_width=0, fill_color=th.tint(color, 0.45),
                      fill_opacity=1)
        r.move_to([x, BASE_Y + (y0 + y1) / 2 * H, 0])
        return r

    base = Line([xs[0] - BW / 2 - 0.3, BASE_Y, 0], [xs[-1] + BW / 2 + 0.3, BASE_Y, 0], color=th.INK_3,
                stroke_width=th.STROKE_AXIS)
    names = VGroup(*[kit.label(f"token {i}", size=th.SIZE_LABEL, color=th.INK_3).move_to([x, BASE_Y - 0.3, 0])
                     for i, x in enumerate(xs)])
    low = VGroup(rect(xs[0], 0, 0.25, th.INFO), rect(xs[1], 0, 0.3, th.INFO), rect(xs[2], 0, 0.15, th.INFO))
    ex_a = rect(xs[1], 0.3, 0.55, th.INFO)  # the part of the excess that refills token 0
    ex_b = rect(xs[1], 0.55, 0.6, th.INFO)  # the part that refills token 2
    outlines = VGroup()
    for x, p in zip(xs, P):
        r = Rectangle(width=BW, height=p * H, stroke_color=th.ACCENT, stroke_width=3.5, fill_opacity=0)
        r.move_to([x, BASE_Y + p * H / 2, 0])
        d = DashedVMobject(r, num_dashes=int(18 + 30 * p))
        d.set_z_index(3)
        outlines.add(d)
    key = VGroup(dash_key("target p", th.ACCENT), swatch("draft q", th.INFO)).arrange(RIGHT, buff=0.5)
    key.move_to([(xs[0] + xs[-1]) / 2, BASE_Y + 0.6 * H + 0.55, 0])
    excess = VGroup(ex_a, ex_b)
    chart = VGroup(base, names, low, excess, outlines, key)
    chart.xs, chart.base, chart.names, chart.low, chart.excess = xs, base, names, low, excess
    chart.ex_a, chart.ex_b, chart.outlines, chart.key = ex_a, ex_b, outlines, key
    chart.rect = rect
    return chart


def side_column(*mobs, buff=0.45):
    """Stacks mobjects in the area right of the proof chart."""
    col = VGroup(*mobs).arrange(DOWN, buff=buff, aligned_edge=LEFT)
    left = CHART_X0 + 2 * (BW + BGAP) + BW / 2 + 0.9
    blocks.fit(col, width=th.SAFE_RIGHT - left, height=th.CONTENT_HEIGHT - 0.2)
    col.move_to([0, th.CONTENT_CENTER_Y, 0]).align_to([left, 0, 0], LEFT)
    return col


class Lecture(kit.Lecture):
    def construct(self):
        self.title_slide(
            narration=(
                "Speculative decoding. "
                "Decode leaves the GPU's arithmetic almost idle while it reads the weights. "
                "In this lecture, you spend that idle compute on guessed tokens. "
                "And you prove that the output is distributed exactly as before."
            )
        )
        self.one_versus_five()
        self.one_round()
        self.never_loses()
        self.accept_rule()
        self.three_tokens()
        self.accept_path()
        self.residual_refills()
        self.alpha_tv()
        self.in_code()
        self.expected_tokens()
        self.saturation()
        self.net_speedup()
        self.optimal_length()
        self.choosing_draft()
        self.batching()
        self.kv_rollback()
        self.state_no_undo()
        self.snapshot_cost()
        self.chunk_commit()
        self.engine_built()
        self.course_end(
            [
                r"Verifying $\gamma + 1$ tokens costs one weight read.",
                r"Accept with $\min(1, p/q)$; resample from $(p - q)_{+}$.",
                r"$\alpha = 1 - D_{\mathrm{TV}}$; turn it off at large batch.",
                r"Roll KV back by length; commit state per chunk.",
            ],
            narration=(
                "Here's what to keep. "
                "Decode is memory bound, so checking γ plus 1 guesses costs one weight read. "
                "Accept with probability min of one and p over q, resample from the residual, and the output is "
                "exactly the target's. "
                "The gain fades at large batch sizes. "
                "Roll the KV cache back by length, and commit the recurrent state one chunk at a time. "
                "That's the end of the course. "
                "Every lever in your engine is code you wrote and a lab tested. "
                "So when an engine surprises you, you know which lever to look at."
            ),
        )

    # Why verifying many tokens costs one weight read ---------------------------------

    def one_versus_five(self):
        self.slide("one-versus-five", "Checking five tokens costs the same 42.2 ms as one",
                   book_section="s-why-verifying-many-tokens-costs-one-weight-read")
        bc = kit.bar_compare(
            [("Read 53.8 GB of weights", 42.2, self.color, "42.2 ms"),
             ("Arithmetic for 5 tokens", 0.86, th.INFO, "0.86 ms")],
            width=7.6, bar_height=0.62, gap=0.9,
        )
        riders = VGroup(*[RoundedRectangle(width=0.42, height=0.42, corner_radius=0.08, stroke_width=0,
                                           fill_color=th.tint(th.INFO, 0.6), fill_opacity=1) for _ in range(5)])
        riders.arrange(RIGHT, buff=0.12).next_to(bc.bars[0], UP, buff=0.12).align_to(bc.bars[0], LEFT).shift(
            RIGHT * 0.3)
        rider_t = kit.label("tokens on board", size=th.SIZE_LABEL, color=th.INK_3)
        rider_t.next_to(riders, RIGHT, buff=0.3)
        fig = VGroup(bc, riders, rider_t)
        t_mem = kit.formula(r"t_{\text{memory}} = \frac{53.8 \times 10^{9}\ \text{B}}{1275 \times 10^{9}\ \text{B/s}}"
                            r" = 42.2\ \text{ms}", size=th.SIZE_MATH_SMALL)
        inten = kit.prose(r"Intensity: 5.0 FLOPs per byte, against a ridge point of 161.", size=th.SIZE_SMALL,
                          width=None, color=th.INK_2)
        result = kit.formula(r"t(1\ \text{token}) = t(5\ \text{tokens}) = 42.2\ \text{ms}", size=th.SIZE_MATH)
        box = SurroundingRectangle(result, color=self.color, buff=0.18, stroke_width=th.STROKE_AXIS)
        col = VGroup(fig, t_mem, inten, VGroup(result, box)).arrange(DOWN, buff=0.5)
        self.place(col)
        with self.narrate(
            "At batch 1, a decode step reads all 53.8 GB of weights. "
            "At the measured 1275 GB/s, that takes 42.2 ms. "
            "Think of it as a bus that crosses town once per step, with one passenger. "
            "Put five tokens on board, and the arithmetic grows to only 0.86 ms. "
            "The pass does 5 FLOPs per byte, against a ridge point of 161. "
            "So checking five tokens costs the same 42.2 ms as generating one."
        ) as t:
            self.play(FadeIn(bc.labels), run_time=0.6)
            t.wait_until("At the measured")
            self.play(GrowFromEdge(bc.bars[0], LEFT), FadeIn(bc.values[0]), FadeIn(t_mem), run_time=1.2)
            t.wait_until("Think of it")
            self.play(FadeIn(riders[0], shift=DOWN * 0.15), FadeIn(rider_t), run_time=0.6)
            t.wait_until("Put five tokens")
            self.play(LaggedStart(*[FadeIn(r, shift=DOWN * 0.15) for r in riders[1:]], lag_ratio=0.25),
                      run_time=1.0)
            self.play(GrowFromEdge(bc.bars[1], LEFT), FadeIn(bc.values[1]), run_time=0.8)
            t.wait_until("The pass does")
            self.play(FadeIn(inten), run_time=0.7)
            t.wait_until("So checking five")
            self.play(FadeIn(result, shift=UP * 0.1), Create(box), run_time=0.9)

    # The algorithm ---------------------------------------------------------------------

    def one_round(self):
        self.slide("one-round", "One round: the draft guesses, the target checks once",
                   book_section="s-the-algorithm")
        lab_w, pre_w, sw, gap = 1.5, 1.9, 1.4, 0.22
        x_lab = 0.0
        x_pre = lab_w + gap + pre_w / 2
        xs = [lab_w + gap + pre_w + gap + sw / 2 + i * (sw + gap) for i in range(4)]
        ys = [1.6, 0.55, -0.5, -1.5]
        names = VGroup(*[kit.label(n, size=th.SIZE_SMALL, color=th.INK_3).move_to([x_lab, y, 0]).align_to(
            [x_lab - lab_w / 2, 0, 0], LEFT) for n, y in zip(["draft", "target", "verdict", "output"], ys)])
        prefix = token("prefix", width=pre_w).move_to([x_pre, ys[0], 0])
        guesses = VGroup(*[token(w, th.INFO, width=sw).move_to([x, ys[0], 0])
                           for w, x in zip(["the", "cat", "sat"], xs)])
        small = kit.label("3 passes", size=th.SIZE_LABEL, color=th.INK_3).move_to([xs[3], ys[0], 0])
        left = x_pre - pre_w / 2
        right = xs[3] + sw / 2
        tgt = RoundedRectangle(width=right - left, height=0.75, corner_radius=0.1, stroke_color=self.color,
                               stroke_width=th.STROKE_BOX, fill_color=th.tint(self.color, 0.1), fill_opacity=1)
        tgt.move_to([(left + right) / 2, ys[1], 0])
        one = kit.label("one pass", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM).move_to([x_pre, ys[1], 0])
        dists = VGroup(*[kit.math_label(rf"p_{i + 1}", size=th.SIZE_BODY, color=self.color).move_to([x, ys[1], 0])
                         for i, x in enumerate(xs)])
        verdicts = VGroup(
            kit.pill("accept", th.GOOD, size=th.SIZE_LABEL).move_to([xs[0], ys[2], 0]),
            kit.pill("accept", th.GOOD, size=th.SIZE_LABEL).move_to([xs[1], ys[2], 0]),
            kit.pill("reject", th.BAD, size=th.SIZE_LABEL).move_to([xs[2], ys[2], 0]),
        )
        dropped = kit.label("discarded", size=th.SIZE_LABEL, color=th.INK_4).move_to([xs[3], ys[2], 0])
        outs = VGroup(token("the", th.GOOD, width=sw).move_to([xs[0], ys[3], 0]),
                      token("cat", th.GOOD, width=sw).move_to([xs[1], ys[3], 0]),
                      token("is", self.color, width=sw).move_to([xs[2], ys[3], 0]))
        result = kit.pill("3 tokens for 1 target pass", self.color, size=th.SIZE_SMALL)
        result.move_to([(left + right) / 2, ys[3] - 1.0, 0])
        board = VGroup(names, prefix, guesses, small, tgt, one, dists, verdicts, dropped, outs, result)
        self.place(board)
        with self.narrate(
            "Here's one round, with γ = 3. "
            "The draft proposes three tokens, one at a time: the, cat, sat. "
            "The target then runs one forward pass over all three, plus the prefix. "
            "That pass gives a distribution at four positions. "
            "The target accepts the, and accepts cat. "
            "It rejects sat, samples is in its place, and discards everything after. "
            "The round emits three tokens for one target pass."
        ) as t:
            self.play(FadeIn(names[0]), FadeIn(prefix), run_time=0.6)
            t.wait_until("The draft proposes")
            for g in guesses:
                self.play(FadeIn(g, shift=RIGHT * 0.15), run_time=0.5)
            self.play(FadeIn(small), run_time=0.4)
            t.wait_until("The target then runs")
            self.play(FadeIn(names[1]), FadeIn(tgt), FadeIn(one), run_time=0.8)
            t.wait_until("That pass gives")
            self.play(LaggedStart(*[FadeIn(d, shift=UP * 0.1) for d in dists], lag_ratio=0.2), run_time=1.0)
            t.wait_until("The target accepts")
            self.play(FadeIn(names[2]), FadeIn(verdicts[0], scale=0.8), run_time=0.5)
            self.play(FadeIn(verdicts[1], scale=0.8), run_time=0.5)
            t.wait_until("It rejects sat")
            self.play(FadeIn(verdicts[2], scale=0.8), guesses[2].animate.set_opacity(0.35), run_time=0.6)
            self.play(FadeIn(dropped), dists[3].animate.set_opacity(0.3), run_time=0.6)
            t.wait_until("The round emits")
            self.play(FadeIn(names[3]), LaggedStart(*[FadeIn(o, shift=DOWN * 0.1) for o in outs], lag_ratio=0.25),
                      run_time=1.0)
            self.play(FadeIn(result, scale=0.9), run_time=0.6)

    def never_loses(self):
        self.slide("never-loses", "A round yields between 1 and γ + 1 tokens", book_section="s-the-algorithm")
        vals = [["fixed", "", "", ""], ["kept", "kept", "fixed", ""], ["kept", "kept", "kept", "bonus"]]
        cols = [[self.color, None, None, None], [th.GOOD, th.GOOD, self.color, None],
                [th.GOOD, th.GOOD, th.GOOD, th.VIOLET]]
        g = kit.grid(3, 4, cell=(1.5, 0.7), values=vals, gap=0.1)
        for r in range(3):
            for c in range(4):
                if g.cell(r, c).text is not None:
                    g.cell(r, c).text.set_z_index(2)
        heads = VGroup(*[kit.label(s, size=th.SIZE_LABEL, color=th.INK_3).next_to(g.cell(0, c), UP, buff=0.18)
                         for c, s in enumerate(["guess 1", "guess 2", "guess 3", "after"])])
        rows = VGroup(*[kit.label(s, size=th.SIZE_SMALL, color=th.INK_2).next_to(g.cell(r, 0), LEFT, buff=0.35)
                        for r, s in enumerate(["first guess rejected", "third guess rejected", "all accepted"])])
        counts = VGroup(*[kit.label(s, size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM).next_to(
            g.cell(r, 3), RIGHT, buff=0.4) for r, s in enumerate(["1 token", "3 tokens", "4 tokens"])])
        key = VGroup(swatch("accepted guess", th.GOOD), swatch("corrected token", self.color),
                     swatch("bonus token", th.VIOLET)).arrange(RIGHT, buff=0.55)
        bound = kit.formula(r"1 \;\le\; \text{tokens per round} \;\le\; \gamma + 1", size=th.SIZE_MATH)
        col = VGroup(VGroup(g, heads, rows, counts), key, bound).arrange(DOWN, buff=0.6)
        self.place(col)
        with self.narrate(
            "A round always yields at least one token. "
            "If the first guess is rejected, the corrected token is still a real sample: what plain decode gives "
            "you. "
            "If some guesses pass, you keep them, plus the correction. "
            "If all three pass, the same forward pass already computed the next distribution. "
            "So you sample a bonus token, for free. "
            "Speculation never loses ground."
        ) as t:
            self.play(FadeIn(g), FadeIn(heads), FadeIn(key), run_time=0.8)
            t.wait_until("If the first guess")
            self.play(FadeIn(rows[0]), g.fill(g.cell(0, 0), cols[0][0]), run_time=0.7)
            self.play(FadeIn(counts[0]), run_time=0.5)
            t.wait_until("If some guesses")
            self.play(FadeIn(rows[1]), *[g.fill(g.cell(1, c), cols[1][c]) for c in range(3)], run_time=0.8)
            self.play(FadeIn(counts[1]), run_time=0.5)
            t.wait_until("If all three")
            self.play(FadeIn(rows[2]), *[g.fill(g.cell(2, c), cols[2][c]) for c in range(3)], run_time=0.8)
            t.wait_until("So you sample")
            self.play(g.fill(g.cell(2, 3), th.VIOLET), FadeIn(counts[2]), run_time=0.7)
            t.wait_until("Speculation never")
            self.play(FadeIn(bound, shift=UP * 0.1), run_time=0.8)

    # The acceptance rule and its proof -------------------------------------------------

    def accept_rule(self):
        self.slide("accept-rule", "Accept a guess with probability min(1, p/q)",
                   book_section="s-the-acceptance-rule-and-its-proof")
        rule = kit.formula(r"u \sim \mathcal{U}(0, 1), \qquad \text{accept } x \text{ if }\;",
                           r"u \le \min\!\left(1, \frac{p(x)}{q(x)}\right)", size=th.SIZE_MATH)
        w = 9.0
        x0 = -w / 2
        cut = x0 + 0.5 * w
        good = Rectangle(width=0.5 * w, height=0.5, stroke_width=0, fill_color=th.tint(th.GOOD, 0.3),
                         fill_opacity=1).move_to([(x0 + cut) / 2, 0, 0])
        bad = Rectangle(width=0.5 * w, height=0.5, stroke_width=0, fill_color=th.tint(th.BAD, 0.2),
                        fill_opacity=1).move_to([(cut + x0 + w) / 2, 0, 0])
        axis = Line([x0, -0.25, 0], [x0 + w, -0.25, 0], color=th.INK_3, stroke_width=th.STROKE_AXIS)
        ticks = VGroup(*[kit.label(s, size=th.SIZE_LABEL, color=th.INK_3).move_to([x, -0.55, 0])
                         for s, x in [("0", x0), ("p/q = 0.5", cut), ("1", x0 + w)]])
        tags = VGroup(kit.label("accept", size=th.SIZE_LABEL, color=th.GOOD, weight=MEDIUM).move_to(good),
                      kit.label("reject", size=th.SIZE_LABEL, color=th.BAD, weight=MEDIUM).move_to(bad))
        cap = kit.label("token 1 of the example: the draft over-proposes it", size=th.SIZE_LABEL, color=th.INK_3)
        cap.next_to(VGroup(good, bad), UP, buff=0.55)
        draws = kit.rng(20).random(6)
        dots = VGroup(*[Dot([x0 + u * w, 0.42, 0], radius=0.09, color=th.GOOD if u <= 0.5 else th.BAD)
                        for u in draws])
        line = VGroup(cap, good, bad, axis, ticks, tags, dots)
        resid = kit.formula(r"p'(y) = \frac{\big(p(y) - q(y)\big)_{+}}{\sum_{z}\big(p(z) - q(z)\big)_{+}}",
                            size=th.SIZE_MATH)
        resid.set_color(RES_C)
        resid_t = kit.label("on a rejection: the positive part of p − q, renormalized", size=th.SIZE_LABEL,
                            color=th.INK_3)
        col = VGroup(rule, line, VGroup(resid_t, resid).arrange(DOWN, buff=0.25)).arrange(DOWN, buff=0.65)
        self.place(col)
        with self.narrate(
            "Here's the rule. "
            "At one position, the target's distribution is p, the draft's is q, and the draft proposed x. "
            "Draw u uniformly between zero and one, and accept x if u is at most min of one and p of x over q of "
            "x. "
            "So if the target likes x at least as much as the draft does, you always accept. "
            "Otherwise, you accept with probability p over q. "
            "On a rejection, you sample from the residual: the positive part of p minus q, renormalized."
        ) as t:
            t.wait_until("Draw u uniformly")
            self.play(FadeIn(rule[0]), run_time=0.7)
            self.play(FadeIn(rule[1]), run_time=0.8)
            t.wait_until("Otherwise, you accept")
            self.play(FadeIn(cap), FadeIn(good), FadeIn(bad), Create(axis), FadeIn(ticks), FadeIn(tags),
                      run_time=0.8)
            self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.3), run_time=1.4)
            t.wait_until("On a rejection")
            self.play(FadeIn(resid_t), FadeIn(resid, shift=UP * 0.1), run_time=0.9)

    def chart_parts(self, chart):
        fam = set(id(m) for m in chart.get_family())
        return tuple(m for m in self.mobjects if id(m) in fam)

    def three_tokens(self):
        self.slide("three-tokens", "A three-token vocabulary, by hand",
                   book_section="s-the-acceptance-rule-and-its-proof")
        chart = proof_chart()
        self.chart = chart
        tab = kit.table(
            [["0", "0.5", "0.25", "1"], ["1", "0.3", "0.6", "0.5"], ["2", "0.2", "0.15", "1"]],
            header=["Token", "$p$", "$q$", "Accept"], align=["l", "r", "r", "r"], size=th.SIZE_SMALL,
            col_gap=0.75, row_gap=0.32,
        )
        note = kit.prose(r"Token 1 is over-proposed, so it's accepted half the time.", size=th.SIZE_SMALL,
                         width=5.6)
        side = side_column(tab, note, buff=0.6)
        band = kit.mark(VGroup(*tab.cells[2]), buff=0.1)
        with self.narrate(
            "Take the lab's three-token vocabulary. "
            "The target puts 0.5 on token zero, 0.3 on token one, and 0.2 on token two. "
            "The draft puts 0.25, 0.6, and 0.15 on them. "
            "Tokens zero and two are under-proposed, so a guess of either is always accepted. "
            "Token one is over-proposed: the draft gives it 0.6, and the target only 0.3. "
            "So the rule accepts token one half the time."
        ) as t:
            self.play(Create(chart.base), FadeIn(chart.names), FadeIn(chart.key), FadeIn(tab.rules),
                      *[FadeIn(c) for c in tab.cells[0]], run_time=0.8)
            t.wait_until("The target puts")
            self.play(Create(chart.outlines), *[FadeIn(tab.cells[r][c]) for r in (1, 2, 3) for c in (0, 1)],
                      run_time=1.2)
            t.wait_until("The draft puts")
            self.play(*[GrowFromEdge(b, DOWN) for b in chart.low], GrowFromEdge(chart.excess, DOWN),
                      *[FadeIn(tab.cells[r][2]) for r in (1, 2, 3)], run_time=1.2)
            t.wait_until("Tokens zero and two")
            self.play(FadeIn(tab.cells[1][3]), FadeIn(tab.cells[3][3]), run_time=0.6)
            t.wait_until("Token one is over")
            self.play(FadeIn(band), Indicate(VGroup(chart.low[1], chart.excess), color=th.INFO,
                                             scale_factor=1.04), run_time=0.9)
            t.wait_until("So the rule accepts")
            self.play(FadeIn(tab.cells[2][3]), FadeIn(note), run_time=0.7)

    def accept_path(self):
        chart = self.chart
        self.slide("accept-path", "The accept path keeps min(p, q)",
                   book_section="s-the-acceptance-rule-and-its-proof", keep=self.chart_parts(chart))
        vals = VGroup(*[kit.label(s, size=th.SIZE_LABEL, color=th.INK, font=th.MONO).move_to(b)
                        for s, b in zip(["0.25", "0.3", "0.15"], chart.low)])
        for v in vals:
            v.set_z_index(4)
        excess = chart.excess
        ex_t = kit.label("0.3 rejected", size=th.SIZE_LABEL, color=th.INK_3)
        ex_t.next_to(excess, RIGHT, buff=0.12).shift(UP * 0.45)
        f1 = kit.formula(r"\Pr[\text{draw } x,\ \text{accept}]", size=th.SIZE_MATH_SMALL)
        f2 = kit.formula(r"= q(x)\,\min\!\left(1, \frac{p(x)}{q(x)}\right)", size=th.SIZE_MATH_SMALL)
        f3 = kit.formula(r"= \min\big(q(x), p(x)\big)", size=th.SIZE_MATH_SMALL, color=ACC_C)
        f4 = kit.formula(r"\alpha = 0.25 + 0.3 + 0.15 = 0.7", size=th.SIZE_MATH_SMALL)
        key = swatch("kept: min(p, q)", ACC_C)
        side = side_column(f1, f2, f3, key, f4, buff=0.4)
        with self.narrate(
            "Now prove that the rule is exact. "
            "A token comes out on one of two paths. "
            "On the accept path, the draft draws x with probability q, and keeps it with probability min of one "
            "and p over q. "
            "Multiply, and q cancels, leaving the smaller of p and q. "
            "On the chart, the draft's bar for token one is cut down to the target's height. "
            "The kept pieces sum to 0.7, the acceptance rate α. "
            "So a rejection happens 0.3 of the time."
        ) as t:
            t.wait_until("On the accept path")
            self.play(FadeIn(f1), FadeIn(f2), run_time=0.8)
            t.wait_until("Multiply, and q cancels")
            self.play(FadeIn(f3, shift=DOWN * 0.1), FadeIn(key), run_time=0.8)
            t.wait_until("On the chart")
            self.play(*[b.animate.set_fill(th.tint(ACC_C, 0.45)) for b in chart.low],
                      excess.animate.set_fill(th.tint(th.INK_4, 0.3)), run_time=1.0)
            self.play(FadeIn(ex_t), run_time=0.5)
            t.wait_until("The kept pieces")
            self.play(LaggedStart(*[FadeIn(v) for v in vals], lag_ratio=0.25), FadeIn(f4), run_time=1.0)
            t.wait_until("So a rejection")
            self.play(Indicate(ex_t, color=th.INK_2), run_time=0.8)
        chart.add(vals)
        self.min_vals = vals

    def residual_refills(self):
        chart = self.chart
        self.slide("residual-refills", "The rejected mass refills exactly the shortfall",
                   book_section="s-the-acceptance-rule-and-its-proof", keep=self.chart_parts(chart))
        xs = chart.xs
        dest_a = [xs[0], BASE_Y + 0.375 * H, 0]
        dest_b = [xs[2], BASE_Y + 0.175 * H, 0]
        va = kit.label("0.25", size=th.SIZE_LABEL, color=th.INK, font=th.MONO).move_to(dest_a).set_z_index(4)
        vb = kit.label("0.05", size=th.SIZE_LABEL, color=th.INK, font=th.MONO)
        vb.move_to([xs[2], BASE_Y + 0.2 * H + 0.25, 0]).set_z_index(4)
        f1 = kit.formula(r"\textstyle\sum_y \big(p(y) - q(y)\big)_{+} = 0.25 + 0.05 = 0.3 = 1 - \alpha",
                         size=th.SIZE_MATH_SMALL)
        f2 = kit.formula(r"(1 - \alpha)\, p'(y) = \big(p(y) - q(y)\big)_{+}", size=th.SIZE_MATH_SMALL)
        f2.set_color(RES_C)
        f3 = kit.formula(r"\Pr[y] =", r"\min(p, q)", r"+", r"\big(p - \min(p, q)\big)", r"=", r"p(y)",
                         size=th.SIZE_MATH_SMALL)
        f3[1].set_color(ACC_C)
        f3[3].set_color(RES_C)
        key = swatch("refilled: (p − q)+", RES_C)
        verdict = kit.callout("key", r"A terrible draft is slow, never wrong.", title="For every $p$ and $q$",
                              width=6.4, size=th.SIZE_SMALL)
        side = side_column(f1, f2, key, f3, verdict, buff=0.42)
        with self.narrate(
            "On the reject path, the residual supplies the rest. "
            "Its unnormalized weights are 0.25 for token zero, and 0.05 for token two. "
            "They sum to 0.3, exactly the rejection probability, and that's no coincidence. "
            "So multiplying the residual by the rejection probability cancels its normalizer. "
            "Each token gets back exactly its shortfall. "
            "Add the two paths, and every token comes out with probability p. "
            "This holds for any draft. "
            "A terrible draft makes speculation slow, never wrong."
        ) as t:
            t.wait_until("Its unnormalized")
            self.play(chart.ex_a.animate.move_to(dest_a).set_fill(th.tint(RES_C, 0.45)),
                      chart.ex_b.animate.move_to(dest_b).set_fill(th.tint(RES_C, 0.45)), FadeIn(key), run_time=1.6)
            self.play(FadeIn(va), FadeIn(vb), run_time=0.5)
            t.wait_until("They sum to 0.3")
            self.play(FadeIn(f1), run_time=0.8)
            t.wait_until("So multiplying")
            self.play(FadeIn(f2, shift=DOWN * 0.1), run_time=0.8)
            t.wait_until("Add the two paths")
            self.play(FadeIn(f3, shift=DOWN * 0.1), run_time=0.8)
            self.play(LaggedStart(*[Indicate(o, color=th.ACCENT, scale_factor=1.04) for o in chart.outlines],
                                  lag_ratio=0.2), run_time=1.2)
            t.wait_until("A terrible draft")
            self.play(FadeIn(verdict, shift=UP * 0.1), run_time=0.8)

    def alpha_tv(self):
        self.slide("alpha-tv", "The acceptance rate is one minus total variation distance",
                   book_section="s-the-acceptance-rule-and-its-proof")
        lines = [
            kit.formula(r"\alpha", r"=", r"\textstyle\sum_y \min\big(p(y), q(y)\big)"),
            kit.formula(r"\alpha", r"=", r"\tfrac{1}{2}\textstyle\sum_y \big(p(y) + q(y) - \lvert p(y) - q(y)\rvert\big)"),
            kit.formula(r"\alpha", r"=", r"\tfrac{1}{2}\big(1 + 1 - \textstyle\sum_y \lvert p(y) - q(y)\rvert\big)"),
            kit.formula(r"\alpha", r"=", r"1 - D_{\mathrm{TV}}(p, q)"),
        ]
        for l in lines[1:]:
            l[0].set_opacity(0)
        maths = VGroup(*lines).arrange(DOWN, buff=0.38, aligned_edge=LEFT)
        for l in lines[1:]:
            l.shift(RIGHT * (lines[0][1].get_left()[0] - l[1].get_left()[0]))
        box = SurroundingRectangle(lines[3][2], color=self.color, buff=0.14, stroke_width=th.STROKE_AXIS)
        ex = kit.formula(r"\lvert p - q\rvert = [0.25,\ 0.3,\ 0.05] \;\Rightarrow\; D_{\mathrm{TV}} = 0.3"
                         r" \;\Rightarrow\; \alpha = 0.7", size=th.SIZE_MATH_SMALL)
        key = kit.callout("key", r"Acceptance is a property of the draft, not a knob. A smarter sampler can't "
                                 r"raise it; a closer draft can.", width=10.5, size=th.SIZE_SMALL)
        col = VGroup(VGroup(maths, box), ex, key).arrange(DOWN, buff=0.55)
        self.place(col)
        with self.narrate(
            "The proof also tells you how often a guess survives. "
            "Start from α as the sum of the minimums. "
            "The minimum of two numbers is half their sum, minus half their gap. "
            "Both distributions sum to one, so α is one minus the total variation distance. "
            "In the example, half the summed gap is 0.3, so α is 0.7, as before. "
            "So no sampler trick raises acceptance; only a closer draft can."
        ) as t:
            t.wait_until("Start from α")
            self.play(FadeIn(lines[0]), run_time=0.7)
            t.wait_until("The minimum of two")
            self.play(FadeIn(lines[1], shift=DOWN * 0.1), lines[0].animate.set_color(th.INK_3), run_time=0.8)
            t.wait_until("Both distributions sum")
            self.play(FadeIn(lines[2], shift=DOWN * 0.1), lines[1].animate.set_color(th.INK_3), run_time=0.8)
            self.play(FadeIn(lines[3], shift=DOWN * 0.1), lines[2].animate.set_color(th.INK_3), Create(box),
                      run_time=0.9)
            t.wait_until("In the example")
            self.play(FadeIn(ex), run_time=0.8)
            t.wait_until("So no sampler")
            self.play(FadeIn(key, shift=UP * 0.1), run_time=0.8)

    def in_code(self):
        self.slide("in-code", "Clamp, then guard the zero sum", book_section="s-the-acceptance-rule-and-its-proof")
        panel = kit.code(CODE, "python")
        notes = kit.bullets(
            [
                r"Clamp before normalizing, or negative entries reach the sampler.",
                r"Guard the zero sum: when $p = q$, it would divide by zero.",
                r"Give the draft and the target the same temperature and truncation.",
            ],
            size=th.SIZE_SMALL, width=4.9,
        )
        layout = VGroup(panel, notes).arrange(RIGHT, buff=0.5)
        self.place(layout)
        with self.narrate(
            "In code, each piece is a few lines. "
            "The accept probability is the minimum of one and the ratio. "
            "The residual clamps p minus q at zero, then normalizes. "
            "Two details cause most bugs. "
            "Clamp before you normalize, or negative entries reach the sampler. "
            "And guard the zero sum: when p equals q, the residual sums to zero. "
            "That branch is never taken, but it mustn't divide by zero. "
            "Finally, give the draft and the target the same sampling settings, or acceptance collapses."
        ) as t:
            self.play(FadeIn(panel.frame), *panel.reveal(1, 2), run_time=0.8)
            t.wait_until("The residual clamps")
            self.play(*panel.reveal(4, 5, 6, 7, 8, 9), run_time=1.0)
            t.wait_until("Clamp before you")
            self.play(panel.highlight(5), FadeIn(notes[0], shift=RIGHT * 0.1), run_time=0.8)
            t.wait_until("And guard the zero")
            self.play(panel.highlight(7, 8), FadeIn(notes[1], shift=RIGHT * 0.1), run_time=0.8)
            t.wait_until("Finally, give")
            self.play(FadeIn(notes[2], shift=RIGHT * 0.1), run_time=0.8)

    # Expected tokens per round -----------------------------------------------------

    def expected_tokens(self):
        self.slide("expected-tokens", "Getting past position i takes i acceptances in a row",
                   book_section="s-expected-tokens-per-round")
        a, g = 0.8, 4
        hs = [a ** i for i in range(g + 1)]
        bars = VGroup()
        tops = VGroup()
        names = VGroup()
        bw, gap, scale = 1.0, 0.45, 3.2
        for i, h in enumerate(hs):
            r = Rectangle(width=bw, height=h * scale, stroke_width=0, fill_color=th.tint(TOK_C, 0.45),
                          fill_opacity=1)
            r.move_to([i * (bw + gap), h * scale / 2, 0])
            bars.add(r)
            tops.add(kit.label(["1", "0.8", "0.64", "0.51", "0.41"][i], size=th.SIZE_SMALL, color=th.INK,
                               font=th.MONO).next_to(r, UP, buff=0.12))
            names.add(kit.math_label(rf"i = {i}", size=th.SIZE_SMALL, color=th.INK_3).move_to(
                [i * (bw + gap), -0.32, 0]))
        base = Line([-bw / 2 - 0.2, 0, 0], [g * (bw + gap) + bw / 2 + 0.2, 0, 0], color=th.INK_3,
                    stroke_width=th.STROKE_AXIS)
        cap = kit.math_label(r"\Pr[N \ge i] = \alpha^{i}", size=th.SIZE_BODY, color=th.INK_2)
        fig = VGroup(base, bars, tops, names)
        cap.next_to(fig, UP, buff=0.35)
        given = kit.formula(r"\alpha = 0.8, \quad \gamma = 4", size=th.SIZE_MATH_SMALL)
        f1 = kit.formula(r"\mathbb{E}[N + 1]", r"= \sum_{i=0}^{\gamma} \alpha^{i}", size=th.SIZE_MATH)
        f1[0].set_color(TOK_C)
        f2 = kit.formula(r"= \frac{1 - \alpha^{\gamma + 1}}{1 - \alpha}", size=th.SIZE_MATH)
        f3 = kit.formula(r"= \frac{1 - 0.8^{5}}{0.2} = 3.36", size=th.SIZE_MATH)
        box = SurroundingRectangle(f3, color=TOK_C, buff=0.14, stroke_width=th.STROKE_AXIS)
        right = VGroup(given, f1, f2, VGroup(f3, box)).arrange(DOWN, buff=0.42, aligned_edge=LEFT)
        f2.shift(RIGHT * (f1[1].get_left()[0] - f2.get_left()[0]))
        VGroup(f3, box).shift(RIGHT * (f1[1].get_left()[0] - f3.get_left()[0]))
        layout = VGroup(VGroup(cap, fig), right).arrange(RIGHT, buff=1.0)
        self.place(layout)
        with self.narrate(
            "Next, how many tokens does a round give you? "
            "Model each guess as accepted independently, with probability α. "
            "Passing i positions takes i acceptances in a row: probability α to the i. "
            "Here α is 0.8, and γ is 4. "
            "The bars shrink geometrically: 1, then 0.8, then 0.64, and so on. "
            "A round emits one more token than it accepts, so the expected count is the sum of these bars. "
            "At these settings, that's 3.36 tokens per target pass."
        ) as t:
            t.wait_until("Model each guess")
            self.play(Create(base), FadeIn(names), run_time=0.7)
            t.wait_until("Passing i")
            self.play(FadeIn(cap), run_time=0.6)
            t.wait_until("Here α is")
            self.play(FadeIn(given), run_time=0.6)
            t.wait_until("The bars shrink")
            for b, tp in zip(bars, tops):
                self.play(GrowFromEdge(b, DOWN), FadeIn(tp), run_time=0.45)
            t.wait_until("A round emits")
            self.play(FadeIn(f1), run_time=0.7)
            self.play(FadeIn(f2, shift=DOWN * 0.1), run_time=0.7)
            t.wait_until("At these settings")
            self.play(FadeIn(f3, shift=DOWN * 0.1), Create(box), run_time=0.9)

    def saturation(self):
        self.slide("saturation", "Acceptance beats length", book_section="s-expected-tokens-per-round")
        ax = kit.axes([0, 12, 1], [0, 8, 1], width=7.6, height=4.4,
                      x_label=r"\text{draft length } \gamma", y_label=r"\mathbb{E}[N+1]",
                      x_numbers=[0, 3, 5, 8, 12], y_numbers=[0, 2, 4, 6, 8])
        specs = [(0.5, self.colors[0]), (0.7, self.colors[1]), (0.9, self.colors[2])]
        curves = VGroup()
        dots = VGroup()
        for a, c in specs:
            curves.add(ax.plot(lambda x, a=a: expected(a, x), x_range=[0, 12], color=c,
                               stroke_width=th.STROKE_CURVE))
            dots.add(VGroup(*[Dot(ax.c2p(gm, expected(a, gm)), radius=0.07, color=c) for gm in (3, 5, 8)]))
        ceils = VGroup(*[DashedLine(ax.c2p(0, 1 / (1 - a)), ax.c2p(12, 1 / (1 - a)), color=c,
                                    stroke_width=th.STROKE_AXIS, dash_length=0.1) for a, c in specs[:2]])
        top9 = kit.label("6.13 at γ = 8", size=th.SIZE_LABEL, color=specs[2][1])
        top9.next_to(ax.c2p(8, expected(0.9, 8)), UL, buff=0.1)
        plot = VGroup(ax, curves, dots, ceils, top9)
        key = VGroup(*[line_key(f"α = {a}", c) for a, c in specs]).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        ceil_f = kit.formula(r"\text{ceiling} = \frac{1}{1 - \alpha}", size=th.SIZE_MATH_SMALL)
        ceil_v = kit.label("2,  3.3,  and 10", size=th.SIZE_SMALL, color=th.INK_2)
        gain = kit.prose(r"At $\alpha = 0.5$: from $\gamma = 3$ to $8$ gains 6\%.", size=th.SIZE_SMALL, width=4.0)
        side = VGroup(key, ceil_f, ceil_v, gain).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        layout = VGroup(plot, side).arrange(RIGHT, buff=0.6)
        self.place(layout)
        with self.narrate(
            "Plot that count against the draft length. "
            "At α = 0.5, the curve flattens almost at once: going from 3 guesses to 8 gains only 6 percent. "
            "One rejection discards everything after it, so the count can never pass one over one minus α. "
            "At α = 0.7, that ceiling is about 3.3. "
            "At α = 0.9, eight guesses yield 6.13 tokens, and the ceiling is 10. "
            "So raise acceptance first, and lengthen the draft second."
        ) as t:
            self.play(Create(ax), run_time=1.0)
            t.wait_until("At α = 0.5, the curve")
            self.play(Create(curves[0]), FadeIn(key[0]), FadeIn(dots[0]), run_time=1.2)
            self.play(FadeIn(gain), run_time=0.6)
            t.wait_until("One rejection discards")
            self.play(Create(ceils[0]), FadeIn(ceil_f), run_time=0.9)
            t.wait_until("At α = 0.7")
            self.play(Create(curves[1]), FadeIn(key[1]), FadeIn(dots[1]), Create(ceils[1]), run_time=1.2)
            t.wait_until("At α = 0.9")
            self.play(Create(curves[2]), FadeIn(key[2]), FadeIn(dots[2]), FadeIn(top9), FadeIn(ceil_v),
                      run_time=1.2)

    # Net speedup, and the optimal draft length -------------------------------------------

    def net_speedup(self):
        self.slide("net-speedup", "The draft isn't free", book_section="s-net-speedup-and-the-optimal-draft-length")
        f = kit.formula(r"\operatorname{speedup} = \frac{", r"1 - \alpha^{\gamma + 1}", r"}{(", r"1 - \alpha",
                        r")(", r"1 + \gamma c", r")}", size=th.SIZE_MATH)
        f[1].set_color(TOK_C)
        f[3].set_color(TOK_C)
        f[5].set_color(COST_C)
        cost = kit.prose(r"$c$: one draft pass, as a fraction of one target pass.", size=th.SIZE_SMALL, width=None,
                         color=th.INK_3)
        given = kit.formula(r"\alpha = 0.8, \quad \gamma = 4, \quad 1 - 0.8^{5} = 0.672", size=th.SIZE_MATH_SMALL)
        bc = kit.bar_compare(
            [("Cheap draft, c = 0.05", 2.80, self.color, "2.80×"),
             ("Costly draft, c = 0.5", 1.12, th.INFO, "1.12×")],
            width=6.4, bar_height=0.6, gap=0.45, max_value=3.0, limit=1.0, limit_label="plain decode",
        )
        calc = VGroup(kit.formula(r"0.672 / (0.2 \times 1.2)", size=th.SIZE_MATH_SMALL, color=th.INK_3),
                      kit.formula(r"0.672 / (0.2 \times 3)", size=th.SIZE_MATH_SMALL, color=th.INK_3))
        for cc, v in zip(calc, bc.values):
            cc.next_to(v, RIGHT, buff=0.4)
        col = VGroup(VGroup(f, cost).arrange(DOWN, buff=0.3), given, VGroup(bc, calc)).arrange(DOWN, buff=0.55)
        self.place(col)
        with self.narrate(
            "But the draft isn't free. "
            "Let c be one draft pass, as a fraction of a target pass. "
            "A round costs one target pass, plus γ draft passes. "
            "Divide the expected tokens by that cost to get the speedup. "
            "At α = 0.8 and γ = 4, a cheap draft with c = 0.05 gives 2.8 times. "
            "Raise c to 0.5, and it falls to 1.12. "
            "An expensive draft destroys the benefit, however good it is."
        ) as t:
            self.play(FadeIn(f[0]), FadeIn(f[1:5]), FadeIn(f[6]), run_time=0.8)
            t.wait_until("Let c be")
            self.play(FadeIn(cost), run_time=0.6)
            t.wait_until("A round costs")
            self.play(FadeIn(f[5]), run_time=0.5)
            self.play(Indicate(f[5], color=COST_C), run_time=0.8)
            t.wait_until("At α = 0.8 and")
            self.play(FadeIn(given), FadeIn(bc.labels), FadeIn(bc.limit), run_time=0.6)
            self.play(GrowFromEdge(bc.bars[0], LEFT), FadeIn(bc.values[0]), FadeIn(calc[0]), run_time=1.0)
            t.wait_until("Raise c to")
            self.play(GrowFromEdge(bc.bars[1], LEFT), FadeIn(bc.values[1]), FadeIn(calc[1]), run_time=1.0)

    def optimal_length(self):
        self.slide("optimal-length", "The best draft length rises steeply with α",
                   book_section="s-net-speedup-and-the-optimal-draft-length")
        c = 0.05
        ax = kit.axes([0, 16, 1], [0, 5, 1], width=7.4, height=4.4,
                      x_label=r"\text{draft length } \gamma", y_label=r"\text{speedup},\ c = 0.05",
                      x_numbers=[0, 4, 8, 12, 16], y_numbers=[0, 1, 2, 3, 4, 5])
        specs = [(0.5, 3), (0.7, 6), (0.8, 8), (0.9, 13)]
        curves = VGroup()
        dots = VGroup()
        for (a, best), col in zip(specs, self.colors):
            curves.add(ax.plot(lambda x, a=a: speedup(a, x, c), x_range=[0.5, 16], color=col,
                               stroke_width=th.STROKE_CURVE))
            dots.add(Dot(ax.c2p(best, speedup(a, best, c)), radius=0.1, color=col).set_z_index(3))
        cap = DashedLine(ax.c2p(0, 5), ax.c2p(16, 5), color=self.colors[2], stroke_width=th.STROKE_AXIS,
                         dash_length=0.1)
        cap_t = kit.label("free-draft bound at α = 0.8: 5×", size=th.SIZE_LABEL, color=th.INK_3)
        cap_t.next_to(ax.c2p(0.4, 5), DOWN, buff=0.12).align_to(ax.c2p(0.4, 5), LEFT)
        plot = VGroup(ax, curves, dots, cap, cap_t)
        sw = [Line(LEFT * 0.22, RIGHT * 0.22, color=col, stroke_width=th.STROKE_CURVE) for col in self.colors]
        tab = kit.table(
            [[VGroup(sw[0], kit.math_label("0.5", size=th.SIZE_SMALL, color=th.INK_2)).arrange(RIGHT, buff=0.15),
              "3", "1.63×"],
             [VGroup(sw[1], kit.math_label("0.7", size=th.SIZE_SMALL, color=th.INK_2)).arrange(RIGHT, buff=0.15),
              "5 to 6", "2.35×"],
             [VGroup(sw[2], kit.math_label("0.8", size=th.SIZE_SMALL, color=th.INK_2)).arrange(RIGHT, buff=0.15),
              "8", "3.09×"],
             [VGroup(sw[3], kit.math_label("0.9", size=th.SIZE_SMALL, color=th.INK_2)).arrange(RIGHT, buff=0.15),
              "13", "4.67×"]],
            header=["$\\alpha$", "Best $\\gamma$", "Speedup"], align=["l", "r", "r"], size=th.SIZE_SMALL,
            col_gap=0.55, row_gap=0.3,
        )
        layout = VGroup(plot, tab).arrange(RIGHT, buff=0.6)
        self.place(layout)
        with self.narrate(
            "A longer draft also costs more draft passes, so there's a best length. "
            "Here's the speedup at c = 0.05. "
            "At α = 0.5, the best is 3 guesses, for 1.63 times. "
            "At α = 0.8, it's 8 guesses, for 3.09 times. "
            "At α = 0.9, it's 13 guesses, for 4.67 times. "
            "The peaks are flat, so a rough choice is fine. "
            "And even a free draft can't beat one over one minus α: 5 times, at α = 0.8."
        ) as t:
            self.play(Create(ax), FadeIn(tab.rules), *[FadeIn(x) for x in tab.cells[0]], run_time=1.0)
            t.wait_until("Here's the speedup")
            self.play(LaggedStart(*[Create(cv) for cv in curves], lag_ratio=0.3), run_time=2.0)
            for i, cue in [(0, "At α = 0.5"), (2, "At α = 0.8, it's"), (3, "At α = 0.9")]:
                t.wait_until(cue)
                self.play(FadeIn(dots[i], scale=0.5), *[FadeIn(x) for x in tab.cells[i + 1]], run_time=0.7)
                if i == 0:
                    self.play(FadeIn(dots[1], scale=0.5), *[FadeIn(x) for x in tab.cells[2]], run_time=0.6)
            t.wait_until("And even a free draft")
            self.play(Create(cap), FadeIn(cap_t), run_time=0.9)

    # Choosing a draft ------------------------------------------------------------------

    def choosing_draft(self):
        self.slide("choosing-draft", "Judge a draft by its cost and its acceptance",
                   book_section="s-choosing-a-draft")
        tab = kit.table(
            [
                ["Small model, same family", "0.022", "0.6 to 0.7", "Yes"],
                ["Multi-token prediction head", "about 0.016", "0.7 to 0.85", "Yes"],
                ["N-gram or prompt lookup", "0", "high when quoting input", "Yes"],
                ["Medusa heads", "no sequential latency", "weaker, offset by a tree", "No, as published"],
            ],
            header=["Draft", "Cost $c$", "Acceptance", "Exact?"],
            size=th.SIZE_SMALL, col_gap=0.55, row_gap=0.34,
        )
        floor =kit.formula(r"c = \frac{1.2\ \text{GB} / 1275\ \text{GB/s}}{42.2\ \text{ms}}"
                            r" = \frac{0.94\ \text{ms}}{42.2\ \text{ms}} = 0.022", size=th.SIZE_MATH_SMALL)
        cheaper = kit.pill("45 × cheaper than the target", self.color, size=th.SIZE_SMALL)
        bottom = VGroup(floor, cheaper).arrange(RIGHT, buff=0.6)
        col = VGroup(tab, bottom).arrange(DOWN, buff=0.7)
        self.place(col)
        band = kit.mark(VGroup(*tab.cells[2]), buff=0.1)
        with self.narrate(
            "Judge any draft by its cost and its acceptance. "
            "A small model from the same family is about 1.2 GB in bf16. "
            "Its memory floor is 0.94 ms, so c is 0.022. "
            "That's 45 times cheaper than the target. "
            "This model also ships a multi-token prediction head: one extra layer, with c about 0.016. "
            "Trained with the model, it accepts 70 to 85 percent of guesses: the strongest option here. "
            "N-gram lookup runs no model, so any acceptance is a win. "
            "Medusa heads draft a tree with no sequential latency, but as published, they aren't exact."
        ) as t:
            self.play(FadeIn(tab.rules), *[FadeIn(x) for x in tab.cells[0]], run_time=0.8)
            t.wait_until("A small model")
            self.play(*[FadeIn(x) for x in tab.cells[1]], run_time=0.7)
            t.wait_until("Its memory floor")
            self.play(FadeIn(floor), run_time=0.8)
            t.wait_until("That's 45 times")
            self.play(FadeIn(cheaper, scale=0.9), run_time=0.6)
            t.wait_until("This model also ships")
            self.play(*[FadeIn(x) for x in tab.cells[2]], run_time=0.7)
            t.wait_until("Trained with")
            self.play(FadeIn(band), run_time=0.6)
            t.wait_until("N-gram lookup")
            self.play(*[FadeIn(x) for x in tab.cells[3]], run_time=0.7)
            t.wait_until("Medusa heads draft")
            self.play(*[FadeIn(x) for x in tab.cells[4]], run_time=0.7)

    # Batching makes it worse ---------------------------------------------------------

    def batching(self):
        self.slide("batching", "At batch 32, the idle compute is gone", book_section="s-batching-makes-it-worse")
        bc = kit.bar_compare(
            [("Batch 1, γ = 4", 5, th.INFO, "5"), ("Batch 32, plain", 32, th.INFO, "32"),
             ("Batch 32, γ = 4", 160, self.color, "160")],
            width=4.6, bar_height=0.5, gap=0.38, max_value=170, limit=161, limit_label="ridge point: 161",
        )
        bc_t = kit.label("tokens per verification pass", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        left = VGroup(bc_t, bc).arrange(DOWN, buff=0.55, aligned_edge=LEFT)
        gen = kit.rng(20)
        counts = []
        for _ in range(6):
            n = 0
            while n < 4 and gen.random() < 0.8:
                n += 1
            counts.append(n + 1)
        g = kit.grid(6, 5, cell=(0.55, 0.42), gap=0.06)
        seqs = VGroup(*[kit.label(f"seq {r + 1}", size=th.SIZE_LABEL, color=th.INK_3).next_to(
            g.cell(r, 0), LEFT, buff=0.2) for r in range(6)])
        g_t = kit.label("ragged acceptance", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        eff = kit.formula(r"\frac{\mathbb{E}[N+1]}{\gamma + 1} = \frac{3.36}{5} = 67\%", size=th.SIZE_MATH_SMALL)
        right = VGroup(g_t, VGroup(g, seqs), eff).arrange(DOWN, buff=0.4)
        layout = VGroup(left, right).arrange(RIGHT, buff=1.0, aligned_edge=UP)
        self.place(layout)
        used = VGroup(*[g.cell(r, c) for r in range(6) for c in range(counts[r])])
        waste = VGroup(*[g.cell(r, c) for r in range(6) for c in range(counts[r], 5)])
        with self.narrate(
            "At batch 1, speculation gives 2 to 3 times. "
            "At batch 32, it often gives nothing. "
            "First, the idle compute is gone. "
            "A plain step at batch 32 already carries 32 tokens. "
            "With γ = 4, verification carries 160, right at the ridge point of 161. "
            "Second, acceptance is ragged. "
            "Each sequence is verified over five positions, but advances by its own count. "
            "The pass pays for the maximum, and delivers the average. "
            "At α = 0.8, that's 67 percent. "
            "So production engines turn speculation off at large batch sizes."
        ) as t:
            self.play(FadeIn(bc_t), FadeIn(bc.labels), FadeIn(bc.limit), GrowFromEdge(bc.bars[0], LEFT),
                      FadeIn(bc.values[0]), run_time=0.9)
            t.wait_until("A plain step")
            self.play(GrowFromEdge(bc.bars[1], LEFT), FadeIn(bc.values[1]), run_time=0.8)
            t.wait_until("With γ = 4, verification")
            self.play(GrowFromEdge(bc.bars[2], LEFT), FadeIn(bc.values[2]), run_time=1.0)
            t.wait_until("Second, acceptance")
            self.play(FadeIn(g_t), FadeIn(g), FadeIn(seqs), run_time=0.8)
            t.wait_until("Each sequence is verified")
            self.play(LaggedStart(*[g.fill(VGroup(*[g.cell(r, c) for c in range(counts[r])]), th.GOOD)
                                    for r in range(6)], lag_ratio=0.15), run_time=1.4)
            t.wait_until("The pass pays")
            self.play(g.fill(waste, th.INK_4, tint=0.3), run_time=0.8)
            t.wait_until("At α = 0.8, that's")
            self.play(FadeIn(eff, shift=UP * 0.1), run_time=0.8)
        del used

    # Rolling back a rejected token -------------------------------------------------------

    def kv_rollback(self):
        self.slide("kv-rollback", "The KV cache rolls back by length",
                   book_section="s-rolling-back-a-rejected-token")
        pages = VGroup(*[kit.grid(1, 4, cell=(0.9, 0.9), gap=0.0) for _ in range(3)]).arrange(RIGHT, buff=0.4)
        slots = [pages[b].cell(0, s) for b in range(3) for s in range(4)]
        for i in range(6):
            pages[i // 4].paint(slots[i], th.INK_2)
        names = VGroup(*[kit.label(f"block {b}", size=th.SIZE_LABEL, color=th.INK_3).next_to(pages[b], DOWN,
                                                                                            buff=0.2)
                         for b in range(3)])
        free = RoundedRectangle(width=3.2, height=1.5, corner_radius=0.12, stroke_color=th.RULE_STRONG,
                                stroke_width=th.STROKE_BOX, fill_color=th.SUNK, fill_opacity=1)
        free_t = kit.label("free list", size=th.SIZE_SMALL, color=th.INK_3)
        key = VGroup(swatch("prefix", th.INK_2), swatch("speculative", th.INFO), swatch("accepted", th.GOOD),
                     swatch("corrected", self.color), swatch("rejected", th.INK_4)).arrange(RIGHT, buff=0.45)
        rule = kit.formula(r"\text{length} \leftarrow \text{prefix} + i + 1 = 6 + 1 + 1 = 8",
                           size=th.SIZE_MATH_SMALL)
        mask = kit.prose(r"No zeroing: attention masks by \texttt{context\_len}, and the next token overwrites the "
                         r"slot.", size=th.SIZE_SMALL, width=10.5)
        row = VGroup(VGroup(pages, names), free).arrange(RIGHT, buff=0.8)
        col = VGroup(key, row, rule, mask).arrange(DOWN, buff=0.55)
        key.shift(UP * 0.4)
        self.place(col)
        free_t.move_to(free.get_top() + DOWN * 0.3)

        def cursor_at(k):
            a = slots[k - 1].get_right()
            return a

        cur = VGroup(
            Triangle(color=th.INK, fill_color=th.INK, fill_opacity=1, stroke_width=0).scale(0.12).rotate(PI),
            kit.label("length", size=th.SIZE_LABEL, color=th.INK_3),
        ).arrange(UP, buff=0.06)
        cur.next_to(pages, UP, buff=0.08).set_x(cursor_at(6)[0])
        with self.narrate(
            "Rejected guesses must also be undone in the engine's state. "
            "The 16 full-attention layers are the easy part. "
            "Here, verification wrote four speculative tokens into the paged cache. "
            "The target kept the first guess, and rejected the second. "
            "So set the length back to the prefix, plus one accepted token, plus the correction. "
            "The last block now holds only rejected tokens, so it returns to the free list. "
            "You don't zero anything: attention masks by context length, and the next token overwrites the slot."
        ) as t:
            self.play(FadeIn(pages), FadeIn(names), FadeIn(cur), FadeIn(free), FadeIn(free_t), FadeIn(key),
                      run_time=0.9)
            t.wait_until("Here, verification")
            self.play(LaggedStart(*[pages[k // 4].fill(slots[k], th.INFO) for k in range(6, 10)], lag_ratio=0.25),
                      cur.animate.set_x(cursor_at(10)[0]), run_time=1.4)
            t.wait_until("The target kept")
            self.play(pages[1].fill(slots[6], th.GOOD), run_time=0.5)
            self.play(*[pages[k // 4].fill(slots[k], th.INK_4, tint=0.3) for k in range(7, 10)], run_time=0.6)
            t.wait_until("So set the length")
            self.play(cur.animate.set_x(cursor_at(8)[0]), pages[1].fill(slots[7], self.color), FadeIn(rule),
                      run_time=1.0)
            t.wait_until("The last block")
            self.play(pages[2].animate.scale(0.7).move_to(free.get_center() + DOWN * 0.2), FadeOut(names[2]),
                      run_time=1.2)
            t.wait_until("You don't zero")
            self.play(FadeIn(mask), run_time=0.7)

    def state_no_undo(self):
        self.slide("state-no-undo", "The recurrent state has no earlier version",
                   book_section="s-rolling-back-a-rejected-token")
        kinds = ["linear", "linear", "linear", "full"]
        boxes = VGroup(*[kit.box(f"{i}", sublabel=k, width=1.25, height=1.15, color=th.INFO if k == "linear"
                                 else self.color) for i, k in enumerate(kinds)]).arrange(RIGHT, buff=0.45)
        arrows = VGroup(*[Arrow(boxes[i].get_right(), boxes[i + 1].get_left(), buff=0.05, color=th.INK_3,
                                stroke_width=3, tip_length=0.16) for i in range(3)])
        lay_t = kit.label("every fourth layer is full attention", size=th.SIZE_SMALL, color=th.INK_2,
                          weight=MEDIUM)
        no_defer = kit.prose(r"Layers 0 to 2 feed layer 3, so they must run on the guesses.", size=th.SIZE_SMALL,
                             width=5.6)
        left = VGroup(lay_t, VGroup(boxes, arrows), no_defer).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        bc = kit.bar_compare(
            [("Gate 0.9", 2.3, th.GOOD, "0.9⁻⁸ = 2.3×"), ("Gate 0.5", 256, th.BAD, "2⁸ = 256×")],
            width=3.6, bar_height=0.55, gap=0.4,
        )
        inv_t = kit.label("error amplified over 8 backward steps", size=th.SIZE_SMALL, color=th.INK_2,
                          weight=MEDIUM)
        bf = kit.prose(r"In bf16, with 8 mantissa bits, 256$\times$ destroys the state.", size=th.SIZE_SMALL,
                       width=5.0)
        right = VGroup(inv_t, bc, bf).arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        layout = VGroup(left, right).arrange(RIGHT, buff=0.9, aligned_edge=UP)
        self.place(layout)
        with self.narrate(
            "The 48 linear-attention layers are the wrinkle. "
            "Their state is overwritten in place, with no earlier version to return to. "
            "You can't defer them, because every fourth layer is full attention, and layers 0, 1, and 2 feed "
            "layer 3. "
            "Running the gated delta update backward is exact on paper, but each step divides by the gate. "
            "Over 8 steps, a slow-forgetting head amplifies rounding error 2.3 times. "
            "A fast-forgetting head amplifies it 256 times, which in bf16 destroys the state."
        ) as t:
            self.play(FadeIn(lay_t), LaggedStart(*[FadeIn(b) for b in boxes], lag_ratio=0.2), run_time=1.0)
            t.wait_until("You can't defer")
            self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.3),
                      Indicate(boxes[3], color=self.color, scale_factor=1.06), run_time=1.2)
            self.play(FadeIn(no_defer), run_time=0.6)
            t.wait_until("Running the gated")
            self.play(FadeIn(inv_t), FadeIn(bc.labels), run_time=0.6)
            t.wait_until("Over 8 steps")
            self.play(GrowFromEdge(bc.bars[0], LEFT), FadeIn(bc.values[0]), run_time=0.7)
            t.wait_until("A fast-forgetting")
            self.play(GrowFromEdge(bc.bars[1], LEFT), FadeIn(bc.values[1]), run_time=1.0)
            self.play(FadeIn(bf), run_time=0.6)

    def snapshot_cost(self):
        self.slide("snapshot-cost", "A snapshot is cheap at batch 1, costly at batch 32",
                   book_section="s-rolling-back-a-rejected-token")
        f = kit.formula(r"t_{\text{snapshot}} = \frac{2 \times 155 \times 10^{6}\ \text{B}}"
                        r"{1275 \times 10^{9}\ \text{B/s}} = 0.24\ \text{ms}", size=th.SIZE_MATH_SMALL)
        state = kit.prose(r"State: 147.8 MiB per sequence. A copy reads and writes it.", size=th.SIZE_SMALL,
                          width=None, color=th.INK_3)
        bc = kit.bar_compare(
            [("Batch 1", 0.24, th.INFO, "0.24 ms  ·  0.6%"), ("Batch 32", 7.8, self.color, "7.8 ms  ·  19%")],
            width=7.0, bar_height=0.6, gap=0.5, max_value=42.2, limit=42.2, limit_label="one 42.2 ms step",
        )
        mem = kit.callout("warning", r"Batch 32 also holds 4.6 GiB of snapshots, taken from memory you wanted "
                                     r"for KV cache.", title="And memory", width=9.0, size=th.SIZE_SMALL)
        col = VGroup(VGroup(state, f).arrange(DOWN, buff=0.3), bc, mem).arrange(DOWN, buff=0.6)
        self.place(col)
        with self.narrate(
            "Snapshot and restore works, and you can price it. "
            "The state is 147.8 MiB per sequence, and a copy reads and writes it. "
            "That takes 0.24 ms. "
            "At batch 1, that's 0.6 percent of a 42.2 ms step. "
            "At batch 32, it's 7.8 ms per round, roughly 19 percent of the step. "
            "It also takes 4.6 GiB of memory you wanted for KV cache. "
            "The cost grows with batch size."
        ) as t:
            t.wait_until("The state is")
            self.play(FadeIn(state), run_time=0.6)
            t.wait_until("That takes")
            self.play(FadeIn(f), run_time=0.8)
            t.wait_until("At batch 1")
            self.play(FadeIn(bc.labels), FadeIn(bc.limit), GrowFromEdge(bc.bars[0], LEFT), FadeIn(bc.values[0]),
                      run_time=0.9)
            t.wait_until("At batch 32")
            self.play(GrowFromEdge(bc.bars[1], LEFT), FadeIn(bc.values[1]), run_time=1.0)
            t.wait_until("It also takes")
            self.play(FadeIn(mem, shift=UP * 0.1), run_time=0.8)

    def chunk_commit(self):
        self.slide("chunk-commit", "Verify as one chunk, commit only the accepted prefix",
                   book_section="s-rolling-back-a-rejected-token")
        s0 = kit.box("S", sublabel="committed", width=2.1, height=1.5, color=th.INFO, size=th.SIZE_HEADING)
        toks = VGroup(*[token(f"t{i + 1}", th.INFO, width=1.0, height=0.62) for i in range(5)]).arrange(RIGHT,
                                                                                                    buff=0.2)
        outs = VGroup(*[kit.math_label(rf"o_{i + 1}", size=th.SIZE_BODY, color=th.INK_2) for i in range(5)])
        for o, tk in zip(outs, toks):
            o.next_to(tk, UP, buff=0.3)
        chunk_t = kit.label("one chunk of γ + 1 = 5 tokens", size=th.SIZE_LABEL, color=th.INK_3)
        chunk_t.next_to(outs, UP, buff=0.3)
        top = VGroup(s0, VGroup(chunk_t, outs, toks)).arrange(RIGHT, buff=2.4)
        brace = Brace(VGroup(*toks[:3]), DOWN, buff=0.12, color=th.GOOD)
        brace_t = kit.label("3 accepted", size=th.SIZE_LABEL, color=th.GOOD).next_to(brace, DOWN, buff=0.1)
        s1 = kit.box("S′", sublabel="after 3 tokens", width=2.1, height=1.5, color=self.color, size=th.SIZE_HEADING)
        s1.next_to(brace_t, DOWN, buff=0.35).align_to(s0, LEFT)
        key = kit.callout("key", r"Nothing is snapshotted, because nothing was overwritten. The kernel needs a "
                                 r"variable commit length: a parameter, not a redesign.", width=11.0,
                          size=th.SIZE_SMALL)
        fig = VGroup(top, brace, brace_t, s1)
        col = VGroup(fig, key).arrange(DOWN, buff=0.6)
        self.place(col)
        read = kit.arrow(s0, toks[0], text="read only", side="above")
        commit = Arrow(brace_t.get_left() + LEFT * 0.1, s1.rect.get_right() + UP * 0.2, buff=0.1, color=self.color,
                       stroke_width=4, tip_length=0.2)
        commit_t = kit.label("update", size=th.SIZE_LABEL, color=self.color)
        commit_t.next_to(commit, DOWN, buff=0.08)
        with self.narrate(
            "The chunked form avoids the problem entirely. "
            "The chunked delta rule computes a chunk's outputs from its starting state, and updates the state "
            "only at the chunk boundary. "
            "So treat the γ plus 1 speculative tokens as one chunk. "
            "Compute all their outputs against the committed state, without changing it. "
            "Once the target decides, apply the update for the accepted tokens only. "
            "Nothing was overwritten, so nothing needs a snapshot."
        ) as t:
            self.play(FadeIn(s0), run_time=0.6)
            t.wait_until("So treat the")
            self.play(FadeIn(toks), FadeIn(chunk_t), run_time=0.8)
            t.wait_until("Compute all their")
            self.play(GrowArrow(read.line), FadeIn(read.label), run_time=0.7)
            self.play(LaggedStart(*[FadeIn(o, shift=UP * 0.1) for o in outs], lag_ratio=0.15),
                      Indicate(s0.rect, color=th.INFO, scale_factor=1.04), run_time=1.2)
            t.wait_until("Once the target")
            self.play(*[toks[i].rect.animate.set_fill(th.tint(th.GOOD, 0.3)).set_stroke(th.GOOD) for i in range(3)],
                      *[toks[i].animate.set_opacity(0.35) for i in range(3, 5)],
                      *[outs[i].animate.set_opacity(0.35) for i in range(3, 5)], FadeIn(brace), FadeIn(brace_t),
                      run_time=0.8)
            self.play(GrowArrow(commit), FadeIn(commit_t), FadeIn(s1), run_time=0.9)
            t.wait_until("Nothing was overwritten")
            self.play(FadeIn(key, shift=UP * 0.1), run_time=0.8)

    def engine_built(self):
        self.slide("engine-built", "Your engine pulls all three levers",
                   book_section="s-the-engine-you-ve-built")
        steps = ["model on disk", "forward pass", "KV cache", "sampling", "CUDA and Triton", "FlashAttention",
                 "paged cache", "continuous batching", "benchmarks", "quantization", "tensor parallelism",
                 "speculation"]
        pills = [kit.pill(s, th.INK_3 if i < len(steps) - 1 else self.color, size=th.SIZE_LABEL, filled=False)
                 for i, s in enumerate(steps)]
        path = VGroup(*[VGroup(*pills[i:i + 4]).arrange(RIGHT, buff=0.2) for i in (0, 4, 8)])
        path.arrange(DOWN, buff=0.22)

        specs = [("More tokens per read", r"Continuous batching.", th.INK_3),
                 ("Fewer bytes per read", r"GQA, paging, quantization, tensor parallelism.", th.INK_3),
                 ("More than one token per read", r"Speculative decoding, this chapter.", self.color)]
        inners = [VGroup(kit.label(tt, size=th.SIZE_LABEL, color=cc, weight=MEDIUM),
                         kit.prose(bb, size=th.SIZE_SMALL, width=3.6)).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
                  for tt, bb, cc in specs]
        w = max(4.2, max(i.width for i in inners) + 0.6)
        h = max(i.height for i in inners) + 0.6
        levers = VGroup()
        for inner, (_, _, cc) in zip(inners, specs):
            rect = RoundedRectangle(width=w, height=h, corner_radius=0.12, stroke_color=cc,
                                    stroke_width=th.STROKE_BOX, fill_color=th.tint(cc, 0.06), fill_opacity=1)
            inner.move_to(rect).align_to(rect, UP).shift(DOWN * 0.3)
            inner.align_to(rect, LEFT).shift(RIGHT * 0.3)
            levers.add(VGroup(rect, inner))
        levers.arrange(RIGHT, buff=0.3)
        col = VGroup(path, levers).arrange(DOWN, buff=0.8)
        self.place(col)
        with self.narrate(
            "This is the last piece of the course, so step back. "
            "You started with a model on disk and one rented A100. "
            "Since then, you've built the forward pass, the KV cache, sampling, and your own kernels. "
            "Then came FlashAttention, a paged cache, continuous batching, honest benchmarks, quantization, and "
            "tensor parallelism. "
            "Chapter 0 named three levers against the memory bottleneck. "
            "Continuous batching serves more tokens per read of the weights. "
            "Grouped-query attention, linear attention, paging, quantization, and tensor parallelism read fewer "
            "bytes. "
            "And speculative decoding gets more than one token per read."
        ) as t:
            self.play(FadeIn(pills[0], shift=RIGHT * 0.1), run_time=0.5)
            t.wait_until("Since then")
            self.play(LaggedStart(*[FadeIn(p, shift=RIGHT * 0.1) for p in pills[1:5]], lag_ratio=0.3), run_time=1.6)
            t.wait_until("Then came")
            self.play(LaggedStart(*[FadeIn(p, shift=RIGHT * 0.1) for p in pills[5:11]], lag_ratio=0.3),
                      run_time=2.2)
            self.play(FadeIn(pills[11], scale=0.9), run_time=0.5)
            for lv, cue in zip(levers, ["Continuous batching serves", "Grouped-query attention", "And speculative"]):
                t.wait_until(cue)
                self.play(FadeIn(lv, shift=UP * 0.1), run_time=0.7)

    def course_end(self, recap, narration):
        """The closing slide for the course's last chapter.

        `closing_slide` shows only "End of the course" when there is no next
        chapter, so this builds the same card with a wrap-up panel instead.
        It keeps the kit's layout and its AI-voice disclosure footer.
        """
        self.slide("where-next", "The end of the course", book_section="s-what-goes-wrong")
        card = kit.where_next(
            recap,
            "You built an inference engine",
            r"From a model on disk to batching, paging, quantization, parallelism, and speculation. "
            r"Every lever is code you wrote and a lab tested.",
            next_label="End of the course",
            color=self.color,
        )
        disclosure = course.config().get("narration_disclosure")
        footer = None
        if disclosure:
            footer = kit.label(disclosure, size=th.SIZE_LABEL, color=th.INK_4)
            footer.move_to([0, 0, 0]).align_to([th.SAFE_LEFT, th.SAFE_BOTTOM, 0], DL)
            blocks.fit(card, th.SAFE_WIDTH, th.CONTENT_HEIGHT - footer.height - 0.35)
        self.place(card, "top", fit=footer is None)
        with self.narrate(narration) as t:
            if footer is not None:
                self.add(footer)
            self.play(FadeIn(card.left_col[0]), run_time=0.4)
            for row in card.items:
                self.play(FadeIn(row, shift=0.1 * RIGHT), run_time=0.5)
            t.wait_until("That's the end")
            self.play(FadeIn(card.right_col, shift=UP * 0.1), run_time=0.7)
