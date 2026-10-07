"""Chapter 11, sampling.

Teaches content/chapters/11-sampling.md in its order. Every number below was
recomputed with Python and numpy (float64) and matches the chapter.

The lab's five-logit row, z = (3, 2, 1, 0, -1):
- exp(z) = 20.086, 7.389, 2.718, 1.000, 0.368; sum 31.561.
- p = (0.6364, 0.2341, 0.0861, 0.0317, 0.0117).
- C = 0.6364, 0.8705, 0.9567, 0.9883, 1.
- p1 / p2 = 2.7183 = e ("2.72").
- Shift by +1 (logits 4..0): softmax unchanged (shift invariance).

Temperature:
- Odds of the top two: e^1 = 2.72 at T = 1, e^2 = 7.39 at T = 0.5,
  e^0.5 = 1.65 at T = 2.
- p(T = 0.5) = (0.8647, 0.1170, 0.0158, 0.0021, 0.0003): the largest bar the
  temperature slide draws.
- p(T = 2): tempered logits 1.5, 1.0, 0.5, 0.0, -0.5; exponentials 4.482,
  2.718, 1.649, 1.000, 0.607, sum 10.455; p = (0.4287, 0.2600, 0.1577,
  0.0956, 0.0580). Raising p to the power 1/2 and renormalizing gives the
  same row; multiplying by 1/2 and renormalizing gives p back.
- Limits: T -> 0 gives a point mass; T -> infinity gives 1/V (1/5 = 0.2 on the
  five-token row, drawn as "1/V"). log 248,320 = 12.4225 ("12.42 nats").

Overflow:
- ln(3.4028e38) = 88.7228 (float32), ln(3.3895e38) = 88.7189 (bfloat16),
  both "88.72"; ln(65,504) = 11.0899 ("11.09"); ln(1.4013e-45) = -103.279
  ("-103.3"). bfloat16 resolves 2^-8 = 0.39%.
- Stable-softmax example: the lab row plus 97, z = (100, 99, 98, 97, 96).
  z - max = (0, -1, -2, -3, -4); exp = 1, 0.368, 0.135, 0.050 (0.0498),
  0.018 (0.0183); p = the lab row's p by shift invariance. The denominator
  lies in [1, V] = [1, 248,320].

Truncation:
- Top-k, k = 2: threshold 2 (the second-largest logit); survivors
  renormalize to 0.6364 / 0.8705 = 0.7311 and 0.2689 (the chapter's top-p
  0.7 figures, the same keep-set).
- Top-p 0.7: C_0 = 0 < 0.7, C_1 = 0.6364 < 0.7 keep ranks 1 and 2;
  C_2 = 0.8705 >= 0.7 drops rank 3. Top-p 0.5: C_1 = 0.6364 >= 0.5 drops
  rank 2. Buggy C_r >= p at 0.7 keeps one token, 0.6364 of the mass.
- Min-p: ln 0.3 = -1.2040 ("1.204"), cut 3 - 1.204 = 1.796, keeps logits 3
  and 2. ln 0.05 = -2.9957 ("3.00"); at T = 1.5 the window is
  1.5 x 2.9957 = 4.4936 ("4.49") logits.

Order (top_p = 0.7, T = 2):
- Right order: C = 0.4287, 0.6886, 0.8463; C_2 = 0.6886 < 0.7 keeps rank 3.
  Renormalized (0.5065, 0.3072, 0.1863), top "0.51".
- Wrong order: top-p on p(T = 1) keeps 2; tempered logits 1.5, 1.0 give
  (0.6225, 0.3775), top "0.62"; they hold 0.4287 + 0.2600 = 0.6886 of the
  tempered mass, under 0.7.

Penalties:
- Repetition: -2 / 1.1 = -1.818 ("-1.82"); -2 x 1.1 = -2.2. Lab, penalty 2:
  [2.0, -2.0, 0.5] -> [1.0, -4.0, 0.5].
- Presence 0.5: e^-0.5 = 0.6065 ("0.607"), a 39.3% ("39%") cut.
  0.5 / 0.7 = 0.7143 ("0.714").

Gumbel-max (kit.rng(11), G = -ln(-ln U)):
- Four races (rows of U from the generator's first 4 x 5 draws); winners
  are ranks 2, 1, 1, 1. Race 1: totals 2.28, 2.36, 1.68, -1.27, -1.65.
- Next 5000 races (U clipped at 1e-12): win shares 0.6360, 0.2300, 0.0860,
  0.0346, 0.0134 against softmax 0.6364, 0.2341, 0.0861, 0.0317, 0.0117.

Cost (A100: 1275 GB/s measured copy, 312 TFLOP/s, ridge 161):
- 248,320 x 4 = 993,280 B = 0.993 MB; read + write 1,986,560 B = 1.99 MB.
- 3V = 744,960 ("745,000") FLOPs; 7.45e5 / 1.99e6 = 0.375 FLOP/B;
  161 / 0.375 = 429.3 ("429").
- 1.99e6 / 1.275e12 = 1.56 us ("1.6"); 7.45e5 / 312e12 = 2.39 ns ("2.4").
- Pipeline: chapter total "at least 26 MB"; 26e6 / 1.275e12 = 20.4 us (the
  chapter says "about 21"). The chapter's table rows sum to 28.4 MB; the
  lecture keeps the chapter's 26 MB and 21 us. 53.8e9 / 1.275e12 = 42.2 ms;
  21 us / 42.2 ms = 0.050%. Twenty launches x 5 to 10 us = 100 to 200 us,
  5 to 10 times the traffic. Batch 128: 128 x 26 MB = 3.33 GB ("3.3");
  3.3e9 / 1.275e12 = 2.59 ms ("2.6"); 2.6 / 42.2 = 6.2% ("6%").
"""

from manim import *

import kit
from kit import theme as th

Z = np.array([3.0, 2.0, 1.0, 0.0, -1.0])
LOGIT_NAMES = ["3", "2", "1", "0", "−1"]

# The chapter's maths colours: \hla temperature, \hlb the mass before a rank,
# \hlc min-p, \hld the Gumbel noise.
T_COL = th.HL_A
C_COL = th.HL_B
M_COL = th.HL_C
G_COL = th.HL_D
Z_COL = th.INFO
OFF = th.INK_4

STABLE = """\
def stable_softmax(x, dim=-1):
    shifted = x - x.max(dim=dim, keepdim=True).values
    exp = shifted.exp()
    return exp / exp.sum(dim=dim, keepdim=True)
"""

TOP_P = """\
sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
sorted_probs = sorted_logits.softmax(dim=-1)
cumulative = sorted_probs.cumsum(dim=-1)      # C_r
remove = cumulative - sorted_probs >= p       # C_{r-1} >= p
remove[..., 0] = False         # always keep the top token
"""

SEED = """\
gen = torch.Generator(device=logits.device).manual_seed(1234)
token = torch.multinomial(probs, num_samples=1, generator=gen)
"""


def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()


def vbars(values, scale, color=None, width=0.7, gap=0.32, fmt="{:.4f}", texts=None, names=None,
          size=th.SIZE_LABEL, colors=None, show_vals=True):
    """Vertical bars on a baseline at y = 0, negative values hanging below it.

    Returns a VGroup with attributes `base`, `bars`, `vals` (a list aligned
    with the bars, None where no value is printed), and `names`.
    """
    color = th.ACCENT if color is None else color
    bars, vals, names_g = VGroup(), VGroup(), VGroup()
    val_list = []
    xs = []
    lowest = 0.0
    for i, v in enumerate(values):
        x = i * (width + gap)
        xs.append(x)
        h = max(abs(v) * scale, 0.02)
        c = colors[i] if colors is not None else color
        r = Rectangle(width=width, height=h, fill_color=c, fill_opacity=0.85, stroke_width=0)
        r.move_to([x, h / 2 if v >= 0 else -h / 2, 0])
        bars.add(r)
        lowest = min(lowest, r.get_bottom()[1])
        text = None
        if show_vals:
            text = texts[i] if texts is not None else fmt.format(v)
        if text:
            t = kit.label(text, size=size, color=th.INK_2)
            t.next_to(r, UP if v >= 0 else DOWN, buff=0.08)
            vals.add(t)
            val_list.append(t)
            if v < 0:
                lowest = min(lowest, t.get_bottom()[1])
        else:
            val_list.append(None)
    base = Line([-width / 2 - 0.15, 0, 0], [xs[-1] + width / 2 + 0.15, 0, 0], color=th.INK_3,
                stroke_width=th.STROKE_AXIS)
    if names:
        for x, nm in zip(xs, names):
            t = kit.label(nm, size=th.SIZE_LABEL, color=th.INK_3)
            t.move_to([x, lowest - 0.25, 0])
            names_g.add(t)
    group = VGroup(base, bars, vals, names_g)
    group.base, group.bars, group.vals, group.names, group.val_list = base, bars, vals, names_g, val_list
    return group


def match(new, ref):
    """Scales and moves `new` so its baseline sits on `ref`'s baseline."""
    new.scale(ref.base.width / new.base.width)
    new.shift(ref.base.get_left() - new.base.get_left())
    return new


def caption(text, color=th.INK_3, size=th.SIZE_LABEL, weight=NORMAL):
    return kit.label(text, size=size, color=color, weight=weight)


def swatch(text, color, size=th.SIZE_LABEL):
    sq = Square(0.26, stroke_width=0, fill_color=color, fill_opacity=0.85)
    return VGroup(sq, kit.label(text, size=size, color=th.INK_2)).arrange(RIGHT, buff=0.15)


def aligned(lines, buff=0.38):
    """Stacks formulas made as (lhs, relation, rhs...) and aligns them on part 1."""
    col = VGroup(*lines).arrange(DOWN, buff=buff, aligned_edge=LEFT)
    x = max(l[1].get_left()[0] for l in lines)
    for l in lines:
        l.shift(RIGHT * (x - l[1].get_left()[0]))
    return col


def mass_bar(probs, keep, width=9.0, height=0.6, labels=None):
    """A horizontal bar of total mass 1, one segment per rank, kept ranks in colour."""
    segs = VGroup()
    x = 0.0
    for i, p in enumerate(probs):
        w = p * width
        col = th.ACCENT if i < keep else OFF
        tint = 0.55 if i % 2 == 0 else 0.35
        r = Rectangle(width=w, height=height, stroke_color=th.CANVAS, stroke_width=1.5,
                      fill_color=th.tint(col, tint), fill_opacity=1)
        r.move_to([x + w / 2, 0, 0])
        segs.add(r)
        x += w
    frame = Rectangle(width=width, height=height, stroke_color=th.INK_3, stroke_width=th.STROKE_AXIS,
                      fill_opacity=0).move_to([width / 2, 0, 0])
    texts = VGroup()
    if labels:
        for i, text in enumerate(labels):
            if text:
                t = kit.label(text, size=th.SIZE_LABEL, color=th.INK)
                t.move_to(segs[i])
                texts.add(t)
    g = VGroup(segs, frame, texts)
    g.segs, g.frame, g.texts, g.unit = segs, frame, texts, width
    return g


class Lecture(kit.Lecture):
    def construct(self):
        self.title_slide(
            narration=(
                "Sampling. "
                "In this lecture, you turn 248,320 raw scores into one token. "
                "You derive softmax and temperature, cut the tail three ways, and work out the order "
                "that makes the knobs mean what they say."
            )
        )
        self.logits_row()
        self.log_odds()
        self.softmax_derived()
        self.temperature()
        self.two_limits()
        self.logits_not_probs()
        self.overflow()
        self.stable_softmax()
        self.top_k_mask()
        self.top_p()
        self.top_p_bug()
        self.min_p()
        self.order()
        self.order_example()
        self.repetition_penalty()
        self.presence_frequency()
        self.gumbel()
        self.seeding()
        self.cost_softmax()
        self.cost_pipeline()
        self.closing_slide(
            [
                r"Softmax makes logit gaps into log-odds; temperature $T$ divides them.",
                r"Subtract the row maximum, in float32: overflow starts at 88.72.",
                r"Truncation is masking to $-\infty$; top-p tests $C_{r-1} < p$.",
                r"Penalties, temperature, top-k, min-p, top-p, then a Gumbel-max draw.",
            ],
            narration=(
                "Here's what to keep. "
                "Softmax turns logit gaps into log-odds, and temperature divides them. "
                "Subtract the row maximum, and run the softmax in float32. "
                "Every truncation is a mask to minus infinity, and top-p tests the mass before each token. "
                "Apply penalties, then temperature, then truncation, and draw with one argmax. "
                "Next, you write your first CUDA kernel."
            ),
            book_section="s-what-goes-wrong",
        )

    # ------------------------------------------------------------------
    # Setup and the distribution

    def logits_row(self):
        self.slide("logits-row", "248,320 raw scores, and three needs", book_section="s-before-you-start")
        hidden = kit.box("Final hidden state", sublabel="5120 numbers", width=3.0)
        head = kit.box("LM head", sublabel="(5120, 248320)", color=Z_COL, width=3.0)
        n = 11
        vals = [""] * n
        vals[8] = "…"
        strip = kit.grid(1, n, cell=(0.38, 0.62), values=[vals], gap=0.04)
        shade = kit.rng(3).normal(0, 1, n)
        for c in range(n):
            if c != 8:
                strip.paint(strip.cell(0, c), Z_COL, tint=float(np.clip(0.25 + 0.18 * shade[c], 0.05, 0.6)))
        strip_t = kit.label("248,320 logits", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        strip_s = kit.label("unnormalized, unbounded", size=th.SIZE_LABEL, color=th.INK_3)
        strip_lab = VGroup(strip_t, strip_s).arrange(DOWN, buff=0.08)
        right = VGroup(strip, strip_lab).arrange(DOWN, buff=0.22)
        row = VGroup(hidden, head, right).arrange(RIGHT, buff=0.9)
        needs = kit.bullets(
            [
                r"\textbf{A distribution}: how likely is each token?",
                r"\textbf{A knob} that trades fidelity against variety.",
                r"\textbf{A draw}, reproducible from a seed, cheap every step.",
            ],
            size=th.SIZE_BODY, width=10.5, marker_color=self.color,
        )
        col = VGroup(row, needs).arrange(DOWN, buff=0.9)
        self.place(col)
        a1 = kit.arrow(hidden, head)
        a2 = kit.arrow(head, strip)
        with self.narrate(
            "The forward pass ends with one number per vocabulary entry. "
            "The last layer multiplies the final hidden state, 5120 numbers wide, by a matrix with one column per"
            " token. "
            "Out come 248,320 logits. "
            "They're raw scores: not probabilities, and nothing constrains their scale. "
            "You need three things from them. "
            "First, a probability distribution. "
            "Second, a knob that trades fidelity against variety. "
            "Third, a draw that a seed can reproduce, cheap enough to run every step."
        ) as t:
            self.play(FadeIn(hidden), run_time=0.6)
            t.wait_until("The last layer")
            self.play(GrowArrow(a1.line), FadeIn(head), run_time=0.9)
            t.wait_until("Out come")
            self.play(GrowArrow(a2.line), LaggedStart(*[FadeIn(c, shift=RIGHT * 0.1) for c in strip.cells[0]],
                                                      lag_ratio=0.08), FadeIn(strip_t), run_time=1.2)
            t.wait_until("They're raw scores")
            self.play(FadeIn(strip_s), run_time=0.6)
            for i, cue in enumerate(["First, a probability", "Second, a knob", "Third, a draw"]):
                t.wait_until(cue)
                self.play(FadeIn(needs[i], shift=RIGHT * 0.12), run_time=0.6)

    def log_odds(self):
        self.slide("log-odds", "A gap of one logit is a factor of e", book_section="s-from-logits-to-a-distribution")
        zc = vbars(Z, 0.6, color=Z_COL, fmt="{:.0f}", width=0.7, gap=0.3)
        zc_t = caption("logits", color=Z_COL, size=th.SIZE_SMALL, weight=MEDIUM)
        zc_t.next_to(zc, UP, buff=1.0).align_to(zc, LEFT)
        left = VGroup(zc_t, zc)
        p = softmax(Z)
        pc = vbars(p, 3.0, names=LOGIT_NAMES, width=0.9, gap=0.4,
                   texts=["0.6364", "0.2341", "0.0861", "0.0317", "0.0117"])
        pc_t = caption("probabilities", color=self.color, size=th.SIZE_SMALL, weight=MEDIUM)
        pc_t.next_to(pc, UP, buff=0.3).align_to(pc, LEFT)
        right = VGroup(pc_t, pc)
        gap = Rectangle(width=2.2, height=0.1, stroke_width=0, fill_opacity=0)
        top = VGroup(left, gap, right).arrange(RIGHT, buff=0.2, aligned_edge=DOWN)
        ratio = kit.formula(r"\frac{0.6364}{0.2341} = 2.72 = e^{1}", size=th.SIZE_MATH)
        shift_note = kit.label("Add 1 to every logit: the same probabilities.", size=th.SIZE_SMALL,
                               color=th.INK_2)
        bottom = VGroup(ratio, shift_note).arrange(RIGHT, buff=1.0)
        col = VGroup(top, bottom).arrange(DOWN, buff=0.6)
        self.place(col)
        ay = pc.base.get_y() + 1.1
        arr = kit.arrow([zc.base.get_right()[0] + 0.25, ay], [pc.base.get_left()[0] - 0.25, ay],
                        text="softmax", color=self.color)
        shifted = match(vbars(Z + 1, 0.6, color=Z_COL, fmt="{:.0f}", width=0.7, gap=0.3), zc)
        top2 = VGroup(pc.bars[0], pc.bars[1])
        with self.narrate(
            "In the lab's five-logit row, the logits count down from 3 to minus 1. "
            "Softmax turns them into probabilities. "
            "The top token gets about 0.64, and the runner-up about 0.23. "
            "Their ratio is 2.72, which is e. "
            "So a gap of one logit is a factor of e in the odds. "
            "Add 1 to every logit, and nothing changes, because only the differences carry information."
        ) as t:
            self.play(FadeIn(zc_t), FadeIn(zc.base), run_time=0.5)
            t.wait_until("In the lab's")
            self.play(*[GrowFromEdge(b, DOWN if v >= 0 else UP) for b, v in zip(zc.bars, Z)], FadeIn(zc.vals),
                      run_time=1.0)
            t.wait_until("Softmax turns")
            self.play(GrowArrow(arr.line), FadeIn(arr.label), FadeIn(pc_t), FadeIn(pc.base), FadeIn(pc.names),
                      run_time=0.7)
            self.play(*[GrowFromEdge(b, DOWN) for b in pc.bars], run_time=0.9)
            t.wait_until("The top token")
            self.play(FadeIn(pc.vals), Indicate(top2, color=self.color, scale_factor=1.04), run_time=0.8)
            t.wait_until("Their ratio")
            self.play(FadeIn(ratio, shift=UP * 0.1), run_time=0.8)
            t.wait_until("Add 1 to every")
            self.play(Transform(zc, shifted), FadeIn(shift_note), run_time=1.2)
            self.play(Indicate(pc.bars, color=self.color, scale_factor=1.03), run_time=0.8)

    def softmax_derived(self):
        self.slide("softmax-derived", "Softmax is the only map that makes gaps into log-odds",
                   book_section="s-from-logits-to-a-distribution")
        l1 = kit.formula(r"\log \frac{p_i}{p_j}", r"=", r"z_i - z_j")
        l2 = kit.formula(r"p_i", r"=", r"p_j\, e^{z_i - z_j}")
        l3 = kit.formula(r"1", r"=", r"\sum_{i=1}^{V} p_i = p_j\, e^{-z_j} \sum_{i=1}^{V} e^{z_i}")
        l4 = kit.formula(r"p_j", r"=", r"\frac{e^{z_j}}{\sum_{i=1}^{V} e^{z_i}}")
        l4.set_color(self.color)
        lines = [l1, l2, l3, l4]
        maths = aligned(lines, buff=0.42)
        tags = VGroup(*[caption(s) for s in ("ask: differences are log-odds", "rearrange",
                                             "sum over every token", "solve: the softmax")])
        x_r = maths.get_right()[0] + 0.8
        for tag, line in zip(tags, lines):
            tag.move_to([x_r + tag.width / 2, line[1].get_y(), 0])
        box = SurroundingRectangle(l4, color=self.color, buff=0.16, corner_radius=0.08, stroke_width=3)
        note = kit.prose(r"Positive and normalized, without imposing either.", size=th.SIZE_SMALL, width=None,
                         color=th.INK_2)
        fig = VGroup(maths, tags, box)
        col = VGroup(fig, note).arrange(DOWN, buff=0.55)
        self.place(col)
        with self.narrate(
            "Derive softmax again, because the derivation tells you where temperature can go. "
            "Ask that logit differences be log-odds, for every pair of tokens. "
            "Rearrange, and each probability is another one times e to the gap. "
            "Sum over every token, so that the left side becomes 1. "
            "Solve for p j, and out comes the softmax. "
            "It's the only map with the property you asked for. "
            "And it comes out positive and normalized, without your imposing either."
        ) as t:
            for line, tag, cue in zip(lines, tags, ["Ask that", "Rearrange", "Sum over", "Solve for"]):
                t.wait_until(cue)
                self.play(FadeIn(line, shift=DOWN * 0.1), FadeIn(tag), run_time=0.8)
            t.wait_until("It's the only map")
            self.play(Create(box), run_time=0.7)
            t.wait_until("And it comes out")
            self.play(FadeIn(note), run_time=0.6)

    # ------------------------------------------------------------------
    # Temperature

    def temperature(self):
        self.slide("temperature", "Temperature divides every gap", book_section="s-temperature")
        f = kit.formula(r"p_i(", r"T", r") = \frac{e^{z_i / ", r"T", r"}}{\sum_{j} e^{z_j / ", r"T", r"}}")
        for k in (1, 3, 5):
            f[k].set_color(T_COL)
        scale = 3.0
        ref = vbars(softmax(Z / 0.5), scale, fmt="{:.2f}", names=LOGIT_NAMES, width=0.8, gap=0.35)
        ref_t = caption("logits", size=th.SIZE_LABEL)
        ref_t.next_to(ref.names, LEFT, buff=0.3)
        lo = kit.formula(r"\log \frac{p_i(T)}{p_j(T)} = \frac{z_i - z_j}{T}", size=th.SIZE_MATH_SMALL)
        for k in (6, 12, 21):
            lo[0][k].set_color(T_COL)
        zoom = VGroup(
            kit.label("Below 1: every gap stretches, sharper.", size=th.SIZE_SMALL, color=th.INK_2),
            kit.label("Above 1: every gap shrinks, flatter.", size=th.SIZE_SMALL, color=th.INK_2),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        t_slot = kit.label("T = 0.50", size=th.SIZE_HEADING, color=T_COL, weight=MEDIUM)
        o_slot = kit.label("top two: 7.39 to 1", size=th.SIZE_SMALL, color=th.INK)
        side = VGroup(t_slot, o_slot, lo, zoom).arrange(DOWN, buff=0.42, aligned_edge=LEFT)
        body = VGroup(VGroup(ref, ref_t), side).arrange(RIGHT, buff=1.2, aligned_edge=DOWN)
        col = VGroup(f, body).arrange(DOWN, buff=0.5)
        self.place(col)
        T = ValueTracker(1.0)
        t_x, t_y = t_slot.get_left()[0], t_slot.get_y()
        o_x, o_y = o_slot.get_left()[0], o_slot.get_y()

        def bars():
            return match(vbars(softmax(Z / T.get_value()), scale, fmt="{:.2f}", names=LOGIT_NAMES, width=0.8,
                               gap=0.35), ref)

        def t_read():
            m = kit.label(f"T = {T.get_value():.2f}", size=th.SIZE_HEADING, color=T_COL, weight=MEDIUM)
            return m.move_to([t_x + m.width / 2, t_y, 0])

        def o_read():
            m = kit.label(f"top two: {np.exp(1 / T.get_value()):.2f} to 1", size=th.SIZE_SMALL, color=th.INK)
            return m.move_to([o_x + m.width / 2, o_y, 0])

        chart = always_redraw(bars)
        tr = always_redraw(t_read)
        orr = always_redraw(o_read)
        with self.narrate(
            "Temperature answers the second need. "
            "Divide every logit by T before the softmax. "
            "At a temperature of 1, the top two keep odds of 2.72 to 1. "
            "Divide by 0.5 instead, and the gap doubles, so the odds rise to 7.39 to 1. "
            "Divide by 2, and the gap halves, so the odds fall to 1.65 to 1. "
            "In log-odds terms, temperature divides every gap by T. "
            "It's a zoom on the gaps between logits."
        ) as t:
            t.wait_until("Divide every logit")
            self.play(FadeIn(f), run_time=0.8)
            t.wait_until("At a temperature of 1")
            self.play(FadeIn(chart), FadeIn(ref_t), FadeIn(tr), FadeIn(orr), run_time=0.8)
            t.wait_until("Divide by 0.5")
            self.play(T.animate.set_value(0.5), run_time=2.0)
            t.wait_until("Divide by 2, and")
            self.play(T.animate.set_value(2.0), run_time=2.5)
            t.wait_until("In log-odds terms")
            self.play(FadeIn(lo), run_time=0.8)
            t.wait_until("It's a zoom")
            self.play(FadeIn(zoom, shift=UP * 0.1), run_time=0.7)

    def two_limits(self):
        self.slide("two-limits", "Temperature 0 is greedy; infinity is noise", book_section="s-temperature")
        states = [
            ([1, 0, 0, 0, 0], r"T \to 0^{+}", "point mass: the argmax", ["1", "", "", "", ""]),
            (list(softmax(Z)), r"T = 1", "the model's distribution", ["0.64", "0.23", "", "", ""]),
            ([0.2] * 5, r"T \to \infty", "uniform: 1/V each", ["", "", "", "", ""]),
        ]
        charts = VGroup()
        for vals, tex, sub, texts in states:
            c = vbars(vals, 2.0, width=0.55, gap=0.2, fmt="{:.2f}", texts=texts)
            head = kit.math_label(tex, size=th.SIZE_BODY, color=T_COL)
            foot = caption(sub)
            g = VGroup(head, c, foot).arrange(DOWN, buff=0.25)
            head.align_to(c.base, DOWN).shift(UP * 2.5)
            g.chart, g.head, g.foot = c, head, foot
            charts.add(g)
        charts.arrange(RIGHT, buff=1.1, aligned_edge=DOWN)
        ent = kit.prose(r"Entropy rises steadily: from 0 to $\log 248{,}320 \approx 12.42$ nats.",
                        size=th.SIZE_SMALL, width=None)
        greedy = kit.callout(
            "note",
            r"Greedy maximizes each token, not the sequence, and on open-ended text it loops.",
            title="Temperature 0 is special-cased to a plain argmax", width=11.5, size=th.SIZE_SMALL,
        )
        col = VGroup(charts, ent, greedy).arrange(DOWN, buff=0.5)
        self.place(col)
        with self.narrate(
            "As T goes to zero, the distribution becomes a point mass on the argmax: greedy decoding. "
            "As T goes to infinity, every token gets 1 over V. "
            "In between, entropy rises steadily, up to about 12.42 nats. "
            "The engine special-cases a temperature of zero to a plain argmax, because the float arithmetic "
            "toward the limit doesn't survive. "
            "But greedy maximizes each token, not the whole sequence, and on open-ended text it falls into loops."
        ) as t:
            self.play(FadeIn(charts[1]), run_time=0.8)
            t.wait_until("As T goes to zero")
            self.play(TransformFromCopy(charts[1].chart, charts[0].chart), FadeIn(charts[0].head),
                      FadeIn(charts[0].foot), run_time=1.2)
            t.wait_until("As T goes to infinity")
            self.play(TransformFromCopy(charts[1].chart, charts[2].chart), FadeIn(charts[2].head),
                      FadeIn(charts[2].foot), run_time=1.2)
            t.wait_until("In between")
            self.play(FadeIn(ent), run_time=0.7)
            t.wait_until("The engine special-cases")
            self.play(FadeIn(greedy.panel), FadeIn(greedy[1]), FadeIn(greedy.tag), run_time=0.7)
            t.wait_until("But greedy maximizes")
            self.play(FadeIn(greedy.content), run_time=0.7)

    def logits_not_probs(self):
        self.slide("logits-not-probs", "Temperature acts on logits, not probabilities", book_section="s-temperature")
        p = softmax(Z)
        p2 = softmax(Z / 2)

        def column(title, tex, vals, verdict, color):
            head = kit.label(title, size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
            f = kit.formula(tex, size=th.SIZE_MATH_SMALL)
            c = vbars(vals, 3.0, names=LOGIT_NAMES, width=0.8, gap=0.35,
                      texts=[f"{vals[0]:.4f}", f"{vals[1]:.4f}", "", "", ""])
            pill = kit.pill(verdict, color, size=th.SIZE_SMALL)
            g = VGroup(head, f, c, pill).arrange(DOWN, buff=0.38)
            g.head, g.f, g.c, g.pill = head, f, c, pill
            return g

        left = column("Power 1/T, then renormalize", r"p_i^{1/T} \propto e^{z_i / T}", p2, "temperature", th.GOOD)
        right = column("Multiply by 1/T, then renormalize",
                       r"\frac{p_i / T}{\sum_j p_j / T} = p_i", p, "a no-op", th.BAD)
        both = VGroup(left, right).arrange(RIGHT, buff=1.6, aligned_edge=UP)
        note = caption("Both at T = 2", size=th.SIZE_SMALL)
        col = VGroup(note, both).arrange(DOWN, buff=0.35)
        self.place(col)
        start = match(vbars(p, 3.0, names=LOGIT_NAMES, width=0.8, gap=0.35,
                            texts=["0.6364", "0.2341", "", "", ""]), left.c)
        with self.narrate(
            "Only one of two similar operations is temperature. "
            "Raise each probability to the power 1 over T, and renormalize. "
            "Each probability is proportional to e to its logit, so that's the softmax of z over T. "
            "At a temperature of 2, the top token drops to about 0.43. "
            "Now multiply by 1 over T instead, and renormalize. "
            "The factor appears above and below the line, and cancels. "
            "The distribution comes back unchanged, so check this first if your knob does nothing."
        ) as t:
            self.play(FadeIn(note), run_time=0.5)
            t.wait_until("Raise each")
            self.play(FadeIn(left.head), FadeIn(start), run_time=0.8)
            t.wait_until("Each probability is")
            self.play(FadeIn(left.f), run_time=0.7)
            t.wait_until("At a temperature of 2")
            self.play(ReplacementTransform(start, left.c), run_time=1.2)
            self.play(FadeIn(left.pill, scale=0.9), run_time=0.5)
            t.wait_until("Now multiply")
            self.play(FadeIn(right.head), FadeIn(right.c), run_time=0.8)
            t.wait_until("The factor appears")
            self.play(FadeIn(right.f), run_time=0.7)
            t.wait_until("The distribution comes back")
            self.play(Indicate(right.c.bars, color=th.BAD, scale_factor=1.03), FadeIn(right.pill, scale=0.9),
                      run_time=0.9)

    # ------------------------------------------------------------------
    # Shift invariance and the max subtraction

    def overflow(self):
        self.slide("overflow", "One logit above 88.72 turns the row into NaN",
                   book_section="s-shift-invariance-and-the-max-subtraction")
        f = kit.formula(r"\operatorname{softmax}(z + c\mathbf{1})_i = \frac{e^{c}\, e^{z_i}}{e^{c} \sum_j e^{z_j}}"
                        r" = \operatorname{softmax}(z)_i", size=th.SIZE_MATH_SMALL)
        w = 11.0
        lo_v, hi_v = -120.0, 120.0

        def fx(v):
            return (v - lo_v) / (hi_v - lo_v) * w - w / 2

        axis = Line([fx(lo_v), 0, 0], [fx(hi_v), 0, 0], color=th.INK_3, stroke_width=th.STROKE_AXIS)
        ticks = VGroup()
        for v in (-100, -50, 0, 50, 100):
            tk = Line([fx(v), -0.08, 0], [fx(v), 0.08, 0], color=th.INK_3, stroke_width=th.STROKE_AXIS)
            num = kit.label(str(v).replace("-", "−"), size=th.SIZE_LABEL, color=th.INK_4)
            num.next_to(tk, DOWN, buff=0.1)
            ticks.add(VGroup(tk, num))
        x_lab = caption("logit x")
        x_lab.next_to(axis, RIGHT, buff=0.2)

        def band(a, b, y, color, tint):
            r = Rectangle(width=fx(b) - fx(a), height=0.24, stroke_width=0, fill_color=th.tint(color, tint),
                          fill_opacity=1)
            return r.move_to([(fx(a) + fx(b)) / 2, y, 0])

        under = band(lo_v, -103.3, 0.3, th.GOOD, 0.5)
        under_t = kit.label("float32: 0 below −103.3", size=th.SIZE_LABEL, color=th.GOOD)
        under_t.next_to(under, UP, buff=0.12).align_to(under, LEFT)
        b16 = band(11.09, hi_v, 0.3, th.BAD, 0.3)
        b16_t = kit.label("float16 overflows above 11.09", size=th.SIZE_LABEL, color=th.BAD)
        b16_t.next_to(b16, UP, buff=0.12).align_to(b16, LEFT)
        b32 = band(88.72, hi_v, 1.2, th.BAD, 0.6)
        b32_t = kit.label("float32 and bfloat16: above 88.72", size=th.SIZE_LABEL, color=th.BAD)
        b32_t.next_to(b32, LEFT, buff=0.2)
        dot = Dot([fx(100), 0, 0], radius=0.1, color=th.BAD)
        dot_t = kit.label("a logit of 100: inf / inf = NaN", size=th.SIZE_SMALL, color=th.BAD, weight=MEDIUM)
        dot_t.move_to([fx(100) - dot_t.width / 2 + 0.3, -0.85, 0])
        line_fig = VGroup(axis, ticks, x_lab, under, under_t, b16, b16_t, b32, b32_t, dot, dot_t)
        note = kit.prose(r"Range is set by the exponent field: 8 bits in float32 and bfloat16, 5 in float16.",
                         size=th.SIZE_SMALL, width=None, color=th.INK_2)
        col = VGroup(f, line_fig, note).arrange(DOWN, buff=0.6)
        self.place(col)
        with self.narrate(
            "Adding a constant to every logit leaves softmax unchanged, because it cancels above and below the "
            "line. "
            "In floating point, though, the constant matters. "
            "For float32, the exponential overflows above 88.72. "
            "The bfloat16 format shares that 8-bit exponent, so it overflows at the same place. "
            "And float16 overflows at 11.09, while a logit of 11 is ordinary. "
            "Underflow is harmless: below minus 103.3, the exponential flushes to zero. "
            "Without a shift, a logit of 100 makes infinity over infinity, which is NaN."
        ) as t:
            self.play(FadeIn(f), run_time=0.9)
            t.wait_until("In floating point")
            self.play(Create(axis), FadeIn(ticks), FadeIn(x_lab), run_time=0.9)
            t.wait_until("For float32")
            self.play(GrowFromEdge(b32, LEFT), FadeIn(b32_t), run_time=0.8)
            t.wait_until("The bfloat16 format")
            self.play(Indicate(b32_t, color=th.BAD, scale_factor=1.05), FadeIn(note), run_time=0.8)
            t.wait_until("And float16")
            self.play(GrowFromEdge(b16, LEFT), FadeIn(b16_t), run_time=0.8)
            t.wait_until("Underflow is harmless")
            self.play(GrowFromEdge(under, RIGHT), FadeIn(under_t), run_time=0.8)
            t.wait_until("Without a shift")
            self.play(FadeIn(dot, scale=0.5), FadeIn(dot_t), run_time=0.7)
            self.play(Flash(dot, color=th.BAD), run_time=0.6)

    def stable_softmax(self):
        self.slide("stable-softmax", "Subtract the maximum, and nothing overflows",
                   book_section="s-shift-invariance-and-the-max-subtraction")
        panel = kit.code(STABLE, "python")
        rows = [
            ["100", "99", "98", "97", "96"],
            ["0", "−1", "−2", "−3", "−4"],
            ["1", "0.368", "0.135", "0.050", "0.018"],
            ["0.6364", "0.2341", "0.0861", "0.0317", "0.0117"],
        ]
        g = kit.grid(4, 5, cell=(1.25, 0.56), values=rows, row_labels=["z", "z − max", "exp", "p"])
        sum_pill = kit.pill("the sum lies between 1 and 248,320", self.color, size=th.SIZE_SMALL)
        right = VGroup(g, sum_pill).arrange(DOWN, buff=0.45)
        layout = VGroup(panel, right).arrange(DOWN, buff=0.55)
        self.place(layout)
        with self.narrate(
            "So subtract the row maximum. "
            "The stable softmax is three lines. "
            "Take a row of the lab's five logits plus 97. "
            "Subtract the maximum, and every exponent is at most zero. "
            "Every term lands between zero and 1, and the largest is exactly 1. "
            "So the sum lies between 1 and 248,320, and can't overflow. "
            "The probabilities are the same as before."
        ) as t:
            t.wait_until("The stable softmax")
            self.play(FadeIn(panel.frame), *panel.reveal(1), run_time=0.7)
            t.wait_until("Take a row")
            self.play(FadeIn(g.row_labels), FadeIn(g.row(0)), run_time=0.8)
            t.wait_until("Subtract the maximum")
            self.play(*panel.reveal(2), panel.highlight(2), FadeIn(g.row(1), shift=DOWN * 0.1), run_time=0.9)
            t.wait_until("Every term lands")
            self.play(*panel.reveal(3), FadeIn(g.row(2), shift=DOWN * 0.1), run_time=0.9)
            self.play(g.fill(g.cell(2, 0), self.color), run_time=0.5)
            t.wait_until("So the sum lies")
            self.play(*panel.reveal(4), FadeIn(sum_pill, scale=0.9), run_time=0.8)
            t.wait_until("The probabilities are")
            self.play(FadeIn(g.row(3), shift=DOWN * 0.1), g.fill(g.row(3), self.color, tint=0.15), run_time=0.9)

    # ------------------------------------------------------------------
    # Truncation

    def top_k_mask(self):
        self.slide("top-k-mask", "Every truncation is a mask to −∞",
                   book_section="s-truncation-one-operation-three-rules")
        f = kit.formula(r"\tilde z_i = \begin{cases} z_i & i \in S \\ -\infty & i \notin S \end{cases}",
                        size=th.SIZE_MATH_SMALL)
        zc = vbars(Z, 0.65, color=Z_COL, fmt="{:.0f}", width=0.7, gap=0.3)
        zc_t = caption("logits, k = 2; dashed: the 2nd-largest", color=Z_COL, size=th.SIZE_SMALL, weight=MEDIUM)
        zc_t.next_to(zc, UP, buff=0.3).align_to(zc, LEFT)
        masked_cols = [Z_COL, Z_COL, OFF, OFF, OFF]
        zm = vbars([3.0, 2.0, -0.01, -0.01, -0.01], 0.65, fmt="{:.0f}", width=0.7, gap=0.3, colors=masked_cols,
                   texts=["3", "2", "−∞", "−∞", "−∞"])
        p = softmax(Z)
        pc = vbars(p, 2.6, names=LOGIT_NAMES, width=0.78, gap=0.3)
        q = np.array([p[0], p[1], 0, 0, 0]) / (p[0] + p[1])
        pq = vbars(q, 2.6, names=LOGIT_NAMES, width=0.78, gap=0.3, texts=["0.7311", "0.2689", "0", "0", "0"])
        pc_t = caption("after the softmax", color=self.color, size=th.SIZE_SMALL, weight=MEDIUM)
        pc_t.next_to(pq, UP, buff=0.3).align_to(pq, LEFT)
        gap = Rectangle(width=1.8, height=0.1, stroke_width=0, fill_opacity=0)
        top = VGroup(VGroup(zc_t, zc), gap, VGroup(pc_t, pq)).arrange(RIGHT, buff=0.2, aligned_edge=DOWN)
        weak = kit.prose(r"Top-k's weakness: $k$ is fixed, while the model's confidence isn't.",
                         size=th.SIZE_SMALL, width=None, color=th.INK_2)
        col = VGroup(f, top, weak).arrange(DOWN, buff=0.5)
        self.place(col)
        match(zm, zc)
        match(pc, pq)
        thr = DashedLine(zc.base.get_left(), zc.base.get_right(), color=th.INK, stroke_width=th.STROKE_AXIS)
        thr.set_y(zc.bars[1].get_top()[1])
        ay = pq.base.get_y() + 1.1
        arr = kit.arrow([zc.base.get_right()[0] + 0.25, ay], [pq.base.get_left()[0] - 0.25, ay],
                        text="softmax", color=self.color)
        with self.narrate(
            "Greedy's loops are why the truncation family exists: top-k, top-p, and min-p all cut off the "
            "unlikely tail. "
            "They differ only in the rule that picks the tokens to keep. "
            "Each filter sets the excluded logits to minus infinity, and lets the softmax renormalize. "
            "With k = 2, the threshold is the second-largest logit, and three logits become minus infinity. "
            "The two survivors renormalize to about 0.73 and 0.27. "
            "The weakness is that k is fixed, while the model's confidence isn't."
        ) as t:
            self.play(FadeIn(zc), FadeIn(zc_t), run_time=0.8)
            t.wait_until("They differ only")
            self.play(FadeIn(f), run_time=0.8)
            t.wait_until("Each filter sets")
            self.play(GrowArrow(arr.line), FadeIn(arr.label), FadeIn(pc), FadeIn(pc_t), run_time=0.9)
            t.wait_until("With k = 2")
            self.play(Create(thr), run_time=0.7)
            self.play(Transform(zc, zm), run_time=0.9)
            t.wait_until("The two survivors")
            self.play(Transform(pc, pq), run_time=1.1)
            t.wait_until("The weakness")
            self.play(FadeIn(weak), run_time=0.6)

    def top_p(self):
        self.slide("top-p", "Top-p keeps the token that tips the bucket", book_section="s-top-p-a-fixed-mass")
        p = softmax(Z)
        unit = 10.0
        full = mass_bar(p, 5, width=unit, labels=["0.6364", "0.2341", "", "", ""])
        kept = mass_bar(p, 2, width=unit, labels=["0.6364", "0.2341", "", "", ""])
        x0 = 0.0
        ticks = VGroup()
        for c, txt in ((0.6364, r"$C_1 = 0.6364$"), (0.8705, r"$C_2 = 0.8705$")):
            tk = Line([x0 + c * unit, -0.42, 0], [x0 + c * unit, -0.3, 0], color=C_COL, stroke_width=3)
            lab = kit.prose(txt, size=th.SIZE_LABEL, color=C_COL, width=None)
            lab.next_to(tk, DOWN, buff=0.08)
            ticks.add(VGroup(tk, lab))
        ticks[0][1].shift(LEFT * 0.75)
        ticks[1][1].shift(RIGHT * 0.45)

        def threshold(v, text):
            ln = DashedLine([v * unit, -0.45, 0], [v * unit, 0.55, 0], color=th.INK, stroke_width=th.STROKE_AXIS)
            lab = kit.label(text, size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM).next_to(ln, UP, buff=0.08)
            return VGroup(ln, lab)

        t70 = threshold(0.7, "p = 0.7")
        t50 = threshold(0.5, "p = 0.5")
        mass_t = caption("probability mass, most likely first", size=th.SIZE_SMALL)
        mass_t.next_to(full, UP, buff=1.0).align_to(full, LEFT)
        bar_fig = VGroup(full, ticks, t70, t50, mass_t)
        rule = kit.formula(r"r \le n(p) \iff", r"C_{r-1}", r"< p", size=th.SIZE_MATH)
        rule[1].set_color(C_COL)
        box = SurroundingRectangle(rule, color=C_COL, buff=0.18, corner_radius=0.08, stroke_width=3)
        rule_t = caption("keep rank r while the mass before it is short of p", size=th.SIZE_SMALL)
        rules = VGroup(VGroup(rule, box), rule_t).arrange(DOWN, buff=0.25)
        res = VGroup(
            kit.label("p = 0.7: two tokens, 0.8705 of the mass", size=th.SIZE_SMALL, color=th.INK_2),
            kit.label("p = 0.5: the top token alone", size=th.SIZE_SMALL, color=th.INK_2),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        bottom = VGroup(rules, res).arrange(RIGHT, buff=1.0)
        col = VGroup(bar_fig, bottom).arrange(DOWN, buff=0.75)
        self.place(col)
        kept.scale(full.frame.width / kept.frame.width)
        kept.shift(full.frame.get_left() - kept.frame.get_left())
        with self.narrate(
            "Top-p keeps a fixed share of probability instead of a fixed count. "
            "Picture pouring tokens into a bucket, most likely first. "
            "Stop as soon as it holds at least p, and keep the token that tips it over. "
            "So keep rank r while the mass before it is still below p. "
            "At p = 0.7, the first token alone holds about 0.64, which is short. "
            "So the second goes in, and the kept mass reaches about 0.87. "
            "At p = 0.5, the first token alone is enough."
        ) as t:
            self.play(FadeIn(full.frame), FadeIn(mass_t), run_time=0.6)
            t.wait_until("Picture pouring")
            for i in range(5):
                anims = [GrowFromEdge(full.segs[i], LEFT)]
                if i < 2:
                    anims.append(FadeIn(full.texts[i]))
                self.play(*anims, run_time=0.45)
            t.wait_until("Stop as soon")
            self.play(Create(t70[0]), FadeIn(t70[1]), run_time=0.7)
            t.wait_until("So keep rank r")
            self.play(FadeIn(rule), Create(box), FadeIn(rule_t), run_time=0.9)
            t.wait_until("At p = 0.7, the first")
            self.play(FadeIn(ticks[0]), Indicate(full.segs[0], color=self.color, scale_factor=1.02), run_time=0.8)
            t.wait_until("So the second goes in")
            self.play(FadeIn(ticks[1]), Transform(full, kept), FadeIn(res[0]), run_time=1.0)
            t.wait_until("At p = 0.5")
            self.play(Create(t50[0]), FadeIn(t50[1]), FadeIn(res[1]), run_time=0.8)

    def top_p_bug(self):
        self.slide("top-p-bug", "Test the mass before the token, not through it", book_section="s-top-p-a-fixed-mass")
        panel = kit.code(TOP_P, "python")
        p = softmax(Z)
        unit = 6.0
        good = mass_bar(p, 2, width=unit)
        bad = mass_bar(p, 1, width=unit)

        def row(bar, title, verdict, color):
            ln = DashedLine([0.7 * unit, -0.4, 0], [0.7 * unit, 0.4, 0], color=th.INK, stroke_width=th.STROKE_AXIS)
            fig = VGroup(bar, ln)
            head = kit.label(title, size=th.SIZE_SMALL, color=th.INK, font=th.MONO)
            res = kit.label(verdict, size=th.SIZE_SMALL, color=color, weight=MEDIUM)
            txt = VGroup(head, res).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
            return VGroup(txt, fig).arrange(RIGHT, buff=0.5)

        r1 = row(good, "cumulative - sorted_probs >= p", "keeps 2 tokens, 0.8705 of the mass", th.GOOD)
        r2 = row(bad, "cumulative >= p", "keeps 1 token, 0.6364 < 0.7", th.BAD)
        rows = VGroup(r1, r2).arrange(DOWN, buff=0.35, aligned_edge=RIGHT)
        tag = caption("at p = 0.7, dashed", size=th.SIZE_LABEL)
        tag.next_to(rows, UP, buff=0.12).align_to(rows, RIGHT)
        col = VGroup(panel, VGroup(tag, rows)).arrange(DOWN, buff=0.45)
        self.place(col)
        with self.narrate(
            "The cumulative sum gives the mass through each token, not before it. "
            "So subtract the token's own probability. "
            "That's the mass before rank r, which you test against p. "
            "The last line always keeps the top token. "
            "Comparing the cumulative sum itself with p is the classic bug. "
            "At p = 0.7, it keeps one token, short of what you asked for. "
            "The output still reads fluently, so the bug survives code review."
        ) as t:
            self.play(FadeIn(panel.frame), *panel.reveal(1, 2, 3), run_time=0.9)
            t.wait_until("So subtract")
            self.play(*panel.reveal(4), panel.highlight(4), run_time=0.7)
            t.wait_until("That's the mass before")
            self.play(FadeIn(tag), FadeIn(r1), run_time=0.8)
            t.wait_until("The last line")
            self.play(*panel.reveal(5), run_time=0.6)
            t.wait_until("Comparing the cumulative")
            self.play(FadeIn(r2[0][0]), FadeIn(r2[1]), run_time=0.8)
            t.wait_until("At p = 0.7, it keeps")
            self.play(FadeIn(r2[0][1]), Indicate(bad.segs[1], color=th.BAD, scale_factor=1.05), run_time=0.9)

    def min_p(self):
        self.slide("min-p", "Min-p keeps a window of logits below the top", book_section="s-min-p-a-logit-window")
        f = kit.formula(r"p_i \ge", r"m", r"\, p_{\max} \iff e^{z_i - z_{\max}} \ge", r"m", r"\iff",
                        r"z_i \ge z_{\max} + \ln", r"m", size=th.SIZE_MATH_SMALL)
        for k in (1, 3, 6):
            f[k].set_color(M_COL)
        box = SurroundingRectangle(f[5:], color=M_COL, buff=0.12, corner_radius=0.06, stroke_width=3)
        w = 9.0
        lo_v, hi_v = -1.5, 3.5

        def fx(v):
            return (v - lo_v) / (hi_v - lo_v) * w

        axis = Line([fx(lo_v), 0, 0], [fx(hi_v), 0, 0], color=th.INK_3, stroke_width=th.STROKE_AXIS)
        dots, nums = VGroup(), VGroup()
        for v, nm in zip(Z, LOGIT_NAMES):
            d = Dot([fx(v), 0, 0], radius=0.13, color=Z_COL)
            n = kit.label(nm, size=th.SIZE_SMALL, color=th.INK_2).next_to(d, DOWN, buff=0.18)
            dots.add(d)
            nums.add(n)
        win = Rectangle(width=fx(3) - fx(1.796), height=1.0, stroke_width=0, fill_color=th.tint(M_COL, 0.3),
                        fill_opacity=1).move_to([(fx(3) + fx(1.796)) / 2, 0, 0])
        win.set_z_index(-1)
        cut = DashedLine([fx(1.796), -0.6, 0], [fx(1.796), 0.6, 0], color=M_COL, stroke_width=th.STROKE_AXIS)
        cut_t = kit.label("cut at 3 − 1.204 = 1.796", size=th.SIZE_LABEL, color=M_COL)
        cut_t.next_to(cut, UP, buff=0.1).shift(LEFT * 0.6)
        ax_t = caption("logits, min-p = 0.3")
        ax_t.next_to(axis, LEFT, buff=0.3)
        line_fig = VGroup(win, axis, dots, nums, cut, cut_t, ax_t)
        bars = kit.bar_compare(
            [("min-p 0.3, T = 1", 1.204, M_COL, "1.204 logits"),
             ("min-p 0.05, T = 1", 2.996, M_COL, "3.00 logits"),
             ("min-p 0.05, T = 1.5", 4.494, T_COL, "4.49 logits")],
            width=5.5, bar_height=0.38, gap=0.2, size=th.SIZE_SMALL,
        )
        bars_t = caption("window width, in original logits", size=th.SIZE_SMALL)
        right = VGroup(bars_t, bars).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        col = VGroup(VGroup(f, box), line_fig, right).arrange(DOWN, buff=0.55)
        self.place(col)
        kept = VGroup(dots[0], dots[1])
        with self.narrate(
            "Min-p keeps every token at least min-p times as likely as the favorite. "
            "The rule is a ratio, so the normalizer cancels. "
            "Take logs, and it becomes a window below the top logit. "
            "At a min-p of 0.3, the window is 1.204 logits wide. "
            "On the five-logit row, the cut lands at 1.796, so the logits 3 and 2 survive. "
            "At a min-p of 0.05, the window is 3 logits wide. "
            "Temperature stretches it: at 1.5, it's 4.49. "
            "So min-p grows only in proportion to temperature, while top-p's nucleus grows without bound."
        ) as t:
            self.play(FadeIn(f[0:3]), run_time=0.8)
            t.wait_until("The rule is a ratio")
            self.play(FadeIn(f[3:5]), run_time=0.7)
            t.wait_until("Take logs")
            self.play(FadeIn(f[5:]), Create(box), run_time=0.8)
            t.wait_until("At a min-p of 0.3")
            self.play(Create(axis), FadeIn(dots), FadeIn(nums), FadeIn(ax_t), run_time=0.8)
            self.play(FadeIn(bars_t), FadeIn(bars.labels[0]), GrowFromEdge(bars.bars[0], LEFT),
                      FadeIn(bars.values[0]), run_time=0.8)
            t.wait_until("On the five-logit row")
            self.play(GrowFromEdge(win, RIGHT), Create(cut), FadeIn(cut_t), run_time=0.9)
            self.play(kept.animate.set_color(self.color), VGroup(*dots[2:]).animate.set_color(OFF), run_time=0.6)
            t.wait_until("At a min-p of 0.05")
            self.play(FadeIn(bars.labels[1]), GrowFromEdge(bars.bars[1], LEFT), FadeIn(bars.values[1]),
                      run_time=0.8)
            t.wait_until("Temperature stretches")
            self.play(FadeIn(bars.labels[2]), GrowFromEdge(bars.bars[2], LEFT), FadeIn(bars.values[2]),
                      run_time=0.9)

    # ------------------------------------------------------------------
    # The order

    def order(self):
        self.slide("order", "Penalties, temperature, truncation, then the draw",
                   book_section="s-the-order-and-which-pairs-commute")
        steps = [("Penalties", "raw logits"), ("Temperature", "divide by T"),
                 ("Truncation", "top-k, min-p, top-p"), ("Softmax, draw", "one token")]
        boxes = VGroup(*[kit.box(a, sublabel=b, color=self.color if i < 3 else th.INFO, width=2.5)
                         for i, (a, b) in enumerate(steps)]).arrange(RIGHT, buff=0.6)
        tab = kit.table(
            [
                ["Repetition penalty, temperature", "Yes", r"both multiply; $T > 0$ keeps the sign"],
                ["Presence or frequency, temperature", "No", "additive against multiplicative"],
                ["Temperature, top-k", "Yes", r"dividing by $T > 0$ keeps the ranking"],
                ["Temperature, min-p", "No", r"the window scales with $T$"],
                ["Temperature, top-p", "No", "the nucleus is a mass, not a ranking"],
                ["Penalties, any truncation", "No", "penalties change the ranking"],
            ],
            header=["Swap", "Commutes?", "Why"],
            size=th.SIZE_SMALL, row_gap=0.2,
        )
        for r in (2, 4, 5, 6):
            tab.cells[r][1].set_color(th.BAD)
        for r in (1, 3):
            tab.cells[r][1].set_color(th.GOOD)
        col = VGroup(boxes, tab).arrange(DOWN, buff=0.6)
        self.place(col)
        arrows = VGroup(*[kit.arrow(a, b) for a, b in zip(boxes[:-1], boxes[1:])])
        with self.narrate(
            "You now have five transforms, and their order matters. "
            "The engine applies penalties to the raw logits first. "
            "Then temperature divides them. "
            "Then the truncations run: top-k, then min-p, then top-p. "
            "Last come the softmax and the draw. "
            "Swapping two adjacent steps changes the output for only three kinds of pair. "
            "Additive penalties against temperature. "
            "Temperature against min-p or top-p. "
            "And penalties against any truncation, because penalties change the ranking. "
            "Top-p goes last, because it's the only one that reads the renormalized masses."
        ) as t:
            for i, cue in enumerate(["The engine applies", "Then temperature", "Then the truncations",
                                     "Last come"]):
                t.wait_until(cue)
                anims = [FadeIn(boxes[i], shift=RIGHT * 0.1)]
                if i:
                    anims.append(GrowArrow(arrows[i - 1].line))
                self.play(*anims, run_time=0.6)
            t.wait_until("Swapping two")
            self.play(FadeIn(tab.rules), FadeIn(VGroup(*tab.cells[0])), FadeIn(VGroup(*tab.cells[1])),
                      FadeIn(VGroup(*tab.cells[3])), run_time=0.9)
            t.wait_until("Additive penalties")
            self.play(FadeIn(VGroup(*tab.cells[2]), shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("Temperature against")
            self.play(FadeIn(VGroup(*tab.cells[4]), shift=RIGHT * 0.1),
                      FadeIn(VGroup(*tab.cells[5]), shift=RIGHT * 0.1), run_time=0.7)
            t.wait_until("And penalties against")
            self.play(FadeIn(VGroup(*tab.cells[6]), shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("Top-p goes last")
            self.play(Indicate(boxes[2], color=self.color, scale_factor=1.05), run_time=0.9)

    def order_example(self):
        self.slide("order-example", "Temperature after top-p narrows the nucleus",
                   book_section="s-the-order-and-which-pairs-commute")
        p = softmax(Z)
        p2 = softmax(Z / 2)
        sc = 3.0
        kw = dict(names=LOGIT_NAMES, width=0.8, gap=0.3)
        right_cols = [self.color] * 3 + [OFF] * 2
        l_start = vbars(p2, sc, colors=right_cols, **kw)
        r3 = p2[:3] / p2[:3].sum()
        l_end = vbars([*r3, 0, 0], sc, texts=["0.5065", "0.3072", "0.1863", "", ""], **kw)
        r_start = vbars(p, sc, colors=[self.color] * 2 + [OFF] * 3, **kw)
        w2 = softmax(np.array([1.5, 1.0]))
        r_end = vbars([*w2, 0, 0, 0], sc, texts=["0.6225", "0.3775", "", "", ""], **kw)

        tops = VGroup(kit.pill("top token 0.51", th.GOOD, size=th.SIZE_SMALL),
                      kit.pill("top token 0.62", th.BAD, size=th.SIZE_SMALL))

        def panel(title, start, sub, color, pill):
            head = kit.label(title, size=th.SIZE_SMALL, color=color, weight=MEDIUM)
            note = kit.prose(sub, size=th.SIZE_LABEL, color=th.INK_3, width=None)
            g = VGroup(head, start, note, pill).arrange(DOWN, buff=0.3)
            return g


        lp = panel("Right: temperature, then top-p", l_start, r"$C_2 = 0.6886 < 0.7$: three tokens survive",
                   th.GOOD, tops[0])
        rp = panel("Wrong: top-p, then temperature", r_start, r"two tokens, holding $0.6886 < 0.7$ of the tempered mass",
                   th.BAD, tops[1])
        lp[2].align_to(lp[1], LEFT)
        both = VGroup(lp, rp).arrange(RIGHT, buff=1.3, aligned_edge=UP)
        given = kit.label("The five logits, top-p = 0.7, temperature 2", size=th.SIZE_SMALL, color=th.INK_2)
        col = VGroup(given, both).arrange(DOWN, buff=0.45)
        self.place(col)
        match(l_end, l_start)
        match(r_end, r_start)
        with self.narrate(
            "Now the swap that changes output visibly: top-p at 0.7 and a temperature of 2. "
            "In the right order, you temper first. "
            "The top two now hold about 0.69, below 0.7, so a third token survives. "
            "Renormalized, the top token gets about 0.51. "
            "In the wrong order, top-p keeps two tokens. "
            "Tempering those two gives the top token about 0.62. "
            "And the parameter no longer describes anything: you asked for more variety, and got a narrower "
            "nucleus."
        ) as t:
            self.play(FadeIn(given), run_time=0.6)
            t.wait_until("In the right order")
            self.play(FadeIn(lp[0]), FadeIn(l_start), run_time=0.8)
            t.wait_until("The top two now")
            self.play(FadeIn(lp[2]), Indicate(l_start.bars[2], color=self.color, scale_factor=1.08), run_time=0.9)
            t.wait_until("Renormalized")
            self.play(Transform(l_start, l_end), FadeIn(tops[0], scale=0.9), run_time=1.0)
            t.wait_until("In the wrong order")
            self.play(FadeIn(rp[0]), FadeIn(r_start), run_time=0.8)
            t.wait_until("Tempering those two")
            self.play(Transform(r_start, r_end), FadeIn(tops[1], scale=0.9), run_time=1.0)
            t.wait_until("And the parameter")
            self.play(FadeIn(rp[2]), run_time=0.7)

    # ------------------------------------------------------------------
    # Penalties

    def repetition_penalty(self):
        self.slide("repetition-penalty", "Divide positive logits, multiply negative ones",
                   book_section="s-penalties-as-logit-transforms")
        w = 11.0
        lo_v, hi_v = -4.5, 2.5

        def fx(v):
            return (v - lo_v) / (hi_v - lo_v) * w

        axis = Line([fx(lo_v), 0, 0], [fx(hi_v), 0, 0], color=th.INK_3, stroke_width=th.STROKE_AXIS)
        ticks = VGroup()
        for v in range(-4, 3):
            tk = Line([fx(v), -0.08, 0], [fx(v), 0.08, 0], color=th.INK_3, stroke_width=th.STROKE_AXIS)
            num = kit.label(str(v).replace("-", "−"), size=th.SIZE_LABEL, color=th.INK_4).next_to(tk, DOWN, buff=0.1)
            ticks.add(VGroup(tk, num))
        start = [(2.0, "seen"), (-2.0, "seen"), (0.5, "unseen")]
        dots, tags = VGroup(), VGroup()
        for v, tag in start:
            d = Dot([fx(v), 0, 0], radius=0.13, color=Z_COL if tag == "seen" else OFF)
            lab = kit.label(tag, size=th.SIZE_LABEL, color=th.INK_3).move_to([fx(v), -0.75, 0])
            dots.add(d)
            tags.add(lab)
        ends = [Dot([fx(1.0), 0, 0], radius=0.13, color=self.color), Dot([fx(-4.0), 0, 0], radius=0.13,
                                                                         color=self.color)]
        a_pos = CurvedArrow(np.array([fx(2.0), 0.18, 0]), np.array([fx(1.0), 0.18, 0]), angle=PI / 2.2,
                            color=self.color, stroke_width=3, tip_length=0.18)
        a_neg = CurvedArrow(np.array([fx(-2.0), 0.18, 0]), np.array([fx(-4.0), 0.18, 0]), angle=PI / 2.6,
                            color=self.color, stroke_width=3, tip_length=0.18)
        lab_pos = kit.label("2 / 2 = 1", size=th.SIZE_LABEL, color=self.color)
        lab_pos.move_to(a_pos.get_center() + UP * 0.55)
        lab_neg = kit.label("−2 × 2 = −4", size=th.SIZE_LABEL, color=self.color)
        lab_neg.move_to(a_neg.get_center() + UP * 0.6)
        line_fig = VGroup(axis, ticks, dots, tags, *ends, a_pos, a_neg, lab_pos, lab_neg)
        lab_case = kit.label("The lab: penalty 2 turns [2.0, −2.0, 0.5] into [1.0, −4.0, 0.5]", size=th.SIZE_SMALL,
                             color=th.INK_2)
        code = kit.mono("torch.where(scores > 0, scores / penalty, scores * penalty)", size=th.SIZE_SMALL,
                        color=th.INK)
        wrong = VGroup(
            kit.formula(r"-2 / 1.1 = -1.82 > -2", size=th.SIZE_MATH_SMALL, color=th.BAD),
            kit.label("divide a negative logit: it goes up", size=th.SIZE_LABEL, color=th.BAD),
        ).arrange(DOWN, buff=0.12)
        right = VGroup(
            kit.formula(r"-2 \times 1.1 = -2.2 < -2", size=th.SIZE_MATH_SMALL, color=th.GOOD),
            kit.label("multiply it: it goes down", size=th.SIZE_LABEL, color=th.GOOD),
        ).arrange(DOWN, buff=0.12)
        pair = VGroup(wrong, right).arrange(RIGHT, buff=1.4)
        col = VGroup(pair, code, line_fig, lab_case).arrange(DOWN, buff=0.5)
        self.place(col)
        with self.narrate(
            "Penalties lower the logits of tokens the model has already produced. "
            "The repetition penalty scales each seen token's logit, with a branch on its sign. "
            "Dividing a positive logit moves it toward zero, which is down. "
            "For a negative logit, dividing moves it up. "
            "Minus 2 divided by 1.1 is minus 1.82, which encourages the token. "
            "So the negative branch multiplies instead. "
            "The lab checks this with a penalty of 2. "
            "The logit 2 halves to 1, minus 2 doubles to minus 4, and the unseen token stays put."
        ) as t:
            t.wait_until("The repetition penalty")
            self.play(FadeIn(code), run_time=0.7)
            t.wait_until("For a negative logit")
            self.play(FadeIn(wrong[1]), run_time=0.5)
            t.wait_until("Minus 2 divided")
            self.play(FadeIn(wrong[0]), run_time=0.7)
            t.wait_until("So the negative branch")
            self.play(FadeIn(right), run_time=0.7)
            t.wait_until("The lab checks")
            self.play(Create(axis), FadeIn(ticks), FadeIn(dots), FadeIn(tags), run_time=0.9)
            t.wait_until("The logit 2 halves")
            self.play(Create(a_pos), FadeIn(lab_pos), TransformFromCopy(dots[0], ends[0]),
                      dots[0].animate.set_color(OFF), run_time=0.9)
            self.play(Create(a_neg), FadeIn(lab_neg), TransformFromCopy(dots[1], ends[1]),
                      dots[1].animate.set_color(OFF), run_time=0.9)
            self.play(FadeIn(lab_case), Indicate(dots[2], color=th.INK_3), run_time=0.7)

    def presence_frequency(self):
        self.slide("presence-frequency", "Presence and frequency subtract a constant",
                   book_section="s-penalties-as-logit-transforms")
        f = kit.formula(r"z_i' = z_i", r"- \alpha_{\text{pres}}\, \mathbf{1}[c_i > 0]", r"- \alpha_{\text{freq}}\, c_i")
        f[1].set_color(self.color)
        f[2].set_color(th.INFO)
        b1 = Brace(f[1], DOWN, color=th.INK_3, buff=0.1)
        t1 = caption("once per distinct token").next_to(b1, DOWN, buff=0.08).align_to(b1, RIGHT)
        b2 = Brace(f[2], DOWN, color=th.INK_3, buff=0.1)
        t2 = caption("grows with the count").next_to(b2, DOWN, buff=0.08).align_to(b2, LEFT).shift(RIGHT * 0.15)
        top = VGroup(f, b1, t1, b2, t2)
        odds = kit.bar_compare(
            [("odds, unpenalized", 1.0, th.INK_4, "1"),
             ("odds, presence 0.5", 0.6065, self.color, "× 0.607: 39% lower")],
            width=5.0, bar_height=0.45, gap=0.25, size=th.SIZE_SMALL,
        )
        temp = kit.bar_compare(
            [("penalty you set", 0.5, self.color, "0.5"),
             ("shift after T = 0.7", 0.714, T_COL, "0.5 / 0.7 = 0.714")],
            width=5.0, bar_height=0.45, gap=0.25, size=th.SIZE_SMALL,
        )
        temp_t = caption("Penalties run before temperature, so the shift is divided by T", size=th.SIZE_SMALL)
        lower = VGroup(odds, VGroup(temp_t, temp).arrange(DOWN, buff=0.25, aligned_edge=LEFT)).arrange(
            DOWN, buff=0.55, aligned_edge=LEFT)
        col = VGroup(top, lower).arrange(DOWN, buff=0.6)
        self.place(col)
        with self.narrate(
            "Presence and frequency penalties subtract a constant instead. "
            "Presence applies once per distinct token seen. "
            "Frequency scales with the count, so it keeps biting. "
            "A presence penalty of 0.5 multiplies the odds by 0.607, a 39 percent cut. "
            "Because penalties run before temperature, the shift is divided by T. "
            "At a temperature of 0.7, a penalty of 0.5 acts like 0.714. "
            "Document that, rather than fixing it."
        ) as t:
            self.play(FadeIn(f[0]), run_time=0.6)
            t.wait_until("Presence applies")
            self.play(FadeIn(f[1]), FadeIn(b1), FadeIn(t1), run_time=0.8)
            t.wait_until("Frequency scales")
            self.play(FadeIn(f[2]), FadeIn(b2), FadeIn(t2), run_time=0.8)
            t.wait_until("A presence penalty")
            self.play(FadeIn(odds.labels), *odds.grow(run_time=0.9), run_time=0.9)
            self.play(FadeIn(odds.values), run_time=0.5)
            t.wait_until("Because penalties run")
            self.play(FadeIn(temp_t), FadeIn(temp.labels[0]), GrowFromEdge(temp.bars[0], LEFT),
                      FadeIn(temp.values[0]), run_time=0.8)
            t.wait_until("At a temperature of 0.7")
            self.play(FadeIn(temp.labels[1]), GrowFromEdge(temp.bars[1], LEFT), FadeIn(temp.values[1]),
                      run_time=0.8)

    # ------------------------------------------------------------------
    # The draw

    def gumbel(self):
        self.slide("gumbel", "Add Gumbel noise, and take the argmax", book_section="s-the-gumbel-max-trick")
        f = kit.formula(r"\operatorname{arg\,max}_i \,(z_i +", r"G_i", r") \sim \operatorname{softmax}(z), \qquad",
                        r"G", r"= -\ln(-\ln U)", size=th.SIZE_MATH_SMALL)
        f[1].set_color(G_COL)
        f[3].set_color(G_COL)
        gen = kit.rng(11)
        races = -np.log(-np.log(gen.random((4, 5))))
        tally_u = np.clip(gen.random((5000, 5)), 1e-12, None)
        wins = np.bincount((Z - np.log(-np.log(tally_u))).argmax(1), minlength=5) / 5000
        # Race panel: axis from -2 to 6 logits.
        y0, y1, hgt = -2.0, 6.0, 4.2
        pitch = 1.2

        def py(v):
            return (v - y0) / (y1 - y0) * hgt

        base = Line([-0.5, py(0), 0], [4 * pitch + 0.5, py(0), 0], color=th.INK_3, stroke_width=th.STROKE_AXIS)
        marks, names = VGroup(), VGroup()
        for i, v in enumerate(Z):
            m = Line([i * pitch - 0.28, py(v), 0], [i * pitch + 0.28, py(v), 0], color=Z_COL, stroke_width=6)
            marks.add(m)
            n = kit.label(LOGIT_NAMES[i], size=th.SIZE_LABEL, color=th.INK_3).move_to([i * pitch, py(y0) - 0.3, 0])
            names.add(n)
        race_t = caption("one race: logit plus noise")
        race_t.move_to([2 * pitch, py(y1) + 0.35, 0])
        frame = Rectangle(width=4 * pitch + 1.0, height=hgt, stroke_width=0, fill_opacity=0)
        frame.move_to([2 * pitch, hgt / 2, 0])
        key = VGroup(swatch("logit", Z_COL), swatch("Gumbel noise", G_COL)).arrange(RIGHT, buff=0.4)
        key.move_to([2 * pitch, py(y0) - 0.85, 0])
        race_fig = VGroup(frame, base, marks, names, race_t, key)

        def race(k):
            g = VGroup()
            totals = Z + races[k]
            win = int(np.argmax(totals))
            for i in range(5):
                col = self.color if i == win else G_COL
                seg = Line([i * pitch, py(Z[i]), 0], [i * pitch, py(totals[i]), 0], color=col, stroke_width=7)
                dot = Dot([i * pitch, py(totals[i]), 0], radius=0.09 if i != win else 0.13, color=col)
                g.add(VGroup(seg, dot))
            tag = kit.label("wins", size=th.SIZE_LABEL, color=self.color, weight=MEDIUM)
            tag.next_to(g[win][1], UP, buff=0.12)
            g.add(tag)
            return g

        tally = vbars(wins, 2.8, fmt="{:.3f}", names=LOGIT_NAMES, width=0.75, gap=0.4)
        targets = VGroup()
        for b, pv in zip(tally.bars, softmax(Z)):
            y = b.get_bottom()[1] + pv * 2.8
            targets.add(Line([b.get_left()[0] - 0.08, y, 0], [b.get_right()[0] + 0.08, y, 0], color=th.INK,
                             stroke_width=3))
        tally_t = caption("win shares over 5000 races")
        tally_t.next_to(tally, UP, buff=0.3).align_to(tally, LEFT)
        tkey = VGroup(Line(LEFT * 0.2, RIGHT * 0.2, color=th.INK, stroke_width=3),
                      kit.label("softmax", size=th.SIZE_LABEL, color=th.INK_2)).arrange(RIGHT, buff=0.15)
        tkey.next_to(tally, DOWN, buff=0.25).align_to(tally, LEFT)
        tally_fig = VGroup(tally_t, tally, targets, tkey)
        body = VGroup(race_fig, tally_fig).arrange(RIGHT, buff=1.4, aligned_edge=DOWN)
        col = VGroup(f, body).arrange(DOWN, buff=0.45)
        self.place(col)
        # Build the races after placing, in the placed coordinates.
        sc = base.width / (4 * pitch + 1.0)
        shift_v = base.get_left() - sc * np.array([-0.5, py(0), 0])

        def placed_race(k):
            g = race(k)
            g.scale(sc, about_point=ORIGIN)
            g.shift(shift_v)
            return g

        r = [placed_race(k) for k in range(4)]
        with self.narrate(
            "The third need is the draw, and it can be a race. "
            "Give every token a random bonus, add it to its logit, and pick the highest total. "
            "Here, the runner-up wins the first race. "
            "With Gumbel-distributed bonuses, each token wins with exactly its softmax probability. "
            "Over 5000 races, the win shares land on the softmax. "
            "The proof never needs the normalizer, and a masked token can never win. "
            "So the draw is one fused pass, with no cumulative sum and no search."
        ) as t:
            self.play(FadeIn(base), FadeIn(marks), FadeIn(names), FadeIn(race_t), run_time=0.8)
            t.wait_until("Give every token")
            self.play(FadeIn(key), LaggedStart(*[Create(s[0]) for s in r[0][:5]], lag_ratio=0.15),
                      run_time=1.2)
            self.play(*[FadeIn(s[1], scale=0.5) for s in r[0][:5]], run_time=0.4)
            t.wait_until("Here, the runner-up")
            self.play(FadeIn(r[0][5]), Indicate(r[0][1], color=self.color), run_time=0.8)
            t.wait_until("With Gumbel-distributed")
            self.play(FadeIn(f), run_time=0.8)
            for k in range(1, 4):
                self.play(FadeOut(r[k - 1]), FadeIn(r[k]), run_time=0.6)
            t.wait_until("Over 5000 races")
            self.play(FadeIn(tally_t), FadeIn(tally.base), FadeIn(tally.names), *[GrowFromEdge(b, DOWN)
                                                                                    for b in tally.bars],
                      run_time=1.0)
            self.play(FadeIn(tally.vals), Create(targets), FadeIn(tkey), run_time=0.8)

    def seeding(self):
        self.slide("seeding", "A seed reproduces only with your own generator",
                   book_section="s-seeding-and-reproducibility")
        panel = kit.code(SEED, "python")
        users = VGroup(*[kit.label(s, size=th.SIZE_LABEL, color=th.INK_2) for s in
                         ("dropout layer", "shuffled dataloader", "another request")]).arrange(DOWN, buff=0.3,
                                                                                               aligned_edge=RIGHT)
        glob = kit.box("Global RNG", sublabel="shared by the process", color=th.BAD, width=3.0)
        left = VGroup(users, glob).arrange(RIGHT, buff=0.7)
        req = kit.label("this request", size=th.SIZE_LABEL, color=th.INK_2)
        own = kit.box("torch.Generator", sublabel="owned by the request", color=th.GOOD, width=3.0)
        right = VGroup(req, own).arrange(RIGHT, buff=0.7)
        mid = VGroup(left, right).arrange(RIGHT, buff=0.9)
        batch = kit.prose(r"Across batch sizes, the last bits differ, and a near-tie argmax flips.",
                          size=th.SIZE_SMALL, width=None, color=th.INK_2)
        key = kit.callout("key", r"Same seed, same batch composition, same engine version: same output.",
                          title="Promise only what you can reproduce", width=11.5, size=th.SIZE_SMALL)
        col = VGroup(panel, mid, batch, key).arrange(DOWN, buff=0.4)
        self.place(col)
        u_arrows = VGroup(*[Arrow(u.get_right(), glob.rect.get_left() + UP * (0.3 - 0.3 * i), buff=0.12,
                                  color=th.INK_4, stroke_width=3, tip_length=0.16) for i, u in enumerate(users)])
        r_arrow = kit.arrow(req, own)
        with self.narrate(
            "A generation should reproduce exactly, given a seed. "
            "That means threading an explicit generator through the sampler. "
            "The global random number generator is shared by the whole process. "
            "A dropout layer, a dataloader, or another request can consume its draws. "
            "An explicit generator is owned by the request. "
            "Across batch sizes, it's harder. "
            "A different reduction order changes the last bits of the logits, and a near-tie argmax flips. "
            "So promise only this: same seed, same batch composition, same engine version, same output."
        ) as t:
            t.wait_until("That means threading")
            self.play(FadeIn(panel.frame), *panel.reveal(1, 2), run_time=0.8)
            t.wait_until("The global random")
            self.play(FadeIn(glob), run_time=0.6)
            t.wait_until("A dropout layer")
            self.play(FadeIn(users), LaggedStart(*[GrowArrow(a) for a in u_arrows], lag_ratio=0.3), run_time=1.0)
            t.wait_until("An explicit generator")
            self.play(FadeIn(req), GrowArrow(r_arrow.line), FadeIn(own), run_time=0.8)
            t.wait_until("A different reduction")
            self.play(FadeIn(batch), run_time=0.7)
            t.wait_until("So promise only")
            self.play(FadeIn(key, shift=UP * 0.1), run_time=0.7)

    # ------------------------------------------------------------------
    # What sampling costs

    def cost_softmax(self):
        self.slide("cost-softmax", "One softmax: 0.375 FLOPs per byte", book_section="s-what-sampling-costs")
        l1 = kit.formula(r"\text{one row}", r"=", r"248{,}320 \times 4 = 993{,}280 \text{ bytes}",
                         size=th.SIZE_MATH_SMALL)
        l2 = kit.formula(r"\text{read + write}", r"=", r"1.99 \text{ MB}", size=th.SIZE_MATH_SMALL)
        l3 = kit.formula(r"3V", r"\approx", r"745{,}000 \text{ FLOPs}", size=th.SIZE_MATH_SMALL)
        l4 = kit.formula(r"\frac{7.45 \times 10^{5}}{1.99 \times 10^{6}}", r"=", r"0.375 \text{ FLOPs per byte}",
                         size=th.SIZE_MATH_SMALL)
        l4[2].set_color(self.color)
        maths = aligned([l1, l2, l3, l4], buff=0.35)
        ridge = kit.prose(r"A100 ridge point: 161. Softmax sits $161 / 0.375 = 429\times$ below it.",
                          size=th.SIZE_SMALL, width=None, color=th.INK_2)
        times = kit.bar_compare(
            [("memory, 1275 GB/s", 1600, th.INFO, "1.6 µs"),
             ("arithmetic, 312 TFLOP/s", 2.4, self.color, "2.4 ns")],
            width=6.0, bar_height=0.5, gap=0.3, size=th.SIZE_SMALL,
        )
        only = kit.pill("Only the bytes matter", self.color, size=th.SIZE_SMALL)
        lower = VGroup(times, only).arrange(RIGHT, buff=0.8)
        col = VGroup(maths, ridge, lower).arrange(DOWN, buff=0.55)
        self.place(col)
        with self.narrate(
            "Now the cost. "
            "One float32 row of logits is a little under 1 MB. "
            "One softmax pass reads it and writes it: 1.99 MB. "
            "The arithmetic is about 745,000 FLOPs. "
            "That's 0.375 FLOPs per byte. "
            "The A100's ridge point is 161, so softmax sits 429 times below it. "
            "At 1275 GB/s, the bytes take 1.6 microseconds. "
            "The arithmetic would take 2.4 nanoseconds. "
            "Only the bytes matter."
        ) as t:
            for line, cue in zip([l1, l2, l3, l4], ["One float32 row", "One softmax pass", "The arithmetic is",
                                                    "That's 0.375"]):
                t.wait_until(cue)
                self.play(FadeIn(line, shift=DOWN * 0.1), run_time=0.7)
            t.wait_until("The A100's ridge")
            self.play(FadeIn(ridge), run_time=0.7)
            t.wait_until("At 1275 GB/s")
            self.play(FadeIn(times.labels[0]), GrowFromEdge(times.bars[0], LEFT), FadeIn(times.values[0]),
                      run_time=0.9)
            t.wait_until("The arithmetic would")
            self.play(FadeIn(times.labels[1]), GrowFromEdge(times.bars[1], LEFT), FadeIn(times.values[1]),
                      run_time=0.6)
            t.wait_until("Only the bytes")
            self.play(FadeIn(only, scale=0.9), run_time=0.5)

    def cost_pipeline(self):
        self.slide("cost-pipeline", "Cheap in bytes, expensive in launches", book_section="s-what-sampling-costs")
        head = kit.prose(r"The whole pipeline at batch 1: at least 26 MB, 0.05\% of a 42.2 ms decode step.",
                         size=th.SIZE_SMALL, width=None, color=th.INK_2)
        bars = kit.bar_compare(
            [("memory traffic, 26 MB", 21, th.INFO, "21 µs"),
             ("launch overhead, about 20 kernels", 200, th.BAD, "100 to 200 µs")],
            width=6.0, bar_height=0.5, gap=0.3, size=th.SIZE_SMALL,
        )
        rng_bar = Rectangle(width=bars.bars[1].width / 2, height=bars.bars[1].height, stroke_width=0,
                            fill_color=th.CANVAS, fill_opacity=0.55)
        rng_bar.align_to(bars.bars[1], RIGHT).align_to(bars.bars[1], UP)
        bars.add(rng_bar)
        more = kit.bullets(
            [
                r"The draw forces a synchronization, and the sort can't be captured in a CUDA graph.",
                r"At batch 128: 3.3 GB, 2.6 ms, 6\% of the step.",
            ],
            size=th.SIZE_SMALL, width=11.5, buff=0.22,
        )
        fused = kit.callout("key", r"Fuse it: one launch, top-p on the top-k survivors, and a Gumbel-max draw that "
                                   r"stays on the device.", title="Which is why it's one kernel", width=11.5,
                            size=th.SIZE_SMALL)
        col = VGroup(head, bars, more, fused).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        self.place(col)
        with self.narrate(
            "The whole pipeline at batch 1 moves at least 26 MB. "
            "That's about 21 microseconds, or 0.05 percent of a decode step. "
            "The pipeline is about twenty kernels, at 5 to 10 microseconds of launch overhead each. "
            "That's 100 to 200 microseconds, five to ten times the traffic. "
            "The draw also forces a synchronization, and the sort blocks CUDA graph capture. "
            "At batch 128, the traffic grows to 6 percent of the step. "
            "So production engines fuse the sampler into one kernel, with a Gumbel-max draw."
        ) as t:
            self.play(FadeIn(head), run_time=0.7)
            t.wait_until("That's about 21")
            self.play(FadeIn(bars.labels[0]), GrowFromEdge(bars.bars[0], LEFT), FadeIn(bars.values[0]),
                      run_time=0.8)
            t.wait_until("The pipeline is about")
            self.play(FadeIn(bars.labels[1]), GrowFromEdge(bars.bars[1], LEFT), FadeIn(rng_bar), run_time=1.0)
            t.wait_until("That's 100 to 200")
            self.play(FadeIn(bars.values[1]), run_time=0.5)
            t.wait_until("The draw also")
            self.play(FadeIn(more[0], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("At batch 128")
            self.play(FadeIn(more[1], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("So production engines")
            self.play(FadeIn(fused, shift=UP * 0.1), run_time=0.8)
