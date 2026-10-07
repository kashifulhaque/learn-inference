"""Chapter 04, RMSNorm and the residual stream.

Teaches content/chapters/04-rmsnorm-and-residuals.md in its order. Every
number below was recomputed with Python, numpy, and torch (bfloat16 where the
chapter's argument is about bfloat16), and each matches the chapter.

How often: 64 layers x 2 norms + 1 final norm = 129 norms per forward pass.

The (3, 4) example:
- Squares 9 and 16; mean of squares (9 + 16) / 2 = 12.5; root 3.5355 (3.536).
- 3 / 3.5355 = 0.8485 (0.849); 4 / 3.5355 = 1.1314 (1.131);
  (0.8485^2 + 1.1314^2) / 2 = 1.0, so the result has RMS 1.
- Scale: (30, 40) gives the same (0.849, 1.131).
- Shift: (4, 5): ms = 20.5, rms 4.5277; outputs 0.8835 (0.883) and 1.1043
  (1.104), which differ from (0.849, 1.131).

LayerNorm: sigma^2 = ms(x) - mu^2. Dropping beta saves 129 x 5120 = 660,480
parameters; 660,480 / 26.9e9 = 0.00246%, "0.0025%".

bfloat16 (1 sign, 8 exponent, 7 stored significand bits; 8 bits of
precision). In [256, 512) the spacing is 2^(8-7) = 2; in [128, 256) it is 1.
torch: bfloat16(256) + bfloat16(1) == 256 (257 rounds back). Vanishing
threshold S/a >= 2^8 = 256 (possible), >= 2^9 = 512 (guaranteed). Bits of
256.0: sign 0, exponent 127 + 8 = 135 = 10000111, significand 0000000.

The stall: summing 5120 ones left to right in torch bfloat16 gives 256 after
256 terms and stays at 256 through term 5120 (checked). ms_hat = 256 s / 5120
= s / 20; normalizer 1 / sqrt(20) = 0.2236 (0.224); output 1 / 0.2236 =
4.472, "about 4.5 times" too large.

Error bounds: log2(5120) = 12.32 ("12.3"); 5120 x 2^-8 = 20;
12.3 x 2^-8 = 0.048; 12.3 x 2^-24 = 7.33e-7 ("7.3 x 10^-7"), below the lab's
1e-5 tolerance. Measured bf16 error "order of 1e-2" is the chapter's figure.

Residual stream: L = 64 x 2 = 128 contributions. Four-channel running sum
(illustrative values): x0 = (1, -0.5, 0.5, 2); + (0.5, 1, -0.5, 0) =
(1.5, 0.5, 0, 2); + (-1, 0.5, 1, 0.5) = (0.5, 1, 1, 2.5); + (0.5, 0.5, 0.5,
-1) = (1, 1.5, 1.5, 1.5).
Growth: sqrt(1 + 128) = sqrt(129) = 11.36, "11.4". Simulation on the slide:
kit.rng(4), d = 5120, x0 and each contribution standard normal; RMS after
0, 8, 32, 64, 128 contributions = 0.99, 3.02, 5.75, 8.12, 11.48 against
sqrt(1 + l) = 1, 3, 5.74, 8.06, 11.36.

Final norm (illustrative softmax, the chapter's "multiply every logit by 2"):
softmax(2, 1, 0) = (0.665, 0.245, 0.090); softmax(4, 2, 0) = (0.867, 0.117,
0.016). The argmax is the same.

Pre-norm versus post-norm: 0.95^64 = 0.0375 ("0.038"); 1.05^64 = 22.70
("22.7"). log10 of each: -1.426 and 1.356 (log-scale plot).

Per-head norm: 24 query heads and 4 key heads over 256; logit = q.k / sqrt(256)
= q.k / 16; |q| = |k| = sqrt(256) = 16; |q.k| <= 16 x 16 = 256;
256 / 16 = 16. Traffic: (24 + 4) x 256 = 7168 elements x 4 B = 28,672 B per
full-attention layer; x 16 = 458,752 B; stream norms 128 x 5120 x 4 =
2,621,440 B; 458,752 / 2,621,440 = 0.175, "about 17%".

Where the time goes: 4 FLOP / 4 B = 1 FLOP/B; ridge 312e12 / 1935e9 = 161.2;
1 FLOP/B x 1935 GB/s = 1.935 TFLOP/s ("1.94"); 1.935 / 312 = 0.62%
("0.6%"). Roofline plot in log10: log10(1.935) = 0.287, log10(161.2) =
2.207, log10(312) = 2.494.
Prefill, 4096 tokens: 4096 x 5120 = 20,971,520 elements x 4 B = 83.9 MB per
norm; / 1275 GB/s = 65.8 us; x 128 = 8.42 ms ("8.4"); Triton at 945 GB/s
(945 / 1275 = 74%): 128 x 83.9 MB / 945 GB/s = 11.36 ms ("11.4").

Fusion: five trips unfused, four fused; 1 - 4/5 = 20%. Intermediate sum
4096 x 5120 x 2 B = 41.9 MB against the 40 MB L2. Measured 1.10x faster when
it spills, 0.92x when it fits (chapter figures).
"""

from manim import *

import kit
from kit import theme as th

MS_C = th.HL_C  # \hlc in the chapter: the mean of squares
EPS_C = th.HL_A  # \hla: epsilon
GAIN_C = th.HL_B  # \hlb: the gain
DELTA_C = th.HL_D  # \hld: a sublayer's contribution

REF = """\
def rms_norm(x, weight, eps=1e-6):
    dtype = x.dtype
    x32 = x.float()
    variance = x32.pow(2).mean(dim=-1, keepdim=True)
    normed = x32 * torch.rsqrt(variance + eps)
    return (normed * weight.float()).to(dtype)
"""

LAYER = """\
residual = x
hidden = self.input_layernorm(x)
hidden = self.mixer(hidden, ...)
x = residual + hidden
return x + self.mlp(self.post_attention_layernorm(x))
"""


def hx(color) -> str:
    return ManimColor(color).to_hex()[1:7].upper()


def C(tex: str, color) -> str:
    """Colours a piece of TeX, inside `formula` or inside `$...$` in prose."""
    return rf"\textcolor[HTML]{{{hx(color)}}}{{{tex}}}"


MS = C(r"\operatorname{ms}(x)", MS_C)
EPS = C(r"\epsilon", EPS_C)


def vec(values, cell=(1.15, 0.6), color=None, size=None):
    """A one-row grid of numbers: a small vector."""
    g = kit.grid(1, len(values), cell=cell, values=[[str(v) for v in values]], size=size)
    if color is not None:
        g.paint(g.row(0), color)
    return g


def swatch(text, color, size=th.SIZE_LABEL):
    sq = Square(0.26, stroke_color=th.RULE_STRONG, stroke_width=th.STROKE_HAIRLINE,
                fill_color=th.tint(color, 0.35), fill_opacity=1)
    return VGroup(sq, kit.label(text, size=size, color=th.INK_2)).arrange(RIGHT, buff=0.16)


def line_key(text, color, size=th.SIZE_LABEL, dashed=False):
    cls = DashedLine if dashed else Line
    rule = cls(LEFT * 0.25, RIGHT * 0.25, color=color, stroke_width=th.STROKE_CURVE)
    return VGroup(rule, kit.label(text, size=size, color=th.INK_2)).arrange(RIGHT, buff=0.16)


def plus_node(r=0.22):
    c = Circle(radius=r, stroke_color=th.INK_2, stroke_width=th.STROKE_BOX, fill_color=th.RAISE, fill_opacity=1)
    p = kit.label("+", size=th.SIZE_SMALL, color=th.INK)
    p.move_to(c)
    return VGroup(c, p)


def elbow(points, color=th.INK_3, tip=True):
    """A path through `points` with square corners, ending in an arrow tip."""
    pts = [np.array([*p[:2], 0.0]) for p in points]
    parts = VGroup()
    for a, b in zip(pts[:-2], pts[1:-1]):
        parts.add(Line(a, b, color=color, stroke_width=th.STROKE_BOX + 1))
    if tip:
        parts.add(Arrow(pts[-2], pts[-1], buff=0, color=color, stroke_width=th.STROKE_BOX + 1, tip_length=0.18,
                        max_tip_length_to_length_ratio=0.5))
    else:
        parts.add(Line(pts[-2], pts[-1], color=color, stroke_width=th.STROKE_BOX + 1))
    return parts


class Lecture(kit.Lecture):
    def construct(self):
        self.title_slide(
            narration=(
                "RMSNorm and the residual stream. "
                "You learn the normalization every layer uses, and why its one sum must run in float32. "
                "Then you follow the residual stream, the running total that keeps a deep model stable."
            )
        )
        self.norms_everywhere()
        self.three_four()
        self.the_formula()
        self.scale_not_shift()
        self.layernorm()
        self.epsilon()
        self.addend_vanishes()
        self.the_stall()
        self.tree_reduction()
        self.reference_code()
        self.residual_layer()
        self.running_sum()
        self.stream_grows()
        self.final_norm()
        self.pre_vs_post()
        self.c_to_the_64()
        self.qk_norm()
        self.bandwidth_bound()
        self.fusion()
        self.what_goes_wrong()
        self.closing_slide(
            [
                r"Divide by $\sqrt{\operatorname{ms}(x) + \epsilon}$, apply a gain.",
                r"Reduce in float32: bf16 stalls near 256 terms.",
                r"Pre-norm keeps an identity path.",
                r"1 FLOP per byte: only fewer bytes help.",
            ],
            narration=(
                "Here's what to keep. "
                "RMSNorm divides each row by its root mean square, and applies a gain. "
                "Its sum runs in float32, because a bfloat16 sum stalls near 256 terms. "
                "The residual stream is a running sum, and pre-norm keeps an identity path through it. "
                "RMSNorm is bandwidth bound, so only fewer bytes make it faster. "
                "Next, rotary embeddings put position into the queries and keys."
            ),
            book_section="s-what-goes-wrong",
        )

    # ------------------------------------------------------------------
    # Introduction and the operation

    def norms_everywhere(self):
        self.slide("norms-everywhere", "Every forward pass runs 129 norms")
        g = kit.grid(2, 64, cell=(0.135, 0.42), gap=0.025)
        final = Rectangle(width=0.32, height=g.height, stroke_color=th.RULE_STRONG,
                          stroke_width=th.STROKE_HAIRLINE, fill_color=th.RAISE, fill_opacity=1)
        final.next_to(g, RIGHT, buff=0.3)
        cap = kit.label("64 layers", size=th.SIZE_LABEL, color=th.INK_3).next_to(g, UP, buff=0.15).align_to(g, LEFT)
        top_t = kit.label("top", size=th.SIZE_LABEL, color=th.INK_3).next_to(final, UP, buff=0.15)
        figure = VGroup(g, final, cap, top_t)
        key = VGroup(swatch("before the token mixer", self.colors[0]), swatch("before the MLP", self.colors[1]),
                     swatch("final norm", self.colors[2])).arrange(RIGHT, buff=0.6)
        count = kit.formula(r"64 \times 2 + 1 = 129", size=th.SIZE_MATH)
        each = kit.prose(r"Each one: 4 operations per element, and one sum over 5120 terms at 8 bits of precision.",
                         size=th.SIZE_SMALL, width=None, color=th.INK_2)
        col = VGroup(figure, key, count, each).arrange(DOWN, buff=0.5)
        self.place(col)
        with self.narrate(
            "This model has 64 layers, and each one normalizes its input twice. "
            "The first norm sits before the token mixer, the sublayer that moves information between tokens. "
            "The second sits before the MLP. "
            "One more norm sits at the top of the stack. "
            "That's 129 norms per forward pass. "
            "Each one does 4 operations per element, and one sum over 5120 terms, at 8 bits of precision. "
            "That's enough to break it."
        ) as t:
            self.play(FadeIn(g), FadeIn(final), FadeIn(cap), FadeIn(top_t), run_time=0.8)
            t.wait_until("The first norm")
            self.play(FadeIn(key[0]), LaggedStart(*[c.square.animate.set_fill(th.tint(self.colors[0], 0.4))
                                                    for c in g.row(0)], lag_ratio=0.03), run_time=1.6)
            t.wait_until("The second sits")
            self.play(FadeIn(key[1]), LaggedStart(*[c.square.animate.set_fill(th.tint(self.colors[1], 0.4))
                                                    for c in g.row(1)], lag_ratio=0.03), run_time=1.6)
            t.wait_until("One more norm")
            self.play(FadeIn(key[2]), final.animate.set_fill(th.tint(self.colors[2], 0.4)), run_time=0.7)
            t.wait_until("That's 129 norms")
            self.play(FadeIn(count, shift=UP * 0.1), run_time=0.8)
            t.wait_until("Each one does")
            self.play(FadeIn(each), run_time=0.8)

    def three_four(self):
        self.slide("three-four", "Normalizing the row (3, 4), step by step", book_section="s-the-operation")
        cell = (1.25, 0.62)
        mobs = [vec([3, 4], cell), vec([9, 16], cell, color=MS_C), vec(["12.5"], cell, color=MS_C),
                vec(["3.536"], cell, color=MS_C), vec(["0.849", "1.131"], cell, color=self.color)]
        names = ["row x", "square each entry", "mean of squares", "square root", "divide x by it"]
        labels = [kit.label(n, size=th.SIZE_SMALL, color=th.INK_2) for n in names]
        rows = VGroup()
        for i, (lab, m) in enumerate(zip(labels, mobs)):
            y = -i * 0.95
            m.move_to([0, y, 0]).align_to([0.0, 0, 0], LEFT)
            lab.move_to([0, y, 0]).align_to([-0.5, 0, 0], RIGHT)
            rows.add(VGroup(lab, m))
        check = kit.formula(r"\sqrt{\tfrac{0.849^2 + 1.131^2}{2}} = 1", size=th.SIZE_MATH_SMALL)
        check_t = kit.label("root mean square of the result", size=th.SIZE_LABEL, color=th.INK_3)
        side = VGroup(check_t, check).arrange(DOWN, buff=0.25)
        side.next_to(mobs[4], RIGHT, buff=1.0)
        layout = VGroup(rows, side)
        self.place(layout)
        ar = Arrow(mobs[4].get_right(), side.get_left(), buff=0.15, color=th.INK_4, stroke_width=3, tip_length=0.18)
        with self.narrate(
            "RMSNorm takes one token's vector, measures the typical size of its entries, and divides by it. "
            "Take a row with two entries, 3 and 4. "
            "Square each entry, to get 9 and 16. "
            "Their mean is 12.5. "
            "Its square root, the root mean square, is about 3.536. "
            "Divide each entry by it, and you get about 0.849 and 1.131. "
            "The root mean square of that result is exactly 1. "
            "Every token leaves with the same typical size, whatever size it arrived with."
        ) as t:
            t.wait_until("Take a row")
            self.play(FadeIn(rows[0]), run_time=0.7)
            for i, cue in zip(range(1, 5), ["Square each", "Their mean", "Its square root", "Divide each"]):
                t.wait_until(cue)
                self.play(FadeIn(labels[i]), TransformFromCopy(mobs[i - 1], mobs[i]), run_time=0.9)
            t.wait_until("The root mean square of that")
            self.play(GrowArrow(ar), FadeIn(side, shift=RIGHT * 0.1), run_time=0.9)
            t.wait_until("Every token leaves")
            self.play(Circumscribe(mobs[4], color=self.color, buff=0.08), run_time=1.2)

    def the_formula(self):
        self.slide("the-formula", "Divide by the root mean square, then apply a gain", book_section="s-the-operation")
        main = kit.formula(rf"y_i = \frac{{x_i}}{{\sqrt{{{MS} + {EPS}}}}}\; {C('w_i', GAIN_C)}",
                           size=th.SIZE_MATH + 8)
        ms_def = kit.formula(rf"{MS} = \frac{{1}}{{d}} \sum_{{j=1}}^{{d}} x_j^2", size=th.SIZE_MATH)
        top = VGroup(main, ms_def).arrange(RIGHT, buff=1.2)
        parts = kit.bullets(
            [
                rf"${MS}$: the mean of squares over $d$ channels. For the residual stream, $d = 5120$.",
                rf"${EPS} = 10^{{-6}}$: the config's \texttt{{rms\_norm\_eps}}.",
                rf"${C('w', GAIN_C)}$: a learned gain, one per channel, initialized to all ones.",
            ],
            size=th.SIZE_SMALL, width=10.5,
        )
        col = VGroup(top, parts).arrange(DOWN, buff=0.8)
        self.place(col)
        with self.narrate(
            "As a formula, divide each element by the root of the mean square plus a small constant. "
            "Then multiply by a gain. "
            "The mean of squares runs over d channels. "
            "For the residual-stream norms, d is 5120. "
            "The constant epsilon is 1e-6 in this config. "
            "The gain w is a learned vector, one value per channel, initialized to all ones."
        ) as t:
            self.play(FadeIn(main, shift=UP * 0.1), run_time=0.9)
            t.wait_until("The mean of squares runs")
            self.play(FadeIn(ms_def), FadeIn(parts[0], shift=RIGHT * 0.1), run_time=0.9)
            t.wait_until("The constant epsilon")
            self.play(FadeIn(parts[1], shift=RIGHT * 0.1), run_time=0.7)
            t.wait_until("The gain w")
            self.play(FadeIn(parts[2], shift=RIGHT * 0.1), run_time=0.7)

    def scale_not_shift(self):
        self.slide("scale-not-shift", "A scale changes nothing; a shift changes the output",
                   book_section="s-the-operation")
        cell = (1.15, 0.6)
        cases = [
            (r"$x$", [3, 4], ["0.849", "1.131"], "same", th.INK_4),
            (r"$10x$", [30, 40], ["0.849", "1.131"], "same output", th.GOOD),
            (r"$x + 1$", [4, 5], ["0.883", "1.104"], "different output", th.BAD),
        ]
        cols = VGroup()
        for head, a, b, verdict, color in cases:
            h = kit.prose(head, size=th.SIZE_BODY, width=None, color=th.INK)
            vin = vec(a, cell)
            vout = vec(b, cell, color=self.color)
            arr = Arrow(UP * 0.45, DOWN * 0.45, buff=0, color=th.INK_4, stroke_width=3, tip_length=0.18)
            nm = kit.label("RMSNorm", size=th.SIZE_LABEL, color=th.INK_3)
            mid = VGroup(arr, nm).arrange(RIGHT, buff=0.15)
            pill = kit.pill(verdict, color, size=th.SIZE_LABEL) if verdict != "same" else VGroup()
            c = VGroup(h, vin, mid, vout).arrange(DOWN, buff=0.3)
            arr_x = vin.get_x()
            mid.shift(RIGHT * (arr_x - arr.get_x()))
            if len(pill):
                pill.next_to(vout, DOWN, buff=0.3)
                c.add(pill)
            c.pill = pill
            cols.add(c)
        cols.arrange(RIGHT, buff=1.0, aligned_edge=UP)
        local = kit.prose(r"\textbf{Row-local}: rows are independent, so the kernel runs one row per thread block.",
                          size=th.SIZE_SMALL, width=None)
        col = VGroup(cols, local).arrange(DOWN, buff=0.6)
        self.place(col)
        with self.narrate(
            "Three properties follow. "
            "First, it's scale invariant. "
            "Multiply the row by 10, and the output doesn't change, up to epsilon. "
            "Whatever magnitude the previous layer produced is discarded. "
            "Second, it's not shift invariant. "
            "Add 1 to every entry, and the output moves. "
            "Third, it's row-local. "
            "Rows are independent, so the kernel runs one row per thread block."
        ) as t:
            self.play(FadeIn(cols[0]), run_time=0.8)
            t.wait_until("Multiply the row")
            self.play(FadeIn(cols[1][:3]), run_time=0.7)
            self.play(TransformFromCopy(cols[1][1], cols[1][3]), run_time=0.8)
            t.wait_until("Whatever magnitude")
            self.play(FadeIn(cols[1].pill, scale=0.9), run_time=0.6)
            t.wait_until("Add 1 to every")
            self.play(FadeIn(cols[2][:3]), run_time=0.7)
            self.play(TransformFromCopy(cols[2][1], cols[2][3]), run_time=0.8)
            self.play(FadeIn(cols[2].pill, scale=0.9), run_time=0.6)
            t.wait_until("Third, it's row-local")
            self.play(FadeIn(local), run_time=0.8)

    # ------------------------------------------------------------------
    # LayerNorm versus RMSNorm

    def layernorm(self):
        self.slide("layernorm", "LayerNorm also subtracts the mean and adds a shift",
                   book_section="s-layernorm-versus-rmsnorm")
        mu = C(r"\mu", th.INFO)
        beta = C(r"\beta_i", th.INFO)
        ln = kit.formula(rf"y_i = \frac{{x_i - {mu}}}{{\sqrt{{\sigma^2 + \epsilon}}}}\,\gamma_i + {beta}",
                         size=th.SIZE_MATH + 4)
        rel = kit.formula(rf"\sigma^2 = {MS} - \mu^2", size=th.SIZE_MATH)
        top = VGroup(ln, rel).arrange(RIGHT, buff=1.4)
        saves = kit.callout(
            "result",
            r"$129 \times 5120 = 660{,}480$ shift parameters, 0.0025\% of 26.9B. One accumulator fewer. "
            r"Simpler, and no slower.",
            title="What dropping the centering saves", width=6.0, size=th.SIZE_SMALL,
        )
        costs = kit.callout(
            "warning",
            r"An invariance. A constant offset added to every channel now survives, and models use it. "
            r"Measured quality: no difference.",
            title="What it costs", width=6.0, size=th.SIZE_SMALL,
        )
        cards = VGroup(saves, costs).arrange(RIGHT, buff=0.4, aligned_edge=UP)
        col = VGroup(top, cards).arrange(DOWN, buff=0.7)
        self.place(col)
        with self.narrate(
            "LayerNorm, from older models, does more. "
            "LayerNorm subtracts the row's mean, divides by the standard deviation, and adds a learned shift. "
            "Set the mean and the shift to zero, and you get RMSNorm. "
            "The variance is the mean of squares minus the squared mean, so the two agree when the mean is zero. "
            "Dropping the shift saves about 0.0025 percent of the parameters, and one accumulator. "
            "The kernel is bandwidth bound, so RMSNorm is simpler and no slower, not much faster. "
            "What it costs is an invariance: a constant offset now survives, and models use it. "
            "The measured quality difference is nil, as an empirical finding."
        ) as t:
            t.wait_until("LayerNorm subtracts")
            self.play(FadeIn(ln, shift=UP * 0.1), run_time=0.9)
            t.wait_until("Set the mean")
            self.play(Indicate(ln, color=th.INFO, scale_factor=1.03), run_time=1.0)
            t.wait_until("The variance is")
            self.play(FadeIn(rel), run_time=0.8)
            t.wait_until("Dropping the shift")
            self.play(FadeIn(saves, shift=UP * 0.1), run_time=0.8)
            t.wait_until("What it costs")
            self.play(FadeIn(costs, shift=UP * 0.1), run_time=0.8)

    def epsilon(self):
        self.slide("epsilon", "Epsilon keeps an all-zero row finite", book_section="s-layernorm-versus-rmsnorm")
        cell = (0.95, 0.56)
        zero = vec([0, 0, 0, 0], cell)
        nan = vec(["NaN"] * 4, cell)
        z_t = kit.label("a padded row", size=th.SIZE_LABEL, color=th.INK_3)
        n_t = kit.label("every later layer, then the logits", size=th.SIZE_LABEL, color=th.BAD)
        left = VGroup(VGroup(z_t, zero).arrange(DOWN, buff=0.15), VGroup(n_t, nan).arrange(DOWN, buff=0.15))
        left.arrange(RIGHT, buff=2.4)
        why = kit.formula(r"\operatorname{ms}(x) = 0 \;\Rightarrow\; 1/\sqrt{0}", size=th.SIZE_MATH_SMALL)
        forms = [
            (rf"\frac{{1}}{{\sqrt{{{MS} + {EPS}}}}}", "matches the checkpoint", th.GOOD),
            (rf"\frac{{1}}{{\sqrt{{\operatorname{{ms}}(x)}} + \epsilon}}", "finite, but wrong", th.BAD),
            (rf"\frac{{1}}{{\sqrt{{\operatorname{{ms}}(x) + d\,\epsilon}}}}", "finite, but wrong", th.BAD),
        ]
        boxes = VGroup()
        for tex, note, color in forms:
            f = kit.formula(tex, size=th.SIZE_MATH_SMALL)
            n = kit.label(note, size=th.SIZE_LABEL, color=color)
            boxes.add(VGroup(f, n).arrange(DOWN, buff=0.25))
        boxes.arrange(RIGHT, buff=1.1, aligned_edge=DOWN)
        col = VGroup(left, boxes).arrange(DOWN, buff=1.1)
        self.place(col)
        ar = kit.arrow(zero, nan, color=th.BAD)
        why.next_to(ar.line, DOWN, buff=0.25)
        with self.narrate(
            "Epsilon keeps the reciprocal square root finite when a row is all zeros. "
            "That happens more than you'd expect: padding, masked tokens, and channels a dead head never writes. "
            "Without epsilon, the mean of squares is zero, and you divide by zero. "
            "The row becomes not a number. "
            "It flows through every later layer and into the logits. "
            "The reference adds epsilon inside the square root. "
            "Adding it outside the root, or scaling it by d, is also finite. "
            "Only the first matches the checkpoint."
        ) as t:
            t.wait_until("That happens more")
            self.play(FadeIn(zero), FadeIn(z_t), run_time=0.7)
            t.wait_until("Without epsilon")
            self.play(FadeIn(why), GrowArrow(ar.line), run_time=0.9)
            t.wait_until("The row becomes")
            self.play(FadeIn(nan), nan.fill(nan.row(0), th.BAD), run_time=0.9)
            t.wait_until("It flows through")
            self.play(FadeIn(n_t), Indicate(nan, color=th.BAD, scale_factor=1.04), run_time=1.0)
            t.wait_until("The reference adds")
            self.play(FadeIn(boxes[0], shift=UP * 0.1), run_time=0.7)
            t.wait_until("Adding it outside")
            self.play(FadeIn(boxes[1], shift=UP * 0.1), FadeIn(boxes[2], shift=UP * 0.1), run_time=0.8)
            t.wait_until("Only the first")
            self.play(Circumscribe(boxes[0], color=th.GOOD, buff=0.12), run_time=1.0)

    # ------------------------------------------------------------------
    # Why the reduction runs in float32

    def addend_vanishes(self):
        self.slide("addend-vanishes", "An addend vanishes once the sum is 256 times larger",
                   book_section="s-why-the-reduction-runs-in-float32")
        bits = list("0") + list("10000111") + list("0000000")
        bg = kit.grid(1, 16, cell=(0.42, 0.52), values=[bits])
        bg.paint(bg.cell(0, 0), th.INK_4)
        bg.paint(bg.block(0, 1, 1, 9), th.INFO)
        bg.paint(bg.block(0, 1, 9, 16), self.color)
        tags = VGroup()
        for (c0, c1), text in (((0, 1), "sign"), ((1, 9), "exponent: 8"), ((9, 16), "significand: 7 stored")):
            tg = kit.label(text, size=th.SIZE_LABEL, color=th.INK_3)
            tg.next_to(bg.block(0, 1, c0, c1), DOWN, buff=0.15)
            tags.add(tg)
        tags[0].align_to(bg.cell(0, 0), LEFT)
        b_t = kit.label("256 in bfloat16", size=th.SIZE_LABEL, color=th.INK_3).next_to(bg, LEFT, buff=0.3)
        bits_fig = VGroup(bg, tags, b_t)

        unit = 0.75
        lo, hi = 250, 264
        X = lambda v: (v - lo) * unit  # noqa: E731
        axis = Line([X(lo), 0, 0], [X(hi), 0, 0], color=th.INK_3, stroke_width=th.STROKE_AXIS)
        reps = [250, 251, 252, 253, 254, 255, 256, 258, 260, 262, 264]
        ticks = VGroup(*[Line([X(v), -0.12, 0], [X(v), 0.12, 0], color=th.INK_2, stroke_width=3) for v in reps])
        nums = VGroup(*[kit.label(str(v), size=th.SIZE_LABEL, color=th.INK_3).next_to([X(v), -0.15, 0], DOWN,
                                                                                       buff=0.4)
                        for v in (252, 256, 260, 264)])
        sp1 = kit.label("spacing 1", size=th.SIZE_LABEL, color=th.INK_3).move_to([X(253), 0.5, 0])
        sp2 = kit.label("spacing 2", size=th.SIZE_LABEL, color=th.INK_3).move_to([X(260), 0.5, 0])
        line_fig = VGroup(axis, ticks, nums, sp1, sp2)
        rule = kit.formula(r"a \le \tfrac{1}{2}\,\mathrm{ulp}(S) = 2^{e-8}", r"\;\Rightarrow\;",
                           r"\frac{S}{a} \ge 2^8 = 256", size=th.SIZE_MATH_SMALL)
        always = kit.label("always vanishes once S/a ≥ 512", size=th.SIZE_LABEL, color=th.INK_3)
        rule_col = VGroup(rule, always).arrange(DOWN, buff=0.2)
        col = VGroup(bits_fig, line_fig, rule_col).arrange(DOWN, buff=0.75)
        self.place(col)
        s_dot = Dot(axis.get_start() + RIGHT * X(256), radius=0.1, color=self.color)
        s_lab = kit.label("S = 256", size=th.SIZE_SMALL, color=self.color, weight=MEDIUM)
        s_lab.next_to(s_dot, UP, buff=0.5).shift(LEFT * 0.7)
        exact = Circle(radius=0.1, stroke_color=th.INK_2, stroke_width=3).move_to(axis.get_start() + RIGHT * X(257))
        e_lab = kit.label("257?", size=th.SIZE_LABEL, color=th.INK_2).next_to(exact, UP, buff=0.5).shift(RIGHT * 0.5)
        hop = CurvedArrow(s_dot.get_center() + UP * 0.15, exact.get_center() + UP * 0.15, angle=-1.6,
                          color=th.INK_3, stroke_width=3, tip_length=0.15)
        back = CurvedArrow(exact.get_center() + DOWN * 0.15, s_dot.get_center() + DOWN * 0.15, angle=-1.6,
                           color=th.BAD, stroke_width=3, tip_length=0.15)
        with self.narrate(
            "Now the numerics. "
            "Picture a calculator with three significant digits. "
            "Ask it for 1000 plus 1, and it shows 1000. "
            "bfloat16 is about that coarse. "
            "It has 7 stored significand bits, so 8 bits of precision with the implicit one. "
            "Between 256 and 512, neighboring values are 2 apart. "
            "Add 1 to 256, and the exact answer sits halfway between two of them. "
            "Rounding returns 256, and the addend is gone. "
            "In general, an addend can vanish once the sum is 256 times larger."
        ) as t:
            t.wait_until("bfloat16 is about")
            self.play(FadeIn(bg), FadeIn(b_t), run_time=0.8)
            t.wait_until("It has 7 stored")
            self.play(LaggedStart(*[FadeIn(tg) for tg in tags], lag_ratio=0.3), run_time=1.0)
            t.wait_until("Between 256 and 512")
            self.play(Create(axis), FadeIn(ticks), FadeIn(nums), run_time=0.9)
            self.play(FadeIn(sp1), FadeIn(sp2), run_time=0.6)
            t.wait_until("Add 1 to 256")
            self.play(FadeIn(s_dot, scale=0.5), FadeIn(s_lab), run_time=0.6)
            self.play(Create(hop), FadeIn(exact), FadeIn(e_lab), run_time=0.9)
            t.wait_until("Rounding returns")
            self.play(Create(back), e_lab.animate.set_color(th.BAD), Indicate(s_dot, color=th.BAD), run_time=1.0)
            t.wait_until("In general")
            self.play(FadeIn(rule, shift=UP * 0.1), run_time=0.8)
            self.play(FadeIn(always), run_time=0.5)

    def the_stall(self):
        self.slide("the-stall", "A left-to-right bf16 sum stalls at 256 terms",
                   book_section="s-why-the-reduction-runs-in-float32")
        ax = kit.axes([0, 5120, 1280], [0, 5120, 1280], width=6.4, height=4.0,
                      x_label=r"\text{terms added } k", y_label=r"\text{running sum } S_k / s",
                      x_numbers=[0, 1280, 2560, 3840, 5120], y_numbers=[0, 2560, 5120])
        exact = ax.plot(lambda x: x, x_range=[0, 5120], color=th.INFO, stroke_width=th.STROKE_CURVE)
        k = ValueTracker(0)

        def bf16_path():
            kk = max(k.get_value(), 1.0)
            pts = [ax.c2p(0, 0), ax.c2p(min(kk, 256), min(kk, 256)), ax.c2p(kk, min(kk, 256))]
            path = VMobject(color=self.color, stroke_width=th.STROKE_CURVE + 1)
            path.set_points_as_corners(pts)
            return path

        stall = always_redraw(bf16_path)
        e_lab = kit.label("exact: 5120 s", size=th.SIZE_LABEL, color=th.INFO)
        e_lab.next_to(ax.c2p(5120, 5120), LEFT, buff=0.2)
        b_lab = kit.label("bfloat16: stuck at 256 s", size=th.SIZE_LABEL, color=self.color)
        b_lab.next_to(ax.c2p(5120, 256), UP, buff=0.15).align_to(ax.c2p(5120, 0), RIGHT)
        plot = VGroup(ax, exact, e_lab, b_lab)
        d = kit.Derivation(
            [
                r"\widehat{\operatorname{ms}} &\approx \frac{256\,s}{5120} = \frac{s}{20}",
                r"\frac{1}{\sqrt{20}} &\approx 0.224",
                r"\frac{1}{0.224} &\approx 4.5\times \text{ too large}",
            ],
            size=th.SIZE_MATH_SMALL, buff=0.45,
        )
        bias = kit.bullets([r"One-sided: a bias, not noise.", r"Fine at 256 channels, broken at 5120."],
                           size=th.SIZE_SMALL, width=4.6, buff=0.2)
        side = VGroup(d, bias).arrange(DOWN, buff=0.6, aligned_edge=LEFT)
        layout = VGroup(plot, side).arrange(RIGHT, buff=0.7)
        self.place(layout)
        with self.narrate(
            "Apply that to the sum inside RMSNorm. "
            "Run it left to right over 5120 squared activations, each about the same size, s. "
            "After k additions, the sum is about k times s. "
            "At 256 terms, it reaches the threshold. "
            "From then on, every addend rounds away, and the sum stalls. "
            "The mean of squares comes out 20 times too small. "
            "So the output is about 4.5 times too large. "
            "Dropping a positive addend only makes the sum smaller, so averaging never cancels it. "
            "And it depends on length: 256 channels are fine, 5120 are crippled."
        ) as t:
            self.play(Create(ax), run_time=0.9)
            t.wait_until("After k additions")
            self.add(stall)
            self.play(Create(exact), k.animate.set_value(256), run_time=t.until("At 256 terms") + 0.4)
            t.wait_until("From then on")
            self.play(k.animate.set_value(5120), run_time=2.5, rate_func=linear)
            self.play(FadeIn(e_lab), FadeIn(b_lab), run_time=0.6)
            t.wait_until("The mean of squares comes")
            self.play(d.show(0), run_time=0.8)
            t.wait_until("So the output is")
            self.play(d.show(1), run_time=0.7)
            self.play(d.show(2), run_time=0.7)
            t.wait_until("Dropping a positive")
            self.play(FadeIn(bias[0], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("And it depends on length")
            self.play(FadeIn(bias[1], shift=RIGHT * 0.1), run_time=0.6)

    def tree_reduction(self):
        self.slide("tree-reduction", "A tree reduction rescues most of it",
                   book_section="s-why-the-reduction-runs-in-float32")
        levels = [8, 4, 2, 1]
        tree = VGroup()
        nodes = []
        w = 0.5
        for li, n in enumerate(levels):
            row = []
            span = 8 * w
            for i in range(n):
                x = -span / 2 + (i + 0.5) * span / n
                sq = Square(0.36, stroke_color=th.RULE_STRONG, stroke_width=th.STROKE_HAIRLINE,
                            fill_color=th.tint(self.colors[1] if li == 0 else self.color, 0.35), fill_opacity=1)
                sq.move_to([x, -li * 0.85, 0])
                row.append(sq)
            nodes.append(row)
        edges = VGroup()
        for li in range(1, len(levels)):
            for i, parent in enumerate(nodes[li]):
                for child in nodes[li - 1][2 * i: 2 * i + 2]:
                    edges.add(Line(child.get_bottom(), parent.get_top(), color=th.INK_4, stroke_width=2))
        tree.add(edges, *[VGroup(*r) for r in nodes])
        tree_t = kit.label("slices, then pairs of partial sums", size=th.SIZE_LABEL, color=th.INK_3)
        tree_fig = VGroup(tree_t, tree).arrange(DOWN, buff=0.3)
        tab = kit.table(
            [
                ["bf16, sequential", r"$5120 \times 2^{-8}$", "20: no bound"],
                ["bf16, pairwise", r"$12.3 \times 2^{-8}$", "about 0.048"],
                ["float32, pairwise", r"$12.3 \times 2^{-24}$", r"about $7.3 \times 10^{-7}$"],
            ],
            header=["Scheme", "Error bound", "Value"],
            size=th.SIZE_SMALL, col_gap=0.5,
        )
        top = VGroup(tree_fig, tab).arrange(RIGHT, buff=0.8)
        key = kit.callout("key", r"bf16 leaves a few percent of low bias, about $10^{-2}$ measured: enough to "
                                 r"flip an argmax. The float32 bound sits far below the lab's $10^{-5}$ tolerance.",
                          title="Why the reference upcasts", width=10.5, size=th.SIZE_SMALL)
        col = VGroup(top, key).arrange(DOWN, buff=0.6)
        self.place(col)
        with self.narrate(
            "So why does the lab measure an error near 1e-2, and not 4.5 times? "
            "Because GPU reductions don't run left to right. "
            "They sum separate slices, then add the partial sums in pairs, like a tournament bracket. "
            "A sequential sum's error bound in bfloat16 is no bound at all. "
            "A tree cuts it to about 0.048. "
            "In float32, it's about 7.3e-7. "
            "What survives in bfloat16 is a few percent of low bias, enough to flip an argmax. "
            "The float32 bound is far below the lab's tolerance, so the reference upcasts."
        ) as t:
            t.wait_until("They sum separate")
            self.play(FadeIn(tree_t), FadeIn(VGroup(*nodes[0])), run_time=0.6)
            for li in range(1, len(levels)):
                new_edges = VGroup(*[e for e in edges if abs(e.get_end()[1] - nodes[li][0].get_top()[1]) < 1e-3])
                self.play(Create(new_edges), FadeIn(VGroup(*nodes[li]), shift=DOWN * 0.05), run_time=0.6)
            t.wait_until("A sequential sum's")
            self.play(FadeIn(tab.rules), *[FadeIn(c) for c in tab.cells[0]], *[FadeIn(c) for c in tab.cells[1]],
                      run_time=0.8)
            t.wait_until("A tree cuts it")
            self.play(*[FadeIn(c) for c in tab.cells[2]], run_time=0.7)
            t.wait_until("In float32, it's")
            self.play(*[FadeIn(c) for c in tab.cells[3]], run_time=0.7)
            t.wait_until("What survives")
            self.play(FadeIn(key, shift=UP * 0.1), run_time=0.8)

    def reference_code(self):
        self.slide("reference-code", "Widen to float32, reduce, round back once",
                   book_section="s-why-the-reduction-runs-in-float32")
        panel = kit.code(REF, "python", caption="reference rms_norm")
        notes = kit.bullets(
            [
                r"\texttt{variance} is the mean of squares: nothing is subtracted.",
                r"float32 lives only in registers. HBM traffic stays bf16 both ways.",
            ],
            size=th.SIZE_SMALL, width=11.0, buff=0.25,
        )
        col = VGroup(panel, notes).arrange(DOWN, buff=0.55)
        self.place(col)
        with self.narrate(
            "The reference code is short. "
            "It remembers the input's data type. "
            "It widens the row to float32. "
            "It reduces in float32, and keeps the last axis, so the result broadcasts. "
            "The variable called variance isn't one: it's the mean of squares. "
            "Then comes one reciprocal square root per row, and the gain. "
            "It rounds back to bfloat16 once, at the end. "
            "The bytes that cross to and from HBM stay bfloat16, so the upcast costs registers, not bandwidth."
        ) as t:
            self.play(FadeIn(panel.frame), *panel.reveal(1), run_time=0.7)
            t.wait_until("It remembers")
            self.play(*panel.reveal(2), run_time=0.6)
            t.wait_until("It widens")
            self.play(*panel.reveal(3), panel.highlight(3), run_time=0.7)
            t.wait_until("It reduces in float32")
            self.play(*panel.reveal(4), panel.highlight(4), run_time=0.7)
            t.wait_until("The variable called")
            self.play(FadeIn(notes[0], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("Then comes one")
            self.play(*panel.reveal(5), run_time=0.6)
            t.wait_until("It rounds back")
            self.play(*panel.reveal(6), panel.highlight(6), run_time=0.7)
            t.wait_until("The bytes that cross")
            self.play(FadeIn(notes[1], shift=RIGHT * 0.1), run_time=0.6)

    # ------------------------------------------------------------------
    # The residual stream

    def residual_layer(self):
        self.slide("residual-layer", "Each sublayer reads a normalized copy and adds back",
                   book_section="s-the-residual-stream")
        panel = kit.code(LAYER, "python", caption="one layer")
        # The stream runs left to right; each branch dips below it and comes back.
        y0, yb = 0.0, -1.35
        stream = Arrow([-6.3, y0, 0], [6.3, y0, 0], buff=0, color=th.INK_2, stroke_width=6, tip_length=0.25)
        s_lab = kit.label("residual stream x", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        s_lab.next_to([-6.3, y0, 0], UP, buff=0.2).align_to([-6.3, 0, 0], LEFT)
        p1, p2 = plus_node().move_to([-0.4, y0, 0]), plus_node().move_to([5.0, y0, 0])
        n1 = kit.box("RMSNorm", color=self.colors[0], width=1.9, height=0.7, size=th.SIZE_LABEL)
        m1 = kit.box("mixer", color=self.colors[1], width=1.6, height=0.7, size=th.SIZE_LABEL)
        n2 = kit.box("RMSNorm", color=self.colors[0], width=1.9, height=0.7, size=th.SIZE_LABEL)
        m2 = kit.box("MLP", color=self.colors[1], width=1.6, height=0.7, size=th.SIZE_LABEL)
        n1.move_to([-3.95, yb, 0])
        m1.move_to([-1.75, yb, 0])
        n2.move_to([1.45, yb, 0])
        m2.move_to([3.65, yb, 0])
        b1 = VGroup(elbow([(-5.4, y0), (-5.4, yb), n1.get_left()]),
                    Arrow(n1.get_right(), m1.get_left(), buff=0, color=th.INK_3, stroke_width=th.STROKE_BOX + 1,
                          tip_length=0.18),
                    elbow([m1.get_right(), (-0.4, yb), p1.get_bottom()]))
        b2 = VGroup(elbow([(0.0, y0), (0.0, yb), n2.get_left()]),
                    Arrow(n2.get_right(), m2.get_left(), buff=0, color=th.INK_3, stroke_width=th.STROKE_BOX + 1,
                          tip_length=0.18),
                    elbow([m2.get_right(), (5.0, yb), p2.get_bottom()]))
        diagram = VGroup(stream, s_lab, b1, b2, n1, m1, n2, m2, p1, p2)
        col = VGroup(panel, diagram).arrange(DOWN, buff=0.6)
        self.place(col)
        with self.narrate(
            "Now, what RMSNorm normalizes. "
            "Here's one layer of the model. "
            "The tensor x passes through untouched by anything except addition. "
            "The mixer reads a normalized copy, computes something, and adds its result back. "
            "Then the MLP does the same, with its own norm. "
            "That running sum is the residual stream. "
            "It's the model's working memory, and it's the reason a 64-layer network trains at all."
        ) as t:
            self.play(FadeIn(panel.frame), *panel.reveal(1), run_time=0.7)
            t.wait_until("The tensor x passes")
            self.play(GrowArrow(stream), FadeIn(s_lab), FadeIn(p1), FadeIn(p2), run_time=1.0)
            t.wait_until("The mixer reads")
            self.play(*panel.reveal(2, 3, 4), panel.highlight(2), run_time=0.8)
            self.play(Create(b1[0]), FadeIn(n1), run_time=0.6)
            self.play(GrowArrow(b1[1]), FadeIn(m1), run_time=0.5)
            self.play(Create(b1[2]), Indicate(p1, color=self.color), run_time=0.7)
            t.wait_until("Then the MLP")
            self.play(*panel.reveal(5), panel.highlight(5), run_time=0.7)
            self.play(Create(b2[0]), FadeIn(n2), run_time=0.5)
            self.play(GrowArrow(b2[1]), FadeIn(m2), run_time=0.5)
            self.play(Create(b2[2]), Indicate(p2, color=self.color), run_time=0.7)
            t.wait_until("That running sum")
            self.play(Indicate(stream, color=self.color, scale_factor=1.02), run_time=1.0)

    def running_sum(self):
        self.slide("running-sum", "The stream is the embedding plus 128 contributions",
                   book_section="s-the-residual-stream")
        cell = (1.0, 0.56)
        x0 = ["1", "-0.5", "0.5", "2"]
        deltas = [["0.5", "1", "-0.5", "0"], ["-1", "0.5", "1", "0.5"], ["0.5", "0.5", "0.5", "-1"]]
        totals = [["1.5", "0.5", "0", "2"], ["0.5", "1", "1", "2.5"], ["1", "1.5", "1.5", "1.5"]]
        rows = VGroup()
        lab_tex = [r"$x^{(0)}$", r"$+\,\Delta^{(1)}$", r"$+\,\Delta^{(2)}$", r"$+\,\Delta^{(3)}$"]
        grids = [vec(x0, cell, color=th.INK_4)] + [vec(dv, cell, color=DELTA_C) for dv in deltas]
        for tex, g in zip(lab_tex, grids):
            lab = kit.prose(tex, size=th.SIZE_SMALL, width=None, color=th.INK)
            rows.add(VGroup(lab, g))
        for r in rows:
            r[1].move_to([0, 0, 0]).align_to([0, 0, 0], LEFT)
            r[0].next_to(r[1], LEFT, buff=0.3)
        rows.arrange(DOWN, buff=0.22)
        for r in rows:
            r[1].align_to(rows[0][1], LEFT)
            r[0].next_to(r[1], LEFT, buff=0.3)
        rule = Line(LEFT * 2.0, RIGHT * 2.0, color=th.INK_3, stroke_width=th.STROKE_AXIS)
        rule.next_to(rows, DOWN, buff=0.25).align_to(rows[0][1], LEFT)
        stream = vec(x0, cell, color=self.color)
        stream.next_to(rule, DOWN, buff=0.25).align_to(rows[0][1], LEFT)
        s_lab = kit.label("stream", size=th.SIZE_SMALL, color=self.color, weight=MEDIUM).next_to(stream, LEFT,
                                                                                                 buff=0.3)
        left = VGroup(rows, rule, stream, s_lab)
        delta = C(r"\Delta^{(\ell)}", DELTA_C)
        sum_f = kit.formula(rf"x^{{(L)}} = x^{{(0)}} + \sum_{{\ell=1}}^{{L}} {delta}",
                            size=th.SIZE_MATH)
        box = SurroundingRectangle(sum_f, color=th.INK_3, buff=0.2, stroke_width=th.STROKE_AXIS)
        el = kit.formula(r"L = 64 \times 2 = 128", size=th.SIZE_MATH_SMALL)
        bus = kit.prose(r"A shared bus, 5120 channels wide: 128 writers append to it, 128 readers read it.",
                        size=th.SIZE_SMALL, width=5.0, color=th.INK_2)
        right = VGroup(VGroup(sum_f, box), el, bus).arrange(DOWN, buff=0.5)
        layout = VGroup(left, right).arrange(RIGHT, buff=1.0)
        self.place(layout)

        def show_total(i):
            new = vec(totals[i], cell, color=self.color).move_to(stream)
            self.play(FadeIn(rows[i + 1], shift=DOWN * 0.08), run_time=0.5)
            self.play(*[Transform(stream.cell(0, c).text, new.cell(0, c).text) for c in range(4)],
                      Indicate(rows[i + 1][1], color=DELTA_C, scale_factor=1.03), run_time=0.8)

        with self.narrate(
            "Write the stream after l sublayers as x of l. "
            "Each sublayer adds its contribution, delta, computed from a normalized copy. "
            "Here's a stream with four channels, starting from the embedding. "
            "The first sublayer adds its delta, and the stream updates. "
            "The second and third do the same. "
            "Nothing is ever overwritten, only accumulated. "
            "Unroll it, and the final state is the embedding plus every contribution. "
            "With 64 layers of two sublayers each, that's 128 contributions. "
            "Think of the stream as a shared bus, 5120 channels wide, that 128 writers append to."
        ) as t:
            t.wait_until("Here's a stream")
            self.play(FadeIn(rows[0]), FadeIn(rule), FadeIn(stream), FadeIn(s_lab), run_time=0.8)
            t.wait_until("The first sublayer")
            show_total(0)
            t.wait_until("The second and third")
            show_total(1)
            show_total(2)
            t.wait_until("Unroll it")
            self.play(FadeIn(sum_f), Create(box), run_time=0.9)
            t.wait_until("With 64 layers")
            self.play(FadeIn(el), run_time=0.6)
            t.wait_until("Think of the stream")
            self.play(FadeIn(bus), run_time=0.7)

    def stream_grows(self):
        self.slide("stream-grows", "The stream grows like the square root of depth",
                   book_section="s-the-residual-stream")
        gen = kit.rng(4)
        d = 5120
        x = gen.standard_normal(d)
        rms = [float(np.sqrt(np.mean(x ** 2)))]
        for _ in range(128):
            x = x + gen.standard_normal(d)
            rms.append(float(np.sqrt(np.mean(x ** 2))))
        ax = kit.axes([0, 128, 16], [0, 12, 2], width=6.8, height=4.0,
                      x_label=r"\text{contributions } \ell", y_label=r"\operatorname{rms}(x^{(\ell)}) / \sigma",
                      x_numbers=[0, 32, 64, 96, 128], y_numbers=[0, 4, 8, 12])
        curve = ax.plot(lambda v: np.sqrt(1 + v), x_range=[0, 128, 0.5], color=self.color,
                        stroke_width=th.STROKE_CURVE)
        dots = VGroup(*[Dot(ax.c2p(i, rms[i]), radius=0.055, color=th.INFO) for i in range(0, 129, 4)])
        end = Dot(ax.c2p(128, np.sqrt(129)), radius=0.09, color=th.INK)
        end_l = kit.label("≈ 11.4", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM).next_to(end, DOWN, buff=0.2)
        plot = VGroup(ax, curve, dots, end, end_l)
        f1 = kit.formula(r"\operatorname{rms}(x^{(L)}) \approx \sqrt{\operatorname{rms}(x^{(0)})^2 + L\sigma^2}",
                         size=th.SIZE_MATH_SMALL)
        f2 = kit.formula(r"= \sqrt{129}\,\sigma \approx 11.4\,\sigma", size=th.SIZE_MATH_SMALL)
        keys = VGroup(line_key("square-root rule", self.color),
                      VGroup(Dot(radius=0.07, color=th.INFO), kit.label("simulated, independent", size=th.SIZE_LABEL,
                                                                          color=th.INK_2)).arrange(RIGHT, buff=0.2))
        keys.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        note = kit.prose(r"A heuristic, not a measurement. Real contributions are correlated, but the direction "
                         r"holds.", size=th.SIZE_SMALL, width=4.8, color=th.INK_3)
        side = VGroup(f1, f2, keys, note).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        f2.shift(RIGHT * 0.6)
        layout = VGroup(plot, side).arrange(RIGHT, buff=0.6)
        self.place(layout)
        with self.narrate(
            "Because nothing subtracts, the stream tends to grow as it climbs the stack. "
            "Suppose the contributions are uncorrelated, and each has root mean square sigma. "
            "Then variances add. "
            "The dots are a simulation with independent random contributions, and the curve is the "
            "square-root rule. "
            "If the embedding also has size sigma, then after 128 contributions the stream is about 11.4 times "
            "larger. "
            "That's a heuristic, not a measurement, but the direction holds in practice. "
            "Later layers see larger inputs, which is why each sublayer normalizes its own input."
        ) as t:
            self.play(Create(ax), run_time=0.9)
            t.wait_until("Then variances add")
            self.play(FadeIn(f1), run_time=0.8)
            t.wait_until("The dots are")
            self.play(LaggedStart(*[FadeIn(dt, scale=0.5) for dt in dots], lag_ratio=0.05), FadeIn(keys[1]),
                      run_time=1.6)
            self.play(Create(curve), FadeIn(keys[0]), run_time=1.2)
            t.wait_until("If the embedding")
            self.play(FadeIn(f2), FadeIn(end, scale=0.5), FadeIn(end_l), run_time=0.8)
            t.wait_until("That's a heuristic")
            self.play(FadeIn(note), run_time=0.7)

    def final_norm(self):
        self.slide("final-norm", "The final norm pins the scale of the logits", book_section="s-the-residual-stream")

        def probs(title, values, color):
            bc = kit.bar_compare([(f"token {n}", v, color, f"{v:.3f}")
                                  for n, v in zip("ABC", values)], max_value=1.0, width=3.6, bar_height=0.45,
                                 gap=0.25)
            head = kit.prose(title, size=th.SIZE_SMALL, width=None, color=th.INK)
            g = VGroup(head, bc).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
            g.bc = bc
            return g

        a = probs(r"softmax of logits $(2, 1, 0)$", [0.665, 0.245, 0.090], th.INFO)
        b = probs(r"every logit $\times 2$: $(4, 2, 0)$", [0.867, 0.117, 0.016], self.color)
        pair = VGroup(a, b).arrange(RIGHT, buff=1.2, aligned_edge=UP)
        temp = kit.prose(r"Doubling every logit halves the sampling temperature.", size=th.SIZE_SMALL, width=None)
        warn = kit.callout("warning", r"Greedy decoding still looks fine, because argmax ignores a positive rescale. "
                                      r"Temperature and top-p don't.",
                           title="Skipping the final norm hides until you sample", width=11.0, size=th.SIZE_SMALL)
        col = VGroup(pair, temp, warn).arrange(DOWN, buff=0.5)
        self.place(col)
        with self.narrate(
            "The final norm runs once, before the head that turns the last hidden state into logits. "
            "Softmax cares about the logits' absolute scale. "
            "Take logits 2, 1, and 0. "
            "The top token gets about 0.67 of the probability. "
            "Double the logits, and it gets about 0.87. "
            "Doubling every logit halves the temperature, so a drifting stream means a drifting temperature. "
            "Skip the final norm, and greedy decoding still looks fine, because argmax ignores a rescale. "
            "Sampling with temperature won't."
        ) as t:
            t.wait_until("Take logits")
            self.play(FadeIn(a[0]), FadeIn(a.bc.labels), *a.bc.grow(), run_time=1.0)
            t.wait_until("The top token")
            self.play(FadeIn(a.bc.values), run_time=0.6)
            t.wait_until("Double the logits")
            self.play(FadeIn(b[0]), FadeIn(b.bc.labels), *b.bc.grow(), run_time=1.0)
            self.play(FadeIn(b.bc.values), run_time=0.6)
            t.wait_until("Doubling every logit")
            self.play(FadeIn(temp), run_time=0.7)
            t.wait_until("Skip the final norm")
            self.play(FadeIn(warn, shift=UP * 0.1), run_time=0.8)

    # ------------------------------------------------------------------
    # Pre-norm versus post-norm

    def placement(self, pre: bool):
        """A small block diagram: pre-norm (norm in the branch) or post-norm (norm on the stream)."""
        y0, yb = 0.0, -1.1
        x_end = 5.2
        stream = Arrow([0, y0, 0], [x_end, y0, 0], buff=0, color=th.INK_2, stroke_width=5, tip_length=0.22)
        plus = plus_node().move_to([3.1 if pre else 2.4, y0, 0])
        parts = VGroup(stream)
        if pre:
            n = kit.box("N", color=self.colors[0], width=0.8, height=0.6, size=th.SIZE_SMALL).move_to([1.0, yb, 0])
            f = kit.box("F", color=self.colors[1], width=0.8, height=0.6, size=th.SIZE_SMALL).move_to([2.1, yb, 0])
            path = VGroup(elbow([(0.3, y0), (0.3, yb), n.get_left()]),
                          Arrow(n.get_right(), f.get_left(), buff=0, color=th.INK_3, stroke_width=3,
                                tip_length=0.15),
                          elbow([f.get_right(), (3.1, yb), plus.get_bottom()]))
            parts.add(path, n, f, plus)
        else:
            f = kit.box("F", color=self.colors[1], width=0.8, height=0.6, size=th.SIZE_SMALL).move_to([1.2, yb, 0])
            n = kit.box("N", color=self.colors[0], width=0.8, height=0.6, size=th.SIZE_SMALL)
            n.move_to([3.9, y0, 0])
            path = VGroup(elbow([(0.3, y0), (0.3, yb), f.get_left()]),
                          elbow([f.get_right(), (2.4, yb), plus.get_bottom()]))
            parts.add(path, f, plus, n)
        return parts

    def pre_vs_post(self):
        self.slide("pre-vs-post", "Pre-norm keeps an identity path for the gradient",
                   book_section="s-pre-norm-versus-post-norm")
        ident = C("I", th.HL_A)
        nprime = C(r"N'_\ell", th.HL_B)
        pre = self.placement(True)
        post = self.placement(False)
        pre_t = kit.label("Pre-norm: this model", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
        post_t = kit.label("Post-norm: the original transformer", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
        pre_f = kit.formula(rf"\prod_{{\ell=1}}^{{L}} \left({ident} + J_\ell\right) = {ident} + "
                            rf"\sum_\ell J_\ell + \cdots", size=th.SIZE_MATH_SMALL)
        post_f = kit.formula(rf"\prod_{{\ell=1}}^{{L}} {nprime} \left(I + J_\ell\right)", size=th.SIZE_MATH_SMALL)
        lhs = kit.formula(r"\frac{\partial x^{(L)}}{\partial x^{(0)}}", size=th.SIZE_MATH_SMALL)
        left = VGroup(pre_t, pre, pre_f).arrange(DOWN, buff=0.45)
        right = VGroup(post_t, post, post_f).arrange(DOWN, buff=0.45)
        pair = VGroup(left, right).arrange(RIGHT, buff=1.1, aligned_edge=UP)
        post_f.align_to(pre_f, DOWN)
        lhs_t = VGroup(kit.label("gradient, top to bottom:", size=th.SIZE_LABEL, color=th.INK_3), lhs).arrange(
            RIGHT, buff=0.3)
        col = VGroup(pair, lhs_t).arrange(DOWN, buff=0.55)
        self.place(col)
        with self.narrate(
            "Why does the norm sit before each sublayer? "
            "This model uses pre-norm: it normalizes the sublayer's input, and leaves the stream alone. "
            "The original transformer used post-norm, which normalizes after the add. "
            "The difference shows up in the gradient through the stack. "
            "For pre-norm, each layer contributes the identity plus its sublayer's Jacobian. "
            "Expand the product, and the leading term is the identity, whatever the depth. "
            "There's always a path to every layer whose gradient is exactly 1. "
            "For post-norm, every factor carries the norm's Jacobian, so no identity term survives."
        ) as t:
            t.wait_until("This model uses pre-norm")
            self.play(FadeIn(pre_t), FadeIn(pre), run_time=1.0)
            t.wait_until("The original transformer")
            self.play(FadeIn(post_t), FadeIn(post), run_time=1.0)
            t.wait_until("The difference shows")
            self.play(FadeIn(lhs_t), run_time=0.7)
            t.wait_until("For pre-norm, each layer")
            self.play(FadeIn(pre_f), run_time=0.8)
            t.wait_until("There's always a path")
            self.play(Indicate(pre[0], color=th.HL_A, scale_factor=1.03), run_time=1.0)
            t.wait_until("For post-norm, every factor")
            self.play(FadeIn(post_f), Indicate(post[-1], color=th.HL_B), run_time=1.0)

    def c_to_the_64(self):
        self.slide("c-to-the-64", "Post-norm's gradient scales like c⁶⁴", book_section="s-pre-norm-versus-post-norm")
        ax = kit.axes([0, 64, 8], [0, 4, 1], width=7.2, height=4.0,
                      x_label=r"\text{layers}", x_numbers=[0, 16, 32, 48, 64])
        ylabs = VGroup()
        for v, txt in zip([0, 1, 2, 3, 4], ["0.01", "0.1", "1", "10", "100"]):
            lab = kit.math_label(txt, size=th.SIZE_LABEL - 2, color=th.INK_3)
            lab.next_to(ax.c2p(0, v), LEFT, buff=0.14)
            ylabs.add(lab)
        y_t = kit.label("gradient scale", size=th.SIZE_LABEL, color=th.INK_3).next_to(ax.c2p(0, 4), UP, buff=0.2)
        up = ax.plot(lambda l: 2 + l * np.log10(1.05), x_range=[0, 64], color=th.BAD, stroke_width=th.STROKE_CURVE)
        down = ax.plot(lambda l: 2 + l * np.log10(0.95), x_range=[0, 64], color=th.INFO, stroke_width=th.STROKE_CURVE)
        flat = DashedLine(ax.c2p(0, 2), ax.c2p(64, 2), color=th.GOOD, stroke_width=th.STROKE_CURVE)
        l_up = kit.label("22.7", size=th.SIZE_SMALL, color=th.BAD, weight=MEDIUM).next_to(ax.c2p(64, 3.356), RIGHT,
                                                                                         buff=0.15)
        l_dn = kit.label("0.038", size=th.SIZE_SMALL, color=th.INFO, weight=MEDIUM).next_to(ax.c2p(64, 0.574),
                                                                                           RIGHT, buff=0.15)
        plot = VGroup(ax, ylabs, y_t, up, down, flat, l_up, l_dn)
        keys = VGroup(line_key("post-norm, c = 1.05", th.BAD), line_key("post-norm, c = 0.95", th.INFO),
                      line_key("pre-norm identity path", th.GOOD, dashed=True)).arrange(DOWN, buff=0.22,
                                                                                        aligned_edge=LEFT)
        f2 = VGroup(kit.formula(r"0.95^{64} \approx 0.038", size=th.SIZE_MATH_SMALL),
                    kit.formula(r"1.05^{64} \approx 22.7", size=th.SIZE_MATH_SMALL)).arrange(DOWN, buff=0.35,
                                                                                            aligned_edge=LEFT)
        warm = kit.prose(r"So post-norm needs learning-rate warmup and careful initialization.",
                         size=th.SIZE_SMALL, width=4.4, color=th.INK_2)
        side = VGroup(keys, f2, warm).arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        layout = VGroup(plot, side).arrange(RIGHT, buff=0.9)
        self.place(layout)
        with self.narrate(
            "If a typical post-norm factor has gain c, the gradient over 64 layers scales like c to the 64th. "
            "Take a 5 percent error in either direction. "
            "At 0.95 per layer, the gradient shrinks to about 0.038. "
            "At 1.05 per layer, it grows to about 22.7. "
            "That's why post-norm transformers need warmup and careful initialization. "
            "Pre-norm's cost is the growing stream, and the final norm handles that."
        ) as t:
            self.play(Create(ax), FadeIn(ylabs), FadeIn(y_t), run_time=0.9)
            t.wait_until("Take a 5 percent")
            self.play(Create(flat), FadeIn(keys[2]), run_time=0.8)
            t.wait_until("At 0.95 per layer")
            self.play(Create(down), FadeIn(keys[1]), run_time=1.2)
            self.play(FadeIn(l_dn), FadeIn(f2[0]), run_time=0.6)
            t.wait_until("At 1.05 per layer")
            self.play(Create(up), FadeIn(keys[0]), run_time=1.2)
            self.play(FadeIn(l_up), FadeIn(f2[1]), run_time=0.6)
            t.wait_until("That's why post-norm")
            self.play(FadeIn(warm), run_time=0.7)
            t.wait_until("Pre-norm's cost")
            self.play(Indicate(flat, color=th.GOOD, scale_factor=1.02), run_time=1.0)

    # ------------------------------------------------------------------
    # The per-head query and key norm

    def qk_norm(self):
        self.slide("qk-norm", "Per-head norms cap every attention logit at 16",
                   book_section="s-the-per-head-query-and-key-norm")
        tab = kit.table(
            [
                [r"\texttt{q}, 24 heads", "24 per token", "256"],
                [r"\texttt{k}, 4 heads", "4 per token", "256"],
                ["residual stream", "1 per token", "5120"],
            ],
            header=["Tensor", "Rows", "Length"],
            size=th.SIZE_SMALL, col_gap=0.45,
        )
        d = kit.Derivation(
            [
                r"\text{logit} &= q \cdot k / \sqrt{256} = q \cdot k / 16",
                r"\lVert q \rVert = \lVert k \rVert &= \sqrt{256} = 16",
                r"\lvert q \cdot k \rvert &\le 16 \times 16 = 256",
                r"\lvert \text{logit} \rvert &\le 256 / 16 = 16",
            ],
            size=th.SIZE_MATH_SMALL, buff=0.4,
        )
        top = VGroup(tab, d).arrange(RIGHT, buff=0.8)
        cost = kit.prose(r"Reduction length 256: right at the stall threshold. Cost: about 17\% more norm traffic.",
                         size=th.SIZE_SMALL, width=None, color=th.INK_3)
        col = VGroup(top, cost).arrange(DOWN, buff=0.7)
        self.place(col)
        box = SurroundingRectangle(d.lines[3], color=self.color, buff=0.12, stroke_width=th.STROKE_AXIS)
        with self.narrate(
            "RMSNorm runs in one more place. "
            "This model normalizes each attention head's queries and keys over the 256-wide head dimension. "
            "There are 24 query heads and 4 key heads. "
            "An attention logit is q dot k over 16. "
            "After the norm, each vector has length 16. "
            "By Cauchy-Schwarz, q dot k is at most 256 in magnitude. "
            "Divide by 16, and every logit is at most 16, so softmax can't overflow or collapse by accident. "
            "These norms add about 17 percent to normalization traffic."
        ) as t:
            t.wait_until("There are 24 query")
            self.play(FadeIn(tab.rules), *[FadeIn(VGroup(*r)) for r in tab.cells], run_time=0.9)
            t.wait_until("An attention logit")
            self.play(d.show(0), run_time=0.8)
            t.wait_until("After the norm")
            self.play(d.show(1), run_time=0.8)
            t.wait_until("By Cauchy-Schwarz")
            self.play(d.show(2), run_time=0.8)
            t.wait_until("Divide by 16")
            self.play(d.show(3), run_time=0.8)
            self.play(Create(box), run_time=0.6)
            t.wait_until("These norms add")
            self.play(FadeIn(cost), run_time=0.7)

    # ------------------------------------------------------------------
    # Where the time goes

    def bandwidth_bound(self):
        self.slide("bandwidth-bound", "At 1 FLOP per byte, RMSNorm waits on memory",
                   book_section="s-where-the-time-goes")
        ridge = np.log10(312e12 / 1935e9)
        ax = kit.axes([0, 4, 1], [0, 4, 1], width=5.6, height=4.0)
        P = lambda a, b: ax.c2p(a + 1, b + 1)  # noqa: E731
        xl = VGroup(*[kit.math_label(s, size=th.SIZE_LABEL - 2, color=th.INK_3).next_to(P(v, -1), DOWN, buff=0.14)
                      for v, s in zip(range(-1, 4), ["0.1", "1", "10", "100", "1000"])])
        yl = VGroup(*[kit.math_label(s, size=th.SIZE_LABEL - 2, color=th.INK_3).next_to(P(-1, v), LEFT, buff=0.14)
                      for v, s in zip(range(-1, 4), ["0.1", "1", "10", "100", "1000"])])
        x_t = kit.label("FLOPs per byte", size=th.SIZE_LABEL, color=th.INK_3).next_to(xl, DOWN, buff=0.15)
        y_t = kit.label("TFLOP/s", size=th.SIZE_LABEL, color=th.INK_3).next_to(P(-1, 3), UP, buff=0.15)
        lg = np.log10(1.935)
        roof = VMobject(color=th.INK_2, stroke_width=th.STROKE_CURVE)
        roof.set_points_as_corners([P(-1, lg - 1), P(ridge, np.log10(312)), P(3, np.log10(312))])
        rl = DashedLine(P(ridge, -1), P(ridge, np.log10(312)), color=th.INK_4, stroke_width=th.STROKE_AXIS)
        rl_t = kit.label("ridge 161", size=th.SIZE_LABEL, color=th.INK_3).next_to(rl, RIGHT, buff=0.1).shift(DOWN * 0.6)
        pt = Dot(P(0, lg), radius=0.1, color=self.color)
        pt_t = kit.label("RMSNorm", size=th.SIZE_SMALL, color=self.color, weight=MEDIUM).next_to(pt, RIGHT, buff=0.2)
        pt_t.shift(DOWN * 0.35)
        plot = VGroup(ax, xl, yl, x_t, y_t, roof, rl, rl_t, pt, pt_t)
        ratio = kit.formula(r"\frac{4\ \text{FLOP}}{4\ \text{bytes}} = 1\ \text{FLOP/byte}", size=th.SIZE_MATH_SMALL)
        share = kit.prose(r"1.94 TFLOP/s at 1935 GB/s: 0.6\% of the A100's 312.", size=th.SIZE_SMALL, width=5.6)
        ex_t = kit.label("128 norms, 4096-token prefill, 83.9 MB each", size=th.SIZE_LABEL, color=th.INK_3)
        bc = kit.bar_compare([("copy ceiling", 8.42, th.INFO, "8.4 ms"), ("Triton kernel", 11.36, self.color,
                                                                          "11.4 ms")],
                             width=2.8, bar_height=0.42, gap=0.22, size=th.SIZE_SMALL)
        side = VGroup(ratio, share, VGroup(ex_t, bc).arrange(DOWN, buff=0.25, aligned_edge=LEFT)).arrange(
            DOWN, buff=0.55, aligned_edge=LEFT)
        layout = VGroup(plot, side).arrange(RIGHT, buff=0.8)
        self.place(layout)
        with self.narrate(
            "Where does the time go? "
            "Per element, RMSNorm moves 4 bytes and does about 4 FLOPs. "
            "That's 1 FLOP per byte. "
            "The A100's ridge point is 161, so RMSNorm sits far down the memory slope. "
            "At the rated bandwidth, it reaches 0.6 percent of peak arithmetic. "
            "For scale, take 128 norms over a 4096-token prefill. "
            "At the copy ceiling, they take 8.4 ms. "
            "The Triton kernel from chapter 13 takes 11.4 ms."
        ) as t:
            self.play(Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(x_t), FadeIn(y_t), run_time=0.9)
            t.wait_until("That's 1 FLOP per byte")
            self.play(FadeIn(ratio), run_time=0.7)
            t.wait_until("The A100's ridge")
            self.play(Create(roof), Create(rl), FadeIn(rl_t), run_time=1.2)
            self.play(FadeIn(pt, scale=0.5), FadeIn(pt_t), run_time=0.6)
            t.wait_until("At the rated")
            self.play(FadeIn(share), run_time=0.7)
            t.wait_until("For scale")
            self.play(FadeIn(ex_t), FadeIn(bc.labels), run_time=0.6)
            t.wait_until("At the copy ceiling")
            self.play(GrowFromEdge(bc.bars[0], LEFT), FadeIn(bc.values[0]), run_time=0.8)
            t.wait_until("The Triton kernel")
            self.play(GrowFromEdge(bc.bars[1], LEFT), FadeIn(bc.values[1]), run_time=0.8)

    def fusion(self):
        self.slide("fusion", "Fusion pays only when the sum spills out of L2",
                   book_section="s-where-the-time-goes")

        def trips(names):
            row = VGroup()
            for n in names:
                c = th.INFO if n.startswith("read") else self.color
                row.add(kit.box(n, color=c, width=1.9, height=0.62, size=th.SIZE_LABEL))
            return row.arrange(RIGHT, buff=0.15)

        un = trips(["read output", "read residual", "write sum", "read sum", "write normed"])
        fu = trips(["read output", "read residual", "write sum", "write normed"])
        un_t = kit.label("Unfused: 5 trips", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        fu_t = kit.label("Fused: 4 trips", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        pred = kit.formula(r"1 - \tfrac{4}{5} = 20\%\ \text{predicted}", size=th.SIZE_MATH_SMALL)
        top = VGroup(VGroup(un_t, un).arrange(DOWN, buff=0.2, aligned_edge=LEFT),
                     VGroup(fu_t, fu).arrange(DOWN, buff=0.2, aligned_edge=LEFT)).arrange(DOWN, buff=0.35,
                                                                                          aligned_edge=LEFT)
        pred.next_to(top[1][1], RIGHT, buff=0.6)
        bc = kit.bar_compare([("sum is 41.9 MB, past the 40 MB L2", 1.10, th.GOOD, "1.10× faster"),
                              ("sum fits in L2", 0.92, th.BAD, "0.92×, slower")],
                             width=4.5, max_value=1.2, limit=1.0, limit_label="unfused", bar_height=0.45, gap=0.3)
        m_t = kit.label("Measured on the A100", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        bottom = VGroup(m_t, bc).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        col = VGroup(VGroup(top, pred), bottom).arrange(DOWN, buff=0.6, aligned_edge=LEFT)
        self.place(col)
        with self.narrate(
            "If bytes are the only lever, stop writing intermediates. "
            "Unfused, the residual add and the norm make five trips. "
            "They read two inputs, write the sum, read it back, and write the result. "
            "Fused, the sum stays in registers: four trips. "
            "That predicts a 20 percent saving. "
            "The measurement disagrees, because the sum, 41.9 MB, barely passes the 40 MB L2. "
            "There, the fusion is 1.1 times faster. "
            "When the sum fits in L2, the fused kernel is slower."
        ) as t:
            t.wait_until("Unfused, the residual")
            self.play(FadeIn(un_t), run_time=0.4)
            t.wait_until("They read two inputs")
            self.play(LaggedStart(*[FadeIn(b, shift=RIGHT * 0.1) for b in un], lag_ratio=0.3), run_time=2.0)
            t.wait_until("Fused, the sum")
            self.play(FadeIn(fu_t), LaggedStart(*[FadeIn(b, shift=RIGHT * 0.1) for b in fu], lag_ratio=0.3),
                      run_time=1.6)
            t.wait_until("That predicts")
            self.play(FadeIn(pred), run_time=0.7)
            t.wait_until("The measurement disagrees")
            self.play(FadeIn(m_t), FadeIn(bc.labels), FadeIn(bc.limit), run_time=0.7)
            t.wait_until("There, the fusion")
            self.play(GrowFromEdge(bc.bars[0], LEFT), FadeIn(bc.values[0]), run_time=0.8)
            t.wait_until("When the sum fits")
            self.play(GrowFromEdge(bc.bars[1], LEFT), FadeIn(bc.values[1]), run_time=0.8)

    def what_goes_wrong(self):
        self.slide("what-goes-wrong", "Six bugs the lab is built to catch", book_section="s-what-goes-wrong")
        bugs = kit.bullets(
            [
                r"Reduce over \texttt{dim=0}: you normalize across the batch, silently.",
                r"Drop \texttt{keepdim}: a shape error, or a wrong-axis broadcast.",
                r"Use \texttt{x.var()}: it subtracts the mean and divides by $d - 1$.",
                r"Mix float32 and bf16: PyTorch promotes without a word.",
                r"Forget to cast back: the next matmul leaves the tensor cores.",
                r"Drop \texttt{eps}: the first all-zero row becomes NaN.",
            ],
            size=th.SIZE_SMALL, width=11.5, buff=0.3,
        )
        self.place(bugs)
        with self.narrate(
            "Here's what the lab is likely to provoke. "
            "Reduce over the wrong axis, and you normalize across the batch, silently. "
            "Drop the keep-dimension flag, and the multiply broadcasts against the wrong axis. "
            "PyTorch's variance subtracts the mean, which RMSNorm doesn't want. "
            "Mix float32 and bfloat16, and PyTorch promotes silently. "
            "Forget to cast back, and the next matmul runs off the tensor cores. "
            "Drop epsilon, and a zero row turns into not-a-number."
        ) as t:
            for row, cue in zip(bugs, ["Reduce over", "Drop the keep", "PyTorch's variance", "Mix float32",
                                       "Forget to cast", "Drop epsilon"]):
                t.wait_until(cue)
                self.play(FadeIn(row, shift=RIGHT * 0.1), run_time=0.6)
