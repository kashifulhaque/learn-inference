"""Chapter 18, quantization on Ampere.

Teaches content/chapters/18-quantization.md in its order. Every number below
was recomputed in Python and numpy, and matches the chapter.

Decode floor: 53.8e9 B / 1275e9 B/s = 42.20 ms (42.2); 1 / 0.0422 = 23.7
tokens/s. int8 at g = 128: 53.8 x 1.015625 / 2 = 27.32 GB (27.3); 27.3 / 1275
= 21.41 ms (21.4); 1 / 0.0214 = 46.7 tokens/s; 42.2 / 21.4 = 1.972 (1.97).

Formats: bf16 1 + 8 + 7 bits, 2^7 = 128 steps per octave; e4m3 1 + 4 + 3,
2^3 = 8 steps per octave; int8 -128..127; int4 -8..7. The spacing figure
draws e4m3-style levels (8 per octave from 0.25 to 8, 40 levels) against an
evenly spaced grid with step 0.25 (33 levels) on [0, 8].

Four weights (0.834, -0.301, 0.027, -1.27): s = 1.27 / 127 = 0.01;
w / s = 83.4, -30.1, 2.7, -127; q = 83, -30, 3, -127; s q = 0.83, -0.30,
0.03, -1.27; errors 0.004, 0.001, 0.003, 0, all <= s / 2 = 0.005.

Error bound: |round(x) - x| <= 1/2, so |w_hat - w| <= s / 2. Uniform
residual: variance s^2 / 12, RMS s / sqrt(12). The residual histogram is
4000 uniform draws (kit.rng(181)) in 10 bins.

Prediction: 127 x sqrt(12) = 439.94 (440). sqrt(2 ln 2n): n = 128: 3.330,
eps = 0.757% (0.76%); n = 1024: 3.905, 0.888% (0.89%); n = 524,288: 5.266,
1.197% (1.20%). 1.197 / 0.757 = 1.58 ("a factor of 1.6").

Outlier, four weights with -12.7: s = 12.7 / 127 = 0.1; q = 8, -3, 0, -127;
s q = 0.8, -0.3, 0, -12.7; errors 0.034, 0.001, 0.027, 0.

Outlier of 500 in a (512, 1024) standard normal: s = 500 / 127 = 3.937,
s / 2 = 1.969; P(|w| < 1.969) = 2 Phi(1.969) - 1 = 0.951 (95.1%; 4.9% left).
Simulated with numpy (seed 0): relative error 0.760 per tensor, 0.0133 per
group of 128, ratio 57.1; the chapter's 0.76 / 0.013 = 58.5, "about 58".
In the simulation the survivors all land on q = +-1 (q = +-2 would need
|w| >= 5.9, which a standard normal essentially never reaches).

Granularity: per tensor 524,288 weights share one scale; per channel 1024;
per group 128, so an outlier damages 127 neighbours. Scale shapes (), (512, 1),
(512, 8) for the lab's (512, 1024) weight; 1024 / 128 = 8 groups per row.

Metadata: b(g) = 1 + 2 / g. g = 32: 1.0625 B, 6.25%; g = 64: 1.03125, 3.125%
(3.13%); g = 128: 1.015625, 1.5625% (1.56%), 50.78% of bf16, 2 / 1.015625 =
1.969 (1.97); per channel at 5120 wide: 1.00039, 0.039% (0.04%).
Lab: 512 x 1024 + 512 x 8 x 2 = 524,288 + 8,192 = 532,480 B; / 1,048,576 =
50.78%.

Intensity: bf16 decode 2 FLOPs / 2 B = 1; int8 3 FLOPs / 1 B = 3; ridge 161;
161 / 3 = 53.7 (54). Prefill at 2048 tokens about 2000. The roofline plot
uses log10 axes: the memory roof at 1935 GB/s and the 312 TFLOP/s compute
roof meet at 312 / 1.935 = 161.2.

Accumulator: 127 x 127 = 16,129; x 5120 = 82,580,480 = 8.26e7; 2^31 =
2,147,483,648 = 2.15e9; ratio 26.0. K = 5120 splits into 5120 / 128 = 40
groups (the strip draws all 40).

FP8 e4m3: 3 mantissa bits plus the implicit 1, worst-case relative rounding
error 2^-4 = 6.25%. FP8 needs compute capability (8, 9) or later; A100 is 8.0.

KV cache: 16 x 4 x 256 x 2 x 2 = 65,536 B = 64 KiB per token; int8 data
32 KiB plus 16 x 4 x 2 x 2 = 256 B of scales = 32.25 KiB; 64 / 32.25 =
1.984 (1.98). At 32k: 32,768 x 65,536 B = 2.0 GiB; 32,768 x 33,024 B =
1.0078 GiB (1.008).
Keys: e^0.1 = 1.105 (10.5%), e^0.3 = 1.350 (35%). The bar figure uses
scores (0.4, 1.3, 0.2, 2.0, 0.9, 1.6), and raises the fifth by 0.3.
Values: n_eff = 100 shrinks the RMS by sqrt(100) = 10. The dot figure draws
100 uniform residuals in [-1/2, 1/2] steps (kit.rng(182)); their mean is
computed in the scene, and the RMS of such a mean is 0.289 / 10 = 0.029.

Three precisions: 53.8e9 / 2 = 26.9e9 parameters. int4: 26.9e9 x
(0.5 + 2 / 128) = 13.87e9 B (13.9 GB); 13.87 / 1275 = 10.88 ms (10.9);
42.2 / 10.9 = 3.87. Freed: 53.8 - 27.3 = 26.5 GB; 26.5e9 / 65,536 =
404,358 (about 404,000) tokens; / 32 = 12,636 (roughly 12,600).
int4 step: 127 / 7 = 18.14 (18.1); sqrt(2 ln 256) / 440 x 18.14 = 13.7%.

Measuring: layer output error under 1%; logit correlation above 0.999 and
top-1 agreement above 0.98; perplexity increase under about 1%.

Calibration curve: 100,000 Student-t draws with 2.5 degrees of freedom
(kit.rng(183)), int8 with s = tau / 127, mean squared error split into
clipping and rounding parts, against tau as a fraction of the observed
maximum. Illustrative only; the slide states no value for the optimum.
"""

from manim import *

import kit
from kit import theme as th

S_COL = th.HL_A  # \hla: the scale s
Z_COL = th.HL_B  # \hlb: the zero point z
N_COL = th.HL_C  # \hlc: n, g, n_eff
B_COL = th.HL_D  # \hld: beta, the largest magnitude

LAB_QUANT = """\
scale = weight.abs().amax(dim=-1, keepdim=True) / 127.0
quantized = (weight / scale).round().clamp(-128, 127)
dequantized = quantized.to(torch.bfloat16) * scale
"""

CAPABILITY = """\
major, minor = torch.cuda.get_device_capability()
has_fp8 = (major, minor) >= (8, 9)    # Ada and Hopper onward
"""


class NumLine(VGroup):
    """A horizontal number line from `lo` to `hi`; `p(v)` is the point for value `v`."""

    def __init__(self, lo, hi, width, ticks=(), majors=(), labels=None, tick_h=0.12, major_h=0.26,
                 color=th.INK_3, label_size=th.SIZE_LABEL, label_color=th.INK_3, tick_stroke=th.STROKE_HAIRLINE):
        super().__init__()
        self.lo, self.hi = lo, hi
        self.line = Line(LEFT * width / 2, RIGHT * width / 2, color=color, stroke_width=th.STROKE_AXIS)
        self.add(self.line)
        self.ticks = VGroup()
        for v in ticks:
            x = self._x(v, width)
            self.ticks.add(Line([x, -tick_h / 2, 0], [x, tick_h / 2, 0], color=color, stroke_width=tick_stroke))
        for v in majors:
            x = self._x(v, width)
            self.ticks.add(Line([x, -major_h / 2, 0], [x, major_h / 2, 0], color=th.INK_2,
                                stroke_width=th.STROKE_AXIS))
        self.add(self.ticks)
        self.labels = VGroup()
        for v, text in (labels or {}).items():
            lab = kit.math_label(text, size=label_size, color=label_color)
            lab.next_to([self._x(v, width), -major_h / 2, 0], DOWN, buff=0.12)
            self.labels.add(lab)
        self.add(self.labels)

    def _x(self, v, width):
        return -width / 2 + (v - self.lo) / (self.hi - self.lo) * width

    def p(self, v):
        a, b = self.line.get_start(), self.line.get_end()
        return a + (v - self.lo) / (self.hi - self.lo) * (b - a)


def swatch(text, color, size=th.SIZE_LABEL):
    sq = Square(0.26, stroke_color=th.RULE_STRONG, stroke_width=th.STROKE_HAIRLINE,
                fill_color=th.tint(color, 0.35), fill_opacity=1)
    return VGroup(sq, kit.label(text, size=size, color=th.INK_2)).arrange(RIGHT, buff=0.14)


def line_key(text, color, size=th.SIZE_LABEL):
    rule = Line(LEFT * 0.25, RIGHT * 0.25, color=color, stroke_width=th.STROKE_CURVE)
    return VGroup(rule, kit.label(text, size=size, color=th.INK_2)).arrange(RIGHT, buff=0.16)


def polyline(ax, xs, ys, color, ymax=None, stroke=th.STROKE_CURVE):
    pts = [ax.c2p(x, y) for x, y in zip(xs, ys) if ymax is None or y <= ymax]
    curve = VMobject(stroke_color=color, stroke_width=stroke)
    curve.set_points_as_corners(pts)
    return curve


class Lecture(kit.Lecture):
    def construct(self):
        self.title_slide(
            narration=(
                "Quantization on Ampere. "
                "In this lecture, you derive the error of rounding to a shared scale, see why one outlier wrecks it, "
                "and count what int8 buys you on an A100."
            )
        )
        self.decode_floor()
        self.bits()
        self.four_weights()
        self.affine()
        self.symmetric()
        self.error_bound()
        self.bound_predicts()
        self.outlier_four()
        self.outlier_500()
        self.granularity()
        self.metadata()
        self.decode_halves()
        self.roofline()
        self.int8_cores()
        self.no_fp8()
        self.activation_outliers()
        self.calibration()
        self.kv_cache()
        self.three_precisions()
        self.measuring()
        self.closing_slide(
            [
                r"Symmetric int8: $s = \beta/127$, error $\le s/2$.",
                r"Groups of 128 contain outliers for 1.56\%.",
                r"Decode: 42.2 to 21.4 ms; prefill flat.",
                r"No FP8 on Ampere; keys need more bits.",
            ],
            narration=(
                "Here's what to keep. "
                "Symmetric int8's error is at most half a step. "
                "One outlier stretches every step it shares, so give each group of 128 weights its own scale. "
                "That costs 1.56 percent. "
                "Weight-only int8 halves the decode floor, and leaves prefill flat. "
                "Ampere has no FP8, and keys need more precision than values. "
                "Next, you split the model across GPUs."
            ),
            book_section="s-what-goes-wrong",
        )

    # ------------------------------------------------------------------
    # Before you start

    def decode_floor(self):
        self.slide("decode-floor", "Decode's floor is the weight bytes", book_section="s-before-you-start")
        eq = kit.formula(r"t_{\text{decode}} \approx", r"\frac{53.8\ \text{GB}}{1275\ \text{GB/s}}",
                         r"= 42.2\ \text{ms}")
        rate = kit.label("23.7 tokens per second, before a single FLOP", size=th.SIZE_SMALL, color=th.INK_3)
        bc = kit.bar_compare(
            [("bf16 weights", 53.8, th.INFO, "53.8 GB  ·  42.2 ms"),
             ("int8 weights", 27.3, self.color, "27.3 GB  ·  21.4 ms")],
            width=7.0, bar_height=0.6, gap=0.45,
        )
        half = kit.pill("half the bytes, half the floor", self.color, size=th.SIZE_SMALL)
        col = VGroup(VGroup(eq, rate).arrange(DOWN, buff=0.3), bc, half).arrange(DOWN, buff=0.65)
        self.place(col)
        with self.narrate(
            "Every decode step reads all 53.8 GB of the model's weights. "
            "At the A100's measured copy bandwidth of 1275 GB/s, that takes 42.2 ms. "
            "So the floor is 23.7 tokens per second, before a single FLOP. "
            "Halve the bytes, and you halve the floor. "
            "That makes quantization the most direct speedup this engine has. "
            "It's also the easiest way to wreck the model without noticing."
        ) as t:
            self.play(FadeIn(eq[0]), FadeIn(bc.labels[0]), GrowFromEdge(bc.bars[0], LEFT), FadeIn(bc.values[0]),
                      run_time=1.0)
            t.wait_until("At the A100's")
            self.play(FadeIn(eq[1:]), run_time=0.8)
            t.wait_until("So the floor")
            self.play(FadeIn(rate), run_time=0.6)
            t.wait_until("Halve the bytes")
            self.play(FadeIn(bc.labels[1]), GrowFromEdge(bc.bars[1], LEFT), run_time=1.0)
            self.play(FadeIn(bc.values[1]), FadeIn(half, scale=0.9), run_time=0.6)

    # ------------------------------------------------------------------
    # What the bits hold

    def bits(self):
        self.slide("bits", "Floats space levels by octave; integers evenly", book_section="s-what-the-bits-hold")
        roles = {"s": th.VIOLET, "e": th.INFO, "m": self.color, "i": th.GOOD}
        specs = [
            ("bf16", "s" + "e" * 8 + "m" * 7, "128 steps per octave"),
            ("FP8 E4M3", "s" + "e" * 4 + "m" * 3, "8 steps per octave"),
            ("int8", "i" * 8, "−128 to 127, step 1"),
            ("int4", "i" * 4, "−8 to 7, step 1"),
        ]
        cw, chh = 0.3, 0.4
        rows = VGroup()
        names, cellrows, descs = VGroup(), VGroup(), VGroup()
        for name, kinds, desc in specs:
            names.add(kit.label(name, size=th.SIZE_SMALL, color=th.INK, font=th.MONO))
            cells = VGroup(*[Rectangle(width=cw, height=chh, stroke_color=th.RULE_STRONG,
                                       stroke_width=th.STROKE_HAIRLINE, fill_color=th.tint(roles[k], 0.4),
                                       fill_opacity=1) for k in kinds]).arrange(RIGHT, buff=0)
            cellrows.add(cells)
            descs.add(kit.label(desc, size=th.SIZE_SMALL, color=th.INK_2))
        name_w = max(n.width for n in names)
        cells_w = 16 * cw
        for i in range(4):
            y = -i * (chh + 0.2)
            names[i].move_to([name_w / 2, y, 0]).align_to([0, 0, 0], LEFT)
            cellrows[i].move_to([0, y, 0]).align_to([name_w + 0.4, 0, 0], LEFT)
            descs[i].move_to([0, y, 0]).align_to([name_w + 0.4 + cells_w + 0.5, 0, 0], LEFT)
            rows.add(VGroup(names[i], cellrows[i], descs[i]))
        key = VGroup(swatch("sign", roles["s"]), swatch("exponent", roles["e"]), swatch("mantissa", roles["m"]),
                     swatch("integer", roles["i"])).arrange(RIGHT, buff=0.5)

        fl = []
        for e in range(-2, 3):
            base = 2.0 ** e
            fl += [base * (1 + k / 8) for k in range(8)]
        fl.append(8.0)
        width = 10.0
        fline = NumLine(0, 8, width, ticks=fl, tick_h=0.3, tick_stroke=2.0, labels={0: "0", 1: "1", 2: "2", 4: "4", 8: "8"})
        iline = NumLine(0, 8, width, ticks=[0.25 * k for k in range(33)], tick_h=0.3, tick_stroke=2.0,
                        labels={0: "0", 1: "1", 2: "2", 4: "4", 8: "8"})
        for ln, c in ((fline, self.color), (iline, th.GOOD)):
            ln.ticks.set_color(c)
        fname = kit.label("FP8: same 8 levels in every octave", size=th.SIZE_LABEL, color=th.INK_2)
        iname = kit.label("int8 with a scale: one fixed step", size=th.SIZE_LABEL, color=th.INK_2)
        fblock = VGroup(fname, fline).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        iblock = VGroup(iname, iline).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        lines = VGroup(fblock, iblock).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        col = VGroup(rows, key, lines).arrange(DOWN, buff=0.45)
        self.place(col)
        with self.narrate(
            "A floating-point number splits its bits into a sign, an exponent, and a mantissa. "
            "In bfloat16, 8 bits are exponent and 7 are mantissa. "
            "FP8 E4M3 has 4 and 3. "
            "An integer format has no exponent: int8 holds minus 128 to 127. "
            "A float puts the same number of levels in every octave, so the gap grows with the value. "
            "An integer spaces its levels evenly. "
            "So a float has constant relative precision, and an integer has constant absolute precision."
        ) as t:
            self.play(FadeIn(key), run_time=0.6)
            t.wait_until("In bfloat16")
            self.play(FadeIn(rows[0]), run_time=0.7)
            t.wait_until("FP8 E4M3")
            self.play(FadeIn(rows[1]), run_time=0.7)
            t.wait_until("An integer format")
            self.play(FadeIn(rows[2]), FadeIn(rows[3]), run_time=0.8)
            t.wait_until("A float puts")
            self.play(FadeIn(fname), Create(fline.line), FadeIn(fline.labels), run_time=0.6)
            self.play(LaggedStart(*[GrowFromCenter(tk) for tk in fline.ticks], lag_ratio=0.04), run_time=1.6)
            t.wait_until("An integer spaces")
            self.play(FadeIn(iname), Create(iline.line), FadeIn(iline.labels), run_time=0.6)
            self.play(LaggedStart(*[GrowFromCenter(tk) for tk in iline.ticks], lag_ratio=0.04), run_time=1.4)

    def four_weights(self):
        self.slide("four-weights", "A shared scale turns integers into weights", book_section="s-what-the-bits-hold")
        eq = kit.formula(r"\hat w = ", r"s", r"\,q,", r"\qquad", r"s", r"= \frac{1.27}{127} = 0.01")
        eq[1].set_color(S_COL)
        eq[4].set_color(S_COL)
        tab = kit.table(
            [
                ["0.834", "83.4", "83", "0.83", "0.004"],
                ["$-0.301$", "$-30.1$", "$-30$", "$-0.30$", "0.001"],
                ["0.027", "2.7", "3", "0.03", "0.003"],
                ["$-1.27$", "$-127$", "$-127$", "$-1.27$", "0"],
            ],
            header=["weight $w$", "$w/s$", "stored $q$", "$s\\,q$", "error"],
            align="r", col_gap=0.55, row_gap=0.3,
        )
        left = VGroup(eq, tab).arrange(DOWN, buff=0.6)
        # Zoom: the step between 0.83 and 0.84.
        zl = NumLine(0.82, 0.85, 4.4, ticks=[0.825, 0.835, 0.845], majors=[0.82, 0.83, 0.84, 0.85],
                     labels={0.82: "0.82", 0.83: "0.83", 0.84: "0.84", 0.85: "0.85"})
        zt = kit.label("Up close: one step is 0.01", size=th.SIZE_SMALL, color=th.INK_2)
        right = VGroup(zt, zl).arrange(DOWN, buff=1.4)
        layout = VGroup(left, right).arrange(RIGHT, buff=0.8)
        self.place(layout)
        dot = Dot(zl.p(0.834) + UP * 0.45, radius=0.09, color=self.color)
        dlab = kit.label("0.834", size=th.SIZE_LABEL, color=self.color).next_to(dot, UP, buff=0.1)
        ghost = Dot(zl.p(0.834) + UP * 0.45, radius=0.09, color=th.INK_4, fill_opacity=0.5)
        err = Line(zl.p(0.83) + UP * 0.45, zl.p(0.834) + UP * 0.45, color=th.BAD, stroke_width=5)
        half = BraceBetweenPoints(zl.p(0.83) + DOWN * 0.7, zl.p(0.835) + DOWN * 0.7, DOWN, color=th.INK_4)
        half_t = kit.prose(r"$s/2 = 0.005$", size=th.SIZE_LABEL, color=th.INK_3, width=None).next_to(half, DOWN,
                                                                                                  buff=0.1)
        err_t = kit.label("error 0.004", size=th.SIZE_LABEL, color=th.BAD)
        cols = [[tab.cells[r][c] for r in range(5)] for c in range(5)]
        with self.narrate(
            "An int8 alone holds only an integer, so you pair it with a scale: how much one integer step is worth. "
            "Take these four weights. "
            "The largest magnitude is 1.27, so the scale is 1.27 / 127, which is 0.01. "
            "Divide each weight by the scale, and round. "
            "Then multiply back. "
            "Up close, 0.834 snaps to the nearer level. "
            "Every error is at most 0.005, half a step. "
            "And since many weights share one scale, storage is about one byte per weight."
        ) as t:
            t.wait_until("Take these four")
            self.play(FadeIn(tab.rules), *[FadeIn(m) for m in cols[0]], run_time=0.8)
            t.wait_until("The largest magnitude")
            self.play(FadeIn(eq), run_time=0.8)
            t.wait_until("Divide each weight")
            self.play(*[FadeIn(m, shift=RIGHT * 0.1) for m in cols[1]], run_time=0.7)
            self.play(*[FadeIn(m, shift=RIGHT * 0.1) for m in cols[2]], run_time=0.7)
            t.wait_until("Then multiply back")
            self.play(*[FadeIn(m, shift=RIGHT * 0.1) for m in cols[3]], run_time=0.7)
            t.wait_until("Up close")
            self.play(FadeIn(zt), FadeIn(zl), FadeIn(dot), FadeIn(dlab), run_time=0.8)
            self.add(ghost)
            self.play(dot.animate.move_to(zl.p(0.83) + UP * 0.45), run_time=0.9)
            err_t.next_to(ghost, RIGHT, buff=0.2)
            self.play(Create(err), FadeIn(err_t), run_time=0.6)
            t.wait_until("Every error is")
            self.play(*[FadeIn(m, shift=RIGHT * 0.1) for m in cols[4]], FadeIn(half), FadeIn(half_t), run_time=0.8)

    # ------------------------------------------------------------------
    # Affine and symmetric

    def affine(self):
        self.slide("affine", "Affine quantization maps a range onto integers",
                   book_section="s-affine-quantization-derived")
        lo, hi = -0.6, 1.4
        real = NumLine(lo, hi, 5.8, majors=[lo, 0, hi], label_size=th.SIZE_SMALL,
                       labels={lo: r"\beta_{\min}", 0: "0", hi: r"\beta_{\max}"})
        ints = NumLine(-128, 127, 5.8, majors=[-128, -52, 127], label_size=th.SIZE_SMALL,
                       labels={-128: r"-128", -52: "z", 127: r"127"})
        ints.labels[1].set_color(Z_COL)
        rn = kit.label("real values w", size=th.SIZE_SMALL, color=th.INK_3)
        inn = kit.label("int8 levels q", size=th.SIZE_SMALL, color=th.INK_3)
        fig = VGroup(VGroup(rn, real).arrange(DOWN, buff=0.25),
                     VGroup(inn, ints).arrange(DOWN, buff=0.25)).arrange(DOWN, buff=1.6)
        rn.shift(RIGHT * 0.9)
        inn.shift(RIGHT * 0.9)
        deq = kit.formula(r"\hat w =", r"s", r"\,(q - ", r"z", r")")
        deq[1].set_color(S_COL)
        deq[3].set_color(Z_COL)
        sz = kit.formula(r"s", r"= \frac{\beta_{\max} - \beta_{\min}}{q_{\max} - q_{\min}}", size=th.SIZE_MATH_SMALL)
        sz[0].set_color(S_COL)
        zz = kit.formula(r"z", r"= \operatorname{round}\!\left(q_{\min} - \frac{\beta_{\min}}{s}\right)",
                         size=th.SIZE_MATH_SMALL)
        zz[0].set_color(Z_COL)
        boxed = VGroup(sz, zz).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        frame = SurroundingRectangle(boxed, color=th.INK_3, buff=0.2, stroke_width=th.STROKE_AXIS)
        quant = kit.formula(r"q = \operatorname{clamp}\!\left(\operatorname{round}\!\left(\frac{w}{s}\right) + z,"
                            r"\ q_{\min},\ q_{\max}\right)", size=th.SIZE_MATH_SMALL)
        warn = kit.callout("warning", r"Without the clamp, an out-of-range value wraps around in int8.",
                           title="Clamp, always", width=6.4, size=th.SIZE_SMALL)
        right = VGroup(deq, VGroup(frame, boxed), quant, warn).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        layout = VGroup(fig, right).arrange(RIGHT, buff=0.7)
        self.place(layout)
        maps = VGroup(*[DashedLine(real.p(a) + DOWN * 0.75, ints.p(b) + UP * 0.2, color=c,
                                   stroke_width=th.STROKE_AXIS, dash_length=0.08)
                        for a, b, c in ((lo, -128, th.INK_3), (hi, 127, th.INK_3), (0, -52, Z_COL))])
        with self.narrate(
            "Now make the recipe general. "
            "An affine map sends a range of reals onto a range of integers. "
            "Dequantization is s times, q minus z. "
            "The zero point z is the integer that means exactly zero. "
            "Pin the two ends of the real range to the two ends of the integer range. "
            "Solving gives the scale and the zero point. "
            "Quantizing inverts the map: divide, round, add z, and clamp. "
            "Skip the clamp, and an out-of-range value silently wraps around."
        ) as t:
            self.play(FadeIn(fig[0]), FadeIn(fig[1]), run_time=0.8)
            t.wait_until("Dequantization is")
            self.play(FadeIn(deq), run_time=0.7)
            t.wait_until("The zero point")
            self.play(Create(maps[2]), Indicate(ints.labels[1], color=Z_COL), run_time=0.9)
            t.wait_until("Pin the two ends")
            self.play(Create(maps[0]), Create(maps[1]), run_time=1.0)
            t.wait_until("Solving gives")
            self.play(FadeIn(sz), run_time=0.7)
            self.play(FadeIn(zz), Create(frame), run_time=0.8)
            t.wait_until("Quantizing inverts")
            self.play(FadeIn(quant), run_time=0.8)
            t.wait_until("Skip the clamp")
            self.play(FadeIn(warn, shift=UP * 0.1), run_time=0.7)

    def symmetric(self):
        self.slide("symmetric", "Symmetric quantization: no zero point, one multiply",
                   book_section="s-symmetric-quantization")
        f = kit.formula(r"s", r"= \frac{", r"\beta", r"}{127}", r"\qquad", r"\hat w = ", r"s", r"\,q",
                        r"\qquad", r"\beta", r"= \max |w|")
        for i in (0, 6):
            f[i].set_color(S_COL)
        for i in (2, 9):
            f[i].set_color(B_COL)
        real = NumLine(-1, 1, 8.0, majors=[-1, 0, 1], labels={-1: r"-\beta", 0: "0", 1: r"\beta"},
                       label_size=th.SIZE_SMALL)
        ints = NumLine(-128, 127, 8.0 * 255 / 254, majors=[-127, 0, 127], labels={-127: "-127", 0: "0", 127: "127"},
                       label_size=th.SIZE_SMALL)
        ints.shift(LEFT * (ints.p(0)[0] - real.p(0)[0]))
        fig = VGroup(real, ints).arrange(DOWN, buff=1.1)
        ints.shift(RIGHT * (real.p(0)[0] - ints.p(0)[0]))
        panel = kit.code(LAB_QUANT, "python", caption="one scale per output row")
        note = kit.pill("round before you clamp", self.color, size=th.SIZE_SMALL)
        bottom = VGroup(panel, note).arrange(RIGHT, buff=0.5)
        col = VGroup(f, fig, bottom).arrange(DOWN, buff=0.55)
        self.place(col)
        maps = VGroup(*[DashedLine(real.p(a) + DOWN * 0.7, ints.p(b) + UP * 0.2, color=th.INK_3,
                                   stroke_width=th.STROKE_AXIS, dash_length=0.08)
                        for a, b in ((-1, -127), (0, 0), (1, 127))])
        maps[1].set_color(Z_COL)
        lost = Dot(ints.p(-128), radius=0.08, color=th.BAD)
        lost_t = kit.label("−128 unused", size=th.SIZE_LABEL, color=th.BAD)
        lost_t.next_to(lost, LEFT, buff=0.25)
        with self.narrate(
            "Weights sit close to symmetric around zero, so they drop the zero point. "
            "Set the range to plus or minus beta, the largest magnitude. "
            "Then z is zero, and dequantization is one multiply. "
            "You use minus 127 to 127, because minus 128 has no positive partner. "
            "The lab's quantizer is three lines, with one scale per output row. "
            "Round before you clamp, or you lose a level at each end."
        ) as t:
            self.play(FadeIn(real), FadeIn(ints), run_time=0.8)
            t.wait_until("Set the range")
            self.play(FadeIn(f[:4]), FadeIn(f[8:]), Create(maps[0]), Create(maps[2]), run_time=1.0)
            t.wait_until("Then z is zero")
            self.play(Create(maps[1]), FadeIn(f[4:8]), run_time=0.9)
            t.wait_until("You use minus")
            self.play(FadeIn(lost, scale=0.5), FadeIn(lost_t), run_time=0.7)
            t.wait_until("The lab's quantizer")
            self.play(FadeIn(panel.frame), *panel.reveal(1), run_time=0.7)
            self.play(*panel.reveal(2), run_time=0.6)
            self.play(*panel.reveal(3), run_time=0.6)
            t.wait_until("Round before")
            self.play(panel.highlight(2), FadeIn(note, scale=0.9), run_time=0.7)

    # ------------------------------------------------------------------
    # The error bound and what it predicts

    def error_bound(self):
        self.slide("error-bound", "The error is at most half a step", book_section="s-the-error-bound")
        ax = kit.axes([0, 4, 1], [-0.5, 0.5, 0.25], width=6.8, height=3.8,
                      x_label=r"w / s", y_label=r"(\hat w - w)/s",
                      x_numbers=[0, 1, 2, 3, 4], y_numbers=[-0.5, 0, 0.5], decimals=0)
        # Redo the y numbers with one decimal.
        for lab in ax.number_labels[5:]:
            ax.number_labels.remove(lab)
            ax.remove(lab)
        for v in (-0.5, 0.5):
            n = kit.math_label(f"{v:+.1f}".replace("+", "+"), size=th.SIZE_LABEL - 2, color=th.INK_3)
            n.next_to(ax.c2p(0, v), LEFT, buff=0.14)
            ax.add(n)
        saw = VGroup()
        for k in range(5):
            a, b = max(0.0, k - 0.5), min(4.0, k + 0.5)
            saw.add(Line(ax.c2p(a, k - a), ax.c2p(b, k - b), color=self.color, stroke_width=th.STROKE_CURVE))
        bands = VGroup(*[DashedLine(ax.c2p(0, v), ax.c2p(4, v), color=th.INK_4, stroke_width=th.STROKE_AXIS,
                                    dash_length=0.08) for v in (-0.5, 0.5)])
        plot = VGroup(ax, saw, bands)
        bound = kit.formula(r"|\hat w - w| \le \frac{", r"s", r"}{2}")
        bound[1].set_color(S_COL)
        bbox = SurroundingRectangle(bound, color=self.color, buff=0.18, stroke_width=3)
        rms = kit.formula(r"\sigma_{\text{err}} = \frac{", r"s", r"}{\sqrt{12}}", size=th.SIZE_MATH_SMALL)
        rms[1].set_color(S_COL)
        r = kit.rng(181)
        x = r.uniform(0, 4, 4000)
        res = np.round(x) - x
        counts, edges = np.histogram(res, bins=10, range=(-0.5, 0.5))
        hax = kit.axes([-0.5, 0.5, 0.5], [0, 600, 200], width=3.4, height=1.4, show_y_axis=False)
        hist = kit.bars(hax, edges, counts, color=self.color, fill_opacity=0.35)
        hcap = kit.label("residuals: flat across the step", size=th.SIZE_LABEL, color=th.INK_3)
        hgroup = VGroup(VGroup(hax, hist), hcap).arrange(DOWN, buff=0.15)
        side = VGroup(VGroup(bbox, bound), hgroup, rms).arrange(DOWN, buff=0.5)
        layout = VGroup(plot, side).arrange(RIGHT, buff=0.8)
        self.place(layout)
        with self.narrate(
            "How wrong can a quantized weight be? "
            "The error is the rounding residual, scaled up by s. "
            "Rounding is never off by more than one half. "
            "So the error is at most half a step, whatever the weight's size. "
            "That's what makes outliers so damaging. "
            "For the typical error, treat the residual as uniform. "
            "Its root-mean-square is s over the square root of 12. "
            "Every later choice is an attempt to make s smaller."
        ) as t:
            self.play(Create(ax), run_time=0.9)
            t.wait_until("The error is the rounding")
            self.play(LaggedStart(*[Create(seg) for seg in saw], lag_ratio=0.5), run_time=2.0)
            t.wait_until("Rounding is never")
            self.play(Create(bands), run_time=0.7)
            t.wait_until("So the error is at most")
            self.play(FadeIn(bound), Create(bbox), run_time=0.8)
            t.wait_until("For the typical")
            self.play(FadeIn(hax), LaggedStart(*[GrowFromEdge(b, DOWN) for b in hist], lag_ratio=0.1),
                      FadeIn(hcap), run_time=1.2)
            t.wait_until("Its root-mean-square")
            self.play(FadeIn(rms), run_time=0.7)

    def bound_predicts(self):
        self.slide("bound-predicts", "More weights per scale, a coarser step", book_section="s-what-the-bound-predicts")
        f = kit.formula(r"\varepsilon(", r"n", r") \approx \frac{\sqrt{2\ln(2", r"n", r")}}{127\sqrt{12}}",
                        r"= \frac{\sqrt{2\ln(2", r"n", r")}}{440}")
        for i in (1, 3, 6):
            f[i].set_color(N_COL)
        bc = kit.bar_compare(
            [("Per group, n = 128", 0.757, self.color, "0.76%"),
             ("Per channel, n = 1024", 0.888, th.INFO, "0.89%"),
             ("Per tensor, n = 524,288", 1.197, th.INFO, "1.20%")],
            width=6.4, bar_height=0.55, gap=0.4, max_value=1.3, limit=1.0, limit_label="lab threshold: 1%",
        )
        cap = kit.label("Predicted relative error, normal weights", size=th.SIZE_SMALL, color=th.INK_3)
        gap = kit.pill("only a factor of 1.6", th.INFO, size=th.SIZE_SMALL)
        col = VGroup(f, VGroup(cap, bc).arrange(DOWN, buff=0.35), gap).arrange(DOWN, buff=0.55)
        self.place(col)
        with self.narrate(
            "The bound also predicts the error before you run anything. "
            "For n normal weights, the largest magnitude grows like the square root of the log of n. "
            "That gives this formula. "
            "Per group of 128, it predicts 0.76 percent. "
            "Per output row, 0.89 percent. "
            "Per tensor, 1.2 percent. "
            "The group figure is under the lab's 1 percent threshold. "
            "The whole spread is only a factor of 1.6. "
            "If weights were Gaussian, per-tensor would be almost fine. "
            "They aren't."
        ) as t:
            t.wait_until("That gives this")
            self.play(FadeIn(f), FadeIn(cap), FadeIn(bc.labels), run_time=0.9)
            for i, cue in enumerate(["Per group of 128", "Per output row", "Per tensor, 1.2"]):
                t.wait_until(cue)
                self.play(GrowFromEdge(bc.bars[i], LEFT), FadeIn(bc.values[i]), run_time=0.8)
            t.wait_until("The group figure")
            self.play(Create(bc.limit), run_time=0.7)
            t.wait_until("The whole spread")
            self.play(FadeIn(gap, scale=0.9), run_time=0.6)

    # ------------------------------------------------------------------
    # Why per-tensor scaling collapses

    def outlier_four(self):
        self.slide("outlier-four", "One outlier stretches the step for everyone",
                   book_section="s-why-per-tensor-scaling-collapses")
        lo, hi, width = -0.4, 0.9, 11.0
        fine = NumLine(lo, hi, width, ticks=[round(lo + 0.01 * k, 2) for k in range(131)], tick_h=0.14,
                       majors=[round(lo + 0.1 * k, 1) for k in range(14)])
        coarse = NumLine(lo, hi, width, majors=[round(lo + 0.1 * k, 1) for k in range(14)],
                         labels={v: f"{v:g}" for v in (-0.4, -0.3, 0, 0.3, 0.6, 0.8, 0.9)})
        t1 = kit.prose(r"Largest weight $-1.27$:\enspace $s = 0.01$", size=th.SIZE_SMALL, width=None)
        t2 = kit.prose(r"Largest weight $-12.7$:\enspace $s = 0.1$", size=th.SIZE_SMALL, width=None)
        t2b = kit.label("the outlier itself stays exact", size=th.SIZE_LABEL, color=th.INK_3)
        top = VGroup(t1, fine).arrange(DOWN, buff=0.75, aligned_edge=LEFT)
        bot = VGroup(VGroup(t2, t2b).arrange(RIGHT, buff=0.5), coarse).arrange(DOWN, buff=0.75, aligned_edge=LEFT)
        col = VGroup(top, bot).arrange(DOWN, buff=0.9, aligned_edge=LEFT)
        self.place(col)
        ws = [0.834, -0.301, 0.027]
        cols = [self.color, th.INFO, th.VIOLET]

        def dots(line, snapped):
            g = VGroup()
            for w, s, c in zip(ws, snapped, cols):
                g.add(Dot(line.p(w) + UP * 0.35, radius=0.085, color=c))
            return g

        d1 = dots(fine, [0.83, -0.30, 0.03])
        d2 = dots(coarse, [0.8, -0.3, 0.0])
        e1 = VGroup(*[kit.label(s, size=th.SIZE_LABEL, color=th.INK_3) for s in ("0.004", "0.001", "0.003")])
        e2 = VGroup(*[kit.label(s, size=th.SIZE_LABEL, color=c) for s, c in
                      (("0.034", th.BAD), ("0.001", th.INK_3), ("0.027: gone", th.BAD))])
        for d, e in zip(d1, e1):
            e.next_to(d, UP, buff=0.12)
        with self.narrate(
            "Real weight matrices hold a few entries far outside the bulk. "
            "Go back to the four weights, where the scale was 0.01. "
            "Now make the last one ten times larger, minus 12.7. "
            "The scale becomes 0.1. "
            "The outlier itself is exact. "
            "But 0.834 now rounds to 0.8, an error of 0.034. "
            "And 0.027 rounds to zero. "
            "It's gone. "
            "A single outlier ruins every weight that shares its scale."
        ) as t:
            t.wait_until("Go back to")
            self.play(FadeIn(t1), FadeIn(fine), run_time=0.8)
            self.play(FadeIn(d1, scale=0.6), FadeIn(e1), run_time=0.7)
            t.wait_until("Now make the last")
            self.play(FadeIn(t2), run_time=0.6)
            t.wait_until("The scale becomes")
            self.play(FadeIn(coarse), FadeIn(d2, scale=0.6), run_time=0.8)
            t.wait_until("The outlier itself")
            self.play(FadeIn(t2b), run_time=0.5)
            t.wait_until("But 0.834")
            self.play(d2[0].animate.move_to(coarse.p(0.8) + UP * 0.35), d2[1].animate.move_to(
                coarse.p(-0.3) + UP * 0.35), run_time=0.9)
            e2[0].next_to(d2[0], UP, buff=0.12)
            e2[1].next_to(d2[1], UP, buff=0.12)
            self.play(FadeIn(e2[0]), FadeIn(e2[1]), run_time=0.5)
            t.wait_until("And 0.027")
            self.play(d2[2].animate.move_to(coarse.p(0.0) + UP * 0.35), run_time=0.8)
            e2[2].next_to(d2[2], UP, buff=0.12)
            t.wait_until("It's gone")
            self.play(FadeIn(e2[2]), Indicate(d2[2], color=th.BAD, scale_factor=1.6), run_time=0.8)

    def outlier_500(self):
        self.slide("outlier-500", "An outlier of 500 zeroes 95% of the matrix",
                   book_section="s-why-per-tensor-scaling-collapses")
        s = 500 / 127
        ax = kit.axes([-5, 5, 1], [0, 0.55, 0.1], width=7.0, height=3.9, show_y_axis=False,
                      x_numbers=[-4, -2, 0, 2, 4], x_label=r"w")
        pdf = lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi)  # noqa: E731
        dens = kit.density(ax, pdf, x_range=[-4.6, 4.6], color=th.INK_3, fill_opacity=0.0)
        zero = kit.tail(ax, dens.curve, -s / 2, s / 2, color=th.INK_4, opacity=0.45)
        tails = VGroup(kit.tail(ax, dens.curve, -4.6, -s / 2, color=self.color, opacity=0.55),
                       kit.tail(ax, dens.curve, s / 2, 4.6, color=self.color, opacity=0.55))
        cuts = VGroup(*[DashedLine(ax.c2p(v, 0), ax.c2p(v, 0.43), color=th.INK_3, stroke_width=th.STROKE_AXIS,
                                   dash_length=0.08) for v in (-s / 2, s / 2)])
        cut_t = kit.label("±1.97: half a step", size=th.SIZE_LABEL, color=th.INK_3)
        cut_t.next_to(cuts[1].get_top(), RIGHT, buff=0.12)
        zero_t = kit.label("95.1% round to 0", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
        zero_t.move_to(ax.c2p(0, 0.5))
        lev = VGroup(*[Dot(ax.c2p(v, 0), radius=0.08, color=self.color) for v in (-s, s)])
        lev_t = kit.label("q = ±1", size=th.SIZE_LABEL, color=self.color).next_to(ax.c2p(s, 0.06), UP, buff=0.05)
        plot = VGroup(ax, dens, zero, tails, cuts, cut_t, zero_t, lev, lev_t)
        f = kit.formula(r"s", r"= \frac{500}{127} = 3.937", size=th.SIZE_MATH_SMALL)
        f[0].set_color(S_COL)
        f2 = kit.formula(r"\frac{s}{2} = 1.969", size=th.SIZE_MATH_SMALL)
        bc = kit.bar_compare([("Per tensor", 0.76, th.BAD, "0.76"), ("Per group of 128", 0.013, self.color, "0.013")],
                             width=2.6, bar_height=0.45, gap=0.3, size=th.SIZE_LABEL)
        bcap = kit.label("relative error", size=th.SIZE_LABEL, color=th.INK_3)
        ratio = kit.pill("about 58×", th.BAD, size=th.SIZE_SMALL)
        side = VGroup(f, f2, VGroup(bcap, bc).arrange(DOWN, buff=0.2, aligned_edge=LEFT), ratio).arrange(
            DOWN, buff=0.4, aligned_edge=LEFT)
        layout = VGroup(plot, side).arrange(RIGHT, buff=0.6)
        self.place(layout)
        with self.narrate(
            "The lab's outlier case is a 512 by 1024 matrix of standard normals, with one entry set to 500. "
            "One scale for the whole tensor makes the step 500 / 127, about 3.94. "
            "Anything smaller than half a step rounds to zero. "
            "That's 95 percent of the matrix. "
            "The rest collapse onto a handful of levels. "
            "The relative error comes out near 0.76. "
            "Per group of 128, it stays near 0.013. "
            "That's a ratio of about 58, caused by a weight that is itself stored exactly."
        ) as t:
            self.play(Create(ax), Create(dens.curve), run_time=1.0)
            t.wait_until("One scale for")
            self.play(FadeIn(f), run_time=0.7)
            t.wait_until("Anything smaller")
            self.play(FadeIn(f2), Create(cuts), FadeIn(cut_t), run_time=0.8)
            t.wait_until("That's 95 percent")
            self.play(FadeIn(zero), FadeIn(zero_t), run_time=0.9)
            t.wait_until("The rest collapse")
            self.play(FadeIn(tails), FadeIn(lev, scale=0.5), FadeIn(lev_t), run_time=0.8)
            t.wait_until("The relative error")
            self.play(FadeIn(bcap), FadeIn(bc.labels[0]), GrowFromEdge(bc.bars[0], LEFT), FadeIn(bc.values[0]),
                      run_time=0.8)
            t.wait_until("Per group of 128")
            self.play(FadeIn(bc.labels[1]), GrowFromEdge(bc.bars[1], LEFT), FadeIn(bc.values[1]), run_time=0.7)
            t.wait_until("That's a ratio")
            self.play(FadeIn(ratio, scale=0.9), run_time=0.6)

    # ------------------------------------------------------------------
    # Per-channel and group-wise scaling

    def granularity(self):
        self.slide("granularity", "Shrink the set of weights that share a scale",
                   book_section="s-per-channel-and-group-wise-scaling")
        R, C, cell = 6, 16, 0.22
        orow, ocol = 2, 9
        titles = ["Per tensor", "Per channel", "Per group of 128"]
        shapes = ["scale shape ()", "scale shape (512, 1)", "scale shape (512, 8)"]
        hits = ["damages 524,288", "damages 1024", "damages 128"]
        grids, panels = [], VGroup()
        for k in range(3):
            g = kit.grid(R, C, cell=cell)
            for r in range(R):
                for c in range(C):
                    if k == 0:
                        tint = 0.3
                    elif k == 1:
                        tint = 0.18 if r % 2 else 0.4
                    else:
                        tint = 0.18 if (r + c // 2) % 2 else 0.4
                    g.paint(g.cell(r, c), th.INFO, tint=tint)
            grids.append(g)
            head = kit.label(titles[k], size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
            shp = kit.label(shapes[k], size=th.SIZE_LABEL, color=th.INK_3, font=th.MONO)
            hit = kit.label(hits[k], size=th.SIZE_LABEL, color=th.BAD)
            panels.add(VGroup(head, g, shp, hit).arrange(DOWN, buff=0.22))
        panels.arrange(RIGHT, buff=0.75)
        for k in (1, 2):
            panels[k].shift(UP * (grids[0].get_y() - grids[k].get_y()))
        bad_key = swatch("outlier", th.BAD)
        bad_key[0].set_fill(th.BAD, opacity=1)
        key = VGroup(swatch("one scale's weights", th.INFO), bad_key,
                     swatch("damaged by it", th.HL_D)).arrange(RIGHT, buff=0.6)
        note = kit.label("Each color band shares one scale. Lab weight: 512 rows × 1024 inputs.",
                         size=th.SIZE_LABEL, color=th.INK_3)
        col = VGroup(panels, key, note).arrange(DOWN, buff=0.55)
        self.place(col)
        outs = [g.cell(orow, ocol) for g in grids]
        regions = [grids[0].block(0, R, 0, C), grids[1].row(orow), grids[2].block(orow, orow + 1, 8, 10)]
        with self.narrate(
            "So shrink the set of weights that share a scale. "
            "Each band of color here is one scale, and the red cell is an outlier. "
            "Per tensor, the outlier damages all half a million weights. "
            "Per channel gives each output row its own scale, so the damage stays in one row. "
            "Per group gives each block of 128 weights its own scale: a 512 by 8 grid of scales for the lab. "
            "Now the outlier damages only its 127 neighbors."
        ) as t:
            self.play(*[FadeIn(VGroup(p[0], p[1])) for p in panels], FadeIn(note), run_time=0.9)
            t.wait_until("Each band")
            self.play(FadeIn(key), *[o.square.animate.set_fill(th.BAD, opacity=1) for o in outs], run_time=0.8)
            t.wait_until("Per tensor, the outlier")
            self.play(grids[0].fill(VGroup(*[c for c in regions[0] if c is not outs[0]]), th.HL_D),
                      FadeIn(panels[0][2]), FadeIn(panels[0][3]), run_time=1.0)
            t.wait_until("Per channel gives")
            self.play(FadeIn(panels[1][2]), run_time=0.5)
            self.play(grids[1].fill(VGroup(*[c for c in regions[1] if c is not outs[1]]), th.HL_D),
                      FadeIn(panels[1][3]), run_time=0.9)
            t.wait_until("Per group gives")
            self.play(Create(grids[2].outline(orow, orow + 1, 8, 10, color=th.INK, stroke=3)),
                      FadeIn(panels[2][2]), run_time=0.7)
            t.wait_until("Now the outlier")
            self.play(grids[2].fill(VGroup(*[c for c in regions[2] if c is not outs[2]]), th.HL_D),
                      FadeIn(panels[2][3]), run_time=0.8)

    def metadata(self):
        self.slide("metadata", "Scales cost 1.56% at a group size of 128",
                   book_section="s-per-channel-and-group-wise-scaling")
        f = kit.formula(r"b(", r"g", r") = 1 + \frac{2}{", r"g", r"}\ \text{bytes per weight}")
        f[1].set_color(N_COL)
        f[3].set_color(N_COL)
        bc = kit.bar_compare(
            [("g = 32", 6.25, th.INFO, "6.25%"), ("g = 64", 3.125, th.INFO, "3.13%"),
             ("g = 128", 1.5625, self.color, "1.56%"), ("Per channel, 5120 wide", 0.039, th.INFO, "0.04%")],
            width=6.2, bar_height=0.5, gap=0.3,
        )
        cap = kit.label("Scale bytes on top of the int8 weights", size=th.SIZE_SMALL, color=th.INK_3)
        lab = kit.prose(r"Lab weight at $g = 128$: $524{,}288 + 8{,}192 = 532{,}480$ bytes, 50.78\% of bf16.",
                        size=th.SIZE_SMALL, width=None)
        col = VGroup(f, VGroup(cap, bc).arrange(DOWN, buff=0.3), lab).arrange(DOWN, buff=0.55)
        self.place(col)
        with self.narrate(
            "Scales cost bytes too. "
            "With int8 weights and 2-byte scales, each weight costs 1 byte, plus 2 over g. "
            "At a group size of 128, the scales add 1.56 percent. "
            "The total is 50.78 percent of bfloat16. "
            "Groups of 32 would add 6.25 percent, four times the metadata, for a small gain in error. "
            "That's why 128 is the standard. "
            "The lab's weight comes to 532,480 bytes, exactly that fraction."
        ) as t:
            t.wait_until("With int8 weights")
            self.play(FadeIn(f), FadeIn(cap), FadeIn(bc.labels), run_time=0.9)
            t.wait_until("At a group size")
            self.play(GrowFromEdge(bc.bars[2], LEFT), FadeIn(bc.values[2]), run_time=0.8)
            t.wait_until("Groups of 32")
            self.play(GrowFromEdge(bc.bars[0], LEFT), FadeIn(bc.values[0]), GrowFromEdge(bc.bars[1], LEFT),
                      FadeIn(bc.values[1]), run_time=0.9)
            t.wait_until("That's why 128")
            self.play(GrowFromEdge(bc.bars[3], LEFT), FadeIn(bc.values[3]),
                      Indicate(bc.labels[2], color=self.color), run_time=0.8)
            t.wait_until("The lab's weight")
            self.play(FadeIn(lab, shift=UP * 0.1), run_time=0.7)

    # ------------------------------------------------------------------
    # Weight-only against weight-and-activation

    def decode_halves(self):
        self.slide("decode-halves", "Weight-only int8 halves decode time",
                   book_section="s-weight-only-against-weight-and-activation")
        w16 = kit.box("W8A16: weight-only", sublabel="int8 weights, bf16 matmul", color=self.color, width=5.2)
        w88 = kit.box("W8A8: weights and activations", sublabel="int8 matmul, int32 accumulator", color=th.INFO,
                      width=5.2)
        boxes = VGroup(w16, w88).arrange(RIGHT, buff=0.6)
        f = kit.formula(r"t_{\text{decode}} \approx \frac{\text{weight bytes}}{\text{bandwidth}}",
                        size=th.SIZE_MATH_SMALL)
        bc = kit.bar_compare(
            [("bf16", 42.2, th.INFO, "42.2 ms  ·  23.7 tokens/s"),
             ("int8, g = 128", 21.4, self.color, "21.4 ms  ·  46.7 tokens/s")],
            width=5.6, bar_height=0.55, gap=0.4,
        )
        cap = kit.label("Decode floor per token at 1275 GB/s", size=th.SIZE_SMALL, color=th.INK_3)
        speed = kit.pill("1.97× faster", self.color, size=th.SIZE_SMALL)
        col = VGroup(boxes, f, VGroup(cap, bc).arrange(DOWN, buff=0.3), speed).arrange(DOWN, buff=0.5)
        self.place(col)
        with self.narrate(
            "Weight-only quantization, W8A16, stores int8 weights and dequantizes them to bfloat16 in registers. "
            "W8A8 quantizes the activations too, and multiplies in int8. "
            "Decode at batch 1 reads every weight once, so its time is bytes over bandwidth. "
            "In bfloat16, that's 42.2 ms per token. "
            "In int8 with groups of 128, the floor drops to 21.4 ms. "
            "That's 1.97 times faster, because nothing else was the bottleneck."
        ) as t:
            self.play(FadeIn(w16, shift=UP * 0.1), run_time=0.7)
            t.wait_until("W8A8 quantizes")
            self.play(FadeIn(w88, shift=UP * 0.1), run_time=0.7)
            t.wait_until("Decode at batch 1")
            self.play(FadeIn(f), FadeIn(cap), FadeIn(bc.labels), run_time=0.8)
            t.wait_until("In bfloat16, that's")
            self.play(GrowFromEdge(bc.bars[0], LEFT), FadeIn(bc.values[0]), run_time=0.8)
            t.wait_until("In int8 with groups")
            self.play(GrowFromEdge(bc.bars[1], LEFT), FadeIn(bc.values[1]), run_time=0.8)
            t.wait_until("That's 1.97")
            self.play(FadeIn(speed, scale=0.9), run_time=0.6)

    def roofline(self):
        self.slide("roofline", "Dequantization is free; prefill gains nothing",
                   book_section="s-weight-only-against-weight-and-activation")
        lg = np.log10
        ax = kit.axes([0, 4, 1], [0, 3, 1], width=6.6, height=4.4, y_label=r"\text{FLOP/s (log)}")
        xt = kit.label("FLOPs per byte (log scale)", size=th.SIZE_LABEL, color=th.INK_3)
        xt.next_to(ax.c2p(2, 0), DOWN, buff=0.6)
        ax.add(xt)
        for v, text in ((0, "1"), (1, "10"), (2, "100"), (3, "1000"), (4, "10^4")):
            n = kit.math_label(text, size=th.SIZE_LABEL - 2, color=th.INK_3).next_to(ax.c2p(v, 0), DOWN, buff=0.16)
            ax.add(n)
        ridge_x = lg(161)
        top = lg(312)
        roof = VGroup(Line(ax.c2p(0, lg(1.935)), ax.c2p(ridge_x, top), color=th.INK_2, stroke_width=th.STROKE_CURVE),
                      Line(ax.c2p(ridge_x, top), ax.c2p(4, top), color=th.INK_2, stroke_width=th.STROKE_CURVE))
        rline = DashedLine(ax.c2p(ridge_x, 0), ax.c2p(ridge_x, top), color=th.INK_4, stroke_width=th.STROKE_AXIS)
        rlab = kit.label("ridge 161", size=th.SIZE_LABEL, color=th.INK_3).next_to(ax.c2p(ridge_x, top), UP, buff=0.15)
        p_bf = Dot(ax.c2p(0, lg(1.935)), radius=0.1, color=th.INFO)
        p_i8 = ax.c2p(lg(3), lg(3 * 1.935))
        p_pf = Dot(ax.c2p(lg(2000), top), radius=0.1, color=th.VIOLET)
        mover = Dot(ax.c2p(0, lg(1.935)), radius=0.1, color=self.color)
        l_bf = kit.label("bf16 decode: 1", size=th.SIZE_LABEL, color=th.INFO).next_to(p_bf, UR, buff=0.08)
        l_bf.shift(DOWN * 0.45 + RIGHT * 0.05)
        l_i8 = kit.label("int8 decode: 3", size=th.SIZE_LABEL, color=self.color)
        l_i8.next_to(p_i8, RIGHT, buff=0.18).shift(DOWN * 0.15)
        l_pf = kit.label("prefill: about 2000", size=th.SIZE_LABEL, color=th.VIOLET).next_to(p_pf, DOWN, buff=0.18)
        plot = VGroup(ax, roof, rline, rlab, p_bf, p_pf, l_bf, l_i8, l_pf)
        side = VGroup(
            kit.prose(r"int8: 1 byte, 3 FLOPs per weight.", size=th.SIZE_SMALL, width=5.4),
            kit.prose(r"$161 / 3 \approx 54\times$ below the ridge.", size=th.SIZE_SMALL, width=5.4),
            kit.prose(r"Prefill is compute bound: same FLOPs, no gain.", size=th.SIZE_SMALL, width=5.4),
            kit.callout("warning", r"Dequantize inside the matmul, in registers.", title="Not before it",
                        width=5.4, size=th.SIZE_SMALL),
        ).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        layout = VGroup(plot, side).arrange(RIGHT, buff=0.5)
        self.place(layout)
        p_i8 = ax.c2p(lg(3), lg(3 * 1.935))
        with self.narrate(
            "Dequantizing adds work, but not enough to matter. "
            "The bfloat16 kernel does 1 FLOP per byte. "
            "The int8 kernel reads 1 byte per weight and does 3 FLOPs: one multiply to dequantize, then the "
            "multiply-add. "
            "That's still 54 times below the ridge point of 161, so the extra multiply is free. "
            "Prefill sits around 2000, far past the ridge. "
            "It's compute bound, and its FLOPs don't change. "
            "One warning: dequantize inside the matmul kernel, never the whole tensor first."
        ) as t:
            self.play(Create(ax), Create(roof), run_time=1.2)
            t.wait_until("The bfloat16 kernel")
            self.play(FadeIn(p_bf, scale=0.5), FadeIn(l_bf), run_time=0.7)
            t.wait_until("The int8 kernel")
            self.add(mover)
            self.play(FadeIn(side[0]), run_time=0.6)
            self.play(mover.animate.move_to(p_i8), run_time=1.0)
            self.play(FadeIn(l_i8), run_time=0.4)
            t.wait_until("That's still 54")
            self.play(Create(rline), FadeIn(rlab), FadeIn(side[1]), run_time=0.8)
            t.wait_until("Prefill sits")
            self.play(FadeIn(p_pf, scale=0.5), FadeIn(l_pf), run_time=0.7)
            t.wait_until("It's compute bound")
            self.play(FadeIn(side[2]), Indicate(p_pf, color=th.VIOLET, scale_factor=1.6), run_time=0.8)
            t.wait_until("One warning")
            self.play(FadeIn(side[3], shift=UP * 0.1), run_time=0.7)

    # ------------------------------------------------------------------
    # Ampere's int8 tensor cores

    def int8_cores(self):
        self.slide("int8-cores", "Integer matmuls need the scales outside the sum",
                   book_section="s-what-ampere-s-int8-tensor-cores-do")
        f = kit.formula(r"y_j = \sum_k x_k W_{kj} =", r"s_x", r"\,", r"s_{w,j}", r"\sum_k q_{x,k}\, q_{w,kj}")
        f[1].set_color(S_COL)
        f[3].set_color(S_COL)
        int_t = kit.label("pure int8 products, int32 sum", size=th.SIZE_LABEL, color=th.INFO)
        head = kit.formula(r"5120 \times 127 \times 127 = 8.26 \times 10^{7}", r"\;\ll\;",
                           r"2^{31} = 2.15 \times 10^{9}", size=th.SIZE_MATH_SMALL)
        head_t = kit.pill("26× headroom", th.GOOD, size=th.SIZE_SMALL)
        headroom = VGroup(head, head_t).arrange(RIGHT, buff=0.5)
        n = 40
        strip_a = VGroup(*[Rectangle(width=0.22, height=0.4, stroke_color=th.RULE_STRONG,
                                     stroke_width=th.STROKE_HAIRLINE, fill_color=th.tint(th.INFO, 0.3),
                                     fill_opacity=1) for _ in range(n)]).arrange(RIGHT, buff=0)
        strip_b = strip_a.copy()
        la = kit.label("Per channel: one int32 sum over k, one float multiply at the end", size=th.SIZE_LABEL,
                       color=th.INK_2)
        lb = kit.label("Per group: stop every 128 steps to convert, scale, and add", size=th.SIZE_LABEL,
                       color=th.INK_2)
        end_a = kit.pill("× scales", S_COL, size=th.SIZE_LABEL)
        ra = VGroup(la, VGroup(strip_a, end_a).arrange(RIGHT, buff=0.25)).arrange(DOWN, buff=0.18,
                                                                                aligned_edge=LEFT)
        rb = VGroup(lb, strip_b).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        col = VGroup(VGroup(f, int_t).arrange(DOWN, buff=0.3), headroom, ra, rb).arrange(DOWN, buff=0.45)
        rb.align_to(ra, LEFT)
        self.place(col)
        int_t.next_to(f[4], DOWN, buff=0.25)
        intpart = SurroundingRectangle(f[4], color=th.INFO, buff=0.1, stroke_width=3)
        marks = VGroup(*[Line(strip_b[i].get_corner(UR) + UP * 0.08, strip_b[i].get_corner(DR) + DOWN * 0.08,
                              color=th.BAD, stroke_width=4) for i in range(n)])
        with self.narrate(
            "Ampere's tensor cores have an integer mode: int8 inputs, int32 accumulation. "
            "To use it, the whole sum must be integers. "
            "Give the activations one scale, and each weight column its own. "
            "Neither depends on k, so both leave the sum. "
            "What remains is a pure integer dot product. "
            "Even in the worst case, it stays 26 times below the int32 limit. "
            "Group-wise scales break this. "
            "They change every 128 steps, so the loop must stop and rescale. "
            "Group-wise scaling is a weight-only technique."
        ) as t:
            t.wait_until("Give the activations")
            self.play(FadeIn(f), run_time=0.9)
            t.wait_until("Neither depends")
            self.play(Indicate(f[1], color=S_COL), Indicate(f[3], color=S_COL), run_time=0.9)
            t.wait_until("What remains")
            self.play(Create(intpart), FadeIn(int_t), run_time=0.7)
            self.play(FadeIn(la), FadeIn(strip_a), FadeIn(end_a, shift=LEFT * 0.1), run_time=0.6)
            t.wait_until("Even in the worst")
            self.play(FadeIn(head), run_time=0.8)
            self.play(FadeIn(head_t, scale=0.9), run_time=0.6)
            t.wait_until("They change every")
            self.play(FadeIn(lb), FadeIn(strip_b), run_time=0.6)
            self.play(LaggedStart(*[Create(m) for m in marks], lag_ratio=0.06), run_time=1.6)

    def no_fp8(self):
        self.slide("no-fp8", "The A100 has no FP8 tensor cores", book_section="s-ampere-has-no-fp8")
        rel = kit.prose(r"E4M3: 3 mantissa bits plus the implicit 1. Worst-case rounding error "
                        r"$2^{-4} = 6.25\%$, \textit{relative}, at every magnitude.", size=th.SIZE_SMALL, width=11.0)
        a = kit.box("Ampere: A100", sublabel="capability 8.0  ·  no FP8", color=th.BAD, width=3.8)
        b = kit.box("Ada", sublabel="8.9  ·  FP8 tensor cores", color=th.GOOD, width=3.8)
        c = kit.box("Hopper", sublabel="FP8 tensor cores", color=th.GOOD, width=3.8)
        chips = VGroup(a, b, c).arrange(RIGHT, buff=0.5)
        here = kit.label("this course", size=th.SIZE_LABEL, color=th.BAD)
        panel = kit.code(CAPABILITY, "python")
        store = kit.prose(r"FP8 storage still works, unpacked in software: the bandwidth saving, no compute "
                          r"speedup. On Ampere, int8 does the same with a faster unpack.", size=th.SIZE_SMALL,
                          width=11.0, color=th.INK_2)
        col = VGroup(rel, VGroup(chips, here).arrange(DOWN, buff=0.15), store, panel).arrange(DOWN, buff=0.5)
        here.align_to(a, LEFT).shift(RIGHT * 0.2)
        self.place(col)
        with self.narrate(
            "FP8 would suit outliers well. "
            "Its precision is relative: E4M3's worst-case rounding error is 6.25 percent at every magnitude. "
            "But the A100 is Ampere, compute capability 8.0, with no FP8 tensor cores. "
            "Ada and Hopper have them. "
            "You can store FP8 weights and unpack them in software, but the math stays in bfloat16. "
            "On Ampere, int8 gives the same saving with a faster unpack. "
            "Check the capability in code first."
        ) as t:
            t.wait_until("Its precision is")
            self.play(FadeIn(rel), run_time=0.8)
            t.wait_until("But the A100")
            self.play(FadeIn(a, shift=UP * 0.1), FadeIn(here), run_time=0.7)
            t.wait_until("Ada and Hopper")
            self.play(FadeIn(b, shift=UP * 0.1), FadeIn(c, shift=UP * 0.1), run_time=0.7)
            t.wait_until("You can store")
            self.play(FadeIn(store), run_time=0.7)
            t.wait_until("Check the capability")
            self.play(FadeIn(panel.frame), *panel.reveal(1, 2), run_time=0.8)
            self.play(panel.highlight(2), run_time=0.5)

    # ------------------------------------------------------------------
    # Activation outliers and calibration

    def activation_outliers(self):
        self.slide("activation-outliers", "Activation outliers sit in a few channels",
                   book_section="s-activation-outliers")
        r = kit.rng(184)
        T, C = 6, 12
        hot = (2, 8)
        mags = np.abs(r.normal(0, 1, (T, C)))
        X = kit.grid(T, C, cell=0.36)
        W = kit.grid(C, 5, cell=0.24)
        for i in range(T):
            for j in range(C):
                tint = 0.85 if j in hot else 0.08 + 0.12 * min(mags[i, j], 2.5) / 2.5
                X.paint(X.cell(i, j), self.color, tint=tint)
        for i in range(C):
            for j in range(5):
                W.paint(W.cell(i, j), th.INFO, tint=0.18)
        xl = kit.prose(r"activations $x$: tokens $\times$ channels", size=th.SIZE_LABEL, width=None, color=th.INK_3)
        wl = kit.prose(r"weights $W$", size=th.SIZE_LABEL, width=None, color=th.INK_3)
        xg = VGroup(xl, X).arrange(DOWN, buff=0.2)
        wg = VGroup(wl, W).arrange(DOWN, buff=0.2)
        times = kit.formula(r"\times", size=th.SIZE_MATH)
        fig = VGroup(xg, times, wg).arrange(RIGHT, buff=0.5)
        hot_t = kit.label("up to 100× typical, same channels every token", size=th.SIZE_LABEL, color=self.color)
        f = kit.formula(r"y = xW = \left(x \operatorname{diag}(\mathbf{c})^{-1}\right)"
                        r"\left(\operatorname{diag}(\mathbf{c})\, W\right)", size=th.SIZE_MATH_SMALL)
        awq = kit.prose(r"\textbf{AWQ}: protect the weight channels that multiply large activations.",
                        size=th.SIZE_SMALL, width=None)
        col = VGroup(fig, hot_t, f, awq).arrange(DOWN, buff=0.4)
        self.place(col)
        hot_t.next_to(X, DOWN, buff=0.2)
        col2 = VGroup(fig, hot_t)
        VGroup(f, awq).arrange(DOWN, buff=0.4).next_to(col2, DOWN, buff=0.45)
        hot_cells = VGroup(*[X.cell(i, j) for j in hot for i in range(T)])
        hot_rows = VGroup(*[W.cell(j, k) for j in hot for k in range(5)])
        with self.narrate(
            "Activations are much harder to quantize than weights. "
            "They hold outliers up to 100 times the typical size. "
            "But the outliers sit in a few channels, the same ones for every token. "
            "SmoothQuant divides each activation channel by a factor c, and multiplies the matching weight row by c. "
            "The product doesn't change. "
            "The loud channels shrink, and both tensors become quantizable. "
            "AWQ uses the same statistics to protect the weights that multiply large activations."
        ) as t:
            self.play(FadeIn(xg), FadeIn(times), FadeIn(wg), run_time=0.9)
            t.wait_until("But the outliers sit")
            self.play(Indicate(hot_cells, color=self.color, scale_factor=1.05), FadeIn(hot_t), run_time=1.0)
            t.wait_until("SmoothQuant divides")
            self.play(FadeIn(f), run_time=0.8)
            t.wait_until("The loud channels")
            self.play(X.fill(hot_cells, self.color, tint=0.2), W.fill(hot_rows, th.INFO, tint=0.6), run_time=1.4)
            t.wait_until("AWQ uses")
            self.play(FadeIn(awq), run_time=0.7)

    def calibration(self):
        self.slide("calibration", "Calibrate the clip, not the maximum", book_section="s-calibration")
        r = kit.rng(183)
        x = r.standard_t(2.5, 100_000)
        m = np.abs(x).max()
        fr = np.linspace(0.2, 1.0, 161)
        clip, rnd = [], []
        for f in fr:
            tau = f * m
            s = tau / 127
            xc = np.clip(x, -tau, tau)
            q = np.round(xc / s) * s
            clip.append(np.mean((xc - x) ** 2))
            rnd.append(np.mean((q - xc) ** 2))
        clip, rnd = np.array(clip), np.array(rnd)
        tot = clip + rnd
        norm = tot.min()
        clip, rnd, tot = clip / norm, rnd / norm, tot / norm
        best = fr[int(np.argmin(tot))]
        # x axis is tau / max - 0.2, so the axis lines start at the plot's corner.
        xs = fr - 0.2
        ax = kit.axes([0, 0.8, 0.2], [0, 3, 1], width=6.6, height=3.8, y_label=r"\text{squared error}")
        xt = kit.prose(r"clipping threshold $\tau$", size=th.SIZE_LABEL, color=th.INK_3, width=None)
        xt.next_to(ax.c2p(0.4, 0), DOWN, buff=0.6)
        ax.add(xt)
        c_clip = polyline(ax, xs, clip, th.INFO, ymax=3)
        c_rnd = polyline(ax, xs, rnd, N_COL, ymax=3)
        c_tot = polyline(ax, xs, tot, self.color, ymax=3)
        vbest = DashedLine(ax.c2p(best - 0.2, 0), ax.c2p(best - 0.2, 1), color=th.INK_3, stroke_width=th.STROKE_AXIS)
        dbest = Dot(ax.c2p(best - 0.2, 1), radius=0.09, color=self.color)
        lbest = kit.math_label(r"\tau^{\star}", size=th.SIZE_SMALL, color=th.INK).next_to(ax.c2p(best - 0.2, 0),
                                                                                         DOWN, buff=0.15)
        lmax = kit.label("observed max", size=th.SIZE_LABEL, color=th.INK_3).next_to(ax.c2p(0.8, 0), DOWN, buff=0.15)
        mx = Line(ax.c2p(0.8, 0), ax.c2p(0.8, 3), color=th.INK_4, stroke_width=th.STROKE_AXIS)
        plot = VGroup(ax, c_clip, c_rnd, c_tot, vbest, dbest, lbest, lmax, mx)
        keys = VGroup(line_key("clipping error", th.INFO), line_key("rounding error", N_COL),
                      line_key("total", self.color)).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        f = kit.formula(r"\tau^{\star} = \arg\min_{\tau}\, \mathbb{E}\big[(Q_\tau(x) - x)^2\big]",
                        size=th.SIZE_MATH_SMALL)
        cap = kit.prose(r"Simulated heavy-tailed activations, int8 with $s = \tau/127$.", size=th.SIZE_LABEL,
                        width=4.4, color=th.INK_3)
        side = VGroup(f, keys, cap).arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        layout = VGroup(plot, side).arrange(RIGHT, buff=0.6)
        self.place(layout)
        with self.narrate(
            "An activation's range is known only once you run data through the model. "
            "Calibration runs a few hundred representative sequences. "
            "Using the observed maximum lets one outlier set the scale forever. "
            "Instead, pick the clipping threshold that minimizes the error. "
            "Raising it clips less, but rounds more coarsely. "
            "The best threshold balances the two, below the maximum. "
            "And calibrate on what you serve."
        ) as t:
            self.play(Create(ax), run_time=0.9)
            t.wait_until("Using the observed")
            self.play(Create(mx), FadeIn(lmax), run_time=0.7)
            t.wait_until("Instead, pick")
            self.play(FadeIn(f), run_time=0.7)
            t.wait_until("Raising it clips")
            self.play(Create(c_clip), FadeIn(keys[0]), run_time=1.0)
            self.play(Create(c_rnd), FadeIn(keys[1]), run_time=1.0)
            t.wait_until("The best threshold")
            self.play(Create(c_tot), FadeIn(keys[2]), run_time=1.0)
            self.play(Create(vbest), FadeIn(dbest, scale=0.5), FadeIn(lbest), FadeIn(cap), run_time=0.7)

    # ------------------------------------------------------------------
    # Quantizing the KV cache

    def kv_cache(self):
        self.slide("kv-cache", "Keys need more precision than values", book_section="s-quantizing-the-kv-cache")
        bc = kit.bar_compare(
            [("bf16", 64, th.INFO, "64 KiB"), ("int8 + scales", 32.25, self.color, "32.25 KiB")],
            width=3.6, bar_height=0.42, gap=0.25, size=th.SIZE_LABEL,
        )
        bcap = kit.label("KV cache per token  ·  32k context: 2.0 → 1.008 GiB", size=th.SIZE_LABEL, color=th.INK_3)
        top = VGroup(bcap, bc).arrange(DOWN, buff=0.2)
        # Keys: attention weights, one score nudged by 0.3.
        scores = np.array([0.4, 1.3, 0.2, 2.0, 0.9, 1.6])
        w = np.exp(scores)
        hmax = 2.0
        wb = VGroup(*[Rectangle(width=0.42, height=hmax * v / w.max(), fill_color=self.color, fill_opacity=0.8,
                                stroke_width=0) for v in w]).arrange(RIGHT, buff=0.14, aligned_edge=DOWN)
        k_head = kit.label("Keys feed a softmax", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
        k_note = kit.prose(r"score error $\delta$: weight $\times\, e^{\delta}$", size=th.SIZE_LABEL, width=None,
                           color=th.INK_3)
        k_vals = kit.label("δ = 0.1: +10.5%   ·   δ = 0.3: +35%", size=th.SIZE_LABEL, color=th.BAD)
        kpan = VGroup(k_head, wb, k_note, k_vals).arrange(DOWN, buff=0.25)
        # Values: 100 residuals average down.
        r = kit.rng(182)
        res = r.uniform(-0.5, 0.5, 100)
        mean = res.mean()
        vl = NumLine(-0.5, 0.5, 4.2, majors=[-0.5, 0, 0.5], labels={-0.5: "-s/2", 0: "0", 0.5: "s/2"})
        jit = r.uniform(0.15, 0.85, 100)
        vdots = VGroup(*[Dot(vl.p(v) + UP * j, radius=0.04, color=th.INFO, fill_opacity=0.7)
                         for v, j in zip(res, jit)])
        vfig = VGroup(vl, vdots)
        v_head = kit.label("Values are averaged", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
        v_note = kit.prose(r"$n_{\text{eff}} = 100$: RMS error $\div\, 10$", size=th.SIZE_LABEL, width=None,
                           color=th.INK_3)
        v_vals = kit.label("no amplification, ever", size=th.SIZE_LABEL, color=th.GOOD)
        vpan = VGroup(v_head, vfig, v_note, v_vals).arrange(DOWN, buff=0.25)
        pans = VGroup(kpan, vpan).arrange(RIGHT, buff=1.2, aligned_edge=UP)
        col = VGroup(top, pans).arrange(DOWN, buff=0.5)
        self.place(col)
        bump = wb[4]
        grown = bump.copy().stretch(np.exp(0.3), 1, about_edge=DOWN).set_fill(th.BAD, opacity=0.8)
        mdot = Dot(vl.p(mean) + DOWN * 0.0, radius=0.11, color=th.GOOD)
        with self.narrate(
            "Quantize the KV cache with one scale per vector, and a token drops from 64 KiB to 32.25 KiB. "
            "But keys and values need different treatment. "
            "Keys feed a softmax, so a score error is exponentiated. "
            "An error of 0.1 changes a token's weight by 10.5 percent. "
            "An error of 0.3 changes it by 35 percent. "
            "Values are averaged, so their errors shrink. "
            "Spread attention over 100 tokens, and the value error falls tenfold. "
            "So keep keys in bfloat16, and put values in int8."
        ) as t:
            self.play(FadeIn(bcap), FadeIn(bc.labels), run_time=0.6)
            self.play(GrowFromEdge(bc.bars[0], LEFT), FadeIn(bc.values[0]), run_time=0.6)
            self.play(GrowFromEdge(bc.bars[1], LEFT), FadeIn(bc.values[1]), run_time=0.6)
            t.wait_until("Keys feed")
            self.play(FadeIn(k_head), LaggedStart(*[GrowFromEdge(b, DOWN) for b in wb], lag_ratio=0.1),
                      FadeIn(k_note), run_time=1.0)
            t.wait_until("An error of 0.3")
            self.play(Transform(bump, grown), FadeIn(k_vals), run_time=0.9)
            t.wait_until("Values are averaged")
            self.play(FadeIn(v_head), FadeIn(vl), LaggedStart(*[FadeIn(d) for d in vdots], lag_ratio=0.01),
                      run_time=1.2)
            t.wait_until("Spread attention")
            self.play(FadeIn(mdot, scale=0.5), FadeIn(v_note), run_time=0.7)
            t.wait_until("So keep keys")
            self.play(FadeIn(v_vals), run_time=0.6)

    # ------------------------------------------------------------------
    # The model at three precisions

    def three_precisions(self):
        self.slide("three-precisions", "int4 is four times smaller, and costs quality",
                   book_section="s-the-model-at-three-precisions")
        bc = kit.bar_compare(
            [("bf16", 53.8, th.INFO, "53.8 GB  ·  42.2 ms"),
             ("int8, g = 128", 27.3, self.color, "27.3 GB  ·  21.4 ms"),
             ("int4, g = 128", 13.9, th.VIOLET, "13.9 GB  ·  10.9 ms")],
            width=6.8, bar_height=0.55, gap=0.35,
        )
        cap = kit.label("26.9 billion weights: size and decode floor", size=th.SIZE_SMALL, color=th.INK_3)
        freed = kit.prose(r"int8 frees 26.5 GB: about 404,000 more tokens of KV cache.", size=th.SIZE_SMALL,
                          width=None)
        f = kit.formula(r"s_{\text{int4}} = \frac{\beta}{7} = 18.1\, s_{\text{int8}}", size=th.SIZE_MATH_SMALL)
        err = kit.prose(r"Predicted error at $g = 128$: 0.76\% $\to$ about 13.7\%.", size=th.SIZE_SMALL, width=None)
        warn = VGroup(f, err).arrange(RIGHT, buff=0.6)
        col = VGroup(VGroup(cap, bc).arrange(DOWN, buff=0.3), freed, warn).arrange(DOWN, buff=0.55)
        self.place(col)
        b0 = bc.bars[0]
        x_cut = b0.get_left()[0] + b0.width * 27.3 / 53.8
        band = Rectangle(width=b0.get_right()[0] - x_cut, height=b0.height + 0.12, stroke_color=th.GOOD,
                         stroke_width=3, fill_opacity=0).move_to([(x_cut + b0.get_right()[0]) / 2, b0.get_y(), 0])
        with self.narrate(
            "Scale it up to the whole model. "
            "In bfloat16, the weights are 53.8 GB, with a 42.2 ms floor. "
            "At int8, they're 27.3 GB, at 21.4 ms. "
            "At int4, they're 13.9 GB, at 10.9 ms. "
            "Moving to int8 frees 26.5 GB, room for about 404,000 more tokens of KV cache. "
            "But int4 costs quality. "
            "Its step is 18.1 times larger, so the predicted error jumps to about 13.7 percent. "
            "That's where round-to-nearest stops working, and GPTQ or AWQ take over."
        ) as t:
            self.play(FadeIn(cap), FadeIn(bc.labels), run_time=0.7)
            for i, cue in enumerate(["In bfloat16, the weights", "At int8, they're", "At int4, they're"]):
                t.wait_until(cue)
                self.play(GrowFromEdge(bc.bars[i], LEFT), FadeIn(bc.values[i]), run_time=0.8)
            t.wait_until("Moving to int8")
            self.play(Create(band), FadeIn(freed), run_time=0.8)
            t.wait_until("Its step is")
            self.play(FadeIn(f), run_time=0.7)
            self.play(FadeIn(err), Indicate(bc.bars[2], color=th.BAD, scale_factor=1.05), run_time=0.8)

    # ------------------------------------------------------------------
    # Measuring the damage

    def measuring(self):
        self.slide("measuring", "Judge a scheme by perplexity, not weight error",
                   book_section="s-measuring-the-damage")
        b1 = kit.box("1  Layer output error", sublabel="one layer: under 1%", width=3.9)
        b2 = kit.box("2  Logits", sublabel="correlation > 0.999, top-1 > 0.98", width=3.9)
        b3 = kit.box("3  Perplexity", sublabel="increase under about 1%", color=self.color, width=3.9)
        row = VGroup(b1, b2, b3).arrange(RIGHT, buff=0.55)
        axis = Arrow(LEFT * 6, RIGHT * 6, buff=0, color=th.INK_4, stroke_width=3, tip_length=0.2)
        axis_t = kit.label("more cost, more useful", size=th.SIZE_LABEL, color=th.INK_3)
        rel = kit.formula(r"\frac{\operatorname{Var}(\Delta y_j)}{\operatorname{Var}(y_j)}",
                          r"= \frac{\lVert x \rVert^2 \sigma_{\text{err}}^2}{\lVert x \rVert^2 \sigma_W^2}",
                          size=th.SIZE_MATH_SMALL)
        rel_t = kit.prose(r"Independent errors: output error equals weight error. AWQ's point is that "
                          r"real errors aren't independent of $x$.", size=th.SIZE_SMALL, width=6.0)
        bottom = VGroup(rel, rel_t).arrange(RIGHT, buff=0.6)
        warn = kit.callout("warning", r"Doing only the first check is how broken quantization schemes ship.",
                           title="The trap", width=9.0, size=th.SIZE_SMALL)
        col = VGroup(row, VGroup(axis, axis_t).arrange(DOWN, buff=0.12), bottom, warn).arrange(DOWN, buff=0.45)
        axis.stretch_to_fit_width(row.width)
        self.place(col)
        with self.narrate(
            "Weight error is only a proxy. "
            "Check the output three ways, in increasing cost. "
            "First, one layer's output error, under 1 percent. "
            "With independent errors, it equals the weight error. "
            "Second, the logits: correlation above 0.999, and top-1 agreement above 0.98. "
            "Third, perplexity on held-out text, the real test. "
            "Doing only the first check is how broken schemes ship."
        ) as t:
            t.wait_until("Check the output")
            self.play(GrowArrow(axis), FadeIn(axis_t), run_time=0.8)
            t.wait_until("First, one layer")
            self.play(FadeIn(b1, shift=UP * 0.1), run_time=0.6)
            t.wait_until("With independent errors")
            self.play(FadeIn(rel), FadeIn(rel_t), run_time=0.8)
            t.wait_until("Second, the logits")
            self.play(FadeIn(b2, shift=UP * 0.1), run_time=0.6)
            t.wait_until("Third, perplexity")
            self.play(FadeIn(b3, shift=UP * 0.1), run_time=0.6)
            t.wait_until("Doing only")
            self.play(FadeIn(warn, shift=UP * 0.1), run_time=0.7)
