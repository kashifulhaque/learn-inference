"""Chapter 10, the roofline.

Teaches content/chapters/10-roofline.md in its order. Every number below was
recomputed in Python (exact arithmetic, then rounded as the chapter does).

Rates (A100 80GB PCIe): C = 312 TFLOP/s; beta = 1935 GB/s rated, 2039 GB/s
for SXM4, 1275 GB/s measured by a plain copy.
- Ridge: 312e12 / 1.935e12 = 161.24 ("161.2", "161"); 312 / 2.039 = 153.02
  ("153.0"); 312 / 1.275 = 244.71 ("244.7", "245").
- Tiny case, 1 GB and 1 GFLOP: 1e9 / 1.935e12 = 0.5168 ms ("0.517 ms");
  1e9 / 312e12 = 3.205 us ("3.2 us"); idle fraction 1 - 3.205e-6 / 5.168e-4
  = 99.38% ("over 99 percent").
- Plot coordinates: slope P = 1.935 x I TFLOP/s; ceiling 312 TFLOP/s.

Bandwidth: 1275 / 1935 = 0.659 ("66 percent"). 4 GB / 6.3 ms = 634.9 GB/s
("635"). Vector add 1304 GB/s counting 3N. Fused RMSNorm 945 GB/s:
945 / 1275 = 74.1% ("74"), 945 / 1935 = 48.8% ("49").

MLP (h = 5120, i = 17408, e = 2):
- Weights 3 x 5120 x 17408 x 2 = 534,773,760 B = 534.8 MB ("535 MB"); one
  token does 6 h i = 534.8 MFLOP ("535 million flops").
- I(T) = 6Thi / (3hie + 2The) = T / (1 + T / 26112); 3i/2 = 26,112.
- I(1) = 0.99996 ("1.0"); I(16) = 15.99 ("16.0"); I(161) = 160.01 ("160");
  I(512) = 502.2 ("502"); I(2048) = 1899.05 ("1899"); I(4096) = 3540.6.
- Ratio to ridge 161.24: 0.0062, 0.099, 0.992, 3.11, 11.78 ("nearly 12");
  memory bound by 161.25 ("161") at T = 1 and 10.08 ("10") at T = 16.
- Relative error T / 26112: 1 -> 0.0038% ("0.004"), 16 -> 0.061% ("0.06"),
  2048 -> 7.84% ("7.8").
- Spill at T = 4096: 4096 x 17408 x 2 = 142.6 MB ("143 MB"), / 40 MB = 3.57
  ("3.6 times"); I_spilled = 6Thi / (3hie + 2The + 4Tie) = 1842.1 ("1842")
  against 3540.6 ("3540").
- One token: 1 / 161.24 = 0.620% of peak ("0.62 percent").

Decode attention (H = 24, H_kv = 4, d = 256, e = 2):
- I = 4HLd / (2 H_kv L d e) = 2H / (e H_kv) = 24 / 4 = 6; int8 (e = 1)
  gives 12; H_kv = 24 gives 1. 161.24 / 6 = 26.87 ("27", "26.9").
- 32k example, 16 layers: B = 16 x 2 x 4 x 32768 x 256 x 2 = 2,147,483,648
  = 2 GiB; F = 16 x 4 x 24 x 32768 x 256 = 1.288e10; memory floor 2.147e9 /
  1.935e12 = 1.110 ms ("1.11 ms"); compute floor 1.288e10 / 312e12 =
  41.30 us ("41.3 us"); ratio 26.87.

RMSNorm (T = 4096, h = 5120): I = 4Th / (2The) = 2/e = 1. Bytes 2 x 4096 x
5120 x 2 = 83,886,080 = 83.9 MB; FLOPs 4 x 4096 x 5120 = 83.9 MFLOP; memory
floor 83.89e6 / 1935e9 = 43.35 us ("43.4"), at 1275 GB/s 65.79 us ("65.8");
compute floor 0.269 us ("0.27"); ratio 161.2 ("160 times" as the chapter
words it). Measured Triton 88.8 us (chapter value; 83.9 MB / 88.8 us =
944.7 GB/s, the "945"). 129 norms = 64 x 2 + 1.

Embedding: B = 2 x 2048 x 5120 x 2 + 8 x 2048 = 41,959,424 B = 41.96 MB
("about 42 MB"); 41.96e6 / 1935e9 = 21.68 us ("21.7"); at 1275 GB/s
32.9 us. Row 5120 x 2 = 10,240 B = 80 lines of 128 B. Table 248,320 x 5120
x 2 = 2.543e9 B ("2.54 GB"); / 40e6 = 63.6 ("63 times", as the chapter).

Launch overhead (N = 26.9e9, 2 FLOPs each): compute floor 53.8e9 / 312e12 =
0.1724 ms ("0.172 ms"); memory floor 53.8e9 / 1935e9 = 27.80 ms ("27.8").
400 launches: 2.0 ms at 5 us (2.0 / 0.1724 = 11.6, "12 times"; 7.2% of the
memory floor, "7"), 4.0 ms at 10 us (23.2, "23"; 14.4%, "14"). Per-token
RMSNorm 2 x 5120 x 2 = 20,480 B / 1935e9 = 10.58 ns ("10.6 ns");
5 us / 10.58 ns = 472 ("470 times").

Idle SMs: u(b) = b / (108 ceil(b / 108)): 8 -> 7.41% ("7.4"); 108 -> 100%;
109 -> 50.46% ("50.5"); 216 -> 100%; 4096 -> 38 waves, 99.81% ("99.8");
1 block -> 1/108 = 0.926% ("0.93").
"""

import math

from manim import *

import kit
from kit import theme as th

# Colour roles, as the chapter's maths colours use them.
COMPUTE = th.HL_A  # \hla: C, peak arithmetic
MEMORY = th.HL_B  # \hlb: beta, bandwidth
INTENSITY = th.HL_C  # \hlc: I
ASYM = th.HL_D  # \hld: 26112
WEIGHTS = th.VIOLET

PEAK = 312.0  # TFLOP/s
BW = 1.935  # TB/s, so P = BW * I in TFLOP/s
RIDGE = PEAK / BW  # 161.24
H_DIM, I_DIM = 5120, 17408


def mlp_i(t):
    """The MLP's exact intensity at t tokens, in flops per byte."""
    return 6 * t * H_DIM * I_DIM / (3 * H_DIM * I_DIM * 2 + 2 * t * H_DIM * 2)


def attainable(i, bw=BW):
    return min(PEAK, bw * i)


def tick_tex(k):
    return {-1: "0.1", 0: "1", 1: "10", 2: "100", 3: "1000"}.get(k, f"10^{{{k}}}")


class Roof(VGroup):
    """A log-log roofline: axes, decade labels, and the two roofs.

    `self.p(i, perf)` maps an intensity (flops per byte) and a throughput
    (TFLOP/s) to a point. Build every derived mobject after `place()`.
    """

    def __init__(self, width=8.0, height=4.4, x_range=(-1, 4), y_range=(-1, 3), bw=BW):
        super().__init__()
        self.xr, self.yr, self.bw = x_range, y_range, bw
        # Data coordinates are decades above the axis minimum, so the axes
        # meet at the bottom-left corner rather than at (1, 1).
        x0, y0 = x_range[0], y_range[0]
        ax = kit.axes([0, x_range[1] - x0, 1], [0, y_range[1] - y0, 1], width=width, height=height)
        self.ax = ax
        nums = VGroup()
        for k in range(x_range[0], x_range[1] + 1):
            n = kit.math_label(tick_tex(k), size=th.SIZE_LABEL, color=th.INK_3)
            n.next_to(ax.c2p(k - x0, 0), DOWN, buff=0.14)
            nums.add(n)
        for k in range(y_range[0], y_range[1] + 1):
            n = kit.math_label(tick_tex(k), size=th.SIZE_LABEL, color=th.INK_3)
            n.next_to(ax.c2p(0, k - y0), LEFT, buff=0.14)
            nums.add(n)
        xt = kit.label("arithmetic intensity, flops per byte (log)", size=th.SIZE_LABEL, color=th.INK_3)
        xt.next_to(nums[: x_range[1] - x_range[0] + 1], DOWN, buff=0.18).align_to(ax.c2p(x_range[1] - x0, 0), RIGHT)
        yt = kit.label("attainable TFLOP/s (log)", size=th.SIZE_LABEL, color=th.INK_3)
        yt.next_to(ax.c2p(0, y_range[1] - y0), UP, buff=0.18).align_to(ax.c2p(0, 0), LEFT)
        self.nums, self.xt, self.yt = nums, xt, yt
        self.add(ax, nums, xt, yt)

    def p(self, i, perf):
        return self.ax.c2p(math.log10(i) - self.xr[0], math.log10(perf) - self.yr[0])

    def ridge(self, bw=None):
        return PEAK / (bw or self.bw)

    def slope(self, bw=None, color=MEMORY, stroke=th.STROKE_CURVE):
        bw = bw or self.bw
        i0 = 10 ** self.xr[0]
        return Line(self.p(i0, bw * i0), self.p(self.ridge(bw), PEAK), color=color, stroke_width=stroke)

    def ceiling(self, bw=None, color=COMPUTE, stroke=th.STROKE_CURVE):
        return Line(self.p(self.ridge(bw), PEAK), self.p(10 ** self.xr[1], PEAK), color=color, stroke_width=stroke)

    def drop(self, i, color=th.INK_4, bw=None):
        """A dashed line from the x-axis up to the roof at intensity i."""
        top = min(PEAK, (bw or self.bw) * i)
        return DashedLine(self.p(i, 10 ** self.yr[0]), self.p(i, top), color=color, stroke_width=th.STROKE_AXIS,
                          dash_length=0.08)

    def dot(self, i, color, radius=0.1, bw=None):
        return Dot(self.p(i, attainable(i, bw or self.bw)), radius=radius, color=color)


def swatch(color, text, size=th.SIZE_LABEL):
    sq = Square(side_length=0.24, fill_color=color, fill_opacity=0.85, stroke_width=0)
    return VGroup(sq, kit.label(text, size=size, color=th.INK_2)).arrange(RIGHT, buff=0.15)


def row(tab, i):
    return VGroup(*tab.cells[i])


class Lecture(kit.Lecture):
    def construct(self):
        self.title_slide(
            narration=(
                "The roofline. "
                "Before you optimize anything, decide what limits it. "
                "In this lecture, you sketch one picture, put two numbers on it, and then place every operation "
                "in the model on it."
            )
        )
        self.two_waits()
        self.roofline_plot()
        self.operations()
        self.two_floors()
        self.ridge_point()
        self.ridge_moves()
        self.count_both()
        self.copy_ceiling()
        self.mlp_intensity()
        self.mlp_limits()
        self.decode_vs_prefill()
        self.decode_attention()
        self.attention_levers()
        self.rmsnorm()
        self.embedding()
        self.launch_overhead()
        self.idle_sms()
        self.what_goes_wrong()
        self.closing_slide(
            [
                r"Floor $\max(F/C,\ B/\beta)$; ridge 161.2.",
                r"MLP: $I \approx T$, the tokens in the pass.",
                r"Decode attention: 6. RMSNorm 1, embedding 0.",
                r"Batch 1: launches and idle SMs dominate.",
            ],
            narration=(
                "Here's what to keep. "
                "The runtime floor is the larger of the compute time and the memory time. "
                "The ridge point is 161 flops per byte, and about 245 against what a copy reaches. "
                "The MLP's intensity is roughly the number of tokens in the pass. "
                "Decode attention sits at 6, whatever the context or the batch. "
                "At batch 1, launches and idle SMs cost more than the arithmetic. "
                "Next, sampling turns the model's scores into one token."
            ),
            book_section="s-what-goes-wrong",
        )

    # Picture the roofline ----------------------------------------------------

    def two_waits(self):
        self.slide("two-waits", "Every operation waits on one of two things",
                   book_section="s-picture-the-roofline")

        def case(title, door, cooks, verdict, verdict_color):
            head = kit.label(title, size=th.SIZE_BODY, color=th.INK, weight=MEDIUM)
            names = VGroup(kit.label("door: bytes", size=th.SIZE_SMALL, color=th.INK_2),
                           kit.label("cooks: flops", size=th.SIZE_SMALL, color=th.INK_2))
            nw = max(n.width for n in names)
            bars = VGroup()
            for k, (n, w, c) in enumerate(zip(names, (door, cooks), (MEMORY, COMPUTE))):
                y = -k * 0.75
                n.move_to([nw - n.width / 2, y, 0])
                b = Rectangle(width=w, height=0.5, fill_color=c, fill_opacity=0.8, stroke_width=0)
                b.move_to([nw + 0.3 + w / 2, y, 0])
                bars.add(b)
            v = kit.label(verdict, size=th.SIZE_SMALL, color=verdict_color, weight=MEDIUM)
            body = VGroup(names, bars)
            g = VGroup(head, body, v).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
            g.bars, g.names, g.head, g.verdict = bars, names, head, v
            return g

        a = case("A crate of ingredients, a little chopping", 3.8, 0.45, "the cooks wait at the door", MEMORY)
        b = case("A handful of ingredients, hours of chopping", 0.45, 3.8, "the door sits idle", COMPUTE)
        cases = VGroup(a, b).arrange(DOWN, buff=0.6, aligned_edge=LEFT)
        ratio = kit.formula(r"\text{arithmetic intensity}", r"=", r"\frac{\text{flops}}{\text{bytes moved}}",
                            size=th.SIZE_MATH)
        ratio[0].set_color(INTENSITY)
        col = VGroup(cases, ratio).arrange(RIGHT, buff=1.0)
        self.place(col)
        with self.narrate(
            "Every operation on a GPU waits on one of two things: arithmetic, or memory traffic. "
            "Picture a kitchen with fast cooks and one narrow delivery door. "
            "Every dish needs ingredients carried through the door, the bytes, and chopping, the flops. "
            "Bring a crate of ingredients for a little chopping, and the cooks stand at the door, waiting. "
            "Bring a handful of ingredients for hours of chopping, and the door sits idle. "
            "What separates the two cases is one ratio: work per ingredient. "
            "On a GPU, that's flops per byte moved, and it's called arithmetic intensity."
        ) as t:
            t.wait_until("Every dish needs")
            self.play(FadeIn(a.names), FadeIn(b.names), run_time=0.6)
            t.wait_until("Bring a crate")
            self.play(FadeIn(a.head), GrowFromEdge(a.bars[0], LEFT), GrowFromEdge(a.bars[1], LEFT), run_time=1.0)
            self.play(FadeIn(a.verdict), run_time=0.5)
            t.wait_until("Bring a handful")
            self.play(FadeIn(b.head), GrowFromEdge(b.bars[0], LEFT), GrowFromEdge(b.bars[1], LEFT), run_time=1.0)
            self.play(FadeIn(b.verdict), run_time=0.5)
            t.wait_until("On a GPU")
            self.play(FadeIn(ratio, shift=UP * 0.1), run_time=0.8)

    def roofline_plot(self):
        self.slide("roofline-plot", "Two lines make the roof", book_section="s-picture-the-roofline")
        roof = Roof(width=8.6, height=4.4)
        self.place(roof)
        slope, ceil = roof.slope(), roof.ceiling()
        r_dot = Dot(roof.p(RIDGE, PEAK), radius=0.1, color=th.INK)
        r_drop = roof.drop(RIDGE)
        r_lab = VGroup(kit.label("ridge point", size=th.SIZE_LABEL, color=th.INK, weight=MEDIUM),
                       kit.label("161 flops per byte", size=th.SIZE_LABEL, color=th.INK_2)).arrange(
            DOWN, buff=0.06, aligned_edge=RIGHT)
        r_lab.next_to(r_dot, LEFT, buff=0.45).shift(UP * 0.2)
        s_lab = kit.label("1935 GB/s × intensity", size=th.SIZE_LABEL, color=MEMORY, weight=MEDIUM)
        angle = math.atan2(*(np.array(slope.get_end() - slope.get_start())[[1, 0]]))
        s_lab.rotate(angle).move_to(roof.p(3.0, 3.0 * BW)).shift(np.array([-math.sin(angle), math.cos(angle), 0]) * 0.3)
        c_lab = kit.label("312 TFLOP/s peak", size=th.SIZE_LABEL, color=COMPUTE, weight=MEDIUM)
        c_lab.next_to(roof.p(2000, PEAK), DOWN, buff=0.15)
        mem = kit.label("memory bound", size=th.SIZE_SMALL, color=MEMORY, weight=SEMIBOLD)
        mem.move_to(roof.p(10 ** 1.3, 0.35))
        comp = kit.label("compute bound", size=th.SIZE_SMALL, color=COMPUTE, weight=SEMIBOLD)
        comp.move_to(roof.p(10 ** 3.15, 8))
        left_area = Polygon(roof.p(0.1, 0.1), roof.p(0.1, BW * 0.1), roof.p(RIDGE, PEAK), roof.p(RIDGE, 0.1),
                            stroke_width=0, fill_color=MEMORY, fill_opacity=0.08)
        right_area = Polygon(roof.p(RIDGE, 0.1), roof.p(RIDGE, PEAK), roof.p(1e4, PEAK), roof.p(1e4, 0.1),
                             stroke_width=0, fill_color=COMPUTE, fill_opacity=0.08)
        with self.narrate(
            "Put arithmetic intensity on the horizontal axis, and the throughput an operation can reach on the "
            "vertical axis. "
            "Both axes are logarithmic. "
            "On the left, the cooks wait at the door. "
            "Each byte unlocks a fixed amount of work, so throughput is bandwidth times intensity. "
            "On the right, the arithmetic units are the limit, at the A100's peak of 312 TFLOP/s. "
            "The bend is the ridge point, at 161 flops per byte. "
            "Left of it, an operation is memory bound. "
            "Right of it, it's compute bound."
        ) as t:
            self.play(Create(roof.ax), FadeIn(roof.nums), FadeIn(roof.xt), FadeIn(roof.yt), run_time=1.0)
            t.wait_until("Each byte unlocks")
            self.play(Create(slope), run_time=1.2)
            self.play(FadeIn(s_lab), run_time=0.5)
            t.wait_until("On the right")
            self.play(Create(ceil), FadeIn(c_lab), run_time=1.0)
            t.wait_until("The bend is")
            self.play(FadeIn(r_dot, scale=0.5), Create(r_drop), FadeIn(r_lab), run_time=0.8)
            t.wait_until("Left of it")
            self.play(FadeIn(left_area), FadeIn(mem), run_time=0.6)
            t.wait_until("Right of it")
            self.play(FadeIn(right_area), FadeIn(comp), run_time=0.6)

    def operations(self):
        self.slide("operations", "Nearly everything sits on the slope", book_section="s-picture-the-roofline")
        roof = Roof(width=8.6, height=4.4)
        self.place(roof)
        roofs = VGroup(roof.slope(), roof.ceiling())
        r_drop = Dot(roof.p(RIDGE, PEAK), radius=0.08, color=th.INK)
        r_tag = kit.label("ridge 161", size=th.SIZE_LABEL, color=th.INK_3)
        r_tag.next_to(r_drop, LEFT, buff=0.2)
        d1, d6, dp = roof.dot(1.0, self.color), roof.dot(6.0, self.color), roof.dot(1899, self.color)
        l1 = VGroup(kit.label("RMSNorm and decode MLP", size=th.SIZE_LABEL, color=th.INK, weight=MEDIUM),
                    kit.label("1 flop per byte", size=th.SIZE_LABEL, color=th.INK_3)).arrange(
            DOWN, buff=0.06, aligned_edge=LEFT)
        l1.next_to(d1, RIGHT, buff=0.3).shift(DOWN * 0.4)
        l6 = VGroup(kit.label("decode attention", size=th.SIZE_LABEL, color=th.INK, weight=MEDIUM),
                    kit.label("6", size=th.SIZE_LABEL, color=th.INK_3)).arrange(DOWN, buff=0.06, aligned_edge=LEFT)
        l6.next_to(d6, RIGHT, buff=0.3).shift(DOWN * 0.4)
        lp = kit.label("prefill MLP, 2048 tokens: 1899", size=th.SIZE_LABEL, color=th.INK, weight=MEDIUM)
        lp.next_to(dp, UP, buff=0.15).align_to(roof.ax, RIGHT)
        emb = Arrow(roof.p(0.6, 0.18), roof.p(0.1, 0.18), buff=0, color=th.INK_3, stroke_width=3, tip_length=0.18)
        emb_t = kit.label("embedding: 0, off the axis", size=th.SIZE_LABEL, color=th.INK_3)
        emb_t.next_to(emb, RIGHT, buff=0.15)
        with self.narrate(
            "Here's where the model's operations land, before you derive any of them. "
            "RMSNorm sits at 1 flop per byte. "
            "So does the MLP when it processes one token, which is decode at batch 1. "
            "Attention during decode sits at 6. "
            "All three are on the slope, far left of the ridge. "
            "A 2048-token prefill puts the MLP at 1899, under the flat roof. "
            "The embedding lookup does no arithmetic at all, so its intensity is 0. "
            "It falls off the left edge of a log axis."
        ) as t:
            self.play(FadeIn(roof), FadeIn(roofs), FadeIn(r_drop), FadeIn(r_tag), run_time=0.8)
            t.wait_until("RMSNorm sits")
            self.play(FadeIn(d1, scale=0.5), run_time=0.6)
            t.wait_until("So does the MLP")
            self.play(FadeIn(l1), run_time=0.6)
            t.wait_until("Attention during decode")
            self.play(FadeIn(d6, scale=0.5), FadeIn(l6), run_time=0.6)
            t.wait_until("All three are")
            self.play(Indicate(roofs[0], color=MEMORY, scale_factor=1.02), run_time=0.9)
            t.wait_until("A 2048-token prefill puts")
            self.play(FadeIn(dp, scale=0.5), FadeIn(lp), run_time=0.7)
            t.wait_until("The embedding lookup")
            self.play(GrowArrow(emb), FadeIn(emb_t), run_time=0.8)

    # The two rates and the ridge point -------------------------------------

    def two_floors(self):
        self.slide("two-floors", "Each rate sets a floor; the larger one wins",
                   book_section="s-the-two-rates-and-the-ridge-point")
        given = kit.prose(r"An operation reads 1 GB and does 1 GFLOP of arithmetic.", size=th.SIZE_SMALL,
                          width=None)
        bc = kit.bar_compare(
            [("Memory: 1 GB at 1935 GB/s", 0.517, MEMORY, "0.517 ms"),
             ("Compute: 1 GFLOP at 312 TFLOP/s", 0.0032, COMPUTE, "3.2 µs")],
            width=6.5, bar_height=0.5, gap=0.35,
        )
        idle = kit.label("arithmetic idle over 99% of the time", size=th.SIZE_SMALL, color=th.BAD, weight=MEDIUM)
        f1 = kit.formula(r"t_{\text{compute}} = \frac{F}{C}", r",\qquad", r"t_{\text{memory}} = \frac{B}{\beta}")
        f1[0].set_color(COMPUTE)
        f1[2].set_color(MEMORY)
        f2 = kit.formula(r"t \;\ge\; \max\!\left(\frac{F}{C},\ \frac{B}{\beta}\right)")
        box = SurroundingRectangle(f2, color=self.color, buff=0.16, corner_radius=0.08, stroke_width=3)
        col = VGroup(given, VGroup(bc, idle).arrange(DOWN, buff=0.3, aligned_edge=LEFT), f1, VGroup(f2, box)
                     ).arrange(DOWN, buff=0.5)
        self.place(col)
        with self.narrate(
            "Start with a tiny case: an operation reads 1 GB, and does a billion flops of arithmetic. "
            "At 1935 GB/s, the bytes take 0.517 ms to arrive. "
            "At 312 TFLOP/s, the arithmetic takes 3.2 µs. "
            "The operation can't finish before its bytes arrive, so the arithmetic sits idle over 99 percent of "
            "the time. "
            "In general, an operation does F flops and moves B bytes, and each rate sets its own floor. "
            "So the runtime floor is the larger one, because the hardware can overlap the two but can't beat either."
        ) as t:
            self.play(FadeIn(given), FadeIn(bc.labels), run_time=0.7)
            t.wait_until("At 1935")
            self.play(GrowFromEdge(bc.bars[0], LEFT), FadeIn(bc.values[0]), run_time=1.0)
            t.wait_until("At 312")
            self.play(GrowFromEdge(bc.bars[1], LEFT), FadeIn(bc.values[1]), run_time=0.6)
            t.wait_until("The operation can't")
            self.play(FadeIn(idle), run_time=0.6)
            t.wait_until("In general")
            self.play(FadeIn(f1), run_time=0.8)
            t.wait_until("So the runtime")
            self.play(FadeIn(f2), Create(box), run_time=0.9)

    def ridge_point(self):
        self.slide("ridge-point", "The ridge point is C divided by β",
                   book_section="s-the-two-rates-and-the-ridge-point")
        d = kit.Derivation(
            [
                r"I &= F / B",
                r"P = F / t &= \min\!\left(C,\ \beta I\right)",
                r"\beta I^{\ast} &= C",
            ],
            size=th.SIZE_MATH,
            buff=0.4,
        )
        result = kit.formula(r"I^{\ast} = \frac{C}{\beta}", r"= \frac{312 \times 10^{12}}{1.935 \times 10^{12}}",
                             r"= 161.2\ \text{flops per byte}", size=th.SIZE_MATH)
        result[0].set_color(INTENSITY)
        box = SurroundingRectangle(result, color=self.color, buff=0.18, corner_radius=0.08, stroke_width=3)
        note = kit.callout("key", r"Below the ridge point, the arithmetic finishes before the next bytes arrive, so "
                                  r"adding compute does nothing.", width=10.5, size=th.SIZE_SMALL)
        col = VGroup(d, VGroup(result, box), note).arrange(DOWN, buff=0.5)
        self.place(col)
        with self.narrate(
            "Define arithmetic intensity, I, as flops divided by bytes. "
            "Divide the flops by the floor, and you get the best throughput an operation can reach. "
            "It's the smaller of the compute peak, and bandwidth times intensity. "
            "The ridge point is where they meet. "
            "Divide 312 TFLOP/s by 1935 GB/s, and you get 161.2 flops per byte. "
            "Below it, the arithmetic finishes before the next bytes arrive, so adding compute does nothing."
        ) as t:
            self.play(d.show(0), run_time=0.8)
            t.wait_until("Divide the flops")
            self.play(d.show(1), run_time=0.9)
            t.wait_until("The ridge point is where")
            self.play(d.show(2), run_time=0.8)
            t.wait_until("Divide 312")
            self.play(FadeIn(result), Create(box), run_time=1.0)
            t.wait_until("Below it")
            self.play(FadeIn(note, shift=UP * 0.1), run_time=0.7)

    def ridge_moves(self):
        self.slide("ridge-moves", "The ridge point moves with the bandwidth",
                   book_section="s-the-two-rates-and-the-ridge-point")
        roof = Roof(width=7.2, height=4.2, x_range=(0, 4), y_range=(0, 3))
        tab = kit.table(
            [
                ["2039 GB/s, SXM4 rated", "153.0"],
                ["1935 GB/s, PCIe rated", "161.2"],
                ["1275 GB/s, measured copy", "244.7"],
            ],
            header=["Bandwidth", "Ridge"],
            size=th.SIZE_SMALL,
            align=["l", "r"],
            row_gap=0.32,
        )
        layout = VGroup(roof, tab).arrange(RIGHT, buff=0.6)
        self.place(layout)
        bw = ValueTracker(BW)
        ceil = always_redraw(lambda: roof.ceiling(bw=bw.get_value()))
        slope = always_redraw(lambda: Line(roof.p(1, bw.get_value()), roof.p(PEAK / bw.get_value(), PEAK),
                                           color=MEMORY, stroke_width=th.STROKE_CURVE))
        rdot = always_redraw(lambda: Dot(roof.p(PEAK / bw.get_value(), PEAK), radius=0.1, color=th.INK))
        rdrop = always_redraw(lambda: roof.drop(PEAK / bw.get_value(), bw=bw.get_value()))
        ghost = VGroup(Line(roof.p(1, BW), roof.p(RIDGE, PEAK), color=MEMORY, stroke_width=2).set_opacity(0.35))
        band = kit.mark(row(tab, 3), color=th.tint(MEMORY, 0.18), buff=0.1)
        with self.narrate(
            "The ridge point belongs to the pair of rates, not to the card alone. "
            "The SXM4 module is rated at 2039 GB/s, which puts the ridge at 153. "
            "This course's PCIe card is rated at 1935, for 161.2. "
            "A plain copy really reaches 1275 GB/s on this card. "
            "The slope drops, and the ridge moves right, to 244.7. "
            "So even more of the model sits on the memory side than the headline suggests."
        ) as t:
            self.play(FadeIn(roof), FadeIn(tab.rules), FadeIn(row(tab, 0)), run_time=0.7)
            self.add(slope, ceil, rdot, rdrop)
            t.wait_until("The SXM4 module")
            self.play(bw.animate.set_value(2.039), FadeIn(row(tab, 1)), run_time=1.0)
            t.wait_until("This course's PCIe")
            self.play(bw.animate.set_value(BW), FadeIn(row(tab, 2)), run_time=1.0)
            t.wait_until("The slope drops")
            self.add(ghost)
            self.play(bw.animate.set_value(1.275), FadeIn(row(tab, 3)), run_time=1.6)
            t.wait_until("So even more")
            self.play(FadeIn(band), run_time=0.6)

    # Why measured bandwidth is 66% of rated ----------------------------------

    def count_both(self):
        self.slide("count-both", "A copy of N bytes moves 2N", book_section="s-why-measured-bandwidth-is-66-of-rated")
        src = kit.grid(1, 8, cell=(0.42, 0.42))
        dst = kit.grid(1, 8, cell=(0.42, 0.42))
        src.paint(src.block(0, 1, 0, 8), MEMORY, tint=0.3)
        src_t = kit.label("source", size=th.SIZE_LABEL, color=th.INK_3)
        dst_t = kit.label("destination", size=th.SIZE_LABEL, color=th.INK_3)
        sm = kit.box("SMs", color=th.INK_3, width=1.6, height=0.9)
        left = VGroup(src_t, src).arrange(DOWN, buff=0.15)
        right = VGroup(dst_t, dst).arrange(DOWN, buff=0.15)
        fig = VGroup(left, sm, right).arrange(RIGHT, buff=1.6)
        sm.align_to(src, DOWN).shift(DOWN * 0.24)
        tab = kit.table(
            [
                ["Copy", "read $N$, write $N$", "$2N$", "1275 GB/s"],
                ["Vector add", "read $2N$, write $N$", "$3N$", "1304 GB/s"],
            ],
            header=["Kernel", "Traffic", "Bytes counted", "Measured"],
            size=th.SIZE_SMALL,
            align=["l", "l", "r", "r"],
        )
        wrong = kit.prose(r"``4 GB in 6.3 ms, so 635 GB/s'' counts only $N$, and halves the result.",
                          size=th.SIZE_SMALL, color=th.BAD, width=None)
        col = VGroup(fig, wrong, tab).arrange(DOWN, buff=0.6)
        self.place(col)
        rd = kit.arrow(src, sm, text="read N", color=MEMORY)
        wr = kit.arrow(sm, dst, text="write N", color=MEMORY)
        with self.narrate(
            "Start with the simplest bandwidth-bound kernel there is: a device-to-device copy. "
            "Copying N bytes moves 2N bytes across the memory bus. "
            "N bytes are read from the source, and N bytes are written to the destination. "
            "The 1275 GB/s figure counts both. "
            "Count only N, and you halve your own result, and conclude the GPU is broken. "
            "The hand-written vector add reads two arrays and writes one, so it moves 3N bytes. "
            "Counted that way, it measures 1304 GB/s, the same as the copy."
        ) as t:
            self.play(FadeIn(left), FadeIn(sm), FadeIn(right), run_time=0.7)
            t.wait_until("N bytes are read")
            self.play(GrowArrow(rd.line), FadeIn(rd.label), run_time=0.7)
            self.play(GrowArrow(wr.line), FadeIn(wr.label), dst.fill(dst.block(0, 1, 0, 8), MEMORY, tint=0.3),
                      run_time=0.9)
            t.wait_until("The 1275")
            self.play(FadeIn(tab.rules), FadeIn(row(tab, 0)), FadeIn(row(tab, 1)), run_time=0.7)
            t.wait_until("Count only N")
            self.play(FadeIn(wrong), run_time=0.7)
            t.wait_until("The hand-written")
            self.play(FadeIn(VGroup(*tab.cells[2][:3])), run_time=0.7)
            t.wait_until("Counted that way")
            self.play(FadeIn(tab.cells[2][3]), run_time=0.6)

    def copy_ceiling(self):
        self.slide("copy-ceiling", "Judge a kernel against the copy, not the spec sheet",
                   book_section="s-why-measured-bandwidth-is-66-of-rated")
        bc = kit.bar_compare(
            [("Rated pin rate", 1935, th.INK_4, "1935 GB/s"),
             ("Plain copy", 1275, MEMORY, "1275 GB/s  ·  66% of rated"),
             ("Fused RMSNorm", 945, self.color, "945 GB/s  ·  74% of copy, 49% of rated")],
            width=5.2, bar_height=0.48, gap=0.3,
        )
        losses = kit.bullets(
            [r"DRAM refresh", r"Row activation and precharge", r"Bus turnaround between reads and writes"],
            size=th.SIZE_SMALL, width=6.2, buff=0.2,
        )
        lhead = kit.label("What the rating ignores", size=th.SIZE_SMALL, color=th.INK, weight=SEMIBOLD)
        left = VGroup(lhead, losses).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        key = kit.callout("key", r"At 85\% of the copy ceiling, a memory-bound kernel is finished. At 30\%, it has a "
                                 r"real problem.", title="When to stop", width=5.6, size=th.SIZE_SMALL)
        lower = VGroup(left, key).arrange(RIGHT, buff=0.6, aligned_edge=UP)
        col = VGroup(bc, lower).arrange(DOWN, buff=0.6)
        self.place(col)
        with self.narrate(
            "Even a perfect copy reaches only 66 percent of the rating. "
            "The rating is a pin rate: memory clock times bus width, assuming the bus never idles. "
            "A real access stream pays for DRAM refresh, for row activation and precharge, and for turning the bus "
            "around between reads and writes. "
            "So measure bandwidth-bound kernels against 1275 GB/s. "
            "The fused RMSNorm from chapter 13 reaches 945 GB/s. "
            "That's 74 percent of the copy ceiling, but only 49 percent of the rating. "
            "At 85 percent of the ceiling, a memory-bound kernel is finished. "
            "Far below that, it has a real problem."
        ) as t:
            self.play(FadeIn(bc.labels[:2]), GrowFromEdge(bc.bars[0], LEFT), FadeIn(bc.values[0]), run_time=0.9)
            self.play(GrowFromEdge(bc.bars[1], LEFT), FadeIn(bc.values[1]), run_time=0.8)
            t.wait_until("A real access")
            self.play(FadeIn(lhead), *[FadeIn(r, shift=RIGHT * 0.1) for r in losses], run_time=0.9)
            t.wait_until("The fused RMSNorm")
            self.play(FadeIn(bc.labels[2]), GrowFromEdge(bc.bars[2], LEFT), run_time=0.8)
            t.wait_until("That's 74 percent")
            self.play(FadeIn(bc.values[2]), run_time=0.6)
            t.wait_until("At 85 percent")
            self.play(FadeIn(key, shift=UP * 0.1), run_time=0.7)

    # The MLP -----------------------------------------------------------------

    def mlp_intensity(self):
        self.slide("mlp-intensity", "Every token in the pass shares one weight read", book_section="s-the-mlp")
        mats = VGroup(*[kit.box(n, sublabel="5120 × 17408" if n != "down" else "17408 × 5120", color=WEIGHTS,
                                width=2.5, height=1.1, size=th.SIZE_SMALL) for n in ("gate", "up", "down")])
        mats.arrange(DOWN, buff=0.2)
        w_lab = kit.label("534.8 MB, read once per pass", size=th.SIZE_LABEL, color=WEIGHTS, weight=MEDIUM)
        w_lab.next_to(mats, DOWN, buff=0.2)
        toks = VGroup(*[RoundedRectangle(width=1.4, height=0.3, corner_radius=0.06, fill_color=th.tint(self.color, 0.4),
                                         fill_opacity=1, stroke_color=self.color, stroke_width=1.5)
                        for _ in range(8)]).arrange(DOWN, buff=0.12)
        t_lab = kit.label("T tokens", size=th.SIZE_LABEL, color=self.color, weight=MEDIUM).next_to(toks, UP, buff=0.15)
        fig = VGroup(VGroup(t_lab, toks), VGroup(mats, w_lab)).arrange(RIGHT, buff=1.2)
        d = kit.Derivation(
            [
                r"F &= 2Thi + 2Thi + 2Tih = 6Thi",
                r"B &= 3hi\,e + 2The",
                r"I(T) &= \frac{6Thi}{3hi\,e + 2The} = \frac{T}{1 + T/26112}",
            ],
            size=th.SIZE_MATH_SMALL,
            buff=0.45,
        )
        approx = kit.prose(r"$I(T) \approx T$: the tokens in the pass.", size=th.SIZE_SMALL, color=INTENSITY,
                           width=None)
        right = VGroup(d, approx).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        layout = VGroup(fig, right).arrange(RIGHT, buff=0.8)
        self.place(layout)
        links = VGroup(*[Line(tk.get_right(), mats.get_left(), color=th.INK_4, stroke_width=1.5) for tk in toks])
        with self.narrate(
            "Now derive where each operation sits, starting with the MLP. "
            "It's a SwiGLU block: gate and up side by side, then down. "
            "With T tokens in the pass, the three matmuls cost 6 T h i flops. "
            "The weights are read once per forward pass, no matter how many tokens are in it. "
            "That's 534.8 MB. "
            "The activations add a little more. "
            "Divide, and intensity is T over 1 plus T over 26,112. "
            "Each token does 2 flops per 2-byte weight, and every token shares one read. "
            "So intensity is roughly T."
        ) as t:
            self.play(FadeIn(mats), run_time=0.8)
            t.wait_until("With T tokens")
            self.play(FadeIn(t_lab), FadeIn(toks[0]), Create(links[0]), d.show(0), run_time=1.0)
            t.wait_until("The weights are read")
            self.play(LaggedStart(*[AnimationGroup(FadeIn(tk), Create(ln)) for tk, ln in zip(toks[1:], links[1:])],
                                  lag_ratio=0.3), run_time=1.8)
            t.wait_until("That's 534.8")
            self.play(FadeIn(w_lab), Indicate(mats, color=WEIGHTS, scale_factor=1.03), run_time=0.9)
            t.wait_until("The activations add")
            self.play(d.show(1), run_time=0.8)
            t.wait_until("Divide, and")
            self.play(d.show(2), run_time=0.9)
            t.wait_until("So intensity is")
            self.play(FadeIn(approx), run_time=0.6)

    def mlp_limits(self):
        self.slide("mlp-limits", "Intensity approaches T from below", book_section="s-the-mlp")
        ax = kit.axes([0, 6, 1], [0, 5, 1], width=6.6, height=4.3)
        nums = VGroup()
        for k in range(0, 7):
            nums.add(kit.math_label(tick_tex(k), size=th.SIZE_LABEL, color=th.INK_3).next_to(ax.c2p(k, 0), DOWN,
                                                                                            buff=0.14))
        for k in range(0, 6):
            nums.add(kit.math_label(tick_tex(k), size=th.SIZE_LABEL, color=th.INK_3).next_to(ax.c2p(0, k), LEFT,
                                                                                            buff=0.14))
        xt = kit.label("tokens in the pass, T", size=th.SIZE_LABEL, color=th.INK_3)
        xt.next_to(nums[:7], DOWN, buff=0.15).align_to(ax.c2p(6, 0), RIGHT)
        yt = kit.label("flops per byte", size=th.SIZE_LABEL, color=th.INK_3)
        yt.next_to(ax.c2p(0, 5), UP, buff=0.15).align_to(ax.c2p(0, 0), LEFT)
        axes_g = VGroup(ax, nums, xt, yt)
        tab = kit.table(
            [["1", "0.004\\%"], ["16", "0.06\\%"], ["2048", "7.8\\%"]],
            header=["$T$", "Error of $I \\approx T$"],
            size=th.SIZE_SMALL, align=["r", "r"],
        )
        spill = kit.prose(r"At $T = 4096$, each $(T, i)$ intermediate is 143 MB, 3.6 times the L2.",
                          size=th.SIZE_SMALL, width=4.6)
        side = VGroup(tab, spill).arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        layout = VGroup(axes_g, side).arrange(RIGHT, buff=0.6)
        self.place(layout)

        def pt(tv, iv):
            return ax.c2p(math.log10(tv), math.log10(iv))

        def curve(f, color, dashed=False):
            pts = [pt(10 ** x, f(10 ** x)) for x in np.linspace(0, 6, 121)]
            c = VMobject(color=color, stroke_width=th.STROKE_CURVE).set_points_smoothly(pts)
            return DashedVMobject(c, num_dashes=60) if dashed else c

        ident = DashedLine(pt(1, 1), pt(1e5, 1e5), color=th.INK_4, stroke_width=th.STROKE_AXIS, dash_length=0.1)
        ident_t = kit.math_label(r"I = T", size=th.SIZE_LABEL, color=th.INK_3).next_to(pt(1e5, 1e5), RIGHT, buff=0.1)
        exact = curve(mlp_i, INTENSITY)
        asym = DashedLine(pt(1, 26112), pt(1e6, 26112), color=ASYM, stroke_width=th.STROKE_AXIS, dash_length=0.1)
        asym_t = kit.label("26,112", size=th.SIZE_LABEL, color=ASYM, weight=MEDIUM).next_to(pt(1, 26112), UR, buff=0.1)

        def spilled(tv):
            return 6 * tv * H_DIM * I_DIM / (3 * H_DIM * I_DIM * 2 + 2 * tv * H_DIM * 2 + 4 * tv * I_DIM * 2)

        sp = curve(spilled, th.BAD).set_z_index(-1)
        d1 = Dot(pt(4096, mlp_i(4096)), radius=0.08, color=INTENSITY)
        d2 = Dot(pt(4096, spilled(4096)), radius=0.08, color=th.BAD)
        t1 = kit.label("3540", size=th.SIZE_LABEL, color=INTENSITY, weight=MEDIUM).next_to(d1, UL, buff=0.08)
        t2 = kit.label("1842, spilled", size=th.SIZE_LABEL, color=th.BAD, weight=MEDIUM).next_to(d2, DR, buff=0.1)
        with self.narrate(
            "The approximation is good while T is far below 26,112. "
            "At 16 tokens, that's 0.06 percent. "
            "At 2048, it's 7.8 percent. "
            "Past about 26,112 tokens, the activations move more bytes than the weights, and intensity levels "
            "off. "
            "The derivation assumes the gate and up outputs never reach HBM. "
            "At 4096 tokens, each one is 143 MB, 3.6 times the L2, so they spill. "
            "Counting those bytes drops intensity from 3540 to 1842. "
            "It's still compute bound, but the floor changes."
        ) as t:
            self.play(Create(ax), FadeIn(nums), FadeIn(xt), FadeIn(yt), run_time=0.8)
            self.play(Create(ident), FadeIn(ident_t), run_time=0.7)
            t.wait_until("At 16 tokens, that")
            self.play(Create(exact), FadeIn(tab.rules), FadeIn(row(tab, 0)), FadeIn(row(tab, 1)), run_time=1.2)
            t.wait_until("At 2048, it's")
            self.play(FadeIn(row(tab, 2)), FadeIn(row(tab, 3)), run_time=0.6)
            t.wait_until("Past about")
            self.play(Create(asym), FadeIn(asym_t), run_time=0.8)
            t.wait_until("At 4096 tokens")
            self.play(FadeIn(spill), run_time=0.7)
            t.wait_until("Counting those bytes")
            self.play(Create(sp), FadeIn(d1), FadeIn(t1), run_time=1.0)
            self.play(FadeIn(d2), FadeIn(t2), run_time=0.6)

    def decode_vs_prefill(self):
        self.slide("decode-vs-prefill", "161 tokens in one pass make the GPU work", book_section="s-the-mlp")
        roof = Roof(width=6.6, height=4.6, x_range=(-1, 4), y_range=(-1, 3))
        tab = kit.table(
            [
                ["1, decode", "1.0", "memory, $161\\times$"],
                ["16", "16.0", "memory, $10\\times$"],
                ["161", "160", "at the ridge"],
                ["512", "502", "compute"],
                ["2048, prefill", "1899", "compute"],
            ],
            header=["Tokens", "$I$", "Bound by"],
            size=th.SIZE_SMALL,
            align=["l", "r", "l"],
            col_gap=0.4,
            row_gap=0.24,
        )
        key = kit.prose(r"One token: 0.62\% of peak.", size=th.SIZE_SMALL, color=self.color, width=None)
        side = VGroup(tab, key).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        layout = VGroup(roof, side).arrange(RIGHT, buff=0.5)
        self.place(layout)
        roofs = VGroup(roof.slope(), roof.ceiling())
        logt = ValueTracker(0.0)
        mover = always_redraw(lambda: roof.dot(mlp_i(10 ** logt.get_value()), self.color, radius=0.11))
        stops = [(1, "1", DR), (16, "16", DR), (161, "161", UL), (512, "512", UP), (2048, "2048", DOWN)]
        marks = []
        for tv, txt, direc in stops:
            dt = roof.dot(mlp_i(tv), self.color, radius=0.07)
            lab = kit.label(txt, size=th.SIZE_LABEL, color=th.INK, weight=MEDIUM).next_to(dt, direc, buff=0.1)
            marks.append(VGroup(dt, lab))
        with self.narrate(
            "Now slide the MLP along the roofline. "
            "At one token, which is decode at batch 1, intensity is 1. "
            "It's memory bound by a factor of 161. "
            "At 16 tokens, it's still memory bound, by 10. "
            "At 161 tokens, it reaches the ridge. "
            "At 512, it's compute bound. "
            "A 2048-token prefill sits at 1899, nearly 12 times past the ridge. "
            "So decode at batch 1 uses 0.62 percent of the tensor cores' peak. "
            "Getting 161 tokens into one pass is what makes the GPU work, and that's why batching exists."
        ) as t:
            self.play(FadeIn(roof), FadeIn(roofs), FadeIn(tab.rules), FadeIn(row(tab, 0)), run_time=0.7)
            t.wait_until("At one token")
            self.add(mover)
            self.play(FadeIn(marks[0]), FadeIn(row(tab, 1)), run_time=0.6)
            for (tv, _, _), mk, r, cue in zip(stops[1:], marks[1:], (2, 3, 4, 5),
                                              ["At 16 tokens", "At 161 tokens", "At 512", "A 2048-token"]):
                t.wait_until(cue)
                self.play(logt.animate.set_value(math.log10(tv)), run_time=1.0)
                self.play(FadeIn(mk), FadeIn(row(tab, r)), run_time=0.5)
            t.wait_until("So decode at batch 1")
            self.play(FadeIn(key, shift=UP * 0.1), Indicate(marks[0], color=self.color), run_time=0.9)

    # Attention during decode -------------------------------------------------

    def decode_attention(self):
        self.slide("decode-attention", "Decode attention's intensity is the group size, 6",
                   book_section="s-attention-during-decode")
        given = kit.prose(r"One step, one layer, one sequence at context $L$: $H = 24$, $H_{kv} = 4$, $d = 256$.",
                          size=th.SIZE_SMALL, width=None)
        fF = kit.formula(r"F = 2HLd + 2HLd = 4HLd", size=th.SIZE_MATH_SMALL)
        fB = kit.formula(r"B = 2 \cdot H_{kv} \cdot L \cdot d \cdot e", size=th.SIZE_MATH_SMALL)
        fI = kit.formula(r"I = \frac{4HLd}{2H_{kv}Lde}", size=th.SIZE_MATH)
        fI2 = kit.formula(r"= \frac{2H}{e\,H_{kv}} = \frac{24}{4} =", r"6", size=th.SIZE_MATH)
        fI2[1].set_color(self.color)
        irow = VGroup(fI, fI2).arrange(RIGHT, buff=0.2)
        maths = VGroup(fF, fB, irow).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        ex_head = kit.label("One sequence at 32k, 16 layers: 2 GiB read", size=th.SIZE_SMALL, color=th.INK,
                            weight=MEDIUM)
        bc = kit.bar_compare([("Memory floor", 1.11, MEMORY, "1.11 ms"), ("Compute floor", 0.0413, COMPUTE, "41.3 µs")],
                             width=3.6, bar_height=0.42, gap=0.25, size=th.SIZE_SMALL)
        ratio = kit.label("ratio 26.9 = 161.2 / 6", size=th.SIZE_SMALL, color=self.color, weight=MEDIUM)
        right = VGroup(ex_head, bc, ratio).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        body = VGroup(maths, right).arrange(RIGHT, buff=0.9)
        col = VGroup(given, body).arrange(DOWN, buff=0.6)
        self.place(col)
        glyphs = fI[0]
        bar = max(glyphs, key=lambda g: g.width)
        num = sorted([g for g in glyphs if g.get_y() > bar.get_y() + 0.05 and g.get_x() > bar.get_left()[0]],
                     key=lambda g: g.get_x())
        den = sorted([g for g in glyphs if g.get_y() < bar.get_y() - 0.05], key=lambda g: g.get_x())
        cancel = VGroup(*[Line(g.get_corner(DL) + DL * 0.03, g.get_corner(UR) + UR * 0.03, color=th.BAD,
                               stroke_width=3) for g in (num[2], num[3], den[-3], den[-2])])
        with self.narrate(
            "Attention during decode is different: each sequence reads its own KV cache, so nothing is shared. "
            "The query is a single token, so the scores and the values each cost 2 H L d flops. "
            "The step reads keys and values for the 4 KV heads only. "
            "Divide, and both L and d cancel. "
            "With 2-byte elements, intensity is 24 / 4, which is 6: exactly the GQA group size. "
            "At 32k context across 16 layers, the step reads 2 GiB. "
            "Its memory floor is 1.11 ms, and its compute floor is 41.3 µs. "
            "That's a factor of 27, which is 161 divided by 6."
        ) as t:
            t.wait_until("Attention during decode")
            self.play(FadeIn(given), run_time=0.7)
            t.wait_until("The query is")
            self.play(FadeIn(fF), run_time=0.8)
            t.wait_until("The step reads")
            self.play(FadeIn(fB), run_time=0.8)
            t.wait_until("Divide, and both")
            self.play(FadeIn(fI), run_time=0.7)
            self.play(LaggedStart(*[Create(c) for c in cancel], lag_ratio=0.25), run_time=1.0)
            t.wait_until("With 2-byte")
            self.play(FadeIn(fI2), run_time=0.8)
            t.wait_until("At 32k context")
            self.play(FadeIn(ex_head), FadeIn(bc.labels), run_time=0.6)
            t.wait_until("Its memory floor")
            self.play(*bc.grow(run_time=1.0), FadeIn(bc.values), run_time=1.0)
            t.wait_until("That's a factor")
            self.play(FadeIn(ratio), run_time=0.6)

    def attention_levers(self):
        self.slide("attention-levers", "Only the dtype and the KV heads move it",
                   book_section="s-attention-during-decode")
        roof = Roof(width=7.0, height=4.3, x_range=(-1, 4), y_range=(-1, 3))
        notes = kit.bullets(
            [
                r"Context 1k or 256k: 6",
                r"Batch 64: 6",
                r"int8 cache, $e = 1$: 12",
                r"24 KV heads: 1",
            ],
            size=th.SIZE_BODY, width=4.8, buff=0.4,
        )
        layout = VGroup(roof, notes).arrange(RIGHT, buff=0.6)
        self.place(layout)
        roofs = VGroup(roof.slope(), roof.ceiling())
        r_drop = VGroup()
        main = roof.dot(6, self.color, radius=0.11)
        m_lab = kit.label("GQA, bf16: 6", size=th.SIZE_LABEL, color=self.color, weight=MEDIUM)
        m_lab.next_to(main, RIGHT, buff=0.25).shift(DOWN * 0.2)
        ctx = kit.label("L = 1k", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        ctx_slot = roof.p(10 ** 0.2, 300)
        ctx.move_to(ctx_slot)
        ctx2 = kit.label("L = 256k", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM).move_to(ctx_slot)
        ctx3 = kit.label("batch 64", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM).move_to(ctx_slot)
        q = roof.dot(12, th.GOOD, radius=0.09)
        q_lab = kit.label("int8: 12", size=th.SIZE_LABEL, color=th.GOOD, weight=MEDIUM).next_to(q, LEFT, buff=0.25)
        m = roof.dot(1, th.BAD, radius=0.09)
        mha_lab = kit.label("24 KV heads: 1", size=th.SIZE_LABEL, color=th.BAD, weight=MEDIUM).next_to(m, RIGHT, buff=0.25).shift(DOWN * 0.2)
        with self.narrate(
            "That cancellation has three consequences. "
            "First, context length doesn't matter. "
            "At 1k context or at 256k, intensity is 6. "
            "Second, batch size doesn't matter. "
            "Each sequence brings its own cache, so flops and bytes grow together, and the ratio stays put. "
            "Third, the cache's number format is the only lever in the formula. "
            "Quantize the cache to int8, and intensity doubles to 12. "
            "Give every query head its own KV head, and it drops to 1. "
            "Decode attention is permanently memory bound, so every fix reads fewer bytes."
        ) as t:
            self.play(FadeIn(roof), FadeIn(roofs), FadeIn(r_drop), FadeIn(main), FadeIn(m_lab), run_time=0.8)
            t.wait_until("At 1k context")
            self.play(FadeIn(ctx), FadeIn(notes[0], shift=RIGHT * 0.1), run_time=0.6)
            self.play(Transform(ctx, ctx2), Indicate(main, color=self.color, scale_factor=1.4), run_time=1.0)
            t.wait_until("Second, batch")
            self.play(Transform(ctx, ctx3), Indicate(main, color=self.color, scale_factor=1.4),
                      FadeIn(notes[1], shift=RIGHT * 0.1), run_time=1.0)
            t.wait_until("Quantize the cache")
            self.play(FadeOut(ctx), TransformFromCopy(main, q), FadeIn(q_lab), FadeIn(notes[2], shift=RIGHT * 0.1),
                      run_time=1.0)
            t.wait_until("Give every query")
            self.play(TransformFromCopy(main, m), FadeIn(mha_lab), FadeIn(notes[3], shift=RIGHT * 0.1), run_time=1.0)
            t.wait_until("Decode attention is")
            self.play(Indicate(roofs[0], color=MEMORY, scale_factor=1.02), run_time=0.9)

    # RMSNorm and the embedding -----------------------------------------------

    def rmsnorm(self):
        self.slide("rmsnorm", "RMSNorm's cost is pure memory traffic", book_section="s-rmsnorm")
        f = kit.formula(r"I = \frac{4Th}{2The} = \frac{2}{e} = 1\ \text{flop per byte}", size=th.SIZE_MATH_SMALL)
        cap = kit.prose(r"At $T = 4096$ rows of $h = 5120$: 83.9 MB moved, 83.9 MFLOP.", size=th.SIZE_SMALL,
                        width=None, color=th.INK_3)
        bc = kit.bar_compare(
            [("Compute floor", 0.27, COMPUTE, "0.27 µs"),
             ("Memory floor at 1935 GB/s", 43.4, MEMORY, "43.4 µs"),
             ("Memory floor at 1275 GB/s", 65.8, MEMORY, "65.8 µs"),
             ("Measured, Triton", 88.8, self.color, "88.8 µs")],
            width=5.6, bar_height=0.44, gap=0.26,
        )
        key = kit.callout("key", r"The only lever is the 83.9 MB: stop writing the output to HBM at all, by fusing "
                                 r"the norm with its neighbors.", width=10.8, size=th.SIZE_SMALL)
        col = VGroup(VGroup(f, cap).arrange(DOWN, buff=0.25), bc, key).arrange(DOWN, buff=0.4)
        self.place(col)
        with self.narrate(
            "RMSNorm rescales each token's vector, then multiplies by a learned gain. "
            "Per row of h values, it does about 4 h flops, and it reads and writes h values of 2 bytes each. "
            "Divide, and both T and h cancel: intensity is 1, at every size. "
            "At 4096 rows of 5120, it moves 83.9 MB. "
            "The memory floor is 43.4 µs at the rated bandwidth. "
            "The compute floor is 0.27 µs, 160 times smaller. "
            "The Triton kernel runs in 88.8 µs. "
            "The only lever is to stop writing the output to HBM at all, by fusing the norm with its neighbors."
        ) as t:
            t.wait_until("Divide, and both")
            self.play(FadeIn(f), run_time=0.8)
            t.wait_until("At 4096 rows")
            self.play(FadeIn(cap), FadeIn(bc.labels), run_time=0.7)
            t.wait_until("The memory floor is")
            self.play(GrowFromEdge(bc.bars[1], LEFT), FadeIn(bc.values[1]), run_time=0.8)
            self.play(GrowFromEdge(bc.bars[2], LEFT), FadeIn(bc.values[2]), run_time=0.8)
            t.wait_until("The compute floor")
            self.play(GrowFromEdge(bc.bars[0], LEFT), FadeIn(bc.values[0]), run_time=0.6)
            t.wait_until("The Triton kernel")
            self.play(GrowFromEdge(bc.bars[3], LEFT), FadeIn(bc.values[3]), run_time=0.8)
            t.wait_until("The only lever")
            self.play(FadeIn(key, shift=UP * 0.1), run_time=0.7)

    def embedding(self):
        self.slide("embedding", "The embedding lookup does no arithmetic", book_section="s-the-embedding-lookup")
        ids = ["17", "3", "17", "9"]
        idg = kit.grid(4, 1, cell=(0.7, 0.42))
        # The ID text sits outside the grid, because animating a cell's fill
        # re-adds its square above the cell's own text.
        id_text = VGroup(*[kit.label(v, size=th.SIZE_LABEL, color=th.INK, font=th.MONO).move_to(idg.cell(k, 0))
                           for k, v in enumerate(ids)])
        idg.add(id_text)
        id_t = kit.label("token IDs", size=th.SIZE_LABEL, color=th.INK_3).next_to(idg, UP, buff=0.15)
        table_rows = ["0", "3", "9", "17", "248,319"]
        tbl = kit.grid(5, 12, cell=(0.3, 0.42), row_labels=table_rows)
        tbl_t = kit.label("table: 248,320 rows, 2.54 GB", size=th.SIZE_LABEL, color=th.INK_3)
        tbl_t.next_to(tbl, UP, buff=0.15).align_to(tbl, RIGHT)
        out = kit.grid(4, 12, cell=(0.3, 0.42))
        out_t = kit.label("output, T rows", size=th.SIZE_LABEL, color=th.INK_3).next_to(out, UP, buff=0.15)
        fig = VGroup(VGroup(id_t, idg), VGroup(tbl_t, tbl), VGroup(out_t, out)).arrange(RIGHT, buff=0.9)
        facts = kit.table(
            [
                ["Intensity", "$0 / B = 0$"],
                ["2048-token prefill", "41.96 MB, 21.7 µs"],
                ["One row", "10,240 bytes: 80 cache lines"],
                ["Table vs 40 MB L2", "63 times larger"],
            ],
            size=th.SIZE_SMALL, align=["l", "l"], row_gap=0.22,
        )
        col = VGroup(fig, facts).arrange(DOWN, buff=0.6)
        self.place(col)
        which = {"3": 1, "9": 2, "17": 3}
        with self.narrate(
            "The embedding lookup is a gather: each token ID picks one row of the table. "
            "It computes nothing, so its intensity is 0. "
            "For a 2048-token prefill, it moves about 42 MB. "
            "That's 21.7 µs, against hundreds of milliseconds in the layers. "
            "Each row is 10,240 contiguous bytes, 80 whole cache lines, so the gather coalesces well. "
            "The table is 2.54 GB, 63 times the L2, so nothing stays resident between passes. "
            "The exception is a repeated token in one batch: its second read hits L2."
        ) as t:
            self.play(FadeIn(idg), FadeIn(id_t), FadeIn(tbl), FadeIn(tbl_t), FadeIn(out), FadeIn(out_t), run_time=0.8)
            for k in range(3):
                r = which[ids[k]]
                self.play(idg.fill(idg.cell(k, 0), self.color, tint=0.4),
                          tbl.fill(tbl.row(r), self.color, tint=0.4), run_time=0.35)
                self.play(out.fill(out.row(k), self.color, tint=0.4), run_time=0.35)
                self.bring_to_front(id_text)
            t.wait_until("It computes nothing")
            self.play(FadeIn(facts.rules), FadeIn(row(facts, 0)), run_time=0.6)
            t.wait_until("For a 2048-token")
            self.play(FadeIn(row(facts, 1)), run_time=0.6)
            t.wait_until("Each row is")
            self.play(FadeIn(row(facts, 2)), Indicate(tbl.row(3), color=self.color, scale_factor=1.05), run_time=0.9)
            t.wait_until("The table is")
            self.play(FadeIn(row(facts, 3)), run_time=0.6)
            t.wait_until("The exception")
            self.play(idg.fill(idg.cell(3, 0), th.GOOD, tint=0.4), tbl.fill(tbl.row(2), th.GOOD, tint=0.4),
                      run_time=0.4)
            self.play(out.fill(out.row(3), th.GOOD, tint=0.4), run_time=0.4)
            self.bring_to_front(id_text)
            self.play(Circumscribe(VGroup(idg.cell(0, 0), idg.cell(2, 0)), color=th.GOOD, buff=0.05), run_time=1.0)

    # Kernel launch overhead and idle SMs ------------------------------------

    def launch_overhead(self):
        self.slide("launch-overhead", "Starting the kernels costs more than their arithmetic",
                   book_section="s-kernel-launch-overhead")

        def strip(n, gap_each):
            parts = VGroup()
            x = 0.0
            for k in range(n):
                if gap_each or k == 0:
                    g = Rectangle(width=0.42, height=0.45, fill_color=th.tint(th.INK_4, 0.4), fill_opacity=1,
                                  stroke_width=0)
                    g.move_to([x + 0.21, 0, 0])
                    parts.add(g)
                    x += 0.42
                kr = Rectangle(width=0.2, height=0.45, fill_color=self.color, fill_opacity=0.85, stroke_width=0)
                kr.move_to([x + 0.1, 0, 0])
                parts.add(kr)
                x += 0.2
            return parts

        s1 = strip(14, True)
        s2 = strip(14, False)
        n1 = kit.label("one launch per kernel", size=th.SIZE_LABEL, color=th.INK_2)
        n2 = kit.label("CUDA graph: one launch", size=th.SIZE_LABEL, color=th.INK_2)
        key = VGroup(swatch(th.tint(th.INK_4, 0.4), "launch, 5 to 10 µs"), swatch(self.color, "kernel work")).arrange(
            RIGHT, buff=0.5)
        names_w = max(n1.width, n2.width)
        for n, s_ in ((n1, s1), (n2, s2)):
            s_.next_to(n, RIGHT, buff=0.3 + names_w - n.width)
        r1, r2 = VGroup(n1, s1), VGroup(n2, s2)
        timeline = VGroup(key, r1, r2).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        bc = kit.bar_compare(
            [("Arithmetic floor", 0.172, COMPUTE, "0.172 ms"),
             ("400 launches at 5 µs", 2.0, th.INK_3, "2.0 ms  ·  12 times"),
             ("400 launches at 10 µs", 4.0, th.INK_3, "4.0 ms  ·  23 times")],
            width=5.0, bar_height=0.42, gap=0.25,
        )
        mem = kit.label("Memory floor: 27.8 ms, so launches add 7 to 14%.", size=th.SIZE_SMALL, color=th.INK_3)
        col = VGroup(timeline, bc, mem).arrange(DOWN, buff=0.5)
        self.place(col)
        with self.narrate(
            "The roofline assumes the GPU is busy, and at batch 1 it often isn't. "
            "Launching a kernel costs 5 to 10 µs, and a forward pass is one stream of dependent kernels, so those "
            "costs add up in series. "
            "A decode step's arithmetic floor is 0.172 ms. "
            "It runs several hundred kernels; take 400. "
            "At 5 µs each, the launches cost 2 ms, 12 times the arithmetic floor. "
            "At 10 µs, they cost 4 ms, 23 times. "
            "Against the 27.8 ms memory floor, that's 7 to 14 percent. "
            "A CUDA graph captures the whole step and replays it as one launch. "
            "Production engines use graphs for decode."
        ) as t:
            t.wait_until("Launching a kernel")
            self.play(FadeIn(key), FadeIn(n1), run_time=0.5)
            self.play(LaggedStart(*[FadeIn(p) for p in s1], lag_ratio=0.12), run_time=2.4)
            t.wait_until("A decode step's")
            self.play(FadeIn(bc.labels), GrowFromEdge(bc.bars[0], LEFT), FadeIn(bc.values[0]), run_time=0.8)
            t.wait_until("At 5 µs each")
            self.play(GrowFromEdge(bc.bars[1], LEFT), FadeIn(bc.values[1]), run_time=0.8)
            t.wait_until("At 10 µs")
            self.play(GrowFromEdge(bc.bars[2], LEFT), FadeIn(bc.values[2]), run_time=0.8)
            t.wait_until("Against the 27.8")
            self.play(FadeIn(mem), run_time=0.6)
            t.wait_until("A CUDA graph")
            self.play(FadeIn(n2), TransformFromCopy(s1, s2), run_time=1.2)

    def idle_sms(self):
        self.slide("idle-sms", "Decode at batch 1 fills 1 of 108 SMs", book_section="s-idle-sms-and-occupancy")
        g1 = kit.grid(9, 12, cell=0.34, gap=0.04)
        g2 = kit.grid(9, 12, cell=0.34, gap=0.04)
        w1 = kit.label("wave 1", size=th.SIZE_LABEL, color=th.INK_3).next_to(g1, UP, buff=0.12)
        w2 = kit.label("wave 2", size=th.SIZE_LABEL, color=th.INK_3).next_to(g2, UP, buff=0.12)
        waves = VGroup(VGroup(w1, g1), VGroup(w2, g2)).arrange(RIGHT, buff=0.4)
        f = kit.formula(r"u(b) = \frac{b}{108 \cdot \lceil b / 108 \rceil}", size=th.SIZE_MATH_SMALL)
        tab = kit.table(
            [["8", "1", "7.4\\%"], ["108", "1", "100\\%"], ["109", "2", "50.5\\%"], ["216", "2", "100\\%"],
             ["4096", "38", "99.8\\%"], ["1, decode", "1", "0.93\\%"]],
            header=["Blocks", "Waves", "Busy"],
            size=th.SIZE_SMALL, align=["l", "r", "r"], row_gap=0.2, col_gap=0.45,
        )
        side = VGroup(f, tab).arrange(DOWN, buff=0.4)
        layout = VGroup(waves, side).arrange(RIGHT, buff=0.5)
        self.place(layout)
        cells1 = [g1.cell(r, c) for r in range(9) for c in range(12)]
        busy = th.tint(self.color, 0.55)

        def show(n1, n2):
            anims = []
            for k, c in enumerate(cells1):
                anims.append(c.square.animate.set_fill(busy if k < n1 else th.RAISE, opacity=1))
            for k in range(108):
                c = g2.cell(k // 12, k % 12)
                anims.append(c.square.animate.set_fill(busy if k < n2 else th.RAISE, opacity=1))
            return anims

        with self.narrate(
            "A kernel's threads come in blocks, and each block runs on one of 108 SMs. "
            "A kernel runs its blocks in waves of at most 108, and while a partial last wave finishes, the other "
            "SMs stand empty. "
            "With 8 blocks, 7.4 percent of the GPU is busy. "
            "With 108, every SM has work. "
            "Add one more block, and a second wave runs almost empty, so utilization halves, to 50.5 percent. "
            "RMSNorm with one block per row fills 99.8 percent of the GPU on a 4096-token prefill. "
            "Decode at batch 1 launches one block, and 107 SMs sit idle. "
            "The fix is more work, which is what batching is for."
        ) as t:
            self.play(FadeIn(waves[0]), run_time=0.6)
            t.wait_until("A kernel runs")
            self.play(FadeIn(f), FadeIn(tab.rules), FadeIn(row(tab, 0)), FadeIn(waves[1]), run_time=0.7)
            t.wait_until("With 8 blocks")
            self.play(*show(8, 0), FadeIn(row(tab, 1)), run_time=0.8)
            t.wait_until("With 108")
            self.play(*show(108, 0), FadeIn(row(tab, 2)), run_time=0.9)
            t.wait_until("Add one more")
            self.play(*show(108, 1), FadeIn(row(tab, 3)), FadeIn(row(tab, 4)), run_time=0.8)
            t.wait_until("RMSNorm with one block")
            self.play(FadeIn(row(tab, 5)), run_time=0.6)
            t.wait_until("Decode at batch 1")
            self.play(*show(1, 0), FadeIn(row(tab, 6)), run_time=0.9)
            self.play(Indicate(g1.cell(0, 0), color=self.color, scale_factor=1.6), run_time=0.8)

    def what_goes_wrong(self):
        self.slide("what-goes-wrong", "Four ways to misjudge a kernel", book_section="s-what-goes-wrong")
        cards = [
            ("Counting only reads", r"Report $N / t$ for a copy, and you halve the measured bandwidth."),
            ("Dropping the factor of 2", r"A matmul is $2mkn$ flops. Count $mkn$, and every intensity halves."),
            ("Charging weights per token", r"Weights are read once per pass. Multiply by $T$, and every pass "
                                           r"looks memory bound."),
            ("Reading all 24 KV heads", r"The cache holds $H_{kv} = 4$. Counting $H$ gives an intensity of 1, "
                                        r"not 6."),
        ]
        boxes = [kit.callout("warning", body, title=title, width=6.0, size=th.SIZE_SMALL) for title, body in cards]
        grid = VGroup(VGroup(boxes[0], boxes[1]).arrange(RIGHT, buff=0.4, aligned_edge=UP),
                      VGroup(boxes[2], boxes[3]).arrange(RIGHT, buff=0.4, aligned_edge=UP)).arrange(DOWN, buff=0.4)
        self.place(grid)
        with self.narrate(
            "Four mistakes account for most wrong verdicts. "
            "First, counting only reads, which halves a copy's measured bandwidth. "
            "Second, dropping the factor of 2 in a matmul, which halves every intensity. "
            "Third, charging the weights once per token, when they're read once per pass. "
            "Fourth, reading all 24 heads of KV, when the cache holds only 4."
        ) as t:
            for b, cue in zip(boxes, ["First, counting", "Second, dropping", "Third, charging", "Fourth, reading"]):
                t.wait_until(cue)
                self.play(FadeIn(b, shift=UP * 0.1), run_time=0.7)
