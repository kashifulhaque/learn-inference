"""Chapter 19, tensor parallelism.

Teaches content/chapters/19-tensor-parallelism.md in its order. Every number
below was recomputed in Python and matches the chapter (rounded as it rounds).

Geometry: d = 5120, d_ff = 17408, 64 layers (16 full attention, 48 linear),
24 query heads, 4 KV heads, d_h = 256; linear attention 48 value heads,
16 key heads, head width 128; vocabulary 248,320; bf16 (2 bytes).

Why a second GPU:
- Weights 26,895,979,520 x 2 = 53.79e9 B = 53.8 GB; half 26.9 GB.
- 19 GiB / 64 KiB = 19 x 2^30 / 65,536 = 311,296 tokens ("about 311,000");
  / 32 sequences = 9728 ("under 10,000").
- 26.9e9 B / 2^30 = 25.05 GiB freed per card ("another 25 GiB").
- Decode floor 53.79e9 / 1275e9 = 42.19 ms (42.2); half 21.09 ms (21.1).
- On-screen bars: 80 GB card; 19 GiB = 20.40 GB; 19 + 25 = 44 GiB = 47.2 GB.

Collectives demo: [1, 2, 3] + [4, 5, 6] = [5, 7, 9].

Ring all-reduce: per-rank bytes 2 (P - 1)/P x S; P = 2 gives exactly S;
P -> infinity gives 2S. Simulated with P = 4: after 3 reduce-scatter hops,
rank r holds chunk (r + 1) mod 4 summed over all 4 ranks; 3 more hops
gather the rest.

Naive versus paired: S_H / S_Y = 17408 / 5120 = 3.4; naive bytes
(3.4 + 2) = 5.4 units of (P - 1)/P S_Y, paired 2; 5.4 / 2 = 2.7.
One all-reduce per sub-block: 2 per layer x 64 = 128 per forward pass.

MLP at P = 2: 17408 / 2 = 8704 channels per rank; gate and up (8704, 5120),
down (5120, 8704), output (T, 5120) partial sum.

Attention: 24 / 4 = 6 query heads per KV head. W_Q 5120 x 6144 (24 x 256),
W_K, W_V 5120 x 1024 (4 x 256), W_O 6144 x 5120. P = 2: 12 query, 2 KV per
rank; P = 4: 6 and 1; P = 8: 4 / 8 is not whole. Replicated KV: 64 KiB ->
128 KiB per token.

Linear attention: state 154,927,104 B = 147.75 MiB (147.8) per sequence;
/ 2 = 73.875 (73.9); / 4 = 36.94 (36.9). 48 / 2 = 24 value heads, 16 / 2 = 8
key heads. gcd(24, 4, 48, 16) = 4.

Replicated pieces: norm weight 5120 x 2 = 10,240 B = 10 KiB. Embedding
248,320 x 5120 x 2 = 2.543e9 B = 2.54 GB. Logits 248,320 x 2 = 496,640 B =
497 KB per token; / 10,240 = 48.5 ("48 times").

Communication volume: S = 5120 x 2 = 10,240 B = 10 KiB. B(P) = 2 x 2 x
(P - 1)/P x 10,240 = 40,960 (P - 1)/P B: P = 2: 20,480 B = 20.0 KiB; P = 4:
30.0 KiB; P = 8: 35.0 KiB; limit 40.0 KiB.

Prefill (T = 4096, P = 2): 64 x 4096 x 20,480 = 5.369e9 B = 5.37 GB.
NVLink 600 GB/s: 5.369e9 / 600e9 = 8.95 ms. PCIe 64 GB/s: 83.9 ms.
Compute 2 x 26.9e9 x 4096 = 2.204e14 FLOP; / (2 x 312e12) = 353.1 ms (353).
Ratios 8.95 / 353 = 2.5%; 83.9 / 353 = 23.8% ("24%").

Decode: 128 x 8 us = 1.024 ms (1.02). Weight read 42.2 / P ms: 21.1, 10.55
(10.6), 5.27 (5.3), 2.64 (2.6). Steps 22.1, 11.6, 6.3, 3.6 ms (chapter
rounding). Speedups 42.2 / 22.12 = 1.91, 3.64, 6.70; at P = 16 the chapter
gives 11.7 (42.2 / 3.6 with the read rounded to 2.6 ms; unrounded it's
42.2 / 3.66 = 11.5). Efficiency 96, 91, 84, 73% as the chapter states.
Fixed share at P = 16: 1.02 / 3.66 = 28%. 1.02 / 42.2 = 2.4%.

Pipeline: one hidden state at T = 4096: 4096 x 5120 x 2 = 41.9e6 B = 42 MB;
5.37e9 / 41.9e6 = 128 times less.

What goes wrong: 6144 / 16 = 384 features = 1.5 heads of 256.
"""

from manim import *

import kit
from kit import theme as th

WEIGHTS = th.INFO
CACHE = th.GOOD
REPLICA = th.INK_4

SPLIT_SRC = """\
def split_column_parallel(weight, world_size):
    # weight is (out_features, in_features): PyTorch
    # stores W transposed, so the column split is dim 0.
    return list(weight.chunk(world_size, dim=0))

def split_row_parallel(weight, world_size):
    # the row split is the input dimension: dim 1.
    return list(weight.chunk(world_size, dim=1))

def row_parallel_forward(x_shards, shards):
    partials = [x @ s.T for x, s in zip(x_shards, shards)]
    return sum(partials)          # this sum is the all-reduce
"""


def tile(width, height, color, text=None, size=th.SIZE_SMALL, tint=0.3):
    """A filled rectangle standing for a tensor block, with an optional label."""
    rect = Rectangle(width=width, height=height, stroke_color=color, stroke_width=th.STROKE_BOX,
                     fill_color=th.tint(color, tint), fill_opacity=1)
    group = VGroup(rect)
    group.rect, group.text = rect, None
    if text:
        t = kit.prose(text, size=size, color=th.INK, width=None) if "$" in text else kit.label(
            text, size=size, color=th.INK)
        t.move_to(rect)
        group.add(t)
        group.text = t
    return group


def seg(x0, width, height, color, tint=0.35):
    """One segment of a horizontal bar, from x0 to x0 + width."""
    r = Rectangle(width=max(width, 0.01), height=height, fill_color=th.tint(color, tint), fill_opacity=1,
                  stroke_color=color, stroke_width=th.STROKE_HAIRLINE)
    r.move_to([x0 + width / 2, 0, 0])
    return r


def swatch(text, color, size=th.SIZE_LABEL):
    sq = Square(0.26, stroke_color=color, stroke_width=th.STROKE_HAIRLINE, fill_color=th.tint(color, 0.35),
                fill_opacity=1)
    return VGroup(sq, kit.label(text, size=size, color=th.INK_2)).arrange(RIGHT, buff=0.15)


def rule_key(text, color, size=th.SIZE_LABEL):
    r = Line(LEFT * 0.25, RIGHT * 0.25, color=color, stroke_width=th.STROKE_CURVE)
    return VGroup(r, kit.label(text, size=size, color=th.INK_2)).arrange(RIGHT, buff=0.15)


class Lecture(kit.Lecture):
    def construct(self):
        self.gpu = [self.color, th.INFO]
        self.title_slide(
            narration=(
                "Tensor parallelism. "
                "In this lecture, you split every weight matrix across GPUs. "
                "You derive the split from block matrix multiplication, count what it costs to communicate, "
                "and find where the link between GPUs stops paying."
            )
        )
        self.one_gpu()
        self.collectives()
        self.two_cuts()
        self.column_parallel()
        self.row_parallel()
        self.pytorch_code()
        self.pairing()
        self.ring()
        self.naive_cost()
        self.mlp()
        self.attention_heads()
        self.head_divisibility()
        self.linear_attention()
        self.replicated()
        self.comm_volume()
        self.prefill()
        self.decode()
        self.three_ways()
        self.what_goes_wrong()
        self.closing_slide(
            [
                r"Row split ends in one all-reduce.",
                r"128 all-reduces per forward pass.",
                r"4 KV heads: $P \le 4$.",
                r"Decode: 1.91$\times$ at $P = 2$.",
            ],
            narration=(
                "Here's what to keep. "
                "A column split needs no communication, and a row split ends in an all-reduce. "
                "Pair them, and a forward pass costs 128 all-reduces. "
                "This model's 4 KV heads cap the degree at 4. "
                "Traffic stays under 40 KiB per token per layer, and decode on 2 GPUs is 1.91 times faster. "
                "Next, speculative decoding spends decode's idle compute on guessed tokens."
            ),
            book_section="s-what-goes-wrong",
        )

    # Introduction -----------------------------------------------------------

    def one_gpu(self):
        self.slide("one-gpu-barely", "A second GPU halves what each card holds")
        u = 0.1  # Manim units per GB: an 80 GB card is 8 units wide.
        h = 0.7

        def card(weights, w_text, cache_parts):
            frame = Rectangle(width=80 * u, height=h, stroke_color=th.INK_3, stroke_width=th.STROKE_BOX,
                              fill_color=th.RAISE, fill_opacity=1).move_to([40 * u, 0, 0])
            frame.set_z_index(1).set_fill(opacity=0)
            back = Rectangle(width=80 * u, height=h, stroke_width=0, fill_color=th.RAISE,
                             fill_opacity=1).move_to([40 * u, 0, 0])
            w = seg(0, weights * u, h, WEIGHTS)
            wl = kit.label(w_text, size=th.SIZE_LABEL, color=th.INK).move_to(w)
            x = weights * u
            cs = VGroup()
            for gb, text, color in cache_parts:
                c = seg(x, gb * u, h, color)
                cl = kit.label(text, size=th.SIZE_LABEL, color=th.INK).move_to(c)
                cs.add(VGroup(c, cl))
                x += gb * u
            g = VGroup(back, VGroup(w, wl), cs, frame)
            g.w, g.cs = VGroup(w, wl), cs
            return g

        one = card(53.8, "weights 53.8 GB", [(20.40, "cache 19 GiB", CACHE)])
        two = card(26.9, "weights 26.9 GB", [(20.40, "19 GiB", CACHE), (26.84, "+25 GiB of cache", th.GOOD)])
        n1 = kit.label("One A100 80GB", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        n2 = kit.label("Two GPUs, each card", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        note1 = kit.label("311,000 tokens of cache: under 10,000 each at batch 32", size=th.SIZE_LABEL,
                          color=th.INK_3)
        r1 = VGroup(n1, one, note1).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        r2 = VGroup(n2, two).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        bars = kit.bar_compare(
            [("Decode floor, one GPU", 42.2, WEIGHTS, "42.2 ms per token"),
             ("Decode floor, two GPUs", 21.1, self.color, "21.1 ms per token")],
            width=4.2, bar_height=0.45, gap=0.25, size=th.SIZE_SMALL,
        )
        col = VGroup(r1, r2, bars).arrange(DOWN, buff=0.55, aligned_edge=LEFT)
        self.place(col)
        two.cs[1].set_z_index(0)
        with self.narrate(
            "Your engine fits on one A100, but barely. "
            "The weights take 53.8 GB in bf16. "
            "That leaves about 19 GiB for the KV cache. "
            "At 64 KiB per token, that holds about 311,000 tokens. "
            "Shared by 32 sequences, that's under 10,000 tokens each. "
            "Now add a second GPU. "
            "Each card holds 26.9 GB of weights, which frees another 25 GiB for cache. "
            "Decode is memory bound, so its floor is reading every weight once. "
            "With each GPU reading half, that floor drops from 42.2 ms to 21.1 ms per token."
        ) as t:
            self.play(FadeIn(n1), FadeIn(one[0]), FadeIn(one[3]), run_time=0.6)
            t.wait_until("The weights take")
            self.play(GrowFromEdge(one.w[0], LEFT), FadeIn(one.w[1]), run_time=0.8)
            t.wait_until("That leaves about")
            self.play(GrowFromEdge(one.cs[0][0], LEFT), FadeIn(one.cs[0][1]), run_time=0.8)
            t.wait_until("At 64 KiB")
            self.play(FadeIn(note1), run_time=0.6)
            t.wait_until("Now add a second")
            self.play(FadeIn(n2), FadeIn(two[0]), FadeIn(two[3]), run_time=0.6)
            t.wait_until("Each card holds")
            self.play(GrowFromEdge(two.w[0], LEFT), FadeIn(two.w[1]), GrowFromEdge(two.cs[0][0], LEFT),
                      FadeIn(two.cs[0][1]), run_time=0.8)
            self.play(GrowFromEdge(two.cs[1][0], LEFT), FadeIn(two.cs[1][1]), run_time=0.8)
            t.wait_until("Decode is memory")
            self.play(FadeIn(bars.labels[0]), GrowFromEdge(bars.bars[0], LEFT), FadeIn(bars.values[0]),
                      run_time=0.8)
            t.wait_until("With each GPU")
            self.play(FadeIn(bars.labels[1]), GrowFromEdge(bars.bars[1], LEFT), FadeIn(bars.values[1]),
                      run_time=0.8)

    # Before you start -------------------------------------------------------

    def collectives(self):
        self.slide("collectives", "Ranks meet in collectives", book_section="s-before-you-start")
        top = kit.prose(r"A \textit{rank} is one GPU's process. The degree $P$ is how many ranks share the model.",
                        size=th.SIZE_SMALL, width=12.0)
        tab = kit.table(
            [
                ["All-gather", "every rank's tensor, concatenated"],
                ["Reduce-scatter", "its own slice of the sum"],
                ["Broadcast", "a copy of one rank's tensor"],
                ["All-reduce", "the sum over all ranks, on every rank"],
            ],
            header=["Collective", "Each rank ends with"],
            size=th.SIZE_SMALL, row_gap=0.3,
        )

        def vec(vals, color=None):
            g = kit.grid(1, 3, cell=(0.62, 0.55), values=[[str(v) for v in vals]])
            if color is not None:
                g.paint(g.row(0), color)
            return g

        g0, g1 = vec([1, 2, 3]), vec([4, 5, 6])
        l0 = kit.label("rank 0", size=th.SIZE_LABEL, color=th.INK_3)
        l1 = kit.label("rank 1", size=th.SIZE_LABEL, color=th.INK_3)
        row0 = VGroup(l0, g0).arrange(RIGHT, buff=0.3)
        row1 = VGroup(l1, g1).arrange(RIGHT, buff=0.3)
        demo = VGroup(row0, row1).arrange(DOWN, buff=1.3, aligned_edge=RIGHT)
        body = VGroup(tab, demo).arrange(RIGHT, buff=1.0)
        col = VGroup(top, body).arrange(DOWN, buff=0.6)
        self.place(col)
        s0, s1 = vec([5, 7, 9], self.color), vec([5, 7, 9], self.color)
        for s_, g_ in ((s0, g0), (s1, g1)):
            s_.match_width(g_).move_to(g_)
        link = DoubleArrow(g0.get_bottom(), g1.get_top(), buff=0.12, color=th.INK_3, stroke_width=3,
                           tip_length=0.18)
        link_t = kit.label("add", size=th.SIZE_LABEL, color=th.INK_3).next_to(link, RIGHT, buff=0.15)
        band = kit.mark(VGroup(*tab.cells[4]), buff=0.12)
        with self.narrate(
            "A rank is one GPU's process, and the degree, P, is how many ranks share the model. "
            "A collective is a step that every rank joins at once. "
            "An all-gather concatenates every rank's tensor. "
            "A reduce-scatter leaves each rank its own slice of the sum. "
            "A broadcast copies one rank's tensor to the rest. "
            "The one you'll meet most is the all-reduce. "
            "Each rank starts with a tensor of the same shape. "
            "Each ends with the elementwise sum."
        ) as t:
            self.play(FadeIn(top), run_time=0.7)
            t.wait_until("A collective is")
            self.play(FadeIn(tab.rules), *[FadeIn(c) for c in tab.cells[0]], run_time=0.7)
            for r, cue in zip((1, 2, 3), ["An all-gather", "A reduce-scatter", "A broadcast copies"]):
                t.wait_until(cue)
                self.play(*[FadeIn(c, shift=RIGHT * 0.1) for c in tab.cells[r]], run_time=0.5)
            t.wait_until("The one you'll meet")
            self.play(*[FadeIn(c, shift=RIGHT * 0.1) for c in tab.cells[4]], FadeIn(band), run_time=0.6)
            t.wait_until("Each rank starts")
            self.play(FadeIn(row0), FadeIn(row1), run_time=0.7)
            t.wait_until("Each ends with")
            self.play(GrowArrow(link), FadeIn(link_t), run_time=0.6)
            self.play(ReplacementTransform(g0, s0), ReplacementTransform(g1, s1), run_time=1.0)

    # Two ways to split a matrix --------------------------------------------

    def two_cuts(self):
        self.slide("two-cuts", "Cut the weight by columns or by rows", book_section="s-two-ways-to-split-a-matrix")
        a, b = self.gpu

        def weight(title, l1, l2):
            g = kit.grid(4, 4, cell=0.5)
            name = kit.label(title, size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
            ins = kit.label("inputs", size=th.SIZE_LABEL, color=th.INK_4).rotate(PI / 2).next_to(g, LEFT, buff=0.15)
            outs = kit.label("outputs", size=th.SIZE_LABEL, color=th.INK_4).next_to(g, UP, buff=0.12)
            notes = VGroup(kit.label(l1, size=th.SIZE_LABEL, color=th.INK_2),
                           kit.label(l2, size=th.SIZE_LABEL, color=th.INK_2)).arrange(DOWN, buff=0.1)
            fig = VGroup(VGroup(g, ins, outs), notes).arrange(DOWN, buff=0.3)
            box = VGroup(name, fig).arrange(DOWN, buff=0.25)
            box.g, box.notes = g, notes
            return box

        colw = weight("Column split", "needs all of X", "makes half of Y, finished")
        roww = weight("Row split", "needs half of X", "makes all of Y, partial")
        pics = VGroup(colw, roww).arrange(RIGHT, buff=2.0, aligned_edge=UP)
        eq = kit.formula(r"y =", r"x_1 w_1 + x_2 w_2", r"+", r"x_3 w_3 + x_4 w_4")
        eq[1].set_color(a)
        eq[3].set_color(b)
        br0 = Brace(eq[1], DOWN, buff=0.1, color=a)
        br1 = Brace(eq[3], DOWN, buff=0.1, color=b)
        t0 = kit.label("GPU 0", size=th.SIZE_LABEL, color=a).next_to(br0, DOWN, buff=0.08)
        t1 = kit.label("GPU 1", size=th.SIZE_LABEL, color=b).next_to(br1, DOWN, buff=0.08)
        sumrow = VGroup(eq, br0, br1, t0, t1)
        col = VGroup(pics, sumrow).arrange(DOWN, buff=0.55)
        self.place(col)
        cg, rg = colw.g, roww.g
        with self.narrate(
            "Think of the weight W as a grid, with one row per input feature and one column per output feature. "
            "Cut it vertically, and each GPU holds half the columns. "
            "Each needs all of the input, and makes half the output columns, finished. "
            "Cut it horizontally, and each GPU holds half the rows. "
            "Each needs only the matching half of the input. "
            "But each makes a full-size output that's only part of the sum. "
            "For one output from four inputs, GPU 0 computes the first two terms, and GPU 1 the last two. "
            "Neither number is y, but their sum is."
        ) as t:
            self.play(FadeIn(colw[0]), FadeIn(colw[1][0]), FadeIn(roww[0]), FadeIn(roww[1][0]), run_time=0.8)
            t.wait_until("Cut it vertically")
            self.play(cg.fill(cg.block(0, 4, 0, 2), a), cg.fill(cg.block(0, 4, 2, 4), b), run_time=0.9)
            t.wait_until("Each needs all")
            self.play(FadeIn(colw.notes), run_time=0.6)
            t.wait_until("Cut it horizontally")
            self.play(rg.fill(rg.block(0, 2, 0, 4), a), rg.fill(rg.block(2, 4, 0, 4), b), run_time=0.9)
            t.wait_until("Each needs only")
            self.play(FadeIn(roww.notes[0]), run_time=0.5)
            t.wait_until("But each makes")
            self.play(FadeIn(roww.notes[1]), run_time=0.5)
            t.wait_until("For one output")
            self.play(FadeIn(eq), run_time=0.8)
            self.play(GrowFromCenter(br0), FadeIn(t0), run_time=0.6)
            self.play(GrowFromCenter(br1), FadeIn(t1), run_time=0.6)
            t.wait_until("Neither number")
            self.play(Indicate(eq[2], color=th.INK, scale_factor=1.6), run_time=1.0)

    # Column and row parallel ----------------------------------------------

    def column_parallel(self):
        self.slide("column-parallel", "Column parallel needs no communication",
                   book_section="s-column-parallel-linear-layers")
        a, b = self.gpu
        x = tile(1.6, 1.2, th.INK_4, "$X$")
        w1 = tile(1.2, 1.6, a, "$W_1$")
        w2 = tile(1.2, 1.6, b, "$W_2$")
        w = VGroup(w1, w2).arrange(RIGHT, buff=0)
        y1 = tile(1.2, 1.2, a, "$XW_1$")
        y2 = tile(1.2, 1.2, b, "$XW_2$")
        y = VGroup(y1, y2).arrange(RIGHT, buff=0)
        times = kit.formula(r"\times")
        eq = kit.formula(r"=")
        prod = VGroup(x, times, w, eq, y).arrange(RIGHT, buff=0.35)
        xd = kit.prose(r"$T \times d_{\mathrm{in}}$", size=th.SIZE_LABEL, color=th.INK_3, width=None)
        xd.next_to(x, DOWN, buff=0.15)
        wd = kit.prose(r"$d_{\mathrm{in}} \times d_{\mathrm{out}}$", size=th.SIZE_LABEL, color=th.INK_3, width=None)
        wd.next_to(w, DOWN, buff=0.15)
        fig = VGroup(prod, xd, wd)
        f = kit.formula(r"XW = \begin{bmatrix} XW_1 & XW_2 & \cdots & XW_P \end{bmatrix}")
        box = SurroundingRectangle(f, color=self.color, buff=0.15, corner_radius=0.08, stroke_width=3)
        pills = VGroup(kit.pill("input replicated", th.INK_3, size=th.SIZE_LABEL),
                       kit.pill("output sharded", th.INK_3, size=th.SIZE_LABEL),
                       kit.pill("no communication", self.color, size=th.SIZE_LABEL)).arrange(RIGHT, buff=0.35)
        col = VGroup(fig, VGroup(f, box), pills).arrange(DOWN, buff=0.6)
        self.place(col)
        w2_home = w2.get_center()
        with self.narrate(
            "Now the algebra. "
            "Write a linear layer as Y equals X times W, for T tokens. "
            "Cut W into P blocks of columns. "
            "Matrix multiplication works block by block, so the product is X times each block, side by side. "
            "Rank p holds one block, receives the whole input, and computes its own output columns. "
            "Each output column needs the full input, but only its own column of W. "
            "So nothing crosses the link."
        ) as t:
            t.wait_until("Write a linear")
            self.play(FadeIn(x), FadeIn(xd), FadeIn(times), FadeIn(w), FadeIn(wd), run_time=0.8)
            t.wait_until("Cut W into")
            self.play(w1.animate.shift(LEFT * 0.08), w2.animate.move_to(w2_home + RIGHT * 0.08), run_time=0.6)
            t.wait_until("Matrix multiplication works")
            self.play(FadeIn(eq), FadeIn(y1, shift=RIGHT * 0.1), FadeIn(y2, shift=RIGHT * 0.1), run_time=0.8)
            self.play(FadeIn(f), Create(box), run_time=0.8)
            t.wait_until("Rank p holds")
            self.play(Indicate(x, color=th.INK_3, scale_factor=1.05), Indicate(w1, color=a, scale_factor=1.06),
                      Indicate(y1, color=a, scale_factor=1.06), run_time=1.2)
            t.wait_until("Each output column")
            self.play(FadeIn(pills[0]), FadeIn(pills[1]), run_time=0.6)
            t.wait_until("So nothing crosses")
            self.play(FadeIn(pills[2], scale=0.9), run_time=0.5)

    def row_parallel(self):
        self.slide("row-parallel", "Row parallel ends in one all-reduce", book_section="s-row-parallel-linear-layers")
        a, b = self.gpu
        x1 = tile(0.8, 1.2, a, "$X_1$")
        x2 = tile(0.8, 1.2, b, "$X_2$")
        x = VGroup(x1, x2).arrange(RIGHT, buff=0)
        w1 = tile(2.2, 0.8, a, "$W_1$")
        w2 = tile(2.2, 0.8, b, "$W_2$")
        w = VGroup(w1, w2).arrange(DOWN, buff=0)
        top = VGroup(x, kit.formula(r"\times"), w).arrange(RIGHT, buff=0.35)
        p0 = tile(2.2, 1.2, a, "$X_1 W_1$")
        p1 = tile(2.2, 1.2, b, "$X_2 W_2$")
        plus = kit.formula(r"+")
        yf = tile(2.2, 1.2, th.INK_3, "$Y$", tint=0.15)
        ar = kit.label("all-reduce", size=th.SIZE_LABEL, color=th.INK_2, weight=MEDIUM)
        arr = Arrow(LEFT * 0.7, RIGHT * 0.7, buff=0, color=th.INK_3, stroke_width=4, tip_length=0.2)
        arrow = VGroup(ar, arr).arrange(DOWN, buff=0.08)
        bottom = VGroup(p0, plus, p1, arrow, yf).arrange(RIGHT, buff=0.3)
        tags = VGroup(kit.label("rank 0: partial", size=th.SIZE_LABEL, color=a).next_to(p0, DOWN, buff=0.12),
                      kit.label("rank 1: partial", size=th.SIZE_LABEL, color=b).next_to(p1, DOWN, buff=0.12))
        f = kit.formula(r"XW = \sum_{p=1}^{P} X_p\, W_p", size=th.SIZE_MATH)
        upper = VGroup(top, f).arrange(RIGHT, buff=1.2)
        warn = kit.prose(r"Add the partial sums; \textbf{never concatenate} them.", size=th.SIZE_SMALL,
                         width=None, color=th.BAD)
        col = VGroup(upper, VGroup(bottom, tags), warn).arrange(DOWN, buff=0.55)
        self.place(col)
        with self.narrate(
            "The row split works the other way. "
            "Cut W into blocks of rows, and cut X into matching blocks of columns. "
            "Each output is a sum over the input features. "
            "Group that sum by block, and the product becomes a sum of block products. "
            "Rank p holds one block of rows, and only its slice of the input. "
            "It computes a partial sum with the full output shape. "
            "Adding them across ranks is an all-reduce. "
            "The partial sums must be added, never concatenated."
        ) as t:
            t.wait_until("Cut W into")
            self.play(FadeIn(top), run_time=0.8)
            t.wait_until("Group that sum")
            self.play(FadeIn(f, shift=DOWN * 0.1), run_time=0.8)
            t.wait_until("It computes a partial")
            self.play(TransformFromCopy(VGroup(x1, w1), p0), FadeIn(tags[0]), run_time=0.9)
            self.play(TransformFromCopy(VGroup(x2, w2), p1), FadeIn(tags[1]), FadeIn(plus), run_time=0.9)
            t.wait_until("Adding them across")
            self.play(GrowArrow(arr), FadeIn(ar), run_time=0.6)
            self.play(FadeIn(yf, shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("The partial sums must")
            self.play(FadeIn(warn, shift=UP * 0.1), run_time=0.7)

    def pytorch_code(self):
        self.slide("pytorch-code", "In PyTorch, column parallel is dim 0",
                   book_section="s-row-parallel-linear-layers")
        panel = kit.code(SPLIT_SRC, "python", caption="lab 19")
        self.place(panel)
        with self.narrate(
            "In code, PyTorch flips the layout, the most common source of confusion here. "
            "A linear layer stores its weight as output features by input features. "
            "So a column split, in the math sense, is a chunk along dimension zero. "
            "The row split runs along the input features, which is dimension one. "
            "The row-parallel forward pass multiplies each input shard by its weight shard. "
            "Then it adds the partial results, and on real GPUs, that sum is the all-reduce."
        ) as t:
            self.play(FadeIn(panel.frame), *panel.reveal(1), run_time=0.7)
            t.wait_until("A linear layer stores")
            self.play(*panel.reveal(2, 3), run_time=0.7)
            t.wait_until("So a column split")
            self.play(*panel.reveal(4), panel.highlight(4), run_time=0.7)
            t.wait_until("The row split runs")
            self.play(*panel.reveal(6, 7, 8), panel.highlight(8), run_time=0.8)
            t.wait_until("The row-parallel forward")
            self.play(*panel.reveal(10, 11), run_time=0.7)
            t.wait_until("Then it adds")
            self.play(*panel.reveal(12), panel.highlight(12), run_time=0.7)

    # The pairing -----------------------------------------------------------

    def pairing(self):
        self.slide("pairing", "Column then row: one all-reduce per block",
                   book_section="s-the-pairing-and-why-one-all-reduce-is-enough")
        a, b = self.gpu
        xb = kit.box("$X$", width=1.1, height=1.0)
        lanes = VGroup()
        for g, c in enumerate(self.gpu):
            cp = kit.box("column parallel", sublabel=f"rank {g}", color=c, width=2.7)
            hb = kit.box(f"$H_{g}$", color=c, width=1.0, height=1.0)
            rp = kit.box("row parallel", sublabel=f"rank {g}", color=c, width=2.7)
            lane = VGroup(cp, hb, rp).arrange(RIGHT, buff=0.75)
            lane.cp, lane.hb, lane.rp = cp, hb, rp
            lanes.add(lane)
        lanes.arrange(DOWN, buff=1.1)
        ar = kit.box("all-reduce", color=th.INK_3, width=2.0, height=1.0)
        yb = kit.box("$Y$", width=1.1, height=1.0)
        right = VGroup(ar, yb).arrange(RIGHT, buff=0.7)
        fig = VGroup(xb, lanes, right).arrange(RIGHT, buff=0.8)
        f = kit.formula(r"X \xrightarrow{\text{column}} H_p \xrightarrow{\text{row}} Y^{(p)}"
                        r"\xrightarrow{\text{all-reduce}} Y", size=th.SIZE_MATH_SMALL)
        col = VGroup(fig, f).arrange(DOWN, buff=0.6)
        self.place(col)
        ins = VGroup(*[kit.arrow(xb, ln.cp) for ln in lanes])
        mids = VGroup(*[kit.arrow(ln.cp, ln.hb) for ln in lanes])
        mids2 = VGroup(*[kit.arrow(ln.hb, ln.rp) for ln in lanes])
        outs = VGroup(*[kit.arrow(ln.rp, ar) for ln in lanes])
        fin = kit.arrow(ar, yb)
        never = kit.label("never gathered", size=th.SIZE_LABEL, color=th.INK_3)
        never.move_to((lanes[0].hb.get_bottom() + lanes[1].hb.get_top()) / 2)
        with self.narrate(
            "Now put the two splits back to back. "
            "A column-parallel layer produces an output sharded along its output dimension. "
            "A row-parallel layer wants an input sharded along its input dimension. "
            "In a two-layer stack, those are the same axis. "
            "So the sharded intermediate is never gathered. "
            "It goes straight into the next layer, in the form that layer wants. "
            "The only communication is one all-reduce at the end, which is what makes tensor parallelism cheap."
        ) as t:
            self.play(FadeIn(xb), *[FadeIn(ln.cp) for ln in lanes], *[GrowArrow(a_.line) for a_ in ins],
                      run_time=0.8)
            self.play(*[GrowArrow(m.line) for m in mids], *[FadeIn(ln.hb) for ln in lanes], run_time=0.7)
            t.wait_until("A row-parallel layer")
            self.play(*[FadeIn(ln.rp) for ln in lanes], run_time=0.7)
            t.wait_until("In a two-layer")
            self.play(*[GrowArrow(m.line) for m in mids2], run_time=0.7)
            t.wait_until("So the sharded")
            self.play(*[Indicate(ln.hb, color=c, scale_factor=1.1) for ln, c in zip(lanes, self.gpu)],
                      FadeIn(never), run_time=1.0)
            t.wait_until("The only communication")
            self.play(*[GrowArrow(o.line) for o in outs], FadeIn(ar),
                      run_time=0.8)
            self.play(GrowArrow(fin.line), FadeIn(yb), run_time=0.6)
            self.play(FadeIn(f, shift=UP * 0.1), run_time=0.7)

    def ring(self):
        self.slide("ring-all-reduce", "A ring all-reduce sends under twice the tensor",
                   book_section="s-the-pairing-and-why-one-all-reduce-is-enough")
        P = 4
        cols = self.colors
        ranks = VGroup()
        grids = []
        for r in range(P):
            g = kit.grid(1, P, cell=(0.5, 0.5))
            for c in range(P):
                g.paint(g.cell(0, c), cols[c], tint=0.12)
            name = kit.label(f"rank {r}", size=th.SIZE_LABEL, color=th.INK_3)
            ranks.add(VGroup(name, g).arrange(DOWN if r < 2 else UP, buff=0.12))
            grids.append(g)
        pos = [(-1.9, 1.25), (1.9, 1.25), (1.9, -1.25), (-1.9, -1.25)]
        for rk, (px, py) in zip(ranks, pos):
            rk.move_to([px, py, 0])
        f = kit.formula(r"2\,", r"\frac{P-1}{P}", r"\,S", size=th.SIZE_MATH)
        f[1].set_color(th.HL_C)
        f[2].set_color(th.HL_D)
        fcap = kit.label("bytes sent per rank", size=th.SIZE_LABEL, color=th.INK_3)
        p2 = kit.prose(r"$P = 2$: exactly one tensor, $S$", size=th.SIZE_SMALL, width=None)
        pinf = kit.prose(r"large $P$: approaches $2S$", size=th.SIZE_SMALL, width=None)
        phase = kit.label("reduce-scatter", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
        phase2 = kit.label("all-gather", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
        key = kit.label("darker: more ranks summed in", size=th.SIZE_LABEL, color=th.INK_3)
        side = VGroup(VGroup(f, fcap).arrange(DOWN, buff=0.15), p2, pinf).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        ringfig = VGroup(ranks, key)
        key.next_to(ranks, DOWN, buff=0.3)
        layout = VGroup(ringfig, side).arrange(RIGHT, buff=1.2)
        self.place(layout)
        phase.move_to(ranks.get_center())
        phase2.move_to(ranks.get_center())
        arrows = VGroup(*[kit.arrow(ranks[r][1], ranks[(r + 1) % P][1]) for r in range(P)])
        count = [[1] * P for _ in range(P)]

        def tint_of(n):
            return 0.12 + 0.5 * (n - 1) / (P - 1)

        def hop(sends, run_time=0.7):
            movers, fills = [], []
            new = [row[:] for row in count]
            for r, c in sends:
                dst = (r + 1) % P
                sq = grids[r].cell(0, c).square
                m = sq.copy().set_z_index(3)
                movers.append(m)
                fills.append((dst, c))
            self.add(*movers)
            self.play(*[m.animate.move_to(grids[(r + 1) % P].cell(0, c)) for m, (r, c) in zip(movers, sends)],
                      run_time=run_time)
            for (r, c), (dst, _) in zip(sends, fills):
                new[dst][c] = P if sends_gather else count[dst][c] + count[r][c]
            self.remove(*movers)
            for dst, c in fills:
                count[dst][c] = new[dst][c]
            self.play(*[grids[dst].fill(grids[dst].cell(0, c), cols[c], tint=tint_of(count[dst][c]))
                        for dst, c in fills], run_time=0.3)

        with self.narrate(
            "What does one all-reduce cost? "
            "A ring all-reduce arranges the ranks in a ring, and cuts the tensor into P pieces. "
            "First comes a reduce-scatter. "
            "Each rank passes a piece to its neighbor, which adds its own copy and passes it on. "
            "After P minus 1 hops, each rank holds one fully summed piece. "
            "Then an all-gather carries the finished pieces around the ring for another P minus 1 hops. "
            "So each rank sends the tensor size, times 2, times P minus 1 over P. "
            "At 2 ranks, that's exactly one tensor's worth. "
            "As P grows, it approaches twice the tensor."
        ) as t:
            self.play(FadeIn(ranks), run_time=0.7)
            self.play(*[GrowArrow(x.line) for x in arrows], FadeIn(key), run_time=0.7)
            t.wait_until("First comes a")
            self.play(FadeIn(phase), run_time=0.4)
            t.wait_until("Each rank passes")
            sends_gather = False
            for s in range(P - 1):
                hop([(r, (r - s) % P) for r in range(P)])
            t.wait_until("Then an all-gather")
            self.play(FadeOut(phase), FadeIn(phase2), run_time=0.4)
            sends_gather = True
            for s in range(P - 1):
                hop([(r, (r + 1 - s) % P) for r in range(P)])
            t.wait_until("So each rank sends")
            self.play(FadeIn(f), FadeIn(fcap), run_time=0.8)
            t.wait_until("At 2 ranks")
            self.play(FadeIn(p2), run_time=0.6)
            t.wait_until("As P grows")
            self.play(FadeIn(pinf), run_time=0.6)

    def naive_cost(self):
        self.slide("naive-cost", "Gathering the middle costs 2.7 times the bytes",
                   book_section="s-the-pairing-and-why-one-all-reduce-is-enough")
        u = 1.3
        h = 0.6
        n1 = seg(0, 3.4 * u, h, th.INFO)
        n1t = kit.label("all-gather H: 3.4", size=th.SIZE_LABEL, color=th.INK).move_to(n1)
        n2 = seg(3.4 * u, 2.0 * u, h, self.color)
        n2t = kit.label("all-reduce: 2", size=th.SIZE_LABEL, color=th.INK).move_to(n2)
        naive = VGroup(n1, n1t, n2, n2t)
        q = seg(0, 2.0 * u, h, self.color).shift(DOWN * 1.1)
        qt = kit.label("all-reduce: 2", size=th.SIZE_LABEL, color=th.INK).move_to(q)
        paired = VGroup(q, qt)
        name1 = kit.label("Naive", size=th.SIZE_SMALL, color=th.INK_2)
        name2 = kit.label("Paired", size=th.SIZE_SMALL, color=th.INK_2)
        name1.next_to(n1, LEFT, buff=0.3)
        name2.next_to(q, LEFT, buff=0.3).align_to(name1, RIGHT)
        v1 = kit.label("5.4", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM).next_to(n2, RIGHT, buff=0.2)
        v2 = kit.label("2", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM).next_to(q, RIGHT, buff=0.2)
        chart = VGroup(name1, name2, naive, paired, v1, v2)
        unit = kit.prose(r"Bytes per rank, in units of $\tfrac{P-1}{P}\,S_Y$, where $S_H = \tfrac{17408}{5120}"
                         r"\,S_Y = 3.4\,S_Y$", size=th.SIZE_SMALL, width=None, color=th.INK_3)
        ratio = kit.pill("2.7× the bytes", self.color, size=th.SIZE_SMALL)
        key = kit.callout("key", r"Attention half or MLP half: 2 per layer, \textbf{128 per forward pass} "
                                 r"through 64 layers.", title="One all-reduce per sub-block", width=10.0,
                          size=th.SIZE_SMALL)
        mid = VGroup(chart, ratio).arrange(RIGHT, buff=0.8)
        col = VGroup(unit, mid, key).arrange(DOWN, buff=0.6)
        self.place(col)
        with self.narrate(
            "The naive alternative gathers the intermediate after the first layer, then all-reduces after the "
            "second. "
            "In the MLP, the intermediate is 17408 wide, against a hidden size of 5120. "
            "So it's 3.4 times the size of the output. "
            "Add the gather to the all-reduce, and the naive scheme moves 2.7 times the bytes. "
            "It also pays a second collective launch, which in decode matters more than the bytes. "
            "The pairing pays one all-reduce per sub-block, the attention half or the MLP half of a layer. "
            "That's 128 per forward pass."
        ) as t:
            self.play(FadeIn(unit), FadeIn(name1), FadeIn(name2), run_time=0.7)
            t.wait_until("So it's 3.4")
            self.play(GrowFromEdge(n1, LEFT), FadeIn(n1t), run_time=0.8)
            t.wait_until("Add the gather")
            self.play(GrowFromEdge(n2, LEFT), FadeIn(n2t), FadeIn(v1), run_time=0.7)
            self.play(GrowFromEdge(q, LEFT), FadeIn(qt), FadeIn(v2), run_time=0.7)
            self.play(FadeIn(ratio, scale=0.9), run_time=0.5)
            t.wait_until("The pairing pays")
            self.play(FadeIn(key, shift=UP * 0.1), run_time=0.8)

    # The MLP ---------------------------------------------------------------

    def mlp(self):
        self.slide("mlp", "The MLP splits by intermediate channel",
                   book_section="s-the-mlp-shards-by-intermediate-dimension")
        a, b = self.gpu
        ident = kit.formula(r"\big(\operatorname{silu}(XW_g) \odot XW_u\big)_{\text{block } p}", r"=",
                            r"\operatorname{silu}(XW_{g,p}) \odot XW_{u,p}", size=th.SIZE_MATH_SMALL)
        n = 16
        strip = kit.grid(1, n, cell=(0.5, 0.5))
        cap = kit.label("intermediate H: 17408 channels", size=th.SIZE_LABEL, color=th.INK_3)
        cap.next_to(strip, UP, buff=0.15)
        br0 = Brace(strip.block(0, 1, 0, n // 2), DOWN, buff=0.1, color=a)
        br1 = Brace(strip.block(0, 1, n // 2, n), DOWN, buff=0.1, color=b)
        l0 = kit.label("rank 0: 8704", size=th.SIZE_LABEL, color=a).next_to(br0, DOWN, buff=0.08)
        l1 = kit.label("rank 1: 8704", size=th.SIZE_LABEL, color=b).next_to(br1, DOWN, buff=0.08)
        fig = VGroup(cap, strip, br0, br1, l0, l1)
        tab = kit.table(
            [
                [r"$W_g$, $W_u$", "(17408, 5120)", "(8704, 5120)"],
                [r"$W_d$", "(5120, 17408)", "(5120, 8704)"],
                ["Output", "($T$, 5120)", "($T$, 5120) partial sum"],
            ],
            header=["At $P = 2$", "Unsharded", "Per rank"],
            size=th.SIZE_SMALL, row_gap=0.22,
        )
        col = VGroup(ident, fig, tab).arrange(DOWN, buff=0.5)
        self.place(col)
        with self.narrate(
            "Now apply the pairing to the MLP, a SwiGLU block. "
            "Shard its gate and up projections by column, so each rank gets 8704 of the 17408 channels. "
            "The activation and the elementwise product act on each channel on its own. "
            "So a rank's block equals that block of the full intermediate, computed without ever forming it. "
            "Then shard the down projection by row. "
            "Each rank's slice is already in place, and the block ends in one all-reduce. "
            "If the activation mixed channels, as a softmax would, you'd need a collective in the middle."
        ) as t:
            self.play(FadeIn(cap), FadeIn(strip), run_time=0.7)
            t.wait_until("Shard its gate")
            self.play(strip.fill(strip.block(0, 1, 0, n // 2), a), strip.fill(strip.block(0, 1, n // 2, n), b),
                      GrowFromCenter(br0), GrowFromCenter(br1), FadeIn(l0), FadeIn(l1), run_time=0.9)
            self.play(FadeIn(tab.rules), *[FadeIn(c) for c in tab.cells[0]], *[FadeIn(c) for c in tab.cells[1]],
                      run_time=0.7)
            t.wait_until("The activation and")
            self.play(LaggedStart(*[Indicate(strip.cell(0, c), color=a if c < n // 2 else b, scale_factor=1.2)
                                    for c in range(n)], lag_ratio=0.08), run_time=2.0)
            t.wait_until("So a rank's block")
            self.play(FadeIn(ident, shift=DOWN * 0.1), run_time=0.8)
            t.wait_until("Then shard the down")
            self.play(*[FadeIn(c) for c in tab.cells[2]], run_time=0.6)
            t.wait_until("Each rank's slice")
            self.play(*[FadeIn(c) for c in tab.cells[3]], run_time=0.6)

    # Attention -------------------------------------------------------------

    def attention_heads(self):
        self.slide("attention-heads", "Attention splits by whole heads", book_section="s-attention-shards-by-head")
        cols = self.colors
        groups = VGroup()
        for gi in range(4):
            grp = VGroup(*[Square(0.4, stroke_color=cols[gi], stroke_width=th.STROKE_BOX,
                                  fill_color=th.tint(cols[gi], 0.25), fill_opacity=1) for _ in range(6)])
            grp.arrange(RIGHT, buff=0.08)
            groups.add(grp)
        groups.arrange(RIGHT, buff=0.5)
        kvs = VGroup(*[kit.box(f"KV head {i}", color=cols[i], size=th.SIZE_LABEL, width=2.4) for i in range(4)])
        for kv, grp in zip(kvs, groups):
            kv.move_to([grp.get_x(), groups.get_y() - 1.5, 0])
        links = VGroup()
        for kv, grp in zip(kvs, groups):
            for sq in grp:
                links.add(Line(sq.get_bottom(), kv.get_top(), stroke_width=1.5, color=cols[list(kvs).index(kv)]))
        q_lab = kit.label("24 query heads, 6 per KV head", size=th.SIZE_LABEL, color=th.INK_3)
        q_lab.next_to(groups, UP, buff=0.2).align_to(groups, LEFT)
        heads = VGroup(q_lab, groups, links, kvs)
        frames = VGroup()
        for g, half in enumerate((VGroup(groups[0], groups[1], kvs[0], kvs[1]),
                                  VGroup(groups[2], groups[3], kvs[2], kvs[3]))):
            fr = SurroundingRectangle(half, buff=0.22, corner_radius=0.12, color=th.INK_3, stroke_width=2)
            nm = kit.label(f"GPU {g}", size=th.SIZE_LABEL, color=th.INK_2, weight=MEDIUM)
            nm.next_to(fr, DOWN, buff=0.12)
            frames.add(VGroup(fr, nm))
        notes = kit.bullets(
            [
                r"$W_Q$, $W_K$, $W_V$: column parallel, cut on head boundaries",
                r"$W_O$: row parallel, then one all-reduce",
                r"Each rank: 12 query heads, 2 KV heads, and their cache",
            ],
            size=th.SIZE_SMALL, width=11.0, buff=0.2,
        )
        col = VGroup(VGroup(heads, frames), notes).arrange(DOWN, buff=0.5)
        self.place(col)
        with self.narrate(
            "Attention isn't one matmul, so it needs its own argument. "
            "Here, 24 query heads share 4 KV heads, six per group. "
            "Each head's output depends only on its own query, and its group's keys and values. "
            "So split by head. "
            "The query, key, and value projections are column parallel, cut on head boundaries. "
            "The output projection takes the concatenated heads, so it's row parallel, with one all-reduce. "
            "At 2 GPUs, each rank owns 12 query heads, 2 KV heads, and their cache. "
            "The cut must fall between heads, because the softmax normalizes within one head."
        ) as t:
            t.wait_until("Here, 24 query")
            self.play(FadeIn(q_lab), LaggedStart(*[FadeIn(s) for g in groups for s in g], lag_ratio=0.03),
                      FadeIn(kvs), run_time=1.2)
            t.wait_until("Each head's output")
            self.play(LaggedStart(*[Create(l) for l in links], lag_ratio=0.02), run_time=1.2)
            t.wait_until("So split by head")
            self.play(*[Create(fr[0]) for fr in frames], *[FadeIn(fr[1]) for fr in frames], run_time=0.9)
            t.wait_until("The query, key")
            self.play(FadeIn(notes[0], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("The output projection")
            self.play(FadeIn(notes[1], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("At 2 GPUs")
            self.play(FadeIn(notes[2], shift=RIGHT * 0.1), run_time=0.6)

    def head_divisibility(self):
        self.slide("head-divisibility", "4 KV heads cap the degree at 4", book_section="s-attention-shards-by-head")
        cols = self.colors
        cw = 0.36
        q = kit.grid(1, 24, cell=(cw, 0.45))
        kv = kit.grid(1, 4, cell=(6 * cw, 0.5), values=[[f"KV {i}" for i in range(4)]])
        for i in range(24):
            q.paint(q.cell(0, i), cols[i // 6], tint=0.3)
        for i in range(4):
            kv.paint(kv.cell(0, i), cols[i], tint=0.3)
        kv.next_to(q, DOWN, buff=0.18)
        ql = kit.label("query heads", size=th.SIZE_LABEL, color=th.INK_3).next_to(q, LEFT, buff=0.2)
        kl = kit.label("KV heads", size=th.SIZE_LABEL, color=th.INK_3).next_to(kv, LEFT, buff=0.2).align_to(ql, RIGHT)
        fig = VGroup(q, kv, ql, kl)
        tab = kit.table(
            [["2", "12", "2"], ["4", "6", "1"], ["8", "—", "—"]],
            header=["Degree $P$", "Query heads per rank", "KV heads per rank"],
            size=th.SIZE_SMALL, align=["c", "c", "c"], row_gap=0.24,
        )
        warn = kit.callout("warning", r"Replicating KV heads at $P = 8$ doubles the cache from 64 KiB to "
                                      r"128 KiB per token.", title="The workaround", width=5.4, size=th.SIZE_SMALL)
        low = VGroup(tab, warn).arrange(RIGHT, buff=0.6)
        pl = kit.label("P = 2", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
        col = VGroup(VGroup(pl, fig).arrange(DOWN, buff=0.45), low).arrange(DOWN, buff=0.6)
        self.place(col)
        pl.align_to(fig, LEFT)
        for i in range(4):
            sq = kv.cell(0, i).square
            kv.cell(0, i).text.move_to(sq.get_left() + RIGHT * sq.width * 0.25)
        y0, y1 = kv.get_bottom()[1] - 0.1, q.get_top()[1] + 0.1
        x0, x1 = q.get_left()[0], q.get_right()[0]

        def cuts(p, color=th.INK):
            out = VGroup()
            for k in range(1, p):
                x = x0 + (x1 - x0) * k / p
                bad = (k * 4) % p != 0
                out.add(Line([x, y0, 0], [x, y1, 0], color=th.BAD if bad else color, stroke_width=4))
            return out

        c2, c4, c8 = cuts(2), cuts(4), cuts(8)
        p4 = kit.label("P = 4", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM).move_to(pl, aligned_edge=LEFT)
        p8 = kit.label("P = 8: KV heads cut in half", size=th.SIZE_SMALL, color=th.BAD,
                       weight=MEDIUM).move_to(pl, aligned_edge=LEFT)
        with self.narrate(
            "A rank must also own every query head of each KV head it holds. "
            "So P must divide the number of KV heads, which is 4. "
            "At 2 ranks, each gets 12 query heads and 2 KV heads. "
            "At 4 ranks, each gets 6 query heads and 1 KV head. "
            "At 8 ranks, there's no valid split. "
            "The usual workaround replicates KV heads. "
            "That doubles the cache from 64 KiB to 128 KiB per token, and gives back the memory you wanted."
        ) as t:
            self.play(FadeIn(fig), run_time=0.8)
            t.wait_until("So P must divide")
            self.play(FadeIn(tab.rules), *[FadeIn(c) for c in tab.cells[0]], run_time=0.6)
            t.wait_until("At 2 ranks")
            self.play(FadeIn(pl), *[Create(l) for l in c2], *[FadeIn(c) for c in tab.cells[1]], run_time=0.8)
            t.wait_until("At 4 ranks")
            self.play(ReplacementTransform(pl, p4), ReplacementTransform(c2, c4[1]), Create(c4[0]), Create(c4[2]),
                      *[FadeIn(c) for c in tab.cells[2]], run_time=0.8)
            t.wait_until("At 8 ranks")
            self.play(ReplacementTransform(p4, p8), ReplacementTransform(c4, VGroup(c8[1], c8[3], c8[5])),
                      *[Create(c8[i]) for i in (0, 2, 4, 6)], *[FadeIn(c) for c in tab.cells[3]], run_time=0.9)
            t.wait_until("The usual workaround")
            self.play(FadeIn(warn, shift=UP * 0.1), run_time=0.7)

    # Linear attention -------------------------------------------------------

    def linear_attention(self):
        self.slide("linear-attention", "The recurrent state shards with its heads",
                   book_section="s-the-hybrid-model-s-linear-attention-layers")
        a, b = self.gpu
        kh = kit.grid(1, 16, cell=(0.36, 0.36))
        vh = kit.grid(3, 16, cell=(0.36, 0.36))
        vh.next_to(kh, DOWN, buff=0.15)
        kl = kit.label("16 key heads", size=th.SIZE_LABEL, color=th.INK_3).next_to(kh, UP, buff=0.12)
        kl.align_to(kh, LEFT)
        vl = kit.label("48 value heads, each with a 128 × 128 state", size=th.SIZE_LABEL, color=th.INK_3)
        vl.next_to(vh, DOWN, buff=0.15).align_to(vh, LEFT)
        fig = VGroup(kh, vh, kl, vl)
        bars = kit.bar_compare(
            [("P = 1", 147.75, th.INK_3, "147.8 MiB"), ("P = 2", 73.875, self.color, "73.9 MiB"),
             ("P = 4", 36.94, self.color, "36.9 MiB")],
            width=3.0, bar_height=0.42, gap=0.25, size=th.SIZE_SMALL,
        )
        bcap = kit.label("state per sequence, per rank", size=th.SIZE_LABEL, color=th.INK_3)
        right = VGroup(bcap, bars).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        top = VGroup(fig, right).arrange(RIGHT, buff=0.8)
        g = kit.formula(r"P \mid \gcd(24,\, 4,\, 48,\, 16) = 4", size=th.SIZE_MATH)
        gbox = SurroundingRectangle(g, color=self.color, buff=0.16, corner_radius=0.08, stroke_width=3)
        gl = kit.label("query heads, KV heads, value heads, key heads", size=th.SIZE_LABEL, color=th.INK_3)
        bottom = VGroup(VGroup(g, gbox), gl).arrange(DOWN, buff=0.2)
        col = VGroup(top, bottom).arrange(DOWN, buff=0.6)
        self.place(col)
        with self.narrate(
            "The other 48 layers run the gated delta rule, with 16 key heads and 48 value heads. "
            "Each value head carries its own state, gate, and step size. "
            "So these layers shard by value head, as attention shards by query head. "
            "The state splits with the heads, and never moves. "
            "Per sequence, it's 147.8 MiB on one GPU. "
            "At 2 GPUs, that's 73.9 MiB per rank, and at 4, it's 36.9. "
            "Every head count must divide evenly, so P must divide their greatest common divisor. "
            "That's 4. "
            "This model supports 1, 2, or 4 GPUs, and nothing above."
        ) as t:
            self.play(FadeIn(kh), FadeIn(kl), FadeIn(vh), FadeIn(vl), run_time=0.8)
            t.wait_until("So these layers")
            self.play(kh.fill(kh.block(0, 1, 0, 8), a), kh.fill(kh.block(0, 1, 8, 16), b),
                      vh.fill(vh.block(0, 3, 0, 8), a), vh.fill(vh.block(0, 3, 8, 16), b), run_time=1.0)
            t.wait_until("Per sequence")
            self.play(FadeIn(bcap), FadeIn(bars.labels[0]), GrowFromEdge(bars.bars[0], LEFT), FadeIn(bars.values[0]),
                      run_time=0.8)
            t.wait_until("At 2 GPUs")
            self.play(FadeIn(bars.labels[1]), GrowFromEdge(bars.bars[1], LEFT), FadeIn(bars.values[1]), run_time=0.6)
            self.play(FadeIn(bars.labels[2]), GrowFromEdge(bars.bars[2], LEFT), FadeIn(bars.values[2]), run_time=0.6)
            t.wait_until("Every head count")
            self.play(FadeIn(g), FadeIn(gl), run_time=0.8)
            t.wait_until("That's 4")
            self.play(Create(gbox), run_time=0.6)

    # What stays replicated ----------------------------------------------------

    def replicated(self):
        self.slide("replicated", "Norms stay whole; logits stay sharded", book_section="s-what-stays-replicated")
        cards = VGroup(
            kit.callout("note", r"Replicated: 10 KiB each.", title="RMSNorm", width=4.1, size=th.SIZE_SMALL),
            kit.callout("note", r"2.54 GB, or split by vocabulary.", title="Input embedding", width=4.1,
                        size=th.SIZE_SMALL),
            kit.callout("key", r"Column parallel; logits stay sharded.", title="Output projection", width=4.1,
                        size=th.SIZE_SMALL),
        ).arrange(RIGHT, buff=0.3, aligned_edge=UP)
        bars = kit.bar_compare(
            [("Gather the logits", 497, th.BAD, "497 KB per token"),
             ("Hidden-state all-reduce", 10.24, th.INFO, "10 KiB per token"),
             ("Vocab-parallel sampling", 0.5, self.color, "two scalars, then one token id")],
            width=5.5, bar_height=0.42, gap=0.25, size=th.SIZE_SMALL,
        )
        col = VGroup(cards, bars).arrange(DOWN, buff=0.6)
        self.place(col)
        with self.narrate(
            "Not every weight is worth splitting. "
            "A norm reduces over the whole hidden vector, and the previous all-reduce already leaves that vector "
            "whole. "
            "So norms stay replicated, at 10 KiB each. "
            "The input embedding costs 2.54 GB per rank, or you can split it by vocabulary for one more all-reduce. "
            "The output projection maps to 248,320 logits, so it's naturally column parallel. "
            "Gathering its output would cost 497 KB per token, 48 times the hidden-state all-reduce. "
            "So keep the logits sharded. "
            "The ranks all-reduce two scalars for the softmax, and broadcast the chosen token."
        ) as t:
            t.wait_until("A norm reduces")
            self.play(FadeIn(cards[0], shift=UP * 0.1), run_time=0.7)
            t.wait_until("The input embedding")
            self.play(FadeIn(cards[1], shift=UP * 0.1), run_time=0.7)
            t.wait_until("The output projection")
            self.play(FadeIn(cards[2], shift=UP * 0.1), run_time=0.7)
            t.wait_until("Gathering its output")
            self.play(FadeIn(bars.labels[:2]), GrowFromEdge(bars.bars[0], LEFT), FadeIn(bars.values[0]),
                      GrowFromEdge(bars.bars[1], LEFT), FadeIn(bars.values[1]), run_time=1.0)
            t.wait_until("The ranks all-reduce")
            self.play(FadeIn(bars.labels[2]), GrowFromEdge(bars.bars[2], LEFT), FadeIn(bars.values[2]), run_time=0.8)

    # Communication volume -----------------------------------------------------

    def comm_volume(self):
        self.slide("comm-volume", "Per-rank traffic never passes 40 KiB", book_section="s-communication-volume")
        ax = kit.axes([0, 16, 2], [0, 45, 10], width=6.6, height=3.9,
                      x_label=r"\text{degree } P", y_label=r"\text{KiB per token per layer}",
                      x_numbers=[2, 4, 6, 8, 10, 12, 14, 16], y_numbers=[0, 10, 20, 30, 40])
        curve = ax.plot(lambda p: 40 * (p - 1) / p, x_range=[1, 16, 0.05], color=self.color,
                        stroke_width=th.STROKE_CURVE)
        cap = DashedLine(ax.c2p(0, 40), ax.c2p(16, 40), color=th.INK_3, stroke_width=th.STROKE_AXIS)
        cap_t = kit.label("40 KiB ceiling", size=th.SIZE_LABEL, color=th.INK_3).next_to(ax.c2p(16, 40), UP,
                                                                                        buff=0.1)
        cap_t.align_to(ax.c2p(16, 40), RIGHT)
        dots = VGroup()
        labs = VGroup()
        for p, txt in [(1, ""), (2, "20.0"), (4, "30.0"), (8, "35.0")]:
            v = 40 * (p - 1) / p
            d = Dot(ax.c2p(p, v), radius=0.08, color=self.color)
            lab = kit.label(txt or "0", size=th.SIZE_LABEL, color=th.INK, weight=MEDIUM).next_to(d, DR, buff=0.08)
            if not txt:
                lab.set_opacity(0)
            dots.add(d)
            labs.add(lab)
        plot = VGroup(ax, curve, cap, cap_t, dots, labs)
        s = kit.prose(r"$S = 5120 \times 2 = 10{,}240$ bytes, one token's hidden state", size=th.SIZE_SMALL,
                      width=4.6)
        f = kit.formula(r"B(P) = 2 \times 2\,\tfrac{P-1}{P} \times 10240", size=th.SIZE_MATH_SMALL)
        f2 = kit.formula(r"= 40960\,\tfrac{P-1}{P} \text{ bytes}", size=th.SIZE_MATH_SMALL)
        f2.next_to(f, DOWN, buff=0.2).align_to(f[0][4], LEFT)
        note = kit.prose(r"Not a bandwidth blow-up: a latency problem.", size=th.SIZE_SMALL, width=4.6)
        side = VGroup(s, VGroup(f, f2), note).arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        layout = VGroup(plot, side).arrange(RIGHT, buff=0.6)
        self.place(layout)
        with self.narrate(
            "Now count bytes. "
            "Per token, each all-reduce carries the hidden state: 5120 elements at 2 bytes, or 10 KiB. "
            "There are two all-reduces per layer, each paying the ring factor. "
            "So per token per layer, each rank sends 40960 bytes, times P minus 1 over P. "
            "At 2 ranks, that's 20 KiB. "
            "At 4, it's 30, and at 8, it's 35. "
            "However many ranks you add, it never passes 40 KiB. "
            "Tensor parallelism has a latency problem, not a bandwidth one."
        ) as t:
            self.play(Create(ax), FadeIn(s), run_time=0.9)
            t.wait_until("There are two")
            self.play(FadeIn(f), run_time=0.7)
            t.wait_until("So per token")
            self.play(FadeIn(f2), Create(curve), FadeIn(dots[0]), FadeIn(labs[0]), run_time=1.2)
            t.wait_until("At 2 ranks")
            self.play(FadeIn(dots[1], scale=0.5), FadeIn(labs[1]), run_time=0.5)
            t.wait_until("At 4, it's")
            self.play(FadeIn(dots[2], scale=0.5), FadeIn(labs[2]), run_time=0.5)
            self.play(FadeIn(dots[3], scale=0.5), FadeIn(labs[3]), run_time=0.5)
            t.wait_until("However many")
            self.play(Create(cap), FadeIn(cap_t), run_time=0.8)
            t.wait_until("Tensor parallelism has")
            self.play(FadeIn(note), run_time=0.6)

    def prefill(self):
        self.slide("prefill", "Prefill needs NVLink, not PCIe", book_section="s-communication-volume")
        vol = kit.formula(r"64 \times 4096 \times 20480 = 5.37 \text{ GB}", size=th.SIZE_MATH_SMALL)
        vcap = kit.label("a 4096-token prefill at P = 2, per rank", size=th.SIZE_LABEL, color=th.INK_3)
        bars = kit.bar_compare(
            [("Compute, 312 TFLOP/s each", 353, th.INK_3, "353 ms"),
             ("PCIe Gen4, 64 GB/s", 83.9, th.BAD, "83.9 ms: 24% of compute"),
             ("NVLink, 600 GB/s", 8.95, self.color, "8.95 ms: 2.5%")],
            width=5.4, bar_height=0.5, gap=0.32, size=th.SIZE_SMALL,
        )
        flops = kit.prose(r"$2 \times 26.9 \times 10^{9} \times 4096 = 2.20 \times 10^{14}$ FLOPs over two A100s",
                          size=th.SIZE_SMALL, width=None, color=th.INK_3)
        col = VGroup(VGroup(vcap, vol).arrange(DOWN, buff=0.15), bars, flops).arrange(DOWN, buff=0.6)
        self.place(col)
        with self.narrate(
            "Take a 4096-token prefill on 2 GPUs. "
            "Across 64 layers, each rank sends 5.37 GB. "
            "Prefill is limited by arithmetic, and on two A100 GPUs, the compute takes 353 ms. "
            "Over PCIe, the general-purpose bus, the transfer takes 83.9 ms, 24 percent of the compute. "
            "Over NVLink, the direct link between GPUs, it takes 8.95 ms, only 2.5 percent. "
            "The sharding that costs nothing over NVLink costs a quarter of your prefill over PCIe."
        ) as t:
            self.play(FadeIn(vcap), run_time=0.5)
            t.wait_until("Across 64 layers")
            self.play(FadeIn(vol), run_time=0.8)
            t.wait_until("Prefill is limited")
            self.play(FadeIn(flops), run_time=0.7)
            self.play(FadeIn(bars.labels[0]), GrowFromEdge(bars.bars[0], LEFT), FadeIn(bars.values[0]), run_time=0.9)
            t.wait_until("Over PCIe")
            self.play(FadeIn(bars.labels[1]), GrowFromEdge(bars.bars[1], LEFT), FadeIn(bars.values[1]), run_time=0.8)
            t.wait_until("Over NVLink")
            self.play(FadeIn(bars.labels[2]), GrowFromEdge(bars.bars[2], LEFT), FadeIn(bars.values[2]), run_time=0.8)

    def decode(self):
        self.slide("decode", "Decode pays a fixed 1 ms per step", book_section="s-communication-volume")
        rows = [(1, 42.2, 0.0, "42.2 ms   1.00×   100%"), (2, 21.1, 1.02, "22.1 ms   1.91×   96%"),
                (4, 10.55, 1.02, "11.6 ms   3.64×   91%"), (8, 5.27, 1.02, "6.3 ms   6.70×   84%"),
                (16, 2.64, 1.02, "3.6 ms   11.7×   73%")]
        u = 0.17
        h = 0.48
        names, reads, comms, vals = VGroup(), VGroup(), VGroup(), VGroup()
        for i, (p, w, c, txt) in enumerate(rows):
            y = -i * (h + 0.28)
            nm = kit.label(f"P = {p}", size=th.SIZE_SMALL, color=th.INK_2)
            nm.move_to([-0.3 - nm.width / 2, y, 0])
            r = seg(0, w * u, h, WEIGHTS).shift(UP * y)
            cc = seg(w * u, max(c, 0.001) * u, h, self.color, tint=0.8).shift(UP * y)
            v = kit.label(txt, size=th.SIZE_SMALL, color=th.INK, font=th.MONO)
            v.move_to([42.2 * u + 0.4 + v.width / 2, y, 0])
            names.add(nm)
            reads.add(r)
            comms.add(cc)
            vals.add(v)
        head = kit.label("step     speedup   efficiency", size=th.SIZE_LABEL, color=th.INK_3, font=th.MONO)
        head.next_to(vals[0], UP, buff=0.25).align_to(vals[0], LEFT)
        chart = VGroup(names, reads, comms, vals, head)
        key = VGroup(swatch("weight read: 42.2 / P ms", WEIGHTS),
                     swatch("128 all-reduces × 8 µs = 1.02 ms", self.color)).arrange(RIGHT, buff=0.7)
        col = VGroup(key, chart).arrange(DOWN, buff=0.5)
        self.place(col)
        with self.narrate(
            "In decode, the tensors are tiny, one token of 10 KiB. "
            "But each all-reduce costs 5 to 10 microseconds of latency, whatever its size. "
            "There are 128 of them per step, so at about 8 microseconds each, that's 1.02 ms. "
            "That barely changes with P, but the weight read shrinks: 42.2 ms, divided by P. "
            "At 2 GPUs, a step takes 22.1 ms, so decode is 1.91 times faster, not 2. "
            "At 16, the fixed term is 28 percent of the step, and efficiency falls to 73 percent. "
            "CUDA graphs shrink that fixed term, which is why production engines capture decode."
        ) as t:
            self.play(FadeIn(names), FadeIn(head), run_time=0.6)
            t.wait_until("There are 128")
            self.play(FadeIn(key[1]), run_time=0.6)
            t.wait_until("That barely changes")
            self.play(FadeIn(key[0]), LaggedStart(*[GrowFromEdge(r, LEFT) for r in reads], lag_ratio=0.2),
                      run_time=1.4)
            t.wait_until("At 2 GPUs")
            self.play(FadeIn(vals[0]), GrowFromEdge(comms[1], LEFT), FadeIn(vals[1]), run_time=0.7)
            self.play(GrowFromEdge(comms[2], LEFT), FadeIn(vals[2]), run_time=0.6)
            self.play(GrowFromEdge(comms[3], LEFT), FadeIn(vals[3]), run_time=0.6)
            t.wait_until("At 16, the fixed")
            self.play(GrowFromEdge(comms[4], LEFT), FadeIn(vals[4]), run_time=0.6)
            self.play(Indicate(comms[4], color=self.color, scale_factor=1.3), run_time=0.8)

    # Three kinds of parallelism -------------------------------------------------

    def three_ways(self):
        self.slide("three-ways", "Only tensor parallelism cuts decode latency",
                   book_section="s-tensor-pipeline-and-data-parallelism")
        a, b = self.gpu

        def gpu(text, color, sub=None):
            return kit.box(text, sublabel=sub, color=color, width=2.6, size=th.SIZE_LABEL)

        def column(title, top, bottom, link, lines, color):
            name = kit.label(title, size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
            pair = VGroup(top, bottom).arrange(DOWN, buff=0.9)
            notes = VGroup(*[kit.label(s, size=th.SIZE_LABEL, color=th.INK_2) for s in lines]).arrange(
                DOWN, buff=0.1, aligned_edge=LEFT)
            c = VGroup(name, pair, notes).arrange(DOWN, buff=0.35)
            c.pair, c.notes = pair, notes
            return c

        data = column("Data", gpu("full copy", REPLICA, "53.8 GB"), gpu("full copy", REPLICA, "53.8 GB"),
                      None, ["no communication", "throughput only"], REPLICA)
        pipe = column("Pipeline", gpu("layers 0 to 31", a), gpu("layers 32 to 63", b),
                      "42 MB, once", ["memory, not latency", "idle bubble in decode"], th.INK_3)
        tens = column("Tensor", gpu("half of every matrix", a), gpu("half of every matrix", b),
                      "128 all-reduces", ["memory and latency", "needs NVLink"], self.color)
        cols = VGroup(data, pipe, tens).arrange(RIGHT, buff=1.0, aligned_edge=UP)
        self.place(cols)
        cols.shift(LEFT * 0.35)
        p_arrow = kit.arrow(pipe.pair[0], pipe.pair[1], text="42 MB, once", side="left")
        t_arrow = VGroup(DoubleArrow(tens.pair[0].get_bottom(), tens.pair[1].get_top(), buff=0.08,
                                     color=self.color, stroke_width=4, tip_length=0.18))
        t_lab = kit.label("128 all-reduces", size=th.SIZE_LABEL, color=self.color)
        t_lab.next_to(t_arrow, RIGHT, buff=0.12)
        frame = SurroundingRectangle(VGroup(tens, t_lab), buff=0.2, corner_radius=0.12, color=self.color, stroke_width=3)
        with self.narrate(
            "There are three ways to use more than one GPU. "
            "Data parallelism gives each GPU a full copy and its own requests. "
            "It needs no communication, but each GPU still reads all 53.8 GB. "
            "Pipeline parallelism splits the layers between GPUs. "
            "At a 4096-token prefill, it sends 42 MB once, 128 times less than tensor parallelism. "
            "But in decode, each stage idles while the other works. "
            "Tensor parallelism alone cuts the bytes each GPU reads per token, so it alone cuts decode latency. "
            "So use tensor parallelism within a node, over NVLink, and the others across nodes and replicas."
        ) as t:
            t.wait_until("Data parallelism gives")
            self.play(FadeIn(data[0]), FadeIn(data.pair), run_time=0.7)
            t.wait_until("It needs no")
            self.play(FadeIn(data.notes), run_time=0.5)
            t.wait_until("Pipeline parallelism splits")
            self.play(FadeIn(pipe[0]), FadeIn(pipe.pair), run_time=0.7)
            t.wait_until("At a 4096-token")
            self.play(GrowArrow(p_arrow.line), FadeIn(p_arrow.label), FadeIn(pipe.notes[0]), run_time=0.7)
            t.wait_until("But in decode")
            self.play(FadeIn(pipe.notes[1]), Indicate(pipe.pair[1], color=th.INK_3, scale_factor=1.05), run_time=0.8)
            t.wait_until("Tensor parallelism alone")
            self.play(FadeIn(tens[0]), FadeIn(tens.pair), GrowFromCenter(t_arrow), FadeIn(t_lab),
                      FadeIn(tens.notes), run_time=0.9)
            t.wait_until("So use tensor")
            self.play(Create(frame), run_time=0.7)

    # Testing and bugs ------------------------------------------------------------

    def what_goes_wrong(self):
        self.slide("what-goes-wrong", "Test on one device; the bugs are silent", book_section="s-what-goes-wrong")
        test = kit.callout("note", r"Simulate every rank on one device. Concatenate column outputs, sum row "
                                   r"partials, and compare to $10^{-3}$ in bf16 or $10^{-4}$ in float32.",
                           title="Test without two GPUs", width=4.6, size=th.SIZE_SMALL)
        bugs = kit.bullets(
            [
                r"Concatenated partial sums: $P$ times too wide",
                r"Wrong rank order: silently wrong",
                r"Split by feature, not head: 1.5 heads at $P = 16$",
                r"Row bias on every rank: added $P$ times",
                r"Sharded norm: the wrong denominator",
                r"Different RNG state: sequences diverge",
            ],
            size=th.SIZE_SMALL, width=7.2, buff=0.24,
        )
        layout = VGroup(test, bugs).arrange(RIGHT, buff=0.6, aligned_edge=UP)
        self.place(layout)
        with self.narrate(
            "You don't need two GPUs to test the partitioning. "
            "Simulate every rank on one device, combine the shards, and compare with the unsharded layer, within "
            "a tolerance. "
            "Concatenating partial sums at least raises a shape error. "
            "Concatenating in the wrong rank order gives silently wrong results. "
            "Splitting by raw feature works at 2 ranks by luck, but at 16, each rank gets one and a half heads. "
            "A row-parallel bias added on every rank is added P times. "
            "A sharded norm divides by the wrong denominator. "
            "And a different random state on each rank picks different tokens."
        ) as t:
            t.wait_until("Simulate every rank")
            self.play(FadeIn(test, shift=UP * 0.1), run_time=0.8)
            for row, cue in zip(bugs, ["Concatenating partial", "Concatenating in the wrong", "Splitting by raw",
                                       "A row-parallel bias", "A sharded norm", "And a different"]):
                t.wait_until(cue)
                self.play(FadeIn(row, shift=RIGHT * 0.1), run_time=0.5)
