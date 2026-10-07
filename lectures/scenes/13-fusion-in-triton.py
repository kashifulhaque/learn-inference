"""Chapter 13, fusion in Triton.

Teaches content/chapters/13-fusion-in-triton.md in its order. Every number
below was recomputed in Python and matches the chapter (H = 5120, b = 2 bytes
for bfloat16, A100 L2 = 40 MB, decimal MB).

Masks and blocks:
- next_power_of_2(5120) = 8192; 8192 - 5120 = 3072 masked lanes. The lane
  strip draws 16 cells of 512 lanes: 10 live, 6 masked.
- next_power_of_2(4097) = 8192; 8192 - 4097 = 4095 masked, nearly half.
- Elements per lane, BLOCK = 8192: 8192 / (32 x 8) = 32; / (32 x 16) = 16;
  / (32 x 32) = 8. Lanes 256, 512, 1024. SwiGLU: 1024 / (32 x 4) = 8.
- Dividing by BLOCK: 5120 / 8192 = 0.625; 1 / sqrt(0.625) = 1.2649 (1.265,
  a 26.5 percent error).

The prediction:
- S = N H b = 4096 x 5120 x 2 = 41,943,040 B = 41.9 MB.
- Unfused 3S + 2S = 5S; fused 4S; 5S / 4S = 1.25; 1 - 4/5 = 20 percent.
- Weight vector H b = 10,240 B = 10.2 KB.

The measurement (tensor sizes N x 5120 x 2 B): 256 rows 2.6 MB, 1024 10.5,
2048 21.0, 4096 41.9, 8192 83.9, 16384 167.8 (4.19x L2, "4x"), 32768 335.5.
Sweep speedups from the chapter's times: 0.074/0.072 = 1.028 (1.02),
0.074/0.081 = 0.914 (0.91), 0.090/0.107 = 0.841 (0.84), 0.148/0.155 = 0.955
(0.95), 0.263/0.252 = 1.044 (chapter 1.05), 0.495/0.444 = 1.115 (1.12),
0.958/0.823 = 1.164 (1.16). Lab points: 0.92 at 4096 rows, 1.10 at 16384.
Plot x = log2(MB) - 1; L2 at log2(40) - 1 = 4.32.

The model, speedup = e (4 + phi) / 4:
- e = 0.92 / 1.00 = 0.92; 1.10 / 1.25 = 0.88; 1.16 / 1.25 = 0.928 (0.93).
- 0.90 x 1.00 = 0.90; 0.90 x 1.25 = 1.125 (1.13).
- Break-even: 1 / 0.90 = 1.111; 4 x 1.111 - 4 = 0.444, phi >= 0.44.
- First unfused kernel at S = 41.9 MB: 3S = 125.8 MB ("126 MB").

SwiGLU:
- 5S / 3S = 1.667; 1.52 / 1.667 = 0.912 (0.91).
- 4096 x 17408 x 2 = 142,606,336 B = 142.6 MB; / 40 MB = 3.57 (3.6x).
- 4096 x 17408 = 71,303,168 elements (71.3M); / 1024 = 69,632 programs
  exactly; / 108 SMs = 644.7 (645).

One block per row: tiled 3S against 2S, half as much again (50 percent).
0.089 vs 0.090 ms, within 1 percent (1.1 percent). Row of 5120 bf16 =
10,240 B = 10 KB. Tiles 2048 + 2048 + 1024 = 5120.
num_warps 8 / 16 / 32: 945 / 924 / 878 GB/s (chapter values; 2S / 0.0888 ms
= 944.7 GB/s).

Ceiling: 945 / 1275 = 0.741 (74 percent); 945 / 1935 = 0.488 (49 percent).
Copy: 2 x 41.9 MB / 0.066 ms = 1271 GB/s from the rounded time; chapter
states 1275.
"""

from manim import *

import kit
from kit import theme as th

PHI_C = th.HL_A  # \hla in the chapter: the HBM fraction
EFF_C = th.HL_B  # \hlb in the chapter: the efficiency

SOFTMAX = """\
@triton.jit
def _softmax_fwd(x_ptr, out_ptr, stride_row, n_cols,
                 BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    offs = row * stride_row + cols
    x = tl.load(x_ptr + offs, mask=mask, other=float("-inf"))
    x = x.to(tl.float32)
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    out = num / tl.sum(num, axis=0)
    tl.store(out_ptr + offs, out, mask=mask)
"""

FUSED = """\
row = tl.program_id(0)
offset = row * stride_row
cols = tl.arange(0, BLOCK)              # (8192,) int32
mask = cols < n_cols                    # 5120 true
x = tl.load(x_ptr + offset + cols, mask=mask, other=0.0)
res = tl.load(res_ptr + offset + cols, mask=mask, other=0.0)
total = x.to(tl.float32) + res.to(tl.float32)
tl.store(new_res_ptr + offset + cols, total, mask=mask)
var = tl.sum(total * total, axis=0) / n_cols
scale = 1.0 / tl.sqrt(var + eps)
w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
tl.store(out_ptr + offset + cols, total * scale * w, mask=mask)
"""

SWIGLU = """\
pid = tl.program_id(0)
offsets = pid * BLOCK + tl.arange(0, BLOCK)    # (1024,)
mask = offsets < n_elements
gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
up = tl.load(up_ptr + offsets, mask=mask, other=0.0)
silu = gate.to(tl.float32) * tl.sigmoid(gate.to(tl.float32))
tl.store(out_ptr + offsets, silu * up, mask=mask)
"""


# Helpers ----------------------------------------------------------------------


def panel(title, width, height, color=None, size=th.SIZE_SMALL, sub=None):
    """A titled region (a memory level, a language) with its label in the top-left corner."""
    stroke = th.RULE_STRONG if color is None else color
    rect = RoundedRectangle(width=width, height=height, corner_radius=0.12, stroke_color=stroke,
                            stroke_width=th.STROKE_BOX, fill_color=th.tint(stroke, 0.06), fill_opacity=1)
    parts = [kit.label(title, size=size, color=th.INK, weight=MEDIUM)]
    if sub:
        parts.append(kit.label(sub, size=th.SIZE_LABEL, color=th.INK_3))
    head = VGroup(*parts).arrange(RIGHT, buff=0.2, aligned_edge=DOWN)
    head.move_to(rect.get_corner(UL) + RIGHT * (head.width / 2 + 0.2) + DOWN * (head.height / 2 + 0.16))
    group = VGroup(rect, head)
    group.rect, group.head = rect, head
    return group


def chip(text, color, w=1.4, h=0.62, size=th.SIZE_SMALL, tint=0.3, mono=True):
    """A tensor in memory: a tinted rectangle with its name in the code font."""
    rect = RoundedRectangle(width=w, height=h, corner_radius=0.06, stroke_color=color, stroke_width=th.STROKE_BOX,
                            fill_color=th.tint(color, tint), fill_opacity=1)
    t = kit.label(text, size=size, color=th.INK, font=th.MONO if mono else th.SANS)
    if t.width > w * 0.88:
        t.scale_to_fit_width(w * 0.88)
    t.move_to(rect)
    g = VGroup(rect, t)
    g.rect, g.text = rect, t
    return g


def cells(n, w, h, color=None, tint=0.35, gap=0.0):
    """A row of `n` cells: lanes of a block, or elements in memory."""
    fill = th.RAISE if color is None else th.tint(color, tint)
    row = VGroup(*[Rectangle(width=w, height=h, stroke_color=th.RULE_STRONG, stroke_width=th.STROKE_HAIRLINE,
                             fill_color=fill, fill_opacity=1) for _ in range(n)])
    row.arrange(RIGHT, buff=gap)
    return row


def flow(a, b, color=th.INK_3, stroke=3):
    """A plain arrow between two points."""
    return Arrow(a, b, buff=0.04, color=color, stroke_width=stroke, tip_length=0.18,
                 max_tip_length_to_length_ratio=0.3, max_stroke_width_to_length_ratio=20)


def packet(color):
    return Square(0.2, stroke_width=0, fill_color=color, fill_opacity=1)


def swatch(text, color, tint=0.35):
    sq = Square(0.28, stroke_color=th.RULE_STRONG, stroke_width=th.STROKE_HAIRLINE,
                fill_color=th.tint(color, tint), fill_opacity=1)
    return VGroup(sq, kit.label(text, size=th.SIZE_LABEL, color=th.INK_2)).arrange(RIGHT, buff=0.15)


class Lecture(kit.Lecture):
    def construct(self):
        self.title_slide(
            narration=(
                "Fusion in Triton. "
                "In this lecture, you write GPU kernels one block at a time, and fuse two of them to move fewer bytes. "
                "Then you find out why the byte count over-promises when the data fits in the cache."
            )
        )
        self.hook()
        self.block_not_thread()
        self.softmax_kernel()
        self.programs_over_rows()
        self.ragged_edge()
        self.constexpr_warps()
        self.triton_trade()
        self.unfused_traffic()
        self.fused_traffic()
        self.fused_kernel()
        self.measurement()
        self.l2_absorbs()
        self.two_number_model()
        self.crossover()
        self.swiglu_traffic()
        self.swiglu_kernel()
        self.one_block_per_row()
        self.honest_measurement()
        self.when_not_to_fuse()
        self.correctness()
        self.closing_slide(
            [
                r"Triton describes one block; mask every load and store.",
                r"RMSNorm fusion: $5S \to 4S$, a $1.25\times$ prediction.",
                r"Measured: about $0.90 \times (4 + \varphi)/4$.",
                r"SwiGLU's dead intermediate: $5S \to 3S$, $1.52\times$.",
            ],
            narration=(
                "Here's what to keep. "
                "A Triton kernel describes one block, and every load and store needs a mask. "
                "Fusing RMSNorm with its residual add cuts traffic from 5S to 4S. "
                "The measured speedup is about 0.90 times the traffic term, and the cache decides how much of the "
                "round trip was real. "
                "SwiGLU's intermediate is dead, so it pays 1.52 times. "
                "Next, FlashAttention fuses attention so the score table never reaches memory at all."
            ),
            book_section="s-what-goes-wrong",
        )

    # ------------------------------------------------------------------
    # Opening

    def hook(self):
        self.slide("fewer-bytes", "Fewer bytes should mean faster", book_section="s-before-you-start")
        k1 = kit.box("add", color=self.colors[1], width=2.6, height=0.85)
        k2 = kit.box("RMSNorm", color=self.colors[1], width=2.6, height=0.85)
        pair = VGroup(k1, k2).arrange(RIGHT, buff=2.2)
        bars = kit.bar_compare(
            [("Byte count predicts", 1.25, th.INK_4, "1.25×"),
             ("A100, 41.9 MB tensor", 0.92, self.color, "0.92×"),
             ("A100, 167.8 MB tensor", 1.10, self.color, "1.10×")],
            max_value=1.3, width=6.2, bar_height=0.48, gap=0.3, limit=1.0, limit_label="no speedup",
        )
        cache = kit.pill("A100 L2 cache: 40 MB", th.INFO, size=th.SIZE_SMALL)
        layout = VGroup(pair, bars, cache).arrange(DOWN, buff=0.65)
        self.place(layout)
        link = flow(k1.get_right(), k2.get_left())
        link_lab = kit.label("via HBM", size=th.SIZE_LABEL, color=th.INK_3).next_to(link, UP, buff=0.1)
        fused = kit.box("add + RMSNorm, one kernel", color=self.color, width=pair.width, height=0.85)
        fused.move_to(pair)
        with self.narrate(
            "A memory-bound kernel at full bandwidth only gets faster by moving fewer bytes. "
            "Fusion merges two kernels into one, so the result passed between them never leaves the chip. "
            "Counting bytes says that fusing RMSNorm with its residual add runs 1.25 times faster. "
            "On the A100, with a 42 MB tensor, it runs at 0.92 times, which is slower. "
            "With a 168 MB tensor, it runs at 1.10 times. "
            "The gap comes from the GPU's cache, which holds 40 MB."
        ) as t:
            self.play(FadeIn(k1), FadeIn(k2), GrowArrow(link), FadeIn(link_lab), run_time=0.8)
            t.wait_until("Fusion merges")
            self.play(ReplacementTransform(VGroup(k1, k2, link, link_lab), fused), run_time=1.0)
            t.wait_until("Counting bytes")
            self.play(FadeIn(bars.labels[0]), FadeIn(bars.limit), GrowFromEdge(bars.bars[0], LEFT),
                      FadeIn(bars.values[0]), run_time=0.9)
            t.wait_until("On the A100")
            self.play(FadeIn(bars.labels[1]), GrowFromEdge(bars.bars[1], LEFT), FadeIn(bars.values[1]), run_time=0.8)
            t.wait_until("With a 168 MB")
            self.play(FadeIn(bars.labels[2]), GrowFromEdge(bars.bars[2], LEFT), FadeIn(bars.values[2]), run_time=0.8)
            t.wait_until("The gap comes")
            self.play(FadeIn(cache, scale=0.9), run_time=0.6)

    # ------------------------------------------------------------------
    # The Triton programming model

    def block_not_thread(self):
        self.slide("block-not-thread", "In Triton, you write what one block does",
                   book_section="s-the-triton-programming-model")
        cuda = panel("CUDA", 6.3, 3.3, color=self.colors[1])
        tri = panel("Triton", 6.3, 3.3, color=self.color)
        panels = VGroup(cuda, tri).arrange(RIGHT, buff=0.3)
        lanes_c = cells(8, 0.6, 0.6, gap=0.08)
        lanes_t = cells(8, 0.6, 0.6, gap=0.08)
        lanes_c.move_to(cuda.rect.get_center() + DOWN * 0.1)
        lanes_t.move_to(tri.rect.get_center() + DOWN * 0.1)
        tids = VGroup(*[kit.label(f"t{i}", size=th.SIZE_LABEL, color=th.INK_3, font=th.MONO).move_to(c)
                        for i, c in enumerate(lanes_c)])
        idx = kit.mono("i = blockIdx.x*blockDim.x + threadIdx.x", size=th.SIZE_LABEL, color=th.INK_2)
        if idx.width > cuda.rect.width - 0.7:
            idx.scale_to_fit_width(cuda.rect.width - 0.7)
        idx.next_to(lanes_c, DOWN, buff=0.35)
        c_note = kit.label("you write one thread", size=th.SIZE_SMALL, color=th.INK_2)
        c_note.next_to(lanes_c, UP, buff=0.25)
        brace = Brace(lanes_t, DOWN, buff=0.12, color=th.INK_3)
        vec = kit.mono("x: BLOCK values", size=th.SIZE_LABEL, color=th.INK_2).next_to(brace, DOWN, 0.1)
        t_note = kit.label("you write one block", size=th.SIZE_SMALL, color=th.INK_2)
        t_note.next_to(lanes_t, UP, buff=0.25)
        jobs = ["assigns lanes", "picks vector widths", "allocates registers", "stages shared memory",
                "inserts barriers"]
        pills = VGroup(*[kit.pill(j, self.color, size=th.SIZE_SMALL, filled=False) for j in jobs])
        prow = VGroup(VGroup(*pills[:3]).arrange(RIGHT, buff=0.25), VGroup(*pills[3:]).arrange(RIGHT, buff=0.25))
        prow.arrange(DOWN, buff=0.2)
        comp = kit.label("The Triton compiler", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
        bottom = VGroup(comp, prow).arrange(DOWN, buff=0.25)
        layout = VGroup(panels, lanes_c, lanes_t, tids, idx, c_note, brace, vec, t_note, bottom)
        bottom.next_to(panels, DOWN, buff=0.45)
        self.place(layout)
        with self.narrate(
            "The two languages differ in the unit you describe. "
            "In CUDA, you write what one thread does. "
            "The block exists only in your index arithmetic, and the barriers are yours to place. "
            "In Triton, you write what one block does. "
            "Your code works on whole vectors, of a length fixed at compile time. "
            "The compiler then assigns lanes and picks vector widths. "
            "It allocates registers, stages shared memory, and inserts the barriers. "
            "Triton hides the thread level, but not the hardware."
        ) as t:
            self.play(FadeIn(cuda), FadeIn(tri), FadeIn(lanes_c), FadeIn(tids), FadeIn(lanes_t), run_time=0.8)
            t.wait_until("In CUDA, you write")
            self.play(lanes_c[3].animate.set_fill(th.tint(self.colors[1], 0.5)), FadeIn(c_note), run_time=0.6)
            t.wait_until("The block exists")
            self.play(FadeIn(idx, shift=UP * 0.1), run_time=0.6)
            t.wait_until("In Triton, you write")
            self.play(*[c.animate.set_fill(th.tint(self.color, 0.4)) for c in lanes_t], FadeIn(t_note), run_time=0.7)
            t.wait_until("Your code works")
            self.play(FadeIn(brace), FadeIn(vec), run_time=0.6)
            t.wait_until("The compiler then")
            self.play(FadeIn(comp), FadeIn(pills[0]), FadeIn(pills[1]), run_time=0.7)
            t.wait_until("It allocates")
            self.play(LaggedStart(*[FadeIn(p, shift=UP * 0.1) for p in pills[2:]], lag_ratio=0.3), run_time=1.0)

    def softmax_kernel(self):
        self.slide("softmax-kernel", "A whole softmax, with no threads in sight",
                   book_section="s-the-triton-programming-model")
        code = kit.code(SOFTMAX, "python", caption="softmax_triton.py, condensed")
        self.place(code)
        with self.narrate(
            "Here's a complete softmax kernel. "
            "It has no thread index, no shared memory, and no barrier, and it handles any row width. "
            "The program ID is CUDA's block index, so each program owns one row. "
            "The next line builds a vector of column indices. "
            "Adding it to a pointer gives a vector of pointers, one per column. "
            "The load reads the whole row, and the cast moves it to float32. "
            "A max and a sum each reduce the vector to a scalar. "
            "The store writes the row out, under the same mask."
        ) as t:
            self.play(FadeIn(code.frame), *code.reveal(1, 2, 3), run_time=0.8)
            t.wait_until("It has no thread")
            self.play(*code.reveal(4, 5, 6, 7, 8, 9, 10, 11, 12, 13), run_time=0.9)
            t.wait_until("The program ID is")
            self.play(code.highlight(4), run_time=0.5)
            t.wait_until("The next line")
            self.play(code.highlight(5), run_time=0.5)
            t.wait_until("Adding it")
            self.play(code.highlight(7), run_time=0.5)
            t.wait_until("The load reads")
            self.play(code.highlight(8, 9), run_time=0.5)
            t.wait_until("A max and a sum")
            self.play(code.highlight(10, 12), run_time=0.5)
            t.wait_until("The store writes")
            self.play(code.highlight(13), run_time=0.5)

    def programs_over_rows(self):
        self.slide("programs-over-rows", "The grid launches one program per row",
                   book_section="s-the-triton-programming-model")
        n_rows, n_cols = 6, 16
        g = kit.grid(n_rows, n_cols, cell=(0.5, 0.44), row_labels=[f"program {r}" for r in range(n_rows)])
        launch = kit.mono("_softmax_fwd[(n_rows,)](x, out, x.stride(0), n_cols, ...)", size=th.SIZE_LABEL,
                          color=th.INK)
        brace = Brace(VGroup(*g.cells[n_rows - 1]), DOWN, buff=0.06, color=th.INK_4)
        b_lab = kit.label("stride_row: elements from one row start to the next", size=th.SIZE_LABEL,
                          color=th.INK_3, font=th.MONO)
        b_lab.next_to(brace, DOWN, buff=0.06)
        addr = kit.mono("x_ptr + row * stride_row + cols", size=th.SIZE_SMALL, color=th.INK)
        scalar = chip("max", self.color, w=1.2, h=0.5, size=th.SIZE_LABEL)
        fig = VGroup(g, brace, b_lab)
        layout = VGroup(launch, fig, addr).arrange(DOWN, buff=0.45)
        scalar.next_to(g, RIGHT, buff=0.9)
        scalar.set_y(g.cell(2, 0).get_y())
        layout.add(scalar)
        self.place(layout)
        idx = VGroup(*[kit.label(str(c), size=th.SIZE_LABEL - 2, color=th.INK, font=th.MONO).move_to(g.cell(2, c))
                       for c in range(n_cols)])
        idx.set_z_index(3)
        ring = g.outline(2, 3, 0, n_cols, color=self.color)
        red = flow(g.cell(2, n_cols - 1).get_right() + RIGHT * 0.05, scalar.get_left(), color=self.color)
        shades = [0.18, 0.4]
        with self.narrate(
            "You choose the grid at launch, in square brackets. "
            "One entry per row launches one program per row, numbered from zero. "
            "Each program finds its row by multiplying its ID by the row stride. "
            "You pass strides in yourself, because the kernel does its own address arithmetic. "
            "Inside a program, every value is a scalar or a vector of the block's length. "
            "A reduction turns the vector into a scalar."
        ) as t:
            self.play(FadeIn(launch), FadeIn(g), run_time=0.8)
            t.wait_until("One entry per row")
            self.play(LaggedStart(*[g.fill(g.row(r), self.color, tint=shades[r % 2]) for r in range(n_rows)],
                                  lag_ratio=0.25), run_time=1.6)
            t.wait_until("Each program finds")
            self.play(Create(ring), FadeIn(brace), FadeIn(b_lab), run_time=0.8)
            t.wait_until("You pass strides")
            self.play(FadeIn(addr, shift=UP * 0.1), run_time=0.6)
            t.wait_until("Inside a program")
            self.play(LaggedStart(*[FadeIn(i) for i in idx], lag_ratio=0.05), run_time=1.0)
            t.wait_until("A reduction turns")
            self.play(GrowArrow(red), FadeIn(scalar, shift=RIGHT * 0.2), run_time=0.8)

    def ragged_edge(self):
        self.slide("ragged-edge", "Masks guard the ragged edge",
                   book_section="s-the-triton-programming-model")
        live, dead = 10, 6
        w, h = 0.62, 0.55
        lanes = cells(live + dead, w, h)
        mem = cells(live + dead, w, h)
        for i, sq in enumerate(mem):
            sq.set_fill(th.tint(self.color if i < live else self.colors[1], 0.25), opacity=1)
        mem.next_to(lanes, DOWN, buff=1.25)
        l_lab = kit.label("lanes of one block", size=th.SIZE_LABEL, color=th.INK_3)
        l_lab.next_to(lanes, LEFT, buff=0.3)
        m_lab = kit.label("memory", size=th.SIZE_LABEL, color=th.INK_3)
        m_lab.next_to(mem, LEFT, buff=0.3)
        b_live = Brace(VGroup(*lanes[:live]), UP, buff=0.08, color=th.INK_4)
        t_live = kit.label("5120 live lanes", size=th.SIZE_LABEL, color=th.INK_2).next_to(b_live, UP, buff=0.06)
        b_dead = Brace(VGroup(*lanes[live:]), UP, buff=0.08, color=th.INK_4)
        t_dead = kit.label("3072 past the row", size=th.SIZE_LABEL, color=th.INK_2).next_to(b_dead, UP, buff=0.06)
        r_lab = kit.label("row r", size=th.SIZE_LABEL, color=th.INK_2).next_to(VGroup(*mem[:live]), DOWN, buff=0.12)
        n_lab = kit.label("row r + 1", size=th.SIZE_LABEL, color=th.INK_2).next_to(VGroup(*mem[live:]), DOWN,
                                                                                  buff=0.12)
        n_lab.align_to(r_lab, DOWN)
        tab = kit.table(
            [["Softmax", "maximum", r"$-\infty$, which never wins"],
             ["RMSNorm", "sum of squares", r"0.0, which adds nothing"]],
            header=["Kernel", "Reduction", "Fill for masked lanes"],
            size=th.SIZE_SMALL,
        )
        fig = VGroup(lanes, mem, l_lab, m_lab, b_live, t_live, b_dead, t_dead, r_lab, n_lab)
        layout = VGroup(fig, tab).arrange(DOWN, buff=0.55)
        self.place(layout)
        reads = VGroup(*[flow(lanes[i].get_bottom(), mem[i].get_top(), color=th.BAD) for i in range(live + dead)])
        gap_y = (lanes.get_bottom()[1] + mem.get_top()[1]) / 2
        bad = kit.label("no mask: reads row r + 1, and nothing faults", size=th.SIZE_LABEL, color=th.BAD)
        bad.move_to([0, gap_y, 0]).align_to(lanes[live - 1], RIGHT).shift(LEFT * 0.15)
        ok = kit.label("masked: no access", size=th.SIZE_LABEL, color=th.INK_3)
        ok.move_to([VGroup(*lanes[live:]).get_x(), gap_y, 0])
        with self.narrate(
            "The block size must be a power of two. "
            "This model's rows are 5120 wide, so the block rounds up to 8192. "
            "That leaves 3072 lanes past the end of the row. "
            "Without a mask, they read the next row, and nothing faults. "
            "A missing mask on the store corrupts it. "
            "The mask switches those lanes off: they skip memory and take a fill value. "
            "Make the fill the identity of the reduction that follows. "
            "That's minus infinity for a maximum, and zero for a sum."
        ) as t:
            self.play(FadeIn(lanes), FadeIn(l_lab), FadeIn(mem), FadeIn(m_lab), FadeIn(r_lab), FadeIn(n_lab),
                      run_time=0.8)
            t.wait_until("This model's rows")
            self.play(*[sq.animate.set_fill(th.tint(self.color, 0.4)) for sq in lanes[:live]],
                      FadeIn(b_live), FadeIn(t_live), run_time=0.7)
            t.wait_until("That leaves 3072")
            self.play(*[sq.animate.set_fill(th.tint(th.INK_4, 0.3)) for sq in lanes[live:]],
                      FadeIn(b_dead), FadeIn(t_dead), run_time=0.7)
            t.wait_until("Without a mask, they")
            self.play(LaggedStart(*[GrowArrow(a) for a in reads[live:]], lag_ratio=0.1), FadeIn(bad), run_time=1.0)
            t.wait_until("The mask switches")
            self.play(FadeOut(reads[live:]), FadeOut(bad), FadeIn(ok), run_time=0.8)
            t.wait_until("Make the fill")
            self.play(FadeIn(tab.rules), FadeIn(VGroup(*tab.cells[0])), run_time=0.6)
            t.wait_until("That's minus infinity")
            self.play(FadeIn(VGroup(*tab.cells[1]), shift=RIGHT * 0.1), run_time=0.5)
            self.play(FadeIn(VGroup(*tab.cells[2]), shift=RIGHT * 0.1), run_time=0.5)

    def constexpr_warps(self):
        self.slide("compile-time-block", "The block size is fixed when the kernel compiles",
                   book_section="s-the-triton-programming-model")
        line = NumberLine(x_range=[4096, 8192, 1024], length=5.6, color=th.INK_3, stroke_width=th.STROKE_AXIS,
                          include_tip=False, tick_size=0.06)
        ends = VGroup(kit.label("4097", size=th.SIZE_LABEL, color=th.INK_3).next_to(line.n2p(4097), DOWN, 0.15),
                      kit.label("8192", size=th.SIZE_LABEL, color=th.INK_3).next_to(line.n2p(8192), DOWN, 0.15))
        widths = [4097, 4800, 5120, 6000, 7300, 8192]
        dots = VGroup(*[Dot(line.n2p(v), radius=0.08, color=self.colors[1]) for v in widths])
        l_head = kit.label("row widths a kernel might see", size=th.SIZE_LABEL, color=th.INK_3)
        target = kit.box("one compile: BLOCK = 8192", color=self.color, size=th.SIZE_SMALL, font=th.MONO)
        left = VGroup(l_head, VGroup(line, ends, dots), target).arrange(DOWN, buff=0.45)
        f = kit.formula(r"\text{elements per lane} = \frac{\texttt{BLOCK}}{32 \times \texttt{num\_warps}}",
                        size=th.SIZE_MATH_SMALL)
        tab = kit.table([["8", "256", "32"], ["16", "512", "16"], ["32", "1024", "8"]],
                        header=["num_warps", "Lanes", "Elements per lane"], align=["c", "c", "c"],
                        size=th.SIZE_SMALL)
        right = VGroup(f, tab).arrange(DOWN, buff=0.45)
        top = VGroup(left, right).arrange(RIGHT, buff=1.0, aligned_edge=UP)
        auto = kit.callout("key", r"Autotune when the shapes vary. The engine runs one hidden size, so it measured "
                                  r"once and hardcodes 8 warps.", title="Autotuning", width=12.0, size=th.SIZE_SMALL)
        layout = VGroup(top, auto).arrange(DOWN, buff=0.5)
        self.place(layout)
        with self.narrate(
            "A compile-time block size lets the compiler size registers exactly, at one compile per value. "
            "Rounding up to the next power of two puts every width from 4097 to 8192 on one compile. "
            "The number of warps sets each lane's share of the block. "
            "With 8 warps, each lane handles 32 elements. "
            "More warps add parallelism inside a row, but fewer rows fit on an SM. "
            "Autotune when shapes vary; the engine runs one size, so it hardcodes 8 warps."
        ) as t:
            self.play(FadeIn(l_head), Create(line), FadeIn(ends), run_time=0.8)
            self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.15), run_time=0.8)
            t.wait_until("Rounding up")
            self.play(*[d.animate.move_to(target.get_top() + DOWN * 0.05) for d in dots], run_time=1.0)
            self.play(FadeOut(dots), FadeIn(target, scale=0.9), run_time=0.5)
            t.wait_until("The number of warps")
            self.play(FadeIn(f), FadeIn(tab.rules), FadeIn(VGroup(*tab.cells[0])), run_time=0.8)
            t.wait_until("With 8 warps")
            self.play(FadeIn(VGroup(*tab.cells[1]), shift=RIGHT * 0.1), run_time=0.5)
            t.wait_until("More warps add")
            self.play(FadeIn(VGroup(*tab.cells[2]), shift=RIGHT * 0.1), FadeIn(VGroup(*tab.cells[3]),
                                                                                shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("Autotune when")
            self.play(FadeIn(auto, shift=UP * 0.1), run_time=0.7)

    def triton_trade(self):
        self.slide("triton-trade", "Triton gives up the last 10% for brevity",
                   book_section="s-the-triton-programming-model")
        tab = kit.table(
            [
                ["Unit you write", "One thread", "One block"],
                ["Tail handling", r"\texttt{if (i < n)}", r"A mask on load and store"],
                ["Reductions", "Shared memory, tree, barriers", r"\texttt{tl.sum}, \texttt{tl.max}"],
                ["Barriers", "You place them", "The compiler inserts them"],
                ["Warp primitives", "Shuffles, ballots", "Not exposed"],
            ],
            header=["", "CUDA", "Triton"],
            size=th.SIZE_SMALL,
        )
        bars = kit.bar_compare(
            [("RMSNorm in CUDA", 35, self.colors[1], "about 35 lines"),
             ("RMSNorm in Triton", 12, self.color, "12 lines, any width")],
            width=4.5, bar_height=0.45, gap=0.28,
        )
        layout = VGroup(tab, bars).arrange(DOWN, buff=0.6)
        self.place(layout)
        with self.narrate(
            "Triton isn't magic, so here's the trade. "
            "Tails become masks, and reductions become one call each. "
            "The compiler places barriers, and warp-level primitives aren't exposed. "
            "The RMSNorm from chapter 12 took about 35 lines of CUDA. "
            "The Triton version takes 12, and handles any width. "
            "What you lose is the last 10 percent, and the means to diagnose it."
        ) as t:
            self.play(FadeIn(tab.rules), FadeIn(VGroup(*tab.cells[0])), FadeIn(VGroup(*tab.cells[1])), run_time=0.8)
            t.wait_until("Tails become")
            self.play(FadeIn(VGroup(*tab.cells[2])), FadeIn(VGroup(*tab.cells[3])), run_time=0.7)
            t.wait_until("The compiler places")
            self.play(FadeIn(VGroup(*tab.cells[4])), FadeIn(VGroup(*tab.cells[5])), run_time=0.7)
            t.wait_until("The RMSNorm from chapter")
            self.play(FadeIn(bars.labels[0]), GrowFromEdge(bars.bars[0], LEFT), FadeIn(bars.values[0]), run_time=0.8)
            t.wait_until("The Triton version")
            self.play(FadeIn(bars.labels[1]), GrowFromEdge(bars.bars[1], LEFT), FadeIn(bars.values[1]), run_time=0.8)

    # ------------------------------------------------------------------
    # The prediction

    def traffic_figure(self, chips, kernels, w=1.55):
        """Kernels on a row above an HBM panel holding tensor chips."""
        hbm = panel("HBM", len(chips) * (w + 0.35) + 0.6, 1.55, color=self.colors[1], size=th.SIZE_LABEL)
        row = VGroup(*[chip(n, c, w=w) for n, c in chips]).arrange(RIGHT, buff=0.35)
        row.move_to(hbm.rect.get_center() + DOWN * 0.18)
        ks = VGroup(*[kit.box(name, color=self.color, width=wk, height=0.8) for name, wk in kernels])
        ks.arrange(RIGHT, buff=1.0)
        ks.next_to(hbm, UP, buff=1.25)
        fig = VGroup(hbm, row, ks)
        fig.hbm, fig.chips, fig.kernels = hbm, row, ks
        return fig

    def link(self, fig, src, dst, dx_src=0.0, dx_dst=0.0, color=th.INK_3):
        """An arrow between a chip (index into fig.chips) and a kernel, either direction."""
        def pt(m, top, dx):
            return (m.get_top() if top else m.get_bottom()) + RIGHT * dx
        if isinstance(src, int):
            a = pt(fig.chips[src], True, dx_src)
            b = pt(dst, False, dx_dst)
        else:
            a = pt(src, False, dx_src)
            b = pt(fig.chips[dst], True, dx_dst)
        return flow(a, b, color=color)

    def send(self, arrows, color, run_time=0.8):
        dots = [packet(color).move_to(a.get_start()) for a in arrows]
        self.play(*[GrowArrow(a) for a in arrows], run_time=0.4)
        self.play(*[d.animate.move_to(a.get_end()) for d, a in zip(dots, arrows)], run_time=run_time)
        self.remove(*dots)

    def unfused_traffic(self):
        self.slide("unfused-traffic", "Unfused, the sum makes a round trip",
                   book_section="s-the-prediction-fusion-saves-a-fifth-of-the-bytes")
        fig = self.traffic_figure(
            [("x", self.colors[1]), ("residual", self.colors[1]), ("total", self.color), ("out", self.colors[1])],
            [("add", 2.6), ("RMSNorm", 2.6)],
        )
        add, norm = fig.kernels
        S = kit.formula(r"S = N H b = 4096 \times 5120 \times 2 = 41.9\ \text{MB}", size=th.SIZE_MATH_SMALL)
        tally = kit.table([["add", "3S"], ["RMSNorm", "2S"], [r"\textbf{Total}", r"\textbf{5S}"]],
                          header=["Kernel", "Traffic"], align=["l", "r"], size=th.SIZE_SMALL)
        right = tally
        body = VGroup(fig, right).arrange(RIGHT, buff=0.9)
        layout = VGroup(S, body).arrange(DOWN, buff=0.5)
        self.place(layout)
        # Kernels sit over the chips they touch.
        add.set_x((fig.chips[0].get_x() + fig.chips[2].get_x()) / 2 - 0.2)
        norm.set_x((fig.chips[2].get_x() + fig.chips[3].get_x()) / 2 + 0.5)
        r_x = self.link(fig, 0, add, dx_dst=-0.6)
        r_res = self.link(fig, 1, add, dx_dst=-0.1)
        w_tot = self.link(fig, add, 2, dx_src=0.7, dx_dst=-0.25)
        r_tot = self.link(fig, 2, norm, dx_src=0.25, dx_dst=-0.5)
        w_out = self.link(fig, norm, 3, dx_src=0.5)
        trip = kit.label("round trip", size=th.SIZE_LABEL, color=th.BAD, weight=MEDIUM)
        trip.next_to(fig.hbm, DOWN, buff=0.15).set_x(fig.chips[2].get_x())
        with self.narrate(
            "Now count bytes. "
            "Write S for one activation tensor: rows, times hidden size, times 2 bytes. "
            "At 4096 rows, S is 41.9 MB. "
            "Unfused, PyTorch runs two kernels. "
            "The add reads x and the residual, and writes their sum: 3S. "
            "The norm reads the sum back and writes the output: 2S. "
            "That's 5S in total. "
            "The sum goes out to memory, and the next kernel reads it straight back."
        ) as t:
            t.wait_until("Write S")
            self.play(FadeIn(S), run_time=0.8)
            t.wait_until("Unfused, PyTorch")
            self.play(FadeIn(fig.hbm), FadeIn(fig.chips), FadeIn(fig.kernels), FadeIn(tally.rules),
                      FadeIn(VGroup(*tally.cells[0])), run_time=0.8)
            t.wait_until("The add reads")
            self.send([r_x, r_res], self.colors[1])
            self.send([w_tot], self.color)
            self.play(FadeIn(VGroup(*tally.cells[1])), run_time=0.4)
            t.wait_until("The norm reads")
            self.send([r_tot], self.color)
            self.send([w_out], self.colors[1])
            self.play(FadeIn(VGroup(*tally.cells[2])), run_time=0.4)
            t.wait_until("That's 5S")
            self.play(FadeIn(VGroup(*tally.cells[3])), run_time=0.5)
            t.wait_until("The sum goes out")
            self.play(w_tot.animate.set_color(th.BAD), r_tot.animate.set_color(th.BAD), FadeIn(trip), run_time=0.7)

    def fused_traffic(self):
        self.slide("fused-traffic", "Fused, the sum stays in registers",
                   book_section="s-the-prediction-fusion-saves-a-fifth-of-the-bytes")
        fig = self.traffic_figure(
            [("x", self.colors[1]), ("residual", self.colors[1]), ("new_res", self.colors[1]),
             ("out", self.colors[1])],
            [("", 7.0)], w=1.75,
        )
        k = fig.kernels[0]
        k.set_x(fig.hbm.get_x())
        name = kit.label("fused add + RMSNorm", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
        reg = chip("total: registers", self.color, w=2.7, h=0.5, size=th.SIZE_LABEL)
        inner = VGroup(name, reg).arrange(RIGHT, buff=0.4)
        inner.move_to(k.rect)
        k.rect.stretch_to_fit_width(inner.width + 0.6)
        bars = kit.bar_compare([("Unfused", 5, th.INK_4, "5S"), ("Fused", 4, self.color, "4S")],
                               width=3.4, bar_height=0.45, gap=0.28)
        ratio = kit.formula(r"\frac{5S}{4S} = 1.25", size=th.SIZE_MATH)
        box = SurroundingRectangle(ratio, color=self.color, buff=0.16, corner_radius=0.08, stroke_width=3)
        side = VGroup(bars, VGroup(ratio, box)).arrange(DOWN, buff=0.6)
        fig.add(inner)
        layout = VGroup(fig, side).arrange(RIGHT, buff=0.8)
        self.place(layout)
        ins = [self.link(fig, 0, k, dx_dst=-1.6), self.link(fig, 1, k, dx_dst=-0.6)]
        outs = [self.link(fig, k, 2, dx_src=0.6), self.link(fig, k, 3, dx_src=1.6)]
        w_note = kit.label("weight: 10.2 KB, left out", size=th.SIZE_LABEL, color=th.INK_3)
        w_note.next_to(fig.hbm, DOWN, buff=0.2).align_to(fig.hbm, LEFT)
        with self.narrate(
            "Fuse them, and one kernel does the same work. "
            "It still writes the sum, because the next layer's residual needs it. "
            "What disappears is the read back: the sum stays in registers for the normalization. "
            "That's 4S instead of 5S. "
            "The weight vector is only 10 KB, too small to count. "
            "So the prediction is 1.25 times faster, at every size."
        ) as t:
            self.play(FadeIn(fig.hbm), FadeIn(fig.chips), FadeIn(k), FadeIn(name), run_time=0.8)
            self.send(ins, self.colors[1])
            t.wait_until("It still writes")
            self.send(outs[:1], self.colors[1])
            self.send(outs[1:], self.colors[1])
            t.wait_until("What disappears")
            self.play(FadeIn(reg, scale=0.9), Indicate(k.rect, color=self.color, scale_factor=1.03), run_time=0.8)
            t.wait_until("That's 4S")
            self.play(FadeIn(bars.labels), *bars.grow(run_time=0.9), run_time=0.9)
            self.play(FadeIn(bars.values), run_time=0.4)
            t.wait_until("The weight vector")
            self.play(FadeIn(w_note), run_time=0.5)
            t.wait_until("So the prediction")
            self.play(FadeIn(ratio), Create(box), run_time=0.8)

    def fused_kernel(self):
        self.slide("fused-kernel", "One line in the fused kernel does the saving",
                   book_section="s-the-fused-kernel-line-by-line")
        code = kit.code(FUSED, "python", caption="rmsnorm_triton.py, kernel body")
        warn = kit.label("Divide by BLOCK instead, and every output is 1.265× too large.", size=th.SIZE_SMALL,
                         color=th.BAD)
        layout = VGroup(code, warn).arrange(DOWN, buff=0.35)
        self.place(layout)
        with self.narrate(
            "Here's the body of the fused kernel. "
            "Each program owns one row, and the mask keeps 5120 of the 8192 lanes live. "
            "Both loads fill with zero, the identity for a sum. "
            "This line is the fusion: the add runs in float32 registers, and nothing after it fetches the sum "
            "from memory. "
            "The first store publishes the new residual, which the kernel never reads back. "
            "The variance divides by the real width, not the block size. "
            "Divide by the block, and every output is off by 26.5 percent."
        ) as t:
            self.play(FadeIn(code.frame), *code.reveal(1, 2, 3, 4), run_time=0.8)
            t.wait_until("Each program owns")
            self.play(code.highlight(4), run_time=0.5)
            t.wait_until("Both loads")
            self.play(*code.reveal(5, 6), run_time=0.6)
            t.wait_until("This line is")
            self.play(*code.reveal(7), run_time=0.5)
            self.play(code.highlight(7, color=th.tint(self.color, 0.3)), code.focus(7), run_time=0.8)
            t.wait_until("The first store")
            self.play(*code.reveal(8), code.focus(7, 8), run_time=0.6)
            t.wait_until("The variance")
            self.play(*code.reveal(9, 10, 11, 12), code.focus(), run_time=0.8)
            self.play(Circumscribe(code.line(9)[-6:], color=th.BAD, buff=0.06), run_time=1.0)
            t.wait_until("Divide by the block")
            self.play(FadeIn(warn, shift=UP * 0.1), run_time=0.6)

    # ------------------------------------------------------------------
    # The measurement and the cache

    def measurement(self):
        self.slide("measurement", "The A100 disagrees with the byte count",
                   book_section="s-the-measurement-fusion-loses-on-small-tensors")
        import math

        sweep = [(2.6, 1.02), (10.5, 0.91), (21.0, 0.84), (41.9, 0.95), (83.9, 1.05), (167.8, 1.12),
                 (335.5, 1.16)]
        ax = kit.axes([0, 8.5, 1], [0, 0.5, 0.1], width=7.6, height=4.0)

        def p(mb, s):
            return ax.c2p(math.log2(mb) - 1, s - 0.8)

        xl = VGroup()
        for mb in (2, 8, 32, 128, 512):
            n = kit.label(str(mb), size=th.SIZE_LABEL, color=th.INK_3).next_to(p(mb, 0.8), DOWN, buff=0.15)
            xl.add(n)
        yl = VGroup()
        for v in (0.8, 0.9, 1.0, 1.1, 1.2, 1.3):
            n = kit.label(f"{v:.1f}", size=th.SIZE_LABEL, color=th.INK_3).next_to(ax.c2p(0, v - 0.8), LEFT, buff=0.15)
            yl.add(n)
        x_t = kit.label("tensor size, MB (log scale)", size=th.SIZE_LABEL, color=th.INK_3)
        x_t.next_to(xl, DOWN, buff=0.15).align_to(ax, RIGHT)
        y_t = kit.label("speedup", size=th.SIZE_LABEL, color=th.INK_3).next_to(yl, UP, buff=0.2).align_to(yl, LEFT)
        frame = VGroup(ax, xl, yl, x_t, y_t)
        even = Line(ax.c2p(0, 0.2), ax.c2p(8.5, 0.2), color=th.INK_3, stroke_width=th.STROKE_AXIS)
        even_l = kit.label("break even", size=th.SIZE_LABEL, color=th.INK_3).next_to(ax.c2p(8.5, 0.2), UP, 0.08)
        even_l.align_to(ax.c2p(8.5, 0), RIGHT)
        pred = DashedLine(ax.c2p(0, 0.45), ax.c2p(8.5, 0.45), color=th.INK_4, stroke_width=th.STROKE_AXIS)
        pred_l = kit.label("byte count: 1.25×", size=th.SIZE_LABEL, color=th.INK_3).next_to(pred, DOWN, 0.1)
        pred_l.align_to(pred, LEFT).shift(RIGHT * 0.15)
        l2x = math.log2(40) - 1
        l2 = DashedLine(ax.c2p(l2x, 0), ax.c2p(l2x, 0.5), color=th.INFO, stroke_width=th.STROKE_AXIS)
        l2_l = kit.label("L2: 40 MB", size=th.SIZE_LABEL, color=th.INFO).next_to(l2, UP, buff=0.08)
        pts = [p(mb, s) for mb, s in sweep]
        dots = VGroup(*[Dot(q, radius=0.08, color=self.color) for q in pts])
        curve = VMobject(color=self.color, stroke_width=th.STROKE_CURVE).set_points_as_corners(pts)
        vals = VGroup()
        sides = [UR, DL, RIGHT, DR, UL, UL, UP]
        for (mb, s), q, side in zip(sweep, pts, sides):
            v = kit.label(f"{s:.2f}", size=th.SIZE_LABEL, color=th.INK_2)
            v.next_to(q, side, buff=0.1)
            vals.add(v)
        note = kit.prose(r"The lab's own run: 0.92$\times$ at 4096 rows and 1.10$\times$ at 16384. "
                         r"Runs differ by a few percent; treat gaps under 5\% as noise.",
                         size=th.SIZE_SMALL, width=3.4, color=th.INK_2)
        plot = VGroup(frame, even, even_l, pred, pred_l, l2, l2_l, dots, curve, vals)
        layout = VGroup(plot, note).arrange(RIGHT, buff=0.6)
        self.place(layout)
        with self.narrate(
            "Here's the measurement, on an A100 with bfloat16 rows. "
            "The byte count predicted 1.25 times at every size. "
            "The lab measures 0.92 times at 4096 rows: a fifth fewer bytes, and slower. "
            "A finer sweep shows the shape. "
            "Mid-sized tensors fall below break even. "
            "From there, the speedup rises with every step, and crosses 1 between 4096 and 8192 rows. "
            "Now look at where it turns: around 40 MB, the size of the A100's L2."
        ) as t:
            self.play(FadeIn(frame), Create(even), FadeIn(even_l), run_time=0.8)
            t.wait_until("The byte count")
            self.play(Create(pred), FadeIn(pred_l), run_time=0.7)
            t.wait_until("The lab measures")
            self.play(FadeIn(note), run_time=0.6)
            t.wait_until("A finer sweep")
            self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots[:3]], lag_ratio=0.3), FadeIn(vals[:3]),
                      run_time=1.0)
            t.wait_until("From there")
            self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots[3:]], lag_ratio=0.3), FadeIn(vals[3:]),
                      Create(curve), run_time=1.6)
            t.wait_until("Now look")
            self.play(Create(l2), FadeIn(l2_l), run_time=0.8)

    def l2_absorbs(self):
        self.slide("l2-absorbs", "L2 was already absorbing the round trip",
                   book_section="s-what-the-byte-count-misses-the-l2-cache")
        add = kit.box("add", color=self.color, width=2.4, height=0.8)
        norm = kit.box("RMSNorm", color=self.color, width=2.4, height=0.8)
        ks = VGroup(add, norm).arrange(RIGHT, buff=3.2)
        l2 = panel("L2", 8.4, 1.25, color=th.INFO, sub="40 MB, write-back")
        hbm = panel("HBM", 8.4, 1.25, color=self.colors[1])
        stack = VGroup(ks, l2, hbm).arrange(DOWN, buff=0.85)
        key = kit.callout("key", r"Count the bytes that reach HBM, not the bytes a kernel touches.",
                          title="The fix to the count", width=3.9, size=th.SIZE_SMALL)
        layout = VGroup(stack, key).arrange(RIGHT, buff=0.5)
        self.place(layout)
        slot = chip("sum", self.color, w=1.2, h=0.5, size=th.SIZE_LABEL)
        slot.move_to(l2.rect.get_center() + DOWN * 0.12 + RIGHT * 0.9)
        h_slot = chip("sum", self.color, w=1.2, h=0.5, size=th.SIZE_LABEL)
        h_slot.move_to(hbm.rect.get_center() + DOWN * 0.12 + RIGHT * 0.9)
        w1 = flow(add.get_bottom() + RIGHT * 0.5, slot.get_top() + LEFT * 0.3, color=self.color)
        r1 = flow(slot.get_top() + RIGHT * 0.3, norm.get_bottom() + LEFT * 0.5, color=self.color)
        ghost = DashedLine(slot.get_bottom(), h_slot.get_top(), color=th.INK_4, stroke_width=th.STROKE_AXIS)
        never = kit.label("small tensor: never reaches HBM", size=th.SIZE_LABEL, color=th.INK_3)
        never.next_to(ghost, LEFT, buff=0.25)
        evict = flow(slot.get_bottom() + LEFT * 0.25, h_slot.get_top() + LEFT * 0.25, color=th.BAD)
        back = flow(h_slot.get_top() + RIGHT * 0.25, slot.get_bottom() + RIGHT * 0.25, color=th.BAD)
        big = kit.label("4× L2: evicted, read back", size=th.SIZE_LABEL, color=th.BAD)
        big.next_to(ghost, LEFT, buff=0.5)
        with self.narrate(
            "The cache explains the gap. "
            "The A100's 40 MB of L2 is shared by all its SMs, and it's write-back. "
            "Follow the sum through the unfused pair. "
            "The add kernel writes it, and the write lands in L2, not in HBM. "
            "The norm kernel runs straight afterward and reads it back. "
            "If those bytes are still in L2, the read never reaches HBM. "
            "The cache had already fused the pair, in effect. "
            "At four times L2, the sum is evicted before the read, and the round trip is real."
        ) as t:
            self.play(FadeIn(ks), FadeIn(l2), FadeIn(hbm), run_time=0.8)
            t.wait_until("The add kernel writes")
            d = packet(self.color).move_to(w1.get_start())
            self.play(GrowArrow(w1), run_time=0.4)
            self.play(d.animate.move_to(slot), run_time=0.7)
            self.play(FadeIn(slot), run_time=0.3)
            self.remove(d)
            t.wait_until("The norm kernel runs")
            d = packet(self.color).move_to(r1.get_start())
            self.play(GrowArrow(r1), run_time=0.4)
            self.play(d.animate.move_to(r1.get_end()), run_time=0.7)
            self.remove(d)
            t.wait_until("If those bytes")
            self.play(Create(ghost), FadeIn(never), FadeIn(h_slot.rect.copy().set_opacity(0)), run_time=0.7)
            t.wait_until("The cache had already")
            self.play(FadeIn(key, shift=LEFT * 0.1), run_time=0.7)
            t.wait_until("At four times")
            self.play(FadeOut(ghost), FadeOut(never), GrowArrow(evict), FadeIn(h_slot), FadeIn(big), run_time=0.8)
            self.play(GrowArrow(back), run_time=0.5)

    def two_number_model(self):
        self.slide("two-number-model", "Two numbers explain every measurement",
                   book_section="s-what-the-byte-count-misses-the-l2-cache")
        f = kit.formula(r"\text{speedup}", r"=", r"e", r"\cdot", r"\frac{4 + \varphi}{4}", size=th.SIZE_MATH)
        f[2].set_color(EFF_C)
        defs = VGroup(
            kit.prose(r"$\varphi$: the share of the sum that really goes through HBM. 0 if L2 absorbs it, 1 if not.",
                      size=th.SIZE_SMALL, width=4.6),
            kit.prose(r"$e$: the fused kernel's efficiency per byte, against the pair it replaces.",
                      size=th.SIZE_SMALL, width=4.6),
        ).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        defs[0][0][0].set_color(PHI_C)
        defs[1][0][0].set_color(EFF_C)
        tab = kit.table(
            [["4096", r"$\approx 0$", "1.00", r"0.92$\times$", "0.92"],
             ["16384", r"$\approx 1$", "1.25", r"1.10$\times$", "0.88"],
             ["32768", r"$\approx 1$", "1.25", r"1.16$\times$", "0.93"]],
            header=["Rows", r"$\varphi$", "Traffic", "Measured", "$e$"],
            align=["r", "c", "r", "r", "r"], size=th.SIZE_SMALL,
        )
        result = kit.formula(r"\text{speedup} \approx 0.90 \times \frac{4 + \varphi}{4}", size=th.SIZE_MATH_SMALL)
        rbox = SurroundingRectangle(result, color=self.color, buff=0.16, corner_radius=0.08, stroke_width=3)
        right = VGroup(tab, VGroup(result, rbox)).arrange(DOWN, buff=0.5)
        left = VGroup(f, defs).arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        layout = VGroup(left, right).arrange(RIGHT, buff=0.8)
        self.place(layout)
        with self.narrate(
            "To turn that into predictions, you need two numbers. "
            "Phi is the share of the sum that really goes through HBM: zero if the cache absorbs it, one if not. "
            "The second, e, is the fused kernel's efficiency per byte. "
            "It holds a whole row in float32 registers, so fewer rows fit on an SM. "
            "At 4096 rows, phi is near zero, and e comes out at 0.92. "
            "At 16384 rows, phi is near one, and e is 0.88. "
            "At the largest size, it's 0.93. "
            "All three sit near 0.90, a stable property of the kernel. "
            "So fusion here buys 0.90 times inside the cache, and 1.13 times past it."
        ) as t:
            self.play(FadeIn(f[:2]), FadeIn(f[3:]), run_time=0.8)
            t.wait_until("Phi is the share")
            self.play(FadeIn(defs[0]), Indicate(f[4], color=PHI_C), run_time=0.8)
            t.wait_until("The second, e")
            self.play(FadeIn(f[2]), FadeIn(defs[1]), run_time=0.8)
            t.wait_until("At 4096 rows, phi")
            self.play(FadeIn(tab.rules), FadeIn(VGroup(*tab.cells[0])), run_time=0.6)
            for r, cue in zip((1, 2, 3), ("At 4096 rows", "At 16384 rows", "At the largest")):
                t.wait_until(cue)
                self.play(FadeIn(VGroup(*tab.cells[r]), shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("All three")
            self.play(Indicate(VGroup(*[tab.cells[r][4] for r in (1, 2, 3)]), color=EFF_C), run_time=0.9)
            t.wait_until("So fusion here")
            self.play(FadeIn(result), Create(rbox), run_time=0.8)

    def crossover(self):
        self.slide("crossover", "Fusion breaks even between one and two times L2",
                   book_section="s-what-the-byte-count-misses-the-l2-cache")
        f = kit.formula(r"\frac{4 + \varphi}{4} \ge \frac{1}{0.90} = 1.111", r"\implies", r"\varphi \ge 0.44",
                        size=th.SIZE_MATH)
        f[2].set_color(PHI_C)
        segs = VGroup(
            Rectangle(width=4.1, height=0.75, stroke_width=0, fill_color=th.tint(th.BAD, 0.25), fill_opacity=1),
            Rectangle(width=4.1, height=0.75, stroke_width=0, fill_color=th.tint(th.INK_4, 0.3), fill_opacity=1),
            Rectangle(width=4.1, height=0.75, stroke_width=0, fill_color=th.tint(th.GOOD, 0.3), fill_opacity=1),
        ).arrange(RIGHT, buff=0)
        seg_t = VGroup(
            kit.label("inside L2: fusion loses", size=th.SIZE_LABEL, color=th.INK),
            kit.label("1–2× L2: break even", size=th.SIZE_LABEL, color=th.INK),
            kit.label("past 2× L2: fusion wins", size=th.SIZE_LABEL, color=th.INK),
        )
        for s, tx in zip(segs, seg_t):
            tx.move_to(s)
        ticks = VGroup(
            kit.label("41.9 MB · 4096 rows", size=th.SIZE_LABEL, color=th.INK_3).next_to(segs[1].get_corner(DL),
                                                                                      DOWN, 0.15),
            kit.label("83.9 MB · 8192 rows", size=th.SIZE_LABEL, color=th.INK_3).next_to(segs[1].get_corner(DR),
                                                                                      DOWN, 0.15),
        )
        band = VGroup(segs, seg_t, ticks)
        bars = kit.bar_compare([("Add kernel at S = 41.9 MB", 125.8, self.color, "3S = 126 MB")],
                               limit=40, limit_label="L2: 40 MB", width=6.0, bar_height=0.55)
        layout = VGroup(f, band, bars).arrange(DOWN, buff=0.75)
        self.place(layout)
        with self.narrate(
            "The model also says where fusion starts to pay. "
            "The traffic term has to cover the 10 percent efficiency loss. "
            "That takes phi of at least 0.44: nearly half the sum must reach HBM. "
            "In the sweep, that happens between one and two times L2, or about 4000 to 8000 rows. "
            "It isn't exactly one times L2, because the sum doesn't get the cache to itself. "
            "At 42 MB, the add kernel alone pushes 126 MB through it. "
            "Late writes survive and early ones don't, so the curve is a smooth ramp, not a step."
        ) as t:
            t.wait_until("The traffic term")
            self.play(FadeIn(f[0]), run_time=0.8)
            t.wait_until("That takes phi")
            self.play(FadeIn(f[1:]), run_time=0.7)
            t.wait_until("In the sweep")
            self.play(LaggedStart(*[FadeIn(VGroup(s, tx)) for s, tx in zip(segs, seg_t)], lag_ratio=0.3),
                      run_time=1.2)
            self.play(FadeIn(ticks), Indicate(segs[1], color=th.INK_3, scale_factor=1.03), run_time=0.8)
            t.wait_until("At 42 MB")
            self.play(FadeIn(bars.labels), FadeIn(bars.limit), *bars.grow(run_time=1.0), run_time=1.0)
            self.play(FadeIn(bars.values), run_time=0.4)

    # ------------------------------------------------------------------
    # SwiGLU

    def swiglu_traffic(self):
        self.slide("swiglu-traffic", "SwiGLU's intermediate is dead, so both trips go",
                   book_section="s-why-the-swiglu-fusion-pays-more")
        fig = self.traffic_figure(
            [("gate", self.colors[1]), ("t", self.color), ("up", self.colors[1]), ("out", self.colors[1])],
            [("silu", 2.2), ("multiply", 2.4)], w=1.35,
        )
        silu, mul = fig.kernels
        tab = kit.table([["silu", "2S"], ["multiply", "3S"], ["Unfused", "5S"], ["Fused", "3S"]],
                        header=["Kernel", "Traffic"], align=["l", "r"], size=th.SIZE_SMALL)
        ratio = kit.formula(r"\frac{5S}{3S} = 1.667", size=th.SIZE_MATH_SMALL)
        side = VGroup(tab, ratio).arrange(DOWN, buff=0.4)
        body = VGroup(fig, side).arrange(RIGHT, buff=0.8)
        facts = VGroup(
            kit.pill("measured 1.52×", self.color, size=th.SIZE_SMALL),
            kit.pill("1.52 / 1.667 = 0.91", EFF_C, size=th.SIZE_SMALL, filled=False),
            kit.pill("142.6 MB each: 3.6× L2", th.INFO, size=th.SIZE_SMALL, filled=False),
        ).arrange(RIGHT, buff=0.3)
        layout = VGroup(body, facts).arrange(DOWN, buff=0.55)
        self.place(layout)
        silu.set_x((fig.chips[0].get_x() + fig.chips[1].get_x()) / 2)
        mul.set_x((fig.chips[1].get_x() + fig.chips[3].get_x()) / 2 + 0.3)
        r_g = self.link(fig, 0, silu, dx_dst=-0.3)
        w_t = self.link(fig, silu, 1, dx_src=0.4)
        r_t = self.link(fig, 1, mul, dx_dst=-0.7)
        r_u = self.link(fig, 2, mul, dx_dst=0.0)
        w_o = self.link(fig, mul, 3, dx_src=0.7)
        dead = VGroup(Cross(fig.chips[1], stroke_color=th.BAD, stroke_width=5, scale_factor=0.8))
        with self.narrate(
            "The lab also fuses the MLP's activation, SwiGLU, and measures 1.52 times. "
            "Unfused, it's two kernels. "
            "The first reads gate and writes the activation: 2S. "
            "The second reads that and up, and writes the output: 3S. "
            "Fused, it's 3S. "
            "Nothing else needs the activation, so both of its trips disappear, and the prediction is 1.667. "
            "The measurement lands at 0.91 of that, the same efficiency as before. "
            "And each tensor is 142.6 MB, far past L2."
        ) as t:
            self.play(FadeIn(fig.hbm), FadeIn(fig.chips), FadeIn(fig.kernels), FadeIn(facts[0]), run_time=0.8)
            t.wait_until("The first reads")
            self.send([r_g], self.colors[1], run_time=0.6)
            self.send([w_t], self.color, run_time=0.6)
            self.play(FadeIn(tab.rules), FadeIn(VGroup(*tab.cells[0])), FadeIn(VGroup(*tab.cells[1])), run_time=0.4)
            t.wait_until("The second reads")
            self.send([r_t, r_u], self.colors[1], run_time=0.6)
            self.send([w_o], self.colors[1], run_time=0.6)
            self.play(FadeIn(VGroup(*tab.cells[2])), FadeIn(VGroup(*tab.cells[3])), run_time=0.4)
            t.wait_until("Fused, it's 3S")
            self.play(FadeIn(VGroup(*tab.cells[4])), run_time=0.5)
            t.wait_until("Nothing else needs")
            self.play(w_t.animate.set_color(th.BAD), r_t.animate.set_color(th.BAD), Create(dead), run_time=0.8)
            self.play(FadeIn(ratio), run_time=0.6)
            t.wait_until("The measurement lands")
            self.play(FadeIn(facts[1], shift=UP * 0.1), run_time=0.6)
            t.wait_until("And each tensor")
            self.play(FadeIn(facts[2], shift=UP * 0.1), run_time=0.6)

    def swiglu_kernel(self):
        self.slide("swiglu-kernel", "A flat kernel walks the tensor in blocks",
                   book_section="s-why-the-swiglu-fusion-pays-more")
        code = kit.code(SWIGLU, "python", caption="_swiglu_fwd, body")
        n = 8
        blocks = VGroup(*[Rectangle(width=1.35, height=0.6, stroke_color=th.RULE_STRONG,
                                    stroke_width=th.STROKE_HAIRLINE, fill_color=th.RAISE, fill_opacity=1)
                          for _ in range(n)]).arrange(RIGHT, buff=0.06)
        tags = VGroup(*[kit.label(f"pid {i}", size=th.SIZE_LABEL, color=th.INK_3, font=th.MONO).move_to(b)
                        for i, b in enumerate(blocks)])
        tags.set_z_index(3)
        dots = kit.label("…", size=th.SIZE_SMALL, color=th.INK_3).next_to(blocks, RIGHT, buff=0.2)
        cap = kit.label("gate, up, and out as flat arrays: 1024 elements per program", size=th.SIZE_LABEL,
                        color=th.INK_3)
        cap.next_to(blocks, UP, buff=0.15).align_to(blocks, LEFT)
        strip = VGroup(cap, blocks, tags, dots)
        facts = VGroup(
            kit.pill("71.3M elements", self.colors[1], size=th.SIZE_SMALL, filled=False),
            kit.pill("69,632 programs", self.color, size=th.SIZE_SMALL),
            kit.pill("645 per SM", self.color, size=th.SIZE_SMALL, filled=False),
            kit.pill("8 elements per lane", th.INK_3, size=th.SIZE_SMALL, filled=False),
        ).arrange(RIGHT, buff=0.25)
        layout = VGroup(code, strip, facts).arrange(DOWN, buff=0.45)
        self.place(layout)
        with self.narrate(
            "The SwiGLU kernel is a flat elementwise stream. "
            "Its grid runs over elements, not rows, so the shape doesn't matter. "
            "Each program takes the next 1024 elements: the global index from chapter 12, one block at a time. "
            "At 4096 tokens, that's 71.3 million elements. "
            "That's 69,632 programs, or 645 per SM. "
            "With no reduction and no wide row in registers, occupancy stays high."
        ) as t:
            self.play(FadeIn(code.frame), *code.reveal(), run_time=0.8)
            t.wait_until("Its grid runs")
            self.play(FadeIn(cap), FadeIn(blocks), run_time=0.6)
            t.wait_until("Each program takes")
            self.play(code.highlight(2), run_time=0.4)
            self.play(LaggedStart(*[AnimationGroup(b.animate.set_fill(th.tint(self.color, 0.35)), FadeIn(tg))
                                    for b, tg in zip(blocks, tags)], lag_ratio=0.2), FadeIn(dots), run_time=1.6)
            t.wait_until("At 4096 tokens")
            self.play(FadeIn(facts[0], shift=UP * 0.1), run_time=0.5)
            t.wait_until("That's 69,632")
            self.play(FadeIn(facts[1], shift=UP * 0.1), FadeIn(facts[2], shift=UP * 0.1), run_time=0.6)
            t.wait_until("With no reduction")
            self.play(FadeIn(facts[3], shift=UP * 0.1), run_time=0.5)

    # ------------------------------------------------------------------
    # One block per row, measuring, and when not to fuse

    def one_block_per_row(self):
        self.slide("one-block-per-row", "A second pass over a row costs 1%, not 50%",
                   book_section="s-one-block-per-row")
        u = 2.4 / 2048  # Manim units per element
        single = Rectangle(width=5120 * u, height=0.5, stroke_color=self.color, stroke_width=th.STROKE_BOX,
                           fill_color=th.tint(self.color, 0.3), fill_opacity=1)
        tiles = VGroup(*[Rectangle(width=c * u, height=0.5, stroke_color=self.colors[1], stroke_width=th.STROKE_BOX,
                                   fill_color=th.tint(self.colors[1], 0.25), fill_opacity=1) for c in (2048, 2048, 1024)])
        tiles.arrange(RIGHT, buff=0.06)
        s_lab = kit.label("single block: one load, held in registers", size=th.SIZE_LABEL, color=th.INK_2)
        t_lab = kit.label("tiled, 2048 at a time: two passes", size=th.SIZE_LABEL, color=th.INK_2)
        s_row = VGroup(s_lab, single).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        t_row = VGroup(t_lab, tiles).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        rows = VGroup(s_row, t_row).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        bars = kit.bar_compare(
            [("Plain copy", 0.066, th.INK_4, "0.066 ms"),
             ("Single block", 0.089, self.color, "0.089 ms · 945 GB/s"),
             ("Tiled", 0.090, self.colors[1], "0.090 ms · 929 GB/s")],
            width=3.6, bar_height=0.42, gap=0.25,
        )
        top = VGroup(rows, bars).arrange(DOWN, buff=0.6)
        warps = kit.label("On these rows, 8 warps beat 16 and 32: 945, 924, and 878 GB/s.", size=th.SIZE_SMALL,
                          color=th.INK_2)
        layout = VGroup(top, warps).arrange(DOWN, buff=0.6)
        self.place(layout)
        pass1 = Arrow(tiles.get_corner(UL) + UP * 0.12, tiles.get_corner(UR) + UP * 0.12, buff=0, color=th.INK_3,
                      stroke_width=3, tip_length=0.15)
        pass2 = Arrow(tiles.get_corner(DL) + DOWN * 0.12, tiles.get_corner(DR) + DOWN * 0.12, buff=0, color=th.INK_3,
                      stroke_width=3, tip_length=0.15)
        with self.narrate(
            "Both RMSNorm kernels give one program a whole row, which stays in registers between the reduction and "
            "the scaling. "
            "Past 16384 columns, the engine falls back to a tiled kernel. "
            "It walks the row in chunks of 2048, once for the sum of squares, and again to scale. "
            "That's 3S of traffic instead of 2S. "
            "Yet the two land within 1 percent of each other. "
            "The re-read row is only 10 KB, touched moments earlier, so it's still in cache. "
            "On these rows, 8 warps also beat 16 and 32."
        ) as t:
            self.play(FadeIn(s_lab), GrowFromEdge(single, LEFT), run_time=0.8)
            t.wait_until("Past 16384")
            self.play(FadeIn(t_lab), LaggedStart(*[GrowFromEdge(r, LEFT) for r in tiles], lag_ratio=0.3),
                      run_time=0.9)
            t.wait_until("It walks the row")
            self.play(GrowArrow(pass1), run_time=0.6)
            self.play(GrowArrow(pass2), run_time=0.6)
            t.wait_until("Yet the two")
            self.play(FadeIn(bars.labels), *bars.grow(run_time=1.0), run_time=1.0)
            self.play(FadeIn(bars.values), run_time=0.4)
            t.wait_until("On these rows")
            self.play(FadeIn(warps, shift=UP * 0.1), run_time=0.6)

    def honest_measurement(self):
        self.slide("honest-measurement", "Time the kernel, and judge it against the copy",
                   book_section="s-measure-against-what-is-reachable")
        alloc = kit.callout("warning", r"At 4096 rows, the fused kernel writes two 41.9 MB tensors. Allocating them "
                                       r"on every call costs about as much as the kernels. Pass in buffers.",
                            title="Don't time the allocator", width=12.0, size=th.SIZE_SMALL)
        bars = kit.bar_compare(
            [("RMSNorm kernel", 945, self.color, "945 GB/s"),
             ("Plain copy: the ceiling", 1275, self.colors[1], "1275 GB/s"),
             ("Rated peak", 1935, th.INK_4, "1935 GB/s")],
            width=5.6, bar_height=0.45, gap=0.28, decimals=0,
        )
        pills = VGroup(kit.pill("74% of the copy", self.color, size=th.SIZE_SMALL),
                       kit.pill("49% of rated", th.INK_3, size=th.SIZE_SMALL, filled=False)
                       ).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        row = VGroup(bars, pills).arrange(RIGHT, buff=0.6)
        layout = VGroup(alloc, row).arrange(DOWN, buff=0.6)
        self.place(layout)
        with self.narrate(
            "Two things keep a measurement honest. "
            "First, allocation. "
            "Allocating two outputs of 42 MB each on every call costs about as much as the kernels, so pass in buffers. "
            "Second, the yardstick. "
            "The RMSNorm kernel moves 945 GB/s. "
            "A plain copy reaches 1275, and no kernel beats it. "
            "The card is rated at 1935. "
            "Against the copy, the kernel reaches 74 percent. "
            "Against the rated peak, it's 49 percent, which makes a good kernel look broken."
        ) as t:
            t.wait_until("First, allocation")
            self.play(FadeIn(alloc, shift=UP * 0.1), run_time=0.7)
            t.wait_until("The RMSNorm kernel")
            self.play(FadeIn(bars.labels[0]), GrowFromEdge(bars.bars[0], LEFT), FadeIn(bars.values[0]), run_time=0.8)
            t.wait_until("A plain copy")
            self.play(FadeIn(bars.labels[1]), GrowFromEdge(bars.bars[1], LEFT), FadeIn(bars.values[1]), run_time=0.8)
            t.wait_until("The card is rated")
            self.play(FadeIn(bars.labels[2]), GrowFromEdge(bars.bars[2], LEFT), FadeIn(bars.values[2]), run_time=0.8)
            t.wait_until("Against the copy")
            self.play(FadeIn(pills[0], scale=0.9), run_time=0.5)
            t.wait_until("Against the rated")
            self.play(FadeIn(pills[1], scale=0.9), run_time=0.5)

    def when_not_to_fuse(self):
        self.slide("when-not-to-fuse", "Six cases where fusion doesn't pay", book_section="s-when-not-to-fuse")
        items = [
            ("Compute bound", "A GEMM is limited by arithmetic, not bytes."),
            ("Intermediate not dead", "It still gets written: 1.10× instead of 1.52×."),
            ("Harder to check", "Keep the unfused path, and test against it."),
            ("Register pressure", "Spills go to HBM. Try a smaller block."),
            ("Mismatched geometry", "Row against column access: up to 6.9×."),
            ("Fits in L2", "Small-batch decode: weight bound anyway."),
        ]
        cards = VGroup()
        for title, body in items:
            rect = RoundedRectangle(width=4.05, height=1.75, corner_radius=0.1, stroke_color=th.RULE_STRONG,
                                    stroke_width=th.STROKE_HAIRLINE, fill_color=th.RAISE, fill_opacity=1)
            head = kit.label(title, size=th.SIZE_SMALL, color=self.color, weight=SEMIBOLD)
            if head.width > 3.6:
                head.scale_to_fit_width(3.6)
            txt = kit.prose(kit.tex_escape(body).replace("×", r"$\times$"), size=th.SIZE_LABEL, width=3.6)
            col = VGroup(head, txt).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
            col.move_to(rect).align_to(rect, UP).shift(DOWN * 0.22)
            col.align_to(rect, LEFT).shift(RIGHT * 0.22)
            cards.add(VGroup(rect, col))
        cards.arrange_in_grid(2, 3, buff=(0.3, 0.35))
        self.place(cards)
        with self.narrate(
            "Fusion loses in six cases. "
            "A compute-bound operation, like a large GEMM, gains almost nothing. "
            "If anything outside reads the intermediate, you still write it, and the saving halves. "
            "Every fused kernel brings new bugs, so test it against the unfused path. "
            "Too much state per lane spills registers to HBM. "
            "Fusing a row-wise pass with a column-wise one gives one of them the wrong access pattern. "
            "And when the working set fits in L2, as in small-batch decode, fusion buys nothing."
        ) as t:
            cues = ["A compute-bound", "If anything outside", "Every fused kernel", "Too much state",
                    "Fusing a row-wise", "And when the working"]
            for card, cue in zip(cards, cues):
                t.wait_until(cue)
                self.play(FadeIn(card, shift=UP * 0.1), run_time=0.5)

    def correctness(self):
        self.slide("correctness", "Test the width that isn't a power of two", book_section="s-correctness-first")
        tab = kit.table(
            [[r"\texttt{(64, 5120)}", "The ordinary case"],
             [r"\texttt{(1, 5120)}", "A grid of one program"],
             [r"\texttt{(4096, 5120)}", "Fills 108 SMs several times"],
             [r"\texttt{(13, 4097)}", "Not a power of two; fewer rows than SMs"]],
            header=["Shape", "What it catches"], size=th.SIZE_SMALL, highlight_rows=[3],
        )
        tab.bands.set_opacity(0)
        total = 8.0
        live = Rectangle(width=total * 4097 / 8192, height=0.6, stroke_color=self.color, stroke_width=th.STROKE_BOX,
                         fill_color=th.tint(self.color, 0.35), fill_opacity=1)
        dead = Rectangle(width=total * 4095 / 8192, height=0.6, stroke_color=th.RULE_STRONG,
                         stroke_width=th.STROKE_BOX, fill_color=th.tint(th.INK_4, 0.3), fill_opacity=1)
        bar = VGroup(live, dead).arrange(RIGHT, buff=0)
        l_t = kit.label("4097 live", size=th.SIZE_SMALL, color=th.INK).move_to(live)
        d_t = kit.label("4095 masked", size=th.SIZE_SMALL, color=th.INK_2).move_to(dead)
        cap = kit.label("One row of (13, 4097): a block of 8192 lanes", size=th.SIZE_LABEL, color=th.INK_3)
        cap.next_to(bar, UP, buff=0.15).align_to(bar, LEFT)
        strip = VGroup(cap, bar, l_t, d_t)
        layout = VGroup(tab, strip).arrange(DOWN, buff=0.7)
        self.place(layout)
        with self.narrate(
            "Check correctness before speed: a small error compounds over every layer. "
            "Compare in float32 against the PyTorch reference, on shapes that exercise the tails. "
            "A single row tests a grid of one program. "
            "A big batch fills every SM several times over. "
            "The important one is 13 rows of width 4097. "
            "Its block rounds up to 8192, so nearly half the lanes are masked. "
            "A kernel with a missing mask passes the other shapes, and fails this one."
        ) as t:
            t.wait_until("Compare in float32")
            self.play(FadeIn(tab.rules), FadeIn(VGroup(*tab.cells[0])), FadeIn(VGroup(*tab.cells[1])), run_time=0.7)
            t.wait_until("A single row")
            self.play(FadeIn(VGroup(*tab.cells[2]), shift=RIGHT * 0.1), run_time=0.5)
            t.wait_until("A big batch")
            self.play(FadeIn(VGroup(*tab.cells[3]), shift=RIGHT * 0.1), run_time=0.5)
            t.wait_until("The important one")
            self.play(FadeIn(VGroup(*tab.cells[4]), shift=RIGHT * 0.1), tab.bands.animate.set_opacity(1),
                      run_time=0.6)
            t.wait_until("Its block rounds")
            self.play(FadeIn(cap), GrowFromEdge(live, LEFT), FadeIn(l_t), run_time=0.8)
            self.play(GrowFromEdge(dead, LEFT), FadeIn(d_t), run_time=0.8)
