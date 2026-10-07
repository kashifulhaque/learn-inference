"""Chapter 14, FlashAttention.

Teaches content/chapters/14-flash-attention.md in its order. Every number
below was recomputed with Python and numpy; each matches the chapter.

What the naive version costs (24 query heads, d = 256, L = 8192):
- 8192 x 8192 = 67,108,864 score entries per head.
- bf16: 67,108,864 x 2 = 134,217,728 B = 134.2 MB per head; x 24 heads =
  3,221,225,472 B = 3.22 GB.
- float32 (x 2) holding scores and probabilities (x 2): 12.88 GB, about
  12.9 GB, against 53.8 GB of weights (chapter value).
- Four passes over bf16 scores: 4 x 134.2 MB = 536.9 MB, about 537 MB per head.
- Inputs 3 L d x 2 B = 3 x 8192 x 256 x 2 = 12,582,912 B = 12.6 MB per head.
- Score share of traffic: 536.9 / (536.9 + 12.6) = 0.977, "97%" as the
  chapter states it.

Online softmax on four scores s = (1, 2, 4, 3), v = (10, 20, 30, 40), blocks
of two (numpy, float64):
- Block 1: m = 2; weights e^-1 = 0.36788, e^0 = 1; l = 1.36788 (1.368);
  u = 0.36788 x 10 + 20 = 23.679 (23.68).
- Block 2: m = 4; new weights 1 and 0.36788; they add 1.368 to l and
  30 + 14.715 = 44.715 (chapter: 44.72) to u.
- alpha = e^(2 - 4) = 0.135335 (0.135); alpha x l = 0.18512 (0.185);
  alpha x u = 3.2046 (3.20); l = 1.553002 (1.553); u = 47.91975 (47.92).
- u / l = 30.8562 (30.86).
- One-pass check: e^-3, e^-2, e^0, e^-1 = 0.0498, 0.1353, 1, 0.3679
  (0.050, 0.135, 1, 0.368); sum 1.553002; weighted sum 47.91975; ratio
  30.8562. Identical to the streaming result. On screen the old block-1
  weights convert as 0.368 x 0.135 = 0.0498 (0.050) and 1 x 0.135 = 0.135.

What lives where (B_r = 128, B_c = 64, d = 128, lab geometry):
- Q_i: 128 x 128 x 2 B = 32 KiB. acc: 128 x 128 x 4 B = 64 KiB.
  m and l: 2 x 128 x 4 B = 1 KiB. S, P: 128 x 64 x 4 B = 32 KiB each.
- d = 256: acc = 128 x 256 x 4 B = 128 KiB.
- Shared memory, bf16, 4 stages: 4 x 64 x 128 x 2 x 2 = 131,072 B = 128 KiB,
  under the 160 KB budget (SHARED_MEMORY_BYTES).

Causal blocks at L = 8192: 8192/128 = 64 query tiles, 8192/64 = 128 key
blocks, 64 x 128 = 8192 pairs; sum_{i=0}^{63} 2(i+1) = 4160; 4160/8192 =
0.5078 (50.8%). The drawn 8 x 16 grid groups 8 x 8 block pairs per square:
full squares (J < 2I) hold 3584 pairs, the two diagonal squares per row hold
52 + 20 pairs, and 3584 + 8 x 72 = 4160.

GQA: 24 query heads / 4 KV heads = 6 per KV head.

IO: per head at d = 128, inputs 3 x 8192 x 128 x 2 B = 6.3 MB. M = 160 x
1024 / 512 = 320 row pairs; M/d = 320/128 = 2.5. K and V for one head:
8192 x 128 x 2 B x 2 = 4.19 MB (4.2 MB); two KV heads 8.4 MB, under 40 MB L2.

Speed (L = 8192, 8 heads, d = 128, causal): 4 x 8192^2 x 128 = 34.36 GFLOP
per head; halved 17.2; x 8 = 137.4 GFLOP; / 312 TFLOP/s = 0.4405 ms (0.44).
137.4 GFLOP / 1.64 ms = 83.8 TFLOP/s = 26.9% of peak (27%). Vendor time
0.63 x 1.64 = 1.033 ms ("near 1 ms"), 133 TFLOP/s = 42.6% (43%).
Exponentials: 8192^2 / 2 x 8 = 268,435,456 (268 million). 64 x 64 tiling:
3.35 ms, 3.35 / 1.64 = 2.04 (a factor of two).

Memory (L = 4096, 8 heads, d = 128): scores 8 x 4096^2 x 4 B = 536,870,912 B
= 512 MiB. Q 8 x 4096 x 128 x 2 B = 8 MiB; K, V with 2 KV heads 2 MiB each;
O 8 MiB; total 20 MiB. (512 + 38) / 20 = 27.5.
"""

from manim import *

import kit
from kit import theme as th

# The chapter's four maths colours: m, l, u, alpha.
M_C = th.HL_A
L_C = th.HL_B
U_C = th.HL_C
A_C = th.HL_D

LOOP_SRC = """\
for start_n in range(0, hi, BLOCK_N):
    # load k and v for this block, then mask
    scores = tl.dot(q, tl.trans(k)) * scale
    m_new = tl.maximum(m_i, tl.max(scores, axis=1))  # stage 1
    correction = tl.exp(m_i - m_new)                 # stage 2
    p = tl.exp(scores - m_new[:, None])
    # stages 3 and 4: rescale, then add the new block
    l_i = l_i * correction + tl.sum(p, axis=1)
    acc = acc * correction[:, None] + tl.dot(p.to(v.dtype), v)
    m_i = m_new

acc = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
"""


def panel(title, width, height, color=None, sub=None, size=th.SIZE_SMALL):
    """A titled region (a memory level or a chip) with its label in the top-left corner."""
    stroke = th.RULE_STRONG if color is None else color
    rect = RoundedRectangle(width=width, height=height, corner_radius=0.12, stroke_color=stroke,
                            stroke_width=th.STROKE_BOX, fill_color=th.tint(stroke, 0.06), fill_opacity=1)
    name = kit.label(title, size=size, color=th.INK, weight=MEDIUM)
    parts = [name]
    if sub:
        parts.append(kit.label(sub, size=th.SIZE_LABEL, color=th.INK_3))
    head = VGroup(*parts).arrange(RIGHT, buff=0.2, aligned_edge=DOWN)
    head.move_to(rect.get_corner(UL) + RIGHT * (head.width / 2 + 0.2) + DOWN * (head.height / 2 + 0.16))
    group = VGroup(rect, head)
    group.rect, group.head = rect, head
    return group


def tile(width, height, color, text=None, size=th.SIZE_LABEL, tint=0.35):
    """A filled rectangle standing for a tensor tile, with an optional label."""
    rect = Rectangle(width=width, height=height, stroke_color=color, stroke_width=th.STROKE_BOX,
                     fill_color=th.tint(color, tint), fill_opacity=1)
    group = VGroup(rect)
    if text:
        t = kit.prose(text, size=size, color=th.INK, width=None) if "$" in text else kit.label(
            text, size=size, color=th.INK)
        t.move_to(rect)
        group.add(t)
    group.rect = rect
    return group


def value_tex(text, size=th.SIZE_HEADING, color=th.INK):
    return kit.math_label(text, size=size, color=color)


class Lecture(kit.Lecture):
    def construct(self):
        self.title_slide(
            narration=(
                "FlashAttention. "
                "Ordinary attention builds a table of scores, one for every pair of tokens. "
                "In this lecture, you build the online softmax from four numbers, and then follow the tiled kernel "
                "that never writes that table to memory."
            )
        )
        self.score_matrix()
        self.four_passes()
        self.softmax_resists()
        self.four_scores()
        self.block_one()
        self.max_moves()
        self.rescale()
        self.one_pass_check()
        self.streaming_max()
        self.rescaling_identity()
        self.corrected_totals()
        self.exact()
        self.tiled_loop()
        self.what_lives_where()
        self.kernel_loop()
        self.causal_blocks()
        self.gqa()
        self.io_complexity()
        self.speed()
        self.memory()
        self.closing_slide(
            [
                r"Naive attention moves $L^2$ scores through HBM four times.",
                r"Keep $m$, $\ell$, $u$ per row; rescale by $\alpha$.",
                r"Exact up to rounding; only $Q$, $K$, $V$, $O$ in HBM.",
                r"0.63$\times$ the vendor speed, 27.5$\times$ less memory.",
            ],
            narration=(
                "Here's what to keep. "
                "Naive attention pushes a square score table through HBM four times, which is 97% of its traffic "
                "at 8192 tokens. "
                "The online softmax keeps a maximum, a sum, and an output per row, and rescales the last two by "
                "alpha whenever the maximum rises. "
                "The result is exact up to rounding, and only Q, K, V, and O ever live in HBM. "
                "The teaching kernel runs at 0.63 times the vendor's speed, with 27.5 times less peak memory."
            ),
            book_section="s-what-goes-wrong",
        )

    # ------------------------------------------------------------------
    # What the naive version costs

    def score_matrix(self):
        self.slide("score-matrix", "The score table grows with the square of L",
                   book_section="s-what-the-naive-version-costs")
        n = 8
        c = 0.42
        scores = kit.grid(n, n, cell=c)
        q = kit.grid(n, 2, cell=c)
        kt = kit.grid(2, n, cell=c)
        q.paint(q.block(0, q.rows, 0, q.cols), self.colors[1])
        kt.paint(kt.block(0, kt.rows, 0, kt.cols), self.colors[2])
        q.next_to(scores, LEFT, buff=0.25)
        kt.next_to(scores, UP, buff=0.25)
        q_lab = kit.prose(r"$Q$: $L \times d$", size=th.SIZE_LABEL, color=th.INK_2, width=None)
        q_lab.next_to(q, DOWN, buff=0.15)
        k_lab = kit.prose(r"$K^\top$: $d \times L$", size=th.SIZE_LABEL, color=th.INK_2, width=None)
        k_lab.next_to(kt, UP, buff=0.12)
        s_lab = kit.prose(r"scores: $L \times L$", size=th.SIZE_LABEL, color=self.color, width=None)
        s_lab.next_to(scores, DOWN, buff=0.15)
        figure = VGroup(scores, q, kt, q_lab, k_lab, s_lab)

        entries = kit.formula(r"8192 \times 8192 = 67{,}108{,}864", size=th.SIZE_MATH_SMALL)
        rows = kit.table(
            [
                ["bf16, one head", "134.2 MB"],
                ["bf16, 24 heads", "3.22 GB"],
                ["float32, scores and probabilities", "12.9 GB"],
                ["the model's weights", "53.8 GB"],
            ],
            header=["One sequence of 8192 tokens", "Bytes"],
            align=["l", "r"],
            size=th.SIZE_SMALL,
        )
        side = VGroup(entries, rows).arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        layout = VGroup(figure, side).arrange(RIGHT, buff=0.8)
        self.place(layout)
        cells = [scores.cell(r, cc) for r in range(n) for cc in range(n)]
        with self.narrate(
            "Attention is the one operation whose intermediate result is larger than the model. "
            "Every query scores against every key, so the scores form a square table. "
            "Take this model's geometry: 24 query heads, head dimension 256, and 8192 tokens. "
            "One head's table holds about 67 million entries. "
            "In bf16, that's 134.2 MB for one head. "
            "Across 24 heads, it's 3.22 GB for a single sequence. "
            "The reference path computes in float32 and holds the scores and the probabilities at once. "
            "That's about 12.9 GB for one sequence, against 53.8 GB of weights."
        ) as t:
            self.play(FadeIn(q), FadeIn(kt), FadeIn(q_lab), FadeIn(k_lab), FadeIn(scores), run_time=0.8)
            t.wait_until("Every query scores")
            self.play(LaggedStart(*[sq.animate.set_fill(th.tint(self.color, 0.35), opacity=1)
                                    for sq in scores.squares()], lag_ratio=0.03),
                      FadeIn(s_lab), run_time=2.0)
            t.wait_until("One head's table")
            self.play(FadeIn(entries, shift=UP * 0.1), FadeIn(rows.rules), FadeIn(VGroup(*rows.cells[0])),
                      run_time=0.8)
            for r, cue in zip(range(1, 5), ["In bf16", "Across 24", "The reference path", "That's about 12.9"]):
                t.wait_until(cue)
                self.play(FadeIn(VGroup(*rows.cells[r]), shift=RIGHT * 0.1), run_time=0.6)

    def four_passes(self):
        self.slide("four-passes", "Four trips through HBM for one table", book_section="s-what-the-naive-version-costs")
        hbm = panel("HBM", 3.4, 2.7, color=self.colors[1])
        chip = panel("On the chip", 3.4, 2.7, color=self.color)
        row = VGroup(hbm, chip).arrange(RIGHT, buff=4.0)
        s_tile = tile(1.1, 1.1, self.color, "$S$")
        s_tile.move_to(hbm.rect.get_center() + DOWN * 0.2)
        sm = tile(1.6, 1.0, self.color, "SM", tint=0.12)
        sm.move_to(chip.rect.get_center() + DOWN * 0.2)
        figure = VGroup(row, s_tile, sm)
        bars = kit.bar_compare(
            [("Q, K, V inputs", 12.6, self.colors[1], "12.6 MB"),
             ("Score traffic", 536.9, self.color, "537 MB")],
            width=4.5, bar_height=0.42, gap=0.25,
        )
        share = kit.pill("97% of the traffic", self.color)
        bottom = VGroup(bars, share).arrange(RIGHT, buff=0.6)
        layout = VGroup(figure, bottom).arrange(DOWN, buff=0.55)
        self.place(layout)
        steps = [("1  write scores", RIGHT), ("2  read for softmax", LEFT),
                 ("3  write probabilities", RIGHT), ("4  read for PV", LEFT)]
        y0 = hbm.rect.get_top()[1] - 0.55
        x_l, x_r = hbm.rect.get_right()[0] + 0.15, chip.rect.get_left()[0] - 0.15
        arrows = []
        for i, (text, d) in enumerate(steps):
            y = y0 - i * 0.6
            a, b = ([x_r, y, 0], [x_l, y, 0]) if d is RIGHT else ([x_l, y, 0], [x_r, y, 0])
            arr = Arrow(a, b, buff=0, color=th.INK_3, stroke_width=3, tip_length=0.18)
            lab = kit.label(text, size=th.SIZE_LABEL, color=th.INK_2)
            lab.next_to(arr, UP, buff=0.04)
            arrows.append(VGroup(arr, lab))
        with self.narrate(
            "The naive path makes four passes over the table. "
            "It writes the scores. "
            "It reads them back for the softmax. "
            "It writes the probabilities. "
            "Then it reads those for the value multiply. "
            "Per head, that's 537 MB of score traffic, while the inputs are only 12.6 MB. "
            "So 97% of the memory traffic serves a table that exists only to be consumed. "
            "That's why long-context prefill runs out of memory before it runs out of time."
        ) as t:
            self.play(FadeIn(hbm), FadeIn(chip), FadeIn(s_tile), FadeIn(sm), run_time=0.8)
            for (text, d), arr, cue in zip(steps, arrows, ["It writes the scores", "It reads them back",
                                                           "It writes the probabilities", "Then it reads"]):
                t.wait_until(cue)
                dot = Square(0.22, stroke_width=0, fill_color=self.color, fill_opacity=1)
                dot.move_to(arr[0].get_start())
                self.play(GrowArrow(arr[0]), FadeIn(arr[1]), run_time=0.5)
                self.play(dot.animate.move_to(arr[0].get_end()), run_time=0.6)
                self.remove(dot)
            t.wait_until("Per head")
            self.play(*bars.grow(run_time=1.0), FadeIn(bars.labels), run_time=1.0)
            self.play(FadeIn(bars.values), run_time=0.4)
            t.wait_until("So 97%")
            self.play(FadeIn(share, scale=0.9), run_time=0.6)

    # ------------------------------------------------------------------
    # Why softmax resists tiling

    def softmax_resists(self):
        self.slide("softmax-resists", "The softmax needs the whole row", book_section="s-why-softmax-resists-tiling")
        n = 16
        row = kit.grid(1, n, cell=(0.62, 0.5))
        row_lab = kit.prose(r"row $i$ of the scores: $L_k$ entries", size=th.SIZE_LABEL, color=th.INK_3, width=None)
        row_lab.next_to(row, UP, buff=0.15).align_to(row, LEFT)
        f = kit.formula(r"m_i = \max_j s_{ij},", r"\qquad", r"\ell_i = \sum_{j=1}^{L_k} e^{s_{ij} - m_i}",
                        size=th.SIZE_MATH_SMALL)
        f[0].set_color(M_C)
        f[2].set_color(L_C)
        two = kit.callout("note", r"Compute $m_i$ and $\ell_i$ first, then the output. Doubles the reads of $K$.",
                          title="Escape 1: two passes", width=5.8, size=th.SIZE_SMALL)
        one = kit.callout("key", r"Repair the running values each time the maximum moves. "
                                 r"The repair costs a multiply.",
                          title="Escape 2: one pass with repair", width=5.8, size=th.SIZE_SMALL)
        cards = VGroup(two, one).arrange(RIGHT, buff=0.4, aligned_edge=UP)
        top = VGroup(row_lab, row)
        layout = VGroup(top, f, cards).arrange(DOWN, buff=0.65)
        self.place(layout)
        box = row.outline(0, 1, 0, 4, color=self.color)
        box_lab = kit.label("one tile: 64 columns", size=th.SIZE_LABEL, color=self.color)
        step = row.cell(0, 1).get_x() - row.cell(0, 0).get_x()
        box_lab.next_to(box, DOWN, buff=0.12)
        with self.narrate(
            "Why not cut the score table into tiles, one at a time? "
            "The two matmuls tile without trouble, but the softmax between them doesn't. "
            "To normalize a row, it needs the row's maximum and the sum of its exponentials. "
            "Both are reductions over the whole row, and a tile gives you only 64 columns of it. "
            "There are two escapes. "
            "You can make two passes, which doubles the reads of K. "
            "Or you can make one pass, and repair the running values each time the maximum moves. "
            "That's the online softmax, and the repair costs a multiply."
        ) as t:
            self.play(FadeIn(row), FadeIn(row_lab), run_time=0.7)
            t.wait_until("The two matmuls")
            self.play(Create(box), FadeIn(box_lab), run_time=0.6)
            t.wait_until("To normalize")
            self.play(FadeIn(f), run_time=0.8)
            t.wait_until("Both are reductions")
            self.play(VGroup(box, box_lab).animate.shift(RIGHT * step * 12), run_time=1.6)
            self.play(VGroup(box, box_lab).animate.shift(LEFT * step * 12), run_time=0.8)
            t.wait_until("You can make two")
            self.play(FadeIn(two, shift=UP * 0.1), run_time=0.6)
            t.wait_until("Or you can make")
            self.play(FadeIn(one, shift=UP * 0.1), run_time=0.6)

    # ------------------------------------------------------------------
    # The online softmax on four scores

    def example_board(self):
        """The worked example: four keys in two blocks, a weight row, and three state boxes."""
        cell = (1.25, 0.6)
        g1 = kit.grid(3, 2, cell=cell, values=[["1", "2"], ["10", "20"], ["", ""]],
                      row_labels=["score s", "value v", "weight"], col_labels=["key 1", "key 2"])
        g2 = kit.grid(3, 2, cell=cell, values=[["4", "3"], ["30", "40"], ["", ""]],
                      col_labels=["key 3", "key 4"])
        g2.next_to(g1, RIGHT, buff=0.4)
        g2.align_to(g1, DOWN)
        b1 = kit.label("block 1", size=th.SIZE_LABEL, color=th.INK_3)
        b2 = kit.label("block 2", size=th.SIZE_LABEL, color=th.INK_3)
        b1.next_to(VGroup(*g1.cells[2]), DOWN, buff=0.14)
        b2.next_to(VGroup(*g2.cells[2]), DOWN, buff=0.14)
        grids = VGroup(g1, g2, b1, b2)
        boxes = VGroup()
        for sym, name, color in [("m", "running maximum", M_C), (r"\ell", "running sum", L_C),
                                 ("u", "running output", U_C)]:
            rect = RoundedRectangle(width=3.6, height=1.35, corner_radius=0.1, stroke_color=color,
                                    stroke_width=th.STROKE_BOX, fill_color=th.tint(color, 0.08), fill_opacity=1)
            s = kit.math_label(sym, size=th.SIZE_BODY, color=color)
            nm = kit.label(name, size=th.SIZE_LABEL, color=th.INK_3)
            head = VGroup(s, nm).arrange(RIGHT, buff=0.15, aligned_edge=DOWN)
            head.move_to(rect.get_top() + DOWN * 0.3)
            b = VGroup(rect, head)
            b.rect = rect
            boxes.add(b)
        boxes.arrange(RIGHT, buff=0.35)
        board = VGroup(grids, boxes)
        board.g1, board.g2, board.boxes = g1, g2, boxes
        return board

    def key_cell(self, board, k):
        g = board.g1 if k < 2 else board.g2
        return g, k % 2

    def weight(self, board, k, text, color=th.INK):
        g, c = self.key_cell(board, k)
        w = kit.label(text, size=th.SIZE_SMALL, color=color, font=th.MONO)
        box = g.cell(2, c).square
        if w.width > box.width * 0.82:
            w.scale_to_fit_width(box.width * 0.82)
        w.move_to(box)
        return w

    def state_value(self, board, i, tex):
        b = board.boxes[i]
        v = value_tex(tex, size=th.SIZE_HEADING, color=th.INK)
        v.move_to(b.rect.get_center() + DOWN * 0.22)
        return v

    def layout_board(self, board, work):
        """Places the keys top left, the work column top right, and the state boxes below."""
        grids, boxes = board
        top = VGroup(grids, work).arrange(RIGHT, buff=0.7, aligned_edge=UP)
        layout = VGroup(top, boxes).arrange(DOWN, buff=0.5)
        boxes.align_to(top, LEFT)
        self.place(layout, "top")
        return layout

    def four_scores(self):
        self.slide("four-scores", "Four scores, two blocks, three running numbers",
                   book_section="s-the-online-softmax-on-four-scores")
        board = self.example_board()
        defs = VGroup(
            kit.prose(r"$m$: the largest score so far", size=th.SIZE_SMALL, width=None),
            kit.prose(r"$\ell = \sum e^{s_j - m}$", size=th.SIZE_SMALL, width=None),
            kit.prose(r"$u = \sum e^{s_j - m}\, v_j$", size=th.SIZE_SMALL, width=None),
        ).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        goal = kit.prose(r"Goal: $\sum_j \mathrm{softmax}(s)_j\, v_j$, one block at a time.", size=th.SIZE_SMALL,
                         width=4.4)
        work = VGroup(goal, defs).arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        self.layout_board(board, work)
        init = [self.state_value(board, 0, r"-\infty"), self.state_value(board, 1, "0"),
                self.state_value(board, 2, "0")]
        with self.narrate(
            "Now run the algorithm by hand. "
            "One query has four scores: 1, 2, 4, and 3. "
            "They arrive in two blocks of two. "
            "Each key carries a value: 10, 20, 30, and 40. "
            "The goal is the softmax-weighted average of the values, seeing one block at a time. "
            "You keep three running numbers. "
            "The running maximum is the largest score so far. "
            "The running sum adds up e to the score minus that maximum. "
            "The running output adds up the same weights times the values. "
            "The maximum starts at minus infinity, and both totals start at zero."
        ) as t:
            g1, g2 = board.g1, board.g2
            frame = VGroup(g1, g2)
            self.play(FadeIn(frame), run_time=0.8)
            t.wait_until("They arrive")
            self.play(FadeIn(board[0][2]), FadeIn(board[0][3]), run_time=0.5)
            t.wait_until("The goal")
            self.play(FadeIn(goal), run_time=0.6)
            for i, cue in enumerate(["The running maximum", "The running sum", "The running output"]):
                t.wait_until(cue)
                self.play(FadeIn(board.boxes[i], shift=UP * 0.1), FadeIn(defs[i]), run_time=0.6)
            t.wait_until("The maximum starts")
            self.play(*[FadeIn(v) for v in init], run_time=0.6)

    def block_one(self):
        self.slide("block-one", "Block 1 sets the maximum to 2", book_section="s-the-online-softmax-on-four-scores")
        board = self.example_board()
        work = VGroup(
            kit.formula(r"m = \max(1, 2) = 2", size=th.SIZE_MATH_SMALL),
            kit.formula(r"\ell = 0.368 + 1 = 1.368", size=th.SIZE_MATH_SMALL),
            kit.formula(r"u = 0.368 \times 10 + 1 \times 20", size=th.SIZE_MATH_SMALL),
            kit.formula(r"= 23.68", size=th.SIZE_MATH_SMALL),
        ).arrange(DOWN, buff=0.38, aligned_edge=LEFT)
        work[3].align_to(work[2], LEFT).shift(RIGHT * 0.45)
        for w, col in zip(work, [M_C, L_C, U_C]):
            w[0][0].set_color(col)
        self.layout_board(board, work)
        old = [self.state_value(board, 0, r"-\infty"), self.state_value(board, 1, "0"),
               self.state_value(board, 2, "0")]
        new = [self.state_value(board, 0, "2"), self.state_value(board, 1, "1.368"),
               self.state_value(board, 2, "23.68")]
        w = [self.weight(board, 0, "0.368"), self.weight(board, 1, "1")]
        outline = board.g1.outline(0, 3, 0, 2, color=self.color)
        with self.narrate(
            "Block 1 holds the scores 1 and 2. "
            "The larger is 2, so the maximum becomes 2. "
            "Measure each score against it. "
            "The first weight is e to the minus 1, or 0.368. "
            "The second is e to the zero, which is 1. "
            "The running sum is their total, 1.368. "
            "The running output weights each value: 0.368 times 10, plus 1 times 20. "
            "That's 23.68."
        ) as t:
            self.play(FadeIn(board), *[FadeIn(o) for o in old], run_time=0.5)
            self.play(Create(outline), run_time=0.6)
            t.wait_until("The larger is 2")
            self.play(FadeIn(work[0]), Transform(old[0], new[0]), Indicate(board.boxes[0].rect, color=M_C),
                      run_time=0.9)
            t.wait_until("The first weight")
            self.play(FadeIn(w[0], scale=0.8), run_time=0.5)
            t.wait_until("The second is")
            self.play(FadeIn(w[1], scale=0.8), run_time=0.5)
            t.wait_until("The running sum is")
            self.play(FadeIn(work[1]), Transform(old[1], new[1]), run_time=0.8)
            t.wait_until("The running output")
            self.play(FadeIn(work[2]), run_time=0.8)
            t.wait_until("That's 23.68")
            self.play(FadeIn(work[3]), Transform(old[2], new[2]), run_time=0.8)

    def max_moves(self):
        self.slide("max-moves", "Block 2 moves the maximum to 4", book_section="s-the-online-softmax-on-four-scores")
        board = self.example_board()
        work = VGroup(
            kit.formula(r"m = \max(2, 4, 3) = 4", size=th.SIZE_MATH_SMALL),
            kit.formula(r"\text{new sum: } 1 + 0.368 = 1.368", size=th.SIZE_MATH_SMALL),
            kit.formula(r"\text{new output: } 30 + 14.72 = 44.72", size=th.SIZE_MATH_SMALL),
        ).arrange(DOWN, buff=0.32, aligned_edge=LEFT)
        work[0][0][0].set_color(M_C)
        clash = kit.label("Different units: you can't add them yet.", size=th.SIZE_SMALL, color=th.BAD,
                          weight=MEDIUM)
        col = VGroup(work, clash).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        self.layout_board(board, col)
        vals = [self.state_value(board, 0, "2"), self.state_value(board, 1, "1.368"),
                self.state_value(board, 2, "23.68")]
        new_m = self.state_value(board, 0, "4")
        w = [self.weight(board, 0, "0.368"), self.weight(board, 1, "1")]
        w_new = [self.weight(board, 2, "1", color=self.color), self.weight(board, 3, "0.368", color=self.color)]
        unit1 = kit.label("against 2", size=th.SIZE_LABEL, color=th.INK_3)
        unit2 = kit.label("against 4", size=th.SIZE_LABEL, color=self.color)
        outline = board.g2.outline(0, 3, 0, 2, color=self.color)
        # The block labels give way to the unit labels.
        unit1.move_to(board[0][2])
        unit2.move_to(board[0][3])
        with self.narrate(
            "Block 2 holds the scores 4 and 3. "
            "Its larger score is 4, so the maximum rises to 4. "
            "Measure the new scores against 4. "
            "They give 1 and 0.368. "
            "So the new block adds 1.368 to the sum, and 44.72 to the output. "
            "But you can't add those to the old totals yet. "
            "The old totals were measured against a maximum of 2, and the new terms against 4. "
            "They're in different units."
        ) as t:
            self.play(FadeIn(board), *[FadeIn(o) for o in vals + w], run_time=0.5)
            self.play(Create(outline), run_time=0.6)
            t.wait_until("Its larger score")
            self.play(FadeIn(work[0]), Transform(vals[0], new_m), Indicate(board.boxes[0].rect, color=M_C),
                      run_time=0.9)
            t.wait_until("They give 1")
            self.play(FadeIn(w_new[0], scale=0.8), FadeIn(w_new[1], scale=0.8), run_time=0.7)
            t.wait_until("So the new block")
            self.play(FadeIn(work[1]), run_time=0.6)
            self.play(FadeIn(work[2]), run_time=0.6)
            t.wait_until("The old totals")
            self.play(FadeOut(board[0][2]), FadeOut(board[0][3]), FadeIn(unit1), FadeIn(unit2), run_time=0.7)
            t.wait_until("They're in different")
            self.play(FadeIn(clash, shift=UP * 0.1), run_time=0.6)

    def rescale(self):
        self.slide("rescale", "One multiply converts the old totals", book_section="s-the-online-softmax-on-four-scores")
        board = self.example_board()
        alpha = kit.formula(r"\alpha = e^{2-4} = 0.135", size=th.SIZE_MATH_SMALL)
        alpha[0][0].set_color(A_C)
        work = VGroup(
            alpha,
            kit.formula(r"\ell = 0.135 \times 1.368 + 1.368", size=th.SIZE_MATH_SMALL),
            kit.formula(r"= 0.185 + 1.368 = 1.553", size=th.SIZE_MATH_SMALL),
            kit.formula(r"u = 0.135 \times 23.68 + 44.72", size=th.SIZE_MATH_SMALL),
            kit.formula(r"= 3.20 + 44.72 = 47.92", size=th.SIZE_MATH_SMALL),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        work[1][0][0].set_color(L_C)
        work[3][0][0].set_color(U_C)
        work[2].shift(RIGHT * 0.45)
        work[4].shift(RIGHT * 0.45)
        result = kit.formula(r"u / \ell = 47.92 / 1.553 = 30.86", size=th.SIZE_MATH_SMALL)
        result_box = SurroundingRectangle(result, color=self.color, buff=0.14, corner_radius=0.08, stroke_width=3)
        res = VGroup(result, result_box)
        col = VGroup(work, res).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        self.layout_board(board, col)
        vals = [self.state_value(board, 0, "4"), self.state_value(board, 1, "1.368"),
                self.state_value(board, 2, "23.68")]
        new_vals = [self.state_value(board, 1, "1.553"), self.state_value(board, 2, "47.92")]
        w_old = [self.weight(board, 0, "0.368"), self.weight(board, 1, "1")]
        w_conv = [self.weight(board, 0, "0.050", color=A_C), self.weight(board, 1, "0.135", color=A_C)]
        w_new = [self.weight(board, 2, "1", color=self.color), self.weight(board, 3, "0.368", color=self.color)]
        times = kit.math_label(r"\text{each} \times 0.135", size=th.SIZE_SMALL, color=A_C)
        times.move_to(board[0][2])
        with self.narrate(
            "Multiply any old term, e to the score minus 2, by e to the 2 minus 4. "
            "You get e to the score minus 4, which is the new unit. "
            "That factor is the correction, alpha, and here it's 0.135. "
            "On screen, the old weights convert: 0.368 becomes 0.050, and 1 becomes 0.135. "
            "The algorithm never touches them one by one. "
            "One multiply converts each whole total, no matter how many terms it holds. "
            "The sum becomes 0.135 times 1.368, plus 1.368, which is 1.553. "
            "The output becomes 0.135 times 23.68, plus 44.72, which is 47.92. "
            "Divide once at the end: 47.92 divided by 1.553 is 30.86."
        ) as t:
            self.play(FadeIn(board), *[FadeIn(o) for o in vals + w_old + w_new], run_time=0.5)
            t.wait_until("That factor")
            self.play(FadeIn(alpha), run_time=0.7)
            t.wait_until("On screen")
            self.play(FadeOut(board[0][2]), FadeIn(times), run_time=0.4)
            self.play(Transform(w_old[0], w_conv[0]), Transform(w_old[1], w_conv[1]), run_time=1.0)
            t.wait_until("The sum becomes")
            self.play(FadeIn(work[1]), run_time=0.6)
            self.play(FadeIn(work[2]), Transform(vals[1], new_vals[0]), run_time=0.8)
            t.wait_until("The output becomes")
            self.play(FadeIn(work[3]), run_time=0.6)
            self.play(FadeIn(work[4]), Transform(vals[2], new_vals[1]), run_time=0.8)
            t.wait_until("Divide once")
            self.play(FadeIn(result), Create(result_box), run_time=0.8)

    def one_pass_check(self):
        self.slide("one-pass-check", "The ordinary softmax gives 30.86 too",
                   book_section="s-the-online-softmax-on-four-scores")
        tab = kit.table(
            [
                ["$s_j$", "1", "2", "4", "3", ""],
                ["$e^{s_j - 4}$", "0.050", "0.135", "1", "0.368", "sum 1.553"],
                ["$v_j$", "10", "20", "30", "40", ""],
            ],
            header=["", "key 1", "key 2", "key 3", "key 4", ""],
            align=["l", "r", "r", "r", "r", "l"],
            size=th.SIZE_SMALL,
            highlight_rows=[1],
        )
        f = kit.formula(r"\frac{0.050 \cdot 10 + 0.135 \cdot 20 + 1 \cdot 30 + 0.368 \cdot 40}{1.553}",
                        r"= \frac{47.92}{1.553} = 30.86", size=th.SIZE_MATH_SMALL)
        same = kit.callout("key", r"The running totals are kept in units of ``relative to the current maximum.'' "
                                  r"When it rises, $\alpha$ is the exchange rate: one multiply per total.",
                           title="Same answer", width=11.0, size=th.SIZE_SMALL)
        layout = VGroup(tab, f, same).arrange(DOWN, buff=0.5)
        self.place(layout)
        with self.narrate(
            "Now check against the ordinary softmax, with all four scores at once. "
            "The maximum is 4. "
            "The exponentials are 0.050, 0.135, 1, and 0.368. "
            "They sum to 1.553, the same running sum you ended with. "
            "The weighted sum of the values is 47.92. "
            "Divide, and you get 30.86. "
            "It's the same answer. "
            "Think of the running totals as kept in units relative to the current maximum. "
            "When the maximum rises, alpha is the exchange rate, and it costs one multiply per total."
        ) as t:
            self.play(FadeIn(tab.rules), *[FadeIn(VGroup(*tab.cells[r])) for r in (0, 1, 3)], run_time=0.8)
            t.wait_until("The exponentials")
            self.play(FadeIn(tab.bands), FadeIn(VGroup(*tab.cells[2][:5])), run_time=0.8)
            t.wait_until("They sum to")
            self.play(FadeIn(tab.cells[2][5]), run_time=0.5)
            t.wait_until("The weighted sum")
            self.play(FadeIn(f[0]), run_time=0.8)
            t.wait_until("Divide, and")
            self.play(FadeIn(f[1]), run_time=0.6)
            t.wait_until("Think of")
            self.play(FadeIn(same, shift=UP * 0.1), run_time=0.7)

    # ------------------------------------------------------------------
    # Stages 1 to 4

    def streaming_max(self):
        self.slide("streaming-max", "The running maximum never decreases",
                   book_section="s-stage-1-the-streaming-maximum")
        f = kit.formula(r"m^{(t+1)}", r"= \max\!\left(m^{(t)},\ \max_{j \in \Delta_{t+1}} s_j\right)",
                        r",\qquad m^{(0)} = -\infty")
        f[0].set_color(M_C)
        steps = VGroup()
        texts = [(r"m^{(0)} = -\infty", 0.0), (r"m^{(1)} = 2", 0.75), (r"m^{(2)} = 4", 1.5)]
        for i, (tex, h) in enumerate(texts):
            b = kit.box(f"${tex}$", color=M_C, size=th.SIZE_BODY, width=2.8, height=0.95)
            b.move_to([i * 3.9, h, 0])
            steps.add(b)
        les = VGroup()
        for a, b in zip(steps[:-1], steps[1:]):
            arr = kit.arrow(a, b, color=M_C)
            les.add(arr)
        tags = VGroup(kit.label("start", size=th.SIZE_LABEL, color=th.INK_3),
                      kit.label("after block 1", size=th.SIZE_LABEL, color=th.INK_3),
                      kit.label("after block 2", size=th.SIZE_LABEL, color=th.INK_3))
        for tag, s in zip(tags, steps):
            tag.next_to(s, DOWN, buff=0.15)
        fig = VGroup(steps, les, tags)
        layout = VGroup(f, fig).arrange(DOWN, buff=0.8)
        self.place(layout)
        with self.narrate(
            "Now the general algebra, in four stages. "
            "Stage 1 tracks the maximum one block at a time. "
            "The new maximum is the larger of the old maximum and the new block's maximum. "
            "Start at minus infinity, so the first block sets it outright. "
            "In the example, it went from minus infinity to 2 to 4. "
            "The maximum never decreases, and that's what makes the rest safe."
        ) as t:
            t.wait_until("The new maximum")
            self.play(FadeIn(f[0:2]), run_time=0.9)
            t.wait_until("Start at minus")
            self.play(FadeIn(f[2]), FadeIn(steps[0]), FadeIn(tags[0]), run_time=0.7)
            t.wait_until("In the example")
            for i in (1, 2):
                self.play(GrowArrow(les[i - 1].line), FadeIn(steps[i], shift=UP * 0.2), FadeIn(tags[i]),
                          run_time=0.8)

    def rescaling_identity(self):
        self.slide("rescaling-identity", "One scalar moves a row to a new shift",
                   book_section="s-stage-2-the-rescaling-identity")
        d = kit.Derivation(
            [
                r"e^{s_j - m_{\mathrm{new}}} &= e^{s_j - m_{\mathrm{old}} + m_{\mathrm{old}} - m_{\mathrm{new}}}",
                r"&= e^{s_j - m_{\mathrm{old}}} \cdot e^{m_{\mathrm{old}} - m_{\mathrm{new}}}",
            ],
            size=th.SIZE_MATH,
            buff=0.4,
        )
        a = kit.formula(r"\alpha", r"= e^{m_{\mathrm{old}} - m_{\mathrm{new}}} \in (0, 1]")
        a[0].set_color(A_C)
        box = SurroundingRectangle(a, color=A_C, buff=0.2, corner_radius=0.08, stroke_width=3)
        notes = kit.bullets(
            [
                r"No $j$ in $\alpha$: one scalar converts a whole row of exponentials.",
                r"It only shrinks the totals, so it can't overflow.",
                r"A big jump underflows $\alpha$ to zero, which is right.",
            ],
            size=th.SIZE_SMALL, width=10.5, buff=0.22,
        )
        layout = VGroup(d, VGroup(a, box), notes).arrange(DOWN, buff=0.5)
        self.place(layout)
        with self.narrate(
            "Stage 2 turns the exchange rate into one line of algebra. "
            "Add and subtract the old maximum in the exponent. "
            "Then split the exponential of a sum into a product. "
            "The second factor is the correction, alpha. "
            "It doesn't depend on j, so one scalar converts a whole row of exponentials. "
            "Because the maximum only rises, alpha sits between 0 and 1. "
            "So the correction only ever shrinks the totals, and it can't overflow. "
            "If the maximum jumps a long way, alpha underflows to zero. "
            "That's right, because the old terms were negligible anyway."
        ) as t:
            t.wait_until("Add and subtract")
            self.play(d.show(0), run_time=1.0)
            t.wait_until("Then split")
            self.play(d.show(1), run_time=1.0)
            t.wait_until("The second factor")
            self.play(FadeIn(a), Create(box), run_time=0.9)
            t.wait_until("It doesn't depend")
            self.play(FadeIn(notes[0], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("So the correction")
            self.play(FadeIn(notes[1], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("If the maximum jumps")
            self.play(FadeIn(notes[2], shift=RIGHT * 0.1), run_time=0.6)

    def corrected_totals(self):
        self.slide("corrected-totals", "The sum and the output share one correction",
                   book_section="s-stage-3-correcting-the-running-sum")
        size = th.SIZE_MATH
        lf = kit.formula(r"\ell^{(t+1)}", r"=", r"\alpha_{t+1}\, \ell^{(t)}",
                         r"+ \sum_{j \in \Delta_{t+1}} e^{s_j - m^{(t+1)}}", size=size)
        uf = kit.formula(r"u^{(t+1)}", r"=", r"\alpha_{t+1}\, u^{(t)}",
                         r"+ \sum_{j \in \Delta_{t+1}} e^{s_j - m^{(t+1)}}\, v_j", size=size)
        lf[0].set_color(L_C)
        uf[0].set_color(U_C)
        for f in (lf, uf):
            f[2][0].set_color(A_C)
            f[2][1:4].set_color(A_C)
        uf.next_to(lf, DOWN, buff=1.3)
        uf.shift(RIGHT * (lf[1].get_x() - uf[1].get_x()))
        b1 = Brace(lf[2], DOWN, color=th.INK_3, buff=0.1)
        t1 = kit.label("old total, rescaled", size=th.SIZE_LABEL, color=th.INK_3).next_to(b1, DOWN, buff=0.08)
        b2 = Brace(lf[3], DOWN, color=th.INK_3, buff=0.1)
        t2 = kit.label("new block, under the new shift", size=th.SIZE_LABEL, color=th.INK_3).next_to(b2, DOWN,
                                                                                                    buff=0.08)
        b3 = Brace(uf[3], DOWN, color=th.INK_3, buff=0.1)
        t3 = kit.prose(r"a small matmul: $P\,V_j$", size=th.SIZE_LABEL, color=th.INK_3, width=None)
        t3.next_to(b3, DOWN, buff=0.08)
        foot = kit.prose(r"Three running quantities per query row, $m$, $\ell$, $u$, and one $\alpha$ shared by two.",
                         size=th.SIZE_SMALL, color=th.INK_2, width=None)
        fig = VGroup(lf, uf, b1, t1, b2, t2, b3, t3)
        layout = VGroup(fig, foot).arrange(DOWN, buff=0.55)
        self.place(layout)
        with self.narrate(
            "Stages 3 and 4 apply that identity to the two totals. "
            "Split the running sum into old terms and new terms. "
            "The old terms are stored under the old shift, so multiply them by alpha. "
            "Then add the new block's terms, computed directly under the new shift. "
            "The output updates the same way, because each term is linear in its exponential. "
            "Its new term is a small matmul: a row of probabilities against a block of values. "
            "That's the whole algorithm. "
            "Three running quantities per query row, and one correction shared by two of them."
        ) as t:
            t.wait_until("Split the running")
            self.play(FadeIn(lf[0:2]), run_time=0.6)
            t.wait_until("The old terms")
            self.play(FadeIn(lf[2]), FadeIn(b1), FadeIn(t1), run_time=0.8)
            t.wait_until("Then add")
            self.play(FadeIn(lf[3]), FadeIn(b2), FadeIn(t2), run_time=0.8)
            t.wait_until("The output updates")
            self.play(FadeIn(uf[0:3]), run_time=0.8)
            t.wait_until("Its new term")
            self.play(FadeIn(uf[3]), FadeIn(b3), FadeIn(t3), run_time=0.8)
            t.wait_until("Three running")
            self.play(FadeIn(foot), run_time=0.6)

    def exact(self):
        self.slide("exact", "Exact, not approximate", book_section="s-why-the-result-is-exact")
        f = kit.formula(
            r"\frac{u^{(T)}}{\ell^{(T)}}", r"=",
            r"\frac{\sum_j e^{s_j - m^{(T)}}\, v_j}{\sum_j e^{s_j - m^{(T)}}}", r"=",
            r"\sum_j \operatorname{softmax}(s)_j\, v_j",
        )
        n1 = kit.label("1. the definition, which every update keeps", size=th.SIZE_LABEL, color=th.INK_3)
        n2 = kit.label("2. shift invariance at the true maximum", size=th.SIZE_LABEL, color=th.INK_3)
        notes = VGroup(n1, n2).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        rnd = VGroup(
            kit.box(r"float32: about $10^{-5}$", size=th.SIZE_SMALL, color=self.colors[1]),
            kit.box(r"bfloat16: about $10^{-2}$", size=th.SIZE_SMALL, color=self.colors[1]),
        ).arrange(RIGHT, buff=0.5)
        rnd_lab = kit.label("Agreement the labs see: rounding, not approximation", size=th.SIZE_SMALL,
                            color=th.INK_2)
        bottom = VGroup(rnd_lab, rnd).arrange(DOWN, buff=0.3)
        layout = VGroup(f, notes, bottom).arrange(DOWN, buff=0.6)
        self.place(layout)
        n1_tick = kit.label("1", size=th.SIZE_LABEL, color=self.color, weight=SEMIBOLD).next_to(f[1], UP, buff=0.15)
        n2_tick = kit.label("2", size=th.SIZE_LABEL, color=self.color, weight=SEMIBOLD).next_to(f[3], UP, buff=0.15)
        with self.narrate(
            "This streaming algorithm is exact, and the proof is two lines. "
            "After the last block, divide the output by the sum. "
            "The first equality is the definition, which every update preserves. "
            "The second is the softmax's shift invariance, at the true row maximum. "
            "No term is dropped, and only the order of floating-point additions changes. "
            "In float32, the labs see agreement to about 10 to the minus 5. "
            "In bfloat16, it's about 10 to the minus 2. "
            "That's rounding, not approximation."
        ) as t:
            t.wait_until("After the last")
            self.play(FadeIn(f[0:3]), run_time=0.8)
            t.wait_until("The first equality")
            self.play(FadeIn(n1), FadeIn(n1_tick), Indicate(f[1], color=self.color), run_time=0.8)
            t.wait_until("The second is")
            self.play(FadeIn(f[3:]), FadeIn(n2), FadeIn(n2_tick), run_time=0.8)
            t.wait_until("In float32")
            self.play(FadeIn(rnd_lab), FadeIn(rnd[0]), run_time=0.6)
            t.wait_until("In bfloat16")
            self.play(FadeIn(rnd[1]), run_time=0.6)

    # ------------------------------------------------------------------
    # The tiled algorithm

    def tiled_loop(self):
        self.slide("tiled-loop", "Keys and values stream past a resident query tile",
                   book_section="s-the-tiled-algorithm")
        hbm = panel("HBM", 5.0, 5.0, color=self.colors[1])
        chip = panel("One program on an SM", 6.4, 5.0, color=self.color)
        row = VGroup(hbm, chip).arrange(RIGHT, buff=0.9)
        self.place(row)
        q = kit.grid(4, 1, cell=(0.75, 0.9), col_labels=["Q"])
        k = kit.grid(8, 1, cell=(0.75, 0.45), col_labels=["K"])
        v = kit.grid(8, 1, cell=(0.75, 0.45), col_labels=["V"])
        o = kit.grid(4, 1, cell=(0.75, 0.9), col_labels=["O"])
        for g, col in ((q, self.colors[1]), (k, self.colors[2]), (v, self.colors[3]), (o, th.INK_4)):
            g.paint(g.block(0, g.rows, 0, g.cols), col, tint=0.18)
        tensors = VGroup(q, k, v, o).arrange(RIGHT, buff=0.4, aligned_edge=UP)
        tensors.move_to(hbm.rect.get_center() + DOWN * 0.25)
        regs = panel("Registers", 6.0, 2.55, size=th.SIZE_LABEL)
        smem = panel("Shared memory", 6.0, 1.45, size=th.SIZE_LABEL)
        inner = VGroup(regs, smem).arrange(DOWN, buff=0.2)
        inner.move_to(chip.rect.get_center() + DOWN * 0.22)
        # Slots inside the registers panel.
        r0 = regs.rect.get_corner(DL) + UP * 0.25 + RIGHT * 0.3
        q_slot = tile(0.75, 0.9, self.colors[1], "$Q_i$", tint=0.35)
        q_slot.move_to(r0 + RIGHT * 0.375 + UP * 0.9)
        u_slot = tile(0.75, 0.9, U_C, "$u$", tint=0.3)
        u_slot.next_to(q_slot, RIGHT, buff=0.75)
        ml = VGroup(tile(0.3, 0.9, M_C, None, tint=0.3), tile(0.3, 0.9, L_C, None, tint=0.3)).arrange(RIGHT, buff=0.08)
        ml.next_to(u_slot, RIGHT, buff=0.75)
        ml_lab = kit.prose(r"$m$, $\ell$", size=th.SIZE_LABEL, color=th.INK_2, width=None).next_to(ml, DOWN,
                                                                                                    buff=0.1)
        s_slot = tile(0.9, 0.9, self.color, "$S, P$", tint=0.35)
        s_slot.next_to(ml, RIGHT, buff=0.7)
        s_ghost = DashedVMobject(Rectangle(width=0.9, height=0.9, stroke_color=self.color, stroke_width=2),
                                 num_dashes=24)
        s_ghost.move_to(s_slot)
        s_ghost_lab = kit.prose("$S, P$", size=th.SIZE_LABEL, color=th.INK_4, width=None).move_to(s_slot)
        s_lab = kit.label("128 × 64", size=th.SIZE_LABEL, color=th.INK_3).next_to(s_slot, DOWN, buff=0.1)
        q_lab = kit.label("128 × d", size=th.SIZE_LABEL, color=th.INK_3).next_to(q_slot, DOWN, buff=0.1)
        u_lab = kit.label("float32", size=th.SIZE_LABEL, color=th.INK_3).next_to(u_slot, DOWN, buff=0.1)
        kv_y = smem.rect.get_center()[1] - 0.15
        k_slot_pos = np.array([smem.rect.get_left()[0] + 1.0, kv_y, 0])
        v_slot_pos = k_slot_pos + RIGHT * 1.1
        k_tag = kit.prose(r"$K_j$, $V_j$: 64 keys each", size=th.SIZE_LABEL, color=th.INK_3, width=None)
        k_tag.move_to(v_slot_pos + RIGHT * (0.5 + k_tag.width / 2 + 0.2))
        note = kit.label("scores never leave the chip", size=th.SIZE_LABEL, color=self.color)
        note.move_to(regs.head).align_to(regs.rect, RIGHT).shift(LEFT * 0.2)
        with self.narrate(
            "Now turn the four stages into a loop over tiles. "
            "Each program owns a tile of 128 query rows. "
            "It loads that tile into registers once, and it stays there for the whole pass. "
            "It sets the maximum to minus infinity, and both totals to zero. "
            "Then it walks over the keys, 64 at a time, staging each key and value block in shared memory. "
            "The tile's scores, 128 by 64, are computed in registers. "
            "They update the maximum, the sum, and the output, and then they're thrown away. "
            "The output accumulator never leaves registers. "
            "After the last block, the program divides once and writes its output to HBM, exactly once. "
            "The outer loop is the launch grid: one program per query tile, all running at once."
        ) as t:
            self.play(FadeIn(hbm), FadeIn(chip), FadeIn(regs), FadeIn(smem), FadeIn(tensors), run_time=0.9)
            t.wait_until("Each program owns")
            self.play(q.fill(q.cell(0, 0), self.colors[1], tint=0.6), run_time=0.5)
            t.wait_until("It loads that")
            qc = q.cell(0, 0).copy()
            self.play(qc.animate.move_to(q_slot), run_time=0.9)
            self.play(FadeIn(q_slot), FadeIn(q_lab), run_time=0.3)
            self.remove(qc)
            t.wait_until("It sets the maximum")
            self.play(FadeIn(u_slot), FadeIn(ml), FadeIn(ml_lab), FadeIn(u_lab), run_time=0.7)
            t.wait_until("Then it walks")
            self.play(FadeIn(k_tag), Create(s_ghost), FadeIn(s_ghost_lab), FadeIn(s_lab), run_time=0.6)
            prev = None
            for j in range(8):
                slow = j == 0
                kc = k.cell(j, 0).copy()
                vc = v.cell(j, 0).copy()
                kc[0].set_fill(th.tint(self.colors[2], 0.6), opacity=1)
                vc[0].set_fill(th.tint(self.colors[3], 0.6), opacity=1)
                anims = [kc.animate.move_to(k_slot_pos), vc.animate.move_to(v_slot_pos),
                         k.fill(k.cell(j, 0), self.colors[2], tint=0.6), v.fill(v.cell(j, 0), self.colors[3], tint=0.6)]
                if prev is not None:
                    anims.append(FadeOut(prev))
                self.play(*anims, run_time=0.9 if slow else 0.45)
                prev = VGroup(kc, vc)
                if slow:
                    t.wait_until("The tile's scores")
                self.play(FadeIn(s_slot, scale=0.8), run_time=0.6 if slow else 0.25)
                if slow:
                    t.wait_until("They update")
                self.play(Indicate(u_slot, color=U_C, scale_factor=1.08), Indicate(ml, color=M_C, scale_factor=1.08),
                          run_time=0.7 if slow else 0.3)
                self.play(FadeOut(s_slot), run_time=0.5 if slow else 0.2)
            t.wait_until("The output accumulator")
            self.play(FadeIn(note), Indicate(u_slot, color=U_C), run_time=0.8)
            t.wait_until("After the last block")
            uc = u_slot.copy()
            self.play(uc.animate.move_to(o.cell(0, 0)), run_time=1.0)
            self.play(FadeOut(uc), o.fill(o.cell(0, 0), U_C, tint=0.45), run_time=0.4)

    def what_lives_where(self):
        self.slide("what-lives-where", "Every tensor has a level, except the scores",
                   book_section="s-what-lives-where")
        levels = [
            ("Registers", "256 KB per SM", self.color),
            ("Shared memory", "160 KB budget", self.colors[2]),
            ("HBM", "80 GB", self.colors[1]),
        ]
        bands = VGroup()
        for name, sub, col in levels:
            rect = RoundedRectangle(width=12.9, height=1.25, corner_radius=0.1, stroke_color=col,
                                    stroke_width=th.STROKE_BOX, fill_color=th.tint(col, 0.06), fill_opacity=1)
            nm = kit.label(name, size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
            sb = kit.label(sub, size=th.SIZE_LABEL, color=th.INK_3)
            head = VGroup(nm, sb).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
            head.move_to(rect.get_left() + RIGHT * (0.25 + head.width / 2))
            bands.add(VGroup(rect, head))
        bands.arrange(DOWN, buff=0.3)
        big = kit.label("At d = 256, this model's head dimension, u doubles to 128 KiB.", size=th.SIZE_LABEL,
                        color=th.BAD)
        big.next_to(bands, DOWN, buff=0.2).align_to(bands, LEFT)
        self.place(VGroup(bands, big))

        def chip(tex, sub, col):
            return kit.box(tex, sublabel=sub, color=col, size=th.SIZE_SMALL, height=0.95)

        reg_items = [chip("$Q_i$", "32 KiB", self.colors[1]), chip("$u$, float32", "64 KiB", U_C),
                     chip(r"$m$, $\ell$", "1 KiB", M_C), chip("$S$, $P$", "32 KiB each", self.color)]
        smem_items = [chip(r"$K_j$, $V_j$ in bf16, 4 stages", "128 KiB", self.colors[2])]
        hbm_items = [chip("$Q$", None, self.colors[1]), chip("$K$", None, self.colors[2]),
                     chip("$V$", None, self.colors[3]), chip("$O$", None, th.INK_3)]
        x0 = bands[0][0].get_left()[0] + 3.3
        for items, band in zip((reg_items, smem_items, hbm_items), bands):
            VGroup(*items).arrange(RIGHT, buff=0.25)
            g = VGroup(*items)
            g.move_to(band[0]).align_to([x0, 0, 0], LEFT)
        no_s = kit.box("$S$: no address", size=th.SIZE_SMALL, color=th.BAD, height=0.95, filled=False)
        no_s.rect.set_stroke(opacity=0.0)
        dashed = DashedVMobject(no_s.rect.copy().set_stroke(color=th.BAD, opacity=1, width=2), num_dashes=40)
        no_s.next_to(VGroup(*hbm_items), RIGHT, buff=0.4)
        dashed.move_to(no_s.rect)
        with self.narrate(
            "Here's where each tensor lives, with the lab's geometry: 128 query rows, 64 keys per block, and "
            "head dimension 128. "
            "In registers, the query tile takes 32 KiB. "
            "The float32 accumulator takes 64 KiB. "
            "The maximum and the sum take 1 KiB together. "
            "The score and probability tiles are 32 KiB each, and they're transient. "
            "At this model's head dimension of 256, the accumulator doubles to 128 KiB. "
            "In shared memory, four pipeline stages of key and value tiles take 128 KiB, under the 160 KB budget. "
            "In HBM, the kernel keeps Q, K, V, and O. "
            "Nothing else. "
            "The score matrix has no address."
        ) as t:
            self.play(LaggedStart(*[FadeIn(b) for b in bands], lag_ratio=0.2), run_time=1.0)
            for item, cue in zip(reg_items, ["In registers", "The float32 accumulator", "The maximum and the sum",
                                             "The score and probability"]):
                t.wait_until(cue)
                self.play(FadeIn(item, shift=DOWN * 0.15), run_time=0.5)
            t.wait_until("At this model's")
            self.play(FadeIn(big), run_time=0.6)
            t.wait_until("In shared memory")
            self.play(FadeIn(smem_items[0], shift=DOWN * 0.15), run_time=0.6)
            t.wait_until("In HBM")
            self.play(LaggedStart(*[FadeIn(i, shift=DOWN * 0.15) for i in hbm_items], lag_ratio=0.2), run_time=0.9)
            t.wait_until("The score matrix")
            self.play(Create(dashed), FadeIn(no_s.text), run_time=0.8)

    def kernel_loop(self):
        self.slide("kernel-loop", "The loop body is stages 1 to 4, in order",
                   book_section="s-the-kernel-line-by-line")
        panel_ = kit.code(LOOP_SRC, "python", caption="flash_attn_triton.py")
        self.place(panel_)
        with self.narrate(
            "Here's the loop body from the Triton kernel. "
            "First, the tile's scores, computed on the tensor cores. "
            "Then stage 1: the new maximum for each row. "
            "Stage 2: the correction factor, one per row. "
            "Then the probabilities, under the new shift. "
            "Stages 3 and 4: the sum and the accumulator, each rescaled, then extended. "
            "Axis 1 reduces along the keys; get the axis wrong, and nothing crashes, but you get plausible "
            "garbage. "
            "All three running values are float32, even with bfloat16 inputs. "
            "After the loop, there's one division and one store."
        ) as t:
            self.play(FadeIn(panel_.frame), *panel_.reveal(1, 2), run_time=0.7)
            t.wait_until("First, the tile's")
            self.play(*panel_.reveal(3), panel_.highlight(3), run_time=0.6)
            t.wait_until("Then stage 1")
            self.play(*panel_.reveal(4), run_time=0.6)
            t.wait_until("Stage 2")
            self.play(*panel_.reveal(5), run_time=0.6)
            t.wait_until("Then the probabilities")
            self.play(*panel_.reveal(6), run_time=0.6)
            t.wait_until("Stages 3 and 4")
            self.play(*panel_.reveal(7, 8, 9, 10), panel_.highlight(8, 9), run_time=0.8)
            t.wait_until("Axis 1")
            self.play(Circumscribe(panel_.line(8)[-7:], color=th.BAD, buff=0.06), run_time=1.2)
            t.wait_until("After the loop")
            self.play(*panel_.reveal(12), run_time=0.6)

    # ------------------------------------------------------------------
    # Causal masking, GQA, IO

    def causal_blocks(self):
        self.slide("causal-blocks", "Causal attention skips half the blocks",
                   book_section="s-causal-masking-and-the-blocks-you-can-skip")
        rows, cols = 8, 16
        g = kit.grid(rows, cols, cell=0.36)
        top = kit.label("key blocks →", size=th.SIZE_LABEL, color=th.INK_3).next_to(g, UP, buff=0.12).align_to(g, LEFT)
        left = kit.label("query tiles →", size=th.SIZE_LABEL, color=th.INK_3).rotate(-PI / 2)
        left.next_to(g, LEFT, buff=0.15).align_to(g, UP)
        scale = kit.label("each square: 8 × 8 block pairs", size=th.SIZE_LABEL, color=th.INK_3)
        scale.next_to(g, DOWN, buff=0.15).align_to(g, LEFT)
        fig = VGroup(g, top, left, scale)
        f = kit.formula(r"\sum_{i=0}^{63} 2(i+1) = 4160", size=th.SIZE_MATH_SMALL)
        lines = VGroup(
            kit.prose(r"64 query tiles $\times$ 128 key blocks $=$ 8192 pairs", size=th.SIZE_SMALL, width=None),
            f,
            kit.prose(r"$4160 / 8192 = 50.8\%$", size=th.SIZE_SMALL, width=None),
        ).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        key = VGroup()
        for text, col, tint in [("runs, no mask", self.color, 0.55), ("on the diagonal, masked", self.color, 0.2),
                                ("never launched", None, None)]:
            sw = Square(0.3, stroke_color=th.RULE_STRONG, stroke_width=th.STROKE_HAIRLINE,
                        fill_color=th.RAISE if col is None else th.tint(col, tint), fill_opacity=1)
            key.add(VGroup(sw, kit.label(text, size=th.SIZE_LABEL, color=th.INK_2)).arrange(RIGHT, buff=0.15))
        key.arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        side = VGroup(lines, key).arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        layout = VGroup(fig, side).arrange(RIGHT, buff=0.7)
        self.place(layout)
        full = [[g.cell(r, c) for c in range(cols) if c < 2 * r] for r in range(rows)]
        diag = [[g.cell(r, c) for c in (2 * r, 2 * r + 1)] for r in range(rows)]
        with self.narrate(
            "Causal attention lets a query see only keys at or before its own position. "
            "Masking after the matmul saves nothing, because you already paid for it. "
            "The saving comes from never launching the blocks above the diagonal. "
            "At 8192 tokens, there are 64 query tiles and 128 key blocks, so a full pass is 8192 block pairs. "
            "Each query tile runs only up to its own diagonal, two key blocks further than the tile before. "
            "Summed over all 64 tiles, that's 4160 block pairs, or 50.8% of the full count. "
            "The extra 0.8% is the diagonal blocks, which still need the elementwise mask. "
            "During decode, the query length is 1, so the kernel must offset the query to the end of the cache."
        ) as t:
            self.play(FadeIn(fig), run_time=0.8)
            t.wait_until("At 8192 tokens")
            self.play(FadeIn(lines[0]), run_time=0.6)
            t.wait_until("Each query tile")
            for r in range(rows):
                anims = [g.fill(VGroup(*diag[r]), self.color, tint=0.2)]
                if full[r]:
                    anims.append(g.fill(VGroup(*full[r]), self.color, tint=0.55))
                self.play(*anims, run_time=0.35)
            self.play(FadeIn(key), run_time=0.5)
            t.wait_until("Summed over")
            self.play(FadeIn(f), run_time=0.6)
            self.play(FadeIn(lines[2]), run_time=0.5)
            t.wait_until("The extra 0.8%")
            self.play(Indicate(VGroup(*[c for d in diag for c in d]), color=self.color, scale_factor=1.05),
                      run_time=1.0)

    def gqa(self):
        self.slide("gqa", "Six query heads read one KV head", book_section="s-grouped-queries-in-the-kernel")
        cols = [self.color, self.colors[1], self.colors[2], self.colors[3]]
        qs = VGroup()
        for gi in range(4):
            grp = VGroup(*[Square(0.36, stroke_color=cols[gi], stroke_width=th.STROKE_BOX,
                                  fill_color=th.tint(cols[gi], 0.25), fill_opacity=1) for _ in range(6)])
            grp.arrange(RIGHT, buff=0.08)
            qs.add(grp)
        qs.arrange(RIGHT, buff=0.45)
        kvs = VGroup(*[kit.box(f"KV head {i}", color=cols[i], size=th.SIZE_LABEL, width=2.2) for i in range(4)])
        for kv, grp in zip(kvs, qs):
            kv.move_to([grp.get_x(), -2.0, 0])
        links = VGroup()
        for kv, grp in zip(kvs, qs):
            for sq in grp:
                links.add(Line(sq.get_bottom(), kv.get_top(), stroke_width=1.5, color=kv.rect.get_stroke_color()))
        q_lab = kit.label("24 query heads, one program each", size=th.SIZE_LABEL, color=th.INK_3)
        q_lab.next_to(qs, UP, buff=0.15).align_to(qs, LEFT)
        code = kit.mono("kv_head = head // kv_group   # kv_group = 6", size=th.SIZE_SMALL, color=th.INK)
        fig = VGroup(q_lab, qs, links, kvs)
        l2 = kit.callout("key", r"Six programs read the same $K$ and $V$ bytes, mostly from the 40 MB L2: "
                                r"the KV read needs a sixth of the bandwidth.", title="Share by indexing",
                         width=11.5, size=th.SIZE_SMALL)
        layout = VGroup(fig, code, l2).arrange(DOWN, buff=0.45)
        self.place(layout)
        with self.narrate(
            "The model has 24 query heads and 4 KV heads, so six query heads share each KV head. "
            "The naive reference copies K and V six times with repeat interleave. "
            "The kernel indexes the shared head instead: the head number, divided by six. "
            "Six programs now read the same key and value bytes. "
            "They run close together, so the reads mostly hit L2, the GPU's 40 MB on-chip cache. "
            "The bandwidth needed for the KV read drops by a factor of six. "
            "A kernel that duplicates the heads gets the cache savings, and none of the bandwidth savings."
        ) as t:
            self.play(FadeIn(q_lab), LaggedStart(*[FadeIn(s) for g in qs for s in g], lag_ratio=0.03),
                      FadeIn(kvs), run_time=1.2)
            t.wait_until("The kernel indexes")
            self.play(FadeIn(code), LaggedStart(*[Create(l) for l in links], lag_ratio=0.02), run_time=1.4)
            t.wait_until("They run close")
            self.play(FadeIn(l2, shift=UP * 0.1), run_time=0.7)

    def io_complexity(self):
        self.slide("io-complexity", "Traffic stays quadratic, and L2 absorbs it",
                   book_section="s-the-io-complexity-result")
        naive = kit.formula(r"\text{naive: }", r"\Theta(L^2 + L d)", size=th.SIZE_MATH_SMALL)
        tiled = kit.formula(r"\text{tiled: }", r"\frac{L}{B_r} \cdot 2 L d = \Theta\!\left(\frac{L^2 d}{M}\right)",
                            size=th.SIZE_MATH_SMALL)
        forms = VGroup(naive, tiled).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        factor = kit.prose(r"$M \approx 320$ row pairs in 160 KB, so $M/d = 320/128 = 2.5$", size=th.SIZE_SMALL,
                           width=None)
        bars = kit.bar_compare([("K and V, one head", 4.2, self.colors[1], "4.2 MB"),
                                ("K and V, two KV heads", 8.4, self.colors[1], "8.4 MB")],
                               limit=40, limit_label="A100 L2: 40 MB", width=7.0, bar_height=0.42, gap=0.25)
        layout = VGroup(forms, factor, bars).arrange(DOWN, buff=0.55)
        self.place(layout)
        with self.narrate(
            "The naive path moves the score matrix a constant number of times, so its traffic grows with L "
            "squared. "
            "The tiled path re-reads all of K and V once per query tile. "
            "That's still quadratic in L, divided by M, the number of key rows that fit on the chip. "
            "So FlashAttention makes peak memory linear, not traffic. "
            "On an A100, about 320 key and value row pairs fit in 160 KB. "
            "Against a head dimension of 128, that predicts only a 2.5 times saving in traffic. "
            "The gap is L2. "
            "One head's keys and values are 4.2 MB, and the lab's two KV heads are 8.4 MB. "
            "That's well under the 40 MB L2, so most re-reads never reach HBM."
        ) as t:
            t.wait_until("The naive path")
            self.play(FadeIn(naive), run_time=0.7)
            t.wait_until("That's still quadratic")
            self.play(FadeIn(tiled), run_time=0.8)
            t.wait_until("On an A100")
            self.play(FadeIn(factor), run_time=0.7)
            t.wait_until("One head's keys")
            self.play(FadeIn(bars.labels), FadeIn(bars.limit), *bars.grow(), run_time=1.0)
            self.play(FadeIn(bars.values), run_time=0.4)

    # ------------------------------------------------------------------
    # Results

    def speed(self):
        self.slide("speed", "0.63× the vendor kernel, and why", book_section="s-speed-0-63x-and-why")
        bars = kit.bar_compare(
            [("Arithmetic floor", 0.44, th.INK_4, "0.44 ms"),
             ("Vendor kernel", 1.03, self.colors[1], "≈ 1 ms, 43% of peak"),
             ("This kernel, 128 × 64 tiles", 1.64, self.color, "1.64 ms, 27% of peak"),
             ("This kernel, 64 × 64 tiles", 3.35, self.color, "3.35 ms")],
            width=5.6, bar_height=0.42, gap=0.25,
        )
        cap = kit.prose(r"$L = 8192$, 8 heads, $d = 128$, causal: 137 GFLOP at 312 TFLOP/s", size=th.SIZE_SMALL,
                        color=th.INK_3, width=None)
        why = kit.bullets(
            [r"268 million exponentials on the special function units",
             r"A rescale of the whole accumulator every iteration",
             r"No warp specialization or per-shape autotuning"],
            size=th.SIZE_SMALL, width=11.0, buff=0.2,
        )
        layout = VGroup(cap, bars, why).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        self.place(layout)
        with self.narrate(
            "Now speed. "
            "At 8192 tokens, 8 heads, head dimension 128, and causal masking, the two matmuls cost 137 GFLOP. "
            "At the A100's 312 TFLOP/s, the floor is 0.44 ms. "
            "This kernel takes 1.64 ms, about 27% of peak. "
            "The vendor kernel lands near 1 ms, about 43%. "
            "The exponentials, 268 million of them, run on the special function units, not the tensor cores. "
            "The rescale touches the whole accumulator every iteration. "
            "And vendor kernels use warp specialization and tile sizes tuned per shape. "
            "One tile size alone matters a lot: at 64 by 64, this kernel takes 3.35 ms, twice as long. "
            "Within 2 times of a vendor kernel, with 40 readable lines, is a good result."
        ) as t:
            self.play(FadeIn(cap), FadeIn(bars.labels[:3]), run_time=0.7)
            t.wait_until("At the A100")
            self.play(GrowFromEdge(bars.bars[0], LEFT), FadeIn(bars.values[0]), run_time=0.8)
            t.wait_until("This kernel takes")
            self.play(GrowFromEdge(bars.bars[2], LEFT), FadeIn(bars.values[2]), run_time=0.8)
            t.wait_until("The vendor kernel")
            self.play(GrowFromEdge(bars.bars[1], LEFT), FadeIn(bars.values[1]), run_time=0.8)
            for i, cue in enumerate(["The exponentials", "The rescale touches", "And vendor kernels"]):
                t.wait_until(cue)
                self.play(FadeIn(why[i], shift=RIGHT * 0.1), run_time=0.5)
            t.wait_until("One tile size")
            self.play(FadeIn(bars.labels[3]), GrowFromEdge(bars.bars[3], LEFT), FadeIn(bars.values[3]), run_time=0.9)

    def memory(self):
        self.slide("memory", "27.5× less peak memory", book_section="s-memory-27-5x")
        bars = kit.bar_compare(
            [("Naive", 550, self.colors[1], "scores 512 + copies 38 MiB"),
             ("FlashAttention", 20, self.color, "Q, K, V, O: 20 MiB")],
            width=6.5, bar_height=0.55, gap=0.35,
        )
        cap = kit.prose(r"Peak allocation at $L = 4096$, 8 heads, $d = 128$", size=th.SIZE_SMALL, color=th.INK_3,
                        width=None)
        f = kit.formula(r"8 \times 4096 \times 4096 \times 4 = 536{,}870{,}912 \text{ bytes} = 512 \text{ MiB}",
                        size=th.SIZE_MATH_SMALL)
        ratio = kit.formula(r"\frac{512 + 38}{20} = 27.5", size=th.SIZE_MATH)
        ratio_box = SurroundingRectangle(ratio, color=self.color, buff=0.18, corner_radius=0.08, stroke_width=3)
        layout = VGroup(cap, f, bars, VGroup(ratio, ratio_box)).arrange(DOWN, buff=0.45)
        self.place(layout)
        with self.narrate(
            "The memory result is the one that changes what you can serve. "
            "The lab measures peak allocation at 4096 tokens, 8 heads, and head dimension 128. "
            "The naive path holds the scores in float32, which is 512 MiB. "
            "FlashAttention's live set is Q, K, V, and O, which is 20 MiB. "
            "Add the 38 MiB of float32 input copies the reference makes, and the ratio is 27.5. "
            "Peak memory, not clock time, is why every serving stack uses FlashAttention. "
            "It's the difference between 8k context and 200k context on the same card."
        ) as t:
            self.play(FadeIn(cap), run_time=0.5)
            t.wait_until("The naive path")
            self.play(FadeIn(f), FadeIn(bars.labels[0]), GrowFromEdge(bars.bars[0], LEFT), FadeIn(bars.values[0]),
                      run_time=1.0)
            t.wait_until("FlashAttention's live")
            self.play(FadeIn(bars.labels[1]), GrowFromEdge(bars.bars[1], LEFT), FadeIn(bars.values[1]), run_time=0.9)
            t.wait_until("Add the 38")
            self.play(FadeIn(ratio), Create(ratio_box), run_time=0.8)
