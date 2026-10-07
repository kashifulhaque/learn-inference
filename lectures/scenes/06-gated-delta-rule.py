"""Chapter 06, linear attention and the gated delta rule.

Teaches content/chapters/06-gated-delta-rule.md in its order: attention as a
running sum, why a running sum can't forget, the delta rule, the forget gate,
the sequential form, the chunked form, what it costs, and the log-space decay
ratios. The rest of the layer (convolution, gated RMSNorm) and head sharing
are left to the reader. Every number below was recomputed with Python and
numpy; chapter values match.

Linear attention (d_k = d_v = 128):
- One read S^T q: 2 x 128 x 128 = 32,768 FLOPs, at any context length.
- One head's state: 128 x 128 = 16,384 float32 numbers = 64 KiB.
- 48 value heads x 48 linear layers x 65,536 B = 150,994,944 B = 144 MiB.
- Convolution cache 10,240 x 4 x 2 B x 48 layers = 3.75 MiB; total
  147.75 MiB ("147.8 MiB", chapter 2).

Two-drawer memory (d_k = d_v = 2, the lecture's own small numbers):
- e1 = (1, 0), e2 = (0, 1), v_a = (1, 2), v_b = (3, -1): S = [[1, 2], [3, -1]].
- k3 = (0.707, 0.707), v_c = (2, 2): k3 v_c^T adds 1.414 to every entry;
  S = [[2.41, 3.41], [4.41, 0.41]]. S^T e1 = (2.41, 3.41) = v_a + 0.707 v_c.
- Interference: SNR ~ d_k / (n - 1); 128 / (n - 1) = 1 at n = 129.
  After 200 writes, 199 / 128 = 1.55 ("about 1.6 times", chapter).

Double write with k = e1: plain sum gives row 1 = (1, 2) + (3, -1) = (4, 1),
the sum. Delta rule at beta = 1: (I - k k^T) erases row 1 to (0, 0), then
adding k v_b^T gives (3, -1); S^T k = (3, -1) = v_b.

Unit keys: eigenvalues 1 - beta along k, 1 on the other d_k - 1 = 127
directions. Norm 2 at beta = 0.8: 1 - 0.8 x 4 = -2.2 (chapter).

One gated delta step (lecture's own numbers): S = [[1, 2], [3, -1]],
alpha = 0.5, beta = 0.5, k = (0.6, 0.8) (unit), v = (2, 1).
- alpha S = [[0.5, 1], [1.5, -0.5]].
- read = (alpha S)^T k = 0.6 (0.5, 1) + 0.8 (1.5, -0.5) = (1.5, 0.2).
- v - read = (0.5, 0.8); delta = 0.5 x that = (0.25, 0.4).
- k delta^T = [[0.15, 0.24], [0.2, 0.32]].
- S_t = [[0.65, 1.24], [1.7, -0.18]]; same as (I - b k k^T) a S + b k v^T.
- S_t^T k = (1.75, 0.6): halfway from (1.5, 0.2) to (2, 1).
- k_perp = (-0.8, 0.6): (alpha S)^T k_perp = S_t^T k_perp = (0.5, -1.1).

Time constants, softplus = 1 so r = e^A_log (chapter table):
- A_log 0: alpha 0.368, n_half 0.69, n_90 2.3.
- A_log -2: r 0.1353, alpha 0.8734, n_half 5.1, n_90 17.0.
- A_log -4: r 0.01832, alpha 0.9819, n_half 37.8 (38), n_90 125.7 (126).
- A_log -6: alpha 0.99752, n_half 279.6 (280), n_90 928.9 (929).
- A_log -8: alpha 0.99966, n_half 2066, n_90 6864.
- e^2 = 7.389 ("7.4").

Ledger illustration: a six-token chunk with every gate 0.8, so
g_t / g_j = 0.8^(t - j): 1, 0.8, 0.64, 0.512 (0.51), 0.4096 (0.41),
0.32768 (0.33).

Chunk loop: 2048 / 64 = 32 iterations instead of 2048.

Cost per token, d = 128, C = 64: chunked 7Cd + 6d^2 = 57,344 + 98,304 =
155,648; sequential 7d^2 = 114,688; ratio 1.357 ("1.36"). Break-even
C = d / 7 = 18.3 ("about 18"). A100: 108 SMs. Lab: more than 5x faster at
2048 tokens.

Log space: float32 smallest subnormal 2^-149 = 1.401e-45 ("1.4 x 10^-45");
alpha = 0.5 gives g_t = 2^-t: 2^-149 still representable, 2^-150 rounds to
0, so g_150 = 0 (chapter: "at t = 150, exactly zero"). log10 g_t =
-0.30103 t crosses -44.85 at t = 149. alpha = 1e-8: ln = -18.42; over 64
tokens -1178.9 ("-1179").
"""

from manim import *

import kit
from kit import theme as th

# The chapter's maths colours: alpha (hla), beta (hlb), the read (hlc), the
# interference (hld).
A_C = th.HL_A
B_C = th.HL_B
R_C = th.HL_C
I_C = th.HL_D

STEP_SRC = """\
def delta_rule_step(state, q, k, v, alpha, beta):
    # state: (batch, heads, k_dim, v_dim), float32
    alpha = alpha[..., None, None].float()   # (b, h, 1, 1)
    beta = beta[..., None].float()           # (b, h, 1)
    q, k, v = q.float(), k.float(), v.float()
    read = torch.einsum("bhkv,bhk->bhv", state, k)
    delta = beta * (v - alpha[..., 0] * read)
    state = alpha * state + torch.einsum(
        "bhk,bhv->bhkv", k, delta)
    out = torch.einsum("bhkv,bhk->bhv", state, q)  # post-write
    return state, out
"""


def num(text, color=th.INK_2, size=th.SIZE_SMALL):
    """A matrix entry in the code font."""
    return kit.label(text, size=size, color=color, font=th.MONO)


def matrix(values, cell=(1.25, 0.78), size=th.SIZE_SMALL):
    """A 2 x 2 (or r x c) state matrix with an entry in every cell."""
    rows, cols = len(values), len(values[0])
    g = kit.grid(rows, cols, cell=cell, values=values, size=size)
    for row in g.cells:
        for c in row:
            if c.text is not None:
                c.text.set_z_index(2)
    return g


def entry(g, r, c, text, color=th.INK_2, size=th.SIZE_SMALL):
    """A new entry for cell (r, c), placed, for a Transform."""
    t = num(text, color=color, size=size)
    box = g.cell(r, c).square
    if t.width > box.width * 0.86:
        t.scale_to_fit_width(box.width * 0.86)
    t.move_to(box)
    return t


def retext(g, values, color=th.INK_2, cells=None):
    """Transforms that rewrite the entries of `g` to `values`."""
    anims = []
    for r, row in enumerate(values):
        for c, v in enumerate(row):
            if cells is not None and (r, c) not in cells:
                continue
            anims.append(Transform(g.cell(r, c).text, entry(g, r, c, v, color=color)))
    return anims


def row_tags(g, texts, color=th.INK_3, size=th.SIZE_SMALL):
    """Maths labels left of each row of a grid."""
    out = VGroup()
    for r, tex in enumerate(texts):
        lab = kit.math_label(tex, size=size, color=color)
        lab.next_to(g.cell(r, 0), LEFT, buff=0.2)
        out.add(lab)
    return out


def name_above(g, tex, color=th.INK, size=th.SIZE_BODY):
    lab = kit.math_label(tex, size=size, color=color)
    lab.next_to(g, UP, buff=0.18)
    return lab


def line(tex, size=th.SIZE_SMALL, color=th.INK_2):
    return kit.prose(tex, size=size, color=color, width=None)


class Lecture(kit.Lecture):
    def construct(self):
        self.title_slide(
            narration=(
                "Linear attention and the gated delta rule. "
                "Three quarters of this model's layers don't keep a KV cache. "
                "Each head keeps one fixed 128 by 128 matrix instead. "
                "You build its update rule, then the chunked form that makes prefill fast."
            ),
            summary=("The recurrence that runs 48 of this model's 64 layers, built up from attention as a "
                     "running sum, and the chunked form that makes prefill fast."),
        )
        self.softmax_keeps_all()
        self.running_sum()
        self.constant_read()
        self.two_drawers()
        self.interference()
        self.double_write()
        self.delta_rule()
        self.erase_then_add()
        self.unit_keys()
        self.forget_gate()
        self.step_read()
        self.step_write()
        self.time_constant()
        self.sequential()
        self.ledger()
        self.triangular_solve()
        self.carried_state()
        self.chunk_cost()
        self.log_space()
        self.closing_slide(
            [
                r"A fixed $128 \times 128$ state per head; reads cost the same at any length.",
                r"Erase along a unit key, then add; $\alpha$ fades the rest.",
                r"Chunks: 1.36$\times$ the FLOPs, over 5$\times$ faster.",
                r"Decay ratios via $\exp(\log g_t - \log g_j)$.",
            ],
            narration=(
                "Here's what to keep. "
                "Each head keeps a fixed 128 by 128 state, so a read costs the same at any length. "
                "The delta rule erases along a unit key, then adds, and the gate fades the rest. "
                "Chunks solve one triangular system each, with 1.36 times the arithmetic and over 5 times the "
                "speed. "
                "And decay ratios come from subtracted logarithms, never a quotient."
            ),
            book_section="s-what-goes-wrong",
        )

    # ------------------------------------------------------------------
    # Attention as a running sum

    def softmax_keeps_all(self):
        self.slide("softmax-keeps-all", "Softmax ties the query to every key",
                   book_section="s-attention-as-a-running-sum")
        size = th.SIZE_MATH
        num_ = kit.formula(r"\sum_{j \le t} \exp\big(", r"q_t", r"^{\top} k_j / \sqrt{d_k}\big)\, v_j", size=size)
        den = kit.formula(r"\sum_{j \le t} \exp\big(", r"q_t", r"^{\top} k_j / \sqrt{d_k}\big)", size=size)
        bar = Line(LEFT, RIGHT, color=th.INK, stroke_width=2.5)
        bar.set_width(max(num_.width, den.width) + 0.2)
        frac = VGroup(num_, bar, den).arrange(DOWN, buff=0.16)
        lhs = kit.formula(r"o_t =", size=size)
        lhs.next_to(bar, LEFT, buff=0.25)
        f = VGroup(lhs, frac)
        dims = line(r"Here $d_k = d_v = 128$.", color=th.INK_3)
        top = VGroup(f, dims).arrange(DOWN, buff=0.35)

        cols = 14
        cache = kit.grid(2, cols, cell=(0.62, 0.5), gap=0.05)
        tags = VGroup(kit.label("keys", size=th.SIZE_LABEL, color=th.INK_3),
                      kit.label("values", size=th.SIZE_LABEL, color=th.INK_3))
        for r, tg in enumerate(tags):
            tg.next_to(cache.cell(r, 0), LEFT, buff=0.2)
        cap = kit.label("KV cache: one key and one value per past token, kept forever", size=th.SIZE_LABEL,
                        color=self.color)
        cap.next_to(cache, DOWN, buff=0.2).align_to(cache, LEFT)
        fig = VGroup(cache, tags, cap)
        layout = VGroup(top, fig).arrange(DOWN, buff=0.7)
        self.place(layout)
        with self.narrate(
            "Start with what you're replacing. "
            "Softmax attention weights every past value by the exponential of a query-key dot product. "
            "Look at where the query sits. "
            "It's inside the exponential, coupled to every key, so you can't pull it out of the sum. "
            "That's why softmax attention keeps every past key and value in the KV cache. "
            "And that cache grows with every token."
        ) as t:
            t.wait_until("Softmax attention weights")
            self.play(FadeIn(f), FadeIn(dims), run_time=0.9)
            t.wait_until("Look at where")
            self.play(num_[1].animate.set_color(self.color), den[1].animate.set_color(self.color), run_time=0.6)
            t.wait_until("It's inside")
            self.play(Circumscribe(num_[1], color=self.color, buff=0.06),
                      Circumscribe(den[1], color=self.color, buff=0.06), run_time=1.2)
            t.wait_until("That's why softmax")
            self.play(FadeIn(cache), FadeIn(tags), FadeIn(cap), run_time=0.6)
            t.wait_until("And that cache grows")
            self.play(LaggedStart(*[cache.fill(cache.col(c), self.color) for c in range(cols)], lag_ratio=0.25),
                      run_time=t.remaining * 0.9)

    def running_sum(self):
        self.slide("running-sum", "Drop the exponential, and the past folds into one matrix",
                   book_section="s-attention-as-a-running-sum")
        d = kit.Derivation(
            [
                r"o_t &= \sum_{j \le t} (q_t^\top k_j)\, v_j",
                r"&= \Big( \sum_{j \le t} k_j v_j^\top \Big)^{\!\top} q_t",
            ],
            size=th.SIZE_MATH_SMALL,
            buff=0.4,
        )
        boxed = kit.formula(r"S_t = S_{t-1} + k_t v_t^\top,", r"\qquad o_t = S_t^\top q_t", size=th.SIZE_MATH_SMALL)
        box = SurroundingRectangle(boxed, color=self.color, buff=0.18, corner_radius=0.08, stroke_width=3)
        left = VGroup(d, VGroup(boxed, box)).arrange(DOWN, buff=0.6, aligned_edge=LEFT)

        n = 4
        c = 0.55
        S = kit.grid(n, n, cell=c)
        s_lab = kit.math_label(r"S \in \mathbb{R}^{d_k \times d_v}", size=th.SIZE_SMALL, color=th.INK)
        s_lab.next_to(S, UP, buff=0.2)
        s_note = kit.label("size never changes", size=th.SIZE_LABEL, color=self.color)
        s_note.next_to(S, DOWN, buff=0.2)
        tile = kit.grid(n, n, cell=c)
        kcol = kit.grid(n, 1, cell=c)
        vrow = kit.grid(1, n, cell=c)
        kcol.paint(kcol.block(0, n, 0, 1), self.colors[1])
        vrow.paint(vrow.block(0, 1, 0, n), self.colors[2])
        kcol.next_to(tile, LEFT, buff=0.15)
        vrow.next_to(tile, UP, buff=0.15)
        k_lab = kit.math_label("k_t", size=th.SIZE_SMALL, color=self.colors[1]).next_to(kcol, LEFT, buff=0.15)
        v_lab = kit.math_label(r"v_t^\top", size=th.SIZE_SMALL, color=self.colors[2]).next_to(vrow, UP, buff=0.12)
        outer = VGroup(tile, kcol, vrow, k_lab, v_lab)
        plus = kit.formula("+", size=th.SIZE_MATH)
        right = VGroup(outer, plus, VGroup(s_lab, S, s_note)).arrange(RIGHT, buff=0.45)
        layout = VGroup(left, right).arrange(RIGHT, buff=0.9)
        self.place(layout)
        rng = kit.rng(6)
        level = np.zeros((n, n))
        with self.narrate(
            "Replace the exponential score with the plain dot product, and drop the denominator. "
            "The output becomes a sum of values, weighted by dot products. "
            "Move the brackets, and the query comes out of the sum. "
            "Call what's left S. "
            "It's a running total: each token adds one outer product, its key times its value, transposed. "
            "Reading is a matrix-vector product with the query. "
            "That's linear attention. "
            "S is 128 by 128, its size never changes, and it replaces the whole KV cache for that head."
        ) as t:
            self.play(FadeIn(S), FadeIn(s_lab), run_time=0.6)
            t.wait_until("The output becomes")
            self.play(d.show(0), run_time=0.9)
            t.wait_until("Move the brackets")
            self.play(d.show(1), run_time=1.0)
            t.wait_until("It's a running total")
            for i in range(3):
                kv = rng.uniform(0.2, 1.0, size=n)
                vv = rng.uniform(0.2, 1.0, size=n)
                prod = np.outer(kv, vv)
                new = tile.copy()
                for r in range(n):
                    for cc in range(n):
                        new.cell(r, cc).square.set_fill(th.tint(self.color, 0.08 + 0.45 * prod[r, cc]), opacity=1)
                self.play(FadeIn(kcol), FadeIn(vrow), FadeIn(k_lab), FadeIn(v_lab), FadeIn(tile), FadeIn(new),
                          run_time=0.5 if i else 0.8)
                level = np.minimum(level + 0.22 * prod, 0.7)
                self.play(new.animate.move_to(S), run_time=0.6 if i else 0.9)
                self.play(FadeOut(new),
                          *[S.cell(r, cc).square.animate.set_fill(th.tint(self.color, 0.1 + level[r, cc]), opacity=1)
                            for r in range(n) for cc in range(n)], run_time=0.4)
            t.wait_until("Reading is a matrix")
            self.play(FadeIn(boxed), Create(box), run_time=0.8)
            t.wait_until("S is 128 by")
            self.play(FadeIn(s_note), Indicate(S, color=self.color, scale_factor=1.05), run_time=1.0)

    def constant_read(self):
        self.slide("constant-read", "A read costs the same at token 10 and token 100,000",
                   book_section="s-attention-as-a-running-sum")
        ax = kit.axes([0, 100000, 25000], [0, 10, 2], width=5.6, height=3.0,
                      x_label=r"\text{context length}", y_label=r"\text{cost of one read}")
        soft = ax.plot(lambda x: 9.0 * x / 100000 + 0.2, x_range=[0, 100000], color=self.colors[1],
                       stroke_width=th.STROKE_CURVE)
        flat = ax.plot(lambda x: 1.2, x_range=[0, 100000], color=self.color, stroke_width=th.STROKE_CURVE)
        s_lab = kit.label("softmax: grows with length", size=th.SIZE_LABEL, color=self.colors[1])
        s_lab.next_to(ax.c2p(52000, 4.9), UL, buff=0.1)
        f_lab = kit.label("linear: 32,768 FLOPs", size=th.SIZE_LABEL, color=self.color)
        f_lab.next_to(ax.c2p(100000, 1.2), UP, buff=0.12).align_to(ax.c2p(100000, 0), RIGHT)
        dots = VGroup(Dot(ax.c2p(10, 1.2), radius=0.08, color=self.color),
                      Dot(ax.c2p(100000, 1.2), radius=0.08, color=self.color))
        plot = VGroup(ax, soft, flat, s_lab, f_lab, dots)
        tab = kit.table(
            [
                [r"one read, $S^\top q$", "32,768 FLOPs"],
                ["one head's state", "16,384 numbers"],
                [r"48 heads $\times$ 48 layers", "144 MiB"],
                ["plus the conv cache", "147.8 MiB"],
            ],
            header=["Per sequence, float32", ""],
            align=["l", "r"],
            size=th.SIZE_SMALL,
        )
        top = VGroup(plot, tab).arrange(RIGHT, buff=0.7)
        warn = kit.callout("warning", r"No exponential, no normalizer: a different operator, trained as itself.",
                           title="Not a cheap softmax", width=11.0, size=th.SIZE_SMALL)
        layout = VGroup(top, warn).arrange(DOWN, buff=0.5)
        self.place(layout)
        with self.narrate(
            "One head's read is 32,768 FLOPs, at token 10 and at token 100,000. "
            "The state holds 16,384 numbers per head, in float32. "
            "Across 48 value heads and 48 linear layers, that's 144 MiB per sequence. "
            "Add a small convolution cache, and you get chapter 2's 147.8 MiB, at any context length. "
            "It isn't a cheap softmax, though: with no exponential and no normalizer, it's a different operator, "
            "trained as itself."
        ) as t:
            self.play(Create(ax), run_time=0.8)
            self.play(Create(soft), FadeIn(s_lab), run_time=1.0)
            t.wait_until("One head's read")
            self.play(Create(flat), FadeIn(f_lab), FadeIn(dots), FadeIn(tab.rules), FadeIn(VGroup(*tab.cells[0])),
                      FadeIn(VGroup(*tab.cells[1])), run_time=1.0)
            for r, cue in zip((2, 3, 4), ["The state holds", "Across 48", "Add a small"]):
                t.wait_until(cue)
                self.play(FadeIn(VGroup(*tab.cells[r]), shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("It isn't a cheap")
            self.play(FadeIn(warn, shift=UP * 0.1), run_time=0.7)

    # ------------------------------------------------------------------
    # Why a running sum can't forget

    def two_drawers(self):
        self.slide("two-drawers", "A third key has no drawer of its own",
                   book_section="s-why-a-running-sum-can-t-forget")
        g = matrix([["0", "0"], ["0", "0"]], cell=(1.5, 0.85))
        tags = row_tags(g, ["e_1", "e_2"])
        sname = name_above(g, "S")
        fig = VGroup(sname, g, tags)
        steps = VGroup(
            line(r"write $v_a = (1, 2)$ under $e_1 = (1, 0)$"),
            line(r"write $v_b = (3, -1)$ under $e_2 = (0, 1)$"),
            line(r"read $e_1$: $(1, 2)$, exactly $v_a$", color=th.GOOD),
            line(r"write $v_c = (2, 2)$ under $k_3 = (0.707, 0.707)$"),
            line(r"read $e_1$: $(2.41, 3.41) = v_a + 0.707\, v_c$", color=th.BAD),
        ).arrange(DOWN, buff=0.38, aligned_edge=LEFT)
        layout = VGroup(fig, steps).arrange(RIGHT, buff=1.1)
        self.place(layout)
        with self.narrate(
            "Picture the state as a filing cabinet, where each key labels a drawer. "
            "Write the value 1, 2 with the first of two perpendicular keys, and it lands in row 1. "
            "Write 3, minus 1 with the second key, and it lands in row 2. "
            "Reading with the first key returns exactly 1, 2. "
            "Now write a third value, 2, 2, with a key halfway between the two. "
            "There's no third drawer, so it lands partly in both rows. "
            "Reading with the first key now gives 2.41, 3.41. "
            "That's the value you asked for, blurred by 0.707 times one you didn't."
        ) as t:
            self.play(FadeIn(fig), run_time=0.7)
            t.wait_until("Write the value 1, 2")
            self.play(FadeIn(steps[0]), g.fill(g.row(0), self.colors[1]), *retext(g, [["1", "2"]]), run_time=0.8)
            t.wait_until("Write 3, minus 1")
            self.play(FadeIn(steps[1]), g.fill(g.row(1), self.colors[2]), *retext(g, [["1", "2"], ["3", "−1"]],
                                                                                    cells={(1, 0), (1, 1)}),
                      run_time=0.8)
            t.wait_until("Reading with the first key returns")
            box = g.outline(0, 1, 0, 2, color=th.GOOD)
            self.play(Create(box), FadeIn(steps[2]), run_time=0.7)
            t.wait_until("Now write a third")
            self.play(FadeOut(box), FadeIn(steps[3]), run_time=0.5)
            t.wait_until("There's no third")
            self.play(g.fill(g.block(0, 2, 0, 2), self.color),
                      *retext(g, [["2.41", "3.41"], ["4.41", "0.41"]], color=th.INK), run_time=1.0)
            t.wait_until("Reading with the first key now")
            box2 = g.outline(0, 1, 0, 2, color=th.BAD)
            self.play(Create(box2), FadeIn(steps[4]), run_time=0.7)

    def interference(self):
        self.slide("interference", "Past about 128 writes, reads come back blurred",
                   book_section="s-why-a-running-sum-can-t-forget")
        f = kit.formula(r"S^\top k_m = v_m \,+", r"\underbrace{\sum_{i \ne m} (k_i^\top k_m)\, v_i}"
                        r"_{\text{interference}}", size=th.SIZE_MATH_SMALL)
        f[1].set_color(I_C)
        snr = kit.formula(r"\mathrm{SNR} \approx \frac{d_k}{n - 1}", size=th.SIZE_MATH_SMALL)
        notes = kit.bullets(
            [
                r"Orthogonal keys read back exactly, but only 128 fit.",
                r"After 200 writes, the noise has about $1.6\times$ the signal's energy.",
            ],
            size=th.SIZE_SMALL, width=5.4, buff=0.3,
        )
        left = VGroup(f, snr, notes).arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        ax = kit.axes([0, 512, 64], [0, 4, 1], width=5.4, height=3.6,
                      x_label=r"\text{writes } n", y_label=r"\mathrm{SNR}",
                      x_numbers=[0, 128, 256, 384, 512], y_numbers=[0, 1, 2, 3, 4])
        curve = ax.plot(lambda n: 128 / (n - 1), x_range=[33, 512], color=I_C, stroke_width=th.STROKE_CURVE)
        h = DashedLine(ax.c2p(0, 1), ax.c2p(512, 1), color=th.INK_4, stroke_width=th.STROKE_AXIS)
        dot = Dot(ax.c2p(129, 1), radius=0.09, color=I_C)
        tag = kit.label("noise = signal at n = 129", size=th.SIZE_LABEL, color=th.INK_2)
        tag.next_to(dot, UR, buff=0.12)
        plot = VGroup(ax, curve, h, dot, tag)
        layout = VGroup(left, plot).arrange(RIGHT, buff=0.7)
        self.place(layout)
        with self.narrate(
            "The same thing happens at full size. "
            "Read back one of n associations, and you get the value you asked for, plus interference. "
            "That's every other value, weighted by how much its key overlaps yours. "
            "If the keys were orthogonal, the interference would vanish, but only 128 orthogonal directions fit. "
            "For random unit keys, the signal-to-noise ratio is about 128 over n minus 1. "
            "Noise matches signal at about 129 writes. "
            "After 200 writes, the noise has about 1.6 times the signal's energy."
        ) as t:
            t.wait_until("Read back one")
            self.play(FadeIn(f[0]), run_time=0.6)
            t.wait_until("That's every other")
            self.play(FadeIn(f[1]), run_time=0.8)
            t.wait_until("If the keys were")
            self.play(FadeIn(notes[0], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("For random unit")
            self.play(FadeIn(snr), Create(ax), run_time=0.9)
            self.play(Create(curve), run_time=1.4)
            t.wait_until("Noise matches")
            self.play(Create(h), FadeIn(dot, scale=0.5), FadeIn(tag), run_time=0.8)
            t.wait_until("After 200 writes")
            self.play(FadeIn(notes[1], shift=RIGHT * 0.1), run_time=0.6)

    def double_write(self):
        self.slide("double-write", "Writing the same key twice returns the sum",
                   book_section="s-why-a-running-sum-can-t-forget")
        g = matrix([["0", "0"], ["0", "0"]], cell=(1.5, 0.85))
        tags = row_tags(g, ["k", r"k_\perp"])
        sname = name_above(g, "S")
        fig = VGroup(sname, g, tags)
        steps = VGroup(
            line(r"token 3 writes $v_a = (1, 2)$ under $k$"),
            line(r"token 90 writes $v_b = (3, -1)$ under $k$"),
        ).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        f = kit.formula(r"S = k v_a^\top + k v_b^\top = k\,(v_a + v_b)^\top", size=th.SIZE_MATH_SMALL)
        r_ = kit.formula(r"S^\top k = v_a + v_b = (4,\ 1)", size=th.SIZE_MATH_SMALL, color=th.BAD)
        right = VGroup(steps, f, r_).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        top = VGroup(fig, right).arrange(RIGHT, buff=1.0)
        key = kit.callout("key", "The read returns the sum, not the update. The state can't express replacement.",
                          title="Plain accumulation can only add", width=10.5, size=th.SIZE_SMALL)
        layout = VGroup(top, key).arrange(DOWN, buff=0.6)
        self.place(layout)
        with self.narrate(
            "The clearest failure is an overwrite. "
            "Say token 3 writes the value 1, 2 under some key. "
            "Token 90 writes 3, minus 1 under the same key, meaning this slot now holds the new value. "
            "Plain accumulation adds the two. "
            "Reading the key returns 4, 1: the sum, not the update. "
            "The state has no way to express replacement, so every overwrite makes the entry worse."
        ) as t:
            self.play(FadeIn(fig), run_time=0.6)
            t.wait_until("Say token 3")
            self.play(FadeIn(steps[0]), g.fill(g.row(0), self.colors[1]), *retext(g, [["1", "2"]]), run_time=0.8)
            t.wait_until("Token 90 writes")
            self.play(FadeIn(steps[1]), run_time=0.6)
            t.wait_until("Plain accumulation adds")
            self.play(g.fill(g.row(0), th.BAD), *retext(g, [["4", "1"]], color=th.INK), FadeIn(f), run_time=1.0)
            t.wait_until("Reading the key")
            self.play(Create(g.outline(0, 1, 0, 2, color=th.BAD)), FadeIn(r_), run_time=0.7)
            t.wait_until("The state has no way")
            self.play(FadeIn(key, shift=UP * 0.1), run_time=0.7)

    # ------------------------------------------------------------------
    # The delta rule

    def delta_rule(self):
        self.slide("delta-rule", "The delta rule writes only what's missing",
                   book_section="s-the-delta-rule-write-only-what-s-missing")
        f = kit.formula(r"S_t = S_{t-1} +", r"\beta_t", r"\, k_t\, \big(v_t -", r"S_{t-1}^\top k_t", r"\big)^\top",
                        size=th.SIZE_MATH)
        f[1].set_color(B_C)
        f[3].set_color(R_C)
        box = SurroundingRectangle(f, color=self.color, buff=0.2, corner_radius=0.08, stroke_width=3)
        brace = Brace(f[3], DOWN, buff=0.12, color=th.INK_3)
        b_lab = kit.label("what the state returns now", size=th.SIZE_LABEL, color=R_C).next_to(brace, DOWN, buff=0.08)
        top = VGroup(VGroup(f, box), brace, b_lab)
        d = kit.Derivation(
            [
                r"\mathcal{L}(S) &= \tfrac{1}{2} \lVert S^\top k_t - v_t \rVert^2",
                r"\nabla_S \mathcal{L} &= k_t\, (S^\top k_t - v_t)^\top",
                r"S_t &= S_{t-1} - \beta_t\, \nabla_S \mathcal{L}",
            ],
            size=th.SIZE_MATH_SMALL, buff=0.32,
        )
        code = kit.mono("beta = torch.sigmoid(b_raw)   # (batch, seq, 48), in (0, 1)", size=th.SIZE_LABEL,
                        color=th.INK_2)
        layout = VGroup(top, d, code).arrange(DOWN, buff=0.5)
        self.place(layout)
        with self.narrate(
            "The delta rule fixes the overwrite. "
            "Before writing, it reads what the state already returns for this key. "
            "Then it writes back only the difference, scaled by a step size, beta. "
            "If the state already returns the right value, nothing is written. "
            "Take the squared error of the read as a loss. "
            "Its gradient is the key times the residual, transposed. "
            "One gradient step of size beta is exactly the delta rule. "
            "In the engine, a sigmoid keeps beta between 0 and 1."
        ) as t:
            t.wait_until("Before writing")
            self.play(FadeIn(f[0]), FadeIn(f[3]), FadeIn(brace), FadeIn(b_lab), run_time=0.9)
            t.wait_until("Then it writes back")
            self.play(FadeIn(f[1]), FadeIn(f[2]), FadeIn(f[4]), Create(box), run_time=0.9)
            t.wait_until("Take the squared")
            self.play(d.show(0), run_time=0.8)
            t.wait_until("Its gradient")
            self.play(d.show(1), run_time=0.8)
            t.wait_until("One gradient step")
            self.play(d.show(2), run_time=0.8)
            t.wait_until("In the engine")
            self.play(FadeIn(code), run_time=0.6)

    def erase_then_add(self):
        self.slide("erase-then-add", "Erase along the key, then add",
                   book_section="s-the-delta-rule-write-only-what-s-missing")
        f = kit.formula(r"S_t =", r"(I - \beta\, k k^\top)\, S_{t-1}", r"+", r"\beta\, k v^\top", size=th.SIZE_MATH)
        b1 = Brace(f[1], DOWN, buff=0.12, color=th.INK_3)
        t1 = kit.label("erase along k", size=th.SIZE_LABEL, color=th.INK_3).next_to(b1, DOWN, buff=0.08)
        b2 = Brace(f[3], DOWN, buff=0.12, color=th.INK_3)
        t2 = kit.label("add", size=th.SIZE_LABEL, color=th.INK_3).next_to(b2, DOWN, buff=0.08)
        top = VGroup(f, b1, t1, b2, t2)
        cell = (1.2, 0.75)
        g0 = matrix([["1", "2"], ["0", "0"]], cell=cell)
        g1 = matrix([["0", "0"], ["0", "0"]], cell=cell)
        g2 = matrix([["3", "−1"], ["0", "0"]], cell=cell)
        g0.paint(g0.row(0), self.colors[1])
        g2.paint(g2.row(0), self.colors[2])
        caps = [kit.label(s, size=th.SIZE_LABEL, color=th.INK_3) for s in
                ("after token 3", "erase, β = 1", "add the new value")]
        cols = VGroup()
        for g, cap in zip((g0, g1, g2), caps):
            cap.next_to(g, DOWN, buff=0.18)
            cols.add(VGroup(g, cap))
        cols.arrange(RIGHT, buff=1.3)
        tags = row_tags(g0, ["k", r"k_\perp"])
        arrows = VGroup(*[Arrow(a[0].get_right(), b[0].get_left(), buff=0.15, color=th.INK_3, stroke_width=3,
                                tip_length=0.18) for a, b in zip(cols[:-1], cols[1:])])
        fig = VGroup(cols, tags, arrows)
        res = kit.formula(r"S^\top k = (3,\ -1) = v_b", size=th.SIZE_MATH_SMALL, color=th.GOOD)
        layout = VGroup(top, fig, res).arrange(DOWN, buff=0.55)
        self.place(layout)
        with self.narrate(
            "Collect the terms in the old state, and the write becomes two moves. "
            "First, multiply by the identity minus beta k k transpose, which erases along the key. "
            "Then add beta times the key, times the new value transposed. "
            "Replay the overwrite with beta equal to 1. "
            "The first write put 1, 2 in the key's row. "
            "The second erases that row to zero, then adds 3, minus 1. "
            "Reading now returns 3, minus 1: a replacement, not a sum."
        ) as t:
            self.play(FadeIn(f), run_time=0.8)
            t.wait_until("First, multiply")
            self.play(FadeIn(b1), FadeIn(t1), Indicate(f[1], color=self.color, scale_factor=1.05), run_time=0.9)
            t.wait_until("Then add beta")
            self.play(FadeIn(b2), FadeIn(t2), Indicate(f[3], color=self.color, scale_factor=1.08), run_time=0.9)
            t.wait_until("The first write")
            self.play(FadeIn(cols[0]), FadeIn(tags), run_time=0.6)
            t.wait_until("The second erases")
            self.play(GrowArrow(arrows[0]), FadeIn(cols[1]), run_time=0.7)
            self.play(GrowArrow(arrows[1]), FadeIn(cols[2]), run_time=0.7)
            t.wait_until("Reading now returns")
            self.play(Create(g2.outline(0, 1, 0, 2, color=th.GOOD)), FadeIn(res), run_time=0.7)

    def unit_keys(self):
        self.slide("unit-keys", "Unit-length keys keep the erase stable",
                   book_section="s-the-delta-rule-write-only-what-s-missing")
        eig = kit.formula(r"\mathrm{eig}(I - \beta\, k k^\top):\quad", r"1 - \beta \text{ along } k,",
                          r"\quad 1 \text{ on the other } d_k - 1", size=th.SIZE_MATH_SMALL)
        nl = NumberLine(x_range=[-3, 1.5, 0.5], length=10.5, color=th.INK_3, stroke_width=th.STROKE_AXIS,
                        include_tip=False, tick_size=0.06)
        ticks = VGroup()
        for v in (-3, -2, -1, 0, 1):
            m = kit.math_label(str(v), size=th.SIZE_LABEL, color=th.INK_3)
            m.next_to(nl.n2p(v), DOWN, buff=0.18)
            ticks.add(m)
        band = Rectangle(width=nl.n2p(1)[0] - nl.n2p(-1)[0], height=0.3, stroke_width=0,
                         fill_color=th.tint(th.GOOD, 0.18), fill_opacity=1)
        band.move_to((nl.n2p(1) + nl.n2p(-1)) / 2)
        band.set_z_index(-1)
        band_lab = kit.label("stable: |λ| ≤ 1", size=th.SIZE_LABEL, color=th.GOOD)
        band_lab.next_to(band, UP, buff=0.12).align_to(band, LEFT).shift(RIGHT * 0.1)

        def mark(v, text, color, up=True):
            d = Dot(nl.n2p(v), radius=0.1, color=color)
            lab = kit.label(text, size=th.SIZE_LABEL, color=color)
            lab.next_to(d, UP if up else DOWN, buff=1.0 if up else 0.65)
            ln = Line(d.get_center(), lab.get_bottom() if up else lab.get_top(), color=color,
                      stroke_width=th.STROKE_HAIRLINE, buff=0.12)
            return VGroup(d, ln, lab)

        m_half = mark(0.5, "unit key, β = 0.5", th.GOOD, up=False)
        m_full = mark(0, "β = 1: wiped", th.GOOD, up=True)
        m_bad = mark(-2.2, "norm 2, β = 0.8: −2.2", th.BAD, up=False)
        fig = VGroup(band, nl, ticks, band_lab, m_half, m_full, m_bad)
        bad_f = kit.formula(r"1 - \beta \lVert k \rVert^2 = 1 - 0.8 \times 4 = -2.2", size=th.SIZE_MATH_SMALL,
                            color=th.BAD)
        code = kit.mono("k = F.normalize(k, dim=-1)   # (batch, seq, 16, 128)", size=th.SIZE_LABEL)
        layout = VGroup(eig, fig, bad_f, code).arrange(DOWN, buff=0.45)
        self.place(layout)
        with self.narrate(
            "That erase factor behaves only if the key has unit length. "
            "With a unit key, it shrinks the key's direction by 1 minus beta, and leaves the other 127 directions "
            "alone. "
            "At beta equal to 1, the key's direction is wiped. "
            "So no eigenvalue is larger than 1 in magnitude, and repeated writes never amplify the state. "
            "Now drop the normalization. "
            "A key of norm 2, at beta 0.8, gives an eigenvalue of minus 2.2. "
            "Each write flips its sign and grows it 2.2 times, until the state overflows to NaN. "
            "So the engine normalizes every key to unit length."
        ) as t:
            t.wait_until("With a unit key")
            self.play(FadeIn(eig), run_time=0.8)
            self.play(Create(nl), FadeIn(ticks), run_time=0.7)
            self.play(FadeIn(m_half), run_time=0.5)
            t.wait_until("At beta equal to 1")
            self.play(FadeIn(m_full), run_time=0.5)
            t.wait_until("So no eigenvalue")
            self.play(FadeIn(band), FadeIn(band_lab), run_time=0.7)
            t.wait_until("A key of norm 2")
            self.play(FadeIn(bad_f), FadeIn(m_bad), run_time=0.8)
            t.wait_until("Each write flips")
            self.play(Indicate(m_bad[0], color=th.BAD, scale_factor=1.8), run_time=1.0)
            t.wait_until("So the engine")
            self.play(FadeIn(code), run_time=0.6)

    # ------------------------------------------------------------------
    # The forget gate

    def forget_gate(self):
        self.slide("forget-gate", "A forget gate fades the whole state first", book_section="s-the-forget-gate")
        f = kit.formula(r"S_t =", r"\alpha_t", r"S_{t-1} +", r"\beta_t", r"\, k_t\, \big(v_t -", r"\alpha_t",
                        r"S_{t-1}^\top k_t", r"\big)^\top", size=th.SIZE_MATH)
        for i in (1, 5):
            f[i].set_color(A_C)
        f[3].set_color(B_C)
        f[6].set_color(R_C)
        box = SurroundingRectangle(f, color=self.color, buff=0.2, corner_radius=0.08, stroke_width=3)
        inner = kit.label("corrects the decayed state, not the old one", size=th.SIZE_LABEL, color=A_C)
        inner.next_to(VGroup(f[5], f[6]), DOWN, buff=0.4)
        top = VGroup(VGroup(f, box), inner)

        n = 3
        rng = kit.rng(11)
        base = rng.uniform(0.35, 0.7, size=(n, n))

        def state(levels, title, color=self.color):
            g = kit.grid(n, n, cell=0.5)
            for r in range(n):
                for c in range(n):
                    if levels[r][c] is None:
                        g.cell(r, c).square.set_fill(th.tint(self.colors[2], 0.4), opacity=1)
                    else:
                        g.cell(r, c).square.set_fill(th.tint(color, levels[r][c]), opacity=1)
            cap = kit.prose(title, size=th.SIZE_SMALL, color=th.INK_2, width=None)
            cap.next_to(g, DOWN, buff=0.2)
            return VGroup(g, cap)

        s0 = state(base.tolist(), r"$S_{t-1}$")
        s1 = state((0.5 * base).tolist(), r"decay: $\times\,\alpha_t$")
        er = (0.5 * base).tolist()
        er[0] = [0.0, 0.0, 0.0]
        s2 = state(er, r"erase along $k_t$")
        ad = [list(r) for r in er]
        ad[0] = [None, None, None]
        s3 = state(ad, r"add $\beta_t k_t v_t^\top$")
        stages = VGroup(s0, s1, s2, s3).arrange(RIGHT, buff=1.0, aligned_edge=UP)
        arrows = VGroup(*[Arrow(a[0].get_right(), b[0].get_left(), buff=0.12, color=th.INK_3, stroke_width=3,
                                tip_length=0.18) for a, b in zip(stages[:-1], stages[1:])])
        foot = kit.label("One gate per token per value head: 48 rates in each layer.", size=th.SIZE_SMALL,
                         color=th.INK_2)
        layout = VGroup(top, VGroup(stages, arrows), foot).arrange(DOWN, buff=0.55)
        self.place(layout)
        with self.narrate(
            "The delta rule only touches a key when it's written again, so everything else stays forever. "
            "The last piece is a forget gate, alpha, a scalar between 0 and 1. "
            "It shrinks the whole state before the write. "
            "Alpha appears inside the correction too, so the delta rule corrects the decayed state, not the old one. "
            "So each token makes three moves: decay, erase along the key, and add. "
            "This is the gated delta rule, the whole update this layer runs. "
            "Each value head gets its own gate, so a layer's 48 heads fade at 48 different rates."
        ) as t:
            self.play(FadeIn(s0), run_time=0.6)
            t.wait_until("The last piece")
            self.play(FadeIn(f[1]), FadeIn(f[2]), FadeIn(f[0]), run_time=0.8)
            t.wait_until("It shrinks")
            self.play(GrowArrow(arrows[0]), FadeIn(s1), run_time=0.8)
            t.wait_until("Alpha appears")
            self.play(FadeIn(f[3:]), Create(box), run_time=0.9)
            self.play(FadeIn(inner), Circumscribe(VGroup(f[5], f[6]), color=A_C, buff=0.06), run_time=1.2)
            t.wait_until("So each token")
            self.play(GrowArrow(arrows[1]), FadeIn(s2), run_time=0.7)
            self.play(GrowArrow(arrows[2]), FadeIn(s3), run_time=0.7)
            t.wait_until("Each value head")
            self.play(FadeIn(foot), run_time=0.6)

    def step_board(self, values, title_tex):
        g = matrix(values, cell=(1.35, 0.8))
        tags = row_tags(g, ["e_1", "e_2"])
        name = name_above(g, title_tex)
        return g, VGroup(name, g, tags)

    def step_read(self):
        self.slide("step-read", "One step by hand: decay, then read", book_section="s-the-forget-gate")
        given = line(r"$\alpha = 0.5$, \quad $\beta = 0.5$, \quad $k = (0.6,\ 0.8)$, \quad $v = (2,\ 1)$",
                     size=th.SIZE_BODY, color=th.INK)
        g, board = self.step_board([["1", "2"], ["3", "−1"]], r"S_{t-1}")
        name_new = name_above(g, r"\alpha S_{t-1}", color=A_C)
        work = VGroup(
            kit.formula(r"\alpha S^\top k = 0.6\,(0.5,\ 1) + 0.8\,(1.5,\ -0.5)", size=th.SIZE_MATH_SMALL),
            kit.formula(r"= (1.5,\ 0.2)", size=th.SIZE_MATH_SMALL, color=R_C),
            kit.formula(r"v - \alpha S^\top k = (0.5,\ 0.8)", size=th.SIZE_MATH_SMALL),
            kit.formula(r"\delta = \beta\,(0.5,\ 0.8) = (0.25,\ 0.4)", size=th.SIZE_MATH_SMALL, color=B_C),
        ).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        work[1].shift(RIGHT * 0.6)
        body = VGroup(board, work).arrange(RIGHT, buff=1.9)
        layout = VGroup(given, body).arrange(DOWN, buff=0.6)
        self.place(layout)
        name_new.move_to(board[0])
        w1 = kit.label("× 0.6", size=th.SIZE_SMALL, color=R_C).next_to(g.cell(0, 1), RIGHT, buff=0.2)
        w2 = kit.label("× 0.8", size=th.SIZE_SMALL, color=R_C).next_to(g.cell(1, 1), RIGHT, buff=0.2)
        with self.narrate(
            "Run one step by hand, from the two-drawer state. "
            "Take alpha and beta both 0.5, a unit key of 0.6, 0.8, and a new value of 2, 1. "
            "First, decay: every entry halves. "
            "Next, read the decayed state with the key. "
            "That's 0.6 times row 1, plus 0.8 times row 2, which gives 1.5, 0.2. "
            "The target is 2, 1, so the state is off by 0.5, 0.8. "
            "Scale that by beta, and the correction to write is 0.25, 0.4."
        ) as t:
            self.play(FadeIn(board), run_time=0.7)
            t.wait_until("Take alpha")
            self.play(FadeIn(given), run_time=0.7)
            t.wait_until("First, decay")
            self.play(g.fill(g.block(0, 2, 0, 2), A_C, tint=0.2), Transform(board[0], name_new),
                      *retext(g, [["0.5", "1"], ["1.5", "−0.5"]], color=th.INK), run_time=1.0)
            t.wait_until("Next, read")
            self.play(FadeIn(w1), FadeIn(w2), run_time=0.6)
            t.wait_until("That's 0.6 times")
            self.play(Indicate(g.row(0), color=R_C, scale_factor=1.04), FadeIn(work[0]), run_time=0.8)
            self.play(Indicate(g.row(1), color=R_C, scale_factor=1.04), run_time=0.6)
            self.play(FadeIn(work[1]), run_time=0.6)
            t.wait_until("The target is")
            self.play(FadeIn(work[2]), run_time=0.7)
            t.wait_until("Scale that")
            self.play(FadeIn(work[3]), run_time=0.7)

    def step_write(self):
        self.slide("step-write", "The write moves the read halfway to v", book_section="s-the-forget-gate")
        cell = (1.2, 0.75)
        a = matrix([["0.5", "1"], ["1.5", "−0.5"]], cell=cell)
        a.paint(a.block(0, 2, 0, 2), A_C, tint=0.2)
        a_name = name_above(a, r"\alpha S_{t-1}", color=A_C, size=th.SIZE_SMALL)
        o = matrix([["0.15", "0.24"], ["0.2", "0.32"]], cell=cell)
        o.paint(o.block(0, 2, 0, 2), B_C, tint=0.2)
        o_name = name_above(o, r"k\,\delta^\top", color=B_C, size=th.SIZE_SMALL)
        s = matrix([["0.65", "1.24"], ["1.7", "−0.18"]], cell=cell)
        s.paint(s.block(0, 2, 0, 2), self.color, tint=0.25)
        s_name = name_above(s, r"S_t", size=th.SIZE_SMALL)
        kcol = VGroup(*[num(x, color=th.INK_3) for x in ("0.6", "0.8")])
        for i, x in enumerate(kcol):
            x.next_to(o.cell(i, 0), LEFT, buff=0.18)
        drow = VGroup(*[num(x, color=th.INK_3) for x in ("0.25", "0.4")])
        for i, x in enumerate(drow):
            x.next_to(o.cell(0, i), UP, buff=0.12)
        o_name.next_to(drow, UP, buff=0.12)
        plus = kit.formula("+", size=th.SIZE_MATH)
        eq = kit.formula("=", size=th.SIZE_MATH)
        A = VGroup(a_name, a)
        O = VGroup(o_name, o, kcol, drow)
        Sg = VGroup(s_name, s)
        row = VGroup(A, plus, O, eq, Sg).arrange(RIGHT, buff=0.5, aligned_edge=DOWN)
        plus.set_y(a.get_y())
        eq.set_y(a.get_y())
        checks = VGroup(
            kit.formula(r"S_t^\top k = (1.75,\ 0.6)", r"\quad\text{halfway from } (1.5,\ 0.2) \text{ to } (2,\ 1)",
                        size=th.SIZE_MATH_SMALL),
            kit.formula(r"k_\perp = (-0.8,\ 0.6):", r"\quad (0.5,\ -1.1) \text{ before and after}",
                        size=th.SIZE_MATH_SMALL),
        ).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        checks[0][0].set_color(self.color)
        layout = VGroup(row, checks).arrange(DOWN, buff=0.8)
        self.place(layout)
        s_vals = [[s.cell(r, c).text for c in range(2)] for r in range(2)]
        with self.narrate(
            "Now write. "
            "The outer product of the key and the correction is a 2 by 2 matrix. "
            "Its rows are 0.6 and 0.8 times the correction. "
            "Add it to the decayed state to get the new state. "
            "Read that with the key, and you get 1.75, 0.6. "
            "That's exactly halfway to the target, because beta is 0.5. "
            "Along the perpendicular direction, the read is 0.5, minus 1.1, before the write and after. "
            "The write disturbs nothing it wasn't aimed at."
        ) as t:
            self.play(FadeIn(A), run_time=0.6)
            t.wait_until("The outer product")
            self.play(FadeIn(plus), FadeIn(o.squares()), FadeIn(kcol), FadeIn(drow), FadeIn(o_name), run_time=0.8)
            t.wait_until("Its rows are")
            self.play(LaggedStart(*[FadeIn(o.cell(r, c).text, scale=0.8) for r in range(2) for c in range(2)],
                                  lag_ratio=0.3), run_time=1.2)
            t.wait_until("Add it to")
            self.play(FadeIn(eq), FadeIn(s.squares()), FadeIn(s_name), run_time=0.6)
            for r in range(2):
                ghosts = VGroup(*[a.cell(r, c).text.copy() for c in range(2)],
                                *[o.cell(r, c).text.copy() for c in range(2)])
                self.play(ghosts.animate.move_to(s.row(r)).set_opacity(0), FadeIn(VGroup(*s_vals[r])), run_time=0.8)
                self.remove(ghosts)
            t.wait_until("Read that with")
            self.play(FadeIn(checks[0][0]), run_time=0.6)
            t.wait_until("That's exactly halfway")
            self.play(FadeIn(checks[0][1]), run_time=0.6)
            t.wait_until("Along the perpendicular")
            self.play(FadeIn(checks[1]), run_time=0.7)

    def time_constant(self):
        self.slide("time-constant", "One number per head sets its memory in tokens",
                   book_section="s-the-forget-gate")
        forms = VGroup(
            kit.formula(r"r_t = \operatorname{softplus}(a_t + b_\Delta)\, e^{A_{\log}} > 0", size=th.SIZE_MATH_SMALL),
            kit.formula(r"\alpha_t = e^{-r_t} \in (0,\ 1)", size=th.SIZE_MATH_SMALL),
            kit.formula(r"\alpha^{n} = e^{-rn}", size=th.SIZE_MATH_SMALL),
            kit.formula(r"n_{90} = \frac{\ln 10}{r}", size=th.SIZE_MATH_SMALL),
        ).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        forms[1][0][0].set_color(A_C)
        tab = kit.table(
            [
                ["$0$", "0.368", "0.69", "2.3"],
                ["$-2$", "0.8734", "5.1", "17"],
                ["$-4$", "0.9819", "38", "126"],
                ["$-6$", "0.99752", "280", "929"],
                ["$-8$", "0.99966", "2066", "6864"],
            ],
            header=[r"$A_{\log}$", r"$\alpha$", r"$n_{1/2}$", r"$n_{90}$"],
            align=["r", "r", "r", "r"],
            size=th.SIZE_SMALL, col_gap=0.75, row_gap=0.24,
        )
        cap = kit.label("assuming softplus returns 1", size=th.SIZE_LABEL, color=th.INK_3)
        cap.next_to(tab, DOWN, buff=0.2).align_to(tab, RIGHT)
        right = VGroup(tab, cap)
        top = VGroup(forms, right).arrange(RIGHT, buff=0.9)
        pill = kit.pill("Two units of A_log: the horizon moves by e² ≈ 7.4", self.color, size=th.SIZE_SMALL)
        layout = VGroup(top, pill).arrange(DOWN, buff=0.5)
        self.place(layout)
        with self.narrate(
            "A gate outside 0 to 1 breaks the layer, so the engine keeps it inside by construction. "
            "So the rate r is positive, and alpha, e to the minus r, lands strictly between 0 and 1. "
            "After n steps, a write has shrunk by alpha to the n. "
            "A tenth survives after the log of 10, over r, steps. "
            "Suppose softplus returns 1. "
            "At A log of minus 2, the 90% horizon is 17 tokens. "
            "At minus 4, it's 126. "
            "At minus 8, it's 6864. "
            "Every 2 units of A log move the horizon by a factor of 7.4."
        ) as t:
            self.play(FadeIn(forms[0]), run_time=0.7)
            t.wait_until("So the rate")
            self.play(FadeIn(forms[1]), run_time=0.7)
            t.wait_until("After n steps")
            self.play(FadeIn(forms[2]), run_time=0.6)
            t.wait_until("A tenth survives")
            self.play(FadeIn(forms[3]), run_time=0.6)
            t.wait_until("Suppose softplus")
            self.play(FadeIn(tab.rules), FadeIn(VGroup(*tab.cells[0])), FadeIn(cap), FadeIn(VGroup(*tab.cells[1])),
                      run_time=0.7)
            for r, cue in zip((2, 3, 5), ["At A log of minus 2", "At minus 4", "At minus 8"]):
                t.wait_until(cue)
                rows = [r] if r != 5 else [4, 5]
                self.play(*[FadeIn(VGroup(*tab.cells[i]), shift=RIGHT * 0.1) for i in rows], run_time=0.6)
            t.wait_until("Every 2 units")
            self.play(FadeIn(pill, scale=0.9), run_time=0.6)

    # ------------------------------------------------------------------
    # The sequential form

    def sequential(self):
        self.slide("sequential", "The sequential form is also the decode path",
                   book_section="s-the-sequential-form")
        panel = kit.code(STEP_SRC, "python", caption="engine/layers/linear_attn.py")
        self.place(panel)
        with self.narrate(
            "Here's the recurrence as code, one token at a time. "
            "It's the reference, and also the decode path. "
            "Everything runs in float32, because bf16 would lose the small corrections. "
            "First, the read: the state times the key. "
            "Then the correction, against the decayed read. "
            "Then the state decays, and the outer product is added. "
            "The output uses the state after this token's own write. "
            "This code is correct, and unusable for prefill. "
            "A 2000-token prompt becomes 2000 sequential steps of tiny kernels, and the GPU waits on launch "
            "latency."
        ) as t:
            self.play(FadeIn(panel.frame), *panel.reveal(1, 2), run_time=0.7)
            t.wait_until("Everything runs")
            self.play(*panel.reveal(3, 4, 5), panel.highlight(5), run_time=0.7)
            t.wait_until("First, the read")
            self.play(*panel.reveal(6), panel.highlight(6, color=th.tint(R_C, 0.18)), run_time=0.6)
            t.wait_until("Then the correction")
            self.play(*panel.reveal(7), run_time=0.6)
            t.wait_until("Then the state decays")
            self.play(*panel.reveal(8, 9), panel.highlight(8, 9, color=th.tint(A_C, 0.16)), run_time=0.7)
            t.wait_until("The output uses")
            self.play(*panel.reveal(10, 11), panel.highlight(10), run_time=0.7)

    # ------------------------------------------------------------------
    # The chunked form

    def ledger(self):
        self.slide("ledger", "Unroll a chunk into a ledger", book_section="s-the-chunked-form")
        forms = VGroup(
            kit.formula(r"g_t = \prod_{i=1}^{t} \alpha_i", size=th.SIZE_MATH_SMALL),
            kit.formula(r"u_t = \beta_t\,(v_t - \alpha_t S_{t-1}^\top k_t)", size=th.SIZE_MATH_SMALL),
        ).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        forms[0][0][0:2].set_color(A_C)
        led = kit.formula(r"S_t =", r"g_t S_0", r"+ \sum_{j \le t}", r"\frac{g_t}{g_j}", r"\, k_j u_j^\top",
                          size=th.SIZE_MATH)
        led[1].set_color(self.colors[1])
        led[3].set_color(A_C)
        lbox = SurroundingRectangle(led, color=self.color, buff=0.18, corner_radius=0.08, stroke_width=3)
        b1 = Brace(led[1], DOWN, buff=0.1, color=th.INK_3)
        b2 = Brace(led[2:], DOWN, buff=0.1, color=th.INK_3)
        for b in (b1, b2):
            b.shift(DOWN * (b.get_top()[1] - lbox.get_bottom()[1] + 0.1))
        t1 = kit.label("opening balance", size=th.SIZE_LABEL, color=th.INK_3).next_to(b1, DOWN, buff=0.08)
        t2 = kit.label("deposits, each decayed", size=th.SIZE_LABEL, color=th.INK_3).next_to(b2, DOWN, buff=0.45)
        left = VGroup(forms, VGroup(VGroup(led, lbox), b1, t1, b2, t2)).arrange(DOWN, buff=0.6, aligned_edge=LEFT)
        n = 6
        vals = [[(f"{0.8 ** (r - c):.2f}".rstrip("0").rstrip(".") if c <= r else "") for c in range(n)]
                for r in range(n)]
        g = kit.grid(n, n, cell=(0.8, 0.56), values=vals, size=th.SIZE_LABEL)
        cap = kit.prose(r"$g_t / g_j$ (row $t$, column $j$), every gate $0.8$", size=th.SIZE_SMALL, color=th.INK_2,
                        width=None)
        cap.next_to(g, UP, buff=0.2)
        texts = VGroup(*[g.cell(r, c).text for r in range(n) for c in range(n) if c <= r])
        fig = VGroup(cap, g)
        layout = VGroup(left, fig).arrange(RIGHT, buff=0.8)
        self.place(layout)
        for r in range(n):
            for c in range(n):
                if c <= r:
                    g.cell(r, c).square.set_fill(th.tint(A_C, 0.06 + 0.4 * 0.8 ** (r - c)), opacity=1)
        texts.set_opacity(0)
        with self.narrate(
            "To process a prompt fast, split it into chunks of 64 tokens, and solve each chunk at once. "
            "The cumulative decay, g, is the product of the gates since the chunk began. "
            "The correction, u, is what step t writes. "
            "Unroll the recurrence: the state at step t is the carried-in state, decayed by g, plus every correction so far. "
            "Each correction decays by the gates since it was written. "
            "Read it as a ledger: an opening balance and deposits, all shrinking. "
            "Here's a six-token chunk with every gate at 0.8. "
            "The decay since a deposit is 1 on the diagonal, then 0.8, 0.64, and so on below it."
        ) as t:
            t.wait_until("The cumulative decay")
            self.play(FadeIn(forms[0]), run_time=0.7)
            t.wait_until("The correction, u")
            self.play(FadeIn(forms[1]), run_time=0.7)
            t.wait_until("Unroll the recurrence")
            self.play(FadeIn(led[:2]), FadeIn(b1), FadeIn(t1), run_time=0.8)
            t.wait_until("Each correction decays")
            self.play(FadeIn(led[2:]), FadeIn(b2), FadeIn(t2), Create(lbox), run_time=0.9)
            t.wait_until("Here's a six-token")
            self.play(FadeIn(fig), run_time=0.7)
            t.wait_until("The decay since")
            for k in range(n):
                diag = [g.cell(r, r - k).text for r in range(k, n)]
                self.play(*[m.animate.set_opacity(1) for m in diag], run_time=0.35)

    def triangular_solve(self):
        self.slide("triangular-solve", "Each chunk is one triangular solve that can't fail",
                   book_section="s-the-chunked-form")
        u = kit.formula(r"u_t = \beta_t\,(v_t - g_t S_0^\top k_t)", r"\;-\; \sum_{j<t} \beta_t \frac{g_t}{g_j}"
                        r"(k_j^\top k_t)\, u_j", size=th.SIZE_MATH_SMALL)
        u[1].set_color(self.color)
        sys_ = kit.formula(r"(I + A)\,U = R", size=th.SIZE_MATH)
        sbox = SurroundingRectangle(sys_, color=self.color, buff=0.16, corner_radius=0.08, stroke_width=3)
        n = 6
        vals = [["1" if r == c else "" for c in range(n)] for r in range(n)]
        m = kit.grid(n, n, cell=0.5, values=vals, size=th.SIZE_LABEL)
        for r in range(n):
            for c in range(r):
                m.cell(r, c).square.set_fill(th.tint(self.color, 0.35), opacity=1)
        m_lab = kit.math_label("I + A", size=th.SIZE_SMALL, color=th.INK).next_to(m, UP, buff=0.15)
        U = kit.grid(n, 1, cell=(0.9, 0.5))
        U_lab = kit.math_label("U", size=th.SIZE_SMALL, color=th.INK).next_to(U, UP, buff=0.15)
        R = kit.grid(n, 1, cell=(0.9, 0.5))
        R.paint(R.block(0, n, 0, 1), self.colors[1], tint=0.25)
        R_lab = kit.math_label("R", size=th.SIZE_SMALL, color=th.INK).next_to(R, UP, buff=0.15)
        eq = kit.formula("=", size=th.SIZE_MATH)
        mat = VGroup(VGroup(m_lab, m), VGroup(U_lab, U), eq, VGroup(R_lab, R)).arrange(RIGHT, buff=0.35)
        key = kit.callout("key", r"Unit diagonal, so $\det(I + A) = 1^C = 1$ for every input. Forward "
                                 r"substitution: a chain of 64, not $L$.",
                          title="The solve can never fail", width=5.6, size=th.SIZE_SMALL)
        body = VGroup(mat, key).arrange(RIGHT, buff=0.7)
        layout = VGroup(u, VGroup(sys_, sbox), body).arrange(DOWN, buff=0.5)
        self.place(layout)
        with self.narrate(
            "Substitute the unrolled state into each correction, and only the carried-in state is left. "
            "Each correction is a known vector, minus earlier corrections, weighted by key overlap and decay. "
            "Collect those weights into a matrix A, and the system is I plus A, times U, equals R. "
            "A is strictly lower triangular, because a correction depends only on earlier ones. "
            "So I plus A has ones on its diagonal, and its determinant is exactly 1. "
            "No keys, gates, or step sizes can make this solve fail. "
            "Forward substitution solves it from the top row down. "
            "The chain is 64 steps long, not the length of the sequence."
        ) as t:
            self.play(FadeIn(u[0]), run_time=0.8)
            t.wait_until("Each correction is")
            self.play(FadeIn(u[1]), run_time=0.8)
            t.wait_until("Collect those")
            self.play(FadeIn(sys_), Create(sbox), FadeIn(m.squares()), FadeIn(m_lab), FadeIn(U), FadeIn(U_lab),
                      FadeIn(eq), FadeIn(R), FadeIn(R_lab), run_time=0.9)
            t.wait_until("A is strictly")
            self.play(Indicate(VGroup(*[m.cell(r, c) for r in range(n) for c in range(r)]), color=self.color,
                               scale_factor=1.04), run_time=1.0)
            t.wait_until("So I plus A")
            self.play(LaggedStart(*[FadeIn(m.cell(r, r).text, scale=0.6) for r in range(n)], lag_ratio=0.2),
                      run_time=1.0)
            t.wait_until("No keys")
            self.play(FadeIn(key, shift=UP * 0.1), run_time=0.7)
            t.wait_until("Forward substitution")
            for r in range(n):
                hl = m.outline(r, r + 1, 0, r + 1, color=self.colors[1], stroke=3)
                self.play(Create(hl), U.fill(U.cell(r, 0), self.color, tint=0.45), run_time=0.35)
                self.play(FadeOut(hl), run_time=0.15)

    def carried_state(self):
        self.slide("carried-state", "Only the state crosses a chunk boundary", book_section="s-the-chunked-form")
        o = kit.formula(r"o_t = g_t S_0^\top q_t + \sum_{j \le t} \frac{g_t}{g_j}\,(q_t^\top k_j)\, u_j",
                        size=th.SIZE_MATH_SMALL)
        sc = kit.formula(r"S_C = g_C S_0 + \sum_{j=1}^{C} \frac{g_C}{g_j}\, k_j u_j^\top", size=th.SIZE_MATH_SMALL)
        forms = VGroup(o, sc).arrange(RIGHT, buff=0.9)
        n = 32
        strip = kit.grid(1, n, cell=(0.3, 0.55), gap=0.04)
        cap = kit.label("2048 tokens in 32 chunks of 64", size=th.SIZE_LABEL, color=th.INK_3)
        cap.next_to(strip, UP, buff=0.75).align_to(strip, LEFT)
        state = kit.box("S", width=0.55, height=0.5, color=self.colors[1], size=th.SIZE_LABEL)
        state.next_to(strip.cell(0, 0), UP, buff=0.12)
        fig = VGroup(cap, strip, state)
        count = kit.pill("32 iterations instead of 2048", self.color, size=th.SIZE_SMALL)
        layout = VGroup(forms, fig, count).arrange(DOWN, buff=0.6)
        self.place(layout)
        o_note = kit.label("one small matmul per chunk", size=th.SIZE_LABEL, color=th.INK_3).next_to(o, DOWN,
                                                                                                    buff=0.15)
        s_note = kit.label("carried to the next chunk", size=th.SIZE_LABEL, color=self.colors[1]).next_to(sc, DOWN,
                                                                                                        buff=0.15)
        with self.narrate(
            "With the corrections solved, the rest is reading the ledger. "
            "Each output is the carried-in state read by the query, plus a masked score matrix times U. "
            "Then the state at the end of the chunk carries forward. "
            "It's the only value that crosses a chunk boundary. "
            "At 2048 tokens and chunks of 64, the loop runs 32 times instead of 2048."
        ) as t:
            self.play(FadeIn(strip), FadeIn(cap), run_time=0.6)
            t.wait_until("Each output")
            self.play(FadeIn(o), run_time=0.8)
            self.play(FadeIn(o_note), run_time=0.5)
            t.wait_until("Then the state")
            self.play(FadeIn(sc), FadeIn(s_note), FadeIn(state), run_time=0.8)
            t.wait_until("It's the only")
            for c in range(n):
                anims = [strip.fill(strip.cell(0, c), self.color, tint=0.45)]
                if c:
                    anims.append(state.animate.next_to(strip.cell(0, c), UP, buff=0.12))
                self.play(*anims, run_time=0.12)
            t.wait_until("At 2048 tokens")
            self.play(FadeIn(count, scale=0.9), run_time=0.6)

    def chunk_cost(self):
        self.slide("chunk-cost", "More arithmetic, in a shape the GPU likes",
                   book_section="s-what-the-chunked-form-costs")
        forms = VGroup(
            kit.formula(r"\text{chunked: } 7Cd + 6d^2", r"\qquad \text{sequential: } 7d^2", size=th.SIZE_MATH_SMALL),
            kit.formula(r"\text{equal at } C = d/7 \approx 18", size=th.SIZE_MATH_SMALL),
        ).arrange(DOWN, buff=0.35)
        bars = kit.bar_compare(
            [("Sequential", 114688, self.colors[1], "114,688 FLOPs per token"),
             ("Chunked, C = 64", 155648, self.color, "155,648 FLOPs per token")],
            width=5.6, bar_height=0.5, gap=0.3,
        )
        ratio = kit.pill("1.36 × the FLOPs", self.color, size=th.SIZE_SMALL)
        mid = VGroup(bars, ratio).arrange(RIGHT, buff=0.5)
        key = kit.callout("key", r"The loop leaves the A100's 108 SMs idle on tiny dependent kernels. The chunks "
                                 r"give the tensor cores dense matmuls: more than $5\times$ faster on 2048 tokens.",
                          title="A shape change, not a FLOP saving", width=11.5, size=th.SIZE_SMALL)
        layout = VGroup(forms, mid, key).arrange(DOWN, buff=0.55)
        self.place(layout)
        with self.narrate(
            "The chunked form doesn't win by doing less work. "
            "Per token, with d of 128, it costs 7 C d, plus 6 d squared. "
            "The sequential loop costs 7 d squared. "
            "They're equal at a chunk size of about 18. "
            "At the engine's chunk size of 64, that's 155,648 FLOPs per token, against 114,688 for the loop. "
            "So chunking does about 1.36 times the arithmetic. "
            "Yet the lab requires it to run more than 5 times faster on 2048 tokens. "
            "The loop's tiny dependent kernels leave the A100's 108 SMs idle, while chunks give the tensor cores "
            "dense matmuls."
        ) as t:
            t.wait_until("Per token")
            self.play(FadeIn(forms[0][0]), run_time=0.7)
            t.wait_until("The sequential loop costs")
            self.play(FadeIn(forms[0][1]), run_time=0.6)
            t.wait_until("They're equal")
            self.play(FadeIn(forms[1]), run_time=0.6)
            t.wait_until("At the engine's")
            self.play(FadeIn(bars.labels), GrowFromEdge(bars.bars[1], LEFT), FadeIn(bars.values[1]), run_time=0.9)
            self.play(GrowFromEdge(bars.bars[0], LEFT), FadeIn(bars.values[0]), run_time=0.8)
            t.wait_until("So chunking")
            self.play(FadeIn(ratio, scale=0.9), run_time=0.5)
            t.wait_until("Yet the lab")
            self.play(FadeIn(key, shift=UP * 0.1), run_time=0.8)

    # ------------------------------------------------------------------
    # Decay ratios and floating point

    def log_space(self):
        self.slide("log-space", "Compute decay ratios in log space",
                   book_section="s-decay-ratios-and-floating-point")
        ax = kit.axes([0, 200, 25], [-60, 0, 10], width=6.0, height=3.6,
                      x_label=r"t", y_label=r"\log_{10} g_t",
                      x_numbers=[0, 50, 100, 150, 200], y_numbers=[-60, -45, -30, -15, 0])
        curve = ax.plot(lambda x: -0.30103 * x, x_range=[0, 149], color=A_C, stroke_width=th.STROKE_CURVE)
        floor = DashedLine(ax.c2p(0, -44.85), ax.c2p(200, -44.85), color=th.BAD, stroke_width=th.STROKE_AXIS)
        f_lab = kit.label("float32 floor, 1.4e-45", size=th.SIZE_LABEL, color=th.BAD)
        f_lab.next_to(ax.c2p(200, -44.85), DOWN, buff=0.12).align_to(ax.c2p(200, 0), RIGHT)
        zero = Dot(ax.c2p(150, -44.85), radius=0.09, color=th.BAD)
        z_lab = kit.label("t = 150: g is 0", size=th.SIZE_LABEL, color=th.BAD)
        z_lab.next_to(zero, UR, buff=0.12)
        a_lab = kit.label("α = 0.5", size=th.SIZE_LABEL, color=A_C).next_to(ax.c2p(60, -18), UR, buff=0.1)
        plot = VGroup(ax, curve, floor, f_lab, zero, z_lab, a_lab)
        nan = kit.formula(r"\frac{g_t}{g_j} = \frac{0}{0} = \text{NaN}", size=th.SIZE_MATH_SMALL, color=th.BAD)
        fix = kit.formula(r"\frac{g_t}{g_j} = \exp(\log g_t - \log g_j) \le 1", size=th.SIZE_MATH_SMALL,
                          color=th.GOOD)
        tiny_w = kit.prose(r"$\alpha = 10^{-8}$: 64 tokens sum to $\log g = -1179$, an ordinary float32.",
                           size=th.SIZE_SMALL, color=th.INK_2, width=4.6)
        side = VGroup(nan, fix, tiny_w).arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        layout = VGroup(plot, side).arrange(RIGHT, buff=0.7)
        self.place(layout)
        with self.narrate(
            "Decay ratios are the numerical trap: compute g as a running product, then divide, and it "
            "underflows. "
            "Take a head with alpha 0.5, so g halves every token. "
            "Float32 bottoms out near 1.4 times 10 to the minus 45. "
            "At token 150, g is exactly zero. "
            "Then a ratio is zero over zero, which is NaN, and it spreads to every later chunk. "
            "The fix is to keep logarithms. "
            "Even with alpha at 10 to the minus 8, a 64-token chunk sums to only minus 1179, an ordinary float32. "
            "Subtract the logs, then exponentiate once. "
            "The result is at most 1, and a tiny true ratio becomes zero, the right answer."
        ) as t:
            self.play(Create(ax), run_time=0.8)
            t.wait_until("Take a head")
            self.play(Create(curve), FadeIn(a_lab), run_time=1.6)
            t.wait_until("Float32 bottoms")
            self.play(Create(floor), FadeIn(f_lab), run_time=0.7)
            t.wait_until("At token 150")
            self.play(FadeIn(zero, scale=0.5), FadeIn(z_lab), run_time=0.6)
            t.wait_until("Then a ratio")
            self.play(FadeIn(nan), run_time=0.7)
            t.wait_until("Even with alpha")
            self.play(FadeIn(tiny_w), run_time=0.7)
            t.wait_until("Subtract the logs")
            self.play(FadeIn(fix), run_time=0.7)
