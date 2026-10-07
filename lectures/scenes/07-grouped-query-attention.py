"""Chapter 07, grouped-query attention.

Teaches content/chapters/07-grouped-query-attention.md in its order. Every
number below was recomputed with Python and numpy; each matches the chapter.

Geometry: d = 5120, H = 24 query heads, H_kv = 4 KV heads, d_h = 256, gate on,
16 of 64 layers full attention, b = 2 bytes (bfloat16).

The lookup (a toy row, not from the chapter): scores s = (2, 0.5, 1, -1),
softmax weights (0.609, 0.136, 0.224, 0.030), shown to two places as
0.61, 0.14, 0.22, 0.03 (sum 1.00); values v = (10, 20, 30, 40); output
sum w v = 16.754, "16.75", inside the values' range [10, 40].

The scale factor:
- Var(q^T k) = d_h = 256, sd = 16. Histogram: 4000 simulated dot products
  of 256-wide standard normal vectors, kit.rng(71): sample sd 16.06.
- Six scores of one row, kit.rng(7) (q and six keys, 256 wide): raw
  (-8.4, -4.0, 18.8, -13.7, 16.3, 26.1), "about minus 14 to 26"; softmax
  (0.000, 0.000, 0.001, 0.000, 0.000, 0.999), largest weight 0.9993,
  "99.9 percent". Divided by 16: (-0.52, -0.25, 1.18, -0.85, 1.02, 1.63);
  softmax (0.046, 0.060, 0.251, 0.033, 0.214, 0.396), shown as 0.05, 0.06,
  0.25, 0.03, 0.21, 0.40 (sum 1.00).
- e^16 = 8,886,110 = 8.9 x 10^6. 1 / sqrt(256) = 1/16 = 0.0625.

The causal mask (toy scores; row 1 is (1, 2, 3, 0.5)):
- Mask, then softmax over keys 0 and 1: (0.269, 0.731), shown 0.27, 0.73,
  sum 1.
- Softmax over all four, then zero keys 2 and 3: (0.085, 0.232), shown
  0.09, 0.23; kept sum 0.317, "0.32".
- Decode: kv_len 10, q_len 1, offset 9; trusting row 0 hides keys 1 to 9.
- Regimes (chapter table): prefill 4 x 4, offset 0, 6 masked (4 x 3 / 2);
  decode 1 x 10, offset 9, 0 masked; chunk 3 x 10, offset 7, queries at 7,
  8, 9 mask 2 + 1 + 0 = 3.

Shapes and parameters:
- q_proj 5120 x 12288 = 62,914,560 (62.9M); k_proj = v_proj = 5120 x 1024
  = 5,242,880 (5.2M) each, 10,485,760 together (10.5M); o_proj 6144 x 5120
  = 31,457,280 (31.5M). Total 104,857,600 (104.9M); x 2 B = 209.7 MB
  ("210 MB"); x 16 layers = 3.36 GB ("3.4 GB").
- 24 x 256 = 6144; 4 x 256 = 1024; 5120 / 24 = 213.33.
- Bars drawn to scale at 2048 channels per unit: 12288 -> 6.0, 6144 -> 3.0,
  5120 -> 2.5, 1024 -> 0.5.

Cache arithmetic, 2 x H_kv x 256 x 2 bytes per layer per token:
- MHA (24): 24,576 B = 24 KiB; x 16 = 384 KiB; x 32,768 = 12 GiB.
- H_kv = 8: 8 KiB; 128 KiB, a third of MHA.
- GQA (4): 4096 B = 4 KiB; 64 KiB; 2 GiB. 64 / 384 = 1/6.
- MQA (1): 1 KiB; 16 KiB; 0.5 GiB. 16 / 384 = 1/24.
- Card: 80 - 53.8 = 26.2 GB, "about 26 GB". MHA: 26.2e9 / (12 x 2^30)
  = 2.03, two sequences. GQA: 11 is chapter 2's batch (23.9 GiB after
  overhead, 2.14 GiB per sequence with state); 26.2 GB / 2 GiB alone would
  be 12.2, so the lecture quotes 11 as chapter 2's figure, as the chapter does.

Decode intensity: FLOPs 4 H L d_h, bytes 2 H_kv L d_h b; I = 2H / (H_kv b)
= 48 / 8 = 6. Ridge 312e12 / 1935e9 = 161.2, "161"; 161.2 / 6 = 26.9,
"27 times"; 6 / 161.2 = 3.7%, "under 4 percent". int8 (b = 1): 12.

Output gate: 5120 x 6144 = 31,457,280 parameters; x 2 B = 62.9 MB; x 16 =
1.007 GB, "about 1.0 GB" of 53.8 GB. sigmoid(-20) = 2.1e-9, under 1e-6.
Sigmoid on the slide: s(4) = 0.982, s(0) = 0.5, s(-8) = 0.0003.

repeat_kv at 32k, one sequence, one layer: 2 x 4 x 32,768 x 256 x 2 =
134,217,728 B = 128 MiB; x 6 = 768 MiB; redundant 640 MiB. 13 // 6 = 2.

Naive path at 8k, 24 heads: 24 x 8192^2 x 2 B = 3 GiB; float32 6 GiB.
Peak "well past 12 GB" and 27.5x are the chapter's measurements.

Decode time (53.8 GB of weights): 53.8e9 / 1935e9 = 27.80 ms, 36.0 tok/s;
53.8e9 / 1275e9 = 42.20 ms, 23.7 tok/s ("24"). At 32k: cache 2.0 GiB plus
state 154,927,104 B = 0.144 GiB; (2 x 2^30 + 154,927,104) / 1275e9 =
1.81 ms, "about 1.8 ms".
"""

from manim import *

import kit
from kit import theme as th
from kit.text import MATH_TEMPLATE

# The chapter's maths colours: \hla score, \hlb weight and H_kv, \hlc output and b, \hld sqrt(d_h).
SCORE_C = th.HL_A
WEIGHT_C = th.HL_B
OUT_C = th.HL_C
SCALE_C = th.HL_D
Q_C = th.VIOLET  # query heads
GATE_C = th.WARN  # the output gate

MASK_SRC = """\
offset = kv_len - q_len
idx_q = torch.arange(q_len, device=device).unsqueeze(-1)
idx_k = torch.arange(kv_len, device=device).unsqueeze(0)
mask = idx_k > idx_q + offset          # True = blocked
scores = scores.masked_fill(mask, float("-inf"))
"""

GATE_SRC = """\
q = self.q_proj(x)                # (batch, seq, 12288)
q, gate = q.chunk(2, dim=-1)      # (batch, seq, 6144) each
# ... attention, producing out: (batch, seq, 6144) ...
out = out * torch.sigmoid(gate)   # elementwise
return self.o_proj(out)           # (batch, seq, 5120)
"""

NAIVE_SRC = """\
scores = torch.matmul(q.float(),
                      k.float().transpose(-1, -2)) * scale
scores = scores.masked_fill(mask, float("-inf"))
weights = torch.softmax(scores, dim=-1)
out = torch.matmul(weights.to(v.dtype), v)
"""


def swatch(text, color, size=th.SIZE_LABEL, tint=0.35):
    sq = Square(0.26, stroke_color=th.RULE_STRONG, stroke_width=th.STROKE_HAIRLINE,
                fill_color=th.tint(color, tint), fill_opacity=1)
    return VGroup(sq, kit.label(text, size=size, color=th.INK_2)).arrange(RIGHT, buff=0.16)


def row(tab, i):
    return VGroup(*tab.cells[i])


def cell_text(g, r, c, mob):
    """Places `mob` in a grid cell, scaled to fit."""
    sq = g.cell(r, c).square
    if mob.width > sq.width * 0.82:
        mob.scale_to_fit_width(sq.width * 0.82)
    if mob.height > sq.height * 0.7:
        mob.scale_to_fit_height(sq.height * 0.7)
    return mob.move_to(sq)


class Lecture(kit.Lecture):
    def construct(self):
        self.title_slide(
            narration=(
                (
                    (
                        'Grouped-query attention. '
                        'You build attention from a lookup, and see why its scores are scaled and masked. '
                        "Then you count what sharing keys and values saves, and why decode attention leaves the GPU's arithmetic mostly idle."
                    )
                )
            )
        )
        self.sixteen_layers()
        self.lookup()
        self.three_properties()
        self.score_spread()
        self.saturation()
        self.mask_before()
        self.mask_offset()
        self.mask_regimes()
        self.shapes()
        self.heads()
        self.sharing()
        self.cache_table()
        self.quality()
        self.intensity()
        self.ridge()
        self.gate()
        self.gate_slice()
        self.repeat_kv()
        self.naive()
        self.decode_time()
        self.closing_slide(
            [
                r"Attention is a weighted average. Divide by $\sqrt{d_h}$ against saturation, not overflow.",
                r"Mask before the softmax, shifted by the cached prefix.",
                r"GQA: 64 KiB per token, and $H / H_{kv} = 6$ FLOPs per byte in decode.",
                r"Chunk the query projection into queries and a gate. Index the shared KV head.",
            ],
            narration=(
                (
                    (
                        "Here's what to keep. "
                        'Attention is a weighted average, and its scale guards against saturation, not overflow. '
                        'Mask before the softmax, shifted by the cached prefix. '
                        'Grouped-query attention costs 64 KiB per token, and decode attention does 6 FLOPs per byte, so only reading fewer bytes helps. '
                        'Chunk the query projection into queries and a gate, and index the shared KV head instead of copying it. '
                        'Next, you assemble the full forward pass.'
                    )
                )
            ),
            book_section="s-what-goes-wrong",
        )

    # ------------------------------------------------------------------
    # Before you start

    def sixteen_layers(self):
        self.slide("sixteen-layers", "Sixteen of the 64 layers keep a KV cache", book_section="s-before-you-start")
        g = kit.grid(1, 64, cell=(0.16, 0.55), gap=0.03)
        full = VGroup(*[g.cell(0, i) for i in range(64) if i % 4 == 3])
        cap = kit.label("64 layers", size=th.SIZE_LABEL, color=th.INK_3).next_to(g, UP, buff=0.12).align_to(g, LEFT)
        key = VGroup(swatch("16 full attention: keys and values, cached", self.color),
                     swatch("48 linear attention (chapter 6)", th.INK_4, tint=0.2)).arrange(RIGHT, buff=0.7)
        tab = kit.table(
            [
                [r"Residual stream width, $d$", "5120"],
                [r"Query heads, $H$", "24"],
                [r"Key and value heads, $H_{kv}$", "4"],
                [r"Head width, $d_h$", "256"],
                ["Output gate", "on every head"],
                ["Cache element", "2 bytes, bfloat16"],
            ],
            header=["Geometry", "Value"],
            align=["l", "r"],
            size=th.SIZE_SMALL,
            row_gap=0.2,
        )
        pre = kit.box("Prefill", sublabel="the whole prompt in one pass", width=4.6, color=Q_C)
        dec = kit.box("Decode", sublabel="one token per pass, reading the cache", width=4.6, color=self.color)
        phases = VGroup(pre, dec).arrange(DOWN, buff=0.35)
        lower = VGroup(tab, phases).arrange(RIGHT, buff=0.8)
        col = VGroup(VGroup(cap, g), key, lower).arrange(DOWN, buff=0.4)
        self.place(col)
        lin = VGroup(*[g.cell(0, i) for i in range(64) if i % 4 != 3])
        with self.narrate(
            (
                (
                    "Sixteen of this model's 64 layers use ordinary softmax attention. "
                    "They own the KV cache: every past token's keys and values, kept so they're never recomputed. "
                    'Prefill pushes the whole prompt through in one pass. '
                    'Decode then generates one token per pass, reading back the cache. '
                    'The residual stream is 5120 wide. '
                    'There are 24 query heads but only 4 key and value heads, each 256 wide. '
                    "Each head's output passes through a gate, and the cache stores 2 bytes per element."
                )
            )
        ) as t:
            self.play(FadeIn(g), FadeIn(cap), run_time=0.6)
            self.play(LaggedStart(*[c.square.animate.set_fill(th.tint(self.color, 0.45)) for c in full],
                                  lag_ratio=0.08), FadeIn(key[0]), run_time=1.4)
            self.play(g.fill(lin, th.INK_4, tint=0.2), FadeIn(key[1]), run_time=0.6)
            self.play(Indicate(full, color=self.color, scale_factor=1.06), run_time=0.9)
            t.wait_until("Prefill pushes")
            self.play(FadeIn(pre, shift=UP * 0.1), run_time=0.6)
            t.wait_until("Decode then")
            self.play(FadeIn(dec, shift=UP * 0.1), run_time=0.6)
            t.wait_until("The residual stream")
            self.play(FadeIn(tab.rules), FadeIn(row(tab, 0)), FadeIn(row(tab, 1)), run_time=0.7)
            t.wait_until("There are 24")
            self.play(LaggedStart(*[FadeIn(row(tab, i)) for i in (2, 3, 4)], lag_ratio=0.4), run_time=1.2)
            t.wait_until("Each head's output")
            self.play(FadeIn(row(tab, 5)), FadeIn(row(tab, 6)), run_time=0.7)

    # ------------------------------------------------------------------
    # Queries, keys, and values

    def lookup(self):
        self.slide("lookup", "Attention is a differentiable lookup", book_section="s-queries-keys-and-values")
        f = kit.formula(
            r"s_{tj} = \frac{q_t^\top k_j}{\sqrt{d_h}},", r"\qquad",
            r"w_{tj} = \frac{e^{s_{tj}}}{\sum_{i \le t} e^{s_{ti}}},", r"\qquad",
            r"o_t = \sum_{j \le t} w_{tj}\, v_j",
            size=th.SIZE_MATH_SMALL,
        )
        f[0].set_color(SCORE_C)
        f[2].set_color(WEIGHT_C)
        f[4].set_color(OUT_C)
        scores = [2.0, 0.5, 1.0, -1.0]
        weights = [0.61, 0.14, 0.22, 0.03]
        g = kit.grid(4, 4, cell=(1.25, 0.62),
                     values=[[f"k{j}" for j in range(4)], ["10", "20", "30", "40"], [""] * 4, [""] * 4],
                     row_labels=["key (label)", "value (contents)", "score", "weight"],
                     col_labels=[f"token {j}" for j in range(4)])
        for j in range(4):
            g.paint(g.cell(0, j), th.INK_4, tint=0.15)
            g.paint(g.cell(1, j), OUT_C, tint=0.15)
        q = kit.box("query  q", sublabel="what token t wants", color=Q_C, width=2.6, height=1.3)
        out = kit.box("output  o", sublabel="weighted average", color=OUT_C, width=2.6, height=1.3)
        o_val = kit.label("16.75", size=th.SIZE_HEADING, color=th.INK, weight=SEMIBOLD)
        row_ = VGroup(q, g, out).arrange(RIGHT, buff=0.6)
        col = VGroup(f, row_).arrange(DOWN, buff=0.6)
        self.place(col)
        o_val.next_to(out, DOWN, buff=0.2)
        s_txt = [cell_text(g, 2, j, kit.label(f"{v:g}", size=th.SIZE_SMALL, color=SCORE_C, font=th.MONO))
                 for j, v in enumerate(scores)]
        w_txt = [cell_text(g, 3, j, kit.label(f"{v:.2f}", size=th.SIZE_SMALL, color=WEIGHT_C, font=th.MONO))
                 for j, v in enumerate(weights)]
        with self.narrate(
            (
                (
                    'Attention is a differentiable lookup in a table of the tokens seen so far. '
                    'Picture a shelf of folders. '
                    'Each past token offers a key, the label on its folder, and a value, its contents. '
                    'The current token arrives with a query, its question. '
                    'So attention scores every key against the query. '
                    'Here, the scores are 2, 0.5, 1, and minus 1. '
                    'A softmax turns them into weights that are positive and sum to 1. '
                    'The output blends the values in those proportions: 16.75.'
                )
            )
        ) as t:
            t.wait_until("Picture a shelf")
            self.play(FadeIn(g), run_time=0.8)
            t.wait_until("Each past token")
            self.play(Indicate(g.row(0), color=th.INK_3, scale_factor=1.04), run_time=0.7)
            self.play(Indicate(g.row(1), color=OUT_C, scale_factor=1.04), run_time=0.6)
            t.wait_until("The current token")
            self.play(FadeIn(q, shift=RIGHT * 0.2), run_time=0.6)
            t.wait_until("So attention scores")
            self.play(FadeIn(f[0]), run_time=0.7)
            t.wait_until("Here, the scores")
            self.play(LaggedStart(*[FadeIn(s, shift=RIGHT * 0.3) for s in s_txt], lag_ratio=0.2), run_time=1.2)
            t.wait_until("A softmax turns")
            self.play(FadeIn(f[1:3]), run_time=0.6)
            self.play(LaggedStart(*[FadeIn(w, scale=0.8) for w in w_txt], lag_ratio=0.15),
                      *[g.fill(g.cell(3, j), WEIGHT_C, tint=0.08 + 0.6 * w) for j, w in enumerate(weights)],
                      run_time=1.0)
            t.wait_until("The output blends")
            self.play(FadeIn(f[3:]), FadeIn(out), run_time=0.6)
            flow = [g.cell(1, j).text.copy() for j in range(4)]
            self.play(LaggedStart(*[c.animate.move_to(out.rect).set_opacity(0) for c in flow], lag_ratio=0.2),
                      run_time=1.2)
            self.remove(*flow)
            self.play(FadeIn(o_val, shift=UP * 0.1), run_time=0.5)

    def three_properties(self):
        self.slide("three-properties", "The output is an average, so the past must be kept",
                   book_section="s-queries-keys-and-values")
        f = kit.formula(r"\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left( \frac{Q K^\top}"
                        r"{\sqrt{d_h}} + M \right) V", size=th.SIZE_MATH_SMALL)
        ax = kit.axes([0, 50, 10], [0, 1, 1], width=7.5, height=0.8, x_numbers=[0, 10, 20, 30, 40, 50],
                      show_y_axis=False)
        hull = Rectangle(width=ax.c2p(40, 0)[0] - ax.c2p(10, 0)[0], height=0.5, stroke_width=0,
                         fill_color=th.tint(OUT_C, 0.25), fill_opacity=1)
        hull.move_to([(ax.c2p(10, 0)[0] + ax.c2p(40, 0)[0]) / 2, ax.c2p(0, 0)[1] + 0.25, 0])
        vals = VGroup(*[Dot(ax.c2p(v, 0) + UP * 0.25, radius=0.09, color=th.INK_3) for v in (10, 20, 30, 40)])
        o_dot = Dot(ax.c2p(16.75, 0) + UP * 0.25, radius=0.12, color=OUT_C)
        vals.set_z_index(2)
        o_dot.set_z_index(3)
        o_lab = kit.label("output 16.75", size=th.SIZE_LABEL, color=OUT_C, weight=MEDIUM).next_to(o_dot, UP,
                                                                                                buff=0.3)
        h_lab = kit.label("range of the values", size=th.SIZE_LABEL, color=th.INK_3).next_to(hull, RIGHT, buff=0.25)
        line = VGroup(ax, hull, vals, o_dot, o_lab, h_lab)
        bul = kit.bullets(
            [
                r"\textbf{A convex combination:} it can't amplify, and it can't output zero.",
                r"\textbf{No fixed-size summary:} every past key and value is kept.",
                r"\textbf{Prefill costs $O(L^2)$, decode $O(L)$}, and decode's bytes are the limit.",
            ],
            size=th.SIZE_SMALL, width=11.5, buff=0.28,
        )
        col = VGroup(f, line, bul).arrange(DOWN, buff=0.55)
        self.place(col)
        with self.narrate(
            (
                (
                    'In matrix form, a whole block of queries runs at once, and M is the causal mask. '
                    'First, the weights are non-negative and sum to 1, so the output is a convex combination. '
                    'It always lands inside the range of the values it read. '
                    "So attention can't amplify, and it can't output zero. "
                    "Second, there's no fixed-size summary of the past, so every key and value has to be kept. "
                    "That's the KV cache. "
                    'Third, prefill is quadratic in the context, and decode is linear, but decode reads every cached byte.'
                )
            )
        ) as t:
            self.play(FadeIn(f), run_time=0.8)
            t.wait_until("First, the weights")
            self.play(FadeIn(ax), FadeIn(vals), run_time=0.7)
            t.wait_until("It always lands")
            self.play(FadeIn(hull), FadeIn(h_lab), run_time=0.6)
            self.play(FadeIn(o_dot, scale=0.5), FadeIn(o_lab), run_time=0.6)
            t.wait_until("So attention can't")
            self.play(FadeIn(bul[0], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("Second, there")
            self.play(FadeIn(bul[1], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("Third, prefill")
            self.play(FadeIn(bul[2], shift=RIGHT * 0.1), run_time=0.6)

    # ------------------------------------------------------------------
    # Where the scale factor comes from

    def score_spread(self):
        self.slide("score-spread", "Raw scores spread by √256 = 16",
                   book_section="s-where-the-scale-factor-comes-from")
        d = kit.Derivation(
            [
                r"q^\top k &= \sum_{i=1}^{d_h} q_i k_i",
                r"\operatorname{Var}(q^\top k) &= \sum_{i=1}^{d_h} \operatorname{Var}(q_i k_i) = d_h",
                r"\operatorname{sd}(q^\top k) &= \sqrt{d_h} = \sqrt{256} = 16",
            ],
            size=th.SIZE_MATH_SMALL, buff=0.45,
        )
        d.lines[2].set_color(SCALE_C)
        assume = VGroup(kit.prose(r"Entries of $q$ and $k$: independent,", size=th.SIZE_SMALL, color=th.INK_3,
                                  width=None),
                        kit.prose(r"mean 0, variance 1.", size=th.SIZE_SMALL, color=th.INK_3, width=None)).arrange(
            DOWN, buff=0.12, aligned_edge=LEFT)
        left = VGroup(assume, d).arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        ax = kit.axes([-64, 64, 16], [0, 0.03, 0.01], width=6.2, height=3.6,
                      x_numbers=[-48, -16, 16, 48], x_label=r"\text{raw score } q^\top k", show_y_axis=False)
        edges = np.arange(-64, 64.01, 4)
        hist = kit.RunningHistogram(ax, edges, color=SCALE_C)
        lo = kit.vline(ax, -16, 0.027, color=th.INK_3)
        hi = kit.vline(ax, 16, 0.027, color=th.INK_3)
        sd_lab = kit.label("±16: one standard deviation", size=th.SIZE_LABEL, color=th.INK_3)
        sd_lab.next_to(ax.c2p(0, 0.027), UP, buff=0.12)
        n_lab = kit.label("4000 simulated scores, 256 channels each", size=th.SIZE_LABEL, color=th.INK_3)
        plot = VGroup(ax, hist.bars, lo, hi, sd_lab)
        right = VGroup(plot, n_lab).arrange(DOWN, buff=0.25)
        layout = VGroup(left, right).arrange(RIGHT, buff=0.8)
        self.place(layout)
        r = kit.rng(71)
        batches = [(r.normal(size=(500, 256)) * r.normal(size=(500, 256))).sum(axis=1) for _ in range(8)]
        with self.narrate(
            (
                'Why divide by the square root of the head width? '
                'Model the entries of q and k as independent, with mean zero and variance 1. '
                'The raw score adds up 256 products, and each product has variance 1. '
                "Independent variances add, so the score's variance is the head width. "
                'Its standard deviation is the square root of that: 16. '
                'The histogram fills in with 4000 simulated scores, and their spread lands on 16.'
            )
        ) as t:
            t.wait_until("Model the entries")
            self.play(FadeIn(assume), run_time=0.6)
            t.wait_until("The raw score adds")
            self.play(d.show(0), run_time=0.8)
            t.wait_until("Independent variances")
            self.play(d.show(1), run_time=0.8)
            t.wait_until("Its standard deviation")
            self.play(d.show(2), run_time=0.8)
            t.wait_until("The histogram")
            self.play(FadeIn(ax), FadeIn(hist.bars), FadeIn(n_lab), run_time=0.6)
            for b in batches:
                hist.add(b)
                self.play(hist.update(), run_time=0.35)
            self.play(Create(lo), Create(hi), FadeIn(sd_lab), run_time=0.7)

    def saturation(self):
        self.slide("saturation", "Unscaled, one token takes all the weight",
                   book_section="s-where-the-scale-factor-comes-from")
        raw = [-8.4, -4.0, 18.8, -13.7, 16.3, 26.1]
        scaled = [-0.52, -0.25, 1.18, -0.85, 1.02, 1.63]
        w_raw = [0.000, 0.000, 0.001, 0.000, 0.000, 0.999]
        w_sc = [0.05, 0.06, 0.25, 0.03, 0.21, 0.40]
        pitch, bw, mh = 1.55, 0.5, 2.6
        base = Line([-0.45, 0, 0], [5 * pitch + 0.65 + bw, 0, 0], color=th.INK_3, stroke_width=th.STROKE_AXIS)
        rb, sb, rl, sl, sv = VGroup(), VGroup(), VGroup(), VGroup(), VGroup()
        for i in range(6):
            x = i * pitch
            for grp, w, dx, col in ((rb, w_raw[i], 0, th.BAD), (sb, w_sc[i], bw + 0.2, WEIGHT_C)):
                h = max(0.02, w * mh)
                grp.add(Rectangle(width=bw, height=h, fill_color=col, fill_opacity=0.85, stroke_width=0).move_to(
                    [x + dx, h / 2, 0]))
            rl.add(kit.label(f"{raw[i]:g}", size=th.SIZE_LABEL, color=th.BAD, font=th.MONO).move_to(
                [x + bw / 2 + 0.1, -0.3, 0]))
            sl.add(kit.label(f"{scaled[i]:.2f}", size=th.SIZE_LABEL, color=WEIGHT_C, font=th.MONO).move_to(
                [x + bw / 2 + 0.1, -0.65, 0]))
            sv.add(kit.label(f"{w_sc[i]:.2f}", size=th.SIZE_LABEL, color=WEIGHT_C, font=th.MONO).next_to(
                sb[i], UP, buff=0.08))
        big = kit.label("0.999", size=th.SIZE_LABEL, color=th.BAD, font=th.MONO, weight=MEDIUM)
        big.next_to(rb[5], UP, buff=0.08)
        r_name = kit.label("raw score", size=th.SIZE_LABEL, color=th.BAD).next_to(rl, LEFT, buff=0.3)
        s_name = kit.label("divided by 16", size=th.SIZE_LABEL, color=WEIGHT_C).next_to(sl, LEFT, buff=0.3)
        r_name.align_to(s_name, RIGHT)
        chart = VGroup(base, rb, sb, rl, sl, sv, big, r_name, s_name)
        y_lab = kit.label("softmax weight", size=th.SIZE_LABEL, color=th.INK_3)
        y_lab.next_to(chart, UP, buff=0.2).align_to(chart, LEFT)
        chart.add(y_lab)
        gap = kit.formula(r"e^{16} \approx 8.9 \times 10^{6}", size=th.SIZE_MATH_SMALL)
        scale = kit.formula(r"\frac{1}{\sqrt{256}} = \frac{1}{16} = 0.0625", size=th.SIZE_MATH_SMALL)
        scale[0].set_color(SCALE_C)
        side = VGroup(gap, scale).arrange(DOWN, buff=0.6, aligned_edge=LEFT)
        top = VGroup(chart, side).arrange(RIGHT, buff=0.8)
        note = kit.callout("key", r"The scale fixes saturation, not overflow: subtracting the row maximum "
                                  r"already prevents overflow.", title="Not about range", width=11.6,
                           size=th.SIZE_SMALL)
        col = VGroup(top, note).arrange(DOWN, buff=0.45)
        self.place(col)
        with self.narrate(
            (
                (
                    'Now see what that spread does to a softmax. '
                    'Here are six raw scores from one row, from about minus 14 to 26. '
                    'Two scores 16 apart get weights in a ratio of about 8.9 million. '
                    'So the largest score takes 99.9 percent of the weight, and the gradient through it is about zero. '
                    'Divide by 16, which is 0.0625, and the weights spread across the row. '
                    'The scale is about saturation, not overflow, because subtracting the row maximum already prevents overflow.'
                )
            )
        ) as t:
            self.play(FadeIn(base), FadeIn(rl), FadeIn(r_name), FadeIn(y_lab), run_time=0.7)
            t.wait_until("Two scores 16 apart")
            self.play(FadeIn(gap), run_time=0.6)
            t.wait_until("So the largest")
            self.play(*[GrowFromEdge(b, DOWN) for b in rb], run_time=0.9)
            self.play(FadeIn(big), run_time=0.5)
            t.wait_until("Divide by 16")
            self.play(FadeIn(scale), TransformFromCopy(rl, sl), FadeIn(s_name), run_time=1.0)
            self.play(*[GrowFromEdge(b, DOWN) for b in sb], run_time=0.9)
            self.play(FadeIn(sv), run_time=0.5)
            t.wait_until("The scale is about")
            self.play(FadeIn(note, shift=UP * 0.1), run_time=0.7)

    # ------------------------------------------------------------------
    # The causal mask

    def mask_before(self):
        self.slide("mask-before", "Mask before the softmax, not after", book_section="s-the-causal-mask")
        S = [[2, 1, 0.5, 3], [1, 2, 3, 0.5], [0.5, 1, 2, 1], [1, 0, 2, 1.5]]
        g = kit.grid(4, 4, cell=(0.95, 0.62), values=[[f"{v:g}" for v in r] for r in S],
                     row_labels=[f"query {i}" for i in range(4)], col_labels=[f"key {j}" for j in range(4)])
        g.paint(g.block(0, 4, 0, 4), SCORE_C, tint=0.12)
        upper = [(r, c) for r in range(4) for c in range(4) if c > r]
        cap = kit.label("scores, before the softmax", size=th.SIZE_LABEL, color=th.INK_3)
        left = VGroup(g, cap).arrange(DOWN, buff=0.25)

        def comp(title, vals, total, color):
            h = kit.label(title, size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
            gg = kit.grid(1, 4, cell=(1.1, 0.62), values=[vals])
            tot = kit.label(f"sum {total}", size=th.SIZE_SMALL, color=color, weight=SEMIBOLD)
            body = VGroup(gg, tot).arrange(RIGHT, buff=0.35)
            return VGroup(h, body).arrange(DOWN, buff=0.18, aligned_edge=LEFT)

        good = comp("Mask, then softmax", ["0.27", "0.73", "0", "0"], "1.00", th.GOOD)
        bad = comp("Softmax, then zero the future", ["0.09", "0.23", "0", "0"], "0.32", th.BAD)
        r1 = kit.label("row 1: query 1 sees keys 0 and 1", size=th.SIZE_LABEL, color=self.color, weight=MEDIUM)
        right = VGroup(r1, good, bad).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        layout = VGroup(left, right).arrange(RIGHT, buff=1.0)
        self.place(layout)
        infs = VGroup()
        for r, c in upper:
            m = kit.math_label(r"-\infty", size=th.SIZE_SMALL, color=th.INK_3).move_to(g.cell(r, c).square)
            infs.add(m)
        olds = VGroup(*[g.cell(r, c).text for r, c in upper])
        box = g.outline(1, 2, 0, 4, color=self.color)
        with self.narrate(
            (
                (
                    'The causal mask stops a token from seeing its own future. '
                    'It sets every score where the key comes after the query to minus infinity, before the softmax. '
                    'The future then adds nothing, and the weights still sum to 1. '
                    'Take row 1, whose query may see keys 0 and 1. '
                    'Its weights come out as 0.27 and 0.73. '
                    'Now zero the future after the softmax instead. '
                    'The denominator still counted the future keys, so the kept weights sum to only 0.32. '
                    'The output shrinks by an amount that depends on the content.'
                )
            )
        ) as t:
            self.play(FadeIn(g), FadeIn(cap), run_time=0.8)
            t.wait_until("It sets every score")
            self.play(olds.animate.set_opacity(0), FadeIn(infs),
                      *[g.fill(g.cell(r, c), th.INK_4, tint=0.3) for r, c in upper], run_time=1.0)
            t.wait_until("Take row 1")
            self.play(Create(box), FadeIn(r1), run_time=0.7)
            t.wait_until("Its weights come")
            self.play(FadeIn(good), run_time=0.7)
            t.wait_until("Now zero the future")
            self.play(FadeIn(bad[0]), FadeIn(bad[1][0]), run_time=0.7)
            t.wait_until("The denominator")
            self.play(FadeIn(bad[1][1], scale=0.8), run_time=0.5)
            self.play(Indicate(bad[1][1], color=th.BAD), run_time=0.8)

    def mask_offset(self):
        self.slide("mask-offset", "A cached prefix shifts the mask", book_section="s-the-causal-mask")
        n = 10
        keys = kit.grid(1, n, cell=(0.72, 0.6), values=[[str(j) for j in range(n)]],
                        col_labels=None)
        keys.paint(keys.block(0, 1, 0, 9), th.INFO, tint=0.25)
        keys.paint(keys.cell(0, 9), self.color, tint=0.45)
        k_lab = kit.label("key positions", size=th.SIZE_SMALL, color=th.INK_2)
        cached = Brace(keys.block(0, 1, 0, 9), UP, buff=0.08, color=th.INK_4)
        cached_t = kit.label("9 cached tokens", size=th.SIZE_LABEL, color=th.INFO).next_to(cached, UP, buff=0.06)
        new_t = kit.label("new", size=th.SIZE_LABEL, color=self.color).next_to(keys.cell(0, 9), UP, buff=0.2)
        new_t.align_to(cached_t, DOWN)

        def mask_row(visible):
            gg = kit.grid(1, n, cell=(0.72, 0.6))
            for j in range(n):
                gg.paint(gg.cell(0, j), th.GOOD if j < visible else th.BAD, tint=0.3 if j < visible else 0.22)
            return gg

        wrong = mask_row(1)
        right = mask_row(10)
        w_lab = kit.label("trusting row number 0", size=th.SIZE_SMALL, color=th.INK_2)
        r_lab = kit.label("offset 9: position 9", size=th.SIZE_SMALL, color=th.INK_2)
        w_note = kit.label("hides keys 1 to 9", size=th.SIZE_SMALL, color=th.BAD, weight=MEDIUM)
        r_note = kit.label("sees all 10", size=th.SIZE_SMALL, color=th.GOOD, weight=MEDIUM)
        grids = VGroup(keys, wrong, right).arrange(DOWN, buff=0.5)
        for name, gg in ((k_lab, keys), (w_lab, wrong), (r_lab, right)):
            name.next_to(gg, LEFT, buff=0.4)
        rows = VGroup(VGroup(k_lab, keys), VGroup(w_lab, wrong), VGroup(r_lab, right))
        w_note.next_to(wrong, RIGHT, buff=0.3)
        r_note.next_to(right, RIGHT, buff=0.3)
        cached.next_to(keys.block(0, 1, 0, 9), UP, buff=0.08)
        cached_t.next_to(cached, UP, buff=0.06)
        new_t.next_to(keys.cell(0, 9), UP, buff=0.2).align_to(cached_t, DOWN)
        f = kit.formula(r"\text{pos}(i) = i + (\text{kv\_len} - \text{q\_len})", size=th.SIZE_MATH_SMALL)
        fig = VGroup(rows, w_note, r_note, cached, cached_t, new_t)
        col = VGroup(fig, f).arrange(DOWN, buff=0.6)
        self.place(col)
        with self.narrate(
            (
                (
                    "During decode, the keys include a cached prefix that the queries don't cover. "
                    'Say nine tokens are cached, and one new token arrives. '
                    'The key tensor has ten rows, but the query tensor has one, numbered 0. '
                    'A mask that trusts that row number hides keys 1 to 9. '
                    'That query really sits at position 9, and it may see all ten keys. '
                    'So shift each row by the key count minus the query count. '
                    'Here the offset is 9.'
                )
            )
        ) as t:
            t.wait_until("Say nine tokens")
            self.play(FadeIn(k_lab), FadeIn(keys), FadeIn(cached), FadeIn(cached_t), FadeIn(new_t), run_time=0.9)
            t.wait_until("A mask that trusts")
            self.play(FadeIn(w_lab), FadeIn(wrong), run_time=0.7)
            self.play(FadeIn(w_note, shift=LEFT * 0.1), run_time=0.5)
            t.wait_until("That query really")
            self.play(FadeIn(r_lab), FadeIn(right), FadeIn(r_note, shift=LEFT * 0.1), run_time=0.8)
            t.wait_until("So shift each row")
            self.play(FadeIn(f, shift=UP * 0.1), run_time=0.8)
            t.wait_until("Here the offset")
            self.play(Indicate(r_lab, color=self.color), run_time=0.8)

    def mask_regimes(self):
        self.slide("mask-regimes", "Three regimes, one offset", book_section="s-the-causal-mask")
        panel = kit.code(MASK_SRC, "python")

        def regime(q_len, kv_len, title, sub, cell=0.4):
            off = kv_len - q_len
            gg = kit.grid(q_len, kv_len, cell=cell)
            for i in range(q_len):
                for j in range(kv_len):
                    blocked = j > i + off
                    gg.paint(gg.cell(i, j), th.BAD if blocked else th.INFO, tint=0.45 if blocked else 0.18)
            h = kit.label(title, size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
            s = kit.label(sub, size=th.SIZE_LABEL, color=th.INK_3)
            return VGroup(h, s, gg).arrange(DOWN, buff=0.15)

        pre = regime(4, 4, "Prefill", "4 × 4, offset 0: 6 masked")
        dec = regime(1, 10, "Decode step", "1 × 10, offset 9: none masked")
        chk = regime(3, 10, "Chunked prefill", "3 × 10, offset 7: 3 masked")
        regs = VGroup(pre, dec, chk).arrange(RIGHT, buff=0.8, aligned_edge=DOWN)
        for r in regs:
            r[0].align_to(regs, UP)
            r[1].next_to(r[0], DOWN, buff=0.15)
        key = VGroup(swatch("visible", th.INFO, tint=0.18), swatch("masked", th.BAD, tint=0.45)).arrange(DOWN, buff=0.2,
                                                                                               aligned_edge=LEFT)
        top = VGroup(panel, key).arrange(RIGHT, buff=0.6)
        col = VGroup(top, regs).arrange(DOWN, buff=0.5)
        self.place(col)
        with self.narrate(
            (
                (
                    "A key is blocked when its index passes the query's index plus the offset. "
                    'Lab 07 checks three regimes. '
                    'Prefill, with 4 queries and 4 keys, masks 6 entries. '
                    'A decode step, with 1 query and 10 keys, masks nothing. '
                    'A chunked prefill, with 3 queries and 10 keys, masks 3. '
                    'Forget the offset, and the mask is right in prefill and wrong in every decode step. '
                    'Nothing crashes. '
                    'The model drifts, because each step hides most of its own context.'
                )
            )
        ) as t:
            self.play(FadeIn(panel.frame), *panel.reveal(1), run_time=0.7)
            t.wait_until("A key is blocked")
            self.play(*panel.reveal(2, 3, 4), panel.highlight(4), run_time=0.8)
            self.play(*panel.reveal(5), run_time=0.5)
            t.wait_until("Prefill, with 4")
            self.play(FadeIn(pre), FadeIn(key), run_time=0.7)
            t.wait_until("A decode step")
            self.play(FadeIn(dec), run_time=0.7)
            t.wait_until("A chunked prefill")
            self.play(FadeIn(chk), run_time=0.7)
            t.wait_until("Forget the offset")
            self.play(Circumscribe(panel.line(4), color=th.BAD, buff=0.06), run_time=1.2)

    # ------------------------------------------------------------------
    # Shapes, end to end

    def shapes(self):
        self.slide("shapes", "The attention block is wider than the stream", book_section="s-shapes-end-to-end")
        u = 1 / 2048  # Manim units per channel
        h = 0.5

        def bar(ch, color, text=None, ticks=0):
            r = Rectangle(width=ch * u, height=h, fill_color=th.tint(color, 0.35), fill_opacity=1,
                          stroke_color=color, stroke_width=th.STROKE_HAIRLINE)
            g = VGroup(r)
            if ticks:
                step = r.width / ticks
                for i in range(1, ticks):
                    x = r.get_left()[0] + i * step
                    g.add(Line([x, -h / 2, 0], [x, h / 2, 0], color=color, stroke_width=1, stroke_opacity=0.6))
            if text:
                lab = kit.label(text, size=th.SIZE_LABEL, color=th.INK, font=th.MONO)
                lab.move_to(r)
                bg = BackgroundRectangle(lab, color=th.tint(color, 0.35), fill_opacity=1, buff=0.04)
                g.add(bg, lab)
            g.rect = r
            return g

        q = bar(6144, Q_C, "q 6144", ticks=24)
        gate = bar(6144, GATE_C, "gate 6144")
        qg = VGroup(q, gate).arrange(RIGHT, buff=0)
        k = bar(1024, self.color, ticks=4)
        v = bar(1024, self.color, ticks=4)
        kv = VGroup(k, v).arrange(RIGHT, buff=0.12)
        kv_t = kit.label("1024 each", size=th.SIZE_LABEL, color=th.INK_2, font=th.MONO)
        specs = [
            ("residual stream x", bar(5120, th.INK_4, "5120"), ""),
            ("query projection", qg, "62.9M"),
            ("key, value projections", VGroup(kv, kv_t), "5.2M each"),
            ("heads merged", bar(6144, Q_C, "6144", ticks=24), ""),
            ("output projection", bar(5120, th.INK_4, "5120"), "31.5M"),
        ]
        rows = VGroup()
        names = VGroup(*[kit.label(s[0], size=th.SIZE_SMALL, color=th.INK_2) for s in specs])
        name_w = max(n.width for n in names)
        params = {}
        for i, (name, (_, b, p)) in enumerate(zip(names, specs)):
            y = -i * 0.8
            name.move_to([name_w - name.width / 2, y, 0])
            b.move_to([name_w + 0.4 + b.width / 2, y, 0])
            if b is specs[2][1]:
                kv.move_to([name_w + 0.4 + kv.width / 2, y, 0])
                kv_t.next_to(kv, RIGHT, buff=0.25)
            if p:
                pl = kit.label(p, size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
                pl.move_to([name_w + 0.4 + 6.0 + 0.5 + pl.width / 2, y, 0])
                params[i] = pl
            rows.add(VGroup(name, b))
        p_head = kit.label("parameters", size=th.SIZE_LABEL, color=th.INK_3).next_to(params[1], UP, buff=0.35)
        p_head.align_to(params[1], LEFT)
        params = VGroup(*[params.get(i, VGroup()) for i in range(5)])
        tot = kit.prose(r"One layer: \textbf{104.9M} parameters, 210 MB in bf16; 3.4 GB over 16 layers.",
                        size=th.SIZE_SMALL, width=None)
        tie = kit.prose(r"$d_h = 256$, not $5120 / 24 = 213.33$.", size=th.SIZE_SMALL, color=th.BAD, width=None)
        fig = VGroup(rows, params, p_head)
        col = VGroup(fig, tot, tie).arrange(DOWN, buff=0.45)
        self.place(col)
        with self.narrate(
            (
                (
                    'Follow the shapes through one layer, with the bars drawn to scale. '
                    'The residual stream is 5120 wide. '
                    'The query projection emits 12288 channels: 24 heads of 256 for the queries, and as many again for the gate. '
                    'The key and value projections emit only 1024 each. '
                    'After attention, the heads merge into 6144 channels, and the output projection brings them back to 5120. '
                    "The head width isn't the hidden size over the head count, which would be 213.33. "
                    'One layer holds 104.9 million parameters, 210 MB in bfloat16. '
                    'Keys and values are only 10.5 million of them.'
                )
            )
        ) as t:
            self.play(FadeIn(rows[0]), run_time=0.7)
            t.wait_until("The query projection")
            self.play(FadeIn(names[1]), GrowFromEdge(qg, LEFT), FadeIn(p_head), FadeIn(params[1]), run_time=1.0)
            t.wait_until("The key and value")
            self.play(FadeIn(names[2]), GrowFromEdge(kv, LEFT), FadeIn(kv_t), FadeIn(params[2]), run_time=0.8)
            t.wait_until("After attention")
            self.play(FadeIn(rows[3]), run_time=0.7)
            self.play(FadeIn(rows[4]), FadeIn(params[4]), run_time=0.7)
            t.wait_until("The head width")
            self.play(FadeIn(tie), run_time=0.7)
            t.wait_until("One layer holds")
            self.play(FadeIn(tot), run_time=0.7)
            t.wait_until("Keys and values are")
            self.play(Indicate(params[2], color=self.color), Indicate(kv, color=self.color), run_time=0.9)

    # ------------------------------------------------------------------
    # Multi-head attention

    def heads(self):
        self.slide("heads", "Heads are free in arithmetic", book_section="s-multi-head-attention")
        r = kit.rng(11)

        def mini(weights, color, w=0.95, hgt=0.7):
            frame = RoundedRectangle(width=w, height=hgt + 0.2, corner_radius=0.06, stroke_color=th.RULE,
                                     stroke_width=th.STROKE_HAIRLINE, fill_color=th.RAISE, fill_opacity=1)
            n = len(weights)
            bw = (w - 0.2) / n
            bars = VGroup()
            top = max(weights)
            for i, x in enumerate(weights):
                bh = max(0.02, hgt * x / top)
                b = Rectangle(width=bw * 0.75, height=bh, fill_color=color, fill_opacity=0.85, stroke_width=0)
                b.move_to([frame.get_left()[0] + 0.1 + bw * (i + 0.5), frame.get_bottom()[1] + 0.1 + bh / 2, 0])
                bars.add(b)
            return VGroup(frame, bars)

        one = mini(r.dirichlet(np.ones(8)), th.INK_3, w=3.2, hgt=1.0)
        one_lab = kit.label("one head, 6144 wide: one ranking of the context", size=th.SIZE_LABEL, color=th.INK_3)
        top = VGroup(one, one_lab).arrange(RIGHT, buff=0.4)
        cols = [self.color, th.INFO, th.GOOD, th.VIOLET]
        minis = VGroup(*[mini(r.dirichlet(np.full(8, 0.6)), cols[i % 4]) for i in range(24)])
        minis.arrange_in_grid(rows=3, cols=8, buff=(0.2, 0.2))
        many_lab = kit.label("24 heads, 256 wide each: 24 separate rankings and averages", size=th.SIZE_LABEL,
                             color=th.INK_3)
        bottom = VGroup(minis, many_lab).arrange(DOWN, buff=0.2)
        f = kit.formula(r"2\,T L\,(H d_h)", r"=", r"H \cdot (2\,T L\, d_h)", size=th.SIZE_MATH_SMALL)
        f_lab = kit.label("score FLOPs: the same either way", size=th.SIZE_LABEL, color=th.INK_3)
        ff = VGroup(f, f_lab).arrange(RIGHT, buff=0.4)
        col = VGroup(top, bottom, ff).arrange(DOWN, buff=0.45)
        self.place(col)
        with self.narrate(
            (
                (
                    'Real attention runs 24 heads in parallel, each with its own queries, keys, and values. '
                    'One head gives each query one ranking of the context, and so one average. '
                    'A token that needs its subject, its topic, and a matching bracket needs three rankings. '
                    'With 24 heads, it gets 24 separate averages for the output projection to combine. '
                    'Splitting the width into heads costs no extra FLOPs. '
                    'What a head gives up is rank: its scores live in a 256-dimensional subspace.'
                )
            )
        ) as t:
            t.wait_until("One head gives")
            self.play(FadeIn(one[0]), LaggedStart(*[GrowFromEdge(b, DOWN) for b in one[1]], lag_ratio=0.1),
                      FadeIn(one_lab), run_time=1.0)
            t.wait_until("With 24 heads")
            self.play(LaggedStart(*[FadeIn(m, scale=0.85) for m in minis], lag_ratio=0.04), run_time=1.6)
            self.play(FadeIn(many_lab), run_time=0.5)
            t.wait_until("Splitting the width")
            self.play(FadeIn(ff, shift=UP * 0.1), run_time=0.8)
            t.wait_until("What a head gives up")
            self.play(Indicate(minis[0], color=self.color, scale_factor=1.15), run_time=0.9)

    # ------------------------------------------------------------------
    # MHA, MQA, and GQA

    def sharing(self):
        self.slide("sharing", "Six query heads share one KV head", book_section="s-mha-mqa-and-gqa")
        side = 0.24

        def scheme(n_kv):
            qs = VGroup(*[Square(side, stroke_color=Q_C, stroke_width=1.5, fill_color=th.tint(Q_C, 0.3),
                                 fill_opacity=1) for _ in range(24)]).arrange(RIGHT, buff=0.05)
            group = 24 // n_kv
            if n_kv == 4:
                for i in range(1, 4):
                    for sq in qs[6 * i:]:
                        sq.shift(RIGHT * 0.2)
            kvs = VGroup()
            for i in range(n_kv):
                members = qs[group * i:group * i + group]
                k = Rectangle(width=max(members.width, side), height=0.34, fill_color=th.tint(self.color, 0.45),
                              fill_opacity=1, stroke_color=self.color, stroke_width=1.5)
                k.move_to([members.get_x(), qs.get_y() - 0.9, 0])
                kvs.add(k)
            links = VGroup(*[Line(sq.get_bottom(), kvs[i // group].get_top(), color=th.INK_4, stroke_width=1.2)
                             for i, sq in enumerate(qs)])
            return VGroup(qs, links, kvs)

        rows = []
        for n_kv, name, desc in [(24, "MHA", "24 KV heads: one each"), (1, "MQA", "1 KV head: shared by all"),
                                 (4, "GQA", "4 KV heads: groups of 6")]:
            d = scheme(n_kv)
            nm = kit.label(name, size=th.SIZE_BODY, color=th.INK, weight=SEMIBOLD)
            ds = kit.label(desc, size=th.SIZE_SMALL, color=self.color if n_kv == 4 else th.INK_2)
            r = VGroup(nm, d, ds).arrange(RIGHT, buff=0.5)
            rows.append(r)
        rows = VGroup(*rows).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        for r in rows:
            r[1].align_to(rows[0][1], LEFT)
            r[2].next_to(rows[0][1], RIGHT, buff=0.5).set_y(r[1].get_y())
        key = VGroup(swatch("query head", Q_C, tint=0.3), swatch("key and value head, cached", self.color,
                                                                   tint=0.45)).arrange(RIGHT, buff=0.6)
        g = kit.formula(r"g = H / H_{kv} = 24 / 4 = 6", size=th.SIZE_MATH_SMALL)
        bottom = VGroup(key, g).arrange(RIGHT, buff=1.0)
        col = VGroup(rows, bottom).arrange(DOWN, buff=0.5)
        self.place(col)
        with self.narrate(
            "How many key and value heads do you need to keep? "
            "Queries are used on the spot, but keys and values stay in the cache for the rest of the sequence. "
            "So let several query heads read the same key and value head. "
            "Multi-head attention gives every query head its own: 24 KV heads. "
            "Multi-query attention shares one set across all of them. "
            "Grouped-query attention sits between. "
            "This model keeps 4 KV heads, so each group of 6 query heads shares one."
        ) as t:
            self.play(FadeIn(key), run_time=0.6)
            for r, cue in zip(rows, ["Multi-head attention", "Multi-query", "Grouped-query attention"]):
                t.wait_until(cue)
                d = r[1]
                self.play(FadeIn(r[0]), FadeIn(d[0]), run_time=0.5)
                self.play(LaggedStart(*[Create(l) for l in d[1]], lag_ratio=0.02), FadeIn(d[2]), FadeIn(r[2]),
                          run_time=0.9)
            t.wait_until("This model keeps")
            self.play(FadeIn(g), Indicate(rows[2][1][2], color=self.color, scale_factor=1.1), run_time=0.9)

    def cache_table(self):
        self.slide("cache-table", "GQA's cache costs 64 KiB per token", book_section="s-mha-mqa-and-gqa")
        f = kit.formula(r"\text{bytes} = 2 \times", r"H_{kv}", r"\times d_h \times", r"b", size=th.SIZE_MATH_SMALL)
        f[1].set_color(WEIGHT_C)
        f[3].set_color(OUT_C)
        f_lab = kit.label("per token, per layer", size=th.SIZE_LABEL, color=th.INK_3)
        ff = VGroup(f, f_lab).arrange(RIGHT, buff=0.4)
        tab = kit.table(
            [
                ["MHA", "24", "24 KiB", "384 KiB", "12 GiB"],
                ["GQA", "4", "4 KiB", "64 KiB", "2 GiB"],
                ["MQA", "1", "1 KiB", "16 KiB", "0.5 GiB"],
            ],
            header=["Scheme", "$H_{kv}$", "Per layer", "Over 16 layers", "At 32k context"],
            align=["l", "r", "r", "r", "r"],
            size=th.SIZE_SMALL,
            col_gap=0.8,
            highlight_rows=[1],
        )
        bc = kit.bar_compare([("GQA", 11, self.color, "11 sequences"), ("MHA", 2, th.INK_4, "2 sequences")],
                             width=5.0, bar_height=0.42, gap=0.22)
        bc_lab = kit.label("At 32k context, in the 26 GB the weights leave", size=th.SIZE_LABEL, color=th.INK_3)
        bcg = VGroup(bc_lab, bc).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        col = VGroup(ff, tab, bcg).arrange(DOWN, buff=0.5)
        self.place(col)
        with self.narrate(
            (
                (
                    'Count the cache bytes per token in one layer: 2, for keys and values, times the KV heads, times 256, times 2 bytes. '
                    "With grouped-query attention, that's 4 KiB per layer. "
                    "Over 16 layers, it's 64 KiB per token, and 2 GiB at 32k context. "
                    'Multi-head attention costs six times that: 12 GiB at 32k. '
                    'Multi-query costs 16 KiB per token. '
                    'The weights leave about 26 GB of the card. '
                    'With GQA at 32k, that holds 11 sequences, the batch chapter 2 derives. '
                    'With MHA, it holds two.'
                )
            )
        ) as t:
            t.wait_until("Count the cache")
            self.play(FadeIn(ff), run_time=0.8)
            t.wait_until("With grouped-query")
            self.play(FadeIn(tab.rules), FadeIn(row(tab, 0)), FadeIn(tab.bands), FadeIn(row(tab, 2)), run_time=0.8)
            t.wait_until("Multi-head attention costs")
            self.play(FadeIn(row(tab, 1), shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("Multi-query costs")
            self.play(FadeIn(row(tab, 3), shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("The weights leave")
            self.play(FadeIn(bc_lab), FadeIn(bc.labels), run_time=0.6)
            t.wait_until("With GQA at 32k")
            self.play(GrowFromEdge(bc.bars[0], LEFT), FadeIn(bc.values[0]), run_time=0.8)
            t.wait_until("With MHA, it holds")
            self.play(GrowFromEdge(bc.bars[1], LEFT), FadeIn(bc.values[1]), run_time=0.6)

    def quality(self):
        self.slide("quality", "A few KV heads keep nearly all the quality", book_section="s-mha-mqa-and-gqa")
        bc = kit.bar_compare(
            [
                ("24 KV heads (MHA)", 384, th.INK_4, "384 KiB"),
                ("8 KV heads", 128, self.color, "128 KiB, a third"),
                ("4 KV heads (GQA)", 64, self.color, "64 KiB, a sixth"),
                ("1 KV head (MQA)", 16, th.BAD, "16 KiB, quality loss"),
            ],
            width=6.2, bar_height=0.45, gap=0.28,
        )
        cap = kit.label("Cache per token over 16 layers", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        chart = VGroup(cap, bc).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        bul = kit.bullets(
            [
                r"The GQA paper mean-pools a trained MHA checkpoint's KV heads, then uptrains briefly.",
                r"This model trained with 4 KV heads from the start: no conversion loss.",
                r"Only 16 of its 64 layers use attention at all.",
            ],
            size=th.SIZE_SMALL, width=11.5, buff=0.25,
        )
        col = VGroup(chart, bul).arrange(DOWN, buff=0.55)
        self.place(col)
        with self.narrate(
            (
                (
                    'The trade costs a little quality. '
                    "The GQA paper mean-pools a trained multi-head checkpoint's KV heads into groups, then uptrains briefly. "
                    'Collapsing all the way to one KV head loses measurable quality. '
                    'A handful of groups recovers nearly all of it, and keeps most of the cache saving. '
                    "This model's case is better. "
                    "It was trained with four KV heads from the start, so there's no conversion loss. "
                    'And only 16 of its layers use attention at all.'
                )
            )
        ) as t:
            self.play(FadeIn(cap), FadeIn(bc.labels[0]), GrowFromEdge(bc.bars[0], LEFT), FadeIn(bc.values[0]),
                      run_time=0.8)
            t.wait_until("The GQA paper")
            self.play(FadeIn(bul[0], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("Collapsing all the way")
            self.play(FadeIn(bc.labels[3]), GrowFromEdge(bc.bars[3], LEFT), FadeIn(bc.values[3]), run_time=0.8)
            t.wait_until("A handful of groups")
            self.play(FadeIn(bc.labels[1:3]), GrowFromEdge(bc.bars[1], LEFT), GrowFromEdge(bc.bars[2], LEFT),
                      FadeIn(bc.values[1:3]), run_time=0.9)
            t.wait_until("It was trained")
            self.play(FadeIn(bul[1], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("And only 16")
            self.play(FadeIn(bul[2], shift=RIGHT * 0.1), run_time=0.6)

    # ------------------------------------------------------------------
    # Why decode attention is permanently memory bound

    def intensity(self):
        self.slide("intensity", "Decode attention does 6 FLOPs per byte",
                   book_section="s-why-decode-attention-is-permanently-memory-bound")
        sz = th.SIZE_MATH

        def m(tex, color=th.INK):
            return MathTex(tex, tex_template=MATH_TEMPLATE, font_size=sz * th.TEX_SCALE, color=color)

        flops = kit.formula(r"\text{FLOPs} = 4 H", r"L", r"d_h", size=th.SIZE_MATH_SMALL)
        byts = kit.formula(r"\text{bytes} = 2", r"H_{kv}", r"L", r"d_h", r"b", size=th.SIZE_MATH_SMALL)
        flops[1].set_color(SCORE_C)
        byts[1].set_color(WEIGHT_C)
        byts[2].set_color(SCORE_C)
        byts[4].set_color(OUT_C)
        defs = VGroup(flops, byts).arrange(RIGHT, buff=1.2)
        lhs = m(r"I =")
        num = VGroup(m("4H"), m("L", SCORE_C), m("d_h")).arrange(RIGHT, buff=0.12)
        den = VGroup(m("2"), m("H_{kv}", WEIGHT_C), m("L", SCORE_C), m("d_h"), m("b", OUT_C)).arrange(RIGHT, buff=0.12)
        bar = Line(LEFT, RIGHT, color=th.INK, stroke_width=3)
        bar.set_width(max(num.width, den.width) + 0.2)
        frac = VGroup(num, bar, den).arrange(DOWN, buff=0.15)
        res = VGroup(m("="), VGroup(m("2H"), Line(LEFT, RIGHT, color=th.INK, stroke_width=3).set_width(1.3),
                                     VGroup(m("H_{kv}", WEIGHT_C), m("b", OUT_C)).arrange(RIGHT, buff=0.1)).arrange(
            DOWN, buff=0.15)).arrange(RIGHT, buff=0.3)
        eq = VGroup(lhs, frac, res).arrange(RIGHT, buff=0.3)
        final = kit.formula(r"b = 2:\quad I = \frac{H}{H_{kv}} = g = 6 \ \text{FLOPs per byte}", size=th.SIZE_MATH)
        box = SurroundingRectangle(final, color=self.color, buff=0.2, corner_radius=0.08, stroke_width=3)
        note = kit.label("one decode step, one sequence, one layer, context L", size=th.SIZE_LABEL, color=th.INK_3)
        col = VGroup(note, defs, eq, VGroup(final, box)).arrange(DOWN, buff=0.5)
        self.place(col)
        crosses = VGroup(*[Line(p.get_corner(DL), p.get_corner(UR), color=th.BAD, stroke_width=4).scale(1.25)
                           for p in (num[1], num[2], den[2], den[3])])
        with self.narrate(
            (
                (
                    'Arithmetic intensity is the FLOPs an operation does per byte it moves from HBM. '
                    'Take one decode step for one sequence in one layer, at context L. '
                    'The scores cost 2 H L d FLOPs, and the weighted sum of values costs the same again. '
                    'The cache read is K and V: the KV heads times L times the head width, at b bytes each. '
                    'Divide the two. '
                    'The head width and the context length both cancel. '
                    'With 2 bytes per element, the intensity is the query heads over the KV heads: the group size. '
                    "For this model, that's exactly 6 FLOPs per byte."
                )
            )
        ) as t:
            self.play(FadeIn(note), run_time=0.5)
            t.wait_until("The scores cost")
            self.play(FadeIn(flops), run_time=0.7)
            t.wait_until("The cache read")
            self.play(FadeIn(byts), run_time=0.7)
            t.wait_until("Divide the two")
            self.play(FadeIn(lhs), TransformFromCopy(flops[1:], num[1:]), FadeIn(num[0]), FadeIn(bar), run_time=0.8)
            self.play(TransformFromCopy(byts[1:], den[1:]), FadeIn(den[0]), run_time=0.8)
            t.wait_until("The head width and")
            self.play(LaggedStart(*[Create(c) for c in crosses], lag_ratio=0.2), run_time=1.0)
            self.play(FadeIn(res, shift=LEFT * 0.1), run_time=0.7)
            t.wait_until("With 2 bytes")
            self.play(FadeIn(final), run_time=0.8)
            t.wait_until("For this model")
            self.play(Create(box), run_time=0.6)

    def ridge(self):
        self.slide("ridge", "27 times below the ridge", book_section="s-why-decode-attention-is-permanently-memory-bound")
        rp = kit.prose(r"A100 ridge point: 312 TFLOP/s $\div$ 1935 GB/s $\approx$ 161 FLOPs per byte",
                       size=th.SIZE_SMALL, width=None, color=th.INK_2)
        bc = kit.bar_compare(
            [
                ("A100 ridge point", 161, th.INK_4, "161"),
                ("Decode attention, bf16 cache", 6, self.color, "6: under 4% of the arithmetic"),
                ("With an int8 cache", 12, th.GOOD, "12"),
            ],
            width=7.0, bar_height=0.48, gap=0.3,
        )
        unit = kit.label("FLOPs per byte", size=th.SIZE_LABEL, color=th.INK_3)
        chart = VGroup(unit, bc).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        bul = kit.bullets(
            [
                r"\textbf{Longer context doesn't help:} $L$ cancelled.",
                r"\textbf{Bigger batches don't help:} each sequence reads its own cache.",
                r"\textbf{Read fewer bytes:} GQA, paged attention (ch.\ 15), int8 caches (ch.\ 18).",
            ],
            size=th.SIZE_SMALL, width=11.5, buff=0.22,
        )
        col = VGroup(rp, chart, bul).arrange(DOWN, buff=0.5)
        self.place(col)
        with self.narrate(
            (
                (
                    'Compare that with the A100. '
                    'Its ridge point, where arithmetic and memory finish together, is about 161 FLOPs per byte. '
                    'Decode attention sits 27 times below the ridge, using under 4 percent of the arithmetic. '
                    "Longer context doesn't help, because L cancelled. "
                    "Batching doesn't help either, because each sequence reads its own cache. "
                    'The only lever left is to read fewer bytes. '
                    'An int8 cache, for example, halves b and doubles the intensity to 12.'
                )
            )
        ) as t:
            t.wait_until("Its ridge point")
            self.play(FadeIn(rp), FadeIn(unit), FadeIn(bc.labels[:2]), GrowFromEdge(bc.bars[0], LEFT),
                      FadeIn(bc.values[0]), run_time=1.0)
            t.wait_until("Decode attention sits")
            self.play(GrowFromEdge(bc.bars[1], LEFT), FadeIn(bc.values[1]), run_time=0.8)
            t.wait_until("Longer context")
            self.play(FadeIn(bul[0], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("Batching doesn't")
            self.play(FadeIn(bul[1], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("The only lever")
            self.play(FadeIn(bul[2], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("An int8 cache")
            self.play(FadeIn(bc.labels[2]), GrowFromEdge(bc.bars[2], LEFT), FadeIn(bc.values[2]), run_time=0.8)

    # ------------------------------------------------------------------
    # The output gate

    def gate(self):
        self.slide("gate", "The output gate gives a head an off switch", book_section="s-the-output-gate")
        panel = kit.code(GATE_SRC, "python")
        ax = kit.axes([-8, 8, 4], [0, 1, 0.5], width=4.6, height=2.2, x_numbers=[-8, -4, 0, 4, 8],
                      y_numbers=[0, 1], x_label=r"\text{gate}", y_label=r"\sigma(\text{gate})")
        curve = ax.plot(lambda x: 1 / (1 + np.exp(-x)), x_range=[-8, 8], color=GATE_C, stroke_width=th.STROKE_CURVE)
        plot = VGroup(ax, curve)
        base = Line(LEFT * 0.6, RIGHT * 0.6, color=th.INK_3, stroke_width=th.STROKE_AXIS)
        out_lab = kit.label("one head's output", size=th.SIZE_LABEL, color=th.INK_3)
        out_box = VGroup(base, out_lab).arrange(DOWN, buff=0.2)
        lower = VGroup(plot, out_box).arrange(RIGHT, buff=1.2, aligned_edge=DOWN)
        col = VGroup(panel, lower).arrange(DOWN, buff=0.5)
        self.place(col)
        x = ValueTracker(4.0)
        sig = lambda v: 1 / (1 + np.exp(-v))  # noqa: E731
        dot = always_redraw(lambda: Dot(ax.c2p(x.get_value(), sig(x.get_value())), radius=0.09, color=th.INK))
        hb = 2.0

        def out_bar():
            hh = max(0.01, hb * sig(x.get_value()))
            return Rectangle(width=0.9, height=hh, fill_color=Q_C, fill_opacity=0.85, stroke_width=0).next_to(
                base, UP, buff=0)

        bar_ = always_redraw(out_bar)
        val = always_redraw(lambda: kit.label(f"× {sig(x.get_value()):.3f}", size=th.SIZE_LABEL, color=th.INK_2,
                                              font=th.MONO).next_to(base, UP,
                                                                    buff=max(0.01, hb * sig(x.get_value())) + 0.12))
        with self.narrate(
            (
                (
                    'The query projection emits 12288 channels, and the second half is a gate. '
                    'Chunk it in two, run attention on the queries, then multiply the output by the sigmoid of the gate. '
                    "Attention returns a weighted average, so a head can't say that nothing here is relevant. "
                    'Without a gate, models dump unwanted attention on the first token, an attention sink. '
                    'A sigmoid near zero switches the head off, per channel and per token. '
                    "It's bounded between 0 and 1, so it can attenuate, but never amplify."
                )
            )
        ) as t:
            self.play(FadeIn(panel.frame), *panel.reveal(1), run_time=0.7)
            t.wait_until("Chunk it in two")
            self.play(*panel.reveal(2, 3), run_time=0.7)
            self.play(*panel.reveal(4, 5), panel.highlight(4), run_time=0.7)
            t.wait_until("Attention returns")
            self.play(Create(ax), Create(curve), FadeIn(out_box), run_time=1.0)
            self.add(dot, bar_, val)
            self.play(FadeIn(dot), run_time=0.3)
            t.wait_until("A sigmoid near zero")
            self.play(x.animate.set_value(-8.0), run_time=2.2)
            t.wait_until("It's bounded")
            self.play(Indicate(curve, color=GATE_C, scale_factor=1.03), run_time=1.0)

    def gate_slice(self):
        self.slide("gate-slice", "Slicing instead of chunking drops the gate", book_section="s-the-output-gate")
        u = 1 / 1400
        h = 0.55

        def seg(ch, color, text):
            r = Rectangle(width=ch * u, height=h, fill_color=th.tint(color, 0.35), fill_opacity=1,
                          stroke_color=color, stroke_width=th.STROKE_HAIRLINE)
            lab = kit.label(text, size=th.SIZE_LABEL, color=th.INK, font=th.MONO).move_to(r)
            return VGroup(r, lab)

        def proj():
            return VGroup(seg(6144, Q_C, "queries 6144"), seg(6144, GATE_C, "gate 6144")).arrange(RIGHT, buff=0)

        top_lab = kit.label("Query projection output, 12288 channels", size=th.SIZE_SMALL, color=th.INK_2)
        p1 = proj()
        c_lab = kit.label("chunk(2): queries, and a gate through a sigmoid", size=th.SIZE_SMALL, color=th.GOOD,
                          weight=MEDIUM)
        p2 = proj()
        s_lab = kit.label("slice [:6144]: same shape, no error, gate gone", size=th.SIZE_SMALL, color=th.BAD,
                          weight=MEDIUM)
        blocks = VGroup(VGroup(top_lab, p1).arrange(DOWN, buff=0.15, aligned_edge=LEFT),
                        VGroup(c_lab, p2).arrange(DOWN, buff=0.15, aligned_edge=LEFT)).arrange(
            DOWN, buff=0.45, aligned_edge=LEFT)
        p3 = proj()
        sl = VGroup(s_lab, p3).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        blocks.add(sl)
        blocks.arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        cost = kit.callout(
            "note",
            r"$5120 \times 6144 = 31{,}457{,}280$ parameters, 62.9 MB per layer: about 1.0 GB of the 53.8 GB "
            r"model, for the ability to output nothing.",
            title="What the gate costs", width=11.6, size=th.SIZE_SMALL,
        )
        col = VGroup(blocks, cost).arrange(DOWN, buff=0.55)
        self.place(col)
        sig_box = kit.box("sigmoid", color=GATE_C, size=th.SIZE_LABEL, height=0.5)
        sig_box.next_to(p2[1], RIGHT, buff=0.3)
        cross = VGroup(Line(p3[1].get_corner(DL), p3[1].get_corner(UR), color=th.BAD, stroke_width=4),
                       Line(p3[1].get_corner(UL), p3[1].get_corner(DR), color=th.BAD, stroke_width=4))
        with self.narrate(
            (
                (
                    'The gate costs one extra 5120 by 6144 matrix per layer. '
                    "That's about 31.5 million parameters, or 62.9 MB in bfloat16. "
                    "Across 16 layers, it's about 1 GB, spent on the ability to output nothing. "
                    'The correct split chunks the projection into queries and a gate. '
                    'If you slice the first 6144 channels instead, the shapes still work. '
                    'Half the projection is discarded, and the gate is never applied. '
                    "Quality drops in a way that's hard to trace."
                )
            )
        ) as t:
            self.play(FadeIn(top_lab), FadeIn(p1), run_time=0.7)
            t.wait_until("That's about 31.5")
            self.play(FadeIn(cost, shift=UP * 0.1), run_time=0.7)
            t.wait_until("The correct split")
            self.play(FadeIn(c_lab), TransformFromCopy(p1, p2), run_time=0.8)
            self.play(p2[1].animate.shift(RIGHT * 0.15), FadeIn(sig_box), run_time=0.6)
            t.wait_until("If you slice")
            self.play(FadeIn(s_lab), TransformFromCopy(p1, p3), run_time=0.8)
            t.wait_until("Half the projection")
            self.play(p3[1].animate.set_opacity(0.25), Create(cross), run_time=0.8)

    # ------------------------------------------------------------------
    # Share KV heads without copying them

    def repeat_kv(self):
        self.slide("repeat-kv", "Index the shared KV head; don't copy it",
                   book_section="s-share-kv-heads-without-copying-them")
        w, gap = 0.3, 0.05
        cols = [self.color, th.INFO, th.GOOD, th.VIOLET]

        def slots(style):
            out = VGroup()
            for i in range(24):
                c = cols[i // 6]
                if style == "view":
                    r = DashedVMobject(Rectangle(width=w, height=0.45, stroke_color=c, stroke_width=2), num_dashes=10)
                else:
                    r = Rectangle(width=w, height=0.45, stroke_color=c, stroke_width=1.5,
                                  fill_color=th.tint(c, 0.4), fill_opacity=1)
                out.add(r)
            out.arrange(RIGHT, buff=gap)
            for i in range(1, 4):
                for r in out[6 * i:]:
                    r.shift(RIGHT * 0.2)
            return out

        view = slots("view")
        copy = slots("copy")
        stored = VGroup()
        for i in range(4):
            grp = view[6 * i:6 * i + 6]
            b = Rectangle(width=grp.width, height=0.45, stroke_color=cols[i], stroke_width=2,
                          fill_color=th.tint(cols[i], 0.4), fill_opacity=1)
            lab = kit.label(f"KV head {i}", size=th.SIZE_LABEL, color=th.INK)
            stored.add(VGroup(b, lab))
        labels = [kit.label(t, size=th.SIZE_SMALL, color=c, weight=MEDIUM) for t, c in
                  (("stored: 4 heads", th.INK_2), ("expand: a view, free", th.GOOD), ("reshape: a copy", th.BAD))]
        stored_y, view_y, copy_y = 0, -1.0, -2.0
        view.move_to([0, view_y, 0])
        copy.move_to([0, copy_y, 0])
        for i in range(4):
            grp = view[6 * i:6 * i + 6]
            stored[i][0].move_to([grp.get_x(), stored_y, 0])
            stored[i][1].move_to(stored[i][0])
        name_x = view.get_left()[0] - 0.4
        for lab, y in zip(labels, (stored_y, view_y, copy_y)):
            lab.move_to([name_x - lab.width / 2, y, 0])
        refs = VGroup(*[Line(view[i].get_top(), stored[i // 6][0].get_bottom(), color=th.INK_4, stroke_width=1.2)
                        for i in range(24)])
        fig = VGroup(*labels, stored, view, copy, refs)
        bc = kit.bar_compare([("Stored, 32k context", 128, self.color, "128 MiB"),
                              ("After the copy", 768, th.BAD, "768 MiB")],
                             width=5.5, bar_height=0.4, gap=0.2)
        idx = kit.mono("kv_head = head // kv_group   # 13 // 6 = 2", size=th.SIZE_SMALL, color=th.INK)
        col = VGroup(fig, bc, idx).arrange(DOWN, buff=0.5)
        self.place(col)
        hit = SurroundingRectangle(view[13], color=th.INK, buff=0.05, stroke_width=3)
        hit2 = SurroundingRectangle(stored[2][0], color=th.INK, buff=0.05, stroke_width=3)
        with self.narrate(
            (
                (
                    'The naive helper copies KV heads in two steps. '
                    'First, expand adds a group axis with stride zero, so the six copies are the same memory. '
                    "That's free. "
                    "Then reshape can't keep that view, so it materializes every copy. "
                    'At 32k context, one layer stores 128 MiB of keys and values. '
                    "After the copy, it's 768 MiB, which undoes what GQA saved. "
                    'A real kernel indexes the shared head instead. '
                    'Query head 13 reads KV head 13 divided by 6, rounded down, which is 2.'
                )
            )
        ) as t:
            self.play(FadeIn(labels[0]), FadeIn(stored), run_time=0.7)
            t.wait_until("First, expand")
            self.play(FadeIn(labels[1]), FadeIn(view), LaggedStart(*[Create(l) for l in refs], lag_ratio=0.02),
                      run_time=1.2)
            t.wait_until("Then reshape")
            self.play(FadeIn(labels[2]), LaggedStart(*[TransformFromCopy(view[i], copy[i]) for i in range(24)],
                                                     lag_ratio=0.03), run_time=1.4)
            t.wait_until("At 32k context")
            self.play(FadeIn(bc.labels[0]), GrowFromEdge(bc.bars[0], LEFT), FadeIn(bc.values[0]), run_time=0.7)
            t.wait_until("After the copy")
            self.play(FadeIn(bc.labels[1]), GrowFromEdge(bc.bars[1], LEFT), FadeIn(bc.values[1]), run_time=0.9)
            t.wait_until("A real kernel")
            self.play(FadeIn(idx), run_time=0.6)
            t.wait_until("Query head 13")
            self.play(Create(hit), Create(hit2), Indicate(refs[13], color=th.INK), run_time=1.0)

    # ------------------------------------------------------------------
    # The naive implementation

    def naive(self):
        self.slide("naive", "Keep the naive path as your ground truth",
                   book_section="s-the-naive-implementation-and-why-you-keep-it")
        panel = kit.code(NAIVE_SRC, "python")
        bc = kit.bar_compare([("Scores, bf16", 3, self.color, "3 GiB"), ("Scores, float32", 6, th.BAD, "6 GiB")],
                             width=5.0, bar_height=0.42, gap=0.22)
        cap = kit.label("8k context, 24 heads, one sequence", size=th.SIZE_LABEL, color=th.INK_3)
        chart = VGroup(cap, bc).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        peak = kit.callout("warning", r"Three separate score-sized tensors. Peak memory: well past 12 GB, "
                                      r"$27.5\times$ a FlashAttention kernel's.", title="Can't serve",
                           width=5.6, size=th.SIZE_SMALL)
        lower = VGroup(chart, peak).arrange(RIGHT, buff=0.7)
        col = VGroup(panel, lower).arrange(DOWN, buff=0.55)
        self.place(col)
        with self.narrate(
            (
                (
                    'The naive implementation is four statements: scores, mask, softmax, and the weighted sum. '
                    'It materializes one score for every query and key pair. '
                    'At 8k context with 24 heads, the scores alone take 3 GiB in bfloat16. '
                    'The listing computes in float32, which doubles that, and it keeps several score-sized tensors. '
                    "Peak memory runs well past 12 GB, 27.5 times a FlashAttention kernel's. "
                    "Keep it anyway: it's the reference every fast kernel gets checked against."
                )
            )
        ) as t:
            self.play(FadeIn(panel.frame), *panel.reveal(), run_time=0.9)
            t.wait_until("It materializes")
            self.play(panel.highlight(1, 2), run_time=0.6)
            t.wait_until("At 8k context")
            self.play(FadeIn(cap), FadeIn(bc.labels[0]), GrowFromEdge(bc.bars[0], LEFT), FadeIn(bc.values[0]),
                      run_time=0.8)
            t.wait_until("The listing computes")
            self.play(FadeIn(bc.labels[1]), GrowFromEdge(bc.bars[1], LEFT), FadeIn(bc.values[1]), run_time=0.8)
            t.wait_until("Peak memory runs")
            self.play(FadeIn(peak, shift=UP * 0.1), run_time=0.7)
            t.wait_until("Keep it anyway")
            self.play(Circumscribe(panel.frame[0], color=th.GOOD, buff=0.05), run_time=1.2)

    # ------------------------------------------------------------------
    # Where decode time goes

    def decode_time(self):
        self.slide("decode-time", "At batch 1, reading the weights dominates", book_section="s-where-decode-time-goes")
        tab = kit.table(
            [
                ["Rated", "1935", "27.8 ms", "36"],
                ["Measured copy", "1275", "42.2 ms", "24"],
            ],
            header=["Bandwidth", "GB/s", "53.8 GB weight read", "Tokens/s"],
            align=["l", "r", "r", "r"],
            size=th.SIZE_SMALL,
            col_gap=0.8,
        )
        u = 9.0 / 44.0
        hgt = 0.6
        w_seg = Rectangle(width=42.2 * u, height=hgt, fill_color=th.INFO, fill_opacity=0.85, stroke_width=0)
        c_seg = Rectangle(width=1.8 * u, height=hgt, fill_color=self.color, fill_opacity=0.9, stroke_width=0)
        c_seg.next_to(w_seg, RIGHT, buff=0)
        w_t = kit.label("weights 42.2 ms", size=th.SIZE_LABEL, color=th.CANVAS, weight=MEDIUM).move_to(w_seg)
        c_t = kit.label("cache + state at 32k: 1.8 ms", size=th.SIZE_LABEL, color=self.color, weight=MEDIUM)
        c_t.next_to(c_seg, DOWN, buff=0.15).align_to(c_seg, RIGHT)
        step_lab = kit.label("One decode step at 1275 GB/s", size=th.SIZE_SMALL, color=th.INK_2)
        stepbar = VGroup(step_lab, VGroup(w_seg, c_seg, w_t, c_t)).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        c_t.next_to(c_seg, DOWN, buff=0.15).align_to(c_seg, RIGHT)
        batch = kit.callout("key", r"Sixteen sequences read the weights once and the cache 16 times: "
                                   r"almost $16\times$ the throughput.", title="The argument for batching",
                            width=11.0, size=th.SIZE_SMALL)
        col = VGroup(tab, stepbar, batch).arrange(DOWN, buff=0.55)
        self.place(col)
        with self.narrate(
            (
                (
                    'At batch size 1, a decode step reads all 53.8 GB of weights, plus the cache and the state. '
                    'At the rated 1935 GB/s, the weight read takes 27.8 ms, or 36 tokens per second. '
                    "At the measured 1275 GB/s, it's 42.2 ms, or 24 tokens per second. "
                    'At 32k context, the cache and state add about 1.8 ms. '
                    "That grows with context and batch, but the weight read doesn't. "
                    'Sixteen sequences read the weights once, so throughput goes up almost 16 times.'
                )
            )
        ) as t:
            t.wait_until("At the rated")
            self.play(FadeIn(tab.rules), FadeIn(row(tab, 0)), FadeIn(row(tab, 1)), run_time=0.8)
            t.wait_until("At the measured")
            self.play(FadeIn(row(tab, 2)), run_time=0.6)
            self.play(FadeIn(step_lab), GrowFromEdge(w_seg, LEFT), run_time=0.8)
            self.play(FadeIn(w_t), run_time=0.4)
            t.wait_until("At 32k context")
            self.play(GrowFromEdge(c_seg, LEFT), FadeIn(c_t), run_time=0.7)
            t.wait_until("Sixteen sequences")
            self.play(FadeIn(batch, shift=UP * 0.1), run_time=0.7)
