"""Chapter 08, the forward pass.

Teaches content/chapters/08-the-forward-pass.md in its order. Every number
below was recomputed in Python and matches the chapter. Geometry: h = 5120,
i = 17408, V = 248,320, 64 layers, 24 query heads and 4 KV heads of width
256, every fourth layer full attention.

Layer schedule: full attention when (l + 1) mod 4 = 0, so l in {3, 7, ..., 63};
64 / 4 = 16 full-attention layers, 64 - 16 = 48 linear-attention layers.

Full-attention shapes: q_proj 12288 = 2 x 6144 (query + output gate);
24 x 256 = 6144; k and v 4 x 256 = 1024 each; 24 / 4 = 6 query heads per KV
head; o_proj 6144 -> 5120.

Linear-attention shapes: in_proj_qkvz 2048 + 2048 + 6144 + 6144 = 16384;
conv over q, k, v: 2048 + 2048 + 6144 = 10240; q, k 16 x 128 = 2048;
v 48 x 128 = 6144; repeat 48 / 16 = 3 times; state (B, 48, 128, 128).
Bars on those two slides are drawn at 3072 channels per Manim unit.

SwiGLU parameters:
- Ungated at 4h: 2 x 5120 x 20480 = 8h^2 = 209,715,200 ("about 210 million").
- Equal count: i = 8/3 x 5120 = 13,653.3 ("13,653"); 3 x 5120 x 13,653.3
  = 209,715,200 by construction.
- At 17408 = 3.4h: 3 x 5120 x 17408 = 267,386,880 ("about 267 million");
  267,386,880 / 209,715,200 - 1 = 0.275 (27.5%).
- x 64 = 17,112,760,320 = 17.1 billion (chapter: 64% of the model).

Alignment: 17408 = 17 x 1024; 17408 / 8 = 2176 = 17 x 128;
17408 / 16 = 1088 = 17 x 64; 13,653 / 128 = 106.66, not whole.
The grid draws 8 shards x 17 cells of 128 channels.

SiLU: bisection on SiLU'(z) = sigma(z)(1 + z(1 - sigma(z))) = 0 gives
z* = -1.27846, SiLU(z*) = -0.27846 ("-1.278", "-0.278").

Output head: 5120 x 248,320 = 1,271,398,400 parameters (1.27 billion);
x 2 bytes = 2.5428e9 B = 2.54 GB. Untied embedding of the same size:
2 x 1,271,398,400 = 2,542,796,800 = 2.54 billion (chapter: 9.5% of the model).

Slice before you project, 2048-token prefill:
- All positions: 2 x 2048 x 5120 x 248,320 = 5.2076e12 FLOPs ("5.21 x 10^12",
  "about 5.21 trillion"); output 2048 x 248,320 x 2 B = 1.017e9 B = 1.02 GB.
- Last position: 2 x 5120 x 248,320 = 2.5428e9 = 2.54 GFLOPs; output
  248,320 x 2 B = 496,640 B = 497 KB.
- 5.2076e12 / 312e12 = 16.69 ms ("16.7 ms", roofline floor).
- Ratio 5.2076e12 / 2.5428e9 = 2048 ("2048 times the arithmetic you need").

Logit example (illustrative, inside the chapter's 10-to-30 range):
z = 24, 22.5, 21, 18, 14, 11; softmax p = 0.784, 0.175, 0.039, 0.0019,
0.0000, 0.0000. Adding c = 4 leaves p unchanged (shift invariance).

Validation (chapter values): max abs diff ~1e-2 on logits of magnitude
10 to 30; correlation > 0.9999; top-1 agreement 1.0; KL from p to q.
bf16: 8 significand bits, eps = 2^-8 = 0.00390625 ("about 0.0039",
"3.9 x 10^-3"). Bisection tolerance 1e-2 = 2.56 eps ("roughly three times").
Prefill-decode reference maxerr 5.96e-07 (float32, small config).

Bisection chart: simulated per-layer differences, kit.rng(19): layers 0-2
uniform in [0.002, 0.006]; from layer 3, 0.014 + 0.0016 (l - 3) plus
uniform [0, 0.003]. Tolerance line at 0.01. First crossing: layer 3.
"""

from manim import *

import kit
from kit import theme as th

UNIT = 3072.0  # channels per Manim unit on the shape-flow slides

BLOCK_SRC = """\
class SwiGLU(nn.Module):
    def __init__(self, h, i, bias=False):  # h=5120, i=17408
        super().__init__()
        self.gate_proj = nn.Linear(h, i, bias=bias)
        self.up_proj   = nn.Linear(h, i, bias=bias)
        self.down_proj = nn.Linear(i, h, bias=bias)

    def forward(self, x):                 # (batch, seq, 5120)
        gate = self.gate_proj(x)          # (batch, seq, 17408)
        up   = self.up_proj(x)            # (batch, seq, 17408)
        hidden = F.silu(gate) * up        # (batch, seq, 17408)
        return self.down_proj(hidden)     # (batch, seq, 5120)
"""

HEAD_SRC = """\
x = self.norm(x)                     # (batch, seq, 5120)
if last_token_only:
    x = x[:, -1:, :]                 # (batch, 1, 5120)
return self.lm_head(x)               # (batch, seq_out, 248320)
"""


Q_COL = th.HL_B
GATE_COL = th.HL_D
KV_COL = th.HL_C
K_COL = th.VIOLET
X_COL = th.INK_3


def seg(channels, color, text=None, h=0.42):
    """A bar whose length is a tensor's width in channels."""
    w = channels / UNIT
    rect = Rectangle(width=w, height=h, stroke_color=color, stroke_width=th.STROKE_HAIRLINE + 0.5,
                     fill_color=th.tint(color, 0.3), fill_opacity=1)
    group = VGroup(rect)
    group.rect = rect
    if text:
        t = kit.label(text, size=th.SIZE_LABEL, color=th.INK)
        if t.width <= w - 0.12:
            t.move_to(rect)
            group.add(t)
    return group


def ticks(bar, n, color=None):
    """n - 1 thin dividers that cut a bar into n heads."""
    r = bar.rect
    color = color or r.get_stroke_color()
    out = VGroup()
    for k in range(1, n):
        x = r.get_left()[0] + r.width * k / n
        out.add(Line([x, r.get_bottom()[1], 0], [x, r.get_top()[1], 0], stroke_width=1.2, color=color))
    bar.add(out)
    return bar


def shape_rows(rows, name_w=3.9, bar_w=6.2, pitch=0.66):
    """Lays out (name, mono, bars, shape) rows in three columns.

    Returns a VGroup of rows; each row has `.name`, `.bars`, `.shape`.
    """
    out = VGroup()
    for k, (name, mono, bars, shape) in enumerate(rows):
        y = -k * pitch
        nm = kit.mono(name, size=th.SIZE_LABEL, color=th.INK) if mono else kit.label(
            name, size=th.SIZE_LABEL, color=th.INK_2)
        nm.move_to([nm.width / 2, y, 0])
        bars.move_to([name_w + bars.width / 2, y, 0])
        sh = kit.mono(shape, size=th.SIZE_LABEL, color=th.INK_3)
        sh.move_to([name_w + bar_w + 0.4 + sh.width / 2, y, 0])
        row = VGroup(nm, bars, sh)
        row.name, row.bars, row.shape = nm, bars, sh
        out.add(row)
    return out


def swatch(text, color, size=th.SIZE_LABEL):
    sq = Square(0.26, stroke_color=th.RULE_STRONG, stroke_width=th.STROKE_HAIRLINE,
                fill_color=th.tint(color, 0.35), fill_opacity=1)
    return VGroup(sq, kit.label(text, size=size, color=th.INK_2)).arrange(RIGHT, buff=0.16)


def plus_node(r=0.22, color=th.INK_2):
    c = Circle(radius=r, stroke_color=color, stroke_width=th.STROKE_BOX + 0.5, fill_color=th.RAISE, fill_opacity=1)
    h = Line(LEFT * r * 0.55, RIGHT * r * 0.55, color=color, stroke_width=3)
    v = Line(DOWN * r * 0.55, UP * r * 0.55, color=color, stroke_width=3)
    g = VGroup(c, h, v)
    g.circle = c
    return g


def vbars(values, color, scale, width=0.3, gap=0.12, tint=0.45):
    """Vertical bars from a baseline at y = 0; negative values hang below it."""
    out = VGroup()
    for k, v in enumerate(values):
        hgt = max(abs(v) * scale, 0.02)
        r = Rectangle(width=width, height=hgt, stroke_color=color, stroke_width=th.STROKE_HAIRLINE,
                      fill_color=th.tint(color, tint), fill_opacity=1)
        x = k * (width + gap) + width / 2
        r.move_to([x, hgt / 2 if v >= 0 else -hgt / 2, 0])
        out.add(r)
    return out


def baseline(bars, color=th.INK_3):
    x0, x1 = bars.get_left()[0] - 0.12, bars.get_right()[0] + 0.12
    return Line([x0, 0, 0], [x1, 0, 0], color=color, stroke_width=th.STROKE_AXIS)


class Lecture(kit.Lecture):
    def construct(self):
        self.title_slide(
            narration=(
                "The forward pass. "
                "In this lecture, you assemble the whole 64-layer model, derive its MLP, and prove its outputs "
                "match the reference."
            )
        )
        self.why_thorough()
        self.whole_model()
        self.decoder_block()
        self.full_attention()
        self.linear_attention()
        self.schedule()
        self.gated_unit()
        self.silu()
        self.swiglu_code()
        self.mlp_params()
        self.alignment()
        self.output_head()
        self.slice_first()
        self.logits()
        self.loading()
        self.compare_logits()
        self.bit_exact()
        self.bisect()
        self.prefill_decode()
        self.what_goes_wrong()
        self.closing_slide(
            [
                r"Every layer: two pre-norm residual sublayers. Full attention at 3, 7, \ldots, 63.",
                r"SwiGLU at $i = 17408$ holds 64\% of the parameters.",
                r"Final norm, slice to the last position, then project.",
                r"Compare logits in float32; bisect to the first bad layer.",
            ],
            narration=(
                "Here's what to keep. "
                "Every layer is two pre-norm residual sublayers, and only the mixer changes, with full attention "
                "at every fourth layer. "
                "The SwiGLU MLP holds 64% of the parameters. "
                "Apply the final norm, slice to the last position, and only then project. "
                "Validate logits in float32 with a tolerance, bisect any mismatch to its first layer, and make "
                "prefill and decode agree. "
                "Next, the KV cache."
            ),
            book_section="s-what-goes-wrong",
        )

    # ------------------------------------------------------------------
    # Introduction

    def why_thorough(self):
        self.slide("why-thorough", "Every later test compares you against you")
        mine = kit.box("Your model", sublabel="this chapter, bf16", color=self.color, width=3.6, height=1.5,
                       size=th.SIZE_BODY)
        later = VGroup(
            kit.box("Fused kernel", sublabel="vs your unfused one", width=4.0),
            kit.box("Paged attention", sublabel="vs your contiguous one", width=4.0),
            kit.box("Quantized matmul", sublabel="vs your bf16 one", width=4.0),
        ).arrange(DOWN, buff=0.35)
        hf = kit.box("Hugging Face transformers", sublabel="the last independent check", color=th.INFO,
                     width=4.4, height=1.5)
        row = VGroup(later, mine, hf).arrange(RIGHT, buff=1.2)
        warn = kit.callout("warning", r"If this model is wrong, every later test still passes.",
                           title="Why it matters now", width=9.0, size=th.SIZE_SMALL)
        col = VGroup(row, warn).arrange(DOWN, buff=0.6)
        self.place(col)
        arrows = VGroup(*[kit.arrow(b, mine, color=th.INK_4) for b in later])
        ref = kit.arrow(hf, mine, color=th.INFO)
        with self.narrate(
            "This chapter runs the whole model for the first time. "
            "From chapter 10 onward, every optimization is checked against the model you finish today. "
            "A fused kernel against your unfused one. "
            "A paged attention kernel against your contiguous one. "
            "A quantized matmul against your bf16 one. "
            "If this model is wrong, every later test still passes, and nothing tells you. "
            "Hugging Face transformers is the last independent check you get, so use it properly."
        ) as t:
            self.play(FadeIn(mine, scale=0.95), run_time=0.8)
            for k, cue in enumerate(["A fused kernel", "A paged attention", "A quantized matmul"]):
                t.wait_until(cue)
                self.play(FadeIn(later[k], shift=RIGHT * 0.15), GrowArrow(arrows[k].line), run_time=0.7)
            t.wait_until("If this model is wrong")
            self.play(FadeIn(warn, shift=UP * 0.1), run_time=0.7)
            t.wait_until("Hugging Face")
            self.play(FadeIn(hf, shift=LEFT * 0.15), GrowArrow(ref.line), run_time=0.8)

    # ------------------------------------------------------------------
    # The whole model, in order

    def whole_model(self):
        self.slide("whole-model", "One prompt's path through the model", book_section="s-the-whole-model-in-order")
        w, hgt = 2.7, 1.6
        emb = kit.box("Embedding", sublabel="chapter 3", width=w, height=hgt, color=th.INFO, size=th.SIZE_BODY)
        layers = kit.box("64 layers", sublabel="norm, mixer, MLP", width=w, height=hgt, color=self.color,
                         size=th.SIZE_BODY)
        back = VGroup(*[layers.rect.copy().set_fill(th.tint(self.color, 0.05)).shift(UR * 0.1 * k)
                        for k in (2, 1)])
        norm = kit.box("Final norm", sublabel="RMSNorm", width=w, height=hgt, color=th.INFO, size=th.SIZE_BODY)
        head = kit.box("Output head", sublabel="new in this chapter", width=w, height=hgt, color=self.color,
                       size=th.SIZE_BODY)
        stack_ = VGroup(back, layers)
        row = VGroup(emb, stack_, norm, head).arrange(RIGHT, buff=0.65)
        ids = kit.mono("token ids  (B, T)", size=th.SIZE_SMALL, color=th.INK_2)
        ids.next_to(emb, UP, buff=1.0)
        shapes = VGroup()
        for bx, s_ in zip((emb, layers, norm, head),
                          ("(B, T, 5120)", "(B, T, 5120)", "(B, T, 5120)", "(B, 1, 248320)")):
            m = kit.mono(s_, size=th.SIZE_LABEL, color=th.INK_3)
            m.next_to(bx, DOWN, buff=0.25)
            shapes.add(m)
        shapes[1].set_y(shapes[0].get_y())
        strip = kit.grid(1, 64, cell=(0.034, 0.2), gap=0.006)
        strip.next_to(layers.rect, UP, buff=0.4)
        strip_t = kit.label("layer 0 to 63", size=th.SIZE_LABEL, color=th.INK_3).next_to(strip, UP, buff=0.1)
        stream = kit.label("residual stream: 5120 per token, never changes width", size=th.SIZE_SMALL,
                           color=th.INK_2)
        scores = kit.label("248,320 scores", size=th.SIZE_SMALL, color=self.color, weight=MEDIUM)
        new = kit.label("MLP: new in this chapter", size=th.SIZE_LABEL, color=self.color, weight=MEDIUM)
        fig = VGroup(ids, row, shapes, strip, strip_t)
        col = VGroup(fig, stream).arrange(DOWN, buff=0.75)
        self.place(col)
        arrows = VGroup(kit.arrow(ids, emb, color=th.INK_4, buff=0.08),
                        *[kit.arrow(a_, b_, color=th.INK_4, buff=0.06)
                          for a_, b_ in ((emb, back), (stack_, norm), (norm, head))])
        under = Line(shapes[0].get_left(), shapes[2].get_right(), color=th.INFO, stroke_width=4)
        under.next_to(shapes[0:3], DOWN, buff=0.14)
        stream.next_to(under, DOWN, buff=0.2)
        scores.next_to(shapes[3], DOWN, buff=0.2)
        new.next_to(layers.rect, DOWN, buff=0.25).set_y(scores.get_y())
        new.next_to(stream, DOWN, buff=0.25)
        with self.narrate(
            "Here's the whole journey of one prompt. "
            "The embedding table from chapter 3 turns each token id into a 5120-wide vector. "
            "That vector starts the token's residual stream: the running vector every layer reads and adds to. "
            "Sixty-four decoder layers each add two updates to it. "
            "Then a final RMSNorm and one large matrix multiply turn the last vector into a score per vocabulary "
            "token. "
            "This chapter adds the two missing parts: the MLP, and the output head."
        ) as t:
            self.play(FadeIn(ids), run_time=0.6)
            t.wait_until("The embedding table")
            self.play(GrowArrow(arrows[0].line), FadeIn(emb), FadeIn(shapes[0]), run_time=0.8)
            dot = Dot(radius=0.11, color=th.INFO).move_to(ids.get_bottom() + DOWN * 0.2)
            self.play(dot.animate.move_to(emb.rect), run_time=0.6)
            t.wait_until("That vector starts")
            self.play(Create(under), FadeIn(stream), run_time=0.9)
            t.wait_until("Sixty-four decoder")
            self.play(GrowArrow(arrows[1].line), FadeIn(stack_), FadeIn(shapes[1]), FadeIn(strip), FadeIn(strip_t),
                      dot.animate.move_to(layers.rect), run_time=0.8)
            self.play(LaggedStart(*[c.square.animate.set_fill(th.tint(self.color, 0.6)) for c in strip.cells[0]],
                                  lag_ratio=0.04), Indicate(layers.rect, color=self.color, scale_factor=1.03),
                      run_time=2.0)
            t.wait_until("Then a final RMSNorm")
            self.play(GrowArrow(arrows[2].line), FadeIn(norm), FadeIn(shapes[2]), dot.animate.move_to(norm.rect),
                      run_time=0.9)
            self.play(GrowArrow(arrows[3].line), FadeIn(head), dot.animate.move_to(head.rect), run_time=0.9)
            self.play(FadeOut(dot, scale=2.5), FadeIn(shapes[3]), FadeIn(scores), run_time=0.8)
            t.wait_until("This chapter adds")
            self.play(FadeIn(new), Circumscribe(layers.rect, color=self.color, buff=0.06),
                      Circumscribe(head.rect, color=self.color, buff=0.06), run_time=1.2)

    # ------------------------------------------------------------------
    # The decoder block

    def decoder_block(self):
        self.slide("decoder-block", "Two pre-norm sublayers, each added back", book_section="s-the-decoder-block")
        ty, by = 0.0, 1.7
        x_in, s1, p1, s2, p2, x_out = -6.4, -5.75, -0.35, 0.1, 5.75, 6.4
        trunk = Line([x_in, ty, 0], [x_out, ty, 0], color=th.INK_3, stroke_width=4)
        n1 = kit.box("norm", width=1.3, height=0.85, color=th.INFO).move_to([-4.85, by, 0])
        mx = kit.box("Mixer", sublabel="across positions", width=2.55, height=1.1, color=self.color)
        mx.move_to([-2.2, by, 0])
        n2 = kit.box("norm", width=1.3, height=0.85, color=th.INFO).move_to([1.25, by, 0])
        mlp = kit.box("MLP", sublabel="across channels", width=2.55, height=1.1, color=th.GOOD)
        mlp.move_to([3.75, by, 0])
        add1 = plus_node().move_to([p1, ty, 0])
        add2 = plus_node().move_to([p2, ty, 0])

        def branch(sx, nb, bx, px):
            up = VMobject(color=th.INK_3, stroke_width=3)
            up.set_points_as_corners([[sx, ty, 0], [sx, by, 0], [nb.rect.get_left()[0], by, 0]])
            mid = Arrow(nb.rect.get_right(), bx.rect.get_left(), buff=0.0, color=th.INK_3, stroke_width=3,
                        tip_length=0.16, max_tip_length_to_length_ratio=0.5)
            over = Line(bx.rect.get_right(), [px, by, 0], color=th.INK_3, stroke_width=3)
            down = Arrow([px, by, 0], [px, ty + 0.22, 0], buff=0.0, color=th.INK_3, stroke_width=3,
                         tip_length=0.18)
            return VGroup(up, mid, over, down)

        b1 = branch(s1, n1, mx, p1)
        b2 = branch(s2, n2, mlp, p2)
        xin = kit.formula("x", size=th.SIZE_MATH, color=th.INK).next_to(trunk, LEFT, buff=0.15)
        xout = kit.formula("x", size=th.SIZE_MATH, color=th.INK).next_to(trunk, RIGHT, buff=0.15)
        diagram = VGroup(trunk, b1, b2, n1, mx, n2, mlp, add1, add2, xin, xout)
        f1 = kit.formula(r"x \leftarrow x + \operatorname{mixer}\big(\operatorname{norm}_1(x)\big)",
                         size=th.SIZE_MATH_SMALL)
        f2 = kit.formula(r"x \leftarrow x + \operatorname{MLP}\big(\operatorname{norm}_2(x)\big)",
                         size=th.SIZE_MATH_SMALL)
        fs = VGroup(f1, f2).arrange(RIGHT, buff=1.2)
        note = kit.label("the residual stream: never normalized", size=th.SIZE_SMALL, color=self.color,
                         weight=MEDIUM)
        col = VGroup(diagram, fs).arrange(DOWN, buff=1.1)
        self.place(col)
        note.next_to(trunk, DOWN, buff=0.45).set_x(0)
        glow = trunk.copy().set_stroke(self.color, width=7)
        with self.narrate(
            "Every one of the 64 layers has the same skeleton. "
            "First, it normalizes the stream, and runs the mixer, which moves information between positions. "
            "The result is added back to the stream. "
            "Then it normalizes again, runs the MLP, and adds that result back too. "
            "This is pre-norm: each normalization sits inside its branch, not after the addition. "
            "Now look at the straight path along the bottom: the residual stream itself is never normalized. "
            "So a signal can cross all 64 layers without a single scaling operation. "
            "Every modern decoder is pre-norm."
        ) as t:
            self.play(Create(trunk), FadeIn(xin), FadeIn(xout), FadeIn(add1), FadeIn(add2), run_time=0.9)
            t.wait_until("First, it normalizes")
            dot = Dot(radius=0.1, color=th.INFO).move_to([x_in, ty, 0])
            self.play(FadeIn(dot), Create(b1[0]), FadeIn(n1), run_time=0.8)
            self.play(GrowArrow(b1[1]), FadeIn(mx), run_time=0.7)
            t.wait_until("The result is added")
            self.play(Create(b1[2]), GrowArrow(b1[3]), FadeIn(f1), run_time=0.8)
            self.play(dot.animate.move_to(add1), Indicate(add1, color=self.color), run_time=0.8)
            t.wait_until("Then it normalizes again")
            self.play(Create(b2[0]), FadeIn(n2), GrowArrow(b2[1]), FadeIn(mlp), run_time=0.9)
            self.play(Create(b2[2]), GrowArrow(b2[3]), FadeIn(f2), run_time=0.7)
            self.play(dot.animate.move_to(add2), Indicate(add2, color=self.color), run_time=0.8)
            t.wait_until("Now look at the straight")
            self.play(dot.animate.move_to([x_out, ty, 0]), run_time=0.6)
            self.play(FadeOut(dot), Create(glow), run_time=1.0)
            self.play(FadeIn(note, shift=UP * 0.08), run_time=0.7)

    def full_attention(self):
        self.slide("full-attention", "Sixteen layers mix with full attention", book_section="s-the-decoder-block")
        q = seg(6144, Q_COL, "q · 6144")
        g = seg(6144, GATE_COL, "gate · 6144")
        qproj = VGroup(q, g).arrange(RIGHT, buff=0)
        heads = ticks(seg(6144, Q_COL, None), 24)
        kv = VGroup(seg(1024, KV_COL, "k"), seg(1024, KV_COL, "v")).arrange(RIGHT, buff=0.2)
        cache = VGroup(*[seg(1024, KV_COL, None) for _ in range(2)]).arrange(RIGHT, buff=0.2)
        ghosts = VGroup(*[c.rect.copy().set_fill(th.tint(KV_COL, 0.12)).set_stroke(opacity=0.5).shift(
            UR * 0.07 * k) for c in cache for k in (3, 2, 1)])
        ghosts.set_z_index(-1)
        cache_g = VGroup(ghosts, cache)
        out = seg(6144, Q_COL, "out · 6144")
        oproj = seg(5120, X_COL, "5120")
        rows = shape_rows([
            ("input x", False, seg(5120, X_COL, "x · 5120"), "(B, T, 5120)"),
            ("q_proj", True, qproj, "(B, T, 12288)"),
            ("24 query heads", False, heads, "(B, 24, T, 256)"),
            ("k_proj, v_proj", True, kv, "(B, 4, T, 256)"),
            ("append to the cache", False, cache_g, "(B, 4, L, 256)"),
            ("attention × sigmoid(gate)", False, out, "(B, T, 6144)"),
            ("o_proj", True, oproj, "(B, T, 5120)"),
        ], bar_w=4.2)
        key = kit.label("B batch · T tokens in this pass · L context tokens", size=th.SIZE_LABEL, color=th.INK_3)
        col = VGroup(rows, key).arrange(DOWN, buff=0.5)
        self.place(col)
        gate_copy = g.copy()
        with self.narrate(
            "Follow the shapes through a full-attention layer. "
            "B is the batch, T is the tokens in this pass, and L is the context length. "
            "The input is 5120 wide per token. "
            "The query projection is twice as wide as the query heads need. "
            "The second half is chapter 7's output gate, which lets a head switch itself off. "
            "The first half splits into 24 query heads of width 256. "
            "Keys and values are much narrower: 4 KV heads, each shared by six query heads. "
            "After per-head norms and RoPE, they join the cache, which holds all L context tokens. "
            "Attention's output is multiplied by the sigmoid of the gate. "
            "The output projection brings it back to the stream's width."
        ) as t:
            t.wait_until("B is the batch")
            self.play(FadeIn(key), run_time=0.6)
            t.wait_until("The input is")
            self.play(FadeIn(rows[0]), run_time=0.6)
            t.wait_until("The query projection")
            self.play(FadeIn(rows[1].name), FadeIn(rows[1].shape), GrowFromEdge(qproj, LEFT), run_time=0.8)
            t.wait_until("The second half")
            self.play(Indicate(g, color=GATE_COL, scale_factor=1.05), run_time=0.8)
            t.wait_until("The first half splits")
            self.play(TransformFromCopy(q, heads), FadeIn(rows[2].name), FadeIn(rows[2].shape), run_time=1.0)
            t.wait_until("Keys and values are much")
            self.play(FadeIn(rows[3]), run_time=0.8)
            t.wait_until("After per-head norms")
            self.play(TransformFromCopy(kv, cache), FadeIn(rows[4].name), FadeIn(rows[4].shape), run_time=0.8)
            self.play(LaggedStart(*[FadeIn(gh) for gh in ghosts], lag_ratio=0.15), run_time=0.8)
            t.wait_until("Attention's output")
            self.play(FadeIn(rows[5].name), FadeIn(rows[5].shape), FadeIn(out), run_time=0.7)
            self.play(gate_copy.animate.move_to(out).set_opacity(0.0), Indicate(out, color=GATE_COL), run_time=1.0)
            self.remove(gate_copy)
            t.wait_until("The output projection")
            self.play(FadeIn(rows[6]), run_time=0.7)

    def linear_attention(self):
        self.slide("linear-attention", "Forty-eight layers mix with the delta rule",
                   book_section="s-the-decoder-block")
        q = seg(2048, Q_COL, "q")
        k = seg(2048, K_COL, "k")
        v = seg(6144, KV_COL, "v · 6144")
        z = seg(6144, GATE_COL, "z · 6144")
        qkvz = VGroup(q, k, v, z).arrange(RIGHT, buff=0)
        conv = VGroup(seg(2048, Q_COL), seg(2048, K_COL), seg(6144, KV_COL, "conv · 10240")).arrange(RIGHT, buff=0)
        hq = ticks(seg(2048, Q_COL), 16)
        hk = ticks(seg(2048, K_COL), 16)
        hv = ticks(seg(6144, KV_COL), 48)
        heads = VGroup(hq, hk, hv).arrange(RIGHT, buff=0.15)
        rq = ticks(seg(6144, Q_COL), 48)
        rk = ticks(seg(6144, K_COL), 48)
        rv = ticks(seg(6144, KV_COL), 48)
        rep = VGroup(rq, rk, rv).arrange(RIGHT, buff=0.15)
        o = ticks(seg(6144, KV_COL), 48)
        st = VGroup(*[Square(0.42, stroke_color=th.INFO, stroke_width=th.STROKE_HAIRLINE + 0.5,
                             fill_color=th.tint(th.INFO, 0.3), fill_opacity=1).shift(UR * 0.06 * j)
                      for j in (2, 1, 0)])
        st_t = kit.label("state", size=th.SIZE_LABEL, color=th.INFO)
        delta = VGroup(o, st, st_t).arrange(RIGHT, buff=0.25)
        gn = seg(6144, KV_COL, "out · 6144")
        op = seg(5120, X_COL, "5120")
        rows = shape_rows([
            ("input x", False, seg(5120, X_COL, "x · 5120"), "(B, T, 5120)"),
            ("in_proj_qkvz", True, qkvz, "(B, T, 16384)"),
            ("causal conv over q, k, v", False, conv, "(B, T, 10240)"),
            ("heads of width 128", False, heads, "(B, 16, T, 128)"),
            ("repeat q and k 3 times", False, rep, "(B, 48, T, 128)"),
            ("delta rule", False, delta, "(B, 48, T, 128)"),
            ("gated norm with silu(z)", False, gn, "(B, T, 48, 128)"),
            ("out_proj", True, op, "(B, T, 5120)"),
        ], bar_w=6.3, pitch=0.6)
        pill = kit.pill("Both mixers: (B, T, 5120) in, (B, T, 5120) out", self.color, size=th.SIZE_SMALL)
        col = VGroup(rows, pill).arrange(DOWN, buff=0.45)
        self.place(col)
        z_copy = z.copy()
        with self.narrate(
            "The other 48 layers use the gated delta rule, where the past lives in a fixed-size state. "
            "One projection produces the inputs for every head. "
            "They split into queries, keys, values, and a gate called z. "
            "A causal convolution mixes each channel with its previous three steps. "
            "Queries and keys form 16 heads, and values form 48. "
            "So each query and key head repeats 3 times to match. "
            "The delta rule updates a 128 by 128 state per head, gated by alpha and beta. "
            "z gates the output through an RMSNorm, and the output projection closes the layer. "
            "Both mixers take the same shape in and give it back, so you can swap one for the other freely."
        ) as t:
            self.play(FadeIn(rows[0]), run_time=0.6)
            t.wait_until("One projection")
            self.play(FadeIn(rows[1].name), FadeIn(rows[1].shape), GrowFromEdge(qkvz, LEFT), run_time=0.9)
            t.wait_until("They split")
            self.play(LaggedStart(*[Indicate(s, scale_factor=1.06) for s in (q, k, v, z)], lag_ratio=0.25),
                      run_time=1.2)
            t.wait_until("A causal convolution")
            self.play(TransformFromCopy(VGroup(q, k, v), conv), FadeIn(rows[2].name), FadeIn(rows[2].shape),
                      run_time=0.9)
            t.wait_until("Queries and keys form")
            self.play(TransformFromCopy(conv, heads), FadeIn(rows[3].name), FadeIn(rows[3].shape), run_time=0.9)
            t.wait_until("So each query")
            self.play(TransformFromCopy(heads, rep), FadeIn(rows[4].name), FadeIn(rows[4].shape), run_time=0.9)
            t.wait_until("The delta rule updates")
            self.play(FadeIn(rows[5]), run_time=0.8)
            self.play(Indicate(st, color=th.INFO, scale_factor=1.1), run_time=0.7)
            t.wait_until("z gates the output")
            self.play(FadeIn(rows[6]), z_copy.animate.move_to(gn).set_opacity(0.0), run_time=1.0)
            self.remove(z_copy)
            self.play(FadeIn(rows[7]), run_time=0.6)
            t.wait_until("Both mixers take")
            self.play(FadeIn(pill, scale=0.95), Indicate(rows[0].bars, color=self.color),
                      Indicate(rows[7].bars, color=self.color), run_time=0.9)

    # ------------------------------------------------------------------
    # The layer schedule

    def schedule(self):
        self.slide("schedule", "Every fourth layer is full attention", book_section="s-the-layer-schedule")
        rule = kit.formula(r"(\ell + 1) \bmod 4 = 0", size=th.SIZE_MATH)
        g = kit.grid(1, 64, cell=(0.17, 0.75), gap=0.03)
        full = [i for i in range(64) if (i + 1) % 4 == 0]
        nums = VGroup()
        for i in (0, 3, 7, 11, 63):
            n = kit.label(str(i), size=th.SIZE_LABEL, color=th.INK_3)
            n.next_to(g.cell(0, i), DOWN, buff=0.12)
            nums.add(n)
        dots = kit.label("…", size=th.SIZE_LABEL, color=th.INK_3).next_to(g.cell(0, 33), DOWN, buff=0.12)
        nums.add(dots)
        strip = VGroup(g, nums)
        key = VGroup(swatch("16 full attention: KV cache", self.color),
                     swatch("48 gated delta rule: fixed state", th.INFO)).arrange(RIGHT, buff=0.8)
        last = kit.callout("key", r"Layer 63 adds what it adds with access to every past token individually.",
                           title="The last layer is full attention", width=6.2, size=th.SIZE_SMALL)
        first = kit.callout("note", r"A mismatch already at layer 0 is upstream of both mixers: the embedding, or "
                                    r"the first norm.", title="The first three aren't", width=6.2, size=th.SIZE_SMALL)
        notes = VGroup(first, last).arrange(RIGHT, buff=0.35, aligned_edge=UP)
        col = VGroup(rule, strip, key, notes).arrange(DOWN, buff=0.45)
        self.place(col)
        box63 = g.outline(0, 1, 63, 64, color=self.color, buff=0.06)
        box02 = g.outline(0, 1, 0, 3, color=th.INK_2, buff=0.06)
        with self.narrate(
            "Which mixer runs where? "
            "A layer is full attention when its index plus one is divisible by 4. "
            "So the full-attention layers are 3, 7, and so on, up to 63. "
            "That's 16 of them, and the other 48 are linear attention. "
            "The first three layers aren't full attention, so a bug in that path can't show up before layer 3. "
            "A mismatch already at layer 0 is upstream of both mixers: the embedding, or the first norm. "
            "And the last layer is full attention, so it sees every past token individually."
        ) as t:
            self.play(FadeIn(g), FadeIn(nums), run_time=0.7)
            t.wait_until("A layer is full")
            self.play(FadeIn(rule, shift=DOWN * 0.1), run_time=0.7)
            t.wait_until("So the full-attention")
            self.play(LaggedStart(*[g.cell(0, i).square.animate.set_fill(th.tint(self.color, 0.45))
                                    for i in full], lag_ratio=0.12), FadeIn(key[0]), run_time=2.0)
            t.wait_until("That's 16 of them")
            self.play(LaggedStart(*[g.cell(0, i).square.animate.set_fill(th.tint(th.INFO, 0.3))
                                    for i in range(64) if i not in full], lag_ratio=0.02), FadeIn(key[1]),
                      run_time=1.4)
            t.wait_until("The first three layers")
            self.play(Create(box02), FadeIn(first, shift=UP * 0.1), run_time=0.8)
            t.wait_until("And the last layer")
            self.play(Create(box63), FadeIn(last, shift=UP * 0.1), run_time=0.8)

    # ------------------------------------------------------------------
    # The SwiGLU MLP

    def gated_unit(self):
        self.slide("gated-unit", "A gate can switch a channel off", book_section="s-the-swiglu-mlp")
        ffn = kit.formula(r"\operatorname{FFN}(x) = W_{\text{down}}\,\phi\big(W_{\text{up}}\,x\big)",
                          size=th.SIZE_MATH_SMALL)
        glu = kit.formula(r"\operatorname{GLU}(x) =", r"\big(W_{\text{up}}\,x\big)", r"\odot",
                          r"\phi\big(W_{\text{gate}}\,x\big)", size=th.SIZE_MATH_SMALL)
        glu[1].set_color(th.HL_B)
        glu[3].set_color(th.HL_A)
        fs = VGroup(ffn, glu).arrange(RIGHT, buff=1.0)
        up = [0.9, -0.6, 1.2, 0.7, -1.0, 0.5, 1.1, -0.8]
        gate = [1.0, 0.05, 0.9, 0.0, 0.8, 0.02, 1.1, 0.6]
        prod = [u * gv for u, gv in zip(up, gate)]
        sc = 1.0
        pu, pg, pp = vbars(up, th.HL_B, sc), vbars(gate, th.HL_A, sc), vbars(prod, self.color, sc)
        panels = VGroup()
        for bars, name in ((pu, "up projection"), (pg, "gate, after φ"), (pp, "product")):
            base = baseline(bars)
            nm = kit.label(name, size=th.SIZE_LABEL, color=th.INK_3)
            nm.next_to(base, DOWN, buff=1.3)
            panels.add(VGroup(base, bars, nm))
        times = kit.formula(r"\odot", size=th.SIZE_MATH)
        eq = kit.formula(r"=", size=th.SIZE_MATH)
        panels_row = VGroup(panels[0], times, panels[1], eq, panels[2]).arrange(RIGHT, buff=0.55)
        times.set_y(0.0 + panels[0][0].get_y())
        eq.set_y(panels[0][0].get_y())
        for p in panels:
            p[2].set_y(panels[0][2].get_y())
        cap = kit.label("eight channels of one position", size=th.SIZE_LABEL, color=th.INK_3)
        col = VGroup(fs, panels_row, cap).arrange(DOWN, buff=0.6)
        self.place(col)
        off = VGroup(*[pp[k] for k in (1, 3, 5)])
        off_marks = VGroup(*[kit.label("off", size=th.SIZE_LABEL, color=th.BAD).next_to(pp[k], UP, buff=0.12)
                             for k in (1, 3, 5)])
        # Every "off" mark sits at the same height, above the tallest neighbour.
        y_off = panels[2][0].get_y() + 1.45
        for m in off_marks:
            m.set_y(y_off)
        with self.narrate(
            "Now the one piece you haven't built: the MLP. "
            "It moves information between channels, at each position on its own. "
            "The original feed-forward block widens the vector, applies a nonlinearity, and narrows it back. "
            "A gated linear unit replaces that nonlinearity with a product of two projections. "
            "The gate passes through the nonlinearity, and the up projection doesn't. "
            "Here are eight channels of each: multiply them, channel by channel. "
            "Where the gate is near zero, the channel switches off, whatever the up projection carried. "
            "A sum of two linear maps can't do that, but a product can."
        ) as t:
            t.wait_until("The original feed-forward")
            self.play(FadeIn(ffn), run_time=0.8)
            t.wait_until("A gated linear unit")
            self.play(FadeIn(glu), run_time=0.8)
            t.wait_until("Here are eight")
            self.play(FadeIn(panels[0][0]), FadeIn(panels[0][2]), FadeIn(panels[1][0]), FadeIn(panels[1][2]),
                      FadeIn(cap), run_time=0.6)
            self.play(LaggedStart(*[GrowFromEdge(b, DOWN if v >= 0 else UP) for b, v in zip(pu, up)],
                                  lag_ratio=0.08),
                      LaggedStart(*[GrowFromEdge(b, DOWN) for b in pg], lag_ratio=0.08), run_time=1.2)
            self.play(FadeIn(times), FadeIn(eq), FadeIn(panels[2][0]), FadeIn(panels[2][2]), run_time=0.5)
            self.play(TransformFromCopy(pu, pp), run_time=1.2)
            t.wait_until("Where the gate is near")
            self.play(Indicate(VGroup(*[pg[k] for k in (1, 3, 5)]), color=th.BAD, scale_factor=1.3),
                      FadeIn(off_marks), Indicate(off, color=th.BAD, scale_factor=1.3), run_time=1.2)

    def silu(self):
        self.slide("silu", "SiLU is smooth, dips below zero, then saturates", book_section="s-the-swiglu-mlp")
        ax = kit.axes([-5, 4, 1], [-1, 4, 1], width=7.0, height=4.4, x_label="z",
                      x_numbers=[-4, -2, 0, 2, 4], y_numbers=[-1, 0, 1, 2, 3, 4])
        relu = ax.plot(lambda x: max(x, 0.0), x_range=[-5, 4, 0.01], color=th.INK_4, stroke_width=3)
        silu = ax.plot(lambda x: x / (1 + np.exp(-x)), x_range=[-5, 4, 0.02], color=self.color,
                       stroke_width=th.STROKE_CURVE)
        zmin, fmin = -1.278, -0.278
        dot = Dot(ax.c2p(zmin, fmin), radius=0.08, color=th.INK)
        mlab = kit.math_label(r"(-1.278,\ -0.278)", size=th.SIZE_LABEL, color=th.INK)
        mlab.next_to(dot, DOWN + LEFT, buff=0.1)
        k1 = VGroup(Line(LEFT * 0.25, RIGHT * 0.25, color=self.color, stroke_width=th.STROKE_CURVE),
                    kit.label("SiLU", size=th.SIZE_LABEL, color=th.INK_2)).arrange(RIGHT, buff=0.15)
        k2 = VGroup(Line(LEFT * 0.25, RIGHT * 0.25, color=th.INK_4, stroke_width=3),
                    kit.label("ReLU", size=th.SIZE_LABEL, color=th.INK_2)).arrange(RIGHT, buff=0.15)
        keys = VGroup(k1, k2).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        plot = VGroup(ax, relu, silu, dot, mlab)
        keys.next_to(ax.c2p(-5, 4), DR, buff=0.2)
        plot.add(keys)
        f = kit.formula(r"\operatorname{SiLU}(z) = z\,\sigma(z) = \frac{z}{1 + e^{-z}}", size=th.SIZE_MATH_SMALL)
        pts = kit.bullets(
            [
                r"\textbf{Smooth everywhere}: no kink at zero.",
                r"\textbf{Non-monotonic}: a minimum of about $-0.278$ at $z \approx -1.278$.",
                r"\textbf{Saturates}: like ReLU at both extremes.",
            ],
            size=th.SIZE_SMALL, width=4.9, buff=0.3,
        )
        side = VGroup(f, pts).arrange(DOWN, buff=0.55, aligned_edge=LEFT)
        layout = VGroup(plot, side).arrange(RIGHT, buff=0.55)
        self.place(layout)
        with self.narrate(
            "SwiGLU is the gated unit whose nonlinearity is SiLU, also called Swish. "
            "SiLU multiplies its input by the logistic sigmoid. "
            "First, it's smooth everywhere: ReLU has a kink at zero, but SiLU's derivative is continuous. "
            "Second, it isn't monotonic. "
            "It dips to a minimum of about minus 0.278. "
            "So small negative inputs keep a little signal, instead of being erased. "
            "Third, it saturates like ReLU: toward zero on the left, and toward z on the right."
        ) as t:
            self.play(Create(ax), FadeIn(keys), run_time=0.9)
            t.wait_until("SiLU multiplies")
            self.play(FadeIn(f), Create(relu), run_time=0.8)
            self.play(Create(silu), run_time=1.4)
            t.wait_until("First, it's smooth")
            self.play(FadeIn(pts[0], shift=RIGHT * 0.1), Indicate(VGroup(dot.copy().move_to(ax.c2p(0, 0))),
                                                                   color=self.color), run_time=0.8)
            t.wait_until("It dips to")
            self.play(FadeIn(dot, scale=0.5), FadeIn(mlab), FadeIn(pts[1], shift=RIGHT * 0.1), run_time=0.8)
            t.wait_until("Third, it saturates")
            self.play(FadeIn(pts[2], shift=RIGHT * 0.1), run_time=0.7)

    def swiglu_code(self):
        self.slide("swiglu-code", "SwiGLU is three bias-free matrices", book_section="s-the-swiglu-mlp")
        panel = kit.code(BLOCK_SRC, "python", caption="engine/layers/mlp.py")
        self.place(panel)
        with self.narrate(
            "Here's the whole block, and it's barely longer than the maths. "
            "The gate and up projections widen each token's vector to the intermediate size. "
            "The down projection narrows them back. "
            "In the forward pass, the gate goes through SiLU and multiplies the up projection. "
            "The product goes back down to the stream's width. "
            "None of the projections has a bias. "
            "With a norm right upstream, a bias adds parameters and buys nothing measurable."
        ) as t:
            self.play(FadeIn(panel.frame), *panel.reveal(1, 2, 3), run_time=0.8)
            t.wait_until("The gate and up")
            self.play(*panel.reveal(4, 5), panel.highlight(4, 5), run_time=0.8)
            t.wait_until("The down projection")
            self.play(*panel.reveal(6), panel.highlight(6), run_time=0.6)
            t.wait_until("In the forward pass")
            self.play(*panel.reveal(8, 9, 10), run_time=0.8)
            self.play(*panel.reveal(11), panel.highlight(11), run_time=0.6)
            t.wait_until("The product goes")
            self.play(*panel.reveal(12), panel.highlight(12), run_time=0.6)
            t.wait_until("None of the projections")
            self.play(Circumscribe(VGroup(panel.line(4), panel.line(6)), color=self.color, buff=0.08), run_time=1.2)

    def mlp_params(self):
        self.slide("mlp-params", "Why 17,408: the gate gets extra parameters", book_section="s-the-swiglu-mlp")
        bc = kit.bar_compare(
            [
                ("Ungated, i = 4h = 20,480", 209715200, th.INFO, "209,715,200"),
                ("Gated, i = 8h / 3 ≈ 13,653", 209715200, th.INK_4, "209,715,200"),
                ("Gated, i = 17,408", 267386880, self.color, "267,386,880"),
            ],
            width=5.2, bar_height=0.55, gap=0.42,
        )
        f1 = kit.formula(r"2 \cdot h \cdot 4h = 8h^2", size=th.SIZE_MATH_SMALL)
        f2 = kit.formula(r"3 \cdot h \cdot i = 3 \cdot 5120 \cdot 17408", size=th.SIZE_MATH_SMALL)
        more = kit.pill("+27.5%", self.color, size=th.SIZE_SMALL)
        fs = VGroup(f1, f2, more).arrange(RIGHT, buff=0.6)
        total = kit.callout("result", r"$64 \times 267{,}386{,}880 = 17.1$ billion parameters: 64\% of the model.",
                            title="Across 64 layers", width=9.6, size=th.SIZE_SMALL)
        col = VGroup(fs, bc, total).arrange(DOWN, buff=0.6)
        self.place(col)
        with self.narrate(
            "The usual intermediate size is 4 times h, but this model uses 17,408. "
            "Two forces set that number, and the first is the parameter budget. "
            "An ungated block has two matrices: about 210 million parameters. "
            "A gated block has three, so to keep the count equal, i would shrink to two-thirds of 4 h. "
            "The original SwiGLU paper used that rule, and this model ignores it. "
            "Its block holds about 267 million parameters, which is 27.5% more. "
            "MLP weights are read once per forward pass, so they're cheap to serve. "
            "Across 64 layers, the MLP holds 17.1 billion parameters: 64% of the model."
        ) as t:
            self.play(FadeIn(bc.labels), run_time=0.7)
            t.wait_until("An ungated block")
            self.play(GrowFromEdge(bc.bars[0], LEFT), FadeIn(bc.values[0]), FadeIn(f1), run_time=0.9)
            t.wait_until("A gated block has three")
            self.play(GrowFromEdge(bc.bars[1], LEFT), FadeIn(bc.values[1]), run_time=0.9)
            t.wait_until("Its block holds")
            self.play(GrowFromEdge(bc.bars[2], LEFT), FadeIn(bc.values[2]), FadeIn(f2), run_time=0.9)
            self.play(FadeIn(more, scale=0.9), run_time=0.5)
            t.wait_until("Across 64 layers")
            self.play(FadeIn(total, shift=UP * 0.1), run_time=0.8)

    def alignment(self):
        self.slide("alignment", "Why 17,408: it divides cleanly", book_section="s-the-swiglu-mlp")
        g = kit.grid(8, 17, cell=(0.42, 0.42), gap=0.04, row_labels=[f"GPU {k}" for k in range(8)])
        cap = kit.label("each cell: 128 channels", size=th.SIZE_LABEL, color=th.INK_3)
        cap.next_to(g, UP, buff=0.2).align_to(g.cells[0][0], LEFT)
        fig = VGroup(g, cap)
        f = kit.formula(r"17408 = 17 \times 1024", size=th.SIZE_MATH)
        lines = kit.bullets(
            [
                r"Divisible by 128 and 256: tensor-core tile widths.",
                r"8 GPUs: $2176 = 17 \times 128$ each.",
                r"16 GPUs: $1088 = 17 \times 64$ each.",
                r"$13{,}653$: none of these.",
            ],
            size=th.SIZE_SMALL, width=4.8, buff=0.3,
        )
        side = VGroup(f, lines).arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        layout = VGroup(fig, side).arrange(RIGHT, buff=0.6)
        self.place(layout)
        splits = VGroup()
        for r in range(8):
            c = g.cell(r, 8).square
            splits.add(DashedLine(c.get_top() + UP * 0.03, c.get_bottom() + DOWN * 0.03, color=th.INK,
                                  stroke_width=3, dash_length=0.06))
        with self.narrate(
            "The second force is hardware alignment. "
            "17,408 is 17 times 1024. "
            "So it's divisible by the tile widths tensor cores want for a bf16 GEMM. "
            "It also splits evenly across GPUs for tensor parallelism. "
            "Across 8 GPUs, each shard is still a multiple of 128. "
            "Across 16 GPUs, each shard is a multiple of 64. "
            "The two-thirds answer has none of these properties."
        ) as t:
            t.wait_until("17,408 is 17")
            self.play(FadeIn(f), run_time=0.7)
            t.wait_until("So it's divisible")
            self.play(FadeIn(lines[0], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("It also splits")
            self.play(FadeIn(g.row_labels), FadeIn(cap), run_time=0.5)
            t.wait_until("Across 8 GPUs")
            self.play(LaggedStart(*[FadeIn(g.row(r)) for r in range(8)], lag_ratio=0.15), run_time=1.4)
            self.play(LaggedStart(*[g.fill(g.row(r), self.colors[r % 4]) for r in range(8)], lag_ratio=0.1),
                      FadeIn(lines[1], shift=RIGHT * 0.1), run_time=1.0)
            t.wait_until("Across 16 GPUs")
            self.play(LaggedStart(*[Create(s) for s in splits], lag_ratio=0.1), FadeIn(lines[2], shift=RIGHT * 0.1),
                      run_time=1.0)
            t.wait_until("The two-thirds answer")
            self.play(FadeIn(lines[3], shift=RIGHT * 0.1), run_time=0.6)

    # ------------------------------------------------------------------
    # The output head

    def output_head(self):
        self.slide("output-head", "A final norm, then one large projection", book_section="s-the-output-head")
        panel = kit.code(HEAD_SRC, "python")
        emb = kit.box("Embedding", sublabel="248,320 × 5120", width=3.6, color=th.INFO)
        head = kit.box("lm_head", sublabel="5120 → 248,320", width=3.6, color=self.color, font=th.MONO)
        plus = kit.formula("+", size=th.SIZE_MATH)
        mats = VGroup(emb, plus, head).arrange(RIGHT, buff=0.4)
        each = kit.label("1,271,398,400 parameters each · 2.54 GB in bf16", size=th.SIZE_SMALL, color=th.INK_2)
        tot = kit.prose(r"Not tied: \textbf{2.54 billion} parameters, 9.5\% of the model, before any layer runs.",
                        size=th.SIZE_SMALL, width=None)
        col = VGroup(panel, mats, each, tot).arrange(DOWN, buff=0.4)
        self.place(col)
        with self.narrate(
            "After layer 63, the stream gets one more normalization. "
            "Sixty-four residual additions can leave it at any magnitude, and the output projection was trained "
            "on a normalized input. "
            "Then keep only the last position, and project. "
            "The output head maps each vector to 248,320 vocabulary scores. "
            "That's about 1.27 billion parameters, or 2.54 GB in bf16. "
            "The embedding table isn't tied to it, so it's a second matrix of the same size. "
            "Together, they're 9.5% of the model."
        ) as t:
            self.play(FadeIn(panel.frame), *panel.reveal(1), run_time=0.8)
            t.wait_until("Sixty-four residual")
            self.play(panel.highlight(1), run_time=0.6)
            t.wait_until("Then keep only")
            self.play(*panel.reveal(2, 3), run_time=0.7)
            self.play(*panel.reveal(4), run_time=0.6)
            t.wait_until("The output head maps")
            self.play(FadeIn(head, shift=UP * 0.1), panel.highlight(4), run_time=0.8)
            t.wait_until("That's about 1.27")
            self.play(FadeIn(each), run_time=0.6)
            t.wait_until("The embedding table isn't")
            self.play(FadeIn(emb, shift=UP * 0.1), FadeIn(plus), run_time=0.8)
            t.wait_until("Together, they")
            self.play(FadeIn(tot), run_time=0.7)

    def slice_first(self):
        self.slide("slice-first", "Slice to the last position before you project",
                   book_section="s-the-output-head")
        rows_n = 12
        g = kit.grid(rows_n, 1, cell=(5.6, 0.26), gap=0.05)
        b_left = Brace(g, LEFT, buff=0.12, color=th.INK_4)
        t_left = kit.label("2048 positions", size=th.SIZE_LABEL, color=th.INK_3)
        t_left.rotate(PI / 2).next_to(b_left, LEFT, buff=0.1)
        b_top = Brace(g, UP, buff=0.12, color=th.INK_4)
        t_top = kit.label("248,320 logits per position", size=th.SIZE_LABEL, color=th.INK_3)
        t_top.next_to(b_top, UP, buff=0.08)
        last_t = kit.label("the one row you need", size=th.SIZE_LABEL, color=self.color, weight=MEDIUM)
        fig = VGroup(g, b_left, t_left, b_top, t_top)
        tab = kit.table(
            [
                ["FLOPs", r"$5.21 \times 10^{12}$", "2.54 G"],
                ["Output", "1.02 GB", "497 KB"],
            ],
            header=["", "All 2048", "Last only"],
            align=["l", "r", "r"],
            size=th.SIZE_SMALL,
        )
        floor = kit.prose(r"At 312 TFLOP/s, projecting all 2048 takes at least 16.7 ms (a roofline floor).",
                          size=th.SIZE_SMALL, width=5.0, color=th.INK_2)
        side = VGroup(tab, floor).arrange(DOWN, buff=0.55, aligned_edge=LEFT)
        layout = VGroup(fig, side).arrange(RIGHT, buff=0.7)
        self.place(layout)
        last_t.next_to(g.cell(rows_n - 1, 0), DOWN, buff=0.15)
        with self.narrate(
            "That slice isn't a micro-optimization. "
            "In a 2048-token prefill, you need only the last position's row. "
            "Project every position, and you do 2048 times the arithmetic you need. "
            "Project only the last one, and it costs 2.54 GFLOPs. "
            "The output tensor drops from 1.02 GB to 497 KB. "
            "At the A100's peak of 312 TFLOP/s, the full projection takes at least 16.7 ms, as a roofline floor."
        ) as t:
            self.play(FadeIn(g), FadeIn(b_left), FadeIn(t_left), FadeIn(b_top), FadeIn(t_top), run_time=0.8)
            t.wait_until("In a 2048-token")
            self.play(Indicate(g.cell(rows_n - 1, 0), color=self.color, scale_factor=1.04), run_time=0.8)
            t.wait_until("Project every position")
            self.play(LaggedStart(*[g.fill(g.cell(r, 0), th.INFO) for r in range(rows_n)], lag_ratio=0.1),
                      FadeIn(tab.rules), FadeIn(VGroup(*tab.cells[0])), FadeIn(tab.cells[1][0]),
                      FadeIn(tab.cells[1][1]), run_time=1.4)
            t.wait_until("Project only the last")
            self.play(g.fill(g.block(0, rows_n - 1, 0, 1), th.INK_4, tint=0.08),
                      g.fill(g.cell(rows_n - 1, 0), self.color, tint=0.5), FadeIn(last_t),
                      FadeIn(tab.cells[1][2]), run_time=1.0)
            t.wait_until("The output tensor drops")
            self.play(FadeIn(VGroup(*tab.cells[2])), run_time=0.7)
            t.wait_until("At the A100's peak")
            self.play(FadeIn(floor), run_time=0.7)

    def logits(self):
        self.slide("logits", "Only differences between logits carry information",
                   book_section="s-the-output-head")
        z = [24, 22.5, 21, 18, 14, 11]
        p = [0.784, 0.175, 0.039, 0.0019, 0.0, 0.0]
        zs = 0.1
        zb = vbars(z, self.color, zs, width=0.42, gap=0.32)
        zb_shift = vbars([v + 4 for v in z], self.color, zs, width=0.42, gap=0.32)
        zb_max = vbars([v - 24 for v in z], self.color, zs, width=0.42, gap=0.32)
        pb = vbars(p, th.INFO, 3.0, width=0.42, gap=0.32)
        zbase, pbase = baseline(zb), baseline(pb)
        zname = kit.label("logits z", size=th.SIZE_SMALL, color=th.INK_2)
        zname.next_to(zbase, DOWN, buff=1.6)
        pname = kit.label("probabilities p", size=th.SIZE_SMALL, color=th.INK_2)
        zl = VGroup(zbase, zb, zb_shift, zb_max, zname)
        pl = VGroup(pbase, pb, pname)
        pname.next_to(pbase, DOWN, buff=1.6)
        zvals = VGroup(*[kit.label(f"{v:g}", size=th.SIZE_LABEL, color=th.INK_3).next_to(b, UP, buff=0.08)
                         for v, b in zip(z, zb)])
        pvals = VGroup(*[kit.label(f"{v:.2f}" if v >= 0.01 else "≈0", size=th.SIZE_LABEL, color=th.INK_3)
                         .next_to(b, UP, buff=0.08) for v, b in zip(p, pb)])
        zl.add(zvals)
        pl.add(pvals)
        arrow_ = Arrow(LEFT * 0.6, RIGHT * 0.6, color=th.INK_3, stroke_width=4, tip_length=0.2)
        soft = kit.label("softmax", size=th.SIZE_LABEL, color=th.INK_3).next_to(arrow_, UP, buff=0.1)
        mid = VGroup(arrow_, soft)
        charts = VGroup(zl, mid, pl).arrange(RIGHT, buff=0.8)
        pl.shift(UP * (zbase.get_y() - pbase.get_y()))
        mid.set_y(zbase.get_y() + 0.8)
        f = kit.formula(r"p_j = \frac{e^{z_j}}{\sum_{m=1}^{V} e^{z_m}}", r"\qquad", r"z \to z + c\mathbf{1}:\ "
                        r"p \text{ unchanged}", size=th.SIZE_MATH_SMALL)
        f[2].set_color(self.color)
        cap = kit.label("six example logits; trained logits typically land between 10 and 30",
                        size=th.SIZE_LABEL, color=th.INK_3)
        col = VGroup(f, charts, cap).arrange(DOWN, buff=0.5)
        self.place(col)
        for m in (zb_shift, zb_max):
            m.set_opacity(0)
        # The shifted bar sets are targets only; keep them out of the scene.
        zl.remove(zb_shift, zb_max)
        plus4 = kit.label("+ 4", size=th.SIZE_SMALL, color=self.color, weight=MEDIUM)
        plus4.next_to(zname, RIGHT, buff=0.35)
        minus = kit.label("− max", size=th.SIZE_SMALL, color=self.color, weight=MEDIUM)
        minus.move_to(plus4)
        with self.narrate(
            "Each output entry is a logit: an unnormalized log-probability, one raw score per vocabulary token. "
            "Exponentiate and normalize, and you get the distribution over the next token. "
            "Softmax is shift-invariant. "
            "Add the same constant to every logit, and the probabilities don't change. "
            "That's why every softmax subtracts the maximum first, which stops the exponentials from overflowing. "
            "So only differences carry information. "
            "Trained logits typically land between about 10 and 30, but one logit's size means nothing alone."
        ) as t:
            self.play(FadeIn(zbase), FadeIn(zname), LaggedStart(*[GrowFromEdge(b, DOWN) for b in zb],
                                                                lag_ratio=0.1), FadeIn(zvals), run_time=1.2)
            t.wait_until("Exponentiate and normalize")
            self.play(FadeIn(f[0]), GrowArrow(arrow_), FadeIn(soft), run_time=0.8)
            self.play(FadeIn(pbase), FadeIn(pname), LaggedStart(*[GrowFromEdge(b, DOWN) for b in pb],
                                                                lag_ratio=0.1), FadeIn(pvals), run_time=1.2)
            t.wait_until("Add the same constant")
            zb_shift.set_opacity(1)
            self.play(Transform(zb, zb_shift), FadeOut(zvals), FadeIn(plus4), run_time=1.0)
            self.play(FadeIn(f[2]), Indicate(pb, color=th.INFO, scale_factor=1.05), run_time=0.9)
            t.wait_until("That's why every softmax")
            zb_max.set_opacity(1)
            self.play(Transform(zb, zb_max), FadeOut(plus4), FadeIn(minus), run_time=1.0)
            t.wait_until("Trained logits")
            self.play(FadeIn(cap), run_time=0.6)

    # ------------------------------------------------------------------
    # Loading weights

    def loading(self):
        self.slide("loading", "Three checks catch nearly every loading bug", book_section="s-loading-weights")
        checks = kit.bullets(
            [
                r"\textbf{Every checkpoint tensor is consumed.} A leftover means a skipped component.",
                r"\textbf{Every parameter is written.} Fill with NaN first; a NaN left over is a gap.",
                r"\textbf{Shapes match exactly.} A weight is stored (out, in).",
            ],
            size=th.SIZE_SMALL, width=6.4, buff=0.4,
        )
        g = kit.grid(4, 5, cell=(0.85, 0.55), gap=0.06, values=[["NaN"] * 5 for _ in range(4)])
        for row in g.cells:
            for c in row:
                c.square.set_fill(th.tint(th.BAD, 0.25), opacity=1)
        gcap = kit.label("parameters, before and after loading", size=th.SIZE_LABEL, color=th.INK_3)
        gcap.next_to(g, UP, buff=0.2)
        fig = VGroup(g, gcap)
        top = VGroup(checks, fig).arrange(RIGHT, buff=0.7)
        warn = kit.callout("warning", r"A parameter left at its random value: slightly worse output. "
                                      r"A transposed square matrix: loads fine, computes something else.",
                           title="Two bugs that don't raise", width=11.5, size=th.SIZE_SMALL)
        col = VGroup(top, warn).arrange(DOWN, buff=0.55)
        self.place(col)
        miss = (2, 3)
        loaded = [(r, c) for r in range(4) for c in range(5) if (r, c) != miss]
        gap_box = g.outline(miss[0], miss[0] + 1, miss[1], miss[1] + 1, color=th.BAD, buff=0.05)
        gap_t = kit.label("never written", size=th.SIZE_LABEL, color=th.BAD, weight=MEDIUM)
        gap_t.next_to(g, DOWN, buff=0.15).align_to(g.cell(*miss), RIGHT)
        with self.narrate(
            "Now load real weights. "
            "The checkpoint's names don't match your modules, so write an explicit mapping, which fails loudly "
            "when a name is missing. "
            "Three checks catch nearly every loading bug. "
            "First, every checkpoint tensor is consumed. "
            "Second, every parameter is written. "
            "Fill parameters with a not-a-number value before loading, and any left afterward points at the gap. "
            "Third, shapes match exactly. "
            "Two bugs don't raise at all: a parameter left at its random value, and a transposed square matrix."
        ) as t:
            t.wait_until("First, every checkpoint")
            self.play(FadeIn(checks[0], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("Second, every parameter")
            self.play(FadeIn(checks[1], shift=RIGHT * 0.1), FadeIn(g), FadeIn(gcap), run_time=0.8)
            t.wait_until("Fill parameters")
            anims = []
            for r, c in loaded:
                cell_ = g.cell(r, c)
                anims.append(AnimationGroup(cell_.square.animate.set_fill(th.tint(th.GOOD, 0.3)),
                                            FadeOut(cell_.text)))
            self.play(LaggedStart(*anims, lag_ratio=0.08), run_time=2.0)
            self.play(Create(gap_box), FadeIn(gap_t), run_time=0.7)
            t.wait_until("Third, shapes match")
            self.play(FadeIn(checks[2], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("Two bugs don't")
            self.play(FadeIn(warn, shift=UP * 0.1), run_time=0.8)

    # ------------------------------------------------------------------
    # Validating against the reference

    def compare_logits(self):
        self.slide("compare-logits", "Compare logits, not generated text",
                   book_section="s-validating-against-the-reference")
        ref = [0.55, 0.2, 0.12, 0.07, 0.04, 0.02]
        yours = [0.42, 0.06, 0.25, 0.03, 0.16, 0.08]
        sc = 3.0
        rb = vbars(ref, th.HL_C, sc, width=0.3, gap=0.32)
        yb = vbars(yours, th.HL_D, sc, width=0.3, gap=0.32).shift(RIGHT * 0.3)
        base = baseline(VGroup(rb, yb))
        key = VGroup(swatch("reference p", th.HL_C), swatch("yours q", th.HL_D)).arrange(RIGHT, buff=0.4)
        same = kit.label("same argmax, different distribution", size=th.SIZE_LABEL, color=th.INK_3)
        fig = VGroup(key, VGroup(base, rb, yb), same).arrange(DOWN, buff=0.3)
        tab = kit.table(
            [
                ["Max abs difference", r"$\sim 10^{-2}$", "least stable"],
                ["Correlation", r"$> 0.9999$", "most sensitive"],
                ["Top-1 agreement", "1.0", "coarsest"],
            ],
            header=["In float32", "Expect", ""],
            size=th.SIZE_SMALL,
        )
        kl = kit.formula(r"D_{\mathrm{KL}}(", r"p", r"\,\|\,", r"q", r")", r"= \sum_{j=1}^{V}", r"p_j",
                         r"\log \frac{", r"p_j", r"}{", r"q_j", r"}", size=th.SIZE_MATH_SMALL)
        for k in (1, 6, 8):
            kl[k].set_color(th.HL_C)
        for k in (3, 10):
            kl[k].set_color(th.HL_D)
        side = VGroup(tab, kl).arrange(DOWN, buff=0.55)
        layout = VGroup(fig, side).arrange(RIGHT, buff=0.8)
        self.place(layout)
        with self.narrate(
            "To check the model, load the same weights into transformers, and run both on the same token ids. "
            "Compare logits, not generated text. "
            "Greedy decoding keeps only the top of the ranking. "
            "Two models can produce the same 200 tokens while disagreeing badly about the distribution underneath. "
            "So measure four things, in float32. "
            "The max absolute difference should be around 10 to the minus 2. "
            "Correlation should be above 0.9999. "
            "Top-1 agreement should be 1. "
            "KL divergence weights each disagreement by the reference probability. "
            "Judge by correlation and KL."
        ) as t:
            self.play(FadeIn(base), FadeIn(key), LaggedStart(*[GrowFromEdge(b, DOWN) for b in rb], lag_ratio=0.1),
                      run_time=1.0)
            t.wait_until("Greedy decoding")
            self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in yb], lag_ratio=0.1), run_time=1.0)
            t.wait_until("Two models can")
            self.play(FadeIn(same), Indicate(VGroup(rb[0], yb[0]), color=self.color), run_time=0.9)
            t.wait_until("So measure four")
            self.play(FadeIn(tab.rules), FadeIn(VGroup(*tab.cells[0])), run_time=0.6)
            for r, cue in zip((1, 2, 3), ("The max absolute", "Correlation should", "Top-1 agreement")):
                t.wait_until(cue)
                self.play(FadeIn(VGroup(*tab.cells[r]), shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("KL divergence weights")
            self.play(FadeIn(kl), run_time=0.8)
            t.wait_until("Judge by")
            self.play(Indicate(VGroup(*tab.cells[2]), color=self.color, scale_factor=1.05),
                      Circumscribe(kl, color=self.color, buff=0.1), run_time=1.2)

    def bit_exact(self):
        self.slide("bit-exact", "Bit-exact agreement is impossible",
                   book_section="s-validating-against-the-reference")
        assoc = kit.formula(r"(a + b) + c \;\ne\; a + (b + c)", size=th.SIZE_MATH)

        def tree(pairs_first, color):
            leaves = VGroup(*[Dot(radius=0.07, color=th.INK_3) for _ in range(8)]).arrange(RIGHT, buff=0.42)
            lines = VGroup()
            nodes = VGroup()

            def node(a, b, y):
                p = np.array([(a[0] + b[0]) / 2, y, 0])
                n = plus_node(r=0.13, color=color).move_to(p)
                lines.add(Line(a, p, color=th.INK_4, stroke_width=2), Line(b, p, color=th.INK_4, stroke_width=2))
                nodes.add(n)
                return p

            pts = [l.get_center() for l in leaves]
            levels = []
            if pairs_first:
                l1 = [node(pts[2 * k], pts[2 * k + 1], 0.6) for k in range(4)]
                l2 = [node(l1[0], l1[1], 1.2), node(l1[2], l1[3], 1.2)]
                root = node(l2[0], l2[1], 1.8)
                levels = [4, 2, 1]
            else:
                halves = []
                for h0 in (0, 4):
                    acc = pts[h0]
                    for j in range(1, 4):
                        acc = node(acc, pts[h0 + j], 0.45 * j)
                    halves.append(acc)
                root = node(halves[0], halves[1], 1.8)
                levels = [6, 1]
            res = kit.label("one sum", size=th.SIZE_LABEL, color=color).next_to(root, UP, buff=0.25)
            g = VGroup(lines, leaves, nodes, res)
            g.levels, g.res, g.lines, g.nodes, g.leaves = levels, res, lines, nodes, leaves
            return g

        ta = tree(True, th.INFO)
        tb = tree(False, self.color)
        ta.res.become(kit.label("your kernel", size=th.SIZE_LABEL, color=th.INFO).move_to(ta.res))
        tb.res.become(kit.label("cuBLAS: split-K", size=th.SIZE_LABEL, color=self.color).move_to(tb.res))
        ne = kit.formula(r"\ne", size=th.SIZE_MATH)
        trees = VGroup(ta, ne, tb).arrange(RIGHT, buff=0.8)
        ne.set_y(ta.nodes[-1].get_y())
        prods = kit.label("8 of the 5120 products in one dot product", size=th.SIZE_LABEL, color=th.INK_3)
        eps = kit.formula(r"\text{bf16: } \varepsilon = 2^{-8} \approx 3.9 \times 10^{-3}", size=th.SIZE_MATH_SMALL)
        key = kit.callout("key", r"Assert a tolerance chosen from the dtype, never equality.", width=7.0,
                          size=th.SIZE_SMALL)
        bottom = VGroup(eps, key).arrange(RIGHT, buff=0.7)
        col = VGroup(assoc, VGroup(trees, prods).arrange(DOWN, buff=0.25), bottom).arrange(DOWN, buff=0.5)
        self.place(col)
        with self.narrate(
            "Why can't the two agree exactly? "
            "First, floating-point addition isn't associative, because each addition rounds. "
            "Each output of a matmul is a sum of 5120 products. "
            "The order a kernel adds them depends on its tiles, its split-K strategy, and its warps. "
            "Your kernel and cuBLAS choose differently, so they land on different sums of the same numbers. "
            "Second, bf16 keeps 8 bits of significand, a relative spacing of about 0.0039. "
            "Every matmul rounds to that precision, and both roundings are equally valid. "
            "So assert a tolerance chosen from the dtype, never equality."
        ) as t:
            t.wait_until("First, floating-point")
            self.play(FadeIn(assoc), run_time=0.8)
            t.wait_until("Each output of a matmul")
            self.play(FadeIn(ta.leaves), FadeIn(tb.leaves), FadeIn(prods), run_time=0.7)
            t.wait_until("The order a kernel")
            self.play(Create(ta.lines), FadeIn(ta.nodes), FadeIn(ta.res), run_time=1.4)
            t.wait_until("Your kernel and cuBLAS")
            self.play(Create(tb.lines), FadeIn(tb.nodes), FadeIn(tb.res), run_time=1.4)
            self.play(FadeIn(ne, scale=0.8), run_time=0.5)
            t.wait_until("Second, bf16 keeps")
            self.play(FadeIn(eps), run_time=0.7)
            t.wait_until("So assert a tolerance")
            self.play(FadeIn(key, shift=UP * 0.1), run_time=0.7)

    # ------------------------------------------------------------------
    # Bisecting a mismatch

    def bisect(self):
        self.slide("bisect", "Find the first layer where the streams diverge",
                   book_section="s-bisecting-a-mismatch")
        r = kit.rng(19)
        n = 24
        diffs = []
        for l in range(n):
            if l < 3:
                diffs.append(float(r.uniform(0.002, 0.006)))
            else:
                diffs.append(0.014 + 0.0016 * (l - 3) + float(r.uniform(0.0, 0.003)))
        ymax = 0.05
        ax = kit.axes([0, n, 4], [0, ymax, 0.01], width=8.5, height=2.0, x_label=r"\text{layer}",
                      y_label=r"\text{max }|\Delta|", x_numbers=[0, 4, 8, 12, 16, 20])
        bars = VGroup()
        bw = ax.c2p(1, 0)[0] - ax.c2p(0, 0)[0]
        for l, d in enumerate(diffs):
            p0 = ax.c2p(l + 0.5, 0)
            hgt = ax.c2p(0, d)[1] - ax.c2p(0, 0)[1]
            rect = Rectangle(width=bw * 0.7, height=hgt, stroke_width=th.STROKE_HAIRLINE, stroke_color=th.INK_4,
                             fill_color=th.tint(th.INK_4, 0.35), fill_opacity=1)
            rect.move_to(p0 + UP * hgt / 2)
            bars.add(rect)
        tol = DashedLine(ax.c2p(0, 0.01), ax.c2p(n, 0.01), color=th.BAD, stroke_width=th.STROKE_AXIS,
                         dash_length=0.08)
        tol_t = kit.math_label(r"\text{tol} = 10^{-2}", size=th.SIZE_LABEL, color=th.BAD)
        tol_t.next_to(tol, RIGHT, buff=0.12)
        plot = VGroup(ax, bars, tol, tol_t)
        tab = kit.table(
            [
                ["Layer 0, immediately", "Embedding, or a transposed weight"],
                ["Layer 3, first full attention", "Output gate split, or mask offset"],
                ["Layer 0, past a fixed position", "Partial rotary boundary"],
                ["Gradual drift across all", "A norm reducing in bf16"],
                ["Right for one token, then wrong", "Cache indexing, decode position"],
            ],
            header=["First divergence", "Likely cause"],
            size=th.SIZE_LABEL,
            row_gap=0.17,
            col_gap=1.0,
        )
        layout = VGroup(plot, tab).arrange(DOWN, buff=0.45)
        self.place(layout)
        cursor = Triangle(color=th.INK, fill_color=th.INK, fill_opacity=1, stroke_width=0).scale(0.1).rotate(PI)
        cursor.next_to(bars[0], UP, buff=0.1)
        stop_t = kit.label("stops at layer 3", size=th.SIZE_LABEL, color=self.color, weight=MEDIUM)
        stop_t.next_to(bars[3], UP, buff=0.45).shift(RIGHT * 0.9)
        band = kit.mark(VGroup(*tab.cells[2]), buff=0.08)
        with self.narrate(
            "When the logits disagree, don't stare at the logits. "
            "Hook every decoder layer in both models, run one prompt, and collect each layer's output. "
            "Then walk the two lists, and return the first layer whose difference crosses a tolerance. "
            "Everything after it is downstream of one bug. "
            "On a stream of order 1, a tolerance of 10 to the minus 2 is about three times bf16's spacing. "
            "Here, the walk stops at layer 3, the first full-attention layer. "
            "That points at the output gate split, or the mask offset."
        ) as t:
            self.play(Create(ax), run_time=0.8)
            t.wait_until("Hook every decoder")
            self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.05), run_time=1.6)
            t.wait_until("Then walk the two")
            self.play(FadeIn(tol), FadeIn(tol_t), FadeIn(cursor), run_time=0.6)
            for l in range(1, 4):
                self.play(cursor.animate.next_to(bars[l], UP, buff=0.1), run_time=0.4)
            self.play(bars[3].animate.set_fill(th.tint(self.color, 0.6)).set_stroke(self.color), run_time=0.5)
            t.wait_until("Everything after it")
            self.play(*[b.animate.set_fill(th.tint(th.BAD, 0.25)) for b in bars[4:]], run_time=0.8)
            t.wait_until("Here, the walk stops")
            self.play(FadeIn(stop_t), FadeIn(tab.rules), *[FadeIn(c) for row in tab.cells for c in row],
                      run_time=1.0)
            t.wait_until("That points at")
            self.play(FadeIn(band), run_time=0.6)

    # ------------------------------------------------------------------
    # Prefill and decode must agree

    def prefill_decode(self):
        self.slide("prefill-decode", "Prefill and decode must give the same numbers",
                   book_section="s-prefill-and-decode-must-agree")
        n, pre = 8, 4
        a = kit.grid(1, n, cell=(1.0, 0.65), gap=0.1, values=[[f"z{k}" for k in range(n)]])
        b = kit.grid(1, n, cell=(1.0, 0.65), gap=0.1, values=[[f"z{k}" if k >= pre - 1 else "" for k in range(n)]])
        la = kit.label("full forward, no cache", size=th.SIZE_SMALL, color=th.INK_2)
        lb = kit.label("prefill 4, then one token at a time", size=th.SIZE_SMALL, color=th.INK_2)
        ra = VGroup(la, a).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        rb = VGroup(lb, b).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        grids = VGroup(ra, rb).arrange(DOWN, buff=1.0, aligned_edge=LEFT)
        for k in range(n):
            b.cell(0, k).text and b.cell(0, k).text.set_opacity(0)
        res = kit.pill("max error 5.96e-07  ·  float32, small config", self.color, size=th.SIZE_SMALL)
        key = kit.callout("key", r"Covers the KV append, state carry, conv window, mask offset, and decode position.",
                          title="The highest-value test in the engine", width=12.0, size=th.SIZE_SMALL)
        col = VGroup(grids, res, key).arrange(DOWN, buff=0.55)
        self.place(col)
        eqs = VGroup(*[kit.formula("=", size=th.SIZE_MATH_SMALL, color=th.INK_3).move_to(
            [a.cell(0, k).get_x(), (a.get_bottom()[1] + lb.get_top()[1]) / 2, 0]) for k in range(pre - 1, n)])
        brace = Brace(b.block(0, 1, 0, pre), DOWN, buff=0.08, color=th.INK_4)
        with self.narrate(
            "One question remains: does your model give the same answer however you feed it the tokens? "
            "Prefill and decode run the same code with different shapes, so they must agree. "
            "First, run a full forward pass with no cache. "
            "Then prefill a prefix into a fresh cache, and feed the remaining tokens one at a time. "
            "Compare the logits at matching positions. "
            "This one test exercises the cache, the state carry, the convolution window, and the mask offset. "
            "The reference implementation agrees to 5.96e-07, in float32. "
            "One implementation must agree with itself to near machine precision."
        ) as t:
            t.wait_until("First, run a full")
            self.play(FadeIn(la), FadeIn(a.squares()), run_time=0.6)
            self.play(LaggedStart(*[FadeIn(a.cell(0, k).text) for k in range(n)], lag_ratio=0.05),
                      a.fill(a.row(0), th.INFO), run_time=0.9)
            t.wait_until("Then prefill a prefix")
            self.play(FadeIn(lb), FadeIn(b.squares()), run_time=0.6)
            self.play(b.fill(b.block(0, 1, 0, pre), self.color, tint=0.15), FadeIn(brace), run_time=0.6)
            b.cell(0, pre - 1).text.set_opacity(1)
            self.play(FadeIn(b.cell(0, pre - 1).text), b.fill(b.cell(0, pre - 1), self.color), run_time=0.6)
            for k in range(pre, n):
                b.cell(0, k).text.set_opacity(1)
                self.play(FadeIn(b.cell(0, k).text), b.fill(b.cell(0, k), self.color), run_time=0.45)
            t.wait_until("Compare the logits")
            self.play(LaggedStart(*[FadeIn(e) for e in eqs], lag_ratio=0.15), run_time=1.0)
            t.wait_until("This one test")
            self.play(FadeIn(key, shift=UP * 0.1), run_time=0.8)
            t.wait_until("The reference implementation")
            self.play(FadeIn(res, scale=0.95), run_time=0.7)

    # ------------------------------------------------------------------
    # What goes wrong

    def what_goes_wrong(self):
        self.slide("what-goes-wrong", "The dangerous bugs still generate English", book_section="s-what-goes-wrong")
        w = 6.25
        cards = VGroup(
            kit.callout("warning", r"Fluent and confident, but unrelated to the prompt. A square matrix doesn't "
                                   r"raise.", title="A transposed weight", width=w, size=th.SIZE_SMALL),
            kit.callout("warning", r"Rotates by head index, not position: no sense of order, worse with longer "
                                   r"prompts.", title="RoPE on the wrong axis", width=w, size=th.SIZE_SMALL),
            kit.callout("warning", r"Right during prefill, wrong on every decode step: drifts within a few dozen "
                                   r"tokens.", title="A mask off by one", width=w, size=th.SIZE_SMALL),
            kit.callout("warning", r"Still runs, but a different model: divergence grows steadily from layer 0 to "
                                   r"63.", title="A norm on the wrong tensor", width=w, size=th.SIZE_SMALL),
        )
        top = VGroup(cards[0], cards[1]).arrange(RIGHT, buff=0.4, aligned_edge=UP)
        bot = VGroup(cards[2], cards[3]).arrange(RIGHT, buff=0.4, aligned_edge=UP)
        grid_ = VGroup(top, bot).arrange(DOWN, buff=0.4)
        self.place(grid_)
        with self.narrate(
            "The dangerous failures still generate English. "
            "A transposed weight gives a model that's fluent and confident, but unrelated to the prompt. "
            "RoPE on the wrong axis rotates by head index instead of position, so the model loses its sense of "
            "order. "
            "A mask off by one is right during prefill and wrong on every decode step, so the model drifts within "
            "a few dozen tokens. "
            "A norm on the wrong tensor still runs, but it's a different model, and the divergence grows with "
            "depth."
        ) as t:
            for card, cue in zip(cards, ["A transposed weight gives", "RoPE on the wrong", "A mask off by",
                                         "A norm on the wrong"]):
                t.wait_until(cue)
                self.play(FadeIn(card, shift=UP * 0.1), run_time=0.7)
