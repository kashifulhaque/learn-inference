"""Chapter 05, rotary position embeddings.

Teaches content/chapters/05-rotary-embeddings.md in its order. Every number
below was recomputed with Python and numpy; each matches the chapter.

Frequencies (rotary dimension d = 64, so 32 pairs, theta_i = Theta^(-i/32)):
- theta_0 = 1 for every Theta; lambda_0 = 2 pi = 6.283, "6.3 positions".
- Theta = 1e7 (the lab's value): lambda_8 = 353.3, lambda_16 = 19,869
  ("about 19,900"), lambda_31 = 2 pi x 1e7^(31/32) = 37,969,062 ("38 million").
- Dial slide, 100 positions at Theta = 1e7: pair 0 turns 100 rad = 15.9 turns
  (ends at 329.6 deg); pair 8 101.9 deg; pair 16 1.81 deg; pair 31
  0.00095 deg.

Slowest pair, i = 31, over 262,144 positions:
- Theta = 1e4: Theta^(31/32) = 7,498.9 (7.50e3); lambda_31 = 47,117
  (4.71e4, "about 47,000"); 262,144 / 47,117 = 5.56 turns ("5.6", "more than
  five"). Positions 1,000 and 48,000 differ by 47,000 = 0.9975 turns, so they
  land 0.9 deg apart. (Chapter question: 200,000 / 47,117 = 4.24 turns.)
- Theta = 1e7: Theta^(31/32) = 6.04e6; lambda_31 = 3.80e7; 262,144 / 3.80e7
  = 0.0069 turns = 2.49 deg ("about 2.5 degrees", "less than 1%").
- Wavelength plot: log10(lambda_i) = log10(2 pi) + i log10(Theta) / 32, a
  straight line from 0.798 to 4.67 (Theta = 1e4) or 7.58 (Theta = 1e7);
  context line at log10(262,144) = 5.42.

Two-dimensional score demo (my own example vectors; theta = 0.3 rad):
- q = (1, 0.4), k = (0.6, 0.9). Score <R(m theta) q, R(n theta) k> for
  (m, n) = (10, 5), (20, 15), (100, 95), (300, 295) is 0.72625 every time
  ("0.726"); at (300, 290), gap 10, it is -0.85725 ("-0.857"), the same as
  (10, 0). The pairs (10, 5) ... (300, 295) are the lab's gap-5 pairs.
- Clock slide: 8 deg per step (illustrative). Query starts at 20 deg, key at
  50 deg; at positions 10 and 15 they sit at 100 and 170 deg, 70 deg apart;
  at 1010 and 1015 they sit at 8100 and 8170 deg = 180 and 250 deg mod 360,
  still 70 deg apart.

Tables:
- 32,768 x 32 x 4 B = 4,194,304 B = 4.19 MB per table; both 8,388,608 B =
  8.4 MB. Weights 53.8 GB (chapter value).
- Table preview: cos(p theta_i) for p = 0..7 and i = 0, 4, ..., 28, Theta =
  1e7, computed in the scene.
- bfloat16 near 100,000: 2^16 <= 100,000 < 2^17 and bf16 keeps 7 explicit
  mantissa bits, so the spacing is 2^(16 - 7) = 512. Round-to-nearest-even of
  99,600 ... 100,700 lands on 99,840, 100,352, or 100,864 (checked by bit
  rounding of the float32 pattern).

Partial rotary:
- rotary_dim = 256 x 0.25 = 64; passthrough 256 - 64 = 192; 32 pairs.
- Heads: 24 query + 4 KV = 28.
- Saving: 3 FLOPs x (256 - 64) channels x 28 heads = 16,128 ("about
  16,000") FLOPs per token per full-attention layer, against 2.54 GFLOP per
  token for lm_head (chapter value).
- Decay plot (illustration of the chapter's "the position-dependent term
  decays as the gap grows"): q and k equal in every pair, Theta = 1e7, the
  rotated score per pair is mean_i cos(delta theta_i), averaged over gaps in
  [0.9 delta, 1.1 delta]. It falls from 1.0 at delta = 0 to about 0.98 at 1,
  0.79 at 10, 0.6 at 1,000, and 0.2 near 262,144. The passthrough part is
  flat by construction.

Multimodal RoPE: mrope_section [11, 11, 10]; 11 + 11 + 10 = 32. Pairs 0-10
read t, 11-21 read h, 22-31 read w. Patch example: row 7, column 12.

Decode positions: prefill 0 .. n - 1; decode passes the cache length. The
bug (position 0 every step) makes the gap wrong by a growing amount.
Rotating keys twice gives an effective gap of 2m - n (chapter statement).
"""

from manim import *

import kit
from kit import theme as th

M_C = th.HL_A  # \hla: the query's position m
N_C = th.HL_B  # \hlb: the key's position n
GAP_C = th.HL_C  # \hlc: the gap m - n
PASS_C = th.HL_D  # \hld: the passthrough part

WHERE = """\
q, k, v, gate = self.project(x)   # q_norm and k_norm inside
q = apply_rotary_partial(q, cos, sin, self.rotary_dim)
k = apply_rotary_partial(k, cos, sin, self.rotary_dim)
if kv_cache is not None:
    k, v = kv_cache.append(layer_idx, k, v)
"""

TABLES = """\
half = rotary_dim // 2
inv_freq = 1.0 / (
    theta ** (torch.arange(0, half, dtype=torch.float32) / half)
)
positions = torch.arange(max_position, dtype=torch.float32)
angles = torch.outer(positions, inv_freq)
return angles.cos().to(dtype), angles.sin().to(dtype)
"""

HALF = """\
def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

rotated = x * cos_full + _rotate_half(x) * sin_full
"""

PARTIAL = """\
rot, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
cos_full = torch.cat((cos, cos), dim=-1)[None, None]
sin_full = torch.cat((sin, sin), dim=-1)[None, None]
rotated = rot * cos_full + _rotate_half(rot) * sin_full
return torch.cat((rotated, passthrough), dim=-1)
"""

POSITIONS = """\
start = cache.length if cache is not None else 0
positions = torch.arange(start, start + seq)
"""


def freq(i, big_theta=1e7, d=64):
    """theta_i = Theta^(-2i/d)."""
    return big_theta ** (-2 * i / d)


def unit(angle):
    return np.array([np.cos(angle), np.sin(angle), 0.0])


def vec(center, angle, r, color, stroke=6):
    """An arrow from `center` at `angle` (radians, counterclockwise from +x)."""
    return Arrow(center, center + r * unit(angle), buff=0, color=color, stroke_width=stroke, tip_length=0.2,
                 max_tip_length_to_length_ratio=0.3, max_stroke_width_to_length_ratio=20)


def dial(r=1.0, ticks=12, color=th.RULE_STRONG):
    """A clock face: a circle with tick marks and a hub, centred at the origin."""
    face = Circle(radius=r, stroke_color=color, stroke_width=th.STROKE_BOX, fill_color=th.RAISE, fill_opacity=1)
    marks = VGroup(*[
        Line(unit(k * TAU / ticks) * r * 0.86, unit(k * TAU / ticks) * r, color=th.INK_4,
             stroke_width=th.STROKE_HAIRLINE)
        for k in range(ticks)
    ])
    hub = Dot(radius=0.05, color=th.INK)
    g = VGroup(face, marks, hub)
    g.face = face
    return g


def swatch(text, color, size=th.SIZE_LABEL):
    sq = Square(0.26, stroke_color=th.RULE_STRONG, stroke_width=th.STROKE_HAIRLINE,
                fill_color=th.tint(color, 0.35), fill_opacity=1)
    return VGroup(sq, kit.label(text, size=size, color=th.INK_2)).arrange(RIGHT, buff=0.16)


def line_key(text, color, size=th.SIZE_LABEL, dashed=False):
    cls = DashedLine if dashed else Line
    rule = cls(LEFT * 0.25, RIGHT * 0.25, color=color, stroke_width=th.STROKE_CURVE)
    return VGroup(rule, kit.label(text, size=size, color=th.INK_2)).arrange(RIGHT, buff=0.16)


def bf16_round(x):
    """Round-to-nearest-even of x to bfloat16, valid for 65,536 <= x < 131,072."""
    return float(np.round(x / 512.0) * 512.0)


class Lecture(kit.Lecture):
    def construct(self):
        self.title_slide(
            narration=(
                "Rotary position embeddings. "
                "Attention can't see word order on its own. "
                "In this lecture, you derive RoPE from one requirement: the score between two tokens should "
                "depend only on how far apart they are. "
                "Then you meet the two twists this model adds."
            )
        )
        self.order_blind()
        self.rotations()
        self.where_it_sits()
        self.clock_hands()
        self.requirement()
        self.complex_solution()
        self.score_demo()
        self.full_head()
        self.frequencies()
        self.wavelengths()
        self.slowest_pair()
        self.tables()
        self.float32_angles()
        self.half_split()
        self.partial_rotary()
        self.score_split()
        self.no_saving()
        self.mrope()
        self.decode_positions()
        self.what_goes_wrong()
        self.closing_slide(
            [
                r"Rotate $q$ and $k$ by position: the score sees only $m - n$.",
                r"$\Theta$ sets the slowest wavelength; it must outlast the context.",
                r"Build the tables once, in float32; keep the checkpoint's pairing.",
                r"64 of 256 channels rotate; mRoPE is plain RoPE for text.",
            ],
            narration=(
                "Here's what to keep. "
                "Rotate the query and the key by their positions, and the score depends only on the gap. "
                "Theta sets the slowest wavelength, which must outlast the context. "
                "Build the angle tables once, in float32, and keep the pairing the checkpoint was trained with. "
                "This model rotates 64 of each head's 256 channels, and its multimodal positions reduce to "
                "plain RoPE for text."
            ),
            book_section="s-what-goes-wrong",
        )

    # Before you start -------------------------------------------------------------

    def order_blind(self):
        self.slide("order-blind", "Attention can't see word order", book_section="s-before-you-start")
        words = ["the", "cat", "sat", "on", "the", "mat"]
        toks = VGroup(*[kit.box(w, width=1.45, height=0.7, font=th.MONO) for w in words]).arrange(RIGHT, buff=0.25)
        outs = VGroup(*[kit.box(f"o({w})", width=1.45, height=0.7, font=th.MONO, color=self.color) for w in words])
        outs.arrange(RIGHT, buff=0.25)
        band = RoundedRectangle(width=toks.width + 0.4, height=0.8, corner_radius=0.12, stroke_width=0,
                                fill_color=th.SUNK, fill_opacity=1)
        band_t = kit.label("attention, with no positions", size=th.SIZE_LABEL, color=th.INK_3)
        band_t.move_to(band)
        mid = VGroup(band, band_t)
        fig = VGroup(toks, mid, outs).arrange(DOWN, buff=0.45)
        f = kit.formula(r"w_{ij} = \operatorname{softmax}_j\!\left(\frac{q_i \cdot k_j}{\sqrt{d}}\right)",
                        size=th.SIZE_MATH_SMALL)
        note = kit.prose(r"No $i$ or $j$ appears outside the vectors themselves.", size=th.SIZE_SMALL,
                         width=None, color=th.INK_3)
        col = VGroup(fig, f, note).arrange(DOWN, buff=0.45)
        self.place(col)
        with self.narrate(
            "Attention has no idea what order its inputs are in. "
            "Each query weights the values by a softmax over its dot products with the keys. "
            "Nothing in that formula mentions a position, only the vectors themselves. "
            "So swap two tokens, and their outputs swap with them, unchanged. "
            "The cat sat on the mat, and the mat sat on the cat, give the same set of outputs, reordered. "
            "The causal mask decides which tokens are visible, but not how far apart they are. "
            "Position has to be put in on purpose."
        ) as t:
            self.play(FadeIn(toks), FadeIn(mid), run_time=0.8)
            self.play(LaggedStart(*[FadeIn(o, shift=DOWN * 0.1) for o in outs], lag_ratio=0.1), run_time=1.0)
            t.wait_until("Each query weights")
            self.play(FadeIn(f), run_time=0.8)
            t.wait_until("Nothing in that formula")
            self.play(FadeIn(note), run_time=0.6)
            t.wait_until("So swap two tokens")
            a, b = toks[1].get_center(), toks[5].get_center()
            oa, ob = outs[1].get_center(), outs[5].get_center()
            self.play(toks[1].animate.move_to(b), toks[5].animate.move_to(a),
                      outs[1].animate.move_to(ob), outs[5].animate.move_to(oa), run_time=1.6)
            t.wait_until("The cat sat")
            self.play(Indicate(VGroup(outs[1], outs[5]), color=self.color, scale_factor=1.06), run_time=1.0)

    def rotations(self):
        self.slide("rotations", "Rotations compose, invert, and keep length", book_section="s-before-you-start")
        mat = kit.formula(r"R(\alpha) = \begin{bmatrix} \cos\alpha & -\sin\alpha \\ "
                          r"\sin\alpha & \cos\alpha \end{bmatrix}")
        props = VGroup(
            kit.formula(r"R(\alpha)R(\beta) = R(\alpha + \beta)", size=th.SIZE_MATH_SMALL),
            kit.formula(r"R(\alpha)^{\top} = R(-\alpha)", size=th.SIZE_MATH_SMALL),
            kit.formula(r"\lVert R(\alpha)x \rVert = \lVert x \rVert", size=th.SIZE_MATH_SMALL),
        ).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        names = VGroup(*[kit.label(s, size=th.SIZE_LABEL, color=th.INK_3) for s in
                         ("compose", "transpose is inverse", "keep length")])
        for n_, p in zip(names, props):
            n_.next_to(p, LEFT, buff=0.4)
        names.align_to(names[1], RIGHT)
        cplx = kit.prose(r"As complex numbers: multiply by $e^{i\alpha}$.", size=th.SIZE_SMALL, width=None,
                         color=th.INK_3)
        left = VGroup(mat, VGroup(names, props), cplx).arrange(DOWN, buff=0.55, aligned_edge=LEFT)
        r = 1.9
        face = Circle(radius=r, stroke_color=th.RULE_STRONG, stroke_width=th.STROKE_HAIRLINE)
        face.set_stroke(opacity=1)
        dashed = DashedVMobject(face, num_dashes=60)
        ax_h = Line(LEFT * (r + 0.3), RIGHT * (r + 0.3), color=th.RULE_STRONG, stroke_width=th.STROKE_HAIRLINE)
        ax_v = Line(DOWN * (r + 0.3), UP * (r + 0.3), color=th.RULE_STRONG, stroke_width=th.STROKE_HAIRLINE)
        plane = VGroup(ax_h, ax_v, dashed)
        layout = VGroup(left, plane).arrange(RIGHT, buff=1.2)
        self.place(layout)
        o = plane.get_center()
        a0, al, be = 15 * DEGREES, 50 * DEGREES, 40 * DEGREES
        x0 = vec(o, a0, r, th.INK_2)
        x1 = vec(o, a0 + al, r, self.colors[1])
        x2 = vec(o, a0 + al + be, r, self.color)
        arc1 = Arc(radius=0.75, start_angle=a0, angle=al, arc_center=o, color=self.colors[1], stroke_width=3)
        arc2 = Arc(radius=1.05, start_angle=a0 + al, angle=be, arc_center=o, color=self.color, stroke_width=3)
        l0 = kit.math_label("x", size=th.SIZE_SMALL, color=th.INK_2).next_to(o + r * unit(a0), RIGHT, buff=0.1)
        l1 = kit.math_label(r"\alpha", size=th.SIZE_SMALL, color=self.colors[1])
        l1.move_to(o + 0.5 * unit(a0 + al / 2))
        l2 = kit.math_label(r"\beta", size=th.SIZE_SMALL, color=self.color)
        l2.move_to(o + 1.32 * unit(a0 + al + be / 2))
        with self.narrate(
            "You need one tool: the rotation matrix. "
            "R of alpha turns any vector in the plane by the angle alpha. "
            "It has three properties you'll use again and again. "
            "First, they compose: turn by alpha, then by beta, and you've turned by their sum. "
            "Second, the transpose undoes a rotation: it's a rotation by minus alpha. "
            "Third, a rotation never changes a vector's length. "
            "As a complex number, multiplying by e to the i alpha makes the same turn."
        ) as t:
            self.play(FadeIn(mat), Create(plane), GrowArrow(x0), FadeIn(l0), run_time=1.0)
            t.wait_until("R of alpha turns")
            self.play(TransformFromCopy(x0, x1), Create(arc1), FadeIn(l1), run_time=1.2)
            t.wait_until("First, they compose")
            self.play(FadeIn(names[0]), FadeIn(props[0]), run_time=0.6)
            self.play(TransformFromCopy(x1, x2), Create(arc2), FadeIn(l2), run_time=1.2)
            t.wait_until("Second, the transpose")
            self.play(FadeIn(names[1]), FadeIn(props[1]), run_time=0.6)
            ghost = x2.copy()
            self.play(Rotate(ghost, -(al + be), about_point=o), run_time=1.0)
            self.remove(ghost)
            t.wait_until("Third, a rotation")
            self.play(FadeIn(names[2]), FadeIn(props[2]), Indicate(dashed, color=self.color, scale_factor=1.0),
                      run_time=1.0)
            t.wait_until("As a complex number")
            self.play(FadeIn(cplx), run_time=0.6)

    def where_it_sits(self):
        self.slide("where-it-sits", "RoPE runs after the norm, before the cache", book_section="s-before-you-start")
        panel = kit.code(WHERE, "python")
        flow = VGroup(
            kit.box("project q, k, v", sublabel="per-head RMSNorm", width=3.4),
            kit.box("RoPE", sublabel="rotate q and k", color=self.color, width=3.4),
            kit.box("KV cache", sublabel="append k and v", color=th.INFO, width=3.4),
        ).arrange(RIGHT, buff=0.9)
        key = kit.callout("key", r"The cache stores keys that are \textbf{already rotated}: each carries its "
                                 r"absolute position.", width=10.5, size=th.SIZE_SMALL)
        col = VGroup(flow, panel, key).arrange(DOWN, buff=0.55)
        self.place(col)
        a1 = kit.arrow(flow[0], flow[1])
        a2 = kit.arrow(flow[1], flow[2])
        with self.narrate(
            "Where does RoPE run? "
            "Right after the per-head RMSNorm on queries and keys, from chapter 4. "
            "And right before the key and value go into the KV cache. "
            "Lines 2 and 3 rotate the query and the key. "
            "Line 5 appends the key and value to the cache. "
            "That order means the cache stores keys that are already rotated, with their absolute positions "
            "baked in. "
            "Chapter 15 comes back to this when it reuses a prefix."
        ) as t:
            t.wait_until("Right after")
            self.play(FadeIn(flow[0]), FadeIn(panel.frame), *panel.reveal(1), run_time=0.8)
            t.wait_until("And right before")
            self.play(GrowArrow(a1.line), FadeIn(flow[1]), GrowArrow(a2.line), FadeIn(flow[2]), run_time=1.0)
            t.wait_until("Lines 2 and 3")
            self.play(*panel.reveal(2, 3), panel.highlight(2, 3), run_time=0.8)
            t.wait_until("Line 5 appends")
            self.play(*panel.reveal(4, 5), panel.highlight(5, color=th.tint(th.INFO, 0.16)), run_time=0.8)
            t.wait_until("That order means")
            self.play(FadeIn(key, shift=UP * 0.1), run_time=0.8)

    # The requirement -----------------------------------------------------------------

    def clock_hands(self):
        self.slide("clock-hands", "Only the angle between the hands matters", book_section="s-the-requirement")
        r = 2.0
        face = dial(r, ticks=24)
        step = 8 * DEGREES
        q0, k0 = 20 * DEGREES, 50 * DEGREES
        pq, pk = ValueTracker(0), ValueTracker(0)
        q_lab = kit.label("query at position 1010", size=th.SIZE_SMALL, color=M_C, weight=MEDIUM)
        k_lab = kit.label("key at position 1015", size=th.SIZE_SMALL, color=N_C, weight=MEDIUM)
        gap = kit.label("gap: 5 steps", size=th.SIZE_SMALL, color=GAP_C, weight=MEDIUM)
        side = VGroup(q_lab, k_lab, gap).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        rule = kit.prose(r"Every step turns a hand by the same small angle.", size=th.SIZE_SMALL, width=4.6,
                         color=th.INK_3)
        side_col = VGroup(side, rule).arrange(DOWN, buff=0.7, aligned_edge=LEFT)
        layout = VGroup(face, side_col).arrange(RIGHT, buff=1.4)
        self.place(layout)
        o = face.get_center()

        def ang_q():
            return q0 + pq.get_value() * step

        def ang_k():
            return k0 + pk.get_value() * step

        hq = always_redraw(lambda: vec(o, ang_q(), r * 0.9, M_C, stroke=7))
        hk = always_redraw(lambda: vec(o, ang_k(), r * 0.72, N_C, stroke=7))
        arc = always_redraw(lambda: Arc(radius=0.55, start_angle=ang_q(), angle=ang_k() - ang_q(), arc_center=o,
                                        color=GAP_C, stroke_width=5))
        q_num = always_redraw(lambda: kit.label(f"query at position {int(round(pq.get_value()))}",
                                                size=th.SIZE_SMALL, color=M_C, weight=MEDIUM).move_to(q_lab,
                                                                                                    aligned_edge=LEFT))
        k_num = always_redraw(lambda: kit.label(f"key at position {int(round(pk.get_value()))}",
                                                size=th.SIZE_SMALL, color=N_C, weight=MEDIUM).move_to(k_lab,
                                                                                                    aligned_edge=LEFT))
        with self.narrate(
            "Before any formula, pin down what you want. "
            "Picture every vector as a clock hand. "
            "Each step along the sequence turns the hand by the same small angle. "
            "A query at position 10 has turned 10 steps, and a key at position 15 has turned 15. "
            "The extra angle between them is 5 steps' worth. "
            "Now move both tokens to positions 1010 and 1015. "
            "Both hands spin a long way, but the angle between them doesn't change. "
            "The model sees how far apart two tokens are, not where they sit."
        ) as t:
            self.play(FadeIn(face), run_time=0.6)
            t.wait_until("Picture every vector")
            self.add(hq, hk)
            self.play(FadeIn(hq), FadeIn(hk), run_time=0.6)
            t.wait_until("Each step along")
            self.play(FadeIn(rule), run_time=0.6)
            t.wait_until("A query at position 10")
            self.add(q_num, k_num)
            self.play(pq.animate.set_value(10), pk.animate.set_value(15), run_time=2.0)
            t.wait_until("The extra angle")
            self.add(arc)
            self.play(FadeIn(arc), FadeIn(gap), run_time=0.7)
            t.wait_until("Now move both")
            self.play(pq.animate.set_value(1010), pk.animate.set_value(1015), run_time=3.5,
                      rate_func=rate_functions.ease_in_out_sine)
            t.wait_until("Both hands spin")
            self.play(Indicate(gap, color=GAP_C, scale_factor=1.1), run_time=1.0)

    def requirement(self):
        self.slide("requirement", "The score may depend only on the gap", book_section="s-the-requirement")
        req = kit.formula(r"\langle f(q,", r"m", r"),\ f(k,", r"n", r")\rangle", r"=", r"g(q, k,", r"m - n", r")")
        req[1].set_color(M_C)
        req[3].set_color(N_C)
        req[7].set_color(GAP_C)
        lead = kit.prose(r"Add a learned vector per position instead:", size=th.SIZE_SMALL, width=None,
                         color=th.INK_2)
        add = kit.formula(r"(q + p_m)\cdot(k + p_n)", r"=", r"q\cdot k", r"+", r"q\cdot p_n + p_m\cdot k", r"+",
                          r"p_m\cdot p_n", size=th.SIZE_MATH_SMALL)
        bad_note = kit.label("depend on m and n separately", size=th.SIZE_LABEL, color=th.BAD)
        col = VGroup(req, VGroup(lead, add).arrange(DOWN, buff=0.35, aligned_edge=LEFT)).arrange(DOWN, buff=1.1)
        self.place(col, "top")
        col.shift(DOWN * 0.4)
        box = SurroundingRectangle(add[4], color=th.BAD, buff=0.1, stroke_width=th.STROKE_AXIS)
        bad_note.next_to(box, DOWN, buff=0.2)
        with self.narrate(
            "Now say it precisely. "
            "Write f of x and p for the vector attention uses at position p. "
            "The score between a query at m and a key at n must depend on the two vectors, and on the gap, "
            "m minus n. "
            "Nothing else is allowed. "
            "That rules out the obvious approach. "
            "Add a learned vector for each position, and expand the score. "
            "The middle terms depend on m and n separately, not on their difference. "
            "The requirement asks the absolute positions to cancel exactly, for every query and key."
        ) as t:
            t.wait_until("The score between")
            self.play(FadeIn(req[:5]), run_time=0.8)
            self.play(FadeIn(req[5:]), run_time=0.8)
            t.wait_until("Nothing else")
            self.play(Circumscribe(req[7], color=GAP_C), run_time=1.0)
            t.wait_until("Add a learned")
            self.play(FadeIn(lead), FadeIn(add), run_time=0.9)
            t.wait_until("The middle terms")
            self.play(Create(box), FadeIn(bad_note), run_time=0.8)

    # The two-dimensional solution ------------------------------------------------

    def complex_solution(self):
        self.slide("complex-solution", "A rotation makes the absolute angles cancel",
                   book_section="s-the-two-dimensional-solution")
        f = kit.formula(r"f(x, p) = x\,e^{i p \theta}")
        d = kit.Derivation([
            r"\langle f(q, m), f(k, n) \rangle &= \operatorname{Re}\!\left[q\,e^{i m \theta}\cdot"
            r"\overline{k\,e^{i n \theta}}\right]",
            r"&= \operatorname{Re}\!\left[q\,\bar{k}\,e^{i m \theta}\,e^{-i n \theta}\right]",
            r"&= \operatorname{Re}\!\left[q\,\bar{k}\,e^{i (m - n) \theta}\right]",
        ], size=th.SIZE_MATH_SMALL, buff=0.45)
        note = kit.prose(r"Only the gap $m - n$ survives.", size=th.SIZE_SMALL, width=None, color=GAP_C)
        col = VGroup(f, d, note).arrange(DOWN, buff=0.6)
        self.place(col)
        box = SurroundingRectangle(d.lines[2][1], color=GAP_C, buff=0.12, stroke_width=th.STROKE_AXIS)
        with self.narrate(
            "The clock picture already holds the answer. "
            "Start with two dimensions, and treat the query and the key as complex numbers. "
            "Rotate each one by its position times a fixed frequency, theta. "
            "Take the real inner product. "
            "Conjugating the key's rotation flips the sign of its angle. "
            "So the two exponentials merge into one, with angle m minus n, times theta. "
            "The absolute positions have cancelled, which is exactly the requirement. "
            "This two-dimensional answer turns out to be the whole problem."
        ) as t:
            t.wait_until("Rotate each one")
            self.play(FadeIn(f), run_time=0.8)
            t.wait_until("Take the real")
            self.play(d.show(0), run_time=0.9)
            t.wait_until("Conjugating")
            self.play(d.show(1), run_time=0.9)
            t.wait_until("So the two exponentials")
            self.play(d.show(2), run_time=0.9)
            t.wait_until("The absolute positions")
            self.play(Create(box), FadeIn(note), run_time=0.8)

    def score_demo(self):
        self.slide("score-demo", "Slide both tokens along: the score holds still",
                   book_section="s-the-two-dimensional-solution")
        theta = 0.3
        qv, kv = np.array([1.0, 0.4]), np.array([0.6, 0.9])
        aq0, ak0 = np.arctan2(qv[1], qv[0]), np.arctan2(kv[1], kv[0])
        lq, lk = np.linalg.norm(qv), np.linalg.norm(kv)
        m, n = ValueTracker(10), ValueTracker(5)

        def score():
            a = (m.get_value() - n.get_value()) * theta
            return (qv @ kv) * np.cos(a) + (qv[0] * kv[1] - qv[1] * kv[0]) * np.sin(a)

        r = 1.55
        face = dial(r * 1.12, ticks=12)
        mat = kit.formula(r"\langle R(m\theta)q,\ R(n\theta)k\rangle", r"=", size=th.SIZE_MATH_SMALL)
        exp1 = kit.formula(r"(q_0k_0 + q_1k_1)\cos(", r"\delta", r"\theta)", size=th.SIZE_MATH_SMALL)
        exp2 = kit.formula(r"+\ (q_0k_1 - q_1k_0)\sin(", r"\delta", r"\theta)", size=th.SIZE_MATH_SMALL)
        for e in (exp1, exp2):
            e[1].set_color(GAP_C)
        maths = VGroup(mat, exp1, exp2).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        exp2.shift(RIGHT * 0.4)
        labels = ["query position m", "key position n", "gap", "score"]
        cols = [M_C, N_C, GAP_C, th.INK]
        names = VGroup(*[kit.label(s, size=th.SIZE_SMALL, color=c) for s, c in zip(labels, cols)])
        names.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        slots = VGroup(*[Rectangle(width=1.3, height=0.4, stroke_width=0, fill_opacity=0).next_to(nm, RIGHT,
                                                                                                  buff=0.5)
                         for nm in names])
        slots.align_to(slots[0], LEFT)
        table = VGroup(names, slots)
        props = kit.prose(r"Orthogonal: lengths unchanged. No learned parameters.", size=th.SIZE_SMALL, width=None,
                          color=th.INK_3)
        right = VGroup(maths, table, props).arrange(DOWN, buff=0.55, aligned_edge=LEFT)
        layout = VGroup(face, right).arrange(RIGHT, buff=1.0)
        self.place(layout)
        o = face.get_center()
        hq = always_redraw(lambda: vec(o, aq0 + m.get_value() * theta, r * lq / 1.08, M_C))
        hk = always_redraw(lambda: vec(o, ak0 + n.get_value() * theta, r * lk / 1.08, N_C))

        def readout(i, fn, color):
            return always_redraw(lambda: kit.label(fn(), size=th.SIZE_SMALL, color=color, font=th.MONO,
                                                   weight=MEDIUM).move_to(slots[i], aligned_edge=LEFT))

        vals = VGroup(
            readout(0, lambda: f"{int(round(m.get_value()))}", M_C),
            readout(1, lambda: f"{int(round(n.get_value()))}", N_C),
            readout(2, lambda: f"{int(round(m.get_value() - n.get_value()))}", GAP_C),
            readout(3, lambda: f"{score():.3f}".replace("-", "−"), th.INK),
        )
        with self.narrate(
            "Here's the same result as matrices, which is what the code uses. "
            "Rotate the query by m theta, and the key by n theta. "
            "The score mixes the ordinary dot product of q and k with the signed area they span. "
            "The cosine and sine of the gap set the mix. "
            "Start with the query at 10 and the key at 5. "
            "Now slide both along, to 300 and 295. "
            "The hands spin, but the score never moves. "
            "Open the gap to 10, and the score changes. "
            "The rotation is orthogonal, so it keeps the norms that chapter 4 set, and it has no learned parameters."
        ) as t:
            self.play(FadeIn(face), FadeIn(mat), run_time=0.8)
            t.wait_until("Rotate the query")
            self.add(hq, hk)
            self.play(FadeIn(hq), FadeIn(hk), run_time=0.7)
            t.wait_until("The score mixes")
            self.play(FadeIn(exp1), run_time=0.7)
            self.play(FadeIn(exp2), run_time=0.7)
            t.wait_until("Start with the query")
            self.add(*vals)
            self.play(FadeIn(names), run_time=0.7)
            t.wait_until("Now slide both")
            self.play(m.animate.set_value(300), n.animate.set_value(295), run_time=4.0,
                      rate_func=rate_functions.ease_in_out_sine)
            t.wait_until("Open the gap")
            self.play(n.animate.set_value(290), run_time=1.6)
            self.play(Indicate(names[3], color=GAP_C), run_time=0.8)
            t.wait_until("The rotation is orthogonal")
            self.play(FadeIn(props), run_time=0.7)

    # Extending to a full head -----------------------------------------------------

    def full_head(self):
        self.slide("full-head", "A head is many pairs, each rotating on its own",
                   book_section="s-extending-to-a-full-head")
        g = kit.grid(8, 8, cell=0.5)
        cap = kit.prose(r"$R_p$, first 8 channels", size=th.SIZE_LABEL, color=th.INK_3, width=None)
        cap.next_to(g, UP, buff=0.2)
        dots = kit.math_label(r"\ddots", size=th.SIZE_HEADING, color=th.INK_3).next_to(g, DR, buff=0.05)
        blocks = []
        for b in range(4):
            lab = kit.math_label(rf"R(p\theta_{b})", size=th.SIZE_LABEL, color=th.INK)
            lab.move_to(g.block(2 * b, 2 * b + 2, 2 * b, 2 * b + 2))
            blocks.append(lab)
        zero = kit.label("zeros elsewhere", size=th.SIZE_LABEL, color=th.INK_4)
        zero.next_to(g, DOWN, buff=0.2).align_to(g, LEFT)
        fig = VGroup(g, cap, dots, *blocks, zero)
        s = kit.formula(r"\langle R_m q, R_n k\rangle", r"=",
                        r"\sum_{i=0}^{d/2-1} \langle R(m\theta_i) q^{(i)},\ R(n\theta_i) k^{(i)}\rangle",
                        size=th.SIZE_MATH_SMALL)
        ident = kit.formula(r"R_m^{\top} R_n = R_{n-m}", size=th.SIZE_MATH_SMALL)
        note = kit.prose(r"Every term depends only on $m - n$, so the sum does too.", size=th.SIZE_SMALL,
                         width=5.6)
        right = VGroup(ident, s, note).arrange(DOWN, buff=0.6, aligned_edge=LEFT)
        layout = VGroup(fig, right).arrange(RIGHT, buff=0.8)
        self.place(layout)
        with self.narrate(
            "A real head isn't two channels wide. "
            "So split it into pairs of channels, and give each pair its own frequency, theta i. "
            "Each pair rotates on its own, which makes the full transform block-diagonal. "
            "There's one small rotation per pair, and zeros everywhere else. "
            "A block-diagonal matrix of rotations is still orthogonal, so the identity carries over to the "
            "whole head. "
            "The score splits into one term per pair. "
            "Every term depends only on m minus n, so the sum does too. "
            "The requirement holds for a head of any width."
        ) as t:
            self.play(FadeIn(g), FadeIn(cap), run_time=0.7)
            t.wait_until("So split it")
            for b in range(4):
                self.play(g.fill(g.block(2 * b, 2 * b + 2, 2 * b, 2 * b + 2), self.colors[b]), FadeIn(blocks[b]),
                          run_time=0.45)
            self.play(FadeIn(dots), run_time=0.3)
            t.wait_until("There's one small")
            self.play(FadeIn(zero), run_time=0.6)
            t.wait_until("A block-diagonal")
            self.play(FadeIn(ident), run_time=0.7)
            t.wait_until("The score splits")
            self.play(FadeIn(s), run_time=0.9)
            t.wait_until("Every term depends")
            self.play(FadeIn(note), run_time=0.7)

    def frequencies(self):
        self.slide("frequencies", "Fast pairs see nearby, slow pairs see far",
                   book_section="s-extending-to-a-full-head")
        f = kit.formula(r"\theta_i = \Theta^{-2i/d} = \Theta^{-i/32}", size=th.SIZE_MATH_SMALL)
        given = kit.prose(r"$d = 64$ rotary channels, 32 pairs, $\Theta = 10^7$ (the lab's value)",
                          size=th.SIZE_SMALL, width=None, color=th.INK_3)
        pairs = [0, 8, 16, 31]
        waves = ["6.3", "353", "19,900", "38 million"]
        r = 1.05
        faces = VGroup(*[dial(r, ticks=12) for _ in pairs]).arrange(RIGHT, buff=0.9)
        names = VGroup()
        for face, i, w in zip(faces, pairs, waves):
            nm = kit.label(f"pair {i}", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
            wl = kit.label(f"λ ≈ {w}", size=th.SIZE_LABEL, color=th.INK_3)
            col = VGroup(nm, wl).arrange(DOWN, buff=0.12).next_to(face, DOWN, buff=0.3)
            names.add(col)
        pos_slot = Rectangle(width=4.0, height=0.45, stroke_width=0, fill_opacity=0)
        head = VGroup(f, given).arrange(DOWN, buff=0.3)
        col = VGroup(head, VGroup(faces, names), pos_slot).arrange(DOWN, buff=0.5)
        self.place(col)
        p = ValueTracker(0)
        hands = VGroup(*[
            always_redraw(lambda face=face, i=i: vec(face.get_center(), PI / 2 + p.get_value() * freq(i), r * 0.88,
                                                     self.colors[k % 4], stroke=6))
            for k, (face, i) in enumerate(zip(faces, pairs))
        ])
        pos = always_redraw(lambda: kit.label(f"position {int(round(p.get_value()))}", size=th.SIZE_SMALL,
                                              color=th.INK, font=th.MONO, weight=MEDIUM).move_to(pos_slot))
        with self.narrate(
            "Why give each pair a different speed? "
            "Think of a clock. "
            "The second hand resolves seconds, and the hour hand resolves hours. "
            "One hyperparameter, rope theta, sets the speeds as a geometric sequence. "
            "With 64 rotary channels, there are 32 frequencies. "
            "Pair 0 always turns one radian per position, a full turn every 6.3 positions. "
            "Watch 100 positions go by. "
            "Pair 0 spins again and again, and pair 31 barely moves. "
            "Fast pairs resolve nearby positions, and slow pairs resolve distant ones."
        ) as t:
            self.play(FadeIn(faces), run_time=0.7)
            self.add(*hands)
            self.play(FadeIn(hands), run_time=0.5)
            t.wait_until("One hyperparameter")
            self.play(FadeIn(f), run_time=0.8)
            t.wait_until("With 64 rotary")
            self.play(FadeIn(given), FadeIn(names), run_time=0.8)
            t.wait_until("Watch 100 positions")
            self.add(pos)
            self.play(p.animate.set_value(100), run_time=6.0, rate_func=linear)
            t.wait_until("Fast pairs resolve")
            self.play(Indicate(names[0], color=self.color), Indicate(names[3], color=self.colors[3]), run_time=1.0)

    # What Theta controls -------------------------------------------------------------

    def wavelengths(self):
        self.slide("wavelengths", "Θ stretches the slow end, not the fast end",
                   book_section="s-what-theta-controls-and-why-long-context-raises-it")
        ax = kit.axes([0, 31, 1], [0, 8, 1], width=7.4, height=4.4,
                      x_label=r"\text{pair } i", y_label=r"\text{wavelength (positions)}",
                      x_numbers=[0, 8, 16, 24, 31])
        ylabs = VGroup(*[kit.math_label(rf"10^{{{k}}}", size=th.SIZE_LABEL - 2, color=th.INK_3)
                         .next_to(ax.c2p(0, k), LEFT, buff=0.14) for k in (0, 2, 4, 6, 8)])
        ax.add(ylabs)
        base = np.log10(2 * np.pi)
        lo = ax.plot(lambda i: base + i * 4 / 32, x_range=[0, 31], color=th.INFO, stroke_width=th.STROKE_CURVE)
        hi = ax.plot(lambda i: base + i * 7 / 32, x_range=[0, 31], color=self.color, stroke_width=th.STROKE_CURVE)
        ctx_y = np.log10(262144)
        ctx = DashedLine(ax.c2p(0, ctx_y), ax.c2p(31, ctx_y), color=th.INK_3, stroke_width=th.STROKE_AXIS)
        ctx_t = kit.label("context: 262,144", size=th.SIZE_LABEL, color=th.INK_3)
        ctx_t.next_to(ax.c2p(0, ctx_y), UR, buff=0.1)
        fast = Dot(ax.c2p(0, base), radius=0.08, color=th.INK)
        end_lo = Dot(ax.c2p(31, base + 31 * 4 / 32), radius=0.08, color=th.INFO)
        end_hi = Dot(ax.c2p(31, base + 31 * 7 / 32), radius=0.08, color=self.color)
        plot = VGroup(ax, lo, hi, ctx, ctx_t, fast, end_lo, end_hi)
        f = kit.formula(r"\lambda_i = \frac{2\pi}{\theta_i} = 2\pi\,\Theta^{2i/d}", size=th.SIZE_MATH_SMALL)
        k1 = line_key("Θ = 10⁴: slowest about 47,000", th.INFO)
        k2 = line_key("Θ = 10⁷: slowest about 38 million", self.color)
        fast_t = kit.prose(r"Fast end: $2\pi \approx 6.3$ for every $\Theta$.", size=th.SIZE_SMALL, width=None,
                           color=th.INK_3)
        side = VGroup(f, k1, k2, fast_t).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        layout = VGroup(plot, side).arrange(RIGHT, buff=0.6)
        self.place(layout)
        with self.narrate(
            "Read each frequency as a wavelength: the positions it takes to complete one full turn. "
            "On a log scale, the wavelengths climb in a straight line. "
            "Theta sets how steep that line is. "
            "The fast end stays at 2 pi, about 6.3 positions, whatever theta is. "
            "The design constraint is at the slow end. "
            "The slowest wavelength has to outlast the context you intend to serve, here 262,144 positions. "
            "At theta of 10,000, the slowest pair turns once every 47,000 positions or so, well short of that. "
            "At 10 million, it takes about 38 million positions."
        ) as t:
            self.play(Create(ax), FadeIn(f), run_time=1.0)
            t.wait_until("On a log scale")
            self.play(Create(hi), FadeIn(k2), run_time=1.2)
            t.wait_until("Theta sets how")
            self.play(Create(lo), FadeIn(k1), run_time=1.2)
            t.wait_until("The fast end")
            self.play(FadeIn(fast, scale=0.5), FadeIn(fast_t), run_time=0.7)
            t.wait_until("The slowest wavelength")
            self.play(Create(ctx), FadeIn(ctx_t), run_time=0.8)
            t.wait_until("At theta of 10,000")
            self.play(FadeIn(end_lo, scale=0.5), Indicate(k1, color=th.INFO), run_time=0.9)
            t.wait_until("At 10 million")
            self.play(FadeIn(end_hi, scale=0.5), Indicate(k2, color=self.color), run_time=0.9)

    def slowest_pair(self):
        self.slide("slowest-pair", "A small Θ lets the slowest hand lap itself",
                   book_section="s-what-theta-controls-and-why-long-context-raises-it")
        r = 1.2
        faces = VGroup(dial(r, ticks=12), dial(r, ticks=12)).arrange(RIGHT, buff=4.2)
        titles = VGroup(kit.label("Θ = 10⁴", size=th.SIZE_SMALL, color=th.INFO, weight=MEDIUM),
                        kit.label("Θ = 10⁷", size=th.SIZE_SMALL, color=self.color, weight=MEDIUM))
        for tt, fc in zip(titles, faces):
            tt.next_to(fc, DOWN, buff=0.25)
        res = VGroup(kit.label("5.6 turns", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM),
                     kit.label("0.0069 turns, about 2.5°", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM))
        for rr, tt in zip(res, titles):
            rr.next_to(tt, DOWN, buff=0.15)
        key = kit.callout("key", r"$\Theta$ lives only in a precomputed table, so it's free to serve. "
                                 r"But serve a model at the $\Theta$ it was trained with.",
                          title="Raise it before training", width=11.0, size=th.SIZE_SMALL)
        top = VGroup(faces, titles, res)
        col = VGroup(top, key).arrange(DOWN, buff=0.5)
        self.place(col)
        lam = [2 * np.pi * 1e4 ** (31 / 32), 2 * np.pi * 1e7 ** (31 / 32)]
        cols = [th.INFO, self.color]
        p = ValueTracker(0)
        hands = VGroup(*[
            always_redraw(lambda j=j: vec(faces[j].get_center(), PI / 2 - TAU * p.get_value() / lam[j], r * 0.88,
                                          cols[j], stroke=6))
            for j in range(2)
        ])
        pos_slot = Rectangle(width=3.2, height=0.4, stroke_width=0, fill_opacity=0)
        pos_slot.move_to(faces.get_center())
        pos = always_redraw(lambda: kit.label(f"position {int(round(p.get_value())):,}", size=th.SIZE_LABEL,
                                              color=th.INK_3, font=th.MONO).move_to(pos_slot))
        c0 = faces[0].get_center()
        marks = VGroup()
        for val, rad in ((1000, r * 1.0), (48000, r * 1.0)):
            a = PI / 2 - TAU * val / lam[0]
            marks.add(Dot(c0 + rad * unit(a), radius=0.07, color=th.BAD))
        mark_t = kit.label("1,000 and 48,000", size=th.SIZE_LABEL, color=th.BAD)
        mark_t.next_to(marks, RIGHT, buff=0.2)
        with self.narrate(
            "Now spin the slowest pair across the whole context. "
            "At theta of 10,000, it completes more than five full turns. "
            "Position 1,000 and position 48,000 land at nearly the same angle. "
            "Every faster pair wraps even more, so the model loses its coarse sense of where it is. "
            "At 10 million, the slowest pair covers less than 1 percent of a turn, about 2.5 degrees. "
            "It has become a steady, coarse position signal. "
            "That's why long-context models raise theta. "
            "It only changes a precomputed table, but a model must be served at the theta it was trained with."
        ) as t:
            self.play(FadeIn(faces), FadeIn(titles), run_time=0.8)
            self.add(*hands, pos)
            self.play(FadeIn(hands), run_time=0.4)
            t.wait_until("At theta of 10,000")
            self.play(p.animate.set_value(262144), run_time=4.0, rate_func=linear)
            self.play(FadeIn(res[0]), run_time=0.5)
            t.wait_until("Position 1,000")
            self.play(FadeIn(marks, scale=0.5), FadeIn(mark_t), run_time=0.8)
            t.wait_until("At 10 million")
            self.play(FadeIn(res[1]), Indicate(hands[1], color=self.color, scale_factor=1.05), run_time=0.9)
            t.wait_until("That's why")
            self.play(FadeIn(key, shift=UP * 0.1), run_time=0.8)

    # Implementation ---------------------------------------------------------------------

    def tables(self):
        self.slide("tables", "Build the angle tables once", book_section="s-implementation")
        panel = kit.code(TABLES, "python")
        rows, ci = 6, [0, 4, 8, 12, 16, 20, 24, 28]
        vals = [[np.cos(p * freq(i)) for i in ci] for p in range(rows)]
        g = kit.grid(rows, len(ci), cell=0.33)
        cap = kit.prose(r"$\cos(p\,\theta_i)$ for positions 0 to 5 (down) and every 4th pair (across).",
                        size=th.SIZE_SMALL, color=th.INK_3, width=6.0)
        size = kit.prose(r"Each table: $32768 \times 32 \times 4$ B $= 4.19$ MB. Both: 8.4 MB.",
                         size=th.SIZE_SMALL, width=6.0)
        notes = VGroup(cap, size).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        right = VGroup(g, notes).arrange(RIGHT, buff=0.5)
        layout = VGroup(panel, right).arrange(DOWN, buff=0.45)
        self.place(layout)

        def color_of(v):
            return th.tint(self.color if v >= 0 else th.INFO, 0.08 + 0.55 * abs(v))

        with self.narrate(
            "The angles depend only on position, never on activations. "
            "So they're the same for every layer, every head, and every request, and you build them once. "
            "The inverse frequencies are theta to the minus i over 32, falling from 1. "
            "Their outer product with the positions gives every angle, p times theta i. "
            "Take the cosine and sine, then cast to the working precision. "
            "The engine builds 32,768 positions by default, so each table is 4.19 MB. "
            "Against 53.8 GB of weights, that's nothing."
        ) as t:
            self.play(FadeIn(panel.frame), *panel.reveal(1), run_time=0.7)
            t.wait_until("The inverse frequencies")
            self.play(*panel.reveal(2, 3, 4), panel.highlight(2, 3, 4), run_time=0.8)
            t.wait_until("Their outer product")
            self.play(*panel.reveal(5, 6), FadeIn(g), FadeIn(cap), run_time=0.8)
            for p in range(rows):
                self.play(*[g.cell(p, c).square.animate.set_fill(color_of(vals[p][c]), opacity=1)
                            for c in range(len(ci))], run_time=0.25)
            t.wait_until("Take the cosine")
            self.play(*panel.reveal(7), run_time=0.6)
            t.wait_until("The engine builds")
            self.play(FadeIn(size), run_time=0.7)

    def float32_angles(self):
        self.slide("float32-angles", "In bfloat16, nearby positions collapse", book_section="s-implementation")
        lo, hi = 99500, 101100
        width = 11.0

        def x_of(v):
            return -width / 2 + (v - lo) / (hi - lo) * width

        line32 = NumberLine(x_range=[lo, hi, 100], length=width, color=th.INK_3, include_ticks=False,
                            stroke_width=th.STROKE_AXIS)
        line16 = line32.copy()
        line16.shift(DOWN * 1.9)
        n32 = kit.label("float32", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
        n16 = kit.label("bfloat16", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
        n32.next_to(line32, UP, buff=0.5).align_to(line32, LEFT)
        n16.next_to(line16, UP, buff=0.5).align_to(line16, LEFT)
        grid_vals = [99840, 100352, 100864]
        ticks = VGroup()
        tick_labels = VGroup()
        for v in grid_vals:
            tk = Line(UP * 0.18, DOWN * 0.18, color=th.INK_3, stroke_width=th.STROKE_AXIS)
            tk.move_to([x_of(v), line16.get_y(), 0])
            ticks.add(tk)
            tl = kit.label(f"{v:,}", size=th.SIZE_LABEL, color=th.INK_3, font=th.MONO)
            tl.next_to(tk, DOWN, buff=0.15)
            tick_labels.add(tl)
        end_labels = VGroup(
            kit.label(f"{lo:,}", size=th.SIZE_LABEL, color=th.INK_3, font=th.MONO),
            kit.label(f"{hi:,}", size=th.SIZE_LABEL, color=th.INK_3, font=th.MONO),
        )
        end_labels[0].next_to(line32.get_left(), DOWN, buff=0.2)
        end_labels[1].next_to(line32.get_right(), DOWN, buff=0.2)
        positions = list(range(99600, 100701, 100))
        dots32 = VGroup(*[Dot([x_of(v), line32.get_y(), 0], radius=0.08, color=self.color) for v in positions])
        gap_t = kit.label("512 apart", size=th.SIZE_LABEL, color=th.BAD)
        brace = BraceBetweenPoints([x_of(grid_vals[0]), line16.get_y() - 0.6, 0],
                                   [x_of(grid_vals[1]), line16.get_y() - 0.6, 0], DOWN, color=th.BAD)
        brace.next_to(tick_labels[:2], DOWN, buff=0.1)
        gap_t.next_to(brace, DOWN, buff=0.08)
        fig = VGroup(line32, line16, n32, n16, ticks, tick_labels, end_labels, dots32, brace, gap_t)
        fix = kit.prose(r"Fix: compute the angles in float32, take $\cos$ and $\sin$ (in $[-1, 1]$), cast after.",
                        size=th.SIZE_SMALL, width=None)
        col = VGroup(fig, fix).arrange(DOWN, buff=0.5)
        self.place(col)
        y16 = line16.get_y()
        targets = [Dot([x_of(bf16_round(v)), y16, 0], radius=0.08, color=th.BAD) for v in positions]
        with self.narrate(
            "Compute those angles in float32. "
            "For pair 0, the angle at position p is p radians. "
            "Near position 100,000, neighboring bfloat16 values sit 512 apart. "
            "So every position there rounds to the nearest multiple of 512. "
            "Whole blocks of neighboring positions collapse onto one angle, and the model can't tell them apart. "
            "Instead, compute the angles in float32, then take the cosine and sine. "
            "Those land between minus one and one, where bfloat16 is comfortable, so cast after."
        ) as t:
            self.play(Create(line32), FadeIn(n32), FadeIn(end_labels), run_time=0.8)
            t.wait_until("For pair 0")
            self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots32], lag_ratio=0.08), run_time=1.2)
            t.wait_until("Near position 100,000")
            self.play(Create(line16), FadeIn(n16), FadeIn(ticks), FadeIn(tick_labels), FadeIn(brace),
                      FadeIn(gap_t), run_time=1.0)
            t.wait_until("So every position")
            self.play(*[TransformFromCopy(d, tg) for d, tg in zip(dots32, targets)], run_time=1.8)
            t.wait_until("Instead, compute")
            self.play(FadeIn(fix, shift=UP * 0.1), run_time=0.8)

    def half_split(self):
        self.slide("half-split", "Half split: channel j pairs with channel j + 32",
                   book_section="s-implementation")
        nch = 64
        cw = 0.17
        inter = kit.grid(1, nch, cell=(cw, 0.42))
        halfg = kit.grid(1, nch, cell=(cw, 0.42))
        n1 = kit.label("interleaved: 2i with 2i + 1", size=th.SIZE_SMALL, color=th.INK_2)
        n2 = kit.label("half split: j with j + 32", size=th.SIZE_SMALL, color=th.INK_2)
        top = VGroup(n1, inter).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        show = [0, 1, 2, 31]
        colors = [self.colors[0], self.colors[1], self.colors[2], self.colors[3]]
        arcs_h = VGroup()
        for j, c in zip(show, colors):
            a, b = halfg.cell(0, j).get_bottom(), halfg.cell(0, j + 32).get_bottom()
            arcs_h.add(ArcBetweenPoints(a, b, angle=PI * 0.3, color=c, stroke_width=4))
        mid = VGroup(n2, VGroup(halfg, arcs_h)).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        grids = VGroup(top, mid).arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        panel = kit.code(HALF, "python")
        outs = VGroup(
            kit.formula(r"\text{out}_j = a c - b s", size=th.SIZE_MATH_SMALL),
            kit.formula(r"\text{out}_{j+32} = a s + b c", size=th.SIZE_MATH_SMALL),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        bottom = VGroup(panel, outs).arrange(RIGHT, buff=0.6)
        col = VGroup(grids, bottom).arrange(DOWN, buff=0.5)
        self.place(col)
        with self.narrate(
            "The derivation paired neighboring channels: 2 i with 2 i plus 1. "
            "Implementations mostly use the half split instead. "
            "Within the rotary block, channel j pairs with channel j plus 32. "
            "The helper splits the block in halves, and returns minus the second half, then the first. "
            "Multiply and add, and each pair gets exactly R of p theta j. "
            "The half split is faster, because both halves are contiguous slices. "
            "Both pairings give identical scores, but the pairing is a contract with the checkpoint. "
            "Get it wrong, and the model writes fluent text with no idea of word order."
        ) as t:
            self.play(FadeIn(top), run_time=0.6)
            self.play(*[inter.fill(VGroup(inter.cell(0, 2 * j), inter.cell(0, 2 * j + 1)), c)
                        for j, c in zip(show, colors)], run_time=1.0)
            t.wait_until("Within the rotary")
            self.play(FadeIn(n2), FadeIn(halfg), run_time=0.6)
            self.play(LaggedStart(*[Create(a) for a in arcs_h], lag_ratio=0.2),
                      *[halfg.fill(VGroup(halfg.cell(0, j), halfg.cell(0, j + 32)), c)
                        for j, c in zip(show, colors)], run_time=1.2)
            t.wait_until("The helper splits")
            self.play(FadeIn(panel.frame), *panel.reveal(1, 2, 3), run_time=0.8)
            t.wait_until("Multiply and add")
            self.play(*panel.reveal(5), panel.highlight(5), run_time=0.7)
            self.play(FadeIn(outs), run_time=0.8)
            t.wait_until("The half split is faster")
            self.play(Indicate(VGroup(*[halfg.cell(0, c) for c in range(32)]), color=self.color,
                               scale_factor=1.02), run_time=1.0)

    # Partial rotary ------------------------------------------------------------------------

    def partial_rotary(self):
        self.slide("partial-rotary", "Only the first 64 of 256 channels rotate", book_section="s-partial-rotary")
        f = kit.formula(r"\text{rotary\_dim} = 256 \times 0.25 = 64", size=th.SIZE_MATH_SMALL)
        g = kit.grid(1, 32, cell=(0.34, 0.6), gap=0.03)
        rot = g.block(0, 1, 0, 8)
        pas = g.block(0, 1, 8, 32)
        b1 = Brace(rot, UP, buff=0.1, color=th.INK_4)
        b2 = Brace(pas, UP, buff=0.1, color=th.INK_4)
        t1 = kit.label("64 rotate: 32 pairs", size=th.SIZE_LABEL, color=self.color).next_to(b1, UP, buff=0.08)
        t2 = kit.label("192 pass through untouched", size=th.SIZE_LABEL, color=PASS_C).next_to(b2, UP, buff=0.08)
        head = kit.label("one head: 256 channels (each cell is 8)", size=th.SIZE_LABEL, color=th.INK_3)
        head.next_to(g, DOWN, buff=0.15).align_to(g, LEFT)
        fig = VGroup(g, b1, b2, t1, t2, head)
        panel = kit.code(PARTIAL, "python")
        col = VGroup(f, fig, panel).arrange(DOWN, buff=0.45)
        self.place(col)
        with self.narrate(
            "This model departs from textbook RoPE in two ways. "
            "First, it rotates only a quarter of each head. "
            "The partial rotary factor is 0.25, and a head is 256 channels wide, so the rotary dimension is 64. "
            "Only the first 64 channels rotate, in 32 pairs. "
            "The other 192 pass through untouched. "
                        "One table serves all 24 query heads and all 4 KV heads, by broadcasting. "
            "Check that boundary against the config. "
            "Splitting at 128, half the head, gives a model that runs, writes fluent text, and ignores word "
            "order."
        ) as t:
            self.play(FadeIn(g), FadeIn(head), run_time=0.7)
            t.wait_until("The partial rotary factor")
            self.play(FadeIn(f), run_time=0.7)
            t.wait_until("Only the first 64")
            self.play(g.fill(rot, self.color), FadeIn(b1), FadeIn(t1), run_time=0.8)
            t.wait_until("The other 192")
            self.play(g.fill(pas, PASS_C, tint=0.18), FadeIn(b2), FadeIn(t2), run_time=0.8)
            self.play(FadeIn(panel.frame), *panel.reveal(1, 2, 3, 4, 5), run_time=0.9)
            t.wait_until("One table serves")
            self.play(panel.highlight(2, 3), run_time=0.6)
            t.wait_until("Check that boundary")
            self.play(panel.highlight(1, color=th.tint(th.BAD, 0.14)), Circumscribe(f, color=th.BAD), run_time=1.0)

    def score_split(self):
        self.slide("score-split", "Each head adds a score that ignores distance", book_section="s-partial-rotary")
        s = kit.formula(r"\text{score}", r"=", r"q_{\text{rot}}^{\top} R_{n-m}\, k_{\text{rot}}", r"+",
                        r"q_{\text{pass}}^{\top} k_{\text{pass}}", size=th.SIZE_MATH_SMALL)
        s[2].set_color(GAP_C)
        s[4].set_color(PASS_C)
        u1 = kit.label("depends on m − n", size=th.SIZE_LABEL, color=GAP_C)
        u2 = kit.label("no position at all", size=th.SIZE_LABEL, color=PASS_C)
        u1.next_to(s[2], DOWN, buff=0.2)
        u2.next_to(s[4], DOWN, buff=0.2)
        top = VGroup(s, u1, u2)
        th7 = np.array([freq(i) for i in range(32)])

        def smooth(lg):
            gap = 10 ** lg
            gs = np.linspace(0.9 * gap, 1.1 * gap, 120)
            return float(np.mean(np.cos(np.outer(gs, th7))))

        xmax = np.log10(262144)
        ax = kit.axes([0, 5.5, 1], [0, 1.1, 0.25], width=8.2, height=3.0,
                      y_label=r"\text{score per channel pair}",
                      y_numbers=[0, 0.5, 1.0], decimals=1)
        xlabs = VGroup(*[kit.math_label(rf"10^{{{k}}}", size=th.SIZE_LABEL - 2, color=th.INK_3)
                         .next_to(ax.c2p(k, 0), DOWN, buff=0.14) for k in range(6)])
        ax.add(xlabs)
        xt = kit.math_label(r"\text{gap } m - n", size=th.SIZE_LABEL, color=th.INK_3)
        xt.next_to(xlabs, DOWN, buff=0.15)
        ax.add(xt)
        rot = ax.plot(smooth, x_range=[0, xmax, xmax / 160], color=GAP_C, stroke_width=th.STROKE_CURVE)
        flat = ax.plot(lambda x: 1.0, x_range=[0, xmax], color=PASS_C, stroke_width=th.STROKE_CURVE)
        k1 = line_key("rotated part", GAP_C)
        k2 = line_key("passthrough part", PASS_C)
        cap = kit.label("q and k equal in every pair; Θ = 10⁷", size=th.SIZE_LABEL, color=th.INK_3)
        side = VGroup(k1, k2, cap).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        plot = VGroup(ax, rot, flat)
        bottom = VGroup(plot, side).arrange(RIGHT, buff=0.5)
        col = VGroup(top, bottom).arrange(DOWN, buff=0.6)
        self.place(col)
        with self.narrate(
            "Why leave three quarters of each head unrotated? "
            "Look at the score. "
            "The transform is a rotation on the first 64 channels, and the identity on the other 192. "
                        "The rotated part depends on the gap. "
            "The passthrough part is a pure content match, with no position at all. "
            "Here's the rotated part for a query and key that match in every pair. "
            "Summed over pairs at different speeds, it fades as the gap grows. "
            "That suits local syntax, but not a head hunting for a definition 80,000 tokens back. "
            "The passthrough channels give that head a retrieval path that distance doesn't weaken."
        ) as t:
            t.wait_until("Look at the score")
            self.play(FadeIn(s[:2]), run_time=0.6)
            t.wait_until("The rotated part depends")
            self.play(FadeIn(s[2]), FadeIn(u1), run_time=0.7)
            t.wait_until("The passthrough part is")
            self.play(FadeIn(s[3:]), FadeIn(u2), run_time=0.7)
            t.wait_until("Here's the rotated part")
            self.play(Create(ax), FadeIn(cap), run_time=0.9)
            t.wait_until("Summed over pairs")
            self.play(Create(rot), FadeIn(k1), run_time=2.0)
            t.wait_until("The passthrough channels")
            self.play(Create(flat), FadeIn(k2), run_time=1.2)

    def no_saving(self):
        self.slide("no-saving", "Partial rotary saves arithmetic, not memory", book_section="s-partial-rotary")
        g = kit.grid(4, 32, cell=(0.26, 0.4), gap=0.02, row_labels=["KV head 0", "KV head 1", "KV head 2",
                                                                     "KV head 3"])
        for r in range(4):
            g.paint(g.block(r, r + 1, 0, 8), self.color)
            g.paint(g.block(r, r + 1, 8, 32), PASS_C, tint=0.18)
        cap = kit.label("one cached key per token: all 256 channels of every KV head", size=th.SIZE_LABEL,
                        color=th.INK_3)
        cache = VGroup(cap, g).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        bc = kit.bar_compare(
            [("rotation skipped", 16128, th.GOOD, "about 16,000 FLOPs"),
             ("lm_head", 2.54e9, th.INFO, "2.54 GFLOP")],
            width=6.0, bar_height=0.5, gap=0.35,
        )
        sub = kit.label("per token, per full-attention layer, against the output projection",
                        size=th.SIZE_LABEL, color=th.INK_3)
        flops = VGroup(bc, sub).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        verdict = kit.pill("an architectural choice, not an optimization", self.color, size=th.SIZE_SMALL)
        col = VGroup(cache, flops, verdict).arrange(DOWN, buff=0.55)
        self.place(col)
        with self.narrate(
            "What does partial rotary save? "
            "Not memory. "
            "The cache still stores all 256 channels of every KV head, because the passthrough channels are part "
            "of the key. "
            "It saves arithmetic, and arithmetic was never the bottleneck. "
            "The rotation costs about 3 FLOPs per channel. "
            "Skipping 192 channels across 28 heads saves about 16,000 FLOPs per token, per full-attention layer. "
            "The output projection alone costs 2.54 GFLOP per token. "
            "Treat partial rotary as an architectural choice, not an optimization."
        ) as t:
            t.wait_until("The cache still")
            self.play(FadeIn(cap), LaggedStart(*[FadeIn(g.row(r)) for r in range(4)], lag_ratio=0.2),
                      FadeIn(g.row_labels), run_time=1.2)
            t.wait_until("Skipping 192")
            self.play(FadeIn(bc.labels), GrowFromEdge(bc.bars[0], LEFT), FadeIn(bc.values[0]), FadeIn(sub),
                      run_time=0.9)
            t.wait_until("The output projection")
            self.play(GrowFromEdge(bc.bars[1], LEFT), FadeIn(bc.values[1]), run_time=1.0)
            t.wait_until("Treat partial")
            self.play(FadeIn(verdict, scale=0.9), run_time=0.6)

    # Multimodal RoPE ---------------------------------------------------------------------

    def mrope(self):
        self.slide("mrope", "Three coordinates, one rotation", book_section="s-multimodal-rope")
        g = kit.grid(1, 32, cell=(0.32, 0.6), gap=0.03)
        secs = [(0, 11, "time t", self.colors[0]), (11, 22, "height h", self.colors[1]),
                (22, 32, "width w", self.colors[2])]
        braces, names = VGroup(), VGroup()
        for a, b, name, c in secs:
            br = Brace(g.block(0, 1, a, b), UP, buff=0.1, color=th.INK_4)
            nm = kit.label(f"{name}: {b - a} pairs", size=th.SIZE_LABEL, color=c, weight=MEDIUM)
            nm.next_to(br, UP, buff=0.08)
            braces.add(br)
            names.add(nm)
        cap = kit.label("32 frequency pairs", size=th.SIZE_LABEL, color=th.INK_3)
        cap.next_to(g, DOWN, buff=0.15).align_to(g, LEFT)
        fig = VGroup(g, braces, names, cap)
        text_row = kit.table(
            [["text token at 42", "42", "42", "42", "standard RoPE, exactly"],
             ["image patch", "t", "7", "12", "row and column kept"]],
            header=["", "t", "h", "w", ""], size=th.SIZE_SMALL, align=["l", "c", "c", "c", "l"], col_gap=0.7,
        )
        col = VGroup(fig, text_row).arrange(DOWN, buff=0.8)
        self.place(col)
        with self.narrate(
            "The second departure is multimodal RoPE. "
            "This model accepts images, and an image patch has three coordinates: time, height, and width. "
                        "So multimodal RoPE keeps the same rotation, and changes where each pair reads its angle. "
            "The 32 pairs split into three sections. "
            "The first 11 read time. "
            "The next 11 read height. "
            "The last 10 read width. "
            "For a text token, all three coordinates hold the same number, so every pair reads the same position. "
            "That's standard RoPE exactly, with no special case. "
            "An image patch at row 7, column 12 keeps its place in the picture."
        ) as t:
            self.play(FadeIn(g), FadeIn(cap), run_time=0.7)
            for (a, b, _, c), br, nm, cue in zip(secs, braces, names,
                                                 ["The first 11", "The next 11", "The last 10"]):
                t.wait_until(cue)
                self.play(g.fill(g.block(0, 1, a, b), c), FadeIn(br), FadeIn(nm), run_time=0.6)
            t.wait_until("For a text token")
            self.play(FadeIn(text_row.rules), *[FadeIn(c) for c in text_row.cells[0]],
                      *[FadeIn(c) for c in text_row.cells[1]], run_time=0.9)
            t.wait_until("An image patch")
            self.play(*[FadeIn(c) for c in text_row.cells[2]], run_time=0.8)

    # Positions during decode -----------------------------------------------------------

    def decode_positions(self):
        self.slide("decode-positions", "Decode passes the cache length as its position",
                   book_section="s-positions-during-decode")
        panel = kit.code(POSITIONS, "python")
        good = kit.grid(1, 9, cell=(0.75, 0.6), gap=0.05, values=[[str(i) for i in range(9)]])
        bad = kit.grid(1, 9, cell=(0.75, 0.6), gap=0.05, values=[[str(i) for i in range(6)] + ["0", "0", "0"]])
        good.paint(good.block(0, 1, 0, 6), self.color)
        bad.paint(bad.block(0, 1, 0, 6), self.color)
        n1 = kit.label("correct", size=th.SIZE_SMALL, color=th.GOOD, weight=MEDIUM)
        n2 = kit.label("bug", size=th.SIZE_SMALL, color=th.BAD, weight=MEDIUM)
        r1 = VGroup(n1, good).arrange(RIGHT, buff=0.4)
        r2 = VGroup(n2, bad).arrange(RIGHT, buff=0.4)
        good.align_to(bad, LEFT)
        rows = VGroup(r1, r2).arrange(DOWN, buff=0.5, aligned_edge=RIGHT)
        pre = Brace(good.block(0, 1, 0, 6), UP, buff=0.1, color=th.INK_4)
        pre_t = kit.label("prefill: 0 to n − 1", size=th.SIZE_LABEL, color=th.INK_3).next_to(pre, UP, buff=0.08)
        dec = Brace(good.block(0, 1, 6, 9), UP, buff=0.1, color=th.INK_4)
        dec_t = kit.label("decode: one per step", size=th.SIZE_LABEL, color=th.INK_3).next_to(dec, UP, buff=0.08)
        fig = VGroup(rows, pre, pre_t, dec, dec_t)
        col = VGroup(panel, fig).arrange(DOWN, buff=0.7)
        self.place(col)
        with self.narrate(
                        "Prefill processes the whole prompt at once, so it passes positions 0 through n minus 1. "
            "Decode generates one token per step, and passes a single position: the current cache length. "
            "One expression covers both phases. "
            "The trap is passing position 0 on every decode step. "
            "Then every generated token behaves as if it's the first, and its gap to every cached key is wrong "
            "by a growing amount. "
            "Output degenerates into repetition after a few dozen tokens. "
            "It looks like a sampling bug. "
            "It isn't."
        ) as t:
            t.wait_until("Prefill processes")
            self.play(FadeIn(n1), FadeIn(good.block(0, 1, 0, 6)), FadeIn(pre), FadeIn(pre_t), run_time=0.8)
            t.wait_until("Decode generates")
            for c in range(6, 9):
                self.play(FadeIn(good.cell(0, c), shift=RIGHT * 0.1), good.fill(good.cell(0, c), th.GOOD),
                          run_time=0.4)
            self.play(FadeIn(dec), FadeIn(dec_t), run_time=0.5)
            t.wait_until("One expression")
            self.play(FadeIn(panel.frame), *panel.reveal(1, 2), run_time=0.8)
            t.wait_until("The trap is")
            self.play(FadeIn(n2), FadeIn(bad.block(0, 1, 0, 6)), run_time=0.6)
            for c in range(6, 9):
                self.play(FadeIn(bad.cell(0, c), shift=RIGHT * 0.1), bad.fill(bad.cell(0, c), th.BAD),
                          run_time=0.4)

    def what_goes_wrong(self):
        self.slide("what-goes-wrong", "Most RoPE bugs still produce fluent text", book_section="s-what-goes-wrong")
        bugs = kit.bullets(
            [
                r"\textbf{Wrong split index:} fluent text that ignores word order.",
                r"\textbf{Rotating the values:} position belongs in the score, not the payload.",
                r"\textbf{Rotating cached keys again:} the gap becomes $2m - n$.",
                r"\textbf{Angle table in bfloat16:} long context quietly stops working.",
                r"\textbf{Tables left in float32:} prefill falls off the tensor cores.",
            ],
            size=th.SIZE_BODY, width=12.0, buff=0.45,
        )
        self.place(bugs)
        with self.narrate(
            "Most of these bugs produce fluent text, so they hide. "
            "Split at the wrong index, and the model ignores word order. "
            "Rotate the values, and you rotate the content the model retrieves. "
            "Rotate keys that are already in the cache, and the angle applies twice. "
            "Build the angle table in bfloat16, and long context quietly stops working. "
            "Forget to cast the tables to the query's precision, and prefill slows down several-fold, with nothing "
            "in the logs."
        ) as t:
            for row, cue in zip(bugs, ["Split at the wrong", "Rotate the values", "Rotate keys that",
                                       "Build the angle", "Forget to cast"]):
                t.wait_until(cue)
                self.play(FadeIn(row, shift=RIGHT * 0.1), run_time=0.6)
