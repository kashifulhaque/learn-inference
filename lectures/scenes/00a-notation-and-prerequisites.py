"""Chapter 00a, notation and prerequisites.

Teaches content/chapters/00a-notation-and-prerequisites.md in its order. Every
number below was recomputed in Python (numpy, float32 where it matters) and
matches the chapter. Illustrative values the chapter doesn't fix are marked.

Shapes:
- head_dim = 256; one step along seq jumps 256 elements = 512 bytes in bf16.
- Toy slice on screen (illustrative): 4 positions x 6 channels; a step along
  seq jumps 6 elements.

Similarity vector (illustrative): q = (1, 2, 0, -1); keys (1, 2, 0, -1),
(0, 1, 1, 0), (-1, -2, 0, 1), (2, 0, 1, 0), (0, 0, 1, 0), (1, 1, 0, 0);
K q = (6, 2, -6, 2, 0, 3).

Intensity: 2nmT / (b nm) = 2T / b; with b = 2, T FLOPs per byte; T = 1 gives
1 FLOP per byte.

Softmax:
- Shift demo (illustrative): softmax(3, 1, 0) = (0.844, 0.114, 0.042), the
  same for (5, 3, 2).
- float32 max 3.4028e38; e^88 = 1.65e38 (finite), e^89 = 4.49e38 (overflows).
- Overflow demo (illustrative), x = (92, 90, 87) in float32: e^x = (inf, inf,
  6.08e37), sum inf, naive softmax (nan, nan, 0). Shifted: (0, -2, -5),
  exponentials (1, 0.1353, 0.006738), sum 1.1421, softmax (0.876, 0.118,
  0.006).

Formats: float32 1/8/23, 24 significand bits, max 3.4e38; float16 1/5/10, 11
bits, max 65504; bfloat16 1/8/7, 8 bits, max 3.39e38 (about 3.4e38).
8 x log10(2) = 2.41, "about 2 decimal digits". 32 - 16 = 16 mantissa bits
deleted (23 - 7).

Rounding:
- Toy format p = 3 (illustrative): gaps 2^(e - 2): 0.25 in [1, 2), 0.5 in
  [2, 4), 1 in [4, 8).
- Unit roundoff 2^-24 = 5.96e-8 (6.0e-8), 2^-11 = 4.88e-4 (4.9e-4), 2^-8 =
  3.906e-3 (3.9e-3).
- ULP(256) = 2^(8 - 8 + 1) = 2; neighbors 255 and 258; 257 is the midpoint;
  bf16 round-to-nearest-even gives 256 (checked by bit-level rounding, and
  1000 further additions of 1 stay at 256). bf16 values 252 to 255 have gap 1.
- float32 ratio 1 / 2^24 = 1 / 16,777,216, "about 16.8 million".

GPU (A100 80GB, chapter values): 108 SMs (drawn as 9 x 12 = 108), warp 32,
block up to 1024 threads = 32 warps, registers 256 KB/SM, shared memory up
to 164 KB/SM, L2 40 MB, HBM 80 GB at 1275 GB/s measured, launch 5 to 10 us,
coalesced vs strided 6.9x, 25% occupancy example.

Units: 2^30 = 1,073,741,824; 2^30 / 10^9 = 1.0737 ("about 7%"); 53.8e9 / 2^30
= 50.105 ("50.1 GiB"); 64 x 1024 = 65,536 bytes; 1935 GB/s = 1.935e12 B/s;
80 x 2^30 - 6 x 2^30 = 74 GiB.

Symbols: N = 64, d = 5120, d_ff = 17,408, V = 248,320, b = 2, h = 24,
h_kv = 4, g = 24 / 4 = 6, d_h = 256, d_r = 64.
"""

from manim import *

import kit
from kit import theme as th
from kit.text import pango

SHAPE_SRC = """\
q = q.view(batch, q_len, heads, head_dim).transpose(1, 2)
# q: (batch, heads, q_len, head_dim)
"""

SUM_C = th.HL_A  # \hla in the chapter: the summed index
KEY_C = th.HL_B
VAL_C = th.HL_C
SHIFT_C = th.HL_D  # \hld: the softmax shift
SIGN_C = th.VIOLET
EXP_C = th.INFO


def es(text, size=th.SIZE_HEADING, color=th.INK_2, marks=None):
    """A monospaced string (an einsum spec) whose glyphs can be coloured by index.

    Glyph indices skip spaces, so write the string without spaces.
    """
    t = pango(Text, text, size, font=th.MONO, color=color, disable_ligatures=True)
    for idx, col in (marks or {}).items():
        t[idx].set_color(col)
    return t


def mono_t2c(text, size=th.SIZE_SMALL, color=th.INK_2, t2c=None):
    """One line of code-font text with substrings coloured."""
    return pango(Text, text, size, font=th.MONO, color=color, t2c=t2c or {}, disable_ligatures=True)


def swatch(text, color, size=th.SIZE_LABEL):
    sq = Square(0.26, stroke_color=th.RULE_STRONG, stroke_width=th.STROKE_HAIRLINE,
                fill_color=th.tint(color, 0.35), fill_opacity=1)
    return VGroup(sq, kit.label(text, size=size, color=th.INK_2)).arrange(RIGHT, buff=0.16)


def paint_all(g, color, tint=0.35):
    g.paint(g.block(0, g.rows, 0, g.cols), color, tint=tint)


def tick_line(x0, x1, scale, ticks, labels, y=0.0, label_size=th.SIZE_LABEL):
    """A horizontal number line from x0 to x1 with ticks and labels below."""
    to_x = lambda v: (v - x0) * scale  # noqa: E731
    line = Line([to_x(x0), y, 0], [to_x(x1), y, 0], color=th.INK_3, stroke_width=th.STROKE_AXIS)
    marks = VGroup(*[Line([to_x(v), y - 0.12, 0], [to_x(v), y + 0.12, 0], color=th.INK, stroke_width=3)
                     for v in ticks])
    nums = VGroup()
    for v in labels:
        n = kit.label(str(v).rstrip("0").rstrip(".") if isinstance(v, float) else str(v), size=label_size,
                      color=th.INK_3)
        n.next_to([to_x(v), y - 0.12, 0], DOWN, buff=0.12)
        nums.add(n)
    group = VGroup(line, marks, nums)
    group.to_x = to_x
    group.line, group.marks, group.nums = line, marks, nums
    return group


class Lecture(kit.Lecture):
    def construct(self):
        self.title_slide(
            narration=(
                "Notation and prerequisites. "
                "This chapter is a reference page, so you can dip into it, rather than read it straight through. "
                "In this lecture, you tour the shapes, the einsum strings, the floating-point facts, and the GPU "
                "words that every later chapter assumes."
            )
        )
        self.shapes_memory()
        self.shape_names()
        self.index_rules()
        self.einsum_rules()
        self.einsum_scores()
        self.outer_state()
        self.similarity()
        self.intensity()
        self.cheap_updates()
        self.softmax_shift()
        self.softmax_overflow()
        self.float_formats()
        self.ulp_roundoff()
        self.vanishing_addend()
        self.sm_warp_block()
        self.memory_hierarchy()
        self.coalescing()
        self.units()
        self.symbols()
        self.what_goes_wrong()
        self.closing_slide(
            [
                r"Shapes are named, outermost first; the last dimension is contiguous.",
                r"An einsum letter missing from the output is summed.",
                r"Subtract the max before exponentiating.",
                r"bfloat16: float32's range, about 2 digits. Accumulate in float32.",
            ],
            narration=(
                "Here's what to keep. "
                "Shapes are named outermost first, and the last dimension is contiguous. "
                "An einsum letter missing from the output is summed. "
                "Subtract the maximum before you exponentiate. "
                "The bf16 format has float32's range but about 2 decimal digits, so accumulate in float32. "
                "Next, you open the model's files on disk."
            ),
            book_section="s-what-goes-wrong",
        )

    # Shapes and index conventions ---------------------------------------------

    def shapes_memory(self):
        self.slide("shapes-memory", "The last dimension sits contiguously in memory",
                   book_section="s-shapes-and-index-conventions")
        tup = mono_t2c("(batch, heads, seq, head_dim)", size=th.SIZE_HEADING, t2c={"head_dim": self.color})
        rows, cols = 4, 6
        g = kit.grid(rows, cols, cell=0.55, values=[[str(c) for c in range(cols)] for _ in range(rows)],
                     row_labels=[f"seq {r}" for r in range(rows)])
        for r in range(rows):
            g.paint(g.row(r), self.colors[r], tint=0.3)
        g_cap = kit.label("one batch, one head: 4 positions × 6 channels", size=th.SIZE_LABEL, color=th.INK_3)
        g_cap.next_to(g, UP, buff=0.2)
        strip = kit.grid(1, rows * cols, cell=(0.42, 0.5), values=[[str(i % cols) for i in range(rows * cols)]])
        for i in range(rows * cols):
            strip.paint(strip.cell(0, i), self.colors[i // cols], tint=0.3)
        mem = kit.label("memory, by address →", size=th.SIZE_LABEL, color=th.INK_3)
        mem.next_to(strip, UP, buff=0.15).align_to(strip, LEFT)
        note = kit.prose(r"With head\_dim $= 256$, one step along seq jumps 512 bytes in bf16.",
                         size=th.SIZE_SMALL, width=None)
        lab_row = VGroup(kit.label("along head_dim: consecutive", size=th.SIZE_LABEL, color=th.GOOD),
                         kit.label("along seq: a jump of 6 per step", size=th.SIZE_LABEL, color=th.BAD))
        lab_row.arrange(RIGHT, buff=1.2)
        col = VGroup(tup, VGroup(g_cap, g), VGroup(mem, strip, lab_row).arrange(DOWN, buff=0.15), note)
        lab_row.align_to(strip, LEFT)
        mem.align_to(strip, LEFT)
        col.arrange(DOWN, buff=0.45)
        self.place(col)
        targets = [strip.block(0, 1, r * cols, (r + 1) * cols) for r in range(rows)]
        along = strip.outline(0, 1, cols, 2 * cols, color=th.GOOD)
        along_t = lab_row[0]
        jumps = VGroup(*[strip.outline(0, 1, c, c + 1, color=th.BAD) for c in (2, 8, 14, 20)])
        jumps_t = lab_row[1]
        with self.narrate(
            "Every chapter writes shapes the same way, because shape confusion is the easiest way to get stuck. A "
            "shape is a tuple of named dimensions, outermost first: batch, heads, sequence, and head dim. Here's "
            "one batch and one head, with 4 positions and 6 channels. In memory, the rows sit end to end, so the "
            "rightmost dimension is contiguous. A kernel that reads along that dimension reads consecutive bytes. "
            "A kernel that reads along the sequence jumps a whole head dim per step. With a head dim of 256, "
            "that's 512 bytes in bf16."
        ) as t:
            t.wait_until("A shape is a tuple")
            self.play(FadeIn(tup, shift=DOWN * 0.1), run_time=0.8)
            t.wait_until("Here's one batch")
            self.play(FadeIn(g), FadeIn(g_cap), run_time=0.8)
            t.wait_until("In memory, the rows")
            self.play(FadeIn(mem), run_time=0.3)
            for r in range(rows):
                self.play(TransformFromCopy(g.row(r), targets[r]), run_time=0.6)
            t.wait_until("A kernel that reads along that")
            self.play(Create(along), FadeIn(along_t), run_time=0.8)
            t.wait_until("A kernel that reads along the sequence")
            self.play(LaggedStart(*[Create(j) for j in jumps], lag_ratio=0.3), FadeIn(jumps_t), run_time=1.2)
            t.wait_until("With a head dim of 256")
            self.play(FadeIn(note, shift=UP * 0.1), run_time=0.7)

    def shape_names(self):
        self.slide("shape-names", "The same names in every chapter", book_section="s-shapes-and-index-conventions")
        tab = kit.table(
            [
                [r"\texttt{batch}, $B$", "independent sequences processed together"],
                [r"\texttt{q\_len}", "queries this call computes"],
                [r"\texttt{kv\_len}", "keys those queries attend to"],
                [r"\texttt{heads}, \texttt{kv\_heads}", "attention heads; they differ in this model"],
                [r"\texttt{head\_dim}", "width of one head: 256"],
                [r"\texttt{hidden}", "width of the residual stream: 5120"],
            ],
            header=["Name", "Meaning"],
            size=th.SIZE_SMALL, row_gap=0.22,
        )
        decode = kit.pill("decode: q_len = 1, kv_len = the whole context", self.color, size=th.SIZE_LABEL)
        panel = kit.code(SHAPE_SRC, "python")
        col = VGroup(tab, decode, panel).arrange(DOWN, buff=0.4)
        self.place(col)
        band = kit.mark(VGroup(*tab.cells[2], *tab.cells[3]), buff=0.08)
        with self.narrate(
            "The same names appear in every chapter. Batch counts the sequences processed together. The query "
            "length is how many queries this call computes, and the key length is how many keys they attend to. "
            "In decode, which generates one token per step, the query length is 1 and the key length is the whole "
            "context. Query heads and KV heads differ in this model. One head is 256 wide, and the residual "
            "stream is 5120 wide. When a listing writes a shape in a comment, that shape is exact."
        ) as t:
            self.play(FadeIn(tab.rules), *[FadeIn(c) for c in tab.cells[0]], *[FadeIn(c) for c in tab.cells[1]],
                      run_time=0.8)
            t.wait_until("The query length is")
            self.play(*[FadeIn(c) for c in tab.cells[2] + tab.cells[3]], run_time=0.8)
            t.wait_until("In decode")
            self.play(FadeIn(band), FadeIn(decode, scale=0.9), run_time=0.8)
            t.wait_until("Query heads and KV")
            self.play(*[FadeIn(c) for c in tab.cells[4]], run_time=0.6)
            t.wait_until("One head is 256")
            self.play(*[FadeIn(c) for c in tab.cells[5] + tab.cells[6]], run_time=0.7)
            t.wait_until("When a listing")
            self.play(FadeIn(panel.frame), *panel.reveal(1), run_time=0.8)
            self.play(*panel.reveal(2), panel.highlight(2), run_time=0.8)

    def index_rules(self):
        self.slide("index-rules", "Vectors are columns", book_section="s-shapes-and-index-conventions")
        rules = VGroup(
            kit.prose(r"$Q[b,h,i,c]$: one element of a tensor.", size=th.SIZE_SMALL, width=5.2),
            kit.prose(r"$q_t$: the query vector at position $t$, of length \texttt{head\_dim}.",
                      size=th.SIZE_SMALL, width=5.2),
            kit.prose(r"$K$: the whole tensor, shape (\texttt{kv\_len}, \texttt{head\_dim}), one key per row.",
                      size=th.SIZE_SMALL, width=5.2),
        ).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        c = 0.5
        q_row = kit.grid(1, 4, cell=c)
        k_col = kit.grid(4, 1, cell=c)
        dot = kit.grid(1, 1, cell=c)
        paint_all(q_row, KEY_C)
        paint_all(k_col, VAL_C)
        paint_all(dot, th.VIOLET)
        eq1 = VGroup(q_row, kit.formula(r"\cdot"), k_col, kit.formula(r"="), dot).arrange(RIGHT, buff=0.25)
        lab1 = kit.prose(r"$q_t^\top k_j$: row times column, a scalar", size=th.SIZE_SMALL, width=None)
        k2 = kit.grid(4, 1, cell=c)
        v_row = kit.grid(1, 4, cell=c)
        outer = kit.grid(4, 4, cell=c)
        paint_all(k2, VAL_C)
        paint_all(v_row, th.GOOD)
        paint_all(outer, th.VIOLET)
        eq2 = VGroup(k2, kit.formula(r"\cdot"), v_row, kit.formula(r"="), outer).arrange(RIGHT, buff=0.25)
        lab2 = kit.prose(r"$k_j v_j^\top$: column times row, a matrix", size=th.SIZE_SMALL, width=None)
        pics = VGroup(VGroup(lab1, eq1).arrange(DOWN, buff=0.3), VGroup(lab2, eq2).arrange(DOWN, buff=0.3))
        pics.arrange(DOWN, buff=0.6)
        layout = VGroup(rules, pics).arrange(RIGHT, buff=0.9)
        self.place(layout)
        with self.narrate(
            "Indices follow three rules. A bracketed subscript indexes a tensor, so Q with four indices is one "
            "element. A plain subscript names a position, so q t is the query vector at position t. And vectors "
            "are columns. So q t transpose times k j is a row times a column: a single number. Flip the order, "
            "and k j times v j transpose is a whole matrix. A capital letter with no indices is the whole tensor, "
            "like K, with one key per row."
        ) as t:
            t.wait_until("A bracketed subscript")
            self.play(FadeIn(rules[0], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("A plain subscript")
            self.play(FadeIn(rules[1], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("So q t transpose")
            self.play(FadeIn(lab1), FadeIn(eq1[:4]), run_time=0.8)
            self.play(TransformFromCopy(VGroup(q_row, k_col), dot), run_time=0.8)
            t.wait_until("Flip the order")
            self.play(FadeIn(lab2), FadeIn(eq2[:4]), run_time=0.8)
            self.play(TransformFromCopy(VGroup(k2, v_row), outer), run_time=1.0)
            t.wait_until("A capital letter")
            self.play(FadeIn(rules[2], shift=RIGHT * 0.1), run_time=0.6)

    # Einstein summation -------------------------------------------------------

    def einsum_rules(self):
        self.slide("einsum-rules", "A letter missing from the output is summed", book_section="s-einstein-summation")
        rules = kit.bullets(
            [
                r"Each input gets one letter per dimension, in order.",
                r"A letter in the inputs but not the output is \textbf{summed}.",
                r"A letter in the output is \textbf{kept}, in the output's order.",
            ],
            size=th.SIZE_SMALL, width=4.9,
        )
        spec = es("td,dk->tk", marks={1: SUM_C, 3: SUM_C})
        c = 0.42
        x = kit.grid(3, 4, cell=c)
        w = kit.grid(4, 3, cell=c)
        y = kit.grid(3, 3, cell=c)
        mats = VGroup(x, kit.formula(r"\cdot"), w, kit.formula(r"="), y).arrange(RIGHT, buff=0.3)
        caps = VGroup(
            kit.prose(r"$x$: $(T, d)$", size=th.SIZE_LABEL, width=None),
            kit.prose(r"$W$: $(d, k)$", size=th.SIZE_LABEL, width=None),
            kit.prose(r"$Y$: $(T, k)$", size=th.SIZE_LABEL, width=None),
        )
        for cap, m in zip(caps, (x, w, y)):
            cap.next_to(m, DOWN, buff=0.18)
        f = kit.formula(r"Y[t,k] = \sum_{c=0}^{d-1} x[t,", r"c", r"]\, W[", r"c", r",k]", size=th.SIZE_MATH_SMALL)
        f[1].set_color(SUM_C)
        f[3].set_color(SUM_C)
        right = VGroup(spec, VGroup(mats, caps), f).arrange(DOWN, buff=0.5)
        layout = VGroup(rules, right).arrange(RIGHT, buff=0.8)
        self.place(layout)
        with self.narrate(
            "Chapters 6 and 7 write contractions with einsum, which names indices instead of arranging "
            "transposes. It has three rules. Each input gets one letter per dimension, in order. A letter that "
            "appears in the inputs but not in the output is summed over. A letter in the output is kept, in the "
            "output's order. Take a matrix multiply, where x carries t and d, and W carries d and k. The output "
            "keeps t and k. The letter d is missing, so it's summed: a row of x times a column of W."
        ) as t:
            for i, cue in enumerate(["Each input gets", "A letter that appears", "A letter in the output"]):
                t.wait_until(cue)
                self.play(FadeIn(rules[i], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("Take a matrix")
            self.play(FadeIn(spec[:5]), FadeIn(mats[:3]), FadeIn(caps[:2]), run_time=0.8)
            t.wait_until("The output keeps")
            self.play(FadeIn(spec[5:]), FadeIn(mats[3:]), FadeIn(caps[2]), run_time=0.8)
            t.wait_until("The letter d is missing")
            self.play(Indicate(spec[1], color=SUM_C), Indicate(spec[3], color=SUM_C), FadeIn(f), run_time=0.8)
            self.play(x.fill(x.row(1), KEY_C), w.fill(w.col(2), VAL_C), run_time=0.5)
            for cc in range(4):
                self.play(x.fill(x.cell(1, cc), SUM_C, tint=0.6), w.fill(w.cell(cc, 2), SUM_C, tint=0.6),
                          run_time=0.3)
            self.play(y.fill(y.cell(1, 2), SUM_C, tint=0.6), run_time=0.5)

    def einsum_scores(self):
        self.slide("einsum-scores", "Attention scores: b and h ride along, d is summed",
                   book_section="s-einstein-summation")
        carried = {0: th.INFO, 1: th.INFO, 5: th.INFO, 6: th.INFO, 11: th.INFO, 12: th.INFO}
        spec = es("bhid,bhjd->bhij", marks={**carried, 3: SUM_C, 8: SUM_C})
        n, dd, c = 5, 4, 0.42
        q = kit.grid(n, dd, cell=c, row_labels=[f"i={i}" for i in range(n)])
        k = kit.grid(n, dd, cell=c, row_labels=[f"j={j}" for j in range(n)])
        s = kit.grid(n, n, cell=c)
        paint_all(q, KEY_C, tint=0.2)
        paint_all(k, VAL_C, tint=0.2)
        names = [kit.prose(r"$Q$: (seq, $d_h$)", size=th.SIZE_LABEL, width=None),
                 kit.prose(r"$K$: (seq, $d_h$)", size=th.SIZE_LABEL, width=None),
                 kit.prose(r"scores: (seq, seq)", size=th.SIZE_LABEL, width=None)]
        blocks = VGroup(*[VGroup(nm, g).arrange(DOWN, buff=0.2) for nm, g in zip(names, (q, k, s))])
        blocks.arrange(RIGHT, buff=0.9, aligned_edge=DOWN)
        f = kit.formula(r"\mathrm{scores}[b,h,i,j] = \sum_{c=0}^{d_h-1} Q[b,h,i,", r"c", r"]\, K[b,h,j,", r"c",
                        r"]", size=th.SIZE_MATH_SMALL)
        f[1].set_color(SUM_C)
        f[3].set_color(SUM_C)
        tip = kit.prose(r"\texttt{einsum} for clarity, \texttt{matmul} in the inner loop.", size=th.SIZE_SMALL,
                        width=None, color=th.INK_3)
        col = VGroup(spec, blocks, f, tip).arrange(DOWN, buff=0.4)
        self.place(col)
        with self.narrate(
            "Attention scores follow the same rules. Both inputs have the shape batch, heads, sequence, head dim. "
            "The letters b and h appear in every term, so they ride along as batch dimensions. Only d is missing "
            "from the output, so only d is summed. Each score is a dot product between query i and key j, for "
            "every pair. One caution: einsum doesn't promise a fast kernel, so use matmul in the inner loop."
        ) as t:
            self.play(FadeIn(spec), run_time=0.6)
            t.wait_until("Both inputs have")
            self.play(FadeIn(blocks[0]), FadeIn(blocks[1]), run_time=0.8)
            t.wait_until("The letters b and h")
            self.play(*[Indicate(spec[i], color=th.INFO) for i in carried], run_time=0.9)
            t.wait_until("Only d is missing")
            self.play(Indicate(spec[3], color=SUM_C), Indicate(spec[8], color=SUM_C), FadeIn(f), FadeIn(blocks[2]),
                      run_time=0.9)
            t.wait_until("Each score is a dot")
            self.play(q.fill(q.row(1), KEY_C, tint=0.55), k.fill(k.row(3), VAL_C, tint=0.55),
                      s.fill(s.cell(1, 3), SUM_C, tint=0.7), run_time=0.7)
            self.play(LaggedStart(*[s.fill(s.cell(i, j), SUM_C, tint=0.3) for i in range(n) for j in range(n)
                                    if (i, j) != (1, 3)], lag_ratio=0.04), run_time=1.6)
            t.wait_until("One caution")
            self.play(FadeIn(tip, shift=UP * 0.1), run_time=0.7)

    def outer_state(self):
        self.slide("outer-state", "Outer products pack many pairs into one matrix",
                   book_section="s-the-linear-algebra-that-keeps-coming-back")
        spec = es("hk,hv->hkv", size=th.SIZE_BODY)
        none = kit.label("no letter missing: nothing summed", size=th.SIZE_LABEL, color=th.INK_3)
        head = VGroup(spec, none).arrange(RIGHT, buff=0.4)
        c = 0.5
        kc = kit.grid(4, 1, cell=c)
        vr = kit.grid(1, 4, cell=c)
        m = kit.grid(4, 4, cell=c)
        paint_all(kc, KEY_C)
        paint_all(vr, VAL_C)
        vr.next_to(m, UP, buff=0.25)
        kc.next_to(m, LEFT, buff=0.25)
        kl = kit.math_label("k", size=th.SIZE_BODY, color=KEY_C).next_to(kc, LEFT, buff=0.15)
        vl = kit.math_label(r"v^\top", size=th.SIZE_BODY, color=VAL_C).next_to(vr, UP, buff=0.12)
        rank = kit.label("rank 1: every column a multiple of k", size=th.SIZE_LABEL, color=th.INK_3)
        rank.next_to(m, DOWN, buff=0.2)
        pic = VGroup(kc, vr, m, kl, vl, rank)
        f1 = kit.formula(r"S = \sum_j k_j v_j^\top", size=th.SIZE_MATH_SMALL)
        f2 = kit.formula(r"S^\top q = \sum_j", r"(q^\top k_j)", r"\,", r"v_j", size=th.SIZE_MATH)
        f2[1].set_color(KEY_C)
        f2[3].set_color(VAL_C)
        w_lab = kit.label("similarity weight", size=th.SIZE_LABEL, color=KEY_C).next_to(f2, DOWN, buff=0.15)
        w_lab.set_x(f2[1].get_x())
        v_lab = kit.label("stored value", size=th.SIZE_LABEL, color=VAL_C).next_to(f2, UP, buff=0.15)
        v_lab.set_x(f2[3].get_x())
        whole = kit.pill("the whole of linear attention", self.color, size=th.SIZE_SMALL)
        right = VGroup(f1, VGroup(f2, w_lab, v_lab), whole).arrange(DOWN, buff=0.55)
        body = VGroup(pic, right).arrange(RIGHT, buff=1.0)
        col = VGroup(head, body).arrange(DOWN, buff=0.5)
        self.place(col)
        with self.narrate(
            "Some einsums sum nothing. With k and v per head, every letter survives, so the result is larger than "
            "either input. Entry i, j is k i times v j: an outer product. Every column is a multiple of k, so it "
            "has rank 1. Add up outer products over many tokens, and you pack many key-value pairs into one "
            "fixed-size matrix, S. Read it back with a query, and you get a blend of the stored values. Each "
            "value is weighted by how similar its key is to the query. That identity is the whole of linear "
            "attention."
        ) as t:
            self.play(FadeIn(head), run_time=0.7)
            t.wait_until("Entry i, j is")
            self.play(FadeIn(kc), FadeIn(vr), FadeIn(kl), FadeIn(vl), FadeIn(m), run_time=0.7)
            self.play(LaggedStart(*[m.fill(m.col(j), th.VIOLET, tint=0.12 + 0.12 * j) for j in range(4)],
                                  lag_ratio=0.3), run_time=1.2)
            t.wait_until("Every column is")
            self.play(FadeIn(rank), run_time=0.6)
            t.wait_until("Add up outer products")
            self.play(FadeIn(f1, shift=DOWN * 0.1), run_time=0.7)
            for tint in (0.3, 0.45, 0.6):
                self.play(m.fill(m.block(0, 4, 0, 4), th.VIOLET, tint=tint), run_time=0.35)
            t.wait_until("Read it back")
            self.play(FadeIn(f2[0]), FadeIn(f2[2:]), FadeIn(v_lab), run_time=0.8)
            t.wait_until("Each value is weighted")
            self.play(FadeIn(f2[1]), FadeIn(w_lab), run_time=0.7)
            t.wait_until("That identity")
            self.play(FadeIn(whole, scale=0.9), run_time=0.6)

    def similarity(self):
        self.slide("similarity", "One matrix multiply scores every key",
                   book_section="s-the-linear-algebra-that-keeps-coming-back")
        keys = [[1, 2, 0, -1], [0, 1, 1, 0], [-1, -2, 0, 1], [2, 0, 1, 0], [0, 0, 1, 0], [1, 1, 0, 0]]
        qv = [1, 2, 0, -1]
        out = [sum(a * b for a, b in zip(k, qv)) for k in keys]  # 6, 2, -6, 2, 0, 3
        c = (0.6, 0.5)
        K = kit.grid(6, 4, cell=c, values=[[str(v) for v in k] for k in keys],
                     row_labels=[f"k{j + 1}" for j in range(6)])
        q = kit.grid(4, 1, cell=c, values=[[str(v)] for v in qv])
        o = kit.grid(6, 1, cell=c, values=[[str(v)] for v in out])
        paint_all(q, th.VIOLET, tint=0.25)
        for row_ in K.cells:
            for cell_ in row_:
                cell_.text.set_z_index(2)
        for j, v in enumerate(out):
            if v > 0:
                o.paint(o.cell(j, 0), th.GOOD, tint=0.12 * v)
            elif v < 0:
                o.paint(o.cell(j, 0), th.BAD, tint=-0.07 * v)
        k_lab = kit.prose(r"$K$: (\texttt{kv\_len}, \texttt{head\_dim})", size=th.SIZE_LABEL, width=None)
        q_lab = kit.math_label("q", size=th.SIZE_BODY, color=th.VIOLET)
        o_lab = kit.prose(r"$Kq$: one score per key", size=th.SIZE_LABEL, width=None)
        k_blk = VGroup(k_lab, K).arrange(DOWN, buff=0.2)
        q_blk = VGroup(q_lab, q).arrange(DOWN, buff=0.2)
        o_blk = VGroup(o_lab, o).arrange(DOWN, buff=0.2)
        times = kit.formula(r"\cdot")
        eq = kit.formula(r"=")
        fig = VGroup(k_blk, times, q_blk, eq, o_blk).arrange(RIGHT, buff=0.4)
        for mob, anchor in ((q_blk, q), (o_blk, o), (times, times), (eq, eq)):
            mob.shift(UP * (K.get_y() - anchor.get_y()))
        fig.add(kit.label("aligned", size=th.SIZE_LABEL, color=th.GOOD).next_to(o.cell(0, 0), RIGHT, buff=0.2))
        fig.add(kit.label("opposed", size=th.SIZE_LABEL, color=th.BAD).next_to(o.cell(2, 0), RIGHT, buff=0.2))
        notes = fig[-2:]
        f = kit.formula(r"(Kq)_j = q^\top k_j", r"\qquad", r"q^\top K^\top", r"\text{ as a row}",
                        size=th.SIZE_MATH_SMALL)
        line = kit.prose(r"Attention is a similarity search expressed as a matrix multiply.", size=th.SIZE_SMALL,
                         width=None)
        col = VGroup(f, fig, line).arrange(DOWN, buff=0.5)
        self.place(col)
        with self.narrate(
            "The second fact: q transpose K transpose is a similarity vector. "
            "K holds one key per row, so K times q has one entry per key. "
            "Entry j is q transpose k j. "
            "It's large and positive when the query and the key point the same way. "
            "It's negative when they're opposed. "
            "Softmax then turns those numbers into weights. "
            "Attention is a similarity search, expressed as a matrix multiply."
        ) as t:
            self.play(FadeIn(k_blk), FadeIn(q_blk), FadeIn(times), run_time=0.8)
            t.wait_until("Entry j is")
            self.play(FadeIn(f), FadeIn(eq), FadeIn(o_lab), FadeIn(o.squares()), run_time=0.7)
            for j in range(6):
                self.play(K.fill(K.row(j), KEY_C, tint=0.4), FadeIn(o.cell(j, 0).text), run_time=0.3)
                self.play(K.fill(K.row(j), KEY_C, tint=0.0), run_time=0.15)
            t.wait_until("It's large and positive")
            self.play(FadeIn(notes[0]), Indicate(o.cell(0, 0), color=th.GOOD), run_time=0.7)
            t.wait_until("It's negative")
            self.play(FadeIn(notes[1]), Indicate(o.cell(2, 0), color=th.BAD), run_time=0.7)
            t.wait_until("Attention is a similarity")
            self.play(FadeIn(line, shift=UP * 0.1), run_time=0.7)

    def intensity(self):
        self.slide("intensity", "The matrix is read once, however many vectors use it",
                   book_section="s-the-linear-algebra-that-keeps-coming-back")
        n, m, T = 4, 5, 4
        c = 0.5
        W = kit.grid(n, m, cell=c)
        paint_all(W, self.color, tint=0.25)
        X = kit.grid(m, T, cell=c)
        w_lab = kit.prose(r"$n \times m$ matrix, $b$ bytes per entry", size=th.SIZE_LABEL, width=None)
        x_lab = kit.prose(r"$T$ vectors", size=th.SIZE_LABEL, width=None)
        w_blk = VGroup(w_lab, W).arrange(DOWN, buff=0.2)
        x_blk = VGroup(x_lab, X).arrange(DOWN, buff=0.2)
        fig = VGroup(w_blk, kit.formula(r"\cdot"), x_blk).arrange(RIGHT, buff=0.4)
        fig[1].move_to([fig[1].get_x(), W.get_y(), 0])

        def counter(tv):
            return VGroup(
                kit.formula(rf"T = {tv}", size=th.SIZE_MATH_SMALL),
                kit.prose(rf"FLOPs: $2nm \times {tv}$", size=th.SIZE_SMALL, width=None),
                kit.prose(r"bytes read: $b \cdot nm$", size=th.SIZE_SMALL, width=None),
                kit.prose(rf"FLOPs per byte at $b = 2$: \textbf{{{tv}}}", size=th.SIZE_SMALL, width=None,
                          color=self.color),
            ).arrange(DOWN, buff=0.28, aligned_edge=LEFT)

        cnt = counter(1)
        top = VGroup(fig, cnt).arrange(RIGHT, buff=1.0)
        d = kit.formula(r"\text{intensity} \approx \frac{2nm", r"T", r"}{", r"b", r"\cdot nm} =", r"\frac{2", r"T",
                        r"}{", r"b", r"}", size=th.SIZE_MATH)
        for i in (1, 6):
            d[i].set_color(SUM_C)
        for i in (3, 8):
            d[i].set_color(KEY_C)
        box = SurroundingRectangle(d[5:], color=self.color, buff=0.12, stroke_width=3)
        col = VGroup(top, VGroup(d, box)).arrange(DOWN, buff=0.6)
        self.place(col)
        slot = cnt.get_corner(UL)
        with self.narrate(
            "Matrix times vector and matrix times matrix cost the same per weight, but not per byte. An n by m "
            "matrix times one vector costs 2 n m FLOPs, and reads n m entries. Times T vectors, it costs T times "
            "the FLOPs, but still reads the matrix once. So the arithmetic per byte grows with T. With 2 bytes "
            "per entry in bf16, it's T FLOPs per byte. Decoding one token at a time sits at 1 FLOP per byte, "
            "which wastes a GPU. Batching raises T, which is why batching fixes it."
        ) as t:
            self.play(FadeIn(w_blk), FadeIn(fig[1]), FadeIn(x_lab), FadeIn(X.col(0)), run_time=0.8)
            self.play(FadeIn(cnt), X.fill(X.col(0), SUM_C), run_time=0.6)
            t.wait_until("Times T vectors")
            for tv in range(2, T + 1):
                new = counter(tv)
                new.shift(slot - new.get_corner(UL))
                self.play(FadeIn(X.col(tv - 1)), X.fill(X.col(tv - 1), SUM_C), Transform(cnt, new),
                          Indicate(W, color=self.color, scale_factor=1.02), run_time=0.8)
            t.wait_until("So the arithmetic per byte")
            self.play(FadeIn(d), run_time=0.8)
            t.wait_until("With 2 bytes per entry")
            self.play(Create(box), run_time=0.6)
            t.wait_until("Decoding one token")
            new = counter(1)
            new.shift(slot - new.get_corner(UL))
            self.play(Transform(cnt, new), *[X.col(j).animate.set_opacity(0.15) for j in range(1, T)],
                      run_time=0.8)
            t.wait_until("Batching raises T")
            new = counter(T)
            new.shift(slot - new.get_corner(UL))
            self.play(Transform(cnt, new), *[X.col(j).animate.set_opacity(1) for j in range(1, T)], run_time=0.8)

    def cheap_updates(self):
        self.slide("cheap-updates", "Rank-one updates and triangular solves stay cheap",
                   book_section="s-the-linear-algebra-that-keeps-coming-back")
        c = 0.5
        S = kit.grid(4, 5, cell=c)
        paint_all(S, th.INFO, tint=0.15)
        f1 = kit.formula(r"S \leftarrow S + k v^\top", size=th.SIZE_MATH_SMALL)
        l1 = kit.prose(r"$nm$ multiply-adds, no factorization.", size=th.SIZE_SMALL, width=None)
        l1b = kit.prose(r"Chapter 6: one per token.", size=th.SIZE_SMALL, width=None, color=th.INK_3)
        h1 = kit.label("Rank-one update", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
        left = VGroup(h1, f1, S, l1, l1b).arrange(DOWN, buff=0.35)
        L = kit.grid(4, 4, cell=c, values=[["", "", "", ""]] * 4)
        for r in range(4):
            for cc in range(4):
                if cc <= r:
                    L.paint(L.cell(r, cc), self.color, tint=0.25)
                else:
                    L.cell(r, cc).square.set_stroke(opacity=0.3)
        x = kit.grid(4, 1, cell=c, values=[[f"x{i + 1}"] for i in range(4)])
        b = kit.grid(4, 1, cell=c, values=[[f"b{i + 1}"] for i in range(4)])
        for g_ in (x, b):
            for r in range(4):
                g_.cell(r, 0).text.set_opacity(0).set_z_index(2)
        tx = VGroup(L, kit.formula(r"\cdot"), x, kit.formula(r"="), b).arrange(RIGHT, buff=0.25)
        h2 = kit.label("Triangular solve", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
        f2 = kit.formula(r"T x = b", size=th.SIZE_MATH_SMALL)
        costs = VGroup(
            kit.prose(r"Forward substitution: $O(n^2)$.", size=th.SIZE_SMALL, width=None, color=th.GOOD),
            kit.prose(r"Explicit inverse: $O(n^3)$, less accurate.", size=th.SIZE_SMALL, width=None, color=th.BAD),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        right = VGroup(h2, f2, tx, costs).arrange(DOWN, buff=0.35)
        layout = VGroup(left, right).arrange(RIGHT, buff=1.4, aligned_edge=UP)
        self.place(layout)
        k_col = VGroup(*[S.cell(r, 0) for r in range(4)])
        with self.narrate(
            "Two facts are about cost. Adding k v transpose to an n by m matrix costs n m multiply-adds, with no "
            "factorization. Chapter 6's delta rule applies one of these rank-one updates per token. Last, a "
            "triangular system doesn't need an inverse. With T lower triangular, you read off x 1, substitute, "
            "read off x 2, and so on. That's forward substitution, at order n squared operations. Forming the "
            "inverse costs order n cubed, and loses accuracy."
        ) as t:
            self.play(FadeIn(h1), FadeIn(f1), FadeIn(S), run_time=0.7)
            t.wait_until("Adding k v transpose")
            self.play(LaggedStart(*[S.fill(S.row(r), th.VIOLET, tint=0.25 + 0.1 * r) for r in range(4)],
                                  lag_ratio=0.2), run_time=1.2)
            self.play(FadeIn(l1), Indicate(k_col, color=th.VIOLET, scale_factor=1.02), run_time=0.7)
            t.wait_until("Chapter 6's delta rule")
            self.play(FadeIn(l1b), run_time=0.5)
            t.wait_until("Last, a triangular")
            self.play(FadeIn(h2), FadeIn(f2), FadeIn(tx), run_time=0.8)
            t.wait_until("With T lower triangular")
            for r in range(4):
                self.play(L.fill(VGroup(*[L.cell(r, cc) for cc in range(r + 1)]), self.color, tint=0.6),
                          b.cell(r, 0).text.animate.set_opacity(1), run_time=0.35)
                self.play(x.fill(x.cell(r, 0), th.GOOD, tint=0.4), x.cell(r, 0).text.animate.set_opacity(1),
                          L.fill(VGroup(*[L.cell(r, cc) for cc in range(r + 1)]), self.color, tint=0.25),
                          run_time=0.35)
            t.wait_until("That's forward substitution")
            self.play(FadeIn(costs[0]), run_time=0.6)
            t.wait_until("Forming the inverse")
            self.play(FadeIn(costs[1]), run_time=0.6)

    # Softmax ----------------------------------------------------------------------

    def softmax_shift(self):
        self.slide("softmax-shift", "Shifting every score changes nothing",
                   book_section="s-softmax-and-why-every-implementation-subtracts-the-max")
        defn = kit.formula(r"\mathrm{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}", size=th.SIZE_MATH_SMALL)
        shift = kit.formula(
            r"\mathrm{softmax}(x + ", r"c", r")_i = \frac{e^{x_i +", r"c", r"}}{\sum_j e^{x_j +", r"c", r"}} = "
            r"\frac{", r"e^{c}", r"e^{x_i}}{", r"e^{c}", r"\sum_j e^{x_j}} = \mathrm{softmax}(x)_i",
            size=th.SIZE_MATH_SMALL,
        )
        for i in (1, 3, 5, 7, 9):
            shift[i].set_color(SHIFT_C)
        scores = [3, 1, 0]
        probs = [0.844, 0.114, 0.042]
        unit_s, unit_p, bw = 0.38, 2.4, 0.7
        base = 0.0

        def bars(values, unit, color, texts):
            g = VGroup()
            labs = VGroup()
            for i, v in enumerate(values):
                r = Rectangle(width=bw, height=max(v * unit, 0.02), fill_color=color, fill_opacity=0.8,
                              stroke_width=0)
                r.move_to([i * (bw + 0.35), base + r.height / 2, 0])
                lab = kit.label(texts[i], size=th.SIZE_LABEL, color=th.INK_2)
                lab.next_to(r, UP, buff=0.1)
                g.add(r)
                labs.add(lab)
            return g, labs

        sb, sl = bars(scores, unit_s, KEY_C, [str(v) for v in scores])
        sb2, sl2 = bars([v + 2 for v in scores], unit_s, KEY_C, [str(v + 2) for v in scores])
        pb, pl = bars(probs, unit_p, self.color, [f"{p:.3f}" for p in probs])
        axis_s = Line([-0.5, 0, 0], [2 * (bw + 0.35) + 0.5, 0, 0], color=th.INK_3, stroke_width=th.STROKE_AXIS)
        axis_p = axis_s.copy()
        ts = kit.label("scores", size=th.SIZE_LABEL, color=th.INK_3).next_to(axis_s, DOWN, buff=0.15)
        tp = kit.label("probabilities", size=th.SIZE_LABEL, color=th.INK_3).next_to(axis_p, DOWN, buff=0.15)
        left = VGroup(axis_s, sb, sl, sb2, sl2, ts)
        right = VGroup(axis_p, pb, pl, tp)
        right.next_to(left, RIGHT, buff=2.2)
        right.shift(UP * (axis_s.get_y() - axis_p.get_y()))
        ay = axis_s.get_y() + 1.0
        arrow = Arrow([left.get_right()[0] + 0.3, ay, 0], [right.get_left()[0] - 0.3, ay, 0],
                      color=th.INK_3, stroke_width=3, buff=0, tip_length=0.18)
        arrow_t = kit.label("softmax", size=th.SIZE_LABEL, color=th.INK_3).next_to(arrow, UP, buff=0.1)
        charts = VGroup(left, right, arrow, arrow_t)
        c_note = kit.label("+ c = 2", size=th.SIZE_SMALL, color=SHIFT_C, weight=MEDIUM)
        col = VGroup(defn, shift, charts).arrange(DOWN, buff=0.55)
        self.place(col)
        c_note.next_to(sb2, LEFT, buff=0.3).shift(UP * 0.3)
        strikes = VGroup(*[Line(shift[i].get_corner(DL), shift[i].get_corner(UR), color=SHIFT_C, stroke_width=4)
                           for i in (7, 9)])
        sb2.set_opacity(0)
        with self.narrate(
            "Attention and sampling both run softmax. Softmax turns scores into probabilities: positive, and "
            "summing to 1. Here are three scores, and their probabilities. Now add the same constant c to every "
            "score. The probabilities don't move. To see why, factor e to the c out of the numerator and the "
            "denominator. It cancels. Mathematically, the shift changes nothing. Numerically, it changes "
            "everything."
        ) as t:
            t.wait_until("Softmax turns")
            self.play(FadeIn(defn), run_time=0.7)
            t.wait_until("Here are three")
            self.play(FadeIn(axis_s), FadeIn(ts), *[GrowFromEdge(b_, DOWN) for b_ in sb], FadeIn(sl), run_time=0.8)
            self.play(GrowArrow(arrow), FadeIn(arrow_t), FadeIn(axis_p), FadeIn(tp),
                      *[GrowFromEdge(b_, DOWN) for b_ in pb], FadeIn(pl), run_time=0.9)
            t.wait_until("Now add the same")
            sb2.set_opacity(0.8)
            self.play(Transform(sb, sb2), Transform(sl, sl2), FadeIn(c_note), run_time=1.0)
            t.wait_until("The probabilities don't")
            self.play(Indicate(pb, color=self.color, scale_factor=1.05), run_time=0.9)
            t.wait_until("To see why")
            self.play(FadeIn(shift[:7]), run_time=0.9)
            self.play(FadeIn(shift[7:]), run_time=0.9)
            t.wait_until("It cancels")
            self.play(Create(strikes), run_time=0.6)

    def softmax_overflow(self):
        self.slide("softmax-overflow", "Subtract the max, or the exponentials overflow",
                   book_section="s-softmax-and-why-every-implementation-subtracts-the-max")
        tab = kit.table(
            [
                [r"\textbf{Naive}", "", "", ""],
                [r"scores $x$", "92", "90", "87"],
                [r"$e^{x}$", r"\texttt{inf}", r"\texttt{inf}", r"$6.1 \times 10^{37}$"],
                ["softmax", r"\texttt{nan}", r"\texttt{nan}", "0"],
                [r"\textbf{Shifted by the max}", "", "", ""],
                [r"$x - \max x$", "0", r"$-2$", r"$-5$"],
                [r"$e^{x - \max x}$", "1", "0.135", "0.0067"],
                ["softmax", "0.876", "0.118", "0.006"],
            ],
            align=["l", "r", "r", "r"], size=th.SIZE_SMALL, row_gap=0.22, col_gap=0.7,
        )
        tab.cells[0][0].set_color(th.BAD)
        tab.cells[4][0].set_color(self.color)
        for r in (2, 3):
            for cell in tab.cells[r][1:3]:
                cell.set_color(th.BAD)
        for cell in tab.cells[7][1:]:
            cell.set_color(self.color)
        limit = kit.prose(r"float32 overflows near $3.4 \times 10^{38}$; $e^{89}$ already exceeds it.",
                          size=th.SIZE_SMALL, width=4.6)
        stable = kit.formula(r"\mathrm{softmax}(x)_i = \frac{e^{x_i - \max_j x_j}}{\sum_j e^{x_j - \max_j x_j}}",
                             size=th.SIZE_MATH_SMALL)
        box = SurroundingRectangle(stable, color=self.color, buff=0.15, stroke_width=3)
        uses = kit.prose(r"Chapter 11: temperature and top-$p$. Chapter 14: a running maximum.",
                         size=th.SIZE_SMALL, width=4.6, color=th.INK_3)
        side = VGroup(limit, VGroup(stable, box), uses).arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        layout = VGroup(tab, side).arrange(RIGHT, buff=1.2)
        self.place(layout)
        naive_t, safe_t = tab.cells[0][0], tab.cells[4][0]
        divider = DashedLine(tab.get_left(), tab.get_right(), color=th.RULE_STRONG,
                             stroke_width=th.STROKE_HAIRLINE).set_y(
            (tab.cells[3][0].get_bottom()[1] + safe_t.get_top()[1]) / 2)
        with self.narrate(
            "In float32, overflow starts at about 3.4 times 10 to the 38th, and e to the 89 already exceeds it. "
            "Take three scores: 92, 90, and 87. Exponentiate them directly, and the first two become infinity. "
            "Infinity over infinity is not a number. Now subtract the maximum first. The scores become 0, minus "
            "2, and minus 5. The largest exponent is e to the 0, so nothing overflows, and tiny terms underflow "
            "harmlessly. Chapter 11 relies on this when it samples, and chapter 14 carries a running maximum."
        ) as t:
            t.wait_until("In float32, overflow")
            self.play(FadeIn(limit), run_time=0.7)
            t.wait_until("Take three scores")
            self.play(FadeIn(tab.rules), FadeIn(naive_t), *[FadeIn(c) for c in tab.cells[1]], run_time=0.8)
            t.wait_until("Exponentiate them")
            self.play(*[FadeIn(c) for c in tab.cells[2]], run_time=0.7)
            t.wait_until("Infinity over infinity")
            self.play(*[FadeIn(c) for c in tab.cells[3]], run_time=0.7)
            t.wait_until("Now subtract")
            self.play(Create(divider), FadeIn(safe_t), FadeIn(VGroup(stable, box)), run_time=0.8)
            t.wait_until("The scores become")
            self.play(*[FadeIn(c) for c in tab.cells[5]], run_time=0.7)
            t.wait_until("The largest exponent")
            self.play(*[FadeIn(c) for c in tab.cells[6]], run_time=0.7)
            self.play(*[FadeIn(c) for c in tab.cells[7]], run_time=0.7)
            t.wait_until("Chapter 11 relies")
            self.play(FadeIn(uses), run_time=0.6)

    # Floating-point formats ---------------------------------------------------------

    def bit_row(self, exp, man, cw=0.3):
        n = 1 + exp + man
        g = kit.grid(1, n, cell=(cw, 0.45))
        g.paint(g.cell(0, 0), SIGN_C, tint=0.45)
        g.paint(g.block(0, 1, 1, 1 + exp), EXP_C, tint=0.35)
        g.paint(g.block(0, 1, 1 + exp, n), self.color, tint=0.3)
        return g

    def float_formats(self):
        self.slide("float-formats", "bfloat16 keeps float32's range, not its precision",
                   book_section="s-floating-point-formats")
        legend = VGroup(swatch("sign", SIGN_C), swatch("exponent: the scale", EXP_C),
                        swatch("mantissa: the digits", self.color)).arrange(RIGHT, buff=0.6)
        specs = [("float32", 8, 23, "24 significand bits  ·  max ≈ 3.4 × 10³⁸"),
                 ("float16", 5, 10, "11 significand bits  ·  max 65,504"),
                 ("bfloat16", 8, 7, "8 significand bits  ·  max ≈ 3.4 × 10³⁸")]
        rows = VGroup()
        for name, e, m_, info in specs:
            g = self.bit_row(e, m_)
            nm = kit.label(name, size=th.SIZE_SMALL, color=th.INK, font=th.MONO)
            cap = kit.label(f"1 + {e} + {m_}  ·  {info}", size=th.SIZE_LABEL, color=th.INK_3)
            cap.next_to(g, DOWN, buff=0.12).align_to(g, LEFT)
            r = VGroup(nm, VGroup(g, cap))
            r.grid, r.name, r.cap = g, nm, cap
            rows.add(r)
        for r in rows:
            r.name.next_to(r.grid, LEFT, buff=0.4)
        rows.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        name_x = max(r.name.get_right()[0] for r in rows)
        for r in rows:
            r[1].shift(RIGHT * (name_x + 0.4 - r.grid.get_left()[0]))
            r.name.align_to(rows[0].name, LEFT)
        note = kit.prose(r"bfloat16 $=$ float32 with 16 mantissa bits deleted: about 2 decimal digits.",
                         size=th.SIZE_SMALL, width=None)
        col = VGroup(legend, rows, note).arrange(DOWN, buff=0.5)
        self.place(col)
        f32, f16, b16 = rows
        cut = f32.grid.block(0, 1, 16, 32)
        with self.narrate(
            "The weights are stored in bf16, so it helps to know what a format stores. Each one has a sign bit, "
            "an exponent that sets the scale, and a mantissa that holds the digits. In float32, that's 8 exponent "
            "bits and 23 mantissa bits. In float16, it's 5 and 10, so its largest value is 65,504. The bf16 "
            "format is float32 with 16 mantissa bits deleted. It keeps all 8 exponent bits, so it keeps float32's "
            "range. What it gives up is precision: about 2 decimal digits."
        ) as t:
            t.wait_until("Each one has a sign")
            self.play(FadeIn(legend), run_time=0.7)
            t.wait_until("In float32, that's")
            self.play(FadeIn(f32.name), FadeIn(f32.grid), run_time=0.7)
            self.play(FadeIn(f32.cap), run_time=0.4)
            t.wait_until("In float16, it's")
            self.play(FadeIn(f16.name), FadeIn(f16.grid), FadeIn(f16.cap), run_time=0.8)
            t.wait_until("The bf16 format is")
            self.play(cut.animate.set_opacity(0.25), run_time=0.6)
            self.play(FadeIn(b16.name), TransformFromCopy(f32.grid.block(0, 1, 0, 16), b16.grid), run_time=1.0)
            self.play(cut.animate.set_opacity(1), run_time=0.4)
            t.wait_until("It keeps all 8")
            self.play(Indicate(f32.grid.block(0, 1, 1, 9), color=EXP_C), Indicate(b16.grid.block(0, 1, 1, 9),
                                                                                 color=EXP_C), run_time=0.9)
            t.wait_until("What it gives up")
            self.play(FadeIn(b16.cap), FadeIn(note), run_time=0.7)

    def ulp_roundoff(self):
        self.slide("ulp-roundoff", "The gap between neighbors grows with the number",
                   book_section="s-rounding-error-and-vanishing-addends")
        vals = [1 + 0.25 * i for i in range(4)] + [2 + 0.5 * i for i in range(4)] + [4, 5, 6, 7, 8]
        nl = tick_line(1, 8, 1.45, vals, [1, 2, 4, 8])
        gaps = VGroup()
        for lo, hi, txt in ((1, 2, "gap 0.25"), (2, 4, "gap 0.5"), (4, 8, "gap 1")):
            b = Brace(Line([nl.to_x(lo), 0.15, 0], [nl.to_x(hi), 0.15, 0]), UP, buff=0.05, color=th.INK_4)
            lab = kit.label(txt, size=th.SIZE_LABEL, color=th.INK_3).next_to(b, UP, buff=0.06)
            gaps.add(VGroup(b, lab))
        toy = kit.label("a toy format with p = 3 significand bits", size=th.SIZE_LABEL, color=th.INK_3)
        toy.next_to(nl, DOWN, buff=0.15).align_to(nl, LEFT)
        line_fig = VGroup(nl, gaps, toy)
        f = VGroup(
            kit.formula(r"\mathrm{ULP} = 2^{\,e - p + 1}", size=th.SIZE_MATH_SMALL),
            kit.formula(r"|\text{relative error}| \le 2^{-p}", size=th.SIZE_MATH_SMALL),
        ).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        tab = kit.table(
            [["float32", "$2^{-24}$", r"$6.0 \times 10^{-8}$"],
             ["float16", "$2^{-11}$", r"$4.9 \times 10^{-4}$"],
             ["bfloat16", "$2^{-8}$", r"$3.9 \times 10^{-3}$"]],
            header=["Format", "Unit roundoff", "Decimal"], align=["l", "r", "r"], size=th.SIZE_SMALL, row_gap=0.2,
        )
        bottom = VGroup(f, tab).arrange(RIGHT, buff=1.0)
        tol = kit.prose(r"Labs accept about $10^{-2}$ in bfloat16, and $10^{-6}$ in float32.", size=th.SIZE_SMALL,
                        width=None)
        col = VGroup(line_fig, bottom, tol).arrange(DOWN, buff=0.5)
        self.place(col)
        with self.narrate(
            "Labs check your output against a tolerance, and here's why. A unit in the last place, or ULP, is the "
            "gap between neighboring representable numbers. In this toy format with 3 significand bits, the gap "
            "doubles at every power of two. With p significand bits, the gap above 2 to the e is 2 to the e minus "
            "p plus 1. Rounding to nearest lands within half a ULP, so the relative error is at most 2 to the "
            "minus p. That constant is the unit roundoff. For bf16, it's about 0.0039. So the course accepts "
            "about 10 to the minus 2 in bf16, and 10 to the minus 6 in float32."
        ) as t:
            t.wait_until("A unit in the last")
            self.play(Create(nl.line), FadeIn(nl.nums), run_time=0.7)
            t.wait_until("In this toy format")
            self.play(LaggedStart(*[Create(m) for m in nl.marks], lag_ratio=0.08), FadeIn(toy), run_time=1.2)
            self.play(LaggedStart(*[FadeIn(g) for g in gaps], lag_ratio=0.4), run_time=1.2)
            t.wait_until("With p significand bits")
            self.play(FadeIn(f[0]), run_time=0.7)
            t.wait_until("Rounding to nearest")
            self.play(FadeIn(f[1]), run_time=0.7)
            t.wait_until("That constant is")
            self.play(FadeIn(tab.rules), *[FadeIn(c) for row in tab.cells for c in row], run_time=0.9)
            t.wait_until("For bf16, it's")
            self.play(Indicate(VGroup(*tab.cells[3]), color=self.color, scale_factor=1.05), run_time=0.8)
            t.wait_until("So the course accepts")
            self.play(FadeIn(tol, shift=UP * 0.1), run_time=0.7)

    def vanishing_addend(self):
        self.slide("vanishing-addend", "In bfloat16, 256 + 1 is 256",
                   book_section="s-rounding-error-and-vanishing-addends")
        ticks = [252, 253, 254, 255, 256, 258, 260]
        nl = tick_line(251.5, 260.5, 1.25, ticks, ticks, label_size=th.SIZE_SMALL)
        under = -0.75
        gap = Brace(Line([nl.to_x(256), under, 0], [nl.to_x(258), under, 0]), DOWN, buff=0.05, color=th.INK_4)
        gap_t = kit.label("ULP = 2", size=th.SIZE_LABEL, color=th.INK_3).next_to(gap, DOWN, buff=0.06)
        gap1 = Brace(Line([nl.to_x(254), under, 0], [nl.to_x(255), under, 0]), DOWN, buff=0.05, color=th.INK_4)
        gap1_t = kit.label("gap 1", size=th.SIZE_LABEL, color=th.INK_3).next_to(gap1, DOWN, buff=0.06)
        mid = DashedLine([nl.to_x(257), -0.25, 0], [nl.to_x(257), 0.25, 0], color=th.INK_4,
                         stroke_width=th.STROKE_AXIS)
        p256 = np.array([nl.to_x(256), 0, 0])
        p257 = np.array([nl.to_x(257), 0, 0])
        hop = CurvedArrow(p256 + UP * 0.25, p257 + UP * 0.25, angle=-PI / 2, color=th.INK_3, stroke_width=3,
                          tip_length=0.16)
        mid_t = kit.label("+1 gives 257: exactly halfway", size=th.SIZE_LABEL, color=th.INK_2)
        mid_t.next_to(hop, UP, buff=0.15).set_x(nl.to_x(258.5))
        back_t = kit.label("ties to even: 256", size=th.SIZE_LABEL, color=th.BAD, weight=MEDIUM)
        back_t.next_to(hop, UP, buff=0.15).set_x(nl.to_x(254.4))
        sum_dot = Dot(p256, radius=0.12, color=self.color)
        ghost = Circle(radius=0.12, color=self.color, stroke_width=3).move_to(p257)
        fig = VGroup(nl, gap, gap_t, gap1, gap1_t, mid, hop, mid_t, back_t, sum_dot, ghost)
        counter_lab = kit.label("after 1000 more additions of 1:", size=th.SIZE_SMALL, color=th.INK_2)
        counter_val = kit.label("sum = 256", size=th.SIZE_SMALL, color=self.color, weight=SEMIBOLD)
        counter = VGroup(counter_lab, counter_val).arrange(RIGHT, buff=0.3)
        key = kit.callout("key", r"In bfloat16, an addend below about $1/256$ of the running sum rounds away. "
                                 r"In float32, the ratio is $1/16{,}777{,}216$.",
                          title="Addends vanish at a ratio of 256", width=11.0, size=th.SIZE_SMALL)
        col = VGroup(fig, counter, key).arrange(DOWN, buff=0.6)
        self.place(col)
        with self.narrate(
            "Take a bf16 sum of 256, with 8 significand bits. Below 256 the gap is 1, but above it the gap is 2, "
            "so the neighbors are 255 and 258. Add 1, and the exact answer is 257: exactly halfway. Round to "
            "nearest even picks 256. Add 1 a thousand more times, and the sum is still 256. In general, an addend "
            "below about the sum divided by 256 vanishes. In float32, the ratio is about one in 16.8 million. "
            "That's why chapter 6's state and kernel reductions accumulate in float32."
        ) as t:
            self.play(Create(nl.line), FadeIn(nl.marks), FadeIn(nl.nums), run_time=0.8)
            self.play(FadeIn(sum_dot, scale=0.5), run_time=0.5)
            t.wait_until("Below 256 the gap")
            self.play(FadeIn(gap1), FadeIn(gap1_t), run_time=0.6)
            self.play(FadeIn(gap), FadeIn(gap_t), run_time=0.6)
            t.wait_until("Add 1, and the exact")
            self.play(Create(hop), Create(mid), run_time=0.8)
            self.play(FadeIn(ghost, scale=0.5), FadeIn(mid_t), run_time=0.6)
            t.wait_until("Round to nearest even")
            moving = ghost.copy()
            self.play(moving.animate.move_to(p256_now := sum_dot.get_center()), run_time=0.8)
            self.remove(moving)
            self.play(FadeIn(back_t), Indicate(sum_dot, color=self.color, scale_factor=1.4), run_time=0.8)
            t.wait_until("Add 1 a thousand")
            self.play(FadeIn(counter), run_time=0.7)
            t.wait_until("In general, an addend")
            self.play(FadeIn(key, shift=UP * 0.1), run_time=0.8)

    # GPU vocabulary -------------------------------------------------------------------

    def sm_warp_block(self):
        self.slide("sm-warp-block", "108 SMs, warps of 32 threads", book_section="s-gpu-vocabulary")
        sms = kit.grid(9, 12, cell=0.3, gap=0.06)
        sm_t = kit.label("A100: 108 streaming multiprocessors", size=th.SIZE_LABEL, color=th.INK_3)
        idle = kit.label("too little work: most SMs sit idle", size=th.SIZE_LABEL, color=self.color)
        sm_blk = VGroup(sm_t, sms, idle).arrange(DOWN, buff=0.2)
        warp = kit.grid(1, 32, cell=(0.19, 0.4))
        warp_t = kit.label("one warp: 32 threads, one instruction", size=th.SIZE_LABEL, color=th.INK_3)
        branch_a = kit.label("if branch: half the lanes masked off", size=th.SIZE_LABEL, color=self.color)
        branch_b = kit.label("else branch: the other half", size=th.SIZE_LABEL, color=th.INFO)
        warp_blk = VGroup(warp_t, warp, branch_a, branch_b).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        block = kit.box("Thread block", sublabel="up to 1024 threads = 32 warps, on one SM", width=6.4,
                        color=th.VIOLET)
        right = VGroup(warp_blk, block).arrange(DOWN, buff=0.7, aligned_edge=LEFT)
        layout = VGroup(sm_blk, right).arrange(RIGHT, buff=0.9)
        self.place(layout)
        a_lanes = VGroup(*[warp.cell(0, i) for i in range(32) if i % 4 < 2])
        b_lanes = VGroup(*[warp.cell(0, i) for i in range(32) if i % 4 >= 2])
        with self.narrate(
            "Chapter 10 and the kernel chapters use GPU words without defining them. The streaming "
            "multiprocessor, or SM, is the GPU's unit of independent execution. An A100 has 108 of them. A kernel "
            "that can't fill all 108 leaves most of the chip idle. A warp is 32 threads that run the same "
            "instruction together. If its threads take different branches, the warp runs both paths in turn, "
            "masking off the inactive threads. That's divergence. A thread block holds up to 1024 threads, runs "
            "on one SM, and cooperates through shared memory."
        ) as t:
            t.wait_until("An A100 has 108")
            self.play(FadeIn(sm_t), LaggedStart(*[FadeIn(c) for row in sms.cells for c in row], lag_ratio=0.01),
                      run_time=1.2)
            t.wait_until("A kernel that can't")
            few = VGroup(*[sms.cell(r, c) for r in range(9) for c in range(12) if r * 12 + c < 20])
            self.play(sms.fill(few, self.color, tint=0.5), FadeIn(idle), run_time=0.8)
            t.wait_until("A warp is 32")
            self.play(FadeIn(warp_t), FadeIn(warp), sms.fill(sms.cell(0, 0), th.VIOLET, tint=0.8), run_time=0.8)
            self.play(warp.fill(warp.block(0, 1, 0, 32), th.GOOD, tint=0.4), run_time=0.5)
            t.wait_until("If its threads")
            self.play(warp.fill(a_lanes, self.color, tint=0.5), warp.fill(b_lanes, th.INK_4, tint=0.1),
                      FadeIn(branch_a), run_time=0.7)
            self.play(warp.fill(b_lanes, th.INFO, tint=0.5), warp.fill(a_lanes, th.INK_4, tint=0.1),
                      FadeIn(branch_b), run_time=0.7)
            t.wait_until("A thread block holds")
            self.play(FadeIn(block, shift=UP * 0.1), run_time=0.7)

    def memory_hierarchy(self):
        self.slide("memory-hierarchy", "Four levels of memory, and HBM is the bottleneck",
                   book_section="s-gpu-vocabulary")
        levels = [("Registers", "256 KB per SM", "one thread  ·  free", 3.4, th.GOOD),
                  ("Shared memory", "up to 164 KB per SM", "one block  ·  tens of cycles", 4.6, th.GOOD),
                  ("L2 cache", "40 MB", "whole GPU  ·  hundreds of cycles", 5.8, th.INFO),
                  ("HBM", "80 GB", "whole GPU  ·  1275 GB/s measured", 7.0, th.BAD)]
        boxes = VGroup(*[kit.box(name, sublabel=size, width=w, height=0.95, color=col)
                         for name, size, _, w, col in levels]).arrange(DOWN, buff=0.18)
        notes = VGroup(*[kit.label(note, size=th.SIZE_LABEL, color=th.INK_3) for _, _, note, _, _ in levels])
        right_x = boxes[-1].get_right()[0] + 0.4
        for n, b in zip(notes, boxes):
            n.move_to([right_x + n.width / 2, b.get_y(), 0])
        hdr = kit.label("scope  ·  cost to reach", size=th.SIZE_LABEL, color=th.INK_4, weight=MEDIUM)
        hdr.next_to(notes, UP, buff=0.35).align_to(notes, LEFT)
        fig = VGroup(boxes, notes, hdr)
        line = kit.prose(r"FlashAttention keeps its working set in shared memory, not HBM.", size=th.SIZE_SMALL,
                         width=None)
        col = VGroup(fig, line).arrange(DOWN, buff=0.45)
        self.place(col)
        bott = SurroundingRectangle(boxes[3], color=th.BAD, buff=0.08, stroke_width=4)
        with self.narrate(
            "Memory comes in four levels. Registers hold 256 KB per SM, and reaching them is free. Shared memory "
            "holds up to 164 KB per SM, tens of cycles away. The L2 cache holds 40 MB, hundreds of cycles away. "
            "HBM holds 80 GB, measured at 1275 GB/s. It holds the weights and the KV cache, and it's the "
            "bottleneck in almost everything this course measures. Shared memory is a scratchpad you fill "
            "explicitly, not a cache. FlashAttention keeps its working set there."
        ) as t:
            self.play(FadeIn(hdr), run_time=0.5)
            for i, cue in enumerate(["Registers hold", "Shared memory holds", "The L2 cache", "HBM holds"]):
                t.wait_until(cue)
                self.play(FadeIn(boxes[i], shift=DOWN * 0.1), FadeIn(notes[i]), run_time=0.6)
            t.wait_until("It holds the weights")
            self.play(Create(bott), run_time=0.7)
            t.wait_until("Shared memory is a scratchpad")
            self.play(Indicate(boxes[1].rect, color=th.GOOD, scale_factor=1.05), run_time=0.8)
            t.wait_until("FlashAttention keeps")
            self.play(FadeIn(line, shift=UP * 0.1), run_time=0.7)

    def coalescing(self):
        self.slide("coalescing", "A warp should read consecutive addresses", book_section="s-gpu-vocabulary")
        nthreads, ncells, cw = 8, 32, 0.32

        def panel(title, stride, color):
            mem = kit.grid(1, ncells, cell=(cw, 0.42))
            threads = VGroup(*[Dot(radius=0.09, color=color).move_to(mem.cell(0, i * stride).get_center() + UP * 0.95)
                               for i in range(nthreads)])
            head = kit.label(title, size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
            head.next_to(threads, UP, buff=0.25).align_to(mem, LEFT)
            p = VGroup(head, threads, mem)
            p.mem, p.threads, p.head, p.stride, p.tone = mem, threads, head, stride, color
            return p

        a = panel("Coalesced: 8 threads, 8 neighbors, 1 transaction", 1, th.GOOD)
        b = panel("Strided: 8 threads, 8 transactions, mostly wasted", 4, th.BAD)
        panels = VGroup(a, b).arrange(DOWN, buff=0.6, aligned_edge=LEFT)
        gap = kit.pill("measured gap on this hardware: 6.9×", self.color, size=th.SIZE_SMALL)
        extras = kit.bullets(
            [r"Each kernel launch costs 5 to 10 µs; CUDA graphs replay many as one.",
             r"Occupancy, the share of thread slots in use, is a means, not a goal."],
            size=th.SIZE_SMALL, width=7.6,
        )
        bottom = VGroup(gap, extras).arrange(RIGHT, buff=0.6)
        col = VGroup(panels, bottom).arrange(DOWN, buff=0.55)
        self.place(col)

        def wiring(p):
            lines = VGroup()
            for i, d in enumerate(p.threads):
                cell = p.mem.cell(0, i * p.stride)
                lines.add(Line(d.get_bottom(), cell.get_top(), color=p.tone, stroke_width=2, buff=0.04))
            used = VGroup(*[p.mem.cell(0, i * p.stride) for i in range(nthreads)])
            if p.stride == 1:
                tx = VGroup(p.mem.outline(0, 1, 0, nthreads, color=th.INK, stroke=3))
            else:
                tx = VGroup(*[p.mem.outline(0, 1, i * p.stride, (i + 1) * p.stride, color=th.INK, stroke=3)
                              for i in range(nthreads)])
            return lines, used, tx

        la, ua, ta = wiring(a)
        lb, ub, tb = wiring(b)
        with self.narrate(
            "When the 32 threads of a warp read consecutive addresses, the hardware merges them into a few wide "
            "transactions. Here are 8 of them. When they read scattered addresses, it issues many narrow "
            "transactions, and wastes most of each. On this hardware, the measured gap is 6.9 times. Each launch "
            "costs 5 to 10 microseconds, so CUDA graphs replay many small kernels as one unit. And occupancy, the "
            "fraction of thread slots in use, is a means, not a goal."
        ) as t:
            self.play(FadeIn(a.head), FadeIn(a.threads), FadeIn(a.mem), run_time=0.7)
            t.wait_until("Here are 8")
            self.play(Create(la), a.mem.fill(ua, th.GOOD), run_time=0.8)
            self.play(Create(ta), run_time=0.6)
            t.wait_until("When they read scattered")
            self.play(FadeIn(b.head), FadeIn(b.threads), FadeIn(b.mem), run_time=0.6)
            self.play(Create(lb), b.mem.fill(ub, th.BAD), run_time=0.8)
            self.play(LaggedStart(*[Create(r) for r in tb], lag_ratio=0.15), run_time=1.0)
            t.wait_until("On this hardware")
            self.play(FadeIn(gap, scale=0.9), run_time=0.6)
            t.wait_until("Each launch costs")
            self.play(FadeIn(extras[0], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("And occupancy")
            self.play(FadeIn(extras[1], shift=RIGHT * 0.1), run_time=0.6)

    # Units --------------------------------------------------------------------------

    def units(self):
        self.slide("units", "GB for vendor figures, GiB for shapes", book_section="s-units")
        bc = kit.bar_compare(
            [("1 GB", 1e9, th.INFO, "10⁹ = 1,000,000,000 bytes"),
             ("1 GiB", 2 ** 30, self.color, "2³⁰ = 1,073,741,824 bytes")],
            width=6.0, bar_height=0.5, gap=0.3,
        )
        pct = kit.label("about 7% apart", size=th.SIZE_LABEL, color=th.INK_3)
        tab = kit.table(
            [
                ["Weights (a vendor-style size)", "53.8 GB", "the same bytes as 50.1 GiB"],
                ["KV cache per token (a shape)", "64 KiB", r"$64 \times 1024 = 65{,}536$ bytes"],
                ["HBM bandwidth (always decimal)", "1935 GB/s", r"$1.935 \times 10^{12}$ bytes/s"],
                [r"The ``80GB'' card in chapter 2", "80 GiB", "minus 6 GiB of headroom"],
            ],
            align=["l", "r", "l"], size=th.SIZE_SMALL, row_gap=0.24, col_gap=0.5,
        )
        col = VGroup(VGroup(bc, pct).arrange(DOWN, buff=0.2), tab).arrange(DOWN, buff=0.6)
        self.place(col)
        with self.narrate(
            "Byte counts come in two conventions, about 7% apart. A gigabyte is 10 to the 9th bytes. A gibibyte "
            "is 2 to the 30th, about 1.074 billion. Use decimal units for vendor figures, and binary units for "
            "shape arithmetic. So the weights are 53.8 GB, the same bytes as 50.1 GiB. The KV cache is 64 KiB per "
            "token, because 64 times 1024 is what the shapes produce. Bandwidth is always decimal. And chapter 2 "
            "treats the 80GB card as 80 GiB, minus 6 GiB of headroom."
        ) as t:
            t.wait_until("A gigabyte is")
            self.play(FadeIn(bc.labels[0]), GrowFromEdge(bc.bars[0], LEFT), FadeIn(bc.values[0]), run_time=0.8)
            t.wait_until("A gibibyte is")
            self.play(FadeIn(bc.labels[1]), GrowFromEdge(bc.bars[1], LEFT), FadeIn(bc.values[1]), FadeIn(pct),
                      run_time=0.8)
            t.wait_until("So the weights")
            self.play(FadeIn(tab.rules), *[FadeIn(c) for c in tab.cells[0]], run_time=0.7)
            t.wait_until("The KV cache is 64")
            self.play(*[FadeIn(c) for c in tab.cells[1]], run_time=0.6)
            t.wait_until("Bandwidth is always")
            self.play(*[FadeIn(c) for c in tab.cells[2]], run_time=0.6)
            t.wait_until("And chapter 2 treats")
            self.play(*[FadeIn(c) for c in tab.cells[3]], run_time=0.6)

    # Symbols ------------------------------------------------------------------------

    def symbols(self):
        self.slide("symbols", "The symbols, with this model's values", book_section="s-symbols-used-across-the-course")
        head = ["Symbol", "Meaning", "Value"]
        t1 = kit.table(
            [["$N$", "decoder layers", "64"],
             ["$d$", "hidden size", "5120"],
             ["$d_{ff}$", "MLP width", "17,408"],
             ["$V$", "vocabulary size", "248,320"],
             ["$b$", "bytes per element", "2"]],
            header=head, align=["l", "l", "r"], size=th.SIZE_SMALL, row_gap=0.26, col_gap=0.45,
        )
        t2 = kit.table(
            [["$h$", "query heads", "24"],
             [r"$h_{kv}$", "KV heads", "4"],
             ["$g$", r"group size, $h / h_{kv}$", "6"],
             ["$d_h$", "head dimension", "256"],
             ["$d_r$", "rotary dimension", "64"]],
            header=head, align=["l", "l", "r"], size=th.SIZE_SMALL, row_gap=0.26, col_gap=0.45,
        )
        tabs = VGroup(t1, t2).arrange(RIGHT, buff=0.9, aligned_edge=UP)
        note = kit.prose(r"Also: $T$ tokens in a pass, $L$ context length, $B$ batch size, $I$ intensity.",
                         size=th.SIZE_SMALL, width=None, color=th.INK_3)
        col = VGroup(tabs, note).arrange(DOWN, buff=0.6)
        self.place(col)
        band = kit.mark(VGroup(*t2.cells[3]), buff=0.08)
        with self.narrate(
            "When a symbol isn't defined nearby, look it up in the chapter's table. The model has 64 decoder "
            "layers, and a hidden size of 5120. Its MLP is 17,408 wide, and its vocabulary holds 248,320 tokens. "
            "Every element takes 2 bytes, in bf16. A full-attention layer has 24 query heads and 4 KV heads. So "
            "each KV head serves a group of 6 query heads. Each head is 256 wide, and 64 of those channels are "
            "rotated."
        ) as t:
            self.play(FadeIn(t1.rules), FadeIn(t2.rules), *[FadeIn(c) for c in t1.cells[0] + t2.cells[0]],
                      run_time=0.7)
            t.wait_until("The model has 64")
            self.play(*[FadeIn(c) for c in t1.cells[1] + t1.cells[2]], run_time=0.6)
            t.wait_until("Its MLP is")
            self.play(*[FadeIn(c) for c in t1.cells[3] + t1.cells[4]], run_time=0.6)
            t.wait_until("Every element takes")
            self.play(*[FadeIn(c) for c in t1.cells[5]], run_time=0.5)
            t.wait_until("A full-attention layer")
            self.play(*[FadeIn(c) for c in t2.cells[1] + t2.cells[2]], run_time=0.6)
            t.wait_until("So each KV head")
            self.play(FadeIn(band), *[FadeIn(c) for c in t2.cells[3]], run_time=0.6)
            t.wait_until("Each head is 256")
            self.play(*[FadeIn(c) for c in t2.cells[4] + t2.cells[5]], FadeIn(note), run_time=0.7)

    # What goes wrong ----------------------------------------------------------------

    def what_goes_wrong(self):
        self.slide("what-goes-wrong", "Overflow is loud; precision loss is silent", book_section="s-what-goes-wrong")
        loud = kit.callout("warning", r"In float16: \texttt{inf}, then \texttt{nan}.", title="Overflow is loud",
                           width=5.4, size=th.SIZE_SMALL)
        quiet = kit.callout("note", r"In bfloat16: finite, plausible, wrong in the third digit. Worse at long "
                                    r"context.", title="Precision loss is silent", width=5.4, size=th.SIZE_SMALL)
        cards = VGroup(loud, quiet).arrange(RIGHT, buff=0.5, aligned_edge=UP)
        spec = es("bhid,bhjd->bhij", size=th.SIZE_BODY)
        want = mono_t2c("expects q: (batch, heads, q_len, head_dim)", size=th.SIZE_SMALL,
                        t2c={"heads, q_len": th.GOOD})
        got = mono_t2c("given q:   (batch, q_len, heads, head_dim)", size=th.SIZE_SMALL,
                       t2c={"q_len, heads": th.BAD})
        runs = kit.label("sizes match, so it runs, and returns a finite, wrong tensor", size=th.SIZE_LABEL,
                         color=th.BAD)
        shapes = VGroup(spec, want, got, runs).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        rule = kit.pill("Assert the shape; don't infer it.", self.color, size=th.SIZE_SMALL)
        col = VGroup(cards, shapes, rule).arrange(DOWN, buff=0.5)
        self.place(col)
        with self.narrate(
            "Numerical failures come in two kinds. Overflow is loud: in float16, it shows up as infinity, then "
            "not a number. Precision loss is silent. In bf16, the output is finite, plausible, and wrong in the "
            "third digit, and it gets worse at long context. Einsum has a silent failure too. If q was never "
            "transposed, its letters line up with the wrong axes. When the sizes happen to match, it still runs, "
            "and returns a wrong tensor. Assert the shape; don't infer it."
        ) as t:
            t.wait_until("Overflow is loud")
            self.play(FadeIn(loud, shift=UP * 0.1), run_time=0.7)
            t.wait_until("Precision loss is silent")
            self.play(FadeIn(quiet, shift=UP * 0.1), run_time=0.7)
            t.wait_until("Einsum has a silent")
            self.play(FadeIn(spec), FadeIn(want), run_time=0.7)
            t.wait_until("If q was never")
            self.play(FadeIn(got, shift=RIGHT * 0.1), run_time=0.7)
            t.wait_until("When the sizes")
            self.play(FadeIn(runs), run_time=0.6)
            t.wait_until("Assert the shape")
            self.play(FadeIn(rule, scale=0.9), run_time=0.6)
