"""Chapter 02, memory arithmetic.

Teaches content/chapters/02-memory-arithmetic.md in its order. Every number
below was recomputed in Python from the config (d = 5120, N = 64, h = 24,
h_kv = 4, d_h = 256, d_ff = 17408, V = 248320, full_attention_interval = 4,
16 linear key heads, 48 linear value heads, 128-wide linear heads, conv
kernel 4, b = 2 bytes) and matches the chapter.

Layers: N_full = 64 / 4 = 16; N_lin = 64 - 16 = 48.
Units: 1 GB = 10^9 bytes, 1 GiB = 2^30 bytes; 2^30 / 10^9 = 1.074 (7%).

Parameters:
- Embedding: 248,320 x 5120 = 1,271,398,400 (1.27B); with the untied output
  projection 2,542,796,800 (2.54B), 9.45% of the total (9.5%).
- MLP: 5120 x 17408 = 89,128,960 per matrix (89.1M); 3 x that = 267,386,880
  (267.4M); x 64 = 17,112,760,320 (17.11B), 63.6% of the total (64%).
- Full attention: q_proj 5120 x 12288 = 62,914,560 (62.9M); k_proj = v_proj
  = 5,242,880 (5.2M); o_proj 6144 x 5120 = 31,457,280 (31.5M); block
  104,857,600 (104.9M). Gate cost = 62,914,560 / 2 = 31,457,280 (31.5M).
- Linear attention: in_proj_qkvz 5120 x 16384 = 83,886,080 (83.9M); conv1d
  10,240 x 4 = 40,960 (0.04M); in_proj_ba 5120 x 96 = 491,520 (0.49M);
  out_proj 31,457,280 (31.5M); block 115,875,840 (115.9M);
  115,875,840 - 104,857,600 = 11,018,240 (11M larger).
- Norms: 64 x 2 x 5120 + 5120 = 660,480.
- Layer totals: 16 x (104,857,600 + 267,386,880) = 5,955,911,680 (5.96B,
  22.1%); 48 x (115,875,840 + 267,386,880) = 18,396,610,560 (18.40B, 68.4%).
- Total 26,895,979,520 (26.9B). Bytes x 2 = 53,791,959,040 = 53.8 GB =
  50.1 GiB (50.098); 80 - 50.098 = 29.9 GiB left ("roughly 30 GiB").

KV cache:
- 2 x 4 x 256 x 2 = 4096 bytes = 4 KiB per token per full layer;
  x 16 = 65,536 bytes = 64 KiB per token.
- 4096 tokens x 64 KiB = 0.25 GiB. Counter on the grid slide: n x 64 KiB,
  20 tokens = 1,280 KiB.
- GQA: 24 / 4 = 6 query heads per KV head; 2 x 24 x 256 x 2 x 16 =
  393,216 bytes = 384 KiB per token, 6x larger.
- Table (64 KiB x L x B): 4k: 0.25 / 2.0 / 8.0 GiB; 32k: 2.0 / 16.0 / 64.0;
  128k: 8.0 / 64.0 / 256.0 GiB.

Recurrent state:
- 48 heads x 128 x 128 x 4 bytes = 3,145,728 (3.0 MiB); conv window
  10,240 x 4 x 2 = 81,920 (80 KiB); one layer 3,227,648 (3.078 MiB);
  x 48 = 154,927,104 bytes = 147.75 MiB (147.8 MiB).

Break-even:
- Dense: 4096 x 64 = 262,144 bytes = 256 KiB per token = 0.25 MiB/token.
- Hybrid: 147.75 MiB + 0.0625 MiB/token. Saving 262,144 - 65,536 = 196,608
  bytes = 192 KiB per token; 154,927,104 / 196,608 = 788.0 tokens, where
  both are 197.0 MiB. Plot ends at L = 1600: dense 400 MiB, hybrid
  247.75 MiB.
- 32k: dense 8.0 GiB, hybrid 2.144 GiB (2.14); 128k: 32.0 vs 8.144 GiB
  (8.14). Ratio tends to 64 / 16 = 4.

Batch:
- C = 80 x 2^30 = 85,899,345,920; W = 53.8e9; O = 6 x 2^30 =
  6,442,450,944; C - W - O = 25,656,894,976 bytes = 23.9 GiB (23.895).
- 32k: 154,927,104 + 65,536 x 32,768 = 2,302,410,752 bytes = 2.14 GiB;
  quotient 11.14, B = 11; state share 154,927,104 / 2,302,410,752 = 6.7%
  (7%); 11 slots use 23.59 GiB, 0.31 GiB unused.
- 4k: 154,927,104 + 65,536 x 4096 = 423,362,560 bytes = 0.394 GiB (0.39);
  quotient 60.6, B = 60; state share 36.6% (37%); 60 slots use 23.66 GiB.
- 40 GiB card: 42,949,672,960 - 53.8e9 - 6,442,450,944 =
  -17,292,777,984 < 0, so B = 0.
- int8 lever: weights 26.9 GB, numerator 48.9 GiB (48.95).

Activations:
- MLP hidden at a 4096-token prefill chunk: 4096 x 17,408 x 2 =
  142,606,336 bytes = 136 MiB; SwiGLU holds two.
- Scores at 8k, 24 heads: 24 x 8192 x 8192 x 2 = 3,221,225,472 = 3.0 GiB.
  Doubling the context multiplies the score count by 4 (square).

Budget: 50.1 + 1.0 + 3.0 + 2.0 + 23.9 = 80.0 GiB. KV only: 25,656,894,976 /
65,536 = 391,493 tokens (391k).

What goes wrong: 53.8 - 50.1 = 3.7 GiB; 3.7 / 2.14 = 1.7, "two more
sequences at 32k" as the chapter says. Batch 8 at 32k: 8 x 2 = 16 GiB.
"""

from manim import *

import kit
from kit import theme as th

# Colour roles, used the same way on every slide.
WEIGHTS = th.INFO
STATE = th.VIOLET
DENSE = th.HL_D
OVERHEAD = th.INK_4
ACT = th.GOOD


def frame_box(width: float, height: float) -> VGroup:
    """A memory card drawn as an empty bar, left edge at x = 0."""
    back = Rectangle(width=width, height=height, stroke_width=0, fill_color=th.RAISE, fill_opacity=1)
    edge = Rectangle(width=width, height=height, stroke_color=th.INK_3, stroke_width=th.STROKE_BOX, fill_opacity=0)
    edge.set_z_index(2)
    g = VGroup(back, edge)
    g.move_to([width / 2, 0, 0])
    return g


def seg(x0: float, width: float, height: float, color, y: float = 0.0, opacity: float = 0.85) -> Rectangle:
    """One filled segment of a bar, from x0 to x0 + width."""
    r = Rectangle(width=max(width, 0.01), height=height, fill_color=color, fill_opacity=opacity, stroke_width=0)
    r.move_to([x0 + width / 2, y, 0])
    return r


def swatch(color, text: str, size: float = th.SIZE_LABEL) -> VGroup:
    sq = Square(side_length=0.24, fill_color=color, fill_opacity=0.85, stroke_width=0)
    name = kit.label(text, size=size, color=th.INK_2)
    return VGroup(sq, name).arrange(RIGHT, buff=0.15)


def row(tab, i: int) -> VGroup:
    return VGroup(*tab.cells[i])


class Lecture(kit.Lecture):
    def construct(self):
        self.title_slide(
            narration=(
                "Memory arithmetic. "
                "How many users can you serve at once on one GPU? "
                "This lecture answers that by counting bytes: the weights, the KV cache, the recurrent state, "
                "and the batch that fits."
            )
        )
        self.budget_question()
        self.layer_split()
        self.memory_map()
        self.embeddings()
        self.mlp()
        self.attention_blocks()
        self.total_params()
        self.weights_on_card()
        self.kv_per_token()
        self.cache_grows()
        self.gqa()
        self.cache_table()
        self.recurrent_state()
        self.break_even()
        self.batch_formula()
        self.batch_fits()
        self.batch_levers()
        self.activations()
        self.budget()
        self.what_goes_wrong()
        self.closing_slide(
            [
                r"26.9B parameters: 53.8 GB, or 50.1 GiB, in bf16. The MLP is 64\% of them.",
                r"KV cache: 64 KiB per token, from 16 layers. Recurrent state: a fixed 147.8 MiB.",
                r"Break-even: fixed state over KV bytes saved per token, 788 tokens.",
                r"$B_{\max}$ is 11 at 32k context, 60 at 4k, and 0 on a 40 GiB card.",
            ],
            narration=(
                "Here's what to keep. "
                "The model holds 26.9 billion parameters, 53.8 GB in bfloat16, and the MLP is 64 percent of them. "
                "The KV cache costs 64 KiB per token, from the 16 full-attention layers only. "
                "The recurrent state is a fixed 147.8 MiB per sequence. "
                "The break-even is the fixed state divided by the KV bytes saved per token: 788 tokens. "
                "And the card fits 11 sequences at 32k context, 60 at 4k, and none on a 40 GB card. "
                "Next, you'll open the tokenizer and the embedding table that cost those first 2.54 billion "
                "parameters."
            ),
            book_section="s-what-goes-wrong",
        )

    # Setup ----------------------------------------------------------------

    def budget_question(self):
        self.slide("budget-question", "Everything has to fit in 80 GiB", book_section="s-before-you-start")
        w, h = 11.0, 0.8
        card = frame_box(w, h)
        name = kit.label("A100 80GB", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        name.next_to(card, UP, buff=0.15).align_to(card, LEFT)
        cap = kit.label("80 GiB", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        cap.next_to(card, UP, buff=0.15).align_to(card, RIGHT)
        ask = kit.label("?", size=th.SIZE_HEADING, color=self.color, weight=SEMIBOLD).move_to(card)
        bar = VGroup(card, name, cap, ask)
        tab = kit.table(
            [
                [r"Weights", r"decimal GB", r"$1\ \mathrm{GB} = 10^9$ bytes"],
                [r"Caches", r"binary GiB", r"$1\ \mathrm{GiB} = 2^{30}$ bytes"],
                [r"Each parameter", r"bfloat16", r"$b = 2$ bytes"],
            ],
            header=["Quantity", "Unit", "Definition"],
            size=th.SIZE_SMALL,
            align=["l", "l", "l"],
        )
        note = kit.prose(r"GiB is 7\% larger than GB: enough to matter in a budget.", size=th.SIZE_SMALL,
                         color=self.color, width=None)
        col = VGroup(bar, tab, note).arrange(DOWN, buff=0.55)
        self.place(col)
        with self.narrate(
            "Suppose you're about to serve this model to real users. "
            "The first question anyone asks is how many of them you can serve at once. "
            "The answer comes from one budget: the 80 GiB of memory on an A100. "
            "Everything the engine needs has to fit inside it. "
            "Every number in this lecture is arithmetic, not measurement. "
            "Two conventions first. "
            "Weights are quoted in decimal GB, a billion bytes each. "
            "Caches are quoted in binary GiB, two to the thirtieth bytes each. "
            "The two differ by 7 percent, which is enough to matter in a budget. "
            "And every parameter is stored in bfloat16, at 2 bytes each."
        ) as t:
            t.wait_until("The first question")
            self.play(FadeIn(card), FadeIn(name), FadeIn(cap), run_time=0.8)
            t.wait_until("Everything the engine")
            self.play(FadeIn(ask, scale=0.6), run_time=0.6)
            t.wait_until("Weights are quoted")
            self.play(FadeIn(tab.rules), FadeIn(row(tab, 0)), FadeIn(row(tab, 1)), run_time=0.8)
            t.wait_until("Caches are quoted")
            self.play(FadeIn(row(tab, 2)), run_time=0.6)
            t.wait_until("The two differ")
            self.play(FadeIn(note), run_time=0.6)
            t.wait_until("And every parameter")
            self.play(FadeIn(row(tab, 3)), run_time=0.6)

    def layer_split(self):
        self.slide("layer-split", "Only every fourth layer is full attention", book_section="s-before-you-start")
        tab = kit.table(
            [
                [r"$d$", r"\texttt{hidden\_size}", "5120"],
                [r"$N$", r"\texttt{num\_hidden\_layers}", "64"],
                [r"$h$", r"\texttt{num\_attention\_heads}", "24"],
                [r"$h_{kv}$", r"\texttt{num\_key\_value\_heads}", "4"],
                [r"$d_h$", r"\texttt{head\_dim}", "256"],
                [r"$d_{ff}$", r"\texttt{intermediate\_size}", "17408"],
                [r"$V$", r"\texttt{vocab\_size}", "248320"],
            ],
            header=["", "Field", "Value"],
            size=th.SIZE_SMALL,
            align=["l", "l", "r"],
            row_gap=0.2,
        )
        g = kit.grid(4, 16, cell=0.34, gap=0.05)
        cap = kit.prose(r"Layers 0 to 63, sixteen per row. Full attention at 3, 7, 11, \dots, 63.",
                        size=th.SIZE_LABEL, color=th.INK_3, width=5.9)
        legend = VGroup(swatch(self.color, "16 full attention"), swatch(STATE, "48 linear attention")).arrange(
            RIGHT, buff=0.5)
        eq = kit.formula(r"N_{\text{full}} = \tfrac{64}{4} = 16,", r"\quad N_{\text{lin}} = 64 - 16 = 48",
                         size=th.SIZE_MATH_SMALL)
        eq[0].set_color(self.color)
        eq[1].set_color(STATE)
        right = VGroup(g, cap, legend, eq).arrange(DOWN, buff=0.35)
        layout = VGroup(tab, right).arrange(RIGHT, buff=0.7)
        self.place(layout)
        full = [g.cell(r, c) for r in range(4) for c in range(16) if c % 4 == 3]
        lin = [g.cell(r, c) for r in range(4) for c in range(16) if c % 4 != 3]
        with self.narrate(
            "Everything in this chapter comes from a handful of config fields. "
            "The hidden size, d, is 5120. "
            "There are 64 layers. "
            "Queries have 24 heads, but keys and values have only 4, each 256 wide. "
            "The MLP is 17408 wide, and the vocabulary holds 248,320 entries. "
            "Chapter 1 showed that this model mixes two kinds of layer. "
            "Full-attention layers look back at every earlier token. "
            "Linear-attention layers keep a fixed-size running summary instead. "
            "Full attention lands on every fourth layer. "
            "So 16 layers are full attention, and the other 48 are linear."
        ) as t:
            self.play(FadeIn(tab.rules), FadeIn(row(tab, 0)), run_time=0.5)
            t.wait_until("The hidden size")
            self.play(FadeIn(row(tab, 1)), run_time=0.5)
            t.wait_until("There are 64")
            self.play(FadeIn(row(tab, 2)), run_time=0.5)
            t.wait_until("Queries have 24")
            self.play(LaggedStart(*[FadeIn(row(tab, i)) for i in (3, 4, 5)], lag_ratio=0.4), run_time=1.2)
            t.wait_until("The MLP is")
            self.play(FadeIn(row(tab, 6)), FadeIn(row(tab, 7)), run_time=0.6)
            t.wait_until("Chapter 1 showed")
            self.play(FadeIn(g), FadeIn(cap), run_time=0.8)
            t.wait_until("Full attention lands")
            self.play(LaggedStart(*[g.fill(c, self.color, tint=0.8) for c in full], lag_ratio=0.15), run_time=2.0)
            t.wait_until("So 16 layers")
            self.play(g.fill(VGroup(*lin), STATE, tint=0.3), FadeIn(legend), run_time=0.8)
            self.play(FadeIn(eq), run_time=0.8)

    def memory_map(self):
        self.slide("memory-map", "Four things compete for the card", book_section="s-where-the-memory-goes")
        colors = [WEIGHTS, self.color, STATE, ACT]
        names = ["Weights", "KV cache", "Recurrent state", "Activations"]
        whats = [swatch(c, n, size=th.SIZE_SMALL) for c, n in zip(colors, names)]
        tab = kit.table(
            [
                [whats[0], "Nothing: fixed for the model", "Whole run"],
                [whats[1], "Every token of every sequence", "Until the sequence finishes"],
                [whats[2], "Number of sequences, not their length", "Until the sequence finishes"],
                [whats[3], "Tokens in one forward pass", "One forward pass"],
            ],
            header=["What", "Grows with", "Lifetime"],
            size=th.SIZE_SMALL,
            row_gap=0.4,
        )
        note = kit.prose(r"Plus a fixed overhead for the driver.", size=th.SIZE_SMALL, color=th.INK_3, width=None)
        col = VGroup(tab, note).arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        self.place(col)
        with self.narrate(
            "Before any counting, here's the map. "
            "Besides a fixed overhead for the driver, four things compete for the card. "
            "The weights are fixed for the model, and they stay for the whole run. "
            "The KV cache grows with every token of every sequence. "
            "The recurrent state grows with the number of sequences, but not with their length. "
            "Activations grow with the tokens in one forward pass, and they live only for that pass. "
            "The lecture works through them in that order: the weights first, then the two per-sequence costs, "
            "then the batch they allow."
        ) as t:
            self.play(FadeIn(tab.rules), FadeIn(row(tab, 0)), run_time=0.6)
            t.wait_until("Besides a fixed")
            self.play(FadeIn(note), run_time=0.5)
            for i, cue in enumerate(["The weights are fixed", "The KV cache grows", "The recurrent state grows",
                                     "Activations grow"]):
                t.wait_until(cue)
                self.play(FadeIn(row(tab, i + 1), shift=RIGHT * 0.12), run_time=0.6)

    # Parameters -------------------------------------------------------------

    def embeddings(self):
        self.slide("embeddings", "The vocabulary alone costs 2.54B parameters",
                   book_section="s-counting-the-parameters")
        method = kit.prose(r"Multiply out each matrix's shape, add up one layer, then multiply by the layers "
                           r"that have it.", size=th.SIZE_SMALL, color=th.INK_3, width=12.0)

        def matrix(title: str):
            r = Rectangle(width=0.9, height=3.0, fill_color=th.tint(WEIGHTS, 0.3), fill_opacity=1,
                          stroke_color=WEIGHTS, stroke_width=th.STROKE_BOX)
            top = kit.math_label(r"d = 5120", size=th.SIZE_LABEL, color=th.INK_3).next_to(r, UP, buff=0.12)
            side = kit.math_label(r"V = 248{,}320", size=th.SIZE_LABEL, color=th.INK_3)
            side.rotate(PI / 2).next_to(r, LEFT, buff=0.12)
            name = kit.label(title, size=th.SIZE_LABEL, color=th.INK_2, weight=MEDIUM).next_to(r, DOWN, buff=0.15)
            return VGroup(r, top, side, name)

        emb, out = matrix("Embedding"), matrix("Output projection")
        mats = VGroup(emb, out).arrange(RIGHT, buff=1.0, aligned_edge=DOWN)
        d = kit.Derivation(
            [
                r"V \times d &= 248{,}320 \times 5120",
                r"&= 1{,}271{,}398{,}400 \approx 1.27\text{B}",
                r"2 \times V \times d &\approx 2.54\text{B}",
            ],
            size=th.SIZE_MATH_SMALL,
        )
        share = kit.prose(r"9.5\% of the model, and none of it does any reasoning.", size=th.SIZE_SMALL,
                          color=self.color, width=5.4)
        right = VGroup(d, share).arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        body = VGroup(mats, right).arrange(RIGHT, buff=1.0)
        col = VGroup(method, body).arrange(DOWN, buff=0.5)
        self.place(col)
        with self.narrate(
            "Now count the weights, since they're the one cost you pay before serving anyone. "
            "The method is the same for every component. "
            "Multiply out each matrix's shape, add up one layer, then multiply by how many layers have it. "
            "Start with the embedding. "
            "It turns each token id into a vector of width d, with one row per vocabulary entry. "
            "That's 248,320 times 5120, about 1.27 billion parameters. "
            "The output projection at the other end is a separate matrix of the same shape. "
            "Together they hold about 2.54 billion. "
            "That's 9.5 percent of the model, spent on the vocabulary, and none of it does any reasoning."
        ) as t:
            t.wait_until("The method is")
            self.play(FadeIn(method), run_time=0.8)
            t.wait_until("Start with the embedding")
            self.play(GrowFromEdge(emb[0], DOWN), FadeIn(emb[3]), run_time=0.8)
            t.wait_until("It turns each")
            self.play(FadeIn(emb[1]), FadeIn(emb[2]), run_time=0.6)
            t.wait_until("That's 248,320")
            self.play(d.show(0), run_time=0.8)
            self.play(d.show(1), run_time=0.8)
            t.wait_until("The output projection")
            self.play(TransformFromCopy(emb, out), run_time=1.0)
            t.wait_until("Together they")
            self.play(d.show(2), run_time=0.8)
            t.wait_until("That's 9.5")
            self.play(FadeIn(share), run_time=0.6)

    def mlp(self):
        self.slide("mlp", "The MLP is 64% of the model", book_section="s-counting-the-parameters")
        boxes = VGroup(
            kit.box("gate_proj", sublabel="17408 × 5120", font=th.MONO, color=WEIGHTS, width=3.2),
            kit.box("up_proj", sublabel="17408 × 5120", font=th.MONO, color=WEIGHTS, width=3.2),
            kit.box("down_proj", sublabel="5120 × 17408", font=th.MONO, color=WEIGHTS, width=3.2),
        ).arrange(RIGHT, buff=0.45)
        each = VGroup(*[kit.label("89.1M", size=th.SIZE_LABEL, color=th.INK_3).next_to(b, DOWN, buff=0.12)
                        for b in boxes])
        one = kit.formula(r"3 \times 5120 \times 17{,}408 = 267{,}386{,}880 \approx 267.4\text{M}",
                          size=th.SIZE_MATH_SMALL)
        tiles = kit.grid(4, 16, cell=0.3, gap=0.05)
        tiles_note = kit.label("one MLP in each of the 64 layers", size=th.SIZE_LABEL, color=th.INK_3)
        all_ = kit.formula(r"64 \times 267{,}386{,}880 = 17{,}112{,}760{,}320 \approx 17.11\text{B}",
                           size=th.SIZE_MATH_SMALL, color=WEIGHTS)
        lower = VGroup(VGroup(tiles, tiles_note).arrange(DOWN, buff=0.15), all_).arrange(RIGHT, buff=0.6)
        col = VGroup(VGroup(boxes, each), one, lower).arrange(DOWN, buff=0.5)
        self.place(col)
        with self.narrate(
            "Every layer ends in an MLP, the feed-forward block. "
            "This model's MLP is SwiGLU, which needs three matrices. "
            "A gate projection and an up projection each map d to 17408. "
            "A down projection maps back. "
            "Each is about 89.1 million parameters, so one MLP holds about 267.4 million. "
            "Every one of the 64 layers has an MLP, whichever kind of attention it uses. "
            "That's 17.11 billion parameters, 64 percent of the whole model. "
            "For the engine, the MLP is the biggest single target."
        ) as t:
            t.wait_until("A gate projection")
            self.play(FadeIn(boxes[0], shift=UP * 0.1), FadeIn(boxes[1], shift=UP * 0.1), run_time=0.8)
            t.wait_until("A down projection")
            self.play(FadeIn(boxes[2], shift=UP * 0.1), run_time=0.6)
            t.wait_until("Each is about")
            self.play(FadeIn(each), run_time=0.5)
            self.play(Write(one), run_time=1.0)
            t.wait_until("Every one of the 64")
            self.play(FadeIn(tiles_note), FadeIn(tiles), run_time=0.5)
            self.play(LaggedStart(*[tiles.fill(tiles.cell(r, c), WEIGHTS, tint=0.55)
                                    for r in range(4) for c in range(16)], lag_ratio=0.04), run_time=2.0)
            t.wait_until("That's 17.11")
            self.play(FadeIn(all_), run_time=0.8)

    def attention_blocks(self):
        self.slide("attention-blocks", "Linear attention saves state, not parameters",
                   book_section="s-counting-the-parameters")
        top = 83.9

        def panel(title, color, total, items):
            head = kit.label(title, size=th.SIZE_SMALL, color=color, weight=SEMIBOLD)
            bars = kit.bar_compare(items, max_value=top, width=3.0, bar_height=0.38, gap=0.24, color=color,
                                   size=th.SIZE_LABEL)
            tot = kit.label(f"Block total  {total}", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
            return VGroup(head, bars, tot).arrange(DOWN, buff=0.35, aligned_edge=LEFT)

        full = panel("Full attention, 16 layers", self.color, "104.9M", [
            ("q_proj + gate", 62.9, None, "62.9M"),
            ("k_proj", 5.2, None, "5.2M"),
            ("v_proj", 5.2, None, "5.2M"),
            ("o_proj", 31.5, None, "31.5M"),
        ])
        lin = panel("Linear attention, 48 layers", STATE, "115.9M", [
            ("in_proj_qkvz", 83.9, None, "83.9M"),
            ("out_proj", 31.5, None, "31.5M"),
            ("in_proj_ba", 0.49, None, "0.49M"),
            ("conv1d", 0.04, None, "0.04M"),
        ])
        panels = VGroup(full, lin).arrange(RIGHT, buff=0.9, aligned_edge=UP)
        note = kit.prose(r"A linear-attention block is \textbf{11M larger}. It saves \textit{state}, the memory "
                         r"kept per conversation.", size=th.SIZE_SMALL, color=th.INK_2, width=12.0)
        col = VGroup(panels, note).arrange(DOWN, buff=0.55)
        self.place(col)
        fb, lb = full[1], lin[1]
        with self.narrate(
            "The attention blocks come next, and there are two kinds. "
            "A full-attention block has four projections. "
            "Its query projection is double width, because it emits queries and an output gate side by side. "
            "Keys and values have only 4 heads, so their projections are small. "
            "The block holds about 104.9 million parameters, and the gate accounts for 31.5 million of them. "
            "A linear-attention block also has four weight tensors. "
            "Its input projection stacks queries, keys, values, and a gate. "
            "It totals about 115.9 million parameters. "
            "So a linear-attention block is 11 million larger, not smaller. "
            "What it saves is state, the memory it keeps per conversation, and that's the point of the design."
        ) as t:
            t.wait_until("A full-attention block")
            self.play(FadeIn(full[0]), FadeIn(fb.labels), *fb.grow(run_time=1.0), run_time=1.0)
            t.wait_until("Its query projection")
            self.play(FadeIn(fb.values[0]), Indicate(fb.bars[0], color=self.color, scale_factor=1.05), run_time=0.8)
            t.wait_until("Keys and values")
            self.play(FadeIn(fb.values[1:]), run_time=0.6)
            t.wait_until("The block holds")
            self.play(FadeIn(full[2]), run_time=0.6)
            t.wait_until("A linear-attention block also")
            self.play(FadeIn(lin[0]), FadeIn(lb.labels), *lb.grow(run_time=1.0), FadeIn(lb.values), run_time=1.0)
            t.wait_until("It totals")
            self.play(FadeIn(lin[2]), run_time=0.6)
            t.wait_until("So a linear-attention")
            self.play(FadeIn(note), run_time=0.8)

    def total_params(self):
        self.slide("total-params", "26.9 billion parameters in all", book_section="s-counting-the-parameters")
        total = 26_895_979_520
        w, h = 11.5, 0.75
        parts = [(2_542_796_800, WEIGHTS, "2.54B"), (5_955_911_680, self.color, "5.96B"),
                 (18_396_610_560, STATE, "18.40B")]
        bar1 = VGroup()
        vals1 = VGroup()
        x = 0.0
        for n, c, txt in parts:
            s = seg(x, w * n / total, h, th.tint(c, 0.5), opacity=1)
            bar1.add(s)
            vals1.add(kit.label(txt, size=th.SIZE_LABEL, color=th.INK, weight=MEDIUM).move_to(s))
            x += w * n / total
        frame1 = frame_box(w, h)
        key1 = VGroup(swatch(WEIGHTS, "Embedding and output"), swatch(self.color, "16 full-attention layers"),
                      swatch(STATE, "48 linear-attention layers")).arrange(RIGHT, buff=0.5)
        g1 = VGroup(frame1, bar1, vals1)
        key1.next_to(g1, DOWN, buff=0.22).align_to(g1, LEFT)
        mlp = 17_112_760_320
        bar2 = VGroup(seg(0, w * mlp / total, h, th.tint(WEIGHTS, 0.5), opacity=1),
                      seg(w * mlp / total, w * (total - mlp) / total, h, th.tint(th.INK_4, 0.35), opacity=1))
        vals2 = VGroup(kit.label("MLP  17.11B  (64%)", size=th.SIZE_LABEL, color=th.INK, weight=MEDIUM).move_to(bar2[0]),
                       kit.label("everything else", size=th.SIZE_LABEL, color=th.INK_2).move_to(bar2[1]))
        g2 = VGroup(frame_box(w, h), bar2, vals2)
        sum_ = kit.formula(r"2.54\text{B} + 5.96\text{B} + 18.40\text{B} + 660{,}480 = 26{,}895{,}979{,}520",
                           size=th.SIZE_MATH_SMALL)
        norms = kit.prose(r"The 660,480 norm parameters are rounding error.", size=th.SIZE_LABEL, color=th.INK_3,
                          width=None)
        col = VGroup(VGroup(g1, key1), sum_, norms, g2).arrange(DOWN, buff=0.4)
        self.place(col)
        with self.narrate(
            "Now add up the pieces. "
            "The 16 full-attention layers, with their MLPs, hold about 5.96 billion parameters. "
            "The 48 linear-attention layers hold about 18.4 billion. "
            "The embedding and output projection add 2.54 billion. "
            "The norms add only 660,480, which is rounding error. "
            "The total is about 26.9 billion parameters, which matches the name on the box. "
            "Cut the same total another way, and the MLP alone is 64 percent of it."
        ) as t:
            self.play(FadeIn(frame1), run_time=0.5)
            t.wait_until("The 16 full-attention")
            self.play(GrowFromEdge(bar1[1], LEFT), FadeIn(vals1[1]), FadeIn(key1[1]), run_time=0.8)
            t.wait_until("The 48 linear")
            self.play(GrowFromEdge(bar1[2], LEFT), FadeIn(vals1[2]), FadeIn(key1[2]), run_time=0.8)
            t.wait_until("The embedding and output")
            self.play(GrowFromEdge(bar1[0], LEFT), FadeIn(vals1[0]), FadeIn(key1[0]), run_time=0.8)
            t.wait_until("The norms add")
            self.play(FadeIn(norms), run_time=0.6)
            t.wait_until("The total is")
            self.play(Write(sum_), run_time=1.2)
            t.wait_until("Cut the same")
            self.play(FadeIn(g2[0]), GrowFromEdge(bar2[0], LEFT), GrowFromEdge(bar2[1], LEFT), run_time=0.9)
            self.play(FadeIn(vals2), run_time=0.5)

    def weights_on_card(self):
        self.slide("weights-on-card", "The weights leave about 30 GiB", book_section="s-counting-the-parameters")
        conv = kit.Derivation(
            [
                r"26{,}895{,}979{,}520 \times 2 &= 53{,}791{,}959{,}040\ \text{bytes}",
                r"&= 53.8\ \mathrm{GB} = 50.1\ \mathrm{GiB}",
            ],
            size=th.SIZE_MATH_SMALL,
        )
        unit = 10.0 / 80  # Manim units per GiB
        hgt = 0.7
        c80 = frame_box(80 * unit, hgt)
        w80 = seg(0, 50.1 * unit, hgt, WEIGHTS)
        w80_t = kit.label("weights 50.1 GiB", size=th.SIZE_LABEL, color=th.CANVAS, weight=MEDIUM).move_to(w80)
        free = kit.label("29.9 GiB left", size=th.SIZE_SMALL, color=self.color, weight=SEMIBOLD)
        free.move_to([(50.1 + 80) / 2 * unit, 0, 0])
        n80 = kit.label("80 GiB card", size=th.SIZE_SMALL, color=th.INK_2).next_to(c80, LEFT, buff=0.3)
        card80 = VGroup(n80, c80, w80, w80_t, free)
        c40 = frame_box(40 * unit, hgt)
        w40 = seg(0, 40 * unit, hgt, WEIGHTS)
        over = seg(40 * unit, 10.1 * unit, hgt, th.BAD)
        over_t = kit.label("10.1 GiB short", size=th.SIZE_LABEL, color=th.BAD, weight=SEMIBOLD)
        over_t.next_to(over, RIGHT, buff=0.2)
        n40 = kit.label("40 GiB card", size=th.SIZE_SMALL, color=th.INK_2).next_to(c40, LEFT, buff=0.3)
        card40 = VGroup(n40, c40, w40, over, over_t)
        card40.shift(DOWN * 1.3)
        n40.align_to(n80, LEFT)
        VGroup(c40, w40, over, over_t).align_to(c80, LEFT)
        cards = VGroup(card80, card40)
        col = VGroup(conv, cards).arrange(DOWN, buff=0.8)
        self.place(col)
        with self.narrate(
            "Convert the count to bytes. "
            "At 2 bytes per parameter, the weights take 53.8 GB. "
            "In binary units that's 50.1 GiB. "
            "Place them on the 80 GiB card, and roughly 30 GiB is left before anything else is allocated. "
            "That's the first and largest line in the budget. "
            "Now try an A100 with 40 GB. "
            "The weights alone are bigger than the card, so it can't serve this model at all."
        ) as t:
            t.wait_until("At 2 bytes")
            self.play(conv.show(0), run_time=1.0)
            t.wait_until("In binary units")
            self.play(conv.show(1), run_time=0.8)
            t.wait_until("Place them")
            self.play(FadeIn(n80), FadeIn(c80), run_time=0.5)
            self.play(GrowFromEdge(w80, LEFT), run_time=1.2)
            self.play(FadeIn(w80_t), FadeIn(free), run_time=0.6)
            t.wait_until("Now try")
            self.play(FadeIn(n40), FadeIn(c40), run_time=0.5)
            self.play(GrowFromEdge(w40, LEFT), run_time=0.9)
            t.wait_until("The weights alone")
            self.play(GrowFromEdge(over, LEFT), FadeIn(over_t), run_time=0.8)

    # KV cache ---------------------------------------------------------------

    def kv_per_token(self):
        self.slide("kv-per-token", "Each token adds 64 KiB of KV cache", book_section="s-kv-cache")
        g = kit.grid(2, 4, cell=(1.3, 0.7), values=[["256"] * 4, ["256"] * 4], row_labels=["K", "V"],
                     col_labels=["1", "2", "3", "4"], size=th.SIZE_SMALL)
        heads = kit.label("KV head", size=th.SIZE_LABEL, color=th.INK_3).next_to(g, UP, buff=0.15)
        g.add(heads)
        cap = kit.label("one token, one full-attention layer", size=th.SIZE_LABEL, color=th.INK_3)
        pic = VGroup(g, cap).arrange(DOWN, buff=0.3)
        f1 = kit.formula(
            r"\underbrace{2}_{K,\,V} \times \underbrace{4}_{h_{kv}} \times \underbrace{256}_{d_h}"
            r" \times \underbrace{2}_{\text{bytes}}",
            r"= 4096\ \text{bytes} = 4\ \mathrm{KiB}",
            size=th.SIZE_MATH_SMALL,
        )
        f2 = kit.formula(r"4096 \times 16 = 65{,}536\ \text{bytes}", r"= 64\ \mathrm{KiB}", size=th.SIZE_MATH_SMALL)
        f2[1].set_color(self.color)
        f2_note = kit.label("only the 16 full-attention layers", size=th.SIZE_LABEL, color=th.INK_3)
        math = VGroup(f1, VGroup(f2, f2_note).arrange(DOWN, buff=0.15, aligned_edge=LEFT)).arrange(
            DOWN, buff=0.6, aligned_edge=LEFT)
        layout = VGroup(pic, math).arrange(RIGHT, buff=0.8)
        self.place(layout)
        with self.narrate(
            "With the weights placed, the next question is what each conversation costs as it grows. "
            "When the model generates a token, each full-attention layer compares its query against the key of "
            "every earlier token, then mixes their values. "
            "Recomputing those keys and values on every step would redo the whole conversation. "
            "So the engine computes them once and stores them. "
            "That store is the KV cache. "
            "Each new token adds one key and one value per KV head, and each is 256 values wide. "
            "At 2 bytes a value, that's 4096 bytes in one layer. "
            "Only the 16 full-attention layers pay it. "
            "So every token costs 64 KiB."
        ) as t:
            t.wait_until("That store is")
            self.play(FadeIn(g), FadeIn(cap), run_time=0.8)
            t.wait_until("Each new token")
            self.play(g.fill(g.row(0), self.color), run_time=0.6)
            self.play(g.fill(g.row(1), self.color, tint=0.2), run_time=0.6)
            t.wait_until("At 2 bytes")
            self.play(Write(f1), run_time=1.4)
            t.wait_until("Only the 16")
            self.play(FadeIn(f2[0]), FadeIn(f2_note), run_time=0.8)
            t.wait_until("So every token")
            self.play(FadeIn(f2[1]), run_time=0.6)

    def cache_grows(self):
        self.slide("cache-grows", "The cache grows one column per token", book_section="s-kv-cache")
        tokens = 20
        g = kit.grid(16, tokens, cell=(0.36, 0.2), gap=0.03)
        side_l = kit.label("16 full-attention layers", size=th.SIZE_LABEL, color=th.INK_3)
        side_l.rotate(PI / 2).next_to(g, LEFT, buff=0.2)
        axis_l = kit.label("tokens", size=th.SIZE_LABEL, color=th.INK_3)
        arr = Arrow(LEFT * 0.6, RIGHT * 0.6, buff=0, color=th.INK_3, stroke_width=3, tip_length=0.16)
        axis = VGroup(axis_l, arr).arrange(RIGHT, buff=0.15).next_to(g, DOWN, buff=0.2)
        fig = VGroup(g, side_l, axis)

        def count(n):
            return kit.label(f"{n} tokens · {n * 64:,} KiB", size=th.SIZE_SMALL, color=self.color, weight=SEMIBOLD)

        counter = count(0)
        lin = kit.box("48 linear layers", sublabel="add nothing per token", color=STATE, width=4.4)
        total = kit.box("4096 tokens", sublabel="4096 × 64 KiB = 0.25 GiB", color=self.color, width=4.4)
        side = VGroup(counter, lin, total).arrange(DOWN, buff=0.6)
        layout = VGroup(fig, side).arrange(RIGHT, buff=0.7)
        self.place(layout)
        counter_slot = counter.get_center()
        with self.narrate(
            "Picture the cache as a grid. "
            "Each row is one of the 16 full-attention layers. "
            "Each column is one token. "
            "Every token you generate fills one more column, 64 KiB at a time. "
            "The 48 linear layers add nothing per token, because their state doesn't grow with the sequence. "
            "After 4096 tokens, one conversation holds 0.25 GiB of KV cache. "
            "Every further 4k tokens adds the same amount again."
        ) as t:
            self.play(FadeIn(g), FadeIn(side_l), run_time=0.8)
            t.wait_until("Each column")
            self.play(FadeIn(axis), FadeIn(counter), run_time=0.6)
            t.wait_until("Every token you")
            for c in range(tokens):
                new = count(c + 1).move_to(counter_slot)
                self.play(g.fill(g.col(c), self.color, tint=0.6), Transform(counter, new), run_time=0.22)
            t.wait_until("The 48 linear")
            self.play(FadeIn(lin, shift=UP * 0.1), run_time=0.7)
            t.wait_until("After 4096")
            self.play(FadeIn(total, shift=UP * 0.1), run_time=0.7)

    def gqa(self):
        self.slide("gqa", "Grouped-query attention makes it six times smaller", book_section="s-kv-cache")
        qs = VGroup(*[Square(side_length=0.32, fill_color=th.tint(WEIGHTS, 0.4), fill_opacity=1,
                             stroke_color=WEIGHTS, stroke_width=1.5) for _ in range(24)])
        qs.arrange(RIGHT, buff=0.1)
        for i in range(1, 4):
            for q in qs[6 * i:]:
                q.shift(RIGHT * 0.22)
        kvs = VGroup()
        for i in range(4):
            group = qs[6 * i:6 * i + 6]
            k = Rectangle(width=group.width, height=0.42, fill_color=th.tint(self.color, 0.45), fill_opacity=1,
                          stroke_color=self.color, stroke_width=2)
            k.move_to([group.get_center()[0], qs.get_center()[1] - 1.5, 0])
            kvs.add(k)
        links = VGroup(*[Line(q.get_bottom(), kvs[i // 6].get_top(), color=th.INK_4, stroke_width=1.5)
                         for i, q in enumerate(qs)])
        ql = kit.label("24 query heads", size=th.SIZE_LABEL, color=WEIGHTS, weight=MEDIUM).next_to(qs, UP, buff=0.18)
        kl = kit.label("4 KV heads, each shared by 6 query heads", size=th.SIZE_LABEL, color=self.color,
                       weight=MEDIUM).next_to(kvs, DOWN, buff=0.18)
        diagram = VGroup(qs, kvs, links, ql, kl)
        bc = kit.bar_compare(
            [("4 KV heads (GQA)", 64, self.color, "64 KiB per token"),
             ("24 KV heads", 384, th.INK_4, "384 KiB per token")],
            width=6.5, bar_height=0.42, gap=0.25,
        )
        col = VGroup(diagram, bc).arrange(DOWN, buff=0.7)
        self.place(col)
        with self.narrate(
            "Grouped-query attention is already doing heavy lifting here. "
            "Several query heads share one KV head. "
            "Each of the 4 KV heads serves 24 / 4 = 6 query heads. "
            "So the cache stores 4 heads instead of 24. "
            "Give every query head its own KV head, and the same 16 layers cost 384 KiB per token. "
            "That's six times larger, for the same model."
        ) as t:
            self.play(FadeIn(qs), FadeIn(ql), run_time=0.8)
            t.wait_until("Several query heads")
            self.play(Create(links), FadeIn(kvs), run_time=1.2)
            t.wait_until("Each of the 4")
            self.play(FadeIn(kl), run_time=0.6)
            t.wait_until("So the cache stores")
            self.play(FadeIn(bc.labels[0]), bc.grow(run_time=0.8)[0], FadeIn(bc.values[0]), run_time=0.8)
            t.wait_until("Give every query")
            self.play(FadeIn(bc.labels[1]), bc.grow(run_time=1.2)[1], run_time=1.2)
            self.play(FadeIn(bc.values[1]), run_time=0.5)

    def cache_table(self):
        self.slide("cache-table", "Batch and context multiply the cache", book_section="s-kv-cache")
        tab = kit.table(
            [
                ["4k", "0.25 GiB", "2.0 GiB", "8.0 GiB"],
                ["32k", "2.0 GiB", "16.0 GiB", "64.0 GiB"],
                ["128k", "8.0 GiB", "64.0 GiB", "256.0 GiB"],
            ],
            header=["Context", "Batch 1", "Batch 8", "Batch 32"],
            size=th.SIZE_BODY,
            align=["l", "r", "r", "r"],
            col_gap=1.1,
            row_gap=0.36,
        )
        corner = tab.cells[3][3]
        hot = kit.mark(corner, color=th.tint(th.BAD, 0.25))
        rule = kit.formula(r"64\ \mathrm{KiB} \times L \times \text{batch}", size=th.SIZE_MATH_SMALL)
        card = kit.label("on a card that holds 80 GiB", size=th.SIZE_SMALL, color=th.BAD, weight=MEDIUM)
        card.next_to(corner, DOWN, buff=0.35).align_to(corner, RIGHT)
        key = kit.callout("key", r"The cache, not the weights, sets your maximum batch size.", width=10.5,
                          size=th.SIZE_SMALL)
        col = VGroup(rule, VGroup(hot, tab, card), key).arrange(DOWN, buff=0.5)
        self.place(col)
        with self.narrate(
            "The cache's size scales with batch size and context length, and with nothing else. "
            "Multiply 64 KiB by the tokens per sequence and by the number of sequences. "
            "One sequence at 4k context holds 0.25 GiB. "
            "Eight sequences at 32k hold 16 GiB. "
            "Now read the bottom-right cell. "
            "32 sequences at 128k context need 256 GiB of cache, on a card with 80. "
            "The cache, not the weights, sets your maximum batch size."
        ) as t:
            self.play(FadeIn(rule), run_time=0.8)
            t.wait_until("Multiply 64")
            self.play(FadeIn(tab.rules), FadeIn(row(tab, 0)), run_time=0.8)
            t.wait_until("One sequence at 4k")
            self.play(FadeIn(row(tab, 1)), run_time=0.6)
            t.wait_until("Eight sequences")
            self.play(FadeIn(row(tab, 2)), run_time=0.6)
            t.wait_until("Now read")
            self.play(FadeIn(row(tab, 3)), run_time=0.6)
            self.play(FadeIn(hot), run_time=0.5)
            t.wait_until("32 sequences at 128k")
            self.play(FadeIn(card, shift=UP * 0.1), Indicate(corner, color=th.BAD), run_time=1.0)
            t.wait_until("The cache, not")
            self.play(FadeIn(key, shift=UP * 0.1), run_time=0.7)

    # Recurrent state ----------------------------------------------------------

    def recurrent_state(self):
        self.slide("recurrent-state", "Linear layers keep a fixed 147.8 MiB", book_section="s-recurrent-state")
        n = 9
        cells = VGroup(*[Square(side_length=0.48, fill_color=th.tint(self.color, 0.5), fill_opacity=1,
                                stroke_color=self.color, stroke_width=1.5) for _ in range(n)]).arrange(RIGHT, buff=0.06)
        slot = Rectangle(width=cells.width + 0.2, height=0.7, stroke_color=th.RULE_STRONG, stroke_width=1.5)
        slot.move_to(cells)
        t_head = kit.label("Full attention: a transcript", size=th.SIZE_SMALL, color=self.color, weight=SEMIBOLD)
        t_sub = kit.label("appends every token", size=th.SIZE_LABEL, color=th.INK_3)
        left = VGroup(t_head, VGroup(slot, cells), t_sub).arrange(DOWN, buff=0.28)
        summ = kit.grid(4, 4, cell=0.4)
        summ.paint(VGroup(*[summ.cell(r, c) for r in range(4) for c in range(4)]), STATE, tint=0.25)
        s_head = kit.label("Linear attention: a summary", size=th.SIZE_SMALL, color=STATE, weight=SEMIBOLD)
        s_sub = kit.label("overwritten every token", size=th.SIZE_LABEL, color=th.INK_3)
        right = VGroup(s_head, summ, s_sub).arrange(DOWN, buff=0.28)
        top = VGroup(left, right).arrange(RIGHT, buff=1.2, aligned_edge=UP)
        tab = kit.table(
            [
                ["Delta-rule state, float32", r"$48 \times 128 \times 128 \times 4$", "3.0 MiB"],
                ["Convolution window, bf16", r"$10{,}240 \times 4 \times 2$", "80 KiB"],
                [r"\textbf{One layer}", "", r"\textbf{3.078 MiB}"],
                [r"\textbf{48 layers}", r"$3{,}227{,}648 \times 48 = 154{,}927{,}104$ bytes",
                 r"\textbf{147.8 MiB}"],
            ],
            header=["Part", "Bytes", "Size"],
            size=th.SIZE_SMALL,
            align=["l", "l", "r"],
            row_gap=0.22,
        )
        col = VGroup(top, tab).arrange(DOWN, buff=0.5)
        self.place(col)
        for c in cells:
            c.save_state()
        shades = [0.15, 0.45, 0.3, 0.55, 0.25, 0.4]
        r = kit.rng(202)
        with self.narrate(
            "The other 48 layers need a matching answer. "
            "Full attention keeps a transcript, and appends to it every token. "
            "Linear attention keeps a fixed-size summary, and overwrites it every token. "
            "So its size never changes. "
            "The summary has two parts. "
            "Each of the 48 value heads keeps a 128 by 128 matrix. "
            "It's stored in float32, 4 bytes per element, for numerical stability. "
            "That's 3 MiB per layer. "
            "A short convolution also keeps its last 4 steps, which adds 80 KiB. "
            "Every one of the 48 linear layers keeps its own copy, for 147.8 MiB per sequence. "
            "A sequence at 128k context carries exactly as much as one ten tokens long."
        ) as t:
            self.play(FadeIn(t_head), FadeIn(slot), FadeIn(t_sub), FadeIn(s_head), FadeIn(summ), FadeIn(s_sub),
                      run_time=0.8)
            t.wait_until("Full attention keeps")
            for c in cells[:5]:
                self.play(FadeIn(c, shift=LEFT * 0.15), run_time=0.3)
            t.wait_until("Linear attention keeps")
            for i, c in enumerate(cells[5:]):
                shade = shades[i % len(shades)]
                picks = VGroup(*[summ.cell(int(a), int(b)) for a, b in r.integers(0, 4, size=(6, 2))])
                self.play(FadeIn(c, shift=LEFT * 0.15), summ.fill(picks, STATE, tint=shade), run_time=0.45)
            t.wait_until("Each of the 48")
            self.play(FadeIn(tab.rules), FadeIn(row(tab, 0)), FadeIn(row(tab, 1)), run_time=0.8)
            t.wait_until("A short convolution")
            self.play(FadeIn(row(tab, 2)), FadeIn(row(tab, 3)), run_time=0.6)
            t.wait_until("Every one of the 48")
            self.play(FadeIn(row(tab, 4)), run_time=0.6)

    def break_even(self):
        self.slide("break-even", "The hybrid wins past 788 tokens", book_section="s-the-break-even-point")
        ax = kit.axes([0, 1600, 200], [0, 400, 100], width=6.6, height=4.0, x_numbers=[0, 400, 800, 1200, 1600],
                      y_numbers=[0, 100, 200, 300, 400], x_label=r"L\ \text{(tokens)}",
                      y_label=r"\text{MiB per sequence}")
        dense = ax.plot(lambda x: 0.25 * x, x_range=[0, 1600], color=DENSE, stroke_width=th.STROKE_CURVE)
        hybrid = ax.plot(lambda x: 147.75 + 0.0625 * x, x_range=[0, 1600], color=self.color,
                         stroke_width=th.STROKE_CURVE)
        win = ax.get_area(hybrid, x_range=[788, 1600], bounded_graph=dense, color=th.GOOD, opacity=0.18)
        win.set_stroke(width=0)
        cross = Dot(ax.c2p(788, 197), radius=0.08, color=th.INK)
        vl = kit.vline(ax, 788, 197)
        cl = kit.label("788 tokens, 197 MiB", size=th.SIZE_LABEL, color=th.INK, weight=MEDIUM)
        cl.next_to(ax.c2p(788, 40), RIGHT, buff=0.12)
        d_l = kit.label("dense: 256 KiB/token", size=th.SIZE_LABEL, color=DENSE, weight=MEDIUM)
        d_l.next_to(ax.c2p(1100, 275), LEFT, buff=0.15).shift(UP * 0.2)
        h_l = VGroup(kit.label("hybrid: 147.8 MiB", size=th.SIZE_LABEL, color=self.color, weight=MEDIUM),
                     kit.label("+ 64 KiB/token", size=th.SIZE_LABEL, color=self.color, weight=MEDIUM)).arrange(
            DOWN, buff=0.08, aligned_edge=RIGHT)
        h_l.next_to(ax.c2p(1600, 247.75), DOWN, buff=0.45).align_to(ax.c2p(1600, 0), RIGHT)
        plot = VGroup(ax, win, dense, hybrid, vl, cross, cl, d_l, h_l)
        f = kit.formula(r"L = \frac{154{,}927{,}104}{196{,}608} = 788", size=th.SIZE_MATH_SMALL)
        f_note = kit.prose(r"fixed state over KV bytes saved per token (192 KiB)", size=th.SIZE_LABEL,
                           color=th.INK_3, width=4.6)
        tab = kit.table(
            [["32k", "8.0 GiB", "2.14 GiB"], ["128k", "32.0 GiB", "8.14 GiB"]],
            header=["Context", "Dense", "Hybrid"],
            size=th.SIZE_SMALL,
            align=["l", "r", "r"],
        )
        ratio = kit.prose(r"Ratio $\to 64/16 = 4$", size=th.SIZE_SMALL, color=th.INK_2, width=None)
        side = VGroup(VGroup(f, f_note).arrange(DOWN, buff=0.2, aligned_edge=LEFT), tab, ratio).arrange(
            DOWN, buff=0.5, aligned_edge=LEFT)
        layout = VGroup(plot, side).arrange(RIGHT, buff=0.6)
        self.place(layout)
        with self.narrate(
            "You now have two per-sequence costs, one that grows and one that's fixed. "
            "Was the trade worth it? "
            "Compare with a hypothetical dense model, where all 64 layers use full attention. "
            "It pays 256 KiB per token, and nothing fixed. "
            "The hybrid pays 64 KiB per token, plus its fixed 147.8 MiB. "
            "The hybrid starts higher but climbs more slowly, so the two lines cross exactly once. "
            "Every token saves the hybrid 192 KiB. "
            "Divide the fixed state by that saving, and the lines cross at 788 tokens. "
            "Above that, the hybrid wins, and the margin keeps growing. "
            "At 128k context, the dense model needs 32 GiB per sequence, and the hybrid about 8. "
            "The ratio tends to 4, the layer ratio of 64 to 16. "
            "Almost every real request is longer than 788 tokens, so in practice the hybrid always wins."
        ) as t:
            self.play(Create(ax), run_time=0.8)
            t.wait_until("It pays 256")
            self.play(Create(dense), FadeIn(d_l), run_time=1.2)
            t.wait_until("The hybrid pays")
            self.play(Create(hybrid), FadeIn(h_l), run_time=1.2)
            t.wait_until("Divide the fixed")
            self.play(Create(vl), FadeIn(cross, scale=0.5), FadeIn(cl), run_time=0.8)
            self.play(Write(f), FadeIn(f_note), run_time=1.0)
            t.wait_until("Above that")
            self.play(FadeIn(win), run_time=0.8)
            t.wait_until("At 128k")
            self.play(FadeIn(tab), run_time=0.8)
            t.wait_until("The ratio tends")
            self.play(FadeIn(ratio), run_time=0.6)

    # Batch --------------------------------------------------------------------

    def batch_formula(self):
        self.slide("batch-formula", "Weights first, then equal slots", book_section="s-the-largest-batch-that-fits")
        unit = 11.5 / 80
        hgt = 0.8
        card = frame_box(80 * unit, hgt)
        w = seg(0, 50.1 * unit, hgt, WEIGHTS)
        o = seg(50.1 * unit, 6 * unit, hgt, OVERHEAD)
        left = seg(56.1 * unit, 23.9 * unit, hgt, self.color, opacity=0.18)
        w_t = kit.label("W = 53.8 GB (50.1 GiB)", size=th.SIZE_LABEL, color=th.CANVAS, weight=MEDIUM).move_to(w)
        o_t = kit.label("O = 6 GiB", size=th.SIZE_LABEL, color=th.INK_2).next_to(o, UP, buff=0.15)
        l_t = kit.label("23.9 GiB left", size=th.SIZE_SMALL, color=self.color, weight=SEMIBOLD).move_to(left)
        c_t = kit.label("C = 80 GiB", size=th.SIZE_LABEL, color=th.INK_2).next_to(card, UP, buff=0.15)
        c_t.align_to(card, LEFT)
        bar = VGroup(card, w, o, left, w_t, o_t, l_t, c_t)
        o_note = kit.label("CUDA context, driver, activation workspace", size=th.SIZE_LABEL, color=th.INK_3)
        o_note.next_to(card, DOWN, buff=0.15).align_to(card, RIGHT)
        bar.add(o_note)
        m = kit.formula(r"M_{\text{seq}}(L) =", r"154{,}927{,}104", r"+", r"65{,}536\,L", size=th.SIZE_MATH_SMALL)
        m[1].set_color(STATE)
        m[3].set_color(self.color)
        b = kit.formula(r"B_{\max} = \left\lfloor \frac{C - W - O}{M_{\text{seq}}(L)} \right\rfloor",
                        size=th.SIZE_MATH)
        col = VGroup(bar, m, b).arrange(DOWN, buff=0.6)
        self.place(col)
        with self.narrate(
            "Now the question the chapter opened with: how many conversations fit at once? "
            "Picture packing a suitcase. "
            "The weights go in first, at 53.8 GB. "
            "Then a fixed overhead of 6 GiB, for the CUDA context, the driver, and an activation workspace. "
            "That leaves about 23.9 GiB. "
            "One sequence costs its recurrent state, plus 64 KiB per token. "
            "Divide the space left by that cost, and round down to whole sequences."
        ) as t:
            t.wait_until("Picture packing")
            self.play(FadeIn(card), FadeIn(c_t), run_time=0.6)
            t.wait_until("The weights go")
            self.play(GrowFromEdge(w, LEFT), run_time=1.0)
            self.play(FadeIn(w_t), run_time=0.4)
            t.wait_until("Then a fixed")
            self.play(GrowFromEdge(o, LEFT), FadeIn(o_t), FadeIn(o_note), run_time=0.8)
            t.wait_until("That leaves")
            self.play(FadeIn(left), FadeIn(l_t), run_time=0.6)
            t.wait_until("One sequence costs")
            self.play(Write(m), run_time=1.2)
            t.wait_until("Divide the space")
            self.play(Write(b), run_time=1.2)

    def batch_fits(self):
        self.slide("batch-fits", "11 sequences at 32k, 60 at 4k", book_section="s-the-largest-batch-that-fits")
        width = 10.5
        unit = width / 23.895  # Manim units per GiB of the space left
        hgt = 0.7
        state = 154_927_104 / 2**30

        def row_bar(kv_gib, count, caption):
            frame = frame_box(width, hgt)
            slots = VGroup()
            per = state + kv_gib
            for i in range(count):
                x = i * per * unit
                s1 = seg(x, state * unit, hgt, STATE, opacity=0.9)
                s2 = seg(x + state * unit, kv_gib * unit, hgt, self.color, opacity=0.75)
                sep = Line([x + per * unit, -hgt / 2, 0], [x + per * unit, hgt / 2, 0], color=th.CANVAS,
                           stroke_width=2)
                slots.add(VGroup(s1, s2, sep))
            cap = kit.label(caption, size=th.SIZE_SMALL, color=th.INK_2).next_to(frame, UP, buff=0.15)
            cap.align_to(frame, LEFT)
            num = kit.label(str(count), size=th.SIZE_HEADING, color=th.INK, weight=SEMIBOLD)
            num.next_to(frame, RIGHT, buff=0.35)
            return VGroup(frame, slots, cap, num)

        r32 = row_bar(2.0, 11, "32k context: 2.14 GiB per sequence, 11.14 fit")
        r4 = row_bar(0.25, 60, "4k context: 0.39 GiB per sequence, 60.6 fit")
        legend = VGroup(swatch(STATE, "recurrent state"), swatch(self.color, "KV cache")).arrange(RIGHT, buff=0.6)
        zero = kit.prose(r"40 GiB card: $C - W - O < 0$ before you divide, so $B_{\max} = 0$.", size=th.SIZE_SMALL,
                         color=th.BAD, width=None)
        col = VGroup(legend, r32, r4, zero).arrange(DOWN, buff=0.55, aligned_edge=LEFT)
        self.place(col)
        with self.narrate(
            "At 32k context, one sequence costs 2.14 GiB. "
            "The space left holds 11.14 of those. "
            "Memory comes in whole sequences, so 11 fit. "
            "At 4k context, one sequence costs only 0.39 GiB. "
            "Now 60 fit. "
            "On a 40 GiB card, the numerator is negative before you divide anything. "
            "The weights alone are bigger than the card, so the answer is 0 at every context length."
        ) as t:
            self.play(FadeIn(legend), FadeIn(r32[0]), FadeIn(r32[2]), FadeIn(r4[0]), run_time=0.6)
            t.wait_until("The space left")
            self.play(LaggedStart(*[FadeIn(s, shift=RIGHT * 0.1) for s in r32[1]], lag_ratio=0.25), run_time=2.0)
            t.wait_until("Memory comes")
            self.play(FadeIn(r32[3], scale=0.7), run_time=0.5)
            t.wait_until("At 4k context")
            self.play(FadeIn(r4[2]), LaggedStart(*[FadeIn(s) for s in r4[1]], lag_ratio=0.08), run_time=2.4)
            t.wait_until("Now 60")
            self.play(FadeIn(r4[3], scale=0.7), run_time=0.5)
            t.wait_until("On a 40 GiB")
            self.play(FadeIn(zero), run_time=0.7)

    def batch_levers(self):
        self.slide("batch-levers", "Three things follow from the formula",
                   book_section="s-the-largest-batch-that-fits")

        def head(n, text):
            num = kit.label(n, size=th.SIZE_BODY, color=self.color, weight=SEMIBOLD)
            name = kit.label(text, size=th.SIZE_SMALL, color=th.INK, weight=SEMIBOLD)
            return VGroup(num, name).arrange(RIGHT, buff=0.25)

        h1 = head("1", "The floor matters")
        v1 = kit.formula(r"\lfloor 11.14 \rfloor = 11", size=th.SIZE_MATH_SMALL)
        h2 = head("2", "The fixed state hurts short requests")
        w = 4.6

        def share(frac, name, pct):
            frame = frame_box(w, 0.42)
            a = seg(0, w * frac, 0.42, STATE, opacity=0.9)
            b = seg(w * frac, w * (1 - frac), 0.42, self.color, opacity=0.6)
            lab = kit.label(name, size=th.SIZE_LABEL, color=th.INK_2).next_to(frame, LEFT, buff=0.25)
            p = kit.label(pct, size=th.SIZE_LABEL, color=STATE, weight=SEMIBOLD).next_to(frame, RIGHT, buff=0.2)
            return VGroup(frame, a, b, lab, p)

        s32 = share(154_927_104 / 2_302_410_752, "32k", "7% fixed")
        s4 = share(154_927_104 / 423_362_560, "4k", "37% fixed")
        v2 = VGroup(s32, s4).arrange(DOWN, buff=0.25)
        s4.shift(RIGHT * (s32[0].get_left()[0] - s4[0].get_left()[0]))
        h3 = head("3", "Every term is a lever")
        v3 = kit.bar_compare(
            [("bf16 weights", 23.9, th.INK_4, "23.9 GiB left"), ("int8 weights", 48.9, th.GOOD, "48.9 GiB left")],
            width=5.0, bar_height=0.4, gap=0.2, size=th.SIZE_LABEL,
        )
        rows = VGroup(*[VGroup(h, v).arrange(RIGHT, buff=0.6) for h, v in ((h1, v1), (h2, v2), (h3, v3))])
        for r in rows:
            r[1].shift(RIGHT * (7.6 - (r[1].get_left()[0] - r[0].get_left()[0])))
        rows.arrange(DOWN, buff=0.6, aligned_edge=LEFT)
        self.place(rows)
        with self.narrate(
            "Three things follow from that formula. "
            "First, the floor matters. "
            "Memory is allocated in whole sequences, so 11.14 sequences is 11. "
            "Chapter 15's paged cache is what lets you stop rounding down so aggressively. "
            "Second, the fixed state is 7 percent of a 32k sequence. "
            "At 4k, it's 37 percent. "
            "The hybrid's constant hurts most exactly where you want large batches, which is short requests. "
            "Third, every term is a lever. "
            "With int8 quantization, each weight takes 1 byte instead of 2. "
            "That moves the space left from 23.9 GiB to 48.9 GiB, which roughly doubles the batch."
        ) as t:
            t.wait_until("First, the floor")
            self.play(FadeIn(h1), run_time=0.5)
            t.wait_until("Memory is allocated")
            self.play(Write(v1), run_time=0.8)
            t.wait_until("Second, the fixed")
            self.play(FadeIn(h2), FadeIn(s32[0]), FadeIn(s32[3]), GrowFromEdge(s32[1], LEFT),
                      GrowFromEdge(s32[2], LEFT), run_time=0.9)
            self.play(FadeIn(s32[4]), run_time=0.4)
            t.wait_until("At 4k, it's")
            self.play(FadeIn(s4[0]), FadeIn(s4[3]), GrowFromEdge(s4[1], LEFT), GrowFromEdge(s4[2], LEFT),
                      run_time=0.9)
            self.play(FadeIn(s4[4]), run_time=0.4)
            t.wait_until("Third, every term")
            self.play(FadeIn(h3), FadeIn(v3.labels[0]), v3.grow(run_time=0.8)[0], FadeIn(v3.values[0]),
                      run_time=0.8)
            t.wait_until("That moves")
            self.play(FadeIn(v3.labels[1]), v3.grow(run_time=1.0)[1], run_time=1.0)
            self.play(FadeIn(v3.values[1]), run_time=0.4)

    # Activations and the budget ---------------------------------------------------

    def activations(self):
        self.slide("activations", "Activations are transient, but not small", book_section="s-activations")
        mlp_a = Rectangle(width=4.2, height=0.5, fill_color=th.tint(ACT, 0.4), fill_opacity=1, stroke_color=ACT,
                          stroke_width=2)
        mlp_b = mlp_a.copy()
        pair = VGroup(mlp_a, mlp_b).arrange(DOWN, buff=0.12)
        ga = kit.label("gate branch", size=th.SIZE_LABEL, color=th.INK_2).move_to(mlp_a)
        gb = kit.label("up branch", size=th.SIZE_LABEL, color=th.INK_2).move_to(mlp_b)
        m_head = kit.label("MLP hidden, 4096-token prefill chunk", size=th.SIZE_SMALL, color=th.INK, weight=SEMIBOLD)
        m_f = kit.formula(r"4096 \times 17{,}408 \times 2 = 136\ \mathrm{MiB}", size=th.SIZE_MATH_SMALL)
        m_n = kit.label("SwiGLU holds two at once", size=th.SIZE_LABEL, color=th.INK_3)
        left = VGroup(m_head, VGroup(pair, ga, gb), m_f, m_n).arrange(DOWN, buff=0.35)
        side = 2.6
        big = Square(side_length=side, fill_color=th.tint(th.BAD, 0.25), fill_opacity=1, stroke_color=th.BAD,
                     stroke_width=2)
        small = Square(side_length=side / 2, fill_color=th.tint(th.BAD, 0.5), fill_opacity=1, stroke_color=th.BAD,
                       stroke_width=2)
        small.align_to(big, DL)
        q_l = kit.math_label(r"q_{\text{len}} = 8192", size=th.SIZE_LABEL, color=th.INK_3)
        q_l.rotate(PI / 2).next_to(big, LEFT, buff=0.12)
        k_l = kit.math_label(r"kv_{\text{len}} = 8192", size=th.SIZE_LABEL, color=th.INK_3).next_to(big, UP, buff=0.12)
        grow_l = kit.label("2× context, 4× scores", size=th.SIZE_LABEL, color=th.BAD, weight=MEDIUM)
        sq = VGroup(big, small, q_l, k_l)
        s_head = kit.label("Attention scores, materialized", size=th.SIZE_SMALL, color=th.INK, weight=SEMIBOLD)
        s_f = kit.formula(r"24 \times 8192 \times 8192 \times 2 = 3.0\ \mathrm{GiB}", size=th.SIZE_MATH_SMALL)
        right = VGroup(s_head, sq, s_f, grow_l).arrange(DOWN, buff=0.3)
        layout = VGroup(left, right).arrange(RIGHT, buff=1.0)
        self.place(layout)
        with self.narrate(
            "The weights and the caches stay on the card while a sequence lives. "
            "Activations, the intermediate tensors a forward pass creates and then frees, don't. "
            "They're transient, but not small. "
            "During prefill, the engine can push a 4096-token chunk through the MLP at once. "
            "Its hidden activation then takes 136 MiB. "
            "SwiGLU holds two of those at once. "
            "Attention scores are worse, if you materialize them. "
            "There's one score for every pair of positions, in every head. "
            "At 8k context with 24 heads, that's 3.0 GiB for one sequence. "
            "Double the context, and the scores grow four times. "
            "Not materializing them is exactly what FlashAttention is for, in chapter 14."
        ) as t:
            t.wait_until("During prefill")
            self.play(FadeIn(m_head), GrowFromEdge(mlp_a, LEFT), FadeIn(ga), run_time=0.8)
            t.wait_until("Its hidden")
            self.play(FadeIn(m_f), run_time=0.7)
            t.wait_until("SwiGLU holds")
            self.play(GrowFromEdge(mlp_b, LEFT), FadeIn(gb), FadeIn(m_n), run_time=0.8)
            t.wait_until("Attention scores are")
            self.play(FadeIn(s_head), run_time=0.5)
            t.wait_until("There's one score")
            self.play(GrowFromEdge(big, DL), FadeIn(q_l), FadeIn(k_l), run_time=1.0)
            t.wait_until("At 8k context")
            self.play(FadeIn(s_f), run_time=0.7)
            t.wait_until("Double the context")
            self.play(FadeIn(small), run_time=0.6)
            self.play(Indicate(big, color=th.BAD, scale_factor=1.03), FadeIn(grow_l), run_time=0.8)

    def budget(self):
        self.slide("budget", "A budget that works", book_section="s-a-budget-that-works")
        unit = 11.5 / 80
        hgt = 0.8
        items = [
            ("Weights, bf16", 50.1, WEIGHTS, "50.1 GiB"),
            ("CUDA context and driver", 1.0, th.INK_3, "~1.0 GiB"),
            ("Activation workspace", 3.0, ACT, "~3.0 GiB"),
            ("Fragmentation headroom", 2.0, th.WARN, "~2.0 GiB"),
            ("Left for cache", 23.9, self.color, "~23.9 GiB"),
        ]
        card = frame_box(80 * unit, hgt)
        segs = VGroup()
        x = 0.0
        for _, size, color, _ in items:
            segs.add(seg(x, size * unit, hgt, color, opacity=0.85 if size < 20 or color is WEIGHTS else 0.35))
            x += size * unit
        bar = VGroup(card, segs)
        rows = [[swatch(c, n, size=th.SIZE_SMALL), s] for n, _, c, s in items]
        tab = kit.table(rows, header=["Item", "Size"], size=th.SIZE_SMALL, align=["l", "r"], row_gap=0.2)
        fits = kit.bullets(
            [
                r"KV cache alone: about 391k tokens at 64 KiB each.",
                r"With 147.8 MiB of state per sequence: batch 11 at 32k, 60 at 4k, or any mix.",
            ],
            size=th.SIZE_SMALL,
            width=5.2,
        )
        lower = VGroup(tab, fits).arrange(RIGHT, buff=0.8, aligned_edge=UP)
        col = VGroup(bar, lower).arrange(DOWN, buff=0.6)
        self.place(col)
        with self.narrate(
            "Now you can put the whole card on one page. "
            "The weights take 50.1 GiB. "
            "The CUDA context and driver take about 1 GiB. "
            "The activation workspace takes about 3 GiB. "
            "Fragmentation headroom takes about 2 GiB more. "
            "That leaves about 23.9 GiB for caches. "
            "If it held only KV cache, that would store about 391k tokens. "
            "Once you subtract each sequence's recurrent state, it holds batch 11 at 32k context, batch 60 at 4k, "
            "or any mix in between."
        ) as t:
            self.play(FadeIn(card), FadeIn(tab.rules), FadeIn(row(tab, 0)), run_time=0.6)
            for i, cue in enumerate(["The weights take", "The CUDA context", "The activation workspace",
                                     "Fragmentation headroom", "That leaves"]):
                t.wait_until(cue)
                self.play(GrowFromEdge(segs[i], LEFT), FadeIn(row(tab, i + 1)), run_time=0.6)
            t.wait_until("If it held")
            self.play(FadeIn(fits[0], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("Once you subtract")
            self.play(FadeIn(fits[1], shift=RIGHT * 0.1), run_time=0.6)

    def what_goes_wrong(self):
        self.slide("what-goes-wrong", "Four ways a budget breaks", book_section="s-what-goes-wrong")
        cards = [
            ("GB is not GiB", r"53.8 GB is 50.1 GiB. Mixing them loses 3.7 GiB: two more sequences at 32k."),
            ("Per sequence, not per batch", r"2 GiB of cache at 32k is one sequence. Batch 8 needs 16 GiB."),
            ("Sizing for the average", r"A 32k prompt in a 4k plan forces a preemption, or an out-of-memory "
                                       r"error mid-pass."),
            ("Ignoring fragmentation", r"Enough free memory in total can still fail one large request."),
        ]
        boxes = [kit.callout("warning", body, title=title, width=6.0, size=th.SIZE_SMALL) for title, body in cards]
        grid = VGroup(VGroup(boxes[0], boxes[1]).arrange(RIGHT, buff=0.4, aligned_edge=UP),
                      VGroup(boxes[2], boxes[3]).arrange(RIGHT, buff=0.4, aligned_edge=UP)).arrange(DOWN, buff=0.4)
        self.place(grid)
        with self.narrate(
            "These mistakes turn a budget that fits on paper into an out-of-memory error. "
            "First, confusing GB with GiB. "
            "53.8 GB of weights is 50.1 GiB. "
            "Treat both as GiB, and you lose 3.7 GiB, which is two more sequences at 32k. "
            "Second, forgetting that the KV cache is per sequence. "
            "Batch 8 at 32k is 16 GiB, not 2. "
            "Third, sizing the cache for the average request. "
            "An engine that assumes 4k and meets a 32k prompt must preempt another sequence, "
            "or run out of memory in the middle of a forward pass. "
            "Fourth, ignoring fragmentation. "
            "An allocator that has served many sizes can fail a large request, even when the free total is enough."
        ) as t:
            for b, cue in zip(boxes, ["First, confusing", "Second, forgetting", "Third, sizing", "Fourth, ignoring"]):
                t.wait_until(cue)
                self.play(FadeIn(b, shift=UP * 0.1), run_time=0.7)
