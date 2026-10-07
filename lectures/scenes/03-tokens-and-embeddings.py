"""Chapter 03, tokens and embeddings.

Teaches content/chapters/03-tokens-and-embeddings.md in its order. Every
number below was recomputed in Python (exact integers, then rounded as the
chapter rounds them) and matches the chapter.

Config: V = 248,320 (vocab_size), d = 5120 (hidden_size), 2 bytes per
bfloat16 value, total weights 53.8 GB and 26.9B parameters (chapters 1 and 2).

UTF-8 (Python str.encode):
- a = 61; é = C3 A9; ж = D0 B6; क = E0 A4 95; 中 = E4 B8 AD;
  U+1F600 = F0 9F 98 80; 👍🏽 = F0 9F 91 8D + F0 9F 8F BD = 8 bytes.
- Lead-byte patterns 0xxxxxxx, 110xxxxx, 1110xxxx, 11110xxx; continuation
  bytes 10xxxxxx. E0 = 1110 0000 (three-byte lead); A4, 95 = 10xx xxxx.
- The render image has fonts for Latin, Cyrillic, and Devanagari, but not
  for CJK or emoji, so those two show as code points on screen.

Byte-pair encoding:
- |V| = 256 + M + S; 248,320 - 256 = 248,064 merges plus special tokens, so
  "roughly 248,000 merges" with a few dozen specials.
- Toy example: corpus low, lower, lowest; merges (l, o), (lo, w), (e, s),
  (es, t) in that order, as the chapter gives them. lowest = 6 bytes, then
  2 tokens (low, est).
- Toy vocabulary IDs (illustrative, labelled on screen as a toy): bytes keep
  their byte values (s = 0x73 = 115, l = 108, o = 111, w = 119, e = 101,
  t = 116); the four merges get IDs 256 to 259 (lo, low, es, est). So
  lowest = [257, 259] and the unseen word slowest = s + low + est =
  [115, 257, 259], which applying the four merges in order produces.
- Invalid UTF-8 example: C3 28 (a two-byte lead followed by a
  non-continuation byte). Null byte 00. JPEG data FF D8 7A (illustrative).

Generation config: eos_token_id = [248046, 248044]; pad_token_id = 248044.

What the vocabulary costs:
- 248,320 x 5120 = 1,271,398,400 = 1.27B parameters.
- x 2 bytes = 2,542,796,800 B = 2.54 GB.
- Untied output projection: x 2 = 5,085,593,600 B = 5.09 GB.
- 5.0856 / 53.8 = 0.0945 = 9.5% (also 9.45% of 53,791,959,040 B exactly).
- 2.54B of 26.9B parameters (2 x 1.27B). Halving the vocabulary returns
  5.09 / 2 = 2.54 GB.

The embedding layer:
- One-hot product: 2Vd = 2 x 248,320 x 5120 = 2,542,796,800 FLOP = 2.54
  GFLOP, and it reads 2.54 GB.
- Gather: 5120 x 2 = 10,240 B = 10 KiB, zero arithmetic.
- Ratio 2,542,796,800 / 10,240 = 248,320 = V exactly.
- 2000-token prompt: 2000 x 10 KiB = 20,000 KiB = 20,480,000 B, "about
  20 MB" (the chapter writes 20 MB).
- sqrt(5120) = 71.554, "about 71.6".

The output projection:
- 2 x 5120 x 248,320 = 2.54 GFLOP; reads 2.54 GB; 1 FLOP per byte.
- A100 ridge: 312e12 / 1935e9 = 161.2, "about 161".
- Decode reads 53.8 - 2.54 = 51.26 = 51.3 GB; 2.54 / 51.26 = 4.96%, "just
  under 5%".
- Without the slice: 2000 x 248,320 x 2 = 993,280,000 B = 993 MB, and
  2 x 2000 x 5120 x 248,320 = 5.0856e12 = 5.09 TFLOP. With it:
  248,320 x 2 = 496,640 B = 497 KB, and 2.54 GFLOP.
- Untying costs 1,271,398,400 / 26,895,979,520 = 4.73%, "4.7%".

Detokenization: "क ok" = E0 A4 95 20 6F 6B; two-byte tokens [E0 A4],
[95 20], [6F 6B]. Decoded alone with errors="replace": "�", "� ", "ok".
Accumulated with errors="ignore": "" then "क " then "ok" (Python checked).
"""

from manim import *

import kit
from kit import theme as th

# The chapter's maths colours: \hla bytes, \hlb merges, \hlc specials, \hld one-hot.
BASE = th.HL_A
MERGE = th.HL_B
SPECIAL = th.HL_C
ONEHOT = th.HL_D
LEAD = th.ACCENT  # a UTF-8 lead byte
CONT = th.INFO  # a UTF-8 continuation byte
DEVA = "Noto Sans Devanagari"

TEMPLATE = """\
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is a KV cache?<|im_end|>
<|im_start|>assistant
"""

EMBED = """\
x = self.embed_tokens(input_ids)
"""

LM_HEAD = """\
x = self.norm(x)
if last_token_only:
    x = x[:, -1:, :]
return self.lm_head(x)
"""

TYING = """\
embed = self.vocab_size * h
lm_head = 0 if self.tie_word_embeddings else self.vocab_size * h
"""


def cell(text, w, h, color=None, tint=0.35, size=th.SIZE_SMALL, font=th.MONO, ink=th.INK,
         stroke=th.RULE_STRONG):
    """A rectangle with optional centred text: a byte, a token, a table row."""
    rect = Rectangle(width=w, height=h, stroke_color=stroke, stroke_width=th.STROKE_HAIRLINE,
                     fill_color=th.RAISE if color is None else th.tint(color, tint), fill_opacity=1)
    g = VGroup(rect)
    g.rect, g.text = rect, None
    if text:
        t = kit.label(text, size=size, color=ink, font=font)
        if t.width > w * 0.88:
            t.scale_to_fit_width(w * 0.88)
        t.move_to(rect)
        g.add(t)
        g.text = t
    return g


def strip(r, n, color, w=0.2, h=0.5, vertical=False):
    """A vector drawn as a strip of cells with random shades from generator `r`."""
    shades = r.uniform(0.12, 0.8, n)
    cells = VGroup(*[Rectangle(width=w, height=h, stroke_color=th.RULE_STRONG, stroke_width=1,
                               fill_color=th.tint(color, float(s)), fill_opacity=1) for s in shades])
    cells.arrange(DOWN if vertical else RIGHT, buff=0)
    return cells


def swatch(text, color, size=th.SIZE_LABEL):
    sq = Square(0.26, stroke_color=th.RULE_STRONG, stroke_width=th.STROKE_HAIRLINE,
                fill_color=th.tint(color, 0.35), fill_opacity=1)
    return VGroup(sq, kit.label(text, size=size, color=th.INK_2)).arrange(RIGHT, buff=0.16)


def row(tab, i):
    return VGroup(*tab.cells[i])


# BPE pieces: a piece of n characters is as wide as n byte cells, so a word
# keeps its width however it's split.
PW, PH, PG = 0.62, 0.62, 0.06


def piece(text, color=None):
    n = len(text)
    return cell(text, n * PW + (n - 1) * PG, PH, color=color, size=th.SIZE_SMALL)


def word(pieces, merged_color):
    return VGroup(*[piece(p, merged_color if len(p) > 1 else None) for p in pieces]).arrange(RIGHT, buff=PG)


def merge_anims(old, new_pieces, new):
    """ReplacementTransforms from the pieces of `old` to `new`, matched by character span."""
    anims = []
    i = 0
    for j, p in enumerate(new_pieces):
        count = 0
        grab = []
        while count < len(p):
            grab.append(old[i])
            count += old.lens[i]
            i += 1
        src = grab[0] if len(grab) == 1 else VGroup(*grab)
        anims.append(ReplacementTransform(src, new[j]))
    return anims


def word_state(pieces, color):
    w = word(pieces, color)
    w.lens = [len(p) for p in pieces]
    return w


class Lecture(kit.Lecture):
    def construct(self):
        self.title_slide(
            narration=(
                "Tokens and embeddings. "
                "The model never sees text, only integers. "
                "In this lecture, you follow a string through the tokenizer and into the embedding table. "
                "Then you find out what that table costs, and why it's the cheapest layer to run."
            )
        )
        self.pipeline()
        self.utf8()
        self.not_a_network()
        self.bpe_round()
        self.bpe_example()
        self.encode_order()
        self.pre_tokenize()
        self.byte_fallback()
        self.token_cost()
        self.chat_template()
        self.stop_tokens()
        self.vocab_cost()
        self.worth_it()
        self.one_hot()
        self.gather()
        self.residual_start()
        self.lm_head()
        self.slice_first()
        self.weight_tying()
        self.detokenize()
        self.closing_slide(
            [
                r"Apply merges in training order.",
                r"No unknown token; unfamiliar text costs more tokens.",
                r"Stop on every EOS ID; print the exact prompt.",
                r"Embedding: a 10 KiB gather. Output projection: 2.54 GB per token.",
            ],
            narration=(
                "Here's what to keep. "
                "Apply the merges in training order. "
                "Byte fallback means no unknown token, but unfamiliar text costs more tokens. "
                "Stop on every end-of-sequence ID, and print the exact prompt when quality drops. "
                "The embedding is a 10 KiB gather, while the output projection streams 2.54 GB per token. "
                "Next, you meet the normalization that every layer starts with."
            ),
            book_section="s-what-goes-wrong",
        )

    # Before you start ------------------------------------------------------------

    def pipeline(self):
        self.slide("pipeline", "The model only ever sees integers", book_section="s-before-you-start")
        text = kit.box('"lowest"', font=th.MONO, size=th.SIZE_BODY)
        tok = kit.box("Tokenizer", sublabel="CPU, no weights", width=2.6)
        ids = VGroup(cell("257", 1.1, 0.6, self.color), cell("259", 1.1, 0.6, self.color)).arrange(DOWN, buff=0.2)
        emb = kit.box("Embedding", sublabel="248,320 × 5120", color=th.INFO, width=2.8)
        r = kit.rng(3)
        vecs = VGroup(strip(r, 9, th.INFO), strip(r, 9, th.INFO)).arrange(DOWN, buff=0.2)
        flow = VGroup(text, tok, ids, emb, vecs).arrange(RIGHT, buff=0.75)
        caps = VGroup(
            kit.label("a string", size=th.SIZE_LABEL, color=th.INK_3).next_to(text, DOWN, buff=0.3),
            kit.label("token IDs (toy)", size=th.SIZE_LABEL, color=th.INK_3).next_to(ids, DOWN, buff=0.3),
            kit.label("5120 values each", size=th.SIZE_LABEL, color=th.INK_3).next_to(vecs, DOWN, buff=0.3),
        )
        low = min(c.get_y() for c in caps)
        for c in caps:
            c.set_y(low)
        w, h = 11.0, 0.6
        frame = Rectangle(width=w, height=h, stroke_color=th.INK_3, stroke_width=th.STROKE_BOX,
                          fill_color=th.RAISE, fill_opacity=1)
        share = Rectangle(width=w * 2.54 / 26.9, height=h, stroke_width=0, fill_color=self.color, fill_opacity=0.8)
        share.align_to(frame, LEFT)
        share_t = kit.label("2.54B (9.5%): embedding and output projection", size=th.SIZE_LABEL,
                            color=th.INK, weight=MEDIUM)
        share_t.next_to(share, RIGHT, buff=0.25)
        total_t = kit.label("26.9B parameters", size=th.SIZE_LABEL, color=th.INK_3)
        total_t.next_to(frame, UP, buff=0.15).align_to(frame, LEFT)
        bar = VGroup(frame, share, share_t, total_t)
        col = VGroup(VGroup(flow, caps), bar).arrange(DOWN, buff=1.1)
        self.place(col)
        arrows = [kit.arrow(a, b) for a, b in ((text, tok), (tok, ids), (ids, emb), (emb, vecs))]
        with self.narrate(
            "The model is a function on integers. "
            "Something outside the network, the tokenizer, turns a string into a list of integer IDs. "
            "On the network side, the first layer is the embedding: a table that maps each ID to a vector. "
            "Every token becomes a point in a space 5120 wide. "
            "The embedding and the output projection together hold 2.54 billion parameters. "
            "That's 9.5 percent of the weights, spent on the vocabulary."
        ) as t:
            self.play(FadeIn(text), FadeIn(caps[0]), run_time=0.6)
            t.wait_until("Something outside")
            self.play(GrowArrow(arrows[0].line), FadeIn(tok), run_time=0.8)
            self.play(GrowArrow(arrows[1].line), FadeIn(ids, shift=RIGHT * 0.1), FadeIn(caps[1]), run_time=0.8)
            self.play(Indicate(tok.rect, color=th.INK_3, scale_factor=1.05), run_time=0.8)
            t.wait_until("On the network side")
            self.play(GrowArrow(arrows[2].line), FadeIn(emb), run_time=0.8)
            t.wait_until("Every token becomes")
            self.play(GrowArrow(arrows[3].line), LaggedStart(*[FadeIn(v, shift=RIGHT * 0.1) for v in vecs],
                                                           lag_ratio=0.3), FadeIn(caps[2]), run_time=1.0)
            t.wait_until("The embedding and the output")
            self.play(FadeIn(frame), FadeIn(total_t), run_time=0.5)
            self.play(GrowFromEdge(share, LEFT), run_time=0.8)
            t.wait_until("That's 9.5 percent")
            self.play(FadeIn(share_t, shift=RIGHT * 0.1), run_time=0.6)

    def utf8(self):
        self.slide("utf8", "UTF-8 spends one to four bytes per character", book_section="s-before-you-start")
        spec = [
            ("a", th.SANS, "ASCII", ["61"]),
            ("é", th.SANS, "Latin accents", ["C3", "A9"]),
            ("ж", th.SANS, "Cyrillic", ["D0", "B6"]),
            ("क", DEVA, "Devanagari, most CJK", ["E0", "A4", "95"]),
            ("U+1F600", th.MONO, "Emoji", ["F0", "9F", "98", "80"]),
        ]
        rows = VGroup()
        leads, conts = [], []
        for i, (ch, font, name, bs) in enumerate(spec):
            size = th.SIZE_LABEL if font == th.MONO else th.SIZE_HEADING
            g = kit.label(ch, size=size, color=th.INK, font=font)
            g.move_to([0.75, -i * 0.85, 0])
            n = kit.label(name, size=th.SIZE_SMALL, color=th.INK_3)
            n.move_to([1.85 + n.width / 2, -i * 0.85, 0])
            cs = VGroup(*[cell(b, 0.78, 0.56, LEAD if k == 0 else CONT, size=th.SIZE_SMALL) for k, b in enumerate(bs)])
            cs.arrange(RIGHT, buff=0.08)
            cs.move_to([6.0 + cs.width / 2, -i * 0.85, 0])
            leads.append(cs[0])
            conts.extend(cs[1:])
            rows.add(VGroup(g, n, cs))
        note = kit.label("Emoji with a skin-tone modifier: 8 bytes, as two code points", size=th.SIZE_LABEL,
                         color=th.INK_3)
        note.next_to(rows, DOWN, buff=0.35).align_to(rows, LEFT)
        left = VGroup(rows, note)

        pats = [("0xxxxxxx", "1 byte", LEAD), ("110xxxxx", "2 bytes", LEAD), ("1110xxxx", "3 bytes", LEAD),
                ("11110xxx", "4 bytes", LEAD), ("10xxxxxx", "continuation", CONT)]
        lines = VGroup()
        for p, d, c in pats:
            pm = kit.mono(p, size=th.SIZE_SMALL, color=c, weight=MEDIUM)
            dm = kit.label(d, size=th.SIZE_SMALL, color=th.INK_2)
            lines.add(VGroup(pm, dm))
        for i, (pm, dm) in enumerate(lines):
            pm.move_to([pm.width / 2, -i * 0.62, 0])
            dm.move_to([2.1 + dm.width / 2, -i * 0.62, 0])
        head = kit.label("The first byte says how many follow", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
        inner = VGroup(head, lines).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        bg = RoundedRectangle(width=inner.width + 0.7, height=inner.height + 0.7, corner_radius=0.12,
                              fill_color=th.SUNK, fill_opacity=1, stroke_width=0).move_to(inner)
        right = VGroup(bg, inner)
        layout = VGroup(left, right).arrange(RIGHT, buff=0.8)
        self.place(layout)
        with self.narrate(
            "UTF-8 encodes each character as one to four bytes. "
            "A plain ASCII letter takes 1 byte. "
            "Accented Latin letters and Cyrillic take 2. "
            "Devanagari and most Chinese characters take 3. "
            "An emoji takes 4. "
            "UTF-8 is a prefix code: the first byte of a character says how many bytes follow. "
            "Every later byte is marked as a continuation. "
            "So you can always tell whether a byte sequence ends mid-character, and that's what makes streaming text possible."
        ) as t:
            t.wait_until("A plain ASCII")
            self.play(FadeIn(rows[0]), run_time=0.6)
            t.wait_until("Accented Latin")
            self.play(LaggedStart(FadeIn(rows[1]), FadeIn(rows[2]), lag_ratio=0.4), run_time=0.9)
            t.wait_until("Devanagari and most")
            self.play(FadeIn(rows[3]), run_time=0.6)
            t.wait_until("An emoji takes 4")
            self.play(FadeIn(rows[4]), FadeIn(note), run_time=0.8)
            t.wait_until("UTF-8 is a prefix code")
            self.play(FadeIn(bg), FadeIn(head), *[FadeIn(l) for l in lines[:4]], run_time=0.8)
            self.play(*[Indicate(c, color=LEAD, scale_factor=1.1) for c in leads], run_time=1.0)
            t.wait_until("Every later byte")
            self.play(FadeIn(lines[4]), *[Indicate(c, color=CONT, scale_factor=1.1) for c in conts], run_time=1.0)

    # What a tokenizer is -------------------------------------------------------

    def not_a_network(self):
        self.slide("not-a-network", "A tokenizer is two tables and a loop",
                   book_section="s-what-a-tokenizer-is-and-isn-t")
        vocab = kit.box("A vocabulary", sublabel="byte strings → integer IDs", width=5.6)
        merges = kit.box("Merge rules, in order", sublabel="pairs of pieces that combine into one", width=5.6)
        loop = kit.box("A greedy loop", sublabel="on the CPU, in microseconds", width=5.6)
        left = VGroup(vocab, merges, loop).arrange(DOWN, buff=0.3)
        pill = cell("5000", 1.3, 0.62, self.color, size=th.SIZE_SMALL)
        pill_t = kit.label("token ID", size=th.SIZE_LABEL, color=th.INK_3).next_to(pill, UP, buff=0.15)
        table = kit.grid(9, 1, cell=(2.4, 0.34))
        table_t = kit.label("embedding table", size=th.SIZE_LABEL, color=th.INK_3).next_to(table, UP, buff=0.15)
        row_t = kit.label("row 5000", size=th.SIZE_LABEL, color=self.color, weight=MEDIUM)
        row_t.next_to(table.cell(5, 0), RIGHT, buff=0.2)
        fig = VGroup(VGroup(pill, pill_t), VGroup(table, table_t, row_t)).arrange(RIGHT, buff=1.1)
        cap = kit.label("The IDs and the table only make sense together", size=th.SIZE_SMALL, color=th.INK_2)
        right = VGroup(fig, cap).arrange(DOWN, buff=0.4)
        top = VGroup(left, right).arrange(RIGHT, buff=0.8)
        key = kit.callout("key", r"It costs microseconds. Getting it wrong costs you the model.",
                          title="A correctness dependency, not a performance one", width=12.4, size=th.SIZE_SMALL)
        col = VGroup(top, key).arrange(DOWN, buff=0.5)
        self.place(col)
        link = kit.arrow(pill, table.cell(5, 0), color=self.color)
        with self.narrate(
            "The tokenizer isn't a neural network. "
            "It has no trained weights, does no floating-point math, and runs on the CPU. "
            "It's two data structures and a greedy loop. "
            "A vocabulary maps byte strings to integer IDs. "
            "An ordered list of merge rules says which pairs of pieces combine into one. "
            "Both are fitted by counting, once, before training, and then frozen. "
            "Token ID 5000 means whatever row 5000 of the table learned it means. "
            "So the tokenizer is a correctness dependency, not a performance one."
        ) as t:
            t.wait_until("It's two data structures")
            self.play(FadeIn(loop, shift=UP * 0.1), run_time=0.6)
            t.wait_until("A vocabulary maps")
            self.play(FadeIn(vocab, shift=UP * 0.1), run_time=0.6)
            t.wait_until("An ordered list")
            self.play(FadeIn(merges, shift=UP * 0.1), run_time=0.6)
            t.wait_until("Both are fitted")
            self.play(*[Indicate(b.rect, color=th.INK_3, scale_factor=1.03) for b in (vocab, merges)], run_time=1.0)
            t.wait_until("Token ID 5000")
            self.play(FadeIn(table), FadeIn(table_t), FadeIn(pill), FadeIn(pill_t), run_time=0.8)
            t.wait_until("Token ID 5000")
            self.play(GrowArrow(link.line), table.fill(table.cell(5, 0), self.color), FadeIn(row_t), run_time=1.0)
            self.play(FadeIn(cap), run_time=0.5)
            t.wait_until("So the tokenizer")
            self.play(FadeIn(key, shift=UP * 0.1), run_time=0.8)

    # Byte-pair encoding ----------------------------------------------------------

    def bpe_round(self):
        self.slide("bpe-round", "Each merge round adds one vocabulary entry",
                   book_section="s-how-byte-pair-encoding-builds-a-vocabulary")
        f = kit.formula(r"|V|", r"=", r"256", r"+", r"M", r"+", r"S")
        f[2].set_color(BASE)
        f[4].set_color(MERGE)
        f[6].set_color(SPECIAL)
        merge = kit.formula(r"(a,\ b) \to ab", size=th.SIZE_MATH_SMALL)
        merge_t = kit.label("one merge", size=th.SIZE_LABEL, color=th.INK_3)
        mgroup = VGroup(merge, merge_t).arrange(DOWN, buff=0.15)
        top = VGroup(f, mgroup).arrange(RIGHT, buff=2.0)
        steps = VGroup(
            kit.box("Count every adjacent pair", sublabel="over the whole corpus", width=3.6),
            kit.box("Add the most frequent", sublabel="as one new ID", width=3.6, color=MERGE),
            kit.box("Rewrite the corpus", sublabel="with the new piece", width=3.6),
        ).arrange(RIGHT, buff=0.7)
        back = CurvedArrow(steps[2].get_bottom() + DOWN * 0.1, steps[0].get_bottom() + DOWN * 0.1, angle=-PI / 5,
                           color=th.INK_4, stroke_width=3, tip_length=0.18)
        back_t = kit.label("repeat, M times", size=th.SIZE_LABEL, color=th.INK_3)
        back_t.next_to(back, DOWN, buff=0.08)
        cycle = VGroup(steps, back, back_t)
        h = 0.6
        segs = VGroup(
            Rectangle(width=0.8, height=h, fill_color=BASE, fill_opacity=0.8, stroke_width=0),
            Rectangle(width=9.6, height=h, fill_color=MERGE, fill_opacity=0.8, stroke_width=0),
            Rectangle(width=0.8, height=h, fill_color=SPECIAL, fill_opacity=0.8, stroke_width=0),
        ).arrange(RIGHT, buff=0.04)
        seg_t = VGroup(
            kit.label("256 bytes", size=th.SIZE_LABEL, color=BASE, weight=MEDIUM),
            kit.label("roughly 248,000 merges", size=th.SIZE_LABEL, color=MERGE, weight=MEDIUM),
            kit.label("a few dozen specials", size=th.SIZE_LABEL, color=SPECIAL, weight=MEDIUM),
        )
        seg_t[0].next_to(segs[0], DOWN, buff=0.15).align_to(segs[0], LEFT)
        seg_t[1].next_to(segs[1], DOWN, buff=0.15)
        seg_t[2].next_to(segs[2], DOWN, buff=0.15).align_to(segs[2], RIGHT)
        total = kit.label("248,320 entries (widths not to scale)", size=th.SIZE_LABEL, color=th.INK_3)
        total.next_to(segs, UP, buff=0.15).align_to(segs, LEFT)
        bar = VGroup(segs, seg_t, total)
        col = VGroup(top, cycle, bar).arrange(DOWN, buff=0.55)
        self.place(col)
        with self.narrate(
            "This model uses byte-level BPE, so the alphabet is the 256 possible byte values. "
            "A merge is an ordered pair of existing entries, and the single new entry they produce. "
            "Count every adjacent pair in the corpus. "
            "Add the most frequent pair as a new entry. "
            "Then rewrite the corpus with the new piece, and go again. "
            "Each round adds exactly one entry. "
            "So the vocabulary is the 256 bytes, plus one entry per merge, plus a few dozen special tokens. "
            "For this model, that's roughly 248,000 learned merges."
        ) as t:
            self.play(FadeIn(f[:2]), run_time=0.5)
            t.wait_until("This model uses")
            self.play(FadeIn(f[2]), FadeIn(segs[0]), FadeIn(seg_t[0]), run_time=0.8)
            t.wait_until("A merge is")
            self.play(FadeIn(mgroup, shift=UP * 0.1), run_time=0.7)
            t.wait_until("Count every adjacent")
            self.play(FadeIn(steps[0], shift=UP * 0.1), run_time=0.6)
            t.wait_until("Add the most frequent")
            self.play(FadeIn(steps[1], shift=UP * 0.1), run_time=0.6)
            t.wait_until("Then rewrite")
            self.play(FadeIn(steps[2], shift=UP * 0.1), run_time=0.6)
            self.play(Create(back), FadeIn(back_t), run_time=0.8)
            t.wait_until("Each round adds")
            self.play(FadeIn(f[3:5]), run_time=0.6)
            t.wait_until("So the vocabulary is")
            self.play(FadeIn(f[5:]), FadeIn(segs[2]), FadeIn(seg_t[2]), FadeIn(total), run_time=0.8)
            t.wait_until("For this model")
            self.play(GrowFromEdge(segs[1], LEFT), FadeIn(seg_t[1]), run_time=1.2)

    def bpe_example(self):
        self.slide("bpe-example", "Four merges turn lowest into two tokens",
                   book_section="s-how-byte-pair-encoding-builds-a-vocabulary")
        states = [
            [list("low"), list("lower"), list("lowest")],
            [["lo", "w"], ["lo", "w", "e", "r"], ["lo", "w", "e", "s", "t"]],
            [["low"], ["low", "e", "r"], ["low", "e", "s", "t"]],
            [["low"], ["low", "e", "r"], ["low", "es", "t"]],
            [["low"], ["low", "e", "r"], ["low", "est"]],
        ]

        def corpus(k):
            ws = VGroup(*[word_state(p, self.color) for p in states[k]]).arrange(RIGHT, buff=0.75)
            return ws

        c0 = corpus(0)
        names = ["(l, o) → lo", "(lo, w) → low", "(e, s) → es", "(es, t) → est"]
        rules = VGroup()
        for i, nm in enumerate(names):
            rk = kit.label(str(i + 1), size=th.SIZE_SMALL, color=th.INK_4, font=th.MONO)
            rt = kit.mono(nm, size=th.SIZE_SMALL, color=th.INK)
            rules.add(VGroup(rk, rt).arrange(RIGHT, buff=0.3))
        rules.arrange(DOWN, buff=0.28, aligned_edge=LEFT)
        rules_h = kit.label("Merges learned, in order", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        rules_g = VGroup(rules_h, rules).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        res = VGroup(
            kit.label("lowest: 6 bytes → 2 tokens", size=th.SIZE_BODY, color=self.color, weight=SEMIBOLD),
            kit.label("Frequency counting, not learning", size=th.SIZE_SMALL, color=th.INK_3),
        ).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        lower = VGroup(rules_g, res).arrange(RIGHT, buff=1.4, aligned_edge=UP)
        c_cap = kit.label("A corpus of mostly low, lower, and lowest, split into bytes", size=th.SIZE_LABEL,
                          color=th.INK_3)
        top = VGroup(c_cap, c0).arrange(DOWN, buff=0.3)
        col = VGroup(top, lower).arrange(DOWN, buff=0.8)
        self.place(col)
        scale = c0[0][0].rect.height / PH
        anchor = c0.get_left()

        def placed(k):
            c = corpus(k)
            c.scale(scale)
            c.move_to(c0)
            c.shift([anchor[0] - c.get_left()[0], 0, 0])
            return c

        current = c0
        with self.narrate(
            "Take a corpus that's mostly the words low, lower, and lowest, and split every word into bytes. "
            "The pair l, o is the most frequent, so it becomes one new piece. "
            "Next, that piece followed by w is the most frequent, and the whole word low becomes one entry. "
            "Then e joins s, and that piece joins t. "
            "The word lowest now costs 2 tokens instead of 6 bytes. "
            "Nothing here is learned: it's frequency counting."
        ) as t:
            self.play(FadeIn(c_cap), LaggedStart(*[FadeIn(p, scale=0.8) for w in c0 for p in w], lag_ratio=0.05),
                      run_time=1.2)
            self.play(FadeIn(rules_h), run_time=0.4)
            for k, cue in ((1, "The pair l, o"), (2, "Next, that piece"), (3, "Then e joins"), (4, "Then e joins")):
                if k != 4:
                    t.wait_until(cue)
                nxt = placed(k)
                anims = []
                for wi in range(3):
                    anims += merge_anims(current[wi], states[k][wi], nxt[wi])
                self.play(*anims, FadeIn(rules[k - 1], shift=RIGHT * 0.1), run_time=0.9)
                for wi in range(3):
                    nxt[wi].lens = [len(p) for p in states[k][wi]]
                current = nxt
            t.wait_until("The word lowest now")
            self.play(FadeIn(res[0], shift=UP * 0.1), Indicate(current[2], color=self.color, scale_factor=1.06),
                      run_time=0.9)
            t.wait_until("Nothing here")
            self.play(FadeIn(res[1]), run_time=0.6)

    def encode_order(self):
        self.slide("encode-order", "Encoding replays the merges in training order",
                   book_section="s-how-encoding-applies-the-merges")
        names = ["(l, o) → lo", "(lo, w) → low", "(e, s) → es", "(es, t) → est"]
        rules = VGroup()
        for i, nm in enumerate(names):
            rk = kit.label(str(i + 1), size=th.SIZE_SMALL, color=th.INK_4, font=th.MONO)
            rt = kit.mono(nm, size=th.SIZE_SMALL, color=th.INK)
            rules.add(VGroup(rk, rt).arrange(RIGHT, buff=0.3))
        rules.arrange(DOWN, buff=0.32, aligned_edge=LEFT)
        rules_h = kit.label("Merge rules, by rank", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        left = VGroup(rules_h, rules).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        states = [list("slowest"), ["s", "lo", "w", "e", "s", "t"], ["s", "low", "e", "s", "t"],
                  ["s", "low", "es", "t"], ["s", "low", "est"]]
        w0 = word_state(states[0], self.color)
        w_cap = kit.label("slowest: never seen in training", size=th.SIZE_LABEL, color=th.INK_3)
        right = VGroup(w_cap, w0).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        top = VGroup(left, right).arrange(RIGHT, buff=1.4).scale(1.3)
        warn = kit.callout("warning", r"Apply the merges in any other order, and the same string gets different "
                                      r"IDs. The model sees nonsense, and nothing raises an error.",
                           title="Order matters", width=12.0, size=th.SIZE_SMALL)
        tip = kit.label("Debugging: encode the prompt, decode it, and compare it with the original.",
                        size=th.SIZE_SMALL, color=th.INK_2)
        col = VGroup(top, warn, tip).arrange(DOWN, buff=0.55)
        self.place(col)
        scale = w0[0].rect.height / PH
        anchor = w0.get_left()

        def placed(k):
            w = word_state(states[k], self.color)
            w.scale(scale)
            w.move_to(w0)
            w.shift([anchor[0] - w.get_left()[0], 0, 0])
            return w

        ptr = SurroundingRectangle(rules[0], color=self.color, buff=0.1, stroke_width=3)
        with self.narrate(
            "Encoding uses them on a string the tokenizer has never seen, like slowest. "
            "Split it into bytes, then apply the merge rules in the order they were learned, until no rule applies. "
            "Rule 1 joins l and o. "
            "Rule 2 joins that piece and w. "
            "The last two rules build the ending. "
            "Order matters, because low can only form after its first half exists. "
            "A tokenizer that applied the merges in any other order would produce different IDs for the same "
            "string. "
            "If generated text looks subtly wrong, check that the prompt round-trips."
        ) as t:
            self.play(FadeIn(left), run_time=0.6)
            t.wait_until("Encoding uses them")
            self.play(FadeIn(w_cap), run_time=0.4)
            t.wait_until("Split it into bytes")
            self.play(LaggedStart(*[FadeIn(p, scale=0.8) for p in w0], lag_ratio=0.1), run_time=0.9)
            current = w0
            for k, cue in ((1, "Rule 1 joins"), (2, "Rule 2 joins"), (3, "The last two"), (4, None)):
                if cue:
                    t.wait_until(cue)
                nxt = placed(k)
                move = Create(ptr) if k == 1 else Transform(
                    ptr, SurroundingRectangle(rules[k - 1], color=self.color, buff=0.1, stroke_width=3))
                self.play(move, run_time=0.5)
                self.play(*merge_anims(current, states[k], nxt), run_time=0.8)
                nxt.lens = [len(p) for p in states[k]]
                current = nxt
            t.wait_until("Order matters")
            self.play(FadeOut(ptr), Indicate(current, color=self.color, scale_factor=1.05), run_time=0.8)
            t.wait_until("A tokenizer that applied")
            self.play(FadeIn(warn, shift=UP * 0.1), run_time=0.8)
            t.wait_until("If generated text")
            self.play(FadeIn(tip), run_time=0.6)

    def pre_tokenize(self):
        self.slide("pre-tokenize", "Merges never cross a chunk boundary",
                   book_section="s-how-encoding-applies-the-merges")
        chunks = ["the", "·quick", "·brown", "·fox"]
        words = VGroup(*[kit.mono(c, size=th.SIZE_HEADING, color=th.INK) for c in chunks])
        words.arrange(RIGHT, buff=0.02, aligned_edge=DOWN)
        spread = VGroup(*[kit.mono(c, size=th.SIZE_HEADING, color=th.INK) for c in chunks])
        spread.arrange(RIGHT, buff=0.75, aligned_edge=DOWN)
        boxes = VGroup(*[SurroundingRectangle(s, buff=0.16, color=self.color, stroke_width=2.5,
                                              fill_color=th.tint(self.color, 0.08), fill_opacity=1, corner_radius=0.08)
                         for s in spread])
        for b in boxes:
            b.set_z_index(-1)
        walls = VGroup()
        for a, b in zip(boxes[:-1], boxes[1:]):
            x = (a.get_right()[0] + b.get_left()[0]) / 2
            walls.add(DashedLine([x, a.get_bottom()[1] - 0.15, 0], [x, a.get_top()[1] + 0.15, 0], color=th.INK_3,
                                 stroke_width=th.STROKE_AXIS, dash_length=0.08))
        wall_t = kit.label("no merge crosses a dashed line", size=th.SIZE_LABEL, color=th.INK_3)
        wall_t.next_to(VGroup(boxes, walls), DOWN, buff=0.3)
        key = kit.label("· marks a space", size=th.SIZE_LABEL, color=th.INK_4)
        key.next_to(VGroup(boxes, walls), UP, buff=0.3).align_to(boxes, RIGHT)
        fig = VGroup(spread, boxes, walls, wall_t, key)
        words.move_to(spread)

        def pair(text, note, color):
            c = cell(text, 1.8, 0.7, color, size=th.SIZE_BODY)
            n = kit.label(note, size=th.SIZE_SMALL, color=th.INK_2)
            return VGroup(c, n).arrange(RIGHT, buff=0.35)

        p1 = pair("·cat", "one ID", self.color)
        p2 = pair("cat", "a different ID", th.INFO)
        pairs = VGroup(p1, p2).arrange(RIGHT, buff=1.2)
        rules = kit.label("Chunks: words with a leading space, runs of digits, runs of punctuation",
                          size=th.SIZE_SMALL, color=th.INK_2)
        col = VGroup(rules, fig, pairs).arrange(DOWN, buff=0.7)
        self.place(col)
        words.scale(spread[1].height / words[1].height).move_to(spread)
        with self.narrate(
            "Before BPE runs, a regular expression splits the text into chunks. "
            "Typically, that's words with a leading space, runs of digits, and runs of punctuation. "
            "Merges never cross a chunk boundary. "
            "That stops the tokenizer from learning a single token for a whole phrase, like the quick brown fox. "
            "It's also why a leading space is usually part of the token. "
            "The word cat with a space in front, and cat without one, are different IDs."
        ) as t:
            self.play(FadeIn(words), run_time=0.6)
            t.wait_until("a regular expression splits")
            self.play(FadeIn(rules), run_time=0.5)
            self.play(*[Transform(w, s) for w, s in zip(words, spread)], run_time=0.9)
            self.play(LaggedStart(*[FadeIn(b) for b in boxes], lag_ratio=0.2), FadeIn(key), run_time=0.9)
            t.wait_until("Merges never cross")
            self.play(LaggedStart(*[Create(w) for w in walls], lag_ratio=0.2), FadeIn(wall_t), run_time=0.9)
            t.wait_until("It's also why")
            self.play(FadeIn(p1, shift=UP * 0.1), run_time=0.6)
            t.wait_until("The word cat")
            self.play(FadeIn(p2, shift=UP * 0.1), run_time=0.6)

    # Why nothing is unrepresentable -----------------------------------------------

    def byte_fallback(self):
        self.slide("byte-fallback", "Every byte has an entry, so nothing is unknown",
                   book_section="s-why-nothing-is-unrepresentable")
        g = kit.grid(16, 16, cell=0.23)
        g_t = kit.label("the 256 base entries, one per byte value", size=th.SIZE_LABEL, color=th.INK_3)
        grid_col = VGroup(g_t, g).arrange(DOWN, buff=0.2)
        inputs = [("valid UTF-8", ["E0", "A4", "95"]), ("invalid UTF-8", ["C3", "28"]), ("a null byte", ["00"]),
                  ("JPEG data", ["FF", "D8", "7A"])]
        ins = VGroup()
        bytes_of = {}
        for name, bs in inputs:
            nm = kit.label(name, size=th.SIZE_LABEL, color=th.INK_2)
            cs = VGroup(*[cell(b, 0.62, 0.48, th.GOOD, size=th.SIZE_LABEL) for b in bs]).arrange(RIGHT, buff=0.06)
            item = VGroup(nm, cs).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
            bytes_of[id(item)] = bs
            ins.add(item)
        ins.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        wl_head = kit.label("Word-level vocabulary", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        known = VGroup(*[cell(w, 1.1, 0.5, None, size=th.SIZE_LABEL) for w in ("the", "cat", "sat", "dog")])
        known.arrange_in_grid(2, 2, buff=0.08)
        unseen = cell("zyxt", 1.3, 0.5, None, size=th.SIZE_LABEL)
        unk = cell("<unk>", 1.3, 0.5, th.BAD, size=th.SIZE_LABEL)
        flow = VGroup(unseen, unk).arrange(RIGHT, buff=0.7)
        lost = kit.label("information destroyed", size=th.SIZE_LABEL, color=th.BAD)
        wl = VGroup(wl_head, known, flow, lost).arrange(DOWN, buff=0.35)
        wl_bg = RoundedRectangle(width=wl.width + 0.6, height=wl.height + 0.6, corner_radius=0.12,
                                 fill_color=th.SUNK, fill_opacity=1, stroke_width=0).move_to(wl)
        right = VGroup(wl_bg, wl)
        layout = VGroup(ins, grid_col, right).arrange(RIGHT, buff=0.7)
        self.place(layout)
        u_arrow = kit.arrow(unseen, unk, color=th.BAD)
        with self.narrate(
            "Every one of the 256 byte values has a vocabulary entry. "
            "So the encoder can always fall back to emitting raw bytes. "
            "That holds for any input at all: valid UTF-8, invalid UTF-8, a null byte, or the middle of a JPEG "
            "pasted into a chat box. "
            "There's no unknown token, because there's nothing unknown. "
            "Word-level vocabularies must decide what to do with a word they never saw. "
            "That means an unknown token, which destroys information before the model sees it."
        ) as t:
            self.play(FadeIn(g_t), FadeIn(g), run_time=0.7)
            t.wait_until("So the encoder")
            self.play(g.fill(g.block(0, 16, 0, 16), BASE, tint=0.18), run_time=0.8)
            t.wait_until("That holds for any")
            for item in ins:
                self.play(FadeIn(item, shift=RIGHT * 0.1), run_time=0.35)
                copies = []
                targets = []
                for b, c in zip(bytes_of[id(item)], item[1]):
                    v = int(b, 16)
                    tgt = g.cell(v // 16, v % 16)
                    dot = c.rect.copy().set_fill(th.tint(th.GOOD, 0.6), opacity=1)
                    copies.append(dot)
                    targets.append(tgt)
                self.play(*[d.animate.move_to(tg).match_width(tg.square).match_height(tg.square)
                            for d, tg in zip(copies, targets)], run_time=0.6)
                self.play(*[FadeOut(d) for d in copies], *[g.fill(tg, th.GOOD, tint=0.7) for tg in targets],
                          run_time=0.3)
            t.wait_until("There's no unknown")
            self.play(Indicate(g, color=BASE, scale_factor=1.03), run_time=0.8)
            t.wait_until("Word-level vocabularies")
            self.play(FadeIn(wl_bg), FadeIn(wl_head), FadeIn(known), run_time=0.7)
            self.play(FadeIn(unseen), run_time=0.5)
            t.wait_until("That means an unknown")
            self.play(GrowArrow(u_arrow.line), FadeIn(unk), run_time=0.7)
            self.play(FadeIn(lost), run_time=0.5)

    def token_cost(self):
        self.slide("token-cost", "Unfamiliar text costs more tokens", book_section="s-why-nothing-is-unrepresentable")
        glyph = kit.label("क", size=th.SIZE_DISPLAY, color=th.INK, font=DEVA)
        bytes_ = VGroup(*[cell(b, 0.9, 0.7, BASE, size=th.SIZE_BODY) for b in ("E0", "A4", "95")]).arrange(
            RIGHT, buff=0.1)
        b_t = kit.label("3 separate byte tokens, if no merge covers it", size=th.SIZE_SMALL, color=th.INK_2)
        right = VGroup(bytes_, b_t).arrange(DOWN, buff=0.25)
        top = VGroup(glyph, right).arrange(RIGHT, buff=1.4)
        tab = kit.table(
            [
                ["English", "about 4 characters per token"],
                ["Code", "fewer: indentation and punctuation split"],
                ["Non-Latin scripts", "often 2 to 3 times more tokens for the same content"],
            ],
            header=["Text", "Average cost"],
            size=th.SIZE_SMALL, col_gap=0.9, row_gap=0.3,
        )
        tag = kit.pill("more tokens: more latency, more money", self.color, size=th.SIZE_SMALL)
        col = VGroup(top, tab, tag).arrange(DOWN, buff=0.6)
        self.place(col)
        link = kit.arrow(glyph, bytes_)
        with self.narrate(
            "The cost moves into the representation instead. "
            "A character the tokenizer has never seen in context becomes three or four separate byte tokens. "
            "The model can still read them, just less efficiently. "
            "English averages about 4 characters per token. "
            "Code averages fewer, because indentation and punctuation split. "
            "Non-Latin scripts often cost 2 to 3 times more tokens for the same content. "
            "For a user, that's a real cost difference: more tokens per message, more latency, and more money."
        ) as t:
            self.play(FadeIn(glyph), run_time=0.6)
            t.wait_until("A character the")
            self.play(GrowArrow(link.line), LaggedStart(*[FadeIn(b, shift=RIGHT * 0.1) for b in bytes_],
                                                         lag_ratio=0.3), run_time=1.0)
            t.wait_until("The model can still")
            self.play(FadeIn(b_t), run_time=0.5)
            self.play(FadeIn(tab.rules), FadeIn(row(tab, 0)), run_time=0.6)
            for i, cue in enumerate(["English averages", "Code averages", "Non-Latin scripts"]):
                t.wait_until(cue)
                self.play(FadeIn(row(tab, i + 1), shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("For a user")
            self.play(FadeIn(tag, scale=0.9), run_time=0.6)

    # Special tokens and the chat template --------------------------------------

    def chat_template(self):
        self.slide("chat-template", "The prompt ends where the reply begins",
                   book_section="s-special-tokens-and-the-chat-template")
        panel = kit.code(TEMPLATE, "text", caption="the rendered prompt")
        a_t = kit.label("parsed as special", size=th.SIZE_SMALL, color=th.INK_2)
        a_c = cell("<|im_end|>", 2.6, 0.55, self.color, size=th.SIZE_SMALL)
        a_n = kit.label("one ID, never made by merging", size=th.SIZE_SMALL, color=self.color, weight=MEDIUM)
        b_t = kit.label("parsing turned off", size=th.SIZE_SMALL, color=th.INK_2)
        b_c = VGroup(*[cell(ch, 0.26, 0.55, th.INK_4, tint=0.25, size=th.SIZE_SMALL) for ch in "<|im_end|>"]).arrange(
            RIGHT, buff=0)
        b_n = kit.label("a run of ordinary tokens", size=th.SIZE_SMALL, color=th.BAD, weight=MEDIUM)
        rows_ = VGroup(VGroup(a_t, a_c, a_n), VGroup(b_t, b_c, b_n))
        for r_ in rows_:
            r_[1].move_to([3.2 + r_[1].width / 2, 0, 0])
            r_[0].move_to([r_[0].width / 2, 0, 0])
            r_[2].move_to([6.3 + r_[2].width / 2, 0, 0])
        rows_[1].shift(DOWN * 0.8)
        col = VGroup(panel, rows_).arrange(DOWN, buff=0.55)
        self.place(col)
        with self.narrate(
            "Special tokens are different. "
            "They're added to the vocabulary after training, and they're matched literally, before BPE runs. "
            "Merging ordinary text can never produce one. "
            "Encode a prompt with special-token parsing turned off, and the marker becomes ordinary tokens. "
            "The model has never seen that pattern, and loses the conversation's structure. "
            "That structure comes from the chat template, which turns a list of messages into a single string. "
            "The last line opens the assistant's turn and never closes it. "
            "The prompt ends exactly where the model continues."
        ) as t:
            t.wait_until("They're added")
            self.play(FadeIn(a_t), FadeIn(a_c, shift=RIGHT * 0.1), run_time=0.7)
            t.wait_until("Merging ordinary text")
            self.play(FadeIn(a_n), run_time=0.5)
            t.wait_until("Encode a prompt")
            self.play(FadeIn(b_t), LaggedStart(*[FadeIn(c) for c in b_c], lag_ratio=0.1), run_time=1.0)
            t.wait_until("The model has never")
            self.play(FadeIn(b_n), run_time=0.5)
            t.wait_until("That structure comes")
            self.play(FadeIn(panel.frame), *panel.reveal(1, 2), run_time=0.8)
            self.play(*panel.reveal(3, 4), run_time=0.7)
            self.play(panel.highlight(2, 4, color=th.tint(self.color, 0.1)), run_time=0.6)
            t.wait_until("The last line opens")
            self.play(*panel.reveal(5), run_time=0.6)
            self.play(panel.highlight(5), run_time=0.6)

    def stop_tokens(self):
        self.slide("stop-tokens", "Stop on either end-of-sequence ID",
                   book_section="s-special-tokens-and-the-chat-template")
        cfg = VGroup(kit.mono("eos_token_id: [248046, 248044]", size=th.SIZE_SMALL, color=th.INK),
                     kit.mono("pad_token_id: 248044", size=th.SIZE_SMALL, color=th.INK_3)).arrange(
            DOWN, buff=0.18, aligned_edge=LEFT)
        cfg_t = kit.label("from generation_config.json", size=th.SIZE_LABEL, color=th.INK_4)
        cfg_g = VGroup(cfg_t, cfg).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        words = ["A", "KV", "cache", "."]
        cw, ch = 1.35, 0.65

        def lane(name):
            nm = kit.label(name, size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
            cs = VGroup(*[cell(w, cw, ch, None, size=th.SIZE_SMALL, font=th.SANS) for w in words])
            cs.add(cell("248044", 1.3, ch, self.color, size=th.SIZE_SMALL))
            cs.arrange(RIGHT, buff=0.06)
            return nm, cs

        n1, l1 = lane("Stop set {248046, 248044}")
        n2, l2 = lane("Checks only 248046")
        done = kit.label("✓ done", size=th.SIZE_SMALL, color=th.GOOD, weight=SEMIBOLD)
        junk = VGroup(*[cell("??", 0.7, ch, th.BAD, tint=0.3, size=th.SIZE_SMALL) for _ in range(2)]).arrange(
            RIGHT, buff=0.06)
        wall = kit.label("… until max_tokens", size=th.SIZE_SMALL, color=th.BAD, weight=MEDIUM)
        g1 = VGroup(n1, l1).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        g2 = VGroup(n2, l2).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        lanes = VGroup(g1, g2).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        done.next_to(l1, RIGHT, buff=0.3)
        junk.next_to(l2, RIGHT, buff=0.06)
        wall.next_to(junk, RIGHT, buff=0.25)
        fig = VGroup(lanes, done, junk, wall)
        warn = kit.callout("warning", r"It degrades quality and looks like a bad model. Print the exact string "
                                      r"you tokenized before you look anywhere else.",
                           title="A wrong template raises no error", width=12.0, size=th.SIZE_SMALL)
        col = VGroup(cfg_g, fig, warn).arrange(DOWN, buff=0.5)
        cfg_g.align_to(fig, LEFT)
        self.place(col)
        with self.narrate(
            "The end-of-sequence setting is a list of two IDs. "
            "When the model emits either one, the reply is finished. "
            "Check only the first, and the engine misses the second. "
            "The generation runs to its token limit, with garbage after a good answer. "
            "The pad ID is 248044, and it matters only when you batch sequences of different lengths. "
            "A wrong template raises no error either, so when quality drops, print the exact string you tokenized."
        ) as t:
            self.play(FadeIn(cfg_t), FadeIn(cfg[0]), run_time=0.6)
            t.wait_until("When the model emits")
            self.play(FadeIn(n1), LaggedStart(*[FadeIn(c) for c in l1], lag_ratio=0.2), run_time=1.2)
            self.play(FadeIn(done, shift=RIGHT * 0.1), run_time=0.5)
            t.wait_until("Check only the first")
            self.play(FadeIn(n2), LaggedStart(*[FadeIn(c) for c in l2], lag_ratio=0.2), run_time=1.2)
            t.wait_until("The generation runs")
            self.play(LaggedStart(*[FadeIn(j, shift=RIGHT * 0.1) for j in junk], lag_ratio=0.3), run_time=1.0)
            self.play(FadeIn(wall), run_time=0.5)
            t.wait_until("The pad ID")
            self.play(FadeIn(cfg[1]), run_time=0.5)
            t.wait_until("A wrong template")
            self.play(FadeIn(warn, shift=UP * 0.1), run_time=0.8)

    # What the vocabulary costs ------------------------------------------------------

    def vocab_cost(self):
        self.slide("vocab-cost", "The vocabulary costs 9.5% of the weights",
                   book_section="s-what-the-vocabulary-costs")

        def matrix(title):
            r = Rectangle(width=0.85, height=2.9, fill_color=th.tint(th.INFO, 0.3), fill_opacity=1,
                          stroke_color=th.INFO, stroke_width=th.STROKE_BOX)
            top = kit.math_label(r"d = 5120", size=th.SIZE_LABEL, color=th.INK_3).next_to(r, UP, buff=0.12)
            side = kit.math_label(r"V = 248{,}320", size=th.SIZE_LABEL, color=th.INK_3)
            side.rotate(PI / 2).next_to(r, LEFT, buff=0.12)
            name = kit.label(title, size=th.SIZE_LABEL, color=th.INK_2, weight=MEDIUM).next_to(r, DOWN, buff=0.15)
            return VGroup(r, top, side, name)

        emb, out = matrix("Embedding"), matrix("Output projection")
        mats = VGroup(emb, out).arrange(RIGHT, buff=0.9, aligned_edge=DOWN)
        d = kit.Derivation(
            [
                r"248{,}320 \times 5120 &= 1{,}271{,}398{,}400 \approx 1.27\text{B}",
                r"\times\ 2\ \text{bytes} &\approx 2.54\ \text{GB}",
                r"\times\ 2\ \text{tables} &\approx 5.09\ \text{GB}",
                r"5.09 / 53.8 &\approx 9.5\%",
            ],
            size=th.SIZE_MATH_SMALL,
        )
        body = VGroup(mats, d).arrange(RIGHT, buff=0.9)
        w, h = 11.5, 0.55
        frame = Rectangle(width=w, height=h, stroke_color=th.INK_3, stroke_width=th.STROKE_BOX, fill_color=th.RAISE,
                          fill_opacity=1)
        seg = Rectangle(width=w * 5.09 / 53.8, height=h, stroke_width=0, fill_color=th.INFO, fill_opacity=0.85)
        seg.align_to(frame, LEFT)
        seg_t = kit.label("5.09 GB of 53.8 GB of weights", size=th.SIZE_LABEL, color=th.INK, weight=MEDIUM)
        seg_t.next_to(seg, RIGHT, buff=0.25)
        bar = VGroup(frame, seg, seg_t)
        note = kit.label("Arithmetic from the config, not measurements", size=th.SIZE_LABEL, color=th.INK_3)
        col = VGroup(body, bar, note).arrange(DOWN, buff=0.5)
        self.place(col)
        with self.narrate(
            "Each vocabulary entry owns a row of the embedding table, and each row is 5120 wide. "
            "Multiply the rows by the width, and the embedding holds about 1.27 billion parameters. "
            "At 2 bytes each in bfloat16, that's 2.54 GB. "
            "This model doesn't tie its weights, so the output projection is a second matrix of exactly the same "
            "size. "
            "Together, they take 5.09 GB. "
            "Divide by the 53.8 GB of weights, and the vocabulary costs 9.5 percent of the model."
        ) as t:
            t.wait_until("Each vocabulary entry")
            self.play(GrowFromEdge(emb[0], DOWN), FadeIn(emb[1:]), run_time=0.9)
            t.wait_until("Multiply the rows")
            self.play(d.show(0), run_time=0.9)
            t.wait_until("At 2 bytes")
            self.play(d.show(1), run_time=0.8)
            t.wait_until("This model doesn't")
            self.play(TransformFromCopy(emb, out), run_time=1.0)
            t.wait_until("Together, they take")
            self.play(d.show(2), run_time=0.8)
            t.wait_until("Divide by the 53.8")
            self.play(d.show(3), FadeIn(frame), run_time=0.8)
            self.play(GrowFromEdge(seg, LEFT), FadeIn(seg_t), FadeIn(note), run_time=0.8)

    def worth_it(self):
        self.slide("worth-it", "A big vocabulary buys shorter sequences",
                   book_section="s-what-the-vocabulary-costs")

        def tokens(pieces, color, label_text):
            cs = VGroup(*[cell(p, 0.32 * len(p) + 0.3, 0.6, color, size=th.SIZE_SMALL) for p in pieces]).arrange(
                RIGHT, buff=0.06)
            nm = kit.label(label_text, size=th.SIZE_SMALL, color=th.INK_2)
            return VGroup(nm, cs)

        big = tokens(["The", "·tokenizer", "·runs", "·on", "·the", "·CPU"], self.color,
                     "Large vocabulary: 6 tokens")
        small = tokens(["The", "·to", "ken", "izer", "·r", "uns", "·on", "·the", "·C", "PU"], th.INFO,
                       "Smaller vocabulary: 10 tokens")
        for g in (big, small):
            g.arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        rows_ = VGroup(big, small).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        ill = kit.label("an illustrative split", size=th.SIZE_LABEL, color=th.INK_4)
        ill.next_to(rows_, UP, buff=0.2).align_to(rows_, RIGHT)
        b1 = kit.box("Prefill", sublabel="work grows with prompt tokens", width=5.4)
        b2 = kit.box("Decode", sublabel="one full read of the weights per token", width=5.4)
        boxes = VGroup(b1, b2).arrange(RIGHT, buff=0.6)
        trade = kit.prose(r"Halve the vocabulary: get 2.54 GB of HBM back, but every sequence gets longer.",
                          size=th.SIZE_SMALL, width=None, color=th.INK_2)
        verdict = kit.pill("The designers took the memory", self.color, size=th.SIZE_SMALL)
        col = VGroup(VGroup(rows_, ill), boxes, trade, verdict).arrange(DOWN, buff=0.5)
        self.place(col)
        with self.narrate(
            "A bigger vocabulary means fewer tokens per document. "
            "Prefill does work in proportion to the number of prompt tokens. "
            "Decode does one full read of the weights for every generated token. "
            "Halving the vocabulary would give back 2.54 GB of HBM. "
            "But it would lengthen every sequence, and that costs prefill FLOPs and decode steps in proportion. "
            "The designers took the memory."
        ) as t:
            self.play(FadeIn(ill), FadeIn(small), run_time=0.8)
            t.wait_until("A bigger vocabulary")
            self.play(FadeIn(big, shift=UP * 0.1), run_time=0.8)
            t.wait_until("Prefill does work")
            self.play(FadeIn(b1, shift=UP * 0.1), run_time=0.6)
            t.wait_until("Decode does one")
            self.play(FadeIn(b2, shift=UP * 0.1), run_time=0.6)
            t.wait_until("Halving the vocabulary")
            self.play(FadeIn(trade), run_time=0.7)
            t.wait_until("The designers took")
            self.play(FadeIn(verdict, scale=0.9), run_time=0.6)

    # The embedding layer -------------------------------------------------------------

    def one_hot(self):
        self.slide("one-hot", "A one-hot product reads every row to keep one",
                   book_section="s-the-embedding-layer")
        n, m, v = 10, 7, 6
        hot = kit.grid(n, 1, cell=(0.42, 0.34), values=[["1" if r == v else "0"] for r in range(n)])
        hot.paint(hot.cell(v, 0), ONEHOT, tint=0.45)
        table = kit.grid(n, m, cell=(0.42, 0.34))
        out = kit.grid(1, m, cell=(0.42, 0.34))
        hot_t = kit.math_label(r"\mathbf{1}_v", size=th.SIZE_SMALL, color=ONEHOT)
        tab_t = kit.math_label(r"E:\ V \times d", size=th.SIZE_SMALL, color=th.INK_2)
        out_t = kit.math_label(r"e = \text{row } v", size=th.SIZE_SMALL, color=th.INK_2)
        pair = VGroup(hot, table).arrange(RIGHT, buff=0.45)
        hot_t.next_to(hot, UP, buff=0.15)
        tab_t.next_to(table, UP, buff=0.15)
        out.next_to(table, DOWN, buff=0.45)
        out_t.next_to(out, LEFT, buff=0.25)
        fig = VGroup(pair, hot_t, tab_t, out, out_t)
        f = kit.formula(r"e = E^{\top}", r"\mathbf{1}_v")
        f[1].set_color(ONEHOT)
        tab = kit.table(
            [["One-hot product", "2.54 GFLOP", "2.54 GB"], ["Gather", "zero", "10 KiB"]],
            header=["Method", "Arithmetic", "Bytes read"],
            size=th.SIZE_SMALL, align=["l", "r", "r"], col_gap=0.7,
        )
        ratio = kit.prose(r"Bytes read: $2.54 \times 10^9 / 10{,}240 = 248{,}320$, which is $V$ exactly.",
                          size=th.SIZE_SMALL, width=6.4, color=self.color)
        side = VGroup(f, tab, ratio).arrange(DOWN, buff=0.55, aligned_edge=LEFT)
        layout = VGroup(fig, side).arrange(RIGHT, buff=1.0)
        self.place(layout)
        with self.narrate(
            "Textbooks write the embedding as a multiply by a one-hot vector: all zeros, except a single 1. "
            "Multiply the transposed table by it, and every row is zeroed out except row v. "
            "That's correct, and a catastrophic way to compute it. "
            "The product does 2.54 GFLOP of arithmetic, and it reads the whole 2.54 GB table. "
            "Every framework uses a gather instead. "
            "It reads row v, and nothing else: 5120 values at 2 bytes, which is 10 KiB. "
            "The ratio of bytes read is 248,320, which is exactly the vocabulary size."
        ) as t:
            self.play(FadeIn(table), FadeIn(tab_t), run_time=0.6)
            t.wait_until("Textbooks write")
            self.play(FadeIn(hot), FadeIn(hot_t), FadeIn(f), run_time=0.8)
            t.wait_until("Multiply the transposed")
            self.play(LaggedStart(*[table.fill(table.row(r), ONEHOT, tint=0.3) for r in range(n)], lag_ratio=0.15),
                      run_time=1.6)
            self.play(FadeIn(out), FadeIn(out_t), out.fill(out.row(0), ONEHOT, tint=0.45), run_time=0.7)
            t.wait_until("The product does")
            self.play(FadeIn(tab.rules), FadeIn(row(tab, 0)), FadeIn(row(tab, 1)), run_time=0.8)
            t.wait_until("Every framework uses")
            self.play(table.fill(VGroup(*[table.row(r) for r in range(n) if r != v]), th.RAISE, tint=None),
                      run_time=0.8)
            self.play(Indicate(table.row(v), color=ONEHOT, scale_factor=1.04), run_time=0.8)
            t.wait_until("It reads row v")
            self.play(FadeIn(row(tab, 2)), run_time=0.6)
            t.wait_until("The ratio of bytes")
            self.play(FadeIn(ratio), run_time=0.7)

    def gather(self):
        self.slide("gather", "In the engine, the embedding is one gather", book_section="s-the-embedding-layer")
        panel = kit.code(EMBED, "python")
        tab = kit.table(
            [
                [r"\texttt{input\_ids}", r"\texttt{(batch, seq)}", "int64"],
                [r"\texttt{embed\_tokens.weight}", r"\texttt{(248320, 5120)}", "bf16"],
                [r"\texttt{x}", r"\texttt{(batch, seq, 5120)}", "bf16"],
            ],
            header=["Tensor", "Shape", "Dtype"],
            size=th.SIZE_SMALL, col_gap=0.6,
        )
        left = VGroup(panel, tab).arrange(DOWN, buff=0.6, aligned_edge=LEFT)
        rows_n = 22
        tall = kit.grid(rows_n, 1, cell=(2.2, 0.17))
        tall_t = kit.label("the 2.54 GB table", size=th.SIZE_LABEL, color=th.INK_3).next_to(tall, UP, buff=0.15)
        picks = [3, 16, 9, 19]
        ids = VGroup(*[cell(f"id {i + 1}", 1.1, 0.5, self.color, size=th.SIZE_LABEL) for i in range(4)]).arrange(
            DOWN, buff=0.35)
        right = VGroup(ids, VGroup(tall, tall_t)).arrange(RIGHT, buff=1.2)
        top = VGroup(left, right).arrange(RIGHT, buff=0.9)
        note = kit.prose(r"A 2000-token prompt reads about 20 MB in scattered 10 KiB rows: no locality, so each row "
                         r"misses L2 and pays a full trip to HBM.", size=th.SIZE_SMALL, width=12.0, color=th.INK_2)
        col = VGroup(top, note).arrange(DOWN, buff=0.55)
        self.place(col)
        links = VGroup(*[Arrow(ids[i].get_right(), tall.cell(p, 0).get_left(), buff=0.08, color=self.color,
                               stroke_width=3, tip_length=0.16, max_tip_length_to_length_ratio=0.2)
                         for i, p in enumerate(picks)])
        with self.narrate(
            "In the engine, the embedding is one line. "
            "It takes integer token IDs, shaped batch by sequence, and indexes the table. "
            "The table is 248,320 rows by 5120, in bf16. "
            "Each ID gathers its own row, wherever it sits in the table. "
            "For a 2000-token prompt, the gather moves about 20 MB, scattered across the 2.54 GB table. "
            "The reads have no locality, so each one misses L2 and pays a full trip to HBM. "
            "Even so, it's the cheapest layer in the model by a wide margin."
        ) as t:
            self.play(FadeIn(panel.frame), *panel.reveal(1), run_time=0.7)
            t.wait_until("It takes integer")
            self.play(FadeIn(tab.rules), FadeIn(row(tab, 0)), FadeIn(row(tab, 1)), run_time=0.7)
            t.wait_until("The table is")
            self.play(FadeIn(row(tab, 2)), FadeIn(tall), FadeIn(tall_t), run_time=0.7)
            t.wait_until("Each ID gathers")
            self.play(FadeIn(row(tab, 3)), FadeIn(ids), run_time=0.5)
            for i, p in enumerate(picks):
                self.play(GrowArrow(links[i]), tall.fill(tall.cell(p, 0), self.color, tint=0.6), run_time=0.45)
            t.wait_until("For a 2000-token")
            self.play(FadeIn(note), run_time=0.8)

    def residual_start(self):
        self.slide("residual-start", "Layer 0 gets no position and no fixed scale",
                   book_section="s-the-embedding-layer")
        words = ["the", "cat", "saw", "the", "dog"]
        r = kit.rng(5)
        pats = {w: r.uniform(0.12, 0.85, 7) for w in sorted(set(words))}
        cols_ = VGroup()
        for w in words:
            c = cell(w, 1.0, 0.55, None, size=th.SIZE_SMALL, font=th.SANS)
            v = VGroup(*[Rectangle(width=0.5, height=0.3, stroke_color=th.RULE_STRONG, stroke_width=1,
                                   fill_color=th.tint(th.INFO, float(s)), fill_opacity=1) for s in pats[w]]).arrange(
                DOWN, buff=0)
            cols_.add(VGroup(c, v).arrange(DOWN, buff=0.25))
        cols_.arrange(RIGHT, buff=0.35)
        pos = VGroup(*[kit.label(f"position {i}", size=th.SIZE_LABEL, color=th.INK_4).next_to(cols_[i], UP, buff=0.12)
                       for i in (0, 3)])
        left = VGroup(cols_, pos)
        same = kit.label("identical vectors", size=th.SIZE_SMALL, color=self.color, weight=MEDIUM)
        same.next_to(cols_, DOWN, buff=0.3)
        left.add(same)

        def vec(scale_):
            pat = pats["cat"]
            return VGroup(*[Rectangle(width=0.5, height=0.3, stroke_color=th.RULE_STRONG, stroke_width=1,
                                      fill_color=th.tint(th.VIOLET, min(0.95, float(s) * scale_)), fill_opacity=1)
                            for s in pat]).arrange(DOWN, buff=0)

        big, small = vec(1.6), vec(0.6)
        normed = vec(1.0)
        in_g = VGroup(big, small).arrange(RIGHT, buff=0.25)
        norm_b = kit.box("RMSNorm", sublabel="divide by its own size", color=th.VIOLET, width=2.9)
        out_g = VGroup(normed, normed.copy()).arrange(RIGHT, buff=0.25)
        right = VGroup(in_g, norm_b, out_g).arrange(RIGHT, buff=0.6)
        r_cap = kit.label("same direction, different size", size=th.SIZE_LABEL, color=th.INK_3)
        r_cap.next_to(in_g, DOWN, buff=0.3)
        o_cap = kit.label("the size drops out", size=th.SIZE_LABEL, color=th.INK_3)
        o_cap.next_to(out_g, DOWN, buff=0.3)
        right.add(r_cap, o_cap)
        top = VGroup(left, right).arrange(RIGHT, buff=0.9)
        note = kit.prose(r"A stray $\sqrt{5120} \approx 71.6$ factor in someone else's code compensates for a small "
                         r"initialization. A pre-norm model like this one doesn't need it.", size=th.SIZE_SMALL,
                         width=12.0, color=th.INK_2)
        col = VGroup(top, note).arrange(DOWN, buff=0.6)
        self.place(col)
        a1 = kit.arrow(in_g, norm_b)
        a2 = kit.arrow(norm_b, out_g)
        same_box = VGroup(SurroundingRectangle(cols_[0], color=self.color, buff=0.08, stroke_width=3),
                          SurroundingRectangle(cols_[3], color=self.color, buff=0.08, stroke_width=3))
        with self.narrate(
            "The embedding's output is the starting value of the residual stream. "
            "First, it carries no position information. "
            "Two occurrences of the same token produce identical vectors, wherever they sit in the prompt. "
            "Rotation and attention tell them apart later. "
            "Second, its absolute scale barely matters. "
            "The first thing layer 0 does is RMSNorm, which divides each row by its own root-mean-square size. "
            "Some architectures scale the embedding by the square root of d, about 71.6, to offset a small initialization. "
            "A pre-norm model like this one doesn't need that."
        ) as t:
            self.play(LaggedStart(*[FadeIn(c[0]) for c in cols_], lag_ratio=0.15), run_time=0.9)
            self.play(LaggedStart(*[FadeIn(c[1], shift=DOWN * 0.1) for c in cols_], lag_ratio=0.15), run_time=1.0)
            t.wait_until("Two occurrences")
            self.play(Create(same_box), FadeIn(pos), FadeIn(same), run_time=0.9)
            t.wait_until("Second, its absolute")
            self.play(FadeIn(in_g), FadeIn(r_cap), run_time=0.7)
            t.wait_until("The first thing layer 0")
            self.play(GrowArrow(a1.line), FadeIn(norm_b), run_time=0.7)
            self.play(GrowArrow(a2.line), TransformFromCopy(in_g, out_g), FadeIn(o_cap), run_time=1.0)
            t.wait_until("Some architectures")
            self.play(FadeIn(note), run_time=0.8)

    # The output projection ------------------------------------------------------------

    def lm_head(self):
        self.slide("lm-head", "The output projection reads all 2.54 GB per token",
                   book_section="s-the-output-projection")
        panel = kit.code(LM_HEAD, "python")
        shape = kit.prose(r"Weight: $248{,}320 \times 5120$, the embedding's shape.",
                          size=th.SIZE_SMALL, width=None, color=th.INK_2)
        d = kit.Derivation(
            [
                r"\text{FLOP} &= 2 \times 5120 \times 248{,}320 \approx 2.54 \times 10^9",
                r"\text{bytes} &\approx 2.54 \times 10^9",
                r"\text{intensity} &= 1\ \text{FLOP per byte}",
            ],
            size=th.SIZE_MATH_SMALL,
        )
        left = VGroup(panel, shape).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        bc = kit.bar_compare(
            [("Output projection", 1, self.color, "1"), ("A100 needs", 161, th.INK_4, "about 161")],
            width=3.4, bar_height=0.5, gap=0.35, size=th.SIZE_SMALL,
        )
        bc_t = kit.label("FLOPs per byte read", size=th.SIZE_LABEL, color=th.INK_3)
        share = kit.prose(r"A decode step reads about 51.3 GB of weights. This matrix is 2.54 GB of it, just "
                          r"under 5\%.", size=th.SIZE_SMALL, width=None, color=th.INK_2)
        right = VGroup(d, bc_t, bc).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        layout = VGroup(VGroup(left, right).arrange(RIGHT, buff=0.9, aligned_edge=UP), share).arrange(DOWN, buff=0.5)
        self.place(layout)
        with self.narrate(
            "After the final norm, the output projection maps the hidden state to one score per vocabulary entry. "
            "Each score is a logit, and the sampler in chapter 11 picks the next token from them. "
            "During decode it runs for a single position, so it does about 2.54 GFLOP. "
            "It also reads 2.54 GB of weights. "
            "That's an arithmetic intensity of 1 FLOP per byte read. "
            "To stay compute bound, the A100 needs about 161 FLOPs per byte. "
            "So the output projection spends nearly all its time waiting for memory. "
            "A decode step reads about 51.3 GB of weights, and this matrix is just under 5 percent of that."
        ) as t:
            self.play(FadeIn(panel.frame), *panel.reveal(1), run_time=0.7)
            t.wait_until("After the final norm")
            self.play(*panel.reveal(2, 3, 4), panel.highlight(4), run_time=0.9)
            t.wait_until("Each score is")
            self.play(FadeIn(shape), run_time=0.6)
            t.wait_until("During decode it runs")
            self.play(d.show(0), run_time=0.8)
            t.wait_until("It also reads")
            self.play(d.show(1), run_time=0.7)
            t.wait_until("That's an arithmetic")
            self.play(d.show(2), FadeIn(bc_t), FadeIn(bc.labels[0]), GrowFromEdge(bc.bars[0], LEFT),
                      FadeIn(bc.values[0]), run_time=0.8)
            t.wait_until("To stay compute")
            self.play(FadeIn(bc.labels[1]), GrowFromEdge(bc.bars[1], LEFT), run_time=1.2)
            self.play(FadeIn(bc.values[1]), run_time=0.4)
            t.wait_until("A decode step reads")
            self.play(FadeIn(share), run_time=0.7)

    def slice_first(self):
        self.slide("slice-first", "Slice to the last position before projecting",
                   book_section="s-the-output-projection")
        n, m = 9, 14
        g = kit.grid(n, m, cell=(0.36, 0.3), gap=0.03)
        rows_t = VGroup(kit.label("position 1", size=th.SIZE_LABEL, color=th.INK_3),
                        kit.label("position 2000", size=th.SIZE_LABEL, color=th.INK_3))
        rows_t[0].next_to(g.cell(0, 0), LEFT, buff=0.2)
        rows_t[1].next_to(g.cell(n - 1, 0), LEFT, buff=0.2)
        top_t = kit.label("248,320 logits per position", size=th.SIZE_LABEL, color=th.INK_3)
        top_t.next_to(g, UP, buff=0.15)
        fig = VGroup(g, rows_t, top_t)
        tab = kit.table(
            [["Output tensor", "993 MB", "497 KB"], ["Arithmetic", "5.09 TFLOP", "2.54 GFLOP"]],
            header=["", "Without the slice", "With the slice"],
            size=th.SIZE_SMALL, align=["l", "r", "r"], col_gap=0.7,
        )
        tag = kit.label("Training needs every position. Inference needs one.", size=th.SIZE_SMALL, color=self.color,
                        weight=SEMIBOLD)
        side = VGroup(tab, tag).arrange(DOWN, buff=0.6, aligned_edge=LEFT)
        layout = VGroup(fig, side).arrange(RIGHT, buff=0.8)
        self.place(layout)
        keep = g.outline(n - 1, n, 0, m, color=self.color)
        with self.narrate(
            "The slice to the last token is what keeps prefill honest. "
            "Without it, a 2000-token prompt computes logits for every position, then keeps only the last row. "
            "That output alone is 993 MB, and computing it takes about 5.09 TFLOP. "
            "Slice first, and the output is 497 KB, for 2.54 GFLOP. "
            "Training needs every position, because every position has a target. "
            "Inference needs one. "
            "Skipping it is silent: prefill just runs slower than your roofline estimate."
        ) as t:
            self.play(FadeIn(g), FadeIn(top_t), FadeIn(rows_t), run_time=0.7)
            t.wait_until("Without it, a 2000")
            self.play(LaggedStart(*[g.fill(g.row(r), th.BAD, tint=0.4) for r in range(n)], lag_ratio=0.12),
                      run_time=1.4)
            self.play(g.fill(g.block(0, n - 1, 0, m), th.INK_4, tint=0.15), g.fill(g.row(n - 1), self.color, tint=0.5),
                      run_time=0.8)
            t.wait_until("That output alone")
            self.play(FadeIn(tab.rules), FadeIn(row(tab, 0)), *[FadeIn(row(tab, i)[:2]) for i in (1, 2)],
                      run_time=0.8)
            t.wait_until("Slice first")
            self.play(Create(keep), *[FadeIn(row(tab, i)[2]) for i in (1, 2)], run_time=0.8)
            t.wait_until("Training needs")
            self.play(FadeIn(tag), run_time=0.6)

    def weight_tying(self):
        self.slide("weight-tying", "This model keeps two separate matrices", book_section="s-the-output-projection")

        def mat(name, color):
            r = Rectangle(width=0.9, height=2.0, fill_color=th.tint(color, 0.3), fill_opacity=1, stroke_color=color,
                          stroke_width=th.STROKE_BOX)
            t_ = kit.math_label(name, size=th.SIZE_SMALL, color=th.INK).move_to(r)
            return VGroup(r, t_)

        tied = mat("E", th.INFO)
        tied_t = kit.label("Tied", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        tied_n = VGroup(kit.prose(r"lookup: $E$", size=th.SIZE_LABEL, color=th.INK_3, width=None),
                        kit.prose(r"projection: $E^{\top}$", size=th.SIZE_LABEL, color=th.INK_3, width=None)).arrange(
            DOWN, buff=0.12)
        tied_g = VGroup(tied_t, tied, tied_n).arrange(DOWN, buff=0.25)
        un1, un2 = mat("E", th.INFO), mat("W", self.color)
        pair = VGroup(un1, un2).arrange(RIGHT, buff=0.35)
        un_t = kit.label("Untied: this model", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        un_n = VGroup(kit.label("gathered a row at a time", size=th.SIZE_LABEL, color=th.INFO),
                      kit.label("streamed in full every step", size=th.SIZE_LABEL, color=self.color)).arrange(
            DOWN, buff=0.12)
        un_g = VGroup(un_t, pair, un_n).arrange(DOWN, buff=0.25)
        left = VGroup(tied_g, un_g).arrange(RIGHT, buff=0.9, aligned_edge=UP)
        panel = kit.code(TYING, "python", caption="ModelConfig.param_counts")
        flag = kit.mono("tie_word_embeddings: false", size=th.SIZE_SMALL, color=th.INK)
        cost = kit.prose(r"Untying costs 1.27B parameters: 4.7\% of 26.9B.", size=th.SIZE_SMALL, width=5.8,
                         color=th.INK_2)
        right = VGroup(flag, cost).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        top = VGroup(VGroup(left, right).arrange(RIGHT, buff=0.9), panel).arrange(DOWN, buff=0.45)
        warn = kit.callout("warning", r"The output weights never load from disk. You get a crash, or a model that "
                                      r"runs at full speed and emits fluent, confident nonsense.",
                           title="Assuming tying on an untied checkpoint", width=12.4, size=th.SIZE_SMALL)
        col = VGroup(top, warn).arrange(DOWN, buff=0.5)
        self.place(col)
        with self.narrate(
            "Weight tying uses the same matrix for both jobs: E for the lookup, and E transposed for the projection. "
            "This model doesn't tie, which is the common choice at this scale. "
            "Untying costs 1.27 billion parameters, which is 4.7 percent of the model. "
            "One is gathered a row at a time, and the other is streamed in full every step. "
            "The engine has to check the flag, not assume it. "
            "Assume tying on an untied checkpoint, and the output weights never load. "
            "You get a crash, or a model that runs at full speed and emits confident nonsense."
        ) as t:
            self.play(FadeIn(tied_g), run_time=0.8)
            t.wait_until("This model doesn't")
            self.play(FadeIn(un_t), FadeIn(pair), FadeIn(flag), run_time=0.8)
            t.wait_until("Untying costs")
            self.play(FadeIn(cost), run_time=0.6)
            t.wait_until("One is gathered")
            self.play(FadeIn(un_n[0]), Indicate(un1, color=th.INFO), run_time=0.7)
            self.play(FadeIn(un_n[1]), Indicate(un2, color=self.color), run_time=0.7)
            t.wait_until("The engine has to")
            self.play(FadeIn(panel.frame), *panel.reveal(1, 2), run_time=0.7)
            self.play(panel.highlight(2), run_time=0.5)
            t.wait_until("Assume tying")
            self.play(FadeIn(warn, shift=UP * 0.1), run_time=0.8)

    # Detokenization ------------------------------------------------------------------

    def detokenize(self):
        self.slide("detokenize", "A character can straddle two tokens",
                   book_section="s-detokenization-is-stateful")
        bw, bh = 0.85, 0.55
        chars = [("क", 3, LEAD), ("·", 1, BASE), ("o", 1, BASE), ("k", 1, BASE)]
        bytes_ = ["E0", "A4", "95", "20", "6F", "6B"]
        bcells = VGroup(*[cell(b, bw, bh, LEAD if i < 3 else BASE, size=th.SIZE_SMALL) for i, b in enumerate(bytes_)])
        bcells.arrange(RIGHT, buff=0.06)
        glyphs = VGroup()
        i = 0
        for ch, n, _c in chars:
            span = VGroup(*bcells[i:i + n])
            gl = kit.label(ch, size=th.SIZE_BODY, color=th.INK,
                           font=DEVA if ch == "क" else th.SANS)
            gl.next_to(span, UP, buff=0.18)
            glyphs.add(gl)
            i += n
        toks = VGroup()
        for k in range(3):
            span = VGroup(bcells[2 * k], bcells[2 * k + 1])
            br = Brace(span, DOWN, buff=0.08, color=th.INK_3)
            tl = kit.label(f"token {k + 1}", size=th.SIZE_LABEL, color=th.INK_3).next_to(br, DOWN, buff=0.06)
            toks.add(VGroup(br, tl))
        top = VGroup(glyphs, bcells, toks)
        lane_names = VGroup(kit.label("Each token alone", size=th.SIZE_SMALL, color=th.BAD, weight=MEDIUM),
                            kit.label("All IDs so far", size=th.SIZE_SMALL, color=th.GOOD, weight=MEDIUM))
        alone = ["�", "� ", "ok"]
        state = ["(nothing yet)", "क ", "ok"]
        out_a = VGroup(*[cell(s.replace(" ", "·"), 1.75, 0.6, th.BAD, tint=0.15, size=th.SIZE_SMALL) for s in alone])
        out_s = VGroup()
        for s in state:
            if s.startswith("("):
                out_s.add(cell(s, 1.75, 0.6, th.GOOD, tint=0.12, size=th.SIZE_LABEL, font=th.SANS, ink=th.INK_3))
            else:
                c = cell(None, 1.75, 0.6, th.GOOD, tint=0.15)
                lab = VGroup(kit.label("क", size=th.SIZE_SMALL, color=th.INK, font=DEVA),
                             kit.label("·", size=th.SIZE_SMALL, color=th.INK, font=th.MONO)).arrange(
                    RIGHT, buff=0.04) if s.startswith("क") else kit.label(s, size=th.SIZE_SMALL, color=th.INK,
                                                                          font=th.MONO)
                lab.move_to(c.rect)
                c.add(lab)
                out_s.add(c)
        for k in range(3):
            x = VGroup(bcells[2 * k], bcells[2 * k + 1]).get_center()[0]
            out_a[k].move_to([x, 0, 0])
            out_s[k].move_to([x, -0.8, 0])
        lanes = VGroup(out_a, out_s)
        lanes.next_to(top, DOWN, buff=0.55)
        for k in range(3):
            x = VGroup(bcells[2 * k], bcells[2 * k + 1]).get_center()[0]
            out_a[k].set_x(x)
            out_s[k].set_x(x)
        lane_names[0].next_to(out_a, LEFT, buff=0.5)
        lane_names[1].next_to(out_s, LEFT, buff=0.5)
        lane_names[1].align_to(lane_names[0], RIGHT)
        fig = VGroup(top, lanes, lane_names)
        steps = kit.bullets(
            [
                r"Append the new ID, and decode \textit{all} the IDs so far.",
                r"Ignore an incomplete tail; emit only the new, complete suffix.",
                r"At the end, \texttt{flush} replaces what's left instead of hiding it.",
            ],
            size=th.SIZE_SMALL, width=12.0, buff=0.22,
        )
        col = VGroup(fig, steps).arrange(DOWN, buff=0.55)
        self.place(col)
        with self.narrate(
            "Streaming runs the tokenizer backwards, and one token isn't one printable unit. "
            "Take the Devanagari letter ka, which is three bytes, followed by a space and the word ok. "
            "With two-byte tokens, the first token carries two-thirds of the letter. "
            "Decode each token on its own, and you get replacement characters at the seams. "
            "Append the new ID, decode all the accumulated IDs, and emit only the suffix that's new and complete. "
            "The first push returns an empty string, and that's correct behavior, not a bug. "
            "The second token completes the letter, and the whole character appears at once. "
            "At the end, flush replaces an incomplete tail instead of hiding it."
        ) as t:
            self.play(FadeIn(bcells), FadeIn(glyphs), run_time=0.8)
            t.wait_until("With two-byte tokens")
            self.play(LaggedStart(*[FadeIn(tk) for tk in toks], lag_ratio=0.3), run_time=0.9)
            t.wait_until("Decode each token")
            self.play(FadeIn(lane_names[0]), LaggedStart(*[FadeIn(o, shift=DOWN * 0.1) for o in out_a],
                                                         lag_ratio=0.3), run_time=1.2)
            t.wait_until("Append the new ID")
            self.play(FadeIn(lane_names[1]), FadeIn(steps[0]), FadeIn(steps[1]), run_time=0.8)
            t.wait_until("The first push")
            self.play(FadeIn(out_s[0], shift=DOWN * 0.1), run_time=0.6)
            t.wait_until("The second token")
            self.play(FadeIn(out_s[1], shift=DOWN * 0.1), FadeIn(out_s[2], shift=DOWN * 0.1), run_time=0.8)
            t.wait_until("At the end, flush")
            self.play(FadeIn(steps[2]), run_time=0.6)
