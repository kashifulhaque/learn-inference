"""Chapter 15, paged attention.

Teaches content/chapters/15-paged-attention.md in its order. Every number
below was recomputed in Python (exact arithmetic, then rounded as the
chapter does).

Per-token KV and the pool (chapter 2's budget):
- One full-attention layer, one token: 2 x 4 x 256 x 2 B = 4096 B = 4 KiB.
  x 16 layers = 65,536 B = 64 KiB per token.
- One 16-token page: 16 x 64 KiB = 1 MiB.
- Pool: 80 x 2^30 - 53.8e9 - 6 x 2^30 = 25,656,894,976 B = 23.89 GiB
  ("23.9") = 24,468.3 MiB ("24,468"). Weights 53.8e9 B = 50.1 GiB.
- 24,468 pages x 16 tokens = 391,488 tokens.
- Recurrent state 154,927,104 B = 147.75 MiB ("147.8").
- One sequence at 262,144 tokens: 16 GiB KV + 0.144 GiB state = 16.14 GiB
  ("16.1 GiB"). 4k: 25,656,894,976 / (4096 x 65,536 + 154,927,104) = 60.6,
  so 60. 32k: 11.14, so 11.

What a contiguous cache wastes (lab mix 37, 5, 64, 200, 1200, 18, 450, 12,
3000, 90):
- Used: 5076 tokens x 64 KiB = 317.25 MiB ("317").
- Contiguous: 10 x 32,768 = 327,680 slots = 20 GiB; waste 1 - 5076/327,680
  = 0.98451 (98.45%). One reservation: 32,768 x 64 KiB = 2 GiB.
- Paged rounding up to 16: 48, 16, 64, 208, 1200, 32, 464, 16, 3008, 96;
  sum 5152 slots = 322 MiB; waste 1 - 5076/5152 = 0.01475 (1.48%).
- Contiguous request = 2 GiB + 147.75 MiB = 2.1443 GiB. Ten: 21.44 GiB
  ("21.4"); eleven: 23.59 ("23.6"); twelve: 25.73 ("25.7") > 23.89.
- Paged, per request: (322 MiB + 10 x 147.75 MiB) / 10 = 179.95 MiB
  ("about 180"); 24,468.3 / 179.95 = 135.97 ("about 135", as the chapter).
- Bar widths on the 11-unit pool bar: contiguous 2.1443 / 23.895 x 11 =
  0.987 per request; paged 179.95 / 24,468.3 x 11 = 0.0809 per request.

Mapping positions to slots, B = 16: 37 // 16 = 2, 37 % 16 = 5; table entry 9
gives 9 x 16 + 5 = 149. (Other table entries on screen are illustrative.)

Choosing the page size, 32k context, 64 KiB per token:
- Tail waste (B - 1)/2: B = 16: 7.5 tokens = 480 KiB; B = 64: 31.5 tokens =
  2016 KiB = 1.97 MiB; B = 256: 127.5 tokens = 7.97 MiB.
- Table entries 32,768 / B: 32,768, 2048, 512, 128 (also loads per decode).
- B = 256: 127.5 / 32,768 = 0.39% ("0.4%").
- 4 bytes of table per 1 MiB of KV: 2^20 / 4 = 262,144.

The free list: 2% of 24,468 pages = 489.4, "about 489". Decode with B = 16
allocates on 1 step in 16, so 15 of 16 steps allocate nothing.

Kernel walk: context length 40 = 2 full blocks + 8 tokens; cdiv(40, 16) = 3.

Prefix sharing (500-token system prompt, 64 requests):
- 500 x 64 KiB = 32,000 KiB = 31.25 MiB per copy.
- 63 x 31.25 MiB = 1968.75 MiB = 1.923 GiB ("1.92"); / 24,468 MiB = 8.05%
  ("about 8%").
- Prefill 2 x 26.9e9 x 500 = 2.69e13 = 26.9 TFLOP; / 312e12 = 86.2 ms ("86").
- 500 // 16 = 31 full blocks (496 tokens), 500 % 16 = 4 tokens left.

The 48 layers that don't page:
- 147.8 MiB / 64 KiB = 2364.8 ("2365", the chapter's figure; the exact
  147.75 MiB gives 2364).
- 24,468 / 147.8 = 165.5, so 165 sequences.
- 64 x 147.75 MiB = 9.234 GiB ("9.24"); / 23.895 GiB = 38.6% ("39%").
  Bar: 64 x 147.75 / 24,468.3 x 11 = 4.25 units.
- 100 tokens x 64 KiB = 6.25 MiB.

Preemption, 2000 tokens of context:
- Swap: 2000 x 64 KiB + 147.75 MiB = 125 + 147.75 = 272.75 MiB ("273").
  At 21 GB/s: 286.0e6 / 21e9 = 13.6 ms; the chapter says "about 13 ms".
- PCIe vs HBM: 1275 / 21 = 60.7 ("roughly 60 times").
- Recompute: 2 x 26.9e9 x 2000 = 107.6 TFLOP; / 312e12 = 344.9 ms ("345").

What goes wrong: (tokens // 16) + 1 over-allocates exactly on multiples of 16;
the chapter's "6% extra" is its own figure (1 / 16 = 6.25%).
"""

from manim import *

import kit
from kit import theme as th

# Colour roles, the same on every slide.
WEIGHTS = th.INFO
OVERHEAD = th.INK_4
STATE = th.HL_C  # \hlc in the chapter: the fixed recurrent-state term
KVC = th.HL_D  # \hld in the chapter: the per-token KV term
FREE = th.RAISE

ALLOC_SRC = """\
def blocks_needed(self, seq_id, extra_tokens):
    table = self.tables.get(seq_id)
    current = table.length if table else 0
    have = len(table.blocks) * self.block_size if table else 0
    short = current + extra_tokens - have
    return max(0, -(-short // self.block_size))

def allocate(self, seq_id, extra_tokens):
    table = self.tables.setdefault(seq_id, BlockTable(blocks=[]))
    need = self.blocks_needed(seq_id, extra_tokens)
    if need > len(self.free): raise OutOfBlocks(...)
    for _ in range(need):
        table.blocks.append(self.free.pop())
    table.length += extra_tokens
"""

RELEASE_SRC = """\
def release(self, seq_id):
    table = self.tables.pop(seq_id, None)
    if table:
        self.free.extend(table.blocks)
"""

KERNEL_SRC = """\
num_blocks = tl.cdiv(context_len, BLOCK_SIZE)
for b in range(0, MAX_BLOCKS):
    if b < num_blocks:
        physical = tl.load(block_table_ptr + seq * stride_tb
                           + b * stride_ts)
        token_pos = b * BLOCK_SIZE + offs_s
        valid = token_pos < context_len
        # k_ptrs and v_ptrs start at physical * stride_kb
        k = tl.load(k_ptrs, mask=valid[:, None], other=0.0)
        scores = tl.sum(k * q[None, :], axis=1)
        scores = tl.where(valid, scores, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        correction = tl.exp(m_i - m_new)
        # ... chapter 14's online softmax, unchanged
"""


def seg(x0, width, height, color, tint=0.45, stroke=None, y=0.0):
    """One segment of a horizontal bar, from x0 to x0 + width."""
    r = Rectangle(width=max(width, 0.01), height=height, fill_color=th.tint(color, tint), fill_opacity=1,
                  stroke_color=stroke or color, stroke_width=th.STROKE_HAIRLINE if stroke is None else 2)
    r.move_to([x0 + width / 2, y, 0])
    return r


def frame_bar(width, height):
    """An empty bar outline, left edge at x = 0, drawn above its contents."""
    back = Rectangle(width=width, height=height, stroke_width=0, fill_color=th.RAISE, fill_opacity=1)
    edge = Rectangle(width=width, height=height, stroke_color=th.INK_3, stroke_width=th.STROKE_BOX, fill_opacity=0)
    edge.set_z_index(2)
    g = VGroup(back, edge)
    g.move_to([width / 2, 0, 0])
    g.back, g.edge = back, edge
    return g


def strip(n, w, h, color=None, filled=0, gap=0.0, tint=0.45):
    """A block as a row of `n` token slots, the first `filled` tinted with `color`."""
    cells = VGroup()
    for i in range(n):
        fill = th.tint(color, tint) if (color is not None and i < filled) else th.RAISE
        cells.add(Rectangle(width=w, height=h, stroke_color=th.RULE_STRONG, stroke_width=th.STROKE_HAIRLINE,
                            fill_color=fill, fill_opacity=1))
    cells.arrange(RIGHT, buff=gap)
    return cells


def swatch(text, color, size=th.SIZE_LABEL, tint=0.45):
    sq = Square(0.26, stroke_color=color, stroke_width=th.STROKE_HAIRLINE, fill_color=th.tint(color, tint),
                fill_opacity=1)
    return VGroup(sq, kit.label(text, size=size, color=th.INK_2)).arrange(RIGHT, buff=0.16)


def raise_text(g):
    """Keeps a grid's cell text above its squares while the squares animate."""
    for row in g.cells:
        for c in row:
            if c.text is not None:
                c.text.set_z_index(3)


def id_cell(text, color=None, w=0.62, h=0.55, size=th.SIZE_SMALL, tint=0.35):
    rect = Rectangle(width=w, height=h, stroke_color=th.RULE_STRONG if color is None else color,
                     stroke_width=th.STROKE_BOX, fill_color=th.RAISE if color is None else th.tint(color, tint),
                     fill_opacity=1)
    t = kit.label(text, size=size, color=th.INK, font=th.MONO)
    t.move_to(rect)
    t.set_z_index(3)
    g = VGroup(rect, t)
    g.rect, g.text = rect, t
    return g


class Lecture(kit.Lecture):
    def construct(self):
        self.title_slide(
            narration=(
                "Paged attention. "
                "A contiguous KV cache wastes most of the memory you give it. "
                "In this lecture, you borrow the operating system's answer: fixed-size blocks, a table per "
                "sequence, and a decode kernel that follows the table."
            )
        )
        self.pool_budget()
        self.reserve_max()
        self.holes()
        self.chat_mix()
        self.eleven()
        self.analogy()
        self.block_pool()
        self.page_cost()
        self.slot_map()
        self.page_size()
        self.alloc_code()
        self.decode_growth()
        self.out_of_blocks()
        self.kernel_walk()
        self.kernel_code()
        self.prefix_share()
        self.prefix_savings()
        self.fixed_state()
        self.preemption()
        self.what_goes_wrong()
        self.closing_slide(
            [
                r"Contiguous wastes 98.45\%; 16-token pages, 1.48\%.",
                r"A page is 1 MiB; the pool holds 391,488 tokens.",
                r"The kernel adds one table load per block.",
                r"The state doesn't page; it caps the batch at 165.",
            ],
            narration=(
                "Here's what to keep. "
                "On the lab's chat mix, a contiguous cache wastes 98.45 percent of its reservation, and 16-token "
                "pages waste 1.48 percent. "
                "On this model, one page is 1 MiB, and the pool holds 391,488 tokens of KV. "
                "The kernel is chapter 14's online softmax, plus one table load per block. "
                "And the recurrent state doesn't page, so it caps the batch at 165."
            ),
            book_section="s-what-goes-wrong",
        )

    # ------------------------------------------------------------------
    # Before you start

    def pool_budget(self):
        self.slide("pool-budget", "Every sequence draws on one 24,468 MiB pool", book_section="s-before-you-start")
        W, H = 12.0, 0.95
        weights = 53.8e9 / 2**30
        unit = W / 80
        bar = frame_bar(W, H)
        s_w = seg(0, weights * unit, H, WEIGHTS)
        s_o = seg(weights * unit, 6 * unit, H, OVERHEAD)
        s_p = seg((weights + 6) * unit, (80 - weights - 6) * unit, H, self.color)
        t_w = kit.label("weights: 53.8 GB = 50.1 GiB", size=th.SIZE_SMALL, color=th.INK).move_to(s_w)
        t_p = kit.label("pool: 23.9 GiB", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM).move_to(s_p)
        t_o = kit.label("6 GiB overhead", size=th.SIZE_LABEL, color=th.INK_3)
        t_o.next_to(s_o, DOWN, buff=0.15)
        card = kit.label("A100: 80 GiB", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        card.next_to(bar, UP, buff=0.15).align_to(bar, LEFT)
        mib = kit.label("= 24,468 MiB for per-sequence memory", size=th.SIZE_SMALL, color=self.color, weight=MEDIUM)
        mib.next_to(bar, UP, buff=0.15).align_to(bar, RIGHT)
        top = VGroup(bar, s_w, s_o, s_p, t_w, t_p, t_o, card, mib)
        kv = kit.box("KV cache", sublabel="64 KiB per token, 16 full-attention layers", color=KVC, width=5.6)
        st = kit.box("Recurrent state", sublabel="147.8 MiB per sequence, 48 linear layers", color=STATE, width=5.6)
        boxes = VGroup(kv, st).arrange(RIGHT, buff=0.6)
        col = VGroup(top, boxes).arrange(DOWN, buff=1.3)
        self.place(col)
        a1 = kit.arrow(s_p.get_bottom() + LEFT * 0.6, kv, color=KVC)
        a2 = kit.arrow(s_p.get_bottom() + RIGHT * 0.3, st, color=STATE)
        with self.narrate(
            "Start with the budget from chapter 2. "
            "The card holds 80 GiB. "
            "The weights take 53.8 GB. "
            "Another 6 GiB goes to the CUDA context, a workspace, and headroom. "
            "That leaves 23.9 GiB, or 24,468 MiB, for per-sequence memory. "
            "The 16 full-attention layers store 64 KiB of keys and values per token. "
            "The other 48 layers keep a fixed recurrent state for each sequence."
        ) as t:
            t.wait_until("The card holds")
            self.play(FadeIn(bar), FadeIn(card), run_time=0.7)
            t.wait_until("The weights take")
            self.play(GrowFromEdge(s_w, LEFT), FadeIn(t_w), run_time=0.8)
            t.wait_until("Another 6 GiB")
            self.play(GrowFromEdge(s_o, LEFT), FadeIn(t_o), run_time=0.6)
            t.wait_until("That leaves")
            self.play(GrowFromEdge(s_p, LEFT), FadeIn(t_p), FadeIn(mib), run_time=0.8)
            t.wait_until("The 16 full-attention")
            self.play(GrowArrow(a1.line), FadeIn(kv, shift=UP * 0.1), run_time=0.7)
            t.wait_until("The other 48")
            self.play(GrowArrow(a2.line), FadeIn(st, shift=UP * 0.1), run_time=0.7)

    # ------------------------------------------------------------------
    # What a contiguous cache wastes

    def reserve_max(self):
        self.slide("reserve-max", "A contiguous cache reserves 2 GiB per request",
                   book_section="s-what-a-contiguous-cache-wastes")
        W, H = 8.4, 0.55
        lengths = [1200, 450, 3000]
        colors = [self.colors[1], self.colors[2], self.colors[3]]
        rows = VGroup()
        used = VGroup()
        names = VGroup()
        for i, (n, c) in enumerate(zip(lengths, colors)):
            back = Rectangle(width=W, height=H, fill_color=th.tint(th.INK_4, 0.18), fill_opacity=1,
                             stroke_color=th.RULE_STRONG, stroke_width=th.STROKE_HAIRLINE)
            back.move_to([W / 2, -i * (H + 0.3), 0])
            u = seg(0, max(n / 32768 * W, 0.08), H, c, tint=0.75, y=back.get_y())
            nm = kit.label(f"{n:,} tokens", size=th.SIZE_SMALL, color=th.INK_2)
            nm.next_to(back, LEFT, buff=0.3)
            rows.add(back)
            used.add(u)
            names.add(nm)
        brace = Brace(rows[0], UP, buff=0.12, color=th.INK_4)
        cap = kit.prose(r"32,768 slots $\times$ 64 KiB $=$ 2 GiB, reserved at admission", size=th.SIZE_SMALL,
                        width=None, color=th.INK_2)
        cap.next_to(brace, UP, buff=0.1)
        gray = kit.label("gray: reserved, never used", size=th.SIZE_LABEL, color=th.INK_3)
        gray.next_to(rows, DOWN, buff=0.2).align_to(rows, RIGHT)
        fig = VGroup(rows, used, names, brace, cap, gray)
        kinds = kit.bullets(
            [
                r"\textbf{Reserved but unused:} 2 GiB per sequence, whatever it turns out to be.",
                r"\textbf{Internal fragmentation:} the last, partly filled unit.",
                r"\textbf{External fragmentation:} free memory in pieces of the wrong size.",
            ],
            size=th.SIZE_SMALL, width=12.4, buff=0.2,
        )
        col = VGroup(fig, kinds).arrange(DOWN, buff=0.6)
        self.place(col)
        with self.narrate(
            "A contiguous cache wastes memory in three ways, and the largest is reservation. "
            "A request that might reach 32,768 tokens reserves room for all of them, before you know its length. "
            "At 64 KiB per token, that's 2 GiB per sequence. "
            "Here are three chat requests. "
            "The colored sliver is what each one uses, and the gray is reserved for nobody. "
            "The second waste is internal fragmentation: the last unit is only partly filled. "
            "The third is external fragmentation."
        ) as t:
            self.play(FadeIn(rows), FadeIn(names), run_time=0.8)
            t.wait_until("A request that might")
            self.play(FadeIn(brace), FadeIn(cap), run_time=0.7)
            t.wait_until("At 64 KiB")
            self.play(FadeIn(kinds[0], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("Here are three")
            self.play(LaggedStart(*[GrowFromEdge(u, LEFT) for u in used], lag_ratio=0.3), run_time=1.2)
            t.wait_until("The colored sliver")
            self.play(FadeIn(gray), Indicate(rows, color=th.INK_3, scale_factor=1.02), run_time=1.0)
            t.wait_until("The second waste")
            self.play(FadeIn(kinds[1], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("The third is")
            self.play(FadeIn(kinds[2], shift=RIGHT * 0.1), run_time=0.6)

    def holes(self):
        self.slide("holes", "6 GiB free, and no room for 2 GiB", book_section="s-what-a-contiguous-cache-wastes")
        # Twelve sequences in 18 cells of 1 GiB; the six one-cell ones finish.
        spans = [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
        cw, ch = 0.62, 0.8
        cells = VGroup(*[Rectangle(width=cw, height=ch, stroke_color=th.RULE_STRONG, stroke_width=th.STROKE_HAIRLINE,
                                   fill_color=FREE, fill_opacity=1) for _ in range(18)]).arrange(RIGHT, buff=0)
        palette = [self.colors[1], self.colors[2], self.colors[3]]
        seqs = VGroup()
        finishing = VGroup()
        x = 0
        for i, n in enumerate(spans):
            block = VGroup(*cells[x:x + n])
            r = Rectangle(width=block.width, height=ch, fill_color=th.tint(palette[i % 3], 0.45), fill_opacity=1,
                          stroke_color=palette[i % 3], stroke_width=th.STROKE_BOX)
            r.move_to(block)
            seqs.add(r)
            if n == 1:
                finishing.add(r)
            x += n
        pool_lab = kit.label("pool, each cell 1 GiB", size=th.SIZE_LABEL, color=th.INK_3)
        pool_lab.next_to(cells, DOWN, buff=0.15).align_to(cells, LEFT)
        free_lab = kit.label("6 GiB free, in six pieces", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
        free_lab.next_to(cells, DOWN, buff=0.15).align_to(cells, RIGHT)
        req = Rectangle(width=2 * cw, height=ch, fill_color=th.tint(self.color, 0.5), fill_opacity=1,
                        stroke_color=self.color, stroke_width=3)
        req_t = kit.label("2 GiB", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM).move_to(req)
        request = VGroup(req, req_t)
        req_lab = kit.label("new request: one contiguous run of 2 GiB", size=th.SIZE_SMALL, color=th.INK_2)
        verdict = kit.label("No hole is big enough: the request fails.", size=th.SIZE_BODY, color=th.BAD,
                            weight=MEDIUM)
        top = VGroup(req_lab, request).arrange(RIGHT, buff=0.4)
        fig = VGroup(cells, seqs, pool_lab, free_lab)
        col = VGroup(top, fig, verdict).arrange(DOWN, buff=0.9)
        top.align_to(fig, LEFT)
        self.place(col)
        hole_x = [finishing[i].get_x() for i in range(len(finishing))]
        with self.narrate(
            "The third waste comes from time. "
            "Sequences finish at different moments, and each one leaves a hole. "
            "Here, six holes add up to 6 GiB of free memory. "
            "A new request needs one contiguous run of 2 GiB. "
            "Every hole is smaller than that, so the request fails, while 6 GiB sits free."
        ) as t:
            self.play(FadeIn(cells), FadeIn(seqs), FadeIn(pool_lab), run_time=0.8)
            t.wait_until("Sequences finish")
            self.play(LaggedStart(*[r.animate.set_fill(FREE, opacity=0).set_stroke(opacity=0) for r in finishing],
                                  lag_ratio=0.25), run_time=1.6)
            t.wait_until("Here, six holes")
            self.play(FadeIn(free_lab), Indicate(VGroup(*[cells[i] for i in (2, 5, 8, 11, 14, 17)]),
                                                  color=self.color, scale_factor=1.05), run_time=1.0)
            t.wait_until("A new request")
            self.play(FadeIn(req_lab), FadeIn(request, shift=DOWN * 0.1), run_time=0.7)
            t.wait_until("Every hole")
            y = request.get_y()
            for hx in hole_x[:3]:
                self.play(request.animate.move_to([hx + cw / 2, cells.get_top()[1] + ch / 2 + 0.08, 0]),
                          run_time=0.45)
                self.play(Indicate(req, color=th.BAD, scale_factor=1.06), run_time=0.35)
            self.play(request.animate.move_to([top[0].get_right()[0] + 0.4 + request.width / 2, y, 0]),
                      FadeIn(verdict, shift=UP * 0.1), run_time=0.8)
            self.play(req.animate.set_stroke(th.BAD), run_time=0.4)

    def chat_mix(self):
        self.slide("chat-mix", "On a chat trace, 98.45% of the reservation is waste",
                   book_section="s-what-a-contiguous-cache-wastes")
        lengths = [37, 5, 64, 200, 1200, 18, 450, 12, 3000, 90]
        rounded = [-(-n // 16) * 16 for n in lengths]
        g = kit.grid(2, 10, cell=(0.95, 0.55), values=[[str(n) for n in lengths], [str(n) for n in rounded]],
                     gap=0.06, row_labels=["used", "paged"])
        raise_text(g)
        for c in range(10):
            g.paint(g.cell(0, c), self.colors[1], tint=0.25)
        tab = kit.table(
            [
                ["Contiguous, 32k max", "327,680", "20 GiB", r"98.45\%"],
                ["Paged, 16-token blocks", "5,152", "322 MiB", r"1.48\%"],
                ["Actually used", "5,076", "317 MiB", "—"],
            ],
            header=["Layout", "Tokens reserved", "Memory", "Waste"],
            align=["l", "r", "r", "r"],
            size=th.SIZE_SMALL,
        )
        col = VGroup(g, tab).arrange(DOWN, buff=0.7)
        self.place(col)
        changed = [c for c in range(10) if rounded[c] != lengths[c]]
        with self.narrate(
            "Now put numbers on it, with the length mix the lab measures. "
            "Ten chat requests, from 5 tokens to 3000, use 5076 tokens in all. "
            "A contiguous cache reserves 32,768 slots for each, or 20 GiB. "
            "That's 98.45 percent waste. "
            "Paged in 16-token blocks, each length rounds up to a multiple of 16. "
            "A 5-token request takes 16 slots. "
            "The ten need 5152 slots, or 322 MiB. "
            "That's 1.48 percent waste."
        ) as t:
            self.play(FadeIn(VGroup(*g.cells[0])), FadeIn(g.row_labels[0]), run_time=0.8)
            t.wait_until("Ten chat requests")
            self.play(FadeIn(tab.rules), FadeIn(VGroup(*tab.cells[0])), FadeIn(VGroup(*tab.cells[3])), run_time=0.7)
            t.wait_until("A contiguous cache")
            self.play(FadeIn(VGroup(*tab.cells[1][:3]), shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("That's 98.45")
            self.play(FadeIn(tab.cells[1][3], scale=1.2), run_time=0.5)
            t.wait_until("Paged in 16-token")
            self.play(FadeIn(VGroup(*g.cells[1])), FadeIn(g.row_labels[1]), run_time=0.7)
            self.play(g.fill(VGroup(*[g.cell(1, c) for c in range(10)]), self.color, tint=0.25), run_time=0.6)
            t.wait_until("A 5-token")
            self.play(Indicate(VGroup(*[g.cell(1, c) for c in changed]), color=self.color, scale_factor=1.06),
                      run_time=1.0)
            t.wait_until("The ten need")
            self.play(FadeIn(VGroup(*tab.cells[2][:3]), shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("That's 1.48")
            self.play(FadeIn(tab.cells[2][3], scale=1.2), run_time=0.5)

    def eleven(self):
        self.slide("eleven", "Contiguous fits 11 requests; paged fits about 135",
                   book_section="s-what-a-contiguous-cache-wastes")
        W, H = 11.0, 0.7
        pool = 25_656_894_976
        contig = 32768 * 65536 + 154_927_104
        paged = (5152 * 65536 + 10 * 154_927_104) / 10
        b1, b2 = frame_bar(W, H), frame_bar(W, H)
        w1, w2 = contig / pool * W, paged / pool * W
        segs1 = VGroup(*[seg(i * w1, w1, H, self.colors[1], tint=0.4 if i % 2 else 0.6) for i in range(11)])
        over = seg(11 * w1, w1, H, th.BAD, tint=0.25, stroke=th.BAD)
        over_t = kit.label("12", size=th.SIZE_LABEL, color=th.BAD).move_to(over)
        nums = VGroup(*[kit.label(str(i + 1), size=th.SIZE_LABEL, color=th.INK_2).move_to(s)
                        for i, s in enumerate(segs1)])
        segs2 = VGroup(*[seg(i * w2, w2, H, self.color, tint=0.35 if i % 2 else 0.6) for i in range(135)])
        for s in segs2:
            s.set_stroke(width=0)
        t1 = kit.label("Contiguous: 2 GiB + 147.8 MiB each", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        t2 = kit.label("Paged: about 180 MiB each, state included", size=th.SIZE_SMALL, color=th.INK_2,
                       weight=MEDIUM)
        r1 = kit.label("11 fit", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
        r2 = kit.label("about 135 fit", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
        g1 = VGroup(b1, segs1, nums, over, over_t)
        g2 = VGroup(b2, segs2)
        t1.next_to(b1, UP, buff=0.15).align_to(b1, LEFT)
        r1.next_to(b1, UP, buff=0.15).align_to(b1, RIGHT)
        t2.next_to(b2, UP, buff=0.15).align_to(b2, LEFT)
        r2.next_to(b2, UP, buff=0.15).align_to(b2, RIGHT)
        top = VGroup(t1, r1, g1)
        bot = VGroup(t2, r2, g2)
        totals = VGroup(
            kit.label("10 requests: 21.4 GiB", size=th.SIZE_SMALL, color=th.INK_2),
            kit.label("11: 23.6 GiB", size=th.SIZE_SMALL, color=th.INK_2),
            kit.label("12: 25.7 GiB, over the 23.9 GiB pool", size=th.SIZE_SMALL, color=th.BAD),
        ).arrange(RIGHT, buff=0.6)
        gap = kit.callout("key", r"A twelvefold gap: the difference between a serving engine and a demo.",
                          width=10.0, size=th.SIZE_SMALL)
        col = VGroup(top, totals, bot, gap).arrange(DOWN, buff=0.5)
        totals.align_to(top, LEFT)
        self.place(col)
        with self.narrate(
            "Each contiguous request costs 2 GiB of KV, plus 147.8 MiB of recurrent state. "
            "Ten requests take 21.4 GiB. "
            "Eleven take 23.6 GiB, and they still fit. "
            "A twelfth brings the total to 25.7 GiB, which doesn't. "
            "Under paging, the same requests need about 180 MiB each, state included. "
            "The card holds about 135 of them. "
            "That twelvefold gap is the difference between a serving engine and a demo."
        ) as t:
            self.play(FadeIn(b1), FadeIn(t1), run_time=0.6)
            t.wait_until("Ten requests")
            self.play(LaggedStart(*[GrowFromEdge(s, LEFT) for s in segs1[:10]], lag_ratio=0.1),
                      FadeIn(nums[:10]), FadeIn(totals[0]), run_time=1.3)
            t.wait_until("Eleven take")
            self.play(GrowFromEdge(segs1[10], LEFT), FadeIn(nums[10]), FadeIn(totals[1]), run_time=0.6)
            t.wait_until("A twelfth")
            self.play(GrowFromEdge(over, LEFT), FadeIn(over_t), FadeIn(totals[2]), run_time=0.7)
            self.play(FadeIn(r1), run_time=0.5)
            t.wait_until("Under paging")
            self.play(FadeIn(b2), FadeIn(t2), run_time=0.6)
            self.play(LaggedStart(*[FadeIn(s) for s in segs2], lag_ratio=0.02), run_time=2.0)
            t.wait_until("The card holds")
            self.play(FadeIn(r2), run_time=0.5)
            t.wait_until("That twelvefold")
            self.play(FadeIn(gap, shift=UP * 0.1), run_time=0.7)

    # ------------------------------------------------------------------
    # The virtual memory analogy

    def analogy(self):
        self.slide("analogy", "The operating system solved this with pages",
                   book_section="s-the-virtual-memory-analogy")
        tab = kit.table(
            [
                ["Process", "Sequence"],
                ["Virtual page", r"Logical block, $\lfloor p / B \rfloor$"],
                ["Physical page frame", "Physical block in the pool"],
                ["Page table", "Block table"],
                ["Page size, often 4 KiB", "Block size, 16 tokens"],
                [r"Shared pages after \texttt{fork}", "Prefix sharing"],
                ["Swapping to disk", "Swapping to host memory"],
            ],
            header=["Operating system", "Paged KV cache"],
            size=th.SIZE_SMALL,
            row_gap=0.14,
        )
        c1 = kit.prose(r"\textbf{No hardware translation:} the kernel reads the table itself.",
                       size=th.SIZE_SMALL, width=None, color=th.INK_2)
        c2 = kit.prose(r"\textbf{No page faults:} the allocator hands out blocks before the write.",
                       size=th.SIZE_SMALL, width=None, color=th.INK_2)
        side = VGroup(c1, c2).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        layout = VGroup(tab, side).arrange(DOWN, buff=0.45)
        self.place(layout)
        rows = [VGroup(*tab.cells[i]) for i in range(len(tab.cells))]
        with self.narrate(
            "Operating systems solved this in the 1960s with virtual memory. "
            "A process sees contiguous memory, backed by physical pages that sit anywhere. "
            "A sequence plays the process, and its logical blocks play virtual pages. "
            "A block table plays the page table, with 16-token blocks as pages. "
            "Shared pages become prefix sharing, and swapping goes to host memory. "
            "Two differences matter for your code. "
            "No hardware translates addresses, so the kernel reads the table itself. "
            "And there are no page faults: the allocator hands out blocks before the write."
        ) as t:
            self.play(FadeIn(tab.rules), FadeIn(rows[0]), run_time=0.6)
            t.wait_until("A sequence plays")
            self.play(FadeIn(rows[1], shift=RIGHT * 0.1), FadeIn(rows[2], shift=RIGHT * 0.1), run_time=0.7)
            t.wait_until("A block table plays")
            self.play(FadeIn(rows[3], shift=RIGHT * 0.1), FadeIn(rows[4], shift=RIGHT * 0.1),
                      FadeIn(rows[5], shift=RIGHT * 0.1), run_time=0.8)
            t.wait_until("Shared pages become")
            self.play(FadeIn(rows[6], shift=RIGHT * 0.1), FadeIn(rows[7], shift=RIGHT * 0.1), run_time=0.7)
            t.wait_until("No hardware")
            self.play(FadeIn(c1, shift=UP * 0.1), run_time=0.6)
            t.wait_until("And there are no")
            self.play(FadeIn(c2, shift=UP * 0.1), run_time=0.6)

    # ------------------------------------------------------------------
    # Blocks and the block table

    def block_pool(self):
        self.slide("block-pool", "Any free block can serve any sequence", book_section="s-blocks-and-the-block-table")
        g = kit.grid(3, 8, cell=0.72, gap=0.09, values=[[str(r * 8 + c) for c in range(8)] for r in range(3)],
                     size=th.SIZE_SMALL)
        raise_text(g)
        shape = kit.mono("(num_blocks, block_size, kv_heads, head_dim)", size=th.SIZE_LABEL, color=th.INK_2)
        pool_t = kit.label("The pool, allocated once at startup", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        left = VGroup(pool_t, g, shape).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        A, B, C = self.colors[1], self.colors[2], self.colors[3]

        def table_row(name, ids, color):
            nm = kit.label(name, size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
            cells = VGroup(*[id_cell(str(i), color) for i in ids]).arrange(RIGHT, buff=0.06)
            row = VGroup(nm, cells).arrange(RIGHT, buff=0.3)
            row.cells = cells
            return row

        ta = table_row("A", [23, 21, 19], A)
        tb = table_row("B", [22, 20], B)
        tc = table_row("C", [20, 22], C)
        tables_t = kit.label("Block tables", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        tables = VGroup(tables_t, ta, tb, tc).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        tc.move_to(tb, aligned_edge=LEFT)
        layout = VGroup(left, tables).arrange(RIGHT, buff=1.0, aligned_edge=UP)
        result = VGroup(
            kit.pill("External fragmentation: gone", self.color, size=th.SIZE_SMALL),
            kit.pill("Internal: at most B − 1 tokens", self.colors[1], size=th.SIZE_SMALL),
        ).arrange(RIGHT, buff=0.5)
        col = VGroup(layout, result).arrange(DOWN, buff=0.6)
        self.place(col)

        def phys(i):
            return g.cell(i // 8, i % 8)

        with self.narrate(
            "Paging divides the pool into fixed-size blocks. "
            "It's allocated once, at startup. "
            "Each sequence keeps a block table, an ordered list of the physical blocks that hold its tokens. "
            "Here, sequences A and B grow in turns, so their blocks aren't adjacent, or in order. "
            "When B finishes, its blocks go back on the free list, and a new sequence, C, takes them. "
            "Every block is interchangeable, so external fragmentation disappears completely. "
            "The only waste left is the unused tail of each sequence's last block."
        ) as t:
            self.play(FadeIn(pool_t), FadeIn(g), FadeIn(shape), run_time=0.8)
            t.wait_until("Each sequence keeps")
            self.play(FadeIn(tables_t), FadeIn(ta[0]), FadeIn(tb[0]), run_time=0.6)
            t.wait_until("Here, sequences A")
            order = [(ta, 0, 23, A), (tb, 0, 22, B), (ta, 1, 21, A), (tb, 1, 20, B), (ta, 2, 19, A)]
            for row, k, pid, col_ in order:
                self.play(g.fill(phys(pid), col_, tint=0.45), FadeIn(row.cells[k], shift=RIGHT * 0.1),
                          run_time=0.45)
            t.wait_until("When B finishes")
            self.play(g.fill(VGroup(phys(22), phys(20)), FREE, tint=None), FadeOut(tb), run_time=0.7)
            self.play(FadeIn(tc[0]), run_time=0.3)
            for k, pid in enumerate([20, 22]):
                self.play(g.fill(phys(pid), C, tint=0.45), FadeIn(tc.cells[k], shift=RIGHT * 0.1), run_time=0.45)
            t.wait_until("Every block is")
            self.play(FadeIn(result[0], scale=0.9), run_time=0.6)
            t.wait_until("The only waste")
            self.play(FadeIn(result[1], scale=0.9), run_time=0.6)

    # ------------------------------------------------------------------
    # What a page costs

    def page_cost(self):
        self.slide("page-cost", "One 16-token page costs exactly 1 MiB",
                   book_section="s-what-a-page-costs-on-this-model")
        n = 16
        g = kit.grid(n, n, cell=0.27)
        tok = kit.label("16 tokens →", size=th.SIZE_LABEL, color=th.INK_3).next_to(g, UP, buff=0.12)
        tok.align_to(g, LEFT)
        lay = kit.label("16 layers →", size=th.SIZE_LABEL, color=th.INK_3).rotate(-PI / 2)
        lay.next_to(g, LEFT, buff=0.12).align_to(g, UP)
        unit = kit.label("each cell: one token, one layer", size=th.SIZE_LABEL, color=th.INK_3)
        unit.next_to(g, DOWN, buff=0.15).align_to(g, LEFT)
        fig = VGroup(g, tok, lay, unit)
        f1 = kit.formula(r"2 \times 4 \times 256 \times 2\ \text{B} = 4\ \text{KiB}", size=th.SIZE_MATH_SMALL)
        f2 = kit.formula(r"16 \times 4\ \text{KiB} = 64\ \text{KiB per token}", size=th.SIZE_MATH_SMALL)
        f3 = kit.formula(r"16 \times 64\ \text{KiB} = 1\ \text{MiB per page}", size=th.SIZE_MATH_SMALL)
        box3 = SurroundingRectangle(f3, color=self.color, buff=0.14, corner_radius=0.08, stroke_width=3)
        f4 = kit.formula(r"24{,}468 \text{ pages} \times 16 = 391{,}488 \text{ tokens}", size=th.SIZE_MATH_SMALL)
        spend = kit.prose(r"Spend it: 60 sequences at 4k, 11 at 32k, or one at 262k (16.1 GiB).",
                          size=th.SIZE_SMALL, width=6.4, color=th.INK_2)
        maths = VGroup(f1, f2, VGroup(f3, box3), f4, spend).arrange(DOWN, buff=0.42, aligned_edge=LEFT)
        layout = VGroup(fig, maths).arrange(RIGHT, buff=0.8)
        self.place(layout)
        cell0 = g.cell(0, 0)
        col0 = g.col(0)
        with self.narrate(
            "In one full-attention layer, a token stores a key and a value for each of 4 KV heads. "
            "Each is 256 wide, in bf16, so that's 4 KiB. "
            "Across the 16 full-attention layers, a token costs 64 KiB. "
            "A block of 16 tokens therefore costs exactly 1 MiB. "
            "The pool holds 24,468 of them. "
            "That's 391,488 tokens of KV. "
            "After each sequence's recurrent state, spend it as you like: 60 sequences at 4k, 11 at 32k, or one at 262k."
        ) as t:
            self.play(FadeIn(fig), run_time=0.8)
            t.wait_until("Each is 256")
            self.play(g.fill(cell0, self.color, tint=0.7), FadeIn(f1), run_time=0.7)
            t.wait_until("Across the 16")
            self.play(g.fill(col0, self.color, tint=0.7), FadeIn(f2), run_time=0.8)
            t.wait_until("A block of 16")
            self.play(LaggedStart(*[g.fill(g.col(c), self.color, tint=0.45) for c in range(1, n)], lag_ratio=0.1),
                      FadeIn(f3), run_time=1.4)
            self.play(Create(box3), run_time=0.5)
            t.wait_until("The pool holds")
            self.play(FadeIn(f4), run_time=0.7)
            t.wait_until("After each sequence")
            self.play(FadeIn(spend), run_time=0.6)

    # ------------------------------------------------------------------
    # Mapping positions to slots

    def slot_map(self):
        self.slide("slot-map", "Position 37 lives in slot 149", book_section="s-mapping-positions-to-slots")
        f = kit.formula(r"\operatorname{slot}(p) =", r"\text{table}[\lfloor p / B \rfloor]", r"\cdot B +",
                        r"(p \bmod B)", size=th.SIZE_MATH)
        f[1].set_color(th.HL_A)
        f[3].set_color(th.HL_B)
        w, h = 0.155, 0.45
        blocks = VGroup(*[strip(16, w, h) for _ in range(4)]).arrange(RIGHT, buff=0.22)
        heads = VGroup(*[kit.label(f"logical {i}", size=th.SIZE_LABEL, color=th.INK_3).next_to(b, UP, buff=0.1)
                         for i, b in enumerate(blocks)])
        seq_t = kit.label("sequence, positions 0 to 63", size=th.SIZE_LABEL, color=th.INK_3)
        seq_t.next_to(blocks, DOWN, buff=0.15).align_to(blocks, LEFT)
        logical = VGroup(blocks, heads, seq_t)
        ids = [4, 12, 9, 2]
        table = VGroup(*[id_cell(str(i), th.HL_A, w=0.7, h=0.6, tint=0.15) for i in ids]).arrange(RIGHT, buff=0.06)
        table_t = kit.label("block table", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        table_t.next_to(table, LEFT, buff=0.3)
        tgroup = VGroup(table, table_t)
        pool = kit.grid(2, 8, cell=0.62, gap=0.08, values=[[str(r * 8 + c) for c in range(8)] for r in range(2)],
                        size=th.SIZE_LABEL)
        raise_text(pool)
        pool_t = kit.label("physical blocks", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        pool_t.next_to(pool, UP, buff=0.15)
        pgroup = VGroup(pool, pool_t)
        lower = VGroup(tgroup, pgroup).arrange(RIGHT, buff=1.4)
        tgroup.align_to(pool.cell(0, 0), UP)
        result = kit.formula(r"9 \times 16 + 5 = 149", size=th.SIZE_MATH)
        res_box = SurroundingRectangle(result, color=self.color, buff=0.14, corner_radius=0.08, stroke_width=3)
        col = VGroup(f, logical, lower, VGroup(result, res_box)).arrange(DOWN, buff=0.5)
        self.place(col)
        pos = blocks[2][5]
        mark37 = kit.label("37", size=th.SIZE_LABEL, color=th.INK, weight=MEDIUM)
        mark37.next_to(pos, DOWN, buff=0.1)
        off = kit.label("block 2, offset 5", size=th.SIZE_LABEL, color=th.HL_B)
        off.next_to(blocks[2], DOWN, buff=0.15).align_to(blocks[2], LEFT).shift(RIGHT * 0.6)
        mark37.next_to(off, LEFT, buff=0.15).set_y(off.get_y())
        a1 = VGroup(Arrow(blocks[2].get_bottom() + DOWN * 0.45, table[2].get_top(), buff=0.08, color=th.HL_A,
                          stroke_width=3, tip_length=0.18))
        a1.line = a1[0]
        target = pool.cell(1, 1)
        a2 = CurvedArrow(table[2].get_bottom() + DOWN * 0.05, target.get_bottom() + DOWN * 0.05, angle=PI / 3,
                         color=th.HL_A, stroke_width=3, tip_length=0.18)
        with self.narrate(
            "A logical position becomes a slot in the pool through one table lookup. "
            "Take position 37, with 16-token blocks. "
            "Divide by 16, and it sits in logical block 2, at offset 5. "
            "The table says logical block 2 lives in physical block 9. "
            "Multiply 9 by 16 to jump to that block's first slot, then add the offset. "
            "The slot is 149. "
            "In code, that's a gather into the table, a multiply, and an add, over every position at once."
        ) as t:
            self.play(FadeIn(f), FadeIn(logical), FadeIn(tgroup), FadeIn(pgroup), run_time=0.9)
            t.wait_until("Take position 37")
            self.play(pos.animate.set_fill(th.tint(th.HL_B, 0.8)), FadeIn(mark37), run_time=0.6)
            t.wait_until("Divide by 16")
            self.play(VGroup(*blocks[2]).animate.set_stroke(th.HL_A, width=2.5), FadeIn(off), run_time=0.7)
            t.wait_until("The table says")
            self.play(GrowArrow(a1.line), Indicate(table[2], color=th.HL_A, scale_factor=1.1), run_time=0.8)
            self.play(Create(a2), pool.fill(target, th.HL_A, tint=0.45), run_time=0.8)
            t.wait_until("Multiply 9")
            self.play(Indicate(f[1], color=th.HL_A), run_time=0.6)
            self.play(Indicate(f[3], color=th.HL_B), run_time=0.6)
            t.wait_until("The slot is")
            self.play(FadeIn(result), Create(res_box), run_time=0.8)

    # ------------------------------------------------------------------
    # Choosing the page size

    def page_size(self):
        self.slide("page-size", "The page size barely matters for memory", book_section="s-choosing-the-page-size")
        tab = kit.table(
            [
                ["1", "0", "32,768", "32,768"],
                ["16", "7.5 tokens (480 KiB)", "2,048", "2,048"],
                ["64", "31.5 tokens (1.97 MiB)", "512", "512"],
                ["256", "127.5 tokens (7.97 MiB)", "128", "128"],
            ],
            header=["Block size $B$", "Avg. tail waste", "Table entries", "Loads per decode"],
            align=["r", "l", "r", "r"],
            size=th.SIZE_SMALL,
            highlight_rows=[1],
        )
        cap = kit.label("At 32k context, against 2 GiB of KV per sequence", size=th.SIZE_SMALL, color=th.INK_3)
        bw = kit.box("Bandwidth: no", sublabel="4 bytes of table locate 1 MiB: 1 to 262,144", color=self.colors[2],
                     width=5.7)
        lat = kit.box("Latency: yes", sublabel="each table read is a dependent load", color=th.BAD, width=5.7)
        costs = VGroup(bw, lat).arrange(RIGHT, buff=0.4)
        col = VGroup(cap, tab, costs).arrange(DOWN, buff=0.5)
        cap.align_to(tab, LEFT)
        self.place(col)
        rows = [VGroup(*tab.cells[i]) for i in range(5)]
        waste = VGroup(*[tab.cells[i][1] for i in range(1, 5)])
        indir = VGroup(*[c for i in range(1, 5) for c in tab.cells[i][2:]])
        with self.narrate(
            "The block size trades two costs. "
            "Smaller blocks waste less: on average, about half a block per sequence. "
            "Larger blocks need fewer table entries, and fewer loads per decode step. "
            "At 32k context, a 16-token block wastes 7.5 tokens on average, or 480 KiB. "
            "Even at 256, the waste is 0.4 percent of the sequence. "
            "The table isn't a bandwidth problem either: 4 bytes of it locate 1 MiB of keys and values. "
            "It's a latency problem. "
            "Each table read is a dependent load, so the memory system can't run ahead."
        ) as t:
            self.play(FadeIn(cap), FadeIn(tab.rules), FadeIn(rows[0]), FadeIn(VGroup(*[r[0] for r in tab.cells[1:]])),
                      run_time=0.8)
            t.wait_until("Smaller blocks")
            self.play(FadeIn(waste, shift=RIGHT * 0.1), run_time=0.7)
            t.wait_until("Larger blocks")
            self.play(FadeIn(indir, shift=RIGHT * 0.1), run_time=0.7)
            t.wait_until("At 32k context")
            self.play(FadeIn(tab.bands), run_time=0.6)
            t.wait_until("Even at 256")
            self.play(Indicate(tab.cells[4][1], color=self.color, scale_factor=1.06), run_time=0.9)
            t.wait_until("The table isn't")
            self.play(FadeIn(bw, shift=UP * 0.1), run_time=0.7)
            t.wait_until("Each table read")
            self.play(FadeIn(lat, shift=UP * 0.1), run_time=0.7)

    # ------------------------------------------------------------------
    # The free list

    def alloc_code(self):
        self.slide("alloc-code", "The allocator counts blocks before it pops", book_section="s-the-free-list")
        panel = kit.code(ALLOC_SRC, "python", caption="PagedCache")
        self.place(panel)
        with self.narrate(
            "In the allocator, the function that counts blocks is the piece worth reading twice. "
            "First, it finds the capacity the sequence already holds, in tokens. "
            "The shortfall is the current length plus the new tokens, minus that capacity. "
            "Floor division of a negated number is a ceiling, so this rounds the shortfall up to whole blocks. "
            "Allocation asks that question first, and raises if the free list is short. "
            "Then it pops one free block per block needed, and appends it to the table. "
            "Last, the length accumulates."
        ) as t:
            self.play(FadeIn(panel.frame), *panel.reveal(1, 2, 3), run_time=0.8)
            t.wait_until("First, it finds")
            self.play(*panel.reveal(4), panel.highlight(4), run_time=0.6)
            t.wait_until("The shortfall is")
            self.play(*panel.reveal(5), run_time=0.6)
            t.wait_until("Floor division")
            self.play(*panel.reveal(6), panel.highlight(6), run_time=0.6)
            t.wait_until("Allocation asks")
            self.play(*panel.reveal(8, 9, 10, 11), run_time=0.8)
            t.wait_until("Then it pops")
            self.play(*panel.reveal(12, 13), panel.highlight(13), run_time=0.7)
            t.wait_until("Last, the length")
            self.play(*panel.reveal(14), run_time=0.5)

    def decode_growth(self):
        self.slide("decode-growth", "Fifteen decode steps in sixteen allocate nothing", book_section="s-the-free-list")
        A = self.colors[1]
        w, h = 0.2, 0.5
        b0 = strip(16, w, h, A, filled=16)
        b1 = strip(16, w, h, A, filled=13)
        slot = DashedVMobject(Rectangle(width=b0.width, height=h, stroke_color=th.INK_4, stroke_width=2),
                              num_dashes=36)
        b2 = strip(16, w, h, A, filled=0)
        row = VGroup(b0, b1, slot).arrange(RIGHT, buff=0.35)
        b2.move_to(slot)
        tags = VGroup(*[id_cell(s, A, w=0.7, h=0.5) for s in ("17", "8")])
        for tg, b in zip(tags, (b0, b1)):
            tg.next_to(b, UP, buff=0.15)
        tbl_t = kit.label("sequence's block table", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        tbl_t.next_to(tags, UP, buff=0.2).align_to(b0, LEFT)
        tail_t = kit.label("tail block", size=th.SIZE_LABEL, color=th.INK_3).next_to(b1, DOWN, buff=0.15)
        top = VGroup(tbl_t, tags, row, tail_t)
        free = VGroup(*[id_cell(s, None, w=0.7, h=0.55) for s in ("3", "11", "6", "14")]).arrange(RIGHT, buff=0.08)
        free_t = kit.label("free list", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        free_t.next_to(free, LEFT, buff=0.35)
        end_t = kit.label("← pop and push at this end", size=th.SIZE_LABEL, color=th.INK_3)
        fgroup = VGroup(free_t, free)
        quiet = kit.label("no new block", size=th.SIZE_SMALL, color=th.GOOD, weight=MEDIUM)
        lifo = kit.callout("key", "Last in, first out: the block freed a moment ago is the next handed out, "
                                  "which suits the L2 cache and the TLB.", width=11.0, size=th.SIZE_SMALL)
        col = VGroup(top, fgroup, lifo).arrange(DOWN, buff=0.75)
        fgroup.align_to(top, LEFT)
        self.place(col)
        b2.move_to(slot)
        end_t.next_to(free, RIGHT, buff=0.95)
        quiet.next_to(b1, DOWN, buff=0.15).align_to(b1, RIGHT)
        new_id = id_cell("21", None, w=0.7, h=0.55)
        new_id.next_to(free, RIGHT, buff=0.08)
        new_id.shift(UP * 1.0)
        with self.narrate(
            "Watch a sequence decode, one token per step. "
            "Inside its tail block, the shortfall is negative, so the count is zero, and nothing is allocated. "
            "That's fifteen steps out of sixteen. "
            "Meanwhile, another sequence finishes, and its block goes back on the end of the free list. "
            "When a token crosses the block boundary, the allocator pops from that same end. "
            "So the free list is last in, first out. "
            "A block freed a moment ago is the next one handed out, which suits the L2 cache and the TLB."
        ) as t:
            self.play(FadeIn(top), FadeIn(fgroup), Create(slot), run_time=0.8)
            t.wait_until("Inside its tail")
            for i in (13, 14, 15):
                self.play(b1[i].animate.set_fill(th.tint(A, 0.45)), run_time=0.4)
                self.wait(0.2)
            self.play(FadeIn(tail_t), run_time=0.3)
            t.wait_until("That's fifteen")
            self.play(FadeIn(quiet, shift=UP * 0.1), run_time=0.5)
            t.wait_until("Meanwhile")
            self.play(FadeIn(new_id, shift=DOWN * 0.2), run_time=0.4)
            self.play(new_id.animate.shift(DOWN * 1.0), FadeIn(end_t), run_time=0.6)
            t.wait_until("When a token crosses")
            tag2 = id_cell("21", A, w=0.7, h=0.5).next_to(b2, UP, buff=0.15).set_y(tags[0].get_y())
            self.play(new_id.animate.move_to(tag2), run_time=0.8)
            self.play(FadeOut(new_id), FadeIn(tag2), FadeOut(slot), FadeIn(b2), FadeOut(quiet), run_time=0.5)
            self.play(b2[0].animate.set_fill(th.tint(A, 0.45)), run_time=0.4)
            t.wait_until("So the free list")
            self.play(FadeIn(lifo, shift=UP * 0.1), run_time=0.7)
            self.play(Indicate(VGroup(tags, tag2), color=A, scale_factor=1.1), run_time=1.0)

    def out_of_blocks(self):
        self.slide("out-of-blocks", "Fail before the write, and rarely", book_section="s-the-free-list")
        steps = VGroup(
            kit.box("Token 501 crosses a boundary", width=3.9),
            kit.box("Free list empty: raise", color=th.BAD, width=3.9),
            kit.box("Scheduler preempts someone", color=self.colors[1], width=3.9),
        ).arrange(RIGHT, buff=0.55)
        W, H = 11.0, 0.6
        bar = frame_bar(W, H)
        used = seg(0, W * 0.98, H, self.color, tint=0.35)
        wm = seg(W * 0.98, W * 0.02, H, th.INK_3, tint=0.5)
        bar_t = kit.label("pool: 24,468 pages", size=th.SIZE_SMALL, color=th.INK_2).next_to(bar, UP, buff=0.12)
        bar_t.align_to(bar, LEFT)
        wm_t = kit.label("watermark: 2%, about 489 pages, kept free", size=th.SIZE_SMALL, color=th.INK)
        wm_t.next_to(bar, UP, buff=0.12).align_to(bar, RIGHT)
        bars = VGroup(bar, used, wm, bar_t, wm_t)
        code = kit.code(RELEASE_SRC, "python", caption="release")
        leak = kit.callout("warning", "Nothing crashes. Capacity shrinks until the engine preempts all the time.",
                           title="A leaked block", width=5.4, size=th.SIZE_SMALL)
        bottom = VGroup(code, leak).arrange(RIGHT, buff=0.5)
        col = VGroup(steps, bars, bottom).arrange(DOWN, buff=0.6)
        self.place(col)
        arrs = VGroup(kit.arrow(steps[0], steps[1]), kit.arrow(steps[1], steps[2]))
        with self.narrate(
            "A sequence that has run for 500 steps crosses a block boundary on its 501st token, and the free list "
            "is empty. "
            "The allocator raises, with nothing half allocated, and the scheduler preempts someone. "
            "Because counting blocks is a pure query, the scheduler can ask before it commits a write. "
            "And because recovery is expensive, it keeps a watermark free: 2 percent of the pool, about 489 pages. "
            "Releasing is the reverse, and it must return every block. "
            "A leaked block doesn't crash anything. "
            "It shrinks capacity until the engine preempts constantly."
        ) as t:
            t.wait_until("A sequence that has")
            self.play(FadeIn(steps[0], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("The allocator raises")
            self.play(GrowArrow(arrs[0].line), FadeIn(steps[1], shift=RIGHT * 0.1), run_time=0.7)
            self.play(GrowArrow(arrs[1].line), FadeIn(steps[2], shift=RIGHT * 0.1), run_time=0.7)
            t.wait_until("And because recovery")
            self.play(FadeIn(bar), FadeIn(bar_t), GrowFromEdge(used, LEFT), run_time=0.8)
            self.play(FadeIn(wm), FadeIn(wm_t), run_time=0.6)
            t.wait_until("Releasing is")
            self.play(FadeIn(code.frame), *code.reveal(), run_time=0.8)
            self.play(code.highlight(2, 4), run_time=0.5)
            t.wait_until("A leaked block")
            self.play(FadeIn(leak, shift=UP * 0.1), run_time=0.7)

    # ------------------------------------------------------------------
    # The kernel

    def kernel_walk(self):
        self.slide("kernel-walk", "The decode kernel follows the table", book_section="s-the-kernel")
        A = self.colors[1]
        ids = [9, 2, 14]
        table = VGroup(*[id_cell(str(i), A, w=0.72, h=0.6) for i in ids]).arrange(RIGHT, buff=0.06)
        t_idx = VGroup(*[kit.label(str(i), size=th.SIZE_LABEL, color=th.INK_4).next_to(c, DOWN, buff=0.1)
                         for i, c in enumerate(table)])
        table_t = kit.label("block table", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        ctx = kit.label("context length 40", size=th.SIZE_LABEL, color=th.INK_3)
        tgroup = VGroup(table_t, VGroup(table, t_idx), ctx).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        pool = kit.grid(3, 6, cell=0.66, gap=0.08, values=[[str(r * 6 + c) for c in range(6)] for r in range(3)],
                        size=th.SIZE_LABEL)
        raise_text(pool)
        others = [0, 4, 5, 7, 11, 12, 16]
        pool.paint(VGroup(*[pool.cell(i // 6, i % 6) for i in others]), th.INK_4, tint=0.2)
        pool.paint(VGroup(*[pool.cell(i // 6, i % 6) for i in ids]), A, tint=0.35)
        pool_t = kit.label("K and V pool", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        pgroup = VGroup(pool_t, pool).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        # The program: one (sequence, head) pair.
        prog = RoundedRectangle(width=4.8, height=2.9, corner_radius=0.12, stroke_color=self.color,
                                stroke_width=th.STROKE_BOX, fill_color=th.tint(self.color, 0.05), fill_opacity=1)
        prog_t = kit.label("one program per sequence and head", size=th.SIZE_LABEL, color=th.INK_2,
                           weight=MEDIUM)
        prog_t.next_to(prog, UP, buff=0.15)
        kv = strip(16, 0.24, 0.5)
        kv.move_to(prog.get_center() + UP * 0.55)
        kv_t = kit.label("this block's keys", size=th.SIZE_LABEL, color=th.INK_3).next_to(kv, DOWN, buff=0.1)
        state = VGroup(kit.box("$q$", color=th.VIOLET, width=0.8, height=0.65),
                       kit.box("$m$", color=th.HL_A, width=0.8, height=0.65),
                       kit.box(r"$\ell$", color=th.HL_B, width=0.8, height=0.65),
                       kit.box("acc", color=th.HL_C, width=1.1, height=0.65)).arrange(RIGHT, buff=0.15)
        state.move_to(prog.get_bottom() + UP * 0.55)
        program = VGroup(prog, prog_t, kv, kv_t, state)
        layout = VGroup(tgroup, pgroup, program).arrange(RIGHT, buff=0.8)
        mask_t = kit.label("masked: past 40", size=th.SIZE_LABEL, color=th.BAD)
        self.place(layout)
        mask_t.next_to(kv, DOWN, buff=0.1).align_to(kv, RIGHT)
        pointer = SurroundingRectangle(table[0], color=self.color, buff=0.06, stroke_width=4)
        with self.narrate(
            "Decode attends one query token against the cached prefix. "
            "One program handles one sequence and one head. "
            "It loads the query once, and sets up the running state: a maximum, a sum, and an accumulator. "
            "Then it loops over the sequence's logical blocks. "
            "For each one, it reads the physical block number from the table. "
            "Every address after that is computed from it, in software. "
            "It loads that block's keys and values, scores them, and updates the running state. "
            "In the last block, a mask drops the slots past the context length. "
            "At the end come one division and one store."
        ) as t:
            self.play(FadeIn(tgroup), FadeIn(pgroup), run_time=0.8)
            t.wait_until("One program handles")
            self.play(FadeIn(prog), FadeIn(prog_t), FadeIn(kv), FadeIn(kv_t), run_time=0.7)
            t.wait_until("It loads the query")
            self.play(LaggedStart(*[FadeIn(s, shift=UP * 0.1) for s in state], lag_ratio=0.2), run_time=1.0)
            t.wait_until("Then it loops")
            self.play(Create(pointer), run_time=0.5)
            for b, pid in enumerate(ids):
                cell = pool.cell(pid // 6, pid % 6)
                if b:
                    self.play(pointer.animate.move_to(table[b]), run_time=0.4)
                arr = SurroundingRectangle(cell, color=self.color, buff=0.05, stroke_width=4)
                if b == 0:
                    t.wait_until("For each one")
                self.play(Create(arr), Indicate(cell, color=self.color, scale_factor=1.12), run_time=0.7)
                if b == 0:
                    t.wait_until("It loads that block")
                n = 16 if b < 2 else 8
                fly = cell.copy()
                self.play(fly.animate.move_to(kv).set_opacity(0), *[kv[i].animate.set_fill(th.tint(A, 0.45))
                                                                       for i in range(n)], run_time=0.7)
                self.remove(fly)
                if b == 2:
                    t.wait_until("In the last block")
                    self.play(*[kv[i].animate.set_fill(th.tint(th.BAD, 0.3)) for i in range(8, 16)],
                              FadeOut(kv_t), FadeIn(mask_t), run_time=0.7)
                self.play(Indicate(state[1:], color=th.HL_C, scale_factor=1.06), run_time=0.5)
                if b < 2:
                    self.play(FadeOut(arr), *[kv[i].animate.set_fill(th.RAISE) for i in range(16)], run_time=0.3)

    def kernel_code(self):
        self.slide("kernel-code", "Only where the keys come from has changed", book_section="s-the-kernel")
        panel = kit.code(KERNEL_SRC, "python", caption="paged decode kernel")
        self.place(panel)
        with self.narrate(
            "Here's that loop in Triton. "
            "It runs to a compile-time maximum with a runtime guard, because Triton needs a static trip count. "
            "This load is the indirection, in one line: the physical block number from the table. "
            "The mask removes the tail of the last block. "
            "The scores are a broadcast multiply and a sum, not a matmul, because there's only one query row. "
            "Past that line, it's chapter 14's online softmax, verbatim. "
            "The only thing that changed is where the keys came from."
        ) as t:
            self.play(FadeIn(panel.frame), *panel.reveal(1, 2, 3), run_time=0.8)
            self.play(panel.highlight(3), run_time=0.5)
            t.wait_until("This load is")
            self.play(*panel.reveal(4, 5), panel.highlight(4, 5, color=th.tint(self.color, 0.3)), run_time=0.8)
            t.wait_until("The mask removes")
            self.play(*panel.reveal(6, 7, 8, 9), run_time=0.8)
            t.wait_until("The scores are")
            self.play(*panel.reveal(10, 11), run_time=0.7)
            t.wait_until("Past that line")
            self.play(*panel.reveal(12, 13, 14), run_time=0.7)
            t.wait_until("The only thing")
            self.play(Circumscribe(panel.lines(4, 5), color=self.color, buff=0.08), run_time=1.2)

    # ------------------------------------------------------------------
    # Prefix sharing and copy-on-write

    def prefix_share(self):
        self.slide("prefix-share", "Two sequences, one copy of the prefix",
                   book_section="s-prefix-sharing-and-copy-on-write")
        A, B = self.colors[1], self.colors[2]
        shared_ids = ["7", "3", "12"]
        phys = VGroup(*[kit.box(i, sublabel="refs 1", color=self.color, width=1.25, height=1.0) for i in shared_ids])
        phys.arrange(RIGHT, buff=0.4)
        pa = kit.box("5", sublabel="partial", color=A, width=1.25, height=1.0)
        pb = kit.box("10", sublabel="copy", color=B, width=1.25, height=1.0)
        pa.next_to(phys, RIGHT, buff=0.7).shift(UP * 0.7)
        pb.next_to(phys, RIGHT, buff=0.7).shift(DOWN * 0.7)
        ta = VGroup(*[id_cell(s, A) for s in ["7", "3", "12", "5"]]).arrange(RIGHT, buff=0.06)
        tb = VGroup(*[id_cell(s, B) for s in ["7", "3", "12", "10"]]).arrange(RIGHT, buff=0.06)
        ta_n = kit.label("A", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
        tb_n = kit.label("B", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
        ga = VGroup(ta_n, ta).arrange(RIGHT, buff=0.25)
        gb = VGroup(tb_n, tb).arrange(RIGHT, buff=0.25)
        ga.next_to(phys, UP, buff=1.2).align_to(phys, LEFT)
        gb.next_to(phys, DOWN, buff=1.2).align_to(phys, LEFT)
        diagram = VGroup(ga, gb, phys, pa, pb)
        rules = kit.bullets(
            [r"Count references per block.", r"Free a block at zero.", r"Copy before writing if the count is above one."],
            size=th.SIZE_SMALL, width=4.4, buff=0.25,
        )
        rules_t = kit.label("Copy-on-write", size=th.SIZE_SMALL, color=self.color, weight=SEMIBOLD)
        side = VGroup(rules_t, rules).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        top = VGroup(diagram, side).arrange(RIGHT, buff=0.8)
        foot = kit.label("A 500-token prompt: 31 full blocks shared, 4 tokens copied", size=th.SIZE_SMALL,
                         color=th.INK_2, weight=MEDIUM)
        col = VGroup(top, foot).arrange(DOWN, buff=0.5)
        self.place(col)
        la = VGroup(*[Line(ta[i].get_bottom(), phys[i].get_top(), color=A, stroke_width=2.5, buff=0.06)
                      for i in range(3)])
        la.add(Line(ta[3].get_bottom(), pa.get_top(), color=A, stroke_width=2.5, buff=0.06))
        lb = VGroup(*[Line(tb[i].get_top(), phys[i].get_bottom(), color=B, stroke_width=2.5, buff=0.06)
                      for i in range(3)])
        lb_copy = Line(tb[3].get_top(), pb.get_bottom(), color=B, stroke_width=2.5, buff=0.06)
        refs2 = [kit.label("refs 2", size=th.SIZE_LABEL, color=th.INK_3).move_to(p.sub) for p in phys]
        with self.narrate(
            "Once addressing is indirect, two sequences can point at the same physical block. "
            "It's how two processes share a page after fork, with three rules. "
            "Keep a reference count per block. "
            "Free a block only when its count reaches zero. "
            "Copy a block before writing to it if its count is above one. "
            "Sharing is page-aligned. "
            "A 500-token system prompt is 31 full blocks, plus 4 tokens in a partial block. "
            "Only the full blocks are shared. "
            "The partial one is copied, because each sequence writes its own next token into it."
        ) as t:
            self.play(FadeIn(ga), FadeIn(phys), FadeIn(pa), Create(la), run_time=1.0)
            t.wait_until("It's how two")
            self.play(FadeIn(gb[0]), FadeIn(tb[:3]), Create(lb), run_time=0.9)
            self.play(FadeIn(rules_t), run_time=0.4)
            t.wait_until("Keep a reference")
            self.play(FadeIn(rules[0], shift=RIGHT * 0.1),
                      *[Transform(p.sub, r) for p, r in zip(phys, refs2)], run_time=0.7)
            t.wait_until("Free a block")
            self.play(FadeIn(rules[1], shift=RIGHT * 0.1), run_time=0.5)
            t.wait_until("Copy a block")
            self.play(FadeIn(rules[2], shift=RIGHT * 0.1), run_time=0.5)
            t.wait_until("A 500-token")
            self.play(FadeIn(foot), run_time=0.6)
            t.wait_until("The partial one")
            self.play(TransformFromCopy(pa, pb), run_time=0.9)
            self.play(FadeIn(tb[3]), Create(lb_copy), run_time=0.6)

    def prefix_savings(self):
        self.slide("prefix-savings", "Sharing saves memory and skips prefill",
                   book_section="s-prefix-sharing-and-copy-on-write")
        g = kit.grid(4, 16, cell=0.34, gap=0.05)
        g.paint(g.block(0, 4, 0, 16), self.color, tint=0.45)
        g_t = kit.label("64 requests, one prefix copy each", size=th.SIZE_LABEL, color=th.INK_3)
        g_t.next_to(g, UP, buff=0.15).align_to(g, LEFT)
        mem_h = kit.label("Memory", size=th.SIZE_SMALL, color=self.color, weight=SEMIBOLD)
        m1 = kit.formula(r"500 \times 64\ \text{KiB} = 31.25\ \text{MiB}", size=th.SIZE_MATH_SMALL)
        m2 = kit.formula(r"63 \times 31.25\ \text{MiB} = 1.92\ \text{GiB}", size=th.SIZE_MATH_SMALL)
        m3 = kit.label("about 8% of the pool", size=th.SIZE_SMALL, color=th.INK_2)
        mem = VGroup(mem_h, VGroup(g_t, g), m1, m2, m3).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        cmp_h = kit.label("Compute", size=th.SIZE_SMALL, color=self.colors[1], weight=SEMIBOLD)
        c1 = kit.formula(r"2 \times 26.9 \times 10^9 \times 500", size=th.SIZE_MATH_SMALL)
        c2 = kit.formula(r"= 26.9\ \text{TFLOP per request}", size=th.SIZE_MATH_SMALL)
        c3 = kit.formula(r"\text{at } 312\ \text{TFLOP/s: } 86\ \text{ms}", size=th.SIZE_MATH_SMALL)
        c4 = kit.callout("result", "Every request after the first skips it: a cut to time to first token.",
                         title="Latency, not only memory", width=5.2, size=th.SIZE_SMALL)
        cmp = VGroup(cmp_h, c1, c2, c3, c4).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        layout = VGroup(mem, cmp).arrange(RIGHT, buff=1.0, aligned_edge=UP)
        self.place(layout)
        saved = VGroup(*[g.cell(r, c) for r in range(4) for c in range(16) if (r, c) != (0, 0)])
        with self.narrate(
            "Say every request carries the same 500-token system prompt. "
            "Each copy of that prefix costs 31.25 MiB. "
            "With 64 concurrent requests, 63 of them stop paying it. "
            "That saves 1.92 GiB, about 8 percent of the pool. "
            "The shared prefix also skips prefill. "
            "Prefill costs about 2 FLOPs per token for each of the 26.9 billion parameters. "
            "So 500 tokens cost 26.9 TFLOP, per request. "
            "That's 86 ms of tensor-core time at 312 TFLOP/s. "
            "Every later request skips it, which cuts time to first token."
        ) as t:
            self.play(FadeIn(mem_h), FadeIn(g_t), FadeIn(g), run_time=0.8)
            t.wait_until("Each copy")
            self.play(FadeIn(m1), run_time=0.6)
            t.wait_until("With 64 concurrent")
            self.play(g.fill(saved, FREE, tint=None), run_time=1.2)
            t.wait_until("That saves")
            self.play(FadeIn(m2), FadeIn(m3), run_time=0.7)
            t.wait_until("The shared prefix")
            self.play(FadeIn(cmp_h), run_time=0.4)
            t.wait_until("Prefill costs")
            self.play(FadeIn(c1), run_time=0.6)
            t.wait_until("So 500 tokens")
            self.play(FadeIn(c2), run_time=0.6)
            t.wait_until("That's 86")
            self.play(FadeIn(c3), run_time=0.6)
            t.wait_until("Every later request")
            self.play(FadeIn(c4, shift=UP * 0.1), run_time=0.7)

    # ------------------------------------------------------------------
    # The 48 layers that don't page

    def fixed_state(self):
        self.slide("fixed-state", "The recurrent state doesn't page", book_section="s-the-48-layers-that-don-t-page")
        f = kit.formula(r"\text{cost}(L) =", r"147.8\ \text{MiB}", r"+", r"L \times 64\ \text{KiB}")
        f[1].set_color(STATE)
        f[3].set_color(KVC)
        eq = kit.formula(r"\frac{147.8\ \text{MiB}}{64\ \text{KiB}} = 2365\ \text{tokens}", size=th.SIZE_MATH_SMALL)
        top = VGroup(f, eq).arrange(RIGHT, buff=1.0)
        W, H = 11.0, 0.6
        pool_mib = 25_656_894_976 / 2**20
        sw = 147.75 / pool_mib * W
        b1, b2 = frame_bar(W, H), frame_bar(W, H)
        s1 = VGroup(*[seg(i * sw, sw, H, STATE, tint=0.45 if i % 2 else 0.7) for i in range(64)])
        rest = seg(64 * sw, W - 64 * sw, H, KVC, tint=0.2)
        rest_t = kit.label("61% left for all their KV", size=th.SIZE_LABEL, color=th.INK_2).move_to(rest)
        s2 = VGroup(*[seg(i * sw, sw, H, STATE, tint=0.45 if i % 2 else 0.7) for i in range(165)])
        for s in list(s1) + list(s2):
            s.set_stroke(width=0)
        g1 = VGroup(b1, s1, rest, rest_t)
        g2 = VGroup(b2, s2)
        l1 = kit.label("Batch 64: 9.24 GiB of state, 39% of the pool", size=th.SIZE_SMALL, color=th.INK_2,
                       weight=MEDIUM)
        l2 = kit.label("State alone: 165 sequences with zero context fill it", size=th.SIZE_SMALL, color=th.INK_2,
                       weight=MEDIUM)
        r1 = VGroup(l1, g1).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        r2 = VGroup(l2, g2).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        why = VGroup(
            kit.box("KV entry", sublabel="written once, read forever", color=KVC, width=5.4),
            kit.box("Recurrent state", sublabel="overwritten in full every step", color=STATE, width=5.4),
        ).arrange(RIGHT, buff=0.5)
        col = VGroup(top, why, r1, r2).arrange(DOWN, buff=0.5)
        self.place(col)
        with self.narrate(
            "The other 48 layers keep a recurrent state per sequence: 147.8 MiB, whatever the context length. "
                        "A KV entry is written once and read forever, so it can live anywhere. "
            "The state is overwritten in full every step, so no part of it can be paged or shared. "
            "In the pool's units, admitting a sequence costs as much as 2365 tokens of KV. "
            "At a batch of 64, the state holds 9.24 GiB, or 39 percent of the pool. "
            "And the state alone caps the batch at 165 sequences, with zero context each."
        ) as t:
            self.play(FadeIn(f), run_time=0.8)
            t.wait_until("A KV entry")
            self.play(FadeIn(why[0], shift=UP * 0.1), run_time=0.6)
            t.wait_until("The state is overwritten")
            self.play(FadeIn(why[1], shift=UP * 0.1), run_time=0.6)
            t.wait_until("In the pool's")
            self.play(FadeIn(eq), run_time=0.7)
            t.wait_until("At a batch of 64")
            self.play(FadeIn(b1), FadeIn(l1), run_time=0.5)
            self.play(LaggedStart(*[FadeIn(s) for s in s1], lag_ratio=0.03), run_time=1.2)
            self.play(FadeIn(rest), FadeIn(rest_t), run_time=0.5)
            t.wait_until("And the state alone")
            self.play(FadeIn(b2), FadeIn(l2), run_time=0.5)
            self.play(LaggedStart(*[FadeIn(s) for s in s2], lag_ratio=0.015), run_time=1.6)

    # ------------------------------------------------------------------
    # Preemption

    def preemption(self):
        self.slide("preemption", "Swap wins the arithmetic; the reference recomputes",
                   book_section="s-preemption-recompute-or-swap")
        cap = kit.label("Preempting a sequence with 2000 tokens of context", size=th.SIZE_SMALL, color=th.INK_3)
        bars = kit.bar_compare(
            [("Swap out", 13, self.colors[1], "273 MiB at 21 GB/s: about 13 ms"),
             ("Swap back in", 13, self.colors[1], "about 13 ms"),
             ("Recompute prefill", 345, self.color, "107.6 TFLOP: 345 ms at peak")],
            width=4.6, bar_height=0.5, gap=0.3,
        )
        why = kit.bullets(
            [
                r"Scattered blocks make many small transfers.",
                r"The swap-in sits on the critical path.",
                r"Pinned host memory is a global resource.",
                r"Recompute is a few lines; swap is a subsystem.",
            ],
            size=th.SIZE_SMALL, width=11.0, buff=0.18,
        )
        why_t = kit.label("Why the reference recomputes anyway", size=th.SIZE_SMALL, color=self.color,
                          weight=SEMIBOLD)
        col = VGroup(cap, bars, VGroup(why_t, why).arrange(DOWN, buff=0.25, aligned_edge=LEFT)).arrange(
            DOWN, buff=0.55, aligned_edge=LEFT)
        self.place(col)
        with self.narrate(
            "When the pool runs dry, a sequence must give way. "
            "You can swap its blocks to host memory, or drop them and redo the prefill later. "
            "Swap moves 273 MiB, state included. "
            "Over PCIe at 21 GB/s, that's about 13 ms out, and 13 back. "
            "Recompute costs 107.6 TFLOP, which is 345 ms at peak. "
            "On paper, swapping wins by an order of magnitude. "
            "The reference still recomputes. "
            "Scattered blocks make many small transfers, and the swap-in sits on the critical path. "
            "Pinned host memory is a global resource. "
            "And recompute is a few lines, while swap is a subsystem."
        ) as t:
            self.play(FadeIn(cap), FadeIn(bars.labels), run_time=0.7)
            t.wait_until("Over PCIe")
            self.play(GrowFromEdge(bars.bars[0], LEFT), FadeIn(bars.values[0]), run_time=0.6)
            self.play(GrowFromEdge(bars.bars[1], LEFT), FadeIn(bars.values[1]), run_time=0.6)
            t.wait_until("Recompute costs")
            self.play(GrowFromEdge(bars.bars[2], LEFT), FadeIn(bars.values[2]), run_time=1.0)
            t.wait_until("The reference still")
            self.play(FadeIn(why_t), run_time=0.5)
            t.wait_until("Scattered blocks")
            self.play(FadeIn(why[0], shift=RIGHT * 0.1), run_time=0.5)
            self.play(FadeIn(why[1], shift=RIGHT * 0.1), run_time=0.5)
            t.wait_until("Pinned host")
            self.play(FadeIn(why[2], shift=RIGHT * 0.1), run_time=0.5)
            t.wait_until("And recompute")
            self.play(FadeIn(why[3], shift=RIGHT * 0.1), run_time=0.5)

    # ------------------------------------------------------------------
    # What goes wrong

    def what_goes_wrong(self):
        self.slide("what-goes-wrong", "Five bugs that don't crash", book_section="s-what-goes-wrong")
        bugs = kit.bullets(
            [
                r"\textbf{Off by one:} 16 tokens must take one block, not two.",
                r"\textbf{Ignoring the current length:} the pool runs dry in seconds.",
                r"\textbf{Releasing twice:} two sequences write the same block.",
                r"\textbf{Assuming block order:} the free list hands out IDs in any order.",
                r"\textbf{A stale context length:} the newest key is dropped, or garbage is read.",
            ],
            size=th.SIZE_BODY, width=12.6, buff=0.42,
        )
        self.place(bugs)
        with self.narrate(
            "These bugs don't crash, so the lab checks each one. "
            "Exactly 16 tokens must take one block, and 17 must take two. "
            "Ignore the current length, and every decode step reallocates, and exhausts the pool in seconds. "
            "Release a sequence twice, and two sequences write the same block. "
            "Assume block order, and you break the first time blocks are reused. "
            "And pass a stale context length, and the model ignores its newest token."
        ) as t:
            for row, cue in zip(bugs, ["Exactly 16", "Ignore the current", "Release a sequence", "Assume block",
                                       "And pass a stale"]):
                t.wait_until(cue)
                self.play(FadeIn(row, shift=RIGHT * 0.1), run_time=0.6)
