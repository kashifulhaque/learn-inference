"""Chapter 16, continuous batching.

Teaches content/chapters/16-continuous-batching.md in its order. Every number
below was recomputed in Python and matches the chapter (rounded as it rounds).

Four requests, two slots (A, C, D need 2 tokens; B needs 6):
- Static: 8 steps, 2 x 8 = 16 slot-steps, 2 + 6 + 2 + 2 = 12 tokens,
  12 / 16 = 75% useful. C and D start at step 7.
- Continuous: 6 steps, 12 slot-steps, 100% useful.

Head-of-line blocking, batch of 16 (15 stop at 20 tokens, one runs to 2000):
- Useful 15 x 20 + 2000 = 2300; paid 16 x 2000 = 32,000; 2300 / 32,000
  = 0.0719, "7.2%". 2000 - 20 = 1980 steps at batch-1 work.
- Queueing: 2000 steps at about 45 ms ~ 91 s (chapter's figure; 2000 x 45.8 ms
  = 91.6 s, 2000 x 45.5 ms = 91 s).

Flat batch drawn on screen: A chunk 5 tokens + B chunk 3 tokens = P = 8;
4 decodes = D = 4; T = 12. Model widths: hidden 5120, 24 query heads, 4 KV
heads, head dim 256, vocabulary 248,320.

LM head: 2 x 5120 x 248,320 = 2.543e9 = 2.54 GFLOP per token;
x 4096 = 10.42e12 = 10.4 TFLOP; / 312e12 = 33.4 ms, "33 ms";
4096 x 248,320 x 2 B = 2.034e9 = 2.03 GB of logits.

Bytes per step, L = 2048 (weights 53.8 GB, KV 64 KiB per token, state
147.8 MiB):
- KV 2048 x 65,536 = 134,217,728 B = 134.2 MB; state 147.8 x 2^20 =
  154,979,533 B = 155.0 MB; sum 289.2 MB.
- Step time = max(bytes / 1275e9, B x 53.8e9 / 312e12):
  B=1: 54.1 GB, 42.4 ms, 24 tok/s, intensity 1.0
  B=8: 56.1 GB, 44.0 ms, 182 tok/s, 7.7
  B=16: 58.4 GB, 45.8 ms, 349 tok/s, 14.7
  B=32: 63.1 GB, 49.5 ms, 647 tok/s, 27.3
  B=64: 72.3 GB, 56.7 ms, 1128.7 tok/s ("1129"), 47.6
- Throughput ratio 47.87 ("48"); ITL ratio 56.7 / 42.4 = 1.337 ("1.34").
- Per-sequence reads at B = 64: 64 x 289.2 MB = 18.5 GB.
- Ridge: 312e12 / 1935e9 = 161.2; 312e12 / 1275e9 = 244.7 ("245").
- Pool: 128 + 147.8 = 275.8 MiB per sequence; 18,120 / 275.8 = 65.7, "65".

Chunked prefill:
- 4000 x 53.8 GFLOP = 215.2 TFLOP; / 312e12 = 0.690 s, "690 ms".
- One token at peak: 53.8e9 / 312e12 = 0.1724 ms ("0.172").
- Meet: 45.8 / 0.172 = 266.3, so T_p + B = 266, T_p ~ 250.
- Step time max(45.8, 0.172 x (T_p + 16)): 128 -> 45.8 ms; 256 -> 46.8;
  512 -> 90.8 ("91"); 1024 -> 178.9 ("179"). Added compute 128 x 0.172 =
  22 ms, 256 -> 44 ms, 512 -> 88 ms, 1024 -> 176 ms.
- Jitter: 45 ms per token becomes 176 ms when a 1024-token chunk joins
  (chapter's fairness figure, the chunk's added compute).

Preemption: 2000 x 0.1724 ms = 344.9 ms, "345 ms" of recompute; swap
2000 x 64 KiB + 147.8 MiB = 125 + 147.8 = 272.8 MiB, "273 MiB", "about 13 ms
each way" over PCIe (chapter's figure). Retries bounded at max_batch_size + 1.

Admission: int(18,120 x 0.02) = 362 pages held back. A growing sequence asks
for a block every 16 steps (16 tokens per block).
"""

import re

from manim import *

import kit
from kit import theme as th
from kit.narration import split_sentences, spoken

LOOP = """\
step():
  for r in running:
    if r.done: retire r, free blocks
    elif r.prefilling: skip for now
    else: reserve 1 token, decode r
  for r in running still prefilling:
    chunk = min(chunk_size, budget, left)
    if reserve(r, chunk): prefill r
  while waiting and budget and room:
    head = waiting[0]    # no skipping
    if not reserve(head, chunk): break
    admit head, prefill a chunk
  if batch empty and running:
    preempt newest, retry
"""

# Spoken fixes for words the shared rules don't cover.
FIXES = [(r"\bLM head\b", "L M head"), (r"\bp99\b", "P ninety-nine")]


def voiced(text):
    """The spoken form of `text`, with this lecture's fixes, one sentence each."""
    out = []
    for s in split_sentences(text):
        for pat, rep in FIXES:
            s = re.sub(pat, rep, s)
        v = spoken(s)
        out.append(v[:1].upper() + v[1:])
    return " ".join(out)


def cell(w, h, color=None, text=None, tint=0.35, size=th.SIZE_SMALL, stroke=th.RULE_STRONG):
    """A rectangle tinted with `color`, with an optional centred label."""
    fill = th.RAISE if color is None else th.tint(color, tint)
    sq = Rectangle(width=w, height=h, stroke_color=stroke, stroke_width=th.STROKE_HAIRLINE,
                   fill_color=fill, fill_opacity=1)
    g = VGroup(sq)
    g.square, g.text = sq, None
    if text:
        t = kit.label(text, size=size, color=th.INK, weight=MEDIUM).move_to(sq)
        t.set_z_index(2)
        g.add(t)
        g.text = t
    return g


def chip(text, color, w=0.62, h=0.52, size=th.SIZE_SMALL, tint=0.3):
    """A request: a rounded tile with a letter."""
    r = RoundedRectangle(width=w, height=h, corner_radius=0.08, stroke_color=color, stroke_width=2,
                         fill_color=th.tint(color, tint), fill_opacity=1)
    t = kit.label(text, size=size, color=th.INK, weight=MEDIUM).move_to(r)
    g = VGroup(r, t)
    g.rect, g.text = r, t
    return g


def panel(title, w, h, color=th.INK_3):
    """A white card with a small heading in its top-left corner."""
    r = RoundedRectangle(width=w, height=h, corner_radius=0.12, stroke_color=th.RULE_STRONG, stroke_width=2,
                         fill_color=th.RAISE, fill_opacity=1)
    head = kit.label(title, size=th.SIZE_LABEL, color=color, weight=SEMIBOLD)
    head.next_to(r.get_corner(UL), DR, buff=0.16)
    g = VGroup(r, head)
    g.rect, g.head = r, head
    return g


def swatch(text, color, size=th.SIZE_LABEL):
    sq = Square(0.26, stroke_color=th.RULE_STRONG, stroke_width=th.STROKE_HAIRLINE,
                fill_color=th.tint(color, 0.35), fill_opacity=1)
    return VGroup(sq, kit.label(text, size=size, color=th.INK_2)).arrange(RIGHT, buff=0.16)


class Lecture(kit.Lecture):
    def construct(self):
        self.req = {"A": self.color, "B": th.INFO, "C": th.GOOD, "D": th.VIOLET}
        self.title_slide(
            narration=(
                "Continuous batching. "
                "Every decode step reads all 53.8 GB of weights, however many sequences it serves. "
                "So the real question is which sequences ride along, and when. "
                "In this lecture, you build the scheduler that decides, before every forward pass."
            )
        )
        self.lifecycle()
        self.static_four()
        self.continuous_four()
        self.head_of_line()
        self.per_iteration()
        self.loop_code()
        self.flat_batch()
        self.lm_head()
        self.decode_first()
        self.bytes_per_step()
        self.throughput_latency()
        self.no_ridge()
        self.prefill_stall()
        self.chunk_size()
        self.mixing()
        self.preemption()
        self.recompute()
        self.admission()
        self.fairness()
        self.closing_slide(
            [
                r"Re-plan every step, not every batch.",
                r"Decode first. Batch 64: $48\times$ throughput, $1.34\times$ ITL.",
                r"About 250 prefill tokens per step ride free.",
                r"Evict the newest to the queue's front; keep a watermark.",
            ],
            narration=(
                "Here's what to keep. "
                "Continuous batching re-plans every step, so finished requests leave and waiting ones join. "
                "Schedule decodes first, because finishing running work frees blocks. "
                "Batch 64 buys 48 times the throughput for 1.34 times the token gap. "
                "Chunked prefill fills the idle tensor cores, and about 250 prompt tokens per step come free. "
                "When memory runs out, evict the newest sequence to the front of the queue, and keep a watermark."
            ),
            book_section="s-what-goes-wrong",
        )

    # Before you start -------------------------------------------------------------

    def lifecycle(self):
        self.slide("lifecycle", "A request waits, prefills, decodes, and retires",
                   book_section="s-before-you-start")
        h = 0.75
        wait = cell(2.5, h, th.INK_4, "waiting", tint=0.2)
        pre = cell(2.4, h, self.color, "prefill")
        decs = VGroup(*[cell(0.72, h, th.INFO, None) for _ in range(6)]).arrange(RIGHT, buff=0.1)
        ret = cell(1.6, h, th.GOOD, "retire", tint=0.25)
        lane = VGroup(wait, pre, decs, ret).arrange(RIGHT, buff=0.1)
        arrive = VGroup(
            Triangle(fill_color=th.INK, fill_opacity=1, stroke_width=0).scale(0.11),
            kit.label("arrives", size=th.SIZE_LABEL, color=th.INK_3),
        ).arrange(DOWN, buff=0.08)
        arrive.next_to(wait, DOWN, buff=0.08).align_to(wait, LEFT).shift(LEFT * 0.12)
        dec_lab = kit.label("decode: one token per step", size=th.SIZE_LABEL, color=th.INFO)
        dec_lab.next_to(decs, DOWN, buff=0.15).align_to(decs, LEFT)
        ttft = BraceBetweenPoints(wait.get_corner(UL) + UP * 0.05, pre.get_corner(UR) + UP * 0.05,
                                  direction=UP, color=th.INK_4)
        ttft_t = kit.label("time to first token (TTFT)", size=th.SIZE_LABEL, color=th.INK_2)
        ttft_t.next_to(ttft, UP, buff=0.1)
        itl = BraceBetweenPoints(decs[2].get_corner(UL) + UP * 0.05, decs[3].get_corner(UL) + UP * 0.05,
                                 direction=UP, color=th.INK_4)
        itl_t = kit.label("inter-token latency (ITL)", size=th.SIZE_LABEL, color=th.INK_2)
        itl_t.next_to(itl, UP, buff=0.1).align_to(itl, LEFT).shift(LEFT * 0.3)
        blocks = kit.label("frees its blocks", size=th.SIZE_LABEL, color=th.GOOD)
        blocks.next_to(ret, UP, buff=0.15).align_to(ret, RIGHT)
        fig = VGroup(lane, arrive, dec_lab, ttft, ttft_t, itl, itl_t, blocks)
        step = kit.prose(r"\textbf{Step}: one forward pass over a set of tokens. The scheduler picks the set "
                         r"before every pass.", size=th.SIZE_SMALL, width=11.0)
        feel = kit.prose(r"Users feel TTFT as sluggishness, and ITL as stuttering.", size=th.SIZE_SMALL,
                         width=None, color=th.INK_3)
        col = VGroup(step, fig, feel).arrange(DOWN, buff=0.7)
        self.place(col)
        with self.narrate(
            "A step is one forward pass over a set of tokens, and the scheduler picks that set before every pass. "
            "A request arrives, and waits in a queue. "
            "Once it's admitted, it prefills: the whole prompt runs through the model and fills its cache. "
            "Then it decodes, one token per step, until it ends. "
            "Finally, it retires and returns its cache blocks. "
            "Time to first token runs from arrival to the first streamed token, so queueing and prefill "
            "dominate it. "
            "Inter-token latency is the gap between later tokens, and it equals one scheduler step. "
            "Users feel the first as sluggishness, and the second as stuttering."
        ) as t:
            self.play(FadeIn(step), run_time=0.8)
            t.wait_until("A request arrives")
            self.play(FadeIn(arrive), FadeIn(wait, shift=RIGHT * 0.1), run_time=0.8)
            t.wait_until("Once it's admitted")
            self.play(FadeIn(pre, shift=RIGHT * 0.1), run_time=0.8)
            t.wait_until("Then it decodes")
            self.play(LaggedStart(*[FadeIn(d, shift=RIGHT * 0.08) for d in decs], lag_ratio=0.25),
                      FadeIn(dec_lab), run_time=1.6)
            t.wait_until("Finally, it retires")
            self.play(FadeIn(ret, shift=RIGHT * 0.1), FadeIn(blocks), run_time=0.8)
            t.wait_until("Time to first token")
            self.play(FadeIn(ttft), FadeIn(ttft_t), run_time=0.8)
            t.wait_until("Inter-token latency is")
            self.play(FadeIn(itl), FadeIn(itl_t), run_time=0.8)
            t.wait_until("Users feel")
            self.play(FadeIn(feel), run_time=0.6)

    # Four requests, two ways ----------------------------------------------------------

    def schedule_grid(self, cols, w=0.86, h=0.6, labels=True):
        """Columns of two slots; `cols[i]` is (slot 1, slot 2), None for idle."""
        columns = VGroup()
        for pair in cols:
            c = VGroup(*[cell(w, h, self.req[x] if x else None, x) for x in pair]).arrange(DOWN, buff=0.06)
            for sq in c:
                if sq.text is None:
                    sq.square.set_fill(th.tint(th.INK_4, 0.08))
            columns.add(c)
        columns.arrange(RIGHT, buff=0.06)
        steps = VGroup(*[kit.label(str(i + 1), size=th.SIZE_LABEL, color=th.INK_3).next_to(c, UP, buff=0.1)
                         for i, c in enumerate(columns)])
        g = VGroup(columns, steps)
        g.columns, g.steps = columns, steps
        if labels:
            rows = VGroup(*[kit.label(f"slot {i + 1}", size=th.SIZE_LABEL, color=th.INK_3)
                            .next_to(columns[0][i], LEFT, buff=0.2) for i in range(2)])
            g.add(rows)
            g.rows = rows
        return g

    def static_four(self):
        self.slide("static-four", "Static batching holds a slot open for B",
                   book_section="s-four-requests-two-ways")
        need = VGroup(*[
            VGroup(chip(x, self.req[x]), kit.label(f"{n} tokens", size=th.SIZE_LABEL, color=th.INK_3))
            .arrange(RIGHT, buff=0.15)
            for x, n in (("A", 2), ("B", 6), ("C", 2), ("D", 2))
        ]).arrange(RIGHT, buff=0.6)
        sched = [("A", "B"), ("A", "B"), (None, "B"), (None, "B"), (None, "B"), (None, "B"), ("C", "D"),
                 ("C", "D")]
        g = self.schedule_grid(sched, w=1.0, h=0.7)
        steps_t = kit.label("step", size=th.SIZE_LABEL, color=th.INK_3).next_to(g.steps, LEFT, buff=0.3)
        steps_t.align_to(g.rows, RIGHT)
        idle = VGroup(*[g.columns[i][0] for i in range(2, 6)])
        result = kit.prose(r"8 steps $\cdot$ $2 \times 8 = 16$ slot-steps paid $\cdot$ 12 tokens made "
                           r"$\cdot$ \textbf{75\% useful}", size=th.SIZE_SMALL, width=None)
        col = VGroup(need, VGroup(g, steps_t), result).arrange(DOWN, buff=0.75)
        self.place(col)
        idle_t = kit.label("idle: A is done", size=th.SIZE_LABEL, color=th.BAD).next_to(g.steps, UP, buff=0.22)
        idle_t.set_x(idle.get_x())
        with self.narrate(
            "Take a GPU with room for two sequences, and four requests that arrive together. "
            "A, C, and D each need 2 output tokens. "
            "B needs 6. "
            "Static batching runs A and B until both finish. "
            "A is done after step 2, but its slot sits empty while B keeps going. "
            "Only at step 7 do C and D get a turn. "
            "That's 8 steps, paying for 16 slot-steps to make 12 tokens. "
            "So only 75 percent of the slots do useful work."
        ) as t:
            self.play(FadeIn(need[0]), FadeIn(need[1]), FadeIn(need[2]), FadeIn(need[3]), run_time=0.8)
            t.wait_until("Static batching runs")
            self.play(FadeIn(g.rows), FadeIn(steps_t), run_time=0.5)
            for i in range(2):
                self.play(FadeIn(g.columns[i]), FadeIn(g.steps[i]), run_time=0.45)
            t.wait_until("A is done")
            for i in range(2, 6):
                self.play(FadeIn(g.columns[i]), FadeIn(g.steps[i]), run_time=0.4)
            self.play(FadeIn(idle_t), *[sq.square.animate.set_stroke(th.BAD, width=2.5) for sq in idle],
                      run_time=0.6)
            t.wait_until("Only at step 7")
            for i in range(6, 8):
                self.play(FadeIn(g.columns[i]), FadeIn(g.steps[i]), run_time=0.45)
            t.wait_until("That's 8 steps")
            self.play(FadeIn(result, shift=UP * 0.1), run_time=0.8)

    def continuous_four(self):
        self.slide("continuous-four", "Continuous batching refills a slot the next step",
                   book_section="s-four-requests-two-ways")
        st = [("A", "B"), ("A", "B"), (None, "B"), (None, "B"), (None, "B"), (None, "B"), ("C", "D"),
              ("C", "D")]
        ct = [("A", "B"), ("A", "B"), ("C", "B"), ("C", "B"), ("D", "B"), ("D", "B")]
        gs = self.schedule_grid(st, w=0.86, h=0.56)
        gc = self.schedule_grid(ct, w=0.86, h=0.56)
        ts = kit.label("Static", size=th.SIZE_SMALL, color=th.INK_2, weight=SEMIBOLD)
        tc = kit.label("Continuous", size=th.SIZE_SMALL, color=self.color, weight=SEMIBOLD)
        rs = kit.stack(kit.label("8 steps, 16 slot-steps", size=th.SIZE_SMALL, color=th.INK_3),
                       kit.label("75% useful", size=th.SIZE_SMALL, color=th.INK_3), buff=0.12)
        rc = kit.stack(kit.label("6 steps, 12 slot-steps", size=th.SIZE_SMALL, color=th.INK_2),
                       kit.label("100% useful", size=th.SIZE_SMALL, color=self.color, weight=SEMIBOLD), buff=0.12)
        top = VGroup(VGroup(ts, gs).arrange(DOWN, buff=0.2, aligned_edge=LEFT), rs).arrange(RIGHT, buff=0.6)
        bot = VGroup(VGroup(tc, gc).arrange(DOWN, buff=0.2, aligned_edge=LEFT), rc).arrange(RIGHT, buff=0.6)
        rc.align_to(rs, LEFT)
        bot[0].align_to(top[0], LEFT)
        key = kit.pill("Re-plan every step, not every batch", self.color, size=th.SIZE_SMALL)
        col = VGroup(top, bot, key).arrange(DOWN, buff=0.55)
        bot[0].align_to(top[0], LEFT)
        rc.align_to(rs, LEFT)
        self.place(col)
        with self.narrate(
            "Continuous batching checks after every step instead. "
            "When A finishes, C takes its slot on step 3. "
            "When C finishes, D takes it on step 5. "
            "D and B both finish on step 6. "
            "That's 6 steps instead of 8. "
            "It pays for 12 slot-steps, and every one of them is useful. "
            "Nothing waits for the slowest member of a batch."
        ) as t:
            self.play(FadeIn(top), FadeIn(tc), FadeIn(gc.rows), run_time=0.8)
            self.play(FadeIn(gc.columns[0]), FadeIn(gc.steps[0]), FadeIn(gc.columns[1]), FadeIn(gc.steps[1]),
                      run_time=0.6)
            t.wait_until("When A finishes")
            self.play(FadeIn(gc.columns[2], shift=DOWN * 0.1), FadeIn(gc.steps[2]), run_time=0.5)
            self.play(FadeIn(gc.columns[3]), FadeIn(gc.steps[3]), run_time=0.4)
            t.wait_until("When C finishes")
            self.play(FadeIn(gc.columns[4], shift=DOWN * 0.1), FadeIn(gc.steps[4]), run_time=0.5)
            t.wait_until("D and B both")
            self.play(FadeIn(gc.columns[5]), FadeIn(gc.steps[5]), run_time=0.5)
            t.wait_until("That's 6 steps")
            self.play(FadeIn(rc), run_time=0.6)
            t.wait_until("Nothing waits")
            self.play(FadeIn(key, scale=0.95), run_time=0.6)

    # What static batching wastes -------------------------------------------------------

    def head_of_line(self):
        self.slide("head-of-line", "One long request leaves 16 slots 7.2% busy",
                   book_section="s-what-static-batching-wastes")
        W, rh = 6.6, 0.2
        rows = VGroup()
        useful = VGroup()
        for i in range(16):
            n = 2000 if i == 15 else 20
            bg = Rectangle(width=W, height=rh, stroke_width=0, fill_color=th.tint(th.INK_4, 0.14), fill_opacity=1)
            fg = Rectangle(width=max(0.03, W * n / 2000), height=rh, stroke_width=0,
                           fill_color=self.color if i < 15 else th.INFO, fill_opacity=0.9)
            fg.align_to(bg, LEFT)
            rows.add(bg)
            useful.add(fg)
            VGroup(bg, fg).shift(DOWN * i * (rh + 0.07))
        ax_l = kit.label("step 0", size=th.SIZE_LABEL, color=th.INK_3).next_to(rows[-1], DOWN, buff=0.15)
        ax_l.align_to(rows, LEFT)
        ax_r = kit.label("step 2000", size=th.SIZE_LABEL, color=th.INK_3).next_to(rows[-1], DOWN, buff=0.15)
        ax_r.align_to(rows, RIGHT)
        short_t = kit.label("15 requests stop at 20", size=th.SIZE_LABEL, color=self.color)
        short_t.next_to(rows[0], UP, buff=0.15).align_to(rows, LEFT)
        long_t = kit.label("1 runs to 2000", size=th.SIZE_LABEL, color=th.INFO)
        long_t.next_to(rows[0], UP, buff=0.15).align_to(rows, RIGHT).align_to(short_t, UP)
        chart = VGroup(rows, useful, ax_l, ax_r, short_t, long_t)
        l1 = kit.formula(r"\text{useful} = 15 \times 20 + 2000 = 2300", size=th.SIZE_MATH_SMALL)
        l2 = kit.formula(r"\text{paid} = 16 \times 2000 = 32{,}000", size=th.SIZE_MATH_SMALL)
        l3 = kit.formula(r"2300 / 32{,}000 = 7.2\%", size=th.SIZE_MATH_SMALL, color=self.color)
        maths = VGroup(l1, l2, l3).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        late = kit.callout("warning", r"A request arriving one step late waits for all 2000 steps: about 91 s.",
                           title="Queueing outside the batch", width=5.4, size=th.SIZE_SMALL)
        side = VGroup(maths, late).arrange(DOWN, buff=0.55, aligned_edge=LEFT)
        layout = VGroup(chart, side).arrange(RIGHT, buff=0.6)
        self.place(layout)
        with self.narrate(
            "Now use realistic lengths, which in chat traffic span two orders of magnitude. "
            "Take a batch of 16, where fifteen requests stop at 20 tokens. "
            "One request runs to 2000. "
            "The useful work is 15 times 20, plus 2000: that's 2300 token-steps. "
            "The batch pays for 16 times 2000, which is 32,000 slot-steps. "
            "So the GPU does useful work only 7.2 percent of the time. "
            "It's worse outside the batch. "
            "A request that arrives one step late waits for the whole batch to drain, about 91 seconds of nothing. "
            "No batch size fixes this."
        ) as t:
            self.play(FadeIn(rows), FadeIn(ax_l), FadeIn(ax_r), run_time=0.8)
            t.wait_until("Take a batch of 16")
            self.play(LaggedStart(*[GrowFromEdge(u, LEFT) for u in useful[:15]], lag_ratio=0.05),
                      FadeIn(short_t), run_time=1.2)
            t.wait_until("One request runs")
            self.play(GrowFromEdge(useful[15], LEFT), FadeIn(long_t), run_time=2.0)
            t.wait_until("The useful work")
            self.play(FadeIn(l1), run_time=0.7)
            t.wait_until("The batch pays")
            self.play(FadeIn(l2), Indicate(rows, color=th.INK_4, scale_factor=1.01), run_time=0.9)
            t.wait_until("So the GPU")
            self.play(FadeIn(l3), run_time=0.7)
            t.wait_until("A request that arrives")
            self.play(FadeIn(late, shift=UP * 0.1), run_time=0.8)

    # Scheduling per iteration ---------------------------------------------------------

    def per_iteration(self):
        self.slide("per-iteration", "Before every pass: retire, decode, prefill, admit",
                   book_section="s-scheduling-per-iteration")
        pw = panel("waiting: no blocks yet", 4.0, 2.1)
        pr = panel("running: holding blocks", 5.0, 2.1, color=self.color)
        pf = panel("finished", 2.4, 2.1, color=th.GOOD)
        panels = VGroup(pw, pr, pf).arrange(RIGHT, buff=0.3)
        names = ["1  Retire", "2  Decode", "3  Continue prefills", "4  Admit"]
        steps = VGroup(*[kit.pill(n, th.INK_4, size=th.SIZE_BODY, filled=False) for n in names])
        steps.arrange(RIGHT, buff=0.4)
        desc = kit.prose(r"A waiting request joins the \textit{next step}, not the next batch.", size=th.SIZE_SMALL, width=None, color=th.INK_3)
        col = VGroup(panels, steps, desc).arrange(DOWN, buff=0.7)
        self.place(col)

        def slot(p, i, n):
            r = p.rect
            x0 = r.get_left()[0] + 0.55
            span = r.width - 1.1
            return np.array([x0 + span * i / max(n - 1, 1), r.get_center()[1] - 0.2, 0])

        run = {x: chip(x, c, w=0.75, h=0.65) for x, c in (("A", self.color), ("B", th.INFO), ("C", th.GOOD), ("D", th.VIOLET))}
        for i, x in enumerate("ABCD"):
            run[x].move_to(slot(pr, i, 4))
        wait = {x: chip(x, th.INK_4, w=0.75, h=0.65) for x in "GFE"}
        for i, x in enumerate("GFE"):
            wait[x].move_to(slot(pw, i + 1, 4))
        done_a = kit.label("done", size=th.SIZE_LABEL, color=th.GOOD).next_to(run["A"], DOWN, buff=0.08)
        pref_c = kit.label("prefill", size=th.SIZE_LABEL, color=th.WARN).next_to(run["C"], DOWN, buff=0.08)
        head_t = kit.label("head", size=th.SIZE_LABEL, color=th.INK_3).next_to(wait["E"], DOWN, buff=0.08)

        def on(i):
            return AnimationGroup(steps[i][0].animate.set_fill(th.tint(self.color, 0.15), opacity=1)
                                  .set_stroke(self.color), steps[i][1].animate.set_color(th.INK))

        with self.narrate(
            "Continuous batching, also called iteration-level scheduling, runs the scheduler before every "
            "forward pass. "
            "Its state is three collections: waiting, running, and finished. "
            "Each step does four things, in order. "
            "First, it retires requests that finished last step, releasing their blocks. "
            "Second, it schedules one decode token for every running request past its prefill. "
            "Third, it continues prefills that still have prompt left. "
            "Fourth, it admits waiting requests from the head of the queue, while the budgets allow. "
            "So a waiting request joins the next step, not the next batch."
        ) as t:
            t.wait_until("Its state is")
            self.play(FadeIn(panels), *[FadeIn(c) for c in run.values()], *[FadeIn(c) for c in wait.values()],
                      FadeIn(done_a), FadeIn(pref_c), FadeIn(head_t), run_time=1.0)
            t.wait_until("Each step does")
            self.play(FadeIn(steps), run_time=0.7)
            t.wait_until("First, it retires")
            self.play(on(0), run_time=0.4)
            self.play(run["A"].animate.move_to(slot(pf, 0, 1)), FadeOut(done_a), run_time=1.0)
            t.wait_until("Second, it schedules")
            self.play(on(1), run_time=0.4)
            self.play(Indicate(run["B"], color=th.INFO), Indicate(run["D"], color=th.VIOLET), run_time=1.0)
            t.wait_until("Third, it continues")
            self.play(on(2), run_time=0.4)
            self.play(Indicate(run["C"], color=th.GOOD), Indicate(pref_c, color=th.WARN), run_time=1.0)
            t.wait_until("Fourth, it admits")
            self.play(on(3), run_time=0.4)
            self.play(wait["E"].animate.move_to(slot(pr, 0, 4)), FadeOut(head_t), run_time=1.0)
            head2 = kit.label("head", size=th.SIZE_LABEL, color=th.INK_3).next_to(wait["F"], DOWN, buff=0.08)
            self.play(wait["E"][0].animate.set_fill(th.tint(th.HL_D, 0.3)).set_stroke(th.HL_D),
                      FadeIn(head2), run_time=0.6)
            t.wait_until("So a waiting request")
            self.play(FadeIn(desc), run_time=0.6)

    def loop_code(self):
        self.slide("loop-code", "The loop, with three limits on every pass",
                   book_section="s-scheduling-per-iteration")
        panel_ = kit.code(LOOP, "text")
        lims = VGroup(
            kit.box("Token budget", sublabel="tokens in one pass", color=self.color, width=3.3),
            kit.box("Batch-size cap", sublabel="concurrent sequences", color=th.INFO, width=3.3),
            kit.box("Block pool", sublabel="caps both, underneath", color=th.GOOD, width=3.3),
        ).arrange(DOWN, buff=0.35)
        layout = VGroup(panel_, lims).arrange(RIGHT, buff=0.5)
        self.place(layout)
        with self.narrate(
            "Here's the loop as pseudocode. "
            "The first pass walks the running list. "
            "Finished requests retire, and the rest each reserve one token for decode. "
            "The second pass gives each unfinished prompt a chunk, capped by the chunk size and the budget. "
            "Then admission takes requests from the head of the queue, and stops at the first one that doesn't fit. "
            "If the batch comes out empty while sequences are running, the scheduler preempts the newest one. "
            "Three limits bound every pass. "
            "A token budget caps the tokens in one forward pass, so prefill can't starve decode. "
            "A batch-size cap limits concurrent sequences. "
            "And the block pool caps everything underneath both."
        ) as t:
            self.play(FadeIn(panel_.frame), *panel_.reveal(1), run_time=0.7)
            t.wait_until("The first pass")
            self.play(*panel_.reveal(2, 3, 4, 5), run_time=0.8)
            t.wait_until("Finished requests retire")
            self.play(panel_.highlight(3, 5), run_time=0.6)
            t.wait_until("The second pass")
            self.play(*panel_.reveal(6, 7, 8), panel_.highlight(7), run_time=0.8)
            t.wait_until("Then admission takes")
            self.play(*panel_.reveal(9, 10, 11, 12), panel_.highlight(10, 11), run_time=0.8)
            t.wait_until("If the batch comes out")
            self.play(*panel_.reveal(13, 14), panel_.highlight(14), run_time=0.8)
            t.wait_until("A token budget")
            self.play(FadeIn(lims[0], shift=LEFT * 0.1), run_time=0.6)
            t.wait_until("A batch-size cap")
            self.play(FadeIn(lims[1], shift=LEFT * 0.1), run_time=0.6)
            t.wait_until("And the block pool")
            self.play(FadeIn(lims[2], shift=LEFT * 0.1), run_time=0.6)

    def flat_batch(self):
        self.slide("flat-batch", "Every token goes in one flat run, with no padding",
                   book_section="s-scheduling-per-iteration")
        w, h = 0.55, 0.5
        spec = [("A", 5, self.color, "prefill chunk"), ("B", 3, th.HL_C, "prefill chunk"),
                ("C", 1, th.INFO, "decode"), ("D", 1, th.INFO, "decode"), ("E", 1, th.INFO, "decode"),
                ("F", 1, th.INFO, "decode")]
        ragged = VGroup()
        rnames = VGroup()
        for name, n, col, kind in spec:
            row = VGroup(*[cell(w, h, col) for _ in range(n)]).arrange(RIGHT, buff=0.04)
            ragged.add(row)
            rnames.add(kit.label(f"{name}  {kind}", size=th.SIZE_LABEL, color=th.INK_3))
        ragged.arrange(DOWN, buff=0.08, aligned_edge=LEFT)
        for nm, row in zip(rnames, ragged):
            nm.next_to(row, LEFT, buff=0.3)
        rnames.align_to(rnames[0], RIGHT)
        for nm, row in zip(rnames, ragged):
            nm.next_to(ragged, LEFT, buff=0.3).set_y(row.get_y())
        rag = VGroup(rnames, ragged).move_to([0, -1.25, 0])
        flat = VGroup(*[cell(w, h, col) for _, n, col, _ in spec for _ in range(n)]).arrange(RIGHT, buff=0.04)
        flat.move_to([-1.2, 1.0, 0])
        bounds = [0, 5, 8]
        ticks = VGroup()
        for b in bounds + [12]:
            x = flat[b].get_left()[0] - 0.02 if b < 12 else flat[11].get_right()[0] + 0.02
            if b in (0, 5, 8):
                ticks.add(Line([x, flat.get_top()[1] + 0.05, 0], [x, flat.get_top()[1] + 0.32, 0],
                               color=th.INK, stroke_width=3))
        cu = kit.label("sequence boundaries, in a side table", size=th.SIZE_LABEL, color=th.INK_2)
        cu.next_to(ticks, UP, buff=0.12).align_to(flat, LEFT)
        bp = Brace(VGroup(*flat[:8]), DOWN, buff=0.08, color=th.INK_4)
        bpt = kit.prose(r"$P = 8$ prefill tokens", size=th.SIZE_LABEL, width=None).next_to(bp, DOWN, buff=0.08)
        bd = Brace(VGroup(*flat[8:]), DOWN, buff=0.08, color=th.INK_4)
        bdt = kit.prose(r"$D = 4$ decodes", size=th.SIZE_LABEL, width=None).next_to(bd, DOWN, buff=0.08)
        tt = kit.prose(r"$T = P + D = 12$", size=th.SIZE_SMALL, width=None).next_to(flat, RIGHT, buff=0.4)
        tab = kit.table(
            [
                ["hidden states", r"$(T, 5120)$"],
                ["$Q$, full-attention layer", r"$(T, 24, 256)$"],
                ["block tables", r"$(D, \text{max blocks})$"],
            ],
            header=["Tensor", "Shape"], size=th.SIZE_SMALL, row_gap=0.2,
        )
        tab.move_to([-0.6, -2.05, 0])
        with self.narrate(
            "The scheduler returns a plan, and the plan fixes the tensor shapes. "
            "Each sequence brings a prefill chunk, or one decode token. "
            "Rather than pad them into a rectangle, the engine concatenates every token into one flat run. "
            "P counts the prefill tokens, D counts the decodes, and T is their sum. "
            "Side tables record where each sequence starts and ends. "
            "So the hidden states are T by 5120, with no padding. "
            "There's no batch dimension anywhere except the block tables. "
            "That's what makes ragged batches free."
        ) as t:
            self.play(FadeIn(rag), run_time=0.9)
            t.wait_until("Rather than pad")
            src = [c for row in ragged for c in row]
            self.play(*[ReplacementTransform(s, d) for s, d in zip(src, flat)], FadeOut(rnames), run_time=1.6)
            t.wait_until("P counts")
            self.play(FadeIn(bp), FadeIn(bpt), FadeIn(bd), FadeIn(bdt), run_time=0.7)
            self.play(FadeIn(tt), run_time=0.5)
            t.wait_until("Side tables")
            self.play(Create(ticks), FadeIn(cu), run_time=0.7)
            t.wait_until("So the hidden states")
            self.play(FadeIn(tab.rules), *[FadeIn(c) for r in tab.cells[:3] for c in r], run_time=0.9)
            t.wait_until("There's no batch")
            self.play(FadeIn(VGroup(*tab.cells[3])), run_time=0.6)

    def lm_head(self):
        self.slide("lm-head", "Slice before the LM head", book_section="s-scheduling-per-iteration")
        w, h = 0.55, 0.5
        cols = [self.color] * 5 + [th.HL_C] * 3 + [th.INFO] * 4
        flat = VGroup(*[cell(w, h, c) for c in cols]).arrange(RIGHT, buff=0.04)
        sampled = [7, 8, 9, 10, 11]
        head = kit.box("LM head: 5120 to 248,320", color=th.HL_D, width=6.9)
        fig = VGroup(flat, head).arrange(DOWN, buff=1.0)
        note = kit.label("B's chunk completes its prompt; A's doesn't", size=th.SIZE_LABEL, color=th.INK_3)
        note.next_to(flat, UP, buff=0.2)
        fig.add(note)
        cost = kit.formula(r"2 \times 5120 \times 248{,}320 = 2.54\ \text{GFLOP per token}",
                           size=th.SIZE_MATH_SMALL)
        warn = kit.callout(
            "warning",
            r"For every token at $T = 4096$: 10.4 TFLOP, 33 ms at peak, and 2.03 GB of logits to hold.",
            title="Unsliced", width=9.6, size=th.SIZE_SMALL)
        col = VGroup(fig, cost, warn).arrange(DOWN, buff=0.5)
        self.place(col)
        arrows = VGroup(*[Arrow(flat[i].get_bottom(), [flat[i].get_x(), head.get_top()[1], 0],
                                buff=0.08, color=th.HL_D, stroke_width=3, tip_length=0.15)
                          for i in sampled])
        dim = VGroup(*[flat[i] for i in range(12) if i not in sampled])
        text = (
            "The row people get wrong is the logits. "
            "The LM head projects each hidden state onto all 248,320 vocabulary tokens. "
            "You need it only for the last token of each decode sequence. "
            "You also need it for a prefill chunk's last token, but only when that chunk completes its prompt. "
            "Computing it for every token costs 2.54 GFLOP per token. "
            "At 4096 tokens in a step, that's 10.4 TFLOP, or 33 ms at peak. "
            "Slicing isn't an optimization; it's a requirement."
        )
        with self.narrate(text, say=voiced(text)) as t:
            self.play(FadeIn(flat), run_time=0.6)
            t.wait_until("The LM head")
            self.play(FadeIn(head, shift=UP * 0.1), run_time=0.7)
            t.wait_until("You need it only")
            self.play(LaggedStart(*[GrowArrow(a) for a in arrows[1:]], lag_ratio=0.2), run_time=1.0)
            t.wait_until("You also need it")
            self.play(GrowArrow(arrows[0]), FadeIn(note), dim.animate.set_opacity(0.35), run_time=1.0)
            t.wait_until("Computing it for every")
            self.play(FadeIn(cost), run_time=0.7)
            t.wait_until("At 4096 tokens")
            self.play(FadeIn(warn, shift=UP * 0.1), run_time=0.8)

    # Decode before prefill ------------------------------------------------------------

    def pool_grid(self, owners, rows=3, cols=8):
        g = VGroup()
        for i in range(rows * cols):
            o = owners[i] if i < len(owners) else None
            g.add(cell(0.5, 0.5, o, None, tint=0.45))
        g.arrange_in_grid(rows, cols, buff=0.05)
        return g

    def decode_first(self):
        self.slide("decode-first", "Decode first, so cache pressure falls",
                   book_section="s-decode-before-prefill")
        own = [self.color] * 6 + [th.INFO] * 5 + [th.GOOD] * 5 + [th.VIOLET] * 4
        left = self.pool_grid(own)
        right = self.pool_grid(own)
        lt = kit.label("Decode first", size=th.SIZE_SMALL, color=th.GOOD, weight=SEMIBOLD)
        rt = kit.label("Admit first", size=th.SIZE_SMALL, color=th.BAD, weight=SEMIBOLD)
        free_l = kit.label("4 blocks free", size=th.SIZE_LABEL, color=th.INK_3)
        free_r = kit.label("4 blocks free", size=th.SIZE_LABEL, color=th.INK_3)
        lcol = VGroup(lt, left, free_l).arrange(DOWN, buff=0.25)
        rcol = VGroup(rt, right, free_r).arrange(DOWN, buff=0.25)
        newc = chip("N", th.HL_D)
        figs = VGroup(lcol, rcol).arrange(RIGHT, buff=1.6)
        newc.next_to(right, RIGHT, buff=0.4)
        chain = kit.prose(r"Admit first: pressure rises $\to$ preemption $\to$ recomputation, burning the "
                          r"compute that would have finished the sequences you had.", size=th.SIZE_SMALL, width=11.0)
        col = VGroup(VGroup(figs, newc), chain).arrange(DOWN, buff=0.7)
        self.place(col)
        free_l2 = kit.label("A finishes: 10 blocks free", size=th.SIZE_LABEL, color=th.GOOD).move_to(free_l)
        free_r2 = kit.label("N takes 4: none free", size=th.SIZE_LABEL, color=th.BAD).move_to(free_r)
        with self.narrate(
            "When memory is tight, who goes first? "
            "The scheduler runs decodes first, then prefills, then admissions. "
            "A running sequence has already paid for its prefill, and it's holding blocks. "
            "Finishing it frees those blocks, so cache pressure falls. "
            "Admit a newcomer ahead of it, and the newcomer takes blocks instead. "
            "Pressure rises, which triggers preemption, which triggers recomputation. "
            "That burns the compute that would have finished the sequences you already had. "
            "Decode priority also helps tail latency: sequences near completion get to complete."
        ) as t:
            self.play(FadeIn(lcol), FadeIn(rcol), run_time=0.9)
            t.wait_until("Finishing it frees")
            self.play(*[left[i].square.animate.set_fill(th.RAISE) for i in range(6)],
                      ReplacementTransform(free_l, free_l2), run_time=1.0)
            t.wait_until("Admit a newcomer")
            self.play(FadeIn(newc, shift=LEFT * 0.1), run_time=0.5)
            self.play(*[right[i].square.animate.set_fill(th.tint(th.HL_D, 0.45)) for i in range(20, 24)],
                      ReplacementTransform(free_r, free_r2), run_time=1.0)
            t.wait_until("Pressure rises")
            self.play(FadeIn(chain), Indicate(right, color=th.BAD, scale_factor=1.03), run_time=1.0)

    # Throughput against latency --------------------------------------------------------

    def bytes_per_step(self):
        self.slide("bytes-per-step", "The weight read is shared by the whole batch",
                   book_section="s-throughput-against-latency")
        eq = kit.formula(r"\text{bytes}(B, L) =", r"53.8\ \text{GB}", r"+\ B\,(", r"L \times 64\ \text{KiB}",
                         r"+", r"147.8\ \text{MiB}", r")")
        eq[1].set_color(th.HL_A)
        eq[3].set_color(th.HL_B)
        eq[5].set_color(th.HL_C)
        k = 0.13  # units per GB
        wbar = Rectangle(width=53.8 * k, height=0.7, stroke_width=0, fill_color=th.HL_A, fill_opacity=0.85)
        wlab = kit.label("weights, read once per step", size=th.SIZE_LABEL, color=th.CANVAS, weight=MEDIUM)
        wlab.move_to(wbar)

        def seqs(B):
            kv = Rectangle(width=B * 0.1342 * k, height=0.7, stroke_width=0, fill_color=th.HL_B, fill_opacity=0.85)
            st = Rectangle(width=B * 0.1550 * k, height=0.7, stroke_width=0, fill_color=th.HL_C, fill_opacity=0.85)
            kv.next_to(wbar, RIGHT, buff=0)
            st.next_to(kv, RIGHT, buff=0)
            return VGroup(kv, st)

        totals = {1: "54.1", 8: "56.1", 16: "58.4", 32: "63.1", 64: "72.3"}

        def tag(B, segs):
            return kit.label(f"B = {B}:  {totals[B]} GB", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM) \
                .next_to(segs, RIGHT, buff=0.25)

        bar = VGroup(wbar, wlab)
        def solid(text, color):
            sq = Square(0.26, stroke_width=0, fill_color=color, fill_opacity=0.85)
            return VGroup(sq, kit.label(text, size=th.SIZE_LABEL, color=th.INK_2)).arrange(RIGHT, buff=0.16)

        keyrow = VGroup(solid("KV cache, L × 64 KiB", th.HL_B), solid("recurrent state, 147.8 MiB", th.HL_C))
        keyrow.arrange(RIGHT, buff=0.6)
        per = kit.prose(r"Per sequence at $L = 2048$: $134.2 + 155.0 = 289.2$ MB. "
                        r"FLOPs: $53.8\ \text{GFLOP} \times B$.", size=th.SIZE_SMALL, width=None)
        probe = VGroup(bar, seqs(64), tag(64, seqs(64)))
        col = VGroup(eq, probe, keyrow, per).arrange(DOWN, buff=0.6)
        self.place(col)
        bar_left = wbar.get_left()
        s = {B: seqs(B) for B in totals}
        tags = {B: tag(B, s[B]) for B in totals}
        probe.remove(*probe[1:])
        with self.narrate(
            "Now the central trade: how large a batch do you want? "
            "Think of a bus, where the weight read is the trip. "
            "Each step reads all 53.8 GB of weights once, however many sequences ride along. "
            "Each passenger adds its own KV cache and recurrent state. "
            "At a context of 2048 tokens, that's 134.2 MB of cache, plus 155.0 MB of state. "
            "So each sequence adds about 289 MB per step. "
            "At batch 64, the step reads 72.3 GB, and most of that is still the weights."
        ) as t:
            del bar_left
            self.play(FadeIn(eq), run_time=0.8)
            t.wait_until("Each step reads")
            self.play(GrowFromEdge(wbar, LEFT), FadeIn(wlab), run_time=1.0)
            t.wait_until("Each passenger")
            self.play(FadeIn(keyrow), GrowFromEdge(s[1], LEFT), FadeIn(tags[1]), run_time=0.8)
            t.wait_until("At a context of 2048")
            self.play(FadeIn(per), run_time=0.7)
            t.wait_until("At batch 64")
            prev, ptag = s[1], tags[1]
            for B in (8, 16, 32, 64):
                self.play(ReplacementTransform(prev, s[B]), ReplacementTransform(ptag, tags[B]), run_time=0.5)
                prev, ptag = s[B], tags[B]

    def throughput_latency(self):
        self.slide("throughput-latency", "Batch 64: 48× the throughput, 1.34× the latency",
                   book_section="s-throughput-against-latency")
        Bs = [1, 8, 16, 32, 64]
        tput = [24, 182, 349, 647, 1129]
        itl = [42.4, 44.0, 45.8, 49.5, 56.7]
        tb = kit.bar_compare([(f"B = {b}", v, self.color, f"{v} tok/s") for b, v in zip(Bs, tput)],
                             width=4.0, bar_height=0.42, gap=0.25)
        ib = kit.bar_compare([(f"B = {b}", v, th.INFO, f"{v:.1f} ms") for b, v in zip(Bs, itl)],
                             width=3.4, bar_height=0.42, gap=0.25, max_value=60)
        th_t = kit.label("Throughput", size=th.SIZE_SMALL, color=self.color, weight=SEMIBOLD)
        it_t = kit.label("Step time = each user's ITL", size=th.SIZE_SMALL, color=th.INFO, weight=SEMIBOLD)
        lcol = VGroup(th_t, tb).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        rcol = VGroup(it_t, ib).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        charts = VGroup(lcol, rcol).arrange(RIGHT, buff=0.9, aligned_edge=UP)
        rule = kit.prose(r"Step time $= \max(\text{bytes} / 1275\ \text{GB/s},\ \text{FLOPs} / 312\ "
                         r"\text{TFLOP/s})$. Arithmetic, not a measurement.", size=th.SIZE_SMALL, width=None,
                         color=th.INK_3)
        p1 = kit.pill("throughput × 48", self.color, size=th.SIZE_SMALL)
        p2 = kit.pill("ITL × 1.34", th.INFO, size=th.SIZE_SMALL)
        pills = VGroup(p1, p2).arrange(RIGHT, buff=0.8)
        col = VGroup(rule, charts, pills).arrange(DOWN, buff=0.55)
        self.place(col)
        with self.narrate(
            "To get the step time, divide the bytes by the measured 1275 GB/s, and the FLOPs by 312 TFLOP/s, "
            "and take the larger. "
            "At batch 1, a step takes 42.4 ms, for 24 tokens per second. "
            "At batch 64, it takes 56.7 ms, for 1129 tokens per second. "
            "Throughput rises 48 times. "
            "Each user's inter-token latency rises only 1.34 times. "
            "Every sequence you add rides on weight bytes that were moving anyway."
        ) as t:
            self.play(FadeIn(rule), FadeIn(th_t), FadeIn(it_t), FadeIn(tb.labels), FadeIn(ib.labels), run_time=0.9)
            t.wait_until("At batch 1")
            self.play(GrowFromEdge(tb.bars[0], LEFT), FadeIn(tb.values[0]), GrowFromEdge(ib.bars[0], LEFT),
                      FadeIn(ib.values[0]), run_time=0.8)
            t.wait_until("At batch 64")
            for i in range(1, 5):
                self.play(GrowFromEdge(tb.bars[i], LEFT), FadeIn(tb.values[i]), GrowFromEdge(ib.bars[i], LEFT),
                          FadeIn(ib.values[i]), run_time=0.5)
            t.wait_until("Throughput rises")
            self.play(FadeIn(p1, scale=0.9), run_time=0.6)
            t.wait_until("Each user's inter-token")
            self.play(FadeIn(p2, scale=0.9), run_time=0.6)

    def no_ridge(self):
        self.slide("no-ridge", "Decode stays memory bound at every batch you can reach",
                   book_section="s-throughput-against-latency")
        Bs = [1, 8, 16, 32, 64]
        inten = [1.0, 7.7, 14.7, 27.3, 47.6]
        bc = kit.bar_compare([(f"B = {b}", v, self.color, f"{v:.1f}") for b, v in zip(Bs, inten)],
                             width=8.0, max_value=260, bar_height=0.4, gap=0.22)
        x0 = bc.bars[0].get_left()[0]
        per = bc.bars[0].width / 1.0
        top_y = bc.bars[0].get_top()[1] + 0.3
        bot_y = bc.bars[-1].get_bottom()[1] - 0.2
        rated = DashedLine([x0 + 161 * per, bot_y, 0], [x0 + 161 * per, top_y, 0], color=th.INK_3,
                           stroke_width=th.STROKE_AXIS, dash_length=0.1)
        meas = DashedLine([x0 + 245 * per, bot_y, 0], [x0 + 245 * per, top_y, 0], color=th.BAD,
                          stroke_width=th.STROKE_AXIS, dash_length=0.1)
        rated_t = kit.label("ridge, rated: 161", size=th.SIZE_LABEL, color=th.INK_3).next_to(rated, UP, buff=0.1)
        meas_t = kit.label("measured: 245", size=th.SIZE_LABEL, color=th.BAD).next_to(meas, DOWN, buff=0.1)
        unit = kit.label("FLOPs per byte", size=th.SIZE_LABEL, color=th.INK_3).next_to(bc, UP, buff=0.5)
        unit.align_to(bc, LEFT)
        chart = VGroup(bc, rated, meas, rated_t, meas_t, unit)
        notes = kit.bullets(
            [
                r"At $B = 64$, per-sequence reads are 18.5 GB against 53.8 GB of weights.",
                r"At $L = 2048$: $18{,}120 / 275.8$ MiB per sequence $= 65$ sequences fit.",
            ],
            size=th.SIZE_SMALL, width=11.5,
        )
        col = VGroup(chart, notes).arrange(DOWN, buff=0.55)
        self.place(col)
        with self.narrate(
            "The trade stops being favorable when the per-sequence term catches up. "
            "At batch 64, the per-sequence reads add up to 18.5 GB, against 53.8 GB of weights. "
            "And you can't batch your way to the ridge point. "
            "Intensity at batch 64 is 47.6 FLOPs per byte. "
            "The ridge is 161 against the rated bandwidth. "
            "Against the bandwidth you can measure, it's 245. "
            "The batch can't grow much further anyway: at 2048 tokens, the pool holds 65 sequences. "
            "So decode is memory bound at every reachable batch size, and the tensor cores sit mostly idle."
        ) as t:
            self.play(FadeIn(bc.labels), FadeIn(unit), run_time=0.6)
            t.wait_until("At batch 64, the per")
            self.play(FadeIn(notes[0]), run_time=0.7)
            t.wait_until("Intensity at batch 64")
            self.play(*[GrowFromEdge(b, LEFT) for b in bc.bars], FadeIn(bc.values), run_time=1.2)
            t.wait_until("The ridge is 161")
            self.play(Create(rated), FadeIn(rated_t), run_time=0.8)
            t.wait_until("Against the bandwidth")
            self.play(Create(meas), FadeIn(meas_t), run_time=0.8)
            t.wait_until("The batch can")
            self.play(FadeIn(notes[1]), run_time=0.7)

    # Chunked prefill ----------------------------------------------------------------

    def prefill_stall(self):
        self.slide("prefill-stall", "A 4000-token prompt stalls every stream for 690 ms",
                   book_section="s-chunked-prefill")
        s = 10.0 / 1000  # units per ms
        h = 0.55
        x_start = 0.0

        def dstep(t0, dur, col=th.INFO, mixed=False):
            r = Rectangle(width=dur * s - 0.04, height=h, stroke_width=0, fill_color=th.tint(col, 0.6),
                          fill_opacity=1)
            r.move_to([x_start + (t0 + dur / 2) * s, 0, 0])
            if not mixed:
                return r
            top = Rectangle(width=dur * s - 0.04, height=h * 0.45, stroke_width=0, fill_color=th.tint(self.color, 0.6),
                            fill_opacity=1)
            top.align_to(r, UP).align_to(r, LEFT)
            return VGroup(r, top)

        lane1 = VGroup()
        t0 = 0.0
        for _ in range(4):
            lane1.add(dstep(t0, 45.8))
            t0 += 45.8
        stall = Rectangle(width=690 * s - 0.04, height=h, stroke_width=0, fill_color=th.tint(self.color, 0.6),
                          fill_opacity=1).move_to([(t0 + 345) * s, 0, 0])
        stall_t = kit.label("4000-token prefill: 690 ms", size=th.SIZE_LABEL, color=th.INK).move_to(stall)
        t1 = t0 + 690
        rest = VGroup()
        while t1 + 45.8 <= 1000:
            rest.add(dstep(t1, 45.8))
            t1 += 45.8
        lane2 = VGroup()
        t2 = 0.0
        while t2 + 46.8 <= 1000:
            lane2.add(dstep(t2, 46.8, mixed=t2 >= 4 * 45.8))
            t2 += 46.8
        lane2.shift(DOWN * 1.5)
        n1 = kit.label("one pass", size=th.SIZE_SMALL, color=th.INK_2).next_to(lane1[0], LEFT, buff=0.3)
        n2 = kit.label("chunked", size=th.SIZE_SMALL, color=th.INK_2).next_to(lane2[0], LEFT, buff=0.3)
        n2.align_to(n1, RIGHT)
        axis = NumberLine(x_range=[0, 1000, 100], length=10.0, color=th.INK_3, stroke_width=th.STROKE_AXIS,
                          include_numbers=False, tick_size=0.05)
        axis.move_to([5.0, -2.4, 0])
        a0 = kit.label("0 ms", size=th.SIZE_LABEL, color=th.INK_3).next_to(axis.n2p(0), DOWN, buff=0.12)
        a1 = kit.label("1000 ms", size=th.SIZE_LABEL, color=th.INK_3).next_to(axis.n2p(1000), DOWN, buff=0.12)
        key = VGroup(swatch("decode tokens", th.INFO), swatch("prompt tokens", self.color)).arrange(RIGHT, buff=0.6)
        key.next_to(axis, DOWN, buff=0.7)
        fig = VGroup(lane1, stall, stall_t, rest, lane2, n1, n2, axis, a0, a1, key)
        cost = kit.formula(r"4000 \times 53.8\ \text{GFLOP} = 215\ \text{TFLOP} \;\to\; 690\ \text{ms at peak}",
                           size=th.SIZE_MATH_SMALL)
        col = VGroup(cost, fig).arrange(DOWN, buff=0.6)
        self.place(col)
        with self.narrate(
            "That's the argument for chunked prefill. "
            "Prefill is the opposite of decode: compute bound, and bursty. "
            "A 4000-token prompt costs 215 TFLOP. "
            "That's 690 ms at the A100's peak, and more in practice. "
            "Run it as one forward pass, and every decode in flight stalls for that long. "
            "Every user watching a stream sees a hitch, because a stranger submitted a long prompt. "
            "Chunked prefill splits the prompt across several steps. "
            "Each step carries a few hundred prompt tokens alongside the decodes, so the streams keep flowing."
        ) as t:
            self.play(FadeIn(axis), FadeIn(a0), FadeIn(a1), FadeIn(n1), FadeIn(key), run_time=0.7)
            self.play(LaggedStart(*[FadeIn(r) for r in lane1], lag_ratio=0.3), run_time=1.0)
            t.wait_until("A 4000-token prompt")
            self.play(FadeIn(cost), run_time=0.7)
            t.wait_until("Run it as one")
            self.play(GrowFromEdge(stall, LEFT), run_time=1.5)
            self.play(FadeIn(stall_t), LaggedStart(*[FadeIn(r) for r in rest], lag_ratio=0.3), run_time=0.8)
            t.wait_until("Every user watching")
            self.play(Indicate(stall, color=th.BAD, scale_factor=1.03), run_time=1.0)
            t.wait_until("Chunked prefill splits")
            self.play(FadeIn(n2), LaggedStart(*[FadeIn(r) for r in lane2], lag_ratio=0.15), run_time=2.0)

    def chunk_size(self):
        self.slide("chunk-size", "About 250 prefill tokens hide under the memory floor",
                   book_section="s-chunked-prefill")
        ax = kit.axes([0, 1100, 100], [0, 200, 50], width=7.4, height=4.2,
                      x_label=r"\text{prefill tokens in the step, } T_p", y_label=r"\text{step time (ms)}",
                      x_numbers=[0, 250, 500, 750, 1000], y_numbers=[0, 50, 100, 150, 200])

        def f(x):
            return max(45.8, 0.172 * (x + 16))

        floor = DashedLine(ax.c2p(0, 45.8), ax.c2p(1100, 45.8), color=th.HL_A, stroke_width=th.STROKE_AXIS)
        comp = DashedLine(ax.c2p(0, 0.172 * 16), ax.c2p(1100, 0.172 * 1116), color=th.HL_B,
                          stroke_width=th.STROKE_AXIS)
        curve = VMobject(color=self.color, stroke_width=th.STROKE_CURVE)
        curve.set_points_as_corners([ax.c2p(0, 45.8), ax.c2p(250.3, 45.8), ax.c2p(1100, f(1100))])
        pts = [(128, "45.8"), (256, "46.8"), (512, "91"), (1024, "179")]
        dots = VGroup(*[Dot(ax.c2p(x, f(x)), radius=0.08, color=self.color) for x, _ in pts])
        dlabs = VGroup()
        for (x, txt), d in zip(pts, dots):
            lab = kit.label(f"{txt} ms", size=th.SIZE_LABEL, color=th.INK_2)
            lab.next_to(d, UP if x < 500 else LEFT, buff=0.15)
            if x == 256:
                lab.next_to(d, DR, buff=0.1)
            dlabs.add(lab)
        plot = VGroup(ax, floor, comp, curve, dots, dlabs)
        rule = kit.formula(r"\max(", r"45.8\ \text{ms}", r",\ ", r"0.172\ \text{ms} \times (T_p + B)", r")",
                           size=th.SIZE_MATH_SMALL)
        rule[1].set_color(th.HL_A)
        rule[3].set_color(th.HL_B)
        b16 = kit.prose(r"$B = 16$ decoding sequences.", size=th.SIZE_SMALL, width=None, color=th.INK_3)
        meet = kit.prose(r"They meet at $T_p + B = 266$.", size=th.SIZE_SMALL, width=None)
        knob = kit.prose(r"Lab: chunks of 256. Engine default: 1024, with a 4096-token budget.",
                         size=th.SIZE_SMALL, width=4.4, color=th.INK_3)
        side = VGroup(rule, b16, meet, knob).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        layout = VGroup(plot, side).arrange(RIGHT, buff=0.5)
        self.place(layout)
        with self.narrate(
            "How large can a chunk be before decoding users notice? "
            "Take 16 sequences decoding. "
            "Their step has a memory floor of 45.8 ms, however little arithmetic it does. "
            "Each prefill token adds 0.172 ms of arithmetic at peak, and almost no new bytes. "
            "The step time is the larger of the two, and they meet at 266 tokens. "
            "So about 250 prefill tokens per step ride free. "
            "At 512, the step takes 91 ms. "
            "At 1024, it takes 179, and every decoding user pays. "
            "Real kernels don't reach peak, so measure the free chunk yourself. "
            "The lab uses chunks of 256, and the engine defaults to 1024, which favors throughput."
        ) as t:
            self.play(Create(ax), FadeIn(b16), run_time=0.9)
            t.wait_until("Their step has")
            self.play(Create(floor), FadeIn(rule[:3]), run_time=0.8)
            t.wait_until("Each prefill token")
            self.play(Create(comp), FadeIn(rule[3:]), run_time=0.9)
            t.wait_until("The step time is")
            self.play(Create(curve), FadeIn(meet), run_time=1.2)
            t.wait_until("So about 250")
            self.play(FadeIn(dots[:2]), FadeIn(dlabs[:2]), run_time=0.6)
            t.wait_until("At 512")
            self.play(FadeIn(dots[2]), FadeIn(dlabs[2]), run_time=0.5)
            t.wait_until("At 1024")
            self.play(FadeIn(dots[3]), FadeIn(dlabs[3]), run_time=0.5)
            t.wait_until("The lab uses")
            self.play(FadeIn(knob), run_time=0.6)

    def mixing(self):
        self.slide("mixing", "A mixed step splits for attention, then rejoins",
                   book_section="s-chunked-prefill")
        cols = [self.color] * 6 + [th.INFO] * 4
        w, h = 0.34, 0.5
        row1 = VGroup(*[cell(w, h, c) for c in cols]).arrange(RIGHT, buff=0.03)
        row2 = VGroup(*[cell(w, h, c) for c in cols]).arrange(RIGHT, buff=0.03)
        b1 = kit.box("causal, variable-length", sublabel="prefill chunk + cached prefix", color=self.color,
                     width=4.2)
        b2 = kit.box("paged single-query kernel", sublabel="decode tokens, chapter 15", color=th.INFO, width=4.2)
        mid = VGroup(b1, b2).arrange(DOWN, buff=0.35)
        flow = VGroup(row1, mid, row2).arrange(RIGHT, buff=1.0)
        t1 = kit.label("one flat run", size=th.SIZE_LABEL, color=th.INK_3).next_to(row1, UP, buff=0.15)
        t2 = kit.label("linear layers, MLP", size=th.SIZE_LABEL, color=th.INK_3).next_to(row2, UP, buff=0.15)
        fig = VGroup(flow, t1, t2)
        lines = kit.bullets(
            [
                r"Chunk $c$ attends to chunks $1 \dots c-1$ in the cache, and causally to itself.",
                r"Write this step's $K$ and $V$ before attention reads them.",
                r"Only the last chunk samples; that request's TTFT gets worse.",
            ],
            size=th.SIZE_SMALL, width=11.5,
        )
        col = VGroup(fig, lines).arrange(DOWN, buff=0.75)
        self.place(col)
        arrows = VGroup(
            kit.arrow(row1, b1, color=self.color), kit.arrow(row1, b2, color=th.INFO),
            kit.arrow(b1, row2, color=self.color), kit.arrow(b2, row2, color=th.INFO),
        )
        for a, src, dst in ((arrows[0], row1, b1), (arrows[1], row1, b2), (arrows[2], b1, row2),
                            (arrows[3], b2, row2)):
            a.line.put_start_and_end_on(
                src.get_right() + RIGHT * 0.1 if src is row1 else src.get_right() + RIGHT * 0.1,
                dst.get_left() + LEFT * 0.1 if dst is not row2 else dst.get_left() + LEFT * 0.1)
        with self.narrate(
            "A step that mixes both kinds of token needs care. "
            "Prefill tokens need causal, variable-length attention over their chunk and its cached prefix. "
            "Decode tokens need the paged single-query kernel from chapter 15. "
            "So you split the batch by kind before attention, and rejoin it for the linear layers and the MLP. "
            "Chunk c must attend to every earlier chunk, and causally within itself. "
            "Write this step's keys and values before attention reads them. "
            "Only the last chunk samples, so that request's first token comes later. "
            "That's the right trade: everyone else stops stuttering."
        ) as t:
            self.play(FadeIn(row1), FadeIn(t1), run_time=0.7)
            t.wait_until("Prefill tokens need")
            self.play(GrowArrow(arrows[0].line), FadeIn(b1), run_time=0.8)
            t.wait_until("Decode tokens need")
            self.play(GrowArrow(arrows[1].line), FadeIn(b2), run_time=0.8)
            t.wait_until("So you split")
            self.play(GrowArrow(arrows[2].line), GrowArrow(arrows[3].line), FadeIn(row2), run_time=0.9)
            self.play(FadeIn(t2), run_time=0.5)
            t.wait_until("Chunk c must")
            self.play(FadeIn(lines[0], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("Write this step's")
            self.play(FadeIn(lines[1], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("Only the last chunk")
            self.play(FadeIn(lines[2], shift=RIGHT * 0.1), run_time=0.6)

    # Preemption ----------------------------------------------------------------------

    def preemption(self):
        self.slide("preemption", "Out of blocks: evict the newest, requeue it at the front",
                   book_section="s-preemption")
        owners = {"A": 9, "B": 8, "C": 8, "D": 7}
        seq = [x for x, n in owners.items() for _ in range(n)]
        grid = VGroup(*[cell(0.48, 0.48, self.req[x], None, tint=0.45) for x in seq])
        grid.arrange_in_grid(4, 8, buff=0.05)
        gtitle = kit.label("block pool: 32 blocks, 0 free", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        rq = VGroup(*[chip(x, self.req[x]) for x in "ABCD"]).arrange(RIGHT, buff=0.2)
        rtitle = kit.label("running, oldest to newest", size=th.SIZE_LABEL, color=th.INK_3)
        right = VGroup(rtitle, rq, gtitle, grid).arrange(DOWN, buff=0.25)
        wq = VGroup(chip("F", th.INK_4), chip("E", th.INK_4)).arrange(RIGHT, buff=0.2)
        wslot = chip("D", th.VIOLET).next_to(wq, RIGHT, buff=0.2)
        wtitle = kit.label("waiting", size=th.SIZE_LABEL, color=th.INK_3)
        wbox = RoundedRectangle(width=3.0, height=1.0, corner_radius=0.1, stroke_color=th.RULE_STRONG,
                                stroke_width=2, fill_opacity=0)
        wbox.move_to(VGroup(wq, wslot))
        front = kit.label("front", size=th.SIZE_LABEL, color=th.INK_3).next_to(wbox, DOWN, buff=0.12)
        front.align_to(wbox, RIGHT)
        left = VGroup(wtitle, VGroup(wbox, wq, front)).arrange(DOWN, buff=0.25)
        layout = VGroup(left, right).arrange(RIGHT, buff=1.2)
        notes = VGroup(
            kit.label("Newest first: the oldest are closest to finishing.", size=th.SIZE_SMALL, color=th.INK_2),
            kit.label("To the front: no arrival can overtake it.", size=th.SIZE_SMALL, color=th.INK_2),
        ).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        col = VGroup(layout, notes).arrange(DOWN, buff=0.5)
        self.place(col)
        wslot.move_to(wq.get_right() + RIGHT * (0.2 + wslot.width / 2))
        need = kit.label("needs a block", size=th.SIZE_LABEL, color=th.BAD).next_to(rq[0], UP, buff=0.12)
        d_cells = VGroup(*[grid[i] for i in range(25, 32)])
        gtitle2 = kit.label("block pool: D's 7 freed, 1 to A", size=th.SIZE_SMALL, color=th.GOOD, weight=MEDIUM).move_to(gtitle)
        rtitle_pos = rtitle.get_center()
        del rtitle_pos
        with self.narrate(
            "Sometimes memory runs out anyway. "
            "A running sequence needs a new block, and none is free. "
            "The scheduler preempts: it takes the blocks of the newest sequence, and sends it back to the front "
            "of the queue. "
            "Newest first, because the oldest sequences are closest to finishing, and finishing them frees blocks. "
            "To the front, because at the back, every arrival could overtake it, and preemption would become "
            "starvation. "
            "The victim restarts from scratch, recomputing its prefill."
        ) as t:
            self.play(FadeIn(right), FadeIn(left), run_time=0.9)
            t.wait_until("A running sequence needs")
            self.play(FadeIn(need), Indicate(rq[0], color=th.BAD), run_time=0.9)
            t.wait_until("The scheduler preempts")
            self.play(*[c.square.animate.set_fill(th.tint(th.BAD, 0.3)) for c in d_cells], run_time=0.6)
            self.play(*[c.square.animate.set_fill(th.RAISE) for c in d_cells],
                      ReplacementTransform(gtitle, gtitle2), run_time=0.7)
            self.play(rq[3].animate.move_to(wslot), run_time=1.0)
            self.play(grid[25].square.animate.set_fill(th.tint(self.color, 0.45)), FadeOut(need), run_time=0.5)
            t.wait_until("Newest first")
            self.play(FadeIn(notes[0], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("To the front")
            self.play(FadeIn(notes[1], shift=RIGHT * 0.1), Indicate(front), run_time=0.7)

    def recompute(self):
        self.slide("recompute", "Recompute costs more, and the reference does it anyway",
                   book_section="s-preemption")
        bc = kit.bar_compare(
            [("Recompute prefill", 345, th.BAD, "≈ 345 ms of arithmetic"),
             ("Swap 273 MiB over PCIe", 13, th.INFO, "≈ 13 ms each way")],
            width=6.5, bar_height=0.6, gap=0.45,
        )
        given = kit.label("One 2000-token sequence, priced in chapter 15", size=th.SIZE_SMALL, color=th.INK_3)
        why = kit.bullets(
            [
                r"Scattered blocks transfer poorly, and a swap-in sits on the critical path.",
                r"Pinned host memory is scarce; recompute is a few lines, not a subsystem.",
                r"Retries stop after \textit{max batch size} $+ 1$ attempts, so preemption ends.",
            ],
            size=th.SIZE_SMALL, width=11.5,
        )
        col = VGroup(given, bc, why).arrange(DOWN, buff=0.6)
        self.place(col)
        with self.narrate(
            "For a 2000-token sequence, recompute costs about 345 ms of prefill arithmetic. "
            "Swapping its 273 MiB to host memory over PCIe takes about 13 ms each way. "
            "The reference recomputes anyway. "
            "Scattered blocks transfer poorly, and a swap-in sits on the critical path. "
            "Pinned host memory is a scarce global resource, and recompute is a few lines instead of a subsystem. "
            "And the retries are bounded by the batch-size cap, so preemption always ends."
        ) as t:
            self.play(FadeIn(given), FadeIn(bc.labels), run_time=0.7)
            t.wait_until("For a 2000-token")
            self.play(GrowFromEdge(bc.bars[0], LEFT), FadeIn(bc.values[0]), run_time=1.0)
            t.wait_until("Swapping its")
            self.play(GrowFromEdge(bc.bars[1], LEFT), FadeIn(bc.values[1]), run_time=0.6)
            t.wait_until("Scattered blocks")
            self.play(FadeIn(why[0], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("Pinned host memory")
            self.play(FadeIn(why[1], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("And the retries")
            self.play(FadeIn(why[2], shift=RIGHT * 0.1), run_time=0.6)

    # Admission control ----------------------------------------------------------------

    def admission(self):
        self.slide("admission", "Admit above a watermark, from the head only",
                   book_section="s-admission-control")
        W, H = 11.0, 0.7
        pool = Rectangle(width=W, height=H, stroke_color=th.RULE_STRONG, stroke_width=2, fill_color=th.RAISE,
                         fill_opacity=1)
        rw = W * 362 / 18120
        reserve = Rectangle(width=rw, height=H, stroke_width=0, fill_color=th.BAD, fill_opacity=0.7)
        reserve.align_to(pool, RIGHT)
        used = Rectangle(width=W - rw - 1.3, height=H, stroke_width=0, fill_color=th.tint(self.color, 0.45),
                         fill_opacity=1).align_to(pool, LEFT)
        ptitle = kit.label("block pool: 18,120 pages", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        ptitle.next_to(pool, UP, buff=0.15).align_to(pool, LEFT)
        rlab = kit.label("2% watermark: 362 pages held back", size=th.SIZE_LABEL, color=th.BAD)
        rlab.next_to(pool, UP, buff=0.15).align_to(pool, RIGHT)
        cand = Rectangle(width=1.8, height=H, stroke_color=th.HL_D, stroke_width=3, fill_color=th.tint(th.HL_D, 0.3),
                         fill_opacity=0.8)
        cand.next_to(used, RIGHT, buff=0).shift(DOWN * 0)
        refuse = kit.label("first chunk would dip into the reserve: refused", size=th.SIZE_LABEL, color=th.HL_D)
        refuse.next_to(pool, DOWN, buff=0.15).align_to(pool, RIGHT)
        free_t = kit.label("free", size=th.SIZE_LABEL, color=th.INK_3)
        free_t.move_to([(used.get_right()[0] + reserve.get_left()[0]) / 2, pool.get_y(), 0])
        top = VGroup(pool, used, reserve, ptitle, rlab, refuse, free_t)
        q = VGroup(chip("700", th.INK_4, w=2.2), chip("40", th.INK_4, w=0.8), chip("40", th.INK_4, w=0.8),
                   chip("300", th.INK_4, w=1.4)).arrange(LEFT, buff=0.2)
        qt = kit.label("waiting queue, prompt lengths; head on the right", size=th.SIZE_LABEL, color=th.INK_3)
        qt.next_to(q, UP, buff=0.15).align_to(q, LEFT)
        stop = kit.label("head doesn't fit: stop", size=th.SIZE_SMALL, color=th.BAD, weight=MEDIUM)
        stop.next_to(q[0], RIGHT, buff=0.4)
        cross = kit.label("not skipped ahead", size=th.SIZE_LABEL, color=th.INK_3)
        cross.next_to(VGroup(q[1], q[2]), DOWN, buff=0.15)
        bottom = VGroup(q, qt, stop, cross)
        grow = kit.prose(r"Admitted sequences keep growing: one more block every 16 steps.", size=th.SIZE_SMALL,
                         width=None, color=th.INK_3)
        col = VGroup(top, grow, bottom).arrange(DOWN, buff=0.75)
        self.place(col)
        cand.next_to(used, RIGHT, buff=0)
        with self.narrate(
            "Preemption is the cure, and admission control is the prevention. "
            "A candidate's first chunk must fit the token budget, and its blocks must fit above a watermark. "
            "At a 2 percent watermark on an 18,120-page pool, 362 pages stay unallocated. "
            "Without it, the scheduler hands out the last block, and the next decode step has nowhere to write. "
            "Admitted sequences keep growing, too, by one block every 16 steps. "
            "Finally, admission only looks at the head of the queue. "
            "If the head doesn't fit, the loop stops, rather than skipping to a smaller request behind it."
        ) as t:
            self.play(FadeIn(pool), FadeIn(ptitle), GrowFromEdge(used, LEFT), FadeIn(free_t), run_time=0.9)
            t.wait_until("A candidate's first")
            self.play(FadeIn(reserve), FadeIn(rlab), run_time=0.7)
            t.wait_until("Without it, the")
            self.play(FadeIn(cand, shift=DOWN * 0.2), run_time=0.6)
            self.play(Indicate(reserve, color=th.BAD, scale_factor=1.2), FadeIn(refuse), run_time=0.8)
            self.play(FadeOut(cand), run_time=0.4)
            t.wait_until("Admitted sequences keep")
            self.play(FadeIn(grow), run_time=0.6)
            t.wait_until("Finally, admission")
            self.play(FadeIn(q), FadeIn(qt), run_time=0.7)
            t.wait_until("If the head")
            self.play(FadeIn(stop), Indicate(q[0], color=th.BAD), run_time=0.8)
            self.play(FadeIn(cross), run_time=0.6)

    # Fairness and the tail ------------------------------------------------------------

    def fairness(self):
        self.slide("fairness", "Shortest-job-first starves the big request",
                   book_section="s-fairness-and-the-tail")
        tab = kit.table(
            [
                ["First come, first served", "Worst", "Best, and bounded"],
                ["Shortest-job-first", "Best", "Unbounded under load"],
                ["Priority classes", "Depends", "Unbounded for low class"],
                ["First come, first served, with aging", "Close to first", "Bounded"],
            ],
            header=["Policy", "Mean latency", "Tail latency"], size=th.SIZE_SMALL, row_gap=0.24,
            highlight_rows=[0],
        )
        jitter = kit.callout(
            "note", r"A stream at 45 ms per token slows to 176 ms per token when someone's 1024-token chunk "
                    r"joins the step. Smaller chunks keep the jitter small.",
            title="Fairness users can feel", width=11.0, size=th.SIZE_SMALL)
        col = VGroup(tab, jitter).arrange(DOWN, buff=0.6)
        self.place(col)
        text = (
            "Why refuse to skip? "
            "Skipping ahead is shortest-job-first. "
            "Under sustained load, it can pass over a large request forever. "
            "Its mean wait improves, and its p99 latency goes to infinity. "
            "The reference serves first come, first served, from the head of the queue. "
            "Your wait is bounded by the work ahead of you. "
            "If you need priorities, add aging: priority rises with waiting time, so nothing waits forever. "
            "One more effect is visible only to users. "
            "A stream arriving at 45 ms per token slows to 176 ms per token when a 1024-token chunk joins the step."
        )
        with self.narrate(text, say=voiced(text)) as t:
            self.play(FadeIn(tab.rules), *[FadeIn(c) for c in tab.cells[0]], run_time=0.7)
            t.wait_until("Skipping ahead is")
            self.play(*[FadeIn(c) for c in tab.cells[2]], run_time=0.6)
            t.wait_until("Its mean wait")
            self.play(Indicate(tab.cells[2][2], color=th.BAD), run_time=0.9)
            t.wait_until("The reference serves")
            self.play(FadeIn(tab.bands), *[FadeIn(c) for c in tab.cells[1]], run_time=0.7)
            t.wait_until("If you need priorities")
            self.play(*[FadeIn(c) for c in tab.cells[3]], *[FadeIn(c) for c in tab.cells[4]], run_time=0.8)
            t.wait_until("A stream arriving")
            self.play(FadeIn(jitter, shift=UP * 0.1), run_time=0.8)
