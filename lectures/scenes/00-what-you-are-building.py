"""Chapter 00, what you are building.

Teaches content/chapters/00-what-you-are-building.md in its order. Every number
below was recomputed in Python and matches the chapter.

Model and running example (chapter values):
- Qwen3.8-27B on one A100 80GB. V = 248,320 ("about a quarter of a million
  candidates"), d = 5120, 64 layers. Running prompt: 2000 tokens.
- Per piece of a reply: 53.8 GB read ("tens of gigabytes") and 26.9e9
  multiply-adds ("tens of billions of multiplications").

Uncached loop:
- sum(range(2000, 2500)) = 1,124,750 token positions, about 1.1 million, for
  500 useful tokens. 500-token reply = 500 sequential steps.

Ridge point (A100 PCIe ratings):
- 312e12 FLOP/s / 1935e9 byte/s = 161.24, "161" (lab target 161.2 +- 2).
- Mixed units: 312 / 1935 = 0.161.
- SXM4 at 2039 GB/s: 312e12 / 2039e9 = 153.02, "153".
- Strip and plot positions use log10: log10(161.24) = 2.207.

Decode at batch 1:
- FLOPs = 2 n m, bytes = 2 n m, I = 1. 1 / 161.24 = 0.62% ("under 1%").
- Whole model: 2 x 26.9e9 = 53.8e9 FLOP; 53.8 GB of weights.
- t_memory = 53.8e9 / 1275e9 = 42.20 ms ("42 ms").
- t_compute = 53.8e9 / 312e12 = 0.1724 ms ("0.17 ms").
- Ratio 42.20 / 0.1724 = 244.7; the chapter rounds to "250 times", which is
  what the slide says ("about 250").
- 1000 / 42.2 = 23.7, "about 24 tokens per second".

Prefill, n = 5120: I = nT / (n + T).
- T = 1: 0.9998 ("1"), memory bound by 161.27 ("161 times").
- T = 16: 15.95 ("16"), 161.24 / 15.95 = 10.1 ("10 times").
- T = 161: 156.09 ("156", about the ridge).
- T = 2000: 5120 x 2000 / 7120 = 1438.2 ("about 1440"); 1438.2 / 161.24 =
  8.92 ("about nine times past the ridge").
- T = 2048: 1462.9 ("~1460").

Hybrid (chapter 2's numbers, as this chapter quotes them):
- 16 full-attention layers at 4 KiB per token each: 16 x 4 = 64 KiB.
- All-full-attention: 64 x 4 = 256 KiB per token.
- Fixed state 154,927,104 B = 147.75 MiB ("147.8"); 48 linear layers.
- Break-even 154,927,104 / 196,608 = 788.0 tokens.
- 32k: (154,927,104 + 65,536 x 32,768) / 2^30 = 2.144 GiB ("2.14");
  262,144 x 32,768 / 2^30 = 8.0 GiB.
"""

from manim import *

import kit
from kit import theme as th

LOOP = """\
tokens = tokenizer.encode(prompt)        # e.g. 2000 token IDs
for _ in range(max_new_tokens):
    logits = model(tokens)               # (len(tokens), 248320)
    next_id = sample(logits[-1])         # one token ID
    tokens.append(next_id)
    if next_id == eos_id:
        break
"""

HARNESS = """\
[PASS] 16 layers use full attention
[FAIL] KV cache grows 64 KiB per token - got 262,144 bytes
      kv_kib_per_token = 256

7/8 checks passed
"""

MEM = th.INFO  # memory-side quantities: HBM, bytes, bandwidth
FLOP_C = th.HL_A  # compute-side quantities, \hla in the chapter
BYTE_C = th.HL_B  # bytes, \hlb in the chapter
ACT_C = th.HL_C  # activations, \hlc in the chapter
STATE = th.VIOLET  # linear-attention state

RIDGE = 312e12 / 1935e9  # 161.24


def swatch(text, color, size=th.SIZE_LABEL):
    sq = Square(0.26, stroke_color=th.RULE_STRONG, stroke_width=th.STROKE_HAIRLINE,
                fill_color=th.tint(color, 0.45), fill_opacity=1)
    return VGroup(sq, kit.label(text, size=size, color=th.INK_2)).arrange(RIGHT, buff=0.16)


def tensor(width, height, rows, color, tint=0.3):
    """A tensor drawn as a tinted block with one stripe per token row."""
    rect = Rectangle(width=width, height=height, stroke_color=color, stroke_width=th.STROKE_BOX,
                     fill_color=th.tint(color, tint), fill_opacity=1)
    lines = VGroup()
    for i in range(1, rows):
        y = rect.get_top()[1] - i * height / rows
        lines.add(Line([rect.get_left()[0], y, 0], [rect.get_right()[0], y, 0], color=color,
                       stroke_width=th.STROKE_HAIRLINE, stroke_opacity=0.6))
    g = VGroup(rect, lines)
    g.rect = rect
    return g


def card(head, body, sub, color, width=4.1, height=1.75):
    """A course-part card: a coloured header, a line of what you build, and its chapters."""
    h = kit.label(head, size=th.SIZE_SMALL, color=color, weight=SEMIBOLD)
    b = kit.label(body, size=th.SIZE_SMALL, color=th.INK_2)
    s = kit.label(sub, size=th.SIZE_LABEL, color=th.INK_3)
    col = VGroup(h, b, s).arrange(DOWN, buff=0.16, aligned_edge=LEFT)
    width = max(width, col.width + 0.6)
    rect = RoundedRectangle(width=width, height=height, corner_radius=0.1, stroke_color=color,
                            stroke_width=th.STROKE_BOX, fill_color=th.tint(color, 0.06), fill_opacity=1)
    col.move_to(rect).align_to(rect, LEFT).shift(RIGHT * 0.3)
    return VGroup(rect, col)


def log_strip(width=11.0, height=0.7):
    """A log-scale intensity strip from 1 to 10,000 FLOPs per byte, with the ridge marked."""
    def x_of(v):
        return width * np.log10(v) / 4.0

    xr = x_of(RIDGE)
    mem = Rectangle(width=xr, height=height, stroke_width=0, fill_color=th.tint(BYTE_C, 0.22), fill_opacity=1)
    mem.move_to([xr / 2, 0, 0])
    comp = Rectangle(width=width - xr, height=height, stroke_width=0, fill_color=th.tint(FLOP_C, 0.22),
                     fill_opacity=1)
    comp.move_to([xr + (width - xr) / 2, 0, 0])
    edge = Rectangle(width=width, height=height, stroke_color=th.INK_3, stroke_width=th.STROKE_AXIS,
                     fill_opacity=0).move_to([width / 2, 0, 0])
    ticks = VGroup()
    nums = VGroup()
    for v, tex in [(1, "1"), (10, "10"), (100, "100"), (1000, "1000"), (10000, "10^4")]:
        x = x_of(v)
        ticks.add(Line([x, -height / 2, 0], [x, -height / 2 - 0.1, 0], color=th.INK_3, stroke_width=th.STROKE_AXIS))
        n = kit.math_label(tex, size=th.SIZE_LABEL - 2, color=th.INK_3)
        n.next_to([x, -height / 2 - 0.1, 0], DOWN, buff=0.1)
        nums.add(n)
    ridge = Line([xr, -height / 2 - 0.15, 0], [xr, height / 2 + 0.15, 0], color=th.INK, stroke_width=4)
    mem_t = kit.label("memory bound", size=th.SIZE_SMALL, color=BYTE_C, weight=MEDIUM).move_to(mem)
    comp_t = kit.label("compute bound", size=th.SIZE_SMALL, color=FLOP_C, weight=MEDIUM).move_to(comp)
    ridge_t = kit.label("161", size=th.SIZE_SMALL, color=th.INK, weight=SEMIBOLD).next_to(ridge, UP, buff=0.08)
    unit = kit.label("FLOPs per byte, log scale", size=th.SIZE_LABEL, color=th.INK_3)
    unit.next_to(nums, DOWN, buff=0.12).align_to(edge, RIGHT)
    g = VGroup(mem, comp, edge, ticks, nums, ridge, mem_t, comp_t, ridge_t, unit)
    g.mem, g.comp, g.frame, g.ticks, g.nums = mem, comp, edge, ticks, nums
    g.ridge, g.mem_t, g.comp_t, g.ridge_t, g.unit = ridge, mem_t, comp_t, ridge_t, unit
    g.x_of = x_of
    return g




class Lecture(kit.Lecture):
    def construct(self):
        self.title_slide(
            narration=(
                "What you are building. "
                "You follow one prompt through an inference engine. "
                "You see why decode waits on memory while prefill keeps the GPU busy, and how a lab reaches the GPU."
            )
        )
        self.engine_intro()
        self.five_steps()
        self.forward_pass()
        self.one_layer()
        self.causal()
        self.loop_code()
        self.quadratic()
        self.two_phases()
        self.chef()
        self.ridge_point()
        self.decode_intensity()
        self.decode_step()
        self.prefill()
        self.levers()
        self.hybrid()
        self.course_map()
        self.engine_step()
        self.lab_run()
        self.what_goes_wrong()
        self.closing_slide(
            [
                r"Prefill runs the prompt once; decode runs one token per step.",
                r"Ridge point $\approx 161$ FLOPs per byte. Decode sits at 1, prefill at about $T$.",
                r"One decode step: 42 ms of memory, 0.17 ms of compute.",
                r"The hybrid grows by 64 KiB per token, not 256.",
            ],
            narration=(
                "Here's what to keep. "
                "Prefill runs the prompt once, and decode runs one token per step against a cache. "
                "The ridge point is about 161 FLOPs per byte. "
                "Decode at batch 1 sits at 1, and prefill at about T. "
                "So one decode step spends about 42 ms on memory, and only 0.17 on compute. "
                "Next, a reference page for the notation and GPU vocabulary ahead."
            ),
            book_section="s-what-goes-wrong",
        )

    # Introduction -----------------------------------------------------------

    def engine_intro(self):
        self.slide("engine-intro", "You build the engine behind a streamed reply",
                   book_section="s-before-you-start")
        chat = RoundedRectangle(width=6.2, height=3.0, corner_radius=0.14, stroke_color=th.RULE_STRONG,
                                stroke_width=th.STROKE_BOX, fill_color=th.RAISE, fill_opacity=1)
        ask = kit.label("Why does decode wait on memory?", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
        ask_bg = RoundedRectangle(width=ask.width + 0.4, height=ask.height + 0.3, corner_radius=0.12,
                                  stroke_width=0, fill_color=th.tint(MEM, 0.14), fill_opacity=1)
        ask.move_to(ask_bg)
        bubble = VGroup(ask_bg, ask)
        lines = VGroup(kit.label("Each step reads every weight once,", size=th.SIZE_SMALL, color=th.INK_2),
                       kit.label("then does one multiply-add with it …", size=th.SIZE_SMALL, color=th.INK_2)).arrange(
            DOWN, buff=0.22, aligned_edge=LEFT)
        inner = VGroup(bubble, lines).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        bubble.align_to(inner, RIGHT)
        inner.move_to(chat)

        def stat(big, small):
            b = kit.label(big, size=th.SIZE_HEADING - 4, color=self.color, weight=SEMIBOLD)
            s = kit.label(small, size=th.SIZE_LABEL, color=th.INK_3)
            return VGroup(b, s).arrange(DOWN, buff=0.08, aligned_edge=LEFT)

        stats = VGroup(
            stat("tens of GB", "of numbers read, for each piece"),
            stat("tens of billions", "of multiplications"),
            stat("1 of 248,320", "candidates picked as the winner"),
        ).arrange(DOWN, buff=0.38, aligned_edge=LEFT)
        top = VGroup(VGroup(chat, inner), stats).arrange(RIGHT, buff=0.8)
        pills = VGroup(kit.pill("Qwen3.8-27B", MEM), kit.pill("one A100 80GB", MEM),
                       kit.pill("rented by the second", th.INK_3)).arrange(RIGHT, buff=0.3)
        build = kit.prose(r"You write the attention kernels, the cache, the scheduler, and the server. "
                          r"Nothing is imported from vLLM or TensorRT.", size=th.SIZE_SMALL, width=12.0)
        col = VGroup(top, pills, build).arrange(DOWN, buff=0.5)
        self.place(col)
        glyphs = [g for ln in lines for g in ln]
        with self.narrate(
            "You type a question, and the reply streams back a few characters at a time. "
            "Behind each piece, a program read tens of gigabytes of numbers. "
            "It did tens of billions of multiplications. "
            "Then it picked one winner from about a quarter of a million candidates, and did it all again. "
            "That program is an inference engine, and in this course, you write one. "
            "It runs Qwen 3.8, a 27-billion-parameter open model, on one rented A100 with 80 GB. "
            "You write the kernels, the cache, the scheduler, and the server, and import nothing from vLLM. "
            "You need linear algebra, calculus, and PyTorch, but no CUDA."
        ) as t:
            self.play(FadeIn(chat), FadeIn(bubble), run_time=0.5)
            self.play(LaggedStart(*[FadeIn(g) for g in glyphs], lag_ratio=0.12), run_time=2.6)
            for i, cue in enumerate(["Behind each piece", "It did tens", "Then it picked"]):
                t.wait_until(cue)
                self.play(FadeIn(stats[i], shift=LEFT * 0.1), run_time=0.6)
            t.wait_until("It runs Qwen")
            self.play(LaggedStart(*[FadeIn(p, scale=0.9) for p in pills], lag_ratio=0.3), run_time=1.0)
            t.wait_until("You write the kernels")
            self.play(FadeIn(build, shift=UP * 0.1), run_time=0.7)

    # From prompt to reply ---------------------------------------------------

    def five_steps(self):
        self.slide("five-steps", "Five steps from prompt to reply", book_section="s-from-prompt-to-reply")
        specs = [("Tokenize", "text → IDs"), ("Prefill", "whole prompt"), ("Sample", "scores → ID"),
                 ("Stream", "ID → text"), ("Decode", "next token")]
        boxes = VGroup(*[kit.box(name, sublabel=sub, width=2.0, height=1.35, size=th.SIZE_BODY) for name, sub in specs])
        boxes.arrange(RIGHT, buff=0.6)
        nums = VGroup(*[kit.label(str(i + 1), size=th.SIZE_SMALL, color=th.INK_4, weight=SEMIBOLD)
                        .next_to(b, UP, buff=0.15) for i, b in enumerate(boxes)])
        prompt = kit.label("2000-token prompt", size=th.SIZE_LABEL, color=th.INK_3)
        prompt.next_to(boxes[0], DOWN, buff=0.3)
        loop_t = kit.label("until end-of-sequence or a length limit", size=th.SIZE_LABEL, color=self.color)
        key = kit.pill("steps 2 and 5: almost all the time", self.color, size=th.SIZE_SMALL)
        key.next_to(boxes, UP, buff=0.75)
        spacer = Rectangle(width=1, height=1.6, stroke_width=0, fill_opacity=0).next_to(boxes, DOWN, buff=0)
        self.place(VGroup(key, boxes, nums, prompt, spacer))
        arrows = VGroup(*[kit.arrow(a, b) for a, b in zip(boxes[:-1], boxes[1:])])
        loop = CurvedArrow(boxes[4].get_bottom() + DOWN * 0.1, boxes[2].get_bottom() + DOWN * 0.1,
                           angle=-PI * 0.55, color=self.color, stroke_width=th.STROKE_BOX + 1, tip_length=0.2)
        loop_t.next_to(loop, DOWN, buff=0.12)
        with self.narrate(
            "What happens between pressing Enter and the first piece of the reply? "
            "First, it splits the text into tokens, short pieces of words, each with an integer ID. "
            "The running example is a 2000-token prompt. "
            "Second, prefill runs the model once over the whole prompt, scoring every possible next token. "
            "Third, the sampler turns those scores into one token ID. "
            "Fourth, the engine turns that ID back into text, and streams it. "
            "Fifth, decode appends the token and runs the model again, until an end-of-sequence token or a "
            "length limit. "
            "Steps 2 and 5 take almost all the time."
        ) as t:
            cues = ["First, it splits", "Second, prefill", "Third, the sampler", "Fourth, the engine", "Fifth, decode"]
            for i, cue in enumerate(cues):
                t.wait_until(cue)
                anims = [FadeIn(boxes[i], shift=UP * 0.1), FadeIn(nums[i])]
                if i:
                    anims.append(GrowArrow(arrows[i - 1].line))
                self.play(*anims, run_time=0.6)
                if i == 0:
                    t.wait_until("The running example")
                    self.play(FadeIn(prompt), run_time=0.5)
            self.play(Create(loop), FadeIn(loop_t), run_time=1.0)
            t.wait_until("Steps 2 and 5")
            self.play(*[b.rect.animate.set_stroke(self.color).set_fill(th.tint(self.color, 0.14)) for b in
                        (boxes[1], boxes[4])], FadeIn(key, scale=0.9), run_time=0.8)

    # What a forward pass computes -------------------------------------------

    def forward_pass(self):
        self.slide("forward-pass", "One function: token IDs in, scores out",
                   book_section="s-what-a-forward-pass-computes")
        h, rows = 2.4, 6
        ids = tensor(0.42, h, rows, th.INK_3, tint=0.12)
        s1 = tensor(1.5, h, rows, MEM)
        s2 = tensor(1.5, h, rows, MEM)
        logits = tensor(3.4, h, rows, self.color, tint=0.25)
        row = VGroup(ids, s1, s2, logits).arrange(RIGHT, buff=1.75)
        ids.shift(LEFT * 0.5)

        def caption(m, top, bottom):
            a = kit.label(top, size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
            b = kit.label(bottom, size=th.SIZE_LABEL, color=th.INK_3, font=th.MONO)
            return VGroup(a, b).arrange(DOWN, buff=0.08).next_to(m, DOWN, buff=0.2)

        caps = VGroup(caption(ids, "token IDs", "(T,)"), caption(s1, "residual stream", "(T, 5120)"),
                      caption(s2, "after 64 layers", "(T, 5120)"), caption(logits, "logits", "(T, 248320)"))
        defs = kit.prose(r"$V = 248{,}320$, the vocabulary size. \quad $d = 5120$, the hidden size.",
                         size=th.SIZE_SMALL, width=None)
        col = VGroup(VGroup(row, caps), defs).arrange(DOWN, buff=0.9)
        self.place(col)
        arrows = VGroup(kit.arrow(ids, s1, text="embed"), kit.arrow(s1, s2, text="64 layers"),
                        kit.arrow(s2, logits, text="project"))
        subs = VGroup(kit.prose(r"$V \times d$ table", size=th.SIZE_LABEL, color=th.INK_3, width=None),
                      kit.prose(r"$d \times V$", size=th.SIZE_LABEL, color=th.INK_3, width=None))
        subs[0].next_to(arrows[0].line, DOWN, buff=0.14)
        subs[1].next_to(arrows[2].line, DOWN, buff=0.14)
        hl = VGroup()
        for m in (ids, s1, s2, logits):
            r = m.rect
            band = Rectangle(width=r.width, height=h / rows, stroke_width=0, fill_color=th.MARK, fill_opacity=0.9)
            band.move_to([r.get_x(), r.get_top()[1] - 3.5 * h / rows, 0])
            hl.add(band)
        with self.narrate(
            "The engine is built around one function, the forward pass. "
            "It takes token IDs, and returns a score for every vocabulary entry, at every position. "
            "First, each ID indexes a row of a V by d embedding table. "
            "V is the vocabulary size, 248,320. "
            "And d is the hidden size, 5120. "
            "The result is the residual stream: one vector per token, which every layer reads and adds to. "
            "Then 64 layers refine it. "
            "Finally, a d by V matrix turns each vector into 248,320 scores, called logits. "
            "A higher logit means a likelier next token."
        ) as t:
            self.play(FadeIn(ids), FadeIn(caps[0]), run_time=0.7)
            t.wait_until("First, each ID")
            self.play(GrowArrow(arrows[0].line), FadeIn(arrows[0].label), FadeIn(subs[0]), run_time=0.7)
            t.wait_until("V is the vocabulary")
            self.play(FadeIn(defs), run_time=0.6)
            t.wait_until("The result is the residual")
            self.play(FadeIn(s1), FadeIn(caps[1]), run_time=0.7)
            self.play(FadeIn(hl[0]), FadeIn(hl[1]), run_time=0.5)
            t.wait_until("Then 64 layers")
            self.play(GrowArrow(arrows[1].line), FadeIn(arrows[1].label), FadeIn(s2), FadeIn(caps[2]),
                      FadeIn(hl[2]), run_time=0.9)
            t.wait_until("Finally, a d by V")
            self.play(GrowArrow(arrows[2].line), FadeIn(arrows[2].label), FadeIn(subs[1]), run_time=0.7)
            self.play(GrowFromEdge(logits, LEFT), FadeIn(caps[3]), FadeIn(hl[3]), run_time=0.9)
            t.wait_until("A higher logit")
            self.play(Indicate(logits.rect, color=self.color, scale_factor=1.04), run_time=0.9)

    def one_layer(self):
        self.slide("one-layer", "Each layer adds a correction to the stream",
                   book_section="s-what-a-forward-pass-computes")
        y0, y1 = -1.2, 0.9

        def branch(x0, name, sub, color):
            norm = kit.box("Norm", width=1.3, height=1.0, size=th.SIZE_SMALL)
            main = kit.box(name, sublabel=sub, color=color, width=2.9, height=1.0, size=th.SIZE_SMALL)
            norm.move_to([x0 + 0.65, y1, 0])
            main.next_to(norm, RIGHT, buff=0.5)
            xp = main.get_right()[0] + 0.6
            plus = VGroup(Circle(radius=0.22, color=th.INK, stroke_width=3, fill_color=th.RAISE, fill_opacity=1),
                          kit.label("+", size=th.SIZE_SMALL, color=th.INK, weight=SEMIBOLD))
            plus.move_to([xp, y0, 0])
            out = VMobject(color=th.INK_3, stroke_width=3)
            out.set_points_as_corners([[x0 - 0.5, y0, 0], [x0 - 0.5, y1, 0], norm.get_left()])
            mid = Arrow(norm.get_right(), main.get_left(), buff=0.02, color=th.INK_3, stroke_width=3, tip_length=0.16)
            back = VMobject(color=th.INK_3, stroke_width=3)
            back.set_points_as_corners([main.get_right(), [xp, y1, 0], [xp, y0 + 0.24, 0]])
            g = VGroup(out, norm, mid, main, back, plus)
            g.norm, g.main, g.plus, g.paths = norm, main, plus, VGroup(out, mid, back)
            return g

        b1 = branch(-5.6, "Token mixer", "attention or recurrence", self.color)
        b2 = branch(b1.plus.get_x() + 1.3, "MLP", "matmuls, nonlinearity", FLOP_C)
        stream = Arrow([-6.6, y0, 0], [b2.plus.get_x() + 1.6, y0, 0], buff=0, color=MEM, stroke_width=6,
                       tip_length=0.25)
        s_lab = kit.label("residual stream", size=th.SIZE_SMALL, color=MEM, weight=MEDIUM)
        s_lab.next_to(stream, DOWN, buff=0.18).align_to(stream, LEFT)
        n1 = kit.label("positions exchange information", size=th.SIZE_LABEL, color=self.color)
        n2 = kit.label("each position on its own", size=th.SIZE_LABEL, color=FLOP_C)
        n1.next_to(b1.main, UP, buff=0.2)
        n2.next_to(b2.main, UP, buff=0.2)
        times = kit.label("× 64 layers", size=th.SIZE_SMALL, color=th.INK_3, weight=MEDIUM)
        times.next_to(stream, DOWN, buff=0.18).align_to(stream, RIGHT)
        self.place(VGroup(stream, s_lab, b1, b2, n1, n2, times))
        dot = Dot(stream.get_start(), radius=0.1, color=MEM)
        with self.narrate(
            "Look inside one layer. "
            "It reads the residual stream, computes a correction, and adds it back. "
            "The first half is the token mixer, the only place where positions exchange information. "
            "Usually it's attention: a weighted blend of what earlier positions hold. "
            "In this model, it's sometimes a recurrence instead. "
            "The second half is the MLP: matrix multiplies around a nonlinearity. "
            "It transforms each position on its own. "
            "Both halves are wrapped in normalization. "
            "And both add to the stream, rather than replacing it."
        ) as t:
            self.play(GrowArrow(stream), FadeIn(s_lab), FadeIn(times), run_time=0.9)
            t.wait_until("It reads the residual")
            self.play(FadeIn(dot), run_time=0.3)
            self.play(dot.animate.move_to(stream.get_end() + LEFT * 0.3), run_time=1.8)
            self.play(FadeOut(dot), run_time=0.3)
            t.wait_until("The first half")
            self.play(Create(b1.paths[0]), FadeIn(b1.norm), GrowArrow(b1.paths[1]), FadeIn(b1.main), run_time=0.9)
            self.play(FadeIn(n1), run_time=0.5)
            t.wait_until("In this model")
            self.play(Indicate(b1.main.sub, color=self.color), run_time=0.8)
            t.wait_until("The second half")
            self.play(Create(b2.paths[0]), FadeIn(b2.norm), GrowArrow(b2.paths[1]), FadeIn(b2.main), run_time=0.9)
            t.wait_until("It transforms each")
            self.play(FadeIn(n2), run_time=0.5)
            t.wait_until("Both halves are wrapped")
            self.play(Indicate(VGroup(b1.norm, b2.norm), color=th.INK, scale_factor=1.08), run_time=0.9)
            t.wait_until("And both add")
            self.play(Create(b1.paths[2]), FadeIn(b1.plus), Create(b2.paths[2]), FadeIn(b2.plus), run_time=1.0)

    def causal(self):
        self.slide("causal", "Position t reads only positions at or before t",
                   book_section="s-what-a-forward-pass-computes")
        n = 8
        g = kit.grid(n, n, cell=0.5)
        top = kit.label("may read positions →", size=th.SIZE_LABEL, color=th.INK_3)
        top.next_to(g, UP, buff=0.15).align_to(g, LEFT)
        left = kit.label("position t →", size=th.SIZE_LABEL, color=th.INK_3).rotate(-PI / 2)
        left.next_to(g, LEFT, buff=0.15).align_to(g, UP)
        fig = VGroup(g, top, left)
        key = kit.callout("key", r"It's a pure function of the token IDs. All the state in a chat is the token "
                                 r"sequence, plus caches that avoid recomputing.",
                          title="The forward pass has no memory of its own", width=6.2, size=th.SIZE_SMALL)
        mask = kit.prose(r"The empty triangle is the future: \textbf{causal masking}.", size=th.SIZE_SMALL,
                         width=6.0)
        side = VGroup(mask, key).arrange(DOWN, buff=0.6, aligned_edge=LEFT)
        self.place(VGroup(fig, side).arrange(RIGHT, buff=0.8))
        lower = [[g.cell(r, c) for c in range(r + 1)] for r in range(n)]
        with self.narrate(
            "Only the token mixer is sequential across tokens, and even there, the dependency runs one way. "
            "Each row here is one position, and the shaded cells are what it may read. "
            "Position t reads positions at or before t, never the future. "
            "That's causal masking, and it's what makes the cache possible. "
            "Notice what the forward pass doesn't have: a memory of its own. "
            "It's a pure function of the token IDs you hand it. "
            "All the state in a chat is the token sequence, plus caches that avoid recomputing."
        ) as t:
            self.play(FadeIn(g), FadeIn(top), FadeIn(left), run_time=0.8)
            t.wait_until("Each row here")
            for r in range(n):
                self.play(g.fill(VGroup(*lower[r]), self.color, tint=0.45), run_time=0.3)
            t.wait_until("Position t reads")
            self.play(Create(g.outline(5, 6, 0, 6, color=self.color)), run_time=0.7)
            t.wait_until("That's causal masking")
            self.play(FadeIn(mask), run_time=0.6)
            t.wait_until("Notice what")
            self.play(FadeIn(key, shift=UP * 0.1), run_time=0.8)

    # What autoregressive means ----------------------------------------------

    def loop_code(self):
        self.slide("loop-code", "Generation is a loop around the forward pass",
                   book_section="s-what-autoregressive-means")
        panel = kit.code(LOOP, "python")
        note = kit.prose(r"Its cost structure is \textbf{the whole course}.", size=th.SIZE_BODY,
                         color=self.color, width=None)
        self.place(VGroup(panel, note).arrange(DOWN, buff=0.5))
        with self.narrate(
            "Why does a reply take one model call per token? "
            "Generation is a loop around the forward pass. "
            "Encode the prompt into token IDs. "
            "Then, every iteration, run the model over all the tokens so far. "
            "Only the last position's logits describe what comes next, so you sample from the last row. "
            "Append the token, and stop at end-of-sequence. "
            "Read this loop carefully, because its cost structure is the whole course."
        ) as t:
            self.play(FadeIn(panel.frame), run_time=0.5)
            t.wait_until("Encode the prompt")
            self.play(*panel.reveal(1), run_time=0.6)
            t.wait_until("Then, every")
            self.play(*panel.reveal(2, 3), panel.highlight(3), run_time=0.8)
            t.wait_until("Only the last")
            self.play(*panel.reveal(4), panel.highlight(4), run_time=0.7)
            t.wait_until("Append the token")
            self.play(*panel.reveal(5, 6, 7), run_time=0.8)
            t.wait_until("Read this loop")
            self.play(FadeIn(note, shift=UP * 0.1), run_time=0.7)

    def quadratic(self):
        self.slide("quadratic", "500 new tokens cost 1.1 million token positions",
                   book_section="s-what-autoregressive-means")
        w0, dw, hgt = 3.6, 0.42, 0.42
        names = ["2000", "2001", "2002", "2003", "…", "2499"]
        rows = VGroup()
        for j, nm in enumerate(names):
            extra = j if j < 4 else (6 if j == 4 else 8)
            old_w = w0 + (extra - 1) * dw if j else w0
            old = Rectangle(width=old_w, height=hgt, stroke_width=0,
                            fill_color=th.tint(self.color if j == 0 else th.INK_4, 0.45), fill_opacity=1)
            parts = [old]
            if j:
                new = Rectangle(width=dw, height=hgt, stroke_width=0, fill_color=th.tint(self.color, 0.55),
                                fill_opacity=1)
                new.next_to(old, RIGHT, buff=0.04)
                parts.append(new)
            lab = kit.label(nm, size=th.SIZE_LABEL, color=th.INK_3, font=th.MONO)
            rows.add(VGroup(lab, VGroup(*parts)))
        for r in rows:
            r[1].next_to(r[0], RIGHT, buff=0.3)
        rows.arrange(DOWN, buff=0.14, aligned_edge=LEFT)
        for r in rows:
            r[1].align_to(rows[0][1], LEFT)
        legend = VGroup(swatch("recomputed from scratch", th.INK_4), swatch("new work", self.color)).arrange(
            RIGHT, buff=0.5)
        left = VGroup(rows, legend).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        f = kit.formula(r"2000 + 2001 + \cdots + 2499", r"= 1{,}124{,}750", size=th.SIZE_MATH_SMALL)
        per = kit.prose(r"token positions, roughly 1.1 million, to produce \textbf{500} useful tokens",
                        size=th.SIZE_SMALL, width=5.4)
        big_o = kit.pill("O(n²): the loop chapter 9 replaces", self.color, size=th.SIZE_LABEL)
        right = VGroup(f, per, big_o).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        self.place(VGroup(left, right).arrange(RIGHT, buff=0.7))
        with self.narrate(
            "The first iteration processes 2000 tokens. "
            "Every iteration after it processes 2001, then 2002, and so on. "
            "Each one recomputes from scratch everything it already computed. "
            "Generate 500 tokens this way, and the model runs over 1,124,750 token positions. "
            "That's roughly 1.1 million positions, to produce 500 useful ones. "
            "This is the O(n²) loop that chapter 9 replaces."
        ) as t:
            self.play(FadeIn(rows[0][0]), GrowFromEdge(rows[0][1], LEFT), run_time=0.8)
            t.wait_until("Every iteration after")
            for r in rows[1:4]:
                self.play(FadeIn(r[0]), GrowFromEdge(r[1], LEFT), run_time=0.5)
            t.wait_until("Each one recomputes")
            self.play(FadeIn(legend), Indicate(VGroup(*[r[1][0] for r in rows[1:4]]), color=th.INK_3,
                                               scale_factor=1.03), run_time=1.0)
            t.wait_until("Generate 500")
            self.play(*[FadeIn(r[0]) for r in rows[4:]], *[GrowFromEdge(r[1], LEFT) for r in rows[4:]], run_time=0.7)
            self.play(Write(f), run_time=1.2)
            t.wait_until("That's roughly")
            self.play(FadeIn(per), run_time=0.6)
            t.wait_until("This is the")
            self.play(FadeIn(big_o, scale=0.9), run_time=0.6)

    def two_phases(self):
        self.slide("two-phases", "Cache keys and values, then decode one token per step",
                   book_section="s-what-autoregressive-means")
        cols, p = 15, 8
        g = kit.grid(2, cols, cell=(0.62, 0.62), gap=0.05, row_labels=["keys", "values"])
        fixed = kit.prose(r"A key or value never changes once it's computed.", size=th.SIZE_BODY, width=None)
        b1 = Brace(g.block(0, 1, 0, p), UP, buff=0.12, color=th.INK_4)
        b1_t = kit.label("prefill: the whole prompt, at once", size=th.SIZE_SMALL, color=th.INK_2)
        b1_t.next_to(b1, UP, buff=0.08)
        b2 = Brace(g.block(1, 2, p, p + 5), DOWN, buff=0.12, color=th.INK_4)
        b2_t = kit.label("decode: one token per step", size=th.SIZE_SMALL, color=th.INK_2)
        b2_t.next_to(b2, DOWN, buff=0.08)
        steps = kit.pill("A 500-token reply takes 500 sequential steps", self.color, size=th.SIZE_SMALL)
        self.place(VGroup(fixed, VGroup(g, b1, b1_t, b2, b2_t), steps).arrange(DOWN, buff=0.6))
        with self.narrate(
            "The fix follows from causal masking. "
            "Position t depends only on earlier positions, and those haven't changed. "
            "In attention, each position computes a key, which later positions match against, and a value, which "
            "it hands over. "
            "Neither changes once computed. "
            "So you keep them in a KV cache, and feed the loop one new token per step. "
            "Prefill is the first iteration: the whole prompt, at once. "
            "Decode is every iteration after it: one token, against the cache. "
            "That's what autoregressive means: the output is fed back as input. "
            "So a 500-token reply takes 500 sequential steps, however large your GPU is."
        ) as t:
            self.play(FadeIn(g), run_time=0.7)
            t.wait_until("Neither changes")
            self.play(FadeIn(fixed), run_time=0.6)
            t.wait_until("Prefill is the first")
            self.play(g.fill(g.block(0, 2, 0, p), self.color, tint=0.45), FadeIn(b1), FadeIn(b1_t), run_time=0.9)
            t.wait_until("Decode is every")
            for c in range(p, p + 5):
                q = kit.pill("new", th.VIOLET, size=th.SIZE_LABEL).next_to(g.cell(0, c), UP, buff=0.15)
                read = g.outline(0, 2, 0, c + 1, color=th.VIOLET, stroke=3)
                self.play(FadeIn(q, shift=DOWN * 0.1), g.fill(g.col(c), th.VIOLET, tint=0.35), run_time=0.3)
                self.play(Create(read), run_time=0.3)
                self.play(FadeOut(q), FadeOut(read), g.fill(g.col(c), self.color, tint=0.45), run_time=0.25)
            self.play(FadeIn(b2), FadeIn(b2_t), run_time=0.6)
            t.wait_until("So a 500-token")
            self.play(FadeIn(steps, scale=0.9), run_time=0.7)

    # Why the GPU waits on memory --------------------------------------------

    def chef(self):
        self.slide("chef", "A GPU computes faster than memory can feed it",
                   book_section="s-why-the-gpu-waits-on-memory")
        hbm = kit.box("HBM", sublabel="80 GB: weights, KV cache", color=MEM, width=3.8, height=1.9,
                      size=th.SIZE_HEADING - 4)
        chip = kit.box("Tensor cores", sublabel="312 TFLOP/s bf16, in 108 SMs", color=self.color, width=4.3,
                       height=1.9, size=th.SIZE_HEADING - 4)
        row = VGroup(hbm, chip).arrange(RIGHT, buff=4.0)

        def role(text):
            return kit.label(text, size=th.SIZE_SMALL, color=th.INK_3, font=th.SERIF, slant="ITALIC")

        r1, r2, r3 = role("the pantry").next_to(hbm, UP, buff=0.2), role("the door"), role("the chef").next_to(
            chip, UP, buff=0.2)
        intensity = kit.callout("definition", r"The FLOPs a piece of work does per byte it moves: chops per "
                                              r"ingredient. A property of the work, not the chip.",
                                title="Arithmetic intensity", width=11.0, size=th.SIZE_SMALL)
        flop = kit.prose(r"A FLOP is one add or one multiply on floating-point numbers.", size=th.SIZE_SMALL,
                         color=th.INK_3, width=None)
        self.place(VGroup(VGroup(row, r1, r3), flop, intensity).arrange(DOWN, buff=0.6))
        door = kit.arrow(hbm, chip, text="1935 GB/s", color=MEM, stroke=6, size=th.SIZE_SMALL)
        r2.next_to(door.label, UP, buff=0.12)
        bw = kit.label("memory bandwidth", size=th.SIZE_LABEL, color=MEM).next_to(door.line, DOWN, buff=0.15)
        start, end = door.line.get_start(), door.line.get_end()
        with self.narrate(
            "A GPU computes far faster than its memory can feed it. "
            "Picture a chef who chops faster than an assistant can carry ingredients through the pantry door. "
            "HBM, the GPU's main memory, is the pantry: 80 GB, holding the weights and the KV cache. "
            "Memory bandwidth is the door: 1935 GB/s, as rated. "
            "The tensor cores are the chef, peaking at 312 TFLOP/s in bfloat16. "
            "A FLOP is one add or one multiply. "
            "If each ingredient gets one chop, the chef mostly waits. "
            "Arithmetic intensity counts chops per ingredient: the FLOPs a piece of work does per byte it moves."
        ) as t:
            t.wait_until("HBM, the GPU's")
            self.play(FadeIn(hbm), FadeIn(r1), run_time=0.7)
            t.wait_until("Memory bandwidth")
            self.play(GrowArrow(door.line), FadeIn(door.label), FadeIn(bw), FadeIn(r2), run_time=0.8)
            t.wait_until("The tensor cores")
            self.play(FadeIn(chip), FadeIn(r3), run_time=0.7)
            t.wait_until("A FLOP is")
            self.play(FadeIn(flop), run_time=0.6)
            t.wait_until("If each ingredient")
            for _ in range(2):
                item = Square(0.2, stroke_width=0, fill_color=MEM, fill_opacity=1).move_to(start)
                self.play(item.animate.move_to(end), run_time=0.7)
                self.play(FadeOut(item), Indicate(chip.rect, color=self.color, scale_factor=1.03), run_time=0.4)
            t.wait_until("Arithmetic intensity counts")
            self.play(FadeIn(intensity, shift=UP * 0.1), run_time=0.8)

    def ridge_point(self):
        self.slide("ridge-point", "The ridge point: about 161 FLOPs per byte",
                   book_section="s-arithmetic-intensity-and-the-ridge-point")
        i_def = kit.formula(r"I", r"=", r"\frac{\text{FLOPs}}{\text{bytes moved}}")
        i_def[2][0:5].set_color(FLOP_C)
        i_def[2][6:].set_color(BYTE_C)
        frac = kit.formula(r"I_{\text{ridge}} = \frac{312 \times 10^{12}\ \text{FLOP/s}}"
                           r"{1935 \times 10^{9}\ \text{byte/s}}")
        res = kit.formula(r"\approx 161\ \text{FLOP/byte}")
        box = SurroundingRectangle(res, color=self.color, buff=0.14, stroke_width=3)
        ridge = VGroup(frac, VGroup(res, box)).arrange(RIGHT, buff=0.3)
        maths = VGroup(i_def, ridge).arrange(RIGHT, buff=1.2)
        strip = log_strip()
        meaning = kit.prose(r"How many FLOPs the chip can do in the time it takes to read one byte.",
                            size=th.SIZE_SMALL, color=th.INK_3, width=None)
        self.place(VGroup(maths, meaning, strip).arrange(DOWN, buff=0.7))
        with self.narrate(
            "How much work per byte does an operation need before memory stops holding it back? "
            "Divide the FLOPs an operation performs by the bytes it moves. "
            "That's its arithmetic intensity. "
            "Now divide the A100's peak FLOPs per second by its rated bandwidth, in base units. "
            "You get the ridge point: about 161 FLOPs per byte. "
            "It's how many FLOPs the chip can do while it reads one byte. "
            "Below 161, an operation is memory bound, and bandwidth sets its speed. "
            "Above it, the operation is compute bound. "
            "Below the ridge, no kernel cleverness helps. "
            "Only moving fewer bytes does."
        ) as t:
            t.wait_until("Divide the FLOPs")
            self.play(FadeIn(i_def), run_time=0.8)
            t.wait_until("Now divide")
            self.play(FadeIn(frac), run_time=0.8)
            t.wait_until("You get the ridge")
            self.play(FadeIn(res), Create(box), run_time=0.8)
            t.wait_until("It's how many")
            self.play(FadeIn(meaning), FadeIn(strip.frame), FadeIn(strip.ticks), FadeIn(strip.nums),
                      FadeIn(strip.unit), run_time=0.8)
            self.play(Create(strip.ridge), FadeIn(strip.ridge_t), run_time=0.6)
            t.wait_until("Below 161")
            self.play(GrowFromEdge(strip.mem, LEFT), FadeIn(strip.mem_t), run_time=0.8)
            t.wait_until("Above it")
            self.play(GrowFromEdge(strip.comp, LEFT), FadeIn(strip.comp_t), run_time=0.8)
            t.wait_until("Only moving")
            self.play(Indicate(strip.mem_t, color=BYTE_C), run_time=0.8)

    # Decode -------------------------------------------------------------------

    def decode_intensity(self):
        self.slide("decode-intensity", "Decode at batch 1 does one FLOP per byte",
                   book_section="s-decode-one-token-per-weight-read")
        n, m = 6, 8
        W = kit.grid(n, m, cell=0.42)
        x = kit.grid(m, 1, cell=0.42)
        y = kit.grid(n, 1, cell=0.42)
        x.paint(x.block(0, m, 0, 1), self.color, tint=0.35)
        times = kit.label("×", size=th.SIZE_HEADING, color=th.INK_3)
        eq = kit.label("=", size=th.SIZE_HEADING, color=th.INK_3)
        mv = VGroup(W, times, x, eq, y).arrange(RIGHT, buff=0.3)
        wl = kit.prose(r"weights, $n \times m$, bf16", size=th.SIZE_LABEL, color=th.INK_3, width=None)
        wl.next_to(W, DOWN, buff=0.18)
        xl = kit.label("one token", size=th.SIZE_LABEL, color=self.color).next_to(x, DOWN, buff=0.18)
        fig = VGroup(mv, wl, xl)
        l0 = kit.formula(r"\text{FLOPs} = 2 \times n \times m", size=th.SIZE_MATH_SMALL)
        l1 = kit.formula(r"\text{bytes} = 2 \times n \times m", size=th.SIZE_MATH_SMALL)
        l2 = kit.formula(r"I_{\text{decode}} = 1\ \text{FLOP/byte}", size=th.SIZE_MATH_SMALL)
        l0[0][0:5].set_color(FLOP_C)
        l1[0][0:5].set_color(BYTE_C)
        w0 = kit.label("one multiply plus one add per weight", size=th.SIZE_LABEL, color=FLOP_C)
        w1 = kit.label("two bytes per bf16 weight", size=th.SIZE_LABEL, color=BYTE_C)
        box = SurroundingRectangle(l2, color=self.color, buff=0.14, stroke_width=3)
        share = kit.pill("1/161 of the arithmetic: under 1%", BYTE_C, size=th.SIZE_SMALL)
        right = VGroup(VGroup(l0, w0).arrange(DOWN, buff=0.1, aligned_edge=LEFT),
                       VGroup(l1, w1).arrange(DOWN, buff=0.1, aligned_edge=LEFT),
                       VGroup(l2, box), share).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        self.place(VGroup(fig, right).arrange(RIGHT, buff=0.9))
        with self.narrate(
            "Where does decode land? "
            "At batch 1, almost all of decode's work multiplies a weight matrix by a single vector. "
            "Each weight costs one multiply and one add, so the FLOPs are 2 × n × m. "
            "Each bfloat16 weight is 2 bytes, so the bytes are also 2 × n × m. "
            "They cancel: decode's intensity is 1 FLOP per byte. "
            "Every weight is read, used for one multiply-add, and discarded. "
            "Against a ridge of 161, that's under 1 percent of the A100's arithmetic."
        ) as t:
            self.play(FadeIn(fig), run_time=0.8)
            t.wait_until("Each weight costs")
            self.play(FadeIn(l0), FadeIn(w0), run_time=0.8)
            t.wait_until("Each bfloat16")
            self.play(FadeIn(l1), FadeIn(w1), run_time=0.8)
            t.wait_until("They cancel")
            self.play(FadeIn(l2), Create(box), run_time=0.8)
            t.wait_until("Every weight is read")
            for r in range(n):
                self.play(W.fill(W.row(r), BYTE_C, tint=0.5), run_time=0.22)
                self.play(W.fill(W.row(r), th.INK_4, tint=0.12), y.fill(y.cell(r, 0), FLOP_C, tint=0.45),
                          run_time=0.25)
            t.wait_until("Against a ridge")
            self.play(FadeIn(share, scale=0.9), run_time=0.7)

    def decode_step(self):
        self.slide("decode-step", "One decode step: 42 ms of memory, 0.17 ms of math",
                   book_section="s-decode-one-token-per-weight-read")
        tm = kit.formula(r"t_{\text{memory}} = \frac{53.8\ \text{GB}}{1275\ \text{GB/s}}", r"\approx 42\ \text{ms}")
        tc = kit.formula(r"t_{\text{compute}} = \frac{53.8\ \text{GFLOP}}{312\ \text{TFLOP/s}}",
                         r"\approx 0.17\ \text{ms}")
        tm[1].set_color(BYTE_C)
        tc[1].set_color(FLOP_C)
        maths = VGroup(tm, tc).arrange(RIGHT, buff=1.0)
        given = kit.prose(r"26.9B parameters: 53.8 GB of bf16 weights, read once per step.", size=th.SIZE_SMALL,
                          color=th.INK_3, width=None)
        note = kit.prose(r"1275 GB/s is what a copy measures; the rating is 1935.", size=th.SIZE_LABEL,
                         color=th.INK_3, width=None)
        bc = kit.bar_compare([("Waiting on memory", 42.2, BYTE_C, "42 ms"),
                              ("Arithmetic", 0.172, FLOP_C, "0.17 ms")], width=7.5, bar_height=0.55, gap=0.35)
        pills = VGroup(kit.pill("about 250 times longer waiting", th.INK_3, size=th.SIZE_SMALL),
                       kit.pill("batch 1: at most about 24 tokens/s", self.color, size=th.SIZE_SMALL)).arrange(
            RIGHT, buff=0.4)
        self.place(VGroup(given, maths, note, bc, pills).arrange(DOWN, buff=0.42))
        with self.narrate(
            "Chapter 2 counts 26.9 billion parameters, which is 53.8 GB in bfloat16. "
            "Each step reads all of them, and does 53.8 GFLOP. "
            "Divide by the 1275 GB/s a copy actually measures, and memory takes about 42 ms. "
            "The arithmetic, at peak, takes about 0.17 ms. "
            "Decode spends about 250 times longer waiting for weights than using them. "
            "These are lower bounds, not benchmarks. "
            "At batch 1, that caps one conversation at about 24 tokens per second."
        ) as t:
            self.play(FadeIn(given), FadeIn(bc.labels), run_time=0.8)
            t.wait_until("Divide by the 1275")
            self.play(FadeIn(tm), run_time=0.8)
            self.play(GrowFromEdge(bc.bars[0], LEFT), FadeIn(bc.values[0]), run_time=1.2)
            self.play(FadeIn(note), run_time=0.6)
            t.wait_until("The arithmetic, at")
            self.play(FadeIn(tc), run_time=0.8)
            self.play(GrowFromEdge(bc.bars[1], LEFT), FadeIn(bc.values[1]), run_time=0.6)
            t.wait_until("Decode spends")
            self.play(FadeIn(pills[0], scale=0.9), run_time=0.6)
            t.wait_until("At batch 1")
            self.play(FadeIn(pills[1], scale=0.9), run_time=0.6)

    # Prefill --------------------------------------------------------------------

    def prefill(self):
        self.slide("prefill", "Prefill shares one weight read across the prompt",
                   book_section="s-prefill-many-tokens-per-weight-read")
        ax = kit.axes([0, 4, 1], [0, 4, 1], width=6.2, height=4.2)
        ticks = VGroup()
        for v, tex in [(0, "1"), (1, "10"), (2, "100"), (3, "1000"), (4, "10^4")]:
            ticks.add(kit.math_label(tex, size=th.SIZE_LABEL - 2).next_to(ax.c2p(v, 0), DOWN, buff=0.14))
            ticks.add(kit.math_label(tex, size=th.SIZE_LABEL - 2).next_to(ax.c2p(0, v), LEFT, buff=0.12))
        xt = kit.label("tokens per weight read, T", size=th.SIZE_LABEL, color=th.INK_3)
        xt.next_to(ticks, DOWN, buff=0.15).align_to(ax, RIGHT)
        yt = kit.label("FLOPs per byte", size=th.SIZE_LABEL, color=th.INK_3).next_to(ax, UP, buff=0.15)
        yt.align_to(ax, LEFT)

        def inten(T):
            return 5120 * T / (5120 + T)

        curve = ax.plot(lambda u: np.log10(inten(10 ** u)), x_range=[0, 4, 0.02], color=self.color,
                        stroke_width=th.STROKE_CURVE)
        yr = np.log10(RIDGE)
        ridge = DashedLine(ax.c2p(0, yr), ax.c2p(4, yr), color=th.INK_3, stroke_width=th.STROKE_AXIS)
        ridge_t = kit.label("ridge, 161", size=th.SIZE_LABEL, color=th.INK_2).next_to(ax.c2p(4, yr), DOWN, buff=0.1)
        ridge_t.align_to(ax.c2p(4, 0), RIGHT)
        pts = [(1, "decode: 1", DR), (16, "16", DR), (161, "161", DR), (2048, "2048", UL)]
        dots, labs = VGroup(), VGroup()
        for T, txt, d in pts:
            p = ax.c2p(np.log10(T), np.log10(inten(T)))
            dots.add(Dot(p, radius=0.08, color=th.INK))
            labs.add(kit.label(txt, size=th.SIZE_LABEL, color=th.INK, weight=MEDIUM).next_to(p, d, buff=0.1))
            if T == 1:
                labs[-1].shift(RIGHT * 0.25 + UP * 0.45)
        plot = VGroup(ax, ticks, xt, yt, ridge, curve, ridge_t, dots, labs)
        f = kit.formula(r"I_{\text{prefill}} = \frac{2nmT}{2nm + 2mT}", r"= \frac{nT}{n + T}",
                        size=th.SIZE_MATH_SMALL)
        approx = kit.formula(r"\approx T \quad \text{when } T \ll n", size=th.SIZE_MATH_SMALL)
        ex = kit.formula(r"\frac{5120 \times 2000}{7120} \approx 1440", size=th.SIZE_MATH_SMALL)
        ex_note = kit.label("about nine times past the ridge", size=th.SIZE_LABEL, color=FLOP_C, weight=MEDIUM)
        same = kit.prose(r"Same matrices, same hardware, same kernels.", size=th.SIZE_SMALL, color=th.INK_3,
                         width=4.6)
        side = VGroup(f, approx, VGroup(ex, ex_note).arrange(DOWN, buff=0.12, aligned_edge=LEFT), same).arrange(
            DOWN, buff=0.4, aligned_edge=LEFT)
        self.place(VGroup(plot, side).arrange(RIGHT, buff=0.6))
        with self.narrate(
            "The same weight matrix multiplies T vectors at once. "
            "The FLOPs grow with T, but one read of the weights serves all T tokens. "
            "Add the activations to the bytes, and the intensity is n T, over n plus T. "
            "That's about T, while T is small against n. "
            "For a 2000-token prompt, use the exact form. "
            "It gives about 1440, roughly nine times past the ridge, so prefill saturates the tensor cores. "
            "On the plot, one token sits at 1. "
            "At 16 tokens, it's still memory bound, by about 10 times. "
            "Around 161 tokens, it reaches the ridge. "
            "Same matrices, same hardware: only the tokens per weight read changed."
        ) as t:
            self.play(Create(ax), FadeIn(ticks), FadeIn(xt), FadeIn(yt), run_time=0.9)
            self.play(Create(ridge), FadeIn(ridge_t), run_time=0.6)
            t.wait_until("Add the activations")
            self.play(FadeIn(f), run_time=0.9)
            t.wait_until("That's about T")
            self.play(FadeIn(approx), Create(curve), run_time=1.4)
            t.wait_until("It gives about 1440")
            self.play(FadeIn(ex), FadeIn(ex_note), FadeIn(dots[3], scale=0.5), FadeIn(labs[3]), run_time=0.8)
            for i, cue in enumerate(["On the plot", "At 16 tokens", "Around 161"]):
                t.wait_until(cue)
                self.play(FadeIn(dots[i], scale=0.5), FadeIn(labs[i]), run_time=0.6)
            t.wait_until("Same matrices")
            self.play(FadeIn(same), run_time=0.6)

    # What follows from the ratio -------------------------------------------------

    def levers(self):
        self.slide("levers", "Every optimization moves one term of the fraction",
                   book_section="s-what-follows-from-the-ratio")
        f = kit.formula(r"I = \frac{\text{FLOPs}}{\text{bytes}}", r"\quad\text{per read of the weights}",
                        size=th.SIZE_MATH)
        f[0][2:7].set_color(FLOP_C)
        f[0][8:].set_color(BYTE_C)
        f[1].set_color(th.INK_3)
        cards = VGroup(
            kit.callout("result", r"Continuous batching: one read serves many sequences. Chapter 16.",
                        title="1. Raise T", width=4.15, size=th.SIZE_SMALL),
            kit.callout("assumption", r"GQA, quantization, paged caches, linear attention. Chapters 6, 7, 15, 18.",
                        title="2. Shrink the bytes", width=4.15, size=th.SIZE_SMALL),
            kit.callout("example", r"Speculative decoding commits several tokens per step. Chapter 20.",
                        title="3. More tokens per read", width=4.15, size=th.SIZE_SMALL),
        ).arrange(RIGHT, buff=0.3, aligned_edge=UP)
        moral = kit.prose(r"Each technique moves \textbf{one term of one fraction}.", size=th.SIZE_BODY,
                          color=self.color, width=None)
        self.place(VGroup(f, cards, moral).arrange(DOWN, buff=0.6))
        with self.narrate(
            "If one fraction sets the bottleneck, how do you beat it? "
            "Each family of optimizations attacks a different term. "
            "First, raise T. "
            "One weight read serves many sequences: that's continuous batching, the largest throughput lever. "
            "Second, shrink the bytes. "
            "Reading half as much runs twice as fast: grouped-query attention, quantization, paged caches, and "
            "linear attention. "
            "Third, get more than one token per read, with speculative decoding. "
            "Nothing in the course is arbitrary: each technique moves one term of one fraction."
        ) as t:
            self.play(FadeIn(f), run_time=0.8)
            for i, cue in enumerate(["First, raise T", "Second, shrink", "Third, get more"]):
                t.wait_until(cue)
                self.play(FadeIn(cards[i], shift=UP * 0.1), run_time=0.7)
                if i == 0:
                    self.play(Indicate(f[0][2:7], color=FLOP_C), run_time=0.7)
                elif i == 1:
                    self.play(Indicate(f[0][8:], color=BYTE_C), run_time=0.7)
            t.wait_until("Nothing in the course")
            self.play(FadeIn(moral, shift=UP * 0.1), run_time=0.7)

    # The model is a hybrid --------------------------------------------------------

    def hybrid(self):
        self.slide("hybrid", "48 of the 64 layers keep a fixed-size state",
                   book_section="s-the-model-is-a-hybrid-and-that-matters")
        g = kit.grid(1, 64, cell=(0.17, 0.62), gap=0.03)
        cap = kit.label("64 layers", size=th.SIZE_LABEL, color=th.INK_3).next_to(g, UP, buff=0.15).align_to(g, LEFT)
        full = VGroup(*[g.cell(0, i) for i in range(64) if i % 4 == 3])
        lin = VGroup(*[g.cell(0, i) for i in range(64) if i % 4 != 3])
        key = VGroup(swatch("16 full attention: cache grows, 4 KiB per token each", self.color),
                     swatch("48 gated delta linear attention: 147.8 MiB state, fixed", STATE)).arrange(
            DOWN, buff=0.2, aligned_edge=LEFT)
        bc = kit.bar_compare([("Full attention everywhere", 8.0, th.INK_4, "8.0 GiB"),
                              ("This hybrid", 2.144, self.color, "2.14 GiB")],
                             width=6.5, bar_height=0.5, gap=0.3)
        bc_t = kit.label("cache per sequence at 32k context", size=th.SIZE_LABEL, color=th.INK_3)
        rates = kit.prose(r"Per token: $64 \times 4 = 256$ KiB against $16 \times 4 = 64$ KiB. "
                          r"The hybrid is ahead past about \textbf{788 tokens}.", size=th.SIZE_SMALL, width=11.5)
        self.place(VGroup(VGroup(cap, g), key, rates, VGroup(bc_t, bc).arrange(DOWN, buff=0.2, aligned_edge=LEFT))
                   .arrange(DOWN, buff=0.45))
        with self.narrate(
            "Why does the token mixer decide how much memory a conversation needs? "
            "Every fourth layer is full attention, 16 in all, and their KV cache grows by 4 KiB per token in each. "
            "The other 48 use gated delta linear attention. "
            "They keep a fixed recurrent state instead: 147.8 MiB per sequence, whatever the context. "
            "Full attention everywhere would cost 256 KiB per token. "
            "This hybrid costs 64 KiB, plus that fixed state. "
            "Past about 788 tokens, the hybrid is ahead, and the gap keeps widening. "
            "At 32k context, it needs 2.14 GiB per sequence, against 8."
        ) as t:
            self.play(FadeIn(g), FadeIn(cap), run_time=0.7)
            t.wait_until("Every fourth layer")
            self.play(LaggedStart(*[c.square.animate.set_fill(th.tint(self.color, 0.55)) for c in full],
                                  lag_ratio=0.08), FadeIn(key[0]), run_time=1.4)
            t.wait_until("The other 48")
            self.play(LaggedStart(*[c.square.animate.set_fill(th.tint(STATE, 0.35)) for c in lin],
                                  lag_ratio=0.015), run_time=1.2)
            t.wait_until("They keep a fixed")
            self.play(FadeIn(key[1]), run_time=0.6)
            t.wait_until("Full attention everywhere")
            self.play(FadeIn(rates), run_time=0.8)
            t.wait_until("At 32k context")
            self.play(FadeIn(bc_t), FadeIn(bc.labels), *bc.grow(run_time=1.0), run_time=1.0)
            self.play(FadeIn(bc.values), run_time=0.4)

    # What you build ---------------------------------------------------------------

    def course_map(self):
        self.slide("course-map", "Six parts, each ending in something that works",
                   book_section="s-what-you-build")
        specs = [
            ("Part 1 · Ground truth", "Weights and arithmetic", "Chapters 1 and 2", MEM),
            ("Part 2 · A forward pass", "Match the reference", "Chapters 3 to 8", MEM),
            ("Part 3 · Making it fast", "Cache, roofline, sampling", "Chapters 9 to 11", self.color),
            ("Part 4 · Kernels", "CUDA, then Triton", "Chapters 12 to 15", self.color),
            ("Part 5 · Serving", "Schedule, benchmark", "Chapters 16 and 17", STATE),
            ("Part 6 · Scaling", "Quantize, shard, speculate", "Chapters 18 to 20", STATE),
        ]
        cards = [card(*s) for s in specs]
        grid = VGroup(VGroup(*cards[:3]).arrange(RIGHT, buff=0.3), VGroup(*cards[3:]).arrange(RIGHT, buff=0.3))
        grid.arrange(DOWN, buff=0.35)
        foot = kit.prose(r"Part 2 is your last free ground truth: your logits against Hugging Face "
                         r"\texttt{transformers}.", size=th.SIZE_SMALL, color=th.INK_3, width=12.0)
        self.place(VGroup(grid, foot).arrange(DOWN, buff=0.5))
        with self.narrate(
            "It has six parts, and each ends with something that works. "
            "Part 1 is ground truth: read the weights, and do the arithmetic. "
            "Part 2 builds a forward pass, and checks its logits against Hugging Face transformers, your last free "
            "ground truth. "
            "Part 3 makes it fast: the cache, the roofline, and sampling. "
            "Part 4 replaces library calls with your own CUDA and Triton kernels. "
            "Part 5 serves many users at once. "
            "Part 6 scales past one card and one token per step."
        ) as t:
            for i, cue in enumerate(["Part 1 is", "Part 2 builds", "Part 3 makes", "Part 4 replaces", "Part 5 serves",
                                     "Part 6 scales"]):
                t.wait_until(cue)
                anims = [FadeIn(cards[i], shift=UP * 0.1)]
                if i == 1:
                    anims.append(FadeIn(foot))
                self.play(*anims, run_time=0.6)

    def engine_step(self):
        self.slide("engine-step", "By the end, one step that repeats", book_section="s-what-you-build")
        server = kit.box("Server", sublabel="a request arrives", width=2.6, height=1.3)
        sched = kit.box("Scheduler", sublabel="who joins the batch", color=STATE, width=3.6, height=1.3)
        sampler = kit.box("Sampler", sublabel="one token per sequence", color=FLOP_C, width=3.6, height=1.3)
        rect = RoundedRectangle(width=4.2, height=3.6, corner_radius=0.1, stroke_color=self.color,
                                stroke_width=th.STROKE_BOX, fill_color=th.tint(self.color, 0.08), fill_opacity=1)
        r_t = kit.label("Model runner", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
        r_s = kit.label("embeddings, 64 layers", size=th.SIZE_LABEL, color=th.INK_3)
        kv = kit.box("paged KV cache", width=3.4, height=0.8, color=MEM, size=th.SIZE_SMALL)
        st = kit.box("state slots", width=3.4, height=0.8, color=MEM, size=th.SIZE_SMALL)
        inner = VGroup(VGroup(r_t, r_s).arrange(DOWN, buff=0.08), kv, st).arrange(DOWN, buff=0.25)
        inner.move_to(rect)
        runner = VGroup(rect, inner)
        mid = VGroup(sched, sampler).arrange(DOWN, buff=1.5)
        fig = VGroup(server, mid, runner).arrange(RIGHT, buff=1.1)
        foot = kit.prose(r"Every chapter builds one of these nouns.", size=th.SIZE_BODY, color=self.color, width=None)
        self.place(VGroup(fig, foot).arrange(DOWN, buff=0.45))
        a1 = kit.arrow(server, sched)
        a2 = kit.arrow(sched, runner)
        a3 = kit.arrow(runner, sampler)
        a4 = kit.arrow(sampler, sched, text="append, free, admit", side="right")
        with self.narrate(
            "By the end, the pieces form one step that repeats. "
            "A request arrives at the server. "
            "The scheduler decides whether it joins the batch, and whether its prompt is chunked. "
            "The model runner walks the 64 layers. "
            "It reads each sequence's KV blocks from the paged cache, and its recurrent state from a fixed slot. "
            "The sampler turns the final logits into one token per sequence. "
            "The scheduler appends those tokens, frees finished sequences, admits waiting ones, and runs the next "
            "step. "
            "Every chapter builds one of those nouns."
        ) as t:
            t.wait_until("A request arrives")
            self.play(FadeIn(server), run_time=0.6)
            t.wait_until("The scheduler decides")
            self.play(GrowArrow(a1.line), FadeIn(sched), run_time=0.8)
            t.wait_until("The model runner")
            self.play(GrowArrow(a2.line), FadeIn(rect), FadeIn(inner[0]), run_time=0.8)
            t.wait_until("It reads each")
            self.play(FadeIn(kv, shift=UP * 0.1), FadeIn(st, shift=UP * 0.1), run_time=0.7)
            t.wait_until("The sampler turns")
            self.play(GrowArrow(a3.line), FadeIn(sampler), run_time=0.8)
            t.wait_until("The scheduler appends")
            self.play(GrowArrow(a4.line), FadeIn(a4.label), run_time=0.8)
            dot = Dot(sched.get_center(), radius=0.12, color=self.color)
            self.play(FadeIn(dot), run_time=0.2)
            for target in (runner, sampler, sched):
                self.play(dot.animate.move_to(target.get_center()), run_time=0.5)
            self.play(FadeOut(dot), run_time=0.2)
            t.wait_until("Every chapter builds")
            self.play(FadeIn(foot), run_time=0.6)

    # How a lab run reaches the GPU -----------------------------------------------

    def lab_run(self):
        self.slide("lab-run", "A lab run: your file goes to an A100 and back",
                   book_section="s-how-a-lab-run-reaches-the-gpu")
        specs = [("Browser", "you click Run"), ("Backend", "file + harness"), ("RunPod", "serverless endpoint"),
                 ("A100 80GB", "run(submission)")]
        boxes = VGroup(*[kit.box(n, sublabel=s, width=2.6, height=1.15, color=self.color if i == 3 else None)
                         for i, (n, s) in enumerate(specs)]).arrange(RIGHT, buff=0.75)
        gap = Rectangle(width=1, height=1.0, stroke_width=0, fill_opacity=0)
        panel = kit.code(HARNESS, "text", line_numbers=False, caption="harness output")
        self.place(VGroup(boxes, gap, panel).arrange(DOWN, buff=0.1))
        arrows = VGroup(*[kit.arrow(a, b) for a, b in zip(boxes[:-1], boxes[1:])])
        y = boxes.get_bottom()[1] - 0.3
        back = Arrow([boxes[3].get_x(), y, 0], [boxes[0].get_x(), y, 0], buff=0, color=self.color,
                     stroke_width=th.STROKE_BOX + 1, tip_length=0.2)
        back_t = kit.label("output streams back, line by line", size=th.SIZE_LABEL, color=self.color)
        back_t.next_to(back, DOWN, buff=0.1)
        with self.narrate(
            "What happens when you run a lab? "
            "When you click Run, the backend packages your file with the lab's test harness. "
            "It ships both to a RunPod serverless endpoint. "
            "There, a container with an A100 imports your file, and calls the harness's run function. "
            "Output streams back line by line, so a long benchmark shows its progress. "
            "The harness prints one pass or fail line per check. "
            "Correctness comes before speed, always. "
            "A fast kernel with wrong answers fails, and that happens to everyone."
        ) as t:
            t.wait_until("When you click")
            self.play(FadeIn(boxes[0]), run_time=0.5)
            self.play(GrowArrow(arrows[0].line), FadeIn(boxes[1]), run_time=0.6)
            t.wait_until("It ships both")
            self.play(GrowArrow(arrows[1].line), FadeIn(boxes[2]), run_time=0.6)
            t.wait_until("There, a container")
            self.play(GrowArrow(arrows[2].line), FadeIn(boxes[3]), run_time=0.6)
            t.wait_until("Output streams")
            self.play(GrowArrow(back), FadeIn(back_t), run_time=0.8)
            t.wait_until("The harness prints")
            self.play(FadeIn(panel.frame), run_time=0.4)
            self.play(*panel.reveal(1), run_time=0.4)
            self.play(*panel.reveal(2, 3), run_time=0.5)
            self.play(*panel.reveal(5), run_time=0.4)
            t.wait_until("Correctness comes")
            self.play(panel.highlight(2, 3, color=th.tint(th.BAD, 0.14)), run_time=0.6)

    # What goes wrong ----------------------------------------------------------------

    def what_goes_wrong(self):
        self.slide("what-goes-wrong", "Three ways to get the ridge point wrong", book_section="s-what-goes-wrong")
        task = kit.prose(r"The lab: write \texttt{gpu\_report} and \texttt{ridge\_point}; land within 2 of 161.2.",
                         size=th.SIZE_SMALL, width=None)
        cards = [
            ("Rated isn't measured", r"Rated at 1935 GB/s; a copy measures 1275. Judge kernels against 1275."),
            ("Mixed units", r"312 / 1935 = 0.161. In base units, the ridge point is 161."),
            ("Two A100 variants", r"The SXM4 module is rated at 2039 GB/s: its ridge point is 153."),
        ]
        boxes = VGroup(*[kit.callout("warning", body, title=title, width=11.5, size=th.SIZE_SMALL)
                         for title, body in cards]).arrange(DOWN, buff=0.3)
        self.place(VGroup(task, boxes).arrange(DOWN, buff=0.45, aligned_edge=LEFT))
        with self.narrate(
            "The lab wants a device report, and a ridge point within 2 of 161.2. "
            "First, comparing against the rated bandwidth. "
            "Judge kernels against the 1275 GB/s a copy measures, or good kernels look broken. "
            "Second, mixing units. "
            "Divide 312 by 1935 without base units, and you get 0.161 instead of 161. "
            "Third, assuming every run gets the same card. "
            "The SXM4 module is rated at 2039 GB/s, so its ridge point is 153."
        ) as t:
            self.play(FadeIn(task), run_time=0.7)
            for b, cue in zip(boxes, ["First, comparing", "Second, mixing", "Third, assuming"]):
                t.wait_until(cue)
                self.play(FadeIn(b, shift=UP * 0.1), run_time=0.7)
