"""Chapter 02, memory arithmetic: SMOKE TEST, not a lecture.

This four-slide scene exercises the kit (title card, code panel, grid with a
box and an arrow, closing card) so the pipeline can be checked end to end.
The pilot author replaces it with the real lecture.

Numbers, from content/chapters/02-memory-arithmetic.md, section "KV cache":

- One token in one full-attention layer: 2 (K and V) x 4 (KV heads) x 256
  (head_dim) x 2 bytes = 4096 bytes = 4 KiB.
- Full-attention layers: 64 / 4 = 16, so 4096 x 16 = 65,536 bytes = 64 KiB
  per token.
- A 4k-token conversation: 4096 x 64 KiB = 262,144 KiB = 0.25 GiB.
"""

from manim import *

import kit
from kit import theme as th

KV_CODE = """\
# One token's keys and values, in bytes
kv_heads = 4            # num_key_value_heads
head_dim = 256          # head_dim
elem = 2                # bytes per bfloat16 value
full_layers = 64 // 4   # every fourth layer: 16

per_layer = 2 * kv_heads * head_dim * elem   # 4096
per_token = per_layer * full_layers          # 65536
print(per_token / 1024, "KiB")               # 64.0 KiB
"""


class Lecture(kit.Lecture):
    def construct(self):
        self.title_slide(
            narration=(
                "Memory arithmetic. How many conversations fit on one GPU comes down to counting bytes, "
                "and this lecture counts the ones the KV cache adds."
            )
        )
        self.kv_per_token()
        self.cache_grows()
        self.closing_slide(
            [
                r"Each token costs $2 \times 4 \times 256 \times 2 = 4096$ bytes per full-attention layer.",
                r"Only 16 of the 64 layers pay it: 64 KiB per token.",
                r"A 4k-token conversation holds 0.25 GiB of cache.",
            ],
            narration=(
                "Here's what to keep. Each token costs 4096 bytes in each full-attention layer. "
                "Only 16 layers pay it, so the cache grows by 64 KiB per token. "
                "A 4k-token conversation holds a quarter of a GiB."
            ),
        )

    def kv_per_token(self):
        self.slide("kv-per-token", "One token's keys and values", book_section="s-kv-cache")
        panel = kit.code(KV_CODE, "python", caption="kv_bytes.py")
        self.place(panel)
        with self.narrate(
            "Each new token adds one key and one value for each of the 4 KV heads. "
            "Each is 256 values wide, at 2 bytes a value. "
            "That's 4096 bytes in one layer. "
            "Only every fourth layer keeps a KV cache, 16 of them, so a token costs 65,536 bytes: 64 KiB."
        ) as t:
            self.play(FadeIn(panel.frame), run_time=0.5)
            self.play(*panel.reveal(1, 2), run_time=0.8)
            t.wait_until("Each is 256")
            self.play(*panel.reveal(3, 4), run_time=0.8)
            t.wait_until("That's 4096")
            self.play(*panel.reveal(7), panel.highlight(7), run_time=0.8)
            t.wait_until("Only every fourth")
            self.play(*panel.reveal(5, 8, 9), run_time=0.8)
            self.play(panel.highlight(8, 9), run_time=0.5)

    def cache_grows(self):
        self.slide("cache-grows", "The cache grows one column per token", book_section="s-kv-cache")
        tokens = 10
        g = kit.grid(16, tokens, cell=(0.5, 0.25), gap=0.04,
                     col_labels=[f"t{i + 1}" for i in range(tokens)])
        layers = kit.label("16 full-attention layers", size=th.SIZE_LABEL, color=th.INK_3)
        layers.rotate(PI / 2).next_to(g, LEFT, buff=0.25)
        figure = VGroup(g, layers)
        note = kit.box("One token", sublabel="16 × 4 KiB = 64 KiB", color=self.color, size=th.SIZE_SMALL)
        total = kit.box("4096 tokens", sublabel="0.25 GiB", size=th.SIZE_SMALL)
        side = VGroup(note, total).arrange(DOWN, buff=1.2, aligned_edge=LEFT)
        layout = VGroup(figure, side).arrange(RIGHT, buff=1.4)
        self.place(layout)
        link = kit.arrow(note, g.col(tokens - 1), color=self.color)
        grow = kit.arrow(note, total, text="× 4096")
        with self.narrate(
            "Picture the cache as a grid. "
            "Each row is one of the 16 full-attention layers, and each column is a token. "
            "Every token you generate fills one more column, 64 KiB at a time. "
            "After 4096 tokens, that's a quarter of a GiB for one conversation."
        ) as t:
            self.play(FadeIn(g), FadeIn(layers), run_time=0.8)
            t.wait_until("Every token")
            for c in range(tokens):
                self.play(g.fill(g.col(c), self.color), run_time=0.18)
            self.play(FadeIn(note), GrowArrow(link.line), run_time=0.6)
            t.wait_until("After 4096")
            self.play(FadeIn(grow), FadeIn(total), run_time=0.8)
