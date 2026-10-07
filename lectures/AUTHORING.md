# Write a lecture

This guide is for anyone writing a lecture scene: one narrated Manim video
per chapter. Read it in full before you write your first scene.

A lecture is a run of slides. Each slide is one Manim section with its own
narration. The build turns the video into a web video, one still image per
slide (the slide's final frame), captions, and a manifest that the site reads.
The site also uses the slide images as a deck, so **every slide's last frame
must be a finished, readable slide on its own**.

`lectures/scenes/02-memory-arithmetic.py` is a four-slide smoke test of the
kit, not a model lecture. Replace it with the real lecture.

## What a lecture teaches

- Teach the chapter, `content/chapters/<slug>.md`, in its order, with its
  scenario and its numbers. You can add a picture the text only describes, or
  draw your own version of a ```` ```viz ```` figure, but never a result the
  chapter doesn't state or a condition it leaves out.
- Every formula, number, and claim must be correct and must match the chapter.
  Compute every number you show or say, and record each one, with how you got
  it, in the scene's module docstring.
- Use the chapter's notation: $d$, $h_{kv}$, $d_h$, $N$, $L$, and so on, as the
  chapter's tables define them. Use its units: decimal GB for weights, binary
  KiB, MiB, and GiB for caches.
- Animate the idea instead of reading a page aloud. A good slide shows
  something prose can't: a cache filling one column per token, a tile moving
  across a matrix, a roofline with the operation landing on one side of the
  ridge.

A lecture runs 10 to 15 minutes, in 14 to 22 slides. The build prints the
length.

## The shape of a scene

A scene lives at `lectures/scenes/<slug>.py` and defines one class named
`Lecture`. The kit reads the slug from the file name, and the chapter's
title, part, and summary from its front matter. The closing slide names the
next chapter in course order.

The following example is a complete three-slide lecture:

```python
"""Chapter 02, memory arithmetic.

Numbers: 2 x 4 x 256 x 2 = 4096 bytes per token per full-attention layer;
64 / 4 = 16 such layers; 4096 x 16 = 65,536 bytes = 64 KiB per token.
"""

from manim import *

import kit
from kit import theme as th

KV = """\
per_layer = 2 * kv_heads * head_dim * elem   # 4096
per_token = per_layer * full_layers          # 65536
"""


class Lecture(kit.Lecture):
    def construct(self):
        self.title_slide()

        self.slide("kv-per-token", "Every token adds 64 KiB", book_section="s-kv-cache")
        panel = kit.code(KV, "python")
        self.place(panel)
        with self.narrate(
            "Each token adds a key and a value per KV head: 4096 bytes in one layer. "
            "Sixteen layers keep a cache, so a token costs 64 KiB."
        ) as t:
            self.play(FadeIn(panel.frame), *panel.reveal(1), run_time=0.8)
            t.wait_until("Sixteen layers")
            self.play(*panel.reveal(2), panel.highlight(2), run_time=0.8)

        self.closing_slide([r"A token costs 64 KiB of KV cache."])
```

## The kit

Import everything from `kit`. The kit's docstrings are the full reference;
this section is the map.

### Scene methods

| Method | What it does |
|---|---|
| `self.slide(id, title, book_section=None, keep=())` | Starts a slide: finishes the previous one, starts a Manim section, fades out the previous slide's mobjects, and draws the header. |
| `self.title_slide(narration=None, summary=None)` | The opening slide: chapter number, part, title, summary, and the course badge. |
| `self.closing_slide(recap, narration=None, next_hook=None, book_section=None)` | The closing slide: what to keep, and the next chapter's title and summary. |
| `with self.narrate(text, say=None) as t:` | Speaks `text` over the animations inside the block, then waits until the voice finishes. |
| `self.place(mob, where="center")` | Scales `mob` down to fit the content area, then positions it. `where` also takes `left`, `right`, `top`, and `top-left`. |
| `self.pause(seconds)` | Holds the frame with no narration. |
| `self.color`, `self.colors` | The course accent, and four data-series colours starting with it. |

The narration handle `t` has `t.duration`, `t.elapsed`, and `t.remaining`,
and the following methods. Each takes a fragment of a sentence's text that
matches exactly one sentence, or a sentence index:

- `t.wait_until("fragment")` holds the frame until that sentence starts.
- `t.until("fragment")` returns the seconds until that sentence starts, for a
  `run_time`.
- `t.until_end("fragment")` returns the seconds until that sentence ends.

We recommend fragments over indices, because an index shifts silently when
you add a sentence earlier in the block. A fragment that matches no sentence,
or more than one, stops the render with the list of sentences.

### Code panels

`kit.code(src, language, highlight_lines=(), size=th.SIZE_CODE, caption=None)`
sets source in the site's code font and syntax colours, one addressable row
per line. Languages are Pygments names: `python` (also for Triton), `cuda`,
`cpp`, `c`, `bash`, and `json`.

| Member | Use |
|---|---|
| `panel.frame` | The background, line numbers, and caption, without the code. Fade it in first to reveal lines one at a time. |
| `panel.line(n)`, `panel.lines(n, ...)` | Line `n`, numbered from 1 as the panel shows it. |
| `panel.reveal(n, ...)` | A list of `FadeIn`s for those lines. |
| `panel.highlight(n, ...)` | An animation that fades in a band behind those lines. |
| `panel.focus(n, ...)` | An animation that dims every other line. `focus()` with no lines undims. |

A code slide shows 14 lines or fewer, each about 64 characters or fewer.
Cut an excerpt down to the lines the narration talks about, and use
`first_line=` to keep the chapter's numbering.

### Diagrams

| Block | Use |
|---|---|
| `kit.box(text, sublabel=None, color=None, width=None, height=None)` | A labelled box: a GPU, a buffer, a layer, a process. `.rect`, `.text`, `.sub`. |
| `kit.arrow(a, b, text=None, side="auto")` | An arrow between the facing edges of two mobjects (or points), with an optional label. `.line`, `.label`. |
| `kit.grid(rows, cols, cell=0.5, values=None, colors=None, gap=0, row_labels=None, col_labels=None)` | A tensor, memory, page table, or tiling. Address cells with `g.cell(r, c)`, `g.row(r)`, `g.col(c)`, and `g.block(r0, r1, c0, c1)`; animate with `g.fill(cells, color)`; frame a tile with `g.outline(...)`. |
| `kit.bar_compare(items, unit, limit=None, limit_label=None)` | Horizontal bars that compare a few numbers, each printed at the bar's end, with an optional capacity line. `bc.grow()` returns the grow animations. |

### Text, maths, and plots

| Block | Use |
|---|---|
| `kit.label(text, size, color, font, weight)` | One line of Pango text with no maths: labels and legends. |
| `kit.mono(text)` | One line in the code font, with no highlighting. |
| `kit.heading(text)` | One serif semibold line. |
| `kit.prose(tex, size, color, width)` | A wrapped serif paragraph with inline `$...$` maths and `\texttt{...}`, set by XeLaTeX. `width=None` sets one line. |
| `kit.formula(*parts, size, colors)` | Display maths (`MathTex`). Split the TeX into parts to address pieces later. |
| `kit.bullets(items)`, `kit.build(rows)` | A bulleted list, and one reveal animation per row. |
| `kit.callout(kind, body, title=None)` | A tinted box: `definition`, `theorem`, `result`, `key`, `example`, `warning`, `note`, `assumption`. |
| `kit.Derivation(steps, mode="stack")` | A multi-line derivation aligned on `&`. `d.show(i)` reveals step `i`. |
| `kit.table(rows, header, align)` | A booktabs-style table. `tab.cells[r]` is row `r`. |
| `kit.mark(mob)`, `kit.underline(mob, color)`, `kit.pill(text, color)` | Emphasis. |
| `kit.axes(...)`, `kit.bars(...)`, `kit.scatter(...)`, `kit.legend(...)`, `kit.vline(...)` | Plots in the site style. |
| `kit.rng(seed)` | A seeded generator. Give every simulation its own seed. |

`kit.theme` (imported as `th`) holds the colour roles (`th.INK`, `th.INK_3`,
`th.RULE`, `th.ACCENT`, `th.GOOD`, `th.BAD`, `th.INFO`, `th.VIOLET`,
`th.MARK`, and the syntax colours), the type scale (`th.SIZE_BODY`,
`th.SIZE_SMALL`, `th.SIZE_LABEL`, `th.SIZE_CODE`, `th.SIZE_MATH`), and the
frame geometry (`th.SAFE_LEFT`, `th.CONTENT_TOP`, and so on). Name a role,
never a hex value. The colours come from the site's light theme in
`frontend/src/index.css`; the accent and fonts come from `course.json`.

## Conventions

### One idea per slide

- A slide makes one point. If its narration needs two paragraphs, it's two
  slides.
- At most about 40 words of on-screen prose, one display formula plus one
  derivation, one code panel of 14 lines or fewer, or one diagram plus three
  short lines. The narration carries the explanation; the slide carries what
  the viewer needs to see.
- Nothing smaller than `th.SIZE_LABEL`. Sentences use `th.SIZE_BODY` or
  `th.SIZE_SMALL`.
- Slide titles are sentence case, fit on one line, and name the point:
  "Every token adds 64 KiB," not "KV cache."

### Slide ids and book sections

- A slide id is lowercase words joined by hyphens, unique in the lecture:
  `kv-per-token`. Don't rename an id once a lecture is published.
- `book_section` is the id of the chapter's `##` section that the slide
  teaches: `s-` followed by the heading slugified. To slugify, lowercase the
  heading; drop `` ` ``, `*`, `_`, `$`, and `\`; replace each run of other
  non-alphanumeric characters with `-`; and trim hyphens from both ends. For
  example, `## KV cache` becomes `s-kv-cache`, and
  `## The largest batch that fits` becomes `s-the-largest-batch-that-fits`.
  The build fails if a `book_section` doesn't match a heading, and lists the
  valid ids.
- The title slide has no `book_section`.

### Narration

- Write narration as plain spoken sentences, in US spelling, following the
  chapter's style guide (`content/STYLE.md`). Each block splits into
  sentences at `.`, `!`, or `?` followed by a space and a capital letter or
  digit, so `0.25` and `e.g. this` stay whole.
- Narration is also the caption text and the slide's speaker notes, so write
  it as a reader should see it: `KV`, `64 KiB`, `bf16`, `σ²`, `24 / 4`,
  `3 ×`. The kit reads symbols aloud through `PRONUNCIATIONS` in
  `kit/narration.py`, then this course's jargon through
  `lectures/pronunciations.json`: "K V", "64 kibibytes", "B F sixteen".
- If the voice would misread something in one place, pass `say=` with the
  same number of sentences. If other chapters need the same fix, add a rule
  to `pronunciations.json`: a `[regex, replacement]` pair, applied in file
  order after the shared rules.
- The voice is VibeVoice-1.5B, a language-model voice that reads decimals and
  most words correctly. It speaks at about 2.6 words a second.
- Write for the ear: short sentences, one number per sentence, and say what a
  result means, not only what it is. Round long numbers in speech ("about 54
  gigabytes") and keep the exact value on screen.
- Speak to the viewer as "you." Don't use the words `simply`, `just`,
  `obviously`, or `easily`.
- Put every animation on a slide inside a `narrate` block, and sync each one
  to the sentence that talks about it with `t.wait_until("...")`.

### The settled final frame

The kit guarantees the following rules, and the build checks them on the
rendered video:

- `slide()` clears the previous slide at the start of the new section, never
  at the end of the old one.
- Every section ends on a frame held still for at least 1.2 seconds. The kit
  adds the hold if the scene didn't leave one, and strips updaters before it.
- The slide image is that final frame. The build checks that the last second
  of each section doesn't change and that the saved image matches the video.

The following rules are yours to follow:

- End each slide on its complete state. Don't fade things out at the end of a
  slide; the next `slide()` does that.
- Don't call `self.next_section()`, and don't play anything before the first
  `slide()` (`title_slide()` is a slide).
- Don't leave anything outside the safe area. A mobject outside it fails the
  render.
- Build fresh mobjects on each slide. To carry a mobject over to the next
  slide, pass it in `keep=`.

### Layout traps

- **Group before you place.** A curve, label, arrow, or band built from
  another mobject's position stays where it was drawn if you then move that
  mobject with `self.place()` or `arrange()`. Put every derived mobject in
  one `VGroup` with its source before placing, or create the derived
  mobjects after placing. `kit.arrow` and `panel.highlight` read positions
  when you call them, so call them after `place()`.
- `kit.label` and slide titles use Pango, which can't set LaTeX. Use Unicode
  in titles: "Why h_kv = 4 matters" reads better as "Why 4 KV heads matter."
- `kit.prose` is LaTeX: escape `%` as `\%`, `&` as `\&`, `#` as `\#`, and `_`
  as `\_` outside maths. Use `\textbf{...}`, `\textit{...}`, and
  `\texttt{...}`.
- `always_redraw` and `ValueTracker` work, and the kit freezes them at the end
  of the slide.

## Build and preview

Run every command from the repository root. The first run builds the Docker
image, which takes a few minutes; after that, each run starts in seconds.
Docker must be running (`open -a Docker`).

To check the layout without the speech server, render only the final frame
of each slide, with estimated narration:

```bash
LECTURE_TTS=estimate python3 lectures/build.py 02-memory-arithmetic --frames
```

This takes under a minute. The frames land in
`lectures/.build/<slug>/<version>-preview/`: `render/frames/*.png` at full
resolution, and `contact.png`, with every slide on one sheet. In this mode,
layout problems print as warnings instead of stopping the render.

To watch the lecture as a low-quality video with silent, estimated narration,
run the following command:

```bash
LECTURE_TTS=estimate python3 lectures/build.py 02-memory-arithmetic --quality l
```

To watch a few slides only, add `--only kv-per-token,cache-grows`. With
`LECTURE_TTS_URL` set instead of `LECTURE_TTS=estimate`, the same commands
use the real voice, and the sentences they synthesise stay cached for the
final render.

To render the final 1080p video, set the speech server and omit `--quality`.
To publish, see `lectures/README.md`.

### Inspect the slides

Open `contact.png` first, then each full-resolution frame that looks crowded.
In Claude Code, open them with the Read tool, which displays images. Look for
the following problems:

- Text touching or overlapping other text, a box, or an arrow.
- Anything cut off at an edge.
- A label far from the thing it labels, or a band or arrow left behind by
  **Group before you place**.
- A slide that reads as half-built, because an element the narration mentions
  is missing.
- Text or code that `place()` scaled down until it's hard to read. Cut content
  instead.

The build also prints warnings for text that overlaps text from another
mobject. Treat every warning as a problem to fix.

## Self-review checklist

Before you hand in a lecture, confirm each of the following:

- [ ] The lecture teaches the chapter in its order, with its numbers, and says
      nothing the chapter doesn't.
- [ ] Every number on screen and in narration is computed, matches the
      chapter, and is recorded in the module docstring.
- [ ] The lecture runs 10 to 15 minutes, in 14 to 22 slides.
- [ ] Each slide makes one point, and at least half the slides animate
      something other than text appearing.
- [ ] Every code panel shows 14 lines or fewer.
- [ ] Every slide id is unique and hyphenated, and every `book_section`
      resolves.
- [ ] Every animation is synced to its sentence with a text fragment.
- [ ] `--frames` shows no warnings, and you've looked at every frame: nothing
      clipped, nothing overlapping, nothing half-built.
- [ ] You've listened to at least the two densest slides with the real voice
      at low quality, and the voice matches what appears.
- [ ] Narration has no LaTeX and no banned words, and every acronym reads
      correctly.
- [ ] Every simulation uses `kit.rng` with its own seed.
