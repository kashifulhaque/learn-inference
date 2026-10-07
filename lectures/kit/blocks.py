"""Slide building blocks: bullets, callout boxes, derivations, tables, cards.

Each function returns a mobject laid out at the origin. It does not add it to
the scene or animate it; position it, then play it in. The few helpers that
return animations say so.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
from manim import (
    DOWN,
    LEFT,
    MEDIUM,
    RIGHT,
    SEMIBOLD,
    UP,
    UR,
    Animation,
    AnimationGroup,
    Circle,
    FadeIn,
    FadeOut,
    Line,
    Mobject,
    Rectangle,
    RoundedRectangle,
    TransformMatchingTex,
    VGroup,
    VMobject,
)

from . import theme as th
from .text import formula, heading, label, prose

# Fitting -------------------------------------------------------------------


def fit(mob: Mobject, width: float | None = None, height: float | None = None) -> Mobject:
    """Scales `mob` down, never up, until it fits `width` by `height`."""
    width = th.SAFE_WIDTH if width is None else width
    height = th.CONTENT_HEIGHT if height is None else height
    factor = min(1.0, width / max(mob.width, 1e-6), height / max(mob.height, 1e-6))
    if factor < 1.0:
        mob.scale(factor)
    return mob


# Bullets -------------------------------------------------------------------


def bullets(
    items: Sequence[str | Mobject],
    size: float = th.SIZE_BODY,
    width: float = th.MEASURE,
    marker_color=th.INK_4,
    buff: float = 0.34,
) -> VGroup:
    """A bulleted list. Each item is prose (inline maths allowed) or a mobject.

    Returns a VGroup of rows; `rows[i]` is item `i` with its marker, ready for
    `build(rows)` to reveal one at a time.
    """
    rows = VGroup()
    indent = 0.42
    for item in items:
        body = prose(item, size=size, width=width - indent) if isinstance(item, str) else item
        dot = Circle(radius=0.055, fill_color=marker_color, fill_opacity=1, stroke_width=0)
        row = VGroup(dot, body)
        body.next_to(dot, RIGHT, buff=indent - 0.11)
        # Centre the marker on the first line of text, not on the paragraph.
        first_line_y = body.get_top()[1] - 0.2 * size / th.SIZE_BODY
        dot.move_to([dot.get_x(), first_line_y, 0])
        rows.add(row)
    rows.arrange(DOWN, buff=buff, aligned_edge=LEFT)
    return rows


def build(rows: VGroup, shift: float = 0.15) -> list[Animation]:
    """Returns one FadeIn per row, for revealing bullets or steps in turn."""
    return [FadeIn(row, shift=RIGHT * shift) for row in rows]


# Callout boxes -------------------------------------------------------------

CALLOUT_KINDS = {
    "definition": ("Definition", None),  # None means the course accent
    "theorem": ("Theorem", None),
    "result": ("Result", None),
    "example": ("Example", th.WARN),
    "warning": ("Watch out", th.BAD),
    "note": ("Note", th.INK_3),
    "assumption": ("Assumption", th.INFO),
    "key": ("Key idea", None),
}


def callout(
    kind: str,
    body: str | Mobject,
    title: str | None = None,
    width: float = 9.5,
    size: float = th.SIZE_BODY,
    pad: float = 0.36,
) -> VGroup:
    """A tinted box with a coloured left rule and a kind label.

    Args:
      kind: One of `CALLOUT_KINDS`: definition, theorem, result, example,
        warning, note, assumption, key.
      body: Prose (inline maths allowed) or any mobject, such as a formula.
      title: Replaces the kind's default label, for example "Why decode waits".
        It may hold inline maths (`"Mean, $\\sigma$ unknown"`).
      width: Outer width in Manim units.
    """
    default_title, color = CALLOUT_KINDS[kind]
    color = th.ACCENT if color is None else color
    name = title or default_title
    # A title with maths goes through LaTeX; plain titles use the sans font.
    if "$" in name:
        tag = prose(rf"\textbf{{{name}}}", size=th.SIZE_LABEL, color=color, width=None)
    else:
        tag = label(name, size=th.SIZE_LABEL, color=color, weight=SEMIBOLD)
    inner = width - 2 * pad - 0.08
    content = prose(body, size=size, width=inner) if isinstance(body, str) else fit(body, width=inner, height=6)
    column = VGroup(tag, content).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
    height = column.height + 2 * pad
    panel = RoundedRectangle(
        width=width,
        height=height,
        corner_radius=0.1,
        fill_color=th.tint(color, 0.07),
        fill_opacity=1,
        stroke_width=0,
    )
    bar = Rectangle(width=0.07, height=height - 0.2, fill_color=color, fill_opacity=1, stroke_width=0)
    bar.move_to(panel.get_left() + RIGHT * (0.035 + 0.02))
    column.move_to(panel.get_center()).align_to(panel, LEFT).shift(RIGHT * (pad + 0.08))
    box = VGroup(panel, bar, column)
    box.panel, box.tag, box.content = panel, tag, content
    return box


# Emphasis ------------------------------------------------------------------


def mark(mob: Mobject, color=th.MARK, buff: float = 0.08) -> Rectangle:
    """A highlighter band behind `mob`, as the site marks a term.

    Add it to the scene before (under) the thing it marks, or play
    `FadeIn(mark(m))`; the kit gives it a low z-index.
    """
    band = Rectangle(
        width=mob.width + 2 * buff,
        height=mob.height + 2 * buff,
        fill_color=color,
        fill_opacity=1,
        stroke_width=0,
    ).move_to(mob)
    band.set_z_index(-1)
    return band


def underline(mob: Mobject, color, buff: float = 0.08, stroke: float = 4) -> Line:
    """A coloured rule under `mob`."""
    return Line(mob.get_corner(DOWN + LEFT), mob.get_corner(DOWN + RIGHT), color=color, stroke_width=stroke).shift(
        DOWN * buff
    )


def pill(text: str, color, size: float = th.SIZE_LABEL, filled: bool = True) -> VGroup:
    """A rounded badge, such as the course name on the title card."""
    word = label(text, size=size, color=th.CANVAS if filled else color, weight=MEDIUM)
    box = RoundedRectangle(
        width=word.width + 0.5,
        height=word.height + 0.3,
        corner_radius=(word.height + 0.3) / 2,
        fill_color=color,
        fill_opacity=1 if filled else 0,
        stroke_color=color,
        stroke_width=0 if filled else 2,
    )
    word.move_to(box)
    return VGroup(box, word)


# Derivations ---------------------------------------------------------------


class Derivation(VGroup):
    """A multi-line derivation, revealed one line at a time.

    Write each step as one TeX string. Put `&` before the relation to align
    the lines on it (`r"\\sum Z_i^2 &= \\dots"`), as in an `aligned` block.

    Two modes:

    - `stack` (default): the lines sit under each other, aligned on the
      relation. `show(i)` fades line `i` in and dims the one before it, so
      the eye lands on the new step while the whole argument stays visible.
    - `transform`: every line sits in the same place. `show(i)` morphs line
      `i-1` into line `i` with `TransformMatchingTex`, so the symbols that
      survive a step visibly stay put.

    Example:
      d = Derivation([r"M_{\\bar X}(t) &= M_{\\sum X_i}(t/n)",
                      r"&= \\exp(\\mu t + \\sigma^2 t^2 / 2n)"]).move_to(ORIGIN)
      self.play(d.show(0)); self.play(d.show(1))
    """

    def __init__(
        self,
        steps: Sequence[str],
        mode: str = "stack",
        size: float = th.SIZE_MATH_SMALL,
        color=th.INK,
        buff: float = 0.32,
        dim: bool = True,
    ):
        super().__init__()
        if mode not in ("stack", "transform"):
            raise ValueError("mode must be 'stack' or 'transform'")
        self.mode = mode
        self.dim_previous = dim
        self.lines: list[VMobject] = []
        for step in steps:
            if "&" in step:
                left, right = step.split("&", 1)
                # MathTex needs a non-empty first part; a blank lhs keeps the
                # relation in the aligned column.
                line = formula(left.strip() or r"\phantom{=}", right.strip(), size=size, color=color)
                if not left.strip():
                    line[0].set_opacity(0)
                line.pivot = line[1]
            else:
                line = formula(step, size=size, color=color)
                line.pivot = None
            self.lines.append(line)
        if mode == "stack":
            VGroup(*self.lines).arrange(DOWN, buff=buff, aligned_edge=LEFT)
            pivoted = [l for l in self.lines if l.pivot is not None]
            if pivoted:
                col = max(l.pivot.get_left()[0] for l in pivoted)
                for line in pivoted:
                    line.shift(RIGHT * (col - line.pivot.get_left()[0]))
        self.add(*self.lines)
        self._shown = -1

    def show(self, i: int, run_time: float | None = None) -> Animation:
        """Returns the animation that reveals step `i`."""
        line = self.lines[i]
        kwargs = {} if run_time is None else {"run_time": run_time}
        prev = self.lines[self._shown] if self._shown >= 0 else None
        self._shown = i
        if self.mode == "transform" and prev is not None:
            line.move_to(prev)
            return TransformMatchingTex(prev, line, **kwargs)
        anims = [FadeIn(line, shift=DOWN * 0.12)]
        if self.mode == "stack" and prev is not None and self.dim_previous:
            anims.append(prev.animate.set_color(th.INK_3))
        return AnimationGroup(*anims, **kwargs)


# Tables --------------------------------------------------------------------


def _cell(value, size: float, color, bold: bool = False) -> Mobject:
    if isinstance(value, Mobject):
        return value
    text = str(value)
    if bold:
        return label(text, size=size, color=color, weight=SEMIBOLD) if "$" not in text else prose(
            text, size=size, color=color, width=None
        )
    return prose(text, size=size, color=color, width=None)


def table(
    rows: Sequence[Sequence],
    header: Sequence | None = None,
    size: float = th.SIZE_SMALL,
    align: str | Sequence[str] = "l",
    col_gap: float = 0.6,
    row_gap: float = 0.26,
    highlight_rows: Sequence[int] = (),
    highlight_color=th.MARK,
) -> VGroup:
    """A booktabs-style table: a rule above, under the header, and below.

    Cells are strings (prose, inline maths allowed) or mobjects. `align` is
    one of `l`, `c`, `r`, or one per column. Returns a VGroup with
    attributes `cells` (list of rows), `rules`, and `bands` (highlights).
    """
    all_rows = ([list(header)] if header is not None else []) + [list(r) for r in rows]
    ncols = max(len(r) for r in all_rows)
    aligns = [align] * ncols if isinstance(align, str) else list(align)
    grid: list[list[Mobject]] = []
    for r, row in enumerate(all_rows):
        is_head = header is not None and r == 0
        grid.append(
            [
                _cell(v, size, th.INK_3 if is_head else th.INK_2, bold=is_head)
                for v in row + [""] * (ncols - len(row))
            ]
        )
    col_w = [max(grid[r][c].width for r in range(len(grid))) for c in range(ncols)]
    row_h = [max(max(m.height for m in row), size * 0.018) for row in grid]
    total_w = sum(col_w) + col_gap * (ncols - 1)
    y = 0.0
    out = VGroup()
    rules = VGroup()
    bands = VGroup()
    top_rule_y = row_gap
    for r, row in enumerate(grid):
        h = row_h[r]
        cy = y - h / 2
        x = -total_w / 2
        for c, cell in enumerate(row):
            a = aligns[c]
            if a == "l":
                cell.move_to([x + cell.width / 2, cy, 0])
            elif a == "r":
                cell.move_to([x + col_w[c] - cell.width / 2, cy, 0])
            else:
                cell.move_to([x + col_w[c] / 2, cy, 0])
            out.add(cell)
            x += col_w[c] + col_gap
        data_index = r - (1 if header is not None else 0)
        if data_index in highlight_rows:
            band = Rectangle(
                width=total_w + 0.3, height=h + row_gap, fill_color=highlight_color, fill_opacity=1, stroke_width=0
            ).move_to([0, cy, 0])
            band.set_z_index(-1)
            bands.add(band)
        y -= h + row_gap
        if header is not None and r == 0:
            rules.add(
                Line([-total_w / 2 - 0.15, y + row_gap / 2, 0], [total_w / 2 + 0.15, y + row_gap / 2, 0],
                     color=th.RULE_STRONG, stroke_width=th.STROKE_HAIRLINE)
            )
    rules.add(
        Line([-total_w / 2 - 0.15, top_rule_y, 0], [total_w / 2 + 0.15, top_rule_y, 0], color=th.INK_3,
             stroke_width=th.STROKE_AXIS)
    )
    rules.add(
        Line([-total_w / 2 - 0.15, y + row_gap / 2, 0], [total_w / 2 + 0.15, y + row_gap / 2, 0], color=th.INK_3,
             stroke_width=th.STROKE_AXIS)
    )
    group = VGroup(bands, rules, out)
    group.cells = grid
    group.rules = rules
    group.bands = bands
    group.move_to([0, 0, 0])
    return group


# Cards ---------------------------------------------------------------------


def title_card(
    title: str,
    summary: str,
    chapter: str,
    part: str | None = None,
    course: str | None = None,
    color=th.ACCENT,
) -> VGroup:
    """The opening card: chapter number and part, title, summary, course badge.

    Args:
      title: The chapter title.
      summary: The chapter's one-sentence summary, set in italics.
      chapter: The kicker's chapter label, such as "Chapter 02".
      part: The part, such as "Part 1 — Ground truth".
      course: The course name for the badge; None leaves the badge off.

    Returns a VGroup with attributes `kicker`, `title`, `rule`, `summary`,
    and `badge` (an empty VGroup without a course).
    """
    kicker_text = chapter + (f"  ·  {part}" if part else "")
    kicker = label(kicker_text, size=th.SIZE_LABEL + 2, color=th.INK_4, weight=MEDIUM)
    name = label(title, size=th.SIZE_DISPLAY, color=th.INK, font=th.SERIF_DISPLAY, weight=SEMIBOLD)
    fit(name, width=th.SAFE_WIDTH - 0.4)
    ask = prose(rf"\textit{{{_tex_escape(summary)}}}", size=th.SIZE_BODY, color=th.INK_3, width=10.8)
    badge = pill(course, color, size=th.SIZE_LABEL) if course else VGroup()
    rule = Line(LEFT * 0.6, RIGHT * 0.6, color=color, stroke_width=5)
    parts = [kicker, name, rule, ask] + ([badge] if course else [])
    group = VGroup(*parts).arrange(DOWN, buff=0.42, aligned_edge=LEFT)
    fit(group, width=th.SAFE_WIDTH - 0.4, height=th.SAFE_TOP - th.SAFE_BOTTOM - 0.4)
    group.move_to([th.SAFE_LEFT + group.width / 2 + 0.4, 0.1, 0])
    group.kicker, group.title, group.rule, group.summary, group.badge = kicker, name, rule, ask, badge
    return group


def where_next(
    recap: Sequence[str],
    next_title: str | None,
    next_hook: str | None = None,
    next_label: str = "Next chapter",
    color=th.ACCENT,
) -> VGroup:
    """The closing card: what this lecture established and where to go next.

    Args:
      recap: Two to four short lines, one per idea to keep. Prose, so inline
        maths is allowed.
      next_title: The next chapter's title, or None for the last chapter.
      next_hook: One sentence on what the next chapter does, such as its
        summary. Prose; escape LaTeX specials.
      next_label: The panel's kicker, such as "Next chapter · 03".
    """
    left_head = label("What to keep", size=th.SIZE_LABEL, color=color, weight=SEMIBOLD)
    items = bullets(recap, size=th.SIZE_SMALL, width=5.9, marker_color=color, buff=0.3)
    left = VGroup(left_head, items).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
    inner = 5.6
    if next_title:
        nh = label(next_label, size=th.SIZE_LABEL - 2, color=th.INK_4, weight=SEMIBOLD)
        nt = heading(next_title, size=th.SIZE_HEADING - 4)
        fit(nt, width=inner)
        part = [nh, nt]
        if next_hook:
            part.append(prose(next_hook, size=th.SIZE_LABEL, color=th.INK_3, width=inner))
        right_col = VGroup(*part).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
    else:
        right_col = VGroup(label("End of the course", size=th.SIZE_LABEL, color=th.INK_3, weight=SEMIBOLD))
    panel = RoundedRectangle(
        width=inner + 0.7,
        height=right_col.height + 0.6,
        corner_radius=0.12,
        fill_color=th.SUNK,
        fill_opacity=1,
        stroke_width=0,
    )
    right_col.move_to(panel)
    right = VGroup(panel, right_col)
    group = VGroup(left, right).arrange(RIGHT, buff=0.6, aligned_edge=UP)
    fit(group, width=th.SAFE_WIDTH, height=th.CONTENT_HEIGHT - 0.2)
    group.left_col, group.right_col = left, right
    group.items = items
    return group


_TEX_SPECIALS = {"%": r"\%", "&": r"\&", "#": r"\#", "_": r"\_", "$": r"\$", "{": r"\{", "}": r"\}"}


def _tex_escape(text: str) -> str:
    """Escapes front-matter text for `prose`: no maths, literal specials."""
    return "".join(_TEX_SPECIALS.get(ch, ch) for ch in text)


def tex_escape(text: str) -> str:
    """Escapes plain text (such as a chapter summary) for `kit.prose`."""
    return _tex_escape(text)


def arrow_note(target: Mobject, text: str, direction=UR, color=th.INK_3, size: float = th.SIZE_LABEL) -> VGroup:
    """A short annotation placed beside `target` with a thin leader line."""
    note = prose(text, size=size, color=color, width=None)
    note.next_to(target, direction, buff=0.5)
    start = note.get_corner(-np.sign(direction) * np.array([1, 1, 0]) if np.any(direction) else DOWN)
    leader = Line(start, target.get_critical_point(direction), color=color, stroke_width=th.STROKE_HAIRLINE, buff=0.08)
    return VGroup(note, leader)


def _slug_ok(value: str) -> bool:
    return re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value) is not None


__all__ = [
    "Derivation",
    "arrow_note",
    "bullets",
    "build",
    "callout",
    "fit",
    "mark",
    "pill",
    "table",
    "tex_escape",
    "title_card",
    "underline",
    "where_next",
]
