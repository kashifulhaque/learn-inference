"""Systems diagrams: boxes, arrows, grids of cells, and bar comparisons.

These are the pictures an inference course draws over and over: a GPU's
memory as labelled boxes, data moving along arrows, a tensor or a KV cache
as a grid of cells, and the numbers that decide a design as bars. Every
function returns a mobject at the origin; position it, then animate it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from manim import (
    DOWN,
    LEFT,
    MEDIUM,
    RIGHT,
    UP,
    AnimationGroup,
    Arrow,
    DashedLine,
    GrowFromEdge,
    Mobject,
    Rectangle,
    RoundedRectangle,
    VGroup,
)

from . import theme as th
from .text import label, prose

# Boxes ----------------------------------------------------------------------


def box(
    text: str,
    width: float | None = None,
    height: float | None = None,
    color=None,
    sublabel: str | None = None,
    size: float = th.SIZE_SMALL,
    font: str | None = None,
    filled: bool = True,
    corner: float = 0.1,
) -> VGroup:
    """A labelled box: a GPU, a buffer, a layer, a process.

    Args:
      text: The label, one line. Text with `$` goes through LaTeX.
      width: Outer width in Manim units. Defaults to the label plus padding.
      height: Outer height. Defaults to the label plus padding.
      color: Border and tint colour. None draws a neutral white card.
      sublabel: A smaller second line, such as a size ("80 GiB").
      size: Label size in `Text` units.
      font: The label's font; defaults to the sans family. Pass `th.MONO`
        for a tensor or function name.
      filled: False leaves the box unfilled, for a dashed-style outline.
      corner: Corner radius.

    Returns:
      A VGroup with attributes `rect`, `text`, and `sub` (None without a
      sublabel).
    """
    name = prose(text, size=size, color=th.INK, width=None) if "$" in text else label(
        text, size=size, color=th.INK, font=font or th.SANS, weight=MEDIUM
    )
    parts = [name]
    sub = None
    if sublabel:
        sub = label(sublabel, size=max(th.SIZE_LABEL, size - 6), color=th.INK_3, font=font or th.SANS)
        parts.append(sub)
    body = VGroup(*parts).arrange(DOWN, buff=0.12)
    w = max(width or 0.0, body.width + 0.6)
    h = max(height or 0.0, body.height + 0.45)
    stroke = th.RULE_STRONG if color is None else color
    fill = th.RAISE if color is None else th.tint(color, 0.10)
    rect = RoundedRectangle(
        width=w,
        height=h,
        corner_radius=min(corner, h / 2),
        stroke_color=stroke,
        stroke_width=th.STROKE_BOX,
        fill_color=fill,
        fill_opacity=1 if filled else 0,
    )
    body.move_to(rect)
    group = VGroup(rect, body)
    group.rect, group.text, group.sub = rect, name, sub
    return group


# Arrows ---------------------------------------------------------------------


def _anchor(m, toward: np.ndarray, vertical: bool) -> np.ndarray:
    if not isinstance(m, Mobject):
        return np.array([*np.asarray(m, dtype=float)[:2], 0.0])
    c = m.get_center()
    if vertical:
        return m.get_top() if toward[1] > c[1] else m.get_bottom()
    return m.get_right() if toward[0] > c[0] else m.get_left()


def arrow(
    a,
    b,
    text: str | None = None,
    color=th.INK_3,
    buff: float = 0.1,
    size: float = th.SIZE_LABEL,
    side: str = "auto",
    stroke: float = th.STROKE_BOX + 1,
) -> VGroup:
    """An arrow from `a` to `b`, with an optional label beside it.

    `a` and `b` are mobjects or points. Between mobjects, the arrow joins
    the facing edges: left and right when they sit side by side, top and
    bottom when one is above the other.

    Args:
      text: A short label, such as "2 TB/s". Text with `$` goes through LaTeX.
      side: Where the label sits: `above`, `below`, `left`, `right`, or
        `auto` (above a horizontal arrow, right of a vertical one).

    Returns:
      A VGroup with attributes `line` (the Arrow) and `label` (or None).
    """
    ca = a.get_center() if isinstance(a, Mobject) else np.array([*a[:2], 0.0])
    cb = b.get_center() if isinstance(b, Mobject) else np.array([*b[:2], 0.0])
    d = cb - ca
    vertical = abs(d[1]) > abs(d[0])
    start = _anchor(a, cb, vertical)
    end = _anchor(b, ca, vertical)
    # Between mobjects that overlap across the arrow, run it straight, at
    # the middle of the overlap.
    if isinstance(a, Mobject) and isinstance(b, Mobject):
        k = 0 if vertical else 1
        lo = max(a.get_corner(DOWN + LEFT)[k], b.get_corner(DOWN + LEFT)[k])
        hi = min(a.get_corner(UP + RIGHT)[k], b.get_corner(UP + RIGHT)[k])
        if hi - lo > 0.2:
            start[k] = end[k] = (lo + hi) / 2
    line = Arrow(
        start,
        end,
        buff=buff,
        color=color,
        stroke_width=stroke,
        tip_length=0.2,
        max_tip_length_to_length_ratio=0.35,
        max_stroke_width_to_length_ratio=20,
    )
    group = VGroup(line)
    group.line, group.label = line, None
    if text:
        lab = prose(text, size=size, color=th.INK_3, width=None) if "$" in text else label(text, size=size, color=th.INK_3)
        where = side
        if where == "auto":
            where = "right" if vertical else "above"
        direction = {"above": UP, "below": DOWN, "left": LEFT, "right": RIGHT}[where]
        lab.next_to(line.get_center(), direction, buff=0.14)
        group.add(lab)
        group.label = lab
    return group


# Grids ----------------------------------------------------------------------


class Grid(VGroup):
    """A grid of cells built by `grid()`: a tensor, a memory, a tiling.

    Attributes:
      cells: `cells[r][c]` is the cell in row `r`, column `c`, a VGroup of
        the square (`.square`) and its text (`.text`, or None).
      rows, cols: The grid's shape.
    """

    def cell(self, r: int, c: int) -> VGroup:
        """The cell at row `r`, column `c` (0-based)."""
        return self.cells[r][c]

    def row(self, r: int) -> VGroup:
        """Every cell in row `r`."""
        return VGroup(*self.cells[r])

    def col(self, c: int) -> VGroup:
        """Every cell in column `c`."""
        return VGroup(*[row[c] for row in self.cells])

    def block(self, r0: int, r1: int, c0: int, c1: int) -> VGroup:
        """The cells in rows `r0` to `r1 - 1` and columns `c0` to `c1 - 1`, like a slice."""
        return VGroup(*[self.cells[r][c] for r in range(r0, r1) for c in range(c0, c1)])

    def squares(self, cells: VGroup | Sequence | None = None) -> VGroup:
        """The squares of `cells` (default: every cell), without their text."""
        if cells is None:
            return VGroup(*[c.square for row in self.cells for c in row])
        if hasattr(cells, "square"):
            return VGroup(cells.square)
        items = cells.submobjects if isinstance(cells, VGroup) else list(cells)
        return VGroup(*[sq for item in items for sq in self.squares(item)])

    def fill(self, cells, color, opacity: float = 1.0, tint: float | None = 0.35, **kwargs):
        """Returns an animation that fills `cells` (a cell, row, column, or block).

        `tint` mixes the colour into the canvas, so text in the cells stays
        readable; pass `tint=None` for the solid colour.
        """
        fill_color = th.tint(color, tint) if tint is not None else color
        squares = self.squares(cells)
        return AnimationGroup(*[s.animate.set_fill(fill_color, opacity=opacity) for s in squares], **kwargs)

    def paint(self, cells, color, tint: float | None = 0.35) -> None:
        """Fills `cells` at once, with no animation (for the starting state)."""
        fill_color = th.tint(color, tint) if tint is not None else color
        for s in self.squares(cells):
            s.set_fill(fill_color, opacity=1)

    def outline(self, r0: int, r1: int, c0: int, c1: int, color=None, buff: float = 0.04, stroke: float = 4) -> Rectangle:
        """Returns a rectangle around a block of cells (a tile), not yet added."""
        block = self.block(r0, r1, c0, c1)
        color = th.ACCENT if color is None else color
        return Rectangle(
            width=block.width + 2 * buff, height=block.height + 2 * buff, stroke_color=color,
            stroke_width=stroke, fill_opacity=0,
        ).move_to(block)


def grid(
    rows: int,
    cols: int,
    cell: float | tuple[float, float] = 0.5,
    values: Sequence[Sequence] | None = None,
    colors: Sequence[Sequence] | None = None,
    gap: float = 0.0,
    size: float | None = None,
    font: str | None = None,
    row_labels: Sequence[str] | None = None,
    col_labels: Sequence[str] | None = None,
    stroke=th.RULE_STRONG,
) -> Grid:
    """A grid of cells for tensors, memory blocks, pages, and tiles.

    Args:
      rows, cols: The grid's shape. Keep it under about 24 columns, so a
        cell stays wide enough to see.
      cell: Cell size in Manim units, or `(width, height)`.
      values: Optional text per cell, `values[r][c]`; None or "" leaves a
        cell blank.
      colors: Optional fill colour per cell, `colors[r][c]`, tinted into
        the canvas; None leaves a cell white.
      gap: Space between cells. 0 draws a shared-edge grid; a small gap
        (0.06) draws separate blocks, such as KV-cache pages.
      size: Text size for values. Defaults to fit the cell, at least
        `th.SIZE_LABEL`.
      font: Font for values; defaults to the mono family.
      row_labels, col_labels: Optional labels left of each row and above
        each column.

    Returns:
      A `Grid` at the origin. Address cells with `cell(r, c)`, `row(r)`,
      `col(c)`, and `block(r0, r1, c0, c1)`; animate with `fill()`.
    """
    cw, ch = (cell, cell) if isinstance(cell, (int, float)) else cell
    size = size or max(th.SIZE_LABEL, min(th.SIZE_SMALL, ch * 48))
    out = Grid()
    out.rows, out.cols = rows, cols
    cells: list[list[VGroup]] = []
    for r in range(rows):
        row = []
        for c in range(cols):
            fill_color = th.RAISE
            if colors is not None and colors[r][c] is not None:
                fill_color = th.tint(colors[r][c], 0.35)
            sq = Rectangle(width=cw, height=ch, stroke_color=stroke, stroke_width=th.STROKE_HAIRLINE,
                           fill_color=fill_color, fill_opacity=1)
            sq.move_to([c * (cw + gap) + cw / 2, -r * (ch + gap) - ch / 2, 0])
            item = VGroup(sq)
            item.square, item.text = sq, None
            v = None if values is None else values[r][c]
            if v not in (None, ""):
                t = label(str(v), size=size, color=th.INK_2, font=font or th.MONO)
                if t.width > cw * 0.86:
                    t.scale_to_fit_width(cw * 0.86)
                t.move_to(sq)
                item.add(t)
                item.text = t
            row.append(item)
            out.add(item)
        cells.append(row)
    out.cells = cells
    out.row_labels = VGroup()
    out.col_labels = VGroup()
    if row_labels:
        for r, text in enumerate(row_labels):
            lab = label(text, size=th.SIZE_LABEL, color=th.INK_3)
            lab.next_to(cells[r][0], LEFT, buff=0.15)
            out.row_labels.add(lab)
        out.add(out.row_labels)
    if col_labels:
        for c, text in enumerate(col_labels):
            lab = label(text, size=th.SIZE_LABEL, color=th.INK_3)
            lab.next_to(cells[0][c], UP, buff=0.12)
            out.col_labels.add(lab)
        out.add(out.col_labels)
    out.move_to([0, 0, 0])
    return out


# Bar comparisons --------------------------------------------------------------


def _fmt(value: float, decimals: int, unit: str) -> str:
    number = f"{value:,.{decimals}f}"
    return f"{number} {unit}".strip()


def bar_compare(
    items: Sequence[tuple],
    unit: str = "",
    max_value: float | None = None,
    width: float = 7.5,
    bar_height: float = 0.5,
    gap: float = 0.32,
    decimals: int = 1,
    limit: float | None = None,
    limit_label: str | None = None,
    color=None,
    size: float = th.SIZE_SMALL,
) -> VGroup:
    """Horizontal bars that compare a few numbers, each labelled with its value.

    Args:
      items: `(label, value)` or `(label, value, colour)` tuples, top to
        bottom. A value label can be overridden with a fourth element, the
        text to print, such as "54 GB".
      unit: Appended to each printed value, such as "GiB".
      max_value: The value at full `width`. Defaults to the largest of the
        values and `limit`.
      width: Length of the longest possible bar in Manim units.
      decimals: Decimal places in printed values.
      limit: Draws a dashed capacity line at this value, such as 80 GiB.
      limit_label: Text above the capacity line.
      color: Bar colour when an item names none; defaults to the accent.

    Returns:
      A VGroup with attributes `labels`, `bars`, `values`, and `limit`
      (or None). `grow()` returns animations that grow the bars from the
      left.
    """
    top = max([float(it[1]) for it in items] + ([float(limit)] if limit is not None else []))
    scale = width / (max_value or top or 1.0)
    names = VGroup(*[label(str(it[0]), size=size, color=th.INK_2) for it in items])
    name_w = max(n.width for n in names)
    x0 = name_w + 0.3
    bars = VGroup()
    values = VGroup()
    for i, it in enumerate(items):
        y = -i * (bar_height + gap)
        c = it[2] if len(it) > 2 and it[2] is not None else (color or th.ACCENT)
        length = max(0.02, float(it[1]) * scale)
        bar = Rectangle(width=length, height=bar_height, fill_color=c, fill_opacity=0.85, stroke_width=0)
        bar.move_to([x0 + length / 2, y, 0])
        names[i].move_to([name_w - names[i].width / 2, y, 0])
        text = it[3] if len(it) > 3 else _fmt(float(it[1]), decimals, unit)
        val = label(text, size=size, color=th.INK, weight=MEDIUM)
        val.next_to(bar, RIGHT, buff=0.16)
        bars.add(bar)
        values.add(val)
    group = VGroup(names, bars, values)
    group.labels, group.bars, group.values, group.limit = names, bars, values, None
    if limit is not None:
        x = x0 + float(limit) * scale
        y_top = bar_height / 2 + 0.25
        y_bot = -(len(items) - 1) * (bar_height + gap) - bar_height / 2 - 0.15
        line = DashedLine([x, y_bot, 0], [x, y_top, 0], color=th.INK_3, stroke_width=th.STROKE_AXIS, dash_length=0.1)
        lim = VGroup(line)
        if limit_label:
            t = label(limit_label, size=th.SIZE_LABEL, color=th.INK_3)
            t.next_to(line, UP, buff=0.1)
            lim.add(t)
        group.add(lim)
        group.limit = lim

    def grow(**kwargs):
        return [GrowFromEdge(b, LEFT, **kwargs) for b in bars]

    group.grow = grow
    group.move_to([0, 0, 0])
    return group


__all__ = ["Grid", "arrow", "bar_compare", "box", "grid"]
