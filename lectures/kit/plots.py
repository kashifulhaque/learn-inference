"""Axes and plots in the site's style, and seeded sampling animations.

Every random draw in a lecture comes from `rng(seed)`, so a re-render draws
the same samples and the slides do not change between renders.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Axes,
    Dot,
    FadeIn,
    Line,
    Rectangle,
    Transform,
    VGroup,
    VMobject,
)

from . import theme as th
from .text import label, math_label

# Randomness ----------------------------------------------------------------


def rng(seed: int) -> np.random.Generator:
    """A seeded generator. Give each simulation in a lecture its own seed."""
    return np.random.default_rng(seed)


# Axes ----------------------------------------------------------------------


def axes(
    x_range: Sequence[float],
    y_range: Sequence[float],
    width: float = 9.0,
    height: float = 4.0,
    x_label: str | None = None,
    y_label: str | None = None,
    x_numbers: Sequence[float] | None = None,
    y_numbers: Sequence[float] | None = None,
    show_y_axis: bool = True,
    number_size: float = th.SIZE_LABEL - 2,
    decimals: int = 0,
) -> Axes:
    """Axes in the site style: thin warm-grey lines, no arrow tips.

    Args:
      x_range: `[min, max, step]`; the step spaces the ticks.
      y_range: `[min, max, step]`.
      x_label: TeX for the x-axis label, placed under the axis at the right.
      y_label: TeX for the y-axis label, placed at the top of the y-axis.
      x_numbers: Tick values to number. Defaults to none.
      y_numbers: Tick values to number. Defaults to none.
      show_y_axis: False draws only the baseline, which suits densities.
      decimals: Decimal places on tick numbers.
    """
    ax = Axes(
        x_range=list(x_range),
        y_range=list(y_range),
        x_length=width,
        y_length=height,
        tips=False,
        axis_config={
            "color": th.INK_3,
            "stroke_width": th.STROKE_AXIS,
            "include_ticks": True,
            "tick_size": 0.06,
            "include_numbers": False,
        },
    )
    if not show_y_axis:
        ax.y_axis.set_opacity(0)
    labels = VGroup()
    if x_numbers is not None:
        for v in x_numbers:
            n = math_label(_fmt(v, decimals), size=number_size, color=th.INK_3)
            n.next_to(ax.c2p(v, y_range[0]), DOWN, buff=0.16)
            labels.add(n)
    if y_numbers is not None and show_y_axis:
        for v in y_numbers:
            n = math_label(_fmt(v, decimals), size=number_size, color=th.INK_3)
            n.next_to(ax.c2p(x_range[0], v), LEFT, buff=0.14)
            labels.add(n)
    ax.number_labels = labels
    ax.add(labels)
    if x_label:
        xl = math_label(x_label, size=th.SIZE_LABEL, color=th.INK_3)
        xl.next_to(ax.c2p(x_range[1], y_range[0]), DOWN, buff=0.5 if x_numbers is not None else 0.2)
        xl.align_to(ax.c2p(x_range[1], 0), RIGHT)
        ax.x_title = xl
        ax.add(xl)
    if y_label and show_y_axis:
        yl = math_label(y_label, size=th.SIZE_LABEL, color=th.INK_3)
        yl.next_to(ax.c2p(x_range[0], y_range[1]), UP, buff=0.15)
        ax.y_title = yl
        ax.add(yl)
    return ax


def _fmt(v: float, decimals: int) -> str:
    if decimals == 0:
        return f"{int(round(v))}"
    return f"{v:.{decimals}f}"


def density(
    ax: Axes,
    f: Callable[[float], float],
    x_range: Sequence[float] | None = None,
    color=th.ACCENT,
    stroke: float = th.STROKE_CURVE,
    fill_opacity: float = 0.14,
) -> VGroup:
    """A density curve with a light fill under it.

    Returns a VGroup `(area, curve)`; animate `Create(result.curve)` and
    `FadeIn(result.area)` separately if you like.
    """
    lo, hi = (ax.x_range[0], ax.x_range[1]) if x_range is None else x_range
    curve = ax.plot(lambda x: float(f(x)), x_range=[lo, hi, (hi - lo) / 400], color=color, stroke_width=stroke)
    area = ax.get_area(curve, x_range=[lo, hi], color=color, opacity=fill_opacity)
    area.set_stroke(width=0)
    group = VGroup(area, curve)
    group.curve, group.area = curve, area
    return group


def tail(ax: Axes, curve: VMobject, a: float, b: float, color=th.BAD, opacity: float = 0.35) -> VMobject:
    """Shades the area under `curve` between `a` and `b`."""
    area = ax.get_area(curve, x_range=[a, b], color=color, opacity=opacity)
    area.set_stroke(width=0)
    return area


def vline(ax: Axes, x: float, y_top: float, color=th.INK_3, stroke: float = th.STROKE_AXIS, dashed: bool = True):
    """A vertical marker from the baseline up to `y_top` at `x`."""
    from manim import DashedLine

    cls = DashedLine if dashed else Line
    return cls(ax.c2p(x, ax.y_range[0]), ax.c2p(x, y_top), color=color, stroke_width=stroke)


def bars(
    ax: Axes,
    edges: Sequence[float],
    heights: Sequence[float],
    color=th.ACCENT,
    fill_opacity: float = 0.35,
    stroke: float = 1.5,
) -> VGroup:
    """Histogram bars from bin edges and heights, in axis coordinates."""
    group = VGroup()
    base = ax.y_range[0]
    for lo, hi, h in zip(edges[:-1], edges[1:], heights):
        p0, p1 = ax.c2p(lo, base), ax.c2p(hi, max(h, base))
        w, hgt = p1[0] - p0[0], max(p1[1] - p0[1], 1e-3)
        r = Rectangle(
            width=w,
            height=hgt,
            fill_color=color,
            fill_opacity=fill_opacity,
            stroke_color=color,
            stroke_width=stroke if h > 0 else 0,
        )
        r.move_to([(p0[0] + p1[0]) / 2, p0[1] + hgt / 2, 0])
        group.add(r)
    return group


class RunningHistogram:
    """A histogram that grows as samples arrive, drawn as a density.

    Example:
      h = RunningHistogram(ax, edges=np.linspace(496, 504, 33), color=blue)
      self.add(h.bars)
      for batch in batches:
          h.add(batch)
          self.play(h.update(), run_time=0.3)
    """

    def __init__(self, ax: Axes, edges: Sequence[float], color=th.ACCENT, fill_opacity: float = 0.35):
        self.ax = ax
        self.edges = np.asarray(edges, dtype=float)
        self.color = color
        self.fill_opacity = fill_opacity
        self.values: list[float] = []
        self.bars = bars(ax, self.edges, np.zeros(len(self.edges) - 1), color=color, fill_opacity=fill_opacity)

    def heights(self) -> np.ndarray:
        if not self.values:
            return np.zeros(len(self.edges) - 1)
        counts, _ = np.histogram(self.values, bins=self.edges)
        widths = np.diff(self.edges)
        return counts / (len(self.values) * widths)

    def add(self, values) -> None:
        self.values.extend(np.atleast_1d(values).tolist())

    def target(self) -> VGroup:
        return bars(self.ax, self.edges, self.heights(), color=self.color, fill_opacity=self.fill_opacity)

    def update(self, **kwargs) -> Transform:
        """Returns the animation that moves the bars to the current counts."""
        return Transform(self.bars, self.target(), **kwargs)


def dots_on_axis(
    ax: Axes,
    values: Sequence[float],
    color=th.ACCENT,
    radius: float = 0.06,
    y: float | None = None,
    opacity: float = 0.9,
) -> VGroup:
    """Dots at `values` along the x-axis, at height `y` (axis units)."""
    y = ax.y_range[0] if y is None else y
    return VGroup(
        *[Dot(ax.c2p(v, y), radius=radius, color=color, fill_opacity=opacity, stroke_width=0) for v in values]
    )


def stacked_dots(
    ax: Axes,
    values: Sequence[float],
    bin_width: float,
    color=th.ACCENT,
    radius: float = 0.055,
    gap: float = 0.02,
) -> VGroup:
    """A dot plot: values binned and stacked upwards, one dot per value.

    Stacks grow in screen units (2 * radius + gap per dot), so choose the
    number of values and the axis height together.
    """
    group = VGroup()
    counts: dict[int, int] = {}
    for v in values:
        k = int(np.floor(v / bin_width))
        c = counts.get(k, 0)
        counts[k] = c + 1
        x = (k + 0.5) * bin_width
        base = ax.c2p(x, ax.y_range[0])
        group.add(
            Dot(base + UP * (radius + c * (2 * radius + gap)), radius=radius, color=color, stroke_width=0)
        )
    return group


def scatter(ax: Axes, xs, ys, color=th.ACCENT, radius: float = 0.05, opacity: float = 0.8) -> VGroup:
    """A scatter plot."""
    return VGroup(
        *[Dot(ax.c2p(x, y), radius=radius, color=color, fill_opacity=opacity, stroke_width=0) for x, y in zip(xs, ys)]
    )


def legend(entries: Sequence[tuple[str, object]], size: float = th.SIZE_LABEL) -> VGroup:
    """A legend: a short coloured rule and a TeX label per entry."""
    rows = VGroup()
    for tex, color in entries:
        swatch = Line(LEFT * 0.22, RIGHT * 0.22, color=color, stroke_width=th.STROKE_CURVE)
        name = math_label(tex, size=size, color=th.INK_2) if "$" not in tex else _prose_label(tex, size)
        rows.add(VGroup(swatch, name).arrange(RIGHT, buff=0.18))
    rows.arrange(DOWN, buff=0.16, aligned_edge=LEFT)
    return rows


def _prose_label(tex: str, size: float):
    from .text import prose

    return prose(tex, size=size, color=th.INK_2, width=None)


class RepeatedSampling:
    """Animates a sampling distribution emerging from repeated samples.

    The top axes show the population. Each call to `one()` draws a sample
    as dots on the top axes, marks the statistic, and drops it onto the
    bottom axes, where the statistics stack up as dots. `many()` then adds
    a large batch at once and turns the stack into a density histogram.
    Both axes should share an x-range, so the narrowing is visible.

    Example:
      demo = RepeatedSampling(self, top, bottom, rng(11),
                              draw=lambda r: r.normal(500, 5, 25),
                              statistic=np.mean, bin_width=0.5,
                              color=self.color, sample_color=self.colors[1])
      for _ in range(4):
          demo.one(run_time=2.0)
      demo.many(2000, edges=np.arange(496, 504.01, 0.5), run_time=2.0)
    """

    def __init__(
        self,
        scene,
        top: Axes,
        bottom: Axes,
        generator: np.random.Generator,
        draw: Callable[[np.random.Generator], np.ndarray],
        statistic: Callable[[np.ndarray], float],
        bin_width: float,
        color=th.ACCENT,
        sample_color=th.ORANGE,
        dot_radius: float = 0.055,
    ):
        self.scene = scene
        self.top = top
        self.bottom = bottom
        self.rng = generator
        self.draw = draw
        self.statistic = statistic
        self.bin_width = bin_width
        self.color = color
        self.sample_color = sample_color
        self.dot_radius = dot_radius
        self.values: list[float] = []
        self.stack = VGroup()
        self.histogram: VGroup | None = None
        self._counts: dict[int, int] = {}

    def _stack_dot(self, value: float) -> Dot:
        k = int(np.floor(value / self.bin_width))
        c = self._counts.get(k, 0)
        self._counts[k] = c + 1
        x = (k + 0.5) * self.bin_width
        base = self.bottom.c2p(x, self.bottom.y_range[0])
        r = self.dot_radius
        return Dot(base + UP * (r + c * (2 * r + 0.02)), radius=r, color=self.color, stroke_width=0)

    def one(self, run_time: float = 2.0, keep_sample: bool = False) -> float:
        """Plays one sample and its statistic; returns the statistic."""
        from manim import FadeOut, LaggedStart

        sample = np.asarray(self.draw(self.rng), dtype=float)
        value = float(self.statistic(sample))
        span = self.top.y_range[1] - self.top.y_range[0]
        heights = self.top.y_range[0] + span * (0.06 + 0.22 * self.rng.random(len(sample)))
        dots = VGroup(
            *[
                Dot(self.top.c2p(v, h), radius=self.dot_radius * 0.9, color=self.sample_color, stroke_width=0)
                for v, h in zip(sample, heights)
            ]
        )
        tick = Line(
            self.top.c2p(value, self.top.y_range[0]),
            self.top.c2p(value, self.top.y_range[0] + 0.45 * span),
            color=self.color,
            stroke_width=5,
        )
        target = self._stack_dot(value)
        marker = Dot(tick.get_bottom(), radius=self.dot_radius * 1.2, color=self.color, stroke_width=0)
        t = run_time
        self.scene.play(LaggedStart(*[FadeIn(d, scale=0.6) for d in dots], lag_ratio=0.04), run_time=0.4 * t)
        self.scene.play(FadeIn(tick, shift=DOWN * 0.1), FadeIn(marker), run_time=0.2 * t)
        self.scene.play(marker.animate.move_to(target).scale(1 / 1.2), run_time=0.25 * t)
        self.stack.add(marker)
        if not keep_sample:
            self.scene.play(FadeOut(dots), FadeOut(tick), run_time=0.15 * t)
        self.values.append(value)
        return value

    def many(self, count: int, edges: Sequence[float], run_time: float = 2.0, fill_opacity: float = 0.35) -> VGroup:
        """Adds `count` more statistics and shows them all as a density histogram.

        The bottom axes' y-range must fit the density (for a normal
        statistic, about 0.4 divided by its standard deviation).
        """
        from manim import FadeOut, ReplacementTransform

        extra = [float(self.statistic(np.asarray(self.draw(self.rng)))) for _ in range(count)]
        self.values.extend(extra)
        edges = np.asarray(edges, dtype=float)
        counts, _ = np.histogram(self.values, bins=edges)
        heights = counts / (len(self.values) * np.diff(edges))
        hist = bars(self.bottom, edges, heights, color=self.color, fill_opacity=fill_opacity)
        start = bars(self.bottom, edges, np.zeros(len(edges) - 1), color=self.color, fill_opacity=fill_opacity)
        self.scene.play(FadeOut(self.stack), FadeIn(start), run_time=0.3 * run_time)
        self.scene.play(ReplacementTransform(start, hist), run_time=0.7 * run_time)
        self.histogram = hist
        return hist


def reveal_dots(dots: VGroup, lag: float = 0.05) -> list:
    """FadeIn animations for a group of dots, to pass to `LaggedStart`."""
    return [FadeIn(d, scale=0.5) for d in dots]


__all__ = [
    "RepeatedSampling",
    "RunningHistogram",
    "axes",
    "bars",
    "density",
    "dots_on_axis",
    "label",
    "legend",
    "reveal_dots",
    "rng",
    "scatter",
    "stacked_dots",
    "tail",
    "vline",
]
