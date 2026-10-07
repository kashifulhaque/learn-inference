"""The lecture kit: shared style and structure for every manim lecture.

Import everything a scene needs from here:

    from manim import *
    import kit
    from kit import theme as th

    class Lecture(kit.Lecture):
        def construct(self):
            self.title_slide()
            ...

The kit is course-agnostic: the course's name, chapter paths, accent
colour, fonts, and voice come from `lectures/course.json`. See
`lectures/AUTHORING.md` for the conventions and a full example.
"""

from . import course, theme
from .blocks import (
    Derivation,
    arrow_note,
    bullets,
    build,
    callout,
    fit,
    mark,
    pill,
    table,
    tex_escape,
    title_card,
    underline,
    where_next,
)
from .code import CodePanel, code
from .diagrams import Grid, arrow, bar_compare, box, grid
from .lecture import Lecture, LectureError, Narration, next_chapter, read_chapter
from .plots import (
    RepeatedSampling,
    RunningHistogram,
    axes,
    bars,
    density,
    dots_on_axis,
    legend,
    rng,
    scatter,
    stacked_dots,
    tail,
    vline,
)
from .text import formula, heading, label, math_label, mono, prose, stack

__all__ = [
    "CodePanel",
    "Derivation",
    "Grid",
    "Lecture",
    "LectureError",
    "Narration",
    "RepeatedSampling",
    "RunningHistogram",
    "arrow",
    "arrow_note",
    "axes",
    "bar_compare",
    "bars",
    "box",
    "build",
    "bullets",
    "callout",
    "code",
    "course",
    "density",
    "dots_on_axis",
    "fit",
    "formula",
    "grid",
    "heading",
    "label",
    "legend",
    "mark",
    "math_label",
    "mono",
    "next_chapter",
    "pill",
    "prose",
    "read_chapter",
    "rng",
    "scatter",
    "stack",
    "stacked_dots",
    "table",
    "tail",
    "tex_escape",
    "theme",
    "title_card",
    "underline",
    "vline",
    "where_next",
]
