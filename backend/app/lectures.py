"""Lectures: one narrated video per chapter.

The render pipeline in `lectures/` writes one manifest per chapter to
`content/lectures/<slug>.json`. A manifest names the video, its poster and
captions, and the sections the video is cut into. Each section is one slide:
its final frame as an image, its title, its narration, where it starts and
ends in the video, and, when the manifest says, the `##` section of the
chapter it teaches. The media live in a public bucket, so a manifest carries
a `base_url`, and every other path in it is relative to that.

A chapter without a manifest, or with one that doesn't parse, has no lecture,
and the reader shows no player. A bad manifest is logged and skipped, never
fatal.

Manifests are read once and cached, like the chapters. `reload_lectures`
clears the cache.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import get_settings

log = logging.getLogger("learn-inference.lectures")

# The reader numbers each `##` section of a chapter `s-1`, `s-2`, and so on
# (see rehypeSectionize in the frontend), so a section links to the chapter
# text only through an anchor of that shape.
_ANCHOR = re.compile(r"^s-[1-9][0-9]*$")


def _lectures_dir() -> Path:
    return get_settings().content_dir / "lectures"


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return float(value)
    return None


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _url(base: str, path: Any) -> str | None:
    """A manifest path resolved against the lecture's `base_url`."""
    path = _text(path)
    if path is None:
        return None
    if path.startswith(("https://", "http://", "/")):
        return path
    return base + path.lstrip("/")


def parse_manifest(data: Any, slug: str) -> dict[str, Any] | None:
    """A manifest in the shape the API serves, or None when it is unusable.

    The file name decides the slug, whatever the manifest says, so a copied
    manifest can't claim another chapter. Every media path comes back as an
    absolute URL. A section missing its id or title, or repeating an id, is
    dropped, and its times are clamped to the video. A `book_anchor` that
    isn't a reader anchor becomes None.
    """
    if not isinstance(data, dict):
        return None
    base = _text(data.get("base_url"))
    video = data.get("video")
    if base is None or not isinstance(video, dict):
        return None
    if not base.endswith("/"):
        base += "/"
    src = _url(base, video.get("src"))
    if src is None:
        return None
    duration = _number(video.get("duration")) or 0.0

    sections: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in data.get("sections") or []:
        if not isinstance(item, dict):
            continue
        sid, title = _text(item.get("id")), _text(item.get("title"))
        if sid is None or title is None or sid in seen:
            continue
        seen.add(sid)
        start = _number(item.get("start")) or 0.0
        end = _number(item.get("end"))
        if end is None or end < start:
            end = start
        if duration:
            start, end = min(start, duration), min(end, duration)
        anchor = _text(item.get("book_anchor"))
        sections.append(
            {
                "id": sid,
                "title": title,
                "start": start,
                "end": end,
                "image": _url(base, item.get("slide")),
                "notes": _text(item.get("notes")) or "",
                "book_section": _text(item.get("book_section")),
                "book_anchor": anchor if anchor and _ANCHOR.match(anchor) else None,
            }
        )

    return {
        "slug": slug,
        "title": _text(data.get("title")) or "",
        "version": str(data.get("version") or ""),
        "base_url": base,
        "video": {
            "src": src,
            "duration": duration,
            "width": int(_number(video.get("width")) or 0),
            "height": int(_number(video.get("height")) or 0),
            "poster": _url(base, video.get("poster")),
        },
        "captions": _url(base, data.get("captions")),
        "sections": sections,
    }


@lru_cache
def load_lectures() -> dict[str, dict[str, Any]]:
    """Every usable lecture on disk, keyed by chapter slug."""
    found: dict[str, dict[str, Any]] = {}
    root = _lectures_dir()
    if not root.is_dir():
        return found
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("Skipping the lecture manifest %s: %s", path, exc)
            continue
        lecture = parse_manifest(data, path.stem)
        if lecture is None:
            log.warning("Skipping the lecture manifest %s: it has no base_url or video", path)
            continue
        claimed = data.get("slug")
        if isinstance(claimed, str) and claimed != path.stem:
            log.warning("The lecture manifest %s names the slug %r; using %r", path, claimed, path.stem)
        found[path.stem] = lecture
    return found


def reload_lectures() -> None:
    load_lectures.cache_clear()


def get_lecture(slug: str) -> dict[str, Any] | None:
    return load_lectures().get(slug)


def summary(lecture: dict[str, Any]) -> dict[str, Any]:
    """A lecture without its sections, for an index of lectures."""
    return {
        "slug": lecture["slug"],
        "title": lecture["title"],
        "version": lecture["version"],
        "duration": lecture["video"]["duration"],
        "poster": lecture["video"]["poster"],
        "sections": len(lecture["sections"]),
    }
