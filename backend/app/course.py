"""Course-wide facts shared by the REST API and the MCP server."""

from __future__ import annotations

import json
from typing import Any

from . import curriculum, db
from .config import Settings


def book_info(settings: Settings) -> dict[str, Any] | None:
    """The PDF book on disk, as the book build's manifest describes it.

    The Docker image builds the PDF and copies it to `book_dir`. A checkout that
    hasn't run the book build has none. Then this is None and the sidebar shows
    no download link, rather than one that leads to a 404.
    """
    try:
        manifest = json.loads((settings.book_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    name = manifest.get("file")
    if not isinstance(name, str) or "/" in name or not (settings.book_dir / name).is_file():
        return None
    return {
        "title": manifest.get("title"),
        "pages": manifest.get("pages"),
        "bytes": manifest.get("bytes"),
        "built": manifest.get("built"),
        "commit": manifest.get("commit"),
        "url": f"/book/{name}",
    }


def progress_summary(user: str) -> dict[str, Any]:
    """How far the user is through the course, for the hub."""
    titles = {c["slug"]: c["title"] for c in curriculum.chapter_list()}
    rows = [r for r in db.get_progress_rows(user) if r["chapter"] in titles]
    last = rows[0] if rows else None  # Rows arrive newest first.
    return {
        "done": sum(1 for r in rows if r["status"] == "done"),
        "total": len(titles),
        "last": {"slug": last["chapter"], "title": titles[last["chapter"]]} if last else None,
    }
