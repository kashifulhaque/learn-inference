"""Tests for the lecture manifests and the routes that serve them.

The lectures come from `fixtures/lectures/`, not `content/lectures/`, so the
tests don't depend on which lectures the render pipeline has produced. The
chapters are the real ones in `content/chapters/`. The fixture holds:

- 14-flash-attention: a good manifest, with a repeated section id and a
  section whose `book_anchor` isn't a reader anchor.
- 02-memory-arithmetic: JSON that doesn't parse.
- 03-tokens-and-embeddings: valid JSON with no video.
- 99-no-such-chapter: a good manifest for a chapter that doesn't exist.

They need only the standard library and the backend's own requirements. Run
them from the repository root:

    .venv/bin/python -m unittest discover -s backend/tests -t . -v

pytest runs them too, with `python -m pytest backend/tests`.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from backend.app import lectures
from backend.app.config import get_settings
from backend.app.main import app

FIXTURE_LECTURES = Path(__file__).parent / "fixtures" / "lectures"
CHAPTER = "14-flash-attention"
BASE = "https://media.example/lectures/14-flash-attention/fixture0001/"


class LectureTestCase(unittest.TestCase):
    """Points the app at a scratch data directory and the fixture lectures."""

    def setUp(self) -> None:
        data = tempfile.TemporaryDirectory()
        self.addCleanup(data.cleanup)
        env = mock.patch.dict(
            os.environ,
            {"DATA_DIR": data.name, "SSO_PROXY_SECRET": "test-secret", "SSO_USERS": "ada:Ada"},
        )
        env.start()
        self.addCleanup(env.stop)
        folder = mock.patch.object(lectures, "_lectures_dir", lambda: FIXTURE_LECTURES)
        folder.start()
        self.addCleanup(folder.stop)
        for clear in (get_settings.cache_clear, lectures.reload_lectures):
            clear()
            self.addCleanup(clear)

    def client(self, signed_in: bool = True) -> TestClient:
        headers = {"Remote-User": "ada", "X-SSO-Proxy": "test-secret"} if signed_in else {}
        client = TestClient(app, headers=headers)
        # Entering the client runs the startup hook, which creates the schema.
        self.enterContext(client)
        return client


class ParseManifestTest(LectureTestCase):
    def test_paths_resolve_against_the_base_url(self) -> None:
        lecture = lectures.get_lecture(CHAPTER)
        assert lecture is not None
        self.assertEqual(lecture["base_url"], BASE)
        self.assertEqual(lecture["video"]["src"], BASE + "lecture.mp4")
        self.assertEqual(lecture["video"]["poster"], BASE + "poster.webp")
        self.assertEqual(lecture["captions"], BASE + "captions.vtt")
        self.assertEqual(lecture["sections"][1]["image"], BASE + "slides/02-score-matrix.webp")
        # An absolute path is left as it is.
        self.assertEqual(lecture["sections"][2]["image"], "https://elsewhere.example/summary.webp")

    def test_sections_are_cleaned(self) -> None:
        lecture = lectures.get_lecture(CHAPTER)
        assert lecture is not None
        sections = lecture["sections"]
        # The repeated id is dropped, and the last section ends with the video.
        self.assertEqual(
            [(s["id"], s["start"], s["end"]) for s in sections],
            [("title", 0.0, 30.0), ("score-matrix", 30.0, 90.0), ("summary", 90.0, 120.0)],
        )
        self.assertEqual(sections[0]["book_anchor"], None)
        self.assertEqual(sections[1]["book_anchor"], "s-2")
        self.assertEqual(sections[1]["book_section"], "s-what-the-naive-version-costs")
        # An anchor the reader never renders is dropped, not passed on.
        self.assertEqual(sections[2]["book_anchor"], None)
        self.assertEqual(sections[2]["notes"], "")

    def test_bad_manifests_are_skipped(self) -> None:
        self.assertEqual(
            sorted(lectures.load_lectures()), [CHAPTER, "99-no-such-chapter"]
        )
        self.assertIsNone(lectures.parse_manifest([], "x"))
        self.assertIsNone(lectures.parse_manifest({"video": {"src": "a.mp4"}}, "x"))
        self.assertIsNone(lectures.parse_manifest({"base_url": "https://m/", "video": {}}, "x"))

    def test_the_file_name_decides_the_slug(self) -> None:
        parsed = lectures.parse_manifest(
            {"slug": "elsewhere", "base_url": "https://m/x", "video": {"src": "a.mp4"}}, "01-real"
        )
        assert parsed is not None
        self.assertEqual(parsed["slug"], "01-real")
        self.assertEqual(parsed["video"]["src"], "https://m/x/a.mp4")


class RoutesTest(LectureTestCase):
    def test_routes_need_a_trusted_sso_identity(self) -> None:
        client = self.client(signed_in=False)
        for path in ("/api/lectures", f"/api/lectures/{CHAPTER}", f"/api/chapters/{CHAPTER}"):
            self.assertEqual(client.get(path).status_code, 401, path)

    def test_the_chapter_payload_carries_its_lecture(self) -> None:
        client = self.client()
        chapter = client.get(f"/api/chapters/{CHAPTER}").json()
        self.assertEqual(chapter["lecture"]["video"]["src"], BASE + "lecture.mp4")
        self.assertEqual(len(chapter["lecture"]["sections"]), 3)
        # A chapter whose manifest is broken has no lecture.
        self.assertIsNone(client.get("/api/chapters/02-memory-arithmetic").json()["lecture"])

    def test_list_and_get(self) -> None:
        client = self.client()
        listing = client.get("/api/lectures").json()["lectures"]
        # The lecture for a chapter that doesn't exist is left out.
        self.assertEqual([entry["slug"] for entry in listing], [CHAPTER])
        entry = listing[0]
        self.assertEqual(entry["sections"], 3)
        self.assertEqual(entry["duration"], 120.0)
        self.assertEqual(entry["chapter_title"], "FlashAttention")
        self.assertEqual(entry["poster"], BASE + "poster.webp")

        one = client.get(f"/api/lectures/{CHAPTER}")
        self.assertEqual(one.status_code, 200)
        self.assertEqual(one.json()["title"], "FlashAttention")
        for missing in ("02-memory-arithmetic", "99-no-such-chapter", "nope"):
            self.assertEqual(client.get(f"/api/lectures/{missing}").status_code, 404, missing)


class ContentManifestsTest(unittest.TestCase):
    """Every manifest in content/lectures/ parses, and its anchors point at a section."""

    def test_every_manifest_is_usable(self) -> None:
        root = Path(__file__).resolve().parents[2] / "content"
        manifests = sorted((root / "lectures").glob("*.json"))
        for path in manifests:
            with self.subTest(path.name):
                lecture = lectures.parse_manifest(json.loads(path.read_text()), path.stem)
                assert lecture is not None
                self.assertTrue(lecture["sections"])
                body = (root / "chapters" / f"{path.stem}.md").read_text()
                headings = _count_h2(body)
                for section in lecture["sections"]:
                    anchor = section["book_anchor"]
                    if anchor:
                        self.assertLessEqual(int(anchor[2:]), headings, section["id"])


def _count_h2(markdown: str) -> int:
    """How many `##` headings a chapter has outside its code fences."""
    count, fenced = 0, False
    for line in markdown.splitlines():
        if line.startswith("```"):
            fenced = not fenced
        elif not fenced and line.startswith("## "):
            count += 1
    return count


if __name__ == "__main__":
    unittest.main()
