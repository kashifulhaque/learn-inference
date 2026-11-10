"""The progress summary endpoint and its path-scoped CORS headers.

The chapters come from a scratch directory written by each test, so the
expected counts don't change when content is added to `content/chapters/`.
Run them like the other tests, from the repository root:

    .venv/bin/python -m unittest discover -s backend/tests -t . -v
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from backend.app import curriculum, db
from backend.app.config import get_settings
from backend.app.main import app

HUB = "https://courses.ifkash.dev"
SLUGS = ("00-intro", "01-next", "02-last")
SIGNED_IN = {"Remote-User": "ada", "X-SSO-Proxy": "test-secret"}


class ProgressSummaryTestCase(unittest.TestCase):
    """Points the app at a scratch data directory and three scratch chapters."""

    def setUp(self) -> None:
        data = tempfile.TemporaryDirectory()
        self.addCleanup(data.cleanup)
        chapters = Path(data.name) / "chapters"
        chapters.mkdir()
        for slug in SLUGS:
            (chapters / f"{slug}.md").write_text(
                f"---\ntitle: Title of {slug}\nslug: {slug}\n---\n\nBody.\n"
            )
        env = mock.patch.dict(
            os.environ,
            {"DATA_DIR": data.name, "SSO_PROXY_SECRET": "test-secret", "SSO_USERS": "ada:Ada"},
        )
        env.start()
        self.addCleanup(env.stop)
        folder = mock.patch.object(curriculum, "_chapters_dir", lambda: chapters)
        folder.start()
        self.addCleanup(folder.stop)
        for clear in (get_settings.cache_clear, curriculum.reload_content):
            clear()
            self.addCleanup(clear)

    def client(self, signed_in: bool = True) -> TestClient:
        client = TestClient(app, headers=SIGNED_IN if signed_in else {})
        # Entering the client runs the startup hook, which creates the schema.
        self.enterContext(client)
        return client

    def seed(self) -> None:
        now = time.time()
        with db.connect() as conn:
            conn.executemany(
                "INSERT INTO progress (user, chapter, status, updated_at) VALUES (?,?,?,?)",
                [
                    ("Ada", "00-intro", "done", now - 30),
                    ("Ada", "01-next", "in_progress", now - 20),
                    # Not in the current list, and newest: ignored.
                    ("Ada", "99-removed", "done", now),
                    # Another user's row: ignored.
                    ("Grace", "02-last", "done", now),
                ],
            )


class SummaryTest(ProgressSummaryTestCase):
    def test_counts_and_last(self) -> None:
        client = self.client()
        self.seed()
        reply = client.get("/api/progress/summary")
        self.assertEqual(reply.status_code, 200)
        self.assertEqual(
            reply.json(),
            {
                "done": 1,
                "total": 3,
                "last": {"slug": "01-next", "title": "Title of 01-next"},
            },
        )

    def test_no_progress(self) -> None:
        reply = self.client().get("/api/progress/summary")
        self.assertEqual(reply.json(), {"done": 0, "total": 3, "last": None})


class CorsTest(ProgressSummaryTestCase):
    def test_signed_out_is_401_with_cors_for_the_hub(self) -> None:
        reply = self.client(signed_in=False).get(
            "/api/progress/summary", headers={"Origin": HUB}
        )
        self.assertEqual(reply.status_code, 401)
        self.assertEqual(reply.headers["access-control-allow-origin"], HUB)
        self.assertEqual(reply.headers["access-control-allow-credentials"], "true")
        self.assertIn("Origin", reply.headers["vary"])

    def test_signed_in_response_carries_cors_for_the_hub(self) -> None:
        reply = self.client().get("/api/progress/summary", headers={"Origin": HUB})
        self.assertEqual(reply.headers["access-control-allow-origin"], HUB)
        self.assertEqual(reply.headers["access-control-allow-credentials"], "true")

    def test_other_origin_gets_no_allow_origin(self) -> None:
        client = self.client()
        for origin in ("https://evil.example", HUB + ".evil.example", "http://courses.ifkash.dev"):
            reply = client.get("/api/progress/summary", headers={"Origin": origin})
            self.assertNotIn("access-control-allow-origin", reply.headers)
            self.assertNotIn("access-control-allow-credentials", reply.headers)
            self.assertIn("Origin", reply.headers["vary"])

    def test_other_paths_get_no_cors_headers(self) -> None:
        reply = self.client().get("/api/health", headers={"Origin": HUB})
        self.assertEqual(reply.status_code, 200)
        self.assertNotIn("access-control-allow-origin", reply.headers)
        self.assertNotIn("Origin", reply.headers.get("vary", ""))


if __name__ == "__main__":
    unittest.main()
