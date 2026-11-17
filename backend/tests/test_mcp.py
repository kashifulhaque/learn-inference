"""The MCP endpoint, its bearer tokens, and the token management API.

The chapters, lab, and lecture come from a scratch directory written by each
test, so the expectations don't change when content is added to `content/`.
Lab runs use a fake provider: no test calls RunPod or costs GPU time. Run the
tests from the repository root:

    .venv/bin/python -m unittest discover -s backend/tests -t . -v
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from unittest import mock

from fastapi.testclient import TestClient

from backend.app import curriculum, db, infra, lectures, mcp_tokens, runs
from backend.app.config import get_settings
from backend.app.main import app
from backend.app.providers import OutOfCredits

SECRET = "test-secret"
SIGNED_IN = {"Remote-User": "ada", "X-SSO-Proxy": SECRET}
OTHER = {"Remote-User": "grace", "X-SSO-Proxy": SECRET}
ACCEPT = "application/json, text/event-stream"

CHAPTER = """---
title: The KV cache
slug: 01-alpha
part: "Part 1 — Ground truth"
summary: Stop recomputing the past.
minutes: 30
gpu: false
objectives:
  - State one fact.
lab: 01-lab
---

# The KV cache

Each new token reads the cached keys.

```python
# A comment in a code block is not a heading.
```

## The block table

A block table maps logical blocks to physical ones.
"""

SECOND = (
    CHAPTER.replace("01-alpha", "02-beta")
    .replace("The KV cache", "Sampling")
    .replace("The block table", "Temperature")
    .replace("A block table maps logical blocks to physical ones.", "Temperature scales logits.")
    .replace("lab: 01-lab\n", "")
)

LECTURE = {
    "slug": "01-alpha",
    "title": "The KV cache",
    "version": "v1",
    "base_url": "https://media.example/01-alpha/v1/",
    "video": {"src": "lecture.mp4", "duration": 60, "width": 1920, "height": 1080},
    "sections": [
        {"id": "s1", "title": "Why cache", "start": 0, "end": 30, "notes": "Narration about paging."},
    ],
}

TESTS = "def check():\n    return 1\n"


def make_provider(events: list[dict[str, Any]], error: Exception | None = None) -> Any:
    class FakeProvider:
        name = "fake"

        async def run_lab(self, **kwargs: Any) -> Any:
            if error is not None:
                raise error
            for event in events:
                yield event

    return FakeProvider()


class MCPTestCase(unittest.TestCase):
    """Points the app at a scratch data directory, chapters, lab, and lecture."""

    def setUp(self) -> None:
        data = tempfile.TemporaryDirectory()
        self.addCleanup(data.cleanup)
        root = Path(data.name)
        (root / "chapters").mkdir()
        (root / "chapters" / "01-alpha.md").write_text(CHAPTER, encoding="utf-8")
        (root / "chapters" / "02-beta.md").write_text(SECOND, encoding="utf-8")
        lab = root / "labs" / "01-lab"
        lab.mkdir(parents=True)
        (lab / "meta.json").write_text(
            json.dumps({"id": "01-lab", "title": "Cache lab", "brief": "Build it.", "hints": ["Think."]}),
            encoding="utf-8",
        )
        (lab / "starter.py").write_text("def build():\n    raise NotImplementedError\n")
        (lab / "solution.py").write_text("def build():\n    return 1\n")
        (lab / "tests.py").write_text(TESTS)
        (lab / "fixture_data.py").write_text("DATA = [1, 2]\n")
        (root / "lectures").mkdir()
        (root / "lectures" / "01-alpha.json").write_text(json.dumps(LECTURE), encoding="utf-8")
        (root / "STYLE.md").write_text("# Style\n\nWrite plainly.\n", encoding="utf-8")

        env = mock.patch.dict(
            os.environ,
            {
                "DATA_DIR": str(root / "data"),
                "CONTENT_DIR": str(root),
                "FRONTEND_DIST": str(root / "missing-dist"),
                "SSO_PROXY_SECRET": SECRET,
                "SSO_USERS": "ada:Ada",
                "MCP_PUBLIC_URL": "",
                "RUNPOD_API_KEY": "",
                "RUNPOD_ENDPOINT_ID": "",
            },
        )
        env.start()
        self.addCleanup(env.stop)
        for clear in (
            get_settings.cache_clear,
            curriculum.reload_content,
            lectures.reload_lectures,
        ):
            clear()
            self.addCleanup(clear)
        # Entering the client runs the lifespan, which creates the schema and
        # starts the MCP session manager.
        self.client = TestClient(app)
        self.enterContext(self.client)

    # --- helpers ------------------------------------------------------------

    def new_token(self, name: str = "laptop", headers: dict | None = None) -> dict:
        response = self.client.post(
            "/api/mcp/tokens", json={"name": name}, headers=headers or SIGNED_IN
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def rpc(
        self,
        token: str | None,
        method: str,
        params: dict | None = None,
        path: str = "/mcp",
        extra: dict | None = None,
    ) -> Any:
        headers = {"Accept": ACCEPT, "Content-Type": "application/json"}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        headers.update(extra or {})
        body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
        return self.client.post(path, headers=headers, json=body, follow_redirects=False)

    def call(self, token: str, tool: str, headers: dict | None = None, **arguments: Any) -> Any:
        response = self.rpc(
            token, "tools/call", {"name": tool, "arguments": arguments}, extra=headers
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()["result"]
        text = result["content"][0]["text"]
        return result.get("isError", False), (text if result.get("isError") else json.loads(text))


class TokenManagementTests(MCPTestCase):
    def test_token_api_needs_sso(self) -> None:
        self.assertEqual(self.client.get("/api/mcp/tokens").status_code, 401)
        self.assertEqual(self.client.post("/api/mcp/tokens", json={"name": "x"}).status_code, 401)
        self.assertEqual(self.client.delete("/api/mcp/tokens/1").status_code, 401)

    def test_create_list_and_revoke(self) -> None:
        listed = self.client.get("/api/mcp/tokens", headers=SIGNED_IN).json()
        self.assertEqual(listed, {"url": "http://testserver/mcp", "tokens": []})

        created = self.new_token()
        self.assertEqual(set(created), {"id", "name", "prefix", "created_at", "token"})
        self.assertTrue(created["token"].startswith("infmcp_"))
        self.assertEqual(created["prefix"], created["token"][:12])

        tokens = self.client.get("/api/mcp/tokens", headers=SIGNED_IN).json()["tokens"]
        self.assertEqual(
            tokens,
            [
                {
                    "id": created["id"],
                    "name": "laptop",
                    "prefix": created["prefix"],
                    "created_at": created["created_at"],
                    "last_used_at": None,
                }
            ],
        )
        # The secret itself is never stored, only its hash, under the mapped name.
        with db.connect() as conn:
            row = dict(conn.execute("SELECT * FROM mcp_tokens").fetchone())
        self.assertNotIn(created["token"], json.dumps(row))
        self.assertEqual(row["user"], "Ada")
        self.assertEqual(len(row["token_hash"]), 64)

        # Another reader neither sees nor revokes it.
        self.assertEqual(self.client.get("/api/mcp/tokens", headers=OTHER).json()["tokens"], [])
        self.assertEqual(
            self.client.delete(f"/api/mcp/tokens/{created['id']}", headers=OTHER).status_code, 404
        )

        response = self.client.delete(f"/api/mcp/tokens/{created['id']}", headers=SIGNED_IN)
        self.assertEqual(response.json(), {"revoked": True})
        self.assertEqual(self.client.get("/api/mcp/tokens", headers=SIGNED_IN).json()["tokens"], [])
        self.assertEqual(
            self.client.delete(f"/api/mcp/tokens/{created['id']}", headers=SIGNED_IN).status_code, 404
        )
        self.assertEqual(self.rpc(created["token"], "ping").status_code, 401)

    def test_public_url_setting(self) -> None:
        with mock.patch.dict(os.environ, {"MCP_PUBLIC_URL": "https://qwen.ifkash.dev/mcp"}):
            get_settings.cache_clear()
            listed = self.client.get("/api/mcp/tokens", headers=SIGNED_IN).json()
        self.assertEqual(listed["url"], "https://qwen.ifkash.dev/mcp")

    def test_name_is_validated(self) -> None:
        for name in ("", "   ", "x" * 65):
            response = self.client.post("/api/mcp/tokens", json={"name": name}, headers=SIGNED_IN)
            self.assertEqual(response.status_code, 422, name)

    def test_at_most_ten_active_tokens(self) -> None:
        for i in range(10):
            self.new_token(f"t{i}")
        response = self.client.post("/api/mcp/tokens", json={"name": "t10"}, headers=SIGNED_IN)
        self.assertEqual(response.status_code, 409)

    def test_cli_creates_lists_and_revokes(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(mcp_tokens.main(["create", "Ada", "--name", "vm"]), 0)
        token = out.getvalue().strip().splitlines()[-1]
        self.assertTrue(token.startswith("infmcp_"))
        self.assertEqual(self.rpc(token, "ping").status_code, 200)

        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(mcp_tokens.main(["list", "Ada"]), 0)
        self.assertIn(token[:12], out.getvalue())
        self.assertNotIn(token, out.getvalue())

        token_id = db.list_tokens("Ada")[0]["id"]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(mcp_tokens.main(["revoke", str(token_id)]), 0)
        self.assertEqual(self.rpc(token, "ping").status_code, 401)


class AuthenticationTests(MCPTestCase):
    def test_rejects_missing_and_bad_tokens(self) -> None:
        response = self.rpc(None, "ping")
        self.assertEqual(response.status_code, 401)
        self.assertTrue(response.headers["www-authenticate"].startswith("Bearer"))

        response = self.rpc("infmcp_not-a-real-token", "ping")
        self.assertEqual(response.status_code, 401)
        self.assertIn('error="invalid_token"', response.headers["www-authenticate"])

        # A token of another course's shape is rejected without a lookup.
        self.assertEqual(self.rpc("rlmcp_whatever", "ping").status_code, 401)
        response = self.rpc(None, "ping", extra={"Authorization": "Basic YWRhOng="})
        self.assertEqual(response.status_code, 401)

    def test_ignores_sso_headers(self) -> None:
        self.assertEqual(self.rpc(None, "ping", extra=SIGNED_IN).status_code, 401)
        # With a valid token, SSO headers can't change who the caller is.
        token = self.new_token(headers=OTHER)["token"]
        _, progress = self.call(token, "get_progress", headers=SIGNED_IN)
        self.assertEqual(progress["reader"], "grace")

    def test_last_used_is_recorded(self) -> None:
        token = self.new_token()["token"]
        self.assertEqual(self.rpc(token, "ping").status_code, 200)
        tokens = self.client.get("/api/mcp/tokens", headers=SIGNED_IN).json()["tokens"]
        self.assertIsNotNone(tokens[0]["last_used_at"])


class ProtocolTests(MCPTestCase):
    def test_initialize_and_list_tools(self) -> None:
        token = self.new_token()["token"]
        for path in ("/mcp", "/mcp/"):
            response = self.rpc(
                token,
                "initialize",
                {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1"},
                },
                path=path,
            )
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()["result"]
            self.assertEqual(result["protocolVersion"], "2025-06-18")
            self.assertIn("tools", result["capabilities"])

        tools = self.rpc(token, "tools/list").json()["result"]["tools"]
        self.assertEqual(
            sorted(tool["name"] for tool in tools),
            sorted(
                [
                    "course_overview", "list_chapters", "get_chapter", "search", "get_lab",
                    "get_solution", "list_lectures", "get_lecture", "get_document",
                    "get_book_info", "get_providers", "get_compute", "get_progress",
                    "set_progress", "get_note", "save_note", "get_draft", "save_draft",
                    "list_runs", "get_run", "run_lab",
                ]
            ),
        )
        by_name = {tool["name"]: tool for tool in tools}
        self.assertTrue(by_name["get_chapter"]["annotations"]["readOnlyHint"])
        self.assertFalse(by_name["run_lab"]["annotations"]["readOnlyHint"])
        self.assertIn("costs real money", by_name["run_lab"]["description"])

        templates = self.rpc(token, "resources/templates/list").json()["result"]
        self.assertEqual(
            sorted(t["uriTemplate"] for t in templates["resourceTemplates"]),
            ["inference://chapters/{slug}", "inference://labs/{lab_id}"],
        )


class ContentToolTests(MCPTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.token = self.new_token()["token"]

    def test_overview(self) -> None:
        _, overview = self.call(self.token, "course_overview")
        self.assertEqual(overview["reader"], "Ada")
        self.assertEqual(overview["counts"], {"parts": 1, "chapters": 2, "labs": 1, "lectures": 1})
        first = overview["parts"][0]["chapters"][0]
        self.assertEqual((first["slug"], first["lab"], first["has_lecture"]), ("01-alpha", "01-lab", True))
        self.assertEqual(sorted(overview["documents"]), ["FACTS.md", "STYLE.md", "lab_common.py"])

    def test_get_chapter_does_not_mark_progress(self) -> None:
        error, chapter = self.call(self.token, "get_chapter", slug="01-alpha")
        self.assertFalse(error)
        self.assertIn("cached keys", chapter["body"])
        self.assertEqual((chapter["next"], chapter["lab"], chapter["lab_title"]), ("02-beta", "01-lab", "Cache lab"))
        self.assertEqual(chapter["lecture"]["sections"], 1)
        self.assertIsNone(chapter["status"])
        self.assertEqual(db.get_progress("Ada"), {})

        error, message = self.call(self.token, "get_chapter", slug="nope")
        self.assertTrue(error)
        self.assertIn("No chapter", message)

        response = self.rpc(self.token, "resources/read", {"uri": "inference://chapters/02-beta"})
        self.assertIn("Temperature scales", response.json()["result"]["contents"][0]["text"])

    def test_list_chapters_headings_skip_code_blocks(self) -> None:
        _, listed = self.call(self.token, "list_chapters", part="part 1")
        self.assertEqual(listed["count"], 2)
        self.assertEqual(listed["chapters"][0]["headings"], ["The block table"])
        _, none = self.call(self.token, "list_chapters", part="part 9")
        self.assertEqual(none["count"], 0)

    def test_search_covers_chapters_labs_and_lectures(self) -> None:
        _, found = self.call(self.token, "search", query="BLOCK table")
        first = found["results"][0]
        self.assertEqual((first["kind"], first["slug"]), ("chapter", "01-alpha"))
        self.assertEqual(first["section"], "The block table")

        _, found = self.call(self.token, "search", query="paging")
        self.assertEqual([(r["kind"], r["slug"]) for r in found["results"]], [("lecture", "01-alpha")])
        self.assertEqual(found["results"][0]["section"], "Why cache")

        _, found = self.call(self.token, "search", query="think")
        self.assertEqual([(r["kind"], r["slug"]) for r in found["results"]], [("lab", "01-lab")])

        _, found = self.call(self.token, "search", query="the", limit=1)
        self.assertGreater(found["total"], 1)
        self.assertEqual(len(found["results"]), 1)
        error, _ = self.call(self.token, "search", query="  ")
        self.assertTrue(error)

    def test_lab_and_solution(self) -> None:
        _, lab = self.call(self.token, "get_lab", lab_id="01-lab")
        self.assertIn("NotImplementedError", lab["starter"])
        self.assertEqual(lab["tests"], TESTS)
        self.assertEqual(lab["fixtures"], {"fixture_data.py": "DATA = [1, 2]\n"})
        self.assertNotIn("solution", lab)
        self.assertEqual(lab["chapters"], ["01-alpha"])
        _, solution = self.call(self.token, "get_solution", lab_id="01-lab")
        self.assertIn("return 1", solution["solution"])
        error, _ = self.call(self.token, "get_lab", lab_id="nope")
        self.assertTrue(error)

        # The site's own lab payload still withholds the checks.
        page = self.client.get("/api/labs/01-lab", headers=SIGNED_IN).json()
        self.assertNotIn("tests", page)
        self.assertNotIn("fixtures", page)
        self.assertNotIn("solution", page)

    def test_lectures(self) -> None:
        _, listed = self.call(self.token, "list_lectures")
        self.assertEqual([item["slug"] for item in listed["lectures"]], ["01-alpha"])
        _, lecture = self.call(self.token, "get_lecture", slug="01-alpha")
        self.assertEqual(lecture["video"]["src"], "https://media.example/01-alpha/v1/lecture.mp4")
        self.assertEqual(lecture["sections"][0]["notes"], "Narration about paging.")
        error, _ = self.call(self.token, "get_lecture", slug="02-beta")
        self.assertTrue(error)

    def test_documents(self) -> None:
        _, style = self.call(self.token, "get_document", name="style.md")
        self.assertEqual(style["name"], "STYLE.md")
        self.assertIn("Write plainly", style["markdown"])

        _, facts = self.call(self.token, "get_document", name="FACTS.md")
        self.assertIn("sections", facts)  # The real fact sheet is too long to send whole.
        heading = facts["sections"][0]
        _, part = self.call(self.token, "get_document", name="FACTS.md", section=heading)
        self.assertEqual(part["section"], heading)
        self.assertTrue(part["markdown"].startswith("#"))
        error, _ = self.call(self.token, "get_document", name="FACTS.md", section="no such heading")
        self.assertTrue(error)

        for name in ("../README.md", "meta.json", "lab_common"):
            error, _ = self.call(self.token, "get_document", name=name)
            self.assertTrue(error, name)

    def test_book_providers_and_compute(self) -> None:
        _, book = self.call(self.token, "get_book_info")
        self.assertEqual(book, {"book": None})
        _, providers = self.call(self.token, "get_providers")
        self.assertEqual(providers["default"], "runpod")
        self.assertFalse(providers["providers"][0]["available"])
        snapshot = {"generated_at": 1.0, "active": 0, "providers": [], "runs": []}
        with mock.patch.object(infra, "snapshot", mock.AsyncMock(return_value=snapshot)):
            _, compute = self.call(self.token, "get_compute")
        self.assertEqual(compute, snapshot)


class ReaderStateTests(MCPTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.token = self.new_token()["token"]

    def test_progress_notes_and_drafts_act_as_the_token_owner(self) -> None:
        _, saved = self.call(self.token, "save_draft", lab_id="01-lab", code="x = 1\n")
        self.assertTrue(saved["saved"])
        self.assertEqual(db.get_draft("Ada", "01-lab"), "x = 1\n")
        _, draft = self.call(self.token, "get_draft", lab_id="01-lab")
        self.assertEqual(draft["code"], "x = 1\n")

        self.call(self.token, "save_note", chapter="01-alpha", body="Remember paging.")
        self.assertEqual(db.get_note("Ada", "01-alpha"), "Remember paging.")
        _, note = self.call(self.token, "get_note", chapter="01-alpha")
        self.assertEqual(note["note"], "Remember paging.")
        _, chapter = self.call(self.token, "get_chapter", slug="01-alpha")
        self.assertEqual(chapter["note"], "Remember paging.")

        _, marked = self.call(self.token, "set_progress", chapter="02-beta", status="in_progress")
        self.assertEqual(marked["status"], "in_progress")
        self.call(self.token, "set_progress", chapter="01-alpha", status="done")
        _, progress = self.call(self.token, "get_progress")
        self.assertEqual(progress["summary"]["done"], 1)
        self.assertEqual(
            [(row["slug"], row["status"]) for row in progress["chapters"]],
            [("01-alpha", "done"), ("02-beta", "in_progress")],
        )
        self.assertEqual(self.db_progress(), {"01-alpha": "done", "02-beta": "in_progress"})

        error, _ = self.call(self.token, "set_progress", chapter="nope", status="done")
        self.assertTrue(error)
        error, _ = self.call(self.token, "set_progress", chapter="01-alpha", status="solved")
        self.assertTrue(error)
        error, _ = self.call(self.token, "save_draft", lab_id="nope", code="x")
        self.assertTrue(error)

        # Another reader sees none of it.
        other = self.new_token(headers=OTHER)["token"]
        _, theirs = self.call(other, "get_draft", lab_id="01-lab")
        self.assertIsNone(theirs["code"])

    def db_progress(self) -> dict[str, str]:
        return db.get_progress("Ada")


class RunLabTests(MCPTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.token = self.new_token()["token"]

    def run_with(self, provider: Any) -> tuple[bool, Any]:
        with mock.patch.object(runs, "get_provider", lambda name=None: provider):
            return self.call(self.token, "run_lab", lab_id="01-lab", code="print('hi')\n")

    def test_a_passing_run_is_recorded_like_the_editor_does(self) -> None:
        provider = make_provider(
            [
                {"type": "job", "provider": "fake", "job_id": "job-1"},
                {"type": "log", "line": "running"},
                {
                    "type": "result",
                    "passed": True,
                    "checks": [{"name": "builds", "passed": True, "detail": "ok"}],
                    "metrics": {"speedup": 2.5},
                },
            ]
        )
        error, result = self.run_with(provider)
        self.assertFalse(error, result)
        self.assertTrue(result["passed"])
        self.assertEqual(result["checks"][0]["name"], "builds")
        self.assertEqual(result["metrics"], {"speedup": 2.5})
        self.assertEqual(result["log"], "running")
        self.assertEqual(result["errors"], [])

        self.assertEqual(db.get_draft("Ada", "01-lab"), "print('hi')\n")
        stored = db.get_run("Ada", result["run_id"])
        assert stored is not None
        self.assertEqual((stored["status"], stored["passed"], stored["job_id"]), ("finished", True, "job-1"))
        # Like the editor's Run button, a pass doesn't mark the chapter done.
        self.assertEqual(db.get_progress("Ada"), {})

        _, listed = self.call(self.token, "list_runs", lab_id="01-lab")
        self.assertEqual([run["id"] for run in listed["runs"]], [result["run_id"]])
        _, one = self.call(self.token, "get_run", run_id=result["run_id"])
        self.assertEqual(one["log"], "running")
        error, _ = self.call(self.token, "get_run", run_id="nope")
        self.assertTrue(error)

    def test_a_provider_error_is_reported_and_recorded(self) -> None:
        error, result = self.run_with(make_provider([{"type": "error", "message": "worker died"}]))
        self.assertFalse(error)
        self.assertIsNone(result["passed"])
        self.assertEqual(result["errors"], ["worker died"])
        stored = db.get_run("Ada", result["run_id"])
        assert stored is not None
        self.assertEqual(stored["status"], "error")

    def test_out_of_credits(self) -> None:
        _, result = self.run_with(make_provider([], error=OutOfCredits("no credit")))
        self.assertTrue(result["out_of_credits"])
        self.assertIn("no credit", result["errors"][0])
        stored = db.get_run("Ada", result["run_id"])
        assert stored is not None
        self.assertEqual(stored["status"], "out_of_credits")

    def test_without_a_runpod_key_nothing_is_sent(self) -> None:
        # The real provider, with no key configured: it fails before any request.
        error, result = self.call(self.token, "run_lab", lab_id="01-lab", code="x = 1\n")
        self.assertFalse(error)
        self.assertEqual(result["provider"], "runpod")
        self.assertIn("RUNPOD_API_KEY", result["errors"][0])

    def test_unknown_lab_and_oversized_code(self) -> None:
        error, _ = self.call(self.token, "run_lab", lab_id="nope", code="x")
        self.assertTrue(error)
        error, message = self.call(self.token, "run_lab", lab_id="01-lab", code="x" * 200_001)
        self.assertTrue(error)
        self.assertIn("longer than", message)


if __name__ == "__main__":
    unittest.main()
