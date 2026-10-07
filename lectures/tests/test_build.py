"""Tests for the host side of the lecture pipeline: config, ids, and publishing.

They need only the Python standard library (plus NumPy for the narration
tests, which are skipped without it) and touch neither the network nor the
repository. Run them from the repository root:

    python3 -m unittest discover -s lectures/tests -v
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

LECTURES = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("lecture_build", LECTURES / "build.py")
course = build.course

SLUG = "02-memory-arithmetic"


class CourseTest(unittest.TestCase):
    def test_slugify_matches_the_stats_rule(self):
        self.assertEqual(course.slugify("Cochran's theorem"), "cochran-s-theorem")
        self.assertEqual(course.slugify("Densities of $t$ and $F$ by transformation"),
                         "densities-of-t-and-f-by-transformation")
        self.assertEqual(course.slugify("`nn.Linear` and *bias*"), "nn-linear-and-bias")

    def test_headings_skip_fenced_code(self):
        md = "## One\n```python\n## not a heading\n```\n~~~\n## nor this\n~~~\n## Two ##\n### Three\n"
        self.assertEqual(course.headings(md), ["One", "Two"])

    def test_section_anchors_are_positional(self):
        anchors = course.section_anchors(SLUG)
        self.assertEqual(anchors["s-before-you-start"], "s-1")
        self.assertEqual(anchors["s-kv-cache"], "s-4")

    def test_chapter_order_and_numbers(self):
        slugs = course.chapter_slugs()
        self.assertLess(slugs.index("00-what-you-are-building"), slugs.index("00a-notation-and-prerequisites"))
        self.assertLess(slugs.index("00a-notation-and-prerequisites"), slugs.index("01-the-model-on-disk"))
        self.assertEqual(course.chapter_number("00a-notation-and-prerequisites"), "00a")
        self.assertEqual(course.next_chapter(SLUG)["slug"], "03-tokens-and-embeddings")
        self.assertIsNone(course.next_chapter(slugs[-1]))

    def test_front_matter_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "c.md"
            p.write_text('---\ntitle: A: b\nsummary: "Quoted: \\"yes\\""\nminutes: 85\ngpu: false\n'
                         "objectives:\n  - One.\n  - 'Two''s'\nlong: >\n  folded\n  text\n---\n# Body\n")
            fm = course.front_matter(p)
        self.assertEqual(fm["title"], "A: b")
        self.assertEqual(fm["summary"], 'Quoted: "yes"')
        self.assertEqual(fm["minutes"], 85)
        self.assertIs(fm["gpu"], False)
        self.assertEqual(fm["objectives"], ["One.", "Two's"])
        self.assertEqual(fm["long"], "folded text")

    def test_urls_come_from_course_json(self):
        cfg = course.config()
        url = course.base_url(SLUG, "abc123def456")
        self.assertEqual(url, f"{cfg['public_base'].rstrip('/')}/{cfg['key_prefix']}/{SLUG}/abc123def456/")
        self.assertEqual(course.manifest_file(SLUG), course.REPO / "content" / "lectures" / f"{SLUG}.json")


class VersionTest(unittest.TestCase):
    def test_version_is_stable_and_depends_on_quality(self):
        target = build.Target(SLUG)
        if not target.scene.exists():
            self.skipTest("no scene for the test chapter")
        self.assertEqual(build.version_of(target, "h"), build.version_of(target, "h"))
        self.assertNotEqual(build.version_of(target, "h"), build.version_of(target, "l"))
        self.assertRegex(build.version_of(target, "h"), r"^[0-9a-f]{12}$")


def fake_build(root: Path, version: str, *, mode: str = "remote", width: int = 1920, suffix: str = "",
               base_url: str | None = None) -> Path:
    """Writes a finished build of SLUG under `root` that looks like build.py's output."""
    work = root / SLUG / f"{version}{suffix}"
    dist = work / "dist"
    (dist / "slides").mkdir(parents=True)
    sections = [
        {"id": "title", "title": "Memory arithmetic", "start": 0.0, "end": 5.0,
         "slide": "slides/01-title.webp", "notes": "Hi.", "book_section": None},
        {"id": "kv", "title": "KV", "start": 5.0, "end": 9.0,
         "slide": "slides/02-kv.webp", "notes": "Bytes.", "book_section": "s-kv-cache"},
    ]
    result = {"title": "Memory arithmetic", "duration": 9.0, "width": width, "height": width * 9 // 16,
              "tts": {"mode": mode}, "sections": sections, "warnings": [], "version": version, "quality": "h"}
    (work / "done.json").write_text(json.dumps(result))
    manifest = build.make_manifest(SLUG, version, result, course.section_anchors(SLUG))
    if base_url:
        manifest["base_url"] = base_url
    (dist / "manifest.json").write_text(json.dumps(manifest))
    for rel, data in (("lecture.mp4", b"v" * 100), ("poster.webp", b"p"), ("captions.vtt", b"WEBVTT\n"),
                      ("slides/01-title.webp", b"a"), ("slides/02-kv.webp", b"b")):
        (dist / rel).write_bytes(data)
    return work


class PublishTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "build"
        self.dest = Path(self.tmp.name) / "content" / f"{SLUG}.json"
        self.version = "abc123def456"
        patcher = mock.patch.object(course, "manifest_file", lambda slug: self.dest)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)
        self.lines: list[str] = []

    def publisher(self, **kw):
        kw.setdefault("upload", False)
        kw.setdefault("dry_run", True)
        kw.setdefault("force", False)
        return build.Publisher(printer=self.lines.append, **kw)

    def test_dry_run_prints_puts_checks_and_manifest_and_writes_nothing(self):
        fake_build(self.root, self.version)
        with mock.patch.object(build, "check_urls") as checks, mock.patch.object(build.subprocess, "run") as run:
            out = self.publisher(upload=True).publish(self.root, build.Target(SLUG), self.version)
        checks.assert_not_called()
        run.assert_not_called()
        self.assertFalse(self.dest.exists())
        puts = [l for l in self.lines if l.startswith("would run: cf r2 objects put")]
        self.assertEqual(len(puts), 6)
        # Slides and captions first, then the video, then the manifest.
        self.assertIn("lecture.mp4", puts[-2])
        self.assertIn("manifest.json", puts[-1])
        self.assertIn(f"lectures/{SLUG}/{self.version}/slides/01-title.webp", puts[0] + puts[1] + puts[2] + puts[3])
        self.assertIn("--bucket-name learn-inference-lectures", puts[0])
        self.assertEqual(len([l for l in self.lines if l.startswith("would check: ")]), 6)
        written = [l for l in self.lines if l.startswith("would write ")]
        self.assertEqual(len(written), 1)
        manifest = json.loads(written[0].split("\n", 1)[1])
        self.assertEqual(manifest["base_url"], course.base_url(SLUG, self.version))
        self.assertEqual(manifest["sections"][1]["book_anchor"], "s-4")
        self.assertNotIn("book_section", manifest["sections"][0])
        self.assertEqual(out["files"], 6)

    def test_publish_without_upload_checks_urls_then_writes_the_manifest(self):
        fake_build(self.root, self.version)
        with mock.patch.object(build, "check_urls") as checks:
            self.publisher(dry_run=False).publish(self.root, build.Target(SLUG), self.version)
        urls = [c[0] for c in checks.call_args[0][0]]
        self.assertTrue(all(u.startswith(course.base_url(SLUG, self.version)) for u in urls))
        self.assertEqual(json.loads(self.dest.read_text())["version"], self.version)
        # Publishing the same version again is a no-op.
        with mock.patch.object(build, "check_urls") as checks:
            out = self.publisher(dry_run=False).publish(self.root, build.Target(SLUG), self.version)
        checks.assert_not_called()
        self.assertTrue(out["skipped"])

    def test_refuses_estimated_narration(self):
        fake_build(self.root, self.version, mode="estimate")
        with self.assertRaisesRegex(build.BuildError, "estimate"):
            self.publisher().publish(self.root, build.Target(SLUG), self.version)

    def test_refuses_estimate_and_preview_directories(self):
        fake_build(self.root, self.version, suffix="-estimate")
        with self.assertRaisesRegex(build.BuildError, "no finished build"):
            self.publisher().publish(self.root, build.Target(SLUG), self.version)

    def test_refuses_low_quality(self):
        fake_build(self.root, self.version, width=854)
        with self.assertRaisesRegex(build.BuildError, "1080p"):
            self.publisher().publish(self.root, build.Target(SLUG), self.version)

    def test_refuses_a_stale_base_url(self):
        fake_build(self.root, self.version, base_url="https://old.example/lectures/x/")
        with self.assertRaisesRegex(build.BuildError, "base_url"):
            self.publisher().publish(self.root, build.Target(SLUG), self.version)

    def test_refuses_missing_slides(self):
        work = fake_build(self.root, self.version)
        (work / "dist" / "slides" / "02-kv.webp").unlink()
        with self.assertRaisesRegex(build.BuildError, "lacks slides/02-kv.webp"):
            self.publisher().publish(self.root, build.Target(SLUG), self.version)

    def test_put_skips_identical_objects_and_never_overwrites(self):
        work = fake_build(self.root, self.version)
        path = work / "dist" / "lecture.mp4"
        pub = self.publisher(upload=True, dry_run=False)
        with mock.patch.object(build, "head", return_value=(200, "video/mp4", 100)), \
                mock.patch.object(build.subprocess, "run") as run:
            pub.put("b", "k", path, "video/mp4", "https://x/k")
        run.assert_not_called()
        with mock.patch.object(build, "head", return_value=(200, "video/mp4", 99)):
            with self.assertRaisesRegex(build.BuildError, "refusing to overwrite"):
                pub.put("b", "k", path, "video/mp4", "https://x/k")

    def test_cli_rejects_upload_without_publish(self):
        with redirect_stdout(io.StringIO()), mock.patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                build.main([SLUG, "--upload"])

    def test_cli_refuses_full_render_in_estimate_mode(self):
        if not build.Target(SLUG).scene.exists():
            self.skipTest("no scene for the test chapter")
        err = io.StringIO()
        with mock.patch.dict("os.environ", {"LECTURE_TTS": "estimate"}), mock.patch.object(sys, "stderr", err), \
                mock.patch.object(build, "ensure_image") as image:
            code = build.main([SLUG])
        self.assertEqual(code, 1)
        image.assert_not_called()
        self.assertIn("--allow-estimate", err.getvalue())


class NarrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("NumPy isn't installed")
        # Import kit.narration without kit/__init__.py, which needs manim.
        pkg = types.ModuleType("kit")
        pkg.__path__ = [str(LECTURES / "kit")]
        saved = sys.modules.get("kit")
        sys.modules["kit"] = pkg
        try:
            cls.narration = importlib.import_module("kit.narration")
        finally:
            if saved is not None:
                sys.modules["kit"] = saved

    def test_course_rules_load_and_compile(self):
        rules = self.narration.course_pronunciations()
        self.assertGreater(len(rules), 20)

    def test_spoken_forms(self):
        spoken = self.narration.spoken
        self.assertEqual(spoken("The KV cache grows 64 KiB per token."),
                         "The K V cache grows sixty-four kibibytes per token.")
        self.assertEqual(spoken("A quarter of a GiB."), "A quarter of a gibibyte.")
        self.assertEqual(spoken("bf16 on an A100 reads 2 TB/s from HBM."),
                         "B F sixteen on an A one hundred reads two terabytes per second from H B M.")
        self.assertEqual(spoken("GQA and RoPE, then RMSNorm."), "G Q A and rope, then R M S norm.")
        self.assertEqual(spoken("It takes 12 µs, about 3 × faster."),
                         "It takes twelve microseconds, about three times faster.")
        self.assertEqual(spoken("24 / 4 = 6"), "twenty-four over four equals six")
        self.assertEqual(spoken("0.25 GiB"), "zero point two five gibibytes")

    def test_numbers_become_words(self):
        spoken = self.narration.spoken
        self.assertEqual(spoken("The vocabulary holds 248,320 entries."),
                         "The vocabulary holds two hundred forty-eight thousand three hundred twenty entries.")
        self.assertEqual(spoken("Two to the 30th, and the 21st step."),
                         "Two to the thirtieth, and the twenty-first step.")
        self.assertEqual(spoken("A 100-token prompt."), "A one hundred-token prompt.")
        self.assertEqual(spoken("A rate of 1e-4."), "A rate of ten to the minus four.")
        # Running the rules twice changes nothing, so say= overrides are safe.
        once = spoken("At p99 the LM waits 176 ms, 20% of a 4096-token step.")
        self.assertEqual(spoken(once), once)

    def test_estimate_seconds(self):
        self.assertAlmostEqual(self.narration.estimate_seconds("one two three four five six seven eight nine ten "
                                                               "eleven twelve thirteen"), 5.0)
        self.assertEqual(self.narration.estimate_seconds("Hi."), 0.5)


if __name__ == "__main__":
    unittest.main()
