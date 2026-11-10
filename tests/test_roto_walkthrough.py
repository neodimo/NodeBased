"""The Roto and RotoPaint walkthrough doubles as a headless test.

tools/roto_walkthrough.py drives each guide workflow through the real viewer with mouse and key events and asserts what
the document should hold afterwards.  A scene that fails an assertion is recorded as broken; this test fails on any of
them, checks that every scene saved a screenshot, and that the README table lists every scene.
"""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent


def load_walkthrough():
    spec = importlib.util.spec_from_file_location("roto_walkthrough", ROOT / "tools" / "roto_walkthrough.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RotoWalkthroughTests(unittest.TestCase):
    def test_every_scene_runs_and_nothing_is_broken(self):
        walkthrough = load_walkthrough()
        with tempfile.TemporaryDirectory() as folder:
            results = walkthrough.run(folder)
            walkthrough.write_readme(results, folder)
            broken = [(r.number, r.title, r.error) for r in results if r.status == "broken"]
            self.assertEqual(broken, [])
            self.assertEqual(len(results), len(walkthrough.SCENES))
            for result in results:
                self.assertTrue(result.screens, f"scene {result.number} saved no screenshot")
                for name in result.screens:
                    self.assertTrue((Path(folder) / name).stat().st_size > 1000, name)
            readme = (Path(folder) / "README.md").read_text()
            for result in results:
                self.assertIn(f"| {result.number} | {result.title} |", readme)
            self.assertTrue(all(page.startswith("https://learn.foundry.com/nuke/content/") for r in results for page in r.pages))


if __name__ == "__main__":
    unittest.main()
