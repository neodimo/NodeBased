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

    def test_past_nuke_scenes_and_the_shape_list_scene(self):
        """Scenes 17 to 19 (snap and fit to edges, track points, fill in-betweens) run on synthetic pictures and work; scene 8 works
        now that the list shows the front-most shape first, and the View column is reported as not applicable."""
        walkthrough = load_walkthrough()
        with tempfile.TemporaryDirectory() as folder:
            results = {r.number: r for r in walkthrough.run(folder, only={8, 17, 18, 19})}
            walkthrough.write_readme(list(results.values()), folder)
            self.assertEqual(sorted(results), [8, 17, 18, 19])
            for number, result in results.items():
                self.assertEqual((number, result.status, result.error), (number, "works", ""))
            self.assertTrue(any("not applicable" in note and "View" in note for note in results[8].notes))
            self.assertFalse(any("differs" in note for note in results[8].notes))
            self.assertEqual(len(results[17].screens), 3)
            self.assertEqual(len(results[18].screens), 4)
            self.assertEqual(len(results[19].screens), 2)
            for number in (17, 18, 19):
                result = results[number]
                self.assertEqual(len(result.captions), len(result.screens))
                self.assertTrue(all(len(caption) > 20 for caption in result.captions), result.captions)
                self.assertTrue(all(page.startswith("https://learn.foundry.com/nuke/content/") for page in result.pages))
            readme = (Path(folder) / "README.md").read_text()
            self.assertIn("## Past Nuke", readme)
            self.assertLess(readme.index("| 8 |"), readme.index("## Past Nuke"))
            for number in (17, 18, 19):
                self.assertGreater(readme.index(f"| {number} |"), readme.index("## Past Nuke"))
            self.assertIn("<sub>", readme)

    def test_the_synthetic_pictures_regenerate_exactly(self):
        walkthrough = load_walkthrough()
        a = walkthrough.stretching_disc(5, 64, 48, 200, 150)
        b = walkthrough.stretching_disc(5, 64, 48, 200, 150)
        self.assertTrue((a == b).all())
        self.assertGreater(float(abs(a - walkthrough.stretching_disc(1, 64, 48, 200, 150)).max()), 0.05)     # the frames differ


if __name__ == "__main__":
    unittest.main()
