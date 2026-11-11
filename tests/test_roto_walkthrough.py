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
                if result.error.startswith("skipped"):      # scene 25 waits for Reduce keys to be on the branch
                    self.assertEqual(result.number, 25)
                    continue
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

    def test_the_second_tour_of_the_past_nuke_tools(self):
        """Scenes 20 to 25 (ripple edit, outline with holes, GPU mattes, stick to plate, shapes from matte, reduce keys) run through the
        real viewer on pictures computed from formulas; each has a caption per screenshot and the README leads with a summary."""
        walkthrough = load_walkthrough()
        with tempfile.TemporaryDirectory() as folder:
            results = {r.number: r for r in walkthrough.run(folder, only={20, 21, 22, 23, 24, 25})}
            walkthrough.write_readme(list(results.values()), folder)
            self.assertEqual(sorted(results), [20, 21, 22, 23, 24, 25])
            for number in (20, 21, 22, 23):
                self.assertEqual((number, results[number].status, results[number].error), (number, "works", ""))
            self.assertNotEqual(results[24].status, "broken", results[24].error)        # works, or differs with the measured drift
            self.assertTrue(any("worst frame" in note for note in results[24].notes), results[24].notes)
            if hasattr(walkthrough.Window, "reduce_roto_keys"):
                self.assertEqual((results[25].status, results[25].error), ("works", ""))
                self.assertTrue(any("Reduce keys at 0.5 px" in note for note in results[25].notes))
            else:
                self.assertIn("not on this branch yet", results[25].error)
            self.assertTrue(any("same offset" in note for note in results[20].notes))
            self.assertTrue(any("3 holes" in note or "2 holes" in note for note in results[21].notes))
            self.assertTrue(any("largest difference between processor and card" in note for note in results[22].notes))
            self.assertTrue(any("mean error" in note for note in results[23].notes))
            for number, result in results.items():
                if result.error.startswith("skipped"):
                    continue
                self.assertEqual(len(result.captions), len(result.screens), number)
                self.assertTrue(all(len(caption) > 20 for caption in result.captions), result.captions)
                for name in result.screens:
                    self.assertTrue((Path(folder) / name).stat().st_size > 1000, name)
            self.assertEqual(len(results[20].screens), 6)
            self.assertEqual(len(results[21].screens), 3)
            self.assertEqual(len(results[23].screens), 4)
            readme = (Path(folder) / "README.md").read_text()
            self.assertIn("## Past Nuke", readme)
            self.assertIn("## Summary of the past-Nuke tools", readme)
            self.assertLess(readme.index("## Summary of the past-Nuke tools"), readme.index("| 20 |"))
            for number in (20, 21, 22, 23, 24, 25):
                self.assertGreater(readme.index(f"| {number} |"), readme.index("## Past Nuke"))
                self.assertIn(f"**{number}. ", readme)
            self.assertIn("Measured:", readme)

    def test_the_synthetic_pictures_regenerate_exactly(self):
        walkthrough = load_walkthrough()
        a = walkthrough.stretching_disc(5, 64, 48, 200, 150)
        b = walkthrough.stretching_disc(5, 64, 48, 200, 150)
        self.assertTrue((a == b).all())
        self.assertGreater(float(abs(a - walkthrough.stretching_disc(1, 64, 48, 200, 150)).max()), 0.05)     # the frames differ
        for make, frames in ((walkthrough.green_screen, (1, 7)), (walkthrough.bending_plate, (1, 10))):
            first, again = make(frames[0]), make(frames[0])
            self.assertTrue((first == again).all())
            self.assertGreater(float(abs(first - make(frames[1])).max()), 0.05)
        self.assertTrue((walkthrough.stick_texture() == walkthrough.stick_texture()).all())
        truth = walkthrough.bend_truth(10, [(100.5, 90.5), (300.5, 200.5)])      # where painted points sit on frame 10: the map is exact
        for (x, y), (px, py) in zip(truth, [(100.5, 90.5), (300.5, 200.5)]):
            bx, by = walkthrough.bend_map(10, x - 0.5, y - 0.5)
            self.assertAlmostEqual(float(bx), px - 0.5, places=3)
            self.assertAlmostEqual(float(by), py - 0.5, places=3)


if __name__ == "__main__":
    unittest.main()
