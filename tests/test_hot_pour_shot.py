"""The small Hot pour shot graph retains its holdout and catcher through evaluation."""
import unittest
from pathlib import Path

import numpy as np

from nodebased import core, scene3d
from nodebased.imaging import Evaluator
ROOT = Path(__file__).resolve().parents[1] / "examples" / "hot_pour_shot"


class HotPourShotTests(unittest.TestCase):
    def test_small_shot_evaluates_with_catcher_and_holdout(self):
        doc = core.load_document(ROOT / "hot_pour_shot.nbcomp")
        doc["nodes"]["render"]["params"].update(width=48, height=27, render_backend="cpu", pt_samples=2)
        scene = Evaluator().evaluate_raster(doc, "scene", frame=1, typed=True)
        self.assertTrue(any(g.shadow_catcher for g in scene.geometries))
        self.assertTrue(any(g.holdout for g in scene.geometries))
        image = Evaluator().evaluate_raster(doc, "render", frame=1).pixels
        self.assertEqual(image.shape, (27, 48, 4))
        self.assertTrue(np.isfinite(image).all())
        self.assertGreater(float(image[..., 3].max()), 0.0)


if __name__ == "__main__":
    unittest.main()
