"""The small Hot pour shot graph retains its holdout and catcher through evaluation."""
import unittest
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from nodebased import core, scene3d
from nodebased.imaging import Evaluator
ROOT = Path(__file__).resolve().parents[1] / "examples" / "hot_pour_shot"


class HotPourShotTests(unittest.TestCase):
    def test_frame_one_evaluates_with_each_pressure_selection(self):
        from nodebased import fluid_gpu_solver

        pressure_modes = ["auto", "cpu"]
        if fluid_gpu_solver.available():
            pressure_modes.append("resident")
        for pressure in pressure_modes:
            with self.subTest(pressure=pressure), patch.dict("os.environ", {"NODEBASED_SIM_CACHE": "0"}):
                doc = core.load_document(ROOT / "hot_pour_shot.nbcomp")
                for key in ("liquid", "steam"):
                    doc["nodes"][key]["params"]["pressure"] = pressure
                doc["nodes"]["render"]["params"].update(
                    width=48, height=27, render_backend="cpu", pt_samples=2)
                doc["nodes"]["plate_fit"]["params"].update(width=48, height=27)
                scene = Evaluator().evaluate_raster(doc, "scene", frame=1, typed=True)
                self.assertTrue(any(g.material == "liquid" and len(g.vertices) > 0
                                    for g in scene.geometries))

    def test_small_shot_evaluates_with_catcher_and_holdout(self):
        started = time.perf_counter()
        doc = core.load_document(ROOT / "hot_pour_shot.nbcomp")
        nodes = doc["nodes"]
        cup = nodes["front_cup"]["params"]
        source = nodes["spout"]["params"]
        self.assertEqual(nodes["liquid"]["params"]["max_size"], 64)
        self.assertEqual(cup["holdout"], "on")
        self.assertAlmostEqual(cup["tx"], 0.0)
        self.assertAlmostEqual(cup["tz"], 0.0)
        self.assertIsNone(nodes["scene"]["inputs"]["object3"],
                          "the plate supplies the glass, so the transparent render must not draw a second one")
        self.assertGreater(source["src_center_y"], cup["ty"] + cup["cyl_height"] / 2)
        self.assertEqual(nodes["surface"]["params"]["material"], "liquid")
        source_x = [point["value"] for point in doc["animation"]["curves"]["spout"]["tx"]["keys"]]
        self.assertTrue(all(abs(x - cup["tx"]) < cup["cyl_radius"] for x in source_x))
        doc["nodes"]["render"]["params"].update(width=48, height=27, render_backend="cpu", pt_samples=2)
        doc["nodes"]["plate_fit"]["params"].update(width=48, height=27)
        # the example pins both solvers to the card ("resident"); this check runs where there is none (CI, 10/15 red)
        doc["nodes"]["liquid"]["params"]["pressure"] = "auto"
        doc["nodes"]["steam"]["params"]["pressure"] = "auto"
        scene = Evaluator().evaluate_raster(doc, "scene", frame=1, typed=True)
        self.assertTrue(any(g.shadow_catcher for g in scene.geometries))
        self.assertTrue(any(g.holdout for g in scene.geometries))
        self.assertTrue(any(g.material == "liquid" and len(g.vertices) > 0 for g in scene.geometries),
                        "the shot renders the cached liquid surface mesh")
        self.assertTrue(scene.volumes, "the steam volume is connected into the shot")
        raw = Evaluator().evaluate_raster(doc, "render", frame=1).pixels
        self.assertEqual(raw.shape, (27, 48, 4))
        self.assertTrue(np.isfinite(raw).all())
        self.assertGreater(float(raw[..., 3].max()), 0.0)
        self.assertGreater(int(((raw[..., :3].max(axis=2) < .01) & (raw[..., 3] > .01)).sum()), 0,
                           "the catcher contributes a black shadow layer")

        # Toggle each switch in the same tiny shot and ensure it changes rendered pixels.
        evaluator = Evaluator()
        caught = evaluator.evaluate_raster(doc, "comp", frame=1).pixels
        self.assertEqual(caught.shape, (27, 48, 4))
        cup = doc["nodes"]["front_cup"]["params"]
        cup["holdout"] = "off"
        cup_visible = evaluator.evaluate_raster(doc, "comp", frame=1).pixels
        self.assertGreater(int((np.abs(cup_visible[..., :3] - caught[..., :3]).max(axis=2) > .01).sum()), 0)
        cup["holdout"] = "on"
        table = doc["nodes"]["table_top"]["params"]
        table["shadow_catcher"] = "off"
        ordinary_table = evaluator.evaluate_raster(doc, "comp", frame=1).pixels
        self.assertGreater(int((np.abs(ordinary_table[..., :3] - caught[..., :3]).max(axis=2) > .01).sum()), 0)
        self.assertLess(time.perf_counter() - started, 60.0)


if __name__ == "__main__":
    unittest.main()
