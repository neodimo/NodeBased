"""Lane 6 A2: fluid artist presets are complete, lightweight graph recipes."""
import unittest
from pathlib import Path

import numpy as np

from nodebased import presets
from nodebased.core import Dispatcher, validate
from nodebased.imaging import Evaluator


class _Pos:
    def x(self):
        return 0

    def y(self):
        return 0


class FluidArtistPresetTests(unittest.TestCase):
    def setUp(self):
        self.presets, errors = presets.load_all(user_directory=Path("/nonexistent"))
        self.fluids = [preset for preset in self.presets if preset.category == "Fluids"]
        self.assertFalse(errors, errors)

    def test_shipped_fluid_presets_solve_five_frames_and_render_pixels(self):
        expected = {"Candle", "Campfire", "Explosion", "Smoke Column", "Dust Hit",
                    "Dam break", "Pour into a glass", "Honey drip", "Ocean splash",
                    "Floating Block", "Moving Rigid Collider"}
        self.assertEqual({preset.name for preset in self.fluids}, expected)
        for preset in self.fluids:
            with self.subTest(preset=preset.name):
                ops = presets.build_ops(preset, [], {}, _Pos())
                dispatcher = Dispatcher()
                dispatcher.execute({"op": "batch", "commands": ops})
                validate(dispatcher.document)
                nodes = dispatcher.document["nodes"]
                solvers = [node for node in nodes.values()
                           if node["type"] in ("FluidSolver3D", "FluidLiquidSolver3D")]
                self.assertEqual(len(solvers), 1)
                self.assertEqual(solvers[0]["params"]["pressure"], "cpu")
                resolution_key = "division_size"
                self.assertGreaterEqual(solvers[0]["params"][resolution_key], 0.2)
                render = next(node_id for node_id, node in nodes.items()
                              if node["type"] == "Render3D")
                image = Evaluator().evaluate_raster(dispatcher.document, render, frame=5)
                self.assertGreater(np.count_nonzero(image.pixels), 0,
                                   f"{preset.name} rendered an empty frame")

    def test_specialty_presets_expose_their_promised_controls(self):
        manifests = {preset.name: preset for preset in self.fluids}
        explosion = manifests["Explosion"]
        source = next(op for op in explosion.ops if op.get("type") == "FluidSolver3D")
        self.assertEqual(source["params"]["fire"], 1)
        self.assertGreater(source["params"]["gas_release"], 0)

        dust = manifests["Dust Hit"]
        self.assertTrue(any(op.get("type") == "FluidCollide3D" and op["params"]["animated"]
                            for op in dust.ops))
        honey = manifests["Honey drip"]
        liquid = next(op for op in honey.ops if op.get("type") == "FluidLiquidSolver3D")
        self.assertGreater(liquid["params"]["viscosity"], 0)
        splash = manifests["Ocean splash"]
        liquid = next(op for op in splash.ops if op.get("type") == "FluidLiquidSolver3D")
        self.assertGreater(liquid["params"]["narrow_band"], 0)
        self.assertTrue(any(op.get("type") == "FluidWhitewater3D" for op in splash.ops))
        rigid = manifests["Floating Block"]
        self.assertTrue(any(op.get("type") == "RigidBody3D" and op["params"]["density"] < 1000
                            for op in rigid.ops))
        self.assertTrue(any(op.get("type") == "RigidSolver3D" for op in rigid.ops))
        smoke = manifests["Moving Rigid Collider"]
        self.assertTrue(any(op.get("type") == "FluidCollide3D" and op["params"]["animated"]
                            for op in smoke.ops))


if __name__ == "__main__":
    unittest.main()
