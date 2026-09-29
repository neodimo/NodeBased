import unittest

import numpy as np

from nodebased.core import Dispatcher, SPECS, bypass_slot
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor


class OpticalEffectsTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((32, 32, 4), np.float32)
        self.image[..., 3] = 1
        self.image[16, 16, :3] = 4

    def test_flare_has_centre_and_ghosts_on_centre_axis(self):
        p = dict(SPECS["Flare"]["params"], position_x=10, position_y=14,
                 size=1, streaks=0, ghosts=2, spread=.5, brightness=1)
        out = Evaluator._flare(np.zeros_like(self.image), p)
        # The optical axis is the line from (10, 14) through the frame centre (16, 16).
        self.assertGreater(float(out[13, 7, 0]), 0)
        self.assertGreater(float(out[13, 7, 0]), float(out[11, 10, 0]))

    def test_glint_respects_highlight_tolerance_and_ray_count(self):
        p = dict(SPECS["Glint"]["params"], tolerance=1, rays=4, length=12)
        low = np.zeros_like(self.image)
        low[..., 3] = 1
        self.assertTrue(np.array_equal(Evaluator._glint(low, p), low))
        high = low.copy(); high[16, 16, :3] = 4
        out = Evaluator._glint(high, p)
        self.assertGreater(float(out[..., 0].sum()), float(high[..., 0].sum()))

    def test_sparkles_are_seeded_and_frame_animated(self):
        p = dict(SPECS["Sparkles"]["params"], tolerance=0, density=.2, size=2, seed=23)
        plate = self.image.copy(); plate[..., :3] = 4
        a = Evaluator._sparkles(plate, p, frame=3)
        b = Evaluator._sparkles(plate, p, frame=3)
        c = Evaluator._sparkles(plate, p, frame=4)
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, c))

    def test_godrays_zero_decay_is_identity_and_moves_energy_inward(self):
        image = np.zeros((9, 17, 4), np.float32); image[..., 3] = 1
        image[4, 2, :3] = 1
        p = dict(SPECS["GodRays"]["params"], center_x=8, center_y=4, steps=8, translate=1)
        p["decay"] = 0
        self.assertTrue(np.array_equal(Evaluator._godrays(image, p), image))
        p["decay"] = .9
        out = Evaluator._godrays(image, p)
        self.assertGreater(float(out[4, 5, 0]), float(image[4, 5, 0]))

    def test_scanned_grain_matches_plate_channel_variances(self):
        rng = np.random.default_rng(7)
        plate = rng.normal(0, [0.04, 0.06, 0.08, 0], (64, 64, 4)).astype(np.float32)
        image = np.ones_like(plate); image[..., 3] = 1
        p = dict(SPECS["ScannedGrain"]["params"], amount=1, response=1, seed=2)
        out = Evaluator._scanned_grain(image, plate, p, frame=5)
        for c in range(3):
            ratio = float(out[..., c].var() / plate[..., c].var())
            self.assertLess(abs(ratio - 1), .10)

    def test_maskmix_and_bypass_contracts_are_registered(self):
        for kind in ("Flare", "Glint", "Sparkles", "GodRays", "VolumeRays", "ScannedGrain"):
            self.assertIn("mix", SPECS[kind]["params"])
            self.assertIn("mask", SPECS[kind]["optional_inputs"])
            node = {"type": kind, "inputs": {slot: "src" for slot in SPECS[kind]["inputs"]}}
            self.assertEqual(bypass_slot(node), SPECS[kind]["inputs"][0])

    def test_nonlocal_effect_graphs_explicitly_fall_back_from_tiles(self):
        for kind in ("Flare", "Glint", "Sparkles", "GodRays", "VolumeRays", "ScannedGrain"):
            d = Dispatcher()
            d.execute({"op": "create", "id": "source", "type": "Constant"})
            d.execute({"op": "create", "id": "fx", "type": kind})
            d.execute({"op": "connect", "id": "fx", "input": "image", "source": "source"})
            self.assertFalse(TileExecutor().supports_tiled(dict(d.document, view="fx"), "fx"), kind)

    def test_nodes_evaluate_through_the_graph_dispatcher(self):
        for kind in ("Flare", "Glint", "Sparkles", "GodRays", "VolumeRays", "ScannedGrain"):
            d = Dispatcher()
            d.execute({"op": "create", "id": "source", "type": "Constant",
                       "params": {"width": 32, "height": 32, "red": 0.2, "green": 0.2, "blue": 0.2}})
            d.execute({"op": "create", "id": "fx", "type": kind})
            d.execute({"op": "connect", "id": "fx", "input": "image", "source": "source"})
            if kind in ("VolumeRays", "ScannedGrain"):
                slot = "matte" if kind == "VolumeRays" else "plate"
                d.execute({"op": "connect", "id": "fx", "input": slot, "source": "source"})
            result = Evaluator().evaluate(dict(d.document, view="fx"))
            self.assertEqual(result.shape, (32, 32, 4), kind)


if __name__ == "__main__":
    unittest.main()
