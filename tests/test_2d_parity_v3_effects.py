import unittest

import numpy as np

from nodebased.core import Dispatcher, SPECS, bypass_slot
from nodebased.imaging import Evaluator, Raster
from nodebased.tiers import Region
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
        one_axis = Evaluator._glint(high, dict(p, rays=1, length=4, falloff=.2))
        two_axes = Evaluator._glint(high, dict(p, rays=2, length=4, falloff=.2))
        self.assertGreater(float(one_axis[16, 18, 0]), float(high[16, 18, 0]))
        self.assertEqual(float(one_axis[18, 16, 0]), float(high[18, 16, 0]))
        self.assertGreater(float(two_axes[18, 16, 0]), float(high[18, 16, 0]))
        self.assertEqual(float(two_axes[18, 18, 0]), float(high[18, 18, 0]))

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

    def test_flare_tracker_link_moves_with_the_selected_point(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "source", "type": "Constant",
                   "params": {"width": 40, "height": 24, "alpha": 1}})
        d.execute({"op": "create", "id": "tracker", "type": "Tracker"})
        d.execute({"op": "connect", "id": "tracker", "input": "image", "source": "source"})
        track = {"name": "track1", "enabled": 1,
                 "x": {"value": 10, "curve": {"interpolation": "linear", "keys": [
                     {"frame": 1, "value": 10}, {"frame": 2, "value": 14}]}},
                 "y": {"value": 10, "curve": {"interpolation": "linear", "keys": [
                     {"frame": 1, "value": 10}, {"frame": 2, "value": 12}]}}}
        d.execute({"op": "set_tracks", "id": "tracker", "tracks": [track]})
        d.execute({"op": "create", "id": "flare", "type": "Flare", "params": {
            "position_x": 8.0, "position_y": 8.0, "tracker_id": "tracker", "track_index": 0,
            "brightness": 1.0, "streaks": 0, "length": 0.0, "rotation": 0.0, "ghosts": 0,
            "spread": 0.0, "size": 1.0, "red": 1.0, "green": 1.0, "blue": 1.0,
            "chromatic_shift": 0.0, "mix": 1.0}})
        d.execute({"op": "connect", "id": "flare", "input": "image", "source": "source"})
        d.execute({"op": "view", "id": "flare"})
        evaluator = Evaluator()
        at_reference = evaluator.evaluate(dict(d.document, view="flare"), frame=1)
        moved = evaluator.evaluate(dict(d.document, view="flare"), frame=2)
        self.assertEqual(np.unravel_index(np.argmax(at_reference[..., 0]), (24, 40)), (8, 8))
        self.assertEqual(np.unravel_index(np.argmax(moved[..., 0]), (24, 40)), (10, 12))

    def test_scanned_grain_raster_path_matches_plate_variance_and_offsets_frames(self):
        rng = np.random.default_rng(2026)
        plate_pixels = rng.normal(0.0, [0.04, 0.06, 0.08, 0.0], (256, 256, 4)).astype(np.float32)
        image_pixels = np.ones((256, 256, 4), np.float32); image_pixels[..., 3] = 1.0
        region = Region(0, 0, 256, 256)
        image = Raster(image_pixels, region, region)
        plate = Raster(plate_pixels, region, region)
        p = dict(SPECS["ScannedGrain"]["params"], amount=1.0, response=1.0, seed=91)
        first = Evaluator._windowed_kernel("ScannedGrain", p, [image, plate], frame=1)
        second = Evaluator._windowed_kernel("ScannedGrain", p, [image, plate], frame=2)
        for channel in range(3):
            variance_ratio = float(first.pixels[..., channel].var() / plate_pixels[..., channel].var())
            self.assertLess(abs(variance_ratio - 1.0), .10)
        self.assertFalse(np.array_equal(first.pixels, second.pixels))


if __name__ == "__main__":
    unittest.main()
