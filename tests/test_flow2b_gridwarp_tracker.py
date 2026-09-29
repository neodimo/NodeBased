import unittest

import numpy as np

from nodebased.core import Dispatcher, SPECS, bypass_slot
from nodebased.imaging import Evaluator
from nodebased.imaging import Raster
from nodebased.tiers import Region
from nodebased.tileexec import TileExecutor
from nodebased.warps import default_grid


def track(name, x, y, moved):
    def curve(a, b):
        return {"value": float(a), "curve": {"interpolation": "linear", "keys": [
            {"frame": 1, "value": float(a)}, {"frame": 20, "value": float(b)}]}}
    return {"name": name, "enabled": 1, "x": curve(x, moved[0]), "y": curve(y, moved[1])}


def orbit_track(name, x, y):
    cx = cy = 32.0
    xkeys, ykeys = [], []
    for frame in range(1, 21):
        angle = (frame - 1) * np.pi / 40
        c, s = np.cos(angle), np.sin(angle)
        shift = (4.0 * (frame - 1) / 19.0, -2.0 * (frame - 1) / 19.0)
        px, py = x-cx, y-cy
        xkeys.append({"frame": frame, "value": float(cx+c*px-s*py+shift[0])})
        ykeys.append({"frame": frame, "value": float(cy+s*px+c*py+shift[1])})
    def curve(keys):
        return {"value": keys[0]["value"], "curve": {"interpolation": "linear", "keys": keys}}
    return {"name": name, "enabled": 1, "x": curve(xkeys), "y": curve(ykeys)}


class GridWarpTrackerTests(unittest.TestCase):
    def graph(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "src", "type": "Constant", "params": {
            "width": 64, "height": 64, "red": 0.2, "green": 0.2, "blue": 0.2, "alpha": 1}})
        d.execute({"op": "create", "id": "trk", "type": "Tracker"})
        d.execute({"op": "connect", "id": "trk", "input": "image", "source": "src"})
        d.execute({"op": "set_tracks", "id": "trk", "tracks": [
            orbit_track("a", 16, 16), orbit_track("b", 48, 16),
            orbit_track("c", 16, 48), orbit_track("d", 48, 48)]})
        d.execute({"op": "create", "id": "warp", "type": "GridWarpTracker", "params": {
            "tracker_id": "trk", "track_indices": "0,1,2,3", "reference_frame": 1}})
        d.execute({"op": "connect", "id": "warp", "input": "image", "source": "src"})
        return d

    def test_grid_follows_known_translation_and_rotation_at_frame_20(self):
        # A known translating, rotating plate stays in lockstep with a directly supplied grid.
        d = self.graph()
        doc = dict(d.document, view="warp")
        self.assertFalse(TileExecutor().supports_tiled(doc, "warp"))
        self.assertEqual(bypass_slot({"type": "GridWarpTracker", "inputs": {"image": "src"}}), "image")
        # Rendering verifies an actual image warp; the known pose keeps the grid's center fixed
        # apart from its explicit translation, within the resampler's half-pixel tolerance.
        coords = default_grid(64, 64, 5, 5)
        source = Evaluator().evaluate_raster(doc, "src", frame=20)
        self.assertEqual(source.pixels.shape, (64, 64, 4))
        for frame in range(1, 21):
            angle = (frame - 1) * np.pi / 40
            c, s = np.cos(angle), np.sin(angle)
            tx, ty = 4.0 * (frame - 1) / 19.0, -2.0 * (frame - 1) / 19.0
            ox, oy = 32 + tx - c*32 + s*32, 32 + ty - s*32 - c*32
            moved = [[[(c*point[0]-s*point[1]+ox), (s*point[0]+c*point[1]+oy)] for point in row] for row in coords]
            expected = Evaluator._warp_node("GridWarp", dict(SPECS["GridWarp"]["params"]),
                                            [source], {"source": coords, "destination": moved}).to_display()
            output = Evaluator().evaluate(doc, frame=frame)
            np.testing.assert_allclose(output, expected, atol=1e-6, err_msg=f"frame {frame}")

    def test_untracked_range_holds_last_grid_pose(self):
        d = self.graph()
        for tr in d.document["node_data"]["trk"]["tracks"]:
            tr["enabled"] = {"value": 1, "curve": {"interpolation": "constant", "keys": [
                {"frame": 1, "value": 1}, {"frame": 11, "value": 0}]}}
        # The absent range holds the last usable tracked pose (frame 10).
        held = Evaluator().evaluate(dict(d.document, view="warp"), frame=15)
        previous = Evaluator().evaluate(dict(d.document, view="warp"), frame=10)
        np.testing.assert_allclose(held, previous, atol=1e-6)

    def test_scanned_grain_controls_presets_alpha_gate_and_irregularity(self):
        rng = np.random.default_rng(44)
        plate = rng.normal(0, .1, (64, 64, 4)).astype(np.float32)
        image = np.zeros_like(plate)
        image[..., 3] = 0
        base = dict(SPECS["ScannedGrain"]["params"], amount=1, response=1)
        gated = Evaluator._scanned_grain(image, plate, dict(base, apply_through_alpha=1), frame=3)
        np.testing.assert_array_equal(gated, image)
        image[..., :3] = .5
        image[..., 3] = 1
        neutral = Evaluator._scanned_grain(image, plate, base, frame=3)
        eight_mm = Evaluator._scanned_grain(image, plate, dict(base, preset="8mm"), frame=3)
        self.assertGreater(float(eight_mm[..., 0].var()), float(neutral[..., 0].var()))
        irregular = Evaluator._scanned_grain(image, plate, dict(base, irregularity=1), frame=3)
        self.assertFalse(np.array_equal(neutral, irregular))

    def test_levelset_distance_matte_dilate_shrink_and_gradient(self):
        pixels = np.zeros((7, 7, 4), np.float32)
        pixels[2:5, 2:5, 3] = 1
        region = Region(0, 0, 7, 7)
        source = Raster(pixels, region, region)
        params = dict(SPECS["LevelSet"]["params"])
        sdf = Evaluator._levelset(source, params)
        self.assertAlmostEqual(float(sdf.pixels[3, 3, 3]), -1.5, places=5)
        self.assertAlmostEqual(float(sdf.pixels[1, 3, 3]), .5, places=5)
        motion = sdf.layers["motion"].pixels
        self.assertLess(float(motion[2, 3, 1]), -.9)
        matte = Evaluator._levelset(source, dict(params, create_matte=1, matt_limit=1.0))
        self.assertEqual(float(matte.pixels[1, 3, 3]), 1.0)
        shrunk = Evaluator._levelset(source, dict(params, create_matte=1, matt_limit=-1.0))
        self.assertEqual(float(shrunk.pixels[2, 3, 3]), 0.0)
        depth_pixels = np.zeros_like(pixels)
        depth_pixels[2:5, 2:5, 2] = 1
        depth_layer = Raster(depth_pixels, region, region)
        multichannel = Raster(pixels.copy(), region, region, {"depth": depth_layer})
        depth_result = Evaluator._levelset(multichannel, dict(params, channel="depth.Z", output="depth.Z"))
        self.assertAlmostEqual(float(depth_result.layers["depth"].pixels[3, 3, 2]), -1.5, places=5)
