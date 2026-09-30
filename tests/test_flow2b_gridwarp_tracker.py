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
    def test_tracker_local_motion_adds_idw_residuals_and_zero_keeps_affine(self):
        source = Raster(np.zeros((64, 64, 4), np.float32), Region(0, 0, 64, 64), Region(0, 0, 64, 64))
        base = np.array([[.5, .5], [63.5, .5], [.5, 63.5], [63.5, 63.5]])
        moved = base + np.array([[0., 0.], [0., 0.], [0., 0.], [12., -6.]])
        data = {"tracker_affine": (1., 0., 0., 1., 3., 2.), "tracker_samples": (base.tolist(), moved.tolist())}
        params = dict(SPECS["GridWarpTracker"]["params"], rows=5, columns=5)
        affine = Evaluator._gridwarp_tracker_controls(source, params, data)
        local = Evaluator._gridwarp_tracker_controls(source, dict(params, local_motion=1.), data)
        np.testing.assert_allclose(np.asarray(affine["destination"])[0, 0], [3.5, 2.5])
        self.assertGreater(float(np.linalg.norm(np.asarray(local["destination"])[-1, -1] -
                                               np.asarray(affine["destination"])[-1, -1])), 5.)
        np.testing.assert_allclose(np.asarray(local["destination"])[-1, -1], moved[-1], atol=.5)

    def test_smartvector_occlusion_and_forward_backward_error_reject_samples(self):
        source = Raster(np.zeros((16, 16, 4), np.float32), Region(0, 0, 16, 16), Region(0, 0, 16, 16))
        flow = np.zeros((16, 16, 4), np.float32); flow[..., 0] = 5.; flow[..., 3] = 1.
        occ = np.zeros_like(flow); occ[..., 3] = 1.
        back = np.zeros_like(flow); back[..., 3] = 1.
        v = Raster(source.pixels, source.data, source.display, {
            "custom.forward": Raster(flow, source.data, source.display),
            "custom.backward": Raster(back, source.data, source.display),
            "vector.occlusion": Raster(occ, source.data, source.display)})
        params = dict(SPECS["GridWarpTracker"]["params"], drive="smartvector", rows=2, columns=2,
                      forward_layer="custom.forward", backward_layer="custom.backward")
        controls = Evaluator._gridwarp_tracker_controls(source, params, None, v, 2)
        np.testing.assert_allclose(np.asarray(controls["destination"]), np.asarray(controls["source"]), atol=1e-6)

    def test_smartvector_occlusion_holds_last_valid_nonzero_motion(self):
        region = Region(0, 0, 16, 16)
        pixels = np.zeros((16, 16, 4), np.float32)
        source = Raster(pixels, region, region)
        def layer(dx, occluded):
            data = np.zeros_like(pixels); data[..., 0] = dx; data[..., 2] = float(occluded); data[..., 3] = 1
            return Raster(data, region, region)
        current = Raster(pixels, region, region, {
            "custom.forward": layer(5., True), "custom.backward": layer(-5., True)})
        old = Raster(pixels, region, region, {
            "custom.forward": layer(3., False), "custom.backward": layer(-3., False)})
        combined = Raster(pixels, region, region, {
            **current.layers, **{f"gridwarp.history.1|{name}": value for name, value in old.layers.items()}})
        params = dict(SPECS["GridWarpTracker"]["params"], drive="smartvector", rows=2, columns=2,
                      forward_layer="custom.forward", backward_layer="custom.backward")
        controls = Evaluator._gridwarp_tracker_controls(source, params, None, combined, 2)
        np.testing.assert_allclose(np.asarray(controls["destination"]), np.asarray(controls["source"]) + [3., 0.], atol=1e-6)

    def test_local_motion_tracks_one_moving_corner_for_twenty_frames(self):
        region = Region(0, 0, 64, 64)
        source = Raster(np.zeros((64, 64, 4), np.float32), region, region)
        points = np.array([[.5, .5], [63.5, .5], [.5, 63.5], [63.5, 63.5]])
        grid = np.asarray(default_grid(64, 64, 5, 5), dtype=float)
        params = dict(SPECS["GridWarpTracker"]["params"], rows=5, columns=5, local_motion=1.)
        for frame in range(1, 21):
            t = (frame - 1) / 19.
            moved = points + np.array([2. * t, -1. * t])
            moved[-1] += np.array([10. * t, -6. * t])
            matrix, offset = Evaluator._gridwarp_fit_tracker((points, moved))
            base = np.einsum("ij,...j->...i", matrix, grid) + offset
            residual = moved - (points @ matrix.T + offset)
            distance = np.linalg.norm(grid.reshape(-1, 2)[:, None, :] - points[None, :, :], axis=-1)
            weights = 1. / np.maximum(distance, 1e-3) ** 2
            expected = base + ((weights @ residual) / weights.sum(axis=1, keepdims=True)).reshape(grid.shape)
            data = {"tracker_points": (points.tolist(), moved.tolist()), "tracker_indices": list(range(4)),
                    "tracker_history": [(frame, moved.tolist())]}
            actual = np.asarray(Evaluator._gridwarp_tracker_controls(source, params, data, frame=frame)["destination"])
            self.assertLessEqual(float(np.max(np.linalg.norm(actual - expected, axis=-1))), .5)

    def test_two_occluded_tracker_points_hold_their_last_valid_motion(self):
        region = Region(0, 0, 64, 64)
        pixels = np.zeros((64, 64, 4), np.float32)
        source = Raster(pixels, region, region)
        refs = np.array([[.5, .5], [63.5, .5], [.5, 63.5], [63.5, 63.5]])
        previous = refs + [2., 1.]
        current = refs + [4., 2.]
        current[:2] += [30., 20.]
        occ = np.zeros_like(pixels)
        occ[0:2, 0:2, 0] = 1.; occ[0:2, 62:64, 0] = 1.
        occ_layer = Raster(occ, region, region)
        vectors = Raster(pixels, region, region, {"vector.occlusion": occ_layer})
        data = {"tracker_points": (refs.tolist(), current.tolist()), "tracker_indices": list(range(4)),
                "tracker_history": [(2, current.tolist()), (1, previous.tolist())]}
        params = dict(SPECS["GridWarpTracker"]["params"], rows=5, columns=5, local_motion=1.)
        controls = Evaluator._gridwarp_tracker_controls(source, params, data, vectors, 2)
        destination = np.asarray(controls["destination"])
        self.assertLessEqual(float(np.linalg.norm(destination[0, 0] - previous[0])), .5)
        self.assertLessEqual(float(np.linalg.norm(destination[0, -1] - previous[1])), .5)

    def test_smartvector_affine_layers_move_every_grid_point_and_missing_layers_hold(self):
        h = w = 96
        region = Region(0, 0, w, h)
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        pixels = np.zeros((h, w, 4), np.float32)
        pixels[..., 0] = xx / w
        pixels[..., 1] = yy / h
        pixels[..., 2] = ((xx.astype(int) // 8 + yy.astype(int) // 8) % 2).astype(np.float32)
        pixels[..., 3] = 1
        source = Raster(pixels, region, region)
        params = dict(SPECS["GridWarpTracker"]["params"], drive="smartvector", rows=5,
                      columns=5, reference_frame=1)
        grid = np.asarray(default_grid(w, h, 5, 5), np.float32)
        matrix_final = np.array([[1.10, 0.08], [-0.04, 0.92]], np.float32)
        offset_final = np.array([2.0, -1.0], np.float32)
        for frame in range(1, 21):
            t = (frame - 1) / 19.0
            matrix = np.eye(2, dtype=np.float32) + (matrix_final - np.eye(2, dtype=np.float32)) * t
            offset = offset_final * t
            field = np.zeros((h, w, 4), np.float32)
            field[..., :2] = np.einsum("ij,hwj->hwi", matrix - np.eye(2, dtype=np.float32),
                                        np.stack((xx, yy), axis=-1)) + offset
            field[..., 3] = 1
            vectors = Raster(pixels, region, region,
                             {"custom.forward": Raster(field, region, region)})
            params["forward_layer"] = "custom.forward"
            controls = Evaluator._gridwarp_tracker_controls(source, params, None, vectors, frame)
            actual = np.asarray(controls["destination"], np.float32)
            expected = np.einsum("ij,hwj->hwi", matrix, grid) + offset
            self.assertLessEqual(float(np.max(np.linalg.norm(actual - expected, axis=-1))), 0.5)
            # The result is an image warp and can be chained into the ordinary GridWarp kernel.
            warped = Evaluator._warp_node("GridWarpTracker", params,
                                          [source, None, vectors], None, frame)
            chained = Evaluator._warp_node("GridWarp", dict(SPECS["GridWarp"]["params"]),
                                           [warped], {"source": controls["source"],
                                                      "destination": controls["destination"]})
            self.assertEqual(chained.pixels.shape, source.pixels.shape)
        no_vectors = Evaluator._gridwarp_tracker_controls(source, params, None, None, 20)
        np.testing.assert_array_equal(np.asarray(no_vectors["destination"]), grid)

    def test_three_or_more_tracks_fit_affine_grid_motion(self):
        d = self.graph()
        matrix = np.array([[1.08, .06], [-.03, .94]])
        offset = np.array([1.5, -2.0])
        for tr in d.document["node_data"]["trk"]["tracks"]:
            x0 = tr["x"]["value"]
            y0 = tr["y"]["value"]
            x1, y1 = matrix @ np.array([x0, y0]) + offset
            tr["x"] = {"value": x0, "curve": {"interpolation": "linear", "keys": [
                {"frame": 1, "value": x0}, {"frame": 20, "value": x1}]}}
            tr["y"] = {"value": y0, "curve": {"interpolation": "linear", "keys": [
                {"frame": 1, "value": y0}, {"frame": 20, "value": y1}]}}
        doc = dict(d.document, view="warp")
        src = default_grid(64, 64, 5, 5)
        source = Evaluator().evaluate_raster(doc, "src", frame=20)
        for frame in range(1, 21):
            t = (frame - 1) / 19.0
            current_matrix = np.eye(2) + (matrix - np.eye(2)) * t
            current_offset = offset * t
            dst = [[(current_matrix @ np.asarray(point) + current_offset).tolist() for point in row]
                   for row in src]
            expected = Evaluator._warp_node("GridWarp", dict(SPECS["GridWarp"]["params"]),
                                            [source], {"source": src, "destination": dst}).to_display()
            result = Evaluator().evaluate(doc, frame=frame)
            np.testing.assert_allclose(result, expected, atol=1e-6, err_msg=f"frame {frame}")

    def test_smartvector_node_connects_and_grid_output_chains_to_gridwarp(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "src", "type": "Constant", "params": {
            "width": 12, "height": 10, "red": .25, "green": .5, "blue": .75, "alpha": 1}})
        d.execute({"op": "create", "id": "vectors", "type": "SmartVector", "params": {
            "reference_frame": 1, "frame_start": 1, "frame_end": 3, "vector_detail": 1}})
        d.execute({"op": "connect", "id": "vectors", "input": "image", "source": "src"})
        d.execute({"op": "create", "id": "warp", "type": "GridWarpTracker", "params": {
            "drive": "smartvector", "reference_frame": 1}})
        d.execute({"op": "connect", "id": "warp", "input": "image", "source": "src"})
        d.execute({"op": "connect", "id": "warp", "input": "vectors", "source": "vectors"})
        d.execute({"op": "create", "id": "grid", "type": "GridWarp"})
        d.execute({"op": "connect", "id": "grid", "input": "image", "source": "warp"})
        result = Evaluator().evaluate(dict(d.document, view="grid"), frame=2)
        self.assertEqual(result.shape, (10, 12, 4))
        np.testing.assert_allclose(result[..., :3], np.broadcast_to([.25, .5, .75], result[..., :3].shape), atol=1e-6)

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
