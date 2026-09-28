import unittest
import copy

import numpy as np

from nodebased.core import Dispatcher, SPECS, bypass_slot, empty_document, validate
from nodebased.imaging import Evaluator
from nodebased.raster import Raster
from nodebased.tiers import Region
from nodebased import shapes
from nodebased.warps import displacement_field, grid_controls


class WarpFieldTests(unittest.TestCase):
    def test_identical_controls_make_an_identity_field(self):
        points = np.array([[0, 0], [20, 0], [0, 20], [20, 20]], dtype=float)
        x, y = np.meshgrid(np.arange(24), np.arange(24))
        field = displacement_field(points, points, x, y)
        np.testing.assert_allclose(field, 0.0, atol=1e-7)

    def test_grid_control_displacement_is_exact_and_decays(self):
        src = np.array([[0, 0], [20, 0], [0, 20], [20, 20]], dtype=float)
        dst = src.copy()
        dst[0] += (3.0, -2.0)
        x = np.array([[3.0, 40.0]])
        y = np.array([[-2.0, 40.0]])
        field = displacement_field(src, dst, x, y, radius=8)
        np.testing.assert_allclose(field[0, 0], (3.0, -2.0), atol=1e-5)
        self.assertLess(float(np.linalg.norm(field[0, 1])), 1e-4)

    def test_bezier_tangent_grid_edges_are_sampled_and_identity_is_exact(self):
        grid = np.zeros((2, 2, 6), dtype=float)
        grid[0, 0, :2] = (0, 0)
        grid[0, 1, :2] = (10, 0)
        grid[1, 0, :2] = (0, 10)
        grid[1, 1, :2] = (10, 10)
        src, dst = grid_controls(grid, grid)
        self.assertGreater(len(src), 4)
        np.testing.assert_array_equal(src, dst)

    def test_grid_controls_reject_mismatched_shapes(self):
        with self.assertRaisesRegex(ValueError, "matching MxNx2"):
            grid_controls(np.zeros((3, 3, 2)), np.zeros((4, 3, 2)))


def point(x, y):
    return {"x": float(x), "y": float(y), "in_x": 0.0, "in_y": 0.0,
            "out_x": 0.0, "out_y": 0.0}


class WarpNodeTests(unittest.TestCase):
    def setUp(self):
        self.width, self.height = 32, 32
        yy, xx = np.mgrid[:self.height, :self.width]
        pixels = np.zeros((self.height, self.width, 4), np.float32)
        pixels[..., 0] = xx / self.width
        pixels[..., 1] = yy / self.height
        pixels[..., 3] = 1.0
        self.image = Raster(pixels, Region(0, 0, self.width, self.height),
                            Region(0, 0, self.width, self.height))

    def grid_data(self, move=(0.0, 0.0)):
        src = [[[8.5, 8.5], [24.5, 8.5]], [[8.5, 24.5], [24.5, 24.5]]]
        dst = copy.deepcopy(src)
        dst[0][0] = [src[0][0][0] + move[0], src[0][0][1] + move[1]]
        return {"source": src, "destination": dst}

    def test_grid_identity_and_exact_control_motion(self):
        params = dict(SPECS["GridWarp"]["params"])
        identity = Evaluator._warp_node("GridWarp", params, [self.image], self.grid_data())
        np.testing.assert_allclose(identity.pixels, self.image.pixels, atol=1e-5)
        moved = Evaluator._warp_node("GridWarp", params, [self.image], self.grid_data((3.0, -2.0)))
        # Destination pixel (11, 6) inverse-samples source pixel (8, 8) exactly.
        np.testing.assert_allclose(moved.pixels[6, 11], self.image.pixels[8, 8], atol=1e-5)
        self.assertLess(abs(float(moved.pixels[28, 28, 0] - self.image.pixels[28, 28, 0])), 1e-3)

    def test_spline_moves_edge_and_mix_zero_is_exact_identity(self):
        edge = self.image.pixels.copy()
        edge[..., :3] = (np.arange(self.width)[None, :, None] >= 8).astype(np.float32)
        source = Raster(edge, self.image.data, self.image.display)
        data = [{"name": "edge",
                 "source": [[8.5, 2.5, 0, 0, 0, 0], [8.5, 29.5, 0, 0, 0, 0]],
                 "destination": [[12.5, 2.5, 0, 0, 0, 0], [12.5, 29.5, 0, 0, 0, 0]]}]
        params = dict(SPECS["SplineWarp"]["params"])
        params["curve_resolution"] = 16
        out = Evaluator._warp_node("SplineWarp", params, [source], data)
        self.assertAlmostEqual(float(out.pixels[16, 12, 0]), 1.0, delta=0.5)
        params["mix"] = 0.0
        identity = Evaluator._warp_node("SplineWarp", params, [source], data)
        np.testing.assert_array_equal(identity.pixels, source.pixels)

    def test_identical_spline_curves_are_pixel_identity(self):
        controls = [[4.5, 4.5, 0, 0, 3, 0], [24.5, 24.5, -3, 0, 0, 0]]
        data = [{"name": "identity", "source": controls, "destination": copy.deepcopy(controls)}]
        output = Evaluator._warp_node("SplineWarp", dict(SPECS["SplineWarp"]["params"]),
                                      [self.image], data)
        np.testing.assert_allclose(output.pixels, self.image.pixels, atol=1e-5)

    def test_spline_stmap_recreates_warped_image(self):
        data = [{"name": "edge",
                 "source": [[8.5, 2.5, 0, 0, 0, 0], [8.5, 29.5, 0, 0, 0, 0]],
                 "destination": [[12.5, 2.5, 0, 0, 0, 0], [12.5, 29.5, 0, 0, 0, 0]]}]
        params = dict(SPECS["SplineWarp"]["params"])
        params["curve_resolution"] = 8
        expected = Evaluator._warp_node("SplineWarp", params, [self.image], data)
        params["output"] = "stmap"
        uv = Evaluator._warp_node("SplineWarp", params, [self.image], data)
        stmap = dict(SPECS["STMap"]["params"])
        actual = Evaluator._uv_node("STMap", stmap, [self.image, uv, None])
        np.testing.assert_allclose(actual.pixels, expected.pixels, atol=1e-5)

    def test_warp_points_interpolate_and_document_validates(self):
        payload = {"pairs": [{"name": "p", "source": [point(1, 2), point(3, 4)],
                              "destination": [point(5, 6), point(7, 8)]}]}
        payload["pairs"][0]["destination"][0]["x"] = {
            "value": 5.0, "curve": {"interpolation": "linear", "keys": [
                {"frame": 1, "value": 5.0}, {"frame": 3, "value": 9.0}]}}
        resolved = shapes.resolve_warp_data("SplineWarp", payload, 2)
        self.assertAlmostEqual(resolved[0]["destination"][0][0], 7.0)
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "c", "type": "Constant"})
        d.execute({"op": "create", "id": "w", "type": "SplineWarp"})
        d.execute({"op": "connect", "id": "w", "input": "image", "source": "c"})
        d.execute({"op": "set_warp_data", "id": "w", "data": payload})
        validate(d.document)

    def test_nodes_register_as_full_frame_mask_mix_and_bypass(self):
        self.assertNotIn("SplineWarp", __import__("nodebased.tiles", fromlist=["SUPPORTED_TILED_KINDS"]).SUPPORTED_TILED_KINDS)
        for kind in ("SplineWarp", "GridWarp"):
            d = Dispatcher()
            d.execute({"op": "create", "id": "c", "type": "Constant", "params": {"width": 8, "height": 8}})
            d.execute({"op": "create", "id": "w", "type": kind})
            d.execute({"op": "connect", "id": "w", "input": "image", "source": "c"})
            self.assertEqual(bypass_slot(d.document["nodes"]["w"]), "image")
            d.execute({"op": "disable", "id": "w", "value": True})
            out = Evaluator().evaluate_raster(d.document, "w")
            src = Evaluator().evaluate_raster(d.document, "c")
            np.testing.assert_array_equal(out.pixels, src.pixels)


if __name__ == "__main__":
    unittest.main()
