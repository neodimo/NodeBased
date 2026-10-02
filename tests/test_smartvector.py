import unittest
import tempfile
from pathlib import Path

import numpy as np

from nodebased.flow_nodes import (accumulated_vectors, homography_from_corners,
                                  inpaint, warp_by_homography, warp_by_flow)
from nodebased.opticalflow import _sample
from nodebased.core import Dispatcher, empty_document
from nodebased.imaging import Evaluator
from nodebased.cachetier import DiskCache
from nodebased import gpu3d, tiles


class SmartVectorMathTests(unittest.TestCase):
    @staticmethod
    def texture(size=64):
        a = np.random.default_rng(91).random((size, size), dtype=np.float32)
        return (a + np.roll(a, 1, 0) + np.roll(a, 1, 1)) / 3

    def test_accumulated_flow_tracks_ten_frame_translation(self):
        base = self.texture()
        yy, xx = np.mgrid[:64, :64].astype(np.float32)
        frames = [_sample(base, xx - i, yy) for i in range(11)]
        forward, backward = accumulated_vectors(frames, 0, vector_detail=3,
                                                smoothness=0.4, reanchor_interval=5)
        measured = forward[10][20:-20, 20:-20, 0]
        self.assertLess(float(np.mean(np.abs(measured - 10))), 0.5)
        self.assertLess(float(np.mean(np.abs(backward[10][20:-20, 20:-20, 0] + 10))), 0.5)

    @unittest.skipUnless(__import__("nodebased.gpu3d", fromlist=["available"]).available(),
                         "wgpu adapter unavailable")
    def test_gpu_accumulation_matches_cpu_and_meets_drift_gates(self):
        base = self.texture(96)
        yy, xx = np.mgrid[:96, :96].astype(np.float32)
        frames = [_sample(base, xx - 0.5*i, yy) for i in range(21)]
        cpu_forward, _ = accumulated_vectors(frames, 0, vector_detail=4, smoothness=0.5,
                                             reanchor_interval=5, backend="cpu")
        gpu_forward, _ = accumulated_vectors(frames, 0, vector_detail=4, smoothness=0.5,
                                             reanchor_interval=5, backend="gpu")
        backend_error = float(np.linalg.norm(gpu_forward[10][20:-20, 20:-20]
                                             - cpu_forward[10][20:-20, 20:-20], axis=2).mean())
        self.assertLess(backend_error, 0.15,
                        f"mean CPU/GPU endpoint disagreement: {backend_error:.4f} px")
        drift = float(np.mean(np.abs(gpu_forward[10][20:-20, 20:-20, 0] - 5.0)))
        self.assertLess(drift, 0.5, f"GPU frame-10 drift: {drift:.4f} px")
        corners = np.asarray([[20, 20], [70, 20], [70, 70], [20, 70]], np.float32)
        errors = []
        for frame in range(1, 21):
            field = gpu_forward[frame]
            displacement = np.stack([_sample(field[..., axis], corners[:, 0], corners[:, 1])
                                     for axis in range(2)], axis=1)
            matrix = homography_from_corners(corners, corners + displacement)
            projected = (matrix @ np.c_[corners, np.ones(4)].T).T
            projected = projected[:, :2] / projected[:, 2:]
            errors.append(np.linalg.norm(projected - (corners + (0.5*frame, 0)), axis=1))
        corner_error = float(np.max(errors))
        self.assertLess(corner_error, 0.5, f"GPU 20-frame corner error: {corner_error:.4f} px")

    def test_flow_warp_moves_reference_mark(self):
        image = np.zeros((32, 32, 4), np.float32); image[10, 8] = (1, 0, 0, 1)
        flow = np.zeros((32, 32, 2), np.float32); flow[..., 0] = 4
        warped = warp_by_flow(image, flow)
        self.assertGreater(warped[10, 12, 0], 0.99)

    def test_corner_pin_homography_tracks_all_four_corners(self):
        src = np.asarray([[2, 2], [29, 2], [29, 29], [2, 29]], np.float32)
        dst = src + np.asarray([3, -2], np.float32)
        h = homography_from_corners(src, dst)
        projected = (h @ np.c_[src, np.ones(4)].T).T
        projected = projected[:, :2] / projected[:, 2:]
        self.assertLess(float(np.max(np.abs(projected - dst))), 1e-4)
        image = np.zeros((32, 32, 4), np.float32); image[12, 12] = (0, 1, 0, 1)
        moved = warp_by_homography(image, h)
        self.assertGreater(moved[10, 15, 1], 0.99)

    def test_corner_tracks_remain_within_half_pixel_for_twenty_frames(self):
        base = self.texture(96)
        yy, xx = np.mgrid[:96, :96].astype(np.float32)
        frames = [_sample(base, xx-i, yy) for i in range(21)]
        forward, _ = accumulated_vectors(frames, 0, vector_detail=4, smoothness=0.5,
                                         reanchor_interval=5)
        corners = np.asarray([[20, 20], [70, 20], [70, 70], [20, 70]], np.float32)
        errors = []
        for frame in range(1, 21):
            field = forward[frame]
            displacement = np.stack([_sample(field[..., axis], corners[:, 0], corners[:, 1])
                                     for axis in range(2)], axis=1)
            matrix = homography_from_corners(corners, corners + displacement)
            projected = (matrix @ np.c_[corners, np.ones(4)].T).T
            projected = projected[:, :2] / projected[:, 2:]
            errors.append(np.linalg.norm(projected - (corners + (frame, 0)), axis=1))
        measured = float(np.max(errors))
        self.assertLess(measured, 0.5, f"maximum 20-frame corner error: {measured:.4f} px")

    def test_inpaint_uses_temporal_observations_and_flat_diffusion(self):
        image = np.zeros((24, 24, 4), np.float32); image[...] = (0.2, 0.4, 0.6, 1)
        matte = np.zeros((24, 24), np.float32); matte[8:16, 8:16] = 1
        neighbour = image.copy()
        result = inpaint(image, matte, [neighbour], "diffusion")
        self.assertLess(float(np.max(np.abs(result - image))), 1e-5)
        spatial = inpaint(image, matte, [], "diffusion")
        self.assertLess(float(np.max(np.abs(spatial - image))), 1e-5)
        ground_truth = np.zeros((24, 24, 4), np.float32); ground_truth[...] = (0.1, 0.3, 0.8, 1)
        occluded = ground_truth.copy(); occluded[8:16, 8:16] = (1, 0, 0, 1)
        recovered = inpaint(occluded, matte, [ground_truth], "diffusion")
        self.assertLess(float(np.mean(np.abs(recovered[8:16, 8:16] - ground_truth[8:16, 8:16]))), 0.01)

    def test_patch_inpaint_recovers_repeating_texture_and_preserves_flat_field(self):
        tile = np.random.default_rng(7).random((8, 8), dtype=np.float32)
        texture = np.tile(tile, (9, 9))[..., None]
        plate = np.repeat(texture, 4, axis=2); plate[..., 3] = 1
        matte = np.zeros((72,72), np.float32); matte[27:45,27:45] = 1
        damaged = plate.copy(); damaged[matte > 0] = 0.5
        restored = inpaint(damaged, matte, method="patch")
        error = float(np.mean(np.abs(restored[matte > 0] - plate[matte > 0])))
        self.assertLess(error, 0.05, f"patch fill mean error/range: {error:.4f}")
        flat = np.empty((24,24,4), np.float32); flat[...] = (.2,.4,.6,1)
        hole = np.zeros((24,24), np.float32); hole[8:16,8:16] = 1
        np.testing.assert_allclose(inpaint(flat, hole, method="patch"), flat, atol=1e-6)

    @unittest.skipUnless(__import__("nodebased.gpu3d", fromlist=["available"]).available(),
                         "wgpu adapter unavailable")
    def test_inpaint_temporal_flow_gpu_tracks_cpu_reference(self):
        base = self.texture(96)
        image = np.repeat(base[...,None], 4, axis=2); image[...,3] = 1
        yy, xx = np.mgrid[:96,:96].astype(np.float32)
        from nodebased.flow_nodes import _sample
        neighbour = image.copy(); neighbour[...,:3] = _sample(image[...,:3], xx-1, yy)
        matte = np.zeros((96,96), np.float32); matte[32:60,32:60] = 1
        from nodebased.opticalflow import flow_pair
        cpu_flow = flow_pair(image, neighbour, backend="cpu")[0]
        gpu_flow = flow_pair(image, neighbour, backend="gpu")[0]
        endpoint_error = float(np.mean(np.linalg.norm(cpu_flow[20:-20,20:-20] -
                                                     gpu_flow[20:-20,20:-20], axis=2)))
        self.assertLess(endpoint_error, 0.15,
                        f"CPU/GPU flow endpoint disagreement: {endpoint_error:.4f} px")
        cpu = inpaint(image, matte, [neighbour], backend="cpu")
        gpu = inpaint(image, matte, [neighbour], backend="gpu")
        error = float(np.mean(np.abs(cpu[matte > 0] - gpu[matte > 0])))
        self.assertLess(error, 0.15, f"CPU/GPU temporal-fill pixel disagreement: {error:.4f}")

    def test_reanchoring_bounds_fifty_frame_drift(self):
        base = self.texture(128)
        yy, xx = np.mgrid[:128, :128].astype(np.float32)
        frames = [_sample(base, xx - 0.5*i, yy) for i in range(51)]
        forward, _ = accumulated_vectors(frames, 0, vector_detail=4, smoothness=0.5,
                                         reanchor_interval=5)
        roi = forward[50][48:80, 70:100, 0]
        error = float(np.mean(np.abs(roi - 25.0)))
        self.assertLess(error, 0.5, f"measured 50-frame mean drift: {error:.4f} px")

    def test_registered_nodes_render_vector_layers_warp_and_inpaint(self):
        dispatcher = Dispatcher(empty_document())
        commands = [{"op": "create", "id": "plate", "type": "Constant"},
                    {"op": "set", "id": "plate", "param": "width", "value": 24},
                    {"op": "set", "id": "plate", "param": "height", "value": 24},
                    {"op": "set", "id": "plate", "param": "red", "value": 0.3},
                    {"op": "create", "id": "vectors", "type": "SmartVector"},
                    {"op": "set", "id": "vectors", "param": "frame_start", "value": 1},
                    {"op": "set", "id": "vectors", "param": "frame_end", "value": 2},
                    # The GPU warp path when a GPU is present; CI runners have no wgpu, and an explicit
                    # "gpu" backend is an error there rather than a fallback (10/2, first v0.33.0 tag).
                    {"op": "set", "id": "vectors", "param": "flow_backend",
                     "value": "gpu" if gpu3d.available() else "cpu"},
                    {"op": "connect", "id": "vectors", "input": "image", "source": "plate"}]
        for ident, node_type in (("distort", "VectorDistort"), ("corner", "VectorCornerPin")):
            commands += [{"op": "create", "id": ident, "type": node_type},
                         {"op": "connect", "id": ident, "input": "image", "source": "plate"},
                         {"op": "connect", "id": ident, "input": "vectors", "source": "vectors"}]
        commands += [{"op": "create", "id": "fill", "type": "Inpaint"},
                     {"op": "connect", "id": "fill", "input": "image", "source": "plate"},
                     {"op": "connect", "id": "fill", "input": "matte", "source": "plate"}]
        commands += [{"op": "set", "id": "distort", "param": "fade_frames", "value": 0}]
        dispatcher.execute({"op": "batch", "commands": commands})
        document = dispatcher.document
        for kind in ("SmartVector", "VectorDistort", "VectorCornerPin"):
            self.assertNotIn(kind, tiles.SUPPORTED_TILED_KINDS)
        # Inpaint joined the tile path in 2D parity plan 13 step E2 (`_temporal_tile`, tiles equal
        # to full-frame), so it is intentionally left out of the exclusion check above.
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        disk = DiskCache(root=Path(temp.name))
        evaluator = Evaluator(cache_bytes=1, disk=disk)
        vector = evaluator.evaluate_raster(document, "vectors", frame=2)
        self.assertIn("smartvector.forward", vector.layers)
        self.assertIn("smartvector.backward", vector.layers)
        self.assertLess(float(np.max(np.abs(vector.layers["smartvector.forward"].pixels[..., :2]))), 1e-5)
        restarted = Evaluator(disk=DiskCache(root=Path(temp.name)))
        cached_vector = restarted.evaluate_raster(document, "vectors", frame=2)
        self.assertIn("smartvector.forward", cached_vector.layers)
        self.assertGreater(restarted.disk_hits, 0)
        for ident in ("distort", "corner", "fill"):
            result = evaluator.evaluate_raster(document, ident, frame=2)
            np.testing.assert_allclose(result.pixels, vector.pixels, atol=1e-5, err_msg=ident)
        for ident in ("vectors", "distort", "corner", "fill"):
            bypass_dispatcher = Dispatcher(document)
            bypass_dispatcher.execute({"op": "disable", "id": ident, "value": True})
            bypass = Evaluator().evaluate_raster(bypass_dispatcher.document, ident, frame=2)
            plate = Evaluator().evaluate_raster(document, "plate", frame=2)
            np.testing.assert_array_equal(bypass.pixels, plate.pixels, err_msg=f"{ident} bypass")


if __name__ == "__main__": unittest.main()
