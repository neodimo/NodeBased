import unittest

import numpy as np

from nodebased import gpu3d
from nodebased.opticalflow import _sample
from nodebased.opticalflow_gpu import (flow_pair_gpu, warp_by_flow_gpu,
                                       warp_by_homography_gpu)


@unittest.skipUnless(gpu3d.available(), "wgpu adapter unavailable")
class OpticalFlowGpuTests(unittest.TestCase):
    def test_gpu_warps_match_cpu_reference(self):
        from nodebased.flow_nodes import warp_by_flow, warp_by_homography
        image = np.random.default_rng(7).random((32, 40, 4), dtype=np.float32)
        yy, xx = np.mgrid[:32, :40].astype(np.float32)
        flow = np.stack((np.full_like(xx, 2.25), np.full_like(yy, -1.5)), axis=2)
        np.testing.assert_allclose(warp_by_flow_gpu(image, flow), warp_by_flow(image, flow), atol=2e-6)
        matrix = np.asarray([[1.0, 0.02, 2.0], [-0.01, 1.0, 1.0], [0.0002, -0.0001, 1.0]])
        np.testing.assert_allclose(warp_by_homography_gpu(image, matrix),
                                   warp_by_homography(image, matrix), atol=1e-5)

    def test_wgpu_translation_and_rotation_accuracy(self):
        size = 128
        rng = np.random.default_rng(1)
        source = rng.random((size, size), dtype=np.float32)
        for _ in range(4):
            source = (source + np.roll(source, 1, 0) + np.roll(source, -1, 0)
                      + np.roll(source, 1, 1) + np.roll(source, -1, 1)) / 5
        yy, xx = np.mgrid[:size, :size].astype(np.float32)
        translated = _sample(source, xx-3.5, yy-1.25)
        forward, _, _ = flow_pair_gpu(source, translated, vector_detail=3, smoothness=1, iterations=8)
        error = np.linalg.norm(forward[16:-16, 16:-16] - (3.5, 1.25), axis=2)
        self.assertLess(float(error.mean()), 0.1)

        cx = cy = (size-1)/2
        angle = 0.05
        c, s = np.cos(angle), np.sin(angle)
        dx, dy = xx-cx, yy-cy
        rotated = _sample(source, c*dx+s*dy+cx, -s*dx+c*dy+cy)
        forward, _, _ = flow_pair_gpu(source, rotated, vector_detail=4, smoothness=1, iterations=8)
        truth = np.stack((c*dx-s*dy-dx, s*dx+c*dy-dy), axis=2)
        error = np.linalg.norm(forward[24:-24, 24:-24] - truth[24:-24, 24:-24], axis=2)
        self.assertLess(float(error.mean()), 0.1)


if __name__ == "__main__":
    unittest.main()
