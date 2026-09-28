import unittest

import numpy as np

from nodebased.opticalflow import _sample, flow_pair


class OpticalFlowCoreTests(unittest.TestCase):
    @staticmethod
    def texture(seed, size=128):
        image = np.random.default_rng(seed).random((size, size), dtype=np.float32)
        for _ in range(4):
            image = (image + np.roll(image, 1, 0) + np.roll(image, -1, 0)
                     + np.roll(image, 1, 1) + np.roll(image, -1, 1)) / 5
        return image

    def test_translated_texture_endpoint_error(self):
        h, w = 128, 144
        source = self.texture(1, h)
        source = np.pad(source, ((0, 0), (0, w-h)), mode="edge")
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        target = _sample(source, xx-3.5, yy-1.25)
        forward, _, _ = flow_pair(source, target, vector_detail=3, smoothness=1.0, iterations=8)
        error = np.linalg.norm(forward[16:-16, 16:-16] - (3.5, 1.25), axis=2)
        self.assertLess(float(error.mean()), 0.1)

    def test_rotation_matches_analytic_field(self):
        size = 128
        source = self.texture(2, size)
        yy, xx = np.mgrid[:size, :size].astype(np.float32)
        cx = cy = (size - 1) / 2
        angle = 0.05
        c, s = np.cos(angle), np.sin(angle)
        dx, dy = xx-cx, yy-cy
        target = _sample(source, c*dx+s*dy+cx, -s*dx+c*dy+cy)
        forward, _, _ = flow_pair(source, target, vector_detail=4, smoothness=1.0, iterations=7)
        truth = np.stack((c*dx-s*dy-dx, s*dx+c*dy-dy), axis=2)
        error = np.linalg.norm(forward[24:-24, 24:-24] - truth[24:-24, 24:-24], axis=2)
        self.assertLess(float(error.mean()), 0.1)

    def test_identical_frames_have_zero_flow_and_no_occlusion(self):
        image = self.texture(3, 64)
        forward, backward, occluded = flow_pair(image, image, vector_detail=3)
        self.assertLess(float(np.max(np.abs(forward))), 1e-5)
        self.assertLess(float(np.max(np.abs(backward))), 1e-5)
        self.assertFalse(np.any(occluded))

    def test_forward_backward_consistency_marks_planted_disocclusion(self):
        size = 80
        rng = np.random.default_rng(8)
        source = rng.random((size, size), dtype=np.float32)
        yy, xx = np.mgrid[:size, :size].astype(np.float32)
        target = _sample(source, xx-5.0, yy)
        target[:, :5] = rng.random((size, 5), dtype=np.float32)
        _, _, occluded = flow_pair(source, target, vector_detail=3, smoothness=1.0, iterations=8)
        self.assertGreater(float(occluded[:, :5].mean()), 0.9)


if __name__ == "__main__":
    unittest.main()
