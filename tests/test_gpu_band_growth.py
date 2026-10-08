"""The smoke path tracer's row bands grow with the measured dispatch time (step Q2) and the picture does not change."""
import sys
import unittest
from unittest import mock

import numpy as np

from nodebased import fluid_gpu_solver, gpupathtrace, pathtrace
from tools.benchmark_volume_majorant import tenth_plume


@unittest.skipUnless(fluid_gpu_solver.available(), "no compute adapter")
class BandGrowthTests(unittest.TestCase):
    WIDTH, HEIGHT, SAMPLES = 64, 48, 4

    def render(self, target):
        if sys.platform == "win32" and "cpu" in str(getattr(fluid_gpu_solver._ctx(), "kind", "")).lower():
            self.skipTest("Windows software adapter: the smoke path tracer stays on the CPU there")
        scene, camera, volume, _ = tenth_plume(16)
        stats = {}
        # two rows per band, so a sample is 24 dispatches until the bands grow
        with mock.patch.object(gpupathtrace, "soft_supported", return_value=True), \
             mock.patch.object(gpupathtrace, "SOFT_SLOWDOWN", 1), \
             mock.patch.object(gpupathtrace, "GPU_PATHS_PER_SUBMISSION", self.WIDTH * 2), \
             mock.patch.object(gpupathtrace, "BAND_TARGET_SECONDS", target):
            image = pathtrace.render(scene, camera, self.WIDTH, self.HEIGHT, ambient=.7, volume=volume, backend="gpu", stats=stats,
                                     settings=pathtrace.PathSettings(sampling="fixed", samples=self.SAMPLES, max_bounces=3, seed=4))
        return image, stats["readbacks"]["waits"]

    def test_bands_grow_and_the_image_is_the_same(self):
        fixed, fixed_waits = self.render(0.0)
        grown, grown_waits = self.render(10.0)          # a target no band reaches: every band grows as far as it may
        self.assertEqual(fixed_waits, self.SAMPLES * (self.HEIGHT // 2))
        self.assertLess(grown_waits, fixed_waits // 2)
        self.assertGreater(float(fixed[..., :3].mean()), 0)
        np.testing.assert_array_equal(fixed, grown)     # sample sums do not depend on how the rows were cut into bands


if __name__ == "__main__":
    unittest.main()
