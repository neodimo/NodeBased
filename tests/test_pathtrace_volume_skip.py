"""Conservative coarse volume majorants and 64-sample GPU image agreement."""
import unittest
from unittest import mock

import numpy as np

from nodebased import fluid_gpu_solver, gpupathtrace, pathtrace, scene3d, volumerender


def plume(n=256):
    x, y, z = np.ogrid[:n, :n, :n]
    density = np.exp(-(((x - n / 2) / (n * 14 / 256)) ** 2 +
                       ((y - n * 100 / 256) / (n * 75 / 256)) ** 2 +
                       ((z - n / 2) / (n * 14 / 256)) ** 2)).astype(np.float32)
    density[density < .1] = 0
    volume = scene3d.Volume(density, .02, (-n * .01,) * 3)
    scene = scene3d.Scene(volumes=(volume,))
    camera = scene3d.Camera(scene3d.Transform3D(position=scene3d.Vec3(0, 0, 8)),
                            scene3d.Vec3(0, 0, 0), 40)
    settings = volumerender.VolumeSettings(absorption=.6, scattering=.8, density_scale=30)
    return scene, camera, settings


class MajorantTests(unittest.TestCase):
    def test_each_cell_bounds_its_trilinear_halo(self):
        scene, _, _ = plume(64)
        density = scene.volumes[0].density
        bounds = gpupathtrace._coarse_volume_majorants(density, 42.0)
        scale = 42.0 / float(density.max())
        for x, y, z in np.ndindex(bounds.shape):
            slices = tuple(slice(max(0, c * 16 - 1), min(64, (c + 1) * 16 + 1))
                           for c in (x, y, z))
            self.assertGreaterEqual(float(bounds[x, y, z]) + 1e-6,
                                    float(density[slices].max()) * scale)


@unittest.skipUnless(fluid_gpu_solver.available(), "no compute adapter")
class ImageTests(unittest.TestCase):
    def test_256_cubed_plume_matches_64_sample_noise(self):
        scene, camera, volume = plume()
        def render(enabled, seed):
            with mock.patch.object(gpupathtrace, "ENABLE_VOLUME_SKIP", enabled), \
                 mock.patch.object(gpupathtrace, "soft_supported", return_value=True):
                return pathtrace.render(
                    scene, camera, 32, 32, ambient=.7, volume=volume, backend="gpu",
                    settings=pathtrace.PathSettings(samples=64, max_bounces=4, seed=seed))[..., :3]
        baseline = render(False, 4)
        repeat = render(False, 5)
        skipped = render(True, 4)
        noise = float(np.abs(baseline - repeat).mean())
        change = float(np.abs(baseline - skipped).mean())
        self.assertGreater(noise, 0)
        self.assertLess(change, noise * 1.25)
        self.assertLess(abs(float(skipped.mean() / baseline.mean()) - 1), .05)
