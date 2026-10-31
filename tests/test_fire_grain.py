"""Lane 6, fire grain step A: the GPU path tracer integrates the fire's emission along each flight instead of summing it at
random tentative collisions.

The old estimator (still the CPU reference, and the GPU's with several volumes) adds `emission / majorant` at every
tentative collision of delta tracking, so a pixel's glow is a handful of large random jumps: the grain of a flame was about as
large as the flame. The new one marches the same regions with stratified jittered samples and weights each by the smoke's
transmittance, which has the same expected value. These tests hold both halves: the mean agrees with the CPU reference, and
the grain falls far below it. Run on the other adapters with `scratch/nb-lanes/run/force-adapter.py integrated|cpu`."""
import os
import unittest
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from unittest import mock

from nodebased import gpupathtrace, pathtrace as pt, ptvolume, scene3d as s, volumerender as vr
from tests.test_3d_pathtrace_gpu_soft import CAMERA, READY
from tests.test_sparse_gpu_pathtrace import blob

FIRE = vr.VolumeSettings(absorption=0.2, scattering=0.9, step_size=0.05, shadow_steps=12, fire_intensity=3.0,
                         temperature_scale=900.0, fire_threshold=500.0, density_scale=6.0)
SIZE = (16, 16)


def collisions():
    """The CPU reference with the old estimator: the glow summed at delta tracking's tentative collisions."""
    return mock.patch.object(ptvolume, "ENABLE_EXPECTED_EMISSION", False)


def trace(scene, backend, samples, seed, settings=FIRE):
    return pt.render(scene, CAMERA, *SIZE, ambient=0.0, volume=settings, backend=backend,
                     settings=pt.PathSettings(samples=samples, seed=seed))


# the fire seen directly: with no gain on the light it casts into the smoke, the picture is the camera rays' own glow
SEEN = replace(FIRE, fire_light=0.0)
SEEN_FLAME = replace(FIRE, fire_intensity=0.0, flame_intensity=2.0, flame_temperature=1800.0, fire_light=0.0)


def grain(scene, backend, samples, settings=FIRE):
    """rms difference of two seeds / sqrt(2): a pure sampling-noise measure, as in examples/fire_shot/render_evidence.py."""
    a, b = (trace(scene, backend, samples, seed, settings)[..., :3] for seed in (1, 2))
    return float(np.sqrt(((a - b) ** 2).mean() / 2)), float((a.mean() + b.mean()) / 2)


def flamed(volume):
    """The blob with a flame channel: burn rate where it is hot, so the flame term of the estimator runs too."""
    flame = np.asarray(volume.temperature, np.float32) * 0.8
    return replace(volume, flame=flame)


FLAME = replace(FIRE, fire_intensity=0.0, flame_intensity=2.0, flame_temperature=1800.0)


@unittest.skipUnless(READY, "no wgpu adapter cleared for smoke in the path tracer")
class FireEmissionEstimator(unittest.TestCase):
    def test_the_mean_glow_matches_the_cpu_delta_tracking_reference(self):
        dense, _ = blob()
        scene = s.Scene(volumes=(dense,))
        with collisions():
            cpu = trace(scene, "cpu", 384, 7)
        gpu = trace(scene, "gpu", 512, 3)
        self.assertGreater(float(cpu[..., :3].max()), 0.05, "the fire must show")
        self.assertAlmostEqual(float(gpu[..., :3].mean()) / float(cpu[..., :3].mean()), 1.0, delta=0.03)

    def test_the_flame_channel_agrees_with_the_cpu_reference_too(self):
        dense, _ = blob()
        scene = s.Scene(volumes=(flamed(dense),))
        with collisions():
            cpu = trace(scene, "cpu", 384, 7, FLAME)
        gpu = trace(scene, "gpu", 512, 3, FLAME)
        self.assertGreater(float(cpu[..., :3].max()), 0.05, "the flame must show")
        self.assertAlmostEqual(float(gpu[..., :3].mean()) / float(cpu[..., :3].mean()), 1.0, delta=0.03)

    def test_the_glow_grain_is_a_fraction_of_the_delta_tracking_grain_at_the_same_samples(self):
        dense, _ = blob()
        scene = s.Scene(volumes=(dense,))
        gpu_grain, gpu_mean = grain(scene, "gpu", 64, SEEN)
        with collisions():
            cpu_grain, cpu_mean = grain(scene, "cpu", 64, SEEN)
        self.assertAlmostEqual(gpu_mean / cpu_mean, 1.0, delta=0.15)
        self.assertLess(gpu_grain, 0.5 * cpu_grain)
        self.assertLess(gpu_grain, 0.45 * gpu_mean, "a flame should no longer be as noisy as it is bright")

    def test_the_grain_of_the_flame_channel_falls_the_same_way(self):
        dense, _ = blob()
        scene = s.Scene(volumes=(flamed(dense),))
        gpu_grain, gpu_mean = grain(scene, "gpu", 64, SEEN_FLAME)
        with collisions():
            cpu_grain, _ = grain(scene, "cpu", 64, SEEN_FLAME)
        self.assertLess(gpu_grain, 0.5 * cpu_grain)
        self.assertLess(gpu_grain, 0.45 * gpu_mean)

    def test_the_cpu_twin_matches_the_card_pixel_for_pixel_on_the_same_seed(self):
        dense, _ = blob()
        scene = s.Scene(volumes=(dense,))
        cpu = trace(scene, "cpu", 32, 5)
        gpu = trace(scene, "gpu", 32, 5)
        self.assertGreater(float(cpu[..., :3].max()), 0.05)
        np.testing.assert_allclose(gpu, cpu, atol=.01, rtol=.01)

    def test_the_cpu_twin_matches_the_card_on_the_flame_channel_too(self):
        dense, _ = blob()
        scene = s.Scene(volumes=(flamed(dense),))
        np.testing.assert_allclose(trace(scene, "gpu", 32, 5, FLAME), trace(scene, "cpu", 32, 5, FLAME), atol=.01, rtol=.01)

    def test_the_old_estimator_switches_keep_the_two_tracers_in_lockstep(self):
        """`ENABLE_EXPECTED_EMISSION` False on both gives the collision-summed glow the evidence compares against. (The card keeps a
        scene's packed form, so each estimator renders its own scene object.)"""
        with mock.patch.object(gpupathtrace, "ENABLE_EXPECTED_EMISSION", False), collisions():
            scene = s.Scene(volumes=(blob()[0],))
            old_gpu = trace(scene, "gpu", 32, 5)
            old_cpu = trace(scene, "cpu", 32, 5)
        new = trace(s.Scene(volumes=(blob()[0],)), "gpu", 32, 5)
        np.testing.assert_allclose(old_gpu, old_cpu, atol=.01, rtol=.01)
        self.assertGreater(float(np.abs(old_gpu - new).max()), 1e-3, "the switch must reach the shader")

    def test_a_sparse_volume_glows_like_the_dense_one_at_the_same_seed(self):
        dense, sparse = blob()
        a = trace(s.Scene(volumes=(dense,)), "gpu", 48, 3)
        b = trace(s.Scene(volumes=(sparse,)), "gpu", 48, 3)
        self.assertGreater(float(a[..., :3].max()), 0.05)
        np.testing.assert_allclose(b, a, atol=2e-4)

    def test_the_render_is_deterministic_for_a_seed(self):
        dense, _ = blob()
        scene = s.Scene(volumes=(dense,))
        np.testing.assert_array_equal(trace(scene, "gpu", 32, 4), trace(scene, "gpu", 32, 4))

    def test_two_volumes_keep_the_collision_estimator_and_agree_with_the_cpu(self):
        dense, _ = blob()
        near = replace(dense, origin=tuple(np.asarray(dense.origin) + (0.0, 0.0, 0.3)))
        scene = s.Scene(volumes=(dense, near))
        cpu = trace(scene, "cpu", 384, 7)
        gpu = trace(scene, "gpu", 1024, 3)
        self.assertGreater(float(cpu[..., :3].max()), 0.05)
        self.assertAlmostEqual(float(gpu[..., :3].mean()) / float(cpu[..., :3].mean()), 1.0, delta=0.05)

    def test_smoke_in_front_of_the_fire_dims_it(self):
        """The weight is the transmittance: raising the density of a volume in the fire's line of sight lowers the glow."""
        dense, _ = blob()
        thin = trace(s.Scene(volumes=(dense,)), "gpu", 128, 3, replace(FIRE, density_scale=2.0))
        thick = trace(s.Scene(volumes=(dense,)), "gpu", 128, 3, replace(FIRE, density_scale=24.0))
        self.assertGreater(float(thin[..., :3].mean()), 1.05 * float(thick[..., :3].mean()))


class ShaderConstants(unittest.TestCase):
    """The quadrature's constants sit in the WGSL and in `ptvolume`, the CPU twin; they must say the same thing."""

    def shader(self):
        with open(gpupathtrace.__file__) as handle:
            return handle.read()

    def test_the_steps_are_four_voxels_for_the_camera_flight_and_sixteen_for_scatter_continuations(self):
        source = self.shader()
        self.assertIn(f"const EMIT_CELLS: f32 = {ptvolume.EMIT_CELLS:.1f};", source)
        self.assertIn(f"const EMIT_CELLS_DEEP: f32 = {ptvolume.EMIT_CELLS_DEEP:.1f};", source)
        self.assertEqual((ptvolume.EMIT_CELLS, ptvolume.EMIT_CELLS_DEEP), (4.0, 16.0))

    def test_the_optical_depth_limits_and_the_blend_match(self):
        source = self.shader()
        self.assertIn(f"const EMIT_TAU: f32 = {ptvolume.EMIT_TAU:.1f};", source)
        self.assertIn(f"if (tau > {ptvolume.EMIT_TAU_CAP:.1f}) {{ break; }}", source)
        self.assertIn(f"({ptvolume.EMIT_SELF} * sigma_t + {1 - ptvolume.EMIT_SELF} * sigma_prev)", source)


if __name__ == "__main__":
    unittest.main()
