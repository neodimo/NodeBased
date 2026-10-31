"""Lane 6, fire in smoke: smoke vertices sample the fire as a light (next-event estimation) and weigh it against the phase-sampled
continuation ray that also reaches it.

Before this step the fire lit smoke only when a continuation ray happened to pass through it, so a small bright flame turned the
light at a smoke vertex into a rare large jump. Now every vertex also draws one point of the fire from the coarse emission grid,
shades it through the smoke by the midpoint rule, and the two strategies share each point by the balance heuristic. These tests hold
the estimator's three promises: the distribution is a distribution (positive wherever the fine field can glow), the sum of the two
strategies keeps the mean of the one it replaces, and at one smoke vertex it is much quieter. They also hold the CPU twin against the
card. Run on the other adapters with `scratch/nb-lanes/run/force-adapter.py integrated|cpu`."""
import os
import unittest
from dataclasses import replace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from nodebased import gpupathtrace, pathtrace as pt, ptvolume, scene3d as s, volumerender as vr
from tests.test_3d_pathtrace_gpu_soft import CAMERA, READY
from tests.test_fire_grain import FIRE, SIZE


def spark(haze=0.35, nx=40, ny=32, nz=24):
    """Thin haze through the whole box and one small, hot, dense knot of fire in it: smoke lit by a compact bright source."""
    x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    hot = (x - 8) ** 2 + (y - 8) ** 2 + (z - 12) ** 2 < 9
    return s.Volume(density=np.where(hot, 2.5, haze).astype(np.float32),
                    temperature=np.where(hot, 1.2, 0.0).astype(np.float32), voxel_size=0.075, origin=(-1.0, -0.7, -0.9))


def trace(backend, samples, seed, output="diffuse", settings=FIRE):
    """A fresh volume per call: the card keeps a scene's packed form, so each estimator renders its own scene object."""
    scene = s.Scene(volumes=(spark(),))
    return pt.render(scene, CAMERA, *SIZE, ambient=0.0, volume=settings, backend=backend, output=output,
                     settings=pt.PathSettings(samples=samples, seed=seed))[..., :3]


class off:
    """Both tracers without next-event sampling of the fire (the estimator this step replaced)."""

    def __enter__(self):
        self.patches = [mock.patch.object(ptvolume, "ENABLE_FIRE_NEE", False), mock.patch.object(gpupathtrace, "ENABLE_FIRE_NEE", False)]
        for patch in self.patches:
            patch.start()

    def __exit__(self, *exc):
        for patch in self.patches:
            patch.stop()


def layer(settings=FIRE, volumes=None):
    return ptvolume.build(s.Scene(volumes=volumes or (spark(),)), settings)


def keys(n, seed=1):
    return np.random.default_rng(seed).integers(0, 2 ** 32, n, dtype=np.uint32)


class EmitterTests(unittest.TestCase):
    def test_it_is_a_probability_distribution_with_no_empty_cell(self):
        emitter = layer().emitter()
        self.assertAlmostEqual(float(emitter.pmf.sum()), 1.0, places=12)
        self.assertTrue((emitter.pmf > 0).all(), "a cell the fine field could glow in must be reachable")
        self.assertTrue((np.diff(emitter.cdf) >= 0).all())
        self.assertEqual(float(emitter.cdf[-1]), 1.0)

    def test_the_fire_holds_nearly_all_the_probability(self):
        la = layer()
        emitter = la.emitter()
        coarse, factor = vr.fire_coarse_emission(la.preps[0].volume, la.settings)
        glow = (coarse.sum(-1) > 0).reshape(-1)
        self.assertGreater(float(emitter.pmf[glow].sum()), 0.95)
        self.assertEqual(factor, emitter.factor)

    def test_a_drawn_point_has_the_density_the_emitter_reports_for_it(self):
        la = layer()
        point, wi, dist, e, pv = ptvolume.fire_nee_sample(la, np.zeros((500, 3)), keys(500), 2)
        np.testing.assert_allclose(la.emitter().density_at(la.preps[0].to_object(point)), pv, rtol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(wi, axis=1), 1.0, atol=1e-9)

    def test_the_density_of_a_cell_centre_is_its_probability_over_the_cell_volume(self):
        la = layer()
        emitter = la.emitter()
        gx, gy, gz = (int(n) for n in emitter.cells)
        centres = np.array([[i, j, k] for i in range(gx) for j in range(gy) for k in range(gz)], np.float64) + 0.5
        density = emitter.density_at(la.preps[0].box_min + centres * emitter.cell_object)
        np.testing.assert_allclose(density * emitter.cell_world ** 3, emitter.pmf, rtol=1e-9)
        self.assertEqual(float(emitter.density_at((la.preps[0].box_min - 1.0)[None, :])[0]), 0.0, "nothing outside the grid")

    def test_it_is_off_without_a_gain_a_fire_or_a_second_volume(self):
        self.assertIsNone(layer(replace(FIRE, fire_light=0.0)).emitter())
        self.assertIsNone(layer(replace(FIRE, fire_intensity=0.0)).emitter())
        near = replace(spark(), origin=(-1.0, -0.7, -0.6))
        self.assertIsNone(layer(volumes=(spark(), near)).emitter())
        with off():
            self.assertIsNone(layer().emitter())

    def test_a_scene_with_splats_does_not_sample_the_fire(self):
        class Splatted:
            splats = object()
            volumes = layer()
        self.assertFalse(pt._fire_nee_on(Splatted))
        Splatted.splats = None
        self.assertTrue(pt._fire_nee_on(Splatted))


class VertexEstimatorTests(unittest.TestCase):
    """One smoke vertex, many hashed numbers: the fire's light there by the two strategies together against the old one."""
    N = 30000
    VERTEX = np.array([-0.6, -0.3, -0.2])
    ARRIVING = np.array([0.0, 0.0, -1.0])

    def estimates(self):
        la = layer()
        n = self.N
        pos, arriving, key = np.tile(self.VERTEX, (n, 1)), np.tile(self.ARRIVING, (n, 1)), keys(n)
        _, wi, dist, e, pv = ptvolume.fire_nee_sample(la, pos, key, 3)
        phase = ptvolume.phase(0.0, (wi * arriving).sum(1))
        sampled = ptvolume.fire_nee_value(la, wi, dist, e, pv, phase, ptvolume.fire_transmittance(la, 0, pos, wi, dist)).mean(1)
        direction, pdf = ptvolume.sample_phase(0.0, arriving, pt.rand(key, 77), pt.rand(key, 78))
        args = (la, 0, pos, direction, np.full(n, np.inf), key, 3, np.ones(n, bool), None)
        weighed = ptvolume.expected_emission(*args, (np.ones(n, bool), pdf)).mean(1)
        plain = ptvolume.expected_emission(*args, None).mean(1)
        return sampled, weighed, plain

    def test_both_strategies_together_keep_the_mean_of_the_continuation_alone(self):
        sampled, weighed, plain = self.estimates()
        together = sampled + weighed
        error = 4 * float(np.sqrt(plain.var() / self.N + together.var() / self.N))
        self.assertAlmostEqual(float(together.mean()), float(plain.mean()), delta=max(error, 0.04 * float(plain.mean())))

    def test_the_two_strategies_together_are_much_quieter_than_the_continuation_alone(self):
        sampled, weighed, plain = self.estimates()
        self.assertLess(float((sampled + weighed).std()), 0.6 * float(plain.std()))

    def test_the_continuation_hands_the_far_field_to_the_sampled_fire_and_never_gains(self):
        _, weighed, plain = self.estimates()
        self.assertLess(float(weighed.mean()), 0.5 * float(plain.mean()))
        self.assertLessEqual(float(weighed.max()), float(plain.max()) * (1 + 1e-9))

    def test_the_midpoint_transmittance_has_the_mean_of_ratio_tracking(self):
        la = layer()
        n = 6000
        pos = np.tile([0.5, 0.3, 0.7], (n, 1))
        direction = np.tile([-0.8, -0.5, -0.33], (n, 1))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        dist = np.full(n, 1.6)
        deterministic = float(ptvolume.fire_transmittance(la, 0, pos, direction, dist)[0])
        ratio = ptvolume.ratio_transmittance(la, 0, pos, direction, dist, keys(n), 5)
        self.assertGreater(deterministic, 0.0)
        self.assertLess(deterministic, 0.9, "the ray must cross smoke")
        self.assertAlmostEqual(float(ratio.mean()), deterministic, delta=max(0.03, 5 * float(ratio.std()) / np.sqrt(n)))
        self.assertGreater(float(ratio.std()), 0.05, "ratio tracking is the noisier of the two")

    def test_a_ray_that_misses_the_smoke_is_not_dimmed(self):
        la = layer()
        out = ptvolume.fire_transmittance(la, 0, np.array([[5.0, 5.0, 5.0]]), np.array([[0.0, 1.0, 0.0]]), np.array([3.0]))
        self.assertEqual(float(out[0]), 1.0)


class CpuRenderTests(unittest.TestCase):
    def test_the_cpu_render_lights_the_haze_with_or_without_the_sampling_to_the_same_mean(self):
        new = np.mean([trace("cpu", 96, seed).mean() for seed in (1, 2)])
        with off():
            old = np.mean([trace("cpu", 96, seed).mean() for seed in (1, 2)])
        self.assertGreater(float(new), 1e-4, "the haze must be lit by the fire")
        self.assertAlmostEqual(float(new / old), 1.0, delta=0.2)


@unittest.skipUnless(READY, "no wgpu adapter cleared for smoke in the path tracer")
class OnTheCard(unittest.TestCase):
    def test_the_card_matches_its_cpu_twin_pixel_for_pixel_on_the_same_seed(self):
        cpu, gpu = trace("cpu", 32, 5), trace("gpu", 32, 5)
        self.assertGreater(float(cpu.max()), 0.005)
        np.testing.assert_allclose(gpu, cpu, atol=.005, rtol=.01)

    def test_the_card_matches_its_cpu_twin_with_the_switch_off_too(self):
        with off():
            cpu, gpu = trace("cpu", 32, 5), trace("gpu", 32, 5)
        np.testing.assert_allclose(gpu, cpu, atol=.005, rtol=.01)

    def test_the_switch_reaches_the_shader(self):
        gpu = trace("gpu", 32, 5)
        with off():
            old = trace("gpu", 32, 5)
        self.assertGreater(float(np.abs(gpu - old).max()), 1e-5)

    def test_the_mean_light_on_the_haze_is_unchanged_at_many_samples(self):
        new = np.mean([trace("gpu", 1536, seed).mean() for seed in (3, 4)])
        with off():
            old = np.mean([trace("gpu", 1536, seed).mean() for seed in (3, 4)])
        self.assertAlmostEqual(float(new / old), 1.0, delta=0.05)

    def test_the_whole_picture_with_the_fire_seen_directly_is_unchanged_too(self):
        new = np.mean([trace("gpu", 1024, seed, "rgba").mean() for seed in (3, 4)])
        with off():
            old = np.mean([trace("gpu", 1024, seed, "rgba").mean() for seed in (3, 4)])
        self.assertAlmostEqual(float(new / old), 1.0, delta=0.05)

    def test_without_a_fire_light_gain_the_switch_changes_nothing(self):
        dark = replace(FIRE, fire_light=0.0)
        a = trace("gpu", 24, 5, settings=dark)
        with off():
            b = trace("gpu", 24, 5, settings=dark)
        np.testing.assert_array_equal(a, b)

    def test_the_gain_scales_the_light_the_fire_casts_on_the_haze(self):
        full = trace("gpu", 512, 3).mean()
        half = trace("gpu", 512, 3, settings=replace(FIRE, fire_light=0.5)).mean()
        none = trace("gpu", 512, 3, settings=replace(FIRE, fire_light=0.0)).mean()
        self.assertLess(float(none), 0.02 * float(full))
        self.assertAlmostEqual(float(half / full), 0.5, delta=0.1)

    def test_the_render_is_deterministic_for_a_seed(self):
        np.testing.assert_array_equal(trace("gpu", 24, 4), trace("gpu", 24, 4))


class ShaderConstants(unittest.TestCase):
    """The estimator's constants sit in the WGSL and in `ptvolume`, the CPU twin; they must say the same thing."""

    def shader(self):
        with open(gpupathtrace.__file__) as handle:
            return handle.read()

    def test_the_shadow_quadrature_matches(self):
        source = self.shader()
        self.assertIn(f"const NEE_STEP_CELLS: f32 = {ptvolume.NEE_STEP_CELLS:.1f};", source)
        self.assertIn(f"const NEE_TAU: f32 = {ptvolume.NEE_TAU:.1f};", source)
        self.assertIn(f"const NEE_MAX_STEPS: f32 = {float(ptvolume.NEE_MAX_STEPS):.1f};", source)

    def test_the_random_stream_and_the_header_size_match(self):
        source = self.shader()
        self.assertIn(f"const NEE_STREAM: u32 = {ptvolume.NEE_STREAM}u;", source)
        self.assertIn(f"const VOL_VECS: u32 = {gpupathtrace.VOL_VECS}u;", source)


if __name__ == "__main__":
    unittest.main()
