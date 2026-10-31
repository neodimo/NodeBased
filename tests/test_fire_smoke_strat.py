"""Lane 6, fire in smoke: the camera ray's first collision in the smoke is stratified over a pixel's samples.

Delta tracking decides, for each of a pixel's samples, whether the camera ray scatters in a thin plume at all: a coin flip per sample,
so a pixel is dark in one sample and bright in the next whatever the lighting at the collision does. This step finds the first collision
by inverting the optical depth (`ptvolume.stratified_event`) from a number that is spread evenly over the pixel's samples
(`ptvolume.stratified_number`), and absorbs by weight at that first vertex instead of by a coin flip. These tests hold the promises: the
numbers are stratified and uniform, the collision has the distribution delta tracking gives, the pixel's scatter count is counted rather
than drawn, the mean image is unchanged, and the CPU twin agrees with the card. Run on the other adapters with
`scratch/nb-lanes/run/force-adapter.py integrated|cpu`."""
import os
import unittest
from dataclasses import replace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from nodebased import gpupathtrace, pathtrace as pt, ptvolume, scene3d as s
from tests.test_3d_pathtrace_gpu_soft import CAMERA, READY
from tests.test_fire_grain import FIRE, SIZE
from tests.test_fire_smoke_nee import layer, spark, trace


class off:
    """Both tracers with delta tracking at the first collision again (fire sampling stays on: the step this one builds on)."""

    def __enter__(self):
        self.patches = [mock.patch.object(ptvolume, "ENABLE_STRATIFIED_FIRST", False),
                        mock.patch.object(gpupathtrace, "ENABLE_STRATIFIED_FIRST", False)]
        for patch in self.patches:
            patch.start()

    def __exit__(self, *exc):
        for patch in self.patches:
            patch.stop()


def thin_trace(backend, samples, seed, output="rgba", haze=0.02):
    """A thin haze through the whole box with the small hot knot of `spark`: most camera rays scatter in the haze with a chance well under 1."""
    scene = s.Scene(volumes=(spark(haze=haze),))
    return pt.render(scene, CAMERA, *SIZE, ambient=0.0, volume=FIRE, backend=backend, output=output,
                     settings=pt.PathSettings(samples=samples, seed=seed))


class StratifiedNumberTests(unittest.TestCase):
    def test_the_bits_are_reversed(self):
        np.testing.assert_array_equal(ptvolume.bit_reverse(np.array([0, 1, 2, 3, 0x80000000], np.uint32)),
                                      np.array([0, 0x80000000, 0x40000000, 0xC0000000, 1], np.uint32))

    def test_a_run_of_samples_covers_every_stratum_once(self):
        for pixel in (0, 7, 12345):
            for count in (8, 64, 128):
                xi = ptvolume.stratified_number(np.full(count, pixel), np.arange(count), 3)
                self.assertEqual(sorted((xi * count).astype(int)), list(range(count)), (pixel, count))

    def test_a_run_that_starts_late_is_stratified_too(self):
        xi = ptvolume.stratified_number(np.full(64, 5), 64 + np.arange(64), 3)
        self.assertEqual(sorted((xi * 64).astype(int)), list(range(64)))

    def test_every_pixel_rotates_the_sequence_differently_and_the_numbers_stay_uniform(self):
        xi = ptvolume.stratified_number(np.arange(4096), np.full(4096, 9), 1)
        self.assertGreaterEqual(float(xi.min()), 0.0)
        self.assertLess(float(xi.max()), 1.0)
        self.assertAlmostEqual(float(xi.mean()), 0.5, delta=0.02)
        self.assertAlmostEqual(float(np.histogram(xi, bins=4, range=(0, 1))[0].min()) / 1024, 1.0, delta=0.12)

    def test_the_seed_changes_the_numbers(self):
        a = ptvolume.stratified_number(np.arange(16), np.zeros(16, int), 1)
        b = ptvolume.stratified_number(np.arange(16), np.zeros(16, int), 2)
        self.assertGreater(float(np.abs(a - b).max()), 0.05)


class EventTests(unittest.TestCase):
    """`stratified_event` against delta tracking on the same rays."""

    def rays(self, oy, oz, count):
        o = np.repeat(np.array([[-1.6, oy, oz]]), count, 0)
        d = np.repeat(np.array([[1.0, 0.0, 0.0]]), count, 0)
        return o, d

    def delta(self, lay, o, d, seed):
        n = len(o)
        keys = pt.path_key(np.arange(n), np.zeros(n, int), seed)
        return ptvolume.free_flight(lay, o, d, np.full(n, np.inf), keys, 0)[0]

    def test_it_collides_where_delta_tracking_does(self):
        lay = layer()
        n = 60000
        xi = (np.arange(n) + 0.5) / n
        for oy, oz in ((0.5, -0.5), (0.0, 0.0), (0.2, 0.1)):
            o, d = self.rays(oy, oz, n)
            strat = ptvolume.stratified_event(lay, 0, o, d, np.full(n, np.inf), xi)
            delta = self.delta(lay, o, d, 11)
            grid = np.linspace(0.5, 4.0, 36)
            gap = max(abs((strat <= g).mean() - (delta <= g).mean()) for g in grid)
            self.assertLess(gap, 0.025, (oy, oz))

    def test_a_homogeneous_slab_collides_with_the_exponential_chance(self):
        thin = s.Volume(density=np.full((20, 20, 20), 0.1, np.float32), voxel_size=0.1, origin=(-1.0, -1.0, -1.0))
        lay = ptvolume.build(s.Scene(volumes=(thin,)), FIRE)
        sigma = lay.sigma_t_unit * FIRE.density_scale * 0.1
        n = 40000
        xi = (np.arange(n) + 0.5) / n
        o = np.repeat(np.array([[-1.5, 0.0, 0.0]]), n, 0)
        d = np.repeat(np.array([[1.0, 0.0, 0.0]]), n, 0)
        t = ptvolume.stratified_event(lay, 0, o, d, np.full(n, np.inf), xi)
        self.assertAlmostEqual(float(np.isfinite(t).mean()), 1 - np.exp(-sigma * 2.0), delta=0.02)
        inside = t[np.isfinite(t)] - 0.5
        self.assertAlmostEqual(float((inside <= 1.0).mean()), (1 - np.exp(-sigma * 1.0)) / (1 - np.exp(-sigma * 2.0)), delta=0.02)

    def test_a_ray_that_stops_before_the_smoke_or_inside_it_collides_less(self):
        lay = layer()
        n = 4000
        xi = (np.arange(n) + 0.5) / n
        o, d = self.rays(0.5, -0.5, n)
        full = ptvolume.stratified_event(lay, 0, o, d, np.full(n, np.inf), xi)
        short = ptvolume.stratified_event(lay, 0, o, d, np.full(n, 0.9), xi)
        before = ptvolume.stratified_event(lay, 0, o, d, np.full(n, 0.5), xi)
        self.assertTrue(np.all(short[np.isfinite(short)] <= 0.9))
        self.assertLess(float(np.isfinite(short).mean()), float(np.isfinite(full).mean()))
        self.assertFalse(np.isfinite(before).any())

    def test_a_ray_that_misses_the_smoke_never_collides(self):
        lay = layer()
        o = np.repeat(np.array([[-1.6, 5.0, 0.0]]), 100, 0)
        d = np.repeat(np.array([[1.0, 0.0, 0.0]]), 100, 0)
        self.assertFalse(np.isfinite(ptvolume.stratified_event(lay, 0, o, d, np.full(100, np.inf), np.linspace(0, 0.99, 100))).any())

    def test_a_pixels_scattering_samples_are_counted_not_drawn(self):
        thin = s.Volume(density=np.full((20, 20, 20), 0.02, np.float32), voxel_size=0.1, origin=(-1.0, -1.0, -1.0))
        lay = ptvolume.build(s.Scene(volumes=(thin,)), FIRE)
        p = 1 - np.exp(-lay.sigma_t_unit * FIRE.density_scale * 0.02 * 2.0)
        counts, drawn = [], []
        for pixel in range(200):
            xi = ptvolume.stratified_number(np.full(64, pixel), np.arange(64), 1)
            o = np.repeat(np.array([[-1.5, 0.0, 0.0]]), 64, 0)
            d = np.repeat(np.array([[1.0, 0.0, 0.0]]), 64, 0)
            counts.append(int(np.isfinite(ptvolume.stratified_event(lay, 0, o, d, np.full(64, np.inf), xi)).sum()))
            drawn.append(int(np.isfinite(self.delta(lay, o, d, pixel)).sum()))
        self.assertAlmostEqual(float(np.mean(counts)) / 64, p, delta=0.02)
        self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertGreater(float(np.var(drawn)), 5.0)


class FreeFlightTests(unittest.TestCase):
    def test_nan_rays_keep_delta_tracking_and_numbered_rays_take_the_stratified_event(self):
        lay = layer()
        n = 3000
        o = np.repeat(np.array([[-1.6, 0.5, -0.5]]), n, 0)
        d = np.repeat(np.array([[1.0, 0.0, 0.0]]), n, 0)
        keys = pt.path_key(np.arange(n), np.zeros(n, int), 5)
        xi = np.where(np.arange(n) % 2 == 0, (np.arange(n) + 0.5) / n, np.nan)
        t, _, owner = ptvolume.free_flight(lay, o, d, np.full(n, np.inf), keys, 0, xi=xi)
        reference = ptvolume.stratified_event(lay, 0, o[::2], d[::2], np.full(n // 2, np.inf), xi[::2])
        np.testing.assert_allclose(t[::2], reference)
        t_delta, _, _ = ptvolume.free_flight(lay, o, d, np.full(n, np.inf), keys, 0)
        np.testing.assert_array_equal(t[1::2], t_delta[1::2])
        self.assertTrue(np.all(owner[np.isfinite(t)] == 0))

    def test_the_switch_turns_the_stratified_event_off(self):
        lay = layer()
        n = 500
        o = np.repeat(np.array([[-1.6, 0.5, -0.5]]), n, 0)
        d = np.repeat(np.array([[1.0, 0.0, 0.0]]), n, 0)
        keys = pt.path_key(np.arange(n), np.zeros(n, int), 5)
        xi = (np.arange(n) + 0.5) / n
        with off():
            t, _, _ = ptvolume.free_flight(lay, o, d, np.full(n, np.inf), keys, 0, xi=xi)
        np.testing.assert_array_equal(t, ptvolume.free_flight(lay, o, d, np.full(n, np.inf), keys, 0)[0])

    def test_two_volumes_keep_delta_tracking(self):
        lay = ptvolume.build(s.Scene(volumes=(spark(), spark())), FIRE)
        n = 200
        o = np.repeat(np.array([[-1.6, 0.5, -0.5]]), n, 0)
        d = np.repeat(np.array([[1.0, 0.0, 0.0]]), n, 0)
        keys = pt.path_key(np.arange(n), np.zeros(n, int), 5)
        xi = (np.arange(n) + 0.5) / n
        np.testing.assert_array_equal(ptvolume.free_flight(lay, o, d, np.full(n, np.inf), keys, 0, xi=xi)[0],
                                      ptvolume.free_flight(lay, o, d, np.full(n, np.inf), keys, 0)[0])


class CpuRenderTests(unittest.TestCase):
    def test_a_thin_plumes_coverage_is_quieter_between_seeds(self):
        def grain():
            a, b = (thin_trace("cpu", 32, seed)[..., 3] for seed in (1, 2))
            return float(np.sqrt(((a - b) ** 2).mean() / 2))
        new = grain()
        with off():
            old = grain()
        self.assertLess(new, 0.5 * old)

    def test_the_cpu_render_keeps_the_mean(self):
        new = np.mean([thin_trace("cpu", 128, seed, "diffuse")[..., :3].mean() for seed in (1, 2, 3)])
        with off():
            old = np.mean([thin_trace("cpu", 128, seed, "diffuse")[..., :3].mean() for seed in (1, 2, 3)])
        self.assertGreater(float(new), 1e-5)
        self.assertAlmostEqual(float(new / old), 1.0, delta=0.15)

    def test_without_fire_sampling_the_first_collision_is_delta_tracked(self):
        dark = replace(FIRE, fire_light=0.0)
        a = trace("cpu", 8, 5, settings=dark)
        with off():
            b = trace("cpu", 8, 5, settings=dark)
        np.testing.assert_array_equal(a, b)


@unittest.skipUnless(READY, "no wgpu adapter cleared for smoke in the path tracer")
class OnTheCard(unittest.TestCase):
    def test_the_card_matches_its_cpu_twin_pixel_for_pixel_on_the_same_seed(self):
        cpu, gpu = trace("cpu", 32, 5), trace("gpu", 32, 5)
        self.assertGreater(float(cpu.max()), 0.005)
        np.testing.assert_allclose(gpu, cpu, atol=.005, rtol=.01)

    def test_the_card_matches_its_cpu_twin_in_a_thin_plume(self):
        cpu, gpu = thin_trace("cpu", 32, 5, "diffuse")[..., :3], thin_trace("gpu", 32, 5, "diffuse")[..., :3]
        self.assertGreater(float(cpu.max()), 0.001)
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

    def test_the_mean_image_is_unchanged_at_many_samples(self):
        new = np.mean([trace("gpu", 1536, seed, "rgba").mean() for seed in (3, 4)])
        with off():
            old = np.mean([trace("gpu", 1536, seed, "rgba").mean() for seed in (3, 4)])
        self.assertAlmostEqual(float(new / old), 1.0, delta=0.05)

    def test_the_mean_light_in_a_thin_plume_is_unchanged_at_many_samples(self):
        new = np.mean([thin_trace("gpu", 1536, seed, "diffuse")[..., :3].mean() for seed in (3, 4)])
        with off():
            old = np.mean([thin_trace("gpu", 1536, seed, "diffuse")[..., :3].mean() for seed in (3, 4)])
        self.assertGreater(float(new), 1e-5)
        self.assertAlmostEqual(float(new / old), 1.0, delta=0.05)

    def test_the_coverage_of_a_thin_plume_is_unchanged_and_much_quieter(self):
        def render():
            return [thin_trace("gpu", 64, seed)[..., 3] for seed in (1, 2)]
        new = render()
        with off():
            old = render()
        self.assertAlmostEqual(float(np.mean(new) / np.mean(old)), 1.0, delta=0.05)
        grain = lambda pair: float(np.sqrt(((pair[0] - pair[1]) ** 2).mean() / 2))
        self.assertLess(grain(new), 0.5 * grain(old))

    def test_a_thin_plume_is_quieter_in_the_beauty_pass(self):
        def grain():
            a, b = (thin_trace("gpu", 64, seed, "diffuse")[..., :3] for seed in (1, 2))
            return float(np.sqrt(((a - b) ** 2).mean() / 2))
        new = grain()
        with off():
            old = grain()
        self.assertLess(new, old)

    def test_the_render_is_deterministic_for_a_seed(self):
        np.testing.assert_array_equal(trace("gpu", 24, 4), trace("gpu", 24, 4))


class ShaderConstants(unittest.TestCase):
    """The estimator's constants sit in the WGSL and in `ptvolume`, the CPU twin; they must say the same thing."""

    def shader(self):
        with open(gpupathtrace.__file__) as handle:
            return handle.read()

    def test_the_quadrature_matches(self):
        source = self.shader()
        self.assertIn(f"const STRAT_CELLS: f32 = {ptvolume.STRAT_CELLS};", source)
        self.assertIn(f"const STRAT_TAU: f32 = {ptvolume.STRAT_TAU};", source)

    def test_the_rotation_salt_matches(self):
        self.assertIn(f"const STRAT_SALT: u32 = 0x{ptvolume.STRAT_SALT:X}u;", self.shader())

    def test_the_switches_start_on(self):
        self.assertTrue(ptvolume.ENABLE_STRATIFIED_FIRST)
        self.assertTrue(gpupathtrace.ENABLE_STRATIFIED_FIRST)


if __name__ == "__main__":
    unittest.main()
