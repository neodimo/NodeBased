"""Lane 6, fire shape step B: the flame glow can take its colour and brightness from the gas temperature.

`volume_flame_follow_temperature` off (the default) keeps the single flame colour, so a saved document renders as before. On, the
burn-rate glow uses the kelvin of the temperature field in the same blackbody table, so a flame that cools with height goes from
yellow-white to orange to dark red. Every renderer reads the same rule: the CPU and GPU raymarchers, the two path tracers (collision
glow and the integrated glow) and the coarse fire-light grid. Run on the other adapters with
`scratch/nb-lanes/run/force-adapter.py integrated|cpu`."""
import os
import unittest
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from unittest import mock

from nodebased import fluid3d, gpu3d, gpupathtrace, particles, pathtrace, scene3d, volumerender
from nodebased.core import SPECS, upgrade_document
from nodebased.imaging import _volume_settings
from tests.test_volume_fire_look import BASE, box

TABLE = volumerender.fire_table(volumerender.VolumeSettings())
GLOW = replace(BASE, flame_intensity=2.0, fire_light=0.0, temperature_scale=1000.0, fire_threshold=500.0)
FOLLOW = replace(GLOW, flame_follow=1)


def cooling_flame(scale=1.0, density=0.25, n=16):
    """A flame block whose temperature falls with height: 2 at the bottom to 0.4 at the top of the block, times `scale`."""
    volume = box(n, density, 0.0, size=2.0)
    temperature = np.zeros((n, n, n), np.float32)
    flame = np.zeros((n, n, n), np.float32)
    ramp = np.linspace(2.0, 0.4, 8, dtype=np.float32)
    temperature[5:11, 4:12, 5:11] = ramp[None, :, None]
    flame[5:11, 4:12, 5:11] = 2.0
    return replace(volume, temperature=temperature * scale, flame=flame)


def raster(volume, settings, backend="cpu"):
    scene = scene3d.Scene(volumes=(volume,))
    render = scene3d.render if backend == "cpu" else gpu3d.render
    return render(scene, scene3d.Camera(), 40, 40, volume=settings)


class FlameColourTests(unittest.TestCase):
    def test_off_is_the_one_flame_colour_and_ignores_the_temperature(self):
        settings = volumerender.VolumeSettings(flame_temperature=1800.0)
        constant = volumerender.fire_radiance(TABLE, np.array([1800.0]))[0]
        np.testing.assert_array_equal(volumerender.flame_colour(settings, TABLE), constant)
        np.testing.assert_array_equal(volumerender.flame_colour(settings, TABLE, np.array([0.1, 5.0])), constant)

    def test_on_follows_the_kelvin_of_the_temperature(self):
        settings = volumerender.VolumeSettings(flame_follow=1, temperature_scale=1000.0, fire_threshold=500.0)
        temperature = np.array([[0.4, 1.0, 2.0], [3.0, 0.2, 1.5]])
        colour = volumerender.flame_colour(settings, TABLE, temperature)
        self.assertEqual(colour.shape, (2, 3, 3))
        np.testing.assert_allclose(colour[0, 1], volumerender.fire_radiance(TABLE, np.array([1000.0]))[0])
        self.assertTrue((colour[0, 0] == 0).all() and (colour[1, 1] == 0).all(), "at or below the threshold: no glow")
        luminance = colour @ np.array([0.2722, 0.6741, 0.0537])
        self.assertGreater(luminance[0, 2], luminance[0, 1])
        self.assertGreater(luminance[1, 0], luminance[0, 2])
        redness = colour[..., 1] / np.maximum(colour[..., 0], 1e-9)
        self.assertGreater(redness[0, 2], redness[0, 1], "a hotter flame is whiter")

    def test_without_a_temperature_field_the_flame_keeps_its_colour(self):
        settings = volumerender.VolumeSettings(flame_follow=1, flame_temperature=1800.0)
        np.testing.assert_array_equal(volumerender.flame_colour(settings, TABLE),
                                      volumerender.fire_radiance(TABLE, np.array([1800.0]))[0])

    def test_the_fire_light_grid_cools_with_the_flame(self):
        flame = np.full((2, 1, 1), 1.5, np.float32)
        temperature = np.array([[[2.0]], [[0.8]]], np.float32)
        density = np.zeros_like(flame)
        off = volumerender._fire_emission(GLOW, volumerender.fire_table(GLOW), density, temperature, flame)
        on = volumerender._fire_emission(FOLLOW, volumerender.fire_table(FOLLOW), density, temperature, flame)
        np.testing.assert_array_equal(off[0], off[1])
        self.assertGreater(float(on[0].sum()), 5 * float(on[1].sum()))


class SettingsTests(unittest.TestCase):
    def test_the_knob_defaults_off_and_an_old_document_renders_with_it_off(self):
        self.assertEqual(SPECS["Render3D"]["params"]["volume_flame_follow_temperature"], 0)
        document = {"version": 1, "nodes": {"r": {"id": "r", "type": "Render3D", "params": {}}}, "connections": []}
        params = upgrade_document(document)["nodes"]["r"]["params"]
        params.pop("volume_flame_follow_temperature", None)
        self.assertEqual(_volume_settings(params).flame_follow, 0)
        params["volume_flame_follow_temperature"] = 1
        self.assertEqual(_volume_settings(params).flame_follow, 1)


class RasterTests(unittest.TestCase):
    def test_off_ignores_the_temperature_and_on_cools_the_glow(self):
        hot, cool = cooling_flame(1.0, 0.0), cooling_flame(0.5, 0.0)
        np.testing.assert_array_equal(raster(hot, GLOW), raster(cool, GLOW))
        on_hot, on_cool = raster(hot, FOLLOW), raster(cool, FOLLOW)
        self.assertGreater(float(on_hot[..., :3].sum()), 1.5 * float(on_cool[..., :3].sum()))
        self.assertGreater(float(on_cool[..., :3].sum()), 0)
        np.testing.assert_array_equal(on_hot[..., 3], raster(hot, GLOW)[..., 3])

    @unittest.skipUnless(gpu3d.available(), "no wgpu adapter")
    def test_the_card_matches_the_cpu_raymarch(self):
        for volume in (cooling_flame(), cooling_flame(0.5)):
            np.testing.assert_allclose(raster(volume, FOLLOW, "gpu"), raster(volume, FOLLOW), atol=.005, rtol=.005)
        volume = cooling_flame()
        np.testing.assert_allclose(raster(volume, GLOW, "gpu"), raster(volume, GLOW), atol=.005, rtol=.005)

    @unittest.skipUnless(gpu3d.available(), "no wgpu adapter")
    def test_a_sparse_volume_follows_too(self):
        dense = cooling_flame()
        dense = replace(dense, density=np.zeros_like(dense.density))
        sparse = scene3d.Volume.from_sparse(dense.to_sparse(), voxel_size=dense.voxel_size, origin=dense.origin)
        cpu = raster(sparse, FOLLOW)
        self.assertGreater(float(cpu[..., :3].sum()), 0)
        np.testing.assert_allclose(raster(sparse, FOLLOW, "gpu"), cpu, atol=.005, rtol=.005)


class PathTraceTests(unittest.TestCase):
    def trace(self, volume, settings, backend, samples=32, seed=5):
        return pathtrace.render(scene3d.Scene(volumes=(volume,)), scene3d.Camera(), 24, 24, settings=pathtrace.PathSettings(
            samples=samples, max_bounces=1, seed=seed), backend=backend, volume=settings)

    def test_the_path_tracer_follows_on_the_cpu(self):
        hot, cool = cooling_flame(1.0, 0.0), cooling_flame(0.5, 0.0)
        self.assertGreater(float(self.trace(hot, GLOW, "cpu")[..., :3].sum()), 0)
        np.testing.assert_array_equal(self.trace(hot, GLOW, "cpu"), self.trace(cool, GLOW, "cpu"))
        self.assertGreater(float(self.trace(hot, FOLLOW, "cpu")[..., :3].sum()),
                           1.5 * float(self.trace(cool, FOLLOW, "cpu")[..., :3].sum()))

    @unittest.skipUnless(gpu3d.available(), "no wgpu adapter")
    def test_the_card_matches_its_cpu_twin_with_and_without_smoke(self):
        if not gpupathtrace.soft_supported(gpu3d._state()):
            self.skipTest("no adapter cleared for smoke in the path tracer")
        for density in (0.0, 0.25):
            for settings in (GLOW, FOLLOW):
                volume = cooling_flame(1.0, density)
                cpu = self.trace(volume, settings, "cpu")
                self.assertGreater(float(cpu[..., :3].sum()), 0)
                np.testing.assert_allclose(self.trace(volume, settings, "gpu"), cpu, atol=.005, rtol=.005)

    @unittest.skipUnless(gpu3d.available(), "no wgpu adapter")
    def test_the_collision_glow_follows_too_with_two_volumes(self):
        """Several volumes keep the collision-summed glow (the integrated one is for a single volume): it follows the temperature as well."""
        if not gpupathtrace.soft_supported(gpu3d._state()):
            self.skipTest("no adapter cleared for smoke in the path tracer")
        a, b = cooling_flame(1.0, 0.25), cooling_flame(0.5, 0.25)
        b = replace(b, origin=tuple(np.asarray(b.origin) + (0.0, 0.0, 0.0)))
        scene = scene3d.Scene(volumes=(a, b))
        path = pathtrace.PathSettings(samples=256, max_bounces=1, seed=3)
        cpu = pathtrace.render(scene, scene3d.Camera(), 16, 16, settings=path, backend="cpu", volume=FOLLOW)
        gpu = pathtrace.render(scene, scene3d.Camera(), 16, 16, settings=path, backend="gpu", volume=FOLLOW)
        self.assertGreater(float(cpu[..., :3].sum()), 0)
        self.assertAlmostEqual(float(gpu[..., :3].mean()) / float(cpu[..., :3].mean()), 1.0, delta=0.06)


class TurbulenceLatticeTests(unittest.TestCase):
    """The shape turbulence blends two curl-noise draws that change only every `pulse_length` frames: they are kept, not rebuilt each
    substep (a 64-cell bake spent 30 of its 40 seconds there), and the values are the ones `particles.turbulence_field` gives."""

    def setUp(self):
        fluid3d._LATTICE_CACHE.clear()

    def test_the_kept_draw_equals_a_fresh_one_and_is_shared(self):
        shape = (12, 10, 8)
        xs, ys, zs = np.meshgrid(*(np.arange(n) + 0.5 for n in shape), indexing="ij")
        fresh = particles.turbulence_field(np.stack((xs, ys, zs), -1).reshape(-1, 3), "curl", 3.0, 2, 1013)
        first = fluid3d.shape_turbulence_lattice(shape, 3.0, 2, 1013)
        np.testing.assert_array_equal(first, fresh)
        self.assertIs(fluid3d.shape_turbulence_lattice(shape, 3.0, 2, 1013), first)
        self.assertFalse(first.flags.writeable)
        self.assertIsNot(fluid3d.shape_turbulence_lattice(shape, 3.0, 2, 1014), first, "another seed is another draw")
        self.assertIsNot(fluid3d.shape_turbulence_lattice(shape, 4.0, 2, 1013), first, "another swirl size too")

    def test_only_a_handful_are_kept_and_big_grids_are_not(self):
        for seed in range(fluid3d._LATTICE_CACHE_LIMIT + 3):
            fluid3d.shape_turbulence_lattice((4, 4, 4), 1.0, 1, seed)
        self.assertEqual(len(fluid3d._LATTICE_CACHE), fluid3d._LATTICE_CACHE_LIMIT)
        with mock.patch.object(fluid3d, "_LATTICE_CACHE_CELLS", 8):
            fluid3d._LATTICE_CACHE.clear()
            fluid3d.shape_turbulence_lattice((4, 4, 4), 1.0, 1, 5)
            self.assertEqual(len(fluid3d._LATTICE_CACHE), 0)

    def test_a_solve_is_bit_identical_with_and_without_the_kept_draws(self):
        from tests.test_fluid3d import ShapeControlTests

        def plume():
            fluid3d._LATTICE_CACHE.clear()
            return ShapeControlTests("test_each_control_at_zero_equals_off").plume(frames=6, turbulence=0.5, swirl_size=3.0, seed=7)[1]

        kept = plume()
        self.assertGreater(len(fluid3d._LATTICE_CACHE), 0, "the solve must have used the kept draws")
        with mock.patch.object(fluid3d, "_LATTICE_CACHE_CELLS", 0):
            rebuilt = plume()
            self.assertEqual(len(fluid3d._LATTICE_CACHE), 0)
        for name in ("density", "u", "v", "w"):
            np.testing.assert_array_equal(kept.arrays[name], rebuilt.arrays[name], name)

if __name__ == "__main__":
    unittest.main()
