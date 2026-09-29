"""Pyro up-res behavior: scaling, mass, repeatability, output cache, registration and bypass."""
import tempfile
import unittest
from unittest import mock

import numpy as np

from nodebased import fluid_upres, scene3d, simcache
from nodebased import fluid_gpu_solver
from nodebased.core import Dispatcher, INPUT_TYPES, OUTPUT_TYPES, SPECS, bypass_slot
from nodebased.imaging import Evaluator
from nodebased.knobs import knob_layout
from nodebased.theme import COLORS
from nodebased.tiers import REGION_RULES
from tests.test_particles_nodes import make, wire


class FluidUpresTests(unittest.TestCase):
    def volume(self, n=12):
        x, y, z = np.meshgrid(*(np.arange(n, dtype=np.float32) for _ in range(3)), indexing="ij")
        d = np.exp(-((x - n * .48) ** 2 + (y - n * .52) ** 2 + (z - n * .5) ** 2) / (n * .09))
        v = np.zeros((*d.shape, 3), np.float32)
        return scene3d.Volume(d.astype(np.float32), voxel_size=.25, origin=(-1, 0, -1),
                              temperature=d * 2, velocity=v, flame=d * .3)

    def test_factor_one_without_turbulence_preserves_cached_channels(self):
        source = self.volume()
        out = fluid_upres.upres_volume(source, {**fluid_upres.DEFAULTS, "upres_factor": 1}, 1)
        np.testing.assert_array_equal(out.density, source.density)
        np.testing.assert_array_equal(out.temperature, source.temperature)
        np.testing.assert_array_equal(out.flame, source.flame)
        np.testing.assert_array_equal(out.velocity, source.velocity)

    def test_factor_two_conserves_integrated_density_mass(self):
        source = self.volume()
        out = fluid_upres.upres_volume(source, {**fluid_upres.DEFAULTS, "upres_factor": 2}, 1)
        before = float(source.density.sum(dtype=np.float64)) * source.voxel_size ** 3
        after = float(out.density.sum(dtype=np.float64)) * out.voxel_size ** 3
        self.assertLessEqual(abs(after - before) / before, .02)
        self.assertEqual(out.density.shape, (24, 24, 24))

    def test_turbulence_adds_high_frequency_detail_over_trilinear_sample(self):
        source = self.volume()
        base = fluid_upres.upres_volume(source, {**fluid_upres.DEFAULTS, "upres_factor": 2}, 3)
        detailed = fluid_upres.upres_volume(source, {**fluid_upres.DEFAULTS, "upres_factor": 2,
                                                     "turbulence": .8, "swirl_size": .3, "seed": 23}, 3)
        def hf(a):
            return sum(float(np.mean(np.diff(a, axis=i) ** 2)) for i in range(3))
        self.assertGreater(hf(detailed.density), hf(base.density))

    def test_deterministic_and_own_cache_round_trip(self):
        original = self.volume(8)
        source = scene3d.Volume(original.density, voxel_size=original.voxel_size,
                                temperature=original.temperature, velocity=original.velocity,
                                flame=original.flame, fuel=original.density * .25)
        params = {**fluid_upres.DEFAULTS, "upres_factor": 2, "turbulence": .4, "seed": 7}
        repeat_a = fluid_upres.upres_volume(source, params, 4)
        repeat_b = fluid_upres.upres_volume(source, params, 4)
        np.testing.assert_array_equal(repeat_a.density, repeat_b.density)
        with tempfile.TemporaryDirectory() as root:
            store = simcache.SimCache(root=root, memory_budget=16 << 20, disk_budget=32 << 20)
            a = fluid_upres.cached_upres(source, params, 4, store)
            b = fluid_upres.cached_upres(source, params, 4, store)
            np.testing.assert_array_equal(a.density, b.density)
            np.testing.assert_array_equal(a.fuel, b.fuel)
            reopened = simcache.SimCache(root=root, memory_budget=16 << 20, disk_budget=32 << 20)
            c = fluid_upres.cached_upres(source, params, 4, reopened)
            np.testing.assert_array_equal(a.density, c.density)
            np.testing.assert_array_equal(a.fuel, c.fuel)

    def test_registered_typed_and_bypass(self):
        kind = "FluidUpres3D"
        self.assertIn(kind, SPECS)
        self.assertEqual(OUTPUT_TYPES[kind], "volume")
        self.assertEqual(SPECS[kind]["inputs"], ["volume"])
        self.assertEqual(INPUT_TYPES["volume"], ("volume",))
        self.assertIn(kind, COLORS)
        self.assertIn(kind, REGION_RULES)
        laid_out = sorted(p for group in knob_layout(kind) for p in group.params)
        self.assertEqual(laid_out, sorted(SPECS[kind]["params"]))
        self.assertEqual(bypass_slot({"type": kind, "inputs": {"volume": "src"}}), "volume")

        d = Dispatcher()
        make(d, p=("Plume3D", {"plume_resolution": 8}), u=(kind, {"upres_factor": "2"}))
        wire(d, "u", "volume", "p")
        enabled = Evaluator().evaluate_raster(d.document, "u", frame=1, typed=True)
        self.assertEqual(enabled.density.shape, (16, 16, 16))
        before = Evaluator().evaluate_raster(d.document, "p", frame=1, typed=True)
        d.execute({"op": "disable", "id": "u", "value": True})
        after = Evaluator().evaluate_raster(d.document, "u", frame=1, typed=True)
        np.testing.assert_array_equal(after.density, before.density)
        self.assertEqual(after.voxel_size, before.voxel_size)

    def test_uses_the_fluid_cache_and_interpolates_the_velocity_guide(self):
        grid = {"division_size": .25, "bounds_min_x": -1., "bounds_min_y": 0., "bounds_min_z": -1.,
                "bounds_max_x": 1., "bounds_max_y": 1., "bounds_max_z": 1., "pressure": "cpu",
                "boundary_y": "closed", "cooling_rate": 0.}
        d = Dispatcher()
        make(d, src=("FluidSource3D", {"src_center_y": .25, "src_radius": .35}),
             sim=("FluidSolver3D", grid), cache=("FluidCache3D", {}),
             up=("FluidUpres3D", {"upres_factor": "2"}))
        wire(d, "sim", "fluid", "src")
        wire(d, "cache", "volume", "sim")
        wire(d, "up", "volume", "cache")
        evaluator = Evaluator()
        coarse = evaluator.evaluate_raster(d.document, "cache", frame=1, typed=True)
        fine = evaluator.evaluate_raster(d.document, "up", frame=1, typed=True)
        self.assertEqual(fine.density.shape, tuple(n * 2 for n in coarse.density.shape))
        before = float(coarse.density.sum(dtype=np.float64)) * coarse.voxel_size ** 3
        after = float(fine.density.sum(dtype=np.float64)) * fine.voxel_size ** 3
        self.assertLessEqual(abs(after - before) / max(before, 1e-20), .02)

    def test_sequential_re_simulation_keeps_density_and_carries_fuel(self):
        params = {**fluid_upres.DEFAULTS, "upres_factor": 2}
        source = self.volume(8)
        source = scene3d.Volume(source.density, voxel_size=source.voxel_size, temperature=source.temperature,
                                velocity=np.zeros((*source.shape, 3), np.float32), flame=source.flame,
                                fuel=source.density * .7)
        current = source
        for frame in range(1, 21):
            # A slightly changing guide stands in for the frame-interpolated cached velocity.
            guide = np.zeros((*source.shape, 3), np.float32)
            guide[..., 1] = np.float32(frame * .001)
            current = fluid_upres.upres_volume(source, params, frame, guide, current if frame > 1 else None)
            original = float(source.density.sum(dtype=np.float64)) * 8
            actual = float(current.density.sum(dtype=np.float64))
            self.assertLessEqual(abs(actual - original) / original, .02)
            self.assertIsNotNone(current.fuel)
        self.assertGreater(float(current.fuel.sum()), 0.0)

    def test_stationary_sequential_detail_has_no_frame_flicker(self):
        source = self.volume(8)
        params = {**fluid_upres.DEFAULTS, "upres_factor": 2, "turbulence": .7,
                  "swirl_size": .3, "seed": 19}
        first = fluid_upres.upres_volume(source, params, 1)
        second = fluid_upres.upres_volume(source, params, 2, previous=first)
        np.testing.assert_allclose(second.density, first.density, rtol=3e-3, atol=1e-9)


@unittest.skipUnless(fluid_gpu_solver.available(), "no compute adapter")
class FluidUpresGpuTests(unittest.TestCase):
    volume = FluidUpresTests.volume
    def test_factors_two_and_four_match_cpu_and_carry_fuel(self):
        source = self.volume(8)
        velocity = np.zeros((*source.shape, 3), np.float32)
        velocity[..., 0] = .025
        source = scene3d.Volume(source.density, voxel_size=source.voxel_size,
                                temperature=source.temperature, flame=source.flame,
                                fuel=source.density * .3, velocity=velocity)
        for factor in (2, 4):
            with self.subTest(factor=factor):
                params = {**fluid_upres.DEFAULTS, "upres_factor": factor}
                prior_cpu = prior_gpu = None
                for frame in range(1, 4):
                    cpu = fluid_upres.upres_volume(source, {**params, "upres_backend": "cpu"}, frame,
                                                   previous=prior_cpu)
                    gpu = fluid_upres.upres_volume(source, {**params, "upres_backend": "gpu"}, frame,
                                                   previous=prior_gpu)
                    for name in ("density", "temperature", "flame", "fuel", "velocity"):
                        np.testing.assert_allclose(getattr(gpu, name), getattr(cpu, name), rtol=.01, atol=2e-4)
                    before = float(source.density.sum(dtype=np.float64)) * source.voxel_size ** 3
                    after = float(gpu.density.sum(dtype=np.float64)) * gpu.voxel_size ** 3
                    self.assertLess(abs(after - before) / before, .01)
                    prior_cpu, prior_gpu = cpu, gpu

    def test_auto_prefers_gpu(self):
        from nodebased import fluid_upres_gpu
        source = self.volume(8)
        with mock.patch.object(fluid_upres_gpu, "reconstruct", wraps=fluid_upres_gpu.reconstruct) as call:
            fluid_upres.upres_volume(source, fluid_upres.DEFAULTS, 1)
        self.assertEqual(call.call_count, 1)

    def test_auto_falls_back_when_adapter_cannot_hold_grid(self):
        from nodebased import fluid_upres_gpu
        source = self.volume(8)
        with mock.patch.object(fluid_upres_gpu, "reconstruct", side_effect=fluid_gpu_solver.Unsupported("no card")):
            auto = fluid_upres.upres_volume(source, fluid_upres.DEFAULTS, 1)
            with self.assertRaises(fluid_gpu_solver.Unsupported):
                fluid_upres.upres_volume(source, {**fluid_upres.DEFAULTS, "upres_backend": "gpu"}, 1)
        cpu = fluid_upres.upres_volume(source, {**fluid_upres.DEFAULTS, "upres_backend": "cpu"}, 1)
        np.testing.assert_array_equal(auto.density, cpu.density)


if __name__ == "__main__":
    unittest.main()
