"""Up-res swirl (Fluids 17 B): divergence-free curl noise and vorticity confinement on the fine grid, gated by the coarse speed."""
import unittest

import numpy as np

from nodebased import core, fluid_gpu_solver, fluid_upres, scene3d

N = 24
H = .1


def plume(speed=1.2, shear=True):
    x, y, z = np.mgrid[:N, :N, :N].astype(np.float32)
    density = np.exp(-(((x - 12) ** 2 + (y - 10) ** 2 + (z - 12) ** 2) / 30)).astype(np.float32)
    velocity = np.zeros((N, N, N, 3), np.float32)
    velocity[..., 1] = speed * density + .1
    if shear:
        velocity[..., 0] = .4 * np.sin(y / 4)
    return scene3d.Volume(density, voxel_size=H, velocity=velocity, fuel=density * .3, temperature=density * 2,
                          flame=density * .2)


def params(**extra):
    return {**fluid_upres.DEFAULTS, "upres_factor": "2", "swirl": 1.0, "swirl_scale": 3.0, **extra}


def box_of(source, factor=2):
    coords = np.argwhere(np.ones((N * factor // 8,) * 3, bool)).astype(np.int32)
    return coords, fluid_upres.swirl_box(coords)


class SwirlMathTests(unittest.TestCase):
    def test_wave_noise_is_divergence_free_and_unit_rms(self):
        K, C, phase = fluid_upres.swirl_waves(3, 4.0, 2, 5)
        self.assertEqual(K.shape, (fluid_upres.SWIRL_WAVES, 3))
        np.testing.assert_allclose(np.sum(K * C, axis=1), 0, atol=1e-6)          # each wave is perpendicular to its direction
        np.testing.assert_allclose(np.sum(np.sum(C * C, axis=1)) / 2, 1, rtol=1e-5)

    def test_waves_are_deterministic_by_seed_and_change_with_frame(self):
        a = fluid_upres.swirl_waves(1, 3.0, 2, 4)
        b = fluid_upres.swirl_waves(1, 3.0, 2, 4)
        c = fluid_upres.swirl_waves(2, 3.0, 2, 4)
        d = fluid_upres.swirl_waves(1, 3.0, 2, 5)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        self.assertFalse(np.array_equal(a[0], c[0]))
        np.testing.assert_array_equal(a[0], d[0])
        self.assertFalse(np.array_equal(a[2], d[2]))

    def test_still_air_gets_no_swirl(self):
        source = plume()
        still = np.zeros_like(source.velocity)
        coords, (low, dims) = box_of(source)
        out = fluid_upres.swirl_dense_cpu(still, 2, low, dims, fluid_upres.swirl_waves(0, 3.0, 2, 1), 2.0, H / 2)
        self.assertEqual(float(np.abs(out).max()), 0.0)

    def test_swirl_scales_with_the_coarse_speed(self):
        coords, (low, dims) = box_of(None)
        waves = fluid_upres.swirl_waves(0, 3.0, 2, 1)
        velocity = plume().velocity
        one = fluid_upres.swirl_dense_cpu(velocity, 2, low, dims, waves, 1.0, H / 2)
        three = fluid_upres.swirl_dense_cpu(velocity * 3, 2, low, dims, waves, 1.0, H / 2)
        np.testing.assert_allclose(three, 3 * one, rtol=2e-3, atol=2e-4)
        self.assertGreater(float(np.abs(one).max()), .1)

    def test_added_velocity_is_limited_by_the_coarse_speed(self):
        coords, (low, dims) = box_of(None)
        velocity = plume().velocity
        out = fluid_upres.swirl_dense_cpu(velocity, 2, low, dims, fluid_upres.swirl_waves(0, 3.0, 2, 1), 6.0, H / 2)
        # Sampled at the coarse nodes, where the guide is exact.
        for cell in ((12, 10, 12), (6, 8, 9), (16, 12, 10)):
            fine = [int(round(c * (N * 2 - 1) / (N - 1))) for c in cell]
            at = tuple(f - int(l) for f, l in zip(fine, low))
            speed = float(np.linalg.norm(velocity[cell]))
            self.assertLessEqual(float(np.linalg.norm(out[at])), fluid_upres.SWIRL_LIMIT * speed * 1.05 + 1e-5)

    def test_swirl_has_curl_where_the_smoke_moves(self):
        coords, (low, dims) = box_of(None)
        with_swirl = fluid_upres.swirl_dense_cpu(plume().velocity, 2, low, dims, fluid_upres.swirl_waves(0, 3.0, 2, 1), 1.0, H / 2)

        def curl(u):
            gx, gy, gz = (np.gradient(u[..., a], H / 2, axis=ax) for a, ax in ((2, 1), (0, 2), (1, 0)))
            return np.abs(gx).mean()
        self.assertGreater(curl(with_swirl), 0)


class SwirlGridTests(unittest.TestCase):
    def two_frames(self, backend, **extra):
        first, second = plume(), plume(1.4)
        p = params(upres_backend=backend, **extra)
        a = fluid_upres.upres_sparse_grid(first, 2, backend, None, None, p, None, 1)
        prior = scene3d.Volume.from_sparse(a, voxel_size=H / 2, origin=first.origin)
        b = fluid_upres.upres_sparse_grid(second, 2, backend, second.velocity, prior, p, first, 2)
        return first, second, a, b

    def test_swirl_off_is_the_old_result_bit_for_bit(self):
        first, second = plume(), plume(1.4)
        off = {**fluid_upres.DEFAULTS, "upres_factor": "2", "upres_backend": "cpu"}
        old = {key: value for key, value in off.items() if key not in ("swirl", "swirl_scale")}
        grids = []
        for p in (old, {**off, "swirl": 0.0}):
            a = fluid_upres.upres_sparse_grid(first, 2, "cpu", None, None, p, None, 1)
            prior = scene3d.Volume.from_sparse(a, voxel_size=H / 2, origin=first.origin)
            grids.append(fluid_upres.upres_sparse_grid(second, 2, "cpu", second.velocity, prior, p, first, 2))
        np.testing.assert_array_equal(grids[0].coords, grids[1].coords)
        for name in grids[0].data:
            np.testing.assert_array_equal(grids[0].data[name], grids[1].data[name])

    def test_swirl_moves_the_smoke_but_keeps_the_flame_and_mass(self):
        _, second, _, off = self.two_frames("cpu", swirl=0.0)
        _, _, _, on = self.two_frames("cpu", swirl=1.0)
        d_off, d_on = off.to_dense()["density"], on.to_dense()["density"]
        self.assertTrue(np.all(np.isfinite(d_on)))
        self.assertGreaterEqual(float(d_on.min()), 0.0)
        self.assertGreater(float(np.abs(d_on - d_off).max()), 1e-3 * float(d_off.max()))
        mass = float(second.density.sum(dtype=np.float64)) * 8
        self.assertLess(abs(float(d_on.sum(dtype=np.float64)) - mass) / mass, .02)

    def test_swirl_is_deterministic(self):
        a = self.two_frames("cpu")[3].to_dense()["density"]
        b = self.two_frames("cpu")[3].to_dense()["density"]
        np.testing.assert_array_equal(a, b)

    def test_still_smoke_is_not_stirred(self):
        still = plume()
        still = scene3d.Volume(still.density, voxel_size=H, velocity=np.zeros_like(still.velocity), fuel=still.fuel,
                               temperature=still.temperature, flame=still.flame)
        out = []
        for swirl in (0.0, 2.0):
            p = params(swirl=swirl, upres_backend="cpu")
            a = fluid_upres.upres_sparse_grid(still, 2, "cpu", None, None, p, None, 1)
            prior = scene3d.Volume.from_sparse(a, voxel_size=H / 2, origin=still.origin)
            out.append(fluid_upres.upres_sparse_grid(still, 2, "cpu", still.velocity, prior, p, still, 2).to_dense()["density"])
        np.testing.assert_allclose(out[0], out[1], atol=1e-6)


class SwirlRegistrationTests(unittest.TestCase):
    def test_knobs_default_off_and_ranges(self):
        params_ = core.SPECS["FluidUpres3D"]["params"]
        self.assertEqual(params_["swirl"], 0.0)
        self.assertEqual(params_["swirl_scale"], 3.0)
        self.assertIn("swirl", core.LIMITS)
        self.assertIn("swirl_scale", core.LIMITS)

    def test_an_old_document_loads_with_swirl_off(self):
        doc = core.empty_document()
        old = {key: value for key, value in core.SPECS["FluidUpres3D"]["params"].items() if key not in ("swirl", "swirl_scale")}
        doc["nodes"]["up"] = {"type": "FluidUpres3D", "name": "up", "pos": [0, 0], "disabled": False,
                              "inputs": {"volume": None}, "params": old}
        loaded = core.upgrade_document(doc)
        core.validate(loaded)
        self.assertEqual(loaded["nodes"]["up"]["params"]["swirl"], 0.0)
        self.assertEqual(loaded["nodes"]["up"]["params"]["swirl_scale"], 3.0)


class SwirlGpuTests(unittest.TestCase):
    def test_dense_swirl_gpu_matches_cpu(self):
        from nodebased.fluid_upres_gpu import swirl_dense
        coords, (low, dims) = box_of(None)
        velocity = plume().velocity
        waves = fluid_upres.swirl_waves(5, 3.0, 2, 7)
        try:
            gpu = swirl_dense(velocity, 2, low, dims, waves, 1.5, fluid_upres.SWIRL_CONFINE * 1.5, fluid_upres.SWIRL_LIMIT, H / 2)
        except fluid_gpu_solver.Unsupported as error:
            self.skipTest(str(error))
        cpu = fluid_upres.swirl_dense_cpu(velocity, 2, low, dims, waves, 1.5, H / 2)
        np.testing.assert_allclose(gpu, cpu, rtol=.01, atol=2e-3)

    def test_sparse_grid_gpu_matches_cpu_with_swirl(self):
        try:
            cpu = SwirlGridTests.two_frames(self, "cpu")
            gpu = SwirlGridTests.two_frames(self, "gpu")
        except fluid_gpu_solver.Unsupported as error:
            self.skipTest(str(error))
        for index in (2, 3):
            np.testing.assert_array_equal(cpu[index].coords, gpu[index].coords)
            for name in ("density", "fuel", "temperature", "flame"):
                np.testing.assert_allclose(gpu[index].data[name], cpu[index].data[name], rtol=.01, atol=2e-4)


if __name__ == "__main__":
    unittest.main()
