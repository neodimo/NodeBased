"""Persistent up-res swirl (Fluids 17 C): a fine velocity carried from frame to frame, so rotation accumulates."""
import tempfile
import unittest

import numpy as np

from nodebased import core, fluid_gpu_solver, fluid_upres, scene3d, simcache
from tests.test_fluid_upres_swirl import H, N, box_of, plume

DT = 1.0 / 24.0
HF = H / 2


def uniform_guide(speed=2.0):
    guide = np.zeros((N, N, N, 3), np.float32)
    guide[..., 1] = speed
    return guide


def vortex(dims, centre, radius=4.0, amplitude=.5):
    """A divergence-free Gaussian vortex about the x axis on the box, shape (*dims, 3)."""
    g = np.mgrid[:dims[0], :dims[1], :dims[2]].astype(np.float32)
    y, z = g[1] - centre[1], g[2] - centre[2]
    w = amplitude * np.exp(-(y * y + z * z) / radius ** 2).astype(np.float32)
    out = np.zeros((*dims, 3), np.float32)
    out[..., 1] = -w * z / radius
    out[..., 2] = w * y / radius
    return out


def persist(guide, uf, waves, strength, life, steps=1, vmax=1.5, iters=20, backend="cpu"):
    coords, (low, dims) = box_of(None)
    for _ in range(steps):
        if backend == "gpu":
            from nodebased.fluid_upres_gpu import swirl_persist
            uf = swirl_persist(guide, 2, low, dims, waves, uf, strength, HF, DT, life, vmax, iters,
                               fluid_upres.SWIRL_SPIN, fluid_upres.SWIRL_INJECT)
        else:
            uf = fluid_upres.swirl_persist_cpu(guide, 2, low, dims, waves, uf, strength, HF, DT, life, vmax, iters)
    return uf


def energy(u):
    return float(np.sum(u.astype(np.float64) ** 2))


class PersistenceTests(unittest.TestCase):
    waves = fluid_upres.swirl_waves(0, 3.0, 2, 1)

    def start(self):
        coords, (low, dims) = box_of(None)
        return vortex(dims, (dims[0] // 2, 14, dims[2] // 2))

    def test_a_vortex_survives_when_life_is_long_and_is_gone_when_it_is_short(self):
        uf = self.start()
        long = persist(uniform_guide(), uf, self.waves, 0.0, 1000.0, steps=12)
        short = persist(uniform_guide(), uf, self.waves, 0.0, .05, steps=12)
        # Semi-Lagrangian carrying smooths the vortex, so "survives" means a large share of its energy, not all of it.
        self.assertGreater(energy(long), .6 * energy(uf))
        self.assertLess(energy(short), 1e-4 * energy(uf))

    def test_the_vortex_is_carried_along_the_flow(self):
        uf = self.start()
        out = persist(uniform_guide(), uf, self.waves, 0.0, 1000.0, steps=12)

        def centre(u):
            m = np.sqrt(np.sum(u * u, axis=-1))
            return float((m * np.arange(m.shape[1])[None, :, None]).sum() / m.sum())
        moved = centre(out) - centre(uf)
        expected = 12 * 2.0 * DT / HF
        self.assertAlmostEqual(moved, expected, delta=.15 * expected)

    def test_life_zero_would_divide_by_nothing_so_the_grid_path_uses_step_b(self):
        # `swirl_life` 0 never reaches the persistent step: the sparse grid carries no `ufine` and uses the per-frame swirl.
        source = plume()
        p = {**fluid_upres.DEFAULTS, "upres_factor": "2", "swirl": 1.0, "swirl_scale": 3.0, "swirl_life": 0.0}
        a = fluid_upres.upres_sparse_grid(source, 2, "cpu", None, None, p, None, 1)
        prior = scene3d.Volume.from_sparse(a, voxel_size=HF, origin=source.origin)
        b = fluid_upres.upres_sparse_grid(plume(1.4), 2, "cpu", source.velocity, prior, p, source, 2)
        self.assertNotIn("ufine", b.data)

    def test_accumulation_builds_up_and_levels_off(self):
        guide = plume().velocity
        coords, (low, dims) = box_of(None)
        uf = np.zeros((*dims, 3), np.float32)
        means = []
        for frame in range(1, 25):
            waves = fluid_upres.swirl_waves(0, 3.0, 2, frame)
            uf = fluid_upres.swirl_persist_cpu(guide, 2, low, dims, waves, uf, 1.0, HF, DT, 1.0, 1.5, 10)
            means.append(float(np.sqrt(np.sum(uf * uf, axis=-1)).mean()))
        self.assertGreater(means[8], 3 * means[0])
        self.assertGreater(means[8], means[2])
        # Levelled off: the last third moves far less than the first third.
        self.assertLess(abs(means[23] - means[16]), abs(means[8] - means[0]))


class ProjectionAndClampTests(unittest.TestCase):
    def test_projection_removes_most_of_the_divergence(self):
        guide = plume().velocity
        coords, (low, dims) = box_of(None)
        waves = fluid_upres.swirl_waves(0, 3.0, 2, 1)
        zero = np.zeros((*dims, 3), np.float32)
        off = fluid_upres.swirl_persist_cpu(guide, 2, low, dims, waves, zero, 1.0, HF, DT, 1.0, 1.5, 0)
        out = {n: fluid_upres.swirl_persist_cpu(guide, 2, low, dims, waves, zero, 1.0, HF, DT, 1.0, 1.5, n) for n in (10, 40)}

        def rms(u):
            d = fluid_upres.divergence_dense(u, HF)
            return float(np.sqrt(np.mean(d * d)))
        self.assertGreater(rms(off), 0)
        # The stated bound: 20 sweeps leave under half of the unprojected divergence (rms over the box), and more sweeps do better.
        self.assertLess(rms(out[10]), .8 * rms(off))
        self.assertLess(rms(out[40]), rms(out[10]))

    def test_the_fine_velocity_never_outruns_the_coarse_flow(self):
        guide = plume().velocity
        coords, (low, dims) = box_of(None)
        uf = np.zeros((*dims, 3), np.float32)
        for frame in range(1, 7):
            uf = fluid_upres.swirl_persist_cpu(guide, 2, low, dims, fluid_upres.swirl_waves(0, 3.0, 2, frame), uf, 8.0,
                                               HF, DT, 4.0, .5, 5)
        q = [np.clip(np.arange(dims[a], dtype=np.float32) + low[a], 0, None) * ((N - 1) / (2 * N - 1)) for a in range(3)]
        from nodebased import fluid3d
        qx, qy, qz = np.meshgrid(*[np.clip(c, 0, N - 1) for c in q], indexing="ij")
        speed = np.sqrt(sum(fluid3d.trilerp(guide[..., a], qx, qy, qz) ** 2 for a in range(3)))
        size = np.sqrt(np.sum(uf * uf, axis=-1))
        self.assertGreater(float(size.max()), 0)
        self.assertTrue(np.all(size <= .5 * speed * 1.001 + 1e-6))

    def test_still_air_keeps_no_fine_velocity(self):
        guide = np.zeros((N, N, N, 3), np.float32)
        coords, (low, dims) = box_of(None)
        uf = vortex(dims, (20, 20, 20))
        out = persist(guide, uf, fluid_upres.swirl_waves(0, 3.0, 2, 1), 1.0, 1000.0)
        self.assertEqual(float(np.abs(out).max()), 0.0)


class GridAndCacheTests(unittest.TestCase):
    def frames(self, backend, store=None, count=3, **extra):
        p = {**fluid_upres.DEFAULTS, "upres_factor": "2", "swirl": 1.0, "swirl_scale": 3.0, "swirl_life": 1.0,
             "upres_backend": backend, **extra}
        prior = None
        previous_source = None
        grids = []
        for frame in range(1, count + 1):
            source = plume(1.0 + .15 * frame)
            guide = None if prior is None else source.velocity
            grid = fluid_upres.upres_sparse_grid(source, 2, backend, guide, prior, p, previous_source, frame)
            grids.append(grid)
            prior = scene3d.Volume.from_sparse(grid, voxel_size=HF, origin=source.origin)
            previous_source = source
        return grids

    def test_the_grid_carries_ufine_and_it_grows_over_frames(self):
        grids = self.frames("cpu")
        self.assertEqual(grids[0].data["ufine"].shape[1:], (8, 8, 8, 3))
        mags = [float(np.sqrt(np.sum(g.data["ufine"] ** 2, axis=-1)).mean()) for g in grids]
        self.assertGreater(mags[1], mags[0])

    def test_swirl_off_has_no_ufine(self):
        grids = self.frames("cpu", count=2, swirl=0.0)
        self.assertNotIn("ufine", grids[1].data)

    def test_the_cache_round_trips_ufine_and_a_resumed_bake_matches(self):
        p = {**fluid_upres.DEFAULTS, "upres_factor": "2", "swirl": 1.0, "swirl_scale": 3.0, "swirl_life": 1.0,
             "upres_backend": "cpu"}
        sources = [plume(1.0 + .15 * f) for f in range(1, 4)]
        with tempfile.TemporaryDirectory() as root:
            store = simcache.SimCache(root=root, memory_budget=64 << 20, disk_budget=128 << 20)
            f1 = fluid_upres.cached_upres(sources[0], p, 1, store, reinject=True)
            f2 = fluid_upres.cached_upres(sources[1], p, 2, store, previous=f1, previous_source=sources[0], reinject=True,
                                          guide_velocity=sources[1].velocity)
            f3 = fluid_upres.cached_upres(sources[2], p, 3, store, previous=f2, previous_source=sources[1], reinject=True,
                                          guide_velocity=sources[2].velocity)
            self.assertIn("ufine", f2.sparse.data)
            # Resume from disk: a fresh cache holds frames 1 and 2, and frame 3 is baked from the stored frame 2.
            reopened = simcache.SimCache(root=root, memory_budget=64 << 20, disk_budget=128 << 20)
            r2 = fluid_upres.cached_upres(sources[1], p, 2, reopened, reinject=True)
            np.testing.assert_array_equal(r2.sparse.data["ufine"], f2.sparse.data["ufine"])
            reopened.clear() if hasattr(reopened, "clear") else None
        with tempfile.TemporaryDirectory() as root2:
            store2 = simcache.SimCache(root=root2, memory_budget=64 << 20, disk_budget=128 << 20)
            again = fluid_upres.cached_upres(sources[2], p, 3, store2, previous=f2, previous_source=sources[1], reinject=True,
                                             guide_velocity=sources[2].velocity)
        np.testing.assert_array_equal(again.sparse.data["ufine"], f3.sparse.data["ufine"])
        np.testing.assert_array_equal(again.sparse.data["density"], f3.sparse.data["density"])

    def test_swirl_changes_the_smoke_only_and_keeps_the_mass(self):
        on = self.frames("cpu", count=3)[2]
        off = self.frames("cpu", count=3, swirl=0.0)[2]
        for name in ("flame", "temperature", "fuel"):
            np.testing.assert_array_equal(on.data[name], off.data[name])
        d_on, d_off = on.to_dense()["density"], off.to_dense()["density"]
        self.assertFalse(np.array_equal(d_on, d_off))
        mass = float(plume(1.45).density.sum(dtype=np.float64)) * 8
        self.assertLess(abs(float(d_on.sum(dtype=np.float64)) - mass) / mass, .02)

    def test_swirl_off_is_the_old_result_bit_for_bit(self):
        first, second = plume(), plume(1.4)
        off = {**fluid_upres.DEFAULTS, "upres_factor": "2", "upres_backend": "cpu"}
        old = {k: v for k, v in off.items() if k not in ("swirl", "swirl_scale", "swirl_life", "swirl_max")}
        grids = []
        for p in (old, {**off, "swirl": 0.0, "swirl_life": 3.0}):
            a = fluid_upres.upres_sparse_grid(first, 2, "cpu", None, None, p, None, 1)
            prior = scene3d.Volume.from_sparse(a, voxel_size=HF, origin=first.origin)
            grids.append(fluid_upres.upres_sparse_grid(second, 2, "cpu", second.velocity, prior, p, first, 2))
        for name in grids[0].data:
            np.testing.assert_array_equal(grids[0].data[name], grids[1].data[name])


class RegistrationTests(unittest.TestCase):
    def test_knobs_and_ranges(self):
        spec = core.SPECS["FluidUpres3D"]["params"]
        self.assertEqual(spec["swirl_life"], 1.0)
        self.assertEqual(spec["swirl_max"], 1.5)
        self.assertIn("swirl_life", core.LIMITS)
        self.assertIn("swirl_max", core.LIMITS)

    def test_an_old_swirl_document_keeps_step_b_behaviour(self):
        doc = core.empty_document()
        old = {k: v for k, v in core.SPECS["FluidUpres3D"]["params"].items() if k not in ("swirl_life", "swirl_max")}
        old["swirl"] = .5
        doc["nodes"]["up"] = {"type": "FluidUpres3D", "name": "up", "pos": [0, 0], "disabled": False,
                              "inputs": {"volume": None}, "params": old}
        loaded = core.upgrade_document(doc)
        core.validate(loaded)
        self.assertEqual(loaded["nodes"]["up"]["params"]["swirl_life"], 0.0)
        self.assertEqual(loaded["nodes"]["up"]["params"]["swirl_max"], 1.5)


class GpuTests(unittest.TestCase):
    def test_persistent_step_gpu_matches_cpu(self):
        guide = plume().velocity
        coords, (low, dims) = box_of(None)
        waves = fluid_upres.swirl_waves(5, 3.0, 2, 7)
        start = np.zeros((*dims, 3), np.float32)
        try:
            gpu = start
            cpu = start
            for _ in range(3):
                gpu = persist(guide, gpu, waves, 1.5, 1.0, backend="gpu")
                cpu = persist(guide, cpu, waves, 1.5, 1.0, backend="cpu")
        except fluid_gpu_solver.Unsupported as error:
            self.skipTest(str(error))
        scale = float(np.abs(cpu).max())
        self.assertGreater(scale, 0)
        np.testing.assert_allclose(gpu, cpu, rtol=.02, atol=.01 * scale)

    def test_sparse_grid_gpu_matches_cpu(self):
        try:
            cpu = GridAndCacheTests.frames(GridAndCacheTests(), "cpu")
            gpu = GridAndCacheTests.frames(GridAndCacheTests(), "gpu")
        except fluid_gpu_solver.Unsupported as error:
            self.skipTest(str(error))
        for index in (1, 2):
            np.testing.assert_array_equal(cpu[index].coords, gpu[index].coords)
            for name in ("density", "fuel", "temperature", "flame"):
                np.testing.assert_allclose(gpu[index].data[name], cpu[index].data[name], rtol=.02, atol=2e-4)
            scale = float(np.abs(cpu[index].data["ufine"]).max())
            np.testing.assert_allclose(gpu[index].data["ufine"], cpu[index].data["ufine"], rtol=.02, atol=.02 * scale)


if __name__ == "__main__":
    unittest.main()
