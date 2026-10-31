"""Up-res reinjection (Lane 6, fire shot smoke up-res): every frame takes its flame, temperature, fuel and new smoke from the coarse
cache, and only the detail seeded into the smoke is carried along the guide."""
import tempfile
import unittest

import numpy as np

from nodebased import fluid_gpu_solver, fluid_upres, scene3d, simcache
from nodebased.imaging import Evaluator

N = 16
H = .25


def blob(cx, cy, cz, radius=3.0):
    x, y, z = np.ogrid[:N, :N, :N]
    return np.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (radius ** 2)).astype(np.float32)


def volume(density, flame=None, velocity_x=0.0):
    velocity = np.zeros((N, N, N, 3), np.float32)
    velocity[..., 0] = velocity_x
    flame = density * .3 if flame is None else flame
    return scene3d.Volume(density.astype(np.float32), voxel_size=H, origin=(-2, 0, -2), temperature=density * 2,
                          velocity=velocity, flame=flame.astype(np.float32), fuel=density * .5)


def mass(volume_, name="density"):
    return float(getattr(volume_, name).sum(dtype=np.float64)) * volume_.voxel_size ** 3


def dense(grid, origin):
    arrays = grid.to_dense()
    return scene3d.Volume(arrays["density"], voxel_size=H / 2, origin=origin, temperature=arrays.get("temperature"),
                          flame=arrays.get("flame"), fuel=arrays.get("fuel"), velocity=arrays.get("velocity"))


def frames(first, second, params, backend="cpu"):
    """Frame 1 from `first`, then frame 2 from `second` with the reinjecting pass; (fine frame 1, fine frame 2) as dense volumes."""
    params = {**fluid_upres.DEFAULTS, "upres_factor": "2", **params}
    a = fluid_upres.upres_sparse_grid(first, 2, backend, None, None, params, None, 1)
    prior = scene3d.Volume.from_sparse(a, voxel_size=H / 2, origin=first.origin)
    b = fluid_upres.upres_sparse_grid(second, 2, backend, second.velocity, prior, params, first, 2)
    return dense(a, first.origin), dense(b, first.origin)


class ReinjectionTests(unittest.TestCase):
    def test_flame_temperature_and_fuel_follow_the_coarse_cache_each_frame(self):
        first = volume(blob(8, 5, 8), velocity_x=.02)
        second_density = blob(8, 6, 8, 3.5)
        second = volume(second_density, flame=second_density * 1.2, velocity_x=.02)
        _, fine = frames(first, second, {})
        for name in ("flame", "temperature", "fuel"):
            with self.subTest(name=name):
                # Corner-aligned resampling shifts the sum a little; an unreinjected pass would sit at the first frame's value.
                self.assertLess(abs(mass(fine, name) / mass(second, name) - 1), .12)
        self.assertGreater(abs(mass(second, "flame") / mass(first, "flame") - 1), .3)

    def test_the_older_carry_only_pass_kept_the_first_frame_flame(self):
        first = volume(blob(8, 5, 8), velocity_x=.02)
        second_density = blob(8, 6, 8, 3.5)
        second = volume(second_density, flame=second_density * 1.2, velocity_x=.02)
        params = {**fluid_upres.DEFAULTS, "upres_factor": "2"}
        a = fluid_upres.upres_sparse_grid(first, 2, "cpu", None, None, params)
        prior = scene3d.Volume.from_sparse(a, voxel_size=H / 2, origin=first.origin)
        old = dense(fluid_upres.upres_sparse_grid(second, 2, "cpu", second.velocity, prior, params), first.origin)
        self.assertGreater(abs(mass(old, "flame") / mass(second, "flame") - 1), .3)

    def test_smoke_that_arrives_after_the_first_frame_is_kept_and_mass_matches(self):
        first = volume(blob(5, 8, 8))
        both = blob(5, 8, 8) + blob(11, 8, 8)
        second = volume(both)
        _, fine = frames(first, second, {})
        self.assertLess(abs(mass(fine) / mass(second) - 1), .01)
        right = fine.density[fine.density.shape[0] // 2:]
        left = fine.density[:fine.density.shape[0] // 2]
        self.assertGreater(float(right.sum()), .6 * float(left.sum()))

    def test_detail_is_seeded_into_new_smoke_and_not_reseeded_into_old_smoke(self):
        params = {"turbulence": 1.5, "swirl_size": 3.0, "grain": 2, "seed": 5}
        still = volume(blob(8, 8, 8))
        smooth_first, smooth_second = frames(still, still, {})
        detailed_first, detailed_second = frames(still, still, params)
        self.assertGreater(float(np.abs(detailed_first.density - smooth_first.density).mean()), 1e-4)
        # Nothing new arrives, so the second frame only carries the first frame's detail (zero guide): no flicker.
        np.testing.assert_allclose(detailed_second.density, detailed_first.density, rtol=5e-3, atol=2e-6)
        grown = volume(blob(8, 8, 8) + blob(12, 8, 8))
        _, with_new = frames(still, grown, params)
        _, smooth_new = frames(still, grown, {})
        right = slice(with_new.density.shape[0] * 3 // 4, None)
        self.assertGreater(float(np.abs(with_new.density[right] - smooth_new.density[right]).mean()), 1e-4)

    def test_detail_changes_from_frame_to_frame_in_the_smoke_that_arrives(self):
        params = {"turbulence": 1.5, "swirl_size": 3.0, "grain": 2, "seed": 5}
        first = volume(blob(4, 8, 8))
        second = volume(blob(4, 8, 8) + blob(12, 8, 8))
        third = volume(blob(4, 8, 8) + blob(12, 8, 8) + blob(8, 12, 8))
        paramset = {**fluid_upres.DEFAULTS, "upres_factor": "2", **params}
        grids, prior, earlier = [], None, None
        for index, source in enumerate((first, second, third), 1):
            grid = fluid_upres.upres_sparse_grid(source, 2, "cpu", source.velocity if prior is not None else None, prior,
                                                 paramset, earlier, index)
            prior = scene3d.Volume.from_sparse(grid, voxel_size=H / 2, origin=first.origin)
            earlier = source
            grids.append(dense(grid, first.origin))
        self.assertLess(abs(mass(grids[2]) / mass(third) - 1), .01)
        self.assertFalse(np.allclose(grids[1].density, grids[2].density, atol=1e-3))


@unittest.skipUnless(fluid_gpu_solver.available(), "no compute adapter")
class ReinjectionGpuTests(unittest.TestCase):
    def test_gpu_matches_the_cpu_for_a_reinjected_frame(self):
        first = volume(blob(8, 5, 8), velocity_x=.02)
        second_density = blob(8, 6, 8, 3.5) + blob(11, 9, 8)
        second = volume(second_density, flame=second_density * 1.2, velocity_x=.02)
        _, cpu = frames(first, second, {}, "cpu")
        _, gpu = frames(first, second, {}, "gpu")
        for name in ("density", "flame", "temperature", "fuel"):
            np.testing.assert_allclose(getattr(gpu, name), getattr(cpu, name), rtol=.01, atol=2e-4, err_msg=name)


class ReinjectionEvaluatorTests(unittest.TestCase):
    def test_the_evaluator_carries_the_coarse_flame_through_the_up_res_node(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "fire_shot"))
        try:
            import make_shot
        finally:
            sys.path.pop(0)
        document = make_shot.build_document(width=64, height=36, division_size=.125, pt_samples=4,
                                            smoke_upres=dict(upres_backend="cpu"))
        with tempfile.TemporaryDirectory() as root:
            evaluator = Evaluator(sim=simcache.SimCache(root=root, memory_budget=1 << 28, disk_budget=1 << 30))
            for frame in (2, 6):
                coarse = evaluator.evaluate_raster(document, "cache", frame=frame, typed=True)
                fine = evaluator.evaluate_raster(document, "upres", frame=frame, typed=True)
                self.assertEqual(fine.density.shape, tuple(2 * n for n in coarse.density.shape))
                self.assertLess(abs(mass(fine) / mass(coarse) - 1), .02)
                self.assertGreater(mass(coarse, "flame"), 0)
                self.assertLess(abs(mass(fine, "flame") / mass(coarse, "flame") - 1), .12)


if __name__ == "__main__":
    unittest.main()
