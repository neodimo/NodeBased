"""Lane L6 step D: the GPU-resident 3D solver (nodebased/fluid_gpu_solver.py) against the CPU reference in fluid3d.

Every test here needs a wgpu adapter and skips without one; the first test prints which one produced the numbers.
"""
import logging
import threading
import unittest

import numpy as np

from nodebased import fluid3d, gpu3d, simcache
from nodebased.cancellation import Cancelled
from tests.test_fluid3d import box_collider, solve

try:
    from nodebased import fluid_gpu_solver as fgs
    HAVE_GPU = fgs.available()
except Exception:                                          # pragma: no cover - no wgpu at all
    fgs, HAVE_GPU = None, False

SMALL = {"nx": 32, "ny": 32, "nz": 32}
OPEN_TOP = {"nx": 24, "ny": 40, "nz": 24, "boundary_x": "open", "boundary_y": "open", "boundary_z": "open"}


def solve_gpu(params, frames, cache=None, cancel=None, sparse=False, **kwargs):
    solver = fgs.GpuSmoke3D(params, cancel=cancel, sparse=sparse, **kwargs)
    cache = cache or simcache.SimCache(enabled=False)
    run = simcache.run_key(None, params)
    state = simcache.solve_to_frame(cache, run, frames, 1, solver.substeps, 0, solver.initial_state, solver.step, cancel)
    return state, solver


def relative_error(a, b):
    return float(np.abs(a - b).max() / max(float(np.abs(a).max()), 1e-6))


@unittest.skipUnless(HAVE_GPU, "no wgpu compute adapter")
class GpuFluidBase(unittest.TestCase):
    pass


class AdapterReport(unittest.TestCase):
    def test_the_log_names_the_device(self):
        if not HAVE_GPU:
            self.skipTest("no wgpu compute adapter")
        report = gpu3d.adapter_report()
        print("\n" + report, flush=True)
        with self.assertLogs("nodebased.fluid_gpu_solver", level="INFO") as logs:
            solver = fgs.GpuSmoke3D(dict(SMALL))
        self.assertIn(solver.adapter_name, "\n".join(logs.output))
        self.assertIn(solver.adapter_name, report)
        self.assertIn(fgs.adapter_name(), report)


class Multigrid(GpuFluidBase):
    """Part 1: the multigrid pressure solve."""

    def rhs_for(self, shape, solid, open_axes, seed=3):
        rng = np.random.default_rng(seed)
        system = fluid3d.Poisson3D(shape, solid, open_axes)
        rhs = rng.standard_normal(shape)
        fluid = np.ones(shape, bool) if solid is None else ~solid
        rhs[~fluid] = 0.0
        if system.singular:
            rhs -= rhs[fluid].mean()
            rhs[~fluid] = 0.0
        return system, rhs

    def test_residual_falls_below_the_tolerance_and_matches_the_cpu_operator(self):
        solid = np.zeros((32, 48, 32), bool)
        solid[10:18, 10:20, 10:20] = True
        for shape, sol, open_axes in [((32, 48, 32), None, (False, True, False)), ((32, 48, 32), solid, (False, True, False)),
                                      ((30, 45, 33), None, (False, False, False)), ((30, 45, 33), None, (True, True, True))]:
            system, rhs = self.rhs_for(shape, sol, open_axes)
            solver = fgs.GpuMultigrid3D()
            q, cycles, residual = solver.solve(rhs, np.zeros(shape), 1e-3, 100, None, system)
            true = rhs - system.apply(q, np.empty(shape))
            if sol is not None:
                true[sol] = 0.0
            self.assertLessEqual(residual, 1e-3, (shape, open_axes))
            self.assertLessEqual(float(np.abs(true).max()), 1.2e-3, (shape, open_axes))     # the float64 residual agrees
            self.assertLessEqual(cycles, 12, "multigrid should need a handful of cycles, conjugate gradient needs hundreds")

    def test_parity_with_the_cpu_conjugate_gradient(self):
        shape = (32, 48, 32)
        system, rhs = self.rhs_for(shape, None, (False, True, False))
        q, _, _ = fgs.GpuMultigrid3D().solve(rhs, np.zeros(shape), 1e-4, 200, None, system)
        cg, _, _ = fluid3d.conjugate_gradient(rhs, np.zeros(shape), 1e-4, 1500, None, system)
        self.assertLess(float(np.abs(q - cg).max()) / float(np.abs(cg).max()), 2e-3)

    def test_same_cycle_count_gives_a_bit_identical_re_solve(self):
        shape = (32, 32, 32)
        system, rhs = self.rhs_for(shape, None, (False, False, False))
        solver = fgs.GpuMultigrid3D()
        first, cycles, _ = solver.solve(rhs, np.zeros(shape), 1e-3, 100, None, system)
        self.assertGreater(solver.cycles, 0)
        again, cycles2, _ = solver.solve(rhs, np.zeros(shape), 1e-3, 100, None, system)
        self.assertEqual(cycles, cycles2)
        np.testing.assert_array_equal(first, again)
        fresh = fgs.GpuMultigrid3D()
        fresh.cycles = solver.cycles
        third, _, _ = fresh.solve(rhs, np.zeros(shape), 1e-3, 100, None, system)
        np.testing.assert_array_equal(first, third)

    def test_plugs_into_the_cpu_solver_as_its_pressure_hook(self):
        params = dict(SMALL)
        cpu, _ = solve(params, 4)
        hooked, solver = solve(params, 4, pressure_solver=fgs.GpuMultigrid3D().solve)
        self.assertLess(relative_error(cpu.arrays["density"], hooked.arrays["density"]), 5e-3)
        self.assertLess(float(np.abs(divergence(hooked)).max()), 2e-3)

    def test_cancellation_stops_the_solve(self):
        shape = (32, 32, 32)
        system, rhs = self.rhs_for(shape, None, (False, False, False))
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(Cancelled):
            fgs.GpuMultigrid3D().solve(rhs, np.zeros(shape), 1e-9, 100, cancel, system)


def divergence(state):
    a = state.arrays
    return fluid3d.divergence(a["u"].astype(np.float64), a["v"].astype(np.float64), a["w"].astype(np.float64))


if __name__ == "__main__":
    unittest.main()
