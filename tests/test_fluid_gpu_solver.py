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


class ResidentSubstep(GpuFluidBase):
    """Part 2: advection, buoyancy, vorticity, fire and boundaries in compute shaders, parity with the CPU reference."""

    def assert_parity(self, params, frames, tolerance=1e-2, names=("u", "v", "w", "density", "temperature", "fuel", "burn"),
                      **kwargs):
        cpu, _ = solve(params, frames, **{k: v() for k, v in kwargs.items()})
        gpu, _ = solve_gpu(params, frames, **{k: v() for k, v in kwargs.items()})
        for name in names:
            err = relative_error(cpu.arrays[name], gpu.arrays[name])
            self.assertLess(err, tolerance, f"{name}: relative error {err:.3g}")
        mass_cpu, mass_gpu = float(cpu.arrays["density"].sum()), float(gpu.arrays["density"].sum())
        self.assertAlmostEqual(mass_gpu / max(mass_cpu, 1e-9), 1.0, delta=2e-3)
        return cpu, gpu

    def test_32_cubed_plume_for_10_frames_matches_the_cpu_solver(self):
        cpu, gpu = self.assert_parity(dict(SMALL, vorticity=0.3), 10)
        self.assertGreater(float(np.abs(gpu.arrays["v"]).max()), 1.0)             # the plume really rose
        self.assertLess(float(np.abs(divergence(gpu)).max()), 2e-3)
        self.assertEqual(gpu.meta["substep_count"], 10)

    def test_open_boundaries_and_semi_lagrangian_substeps(self):
        self.assert_parity(dict(OPEN_TOP, advection="semi_lagrangian", substeps=2), 5)

    def test_fire_with_expansion_ambient_temperature_and_decay(self):
        params = dict(nx=24, ny=40, nz=28, fire=1, source_fuel=1.0, burn_expansion=0.4, dissipation=0.05,
                      cooling_rate=0.1, ambient_temperature=0.3, boundary_z="open")
        cpu, gpu = self.assert_parity(params, 8)
        self.assertGreater(float(gpu.arrays["burn"].max()), 0.1)                  # flames exist to compare

    def test_a_solid_collider_and_the_force_list(self):
        forces = lambda: [fluid3d.Force("wind", {"dir_x": 1, "dir_y": 0, "dir_z": 0, "strength": 0.2}),
                          fluid3d.Force("drag", {"drag": 0.1}),
                          fluid3d.Force("gravity", {"dir_x": 0, "dir_y": -1, "dir_z": 0, "strength": 0.3}),
                          fluid3d.Force("turbulence", {"strength": 0.3, "turbulence_scale": 6.0, "turbulence_speed": 0.5}, seed=3)]
        colliders = lambda: [box_collider((10, 14, 10), (20, 20, 20))]
        params = dict(nx=32, ny=40, nz=32, boundary_x="open")
        cpu, gpu = self.assert_parity(params, 6, forces=forces, colliders=colliders)
        solid = fluid3d.fill_interior(fluid3d.voxelize_surface(fluid3d.np.asarray(box_collider((10, 14, 10), (20, 20, 20)).track.at(1)),
                                                              (32, 40, 32)))
        self.assertEqual(float(np.abs(gpu.arrays["density"][solid]).max()), 0.0)   # no smoke inside the solid

    def test_the_first_frame_fixes_the_cycle_count_and_records_it(self):
        state, solver = solve_gpu(dict(SMALL), 3)
        self.assertGreater(state.meta["mg_cycles"], 0)
        self.assertLessEqual(state.meta["mg_cycles"], 12)
        self.assertLessEqual(state.meta["cg_residual"], 1e-3)

    def test_same_inputs_are_bit_identical_and_a_checkpoint_resumes_exactly(self):
        params = dict(SMALL, substeps=2)
        a, _ = solve_gpu(params, 6)
        b, _ = solve_gpu(params, 6)
        self.assertEqual(a, b)                                                     # two solvers, same bits
        cache = simcache.SimCache()
        solve_gpu(params, 3, cache=cache)                                          # frames 1 to 3 checkpointed
        resumed, solver = solve_gpu(params, 6, cache=cache)                        # a new solver resumes from the cache
        self.assertEqual(resumed, a)
        self.assertEqual(resumed.meta["mg_cycles"], a.meta["mg_cycles"])

    def test_cancellation_between_substeps(self):
        cancel = threading.Event()
        solver = fgs.GpuSmoke3D(dict(SMALL), cancel=cancel)
        state = solver.initial_state(0)
        state = solver.step(state, 1, 0, 0)
        state.arrays                                                               # read it back before the next step
        cancel.set()
        with self.assertRaises(Cancelled):
            solver.step(state, 1, 1, 0)

    def test_no_per_substep_readback_beyond_the_residual(self):
        solver = fgs.GpuSmoke3D(dict(SMALL))
        state = solver.initial_state(0)
        state = solver.step(state, 1, 0, 0)                                        # uploads and calibrates
        before = dict(fgs.STATS)
        for substep in range(1, 4):
            state = solver.step(state, 1, substep, 0)
        reads = fgs.STATS["readbacks"] - before["readbacks"]
        self.assertLessEqual(reads, 3 * 3, "one residual read per substep, plus at most a few for a cycle top-up")
        self.assertEqual(reads >= 3, True)


class FallbackAndBudget(unittest.TestCase):
    def test_a_grid_over_the_budget_is_unsupported_and_falls_back_to_the_cpu(self):
        if not HAVE_GPU:
            self.skipTest("no wgpu compute adapter")
        with self.assertRaises(fgs.Unsupported):
            fgs.GpuSmoke3D(dict(SMALL), memory_budget=1 << 20)
        ok, reason = fgs.fits((32, 32, 32), memory_budget=1 << 20)
        self.assertFalse(ok)
        self.assertIn("budget", reason)
        with self.assertLogs("nodebased.fluid_gpu_solver", level="WARNING"):
            solver = fgs.create_solver(dict(SMALL), memory_budget=1 << 20)
        self.assertIsInstance(solver, fluid3d.Smoke3D)
        self.assertNotIsInstance(solver, fgs.GpuSmoke3D)
        self.assertIn("budget", solver.fallback_reason)
        state, _ = solve(dict(SMALL), 2)                                           # and it solves
        self.assertGreater(float(state.arrays["density"].sum()), 0.0)

    def test_no_compute_adapter_is_unsupported(self):
        saved = dict(fgs._CTX) if fgs else {}
        try:
            fgs._CTX.clear()
            import unittest.mock as mock
            with mock.patch("nodebased.gpu3d._state", side_effect=RuntimeError("no adapter here")):
                self.assertFalse(fgs.available())
                with self.assertRaises(fgs.Unsupported):
                    fgs.GpuSmoke3D(dict(SMALL))
                with self.assertLogs("nodebased.fluid_gpu_solver", level="WARNING"):
                    solver = fgs.create_solver(dict(SMALL))
                self.assertIsInstance(solver, fluid3d.Smoke3D)
                self.assertIn("no adapter here", solver.fallback_reason)
                self.assertFalse(fgs.fits((32, 32, 32))[0])
        finally:
            fgs._CTX.clear()
            fgs._CTX.update(saved)


class NodeBackend(GpuFluidBase):
    """The FluidSolver3D `pressure` knob reaches the resident solver, and auto falls back when it cannot run."""

    def test_pressure_resident_runs_through_the_graph(self):
        from tests.test_fluid3d_nodes import at, plume
        from nodebased.imaging import Evaluator
        results = {}
        for choice in ("cpu", "resident"):
            d = plume(pressure=choice)
            results[choice] = at(Evaluator(), d, "c", 4).density
        self.assertLess(relative_error(results["cpu"], results["resident"]), 1e-2)

    def test_resident_is_refused_with_the_reason_when_it_cannot_run(self):
        import unittest.mock as mock
        params = {"pressure": "resident", "division_size": 0.125, "bounds_min_x": -1.0, "bounds_min_y": 0.0,
                  "bounds_min_z": -1.0, "bounds_max_x": 1.0, "bounds_max_y": 3.0, "bounds_max_z": 1.0}
        with mock.patch.object(fgs, "fits", return_value=(False, "no compute adapter")):
            with self.assertRaisesRegex(ValueError, "no compute adapter"):
                fluid3d.resolve_backend(params, 16 * 24 * 16)
            params["pressure"] = "auto"
            self.assertEqual(fluid3d.resolve_backend(params, 100), "cpu")


def divergence(state):
    a = state.arrays
    return fluid3d.divergence(a["u"].astype(np.float64), a["v"].astype(np.float64), a["w"].astype(np.float64))


if __name__ == "__main__":
    unittest.main()
