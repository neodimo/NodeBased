"""Lane L6 step 2: the 2D smoke solver spike (nodebased/fluid2d.py), driven through simcache.solve_to_frame."""
import tempfile
import threading
import time
import unittest

import numpy as np

from nodebased import fluid2d, simcache
from nodebased.cancellation import Cancelled

SMALL = {"nx": 48, "ny": 48}


def solve(params, frames, cache=None, cancel=None):
    solver = fluid2d.Smoke2D(params, cancel=cancel)
    cache = cache or simcache.SimCache(enabled=False)
    run = simcache.run_key(None, params)
    return simcache.solve_to_frame(cache, run, frames, 1, solver.substeps, 0, solver.initial_state,
                                   solver.step, cancel), solver


def blob(solver, state, value=1.0):
    state.arrays["density"][20:26, 20:26] = value
    return state


class MassTests(unittest.TestCase):
    def test_density_mass_is_conserved_under_advection_away_from_walls(self):
        solver = fluid2d.Smoke2D({**SMALL, "source_density": 0.0, "source_temperature": 0.0})
        state = blob(solver, solver.initial_state())
        state.arrays["u"][:] = 0.7
        state.arrays["v"][:] = 0.4
        before = float(state.arrays["density"].sum(dtype=np.float64))
        u, v, density, _ = solver._advect(state.arrays["u"], state.arrays["v"], state.arrays["density"],
                                          state.arrays["temperature"], 1.0)
        after = float(density.sum(dtype=np.float64))
        self.assertAlmostEqual(after / before, 1.0, delta=1e-4)

    def test_density_mass_grows_only_by_the_source(self):
        params = {**SMALL, "source_density": 2.0}
        state, solver = solve(params, 10)
        cells = int(solver._source.sum())
        added = 10 * 2.0 * cells
        total = float(state.arrays["density"].sum(dtype=np.float64))
        self.assertAlmostEqual(total / added, 1.0, delta=0.10)   # semi-Lagrangian is not exactly conservative: drift stays within a few percent


class ProjectionTests(unittest.TestCase):
    def test_divergence_after_projection_is_below_tolerance_on_every_cell(self):
        params = {**SMALL, "tolerance": 1e-4}
        state, solver = solve(params, 15)
        div = fluid2d.divergence(state.arrays["u"].astype(np.float64), state.arrays["v"].astype(np.float64))
        self.assertLessEqual(float(np.abs(div).max()), 1e-4 + 1e-5)
        self.assertGreater(float(np.abs(state.arrays["v"]).max()), 0.01)     # not a trivially still field

    def test_walls_have_no_normal_velocity(self):
        state, _ = solve(SMALL, 8)
        self.assertEqual(float(np.abs(state.arrays["u"][:, [0, -1]]).max()), 0.0)
        self.assertEqual(float(np.abs(state.arrays["v"][[0, -1], :]).max()), 0.0)

    def test_pressure_solver_hook_is_used_and_cg_stays_the_reference(self):
        calls = []

        def hook(rhs, x0, tol, cap, cancel):
            calls.append(1)
            return fluid2d.conjugate_gradient(rhs, x0, tol, cap, cancel)

        solver = fluid2d.Smoke2D(SMALL, pressure_solver=hook)
        solver.step(solver.initial_state(), 1, 0, 0)
        self.assertEqual(len(calls), 1)


class DeterminismTests(unittest.TestCase):
    def test_two_runs_are_bit_identical(self):
        a, _ = solve(SMALL, 12)
        b, _ = solve(SMALL, 12)
        for name in fluid2d.ARRAYS:
            self.assertEqual(a.arrays[name].tobytes(), b.arrays[name].tobytes(), name)
        self.assertEqual(a.meta, b.meta)

    def test_split_solve_through_disk_matches_a_direct_solve(self):
        params = {**SMALL, "substeps": 2}
        direct, _ = solve(params, 10)
        with tempfile.TemporaryDirectory() as root:
            solve(params, 4, cache=simcache.SimCache(root))
            resumed, _ = solve(params, 10, cache=simcache.SimCache(root))
        for name in fluid2d.ARRAYS:
            self.assertEqual(direct.arrays[name].tobytes(), resumed.arrays[name].tobytes(), name)

    def test_checkpoint_and_restore_round_trip(self):
        state, solver = solve(SMALL, 5)
        copy = solver.checkpoint(state)
        restored = solver.restore(copy)
        state.arrays["density"][:] = 0.0                      # the checkpoint shares nothing with the source
        self.assertGreater(float(restored.arrays["density"].sum()), 0.0)
        again = solver.step(restored, 6, 0, 0)
        same = solver.step(solver.restore(copy), 6, 0, 0)
        self.assertEqual(again.arrays["v"].tobytes(), same.arrays["v"].tobytes())


class CancelTests(unittest.TestCase):
    def test_cancel_between_substeps_returns_promptly_and_keeps_banked_frames(self):
        cancel = threading.Event()
        params = {"nx": 128, "ny": 128, "substeps": 4}
        solver = fluid2d.Smoke2D(params, cancel=cancel)
        cache = simcache.SimCache(enabled=False)
        run = simcache.run_key(None, params)
        count = []
        stamp = []

        def step(state, frame, substep, seed):
            count.append(1)
            if len(count) == 10:               # raised mid-frame 3, after two whole frames were banked
                stamp.append(time.perf_counter())
                cancel.set()
            return solver.step(state, frame, substep, seed)

        with self.assertRaises(Cancelled):
            simcache.solve_to_frame(cache, run, 500, 1, solver.substeps, 0, solver.initial_state, step, cancel)
        self.assertLess(time.perf_counter() - stamp[0], 0.5)     # one 128 x 128 substep is milliseconds
        self.assertEqual(len(count), 10)
        self.assertEqual(cache.latest_at_or_before(run, 499)[0], 2)

    def test_cancel_inside_the_pressure_solve(self):
        cancel = threading.Event()
        cancel.set()
        rhs = np.random.default_rng(0).standard_normal((64, 64))
        rhs -= rhs.mean()
        with self.assertRaises(Cancelled):
            fluid2d.conjugate_gradient(rhs, np.zeros_like(rhs), 1e-12, 500, cancel)


class PlumeTests(unittest.TestCase):
    def test_a_buoyant_plume_rises(self):
        solver = fluid2d.Smoke2D({"nx": 64, "ny": 96})
        state = solver.initial_state()
        heights = []
        for frame in range(1, 51):
            state = solver.step(state, frame, 0, 0)
            if frame in (10, 50):
                heights.append(fluid2d.centre_of_mass(state.arrays["density"])[1])
        self.assertGreater(heights[1], heights[0] + 2.0)
        self.assertGreater(heights[1], 0.12 * 96 + 4.0)       # well above the source row
        cold = fluid2d.Smoke2D({"nx": 64, "ny": 96, "buoyancy_temperature": 0.0})
        still = cold.initial_state()
        for frame in range(1, 51):
            still = cold.step(still, frame, 0, 0)
        self.assertGreater(heights[1], fluid2d.centre_of_mass(still.arrays["density"])[1] + 5.0)

    def test_centre_of_mass_of_empty_grid_is_none(self):
        self.assertIsNone(fluid2d.centre_of_mass(np.zeros((4, 4))))


class GpuPressureParityTests(unittest.TestCase):
    """The wgpu red-black pressure solve (tools/fluid_gpu.py) against the NumPy reference."""

    def test_gpu_pressure_solve_meets_the_tolerance_and_matches_numpy(self):
        import os
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
        try:
            from fluid_gpu import GpuPressure
            gpu = GpuPressure()
        except Exception as error:                    # no wgpu or no adapter on this machine
            self.skipTest(f"no wgpu adapter: {error}")
        params = {"nx": 96, "ny": 96}
        solver = fluid2d.Smoke2D(params)
        state = solver.initial_state()
        for frame in range(1, 9):
            state = solver.step(state, frame, 0, 0)
        reference = solver.step(state, 9, 0, 0)
        solver.pressure_solver = gpu.solve
        other = solver.step(state, 9, 0, 0)
        div = fluid2d.divergence(other.arrays["u"].astype(np.float64), other.arrays["v"].astype(np.float64))
        self.assertLessEqual(float(np.abs(div).max()), 1e-3 + 1e-5)
        for name in ("u", "v"):
            self.assertLess(float(np.abs(reference.arrays[name] - other.arrays[name]).max()), 5e-3, name)


if __name__ == "__main__":
    unittest.main()
