"""Lane L6 step E, part 1: the FLIP/PIC liquid solver (nodebased/flip3d.py) on a small grid.

A dam break conserves its volume within tolerance and settles flat, a drop into a tank splashes, solids are respected,
runs are deterministic and resumable, the particle count per cell stays within 3 and 12 (interior gaps refilled, crowded
cells thinned), the pressure system is symmetric and solved to tolerance, and a solve is cancellable.
"""
import threading
import unittest

import numpy as np

from nodebased import fluid3d, flip3d
from nodebased.cancellation import Cancelled

SHAPE = (16, 16, 8)
G = {"nx": SHAPE[0], "ny": SHAPE[1], "nz": SHAPE[2], "gravity": 0.3, "flip_ratio": 0.9, "substeps": 2}


class BoxSource(fluid3d.Source):
    """A liquid source over a box of whole cells [lo, hi)."""

    def __init__(self, lo, hi, velocity=(0.0, 0.0, 0.0), **kw):
        super().__init__("sphere", fluid_type="liquid", velocity=velocity, **kw)
        self.lo, self.hi = lo, hi

    def footprint(self, solver, frame):
        ii, jj, kk = np.meshgrid(*(np.arange(self.lo[a], self.hi[a]) for a in range(3)), indexing="ij")
        flat = np.ravel_multi_index((ii.ravel(), jj.ravel(), kk.ravel()), solver.shape).astype(np.intp)
        return flat, np.ones(len(flat)), None


class BoxCollider(fluid3d.Collider):
    def __init__(self, lo, hi):
        self.track, self.animated_flag, self._cache = None, False, {}
        self.lo, self.hi = lo, hi

    def mask(self, solver, frame, substep=0, substeps=1):
        solid = np.zeros(solver.shape, bool)
        solid[self.lo[0]:self.hi[0], self.lo[1]:self.hi[1], self.lo[2]:self.hi[2]] = True
        return solid, None


def run(solver, frames, every=None, seed=0):
    state = solver.initial_state(seed)
    seen = []
    for frame in range(1, frames + 1):
        for substep in range(solver.substeps):
            state = solver.step(state, frame, substep, seed)
        if every and frame % every == 0:
            seen.append((frame, state))
    return state, seen


def columns(pos, shape):
    col = np.floor(pos[:, 0]).astype(int) * shape[2] + np.floor(pos[:, 2]).astype(int)
    return col


class DamBreakTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = flip3d.Liquid3D(G, sources=[BoxSource((0, 0, 0), (5, 10, 8))])
        cls.start = 5 * 10 * 8 * 8
        cls.state, cls.seen = run(cls.solver, 150, every=10)

    def test_seeds_eight_particles_per_cell(self):
        state = self.solver.step(self.solver.initial_state(0), 1, 0, 0)
        self.assertEqual(len(state.arrays["position"]), self.start)
        self.assertEqual(len(np.unique(state.arrays["id"])), self.start)

    def test_volume_is_conserved_within_tolerance(self):
        ratio = len(self.state.arrays["position"]) / self.start
        self.assertGreater(ratio, 0.92)
        self.assertLess(ratio, 1.05)

    def test_settles_flat(self):
        pos = self.state.arrays["position"]
        col = columns(pos, SHAPE)
        top = np.array([np.percentile(pos[col == c, 1], 95) for c in np.unique(col)])
        depth = len(pos) / 8 / (SHAPE[0] * SHAPE[2])            # a flat pool of this volume
        self.assertLess(abs(top.mean() - depth), 0.6)
        self.assertLess(top.std(), 0.7)
        speed = lambda s: np.linalg.norm(s.arrays["velocity"], axis=1).mean()
        peak = max(speed(s) for _, s in self.seen)
        self.assertLess(speed(self.state), 0.5 * peak)

    def test_the_column_falls_and_spreads(self):
        pos = self.state.arrays["position"]
        self.assertGreater(pos[:, 0].max(), 0.9 * SHAPE[0])       # reached the far wall
        self.assertLess(pos[:, 1].mean(), 4.0)                     # from 5 cells of mean height at the start

    def test_particles_stay_in_the_domain(self):
        pos = self.state.arrays["position"]
        self.assertTrue(np.all(pos >= 0) and np.all(pos < np.array(SHAPE)))

    def test_count_per_cell_stays_within_the_limits(self):
        pos = self.state.arrays["position"]
        cells = np.ravel_multi_index(tuple(np.floor(pos).astype(int).T), SHAPE)
        self.assertLessEqual(np.bincount(cells).max(), flip3d.MAX_PER_CELL)


class DropTests(unittest.TestCase):
    def test_a_drop_into_a_tank_splashes(self):
        pool = 4
        solver = flip3d.Liquid3D({**G, "nx": 16, "ny": 24, "nz": 16, "gravity": 0.15, "flip_ratio": 0.95},
                                 sources=[BoxSource((0, 0, 0), (16, pool, 16)), BoxSource((6, 14, 6), (10, 18, 10))])
        state = solver.initial_state(0)
        peak_after, drop_low = 0.0, 99.0
        for frame in range(1, 61):
            for substep in range(solver.substeps):
                state = solver.step(state, frame, substep, 0)
            pos, ids = state.arrays["position"], state.arrays["id"]
            drop = ids >= 16 * pool * 16 * 8
            if frame > 15:
                peak_after = max(peak_after, float(pos[:, 1].max()))
                drop_low = min(drop_low, float(pos[drop, 1].mean()))
        self.assertGreater(peak_after, pool + 2.0)                 # crown and rebound rise above the pool
        self.assertLess(drop_low, pool + 1.5)                      # the drop went into the pool
        pos = state.arrays["position"]
        self.assertGreater(float(pos[:, 1].min()), 0.0)
        col = columns(pos, (16, 24, 16))
        top = np.array([pos[col == c, 1].max() for c in np.unique(col)])
        self.assertGreater(top.max() - top.min(), 0.5)             # the free surface is disturbed, above the floor
        self.assertGreater(top.min(), 1.0)


class SolidTests(unittest.TestCase):
    def test_solids_are_respected(self):
        block = ((6, 0, 0), (9, 6, 8))
        solver = flip3d.Liquid3D(G, sources=[BoxSource((0, 0, 0), (5, 10, 8))], colliders=[BoxCollider(*block)])
        state = solver.initial_state(0)
        for frame in range(1, 61):
            for substep in range(solver.substeps):
                state = solver.step(state, frame, substep, 0)
            pos = state.arrays["position"]
            inside = ((pos[:, 0] >= 6) & (pos[:, 0] < 9) & (pos[:, 1] < 6))
            self.assertFalse(inside.any(), frame)
        pos = state.arrays["position"]
        self.assertGreater(int((pos[:, 0] > 9).sum()), 0)          # some liquid climbed over the block


class DeterminismTests(unittest.TestCase):
    def make(self):
        return flip3d.Liquid3D(G, sources=[BoxSource((0, 0, 0), (4, 8, 8))])

    def test_same_seed_same_state_and_resume_matches(self):
        a, _ = run(self.make(), 20, seed=3)
        b, _ = run(self.make(), 20, seed=3)
        self.assertEqual(a, b)
        solver = self.make()
        state = solver.initial_state(3)
        for frame in range(1, 11):
            for substep in range(solver.substeps):
                state = solver.step(state, frame, substep, 3)
        state = solver.restore(solver.checkpoint(state))
        for frame in range(11, 21):
            for substep in range(solver.substeps):
                state = solver.step(state, frame, substep, 3)
        self.assertEqual(state, a)

    def test_the_step_does_not_modify_its_input(self):
        solver = self.make()
        state = solver.step(solver.initial_state(0), 1, 0, 0)
        before = solver.checkpoint(state)
        solver.step(state, 2, 0, 0)
        self.assertEqual(state, before)

    def test_flip_ratio_changes_the_result(self):
        pic, _ = run(flip3d.Liquid3D({**G, "flip_ratio": 0.0}, sources=[BoxSource((0, 0, 0), (4, 8, 8))]), 15)
        flip, _ = run(flip3d.Liquid3D({**G, "flip_ratio": 1.0}, sources=[BoxSource((0, 0, 0), (4, 8, 8))]), 15)
        self.assertFalse(np.array_equal(pic.arrays["position"], flip.arrays["position"]))

    def test_cancellation(self):
        solver = self.make()
        cancel = threading.Event()
        solver.cancel = cancel
        state = solver.initial_state(0)
        state = solver.step(state, 1, 0, 0)
        cancel.set()
        with self.assertRaises(Cancelled):
            solver.step(state, 2, 0, 0)


class MaintenanceTests(unittest.TestCase):
    def test_crowded_cells_are_thinned_and_interior_gaps_refilled(self):
        solver = flip3d.Liquid3D({**G, "nx": 8, "ny": 8, "nz": 8}, sources=[])
        rng = np.random.default_rng(0)
        # a 5-cube of liquid with 8 per cell, then 30 particles piled into one cell, and one interior cell holding 1
        cells = np.stack(np.meshgrid(*[np.arange(1, 6)] * 3, indexing="ij"), axis=-1).reshape(-1, 3).astype(float)
        pos = np.repeat(cells, 8, axis=0) + rng.random((len(cells) * 8, 3))
        pos = np.concatenate((pos, np.full((30, 3), 2.5) + rng.random((30, 3)) * 0.4 - 0.2))
        keep = ~((np.floor(pos[:, 0]) == 3) & (np.floor(pos[:, 1]) == 3) & (np.floor(pos[:, 2]) == 3))
        pos = np.concatenate((pos[keep], np.array([[3.5, 3.5, 3.5]])))
        vel = np.zeros_like(pos)
        ids = np.arange(len(pos), dtype=np.int64)
        age = np.zeros(len(pos), np.int32)
        pos2, vel2, ids2, age2, next_id = solver._maintain(pos, vel, ids, age, len(pos), None, rng)
        cell = np.ravel_multi_index(tuple(np.floor(pos2).astype(int).T), (8, 8, 8))
        count = np.bincount(cell, minlength=512).reshape(8, 8, 8)
        self.assertLessEqual(count.max(), solver.max_per_cell)
        self.assertEqual(count[2, 2, 2], solver.max_per_cell)      # the pile was cut to the maximum
        self.assertEqual(count[3, 3, 3], flip3d.MIN_PER_CELL)      # the gap was topped up
        self.assertEqual(len(np.unique(ids2)), len(ids2))
        self.assertEqual(next_id - len(pos), int((ids2 >= len(pos)).sum()))   # ids grow by one per new particle


class PressureTests(unittest.TestCase):
    def test_the_liquid_system_is_symmetric_and_solved_to_tolerance(self):
        rng = np.random.default_rng(1)
        shape = (8, 8, 8)
        liquid = np.zeros(shape, bool)
        liquid[1:6, 1:4, 1:7] = True                                 # air above and to the sides
        solid = np.zeros(shape, bool)
        solid[4, 1, 1:5] = False
        system = flip3d.LiquidPoisson(shape, liquid, None)
        x, y = rng.random(shape) * liquid, rng.random(shape) * liquid
        ax, ay = np.empty(shape), np.empty(shape)
        system.apply(x, ax)
        system.apply(y, ay)
        self.assertAlmostEqual(float((x * ay).sum()), float((ax * y).sum()), places=9)
        self.assertGreater(float((x * ax).sum()), 0.0)
        self.assertFalse(system.singular)
        rhs = rng.random(shape) * liquid
        q, iterations, residual = fluid3d.conjugate_gradient(rhs, np.zeros(shape), 1e-6, 500, None, system)
        self.assertLessEqual(residual, 1e-6)
        check = np.empty(shape)
        system.apply(q, check)
        self.assertLess(float(np.abs(check - rhs).max()), 1e-5)

    def test_a_full_container_is_singular(self):
        liquid = np.ones((6, 6, 6), bool)
        self.assertTrue(flip3d.LiquidPoisson((6, 6, 6), liquid, None).singular)

    def test_liquid_rests_in_a_full_tank(self):
        solver = flip3d.Liquid3D({**G, "nx": 6, "ny": 6, "nz": 6}, sources=[BoxSource((0, 0, 0), (6, 6, 6))])
        state, _ = run(solver, 10)
        self.assertLess(float(np.abs(state.arrays["velocity"]).max()), 0.05)


if __name__ == "__main__":
    unittest.main()
