"""Lane L6 step C: the 3D smoke and fire solver (nodebased/fluid3d.py), driven through simcache.solve_to_frame."""
import threading
import time
import unittest

import numpy as np

from nodebased import fluid3d, simcache
from nodebased.cancellation import Cancelled

SMALL = {"nx": 20, "ny": 28, "nz": 20}


def solve(params, frames, cache=None, cancel=None, **kwargs):
    solver = fluid3d.Smoke3D(params, cancel=cancel, **kwargs)
    cache = cache or simcache.SimCache(enabled=False)
    run = simcache.run_key(None, params)
    state = simcache.solve_to_frame(cache, run, frames, 1, solver.substeps, 0, solver.initial_state,
                                    solver.step, cancel)
    return state, solver


def box_triangles(lo, hi):
    """The 12 triangles of an axis-aligned box, outward winding, in cell coordinates."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    c = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
             (2, 3, 7), (2, 7, 6), (1, 2, 6), (1, 6, 5), (0, 4, 7), (0, 7, 3)]
    return np.array([c[list(f)] for f in faces])


def box_collider(lo, hi, **kwargs):
    tri = box_triangles(lo, hi)
    return fluid3d.Collider(fluid3d.GeometryTrack(lambda frame: tri), **kwargs)


def divergence_of(state):
    a = state.arrays
    return fluid3d.divergence(a["u"].astype(np.float64), a["v"].astype(np.float64), a["w"].astype(np.float64))


class VoxeliserTests(unittest.TestCase):
    def test_a_closed_box_fills_and_its_surface_is_a_shell(self):
        tri = box_triangles((4.2, 4.2, 4.2), (9.8, 9.8, 9.8))
        surface = fluid3d.voxelize_surface(tri, (16, 16, 16))
        filled = fluid3d.fill_interior(surface)
        self.assertEqual(int(filled.sum()), 6 ** 3)                       # cells 4..9 on every axis
        self.assertTrue(filled[4:10, 4:10, 4:10].all())
        self.assertLess(int(surface.sum()), int(filled.sum()))            # the shell is hollow
        self.assertFalse(surface[6:8, 6:8, 6:8].any())

    def test_surface_voxelisation_is_conservative(self):
        # a triangle that only grazes a corner of a cell still marks it
        tri = np.array([[[3.999, 3.5, 3.5], [5.0, 3.5, 3.5], [5.0, 4.5, 3.5]]])
        surface = fluid3d.voxelize_surface(tri, (8, 8, 8))
        self.assertTrue(surface[3, 3, 3] and surface[4, 3, 3] and surface[4, 4, 3])

    def test_an_open_mesh_encloses_nothing(self):
        tri = box_triangles((4.2, 4.2, 4.2), (9.8, 9.8, 9.8))[2:]        # two faces missing
        surface = fluid3d.voxelize_surface(tri, (16, 16, 16))
        self.assertEqual(int(fluid3d.fill_interior(surface).sum()), int(surface.sum()))


class MassTests(unittest.TestCase):
    def test_density_mass_is_conserved_under_advection_away_from_walls(self):
        for advection in fluid3d.ADVECTIONS:
            solver = fluid3d.Smoke3D({**SMALL, "advection": advection, "default_source": 0})
            state = solver.initial_state()
            state.arrays["density"][8:12, 8:12, 8:12] = 1.0
            a = {name: state.arrays[name].copy() for name in ("u", "v", "w", "density", "temperature", "fuel")}
            a["u"][:] = 0.7
            a["v"][:] = 0.4
            a["w"][:] = -0.3
            before = float(a["density"].sum(dtype=np.float64))
            solver._advect(a, 1.0, 0.0)
            after = float(a["density"].sum(dtype=np.float64))
            self.assertAlmostEqual(after / before, 1.0, delta=2e-3, msg=advection)

    def test_density_mass_grows_only_by_the_source(self):
        for advection in fluid3d.ADVECTIONS:
            params = {**SMALL, "source_density": 2.0, "advection": advection}
            state, solver = solve(params, 10)
            cells = len(solver.sources[0].footprint(solver, 1)[0])
            added = 10 * 2.0 * cells
            total = float(state.arrays["density"].sum(dtype=np.float64))
            # semi-Lagrangian is not conservative under compression; measured drift on this fast, compact
            # plume is +11 percent (semi-Lagrangian) and +16 percent (MacCormack, rescaled to the same total)
            self.assertAlmostEqual(total / added, 1.0, delta=0.20, msg=advection)

    def test_maccormack_keeps_sharper_edges_than_semi_lagrangian(self):
        peaks = {}
        for advection in fluid3d.ADVECTIONS:
            solver = fluid3d.Smoke3D({**SMALL, "advection": advection, "default_source": 0})
            state = solver.initial_state()
            state.arrays["density"][8:12, 8:12, 8:12] = 1.0
            a = {name: state.arrays[name].copy() for name in ("u", "v", "w", "density", "temperature", "fuel")}
            a["u"][:] = 0.37
            for _ in range(6):
                solver._advect(a, 1.0, 0.0)
            peaks[advection] = float(a["density"].max())
        self.assertGreater(peaks["maccormack"], peaks["semi_lagrangian"] + 0.05)


class ProjectionTests(unittest.TestCase):
    def test_divergence_after_projection_is_below_tolerance_on_every_cell(self):
        state, solver = solve({**SMALL, "tolerance": 1e-4}, 12)
        div = divergence_of(state)
        self.assertLessEqual(float(np.abs(div).max()), 1e-4 + 1e-5)
        self.assertGreater(float(np.abs(state.arrays["v"]).max()), 0.01)     # not a trivially still field

    def test_walls_have_no_normal_velocity(self):
        state, _ = solve(SMALL, 8)
        a = state.arrays
        for name, axis in (("u", 0), ("v", 1), ("w", 2)):
            for index in (0, -1):
                self.assertEqual(float(np.abs(np.take(a[name], index, axis=axis)).max()), 0.0)

    def test_pressure_solver_hook_is_used_and_cg_stays_the_reference(self):
        calls = []

        def hook(rhs, x0, tol, cap, cancel, system):
            calls.append((rhs.shape, type(system).__name__))
            return fluid3d.conjugate_gradient(rhs, x0, tol, cap, cancel, system)
        solver = fluid3d.Smoke3D(SMALL, pressure_solver=hook)
        state = solver.initial_state()
        for frame in range(1, 4):
            state = solver.step(state, frame, 0, 0)
        self.assertEqual(calls[0], ((20, 28, 20), "Poisson3D"))
        self.assertEqual(len(calls), 3)
        self.assertIs(fluid3d.Smoke3D(SMALL).pressure_solver, fluid3d.conjugate_gradient)

    def test_the_seven_point_operator_is_symmetric_and_annihilates_constants_in_a_closed_box(self):
        system = fluid3d.Poisson3D((6, 7, 8))
        rng = np.random.default_rng(3)
        p, q = rng.random((6, 7, 8)), rng.random((6, 7, 8))
        out_p, out_q = np.empty_like(p), np.empty_like(q)
        system.apply(p, out_p)
        system.apply(q, out_q)
        self.assertAlmostEqual(float((out_p * q).sum()), float((p * out_q).sum()), places=9)
        self.assertLess(float(np.abs(system.apply(np.ones((6, 7, 8)), out_p)).max()), 1e-12)


class BuoyancyTests(unittest.TestCase):
    def test_a_hot_plume_rises_and_a_cold_one_does_not(self):
        hot = fluid3d.Smoke3D({"nx": 24, "ny": 40, "nz": 24})
        cold = fluid3d.Smoke3D({"nx": 24, "ny": 40, "nz": 24, "buoyancy_temperature": 0.0})
        heights = []
        state, still = hot.initial_state(), cold.initial_state()
        for frame in range(1, 41):
            state = hot.step(state, frame, 0, 0)
            still = cold.step(still, frame, 0, 0)
            if frame in (10, 40):
                heights.append(fluid3d.centre_of_mass(state.arrays["density"])[1])
        self.assertGreater(heights[1], heights[0] + 2.0)
        self.assertGreater(heights[1], 0.12 * 40 + 4.0)
        self.assertGreater(heights[1], fluid3d.centre_of_mass(still.arrays["density"])[1] + 5.0)

    def test_dissipation_and_cooling_remove_density_and_heat(self):
        base = {**SMALL, "source_density": 0.0, "source_temperature": 0.0, "vorticity": 0.0,
                "buoyancy_temperature": 0.0, "buoyancy_density": 0.0}
        for name, key, rate in (("density", "dissipation", 0.2), ("temperature", "cooling_rate", 0.3)):
            solver = fluid3d.Smoke3D({**base, key: rate})
            state = solver.initial_state()
            state.arrays[name][8:12, 8:12, 8:12] = 1.0
            after = solver.step(state, 1, 0, 0).arrays[name][8:12, 8:12, 8:12].mean()
            self.assertAlmostEqual(float(after), float(np.exp(-rate)), delta=0.02)


class ColliderTests(unittest.TestCase):
    PARAMS = {"nx": 24, "ny": 40, "nz": 24, "tolerance": 1e-4}

    def run_with_box(self, frames=45):
        collider = box_collider((6.0, 14.0, 6.0), (18.0, 18.0, 18.0))
        return solve(self.PARAMS, frames, colliders=[collider])

    def test_density_stays_out_of_the_solid_and_the_plume_is_blocked(self):
        state, solver = self.run_with_box()
        solid, _, _ = solver._solid_for(1)
        self.assertTrue(solid[8:16, 14:18, 8:16].all())
        self.assertEqual(float(state.arrays["density"][solid].max()), 0.0)
        free, _ = solve(self.PARAMS, 45)
        above = float(state.arrays["density"][:, 20:, :].sum())
        above_free = float(free.arrays["density"][:, 20:, :].sum())
        self.assertLess(above, 0.5 * above_free)                       # the roof holds most of it back
        self.assertGreater(float(state.arrays["density"][:, :14, :].sum()), float(free.arrays["density"][:, :14, :].sum()))

    def test_no_flow_through_the_solid_and_divergence_is_below_tolerance_in_the_fluid(self):
        state, solver = self.run_with_box(30)
        solid, _, _ = solver._solid_for(1)
        a = state.arrays
        inner_u = a["u"][1:-1]
        near = solid[:-1] | solid[1:]
        self.assertEqual(float(np.abs(inner_u[near]).max()), 0.0)
        near_v = solid[:, :-1] | solid[:, 1:]
        self.assertEqual(float(np.abs(a["v"][:, 1:-1][near_v]).max()), 0.0)
        div = divergence_of(state)
        self.assertLessEqual(float(np.abs(div[~solid]).max()), 1e-4 + 1e-5)

    def test_a_moving_collider_pushes_the_fluid(self):
        # a slab moving up at 0.5 cells per frame, sampled per frame
        frames = {}

        def provider(frame):
            y = 4.0 + 0.5 * frame
            return box_triangles((6.0, y, 6.0), (18.0, y + 2.0, 18.0))
        track = fluid3d.GeometryTrack(provider, animated=True, start_frame=1)
        collider = fluid3d.Collider(track, animated=True)
        params = {**self.PARAMS, "default_source": 0, "vorticity": 0.0}
        solver = fluid3d.Smoke3D(params, colliders=[collider])
        state = solver.initial_state()
        for frame in range(1, 6):
            state = solver.step(state, frame, 0, 0)
        solid, velocity, _ = solver._solid_for(5)
        self.assertIsNotNone(velocity)
        self.assertAlmostEqual(float(velocity[solid][:, 1].mean()), 0.5, places=3)
        self.assertGreater(float(state.arrays["v"][8:16, :, 8:16].max()), 0.2)   # air above is pushed up

    def test_a_static_collider_with_animated_on_matches_off(self):
        # a track that never actually changes shape or position: Collider.animated is (animated_flag and
        # track.animated), so "on" against a non-animated track is exactly the frozen path, bit-identical
        off = box_collider((6.0, 14.0, 6.0), (18.0, 18.0, 18.0), animated=False)
        on = box_collider((6.0, 14.0, 6.0), (18.0, 18.0, 18.0), animated=True)
        self.assertFalse(on.animated)
        state_off, _ = solve(self.PARAMS, 20, colliders=[off])
        state_on, _ = solve(self.PARAMS, 20, colliders=[on])
        for name in ("density", "temperature", "u", "v", "w"):
            self.assertTrue(np.array_equal(state_off.arrays[name], state_on.arrays[name]), name)

    def test_a_swept_collider_leaves_a_wake_in_still_smoke(self):
        # a block sweeping through a still, uniform smoke slab pushes it out of its path: nothing survives
        # inside the block, and the region it has already passed through ("behind") is thinner than the
        # region it has not reached yet ("ahead"), which stays close to the original fill
        def provider(frame):
            x = 2.0 + 2.0 * frame
            return box_triangles((x, 10.0, 6.0), (x + 4.0, 14.0, 18.0))
        track = fluid3d.GeometryTrack(provider, animated=True, start_frame=1)
        collider = fluid3d.Collider(track, animated=True)
        params = {**self.PARAMS, "nx": 40, "default_source": 0, "vorticity": 0.0, "boundary_x": "open"}
        solver = fluid3d.Smoke3D(params, colliders=[collider])
        state = solver.initial_state()
        state.arrays["density"][:, 8:16, 4:20] = 1.0
        ahead_before = float(state.arrays["density"][20:30, 10:14, 6:18].sum())
        behind_before = float(state.arrays["density"][0:8, 10:14, 6:18].sum())
        for frame in range(1, 7):
            state = solver.step(state, frame, 0, 0)
        solid, _, _ = solver._solid_for(6, 0)
        self.assertTrue(solid.any())
        self.assertEqual(float(state.arrays["density"][solid].max()), 0.0)
        ahead = float(state.arrays["density"][20:30, 10:14, 6:18].sum())        # the sweep has not reached here
        behind = float(state.arrays["density"][0:8, 10:14, 6:18].sum())        # the sweep already passed through
        self.assertGreater(ahead, 0.85 * ahead_before)
        self.assertLess(behind, 0.85 * behind_before)


class FireTests(unittest.TestCase):
    PARAMS = {"nx": 20, "ny": 32, "nz": 20, "fire": 1, "source_fuel": 1.0, "source_density": 0.2,
              "source_temperature": 1.0}

    def test_fuel_ignites_above_the_ignition_temperature_and_is_consumed(self):
        burning, _ = solve({**self.PARAMS, "ignition_temperature": 0.5}, 12)
        cold, _ = solve({**self.PARAMS, "ignition_temperature": 50.0}, 12)
        self.assertGreater(float(burning.arrays["burn"].max()), 0.0)
        self.assertEqual(float(cold.arrays["burn"].max()), 0.0)
        self.assertLess(float(burning.arrays["fuel"].sum()), 0.5 * float(cold.arrays["fuel"].sum()))
        self.assertGreater(float(burning.arrays["temperature"].max()), float(cold.arrays["temperature"].max()) + 0.5)
        self.assertGreater(float(burning.arrays["density"].sum()), float(cold.arrays["density"].sum()))  # soot yield

    def test_without_fire_nothing_burns_even_with_fuel(self):
        state, _ = solve({**self.PARAMS, "fire": 0}, 6)
        self.assertEqual(float(state.arrays["burn"].max()), 0.0)
        self.assertGreater(float(state.arrays["fuel"].sum()), 0.0)

    def test_no_fuel_and_cold_fuel_make_no_flame(self):
        no_fuel, _ = solve({**self.PARAMS, "source_fuel": 0.0}, 5)
        cold, _ = solve({**self.PARAMS, "ignition_temperature": 100.0}, 5)
        self.assertEqual(float(no_fuel.arrays["burn"].max()), 0.0)
        self.assertEqual(float(cold.arrays["burn"].max()), 0.0)

    def test_inefficient_fuel_is_retained_and_lifespan_slows_its_consumption(self):
        params = {**self.PARAMS, "default_source": 0, "vorticity": 0.0, "buoyancy_temperature": 0.0}
        solver = fluid3d.Smoke3D(params)
        initial = solver.initial_state()
        initial.arrays["fuel"][8:12, 8:12, 8:12] = 1.0
        initial.arrays["temperature"][8:12, 8:12, 8:12] = 1.0
        quick = solver.step(initial, 1, 0, 0)
        slow_solver = fluid3d.Smoke3D({**params, "flame_lifespan": 3.0, "fuel_inefficiency": 0.25})
        slow = slow_solver.step(initial, 1, 0, 0)
        self.assertGreater(float(slow.arrays["fuel"].sum()), float(quick.arrays["fuel"].sum()))
        self.assertGreater(float(slow.arrays["burn"].max()), 0.0)

    def test_burning_fuel_conserves_fuel_plus_consumed(self):
        solver = fluid3d.Smoke3D({**self.PARAMS, "default_source": 0, "ignition_temperature": 0.5, "vorticity": 0.0,
                                  "buoyancy_temperature": 0.0})
        state = solver.initial_state()
        state.arrays["fuel"][8:12, 8:12, 8:12] = 1.0
        state.arrays["temperature"][8:12, 8:12, 8:12] = 1.0
        before = float(state.arrays["fuel"].sum(dtype=np.float64))
        after = solver.step(state, 1, 0, 0)
        consumed = float(after.arrays["burn"].sum(dtype=np.float64)) * solver.dt
        self.assertAlmostEqual(float(after.arrays["fuel"].sum(dtype=np.float64)) + consumed, before, delta=before * 0.02)
        self.assertAlmostEqual(consumed, before * (1 - np.exp(-0.6)), delta=before * 0.03)

    def test_expansion_makes_a_divergence_source_of_the_burn_rate(self):
        params = {**self.PARAMS, "burn_expansion": 0.4, "boundary_y": "open", "tolerance": 1e-4}
        state, solver = solve(params, 10)
        expected = state.arrays["burn"].astype(np.float64) * 0.4
        self.assertGreater(float(expected.max()), 0.05)
        residual = divergence_of(state) - expected
        self.assertLessEqual(float(np.abs(residual).max()), 1e-4 + 1e-5)

    def test_gas_release_adds_positive_expansion_and_combustion_off_is_the_smoke_baseline(self):
        base = {**self.PARAMS, "boundary_y": "open", "tolerance": 1e-4}
        no_release, _ = solve({**base, "gas_release": 0.0}, 7)
        release, _ = solve({**base, "gas_release": 0.8}, 7)
        positive_no = float(np.maximum(divergence_of(no_release), 0.0).sum(dtype=np.float64))
        positive_yes = float(np.maximum(divergence_of(release), 0.0).sum(dtype=np.float64))
        self.assertGreater(positive_yes, positive_no + 0.1)
        baseline, _ = solve({**base, "fire": 0, "source_fuel": 0.0}, 7)
        disabled, _ = solve({**base, "fire": 0, "source_fuel": 1.0}, 7)
        for name in ("u", "v", "w", "density", "temperature", "burn"):
            np.testing.assert_array_equal(disabled.arrays[name], baseline.arrays[name])

    def test_combustion_is_repeatable_and_has_a_small_reference_render(self):
        from pathlib import Path
        import nodebased
        a, _ = solve({**self.PARAMS, "gas_release": 0.25}, 6)
        b, _ = solve({**self.PARAMS, "gas_release": 0.25}, 6)
        self.assertEqual(a, b)
        root = Path(nodebased.__file__).resolve().parents[1]
        image = root / "docs" / "images" / "fluid_combustion.png"
        self.assertTrue(image.is_file())
        self.assertLess(image.stat().st_size, 200_000)
        self.assertIn("images/fluid_combustion.png", (root / "docs" / "FLUIDS_SPIKE.md").read_text())


class BoundaryTests(unittest.TestCase):
    def test_an_open_boundary_lets_smoke_leave_and_a_closed_one_keeps_it(self):
        base = {"nx": 16, "ny": 24, "nz": 16, "source_y": 0.2, "source_radius": 0.1}
        closed, _ = solve(base, 70)
        opened, _ = solve({**base, "boundary_y": "open"}, 70)
        closed_mass = float(closed.arrays["density"].sum(dtype=np.float64))
        open_mass = float(opened.arrays["density"].sum(dtype=np.float64))
        self.assertLess(open_mass, 0.7 * closed_mass)
        self.assertLessEqual(float(np.abs(divergence_of(opened)).max()), 1e-3 + 1e-5)
        self.assertGreater(float(np.abs(opened.arrays["v"][:, -1, :]).max()), 0.01)      # air actually crosses the top

    def test_open_side_walls_are_not_walls(self):
        state, _ = solve({"nx": 16, "ny": 24, "nz": 16, "boundary_x": "open", "boundary_z": "open"}, 20)
        self.assertGreater(float(np.abs(state.arrays["u"][[0, -1]]).max()), 0.0)


class DeterminismTests(unittest.TestCase):
    def test_two_runs_are_bit_identical(self):
        a, _ = solve(SMALL, 8)
        b, _ = solve(SMALL, 8)
        self.assertEqual(a, b)

    def test_scrub_and_jump_agree_through_a_cache(self):
        params = {**SMALL, "substeps": 2}
        cache = simcache.SimCache(enabled=False)
        solver = fluid3d.Smoke3D(params)
        run = simcache.run_key(None, params)
        for frame in (1, 2, 3, 4, 5, 6):        # scrubbed forward frame by frame
            scrubbed = simcache.solve_to_frame(cache, run, frame, 1, 2, 0, solver.initial_state, solver.step)
        jumped = simcache.solve_to_frame(simcache.SimCache(enabled=False), run, 6, 1, 2, 0,
                                         solver.initial_state, solver.step)
        self.assertEqual(scrubbed, jumped)
        # a jump after a scrub back is served from the checkpoints, still the same
        again = simcache.solve_to_frame(cache, run, 6, 1, 2, 0, solver.initial_state, solver.step)
        self.assertEqual(again, jumped)

    def test_resuming_from_a_checkpoint_equals_the_straight_solve(self):
        solver = fluid3d.Smoke3D(SMALL)
        state = solver.initial_state()
        checkpoints = {}
        for frame in range(1, 7):
            state = solver.step(state, frame, 0, 0)
            checkpoints[frame] = solver.checkpoint(state)
        resumed = solver.restore(checkpoints[3])
        for frame in range(4, 7):
            resumed = solver.step(resumed, frame, 0, 0)
        self.assertEqual(resumed, state)

    def _moving_collider(self):
        def provider(frame):
            x = 2.0 + 0.6 * frame
            return box_triangles((x, 10.0, 6.0), (x + 4.0, 14.0, 18.0))
        track = fluid3d.GeometryTrack(provider, animated=True, start_frame=1)
        return fluid3d.Collider(track, animated=True)

    def test_a_moving_collider_solve_is_bit_identical_across_runs(self):
        params = {**SMALL, "boundary_x": "open", "substeps": 2}
        a, _ = solve(params, 6, colliders=[self._moving_collider()])
        b, _ = solve(params, 6, colliders=[self._moving_collider()])
        self.assertEqual(a, b)

    def test_a_moving_collider_resuming_from_a_checkpoint_equals_the_straight_solve(self):
        params = {**SMALL, "boundary_x": "open", "substeps": 2}
        solver = fluid3d.Smoke3D(params, colliders=[self._moving_collider()])
        state = solver.initial_state()
        checkpoints = {}
        for frame in range(1, 7):
            for substep in range(solver.substeps):
                state = solver.step(state, frame, substep, 0)
            checkpoints[frame] = solver.checkpoint(state)
        resumed = solver.restore(checkpoints[3])
        for frame in range(4, 7):
            for substep in range(solver.substeps):
                resumed = solver.step(resumed, frame, substep, 0)
        self.assertEqual(resumed, state)

    def test_seeded_source_noise_is_repeatable_and_seed_dependent(self):
        def run(seed):
            source = fluid3d.Source("sphere", center=(10, 5, 10), radius=4, noise_amount=0.8, noise_scale=3.0, seed=seed)
            solver = fluid3d.Smoke3D(SMALL, sources=[source])
            state = solver.initial_state()
            for frame in range(1, 4):
                state = solver.step(state, frame, 0, seed)
            return state.arrays["density"]
        self.assertTrue(np.array_equal(run(1), run(1)))
        self.assertFalse(np.array_equal(run(1), run(2)))


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_and_restore_are_exact_and_independent(self):
        state, solver = solve(SMALL, 5)
        saved = solver.checkpoint(state)
        self.assertEqual(saved, state)
        state.arrays["density"][...] = -1.0
        self.assertNotEqual(saved, state)                       # a checkpoint never shares memory with the solver
        restored = solver.restore(saved)
        self.assertEqual(restored, saved)
        restored.arrays["u"][...] = 9.0
        self.assertNotEqual(restored.arrays["u"].max(), saved.arrays["u"].max())

    def test_a_checkpoint_survives_the_disk_cache(self):
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            params = SMALL
            cache = simcache.SimCache(root=root, memory_budget=1)       # everything spills to disk
            solver = fluid3d.Smoke3D(params)
            run = simcache.run_key(None, params)
            first = simcache.solve_to_frame(cache, run, 4, 1, 1, 0, solver.initial_state, solver.step)
            fresh = simcache.SimCache(root=root)
            stored = fresh.get(run, 4)
            self.assertEqual(stored, first)


class CancellationTests(unittest.TestCase):
    def test_a_cancel_during_the_pressure_solve_returns_promptly_and_keeps_banked_frames(self):
        cancel = threading.Event()
        params = {"nx": 48, "ny": 48, "nz": 48, "tolerance": 1e-9, "max_iterations": 5000}
        solver = fluid3d.Smoke3D(params, cancel=cancel)
        cache = simcache.SimCache(enabled=False)
        run = simcache.run_key(None, params)
        simcache.solve_to_frame(cache, run, 2, 1, 1, 0, solver.initial_state, solver.step, cancel)
        threading.Timer(0.3, cancel.set).start()
        started = time.perf_counter()
        with self.assertRaises(Cancelled):
            simcache.solve_to_frame(cache, run, 5, 1, 1, 0, solver.initial_state, solver.step, cancel)
        self.assertLess(time.perf_counter() - started, 3.0)
        self.assertIsNotNone(cache.get(run, 2))

    def test_a_set_cancel_stops_before_the_first_substep(self):
        cancel = threading.Event()
        cancel.set()
        solver = fluid3d.Smoke3D(SMALL, cancel=cancel)
        with self.assertRaises(Cancelled):
            solver.step(solver.initial_state(), 1, 0, 0)


class ValidationTests(unittest.TestCase):
    def test_bad_grid_and_bad_advection_are_refused(self):
        with self.assertRaises(ValueError):
            fluid3d.Smoke3D({"nx": 3})
        with self.assertRaises(ValueError):
            fluid3d.Smoke3D({"advection": "bfecc"})


class SourceMotionTests(unittest.TestCase):
    def test_a_moving_source_geometry_hands_its_motion_to_the_emitted_velocity(self):
        def provider(frame):
            return box_triangles((8.0 + 0.5 * frame, 4.0, 8.0), (12.0 + 0.5 * frame, 8.0, 12.0))
        track = fluid3d.GeometryTrack(provider, animated=True, start_frame=1)
        source = fluid3d.Source("volume", track=track, inherit_velocity=1.0, density=1.0)
        solver = fluid3d.Smoke3D({**SMALL, "default_source": 0, "vorticity": 0.0, "buoyancy_temperature": 0.0},
                                 sources=[source])
        flat, weight, motion = source.footprint(solver, 3)
        self.assertGreater(len(flat), 30)
        np.testing.assert_allclose(motion[:, 0].mean(), 0.5, atol=1e-6)
        np.testing.assert_allclose(motion[:, 1:], 0.0, atol=1e-6)
        state = solver.step(solver.initial_state(), 3, 0, 0)
        self.assertGreater(float(state.arrays["u"][8:14, 4:8, 8:12].max()), 0.1)   # air is dragged along +x

    def test_a_static_source_geometry_has_no_motion(self):
        tri = box_triangles((8.0, 4.0, 8.0), (12.0, 8.0, 12.0))
        track = fluid3d.GeometryTrack(lambda frame: tri)
        np.testing.assert_array_equal(track.motion(5), np.zeros((12, 3)))


class ForceTests(unittest.TestCase):
    def solver(self, force, **params):
        p = {**SMALL, "default_source": 0, "vorticity": 0.0, "buoyancy_temperature": 0.0, "buoyancy_density": 0.0}
        return fluid3d.Smoke3D({**p, **params}, forces=[force])

    def test_gravity_and_wind_accelerate_along_their_direction(self):
        force = fluid3d.Force("wind", {"dir_x": 1.0, "dir_y": 0.0, "dir_z": 0.0, "strength": 0.2})
        solver = self.solver(force, boundary_x="open")
        state = solver.initial_state()
        for frame in range(1, 4):
            state = solver.step(state, frame, 0, 0)
        self.assertGreater(float(state.arrays["u"][10, 10, 10]), 0.3)
        self.assertLess(abs(float(state.arrays["v"][10, 10, 10])), 1e-3)

    def test_gravity_weighs_the_smoke_and_does_not_move_clean_air(self):
        force = fluid3d.Force("gravity", {"dir_x": 0.0, "dir_y": -1.0, "dir_z": 0.0, "strength": 0.3})
        solver = self.solver(force)
        state = solver.initial_state()
        self.assertEqual(float(np.abs(solver.step(state, 1, 0, 0).arrays["v"]).max()), 0.0)   # no smoke, no weight
        state.arrays["density"][8:12, 12:16, 8:12] = 1.0
        after = solver.step(state, 1, 0, 0)
        self.assertLess(float(after.arrays["v"][8:12, 12:16, 8:12].mean()), -0.05)

    def test_drag_slows_the_flow(self):
        force = fluid3d.Force("drag", {"drag": 0.5})
        solver = self.solver(force, boundary_x="open")
        state = solver.initial_state()
        state.arrays["u"][:] = 1.0
        after = solver.step(state, 1, 0, 0).arrays["u"][10, 10, 10]
        self.assertAlmostEqual(float(after), float(np.exp(-0.5)), delta=0.02)

    def test_turbulence_is_divergence_free_before_projection_and_repeatable(self):
        force = fluid3d.Force("turbulence", {"strength": 0.3, "turbulence_scale": 4.0, "turbulence_speed": 0.1}, seed=5)
        a = force._turbulence((20, 28, 20), 4.0, 2.4)
        b = fluid3d.Force("turbulence", {}, seed=5)._turbulence((20, 28, 20), 4.0, 2.4)
        self.assertTrue(np.array_equal(a, b))
        self.assertGreater(float(np.abs(a).max()), 0.1)
        c = fluid3d.Force("turbulence", {}, seed=6)._turbulence((20, 28, 20), 4.0, 2.4)
        self.assertFalse(np.array_equal(a, c))
        div = np.gradient(a[0], axis=0) + np.gradient(a[1], axis=1) + np.gradient(a[2], axis=2)
        self.assertLess(float(np.abs(div[2:-2, 2:-2, 2:-2]).max()), 0.35 * float(np.abs(a).max()))

    def test_force_frame_range_is_honoured(self):
        force = fluid3d.Force("wind", {"dir_x": 1.0, "dir_y": 0.0, "dir_z": 0.0, "strength": 0.2,
                                       "from_frame": 5, "to_frame": 9})
        solver = self.solver(force, boundary_x="open")
        state = solver.step(solver.initial_state(), 1, 0, 0)
        self.assertEqual(float(np.abs(state.arrays["u"]).max()), 0.0)


def _high_frequency_energy(field):
    """Sum of the power spectrum outside the inner half of each axis' frequency range: a coarse
    "how much fine-grained structure" measure that does not care about phase."""
    spectrum = np.fft.fftn(field.astype(np.float64))
    power = np.abs(spectrum) ** 2
    freqs = [np.fft.fftfreq(n) for n in field.shape]
    radius = np.sqrt(sum(g * g for g in np.meshgrid(*freqs, indexing="ij")))
    return float(power[radius > 0.2].sum())


class ShapeControlTests(unittest.TestCase):
    """Lane 6, Pyro production step 2: dissipation's control field, disturbance, shredding, turbulence and
    confinement (docs/FLUIDS_SPIKE.md "Shape controls"). `vorticity` is confinement's own param name."""

    PARAMS = {**SMALL, "vorticity": 0.0, "buoyancy_temperature": 0.0, "buoyancy_density": 0.0}

    def plume(self, frames=8, **extra):
        solver = fluid3d.Smoke3D({**self.PARAMS, **extra})
        state = solver.initial_state()
        for frame in range(1, frames + 1):
            state = solver.step(state, frame, 0, 0)
        return solver, state

    def test_each_control_at_zero_equals_off(self):
        _, base = self.plume()
        for name in ("disturbance", "shredding", "turbulence", "dissipation"):
            _, state = self.plume(**{name: 0.0})
            self.assertTrue(np.array_equal(state.arrays["density"], base.arrays["density"]), name)
            self.assertTrue(np.array_equal(state.arrays["u"], base.arrays["u"]), name)

    def test_disturbance_raises_high_frequency_energy(self):
        # Grow a plume with no disturbance, then compare one more substep with it on and off: density lags a
        # substep behind velocity (semi-Lagrangian advection reads the old field), so the kicks show up in the
        # velocity spectrum straight away.
        base, state = self.plume(frames=15, disturbance=0.0)
        off = fluid3d.Smoke3D({**self.PARAMS, "disturbance": 0.0})
        on = fluid3d.Smoke3D({**self.PARAMS, "disturbance": 6.0, "disturbance_size": 2.0})
        after_off = off.step(state, 16, 0, 0)
        after_on = on.step(state, 16, 0, 0)
        self.assertGreater(_high_frequency_energy(after_on.arrays["u"]),
                            10.0 * _high_frequency_energy(after_off.arrays["u"]))

    def test_disturbance_field_limits_the_effect_to_where_the_field_is_in_range(self):
        solver = fluid3d.Smoke3D({**self.PARAMS, "disturbance": 5.0, "disturbance_size": 4.0,
                                  "disturbance_field": "temperature", "disturbance_range_lo": 0.5,
                                  "disturbance_range_hi": 1000.0, "disturbance_ramp": 0.0})
        a = {name: array.copy() for name, array in solver.initial_state().arrays.items()}
        a["temperature"][:] = 0.0
        a["temperature"][10:, :, :] = 1.0                 # a hot half and a cool half
        solver._disturb(a, frame=3, substep=0, dt=solver.dt)
        # a buffer of one cell either side of the boundary, since add_cell_force spreads a cell's kick to
        # both of its faces
        self.assertEqual(float(np.abs(a["u"][:9]).max()), 0.0)
        self.assertGreater(float(np.abs(a["u"][11:]).max()), 0.0)

    def test_shredding_stretches_a_3d_flow_and_is_exactly_zero_for_a_flat_one(self):
        solver = fluid3d.Smoke3D({**self.PARAMS, "shredding": 2.0})
        a = {name: array.copy() for name, array in solver.initial_state().arrays.items()}

        def axes(shape):
            i = np.arange(shape[0])[:, None, None]
            j = np.arange(shape[1])[None, :, None]
            k = np.arange(shape[2])[None, None, :]
            return i, j, k

        # a helical (ABC-like) flow: real 3D structure, genuine vortex stretching
        for name, fn in (("u", lambda i, j, k, s: np.sin(k * 2 * np.pi / s[2]) + np.cos(j * 2 * np.pi / s[1])),
                         ("v", lambda i, j, k, s: np.sin(i * 2 * np.pi / s[0]) + np.cos(k * 2 * np.pi / s[2])),
                         ("w", lambda i, j, k, s: np.sin(j * 2 * np.pi / s[1]) + np.cos(i * 2 * np.pi / s[0]))):
            shape = a[name].shape
            a[name][:] = fn(*axes(shape), shape)
        before = {name: a[name].copy() for name in ("u", "v", "w")}
        solver._shred(a, solver.dt)
        for name in ("u", "v", "w"):
            self.assertGreater(float(np.abs(a[name] - before[name]).max()), 0.0, name)

        flat = {name: array.copy() for name, array in solver.initial_state().arrays.items()}
        for name in ("u", "v", "w"):                      # depends only on i and j: no z-variation at all
            shape = flat[name].shape
            i, j, _ = axes(shape)
            flat[name][:] = 0.05 * j + 0.0 * i
        before_flat = {name: flat[name].copy() for name in ("u", "v", "w")}
        solver._shred(flat, solver.dt)
        for name in ("u", "v", "w"):
            self.assertEqual(float(np.abs(flat[name] - before_flat[name]).max()), 0.0, name)

    def test_turbulence_is_seeded_and_repeatable(self):
        _, a = self.plume(turbulence=0.5, swirl_size=3.0, seed=7)
        _, b = self.plume(turbulence=0.5, swirl_size=3.0, seed=7)
        _, c = self.plume(turbulence=0.5, swirl_size=3.0, seed=8)
        self.assertTrue(np.array_equal(a.arrays["density"], b.arrays["density"]))
        self.assertFalse(np.array_equal(a.arrays["density"], c.arrays["density"]))

    def test_turbulence_field_limits_the_effect_to_where_the_field_is_in_range(self):
        solver = fluid3d.Smoke3D({**self.PARAMS, "turbulence": 5.0, "swirl_size": 3.0,
                                  "turbulence_field": "density", "turbulence_range_lo": 0.5,
                                  "turbulence_range_hi": 1000.0, "turbulence_ramp": 0.0})
        a = {name: array.copy() for name, array in solver.initial_state().arrays.items()}
        a["density"][:] = 0.0
        a["density"][10:, :, :] = 1.0
        solver._shape_turbulence(a, frame=3, substep=0, dt=solver.dt)
        self.assertEqual(float(np.abs(a["u"][:9]).max()), 0.0)
        self.assertGreater(float(np.abs(a["u"][11:]).max()), 0.0)

    def test_confinement_preserves_more_vorticity_over_50_frames_than_none(self):
        def run(vorticity):
            solver = fluid3d.Smoke3D({**SMALL, "vorticity": vorticity})   # buoyancy on: a real plume with real vorticity
            state = solver.initial_state()
            for frame in range(1, 51):
                state = solver.step(state, frame, 0, 0)
            wx, wy, wz = solver._curl(*solver._cell_velocity(state.arrays))
            return float(np.sum(wx * wx + wy * wy + wz * wz))

        self.assertGreater(run(1.0), run(0.0))

    def test_determinism(self):
        params = {**self.PARAMS, "disturbance": 1.0, "disturbance_size": 3.0, "shredding": 0.5,
                  "turbulence": 0.5, "swirl_size": 2.0,
                  "dissipation": 0.1, "dissipation_field": "temperature", "dissipation_range_lo": 0.2,
                  "dissipation_range_hi": 1000.0}
        _, a = self.plume(**params)
        _, b = self.plume(**params)
        self.assertTrue(np.array_equal(a.arrays["density"], b.arrays["density"]))
        self.assertTrue(np.array_equal(a.arrays["u"], b.arrays["u"]))


class ShapeControlGpuParityTests(unittest.TestCase):
    """The wgpu pressure hook (nodebased/fluid_gpu3d.py) still agrees with the CPU CG reference once the
    shape controls above are active: they run in this same Python step regardless of which pressure
    backend resolves the projection, so "GPU" here means the pressure solve, not the shaping."""

    def gpu(self):
        from nodebased import fluid_gpu3d
        try:
            return fluid_gpu3d.GpuPressure3D()
        except Exception as error:
            self.skipTest(f"no wgpu adapter: {error}")

    def test_shape_controls_still_meet_the_tolerance_on_the_gpu_pressure_hook(self):
        gpu = self.gpu()
        params = {"nx": 20, "ny": 28, "nz": 20, "disturbance": 1.0, "disturbance_size": 3.0,
                  "shredding": 0.3, "turbulence": 0.4, "swirl_size": 2.5,
                  "dissipation": 0.15, "dissipation_field": "density", "dissipation_range_lo": 0.2,
                  "dissipation_range_hi": 1000.0}
        solver = fluid3d.Smoke3D(params)
        state = solver.initial_state()
        for frame in range(1, 6):
            state = solver.step(state, frame, 0, 0)
        reference = solver.step(state, 6, 0, 0)
        solver.pressure_solver = gpu.solve
        other = solver.step(state, 6, 0, 0)
        div = divergence_of(other)
        self.assertLessEqual(float(np.abs(div).max()), 1e-3 + 1e-5)
        for name in ("u", "v", "w"):
            self.assertLess(float(np.abs(reference.arrays[name] - other.arrays[name]).max()), 5e-3)


class GpuPressureParityTests(unittest.TestCase):
    """The wgpu red-black pressure solve (nodebased/fluid_gpu3d.py) against the NumPy reference."""

    def gpu(self):
        from nodebased import fluid_gpu3d
        try:
            return fluid_gpu3d.GpuPressure3D()
        except Exception as error:                    # no wgpu or no adapter on this machine
            self.skipTest(f"no wgpu adapter: {error}")

    def test_gpu_pressure_solve_meets_the_tolerance_and_matches_numpy_with_a_solid_and_an_open_top(self):
        gpu = self.gpu()
        params = {"nx": 24, "ny": 32, "nz": 24, "boundary_y": "open"}
        collider = box_collider((6.0, 10.0, 6.0), (14.0, 14.0, 14.0))
        solver = fluid3d.Smoke3D(params, colliders=[collider])
        state = solver.initial_state()
        for frame in range(1, 9):
            state = solver.step(state, frame, 0, 0)
        reference = solver.step(state, 9, 0, 0)
        solver.pressure_solver = gpu.solve
        other = solver.step(state, 9, 0, 0)
        solid, _, _ = solver._solid_for(9)
        div = divergence_of(other)
        self.assertLessEqual(float(np.abs(div[~solid]).max()), 1e-3 + 1e-5)
        for name in ("u", "v", "w"):
            self.assertLess(float(np.abs(reference.arrays[name] - other.arrays[name]).max()), 5e-3)

    def test_gpu_closed_box_matches_too(self):
        gpu = self.gpu()
        solver = fluid3d.Smoke3D({"nx": 20, "ny": 24, "nz": 20})
        state = solver.initial_state()
        for frame in range(1, 7):
            state = solver.step(state, frame, 0, 0)
        reference = solver.step(state, 7, 0, 0)
        solver.pressure_solver = gpu.solve
        other = solver.step(state, 7, 0, 0)
        self.assertLessEqual(float(np.abs(divergence_of(other)).max()), 1e-3 + 1e-5)
        for name in ("u", "v", "w"):
            self.assertLess(float(np.abs(reference.arrays[name] - other.arrays[name]).max()), 5e-3)


if __name__ == "__main__":
    unittest.main()
