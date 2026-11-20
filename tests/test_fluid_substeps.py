"""Fluids 18 C: the automatic substep rule of FluidSolver3D and the backend threshold (docs/FLUIDS_SPIKE.md, "Fluids 18 C").

substeps = ceil(fastest speed in cells per frame / target), at least the node's own Substeps (the floor), at most the cap (unless the floor is
higher). A node saved before the rule (schema 39) keeps its stored substeps through `auto_substeps` 0 and bakes the same pixels and the same cache.
"""
from __future__ import annotations

import copy
import json
import unittest

import numpy as np

from nodebased import fluid3d, simcache
from nodebased.core import SCHEMA_VERSION, SPECS, Dispatcher, upgrade_document, validate
from nodebased.imaging import Evaluator
from tests.test_fluid3d_nodes import GRID, at
from tests.test_particles_nodes import make, set_, wire


def smoke(**params):
    base = {"nx": 12, "ny": 16, "nz": 12, "default_source": 1, "auto_substeps": 1, "boundary_y": "open"}
    return fluid3d.Smoke3D({**base, **params})


class RuleTests(unittest.TestCase):
    def test_still_air_takes_one_substep(self):
        self.assertEqual(fluid3d.substeps_for_speed(0.0), 1)
        self.assertEqual(fluid3d.substeps_for_speed(1.9), 1)
        self.assertEqual(fluid3d.substeps_for_speed(2.0), 1)

    def test_a_fast_plume_gets_more_substeps_from_the_speed(self):
        self.assertEqual(fluid3d.substeps_for_speed(2.1), 2)
        self.assertEqual(fluid3d.substeps_for_speed(7.5), 4)
        self.assertEqual(fluid3d.substeps_for_speed(15.0), 8)
        self.assertEqual(fluid3d.substeps_for_speed(15.0, target=3.0), 5)

    def test_the_cap_holds(self):
        self.assertEqual(fluid3d.substeps_for_speed(500.0), 8)
        self.assertEqual(fluid3d.substeps_for_speed(500.0, cap=5), 5)

    def test_the_explicit_substeps_is_a_floor_even_above_the_cap(self):
        self.assertEqual(fluid3d.substeps_for_speed(0.0, floor=3), 3)
        self.assertEqual(fluid3d.substeps_for_speed(15.0, floor=3), 8)
        self.assertEqual(fluid3d.substeps_for_speed(500.0, floor=12), 12)

    def test_the_solver_plans_each_frame_from_its_state(self):
        solver = smoke()
        still = solver.initial_state()
        self.assertEqual(solver.plan_frame(still, 1), 1)
        self.assertEqual(solver.dt, 1.0)
        fast = solver.initial_state()
        fast.arrays["v"][...] = 9.0
        self.assertEqual(solver.plan_frame(fast, 1), 5)
        self.assertAlmostEqual(solver.dt, 0.2)
        self.assertEqual(solver.substeps, 5)
        off = smoke(auto_substeps=0, substeps=2)
        self.assertEqual(off.plan_frame(fast, 1), 2)

    def test_an_emitter_speed_counts_before_the_first_step(self):
        solver = smoke(default_source=0)
        solver.sources = [fluid3d.Source("sphere", center=(6, 3, 6), radius=2.0, velocity=(0.0, 12.0, 0.0))]
        solver.voxel = 1.0
        self.assertEqual(solver.plan_frame(solver.initial_state(), 1), 6)

    def test_a_frame_from_a_checkpoint_matches_the_same_frame_in_a_run(self):
        def solve(target, cache, solver):
            return simcache.solve_to_frame(cache, "run", target, 1, 1, 0, solver.initial_state, solver.step, plan=solver.plan_frame)
        kwargs = dict(source_density=1.0, source_temperature=1.0, buoyancy_temperature=3.0)
        straight_solver = smoke(**kwargs)
        straight = solve(5, simcache.SimCache(), straight_solver)
        cache = simcache.SimCache()
        resumed_solver = smoke(**kwargs)
        solve(3, cache, resumed_solver)
        again = smoke(**kwargs)
        resumed = solve(5, cache, again)
        for name in ("density", "temperature", "u", "v", "w"):
            np.testing.assert_array_equal(straight.arrays[name], resumed.arrays[name], name)
        self.assertEqual(straight.meta["substeps_used"], resumed.meta["substeps_used"])
        self.assertGreater(straight.meta["substeps_used"], 1, "the buoyant plume is fast enough to need more than one substep")


def node_plume(**solver):
    d = Dispatcher()
    make(d, src=("FluidSource3D", {"src_center_y": 0.4, "src_radius": 0.3, "src_vel_y": 0.3}),
         sol=("FluidSolver3D", {**GRID, "auto_substeps": 1, **solver}), c=("FluidCache3D", {}))
    wire(d, "sol", "fluid", "src")
    wire(d, "c", "volume", "sol")
    return d


def solved_steps(d, frame, key="c"):
    before = fluid3d.SOLVER_STATS["steps"]
    volume = at(Evaluator(), d, key, frame)
    return volume, fluid3d.SOLVER_STATS["steps"] - before


class NodeTests(unittest.TestCase):
    def test_a_new_node_has_the_rule_on_and_an_old_one_off(self):
        spec = SPECS["FluidSolver3D"]["params"]
        self.assertEqual((spec["auto_substeps"], spec["substep_target"], spec["substep_cap"]), (1, 2.0, 8))
        d = node_plume()
        legacy = copy.deepcopy(d.document)
        legacy["version"] = 38
        for node in legacy["nodes"].values():
            if node["type"] == "FluidSolver3D":
                for name in ("auto_substeps", "substep_target", "substep_cap"):
                    del node["params"][name]
        upgraded = upgrade_document(json.loads(json.dumps(legacy)))
        validate(upgraded)
        self.assertEqual(upgraded["version"], SCHEMA_VERSION)
        params = upgraded["nodes"]["sol"]["params"]
        self.assertEqual((params["auto_substeps"], params["substep_target"], params["substep_cap"]), (0, 2.0, 8))

    def test_a_fast_scene_takes_more_steps_than_frames_and_obeys_the_cap(self):
        d = node_plume(substep_cap=3)
        _, taken = solved_steps(d, 4)
        self.assertGreater(taken, 4, "the emitter's speed asks for more than one substep a frame")
        self.assertLessEqual(taken, 4 * 3, "the cap holds")

    def test_still_air_takes_one_step_per_frame(self):
        d = Dispatcher()
        make(d, src=("FluidSource3D", {"src_center_y": 0.4, "src_radius": 0.3, "src_vel_y": 0.0, "src_temperature": 0.0, "src_density": 0.0}),
             sol=("FluidSolver3D", {**GRID, "auto_substeps": 1, "cooling_rate": 0.0}))
        wire(d, "sol", "fluid", "src")
        _, taken = solved_steps(d, 3, "sol")
        self.assertEqual(taken, 3)

    def test_the_explicit_substeps_keep_at_least_that_many(self):
        d = node_plume(substeps=3)
        _, taken = solved_steps(d, 3)
        self.assertGreaterEqual(taken, 3 * 3)

    def test_a_saved_scene_renders_the_same_pixels_and_cache_after_the_migration(self):
        d = node_plume(auto_substeps=0, substeps=2)
        reference = at(Evaluator(), d, "c", 4)
        legacy = json.loads(json.dumps(d.document))
        legacy["version"] = 38
        legacy["nodes"]["sol"]["params"].pop("auto_substeps")
        legacy["nodes"]["sol"]["params"].pop("substep_target")
        legacy["nodes"]["sol"]["params"].pop("substep_cap")
        upgraded = upgrade_document(legacy)
        migrated = Dispatcher()
        migrated.document = upgraded
        again = at(Evaluator(), migrated, "c", 4)
        np.testing.assert_array_equal(reference.density, again.density)
        np.testing.assert_array_equal(reference.velocity, again.velocity)
        before, after = (v.stream.run if hasattr(v, "stream") and v.stream is not None else None for v in (reference, again))
        self.assertEqual(before, after, "the legacy run keeps its cache key, so saved frames are not re-baked")
        _, taken = solved_steps(migrated, 5)
        self.assertEqual(taken, 5 * 2, "five frames of the stored 2 substeps")


try:
    from nodebased import fluid_gpu_solver as fgs
    HAVE_GPU = fgs.available()
except Exception:        # pragma: no cover - no wgpu on this machine
    HAVE_GPU = False


@unittest.skipUnless(HAVE_GPU, "no wgpu adapter")
class GpuTests(unittest.TestCase):
    def test_the_card_reads_the_same_speed_as_its_own_readback(self):
        params = {"nx": 16, "ny": 24, "nz": 16, "default_source": 1, "auto_substeps": 1, "boundary_y": "open",
                  "buoyancy_temperature": 4.0, "source_radius": 0.2}
        solver = fgs.GpuSmoke3D(params)
        state = solver.initial_state()
        for frame in (1, 2, 3):
            n = solver.plan_frame(state, frame)
            for sub in range(n):
                state = solver.step(state, frame, sub, 0)
            on_card = solver.max_face_speed(state)                      # reduced on the card, before any readback
            host = max(float(np.abs(state.arrays[name]).max()) for name in ("u", "v", "w"))
            self.assertAlmostEqual(on_card, host, places=5)
        self.assertGreater(on_card, 2.0, "the buoyant plume outruns two cells a substep")

    def test_the_card_and_the_processor_pick_the_same_substeps(self):
        params = {"nx": 16, "ny": 24, "nz": 16, "default_source": 1, "auto_substeps": 1, "boundary_y": "open",
                  "buoyancy_temperature": 4.0, "source_radius": 0.2}
        used = {}
        for name, make_solver in (("cpu", lambda: fluid3d.Smoke3D(params)), ("gpu", lambda: fgs.GpuSmoke3D(params))):
            solver = make_solver()
            state = solver.initial_state()
            counts = []
            for frame in (1, 2, 3, 4):
                n = solver.plan_frame(state, frame)
                for sub in range(n):
                    state = solver.step(state, frame, sub, 0)
                counts.append(n)
            used[name] = counts
        self.assertEqual(used["cpu"][0], used["gpu"][0])
        for a, b in zip(used["cpu"], used["gpu"]):
            self.assertLessEqual(abs(a - b), 1, used)


class BackendTests(unittest.TestCase):
    def test_the_automatic_choice_follows_the_threshold(self):
        params = dict(SPECS["FluidSolver3D"]["params"])
        threshold = fluid3d.GPU_AUTO_CELLS
        self.assertEqual(fluid3d.resolve_backend({**params, "pressure": "cpu"}, threshold * 10), "cpu")
        self.assertEqual(fluid3d.resolve_backend({**params, "pressure": "auto"}, 1000, (10, 10, 10)), "cpu")
        shape = (32, 32, 32)
        cells = threshold
        resolved = fluid3d.resolve_backend({**params, "pressure": "auto"}, cells, shape)
        self.assertIn(resolved, ("resident", "gpu", "cpu"))
        below = fluid3d.resolve_backend({**params, "pressure": "auto"}, threshold - 1, shape)
        self.assertEqual(below, "cpu")


if __name__ == "__main__":
    unittest.main()
