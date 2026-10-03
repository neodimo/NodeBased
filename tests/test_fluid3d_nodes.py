"""Lane L6 step C, part 2: FluidSource3D, FluidForce3D, FluidCollide3D, FluidSolver3D and FluidCache3D as graph nodes.

Registration, typed wiring, the solver's Volume output, forces and colliders through the graph, scrubbing without
re-solving (counted through `fluid3d.SOLVER_STATS`), invalidation, cancellation, budgets, restart from disk, bypass,
old documents, and a solved frame drawn by Render3D's raymarch.
"""
import json
import tempfile
import threading
import unittest

import numpy as np

from nodebased import fluid3d, scene3d, simcache
from nodebased.cancellation import Cancelled
from nodebased.core import (Dispatcher, INPUT_TYPES, OUTPUT_TYPES, SPECS, bypass_slot, upgrade_document,
                            validate)
from nodebased.imaging import Evaluator
from nodebased.knobs import knob_layout
from nodebased.theme import COLORS
from nodebased.tiers import REGION_RULES
from tests.test_particles_nodes import make, set_, wire
from tests.waiting import pause

KINDS = ("FluidSource3D", "FluidForce3D", "FluidCollide3D", "FluidSolver3D", "FluidCache3D")
# a small, quick grid: 16 x 24 x 16 cells of 0.125
GRID = {"division_size": 0.125, "bounds_min_x": -1.0, "bounds_min_y": 0.0, "bounds_min_z": -1.0,
        "bounds_max_x": 1.0, "bounds_max_y": 3.0, "bounds_max_z": 1.0, "boundary_y": "closed",
        "cooling_rate": 0.0, "auto_resize": 0}


def at(evaluator, d, key, frame, **kwargs):
    return evaluator.evaluate_raster(d.document, key, frame=frame, typed=True, **kwargs)


def set_key(d, key, param, frame, value):
    d.execute({"op": "set_key", "id": key, "param": param, "frame": frame, "value": value})


def steps():
    return fluid3d.SOLVER_STATS["steps"]


def plume(**solver):
    """Source -> Solver (-> Cache), the smallest working graph."""
    d = Dispatcher()
    make(d, src=("FluidSource3D", {"src_center_y": 0.4, "src_radius": 0.3}),
         sol=("FluidSolver3D", {**GRID, **solver}), c=("FluidCache3D", {}))
    wire(d, "sol", "fluid", "src")
    wire(d, "c", "volume", "sol")
    return d


def height(volume):
    return fluid3d.centre_of_mass(volume.density)[1] * volume.voxel_size


class RegistrationTests(unittest.TestCase):
    def test_kinds_are_registered_everywhere(self):
        for kind in KINDS:
            self.assertIn(kind, SPECS)
            self.assertIn(kind, COLORS)
            self.assertIn(kind, REGION_RULES)
            laid_out = sorted(p for group in knob_layout(kind) for p in group.params)
            self.assertEqual(laid_out, sorted(SPECS[kind]["params"]), kind)
        for kind in ("FluidSource3D", "FluidForce3D", "FluidCollide3D"):
            self.assertEqual(OUTPUT_TYPES[kind], "fluid")
        for kind in ("FluidSolver3D", "FluidCache3D"):
            self.assertEqual(OUTPUT_TYPES[kind], "volume")
        self.assertEqual(SPECS["FluidSource3D"]["inputs"], [])
        self.assertEqual(SPECS["FluidSource3D"]["optional_inputs"], ["geo"])
        self.assertEqual(SPECS["FluidForce3D"]["inputs"], ["fluid"])
        self.assertEqual(SPECS["FluidCollide3D"]["optional_inputs"], ["geometry"])
        self.assertEqual(SPECS["FluidSolver3D"]["inputs"], ["fluid"])
        self.assertEqual(SPECS["FluidCache3D"]["inputs"], ["volume"])
        self.assertEqual(INPUT_TYPES["fluid"], ("fluid",))
        self.assertEqual(INPUT_TYPES["volume"], ("volume",))

    def test_houdini_pyro_and_nuke_knob_vocabulary_is_present(self):
        source = SPECS["FluidSource3D"]["params"]
        for name in ("fluid_emit_from", "src_radius", "src_falloff", "src_density", "src_temperature", "src_fuel",
                     "src_vel_x", "src_inherit_velocity", "src_noise_amount", "src_noise_scale", "start_frame",
                     "end_frame", "tx", "rx", "sx", "uscale", "pivot_x"):
            self.assertIn(name, source)
        solver = SPECS["FluidSolver3D"]["params"]
        for name in ("division_size", "bounds_min_x", "bounds_max_z", "start_frame", "substeps", "seed", "advection",
                     "vorticity", "dissipation", "cooling_rate", "boundary_x", "boundary_y", "boundary_z",
                     "tolerance", "max_iterations", "pressure", "fire", "ignition_temperature", "burn_rate",
                     "burn_heat", "burn_expansion"):
            self.assertIn(name, solver)
        for name in ("force_kind", "buoyancy_lift", "ambient_temperature", "strength", "turbulence_scale",
                     "turbulence_speed", "drag"):
            self.assertIn(name, SPECS["FluidForce3D"]["params"])
        for name in ("cache_memory_mb", "cache_disk_mb", "cache_precision", "cache_channels"):
            self.assertIn(name, SPECS["FluidCache3D"]["params"])
        self.assertIn("animated", SPECS["FluidCollide3D"]["params"])

    def test_typed_wiring(self):
        d = Dispatcher()
        make(d, src=("FluidSource3D", {}), f=("FluidForce3D", {}), col=("FluidCollide3D", {}),
             sol=("FluidSolver3D", {}), c=("FluidCache3D", {}), s=("Scene3D", {}), a=("Axis3D", {}),
             card=("Card3D", {}), g=("Grade", {}), p=("Plume3D", {}), cam=("Camera3D", {}))
        wire(d, "f", "fluid", "src")
        wire(d, "col", "fluid", "f")
        wire(d, "col", "geometry", "card")
        wire(d, "sol", "fluid", "col")
        wire(d, "c", "volume", "sol")
        wire(d, "s", "object0", "sol")
        wire(d, "s", "object1", "c")
        wire(d, "a", "object", "c")
        wire(d, "src", "geo", "card")
        wire(d, "c", "volume", "p")                              # a Plume3D is a volume too
        validate(d.document)
        for target, slot, source in (("sol", "fluid", "card"), ("sol", "fluid", "c"), ("c", "volume", "src"),
                                     ("g", "image", "sol"), ("f", "fluid", "p"), ("src", "geo", "sol")):
            with self.assertRaises(ValueError, msg=(target, slot, source)):
                wire(d, target, slot, source)

    def test_nodes_are_rounded_3d_nodes(self):
        from nodebased.app import node_form
        for kind in KINDS:
            self.assertEqual(node_form({"type": kind}), "round", kind)

    def test_defaults_validate_and_survive_a_json_round_trip(self):
        d = plume()
        reloaded = upgrade_document(json.loads(json.dumps(d.document)))
        validate(reloaded)
        self.assertEqual(reloaded["nodes"]["sol"]["params"], {**SPECS["FluidSolver3D"]["params"], **GRID})

    def test_the_default_solver_grid_and_the_derived_resolution(self):
        self.assertEqual(fluid3d.resolution(SPECS["FluidSolver3D"]["params"]), (20, 30, 20))
        self.assertEqual(fluid3d.resolution({**SPECS["FluidSolver3D"]["params"], "division_size": 0.3}), (7, 10, 7))
        self.assertEqual(fluid3d.resolution({**SPECS["FluidSolver3D"]["params"], "division_size": 50.0}), (4, 4, 4))
        with self.assertRaisesRegex(ValueError, "bounds max must be above"):
            fluid3d.resolution({**SPECS["FluidSolver3D"]["params"], "bounds_max_y": -1.0})

    def test_a_grid_past_the_cpu_cap_is_refused_with_the_numbers(self):
        d = plume(division_size=0.001)
        with self.assertRaisesRegex(ValueError, r"cells; the CPU reference solver stops at 16,777,216"):
            at(Evaluator(), d, "sol", 3)

    def test_bypass_slots(self):
        def node(kind, **wired):
            slots = SPECS[kind]["inputs"] + SPECS[kind].get("optional_inputs", [])
            return {"type": kind, "inputs": {slot: wired.get(slot) for slot in slots}}
        self.assertIsNone(bypass_slot(node("FluidSource3D", geo="g")))
        self.assertEqual(bypass_slot(node("FluidForce3D", fluid="a")), "fluid")
        self.assertEqual(bypass_slot(node("FluidCollide3D", fluid="a", geometry="g")), "fluid")
        self.assertEqual(bypass_slot(node("FluidSolver3D", fluid="a")), "fluid")
        self.assertEqual(bypass_slot(node("FluidCache3D", volume="v")), "volume")
        with self.assertRaises(ValueError):
            d = plume()
            d.execute({"op": "disable", "id": "src", "value": True})


class SolverNodeTests(unittest.TestCase):
    def test_output_is_a_volume_on_the_solver_grid_with_world_unit_velocity(self):
        d = plume()
        volume = at(Evaluator(), d, "sol", 6)
        self.assertIsInstance(volume, scene3d.Volume)
        self.assertEqual(volume.shape, (16, 24, 16))
        self.assertEqual(volume.voxel_size, 0.125)
        self.assertEqual(volume.origin, (-1.0, 0.0, -1.0))
        self.assertEqual(volume.velocity.shape, (16, 24, 16, 3))
        self.assertGreater(float(volume.density.max()), 0.1)
        self.assertGreater(float(volume.temperature.max()), 0.1)
        self.assertGreater(float(volume.velocity[..., 1].max()), 0.0)       # the plume moves up
        state = fluid3d.solve_frame(volume.stream, 6, simcache.SimCache(enabled=False))
        cells_per_frame = 0.5 * (state.arrays["v"][:, :-1] + state.arrays["v"][:, 1:])
        np.testing.assert_allclose(volume.velocity[..., 1], cells_per_frame * 0.125 * 24.0, rtol=1e-5, atol=1e-6)

    def test_the_plume_rises_over_time(self):
        d = plume()
        ev = Evaluator()
        early, late = at(ev, d, "sol", 4), at(ev, d, "sol", 24)
        self.assertGreater(height(late), height(early) + 0.15)

    def test_frames_before_the_start_frame_are_empty(self):
        d = plume(start_frame=5)
        ev = Evaluator()
        self.assertEqual(float(at(ev, d, "sol", 3).density.max()), 0.0)
        self.assertGreater(float(at(ev, d, "sol", 8).density.max()), 0.0)

    def test_same_graph_same_volume_across_fresh_evaluators_and_scrub_versus_jump(self):
        d = plume(substeps=2, seed=3)
        jumped = at(Evaluator(), d, "sol", 9)
        ev = Evaluator()
        for frame in range(1, 10):
            scrubbed = at(ev, d, "sol", frame)
        self.assertEqual(jumped.fingerprint(), scrubbed.fingerprint())
        self.assertEqual(at(Evaluator(), d, "sol", 9).fingerprint(), jumped.fingerprint())

    def test_solver_knobs_and_source_knobs_change_the_run(self):
        base = at(Evaluator(), plume(), "sol", 6)
        for change in ({"vorticity": 0.0}, {"substeps": 2}, {"advection": "semi_lagrangian"},
                       {"boundary_x": "open"}, {"cooling_rate": 0.2}, {"disturbance": 1.0},
                       {"shredding": 0.5}, {"turbulence": 0.4},
                       {"dissipation": 0.15, "dissipation_field": "density"}):
            other = at(Evaluator(), plume(**change), "sol", 6)
            self.assertNotEqual(other.stream.run, base.stream.run, change)
            self.assertNotEqual(other.fingerprint(), base.fingerprint(), change)
        d = plume()
        set_(d, "src", src_density=2.0)
        self.assertNotEqual(at(Evaluator(), d, "sol", 6).stream.run, base.stream.run)

    def test_shape_controls_reach_the_cpu_solver_through_the_graph(self):
        # Lane 6, Pyro production step 2 (docs/FLUIDS_SPIKE.md "Shape controls"): FluidStream.solver() must
        # carry these knobs from the resolved node params into the Smoke3D it builds, not just into the run
        # identity above.
        quiet = at(Evaluator(), plume(disturbance=0.0), "sol", 4)
        kicked = at(Evaluator(), plume(disturbance=6.0, disturbance_size=2.0), "sol", 4)
        self.assertFalse(np.array_equal(quiet.velocity, kicked.velocity))

    def test_the_resident_gpu_solver_accepts_an_active_shape_control(self):
        from nodebased import fluid_gpu_solver
        if not fluid_gpu_solver.available():
            self.skipTest("no wgpu compute adapter")
        d = plume(pressure="resident", disturbance=1.0)
        shaped = at(Evaluator(), d, "sol", 3)
        self.assertGreater(float(shaped.density.sum()), 0.0)
        # the same grid and backend with every shape control at its default still solves
        at(Evaluator(), plume(pressure="resident"), "sol", 3)

    def test_a_disabled_solver_contributes_nothing(self):
        d = plume()
        d.execute({"op": "disable", "id": "sol", "value": True})
        self.assertIsNone(at(Evaluator(), d, "sol", 4))
        self.assertIsNone(at(Evaluator(), d, "c", 4))

    def test_open_top_lets_smoke_leave(self):
        closed = at(Evaluator(), plume(), "sol", 60)
        opened = at(Evaluator(), plume(boundary_y="open"), "sol", 60)
        self.assertLess(float(opened.density.sum()), 0.85 * float(closed.density.sum()))

    def test_fire_ignites_and_carries_a_flame_channel(self):
        d = Dispatcher()
        make(d, src=("FluidSource3D", {"src_center_y": 0.4, "src_radius": 0.3, "src_fuel": 1.0}),
             sol=("FluidSolver3D", {**GRID, "fire": 1}))
        wire(d, "sol", "fluid", "src")
        volume = at(Evaluator(), d, "sol", 10)
        self.assertGreater(float(volume.flame.max()), 0.0)
        cold = Dispatcher()
        make(cold, src=("FluidSource3D", {"src_center_y": 0.4, "src_radius": 0.3, "src_fuel": 1.0}),
             sol=("FluidSolver3D", {**GRID, "fire": 1, "ignition_temperature": 99.0}))
        wire(cold, "sol", "fluid", "src")
        self.assertEqual(float(at(Evaluator(), cold, "sol", 10).flame.max()), 0.0)


class SourceNodeTests(unittest.TestCase):
    def solved(self, **source):
        d = Dispatcher()
        make(d, src=("FluidSource3D", source), sol=("FluidSolver3D", {**GRID}))
        wire(d, "sol", "fluid", "src")
        return d

    def footprint(self, d, frame=1):
        volume = at(Evaluator(), d, "sol", frame)
        solver = volume.stream.solver()
        return solver.sources[0].footprint(solver, frame)

    def test_point_and_sphere_footprints(self):
        point = self.footprint(self.solved(fluid_emit_from="point", src_center_y=1.0))
        self.assertEqual(len(point[0]), 1)
        sphere = self.footprint(self.solved(fluid_emit_from="sphere", src_center_y=1.0, src_radius=0.5, src_falloff=0.0))
        expected = 4.0 / 3.0 * np.pi * 4 ** 3                                   # radius 4 cells
        self.assertAlmostEqual(len(sphere[0]) / expected, 1.0, delta=0.15)
        self.assertTrue(np.all(sphere[1] == 1.0))
        soft = self.footprint(self.solved(fluid_emit_from="sphere", src_center_y=1.0, src_radius=0.5, src_falloff=1.0))
        self.assertLess(float(soft[1].min()), 0.2)                                # falls off to the rim

    def test_the_transform_block_moves_the_source(self):
        a = at(Evaluator(), self.solved(src_center_y=1.0), "sol", 3)
        b = at(Evaluator(), self.solved(src_center_y=1.0, tx=0.5, ty=0.25), "sol", 3)
        ca, cb = fluid3d.centre_of_mass(a.density), fluid3d.centre_of_mass(b.density)
        self.assertAlmostEqual((cb[0] - ca[0]) * 0.125, 0.5, delta=0.05)
        self.assertAlmostEqual((cb[1] - ca[1]) * 0.125, 0.25, delta=0.05)

    def test_start_and_end_frame_window_the_emission(self):
        d = self.solved(src_center_y=1.0, start_frame=3, end_frame=5)
        ev = Evaluator()
        totals = [float(at(ev, d, "sol", f).density.sum()) for f in (2, 3, 5, 6, 9)]
        self.assertEqual(totals[0], 0.0)
        self.assertGreater(totals[1], 0.0)
        self.assertGreater(totals[2], totals[1])
        self.assertAlmostEqual(totals[3], totals[2], delta=0.05 * totals[2])     # emission stopped after frame 5
        self.assertAlmostEqual(totals[4], totals[2], delta=0.05 * totals[2])

    def test_noise_is_seeded_and_changes_the_emission(self):
        plain = at(Evaluator(), self.solved(src_center_y=1.0, src_radius=0.5), "sol", 3)
        noisy = at(Evaluator(), self.solved(src_center_y=1.0, src_radius=0.5, src_noise_amount=0.9), "sol", 3)
        again = at(Evaluator(), self.solved(src_center_y=1.0, src_radius=0.5, src_noise_amount=0.9), "sol", 3)
        self.assertNotEqual(plain.fingerprint(), noisy.fingerprint())
        self.assertEqual(noisy.fingerprint(), again.fingerprint())

    def emit_from_geometry(self, mode):
        d = Dispatcher()
        make(d, cube=("Cube3D", {"cube_size": 1.0, "ty": 1.0}),
             src=("FluidSource3D", {"fluid_emit_from": mode}), sol=("FluidSolver3D", {**GRID}))
        wire(d, "src", "geo", "cube")
        wire(d, "sol", "fluid", "src")
        return self.footprint(d)

    def test_surface_and_volume_emission_from_geometry(self):
        surface = self.emit_from_geometry("surface")
        volume = self.emit_from_geometry("volume")
        cells = 8                                                                 # a 1.0 cube on 0.125 cells
        self.assertGreaterEqual(len(volume[0]), (cells - 1) ** 3)
        self.assertLessEqual(len(volume[0]), (cells + 2) ** 3)
        self.assertGreater(len(volume[0]), len(surface[0]))
        self.assertGreaterEqual(len(surface[0]), 6 * (cells - 2) ** 2)
        solver = np.zeros((16, 24, 16), bool)
        solver.reshape(-1)[volume[0]] = True
        ii, jj, kk = np.nonzero(solver)
        # the cube sits at y 0.5 .. 1.5 and x, z -0.5 .. 0.5
        self.assertAlmostEqual((ii.min() * 0.125 - 1.0), -0.5, delta=0.13)
        self.assertAlmostEqual((jj.min() * 0.125), 0.5, delta=0.13)


class ForceNodeTests(unittest.TestCase):
    def forced(self, *forces, **solver):
        d = Dispatcher()
        make(d, src=("FluidSource3D", {"src_center_y": 0.5, "src_radius": 0.3}),
             sol=("FluidSolver3D", {**GRID, **solver}))
        previous = "src"
        for index, (kind, params) in enumerate(forces):
            key = f"f{index}"
            make(d, **{key: ("FluidForce3D", {"force_kind": kind, **params})})
            wire(d, key, "fluid", previous)
            previous = key
        wire(d, "sol", "fluid", previous)
        return d

    def test_wind_pushes_the_plume_sideways_and_gravity_pulls_it_down(self):
        ev = Evaluator()
        base = at(ev, self.forced(), "sol", 25)
        windy = at(ev, self.forced(("wind", {"force_dir_x": 1.0, "force_dir_y": 0.0, "strength": 0.05}),
                                   boundary_x="open"), "sol", 25)
        base_open = at(ev, self.forced(boundary_x="open"), "sol", 25)
        heavy = at(ev, self.forced(("gravity", {"force_dir_y": -1.0, "strength": 0.3})), "sol", 25)
        self.assertGreater(fluid3d.centre_of_mass(windy.density)[0],
                           fluid3d.centre_of_mass(base_open.density)[0] + 1.0)
        self.assertLess(height(heavy), height(base) - 0.1)

    def test_a_buoyancy_force_replaces_the_built_in_lift(self):
        ev = Evaluator()
        lifted = at(ev, self.forced(), "sol", 12)
        still = at(ev, self.forced(("buoyancy", {"buoyancy_lift": 0.0, "buoyancy_settle": 0.0})), "sol", 12)
        stronger = at(ev, self.forced(("buoyancy", {"buoyancy_lift": 0.25})), "sol", 12)
        self.assertLess(height(still), height(lifted) - 0.2)
        self.assertGreater(height(stronger), height(lifted))

    def test_drag_and_turbulence_change_the_flow(self):
        ev = Evaluator()
        base = at(ev, self.forced(), "sol", 15)
        draggy = at(ev, self.forced(("drag", {"drag": 1.5})), "sol", 15)
        turbulent = at(ev, self.forced(("turbulence", {"strength": 0.08, "turbulence_scale": 0.4})), "sol", 15)
        self.assertLess(float(np.abs(draggy.velocity).max()), float(np.abs(base.velocity).max()))
        self.assertNotEqual(turbulent.fingerprint(), base.fingerprint())
        again = at(Evaluator(), self.forced(("turbulence", {"strength": 0.08, "turbulence_scale": 0.4})), "sol", 15)
        self.assertEqual(turbulent.fingerprint(), again.fingerprint())

    def test_a_force_knob_and_the_frame_window_change_the_run(self):
        a = at(Evaluator(), self.forced(("wind", {"strength": 0.05})), "sol", 6)
        b = at(Evaluator(), self.forced(("wind", {"strength": 0.06})), "sol", 6)
        late = at(Evaluator(), self.forced(("wind", {"strength": 0.05, "from_frame": 50})), "sol", 6)
        plain = at(Evaluator(), self.forced(), "sol", 6)
        self.assertNotEqual(a.stream.run, b.stream.run)
        self.assertNotEqual(a.fingerprint(), late.fingerprint())
        self.assertEqual(late.fingerprint(), plain.fingerprint())               # inactive at frame 6: same arrays
        self.assertNotEqual(a.stream.run, plain.stream.run)

    def test_a_bypassed_force_passes_the_chain(self):
        d = self.forced(("wind", {"strength": 0.05}))
        d.execute({"op": "disable", "id": "f0", "value": True})
        bypassed = at(Evaluator(), d, "sol", 8)
        plain = at(Evaluator(), self.forced(), "sol", 8)
        self.assertEqual(bypassed.fingerprint(), plain.fingerprint())


class ColliderNodeTests(unittest.TestCase):
    def graph(self, *, collide=True, **collider):
        d = Dispatcher()
        make(d, src=("FluidSource3D", {"src_center_y": 0.4, "src_radius": 0.3}),
             cube=("Cube3D", {"cube_size": 1.0, "ty": 1.2}), col=("FluidCollide3D", collider),
             sol=("FluidSolver3D", {**GRID}))
        wire(d, "col", "fluid", "src")
        wire(d, "col", "geometry", "cube")
        wire(d, "sol", "fluid", "col" if collide else "src")
        return d

    def test_the_collider_blocks_the_plume_and_density_stays_out_of_the_solid(self):
        ev = Evaluator()
        blocked = at(ev, self.graph(), "sol", 40)
        free = at(ev, self.graph(collide=False), "sol", 40)
        solid, _, _ = blocked.stream.solver()._solid_for(40)
        self.assertGreater(int(solid.sum()), 100)
        self.assertEqual(float(blocked.density[solid].max()), 0.0)
        y = (np.arange(24) + 0.5) * 0.125
        above = y > 1.8
        self.assertLess(float(blocked.density[:, above, :].sum()), 0.5 * float(free.density[:, above, :].sum()))

    def test_an_unwired_collider_and_a_bypassed_one_do_nothing(self):
        plain = at(Evaluator(), self.graph(collide=False), "sol", 6)
        d = self.graph()
        d.execute({"op": "disable", "id": "col", "value": True})
        self.assertEqual(at(Evaluator(), d, "sol", 6).fingerprint(), plain.fingerprint())
        d = Dispatcher()
        make(d, src=("FluidSource3D", {"src_center_y": 0.4, "src_radius": 0.3}), col=("FluidCollide3D", {}),
             sol=("FluidSolver3D", {**GRID}))
        wire(d, "col", "fluid", "src")
        wire(d, "sol", "fluid", "col")
        self.assertEqual(float(at(Evaluator(), d, "sol", 6).density.sum()), float(plain.density.sum()))

    def test_the_collider_geometry_is_part_of_the_run(self):
        a = at(Evaluator(), self.graph(), "sol", 4)
        d = self.graph()
        set_(d, "cube", ty=1.5)
        self.assertNotEqual(at(Evaluator(), d, "sol", 4).stream.run, a.stream.run)

    def test_animated_collider_pushes_the_plume_aside_and_is_off_by_default(self):
        # bit-identical to a frozen collider when "animated" is left at its default (0)
        default = at(Evaluator(), self.graph(), "sol", 30)
        off = at(Evaluator(), self.graph(animated=0), "sol", 30)
        self.assertEqual(default.fingerprint(), off.fingerprint())
        # a gentle drift (0.02 world units, well under a cell, per frame): fast enough to differ from frozen
        # over 30 frames, slow enough not to violate the explicit scheme's own stability limit
        moving = self.graph(animated=1)
        for frame in (1, 60):
            set_key(moving, "cube", "tx", frame, 0.02 * (frame - 1))
        pushed = at(Evaluator(), moving, "sol", 30)
        self.assertNotEqual(pushed.fingerprint(), off.fingerprint())

    def test_editing_a_collider_keyframe_invalidates_the_cache_and_an_unrelated_node_does_not(self):
        d = self.graph(animated=1)
        d.execute({"op": "time", "last": 60})    # the digest hashes the collider's motion over the document's
        for frame in (1, 60):                    # own time range, so the range must cover the edited keyframe
            set_key(d, "cube", "tx", frame, 0.02 * (frame - 1))
        ev = Evaluator()
        run_before = at(ev, d, "sol", 20).stream.run
        # an unrelated node, wired to nothing this graph uses: no effect on the run
        make(d, spare=("Card3D", {}))
        set_(d, "spare", tx=5.0)
        self.assertEqual(at(ev, d, "sol", 20).stream.run, run_before)
        # editing the collider's own animation keyframe changes the run, at any frame
        set_key(d, "cube", "tx", 60, 0.5)
        self.assertNotEqual(at(ev, d, "sol", 20).stream.run, run_before)


class CacheNodeTests(unittest.TestCase):
    def evaluator(self, root=None):
        return Evaluator(sim=simcache.SimCache(root)) if root else Evaluator()

    def test_scrubbing_back_and_replaying_never_re_solves(self):
        d = plume()
        ev = self.evaluator()
        start = steps()
        at(ev, d, "c", 12)
        solved = steps() - start
        self.assertEqual(solved, 12)                       # one substep per frame, the solver node did not solve twice
        for frame in (3, 12, 7, 1, 12):
            at(ev, d, "c", frame)
        self.assertEqual(steps() - start, solved)
        at(ev, d, "c", 14)
        self.assertEqual(steps() - start, solved + 2)      # only the new frames

    def test_the_cache_serves_what_the_solver_makes(self):
        d = plume()
        ev = Evaluator()
        cached, direct = at(ev, d, "c", 8), at(ev, d, "sol", 8)
        np.testing.assert_array_equal(cached.density, direct.density)
        np.testing.assert_array_equal(cached.velocity, direct.velocity)
        self.assertEqual(cached.stream.run, direct.stream.run)

    def test_precision_and_channels(self):
        exact = at(Evaluator(), plume(), "c", 8)
        d = plume()
        set_(d, "c", cache_precision="float16", cache_channels="density_temperature")
        low = at(Evaluator(), d, "c", 8)
        self.assertIsNone(low.velocity)
        self.assertIsNone(low.flame)
        self.assertIsNotNone(low.temperature)
        self.assertEqual(low.density.dtype, np.float32)
        np.testing.assert_allclose(low.density, exact.density, rtol=2e-3, atol=1e-4)
        self.assertFalse(np.array_equal(low.density, exact.density))              # it really was quantised
        set_(d, "c", cache_channels="density")
        self.assertIsNone(at(Evaluator(), d, "c", 8).temperature)
        self.assertIsNotNone(exact.velocity)
        self.assertIsNotNone(exact.flame)
        # a fresh solve and a cache hit are the same quantised copy
        ev = Evaluator()
        self.assertEqual(at(ev, d, "c", 8).fingerprint(), at(ev, d, "c", 8).fingerprint())

    def test_a_cache_restarts_from_disk_without_solving(self):
        with tempfile.TemporaryDirectory() as root:
            d = plume()
            first = at(self.evaluator(root), d, "c", 10)
            start = steps()
            again = at(self.evaluator(root), d, "c", 10)
            self.assertEqual(steps(), start)
            self.assertEqual(again.fingerprint(), first.fingerprint())
            further = at(self.evaluator(root), d, "c", 12)          # resumes from the checkpoint at frame 10
            self.assertEqual(steps() - start, 2)
            straight = at(Evaluator(), d, "sol", 12)
            self.assertEqual(further.density.tobytes(), straight.density.tobytes())

    def test_budgets_bound_the_stores_and_a_zero_disk_budget_keeps_nothing_on_disk(self):
        with tempfile.TemporaryDirectory() as root:
            d = plume()
            set_(d, "c", cache_disk_mb=0)
            ev = self.evaluator(root)
            at(ev, d, "c", 6)
            store = ev.sim_store(256, 0)
            self.assertFalse(store.enabled)
            self.assertEqual(store.stats()["disk_bytes"], 0)
            self.assertIs(ev.sim_store(256, 0), store)
            self.assertIsNot(ev.sim_store(256, 64), store)

    def test_a_bypassed_cache_passes_its_input(self):
        d = plume()
        d.execute({"op": "disable", "id": "c", "value": True})
        ev = Evaluator()
        start = steps()
        value = at(ev, d, "c", 5)
        self.assertEqual(value.fingerprint(), at(ev, d, "sol", 5).fingerprint())
        self.assertEqual(steps() - start, 5)                # solved by the solver node, not a cache

    def test_cancellation_reaches_the_cache_node_and_keeps_banked_frames(self):
        d = plume()
        ev = Evaluator()
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(Cancelled):
            at(ev, d, "c", 20, cancel=cancel)
        original = fluid3d.Smoke3D.step
        cancel = threading.Event()
        calls = [0]

        def counting(solver, state, frame, substep, seed):
            calls[0] += 1
            result = original(solver, state, frame, substep, seed)
            if calls[0] == 6:
                cancel.set()
            return result
        fluid3d.Smoke3D.step = counting
        try:
            with self.assertRaises(Cancelled):
                at(ev, d, "c", 30, cancel=cancel)
        finally:
            fluid3d.Smoke3D.step = original
        start = steps()
        at(ev, d, "c", 5)                                    # the frames solved before the cancel are banked
        self.assertEqual(steps(), start)


class RenderTests(unittest.TestCase):
    def test_a_solved_frame_renders_through_the_raymarch(self):
        d = plume()
        make(d, s=("Scene3D", {}), cam=("Camera3D", {"tz": 6.0, "ty": 1.4}),
             rd=("Render3D", {"width": 32, "height": 32, "samples": 1, "volume_density_scale": 8.0}))
        wire(d, "s", "object0", "c")
        wire(d, "rd", "scene", "s")
        wire(d, "rd", "camera", "cam")
        ev = Evaluator()
        image = ev.evaluate(dict(d.document, view="rd"), frame=14)
        self.assertGreater(float(image[..., 3].max()), 0.05)                       # smoke coverage
        self.assertGreater(float(image[..., :3].max()), 0.0)
        empty = ev.evaluate(dict(d.document, view="rd"), frame=0)
        self.assertEqual(float(empty[..., 3].max()), 0.0)                          # before the start frame: nothing
        d2 = plume()
        make(d2, s=("Scene3D", {}), cam=("Camera3D", {"tz": 6.0, "ty": 1.4}),
             rd=("Render3D", {"width": 32, "height": 32, "samples": 1, "volume_density_scale": 8.0,
                              "render_output": "volume_density"}))
        wire(d2, "s", "object0", "sol")
        wire(d2, "rd", "scene", "s")
        wire(d2, "rd", "camera", "cam")
        density_pass = ev.evaluate(dict(d2.document, view="rd"), frame=14)
        self.assertGreater(float(density_pass[..., 0].max()), 0.0)


class OldDocumentTests(unittest.TestCase):
    def test_documents_without_fluid_nodes_load_unchanged(self):
        d = Dispatcher()
        make(d, p=("Plume3D", {}), e=("ParticleEmitter3D", {}), g=("Grade", {}))
        before = json.dumps(d.document, sort_keys=True)
        reloaded = upgrade_document(json.loads(before))
        validate(reloaded)
        self.assertEqual(json.dumps(reloaded, sort_keys=True), before)

    def test_a_document_saved_with_fluid_nodes_reloads_and_evaluates(self):
        d = plume()
        first = at(Evaluator(), d, "c", 5)
        reloaded = upgrade_document(json.loads(json.dumps(d.document)))
        validate(reloaded)
        second = Evaluator().evaluate_raster(reloaded, "c", frame=5, typed=True)
        self.assertEqual(first.fingerprint(), second.fingerprint())

    def test_a_collider_saved_with_the_old_velocity_from_motion_key_upgrades_to_animated(self):
        d = Dispatcher()
        make(d, src=("FluidSource3D", {}), cube=("Cube3D", {}), col=("FluidCollide3D", {}))
        wire(d, "col", "fluid", "src")
        wire(d, "col", "geometry", "cube")
        doc = json.loads(json.dumps(d.document))
        doc["nodes"]["col"]["params"] = {"velocity_from_motion": 1}
        reloaded = upgrade_document(doc)
        validate(reloaded)
        self.assertEqual(reloaded["nodes"]["col"]["params"], {"animated": 1})


class PropertiesPanelTests(unittest.TestCase):
    def test_the_solver_panel_shows_the_derived_resolution_read_only(self):
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QApplication, QLabel
        from nodebased.app import Window
        app = QApplication.instance() or QApplication([])
        window = Window()
        window.show()
        try:
            for _ in range(2000):
                if window.frame is not None:
                    break
                pause(10)
            window.command({"op": "create", "id": "sol", "type": "FluidSolver3D", "pos": [3000, 3000],
                            "params": {}}, render=False)
            window.graph.items_by_id["sol"].setSelected(True)
            app.processEvents()
            labels = {l.objectName(): l for l in window.findChildren(QLabel) if l.objectName()}
            self.assertEqual(labels["resolution-readout"].text(), "20 x 30 x 20  (12,000 cells)")
            window.command({"op": "set", "id": "sol", "param": "division_size", "value": 0.2}, render=False)
            window.graph.items_by_id["sol"].setSelected(False)
            window.graph.items_by_id["sol"].setSelected(True)
            app.processEvents()
            labels = {l.objectName(): l for l in window.findChildren(QLabel) if l.objectName()}
            self.assertEqual(labels["resolution-readout"].text(), "10 x 15 x 10  (1,500 cells)")
        finally:
            window.saved_document = window.dispatcher.document
            window.close()
            window.executor.shutdown(wait=True, cancel_futures=True)
            app.processEvents()


if __name__ == "__main__":
    unittest.main()
