"""Lane L6 step E, part 2: FluidLiquidSolver3D, FluidSurface3D and FluidFoam3D as graph nodes, and the liquid mode of
FluidSource3D.

Registration, typed wiring, the ParticleInstance (with its signed-distance Volume) a liquid solves to, smoke and
liquid sources kept apart, forces and colliders through the graph, scrubbing without re-solving through
ParticleCache3D, invalidation, cancellation, bypass, old documents, and a liquid drawn by Render3D.
"""
import json
import threading
import unittest

import numpy as np

from nodebased import flip3d, fluid3d, scene3d, simcache
from nodebased.cancellation import Cancelled
from nodebased.core import (Dispatcher, INPUT_TYPES, OUTPUT_TYPES, SPECS, bypass_slot, upgrade_document, validate)
from nodebased.imaging import Evaluator
from nodebased.knobs import knob_layout
from nodebased.theme import COLORS
from nodebased.tiers import REGION_RULES
from tests.test_particles_nodes import make, set_, wire

KINDS = ("FluidLiquidSolver3D", "FluidSurface3D", "FluidFoam3D")
# 16 x 16 x 16 cells of 0.125
GRID = {"division_size": 0.125, "bounds_min_x": -1.0, "bounds_min_y": 0.0, "bounds_min_z": -1.0,
        "bounds_max_x": 1.0, "bounds_max_y": 2.0, "bounds_max_z": 1.0}
BLOB = {"fluid_type": "liquid", "src_center_x": 0.0, "src_center_y": 1.0, "src_center_z": 0.0, "src_radius": 0.45,
        "end_frame": 1}


def at(evaluator, d, key, frame, **kwargs):
    return evaluator.evaluate_raster(d.document, key, frame=frame, typed=True, **kwargs)


def steps():
    return flip3d.SOLVER_STATS["steps"]


def liquid(**solver):
    """Source -> liquid solver (-> surface, foam, cache): the smallest working graph."""
    d = Dispatcher()
    make(d, src=("FluidSource3D", BLOB), sol=("FluidLiquidSolver3D", {**GRID, **solver}),
         sf=("FluidSurface3D", {}), foam=("FluidFoam3D", {}), c=("ParticleCache3D", {}))
    wire(d, "sol", "fluid", "src")
    wire(d, "sf", "particles", "sol")
    wire(d, "foam", "particles", "sol")
    wire(d, "c", "particles", "sol")
    return d


class RegistrationTests(unittest.TestCase):
    def test_kinds_are_registered_everywhere(self):
        for kind in KINDS:
            self.assertIn(kind, SPECS)
            self.assertIn(kind, COLORS)
            self.assertIn(kind, REGION_RULES)
            laid_out = sorted(p for group in knob_layout(kind) for p in group.params)
            self.assertEqual(laid_out, sorted(SPECS[kind]["params"]), kind)
        self.assertEqual(OUTPUT_TYPES["FluidLiquidSolver3D"], "particles")
        self.assertEqual(OUTPUT_TYPES["FluidSurface3D"], "geometry")
        self.assertEqual(OUTPUT_TYPES["FluidFoam3D"], "particles")
        self.assertEqual(SPECS["FluidLiquidSolver3D"]["inputs"], ["fluid"])
        self.assertEqual(SPECS["FluidSurface3D"]["inputs"], ["particles"])
        self.assertEqual(SPECS["FluidFoam3D"]["inputs"], ["particles"])
        self.assertEqual(SPECS["FluidSource3D"]["params"]["fluid_type"], "smoke")
        for name in ("flip_ratio", "particles_per_cell", "liquid_gravity", "viscosity"):
            self.assertIn(name, SPECS["FluidLiquidSolver3D"]["params"])
        self.assertEqual(SPECS["FluidLiquidSolver3D"]["params"]["particles_per_cell"], 8)
        self.assertEqual(SPECS["FluidLiquidSolver3D"]["params"]["viscosity"], 0.0)          # viscosity is off by default

    def test_typed_wiring(self):
        d = Dispatcher()
        make(d, src=("FluidSource3D", {}), col=("FluidCollide3D", {}), sol=("FluidLiquidSolver3D", {}),
             sf=("FluidSurface3D", {}), foam=("FluidFoam3D", {}), c=("ParticleCache3D", {}),
             r=("ParticleRender3D", {}), s=("Scene3D", {}), card=("Card3D", {}), g=("Grade", {}))
        wire(d, "col", "fluid", "src")
        wire(d, "sol", "fluid", "col")
        wire(d, "sf", "particles", "sol")
        wire(d, "foam", "particles", "sol")
        wire(d, "c", "particles", "foam")
        wire(d, "r", "particles", "sol")
        wire(d, "s", "object0", "sf")
        wire(d, "s", "object1", "r")
        validate(d.document)
        for target, slot, source in (("sol", "fluid", "card"), ("sol", "fluid", "r"), ("sf", "particles", "card"),
                                     ("foam", "particles", "sf"), ("g", "image", "sol"), ("g", "image", "sf")):
            with self.assertRaises(ValueError, msg=(target, slot, source)):
                wire(d, target, slot, source)

    def test_nodes_are_rounded_3d_nodes(self):
        from nodebased.app import node_form
        for kind in KINDS:
            self.assertEqual(node_form({"type": kind}), "round", kind)

    def test_bypass_slots(self):
        def node(kind, **wired):
            slots = SPECS[kind]["inputs"] + SPECS[kind].get("optional_inputs", [])
            return dict(type=kind, inputs={slot: wired.get(slot) for slot in slots})
        self.assertEqual(bypass_slot(node("FluidLiquidSolver3D", fluid="a")), "fluid")
        self.assertEqual(bypass_slot(node("FluidSurface3D", particles="a")), "particles")
        self.assertEqual(bypass_slot(node("FluidFoam3D", particles="a")), "particles")


class SolveTests(unittest.TestCase):
    def test_a_liquid_solves_to_particles_with_a_signed_distance_volume(self):
        d = liquid()
        inst = at(Evaluator(), d, "sol", 1)
        self.assertIsInstance(inst, scene3d.ParticleInstance)
        self.assertGreater(len(inst), 1000)
        self.assertIsInstance(inst.stream, flip3d.LiquidStream)
        self.assertEqual(len(inst.positions) % 8, 0)                      # 8 per cell by default
        vol = inst.surface
        self.assertIsInstance(vol, scene3d.Volume)
        self.assertEqual(vol.density.shape, (16, 16, 16))
        self.assertLess(float(vol.density.min()), 0.0)                    # negative inside the liquid
        self.assertGreater(float(vol.density.max()), 0.0)
        inside = vol.density < 0
        x = (np.arange(16)[:, None, None] + 0.5) * 0.125 - 1.0
        y = (np.arange(16)[None, :, None] + 0.5) * 0.125
        z = (np.arange(16)[None, None, :] + 0.5) * 0.125 - 1.0
        centre = np.sqrt(x ** 2 + (y - 1.0) ** 2 + z ** 2)
        self.assertLess(float(centre[inside].max()), 0.7)                 # the blob, not the tank
        self.assertLess(float(np.abs(centre[inside].mean() - 0.3)), 0.15)
        pos = inst.positions
        self.assertTrue(np.all(pos >= np.array((-1.0, 0.0, -1.0))) and np.all(pos <= np.array((1.0, 2.0, 1.0))))

    def test_the_sdf_output_can_be_switched_off(self):
        inst = at(Evaluator(), liquid(liquid_sdf=0), "sol", 2)
        self.assertIsNone(inst.surface)

    def test_the_blob_falls_and_lands(self):
        ev = Evaluator()
        d = liquid()
        first = at(ev, d, "c", 1)
        later = at(ev, d, "c", 24)
        self.assertLess(float(later.positions[:, 1].mean()), float(first.positions[:, 1].mean()) - 0.4)
        self.assertGreater(float(later.positions[:, 1].min()), 0.0)

    def test_smoke_and_liquid_sources_stay_apart(self):
        d = Dispatcher()
        make(d, smoke=("FluidSource3D", {"src_center_y": 0.4}), wet=("FluidSource3D", BLOB),
             both=("FluidCollide3D", {}), sol=("FluidSolver3D", {**GRID, "bounds_max_y": 3.0}),
             lq=("FluidLiquidSolver3D", GRID))
        wire(d, "both", "fluid", "smoke")
        wire(d, "sol", "fluid", "wet")                   # a liquid source into the smoke solver
        wire(d, "lq", "fluid", "both")                   # a smoke source into the liquid solver
        ev = Evaluator()
        self.assertEqual(float(at(ev, d, "sol", 4).density.max()), 0.0)
        self.assertEqual(len(at(ev, d, "lq", 4)), 0)
        wire(d, "both", "fluid", "wet")                  # a chain holding one of each
        self.assertGreater(len(at(ev, d, "lq", 4)), 100)

    def test_a_still_source_fills_once_and_one_with_a_velocity_pours(self):
        still = at(Evaluator(), liquid(), "sol", 6)
        d = Dispatcher()
        make(d, src=("FluidSource3D", {**BLOB, "end_frame": 1000000, "src_vel_y": -0.02, "src_radius": 0.25,
                                       "src_center_y": 1.6}),
             sol=("FluidLiquidSolver3D", GRID))
        wire(d, "sol", "fluid", "src")
        ev = Evaluator()
        few, many = at(ev, d, "sol", 3), at(ev, d, "sol", 18)
        self.assertGreater(len(many), 1.3 * len(few))
        self.assertGreater(len(still), 0)
        self.assertLess(len(at(Evaluator(), liquid(), "sol", 20)), 1.1 * len(still))


class ForceAndColliderTests(unittest.TestCase):
    def test_a_wind_force_pushes_the_liquid(self):
        ev = Evaluator()
        plain = at(ev, liquid(), "sol", 12)
        d = liquid()
        make(d, wind=("FluidForce3D", {"force_kind": "wind", "force_dir_x": 1.0, "force_dir_y": 0.0,
                                       "strength": 0.05}))
        wire(d, "wind", "fluid", "src")
        wire(d, "sol", "fluid", "wind")
        pushed = at(ev, d, "sol", 12)
        self.assertGreater(float(pushed.positions[:, 0].mean()), float(plain.positions[:, 0].mean()) + 0.05)

    def test_a_collider_keeps_the_liquid_out_of_a_solid(self):
        d = liquid()
        make(d, cube=("Cube3D", {"cube_size": 0.6, "ty": 0.3}), col=("FluidCollide3D", {}))
        wire(d, "col", "fluid", "src")
        wire(d, "col", "geometry", "cube")
        wire(d, "sol", "fluid", "col")
        ev = Evaluator()
        inst = at(ev, d, "sol", 24)
        solid, _, _ = inst.stream.solver()._solid_for(24)
        self.assertGreater(int(solid.sum()), 20)
        cells = np.floor((inst.positions - np.array(inst.stream.origin)) / inst.stream.voxel).astype(int)
        self.assertFalse(bool(solid[tuple(cells.T)].any()))
        held = at(ev, d, "sol", 12)                       # the blob lands on the block, so it is still higher up
        free = at(ev, liquid(), "sol", 12)
        self.assertGreater(float(held.positions[:, 1].mean()), float(free.positions[:, 1].mean()) + 0.1)

    def test_particle_force_nodes_pass_a_liquid_on_unchanged(self):
        d = liquid()
        make(d, gr=("ParticleGravity3D", {}))
        wire(d, "gr", "particles", "sol")
        ev = Evaluator()
        through, direct = at(ev, d, "gr", 5), at(ev, d, "sol", 5)
        np.testing.assert_array_equal(through.positions, direct.positions)


class CacheAndRunTests(unittest.TestCase):
    def test_scrubbing_back_and_replaying_never_re_solves(self):
        d = liquid()
        ev = Evaluator()
        start = steps()
        at(ev, d, "c", 6)
        solved = steps() - start
        self.assertEqual(solved, 12)                       # two substeps per frame, the solver node did not solve twice
        for frame in (2, 6, 4, 1, 6):
            at(ev, d, "c", frame)
        self.assertEqual(steps() - start, solved)
        at(ev, d, "c", 7)
        self.assertEqual(steps() - start, solved + 2)

    def test_the_cache_serves_what_the_solver_makes(self):
        d = liquid()
        ev = Evaluator()
        cached, direct = at(ev, d, "c", 5), at(ev, d, "sol", 5)
        np.testing.assert_array_equal(cached.positions, direct.positions)
        np.testing.assert_array_equal(cached.ids, direct.ids)
        self.assertEqual(cached.stream.run, direct.stream.run)

    def test_editing_a_knob_abandons_the_run(self):
        base = at(Evaluator(), liquid(), "sol", 2).stream.run
        for name, value in (("flip_ratio", 0.5), ("liquid_gravity", 1.0), ("particles_per_cell", 27),
                            ("viscosity", 0.5), ("substeps", 3)):
            other = at(Evaluator(), liquid(**{name: value}), "sol", 2).stream.run
            self.assertNotEqual(base, other, name)

    def test_deterministic_across_evaluators(self):
        a = at(Evaluator(), liquid(), "sol", 8)
        b = at(Evaluator(), liquid(), "sol", 8)
        np.testing.assert_array_equal(a.positions, b.positions)
        np.testing.assert_array_equal(a.surface.density, b.surface.density)
        set_(d := liquid(), "sol", seed=5)
        c = at(Evaluator(), d, "sol", 8)
        self.assertLess(abs(len(c) - len(a)), 0.03 * len(a))       # a different seed jitters the particles
        self.assertFalse(np.array_equal(c.positions, a.positions))

    def test_cancellation(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(Cancelled):
            at(Evaluator(), liquid(), "sol", 4, cancel=cancel)

    def test_a_grid_past_the_cap_is_refused_with_the_numbers(self):
        d = liquid(division_size=0.005)
        with self.assertRaisesRegex(ValueError, r"cells; the CPU reference solver stops at 4,194,304"):
            at(Evaluator(), d, "sol", 2)

    def test_the_multigrid_gpu_solvers_are_refused(self):
        with self.assertRaisesRegex(ValueError, "smoke solver"):
            at(Evaluator(), liquid(pressure="resident"), "sol", 2)


class BypassAndOldDocumentTests(unittest.TestCase):
    def test_bypassed_liquid_nodes_contribute_nothing(self):
        d = liquid()
        ev = Evaluator()
        d.execute({"op": "disable", "id": "sol", "value": True})
        self.assertEqual(len(at(ev, d, "sol", 3)), 0)
        self.assertEqual(len(at(ev, d, "c", 3)), 0)
        self.assertEqual(len(at(ev, d, "sf", 3).vertices), 0)
        d = liquid()
        d.execute({"op": "disable", "id": "sf", "value": True})
        d.execute({"op": "disable", "id": "foam", "value": True})
        self.assertEqual(len(at(ev, d, "sf", 3).vertices), 0)
        self.assertEqual(len(at(ev, d, "foam", 3)), 0)
        self.assertGreater(len(at(ev, d, "sol", 3)), 0)

    def test_documents_saved_before_liquids_load_and_evaluate_unchanged(self):
        d = Dispatcher()
        make(d, src=("FluidSource3D", {"src_center_y": 0.4, "src_radius": 0.3}),
             sol=("FluidSolver3D", {**GRID, "bounds_max_y": 3.0, "boundary_y": "closed", "cooling_rate": 0.0}))
        wire(d, "sol", "fluid", "src")
        document = json.loads(json.dumps(d.document))
        del document["nodes"]["src"]["params"]["fluid_type"]        # a source saved before the knob existed
        reloaded = upgrade_document(document)
        validate(reloaded)
        self.assertEqual(reloaded["nodes"]["src"]["params"]["fluid_type"], "smoke")
        vol = Evaluator().evaluate_raster(reloaded, "sol", frame=6, typed=True)
        self.assertGreater(float(vol.density.max()), 0.0)


class RenderTests(unittest.TestCase):
    def test_a_liquid_renders_as_particles_and_as_a_mesh(self):
        d = liquid()
        make(d, s=("Scene3D", {}), cam=("Camera3D", {"tz": 5.0, "ty": 1.0}), r=("ParticleRender3D", {"representation": "spheres"}),
             rd=("Render3D", {"width": 48, "height": 48, "samples": 1}))
        wire(d, "r", "particles", "sol")
        wire(d, "s", "object0", "r")
        wire(d, "rd", "scene", "s")
        wire(d, "rd", "camera", "cam")
        ev = Evaluator()
        points = ev.evaluate(dict(d.document, view="rd"), frame=4)
        self.assertGreater(float(points[..., 3].max()), 0.5)
        make(d, s2=("Scene3D", {}), rd2=("Render3D", {"width": 48, "height": 48, "samples": 1}))
        wire(d, "s2", "object0", "sf")
        wire(d, "rd2", "scene", "s2")
        wire(d, "rd2", "camera", "cam")
        mesh = ev.evaluate(dict(d.document, view="rd2"), frame=4)
        self.assertGreater(float(mesh[..., 3].max()), 0.5)


if __name__ == "__main__":
    unittest.main()
