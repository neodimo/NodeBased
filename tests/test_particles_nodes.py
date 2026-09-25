"""Lane L5 step 2a, part 2: ParticleEmitter3D and ParticleCache3D as graph nodes.

Registration, typed wiring, scrubbing without re-solving (counted through `particles.SOLVER_STATS`),
invalidation, cancellation, budgets, restart from disk, bypass and old documents.
"""
import json
import tempfile
import threading
import unittest

import numpy as np

from nodebased import particles, scene3d, simcache
from nodebased.cancellation import Cancelled
from nodebased.core import (Dispatcher, INPUT_TYPES, OUTPUT_TYPES, SPECS, bypass_slot, upgrade_document,
                            validate)
from nodebased.imaging import Evaluator
from nodebased.knobs import knob_layout
from nodebased.theme import COLORS
from nodebased.tiers import REGION_RULES


def make(d, **nodes):
    for key, (kind, params) in nodes.items():
        d.execute({"op": "create", "id": key, "type": kind, "params": params})


def wire(d, target, slot, source):
    d.execute({"op": "connect", "id": target, "input": slot, "source": source})


def set_(d, key, **params):
    for name, value in params.items():
        d.execute({"op": "set", "id": key, "param": name, "value": value})


def steps():
    return particles.SOLVER_STATS["steps"]


def at(evaluator, d, key, frame, **kwargs):
    return evaluator.evaluate_raster(d.document, key, frame=frame, typed=True, **kwargs)


class RegistrationTests(unittest.TestCase):
    def test_kinds_are_registered_everywhere(self):
        for kind in ("ParticleEmitter3D", "ParticleCache3D"):
            self.assertIn(kind, SPECS)
            self.assertEqual(OUTPUT_TYPES[kind], "particles")
            self.assertIn(kind, COLORS)
            self.assertIn(kind, REGION_RULES)
            laid_out = sorted(p for group in knob_layout(kind) for p in group.params)
            self.assertEqual(laid_out, sorted(SPECS[kind]["params"]))
        self.assertEqual(SPECS["ParticleEmitter3D"]["inputs"], [])
        self.assertEqual(SPECS["ParticleEmitter3D"]["optional_inputs"], ["geo"])
        self.assertEqual(SPECS["ParticleCache3D"]["inputs"], ["particles"])
        self.assertEqual(INPUT_TYPES["particles"], ("particles",))
        for slot in ["object"] + [f"object{i}" for i in range(8)]:
            self.assertIn("particles", INPUT_TYPES[slot])

    def test_nuke_and_houdini_knob_vocabulary_is_present(self):
        params = SPECS["ParticleEmitter3D"]["params"]
        for name in ("emit_from", "emit_rate", "start_frame", "life", "life_variance", "emit_speed",
                     "speed_variance", "spread", "particle_size", "size_variance", "seed", "substeps",
                     "red", "green", "blue", "alpha", "emit_dir_x", "tx", "rx", "sx", "uscale", "pivot_x"):
            self.assertIn(name, params)

    def test_typed_wiring(self):
        d = Dispatcher()
        make(d, e=("ParticleEmitter3D", {}), c=("ParticleCache3D", {}), s=("Scene3D", {}),
             a=("Axis3D", {}), r=("Render3D", {}), card=("Card3D", {}), g=("Grade", {}))
        wire(d, "c", "particles", "e")
        wire(d, "s", "object0", "e")
        wire(d, "s", "object1", "c")
        wire(d, "a", "object", "c")
        wire(d, "e", "geo", "card")
        validate(d.document)
        for target, slot, source in (("r", "scene", "e"), ("c", "particles", "card"), ("e", "geo", "c"),
                                     ("g", "image", "e")):
            with self.assertRaises(ValueError, msg=(target, slot, source)):
                wire(d, target, slot, source)

    def test_nodes_are_rounded_3d_nodes(self):
        from nodebased.app import node_form
        self.assertEqual(node_form({"type": "ParticleEmitter3D"}), "round")
        self.assertEqual(node_form({"type": "ParticleCache3D"}), "round")

    def test_defaults_validate_and_survive_a_json_round_trip(self):
        d = Dispatcher()
        make(d, e=("ParticleEmitter3D", {}), c=("ParticleCache3D", {}))
        wire(d, "c", "particles", "e")
        reloaded = upgrade_document(json.loads(json.dumps(d.document)))
        validate(reloaded)
        self.assertEqual(reloaded["nodes"]["e"]["params"], SPECS["ParticleEmitter3D"]["params"])


class EmitterNodeTests(unittest.TestCase):
    def setUp(self):
        self.d = Dispatcher()
        make(self.d, e=("ParticleEmitter3D", {"emit_rate": 4.0, "life": 1000.0, "emit_speed": 0.5, "seed": 3}))
        self.ev = Evaluator()

    def test_output_is_a_particle_instance_with_the_solved_arrays(self):
        value = at(self.ev, self.d, "e", 10)
        self.assertIsInstance(value, scene3d.ParticleInstance)
        self.assertEqual(len(value), 40)
        self.assertEqual(value.positions.shape, (40, 3))
        self.assertEqual(value.colors.shape, (40, 4))
        self.assertEqual(value.ids.tolist(), list(range(40)))
        np.testing.assert_allclose(value.ages[:3], [10.0, 10.0, 10.0])   # four births per frame
        self.assertFalse(value.positions.flags.writeable)
        before = at(self.ev, self.d, "e", 9)
        self.assertEqual(len(before), 36)

    def test_same_seed_same_particles_across_fresh_evaluators(self):
        a = at(Evaluator(), self.d, "e", 20)
        b = at(Evaluator(), self.d, "e", 20)
        self.assertEqual(a.positions.tobytes(), b.positions.tobytes())
        set_(self.d, "e", seed=4)
        set_(self.d, "e", spread=45.0)
        c = at(Evaluator(), self.d, "e", 20)
        self.assertNotEqual(a.positions.tobytes(), c.positions.tobytes())

    def test_scrubbing_back_reads_the_cache_and_does_not_re_solve(self):
        start = steps()
        at(self.ev, self.d, "e", 100)
        self.assertEqual(steps() - start, 100)
        start = steps()
        fifty = at(self.ev, self.d, "e", 50)
        self.assertEqual(steps() - start, 0)
        self.assertEqual(len(fifty), 200)
        at(self.ev, self.d, "e", 3)
        at(self.ev, self.d, "e", 100)
        self.assertEqual(steps() - start, 0)
        at(self.ev, self.d, "e", 101)                  # one frame beyond: exactly one frame of solving
        self.assertEqual(steps() - start, 1)

    def test_jumping_ahead_solves_forward_once_and_then_never_again(self):
        start = steps()
        at(self.ev, self.d, "e", 60)
        at(self.ev, self.d, "e", 30)
        at(self.ev, self.d, "e", 60)
        self.assertEqual(steps() - start, 60)

    def test_changing_any_upstream_knob_or_the_seed_invalidates(self):
        at(self.ev, self.d, "e", 40)
        for name, value in (("emit_rate", 5.0), ("seed", 9), ("life", 999.0), ("spread", 10.0),
                            ("tx", 1.0), ("emit_speed", 0.6), ("substeps", 2), ("particle_size", 0.3),
                            ("emit_from", "vertices"), ("start_frame", 2), ("red", 0.5)):
            with self.subTest(knob=name):
                at(self.ev, self.d, "e", 40)
                start = steps()
                at(self.ev, self.d, "e", 40)
                self.assertEqual(steps() - start, 0)
                set_(self.d, "e", **{name: value})
                start = steps()
                at(self.ev, self.d, "e", 40)
                self.assertGreater(steps() - start, 0)

    def test_an_unrelated_node_change_does_not_invalidate(self):
        make(self.d, g=("Constant", {}))
        at(self.ev, self.d, "e", 25)
        set_(self.d, "g", red=0.25)
        start = steps()
        at(self.ev, self.d, "e", 25)
        self.assertEqual(steps() - start, 0)

    def test_the_emission_geometry_is_part_of_the_run(self):
        make(self.d, card=("Card3D", {"card_width": 2.0}))
        set_(self.d, "e", emit_from="surface")
        wire(self.d, "e", "geo", "card")
        first = at(self.ev, self.d, "e", 5)
        set_(self.d, "card", card_width=8.0)
        start = steps()
        second = at(self.ev, self.d, "e", 5)
        self.assertEqual(steps() - start, 5)
        self.assertGreater(float(np.abs(second.positions[:, 0]).max()), float(np.abs(first.positions[:, 0]).max()))

    def test_emission_from_wired_geometry_and_its_transform(self):
        make(self.d, cube=("Cube3D", {"cube_size": 2.0, "tx": 10.0}))
        set_(self.d, "e", emit_from="volume", emit_speed=0.0, emit_rate=200.0)
        wire(self.d, "e", "geo", "cube")
        value = at(self.ev, self.d, "e", 1)
        self.assertEqual(len(value), 200)
        self.assertGreaterEqual(float(value.positions[:, 0].min()), 9.0 - 1e-5)
        self.assertLessEqual(float(value.positions[:, 0].max()), 11.0 + 1e-5)

    def test_cancellation_stops_the_solve_and_keeps_the_frames_already_banked(self):
        cancel = threading.Event()
        cancel.set()
        start = steps()
        with self.assertRaises(Cancelled):
            at(self.ev, self.d, "e", 50, cancel=cancel)
        self.assertEqual(steps() - start, 0)

        original = particles.ParticleEmitter.step
        cancel = threading.Event()
        calls = [0]

        def counting(emitter, state, frame, substep, seed):
            calls[0] += 1
            result = original(emitter, state, frame, substep, seed)
            if calls[0] == 30:
                cancel.set()
            return result
        particles.ParticleEmitter.step = counting
        try:
            with self.assertRaises(Cancelled):
                at(self.ev, self.d, "e", 100, cancel=cancel)
        finally:
            particles.ParticleEmitter.step = original
        self.assertEqual(calls[0], 30)
        start = steps()
        value = at(self.ev, self.d, "e", 100)
        self.assertEqual(steps() - start, 70)                 # frames 1-30 were kept, 31-100 solved now
        self.assertEqual(len(value), 400)

    def test_a_node_below_the_start_frame_is_empty(self):
        set_(self.d, "e", start_frame=20)
        self.assertEqual(len(at(self.ev, self.d, "e", 19)), 0)
        self.assertEqual(len(at(self.ev, self.d, "e", 20)), 4)


class SceneWiringTests(unittest.TestCase):
    def test_the_emitter_sits_under_scene_and_axis_parents(self):
        d = Dispatcher()
        make(d, e=("ParticleEmitter3D", {"emit_rate": 3.0, "emit_speed": 0.0, "life": 100.0}),
             ax=("Axis3D", {"tx": 10.0, "ry": 90.0}), s=("Scene3D", {"ty": 2.0}))
        wire(d, "ax", "object", "e")
        wire(d, "s", "object0", "ax")
        scene = Evaluator().evaluate_raster(d.document, "s", frame=2, typed=True)
        self.assertEqual(len(scene.particles), 1)
        instance = scene.particles[0]
        self.assertEqual(len(instance), 6)
        expected = scene3d.Transform3D(position=scene3d.Vec3(0, 2, 0)).matrix() @ scene3d.Transform3D(
            position=scene3d.Vec3(10, 0, 0), rotation=scene3d.Vec3(0, 90, 0)).matrix()
        np.testing.assert_allclose(instance.matrix, expected, atol=1e-6)

    def test_a_scene_with_no_particles_has_an_empty_particle_tuple(self):
        d = Dispatcher()
        make(d, card=("Card3D", {}), s=("Scene3D", {}))
        wire(d, "s", "object0", "card")
        scene = Evaluator().evaluate_raster(d.document, "s", typed=True)
        self.assertEqual(scene.particles, ())
        self.assertEqual(scene3d.Scene().particles, ())


class CacheNodeTests(unittest.TestCase):
    def setUp(self):
        self.d = Dispatcher()
        make(self.d, e=("ParticleEmitter3D", {"emit_rate": 50.0, "life": 1000.0, "seed": 2}),
             c=("ParticleCache3D", {}))
        wire(self.d, "c", "particles", "e")
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)

    def evaluator(self):
        return Evaluator(sim=simcache.SimCache(self.root.name))

    def test_output_matches_the_emitter_and_keeps_its_parent_matrix(self):
        ev = self.evaluator()
        direct = at(ev, self.d, "e", 12)
        cached = at(ev, self.d, "c", 12)
        self.assertEqual(direct.positions.tobytes(), cached.positions.tobytes())
        self.assertEqual(cached.stream.run, direct.stream.run)

    def test_frames_are_checkpointed_to_disk_and_a_new_session_does_not_re_solve(self):
        ev = self.evaluator()
        at(ev, self.d, "c", 40)
        self.assertGreaterEqual(ev.sim_store(256, 2048).stats()["disk_entries"], 40)
        again = self.evaluator()                                   # a restart: nothing in memory
        start = steps()
        value = at(again, self.d, "c", 25)
        self.assertEqual(steps() - start, 0)
        self.assertEqual(len(value), 1250)
        at(again, self.d, "c", 41)
        self.assertEqual(steps() - start, 1)

    def test_scrub_to_frame_50_after_solving_to_100_does_not_re_solve(self):
        ev = self.evaluator()
        at(ev, self.d, "c", 100)
        start = steps()
        self.assertEqual(len(at(ev, self.d, "c", 50)), 2500)
        self.assertEqual(steps() - start, 0)

    def test_changing_the_emitter_invalidates_the_cache_node_too(self):
        ev = self.evaluator()
        at(ev, self.d, "c", 20)
        set_(self.d, "e", seed=5)
        start = steps()
        at(ev, self.d, "c", 20)
        self.assertEqual(steps() - start, 20)

    def test_memory_and_disk_budgets_bound_the_stores(self):
        set_(self.d, "e", emit_rate=2000.0)
        set_(self.d, "c", cache_memory_mb=1, cache_disk_mb=3)
        ev = self.evaluator()
        at(ev, self.d, "c", 40)
        stats = ev.sim_store(1, 3).stats()
        self.assertLessEqual(stats["memory_bytes"], 1 << 20)
        self.assertLessEqual(stats["disk_bytes"], 3 << 20)
        self.assertGreater(stats["evictions"], 0)                  # the disk budget was actually hit
        self.assertGreater(stats["writes"], 0)

    def test_a_disk_budget_of_zero_keeps_the_cache_in_memory_only(self):
        set_(self.d, "c", cache_disk_mb=0)
        ev = self.evaluator()
        at(ev, self.d, "c", 10)
        store = ev.sim_store(256, 0)
        self.assertFalse(store.enabled)
        self.assertEqual(store.stats()["disk_entries"], 0)
        start = steps()
        at(ev, self.d, "c", 5)
        self.assertEqual(steps() - start, 0)

    def test_cancellation_reaches_the_cache_node(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(Cancelled):
            at(self.evaluator(), self.d, "c", 30, cancel=cancel)


class BypassTests(unittest.TestCase):
    def test_bypass_slots(self):
        def node(kind, **wired):
            slots = SPECS[kind]["inputs"] + SPECS[kind].get("optional_inputs", [])
            return {"type": kind, "inputs": {slot: wired.get(slot) for slot in slots}}
        self.assertEqual(bypass_slot(node("ParticleEmitter3D", geo="g")), "geo")
        self.assertEqual(bypass_slot(node("ParticleEmitter3D")), "geo")
        self.assertEqual(bypass_slot(node("ParticleCache3D", particles="p")), "particles")

    def test_a_bypassed_emitter_passes_its_geometry_input(self):
        d = Dispatcher()
        make(d, card=("Card3D", {"tx": 3.0}), e=("ParticleEmitter3D", {}), s=("Scene3D", {}))
        wire(d, "e", "geo", "card")
        wire(d, "s", "object0", "e")
        d.execute({"op": "disable", "id": "e", "value": True})
        ev = Evaluator()
        passed = at(ev, d, "e", 5)
        card = at(ev, d, "card", 5)
        self.assertIsInstance(passed, scene3d.Geometry)
        np.testing.assert_array_equal(passed.vertices, card.vertices)
        scene = at(ev, d, "s", 5)
        self.assertEqual(len(scene.geometries), 1)
        self.assertEqual(scene.particles, ())

    def test_a_bypassed_emitter_with_nothing_wired_passes_nothing(self):
        d = Dispatcher()
        make(d, e=("ParticleEmitter3D", {}), s=("Scene3D", {}))
        wire(d, "s", "object0", "e")
        d.execute({"op": "disable", "id": "e", "value": True})
        start = steps()
        self.assertIsNone(at(Evaluator(), d, "e", 5))
        scene = at(Evaluator(), d, "s", 5)
        self.assertEqual((scene.geometries, scene.particles), ((), ()))
        self.assertEqual(steps() - start, 0)

    def test_a_bypassed_cache_node_passes_the_emitter_through_untouched(self):
        d = Dispatcher()
        make(d, e=("ParticleEmitter3D", {"emit_rate": 5.0}), c=("ParticleCache3D", {}))
        wire(d, "c", "particles", "e")
        d.execute({"op": "disable", "id": "c", "value": True})
        ev = Evaluator()
        passed, source = at(ev, d, "c", 8), at(ev, d, "e", 8)
        self.assertEqual(passed.positions.tobytes(), source.positions.tobytes())
        self.assertEqual(len(ev.sim_store(256, 2048)._memory), 0)     # the cache node solved nothing

    def test_a_cache_node_downstream_of_a_bypassed_emitter_passes_that_geometry(self):
        d = Dispatcher()
        make(d, card=("Card3D", {}), e=("ParticleEmitter3D", {}), c=("ParticleCache3D", {}))
        wire(d, "e", "geo", "card")
        wire(d, "c", "particles", "e")
        d.execute({"op": "disable", "id": "e", "value": True})
        self.assertIsInstance(at(Evaluator(), d, "c", 3), scene3d.Geometry)


class OldDocumentTests(unittest.TestCase):
    def test_a_document_without_particle_nodes_loads_and_renders_unchanged(self):
        d = Dispatcher()
        make(d, card=("Card3D", {"tz": -1.0}), s=("Scene3D", {}), cam=("Camera3D", {}),
             r=("Render3D", {"width": 96, "height": 54}))
        wire(d, "s", "object0", "card")
        wire(d, "r", "scene", "s")
        wire(d, "r", "camera", "cam")
        before = Evaluator().evaluate(dict(d.document, view="r"), frame=1)
        reloaded = upgrade_document(json.loads(json.dumps(d.document)))
        validate(reloaded)
        after = Evaluator().evaluate(dict(reloaded, view="r"), frame=1)
        np.testing.assert_array_equal(before, after)
        scene = Evaluator().evaluate_raster(reloaded, "s", typed=True)
        reference = scene3d.render(scene3d.Scene(scene.geometries, scene.lights, scene.splats),
                                   scene3d.Camera(), 96, 54, (0.0, 0.0, 0.0, 0.0), samples=2, shadows=True)
        np.testing.assert_array_equal(before, reference)


if __name__ == "__main__":
    unittest.main()
