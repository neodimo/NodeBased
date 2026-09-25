"""Lane L5 step 2c, part 1: ParticleBounce3D collides particles with geometry.

Geometry is a Card3D floor (rotated flat) or wall; the solver sweeps every substep's motion against its
triangles, so the checks are analytic heights and speeds, counts, and 200-frame no-tunnelling runs.
"""
import unittest

import numpy as np

from nodebased import particles
from nodebased.core import CHOICES, LIMITS, OUTPUT_TYPES, SPECS, Dispatcher, bypass_slot, validate
from nodebased.imaging import Evaluator
from nodebased.knobs import knob_layout
from nodebased.theme import COLORS
from nodebased.tiers import REGION_RULES

G = 0.02


def make(d, **nodes):
    for key, (kind, params) in nodes.items():
        d.execute({"op": "create", "id": key, "type": kind, "params": params})


def wire(d, target, slot, source):
    d.execute({"op": "connect", "id": target, "input": slot, "source": source})


def set_(d, key, **params):
    for name, value in params.items():
        d.execute({"op": "set", "id": key, "param": name, "value": value})


def at(evaluator, d, key, frame):
    return evaluator.evaluate_raster(d.document, key, frame=frame, typed=True)


FLOOR = {"card_width": 4000.0, "card_height": 4000.0, "rx": -90.0}


def dropped(emitter=None, bounce=None, gravity=G, floor=None, substeps=16, scene=False):
    """One particle emitted at rest 1 unit above a horizontal floor, gravity, then a bounce node."""
    d = Dispatcher()
    make(d, e=("ParticleEmitter3D", {"emit_rate": 1.0, "emit_speed": 0.0, "life": 1000.0, "ty": 1.0,
                                    "substeps": substeps, **(emitter or {})}),
         g=("ParticleGravity3D", {"strength": gravity}),
         floor=("Card3D", floor or FLOOR),
         b=("ParticleBounce3D", {"bounce": 0.5, "friction": 0.0, **(bounce or {})}))
    wire(d, "g", "particles", "e")
    wire(d, "b", "particles", "g")
    if scene:
        make(d, s=("Scene3D", {}))
        wire(d, "s", "object0", "floor")
        wire(d, "b", "geometry", "s")
    else:
        wire(d, "b", "geometry", "floor")
    return d


def heights(d, frames, key="b"):
    ev = Evaluator()
    return np.array([at(ev, d, key, f).positions[0, 1] for f in frames])


class RegistrationTests(unittest.TestCase):
    def test_bounce_and_render_kinds_are_registered_everywhere(self):
        for kind in ("ParticleBounce3D", "ParticleRender3D"):
            self.assertIn(kind, SPECS)
            self.assertEqual(OUTPUT_TYPES[kind], "particles")
            self.assertEqual(SPECS[kind]["inputs"], ["particles"])
            self.assertIn(kind, COLORS)
            self.assertIn(kind, REGION_RULES)
            laid_out = sorted(p for group in knob_layout(kind) for p in group.params)
            self.assertEqual(laid_out, sorted(SPECS[kind]["params"]))
            for name, value in SPECS[kind]["params"].items():
                if not isinstance(value, str):
                    self.assertIn(name, LIMITS, (kind, name))
        self.assertEqual(SPECS["ParticleBounce3D"]["optional_inputs"], ["geometry"])
        self.assertEqual(SPECS["ParticleRender3D"]["optional_inputs"], ["image"])
        self.assertEqual(CHOICES["representation"], ["points", "spheres", "cards"])
        self.assertIn("ParticleBounce3D", particles.FORCE_KINDS)
        for kind in ("ParticleBounce3D", "ParticleRender3D"):
            self.assertEqual(bypass_slot({"type": kind, "inputs": {"particles": "e"}, "params": {}}), "particles")

    def test_wiring_validates_and_rejects_a_wrong_type(self):
        d = dropped()
        validate(d.document)
        make(d, img=("Constant", {}))
        with self.assertRaises(ValueError):
            wire(d, "b", "geometry", "img")
            validate(d.document)


class BounceHeightTests(unittest.TestCase):
    def test_restitution_half_rebounds_to_a_quarter_of_the_drop_height(self):
        y = heights(dropped(), range(1, 40))
        impact = int(np.argmax(y < 0.05))
        apex = y[impact + 1:impact + 12].max()
        # Rebound speed is e times the impact speed, so the apex is e squared times the drop height.
        self.assertAlmostEqual(apex, 0.25, delta=0.25 * 0.03)
        second = y[impact + 12:impact + 24].max()
        self.assertAlmostEqual(second, 0.0625, delta=0.0625 * 0.08)

    def test_restitution_one_returns_to_the_drop_height_and_zero_does_not_bounce(self):
        y = heights(dropped(bounce={"bounce": 1.0}), range(1, 45))
        impact = int(np.argmax(y < 0.05))
        self.assertAlmostEqual(y[impact + 1:].max(), 1.0, delta=0.03)
        dead = heights(dropped(bounce={"bounce": 0.0}), range(1, 45))
        self.assertLess(dead[impact + 2:].max(), 0.01)

    def test_substeps_only_refine_the_answer(self):
        for substeps in (4, 32):
            y = heights(dropped(substeps=substeps), range(1, 40))
            impact = int(np.argmax(y < 0.05))
            self.assertAlmostEqual(y[impact + 1:impact + 12].max(), 0.25, delta=0.25 * 0.06, msg=substeps)

    def test_the_particle_comes_to_rest_on_the_floor_and_stays_above_it(self):
        y = heights(dropped(), range(1, 90))
        self.assertLess(y[-1], 1e-3)
        self.assertGreaterEqual(y.min(), 0.0)
        np.testing.assert_allclose(y[-6:], y[-1], atol=2e-4)

    def test_a_translated_and_tilted_collider_is_used_in_world_space(self):
        d = dropped(floor={**FLOOR, "ty": -2.0})
        y = heights(d, range(1, 60))
        self.assertGreaterEqual(y.min(), -2.0)
        self.assertLess(y.min(), -1.9)                     # it lands on the floor at y = -2, not at 0
        wall = dropped(emitter={"ty": 0.0, "emit_speed": 0.2, "emit_dir_x": 1.0, "emit_dir_y": 0.0},
                       gravity=0.0, floor={"card_width": 50.0, "card_height": 50.0, "ry": 90.0, "tx": 1.0},
                       bounce={"bounce": 0.5})
        ev = Evaluator()
        xs = [at(ev, wall, "b", f).positions[0, 0] for f in range(1, 16)]
        self.assertGreater(max(xs[:6]), 0.9)               # it reaches the wall at x = 1
        self.assertLess(xs[-1], max(xs))                   # and comes back
        v = at(ev, wall, "b", 15).velocities[0]
        self.assertAlmostEqual(float(v[0]), -0.1, delta=1e-3)   # 0.5 x 0.2, reversed

    def test_a_scene_can_be_the_collider(self):
        a = heights(dropped(), range(1, 30))
        b = heights(dropped(scene=True), range(1, 30))
        np.testing.assert_allclose(a, b, atol=1e-6)

    def test_no_collider_wired_leaves_the_motion_alone(self):
        d = dropped()
        d.document["nodes"]["b"]["inputs"].pop("geometry")
        y = heights(d, range(1, 20))
        self.assertLess(y[-1], -0.5 * G * 18 ** 2 + 1.0 + 0.05)       # falls straight through nothing


class FrictionTests(unittest.TestCase):
    def sliding(self, friction, frames=30):
        d = dropped(emitter={"ty": 0.0005, "emit_speed": 1.0, "emit_dir_x": 1.0, "emit_dir_y": 0.0},
                    bounce={"bounce": 0.5, "friction": friction})
        ev = Evaluator()
        return at(ev, d, "b", frames)

    def test_friction_slows_a_sliding_particle(self):
        free, rough = self.sliding(0.0), self.sliding(0.5)
        self.assertAlmostEqual(float(free.velocities[0, 0]), 1.0, delta=1e-4)
        self.assertLess(float(rough.velocities[0, 0]), 0.95)
        self.assertLess(float(rough.positions[0, 0]), float(free.positions[0, 0]))
        self.assertLess(float(rough.positions[0, 1]), 0.01)              # it stayed on the floor

    def test_coulomb_deceleration_is_friction_times_gravity_whatever_the_substeps(self):
        # About 27 frames of contact at mu * g = 0.5 * 0.02 per frame squared.
        expected = 1.0 - 0.5 * G * (30 - 2.0)
        for substeps in (4, 16):
            d = dropped(emitter={"ty": 0.0005, "emit_speed": 1.0, "emit_dir_x": 1.0, "emit_dir_y": 0.0,
                                 "substeps": substeps},
                        bounce={"bounce": 0.5, "friction": 0.5})
            v = at(Evaluator(), d, "b", 30).velocities[0]
            self.assertAlmostEqual(float(v[0]), expected, delta=0.03, msg=substeps)

    def test_a_strong_enough_friction_stops_the_slide_without_reversing_it(self):
        v = self.sliding(50.0, frames=40).velocities[0]
        self.assertGreaterEqual(float(v[0]), 0.0)
        self.assertAlmostEqual(float(v[0]), 0.0, delta=1e-6)


class KillTests(unittest.TestCase):
    def test_kill_on_collision_removes_the_particle_at_the_floor(self):
        d = dropped(bounce={"kill_on_collision": 1})
        ev = Evaluator()
        first = [int((at(ev, d, "b", f).ids == 0).sum()) for f in range(1, 16)]   # particle 0 alone
        self.assertEqual(first[0], 1)
        self.assertEqual(first[8], 1)                                    # still falling at frame 9
        self.assertEqual(first[-1], 0)                                   # gone after the ten-frame fall
        self.assertAlmostEqual(first.index(0) + 1, 11, delta=1)
        self.assertEqual(int((at(ev, d, "b", 40).ids == 0).sum()), 0)

    def test_only_the_colliding_particles_are_killed(self):
        d = dropped(emitter={"emit_rate": 5.0})
        set_(d, "b", kill_on_collision=1)
        ev = Evaluator()
        seen = [len(at(ev, d, "b", f)) for f in (2, 6, 14, 30)]
        self.assertEqual(seen[0], 5 * 2 - 0)                             # nothing has landed yet
        self.assertGreater(seen[1], 0)
        self.assertLess(seen[3], seen[2] + 1)
        d2 = dropped(emitter={"emit_rate": 5.0})
        keep = len(at(Evaluator(), d2, "b", 30))
        self.assertGreater(keep, seen[3])                                # bouncing ones are still alive


class NoTunnellingTests(unittest.TestCase):
    def test_no_particle_passes_through_the_floor_over_200_frames_at_the_default_timestep(self):
        d = Dispatcher()
        make(d, e=("ParticleEmitter3D", {"emit_rate": 60.0, "emit_speed": 3.0, "spread": 180.0, "life": 400.0,
                                        "ty": 2.0, "seed": 7, "max_particles": 20000}),
             g=("ParticleGravity3D", {"strength": 0.05}), floor=("Card3D", {**FLOOR, "card_width": 1.0e5,
                                                                              "card_height": 1.0e5}),
             b=("ParticleBounce3D", {"bounce": 0.6, "friction": 0.2}))
        self.assertEqual(SPECS["ParticleEmitter3D"]["params"]["substeps"], 1)
        wire(d, "g", "particles", "e")
        wire(d, "b", "particles", "g")
        wire(d, "b", "geometry", "floor")
        ev = Evaluator()
        lowest, alive = 0.0, 0
        for frame in range(1, 201):
            value = at(ev, d, "b", frame)
            if len(value):
                lowest = min(lowest, float(value.positions[:, 1].min()))
                alive = max(alive, len(value))
        self.assertGreater(alive, 1000)
        self.assertGreaterEqual(lowest, -1e-6)

    def test_a_very_fast_particle_is_stopped_by_a_thin_floor(self):
        d = dropped(emitter={"emit_speed": 500.0, "emit_dir_y": -1.0, "substeps": 1}, gravity=0.0,
                    bounce={"bounce": 0.5})
        ev = Evaluator()
        for frame in range(1, 30):
            self.assertGreaterEqual(float(at(ev, d, "b", frame).positions[0, 1]), 0.0, frame)
        self.assertGreater(float(at(ev, d, "b", 3).velocities[0, 1]), 0.0)        # it rebounded

    def test_a_particle_in_a_v_shaped_corner_never_leaves_the_pair_of_walls(self):
        d = Dispatcher()
        make(d, e=("ParticleEmitter3D", {"emit_rate": 30.0, "emit_speed": 0.5, "spread": 180.0, "life": 500.0,
                                        "ty": 0.5, "seed": 3}),
             g=("ParticleGravity3D", {"strength": 0.03}),
             f=("Card3D", FLOOR), w=("Card3D", {"card_width": 4000.0, "card_height": 4000.0, "ry": 90.0, "tx": -1.0}),
             m=("MergeGeo3D", {}), b=("ParticleBounce3D", {"bounce": 0.7, "friction": 0.0}))
        wire(d, "g", "particles", "e")
        wire(d, "b", "particles", "g")
        wire(d, "m", "geo0", "f")
        wire(d, "m", "geo1", "w")
        wire(d, "b", "geometry", "m")
        ev = Evaluator()
        for frame in range(1, 120, 7):
            value = at(ev, d, "b", frame)
            if len(value):
                self.assertGreaterEqual(float(value.positions[:, 1].min()), -1e-6, frame)
                self.assertGreaterEqual(float(value.positions[:, 0].min()), -1.0 - 1e-6, frame)


class SeededSelectionTests(unittest.TestCase):
    def test_probability_zero_lets_everything_through_and_the_window_is_honoured(self):
        d = dropped(bounce={"probability": 0.0})
        self.assertLess(heights(d, [30])[0], 0.0)
        d = dropped(bounce={"from_frame": 20})
        self.assertLess(heights(d, [14])[0], 0.5)
        y = heights(d, range(1, 26))
        self.assertLess(y.min(), -0.1)                                   # first impact was before frame 20


class DeterminismTests(unittest.TestCase):
    def big(self):
        d = Dispatcher()
        make(d, e=("ParticleEmitter3D", {"emit_rate": 40.0, "emit_speed": 0.4, "spread": 90.0, "life": 200.0,
                                        "ty": 1.0, "seed": 11, "substeps": 2}),
             g=("ParticleGravity3D", {"strength": 0.03}), floor=("Card3D", {**FLOOR, "card_width": 60.0,
                                                                              "card_height": 60.0}),
             b=("ParticleBounce3D", {"bounce": 0.5, "friction": 0.3}))
        wire(d, "g", "particles", "e")
        wire(d, "b", "particles", "g")
        wire(d, "b", "geometry", "floor")
        return d

    def test_same_seed_same_arrays_with_collisions(self):
        d = self.big()
        a, b = at(Evaluator(), d, "b", 60), at(Evaluator(), d, "b", 60)
        self.assertGreater(len(a), 500)
        np.testing.assert_array_equal(a.positions, b.positions)
        np.testing.assert_array_equal(a.velocities, b.velocities)

    def test_solving_in_one_jump_equals_scrubbing(self):
        d = self.big()
        jump = at(Evaluator(), d, "b", 45)
        ev = Evaluator()
        for frame in range(1, 46):
            walked = at(ev, d, "b", frame)
        order_a, order_b = np.argsort(jump.ids), np.argsort(walked.ids)
        np.testing.assert_array_equal(jump.positions[order_a], walked.positions[order_b])

    def test_a_cache_behind_a_bounce_scrubs_without_re_solving_and_a_knob_change_re_solves(self):
        d = self.big()
        make(d, c=("ParticleCache3D", {}))
        wire(d, "c", "particles", "b")
        ev = Evaluator()
        first = at(ev, d, "c", 30)
        before = particles.SOLVER_STATS["steps"]
        again = at(ev, d, "c", 30)
        self.assertEqual(particles.SOLVER_STATS["steps"], before)
        np.testing.assert_array_equal(first.positions, again.positions)
        set_(d, "b", bounce=0.9)
        changed = at(ev, d, "c", 30)
        self.assertGreater(particles.SOLVER_STATS["steps"], before)
        self.assertNotEqual(changed.stream.run, first.stream.run)

    def test_moving_the_collider_changes_the_run(self):
        d = self.big()
        make(d, c=("ParticleCache3D", {}))
        wire(d, "c", "particles", "b")
        ev = Evaluator()
        run = at(ev, d, "c", 5).stream.run
        set_(d, "floor", ty=-1.0)
        self.assertNotEqual(at(ev, d, "c", 5).stream.run, run)

    def test_a_bypassed_bounce_passes_the_particles_through(self):
        d = dropped()
        d.document["nodes"]["b"]["disabled"] = True
        y = heights(d, [30])
        self.assertLess(y[0], 0.0)


if __name__ == "__main__":
    unittest.main()
