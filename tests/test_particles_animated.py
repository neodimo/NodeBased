"""Lane L4 step A of 2: moving colliders and moving emitters for particles.

`ParticleBounce3D.animated` resamples its geometry per frame and interpolates per substep, refits
its BVH when the collider's own triangle count is unchanged, and gives a hit the collider's own
velocity there so a moving surface throws particles. `ParticleEmitter3D.animated` resamples its
"geo" input per frame instead of freezing it at the start frame, and `inherit_velocity` carries the
emitter's own moving transform into a newborn particle's velocity. Off by default, so an existing
document solves bit-identically (`tests.test_particles_bounce`, `tests.test_particles_nodes` cover
that unchanged). See docs/SIMULATION.md, "Bounce and collisions (animated)" and "The emitter model".
"""
import json
import unittest

import numpy as np

from nodebased import particles, raytrace
from nodebased.core import CHOICES, LIMITS, OUTPUT_TYPES, SPECS, Dispatcher, bypass_slot, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.knobs import knob_layout
from nodebased.theme import COLORS
from nodebased.tiers import REGION_RULES

G = 0.02
FLOOR = {"card_width": 4000.0, "card_height": 4000.0, "rx": -90.0}


def make(d, **nodes):
    for key, (kind, params) in nodes.items():
        d.execute({"op": "create", "id": key, "type": kind, "params": params})


def wire(d, target, slot, source):
    d.execute({"op": "connect", "id": target, "input": slot, "source": source})


def set_(d, key, **params):
    for name, value in params.items():
        d.execute({"op": "set", "id": key, "param": name, "value": value})


def set_key(d, key, param, frame, value):
    d.execute({"op": "set_key", "id": key, "param": param, "frame": frame, "value": value})


def at(evaluator, d, key, frame):
    return evaluator.evaluate_raster(d.document, key, frame=frame, typed=True)


class RegistrationTests(unittest.TestCase):
    def test_animated_and_inherit_velocity_are_registered(self):
        for kind, names in (("ParticleBounce3D", ("animated",)),
                            ("ParticleEmitter3D", ("animated", "inherit_velocity"))):
            for name in names:
                self.assertIn(name, SPECS[kind]["params"])
                self.assertIn(name, LIMITS)
            laid_out = sorted(p for group in knob_layout(kind) for p in group.params)
            self.assertEqual(laid_out, sorted(SPECS[kind]["params"]))
        self.assertEqual(SPECS["ParticleBounce3D"]["params"]["animated"], 0)
        self.assertEqual(SPECS["ParticleEmitter3D"]["params"]["animated"], 0)
        self.assertEqual(SPECS["ParticleEmitter3D"]["params"]["inherit_velocity"], 0.0)


class BvhRefitTests(unittest.TestCase):
    def test_refit_matches_a_fresh_build_and_reuses_the_tree(self):
        lo = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [5.0, 5.0, 0.0]])
        hi = lo + 1.0
        bvh = raytrace.Bvh.build(lo, hi)
        moved_lo, moved_hi = lo + 2.0, hi + 2.0
        refit = bvh.refit(moved_lo, moved_hi)
        rebuilt = raytrace.Bvh.build(moved_lo, moved_hi)
        np.testing.assert_array_equal(refit.left, bvh.left)
        np.testing.assert_array_equal(refit.right, bvh.right)
        np.testing.assert_array_equal(refit.prim_order, bvh.prim_order)
        np.testing.assert_array_equal(refit.node_lo, rebuilt.node_lo)
        np.testing.assert_array_equal(refit.node_hi, rebuilt.node_hi)

    def test_refit_rejects_a_changed_primitive_count(self):
        bvh = raytrace.Bvh.build(np.zeros((3, 3)), np.ones((3, 3)))
        with self.assertRaises(ValueError):
            bvh.refit(np.zeros((2, 3)), np.ones((2, 3)))


class NoTunnellingMovingColliderTests(unittest.TestCase):
    def test_a_translating_sphere_never_leaves_a_static_particle_inside_it(self):
        """A sphere sweeps at 10 units/frame through a field of motionless particles; over 200
        frames none is ever found inside it (a swept-prism, not a two-snapshot, test)."""
        d = Dispatcher()
        make(d, e=("ParticleEmitter3D", {"emit_from": "volume", "emit_rate": 400.0, "emit_speed": 0.0,
                                        "life": 100000.0, "max_particles": 5000, "seed": 5}),
             field=("Cube3D", {"cube_size": 40.0}),
             sphere=("Sphere3D", {"sphere_radius": 1.0, "segments": 10, "tx": -60.0}),
             b=("ParticleBounce3D", {"bounce": 0.4, "friction": 0.0, "animated": 1}))
        wire(d, "e", "geo", "field")
        wire(d, "b", "particles", "e")
        wire(d, "b", "geometry", "sphere")
        for frame in (1, 200):
            set_key(d, "sphere", "tx", frame, -60.0 + 10.0 * (frame - 1))
        ev = Evaluator()
        radius = 1.0
        for frame in range(1, 201):
            value = at(ev, d, "b", frame)
            if not len(value):
                continue
            centre = np.array((-60.0 + 10.0 * (frame - 1), 0.0, 0.0))
            depth = radius - np.linalg.norm(value.positions - centre, axis=1)
            self.assertLess(float(depth.max()), 1e-3, frame)

    def test_a_static_collider_is_unaffected_by_turning_animated_on(self):
        d = Dispatcher()
        make(d, e=("ParticleEmitter3D", {"emit_rate": 1.0, "emit_speed": 0.0, "life": 1000.0, "ty": 1.0}),
             g=("ParticleGravity3D", {"strength": G}), floor=("Card3D", FLOOR),
             b=("ParticleBounce3D", {"bounce": 0.5, "friction": 0.0}))
        wire(d, "g", "particles", "e")
        wire(d, "b", "particles", "g")
        wire(d, "b", "geometry", "floor")
        frozen = np.array([at(Evaluator(), d, "b", f).positions[0, 1] for f in range(1, 40)])
        set_(d, "b", animated=1)
        animated_off_motion = np.array([at(Evaluator(), d, "b", f).positions[0, 1] for f in range(1, 40)])
        np.testing.assert_allclose(frozen, animated_off_motion, atol=1e-9)


class RotatingPaddleTests(unittest.TestCase):
    def test_a_rotating_paddle_throws_a_resting_particle(self):
        """A Card3D "paddle" rotates about Y at a constant rate; a particle resting where the
        sweeping face reaches it is thrown, its speed of the same order as the paddle's own local
        speed there (bounce close to 1 makes the elastic-collision factor close to 2x)."""
        d = Dispatcher()
        radius, degrees_per_frame = 2.0, 6.0
        make(d, e=("ParticleEmitter3D", {"emit_rate": 1.0, "emit_speed": 0.0, "life": 100.0,
                                        "tx": radius, "ty": 0.0, "tz": -0.06, "substeps": 8}),
             paddle=("Card3D", {"card_width": 8.0, "card_height": 1.0}),
             b=("ParticleBounce3D", {"bounce": 0.9, "friction": 0.0, "animated": 1}))
        wire(d, "b", "particles", "e")
        wire(d, "b", "geometry", "paddle")
        # The paddle holds still through frame 1 (so the one particle, emitted over frame 1's own
        # substeps, is already alive) and only starts sweeping from frame 2 onward.
        for frame in (2, 45):
            set_key(d, "paddle", "ry", frame, degrees_per_frame * (frame - 2))
        ev = Evaluator()
        speeds = []
        for frame in range(1, 20):
            value = at(ev, d, "b", frame)
            if len(value):
                speeds.append(float(np.linalg.norm(value.velocities[0])))
        expected = math_radians_speed(radius, degrees_per_frame)
        self.assertGreater(max(speeds), 0.3 * expected)
        self.assertLess(max(speeds), 4.0 * expected)

    def test_a_frozen_paddle_never_throws_the_same_particle(self):
        d = Dispatcher()
        radius, degrees_per_frame = 2.0, 6.0
        make(d, e=("ParticleEmitter3D", {"emit_rate": 1.0, "emit_speed": 0.0, "life": 100.0,
                                        "tx": radius, "ty": 0.0, "tz": -0.06}),
             paddle=("Card3D", {"card_width": 8.0, "card_height": 1.0}),
             b=("ParticleBounce3D", {"bounce": 0.9, "friction": 0.0}))
        wire(d, "b", "particles", "e")
        wire(d, "b", "geometry", "paddle")
        for frame in (2, 45):
            set_key(d, "paddle", "ry", frame, degrees_per_frame * (frame - 2))
        ev = Evaluator()
        speeds = [float(np.linalg.norm(at(ev, d, "b", f).velocities[0])) for f in range(1, 20)]
        self.assertLess(max(speeds), 1e-6)


def math_radians_speed(radius, degrees_per_frame):
    import math
    return radius * math.radians(degrees_per_frame)


def _fake_geometry(frame, vertices, triangles):
    from nodebased.scene3d import Geometry
    parent = np.eye(4)
    parent[0, 3] = float(frame)               # translates along X by the frame number
    return Geometry(vertices, triangles, (1.0, 1.0, 1.0, 1.0), parent=parent)


class RefitPathTests(unittest.TestCase):
    def test_a_constant_triangle_count_reuses_and_refits_the_bvh(self):
        collider = particles.ParticleCollider("ParticleBounce3D", {"animated": 1, "bounce": 0.5,
                                              "friction": 0.0, "probability": 1.0, "from_frame": -10 ** 6,
                                              "to_frame": 10 ** 6, "seed": 0})
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32)
        triangles = np.array([[0, 1, 2]], np.int32)

        def provider(frame):
            return _fake_geometry(frame, vertices, triangles)
        collider.track = particles.GeometryTrack(provider, True, 1)
        origins = np.zeros((1, 3))
        displacements = np.zeros((1, 3))
        collider.first_hit(origins, displacements, 1, 0, 4)
        first_bvh, first_count = collider._object_bvh[0]
        collider.first_hit(origins, displacements, 1, 1, 4)
        second_bvh, second_count = collider._object_bvh[0]
        self.assertEqual(first_count, second_count)
        self.assertIs(first_bvh.left, second_bvh.left)
        self.assertIs(first_bvh.prim_order, second_bvh.prim_order)

    def test_a_changed_triangle_count_rebuilds(self):
        collider = particles.ParticleCollider("ParticleBounce3D", {"animated": 1, "bounce": 0.5,
                                              "friction": 0.0, "probability": 1.0, "from_frame": -10 ** 6,
                                              "to_frame": 10 ** 6, "seed": 0})
        one_v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32)
        one_t = np.array([[0, 1, 2]], np.int32)
        two_v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [5.0, 5.0, 5.0]], np.float32)
        two_t = np.array([[0, 1, 2], [0, 1, 3]], np.int32)

        def provider(frame):
            return _fake_geometry(frame, one_v, one_t) if frame < 5 else _fake_geometry(frame, two_v, two_t)
        collider.track = particles.GeometryTrack(provider, True, 1)
        origins, displacements = np.zeros((1, 3)), np.zeros((1, 3))
        collider.first_hit(origins, displacements, 1, 0, 1)
        self.assertEqual(collider._object_bvh[0][1], 1)          # one triangle
        collider.first_hit(origins, displacements, 5, 0, 1)
        self.assertEqual(collider._object_bvh[0][1], 2)          # two triangles: rebuilt


class InheritVelocityTests(unittest.TestCase):
    def dropped_from_a_translating_emitter(self, inherit):
        d = Dispatcher()
        make(d, e=("ParticleEmitter3D", {"emit_rate": 1.0, "emit_speed": 0.0, "life": 100.0,
                                        "animated": 1, "inherit_velocity": inherit}))
        for frame in (1, 20):
            set_key(d, "e", "tx", frame, 3.0 * (frame - 1))
        ev = Evaluator()
        return at(ev, d, "e", 3)

    def test_inherit_velocity_one_carries_the_emitters_own_speed(self):
        carried = self.dropped_from_a_translating_emitter(1.0)
        none = self.dropped_from_a_translating_emitter(0.0)
        self.assertAlmostEqual(float(carried.velocities[0, 0]), 3.0, delta=0.05)
        self.assertAlmostEqual(float(none.velocities[0, 0]), 0.0, delta=1e-6)


class AnimatedEmitterGeometryTests(unittest.TestCase):
    def test_animated_emitter_geometry_samples_the_current_frame_not_the_start_frame(self):
        d = Dispatcher()
        make(d, card=("Card3D", {"card_width": 0.001, "card_height": 0.001}),
             e=("ParticleEmitter3D", {"emit_from": "vertices", "emit_rate": 1.0, "emit_speed": 0.0,
                                      "life": 100.0, "animated": 1}))
        wire(d, "e", "geo", "card")
        for frame in (1, 10):
            set_key(d, "card", "tx", frame, 9.0 * (frame - 1))
        ev = Evaluator()
        early = at(ev, d, "e", 1).positions[0, 0]
        late = at(ev, d, "e", 9)
        # The 9th particle born (frame 9) is placed at the geometry's frame-9 position, not frame 1's.
        self.assertAlmostEqual(float(early), 0.0, delta=0.01)
        self.assertGreater(float(late.positions[-1, 0]), 5.0)

    def test_animated_off_keeps_the_geometry_frozen_at_the_start_frame(self):
        d = Dispatcher()
        make(d, card=("Card3D", {"card_width": 0.001, "card_height": 0.001}),
             e=("ParticleEmitter3D", {"emit_from": "vertices", "emit_rate": 1.0, "emit_speed": 0.0,
                                      "life": 100.0}))
        wire(d, "e", "geo", "card")
        for frame in (1, 10):
            set_key(d, "card", "tx", frame, 9.0 * (frame - 1))
        ev = Evaluator()
        late = at(ev, d, "e", 9)
        self.assertAlmostEqual(float(late.positions[-1, 0]), 0.0, delta=0.01)


class IdentityTests(unittest.TestCase):
    def big_with_animated_bounce(self):
        d = Dispatcher()
        make(d, e=("ParticleEmitter3D", {"emit_rate": 20.0, "emit_speed": 0.4, "spread": 90.0, "life": 100.0,
                                        "ty": 1.0, "seed": 3}),
             floor=("Card3D", {**FLOOR, "card_width": 40.0, "card_height": 40.0}),
             b=("ParticleBounce3D", {"bounce": 0.5, "friction": 0.1, "animated": 1}),
             c=("ParticleCache3D", {}))
        wire(d, "b", "particles", "e")
        wire(d, "b", "geometry", "floor")
        wire(d, "c", "particles", "b")
        for frame in (1, 60):
            set_key(d, "floor", "ty", frame, -0.01 * (frame - 1))
        return d

    def test_changing_a_collider_keyframe_invalidates_the_cache_and_an_unrelated_node_does_not(self):
        d = self.big_with_animated_bounce()
        ev = Evaluator()
        run_before = at(ev, d, "c", 30).stream.run
        # An unrelated node: a second, unwired card. No effect on the run.
        make(d, spare=("Card3D", {}))
        set_(d, "spare", tx=5.0)
        self.assertEqual(at(ev, d, "c", 30).stream.run, run_before)
        # Editing the collider's animation keyframe changes the run.
        set_key(d, "floor", "ty", 60, -2.0)
        self.assertNotEqual(at(ev, d, "c", 30).stream.run, run_before)

    def test_the_same_animation_produces_the_same_run_at_different_frames(self):
        d = self.big_with_animated_bounce()
        ev = Evaluator()
        run_5 = at(ev, d, "c", 5).stream.run
        run_50 = at(ev, d, "c", 50).stream.run
        self.assertEqual(run_5, run_50)


class DeterminismTests(unittest.TestCase):
    def test_animated_solves_are_deterministic_and_scrub_equals_a_jump(self):
        d = Dispatcher()
        make(d, e=("ParticleEmitter3D", {"emit_rate": 15.0, "emit_speed": 0.3, "spread": 60.0, "life": 60.0,
                                        "ty": 1.0, "seed": 9}),
             floor=("Card3D", {**FLOOR, "card_width": 40.0, "card_height": 40.0}),
             b=("ParticleBounce3D", {"bounce": 0.5, "friction": 0.1, "animated": 1}))
        wire(d, "b", "particles", "e")
        wire(d, "b", "geometry", "floor")
        for frame in (1, 40):
            set_key(d, "floor", "ty", frame, -0.02 * (frame - 1))
        jump = at(Evaluator(), d, "b", 30)
        ev = Evaluator()
        for frame in range(1, 31):
            walked = at(ev, d, "b", frame)
        order_a, order_b = np.argsort(jump.ids), np.argsort(walked.ids)
        np.testing.assert_array_equal(jump.positions[order_a], walked.positions[order_b])


class OldDocumentTests(unittest.TestCase):
    def test_a_document_without_the_new_knobs_loads_and_solves_as_before(self):
        d = Dispatcher()
        make(d, e=("ParticleEmitter3D", {"emit_rate": 1.0, "emit_speed": 0.0, "life": 1000.0, "ty": 1.0}),
             g=("ParticleGravity3D", {"strength": G}), floor=("Card3D", FLOOR),
             b=("ParticleBounce3D", {"bounce": 0.5, "friction": 0.0}))
        wire(d, "g", "particles", "e")
        wire(d, "b", "particles", "g")
        wire(d, "b", "geometry", "floor")
        before = at(Evaluator(), d, "b", 20).positions.copy()
        stripped = json.loads(json.dumps(d.document))
        for key in ("e", "b"):
            for name in ("animated", "inherit_velocity"):
                stripped["nodes"][key]["params"].pop(name, None)
        reloaded = upgrade_document(stripped)
        validate(reloaded)
        self.assertEqual(reloaded["nodes"]["b"]["params"]["animated"], 0)
        self.assertEqual(reloaded["nodes"]["e"]["params"]["inherit_velocity"], 0.0)
        after = Evaluator().evaluate_raster(reloaded, "b", frame=20, typed=True).positions
        np.testing.assert_array_equal(before, after)


if __name__ == "__main__":
    unittest.main()
