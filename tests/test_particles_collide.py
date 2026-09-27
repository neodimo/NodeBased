"""Lane L4 step B of 2: particles colliding with each other.

`ParticleCollide3D` resolves particles against each other as spheres of `collide_radius` or
`radius_from_size`, with a uniform-grid broad phase rebuilt every substep and `iterations` passes
of position-based contact correction (`restitution`, `friction`, `sleep_threshold`). It composes
with `ParticleBounce3D` (moving or not) and `ParticleCache3D`. See docs/SIMULATION.md,
"Particle-particle collisions".
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
FLOOR = {"card_width": 4000.0, "card_height": 4000.0, "rx": -90.0}
BOX_WALLS = (
    ("floor", {"card_width": 3.0, "card_height": 3.0, "rx": -90.0}),
    ("w1", {"card_width": 3.0, "card_height": 1.6, "ry": 0.0, "tz": -1.5, "ty": 0.8}),
    ("w2", {"card_width": 3.0, "card_height": 1.6, "ry": 180.0, "tz": 1.5, "ty": 0.8}),
    ("w3", {"card_width": 3.0, "card_height": 1.6, "ry": 90.0, "tx": -1.5, "ty": 0.8}),
    ("w4", {"card_width": 3.0, "card_height": 1.6, "ry": -90.0, "tx": 1.5, "ty": 0.8}),
)


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


def collider(**overrides):
    return particles.ParticleSelfCollider("ParticleCollide3D",
                                          {**SPECS["ParticleCollide3D"]["params"], **overrides})


def box_document(iterations=4, restitution=0.2, friction=0.5, sleep_threshold=0.02, radius=0.03,
                 bounce=0.2, bounce_friction=0.5, seed=5, max_particles=2000, emit_rate=200.0):
    d = Dispatcher()
    make(d, e=("ParticleEmitter3D", {"emit_rate": emit_rate, "emit_speed": 0.1, "spread": 40.0,
                                    "life": 1.0e6, "ty": 1.6, "seed": seed, "substeps": 2,
                                    "particle_size": radius * 2.0, "max_particles": max_particles}),
         g=("ParticleGravity3D", {"strength": G}),
         m=("MergeGeo3D", {}),
         b=("ParticleBounce3D", {"bounce": bounce, "friction": bounce_friction}),
         c=("ParticleCollide3D", {"radius_from_size": 1, "iterations": iterations,
                                  "restitution": restitution, "friction": friction,
                                  "sleep_threshold": sleep_threshold}))
    for name, params in BOX_WALLS:
        make(d, **{name: ("Card3D", params)})
    wire(d, "g", "particles", "e")
    wire(d, "b", "particles", "g")
    for i, (name, _) in enumerate(BOX_WALLS):
        wire(d, "m", f"geo{i}", name)
    wire(d, "b", "geometry", "m")
    wire(d, "c", "particles", "b")
    return d


def overlaps(positions, diameter):
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    return diameter - dist.min(axis=1)


class RegistrationTests(unittest.TestCase):
    def test_collide_is_registered_everywhere(self):
        kind = "ParticleCollide3D"
        self.assertIn(kind, SPECS)
        self.assertEqual(OUTPUT_TYPES[kind], "particles")
        self.assertEqual(SPECS[kind]["inputs"], ["particles"])
        self.assertNotIn("optional_inputs", SPECS[kind])
        self.assertIn(kind, COLORS)
        self.assertIn(kind, REGION_RULES)
        laid_out = sorted(p for group in knob_layout(kind) for p in group.params)
        self.assertEqual(laid_out, sorted(SPECS[kind]["params"]))
        for name, value in SPECS[kind]["params"].items():
            if not isinstance(value, str):
                self.assertIn(name, LIMITS, name)
        self.assertIn(kind, particles.FORCE_KINDS)
        self.assertEqual(bypass_slot({"type": kind, "inputs": {"particles": "e"}, "params": {}}),
                         "particles")

    def test_wiring_validates(self):
        d = box_document()
        validate(d.document)


class HeadOnPhysicsTests(unittest.TestCase):
    """Direct tests of the solver on two particles, no geometry involved."""

    def resolve(self, restitution, dx=0.3, vx=1.0, radius=0.5):
        force = collider(collide_radius=radius, restitution=restitution, friction=0.0, iterations=4)
        ids = np.array([0, 1], dtype=np.int64)
        position = np.array([[-dx, 0, 0], [dx, 0, 0]], dtype=np.float32)
        velocity = np.array([[vx, 0, 0], [-vx, 0, 0]], dtype=np.float32)
        size = np.full(2, radius * 2.0, np.float32)
        return particles.resolve_particle_collisions(force, 1, ids, position, velocity, size)

    def test_restitution_one_exchanges_velocities_head_on(self):
        pos, vel = self.resolve(1.0)
        np.testing.assert_allclose(vel[0], [-1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(vel[1], [1.0, 0.0, 0.0], atol=1e-6)
        # they end up just touching, not still overlapping
        self.assertAlmostEqual(float(np.linalg.norm(pos[0] - pos[1])), 1.0, delta=1e-6)

    def test_restitution_zero_gives_no_rebound(self):
        pos, vel = self.resolve(0.0)
        # neither reverses direction: A keeps moving toward +x (or stops), never back toward -x
        self.assertGreaterEqual(float(vel[0, 0]), -1e-6)
        self.assertLessEqual(float(vel[1, 0]), 1e-6)
        np.testing.assert_allclose(vel[0], vel[1] * -1.0, atol=1e-6)   # they end up co-moving

    def test_no_overlap_no_effect(self):
        force = collider(collide_radius=0.1)
        ids = np.array([0, 1], dtype=np.int64)
        position = np.array([[-1.0, 0, 0], [1.0, 0, 0]], dtype=np.float32)
        velocity = np.array([[1.0, 0, 0], [-1.0, 0, 0]], dtype=np.float32)
        size = np.full(2, 0.2, np.float32)
        pos, vel = particles.resolve_particle_collisions(force, 1, ids, position, velocity, size)
        np.testing.assert_array_equal(pos, position)
        np.testing.assert_array_equal(vel, velocity)

    def test_order_of_the_arrays_does_not_change_the_result(self):
        force = collider(restitution=0.6, friction=0.2, collide_radius=0.5)
        ids = np.array([0, 1], dtype=np.int64)
        position = np.array([[-0.3, 0.01, 0], [0.3, -0.02, 0.01]], dtype=np.float32)
        velocity = np.array([[1.0, 0.1, 0], [-1.0, 0.0, -0.05]], dtype=np.float32)
        size = np.full(2, 1.0, np.float32)
        pos_a, vel_a = particles.resolve_particle_collisions(force, 1, ids, position, velocity, size)
        pos_b, vel_b = particles.resolve_particle_collisions(
            force, 1, ids[::-1].copy(), position[::-1].copy(), velocity[::-1].copy(), size[::-1].copy())
        np.testing.assert_allclose(pos_a, pos_b[::-1], atol=1e-6)
        np.testing.assert_allclose(vel_a, vel_b[::-1], atol=1e-6)

    def test_probability_and_frame_window(self):
        force = collider(collide_radius=0.5, restitution=1.0, probability=0.0)
        ids = np.array([0, 1], dtype=np.int64)
        position = np.array([[-0.3, 0, 0], [0.3, 0, 0]], dtype=np.float32)
        velocity = np.array([[1.0, 0, 0], [-1.0, 0, 0]], dtype=np.float32)
        size = np.full(2, 1.0, np.float32)
        pos, vel = particles.resolve_particle_collisions(force, 1, ids, position, velocity, size)
        np.testing.assert_array_equal(vel, velocity)      # probability 0: nothing selected, no effect
        force = collider(collide_radius=0.5, restitution=1.0, from_frame=10)
        pos, vel = particles.resolve_particle_collisions(force, 1, ids, position, velocity, size)
        np.testing.assert_array_equal(vel, velocity)       # frame 1 is before the window
        pos, vel = particles.resolve_particle_collisions(force, 10, ids, position, velocity, size)
        np.testing.assert_allclose(vel[0], [-1.0, 0.0, 0.0], atol=1e-6)


class PileSettlingTests(unittest.TestCase):
    def test_a_pile_settles_with_bounded_overlap_and_low_kinetic_energy(self):
        d = box_document()
        ev = Evaluator()
        value = at(ev, d, "c", 150)
        self.assertEqual(len(value), 2000)
        positions = value.positions.astype(np.float64)
        velocities = value.velocities.astype(np.float64)
        diameter = 0.06
        overlap = overlaps(positions, diameter)
        kinetic_energy = float(0.5 * (velocities ** 2).sum())
        self.assertLess(kinetic_energy, 1.0)
        # stayed inside the box: never below the floor, never outside the walls
        self.assertGreaterEqual(float(positions[:, 1].min()), -1e-3)
        self.assertLess(float(np.abs(positions[:, 0]).max()), 1.5)
        self.assertLess(float(np.abs(positions[:, 2]).max()), 1.5)
        # colliding with each other spreads the pile out: far less overlap than with no collision
        # at all (every particle landing on top of the last one, near-total overlap)
        d_bypassed = box_document()
        d_bypassed.document["nodes"]["c"]["disabled"] = True
        baseline = at(Evaluator(), d_bypassed, "c", 150).positions.astype(np.float64)
        baseline_overlap = overlaps(baseline, diameter)
        self.assertLess(overlap.max(), 0.7 * baseline_overlap.max())

    def test_kinetic_energy_keeps_falling_as_the_pile_settles(self):
        d = box_document()
        ev = Evaluator()
        early = at(ev, d, "c", 20).velocities.astype(np.float64)
        late = at(ev, d, "c", 150).velocities.astype(np.float64)
        self.assertLess(float(0.5 * (late ** 2).sum()), float(0.5 * (early ** 2).sum()))


class BypassTests(unittest.TestCase):
    def test_a_bypassed_collide_passes_the_particles_through_unchanged(self):
        d = box_document()
        d.document["nodes"]["c"]["disabled"] = True
        ev = Evaluator()
        bypassed = at(ev, d, "c", 60)
        upstream = at(ev, d, "b", 60)
        np.testing.assert_array_equal(bypassed.positions, upstream.positions)
        np.testing.assert_array_equal(bypassed.velocities, upstream.velocities)


class DeterminismTests(unittest.TestCase):
    def test_same_seed_same_arrays(self):
        d = box_document()
        a = at(Evaluator(), d, "c", 60)
        b = at(Evaluator(), d, "c", 60)
        np.testing.assert_array_equal(a.positions, b.positions)
        np.testing.assert_array_equal(a.velocities, b.velocities)

    def test_solving_in_one_jump_equals_scrubbing(self):
        d = box_document()
        jump = at(Evaluator(), d, "c", 45)
        ev = Evaluator()
        for frame in range(1, 46):
            walked = at(ev, d, "c", frame)
        order_a, order_b = np.argsort(jump.ids), np.argsort(walked.ids)
        np.testing.assert_array_equal(jump.positions[order_a], walked.positions[order_b])

    def test_changing_a_knob_changes_the_run(self):
        d = box_document()
        run = at(Evaluator(), d, "c", 10).stream.run
        set_(d, "c", restitution=0.9)
        self.assertNotEqual(at(Evaluator(), d, "c", 10).stream.run, run)


class CacheRoundTripTests(unittest.TestCase):
    def test_a_cache_behind_a_collide_scrubs_without_re_solving(self):
        d = box_document(max_particles=200, emit_rate=40.0)
        make(d, cache=("ParticleCache3D", {}))
        wire(d, "cache", "particles", "c")
        ev = Evaluator()
        first = at(ev, d, "cache", 30)
        before = particles.SOLVER_STATS["steps"]
        again = at(ev, d, "cache", 30)
        self.assertEqual(particles.SOLVER_STATS["steps"], before)
        np.testing.assert_array_equal(first.positions, again.positions)
        set_(d, "c", restitution=0.9)
        changed = at(ev, d, "cache", 30)
        self.assertGreater(particles.SOLVER_STATS["steps"], before)
        self.assertNotEqual(changed.stream.run, first.stream.run)


class MovingColliderCompatibilityTests(unittest.TestCase):
    def test_a_moving_collider_and_self_collision_chain_without_tunnelling(self):
        """A sphere sweeps through a field of particles that also collide with each other; every
        particle stays outside the sphere's swept volume over the run, matching the plain moving-
        collider guarantee in tests/test_particles_animated.py."""
        d = Dispatcher()
        make(d, e=("ParticleEmitter3D", {"emit_from": "volume", "emit_rate": 100.0, "emit_speed": 0.0,
                                        "life": 100000.0, "max_particles": 400, "seed": 5,
                                        "particle_size": 0.3}),
             field=("Cube3D", {"cube_size": 40.0}),
             sphere=("Sphere3D", {"sphere_radius": 1.0, "segments": 10, "tx": -60.0}),
             b=("ParticleBounce3D", {"bounce": 0.4, "friction": 0.0, "animated": 1}),
             c=("ParticleCollide3D", {"radius_from_size": 1, "iterations": 2, "restitution": 0.3,
                                      "friction": 0.2, "sleep_threshold": 0.0}))
        wire(d, "e", "geo", "field")
        wire(d, "b", "particles", "e")
        wire(d, "b", "geometry", "sphere")
        wire(d, "c", "particles", "b")
        for frame in (1, 60):
            d.execute({"op": "set_key", "id": "sphere", "param": "tx", "frame": frame,
                      "value": -60.0 + 10.0 * (frame - 1)})
        ev = Evaluator()
        radius = 1.0
        for frame in range(1, 61):
            value = at(ev, d, "c", frame)
            if not len(value):
                continue
            centre = np.array((-60.0 + 10.0 * (frame - 1), 0.0, 0.0))
            depth = radius - np.linalg.norm(value.positions - centre, axis=1)
            self.assertLess(float(depth.max()), 1e-2, frame)


if __name__ == "__main__":
    unittest.main()
