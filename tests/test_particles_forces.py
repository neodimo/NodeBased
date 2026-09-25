"""Lane L5 step 2b: gravity, drag, wind and turbulence force nodes.

Each force takes a particle set in and out, so they chain like Nuke's. The solve stays deterministic
and a force change reaches a downstream ParticleCache3D as a new run.
"""
import unittest

import numpy as np

from nodebased import particles
from nodebased.core import (Dispatcher, INPUT_TYPES, OUTPUT_TYPES, SPECS, CHOICES, LIMITS, bypass_slot,
                            validate)
from nodebased.imaging import Evaluator
from nodebased.knobs import knob_layout
from nodebased.theme import COLORS
from nodebased.tiers import REGION_RULES

FORCES = ("ParticleGravity3D", "ParticleDrag3D", "ParticleWind3D", "ParticleTurbulence3D")


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


def by_id(instance):
    order = np.argsort(instance.ids)
    return {"position": instance.positions[order], "velocity": instance.velocities[order],
            "ages": instance.ages[order], "ids": instance.ids[order]}


def resting(**extra):
    """An emitter whose particles are born at rest at the origin, one per substep set of frames."""
    return ("ParticleEmitter3D", {"emit_rate": 1.0, "emit_speed": 0.0, "life": 1000.0, **extra})


def cloud(count=2000, **extra):
    """Emit `count` resting particles on a large cube's surface in the first frame."""
    return {"e": ("ParticleEmitter3D", {"emit_from": "surface", "emit_rate": float(count),
                                        "emit_speed": 0.0, "life": 1000.0, **extra}),
            "cube": ("Cube3D", {"cube_size": 20.0})}


def build_cloud(count=2000, forces=(), emitter=None):
    d = Dispatcher()
    make(d, **cloud(count, **(emitter or {})))
    wire(d, "e", "geo", "cube")
    last = "e"
    for key, (kind, params) in forces:
        make(d, **{key: (kind, params)})
        wire(d, key, "particles", last)
        last = key
    return d, last


class RegistrationTests(unittest.TestCase):
    def test_force_kinds_are_registered_everywhere(self):
        for kind in FORCES:
            self.assertIn(kind, SPECS)
            self.assertEqual(OUTPUT_TYPES[kind], "particles")
            self.assertEqual(SPECS[kind]["inputs"], ["particles"])
            self.assertIn(kind, COLORS)
            self.assertIn(kind, REGION_RULES)
            laid_out = sorted(p for group in knob_layout(kind) for p in group.params)
            self.assertEqual(laid_out, sorted(SPECS[kind]["params"]))
            for name in ("probability", "from_frame", "to_frame", "seed"):
                self.assertIn(name, SPECS[kind]["params"])
            for name in SPECS[kind]["params"]:
                if not isinstance(SPECS[kind]["params"][name], str):
                    self.assertIn(name, LIMITS, (kind, name))
        self.assertIn("turb_mode", CHOICES)
        self.assertEqual(SPECS["ParticleGravity3D"]["params"]["gravity_y"], -1.0)

    def test_forces_wire_in_a_chain_and_to_a_cache(self):
        d = Dispatcher()
        make(d, e=("ParticleEmitter3D", {}), g=("ParticleGravity3D", {}), w=("ParticleWind3D", {}),
             t=("ParticleTurbulence3D", {}), r=("ParticleDrag3D", {}), c=("ParticleCache3D", {}),
             s=("Scene3D", {}), card=("Card3D", {}))
        for target, source in (("g", "e"), ("w", "g"), ("t", "w"), ("r", "t"), ("c", "r")):
            wire(d, target, "particles", source)
        wire(d, "s", "object0", "r")
        validate(d.document)
        with self.assertRaises(ValueError):
            wire(d, "g", "particles", "card")
            validate(d.document)

    def test_bypass_slot_is_particles(self):
        d = Dispatcher()
        make(d, e=("ParticleEmitter3D", {}), g=("ParticleGravity3D", {}))
        wire(d, "g", "particles", "e")
        for kind in FORCES:
            self.assertEqual(bypass_slot({"type": kind, "inputs": {"particles": "e"}, "params": {}}),
                             "particles")


class GravityTests(unittest.TestCase):
    def test_free_fall_matches_the_analytic_position(self):
        substeps, g = 16, 0.02
        d = Dispatcher()
        make(d, e=resting(substeps=substeps), g=("ParticleGravity3D", {"strength": g}))
        wire(d, "g", "particles", "e")
        fall = by_id(at(Evaluator(), d, "g", 12))
        n = np.rint(fall["ages"] * substeps).astype(int)
        self.assertGreater(len(n), 5)
        dt = 1.0 / substeps
        # The integrator's own closed form (semi-implicit Euler), exact up to float32 rounding.
        discrete = -g * dt * dt * n * (n + 1) / 2.0
        np.testing.assert_allclose(fall["position"][:, 1], discrete, rtol=1e-4, atol=1e-6)
        # And the analytic 1/2 g t^2, within the first-order timestep error g t dt / 2.
        t = n * dt
        analytic = -0.5 * g * t * t
        self.assertLessEqual(np.abs(fall["position"][:, 1] - analytic).max(),
                             0.5 * g * t.max() * dt + 1e-6)
        np.testing.assert_allclose(fall["position"][:, [0, 2]], 0.0, atol=1e-7)
        np.testing.assert_allclose(fall["velocity"][:, 1], -g * t, rtol=1e-4, atol=1e-6)

    def test_gravity_vector_and_strength(self):
        d = Dispatcher()
        make(d, e=resting(substeps=4),
             g=("ParticleGravity3D", {"gravity_x": 1.0, "gravity_y": 0.0, "gravity_z": 0.0, "strength": 0.1}))
        wire(d, "g", "particles", "e")
        v = by_id(at(Evaluator(), d, "g", 6))
        self.assertTrue((v["velocity"][:, 0] > 0).all())
        np.testing.assert_allclose(v["velocity"][:, 1:], 0.0, atol=1e-7)
        np.testing.assert_allclose(v["velocity"][:, 0], 0.1 * v["ages"], rtol=1e-4)


class DragTests(unittest.TestCase):
    def test_speed_decays_exponentially(self):
        substeps, drag = 4, 0.1
        d = Dispatcher()
        make(d, e=("ParticleEmitter3D", {"emit_rate": 1.0, "emit_speed": 1.0, "life": 1000.0,
                                         "substeps": substeps}),
             r=("ParticleDrag3D", {"drag": drag}))
        wire(d, "r", "particles", "e")
        v = by_id(at(Evaluator(), d, "r", 20))
        speed = np.linalg.norm(v["velocity"], axis=1)
        np.testing.assert_allclose(speed, np.exp(-drag * v["ages"]), rtol=1e-4)
        self.assertLess(speed[0], speed[-1])   # older particle is slower

    def test_quadratic_term_slows_fast_particles_more(self):
        d = Dispatcher()
        make(d, e=("ParticleEmitter3D", {"emit_rate": 1.0, "emit_speed": 4.0, "life": 1000.0}),
             r=("ParticleDrag3D", {"drag": 0.0, "drag_quadratic": 0.5}))
        wire(d, "r", "particles", "e")
        v = by_id(at(Evaluator(), d, "r", 3))
        self.assertLess(np.linalg.norm(v["velocity"][0]), 4.0)
        self.assertGreater(np.linalg.norm(v["velocity"][0]), 0.0)


class WindTests(unittest.TestCase):
    def test_wind_accelerates_along_its_direction(self):
        d = Dispatcher()
        make(d, e=resting(substeps=2),
             w=("ParticleWind3D", {"wind_x": 0.0, "wind_y": 3.0, "wind_z": 4.0, "strength": 0.05}))
        wire(d, "w", "particles", "e")
        v = by_id(at(Evaluator(), d, "w", 8))
        expected = np.array((0.0, 0.6, 0.8))
        for row, age in zip(v["velocity"], v["ages"]):
            np.testing.assert_allclose(row, expected * 0.05 * age, rtol=1e-4, atol=1e-7)

    def test_gust_is_repeatable_and_modulates_strength(self):
        def run(gust, seed=0):
            d = Dispatcher()
            make(d, e=resting(),
                 w=("ParticleWind3D", {"strength": 0.05, "wind_gust": gust, "wind_gust_rate": 0.4, "seed": seed}))
            wire(d, "w", "particles", "e")
            return by_id(at(Evaluator(), d, "w", 20))["velocity"]
        steady, gusty, again, other = run(0.0), run(0.8), run(0.8), run(0.8, seed=5)
        np.testing.assert_array_equal(gusty, again)
        self.assertFalse(np.allclose(gusty, steady))
        self.assertFalse(np.allclose(gusty, other))
        self.assertTrue((gusty[:, 0] > 0).all())    # a gust of 0.8 never reverses the wind


class TurbulenceTests(unittest.TestCase):
    def test_field_is_repeatable_with_zero_mean(self):
        points = np.random.default_rng(3).uniform(-50, 50, (20000, 3))
        for mode in ("curl", "gradient"):
            a = particles.turbulence_field(points, mode, 2.0, 3, 7)
            b = particles.turbulence_field(points.copy(), mode, 2.0, 3, 7)
            np.testing.assert_array_equal(a, b)
            self.assertGreater(a.std(), 0.05)
            self.assertLess(np.abs(a.mean(axis=0)).max(), 0.05 * a.std())
            other = particles.turbulence_field(points, mode, 2.0, 3, 8)
            self.assertFalse(np.allclose(a, other))

    def test_curl_field_is_divergence_free(self):
        points = np.random.default_rng(4).uniform(-5, 5, (200, 3))
        h = 1e-2
        divergence = 0.0
        for axis in range(3):
            step = np.zeros(3)
            step[axis] = h
            hi = particles.turbulence_field(points + step, "curl", 1.0, 2, 1)[:, axis]
            lo = particles.turbulence_field(points - step, "curl", 1.0, 2, 1)[:, axis]
            divergence = divergence + (hi - lo) / (2 * h)
        typical = np.abs(particles.turbulence_field(points, "curl", 1.0, 2, 1)).mean()
        self.assertLess(np.abs(divergence).mean(), 0.05 * typical)

    def test_node_is_repeatable_and_has_zero_mean_over_many_particles(self):
        knobs = {"turb_size": 2.0, "strength": 0.05, "octaves": 2, "seed": 3}
        results = []
        for _ in range(2):
            d, last = build_cloud(3000, [("t", ("ParticleTurbulence3D", knobs))])
            results.append(by_id(at(Evaluator(), d, last, 2)))
        np.testing.assert_array_equal(results[0]["velocity"], results[1]["velocity"])
        np.testing.assert_array_equal(results[0]["position"], results[1]["position"])
        velocity = results[0]["velocity"]
        self.assertGreater(velocity.std(), 1e-3)
        self.assertLess(np.abs(velocity.mean(axis=0)).max(), 0.1 * velocity.std())


class ProbabilityTests(unittest.TestCase):
    def test_fraction_of_particles_affected(self):
        d, last = build_cloud(4000, [("g", ("ParticleGravity3D", {"probability": 0.3, "seed": 2}))])
        v = by_id(at(Evaluator(), d, last, 1))
        affected = (v["velocity"][:, 1] != 0).mean()
        self.assertAlmostEqual(affected, 0.3, delta=0.04)

    def test_selection_is_stable_over_life_and_seeded(self):
        def chosen(seed, frame):
            d, last = build_cloud(1000, [("g", ("ParticleGravity3D", {"probability": 0.5, "seed": seed}))])
            v = by_id(at(Evaluator(), d, last, frame))
            return (v["velocity"][:, 1] != 0)
        first, later, other = chosen(1, 1), chosen(1, 3), chosen(2, 1)
        self.assertTrue((first <= later[:len(first)]).all() and (later[:len(first)] <= first).all())
        self.assertFalse((first == other).all())

    def test_zero_and_full_probability(self):
        for probability, expect in ((0.0, 0.0), (1.0, 1.0)):
            d, last = build_cloud(500, [("g", ("ParticleGravity3D", {"probability": probability}))])
            v = by_id(at(Evaluator(), d, last, 1))
            self.assertEqual((v["velocity"][:, 1] != 0).mean(), expect)


class WindowTests(unittest.TestCase):
    def test_force_acts_only_between_its_frames(self):
        d = Dispatcher()
        make(d, e=resting(), g=("ParticleGravity3D", {"strength": 0.1, "from_frame": 5, "to_frame": 6}))
        wire(d, "g", "particles", "e")
        ev = Evaluator()
        self.assertEqual(by_id(at(ev, d, "g", 4))["velocity"][:, 1].min(), 0.0)
        v6 = by_id(at(ev, d, "g", 6))["velocity"][:, 1]
        v9 = by_id(at(ev, d, "g", 9))["velocity"][:, 1]
        np.testing.assert_allclose(v6.min(), -0.2, rtol=1e-4)   # frames 5 and 6 kick once each
        self.assertEqual(v9.min(), v6.min())


class ChainTests(unittest.TestCase):
    def velocities(self, forces, frame=6, substeps=2):
        d, last = build_cloud(300, forces, emitter={"substeps": substeps})
        return by_id(at(Evaluator(), d, last, frame))["velocity"].astype(np.float64)

    def test_chained_forces_sum_their_accelerations(self):
        g = ("g", ("ParticleGravity3D", {"strength": 0.03}))
        w = ("w", ("ParticleWind3D", {"strength": 0.02, "wind_y": 1.0}))
        both = self.velocities([g, w])
        swapped = self.velocities([w, g])
        only_g, only_w = self.velocities([g]), self.velocities([w])
        np.testing.assert_allclose(both, only_g + only_w, atol=2e-6)
        np.testing.assert_allclose(swapped, both, atol=2e-6)

    def test_turbulence_chains_with_gravity_over_one_step(self):
        # Turbulence reads position, so accelerations only add exactly while the particles have not
        # yet moved: one substep from rest.
        g = ("g", ("ParticleGravity3D", {"strength": 0.03}))
        t = ("t", ("ParticleTurbulence3D", {"turb_size": 3.0, "strength": 0.04}))
        both = self.velocities([g, t], frame=1, substeps=1)
        np.testing.assert_allclose(both, self.velocities([g], 1, 1) + self.velocities([t], 1, 1), atol=2e-6)
        self.assertGreater(np.abs(both).max(), 0.01)


class DeterminismAndBypassTests(unittest.TestCase):
    def test_same_seed_same_arrays_with_forces(self):
        def run():
            d, last = build_cloud(800, [("g", ("ParticleGravity3D", {"probability": 0.6})),
                                        ("t", ("ParticleTurbulence3D", {"turb_size": 2.0})),
                                        ("r", ("ParticleDrag3D", {}))], emitter={"seed": 4, "substeps": 2})
            return at(Evaluator(), d, last, 5)
        a, b = run(), run()
        for name in ("positions", "velocities", "ids", "ages"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))

    def test_solving_in_one_jump_equals_scrubbing(self):
        d, last = build_cloud(200, [("g", ("ParticleGravity3D", {})), ("t", ("ParticleTurbulence3D", {}))])
        jump = at(Evaluator(), d, last, 7)
        ev = Evaluator()
        for frame in range(1, 8):
            walked = at(ev, d, last, frame)
        np.testing.assert_array_equal(jump.positions, walked.positions)

    def test_disabled_force_passes_the_particles(self):
        d, last = build_cloud(300, [("g", ("ParticleGravity3D", {"strength": 0.5}))])
        plain = at(Evaluator(), d, "e", 4)
        d.document["nodes"]["g"]["disabled"] = True
        passed = at(Evaluator(), d, "g", 4)
        np.testing.assert_array_equal(plain.positions, passed.positions)
        np.testing.assert_array_equal(plain.velocities, passed.velocities)

    def test_disabled_force_in_the_middle_of_a_chain_is_skipped(self):
        forces = [("g", ("ParticleGravity3D", {"strength": 0.05})), ("w", ("ParticleWind3D", {}))]
        d, last = build_cloud(200, forces)
        d.document["nodes"]["w"]["disabled"] = True
        d2, last2 = build_cloud(200, forces[:1])
        a, b = at(Evaluator(), d, last, 4), at(Evaluator(), d2, last2, 4)
        np.testing.assert_array_equal(a.velocities, b.velocities)


class CacheInvalidationTests(unittest.TestCase):
    def setUp(self):
        self.d = Dispatcher()
        make(self.d, e=resting(), g=("ParticleGravity3D", {"strength": 0.02}), c=("ParticleCache3D", {}))
        wire(self.d, "g", "particles", "e")
        wire(self.d, "c", "particles", "g")

    def test_force_knob_between_emitter_and_cache_invalidates(self):
        ev = Evaluator()
        first = by_id(at(ev, self.d, "c", 6))
        before = particles.SOLVER_STATS["steps"]
        again = by_id(at(ev, self.d, "c", 6))
        self.assertEqual(particles.SOLVER_STATS["steps"], before)     # cached: no solving
        np.testing.assert_array_equal(first["position"], again["position"])
        set_(self.d, "g", strength=0.2)
        after = by_id(at(ev, self.d, "c", 6))
        self.assertGreater(particles.SOLVER_STATS["steps"], before)   # a new run was solved
        self.assertLess(after["position"][:, 1].min(), first["position"][:, 1].min())
        settled = particles.SOLVER_STATS["steps"]
        at(ev, self.d, "c", 6)
        self.assertEqual(particles.SOLVER_STATS["steps"], settled)

    def test_emitter_is_not_solved_twice_behind_a_cache(self):
        ev = Evaluator()
        before = particles.SOLVER_STATS["steps"]
        at(ev, self.d, "c", 5)
        self.assertEqual(particles.SOLVER_STATS["steps"] - before, 5)   # one solve, five frames

    def test_reordering_or_adding_a_force_changes_the_run(self):
        ev = Evaluator()
        base = at(ev, self.d, "c", 3).stream.run
        make(self.d, w=("ParticleWind3D", {}))
        wire(self.d, "w", "particles", "g")
        wire(self.d, "c", "particles", "w")
        self.assertNotEqual(at(ev, self.d, "c", 3).stream.run, base)


if __name__ == "__main__":
    unittest.main()
