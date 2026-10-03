"""Whitewater secondary-particle behavior and its deterministic cache representation."""
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from nodebased import simcache, whitewater
from nodebased.core import Dispatcher, INPUT_TYPES, OUTPUT_TYPES, SPECS
from nodebased.imaging import Evaluator
from nodebased.knobs import knob_layout
from nodebased.scene3d import ParticleInstance, Volume
from tests.test_particles_nodes import make, wire


def liquid(positions, velocities, phi=None, ids=None):
    positions = np.asarray(positions, np.float32)
    velocities = np.asarray(velocities, np.float32)
    if phi is None:
        axis = np.indices((12, 12, 12), dtype=np.float32)
        phi = axis[1] / 10.0 - 0.4
    surface = Volume(phi, voxel_size=0.1, origin=(-0.1, -0.1, -0.1))
    n = len(positions)
    return ParticleInstance(positions, np.full(n, .05, np.float32), np.ones((n, 4), np.float32),
                            velocities=velocities, ids=np.arange(n, dtype=np.int64) if ids is None else ids,
                            surface=surface, frame=1)


class WhitewaterTests(unittest.TestCase):
    def test_node_registration_and_panel_params(self):
        self.assertIn("FluidWhitewater3D", SPECS)
        self.assertEqual(OUTPUT_TYPES["FluidWhitewater3D"], "particles")
        self.assertEqual(INPUT_TYPES["particles"], ("particles",))
        laid_out = [name for group in knob_layout("FluidWhitewater3D") for name in group.params]
        self.assertEqual(sorted(laid_out), sorted(SPECS["FluidWhitewater3D"]["params"]))

    def test_still_tank_emits_nothing(self):
        points = np.array([[.1 + (i % 3) * .02, .2, .1 + (i // 3) * .02] for i in range(9)])
        source = liquid(points, np.zeros_like(points))
        solver = whitewater.FluidWhitewater3D()
        result = solver.step(solver.initial_state(), source, 1)
        self.assertEqual(len(result.ids), 0)

    def test_impact_emits_surface_spray_and_foam_then_foam_dissipates(self):
        angle = np.linspace(-.5, .5, 12)
        points = np.column_stack((.2 + .15 * np.cos(angle), .3 + .15 * np.sin(angle), np.full(12, .1)))
        axes = [np.arange(12, dtype=np.float32) * .1 - .1 for _ in range(3)]
        x, y, z = np.meshgrid(*axes, indexing="ij")
        phi = np.sqrt((x - .2) ** 2 + (y - .3) ** 2 + (z - .1) ** 2) - .15
        normals = points - np.array([.2, .3, .1])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        velocity = normals * 3.0
        # Add converging local motion as well as outward wave speed: the reference model
        # uses both wave-crest and trapped-air potential at a breaking impact.
        velocity[:, 2] += np.where(np.arange(12) % 2, -1.0, 1.0)
        source = liquid(points, velocity, phi)
        solver = whitewater.FluidWhitewater3D({"max_particles": 60,
                                               "spray_threshold": .01, "foam_threshold": .01, "foam_lifespan": .04,
                                               "particle_lifespan": 10., "foam_emission": 1.,
                                               "spray_emission": 1., "wave_crest_min": 0.0,
                                               "wave_crest_max": .2, "kinetic_energy_min": 0.0,
                                               "kinetic_energy_max": 5.})
        first = solver.step(solver.initial_state(), source, 1)
        self.assertIn(whitewater.FOAM, first.kinds)
        self.assertIn(whitewater.SPRAY, first.kinds)
        self.assertEqual(solver.stats["foam"], int(np.count_nonzero(first.kinds == whitewater.FOAM)))
        self.assertEqual(solver.stats["spray"], int(np.count_nonzero(first.kinds == whitewater.SPRAY)))
        still = liquid(points, np.zeros_like(points))
        dead = solver.step(first, still, 2)
        self.assertEqual(int(np.count_nonzero(dead.kinds == whitewater.FOAM)), 0)

    def test_bubbles_rise_and_become_surface_foam(self):
        p = whitewater.whitewater_defaults()
        p.update({"foam_emission": 0., "spray_emission": 0., "bubbles_emission": 1.,
                  "bubbles_threshold": .001, "max_particles": 30, "surface_band": .04,
                  "bubble_buoyancy": .1, "bubble_drag": 0., "particle_lifespan": 10.})
        pts = np.array([[.1 + i * .01, .05, .1] for i in range(8)])
        vel = np.array([[.1 * (-1.) ** i, 0., 0.] for i in range(8)])
        # Negative SDF throughout the seed region; the next liquid has a raised interface.
        deep = liquid(pts, vel, np.full((12, 12, 12), -0.2, np.float32))
        solver = whitewater.FluidWhitewater3D(p, fps=1.)
        bubbles = solver.step(solver.initial_state(), deep, 1)
        self.assertGreater(np.count_nonzero(bubbles.kinds == whitewater.BUBBLE), 0)
        before_y = bubbles.positions[:, 1].copy()
        surface_phi = np.zeros((12, 12, 12), np.float32)
        surfaced = liquid(pts, vel, surface_phi)
        foam = solver.step(bubbles, surfaced, 2)
        self.assertTrue(np.all(foam.positions[:, 1] > before_y))
        self.assertTrue(np.all(foam.kinds == whitewater.FOAM))

    def test_paper_potentials_use_approaching_pairs_convex_crests_and_energy(self):
        positions = np.array([[-.05, 0., 0.], [.05, 0., 0.]])
        normals = np.tile((0., 1., 0.), (2, 1))
        approaching = np.array([[1., 0., 0.], [-1., 0., 0.]])
        trapped, crest, energy = whitewater._emission_potentials(positions, approaching, normals, .2)
        self.assertAlmostEqual(float(trapped[0]), 2.0, places=7)  # |Δv|(1 - v̂·x̂)W, W = 0.5
        self.assertAlmostEqual(float(crest[0]), 0., places=7)
        self.assertAlmostEqual(float(energy[0]), .5, places=7)
        _, _, quarter_mass_energy = whitewater._emission_potentials(
            positions, approaching, normals, .2, particle_mass=.25)
        self.assertAlmostEqual(float(quarter_mass_energy[0]), .125, places=7)
        separating = -approaching
        trapped_out, _, _ = whitewater._emission_potentials(positions, separating, normals, .2)
        self.assertAlmostEqual(float(trapped_out[0]), 0., places=7)
        # A convex crest has neighbours on the inward side, changing normals, and outward velocity.
        crest_positions = np.array([[0., 0., 0.], [0., -.05, 0.], [.02, -.04, 0.]])
        crest_normals = np.array([[0., 1., 0.], [0., .7, .7], [.7, .7, 0.]])
        crest_velocity = np.tile((0., 2., 0.), (3, 1))
        _, wave, _ = whitewater._emission_potentials(crest_positions, crest_velocity,
                                                       crest_normals, .2)
        self.assertGreater(float(wave[0]), 0.)
        self.assertEqual(float(whitewater._map_potential(np.array([0., 1., 2.]), .5, 1.5)[0]), 0.)
        np.testing.assert_allclose(whitewater._map_potential(np.array([0., 1., 2.]), .5, 1.5),
                                   [0., .5, 1.])

    def test_spray_sweeps_moving_liquid_collider_and_receives_its_velocity(self):
        triangle = np.array([[[-1., 0., -1.], [1., 0., -1.], [0., 0., 1.]]])
        track = SimpleNamespace(at=lambda frame: triangle,
                                motion=lambda frame: np.tile((0., .05, 0.), (len(triangle), 1)))
        collider = SimpleNamespace(track=track, animated=True)
        solver = whitewater.FluidWhitewater3D({"spray_emission": 0., "foam_emission": 0.,
                                               "bubbles_emission": 0., "gravity": 0.}, fps=24.,
                                              colliders=(collider,))
        state = whitewater.WhitewaterState(np.array([[0., .1, 0.]]), np.array([[0., -5., 0.]]),
                                           np.array([.02]), np.array([0.]), np.array([1.]),
                                           np.array([1]), np.array([whitewater.SPRAY], np.uint8), 2)
        source = liquid([[.5, .5, .5]], [[0., 0., 0.]], phi=None)
        result = solver.step(state, source, 2)
        self.assertGreater(result.positions[0, 1], 0.)
        self.assertGreater(result.velocities[0, 1], 1.2)  # rebound follows the collider's upward motion

    def test_spray_that_enters_the_pool_returns_as_liquid(self):
        axes = np.arange(12, dtype=np.float32) * 0.1 - 0.1
        x, y, z = np.meshgrid(axes, axes, axes, indexing="ij")
        phi = y - 0.2
        source = liquid([[0.1, 0.25, 0.1]], [[0., -5., 0.]], phi)
        solver = whitewater.FluidWhitewater3D({"foam_emission": 0., "spray_emission": 0.,
                                                "bubbles_emission": 0., "gravity": 0.})
        state = whitewater.WhitewaterState(np.array([[0.1, 0.25, 0.1]], np.float32),
                                           np.array([[0., -5., 0.]], np.float32), np.array([.02], np.float32),
                                           np.array([0.], np.float32), np.array([10.], np.float32),
                                           np.array([1], np.int64), np.array([whitewater.SPRAY], np.uint8), 2)
        returned = solver.step(state, source, 1)
        self.assertEqual(returned.kinds.tolist(), [whitewater.LIQUID])
        self.assertEqual(solver.stats["liquid"], 1)
        self.assertLess(float(returned.positions[0, 1]), 0.12)

    def test_caps_determinism_type_attribute_and_cache_round_trip(self):
        points = np.array([[.1 + i * .002, .3, .1] for i in range(40)])
        source = liquid(points, np.tile((0., 4., 0.), (len(points), 1)))
        params = {"foam_threshold": .01, "spray_threshold": .01, "max_particles": 5,
                  "foam_emission": 1., "spray_emission": 1.}
        solver = whitewater.FluidWhitewater3D(params, seed=23)
        a = solver.step(solver.initial_state(), source, 8)
        b = solver.step(solver.initial_state(), source, 8)
        self.assertLessEqual(len(a.ids), 5)
        for x, y in zip(a.__dict__.values(), b.__dict__.values()):
            np.testing.assert_array_equal(x, y)
        instance = whitewater.instance_from_state(a, source, 8)
        np.testing.assert_array_equal(instance.whitewater_type, a.kinds)
        np.testing.assert_allclose(instance.colors[:, 3], np.clip(1.0 - a.ages / a.lifetimes, 0.0, 1.0))
        self.assertFalse(instance.whitewater_type.flags.writeable)
        self.assertIsNone(instance.stream)  # its own node cache owns this output; avoid re-solving it as raw FLIP
        with tempfile.TemporaryDirectory() as folder:
            cache = simcache.SimCache(root=folder)
            saved = simcache.State({"position": a.positions, "velocity": a.velocities, "size": a.sizes,
                                    "age": a.ages, "life": a.lifetimes, "id": a.ids, "kind": a.kinds},
                                   {"next_id": a.next_id})
            cache.put("whitewater-test", 8, saved)
            loaded = cache.get("whitewater-test", 8)
            for name, expected in (("position", a.positions), ("kind", a.kinds), ("id", a.ids)):
                np.testing.assert_array_equal(loaded.arrays[name], expected)

    def test_graph_node_solves_from_liquid_frames_and_caches_output(self):
        with tempfile.TemporaryDirectory() as folder:
            d = Dispatcher()
            make(d, src=("FluidSource3D", {"fluid_type": "liquid", "src_center_y": .8,
                                           "src_radius": .22, "src_vel_y": 0., "end_frame": 1}),
                 liquid=("FluidLiquidSolver3D", {"division_size": .2, "bounds_min_x": -.6,
                    "bounds_min_y": 0., "bounds_min_z": -.6, "bounds_max_x": .6,
                    "bounds_max_y": 1.2, "bounds_max_z": .6, "substeps": 1, "particles_per_cell": 1}),
                 white=("FluidWhitewater3D", {"max_particles": 12}))
            wire(d, "liquid", "fluid", "src")
            wire(d, "white", "particles", "liquid")
            first_eval = Evaluator(sim=simcache.SimCache(root=folder))
            first = first_eval.evaluate_raster(d.document, "white", frame=1, typed=True)
            second_eval = Evaluator(sim=simcache.SimCache(root=folder))
            second = second_eval.evaluate_raster(d.document, "white", frame=1, typed=True)
            self.assertIsInstance(first, ParticleInstance)
            np.testing.assert_array_equal(first.whitewater_type, second.whitewater_type)
            np.testing.assert_array_equal(first.positions, second.positions)
            self.assertGreater(second_eval._sim_stores[(256, 2048)].disk_hits, 0)


if __name__ == "__main__":
    unittest.main()
