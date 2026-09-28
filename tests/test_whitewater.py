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
        phi = 0.4 - axis[1] / 10.0
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
        points = np.array([[.1 + i * .018, .3, .1] for i in range(12)])
        velocity = np.tile((0., 3., 0.), (len(points), 1))
        source = liquid(points, velocity)
        solver = whitewater.FluidWhitewater3D({"max_particles": 60, "foam_threshold": .1,
                                               "spray_threshold": .1, "foam_lifespan": .04,
                                               "particle_lifespan": 10., "foam_emission": 1.,
                                               "spray_emission": 1.})
        first = solver.step(solver.initial_state(), source, 1)
        self.assertIn(whitewater.FOAM, first.kinds)
        self.assertIn(whitewater.SPRAY, first.kinds)
        still = liquid(points, np.zeros_like(points))
        dead = solver.step(first, still, 2)
        self.assertEqual(int(np.count_nonzero(dead.kinds == whitewater.FOAM)), 0)

    def test_bubbles_rise_and_become_surface_foam(self):
        p = whitewater.whitewater_defaults()
        p.update({"foam_emission": 0., "spray_emission": 0., "bubbles_emission": 1.,
                  "bubbles_threshold": .001, "max_particles": 30, "surface_band": .04,
                  "bubble_buoyancy": 10., "bubble_drag": 0., "particle_lifespan": 10.})
        pts = np.array([[.1 + i * .01, .05, .1] for i in range(8)])
        vel = np.array([[(-1.) ** i, 0., 0.] for i in range(8)])
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
