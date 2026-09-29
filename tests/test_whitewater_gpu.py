"""GPU whitewater emission/motion against the CPU reference and cache replay."""
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
from nodebased import simcache, whitewater
from nodebased.fluid_gpu_whitewater import GpuWhitewater3D
from tests.test_whitewater import liquid


class GpuWhitewaterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.gpu = GpuWhitewater3D()
        except Exception as exc:
            raise unittest.SkipTest(str(exc))

    def test_three_potentials_match_reference(self):
        rng = np.random.default_rng(19)
        pos = rng.uniform(.1, .4, (120, 3))
        vel = rng.normal(size=(120, 3))
        normal = rng.normal(size=(120, 3))
        cpu = whitewater._emission_potentials(pos, vel, normal, .12, .004)
        gpu = self.gpu.potentials(pos, vel, normal, .12, .004)
        for a, b in zip(cpu, gpu):
            np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-5)

    def test_counts_replay_and_disk_reopen(self):
        rng = np.random.default_rng(21)
        points = rng.uniform((.1, .24, .1), (.5, .38, .5), (160, 3))
        velocity = rng.normal(0, 2, (160, 3))
        source = liquid(points, velocity)
        params = {'max_particles': 90, 'foam_threshold': .01, 'spray_threshold': .01,
                  'bubbles_threshold': .001, 'foam_emission': 1., 'spray_emission': 1.,
                  'bubbles_emission': 1., 'kinetic_energy_min': 0., 'kinetic_energy_max': 1.}
        results = {}
        for backend in ('cpu', 'gpu'):
            solver = whitewater.FluidWhitewater3D({**params, 'whitewater_backend': backend}, seed=42)
            state = solver.initial_state()
            for frame in range(1, 5):
                state = solver.step(state, source, frame)
            results[backend] = state
        for kind in (whitewater.FOAM, whitewater.SPRAY, whitewater.BUBBLE):
            a = np.count_nonzero(results['cpu'].kinds == kind)
            b = np.count_nonzero(results['gpu'].kinds == kind)
            self.assertLessEqual(abs(a - b), .03 * max(1, a), (kind, a, b))
        gpu = whitewater.FluidWhitewater3D({**params, 'whitewater_backend': 'gpu'}, seed=42)
        replay = gpu.initial_state()
        for frame in range(1, 5):
            replay = gpu.step(replay, source, frame)
        for key in ('positions', 'velocities', 'kinds', 'ids'):
            np.testing.assert_array_equal(getattr(results['gpu'], key), getattr(replay, key))
        state = results['gpu']
        with tempfile.TemporaryDirectory() as folder:
            cache = simcache.SimCache(root=folder)
            cache.put('whitewater-gpu', 4, simcache.State({
                'position': state.positions, 'velocity': state.velocities, 'kind': state.kinds,
                'id': state.ids, 'age': state.ages, 'life': state.lifetimes, 'size': state.sizes},
                {'next_id': state.next_id}))
            reopened = simcache.SimCache(root=folder).get('whitewater-gpu', 4)
            np.testing.assert_array_equal(reopened.arrays['position'], state.positions)
            np.testing.assert_array_equal(reopened.arrays['kind'], state.kinds)
            self.assertEqual(reopened.meta['next_id'], state.next_id)

    def test_gpu_motion_preserves_type_rules_and_collider_response(self):
        triangle = np.array([[[-1., 0., -1.], [1., 0., -1.], [0., 0., 1.]]])
        track = SimpleNamespace(at=lambda frame: triangle,
                                motion=lambda frame: np.tile((0., .05, 0.), (len(triangle), 1)))
        collider = SimpleNamespace(track=track, animated=True)
        solver = whitewater.FluidWhitewater3D({
            'whitewater_backend': 'gpu', 'foam_emission': 0., 'spray_emission': 0.,
            'bubbles_emission': 0., 'gravity': 0., 'surface_band': .2,
        }, fps=24., colliders=(collider,))
        self.assertEqual(solver.backend, 'gpu')
        state = whitewater.WhitewaterState(
            np.array([[0., .1, 0.]], np.float32), np.array([[0., -5., 0.]], np.float32),
            np.array([.02]), np.array([0.]), np.array([1.]), np.array([1]),
            np.array([whitewater.SPRAY], np.uint8), 2)
        source = liquid([[.5, .5, .5]], [[0., 0., 0.]], phi=None)
        result = solver.step(state, source, 2)
        self.assertGreater(result.positions[0, 1], 0.)
        self.assertGreater(result.velocities[0, 1], 1.2)
