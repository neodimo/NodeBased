"""GPU viscosity tracks the CPU liquid over the existing blob and still-tank scenes."""
import unittest
import numpy as np
from nodebased import flip3d, fluid3d
from nodebased.fluid_gpu_viscosity import GpuViscosity3D
from nodebased.fluid_gpu3d import GpuPressure3D
from tests.test_flip3d import BoxSource


def measures(state):
    pos = state.arrays['position']
    speed = np.linalg.norm(state.arrays['velocity'], axis=1)
    return np.array([np.ptp(pos[:, 1]), len(pos) / 2.0, speed.mean()])


class GpuViscosityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.gpu = GpuViscosity3D()
        except Exception as exc:
            raise unittest.SkipTest(str(exc))

    def test_weighted_face_solve_matches_cpu(self):
        rng = np.random.default_rng(11)
        solver = flip3d.Liquid3D({'nx': 8, 'ny': 8, 'nz': 8})
        for shape in ((9, 8, 8), (8, 9, 8), (8, 8, 9)):
            rhs = rng.normal(size=shape)
            coeff = rng.uniform(1.0, 10.0, size=shape)
            fields = {name: rhs.copy() for name in 'uvw'}
            solver._viscosity(fields, 10.0, {name: coeff for name in 'uvw'})
            np.testing.assert_allclose(self.gpu.solve(rhs, 10.0, coeff), fields['u'], rtol=1e-5, atol=1e-7)
            np.testing.assert_array_equal(self.gpu.solve(rhs, 0.0, coeff), rhs)

    def test_eight_frame_blob_and_still_tank(self):
        scenarios = (
            ('blob', fluid3d.Source('sphere', center=(8, 8, 8), radius=4,
                                    fluid_type='liquid', start_frame=1, end_frame=1)),
            ('tank', BoxSource((2, 2, 2), (14, 12, 14))),
        )
        for name, source in scenarios:
            states = {}
            for backend in ('cpu', 'gpu'):
                params = {'nx': 16, 'ny': 16, 'nz': 16, 'gravity': 0.0,
                          'particles_per_cell': 2, 'viscosity': 10.0,
                          'viscosity_by_attribute': 'temperature', 'backend': backend}
                pressure = GpuPressure3D().solve if backend == 'gpu' else None
                solver = flip3d.Liquid3D(params, pressure_solver=pressure, sources=[source])
                state = solver.initial_state()
                for frame in range(1, 9):
                    state = solver.step(state, frame, 0, 4)
                    if frame == 1 and name == 'blob':
                        pos = state.arrays['position']
                        state.arrays['velocity'][:, 1] = ((pos[:, 0] - 8.0) * .08).astype(np.float32)
                states[backend] = measures(state)
            cpu, gpu = states['cpu'], states['gpu']
            for index, metric in enumerate(('height', 'volume', 'mean speed')):
                with self.subTest(scene=name, metric=metric):
                    self.assertLessEqual(abs(cpu[index] - gpu[index]), .02 * max(abs(cpu[index]), 1e-6))
