"""GPU FLIP transfer kernels compared with Liquid3D's CPU reference."""
import unittest

import numpy as np

from nodebased import flip3d
from nodebased import fluid_gpu_solver as fgs
from nodebased.flip_gpu_transfer import GpuFlipTransfers
from tests.test_flip3d import BoxSource


@unittest.skipUnless(fgs.available(), "no wgpu compute adapter")
class GpuFlipTransferTests(unittest.TestCase):
    def test_gpu_extrapolation_matches_cpu_reference(self):
        rng = np.random.default_rng(62)
        field = rng.normal(size=(9, 7, 5))
        valid = rng.random(field.shape) > 0.75
        valid[4, 3, 2] = True
        expected, expected_valid = flip3d.extrapolate(field, valid, 4)
        from nodebased.flip_gpu_extrapolate import extrapolate
        actual, actual_valid = extrapolate(field, valid, 4)
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
        np.testing.assert_array_equal(actual_valid, expected_valid)

    def test_particle_grid_and_grid_particle_transfers_match_cpu(self):
        rng = np.random.default_rng(24)
        shape = (16, 12, 8)
        pos = rng.uniform((1.0, 1.0, 1.0), (15.0, 11.0, 7.0), (240, 3))
        vel = rng.normal(0.0, 0.3, pos.shape)
        solver = flip3d.Liquid3D({"nx": shape[0], "ny": shape[1], "nz": shape[2], "flip_ratio": 0.9})
        stencils = solver._stencils(pos)
        cpu_u, cpu_v, cpu_w, cpu_valid, cpu_old = solver._to_grid(vel, stencils)
        gpu = GpuFlipTransfers(shape)
        (gpu_u, gpu_v, gpu_w), valid, mask, old = gpu.to_grid(pos, vel)
        for a, b, c, name in zip((cpu_u, cpu_v, cpu_w), (gpu_u, gpu_v, gpu_w),
                                 (old[0], old[1], old[2]), ("u", "v", "w")):
            np.testing.assert_allclose(b, a, atol=5e-5, rtol=2e-5, err_msg=name)
            np.testing.assert_array_equal(valid[name], cpu_valid[name])
            np.testing.assert_allclose(c, cpu_old[name], atol=5e-5, rtol=2e-5)
        np.testing.assert_array_equal(mask, solver._classify(pos, None))
        new_fields = {"u": gpu_u, "v": gpu_v, "w": gpu_w}
        old_fields = dict(zip("uvw", old))
        cpu_vel = solver._from_grid(vel, new_fields, old_fields, stencils)
        cpu_pos = solver._advect(pos.copy(), cpu_vel.copy(), new_fields, None, 0.5)
        gpu_pos, gpu_vel = gpu.from_grid(pos.copy(), vel.copy(), new_fields, old_fields, 0.9, 0.5)
        np.testing.assert_allclose(gpu_vel, cpu_vel, atol=5e-5, rtol=2e-5)
        np.testing.assert_allclose(gpu_pos, cpu_pos, atol=2e-4, rtol=2e-5)

    def test_gpu_transfer_dam_break_tracks_cpu_particles(self):
        params = {"nx": 16, "ny": 16, "nz": 8, "gravity": 0.08, "substeps": 1,
                  "flip_ratio": 0.8, "max_iterations": 300}
        source = BoxSource((0, 0, 0), (5, 8, 8))
        cpu = flip3d.Liquid3D(params, sources=[source])
        gpu = flip3d.Liquid3D({**params, "backend": "gpu"}, sources=[source])
        a, b = cpu.initial_state(0), gpu.initial_state(0)
        for frame in range(1, 4):
            a = cpu.step(a, frame, 0, 0)
            b = gpu.step(b, frame, 0, 0)
        self.assertEqual(len(a.arrays["position"]), len(b.arrays["position"]))
        np.testing.assert_allclose(b.arrays["position"], a.arrays["position"], atol=2e-3, rtol=2e-3)
        np.testing.assert_allclose(b.arrays["velocity"], a.arrays["velocity"], atol=2e-3, rtol=2e-3)

    def test_gpu_open_bottom_records_drained_liquid_mass(self):
        from nodebased.simcache import State
        solver = flip3d.Liquid3D({"nx": 8, "ny": 8, "nz": 8, "gravity": 0.0,
                                  "backend": "gpu", "boundary_y_min": "open"})
        arrays = flip3d.empty_arrays()
        arrays.update(position=np.tile((3.5, 0.2, 3.5), (8, 1)).astype(np.float32),
                      velocity=np.tile((0., -20., 0.), (8, 1)).astype(np.float32),
                      id=np.arange(8, dtype=np.int64), age=np.zeros(8, np.int32),
                      temperature=np.ones(8, np.float32))
        result = solver.step(State(arrays, {"next_id": 8, "substep_count": 0}), 1, 0, 0)
        self.assertEqual(len(result.arrays["position"]), 0)
        self.assertEqual(result.meta["escaped_mass"], 1.0)
