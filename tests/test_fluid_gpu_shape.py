"""Eight-frame Shape parity on the resident smoke solver."""
import unittest

import numpy as np

from nodebased import fluid_gpu_solver
from tests.test_fluid3d import solve
from tests.test_fluid_gpu_solver import solve_gpu


@unittest.skipUnless(fluid_gpu_solver.available(), "no compute adapter")
class ResidentShapeTests(unittest.TestCase):
    BASE = {"nx": 20, "ny": 28, "nz": 20, "boundary_y": "open"}

    def test_each_shape_control_matches_cpu_after_eight_frames(self):
        controls = (
            {"disturbance": 0.8, "disturbance_size": 3.0},
            {"disturbance": 0.8, "disturbance_field": "density",
             "disturbance_range_lo": 0.01, "disturbance_range_hi": 2.0, "disturbance_ramp": 0.1},
            {"shredding": 0.4},
            {"turbulence": 0.3, "swirl_size": 2.0, "grain": 2, "pulse_length": 3.0},
            {"turbulence": 0.3, "swirl_size": 2.0, "turbulence_field": "temperature",
             "turbulence_range_lo": 0.01, "turbulence_range_hi": 2.0},
            {"dissipation": 0.2, "dissipation_field": "speed",
             "dissipation_range_lo": 0.0, "dissipation_range_hi": 4.0},
            {"dissipation": 0.2, "dissipation_field": "vorticity",
             "dissipation_range_lo": 0.0, "dissipation_range_hi": 4.0},
            {"vorticity": 0.5},
        )
        for change in controls:
            with self.subTest(change=change):
                params = {**self.BASE, **change}
                cpu, _ = solve(params, 8)
                gpu, _ = solve_gpu(params, 8)
                for name in ("density",):
                    expected = float(cpu.arrays[name].mean())
                    actual = float(gpu.arrays[name].mean())
                    self.assertLess(abs(actual - expected) / max(abs(expected), 1e-7), .05)
                def speed(state):
                    a = state.arrays
                    return float(np.mean(np.sqrt((.5 * (a["u"][:-1] + a["u"][1:])) ** 2 +
                                                (.5 * (a["v"][:, :-1] + a["v"][:, 1:])) ** 2 +
                                                (.5 * (a["w"][:, :, :-1] + a["w"][:, :, 1:])) ** 2)))
                self.assertLess(abs(speed(gpu) - speed(cpu)) / max(speed(cpu), 1e-7), .05)

    def test_zero_shape_strengths_leave_existing_gpu_result_exact(self):
        base, _ = solve_gpu(self.BASE, 8)
        for name in ("disturbance", "shredding", "turbulence"):
            with self.subTest(name=name):
                zero, _ = solve_gpu({**self.BASE, name: 0.0}, 8)
                for field in ("density", "u", "v", "w"):
                    np.testing.assert_array_equal(base.arrays[field], zero.arrays[field])
