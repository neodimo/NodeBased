"""Lane 6 step N3: what broke when the fluid features met in the Hot pour scene, one test per fix."""
import types
import unittest

import numpy as np

from nodebased import flip3d, fluid3d
from tests.test_fluid3d import box_collider, box_triangles


def stub(shape, origin=(0.0, 0.0, 0.0), voxel=1.0):
    return types.SimpleNamespace(shape=tuple(shape), origin=np.asarray(origin, float), voxel=float(voxel))


class ColliderCrossingTheDomainTests(unittest.TestCase):
    def test_a_table_that_runs_off_the_bottom_of_the_domain_is_solid_down_to_the_floor(self):
        # a slab whose top is at y = 6.5 and whose underside is below the domain: before the fix only its top
        # face's cells (one layer) were solid, because the flood fill escaped through the cut underside
        collider = box_collider((2.2, -5.0, 2.2), (12.8, 6.5, 12.8))
        solid, _velocity = collider.mask(stub((16, 16, 16)), 1)
        self.assertTrue(solid[3:12, 0:6, 3:12].all(), "the slab must be solid from the domain floor up to its top")
        self.assertFalse(solid[3:12, 7:, 3:12].any())

    def test_a_box_inside_the_domain_fills_exactly_as_before(self):
        collider = box_collider((4.2, 4.2, 4.2), (9.8, 9.8, 9.8))
        solid, _velocity = collider.mask(stub((16, 16, 16)), 1)
        self.assertEqual(int(solid.sum()), 6 ** 3)

    def test_a_wall_that_runs_off_the_side_is_solid_up_to_the_edge(self):
        collider = box_collider((-9.0, 2.2, 2.2), (4.8, 9.8, 9.8))
        solid, _velocity = collider.mask(stub((16, 16, 16)), 1)
        self.assertTrue(solid[0:4, 3:9, 3:9].all())


class WideColliderDoesNotInflateTheDomainTests(unittest.TestCase):
    """A table far wider than the liquid used to pull the adaptive box out to the table's whole width."""

    def test_liquid_domain_stays_near_the_liquid_when_a_wide_table_is_in_the_scene(self):
        table = box_collider((-120.0, -6.0, -120.0), (120.0, 0.5, 120.0))
        source = fluid3d.Source(center=(0.0, 14.0, 0.0), radius=3.0, fluid_type="liquid", end_frame=1)
        solver = flip3d.Liquid3D({"nx": 32, "ny": 32, "nz": 32, "origin_x": -16.0, "origin_y": -2.0, "origin_z": -16.0,
                                  "voxel_size": 1.0, "auto_resize": 1, "padding": 8, "max_size": 256,
                                  "particles_per_cell": 2, "substeps": 1, "gravity": 0.02, "start_frame": 1,
                                  "max_iterations": 40}, sources=[source], colliders=[table])
        state = solver.initial_state()
        for frame in range(1, 4):
            state = solver.step(state, frame=frame)
        shape = state.meta["domain_shape"]
        self.assertLessEqual(shape[0], 64, f"the box grew to {shape} to cover the table")
        self.assertLessEqual(shape[2], 64)
        solid = solver._solid_for(3)[0]
        self.assertTrue(solid[:, :2, :].any(), "the table is still in the box below the liquid")

    def test_smoke_domain_stays_near_the_smoke_when_a_wide_table_is_in_the_scene(self):
        table = box_collider((-120.0, -6.0, -120.0), (120.0, 0.5, 120.0))
        solver = fluid3d.Smoke3D({"nx": 16, "ny": 16, "nz": 16, "default_source": 0, "auto_resize": 1, "padding": 8,
                                  "max_size": 256, "boundary_x": "open", "boundary_y": "open", "boundary_z": "open"},
                                 colliders=[table])
        state = solver.initial_state()
        state.arrays["density"][6:10, 3:7, 6:10] = 1.0
        resized = solver._resize_active_domain(state, frame=1)
        shape = resized.meta["domain_shape"]
        self.assertLessEqual(max(shape), 48, f"the box grew to {shape} to cover the table")


if __name__ == "__main__":
    unittest.main()
