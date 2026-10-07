"""Lane 6 step N3: what broke when the fluid features met in the Hot pour scene, one test per fix."""
import types
import unittest

import numpy as np

from nodebased import fluid3d
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


if __name__ == "__main__":
    unittest.main()
