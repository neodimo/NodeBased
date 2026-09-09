from __future__ import annotations

import unittest

import numpy as np

from carta import FootprintBatch, ProjectiveMap, Space


class CoreContractTest(unittest.TestCase):
    def test_affine_jacobian_transports_basis(self) -> None:
        source = Space("source", 2, ("m", "m"))
        target = Space("target", 2, ("m", "m"))
        transform = ProjectiveMap(
            source, target, np.array([[2.0, 0.5, 1.0], [0.0, 3.0, -2.0], [0, 0, 1]])
        )
        request = FootprintBatch(source, np.array([[0.2, 0.4]]), np.array([np.eye(2)]))
        mapped = transform.pullback(request)
        np.testing.assert_allclose(mapped.basis[0], [[2.0, 0.5], [0.0, 3.0]])

    def test_singular_projective_map_is_rejected(self) -> None:
        space = Space("space", 2, ("u", "u"))
        with self.assertRaisesRegex(ValueError, "invertible"):
            ProjectiveMap(space, space, np.zeros((3, 3)))

    def test_scalar_footprint_shape_is_not_accepted(self) -> None:
        space = Space("space", 2, ("u", "u"))
        with self.assertRaisesRegex(ValueError, "centers"):
            FootprintBatch(space, np.array([0.5, 0.5]), np.eye(2))


if __name__ == "__main__":
    unittest.main()
