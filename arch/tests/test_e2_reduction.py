from __future__ import annotations

import inspect
import math
import unittest

import numpy as np

from carta import Fidelity, RayBatch, Space, reduce_rays
from carta import ops
from carta.adapters import GaussianVolumeChart, TriangleSurfaceChart
from carta.representations import GaussianVolume, TriangleSurface


WORLD = Space("test.world", 3, ("m", "m", "m"))


def rays(origins: np.ndarray, directions: np.ndarray | None = None) -> RayBatch:
    origins = np.asarray(origins, dtype=float)
    if directions is None:
        directions = np.broadcast_to([0.0, 0.0, 1.0], origins.shape).copy()
    return RayBatch(
        WORLD,
        origins,
        np.asarray(directions, dtype=float),
        np.zeros(len(origins)),
        np.full(len(origins), 10.0),
    )


def covering_triangle(depth: float) -> np.ndarray:
    return np.array([[-2.0, -2.0, depth], [2.0, -2.0, depth], [0.0, 2.0, depth]])


class VisibilityReductionTest(unittest.TestCase):
    def test_r1_one_fold_serves_surface_and_volume_without_type_switches(self) -> None:
        surface = TriangleSurfaceChart(
            TriangleSurface(
                WORLD,
                np.array([covering_triangle(1.0)]),
                np.array([[1.0, 0.0, 0.0, 1.0]]),
            )
        )
        volume = GaussianVolumeChart(
            GaussianVolume(
                WORLD,
                (-1.0, -1.0, 0.5),
                (1.0, 1.0, 1.5),
                (0.0, 0.0, 1.0),
                (0.5, 0.5, 0.2),
                2.0,
                (0.2, 0.4, 0.8),
            ),
            integration_steps=16,
        )
        query = rays(np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]))
        results = [reduce_rays(source, query) for source in (surface, volume)]
        for result in results:
            self.assertEqual(result.values.shape, (2, 4))
            self.assertEqual(result.transmittance.shape, (2,))
        source = inspect.getsource(ops).lower()
        self.assertNotIn("trianglesurface", source)
        self.assertNotIn("gaussianvolume", source)
        self.assertNotIn(".adapters", source)
        self.assertNotIn(".representations", source)

    def test_r2_opaque_near_surface_occludes_far_surface(self) -> None:
        # Payload order is deliberately far then near.
        surface = TriangleSurface(
            WORLD,
            np.array([covering_triangle(2.0), covering_triangle(1.0)]),
            np.array([[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0]]),
        )
        result = reduce_rays(
            TriangleSurfaceChart(surface), rays(np.array([[0.0, 0.0, 0.0]]))
        )
        np.testing.assert_allclose(result.values[0], [1.0, 0.0, 0.0, 1.0])
        self.assertEqual(result.transmittance[0], 0.0)
        self.assertEqual(result.fidelity, Fidelity.EXACT)

    def test_r3_transparent_surface_reduction_is_depth_ordered(self) -> None:
        surface = TriangleSurface(
            WORLD,
            np.array([covering_triangle(2.0), covering_triangle(1.0)]),
            np.array([[0.0, 0.0, 1.0, 0.5], [1.0, 0.0, 0.0, 0.5]]),
        )
        result = reduce_rays(
            TriangleSurfaceChart(surface), rays(np.array([[0.0, 0.0, 0.0]]))
        )
        np.testing.assert_allclose(result.values[0], [0.5, 0.0, 0.25, 0.75])
        self.assertAlmostEqual(result.transmittance[0], 0.25)

    def test_r4_volume_integration_converges_to_analytic_optical_depth(self) -> None:
        sigma_z = 0.15
        peak_density = 3.0
        volume = GaussianVolume(
            WORLD,
            (-0.5, -0.5, 0.0),
            (0.5, 0.5, 2.0),
            (0.0, 0.0, 1.0),
            (0.2, 0.2, sigma_z),
            peak_density,
            (0.25, 0.5, 1.0),
        )
        query = rays(np.array([[0.0, 0.0, -1.0]]))
        low = reduce_rays(GaussianVolumeChart(volume, 8), query)
        high = reduce_rays(GaussianVolumeChart(volume, 128), query)
        optical_depth = (
            peak_density
            * sigma_z
            * math.sqrt(2.0 * math.pi)
            * math.erf(1.0 / (math.sqrt(2.0) * sigma_z))
        )
        expected_alpha = 1.0 - math.exp(-optical_depth)
        low_error = abs(low.values[0, 3] - expected_alpha)
        high_error = abs(high.values[0, 3] - expected_alpha)
        self.assertLess(high_error, 1e-6)
        self.assertLess(high_error, low_error)
        self.assertEqual(high.fidelity, Fidelity.APPROXIMATE)

    def test_r5_adapters_generate_contributions_but_are_not_sampleable(self) -> None:
        surface = TriangleSurfaceChart(
            TriangleSurface(
                WORLD,
                np.array([covering_triangle(1.0)]),
                np.array([[1.0, 1.0, 1.0, 1.0]]),
            )
        )
        volume = GaussianVolumeChart(
            GaussianVolume(
                WORLD,
                (-1.0, -1.0, 0.0),
                (1.0, 1.0, 2.0),
                (0.0, 0.0, 1.0),
                (1.0, 1.0, 1.0),
                1.0,
                (1.0, 1.0, 1.0),
            )
        )
        self.assertTrue(callable(surface.ray_contributions))
        self.assertTrue(callable(volume.ray_contributions))
        self.assertFalse(hasattr(surface, "sample"))
        self.assertFalse(hasattr(volume, "sample"))

    def test_r6_valid_miss_is_known_transparent(self) -> None:
        surface = TriangleSurface(
            WORLD,
            np.array([covering_triangle(1.0)]),
            np.array([[1.0, 0.0, 0.0, 1.0]]),
        )
        result = reduce_rays(
            TriangleSurfaceChart(surface), rays(np.array([[10.0, 10.0, 0.0]]))
        )
        np.testing.assert_array_equal(result.values[0], np.zeros(4))
        self.assertEqual(result.validity[0], 1.0)
        self.assertEqual(result.transmittance[0], 1.0)

    def test_unsorted_operation_ir_is_rejected(self) -> None:
        from carta import OrderedContributions

        with self.assertRaisesRegex(ValueError, "ordered"):
            OrderedContributions(
                np.array([[2.0, 1.0]]),
                np.zeros((1, 2, 4)),
                np.ones((1, 2), dtype=bool),
                np.ones(1),
                Fidelity.EXACT,
            )


if __name__ == "__main__":
    unittest.main()
