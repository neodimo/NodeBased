from __future__ import annotations

import inspect
import unittest

import numpy as np

from carta import Fidelity, ProjectiveMap, Space, output_footprints, over, render, warp_sample
from carta import ops
from carta.adapters import GaussianSplatChart, RasterChart, SDFChart
from carta.representations import CircleSDF, GaussianSplats, Raster


UV = Space("test.uv", 2, ("normalized", "normalized"))
OUTPUT = Space("test.output", 2, ("normalized", "normalized"))


def identity_pullback() -> ProjectiveMap:
    return ProjectiveMap(OUTPUT, UV, np.eye(3))


def downsample_2x(values: np.ndarray, high_size: int) -> np.ndarray:
    channels = values.shape[1]
    return values.reshape(high_size // 2, 2, high_size // 2, 2, channels).mean(axis=(1, 3))


class ChartTest(unittest.TestCase):
    def test_s1_one_operation_path_serves_three_unlike_representations(self) -> None:
        pixels = np.ones((8, 8, 4)) * np.array([0.1, 0.2, 0.3, 1.0])
        fields = [
            RasterChart(Raster(UV, pixels)).reconstruct(),
            SDFChart(CircleSDF(UV, (0.5, 0.5), 0.3, (1.0, 0.0, 0.0, 0.8))).reconstruct(),
            GaussianSplatChart(
                GaussianSplats(UV, np.array([[0.5, 0.5]]), np.array([0.1]),
                               np.array([[0.0, 1.0, 0.0, 0.7]]))
            ).reconstruct(),
        ]
        projective = ProjectiveMap(
            OUTPUT,
            UV,
            np.array([[0.90, 0.04, 0.02], [-0.03, 0.92, 0.04], [0.08, 0.03, 1.0]]),
        )
        batches = [render(field, projective, OUTPUT, 11, 7) for field in fields]
        for batch in batches:
            self.assertEqual(batch.values.shape, (77, 4))
            self.assertEqual(batch.validity.shape, (77,))
        combined = over(batches[2], over(batches[1], batches[0]))
        self.assertEqual(combined.values.shape, (77, 4))
        source = inspect.getsource(ops).lower()
        self.assertNotIn("raster", source)
        self.assertNotIn("splat", source)
        self.assertNotIn("sdf", source)

    def test_s2_footprint_filter_reduces_checkerboard_aliasing(self) -> None:
        yy, xx = np.indices((63, 63))
        checker = ((xx + yy) % 2).astype(float)
        pixels = np.repeat(checker[:, :, None], 4, axis=2)
        pixels[:, :, 3] = 1.0
        chart = RasterChart(Raster(UV, pixels))
        point = render(chart.reconstruct("point"), identity_pullback(), OUTPUT, 7, 7)
        filtered = render(chart.reconstruct("box16"), identity_pullback(), OUTPUT, 7, 7)
        point_error = np.mean(np.abs(point.values[:, 0] - 0.5))
        filtered_error = np.mean(np.abs(filtered.values[:, 0] - 0.5))
        self.assertGreater(point_error, 0.35)
        self.assertLess(filtered_error, point_error * 0.3)
        self.assertEqual(filtered.fidelity, Fidelity.FILTERED)

    def test_s3_analytic_field_is_consistent_across_resolution(self) -> None:
        field = SDFChart(
            CircleSDF(UV, (0.48, 0.52), 0.27, (0.2, 0.6, 1.0, 0.9))
        ).reconstruct()
        low = render(field, identity_pullback(), OUTPUT, 32, 32).values
        high = render(field, identity_pullback(), OUTPUT, 64, 64).values
        reduced = downsample_2x(high, 64).reshape(-1, 4)
        self.assertLess(np.mean(np.abs(low - reduced)), 0.01)

    def test_s4_projective_maps_compose_before_one_sample_call(self) -> None:
        middle = Space("test.middle", 2, ("normalized", "normalized"))
        first = ProjectiveMap(
            OUTPUT, middle,
            np.array([[0.9, 0.1, 0.02], [-0.05, 1.1, 0.03], [0.08, 0.03, 1.0]]),
        )
        second = ProjectiveMap(
            middle, UV,
            np.array([[1.0, -0.07, 0.04], [0.02, 0.95, 0.01], [-0.04, 0.02, 1.0]]),
        )
        composed = first.then(second)
        footprints = output_footprints(OUTPUT, 9, 5)
        expected = second.apply(first.apply(footprints.centers))
        np.testing.assert_allclose(composed.apply(footprints.centers), expected, atol=1e-12)

        class CountingField:
            space = UV

            def __init__(self) -> None:
                self.calls = 0

            def sample(self, request):
                self.calls += 1
                values = np.column_stack((request.centers, np.zeros(request.count),
                                          np.ones(request.count)))
                from carta import SampleBatch
                return SampleBatch(values, np.ones(request.count), Fidelity.EXACT)

        field = CountingField()
        result = warp_sample(field, composed, footprints)
        self.assertEqual(field.calls, 1)
        np.testing.assert_allclose(result.values[:, :2], expected, atol=1e-12)

    def test_s5_splat_resolution_failure_is_reported(self) -> None:
        splats = GaussianSplats(
            UV,
            np.array([[0.5, 0.5], [0.31, 0.69]]),
            np.array([0.006, 0.008]),
            np.array([[1.0, 0.2, 0.1, 1.0], [0.1, 0.7, 1.0, 0.8]]),
        )
        field = GaussianSplatChart(splats).reconstruct()
        low = render(field, identity_pullback(), OUTPUT, 32, 32)
        high = render(field, identity_pullback(), OUTPUT, 64, 64)
        reduced = downsample_2x(high.values, 64).reshape(-1, 4)
        difference = np.abs(low.values - reduced)
        active = np.maximum(low.values[:, 3], reduced[:, 3]) > 1e-4
        # A whole-frame mean is dominated by empty support. Measure the region
        # the representation actually covers as well as its worst discrepancy.
        self.assertGreater(np.mean(difference[active]), 0.02)
        self.assertGreater(np.max(difference), 0.05)
        self.assertEqual(low.fidelity, Fidelity.HEURISTIC)
        self.assertTrue(any("footprint-inconsistent" in note for note in low.notes))

    def test_space_mismatch_is_rejected(self) -> None:
        other = Space("other", 2, ("normalized", "normalized"))
        field = SDFChart(CircleSDF(UV, (0.5, 0.5), 0.2, (1, 1, 1, 1))).reconstruct()
        with self.assertRaisesRegex(ValueError, "expects"):
            warp_sample(field, identity_pullback(), output_footprints(other, 2, 2))


if __name__ == "__main__":
    unittest.main()
