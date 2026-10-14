"""Rendering 13, lens on the graphics card (step G1): the wgpu twin of Undistort and Redistort.

The CPU warp (`Evaluator._resample` on `lensmodel.sample_coordinates`) is the reference. The agreement test runs
once per adapter this machine can name: the default one (the RTX 3080 Ti here; whatever `force-adapter.py` forces),
the integrated GPU and the software one (llvmpipe). An adapter that is missing, or that lacks what the shader
needs (a Windows software adapter with too few storage buffers, say), makes that adapter's tests skip with the
reason in the message; a skip never reads as a pass.

Tolerances, stated once:
* INTERIOR 1e-4 per channel wherever every tap of the filter lies at least EDGE_BAND pixels inside the source.
* EDGE: within EDGE_BAND pixels of the source's edge, or where the inverse solve is near singular, float32 and
  float64 may round a sample position to different sides of the black border. Bilinear and cubic are continuous
  across it, so they stay within EDGE_TOLERANCE; nearest may pick the neighbouring pixel, so at most
  NEAREST_MISMATCH of its pixels may differ at all.
"""
import unittest
from unittest import mock

import numpy as np

import tests.isolation  # noqa: F401  (Qt settings of the test run stay out of the user's real ones)
from nodebased import gpu3d, gpulens, lensmodel
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator, Region

INTERIOR = 1e-4
EDGE_TOLERANCE = 5e-4
NEAREST_MISMATCH = 0.001
ROUND_TRIP = 1e-3
LENSES = {
    "barrel": lensmodel.Lens(-0.04, 0.004, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    "everything": lensmodel.Lens(-0.05, 0.006, -0.0004, 0.002, -0.0015, 0.015, -0.01, 1.0),
    "anamorphic": lensmodel.Lens(-0.03, 0.002, 0.0, 0.001, 0.0005, -0.01, 0.012, 1.33),
    "pincushion": lensmodel.Lens(0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
}


def smooth_image(width, height):
    """Smooth, bounded content: a sampling position that is off by 1e-4 pixel changes it by well under 1e-4."""
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    return np.stack([0.5 + 0.4 * np.sin(xx / 37.0), 0.5 + 0.4 * np.cos(yy / 41.0),
                     0.5 + 0.3 * np.sin((xx + yy) / 53.0), np.full_like(xx, 0.9)], axis=-1).astype(np.float32)


def adapter_or_skip(case, choice):
    try:
        return gpu3d._state(choice)
    except Exception as exc:
        case.skipTest(f"{choice} adapter not usable here: {exc}")


def interior_mask(sx, sy, width, height):
    margin = gpulens.EDGE_BAND
    return (sx >= margin) & (sx <= width - 1 - margin) & (sy >= margin) & (sy <= height - 1 - margin)


class AgreementBase:
    """CPU against GPU on one adapter: every lens, both directions, every filter, with an overscan data window."""
    choice = "default"
    SIZE = (160, 90)

    def setUp(self):
        adapter_or_skip(self, self.choice)

    def compare(self, lens, direction, filter_name, display, source_data):
        pixels = smooth_image(source_data.width, source_data.height)
        if direction == "undistort":
            out = Region(*lensmodel.undistorted_window(lens, source_data, display))
        else:
            out = display
        sx, sy = lensmodel.sample_coordinates(lens, out, display, source_data, direction)
        cpu = Evaluator._resample(pixels, sx, sy, filter_name).astype(np.float32)
        try:
            gpu = gpulens.warp(lens, pixels, out, display, source_data, direction, filter_name, choice=self.choice)
        except gpulens.Unsupported as exc:
            self.skipTest(f"{self.choice} adapter cannot run a {out.width} x {out.height} lens warp: {exc}")
        self.assertEqual(gpu.shape, cpu.shape)
        self.assertTrue(np.isfinite(gpu).all(), "the GPU path produced NaN or infinity")
        inside = interior_mask(sx, sy, source_data.width, source_data.height)
        difference = np.abs(gpu - cpu).max(axis=2)
        self.assertGreater(inside.mean(), 0.3, "the test frame has too little interior to mean anything")
        self.assertLessEqual(float(difference[inside].max()), INTERIOR,
                             f"{self.choice} {direction} {filter_name}: interior differs from the CPU by more than {INTERIOR}")
        edge = difference[~inside]
        if filter_name == "nearest":
            self.assertLessEqual(float((edge > 1e-6).mean()) if edge.size else 0.0, NEAREST_MISMATCH)
        elif edge.size:
            self.assertLessEqual(float(edge.max()), EDGE_TOLERANCE,
                                 f"{self.choice} {direction} {filter_name}: edge differs by more than {EDGE_TOLERANCE}")
        return float(difference[inside].max()), float(edge.max()) if edge.size else 0.0

    def test_every_lens_direction_and_filter_agrees_with_the_cpu(self):
        width, height = self.SIZE
        display = Region(0, 0, width, height)
        for name, lens in LENSES.items():
            for direction in ("undistort", "redistort"):
                for filter_name in ("nearest", "bilinear", "cubic"):
                    with self.subTest(lens=name, direction=direction, filter=filter_name):
                        self.compare(lens, direction, filter_name, display, display)

    def test_overscan_source_and_offset_display_agree_with_the_cpu(self):
        """A redistort reads an undistorted image whose data window reaches beyond the format, and a display
        window that does not start at the origin moves every offset the shader adds."""
        width, height = self.SIZE
        for display in (Region(0, 0, width, height), Region(32, -16, width, height)):
            overscan = Region(display.x - 20, display.y - 14, width + 40, height + 28)
            for name in ("everything", "anamorphic"):
                for filter_name in ("bilinear", "cubic"):
                    with self.subTest(display=(display.x, display.y), lens=name, filter=filter_name):
                        self.compare(LENSES[name], "redistort", filter_name, display, overscan)
                        self.compare(LENSES[name], "undistort", filter_name, display, display)

    def test_full_hd_and_4k_agree_with_the_cpu(self):
        """Sample positions near 4000 pixels are where float32 loses the most; the stated tolerance still holds."""
        for width, height in ((1920, 1080), (3840, 2160)):
            display = Region(0, 0, width, height)
            for direction in ("undistort", "redistort"):
                with self.subTest(size=(width, height), direction=direction):
                    self.compare(LENSES["everything"], direction, "bilinear", display, display)

    def test_the_stated_tolerances_are_tight_enough_to_fail(self):
        """The comparison is not vacuous: a lens that differs slightly from the one the GPU ran is caught."""
        width, height = self.SIZE
        display = Region(0, 0, width, height)
        pixels = smooth_image(width, height)
        lens, other = LENSES["everything"], lensmodel.Lens(-0.05, 0.006, -0.0004, 0.002, -0.0015, 0.015, -0.01, 1.02)
        sx, sy = lensmodel.sample_coordinates(other, display, display, display, "redistort")
        cpu = Evaluator._resample(pixels, sx, sy, "bilinear")
        gpu = gpulens.warp(lens, pixels, display, display, display, "redistort", "bilinear", choice=self.choice)
        inside = interior_mask(sx, sy, width, height)
        self.assertGreater(float(np.abs(gpu - cpu).max(axis=2)[inside].max()), INTERIOR)


class DefaultAdapterAgreement(AgreementBase, unittest.TestCase):
    choice = "default"


class IntegratedAdapterAgreement(AgreementBase, unittest.TestCase):
    choice = "integrated"


class SoftwareAdapterAgreement(AgreementBase, unittest.TestCase):
    choice = "cpu"


def lens_graph(backend, filter_name="bilinear", width=192, height=108, **lens):
    d = Dispatcher()
    d.execute(dict(op="create", id="plate", type="Checker", params=dict(width=width, height=height, size=24)))
    knobs = dict(lens_preset="custom", lens_k1=-0.04, lens_k2=0.004, filter=filter_name, lens_backend=backend, **lens)
    d.execute(dict(op="create", id="undistort", type="Undistort", params=knobs))
    d.execute(dict(op="connect", id="undistort", input="image", source="plate"))
    d.execute(dict(op="create", id="redistort", type="Redistort", params=knobs))
    d.execute(dict(op="connect", id="redistort", input="image", source="undistort"))
    return d


def raster(d, target):
    return Evaluator().evaluate_raster(dict(d.document, view=target), target, frame=1, tier=1)


class NodePathTests(unittest.TestCase):
    """The nodes through the evaluator: the backend knob, the round trip and the fallback."""

    def test_the_knob_exists_with_auto_as_default(self):
        from nodebased.core import CHOICES, SPECS
        for kind in ("Undistort", "Redistort"):
            self.assertEqual(SPECS[kind]["params"]["lens_backend"], "auto")
        self.assertEqual(CHOICES["lens_backend"], ["auto", "cpu", "gpu"])

    def test_gpu_round_trip_returns_the_plate(self):
        """Undistort then Redistort on the card returns a smooth plate to within 1e-3 away from the border."""
        adapter_or_skip(self, "default")
        yy, xx = np.mgrid[:108, :192].astype(np.float32)
        plate = np.stack([0.5 + 0.4 * np.sin(xx / 37.0), 0.5 + 0.4 * np.cos(yy / 41.0),
                          0.5 + 0.3 * np.sin((xx + yy) / 53.0), np.ones_like(xx)], axis=-1).astype(np.float32)
        for filter_name in ("bilinear", "cubic"):
            with self.subTest(filter=filter_name):
                out = gpulens.warp(LENSES["everything"], plate, Region(*lensmodel.undistorted_window(
                    LENSES["everything"], Region(0, 0, 192, 108), Region(0, 0, 192, 108))),
                    Region(0, 0, 192, 108), Region(0, 0, 192, 108), "undistort", filter_name)
                grown = Region(*lensmodel.undistorted_window(
                    LENSES["everything"], Region(0, 0, 192, 108), Region(0, 0, 192, 108)))
                back = gpulens.warp(LENSES["everything"], out, Region(0, 0, 192, 108), Region(0, 0, 192, 108),
                                    grown, "redistort", filter_name)
                core = (slice(12, -12), slice(12, -12))
                self.assertLessEqual(float(np.abs(back[core] - plate[core]).max()), ROUND_TRIP)

    def test_evaluator_gpu_matches_cpu_through_the_nodes(self):
        adapter_or_skip(self, "default")
        for filter_name in ("bilinear", "cubic"):
            with self.subTest(filter=filter_name):
                cpu = raster(lens_graph("cpu", filter_name), "redistort")
                gpu = raster(lens_graph("gpu", filter_name), "redistort")
                self.assertEqual(cpu.data, gpu.data)
                self.assertEqual(cpu.display, gpu.display)
                # A checker has hard edges: a position 1e-4 pixel off moves a bilinear sample by about 1e-4.
                self.assertLessEqual(float(np.abs(cpu.pixels - gpu.pixels).max()), 5e-3)

    def test_wgpu_missing_falls_back_to_the_cpu(self):
        reference = raster(lens_graph("cpu"), "undistort")
        with mock.patch.object(gpu3d, "_state", side_effect=RuntimeError("wgpu unavailable (default): no module")), \
                mock.patch.object(gpulens, "AUTO_MIN_PIXELS", 0):
            automatic = raster(lens_graph("auto"), "undistort")
            self.assertTrue(np.array_equal(automatic.pixels, reference.pixels))
            with self.assertRaisesRegex(ValueError, "GPU Undistort unavailable: wgpu unavailable"):
                raster(lens_graph("gpu"), "undistort")

    def test_a_failing_card_falls_back_in_auto_and_reports_in_gpu(self):
        reference = raster(lens_graph("cpu"), "redistort")
        with mock.patch.object(gpulens, "warp", side_effect=RuntimeError("device lost")), \
                mock.patch.object(gpulens, "AUTO_MIN_PIXELS", 0):
            self.assertTrue(np.array_equal(raster(lens_graph("auto"), "redistort").pixels, reference.pixels))
            with self.assertRaisesRegex(ValueError, "GPU (Un|Re)distort failed: device lost"):
                raster(lens_graph("gpu"), "redistort")

    def test_auto_keeps_small_images_on_the_cpu_and_cpu_never_asks_for_the_card(self):
        with mock.patch.object(gpulens, "warp", side_effect=AssertionError("the card was used")):
            raster(lens_graph("auto"), "undistort")          # 192 x 108 is below AUTO_MIN_PIXELS
            with mock.patch.object(gpulens, "AUTO_MIN_PIXELS", 0):
                raster(lens_graph("cpu"), "undistort")

    def test_auto_uses_the_card_for_a_big_image(self):
        adapter_or_skip(self, "default")
        with mock.patch.object(gpulens, "warp", wraps=gpulens.warp) as warp:
            raster(lens_graph("auto", width=480, height=272), "redistort")
        self.assertTrue(warp.called)

    def test_an_identity_lens_passes_the_image_through_on_either_backend(self):
        for backend in ("cpu", "gpu"):
            d = Dispatcher()
            d.execute(dict(op="create", id="plate", type="Checker", params=dict(width=64, height=48, size=8)))
            d.execute(dict(op="create", id="u", type="Undistort", params=dict(lens_backend=backend)))
            d.execute(dict(op="connect", id="u", input="image", source="plate"))
            self.assertTrue(np.array_equal(raster(d, "u").pixels, raster(d, "plate").pixels))

    def test_an_oversize_source_is_unsupported_not_a_crash(self):
        """A frame wider than any adapter's texture is refused by name, and the node then runs on the CPU."""
        adapter_or_skip(self, "default")
        wide = Region(0, 0, 70000, 2)
        with self.assertRaisesRegex(gpulens.Unsupported, "texture size"):
            gpulens.warp(LENSES["barrel"], np.zeros((2, 70000, 4), np.float32), Region(0, 0, 8, 8),
                         wide, wide, "redistort", "bilinear")


if __name__ == "__main__":
    unittest.main()
