"""Step D2: close the partial rows MatchGrade/ColorTransfer, Blend, ContactSheet,
Encryptomatte, Erode (filter) and the HistEQ/Histogram/MinColor/Sampler group."""
import unittest

import numpy as np

from nodebased.core import Dispatcher, empty_document
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor


class MatchGradeBakeTests(unittest.TestCase):
    def test_matchgrade_bakes_and_keeps_working_without_a_reference(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "src", "type": "Checker", "params": {
            "width": 16, "height": 8, "size": 1}})
        d.execute({"op": "create", "id": "target", "type": "Grade", "params": {
            "exposure": 0, "multiply": 2, "offset": 0.1}})
        d.execute({"op": "connect", "id": "target", "input": "image", "source": "src"})
        d.execute({"op": "create", "id": "match", "type": "MatchGrade"})
        d.execute({"op": "connect", "id": "match", "input": "image", "source": "src"})
        d.execute({"op": "connect", "id": "match", "input": "reference", "source": "target"})
        live = Evaluator().evaluate(dict(d.document, view="match"))

        frame = int(d.document["time"]["current"])
        src_px = Evaluator().evaluate_raster(d.document, target="src", frame=frame).pixels
        dst_px = Evaluator().evaluate_raster(d.document, target="target", frame=frame).pixels
        bake = []
        for channel, c in zip("rgb", range(3)):
            sm, tm = float(src_px[..., c].mean()), float(dst_px[..., c].mean())
            ss, ts = float(src_px[..., c].std()), float(dst_px[..., c].std())
            gain = ts / ss
            bake.append({"op": "set", "id": "match", "param": f"grade_gain_{channel}", "value": gain})
            bake.append({"op": "set", "id": "match", "param": f"grade_offset_{channel}", "value": tm - sm * gain})
        bake.append({"op": "set", "id": "match", "param": "match_analyzed", "value": 1})
        d.execute({"op": "batch", "commands": bake})

        baked_with_reference = Evaluator().evaluate(dict(d.document, view="match"))
        np.testing.assert_allclose(baked_with_reference, live, atol=1e-5)

        d.execute({"op": "connect", "id": "match", "input": "reference"})
        self.assertIsNone(d.document["nodes"]["match"]["inputs"].get("reference"))
        baked_without_reference = Evaluator().evaluate(dict(d.document, view="match"))
        np.testing.assert_allclose(baked_without_reference, live, atol=1e-5)

    def test_matchgrade_without_reference_or_bake_raises_a_clear_error(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "src", "type": "Checker", "params": {
            "width": 8, "height": 4, "size": 1}})
        d.execute({"op": "create", "id": "match", "type": "MatchGrade"})
        d.execute({"op": "connect", "id": "match", "input": "image", "source": "src"})
        with self.assertRaises(ValueError):
            Evaluator().evaluate(dict(d.document, view="match"))

    def test_matchgrade_bypass_still_passes_image_with_reference_unwired(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "src", "type": "Checker", "params": {
            "width": 8, "height": 4, "size": 1}})
        d.execute({"op": "create", "id": "match", "type": "MatchGrade"})
        d.execute({"op": "connect", "id": "match", "input": "image", "source": "src"})
        d.execute({"op": "disable", "id": "match", "value": True})
        np.testing.assert_array_equal(Evaluator().evaluate(dict(d.document, view="match")),
                                      Evaluator().evaluate(dict(d.document, view="src")))


class ErodeFilterDistanceFalloffTests(unittest.TestCase):
    def disc(self, radius=21):
        yy, xx = np.mgrid[:radius, :radius]
        alpha = (((xx - 10) ** 2 + (yy - 10) ** 2) <= 25).astype(np.float32)
        image = np.zeros((radius, radius, 4), np.float32)
        image[..., 3] = alpha
        return image

    def graph(self, params, width=21, height=21):
        d = Dispatcher()
        d.execute({"op": "create", "id": "src", "type": "Constant", "params": {"width": width, "height": height}})
        d.execute({"op": "create", "id": "fx", "type": "ErodeFilter", "params": params})
        d.execute({"op": "connect", "id": "fx", "input": "image", "source": "src"})
        return d.document

    def both_paths(self, document):
        full = Evaluator().evaluate(dict(document, view="fx"))
        tiles = TileExecutor(evaluator=Evaluator(), tile_edge=5)
        self.assertTrue(tiles.supports_tiled(dict(document, view="fx"), "fx"))
        region = tiles.canvas_region(document, "fx", frame=1, tier=1)
        tiled = tiles.compose_region(document, "fx", region, frame=1, tier=1).pixels
        np.testing.assert_allclose(tiled, full, atol=1e-6)
        return full

    def test_fractional_size_blends_the_two_surrounding_integer_radii(self):
        image = self.disc()
        # Eroding the radius-5 disc by radius 2 leaves a radius-3 disc (offset 3 is the last inside
        # pixel); by radius 3 it leaves a radius-2 disc (offset 3 is already outside). A size of 2.5
        # sits exactly halfway between the two integer radii, so a box fractional falloff must be the
        # plain average of those two hard results at the boundary pixel.
        result = Evaluator._erode_filter(image, {"filter_size": 2.5, "filter_type": "box"})
        self.assertEqual(float(result[10, 12, 3]), 1.0)     # inside both radii
        self.assertAlmostEqual(float(result[10, 13, 3]), 0.5, places=6)
        self.assertEqual(float(result[10, 14, 3]), 0.0)     # outside both radii

    def test_filter_kernels_order_box_below_triangle_below_quadratic(self):
        # At a quarter of the way from radius 2 to radius 3 (size 2.25), the smoother kernels keep
        # more of the unfiltered (radius-2) value at the boundary pixel than the box kernel does:
        # box < triangle < quadratic, matching Nuke's documented smoothness ordering.
        image = self.disc()
        values = {}
        for filter_type in ("box", "triangle", "quadratic"):
            result = Evaluator._erode_filter(image, {"filter_size": 2.25, "filter_type": filter_type})
            values[filter_type] = float(result[10, 13, 3])
        self.assertAlmostEqual(values["box"], 0.75, places=6)
        self.assertAlmostEqual(values["triangle"], 0.84375, places=6)
        self.assertAlmostEqual(values["quadratic"], 0.896484375, places=6)
        self.assertLess(values["box"], values["triangle"])
        self.assertLess(values["triangle"], values["quadratic"])

    def test_gaussian_softens_the_fractional_result_further(self):
        image = self.disc()
        box = Evaluator._erode_filter(image, {"filter_size": 2.5, "filter_type": "box"})
        gaussian = Evaluator._erode_filter(image, {"filter_size": 2.5, "filter_type": "gaussian"})
        self.assertFalse(np.array_equal(box, gaussian))
        # The Gaussian blur pass cannot manufacture matte outside the original disc's extent.
        self.assertTrue(np.all(gaussian[..., 3] <= 1.0 + 1e-6))

    def test_tile_path_matches_full_frame_for_every_filter_kind(self):
        for filter_type in ("box", "triangle", "quadratic", "gaussian"):
            with self.subTest(filter_type=filter_type):
                self.both_paths(self.graph({"filter_size": 2.5, "filter_type": filter_type}))

    def test_whole_integer_size_is_unaffected_by_the_new_blend(self):
        image = self.disc()
        result = Evaluator._erode_filter(image, {"filter_size": 2.0, "filter_type": "box"})
        self.assertEqual(float(result[10, 13, 3]), 1.0)
        self.assertEqual(float(result[10, 14, 3]), 0.0)
