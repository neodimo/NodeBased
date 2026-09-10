"""Data window vs. display window — the contract in docs/BOUNDING_BOX.md.

The failure these tests exist to prevent is silent: a plate rendered with overscan looks correct
until someone pans, stabilises or motion-blurs it, at which point the margin that was rendered for
exactly that purpose turns out to have been thrown away at ingest and the edge goes black. Nothing
raises; the shot is just wrong.

Every geometric claim here is checked against pixel values, not only against shapes, because a
window that is the right size and the wrong offset produces correctly-shaped garbage.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from nodebased.media import read_media, read_media_raster, read_media_region, write_exr
from nodebased.raster import Raster, scale_window
from nodebased.tiers import Region
from nodebased.tiles import TileRegion
from nodebased.tileexec import TileExecutor


def overscan_exr(directory, frame=64, margin=8, inside=0.1, outside=0.9):
    """A real EXR whose data window exceeds its display window on every side.

    The margin is written with a distinct value so a test can tell "the overscan survived" from
    "something edge-extended the frame", which would produce the inside value instead.
    """
    import OpenImageIO as oiio

    path = str(Path(directory) / "overscan.exr")
    size = frame + 2 * margin
    spec = oiio.ImageSpec(size, size, 4, oiio.FLOAT)
    spec.x, spec.y = -margin, -margin
    spec.full_x, spec.full_y = 0, 0
    spec.full_width, spec.full_height = frame, frame
    spec.channelnames = ["R", "G", "B", "A"]
    pixels = np.zeros((size, size, 4), np.float32)
    pixels[..., 3] = 1.0
    pixels[..., :3] = outside
    pixels[margin:margin + frame, margin:margin + frame, :3] = inside
    writer = oiio.ImageOutput.create(path)
    if writer is None or not writer.open(path, spec):
        raise AssertionError("Cannot write the overscan fixture: " + oiio.geterror())
    writer.write_image(pixels)
    writer.close()
    return path


class IngestTests(unittest.TestCase):
    def test_a_data_window_larger_than_the_frame_survives_the_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            raster = read_media_raster(overscan_exr(directory))
        self.assertEqual(raster.data, Region(-8, -8, 80, 80))
        self.assertEqual(raster.display, Region(0, 0, 64, 64))
        self.assertTrue(raster.has_overscan)
        self.assertEqual(raster.pixels.shape, (80, 80, 4))
        # The margin is the value that was written there, not an edge-extension of the frame.
        self.assertAlmostEqual(float(raster.pixels[0, 0, 0]), 0.9, places=5)
        self.assertAlmostEqual(float(raster.pixels[40, 40, 0]), 0.1, places=5)

    def test_read_media_still_returns_the_frame_so_existing_callers_are_unaffected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = overscan_exr(directory)
            frame = read_media(path)
        self.assertEqual(frame.shape, (64, 64, 4))
        # Every pixel of the display window came from inside the data window's centre.
        self.assertTrue(np.allclose(frame[..., 0], 0.1, atol=1e-5))

    def test_a_file_without_overscan_reports_coincident_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plain.exr"
            write_exr(path, np.ones((6, 5, 4), np.float32))
            raster = read_media_raster(str(path))
        self.assertEqual(raster.data, raster.display)
        self.assertFalse(raster.has_overscan)

    def test_bounded_read_acquires_overscan_without_rebasing_to_display(self):
        with tempfile.TemporaryDirectory() as directory:
            path = overscan_exr(directory)
            part = read_media_region(path, Region(-8, -8, 8, 8))
        self.assertEqual(part.data, Region(-8, -8, 8, 8))
        self.assertEqual(part.pixels.shape, (8, 8, 4))
        self.assertTrue(np.allclose(part.pixels[..., 0], 0.9, atol=1e-5))


class RasterTests(unittest.TestCase):
    def test_fit_places_pixels_by_coordinate_and_zero_fills_the_rest(self):
        raster = Raster(np.full((2, 2, 4), 0.5, np.float32), Region(3, 1, 2, 2), Region(0, 0, 6, 6))
        placed = raster.fit(Region(0, 0, 6, 6))
        self.assertEqual(placed.shape, (6, 6, 4))
        self.assertTrue(np.allclose(placed[1:3, 3:5], 0.5))
        # Everything else is transparent black: "no pixel here", not an invented edge value.
        self.assertEqual(float(placed.sum()) , float(placed[1:3, 3:5].sum()))

    def test_a_window_that_disagrees_with_its_pixels_is_rejected_on_construction(self):
        with self.assertRaisesRegex(ValueError, "does not match pixel extent"):
            Raster(np.zeros((2, 2, 4), np.float32), Region(0, 0, 3, 3))

    def test_decimated_window_extent_comes_from_the_decimated_array(self):
        """`Region.scaled` rounds both edges outward and can land a pixel wider than the array that
        decimation actually produces when the origin is not a multiple of the tier. The window has
        to follow the pixels, or a raster is one row out of step with its own data."""
        odd = Region(-7, -7, 73, 73)
        # floor(-7 / 2)=-4 and ceil(66 / 2)=33: the outward-mapped interval is 37 pixels.
        self.assertEqual(odd.scaled(2).width, 37)
        self.assertEqual(scale_window(odd, 2, 37, 37), Region(-4, -4, 37, 37))


class GraphTests(unittest.TestCase):
    """The point of the whole exercise: overscan has to be reachable by the nodes downstream."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = overscan_exr(self.directory.name)

    def read_graph(self):
        dispatcher = Dispatcher()
        dispatcher.execute({"op": "create", "type": "Read", "id": "plate", "params": {"path": self.path}})
        return dispatcher

    def test_a_pan_reveals_rendered_overscan_instead_of_black(self):
        dispatcher = self.read_graph()
        dispatcher.execute({"op": "create", "type": "Transform", "id": "pan",
                            "params": {"translate_x": 8.0, "translate_y": 8.0}})
        dispatcher.execute({"op": "connect", "id": "pan", "input": "image", "source": "plate"})
        panned = Evaluator().evaluate(dispatcher.document, "pan")
        self.assertEqual(panned.shape, (64, 64, 4))
        # This corner is what a clip-to-frame reader could only ever make transparent black.
        self.assertAlmostEqual(float(panned[0, 0, 0]), 0.9, places=5)
        self.assertAlmostEqual(float(panned[0, 0, 3]), 1.0, places=5)

    def test_a_blur_at_the_frame_edge_pulls_real_overscan_rather_than_extending_the_edge(self):
        dispatcher = self.read_graph()
        dispatcher.execute({"op": "create", "type": "Blur", "id": "soft", "params": {"radius": 4.0}})
        dispatcher.execute({"op": "connect", "id": "soft", "input": "image", "source": "plate"})
        blurred = Evaluator().evaluate(dispatcher.document, "soft")
        # The frame edge averages the 0.1 interior with the 0.9 margin, so it must land above the
        # interior value. Clipping at ingest would have edge-padded 0.1 and produced exactly 0.1.
        self.assertGreater(float(blurred[0, 32, 0]), 0.15)
        self.assertAlmostEqual(float(blurred[32, 32, 0]), 0.1, places=4)

    def test_the_data_window_travels_through_a_grade_untouched(self):
        dispatcher = self.read_graph()
        dispatcher.execute({"op": "create", "type": "Grade", "id": "g", "params": {"multiply": 2.0}})
        dispatcher.execute({"op": "connect", "id": "g", "input": "image", "source": "plate"})
        graded = Evaluator().evaluate_raster(dispatcher.document, "g")
        self.assertEqual(graded.data, Region(-8, -8, 80, 80))
        self.assertEqual(graded.display, Region(0, 0, 64, 64))
        self.assertAlmostEqual(float(graded.pixels[0, 0, 0]), 1.8, places=5)

    def test_merge_unions_data_windows_but_still_refuses_mismatched_frames(self):
        dispatcher = self.read_graph()
        dispatcher.execute({"op": "create", "type": "Constant", "id": "bg",
                            "params": {"width": 64, "height": 64, "alpha": 1.0,
                                       "red": 0.0, "green": 0.0, "blue": 0.0}})
        dispatcher.execute({"op": "create", "type": "Merge", "id": "m"})
        dispatcher.execute({"op": "connect", "id": "m", "input": "A", "source": "plate"})
        dispatcher.execute({"op": "connect", "id": "m", "input": "B", "source": "bg"})
        merged = Evaluator().evaluate_raster(dispatcher.document, "m")
        # A (80x80 with overscan) over B (64x64): the result covers both.
        self.assertEqual(merged.data, Region(-8, -8, 80, 80))
        self.assertEqual(merged.display, Region(0, 0, 64, 64))

        mismatched = Dispatcher()
        mismatched.execute({"op": "create", "type": "Constant", "id": "big", "params": {"width": 64, "height": 64}})
        mismatched.execute({"op": "create", "type": "Constant", "id": "small", "params": {"width": 16, "height": 16}})
        mismatched.execute({"op": "create", "type": "Merge", "id": "m"})
        mismatched.execute({"op": "connect", "id": "m", "input": "A", "source": "big"})
        mismatched.execute({"op": "connect", "id": "m", "input": "B", "source": "small"})
        with self.assertRaisesRegex(ValueError, "matching formats"):
            Evaluator().evaluate(mismatched.document, "m")

    def test_overscan_survives_the_proxy_tiers_in_proportion(self):
        dispatcher = self.read_graph()
        evaluator = Evaluator()
        for tier, expected in ((2, Region(-4, -4, 40, 40)), (4, Region(-2, -2, 20, 20))):
            with self.subTest(tier=tier):
                raster = evaluator.evaluate_raster(dispatcher.document, "plate", tier=tier)
                self.assertEqual(raster.data, expected)
                self.assertEqual(raster.display, Region(0, 0, 64 // tier, 64 // tier))
                self.assertAlmostEqual(float(raster.pixels[0, 0, 0]), 0.9, places=5)

    def test_tile_read_requests_only_an_overscan_region_in_source_coordinates(self):
        dispatcher = self.read_graph()
        request = TileRegion(-8, -8, 8, 8, full_x=-8, full_y=-8,
                             full_width=80, full_height=80)
        result = TileExecutor(tile_edge=64).compose_region(dispatcher.document, "plate", request)
        self.assertTrue(result.tiled)
        self.assertEqual(result.region, request)
        self.assertEqual(result.pixels.shape, (8, 8, 4))
        self.assertTrue(np.allclose(result.pixels[..., 0], 0.9, atol=1e-5))


class CropShrinksTheWorkTests(unittest.TestCase):
    """Omid's requirement, stated as a measurement rather than a claim: after a Crop, downstream
    nodes must not be computing over the discarded area."""

    def graph(self):
        dispatcher = Dispatcher()
        dispatcher.execute({"op": "create", "type": "Checker", "id": "plate",
                            "params": {"width": 512, "height": 512, "size": 16}})
        dispatcher.execute({"op": "create", "type": "Crop", "id": "crop",
                            "params": {"x": 200, "y": 200, "width": 64, "height": 64}})
        dispatcher.execute({"op": "connect", "id": "crop", "input": "image", "source": "plate"})
        dispatcher.execute({"op": "create", "type": "Blur", "id": "soft", "params": {"radius": 4.0}})
        dispatcher.execute({"op": "connect", "id": "soft", "input": "image", "source": "crop"})
        return dispatcher

    def test_a_crop_shrinks_its_data_window_and_the_blur_below_it_runs_on_that_much(self):
        dispatcher = self.graph()
        evaluator = Evaluator()
        cropped = evaluator.evaluate_raster(dispatcher.document, "crop")
        self.assertEqual(cropped.data, Region(200, 200, 64, 64))
        self.assertEqual(cropped.display, Region(0, 0, 512, 512))
        blurred = evaluator.evaluate_raster(dispatcher.document, "soft")
        # 64x64 of real work instead of 512x512 — a 64x reduction, and the number is the array's
        # own size rather than an estimate.
        self.assertEqual(blurred.pixels.shape, (64, 64, 4))
        self.assertEqual(blurred.data, Region(200, 200, 64, 64))

    def test_the_visible_frame_is_unchanged_by_the_smaller_data_window(self):
        """The optimisation has to be invisible in the output, or it is not an optimisation."""
        dispatcher = self.graph()
        result = Evaluator().evaluate(dispatcher.document, "soft")
        self.assertEqual(result.shape, (512, 512, 4))
        self.assertEqual(float(np.abs(result[:200]).sum()), 0.0)
        self.assertEqual(float(np.abs(result[264:]).sum()), 0.0)
        self.assertGreater(float(np.abs(result[200:264, 200:264]).sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
