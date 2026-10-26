"""Fidelity step C, part 6: "clip to" on the Draw nodes changes pixels. No clip, bbox, format, union bbox + format and
intersect bbox + format each decide the rectangle a Draw node draws into and outputs. Pixel assertions per mode, the
window each one sets, mask and mix, Roto, the tile path (a clip other than the format is rendered whole), bypass and old
documents."""
import copy
import unittest

import numpy as np

from nodebased import ops2d_chrome as chrome
from nodebased.core import SPECS, Dispatcher, bypass_slot, empty_document, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor
from nodebased.tiers import Region
from tests.test_roto_ui import triangle

W, H = 40, 24
FORMAT = Region(0, 0, W, H)
MODES = chrome.CLIP_TYPES
RECT = dict(box_x=0, box_y=0, box_width=W, box_height=H, softness=0.0, red=0.9, green=0.4, blue=0.1, alpha=1.0)


def graph(crop_box=None, crop_params=None, **shape):
    """A Constant plate (optionally cut by a Crop into a data window of its own) under a Rectangle."""
    d = Dispatcher(empty_document())
    d.execute({"op": "create", "id": "plate", "type": "Constant",
               "params": {"width": W, "height": H, "red": 0.2, "green": 0.3, "blue": 0.5, "alpha": 1.0}})
    source = "plate"
    if crop_box is not None:
        d.execute({"op": "create", "id": "crop", "type": "Crop",
                   "params": {"x": crop_box[0], "y": crop_box[1], "width": crop_box[2], "height": crop_box[3], **(crop_params or {})}})
        d.execute({"op": "connect", "id": "crop", "input": "image", "source": "plate"})
        source = "crop"
    params = {"width": W, "height": H, **RECT, **shape}
    d.execute({"op": "create", "id": "rect", "type": "Rectangle", "params": params})
    d.execute({"op": "connect", "id": "rect", "input": "image", "source": source})
    d.execute({"op": "view", "id": "rect"})
    return d.document


def render(doc, target="rect"):
    return Evaluator().evaluate_raster(doc, target=target)


def expected_over(doc, region, params):
    """The rectangle shape drawn over the plate (cut to its own data window), over `region`."""
    background = render(doc, "crop" if "crop" in doc["nodes"] else "plate").fit(region)
    shape = Evaluator._draw_shape("Rectangle", params, region.x, region.y, region.width, region.height, 1)
    return (shape + background * (1 - shape[..., 3:4])).astype(np.float32)


class FormatIsTheOldBehaviourTests(unittest.TestCase):
    def test_format_draws_into_the_nodes_own_format(self):
        raster = render(graph(cliptype="format"))
        self.assertEqual((raster.data, raster.display), (FORMAT, FORMAT))
        self.assertEqual(SPECS["Rectangle"]["params"]["cliptype"], "format")

    def test_the_default_and_an_explicit_format_agree_to_the_pixel(self):
        np.testing.assert_array_equal(render(graph()).pixels, render(graph(cliptype="format")).pixels)

    def test_with_nothing_wired_and_the_shape_inside_the_format_every_mode_draws_the_same(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "rect", "type": "Rectangle",
                   "params": {"width": W, "height": H, **RECT, "box_x": 5, "box_y": 4, "box_width": 20, "box_height": 10}})
        d.execute({"op": "view", "id": "rect"})
        base = render(d.document).pixels
        for mode in MODES:
            with self.subTest(mode=mode):
                d.execute({"op": "set", "id": "rect", "param": "cliptype", "value": mode})
                raster = render(d.document)
                self.assertEqual(raster.data, FORMAT)
                np.testing.assert_array_equal(raster.pixels, base)


class IncomingBoxSmallerThanFormatTests(unittest.TestCase):
    BOX = (10, 6, 20, 12)

    def test_each_mode_sets_its_window_and_draws_into_it(self):
        incoming = Region(*self.BOX)
        windows = {"no clip": FORMAT, "bbox": incoming, "format": FORMAT, "union bbox + format": FORMAT,
                   "intersect bbox + format": incoming}
        for mode, window in windows.items():
            with self.subTest(mode=mode):
                doc = graph(crop_box=self.BOX, cliptype=mode)
                raster = render(doc)
                self.assertEqual(raster.data, window)
                self.assertEqual(raster.display, FORMAT)
                params = {**SPECS["Rectangle"]["params"], "width": W, "height": H, **RECT}
                np.testing.assert_allclose(raster.pixels, expected_over(doc, window, params), atol=1e-6)

    def test_bbox_drops_the_shape_outside_the_incoming_box(self):
        whole = render(graph(crop_box=self.BOX, cliptype="format"))
        clipped = render(graph(crop_box=self.BOX, cliptype="bbox"))
        outside = whole.fit(FORMAT).copy()
        outside[6:18, 10:30] = 0
        self.assertGreater(float(outside[..., 3].max()), 0.9)                   # format draws the shape out there
        self.assertEqual(clipped.fit(FORMAT)[0, 0, 3], 0.0)                     # bbox does not
        np.testing.assert_array_equal(clipped.pixels, whole.fit(Region(*self.BOX)))


class IncomingBoxLargerThanFormatTests(unittest.TestCase):
    BOX = (-6, -4, W + 14, H + 9)
    CROP = dict(intersect=0)

    def doc(self, mode, **shape):
        return graph(crop_box=self.BOX, crop_params=self.CROP, cliptype=mode, **shape)

    def test_the_crop_really_gives_the_plate_overscan(self):
        raster = render(self.doc("format"), "crop")
        self.assertEqual(raster.data, Region(*self.BOX))

    def test_each_mode_sets_its_window(self):
        incoming = Region(*self.BOX)
        windows = {"bbox": incoming, "format": FORMAT, "union bbox + format": incoming.union(FORMAT),
                   "intersect bbox + format": FORMAT, "no clip": incoming.union(FORMAT)}
        for mode, window in windows.items():
            with self.subTest(mode=mode):
                self.assertEqual(render(self.doc(mode)).data, window)

    def test_the_shape_is_drawn_over_the_whole_larger_window(self):
        shape = dict(box_x=-6, box_y=-4, box_width=W + 14, box_height=H + 9)
        doc = self.doc("bbox", **shape)
        raster = render(doc)
        params = {**SPECS["Rectangle"]["params"], "width": W, "height": H, **RECT, **shape}
        np.testing.assert_allclose(raster.pixels, expected_over(doc, Region(*self.BOX), params), atol=1e-6)
        self.assertGreater(float(raster.pixels[0, 0, 3]), 0.99)                 # drawn out in the overscan
        self.assertEqual(float(render(self.doc("format", **shape)).pixels[0, 0, 3]), 1.0)


class NoClipTests(unittest.TestCase):
    def test_a_rectangle_standing_out_of_the_format_is_drawn_whole_with_no_clip(self):
        shape = dict(box_x=30, box_y=10, box_width=20, box_height=20)
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "rect", "type": "Rectangle", "params": {"width": W, "height": H, **RECT, **shape}})
        d.execute({"op": "view", "id": "rect"})
        clipped = render(d.document)
        self.assertEqual(clipped.data, FORMAT)
        d.execute({"op": "set", "id": "rect", "param": "cliptype", "value": "no clip"})
        free = render(d.document)
        self.assertEqual(free.data, Region(0, 0, 50, 30))
        self.assertEqual(free.display, FORMAT)
        np.testing.assert_array_equal(free.fit(FORMAT), clipped.pixels)         # inside the format nothing changed
        self.assertTrue(np.all(free.pixels[10:30, 30:50, 3] == 1.0))            # the rest of the rectangle exists out there
        self.assertEqual(float(free.pixels[0, 0, 3]), 0.0)
        d.execute({"op": "set", "id": "rect", "param": "cliptype", "value": "union bbox + format"})
        self.assertEqual(render(d.document).data, FORMAT)                       # union is only the box and the format

    def test_shapes_with_no_extent_of_their_own_stay_in_the_union(self):
        for kind in ("Ramp", "Noise", "Grid", "Text"):
            self.assertIsNone(chrome.natural_box(kind, SPECS[kind]["params"]))
        self.assertEqual(chrome.natural_box("Radial", dict(SPECS["Radial"]["params"], box_x=3, box_y=4, box_width=5, box_height=6)),
                         Region(3, 4, 5, 6))


class EveryDrawNodeTests(unittest.TestCase):
    def test_every_draw_node_honours_the_clip(self):
        boxes = {"Ramp": dict(p0_x=0, p0_y=0, p1_x=W - 1, p1_y=0), "Radial": dict(box_x=2, box_y=2, box_width=30, box_height=18),
                 "Rectangle": dict(box_x=2, box_y=2, box_width=30, box_height=18), "Noise": dict(size=9.0),
                 "Grid": dict(spacing_x=8, spacing_y=8, line_width=2.0), "Text": dict(message="Ab", font_size=20.0, box_x=2, box_y=2,
                                                                                          box_width=36, box_height=20)}
        for kind in chrome.DRAW_KINDS_WITH_CHROME:
            with self.subTest(kind=kind):
                d = Dispatcher(empty_document())
                d.execute({"op": "create", "id": "plate", "type": "Constant", "params": {"width": W, "height": H, "red": 0.2, "green": 0.3, "blue": 0.5}})
                d.execute({"op": "create", "id": "crop", "type": "Crop", "params": {"x": 10, "y": 6, "width": 20, "height": 12}})
                d.execute({"op": "connect", "id": "crop", "input": "image", "source": "plate"})
                d.execute({"op": "create", "id": "n", "type": kind, "params": {"width": W, "height": H, **boxes[kind], "cliptype": "bbox"}})
                d.execute({"op": "connect", "id": "n", "input": "image", "source": "crop"})
                d.execute({"op": "view", "id": "n"})
                bbox = render(d.document, "n")
                self.assertEqual(bbox.data, Region(10, 6, 20, 12))
                d.execute({"op": "set", "id": "n", "param": "cliptype", "value": "format"})
                whole = render(d.document, "n")
                np.testing.assert_allclose(bbox.pixels, whole.fit(Region(10, 6, 20, 12)), atol=5e-3 if kind == "Text" else 1e-6)


class GatingTests(unittest.TestCase):
    def test_mask_and_mix_work_inside_the_clip(self):
        doc = graph(crop_box=(10, 6, 20, 12), cliptype="bbox", mix=0.5)
        d = Dispatcher(doc)
        d.execute({"op": "create", "id": "m", "type": "Constant", "params": {"width": W, "height": H, "red": 1, "green": 1, "blue": 1, "alpha": 0.5}})
        d.execute({"op": "connect", "id": "rect", "input": "mask", "source": "m"})
        gated = render(d.document)
        d.document["nodes"]["rect"]["inputs"]["mask"] = None
        d.document["nodes"]["rect"]["params"]["mix"] = 1.0
        ungated = render(d.document)
        background = render(d.document, "crop").fit(gated.data)
        np.testing.assert_allclose(gated.pixels, background * 0.75 + ungated.pixels * 0.25, atol=1e-6)

    def test_bypass_passes_the_image(self):
        doc = graph(crop_box=(10, 6, 20, 12), cliptype="union bbox + format")
        doc["nodes"]["rect"]["disabled"] = True
        np.testing.assert_array_equal(render(doc).pixels, render(doc, "crop").pixels)
        self.assertEqual(bypass_slot(doc["nodes"]["rect"]), "image")


class TilePathTests(unittest.TestCase):
    def test_the_format_stays_tiled_and_any_other_clip_is_rendered_whole_to_the_same_pixels(self):
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=16)
        self.assertTrue(executor.supports_tiled(graph(), "rect"))
        for mode in MODES:
            with self.subTest(mode=mode):
                for crop_box, crop_params in (((10, 6, 20, 12), None), ((-6, -4, W + 14, H + 9), dict(intersect=0))):
                    doc = graph(crop_box=crop_box, crop_params=crop_params, cliptype=mode)
                    self.assertEqual(executor.supports_tiled(doc, "rect"), mode == "format")
                    expected = render(doc).to_display()
                    got = executor.compose(doc, "rect", frame=1, tier=1).pixels
                    np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_needs_full_frame_follows_the_clip(self):
        for kind in chrome.DRAW_KINDS_WITH_CHROME + ("Roto", "RotoPaint"):
            params = dict(SPECS[kind]["params"])
            self.assertFalse(chrome.needs_full_frame(kind, params), kind)
            self.assertTrue(chrome.needs_full_frame(kind, dict(params, cliptype="bbox")), kind)


class RotoTests(unittest.TestCase):
    def doc(self, mode, crop_box=(4, 3, 18, 12), crop_params=None):
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "plate", "type": "Constant", "params": {"width": 32, "height": 24, "red": 0.2, "green": 0.3, "blue": 0.5}})
        d.execute({"op": "create", "id": "crop", "type": "Crop", "params": {"x": crop_box[0], "y": crop_box[1], "width": crop_box[2],
                                                                            "height": crop_box[3], **(crop_params or {})}})
        d.execute({"op": "connect", "id": "crop", "input": "image", "source": "plate"})
        d.execute({"op": "create", "id": "r", "type": "Roto", "params": {"cliptype": mode, "output": "rgba"}})
        d.execute({"op": "connect", "id": "r", "input": "bg", "source": "crop"})
        shape = triangle()
        shape["points"] = [dict(point, x=point["x"] * 0.4, y=point["y"] * 0.4) for point in shape["points"]]
        d.execute({"op": "set_shapes", "id": "r", "shapes": [shape]})
        d.execute({"op": "view", "id": "r"})
        return d.document

    def test_roto_clips_to_the_incoming_box(self):
        whole = render(self.doc("format"), "r")
        for mode, window in (("bbox", Region(4, 3, 18, 12)), ("intersect bbox + format", Region(4, 3, 18, 12)),
                             ("union bbox + format", Region(0, 0, 32, 24)), ("no clip", Region(0, 0, 32, 24))):
            with self.subTest(mode=mode):
                raster = render(self.doc(mode), "r")
                self.assertEqual(raster.data, window)
                np.testing.assert_allclose(raster.pixels, whole.fit(window), atol=1e-6)

    def test_roto_with_a_larger_incoming_box_keeps_the_plates_overscan_in_the_union(self):
        raster = render(self.doc("union bbox + format", crop_box=(-5, -3, 45, 32), crop_params=dict(intersect=0)), "r")
        self.assertEqual(raster.data, Region(-5, -3, 45, 32))
        self.assertEqual(render(self.doc("intersect bbox + format", crop_box=(-5, -3, 45, 32), crop_params=dict(intersect=0)), "r").data,
                         Region(0, 0, 32, 24))


class DocumentTests(unittest.TestCase):
    def test_an_old_document_with_the_stored_clip_draws_the_same_when_it_is_format(self):
        doc = graph()
        old = copy.deepcopy(doc)
        loaded = upgrade_document(old)
        validate(loaded)
        np.testing.assert_array_equal(render(loaded).pixels, render(doc).pixels)


if __name__ == "__main__":
    unittest.main()
