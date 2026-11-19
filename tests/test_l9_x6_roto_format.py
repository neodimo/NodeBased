"""2D behaviour, lane 9 step X6: Roto's `format` is the format chooser (Nuke 17.0 Reference Guide, Roto, `format`: "format dropdown", default root.format).

Before this step Roto held its window as two integer fields. It now has the chooser Reformat uses: a named format sets the window, Custom leaves
the typed width and height alone. A document saved before the chooser loads as Custom with the window it had, and draws the same pixels on the
evaluator and in tiles.
"""
import copy
import unittest

import numpy as np

from nodebased import knobs
from nodebased.core import CHOICES, REFORMAT_FORMATS, SPECS, Dispatcher, empty_document, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor
from tests.test_roto_ui import triangle


def roto_document(**params):
    d = Dispatcher(empty_document())
    d.execute({"op": "create", "id": "r", "type": "Roto", "params": params})
    shape = triangle()
    shape["points"] = [dict(point, x=point["x"] * 0.25, y=point["y"] * 0.25) for point in shape["points"]]
    d.execute({"op": "set_shapes", "id": "r", "shapes": [shape]})
    d.execute({"op": "view", "id": "r"})
    return d.document


def render(doc):
    return Evaluator().evaluate_raster(doc, target="r").pixels


class RotoFormatTests(unittest.TestCase):
    def test_the_default_is_custom_and_the_choices_are_the_format_list(self):
        self.assertEqual(SPECS["Roto"]["params"]["format"], "Custom")
        self.assertEqual(CHOICES["format"], list(REFORMAT_FORMATS) + ["Custom"])

    def test_a_named_format_sets_the_window(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "r", "type": "Roto", "params": {}})
        d.execute({"op": "set", "id": "r", "param": "format", "value": "HD_720"})
        params = d.document["nodes"]["r"]["params"]
        self.assertEqual((params["width"], params["height"]), (1280, 720))
        validate(d.document)
        self.assertNotIn("pixel_aspect", params)

    def test_creating_with_a_named_format_sets_the_window(self):
        doc = roto_document(format="HD_1080")
        params = doc["nodes"]["r"]["params"]
        self.assertEqual((params["width"], params["height"]), (1920, 1080))
        self.assertEqual(render(doc).shape[:2], (1080, 1920))

    def test_custom_keeps_the_typed_window(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "r", "type": "Roto", "params": {"width": 300, "height": 200}})
        d.execute({"op": "set", "id": "r", "param": "format", "value": "Custom"})
        params = d.document["nodes"]["r"]["params"]
        self.assertEqual((params["width"], params["height"]), (300, 200))

    def test_the_chooser_leads_the_window_fields_on_the_panel(self):
        first = [g.params for g in knobs.knob_layout("Roto")]
        at = first.index(("format",))
        self.assertEqual(first[at + 1], ("width",))
        self.assertEqual(first[at + 2], ("height",))

    def test_an_unknown_format_name_is_refused(self):
        doc = roto_document()
        doc["nodes"]["r"]["params"]["format"] = "NotAFormat"
        with self.assertRaises(ValueError):
            validate(doc)


class OldDocumentTests(unittest.TestCase):
    def test_a_roto_saved_without_the_chooser_loads_as_custom_and_draws_the_same(self):
        doc = roto_document(width=96, height=64)
        before = render(doc)
        old = copy.deepcopy(doc)
        old["nodes"]["r"]["params"].pop("format")
        loaded = upgrade_document(old)
        validate(loaded)
        self.assertEqual(loaded["nodes"]["r"]["params"]["format"], "Custom")
        np.testing.assert_array_equal(render(loaded), before)
        self.assertEqual(before.shape[:2], (64, 96))

    def test_tiles_of_7_16_and_32_equal_the_frame_for_a_named_format_and_for_a_saved_one(self):
        named = roto_document(format="HD_720")
        legacy = copy.deepcopy(roto_document(width=96, height=64))
        legacy["nodes"]["r"]["params"].pop("format")
        legacy = upgrade_document(legacy)
        for label, doc in (("named", named), ("legacy", legacy)):
            whole = render(doc)
            for edge in (7, 16, 32):
                with self.subTest(doc=label, edge=edge):
                    executor = TileExecutor(evaluator=Evaluator(), tile_edge=edge)
                    region = executor.canvas_region(doc, "r", frame=1, tier=1)
                    got = executor.compose_region(doc, "r", region, frame=1, tier=1).pixels
                    np.testing.assert_allclose(got, whole, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
