"""Lane L2 step 2c5 of 4 (the last of group c): Reformat, CornerPin. Worked-example assertions
against the brief's own scenarios, evaluator/tile-path parity (via the documented full-frame
fallback), cache-digest proof, bypass and CHOICES/LIMITS coverage. See docs/PARITY_2D.md for the
audit these flip."""
import unittest

import numpy as np

from nodebased.core import Dispatcher, LIMITS, REFORMAT_FORMATS, SPECS, bypass_slot
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor, SUPPORTED_TILED_KINDS

GROUP_2C5 = ("Reformat", "CornerPin")


class Graph:
    def __init__(self):
        self.d = Dispatcher()

    def add(self, key, kind, params=None, **inputs):
        self.d.execute(dict(op="create", id=key, type=kind, params=params or {}))
        for slot, source in inputs.items():
            self.d.execute(dict(op="connect", id=key, input=slot, source=source))
        return key

    def set(self, node_id, param, value):
        self.d.execute(dict(op="set", id=node_id, param=param, value=value))

    def bypass(self, key, value=True):
        self.d.execute(dict(op="disable", id=key, value=value))

    @property
    def doc(self):
        return self.d.document


def evaluator_pixels(document, target, frame=1):
    return Evaluator().evaluate(dict(document, view=target), target, frame=frame)


def tile_fallback_result(document, target, frame=1):
    """Reformat/CornerPin evaluate their input at the *same* frame but are coordinate-dependent
    (Reformat also changes the canvas size outright), which this tile executor has no per-tile
    notion of, like Transform/Crop/Mirror/the group c4 time nodes before them, so they fall back
    to the full-frame evaluator. Asserts the fallback actually happened."""
    executor = TileExecutor(evaluator=Evaluator())
    document = dict(document, view=target)
    assert not executor.supports_tiled(document, target), target
    return executor.compose(document, target, frame=frame, tier=1)


class ReformatWorkedExampleTests(unittest.TestCase):
    def test_hd720_from_1920x1080_gives_1280x720_and_keeps_centred_content_centred(self):
        g = Graph()
        g.add("plate", "Constant", dict(width=1920, height=1080, red=0.0, green=0.0, blue=0.0, alpha=1.0))
        g.add("mark", "Rectangle", dict(width=1920, height=1080, box_x=910, box_y=490,
                                        box_width=100, box_height=100, softness=0.0,
                                        red=1.0, green=1.0, blue=1.0, alpha=1.0), image="plate")
        g.add("node", "Reformat", dict(format="HD_720"), image="mark")
        out = evaluator_pixels(g.doc, "node")
        self.assertEqual(out.shape[:2], (720, 1280))
        # Centre of the target format, safely inside the shrunk marker square (which spans
        # roughly x=606..673, y=326..393 after the uniform 2/3 HD_1080->HD_720 scale).
        self.assertGreater(float(out[360, 640, 0]), 0.9)
        # A corner well outside the marker stays black.
        self.assertLess(float(out[10, 10, 0]), 0.1)

    def test_resize_none_keeps_the_pixels_and_only_changes_the_window(self):
        g = Graph()
        g.add("plate", "Checker", dict(width=960, height=540, size=64))
        g.add("node", "Reformat", dict(format="Custom", width=1200, height=800,
                                       resize_type="none", center=0), image="plate")
        out = evaluator_pixels(g.doc, "node")
        plate = evaluator_pixels(g.doc, "plate")
        self.assertEqual(out.shape[:2], (800, 1200))
        np.testing.assert_array_equal(out[:540, :960], plate)
        self.assertTrue(np.all(out[540:, :] == 0))
        self.assertTrue(np.all(out[:, 960:] == 0))

    def test_named_format_resolves_width_height_at_creation_and_via_set(self):
        g = Graph()
        node = g.add("node", "Reformat", dict(format="UHD_4K"))
        params = g.doc["nodes"][node]["params"]
        self.assertEqual((params["width"], params["height"]), (3840, 2160))
        g.set(node, "format", "2K_DCP")
        params = g.doc["nodes"][node]["params"]
        self.assertEqual((params["width"], params["height"]), (2048, 1080))

    def test_custom_format_leaves_width_height_alone(self):
        g = Graph()
        node = g.add("node", "Reformat", dict(format="Custom", width=333, height=222))
        params = g.doc["nodes"][node]["params"]
        self.assertEqual((params["width"], params["height"]), (333, 222))


class CornerPinWorkedExampleTests(unittest.TestCase):
    def test_to_points_equal_from_points_is_identity(self):
        g = Graph()
        g.add("plate", "Checker", dict(width=960, height=540, size=64))
        g.add("node", "CornerPin", image="plate")  # default params: to == from
        out = evaluator_pixels(g.doc, "node")
        plate = evaluator_pixels(g.doc, "plate")
        np.testing.assert_array_equal(out, plate)

    def test_pinning_the_top_edge_inward_by_25_percent_moves_a_known_pixel(self):
        g = Graph()
        g.add("plate", "Checker", dict(width=960, height=540, size=64))
        # Corners placed at pixel *centres* (not edges) so the correspondence used to verify the
        # warp lands exactly on a sampled pixel regardless of filter, with no rounding ambiguity.
        g.add("node", "CornerPin", dict(
            from1_x=0.5, from1_y=0.5, from2_x=959.5, from2_y=0.5,
            from3_x=0.5, from3_y=539.5, from4_x=959.5, from4_y=539.5,
            # Top edge (points 1, 2) pinned inward by 25% of the 960px width (240px) on each side;
            # bottom edge (points 3, 4) unchanged.
            to1_x=240.5, to1_y=0.5, to2_x=719.5, to2_y=0.5,
            to3_x=0.5, to3_y=539.5, to4_x=959.5, to4_y=539.5,
            filter="nearest"), image="plate")
        out = evaluator_pixels(g.doc, "node")
        plate = evaluator_pixels(g.doc, "plate")
        # The pixel that sat at the top-left corner (0, 0) is now at the computed position (240, 0).
        np.testing.assert_allclose(out[0, 240], plate[0, 0], atol=1e-6)

    def test_inverse_direction_swaps_which_quad_is_the_pre_warp_side(self):
        g = Graph()
        g.add("plate", "Checker", dict(width=960, height=540, size=64))
        forward = dict(from1_x=0.5, from1_y=0.5, from2_x=959.5, from2_y=0.5,
                       from3_x=0.5, from3_y=539.5, from4_x=959.5, from4_y=539.5,
                       to1_x=240.5, to1_y=0.5, to2_x=719.5, to2_y=0.5,
                       to3_x=0.5, to3_y=539.5, to4_x=959.5, to4_y=539.5, filter="nearest")
        g.add("node_forward", "CornerPin", forward, image="plate")
        inverse_params = dict(forward, direction="inverse")
        g.add("node_inverse", "CornerPin", inverse_params, image="plate")
        forward_out = evaluator_pixels(g.doc, "node_forward")
        inverse_out = evaluator_pixels(g.doc, "node_inverse")
        self.assertFalse(np.array_equal(forward_out, inverse_out))


class TilePathParityTests(unittest.TestCase):
    def test_both_kinds_fall_back_and_match_the_evaluator(self):
        for kind, params in (("Reformat", dict(format="HD_720")),
                             ("CornerPin", dict(to1_x=100.0))):
            with self.subTest(kind=kind):
                self.assertNotIn(kind, SUPPORTED_TILED_KINDS)
                g = Graph()
                g.add("plate", "Checker", dict(width=960, height=540, size=64))
                g.add("node", kind, params, image="plate")
                ev = evaluator_pixels(g.doc, "node")
                result = tile_fallback_result(g.doc, "node")
                self.assertFalse(result.tiled)
                np.testing.assert_allclose(result.pixels, ev, atol=1e-6)
                self.assertEqual((result.canvas_height, result.canvas_width), ev.shape[:2])


class BypassTests(unittest.TestCase):
    def test_bypassed_kind_passes_the_input_through_unreformatted(self):
        for kind, params in (("Reformat", dict(format="HD_720")),
                             ("CornerPin", dict(to1_x=100.0))):
            with self.subTest(kind=kind):
                g = Graph()
                g.add("plate", "Checker", dict(width=960, height=540, size=64))
                g.add("node", kind, params, image="plate")
                plate = evaluator_pixels(g.doc, "plate")
                warped = evaluator_pixels(g.doc, "node")
                self.assertFalse(np.array_equal(warped.shape, plate.shape) and
                                 np.array_equal(warped, plate))
                g.bypass("node")
                bypassed = evaluator_pixels(g.doc, "node")
                np.testing.assert_array_equal(bypassed, plate)

    def test_bypass_slot_is_the_single_image_input(self):
        for kind in GROUP_2C5:
            with self.subTest(kind=kind):
                node = dict(type=kind, inputs={"image": "x"})
                self.assertEqual(bypass_slot(node), "image")


class CacheTests(unittest.TestCase):
    def test_editing_a_reformat_param_misses_the_cache_then_hits_on_revisit(self):
        g = Graph()
        g.add("plate", "Checker", dict(width=960, height=540, size=64))
        g.add("node", "Reformat", dict(format="HD_720"), image="plate")
        ev = Evaluator()
        doc = dict(g.doc, view="node")
        first = ev.evaluate(doc, "node", frame=1)
        misses_after_first = ev.misses
        g.set("node", "format", "HD_1080")
        doc = dict(g.doc, view="node")
        second = ev.evaluate(doc, "node", frame=1)
        self.assertGreater(ev.misses, misses_after_first)
        self.assertNotEqual(first.shape, second.shape)
        g.set("node", "format", "HD_720")
        doc = dict(g.doc, view="node")
        third = ev.evaluate(doc, "node", frame=1)
        np.testing.assert_array_equal(first, third)

    def test_editing_a_cornerpin_point_misses_the_cache_then_hits_on_revisit(self):
        g = Graph()
        g.add("plate", "Checker", dict(width=960, height=540, size=64))
        g.add("node", "CornerPin", image="plate")
        ev = Evaluator()
        doc = dict(g.doc, view="node")
        first = ev.evaluate(doc, "node", frame=1)
        misses_after_first = ev.misses
        g.set("node", "to1_x", 200.0)
        doc = dict(g.doc, view="node")
        second = ev.evaluate(doc, "node", frame=1)
        self.assertGreater(ev.misses, misses_after_first)
        self.assertFalse(np.array_equal(first, second))
        g.set("node", "to1_x", 0.0)
        doc = dict(g.doc, view="node")
        third = ev.evaluate(doc, "node", frame=1)
        np.testing.assert_array_equal(first, third)


class SpecCoverageTests(unittest.TestCase):
    def test_nodes_are_registered_with_an_image_input_optional_mask_and_mix(self):
        for kind in GROUP_2C5:
            with self.subTest(kind=kind):
                self.assertIn(kind, SPECS)
                self.assertEqual(SPECS[kind]["inputs"], ["image"])
                self.assertIn("mask", SPECS[kind].get("optional_inputs", []))
                self.assertIn("mix", SPECS[kind]["params"])

    def test_limits_cover_every_new_param(self):
        for name in ("width", "height", "pixel_aspect", "scale", "center", "flip", "flop", "turn",
                    "preserve_bbox"):
            self.assertIn(name, LIMITS)
        for i in (1, 2, 3, 4):
            for axis in ("x", "y"):
                self.assertIn(f"from{i}_{axis}", LIMITS)
                self.assertIn(f"to{i}_{axis}", LIMITS)

    def test_reformat_formats_table_matches_choices(self):
        from nodebased.core import CHOICES
        self.assertEqual(set(CHOICES["format"]), set(REFORMAT_FORMATS) | {"Custom"})

    def test_dispatcher_creates_every_new_node_with_valid_defaults(self):
        for kind in GROUP_2C5:
            with self.subTest(kind=kind):
                d = Dispatcher()
                result = d.execute(dict(op="create", type=kind))
                self.assertIn("id", result["result"])


if __name__ == "__main__":
    unittest.main()
