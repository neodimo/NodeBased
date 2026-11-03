"""2D behaviour, lane 9 step E: Roto and RotoPaint compute what Nuke computes and their main tabs read like Nuke's.

* Every Roto case of tools/nuke_fidelity_cases_filter.py (shapes written in the node's payload, expected pixels from the guide's Roto page)
  passes or is listed `not scored` with its reason.
* Roto's replace: with replace off the shapes are drawn into the plate's alpha (it stays where they do not reach); a document saved before
  carries `legacy_roto_alpha` = 1 and draws the matte alone, bit for bit as it did. The tile path equals the evaluator for both.
"""
import copy
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from nodebased import knobs
from nodebased.core import SPECS, Dispatcher, empty_document, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor

N = 24


def pt(x, y):
    return {"x": float(x), "y": float(y), "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0}


def box(x0, y0, x1, y1, **extra):
    shape = {"name": "box", "mode": "union", "opacity": 1.0, "feather": 0.0, "feather_model": "edge",
             "points": [pt(x0, y0), pt(x1, y0), pt(x1, y1), pt(x0, y1)]}
    shape.update(extra)
    return shape


def document(shapes=None, plate=(0.2, 0.4, 0.6, 0.8), **params):
    d = Dispatcher(empty_document())
    d.execute({"op": "create", "id": "plate", "type": "Constant", "params": {"width": N, "height": N, "red": plate[0], "green": plate[1],
                                                                              "blue": plate[2], "alpha": plate[3]}})
    d.execute({"op": "create", "id": "r", "type": "Roto", "params": {"width": N, "height": N, **params}})
    d.execute({"op": "connect", "id": "r", "input": "bg", "source": "plate"})
    d.execute({"op": "set_shapes", "id": "r", "shapes": shapes or [box(5, 4, 17, 15, feather=3.0)]})
    d.execute({"op": "view", "id": "r"})
    return d.document


def render(doc):
    return Evaluator().evaluate_raster(doc, target="r").pixels


def tiled(doc, edge):
    executor = TileExecutor(evaluator=Evaluator(), tile_edge=edge)
    region = executor.canvas_region(doc, "r", frame=1, tier=1)
    return executor.compose_region(doc, "r", region, frame=1, tier=1).pixels


class BehaviourCaseTests(unittest.TestCase):
    def test_every_roto_case_passes_or_states_why_it_is_not_scored(self):
        import audit_nuke_fidelity as audit
        import nuke_fidelity_cases as cases
        mine = cases.cases_for("Roto")
        self.assertGreaterEqual(len(mine), 20)
        for case in mine:
            result = audit.run_case(case)
            self.assertIn(result["status"], ("pass", "not scored"), f"{case.id}: {result}")
            if result["status"] == "not scored":
                self.assertTrue(case.unscored, case.id)


class ReplaceAndAlphaTests(unittest.TestCase):
    def test_the_matte_goes_over_the_plates_alpha(self):
        px = render(document([box(5, 4, 17, 15)]))
        np.testing.assert_allclose(px[0, 0, 3], 0.8, atol=1e-6)            # outside the shape the plate's alpha stays
        np.testing.assert_allclose(px[8, 8, 3], 1.0, atol=1e-6)            # inside, a full matte over 0.8
        half = render(document([box(5, 4, 17, 15, opacity=0.5)]))
        np.testing.assert_allclose(half[8, 8, 3], 0.5 + 0.8 * 0.5, atol=1e-6)

    def test_replace_discards_the_plates_alpha(self):
        px = render(document([box(5, 4, 17, 15)], replace=1))
        self.assertEqual(float(px[0, 0, 3]), 0.0)
        self.assertEqual(float(px[8, 8, 3]), 1.0)

    def test_an_older_document_draws_the_matte_alone_bit_for_bit(self):
        new = document()
        old = copy.deepcopy(new)
        old["nodes"]["r"]["params"].pop("legacy_roto_alpha")
        loaded = upgrade_document(old)
        validate(loaded)
        self.assertEqual(loaded["nodes"]["r"]["params"]["legacy_roto_alpha"], 1)
        self.assertEqual(SPECS["Roto"]["params"]["legacy_roto_alpha"], 0)
        shapes = loaded["node_data"]["r"]["shapes"]
        matte = Evaluator().evaluate_raster(dict(loaded, view="r"), target="r").pixels
        # Reference: the coverage of the same shapes with no plate at all, written as the alpha channel.
        bare = Dispatcher(empty_document())
        bare.execute({"op": "create", "id": "r", "type": "Roto", "params": {"width": N, "height": N}})
        bare.execute({"op": "set_shapes", "id": "r", "shapes": shapes})
        bare.execute({"op": "view", "id": "r"})
        alone = Evaluator().evaluate_raster(bare.document, target="r").pixels
        np.testing.assert_array_equal(matte[..., 3], alone[..., 3])
        np.testing.assert_array_equal(matte[..., :3], render(new)[..., :3])

    def test_the_tile_path_equals_the_evaluator(self):
        for options in (dict(), dict(replace=1), dict(premultiply_mode="rgb"), dict(premultiply_mode="rgba"), dict(opacity=0.5)):
            for legacy in (0, 1):
                doc = document(**options, legacy_roto_alpha=legacy)
                expected = render(doc)
                for edge in (7, 16, 32):
                    with self.subTest(options=options, legacy=legacy, tile=edge):
                        np.testing.assert_allclose(tiled(doc, edge), expected, atol=1e-6)

    def test_premultiply_rgba_squares_the_alpha(self):
        doc = document([box(5, 4, 17, 15, feather=3.0)], plate=(0, 0, 0, 0), premultiply_mode="rgba")
        plain = document([box(5, 4, 17, 15, feather=3.0)], plate=(0, 0, 0, 0), premultiply_mode="none")
        np.testing.assert_allclose(render(doc)[..., 3], render(plain)[..., 3] ** 2, atol=1e-6)


# ---------------------------------------------------------------------------------------------------- rules 4 and 5: Roto
def labels_of(kind):
    return [(group.params[0], group.label) for group in knobs.knob_layout(kind) if group.kind != "legacy"]


class RotoMainTabTests(unittest.TestCase):
    NUKE_ORDER = ["output", "premultiply_mode", "cliptype", "replace", "width", "height", "maskChannelInput", "inject", "invert_mask", "fringe",
                  "outputMask", "opacity", "shape_motionblur_on", "global_motionblur_on", "global_motionblur", "global_motionblur_shutter",
                  "global_motionblur_shutter_offset_type", "global_motionblur_shutter_offset", "channels"]

    def test_the_main_tab_follows_the_guides_order_and_labels(self):
        got = labels_of("Roto")
        self.assertEqual([name for name, _ in got[:len(self.NUKE_ORDER)]], self.NUKE_ORDER)
        by_name = dict(got)
        for name, label in (("output", "Output"), ("premultiply_mode", "Premultiply"), ("cliptype", "Clip to"), ("replace", "Replace"),
                            ("maskChannelInput", "Mask"), ("inject", "Inject"), ("invert_mask", "Invert"), ("fringe", "Fringe"),
                            ("outputMask", "Output mask"), ("opacity", "Opacity")):
            self.assertEqual(by_name[name], label)
        self.assertEqual(len({label for _, label in got}), len(got), "no two knobs read the same")

    def test_the_old_flag_is_laid_out_once(self):
        flat = [p for group in knobs.knob_layout("Roto") for p in group.params]
        self.assertEqual(flat.count("legacy_roto_alpha"), 1)


def differs(a, b):
    return bool(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max() > 1e-6)


class RotoKnobsActTests(unittest.TestCase):
    """Every main-tab knob moved off its default changes the pixels (rule 4)."""

    def test_output_picks_the_channels_the_shapes_go_into(self):
        base = document(output="alpha")
        self.assertTrue(differs(render(base), render(document(output="rgb"))))
        self.assertTrue(differs(render(base), render(document(output="none"))))

    def test_premultiply_multiplies_the_colour_by_the_matte(self):
        plain = render(document())
        rgb = render(document(premultiply_mode="rgb"))
        self.assertTrue(differs(plain, rgb))
        self.assertLess(float(rgb[0, 0, :3].max()), float(plain[0, 0, :3].max()))      # no shape there: black

    def test_replace_discards_the_incoming_channels(self):
        self.assertTrue(differs(render(document()), render(document(replace=1))))

    def test_clip_to_cuts_the_output_to_the_window(self):
        def cropped(cliptype):
            d = Dispatcher(empty_document())
            d.execute({"op": "create", "id": "plate", "type": "Constant",
                       "params": {"width": N, "height": N, "red": 0.2, "green": 0.4, "blue": 0.6, "alpha": 1.0}})
            d.execute({"op": "create", "id": "crop", "type": "Crop", "params": {"x": 4, "y": 4, "width": 10, "height": 10}})
            d.execute({"op": "connect", "id": "crop", "input": "image", "source": "plate"})
            d.execute({"op": "create", "id": "r", "type": "Roto", "params": {"width": N, "height": N, "cliptype": cliptype}})
            d.execute({"op": "connect", "id": "r", "input": "bg", "source": "crop"})
            d.execute({"op": "set_shapes", "id": "r", "shapes": [box(2, 2, 18, 18)]})
            d.execute({"op": "view", "id": "r"})
            raster = Evaluator().evaluate_raster(d.document, target="r")
            return raster.pixels.shape, raster.pixels
        shape_format, _ = cropped("format")
        shape_bbox, _ = cropped("bbox")
        self.assertNotEqual(shape_format, shape_bbox)

    def test_opacity_scales_the_matte(self):
        full = render(document(replace=1))
        half = render(document(replace=1, opacity=0.5))
        np.testing.assert_allclose(half[..., 3], full[..., 3] * 0.5, atol=1e-6)


# ---------------------------------------------------------------------------------------------------- RotoPaint
def paint_item(**extra):
    item = {"kind": "stroke", "name": "stroke1", "points": [{"x": 10.0, "y": 10.0, "pressure": 1.0}],
            "brush": {"size": 8.0, "hardness": 1.0, "opacity": 1.0, "spacing": 0.2, "strength": 0.2}, "tool": "paint",
            "lifetime": {"mode": "all"}, "color": [1.0, 0.0, 0.0, 1.0], "source_offset": [0.0, 0.0], "source_frame": "relative",
            "opacity": 1.0, "blend": "over", "visible": True, "follow_track": None, "patch_blend": 0.0}
    item.update(extra)
    return item


def paint_document(items=None, plate=(0.2, 0.4, 0.6, 0.8), **params):
    d = Dispatcher(empty_document())
    d.execute({"op": "create", "id": "plate", "type": "Constant", "params": {"width": N, "height": N, "red": plate[0], "green": plate[1],
                                                                              "blue": plate[2], "alpha": plate[3]}})
    d.execute({"op": "create", "id": "r", "type": "RotoPaint", "params": params})
    d.execute({"op": "connect", "id": "r", "input": "image", "source": "plate"})
    d.execute({"op": "set_paint_items", "id": "r", "items": [paint_item()] if items is None else items})
    d.execute({"op": "view", "id": "r"})
    return d.document


class RotoPaintBehaviourTests(unittest.TestCase):
    def test_every_rotopaint_case_passes_or_states_why_it_is_not_scored(self):
        import audit_nuke_fidelity as audit
        import nuke_fidelity_cases as cases
        mine = cases.cases_for("RotoPaint")
        self.assertGreaterEqual(len(mine), 6)
        for case in mine:
            result = audit.run_case(case)
            self.assertIn(result["status"], ("pass", "not scored"), f"{case.id}: {result}")
            if result["status"] == "not scored":
                self.assertTrue(case.unscored, case.id)

    def test_a_harder_brush_paints_more_of_a_pixel_inside_the_rim(self):
        def red_at(hardness, x):
            item = paint_item(brush={"size": 12.0, "hardness": hardness, "opacity": 1.0, "spacing": 0.2, "strength": 0.2})
            return float(render_paint(paint_document([item]))[10, x, 0])
        for x in (13, 14, 15):                                  # 3 to 5 pixels from the centre of a radius-6 brush
            levels = [red_at(h, x) for h in (0.0, 0.25, 0.5, 0.75, 1.0)]
            self.assertEqual(levels, sorted(levels), f"column {x}: the coverage never falls as the hardness rises")
        self.assertLess(red_at(0.0, 15), red_at(1.0, 15))

    def test_a_softer_brush_falls_off_away_from_the_centre(self):
        item = paint_item(brush={"size": 14.0, "hardness": 0.2, "opacity": 1.0, "spacing": 0.2, "strength": 0.2})
        row = render_paint(paint_document([item]))[10, 10:19, 0]
        self.assertTrue(np.all(np.diff(row) <= 1e-6), "the red never rises moving away from the stroke's centre")

    def test_the_tile_path_equals_the_evaluator(self):
        doc = paint_document([paint_item(opacity=0.6)])
        expected = render_paint(doc)
        for edge in (7, 16, 32):
            executor = TileExecutor(evaluator=Evaluator(), tile_edge=edge)
            region = executor.canvas_region(doc, "r", frame=1, tier=1)
            with self.subTest(tile=edge):
                np.testing.assert_allclose(executor.compose_region(doc, "r", region, frame=1, tier=1).pixels, expected, atol=1e-6)


def render_paint(doc):
    return Evaluator().evaluate_raster(doc, target="r").pixels


class RotoPaintMainTabTests(unittest.TestCase):
    NUKE_ORDER = ["output", "premultiply_mode", "cliptype", "replace", "maskChannelInput", "inject", "invert_mask", "fringe", "outputMask", "opacity",
                  "shape_motionblur_on", "global_motionblur_on", "global_motionblur", "global_motionblur_shutter",
                  "global_motionblur_shutter_offset_type", "global_motionblur_shutter_offset", "channels"]

    def test_the_main_tab_follows_the_guides_order_and_labels(self):
        got = labels_of("RotoPaint")
        self.assertEqual([name for name, _ in got[:len(self.NUKE_ORDER)]], self.NUKE_ORDER)
        by_name = dict(got)
        for name, label in (("output", "Output"), ("premultiply_mode", "Premultiply"), ("cliptype", "Clip to"), ("replace", "Replace"),
                            ("outputMask", "Output mask"), ("opacity", "Opacity")):
            self.assertEqual(by_name[name], label)


class RotoPaintKnobsActTests(unittest.TestCase):
    def test_output_premultiply_replace_and_opacity_change_the_pixels(self):
        base = render_paint(paint_document())
        for params in (dict(output="alpha"), dict(premultiply_mode="rgb"), dict(replace=1), dict(opacity=0.5)):
            with self.subTest(params=params):
                self.assertTrue(differs(base, render_paint(paint_document(**params))))


class RotoPaintReplaceTests(unittest.TestCase):
    def test_replace_draws_the_strokes_on_a_cleared_image(self):
        plain = render_paint(paint_document())
        cleared = render_paint(paint_document(replace=1))
        self.assertTrue(np.all(cleared[0, 0] == 0.0))                    # away from the stroke the image is black and transparent
        np.testing.assert_allclose(plain[0, 0], [0.16, 0.32, 0.48, 0.8], atol=1e-6)
        np.testing.assert_allclose(cleared[10, 10], [1.0, 0.0, 0.0, 1.0], atol=1e-6)    # a full red stroke is red either way

    def test_a_clone_stroke_still_reads_the_plate_when_replace_is_on(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "plate", "type": "Checker", "params": {"width": N, "height": N, "size": 2}})
        d.execute({"op": "create", "id": "r", "type": "RotoPaint", "params": {"replace": 1}})
        d.execute({"op": "connect", "id": "r", "input": "image", "source": "plate"})
        d.execute({"op": "set_paint_items", "id": "r", "items": [paint_item(tool="clone", source_offset=[2.0, 0.0])]})
        d.execute({"op": "view", "id": "r"})
        px = render_paint(d.document)
        self.assertGreater(float(px[10, 10, 3]), 0.0)                    # something was cloned in
        self.assertEqual(float(px[0, 0, 3]), 0.0)

    def test_an_older_document_ignores_replace_bit_for_bit(self):
        new = paint_document(replace=1)
        old = copy.deepcopy(new)
        old["nodes"]["r"]["params"].pop("legacy_paint_replace")
        loaded = upgrade_document(old)
        validate(loaded)
        self.assertEqual(loaded["nodes"]["r"]["params"]["legacy_paint_replace"], 1)
        self.assertEqual(SPECS["RotoPaint"]["params"]["legacy_paint_replace"], 0)
        np.testing.assert_array_equal(render_paint(dict(loaded, view="r")), render_paint(paint_document(replace=0)))

    def test_the_tile_path_equals_the_evaluator_with_replace(self):
        doc = paint_document(replace=1)
        expected = render_paint(doc)
        for edge in (7, 16, 32):
            executor = TileExecutor(evaluator=Evaluator(), tile_edge=edge)
            region = executor.canvas_region(doc, "r", frame=1, tier=1)
            with self.subTest(tile=edge):
                np.testing.assert_allclose(executor.compose_region(doc, "r", region, frame=1, tier=1).pixels, expected, atol=1e-6)
