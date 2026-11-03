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
