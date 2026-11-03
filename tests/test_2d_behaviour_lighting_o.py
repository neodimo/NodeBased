"""2D behaviour, lane 9 step O: Emboss and BumpBoss get Nuke's missing knobs.

* Emboss Type: traditional adds the relief to the picture (a flat picture is unchanged), effect outputs the relief alone about mid grey; the
  Edge Detector (simple, sobel, prewitt) reads the same ramp the same and a corner differently; angle defaults to 0.
* BumpBoss: the light leans from the image center toward the light position; light height zero gives no emboss, negative flips it; bump size and
  intensity scale the relief; the minimum shadow is a floor under the darkest factor; the center and light position each change the picture;
  graphs saved before keep their grey emboss bit for bit; tiles equal the full frame, with and without a mask.
"""
import copy
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from nodebased import core, knobs, lightingbehaviour
from nodebased.core import SPECS
from nodebased.imaging import Evaluator
from tests.test_2d_behaviour_filter_a import Graph, differs, evaluator_pixels, kern, old_document, source_graph, texture, tile_pixels


def step_picture(h=9, w=9):
    """A vertical step in red (dark left, bright right) with the other channels mid grey and alpha 1."""
    image = np.full((h, w, 4), 0.5, np.float32)
    image[..., 3] = 1.0
    image[..., 0] = 0.0
    image[:, w // 2:, 0] = 1.0
    return image


def ramp_picture(h=9, w=9):
    image = np.zeros((h, w, 4), np.float32)
    image[..., :3] = np.linspace(0.0, 0.5, w)[None, :, None]
    image[..., 3] = 1.0
    return image


class EmbossTypeTests(unittest.TestCase):
    def test_a_flat_picture_is_unchanged_by_traditional_and_grey_by_effect(self):
        flat = np.full((9, 9, 4), 0.37, np.float32)
        np.testing.assert_allclose(kern("Emboss", flat, emboss_type="traditional"), flat, atol=1e-6)
        np.testing.assert_allclose(kern("Emboss", flat, emboss_type="effect")[..., :3], 0.5, atol=1e-6)

    def test_traditional_is_the_picture_plus_the_relief_the_effect_shows(self):
        pixels = np.clip(texture(), 0.0, 1.0)
        traditional = kern("Emboss", pixels, emboss_type="traditional", width=0.1, angle=0.0)
        effect = kern("Emboss", pixels, emboss_type="effect", width=0.1, angle=0.0)
        mean_rgb = (pixels[..., :3].astype(np.float64)).mean(axis=2)
        # the effect relief is inside the clip for a small width, so the two differ by the picture and mid grey
        np.testing.assert_allclose(traditional[..., 0] - pixels[..., 0], effect[..., 0] - 0.5, atol=1e-5)
        self.assertTrue(differs(traditional, effect))
        self.assertTrue(np.isfinite(mean_rgb).all())

    def test_the_emboss_type_acts(self):
        pixels = texture()
        self.assertTrue(differs(kern("Emboss", pixels, emboss_type="traditional"), kern("Emboss", pixels, emboss_type="effect")))

    def test_the_new_node_starts_at_the_guide_defaults(self):
        params = SPECS["Emboss"]["params"]
        self.assertEqual((params["emboss_type"], params["angle"], params["width"], params["edge_detector"]), ("traditional", 0.0, 1.0, "sobel"))


class EdgeDetectorTests(unittest.TestCase):
    def test_the_three_detectors_read_a_ramp_alike_and_a_corner_differently(self):
        ramp = ramp_picture()
        reads = [kern("Emboss", ramp, emboss_type="effect", edge_detector=d, angle=0.0)[4, 4, 0] for d in ("simple", "sobel", "prewitt")]
        np.testing.assert_allclose(reads, reads[0], atol=1e-6)
        corner = np.zeros((9, 9, 4), np.float32)
        corner[..., 3] = 1.0
        corner[4:, 4:, :3] = 1.0
        out = {d: kern("Emboss", corner, emboss_type="effect", edge_detector=d, angle=45.0) for d in ("simple", "sobel", "prewitt")}
        self.assertTrue(differs(out["simple"], out["sobel"]))
        self.assertTrue(differs(out["sobel"], out["prewitt"]))

    def test_sobel_weights_the_middle_row_twice_as_much_as_prewitt(self):
        line = np.zeros((9, 9, 4), np.float32)
        line[..., 3] = 1.0
        line[4, 4:, :3] = 1.0                                   # a one-row step: only the row it is on and its neighbours see it
        sobel = kern("Emboss", line, emboss_type="effect", edge_detector="sobel", angle=0.0)[4, 4, 0] - 0.5
        prewitt = kern("Emboss", line, emboss_type="effect", edge_detector="prewitt", angle=0.0)[4, 4, 0] - 0.5
        self.assertAlmostEqual(sobel / prewitt, (2.0 / 4.0) / (1.0 / 3.0), places=5)


class BumpBossTests(unittest.TestCase):
    def lit(self, image=None, **knobs_):
        return kern("BumpBoss", step_picture() if image is None else image, height_channel="rgba.red", **knobs_)

    def test_the_new_node_starts_at_the_guide_defaults(self):
        p = SPECS["BumpBoss"]["params"]
        self.assertEqual((p["lightposition_x"], p["lightposition_y"], p["center_x"], p["center_y"]), (480.0, 360.0, 320.0, 240.0))
        self.assertEqual((p["bumpsize"], p["light_height"], p["intensity"], p["minshadow"], p["legacy_bumpboss"]), (1.0, 1.0, 1.0, 0.0, 0))

    def test_light_height_zero_is_level_with_the_image_and_gives_no_emboss(self):
        image = step_picture()
        np.testing.assert_allclose(self.lit(light_height=0.0), image, atol=1e-6)

    def test_a_negative_height_puts_the_light_behind_and_flips_the_relief(self):
        image = step_picture().astype(np.float64)
        front = self.lit(light_height=1.0)[..., 1] / image[..., 1] - 1.0
        behind = self.lit(light_height=-1.0)[..., 1] / image[..., 1] - 1.0
        np.testing.assert_allclose(front, -behind, atol=1e-6)
        self.assertGreater(float(np.abs(front).max()), 0.05)

    def test_the_light_leaning_toward_the_step_brightens_and_leaning_away_darkens_it(self):
        image = step_picture()
        right = self.lit(lightposition_x=480.0, lightposition_y=240.0)        # leans toward +x, up the slope of the step
        left = self.lit(lightposition_x=160.0, lightposition_y=240.0)
        self.assertGreater(float(right[4, 4, 1]), 0.5)
        self.assertLess(float(left[4, 4, 1]), 0.5)
        np.testing.assert_allclose(right[:, 0, 1], image[:, 0, 1], atol=1e-6)         # away from the step: no slope, no change

    def test_a_light_straight_above_the_centre_leans_up_the_rows_and_a_horizontal_step_lights_by_it(self):
        flat_rows = step_picture().transpose(1, 0, 2).copy()                  # a step across the rows
        up = self.lit(flat_rows, lightposition_x=320.0, lightposition_y=480.0)        # Nuke's y runs up: up the picture, toward row 0
        down = self.lit(flat_rows, lightposition_x=320.0, lightposition_y=0.0)
        self.assertLess(float(up[4, 4, 1]), 0.5)                              # the step brightens toward larger row numbers
        self.assertGreater(float(down[4, 4, 1]), 0.5)

    def test_the_light_on_the_centre_has_no_direction_and_gives_no_emboss(self):
        np.testing.assert_allclose(self.lit(lightposition_x=320.0, lightposition_y=240.0), step_picture(), atol=1e-6)

    def test_the_distance_of_the_light_does_not_change_the_lean(self):
        near = self.lit(lightposition_x=400.0, lightposition_y=240.0)
        far = self.lit(lightposition_x=900.0, lightposition_y=240.0)
        np.testing.assert_allclose(near, far, atol=1e-6)

    def test_bump_size_and_intensity_scale_the_relief(self):
        image = step_picture().astype(np.float64)
        base = self.lit(lightposition_y=240.0)[4, 4, 1] / image[4, 4, 1] - 1.0
        for knob in ("bumpsize", "intensity"):
            with self.subTest(knob=knob):
                double = self.lit(lightposition_y=240.0, **{knob: 2.0})[4, 4, 1] / image[4, 4, 1] - 1.0
                self.assertAlmostEqual(double, 2.0 * base, places=5)

    def test_the_minimum_shadow_is_the_floor_under_the_darkest_factor(self):
        image = step_picture().astype(np.float64)
        dark = self.lit(lightposition_x=160.0, lightposition_y=240.0, intensity=8.0)         # factor driven under zero at the step
        self.assertLessEqual(float(dark[4, 4, 1]), 1e-6)
        floored = self.lit(lightposition_x=160.0, lightposition_y=240.0, intensity=8.0, minshadow=0.6)
        np.testing.assert_allclose(floored[..., 1], np.maximum(0.6, (dark[..., 1] / image[..., 1])) * image[..., 1], atol=1e-6)
        self.assertGreaterEqual(float(floored[..., 1].min()), 0.6 * 0.5 - 1e-6)

    def test_the_alpha_and_the_height_channel_stay(self):
        image = step_picture()
        image[..., 3] = 0.25
        out = self.lit(image, lightposition_y=240.0)
        np.testing.assert_allclose(out[..., 3], 0.25, atol=1e-7)

    def test_every_paired_knob_acts(self):
        image = np.clip(texture(), 0.0, 1.0)
        base = dict(height_channel="rgba.red", lightposition_y=240.0)
        for knob, value in (("lightposition_x", 160.0), ("lightposition_y", 400.0), ("bumpsize", 3.0), ("light_height", 2.0), ("intensity", 3.0),
                            ("minshadow", 1.0), ("center_x", 600.0), ("center_y", 100.0), ("light_angle", 90.0), ("height_channel", "rgba.green")):
            with self.subTest(knob=knob):
                self.assertTrue(differs(kern("BumpBoss", image, **base), kern("BumpBoss", image, **{**base, knob: value})), knob)

    def test_a_saved_node_keeps_the_grey_emboss_bit_for_bit(self):
        image = np.clip(texture(), 0.0, 1.0)
        params = {"height_channel": "rgba.red", "light_angle": 40.0, "mix": 1.0}
        old = Evaluator._bump_boss(image, params)                         # a parameter dict that predates the knobs
        h = image[..., 0]
        a = np.pad(h, 1, mode="edge")
        dx, dy = (a[1:-1, 2:] - a[1:-1, :-2]) * 0.5, (a[2:, 1:-1] - a[:-2, 1:-1]) * 0.5
        expect = np.clip(0.5 + np.cos(np.radians(40.0)) * dx + np.sin(np.radians(40.0)) * dy, 0, 1)
        np.testing.assert_array_equal(old[..., 0], expect.astype(np.float32))
        loaded = old_document("BumpBoss", {"light_angle": 40.0}, ("lightposition_x", "lightposition_y", "bumpsize", "light_height", "intensity", "minshadow",
                                                                    "center_x", "center_y", "legacy_bumpboss"))
        self.assertEqual(loaded["nodes"]["n"]["params"]["legacy_bumpboss"], 1)
        self.assertEqual(loaded["nodes"]["n"]["params"]["light_angle"], 40.0)

    def test_a_saved_emboss_keeps_the_relief_alone_bit_for_bit(self):
        loaded = old_document("Emboss", {"angle": 135.0, "width": 2.0}, ("emboss_type", "edge_detector"))
        params = loaded["nodes"]["n"]["params"]
        self.assertEqual((params["emboss_type"], params["edge_detector"], params["angle"]), ("effect", "simple", 135.0))
        image = np.clip(texture(), 0.0, 1.0)
        lum = image[..., :3].mean(axis=2)
        a = np.pad(lum, 1, mode="edge")
        dx, dy = (a[1:-1, 2:] - a[1:-1, :-2]) * 0.5, (a[2:, 1:-1] - a[:-2, 1:-1]) * 0.5
        expect = np.clip(0.5 + 2.0 * (np.cos(np.radians(135.0)) * dx + np.sin(np.radians(135.0)) * dy), 0, 1)
        np.testing.assert_array_equal(Evaluator._emboss(image, {"angle": 135.0, "width": 2.0})[..., 0], expect.astype(np.float32))
        np.testing.assert_array_equal(Evaluator._emboss(image, params)[..., 0], expect.astype(np.float32))


class TileTests(unittest.TestCase):
    def test_tiles_equal_the_full_frame_with_and_without_a_mask(self):
        for kind, params in (("Emboss", {"edge_detector": "sobel", "width": 2.0}), ("Emboss", {"emboss_type": "effect", "edge_detector": "prewitt"}),
                             ("BumpBoss", {"lightposition_y": 240.0, "intensity": 2.0}), ("BumpBoss", {"minshadow": 0.4, "light_height": -1.5})):
            for masked in (False, True):
                doc = source_graph(kind, params, masked).doc
                full = evaluator_pixels(doc, "node")
                for edge in (7, 16, 32):
                    with self.subTest(kind=kind, params=params, masked=masked, edge=edge):
                        np.testing.assert_allclose(tile_pixels(doc, "node", edge), full, atol=1e-6)


class LayoutTests(unittest.TestCase):
    def shown(self, kind):
        return [g.label for g in knobs.KNOB_LAYOUT[kind] if g.kind != "legacy" and not g.params[0].startswith("legacy_")]

    def test_emboss_main_tab_follows_the_guide(self):
        self.assertEqual(self.shown("Emboss")[:5], ["Channels", "Emboss type", "Angle", "Width", "Edge detector"])

    def test_bumpboss_main_tab_follows_the_guide(self):
        self.assertEqual(self.shown("BumpBoss")[:7], ["Channels", "Light position xy", "Bump size", "Light height", "Light intensity", "Minimum shadow intensity",
                                                      "Image center xy"])

    def test_every_new_param_is_laid_out_once_and_has_limits_or_choices(self):
        new = {"Emboss": ("emboss_type", "edge_detector"),
               "BumpBoss": ("lightposition_x", "lightposition_y", "bumpsize", "light_height", "intensity", "minshadow", "center_x", "center_y", "legacy_bumpboss")}
        for kind, names in new.items():
            placed = [name for g in knobs.KNOB_LAYOUT[kind] for name in g.params]
            for name in names:
                with self.subTest(kind=kind, name=name):
                    self.assertEqual(placed.count(name), 1)
                    self.assertIn(name, SPECS[kind]["params"])
                    self.assertTrue(name in core.NODE_LIMITS.get(kind, {}) or name in core.LIMITS or name in core.CHOICES)


if __name__ == "__main__":
    unittest.main()
