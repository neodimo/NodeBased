"""2D behaviour, lane 9 step X4: the edge-detect page's knobs on Emboss and EdgeDetect.

Nuke 17.0 Reference Guide, Emboss and EdgeDetect pages: Threshold, Output (the output channel), Erode size, and a mask channel per block
("The channel to use as a mask. By default, the effect is limited to the non-black areas of this channel").

* every new default leaves the picture a saved node drew, and a document saved before the knobs loads with them;
* each knob changes pixels in the direction its sentence states;
* the evaluator and tiles of 7, 16 and 32 agree, with and without a mask;
* the panel lists the knobs in the guide's order.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from nodebased import knobs
from nodebased.core import CHOICES, SPECS
from nodebased.imaging import Evaluator
from tests.test_2d_behaviour_filter_a import differs, evaluator_pixels, old_document, source_graph, texture, tile_pixels

NEW_EMBOSS = {"threshold": 0.0, "edge_erode_size": 0.0, "edge_mask_channel": "none", "edge_output_channel": "none",
              "output_mask_channel": "none", "blur_mask_channel": "none"}
NEW_EDGE = {"edge_output_channel": "none", "edge_mask_channel": "none"}


def kernel(kind, picture, **knob_values):
    params = {**SPECS[kind]["params"], **knob_values}
    slots = [picture] + [None] * (len(SPECS[kind]["inputs"]) + len(SPECS[kind].get("optional_inputs", [])) - 1)
    return np.asarray(Evaluator._kernel(kind, params, slots), np.float64)


def step_picture(h=24, w=30):
    """A vertical step, 0.2 on the left half and 0.8 on the right, alpha 1; a red ramp in green and blue keeps the mask channels apart."""
    image = np.ones((h, w, 4), np.float32)
    image[..., :3] = np.where(np.arange(w) < w // 2, 0.2, 0.8).astype(np.float32)[None, :, None]
    return image


def masked_step(mask_channel, left=0.0, right=1.0):
    """The step picture whose `mask_channel` (0 red .. 3 alpha) is `left` on the left half and `right` on the right half."""
    image = step_picture()
    half = image.shape[1] // 2
    image[:, :half, mask_channel] = left
    image[:, half:, mask_channel] = right
    return image


class SavedNodesTests(unittest.TestCase):
    def test_new_nodes_carry_every_knob_at_the_value_that_changes_nothing(self):
        for name, value in NEW_EMBOSS.items():
            self.assertEqual(SPECS["Emboss"]["params"][name], value, name)
        for name, value in NEW_EDGE.items():
            self.assertEqual(SPECS["EdgeDetect"]["params"][name], value, name)

    def test_a_saved_node_loads_with_the_knobs_off_and_renders_as_before(self):
        for kind, new, params in (("Emboss", NEW_EMBOSS, dict(emboss_type="traditional", angle=0.0, width=2.0)),
                                  ("EdgeDetect", NEW_EDGE, dict(edge_type="Sobel", edge_erode_size=1.0))):
            doc = old_document(kind, params, drop=list(new))
            for name, value in new.items():
                self.assertEqual(doc["nodes"]["n"]["params"][name], value, (kind, name))
            fresh = old_document(kind, dict(params, **new), drop=[])
            np.testing.assert_array_equal(evaluator_pixels(doc, "n"), evaluator_pixels(fresh, "n"))

    def test_the_old_kernel_equals_the_new_one_at_the_defaults(self):
        pic = np.clip(texture(24, 30), 0.0, 1.0)
        for kind, new in (("Emboss", NEW_EMBOSS), ("EdgeDetect", NEW_EDGE)):
            params = {**SPECS[kind]["params"]}
            for name in new:
                params.pop(name, None)
            slots = [pic] + [None] * (len(SPECS[kind]["inputs"]) + len(SPECS[kind].get("optional_inputs", [])) - 1)
            old = np.asarray(Evaluator._kernel(kind, params, slots), np.float64)
            np.testing.assert_array_equal(old, kernel(kind, pic))

    def test_the_new_choices_are_listed(self):
        for name in ("edge_mask_channel", "output_mask_channel", "blur_mask_channel"):
            self.assertIn("rgba.alpha", CHOICES[name])
            self.assertEqual(CHOICES[name][0], "none")
        self.assertEqual(CHOICES["edge_output_channel"][0], "none")


class EdgeDetectX4Tests(unittest.TestCase):
    BASE = dict(edge_type="Sobel")

    def test_the_output_channel_takes_the_response_alone(self):
        pic = step_picture()
        whole = kernel("EdgeDetect", pic, **self.BASE)
        one = kernel("EdgeDetect", pic, edge_output_channel="rgba.alpha", **self.BASE)
        np.testing.assert_allclose(one[..., :3], pic[..., :3], atol=1e-6)               # the colour keeps the picture
        np.testing.assert_allclose(one[..., 3], whole[..., 0], atol=1e-6)               # alpha holds the response
        self.assertGreater(float(one[12, 14, 3]), 0.5)                                  # the step is bright in alpha

    def test_each_output_channel_names_its_own_channel(self):
        pic = step_picture()
        for index, token in enumerate(("rgba.red", "rgba.green", "rgba.blue", "rgba.alpha")):
            out = kernel("EdgeDetect", pic, edge_output_channel=token, **self.BASE)
            others = [c for c in range(4) if c != index]
            np.testing.assert_allclose(out[..., others], pic[..., others], atol=1e-6, err_msg=token)
            self.assertGreater(float(out[12, 14, index]), 0.5, token)

    def test_the_mask_channel_limits_the_edges_to_where_it_is_not_black(self):
        pic = masked_step(2, left=1.0, right=0.0)                  # blue is the mask: on the left half, black on the right
        plain = kernel("EdgeDetect", pic, **self.BASE)
        masked = kernel("EdgeDetect", pic, edge_mask_channel="rgba.blue", **self.BASE)
        self.assertTrue(differs(plain, masked))
        np.testing.assert_allclose(masked[:, 17:, :3], 0.0, atol=1e-6)                  # no edge where the mask is black
        np.testing.assert_allclose(masked[:, 2:10, :3], plain[:, 2:10, :3], atol=1e-6)  # the response is whole where it is white

    def test_a_mask_of_one_changes_nothing(self):
        pic = np.clip(texture(24, 30), 0.0, 1.0)
        pic[..., 3] = 1.0
        np.testing.assert_allclose(kernel("EdgeDetect", pic, edge_mask_channel="rgba.alpha"), kernel("EdgeDetect", pic), atol=1e-6)

    def test_tiles_of_7_16_and_32_equal_the_frame_with_and_without_a_mask(self):
        for masked in (False, True):
            g = source_graph("EdgeDetect", dict(self.BASE, edge_output_channel="rgba.blue", edge_mask_channel="rgba.alpha", edge_erode_size=1.0,
                                                edge_blur_size=2.0), masked)
            whole = evaluator_pixels(g.doc, "node")
            for edge in (7, 16, 32):
                with self.subTest(masked=masked, edge=edge):
                    np.testing.assert_allclose(tile_pixels(g.doc, "node", edge), whole, atol=1e-6)


class EmbossX4Tests(unittest.TestCase):
    BASE = dict(emboss_type="effect", angle=0.0, width=1.0, edge_detector="sobel", edge_channels="rgb")

    def test_a_threshold_above_the_slope_leaves_the_picture_flat(self):
        pic = step_picture()
        plain = kernel("Emboss", pic, **self.BASE)
        high = kernel("Emboss", pic, threshold=5.0, **self.BASE)
        self.assertGreater(float(np.abs(plain[12, 14, 0] - 0.5)), 0.05)                 # the step has a relief
        np.testing.assert_allclose(high[..., :3], 0.5, atol=1e-6)                       # and a big threshold removes all of it

    def test_a_threshold_takes_that_much_off_the_relief_and_keeps_its_sign(self):
        pic = step_picture()
        plain = kernel("Emboss", pic, emboss_type="traditional", **{k: v for k, v in self.BASE.items() if k != "emboss_type"})
        low = kernel("Emboss", pic, emboss_type="traditional", threshold=0.05, **{k: v for k, v in self.BASE.items() if k != "emboss_type"})
        relief = plain[..., 0] - pic[..., 0]
        cut = low[..., 0] - pic[..., 0]
        np.testing.assert_allclose(cut, np.sign(relief) * np.maximum(np.abs(relief) - 0.05, 0.0), atol=1e-6)

    def test_the_erode_size_thins_the_relief(self):
        pic = step_picture()
        plain = kernel("Emboss", pic, **self.BASE)
        eroded = kernel("Emboss", pic, edge_erode_size=2.0, **self.BASE)
        thickened = kernel("Emboss", pic, edge_erode_size=-2.0, **self.BASE)
        self.assertTrue(differs(plain, eroded))
        width = lambda a: int((np.abs(a[12, :, 0] - 0.5) > 1e-4).sum())
        self.assertNotEqual(width(thickened), width(eroded))
        self.assertGreater(width(thickened), width(plain))                              # a negative size grows it

    def test_the_output_channel_takes_the_relief_alone(self):
        pic = step_picture()
        whole = kernel("Emboss", pic, **self.BASE)
        one = kernel("Emboss", pic, edge_output_channel="rgba.alpha", **self.BASE)
        np.testing.assert_allclose(one[..., :3], pic[..., :3], atol=1e-6)
        np.testing.assert_allclose(one[..., 3], whole[..., 0], atol=1e-6)

    def test_the_edge_mask_limits_the_relief(self):
        pic = masked_step(1, left=0.0, right=1.0)                   # green is the mask: black on the left, white on the right
        pic[..., 1] = np.where(np.arange(pic.shape[1]) < 15, 0.0, 1.0)[None, :]
        plain = kernel("Emboss", pic, edge_channels="red", **{k: v for k, v in self.BASE.items() if k != "edge_channels"})
        masked = kernel("Emboss", pic, edge_channels="red", edge_mask_channel="rgba.green", **{k: v for k, v in self.BASE.items() if k != "edge_channels"})
        self.assertTrue(differs(plain, masked))
        np.testing.assert_allclose(masked[:, :14, 0], 0.5, atol=1e-6)                   # no relief where the mask is black

    def test_the_output_mask_lets_the_picture_show_through_where_it_is_black(self):
        pic = masked_step(2, left=0.0, right=1.0)
        pic[..., 2] = np.where(np.arange(pic.shape[1]) < 15, 0.0, 1.0)[None, :]
        out = kernel("Emboss", pic, output_mask_channel="rgba.blue", **self.BASE)
        np.testing.assert_allclose(out[:, :14], pic[:, :14], atol=1e-6)                 # the picture, untouched, where the mask is black
        self.assertTrue(differs(out[:, 16:], pic[:, 16:]))

    def test_the_blur_mask_limits_the_pre_blur(self):
        pic = np.clip(texture(24, 30), 0.0, 1.0)
        pic[..., 3] = np.where(np.arange(30) < 15, 0.0, 1.0)[None, :]
        args = dict(blur_size=4.0, **self.BASE)
        plain = kernel("Emboss", pic, **args)
        masked = kernel("Emboss", pic, blur_mask_channel="rgba.alpha", **args)
        unblurred = kernel("Emboss", pic, **self.BASE)
        self.assertTrue(differs(plain, masked))
        np.testing.assert_allclose(masked[:, 3:11], unblurred[:, 3:11], atol=1e-5)      # no blur where the mask is black, so the relief is the unblurred one

    def test_tiles_of_7_16_and_32_equal_the_frame_with_and_without_a_mask(self):
        for masked in (False, True):
            g = source_graph("Emboss", dict(self.BASE, threshold=0.02, edge_erode_size=2.0, edge_output_channel="rgba.green",
                                            edge_mask_channel="rgba.alpha", output_mask_channel="rgba.blue", blur_mask_channel="rgba.red",
                                            blur_size=2.0), masked)
            whole = evaluator_pixels(g.doc, "node")
            for edge in (7, 16, 32):
                with self.subTest(masked=masked, edge=edge):
                    np.testing.assert_allclose(tile_pixels(g.doc, "node", edge), whole, atol=1e-6)


class PanelX4Tests(unittest.TestCase):
    def test_emboss_lists_the_knobs_in_the_guides_order(self):
        order = [g.params[0] for g in knobs.KNOB_LAYOUT["Emboss"]]
        want = ["emboss_type", "angle", "width", "edge_mask_channel", "edge_detector", "edge_channels", "threshold", "edge_output_channel",
                "output_mask_channel", "edge_erode_size", "blur_size", "blur_quality", "blur_mask_channel"]
        self.assertEqual([n for n in order if n in want], want)

    def test_edgedetect_lists_the_knobs_in_the_guides_order(self):
        order = [g.params[0] for g in knobs.KNOB_LAYOUT["EdgeDetect"]]
        want = ["edge_type", "threshold", "edge_output_channel", "edge_mask_channel", "edge_erode_size", "edge_blur_size", "edge_blur_quality"]
        self.assertEqual([n for n in order if n in want], want)


if __name__ == "__main__":
    unittest.main()
