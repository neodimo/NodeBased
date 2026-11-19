"""2D behaviour, lane 2 step R: Kronos and OFlow's Resampling and luminance weights.

Resampling (`resampleType`) is the filter the motion warp samples with; Weight Red, Green and Blue (`weightRed`, `weightGreen`, `weightBlue`) are
the weights of the luminance the motion is estimated on. The guide names the filters and gives no weights, so the Lanczos kernels are the ones
Transform and MotionBlur already use (`transformbehaviour`), cited as convention. A document saved before this step keeps its pixels: bilinear
and the Rec. 709 weights it always read.
"""
import copy
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from nodebased import ops2d_chrome, retimeknobs
from nodebased.core import SPECS, CHOICES, LIMITS
from nodebased.flow_nodes import warp_by_flow
from nodebased.opticalflow import flow_pair

H, W = 24, 32


def step_edge():
    out = np.zeros((H, W, 4), np.float32)
    out[:, W // 2:, :] = 1.0
    return out


def uniform(dx):
    flow = np.zeros((H, W, 2), np.float32)
    flow[..., 0] = dx
    return flow


class ResamplingFilters(unittest.TestCase):
    def test_bilinear_is_the_existing_warp_bit_for_bit(self):
        image = np.random.default_rng(3).random((H, W, 4)).astype(np.float32)
        flow = np.random.default_rng(4).normal(0, 1.3, (H, W, 2)).astype(np.float32)
        self.assertTrue(np.array_equal(retimeknobs.warp_resampled(image, flow, "Bilinear"), warp_by_flow(image, flow)))

    def test_half_pixel_step_edge_bilinear_never_rings_lanczos_does(self):
        # A step from 0 to 1 moved half a pixel. Bilinear mixes the two pixels at the edge and stays inside 0..1; Lanczos has negative lobes, so the
        # bright side overshoots 1 and the dark side undershoots 0 (the statable amount: more than 2 percent for Lanczos4, none for bilinear).
        image, flow = step_edge(), uniform(0.5)
        bilinear = retimeknobs.warp_resampled(image, flow, "Bilinear")[:, 2:-2, 0]
        self.assertLessEqual(float(bilinear.max()), 1.0 + 1e-6)
        self.assertGreaterEqual(float(bilinear.min()), -1e-6)
        for name in ("Lanczos4", "Lanczos6"):
            out = retimeknobs.warp_resampled(image, flow, name)[:, 2:-2, 0]
            self.assertGreater(float(out.max()), 1.02, name)
            self.assertLess(float(out.min()), -0.02, name)
            far_dark, far_bright = out[:, :4], out[:, -4:]
            np.testing.assert_allclose(far_dark, 0.0, atol=1e-6)         # a flat area stays flat (the weights sum to one)
            np.testing.assert_allclose(far_bright, 1.0, atol=1e-6)

    def test_whole_pixel_move_is_exact_in_every_filter(self):
        image = np.random.default_rng(5).random((H, W, 4)).astype(np.float32)
        flow = uniform(3.0)
        reference = retimeknobs.warp_resampled(image, flow, "Bilinear")
        for name in ("Lanczos4", "Lanczos6"):
            np.testing.assert_allclose(retimeknobs.warp_resampled(image, flow, name)[:, 5:-5], reference[:, 5:-5], atol=1e-5)

    def test_the_knob_changes_a_retimed_picture(self):
        # Both frames the same step edge, vectors given (FgVecs): the retimed frame at alpha 0.5 moves each half a pixel, so the filter shows.
        edge = step_edge()
        fw = uniform(1.0)
        outs = {}
        for name in retimeknobs.RESAMPLES:
            params = dict(SPECS["Kronos"]["params"], resample_type=name, interpolation="motion")
            outs[name] = retimeknobs.retimed_picture(edge, edge, 0.5, "motion", params, flow_backend="cpu", given=(fw, -fw))
        self.assertFalse(np.allclose(outs["Bilinear"], outs["Lanczos4"], atol=1e-3))
        self.assertFalse(np.allclose(outs["Lanczos4"], outs["Lanczos6"], atol=1e-4))
        self.assertLessEqual(float(outs["Bilinear"].max()), 1.0 + 1e-6)
        self.assertGreater(float(outs["Lanczos4"].max()), 1.02)

    def test_a_whole_frame_and_the_blend_method_ignore_the_filter(self):
        image = np.random.default_rng(6).random((H, W, 4)).astype(np.float32)
        other = np.random.default_rng(7).random((H, W, 4)).astype(np.float32)
        a = dict(SPECS["Kronos"]["params"], resample_type="Bilinear")
        b = dict(SPECS["Kronos"]["params"], resample_type="Lanczos6")
        for method, whole in (("blend", False), ("motion", True)):
            x = retimeknobs.retimed_picture(image, other, 0.5, method, a, flow_backend="cpu", sample_whole=whole)
            y = retimeknobs.retimed_picture(image, other, 0.5, method, b, flow_backend="cpu", sample_whole=whole)
            self.assertTrue(np.array_equal(x, y), method)


class LuminanceWeights(unittest.TestCase):
    @staticmethod
    def pair(shift):
        yy, xx = np.mgrid[:H, :W].astype(np.float64)

        def red(dx):          # a smooth picture of a few sinusoids, evaluated `dx` pixels to the left: the same picture moved `dx` pixels right
            x = xx - dx
            return (0.5 + 0.18 * np.sin(0.45 * x + 0.3 * yy + 0.7) + 0.15 * np.cos(0.31 * x - 0.52 * yy + 1.9)
                    + 0.1 * np.sin(0.22 * x + 0.61 * yy)).astype(np.float32)
        a = np.zeros((H, W, 4), np.float32)
        b = np.zeros((H, W, 4), np.float32)
        a[..., 0] = red(0)
        b[..., 0] = red(shift)                                           # red moves `shift` pixels right; green and blue never change
        a[..., 1] = b[..., 1] = 0.25
        a[..., 2] = b[..., 2] = 0.5
        a[..., 3] = b[..., 3] = 1.0
        return a, b

    def flow_size(self, weights):
        a, b = self.pair(3)
        grey = retimeknobs.weighted_grey
        flow, _, _ = flow_pair(grey(a, weights), grey(b, weights), vector_detail=3, smoothness=1.0, backend="cpu")
        return float(np.abs(flow[4:-4, 4:-4, 0]).mean())

    def test_motion_in_one_channel_is_seen_only_with_that_channel_weighted(self):
        self.assertGreater(self.flow_size((1.0, 0.0, 0.0)), 1.5)                 # red alone: the 3 pixel move is found
        self.assertLess(self.flow_size((0.0, 1.0, 0.0)), 1e-4)                   # green alone: a flat picture, no motion
        self.assertLess(self.flow_size((0.0, 0.0, 1.0)), 1e-4)
        self.assertLess(self.flow_size((0.0, 0.0, 0.0)), 1e-4)

    def test_the_weights_reach_the_retimed_picture(self):
        a, b = self.pair(4)
        outs = {}
        for name, weights in (("red", (1.0, 0.0, 0.0)), ("green", (0.0, 1.0, 0.0))):
            params = dict(SPECS["OFlow"]["params"], interpolation="motion", vector_detail=3, weight_red=weights[0], weight_green=weights[1],
                          weight_blue=weights[2])
            outs[name] = retimeknobs.retimed_picture(a, b, 0.5, "motion", params, flow_backend="cpu")
        self.assertFalse(np.allclose(outs["red"], outs["green"], atol=1e-3))
        # Green alone sees no motion: the picture is the plain cross-dissolve (a ghost); with red weighted the picture is the moved one.
        ghost = 0.5 * (a + b)
        np.testing.assert_allclose(outs["green"][4:-4, 4:-4, 0], ghost[4:-4, 4:-4, 0], atol=1e-4)
        self.assertGreater(float(np.abs(outs["red"][4:-4, 4:-4, 0] - ghost[4:-4, 4:-4, 0]).mean()), 0.01)

    def test_rec709_weights_pass_the_picture_through_untouched(self):
        image = np.random.default_rng(1).random((H, W, 4)).astype(np.float32)
        self.assertIs(retimeknobs.weighted_grey(image, retimeknobs.REC709), image)
        grey = retimeknobs.weighted_grey(image, (0.3, 0.6, 0.1))
        expected = image[..., 0] * np.float32(0.3) + image[..., 1] * np.float32(0.6) + image[..., 2] * np.float32(0.1)
        for channel in range(3):
            np.testing.assert_allclose(grey[..., channel], expected, atol=1e-6)
        self.assertTrue(np.array_equal(grey[..., 3], image[..., 3]))


class DefaultsAndOlderDocuments(unittest.TestCase):
    def test_a_new_node_has_the_guides_defaults(self):
        for kind in ("Kronos", "OFlow"):
            p = SPECS[kind]["params"]
            self.assertEqual((p["weight_red"], p["weight_green"], p["weight_blue"]), (0.3, 0.6, 0.1), kind)
            self.assertEqual(p["resample_type"], "Bilinear", kind)
            self.assertEqual((p["speed"], p["input_speed"]), (0.5, 0.5), kind)
            self.assertEqual(CHOICES["resample_type"], ["Bilinear", "Lanczos4", "Lanczos6"])
            self.assertEqual(LIMITS["weight_red"], (0.0, 1.0))

    def test_a_saved_node_gets_the_rec709_weights_and_bilinear(self):
        for kind in ("Kronos", "OFlow"):
            node = {"type": kind, "params": {"speed": 1.0}}
            ops2d_chrome.upgrade_nodes({"n": node})
            p = node["params"]
            self.assertEqual((p["weight_red"], p["weight_green"], p["weight_blue"]), retimeknobs.REC709, kind)
            self.assertEqual(p["resample_type"], "Bilinear", kind)
            self.assertEqual(p["speed"], 1.0, kind)                       # a saved speed stays what it was

    def test_a_saved_node_estimates_exactly_as_before(self):
        # The old estimator read Rec. 709 luminance inside flow_pair; a saved node's weights are those, so the picture handed to it is unchanged.
        a, b = LuminanceWeights.pair(3)
        node = {"type": "Kronos", "params": {}}
        ops2d_chrome.upgrade_nodes({"n": node})
        params = dict(SPECS["Kronos"]["params"], **{k: node["params"][k] for k in ("weight_red", "weight_green", "weight_blue", "resample_type", "vector_density", "legacy_vector_density", "motion_method")})
        params.update(interpolation="motion", vector_detail=3)
        new = retimeknobs.retimed_picture(a, b, 0.5, "motion", params, flow_backend="cpu")
        fw, bw, occ = flow_pair(a, b, vector_detail=3, smoothness=float(params["smoothness"]), backend="cpu")
        wa, wb = warp_by_flow(a, fw * np.float32(0.5)), warp_by_flow(b, bw * np.float32(0.5))
        old = wa * np.float32(0.5) + wb * np.float32(0.5)
        old[occ] = wb[occ]
        self.assertTrue(np.array_equal(new, old))

    def test_both_nodes_are_whole_frame_in_the_tiled_path(self):
        from nodebased import tiers
        # The retime reads frames from the evaluator and runs `retimed_picture` once on the whole window (the tile rule is the identity), so the
        # evaluator and the tile path are the same code.
        for kind in ("Kronos", "OFlow"):
            self.assertIs(tiers.REGION_RULES[kind], tiers._identity, kind)


class PanelOrder(unittest.TestCase):
    def test_resampling_follows_smoothness_and_the_weights_follow_flicker_compensation(self):
        from nodebased import knobs
        for kind in ("Kronos", "OFlow"):
            names = [g.params[0] for g in knobs.KNOB_LAYOUT[kind]]
            self.assertEqual(names.index("resample_type"), names.index("smoothness") + 1, kind)
            self.assertLess(names.index("flicker_compensation"), names.index("weight_red"), kind)
            self.assertEqual(names.index("weight_green"), names.index("weight_red") + 1, kind)
            self.assertEqual(names.index("weight_blue"), names.index("weight_green") + 1, kind)


class ResetButton(unittest.TestCase):
    """Reset (`resetInputRange`) writes the clip's first and last frame into Input Range, on both retimers, in one undo step."""

    def test_the_range_is_the_clips_frames(self):
        self.assertEqual(retimeknobs.reset_range({"time": {"first": 5, "last": 42}}), (5.0, 42.0))

    def test_the_button_sets_the_input_range_in_one_step(self):
        from PySide6.QtWidgets import QApplication, QPushButton
        app = QApplication.instance() or QApplication([])
        from nodebased.app import Window
        window = Window()
        try:
            window.dispatcher.execute({"op": "time", "first": 3, "last": 27, "current": 3})
            for kind in ("Kronos", "OFlow"):
                key = f"reset_{kind}"
                window.dispatcher.execute({"op": "create", "id": key, "type": kind, "pos": [0, 0]})
                window.dispatcher.execute({"op": "set", "id": key, "param": "input_start", "value": 9.0})
                window.dispatcher.execute({"op": "set", "id": key, "param": "input_end", "value": 12.0})
                panel = window.build_node_panel(key)
                button = panel.findChild(QPushButton, "retime-reset-input-range")
                self.assertIsNotNone(button, kind)
                button.click()
                app.processEvents()
                params = window.dispatcher.document["nodes"][key]["params"]
                self.assertEqual((params["input_start"], params["input_end"]), (3.0, 27.0), kind)
        finally:
            window.saved_document = window.dispatcher.document
            window.close()
            app.processEvents()


if __name__ == "__main__":
    unittest.main()
