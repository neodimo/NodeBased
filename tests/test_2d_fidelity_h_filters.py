"""Fidelity step H, part 5: CurveTool takes its Curve Type, Go! button, frames for the base average and channels; AddMix its
channels, premultiplied, nonlinear and lookup curves; DropShadow its enable switch, inherit input colour, shrink / expand and `bg`
input with an operation; Glow its saturation, width channel, non linear, effect only, filter, quality and crop to format; Median
ignore top line; Shuffle a second output layer; STMap its channels, blur, mask channel, invert and premultiplied; KeyMix its
channels, mask channel and bbox. Value and pixel assertions per option, the tile path against the evaluator, old documents,
bypass and refusal of bad settings."""
import copy
import math
import unittest

import numpy as np

from nodebased import addmixknobs, blurknobs, curvetoolknobs, glowknobs, keymixknobs, mergeknobs
from nodebased.core import SPECS, bypass_slot, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.raster import Raster
from nodebased.tileexec import TileExecutor
from tests.test_2d_parity_group_3b import Graph
from tests.test_2d_parity_step_4b import Graph as TimeGraph, frame_plate

N = 16


def render(g, key, frame=1):
    return Evaluator().evaluate_raster(dict(g.doc, view=key), frame=frame)


def pixels(g, key, frame=1):
    return render(g, key, frame).pixels


def tiled(g, key, tile_edge=5, frame=1):
    executor = TileExecutor(evaluator=Evaluator(), tile_edge=tile_edge)
    document = dict(g.doc, view=key)
    assert executor.supports_tiled(document, key), key
    region = executor.canvas_region(document, key, frame=frame, tier=1)
    return executor.compose_region(document, key, region, frame=frame, tier=1).pixels


def constant(g, key, rgba, size=N):
    return g.add(key, "Constant", dict(width=size, height=size, red=rgba[0], green=rgba[1], blue=rgba[2], alpha=rgba[3]))


# =============================================================================================== CurveTool
class CurveToolTests(unittest.TestCase):
    def graph(self, **params):
        g = TimeGraph()
        frame_plate(g, "plate", base=0.0, scale=0.05, frames=range(1, 7))        # red is 0.05 * frame, green 0.4, blue 0.6
        g.add("ct", "CurveTool", dict(frame_start=1, frame_end=4, **params), image="plate")
        return g

    @staticmethod
    def go(g, key="ct"):
        commands = curvetoolknobs.analyze(g.doc, Evaluator(), key)
        g.d.execute({"op": "batch", "commands": commands})
        return commands

    @staticmethod
    def keyed(commands):
        return {c["param"] for c in commands}

    @staticmethod
    def at(g, name, frame):
        from nodebased.animation import resolve_document
        return resolve_document(g.doc, frame)["nodes"]["ct"]["params"][name]

    def test_each_curve_type_keys_only_its_own_results(self):
        for curve_type, names in curvetoolknobs.RESULTS.items():
            with self.subTest(curve_type=curve_type):
                g = self.graph(curve_type=curve_type)
                self.assertEqual(self.keyed(self.go(g)), set(names))

    def test_avg_intensities_measure_each_frame(self):
        g = self.graph(curve_type="Avg Intensities")
        self.go(g)
        for frame in (1, 3, 4):
            self.assertAlmostEqual(self.at(g, "average_r", frame), 0.05 * frame, places=5)
            self.assertAlmostEqual(self.at(g, "average_g", frame), 0.4, places=5)
            self.assertAlmostEqual(self.at(g, "average_luminance", frame), (0.05 * frame + 0.4 + 0.6) / 3, places=5)

    def test_exposure_difference_is_taken_from_the_previous_frame_or_the_base_average(self):
        g = self.graph(curve_type="Exposure Difference")
        self.go(g)
        luma = lambda f: (0.05 * f + 0.4 + 0.6) / 3
        self.assertAlmostEqual(self.at(g, "exposure_diff", 1), 0.0, places=6)
        self.assertAlmostEqual(self.at(g, "exposure_diff", 3), luma(3) - luma(2), places=5)
        g = self.graph(curve_type="Exposure Difference", avg_frames=2)
        self.go(g)
        base = (luma(1) + luma(2)) / 2
        for frame in (1, 2, 3, 4):
            self.assertAlmostEqual(self.at(g, "exposure_diff", frame), luma(frame) - base, places=5)
        self.assertEqual(curvetoolknobs.exposure_base([1.0, 3.0, 5.0], 2), 2.0)
        self.assertIsNone(curvetoolknobs.exposure_base([1.0, 3.0], 0))

    def test_max_luma_pixel_and_auto_crop(self):
        g = TimeGraph()
        g.add("bar", "Rectangle", dict(width=N, height=N, box_x=3.0, box_y=5.0, box_width=6.0, box_height=4.0,
                                       red=1.0, green=1.0, blue=1.0, alpha=1.0))
        g.add("ct", "CurveTool", dict(frame_start=1, frame_end=1, curve_type="Auto Crop"), image="bar")
        self.go(g)
        self.assertEqual([self.at(g, name, 1) for name in ("crop_x", "crop_y", "crop_width", "crop_height")], [3.0, 5.0, 6.0, 4.0])
        g.d.execute({"op": "set", "id": "ct", "param": "curve_type", "value": "Max Luma Pixel"})
        self.go(g)
        self.assertEqual(self.at(g, "max_value", 1), 1.0)
        self.assertEqual(self.at(g, "min_value", 1), 0.0)

    def test_channels_restrict_what_is_measured(self):
        g = self.graph(curve_type="Avg Intensities", channels="red")
        self.go(g)
        self.assertAlmostEqual(self.at(g, "average_r", 2), 0.1, places=5)
        self.assertEqual(self.at(g, "average_g", 2), 0.0)
        self.assertEqual(self.at(g, "average_a", 2), 0.0)
        self.assertAlmostEqual(self.at(g, "average_luminance", 2), 0.1, places=5)                # the mean of the selected colour channels
        g = self.graph(curve_type="Avg Intensities", channels="rgb")
        self.go(g)
        self.assertEqual(self.at(g, "average_a", 2), 0.0)
        self.assertAlmostEqual(self.at(g, "average_luminance", 2), (0.1 + 0.4 + 0.6) / 3, places=5)
        g = self.graph(curve_type="Avg Intensities")
        self.go(g)
        self.assertEqual(self.at(g, "average_a", 2), 1.0)

    def test_channels_decide_what_the_auto_crop_sees(self):
        image = np.zeros((10, 10, 4), np.float32)
        image[2:5, 3:8, 3] = 1.0                                                               # coverage only in alpha
        image[2:5, 3:8, 0] = 0.5
        with_alpha = curvetoolknobs.metrics_for(image, dict(box_x=0, box_y=0, box_width=0, box_height=0, autocrop_mode="alpha",
                                                            autocrop_color_r=0.0, autocrop_color_g=0.0, autocrop_color_b=0.0,
                                                            autocrop_tolerance=0.0, channels="rgba"))
        self.assertEqual((with_alpha["crop_x"], with_alpha["crop_width"]), (3.0, 5.0))
        without = curvetoolknobs.metrics_for(image, dict(box_x=0, box_y=0, box_width=0, box_height=0, autocrop_mode="alpha",
                                                         autocrop_color_r=0.0, autocrop_color_g=0.0, autocrop_color_b=0.0,
                                                         autocrop_tolerance=0.0, channels="rgb"))
        self.assertEqual(without["crop_width"], 0.0)

    def test_the_range_and_the_input_are_checked(self):
        g = self.graph()
        g.d.execute({"op": "set", "id": "ct", "param": "frame_end", "value": 0})
        with self.assertRaisesRegex(ValueError, "end frame"):
            curvetoolknobs.analyze(g.doc, Evaluator(), "ct")
        lonely = TimeGraph()
        lonely.add("ct", "CurveTool", {})
        with self.assertRaisesRegex(ValueError, "Connect an image"):
            curvetoolknobs.analyze(lonely.doc, Evaluator(), "ct")

    def test_cancel_stops_the_analysis(self):
        g = self.graph()
        self.assertIsNone(curvetoolknobs.analyze(g.doc, Evaluator(), "ct", cancel=lambda: True))

    def test_the_go_button_keys_the_curve_types_results(self):
        from PySide6.QtWidgets import QPushButton
        from tests.test_desktop import APP, Window, release_window
        g = self.graph(curve_type="Avg Intensities")
        window = Window(g.doc)
        self.addCleanup(release_window, self)
        window.show()
        APP.processEvents()
        panel = window.build_node_panel("ct")
        button = panel.findChild(QPushButton, "curvetool-go")
        self.assertEqual(button.text(), "Go!")
        button.click()
        APP.processEvents()
        from nodebased.animation import resolve_document
        resolved = resolve_document(window.dispatcher.document, 3)["nodes"]["ct"]["params"]
        self.assertAlmostEqual(resolved["average_r"], 0.15, places=5)
        self.assertEqual(resolved["crop_width"], 0.0)                                           # not this curve type's result

    def test_an_older_curvetool_loads_with_the_new_knobs(self):
        g = self.graph()
        doc = copy.deepcopy(g.doc)
        for name in curvetoolknobs.PARAMS:
            del doc["nodes"]["ct"]["params"][name]
        loaded = upgrade_document(doc)
        self.assertEqual({name: loaded["nodes"]["ct"]["params"][name] for name in curvetoolknobs.PARAMS}, curvetoolknobs.PARAMS)
        validate(loaded)


# ================================================================================================= AddMix
class AddMixTests(unittest.TestCase):
    A = (0.8, 0.4, 0.2, 0.5)
    B = (0.1, 0.2, 0.3, 1.0)

    def graph(self, **params):
        g = Graph()
        g.add("a", "Constant", dict(width=N, height=N, red=self.A[0], green=self.A[1], blue=self.A[2], alpha=self.A[3]))
        g.add("b", "Constant", dict(width=N, height=N, red=self.B[0], green=self.B[1], blue=self.B[2], alpha=self.B[3]))
        g.add("mix", "AddMix", params, A="a", B="b")
        return g

    def out(self, **params):
        return pixels(self.graph(**params), "mix")[3, 3]

    def a_straight(self):
        return np.array(pixels(self.graph(), "a")[3, 3])              # what the Constant holds (it premultiplies its colour)

    def test_the_default_is_a_premultiplied_a_over_b(self):
        a, b = self.a_straight(), np.array(self.B, np.float32)
        expected = a * 0.5 * np.array([1, 1, 1, 0]) + b * 0.5 + np.array([0, 0, 0, 0.5])
        # rgb: A's colour times its alpha plus B times (1 - alpha); alpha: a + b (1 - a)
        expected = np.array([a[0] * 0.5 + b[0] * 0.5, a[1] * 0.5 + b[1] * 0.5, a[2] * 0.5 + b[2] * 0.5, 0.5 + 1.0 * 0.5])
        np.testing.assert_allclose(self.out(), expected, atol=1e-6)

    def test_the_lookup_curves_weight_a_and_b(self):
        a, b = self.a_straight(), np.array(self.B, np.float32)
        additive = self.out(lut_b="0:1 1:1")                                                    # B untouched: A * alpha + B
        np.testing.assert_allclose(additive[:3], a[:3] * 0.5 + b[:3], atol=1e-6)
        np.testing.assert_allclose(additive[3], 0.5 + 1.0, atol=1e-6)
        whole_a = self.out(lut_a="0:1 1:1")                                                     # A whole: A + B * (1 - alpha)
        np.testing.assert_allclose(whole_a[:3], a[:3] + b[:3] * 0.5, atol=1e-6)
        stepped = self.out(lut_a="0:0 0.25:0 0.75:1 1:1", lut_b="0:1 1:1")
        np.testing.assert_allclose(stepped[:3], a[:3] * 0.5 + b[:3], atol=1e-6)                 # the curve at 0.5 is half way up the ramp
        beyond = self.out(lut_a="0:0 0.25:0.2", lut_b="0:1 0.25:0.4")                           # held flat beyond the last point
        np.testing.assert_allclose(beyond[:3], a[:3] * 0.2 + b[:3] * 0.4, atol=1e-6)

    def test_premultiplied_divides_a_by_its_alpha_first(self):
        a, b = self.a_straight(), np.array(self.B, np.float32)
        got = self.out(premultiplied=1, lut_a="0:1 1:1", lut_b="0:1 1:1")
        np.testing.assert_allclose(got[:3], a[:3] / 0.5 + b[:3], atol=1e-6)
        zero = Graph()
        zero.add("a", "Constant", dict(width=N, height=N, red=0.5, green=0.5, blue=0.5, alpha=0.0))
        zero.add("b", "Constant", dict(width=N, height=N, red=0.1, green=0.2, blue=0.3, alpha=1.0))
        zero.add("mix", "AddMix", dict(premultiplied=1, lut_a="0:1 1:1", lut_b="0:1 1:1"), A="a", B="b")
        np.testing.assert_allclose(pixels(zero, "mix")[3, 3, :3], [0.1, 0.2, 0.3], atol=1e-6)      # no division by zero

    def test_nonlinear_remaps_the_lookup_index(self):
        a, b = self.a_straight(), np.array(self.B, np.float32)
        log = math.log1p(0.5)
        index = log / (1.0 + log)
        got = self.out(nonlinear=1)
        np.testing.assert_allclose(got[:3], a[:3] * index + b[:3] * (1.0 - index), atol=1e-6)
        self.assertFalse(np.allclose(got, self.out()))

    def test_the_channel_sets_choose_what_is_read_and_written(self):
        b = np.array(self.B, np.float32)
        no_alpha = self.out(Achannels="rgb")                                                    # A's alpha counts as zero: B alone
        np.testing.assert_allclose(no_alpha, b, atol=1e-6)
        kept_alpha = self.out(merge_output="rgb")
        self.assertAlmostEqual(float(kept_alpha[3]), self.B[3], places=6)                        # alpha is B's
        np.testing.assert_allclose(kept_alpha[:3], self.out()[:3], atol=1e-6)
        red_only = self.out(merge_output="red")
        np.testing.assert_allclose(red_only[1:], b[1:], atol=1e-6)
        self.assertAlmostEqual(float(red_only[0]), float(self.out()[0]), places=6)
        with self.assertRaisesRegex(ValueError, "layer"):
            self.out(Achannels="rgb, depth.Z")

    def test_bad_curves_are_refused(self):
        for bad in ("0:1", "a:b c:d", "0:1 0:0", ""):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.out(lut_a=bad)

    def test_mask_and_mix_still_gate_the_result(self):
        full = pixels(self.graph(lut_b="0:1 1:1"), "mix")
        half = pixels(self.graph(lut_b="0:1 1:1", mix=0.5), "mix")
        b = pixels(self.graph(), "b")
        np.testing.assert_allclose(half, 0.5 * full + 0.5 * b, atol=1e-6)

    def test_tiles_equal_the_evaluator(self):
        g = Graph()
        g.add("a", "Checker", dict(width=N, height=N, size=3))
        g.add("b", "Ramp", dict(width=N, height=N))
        g.add("mix", "AddMix", dict(premultiplied=1, nonlinear=1, lut_a="0:0.2 1:1", lut_b="0:1 1:0.3", merge_output="rgb", mix=0.7),
              A="a", B="b")
        np.testing.assert_allclose(tiled(g, "mix"), pixels(g, "mix"), atol=1e-6)

    def test_an_older_addmix_draws_the_same_and_bypass_passes_b(self):
        g = self.graph()
        before = pixels(g, "mix")
        doc = copy.deepcopy(g.doc)
        for name in addmixknobs.PARAMS:
            del doc["nodes"]["mix"]["params"][name]
        loaded = upgrade_document(doc)
        np.testing.assert_array_equal(Evaluator().evaluate_raster(dict(loaded, view="mix"), frame=1).pixels, before)
        self.assertTrue(addmixknobs.is_default(loaded["nodes"]["mix"]["params"]))
        g.bypass("mix")
        self.assertEqual(bypass_slot(g.doc["nodes"]["mix"]), "B")
        np.testing.assert_array_equal(pixels(g, "mix"), pixels(g, "b"))


# ============================================================================================== DropShadow
class DropShadowTests(unittest.TestCase):
    P = {"angle": 0.0, "distance": 6.0, "shadow_size": 0.0, "opacity": 1.0, "red": 0.0, "green": 0.0, "blue": 0.0}

    def square(self, colour=(1.0, 0.0, 0.0)):
        image = np.zeros((32, 32, 4), np.float32)
        image[10:16, 6:12] = (*colour, 1.0)
        return image

    def shadow(self, image=None, **extra):
        p = {**self.P, **extra}
        return Evaluator._drop_shadow(self.square() if image is None else image, p)

    def test_the_switch_off_draws_the_input_alone(self):
        image = self.square()
        np.testing.assert_array_equal(self.shadow(image, enable_dropshadow_effect=0), image)
        self.assertGreater(float(np.abs(self.shadow(image) - image).sum()), 1.0)

    def test_shrink_expand_grows_and_shrinks_the_shadow(self):
        image = self.square()
        cover = lambda out: int(((out[..., 3] > 0.5) & (image[..., 3] < 0.5)).sum())         # shadow pixels outside the input
        base = cover(self.shadow(image))
        wide, narrow = cover(self.shadow(image, shrink_expand=2.0)), cover(self.shadow(image, shrink_expand=-2.0))
        self.assertGreater(wide, base)
        self.assertLess(narrow, base)
        grown = self.shadow(image, shrink_expand=2.0, distance=8.0)
        self.assertEqual(float(grown[12, 20, 3]), 1.0)                                         # 2 px beyond where the plain shadow ends (x 19)
        self.assertEqual(float(self.shadow(image, distance=8.0)[12, 20, 3]), 0.0)
        np.testing.assert_array_equal(self.shadow(image, shrink_expand=0.0), self.shadow(image))

    def test_inherit_input_color_casts_the_inputs_colour(self):
        image = self.square((1.0, 0.0, 0.0))
        plain, inherited = self.shadow(image, red=0.0, green=1.0, blue=0.0), self.shadow(image, red=0.0, green=1.0, blue=0.0, inherit_input_color=1)
        np.testing.assert_allclose(plain[12, 14, :3], [0.0, 1.0, 0.0], atol=1e-6)               # the tint
        np.testing.assert_allclose(inherited[12, 14, :3], [1.0, 0.0, 0.0], atol=1e-6)           # the red of the square
        np.testing.assert_allclose(inherited[12, 14, 3], 1.0, atol=1e-6)
        np.testing.assert_array_equal(inherited[10:16, 6:12], image[10:16, 6:12])                # the input itself is untouched

    def graph(self, operation="over", **params):
        g = Graph()
        g.add("src", "Rectangle", dict(width=N, height=N, box_x=2.0, box_y=4.0, box_width=4.0, box_height=4.0,
                                       red=1.0, green=0.0, blue=0.0, alpha=1.0))
        constant(g, "bg", (0.5, 0.5, 0.5, 1.0))
        g.add("sh", "DropShadow", {**dict(angle=0.0, distance=4.0, shadow_size=0.0, opacity=1.0, operation=operation), **params},
              image="src", bg="bg")
        return g

    def test_the_bg_input_gets_the_shadowed_picture_merged_over_it(self):
        g = self.graph("over")
        out = pixels(g, "sh")
        np.testing.assert_allclose(out[5, 4, :3], [1.0, 0.0, 0.0], atol=1e-6)                   # the square, over the grey
        np.testing.assert_allclose(out[5, 8, :3], [0.0, 0.0, 0.0], atol=1e-6)                   # its (black) shadow, opaque
        np.testing.assert_allclose(out[12, 12, :3], [0.5, 0.5, 0.5], atol=1e-6)                 # away from both: the grey
        np.testing.assert_allclose(out[:, :, 3], 1.0, atol=1e-6)
        under = pixels(self.graph("under"), "sh")
        np.testing.assert_allclose(under[5, 4, :3], [0.5, 0.5, 0.5], atol=1e-6)                 # the opaque grey is over the shadowed square
        plus = pixels(self.graph("plus"), "sh")
        np.testing.assert_allclose(plus[5, 4, :3], [1.5, 0.5, 0.5], atol=1e-6)
        self.assertFalse(np.array_equal(out, under))

    def test_without_a_bg_the_operation_is_ignored(self):
        g = Graph()
        g.add("src", "Rectangle", dict(width=N, height=N, box_x=2.0, box_y=4.0, box_width=4.0, box_height=4.0,
                                       red=1.0, green=0.0, blue=0.0, alpha=1.0))
        g.add("a", "DropShadow", dict(angle=0.0, distance=4.0, shadow_size=0.0, opacity=1.0, operation="over"), image="src")
        g.add("b", "DropShadow", dict(angle=0.0, distance=4.0, shadow_size=0.0, opacity=1.0, operation="plus"), image="src")
        np.testing.assert_array_equal(pixels(g, "a"), pixels(g, "b"))

    def test_the_tiles_agree_with_every_new_knob(self):
        for params in (dict(shrink_expand=2.0), dict(shrink_expand=-1.0, shadow_size=2.0), dict(inherit_input_color=1),
                       dict(enable_dropshadow_effect=0)):
            with self.subTest(params=params):
                g = self.graph("over", **params)
                np.testing.assert_allclose(tiled(g, "sh", tile_edge=4), pixels(g, "sh"), atol=1e-6)
        mask = self.graph("over", mix=0.6)
        mask.add("gate", "Rectangle", dict(width=N, height=N, box_x=0.0, box_y=0.0, box_width=8.0, box_height=float(N),
                                           red=1.0, green=1.0, blue=1.0, alpha=1.0))
        mask.d.execute({"op": "connect", "id": "sh", "input": "mask", "source": "gate"})
        np.testing.assert_allclose(tiled(mask, "sh", tile_edge=4), pixels(mask, "sh"), atol=1e-6)

    def test_an_older_dropshadow_draws_the_same_and_bypass_passes_the_input(self):
        g = Graph()
        g.add("src", "Rectangle", dict(width=N, height=N, box_x=2.0, box_y=4.0, box_width=4.0, box_height=4.0,
                                       red=1.0, green=0.0, blue=0.0, alpha=1.0))
        g.add("sh", "DropShadow", dict(angle=0.0, distance=4.0), image="src")
        before = pixels(g, "sh")
        doc = copy.deepcopy(g.doc)
        for name in ("enable_dropshadow_effect", "inherit_input_color", "shrink_expand", "operation"):
            del doc["nodes"]["sh"]["params"][name]
        loaded = upgrade_document(doc)
        np.testing.assert_array_equal(Evaluator().evaluate_raster(dict(loaded, view="sh"), frame=1).pixels, before)
        g.bypass("sh")
        np.testing.assert_array_equal(pixels(g, "sh"), pixels(g, "src"))
        self.assertEqual(SPECS["DropShadow"]["optional_inputs"], ["mask", "bg"])


# ==================================================================================================== Glow
class GlowTests(unittest.TestCase):
    P = {"glow_threshold": 0.0, "glow_size": 3.0, "brightness": 1.0, "red": 1.0, "green": 1.0, "blue": 1.0, "channels": "rgb",
         "kernel_filter": "box", "quality": 1000.0, "crop": 1}

    def spot(self):
        image = np.zeros((21, 21, 4), np.float32)
        image[10, 10] = (0.9, 0.3, 0.1, 1.0)
        image[..., 3] = 1.0
        return image

    def glow(self, image=None, **extra):
        return Evaluator._glow(self.spot() if image is None else image, {**self.P, **extra})

    def test_saturation_moves_the_glow_toward_grey_or_away_from_it(self):
        base = self.glow()
        grey = self.glow(saturation=0.0)
        luma = float((base[..., :3] - self.spot()[..., :3])[10, 9] @ np.array([0.2126, 0.7152, 0.0722]))
        added = (grey[..., :3] - self.spot()[..., :3])[10, 9]
        np.testing.assert_allclose(added, [luma] * 3, atol=1e-6)
        np.testing.assert_array_equal(self.glow(saturation=1.0), base)
        vivid = (self.glow(saturation=2.0)[..., :3] - self.spot()[..., :3])[10, 9]
        plain = (base[..., :3] - self.spot()[..., :3])[10, 9]
        self.assertGreater(float(vivid.max() - vivid.min()), float(plain.max() - plain.min()))

    def test_effect_only_passes_the_glow_without_the_input(self):
        image = self.spot()
        image[..., :3] += 0.2                                                                    # a plate the glow would be added to
        added = self.glow(image)
        alone = self.glow(image, effect_only=1)
        glow = added[..., :3] - image[..., :3]
        np.testing.assert_allclose(alone[..., :3], glow, atol=1e-6)
        np.testing.assert_array_equal(alone[..., 3], image[..., 3])
        self.assertEqual(float(alone[0, 0, 0]), float(alone[0, 1, 0]))

    def test_nonlinear_blurs_the_encoded_highlight(self):
        image = self.spot()
        got = self.glow(image, nonlinear=1)[..., :3] - image[..., :3]
        bright = np.clip(image[..., :3], 0.0, None)
        encoded = blurknobs.filtered(mergeknobs.to_srgb(bright), 3.0, "box", 1000.0, (0, 0))
        np.testing.assert_allclose(got, mergeknobs.from_srgb(encoded), atol=1e-5)
        self.assertFalse(np.allclose(got, self.glow(image)[..., :3] - image[..., :3]))

    def test_the_filter_shapes_and_quality_limit_follow_blur(self):
        shapes = {name: self.glow(kernel_filter=name) for name in blurknobs.FILTERS}
        for name, out in shapes.items():
            expected = blurknobs.filtered(np.clip(self.spot()[..., :3], 0.0, None), 3.0, name, 1000.0, (0, 0))
            np.testing.assert_allclose(out[..., :3] - self.spot()[..., :3], expected, atol=1e-6, err_msg=name)
        self.assertFalse(np.allclose(shapes["box"], shapes["gaussian"]))
        limited = self.glow(glow_size=9.0, quality=3.0)
        free = self.glow(glow_size=9.0, quality=1000.0)
        self.assertFalse(np.allclose(limited, free))
        self.assertTrue(np.isfinite(limited).all())

    def test_crop_off_lets_the_window_grow_by_the_reach(self):
        g = Graph()
        g.add("src", "Rectangle", dict(width=N, height=N, box_x=6.0, box_y=6.0, box_width=3.0, box_height=3.0, red=2.0, green=2.0,
                                       blue=2.0, alpha=1.0))
        g.add("glow", "Glow", dict(glow_threshold=0.0, glow_size=4.0, crop=0, mix=1.0), image="src")
        grown = render(g, "glow")
        self.assertEqual((grown.data.width, grown.data.height), (N + 8, N + 8))
        g.d.execute({"op": "set", "id": "glow", "param": "crop", "value": 1})
        self.assertEqual(render(g, "glow").data.width, N)

    def test_the_width_channel_scales_the_size_per_pixel(self):
        image = self.spot()
        everywhere = np.ones((21, 21, 1), np.float32)
        np.testing.assert_allclose(Evaluator._glow(image, {**self.P}, (0, 0), everywhere), self.glow(image), atol=1e-6)
        nowhere = Evaluator._glow(image, {**self.P}, (0, 0), np.zeros((21, 21, 1), np.float32))
        np.testing.assert_allclose(nowhere, self.glow(image, glow_size=0.0), atol=1e-6)              # the highlight, with no spread
        half = Evaluator._glow(image, {**self.P}, (0, 0), np.full((21, 21, 1), 0.5, np.float32))
        np.testing.assert_allclose(half, self.glow(image, glow_size=1.5), atol=1e-5)                   # level 4 of 8 is exactly half the size
        left = np.zeros((21, 21, 1), np.float32)
        left[:, :10] = 1.0
        mixed = Evaluator._glow(image, {**self.P}, (0, 0), left)
        np.testing.assert_allclose(mixed[:, :10], self.glow(image)[:, :10], atol=1e-6)
        np.testing.assert_allclose(mixed[:, 10:], nowhere[:, 10:], atol=1e-6)
        self.assertEqual(glowknobs.LEVELS, 8)

    def test_the_width_channel_reads_a_channel_of_the_input(self):
        g = Graph()
        g.add("src", "Rectangle", dict(width=N, height=N, box_x=7.0, box_y=7.0, box_width=2.0, box_height=2.0, red=2.0, green=2.0,
                                       blue=2.0, alpha=1.0))
        g.add("glow", "Glow", dict(glow_threshold=0.0, glow_size=3.0, kernel_filter="box", quality=1000.0), image="src")
        wide = pixels(g, "glow")
        g.d.execute({"op": "set", "id": "glow", "param": "width_channel", "value": "rgba.green"})
        np.testing.assert_allclose(pixels(g, "glow")[3, 3], wide[3, 3], atol=1e-5)                    # where the plate is 0 the channel is 0
        g.d.execute({"op": "set", "id": "glow", "param": "width_channel", "value": "nonsense.layer"})
        with self.assertRaisesRegex(ValueError, "No channel"):
            pixels(g, "glow")

    def test_the_tiles_agree_with_the_new_knobs(self):
        for params in (dict(saturation=0.3, effect_only=1), dict(nonlinear=1, kernel_filter="triangle"),
                       dict(glow_size=9.0, quality=3.0, kernel_filter="quadratic"), dict(width_channel="rgba.red")):
            with self.subTest(params=params):
                g = Graph()
                g.add("src", "Checker", dict(width=N, height=N, size=3))
                g.add("glow", "Glow", {**dict(glow_threshold=0.1, glow_size=4.0), **params}, image="src")
                np.testing.assert_allclose(tiled(g, "glow", tile_edge=4), pixels(g, "glow"), atol=1e-5)

    def test_an_older_glow_keeps_a_box_blur_and_a_new_one_follows_nuke(self):
        g = Graph()
        g.add("src", "Checker", dict(width=N, height=N, size=3))
        g.add("glow", "Glow", dict(glow_threshold=0.0, glow_size=4.0), image="src")
        doc = copy.deepcopy(g.doc)
        for name in ("saturation", "width_channel", "nonlinear", "effect_only", "kernel_filter", "quality", "crop"):
            del doc["nodes"]["glow"]["params"][name]
        loaded = upgrade_document(doc)
        params = loaded["nodes"]["glow"]["params"]
        self.assertEqual((params["kernel_filter"], params["quality"], params["crop"]), ("box", blurknobs.NO_QUALITY_LIMIT, 1))
        expected = Evaluator._glow(pixels(g, "src"), {"glow_threshold": 0.0, "glow_size": 4.0, "brightness": 1.0, "channels": "rgb"})
        np.testing.assert_array_equal(Evaluator().evaluate_raster(dict(loaded, view="glow"), frame=1).pixels, expected)
        new = SPECS["Glow"]["params"]
        self.assertEqual((new["kernel_filter"], new["quality"], new["crop"], new["saturation"], new["effect_only"]), ("gaussian", 15.0, 1, 1.0, 0))
        g.bypass("glow")
        np.testing.assert_array_equal(pixels(g, "glow"), pixels(g, "src"))


# ================================================================================================== Median
class MedianTests(unittest.TestCase):
    def rows(self):
        image = np.zeros((9, 5, 4), np.float32)
        image[3:5] = 1.0                       # rows 3 and 4 are 1, the rest 0
        image[..., 3] = 1.0
        return image

    def test_ignore_top_line_leaves_the_top_row_of_the_window_out(self):
        image = self.rows()
        plain = Evaluator._median(image, {"median_size": 1.0, "channels": "rgb"})
        ignoring = Evaluator._median(image, {"median_size": 1.0, "channels": "rgb", "ignore_top_line": 1})
        # Pixel row 4: the window is rows 3 to 5, values 1, 1, 0; without the top row (row 3) it is 1, 0 and the median is 0.5.
        self.assertEqual(float(plain[4, 2, 0]), 1.0)
        self.assertEqual(float(ignoring[4, 2, 0]), 0.5)
        # Pixel row 6: rows 5 to 7 all 0 either way.
        self.assertEqual(float(ignoring[6, 2, 0]), 0.0)
        np.testing.assert_array_equal(Evaluator._median(image, {"median_size": 1.0, "channels": "rgb", "ignore_top_line": 0}), plain)

    def test_it_is_a_graph_knob_and_the_tiles_agree(self):
        g = Graph()
        g.add("src", "Checker", dict(width=N, height=N, size=3))
        g.add("m", "Median", dict(median_size=2.0, ignore_top_line=1), image="src")
        np.testing.assert_allclose(tiled(g, "m", tile_edge=5), pixels(g, "m"), atol=1e-6)
        plain = Graph()
        plain.add("src", "Checker", dict(width=N, height=N, size=3))
        plain.add("m", "Median", dict(median_size=2.0), image="src")
        self.assertFalse(np.array_equal(pixels(g, "m"), pixels(plain, "m")))

    def test_an_older_median_draws_the_same(self):
        g = Graph()
        g.add("src", "Checker", dict(width=N, height=N, size=3))
        g.add("m", "Median", dict(median_size=1.0), image="src")
        before = pixels(g, "m")
        doc = copy.deepcopy(g.doc)
        del doc["nodes"]["m"]["params"]["ignore_top_line"]
        loaded = upgrade_document(doc)
        self.assertEqual(loaded["nodes"]["m"]["params"]["ignore_top_line"], 0)
        np.testing.assert_array_equal(Evaluator().evaluate_raster(dict(loaded, view="m"), frame=1).pixels, before)


# ================================================================================================== Shuffle
class ShuffleTests(unittest.TestCase):
    def graph(self, **params):
        g = Graph()
        constant(g, "src", (0.2, 0.4, 0.6, 1.0))
        g.add("sh", "Shuffle", params, image="src")
        return g

    def test_the_second_output_layer_has_its_own_routing(self):
        g = self.graph(out2="swapped", red_from2="B", green_from2="G", blue_from2="R", alpha_from2="1")
        out = render(g, "sh")
        np.testing.assert_allclose(out.pixels[3, 3], [0.2, 0.4, 0.6, 1.0], atol=1e-6)               # out1 is the plain routing
        np.testing.assert_allclose(out.layers["swapped"].pixels[3, 3], [0.6, 0.4, 0.2, 1.0], atol=1e-6)

    def test_the_layer_can_be_read_by_a_later_shuffle(self):
        g = self.graph(out2="swapped", red_from2="B", green_from2="R", blue_from2="G", alpha_from2="A")
        g.add("again", "Shuffle", dict(layer="swapped"), image="sh")
        np.testing.assert_allclose(pixels(g, "again")[3, 3], [0.6, 0.2, 0.4, 1.0], atol=1e-6)

    def test_none_adds_no_layer_and_rgba_is_refused(self):
        self.assertFalse(render(self.graph(), "sh").layers)
        self.assertFalse(render(self.graph(out2="none", red_from2="B"), "sh").layers)
        with self.assertRaisesRegex(ValueError, "other than rgba"):
            render(self.graph(out2="rgba"), "sh")

    def test_a_second_layer_renders_whole_and_tiles_agree_otherwise(self):
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=4)
        plain = self.graph()
        self.assertTrue(executor.supports_tiled(dict(plain.doc, view="sh"), "sh"))
        np.testing.assert_allclose(tiled(plain, "sh"), pixels(plain, "sh"), atol=1e-6)
        with_layer = self.graph(out2="more")
        self.assertFalse(executor.supports_tiled(dict(with_layer.doc, view="sh"), "sh"))
        result = executor.compose(dict(with_layer.doc, view="sh"), "sh", frame=1, tier=1)
        np.testing.assert_allclose(result.pixels, pixels(with_layer, "sh"), atol=1e-6)

    def test_an_older_shuffle_draws_the_same_and_bypass_passes_the_input(self):
        g = self.graph(red_from="B", blue_from="R")
        before = pixels(g, "sh")
        doc = copy.deepcopy(g.doc)
        for name in ("out2", "red_from2", "green_from2", "blue_from2", "alpha_from2"):
            del doc["nodes"]["sh"]["params"][name]
        loaded = upgrade_document(doc)
        np.testing.assert_array_equal(Evaluator().evaluate_raster(dict(loaded, view="sh"), frame=1).pixels, before)
        validate(loaded)
        g.bypass("sh")
        np.testing.assert_array_equal(pixels(g, "sh"), pixels(g, "src"))


# =================================================================================================== STMap
class STMapTests(unittest.TestCase):
    W = H = 8

    bars = False

    def source(self):
        """A horizontal ramp: pixel x holds (x + 0.5) / 8 in red and green, 0.25 in blue, alpha 1; with `bars` on, alternate
        columns are 0 and 1 in red and green."""
        image = np.zeros((self.H, self.W, 4), np.float32)
        ramp = (np.arange(self.W, dtype=np.float32) + 0.5) / self.W
        if self.bars:
            ramp = (np.arange(self.W) % 2).astype(np.float32)
        image[..., 0] = ramp[None, :]
        image[..., 1] = ramp[None, :]
        image[..., 2] = 0.25
        image[..., 3] = 1.0
        return Raster.of(image)

    def identity_map(self, alpha=1.0):
        image = np.zeros((self.H, self.W, 4), np.float32)
        image[..., 0] = ((np.arange(self.W, dtype=np.float32) + 0.5) / self.W)[None, :]
        image[..., 1] = 1.0 - ((np.arange(self.H, dtype=np.float32) + 0.5) / self.H)[:, None]
        image[..., 3] = alpha
        return image

    def warp(self, uv, mask=None, **extra):
        p = {**SPECS["STMap"]["params"], **extra}
        inputs = [self.source(), Raster.of(uv)] + ([mask] if mask is not None else [])
        return Evaluator._uv_node("STMap", p, inputs).pixels

    def test_the_identity_map_leaves_the_picture(self):
        np.testing.assert_allclose(self.warp(self.identity_map()), self.source().pixels, atol=1e-5)

    def test_premultiplied_divides_the_map_by_its_alpha(self):
        half = self.identity_map(0.5)
        half[..., :2] *= 0.5                                        # a premultiplied half-alpha identity map
        np.testing.assert_allclose(self.warp(half, premultiplied=1), self.source().pixels, atol=1e-5)
        self.assertFalse(np.allclose(self.warp(half), self.source().pixels, atol=1e-3))

    def test_blur_channel_blurs_by_a_per_pixel_radius(self):
        self.bars = True                                            # a ramp is unchanged by a box blur, so use alternate columns
        uv = self.identity_map()
        uv[..., 2] = 0.0
        uv[:, 4:, 2] = 2.0                                          # radius 2 (times blur_scale 1) on the right half only
        got = self.warp(uv, blur_channel="rgba.blue")
        base = self.source().pixels
        np.testing.assert_allclose(got[:, :4], base[:, :4], atol=1e-5)
        expected = keymixknobs.variable_box_blur(base, np.where(np.arange(self.W)[None, :] >= 4, 2.0, 0.0) * np.ones((self.H, 1)))
        np.testing.assert_allclose(got, expected, atol=1e-5)
        self.assertFalse(np.allclose(got[:, 5], base[:, 5], atol=1e-4))
        scaled = self.warp(uv, blur_channel="rgba.blue", blur_scale=0.5)
        np.testing.assert_allclose(scaled[:, 5], keymixknobs.variable_box_blur(base, np.full((self.H, self.W), 1.0))[:, 5], atol=1e-5)

    def test_the_blur_is_a_window_mean_inside_the_picture(self):
        line = np.zeros((1, 9, 4), np.float32)
        line[0, 4] = 1.0
        got = keymixknobs.variable_box_blur(line, np.full((1, 9), 1.0))
        np.testing.assert_allclose(got[0, 3:6, 0], [1 / 3, 1 / 3, 1 / 3], atol=1e-6)
        np.testing.assert_allclose(got[0, 0, 0], 0.0, atol=1e-6)
        edge = np.zeros((1, 4, 4), np.float32)
        edge[0, 0] = 1.0
        self.assertAlmostEqual(float(keymixknobs.variable_box_blur(edge, np.full((1, 4), 1.0))[0, 0, 0]), 0.5, places=6)  # two pixels exist
        self.assertIs(keymixknobs.variable_box_blur(line, np.zeros((1, 9))), line)

    def test_channels_limit_what_the_map_replaces(self):
        uv = self.identity_map()
        uv[..., 0] = 0.5 / self.W                                   # every pixel reads the first column
        full = self.warp(uv)
        red_only = self.warp(uv, channels="red")
        base = self.source().pixels
        np.testing.assert_allclose(red_only[..., 0], full[..., 0], atol=1e-6)
        np.testing.assert_allclose(red_only[..., 1:], base[..., 1:], atol=1e-6)
        with self.assertRaisesRegex(ValueError, "layer"):
            self.warp(uv, channels="rgb, depth.Z")

    def test_the_mask_channel_and_its_invert(self):
        uv = self.identity_map()
        uv[..., 0] = 0.5 / self.W                                   # a map that changes the picture everywhere
        mask = np.zeros((self.H, self.W, 4), np.float32)
        mask[:, :4, 3] = 1.0                                        # alpha: left half
        mask[:, :, 0] = 0.0
        mask[:, 4:, 0] = 1.0                                        # red: right half
        full, base = self.warp(uv), self.source().pixels
        by_alpha = self.warp(uv, Raster.of(mask))
        np.testing.assert_allclose(by_alpha[:, :4], full[:, :4], atol=1e-6)
        np.testing.assert_allclose(by_alpha[:, 4:], base[:, 4:], atol=1e-6)
        by_red = self.warp(uv, Raster.of(mask), maskChannelInput="rgba.red")
        np.testing.assert_allclose(by_red[:, 4:], full[:, 4:], atol=1e-6)
        np.testing.assert_allclose(by_red[:, :4], base[:, :4], atol=1e-6)
        inverted = self.warp(uv, Raster.of(mask), invert_mask=1)
        np.testing.assert_allclose(inverted[:, :4], base[:, :4], atol=1e-6)
        np.testing.assert_allclose(inverted[:, 4:], full[:, 4:], atol=1e-6)

    def test_an_older_stmap_draws_the_same(self):
        g = Graph()
        g.add("src", "Checker", dict(width=N, height=N, size=3))
        g.add("uv", "Ramp", dict(width=N, height=N))
        g.add("st", "STMap", dict(uv_layer=""), image="src", uv="uv")
        before = pixels(g, "st")
        doc = copy.deepcopy(g.doc)
        for name in keymixknobs.STMAP:
            if name not in ("channels", "maskChannelInput", "invert_mask", "premultiplied"):
                del doc["nodes"]["st"]["params"][name]
        loaded = upgrade_document(doc)
        np.testing.assert_array_equal(Evaluator().evaluate_raster(dict(loaded, view="st"), frame=1).pixels, before)
        g.bypass("st")
        np.testing.assert_array_equal(pixels(g, "st"), pixels(g, "src"))


# ================================================================================================== KeyMix
class KeyMixTests(unittest.TestCase):
    def graph(self, mask=True, **params):
        g = Graph()
        constant(g, "a", (1.0, 0.0, 0.0, 0.5))
        constant(g, "b", (0.0, 0.0, 1.0, 1.0))
        g.add("km", "Keymix", params, A="a", B="b")
        if mask:
            g.add("m", "Rectangle", dict(width=N, height=N, box_x=0.0, box_y=0.0, box_width=8.0, box_height=float(N),
                                         red=0.0, green=0.0, blue=0.0, alpha=1.0))
            g.d.execute({"op": "connect", "id": "km", "input": "mask", "source": "m"})
        return g

    def test_the_default_mixes_a_over_b_by_the_masks_alpha(self):
        out = pixels(self.graph(), "km")
        a, b = pixels(self.graph(), "a")[0, 0], pixels(self.graph(), "b")[0, 0]
        np.testing.assert_allclose(out[5, 3], a, atol=1e-6)
        np.testing.assert_allclose(out[5, 12], b, atol=1e-6)

    def test_the_mask_channel_chooses_the_matte(self):
        by_red = pixels(self.graph(maskChannelInput="rgba.red"), "km")                           # the mask's red is 0 everywhere
        np.testing.assert_allclose(by_red[5, 3], pixels(self.graph(), "b")[0, 0], atol=1e-6)
        by_lum = pixels(self.graph(maskChannelInput="luminance"), "km")
        np.testing.assert_allclose(by_lum[5, 3], pixels(self.graph(), "b")[0, 0], atol=1e-6)
        alpha_word = pixels(self.graph(maskChannelInput="alpha"), "km")
        np.testing.assert_array_equal(alpha_word, pixels(self.graph(), "km"))

    def test_with_no_mask_the_channel_is_read_from_a(self):
        none = pixels(self.graph(mask=False), "km")
        np.testing.assert_allclose(none[5, 3], pixels(self.graph(), "a")[0, 0], atol=1e-6)         # the alpha default leaves A whole
        red = pixels(self.graph(mask=False, maskChannelInput="rgba.red"), "km")                    # A's red (0.5 after the alpha) as the matte
        a, b = pixels(self.graph(), "a")[0, 0], pixels(self.graph(), "b")[0, 0]
        np.testing.assert_allclose(red[5, 3], a * a[0] + b * (1 - a[0]), atol=1e-6)

    def test_invert_still_turns_the_matte_over(self):
        inverted = pixels(self.graph(invert_mask=1), "km")
        np.testing.assert_allclose(inverted[5, 12], pixels(self.graph(), "a")[0, 0], atol=1e-6)
        np.testing.assert_allclose(inverted[5, 3], pixels(self.graph(), "b")[0, 0], atol=1e-6)

    def test_channels_keep_the_others_from_b(self):
        a, b = pixels(self.graph(), "a")[0, 0], pixels(self.graph(), "b")[0, 0]
        out = pixels(self.graph(channels="rgb"), "km")[5, 3]
        np.testing.assert_allclose(out[:3], a[:3], atol=1e-6)
        self.assertAlmostEqual(float(out[3]), float(b[3]), places=6)                               # alpha is B's
        red = pixels(self.graph(channels="red"), "km")[5, 3]
        np.testing.assert_allclose(red, [a[0], b[1], b[2], b[3]], atol=1e-6)
        with self.assertRaisesRegex(ValueError, "layer"):
            pixels(self.graph(channels="rgb, depth.Z"), "km")

    def test_bbox_sets_the_data_window_of_the_result(self):
        g = Graph()
        constant(g, "a0", (1.0, 0.0, 0.0, 1.0))
        g.add("a", "Crop", dict(x=2, y=2, width=8, height=8), image="a0")
        constant(g, "b0", (0.0, 0.0, 1.0, 1.0))
        g.add("b", "Crop", dict(x=6, y=6, width=8, height=8), image="b0")
        g.add("km", "Keymix", {}, A="a", B="b")
        window = lambda: render(g, "km").data
        self.assertEqual((window().x, window().y, window().width, window().height), (2, 2, 12, 12))      # the union
        for choice, expected in (("intersection", (6, 6, 4, 4)), ("A", (2, 2, 8, 8)), ("B", (6, 6, 8, 8))):
            g.d.execute({"op": "set", "id": "km", "param": "merge_bbox", "value": choice})
            data = window()
            self.assertEqual((data.x, data.y, data.width, data.height), expected, choice)

    def test_the_tiles_agree_for_the_channel_knobs(self):
        for params in (dict(), dict(maskChannelInput="rgba.red"), dict(channels="rgb", invert_mask=1, mix=0.7),
                       dict(maskChannelInput="luminance")):
            with self.subTest(params=params):
                g = self.graph(**params)
                np.testing.assert_allclose(tiled(g, "km", tile_edge=5), pixels(g, "km"), atol=1e-6)

    def test_an_older_keymix_draws_the_same_and_bypass_passes_b(self):
        g = self.graph(invert_mask=1)
        before = pixels(g, "km")
        doc = copy.deepcopy(g.doc)
        for name in keymixknobs.KEYMIX:
            del doc["nodes"]["km"]["params"][name]
        loaded = upgrade_document(doc)
        self.assertEqual(loaded["nodes"]["km"]["params"]["maskChannelInput"], "rgba.alpha")
        np.testing.assert_array_equal(Evaluator().evaluate_raster(dict(loaded, view="km"), frame=1).pixels, before)
        g.bypass("km")
        np.testing.assert_array_equal(pixels(g, "km"), pixels(g, "b"))


if __name__ == "__main__":
    unittest.main()
