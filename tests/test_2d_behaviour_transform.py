"""2D behaviour, lane 9 step B: Transform, Reformat, CornerPin, Crop, Mirror, Position, Tile, AdjustBBox and BlackOutside compute what the
Nuke 17.0 Reference Guide says they compute.

As in step A: the behaviour cases pass where the guide (or a published form) fixes the pixels, every main-tab knob NodeBased pairs with a
Nuke knob changes pixels, the kernels this step changed agree between the evaluator and the tile path (Transform, Crop and Mirror are
the precedented full-frame exclusions), a document saved before the change renders bit for bit as it did, and each node's main tab
lists its knobs in Nuke's order.
"""
import copy
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from nodebased import core, knobs, transformbehaviour
from nodebased.core import SPECS, TRANSFORM_FILTERS, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.raster import Raster
from tests.test_2d_parity_group_c4 import Graph as KeyGraph
from tests.test_2d_behaviour_filter_a import differs, impulse, kern, old_document, render, texture

import audit_nuke_fidelity as audit  # noqa: E402
import nuke_fidelity_cases as ruler  # noqa: E402

STEP_B_NODES = ("Transform", "Reformat", "CornerPin2D", "Crop", "Mirror", "Position", "Tile", "AdjustBBox", "BlackOutside")
DONE_NODES = STEP_B_NODES


def labels(kind):
    """The labels of a node's main tab in the order the panel lays them out (the mask/mix strip and the legacy switches left off)."""
    out = []
    for group in knobs.KNOB_LAYOUT[kind]:
        if group.kind == "legacy" or group.params[0].startswith("legacy_") or group.params == ("mix",):
            continue
        out.append(group.label)
    return out


# ------------------------------------------------------------------------------------------------------------- the cases
class CaseTests(unittest.TestCase):
    def test_every_scored_case_of_the_step_passes(self):
        scored = [c for c in ruler.CASES if c.nuke in DONE_NODES and not c.unscored]
        self.assertGreaterEqual(len(scored), 49)
        for case in scored:
            with self.subTest(case=case.id):
                result = audit.run_case(case)
                self.assertEqual(result["status"], "pass", (case.id, result["max_abs_error"], result["worst"], case.statement))

    def test_every_unscored_case_states_its_reason(self):
        unscored = [c for c in ruler.CASES if c.nuke in DONE_NODES and c.unscored]
        self.assertGreaterEqual(len(unscored), 9)
        for case in unscored:
            with self.subTest(case=case.id):
                self.assertGreater(len(case.unscored), 40, case.id)
                self.assertEqual(audit.run_case(case)["status"], "not scored")

    def test_every_node_of_the_step_has_a_case(self):
        for node in DONE_NODES:
            with self.subTest(node=node):
                self.assertTrue(ruler.cases_for(node), node)


# --------------------------------------------------------------------------------------------------------------- Transform
def transform(pixels, **knobs_):
    return kern("Transform", pixels, **{"filter": "impulse", **knobs_})


def step_picture(n=24):
    """A picture that is 0 on the left half and 1 on the right half, every channel."""
    pixels = np.zeros((n, n, 4), np.float32)
    pixels[:, n // 2:] = 1.0
    return pixels


class TransformFilterTests(unittest.TestCase):
    NEW_FILTERS = ("impulse", "keys", "simon", "rifman", "mitchell", "parzen", "lanczos4", "lanczos6", "sinc4")

    def test_the_filter_list_holds_the_guides_filters_apart_from_notch(self):
        for name in ("nearest", "bilinear", "cubic", *self.NEW_FILTERS):
            self.assertIn(name, TRANSFORM_FILTERS)
        self.assertNotIn("notch", TRANSFORM_FILTERS)             # the guide draws no curve for it; see the Notch case

    def test_impulse_carries_the_original_values(self):
        pixels = texture()
        out = transform(pixels, scale=1.7, center_x=20.0, center_y=18.0)
        values = {float(v) for v in np.unique(pixels)}
        inside = out[..., 3] > 0
        self.assertTrue(inside.any())
        for channel in range(4):
            self.assertTrue(all(float(v) in values for v in np.unique(out[..., channel][inside])), channel)

    def test_impulse_is_nearest(self):
        pixels = texture()
        for knobs_ in (dict(translate_x=2.4, translate_y=-1.3), dict(rotate=17.0, center_x=20.0, center_y=18.0), dict(scale=0.6)):
            np.testing.assert_array_equal(kern("Transform", pixels, filter="impulse", **knobs_),
                                          kern("Transform", pixels, filter="nearest", **knobs_))

    def test_every_filter_reproduces_a_constant_picture(self):
        flat = np.full((20, 24, 4), 0.37, np.float32)
        for name in self.NEW_FILTERS + ("bilinear", "cubic"):
            with self.subTest(filter=name):
                out = kern("Transform", flat, filter=name, translate_x=3.3, translate_y=1.6, rotate=11.0, scale=1.2,
                           center_x=12.0, center_y=10.0, black_outside=0)
                np.testing.assert_allclose(out, flat, atol=2e-6)

    def test_the_identity_transform_is_exact_for_every_filter(self):
        pixels = texture()
        for name in TRANSFORM_FILTERS:
            with self.subTest(filter=name):
                np.testing.assert_array_equal(kern("Transform", pixels, filter=name), pixels)

    def test_smoothing_and_sharpening_follow_the_guides_order(self):
        step = step_picture()

        def overshoot(name):
            out = kern("Transform", step, filter=name, translate_x=0.5, black_outside=0)[10, :, 0]
            return float(max(out.max() - 1.0, -out.min(), 0.0))
        self.assertEqual(overshoot("cubic"), overshoot("keys"))                 # NodeBased's cubic is the Catmull-Rom spline
        self.assertGreater(overshoot("keys"), 0.0)
        self.assertGreater(overshoot("simon"), overshoot("keys") + 1e-4)
        self.assertGreater(overshoot("rifman"), overshoot("simon") + 1e-4)
        self.assertAlmostEqual(overshoot("mitchell") * 0 + overshoot("parzen"), 0.0, places=6)    # the B-spline never overshoots
        self.assertLess(overshoot("lanczos4"), overshoot("sinc4") + 1e-9)

    def test_parzen_smooths_most_and_mitchell_blurs(self):
        dot = impulse(21)
        peak = {name: float(kern("Transform", dot, filter=name, translate_x=0.5, black_outside=0)[..., 0].max())
                for name in ("impulse", "cubic", "keys", "mitchell", "parzen", "lanczos4", "sinc4")}
        self.assertEqual(peak["impulse"], 1.0)
        self.assertLess(peak["parzen"], peak["mitchell"])
        self.assertLess(peak["mitchell"], peak["cubic"])
        self.assertGreater(peak["cubic"], peak["parzen"])
        self.assertEqual(min(peak, key=peak.get), "parzen")

    def test_clamp_keeps_a_sharpening_filter_inside_the_range_of_the_input(self):
        step = step_picture()
        free = kern("Transform", step, filter="rifman", translate_x=0.5, black_outside=0)
        held = kern("Transform", step, filter="rifman", translate_x=0.5, black_outside=0, clamp=1)
        self.assertGreater(float(free.max()), 1.0 + 1e-3)
        self.assertLess(float(free.min()), -1e-3)
        self.assertLessEqual(float(held.max()), 1.0)
        self.assertGreaterEqual(float(held.min()), 0.0)
        np.testing.assert_allclose(held, np.clip(free, 0.0, 1.0), atol=1e-6)

    def test_every_new_filter_gives_the_same_pixels_on_the_planned_path(self):
        pixels = texture(40, 48)
        for name in self.NEW_FILTERS:
            with self.subTest(filter=name):
                out = kern("Transform", pixels, filter=name, rotate=9.0, scale=1.1, center_x=24.0, center_y=20.0)
                self.assertTrue(np.isfinite(out).all())
                self.assertEqual(out.shape, pixels.shape)


class TransformMatrixTests(unittest.TestCase):
    def test_translate_moves_right_and_up_the_screen(self):
        dot = impulse(11)                                       # the dot at row 5, column 5
        out = transform(dot, translate_x=2.0, translate_y=3.0)
        self.assertEqual(tuple(np.argwhere(out[..., 0] > 0.5)[0]), (2, 7))

    def test_rotate_is_counter_clockwise_on_the_screen(self):
        pixels = np.zeros((11, 11, 4), np.float32)
        pixels[5, 8] = 1.0                                      # right of the centre pixel
        out = transform(pixels, rotate=90.0, center_x=5.5, center_y=5.5)
        self.assertEqual(tuple(np.argwhere(out[..., 0] > 0.5)[0]), (2, 5))     # above it

    def test_scale_about_the_center_leaves_the_center_in_place(self):
        pixels = np.zeros((20, 20, 4), np.float32)
        pixels[9:11, 9:11] = 1.0                                # the four pixels around the point (10, 10)
        out = transform(pixels, scale=3.0, center_x=10.0, center_y=10.0)
        box = np.argwhere(out[..., 0] > 0.5)
        self.assertEqual((box.min(axis=0).tolist(), box.max(axis=0).tolist()), ([7, 7], [12, 12]))

    def test_the_matrix_applies_scale_then_skew_then_rotate_then_translate(self):
        p = {**SPECS["Transform"]["params"], "scale_mode": "xy", "scale_x": 2.0, "scale_y": 0.5, "skew_x": 0.3, "skew_y": 0.1,
             "rotate": 25.0, "translate_x": 4.0, "translate_y": -3.0, "center_x": 6.0, "center_y": 5.0}
        forward = Evaluator._transform_forward_matrix(p)
        theta = np.radians(-25.0)                              # the screen's counter-clockwise is the rows' negative angle
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        skew_x = np.array([[1.0, -0.3], [0.0, 1.0]])
        skew_y = np.array([[1.0, 0.0], [-0.1, 1.0]])
        linear = rotation @ (skew_y @ skew_x) @ np.diag([2.0, 0.5])
        point = np.array([9.0, 2.0])
        expected = np.array([6.0 + 4.0, 5.0 + 3.0]) + linear @ (point - np.array([6.0, 5.0]))
        np.testing.assert_allclose(forward @ np.array([*point, 1.0]), np.array([*expected, 1.0]), atol=1e-9)

    def test_skew_order_and_the_two_skews_act_differently(self):
        pixels = texture()
        base = dict(skew_x=0.3, skew_y=0.2, center_x=20.0, center_y=18.0)
        xy = transform(pixels, skew_order="XY", **base)
        yx = transform(pixels, skew_order="YX", **base)
        self.assertTrue(differs(xy, yx))
        self.assertTrue(differs(transform(pixels, skew_x=0.3, center_x=20.0, center_y=18.0),
                                transform(pixels, skew_y=0.3, center_x=20.0, center_y=18.0)))

    def test_invert_inverts_the_whole_transform(self):
        pixels = texture()
        moved = transform(pixels, translate_x=3.0, translate_y=-2.0)
        back = transform(moved, translate_x=3.0, translate_y=-2.0, invert=1)
        np.testing.assert_array_equal(back[2:-2, 3:-3], pixels[2:-2, 3:-3])
        np.testing.assert_array_equal(transform(pixels, rotate=30.0, scale=1.3, center_x=20.0, center_y=18.0, invert=1),
                                      kern("Transform", pixels, filter="impulse", rotate=30.0, scale=1.3, center_x=20.0, center_y=18.0,
                                           invert=1))
        forward = Evaluator._transform_forward_matrix({**SPECS["Transform"]["params"], "rotate": 30.0, "scale": 1.3, "translate_x": 5.0,
                                                       "center_x": 9.0, "center_y": 4.0})
        inverted = Evaluator._transform_forward_matrix({**SPECS["Transform"]["params"], "rotate": 30.0, "scale": 1.3, "translate_x": 5.0,
                                                        "center_x": 9.0, "center_y": 4.0, "invert": 1})
        np.testing.assert_allclose(forward @ inverted, np.eye(3), atol=1e-9)


class TransformEdgeTests(unittest.TestCase):
    def test_black_outside_is_on_by_default_and_renders_black(self):
        self.assertEqual(SPECS["Transform"]["params"]["black_outside"], 1)
        pixels = texture() + 0.2
        out = transform(pixels, translate_x=4.0)
        np.testing.assert_array_equal(out[:, :4, :3], np.zeros_like(out[:, :4, :3]))

    def test_unchecked_it_fills_the_outside_with_the_outermost_pixels(self):
        pixels = texture()
        out = transform(pixels, translate_x=4.0, black_outside=0)
        np.testing.assert_array_equal(out[:, :4], np.repeat(pixels[:, :1], 4, axis=1))
        np.testing.assert_array_equal(out[:, 4:], pixels[:, :-4])

    def test_it_carries_the_edge_for_a_rotation_too(self):
        pixels = texture() + 1.0
        out = transform(pixels, rotate=20.0, center_x=20.0, center_y=18.0, black_outside=0)
        self.assertGreater(float(out.min()), 0.0)               # nothing black anywhere in the frame

    def test_the_older_reading_paints_black_outside_opaque_and_clamp_extends_the_edge(self):
        pixels = texture() + 0.2
        on = kern("Transform", pixels, filter="impulse", translate_x=4.0, black_outside=1, legacy_transform_edges=1)
        np.testing.assert_array_equal(on[:, :4], np.broadcast_to(np.float32([0, 0, 0, 1]), on[:, :4].shape))
        off = kern("Transform", pixels, filter="impulse", translate_x=4.0, black_outside=0, legacy_transform_edges=1)
        np.testing.assert_array_equal(off[:, :4], np.zeros_like(off[:, :4]))
        extended = kern("Transform", pixels, filter="impulse", translate_x=4.0, clamp=1, black_outside=0, legacy_transform_edges=1)
        np.testing.assert_array_equal(extended[:, :4], np.repeat(pixels[:, :1], 4, axis=1))

    def test_the_older_axes_move_down_and_turn_clockwise(self):
        dot = impulse(11)
        out = kern("Transform", dot, filter="impulse", translate_y=3.0, legacy_transform_axes=1)
        self.assertEqual(tuple(np.argwhere(out[..., 0] > 0.5)[0]), (8, 5))
        pixels = np.zeros((11, 11, 4), np.float32)
        pixels[5, 8] = 1.0
        out = kern("Transform", pixels, filter="impulse", rotate=90.0, center_x=5.5, center_y=5.5, legacy_transform_axes=1)
        self.assertEqual(tuple(np.argwhere(out[..., 0] > 0.5)[0]), (8, 5))


class TransformShutterTests(unittest.TestCase):
    def test_the_shutter_windows_the_guide_gives(self):
        window = transformbehaviour.shutter_window
        self.assertEqual(window(30, 1.0, "centred", 0.0, False), (29.5, 30.5))
        self.assertEqual(window(30, 1.0, "start", 0.0, False), (30.0, 31.0))
        self.assertEqual(window(30, 1.0, "end", 0.0, False), (29.0, 30.0))
        self.assertEqual(window(30, 1.0, "custom", -0.5, False), (29.5, 30.5))
        self.assertEqual(window(30, 1.0, "start", 0.0, True), (29.0, 30.0))           # the older reading had start and end the other way round
        self.assertEqual(window(30, 1.0, "end", 0.0, True), (30.0, 31.0))
        self.assertEqual(window(30, 1.0, "custom", 2.0, True), (31.5, 32.5))

    def test_the_defaults_are_the_guides(self):
        params = SPECS["Transform"]["params"]
        self.assertEqual((params["shutter"], params["shutter_offset"], params["motionblur"]), (0.5, "start", 0))

    def moving(self, **params):
        g = KeyGraph()
        g.add("plate", "Checker", dict(width=40, height=30, size=4))
        g.add("node", "Transform", {"filter": "impulse", **params}, image="plate")
        g.key("node", "translate_x", 1, 0.0)
        g.key("node", "translate_x", 11, 10.0)
        return g

    def raster(self, g, frame):
        return Evaluator().evaluate_raster(dict(g.doc, view="node"), target="node", frame=frame)

    def test_motion_blur_averages_the_transform_over_the_shutter(self):
        g = self.moving(motionblur=1, samples=2, shutter=2.0, shutter_offset="start")
        blurred = self.raster(g, 5)
        sharp = self.moving()
        # samples=2 over frames 5 to 7 (start opens the shutter at the frame): sampled at 5.5 and 6.5, translate 4.5 and 5.5
        samples = [Evaluator().evaluate_raster(dict(sharp.doc, view="node"), target="node", frame=f) for f in (5.5, 6.5)]
        union = samples[0].data.union(samples[1].data)
        mean = np.mean(np.stack([s.fit(union) for s in samples]), axis=0)
        self.assertEqual(blurred.data, union)
        np.testing.assert_allclose(blurred.pixels, mean, atol=1e-5)

    def test_a_still_transform_is_not_blurred(self):
        g = KeyGraph()
        g.add("plate", "Checker", dict(width=40, height=30, size=4))
        g.add("node", "Transform", {"filter": "impulse", "translate_x": 3.0, "motionblur": 1, "samples": 4, "shutter": 2.0}, image="plate")
        g.add("ref", "Transform", {"filter": "impulse", "translate_x": 3.0}, image="plate")
        np.testing.assert_allclose(self.raster(g, 4).pixels, Evaluator().evaluate_raster(dict(g.doc, view="ref"), target="ref", frame=4).pixels,
                                   atol=1e-6)

    def test_start_end_and_centred_blur_different_frames(self):
        # The picture moves right with the frame, so a shutter that opens later sits further right.
        left = {offset: self.raster(self.moving(motionblur=1, samples=4, shutter=2.0, shutter_offset=offset), 6).data.x
                for offset in ("centred", "start", "end")}
        self.assertLess(left["end"], left["centred"])
        self.assertLess(left["centred"], left["start"])


# ---------------------------------------------------------------------------------------------------------------- the knobs
class KnobsActTests(unittest.TestCase):
    """Every Transform main-tab knob NodeBased pairs with a Nuke knob changes the pixels (the direction where the guide gives one)."""
    BASE = dict(filter="cubic", center_x=20.0, center_y=18.0)
    ACTS = (("translate_x", 2.5), ("translate_y", -1.5), ("rotate", 12.0), ("scale", 1.4), ("skew_x", 0.25), ("skew_y", 0.25),
            ("invert", 1), ("filter", "rifman"), ("clamp", 1), ("black_outside", 0), ("center_x", 5.0))

    def test_every_listed_knob_changes_the_pixels(self):
        pixels = texture(37, 45)
        shifted = dict(self.BASE, translate_x=3.5, rotate=7.0)
        for name, value in self.ACTS:
            base = dict(shifted)
            if name in ("translate_x", "translate_y", "rotate"):
                base.pop(name, None)
            if name == "black_outside":
                base["black_outside"] = 1
            if name == "clamp":
                base["filter"] = "rifman"
            with self.subTest(knob=name):
                self.assertIn(name, SPECS["Transform"]["params"])
                self.assertTrue(differs(kern("Transform", pixels, **base), kern("Transform", pixels, **{**base, name: value})), name)

    def test_skew_order_changes_the_pixels_with_both_skews_on(self):
        pixels = texture(37, 45)
        base = dict(self.BASE, skew_x=0.3, skew_y=0.2)
        self.assertTrue(differs(kern("Transform", pixels, **base, skew_order="XY"), kern("Transform", pixels, **base, skew_order="YX")))

    def test_scale_x_and_y_scale_one_axis_each(self):
        pixels = texture(37, 45)
        base = dict(self.BASE, scale_mode="xy")
        wide = kern("Transform", pixels, **base, scale_x=1.5, scale_y=1.0)
        tall = kern("Transform", pixels, **base, scale_x=1.0, scale_y=1.5)
        self.assertTrue(differs(wide, tall))

    def test_scale_splits_into_a_width_and_a_height(self):
        self.assertEqual(core.SPLITTABLE["Transform"], {"scale": "wh"})
        pixels = texture(37, 45)
        base = dict(self.BASE, filter="impulse")
        self.assertEqual(knobs.split_layout_problems(), [])
        wide = kern("Transform", pixels, **base, scale=[1.5, 1.0])
        self.assertTrue(differs(wide, kern("Transform", pixels, **base, scale=[1.0, 1.5])))
        # equal entries are the plain scale, and a list is the older per-axis scale
        np.testing.assert_array_equal(kern("Transform", pixels, **base, scale=[1.3, 1.3]), kern("Transform", pixels, **base, scale=1.3))
        np.testing.assert_array_equal(wide, kern("Transform", pixels, **base, scale_mode="xy", scale_x=1.5, scale_y=1.0))

    def test_every_main_tab_knob_is_laid_out_in_nuke_order(self):
        self.assertEqual(labels("Transform"), ["Translate", "Rotate", "Scale", "Older scale mode", "Older scale X, Y", "Skew X", "Skew Y", "Skew order",
                                               "Center", "Invert", "Filter", "Clamp", "Black outside", "Motion blur", "Shutter",
                                               "Shutter offset", "Samples"])   # the custom offset shares the Shutter offset row (panel sweep, 11/4)


# ----------------------------------------------------------------------------------------------------- an older document
class OlderDocumentTests(unittest.TestCase):
    LEGACY_FLAGS = ("legacy_transform_edges", "legacy_shutter_window", "legacy_transform_axes")

    def test_a_transform_saved_before_the_step_is_given_the_older_readings(self):
        loaded = old_document("Transform", dict(translate_x=3.0, translate_y=2.0, rotate=10.0, filter="cubic", black_outside=1),
                              self.LEGACY_FLAGS)
        params = loaded["nodes"]["n"]["params"]
        for flag in self.LEGACY_FLAGS:
            self.assertEqual(params[flag], 1, flag)

    def test_it_draws_the_same_pixels_as_the_older_kernel(self):
        knobs_ = dict(translate_x=3.0, translate_y=2.0, rotate=10.0, filter="cubic", black_outside=1, center_x=20.0, center_y=18.0)
        loaded = old_document("Transform", knobs_, self.LEGACY_FLAGS)
        pixels = render(loaded, "n")
        explicit = old_document("Transform", knobs_, ())
        for flag in self.LEGACY_FLAGS:
            explicit["nodes"]["n"]["params"][flag] = 1
        np.testing.assert_array_equal(pixels, render(explicit, "n"))
        again = copy.deepcopy(loaded)
        for flag in self.LEGACY_FLAGS:
            again["nodes"]["n"]["params"].pop(flag)                # a node that never carried the flags reads them as set
        np.testing.assert_array_equal(pixels, render(again, "n"))

    def test_the_older_reading_agrees_with_the_axes_of_tracker_and_stabilize(self):
        # Tracker and Stabilize share the matrix and carry no flag: they keep the older directions.
        p = {**SPECS["Transform"]["params"], "translate_y": 3.0, "rotate": 20.0}
        p.pop("legacy_transform_axes")
        np.testing.assert_allclose(Evaluator._transform_forward_matrix(p),
                                   Evaluator._transform_forward_matrix({**p, "legacy_transform_axes": 1}), atol=0)

    def test_a_new_transform_reads_the_new_way(self):
        new = SPECS["Transform"]["params"]
        for flag in self.LEGACY_FLAGS:
            self.assertEqual(new[flag], 0, flag)
        g = KeyGraph()
        g.add("plate", "Checker", dict(width=20, height=20, size=4))
        g.add("node", "Transform", {}, image="plate")
        validate(upgrade_document(copy.deepcopy(g.doc)))


# ---------------------------------------------------------------------------------------------------------------- Reformat
def windowed(kind, picture, **knobs_):
    """One windowed node on `picture` (data window = display window): the Raster it makes."""
    return Evaluator._windowed_kernel(kind, {**SPECS[kind]["params"], **knobs_}, [Raster.of(np.asarray(picture, np.float32))])


def reformat(picture, **knobs_):
    return windowed("Reformat", picture, **{"reformat_type": "to_box", "box_fixed": 1, "filter": "nearest", **knobs_})


class ReformatTests(unittest.TestCase):
    def test_force_this_shape_off_keeps_the_shape_of_the_input(self):
        pixels = texture(12, 16)
        for resize, box_, expected in (("width", (40, 40), (40, 30)), ("height", (40, 24), (32, 24)), ("fit", (40, 40), (40, 30)),
                                       ("fill", (40, 40), (53, 40)), ("distort", (40, 40), (40, 40)), ("none", (40, 40), (40, 40))):
            with self.subTest(resize=resize):
                out = reformat(pixels, width=box_[0], height=box_[1], resize_type=resize, box_fixed=0)
                self.assertEqual((out.display.width, out.display.height), expected)
                self.assertEqual(out.pixels.shape[:2], (expected[1], expected[0]))

    def test_force_this_shape_on_is_exactly_the_box(self):
        out = reformat(texture(12, 16), width=40, height=40, resize_type="width", box_fixed=1)
        self.assertEqual((out.display.width, out.display.height), (40, 40))

    def test_a_turned_box_is_solved_against_the_turned_picture(self):
        # box 30 wide, 60 tall: the working format is 60 wide, 30 tall, the 16 x 12 picture is 60 x 45 at resize width, turned 45 x 60.
        self.assertEqual(transformbehaviour.reformat_box_shape(16, 12, 30, 60, "width", True), (45, 60))
        out = reformat(texture(12, 16), width=30, height=60, resize_type="width", box_fixed=0, turn=1)
        self.assertEqual(out.pixels.shape[:2][::-1], (45, 60))

    def test_scale_type_rounds_to_whole_pixels(self):
        out = windowed("Reformat", texture(12, 16), reformat_type="scale", scale=0.7)
        self.assertEqual((out.display.width, out.display.height), (11, 8))

    def test_scale_splits_into_a_width_and_a_height(self):
        from nodebased import core
        self.assertEqual(core.SPLITTABLE["Reformat"], {"scale": "wh"})
        out = windowed("Reformat", texture(12, 16), reformat_type="scale", scale=[0.5, 0.25], filter="nearest")
        self.assertEqual((out.display.width, out.display.height), (8, 3))
        same = windowed("Reformat", texture(12, 16), reformat_type="scale", scale=[0.5, 0.5], filter="nearest")
        plain = windowed("Reformat", texture(12, 16), reformat_type="scale", scale=0.5, filter="nearest")
        np.testing.assert_array_equal(same.pixels, plain.pixels)
        self.assertEqual(knobs.split_layout_problems(), [])

    def test_center_off_lines_up_the_lower_left_corners(self):
        pixels = texture(12, 16) + 1.0
        out = reformat(pixels, width=32, height=48, resize_type="fit", center=0, legacy_reformat_layout=0).pixels
        self.assertEqual(float(out[:24, :, :3].max()), 0.0)           # the upper half is empty
        self.assertGreater(float(out[24:, :, :3].min()), 0.0)

    def test_turn_rotates_counter_clockwise_and_a_double_turn_equals_flip_and_flop(self):
        pixels = texture(12, 16)
        turned = reformat(pixels, width=12, height=16, resize_type="distort", turn=1, legacy_reformat_layout=0).pixels
        np.testing.assert_array_equal(turned, np.rot90(pixels, 1))
        both = reformat(pixels, width=16, height=12, resize_type="distort", flip=1, flop=1).pixels
        np.testing.assert_array_equal(both, np.rot90(pixels, 2))

    def test_flip_and_flop_after_a_turn(self):
        pixels = texture(12, 16)
        turned = np.rot90(pixels, 1)
        np.testing.assert_array_equal(reformat(pixels, width=12, height=16, resize_type="distort", turn=1, flip=1,
                                               legacy_reformat_layout=0).pixels, turned[::-1])
        np.testing.assert_array_equal(reformat(pixels, width=12, height=16, resize_type="distort", turn=1, flop=1,
                                               legacy_reformat_layout=0).pixels, turned[:, ::-1])

    def test_black_outside_off_fills_the_letterbox_with_the_outermost_pixels(self):
        pixels = texture(12, 16)
        black = reformat(pixels, width=32, height=48, resize_type="fit", black_outside=1).pixels
        edge = reformat(pixels, width=32, height=48, resize_type="fit", black_outside=0).pixels
        self.assertEqual(float(np.abs(black[:12, :, :3]).max()), 0.0)
        np.testing.assert_array_equal(edge[:12], np.repeat(edge[12:13], 12, axis=0))
        np.testing.assert_array_equal(edge[12:36], black[12:36])
        np.testing.assert_array_equal(edge[36:], np.repeat(edge[35:36], 12, axis=0))

    def test_clamp_keeps_a_sharpening_filter_inside_the_range_of_the_input(self):
        step = step_picture(16)
        free = reformat(step, width=24, height=24, resize_type="distort", filter="rifman").pixels
        held = reformat(step, width=24, height=24, resize_type="distort", filter="rifman", clamp=1).pixels
        self.assertGreater(float(free.max()), 1.0 + 1e-3)
        self.assertLessEqual(float(held.max()), 1.0)
        self.assertGreaterEqual(float(held.min()), 0.0)

    def test_every_filter_makes_a_picture_of_the_same_size_and_keeps_a_constant(self):
        flat = np.full((12, 16, 4), 0.37, np.float32)
        for name in TRANSFORM_FILTERS:
            with self.subTest(filter=name):
                out = reformat(flat, width=24, height=18, resize_type="distort", filter=name, black_outside=0).pixels
                self.assertEqual(out.shape, (18, 24, 4))
                np.testing.assert_allclose(out, 0.37, atol=2e-6)

    def test_the_defaults_follow_nuke(self):
        params = SPECS["Reformat"]["params"]
        self.assertEqual((params["filter"], params["box_fixed"], params["clamp"], params["black_outside"], params["center"]),
                         ("cubic", 0, 0, 1, 1))

    ACTS = (("reformat_type", "scale", {"scale": 0.5}), ("width", 20, {}), ("height", 30, {}), ("box_fixed", 0, {"resize_type": "width", "height": 30}),
            ("scale", 0.5, {"reformat_type": "scale"}), ("resize_type", "fill", {"height": 30}), ("center", 0, {"width": 32, "height": 40}),
            ("flip", 1, {}), ("flop", 1, {}), ("turn", 1, {}), ("filter", "keys", {"width": 20, "height": 15, "resize_type": "distort"}),
            ("clamp", 1, {"filter": "rifman", "width": 24, "height": 18, "resize_type": "distort"}),
            ("black_outside", 0, {"width": 32, "height": 40, "resize_type": "fit"}))

    def test_every_listed_knob_changes_the_output(self):
        pixels = texture(12, 16)
        for name, value, base in self.ACTS:
            base_params = {"width": 24, "height": 18, "resize_type": "fit", **base}
            before = reformat(pixels, **base_params)
            after = reformat(pixels, **{**base_params, name: value})
            with self.subTest(knob=name):
                self.assertIn(name, SPECS["Reformat"]["params"])
                same_size = before.pixels.shape == after.pixels.shape
                self.assertTrue(not same_size or differs(before.pixels, after.pixels), name)

    def test_preserve_bounding_box_keeps_pixels_outside_the_format(self):
        pixels = texture(12, 16) + 1.0
        clipped = reformat(pixels, width=8, height=6, resize_type="none", preserve_bbox=0)
        kept = reformat(pixels, width=8, height=6, resize_type="none", preserve_bbox=1)
        self.assertEqual(clipped.data.width, 8)
        self.assertGreater(kept.data.width, 8)

    def test_a_named_format_sets_the_output_size(self):
        out = windowed("Reformat", texture(12, 16), reformat_type="to_format", width=64, height=36, filter="nearest")
        self.assertEqual((out.display.width, out.display.height), (64, 36))

    def test_the_main_tab_follows_nukes_order(self):
        self.assertEqual(labels("Reformat"), ["Type", "Output format", "Width/height", "Force this shape", "Pixel aspect", "Scale",
                                              "Resize type", "Center", "Flip", "Flop", "Turn", "Filter", "Clamp", "Black outside",
                                              "Preserve bounding box"])

    def test_an_older_reformat_is_given_the_older_readings(self):
        loaded = old_document("Reformat", dict(reformat_type="to_box", width=32, height=40, resize_type="fit", center=0, filter="nearest"),
                              ("legacy_reformat_layout", "box_fixed", "clamp", "black_outside"))
        params = loaded["nodes"]["n"]["params"]
        self.assertEqual((params["legacy_reformat_layout"], params["box_fixed"], params["clamp"], params["black_outside"]), (1, 1, 0, 1))
        pixels = Evaluator().evaluate_raster(dict(loaded, view="n"), target="n").pixels
        self.assertEqual(pixels.shape[:2], (40, 32))
        self.assertEqual(float(np.abs(pixels[28:, :, :3]).max()), 0.0)          # the picture is lined up with the top left: the lower rows are empty
        self.assertGreater(float(np.abs(pixels[:20, :, :3]).max()), 0.0)

    def test_the_older_turn_was_the_transpose(self):
        pixels = texture(12, 16)
        out = reformat(pixels, width=12, height=16, resize_type="distort", turn=1, legacy_reformat_layout=1).pixels
        np.testing.assert_array_equal(out, np.transpose(pixels, (1, 0, 2)))


# --------------------------------------------------------------------------------------------------------------- the windows
def graph_raster(kind, picture_params=None, frame=1, **knobs_):
    g = KeyGraph()
    g.add("plate", "Checker", picture_params or dict(width=24, height=16, size=3))
    g.add("node", kind, knobs_, image="plate")
    return Evaluator().evaluate_raster(dict(g.doc, view="node"), target="node", frame=frame), g


class EdgeWindowTests(unittest.TestCase):
    def test_black_outside_off_fills_the_whole_format_for_transform(self):
        r, _ = graph_raster("Transform", translate_x=5.0, filter="impulse", black_outside=0)
        self.assertTrue(r.data.intersect(r.display) == r.display)
        np.testing.assert_array_equal(r.fit(r.display)[:, :5], np.repeat(r.fit(r.display)[:, 5:6], 5, axis=1))

    def test_black_outside_on_keeps_the_moved_pictures_own_window(self):
        r, _ = graph_raster("Transform", translate_x=5.0, filter="impulse", black_outside=1)
        self.assertEqual(r.data.x, 4)                      # one pixel of reach for the filter's support
        self.assertEqual(float(np.abs(r.fit(r.display)[:, :4]).max()), 0.0)

    def test_the_older_window_stays_the_moved_pictures_own(self):
        r, _ = graph_raster("Transform", translate_x=5.0, filter="impulse", black_outside=0, legacy_transform_edges=1)
        self.assertEqual(r.data.x, 4)

    def test_cornerpin_black_outside_off_fills_the_whole_format(self):
        pts = {"from1_x": 0.0, "from1_y": 0.0, "from2_x": 24.0, "from2_y": 0.0, "from3_x": 0.0, "from3_y": 16.0, "from4_x": 24.0, "from4_y": 16.0,
               "to1_x": 5.0, "to1_y": 0.0, "to2_x": 29.0, "to2_y": 0.0, "to3_x": 5.0, "to3_y": 16.0, "to4_x": 29.0, "to4_y": 16.0}
        r, _ = graph_raster("CornerPin", filter="nearest", black_outside=0, **pts)
        self.assertTrue(r.data.intersect(r.display) == r.display)
        old, _ = graph_raster("CornerPin", filter="nearest", black_outside=0, legacy_transform_edges=1, **pts)
        self.assertGreater(old.data.x, r.data.x - 1)
        self.assertGreater(old.data.x, 0)


# ---------------------------------------------------------------------------------------------------------------- CornerPin
PIN_FRAME = {"from1_x": 0.0, "from1_y": 0.0, "from2_x": 16.0, "from2_y": 0.0, "from3_x": 0.0, "from3_y": 12.0, "from4_x": 16.0, "from4_y": 12.0}


def pin_params(**to):
    p = {**PIN_FRAME, "to1_x": 0.0, "to1_y": 0.0, "to2_x": 16.0, "to2_y": 0.0, "to3_x": 0.0, "to3_y": 12.0, "to4_x": 16.0, "to4_y": 12.0}
    p.update(to)
    return p


def pin(picture, **knobs_):
    """The CornerPin made on `picture` as it appears in the format (zero where the data window does not reach)."""
    raster = windowed("CornerPin", picture, **{"filter": "nearest", **knobs_})
    canvas = np.zeros(picture.shape, np.float64)
    shown = raster.display.intersect(raster.data)
    canvas[shown.y:shown.bottom, shown.x:shown.right] = raster.pixels[shown.y - raster.data.y:shown.bottom - raster.data.y,
                                                                    shown.x - raster.data.x:shown.right - raster.data.x]
    return canvas


class CornerPinTests(unittest.TestCase):
    def test_the_to_points_on_the_from_points_leave_the_picture_alone(self):
        pixels = texture(12, 16)
        for name in TRANSFORM_FILTERS:
            if name in ("mitchell", "parzen"):
                continue                       # these two smooth even at whole-pixel positions (B is not 0)
            with self.subTest(filter=name):
                np.testing.assert_allclose(pin(pixels, filter=name, **pin_params()), pixels, atol=1e-6)

    def test_mitchell_and_parzen_smooth_the_identity_pin_and_keep_the_total(self):
        pixels = np.zeros((12, 16, 4), np.float32)
        pixels[6, 8] = 1.0
        for name in ("mitchell", "parzen"):
            with self.subTest(filter=name):
                out = pin(pixels, filter=name, **pin_params())
                self.assertLess(float(out[..., 0].max()), 1.0)
                self.assertAlmostEqual(float(out[..., 0].sum()), 1.0, places=5)

    def test_every_filter_reproduces_a_constant_inside_the_pin(self):
        flat = np.full((12, 16, 4), 0.37, np.float32)
        for name in TRANSFORM_FILTERS:
            with self.subTest(filter=name):
                out = pin(flat, filter=name, black_outside=0, **pin_params(to1_x=1.3, to4_y=11.2))
                np.testing.assert_allclose(out, 0.37, atol=2e-6)

    def test_the_four_corners_land_on_the_to_points(self):
        pixels = np.zeros((12, 16, 4), np.float32)
        pixels[:] = (0.0, 0.0, 0.0, 1.0)
        pixels[0, 0] = (1, 0, 0, 1)                           # the pixel at the first from point
        pixels[0, 15] = (0, 1, 0, 1)
        pixels[11, 0] = (0, 0, 1, 1)
        pixels[11, 15] = (1, 1, 1, 1)
        out = pin(pixels, **pin_params(to1_x=2.0, to1_y=1.0, to2_x=14.0, to2_y=2.0, to3_x=1.0, to3_y=10.0, to4_x=15.0, to4_y=11.0))
        for (y, x), colour in (((1, 2), (1, 0, 0)), ((2, 13), (0, 1, 0)), ((10, 1), (0, 0, 1)), ((10, 14), (1, 1, 1))):
            near = out[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2, :3].reshape(-1, 3)
            self.assertTrue(any(tuple(np.round(v).astype(int)) == colour for v in near), (y, x, colour))     # on or next to the to point

    def test_invert_swaps_the_two_quads(self):
        pixels = texture(12, 16)
        moved = pin_params(to1_x=3.0, to2_x=19.0, to3_x=3.0, to4_x=19.0)
        forward = pin(pixels, **moved)
        back = pin(forward.astype(np.float32), direction="inverse", **moved)
        np.testing.assert_array_equal(back[:, :12], pixels[:, :12])

    def test_black_outside_default_and_off(self):
        self.assertEqual(SPECS["CornerPin"]["params"]["black_outside"], 1)
        pixels = texture(12, 16) + 1.0
        moved = pin_params(to1_x=3.0, to2_x=19.0, to3_x=3.0, to4_x=19.0)
        black = pin(pixels, **moved)
        edge = pin(pixels, black_outside=0, **moved)
        self.assertEqual(float(np.abs(black[:, :3, :3]).max()), 0.0)
        np.testing.assert_array_equal(edge[:, :3], np.repeat(pixels[:, :1], 3, axis=1))

    def test_clamp_keeps_a_sharpening_filter_inside_the_range(self):
        step = step_picture(16)[:12]
        moved = pin_params(to1_x=0.5, to3_x=0.5, to2_x=16.5, to4_x=16.5)
        free = pin(step, filter="rifman", black_outside=0, **moved)
        held = pin(step, filter="rifman", black_outside=0, clamp=1, **moved)
        self.assertGreater(float(free.max()), 1.0 + 1e-3)
        self.assertLessEqual(float(held.max()), 1.0)

    def test_the_shutter_follows_the_guide(self):
        # The picture moves right with the frame; a shutter that opens at the frame (start) sits further right than one that closes there.
        left = {}
        for offset in ("end", "centred", "start"):
            g = KeyGraph()
            g.add("plate", "Checker", dict(width=40, height=30, size=4))
            g.add("node", "CornerPin", {"filter": "nearest", "motionblur": 1, "samples": 4, "shutter": 2.0, "shutter_offset": offset,
                                       "from1_x": 0.0, "from1_y": 0.0, "from2_x": 40.0, "from2_y": 0.0, "from3_x": 0.0, "from3_y": 30.0,
                                       "from4_x": 40.0, "from4_y": 30.0, "to1_x": 0.0, "to1_y": 0.0, "to2_x": 40.0, "to2_y": 0.0,
                                       "to3_x": 0.0, "to3_y": 30.0, "to4_x": 40.0, "to4_y": 30.0}, image="plate")
            for corner in (1, 2, 3, 4):
                g.key("node", f"to{corner}_x", 1, [0.0, 40.0, 0.0, 40.0][corner - 1])
                g.key("node", f"to{corner}_x", 11, [10.0, 50.0, 10.0, 50.0][corner - 1])
            left[offset] = Evaluator().evaluate_raster(dict(g.doc, view="node"), target="node", frame=6).data.x
        self.assertLess(left["end"], left["centred"])
        self.assertLess(left["centred"], left["start"])

    ACTS = (("to1_x", 2.5), ("to1_y", 1.5), ("to2_x", 13.0), ("to2_y", 2.0), ("to3_x", 1.0), ("to3_y", 10.0), ("to4_x", 14.0), ("to4_y", 10.5),
            ("from1_x", 1.5), ("from2_y", 1.5), ("from3_x", 1.5), ("from4_y", 10.5), ("direction", "inverse"), ("filter", "rifman"),
            ("clamp", 1), ("black_outside", 0), ("enable1", 0))

    def test_every_listed_knob_changes_the_pixels(self):
        pixels = texture(12, 16)
        base = pin_params(to1_x=1.5, to2_x=14.5, to3_x=0.5, to4_x=15.5, to1_y=0.5, to4_y=11.5)
        base_extra = {"filter": "cubic"}
        for name, value in self.ACTS:
            kw = dict(base_extra)
            with self.subTest(knob=name):
                self.assertIn(name, SPECS["CornerPin"]["params"])
                a = pin(pixels, **{**base, **kw, **({"filter": "rifman"} if name == "clamp" else {})})
                b = pin(pixels, **{**base, **kw, **({"filter": "rifman"} if name == "clamp" else {}), name: value})
                self.assertTrue(differs(a, b), name)

    def test_the_main_tab_follows_nukes_order(self):
        wanted = [label for i in (1, 2, 3, 4) for label in (f"To {i}", f"Enable {i}")]
        wanted += [f"Extra matrix {r},{c}" for r in range(3) for c in range(3)]
        wanted += ["Invert", "Filter", "Clamp", "Black outside", "Motion blur", "Shutter", "Shutter offset",
                   "Samples", "From 1", "From 2", "From 3", "From 4"]
        self.assertEqual(labels("CornerPin"), wanted)

    def test_a_cornerpin_saved_before_the_step_keeps_the_older_readings(self):
        g = KeyGraph()
        g.add("plate", "Checker", dict(width=24, height=16, size=3))
        g.add("n", "CornerPin", {"filter": "nearest", "black_outside": 0, **pin_params(to1_x=3.0)}, image="plate")
        old = copy.deepcopy(g.doc)
        for flag in ("legacy_shutter_window", "legacy_transform_edges"):
            old["nodes"]["n"]["params"].pop(flag, None)
        loaded = upgrade_document(old)
        validate(loaded)
        self.assertEqual((loaded["nodes"]["n"]["params"]["legacy_shutter_window"], loaded["nodes"]["n"]["params"]["legacy_transform_edges"]), (1, 1))


# ----------------------------------------------------------------------------------------------------- the window nodes
def partial(kind, window=(4, 3, 8, 6), size=(16, 12), **knobs_):
    """`kind` run on a picture that covers `window` (x, y, w, h) of a `size` format: the Raster it makes."""
    from nodebased.tiers import Region
    x, y, w, h = window
    pixels = texture(size[1], size[0])[y:y + h, x:x + w]
    source = Raster(np.ascontiguousarray(pixels), Region(x, y, w, h), Region(0, 0, size[0], size[1]))
    return Evaluator._windowed_kernel(kind, {**SPECS[kind]["params"], **knobs_}, [source]), source


class CropTests(unittest.TestCase):
    def test_the_defaults_and_the_order_of_the_tab(self):
        self.assertEqual(labels("Crop"), ["Box", "Softness", "Reformat", "Intersect", "Black outside"])
        self.assertEqual((SPECS["Crop"]["params"]["crop_softness"], SPECS["Crop"]["params"]["reformat"], SPECS["Crop"]["params"]["crop"]),
                         (0.0, 0, 1))

    def test_softness_zero_is_no_vignette_and_larger_values_fade_more(self):
        pixels = texture(37, 45) + 1.0
        box_ = dict(x=5, y=4, width=30, height=24)
        plain = kern("Crop", pixels, **box_)
        soft2 = kern("Crop", pixels, crop_softness=2.0, **box_)
        soft6 = kern("Crop", pixels, crop_softness=6.0, **box_)
        np.testing.assert_array_equal(plain, kern("Crop", pixels, crop_softness=0.0, **box_))
        self.assertGreater(float(plain[4:28, 5:35, :3].sum()), float(soft2[4:28, 5:35, :3].sum()))
        self.assertGreater(float(soft2[4:28, 5:35, :3].sum()), float(soft6[4:28, 5:35, :3].sum()))
        self.assertLess(float(soft6[4, 20, 0]), float(plain[4, 20, 0]))                    # the edge fades
        np.testing.assert_allclose(soft6[16, 20], plain[16, 20], atol=1e-6)               # the middle does not

    def test_reformat_makes_the_box_the_format(self):
        r, _ = graph_raster("Crop", x=4, y=3, width=10, height=8, reformat=1)
        self.assertEqual((r.display.x, r.display.y, r.display.width, r.display.height), (4, 3, 10, 8))
        r, _ = graph_raster("Crop", x=4, y=3, width=10, height=8, reformat=0)
        self.assertEqual((r.display.width, r.display.height), (24, 16))

    def test_intersect_cuts_the_box_to_the_incoming_picture(self):
        on, _ = graph_raster("Crop", x=-4, y=-3, width=40, height=30, intersect=1)
        off, _ = graph_raster("Crop", x=-4, y=-3, width=40, height=30, intersect=0)
        self.assertEqual((on.data.x, on.data.y, on.data.width, on.data.height), (0, 0, 24, 16))
        self.assertEqual((off.data.x, off.data.y, off.data.width, off.data.height), (-4, -3, 40, 30))

    def test_black_outside_decides_what_the_box_holds_beyond_the_picture(self):
        black = partial("Crop", window=(6, 4, 4, 4), x=4, y=3, width=8, height=6, intersect=0, crop=1)[0]
        edge, source = partial("Crop", window=(6, 4, 4, 4), x=4, y=3, width=8, height=6, intersect=0, crop=0)
        self.assertEqual(float(np.abs(black.pixels[0, :, :3]).max()), 0.0)                 # the top row of the box lies above the picture
        np.testing.assert_array_equal(edge.pixels[0, 2:6], source.pixels[0, :4])           # it repeats the picture's top row there

    def test_every_listed_knob_changes_the_pixels(self):
        pixels = texture(37, 45) + 1.0
        base = dict(x=5, y=4, width=30, height=24)
        for name, value in (("x", 8), ("y", 7), ("width", 20), ("height", 12), ("crop_softness", 3.0)):
            with self.subTest(knob=name):
                self.assertTrue(differs(kern("Crop", pixels, **base), kern("Crop", pixels, **{**base, name: value})), name)


class MirrorTests(unittest.TestCase):
    def test_the_tab_is_nukes(self):
        self.assertEqual(labels("Mirror"), ["Horizontal", "Vertical"])

    def test_horizontal_flops_and_vertical_flips(self):
        pixels = texture(12, 16)
        np.testing.assert_array_equal(kern("Mirror", pixels, flip_x=1), pixels[:, ::-1])
        np.testing.assert_array_equal(kern("Mirror", pixels, flip_y=1), pixels[::-1])
        np.testing.assert_array_equal(kern("Mirror", pixels, flip_x=1, flip_y=1), pixels[::-1, ::-1])
        np.testing.assert_array_equal(kern("Mirror", pixels), pixels)

    def test_a_partial_picture_goes_to_the_other_side_of_the_format_centre(self):
        made, source = partial("Mirror", window=(2, 3, 6, 6), flip_x=1)
        self.assertEqual((made.data.x, made.data.y, made.data.width, made.data.height), (8, 3, 6, 6))
        np.testing.assert_array_equal(made.pixels, source.pixels[:, ::-1])
        made, source = partial("Mirror", window=(2, 3, 6, 6), flip_y=1)
        self.assertEqual((made.data.x, made.data.y), (2, 3))                                # rows 3 to 8 of 12 are their own mirror
        made, source = partial("Mirror", window=(2, 1, 6, 4), flip_y=1)
        self.assertEqual(made.data.y, 7)

    def test_the_older_window_flipped_inside_itself(self):
        made, source = partial("Mirror", window=(2, 3, 6, 6), flip_x=1, legacy_mirror_window=1)
        self.assertEqual((made.data.x, made.data.y), (2, 3))
        np.testing.assert_array_equal(made.pixels, source.pixels[:, ::-1])

    def test_mix_dissolves_to_the_original(self):
        pixels = texture(12, 16)
        np.testing.assert_allclose(kern("Mirror", pixels, flip_x=1, mix=0.5), 0.5 * pixels + 0.5 * pixels[:, ::-1], atol=1e-6)

    def test_an_older_mirror_is_given_the_older_window(self):
        loaded = old_document("Mirror", dict(flip_x=1), ("legacy_mirror_window",))
        self.assertEqual(loaded["nodes"]["n"]["params"]["legacy_mirror_window"], 1)


class PositionTests(unittest.TestCase):
    def test_the_tab_is_nukes(self):
        self.assertEqual(labels("Position"), ["Translate"])

    def test_positive_values_move_up_and_right_and_the_gap_repeats_the_edge(self):
        made, source = partial("Position", translate_x=3, translate_y=2)
        self.assertEqual((made.data.x, made.data.y, made.data.width, made.data.height), (4, 1, 11, 8))     # the union of the two
        union = made.data
        # the moved picture sits 3 right, 2 up
        moved_rows = slice(3 - 2 - union.y, 3 - 2 - union.y + 6)
        moved_cols = slice(4 + 3 - union.x, 4 + 3 - union.x + 8)
        np.testing.assert_array_equal(made.pixels[moved_rows, moved_cols], source.pixels)
        np.testing.assert_array_equal(made.pixels[:, 0], made.pixels[:, 3])                # the strip left of the moved picture repeats its left column

    def test_negative_values_move_down_and_left(self):
        made, _ = partial("Position", translate_x=-3, translate_y=-2)
        self.assertEqual((made.data.x, made.data.y, made.data.width, made.data.height), (1, 3, 11, 8))

    def test_zero_is_the_identity(self):
        made, source = partial("Position")
        self.assertEqual(made.data, source.data)
        np.testing.assert_array_equal(made.pixels, source.pixels)

    def test_the_display_window_stays_put(self):
        made, source = partial("Position", translate_x=3, translate_y=2)
        self.assertEqual(made.display, source.display)

    def test_the_older_position_moved_a_positive_y_down_with_no_fill(self):
        made, source = partial("Position", translate_x=3, translate_y=2, legacy_position=1)
        self.assertEqual((made.data.x, made.data.y, made.data.width, made.data.height), (7, 5, 8, 6))
        np.testing.assert_array_equal(made.pixels, source.pixels)

    def test_an_older_position_is_given_the_older_reading(self):
        loaded = old_document("Position", dict(translate_x=3, translate_y=2), ("legacy_position",))
        self.assertEqual(loaded["nodes"]["n"]["params"]["legacy_position"], 1)


class AdjustBBoxTests(unittest.TestCase):
    def test_the_tab_is_nukes_and_the_default_adds_25(self):
        self.assertEqual(labels("AdjustBBox"), ["Add pixels", "Clip to format"])
        self.assertEqual(SPECS["AdjustBBox"]["params"]["numpixels"], 25)

    def test_growing_adds_pixels_to_each_side_and_repeats_the_edge(self):
        made, source = partial("AdjustBBox", numpixels=2)
        self.assertEqual((made.data.x, made.data.y, made.data.width, made.data.height), (2, 1, 12, 10))
        np.testing.assert_array_equal(made.pixels[2:8, 2:10], source.pixels)
        np.testing.assert_array_equal(made.pixels[0], np.concatenate([np.repeat(source.pixels[:1, :1], 2, axis=1), source.pixels[:1],
                                                                       np.repeat(source.pixels[:1, -1:], 2, axis=1)], axis=1)[0])

    def test_the_older_growth_was_empty(self):
        made, source = partial("AdjustBBox", numpixels=2, legacy_adjustbbox=1)
        self.assertEqual(float(np.abs(made.pixels[:2]).max()), 0.0)
        np.testing.assert_array_equal(made.pixels[2:8, 2:10], source.pixels)

    def test_a_negative_number_crops_the_box(self):
        made, source = partial("AdjustBBox", numpixels=-1)
        self.assertEqual((made.data.x, made.data.y, made.data.width, made.data.height), (5, 4, 6, 4))
        np.testing.assert_array_equal(made.pixels, source.pixels[1:-1, 1:-1])

    def test_the_two_values_of_a_split_number_are_the_width_and_the_height(self):
        made, _ = partial("AdjustBBox", numpixels=[3, 1])
        self.assertEqual((made.data.x, made.data.y, made.data.width, made.data.height), (1, 2, 14, 8))
        equal, _ = partial("AdjustBBox", numpixels=[2, 2])
        scalar, _ = partial("AdjustBBox", numpixels=2)
        self.assertEqual(equal.data, scalar.data)
        np.testing.assert_array_equal(equal.pixels, scalar.pixels)

    def test_clip_to_format_cuts_the_box_to_the_format(self):
        made, _ = partial("AdjustBBox", numpixels=6, clip_to_format=1)
        self.assertEqual((made.data.x, made.data.y, made.data.width, made.data.height), (0, 0, 16, 12))

    def test_the_number_is_split_in_the_layout(self):
        from nodebased import core
        self.assertEqual(core.SPLITTABLE["AdjustBBox"], {"numpixels": "wh"})
        self.assertEqual(knobs.split_layout_problems(), [])


class BlackOutsideTests(unittest.TestCase):
    def test_the_box_is_kept_and_a_ring_of_black_surrounds_it(self):
        made, source = partial("BlackOutside")
        self.assertEqual((made.data.x, made.data.y, made.data.width, made.data.height), (3, 2, 10, 8))
        np.testing.assert_array_equal(made.pixels[1:-1, 1:-1], source.pixels)
        ring = made.pixels.copy()
        ring[1:-1, 1:-1] = 0.0
        self.assertEqual(float(np.abs(ring).max()), 0.0)

    def test_it_blacks_out_the_stripes_a_grown_box_repeats(self):
        grown, _ = partial("AdjustBBox", numpixels=2)
        from nodebased.tiers import Region
        again = Evaluator._windowed_kernel("BlackOutside", dict(SPECS["BlackOutside"]["params"]), [grown])
        self.assertEqual(again.data, grown.data.expand(1, 1))
        self.assertEqual(float(np.abs(again.pixels[0]).max()), 0.0)


class TileTests(unittest.TestCase):
    def tile(self, pixels, **knobs_):
        return np.asarray(windowed("Tile", pixels, **{"filter": "bilinear", **knobs_}).pixels, np.float64)

    def test_the_tab_is_nukes(self):
        self.assertEqual(labels("Tile"), ["Rows", "Mirror alternate rows", "Columns", "Mirror alternate columns", "Filter"])
        self.assertEqual((SPECS["Tile"]["params"]["rows"], SPECS["Tile"]["params"]["columns"], SPECS["Tile"]["params"]["filter"]),
                         (1, 1, "cubic"))

    def test_every_tile_is_the_same_scaled_picture(self):
        pixels = texture(24, 36)
        out = self.tile(pixels, rows=2, columns=3)
        first = out[:12, :12]
        for row in range(2):
            for column in range(3):
                np.testing.assert_array_equal(out[row * 12:(row + 1) * 12, column * 12:(column + 1) * 12], first)

    def test_mirror_flips_alternate_columns_and_rows(self):
        pixels = texture(24, 36)
        out = self.tile(pixels, rows=2, columns=3, mirror_x=1, mirror_y=1)
        plain = self.tile(pixels, rows=2, columns=3)
        first = plain[:12, :12]
        np.testing.assert_array_equal(out[:12, :12], first)
        np.testing.assert_array_equal(out[:12, 12:24], first[:, ::-1])
        np.testing.assert_array_equal(out[12:, :12], first[::-1])
        np.testing.assert_array_equal(out[12:, 12:24], first[::-1, ::-1])
        np.testing.assert_array_equal(out[:12, 24:], first)

    def test_the_filter_changes_the_pixels_and_every_filter_keeps_a_constant(self):
        pixels = texture(24, 36)
        self.assertTrue(differs(self.tile(pixels, rows=2, columns=2, filter="bilinear"), self.tile(pixels, rows=2, columns=2, filter="cubic")))
        self.assertTrue(differs(self.tile(pixels, rows=2, columns=2, filter="keys"), self.tile(pixels, rows=2, columns=2, filter="parzen")))
        flat = np.full((24, 36, 4), 0.37, np.float32)
        for name in TRANSFORM_FILTERS:
            with self.subTest(filter=name):
                np.testing.assert_allclose(self.tile(flat, rows=3, columns=2, filter=name), 0.37, atol=2e-6)

    def test_rows_and_columns_act(self):
        pixels = texture(24, 36)
        self.assertTrue(differs(self.tile(pixels, rows=1, columns=1), self.tile(pixels, rows=2, columns=1)))
        self.assertTrue(differs(self.tile(pixels, rows=1, columns=1), self.tile(pixels, rows=1, columns=2)))

    def test_an_older_tile_is_given_bilinear_and_draws_the_same(self):
        loaded = old_document("Tile", dict(rows=2, columns=2, mirror_x=0, mirror_y=0), ("filter",))
        self.assertEqual(loaded["nodes"]["n"]["params"]["filter"], "bilinear")
        pixels = render(loaded, "src")
        out = render(loaded, "n")
        # the older sampling, written out: bilinear at pixel centres, edge pixels held
        h, w = pixels.shape[:2]
        expect = np.empty_like(pixels)
        for row in range(2):
            for column in range(2):
                y0, y1, x0, x1 = row * h // 2, (row + 1) * h // 2, column * w // 2, (column + 1) * w // 2
                sx = np.clip((np.arange(x1 - x0, dtype=np.float32) + .5) * w / (x1 - x0) - .5, 0, w - 1)
                sy = np.clip((np.arange(y1 - y0, dtype=np.float32) + .5) * h / (y1 - y0) - .5, 0, h - 1)
                xlo, ylo = np.floor(sx).astype(int), np.floor(sy).astype(int)
                xhi, yhi = np.minimum(xlo + 1, w - 1), np.minimum(ylo + 1, h - 1)
                fx, fy = (sx - xlo)[None, :, None], (sy - ylo)[:, None, None]
                top = pixels[ylo[:, None], xlo[None, :]] * (1 - fx) + pixels[ylo[:, None], xhi[None, :]] * fx
                bottom = pixels[yhi[:, None], xlo[None, :]] * (1 - fx) + pixels[yhi[:, None], xhi[None, :]] * fx
                expect[y0:y1, x0:x1] = top * (1 - fy) + bottom * fy
        np.testing.assert_array_equal(out, expect)


# ----------------------------------------------------------------------------------------------- the full-frame exclusion
class ExcludedFromTilesTests(unittest.TestCase):
    """Like Transform, Crop and Mirror before them, these nodes change or depend on the data window or the canvas origin, so a graph
    holding one runs on the full-frame evaluator (docs/PARITY_2D.md); the tile executor must not claim it."""

    def test_the_tile_executor_leaves_each_node_to_the_full_frame_evaluator(self):
        from nodebased.tileexec import TileExecutor
        for kind in ("Transform", "Reformat", "CornerPin", "Crop", "Mirror", "Position", "Tile", "AdjustBBox", "BlackOutside"):
            with self.subTest(kind=kind):
                g = KeyGraph()
                g.add("plate", "Checker", dict(width=24, height=16, size=3))
                g.add("node", kind, {}, image="plate")
                executor = TileExecutor(evaluator=Evaluator(), tile_edge=16)
                self.assertFalse(executor.supports_tiled(dict(g.doc, view="node"), "node"), kind)


if __name__ == "__main__":
    unittest.main()
