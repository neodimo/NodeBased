"""2D behaviour, lane 9 step C: the Draw nodes (Rectangle, Radial, Ramp, Noise, Grid, Grain, Dither) and Defocus, ZDefocus and EdgeExtend
compute what the Nuke 17.0 Reference Guide says they compute.

As in steps A and B: the behaviour cases pass where the guide (or a published form) fixes the pixels, every main-tab knob NodeBased pairs
with a Nuke knob changes pixels, the kernels this step changed give the same pixels in tiles of 7, 16 and 32 as on the whole frame (with
and without a mask), a document saved before the change renders bit for bit as it did, and each node's main tab lists its knobs in
Nuke's order.
"""
import copy
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from nodebased import core, drawbehaviour, knobs
from nodebased.core import SPECS, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.raster import Raster
from tests.test_2d_behaviour_filter_a import differs, evaluator_pixels, old_document, render, source_graph, tile_pixels

import audit_nuke_fidelity as audit  # noqa: E402
import nuke_fidelity_cases as ruler  # noqa: E402

DRAW_NODES = ("Rectangle", "Radial", "Ramp")
DONE_NODES = DRAW_NODES
W, H = 45, 37


def labels(kind):
    """The labels of a node's main tab in the order the panel lays them out (the mix and the legacy switches left off)."""
    return [g.label for g in knobs.KNOB_LAYOUT[kind] if g.kind != "legacy" and not g.params[0].startswith("legacy_") and g.params != ("mix",)]


def background(h=H, w=W):
    """A picture with a varied colour and a partial alpha for a shape to be drawn over."""
    y, x = np.mgrid[0:h, 0:w]
    base = 0.25 + 0.2 * np.sin(x / 4.0) + 0.2 * np.cos(y / 5.0)
    return np.stack([base, base * 0.5, 1.0 - base, 0.5 + 0.4 * np.cos((x + y) / 7.0)], axis=-1).astype(np.float32)


def draw(kind, pixels=None, **knobs_):
    """One draw node over `pixels` (a blank picture of the node's own format when none), through the windowed kernel."""
    pixels = background() if pixels is None else pixels
    params = {**SPECS[kind]["params"], "width": pixels.shape[1], "height": pixels.shape[0], **knobs_}
    return np.asarray(Evaluator._windowed_kernel(kind, params, [Raster.of(np.asarray(pixels, np.float32))]).pixels, np.float64)


def shape(kind, w=W, h=H, **knobs_):
    """The shape alone, as the kernel draws it before it is composited (premultiplied RGBA)."""
    params = {**SPECS[kind]["params"], "width": w, "height": h, **knobs_}
    return np.asarray(Evaluator._draw_shape(kind, params, 0, 0, w, h, 1), np.float64)


# ------------------------------------------------------------------------------------------------------------- the cases
class CaseTests(unittest.TestCase):
    def test_every_scored_case_of_the_step_passes(self):
        scored = [c for c in ruler.CASES if c.nuke in DONE_NODES and not c.unscored]
        self.assertGreaterEqual(len(scored), 19)
        for case in scored:
            with self.subTest(case=case.id):
                result = audit.run_case(case)
                self.assertEqual(result["status"], "pass", (case.id, result["max_abs_error"], result["worst"], case.statement))

    def test_every_unscored_case_states_its_reason(self):
        unscored = [c for c in ruler.CASES if c.nuke in DONE_NODES and c.unscored]
        self.assertGreaterEqual(len(unscored), 8)
        for case in unscored:
            with self.subTest(case=case.id):
                self.assertGreater(len(case.unscored), 40, case.id)
                self.assertEqual(audit.run_case(case)["status"], "not scored")

    def test_every_node_of_the_step_has_a_case(self):
        for node in DONE_NODES:
            with self.subTest(node=node):
                self.assertTrue(ruler.cases_for(node), node)


# ------------------------------------------------------------------------------------------- Rectangle, Radial and Ramp
class AxesTests(unittest.TestCase):
    def test_a_rectangle_counts_y_from_the_bottom(self):
        out = shape("Rectangle", box_x=5.0, box_y=0.0, box_width=10.0, box_height=4.0, softness=0.0)[..., 3]
        self.assertEqual(set(np.argwhere(out > 0)[:, 0]), set(range(H - 4, H)))
        self.assertEqual(set(np.argwhere(out > 0)[:, 1]), set(range(5, 15)))

    def test_the_older_reading_counts_y_from_the_top(self):
        out = shape("Rectangle", box_x=5.0, box_y=0.0, box_width=10.0, box_height=4.0, softness=0.0, legacy_draw_axes=1)[..., 3]
        self.assertEqual(set(np.argwhere(out > 0)[:, 0]), set(range(0, 4)))

    def test_a_ramps_point_0_is_the_bottom_end(self):
        out = shape("Ramp", p0_x=0.0, p0_y=0.0, p1_x=0.0, p1_y=float(H), color0_red=0.0, color0_green=0.0, color0_blue=0.0,
                    color1_red=1.0, color1_green=1.0, color1_blue=1.0)[..., 0]
        self.assertLess(out[-1, 0], 0.05)                       # the bottom row is the point 0 colour
        self.assertGreater(out[0, 0], 0.95)                     # the top row is the point 1 colour
        self.assertTrue(np.all(np.diff(out[:, 0]) < 0))         # rows go from point 1 (top) down to point 0

    def test_a_radials_area_counts_y_from_the_bottom(self):
        low = shape("Radial", box_x=10.0, box_y=0.0, box_width=20.0, box_height=10.0, softness=0.0)[..., 3]
        rows = np.argwhere(low > 0)[:, 0]
        self.assertEqual((rows.min(), rows.max()), (H - 10, H - 1))

    def test_the_ramp_is_measured_at_the_pixel_centre(self):
        out = shape("Ramp", p0_x=0.0, p0_y=0.0, p1_x=10.0, p1_y=0.0, color0_red=0.0, color1_red=1.0)[..., 0]
        np.testing.assert_allclose(out[0, :10], (np.arange(10) + 0.5) / 10.0, atol=1e-6)
        older = shape("Ramp", p0_x=0.0, p0_y=0.0, p1_x=10.0, p1_y=0.0, color0_red=0.0, color1_red=1.0, legacy_draw_axes=1)[..., 0]
        np.testing.assert_allclose(older[0, :10], np.arange(10) / 10.0, atol=1e-6)


class RampTypeTests(unittest.TestCase):
    """The five ramp types: the page gives the words, so the tests check the ends, which end each eases into and the symmetry."""
    KNOBS = dict(p0_x=0.0, p0_y=0.0, p1_x=0.0, p1_y=float(H), color0_red=0.0, color0_green=0.0, color0_blue=0.0, color1_red=1.0,
                 color1_green=1.0, color1_blue=1.0)

    def column(self, kind, **more):
        return shape("Ramp", ramp_type=kind, **self.KNOBS, **more)[::-1, 0, 0]                  # from point 0 (bottom) to point 1 (top)

    def test_every_type_keeps_both_end_colours_and_rises(self):
        for kind in drawbehaviour.RAMP_TYPES:
            with self.subTest(type=kind):
                col = self.column(kind)
                self.assertLess(col[0], 0.05)
                self.assertGreater(col[-1], 0.95)
                self.assertTrue(np.all(np.diff(col) > 0))

    def test_smooth0_eases_into_point_0_and_smooth1_into_point_1(self):
        linear, s0, s1 = (self.column(k) for k in ("linear", "smooth0", "smooth1"))
        self.assertTrue(np.all(s0[1:-1] < linear[1:-1]))        # colours at the point 0 end are spread wider
        self.assertTrue(np.all(s1[1:-1] > linear[1:-1]))

    def test_smooth_eases_into_both_ends_and_is_symmetric(self):
        linear, smooth = self.column("linear"), self.column("smooth")
        n = len(smooth)
        self.assertLess(smooth[2], linear[2])
        self.assertGreater(smooth[n - 3], linear[n - 3])
        np.testing.assert_allclose(smooth + smooth[::-1], 1.0, atol=1e-6)

    def test_plinear_has_the_linear_ends_and_a_darker_middle(self):
        linear, plinear = self.column("linear"), self.column("plinear")
        self.assertTrue(np.all(plinear[1:-1] < linear[1:-1]))
        self.assertAlmostEqual(float(plinear[0]), float(linear[0]), delta=0.02)

    def test_the_older_ramp_is_the_linear_one(self):
        np.testing.assert_array_equal(self.column("smooth", legacy_draw_axes=1)[::-1], self.column("linear", legacy_draw_axes=1)[::-1])


class RectangleTests(unittest.TestCase):
    AREA = dict(box_x=10.0, box_y=8.0, box_width=20.0, box_height=16.0)

    def test_softness_zero_is_a_hard_edge(self):
        alpha = shape("Rectangle", softness=0.0, **self.AREA)[..., 3]
        self.assertEqual(set(np.unique(alpha)), {0.0, 1.0})

    def test_more_softness_widens_the_fade_and_the_fade_is_symmetric(self):
        widths = []
        for soft in (0.1, 0.3, 0.6):
            alpha = shape("Rectangle", softness=soft, **self.AREA)[..., 3]
            widths.append(int(((alpha > 0) & (alpha < 1)).sum()))
            np.testing.assert_allclose(alpha[:, 10:30], alpha[::-1, 10:30][::-1], atol=0)         # (self-check of the slice)
            row = alpha[H - 8 - 8]
            left, right = row[10:20], row[20:30][::-1]
            np.testing.assert_allclose(left, right, atol=1e-6)
        self.assertEqual(widths, sorted(widths))
        self.assertLess(widths[0], widths[-1])

    def test_a_ramp_runs_from_color_0_at_point_0_to_color_at_point_1(self):
        out = shape("Rectangle", box_x=0.0, box_y=0.0, box_width=float(W), box_height=float(H), softness=0.0, ramp="linear",
                    p0_x=0.0, p0_y=0.0, p1_x=0.0, p1_y=float(H), color0_red=0.0, color0_green=0.0, color0_blue=0.0, color0_alpha=1.0,
                    red=1.0, green=1.0, blue=1.0, alpha=1.0)[..., 0]
        self.assertLess(out[-1, 0], 0.05)
        self.assertGreater(out[0, 0], 0.95)

    def test_the_ramp_types_change_the_picture(self):
        base = dict(box_x=0.0, box_y=0.0, box_width=float(W), box_height=float(H), softness=0.0, p0_x=0.0, p0_y=0.0, p1_x=0.0, p1_y=float(H),
                    color0_red=0.0, color0_green=0.0, color0_blue=0.0)
        outs = {kind: shape("Rectangle", ramp=kind, **base) for kind in ("linear", "smooth0", "smooth1", "smooth")}
        for a in outs:
            for b in outs:
                if a < b:
                    self.assertTrue(differs(outs[a], outs[b]), (a, b))

    def test_invert_inverts_the_coverage_before_the_colour_alpha(self):
        plain = shape("Rectangle", softness=0.0, alpha=0.6, **self.AREA)[..., 3]
        inverted = shape("Rectangle", softness=0.0, alpha=0.6, invert=1, **self.AREA)[..., 3]
        np.testing.assert_allclose(inverted, 0.6 - plain, atol=1e-6)


class RadialTests(unittest.TestCase):
    AREA = dict(box_x=5.0, box_y=4.0, box_width=35.0, box_height=29.0)

    def test_the_soft_ramp_is_full_at_the_centre_and_falls_to_nothing_at_the_edge(self):
        alpha = shape("Radial", softness=1.0, **self.AREA)[..., 3]
        cy, cx = H - 1 - (4.0 + 14.0), 5.0 + 17.0
        self.assertGreater(alpha[int(cy), int(cx)], 0.95)
        row = alpha[int(cy)]
        self.assertTrue(np.all(np.diff(row[int(cx):40]) <= 1e-9))                 # never rises going outward
        self.assertEqual(row[0], 0.0)
        self.assertEqual(row[42], 0.0)

    def test_softness_zero_is_a_hard_edge_and_more_softness_softens_it(self):
        hard = shape("Radial", softness=0.0, **self.AREA)[..., 3]
        self.assertEqual(set(np.unique(hard)), {0.0, 1.0})
        counts = [int(((a > 0) & (a < 1)).sum()) for a in (shape("Radial", softness=s, **self.AREA)[..., 3] for s in (0.2, 0.6, 1.0))]
        self.assertEqual(counts, sorted(counts))

    def test_perceptually_linear_keeps_the_centre_and_edge_and_darkens_the_middle(self):
        plain = shape("Radial", softness=1.0, **self.AREA)[..., 3]
        plin = shape("Radial", softness=1.0, plinear=1, **self.AREA)[..., 3]
        self.assertTrue(differs(plain, plin))
        inside = (plain > 0.05) & (plain < 0.95)
        self.assertTrue(np.all(plin[inside] < plain[inside]))
        self.assertAlmostEqual(float(plin.max()), float(plain.max()), delta=0.02)
        np.testing.assert_array_equal(plin == 0, plain == 0)

    def test_the_colour_tab_ramps_the_colour_across_the_ellipse(self):
        flat = shape("Radial", softness=0.0, red=1.0, green=1.0, blue=1.0, alpha=1.0, **self.AREA)
        ramped = shape("Radial", softness=0.0, ramp="linear", p0_x=0.0, p0_y=0.0, p1_x=0.0, p1_y=float(H), color0_red=0.0, color0_green=0.0,
                       color0_blue=0.0, color0_alpha=1.0, red=1.0, green=1.0, blue=1.0, alpha=1.0, **self.AREA)
        np.testing.assert_array_equal(flat[..., 3], ramped[..., 3])
        inside = flat[..., 3] > 0
        self.assertTrue(differs(flat[inside], ramped[inside]))
        col = np.where(inside[:, 22])[0]
        self.assertGreater(ramped[col.min(), 22, 0], ramped[col.max(), 22, 0])          # brighter towards the top (point 1)


# --------------------------------------------------------------------------------------------------- every knob acts
class KnobsActTests(unittest.TestCase):
    """Every main-tab knob NodeBased pairs with a Nuke knob changes the pixels."""
    AREA = dict(box_x=10.0, box_y=8.0, box_width=20.0, box_height=16.0)
    COLOUR_RAMP = dict(ramp="linear", p0_x=0.0, p0_y=0.0, p1_x=0.0, p1_y=float(H), color0_red=0.0, color0_green=0.0, color0_blue=0.0)
    CHROME = (("replace", 1), ("invert", 1), ("opacity", 0.5), ("premult", "rgba.alpha"), ("output", "alpha"))

    def acts(self, kind, base, changes):
        reference = draw(kind, **base)
        for name, value in changes:
            with self.subTest(kind=kind, knob=name):
                self.assertIn(name, SPECS[kind]["params"], name)
                self.assertTrue(differs(reference, draw(kind, **{**base, name: value})), name)

    def test_rectangle(self):
        base = dict(softness=0.0, alpha=0.7, **self.AREA)
        self.acts("Rectangle", base, [("box_x", 12.0), ("box_y", 10.0), ("box_width", 14.0), ("box_height", 9.0), ("softness", 0.5), ("red", 0.2),
                                      ("green", 0.2), ("blue", 0.2), ("alpha", 0.3), *self.CHROME])
        ramped = {**base, **self.COLOUR_RAMP}
        self.acts("Rectangle", {**base}, [("ramp", "linear")])
        self.acts("Rectangle", ramped, [("ramp", "smooth"), ("p0_y", 10.0), ("p1_y", 20.0), ("p0_x", 0.0 + 1e-9 + 3.0), ("color0_red", 0.5),
                                        ("color0_alpha", 0.2), ("red", 0.3)])

    def test_radial(self):
        base = dict(softness=0.6, alpha=0.7, box_x=5.0, box_y=4.0, box_width=35.0, box_height=29.0)
        self.acts("Radial", base, [("box_x", 8.0), ("box_y", 7.0), ("box_width", 20.0), ("box_height", 12.0), ("softness", 1.0), ("plinear", 1),
                                   ("red", 0.2), ("alpha", 0.3), *self.CHROME])
        self.acts("Radial", base, [("ramp", "linear")])
        self.acts("Radial", {**base, **self.COLOUR_RAMP}, [("ramp", "smooth0"), ("p0_y", 10.0), ("p1_y", 20.0), ("color0_red", 0.5), ("color0_alpha", 0.2)])

    def test_ramp(self):
        base = dict(p0_x=0.0, p0_y=0.0, p1_x=0.0, p1_y=float(H), color1_alpha=0.8)
        self.acts("Ramp", base, [("p0_y", 10.0), ("p1_y", 20.0), ("p0_x", 8.0), ("p1_x", 20.0), ("ramp_type", "smooth"), ("color0_red", 0.5),
                                 ("color0_alpha", 0.2), ("color1_red", 0.3), ("color1_green", 0.3), ("color1_blue", 0.3), ("color1_alpha", 0.4), *self.CHROME])

    def test_the_main_tabs_lay_their_knobs_out_in_nuke_order(self):
        head = ["Output", "Premult", "Clip to", "Replace", "Invert", "Opacity", "Mask", "Inject", "Invert"]
        self.assertEqual(labels("Rectangle")[:16], head + ["Area (x, y, w, h)", "Softness", "Ramp", "Color", "Point 1 xy", "Color 0", "Point 0 xy"])
        self.assertEqual(labels("Radial")[:17], head + ["Area (x, y, w, h)", "Softness", "Perceptually linear", "Ramp", "Color", "Point 1 xy", "Color 0",
                                                         "Point 0 xy"])
        self.assertEqual(labels("Ramp")[:13], head + ["Point 0 xy", "Point 1 xy", "Type", "Color"])

    def test_the_ramp_type_choices_are_the_guides(self):
        self.assertEqual(core.CHOICES["ramp_type"], ["linear", "plinear", "smooth0", "smooth1", "smooth"])


# ------------------------------------------------------------------------------------------------------------- the tiles
class TileTests(unittest.TestCase):
    """The kernels this step changed give the same pixels in tiles of 7, 16 and 32 as on the whole frame, with and without a mask."""
    CHANGED = {
        "Rectangle": dict(box_x=9.0, box_y=6.0, box_width=25.0, box_height=18.0, softness=0.4, alpha=0.8),
        "Rectangle ramp": dict(box_x=9.0, box_y=6.0, box_width=25.0, box_height=18.0, softness=0.2, ramp="smooth", p0_x=5.0, p0_y=3.0, p1_x=30.0,
                               p1_y=30.0, color0_red=0.2, color0_green=0.4, color0_blue=0.6),
        "Rectangle inverted": dict(box_x=9.0, box_y=6.0, box_width=25.0, box_height=18.0, softness=0.0, alpha=0.6, invert=1),
        "Radial": dict(box_x=4.0, box_y=3.0, box_width=36.0, box_height=30.0, softness=0.7, plinear=1, alpha=0.9),
        "Radial ramp": dict(box_x=4.0, box_y=3.0, box_width=36.0, box_height=30.0, softness=0.3, ramp="smooth1", p0_x=5.0, p0_y=3.0, p1_x=30.0,
                            p1_y=30.0, color0_red=0.2, color0_green=0.4, color0_blue=0.6, invert=1),
        "Ramp": dict(p0_x=3.0, p0_y=4.0, p1_x=40.0, p1_y=30.0, color0_red=0.2, color1_blue=0.3),
        "Ramp smooth": dict(p0_x=3.0, p0_y=4.0, p1_x=40.0, p1_y=30.0, ramp_type="smooth0", color1_alpha=0.7),
        "Ramp plinear": dict(p0_x=3.0, p0_y=4.0, p1_x=40.0, p1_y=30.0, ramp_type="plinear", opacity=0.8),
    }
    CHANGED_EXTRA = {}

    def test_tiles_equal_the_whole_frame(self):
        for label, params in {**self.CHANGED, **self.CHANGED_EXTRA}.items():
            kind = label.split()[0]
            for masked in (False, True):
                g = source_graph(kind, {**params, "width": W, "height": H}, masked)
                whole = evaluator_pixels(g.doc, "node")
                for edge in (7, 16, 32):
                    with self.subTest(kind=label, masked=masked, tile=edge):
                        np.testing.assert_allclose(tile_pixels(g.doc, "node", edge), whole, atol=1e-5)


# ----------------------------------------------------------------------------------------------------- an older document
class OlderDocumentTests(unittest.TestCase):
    FLAGS = {"Rectangle": ("legacy_draw_axes",), "Radial": ("legacy_draw_axes", "plinear", "ramp"), "Ramp": ("legacy_draw_axes", "ramp_type")}
    KNOBS = {"Rectangle": dict(box_x=9.0, box_y=6.0, box_width=25.0, box_height=18.0, softness=0.4, alpha=0.8, width=W, height=H),
             "Radial": dict(box_x=4.0, box_y=3.0, box_width=36.0, box_height=30.0, softness=0.7, alpha=0.9, width=W, height=H),
             "Ramp": dict(p0_x=3.0, p0_y=4.0, p1_x=40.0, p1_y=30.0, color0_red=0.2, width=W, height=H)}

    def test_a_node_saved_before_the_step_is_given_the_older_readings(self):
        for kind, flags in self.FLAGS.items():
            with self.subTest(kind=kind):
                params = old_document(kind, self.KNOBS[kind], flags)["nodes"]["n"]["params"]
                self.assertEqual(params["legacy_draw_axes"], 1)
                self.assertEqual(params.get("plinear", 0), 0)
                self.assertEqual(params.get("ramp", "none") if kind != "Ramp" else "none", "none")

    def test_it_draws_the_same_pixels_as_the_older_kernel(self):
        # The older Rectangle counted rows from the top and the older Ramp sampled at the index: written out here from that reading.
        ys, xs = np.mgrid[0:H, 0:W]
        old = old_document("Rectangle", self.KNOBS["Rectangle"], self.FLAGS["Rectangle"])
        pixels, plate = render(old, "n"), render(old, "src")
        dist = np.minimum(np.minimum(xs - 9.0, 9.0 + 25.0 - 1.0 - xs), np.minimum(ys - 6.0, 6.0 + 18.0 - 1.0 - ys))
        alpha = (np.clip(dist / (0.4 * 18.0 / 2.0), 0.0, 1.0) * 0.8)[..., None]
        want = np.concatenate([np.ones((H, W, 3)) * alpha, alpha], axis=-1) + plate * (1.0 - alpha)
        np.testing.assert_allclose(pixels, want, atol=1e-6)
        again = copy.deepcopy(old)
        again["nodes"]["n"]["params"].pop("legacy_draw_axes")                      # a node that never carried the flag reads it as set
        np.testing.assert_array_equal(render(again, "n"), pixels)
        # the older Radial: rows from the top, an ellipse centred on the area
        old_radial = old_document("Radial", self.KNOBS["Radial"], self.FLAGS["Radial"])
        cx, cy = 4.0 + 35.0 / 2.0, 3.0 + 29.0 / 2.0
        d = np.sqrt(((xs - cx) / 18.0) ** 2 + ((ys - cy) / 15.0) ** 2)
        alpha = (np.clip((1.0 - d) / 0.7, 0.0, 1.0) * 0.9)[..., None]
        want = np.concatenate([np.ones((H, W, 3)) * alpha, alpha], axis=-1) + render(old_radial, "src") * (1.0 - alpha)
        np.testing.assert_allclose(render(old_radial, "n"), want, atol=1e-6)
        # the older ramp: index sampling, linear, rows from the top
        shown = render(old_document("Ramp", self.KNOBS["Ramp"], self.FLAGS["Ramp"]), "n")
        t = np.clip(((xs - 3.0) * 37.0 + (ys - 4.0) * 26.0) / (37.0 ** 2 + 26.0 ** 2), 0.0, 1.0)
        np.testing.assert_allclose(shown[..., 0], 0.2 * (1.0 - t) + t, atol=1e-5)

    def test_a_new_node_reads_the_new_way(self):
        for kind in DRAW_NODES:
            with self.subTest(kind=kind):
                self.assertEqual(SPECS[kind]["params"]["legacy_draw_axes"], 0)
        self.assertEqual(SPECS["Ramp"]["params"]["ramp_type"], "linear")
        self.assertEqual(SPECS["Radial"]["params"]["plinear"], 0)
        validate(upgrade_document(copy.deepcopy(source_graph("Radial", {"width": W, "height": H}, False).doc)))


if __name__ == "__main__":
    unittest.main()
