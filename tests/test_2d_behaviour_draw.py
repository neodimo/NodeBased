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

DRAW_NODES = ("Rectangle", "Radial", "Ramp", "Noise", "Grid")
DONE_NODES = DRAW_NODES + ("Grain", "Dither")
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
        self.assertGreaterEqual(len(scored), 24)
        for case in scored:
            with self.subTest(case=case.id):
                result = audit.run_case(case)
                self.assertEqual(result["status"], "pass", (case.id, result["max_abs_error"], result["worst"], case.statement))

    def test_every_unscored_case_states_its_reason(self):
        unscored = [c for c in ruler.CASES if c.nuke in DONE_NODES and c.unscored]
        self.assertGreaterEqual(len(unscored), 13)
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


# ------------------------------------------------------------------------------------------------------------------ Noise
def noise(w=48, h=40, **knobs_):
    """The coverage the Noise node draws (its alpha), over absolute pixels."""
    base = dict(size=9.0, octaves=3, seed=3, gamma=1.0, xrotate=20.0, yrotate=15.0, center_x=24.0, center_y=20.0)
    return shape("Noise", w=w, h=h, **{**base, **knobs_})[..., 3]


class NoiseTests(unittest.TestCase):
    def test_the_same_settings_give_the_same_pixels_and_another_seed_another_pattern(self):
        np.testing.assert_array_equal(noise(), noise())
        self.assertTrue(differs(noise(), noise(seed=4)))

    def test_the_values_are_between_0_and_1(self):
        for kind in ("fBm", "turbulence"):
            out = noise(type=kind)
            self.assertGreaterEqual(float(out.min()), 0.0)
            self.assertLessEqual(float(out.max()), 1.0)

    def test_turbulence_is_not_fbm(self):
        self.assertTrue(differs(noise(type="turbulence"), noise(type="fBm")))

    def test_gain_multiplies_the_amplitude_of_each_octave(self):
        np.testing.assert_allclose(noise(octaves=6, gain=0.0), noise(octaves=1), atol=1e-12)
        self.assertTrue(differs(noise(octaves=6, gain=0.5), noise(octaves=1)))

    def test_more_octaves_add_finer_detail(self):
        def roughness(a):
            return float(np.abs(np.diff(a, axis=1)).mean())
        self.assertGreater(roughness(noise(octaves=5, gain=0.7)), roughness(noise(octaves=1)))

    def test_lower_lacunarity_is_smoother_and_higher_is_more_granular(self):
        def roughness(a):
            return float(np.abs(np.diff(a, axis=1)).mean() + np.abs(np.diff(a, axis=0)).mean())
        low, mid, high = (roughness(noise(octaves=4, gain=0.7, lacunarity=value, nyquist=0)) for value in (1.2, 2.0, 3.0))
        self.assertLess(low, mid)
        self.assertLess(mid, high)

    def test_gamma_is_the_power_the_output_is_raised_to(self):
        plain = noise(gamma=1.0)
        np.testing.assert_allclose(noise(gamma=0.5), plain ** 2.0, atol=1e-6)
        np.testing.assert_allclose(noise(gamma=2.0), plain ** 0.5, atol=1e-6)

    def test_nyquist_stops_octaves_finer_than_a_pixel(self):
        fine = dict(size=1.5, octaves=8, lacunarity=2.0)
        self.assertTrue(differs(noise(nyquist=1, **fine), noise(nyquist=0, **fine)))
        np.testing.assert_allclose(noise(nyquist=1, **fine), noise(nyquist=1, **{**fine, "octaves": 2}), atol=1e-12)

    def test_translate_moves_the_pattern_right_and_up(self):
        base, moved = noise(), noise(translate_x=3.0, translate_y=2.0)
        np.testing.assert_allclose(moved[:-2, 3:], base[2:, :-3], atol=1e-5)               # up on the screen is towards row 0

    def test_rotate_turns_the_pattern_about_the_center_counter_clockwise(self):
        base = noise(w=49, h=41, center_x=24.0, center_y=20.0)
        half_turn = noise(w=49, h=41, center_x=24.0, center_y=20.0, rotate_x=180.0)
        np.testing.assert_allclose(half_turn, base[::-1, ::-1], atol=1e-5)
        quarter = noise(w=41, h=41, center_x=20.0, center_y=20.0, rotate_x=90.0)
        base_square = noise(w=41, h=41, center_x=20.0, center_y=20.0)
        np.testing.assert_allclose(quarter, np.rot90(base_square, 1), atol=1e-5)          # counter-clockwise on the screen

    def test_scaling_stretches_the_pattern_about_the_center(self):
        base = noise(w=49, h=41, center_x=24.0, center_y=20.0)
        doubled = noise(w=49, h=41, center_x=24.0, center_y=20.0, scaling=2.0)
        np.testing.assert_allclose(doubled[20, 24::2][:10], base[20, 24:34], atol=1e-5)    # the pattern at the center is the base pattern half as wide

    def test_the_x_and_y_sizes_stretch_the_pattern_one_axis_each(self):
        wide = noise(size=[30.0, 4.0], octaves=1)
        self.assertGreater(float(np.abs(np.diff(wide, axis=0)).mean()), 3.0 * float(np.abs(np.diff(wide, axis=1)).mean()))
        tall = noise(size=[4.0, 30.0], octaves=1)
        self.assertGreater(float(np.abs(np.diff(tall, axis=1)).mean()), 3.0 * float(np.abs(np.diff(tall, axis=0)).mean()))
        np.testing.assert_allclose(noise(size=[9.0, 9.0]), noise(size=9.0), atol=0)

    def test_the_size_splits_into_an_x_and_a_y(self):
        self.assertEqual(core.SPLITTABLE["Noise"], {"size": "wh"})
        self.assertEqual(knobs.split_layout_problems(), [])

    def test_zoffset_moves_through_the_noise_smoothly(self):
        base = noise()
        near, far = noise(z_slice=0.05), noise(z_slice=2.5)
        self.assertLess(float(np.abs(near - base).mean()), 0.2 * float(np.abs(far - base).mean()))
        self.assertTrue(differs(base, far))

    def test_xrotate_and_yrotate_turn_the_volume_not_the_picture(self):
        flat = noise(xrotate=0.0, yrotate=0.0, z_slice=0.37)
        self.assertTrue(differs(flat, noise(xrotate=30.0, yrotate=0.0, z_slice=0.37)))
        self.assertTrue(differs(flat, noise(xrotate=0.0, yrotate=30.0, z_slice=0.37)))
        # the picture stays upright: a turned volume gives the same pattern at the same place when z is zero
        np.testing.assert_allclose(noise(xrotate=30.0, yrotate=0.0, z_slice=0.0)[:, 24 * 0 + 24], noise(xrotate=30.0, yrotate=0.0, z_slice=0.0)[:, 24], atol=0)

    def test_the_noise_is_the_coverage_of_the_colour_and_draws_over_the_input(self):
        params = dict(size=9.0, octaves=2, gamma=1.0, red=0.5, green=0.25, blue=0.75, alpha=0.8)
        plate = background(40, 48)
        out = draw("Noise", plate, **params)
        cover = noise(octaves=2, size=9.0, seed=0, xrotate=30.0, yrotate=30.0, center_x=480.0, center_y=270.0) if False else shape("Noise", w=48, h=40, **params)[..., 3] / 0.8
        want = np.concatenate([np.array([0.5, 0.25, 0.75]) * 0.8 * cover[..., None], (0.8 * cover)[..., None]], axis=-1)
        np.testing.assert_allclose(out, want + plate * (1.0 - want[..., 3:4]), atol=1e-5)

    def test_invert_inverts_the_coverage(self):
        plain, inverted = noise(), shape("Noise", w=48, h=40, size=9.0, octaves=3, seed=3, gamma=1.0, xrotate=20.0, yrotate=15.0, center_x=24.0,
                                         center_y=20.0, invert=1)[..., 3]
        np.testing.assert_allclose(inverted, 1.0 - plain, atol=1e-6)

    def test_the_ramp_colours_the_noise_from_color_0_to_color(self):
        ramped = shape("Noise", w=48, h=40, size=9.0, octaves=1, gamma=1.0, ramp="linear", p0_x=0.0, p0_y=0.0, p1_x=0.0, p1_y=40.0,
                       color0_red=0.0, color0_green=0.0, color0_blue=0.0, red=1.0, green=1.0, blue=1.0)
        flat = shape("Noise", w=48, h=40, size=9.0, octaves=1, gamma=1.0)
        np.testing.assert_allclose(ramped[..., 3], flat[..., 3], atol=1e-6)
        straight = ramped[..., 0] / np.maximum(ramped[..., 3], 1e-9)
        self.assertLess(float(straight[-1].mean()), 0.1)                                   # the bottom is color 0
        self.assertGreater(float(straight[0].mean()), 0.9)                                 # the top is color

    def test_older_noise_is_opaque_grey_on_a_2d_slice(self):
        out = shape("Noise", w=48, h=40, size=9.0, octaves=2, legacy_draw_axes=1)
        np.testing.assert_array_equal(out[..., 3], 1.0)
        np.testing.assert_array_equal(out[..., 0], out[..., 1])


# ------------------------------------------------------------------------------------------------------------------- Grid
def grid_lines(**knobs_):
    out = shape("Grid", w=40, h=24, **{"number_x": 4, "number_y": 3, "line_width": 1.0, "grid_offset_x": 0.0, "grid_offset_y": 0.0, **knobs_})[..., 3]
    return out, np.where(out.min(axis=0) > 0)[0], np.where(out.min(axis=1) > 0)[0]            # a vertical line fills its column, a horizontal one its row


class GridTests(unittest.TestCase):
    def test_number_is_the_count_of_squares_across_the_format(self):
        out, cols, rows = grid_lines()
        self.assertEqual(list(cols), [0, 10, 20, 30])                          # 40 pixels in 4 squares
        self.assertEqual(sorted(24 - 1 - rows), [0, 8, 16])                    # 24 pixels in 3 squares, rows counted from the bottom
        self.assertEqual(list(grid_lines(number_x=8, number_y=6)[1]), [0, 5, 10, 15, 20, 25, 30, 35])

    def test_size_is_the_line_thickness_in_pixels(self):
        for size in (1, 2, 3, 5):
            out, cols, rows = grid_lines(line_width=float(size))
            self.assertEqual(len(cols), 4 * size, size)

    def test_translate_moves_the_whole_grid_right_and_up(self):
        base = grid_lines()[0]
        moved = grid_lines(grid_offset_x=3.0, grid_offset_y=2.0)[0]
        np.testing.assert_array_equal(moved[:-2, 3:], base[2:, :-3])

    def test_the_lines_are_in_the_colour(self):
        out = shape("Grid", w=40, h=24, number_x=4, number_y=3, red=1.0, green=0.5, blue=0.25, alpha=0.8)
        self.assertTrue(np.allclose(out[0, 0], [0.8, 0.4, 0.2, 0.8]))

    def test_invert_fills_the_squares_and_leaves_holes_for_the_lines(self):
        plain = grid_lines()[0]
        inverted = shape("Grid", w=40, h=24, number_x=4, number_y=3, invert=1)[..., 3]
        np.testing.assert_allclose(inverted, 1.0 - plain, atol=1e-6)

    def test_the_ramp_colours_the_lines(self):
        ramped = shape("Grid", w=40, h=24, number_x=4, number_y=3, ramp="linear", p0_x=0.0, p0_y=0.0, p1_x=0.0, p1_y=24.0, color0_red=0.0,
                       color0_green=0.0, color0_blue=0.0)
        flat = shape("Grid", w=40, h=24, number_x=4, number_y=3)
        np.testing.assert_array_equal(ramped[..., 3], flat[..., 3])
        self.assertTrue(differs(ramped, flat))
        self.assertLess(float(ramped[23, 0, 0]), float(ramped[1, 0, 0]))                  # the bottom (point 0) is the darker colour

    def test_the_older_grid_counts_from_the_top_and_defaults_to_the_spacing(self):
        self.assertEqual(SPECS["Grid"]["params"]["number_x"], 10)
        out = shape("Grid", w=40, h=24, number_x=0, number_y=0, spacing_x=10.0, spacing_y=8.0, legacy_draw_axes=1)[..., 3]
        self.assertEqual(sorted(np.where(out.min(axis=1) > 0)[0]), [0, 8, 16])


class NoiseGridKnobsActTests(unittest.TestCase):
    def acts(self, kind, base, changes):
        reference = draw(kind, **base)
        for name, value in changes:
            with self.subTest(kind=kind, knob=name):
                self.assertIn(name, SPECS[kind]["params"], name)
                self.assertTrue(differs(reference, draw(kind, **{**base, name: value})), name)

    CHROME = (("replace", 1), ("invert", 1), ("opacity", 0.5), ("premult", "rgba.alpha"), ("output", "alpha"))

    def test_noise(self):
        base = dict(size=9.0, octaves=3, seed=3, gamma=1.0, center_x=20.0, center_y=18.0, alpha=0.8)
        self.acts("Noise", base, [("type", "turbulence"), ("size", 6.0), ("size", [20.0, 5.0]), ("z_slice", 1.3), ("octaves", 1),
                                  ("lacunarity", 3.0), ("gain", 0.2), ("gamma", 0.5), ("translate_x", 3.0), ("translate_y", 2.0), ("rotate_x", 25.0),
                                  ("rotate_y", 25.0), ("scaling", 1.6), ("skew", 20.0), ("xrotate", 0.0), ("yrotate", 0.0),
                                  ("red", 0.3), ("alpha", 0.4), ("seed", 5), ("ramp", "linear"), *self.CHROME])
        self.acts("Noise", {**base, "rotate_x": 30.0}, [("center_x", 4.0), ("center_y", 5.0)])
        self.acts("Noise", {**base, "size": 1.5, "octaves": 8}, [("nyquist", 0)])
        ramped = {**base, "ramp": "linear", "p0_x": 0.0, "p0_y": 0.0, "p1_x": 0.0, "p1_y": 30.0}
        self.acts("Noise", ramped, [("p0_y", 10.0), ("p1_y", 20.0), ("color0_red", 0.6), ("ramp", "smooth")])

    def test_grid(self):
        base = dict(number_x=4, number_y=3, alpha=0.9)
        self.acts("Grid", base, [("number_x", 6), ("number_y", 5), ("line_width", 3.0), ("grid_offset_x", 2.0), ("grid_offset_y", 2.0), ("red", 0.2),
                                 ("alpha", 0.4), ("ramp", "linear"), *self.CHROME])
        self.acts("Grid", {**base, "ramp": "linear", "p0_x": 0.0, "p0_y": 0.0, "p1_x": 0.0, "p1_y": 30.0},
                  [("p0_y", 10.0), ("p1_y", 20.0), ("color0_red", 0.6), ("ramp", "smooth1")])

    def test_the_main_tabs_lay_their_knobs_out_in_nuke_order(self):
        head = ["Output", "Premult", "Clip to", "Replace", "Invert", "Opacity", "Mask", "Inject", "Invert"]
        self.assertEqual(labels("Noise")[:29], head + ["Type", "X/ysize", "Z", "Octaves", "Clip at Nyquist limit", "Lacunarity", "Gain", "Gamma",
                                                       "Translate xy", "Rotate", "Scale", "Skew", "Center xy", "Xrotate", "Yrotate", "Ramp", "Color",
                                                       "Point 1", "Color 0", "Point 0"])
        self.assertEqual(labels("Grid")[:15], head + ["Number", "Size", "Translate", "Ramp", "Color", "Point 1"])


# --------------------------------------------------------------------------------------------------------- Dither and Grain
def kernel(kind, pixels, frame=1, **knobs_):
    spec = SPECS[kind]
    slots = [np.asarray(pixels, np.float32)] + [None] * (len(spec["inputs"]) + len(spec.get("optional_inputs", [])) - 1)
    return np.asarray(Evaluator._kernel(kind, {**spec["params"], **knobs_}, slots, frame=frame), np.float64)


def flat(value=0.4, alpha=1.0, h=40, w=48):
    picture = np.full((h, w, 4), value, np.float32)
    picture[..., 3] = alpha
    return picture


class DitherTests(unittest.TestCase):
    PICTURE = None

    @classmethod
    def setUpClass(cls):
        cls.PICTURE = np.clip(background(48, 64), 0.0, 1.0)

    def delta(self, **knobs_):
        return kernel("Dither", self.PICTURE, **{"channels": "rgba", **knobs_}) - self.PICTURE

    def test_up_to_half_the_amount_is_added_or_taken_away(self):
        for amount in (0.1, 0.4, 1.0):
            with self.subTest(amount=amount):
                d = self.delta(dither_amount=amount)
                self.assertLessEqual(float(np.abs(d).max()), 0.5 * amount + 1e-6)
                self.assertGreater(float(np.abs(d).max()), 0.45 * amount)             # and nearly all of it is used

    def test_the_noise_has_no_bias(self):
        d = self.delta(dither_amount=0.4)
        self.assertLess(abs(float(d.mean())), 0.01 * 0.4)

    def test_amount_scales_the_noise_and_zero_adds_nothing(self):
        np.testing.assert_allclose(self.delta(dither_amount=0.2), 2.0 * self.delta(dither_amount=0.1), atol=1e-6)
        np.testing.assert_array_equal(self.delta(dither_amount=0.0), 0.0)

    def test_monodither_uses_one_pattern_for_all_channels(self):
        mono = self.delta(monodither=1)
        np.testing.assert_allclose(mono[..., 0], mono[..., 1], atol=1e-6)
        np.testing.assert_allclose(mono[..., 0], mono[..., 2], atol=1e-6)
        apart = self.delta(monodither=0)
        self.assertTrue(differs(apart[..., 0], apart[..., 1]))

    def test_the_seed_changes_the_pattern_and_the_same_seed_repeats_it(self):
        self.assertTrue(differs(self.delta(seed=1), self.delta(seed=2)))
        np.testing.assert_array_equal(self.delta(seed=5), self.delta(seed=5))

    def test_the_pattern_changes_each_frame_unless_the_seed_is_static(self):
        a = kernel("Dither", self.PICTURE, frame=1, channels="rgba") - self.PICTURE
        b = kernel("Dither", self.PICTURE, frame=2, channels="rgba") - self.PICTURE
        self.assertTrue(differs(a, b))
        a = kernel("Dither", self.PICTURE, frame=1, channels="rgba", static_seed=1) - self.PICTURE
        b = kernel("Dither", self.PICTURE, frame=2, channels="rgba", static_seed=1) - self.PICTURE
        np.testing.assert_array_equal(a, b)

    def test_channels_limits_where_it_is_applied(self):
        out = kernel("Dither", self.PICTURE, channels="rgb")
        np.testing.assert_array_equal(out[..., 3], self.PICTURE[..., 3])
        self.assertTrue(differs(out[..., :3], self.PICTURE[..., :3]))

    def test_bits_rounds_the_dithered_values_when_asked(self):
        out = kernel("Dither", self.PICTURE, bits=4, channels="rgba")
        np.testing.assert_allclose(out * 15.0, np.round(out * 15.0), atol=1e-4)
        self.assertTrue(differs(out, kernel("Dither", self.PICTURE, bits=0, channels="rgba")))

    def test_an_older_dither_rounds_to_bits_after_triangular_noise(self):
        out = kernel("Dither", self.PICTURE, legacy_dither=1, bits=5, dither_amount=1.0, seed=2, channels="rgb")
        levels = 31.0
        iy, ix = np.mgrid[0:48, 0:64]
        for c in range(3):
            noise = Evaluator._hash_lattice(ix, iy, c, 2) + Evaluator._hash_lattice(ix, iy, c, 2 + 7919) - 1.0
            want = np.clip(np.round(self.PICTURE[..., c].astype(np.float64) * levels + noise), 0.0, levels) / levels
            np.testing.assert_allclose(out[..., c], want, atol=1e-6)


class GrainTests(unittest.TestCase):
    GREY = flat(0.5, 1.0)

    def delta(self, picture=None, **knobs_):
        picture = self.GREY if picture is None else picture
        return kernel("Grain", picture, **{"minimum_red": -1000.0, "minimum_green": -1000.0, "minimum_blue": -1000.0, **knobs_}) - picture

    def test_zero_intensity_adds_nothing(self):
        d = self.delta(red_intensity=0.0, green_intensity=0.0, blue_intensity=0.0)
        np.testing.assert_array_equal(d, 0.0)

    def test_the_grain_has_no_bias(self):
        d = self.delta(flat(0.5, 1.0, h=160, w=200))
        for c in range(3):
            self.assertLess(abs(float(d[..., c].mean())), 0.1 * float(d[..., c].std()) + 1e-9)
            self.assertGreater(float(d[..., c].std()), 0.0)

    def test_each_channel_has_its_own_size_irregularity_and_intensity(self):
        base = self.delta()
        for knob, value, channel in (("red_size", 7.0, 0), ("green_i", 0.1, 1), ("blue_intensity", 0.2, 2)):
            with self.subTest(knob=knob):
                moved = self.delta(**{knob: value})
                for c in range(4):
                    same = np.array_equal(moved[..., c], base[..., c])
                    self.assertEqual(same, c != channel, (knob, c))

    def test_intensity_scales_the_grain_and_a_larger_size_gives_a_coarser_grain(self):
        np.testing.assert_allclose(self.delta(red_intensity=0.8)[..., 0], 2.0 * self.delta(red_intensity=0.4)[..., 0], atol=1e-6)

        def roughness(d):
            return float(np.abs(np.diff(d[..., 0], axis=1)).mean() / d[..., 0].std())
        self.assertLess(roughness(self.delta(red_size=8.0)), roughness(self.delta(red_size=1.0)))

    def test_irregularity_bends_the_pattern(self):
        self.assertTrue(differs(self.delta(red_i=0.0)[..., 0], self.delta(red_i=1.0)[..., 0]))
        self.assertAlmostEqual(float(self.delta(red_i=0.0)[..., 0].std()), float(self.delta(red_i=1.0)[..., 0].std()), delta=0.2 * float(self.delta()[..., 0].std()))

    def test_the_pattern_changes_each_frame_and_holds_when_the_seed_is_minus_the_frame(self):
        self.assertTrue(differs(kernel("Grain", self.GREY, frame=1) - self.GREY, kernel("Grain", self.GREY, frame=2) - self.GREY))
        a = kernel("Grain", self.GREY, frame=1, seed=-1) - self.GREY
        b = kernel("Grain", self.GREY, frame=2, seed=-2) - self.GREY
        np.testing.assert_array_equal(a, b)
        self.assertTrue(differs(self.delta(seed=134), self.delta(seed=135)))

    def test_black_is_the_amount_of_grain_at_the_black_point(self):
        dark, light = flat(0.0, 1.0), flat(1.0, 1.0)
        self.assertEqual(float(np.abs(self.delta(dark, black=0.0)[..., :3]).max()), 0.0)             # nothing at black
        at_black = self.delta(dark, black=1.0)[..., :3]
        np.testing.assert_allclose(np.abs(at_black).max(), np.abs(self.delta(light, black=0.0)[..., :3]).max(), rtol=0.2)
        half = self.delta(dark, black=0.5)[..., 0]
        np.testing.assert_allclose(half, 0.5 * self.delta(dark, black=1.0)[..., 0], atol=1e-6)

    def test_the_grain_grows_with_the_brightness_of_the_pixel(self):
        low, high = self.delta(flat(0.2, 1.0))[..., 0], self.delta(flat(0.8, 1.0))[..., 0]
        np.testing.assert_allclose(4.0 * low, high, atol=1e-6)

    def test_apply_only_through_alpha_scales_by_the_alpha_and_invert_by_one_minus_it(self):
        half = flat(0.5, 0.5)
        full = self.delta(flat(0.5, 1.0), maskgrain=1)[..., 0]
        np.testing.assert_allclose(self.delta(half, maskgrain=1)[..., 0], 0.5 * full, atol=1e-6)
        np.testing.assert_allclose(self.delta(half, maskgrain=1, invert_mask=1)[..., 0], 0.5 * full, atol=1e-6)
        np.testing.assert_array_equal(self.delta(flat(0.5, 0.0), maskgrain=1)[..., :3], 0.0)
        np.testing.assert_array_equal(self.delta(flat(0.5, 1.0), maskgrain=1, invert_mask=1)[..., :3], 0.0)
        np.testing.assert_allclose(self.delta(half, maskgrain=0)[..., 0], full, atol=1e-6)         # unchecked: the whole image, alpha ignored

    def test_minimum_raises_every_value_below_it(self):
        out = kernel("Grain", np.full((8, 8, 4), 0.01, np.float32), red_intensity=0.0, green_intensity=0.0, blue_intensity=0.0,
                     minimum_red=0.2, minimum_green=0.3, minimum_blue=0.4)
        np.testing.assert_allclose(out[0, 0, :3], [0.2, 0.3, 0.4], atol=1e-6)
        self.assertAlmostEqual(float(out[0, 0, 3]), 0.01, places=6)
        self.assertEqual(SPECS["Grain"]["params"]["minimum_red"], 0.0)

    def test_the_defaults_are_the_guides(self):
        p = SPECS["Grain"]["params"]
        self.assertEqual((p["seed"], p["red_size"], p["green_size"], p["blue_size"]), (134, 3.3, 2.9, 2.5))
        self.assertEqual((p["red_i"], p["green_i"], p["blue_i"]), (0.6, 0.6, 0.6))
        self.assertEqual((p["red_intensity"], p["green_intensity"], p["blue_intensity"]), (0.416, 0.46, 0.85))
        self.assertEqual((p["black"], p["maskgrain"], p["invert_mask"]), (0.0, 1, 0))
        d = SPECS["Dither"]["params"]
        self.assertEqual((d["dither_amount"], d["monodither"], d["seed"], d["static_seed"]), (0.1, 0, 1, 0))

    def test_an_older_grain_is_the_unweighted_noise_at_the_older_scale(self):
        picture = flat(0.5, 1.0)
        out = kernel("Grain", picture, legacy_grain=1, red_intensity=0.05, green_intensity=0.0, blue_intensity=0.0, red_size=3.0, seed=7, frame=2)
        gy, gx = np.mgrid[0:40, 0:48].astype(np.float64)
        want = 0.5 + (Evaluator._value_noise(gx / 3.0, gy / 3.0, 0, 9) - 0.5) * np.sqrt(12.0) * 0.05
        np.testing.assert_allclose(out[..., 0], want, atol=1e-6)
        np.testing.assert_array_equal(out[..., 1:], picture[..., 1:])


class DitherGrainKnobsActTests(unittest.TestCase):
    PICTURE = np.clip(background(40, 48), 0.0, 1.0)

    def acts(self, kind, base, changes):
        reference = kernel(kind, self.PICTURE, **base)
        for name, value in changes:
            with self.subTest(kind=kind, knob=name):
                self.assertIn(name, SPECS[kind]["params"], name)
                self.assertTrue(differs(reference, kernel(kind, self.PICTURE, **{**base, name: value})), name)

    def test_dither(self):
        self.acts("Dither", dict(channels="rgb"), [("dither_amount", 0.3), ("monodither", 1), ("seed", 4), ("channels", "rgba"), ("bits", 4)])
        self.acts("Dither", dict(channels="rgb", static_seed=0), [])
        self.assertTrue(differs(kernel("Dither", self.PICTURE, frame=1, static_seed=0), kernel("Dither", self.PICTURE, frame=2, static_seed=0)))

    def test_grain(self):
        self.acts("Grain", dict(), [("seed", 5), ("red_size", 6.0), ("green_size", 6.0), ("blue_size", 6.0), ("red_i", 0.1), ("green_i", 0.1),
                                    ("blue_i", 0.1), ("red_intensity", 0.9), ("green_intensity", 0.9), ("blue_intensity", 0.2), ("black", 0.6),
                                    ("minimum_red", 0.7), ("minimum_green", 0.7), ("minimum_blue", 0.7), ("maskgrain", 0), ("invert_mask", 1)])

    def test_the_main_tabs_lay_their_knobs_out_in_nuke_order(self):
        self.assertEqual(labels("Dither")[:5], ["Channels", "Amount", "Monodither", "Seed", "Static Seed"])
        self.assertEqual(labels("Grain")[:14], ["Seed", "Size red", "Size green", "Size blue", "Irregularity red", "Irregularity green", "Irregularity blue",
                                                "Intensity red", "Intensity green", "Intensity blue", "Black", "Minimum rgb", "Apply only through alpha",
                                                "Invert alpha"])


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
    CHANGED_EXTRA = {
        "Noise": dict(size=7.0, octaves=3, gamma=1.0, seed=4, translate_x=2.0, translate_y=1.0, rotate_x=15.0, scaling=1.3, center_x=20.0, center_y=18.0),
        "Noise split": dict(size=[11.0, 5.0], octaves=2, gamma=0.7, z_slice=0.4, ramp="smooth", p0_x=3.0, p0_y=4.0, p1_x=40.0, p1_y=30.0, alpha=0.8),
        "Noise inverted": dict(size=7.0, octaves=2, invert=1, opacity=0.7, red=0.4),
        "Grid": dict(number_x=5, number_y=4, line_width=2.0, grid_offset_x=1.0, grid_offset_y=2.0, alpha=0.8),
        "Dither": dict(dither_amount=0.2, seed=3, channels="rgba"),
        "Dither mono": dict(dither_amount=0.3, monodither=1, static_seed=1, bits=6),
        "Grain": dict(red_size=2.5, green_size=4.0, blue_size=6.0, red_i=0.8, green_i=0.2, blue_i=0.5, seed=9, black=0.2,
                      minimum_red=0.05, minimum_green=0.1, minimum_blue=0.02),
        "Grain inverted": dict(maskgrain=1, invert_mask=1, red_intensity=0.7, blue_intensity=0.2),
        "Grid ramp": dict(number_x=6, number_y=5, line_width=1.0, ramp="linear", p0_x=3.0, p0_y=4.0, p1_x=40.0, p1_y=30.0, color0_red=0.3, invert=1),
    }

    def test_tiles_equal_the_whole_frame(self):
        for label, params in {**self.CHANGED, **self.CHANGED_EXTRA}.items():
            kind = label.split()[0]
            for masked in (False, True):
                g = source_graph(kind, {**params, **({"width": W, "height": H} if kind in DRAW_NODES else {})}, masked)
                whole = evaluator_pixels(g.doc, "node")
                for edge in (7, 16, 32):
                    with self.subTest(kind=label, masked=masked, tile=edge):
                        np.testing.assert_allclose(tile_pixels(g.doc, "node", edge), whole, atol=1e-5)


# ----------------------------------------------------------------------------------------------------- an older document
class OlderDocumentTests(unittest.TestCase):
    FLAGS = {"Rectangle": ("legacy_draw_axes",), "Radial": ("legacy_draw_axes", "plinear", "ramp"), "Ramp": ("legacy_draw_axes", "ramp_type"),
             "Noise": ("legacy_draw_axes", "red", "green", "blue", "alpha"), "Grid": ("legacy_draw_axes", "ramp", "p0_x", "p0_y", "p1_x", "p1_y")}
    KNOBS = {"Rectangle": dict(box_x=9.0, box_y=6.0, box_width=25.0, box_height=18.0, softness=0.4, alpha=0.8, width=W, height=H),
             "Radial": dict(box_x=4.0, box_y=3.0, box_width=36.0, box_height=30.0, softness=0.7, alpha=0.9, width=W, height=H),
             "Ramp": dict(p0_x=3.0, p0_y=4.0, p1_x=40.0, p1_y=30.0, color0_red=0.2, width=W, height=H),
             "Noise": dict(size=8.0, octaves=3, seed=2, gamma=1.0, width=W, height=H),
             "Grid": dict(spacing_x=9.0, spacing_y=8.0, number_x=0, number_y=0, line_width=2.0, width=W, height=H)}

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

    def test_noise_and_grid_saved_before_the_step_keep_their_pixels(self):
        for kind in ("Noise", "Grid"):
            with self.subTest(kind=kind):
                old = old_document(kind, self.KNOBS[kind], self.FLAGS[kind])
                params = old["nodes"]["n"]["params"]
                self.assertEqual(params["legacy_draw_axes"], 1)
                pixels = render(old, "n")
                again = copy.deepcopy(old)
                again["nodes"]["n"]["params"].pop("legacy_draw_axes")
                np.testing.assert_array_equal(render(again, "n"), pixels)
                new = copy.deepcopy(old)
                new["nodes"]["n"]["params"]["legacy_draw_axes"] = 0
                self.assertTrue(differs(render(new, "n"), pixels))

    def test_a_new_node_reads_the_new_way(self):
        for kind in DRAW_NODES:
            with self.subTest(kind=kind):
                self.assertEqual(SPECS[kind]["params"]["legacy_draw_axes"], 0)
        self.assertEqual(SPECS["Ramp"]["params"]["ramp_type"], "linear")
        self.assertEqual(SPECS["Radial"]["params"]["plinear"], 0)
        validate(upgrade_document(copy.deepcopy(source_graph("Radial", {"width": W, "height": H}, False).doc)))


if __name__ == "__main__":
    unittest.main()
