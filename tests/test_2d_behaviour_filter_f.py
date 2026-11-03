"""2D behaviour, lane 9 step F: Tracker, DropShadow, DirBlur compute what the Nuke 17.0 Reference Guide says they compute.

* the behaviour cases of tools/nuke_fidelity_cases_filter.py for the three nodes pass, and the unscored ones say why;
* Tracker: a feature that moves by a known sub-pixel step per frame is found within a stated tolerance, match-move applies that motion and stabilise
  removes it, and the main tab lists the knobs in Nuke's order;
* DropShadow: angle reads as the guide's (0 left, 180 right) in a new node and as it did in a saved one; distance, softness, opacity, colour and
  `operation` (with a background wired) act; the tile path equals the evaluator;
* DirBlur: linear, zoom and radial; UseTarget, Samples and pixeloffset act; the knobs it does not yet act on are listed with what they need.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from nodebased import filterbehaviour, knobs
from nodebased.core import SPECS, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.tracker import analyse
from tests.test_2d_behaviour_filter_a import Graph, differs, evaluator_pixels, kern, old_document, tile_pixels, texture

import audit_nuke_fidelity as audit  # noqa: E402
import nuke_fidelity_cases as ruler  # noqa: E402
import nuke_fidelity_requirements as requirements  # noqa: E402

STEP_F_NODES = ("Tracker", "DropShadow", "DirBlur")
SOURCE_IDS = {"Tracker": "tracker_", "DropShadow": "dropshadow_", "DirBlur": "dirblur_"}


def step_cases(node):
    return [c for c in ruler.CASES if c.nuke == node and c.id.startswith(SOURCE_IDS[node])]


def labels(kind):
    """The labels of a node's main tab in the order the panel lays them out (the mix and the legacy switches left off)."""
    return [g.label for g in knobs.KNOB_LAYOUT[kind] if not g.params[0].startswith("legacy_") and g.params != ("mix",)]


def layout_order(nuke_key):
    """(knobs in order, knobs paired) by the audit's own rule for the node's layout."""
    import json
    docs = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "parity")
    nuke = json.load(open(os.path.join(docs, "nuke_nodes.json")))
    nuke = nuke.get("nodes", nuke)[nuke_key] if "nodes" in nuke else nuke[nuke_key]
    pair = json.load(open(os.path.join(docs, "nodebased_to_nuke.json")))["pairs"][nuke_key]
    wanted = [k for k in nuke["knobs"] if k["counted"] and not k["role"].startswith("chrome:")]
    fraction, detail = requirements.layout_component(pair["nodebased"], pair["knobs"], wanted)
    return detail["layout_order"], detail["layout_labels"], detail["label_mismatches"]


class CaseTests(unittest.TestCase):
    def test_every_scored_case_of_the_step_passes(self):
        for node in STEP_F_NODES:
            scored = [c for c in step_cases(node) if not c.unscored]
            self.assertGreaterEqual(len(scored), 3, node)
            for case in scored:
                with self.subTest(case=case.id):
                    result = audit.run_case(case)
                    self.assertEqual(result["status"], "pass", (case.id, result["max_abs_error"], result["worst"], case.statement))

    def test_every_unscored_case_states_its_reason(self):
        unscored = [c for node in STEP_F_NODES for c in step_cases(node) if c.unscored]
        self.assertGreaterEqual(len(unscored), 3)
        for case in unscored:
            with self.subTest(case=case.id):
                self.assertGreater(len(case.unscored), 40, case.id)
                self.assertEqual(audit.run_case(case)["status"], "not scored")


# ------------------------------------------------------------------------------------------------------------ Tracker
SIZE = 64
VELOCITY = (0.4, -0.25)                                   # the feature's step per frame: 0.4 pixel right, 0.25 up (rows count down)
TRACK_TOLERANCE = 0.2                                     # pixels; the sub-pixel refinement of the match is good to about 0.13 on this footage


def feature_frame(frame):
    """Two soft blobs at a position the test writes: (30, 28) at frame 1 plus VELOCITY per frame. Drawn analytically at each position, so the
    sub-pixel truth is exact (no resampling in the footage)."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    cx, cy = 30.0 + VELOCITY[0] * (frame - 1), 28.0 + VELOCITY[1] * (frame - 1)
    grey = (np.exp(-((xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2) / (2 * 3.0 ** 2))
            + 0.6 * np.exp(-((xx + 0.5 - cx - 7) ** 2 + (yy + 0.5 - cy + 4) ** 2) / (2 * 2.0 ** 2)))
    image = np.ones((SIZE, SIZE, 4), np.float32)
    image[..., :3] = grey[..., None]
    return image


def keyed(values):
    frames = sorted(values)
    return {"value": float(values[frames[0]]), "curve": {"interpolation": "linear",
                                                          "keys": [{"frame": f, "value": float(values[f])} for f in frames]}}


def tracked_node(positions, mode, kind="Tracker", source=None, **params):
    g = Graph()
    g.add("plate", "Ramp", dict(width=24, height=18, p0_x=0.0, p0_y=0.0, p1_x=23.0, p1_y=17.0))
    g.add("node", kind, {"mode": mode, "filter": "bilinear", **params}, image="plate")
    g.d.execute({"op": "set_tracks", "id": "node", "tracks": [{"name": "a", "enabled": 1, "x": keyed({f: p[0] for f, p in positions.items()}),
                                                                "y": keyed({f: p[1] for f, p in positions.items()})}]})
    return g


def display_pixels(g, target, frame):
    raster = Evaluator().evaluate_raster(dict(g.doc, view=target), target=target, frame=frame)
    shown, data = raster.display, raster.data
    canvas = np.zeros((shown.height, shown.width, 4), np.float64)
    both = shown.intersect(data)
    canvas[both.y - shown.y:both.bottom - shown.y, both.x - shown.x:both.right - shown.x] = \
        raster.pixels[both.y - data.y:both.bottom - data.y, both.x - data.x:both.right - data.x]
    return canvas


class TrackerTests(unittest.TestCase):
    def solved(self):
        frames = {f: feature_frame(f) for f in range(1, 7)}
        return analyse(frames, 1, (30.5, 28.5), pattern_radius=8, search_radius=14, last_frame=6)

    def test_one_track_recovers_the_known_subpixel_step_per_frame(self):
        result = self.solved()
        self.assertEqual(sorted(result), [1, 2, 3, 4, 5, 6])
        for frame, found in result.items():
            with self.subTest(frame=frame):
                self.assertAlmostEqual(found[0], 30.5 + VELOCITY[0] * (frame - 1), delta=TRACK_TOLERANCE)
                self.assertAlmostEqual(found[1], 28.5 + VELOCITY[1] * (frame - 1), delta=TRACK_TOLERANCE)
        # the motion adds up: five frames later the track has travelled five steps, to the same tolerance
        self.assertAlmostEqual(result[6][0] - result[1][0], 5 * VELOCITY[0], delta=TRACK_TOLERANCE)
        self.assertAlmostEqual(result[6][1] - result[1][1], 5 * VELOCITY[1], delta=TRACK_TOLERANCE)

    def test_match_move_and_stabilise_apply_and_remove_the_tracked_motion(self):
        result = self.solved()
        positions = {f: result[f][:2] for f in (1, 4)}
        step = (positions[4][0] - positions[1][0], positions[4][1] - positions[1][1])
        plate = display_pixels(tracked_node(positions, "none"), "plate", 4)
        moved = display_pixels(tracked_node(positions, "match_move"), "node", 4)
        removed = display_pixels(tracked_node(positions, "stabilise"), "node", 4)
        ruler_shift = __import__("nuke_fidelity_cases_filter").bilinear_shift
        interior = (slice(5, -5), slice(5, -5), slice(0, 3))
        np.testing.assert_allclose(moved[interior], ruler_shift(plate, *step)[interior], atol=1e-5)
        np.testing.assert_allclose(removed[interior], ruler_shift(plate, -step[0], -step[1])[interior], atol=1e-5)
        # the motion it moved by is the feature's own, to the tracking tolerance
        self.assertAlmostEqual(step[0], 3 * VELOCITY[0], delta=TRACK_TOLERANCE)
        self.assertAlmostEqual(step[1], 3 * VELOCITY[1], delta=TRACK_TOLERANCE)

    def test_stabilise_undoes_what_match_move_did(self):
        positions = {1: (10.0, 8.0), 3: (12.4, 6.9)}
        g = tracked_node(positions, "match_move")
        g.add("back", "Tracker", {"mode": "stabilise", "filter": "bilinear"}, image="node")
        g.d.execute({"op": "set_tracks", "id": "back", "tracks": g.doc["node_data"]["node"]["tracks"]})
        plate = display_pixels(g, "plate", 3)
        again = display_pixels(g, "back", 3)
        interior = (slice(4, -4), slice(4, -4), slice(0, 3))
        np.testing.assert_allclose(again[interior], plate[interior], atol=1e-5)          # a ramp is linear: the two resamples cancel exactly

    def test_a_track_that_has_not_moved_leaves_the_plate_alone(self):
        positions = {1: (10.0, 8.0), 3: (10.0, 8.0)}
        np.testing.assert_array_equal(display_pixels(tracked_node(positions, "match_move"), "node", 3),
                                      display_pixels(tracked_node(positions, "match_move"), "plate", 3))

    def test_the_main_tab_lists_the_knobs_in_nukes_order(self):
        self.assertEqual(labels("Tracker"), [
            "Track channels", "Pre-track filter", "Adjust for luminance changes", "Max error", "Clamp super-white, sub-zero footage",
            "Pattern Radius", "Search Radius", "Show error on track paths", "Hide progress bar", "Snap to markers", "Predict track",
            "Pattern grab behaviour", "Every n frames", "When error above", "When error below", "Update the pattern every frame",
            "Create new key when track is moved", "Auto-tracks delete keyframes", "Transform", "Reference Frame", "Jitter period", "Smoothing",
            "Apply Translate", "Apply Rotate", "Apply Scale", "Live-link transform", "Translate", "Rotate", "Scale", "Skew X", "Skew Y",
            "Skew order", "Center", "Filter", "Clamp", "Black outside", "Motion blur", "Shutter", "Shutter Offset", "Custom shutter offset",
            "Samples"])
        kept, total = layout_order("Tracker")[0]
        self.assertEqual(kept, total)
        sections = [g.section for g in knobs.KNOB_LAYOUT["Tracker"] if g.section]
        self.assertEqual(sections, ["Matching", "Auto-tracking", "Keyframes", "Transform"])

    def test_the_tracker_tab_still_lays_every_param_out_once(self):
        placed = [p for g in knobs.KNOB_LAYOUT["Tracker"] for p in g.params]
        self.assertEqual(len(placed), len(set(placed)))
        self.assertEqual(set(SPECS["Tracker"]["params"]) - set(placed), set())


# ---------------------------------------------------------------------------------------------------------- DropShadow
def shadow(pixels=None, **params):
    pixels = texture(24, 32) if pixels is None else pixels
    return kern("DropShadow", pixels, **params)


def square(n=24, lo=9, hi=15):
    pixels = np.zeros((n, n, 4), np.float32)
    pixels[lo:hi, lo:hi] = (0.8, 0.4, 0.2, 1.0)
    return pixels


def alpha_box(out, threshold=1e-6):
    ys, xs = np.nonzero(out[..., 3] > threshold)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


class DropShadowTests(unittest.TestCase):
    BASE = dict(distance=4.0, shadow_size=0.0, opacity=1.0, red=0.0, green=0.0, blue=0.0)

    def test_angle_180_casts_right_and_0_casts_left(self):
        self.assertEqual(alpha_box(shadow(square(), **self.BASE, angle=180.0)), (9, 9, 18, 14))
        self.assertEqual(alpha_box(shadow(square(), **self.BASE, angle=0.0)), (5, 9, 14, 14))

    def test_angle_turns_the_shadow_one_way_through_vertical(self):
        down = alpha_box(shadow(square(), **self.BASE, angle=270.0))
        up = alpha_box(shadow(square(), **self.BASE, angle=90.0))
        self.assertEqual(sorted([down[3], up[3]]), [14, 18])                 # one casts four rows down, the other four up
        self.assertEqual(sorted([down[1], up[1]]), [5, 9])
        self.assertEqual(down[3] - 14, 4, "270 casts down (the default 225 is down and to the left, and the reading below holds it)")

    def test_distance_moves_the_shadow_that_many_pixels(self):
        for distance in (2.0, 5.0, 7.0):
            with self.subTest(distance=distance):
                box = alpha_box(shadow(square(), **{**self.BASE, "distance": distance}, angle=180.0))
                self.assertEqual(box[2], 14 + int(distance))

    def test_softness_spreads_the_shadow_and_keeps_what_it_carries(self):
        base = {**self.BASE, "distance": 12.0}                               # far enough that the shadow does not fall behind the picture
        hard = shadow(square(40), **base, angle=180.0)
        soft = shadow(square(40), **{**base, "shadow_size": 3.0}, angle=180.0)
        self.assertEqual(alpha_box(hard)[1], 9)
        self.assertLess(alpha_box(soft, 1e-4)[1], 9)                         # the halo reaches above the shadow's top edge (the hard shadow does not)
        shadow_only = lambda out: float((out[..., 3] - square(40)[..., 3]).sum())
        self.assertAlmostEqual(shadow_only(soft), shadow_only(hard), delta=0.01 * shadow_only(hard))
        self.assertTrue(differs(hard, soft))

    def test_opacity_scales_the_shadow_linearly(self):
        full = shadow(square(), **self.BASE, angle=180.0)[..., 3] - square()[..., 3]
        half = shadow(square(), **{**self.BASE, "opacity": 0.5}, angle=180.0)[..., 3] - square()[..., 3]
        np.testing.assert_allclose(half, 0.5 * full, atol=1e-6)

    def test_colour_tints_the_shadow(self):
        out = shadow(square(), **{**self.BASE, "red": 1.0, "green": 0.0, "blue": 0.0}, angle=180.0)
        self.assertAlmostEqual(float(out[11, 17, 0]), 1.0, places=5)         # a shadow-only pixel, 4 to the right of the square's edge
        self.assertAlmostEqual(float(out[11, 17, 1]), 0.0, places=5)

    def test_enable_off_passes_the_input(self):
        pixels = texture(24, 32)
        np.testing.assert_array_equal(shadow(pixels, enable_dropshadow_effect=0), pixels.astype(np.float64))

    def test_a_node_saved_before_the_guides_angle_keeps_its_picture(self):
        pixels = texture(24, 32)
        old = SPECS["DropShadow"]["params"]
        self.assertEqual(old["legacy_dropshadow_angle"], 0)
        self.assertEqual(old["angle"], 225.0)
        # an older node: no flag, the angle counted from the right (-45 was the default, down and to the right)
        older = {k: v for k, v in old.items() if k != "legacy_dropshadow_angle"}
        spec_old = Evaluator._kernel("DropShadow", {**older, "angle": -45.0}, [pixels, None, None])
        spec_new = Evaluator._kernel("DropShadow", {**old, "angle": 225.0}, [pixels, None, None])
        np.testing.assert_array_equal(spec_old, spec_new)                    # the default look is the same shadow
        loaded = old_document("DropShadow", dict(angle=30.0, distance=6.0, shadow_size=2.0), ("legacy_dropshadow_angle",))
        self.assertEqual(loaded["nodes"]["n"]["params"]["legacy_dropshadow_angle"], 1)
        plate = texture(24, 32)                                              # partly clear, so the shadow shows
        a = Evaluator._kernel("DropShadow", loaded["nodes"]["n"]["params"], [plate, None, None])
        b = Evaluator._kernel("DropShadow", {**loaded["nodes"]["n"]["params"], "legacy_dropshadow_angle": 1}, [plate, None, None])
        np.testing.assert_array_equal(a, b)
        c = Evaluator._kernel("DropShadow", {**loaded["nodes"]["n"]["params"], "legacy_dropshadow_angle": 0}, [plate, None, None])
        self.assertTrue(differs(a, c))

    def test_operation_merges_the_shadowed_picture_over_the_bg_input(self):
        g = Graph()
        g.add("fg", "Constant", dict(width=24, height=24, red=0.5, green=0.5, blue=0.5, alpha=0.5))
        g.add("back", "Constant", dict(width=24, height=24, red=0.2, green=0.4, blue=0.8, alpha=1.0))
        g.add("over", "DropShadow", dict(distance=0.0, operation="over"), image="fg", bg="back")
        g.add("plus", "DropShadow", dict(distance=0.0, operation="plus"), image="fg", bg="back")
        g.add("alone", "DropShadow", dict(distance=0.0, operation="plus"), image="fg")
        over, plus, alone = evaluator_pixels(g.doc, "over"), evaluator_pixels(g.doc, "plus"), evaluator_pixels(g.doc, "alone")
        self.assertTrue(differs(over, plus))
        self.assertTrue(differs(over, alone))
        self.assertGreater(float(over[12, 12, 2]), float(alone[12, 12, 2]))              # the blue background shows through the half-clear picture
        g.add("clear", "Constant", dict(width=24, height=24, red=0, green=0, blue=0, alpha=0.0))
        g.add("nothing", "DropShadow", dict(distance=0.0, operation="over"), image="fg", bg="clear")
        np.testing.assert_allclose(evaluator_pixels(g.doc, "nothing"), alone, atol=1e-6)  # over a transparent background leaves the shadowed picture
        # with no background wired the knob has nothing to merge with, and says so in the audit's list of knobs a one-frame probe cannot move
        self.assertIn("operation", requirements.NOT_PIXEL["DropShadow"])
        np.testing.assert_array_equal(evaluator_pixels(g.doc, "alone"),
                                      Evaluator._kernel("DropShadow", {**SPECS["DropShadow"]["params"], "distance": 0.0}, [evaluator_pixels(g.doc, "fg"), None, None]))

    def test_the_tile_path_equals_the_evaluator_in_tiles_of_7_16_and_32(self):
        for masked in (False, True):
            g = Graph()
            g.add("plate", "Checker", dict(width=45, height=37, size=5))
            inputs = {"image": "plate"}
            if masked:
                g.add("matte", "Constant", dict(width=45, height=37, red=1, green=1, blue=1, alpha=0.5))
                inputs["mask"] = "matte"
            g.add("node", "DropShadow", dict(angle=130.0, distance=6.0, shadow_size=3.0, shrink_expand=1.0), **inputs)
            whole = evaluator_pixels(g.doc, "node")
            for edge in (7, 16, 32):
                with self.subTest(edge=edge, masked=masked):
                    np.testing.assert_allclose(tile_pixels(g.doc, "node", edge), whole, atol=1e-6)

    def test_the_main_tab_reads_like_nukes(self):
        self.assertEqual(labels("DropShadow"), ["Enable drop shadow", "Inherit input color", "Color", "Opacity", "Angle", "Distance", "Softness",
                                                "Shrink / expand", "Input"])
        kept, total = layout_order("DropShadow")[0]
        self.assertEqual(kept, total)
        self.assertEqual(layout_order("DropShadow")[2], [])


# ------------------------------------------------------------------------------------------------------------- DirBlur
def dir_blur(pixels, **params):
    return kern("DirBlur", pixels, **params)


def dot(n=33):
    pixels = np.zeros((n, n, 4), np.float32)
    pixels[n // 2, n // 2] = 1.0
    return pixels


class DirBlurTests(unittest.TestCase):
    def test_linear_smear_runs_along_the_angle(self):
        across = dir_blur(dot(), blur_type="linear", length=7.0, angle=0.0)[..., 0]
        down = dir_blur(dot(), blur_type="linear", length=7.0, angle=90.0)[..., 0]
        self.assertEqual(int((across[16] > 1e-9).sum()), 7)
        self.assertEqual(int((across > 1e-9).sum()), 7)
        self.assertEqual(int((down[:, 16] > 1e-9).sum()), 7)
        self.assertAlmostEqual(float(across.sum()), 1.0, places=5)

    def test_a_diagonal_angle_smears_along_the_diagonal(self):
        out = dir_blur(dot(), blur_type="linear", length=9.0, angle=45.0)[..., 0]
        ys, xs = np.mgrid[0:33, 0:33]
        dx, dy = (xs - 16) * out, (16 - ys) * out                              # up is positive: 45 degrees runs up and to the right
        self.assertAlmostEqual(float((dx * (xs - 16)).sum()), float((dy * (16 - ys)).sum()), delta=0.05)       # as far up as across
        self.assertGreater(float((dx * (16 - ys)).sum()), 1.0)                 # and up goes with right

    def test_flat_pictures_stay_flat_for_every_type(self):
        flat = np.full((25, 25, 4), 0.37, np.float32)
        for kind in ("linear", "zoom", "radial"):
            with self.subTest(kind=kind):
                np.testing.assert_allclose(dir_blur(flat, blur_type=kind, length=12.0, angle=20.0, center_x=12.5, center_y=12.5), flat, atol=1e-6)

    def test_zoom_and_radial_leave_the_centre_pixel_and_smear_more_the_further_out(self):
        picture = texture(33, 33)
        for kind in ("zoom", "radial"):
            with self.subTest(kind=kind):
                out = dir_blur(picture, blur_type=kind, length=20.0, angle=30.0, center_x=16.5, center_y=16.5)
                np.testing.assert_allclose(out[16, 16], picture[16, 16], atol=1e-6)
        out = dir_blur(picture, blur_type="zoom", length=20.0, center_x=16.5, center_y=16.5)
        change = np.abs(out - picture)[..., :3].mean(axis=2)
        near = change[13:20, 13:20].mean()
        far = np.concatenate([change[:6].ravel(), change[-6:].ravel()]).mean()
        self.assertGreater(far, 3 * near)                                      # a zoom smears more the further a pixel is from the centre

    def test_zoom_keeps_each_pixel_on_its_ray_from_the_centre(self):
        picture = texture(33, 33)
        other = picture.copy()
        other[:16] = np.roll(other[:16], 3, axis=0)                            # change the rows above the centre row only
        a = dir_blur(picture, blur_type="zoom", length=20.0, center_x=16.5, center_y=16.5)
        b = dir_blur(other, blur_type="zoom", length=20.0, center_x=16.5, center_y=16.5)
        np.testing.assert_allclose(a[16], b[16], atol=1e-6)                    # the centre row's rays never leave the row

    def test_use_target_sets_the_direction_from_blurcenter_to_the_target(self):
        base = dict(blur_type="linear", length=9.0, angle=0.0, center_x=16.5, center_y=16.5)
        plain = dir_blur(dot(), **base)[..., 0]
        vertical = dir_blur(dot(), **base, UseTarget=1, target="16.5,30.5")[..., 0]
        self.assertTrue(differs(plain, vertical))
        self.assertEqual(int((vertical > 1e-9).sum()), int((dir_blur(dot(), **{**base, "angle": 90.0})[..., 0] > 1e-9).sum()))
        np.testing.assert_allclose(vertical, dir_blur(dot(), **{**base, "angle": 90.0})[..., 0], atol=1e-6)
        np.testing.assert_allclose(dir_blur(dot(), **base, UseTarget=0, target="16.5,30.5")[..., 0], plain, atol=0)      # off: the angle decides
        np.testing.assert_allclose(dir_blur(dot(), **base, UseTarget=1, target="")[..., 0], plain, atol=0)               # nothing to aim at

    def test_samples_set_the_number_of_taps_of_a_zoom_and_a_radial_smear(self):
        picture = texture(33, 33)
        for kind in ("zoom", "radial"):
            with self.subTest(kind=kind):
                base = dict(blur_type=kind, length=30.0, angle=40.0, center_x=16.5, center_y=16.5)
                auto, few = dir_blur(picture, **base), dir_blur(picture, **base, Samples=2)
                one = dir_blur(picture, **base, Samples=1)
                self.assertTrue(differs(auto, few))
                np.testing.assert_allclose(one, picture, atol=1e-6)            # one tap on the pixel itself smears nothing
                np.testing.assert_allclose(dir_blur(picture, **base, Samples=0), auto, atol=0)

    def test_pixeloffset_moves_the_zoom_samples_outward(self):
        picture = texture(33, 33)
        base = dict(blur_type="zoom", length=20.0, center_x=16.5, center_y=16.5)
        plain = dir_blur(picture, **base)
        np.testing.assert_allclose(dir_blur(picture, **base, pixeloffset=0.0), plain, atol=0)
        self.assertTrue(differs(plain, dir_blur(picture, **base, pixeloffset=2.0)))
        np.testing.assert_allclose(dir_blur(picture, **base, pixeloffset=2.0)[16, 16], plain[16, 16], atol=1e-5)   # no direction at the centre itself

    def test_the_tile_path_equals_the_evaluator_for_linear_with_a_target(self):
        for masked in (False, True):
            g = Graph()
            g.add("plate", "Checker", dict(width=45, height=37, size=5))
            inputs = {"image": "plate"}
            if masked:
                g.add("matte", "Constant", dict(width=45, height=37, red=1, green=1, blue=1, alpha=0.5))
                inputs["mask"] = "matte"
            g.add("node", "DirBlur", dict(blur_type="linear", length=9.0, center_x=10.0, center_y=8.0, UseTarget=1, target="30,20"), **inputs)
            whole = evaluator_pixels(g.doc, "node")
            for edge in (7, 16, 32):
                with self.subTest(edge=edge, masked=masked):
                    np.testing.assert_allclose(tile_pixels(g.doc, "node", edge), whole, atol=1e-6)

    def test_the_main_tab_reads_like_nukes(self):
        self.assertEqual(labels("DirBlur")[:13], ["Channels", "BlurType", "BlurLayer", "Holdout", "BlurCenter", "BlurLength", "BlurAngle", "Target",
                                             "UseTarget", "Samples", "PixelAspect", "Quality", "Pixel offset"])
        kept, total = layout_order("DirBlur")[0]
        self.assertEqual(kept, total)
        self.assertEqual(layout_order("DirBlur")[2], [])

    def test_the_knobs_that_do_not_act_yet_are_listed_with_what_they_need(self):
        listed = requirements.NOT_PIXEL["DirBlur"]
        for name in ("BlurCenter", "Samples", "pixeloffset"):
            with self.subTest(knob=name):
                self.assertGreater(len(listed[name]), 30)
        # holdout, PixelAspect and Quality have no effect yet: the audit keeps them in its list of knobs that change no pixels
        spec = SPECS["DirBlur"]["params"]
        for name in ("holdout", "PixelAspect", "Quality"):
            self.assertIn(name, spec)


class FilterHelpersTests(unittest.TestCase):
    def test_parse_point(self):
        self.assertEqual(filterbehaviour.parse_point("3, 4.5"), (3.0, 4.5))
        self.assertEqual(filterbehaviour.parse_point([1, 2]), (1.0, 2.0))
        self.assertIsNone(filterbehaviour.parse_point(""))
        self.assertIsNone(filterbehaviour.parse_point("3"))
        self.assertIsNone(filterbehaviour.parse_point("a,b"))


if __name__ == "__main__":
    unittest.main()
