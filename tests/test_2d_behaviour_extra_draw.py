"""2D behaviour, lane 2 step H: Flare, Glint, Sparkles and ScannedGrain compute what the Nuke 17.0 Reference Guide says they compute.

As tools/nuke_fidelity_cases_extra.py and docs/parity/nodebased_to_nuke.json frame them:

* the behaviour cases pass where the guide (or a long-established form) fixes the pixels, and the unscored ones say why;
* every main-tab knob NodeBased pairs with a Nuke knob changes pixels, or is listed in NOT_PIXEL with the state it needs;
* the four nodes stay whole-frame jobs (the tile executor hands their graphs to the full-frame evaluator, as before);
* a document saved before the change renders bit for bit as it did (tests/legacy_behaviour_d.py holds the older kernels);
* each node's main tab reads in the guide's order.
"""
import copy
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from nodebased import core, drawfxbehaviour, knobs
from nodebased.core import LIMITS, SPECS, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor
from tests.legacy_behaviour_d import Legacy
from tests.test_2d_behaviour_filter_a import Graph, differs, texture

import audit_nuke_fidelity as audit  # noqa: E402
import nuke_fidelity_cases as ruler  # noqa: E402

STEP_NODES = ("Flare", "Glint", "Sparkles", "ScannedGrain")
FRAME = 1                                           # the frame `render` evaluates when none is given


def render(document, target):
    return Evaluator().evaluate_raster(dict(document, view=target), target=target).pixels


def black(h=12, w=16):
    image = np.zeros((h, w, 4), np.float32)
    image[..., 3] = 1.0
    return image


def run(kind, slots, **knobs_):
    spec = SPECS[kind]
    slots = list(slots) + [None] * (len(spec["inputs"]) + len(spec.get("optional_inputs", [])) - len(slots))
    return np.asarray(Evaluator._kernel(kind, {**spec["params"], **knobs_}, slots), np.float64)


def saved_before(kind, new_names, **params):
    """A plate-fed node as an older document held it (the step's params removed), loaded."""
    g = Graph()
    g.add("plate", "Checker", dict(width=45, height=37, size=5))
    inputs = {"image": "plate"}
    if kind == "ScannedGrain":
        g.add("grain", "Checker", dict(width=45, height=37, size=3))
        inputs["plate"] = "grain"
    g.add("node", kind, params, **inputs)
    old = copy.deepcopy(g.doc)
    for name in new_names:
        old["nodes"]["node"]["params"].pop(name, None)
    loaded = upgrade_document(old)
    validate(loaded)
    return loaded


def new_names(kind):
    return list(drawfxbehaviour.NEW[kind])


# ------------------------------------------------------------------------------------------------------------- the cases
class CaseTests(unittest.TestCase):
    def test_every_scored_case_of_the_step_passes(self):
        scored = [c for c in ruler.CASES if c.nuke in STEP_NODES and not c.unscored]
        self.assertGreaterEqual(len(scored), 30)
        for case in scored:
            with self.subTest(case=case.id):
                result = audit.run_case(case)
                self.assertEqual(result["status"], "pass", (case.id, result["max_abs_error"], result["worst"], case.statement))

    def test_every_unscored_case_states_its_reason(self):
        for case in [c for c in ruler.CASES if c.nuke in STEP_NODES and c.unscored]:
            with self.subTest(case=case.id):
                self.assertGreater(len(case.unscored), 40, case.id)
                self.assertEqual(audit.run_case(case)["status"], "not scored")

    def test_every_node_of_the_step_has_a_case(self):
        for node in STEP_NODES:
            with self.subTest(node=node):
                self.assertTrue([c for c in ruler.cases_for(node) if not c.unscored], node)

    def test_the_cases_bite(self):
        """Mutating a knob the case sets makes it fail: the cases would catch a kernel that stopped doing what they state."""
        import dataclasses
        for cid, over in (("flare_ring_color_at_the_middle_radius", {"flare_ring_color": 0.7}), ("glint_four_rays", {"length": 5.0}),
                          ("glint_rotation_90", {"rotation": 0.0}), ("sparkles_origin_holds_the_ray_colour", {"spark_color": 0.7}),
                          ("scannedgrain_amount_per_channel", {"grain_red": 0.3}), ("scannedgrain_low_clip_stops_subtracting", {"grain_low_clip": 0.3})):
            case = [c for c in ruler.CASES if c.id == cid][0]
            bad = dataclasses.replace(case, id=cid + "_mutated", nb_params={**case.nb_params, **over})
            with self.subTest(case=cid):
                self.assertEqual(audit.run_case(bad)["status"], "fail")


# --------------------------------------------------------------------------------------------------------------- Flare
def flare(image=None, **knobs_):
    return run("Flare", [black(30, 30) if image is None else image], **{"position_x": 15.0, "position_y": 15.0, "flare_radius_inner": 2.0,
                                                                         "flare_radius_middle": 4.0, "flare_radius_outer": 8.0, **knobs_})


class FlareTests(unittest.TestCase):
    def test_a_new_node_has_the_guides_defaults(self):
        p = SPECS["Flare"]["params"]
        self.assertEqual((p["legacy_flare"], p["flare_radius_inner"], p["flare_radius_middle"], p["flare_radius_outer"], p["flare_size_mult"]), (0, 80.0, 100.0, 120.0, 1.0))
        self.assertEqual((p["flare_ring_color"], p["flare_inner_color"], p["flare_outer_falloff"], p["flare_inner_falloff"], p["flare_chroma_shift"], p["brightness"]),
                         (1.0, 0.5, 1.0, 1.0, 0.0, 1.0))

    def test_the_flare_fades_from_the_inner_colour_through_the_ring_to_black(self):
        row = flare()[15, :, 0]
        radii = np.abs(np.arange(30) - 15)
        inner = row[radii <= 2]
        self.assertTrue(np.allclose(inner, 0.5))                                        # the hole
        rising = [row[15 + r] for r in (2, 3, 4)]
        self.assertTrue(rising[0] < rising[1] < rising[2])                                # inner colour up to the ring colour
        self.assertAlmostEqual(rising[2], 1.0, places=6)
        falling = [row[15 + r] for r in (4, 5, 6, 7, 8)]
        self.assertTrue(all(a > b for a, b in zip(falling, falling[1:])))                 # the ring colour down to black
        self.assertEqual(falling[-1], 0.0)

    def test_the_falloffs_shape_the_fade_but_not_the_landmarks(self):
        base, outer, inner = flare(), flare(flare_outer_falloff=3.0), flare(flare_inner_falloff=3.0)
        for other in (outer, inner):
            for r in (0, 2, 4, 8, 10):                                                    # the hole, the inner radius, the ring, the outer radius, beyond
                self.assertAlmostEqual(float(other[15, 15 + r, 0]), float(base[15, 15 + r, 0]), places=6)
        self.assertLess(float(outer[15, 15 + 6, 0]), float(base[15, 15 + 6, 0]))          # a steeper outer fade is darker between the radii
        self.assertLess(float(inner[15, 15 + 3, 0]), float(base[15, 15 + 3, 0]))

    def test_chroma_shift_moves_red_and_blue_apart_along_the_angle(self):
        out = flare(flare_chroma_shift=3.0)
        xs = np.arange(30)
        centre = lambda c: float((out[15, :, c] * xs).sum() / out[15, :, c].sum())
        self.assertGreater(centre(0), centre(1) + 2.5)
        self.assertLess(centre(2), centre(1) - 2.5)
        up = flare(flare_chroma_shift=3.0, flare_chroma_angle=90.0)
        ys = np.arange(30)
        centre_y = lambda c: float((up[:, 15, c] * ys).sum() / up[:, 15, c].sum())
        self.assertLess(centre_y(0), centre_y(1) - 2.5)                                    # 90 degrees anticlockwise is up the screen

    def test_flare_offset_moves_beyond_position_from_the_image_centre(self):
        at_source = flare(position_x=10.0, flare_offset=0.0)
        moved = flare(position_x=10.0, flare_offset=1.0)
        self.assertTrue(differs(at_source, moved))
        self.assertGreater(float(moved[15, 5, 0]), 0.0)

    def test_anamorph_extends_the_horizontal_radius(self):
        out = flare(flare_anamorph=2.0)
        self.assertGreater(float(out[15, 22, 0]), 0.0)
        self.assertEqual(float(out[24, 15, 0]), 0.0)

    def test_polygon_shape_controls_change_the_flare_outline(self):
        circle = flare(flattening=0.0)
        polygon = flare(flattening=1.0, corners=5)
        self.assertTrue(differs(circle, polygon))
        self.assertTrue(differs(polygon, flare(flattening=1.0, corners=8)))
        self.assertTrue(differs(polygon, flare(flattening=1.0, corners=5, sharpness=2.0)))
        self.assertTrue(differs(polygon, flare(flattening=1.0, corners_angle=30.0)))
        self.assertTrue(differs(flare(position_x=10.0, flattening=1.0),
                                flare(position_x=10.0, flattening=1.0, corners_angle_add=1)))

    def test_asymmetry_controls_dark_sections_around_the_flare(self):
        full = flare(asym_multiplier=1.0)
        dark = flare(asym_multiplier=0.0)
        self.assertTrue(differs(full, dark))
        self.assertTrue(differs(dark, flare(asym_multiplier=0.0, asym_repeat=3)))
        self.assertTrue(differs(dark, flare(asym_multiplier=0.0, asym_falloff=10.0)))
        self.assertTrue(differs(dark, flare(asym_multiplier=0.0, asym_angle=35.0)))

    def test_visibility_range_and_channel_mask_gate_the_flare(self):
        off_centre = dict(position_x=25.0)
        invisible = flare(**off_centre, vis_in=0.1, vis_out=0.3)
        visible = flare(**off_centre, vis_in=0.7, vis_out=0.9)
        self.assertTrue(differs(invisible, visible))
        plate = texture(30, 30)
        unmasked = flare(image=plate)
        masked = flare(image=plate, vis_mask="rgba.alpha", vis_mask_size=0.0)
        softened = flare(image=plate, vis_mask="rgba.alpha", vis_mask_size=4.0)
        self.assertTrue(differs(unmasked, masked))
        self.assertTrue(differs(masked, softened))

    def test_the_flare_is_added_to_the_picture(self):
        image = texture(30, 30)
        np.testing.assert_allclose(flare(image)[..., :3] - image[..., :3], flare()[..., :3] - black(30, 30)[..., :3], atol=1e-5)
        np.testing.assert_array_equal(flare(image)[..., 3], image[..., 3])

    def test_every_paired_knob_changes_the_pixels(self):
        base = flare()
        for name, value in {"position_x": 10.0, "flare_radius_inner": 1.0, "flare_radius_middle": 5.0, "flare_radius_outer": 12.0, "flare_size_mult": 1.5,
                            "flare_ring_color": 0.4, "flare_inner_color": 0.9, "flare_outer_falloff": 2.0, "flare_inner_falloff": 2.0, "flare_chroma_shift": 2.0,
                            "flare_anamorph": 1.5, "flare_chroma_spread": 0.5,
                            "brightness": 2.0}.items():
            with self.subTest(knob=name):
                self.assertTrue(differs(flare(**{name: value}), base), name)
        shifted = flare(flare_chroma_shift=2.0)
        self.assertTrue(differs(flare(flare_chroma_shift=2.0, flare_chroma_angle=45.0), shifted))
        self.assertTrue(differs(flare(position_x=10.0, flare_chroma_shift=2.0, flare_chroma_angle_add=1),
                                flare(position_x=10.0, flare_chroma_shift=2.0)))
        self.assertTrue(differs(flare(position_x=10.0, flare_offset=0.5), flare(position_x=10.0)))

    def test_a_document_saved_before_the_step_renders_as_it_did(self):
        old = dict(position_x=20.0, position_y=15.0, brightness=1.3, streaks=5, length=60.0, rotation=20.0, ghosts=3, spread=0.5, size=9.0, chromatic_shift=3.0)
        loaded = saved_before("Flare", new_names("Flare"), **old)
        self.assertEqual(loaded["nodes"]["node"]["params"]["legacy_flare"], 1)
        image = render(loaded, "plate")
        h, w = image.shape[:2]
        expected = Legacy._flare(image, loaded["nodes"]["node"]["params"], (0, 0), (w, h))
        np.testing.assert_array_equal(render(loaded, "node"), expected)

    def test_the_node_is_a_whole_frame_job_as_before(self):
        g = Graph()
        g.add("plate", "Checker", dict(width=45, height=37, size=5))
        g.add("node", "Flare", {}, image="plate")
        self.assertFalse(TileExecutor(evaluator=Evaluator()).supports_tiled(dict(g.doc, view="node"), "node"))

    def test_the_main_tab_reads_in_the_guides_order(self):
        labels = [group.label for group in knobs.knob_layout("Flare")]
        self.assertEqual(labels[:10], ["Position", "Offset", "Radius", "Size mult", "Anamorph", "Ring color", "Inner color", "Outer falloff", "Inner falloff", "Chroma spread"])
        self.assertEqual(labels[-1], "Mix")


# --------------------------------------------------------------------------------------------------------------- Glint
def dot(n=41):
    image = black(n, n)
    image[n // 2, n // 2, :3] = 1.0
    return image


def glint(image=None, **knobs_):
    return run("Glint", [dot() if image is None else image], **{"tolerance": 0.0, "length": 12.0, "glint_steps": 12, **knobs_})


def lit(out, c=20):
    """The lit pixels of a Glint on `dot()` other than the highlight, as (dx, dy) offsets from it."""
    ys, xs = np.nonzero(out[..., 0] > 1e-6)
    return [(int(x - c), int(y - c)) for x, y in zip(xs, ys) if (x, y) != (c, c)]


class GlintTests(unittest.TestCase):
    def test_a_new_node_has_the_guides_defaults(self):
        p = SPECS["Glint"]["params"]
        self.assertEqual((p["legacy_glint"], p["rays"], p["length"], p["tolerance"], p["rotation"]), (0, 2, 50.0, 0.5, 0.0))
        self.assertEqual((p["glint_oddlen"], p["glint_from_color"], p["glint_to_color"], p["glint_steps"], p["glint_max"], p["glint_effect_only"]), (1.0, 1.0, 0.0, 5, 1, 0))

    def test_n_rays_leave_each_highlight_evenly_spread(self):
        for rays in (2, 3, 5, 6):
            out = glint(rays=rays, glint_steps=12)
            pixels = [(dx, dy) for dx, dy in lit(out) if math.hypot(dx, dy) >= 4]
            expected = [math.radians(360.0 * j / rays) for j in range(rays)]
            for dx, dy in pixels:
                angle = math.atan2(dx, -dy)
                gap = min(abs((angle - e + math.pi) % (2 * math.pi) - math.pi) for e in expected)
                self.assertLess(math.degrees(gap), 20.0, (rays, dx, dy))
            for e in expected:                                                              # and every ray is there
                near = [(dx, dy) for dx, dy in pixels if abs((math.atan2(dx, -dy) - e + math.pi) % (2 * math.pi) - math.pi) < math.radians(20)]
                self.assertTrue(near, (rays, math.degrees(e)))

    def test_positive_rotation_turns_the_star_clockwise(self):
        for turn, expect in ((0.0, lambda dx, dy: dx == 0 and dy < 0), (90.0, lambda dx, dy: dy == 0 and dx > 0), (-90.0, lambda dx, dy: dy == 0 and dx < 0),
                             (180.0, lambda dx, dy: dx == 0 and dy > 0)):
            with self.subTest(rotation=turn):
                pixels = lit(glint(rays=1, rotation=turn))
                self.assertTrue(pixels)
                self.assertTrue(all(expect(dx, dy) for dx, dy in pixels), pixels)

    def test_a_star_turned_by_the_gap_between_its_rays_is_the_same_star(self):
        np.testing.assert_allclose(glint(rays=4, rotation=90.0), glint(rays=4), atol=1e-6)
        self.assertTrue(differs(glint(rays=4, rotation=45.0), glint(rays=4)))

    def test_the_rays_reach_the_length(self):
        for length in (5.0, 9.0, 12.0):
            reach = max(max(abs(dx), abs(dy)) for dx, dy in lit(glint(rays=1, length=length, glint_steps=4, glint_to_color=1.0)))
            self.assertEqual(reach, length)

    def test_the_colour_runs_from_the_first_copy_to_the_last(self):
        out = glint(rays=1, length=4.0, glint_steps=4, glint_from_color=2.0, glint_to_color=0.5)
        np.testing.assert_allclose([out[19 - k, 20, 0] for k in range(4)], [2.0, 1.5, 1.0, 0.5], atol=1e-6)

    def test_max_takes_the_brightest_copy_and_off_adds_them(self):
        kw = dict(rays=1, length=2.0, glint_steps=4)
        on, off = glint(glint_max=1, **kw), glint(glint_max=0, **kw)
        self.assertTrue(differs(on, off))
        self.assertGreaterEqual(float(off[..., 0].max()), float(on[..., 0].max()))

    def test_odd_ray_length_shortens_every_other_ray(self):
        reach = lambda out, axis, sign: max(sign * d[axis] for d in lit(out) if d[1 - axis] == 0 and sign * d[axis] > 0)
        out = glint(rays=4, length=12.0, glint_steps=12, glint_to_color=1.0, glint_oddlen=0.5)
        self.assertEqual(reach(out, 1, -1), 12)                                              # the first ray (up) keeps its length
        self.assertEqual(reach(out, 0, 1), 6)                                                 # the second (right) is half as long

    def test_effect_only_leaves_out_the_highlight(self):
        with_pic, only = glint(rays=1, length=4.0, glint_steps=4), glint(rays=1, length=4.0, glint_steps=4, glint_effect_only=1)
        self.assertEqual(float(with_pic[20, 20, 0]), 1.0)
        self.assertEqual(float(only[20, 20, 0]), 0.0)
        np.testing.assert_allclose(with_pic[19, 20, :3], only[19, 20, :3], atol=1e-6)

    def test_only_what_is_above_the_tolerance_makes_rays(self):
        np.testing.assert_array_equal(glint(tolerance=1.5), dot())
        self.assertTrue(lit(glint(tolerance=0.5)))

    def test_every_paired_knob_changes_the_pixels(self):
        image = texture(24, 32)
        base = run("Glint", [image], tolerance=0.5, length=5.0)
        for name, value in {"rays": 5, "length": 9.0, "glint_oddlen": 2.0, "rotation": 30.0, "glint_from_color": 3.0, "glint_to_color": 1.0, "glint_steps": 3,
                            "glint_max": 0, "glint_effect_only": 1, "tolerance": 1.5, "mix": 0.5}.items():
            with self.subTest(knob=name):
                self.assertTrue(differs(run("Glint", [image], **{"tolerance": 0.5, "length": 5.0, name: value}), base), name)

    def test_the_channels_knob_and_the_mask_strip_limit_the_rays(self):
        for masked in (False, True):
            g = Graph()
            g.add("plate", "Constant", dict(width=41, height=41, red=0.0, green=0.0, blue=0.0, alpha=1.0))
            g.add("dot", "Rectangle", dict(width=41, height=41, box_x=20.0, box_y=20.0, box_width=1.0, box_height=1.0), image="plate")
            inputs = {"image": "dot"}
            if masked:
                g.add("matte", "Constant", dict(width=41, height=41, red=1, green=1, blue=1, alpha=0.5))
                inputs["mask"] = "matte"
            g.add("all", "Glint", dict(tolerance=0.0, length=12.0, glint_steps=12), **inputs)
            g.add("red", "Glint", dict(tolerance=0.0, length=12.0, glint_steps=12, channels="red"), **inputs)
            everything, only_red = render(g.doc, "all"), render(g.doc, "red")
            plain = render(g.doc, "dot")
            with self.subTest(masked=masked):
                self.assertTrue(differs(everything[..., 1], plain[..., 1]))
                np.testing.assert_array_equal(only_red[..., 1], plain[..., 1])
                np.testing.assert_array_equal(only_red[..., 2], plain[..., 2])
                np.testing.assert_allclose(only_red[..., 0], everything[..., 0], atol=1e-6)

    def test_a_document_saved_before_the_step_renders_as_it_did(self):
        old = dict(tolerance=0.5, rays=3, length=9.0, rotation=15.0, falloff=0.5)
        loaded = saved_before("Glint", new_names("Glint") + [], **old)
        self.assertEqual(loaded["nodes"]["node"]["params"]["legacy_glint"], 1)
        image = render(loaded, "plate")
        np.testing.assert_array_equal(render(loaded, "node"), Legacy._glint(image, loaded["nodes"]["node"]["params"]))

    def test_a_saved_node_that_lacks_the_three_changed_defaults_gets_the_older_ones(self):
        loaded = saved_before("Glint", new_names("Glint") + ["tolerance", "rays", "length"])
        p = loaded["nodes"]["node"]["params"]
        self.assertEqual((p["rays"], p["length"], p["tolerance"], p["legacy_glint"]), (4, 12.0, 1.0, 1))

    def test_the_node_is_a_whole_frame_job_as_before(self):
        g = Graph()
        g.add("plate", "Checker", dict(width=45, height=37, size=5))
        g.add("node", "Glint", {}, image="plate")
        self.assertFalse(TileExecutor(evaluator=Evaluator()).supports_tiled(dict(g.doc, view="node"), "node"))

    def test_the_main_tab_reads_in_the_guides_order(self):
        labels = [group.label for group in knobs.knob_layout("Glint")]
        self.assertEqual(labels[:11], ["Channels", "No. of rays", "Length", "Odd ray length", "Rotation", "From color", "To color", "Steps", "Max", "Effect only", "Tolerance"])
        self.assertEqual(labels[-1], "Mix")


# ------------------------------------------------------------------------------------------------------------- Sparkles
def sparkles(image=None, **knobs_):
    return run("Sparkles", [black(41, 41) if image is None else image], **{"spark_x": 20.0, "spark_y": 20.0, "spark_size": 15.0, "spark_ray_nb": 5, "seed": 23, **knobs_})


def along_ray(out, seed, ray, rays, rotation=0.0, steps=14):
    """The values of `out` at integer distances 1..steps along ray `ray`, nearest pixel; the ray's own direction from the seed rule."""
    angle = math.radians(drawfxbehaviour.spark_angles(seed, rays, rotation)[ray])
    return [float(out[int(round(20 - d * math.sin(angle))), int(round(20 + d * math.cos(angle))), 0]) for d in range(1, steps + 1)]


class SparklesTests(unittest.TestCase):
    def test_a_new_node_has_the_guides_defaults(self):
        p = SPECS["Sparkles"]["params"]
        self.assertEqual((p["legacy_sparkles"], p["spark_size"], p["spark_rotation"], p["spark_ray_nb"], p["spark_thick"], p["spark_ray_falloff"]), (0, 300.0, 0.0, 50, 0.2, 2.0))
        self.assertEqual((p["spark_brightness"], p["spark_color"]), (1.0, 1.0))

    def test_the_seed_sets_the_ray_directions(self):
        a, b = drawfxbehaviour.spark_angles(23, 6, 0.0), drawfxbehaviour.spark_angles(23, 6, 0.0)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.allclose(a, drawfxbehaviour.spark_angles(24, 6, 0.0)))
        self.assertTrue(differs(sparkles(seed=1), sparkles(seed=2)))
        np.testing.assert_array_equal(sparkles(seed=7), sparkles(seed=7))
        stratified = drawfxbehaviour.spark_angles(23, 6, 0.0)
        for i, angle in enumerate(stratified):                                                # one ray in each sixth of the circle
            self.assertTrue(60.0 * i <= angle < 60.0 * (i + 1))

    def test_every_ray_leaves_the_position_and_fades_away_from_it(self):
        out = sparkles(spark_ray_nb=1, spark_thick=3.0)
        self.assertAlmostEqual(float(out[20, 20, 0]), 1.0, places=6)
        angle = math.radians(float(drawfxbehaviour.spark_angles(23, 1, 0.0)[0]))
        for d in range(1, 15):
            y, x = int(round(20 - d * math.sin(angle))), int(round(20 + d * math.cos(angle)))
            along = (x - 20) * math.cos(angle) - (y - 20) * math.sin(angle)
            self.assertAlmostEqual(float(out[y, x, 0]), max(0.0, 1.0 - along / 15.0) ** 2, places=5, msg=d)          # the stated fade, 2 being the default falloff

    def test_fireworks_fade_toward_the_origin_and_sparkles_peak_in_the_middle(self):
        rays = sparkles(spark_ray_nb=1, spark_thick=3.0)
        fireworks = sparkles(spark_ray_nb=1, spark_thick=3.0, spark_mode="Fireworks")
        stars = sparkles(spark_ray_nb=1, spark_thick=3.0, spark_mode="Sparkles")
        angle = math.radians(float(drawfxbehaviour.spark_angles(23, 1, 0.0)[0]))
        def sample(image, d):
            y, x = int(round(20 - d * math.sin(angle))), int(round(20 + d * math.cos(angle)))
            return float(image[y, x, 0])
        self.assertGreater(sample(fireworks, 12), sample(fireworks, 2))
        self.assertGreater(sample(stars, 7), sample(stars, 2))
        self.assertGreater(sample(stars, 7), sample(stars, 12))
        self.assertTrue(differs(rays, fireworks))

    def test_triangle_shapes_taper_toward_the_ray_tip(self):
        round_rays = sparkles(spark_ray_nb=1, spark_thick=5.0)
        triangle = sparkles(spark_ray_nb=1, spark_thick=5.0, spark_triangle_think=1)
        self.assertTrue(differs(round_rays, triangle))

    def test_the_rays_end_at_the_size(self):
        out = sparkles(spark_ray_nb=40, spark_size=9.0)
        ys, xs = np.nonzero(out[..., 0] > 1e-6)
        self.assertLessEqual(float(np.hypot(xs - 20, ys - 20).max()), 9.0 + 1e-6)
        self.assertGreater(float(np.hypot(xs - 20, ys - 20).max()), 7.0)

    def test_more_rays_light_more_directions(self):
        count = lambda n: len({round(math.degrees(math.atan2(-(y - 20), x - 20)) / 12.0) for y, x in np.argwhere(sparkles(spark_ray_nb=n, spark_size=14.0)[..., 0] > 0.05)
                               if math.hypot(x - 20, y - 20) > 5})
        self.assertLess(count(2), count(8))
        self.assertEqual(float(sparkles(spark_ray_nb=0)[..., :3].max()), 0.0)

    def test_thicker_rays_light_more_pixels(self):
        lit_pixels = lambda t: int((sparkles(spark_thick=t)[..., 0] > 1e-6).sum())
        self.assertLess(lit_pixels(0.2), lit_pixels(2.0))
        self.assertLess(lit_pixels(2.0), lit_pixels(6.0))

    def test_rotation_turns_the_rays_anticlockwise(self):
        base = drawfxbehaviour.spark_angles(23, 5, 0.0)
        turned = drawfxbehaviour.spark_angles(23, 5, 40.0)
        np.testing.assert_allclose(turned - base, 40.0)
        out = sparkles(spark_rotation=40.0, spark_thick=1.0)
        for ray in range(5):
            self.assertGreater(along_ray(out, 23, ray, 5, rotation=40.0)[2], 0.2)

    def test_relative_rotation_tracks_the_position_direction(self):
        direct = drawfxbehaviour.spark_angles(23, 5, 0.0)
        relative = drawfxbehaviour.spark_angles(23, 5, -45.0)
        self.assertTrue(differs(sparkles(spark_x=30.0, spark_y=30.0, spark_rotation=0.0, spark_relative_rotation=1),
                                sparkles(spark_x=30.0, spark_y=30.0, spark_rotation=0.0)))
        self.assertTrue(np.allclose(relative - direct, -45.0))

    def test_random_thickness_is_seeded_and_changes_ray_shape(self):
        a = sparkles(spark_thick=5.0, spark_random_thick=1)
        b = sparkles(spark_thick=5.0, spark_random_thick=1)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(differs(a, sparkles(spark_thick=5.0)))

    def test_anamorphic_stretches_the_ray_field(self):
        self.assertTrue(differs(sparkles(spark_anamorphic=2.0), sparkles()))

    def test_sparkle_falloff_brings_the_sparkles_closer_to_the_origin(self):
        lower = sparkles(spark_mode="Sparkles", spark_falloff=0.0, spark_thick=3.0)
        higher = sparkles(spark_mode="Sparkles", spark_falloff=0.05, spark_thick=3.0)
        self.assertTrue(differs(lower, higher))
        self.assertGreater(along_ray(higher, 23, 0, 5)[1], along_ray(lower, 23, 0, 5)[1])

    def test_chroma_shift_and_spread_change_the_rgb_ray_field(self):
        base = sparkles(spark_ray_nb=1, spark_thick=3.0)
        self.assertTrue(differs(base, sparkles(spark_ray_nb=1, spark_thick=3.0, spark_chroma_shift=1.5)))
        self.assertTrue(differs(base, sparkles(spark_ray_nb=1, spark_thick=3.0, spark_chroma_spread=0.5)))

    def test_visibility_channel_and_blur_limit_the_effect(self):
        plate = texture(41, 41)
        unmasked = sparkles(image=plate)
        masked = sparkles(image=plate, vismask="rgba.alpha", vismask_size=0.0)
        blurred = sparkles(image=plate, vismask="rgba.alpha", vismask_size=4.0)
        self.assertTrue(differs(unmasked, masked))
        self.assertTrue(differs(masked, blurred))

    def test_broken_ray_settings_remove_seeded_segments_after_the_start(self):
        base = sparkles(spark_ray_nb=8, spark_thick=3.0)
        broken = sparkles(spark_ray_nb=8, spark_thick=3.0, broken_rays=1, broken_affected=100.0,
                          broken_start=0.25, broken_holes=3.0)
        later = sparkles(spark_ray_nb=8, spark_thick=3.0, broken_rays=1, broken_affected=100.0,
                         broken_start=0.65, broken_holes=5.0)
        self.assertTrue(differs(base, broken))
        self.assertTrue(differs(broken, later))

    def test_random_broken_segments_are_repeatable_and_seed_sensitive(self):
        args = dict(spark_ray_nb=8, spark_thick=3.0, broken_rays=1, broken_affected=100.0,
                    broken_start=0.2, broken_holes=4.0, random_broken_start=1, random_broken_holes=1)
        a, b = sparkles(seed=7, **args), sparkles(seed=7, **args)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(differs(a, sparkles(seed=8, **args)))

    def test_sparks_direction_clips_emission_to_the_open_angle(self):
        full = sparkles(spark_ray_nb=12, spark_thick=3.0)
        right = sparkles(spark_ray_nb=12, spark_thick=3.0, sparks=1, sparks_direction=0.0, sparks_angle=45.0)
        left = sparkles(spark_ray_nb=12, spark_thick=3.0, sparks=1, sparks_direction=180.0, sparks_angle=45.0)
        self.assertTrue(differs(full, right))
        self.assertTrue(differs(right, left))

    def test_brightness_and_colour_scale_the_rays(self):
        base = sparkles()
        np.testing.assert_allclose(sparkles(spark_brightness=2.0)[..., :3], 2.0 * base[..., :3] - black(41, 41)[..., :3] * 0.0, atol=1e-5)
        np.testing.assert_allclose(sparkles(spark_color=0.5)[..., :3], 0.5 * base[..., :3], atol=1e-5)
        np.testing.assert_array_equal(sparkles()[..., 3], black(41, 41)[..., 3])

    def test_the_rays_are_added_to_the_picture(self):
        image = texture(41, 41)
        np.testing.assert_allclose(sparkles(image)[..., :3] - image[..., :3], sparkles()[..., :3], atol=1e-5)

    def test_every_paired_knob_changes_the_pixels(self):
        base = sparkles()
        for name, value in {"spark_x": 12.0, "spark_size": 8.0, "spark_rotation": 25.0, "spark_ray_nb": 9, "spark_thick": 3.0, "spark_ray_falloff": 0.5, "seed": 99,
                            "spark_brightness": 2.0, "spark_color": 0.5, "spark_mode": "Fireworks", "spark_triangle_think": 1,
                            "spark_anamorphic": 2.0}.items():
            with self.subTest(knob=name):
                self.assertTrue(differs(sparkles(**{name: value}), base), name)

    def test_a_document_saved_before_the_step_renders_as_it_did(self):
        old = dict(tolerance=0.2, size=2.0, density=0.3, seed=11)
        loaded = saved_before("Sparkles", new_names("Sparkles"), **old)
        self.assertEqual(loaded["nodes"]["node"]["params"]["legacy_sparkles"], 1)
        image = render(loaded, "plate")
        np.testing.assert_array_equal(render(loaded, "node"), Legacy._sparkles(image, loaded["nodes"]["node"]["params"], frame=FRAME))

    def test_the_node_is_a_whole_frame_job_as_before(self):
        g = Graph()
        g.add("plate", "Checker", dict(width=45, height=37, size=5))
        g.add("node", "Sparkles", {}, image="plate")
        self.assertFalse(TileExecutor(evaluator=Evaluator()).supports_tiled(dict(g.doc, view="node"), "node"))

    def test_the_main_tab_reads_in_the_guides_order(self):
        labels = [group.label for group in knobs.knob_layout("Sparkles")]
        self.assertEqual(labels[:28], ["Sparkle type", "Triangle shapes", "Position xy", "Size", "Anamorphic", "Rotation", "Relative rotation", "Rays Nb", "Thickness", "Random thickness", "Sparkle falloff", "Ray falloff", "Seed", "Brightness", "Color", "Chroma shift", "Chroma spread", "Mask", "Mask blur", "Broken rays", "% ray affected", "Break start", "Random start", "Breaks", "Random breaks", "Sparks (direction clip)", "Direction", "Open angle"])
        self.assertEqual(labels[-1], "Mix")


# --------------------------------------------------------------------------------------------------------------- ScannedGrain
def grain_plate(h=12, w=16):
    """A fixed grain plate written here: a hash of the position, no random source, mean about 0.5."""
    y, x = np.mgrid[0:h, 0:w]
    plate = np.zeros((h, w, 4), np.float32)
    for c in range(3):
        plate[..., c] = 0.5 + 0.25 * np.sin(1.7 * x + 2.3 * y + 1.1 * c) * np.cos(0.9 * x - 1.3 * y + 0.7 * c)
    plate[..., 3] = 1.0
    return plate


def scanned(image, plate=None, frame=None, **knobs_):
    spec = SPECS["ScannedGrain"]
    p = {**spec["params"], "seed": 0, **knobs_}
    return np.asarray(Evaluator._scanned_grain(image, grain_plate() if plate is None else plate, p, frame), np.float64)


def level(value, h=12, w=16):
    image = np.zeros((h, w, 4), np.float32)
    image[..., :3] = value
    image[..., 3] = 1.0
    return image


class ScannedGrainTests(unittest.TestCase):
    def test_a_new_node_has_the_guides_defaults(self):
        p = SPECS["ScannedGrain"]["params"]
        self.assertEqual((p["legacy_scannedgrain"], p["fullGrain"], p["grain_offset"], p["grain_red"], p["grain_green"], p["grain_blue"]), (0, "", 0.5, 0.3, 0.4, 0.5))
        self.assertEqual((p["grain_saturation"], p["grain_low_clip"]), (1.0, 0.0))

    def test_the_grain_is_the_plate_minus_the_offset_times_the_amount(self):
        plate = grain_plate()
        out = scanned(level(0.7))
        for c, amount in enumerate((0.3, 0.4, 0.5)):
            np.testing.assert_allclose(out[..., c] - 0.7, (plate[..., c] - 0.5) * amount, atol=1e-6)
        np.testing.assert_array_equal(out[..., 3], 1.0)

    def test_the_same_grain_is_added_at_every_input_level(self):
        lows, highs = scanned(level(0.2)), scanned(level(5.0))
        np.testing.assert_allclose(lows[..., :3] - 0.2, highs[..., :3] - 5.0, atol=1e-5)

    def test_offset_amount_and_saturation_act_as_stated(self):
        image = level(0.8)
        grain = lambda **kw: scanned(image, **kw)[..., :3] - 0.8
        np.testing.assert_allclose(grain(grain_offset=0.4)[..., 0], (grain_plate()[..., 0] - 0.4) * 0.3, atol=1e-6)
        np.testing.assert_allclose(grain(grain_red=1.0, grain_green=0.0, grain_blue=2.0)[..., 1], 0.0, atol=1e-6)
        mono = grain(grain_saturation=0.0, grain_red=1.0, grain_green=1.0, grain_blue=1.0)
        np.testing.assert_allclose(mono[..., 0], mono[..., 1], atol=1e-6)
        np.testing.assert_allclose(mono[..., 1], mono[..., 2], atol=1e-6)
        full = grain(grain_saturation=1.0, grain_red=1.0, grain_green=1.0, grain_blue=1.0)
        boosted = grain(grain_saturation=2.0, grain_red=1.0, grain_green=1.0, grain_blue=1.0)
        self.assertGreater(float(np.abs(boosted[..., 0] - boosted[..., 1]).max()), float(np.abs(full[..., 0] - full[..., 1]).max()))

    def test_low_clip_drops_negative_grain_below_the_threshold(self):
        plate = grain_plate()
        below, above = scanned(level(0.2), grain_low_clip=0.5)[..., :3] - 0.2, scanned(level(0.9), grain_low_clip=0.5)[..., :3] - 0.9
        self.assertGreaterEqual(float(below.min()), 0.0)
        self.assertLess(float(above.min()), 0.0)
        np.testing.assert_allclose(below[..., 0], np.maximum((plate[..., 0] - 0.5) * 0.3, 0.0), atol=1e-6)

    def test_a_different_frame_gives_a_different_grain_and_a_frozen_seed_the_same(self):
        image = level(0.5)
        self.assertTrue(differs(scanned(image, frame=1, seed=0), scanned(image, frame=2, seed=0)))
        np.testing.assert_array_equal(scanned(image, frame=1, seed=4), scanned(image, frame=5, seed=0))      # seed + frame is what moves the plate
        np.testing.assert_array_equal(scanned(image, frame=3, seed=-3), scanned(image, frame=9, seed=-9))     # minus the frame freezes the grain

    def test_the_plate_is_wrapped_to_the_picture(self):
        small = grain_plate(5, 7)
        out = scanned(level(0.5), plate=small)
        self.assertEqual(out.shape, (12, 16, 4))
        np.testing.assert_allclose(out[5, 7, :3], out[0, 0, :3], atol=1e-6)

    def test_every_paired_knob_changes_the_pixels(self):
        image = texture(12, 16)
        image[..., :3] = np.abs(image[..., :3]) * 0.4
        base = scanned(image)
        for name, value in {"grain_offset": 0.3, "grain_red": 1.0, "grain_green": 1.0, "grain_blue": 0.0, "grain_saturation": 0.0, "grain_low_clip": 2.0, "seed": 3}.items():
            with self.subTest(knob=name):
                self.assertTrue(differs(scanned(image, **{name: value}), base), name)

    def test_the_grain_file_path_is_recorded_and_the_plate_input_supplies_the_grain(self):
        self.assertIn("fullGrain", SPECS["ScannedGrain"]["params"])
        self.assertEqual(SPECS["ScannedGrain"]["params"]["fullGrain"], "")
        np.testing.assert_array_equal(scanned(level(0.5), fullGrain="/some/grain.%04d.rgb"), scanned(level(0.5)))

    def test_a_document_saved_before_the_step_renders_as_it_did(self):
        old = dict(amount=1.3, response=0.8, seed=5, preset="8mm", irregularity=0.4, minimum=0.1, apply_through_alpha=1)
        loaded = saved_before("ScannedGrain", new_names("ScannedGrain"), **old)
        p = loaded["nodes"]["node"]["params"]
        self.assertEqual(p["legacy_scannedgrain"], 1)
        image = render(loaded, "plate")
        plate = render(loaded, "grain")
        np.testing.assert_array_equal(render(loaded, "node"), Legacy._scanned_grain(image, plate, p, frame=FRAME))

    def test_the_node_is_a_whole_frame_job_as_before(self):
        g = Graph()
        g.add("plate", "Checker", dict(width=45, height=37, size=5))
        g.add("grain", "Checker", dict(width=45, height=37, size=3))
        g.add("node", "ScannedGrain", {}, image="plate", plate="grain")
        self.assertFalse(TileExecutor(evaluator=Evaluator()).supports_tiled(dict(g.doc, view="node"), "node"))

    def test_the_main_tab_reads_in_the_guides_order(self):
        labels = [group.label for group in knobs.knob_layout("ScannedGrain")]
        self.assertEqual(labels[:6], ["Grain", "Offset", "Amount", "Saturation", "Low clip", "Seed"])
        self.assertEqual(labels[-1], "Mix")


# ------------------------------------------------------------------------------------------------------------ the registry
class RegistryTests(unittest.TestCase):
    def test_every_new_param_has_limits_and_a_panel_row(self):
        for kind in STEP_NODES:
            laid_out = {name for group in knobs.knob_layout(kind) for name in group.params}
            for name in drawfxbehaviour.NEW[kind]:
                with self.subTest(kind=kind, param=name):
                    self.assertIn(name, SPECS[kind]["params"])
                    self.assertIn(name, laid_out)
                    if name in ("spark_mode", "vismask", "vis_mask"):
                        self.assertIn(name, core.CHOICES)
                    elif name != "fullGrain":
                        self.assertIn(name, LIMITS)

    def test_a_new_node_validates_and_a_saved_one_is_flagged_legacy(self):
        for kind in STEP_NODES:
            with self.subTest(kind=kind):
                loaded = saved_before(kind, new_names(kind))
                self.assertEqual(loaded["nodes"]["node"]["params"][drawfxbehaviour.FLAGS[kind]], 1)
                fresh = Graph()
                fresh.add("plate", "Checker", dict(width=8, height=8, size=2))
                fresh.add("node", kind, {}, image="plate")
                self.assertEqual(fresh.doc["nodes"]["node"]["params"][drawfxbehaviour.FLAGS[kind]], 0)

    def test_the_unpaired_older_knobs_have_a_reason(self):
        import json
        pairs = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "parity", "nodebased_to_nuke.json")))["pairs"]
        for kind in STEP_NODES:
            for entry in pairs[kind]["knobs"]:
                if entry["kind"] == "extra":
                    with self.subTest(kind=kind, knob=entry["nodebased"]):
                        self.assertGreater(len(entry.get("reason", "")), 20)


if __name__ == "__main__":
    unittest.main()
