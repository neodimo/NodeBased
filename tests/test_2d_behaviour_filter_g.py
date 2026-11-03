"""2D behaviour, lane 9 step G: Bilateral, Sharpen and VectorCornerPin take the knobs and inputs Nuke's Reference Guide lists.

* Bilateral: a step edge stays sharp while flat noise smooths; the guide input takes the edges from the guide image; Size is the window,
  Positional Sigma the falloff, Filter gaussian or median; the older reading is kept for saved graphs (legacy_bilateral); the tile path
  equals the evaluator, with the guide and a mask;
* Sharpen: the quality limit acts once the size passes it;
* VectorCornerPin: the user pins, the From quad, the baked To pins and the Output choice act with no SmartVector wired (the pins stay put); graphs
  saved before keep the older corner points; the main tab lists the knobs in Nuke's order. What the vectors add (the pins moving from the reference
  frame to this one) needs vectors with motion, which the fixtures lack; tests/test_smartvector.py runs the node on a SmartVector rig whose motion is
  zero.
"""
import copy
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from nodebased import bilateralknobs, knobs
from nodebased.core import SPECS, upgrade_document, validate
from nodebased.imaging import Evaluator
from tests.test_2d_behaviour_filter_f import layout_order
from tests.test_2d_behaviour_filter_a import Graph, differs, evaluator_pixels, texture, tile_pixels

import audit_nuke_fidelity as audit  # noqa: E402
import nuke_fidelity_cases as ruler  # noqa: E402
import nuke_fidelity_requirements as requirements  # noqa: E402


def labels(kind):
    return [g.label for g in knobs.KNOB_LAYOUT[kind] if not g.params[0].startswith("legacy_") and g.params != ("mix",)]


def bilateral_cases():
    return [c for c in ruler.CASES if c.nuke == "Bilateral" and c.id.startswith("bilateral_")]


def step_picture(edge_at, h=12, w=16):
    """A clean step: 0 left of `edge_at`, 1 from it on, opaque."""
    img = np.zeros((h, w, 4), np.float32)
    img[:, edge_at:, :3] = 1.0
    img[..., 3] = 1.0
    return img


def noise(img, amount=0.03, seed=3):
    rng = np.random.RandomState(seed)
    out = img.copy()
    out[..., :3] += rng.normal(0, amount, img[..., :3].shape).astype(np.float32)
    return out


def bilateral(img, guide=None, **knob):
    return Evaluator._bilateral(img, {**SPECS["Bilateral"]["params"], **knob}, guide)


def legacy_bilateral(image, p):
    """The filter exactly as it was before step G: radius ceil(size), spatial exp(-d^2 / 2 r^2), colour distance over the picture's RGB."""
    import math
    radius = int(math.ceil(max(0.0, float(p.get("spatial_size", 3.0)))))
    if radius == 0:
        return image.copy()
    sigma = max(1e-5, float(p.get("colour_sigma", 0.1)))
    h, w = image.shape[:2]
    pad = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    out = np.zeros_like(image, dtype=np.float64)
    total = np.zeros((h, w, 1), dtype=np.float64)
    center = image.astype(np.float64)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            spatial = math.exp(-(dx * dx + dy * dy) / max(1.0, 2.0 * radius * radius))
            sample = pad[radius + dy:radius + dy + h, radius + dx:radius + dx + w].astype(np.float64)
            diff = sample[..., :3] - center[..., :3]
            weight = spatial * np.exp(-np.sum(diff * diff, axis=2, keepdims=True) / (2 * sigma * sigma))
            out += sample * weight
            total += weight
    return (out / np.maximum(total, 1e-20)).astype(np.float32)


class CaseTests(unittest.TestCase):
    def test_every_scored_case_of_the_node_passes(self):
        scored = [c for c in bilateral_cases() if not c.unscored]
        self.assertGreaterEqual(len(scored), 2)
        for case in scored:
            with self.subTest(case=case.id):
                result = audit.run_case(case)
                self.assertEqual(result["status"], "pass", (case.id, result["max_abs_error"], result["worst"], case.statement))

    def test_every_unscored_case_states_its_reason(self):
        unscored = [c for c in bilateral_cases() if c.unscored]
        self.assertGreaterEqual(len(unscored), 2)
        for case in unscored:
            with self.subTest(case=case.id):
                self.assertGreater(len(case.unscored), 40, case.id)
                self.assertEqual(audit.run_case(case)["status"], "not scored")


class BilateralTests(unittest.TestCase):
    def test_a_step_edge_stays_sharp_while_flat_noise_smooths(self):
        clean = step_picture(8)
        noisy = noise(clean)
        out = bilateral(noisy, spatial_size=5.0, colour_sigma=0.2)
        flat_before = float(np.abs(noisy[:, :5, 0] - clean[:, :5, 0]).mean())
        flat_after = float(np.abs(out[:, :5, 0] - clean[:, :5, 0]).mean())
        self.assertLess(flat_after, flat_before * 0.6)                         # the noise is smoothed
        self.assertGreater(float(out[:, 8, 0].mean()) - float(out[:, 7, 0].mean()), 0.9)      # the edge keeps its full height
        plain = Evaluator._blur(noisy, {"radius": 2.0, "channels": "rgba"})
        self.assertLess(float(plain[:, 8, 0].mean()) - float(plain[:, 7, 0].mean()), 0.7)   # an ordinary blur does not

    def test_size_is_the_window_in_pixels(self):
        img = step_picture(8)
        img[6, 8] = 0.5                                       # one different pixel
        self.assertEqual(bilateralknobs.radius({"spatial_size": 1.0, "legacy_bilateral": 0}), 0)
        self.assertEqual(bilateralknobs.radius({"spatial_size": 3.0, "legacy_bilateral": 0}), 1)
        self.assertEqual(bilateralknobs.radius({"spatial_size": 5.0, "legacy_bilateral": 0}), 2)
        np.testing.assert_array_equal(bilateral(img, spatial_size=1.0), img)
        near = bilateral(img, spatial_size=3.0, colour_sigma=10.0, positional_sigma=1.0)
        # A 3 by 3 window reaches one pixel: pixels two away from the odd one are untouched by it.
        far = bilateral(img, spatial_size=3.0, colour_sigma=10.0, positional_sigma=1.0)[6, 10]
        self.assertTrue(differs(near[6, 9], img[6, 9]))
        np.testing.assert_allclose(far, bilateral(np.where(np.arange(16)[None, :, None] == 8, 1.0, img), spatial_size=3.0, colour_sigma=10.0, positional_sigma=1.0)[6, 10],
                                   atol=1e-6)

    def test_positional_sigma_widens_the_falloff(self):
        img = noise(step_picture(8), 0.1)
        tight = bilateral(img, spatial_size=7.0, colour_sigma=10.0, positional_sigma=0.02)
        wide = bilateral(img, spatial_size=7.0, colour_sigma=10.0, positional_sigma=1.5)
        self.assertLess(float(np.abs(tight - img).max()), 1e-3)            # nearly nothing but the pixel itself counts
        self.assertGreater(float(np.abs(wide - img).mean()), 0.02)

    def test_colour_sigma_lets_blur_cross_an_edge(self):
        img = noise(step_picture(8), 0.01)
        keep = bilateral(img, spatial_size=5.0, colour_sigma=0.05)
        cross = bilateral(img, spatial_size=5.0, colour_sigma=5.0)
        jump = lambda a: float(a[:, 8, 0].mean() - a[:, 7, 0].mean())
        self.assertGreater(jump(keep), 0.95)
        self.assertLess(jump(cross), 0.6)

    def test_median_takes_the_middle_value_and_drops_a_hot_pixel(self):
        img = step_picture(8)
        img[5, 3, :3] = 9.0
        gaussian = bilateral(img, spatial_size=3.0, colour_sigma=20.0, positional_sigma=1.0)
        median = bilateral(img, spatial_size=3.0, colour_sigma=20.0, positional_sigma=1.0, bilateral_filter="median")
        self.assertGreater(float(gaussian[5, 4, 0]), 0.5)                  # the average is dragged up by the hot pixel
        self.assertAlmostEqual(float(median[5, 4, 0]), 0.0, places=5)      # the middle value is not
        self.assertAlmostEqual(float(median[5, 3, 0]), 0.0, places=5)

    def test_the_guide_decides_where_the_edge_is(self):
        # The picture has its step at column 4; the guide has its at column 8. The picture's own edge is no edge to the weights.
        picture = step_picture(4)
        guide = step_picture(8)
        alone = bilateral(picture, spatial_size=5.0, colour_sigma=0.2, positional_sigma=0.6)
        guided = bilateral(picture, guide, spatial_size=5.0, colour_sigma=0.2, positional_sigma=0.6)
        jump = lambda a, x: float(a[:, x, 0].mean() - a[:, x - 1, 0].mean())
        self.assertGreater(jump(alone, 4), 0.99)                           # unguided: the picture's edge stays hard
        self.assertLess(jump(guided, 4), 0.6)                              # guided: it is blurred, nothing in the guide marks it
        # Guide and picture agreeing on the edge: the edge is as hard as ever.
        agreed = bilateral(picture, step_picture(4), spatial_size=5.0, colour_sigma=0.2, positional_sigma=0.6)
        self.assertGreater(jump(agreed, 4), 0.99)

    def test_the_edge_follows_the_guide_across_a_soft_ramp(self):
        h, w = 12, 16
        ramp = np.zeros((h, w, 4), np.float32)
        ramp[..., :3] = np.clip((np.arange(w) - 5) / 6.0, 0, 1)[None, :, None]
        ramp[..., 3] = 1.0
        guide = step_picture(8)
        p = dict(spatial_size=5.0, colour_sigma=0.3, positional_sigma=0.6)
        jump = lambda a: float(a[:, 8, 0].mean() - a[:, 7, 0].mean())
        self.assertGreater(jump(bilateral(ramp, guide, **p)), jump(bilateral(ramp, **p)) + 0.02)

    def test_guide_channels_pick_the_channels_the_weights_read(self):
        picture = noise(step_picture(8), 0.02)
        guide = step_picture(8)
        guide[..., 3] = np.where(np.arange(16)[None, :] < 4, 0.0, 1.0)         # alpha steps at column 4, the colours at 8
        p = dict(spatial_size=5.0, colour_sigma=0.2, positional_sigma=0.6)
        by_colour = bilateral(picture, guide, guide_channels="rgb", **p)
        by_alpha = bilateral(picture, guide, guide_channels="alpha", **p)
        self.assertTrue(differs(by_colour, by_alpha, 1e-3))
        jump = lambda a, x: float(a[:, x, 0].mean() - a[:, x - 1, 0].mean())
        self.assertGreater(jump(by_colour, 8), 0.8)                            # read through the colours, the edge at 8 is hard
        self.assertLess(jump(by_alpha, 8), 0.5)                                # read through the alpha, the colour edge counts for nothing

    def test_every_paired_knob_changes_the_picture(self):
        picture, guide = noise(texture(14, 18), 0.0), step_picture(9, 14, 18)
        base = bilateral(picture, guide)
        for name, value in (("spatial_size", 5.0), ("colour_sigma", 1.5), ("positional_sigma", 1.2), ("bilateral_filter", "median"),
                            ("guide_channels", "alpha")):
            with self.subTest(knob=name):
                self.assertTrue(differs(bilateral(picture, guide, **{name: value}), base), name)

    def test_defaults_are_nukes(self):
        params = SPECS["Bilateral"]["params"]
        self.assertEqual((params["spatial_size"], params["positional_sigma"], params["colour_sigma"], params["bilateral_filter"], params["guide_channels"]),
                         (3.0, 0.4, 0.4, "gaussian", "all"))
        self.assertIn("guide", SPECS["Bilateral"]["optional_inputs"])

    def test_the_main_tab_lists_the_knobs_in_nukes_order(self):
        self.assertEqual(labels("Bilateral")[:5], ["Guide channels", "Size", "Positional sigma", "Colour sigma", "Filter"])
        kept, total = layout_order("Bilateral")[0]
        self.assertEqual(kept, total)
        self.assertEqual(layout_order("Bilateral")[2], [])


class BilateralGraphTests(unittest.TestCase):
    def graph(self, guided=True, masked=False, **params):
        g = Graph()
        g.add("plate", "Checker", dict(width=45, height=37, size=5))
        g.add("shade", "Grade", dict(exposure=-0.5), image="plate")
        inputs = {"image": "shade"}
        if guided:
            g.add("gsrc", "Checker", dict(width=45, height=37, size=9))
            inputs["guide"] = "gsrc"
        if masked:
            g.add("matte", "Constant", dict(width=45, height=37, red=1, green=1, blue=1, alpha=0.5))
            inputs["mask"] = "matte"
        g.add("node", "Bilateral", {"spatial_size": 5.0, "colour_sigma": 0.3, **params}, **inputs)
        return g

    def test_wiring_the_guide_changes_the_picture(self):
        unguided = evaluator_pixels(self.graph(False).doc, "node")
        guided = evaluator_pixels(self.graph(True).doc, "node")
        self.assertTrue(differs(unguided, guided))

    def test_the_tile_path_equals_the_evaluator_with_the_guide_and_a_mask(self):
        for masked in (False, True):
            doc = self.graph(True, masked).doc
            full = evaluator_pixels(doc, "node")
            for edge in (7, 16, 32):
                with self.subTest(masked=masked, edge=edge):
                    np.testing.assert_allclose(tile_pixels(doc, "node", edge), full, atol=1e-6)

    def test_the_tile_path_equals_the_evaluator_for_a_median_without_a_guide(self):
        doc = self.graph(False, False, bilateral_filter="median").doc
        full = evaluator_pixels(doc, "node")
        for edge in (7, 16, 32):
            with self.subTest(edge=edge):
                np.testing.assert_allclose(tile_pixels(doc, "node", edge), full, atol=1e-6)

    def test_a_saved_graph_keeps_its_picture(self):
        g = self.graph(False)
        old = copy.deepcopy(g.doc)
        for name in ("positional_sigma", "bilateral_filter", "guide_channels", "legacy_bilateral"):
            old["nodes"]["node"]["params"].pop(name, None)
        old["nodes"]["node"]["params"]["colour_sigma"] = 0.3
        loaded = upgrade_document(old)
        validate(loaded)
        params = loaded["nodes"]["node"]["params"]
        self.assertEqual(params["legacy_bilateral"], 1)
        source = evaluator_pixels(loaded, "shade")
        np.testing.assert_array_equal(evaluator_pixels(loaded, "node"), legacy_bilateral(source, params))

    def test_a_saved_graph_without_a_colour_sigma_keeps_nukes_older_default(self):
        old = copy.deepcopy(self.graph(False).doc)
        old["nodes"]["node"]["params"].pop("colour_sigma", None)
        old["nodes"]["node"]["params"].pop("legacy_bilateral", None)
        loaded = upgrade_document(old)
        self.assertEqual(loaded["nodes"]["node"]["params"]["colour_sigma"], 0.1)

    def test_denoise_is_unchanged(self):
        image = texture(20, 24)
        out = Evaluator._bilateral(image, {"spatial_size": 3.0, "colour_sigma": 0.2})
        np.testing.assert_array_equal(out, legacy_bilateral(image, {"spatial_size": 3.0, "colour_sigma": 0.2}))


class SharpenQualityTests(unittest.TestCase):
    """The guide's `quailty`: 'the image is scaled down during the calculation, so that the blur filter is no larger than this number, and then
    scaled up with linear interpolation after filtering'. It acts once the size passes it."""

    def sharpen(self, **knob):
        picture = texture(40, 56)
        spec = SPECS["Sharpen"]
        return np.asarray(Evaluator._kernel("Sharpen", {**spec["params"], **knob}, [picture, None]), np.float64)

    def test_quality_acts_once_the_size_passes_it(self):
        self.assertTrue(differs(self.sharpen(sharpen_size=40.0, quality=4.0), self.sharpen(sharpen_size=40.0, quality=15.0)))
        self.assertTrue(differs(self.sharpen(sharpen_size=40.0, quality=15.0), self.sharpen(sharpen_size=40.0, quality=100.0)))

    def test_quality_changes_nothing_while_the_size_fits_within_it(self):
        np.testing.assert_array_equal(self.sharpen(sharpen_size=3.0, quality=4.0), self.sharpen(sharpen_size=3.0, quality=15.0))

    def test_the_default_quality_is_nukes(self):
        self.assertEqual(SPECS["Sharpen"]["params"]["quality"], 15.0)

    def test_the_node_lists_the_knob_with_the_state_it_needs(self):
        self.assertIn("quailty", requirements.NOT_PIXEL["Sharpen"])


# ------------------------------------------------------------------------------------------------------ VectorCornerPin
W, H = 16, 12
FRAME = ((0, 0), (W, 0), (W, H), (0, H))


def pin_params(points, side="user"):
    return {f"{side}{i}_{axis}": float(v) for i, point in enumerate(points, 1) for axis, v in zip("xy", point)}


def pin_graph(vectors=False, **params):
    g = Graph()
    g.add("plate", "Ramp", dict(width=W, height=H, p0_x=0.0, p0_y=0.0, p1_x=15.0, p1_y=11.0))
    inputs = {"image": "plate"}
    g.add("node", "VectorCornerPin", params, **inputs)
    return g


def pinned(**params):
    g = pin_graph(**params)
    return Evaluator().evaluate_raster(dict(g.doc, view="node"), target="node", frame=1).pixels


def plate():
    g = pin_graph()
    return Evaluator().evaluate_raster(dict(g.doc, view="plate"), target="plate", frame=1).pixels


class VectorCornerPinTests(unittest.TestCase):
    def test_default_pins_are_the_identity(self):
        np.testing.assert_array_equal(pinned(), plate())

    def test_user_pins_move_the_picture(self):
        shifted = tuple((x + 3, y + 2) for x, y in FRAME)
        out, src = pinned(**pin_params(shifted)), plate()
        np.testing.assert_allclose(out[:H - 2, 3:], src[2:, :W - 3], atol=1e-6)         # up 2 (rows count down), right 3
        self.assertEqual(float(np.abs(out[H - 2:]).max()), 0.0)                       # nothing was there below

    def test_each_user_pin_changes_the_picture(self):
        base = pinned()
        for i in (1, 2, 3, 4):
            with self.subTest(pin=i):
                moved = pin_params(FRAME)
                moved[f"user{i}_x"] += 2.0
                self.assertTrue(differs(pinned(**moved), base))

    def test_each_from_corner_changes_the_picture(self):
        base = pinned()
        for i in (1, 2, 3, 4):
            with self.subTest(corner=i):
                moved = pin_params(FRAME, "from")
                moved[f"from{i}_y"] += 2.0
                self.assertTrue(differs(pinned(**moved), base))

    def test_the_to_pins_act_once_use_baked_is_on(self):
        shifted = pin_params(tuple((x + 3, y) for x, y in FRAME), "to")
        self.assertFalse(differs(pinned(**shifted), plate()))                      # the vectors' pins (here the user pins) still rule
        self.assertTrue(differs(pinned(use_baked=1, **shifted), plate()))
        np.testing.assert_allclose(pinned(use_baked=1, **shifted)[:, 3:], plate()[:, :W - 3], atol=1e-6)

    def test_the_output_choice_gives_the_picture_or_a_map(self):
        picture = pinned()
        for mode in ("st-map", "st-map inverse"):
            with self.subTest(mode=mode):
                maps = pinned(output_mode=mode)
                self.assertTrue(differs(maps, picture))
                np.testing.assert_allclose(maps[3, 5, :2], [(5 + 0.5) / W, 1 - (3 + 0.5) / H], atol=1e-6)     # identity pins: each pixel its own centre
                self.assertEqual(float(maps[3, 5, 3]), 1.0)

    def test_the_st_map_of_a_move_and_its_inverse_undo_each_other(self):
        shifted = pin_params(tuple((x + 3, y) for x, y in FRAME))
        forward, inverse = pinned(output_mode="st-map", **shifted), pinned(output_mode="st-map inverse", **shifted)
        # Output pixel (x, row) takes source column x - 3; a source column c lands on output column c + 3.
        self.assertAlmostEqual(float(forward[4, 8, 0]), (8 - 3 + 0.5) / W, places=5)
        self.assertAlmostEqual(float(inverse[4, 8, 0]), (8 + 3 + 0.5) / W, places=5)

    def test_a_saved_graph_keeps_its_older_corner_points_and_its_vector_requirement(self):
        g = pin_graph()
        old = copy.deepcopy(g.doc)
        for name in list(old["nodes"]["node"]["params"]):
            if name.startswith(("user", "to", "from")) or name in ("output_mode", "use_baked", "legacy_vectorcornerpin"):
                old["nodes"]["node"]["params"].pop(name)
        loaded = upgrade_document(old)
        validate(loaded)
        params = loaded["nodes"]["node"]["params"]
        self.assertEqual(params["legacy_vectorcornerpin"], 1)
        with self.assertRaises(ValueError):
            Evaluator().evaluate_raster(dict(loaded, view="node"), target="node", frame=1)       # the older node needs a SmartVector

    def test_the_new_node_does_not_need_a_smartvector(self):
        self.assertEqual(SPECS["VectorCornerPin"]["params"]["legacy_vectorcornerpin"], 0)
        self.assertEqual(pinned().shape, (H, W, 4))

    def test_the_main_tab_lists_the_knobs_in_nukes_order(self):
        self.assertEqual(labels("VectorCornerPin")[:15], [
            "Reference Frame", "User1 xy", "User2 xy", "User3 xy", "User4 xy", "To1 xy", "To2 xy", "To3 xy", "To4 xy", "Use baked pins",
            "From1 xy", "From2 xy", "From3 xy", "From4 xy", "Output"])
        kept, total = layout_order("VectorCornerPin")[0]
        self.assertEqual(kept, total)

    def test_the_cases_of_the_node_pass(self):
        cases = [c for c in ruler.CASES if c.nuke == "VectorCornerPin"]
        self.assertGreaterEqual(len([c for c in cases if not c.unscored]), 4)
        for case in cases:
            if case.unscored:
                continue
            with self.subTest(case=case.id):
                result = audit.run_case(case)
                self.assertEqual(result["status"], "pass", (case.id, result["max_abs_error"], result["worst"]))

    def test_the_left_out_knobs_are_listed(self):
        self.assertIn("to1", requirements.NOT_PIXEL["VectorCornerPin"])


if __name__ == "__main__":
    unittest.main()
