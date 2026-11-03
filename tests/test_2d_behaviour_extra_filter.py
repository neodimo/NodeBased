"""2D behaviour, lane 2 step G: EdgeBlur, GodRays, VolumeRays and BumpBoss compute what the Nuke 17.0 Reference Guide says they compute.

As tools/nuke_fidelity_cases_extra.py and docs/parity/nodebased_to_nuke.json frame them:

* the behaviour cases pass where the guide (or a long-established form) fixes the pixels, and the unscored ones say why;
* every main-tab knob NodeBased pairs with a Nuke knob changes pixels, or is listed in NOT_PIXEL with the state it needs;
* the kernels this step changed give the same pixels in tiles of 7, 16 and 32 as on the whole frame, with and without a mask;
* a document saved before the change renders bit for bit as it did;
* each node's main tab reads in the guide's order.
"""
import copy
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from nodebased import knobs
from nodebased.core import SPECS, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor
from tests.test_2d_behaviour_filter_a import Graph, differs, evaluator_pixels, kern, texture, tile_pixels

import audit_nuke_fidelity as audit  # noqa: E402
import nuke_fidelity_cases as ruler  # noqa: E402

STEP_NODES = ("EdgeBlur", "GodRays", "VolumeRays", "BumpBoss")
CASED_NODES = ("EdgeBlur", "GodRays")


def render(document, target):
    return Evaluator().evaluate_raster(dict(document, view=target), target=target).pixels


def matte_plate(masked):
    """A checker plate cut by a 20 x 18 rectangle matte, feeding what the tests add, with a half-strength constant mask when asked."""
    g = Graph()
    g.add("plate", "Checker", dict(width=45, height=37, size=5))
    g.add("cut", "Rectangle", dict(width=45, height=37, box_x=12.0, box_y=9.0, box_width=20.0, box_height=18.0))
    g.add("src", "Merge", dict(operation="mask"), A="cut", B="plate")
    inputs = {"image": "src"}
    if masked:
        g.add("matte", "Constant", dict(width=45, height=37, red=1, green=1, blue=1, alpha=0.5))
        inputs["mask"] = "matte"
    return g, inputs


def edge_fixture():
    return ruler.make_fixtures()["E"]


# ------------------------------------------------------------------------------------------------------------- the cases
class CaseTests(unittest.TestCase):
    def test_every_scored_case_of_the_step_passes(self):
        scored = [c for c in ruler.CASES if c.nuke in STEP_NODES and not c.unscored]
        self.assertGreaterEqual(len(scored), 8)
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
        for node in CASED_NODES:
            with self.subTest(node=node):
                self.assertTrue(ruler.cases_for(node), node)


# --------------------------------------------------------------------------------------------------------------- EdgeBlur
class EdgeBlurTests(unittest.TestCase):
    def test_a_new_node_has_the_guides_defaults(self):
        p = SPECS["EdgeBlur"]["params"]
        self.assertEqual((p["edgeblur_size"], p["edge_mult"], p["kernel_filter"], p["quality"], p["crop"], p["edge_matte"]),
                         (3.0, 2.0, "gaussian", 15.0, 1, "rgba.alpha"))
        self.assertEqual((p["edgeblur_tint"], p["edgeblur_brightness"]), (1.0, 1.0))

    def test_the_default_gaussian_band_equals_a_plain_blur_of_the_same_size(self):
        src = edge_fixture()
        edged = kern("EdgeBlur", src, edgeblur_size=2.0, edge_mult=1.0, kernel_filter="gaussian")
        blurred = kern("Blur", src, radius=2.0, kernel_filter="gaussian")
        band = np.zeros(src.shape[:2], bool)
        alpha = src[..., 3]
        for y in range(2, src.shape[0] - 2):
            for x in range(2, src.shape[1] - 2):
                window = alpha[y - 2:y + 3, x - 2:x + 3]
                band[y, x] = window.min() != window.max()
        self.assertTrue(band.any())
        np.testing.assert_allclose(edged[band], blurred[band], atol=1e-6)
        far = np.zeros_like(band)
        far[:, :2] = far[:, -2:] = True
        np.testing.assert_array_equal(edged[far], src[far])

    def test_the_filter_choice_changes_the_blur(self):
        src = edge_fixture()
        shapes = {name: kern("EdgeBlur", src, edgeblur_size=3.0, edge_mult=1.0, kernel_filter=name) for name in ("box", "triangle", "quadratic", "gaussian")}
        for a in shapes:
            for b in shapes:
                if a < b:
                    with self.subTest(a=a, b=b):
                        self.assertTrue(differs(shapes[a], shapes[b]))

    def test_a_lower_edge_mult_narrows_the_band(self):
        src = edge_fixture()
        wide, narrow = kern("EdgeBlur", src, edgeblur_size=2.0, edge_mult=2.0), kern("EdgeBlur", src, edgeblur_size=2.0, edge_mult=0.5)
        changed = lambda out: (np.abs(out - src).max(axis=2) > 1e-6)
        self.assertLess(changed(narrow).sum(), changed(wide).sum())
        self.assertTrue((changed(narrow) <= changed(wide)).all())
        np.testing.assert_array_equal(kern("EdgeBlur", src, edgeblur_size=2.0, edge_mult=0.0), src)

    def test_the_matte_channel_picks_where_the_edges_are(self):
        src = edge_fixture().copy()
        src[..., 0] = src[..., 3]                  # the red channel carries the same matte
        src[..., 3] = 1.0                          # and the alpha has no edge at all
        by_alpha = kern("EdgeBlur", src, edgeblur_size=2.0, edge_mult=1.0, edge_matte="rgba.alpha")
        by_red = kern("EdgeBlur", src, edgeblur_size=2.0, edge_mult=1.0, edge_matte="rgba.red")
        np.testing.assert_array_equal(by_alpha, src)
        self.assertTrue(differs(by_red, src))

    def test_tint_and_brightness_multiply_the_edge_blur_only(self):
        src = edge_fixture()
        base = kern("EdgeBlur", src, edgeblur_size=1.0, edge_mult=1.0, kernel_filter="box")
        tinted = kern("EdgeBlur", src, edgeblur_size=1.0, edge_mult=1.0, kernel_filter="box", edgeblur_tint=0.0)
        bright = kern("EdgeBlur", src, edgeblur_size=1.0, edge_mult=1.0, kernel_filter="box", edgeblur_brightness=0.0)
        edge = np.abs(base - src).max(axis=2) > 1e-6
        self.assertTrue(edge.any())
        np.testing.assert_array_equal(tinted[~edge], src[~edge])
        np.testing.assert_array_equal(bright[~edge], src[~edge])
        np.testing.assert_allclose(tinted[..., :3][edge], 0.0, atol=1e-6)
        np.testing.assert_allclose(tinted[..., 3][edge], base[..., 3][edge], atol=1e-6)
        np.testing.assert_allclose(bright[edge], 0.0, atol=1e-6)

    def test_the_channels_knob_limits_the_blur(self):
        src = edge_fixture()
        colour = kern("EdgeBlur", src, edgeblur_size=2.0, edge_mult=1.0, channels="rgb")
        np.testing.assert_array_equal(colour[..., 3], src[..., 3])
        self.assertTrue(differs(colour[..., :3], src[..., :3]))
        alpha = kern("EdgeBlur", src, edgeblur_size=2.0, edge_mult=1.0, channels="alpha")
        np.testing.assert_array_equal(alpha[..., :3], src[..., :3])
        self.assertTrue(differs(alpha[..., 3], src[..., 3]))

    def test_crop_off_grows_the_window_by_the_blur(self):
        g, inputs = matte_plate(False)
        g.add("node", "EdgeBlur", dict(edgeblur_size=4.0, crop=1), **inputs)
        kept = Evaluator().evaluate_raster(dict(g.doc, view="node"), target="node")
        g.d.execute(dict(op="set", id="node", param="crop", value=0))
        grown = Evaluator().evaluate_raster(dict(g.doc, view="node"), target="node")
        self.assertEqual((kept.data.width, kept.data.height), (45, 37))
        self.assertGreater(grown.data.width, kept.data.width)
        self.assertGreater(grown.data.height, kept.data.height)

    def test_a_document_saved_before_the_step_renders_as_it_did(self):
        g, inputs = matte_plate(False)
        g.add("node", "EdgeBlur", {}, **inputs)
        old = copy.deepcopy(g.doc)
        for name in ("edge_matte", "edgeblur_tint", "edgeblur_brightness", "kernel_filter", "quality", "crop", "edgeblur_size", "edge_mult"):
            old["nodes"]["node"]["params"].pop(name, None)
        loaded = upgrade_document(old)
        validate(loaded)
        p = loaded["nodes"]["node"]["params"]
        self.assertEqual((p["edgeblur_size"], p["edge_mult"], p["kernel_filter"], p["quality"]), (4.0, 1.0, "gaussian", 1000.0))
        src = render(loaded, "src")
        band_alpha = src[..., 3:4]
        band = np.clip(Evaluator._box_extreme(band_alpha, 4.0, use_max=True) - Evaluator._box_extreme(band_alpha, 4.0, use_max=False), 0.0, 1.0)
        blurred = Evaluator._gaussian_2d(src, 4.0)
        expected = src.copy()
        for c in range(4):
            expected[..., c] = src[..., c] + band[..., 0] * (blurred[..., c] - src[..., c])
        np.testing.assert_array_equal(render(loaded, "node"), expected.astype(np.float32))

    def test_tiles_of_7_16_and_32_equal_the_whole_frame_with_and_without_a_mask(self):
        variants = {"default gaussian": dict(), "box": dict(edgeblur_size=3.0, kernel_filter="box", edge_mult=1.0),
                    "scaled by quality": dict(edgeblur_size=9.0, kernel_filter="triangle", quality=2.0, edge_mult=1.0),
                    "wide band": dict(edgeblur_size=2.0, edge_mult=5.0, edgeblur_tint=0.6, edgeblur_brightness=1.3)}
        for label, params in variants.items():
            for masked in (False, True):
                g, inputs = matte_plate(masked)
                g.add("node", "EdgeBlur", params, **inputs)
                whole = evaluator_pixels(g.doc, "node")
                for edge in (7, 16, 32):
                    with self.subTest(variant=label, masked=masked, tile=edge):
                        np.testing.assert_allclose(tile_pixels(g.doc, "node", edge), whole, atol=1e-5)

    def test_the_main_tab_reads_in_the_guides_order(self):
        labels = [group.label for group in knobs.knob_layout("EdgeBlur")]
        self.assertEqual(labels, ["Channels", "Matte input", "Size", "Filter", "Quality", "Crop to format", "Edge mult", "Tint", "Brightness",
                                  "Mask", "Inject", "Invert", "Fringe", "Mix"])


# --------------------------------------------------------------------------------------------------------------- GodRays
def point(x=12, y=6, h=12, w=16):
    out = np.zeros((h, w, 4), np.float32)
    out[y, x] = 1.0
    return out


def rays(image, **knobs):
    return kern("GodRays", image, **{"legacy_godrays": 0, "rays_max": 1, **knobs})


class GodRaysTests(unittest.TestCase):
    def test_a_new_node_has_the_guides_knobs_and_defaults(self):
        p = SPECS["GodRays"]["params"]
        for name, value in (("legacy_godrays", 0), ("steps", 5), ("rays_translate_x", 0.0), ("rays_translate_y", 0.0), ("rays_rotate", 0.0),
                            ("rays_scale", 1.0), ("rays_skew", 0.0), ("rays_from_color", 1.0), ("rays_to_color", 1.0), ("rays_gamma", 1.0), ("rays_max", 0)):
            self.assertEqual(p[name], value, name)

    def test_the_default_node_leaves_the_picture_alone(self):
        pixels = texture()
        np.testing.assert_allclose(kern("GodRays", pixels), pixels, atol=1e-6)

    def test_scale_below_one_pulls_the_pixel_toward_the_centre_and_above_one_pushes_it_out(self):
        base = dict(center_x=4.0, center_y=6.0, steps=2)
        down = rays(point(), rays_scale=0.5, **base)
        up = rays(point(x=6), rays_scale=2.0, **base)
        self.assertEqual({x for x in range(16) if down[6, x, 0] > 0.5}, {8, 10, 12})
        self.assertEqual({x for x in range(16) if up[6, x, 0] > 0.5}, {6, 7, 8})              # 4 + 2 * 1, 4 + 2 * 1.5, 4 + 2 * 2
        self.assertAlmostEqual(float(down[:, :, 0].sum()), 3.0, places=4)                    # nowhere else

    def test_the_average_is_the_mean_of_the_samples_taken_singly(self):
        mean = rays(point(), rays_max=0, center_x=4.0, center_y=6.0, steps=2, rays_scale=0.5)
        for x in (8, 10, 12):
            self.assertAlmostEqual(float(mean[6, x, 0]), 1.0 / 3.0, places=5)                # steps + 1 samples
        self.assertAlmostEqual(float(mean[:, :, 0].sum()), 1.0, places=5)

    def test_translate_moves_the_samples_along_the_vector(self):
        out = rays(point(), rays_translate_x=4.0, steps=2)
        self.assertEqual({x for x in range(16) if out[6, x, 0] > 0.5}, {12, 14})            # the third sample, at x 16, is off the picture
        down = rays(point(), rays_translate_y=3.0, steps=1)
        self.assertEqual({y for y in range(12) if down[y, 12, 0] > 0.5}, {6, 9})

    def test_rotate_turns_the_samples_about_the_centre(self):
        out = rays(point(), rays_rotate=90.0, center_x=8.0, center_y=6.0, steps=1)
        self.assertGreater(float(out[6, 12, 0]), 0.99)
        self.assertGreater(float(out[10, 8, 0]), 0.99)                                       # a quarter turn: 4 right of the centre becomes 4 below it

    def test_skew_shears_the_samples_about_the_centre(self):
        out = rays(point(y=9), rays_skew=1.0, center_x=12.0, center_y=6.0, steps=1)
        self.assertGreater(float(out[9, 12, 0]), 0.99)
        self.assertGreater(float(out[9, 15, 0]), 0.99)                                       # 3 below the centre shears 3 to the right

    def test_the_colours_run_from_the_first_sample_to_the_last_through_the_gamma_space(self):
        plain = rays(point(), center_x=4.0, center_y=6.0, steps=2, rays_scale=0.5, rays_from_color=0.5, rays_to_color=2.0)
        self.assertAlmostEqual(float(plain[6, 12, 0]), 0.5, places=5)
        self.assertAlmostEqual(float(plain[6, 10, 0]), 1.25, places=5)                       # halfway, in linear space
        self.assertAlmostEqual(float(plain[6, 8, 0]), 2.0, places=5)
        self.assertAlmostEqual(float(plain[6, 12, 3]), 1.0, places=5)                        # the alpha is not multiplied
        gamma = rays(point(), center_x=4.0, center_y=6.0, steps=2, rays_scale=0.5, rays_from_color=0.5, rays_to_color=2.0, rays_gamma=2.0)
        self.assertAlmostEqual(float(gamma[6, 10, 0]), ((0.5 ** 0.5 + 2.0 ** 0.5) / 2) ** 2, places=5)
        self.assertAlmostEqual(float(gamma[6, 12, 0]), 0.5, places=5)

    def test_every_paired_knob_changes_the_pixels(self):
        pixels = texture()
        base = kern("GodRays", pixels, center_x=20.0, center_y=15.0, rays_scale=0.8)
        moves = {"steps": 9, "center_x": 25.0, "center_y": 9.0, "rays_translate_x": 5.0, "rays_translate_y": 4.0, "rays_rotate": 15.0,
                 "rays_scale": 1.3, "rays_skew": 0.3, "rays_from_color": 0.4, "rays_to_color": 2.0, "rays_max": 1}
        for name, value in moves.items():
            with self.subTest(knob=name):
                self.assertTrue(differs(kern("GodRays", pixels, **{"center_x": 20.0, "center_y": 15.0, "rays_scale": 0.8, name: value}), base), name)
        shaded = kern("GodRays", pixels, center_x=20.0, center_y=15.0, rays_scale=0.8, rays_from_color=0.4)
        self.assertTrue(differs(kern("GodRays", pixels, center_x=20.0, center_y=15.0, rays_scale=0.8, rays_from_color=0.4, rays_gamma=2.0), shaded))

    def test_a_document_saved_before_the_step_renders_as_it_did(self):
        g = Graph()
        g.add("plate", "Checker", dict(width=45, height=37, size=5))
        g.add("node", "GodRays", dict(center_x=20.0, center_y=15.0, steps=8, decay=0.8, translate=0.9), image="plate")
        old = copy.deepcopy(g.doc)
        for name in list(old["nodes"]["node"]["params"]):
            if name.startswith("rays_") or name == "legacy_godrays":
                del old["nodes"]["node"]["params"][name]
        loaded = upgrade_document(old)
        validate(loaded)
        self.assertEqual(loaded["nodes"]["node"]["params"]["legacy_godrays"], 1)
        image = render(loaded, "plate")
        h, w = image.shape[:2]
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        acc = np.zeros_like(image[..., :3])
        bright = np.maximum(image[..., :3], 0.0)
        for i in range(1, 9):
            t = 0.9 * i / 8
            sx = np.clip(np.rint(xx + (xx - 20.0) * t).astype(int), 0, w - 1)
            sy = np.clip(np.rint(yy + (yy - 15.0) * t).astype(int), 0, h - 1)
            acc += bright[sy, sx] * (0.8 ** i)
        expected = image.copy()
        expected[..., :3] += acc / 8
        np.testing.assert_array_equal(render(loaded, "node"), expected)

    def test_the_node_is_a_whole_frame_job_as_before(self):
        g = Graph()
        g.add("plate", "Checker", dict(width=45, height=37, size=5))
        g.add("node", "GodRays", {}, image="plate")
        self.assertFalse(TileExecutor(evaluator=Evaluator()).supports_tiled(dict(g.doc, view="node"), "node"))

    def test_the_main_tab_reads_in_the_guides_order(self):
        labels = [group.label for group in knobs.knob_layout("GodRays")]
        self.assertEqual(labels[:11], ["Channels", "Translate xy", "Rotate", "Scale", "Skew", "Center", "From color", "To color", "Gamma", "Steps", "Max"])
        self.assertEqual(labels[-1], "Mix")


if __name__ == "__main__":
    unittest.main()
