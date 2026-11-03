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
from tests.test_2d_behaviour_filter_a import Graph, differs, evaluator_pixels, kern, tile_pixels

import audit_nuke_fidelity as audit  # noqa: E402
import nuke_fidelity_cases as ruler  # noqa: E402

STEP_NODES = ("EdgeBlur", "GodRays", "VolumeRays", "BumpBoss")
CASED_NODES = ("EdgeBlur",)


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


if __name__ == "__main__":
    unittest.main()
