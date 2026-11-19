"""2D behaviour, lane 9 step X6: VectorBlur's `normalize` (Nuke 17.0 Reference Guide, VectorBlur: normalize, default enabled).

The guide: "When enabled, normalize the results to compensate for tearing (holes appearing behind moving objects) and collisions (objects moving on top of
one another) in the rendered output. Normalizing can exacerbate the effects of tearing and collisions, but it tends to give a softer blur than not
normalizing." The node "generates motion blur by blurring each pixel into a straight line", a scatter. On is the blur NodeBased always did. Off draws
each source pixel along its vector with the energy of the line shared along it and no division by what arrives. The guide gives no formula, so what
is tested is what its two sentences say: a uniform field gives the same picture either way, a hole stays dark and a pile-up is brighter with it off,
and a saved node keeps its picture.

`mv_presets` stays unsupported: the guide says each preset holds "a scale and offset specific to the renderer" and gives no values.
"""
import copy
import json
import os
import unittest

import numpy as np

from nodebased import knobs
from nodebased.core import LIMITS, SPECS, upgrade_document, validate
from nodebased.imaging import Evaluator
from tests.test_2d_behaviour_filter_a import Graph, evaluator_pixels

W, H = 40, 12
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_pairs():
    with open(os.path.join(HERE, "docs", "parity", "nodebased_to_nuke.json")) as handle:
        return json.load(handle)


def plate(g, value=0.6):
    g.add("plate", "Constant", dict(width=W, height=H, red=value, green=value, blue=value, alpha=1.0))


def kern(picture, u, v, **knob_values):
    params = {**SPECS["VectorBlur"]["params"], "vector_scale": 1.0, "vector_alpha": "none", "vector_blur_type": "uniform", **knob_values}
    return np.asarray(Evaluator._vector_blur(picture, u, v, params), np.float64)


def opaque(value=0.6):
    picture = np.full((H, W, 4), value, np.float32)
    picture[..., 3] = 1.0
    return picture


def field(left, right):
    u = np.zeros((H, W), np.float32)
    u[:, :W // 2] = left
    u[:, W // 2:] = right
    return u, np.zeros((H, W), np.float32)


class NormalizeTests(unittest.TestCase):
    def test_the_default_is_on_as_the_guide_says_and_the_limits_are_a_check_box(self):
        self.assertEqual(SPECS["VectorBlur"]["params"]["vector_normalize"], 1)
        self.assertEqual(LIMITS["vector_normalize"], (0, 1))

    def test_a_uniform_field_gives_the_same_picture_on_or_off(self):
        ramp = opaque()
        ramp[..., :3] = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :, None]
        u, v = field(3.0, 3.0)
        on = kern(ramp, u, v, vector_normalize=1)
        off = kern(ramp, u, v, vector_normalize=0)
        np.testing.assert_allclose(off[:, 8:W - 8, :], on[:, 8:W - 8, :], atol=1e-5)

    def test_a_hole_stays_dark_and_transparent_with_it_off(self):
        # the right half moves right by 6, the left half stays: the columns at the seam are left by the moving pixels, and only each pixel's
        # own first sample (one in seven) lands on them
        u, v = field(0.0, 6.0)
        on = kern(opaque(), u, v, vector_normalize=1)
        off = kern(opaque(), u, v, vector_normalize=0)
        seam = W // 2
        self.assertLess(float(off[:, seam, 3].mean()), 0.2)               # a hole behind the moving pixels
        self.assertGreater(float(on[:, seam, 3].mean()), 0.99)            # normalizing fills it
        self.assertLess(float(off[:, seam, 0].mean()), float(on[:, seam, 0].mean()))

    def test_a_pile_up_is_brighter_with_it_off(self):
        # the left half moves right by 6 into the right half, which stays: the lines of the left half land on top of the right half's own pixels
        u, v = field(6.0, 0.0)
        off = kern(opaque(), u, v, vector_normalize=0)
        seam = off[:, W // 2:W // 2 + 4, 3]
        self.assertGreater(float(seam.max()), 1.05)                       # collisions: more than one pixel's energy arrives
        on = kern(opaque(), u, v, vector_normalize=1)
        self.assertLess(float(on[:, W // 2:W // 2 + 4, 3].max()), 1.0001)

    def test_the_energy_of_a_picture_is_kept_when_nothing_leaves_it(self):
        picture = np.zeros((H, W, 4), np.float32)
        picture[4:8, 10:14] = 1.0
        u, v = field(5.0, 5.0)
        off = kern(picture, u, v, vector_normalize=0)
        self.assertAlmostEqual(float(off[..., 0].sum()), float(picture[..., 0].sum()), places=3)

    def test_soft_lines_widen_the_streak_with_it_off(self):
        picture = np.zeros((H, W, 4), np.float32)
        picture[6, 10] = 1.0
        u, v = field(5.0, 5.0)
        thin = kern(picture, u, v, vector_normalize=0, soft_lines=0)
        soft = kern(picture, u, v, vector_normalize=0, soft_lines=1, blur_inside=0)
        self.assertEqual(int((thin[..., 0] > 1e-6).sum(axis=0).max()), 1)
        self.assertGreater(int((soft[..., 0] > 1e-6).sum(axis=0).max()), 1)

    def test_the_gaussian_profile_weights_the_middle_of_the_line_with_it_off(self):
        picture = np.zeros((H, W, 4), np.float32)
        picture[6, 10] = 1.0
        u, v = field(8.0, 8.0)
        flat = kern(picture, u, v, vector_normalize=0, vector_blur_type="uniform")[6, :, 0]
        bell = kern(picture, u, v, vector_normalize=0, vector_blur_type="gaussian", motion_falloff=0.2)[6, :, 0]
        self.assertAlmostEqual(float(flat.sum()), float(bell.sum()), places=4)
        self.assertGreater(float(bell[10:19].max()), float(flat[10:19].max()))

    def test_backward_and_offset_still_steer_the_line(self):
        picture = np.zeros((H, W, 4), np.float32)
        picture[6, 20] = 1.0
        u, v = field(4.0, 4.0)
        forward = kern(picture, u, v, vector_normalize=0, vector_method="forward")[6, :, 0]
        backward = kern(picture, u, v, vector_normalize=0, vector_method="backward")[6, :, 0]
        self.assertGreater(float(forward[20:25].sum()), 0.99)
        self.assertGreater(float(backward[16:21].sum()), 0.99)


class NodeTests(unittest.TestCase):
    def graph(self, **knob):
        g = Graph()
        plate(g)
        g.add("vec", "Constant", dict(width=W, height=H, red=4.0, green=0.0, blue=0.0, alpha=1.0))
        g.add("node", "VectorBlur", dict(knob), image="plate", uv="vec")
        return g

    def test_the_node_renders_both_ways_and_a_saved_node_keeps_its_picture(self):
        g = self.graph(vector_normalize=0)
        off = evaluator_pixels(g.doc, "node")
        self.assertEqual(off.shape[:2], (H, W))
        saved = self.graph()
        before = evaluator_pixels(saved.doc, "node")
        old = copy.deepcopy(saved.doc)
        old["nodes"]["node"]["params"].pop("vector_normalize")
        loaded = upgrade_document(old)
        validate(loaded)
        self.assertEqual(loaded["nodes"]["node"]["params"]["vector_normalize"], 1)
        np.testing.assert_array_equal(evaluator_pixels(loaded, "node"), before)

    def test_the_panel_lists_normalize_after_blur_inside(self):
        labels = [g.label for g in knobs.knob_layout("VectorBlur")]
        self.assertEqual(labels[labels.index("Blur inside") + 1], "Normalize")

    def test_mv_presets_stays_unsupported_with_its_reason(self):
        pairs = _load_pairs()["pairs"]["VectorBlur"]["knobs"]
        entry = [e for e in pairs if e.get("nuke") == ["mv_presets"]][0]
        self.assertEqual(entry["kind"], "unsupported")
        self.assertIn("gives no values", entry["reason"])
        self.assertNotIn("mv_presets", SPECS["VectorBlur"]["params"])


if __name__ == "__main__":
    unittest.main()
