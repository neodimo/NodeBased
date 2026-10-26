"""Fidelity step E, part 1: Grade takes Nuke's levels, gamma, reverse and clamps. Pixels by hand, the two ways round (reverse),
tiles against the evaluator, bypass, masks and mix, and old documents."""
import copy
import unittest

import numpy as np

from nodebased import gradeknobs
from nodebased.core import SPECS, Dispatcher, bypass_slot, empty_document, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.raster import Raster
from nodebased.tileexec import TileExecutor
from tests.test_2d_parity_group_c4 import Graph

H, W = 12, 20


def grade(pixels, **knobs):
    return Evaluator._windowed_kernel("Grade", {**SPECS["Grade"]["params"], **knobs}, [Raster.of(np.asarray(pixels, np.float32))]).pixels


def one(*rgba):
    return np.array([[list(rgba)]], np.float32)


class GradeFormulaTests(unittest.TestCase):
    def test_levels_map_the_black_and_white_points_onto_lift_and_gain(self):
        # blackpoint 0.2 -> lift 0.1, whitepoint 0.8 -> gain 0.9: a ramp 0.2 .. 0.8 becomes 0.1 .. 0.9.
        out = grade(one(0.2, 0.5, 0.8, 1.0), blackpoint=0.2, whitepoint=0.8, black=0.1, white=0.9)
        np.testing.assert_allclose(out[0, 0], [0.1, 0.5, 0.9, 1.0], atol=1e-6)

    def test_multiply_and_add_follow_nuke_after_the_levels(self):
        out = grade(one(0.5, 0.5, 0.5, 1.0), multiply=2.0, offset=0.25, black_clamp=0)
        np.testing.assert_allclose(out[0, 0, :3], [1.25] * 3, atol=1e-6)

    def test_gamma_lightens_midtones_and_leaves_zero_and_negatives_alone(self):
        out = grade(one(0.25, 0.0, -0.5, 1.0), gamma=2.0, black_clamp=0)
        np.testing.assert_allclose(out[0, 0], [0.5, 0.0, -0.5, 1.0], atol=1e-6)

    def test_the_clamps(self):
        pixel = one(-0.5, 0.5, 1.5, 0.7)
        np.testing.assert_allclose(grade(pixel, black_clamp=0)[0, 0], [-0.5, 0.5, 1.5, 0.7], atol=1e-7)
        np.testing.assert_allclose(grade(pixel, black_clamp=1)[0, 0], [0.0, 0.5, 1.5, 0.7], atol=1e-7)
        np.testing.assert_allclose(grade(pixel, black_clamp=0, white_clamp=1)[0, 0], [-0.5, 0.5, 1.0, 0.7], atol=1e-7)
        np.testing.assert_allclose(grade(pixel, black_clamp=1, white_clamp=1)[0, 0], [0.0, 0.5, 1.0, 0.7], atol=1e-7)

    def test_a_new_node_clamps_black_by_default_as_nuke_does(self):
        self.assertEqual(SPECS["Grade"]["params"]["black_clamp"], 1)
        self.assertEqual(SPECS["Grade"]["params"]["white_clamp"], 0)
        np.testing.assert_allclose(grade(one(-0.3, 0.2, 0.4, 1.0))[0, 0], [0.0, 0.2, 0.4, 1.0], atol=1e-7)

    def test_reverse_undoes_the_same_grade(self):
        knobs = dict(blackpoint=0.05, whitepoint=1.5, black=0.1, white=0.9, multiply=1.3, offset=0.07, gamma=1.8,
                     exposure=0.4, black_clamp=0)
        rng = np.random.default_rng(3)
        source = rng.uniform(-0.5, 4.0, (H, W, 4)).astype(np.float32)
        source[..., 3] = rng.uniform(0.0, 1.0, (H, W))
        forward = grade(source, **knobs)
        back = grade(forward, **knobs, reverse=1)
        np.testing.assert_allclose(back[..., :3], source[..., :3], rtol=2e-4, atol=2e-4)
        np.testing.assert_array_equal(back[..., 3], source[..., 3])

    def test_the_reverse_of_a_flat_ramp_leaves_the_colour_alone(self):
        out = grade(one(0.3, 0.6, 0.9, 1.0), multiply=0.0, reverse=1, black_clamp=0)
        np.testing.assert_allclose(out[0, 0], [0.3, 0.6, 0.9, 1.0], atol=1e-6)

    def test_a_white_point_equal_to_the_black_point_stays_finite(self):
        out = grade(one(0.1, 0.5, 2.0, 1.0), blackpoint=0.5, whitepoint=0.5)
        self.assertTrue(np.isfinite(out).all())

    def test_alpha_is_never_touched(self):
        out = grade(one(0.2, 0.2, 0.2, 0.37), multiply=3.0, gamma=2.0, white_clamp=1, black=0.2)
        self.assertEqual(float(out[0, 0, 3]), np.float32(0.37))


class GradeKeepsItsOldKnobsTests(unittest.TestCase):
    def test_exposure_is_a_stop_count(self):
        out = grade(one(0.2, 0.4, 0.6, 1.0), exposure=1.0)
        np.testing.assert_allclose(out[0, 0, :3], [0.4, 0.8, 1.2], atol=1e-6)

    def test_exposure_multiply_and_gamma_combine(self):
        out = grade(one(0.25, 0.25, 0.25, 1.0), exposure=1.0, multiply=2.0, gamma=2.0)
        np.testing.assert_allclose(out[0, 0, :3], [1.0, 1.0, 1.0], atol=1e-6)         # (0.25 * 2 * 2) ** 0.5

    def test_an_older_offset_follows_alpha_and_a_new_offset_is_flat(self):
        half = one(0.4, 0.4, 0.4, 0.5)
        np.testing.assert_allclose(grade(half, offset=0.2, offset_by_alpha=1)[0, 0, :3], [0.5] * 3, atol=1e-6)      # 0.4 + 0.5 * 0.2
        np.testing.assert_allclose(grade(half, offset=0.2, offset_by_alpha=0)[0, 0, :3], [0.6] * 3, atol=1e-6)      # 0.4 + 0.2

    def test_levels_with_an_alpha_following_offset(self):
        out = grade(one(0.5, 0.5, 0.5, 0.5), black=0.1, offset=0.2, offset_by_alpha=1, black_clamp=0)
        # slope = (1 - 0.1) = 0.9: 0.5 * 0.9 + 0.1 (lift) + 0.5 * 0.2 (offset scaled by alpha)
        np.testing.assert_allclose(out[0, 0, :3], [0.5 * 0.9 + 0.1 + 0.5 * 0.2] * 3, atol=1e-6)

    def test_the_legacy_arithmetic_runs_byte_for_byte_when_the_new_knobs_are_off(self):
        rng = np.random.default_rng(5)
        source = rng.uniform(-1.0, 3.0, (H, W, 4)).astype(np.float32)
        p = {**SPECS["Grade"]["params"], **gradeknobs.OLD, "exposure": 0.7, "multiply": 1.3, "offset": -0.15}
        expected = source.copy()
        expected[..., :3] = source[..., :3] * np.float32(2.0 ** 0.7) * np.float32(1.3) + source[..., 3:4] * np.float32(-0.15)
        np.testing.assert_array_equal(Evaluator._grade(source, p), Evaluator._grade(source, dict(p)))
        np.testing.assert_allclose(Evaluator._grade(source, p), expected, rtol=1e-6, atol=1e-6)
        self.assertTrue(gradeknobs.is_legacy(p))
        self.assertFalse(gradeknobs.is_legacy({**p, "black_clamp": 1}))


class GradeDocumentTests(unittest.TestCase):
    def test_an_older_document_gets_the_old_values_and_draws_the_same(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "src", "type": "Constant", "params": {"width": W, "height": H, "red": -0.4, "green": 0.3, "blue": 1.6, "alpha": 0.6}})
        d.execute({"op": "create", "id": "n", "type": "Grade", "params": {"exposure": 0.3, "multiply": 1.2, "offset": 0.1}})
        d.execute({"op": "connect", "id": "n", "input": "image", "source": "src"})
        old = copy.deepcopy(d.document)
        for name in gradeknobs.NEW:
            old["nodes"]["n"]["params"].pop(name)
        loaded = upgrade_document(old)
        validate(loaded)
        for name, value in gradeknobs.OLD.items():
            self.assertEqual(loaded["nodes"]["n"]["params"][name], value)
        pixels = Evaluator().evaluate_raster(dict(loaded, view="n"), target="n").pixels
        src = Evaluator().evaluate_raster(dict(loaded, view="n"), target="src").pixels
        expected = src.copy()
        expected[..., :3] = src[..., :3] * np.float32(2.0 ** 0.3) * np.float32(1.2) + src[..., 3:4] * np.float32(0.1)
        np.testing.assert_allclose(pixels, expected, rtol=1e-6, atol=1e-6)
        self.assertLess(float(pixels[0, 0, 0]), 0.0)                # an older Grade never clamped

    def test_every_new_knob_validates_and_has_limits(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "n", "type": "Grade", "params": {"blackpoint": -2.0, "whitepoint": 8.0, "black": -1.0, "white": 12.0,
                                                                        "gamma": 2.2, "reverse": 1, "black_clamp": 0, "white_clamp": 1}})
        validate(d.document)

    def test_bypass_passes_the_image(self):
        g = Graph()
        g.add("src", "Constant", dict(width=W, height=H, red=-0.4, green=0.3, blue=1.6, alpha=0.6))
        g.add("n", "Grade", dict(white_clamp=1, gamma=2.0), image="src")
        g.bypass("n")
        self.assertEqual(bypass_slot(g.doc["nodes"]["n"]), "image")
        a = Evaluator().evaluate(dict(g.doc, view="n"), "n", frame=1)
        b = Evaluator().evaluate(dict(g.doc, view="src"), "src", frame=1)
        np.testing.assert_array_equal(a, b)


class GradeTileTests(unittest.TestCase):
    def compare(self, knobs, with_mask=False, mix=1.0):
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "src", "type": "Ramp", "params": {
            "width": W, "height": H, "p1_x": W - 1.0, "color0_red": -0.5, "color1_red": 3.0, "color0_green": 0.2, "color1_green": 0.9,
            "color0_alpha": 0.4, "color1_alpha": 1.0}})
        d.execute({"op": "create", "id": "n", "type": "Grade", "params": {**knobs, "mix": mix}})
        d.execute({"op": "connect", "id": "n", "input": "image", "source": "src"})
        if with_mask:
            d.execute({"op": "create", "id": "m", "type": "Radial", "params": {"width": W, "height": H}})
            d.execute({"op": "connect", "id": "n", "input": "mask", "source": "m"})
        d.execute({"op": "view", "id": "n"})
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=8)
        expected = Evaluator().evaluate_raster(d.document, target="n").pixels
        region = executor.canvas_region(d.document, "n", frame=1, tier=1)
        result = executor.compose_region(d.document, "n", region, frame=1, tier=1)
        np.testing.assert_array_equal(result.pixels, expected)

    def test_tiles_equal_the_evaluator_for_every_new_knob(self):
        for knobs in ({"blackpoint": 0.1, "whitepoint": 2.0, "black": 0.05, "white": 0.95},
                      {"gamma": 1.9}, {"gamma": 0.6, "reverse": 1, "multiply": 1.4, "offset": 0.1},
                      {"white_clamp": 1, "black_clamp": 0}, {"black_clamp": 1, "exposure": 0.5},
                      {"offset": 0.2, "offset_by_alpha": 1, "gamma": 1.4}):
            with self.subTest(knobs=knobs):
                self.compare(knobs)

    def test_tiles_equal_the_evaluator_with_a_mask_and_a_mix(self):
        self.compare({"gamma": 1.5, "white_clamp": 1}, with_mask=True, mix=0.6)


if __name__ == "__main__":
    unittest.main()
