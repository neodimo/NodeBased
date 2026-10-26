"""Fidelity step E, part 4: the shared mask strip's `fringe` knob. The effect applies only on the soft edge of the mask."""
import copy
import unittest

import numpy as np

from nodebased import ops2d_chrome as chrome
from nodebased.core import SPECS, Dispatcher, empty_document, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.raster import Raster
from nodebased.tileexec import TileExecutor

H, W = 6, 8


def plate(value=0.5, alpha=1.0):
    pixels = np.full((H, W, 4), value, np.float32)
    pixels[..., 3] = alpha
    return Raster.of(pixels)


def matte(values):
    """A mask whose alpha is `values` (an (H, W) array)."""
    pixels = np.zeros((H, W, 4), np.float32)
    pixels[..., 3] = values
    return Raster.of(pixels)


def run(kind, inputs, **knobs):
    return Evaluator._windowed_kernel(kind, {**SPECS[kind]["params"], **knobs}, inputs).pixels


class FringeWeightTests(unittest.TestCase):
    def test_the_weight_is_zero_at_full_and_empty_and_one_at_half(self):
        weights = chrome.fringe_of(np.array([0.0, 0.25, 0.5, 0.75, 1.0, -2.0, 7.0], np.float32))
        np.testing.assert_allclose(weights, [0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0], atol=1e-7)


class FringeEffectTests(unittest.TestCase):
    def ramp(self):
        return np.tile(np.linspace(0.0, 1.0, W, dtype=np.float32), (H, 1))

    def test_a_filter_applies_only_on_the_soft_edge(self):
        alpha = self.ramp()
        source = plate(0.5)
        full = run("Grade", [source, matte(alpha)], multiply=2.0, white_clamp=0, fringe=0)
        edge = run("Grade", [source, matte(alpha)], multiply=2.0, white_clamp=0, fringe=1)
        graded = 1.0                                              # 0.5 * 2
        np.testing.assert_allclose(full[0, :, 0], 0.5 + (graded - 0.5) * alpha[0], atol=1e-6)
        weight = 1.0 - np.abs(2.0 * alpha[0] - 1.0)
        np.testing.assert_allclose(edge[0, :, 0], 0.5 + (graded - 0.5) * weight, atol=1e-6)
        self.assertAlmostEqual(float(edge[0, 0, 0]), 0.5, places=6)       # mask empty: no edge
        self.assertAlmostEqual(float(edge[0, -1, 0]), 0.5, places=6)      # mask full: no edge

    def test_a_hard_mask_has_no_edge_so_the_node_does_nothing(self):
        hard = matte((self.ramp() > 0.5).astype(np.float32))
        source = plate(0.4)
        out = run("Blur", [source, hard], radius=3.0, fringe=1)
        np.testing.assert_array_equal(out, source.pixels)

    def test_mix_scales_the_fringe(self):
        alpha = np.full((H, W), 0.5, np.float32)
        source = plate(0.2)
        out = run("Grade", [source, matte(alpha)], multiply=3.0, fringe=1, mix=0.5)
        np.testing.assert_allclose(out[..., 0], 0.2 + (0.6 - 0.2) * 0.5, atol=1e-6)

    def test_invert_is_applied_before_the_edge_is_taken(self):
        alpha = np.full((H, W), 0.25, np.float32)
        source = plate(0.2)
        plain = run("Grade", [source, matte(alpha)], multiply=3.0, fringe=1)
        inverted = run("Grade", [source, matte(1.0 - alpha)], multiply=3.0, fringe=1, invert_mask=1)
        np.testing.assert_allclose(plain, inverted, atol=1e-6)

    def test_merge_gates_against_b_on_the_edge_only(self):
        a, b = plate(0.8), plate(0.1)
        alpha = np.full((H, W), 0.5, np.float32)
        out = run("Merge", [a, b, matte(alpha)], operation="plus", fringe=1)
        np.testing.assert_allclose(out[..., 0], 0.1 + (0.9 - 0.1) * 1.0, atol=1e-6)
        wide = run("Merge", [a, b, matte(alpha)], operation="plus", fringe=0)
        np.testing.assert_allclose(wide[..., 0], 0.1 + (0.9 - 0.1) * 0.5, atol=1e-6)

    def test_fringe_without_any_mask_changes_nothing(self):
        source = plate(0.5)
        np.testing.assert_array_equal(run("Grade", [source], multiply=2.0, fringe=1), run("Grade", [source], multiply=2.0, fringe=0))

    def test_a_mask_channel_of_the_image_can_have_a_fringe(self):
        source = plate(0.5, alpha=0.5)
        out = run("Grade", [source], multiply=2.0, white_clamp=0, maskChannelInput="rgba.alpha", fringe=1)
        np.testing.assert_allclose(out[..., 0], 0.5 + (1.0 - 0.5) * 1.0, atol=1e-6)

    def test_inject_still_writes_the_whole_mask(self):
        alpha = np.full((H, W), 0.25, np.float32)
        raster = Evaluator._windowed_kernel("Grade", {**SPECS["Grade"]["params"], "fringe": 1, "inject": 1, "multiply": 2.0},
                                            [plate(0.5), matte(alpha)])
        np.testing.assert_allclose(raster.layers["mask"].pixels[..., 3], 0.25, atol=1e-7)


class FringeKnobTests(unittest.TestCase):
    def kinds_with_the_strip(self):
        return [kind for kind, entry in chrome.FILTER_CHROME.items() if entry[1] and kind in SPECS and kind not in chrome.NO_FRINGE]

    def test_every_node_with_the_mask_strip_has_a_fringe_knob_that_is_off(self):
        kinds = self.kinds_with_the_strip()
        self.assertGreater(len(kinds), 40)
        for kind in kinds:
            with self.subTest(kind=kind):
                self.assertEqual(SPECS[kind]["params"]["fringe"], 0)

    def test_the_draw_generators_have_none_because_nukes_do_not(self):
        for kind in chrome.NO_FRINGE:
            with self.subTest(kind=kind):
                self.assertNotIn("fringe", SPECS[kind]["params"])
                self.assertIn("maskChannelInput", SPECS[kind]["params"])

    def test_an_older_document_gets_the_knob_off_and_validates(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "n", "type": "Blur"})
        old = copy.deepcopy(d.document)
        old["nodes"]["n"]["params"].pop("fringe")
        loaded = upgrade_document(old)
        validate(loaded)
        self.assertEqual(loaded["nodes"]["n"]["params"]["fringe"], 0)


class FringeTileTests(unittest.TestCase):
    def test_tiles_equal_the_evaluator(self):
        for kind, extra in (("Grade", {"multiply": 2.0}), ("Blur", {"radius": 3.0}), ("Sharpen", {"sharpen_amount": 1.5}),
                            ("Soften", {"soften_size": 3.0}), ("Invert", {})):
            with self.subTest(kind=kind):
                d = Dispatcher(empty_document())
                d.execute({"op": "create", "id": "src", "type": "Ramp", "params": {
                    "width": 20, "height": 12, "p1_x": 19.0, "color0_red": 0.0, "color1_red": 1.0, "color0_green": 0.2, "color1_green": 0.9}})
                d.execute({"op": "create", "id": "m", "type": "Ramp", "params": {
                    "width": 20, "height": 12, "p1_x": 0.0, "p1_y": 11.0, "color0_alpha": 0.0, "color1_alpha": 1.0}})
                d.execute({"op": "create", "id": "n", "type": kind, "params": {**extra, "fringe": 1, "mix": 0.8}})
                d.execute({"op": "connect", "id": "n", "input": "image", "source": "src"})
                d.execute({"op": "connect", "id": "n", "input": "mask", "source": "m"})
                d.execute({"op": "view", "id": "n"})
                executor = TileExecutor(evaluator=Evaluator(), tile_edge=8)
                expected = Evaluator().evaluate_raster(d.document, target="n").pixels
                region = executor.canvas_region(d.document, "n", frame=1, tier=1)
                np.testing.assert_allclose(executor.compose_region(d.document, "n", region, frame=1, tier=1).pixels, expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
