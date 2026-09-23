"""Lane L2 step 2c1: Erode, Dilate, Median, Sharpen, Glow, Mirror. Pixel assertions against
hand-computed values, evaluator/tile-path parity for the five padded filters (including at tile
seams), mask + mix, bypass and CHOICES/LIMITS coverage. See docs/PARITY_2D.md for the audit these
flip."""
import unittest

import numpy as np

from nodebased.core import CHOICES, Dispatcher, LIMITS, SPECS, bypass_slot
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor, SUPPORTED_TILED_KINDS

GROUP_C1 = ("Erode", "Dilate", "Median", "Sharpen", "Glow", "Mirror")
# Mirror is excluded from the tile path (canvas-origin-dependent, like Transform/Crop).
GROUP_C1_TILED = ("Erode", "Dilate", "Median", "Sharpen", "Glow")


class Graph:
    def __init__(self):
        self.d = Dispatcher()

    def add(self, key, kind, params=None, **inputs):
        self.d.execute(dict(op="create", id=key, type=kind, params=params or {}))
        for slot, source in inputs.items():
            self.d.execute(dict(op="connect", id=key, input=slot, source=source))
        return key

    def bypass(self, key, value=True):
        self.d.execute(dict(op="disable", id=key, value=value))

    @property
    def doc(self):
        return self.d.document


def evaluator_pixels(document, target):
    return Evaluator().evaluate(dict(document, view=target))


def tile_pixels(document, target):
    executor = TileExecutor(evaluator=Evaluator())
    document = dict(document, view=target)
    assert executor.supports_tiled(document, target), target
    region = executor.canvas_region(document, target, frame=1, tier=1)
    return executor.compose_region(document, target, region, frame=1, tier=1).pixels


def _single_white_pixel(size=7):
    # A single white (including alpha) pixel on an otherwise fully black (including alpha) frame.
    frame = np.zeros((size, size, 4), dtype=np.float32)
    mid = size // 2
    frame[mid, mid] = 1.0
    return frame


class KernelPixelTests(unittest.TestCase):
    """Hand-computed pixel assertions, per the lane brief's worked examples."""

    def test_erode_by_one_of_a_single_white_pixel_on_black_is_all_black(self):
        white = _single_white_pixel()
        out = Evaluator._kernel("Erode", {"erode_size": 1.0, "channels": "rgba", "mix": 1.0}, [white])
        np.testing.assert_allclose(out, 0.0)

    def test_dilate_by_one_makes_a_three_by_three_block(self):
        white = _single_white_pixel()
        out = Evaluator._kernel("Dilate", {"dilate_size": 1.0, "channels": "rgba", "mix": 1.0}, [white])
        mid = white.shape[0] // 2
        block = out[mid - 1:mid + 2, mid - 1:mid + 2]
        np.testing.assert_allclose(block, 1.0)
        outside = out.copy()
        outside[mid - 1:mid + 2, mid - 1:mid + 2] = 0.0
        np.testing.assert_allclose(outside[..., :3], 0.0)

    def test_erode_negative_size_dilates_exactly_like_dilate(self):
        white = _single_white_pixel()
        eroded_negative = Evaluator._kernel("Erode", {"erode_size": -1.0, "channels": "rgba", "mix": 1.0}, [white])
        dilated = Evaluator._kernel("Dilate", {"dilate_size": 1.0, "channels": "rgba", "mix": 1.0}, [white])
        np.testing.assert_allclose(eroded_negative, dilated)

    def test_median_of_a_salt_pixel_removes_it(self):
        frame = np.full((5, 5, 4), 0.2, dtype=np.float32)
        frame[..., 3] = 1.0
        frame[2, 2, :3] = 0.9  # an isolated bright "salt" pixel
        out = Evaluator._kernel("Median", {"median_size": 1.0, "channels": "rgb", "mix": 1.0}, [frame])
        np.testing.assert_allclose(out[2, 2, :3], 0.2)

    def test_sharpen_of_a_flat_image_is_identity(self):
        flat = np.full((6, 6, 4), 0.4, dtype=np.float32)
        flat[..., 3] = 1.0
        out = Evaluator._kernel(
            "Sharpen", {"sharpen_amount": 3.0, "sharpen_size": 4.0, "channels": "rgb", "mix": 1.0}, [flat])
        np.testing.assert_allclose(out, flat, atol=1e-6)

    def test_glow_of_a_black_image_is_black(self):
        black = np.zeros((8, 8, 4), dtype=np.float32)
        black[..., 3] = 1.0
        out = Evaluator._kernel(
            "Glow", {"glow_threshold": 1.0, "glow_size": 8.0, "brightness": 2.0,
                    "red": 1.0, "green": 1.0, "blue": 1.0, "channels": "rgb", "mix": 1.0}, [black])
        np.testing.assert_allclose(out, black)

    def test_glow_brightens_a_pixel_above_threshold(self):
        bright = np.zeros((9, 9, 4), dtype=np.float32)
        bright[..., 3] = 1.0
        bright[4, 4, :3] = 3.0
        out = Evaluator._kernel(
            "Glow", {"glow_threshold": 1.0, "glow_size": 4.0, "brightness": 1.0,
                    "red": 1.0, "green": 1.0, "blue": 1.0, "channels": "rgb", "mix": 1.0}, [bright])
        # The glow source pixel itself gains extra brightness on top of the input.
        self.assertGreater(float(out[4, 4, 0]), float(bright[4, 4, 0]))
        # A neighbour that started at zero picks up bloom from the blurred bright region.
        self.assertGreater(float(out[4, 5, 0]), 0.0)

    def test_mirror_horizontal_moves_a_pixel_at_x_to_width_minus_one_minus_x(self):
        width, height = 6, 4
        frame = np.zeros((height, width, 4), dtype=np.float32)
        frame[..., 3] = 1.0
        frame[1, 2] = [1.0, 0.5, 0.25, 1.0]
        out = Evaluator._kernel("Mirror", {"flip_x": 1, "flip_y": 0, "mix": 1.0}, [frame])
        np.testing.assert_allclose(out[1, width - 1 - 2], [1.0, 0.5, 0.25, 1.0])
        self.assertAlmostEqual(float(out[1, 2, 0]), 0.0)

    def test_mirror_vertical_moves_a_pixel_at_y_to_height_minus_one_minus_y(self):
        width, height = 5, 8
        frame = np.zeros((height, width, 4), dtype=np.float32)
        frame[..., 3] = 1.0
        frame[3, 1] = [0.2, 0.4, 0.6, 1.0]
        out = Evaluator._kernel("Mirror", {"flip_x": 0, "flip_y": 1, "mix": 1.0}, [frame])
        np.testing.assert_allclose(out[height - 1 - 3, 1], [0.2, 0.4, 0.6, 1.0])

    def test_mirror_with_no_flip_is_identity(self):
        frame = np.random.default_rng(0).random((4, 5, 4)).astype(np.float32)
        out = Evaluator._kernel("Mirror", {"flip_x": 0, "flip_y": 0, "mix": 1.0}, [frame])
        np.testing.assert_allclose(out, frame)


class MaskMixTests(unittest.TestCase):
    """mix=0 is identity to the untouched source; a zero mask hides the effect."""

    def _single_image_graph(self, kind, params):
        g = Graph()
        g.add("plate", "Constant", dict(red=0.4, green=0.4, blue=0.4, alpha=1.0))
        g.add("matte", "Constant", dict(alpha=0.0))
        g.add("node", kind, params, image="plate", mask="matte")
        return g

    def _params(self, mix):
        return {"Erode": {"erode_size": 3.0, "mix": mix}, "Dilate": {"dilate_size": 3.0, "mix": mix},
               "Median": {"median_size": 3.0, "mix": mix},
               "Sharpen": {"sharpen_amount": 3.0, "sharpen_size": 3.0, "mix": mix},
               "Glow": {"glow_threshold": -1.0, "glow_size": 3.0, "brightness": 2.0, "mix": mix},
               "Mirror": {"flip_x": 1, "mix": mix}}

    def test_kinds_mix_zero_is_identity(self):
        params = self._params(0.0)
        for kind in GROUP_C1:
            with self.subTest(kind=kind):
                g = Graph()
                g.add("plate", "Checker", dict(width=16, height=12, size=4))
                g.add("node", kind, params[kind], image="plate")
                plate = evaluator_pixels(g.doc, "plate")
                out = evaluator_pixels(g.doc, "node")
                np.testing.assert_allclose(out, plate)

    def test_kinds_zero_mask_hides_the_effect(self):
        params = self._params(1.0)
        for kind in GROUP_C1:
            with self.subTest(kind=kind):
                g = self._single_image_graph(kind, params[kind])
                plate = evaluator_pixels(g.doc, "plate")
                out = evaluator_pixels(g.doc, "node")
                np.testing.assert_allclose(out, plate)


class TilePathParityTests(unittest.TestCase):
    """Every padded-filter kind renders byte-identical pixels on the evaluator and the tile path,
    including at tile seams (a canvas larger than one tile forces the tile executor to stitch)."""

    def test_padded_filters_match_across_both_paths(self):
        cases = {
            "Erode": dict(erode_size=2.0, channels="rgba", mix=1.0),
            "Dilate": dict(dilate_size=2.0, channels="rgba", mix=1.0),
            "Median": dict(median_size=2.0, channels="rgba", mix=1.0),
            "Sharpen": dict(sharpen_amount=1.5, sharpen_size=3.0, channels="rgb", mix=1.0),
            "Glow": dict(glow_threshold=0.2, glow_size=5.0, brightness=1.5, channels="rgb", mix=1.0),
        }
        for kind, params in cases.items():
            with self.subTest(kind=kind):
                self.assertIn(kind, SUPPORTED_TILED_KINDS)
                g = Graph()
                g.add("plate", "Checker", dict(width=48, height=32, size=8))
                g.add("matte", "Constant", dict(width=48, height=32, red=1, green=1, blue=1, alpha=0.5))
                g.add("node", kind, params, image="plate", mask="matte")
                ev = evaluator_pixels(g.doc, "node")
                ti = tile_pixels(g.doc, "node")
                np.testing.assert_allclose(ti, ev, atol=1e-6)

    def test_padded_filters_are_seamless_across_multiple_tiles(self):
        # A canvas well larger than the default tile edge forces the executor to compose several
        # tiles; identical output to the full-frame evaluator is the seam assertion.
        cases = {
            "Erode": dict(erode_size=3.0, channels="rgba", mix=1.0),
            "Dilate": dict(dilate_size=3.0, channels="rgba", mix=1.0),
            "Median": dict(median_size=3.0, channels="rgba", mix=1.0),
            "Sharpen": dict(sharpen_amount=2.0, sharpen_size=5.0, channels="rgb", mix=1.0),
            "Glow": dict(glow_threshold=0.1, glow_size=6.0, brightness=1.2, channels="rgb", mix=1.0),
        }
        for kind, params in cases.items():
            with self.subTest(kind=kind):
                g = Graph()
                g.add("plate", "Checker", dict(width=257, height=193, size=11))
                g.add("node", kind, params, image="plate")
                ev = evaluator_pixels(g.doc, "node")
                ti = tile_pixels(g.doc, "node")
                np.testing.assert_allclose(ti, ev, atol=1e-6)

    def test_mirror_is_not_tile_native_and_falls_back(self):
        self.assertNotIn("Mirror", SUPPORTED_TILED_KINDS)


class BypassTests(unittest.TestCase):
    """Every group-c1 kind passes its input untouched when bypassed, on both evaluation paths."""

    def visible_params(self, kind):
        return {"Erode": dict(erode_size=4.0), "Dilate": dict(dilate_size=4.0),
               # A checker cell is 8px; the median window must exceed that to disturb a cell's
               # interior, so this needs a bigger radius than Erode/Dilate need to visibly differ.
               "Median": dict(median_size=9.0), "Sharpen": dict(sharpen_amount=3.0, sharpen_size=4.0),
               "Glow": dict(glow_threshold=-1.0, brightness=2.0), "Mirror": dict(flip_x=1)}[kind]

    def test_bypassed_kind_equals_its_input_on_both_paths(self):
        for kind in GROUP_C1:
            with self.subTest(kind=kind):
                g = Graph()
                g.add("plate", "Checker", dict(width=32, height=24, size=8))
                g.add("half", "Grade", dict(exposure=-1.0), image="plate")
                g.add("node", kind, self.visible_params(kind), image="half")
                upstream = evaluator_pixels(g.doc, "half")
                enabled = evaluator_pixels(g.doc, "node")
                self.assertFalse(np.array_equal(enabled, upstream), f"{kind} must visibly change its input")
                g.bypass("node")
                self.assertTrue(np.array_equal(evaluator_pixels(g.doc, "node"), upstream))
                if kind in GROUP_C1_TILED:
                    self.assertTrue(np.array_equal(tile_pixels(g.doc, "node"), upstream))

    def test_bypass_slot_is_the_single_image_input(self):
        for kind in GROUP_C1:
            with self.subTest(kind=kind):
                node = dict(type=kind, inputs={"image": "x", "mask": None})
                self.assertEqual(bypass_slot(node), "image")


class SpecCoverageTests(unittest.TestCase):
    def test_all_six_nodes_are_registered_with_mask_and_mix(self):
        for kind in GROUP_C1:
            with self.subTest(kind=kind):
                self.assertIn(kind, SPECS)
                self.assertIn("mask", SPECS[kind].get("optional_inputs", []))
                self.assertIn("mix", SPECS[kind]["params"])
                self.assertEqual(LIMITS["mix"], (0, 1))

    def test_choices_and_limits_cover_every_new_ranged_param(self):
        self.assertIn("channels", CHOICES)  # reused from group (b), not re-declared
        for name in ("erode_size", "dilate_size", "median_size", "sharpen_amount", "sharpen_size",
                    "glow_threshold", "glow_size", "brightness", "flip_x", "flip_y"):
            self.assertIn(name, LIMITS)

    def test_dispatcher_creates_every_new_node_with_valid_defaults(self):
        for kind in GROUP_C1:
            with self.subTest(kind=kind):
                d = Dispatcher()
                result = d.execute(dict(op="create", type=kind))
                self.assertIn("id", result["result"])


if __name__ == "__main__":
    unittest.main()
