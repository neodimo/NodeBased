"""Lane L2 step 2b: Invert, Clamp, Multiply, Add, Gamma, Saturation, Dissolve, Keymix, Copy,
ChannelMerge. Pixel assertions against hand-computed values, evaluator/tile-path parity, mask +
mix, bypass and CHOICES/LIMITS coverage. See docs/PARITY_2D.md for the audit these flip."""
import unittest

import numpy as np

from nodebased.core import CHOICES, Dispatcher, LIMITS, SPECS, bypass_slot
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor, SUPPORTED_TILED_KINDS

GROUP_B_SINGLE = ("Invert", "Clamp", "Multiply", "Add", "Gamma", "Saturation")
GROUP_B_MERGE = ("Dissolve", "Keymix", "Copy", "ChannelMerge")


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


class KernelPixelTests(unittest.TestCase):
    """Hand-computed pixel assertions, per the lane brief's worked examples."""

    def test_invert_of_quarter_is_three_quarters_alpha_untouched_by_default(self):
        rgba = np.array([[[0.25, 0.25, 0.25, 0.6]]], dtype=np.float32)
        out = Evaluator._kernel("Invert", {"channels": "rgb", "mix": 1.0}, [rgba])
        np.testing.assert_allclose(out[0, 0, :3], 0.75)
        self.assertAlmostEqual(float(out[0, 0, 3]), 0.6)

    def test_invert_rgba_also_inverts_alpha(self):
        rgba = np.array([[[0.25, 0.25, 0.25, 0.6]]], dtype=np.float32)
        out = Evaluator._kernel("Invert", {"channels": "rgba", "mix": 1.0}, [rgba])
        np.testing.assert_allclose(out[0, 0], [0.75, 0.75, 0.75, 0.4], atol=1e-6)

    def test_clamp_squashes_out_of_range_values(self):
        rgba = np.array([[[-0.5, 1.5, 0.5, 1.0]]], dtype=np.float32)
        out = Evaluator._kernel(
            "Clamp", {"minimum": 0.0, "maximum": 1.0, "clamp_min": 1, "clamp_max": 1,
                     "channels": "rgb", "mix": 1.0}, [rgba])
        np.testing.assert_allclose(out[0, 0, :3], [0.0, 1.0, 0.5])

    def test_clamp_can_disable_either_bound(self):
        rgba = np.array([[[-0.5, 1.5, 0.5, 1.0]]], dtype=np.float32)
        out = Evaluator._kernel(
            "Clamp", {"minimum": 0.0, "maximum": 1.0, "clamp_min": 0, "clamp_max": 1,
                     "channels": "rgb", "mix": 1.0}, [rgba])
        self.assertAlmostEqual(float(out[0, 0, 0]), -0.5)
        self.assertAlmostEqual(float(out[0, 0, 1]), 1.0)

    def test_multiply_scales_selected_channels(self):
        rgba = np.array([[[0.2, 0.4, 0.6, 1.0]]], dtype=np.float32)
        out = Evaluator._kernel("Multiply", {"multiply": 2.0, "channels": "rgb", "mix": 1.0}, [rgba])
        np.testing.assert_allclose(out[0, 0], [0.4, 0.8, 1.2, 1.0], atol=1e-6)

    def test_add_offsets_selected_channels(self):
        rgba = np.array([[[0.2, 0.4, 0.6, 1.0]]], dtype=np.float32)
        out = Evaluator._kernel("Add", {"offset": 0.1, "channels": "rgb", "mix": 1.0}, [rgba])
        np.testing.assert_allclose(out[0, 0], [0.3, 0.5, 0.7, 1.0], atol=1e-6)

    def test_gamma_two_of_a_quarter_is_a_half(self):
        rgba = np.array([[[0.25, 0.25, 0.25, 1.0]]], dtype=np.float32)
        out = Evaluator._kernel("Gamma", {"gamma": 2.0, "channels": "rgb", "mix": 1.0}, [rgba])
        np.testing.assert_allclose(out[0, 0, :3], 0.5, atol=1e-6)

    def test_saturation_zero_desaturates_to_luma(self):
        rgba = np.array([[[1.0, 0.0, 0.0, 1.0]]], dtype=np.float32)
        out = Evaluator._kernel("Saturation", {"saturation": 0.0, "mix": 1.0}, [rgba])
        expected = 0.2126
        np.testing.assert_allclose(out[0, 0, :3], expected, atol=1e-4)
        self.assertAlmostEqual(float(out[0, 0, 3]), 1.0)

    def test_dissolve_a_quarter_between_black_and_white_is_a_quarter(self):
        black = np.array([[[0.0, 0.0, 0.0, 0.0]]], dtype=np.float32)
        white = np.array([[[1.0, 1.0, 1.0, 1.0]]], dtype=np.float32)
        out = Evaluator._kernel("Dissolve", {"which": 0.25, "mix": 1.0}, [black, white])
        np.testing.assert_allclose(out[0, 0], 0.25, atol=1e-6)

    def test_dissolve_which_zero_is_a_and_which_one_is_b(self):
        a = np.array([[[1.0, 0.0, 0.0, 1.0]]], dtype=np.float32)
        b = np.array([[[0.0, 1.0, 0.0, 1.0]]], dtype=np.float32)
        out0 = Evaluator._kernel("Dissolve", {"which": 0.0, "mix": 1.0}, [a, b])
        out1 = Evaluator._kernel("Dissolve", {"which": 1.0, "mix": 1.0}, [a, b])
        np.testing.assert_allclose(out0, a)
        np.testing.assert_allclose(out1, b)

    def test_keymix_copies_a_only_where_the_mask_channel_is_set(self):
        a = np.array([[[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]], dtype=np.float32)
        b = np.array([[[0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]], dtype=np.float32)
        mask = np.array([[[0, 0, 0, 1], [0, 0, 0, 0]]], dtype=np.float32)
        out = Evaluator._kernel("Keymix", {"mix": 1.0, "invert_mask": 0}, [a, b, mask])
        np.testing.assert_allclose(out[0, 0], a[0, 0])  # mask=1: A copied through
        np.testing.assert_allclose(out[0, 1], b[0, 1])  # mask=0: B kept

    def test_keymix_invert_mask_flips_which_side_wins(self):
        a = np.array([[[1.0, 0.0, 0.0, 1.0]]], dtype=np.float32)
        b = np.array([[[0.0, 1.0, 0.0, 1.0]]], dtype=np.float32)
        mask = np.array([[[0, 0, 0, 1]]], dtype=np.float32)
        out = Evaluator._kernel("Keymix", {"mix": 1.0, "invert_mask": 1}, [a, b, mask])
        np.testing.assert_allclose(out[0, 0], b[0, 0])

    def test_keymix_with_no_mask_wired_copies_a_everywhere_at_mix_one(self):
        a = np.array([[[1.0, 0.0, 0.0, 1.0]]], dtype=np.float32)
        b = np.array([[[0.0, 1.0, 0.0, 1.0]]], dtype=np.float32)
        out = Evaluator._kernel("Keymix", {"mix": 1.0, "invert_mask": 0}, [a, b])
        np.testing.assert_allclose(out[0, 0], a[0, 0])

    def test_copy_moves_named_channels_from_a_and_leaves_the_rest_as_b(self):
        a = np.array([[[1.0, 0.2, 0.3, 0.9]]], dtype=np.float32)
        b = np.array([[[0.0, 0.5, 0.6, 0.1]]], dtype=np.float32)
        out = Evaluator._kernel(
            "Copy", {"copy_red": "A.r", "copy_green": "none", "copy_blue": "none",
                    "copy_alpha": "A.a", "mix": 1.0}, [a, b])
        np.testing.assert_allclose(out[0, 0], [1.0, 0.5, 0.6, 0.9], atol=1e-6)

    def test_copy_with_every_source_none_is_identity_on_b(self):
        a = np.array([[[1.0, 0.2, 0.3, 0.9]]], dtype=np.float32)
        b = np.array([[[0.0, 0.5, 0.6, 0.1]]], dtype=np.float32)
        out = Evaluator._kernel(
            "Copy", {"copy_red": "none", "copy_green": "none", "copy_blue": "none",
                    "copy_alpha": "none", "mix": 1.0}, [a, b])
        np.testing.assert_allclose(out[0, 0], b[0, 0])

    def test_channel_merge_combines_the_chosen_channels_with_the_chosen_operation(self):
        a = np.array([[[0.3, 0.0, 0.0, 1.0]]], dtype=np.float32)
        b = np.array([[[0.0, 0.4, 0.0, 0.5]]], dtype=np.float32)
        out = Evaluator._kernel(
            "ChannelMerge", {"a_channel": "A.r", "b_channel": "B.g", "out_channel": "A",
                            "operation": "plus", "mix": 1.0}, [a, b])
        self.assertAlmostEqual(float(out[0, 0, 3]), 0.7, places=6)
        # Untouched channels stay B's own.
        np.testing.assert_allclose(out[0, 0, :3], b[0, 0, :3])

    def test_channel_merge_multiply_operation(self):
        a = np.array([[[0.5, 0.0, 0.0, 1.0]]], dtype=np.float32)
        b = np.array([[[0.0, 0.4, 0.0, 1.0]]], dtype=np.float32)
        out = Evaluator._kernel(
            "ChannelMerge", {"a_channel": "A.r", "b_channel": "B.g", "out_channel": "R",
                            "operation": "multiply", "mix": 1.0}, [a, b])
        self.assertAlmostEqual(float(out[0, 0, 0]), 0.2, places=6)


class MaskMixTests(unittest.TestCase):
    """mix=0 is identity to the untouched source/B; a zero mask hides the effect."""

    def _single_image_graph(self, kind, params):
        g = Graph()
        g.add("plate", "Constant", dict(red=0.4, green=0.4, blue=0.4, alpha=1.0))
        g.add("matte", "Constant", dict(alpha=0.0))
        g.add("node", kind, params, image="plate", mask="matte")
        return g

    def test_single_image_kinds_mix_zero_is_identity(self):
        params = {"Invert": {"mix": 0.0}, "Clamp": {"minimum": 0.9, "maximum": 0.9, "clamp_min": 1,
                                                    "clamp_max": 1, "mix": 0.0},
                 "Multiply": {"multiply": 5.0, "mix": 0.0}, "Add": {"offset": 5.0, "mix": 0.0},
                 "Gamma": {"gamma": 5.0, "mix": 0.0}, "Saturation": {"saturation": 0.0, "mix": 0.0}}
        for kind in GROUP_B_SINGLE:
            with self.subTest(kind=kind):
                g = Graph(); g.add("plate", "Constant", dict(red=0.4, green=0.4, blue=0.4, alpha=1.0))
                g.add("node", kind, params[kind], image="plate")
                plate = evaluator_pixels(g.doc, "plate")
                out = evaluator_pixels(g.doc, "node")
                np.testing.assert_allclose(out, plate)

    def test_single_image_kinds_zero_mask_hides_the_effect(self):
        params = {"Invert": {"mix": 1.0}, "Multiply": {"multiply": 5.0, "mix": 1.0},
                 "Add": {"offset": 5.0, "mix": 1.0}, "Gamma": {"gamma": 5.0, "mix": 1.0},
                 "Saturation": {"saturation": 0.0, "mix": 1.0},
                 "Clamp": {"minimum": 0.9, "maximum": 0.9, "clamp_min": 1, "clamp_max": 1, "mix": 1.0}}
        for kind in GROUP_B_SINGLE:
            with self.subTest(kind=kind):
                g = self._single_image_graph(kind, params[kind])
                plate = evaluator_pixels(g.doc, "plate")
                out = evaluator_pixels(g.doc, "node")
                np.testing.assert_allclose(out, plate)

    def test_merge_family_mix_zero_is_b(self):
        params = {"Dissolve": {"which": 0.5, "mix": 0.0}, "Keymix": {"mix": 0.0},
                 "Copy": {"copy_red": "A.r", "mix": 0.0},
                 "ChannelMerge": {"a_channel": "A.r", "b_channel": "B.g", "out_channel": "R",
                                  "operation": "plus", "mix": 0.0}}
        for kind in GROUP_B_MERGE:
            with self.subTest(kind=kind):
                g = Graph()
                g.add("a", "Constant", dict(red=1.0, green=0.0, blue=0.0, alpha=1.0))
                g.add("b", "Constant", dict(red=0.0, green=1.0, blue=0.0, alpha=1.0))
                g.add("node", kind, params[kind], A="a", B="b")
                b = evaluator_pixels(g.doc, "b")
                out = evaluator_pixels(g.doc, "node")
                np.testing.assert_allclose(out, b)


class TilePathParityTests(unittest.TestCase):
    """Every group-b kind renders byte-identical pixels on the evaluator and the tile path."""

    def test_single_image_kinds_match_across_both_paths(self):
        cases = {
            "Invert": dict(channels="rgba", mix=0.7),
            "Clamp": dict(minimum=0.1, maximum=0.8, clamp_min=1, clamp_max=1, channels="rgb", mix=1.0),
            "Multiply": dict(multiply=1.7, channels="rgba", mix=1.0),
            "Add": dict(offset=0.2, channels="rgb", mix=1.0),
            "Gamma": dict(gamma=2.2, channels="rgb", mix=1.0),
            "Saturation": dict(saturation=0.3, mix=1.0),
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

    def test_merge_family_kinds_match_across_both_paths(self):
        cases = {
            "Dissolve": dict(which=0.3, mix=1.0),
            "Keymix": dict(invert_mask=0, mix=1.0),
            "Copy": dict(copy_red="A.r", copy_green="none", copy_blue="A.b", copy_alpha="A.a", mix=1.0),
            "ChannelMerge": dict(a_channel="A.r", b_channel="B.g", out_channel="B",
                                 operation="screen", mix=1.0),
        }
        for kind, params in cases.items():
            with self.subTest(kind=kind):
                self.assertIn(kind, SUPPORTED_TILED_KINDS)
                g = Graph()
                g.add("a_plate", "Checker", dict(width=40, height=24, size=6))
                g.add("a", "Grade", dict(exposure=0.3), image="a_plate")
                g.add("b_plate", "Constant", dict(width=40, height=24, red=0.2, green=0.6, blue=0.1, alpha=0.8))
                g.add("b", "Grade", dict(exposure=-0.1), image="b_plate")
                g.add("matte", "Constant", dict(width=40, height=24, red=1, green=1, blue=1, alpha=0.4))
                g.add("node", kind, params, A="a", B="b", mask="matte")
                ev = evaluator_pixels(g.doc, "node")
                ti = tile_pixels(g.doc, "node")
                np.testing.assert_allclose(ti, ev, atol=1e-6)


class BypassTests(unittest.TestCase):
    """Dissolve/Keymix/Copy/ChannelMerge follow Merge's own bypass convention exactly."""

    def merge_family_graph(self, kind, params=None):
        g = Graph()
        g.add("bg_plate", "Checker", dict(width=32, height=24, size=8))
        g.add("bg", "Grade", dict(exposure=1.0), image="bg_plate")
        g.add("fg_plate", "Constant", dict(red=.8, green=.1, blue=.1, alpha=.5))
        g.add("fg", "ColorCorrect", dict(saturation=.5), image="fg_plate")
        g.add("node", kind, params or {}, A="fg", B="bg")
        return g

    def test_bypass_slot_names_b_or_a_for_every_merge_family_kind(self):
        for kind in GROUP_B_MERGE:
            with self.subTest(kind=kind):
                slots = SPECS[kind]["inputs"] + SPECS[kind].get("optional_inputs", [])
                node = dict(type=kind, inputs={slot: None for slot in slots})
                node["inputs"]["A"] = "fg"
                node["inputs"]["B"] = "bg"
                self.assertEqual(bypass_slot(node), "B")
                node["inputs"]["B"] = None
                self.assertEqual(bypass_slot(node), "A")

    def test_bypassed_node_passes_b_and_never_touches_a_on_both_paths(self):
        for kind in GROUP_B_MERGE:
            with self.subTest(kind=kind):
                g = self.merge_family_graph(kind)
                background = evaluator_pixels(g.doc, "bg")
                g.bypass("node")
                for label, pixels in (("evaluator", evaluator_pixels), ("tiles", tile_pixels)):
                    with self.subTest(path=label):
                        np.testing.assert_allclose(pixels(g.doc, "node"), background)
                # A is not an ancestor of a bypassed node: a broken A branch cannot break the picture.
                g.d.execute(dict(op="connect", id="fg", input="image", source=None))
                np.testing.assert_allclose(evaluator_pixels(g.doc, "node"), background)

    def test_mask_is_ignored_when_bypassed(self):
        for kind in GROUP_B_MERGE:
            with self.subTest(kind=kind):
                g = self.merge_family_graph(kind)
                g.d.execute(dict(op="create", id="matte", type="Constant", params=dict(alpha=.25)))
                g.d.execute(dict(op="connect", id="node", input="mask", source="matte"))
                background = evaluator_pixels(g.doc, "bg")
                g.bypass("node")
                np.testing.assert_allclose(evaluator_pixels(g.doc, "node"), background)


class SpecCoverageTests(unittest.TestCase):
    def test_all_ten_nodes_are_registered_with_mask_and_mix(self):
        for kind in GROUP_B_SINGLE + GROUP_B_MERGE:
            with self.subTest(kind=kind):
                self.assertIn(kind, SPECS)
                self.assertIn("mask", SPECS[kind].get("optional_inputs", []))
                self.assertIn("mix", SPECS[kind]["params"])
                self.assertEqual(LIMITS["mix"], (0, 1))

    def test_choices_and_limits_cover_every_new_enum_and_ranged_param(self):
        for name in ("channels", "copy_red", "copy_green", "copy_blue", "copy_alpha",
                    "a_channel", "b_channel", "out_channel"):
            self.assertIn(name, CHOICES)
        for name in ("minimum", "maximum", "clamp_min", "clamp_max", "invert_mask"):
            self.assertIn(name, LIMITS)

    def test_dispatcher_creates_every_new_node_with_valid_defaults(self):
        for kind in GROUP_B_SINGLE + GROUP_B_MERGE:
            with self.subTest(kind=kind):
                d = Dispatcher()
                result = d.execute(dict(op="create", type=kind))
                self.assertIn("id", result["result"])


if __name__ == "__main__":
    unittest.main()
