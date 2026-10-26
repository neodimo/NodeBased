"""Fidelity step B, part 1: Nuke's shared filter and colour chrome (channels, mask channel, invert mask,
inject, un-premultiply by, mix) on the 2D nodes. Pixel assertions per option on a representative node of each
family, the evaluator and tile paths on the same pixels, bypass, and old documents."""
import copy
import unittest

import numpy as np

from nodebased import core, ops2d_chrome as chrome
from nodebased.core import SPECS, Dispatcher, bypass_slot, load_document, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.raster import Raster
from nodebased.tileexec import SUPPORTED_TILED_KINDS, TileExecutor
from tests.test_2d_parity_group_3b import Graph, evaluator_pixels, evaluator_raster

H, W = 24, 32


def image(seed=3):
    rng = np.random.default_rng(seed)
    pixels = rng.random((H, W, 4), dtype=np.float32) * 2.0 - 0.25   # HDR with negatives
    pixels[..., 3] = rng.random((H, W)).astype(np.float32) * 0.8 + 0.2
    return pixels


def matte(seed=9):
    rng = np.random.default_rng(seed)
    return rng.random((H, W, 4), dtype=np.float32)


def render(kind, params=None, source=None, mask=None, second=None, layers=None):
    """The node's whole-image kernel on Rasters, the way the evaluator calls it."""
    spec = SPECS[kind]
    p = dict(spec["params"])
    p.update(params or {})
    slots = list(spec["inputs"]) + list(spec.get("optional_inputs", []))
    inputs = [None] * len(slots)
    source = image() if source is None else source
    inputs[0] = Raster(source, layers=layers)
    if "B" in slots:
        inputs[slots.index("B")] = Raster.of(source if second is None else second)
        inputs[0] = Raster.of(source if second is None else second[::-1].copy())
    if mask is not None:
        inputs[slots.index("mask")] = Raster.of(mask)
    return Evaluator._windowed_kernel(kind, p, inputs)


class DefaultsTests(unittest.TestCase):
    def test_every_default_leaves_the_kernel_alone(self):
        for kind in chrome.FILTER_CHROME:
            cfg = chrome.filter_config(kind, SPECS[kind]["params"])
            for has_mask in (False, True):
                with self.subTest(kind=kind, has_mask=has_mask):
                    # A kind whose own kernel ignores the mask input (Roto) has the wrapper gate a wired mask.
                    self.assertEqual(cfg.active(has_mask), has_mask and kind in chrome.WRAPPER_GATED)

    def test_the_wrapper_with_nothing_changed_gives_the_kernels_own_pixels_with_a_mask_and_mix(self):
        """The wrapper's gating is the kernels' own: forced on, with every knob at its default, it reproduces
        the old path for every node it covers, with a wired mask and a mix below one."""
        rng_image, rng_mask = image(), matte()
        for kind in chrome.FILTER_CHROME:
            if kind in chrome.MIXLESS:
                continue                                  # no mix to reproduce, and (Roto) no mask gating of its own
            spec = SPECS[kind]
            slots = list(spec["inputs"]) + list(spec.get("optional_inputs", []))
            inputs = [None] * len(slots)
            for position, slot in enumerate(slots):
                if slot == "mask":
                    inputs[position] = Raster.of(rng_mask)
                elif slot in spec["inputs"] or slot in ("depth", "uv", "kernel"):
                    inputs[position] = Raster.of(rng_image if position % 2 == 0 else rng_image[::-1].copy())
            p = dict(spec["params"], mix=0.4)
            if kind in chrome.DRAW_KINDS_WITH_CHROME:       # a Draw node draws at its own format; the image is optional
                p.update(width=W, height=H)
                if kind == "Text":
                    p.update(box_width=W - 4, box_height=H - 4, box_x=2, box_y=2, font_size=12.0)
            with self.subTest(kind=kind):
                native = Evaluator._windowed_kernel_inner(kind, p, inputs)
                forced = chrome.run_evaluator(kind, p, inputs, lambda k, q, i: Evaluator._windowed_kernel_inner(k, q, i),
                                              spec, force=True)
                self.assertEqual(native.data, forced.data)
                np.testing.assert_allclose(forced.pixels, native.pixels, atol=1e-6)

    def test_the_nuke_defaults_are_the_ones_in_the_specs(self):
        for kind, (channels, mask, unpremult) in chrome.FILTER_CHROME.items():
            params = SPECS[kind]["params"]
            if mask:
                self.assertEqual((params["maskChannelInput"], params["invert_mask"], params["inject"]), ("none", 0, 0), kind)
            if unpremult:
                self.assertEqual((params["unpremult"], params["invert_unpremult"]), ("none", 0), kind)
            if channels is not None and kind not in chrome.NATIVE_CHANNELS:
                self.assertEqual(params["channels"], channels, kind)
            if kind not in chrome.MIXLESS:                  # Nuke's Roto, Premult and Unpremult have no mix
                self.assertIn("mix", params, kind)

    def test_an_old_document_without_the_knobs_loads_with_the_defaults_and_renders_the_same(self):
        graph = Graph()
        graph.add("plate", "Rectangle", dict(width=64, height=48, box_x=8, box_y=8, box_width=40, box_height=30))
        graph.add("grade", "Grade", dict(exposure=0.5), image="plate")
        graph.add("blur", "Blur", dict(radius=3.0), image="grade")
        before = evaluator_pixels(graph.doc, "blur")
        old = copy.deepcopy(graph.doc)
        for node in old["nodes"].values():
            for name in list(node["params"]):
                if name in chrome.FILTER_PARAM_NAMES + chrome.DRAW_PARAM_NAMES and name not in ("invert",):
                    del node["params"][name]
        with self.assertRaises(ValueError):
            validate(old)
        upgraded = upgrade_document(old)
        validate(upgraded)
        self.assertEqual(upgraded["nodes"]["grade"]["params"]["channels"], "rgb")
        self.assertEqual(upgraded["nodes"]["blur"]["params"]["maskChannelInput"], "none")
        np.testing.assert_array_equal(evaluator_pixels(upgraded, "blur"), before)

    def test_old_documents_inside_a_group_get_the_knobs_too(self):
        nodes = {"g": {"type": "Group", "params": {}, "graph": {"nodes": {
            "inner": {"type": "Grade", "params": {"exposure": 0.0, "multiply": 1.0, "offset": 0.0, "mix": 1.0}}}}}}
        chrome.upgrade_nodes(nodes)
        self.assertEqual(nodes["g"]["graph"]["nodes"]["inner"]["params"]["maskChannelInput"], "none")


class ChannelsTests(unittest.TestCase):
    def test_a_single_channel_takes_only_that_channels_result(self):
        source = image()
        full = render("Grade", dict(multiply=2.0, offset=0.1)).pixels
        for channels, picked in (("red", [0]), ("green", [1]), ("blue", [2]), ("rgb", [0, 1, 2]), ("rgba", [0, 1, 2, 3]),
                                 ("none", [])):
            with self.subTest(channels=channels):
                out = render("Grade", dict(multiply=2.0, offset=0.1, channels=channels)).pixels
                expected = source.copy()
                for c in picked:
                    expected[..., c] = full[..., c]
                np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_alpha_only_and_all_on_a_node_whose_kernel_has_no_alpha_work(self):
        source = image()
        out = render("Grade", dict(multiply=2.0, channels="alpha")).pixels
        np.testing.assert_array_equal(out, source)            # Grade leaves alpha alone, so alpha-only changes nothing
        np.testing.assert_allclose(render("Grade", dict(multiply=2.0, channels="all")).pixels,
                                   render("Grade", dict(multiply=2.0, channels="rgba")).pixels, atol=1e-6)

    def test_a_node_with_its_own_rgb_rgba_alpha_choice_keeps_it_and_gains_the_rest(self):
        source = image()
        # rgb / rgba / alpha stay the kernel's own: Add changes exactly those channels.
        for channels, picked in (("rgb", [0, 1, 2]), ("rgba", [0, 1, 2, 3]), ("alpha", [3])):
            out = render("Add", dict(offset=0.5, channels=channels)).pixels
            expected = source.copy()
            expected[..., picked] += 0.5
            np.testing.assert_allclose(out, expected, atol=1e-6, err_msg=channels)
        # The new words are the wrapper's.
        for channels, picked in (("red", [0]), ("blue", [2]), ("all", [0, 1, 2, 3]), ("none", [])):
            out = render("Add", dict(offset=0.5, channels=channels)).pixels
            expected = source.copy()
            expected[..., picked] += 0.5
            np.testing.assert_allclose(out, expected, atol=1e-6, err_msg=channels)

    def test_a_blur_restricted_to_alpha_blurs_alpha_only(self):
        source = image()
        both = render("Blur", dict(radius=2.0, channels="all")).pixels
        only = render("Blur", dict(radius=2.0, channels="alpha")).pixels
        np.testing.assert_array_equal(only[..., :3], source[..., :3])
        np.testing.assert_allclose(only[..., 3], both[..., 3], atol=1e-6)

    def test_a_named_layer_is_processed_and_the_picture_is_left_alone(self):
        source = image()
        layer = Raster.of(image(seed=17))
        out = render("Add", dict(offset=0.25, channels="extra"), layers={"extra": layer})
        np.testing.assert_allclose(out.pixels, source, atol=1e-6)
        expected = layer.pixels + 0.25        # a named layer is processed whole: all four of its channels
        np.testing.assert_allclose(out.layers["extra"].pixels, expected, atol=1e-6)

    def test_one_component_of_a_layer(self):
        layer = Raster.of(image(seed=17))
        out = render("Add", dict(offset=0.25, channels="extra.g"), layers={"extra": layer})
        expected = layer.pixels.copy()
        expected[..., 1] += 0.25
        np.testing.assert_allclose(out.layers["extra"].pixels, expected, atol=1e-6)

    def test_an_unknown_layer_names_the_ones_there_are(self):
        with self.assertRaisesRegex(ValueError, "no such layer"):
            render("Add", dict(offset=0.25, channels="ghost"), layers={"extra": Raster.of(image())})

    def test_channel_text_parses(self):
        self.assertEqual(chrome.parse_channels("rgb").rgba, (True, True, True, False))
        self.assertEqual(chrome.parse_channels("red, alpha").rgba, (True, False, False, True))
        self.assertEqual(chrome.parse_channels("rgba.green").rgba, (False, True, False, False))
        self.assertEqual(chrome.parse_channels("depth").layers, ("depth",))
        self.assertTrue(chrome.parse_channels("all").everything)


class MaskTests(unittest.TestCase):
    def _gate(self, source, filtered, m, mix=1.0):
        gate = m * mix
        return filtered * gate + source * (1.0 - gate)

    def test_the_default_mask_channel_is_the_wired_masks_alpha(self):
        source, mask = image(), matte()
        filtered = render("Add", dict(offset=0.5)).pixels
        out = render("Add", dict(offset=0.5, mix=0.7), mask=mask).pixels
        np.testing.assert_allclose(out, self._gate(source, filtered, mask[..., 3:4], 0.7), atol=1e-6)

    def test_other_channels_of_the_wired_mask(self):
        source, mask = image(), matte()
        filtered = render("Add", dict(offset=0.5)).pixels
        for token, channel in (("rgba.red", 0), ("rgba.green", 1), ("rgba.blue", 2), ("rgba.alpha", 3)):
            with self.subTest(token=token):
                out = render("Add", dict(offset=0.5, maskChannelInput=token), mask=mask).pixels
                np.testing.assert_allclose(out, self._gate(source, filtered, mask[..., channel:channel + 1]), atol=1e-6)
        luminance = 0.2126 * mask[..., 0:1] + 0.7152 * mask[..., 1:2] + 0.0722 * mask[..., 2:3]
        out = render("Add", dict(offset=0.5, maskChannelInput="rgba.luminance"), mask=mask).pixels
        np.testing.assert_allclose(out, self._gate(source, filtered, luminance), atol=1e-5)

    def test_a_channel_of_the_image_itself_masks_when_no_mask_is_wired(self):
        source = image()
        filtered = render("Add", dict(offset=0.5)).pixels
        out = render("Add", dict(offset=0.5, maskChannelInput="rgba.alpha")).pixels
        np.testing.assert_allclose(out, self._gate(source, filtered, source[..., 3:4]), atol=1e-6)

    def test_invert_mask(self):
        source, mask = image(), matte()
        filtered = render("Add", dict(offset=0.5)).pixels
        out = render("Add", dict(offset=0.5, invert_mask=1, mix=0.6), mask=mask).pixels
        np.testing.assert_allclose(out, self._gate(source, filtered, 1.0 - mask[..., 3:4], 0.6), atol=1e-6)
        inverted_channel = render("Add", dict(offset=0.5, invert_mask=1, maskChannelInput="rgba.red"), mask=mask).pixels
        np.testing.assert_allclose(inverted_channel, self._gate(source, filtered, 1.0 - mask[..., 0:1]), atol=1e-6)

    def test_invert_mask_without_a_mask_changes_nothing(self):
        np.testing.assert_array_equal(render("Add", dict(offset=0.5, invert_mask=1)).pixels, render("Add", dict(offset=0.5)).pixels)

    def test_inject_writes_the_matte_as_the_mask_layer(self):
        mask = matte()
        out = render("Add", dict(offset=0.5, inject=1, invert_mask=1, maskChannelInput="rgba.green"), mask=mask)
        np.testing.assert_allclose(out.layers["mask"].pixels[..., 3:4], 1.0 - mask[..., 1:2], atol=1e-6)
        plain = render("Add", dict(offset=0.5), mask=mask)
        self.assertFalse(plain.layers)

    def test_a_blur_with_a_mask_covers_the_source_where_the_mask_is_empty(self):
        source, mask = image(), matte()
        filtered = render("Blur", dict(radius=2.0)).pixels
        out = render("Blur", dict(radius=2.0, invert_mask=1), mask=mask).pixels
        np.testing.assert_allclose(out, self._gate(source, filtered, 1.0 - mask[..., 3:4]), atol=1e-6)

    def test_merge_gates_against_its_background(self):
        a, b, mask = image(1), image(2), matte()
        spec_p = dict(SPECS["Merge"]["params"], operation="plus")
        merged = Evaluator._windowed_kernel("Merge", spec_p, [Raster.of(a), Raster.of(b), None]).pixels
        out = Evaluator._windowed_kernel("Merge", dict(spec_p, invert_mask=1, mix=0.5),
                                         [Raster.of(a), Raster.of(b), Raster.of(mask)]).pixels
        np.testing.assert_allclose(out, self._gate(b, merged, 1.0 - mask[..., 3:4], 0.5), atol=1e-5)


class UnpremultTests(unittest.TestCase):
    def test_a_kernel_that_does_not_commute_with_the_alpha_sees_the_straight_colour(self):
        source = image()
        alpha = source[..., 3:4]
        gamma = 2.0
        out = render("Gamma", dict(gamma=gamma, unpremult="rgba.alpha")).pixels
        straight = source[..., :3] / alpha
        expected_rgb = np.sign(straight) * np.abs(straight) ** (1.0 / gamma) * alpha
        np.testing.assert_allclose(out[..., :3], expected_rgb, rtol=1e-4, atol=1e-5)
        np.testing.assert_array_equal(out[..., 3], source[..., 3])
        without = render("Gamma", dict(gamma=gamma)).pixels
        self.assertGreater(float(np.abs(out - without).max()), 0.01)

    def test_invert_unpremult_premultiplies_first(self):
        source = image()
        alpha = source[..., 3:4]
        out = render("Gamma", dict(gamma=2.0, unpremult="rgba.alpha", invert_unpremult=1)).pixels
        premultiplied = source[..., :3] * alpha
        expected_rgb = np.sign(premultiplied) * np.abs(premultiplied) ** 0.5 / alpha
        np.testing.assert_allclose(out[..., :3], expected_rgb, rtol=1e-4, atol=1e-5)

    def test_a_zero_channel_is_left_alone_instead_of_dividing(self):
        source = image()
        source[0, 0, 3] = 0.0
        out = render("Add", dict(offset=0.5, unpremult="rgba.alpha"), source=source).pixels
        self.assertTrue(np.all(np.isfinite(out)))

    def test_a_linear_kernel_commutes_with_it(self):
        source = image()
        np.testing.assert_allclose(render("Multiply", dict(multiply=3.0, unpremult="rgba.alpha")).pixels,
                                   render("Multiply", dict(multiply=3.0)).pixels, rtol=1e-4, atol=1e-5)


class TileTests(unittest.TestCase):
    """The tile path and the evaluator give the same pixels with each chrome option, across several tiles."""

    def _graph(self, kind, params, masked, wires=None):
        g = Graph()
        g.add("plate", "Rectangle", dict(width=257, height=193, box_x=40, box_y=30, box_width=150, box_height=110,
                                        red=0.9, green=0.5, blue=0.2, alpha=0.8, softness=0.3))
        g.add("ramp", "Ramp", dict(width=257, height=193))
        g.add("matte", "Radial", dict(width=257, height=193, box_x=20, box_y=20, box_width=200, box_height=150, softness=0.8))
        inputs = dict(wires or {"image": "plate"})
        if masked:
            inputs["mask"] = "matte"
        g.add("node", kind, params, **inputs)
        return g

    def _tiled(self, document):
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=64)
        document = dict(document, view="node")
        self.assertTrue(executor.supports_tiled(document, "node"))
        region = executor.canvas_region(document, "node", frame=1, tier=1)
        return executor.compose_region(document, "node", region, frame=1, tier=1).pixels

    def test_each_option_on_a_pointwise_and_a_neighbourhood_node(self):
        options = (dict(channels="red"), dict(channels="alpha"), dict(channels="none"),
                   dict(maskChannelInput="rgba.red"), dict(invert_mask=1), dict(maskChannelInput="rgba.blue", invert_mask=1),
                   dict(unpremult="rgba.alpha"), dict(unpremult="rgba.alpha", invert_unpremult=1),
                   dict(channels="green", unpremult="rgba.alpha", invert_mask=1, maskChannelInput="rgba.alpha"),
                   dict(inject=1, invert_mask=1))
        kernels = (("Gamma", dict(gamma=1.8)), ("Add", dict(offset=0.2)), ("Saturation", dict(saturation=0.3)),
                   ("Blur", dict(radius=6.0)), ("Sharpen", dict(sharpen_amount=1.5)))
        for kind, kernel_params in kernels:
            self.assertIn(kind, SUPPORTED_TILED_KINDS)
            for option in options:
                if kind in ("Blur", "Sharpen") and "unpremult" in option:
                    continue         # these two have no (un)premult knob in Nuke
                for masked in (False, True):
                    with self.subTest(kind=kind, option=option, masked=masked):
                        params = dict(kernel_params, **option)
                        params = {k: v for k, v in params.items() if k in SPECS[kind]["params"]}
                        g = self._graph(kind, params, masked)
                        np.testing.assert_allclose(self._tiled(g.doc), evaluator_pixels(g.doc, "node"), atol=1e-5)

    def test_merge_family_on_tiles(self):
        for option in (dict(invert_mask=1), dict(maskChannelInput="rgba.red", mix=0.6)):
            for masked in (False, True):
                with self.subTest(option=option, masked=masked):
                    g = self._graph("Merge", dict(operation="plus", **option), masked, wires=dict(A="plate", B="ramp"))
                    np.testing.assert_allclose(self._tiled(g.doc), evaluator_pixels(g.doc, "node"), atol=1e-5)

    def test_the_inject_layer_is_the_same_on_tiles(self):
        g = self._graph("Add", dict(offset=0.2, inject=1, invert_mask=1, maskChannelInput="rgba.red"), True)
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=64)
        document = dict(g.doc, view="node")
        region = executor.canvas_region(document, "node", frame=1, tier=1)
        tiled = executor.compose_region(document, "node", region, frame=1, tier=1)
        full = evaluator_raster(g.doc, "node")
        np.testing.assert_allclose(tiled.layers["mask"][..., 3], full.layers["mask"].pixels[..., 3], atol=1e-6)

    def test_a_chrome_knob_that_names_a_layer_sends_the_graph_to_the_evaluator(self):
        g = self._graph("Add", dict(offset=0.2, channels="depth"), False)
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=64)
        self.assertFalse(executor.supports_tiled(dict(g.doc, view="node"), "node"))
        g = self._graph("Add", dict(offset=0.2, maskChannelInput="depth.Z"), False)
        self.assertFalse(executor.supports_tiled(dict(g.doc, view="node"), "node"))


class BypassAndMixTests(unittest.TestCase):
    def test_bypass_passes_the_input_through_with_every_chrome_knob_set(self):
        g = Graph()
        g.add("plate", "Rectangle", dict(width=64, height=48, box_x=4, box_y=4, box_width=50, box_height=30, red=0.4, green=0.3, blue=0.2, alpha=0.9))
        g.add("node", "Gamma", dict(gamma=2.0, channels="red", maskChannelInput="rgba.alpha", invert_mask=1, inject=1,
                                    unpremult="rgba.alpha"), image="plate")
        base = evaluator_pixels(g.doc, "plate")
        self.assertFalse(np.array_equal(evaluator_pixels(g.doc, "node"), base))
        g.bypass("node")
        np.testing.assert_array_equal(evaluator_pixels(g.doc, "node"), base)
        self.assertEqual(bypass_slot(g.doc["nodes"]["node"]), "image")

    def test_mix_zero_returns_the_input_even_with_chrome_set(self):
        source = image()
        out = render("Gamma", dict(gamma=2.0, mix=0.0, channels="red", unpremult="rgba.alpha", invert_mask=1),
                     mask=matte()).pixels
        np.testing.assert_allclose(out, source, atol=1e-6)

    def test_open_choice_values_are_accepted_by_validation(self):
        g = Graph()
        g.add("plate", "Rectangle", dict(width=64, height=48))
        g.add("node", "Gamma", dict(gamma=2.0, channels="depth.Z, extra", maskChannelInput="matte.g"), image="plate")
        validate(g.doc)


if __name__ == "__main__":
    unittest.main()
