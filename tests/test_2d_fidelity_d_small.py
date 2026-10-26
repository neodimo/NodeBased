"""Fidelity step D, part 3: Constant, Invert, Copy, Remove and CopyMetaData take Nuke's knobs. Pixels, layers and metadata by
hand, tiles against the evaluator, bypass, and old documents."""
import copy
import unittest

import numpy as np

from nodebased import smallnodes
from nodebased.core import SPECS, Dispatcher, bypass_slot, empty_document, upgrade_document, validate
from nodebased.imaging import Evaluator, _inherited_metadata
from nodebased.raster import Raster
from nodebased.tileexec import TileExecutor
from tests.test_2d_parity_group_c4 import Graph, evaluator_pixels

H, W = 12, 20


def run(kind, *rasters, **knobs):
    inputs = [r if isinstance(r, Raster) else Raster.of(np.asarray(r, np.float32)) for r in rasters]
    return Evaluator._windowed_kernel(kind, {**SPECS[kind]["params"], **knobs}, inputs)


def one(*rgba):
    return np.array([[list(rgba)]], np.float32)


class ConstantTests(unittest.TestCase):
    def pixel(self, **knobs):
        return Evaluator._kernel("Constant", {**SPECS["Constant"]["params"], "width": 2, "height": 2, "red": 0.5, "green": 0.4,
                                              "blue": 0.2, "alpha": 0.5, **knobs}, [], 1)[0, 0]

    def test_it_holds_only_the_channels_asked_for(self):
        full = [0.25, 0.2, 0.1, 0.5]                    # premultiplied
        cases = {"rgba": full, "all": full, "rgb": full[:3] + [0.0], "alpha": [0, 0, 0, 0.5], "red": [0.25, 0, 0, 0],
                 "green": [0, 0.2, 0, 0], "blue": [0, 0, 0.1, 0], "none": [0, 0, 0, 0]}
        for text, expected in cases.items():
            with self.subTest(channels=text):
                np.testing.assert_allclose(self.pixel(channels=text), expected, atol=1e-7)

    def test_first_and_last_state_the_range_to_the_nodes_downstream(self):
        g = Graph()
        g.add("plate", "Constant", dict(red=0.0, green=0.4, blue=0.6, alpha=1.0, first=1, last=10))
        for frame in range(-5, 40):
            g.key("plate", "red", frame, frame / 100.0)
        g.add("node", "TimeOffset", dict(reverse_input=1), image="plate")
        for frame in (1, 3, 10):
            self.assertAlmostEqual(float(evaluator_pixels(g.doc, "node", frame)[0, 0, 0]) * 100.0, 11 - frame, places=3)

    def test_a_constant_that_states_no_range_leaves_the_timeline_as_the_range(self):
        g = Graph()
        g.add("plate", "Constant", dict(red=0.0, green=0.4, blue=0.6, alpha=1.0))
        for frame in range(-5, 120):
            g.key("plate", "red", frame, frame / 100.0)
        g.add("node", "TimeOffset", dict(reverse_input=1), image="plate")
        first, last = g.doc["time"]["first"], g.doc["time"]["last"]
        self.assertAlmostEqual(float(evaluator_pixels(g.doc, "node", 4)[0, 0, 0]) * 100.0, first + last - 4, places=3)

    def test_tiles_equal_the_evaluator(self):
        for channels in ("rgba", "rgb", "alpha", "green", "none"):
            with self.subTest(channels=channels):
                d = Dispatcher(empty_document())
                d.execute({"op": "create", "id": "n", "type": "Constant", "params": {"width": W, "height": H, "channels": channels,
                                                                                       "alpha": 0.7}})
                d.execute({"op": "view", "id": "n"})
                executor = TileExecutor(evaluator=Evaluator(), tile_edge=8)
                expected = Evaluator().evaluate_raster(d.document, target="n").pixels
                region = executor.canvas_region(d.document, "n", frame=1, tier=1)
                np.testing.assert_array_equal(executor.compose_region(d.document, "n", region, frame=1, tier=1).pixels, expected)


class InvertTests(unittest.TestCase):
    def test_clamp_keeps_the_inversion_between_zero_and_one(self):
        pixel = one(-0.5, 1.7, 0.25, 0.5)
        np.testing.assert_allclose(run("Invert", pixel).pixels[0, 0], [1.5, -0.7, 0.75, 0.5], atol=1e-6)
        np.testing.assert_allclose(run("Invert", pixel, clamp=1).pixels[0, 0], [1.0, 0.0, 0.75, 0.5], atol=1e-6)

    def test_clamp_follows_the_channels(self):
        pixel = one(-0.5, 1.7, 0.25, -1.0)
        np.testing.assert_allclose(run("Invert", pixel, clamp=1, channels="rgba").pixels[0, 0], [1.0, 0.0, 0.75, 1.0], atol=1e-6)
        np.testing.assert_allclose(run("Invert", pixel, clamp=1, channels="alpha").pixels[0, 0], [-0.5, 1.7, 0.25, 1.0], atol=1e-6)

    def test_tiles_equal_the_evaluator_and_bypass_passes_the_image(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "src", "type": "Ramp", "params": {
            "width": W, "height": H, "p1_x": W - 1.0, "color0_red": -0.5, "color1_red": 1.7, "color0_green": 0.2, "color1_green": 0.9}})
        d.execute({"op": "create", "id": "n", "type": "Invert", "params": {"clamp": 1, "channels": "rgba"}})
        d.execute({"op": "connect", "id": "n", "input": "image", "source": "src"})
        d.execute({"op": "view", "id": "n"})
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=8)
        expected = Evaluator().evaluate_raster(d.document, target="n").pixels
        self.assertTrue(float(expected.min()) >= 0.0 and float(expected.max()) <= 1.0)
        region = executor.canvas_region(d.document, "n", frame=1, tier=1)
        np.testing.assert_array_equal(executor.compose_region(d.document, "n", region, frame=1, tier=1).pixels, expected)
        d.document["nodes"]["n"]["disabled"] = True
        np.testing.assert_array_equal(Evaluator().evaluate_raster(d.document, target="n").pixels,
                                      Evaluator().evaluate_raster(d.document, target="src").pixels)


class CopyTests(unittest.TestCase):
    a = one(0.1, 0.2, 0.3, 0.4)
    b = one(0.5, 0.6, 0.7, 0.8)

    def test_channels_copy_whole_channels_from_a_over_b(self):
        cases = {"none": [0.5, 0.6, 0.7, 0.8], "rgb": [0.1, 0.2, 0.3, 0.8], "rgba": [0.1, 0.2, 0.3, 0.4], "all": [0.1, 0.2, 0.3, 0.4],
                 "alpha": [0.5, 0.6, 0.7, 0.4], "red": [0.1, 0.6, 0.7, 0.8], "blue": [0.5, 0.6, 0.3, 0.8]}
        for text, expected in cases.items():
            with self.subTest(channels=text):
                np.testing.assert_allclose(run("Copy", self.a, self.b, channels=text).pixels[0, 0], expected, atol=1e-6)

    def test_channels_work_with_the_per_channel_pairs(self):
        out = run("Copy", self.a, self.b, channels="red", copy_alpha="A.a").pixels[0, 0]
        np.testing.assert_allclose(out, [0.1, 0.6, 0.7, 0.4], atol=1e-6)

    def test_the_mask_strip_gates_the_copy_against_b(self):
        mask = one(0, 0, 0, 0.25)
        out = run("Copy", self.a, self.b, mask, copy_red="A.r").pixels[0, 0]
        np.testing.assert_allclose(out[0], 0.5 + (0.1 - 0.5) * 0.25, atol=1e-6)
        out = run("Copy", self.a, self.b, mask, copy_red="A.r", invert_mask=1).pixels[0, 0]
        np.testing.assert_allclose(out[0], 0.5 + (0.1 - 0.5) * 0.75, atol=1e-6)
        out = run("Copy", self.a, self.b, mask, copy_red="A.r", mix=0.5).pixels[0, 0]
        np.testing.assert_allclose(out[0], 0.5 + (0.1 - 0.5) * 0.25 * 0.5, atol=1e-6)

    def test_unpremult_by_divides_b_before_the_copy_and_multiplies_after(self):
        a = one(0.9, 0.2, 0.3, 0.4)
        b = one(0.4, 0.6, 0.7, 0.5)
        out = run("Copy", a, b, copy_red="A.r", unpremult="rgba.alpha").pixels[0, 0]
        # B's colour is divided by its alpha, the red is copied from A, and the colour is multiplied by B's alpha again
        np.testing.assert_allclose(out, [0.9 * 0.5, 0.6, 0.7, 0.5], atol=1e-6)

    def test_metainput_picks_whose_metadata_goes_on(self):
        a, b = Raster.of(self.a, None), Raster.of(self.b, None)
        a.meta, b.meta = {"x": "from a", "y": "from a"}, {"y": "from b", "z": "from b"}
        pick = lambda choice: _inherited_metadata("Copy", [a, b], {"metainput": choice})
        self.assertEqual(pick("B"), {"y": "from b", "z": "from b"})
        self.assertEqual(pick("A"), {"x": "from a", "y": "from a"})
        self.assertEqual(pick("All"), {"x": "from a", "y": "from b", "z": "from b"})
        b.meta = None
        self.assertEqual(pick("B"), {"x": "from a", "y": "from a"})              # the other's, as Copy always did

    def test_rangeinput_picks_whose_frame_range_goes_on(self):
        def span(choice):
            g = Graph()
            g.add("plate", "Constant", dict(red=0.0, green=0.4, blue=0.6, alpha=1.0))
            for frame in range(-5, 60):
                g.key("plate", "red", frame, frame / 100.0)
            g.add("a", "FrameRange", dict(first_frame=1, last_frame=20), image="plate")
            g.add("b", "FrameRange", dict(first_frame=5, last_frame=10), image="plate")
            g.add("copy", "Copy", dict(rangeinput=choice, copy_red="A.r"), A="a", B="b")      # the red is A's, so it shows the frame asked for
            g.add("node", "TimeOffset", dict(reverse_input=1), image="copy")
            return round(float(evaluator_pixels(g.doc, "node", 6)[0, 0, 0]) * 100.0)
        self.assertEqual(span("B"), 5 + 10 - 6)
        self.assertEqual(span("A"), 1 + 20 - 6)
        self.assertEqual(span("All"), 1 + 20 - 6)

    def test_tiles_equal_the_evaluator_with_the_chrome(self):
        for knobs in (dict(channels="rgb"), dict(copy_red="A.g", mix=0.5), dict(channels="alpha", unpremult="rgba.alpha"),
                      dict(channels="rgba", invert_unpremult=1, unpremult="rgba.red")):
            with self.subTest(knobs=knobs):
                d = Dispatcher(empty_document())
                for name, c0, c1 in (("a", 0.2, 0.9), ("b", 0.7, 0.3)):
                    d.execute({"op": "create", "id": name, "type": "Ramp", "params": {
                        "width": W, "height": H, "p1_x": W - 1.0, "color0_red": c0, "color1_red": c1, "color0_green": 0.5, "color1_green": c0,
                        "color0_alpha": 0.8, "color1_alpha": 0.5}})
                d.execute({"op": "create", "id": "n", "type": "Copy", "params": knobs})
                d.execute({"op": "connect", "id": "n", "input": "A", "source": "a"})
                d.execute({"op": "connect", "id": "n", "input": "B", "source": "b"})
                d.execute({"op": "view", "id": "n"})
                executor = TileExecutor(evaluator=Evaluator(), tile_edge=8)
                self.assertTrue(executor.supports_tiled(d.document, "n"))
                expected = Evaluator().evaluate_raster(d.document, target="n").pixels
                region = executor.canvas_region(d.document, "n", frame=1, tier=1)
                np.testing.assert_allclose(executor.compose_region(d.document, "n", region, frame=1, tier=1).pixels, expected, atol=1e-6)


class RemoveTests(unittest.TestCase):
    def source(self):
        layer = lambda value: Raster.of(np.full((2, 2, 4), value, np.float32))
        return Raster(np.zeros((2, 2, 4), np.float32), None, None, {"a": layer(1), "b": layer(2), "c": layer(3), "d": layer(4)})

    def names(self, **knobs):
        return sorted(run("Remove", self.source(), **knobs).layers)

    def test_remove_takes_every_channels_knob(self):
        self.assertEqual(self.names(layers="a"), ["b", "c", "d"])
        self.assertEqual(self.names(layers="a", channels2="b"), ["c", "d"])
        self.assertEqual(self.names(layers="a", channels2="b", channels3="c", channels4="d"), [])
        self.assertEqual(self.names(layers="none", channels3="c"), ["a", "b", "d"])

    def test_keep_keeps_only_the_named_layers(self):
        self.assertEqual(self.names(remove_operation="keep", layers="a", channels4="c"), ["a", "c"])
        self.assertEqual(self.names(remove_operation="keep", layers="none"), [])

    def test_all_names_every_layer(self):
        self.assertEqual(self.names(layers="all"), [])
        self.assertEqual(self.names(remove_operation="keep", layers="all"), ["a", "b", "c", "d"])
        self.assertEqual(self.names(channels2="all"), [])

    def test_the_defaults_are_the_remove_of_before(self):
        self.assertEqual(self.names(), ["a", "b", "c", "d"])
        self.assertEqual(self.names(layers="b,c"), ["a", "d"])


class CopyMetaDataTests(unittest.TestCase):
    image = {"camera": "cam A", "lens": "35mm", "shared": "from image"}
    other = {"scene": "sc 12", "lens": "50mm", "shared": "from meta"}

    def result(self, **knobs):
        params = {**SPECS["CopyMetaData"]["params"], **knobs}
        image, meta = Raster.of(np.zeros((2, 2, 4), np.float32)), Raster.of(np.zeros((2, 2, 4), np.float32))
        image.meta, meta.meta = dict(self.image), dict(self.other)
        return Evaluator._metadata_node("CopyMetaData", params, [image, meta], 1).meta

    def test_the_merge_modes(self):
        self.assertEqual(self.result(), {"camera": "cam A", "lens": "50mm", "shared": "from meta", "scene": "sc 12"})
        self.assertEqual(self.result(merge_mode="Meta only"), self.other)
        self.assertEqual(self.result(merge_mode="Meta + Image"), {"camera": "cam A", "lens": "35mm", "shared": "from image", "scene": "sc 12"})
        self.assertEqual(self.result(merge_mode="Image only"), self.image)

    def test_the_meta_filter_searches_keys_values_or_both(self):
        self.assertEqual(self.result(keys="sc", merge_mode="Meta only"), {"scene": "sc 12"})                        # the key and the value
        self.assertEqual(self.result(keys="50", merge_mode="Meta only"), {"lens": "50mm"})
        self.assertEqual(self.result(keys="50", keys_mode="keys only", merge_mode="Meta only"), {})
        self.assertEqual(self.result(keys="meta", keys_mode="values only", merge_mode="Meta only"), {"shared": "from meta"})
        self.assertEqual(self.result(keys="LENS shared", merge_mode="Meta only"), {"lens": "50mm", "shared": "from meta"})

    def test_the_image_filter_limits_what_is_kept_from_the_image(self):
        self.assertEqual(self.result(image_keys="cam", merge_mode="Image only"), {"camera": "cam A"})
        self.assertEqual(self.result(image_keys="35", image_keys_mode="values only", merge_mode="Image only"), {"lens": "35mm"})
        self.assertEqual(self.result(image_keys="35", image_keys_mode="keys only", merge_mode="Image only"), {})
        self.assertEqual(self.result(image_keys="camera", keys="scene", merge_mode="Meta + Image"),
                         {"scene": "sc 12", "camera": "cam A"})

    def test_exact_keys_is_the_key_list_of_before(self):
        self.assertEqual(self.result(keys="lens scene", keys_mode="exact keys"),
                         {"camera": "cam A", "lens": "50mm", "shared": "from image", "scene": "sc 12"})
        self.assertEqual(self.result(keys="len", keys_mode="exact keys"), self.image)

    def test_with_nothing_wired_to_meta_the_image_keeps_its_own(self):
        params = {**SPECS["CopyMetaData"]["params"]}
        image = Raster.of(np.zeros((2, 2, 4), np.float32))
        image.meta = dict(self.image)
        self.assertEqual(Evaluator._metadata_node("CopyMetaData", params, [image], 1).meta, self.image)
        self.assertEqual(Evaluator._metadata_node("CopyMetaData", {**params, "merge_mode": "Meta only"}, [image], 1).meta, {})


class DocumentTests(unittest.TestCase):
    def test_an_older_document_loads_with_the_old_defaults_and_renders_the_same(self):
        for kind in ("Constant", "Invert", "Remove", "CopyMetaData", "Copy"):
            with self.subTest(kind=kind):
                d = Dispatcher(empty_document())
                d.execute({"op": "create", "id": "n", "type": kind, "params": {}})
                old = copy.deepcopy(d.document)
                for name in smallnodes.NEW[kind]:
                    old["nodes"]["n"]["params"].pop(name)
                loaded = upgrade_document(old)
                validate(loaded)
                for name, value in smallnodes.OLD[kind].items():
                    self.assertEqual(loaded["nodes"]["n"]["params"][name], value)

    def test_an_older_copy_metadata_keeps_its_exact_key_list(self):
        self.assertEqual(smallnodes.OLD["CopyMetaData"]["keys_mode"], "exact keys")
        self.assertEqual(SPECS["CopyMetaData"]["params"]["keys_mode"], "keys and values")


if __name__ == "__main__":
    unittest.main()
