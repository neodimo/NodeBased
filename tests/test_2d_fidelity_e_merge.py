"""Fidelity step E, part 2: Merge takes Nuke's knobs. Operation math against the guide's algorithms (tools/nuke_fidelity_cases.py holds all
thirty), the video colorspace, alpha masking, the channel sets, the output box, layers, metadata and range, tiles against the
evaluator, bypass and old documents."""
import copy
import unittest

import numpy as np

from nodebased import mergeknobs
from nodebased.core import SPECS, Dispatcher, bypass_slot, empty_document, upgrade_document, validate
from nodebased.imaging import Evaluator, _input_clip_range
from nodebased.raster import Raster
from nodebased.tileexec import TileExecutor
from nodebased.tiers import Region

H, W = 6, 8


def px(*v):
    return np.array([[list(v)]], np.float32)


def merge(a, b, mask=None, **knobs):
    inputs = [Raster.of(a), Raster.of(b)] + ([] if mask is None else [Raster.of(mask)])
    return Evaluator._windowed_kernel("Merge", {**SPECS["Merge"]["params"], **knobs}, inputs)


class OperationFormulaTests(unittest.TestCase):
    """The thirty operations against tools/nuke_fidelity_cases.merge_reference, on a fixture covering negatives and HDR."""
    def test_every_scored_operation_matches_the_guide(self):
        from tools.nuke_fidelity_cases import MERGE_OPERATIONS_GUIDE, MERGE_UNSCORED, make_fixtures, merge_reference
        fx = make_fixtures()
        for op in MERGE_OPERATIONS_GUIDE:
            if op in MERGE_UNSCORED:
                continue
            with self.subTest(op=op):
                reference, valid = merge_reference(op, fx["A"], fx["B"])
                got = Evaluator._kernel("Merge", {**SPECS["Merge"]["params"], "operation": op}, [fx["A"], fx["B"]])
                np.testing.assert_allclose(got[valid], reference[valid], rtol=1e-5, atol=1e-5)
                self.assertTrue(np.isfinite(got).all(), op)

    def test_hand_computed_values(self):
        a, b = px(0.5, -0.5, 2.0, 0.5), px(-0.25, -0.5, 0.0, 1.0)
        cases = {"multiply": (-0.125, -0.5, 0.0, 0.5), "divide": (0.5, -0.5, 2.0, 0.5), "screen": (0.5, -0.5, 2.0, 1.0),
                 "geometric": (0.0, 0.0, 0.0, 2 * 0.5 / 1.5)}
        for op, expected in cases.items():
            with self.subTest(op=op):
                np.testing.assert_allclose(Evaluator._kernel("Merge", {**SPECS["Merge"]["params"], "operation": op}, [a, b])[0, 0],
                                           expected, atol=1e-6)


class VideoColorspaceTests(unittest.TestCase):
    def test_the_curve_round_trips_and_is_odd_about_zero(self):
        x = np.array([-4.0, -0.5, -0.001, 0.0, 0.001, 0.18, 0.5, 1.0, 12.0], np.float32)
        np.testing.assert_allclose(mergeknobs.from_srgb(mergeknobs.to_srgb(x)), x, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(mergeknobs.to_srgb(-x), -mergeknobs.to_srgb(x), atol=1e-7)
        self.assertAlmostEqual(float(mergeknobs.to_srgb(np.float32(0.18))), 0.4614, places=3)
        self.assertAlmostEqual(float(mergeknobs.to_srgb(np.float32(1.0))), 1.0, places=6)

    def test_plus_happens_in_the_video_colorspace(self):
        a, b = px(0.2, 0.2, 0.2, 1.0), px(0.1, 0.1, 0.1, 1.0)
        linear = merge(a, b, operation="plus").pixels
        video = merge(a, b, operation="plus", sRGB=1).pixels
        expected = mergeknobs.from_srgb(mergeknobs.to_srgb(np.float32(0.2)) + mergeknobs.to_srgb(np.float32(0.1)))
        np.testing.assert_allclose(video[0, 0, :3], [expected] * 3, rtol=1e-5)
        self.assertNotAlmostEqual(float(video[0, 0, 0]), float(linear[0, 0, 0]), places=2)
        self.assertEqual(float(video[0, 0, 3]), 2.0)                       # alpha is not run through the curve

    def test_a_copy_through_the_curve_changes_nothing(self):
        a, b = px(0.37, -0.2, 3.5, 0.6), px(0.1, 0.2, 0.3, 1.0)
        np.testing.assert_allclose(merge(a, b, operation="copy", sRGB=1).pixels, a, rtol=1e-5, atol=1e-6)


class AlphaMaskingTests(unittest.TestCase):
    def test_an_image_is_unchanged_where_the_other_has_no_alpha_and_alpha_screens(self):
        a = np.array([[[0.5, 0.5, 0.5, 0.0], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]], np.float32)
        b = np.array([[[0.2, 0.2, 0.2, 0.5], [0.2, 0.2, 0.2, 0.0], [0.2, 0.2, 0.2, 0.5]]], np.float32)
        out = merge(a, b, operation="plus", screen_alpha=1).pixels
        np.testing.assert_allclose(out[0, 0, :3], [0.2] * 3, atol=1e-6)           # A has no alpha: B comes through
        np.testing.assert_allclose(out[0, 1, :3], [0.5] * 3, atol=1e-6)           # B has no alpha: A comes through
        np.testing.assert_allclose(out[0, 2, :3], [0.7] * 3, atol=1e-6)           # both: the operation
        np.testing.assert_allclose(out[0, :, 3], [0.5, 0.5, 0.75], atol=1e-6)     # a + b - a*b
        plain = merge(a, b, operation="plus").pixels
        np.testing.assert_allclose(plain[0, 0, :3], [0.7] * 3, atol=1e-6)
        np.testing.assert_allclose(plain[0, 2, 3], 1.0, atol=1e-6)               # the same math on alpha as on colour

    def test_it_changes_nothing_on_the_operations_that_already_do_that(self):
        rng = np.random.default_rng(1)
        a, b = rng.uniform(0, 1, (H, W, 4)).astype(np.float32), rng.uniform(0, 1, (H, W, 4)).astype(np.float32)
        for op in ("over", "under", "atop", "in", "out", "mask", "stencil", "xor"):
            with self.subTest(op=op):
                np.testing.assert_array_equal(merge(a, b, operation=op, screen_alpha=1).pixels, merge(a, b, operation=op).pixels)


class ChannelSetTests(unittest.TestCase):
    def test_a_channel_left_out_of_a_counts_as_zero(self):
        a, b = px(0.5, 0.4, 0.3, 0.5), px(0.1, 0.1, 0.1, 1.0)
        out = merge(a, b, operation="plus", Achannels="red").pixels
        np.testing.assert_allclose(out[0, 0], [0.6, 0.1, 0.1, 1.0], atol=1e-6)
        none = merge(a, b, operation="over", Achannels="none").pixels
        np.testing.assert_allclose(none[0, 0], b[0, 0], atol=1e-6)

    def test_a_without_its_alpha_covers_nothing_so_over_adds(self):
        a, b = px(0.5, 0.4, 0.3, 0.5), px(0.1, 0.1, 0.1, 1.0)
        out = merge(a, b, operation="over", Achannels="rgb").pixels
        np.testing.assert_allclose(out[0, 0, :3], [0.6, 0.5, 0.4], atol=1e-6)         # A + B(1 - 0)

    def test_a_channel_left_out_of_b_counts_as_zero(self):
        a, b = px(0.5, 0.4, 0.3, 1.0), px(0.1, 0.2, 0.3, 1.0)
        out = merge(a, b, operation="plus", Bchannels="green").pixels
        np.testing.assert_allclose(out[0, 0], [0.5, 0.6, 0.3, 1.0], atol=1e-6)

    def test_output_writes_only_the_chosen_channels_and_the_rest_are_b(self):
        a, b = px(0.5, 0.4, 0.3, 0.5), px(0.1, 0.2, 0.3, 1.0)
        full = merge(a, b, operation="plus").pixels
        for channels, written in (("alpha", [3]), ("rgb", [0, 1, 2]), ("green", [1]), ("none", [])):
            with self.subTest(output=channels):
                out = merge(a, b, operation="plus", merge_output=channels).pixels[0, 0]
                for c in range(4):
                    self.assertAlmostEqual(float(out[c]), float(full[0, 0, c] if c in written else b[0, 0, c]), places=6)

    def test_the_mask_and_mix_gate_against_b_with_the_new_knobs_on(self):
        a, b = px(0.5, 0.4, 0.3, 0.5), px(0.1, 0.2, 0.3, 1.0)
        mask = px(0, 0, 0, 0.5)
        out = merge(a, b, mask, operation="plus", sRGB=1, mix=0.5).pixels
        full = merge(a, b, operation="plus", sRGB=1).pixels
        np.testing.assert_allclose(out, b + (full - b) * 0.25, atol=1e-6)


class LayerTests(unittest.TestCase):
    def raster(self, value, alpha=1.0, layers=None):
        pixels = np.full((H, W, 4), value, np.float32)
        pixels[..., 3] = alpha
        return Raster(pixels, Region(0, 0, W, H), Region(0, 0, W, H), layers)

    def test_a_named_layer_can_be_the_a_input_and_the_output(self):
        a = self.raster(0.5, layers={"fx": self.raster(0.25)})
        b = self.raster(0.1, layers={"fx": self.raster(0.1)})
        out = Evaluator._windowed_kernel("Merge", {**SPECS["Merge"]["params"], "operation": "plus", "Achannels": "fx", "merge_output": "fx"},
                                         [a, b])
        np.testing.assert_allclose(out.pixels[..., :3], 0.1)                      # the main picture is B, untouched
        np.testing.assert_allclose(out.layers["fx"].pixels[..., :3], 0.1 + 0.25, atol=1e-6)

    def test_also_merge_merges_the_other_layers_the_same_way(self):
        a = self.raster(0.5, layers={"fx": self.raster(0.25), "depth": self.raster(0.7)})
        b = self.raster(0.1, layers={"fx": self.raster(0.1)})
        out = Evaluator._windowed_kernel("Merge", {**SPECS["Merge"]["params"], "operation": "plus", "also_merge": "fx"}, [a, b])
        np.testing.assert_allclose(out.pixels[..., :3], 0.6, atol=1e-6)
        np.testing.assert_allclose(out.layers["fx"].pixels[..., :3], 0.35, atol=1e-6)
        self.assertNotIn("depth", out.layers)
        everything = Evaluator._windowed_kernel("Merge", {**SPECS["Merge"]["params"], "operation": "plus", "also_merge": "all"}, [a, b])
        np.testing.assert_allclose(everything.layers["depth"].pixels[..., :3], 0.7, atol=1e-6)      # B has none: B counts as zero

    def test_a_layer_the_input_does_not_carry_is_named_in_the_error(self):
        a = self.raster(0.5)
        with self.assertRaisesRegex(ValueError, "nope"):
            Evaluator._windowed_kernel("Merge", {**SPECS["Merge"]["params"], "Achannels": "nope"}, [a, self.raster(0.1)])


class BoundingBoxTests(unittest.TestCase):
    def rasters(self):
        display = Region(0, 0, 16, 12)
        a = Raster(np.full((4, 6, 4), 0.5, np.float32), Region(2, 2, 6, 4), display)
        b = Raster(np.full((6, 6, 4), 0.25, np.float32), Region(5, 3, 6, 6), display)
        return a, b

    def test_the_output_box_is_the_union_the_intersection_a_or_b(self):
        a, b = self.rasters()
        expected = {"union": a.data.union(b.data), "intersection": a.data.intersect(b.data), "A": a.data, "B": b.data}
        for choice, region in expected.items():
            with self.subTest(set_bbox_to=choice):
                out = Evaluator._windowed_kernel("Merge", {**SPECS["Merge"]["params"], "operation": "plus", "merge_bbox": choice}, [a, b])
                self.assertEqual(out.data, region)
                self.assertEqual(out.pixels.shape[:2], (region.height, region.width))

    def test_pixels_inside_the_box_are_the_merge_and_nothing_outside_is_kept(self):
        a, b = self.rasters()
        out = Evaluator._windowed_kernel("Merge", {**SPECS["Merge"]["params"], "operation": "plus", "merge_bbox": "intersection"}, [a, b])
        np.testing.assert_allclose(out.pixels, 0.75, atol=1e-6)


class MetadataAndRangeTests(unittest.TestCase):
    def test_metadata_follows_metainput(self):
        a = Raster.of(np.zeros((2, 2, 4), np.float32)); a.meta = {"who": "a", "only_a": 1}
        b = Raster.of(np.zeros((2, 2, 4), np.float32)); b.meta = {"who": "b"}
        from nodebased.imaging import _inherited_metadata
        base = SPECS["Merge"]["params"]
        self.assertEqual(_inherited_metadata("Merge", [a, b], {**base, "metainput": "B"}), {"who": "b"})
        self.assertEqual(_inherited_metadata("Merge", [a, b], {**base, "metainput": "A"}), {"who": "a", "only_a": 1})
        self.assertEqual(_inherited_metadata("Merge", [a, b], {**base, "metainput": "All"}), {"who": "b", "only_a": 1})

    def test_range_follows_rangeinput(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "a", "type": "Constant", "params": {"first": 1, "last": 10}})
        d.execute({"op": "create", "id": "b", "type": "Constant", "params": {"first": 5, "last": 40}})
        d.execute({"op": "create", "id": "m", "type": "Merge"})
        d.execute({"op": "connect", "id": "m", "input": "A", "source": "a"})
        d.execute({"op": "connect", "id": "m", "input": "B", "source": "b"})
        for choice, span in (("B", (5, 40)), ("A", (1, 10)), ("All", (1, 40))):
            with self.subTest(rangeinput=choice):
                d.execute({"op": "set", "id": "m", "param": "rangeinput", "value": choice})
                self.assertEqual(_input_clip_range(d.document, "m"), span)


class MergeDocumentTests(unittest.TestCase):
    def graph(self, **knobs):
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "a", "type": "Ramp", "params": {
            "width": 20, "height": 12, "p1_x": 19.0, "color0_red": 0.0, "color1_red": 3.0, "color0_green": 0.2, "color1_green": 0.9,
            "color0_alpha": 0.3, "color1_alpha": 1.0}})
        d.execute({"op": "create", "id": "b", "type": "Constant", "params": {"width": 20, "height": 12, "red": 0.3, "green": 0.2, "blue": 0.1, "alpha": 0.8}})
        d.execute({"op": "create", "id": "m", "type": "Merge", "params": {"operation": "plus", **knobs}})
        d.execute({"op": "connect", "id": "m", "input": "A", "source": "a"})
        d.execute({"op": "connect", "id": "m", "input": "B", "source": "b"})
        d.execute({"op": "view", "id": "m"})
        return d

    def test_tiles_equal_the_evaluator_for_every_array_level_knob(self):
        for knobs in ({"sRGB": 1}, {"screen_alpha": 1}, {"Achannels": "rgb", "Bchannels": "rgba"}, {"merge_output": "rgb"},
                      {"sRGB": 1, "screen_alpha": 1, "Achannels": "red", "merge_output": "red", "mix": 0.7},
                      {"operation": "multiply"}, {"operation": "screen", "sRGB": 1}, {"operation": "hard-light"}):
            with self.subTest(knobs=knobs):
                d = self.graph(**knobs)
                executor = TileExecutor(evaluator=Evaluator(), tile_edge=8)
                expected = Evaluator().evaluate_raster(d.document, target="m").pixels
                region = executor.canvas_region(d.document, "m", frame=1, tier=1)
                self.assertTrue(executor.supports_tiled(d.document, "m"))
                np.testing.assert_allclose(executor.compose_region(d.document, "m", region, frame=1, tier=1).pixels, expected, atol=1e-6)

    def test_a_box_other_than_the_union_and_layers_fall_back_to_the_whole_image(self):
        for knobs in ({"merge_bbox": "A"}, {"merge_bbox": "intersection"}, {"also_merge": "all"}, {"merge_output": "fx"}):
            with self.subTest(knobs=knobs):
                d = self.graph(**knobs)
                self.assertFalse(TileExecutor(evaluator=Evaluator(), tile_edge=8).supports_tiled(d.document, "m"))

    def test_bypass_passes_b(self):
        d = self.graph(sRGB=1, merge_output="alpha")
        d.execute({"op": "disable", "id": "m", "value": True})
        self.assertEqual(bypass_slot(d.document["nodes"]["m"]), "B")
        np.testing.assert_array_equal(Evaluator().evaluate(dict(d.document, view="m"), "m", frame=1),
                                      Evaluator().evaluate(dict(d.document, view="b"), "b", frame=1))

    def test_an_older_document_gets_the_defaults_and_draws_the_same(self):
        d = self.graph(operation="over")
        old = copy.deepcopy(d.document)
        for name in mergeknobs.NEW:
            old["nodes"]["m"]["params"].pop(name)
        loaded = upgrade_document(old)
        validate(loaded)
        for name, value in mergeknobs.OLD.items():
            self.assertEqual(loaded["nodes"]["m"]["params"][name], value)
        a = Evaluator().evaluate_raster(dict(loaded, view="a"), target="a").pixels
        b = Evaluator().evaluate_raster(dict(loaded, view="b"), target="b").pixels
        got = Evaluator().evaluate_raster(dict(loaded, view="m"), target="m").pixels
        np.testing.assert_array_equal(got, Evaluator._merge_gated("over", a, b, 1.0, None))

    def test_the_new_knobs_validate(self):
        d = self.graph(sRGB=1, screen_alpha=1, merge_bbox="intersection", metainput="All", rangeinput="A", Achannels="rgb",
                       Bchannels="red", merge_output="alpha", also_merge="all")
        validate(d.document)
        with self.assertRaises(ValueError):
            Dispatcher(empty_document()).execute({"op": "create", "id": "m", "type": "Merge", "params": {"merge_bbox": "nowhere"}})


if __name__ == "__main__":
    unittest.main()
