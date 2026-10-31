"""Fidelity step L: TVIScale is Nuke's edge-preserving two-times enlargement.

The solver is a deterministic total-variation regularization of the bilinear 2x enlargement
(nodebased/tviscale.py). There is no Nuke render to compare against, so these tests assert the stated
behaviour (Lambda keeps input noise, Iterations flatten noisy areas, an edge stays sharper than a box blur,
zero iterations is the bilinear target) and the plumbing (2x window, channels, mask strip, mix, bypass, tiles,
named layers, saved graphs from before the change), not pixel equality with Nuke.
"""
import copy
import unittest

import numpy as np

from nodebased.core import Dispatcher, SCHEMA_VERSION, SPECS, bypass_slot, upgrade_document
from nodebased.imaging import Evaluator
from nodebased.raster import Raster
from nodebased.tileexec import TileExecutor
from nodebased.tiers import Region


def params(**changes):
    return {**SPECS["TVIScale"]["params"], **changes}


def run(pixels, mask=None, **changes):
    inputs = [Raster.of(np.asarray(pixels, np.float32))] + ([] if mask is None else [mask])
    return Evaluator._windowed_kernel("TVIScale", params(**changes), inputs, frame=1)


def noisy_flat(size=16, level=0.5, amount=0.05, seed=3):
    rng = np.random.default_rng(seed)
    a = np.full((size, size, 4), level, np.float32)
    a[..., :3] += rng.normal(0, amount, (size, size, 3)).astype(np.float32)
    a[..., 3] = 1.0
    return a


def box_blur(a):
    p = np.pad(a, ((1, 1), (1, 1), (0, 0)), mode="edge")
    return sum(p[dy:dy + a.shape[0], dx:dx + a.shape[1]] for dy in range(3) for dx in range(3)) / 9.0


def interior_std(a):
    return float(a[3:-3, 3:-3, :3].std())


class Graph:
    def __init__(self):
        self.d = Dispatcher()

    def add(self, key, kind, params=None, **inputs):
        self.d.execute(dict(op="create", id=key, type=kind, params=params or {}))
        for slot, source in inputs.items():
            self.d.execute(dict(op="connect", id=key, input=slot, source=source))
        return key

    def set(self, key, **values):
        for name, value in values.items():
            self.d.execute(dict(op="set", id=key, param=name, value=value))

    @property
    def doc(self):
        return self.d.document


def evaluate(document, target):
    return Evaluator().evaluate(dict(document, view=target), frame=1)


def tiles(document, target, tile_edge):
    executor = TileExecutor(evaluator=Evaluator(), tile_edge=tile_edge)
    document = dict(document, view=target)
    assert executor.supports_tiled(document, target)
    region = executor.canvas_region(document, target, frame=1, tier=1)
    return executor.compose_region(document, target, region, frame=1, tier=1).pixels


def noisy_graph(width=40, height=24, **tvi):
    g = Graph()
    g.add("grain", "Checker", dict(width=width, height=height, size=3))
    g.add("node", "TVIScale", tvi, image="grain")
    return g


class DefaultsAndSize(unittest.TestCase):
    def test_new_node_defaults_are_nukes(self):
        p = SPECS["TVIScale"]["params"]
        self.assertEqual((p["tvi_iterations"], p["tvi_lambda"], p["tvi_a"]), (5, 5.0, 0.0001))
        self.assertEqual((p["channels"], p["maskChannelInput"], p["invert_mask"], p["inject"], p["fringe"], p["mix"]),
                         ("all", "none", 0, 0, 0, 1.0))
        self.assertEqual(p["legacy_scale"], 0)

    def test_output_window_is_exactly_twice_the_input_display(self):
        for w, h in ((8, 8), (13, 7), (1, 1)):
            out = run(np.zeros((h, w, 4), np.float32))
            self.assertEqual((out.display.width, out.display.height), (2 * w, 2 * h))
            self.assertEqual(out.pixels.shape, (2 * h, 2 * w, 4))
            self.assertEqual((out.data.width, out.data.height), (2 * w, 2 * h))

    def test_power_does_not_resize_a_new_node(self):
        for power in (-2, 0, 3):
            self.assertEqual(run(np.zeros((4, 6, 4), np.float32), power=power).pixels.shape, (8, 12, 4))

    def test_graph_evaluation_is_twice_the_plate(self):
        g = noisy_graph()
        self.assertEqual(evaluate(g.doc, "node").shape[:2], (48, 80))


class Solver(unittest.TestCase):
    def test_zero_iterations_is_the_bilinear_enlargement(self):
        a = noisy_flat()
        coords = (np.arange(32, dtype=np.float32) + 0.5) / 2 - 0.5
        bilinear = Evaluator._resample(a, np.tile(coords, (32, 1)), np.tile(coords[:, None], (1, 32)), "bilinear", clamp=True)
        np.testing.assert_array_equal(run(a, tvi_iterations=0).pixels, bilinear)
        # Edge pixels are carried outward: a flat picture stays flat to its border.
        flat = run(np.full((4, 4, 4), 0.5, np.float32), tvi_iterations=0).pixels
        np.testing.assert_allclose(flat, 0.5)

    def test_more_iterations_flatten_a_noisy_flat_area(self):
        a = noisy_flat()
        stds = [interior_std(run(a, tvi_iterations=n).pixels) for n in (0, 1, 3, 10, 30)]
        self.assertTrue(all(later < earlier for earlier, later in zip(stds, stds[1:])), stds)
        self.assertLess(stds[-1], 0.85 * stds[0])

    def test_larger_lambda_keeps_more_of_the_input_noise(self):
        a = noisy_flat()
        stds = [interior_std(run(a, tvi_lambda=lam).pixels) for lam in (0.5, 5.0, 50.0, 500.0)]
        self.assertTrue(all(later > earlier for earlier, later in zip(stds, stds[1:])), stds)

    def test_a_changes_the_result_and_zero_is_floored_not_divided(self):
        a = noisy_flat()
        small, large = run(a, tvi_a=0.0001).pixels, run(a, tvi_a=1.0).pixels
        self.assertGreater(float(np.abs(small - large).max()), 1e-3)
        zero = run(a, tvi_a=0.0).pixels
        self.assertTrue(np.isfinite(zero).all())
        # A tiny but positive A is the same solver, so zero (floored) sits next to it.
        self.assertLess(float(np.abs(zero - small).max()), 1e-2)

    def test_sharp_edge_stays_sharper_than_a_box_blur(self):
        a = np.zeros((16, 16, 4), np.float32)
        a[:, 8:, :3] = 1.0
        a[..., 3] = 1.0
        rng = np.random.default_rng(5)
        a[..., :3] += rng.normal(0, 0.02, (16, 16, 3)).astype(np.float32)
        tv = run(a, tvi_iterations=8).pixels
        base = run(a, tvi_iterations=0).pixels
        box = box_blur(base)
        step = lambda x: float(np.abs(np.diff(x[8:24, :, 0], axis=1)).max())     # steepest horizontal step through the edge
        self.assertGreater(step(tv), 1.3 * step(box))
        self.assertGreater(step(tv), 0.5)
        # ...while the noise on the flat sides is still smoothed.
        self.assertLess(float(tv[8:24, 2:10, 0].std()), float(base[8:24, 2:10, 0].std()))
        self.assertLess(float(tv[8:24, 22:30, 0].std()), float(base[8:24, 22:30, 0].std()))

    def test_result_is_bounded_by_the_input_range_and_deterministic(self):
        a = noisy_flat()
        first, second = run(a).pixels, run(a).pixels
        np.testing.assert_array_equal(first, second)
        base = run(a, tvi_iterations=0).pixels
        self.assertLessEqual(float(first.max()), float(base.max()) + 1e-6)
        self.assertGreaterEqual(float(first.min()), float(base.min()) - 1e-6)


class Chrome(unittest.TestCase):
    def test_channels_limit_the_smoothing(self):
        a = noisy_flat()
        full = run(a).pixels
        base = run(a, tvi_iterations=0).pixels
        red = run(a, channels="red").pixels
        np.testing.assert_array_equal(red[..., 0], full[..., 0])
        np.testing.assert_array_equal(red[..., 1:], base[..., 1:])
        self.assertFalse(np.array_equal(full[..., 1], base[..., 1]))
        np.testing.assert_array_equal(run(a, channels="none").pixels, base)
        alpha_only = run(a, channels="alpha").pixels
        np.testing.assert_array_equal(alpha_only[..., :3], base[..., :3])

    def test_mask_gates_the_smoothing_on_the_enlarged_grid(self):
        a = noisy_flat()
        mask = np.zeros((16, 16, 4), np.float32)
        mask[:, 8:, 3] = 1.0
        full = run(a).pixels
        base = run(a, tvi_iterations=0).pixels
        out = run(a, mask=Raster.of(mask)).pixels
        np.testing.assert_array_equal(out[:, :14], base[:, :14])         # outside the matte: the plain enlargement
        np.testing.assert_allclose(out[:, 18:], full[:, 18:], atol=1e-6)  # inside it: the smoothed picture
        inverted = run(a, mask=Raster.of(mask), invert_mask=1).pixels
        np.testing.assert_array_equal(inverted[:, 18:], base[:, 18:])
        np.testing.assert_allclose(inverted[:, :14], full[:, :14], atol=1e-6)

    def test_mask_channel_without_a_mask_input_reads_the_image(self):
        a = noisy_flat()
        a[..., 0] = 0.0
        out = run(a, maskChannelInput="rgba.red").pixels
        np.testing.assert_array_equal(out, run(a, tvi_iterations=0).pixels)

    def test_fringe_only_works_the_soft_edge_of_the_mask(self):
        a = noisy_flat()
        hard = np.zeros((16, 16, 4), np.float32)
        hard[:, 8:, 3] = 1.0
        base = run(a, tvi_iterations=0).pixels
        # A hard mask is only soft where the enlargement blends its two sides (two columns): fringe leaves the
        # plain enlargement everywhere else.
        fringed = run(a, mask=Raster.of(hard), fringe=1).pixels
        np.testing.assert_array_equal(fringed[:, :15], base[:, :15])
        np.testing.assert_array_equal(fringed[:, 17:], base[:, 17:])
        self.assertFalse(np.array_equal(fringed[:, 15:17], base[:, 15:17]))
        soft = np.zeros((16, 16, 4), np.float32)
        soft[..., 3] = np.linspace(0, 1, 16, dtype=np.float32)[None, :]
        edge = run(a, mask=Raster.of(soft), fringe=1).pixels
        plain = run(a, mask=Raster.of(soft)).pixels
        self.assertFalse(np.array_equal(edge, base))
        self.assertFalse(np.array_equal(edge, plain))

    def test_mix_dissolves_between_the_enlargement_and_the_smoothed_picture(self):
        a = noisy_flat()
        full, base = run(a).pixels, run(a, tvi_iterations=0).pixels
        np.testing.assert_array_equal(run(a, mix=0.0).pixels, base)
        np.testing.assert_allclose(run(a, mix=0.25).pixels, 0.25 * full + 0.75 * base, atol=1e-6)

    def test_inject_writes_the_enlarged_matte_as_the_mask_layer(self):
        a = noisy_flat()
        mask = np.zeros((16, 16, 4), np.float32)
        mask[..., 3] = 0.5
        out = run(a, mask=Raster.of(mask), inject=1)
        self.assertIn("mask", out.layers)
        self.assertEqual(out.layers["mask"].pixels.shape, (32, 32, 4))
        np.testing.assert_allclose(out.layers["mask"].pixels[..., 3], 0.5)
        self.assertIsNone(run(a, mask=Raster.of(mask)).layers)

    def test_mismatched_mask_display_is_refused(self):
        with self.assertRaises(ValueError):
            run(noisy_flat(), mask=Raster.of(np.zeros((8, 8, 4), np.float32)))

    def test_a_named_layer_is_enlarged_and_can_be_smoothed(self):
        a = noisy_flat()
        depth = Raster.of(noisy_flat(seed=9))
        source = Raster(a, Region(0, 0, 16, 16), Region(0, 0, 16, 16), {"depth": depth})
        out = Evaluator._windowed_kernel("TVIScale", params(channels="rgba"), [source], frame=1)
        self.assertEqual(out.layers["depth"].pixels.shape, (32, 32, 4))
        plain = out.layers["depth"].pixels
        out = Evaluator._windowed_kernel("TVIScale", params(channels="rgba, depth"), [source], frame=1)
        self.assertLess(interior_std(out.layers["depth"].pixels), interior_std(plain))
        with self.assertRaises(ValueError):
            Evaluator._windowed_kernel("TVIScale", params(channels="missing"), [source], frame=1)


class GraphLevel(unittest.TestCase):
    def test_every_knob_changes_the_pixels(self):
        g = noisy_graph()
        before = evaluate(g.doc, "node")
        g.add("mask", "Constant", dict(width=40, height=24, red=0.0, green=0.0, blue=0.0, alpha=0.0))
        for name, value in (("tvi_iterations", 12), ("tvi_lambda", 0.5), ("tvi_a", 0.5), ("channels", "red"),
                            ("mix", 0.5), ("filter", "nearest")):
            with self.subTest(knob=name):
                h = noisy_graph()
                h.set("node", **{name: value})
                self.assertFalse(np.array_equal(evaluate(h.doc, "node"), before))
        with self.subTest(knob="mask"):
            g.d.execute(dict(op="connect", id="node", input="mask", source="mask"))
            self.assertFalse(np.array_equal(evaluate(g.doc, "node"), before))
        with self.subTest(knob="maskChannelInput"):
            h = noisy_graph()
            h.set("node", maskChannelInput="rgba.green")
            self.assertFalse(np.array_equal(evaluate(h.doc, "node"), before))

    def test_full_frame_equals_tiles_on_a_multi_tile_image(self):
        for extra in ({}, {"tvi_iterations": 9, "tvi_lambda": 1.0}, {"channels": "rgb", "mix": 0.6}):
            g = noisy_graph(**extra)
            full = evaluate(g.doc, "node")
            for edge in (16, 7, 32):
                with self.subTest(extra=extra, tile_edge=edge):
                    np.testing.assert_array_equal(tiles(g.doc, "node", edge), full)

    def test_tiles_with_a_mask_equal_full_frame(self):
        g = noisy_graph()
        g.add("mask", "Ramp", dict(width=40, height=24))
        g.d.execute(dict(op="connect", id="node", input="mask", source="mask"))
        full = evaluate(g.doc, "node")
        np.testing.assert_array_equal(tiles(g.doc, "node", 11), full)

    def test_inject_and_layer_channels_use_the_whole_image_path(self):
        g = noisy_graph(inject=1)
        g.add("mask", "Constant", dict(width=40, height=24, alpha=1.0))
        g.d.execute(dict(op="connect", id="node", input="mask", source="mask"))
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=16)
        self.assertFalse(executor.supports_tiled(dict(g.doc, view="node"), "node"))
        h = noisy_graph(channels="rgba, depth.Z")
        self.assertFalse(executor.supports_tiled(dict(h.doc, view="node"), "node"))

    def test_bypass_passes_the_unscaled_input_on_both_paths(self):
        g = noisy_graph()
        plate = evaluate(g.doc, "grain")
        g.d.execute(dict(op="disable", id="node", value=True))
        self.assertEqual(bypass_slot(g.doc["nodes"]["node"]), "image")
        np.testing.assert_array_equal(evaluate(g.doc, "node"), plate)
        np.testing.assert_array_equal(tiles(g.doc, "node", 16), plate)

    def test_two_chained_scales_make_four_times(self):
        g = noisy_graph()
        g.add("again", "TVIScale", {}, image="node")
        self.assertEqual(evaluate(g.doc, "again").shape[:2], (96, 160))
        np.testing.assert_array_equal(tiles(g.doc, "again", 24), evaluate(g.doc, "again"))


class OldDocuments(unittest.TestCase):
    def old_document(self, mask=False, **tvi):
        g = Graph()
        g.add("plate", "Checker", dict(width=8, height=8, size=2))
        g.add("node", "TVIScale", dict(tvi), image="plate")
        if mask:
            g.add("mask", "Constant", dict(width=8, height=8, red=1.0, green=1.0, blue=1.0, alpha=1.0))
            g.d.execute(dict(op="connect", id="node", input="mask", source="mask"))
        document = copy.deepcopy(g.doc)
        document["version"] = 27
        node = document["nodes"]["node"]["params"]
        for name in ("legacy_scale", "tvi_iterations", "tvi_lambda", "tvi_a", "channels", "maskChannelInput",
                     "invert_mask", "inject", "fringe"):
            node.pop(name, None)
        return document

    def test_schema_version_moved_and_old_nodes_become_legacy(self):
        upgraded = upgrade_document(self.old_document(power=1, filter="nearest"))
        self.assertEqual(upgraded["version"], SCHEMA_VERSION)
        params = upgraded["nodes"]["node"]["params"]
        self.assertEqual(params["legacy_scale"], 1)
        self.assertEqual(params["power"], 1)
        self.assertEqual(params["tvi_iterations"], 5)

    def test_missing_power_in_an_old_node_stays_the_identity(self):
        document = self.old_document()
        document["nodes"]["node"]["params"].pop("power", None)
        upgraded = upgrade_document(document)
        self.assertEqual(upgraded["nodes"]["node"]["params"]["power"], 0)
        np.testing.assert_array_equal(evaluate(upgraded, "node"), evaluate(upgraded, "plate"))

    def test_old_power_and_filter_pixels_are_unchanged(self):
        plate = None
        for power, shape in ((1, (16, 16)), (-1, (4, 4)), (2, (32, 32)), (0, (8, 8))):
            with self.subTest(power=power):
                upgraded = upgrade_document(self.old_document(power=power, filter="nearest"))
                plate = evaluate(upgraded, "plate")
                out = evaluate(upgraded, "node")
                self.assertEqual(out.shape[:2], shape)
                if power == 1:
                    np.testing.assert_array_equal(out, np.repeat(np.repeat(plate, 2, axis=0), 2, axis=1))
                if power == -1:
                    np.testing.assert_array_equal(out, plate[::2, ::2])
                if power == 0:
                    np.testing.assert_array_equal(out, plate)
                np.testing.assert_array_equal(tiles(upgraded, "node", 3), out)

    def test_old_mask_and_mix_blend_against_the_unscaled_picture_as_before(self):
        document = self.old_document(mask=True, power=1, filter="nearest", mix=0.5)
        upgraded = upgrade_document(document)
        out = evaluate(upgraded, "node")
        plate = evaluate(upgraded, "plate")
        scaled = np.repeat(np.repeat(plate, 2, axis=0), 2, axis=1)
        # The old node mixed against the unscaled picture at the origin and gated by the unscaled mask, so only the
        # mask's own 8x8 corner of the 16x16 result is non-black.
        expected = np.zeros_like(scaled)
        expected[:8, :8] = 0.5 * scaled[:8, :8] + 0.5 * plate
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_a_node_created_now_is_not_legacy(self):
        g = noisy_graph()
        self.assertEqual(g.doc["nodes"]["node"]["params"]["legacy_scale"], 0)
        self.assertEqual(upgrade_document(copy.deepcopy(g.doc))["nodes"]["node"]["params"]["legacy_scale"], 0)


if __name__ == "__main__":
    unittest.main()
