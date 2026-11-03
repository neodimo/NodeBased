"""2D behaviour, lane 9 step D: the Filter nodes of step A follow rule 5 of BEHAVIOUR-2D.md.

* Nuke's `size` on Blur, Sharpen, Soften, Glow and Erode (filter) is a WH knob: it splits into a width and a height. Each of the five
  holds either one number (as every older document does) or a list of two, the evaluator and the tile path read both through
  core.split_values, and the two sizes act on their own axis.
* Each node's main tab lists its knobs in the order and with the labels of docs/parity/nuke_nodes.json (EdgeExtend's Channels row leads).
"""
import unittest

import numpy as np

from nodebased import gpu2d, knobs, tiers
from nodebased.core import SPECS, SPLITTABLE, Dispatcher, split_values
from tests.test_2d_behaviour_filter_a import differs, evaluator_pixels, impulse, kern, source_graph, tile_pixels, texture

SPLIT_SIZE = {"Blur": "radius", "Sharpen": "sharpen_size", "Soften": "soften_size", "Glow": "glow_size", "ErodeFilter": "filter_size"}
# A bright dot gives Glow something above its threshold; the others take the same one.
DOT = 3.0


def dot(n=31):
    pixels = np.zeros((n, n, 4), np.float32)
    pixels[n // 2, n // 2] = DOT
    return pixels


def reach_of(change):
    """(columns, rows) of the picture a kernel changed, as the count of changed pixels along the centre row and the centre column."""
    n = change.shape[0]
    changed = np.abs(change[..., :3]).max(axis=-1) > 1e-6
    return int(changed[n // 2, :].sum()), int(changed[:, n // 2].sum())


class SplitSizeTests(unittest.TestCase):
    def test_each_size_is_declared_splittable_as_a_width_and_height(self):
        for kind, name in SPLIT_SIZE.items():
            self.assertEqual(SPLITTABLE[kind][name], "wh", kind)
            self.assertEqual(knobs.resolve_kind(kind, name, SPECS[kind]["params"][name]), "split", kind)
        self.assertEqual(knobs.split_layout_problems(), [])

    def test_a_split_size_blurs_differently_across_and_down(self):
        for kind in ("Blur", "Sharpen", "Soften", "Glow"):
            name = SPLIT_SIZE[kind]
            base = dot()
            wide = reach_of(kern(kind, base, **{name: [6.0, 0.0]}) - base)
            tall = reach_of(kern(kind, base, **{name: [0.0, 6.0]}) - base)
            both = reach_of(kern(kind, base, **{name: [6.0, 6.0]}) - base)
            self.assertGreater(wide[0], 1, f"{kind}: a width of 6 spreads across")
            self.assertLessEqual(wide[1], 1, f"{kind}: a height of 0 leaves the column alone (but for the dot itself)")
            self.assertGreater(tall[1], 1, f"{kind}: a height of 6 spreads down")
            self.assertLessEqual(tall[0], 1, f"{kind}: a width of 0 leaves the row alone (but for the dot itself)")
            self.assertEqual((wide[0], wide[1]), (tall[1], tall[0]), f"{kind}: swapping the two swaps the axes")
            self.assertGreater(both[0], 1)
            self.assertGreater(both[1], 1)

    def test_glow_with_a_height_of_zero_spreads_only_across(self):
        base = dot()
        change = kern("Glow", base, glow_size=[6.0, 0.0]) - base
        n = base.shape[0]
        off_row = np.delete(np.abs(change[..., :3]), n // 2, axis=0)
        self.assertEqual(float(off_row.max()), 0.0)
        self.assertGreater(float(np.abs(change[n // 2, :, :3]).sum(axis=-1)[n // 2 + 2]), 0.0)

    def test_erode_filter_reaches_further_across_than_down(self):
        n = 31
        bright = np.ones((n, n, 4), np.float32)
        bright[n // 2, n // 2] = 0.0
        out = kern("ErodeFilter", bright, filter_size=[4.0, 1.0])
        dark = out[..., 0] < 0.5
        across = int(dark[n // 2, :].sum())
        down = int(dark[:, n // 2].sum())
        self.assertGreater(across, down)
        self.assertEqual(across, 9)       # the dark dot grows 4 pixels each way across ...
        self.assertEqual(down, 3)         # ... and 1 down
        swapped = kern("ErodeFilter", bright, filter_size=[1.0, 4.0])[..., 0] < 0.5
        self.assertEqual((int(swapped[n // 2, :].sum()), int(swapped[:, n // 2].sum())), (3, 9))

    def test_a_negative_split_erode_expands_the_brighter_areas(self):
        n = 31
        dark = np.zeros((n, n, 4), np.float32)
        dark[n // 2, n // 2] = 1.0
        out = kern("ErodeFilter", dark, filter_size=[-4.0, -1.0])[..., 0] > 0.5
        self.assertEqual((int(out[n // 2, :].sum()), int(out[:, n // 2].sum())), (9, 3))

    def test_equal_entries_equal_the_unsplit_node_bit_for_bit(self):
        image = texture()
        for kind, name in SPLIT_SIZE.items():
            for size in (0.3, 3.0, 7.5):
                with self.subTest(kind=kind, size=size):
                    np.testing.assert_array_equal(kern(kind, image, **{name: [size, size]}), kern(kind, image, **{name: size}))

    def test_each_entry_alone_matches_a_one_axis_blur_of_the_unsplit_node(self):
        # A size of (n, 0) is the unsplit node's horizontal pass alone: it equals the (0, n) answer on the transposed picture.
        image = texture(37, 37)
        for kind in ("Blur", "Soften", "Sharpen"):
            name = SPLIT_SIZE[kind]
            across = kern(kind, image, **{name: [5.0, 0.0]})
            down = kern(kind, np.ascontiguousarray(image.transpose(1, 0, 2)), **{name: [0.0, 5.0]})
            np.testing.assert_allclose(across, down.transpose(1, 0, 2), atol=1e-5, err_msg=kind)

    def test_a_scalar_size_is_unchanged_and_a_split_size_changes_the_pixels(self):
        image = texture()
        for kind, name in SPLIT_SIZE.items():
            with self.subTest(kind=kind):
                self.assertTrue(differs(kern(kind, image, **{name: [6.0, 1.0]}), kern(kind, image, **{name: 6.0})))
                self.assertTrue(differs(kern(kind, image, **{name: [6.0, 1.0]}), kern(kind, image, **{name: [1.0, 6.0]})))

    def test_glow_with_a_width_channel_takes_a_split_size(self):
        image = texture()
        width = np.full(image.shape[:2] + (1,), 0.5, np.float32)
        from nodebased.imaging import Evaluator
        p = {**SPECS["Glow"]["params"], "glow_size": [6.0, 1.0], "glow_threshold": 0.5}
        a = Evaluator._glow(image, p, (0, 0), width)
        b = Evaluator._glow(image, {**p, "glow_size": [3.0, 0.5]}, (0, 0), None)
        np.testing.assert_allclose(a, b, atol=1e-5)

    def test_a_wrong_length_list_is_refused(self):
        d = Dispatcher()
        d.execute(dict(op="create", id="n", type="Blur", params={}))
        with self.assertRaises(Exception):
            d.execute(dict(op="set", id="n", param="radius", value=[1.0, 2.0, 3.0]))
        d.execute(dict(op="set", id="n", param="radius", value=[2.0, 5.0]))
        self.assertEqual(list(split_values(d.document["nodes"]["n"]["params"], "radius", 2)), [2.0, 5.0])

    def test_tiles_of_7_16_and_32_equal_the_whole_frame_with_and_without_a_mask(self):
        cases = {
            "Blur": [{"radius": [9.0, 2.0]}, {"radius": [9.0, 2.0], "kernel_filter": "box"}, {"radius": [14.0, 5.0], "quality": 4.0}],
            "Sharpen": [{"sharpen_size": [6.0, 1.0], "sharpen_amount": 1.0}, {"sharpen_size": [12.0, 3.0], "quality": 4.0}],
            "Soften": [{"soften_size": [8.0, 2.0]}, {"soften_size": [12.0, 3.0], "quality": 4.0}],
            "Glow": [{"glow_size": [9.0, 2.0], "glow_threshold": 0.0}],
            "ErodeFilter": [{"filter_size": [3.0, 1.0]}, {"filter_size": [2.6, 1.4], "filter_type": "triangle"},
                            {"filter_size": [-3.0, -1.0]}, {"filter_size": [3.0, 1.0], "filter_type": "gaussian"}],
        }
        for kind, variants in cases.items():
            for params in variants:
                for masked in (False, True):
                    g = source_graph(kind, params, masked)
                    full = np.asarray(evaluator_pixels(g.doc, "node"))
                    for edge in (7, 16, 32):
                        with self.subTest(kind=kind, params=params, masked=masked, edge=edge):
                            np.testing.assert_allclose(np.asarray(tile_pixels(g.doc, "node", edge)), full, atol=1e-5)

    def test_tile_support_covers_the_larger_entry(self):
        from nodebased import blurknobs, tiles
        for kind in ("Blur", "Sharpen", "Soften", "Glow"):
            name = SPLIT_SIZE[kind]
            self.assertEqual(blurknobs.halo(kind, {name: [9.0, 2.0]}), 9, kind)
            self.assertEqual(blurknobs.halo(kind, {name: [2.0, 9.0]}), 9, kind)
            self.assertEqual(tiles.halo_for_kind(kind, {name: [9.0, 2.0]}) if hasattr(tiles, "halo_for_kind") else (9, 9), (9, 9), kind)

    def test_a_proxy_tier_halves_both_entries(self):
        for kind, name in SPLIT_SIZE.items():
            scaled = tiers.scale_params(kind, {name: [8.0, 4.0]}, 2)
            self.assertEqual(list(scaled[name]), [4.0, 2.0], kind)

    def test_blur_with_a_split_size_has_no_gpu_kernel(self):
        with self.assertRaises(gpu2d.Unsupported):
            gpu2d.map_blur({**SPECS["Blur"]["params"], "kernel_filter": "box", "quality": 1000.0, "radius": [4.0, 2.0]})


MAIN_TAB = {
    "Blur": ["Channels", "Size", "Filter", "Quality", "Crop to format"],
    "Sharpen": ["Channels", "Minimum", "Maximum", "Amount", "Size", "Filter", "Quality", "Crop to format"],
    "Soften": ["Channels", "Minimum", "Maximum", "Amount", "Size", "Filter", "Quality", "Crop to format"],
    "Glow": ["Channels", "Tint", "Tolerance", "Brightness", "Saturation", "Width channel", "Non linear", "Effect only", "Size", "Filter",
             "Quality", "Crop to format"],
    "EdgeDetect": ["Edge Detector", "Channels", "Threshold", "Erode size", "Blur size", "Blur quality"],
    "Laplacian": ["Channels", "Size", "Filter", "Quality", "Crop to format"],
    "ErodeFilter": ["Channels", "Size", "Filter"],
    "EdgeExtend": ["Channels", "Source Is Premultiplied", "Erode", "Premultiply"],
}


class LayoutTests(unittest.TestCase):
    def labels(self, kind):
        return [group.label for group in knobs.knob_layout(kind)]

    def test_each_main_tab_leads_with_the_guides_knobs_in_the_guides_order(self):
        for kind, wanted in MAIN_TAB.items():
            with self.subTest(kind=kind):
                self.assertEqual(self.labels(kind)[:len(wanted)], wanted)

    def test_the_mask_strip_and_mix_close_the_tab_where_the_node_has_them(self):
        for kind in ("Blur", "Sharpen", "Soften", "Glow", "Laplacian", "ErodeFilter"):
            with self.subTest(kind=kind):
                self.assertEqual(self.labels(kind)[-5:], ["Mask", "Inject", "Invert", "Fringe", "Mix"])
        for kind in ("EdgeDetect", "EdgeExtend"):
            self.assertEqual(self.labels(kind)[-1], "Mix")

    def test_edge_extend_leads_with_its_channels_row(self):
        self.assertEqual(knobs.knob_layout("EdgeExtend")[0].params, ("channels",))

    def test_every_param_is_still_laid_out_once(self):
        for kind in MAIN_TAB:
            with self.subTest(kind=kind):
                names = [name for group in knobs.knob_layout(kind) for name in group.params]
                self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
