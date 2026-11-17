"""Lane 2 step W1 (November 17, 2026): the colour, keying and channel panels read as the guide's.

Covers the defaults that follow Nuke (`channels` all on Invert, Multiply, Add, Gamma, Clamp and Posterize; IBKGizmo's screen type
and luminance level; ChromaKeyer's despill bias as a colour) with the older documents keeping their pixels, and the labels and order
the review book's worklist asked for.
"""
import copy
import unittest

import numpy as np

from nodebased import knobs
from nodebased.core import SPECS, upgrade_document
from nodebased.imaging import Evaluator
from tests.test_2d_parity_group_3b import Graph, evaluator_pixels, kernel, tile_pixels

ALL_KINDS = ("Invert", "Multiply", "Add", "Gamma", "Clamp", "Posterize")


def image():
    rng = np.random.RandomState(3)
    out = rng.rand(12, 16, 4).astype(np.float32) * 0.8 + 0.1
    return out


def labels(kind):
    return [g.label for g in knobs.knob_layout(kind) if g.kind != "legacy"]


class ChannelsDefaultTests(unittest.TestCase):
    def test_a_new_node_starts_on_all_channels(self):
        for kind in ALL_KINDS:
            with self.subTest(kind=kind):
                self.assertEqual(SPECS[kind]["params"]["channels"], "all")

    def test_all_reaches_alpha_and_rgb_leaves_it(self):
        src = image()
        for kind, params in (("Invert", {}), ("Multiply", {"multiply": 0.5}), ("Add", {"offset": 0.1}), ("Gamma", {"gamma": 2.0}),
                             ("Posterize", {"colors": 3})):
            with self.subTest(kind=kind):
                everything = kernel(kind, src, **params)
                self.assertFalse(np.array_equal(everything[..., 3], src[..., 3]), "the default reaches alpha")
                rgb_only = kernel(kind, src, channels="rgb", **params)
                np.testing.assert_array_equal(rgb_only[..., 3], src[..., 3])
                np.testing.assert_array_equal(rgb_only[..., :3], everything[..., :3])
                rgba = kernel(kind, src, channels="rgba", **params)
                np.testing.assert_array_equal(rgba, everything)

    def test_clamp_all_reaches_alpha(self):
        src = image() * 3.0
        out = kernel("Clamp", src)
        self.assertLessEqual(float(out.max()), 1.0)
        self.assertLessEqual(float(out[..., 3].max()), 1.0)

    def test_evaluator_and_tiles_agree_with_the_default(self):
        for kind, params in (("Invert", {}), ("Multiply", {"multiply": 0.5}), ("Gamma", {"gamma": 2.0})):
            with self.subTest(kind=kind):
                g = Graph()
                g.add("src", "Constant", {"red": 0.4, "green": 0.5, "blue": 0.6, "alpha": 0.75})
                g.add("node", kind, params, image="src")
                full = evaluator_pixels(g.doc, "node")
                np.testing.assert_allclose(tile_pixels(g.doc, "node"), full, atol=1e-6)

    def test_a_document_saved_before_keeps_rgb(self):
        for kind in ALL_KINDS:
            with self.subTest(kind=kind):
                g = Graph()
                g.add("node", kind, {})
                doc = copy.deepcopy(g.doc)
                doc["nodes"]["node"]["params"].pop("channels", None)
                doc = upgrade_document(doc)
                self.assertEqual(doc["nodes"]["node"]["params"]["channels"], "rgb")

    def test_a_saved_all_stays_all(self):
        g = Graph()
        g.add("node", "Multiply", {"channels": "all"})
        doc = upgrade_document(copy.deepcopy(g.doc))
        self.assertEqual(doc["nodes"]["node"]["params"]["channels"], "all")


class LabelsTests(unittest.TestCase):
    def test_value_nodes_lead_with_channels_and_call_the_number_value(self):
        for kind in ("Multiply", "Add", "Gamma"):
            with self.subTest(kind=kind):
                names = labels(kind)
                self.assertEqual(names[:2], ["Channels", "Value"])

    def test_copy_rows_read_as_nuke_and_say_their_destination(self):
        names = labels("Copy")
        self.assertEqual(names[:4], ["Copy channel"] * 4)
        helps = [g.help for g in knobs.knob_layout("Copy")[:4]]
        for word, text in zip(("red", "green", "blue", "alpha"), helps):
            self.assertIn(word, text)

    def test_clamp_labels_are_the_guides(self):
        names = labels("Clamp")
        self.assertIn("MinClampTo", names)
        self.assertIn("MaxClampTo", names)

    def test_keylight_crop_and_gain_labels(self):
        names = labels("ScreenKeyer")
        for word in ("Left", "Right", "Bottom", "Top", "X Method", "Y Method", "Highlights Gain", "Midtones Gain", "Screen Dilate"):
            self.assertIn(word, names)

    def test_hsv_tool_labels_and_headings(self):
        names = labels("HSVTool")
        self.assertEqual(names.count("Adjustment"), 2)
        self.assertEqual(names.count("Range Rolloff"), 3)
        self.assertEqual(names.count("Force Value"), 2)
        headings = [g.section for g in knobs.knob_layout("HSVTool") if g.section]
        for word in ("Hue", "Saturation", "Brightness"):
            self.assertIn(word, headings)

    def test_remove_and_colormatrix_follow_the_guide(self):
        self.assertEqual(labels("Remove")[:2], ["Operation", "Channels"])
        self.assertEqual(labels("ColorMatrix")[0], "Channels")
        self.assertEqual(labels("Expression").count("Channels"), 4)


class IBKGizmoTests(unittest.TestCase):
    def test_new_node_starts_on_the_guides_values(self):
        p = SPECS["IBKGizmo"]["params"]
        self.assertEqual(p["screen_type"], "blue")
        self.assertEqual(p["luminance_level"], 0.0)

    def test_older_document_keeps_green_and_its_level(self):
        g = Graph()
        g.add("node", "IBKGizmo", {})
        doc = copy.deepcopy(g.doc)
        for name in ("screen_type", "luminance_level", "luminance_level_enable"):
            doc["nodes"]["node"]["params"].pop(name, None)
        params = upgrade_document(doc)["nodes"]["node"]["params"]
        self.assertEqual(params["screen_type"], "green")
        self.assertEqual(params["luminance_level"], 1.0)
        self.assertEqual(params["luminance_level_enable"], 1)


class ChromaKeyerDespillBiasTests(unittest.TestCase):
    def _screen(self):
        out = np.zeros((12, 16, 4), np.float32)
        out[..., 0] = 0.3
        out[..., 1] = 0.7
        out[..., 2] = 0.1
        out[..., 3] = 1.0
        return out

    def test_the_bias_is_a_colour_that_acts(self):
        p = SPECS["ChromaKeyer"]["params"]
        for name in ("despill_bias_red", "despill_bias_green", "despill_bias_blue"):
            self.assertEqual(p[name], 0.5)
        src = self._screen()
        base = kernel("ChromaKeyer", src, override_despill_bias=1, premultiply=0)
        # the screen is green: the bias weighs red against blue (the two other channels, in that order)
        toward_blue = kernel("ChromaKeyer", src, override_despill_bias=1, premultiply=0, despill_bias_red=0.0, despill_bias_blue=1.0)
        toward_red = kernel("ChromaKeyer", src, override_despill_bias=1, premultiply=0, despill_bias_red=1.0, despill_bias_blue=0.0)
        self.assertFalse(np.allclose(toward_blue[..., 1], toward_red[..., 1]))
        # a limit weighted by the larger blue share is lower here (blue 0.1 against red 0.3), so more green is taken away
        self.assertLess(float(toward_blue[0, 0, 1]), float(toward_red[0, 0, 1]))
        np.testing.assert_allclose(base[..., 1], kernel("ChromaKeyer", src, override_despill_bias=1, premultiply=0, despill_bias_red=0.5,
                                                        despill_bias_blue=0.5)[..., 1])

    def test_a_node_saved_before_keeps_the_single_number(self):
        g = Graph()
        g.add("node", "ChromaKeyer", {"override_despill_bias": 1, "despill_bias": 0.9})
        doc = copy.deepcopy(g.doc)
        for name in ("despill_bias_red", "despill_bias_green", "despill_bias_blue", "legacy_despill_bias"):
            doc["nodes"]["node"]["params"].pop(name, None)
        params = upgrade_document(doc)["nodes"]["node"]["params"]
        self.assertEqual(params["legacy_despill_bias"], 1)
        full = {**SPECS["ChromaKeyer"]["params"], **params, "premultiply": 0}
        src = self._screen()
        out = Evaluator._kernel("ChromaKeyer", full, [src])
        scalar = Evaluator._despill(src[..., :3], np.array([0.1, 0.8, 0.2], np.float32), 0.9)
        np.testing.assert_allclose(out[..., :3], scalar, atol=1e-6)

    def test_tiles_agree(self):
        g = Graph()
        g.add("src", "Constant", {"red": 0.3, "green": 0.7, "blue": 0.1, "alpha": 1.0})
        g.add("node", "ChromaKeyer", {"override_despill_bias": 1, "despill_bias_red": 0.2, "despill_bias_blue": 0.9}, image="src")
        np.testing.assert_allclose(tile_pixels(g.doc, "node"), evaluator_pixels(g.doc, "node"), atol=1e-6)


def two_motions(height=96, width=128):
    """Two frames of a plate whose left half moves 3 pixels right and whose right half moves 3 pixels left."""
    rng = np.random.RandomState(11)
    texture = rng.rand(height, width + 16).astype(np.float32)
    kernel_1d = np.array([1.0, 2.0, 1.0], np.float32) / 4.0
    for _ in range(3):                                   # a few binomial passes along each axis: a soft texture the estimator can follow
        texture = np.apply_along_axis(lambda line: np.convolve(np.pad(line, 1, mode="edge"), kernel_1d, mode="valid"), 1, texture)
        texture = np.apply_along_axis(lambda line: np.convolve(np.pad(line, 1, mode="edge"), kernel_1d, mode="valid"), 0, texture)
    texture = (texture - texture.min()) / (texture.max() - texture.min())
    first = np.zeros((height, width), np.float32)
    second = np.zeros((height, width), np.float32)
    half = width // 2
    first[:, :half] = texture[:, 8:8 + half]
    second[:, :half] = texture[:, 8 - 3:8 - 3 + half]
    first[:, half:] = texture[:, 8 + half:8 + width]
    second[:, half:] = texture[:, 8 + half + 3:8 + width + 3]
    return np.repeat(first[..., None], 4, axis=2), np.repeat(second[..., None], 4, axis=2)


class VectorDetailTests(unittest.TestCase):
    """Vector Detail is the density of the vectors relative to the image (Kronos 0.3, OFlow 0.2 on a new node)."""

    def _params(self, kind, **over):
        return {**SPECS[kind]["params"], "flow_backend": "cpu", **over}

    def _flow(self, kind, **over):
        from nodebased import retimeknobs
        a, b = two_motions()
        return retimeknobs.estimate_flows(a, b, self._params(kind, **over), "cpu")[0]

    def test_the_guides_defaults(self):
        self.assertEqual(SPECS["Kronos"]["params"]["vector_density"], 0.3)
        self.assertEqual(SPECS["OFlow"]["params"]["vector_density"], 0.2)
        self.assertEqual(SPECS["Kronos"]["params"]["legacy_vector_density"], 0)

    def test_a_lower_value_gives_a_smoother_field_on_two_motions(self):
        def roughness(flow):
            # how much the horizontal vector changes from pixel to pixel inside the two halves, which each move as one
            left, right = flow[10:86, 6:54, 0], flow[10:86, 74:122, 0]
            return float(np.abs(np.diff(left, axis=1)).mean() + np.abs(np.diff(right, axis=1)).mean())
        fields = {density: self._flow("Kronos", vector_density=density) for density in (1.0, 0.5, 0.25)}
        self.assertLess(roughness(fields[0.5]), roughness(fields[1.0]))
        self.assertLess(roughness(fields[0.25]), roughness(fields[0.5]))
        # each still finds both motions: the left half moves right by 3, the right half left by 3
        for flow in fields.values():
            self.assertAlmostEqual(float(flow[10:86, 6:54, 0].mean()), 3.0, delta=0.3)
            self.assertAlmostEqual(float(flow[10:86, 74:122, 0].mean()), -3.0, delta=0.3)

    def test_density_one_is_the_full_size_estimate(self):
        from nodebased import retimeknobs
        a, b = two_motions()
        p = self._params("Kronos", vector_density=1.0)
        old = self._params("Kronos", vector_density=0.2, legacy_vector_density=1)
        np.testing.assert_array_equal(retimeknobs.estimate_flows(a, b, p, "cpu")[0], retimeknobs.estimate_flows(a, b, old, "cpu")[0])

    def test_a_saved_node_keeps_the_full_size_estimate(self):
        for kind in ("Kronos", "OFlow"):
            with self.subTest(kind=kind):
                g = Graph()
                g.add("node", kind, {})
                doc = copy.deepcopy(g.doc)
                for name in ("vector_density", "legacy_vector_density"):
                    doc["nodes"]["node"]["params"].pop(name, None)
                params = upgrade_document(doc)["nodes"]["node"]["params"]
                self.assertEqual(params["legacy_vector_density"], 1)
                self.assertEqual(params["vector_density"], 1.0)


class SetBboxTests(unittest.TestCase):
    """Copy and ChannelMerge's Set bbox to: the data window of the result is the union of the two inputs', their intersection, A's or B's."""

    def _rasters(self):
        from nodebased.raster import Raster
        from nodebased.tiers import Region
        display = Region(0, 0, 16, 12)
        a_data, b_data = Region(0, 0, 8, 12), Region(4, 0, 12, 12)
        a = Raster(np.full((12, 8, 4), 0.25, np.float32), a_data, display)
        b = Raster(np.full((12, 12, 4), 0.75, np.float32), b_data, display)
        return a, b, {"union": Region(0, 0, 16, 12), "intersection": Region(4, 0, 4, 12), "A": a_data, "B": b_data}

    def test_the_window_follows_the_choice(self):
        a, b, windows = self._rasters()
        for kind in ("Copy", "ChannelMerge"):
            for choice, region in windows.items():
                with self.subTest(kind=kind, bbox=choice):
                    params = dict(SPECS[kind]["params"], merge_bbox=choice)
                    out = Evaluator._windowed_kernel(kind, params, [a, b, None])
                    self.assertEqual(out.data, region)

    def test_union_is_the_default_and_an_older_node_is_the_union(self):
        for kind in ("Copy", "ChannelMerge"):
            with self.subTest(kind=kind):
                self.assertEqual(SPECS[kind]["params"]["merge_bbox"], "union")
                g = Graph()
                g.add("node", kind, {})
                doc = copy.deepcopy(g.doc)
                doc["nodes"]["node"]["params"].pop("merge_bbox", None)
                self.assertEqual(upgrade_document(doc)["nodes"]["node"]["params"]["merge_bbox"], "union")

    def test_a_window_other_than_the_union_leaves_the_tile_path(self):
        from nodebased import ops2d_chrome
        for kind in ("Copy", "ChannelMerge"):
            self.assertFalse(ops2d_chrome.needs_full_frame(kind, {"merge_bbox": "union"}))
            self.assertTrue(ops2d_chrome.needs_full_frame(kind, {"merge_bbox": "intersection"}))

    def test_the_panel_has_the_knob_in_the_guides_place(self):
        self.assertEqual(labels("Copy").index("Set bbox to") - labels("Copy").index("Layer copy"), 1)
        self.assertIn("Set bbox to", labels("ChannelMerge"))


if __name__ == "__main__":
    unittest.main()
