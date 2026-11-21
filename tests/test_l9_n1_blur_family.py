"""Lane 9, step N1 (Fri 11/20): Blur, Sharpen, Soften and Laplacian filter the way Nuke 17.0v6 renders them.

The numbers below are read from real Nuke renders of an impulse (the EXRs are not in git): the blur's weights for each filter and size, Soften and
Sharpen's `minimum` / `maximum`, and Laplacian's sign. A document saved before the change keeps its pixels through `legacy_kernel` 1, which the
version 40 migration sets on saved nodes of those four kinds.
"""
import copy
import unittest

import numpy as np

from nodebased import blurknobs
from nodebased.core import SCHEMA_VERSION, SPECS, Dispatcher, upgrade_document, validate
from nodebased.imaging import Evaluator

from tests.test_2d_behaviour_filter_a import Graph as _Graph, evaluator_pixels, texture, tile_pixels


class Graph(_Graph):
    legacy_default = False


def kern(kind, pixels, **knobs):
    """One kernel on the evaluator's own path from the node's SPECS defaults (a node made now: `legacy_kernel` 0) plus `knobs`."""
    spec = SPECS[kind]
    slots = [pixels] + [None] * (len(spec["inputs"]) + len(spec.get("optional_inputs", [])) - 1)
    return np.asarray(Evaluator._kernel(kind, {**spec["params"], **knobs}, slots), np.float64)


def source_graph(kind, params, masked):
    g = Graph()
    g.add("plate", "Checker", dict(width=45, height=37, size=5))
    g.add("shade", "Grade", dict(exposure=-0.5), image="plate")
    inputs = {"image": "shade"}
    if masked:
        g.add("matte", "Constant", dict(width=45, height=37, red=1, green=1, blue=1, alpha=0.5))
        inputs["mask"] = "matte"
    g.add("node", kind, params, **inputs)
    return g


def impulse(h=25, w=41):
    img = np.zeros((h, w, 4), np.float32)
    img[h // 2, w // 2, :3] = 1.0
    img[..., 3] = 1.0
    return img


def row(out, n):
    """The centre row of a blurred impulse, `n` pixels each way from the centre, red channel."""
    y, x = out.shape[0] // 2, out.shape[1] // 2
    return out[y, x - n:x + n + 1, 0].astype(np.float64)


# One row of Nuke's 2D response to an impulse is the 1-D weight at each offset times the weight at the centre.
NUKE_IMPULSE_ROWS = {
    # (filter, size): centre row, offsets -R..R, from Nuke 17.0v6
    ("box", 3.0): [0.11111, 0.11111, 0.11111],
    ("box", 4.0): [0.03125, 0.0625, 0.0625, 0.0625, 0.03125],
    ("triangle", 3.0): [0.03704, 0.07407, 0.11111, 0.07407, 0.03704],
    ("triangle", 4.0): [0.01562, 0.03125, 0.04688, 0.0625, 0.04688, 0.03125, 0.01562],
    ("quadratic", 3.0): [0.00116, 0.01042, 0.02894, 0.05324, 0.0625, 0.05324, 0.02894, 0.01042, 0.00116],
    ("gaussian", 3.0): [0.00071, 0.00613, 0.02861, 0.0721, 0.09812, 0.0721, 0.02861, 0.00613, 0.00071],
    ("gaussian", 4.0): [0.00011, 0.00072, 0.00345, 0.0116, 0.02759, 0.04639, 0.05517, 0.04639, 0.02759, 0.0116, 0.00345, 0.00072, 0.00011],
    ("box", 2.5): [0.12, 0.16, 0.12],               # 0.3 : 0.4 : 0.3 each way
}


class WeightsAreNukes(unittest.TestCase):
    def test_impulse_rows_equal_nukes_renders(self):
        for (shape, size), expected in NUKE_IMPULSE_ROWS.items():
            n = len(expected) // 2
            with self.subTest(shape=shape, size=size):
                out = kern("Blur", impulse(), radius=size, kernel_filter=shape, quality=15.0)
                np.testing.assert_allclose(row(out, n), expected, atol=2e-5)
                # nothing further out
                self.assertAlmostEqual(float(out[12, 20 + n + 1, 0]), 0.0, places=7)

    def test_a_box_of_size_one_or_less_is_the_picture_and_so_is_size_zero(self):
        pixels = texture()
        for size in (0.0, 0.5, 1.0):
            np.testing.assert_array_equal(kern("Blur", pixels, radius=size, kernel_filter="box"), pixels)

    def test_edge_pixels_repeat_past_the_border(self):
        pixels = np.zeros((6, 7, 4), np.float32)
        pixels[..., 0] = np.arange(42).reshape(6, 7)
        out = kern("Blur", pixels, radius=3.0, kernel_filter="box")
        self.assertAlmostEqual(float(out[0, 0, 0]), 2.6666667, places=5)       # Nuke's 3 x 3 average of this ramp at the corner
        self.assertAlmostEqual(float(out[5, 0, 0]), 33.0, places=5)

    def test_support_is_the_reach_the_tiles_hold(self):
        self.assertEqual([blurknobs.support("box", s) for s in (1.0, 1.5, 2.0, 3.0, 4.0)], [0, 1, 1, 1, 2])
        self.assertEqual([blurknobs.support("triangle", s) for s in (1.0, 2.5, 3.0, 4.0)], [0, 2, 2, 3])
        self.assertEqual([blurknobs.support("quadratic", s) for s in (0.5, 1.0, 3.0, 4.0)], [0, 1, 4, 5])
        self.assertEqual([blurknobs.support("gaussian", s) for s in (0.5, 1.0, 2.5, 3.0, 4.0, 10.0)], [0, 1, 3, 4, 6, 15])
        for shape in blurknobs.FILTERS:
            for size in (1.5, 2.5, 3.0, 7.3):
                w = blurknobs.nuke_weights(shape, size)
                self.assertEqual(len(w), 2 * blurknobs.support(shape, size) + 1)
                self.assertAlmostEqual(float(w.sum()), 1.0, places=12)


class SoftenSharpenLimits(unittest.TestCase):
    """`minimum` and `maximum` clamp the change: sign(d) * min(maximum, max(|d| - minimum, 0)), as Nuke renders it."""

    @staticmethod
    def pulses(heights):
        img = np.zeros((9, 5 * len(heights), 4), np.float32)
        img[..., 3] = 1.0
        for i, h in enumerate(heights):
            img[:, i * 5 + 2, 0] = h
        return img

    def check(self, kind, params, expected):
        heights = [0.3, 1.0, 2.0, 6.0]
        out = kern(kind, self.pulses(heights), kernel_filter="box", channels="rgba", **params)
        got = [(float(out[4, i * 5 + 2, 0]), float(out[4, i * 5 + 3, 0])) for i in range(4)]
        for (c, s), (ec, es) in zip(got, expected):
            self.assertAlmostEqual(c, ec, places=3)
            self.assertAlmostEqual(s, es, places=3)

    def test_soften_values_from_nuke(self):
        base = dict(soften_size=3.0, minimum=0.0, maximum=1.0)
        self.check("Soften", dict(base, amount=1.0), [(0.1, 0.1), (0.3333, 0.3333), (1.0, 0.6667), (5.0, 1.0)])
        self.check("Soften", dict(base, amount=2.0), [(-0.1, 0.2), (-0.3333, 0.6667), (0.0, 1.3333), (4.0, 2.0)])
        self.check("Soften", dict(base, amount=0.5), [(0.2, 0.05), (0.6667, 0.1667), (1.5, 0.3333), (5.5, 0.5)])
        self.check("Soften", dict(base, amount=0.5, minimum=0.2), [(0.3, 0.0), (0.7667, 0.0667), (1.5, 0.2333), (5.5, 0.5)])

    def test_sharpen_values_from_nuke(self):
        base = dict(sharpen_size=3.0, minimum=0.0, maximum=1.0)
        self.check("Sharpen", dict(base, sharpen_amount=1.0), [(0.5, -0.1), (1.6667, -0.3333), (3.0, -0.6667), (7.0, -1.0)])
        self.check("Sharpen", dict(base, sharpen_amount=2.0), [(0.7, -0.2), (2.3333, -0.6667), (4.0, -1.3333), (8.0, -2.0)])
        self.check("Sharpen", dict(base, sharpen_amount=1.0, minimum=0.5), [(0.3, 0.0), (1.1667, 0.0), (2.8333, -0.1667), (7.0, -1.0)])

    def test_the_effect_is_not_gated_by_the_pixel_value(self):
        pixels = texture()
        out = kern("Soften", pixels, soften_size=3.0, kernel_filter="gaussian", amount=1.0, minimum=0.0, maximum=1000.0, channels="rgba")
        blurred = blurknobs.filtered(pixels, 3.0, "gaussian", 15.0, nuke=True)
        np.testing.assert_allclose(out, blurred, atol=2e-5)


class LaplacianIsPictureMinusBlur(unittest.TestCase):
    def test_an_impulse_comes_out_positive_at_its_centre(self):
        out = kern("Laplacian", impulse(), laplacian_size=3.0, kernel_filter="box", channels="rgba")
        np.testing.assert_allclose(row(out, 1), [-0.11111, 0.88889, -0.11111], atol=2e-5)

    def test_gaussian_size_three_matches_nukes_centre_row(self):
        out = kern("Laplacian", impulse(), laplacian_size=3.0, kernel_filter="gaussian", channels="rgba")
        np.testing.assert_allclose(row(out, 4), [-0.00071, -0.00613, -0.02861, -0.0721, 0.90188, -0.0721, -0.02861, -0.00613, -0.00071], atol=2e-5)

    def test_flat_picture_gives_zero_and_an_identity_blur_passes_the_picture_through(self):
        flat = np.full((9, 21, 4), 0.5, np.float32)
        np.testing.assert_allclose(kern("Laplacian", flat, laplacian_size=3.0, kernel_filter="box", channels="rgba"), 0.0, atol=1e-6)
        # Nuke: where the blur is the identity (box of size 1) the node returns the picture, not zero
        np.testing.assert_array_equal(kern("Laplacian", flat, laplacian_size=1.0, kernel_filter="box", channels="rgba"), flat)


class TilesEqualTheWholeFrame(unittest.TestCase):
    CHANGED = {
        "Blur box": dict(radius=4.0, kernel_filter="box", quality=15.0),
        "Blur box fraction": dict(radius=2.5, kernel_filter="box", quality=15.0),
        "Blur triangle": dict(radius=5.0, kernel_filter="triangle", quality=15.0),
        "Blur quadratic": dict(radius=3.0, kernel_filter="quadratic", quality=15.0),
        "Blur gaussian": dict(radius=4.0, kernel_filter="gaussian", quality=15.0),
        "Blur scaled": dict(radius=9.0, kernel_filter="gaussian", quality=2.0),
        "Sharpen": dict(sharpen_size=3.0, kernel_filter="box", channels="rgba", minimum=0.0, maximum=1.0),
        "Soften": dict(soften_size=4.0, kernel_filter="gaussian", channels="rgba", amount=0.7, minimum=0.1, maximum=0.5),
        "Laplacian": dict(laplacian_size=3.0, kernel_filter="box", channels="rgba"),
        "Laplacian gaussian": dict(laplacian_size=3.0, kernel_filter="gaussian", channels="rgba"),
    }

    def test_tiles_of_7_16_and_32_equal_the_frame_with_and_without_a_mask(self):
        for label, params in self.CHANGED.items():
            kind = label.split()[0]
            for masked in (False, True):
                g = source_graph(kind, params, masked)
                self.assertEqual(g.doc["nodes"]["node"]["params"]["legacy_kernel"], 0)
                whole = evaluator_pixels(g.doc, "node")
                for edge in (7, 16, 32):
                    with self.subTest(kind=label, masked=masked, tile=edge):
                        np.testing.assert_allclose(tile_pixels(g.doc, "node", edge), whole, atol=1e-5)


def box_mean_radius(a, r):
    a = a.astype(np.float64)
    p = np.pad(a, ((r, r), (r, r), (0, 0)), mode="edge")
    h, w = a.shape[:2]
    return sum(p[dy:dy + h, dx:dx + w] for dy in range(2 * r + 1) for dx in range(2 * r + 1)) / float((2 * r + 1) ** 2)


class OlderDocumentsKeepTheirPixels(unittest.TestCase):
    @staticmethod
    def saved_before(kind, params, drop):
        g = Graph()
        g.add("src", "Checker", dict(width=45, height=37, size=5))
        g.add("n", kind, params, image="src")
        old = copy.deepcopy(g.doc)
        for name in drop:
            old["nodes"]["n"]["params"].pop(name, None)
        old["version"] = 38
        loaded = upgrade_document(old)
        validate(loaded)
        return loaded

    @staticmethod
    def render(document, target):
        return Evaluator().evaluate_raster(dict(document, view=target), target=target).pixels

    def test_schema_is_forty_and_the_flag_is_set_on_the_four_kinds(self):
        self.assertGreaterEqual(SCHEMA_VERSION, 40)
        for kind in ("Blur", "Sharpen", "Soften", "Laplacian"):
            self.assertEqual(SPECS[kind]["params"]["legacy_kernel"], 0, kind)
            loaded = self.saved_before(kind, {}, ("legacy_kernel",))
            self.assertEqual(loaded["nodes"]["n"]["params"]["legacy_kernel"], 1, kind)

    def test_an_older_blur_is_the_integer_radius_box_as_before(self):
        loaded = self.saved_before("Blur", dict(radius=2.0), ("legacy_kernel", "kernel_filter", "quality", "crop"))
        src = self.render(loaded, "src")
        np.testing.assert_allclose(self.render(loaded, "n"), box_mean_radius(src, 2), atol=1e-5)

    def test_an_older_blur_in_a_group_is_set_too(self):
        g = Graph()
        g.add("src", "Checker", dict(width=45, height=37, size=5))
        g.add("n", "Blur", dict(radius=2.0), image="src")
        doc = copy.deepcopy(g.doc)
        inner = copy.deepcopy(doc["nodes"]["n"])
        inner["params"].pop("legacy_kernel")
        doc["nodes"]["grp"] = {"type": "Group", "params": {}, "inputs": {}, "graph": {"nodes": {"inner": inner}}}
        doc["version"] = 38
        loaded = upgrade_document(doc)
        self.assertEqual(loaded["nodes"]["grp"]["graph"]["nodes"]["inner"]["params"]["legacy_kernel"], 1)

    def test_an_older_laplacian_keeps_blur_minus_picture_and_an_older_sharpen_its_value_range(self):
        loaded = self.saved_before("Laplacian", dict(channels="rgba", laplacian_size=2.0),
                                   ("legacy_kernel", "kernel_filter", "quality", "crop"))
        src = self.render(loaded, "src")
        # an older node: box of integer radius 2, blur minus picture
        np.testing.assert_allclose(self.render(loaded, "n"), box_mean_radius(src, 2) - src, atol=1e-5)
        loaded = self.saved_before("Sharpen", dict(sharpen_size=2.0, sharpen_amount=1.0, channels="rgba"),
                                   ("legacy_kernel", "kernel_filter", "quality", "crop", "minimum", "maximum"))
        params = loaded["nodes"]["n"]["params"]
        self.assertEqual((params["legacy_kernel"], params["kernel_filter"]), (1, "box"))
        src = self.render(loaded, "src")
        np.testing.assert_allclose(self.render(loaded, "n"), src + (src - box_mean_radius(src, 2)), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
