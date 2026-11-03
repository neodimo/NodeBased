"""2D behaviour, lane 9 step A: Blur, Erode, ErodeFilter, Median, Sharpen, Soften, Glow, Convolve, Matrix, Laplacian, EdgeDetect and
Emboss compute what the Nuke 17.0 Reference Guide says they compute.

Four things, as tools/nuke_fidelity_cases_filter.py and docs/parity/nodebased_to_nuke.json frame them:

* the behaviour cases pass where the guide (or a long-established form) fixes the pixels, and the unscored ones say why;
* every main-tab knob NodeBased pairs with a Nuke knob changes pixels, in the direction the guide states where it states one;
* the kernels this step changed give the same pixels in tiles of 7, 16 and 32 as on the whole frame, with and without a mask;
* a document saved before the change renders bit for bit as it did.
"""
import copy
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from nodebased import filterbehaviour
from nodebased.core import SPECS, Dispatcher, bypass_slot, empty_document, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.raster import Raster
from nodebased.tileexec import TileExecutor

import audit_nuke_fidelity as audit  # noqa: E402
import nuke_fidelity_cases as ruler  # noqa: E402

STEP_A_NODES = ("Blur", "Erode (fast)", "Erode (filter)", "Median", "Sharpen", "Soften", "Glow", "Convolve", "Matrix", "Laplacian",
                "EdgeDetect", "Emboss")


def texture(h=37, w=45, seed=7):
    """A deterministic textured picture with values below 0 and above 1, and a partial alpha."""
    rng = np.random.RandomState(seed)
    y, x = np.mgrid[0:h, 0:w]
    base = 0.5 + 0.4 * np.sin(x / 3.1) * np.cos(y / 2.3)
    pixels = np.stack([base, base[::-1], base[:, ::-1]], axis=-1) + rng.uniform(-0.3, 0.3, (h, w, 3))
    pixels[3, 4] = 6.0                                                      # a hot pixel
    alpha = 0.55 + 0.4 * np.cos(x / 5.0 + y / 4.0)
    return np.concatenate([pixels, alpha[..., None]], axis=-1).astype(np.float32)


def kern(kind, pixels, **knobs):
    """One kernel on the evaluator's own path, from the node's SPECS defaults plus `knobs`."""
    spec = SPECS[kind]
    slots = [pixels] + [None] * (len(spec["inputs"]) + len(spec.get("optional_inputs", [])) - 1)
    return np.asarray(Evaluator._kernel(kind, {**spec["params"], **knobs}, slots), np.float64)


def differs(a, b, tol=1e-6):
    return float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max()) > tol


def impulse(n=21, channel_value=1.0):
    pixels = np.zeros((n, n, 4), np.float32)
    pixels[n // 2, n // 2] = channel_value
    return pixels


class Graph:
    def __init__(self):
        self.d = Dispatcher()

    def add(self, key, kind, params=None, **inputs):
        self.d.execute(dict(op="create", id=key, type=kind, params=params or {}))
        for slot, source in inputs.items():
            self.d.execute(dict(op="connect", id=key, input=slot, source=source))
        return key

    @property
    def doc(self):
        return self.d.document


def evaluator_pixels(document, target):
    return Evaluator().evaluate(dict(document, view=target))


def tile_pixels(document, target, edge):
    executor = TileExecutor(evaluator=Evaluator(), tile_edge=edge)
    document = dict(document, view=target)
    assert executor.supports_tiled(document, target), target
    region = executor.canvas_region(document, target, frame=1, tier=1)
    return executor.compose_region(document, target, region, frame=1, tier=1).pixels


def source_graph(kind, params, masked, image=None):
    """A checker canvas (or `image`) feeding `kind`, with a half-strength constant mask on the mask socket when asked."""
    g = Graph()
    g.add("plate", "Checker", dict(width=45, height=37, size=5))
    g.add("shade", "Grade", dict(exposure=-0.5), image="plate")
    inputs = {"image": "shade"}
    if masked:
        g.add("matte", "Constant", dict(width=45, height=37, red=1, green=1, blue=1, alpha=0.5))
        inputs["mask"] = "matte"
    g.add("node", kind, params, **inputs)
    return g


# ------------------------------------------------------------------------------------------------------------- the cases
class CaseTests(unittest.TestCase):
    def test_every_scored_case_of_the_step_passes(self):
        scored = [c for c in ruler.CASES if c.nuke in STEP_A_NODES and not c.unscored]
        self.assertGreaterEqual(len(scored), 20)
        for case in scored:
            with self.subTest(case=case.id):
                result = audit.run_case(case)
                self.assertEqual(result["status"], "pass", (case.id, result["max_abs_error"], result["worst"], case.statement))

    def test_every_unscored_case_states_its_reason(self):
        unscored = [c for c in ruler.CASES if c.nuke in STEP_A_NODES and c.unscored]
        self.assertGreaterEqual(len(unscored), 10)
        for case in unscored:
            with self.subTest(case=case.id):
                self.assertGreater(len(case.unscored), 40, case.id)
                self.assertEqual(audit.run_case(case)["status"], "not scored")

    def test_case_ids_are_unique_and_sourced(self):
        ids = [c.id for c in ruler.CASES]
        self.assertEqual(len(ids), len(set(ids)))
        for case in ruler.CASES:
            self.assertIn(case.source, ("guide", "convention"), case.id)
            if case.source == "convention":
                self.assertTrue(case.statement, case.id)

    def test_every_node_of_the_step_has_a_case(self):
        for node in STEP_A_NODES:
            with self.subTest(node=node):
                self.assertTrue(ruler.cases_for(node), node)


def old_document(kind, params, drop, source="Checker"):
    """A document saved before the step: `kind` fed by a picture, with the knobs in `drop` (and the legacy flags) not yet in it."""
    g = Graph()
    g.add("src", source, dict(width=45, height=37, size=5) if source == "Checker" else {})
    g.add("n", kind, params, image="src")
    old = copy.deepcopy(g.doc)
    for name in drop:
        old["nodes"]["n"]["params"].pop(name, None)
    loaded = upgrade_document(old)
    validate(loaded)
    return loaded


def render(document, target):
    return Evaluator().evaluate_raster(dict(document, view=target), target=target).pixels


# ----------------------------------------------------------------------------------------------------------------- Blur
class BlurTests(unittest.TestCase):
    FILTERS = ("box", "triangle", "quadratic", "gaussian")

    def test_size_zero_is_the_identity_for_every_filter(self):
        pixels = texture()
        for shape in self.FILTERS:
            with self.subTest(filter=shape):
                np.testing.assert_array_equal(kern("Blur", pixels, radius=0.0, kernel_filter=shape), pixels)

    def test_box_of_size_n_averages_the_pixels_within_n(self):
        for n in (1, 2, 3):
            with self.subTest(size=n):
                out = kern("Blur", impulse(21), radius=float(n), kernel_filter="box", quality=1000.0)[..., 0]
                mid = 10
                block = out[mid - n:mid + n + 1, mid - n:mid + n + 1]
                np.testing.assert_allclose(block, 1.0 / (2 * n + 1) ** 2, atol=1e-7)
                self.assertEqual(int((out > 1e-9).sum()), (2 * n + 1) ** 2)

    def test_every_filter_is_normalised_symmetric_and_separable(self):
        for shape in self.FILTERS:
            for size in (1.0, 2.5, 4.0):
                with self.subTest(filter=shape, size=size):
                    out = kern("Blur", impulse(31), radius=size, kernel_filter=shape, quality=1000.0)[..., 0]
                    self.assertAlmostEqual(float(out.sum()), 1.0, places=5)
                    np.testing.assert_allclose(out, out[::-1, ::-1], atol=1e-7)
                    np.testing.assert_allclose(out, out.T, atol=1e-7)
                    singular = np.linalg.svd(out.astype(np.float64), compute_uv=False)
                    self.assertLess(singular[1] / singular[0], 1e-5)           # rank one: the product of two 1-D kernels

    def test_a_flat_picture_stays_flat_at_any_size_and_quality(self):
        flat = np.full((25, 25, 4), 0.37, np.float32)
        for shape in self.FILTERS:
            for quality in (2.0, 15.0):
                with self.subTest(filter=shape, quality=quality):
                    np.testing.assert_allclose(kern("Blur", flat, radius=9.0, kernel_filter=shape, quality=quality), flat, atol=1e-6)

    def test_the_blur_widens_with_size(self):
        for shape in self.FILTERS:
            with self.subTest(filter=shape):
                peaks = [float(kern("Blur", impulse(41), radius=s, kernel_filter=shape, quality=1000.0)[20, 20, 0]) for s in (1.0, 2.0, 4.0, 8.0)]
                self.assertEqual(peaks, sorted(peaks, reverse=True))

    def test_a_filter_that_fits_within_quality_is_not_scaled(self):
        pixels = texture()
        for shape in self.FILTERS:
            with self.subTest(filter=shape):
                np.testing.assert_array_equal(kern("Blur", pixels, radius=5.0, kernel_filter=shape, quality=15.0),
                                              kern("Blur", pixels, radius=5.0, kernel_filter=shape, quality=1000.0))

    def test_a_size_above_quality_scales_down_and_stays_smooth_and_bounded(self):
        pixels = texture()
        direct = kern("Blur", pixels, radius=8.0, kernel_filter="box", quality=1000.0)
        scaled = kern("Blur", pixels, radius=8.0, kernel_filter="box", quality=2.0)
        self.assertTrue(differs(direct, scaled, 1e-4))
        self.assertGreaterEqual(float(scaled.min()), float(pixels.min()) - 1e-5)
        self.assertLessEqual(float(scaled.max()), float(pixels.max()) + 1e-5)
        # smoother than the source: the average step between neighbours falls
        self.assertLess(np.abs(np.diff(scaled[..., 0], axis=1)).mean(), np.abs(np.diff(pixels[..., 0], axis=1)).mean())

    def test_crop_on_keeps_the_window_off_grows_it_by_the_reach(self):
        raster = Raster.of(texture(12, 16))
        on = Evaluator._windowed_kernel("Blur", {**SPECS["Blur"]["params"], "radius": 3.0, "crop": 1}, [raster])
        off = Evaluator._windowed_kernel("Blur", {**SPECS["Blur"]["params"], "radius": 3.0, "crop": 0}, [raster])
        self.assertEqual((on.pixels.shape[1], on.pixels.shape[0]), (16, 12))
        self.assertEqual((off.pixels.shape[1], off.pixels.shape[0]), (16 + 6, 12 + 6))

    def test_channels_limit_the_blur_to_the_channels_named(self):
        for channels, blurred in (("rgb", (0, 1, 2)), ("alpha", (3,)), ("red", (0,))):
            with self.subTest(channels=channels):
                g = source_graph("Blur", dict(radius=3.0, kernel_filter="box", channels=channels), False)
                plain = evaluator_pixels(source_graph("Blur", dict(radius=3.0, kernel_filter="box"), False).doc, "node")
                src = evaluator_pixels(g.doc, "shade")
                out = evaluator_pixels(g.doc, "node")
                for c in range(4):
                    if c in blurred:
                        np.testing.assert_allclose(out[..., c], plain[..., c], atol=1e-6)
                    else:
                        np.testing.assert_array_equal(out[..., c], src[..., c])


# --------------------------------------------------------------------------------------------------------------- Erode
class ErodeFastTests(unittest.TestCase):
    def test_positive_size_grows_the_light_pixels_negative_the_dark_ones(self):
        dot = impulse(9)
        grown = kern("Erode", dot, erode_size=1.0)
        self.assertEqual(int((grown[..., 0] > 0.5).sum()), 9)                       # the white dot covers 3 x 3
        np.testing.assert_array_equal(kern("Erode", dot, erode_size=-1.0), np.zeros_like(dot))   # dark pixels grew over it
        hole = 1.0 - dot
        np.testing.assert_array_equal(kern("Erode", hole, erode_size=1.0), np.ones_like(hole))
        self.assertEqual(int((kern("Erode", hole, erode_size=-1.0)[..., 0] < 0.5).sum()), 9)

    def test_size_zero_is_the_identity_and_the_default(self):
        pixels = texture()
        self.assertEqual(SPECS["Erode"]["params"]["erode_size"], 0.0)
        np.testing.assert_array_equal(kern("Erode", pixels), pixels)

    def test_a_larger_size_reaches_further(self):
        counts = [int((kern("Erode", impulse(15), erode_size=s)[..., 0] > 0.5).sum()) for s in (1.0, 2.0, 3.0)]
        self.assertEqual(counts, [9, 25, 49])

    def test_channels_limit_it_to_the_channels_named(self):
        pixels = texture()
        out = kern("Erode", pixels, erode_size=2.0, channels="rgb")
        np.testing.assert_array_equal(out[..., 3], pixels[..., 3])
        self.assertTrue(differs(out[..., :3], pixels[..., :3]))

    def test_the_older_reading_keeps_positive_as_shrinking(self):
        dot = impulse(9)
        np.testing.assert_array_equal(kern("Erode", dot, erode_size=1.0, legacy_erode_sign=1), np.zeros_like(dot))
        pixels = texture()
        for size in (-2.0, 1.0, 3.0):
            np.testing.assert_array_equal(kern("Erode", pixels, erode_size=size, legacy_erode_sign=1),
                                          Evaluator._morph(pixels, size, "rgba"))

    def test_an_older_document_is_given_the_older_reading_and_draws_the_same(self):
        loaded = old_document("Erode", dict(erode_size=2.0, channels="rgba"), ("legacy_erode_sign",))
        self.assertEqual(loaded["nodes"]["n"]["params"]["legacy_erode_sign"], 1)
        source = render(loaded, "src")
        np.testing.assert_array_equal(render(loaded, "n"), Evaluator._morph(source, 2.0, "rgba"))


class ErodeFilterTests(unittest.TestCase):
    FILTERS = ("box", "triangle", "quadratic", "gaussian")

    def test_positive_size_lets_darker_areas_expand_negative_the_brighter(self):
        pixels = np.clip(texture(), 0.0, 1.0)
        for shape in self.FILTERS:
            with self.subTest(filter=shape):
                eroded = kern("ErodeFilter", pixels, filter_size=2.0, filter_type=shape)
                grown = kern("ErodeFilter", pixels, filter_size=-2.0, filter_type=shape)
                self.assertTrue(differs(eroded, pixels, 1e-3))
                self.assertTrue(differs(grown, pixels, 1e-3))
                if shape != "gaussian":                                              # the gaussian blurs after the min, so it can overshoot
                    self.assertLessEqual(float((eroded - pixels).max()), 1e-6)
                    self.assertGreaterEqual(float((grown - pixels).min()), -1e-6)
                self.assertLess(float(eroded.mean()), float(pixels.mean()))
                self.assertGreater(float(grown.mean()), float(pixels.mean()))

    def test_a_negative_size_is_the_mirror_of_the_positive_one(self):
        pixels = texture()
        for shape in self.FILTERS:
            with self.subTest(filter=shape):
                np.testing.assert_array_equal(kern("ErodeFilter", pixels, filter_size=-3.0, filter_type=shape),
                                              -kern("ErodeFilter", -pixels, filter_size=3.0, filter_type=shape))

    def test_size_below_half_a_pixel_is_the_identity(self):
        pixels = texture()
        for size in (0.0, 0.3, -0.3):
            np.testing.assert_array_equal(kern("ErodeFilter", pixels, filter_size=size), pixels)

    def test_the_filter_changes_the_pixels(self):
        pixels = texture()
        results = [kern("ErodeFilter", pixels, filter_size=2.3, filter_type=shape) for shape in self.FILTERS]
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                self.assertTrue(differs(results[i], results[j]), (self.FILTERS[i], self.FILTERS[j]))

    def test_the_default_size_is_one_and_erodes(self):
        self.assertEqual(SPECS["ErodeFilter"]["params"]["filter_size"], 1.0)
        dot = impulse(9)
        np.testing.assert_array_equal(kern("ErodeFilter", dot), np.zeros_like(dot))       # a one-pixel light dot is eroded away


# -------------------------------------------------------------------------------------------------------------- Median
class MedianTests(unittest.TestCase):
    def test_size_is_the_width_of_the_filter_area(self):
        rng = np.random.RandomState(3)
        pixels = rng.uniform(0, 1, (21, 21, 4)).astype(np.float32)
        for size, reach in ((3.0, 1), (5.0, 2), (7.0, 3)):
            with self.subTest(size=size):
                out = kern("Median", pixels, median_size=size)
                y, x = 10, 10
                window = pixels[y - reach:y + reach + 1, x - reach:x + reach + 1]
                np.testing.assert_allclose(out[y, x], np.median(window.reshape(-1, 4), axis=0), atol=1e-6)

    def test_the_default_is_size_three_the_three_by_three_area(self):
        self.assertEqual(SPECS["Median"]["params"]["median_size"], 3.0)

    def test_size_one_is_the_identity(self):
        pixels = texture()
        np.testing.assert_array_equal(kern("Median", pixels, median_size=1.0), pixels)

    def test_a_single_hot_pixel_is_removed(self):
        pixels = np.full((15, 15, 4), 0.2, np.float32)
        pixels[7, 7, :3] = 50.0
        out = kern("Median", pixels, median_size=3.0)
        np.testing.assert_allclose(out[..., :3], 0.2, atol=1e-6)

    def test_ignore_top_line_changes_the_pixels(self):
        pixels = texture()
        self.assertTrue(differs(kern("Median", pixels, median_size=3.0), kern("Median", pixels, median_size=3.0, ignore_top_line=1)))

    def test_the_older_reading_keeps_size_as_a_radius(self):
        pixels = texture()
        np.testing.assert_array_equal(kern("Median", pixels, median_size=1.0, legacy_median_radius=1),
                                      kern("Median", pixels, median_size=3.0, legacy_median_radius=0))
        self.assertEqual(filterbehaviour.median_support({"median_size": 2.0, "legacy_median_radius": 1}), 2)
        self.assertEqual(filterbehaviour.median_support({"median_size": 2.0, "legacy_median_radius": 0}), 1)
        self.assertEqual(filterbehaviour.median_support({"median_size": 3.0}), 3)         # a parameter dict with no flag is an older node

    def test_an_older_document_is_given_the_older_reading_and_draws_the_same(self):
        loaded = old_document("Median", dict(median_size=1.0, channels="rgba"), ("legacy_median_radius",))
        self.assertEqual(loaded["nodes"]["n"]["params"]["legacy_median_radius"], 1)
        fresh = copy.deepcopy(loaded)
        fresh["nodes"]["n"]["params"].update(median_size=3.0, legacy_median_radius=0)
        np.testing.assert_array_equal(render(loaded, "n"), render(fresh, "n"))


# -------------------------------------------------------------------------------------------------------------- Matrix
def matrix_weights(rows):
    flat = [float(v) for row in rows for v in row]
    return {f"weight{i}": flat[i] for i in range(len(flat))}


class MatrixTests(unittest.TestCase):
    EDGE = ((-1, -1, -1), (-1, 8, -1), (-1, -1, -1))

    def test_the_guides_edge_detection_example(self):
        pixels = np.clip(texture(), 0.0, 1.0)
        out = kern("Matrix", pixels, **matrix_weights(self.EDGE))
        centre = pixels[10, 10, :3] * 9 - pixels[9:12, 9:12, :3].sum(axis=(0, 1))
        np.testing.assert_allclose(out[10, 10, :3], np.clip(centre, 0, 1), atol=1e-5)

    def test_results_outside_zero_to_one_are_clipped(self):
        out = kern("Matrix", texture(), **matrix_weights(self.EDGE))
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_normalize_divides_the_matrix_by_its_sum(self):
        pixels = np.clip(texture(), 0.0, 1.0)
        ones = matrix_weights(((1, 1, 1), (1, 1, 1), (1, 1, 1)))
        out = kern("Matrix", pixels, normalize=1, **ones)
        np.testing.assert_allclose(out[10, 10, :3], pixels[9:12, 9:12, :3].mean(axis=(0, 1)), atol=1e-5)
        twos = matrix_weights(((2, 2, 2), (2, 2, 2), (2, 2, 2)))
        np.testing.assert_allclose(kern("Matrix", pixels, normalize=1, **twos), out, atol=1e-6)
        self.assertTrue(differs(kern("Matrix", pixels, normalize=0, **ones), out))

    def test_a_flat_picture_keeps_its_brightness_under_a_normalised_matrix(self):
        flat = np.full((15, 15, 4), 0.37, np.float32)
        out = kern("Matrix", flat, normalize=1, **matrix_weights(((1, 2, 1), (2, 4, 2), (1, 2, 1))))
        np.testing.assert_allclose(out, flat, atol=1e-6)

    def test_the_older_reading_does_not_clip(self):
        pixels = texture()
        out = kern("Matrix", pixels, legacy_matrix_unclipped=1, **matrix_weights(self.EDGE))
        self.assertTrue(float(out.min()) < 0.0 or float(out.max()) > 1.0)
        clipped = kern("Matrix", pixels, legacy_matrix_unclipped=0, **matrix_weights(self.EDGE))
        np.testing.assert_array_equal(clipped, np.clip(out, 0.0, 1.0).astype(np.float32))

    def test_the_identity_matrix_leaves_the_picture_alone(self):
        pixels = texture()
        np.testing.assert_array_equal(kern("Matrix", pixels), pixels)

    def test_an_older_document_is_given_the_older_reading_and_draws_the_same(self):
        params = dict(matrix_size="3", normalize=0, **matrix_weights(self.EDGE))
        loaded = old_document("Matrix", params, ("legacy_matrix_unclipped",))
        self.assertEqual(loaded["nodes"]["n"]["params"]["legacy_matrix_unclipped"], 1)
        source = render(loaded, "src")
        expected = Evaluator._matrix(source, {**params, "legacy_matrix_unclipped": 1})
        np.testing.assert_array_equal(render(loaded, "n"), expected)
        self.assertTrue(float(expected.min()) < 0.0 or float(expected.max()) > 1.0)


# ----------------------------------------------------------------------------------------------------------- Laplacian
class LaplacianTests(unittest.TestCase):
    def test_it_is_the_blur_minus_the_original(self):
        pixels = texture()
        for shape in ("box", "triangle", "quadratic", "gaussian"):
            with self.subTest(filter=shape):
                out = kern("Laplacian", pixels, laplacian_size=2.0, kernel_filter=shape, quality=1000.0)
                blurred = kern("Blur", pixels, radius=2.0, kernel_filter=shape, quality=1000.0)
                np.testing.assert_allclose(out, blurred - pixels, atol=1e-5)

    def test_a_flat_picture_has_no_laplacian(self):
        flat = np.full((15, 15, 4), 0.37, np.float32)
        np.testing.assert_allclose(kern("Laplacian", flat), 0.0, atol=1e-6)

    def test_a_light_dot_goes_negative_and_its_surround_positive(self):
        out = kern("Laplacian", impulse(21), laplacian_size=2.0, kernel_filter="box", quality=1000.0)[..., 0]
        self.assertLess(float(out[10, 10]), 0.0)                 # the blurred dot is darker than the original
        self.assertGreater(float(out[10, 11]), 0.0)

    def test_size_filter_quality_and_crop_act(self):
        pixels = texture()
        base = kern("Laplacian", pixels)
        self.assertTrue(differs(base, kern("Laplacian", pixels, laplacian_size=6.0)))
        self.assertTrue(differs(base, kern("Laplacian", pixels, kernel_filter="box")))
        self.assertTrue(differs(kern("Laplacian", pixels, laplacian_size=12.0), kern("Laplacian", pixels, laplacian_size=12.0, quality=2.0)))
        raster = Raster.of(texture(12, 16))
        on = Evaluator._windowed_kernel("Laplacian", {**SPECS["Laplacian"]["params"], "crop": 1}, [raster])
        off = Evaluator._windowed_kernel("Laplacian", {**SPECS["Laplacian"]["params"], "crop": 0}, [raster])
        self.assertEqual(on.pixels.shape[:2], (12, 16))
        self.assertEqual(off.pixels.shape[:2], (12 + 6, 16 + 6))

    def test_the_defaults_follow_nuke(self):
        params = SPECS["Laplacian"]["params"]
        self.assertEqual((params["laplacian_size"], params["kernel_filter"], params["quality"], params["crop"]), (3.0, "gaussian", 15.0, 1))

    def test_the_older_reading_is_the_fixed_cross_and_not_the_blur(self):
        pixels = texture()
        out = kern("Laplacian", pixels, channels="rgb", legacy_laplacian=1)
        weights = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float32)
        padded = np.pad(pixels, ((1, 1), (1, 1), (0, 0)), mode="edge")
        expected = pixels.copy()
        raw = sum(padded[y:y + 37, x:x + 45] * weights[y, x] for y in range(3) for x in range(3))
        expected[..., :3] = raw[..., :3]
        np.testing.assert_array_equal(out, expected)

    def test_an_older_document_is_given_the_older_reading_and_draws_the_same(self):
        loaded = old_document("Laplacian", dict(channels="rgb"),
                              ("legacy_laplacian", "laplacian_size", "kernel_filter", "quality", "crop"))
        params = loaded["nodes"]["n"]["params"]
        self.assertEqual(params["legacy_laplacian"], 1)
        source = render(loaded, "src")
        np.testing.assert_array_equal(render(loaded, "n"), Evaluator._matrix(source, {"channels": "rgb"}, laplacian=True))


# ------------------------------------------------------------------------------------------------------ Sharpen, Soften
WIDE = {"minimum": -1.0e6, "maximum": 1.0e6}


class SharpenSoftenTests(unittest.TestCase):
    """The guide says only that Sharpen and Soften 'use Laplacian' (blur minus the original); the facts that follow are tested, the
    exact scale of `amount` is not scored (see the cases module)."""

    def sharpen(self, pixels, **knobs):
        return kern("Sharpen", pixels, kernel_filter="box", quality=1000.0, **WIDE, **knobs)

    def soften(self, pixels, **knobs):
        return kern("Soften", pixels, kernel_filter="box", quality=1000.0, **WIDE, **knobs)

    def test_a_flat_picture_is_unchanged_by_both(self):
        flat = np.full((15, 15, 4), 0.37, np.float32)
        np.testing.assert_allclose(self.sharpen(flat, sharpen_size=3.0, sharpen_amount=2.0, channels="rgba"), flat, atol=1e-6)
        np.testing.assert_allclose(self.soften(flat, soften_size=3.0), flat, atol=1e-6)

    def test_amount_zero_changes_nothing(self):
        pixels = texture()
        np.testing.assert_array_equal(self.sharpen(pixels, sharpen_amount=0.0, sharpen_size=2.0), pixels)
        np.testing.assert_array_equal(self.soften(pixels, soften_size=2.0, amount=0.0), pixels)

    def test_sharpen_raises_contrast_and_soften_lowers_it(self):
        step = np.zeros((9, 16, 4), np.float32)
        step[:, 8:] = 0.5
        sharp = self.sharpen(step, sharpen_size=2.0, sharpen_amount=1.0, channels="rgba")
        soft = self.soften(step, soften_size=2.0, channels="rgba")
        self.assertLess(float(sharp[4, 7, 0]), 0.0)                       # the dark side undershoots
        self.assertGreater(float(sharp[4, 8, 0]), 0.5)                    # the light side overshoots
        self.assertGreater(float(soft[4, 7, 0]), 0.0)                     # the dark side is lifted
        self.assertLess(float(soft[4, 8, 0]), 0.5)
        pixels = texture()
        self.assertGreater(float(self.sharpen(pixels, sharpen_size=2.0, sharpen_amount=1.0, channels="rgb")[..., 0].std()), float(pixels[..., 0].std()))
        self.assertLess(float(self.soften(pixels, soften_size=2.0, channels="rgb")[..., 0].std()), float(pixels[..., 0].std()))

    def test_amount_scales_the_effect_linearly(self):
        pixels = texture()
        one = self.sharpen(pixels, sharpen_amount=1.0, sharpen_size=2.0) - pixels
        two = self.sharpen(pixels, sharpen_amount=2.0, sharpen_size=2.0) - pixels
        np.testing.assert_allclose(two, 2.0 * one, atol=1e-5)
        half = self.soften(pixels, soften_size=2.0, amount=0.5) - pixels
        full = self.soften(pixels, soften_size=2.0, amount=1.0) - pixels
        np.testing.assert_allclose(full, 2.0 * half, atol=1e-5)

    def test_sharpen_and_soften_are_opposite_directions_of_one_laplacian(self):
        pixels = texture()
        laplacian = kern("Laplacian", pixels, laplacian_size=2.0, kernel_filter="box", quality=1000.0)
        sharp = self.sharpen(pixels, sharpen_amount=1.0, sharpen_size=2.0, channels="rgba")
        soft = self.soften(pixels, soften_size=2.0, channels="rgba")
        np.testing.assert_allclose(sharp, pixels - laplacian, atol=1e-5)
        np.testing.assert_allclose(soft, pixels + laplacian, atol=1e-5)

    def test_the_value_range_confines_the_effect(self):
        pixels = texture()
        for kind, extra in (("Sharpen", dict(sharpen_amount=1.0, sharpen_size=2.0, channels="rgb")), ("Soften", dict(soften_size=2.0, channels="rgb"))):
            with self.subTest(kind=kind):
                out = kern(kind, pixels, kernel_filter="box", quality=1000.0, minimum=0.0, maximum=1.0, **extra)
                outside = (pixels[..., :3] < 0.0) | (pixels[..., :3] > 1.0)
                self.assertTrue(outside.any())
                np.testing.assert_array_equal(out[..., :3][outside], pixels[..., :3][outside])
                self.assertTrue(differs(out[..., :3][~outside], pixels[..., :3][~outside]))

    def test_the_defaults_are_the_guides_for_the_range_and_filter(self):
        for kind in ("Sharpen", "Soften"):
            params = SPECS[kind]["params"]
            self.assertEqual((params["minimum"], params["maximum"], params["kernel_filter"], params["quality"], params["crop"]),
                             (0.0, 1.0, "gaussian", 15.0, 1))


# ------------------------------------------------------------------------------------------------------------------ Glow
class GlowTests(unittest.TestCase):
    BASE = dict(glow_threshold=0.5, glow_size=4.0, kernel_filter="box", quality=1000.0, channels="rgb")

    def glow(self, pixels, **knobs):
        return kern("Glow", pixels, **{**self.BASE, **knobs})

    def effect(self, pixels, **knobs):
        return self.glow(pixels, effect_only=1, **knobs)

    def test_nothing_above_the_tolerance_means_no_glow(self):
        pixels = texture()
        np.testing.assert_array_equal(self.glow(pixels, glow_threshold=100.0), pixels)

    def test_the_tolerance_is_a_low_threshold_below_which_nothing_glows(self):
        dark = np.full((15, 15, 4), 0.4, np.float32)
        np.testing.assert_array_equal(self.glow(dark, glow_threshold=0.5), dark)
        bright = dark.copy()
        bright[7, 7, :3] = 3.0
        self.assertTrue(differs(self.glow(bright, glow_threshold=0.5), bright))
        loud = self.effect(texture(), glow_threshold=0.0)[..., :3].sum()
        quiet = self.effect(texture(), glow_threshold=1.0)[..., :3].sum()
        self.assertGreater(float(loud), float(quiet))

    def test_effect_only_passes_the_glow_alone(self):
        pixels = texture()
        full = self.glow(pixels)
        only = self.effect(pixels)
        np.testing.assert_allclose(only[..., :3], full[..., :3] - pixels[..., :3], atol=1e-5)
        np.testing.assert_array_equal(only[..., 3], pixels[..., 3])

    def test_brightness_scales_the_glow_and_zero_removes_it(self):
        pixels = texture()
        one = self.effect(pixels, brightness=1.0)[..., :3]
        np.testing.assert_allclose(self.effect(pixels, brightness=2.5)[..., :3], 2.5 * one, atol=1e-5)
        np.testing.assert_array_equal(self.glow(pixels, brightness=0.0), pixels)

    def test_tint_colours_the_glow_channel_by_channel(self):
        pixels = texture()
        white = self.effect(pixels)[..., :3]
        tinted = self.effect(pixels, red=0.5, green=0.0, blue=2.0)[..., :3]
        np.testing.assert_allclose(tinted, white * np.array([0.5, 0.0, 2.0], np.float32), atol=1e-5)

    def test_saturation_zero_gives_a_grey_glow(self):
        pixels = texture()
        grey = self.effect(pixels, saturation=0.0)[..., :3]
        np.testing.assert_allclose(grey[..., 0], grey[..., 1], atol=1e-5)
        np.testing.assert_allclose(grey[..., 1], grey[..., 2], atol=1e-5)
        colour = self.effect(pixels, saturation=1.0)[..., :3]
        self.assertGreater(float(np.abs(colour[..., 0] - colour[..., 1]).max()), 1e-3)

    def test_a_larger_size_spreads_the_glow_further(self):
        dot = np.zeros((41, 41, 4), np.float32)
        dot[20, 20, :3] = 5.0
        reach = lambda size: int((self.effect(dot, glow_size=size)[..., 0] > 1e-9).sum())
        self.assertLess(reach(2.0), reach(5.0))
        self.assertLess(reach(5.0), reach(9.0))

    def test_the_other_knobs_act(self):
        pixels = texture()
        base = self.effect(pixels)
        for name, value in (("kernel_filter", "gaussian"), ("nonlinear", 1), ("width_channel", "rgba.alpha"), ("quality", 2.0)):
            with self.subTest(knob=name):
                extra = {"glow_size": 12.0} if name == "quality" else {}
                reference = self.effect(pixels, **extra)
                self.assertTrue(differs(self.effect(pixels, **{name: value}, **extra), reference if extra else base), name)


# ------------------------------------------------------------------------------------------------------------- Convolve
def small_filter_raster(weights, channel=3, display=None, at=(0, 0)):
    """A filter image whose chosen channel holds `weights`, in a data window of its own (the other channels hold distractors)."""
    from nodebased.tiers import Region
    h, w = weights.shape
    pixels = np.full((h, w, 4), 0.123, np.float32)
    pixels[..., channel] = weights
    data = Region(at[0], at[1], w, h)
    return Raster(pixels, data, display or data)


class ConvolveTests(unittest.TestCase):
    def reference(self, image, weights):
        from nuke_fidelity_cases_filter import correlate
        return correlate(image, weights)

    def run_convolve(self, image, kernel, **knobs):
        return Evaluator._convolve(image, kernel.pixels, {**SPECS["Convolve"]["params"], **knobs}, kernel)

    def test_each_pixel_is_the_sum_of_its_neighbours_times_the_filter(self):
        image = texture()
        weights = np.array([[1, 2, 0], [0, 4, 1], [3, 0, 2]], np.float64)
        out = self.run_convolve(image, small_filter_raster(weights), normalize=0)
        expected = self.reference(image, weights)
        np.testing.assert_allclose(out[2:-2, 2:-2], expected[2:-2, 2:-2], atol=1e-4)

    def test_normalize_divides_the_filter_by_its_sum(self):
        image = texture()
        weights = np.array([[1, 2, 1], [2, 8, 2], [1, 2, 1]], np.float64)
        on = self.run_convolve(image, small_filter_raster(weights), normalize=1)
        np.testing.assert_allclose(on[2:-2, 2:-2], self.reference(image, weights / weights.sum())[2:-2, 2:-2], atol=1e-4)
        scaled = self.run_convolve(image, small_filter_raster(weights * 5), normalize=1)
        np.testing.assert_allclose(scaled, on, atol=1e-5)
        off = self.run_convolve(image, small_filter_raster(weights), normalize=0)
        np.testing.assert_allclose(off[2:-2, 2:-2], float(weights.sum()) * on[2:-2, 2:-2], atol=1e-3)
        flat = np.full((15, 15, 4), 0.37, np.float32)
        np.testing.assert_allclose(self.run_convolve(flat, small_filter_raster(weights)), flat, atol=1e-5)

    def test_a_filter_that_sums_to_zero_works_when_not_normalised(self):
        image = texture()
        edge = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float64)
        out = self.run_convolve(image, small_filter_raster(edge), normalize=0)
        np.testing.assert_allclose(out[2:-2, 2:-2], self.reference(image, edge)[2:-2, 2:-2], atol=1e-4)
        self.assertTrue(np.isfinite(self.run_convolve(image, small_filter_raster(edge), normalize=1)).all())

    def test_the_filter_channel_is_the_alpha_by_default_and_can_be_changed(self):
        image = texture()
        weights = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.float64)
        self.assertEqual(SPECS["Convolve"]["params"]["filter_channel"], "rgba.alpha")
        by_alpha = self.run_convolve(image, small_filter_raster(weights, channel=3))
        by_red = self.run_convolve(image, small_filter_raster(weights, channel=0), filter_channel="rgba.red")
        np.testing.assert_allclose(by_alpha, by_red, atol=1e-6)
        distractor = self.run_convolve(image, small_filter_raster(weights, channel=0))      # alpha is the flat 0.123 box
        self.assertTrue(differs(distractor, by_alpha))

    def test_use_input_channels_filters_each_channel_with_its_own(self):
        from nodebased.tiers import Region
        image = texture()
        rng = np.random.RandomState(5)
        planes = rng.uniform(0.1, 1.0, (3, 3, 4)).astype(np.float32)
        kernel = Raster(planes, Region(0, 0, 3, 3), Region(0, 0, 3, 3))
        out = self.run_convolve(image, kernel, use_input_channels=1)
        for c in range(4):
            expected = self.reference(image[..., c:c + 1], (planes[..., c] / planes[..., c].sum()).astype(np.float64))
            np.testing.assert_allclose(out[2:-2, 2:-2, c], expected[2:-2, 2:-2, 0], atol=1e-4)

    def test_the_filter_is_centred_on_the_middle_of_its_box_at_any_size(self):
        image = texture()
        for n in (1, 5, 9, 15, 21):                   # 21 x 21 takes the FFT path
            with self.subTest(size=n):
                weights = np.zeros((n, n), np.float64)
                weights[n // 2, n // 2] = 1.0
                np.testing.assert_allclose(self.run_convolve(image, small_filter_raster(weights)), image, atol=1e-4)
        weights = np.random.RandomState(2).uniform(0.0, 1.0, (21, 21))
        out = self.run_convolve(image, small_filter_raster(weights), normalize=1)
        np.testing.assert_allclose(out[12:-12, 12:-12], self.reference(image, weights / weights.sum())[12:-12, 12:-12], atol=1e-3)

    def test_filter_bounds_format_lets_the_filter_sit_off_centre(self):
        from nodebased.tiers import Region
        image = texture()
        display = Region(0, 0, 5, 5)
        kernel = small_filter_raster(np.ones((1, 1)), display=display, at=(3, 2))        # one pixel, 1 right of the format's centre
        shape = self.run_convolve(image, kernel, filter_bounds="shape")
        np.testing.assert_allclose(shape, image, atol=1e-6)
        fmt = self.run_convolve(image, kernel, filter_bounds="format")
        np.testing.assert_allclose(fmt[:, :-1], image[:, 1:], atol=1e-6)                   # each pixel takes its right neighbour

    def test_the_older_reading_is_the_top_left_corner_of_the_red_channel(self):
        image = texture()
        kernel_pixels = np.random.RandomState(4).uniform(0.1, 1.0, (8, 8, 4)).astype(np.float32)
        out = Evaluator._convolve(image, kernel_pixels, {"kernel_size": "3", "legacy_convolve": 1})
        weights = kernel_pixels[:3, :3, 0].astype(np.float32)
        weights /= weights.sum()
        padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode="edge")
        expected = np.zeros_like(image)
        for y in range(3):
            for x in range(3):
                expected += padded[y:y + 37, x:x + 45] * weights[y, x]
        np.testing.assert_array_equal(out, expected.astype(np.float32))

    def test_an_older_document_is_given_the_older_reading_and_draws_the_same(self):
        g = Graph()
        g.add("src", "Checker", dict(width=45, height=37, size=5))
        g.add("filt", "Checker", dict(width=8, height=8, size=2))
        g.add("n", "Convolve", dict(kernel_size="3"), image="src", kernel="filt")
        old = copy.deepcopy(g.doc)
        for name in ("legacy_convolve", "normalize", "filter_channel", "use_input_channels", "filter_bounds"):
            old["nodes"]["n"]["params"].pop(name, None)
        loaded = upgrade_document(old)
        validate(loaded)
        self.assertEqual(loaded["nodes"]["n"]["params"]["legacy_convolve"], 1)
        image, kernel = render(loaded, "src"), render(loaded, "filt")
        np.testing.assert_array_equal(render(loaded, "n"), Evaluator._convolve(image, kernel, {"kernel_size": "3", "legacy_convolve": 1}))

    def test_a_new_node_with_a_single_white_pixel_filter_is_the_identity_on_the_whole_frame(self):
        g = Graph()
        g.add("src", "Checker", dict(width=12, height=10, size=1))
        g.add("k", "Constant", dict(width=1, height=1, red=1.0, green=1.0, blue=1.0, alpha=1.0))
        g.add("fx", "Convolve", {}, image="src", kernel="k")
        np.testing.assert_array_equal(evaluator_pixels(g.doc, "fx"), evaluator_pixels(g.doc, "src"))
        executor = TileExecutor(evaluator=Evaluator())
        self.assertFalse(executor.supports_tiled(dict(g.doc, view="fx"), "fx"))          # a whole-frame job

    def test_a_new_node_is_a_whole_frame_job_and_an_older_one_keeps_its_tiles(self):
        from nodebased import ops2d_chrome
        self.assertTrue(ops2d_chrome.needs_full_frame("Convolve", dict(SPECS["Convolve"]["params"])))
        self.assertFalse(ops2d_chrome.needs_full_frame("Convolve", dict(SPECS["Convolve"]["params"], legacy_convolve=1)))


# ------------------------------------------------------------------------------------------------------------ EdgeDetect
class EdgeDetectTests(unittest.TestCase):
    def step(self):
        pixels = np.zeros((16, 32, 4), np.float32)
        pixels[:, 16:, :3] = 1.0
        pixels[..., 3] = 1.0
        return pixels

    def test_a_flat_picture_has_no_edges(self):
        flat = np.full((15, 15, 4), 0.37, np.float32)
        for method in ("Sobel", "Prewitt", "Laplacian"):
            np.testing.assert_allclose(kern("EdgeDetect", flat, edge_type=method)[..., :3], 0.0, atol=1e-6)

    def test_a_step_gives_an_edge_on_both_sides_of_it_and_nowhere_else(self):
        for method in ("Sobel", "Prewitt"):
            with self.subTest(method=method):
                out = kern("EdgeDetect", self.step(), edge_type=method)[..., 0]
                self.assertGreater(float(out[8, 15]), 0.0)
                self.assertGreater(float(out[8, 16]), 0.0)
                np.testing.assert_allclose(out[8, 15], out[8, 16], atol=1e-6)
                far = np.concatenate([out[:, :14], out[:, 18:]], axis=1)
                np.testing.assert_allclose(far, 0.0, atol=1e-6)

    def test_the_detectors_differ(self):
        pixels = texture()
        sobel, prewitt, laplacian = (kern("EdgeDetect", pixels, edge_type=m) for m in ("Sobel", "Prewitt", "Laplacian"))
        self.assertTrue(differs(sobel, prewitt))
        self.assertTrue(differs(sobel, laplacian))

    def test_threshold_is_subtracted_and_clipped_at_zero(self):
        pixels = texture()
        plain = kern("EdgeDetect", pixels)[..., :3]
        out = kern("EdgeDetect", pixels, threshold=0.4)[..., :3]
        np.testing.assert_allclose(out, np.maximum(plain - 0.4, 0.0), atol=1e-5)

    def test_blur_size_smooths_the_noise_out_of_the_edges(self):
        pixels = texture()
        plain = kern("EdgeDetect", pixels)[..., 0]
        blurred = kern("EdgeDetect", pixels, edge_blur_size=3.0)[..., 0]
        self.assertTrue(differs(plain, blurred))
        self.assertLess(float(blurred.mean()), float(plain.mean()))
        np.testing.assert_allclose(kern("EdgeDetect", pixels, edge_blur_size=3.0)[..., 0],
                                   kern("EdgeDetect", kern("Blur", pixels, radius=3.0, kernel_filter="gaussian", quality=15.0))[..., 0], atol=1e-5)

    def test_blur_quality_limits_a_wide_blur(self):
        pixels = texture()
        self.assertTrue(differs(kern("EdgeDetect", pixels, edge_blur_size=12.0, edge_blur_quality=15.0),
                                kern("EdgeDetect", pixels, edge_blur_size=12.0, edge_blur_quality=2.0)))

    def test_erode_size_thins_the_edges_and_a_negative_one_thickens_them(self):
        pixels = self.step()
        base = int((kern("EdgeDetect", pixels)[..., 0] > 1e-6).sum())
        thinner = int((kern("EdgeDetect", pixels, edge_erode_size=1.0)[..., 0] > 1e-6).sum())
        thicker = int((kern("EdgeDetect", pixels, edge_erode_size=-1.0)[..., 0] > 1e-6).sum())
        self.assertLess(thinner, base)
        self.assertGreater(thicker, base)

    def test_the_new_knobs_default_to_doing_nothing(self):
        params = SPECS["EdgeDetect"]["params"]
        self.assertEqual((params["edge_blur_size"], params["edge_erode_size"]), (0.0, 0.0))
        pixels = texture()
        np.testing.assert_array_equal(kern("EdgeDetect", pixels), kern("EdgeDetect", pixels, edge_blur_size=0.0, edge_erode_size=0.0))

    def test_an_older_document_draws_the_same(self):
        loaded = old_document("EdgeDetect", dict(edge_type="Prewitt", threshold=0.1),
                              ("edge_blur_size", "edge_erode_size", "edge_blur_quality"))
        source = render(loaded, "src")
        np.testing.assert_array_equal(render(loaded, "n"), Evaluator._edge_detect(source, {"edge_type": "Prewitt", "threshold": 0.1}))


# -------------------------------------------------------------------------------------------------------------- Emboss
OLD_EMBOSS = {"emboss_type": "effect", "edge_detector": "simple"}      # the relief alone, read with the central difference (an older document's node)


class EmbossTests(unittest.TestCase):
    """The guide gives no formula for Emboss; these are the properties of the directional relief NodeBased draws."""

    def relief(self, pixels, **knobs):
        return kern("Emboss", pixels, **{**OLD_EMBOSS, **knobs})[..., 0] - 0.5

    def test_a_flat_picture_is_a_flat_grey_relief(self):
        flat = np.full((15, 15, 4), 0.37, np.float32)
        np.testing.assert_allclose(kern("Emboss", flat, **OLD_EMBOSS)[..., :3], 0.5, atol=1e-6)

    def test_turning_the_angle_by_180_degrees_flips_the_relief(self):
        pixels = np.clip(texture(), 0.0, 1.0)
        for angle in (0.0, 35.0, 135.0):
            with self.subTest(angle=angle):
                np.testing.assert_allclose(self.relief(pixels, angle=angle, width=0.2), -self.relief(pixels, angle=angle + 180.0, width=0.2), atol=1e-5)

    def test_a_quarter_turn_reads_the_gradient_across_the_other_axis(self):
        ramp = np.zeros((12, 16, 4), np.float32)
        ramp[..., :3] = np.linspace(0.0, 0.5, 16)[None, :, None]
        self.assertGreater(float(np.abs(self.relief(ramp, angle=0.0, width=1.0)[6, 8])), 0.01)
        np.testing.assert_allclose(self.relief(ramp, angle=90.0, width=1.0)[6, 8], 0.0, atol=1e-6)

    def test_width_scales_the_relief_until_it_clips(self):
        pixels = np.clip(texture(), 0.0, 1.0)
        small = self.relief(pixels, angle=0.0, width=0.1)
        double = self.relief(pixels, angle=0.0, width=0.2)
        np.testing.assert_allclose(double, 2.0 * small, atol=1e-5)

    def test_angle_and_width_act(self):
        pixels = texture()
        base = kern("Emboss", pixels, angle=0.0, width=1.0)
        self.assertTrue(differs(base, kern("Emboss", pixels, angle=60.0, width=1.0)))
        self.assertTrue(differs(base, kern("Emboss", pixels, angle=0.0, width=3.0)))


# --------------------------------------------------------------------------------------------- every paired main-tab knob
class KnobsActTests(unittest.TestCase):
    """Every main-tab knob the mapping pairs with a NodeBased param changes the pixels when moved from its default."""
    # kind: [(param, changed value, base params the change is measured against)]
    ACTS = {
        "Blur": [("radius", 4.0, {}), ("kernel_filter", "box", {"radius": 4.0}), ("quality", 2.0, {"radius": 8.0})],
        "Erode": [("erode_size", 2.0, {}), ("erode_size", -2.0, {})],
        "ErodeFilter": [("filter_size", 3.0, {}), ("filter_type", "triangle", {"filter_size": 2.3})],
        "Median": [("median_size", 5.0, {}), ("ignore_top_line", 1, {})],
        "Sharpen": [("sharpen_amount", 2.0, {"sharpen_size": 2.0}), ("sharpen_size", 4.0, {}), ("kernel_filter", "box", {"sharpen_size": 3.0}),
                    ("quality", 2.0, {"sharpen_size": 9.0}), ("minimum", -10.0, {"sharpen_size": 2.0}), ("maximum", 100.0, {"sharpen_size": 2.0})],
        "Soften": [("amount", 0.5, {"soften_size": 3.0}), ("soften_size", 6.0, {}), ("kernel_filter", "box", {"soften_size": 3.0}),
                   ("quality", 2.0, {"soften_size": 9.0}), ("minimum", -10.0, {"soften_size": 3.0}), ("maximum", 100.0, {"soften_size": 3.0})],
        "Glow": [("glow_threshold", 1.0, {"glow_threshold": 0.0}), ("glow_size", 12.0, {}), ("brightness", 2.0, {}), ("red", 0.0, {}),
                 ("saturation", 0.0, {}), ("nonlinear", 1, {}), ("effect_only", 1, {}), ("kernel_filter", "box", {}), ("quality", 2.0, {"glow_size": 12.0}),
                 ("width_channel", "rgba.alpha", {})],
        "Matrix": [("normalize", 1, matrix_weights(((1, 1, 1), (1, 2, 1), (1, 1, 1)))), ("weight0", 0.5, {}), ("matrix_size", "5", {"weight12": 2.0, "weight0": 1.0})],
        "Laplacian": [("laplacian_size", 6.0, {}), ("kernel_filter", "box", {}), ("quality", 2.0, {"laplacian_size": 12.0})],
        "EdgeDetect": [("edge_type", "Prewitt", {}), ("threshold", 0.2, {}), ("edge_blur_size", 3.0, {}), ("edge_erode_size", 1.0, {}),
                       ("edge_blur_quality", 2.0, {"edge_blur_size": 12.0})],
        "Emboss": [("angle", 60.0, {"angle": 0.0}), ("width", 3.0, {})],
    }

    def test_every_listed_knob_changes_the_pixels(self):
        pixels = texture()
        for kind, acts in self.ACTS.items():
            for name, value, base in acts:
                with self.subTest(kind=kind, knob=name, value=value):
                    self.assertIn(name, SPECS[kind]["params"])
                    self.assertTrue(differs(kern(kind, pixels, **base), kern(kind, pixels, **{**base, name: value})), (kind, name))

    def test_mix_dissolves_between_the_original_and_the_effect(self):
        pixels = texture()
        for kind, base in (("Blur", {"radius": 4.0}), ("Erode", {"erode_size": 2.0}), ("Median", {}), ("Sharpen", {"sharpen_size": 2.0}),
                           ("Soften", {"soften_size": 3.0}), ("Glow", {"glow_threshold": 0.0}), ("Laplacian", {}), ("EdgeDetect", {}),
                           ("Emboss", {}), ("ErodeFilter", {"filter_size": 2.0})):
            with self.subTest(kind=kind):
                full = kern(kind, pixels, **base)
                half = kern(kind, pixels, mix=0.5, **base)
                np.testing.assert_array_equal(kern(kind, pixels, mix=0.0, **base), pixels)
                np.testing.assert_allclose(half, 0.5 * pixels + 0.5 * full, atol=1e-5)

    def test_crop_acts_on_the_window_of_every_blur_made_node(self):
        raster = Raster.of(texture(12, 16))
        for kind in ("Blur", "Sharpen", "Soften", "Glow", "Laplacian"):
            with self.subTest(kind=kind):
                size = {"Blur": "radius", "Sharpen": "sharpen_size", "Soften": "soften_size", "Glow": "glow_size", "Laplacian": "laplacian_size"}[kind]
                on = Evaluator._windowed_kernel(kind, {**SPECS[kind]["params"], size: 3.0, "crop": 1}, [raster])
                off = Evaluator._windowed_kernel(kind, {**SPECS[kind]["params"], size: 3.0, "crop": 0}, [raster])
                self.assertEqual(on.pixels.shape[:2], (12, 16))
                self.assertGreater(off.pixels.shape[0], 12)
                self.assertGreater(off.pixels.shape[1], 16)


# ------------------------------------------------------------------------------------------------------------- the tiles
class TileTests(unittest.TestCase):
    """The kernels this step changed give the same pixels in tiles of 7, 16 and 32 as on the whole frame."""
    CHANGED = {
        "Erode": dict(erode_size=3.0, channels="rgba"),
        "ErodeFilter": dict(filter_size=-3.0, filter_type="box"),
        "ErodeFilter ": dict(filter_size=2.5, filter_type="triangle"),
        "Median": dict(median_size=7.0, channels="rgba"),
        "Median top": dict(median_size=5.0, ignore_top_line=1),
        "Matrix": dict(matrix_size="5", **matrix_weights(((0, 0, 1, 0, 0), (0, 1, 2, 1, 0), (1, 2, -8, 2, 1), (0, 1, 2, 1, 0), (0, 0, 1, 0, 0)))),
        "Laplacian": dict(laplacian_size=4.0, kernel_filter="box", quality=1000.0),
        "Laplacian gaussian": dict(laplacian_size=3.0),
        "Laplacian scaled": dict(laplacian_size=9.0, kernel_filter="triangle", quality=2.0),
        "EdgeDetect": dict(edge_type="Sobel", threshold=0.05, edge_blur_size=2.0, edge_erode_size=1.0),
        "EdgeDetect scaled": dict(edge_type="Prewitt", edge_blur_size=9.0, edge_blur_quality=2.0, edge_erode_size=-2.0),
    }

    def test_tiles_equal_the_whole_frame(self):
        for label, params in self.CHANGED.items():
            kind = label.split()[0]
            for masked in (False, True):
                g = source_graph(kind, params, masked)
                whole = evaluator_pixels(g.doc, "node")
                for edge in (7, 16, 32):
                    with self.subTest(kind=label, masked=masked, tile=edge):
                        np.testing.assert_allclose(tile_pixels(g.doc, "node", edge), whole, atol=1e-5)
