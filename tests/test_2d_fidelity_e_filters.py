"""Fidelity step E, part 3: Blur, Sharpen and Soften take Nuke's filter shapes, quality limit, crop to format, and (the latter two)
a value range and Soften's amount. Kernel shapes by hand on an impulse, the quality limit, the growing window, tiles against the
evaluator, bypass and old documents."""
import copy
import math
import unittest

import numpy as np

from nodebased import blurknobs
from nodebased.core import SPECS, Dispatcher, bypass_slot, empty_document, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.raster import Raster
from nodebased.tileexec import TileExecutor
from nodebased.tiers import Region

N = 21


def impulse(n=N):
    pixels = np.zeros((n, n, 4), np.float32)
    pixels[n // 2, n // 2] = 1.0
    return pixels


def run(kind, pixels, **knobs):
    return Evaluator._windowed_kernel(kind, {**SPECS[kind]["params"], **knobs}, [Raster.of(np.asarray(pixels, np.float32))]).pixels


def blur(pixels, **knobs):
    return run("Blur", pixels, **knobs)


def row(pixels, channel=0):
    return pixels[N // 2, :, channel]


class KernelShapeTests(unittest.TestCase):
    """The response of a single bright pixel is the kernel itself; the 2-D response is the product of two 1-D kernels."""
    def assert_kernel(self, filter_name, size, expected):
        out = blur(impulse(), radius=size, kernel_filter=filter_name, quality=1000.0)
        k = np.array(expected, np.float64)
        k = k / k.sum()
        centre = N // 2
        reach = (len(k) - 1) // 2
        np.testing.assert_allclose(row(out)[centre - reach:centre + reach + 1], k * k[reach], atol=1e-6)
        np.testing.assert_allclose(out[centre - reach:centre + reach + 1, centre, 0], k * k[reach], atol=1e-6)
        self.assertAlmostEqual(float(out[..., 0].sum()), 1.0, places=5)                 # the energy stays
        outside = out[..., 0].copy()
        outside[centre - reach:centre + reach + 1, centre - reach:centre + reach + 1] = 0
        self.assertEqual(float(np.abs(outside).max()), 0.0)

    def test_box_weighs_everything_within_the_radius_the_same(self):
        self.assert_kernel("box", 2, [1, 1, 1, 1, 1])

    def test_triangle_is_a_tent(self):
        # size 3: weight 1 - |k| / 3
        self.assert_kernel("triangle", 3, [0, 1 / 3, 2 / 3, 1, 2 / 3, 1 / 3, 0])

    def test_quadratic_is_a_quadratic_b_spline(self):
        # size 3: u = 1.5 k / 3 = k / 2, weights 0.75 - u^2 (|u| <= 0.5), 0.5 (1.5 - |u|)^2 (to 1.5)
        self.assert_kernel("quadratic", 3, [0, 0.125, 0.5, 0.75, 0.5, 0.125, 0])

    def test_gaussian_has_sigma_a_third_of_the_size_and_stops_at_the_size(self):
        k = [math.exp(-0.5 * (j / 1.0) ** 2) for j in range(-3, 4)]                      # size 3: sigma 1, reach 3
        self.assert_kernel("gaussian", 3, k)

    def test_every_filter_keeps_a_flat_picture_flat_and_is_symmetric(self):
        flat = np.full((N, N, 4), 0.37, np.float32)
        for name in blurknobs.FILTERS:
            with self.subTest(filter=name):
                np.testing.assert_allclose(blur(flat, radius=4.5, kernel_filter=name), flat, atol=1e-6)
                out = blur(impulse(), radius=4, kernel_filter=name, quality=1000.0)
                np.testing.assert_allclose(out, out[::-1, ::-1], atol=1e-7)
                np.testing.assert_allclose(out, out.transpose(1, 0, 2), atol=1e-7)

    def test_the_filters_blur_differently(self):
        peaks = {name: float(blur(impulse(), radius=4, kernel_filter=name, quality=1000.0)[N // 2, N // 2, 0]) for name in blurknobs.FILTERS}
        self.assertEqual(len(set(round(v, 6) for v in peaks.values())), 4)
        self.assertLess(peaks["box"], peaks["gaussian"])                                  # the box spreads the energy flatter

    def test_below_half_a_pixel_every_filter_is_the_identity(self):
        source = np.random.default_rng(1).random((9, 9, 4)).astype(np.float32)
        for name in blurknobs.FILTERS:
            np.testing.assert_array_equal(blur(source, radius=0.4, kernel_filter=name), source)

    def test_an_unknown_filter_is_refused(self):
        with self.assertRaises(ValueError):
            blurknobs.weights("lanczos", 3.0)


class SoftenAndBlurShareTheGaussianTests(unittest.TestCase):
    def test_a_gaussian_blur_is_a_soften_of_the_same_size(self):
        source = np.random.default_rng(2).random((17, 19, 4)).astype(np.float32)
        np.testing.assert_array_equal(blur(source, radius=5.0, kernel_filter="gaussian"),
                                      run("Soften", source, soften_size=5.0, channels="rgba"))

    def test_the_legacy_soften_arithmetic_is_unchanged(self):
        source = np.random.default_rng(3).random((15, 15, 4)).astype(np.float32)
        size = 4.0
        radius, sigma = 4, size / 3.0
        expected = Evaluator._gaussian_axis(Evaluator._gaussian_axis(source, radius, sigma, axis=1), radius, sigma, axis=0)
        np.testing.assert_array_equal(run("Soften", source, soften_size=size), expected)


class QualityTests(unittest.TestCase):
    def test_a_filter_within_the_limit_is_the_plain_filter(self):
        source = np.random.default_rng(4).random((32, 32, 4)).astype(np.float32)
        self.assertEqual(blurknobs.scale_factor("Blur", {"radius": 8.0, "quality": 8.0}), 1)
        np.testing.assert_array_equal(blur(source, radius=8.0, kernel_filter="gaussian", quality=8.0),
                                      blur(source, radius=8.0, kernel_filter="gaussian", quality=1000.0))

    def test_the_scale_factor_is_the_whole_number_that_fits_the_filter_in_the_limit(self):
        for size, quality, factor in ((40, 10, 4), (41, 10, 5), (15, 15, 1), (16, 15, 2), (100, 15, 7)):
            self.assertEqual(blurknobs.scale_factor("Blur", {"radius": float(size), "quality": float(quality)}), factor)

    def test_a_wide_blur_scaled_down_still_blurs_a_smooth_picture_alike(self):
        n = 96
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        bump = np.exp(-(((xx - 48) ** 2 + (yy - 48) ** 2) / (2 * 14.0 ** 2))).astype(np.float32)
        source = np.stack([bump, bump * 0.5, bump * 0.25, np.ones_like(bump)], axis=-1)
        exact = blur(source, radius=24.0, kernel_filter="gaussian", quality=1000.0)
        scaled = blur(source, radius=24.0, kernel_filter="gaussian", quality=8.0)
        self.assertEqual(scaled.shape, exact.shape)
        self.assertLess(float(np.abs(scaled - exact).max()), 0.02)
        self.assertGreater(float(np.abs(scaled - source).max()), 0.05)                    # and it did blur

    def test_a_flat_picture_and_a_ramp_survive_the_scaling(self):
        flat = np.full((50, 70, 4), 0.4, np.float32)
        np.testing.assert_allclose(blur(flat, radius=30.0, quality=5.0), flat, atol=1e-6)
        ramp = np.zeros((40, 64, 4), np.float32)
        ramp[..., 0] = np.linspace(0.0, 1.0, 64, dtype=np.float32)[None, :]
        out = blur(ramp, radius=12.0, kernel_filter="box", quality=4.0)
        np.testing.assert_allclose(out[:, 24:40, 0], ramp[:, 24:40, 0], atol=2e-3)       # the middle of a ramp does not move

    def test_odd_sizes_work(self):
        out = blur(np.random.default_rng(5).random((37, 53, 4)).astype(np.float32), radius=20.0, quality=6.0)
        self.assertEqual(out.shape, (37, 53, 4))
        self.assertTrue(np.isfinite(out).all())


class CropToFormatTests(unittest.TestCase):
    def raster(self):
        pixels = np.ones((8, 10, 4), np.float32)
        return Raster(pixels, Region(10, 8, 10, 8), Region(0, 0, 40, 30))

    def test_crop_on_keeps_the_window_off_grows_it_by_the_reach(self):
        for kind, knob in (("Blur", "radius"), ("Sharpen", "sharpen_size"), ("Soften", "soften_size")):
            with self.subTest(kind=kind):
                on = Evaluator._windowed_kernel(kind, {**SPECS[kind]["params"], knob: 3.0, "crop": 1}, [self.raster()])
                off = Evaluator._windowed_kernel(kind, {**SPECS[kind]["params"], knob: 3.0, "crop": 0}, [self.raster()])
                self.assertEqual(on.data, Region(10, 8, 10, 8))
                self.assertEqual(off.data, Region(7, 5, 16, 14))

    def test_the_grown_edge_is_transparent_black_blurred_in(self):
        out = Evaluator._windowed_kernel("Blur", {**SPECS["Blur"]["params"], "radius": 3.0, "crop": 0, "kernel_filter": "box"}, [self.raster()])
        corner = out.pixels[0, 0]
        self.assertGreater(float(corner[3]), 0.0)
        self.assertLess(float(corner[3]), 0.2)                                        # a sliver of the picture reached the new edge
        np.testing.assert_allclose(out.pixels[7, 8], 1.0, atol=1e-6)                   # deep inside it is untouched

    def test_a_picture_the_filter_does_not_change_is_not_grown_when_it_does_nothing(self):
        out = Evaluator._windowed_kernel("Blur", {**SPECS["Blur"]["params"], "radius": 0.3, "crop": 0}, [self.raster()])
        self.assertEqual(out.data, Region(10, 8, 10, 8))


class SharpenTests(unittest.TestCase):
    def test_it_adds_the_detail_back_scaled_by_amount(self):
        source = impulse()
        blurred = blur(source, radius=2.0, kernel_filter="gaussian", quality=1000.0)
        for amount in (0.0, 0.5, 1.0, 2.0):
            with self.subTest(amount=amount):
                out = run("Sharpen", source, sharpen_size=2.0, sharpen_amount=amount, kernel_filter="gaussian", minimum=-10.0, maximum=10.0)
                np.testing.assert_allclose(out[..., :3], source[..., :3] + amount * (source[..., :3] - blurred[..., :3]), atol=1e-6)
                np.testing.assert_array_equal(out[..., 3], source[..., 3])                  # channels rgb: alpha is not sharpened

    def test_the_filter_picks_the_blur_it_subtracts(self):
        source = impulse()
        a = run("Sharpen", source, sharpen_size=3.0, kernel_filter="box", minimum=-10.0, maximum=10.0)
        b = run("Sharpen", source, sharpen_size=3.0, kernel_filter="gaussian", minimum=-10.0, maximum=10.0)
        self.assertGreater(float(np.abs(a - b).max()), 0.01)

    def test_values_outside_the_range_pass_unchanged(self):
        source = np.zeros((N, N, 4), np.float32)
        source[:, N // 2:, :3] = 0.5
        source[N // 2, N // 2 + 4, :3] = 3.0                                              # a bright pixel above the range
        source[N // 2, N // 2 - 4, :3] = -0.5                                             # and a dark one below it
        out = run("Sharpen", source, sharpen_size=2.0, sharpen_amount=1.5, minimum=0.0, maximum=1.0)
        self.assertEqual(float(out[N // 2, N // 2 + 4, 0]), 3.0)
        self.assertEqual(float(out[N // 2, N // 2 - 4, 0]), -0.5)
        inside = (source[..., 0] >= 0.0) & (source[..., 0] <= 1.0)
        self.assertGreater(float(np.abs(out[..., 0] - source[..., 0])[inside].max()), 0.01)    # while the rest was sharpened
        wide = run("Sharpen", source, sharpen_size=2.0, sharpen_amount=1.5, minimum=-10.0, maximum=10.0)
        self.assertNotEqual(float(wide[N // 2, N // 2 + 4, 0]), 3.0)

    def test_a_flat_picture_is_unchanged(self):
        flat = np.full((N, N, 4), 0.6, np.float32)
        np.testing.assert_allclose(run("Sharpen", flat, sharpen_size=3.0, sharpen_amount=2.0), flat, atol=1e-6)


class SoftenTests(unittest.TestCase):
    def test_amount_goes_from_the_picture_to_the_soft_one_and_past(self):
        source = impulse()
        soft = run("Soften", source, soften_size=3.0, amount=1.0, minimum=-10.0, maximum=10.0)
        for amount in (0.0, 0.5, 2.0):
            with self.subTest(amount=amount):
                out = run("Soften", source, soften_size=3.0, amount=amount, minimum=-10.0, maximum=10.0)
                np.testing.assert_allclose(out, source + amount * (soft - source), atol=1e-6)

    def test_the_filter_shapes_the_softness(self):
        a = run("Soften", impulse(), soften_size=3.0, kernel_filter="box", minimum=-10.0, maximum=10.0)
        b = run("Soften", impulse(), soften_size=3.0, kernel_filter="triangle", minimum=-10.0, maximum=10.0)
        self.assertGreater(float(np.abs(a - b).max()), 0.001)

    def test_values_outside_the_range_pass_unchanged(self):
        source = np.zeros((N, N, 4), np.float32)
        source[N // 2, N // 2, :3] = 4.0
        source[N // 2, N // 2 + 1, :3] = 0.5
        out = run("Soften", source, soften_size=2.0, minimum=0.0, maximum=1.0)
        self.assertEqual(float(out[N // 2, N // 2, 0]), 4.0)
        self.assertNotEqual(float(out[N // 2, N // 2 + 1, 0]), 0.5)

    def test_channels_still_limit_the_effect(self):
        source = impulse()
        out = run("Soften", source, soften_size=2.0, channels="alpha", minimum=-10.0, maximum=10.0)
        np.testing.assert_array_equal(out[..., :3], source[..., :3])
        self.assertNotEqual(float(out[N // 2, N // 2, 3]), 1.0)


class FilterDocumentTests(unittest.TestCase):
    def graph(self, kind, **knobs):
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "src", "type": "Ramp", "params": {
            "width": 24, "height": 14, "p1_x": 23.0, "color0_red": 0.0, "color1_red": 1.0, "color0_green": 0.9, "color1_green": 0.1,
            "color0_alpha": 0.5, "color1_alpha": 1.0}})
        d.execute({"op": "create", "id": "n", "type": kind, "params": knobs})
        d.execute({"op": "connect", "id": "n", "input": "image", "source": "src"})
        d.execute({"op": "view", "id": "n"})
        return d

    def tiles_equal(self, d, tiled=True):
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=8)
        expected = Evaluator().evaluate_raster(d.document, target="n").pixels
        self.assertEqual(executor.supports_tiled(d.document, "n"), tiled)
        region = executor.canvas_region(d.document, "n", frame=1, tier=1)
        got = executor.compose_region(d.document, "n", region, frame=1, tier=1).pixels
        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_tiles_equal_the_evaluator_for_every_filter(self):
        for kind, size in (("Blur", {"radius": 3.0}), ("Sharpen", {"sharpen_size": 3.0, "sharpen_amount": 1.2}), ("Soften", {"soften_size": 3.0})):
            for name in blurknobs.FILTERS:
                with self.subTest(kind=kind, filter=name):
                    self.tiles_equal(self.graph(kind, **size, kernel_filter=name))

    def test_tiles_equal_the_evaluator_with_the_value_range_and_amount(self):
        self.tiles_equal(self.graph("Sharpen", sharpen_size=2.0, minimum=0.2, maximum=0.8, sharpen_amount=2.0))
        self.tiles_equal(self.graph("Soften", soften_size=2.0, minimum=0.2, maximum=0.8, amount=0.6))

    def test_a_growing_window_runs_on_the_whole_image_but_a_scaled_picture_stays_on_tiles(self):
        for kind, knob in (("Blur", "radius"), ("Sharpen", "sharpen_size"), ("Soften", "soften_size")):
            with self.subTest(kind=kind, what="crop off"):
                d = self.graph(kind, **{knob: 3.0}, crop=0)
                self.assertFalse(TileExecutor(evaluator=Evaluator(), tile_edge=8).supports_tiled(d.document, "n"))
            with self.subTest(kind=kind, what="quality"):
                d = self.graph(kind, **{knob: 20.0}, quality=4.0)
                self.tiles_equal(d)

    def test_tiles_equal_the_evaluator_when_quality_scales_the_picture_down(self):
        for kind, size in (("Blur", "radius"), ("Sharpen", "sharpen_size"), ("Soften", "soften_size")):
            for name in blurknobs.FILTERS:
                for edge in (8, 16, 24):
                    with self.subTest(kind=kind, filter=name, tile=edge):
                        d = self.graph(kind, **{size: 11.0}, kernel_filter=name, quality=3.0, mix=0.9)
                        executor = TileExecutor(evaluator=Evaluator(), tile_edge=edge)
                        expected = Evaluator().evaluate_raster(d.document, target="n").pixels
                        self.assertTrue(executor.supports_tiled(d.document, "n"))
                        region = executor.canvas_region(d.document, "n", frame=1, tier=1)
                        got = executor.compose_region(d.document, "n", region, frame=1, tier=1).pixels
                        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_a_piece_of_a_picture_blurs_to_what_the_whole_picture_does_inside_the_halo(self):
        rng = np.random.default_rng(11)
        whole = rng.random((60, 90, 4)).astype(np.float32)
        for name in blurknobs.FILTERS:
            for size, quality in ((12.0, 4.0), (21.0, 5.0), (9.0, 2.0)):
                p = {"radius": size, "kernel_filter": name, "quality": quality}
                expected = blurknobs.blur(whole, p, (0, 0))
                halo = blurknobs.halo("Blur", p)
                for x0, y0 in ((halo + 3, halo + 1), (7 * halo + 2, 2 * halo + 5)):       # origins on and off the block grid
                    core_w, core_h = 20, 18
                    x1, y1 = min(90, x0 + core_w + halo), min(60, y0 + core_h + halo)
                    if x0 + core_w >= 90 or y0 + core_h >= 60:
                        continue
                    piece = whole[y0 - halo:y1, x0 - halo:x1]
                    got = blurknobs.blur(piece, p, (x0 - halo, y0 - halo))
                    with self.subTest(filter=name, size=size, quality=quality, origin=(x0, y0)):
                        np.testing.assert_allclose(got[halo:halo + core_h, halo:halo + core_w], expected[y0:y0 + core_h, x0:x0 + core_w], atol=1e-6)

    def test_the_halo_is_the_reach_until_quality_scales_the_picture(self):
        self.assertEqual(blurknobs.halo("Blur", {"radius": 8.0, "quality": 15.0}), 8)
        self.assertEqual(blurknobs.halo("Blur", {"radius": 0.3, "quality": 15.0}), 0)
        self.assertEqual(blurknobs.halo("Blur", {"radius": 40.0, "quality": 10.0}), 4 * (10 + 2))
        self.assertEqual(blurknobs.halo("Soften", {"soften_size": 41.0, "quality": 10.0}), 5 * (9 + 2))

    def test_bypass_passes_the_image(self):
        for kind in blurknobs.KINDS:
            with self.subTest(kind=kind):
                d = self.graph(kind, crop=0)
                d.execute({"op": "disable", "id": "n", "value": True})
                self.assertEqual(bypass_slot(d.document["nodes"]["n"]), "image")
                np.testing.assert_array_equal(Evaluator().evaluate(dict(d.document, view="n"), "n", frame=1),
                                              Evaluator().evaluate(dict(d.document, view="src"), "src", frame=1))

    def test_an_older_document_gets_the_old_values_and_draws_the_same(self):
        for kind in blurknobs.KINDS:
            with self.subTest(kind=kind):
                d = self.graph(kind)
                old = copy.deepcopy(d.document)
                for name in blurknobs.NEW[kind]:
                    old["nodes"]["n"]["params"].pop(name)
                loaded = upgrade_document(old)
                validate(loaded)
                for name, value in blurknobs.OLD[kind].items():
                    self.assertEqual(loaded["nodes"]["n"]["params"][name], value)
                source = Evaluator().evaluate_raster(dict(loaded, view="src"), target="src").pixels
                got = Evaluator().evaluate_raster(dict(loaded, view="n"), target="n").pixels
                params = loaded["nodes"]["n"]["params"]
                if kind == "Blur":
                    expected = Evaluator._box_blur_axis(Evaluator._box_blur_axis(source, params["radius"], axis=1), params["radius"], axis=0)
                elif kind == "Sharpen":
                    box = Evaluator._box_blur_axis(Evaluator._box_blur_axis(source, params["sharpen_size"], axis=1), params["sharpen_size"], axis=0)
                    expected = source.copy()
                    expected[..., :3] = source[..., :3] + (source[..., :3] - box[..., :3]) * params["sharpen_amount"]
                else:
                    r, sigma = int(math.ceil(params["soften_size"])), params["soften_size"] / 3.0
                    expected = Evaluator._gaussian_axis(Evaluator._gaussian_axis(source, r, sigma, axis=1), r, sigma, axis=0)
                np.testing.assert_array_equal(got, expected)

    def test_a_new_node_follows_nuke(self):
        self.assertEqual(SPECS["Blur"]["params"]["kernel_filter"], "gaussian")
        self.assertEqual(SPECS["Blur"]["params"]["quality"], 15.0)
        self.assertEqual(SPECS["Blur"]["params"]["crop"], 1)
        self.assertEqual((SPECS["Sharpen"]["params"]["minimum"], SPECS["Sharpen"]["params"]["maximum"]), (0.0, 1.0))
        self.assertEqual(SPECS["Soften"]["params"]["amount"], 1.0)

    def test_the_new_knobs_validate_and_a_bad_filter_is_refused(self):
        d = self.graph("Soften", kernel_filter="quadratic", quality=3.0, crop=0, minimum=-1.0, maximum=2.0, amount=0.5)
        validate(d.document)
        with self.assertRaises(ValueError):
            self.graph("Blur", kernel_filter="lanczos")


if __name__ == "__main__":
    unittest.main()
