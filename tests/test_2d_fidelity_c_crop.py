"""Fidelity step C, part 4: Crop takes Nuke's softness, reformat, intersect and black-outside knobs. Pixel assertions per
option, the window and display each one sets, mask and mix, bypass, proxy tiers, the tile path (Crop is evaluated whole,
like Transform and Mirror) and old documents."""
import copy
import unittest

import numpy as np

from nodebased import tiers
from nodebased.core import SPECS, Dispatcher, bypass_slot, empty_document, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor
from nodebased.tiers import Region

W, H = 20, 12


def source_pixels():
    rng = np.random.default_rng(5)
    return (rng.random((H, W, 4)) * 0.8 + 0.1).astype(np.float32)


def document(box=(4, 3, 10, 6), bypass=False, mask=None, **params):
    d = Dispatcher(empty_document())
    d.execute({"op": "create", "id": "src", "type": "Checker", "params": {"width": W, "height": H, "size": 3}})
    d.execute({"op": "create", "id": "crop", "type": "Crop", "params": {"x": box[0], "y": box[1], "width": box[2], "height": box[3], **params}})
    d.execute({"op": "connect", "id": "crop", "input": "image", "source": "src"})
    if mask is not None:
        d.execute({"op": "create", "id": "m", "type": "Constant", "params": {"width": W, "height": H, "red": 1, "green": 1, "blue": 1, "alpha": mask}})
        d.execute({"op": "connect", "id": "crop", "input": "mask", "source": "m"})
    if bypass:
        d.execute({"op": "disable", "id": "crop", "disabled": True}) if False else None
        d.document["nodes"]["crop"]["disabled"] = True
    d.execute({"op": "view", "id": "crop"})
    return d.document


def render(doc, target="crop"):
    return Evaluator().evaluate_raster(doc, target=target)


class DefaultsTests(unittest.TestCase):
    def test_the_specs_carry_the_defaults(self):
        params = SPECS["Crop"]["params"]
        self.assertEqual((params["crop_softness"], params["reformat"], params["intersect"], params["crop"]), (0.0, 0, 1, 1))

    def test_default_crop_is_the_box_cut_from_the_picture(self):
        raster = render(document())
        source = render(document(), "src")
        self.assertEqual(raster.data, Region(4, 3, 10, 6))
        self.assertEqual(raster.display, source.display)
        np.testing.assert_array_equal(raster.pixels, source.pixels[3:9, 4:14])

    def test_a_box_beyond_the_picture_is_cut_to_it_by_default(self):
        raster = render(document(box=(-5, -5, 100, 100)))
        self.assertEqual(raster.data, Region(0, 0, W, H))


class SoftnessTests(unittest.TestCase):
    def expected_vignette(self, box, softness):
        x, y, w, h = box
        xs = np.arange(x, x + w) + 0.5
        ys = np.arange(y, y + h) + 0.5
        fx = np.clip(np.minimum(xs - x, x + w - xs) / softness, 0, 1)
        fy = np.clip(np.minimum(ys - y, y + h - ys) / softness, 0, 1)
        return (fy[:, None] * fx[None, :])[..., None]

    def test_the_edges_fade_to_black_over_the_softness(self):
        plain = render(document()).pixels
        for softness in (1.0, 2.5, 4.0):
            with self.subTest(softness=softness):
                soft = render(document(crop_softness=softness))
                self.assertEqual(soft.data, Region(4, 3, 10, 6))                       # the box does not move or grow
                np.testing.assert_allclose(soft.pixels, plain * self.expected_vignette((4, 3, 10, 6), softness), atol=1e-6)

    def test_the_centre_is_untouched_and_the_edges_are_darker(self):
        plain = render(document()).pixels
        soft = render(document(crop_softness=2.0)).pixels
        np.testing.assert_array_equal(soft[2:4, 3:7], plain[2:4, 3:7])
        self.assertLess(float(soft[0, 0, 3]), float(plain[0, 0, 3]))
        self.assertLess(float(soft[0, 5, 0]), float(plain[0, 5, 0]) + 1e-9)

    def test_softness_zero_is_exactly_the_old_crop(self):
        np.testing.assert_array_equal(render(document(crop_softness=0.0)).pixels, render(document()).pixels)

    def test_the_array_kernel_agrees_with_the_window_path(self):
        pixels = source_pixels()
        p = dict(SPECS["Crop"]["params"], x=4, y=3, width=10, height=6, crop_softness=3.0)
        legacy = Evaluator._crop(pixels, p)
        raster_doc = document(crop_softness=3.0)
        region = render(raster_doc)
        source = render(raster_doc, "src")
        full = Evaluator._crop(source.pixels, p)
        np.testing.assert_allclose(full[3:9, 4:14], region.pixels, atol=1e-6)
        self.assertEqual(float(legacy[0, 0, 0]), 0.0)

    def test_proxy_tiers_scale_the_softness_with_the_box(self):
        scaled = tiers.scale_params("Crop", dict(SPECS["Crop"]["params"], x=8, y=8, width=64, height=32, crop_softness=8.0), 4)
        self.assertEqual((scaled["x"], scaled["width"]), (2, 16))
        self.assertAlmostEqual(scaled["crop_softness"], 2.0)


class IntersectAndBlackOutsideTests(unittest.TestCase):
    BOX = (-4, -3, W + 9, H + 7)

    def test_intersect_off_lets_the_box_extend_past_the_picture_filled_with_black(self):
        raster = render(document(box=self.BOX, intersect=0))
        self.assertEqual(raster.data, Region(*self.BOX))
        source = render(document(), "src").pixels
        np.testing.assert_array_equal(raster.pixels[3:3 + H, 4:4 + W], source)
        outside = raster.pixels.copy()
        outside[3:3 + H, 4:4 + W] = 0
        self.assertEqual(float(np.abs(outside).max()), 0.0)

    def test_black_outside_off_fills_the_extra_area_with_the_outermost_pixels(self):
        raster = render(document(box=self.BOX, intersect=0, crop=0))
        source = render(document(), "src").pixels
        expected = np.pad(source, ((3, 4), (4, 5), (0, 0)), mode="edge")
        self.assertEqual(raster.data, Region(*self.BOX))
        np.testing.assert_array_equal(raster.pixels, expected)

    def test_black_outside_off_does_nothing_while_intersect_is_on(self):
        np.testing.assert_array_equal(render(document(crop=0)).pixels, render(document()).pixels)
        np.testing.assert_array_equal(render(document(box=self.BOX, crop=0)).pixels, render(document(box=self.BOX)).pixels)

    def test_a_box_inside_the_picture_is_the_same_either_way(self):
        np.testing.assert_array_equal(render(document(intersect=0)).pixels, render(document()).pixels)
        np.testing.assert_array_equal(render(document(intersect=0, crop=0)).pixels, render(document()).pixels)

    def test_softness_applies_to_the_extended_area_too(self):
        edge = render(document(box=self.BOX, intersect=0, crop=0)).pixels
        soft = render(document(box=self.BOX, intersect=0, crop=0, crop_softness=3.0)).pixels
        self.assertLess(float(soft[0, 0, 3]), float(edge[0, 0, 3]))
        np.testing.assert_array_equal(soft[8:12, 12:20], edge[8:12, 12:20])


class ReformatTests(unittest.TestCase):
    def test_reformat_makes_the_box_the_output_format(self):
        raster = render(document(reformat=1))
        self.assertEqual(raster.display, Region(4, 3, 10, 6))
        self.assertEqual(raster.data, Region(4, 3, 10, 6))
        self.assertEqual(render(document()).display, Region(0, 0, W, H))

    def test_to_display_of_a_reformatted_crop_is_the_boxs_pixels(self):
        pixels = render(document(reformat=1)).to_display()
        np.testing.assert_array_equal(pixels, render(document(), "src").pixels[3:9, 4:14])

    def test_reformat_with_a_larger_box_pads_the_format_with_black(self):
        raster = render(document(box=(-4, -3, W + 9, H + 7), intersect=0, reformat=1))
        self.assertEqual(raster.display, Region(-4, -3, W + 9, H + 7))
        self.assertEqual(raster.to_display().shape[:2], (H + 7, W + 9))


class GatingAndPathsTests(unittest.TestCase):
    def test_mask_and_mix_still_gate_the_result(self):
        source = render(document(), "src")
        cropped = render(document(crop_softness=2.0))
        gated = render(document(crop_softness=2.0, mask=0.5, mix=0.5))
        box = Region(0, 0, W, H)
        expected = source.fit(box) * (1 - 0.25) + cropped.fit(box) * 0.25
        np.testing.assert_allclose(gated.fit(box), expected, atol=1e-6)

    def test_bypass_passes_the_image(self):
        doc = document(crop_softness=3.0, intersect=0, reformat=1, bypass=True)
        np.testing.assert_array_equal(render(doc).pixels, render(doc, "src").pixels)
        self.assertEqual(bypass_slot(doc["nodes"]["crop"]), "image")

    def test_the_tile_executor_renders_crop_whole_and_matches_the_evaluator(self):
        for options in (dict(), dict(crop_softness=2.0), dict(intersect=0, crop=0, box=(-4, -3, W + 9, H + 7))):
            with self.subTest(options=options):
                doc = document(**options)
                executor = TileExecutor(evaluator=Evaluator(), tile_edge=8)
                self.assertFalse(executor.supports_tiled(doc, "crop"))               # Crop is outside the tiled set, like Transform
                expected = render(doc)
                got = executor.compose(doc, "crop", frame=1, tier=1).pixels
                self.assertEqual(got.shape[:2], (expected.display.height, expected.display.width))
                np.testing.assert_allclose(got, expected.to_display(), atol=1e-6)


class DocumentTests(unittest.TestCase):
    def test_a_crop_saved_before_the_knobs_existed_loads_with_the_defaults_and_crops_the_same(self):
        doc = document()
        old = copy.deepcopy(doc)
        for name in ("crop_softness", "reformat", "intersect", "crop"):
            old["nodes"]["crop"]["params"].pop(name, None)
        loaded = upgrade_document(old)
        validate(loaded)
        params = loaded["nodes"]["crop"]["params"]
        self.assertEqual((params.get("crop_softness", 0.0), params.get("reformat", 0), params.get("intersect", 1), params.get("crop", 1)),
                         (0.0, 0, 1, 1))
        np.testing.assert_array_equal(render(loaded).pixels, render(doc).pixels)


if __name__ == "__main__":
    unittest.main()
