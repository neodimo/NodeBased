"""Step 5c: Shuffle layers, STMap, IDistort, VectorBlur. The maps and vector fields arrive as
named layers of a real multichannel EXR (or a wired uv image), so the layer plumbing from Write and
Read through the graph is exercised end to end. Pixel assertions throughout."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodebased.core import SPECS, bypass_slot
from nodebased.imaging import Evaluator
from nodebased.media import write_exr
from nodebased.tileexec import SUPPORTED_TILED_KINDS, TileExecutor
from nodebased.tiers import Region, input_regions, scale_params
from tests.test_2d_parity_group_3b import Graph, evaluator_pixels, evaluator_raster, tile_pixels

W, H = 24, 16
KINDS = ("STMap", "IDistort", "VectorBlur")


def rgba(fill=(0, 0, 0, 1.0)):
    frame = np.zeros((H, W, 4), np.float32)
    frame[:] = fill
    return frame


def dot_image(x=10, y=8, value=1.0):
    frame = rgba((0, 0, 0, 0))
    frame[y, x] = (value, value, value, 1.0)
    return frame


def texture():
    yy, xx = np.mgrid[0:H, 0:W]
    frame = np.zeros((H, W, 4), np.float32)
    frame[..., 0] = xx / W
    frame[..., 1] = yy / H
    frame[..., 2] = ((xx * 3 + yy * 5) % 7) / 7.0
    frame[..., 3] = 1.0
    return frame


def constant_layer(a, b):
    layer = rgba()
    layer[..., 0], layer[..., 1] = a, b
    return layer


def identity_map():
    """Absolute normalised coordinates of every pixel centre; v runs bottom to top."""
    yy, xx = np.mgrid[0:H, 0:W]
    return np.stack([(xx + 0.5) / W, 1.0 - (yy + 0.5) / H, np.zeros((H, W)), np.ones((H, W))],
                    axis=2).astype(np.float32)


class Scratch(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.count = 0

    def exr(self, beauty, **layers):
        self.count += 1
        path = Path(self._dir.name) / f"src{self.count}.exr"
        write_exr(path, beauty, bits="float", layers=layers or None)
        return str(path)

    def read(self, g, key, beauty, **layers):
        return g.add(key, "Read", {"path": self.exr(beauty, **layers)})

    def run_node(self, kind, beauty, params=None, uv_image=None, mask=None, layers=None):
        """Evaluate `kind` over a Read of `beauty` (with named `layers`); `uv_image`/`mask` wire
        extra Reads into the uv and mask slots."""
        g = Graph()
        self.read(g, "src", beauty, **(layers or {}))
        wires = {"image": "src"}
        if uv_image is not None:
            self.read(g, "uv", uv_image)
            wires["uv"] = "uv"
        if mask is not None:
            self.read(g, "mask", mask)
            wires["mask"] = "mask"
        g.add("node", kind, params or {}, **wires)
        return g, evaluator_raster(g.doc, "node")


class ShuffleLayerTests(Scratch):
    def test_a_named_layer_is_routed_to_rgba(self):
        g = Graph()
        self.read(g, "src", texture(), motion=constant_layer(3.0, -2.0), depth=rgba((7, 7, 7, 1)))
        g.add("s", "Shuffle", dict(layer="motion", red_from="R", green_from="G", blue_from="0", alpha_from="1"),
              image="src")
        out = evaluator_pixels(g.doc, "s")
        np.testing.assert_array_equal(out[..., 0], np.full((H, W), 3.0, np.float32))
        np.testing.assert_array_equal(out[..., 1], np.full((H, W), -2.0, np.float32))
        np.testing.assert_array_equal(out[..., 2:], np.stack([np.zeros((H, W)), np.ones((H, W))], 2))
        g.add("d", "Shuffle", dict(layer="depth", red_from="R", green_from="R", blue_from="R", alpha_from="1"),
              image="src")
        np.testing.assert_array_equal(evaluator_pixels(g.doc, "d")[..., 0], np.full((H, W), 7.0, np.float32))

    def test_empty_or_rgba_layer_shuffles_the_beauty_as_before(self):
        g = Graph()
        self.read(g, "src", texture(), motion=constant_layer(3.0, -2.0))
        for layer in ("", "rgba"):
            g.add("s" + layer, "Shuffle", dict(layer=layer, red_from="B", green_from="G", blue_from="R"), image="src")
            out = evaluator_pixels(g.doc, "s" + layer)
            np.testing.assert_array_equal(out[..., 0], texture()[..., 2])
            np.testing.assert_array_equal(out[..., 2], texture()[..., 0])

    def test_unknown_layer_is_an_error_that_lists_what_exists(self):
        g = Graph()
        self.read(g, "src", texture(), motion=constant_layer(1.0, 1.0))
        g.add("s", "Shuffle", dict(layer="normals"), image="src")
        with self.assertRaisesRegex(ValueError, r"no layer 'normals'.*available: rgba, motion"):
            evaluator_pixels(g.doc, "s")
        plain = Graph()
        self.read(plain, "src", texture())
        plain.add("s", "Shuffle", dict(layer="motion"), image="src")
        with self.assertRaisesRegex(ValueError, "no layer 'motion'.*none"):
            evaluator_pixels(plain.doc, "s")

    def test_layer_plumbing_on_the_tile_path(self):
        """A layered shuffle is refused by the tile path (a tile carries one RGBA array), so the
        caller falls back to the evaluator; an unlayered one stays tiled and equals the evaluator."""
        g = Graph()
        self.read(g, "src", texture(), motion=constant_layer(3.0, -2.0))
        g.add("plain", "Shuffle", dict(red_from="B", blue_from="R"), image="src")
        g.add("layered", "Shuffle", dict(layer="motion"), image="src")
        executor = TileExecutor(evaluator=Evaluator())
        self.assertIn("Shuffle", SUPPORTED_TILED_KINDS)
        self.assertTrue(executor.supports_tiled(dict(g.doc, view="plain"), "plain"))
        self.assertFalse(executor.supports_tiled(dict(g.doc, view="layered"), "layered"))
        np.testing.assert_array_equal(tile_pixels(g.doc, "plain"), evaluator_pixels(g.doc, "plain"))

    def test_bypass_and_old_documents(self):
        g = Graph()
        self.read(g, "src", texture(), motion=constant_layer(3.0, -2.0))
        g.add("s", "Shuffle", dict(layer="motion"), image="src")
        g.bypass("s")
        np.testing.assert_array_equal(evaluator_pixels(g.doc, "s"), texture())
        old = Graph()
        self.read(old, "src", texture())
        old.add("s", "Shuffle", dict(red_from="B", blue_from="R"), image="src")
        del old.doc["nodes"]["s"]["params"]["layer"]      # a document written before step 5c
        out = evaluator_pixels(old.doc, "s")
        np.testing.assert_array_equal(out[..., 0], texture()[..., 2])


class STMapTests(Scratch):
    def test_identity_map_is_identity_from_a_layer_and_from_a_wired_image(self):
        for filter_ in ("nearest", "bilinear"):
            with self.subTest(filter=filter_, source="layer"):
                _, raster = self.run_node("STMap", texture(), dict(uv_layer="uv", filter=filter_),
                                          layers=dict(uv=identity_map()))
                np.testing.assert_allclose(raster.pixels, texture(), atol=1e-6)
            with self.subTest(filter=filter_, source="wired image"):
                _, raster = self.run_node("STMap", texture(), dict(filter=filter_), uv_image=identity_map())
                np.testing.assert_allclose(raster.pixels, texture(), atol=1e-6)

    def test_a_constant_offset_in_the_map_shifts_by_that_many_pixels(self):
        shifted = identity_map()
        shifted[..., 0] += 3.0 / W        # sample 3 pixels to the right, so the picture moves left by 3
        shifted[..., 1] -= 2.0 / H        # v runs bottom to top: v down by 2/H samples 2 rows further down
        _, raster = self.run_node("STMap", texture(), dict(uv_layer="uv", filter="nearest"),
                                  layers=dict(uv=shifted))
        want = np.zeros_like(texture())
        want[:H - 2, :W - 3] = texture()[2:, 3:]
        np.testing.assert_allclose(raster.pixels, want, atol=1e-6)

    def test_out_of_range_coordinates_are_black_or_clamped(self):
        far = identity_map()
        far[..., 0] += 10.0
        _, black = self.run_node("STMap", texture(), dict(uv_layer="uv", uv_outside="black", filter="nearest"),
                                 layers=dict(uv=far))
        self.assertEqual(float(np.abs(black.pixels).max()), 0.0)
        _, clamped = self.run_node("STMap", texture(), dict(uv_layer="uv", uv_outside="clamp", filter="nearest"),
                                   layers=dict(uv=far))
        np.testing.assert_allclose(clamped.pixels, np.repeat(texture()[:, -1:], W, axis=1), atol=1e-6)

    def test_output_window_is_the_maps_and_a_missing_map_is_an_error(self):
        g, raster = self.run_node("STMap", texture(), dict(uv_layer="uv"), layers=dict(uv=identity_map()))
        self.assertEqual(raster.data, Region(0, 0, W, H))
        bare = Graph()
        self.read(bare, "src", texture())
        bare.add("n", "STMap", image="src")
        with self.assertRaisesRegex(ValueError, "connect the uv input or choose a uv_layer"):
            evaluator_pixels(bare.doc, "n")
        with self.assertRaisesRegex(ValueError, "no layer 'nope'"):
            self.run_node("STMap", texture(), dict(uv_layer="nope"), layers=dict(uv=identity_map()))


class IDistortTests(Scratch):
    def test_constant_offset_map_moves_the_picture_by_that_many_pixels(self):
        _, raster = self.run_node("IDistort", dot_image(10, 8), dict(uv_layer="motion", filter="nearest"),
                                  layers=dict(motion=constant_layer(5.0, -2.0)))
        want = dot_image(15, 6)
        np.testing.assert_allclose(raster.pixels, want, atol=1e-6)

    def test_zero_map_is_identity_and_scale_and_offset_apply_before_sampling(self):
        _, zero = self.run_node("IDistort", texture(), dict(uv_layer="motion"),
                                layers=dict(motion=constant_layer(0.0, 0.0)))
        np.testing.assert_allclose(zero.pixels, texture(), atol=1e-6)
        _, scaled = self.run_node("IDistort", dot_image(10, 8), dict(uv_layer="motion", filter="nearest", uv_scale_x=2.0,
                                                                     uv_scale_y=-1.0, uv_offset_x=1.0, uv_offset_y=0.0),
                                  layers=dict(motion=constant_layer(2.0, 3.0)))
        # d = ((2 + 1) * 2, (3 + 0) * -1) = (6, -3)
        np.testing.assert_allclose(scaled.pixels, dot_image(16, 5), atol=1e-6)

    def test_fractional_offsets_interpolate_and_a_wired_map_works(self):
        _, raster = self.run_node("IDistort", dot_image(10, 8), dict(filter="bilinear"),
                                  uv_image=constant_layer(0.5, 0.0))
        self.assertAlmostEqual(float(raster.pixels[8, 10, 0]), 0.5, places=5)
        self.assertAlmostEqual(float(raster.pixels[8, 11, 0]), 0.5, places=5)
        self.assertAlmostEqual(float(raster.pixels.sum(axis=(0, 1))[0]), 1.0, places=5)

    def test_keeps_the_source_data_window(self):
        _, raster = self.run_node("IDistort", texture(), dict(uv_layer="motion"),
                                  layers=dict(motion=constant_layer(30.0, 0.0)))
        self.assertEqual(raster.data, Region(0, 0, W, H))
        self.assertEqual(float(np.abs(raster.pixels).max()), 0.0)   # everything moved out of frame


class VectorBlurTests(Scratch):
    def test_a_uniform_field_gives_every_pixel_the_same_streak(self):
        _, raster = self.run_node("VectorBlur", dot_image(8, 8), dict(uv_layer="motion"),
                                  layers=dict(motion=constant_layer(6.0, 0.0)))
        row = raster.pixels[8, :, 0]
        np.testing.assert_allclose(row[8:15], np.full(7, 1 / 7, np.float32), atol=1e-6)   # forward: with the motion
        self.assertEqual(float(np.abs(row[:8]).max()), 0.0)
        self.assertEqual(float(np.abs(row[15:]).max()), 0.0)
        self.assertEqual(float(np.abs(raster.pixels[[0, 7, 9, 15]]).max()), 0.0)
        # one vertical column of pixels: the same field blurs a vertical line into one streak per row
        line = rgba((0, 0, 0, 0))
        line[:, 4] = 1.0
        _, streaks = self.run_node("VectorBlur", line, dict(uv_layer="motion"), layers=dict(motion=constant_layer(6.0, 0.0)))
        np.testing.assert_allclose(streaks.pixels[:, 4:11, 0], np.full((H, 7), 1 / 7, np.float32), atol=1e-6)

    def test_backward_method_streaks_the_other_way_and_scale_and_max_length_apply(self):
        layers = dict(motion=constant_layer(6.0, 0.0))
        _, back = self.run_node("VectorBlur", dot_image(14, 8), dict(uv_layer="motion", vector_method="backward"),
                                layers=layers)
        np.testing.assert_allclose(back.pixels[8, 8:15, 0], np.full(7, 1 / 7, np.float32), atol=1e-6)
        self.assertEqual(float(np.abs(back.pixels[8, 15:]).max()), 0.0)
        _, half = self.run_node("VectorBlur", dot_image(8, 8), dict(uv_layer="motion", vector_scale=0.5), layers=layers)
        np.testing.assert_allclose(half.pixels[8, 8:12, 0], np.full(4, 0.25, np.float32), atol=1e-6)
        _, capped = self.run_node("VectorBlur", dot_image(8, 8), dict(uv_layer="motion", max_length=2.0), layers=layers)
        np.testing.assert_allclose(capped.pixels[8, 8:11, 0], np.full(3, 1 / 3, np.float32), atol=1e-6)
        self.assertEqual(float(np.abs(capped.pixels[8, 11:]).max()), 0.0)

    def test_offset_shifts_the_shutter_and_zero_vectors_leave_the_picture_alone(self):
        _, shifted = self.run_node("VectorBlur", dot_image(8, 8), dict(uv_layer="motion", vector_offset=1.0),
                                   layers=dict(motion=constant_layer(4.0, 0.0)))
        np.testing.assert_allclose(shifted.pixels[8, 12:17, 0], np.full(5, 0.2, np.float32), atol=1e-6)
        _, still = self.run_node("VectorBlur", texture(), dict(uv_layer="motion"),
                                 layers=dict(motion=constant_layer(0.0, 0.0)))
        np.testing.assert_allclose(still.pixels, texture(), atol=1e-6)

    def test_alpha_weighting_keeps_the_matte_and_smears_only_colour(self):
        source = rgba((0, 0, 0, 0))
        source[8, 8] = (1.0, 0.5, 0.25, 1.0)
        layers = dict(motion=constant_layer(4.0, 0.0))
        _, plain = self.run_node("VectorBlur", source, dict(uv_layer="motion"), layers=layers)
        _, weighted = self.run_node("VectorBlur", source, dict(uv_layer="motion", vector_alpha="weighted"), layers=layers)
        np.testing.assert_allclose(plain.pixels[8, 8:13, 3], np.full(5, 0.2, np.float32), atol=1e-6)
        np.testing.assert_array_equal(weighted.pixels[..., 3], source[..., 3])       # matte untouched
        np.testing.assert_allclose(weighted.pixels[8, 8], (1.0, 0.5, 0.25, 1.0), atol=1e-6)
        self.assertEqual(float(np.abs(weighted.pixels[8, 9:, :3]).max()), 0.0)        # no alpha, no colour kept

    def test_streak_length_from_a_wired_diagonal_field(self):
        _, raster = self.run_node("VectorBlur", dot_image(6, 4), dict(vector_scale=1.0), uv_image=constant_layer(3.0, 4.0))
        red = raster.pixels[..., 0]
        self.assertAlmostEqual(float(red.sum()), 1.0, places=5)     # every sample's weight lands somewhere
        hits = np.argwhere(red > 1e-6)
        # length 5 -> 6 samples spaced one pixel apart from (6, 4) to (9, 8); bilinear puts the
        # off-grid ones across two pixels, so the box reaches at most one pixel past that line.
        self.assertEqual(hits.min(axis=0).tolist(), [4, 6])
        self.assertEqual(hits.max(axis=0).tolist(), [8, 9])
        self.assertGreater(float(red[4, 6]), 0.15)                    # the dot's own pixel: sample at t=0


class SharedBehaviourTests(Scratch):
    def _params(self, kind):
        return {"STMap": dict(uv_layer="uv", filter="nearest"), "IDistort": dict(uv_layer="uv", filter="nearest"),
                "VectorBlur": dict(uv_layer="uv")}[kind]

    def _layers(self, kind):
        return dict(uv=identity_map() if kind == "STMap" else constant_layer(4.0, 1.0))

    def test_mix_and_mask_gate_the_result_against_the_untouched_source(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                params = self._params(kind)
                _, full = self.run_node(kind, texture(), params, layers=self._layers(kind))
                _, none = self.run_node(kind, texture(), dict(params, mix=0.0), layers=self._layers(kind))
                np.testing.assert_allclose(none.pixels, texture(), atol=1e-6)
                _, half = self.run_node(kind, texture(), dict(params, mix=0.5), layers=self._layers(kind))
                np.testing.assert_allclose(half.pixels, 0.5 * full.pixels + 0.5 * texture(), atol=1e-6)
                _, hidden = self.run_node(kind, texture(), params, mask=rgba((0, 0, 0, 0)), layers=self._layers(kind))
                np.testing.assert_allclose(hidden.pixels, texture(), atol=1e-6)
                _, open_ = self.run_node(kind, texture(), params, mask=rgba(), layers=self._layers(kind))
                np.testing.assert_allclose(open_.pixels, full.pixels, atol=1e-6)

    def test_bypass_passes_the_image_input(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                g = Graph()
                self.read(g, "src", texture(), **self._layers(kind))
                g.add("n", kind, self._params(kind), image="src")
                g.bypass("n")
                self.assertEqual(bypass_slot(g.doc["nodes"]["n"]), "image")
                np.testing.assert_array_equal(evaluator_pixels(g.doc, "n"), texture())

    def test_registered_with_a_region_rule_and_kept_off_the_tile_path(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                self.assertNotIn(kind, SUPPORTED_TILED_KINDS)
                g = Graph()
                self.read(g, "src", texture(), **self._layers(kind))
                g.add("n", kind, self._params(kind), image="src")
                self.assertFalse(TileExecutor(evaluator=Evaluator()).supports_tiled(dict(g.doc, view="n"), "n"))
                self.assertEqual(SPECS[kind]["optional_inputs"], ["uv", "mask"])
        region = Region(10, 20, 30, 40)
        self.assertEqual(input_regions("STMap", dict(SPECS["STMap"]["params"]), region, 3), [None, region, region])
        self.assertEqual(input_regions("IDistort", dict(SPECS["IDistort"]["params"]), region, 3), [None, region, region])
        blur = input_regions("VectorBlur", dict(SPECS["VectorBlur"]["params"], max_length=8.0), region, 3)
        self.assertEqual(blur, [region.expand(9, 9), region, region])
        self.assertEqual(input_regions("VectorBlur", dict(SPECS["VectorBlur"]["params"], max_length=0.0), region, 3)[0], None)

    def test_proxy_tier_scales_the_pixel_measures_stored_in_data(self):
        scaled = scale_params("IDistort", dict(SPECS["IDistort"]["params"], uv_scale_x=2.0), 2)
        self.assertEqual((scaled["uv_scale_x"], scaled["uv_scale_y"]), (1.0, 0.5))
        blur = scale_params("VectorBlur", dict(SPECS["VectorBlur"]["params"]), 4)
        self.assertEqual((blur["vector_scale"], blur["max_length"]), (0.25, 25.0))
        stm = dict(SPECS["STMap"]["params"])
        self.assertEqual(scale_params("STMap", stm, 4), stm)     # normalised coordinates do not change

    def test_old_documents_without_the_new_nodes_still_evaluate(self):
        g = Graph()
        self.read(g, "src", texture())
        g.add("g", "Grade", image="src")
        np.testing.assert_allclose(evaluator_pixels(g.doc, "g"), texture(), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
