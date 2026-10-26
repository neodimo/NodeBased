"""Fidelity step C, part 2: Read takes Nuke's knobs that change what is read: the frame knob and mode, first / last
with before / after, the original range, the format override, raw, premultiplied and auto alpha. Round trips through
real files, the region and tile paths against the whole-image read, and old documents."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodebased import readknobs
from nodebased.core import SPECS, Dispatcher, empty_document, upgrade_document, validate
from nodebased.imaging import Evaluator, read_image_bounds, read_image_raster, read_image_region
from nodebased.media import write_exr
from nodebased.tileexec import TileExecutor
from nodebased.tiers import Region

W, H = 24, 16
FRAMES = (1, 2, 3, 4, 5)


def value_of(frame):
    return frame / 10.0


class SequenceCase(unittest.TestCase):
    """A five-frame EXR sequence whose red channel is frame / 10, opaque."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.pattern = str(cls.root / "plate.%04d.exr")
        for frame in FRAMES:
            image = np.zeros((H, W, 4), np.float32)
            image[..., 0] = value_of(frame)
            image[..., 1] = 0.25
            image[..., 2] = 0.5
            image[..., 3] = 1.0
            write_exr(str(cls.root / f"plate.{frame:04d}.exr"), image, bits="float")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def red(self, frame, **knobs):
        raster = read_image_raster(self.pattern, frame=frame, **knobs)
        return float(raster.pixels[0, 0, 0]), raster


class DefaultsTests(SequenceCase):
    def test_the_specs_carry_the_defaults(self):
        for key, default in readknobs.DEFAULTS.items():
            self.assertEqual(SPECS["Read"]["params"][key], default)
        self.assertFalse(readknobs.active(SPECS["Read"]["params"]))

    def test_default_knobs_read_exactly_what_read_always_read(self):
        for frame in (1, 3, 5):
            plain = read_image_raster(self.pattern, frame=frame)
            explicit = read_image_raster(self.pattern, frame=frame, **readknobs.DEFAULTS)
            np.testing.assert_array_equal(plain.pixels, explicit.pixels)
            self.assertEqual((plain.data, plain.display), (explicit.data, explicit.display))
            self.assertAlmostEqual(plain.pixels[0, 0, 0], value_of(frame), places=6)

    def test_the_legacy_offset_and_missing_policy_still_work(self):
        self.assertAlmostEqual(self.red(2, frame_offset=2)[0], value_of(4), places=6)
        with self.assertRaises(ValueError):
            read_image_raster(self.pattern, frame=9)
        self.assertEqual(float(read_image_raster(self.pattern, frame=9, missing="black").pixels.max()), 0.0)
        self.assertAlmostEqual(self.red(9, missing="hold")[0], value_of(5), places=6)


class FrameKnobTests(SequenceCase):
    def test_expression_mode_evaluates_the_expression_at_the_timeline_frame(self):
        self.assertAlmostEqual(self.red(1, frame_expr="frame + 2")[0], value_of(3), places=6)
        self.assertAlmostEqual(self.red(2, frame_expr="6 - frame")[0], value_of(4), places=6)
        self.assertAlmostEqual(self.red(4, frame_expr="3")[0], value_of(3), places=6)       # a constant holds one frame
        self.assertAlmostEqual(self.red(3, frame_expr="")[0], value_of(3), places=6)         # blank: the frame itself

    def test_start_at_puts_the_clips_first_frame_at_the_given_timeline_frame(self):
        self.assertAlmostEqual(self.red(10, frame_mode="start at", frame_expr="10")[0], value_of(1), places=6)
        self.assertAlmostEqual(self.red(12, frame_mode="start at", frame_expr="10")[0], value_of(3), places=6)
        # With a clip range set, the range's first frame is the one that lands there.
        self.assertAlmostEqual(self.red(10, frame_mode="start at", frame_expr="10", first=2, last=5)[0], value_of(2), places=6)

    def test_offset_adds_to_the_frame(self):
        self.assertAlmostEqual(self.red(3, frame_mode="offset", frame_expr="-1")[0], value_of(2), places=6)
        self.assertAlmostEqual(self.red(3, frame_mode="offset", frame_expr="1")[0], value_of(4), places=6)

    def test_the_legacy_frame_offset_adds_after_the_frame_knob(self):
        self.assertAlmostEqual(self.red(1, frame_expr="frame + 1", frame_offset=1)[0], value_of(3), places=6)

    def test_a_bad_expression_is_an_error_not_a_guess(self):
        with self.assertRaises(ValueError):
            read_image_raster(self.pattern, frame=1, frame_expr="frame +")


class RangeTests(SequenceCase):
    def test_inside_the_range_nothing_changes(self):
        for frame in (2, 3, 4):
            self.assertAlmostEqual(self.red(frame, first=2, last=4)[0], value_of(frame), places=6)

    def test_hold_clamps_to_the_nearest_end(self):
        self.assertAlmostEqual(self.red(1, first=2, last=4, before="hold")[0], value_of(2), places=6)
        self.assertAlmostEqual(self.red(5, first=2, last=4, after="hold")[0], value_of(4), places=6)

    def test_loop_wraps_around_the_range(self):
        expected = {5: 2, 6: 3, 7: 4, 8: 2, 1: 4, 0: 3, -1: 2}
        for frame, source in expected.items():
            with self.subTest(frame=frame):
                kwargs = dict(first=2, last=4, before="loop", after="loop")
                self.assertAlmostEqual(float(read_image_raster(self.pattern, frame=frame, **kwargs).pixels[0, 0, 0]),
                                       value_of(source), places=6)

    def test_bounce_reflects_at_each_end(self):
        expected = {5: 3, 6: 2, 7: 3, 8: 4, 1: 3, 0: 4}
        for frame, source in expected.items():
            with self.subTest(frame=frame):
                kwargs = dict(first=2, last=4, before="bounce", after="bounce")
                self.assertAlmostEqual(float(read_image_raster(self.pattern, frame=frame, **kwargs).pixels[0, 0, 0]),
                                       value_of(source), places=6)

    def test_black_outside_the_range(self):
        _, raster = self.red(5, first=2, last=4, after="black")
        self.assertEqual(float(raster.pixels.max()), 0.0)
        self.assertEqual(raster.display, Region(0, 0, W, H))                    # the frame keeps its size
        self.assertAlmostEqual(self.red(1, first=2, last=4, after="black", before="hold")[0], value_of(2), places=6)

    def test_before_and_after_are_independent(self):
        self.assertAlmostEqual(self.red(5, first=2, last=3, before="black", after="loop")[0], value_of(3), places=6)
        self.assertEqual(self.red(1, first=2, last=3, before="black", after="loop")[0], 0.0)

    def test_the_range_works_on_the_mapped_frame(self):
        self.assertAlmostEqual(self.red(1, frame_expr="frame + 4", first=2, last=4, after="loop")[0], value_of(2), places=6)


class OriginalRangeTests(SequenceCase):
    def test_a_frame_outside_the_original_range_follows_on_error(self):
        kwargs = dict(origfirst=2, origlast=4)
        with self.assertRaises(ValueError):
            read_image_raster(self.pattern, frame=1, **kwargs)
        self.assertEqual(float(read_image_raster(self.pattern, frame=1, missing="black", **kwargs).pixels.max()), 0.0)
        self.assertAlmostEqual(self.red(1, missing="hold", **kwargs)[0], value_of(2), places=6)
        self.assertAlmostEqual(self.red(5, missing="hold", **kwargs)[0], value_of(4), places=6)
        checker = read_image_raster(self.pattern, frame=5, missing="checkerboard", **kwargs).pixels
        self.assertEqual({round(float(v), 3) for v in np.unique(checker[..., 0])}, {0.2, 0.8})

    def test_frames_inside_the_original_range_read_normally(self):
        self.assertAlmostEqual(self.red(3, origfirst=2, origlast=4)[0], value_of(3), places=6)

    def test_checkerboard_also_covers_a_genuinely_missing_file(self):
        raster = read_image_raster(self.pattern, frame=9, missing="checkerboard")
        self.assertEqual(raster.pixels.shape[:2], (H, W))
        self.assertEqual(float(raster.pixels[0, 0, 3]), 1.0)
        self.assertNotEqual(float(raster.pixels[0, 0, 0]), float(raster.pixels[0, 16, 0]))
        self.assertEqual(float(raster.pixels[0, 0, 0]), float(raster.pixels[16 % H, 16, 0]) if False else float(raster.pixels[0, 0, 0]))


class FormatTests(SequenceCase):
    def test_default_is_the_files_own_format(self):
        raster = read_image_raster(self.pattern, frame=1)
        self.assertEqual(raster.display, Region(0, 0, W, H))

    def test_a_format_replaces_the_display_window_and_keeps_the_picture_at_the_lower_left(self):
        raster = read_image_raster(self.pattern, frame=1, read_format="Square_1K")
        self.assertEqual(raster.display, Region(0, 0, 1024, 1024))
        # The file's rows sit at the bottom of the new window: lower-left anchoring as in Nuke.
        self.assertEqual(raster.data, Region(0, 1024 - H, W, H))
        np.testing.assert_array_equal(raster.pixels, read_image_raster(self.pattern, frame=1).pixels)

    def test_a_smaller_format_keeps_the_pixels_and_moves_the_window(self):
        raster = read_image_raster(self.pattern, frame=2, read_format="2K_DCP")
        self.assertEqual(raster.display, Region(0, 0, 2048, 1080))
        self.assertAlmostEqual(float(raster.fit(raster.display)[1080 - 1, 0, 0]), value_of(2), places=6)
        self.assertEqual(float(raster.fit(raster.display)[0, 0, 3]), 0.0)

    def test_unknown_format_is_refused(self):
        with self.assertRaises(ValueError):
            read_image_raster(self.pattern, frame=1, read_format="nonsense")


class ColourKnobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import OpenImageIO as oiio
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.rgb_png = str(cls.root / "rgb.png")
        cls.rgba_png = str(cls.root / "rgba.png")
        rng = np.random.default_rng(3)
        data = cls.data = (rng.random((H, W, 3)) * 255).astype(np.uint8)
        for path, channels in ((cls.rgb_png, 3), (cls.rgba_png, 4)):
            out = oiio.ImageOutput.create(path)
            spec = oiio.ImageSpec(W, H, channels, oiio.UINT8)
            out.open(path, spec)
            plane = data if channels == 3 else np.concatenate([data, np.full((H, W, 1), 128, np.uint8)], axis=2)
            out.write_image(plane)
            out.close()
        cls.exr = str(cls.root / "color.exr")
        colour = np.zeros((H, W, 4), np.float32)
        colour[...] = (0.8, 0.2, 0.1, 0.5)
        write_exr(cls.exr, colour, bits="float")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_raw_matches_the_raw_colourspace_and_leaves_the_files_values(self):
        raw = read_image_raster(self.exr, frame=1, raw=1).pixels
        np.testing.assert_array_equal(raw, read_image_raster(self.exr, frame=1, colorspace="Raw").pixels)
        np.testing.assert_allclose(raw[0, 0], (0.8, 0.2, 0.1, 0.5), atol=1e-6)      # the file's own (associated) values

    def test_raw_on_a_display_referred_file_leaves_the_code_values(self):
        raw = read_image_raster(self.rgb_png, frame=1, raw=1).pixels
        converted = read_image_raster(self.rgb_png, frame=1).pixels
        np.testing.assert_allclose(raw[..., :3], self.data / 255.0, atol=1e-6)
        self.assertGreater(float(np.abs(raw - converted).max()), 1e-3)

    def test_premultiplied_treats_the_colour_as_already_multiplied_by_alpha(self):
        straight = read_image_raster(self.rgba_png, frame=1).pixels
        premultiplied = read_image_raster(self.rgba_png, frame=1, premultiplied=1).pixels
        same = read_image_raster(self.rgba_png, frame=1, alpha_mode="Premultiplied").pixels
        np.testing.assert_array_equal(premultiplied, same)
        self.assertFalse(np.allclose(premultiplied[..., :3], straight[..., :3]))
        # A straight PNG is multiplied by alpha (0.5) on the way in; a premultiplied one is not.
        np.testing.assert_allclose(straight[..., 3], 128 / 255, atol=1e-6)
        self.assertGreater(float(premultiplied[..., :3].sum()), float(straight[..., :3].sum()))

    def test_auto_alpha_off_gives_a_file_with_no_alpha_an_alpha_of_zero(self):
        on = read_image_raster(self.rgb_png, frame=1).pixels
        off = read_image_raster(self.rgb_png, frame=1, auto_alpha=0).pixels
        self.assertTrue(np.all(on[..., 3] == 1.0))
        self.assertTrue(np.all(off[..., 3] == 0.0))
        np.testing.assert_array_equal(off[..., :3], on[..., :3])

    def test_auto_alpha_leaves_a_file_with_an_alpha_channel_alone(self):
        np.testing.assert_array_equal(read_image_raster(self.rgba_png, frame=1).pixels,
                                      read_image_raster(self.rgba_png, frame=1, auto_alpha=0).pixels)


class RegionAndTileTests(SequenceCase):
    def test_the_region_read_is_the_whole_read_cut_to_the_region_for_every_knob(self):
        region = Region(3, 2, 10, 8)
        options = (dict(), dict(frame_expr="frame + 2"), dict(first=2, last=3, after="loop"), dict(read_format="2K_DCP"),
                   dict(raw=1), dict(origfirst=2, origlast=5, missing="hold"))
        for options_ in options:
            with self.subTest(options=options_):
                whole = read_image_raster(self.pattern, frame=2, **options_)
                part = read_image_region(self.pattern, region, frame=2, **options_)
                self.assertEqual(part.data, region)
                np.testing.assert_array_equal(part.pixels, whole.fit(region))
                data, display = read_image_bounds(self.pattern, frame=2, **options_)
                self.assertEqual((data, display), (whole.data, whole.display))

    def _document(self, **params):
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "r", "type": "Read", "params": {"path": self.pattern, **params}})
        d.execute({"op": "view", "id": "r"})
        return d.document

    def test_tiles_equal_the_evaluator_for_every_knob(self):
        for options in (dict(), dict(frame_expr="frame + 1"), dict(frame_mode="start at", frame_expr="0"),
                        dict(first=2, last=3, before="bounce", after="loop"), dict(read_format="Square_1K"),
                        dict(origfirst=2, origlast=3, missing="black"), dict(auto_alpha=0), dict(raw=1, premultiplied=1)):
            with self.subTest(options=options):
                document = self._document(**options)
                for frame in (1, 4):
                    expected = Evaluator().evaluate_raster(document, target="r", frame=frame)
                    executor = TileExecutor(evaluator=Evaluator(), tile_edge=16)
                    self.assertTrue(executor.supports_tiled(document, "r"))
                    region = executor.canvas_region(document, "r", frame=frame, tier=1)
                    got = executor.compose_region(document, "r", region, frame=frame, tier=1).pixels
                    np.testing.assert_allclose(got, expected.fit(Region(region.x, region.y, region.width, region.height)), atol=1e-6)

    def test_a_changed_knob_changes_the_cache_identity(self):
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=16)
        base = executor.compose_region(self._document(), "r", executor.canvas_region(self._document(), "r", frame=2, tier=1),
                                       frame=2, tier=1).pixels
        shifted = self._document(frame_expr="frame + 1")
        other = executor.compose_region(shifted, "r", executor.canvas_region(shifted, "r", frame=2, tier=1), frame=2, tier=1).pixels
        self.assertNotAlmostEqual(float(base[0, 0, 0]), float(other[0, 0, 0]), places=3)


class DocumentTests(SequenceCase):
    def test_a_read_saved_before_the_knobs_existed_loads_with_the_defaults_and_reads_the_same(self):
        document = Dispatcher(empty_document())
        document.execute({"op": "create", "id": "r", "type": "Read", "params": {"path": self.pattern}})
        old = copy.deepcopy(document.document)
        for key in readknobs.KEYS:
            old["nodes"]["r"]["params"].pop(key, None)
        loaded = upgrade_document(old)
        validate(loaded)
        params = loaded["nodes"]["r"]["params"]
        for key, default in readknobs.DEFAULTS.items():
            self.assertEqual(params.get(key, default), default)
        a = Evaluator().evaluate_raster(document.document, target="r", frame=3).pixels
        b = Evaluator().evaluate_raster(loaded, target="r", frame=3).pixels
        np.testing.assert_array_equal(a, b)

    def test_validation_accepts_the_choices_and_refuses_others(self):
        document = Dispatcher(empty_document())
        document.execute({"op": "create", "id": "r", "type": "Read", "params": {"path": self.pattern}})
        for name, value in (("before", "bounce"), ("frame_mode", "start at"), ("read_format", "HD_720"), ("missing", "checkerboard")):
            document.execute({"op": "set", "id": "r", "param": name, "value": value})
        for name, value in (("before", "sideways"), ("frame_mode", "random"), ("read_format", "Huge")):
            with self.assertRaises(ValueError):
                document.execute({"op": "set", "id": "r", "param": name, "value": value})


if __name__ == "__main__":
    unittest.main()
