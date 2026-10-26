"""Fidelity step C, part 3: Write takes Nuke's knobs that change what is written: channels, the output transform
(colorspace, raw), premultiplied, views, the frame number in the file name, create directories, the range limit and the EXR
compression. Round trips through real files; the window's render uses the same code."""
import tests.isolation  # noqa: F401  (keeps the window's layout out of the real Qt settings)

import copy
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

from nodebased import writenode
from nodebased.color import INPUT_SPACES, WORKING, processor
from nodebased.core import SPECS, Dispatcher, empty_document, upgrade_document, validate
from nodebased.imaging import read_image_raster, write_png
from nodebased.media import raster_layer_arrays, write_exr
from nodebased.raster import Raster

H, W = 12, 20


def make_raster(alpha=0.5, with_layers=False):
    rng = np.random.default_rng(11)
    colour = (rng.random((H, W, 3)) * 0.7 + 0.05).astype(np.float32)
    pixels = np.concatenate([colour * alpha, np.full((H, W, 1), alpha, np.float32)], axis=2)
    layers = None
    if with_layers:
        depth = np.zeros((H, W, 4), np.float32)
        depth[..., 0] = np.linspace(1, 5, W)[None, :]
        normals = np.zeros((H, W, 4), np.float32)
        normals[..., :3] = 0.5
        layers = {"depth": Raster(depth), "normals": Raster(normals)}
    return Raster(pixels, layers=layers)


def channel_names(path):
    import OpenImageIO as oiio
    source = oiio.ImageInput.open(str(path))
    try:
        return list(source.spec().channelnames)
    finally:
        source.close()


def tag_of(path):
    import OpenImageIO as oiio
    source = oiio.ImageInput.open(str(path))
    try:
        return source.spec().get_string_attribute("oiio:ColorSpace")
    finally:
        source.close()


def png_pixels(path):
    """The 8-bit values stored in a PNG, alpha left unassociated (OIIO would otherwise premultiply on reading)."""
    import OpenImageIO as oiio
    options = oiio.ImageSpec()
    options.attribute("oiio:UnassociatedAlpha", 1)
    source = oiio.ImageInput.open(str(path), options)
    try:
        return source.read_image(0, 0, 0, source.spec().nchannels, oiio.UINT8)
    finally:
        source.close()


def space_of(path):
    """The colour space an EXR declares, in NodeBased's names (OIIO stores the tag under its interop spelling)."""
    from nodebased.media import _TAGGED_SPACES
    return _TAGGED_SPACES[tag_of(path).lower()]


def params(**knobs):
    return {**SPECS["Write"]["params"], **knobs}


class Folder(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)

    def path(self, name):
        return str(self.root / name)


class DefaultsTests(Folder):
    def test_the_specs_carry_the_defaults(self):
        for key, default in writenode.DEFAULTS.items():
            self.assertEqual(SPECS["Write"]["params"][key], default)
        self.assertFalse(writenode.active(SPECS["Write"]["params"]))

    def test_default_knobs_write_the_same_bytes_as_before(self):
        raster = make_raster(with_layers=True)
        writenode.write_raster(raster, self.path("a.exr"), "exr", "half", params())
        write_exr(self.path("b.exr"), raster.to_display(), bits="half", layers=raster_layer_arrays(raster))
        self.assertEqual(Path(self.path("a.exr")).read_bytes(), Path(self.path("b.exr")).read_bytes())
        writenode.write_raster(raster, self.path("a.png"), "png", "half", params())
        write_png(self.path("b.png"), raster.to_display())
        self.assertEqual(Path(self.path("a.png")).read_bytes(), Path(self.path("b.png")).read_bytes())


class ChannelTests(Folder):
    def test_exr_channels(self):
        raster = make_raster(with_layers=True)
        cases = {"all": ["R", "G", "B", "A", "depth.Z", "normals.X", "normals.Y", "normals.Z"],
                 "rgba": ["R", "G", "B", "A"], "rgb": ["R", "G", "B"], "alpha": ["A"], "red": ["R"],
                 "red green": ["R", "G"], "rgb depth": ["R", "G", "B", "depth.Z"], "depth": ["depth.Z"],
                 "rgba normals": ["R", "G", "B", "A", "normals.X", "normals.Y", "normals.Z"]}
        for text, expected in cases.items():
            with self.subTest(channels=text):
                target = self.path(f"c{abs(hash(text))}.exr")
                writenode.write_raster(raster, target, "exr", "float", params(channels=text))
                self.assertEqual(channel_names(target), expected)

    def test_the_channels_written_hold_the_pictures_values(self):
        raster = make_raster()
        target = self.path("rgb.exr")
        writenode.write_raster(raster, target, "exr", "float", params(channels="rgb"))
        back = read_image_raster(target, frame=1).pixels
        np.testing.assert_allclose(back[..., :3], raster.pixels[..., :3], atol=1e-5)
        self.assertTrue(np.all(back[..., 3] == 1.0))            # no alpha in the file: read back opaque

    def test_nothing_selected_is_refused(self):
        with self.assertRaises(ValueError):
            writenode.write_raster(make_raster(), self.path("none.exr"), "exr", "half", params(channels="none"))

    def test_png_channels(self):
        import OpenImageIO as oiio
        raster = make_raster(alpha=1.0)
        for text, count in (("rgba", 4), ("all", 4), ("rgb", 3)):
            with self.subTest(channels=text):
                target = self.path(f"c_{text}.png")
                writenode.write_raster(raster, target, "png", "half", params(channels=text))
                source = oiio.ImageInput.open(target)
                self.assertEqual(source.spec().nchannels, count)
                source.close()
        writenode.write_raster(raster, self.path("red.png"), "png", "half", params(channels="red"))
        image = oiio.ImageBuf(self.path("red.png")).get_pixels(oiio.UINT8)
        self.assertTrue(np.all(image[..., 1:3] == 0))
        self.assertTrue(np.any(image[..., 0] > 0))


class OutputTransformTests(Folder):
    def test_exr_in_each_space_reads_back_to_the_working_values(self):
        raster = make_raster(alpha=0.8)
        for space, tag in writenode.EXR_TAGS.items():
            with self.subTest(space=space):
                target = self.path(f"s_{tag}.exr")
                writenode.write_raster(raster, target, "exr", "float", params(colorspace=space))
                self.assertEqual(space_of(target), space)
                back = read_image_raster(target, frame=1).pixels
                np.testing.assert_allclose(back, raster.pixels, atol=2e-4)

    def test_the_file_holds_the_converted_numbers(self):
        raster = make_raster(alpha=1.0)
        target = self.path("srgb.exr")
        writenode.write_raster(raster, target, "exr", "float", params(colorspace="sRGB"))
        import OpenImageIO as oiio
        stored = oiio.ImageBuf(target).get_pixels(oiio.FLOAT)
        expected = np.ascontiguousarray(raster.pixels.copy())
        processor(WORKING, INPUT_SPACES["sRGB"]).applyRGBA(expected)
        np.testing.assert_allclose(stored, expected, atol=1e-5)
        self.assertFalse(np.allclose(stored[..., :3], raster.pixels[..., :3], atol=1e-2))

    def test_raw_writes_the_working_values_untouched_and_says_so(self):
        raster = make_raster(alpha=0.8)
        target = self.path("raw.exr")
        writenode.write_raster(raster, target, "exr", "float", params(raw=1, colorspace="sRGB"))      # raw wins over the space
        import OpenImageIO as oiio
        np.testing.assert_allclose(oiio.ImageBuf(target).get_pixels(oiio.FLOAT), raster.pixels, atol=1e-6)
        self.assertEqual(space_of(target), "Raw")

    def test_png_auto_is_the_srgb_display_view_and_raw_is_the_linear_numbers(self):
        import OpenImageIO as oiio
        raster = make_raster(alpha=1.0)
        writenode.write_raster(raster, self.path("auto.png"), "png", "half", params())
        writenode.write_raster(raster, self.path("raw.png"), "png", "half", params(raw=1))
        raw = oiio.ImageBuf(self.path("raw.png")).get_pixels(oiio.UINT8)
        np.testing.assert_array_equal(raw[..., :3], (np.clip(raster.pixels[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8))
        auto = oiio.ImageBuf(self.path("auto.png")).get_pixels(oiio.UINT8)
        self.assertGreater(float(np.abs(auto.astype(int) - raw.astype(int)).max()), 20)

    def test_png_in_linear_rec709_is_the_matrix_of_the_working_colour(self):
        import OpenImageIO as oiio
        raster = make_raster(alpha=1.0)
        writenode.write_raster(raster, self.path("lin.png"), "png", "half", params(colorspace="Linear Rec.709"))
        expected = np.ascontiguousarray(raster.pixels.copy())
        processor(WORKING, INPUT_SPACES["Linear Rec.709"]).applyRGBA(expected)
        stored = oiio.ImageBuf(self.path("lin.png")).get_pixels(oiio.UINT8)
        np.testing.assert_array_equal(stored[..., :3], (np.clip(expected[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8))

    def test_premultiplied_transforms_the_colour_as_it_is_instead_of_the_straight_colour(self):
        import OpenImageIO as oiio
        raster = make_raster(alpha=0.5)
        writenode.write_raster(raster, self.path("straight.png"), "png", "half", params(colorspace="Linear Rec.709"))
        writenode.write_raster(raster, self.path("premult.png"), "png", "half", params(colorspace="Linear Rec.709", premultiplied=1))
        straight = png_pixels(self.path("straight.png"))
        premult = png_pixels(self.path("premult.png"))
        proc = processor(WORKING, INPUT_SPACES["Linear Rec.709"])
        divided = raster.pixels.copy()
        divided[..., :3] /= raster.pixels[..., 3:4]
        as_is = np.ascontiguousarray(raster.pixels.copy())
        for array in (divided, as_is):
            proc.applyRGBA(array)
        eight = lambda a: (np.clip(a[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)
        np.testing.assert_array_equal(straight[..., :3], eight(divided))        # the transform ran on the straight colour
        np.testing.assert_array_equal(premult[..., :3], eight(as_is))           # ... or on the colour as it is
        np.testing.assert_array_equal(premult[..., 3], straight[..., 3])
        self.assertFalse(np.array_equal(straight[..., :3], premult[..., :3]))

    def test_exr_premultiplied_on_a_nonlinear_space_differs_from_the_straight_transform(self):
        raster = make_raster(alpha=0.5)
        writenode.write_raster(raster, self.path("a.exr"), "exr", "float", params(colorspace="sRGB"))
        writenode.write_raster(raster, self.path("b.exr"), "exr", "float", params(colorspace="sRGB", premultiplied=1))
        import OpenImageIO as oiio
        a = oiio.ImageBuf(self.path("a.exr")).get_pixels(oiio.FLOAT)
        b = oiio.ImageBuf(self.path("b.exr")).get_pixels(oiio.FLOAT)
        self.assertGreater(float(np.abs(a[..., :3] - b[..., :3]).max()), 1e-2)

    def test_exr_compression_is_recorded_and_readable(self):
        import OpenImageIO as oiio
        raster = make_raster()
        for name in ("zips", "piz", "none", "rle"):
            with self.subTest(compression=name):
                target = self.path(f"k_{name}.exr")
                writenode.write_raster(raster, target, "exr", "half", params(exr_compression=name))
                source = oiio.ImageInput.open(target)
                self.assertTrue(source.spec().get_string_attribute("compression").startswith(name))
                source.close()
                np.testing.assert_allclose(read_image_raster(target, frame=1).pixels, raster.pixels, atol=2e-3)


class FrameAndRangeTests(unittest.TestCase):
    def test_frames_to_render(self):
        time = {"first": 1, "last": 10, "current": 4}
        self.assertEqual(writenode.frames_to_render(params(), time, single=True), [4])
        self.assertEqual(writenode.frames_to_render(params(), time, single=False), list(range(1, 11)))
        limited = params(use_limit=1, first=3, last=5)
        self.assertEqual(writenode.frames_to_render(limited, time, single=False), [3, 4, 5])
        self.assertEqual(writenode.frames_to_render(limited, time, single=True), [4])
        self.assertEqual(writenode.frames_to_render(limited, dict(time, current=9), single=True), [])
        self.assertEqual(writenode.frames_to_render(params(use_limit=1, first=5, last=3), time, single=False), [3, 4, 5])

    def test_the_frame_number_written_into_the_name(self):
        number = writenode.file_frame_number
        self.assertEqual(number(params(), 7, 1), 7)
        self.assertEqual(number(params(frame_expr="frame + 100"), 7, 1), 107)
        self.assertEqual(number(params(frame_expr="1000"), 7, 1), 1000)
        self.assertEqual(number(params(frame_mode="start at", frame_expr="1001"), 3, 3), 1001)
        self.assertEqual(number(params(frame_mode="start at", frame_expr="1001"), 5, 3), 1003)
        self.assertEqual(number(params(frame_mode="offset", frame_expr="-2"), 7, 1), 5)
        with self.assertRaises(ValueError):
            number(params(frame_expr="frame +"), 7, 1)

    def test_only_the_main_view_exists(self):
        writenode.check_views(params())
        with self.assertRaises(ValueError):
            writenode.check_views(params(views="left"))
        document = Dispatcher(empty_document())
        document.execute({"op": "create", "id": "w", "type": "Write"})
        with self.assertRaises(ValueError):
            document.execute({"op": "set", "id": "w", "param": "views", "value": "left"})


class DocumentTests(unittest.TestCase):
    def test_a_write_saved_before_the_knobs_existed_loads_with_the_defaults(self):
        document = Dispatcher(empty_document())
        document.execute({"op": "create", "id": "w", "type": "Write", "params": {"path": "/tmp/out.exr"}})
        old = copy.deepcopy(document.document)
        for key in writenode.KEYS:
            old["nodes"]["w"]["params"].pop(key, None)
        loaded = upgrade_document(old)
        validate(loaded)
        for key, default in writenode.DEFAULTS.items():
            self.assertEqual(loaded["nodes"]["w"]["params"][key], default)
        self.assertEqual(loaded["nodes"]["w"]["params"]["path"], "/tmp/out.exr")

    def test_channels_take_a_layer_name_and_other_words_are_checked_when_it_runs(self):
        document = Dispatcher(empty_document())
        document.execute({"op": "create", "id": "w", "type": "Write"})
        for value in ("rgb", "depth", "rgb normals.X", "red green"):
            document.execute({"op": "set", "id": "w", "param": "channels", "value": value})
        with self.assertRaises(ValueError):
            document.execute({"op": "set", "id": "w", "param": "colorspace", "value": "Nonsense"})


class WindowTests(unittest.TestCase):
    """The window's Write render runs the same code on a real document."""

    def setUp(self):
        from tests.test_2d_parity_feedback_y3 import close_window, make_window
        self.window = make_window(self)
        self.addCleanup(close_window, self)
        self.dir = Path(tempfile.mkdtemp(prefix="nb-c3-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.dir, ignore_errors=True))
        self.window.set_time(first=1, last=5, current=2)

    def set(self, **knobs):
        for name, value in knobs.items():
            self.window.command({"op": "set", "id": "writer", "param": name, "value": value})

    def files(self, folder=None):
        return sorted(p.name for p in (folder or self.dir).iterdir())

    def test_frame_numbers_and_the_range_limit_decide_the_file_names(self):
        self.set(path=str(self.dir / "render.%04d.exr"), frame_expr="frame + 100", use_limit=1, first=2, last=3)
        self.window.render_write("writer", single=False)
        self.assertEqual(self.files(), ["render.0102.exr", "render.0103.exr"])

    def test_start_at_numbers_the_first_rendered_frame(self):
        self.set(path=str(self.dir / "render.%04d.exr"), frame_mode="start at", frame_expr="1001")
        self.window.render_write("writer", single=False)
        self.assertEqual(self.files(), [f"render.{n:04d}.exr" for n in range(1001, 1006)])

    def test_create_directories_makes_the_folder_or_refuses_without_it(self):
        deep = self.dir / "a" / "b"
        self.set(path=str(deep / "out.exr"), create_directories=0)
        with unittest.mock.patch("nodebased.app.QMessageBox.warning") as warning:
            self.window.render_write("writer", single=True)
        self.assertTrue(warning.called)
        self.assertFalse(deep.exists())
        self.set(create_directories=1)
        self.window.render_write("writer", single=True)
        self.assertEqual(self.files(deep), ["out.exr"])

    def test_channels_and_the_output_transform_reach_the_file(self):
        target = self.dir / "out.exr"
        self.set(path=str(target), channels="rgb", colorspace="sRGB")
        self.window.render_write("writer", single=True)
        self.assertEqual(channel_names(target), ["R", "G", "B"])
        self.assertEqual(space_of(target), "sRGB")

    def test_an_empty_range_says_so_and_writes_nothing(self):
        self.set(path=str(self.dir / "out.exr"), use_limit=1, first=4, last=5)
        with unittest.mock.patch("nodebased.app.QMessageBox.warning") as warning:
            self.window.render_write("writer", single=True)
        self.assertTrue(warning.called)
        self.assertEqual(self.files(), [])


if __name__ == "__main__":
    unittest.main()
