"""Fidelity step M: Write delivers TIFF, JPEG and DPX files and takes Nuke's EXR options (autocrop, metadata, interleave,
first part, full layer names, truncated channel names and the DWAA / DWAB compression level).

Every case writes a real file and reads it back with OpenImageIO; the options are checked in the file, not in the knob."""
import tests.isolation  # noqa: F401  (keeps the window's layout out of the real Qt settings)

import copy
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodebased import core, writeformats, writenode
from nodebased.core import SPECS, Dispatcher, empty_document, upgrade_document, validate
from nodebased.imaging import read_image_raster, write_png
from nodebased.media import raster_layer_arrays, write_exr
from nodebased.raster import Raster
from nodebased.tiers import Region

H, W = 24, 40


def params(**knobs):
    return {**SPECS["Write"]["params"], **knobs}


def make_raster(alpha=0.5, with_layers=False, meta=None):
    """A smooth premultiplied picture (so JPEG's loss stays small) with a colour ramp that stays inside the sRGB gamut (so no
    colour is clipped on its way out), optionally with depth and normals."""
    ramp = np.linspace(0.25, 0.55, W, dtype=np.float32)[None, :, None]
    down = np.linspace(0.25, 0.55, H, dtype=np.float32)[:, None, None]
    colour = np.concatenate((np.broadcast_to(ramp, (H, W, 1)), np.broadcast_to(down, (H, W, 1)),
                             np.full((H, W, 1), 0.3, np.float32)), axis=2)
    pixels = np.concatenate((colour * alpha, np.full((H, W, 1), alpha, np.float32)), axis=2)
    layers = None
    if with_layers:
        depth = np.zeros((H, W, 4), np.float32)
        depth[..., 0] = np.linspace(1, 5, W)[None, :]
        normals = np.zeros((H, W, 4), np.float32)
        normals[..., :3] = 0.5
        layers = {"depth": Raster(depth), "normals": Raster(normals)}
    return Raster(pixels, layers=layers, meta=meta)


def read_float(path, subimage=0, unassociated=True):
    import OpenImageIO as oiio
    options = oiio.ImageSpec()
    if unassociated:
        options.attribute("oiio:UnassociatedAlpha", 1)
    source = oiio.ImageInput.open(str(path), options)
    try:
        source.seek_subimage(subimage, 0)
        return source.read_image(subimage, 0, 0, source.spec().nchannels, oiio.FLOAT)
    finally:
        source.close()


def spec_of(path, subimage=0):
    import OpenImageIO as oiio
    source = oiio.ImageInput.open(str(path))
    try:
        source.seek_subimage(subimage, 0)
        return source.spec()
    finally:
        source.close()


def attribute(path, name, subimage=0):
    spec = spec_of(path, subimage)
    for entry in spec.extra_attribs:
        if entry.name == name:
            return entry.value
    return None


def parts_of(path):
    import OpenImageIO as oiio
    source = oiio.ImageInput.open(str(path))
    found = []
    try:
        index = 0
        while source.seek_subimage(index, 0):
            spec = source.spec()
            found.append((spec.get_string_attribute("name"), list(spec.channelnames)))
            index += 1
    finally:
        source.close()
    return found


def jpeg_sampling(path):
    """The (horizontal, vertical) sampling factor of each component in a JPEG's frame header."""
    data = Path(path).read_bytes()
    match = re.search(rb"\xff[\xc0\xc2]", data)
    position = match.end()
    count = data[position + 7]
    return [((data[position + 8 + 3 * i + 1] >> 4), (data[position + 8 + 3 * i + 1] & 15)) for i in range(count)]


def dpx_ten_bit_samples(path, channels):
    """The 10-bit code values of a filled-method-A DPX, decoded from its bytes: HxWxchannels. (OpenImageIO's own reader mis-reads
    some widths of a 10-bit RGBA file, so the alpha test decodes the file itself; each line is padded to whole 32-bit words.)"""
    import struct
    data = Path(path).read_bytes()
    big = data[:4] == b"SDPX"
    order = ">" if big else "<"
    offset = struct.unpack(order + "I", data[4:8])[0]
    width, height = struct.unpack(order + "II", data[772:780])
    per_line = -(-width * channels // 3)
    words = np.frombuffer(data[offset:offset + 4 * per_line * height], dtype=order + "u4").reshape(height, per_line)
    samples = np.stack(((words >> 22) & 1023, (words >> 12) & 1023, (words >> 2) & 1023), axis=2).reshape(height, -1)
    return samples[:, :width * channels].reshape(height, width, channels)


class Folder(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)

    def path(self, name):
        return str(self.root / name)


def sRGB_expected(raster, channels=3):
    """What a straight, sRGB-encoded, clamped picture holds: the same colour pipeline the 8-bit PNG goes through."""
    colour, alpha = writenode.straight_output(raster.to_display(), "sRGB", False)
    return np.concatenate((np.clip(colour, 0, 1), alpha), axis=2)[..., :channels]


class FormatChoiceTests(Folder):
    def test_the_file_type_choices_and_the_extensions_auto_reads(self):
        self.assertEqual(SPECS["Write"]["params"]["file_type"], "Auto")
        self.assertEqual(core.CHOICES["file_type"], ["Auto", "exr", "png", "tiff", "jpeg", "dpx"])
        for name, expected in (("a.tif", "tiff"), ("a.TIFF", "tiff"), ("a.jpg", "jpeg"), ("a.jpeg", "jpeg"), ("a.dpx", "dpx"),
                               ("a.exr", "exr"), ("a.png", "png")):
            path, file_type, _bits = writenode.target_of(params(path=self.path(name)))
            self.assertEqual((Path(path).name, file_type), (name, expected))
        with self.assertRaises(ValueError):
            writenode.target_of(params(path=self.path("a.mov")))

    def test_an_explicit_choice_makes_the_extension_agree(self):
        path, file_type, _bits = writenode.target_of(params(path=self.path("a.exr"), file_type="tiff"))
        self.assertEqual((Path(path).suffix, file_type), (".tif", "tiff"))
        path, _t, _b = writenode.target_of(params(path=self.path("a.tiff"), file_type="tiff"))
        self.assertEqual(Path(path).suffix, ".tiff")            # a name the format answers to is left as it is
        path, _t, _b = writenode.target_of(params(path=self.path("a.png"), file_type="jpeg"))
        self.assertEqual(Path(path).suffix, ".jpg")
        path, _t, _b = writenode.target_of(params(path=self.path("a.png"), file_type="dpx"))
        self.assertEqual(Path(path).suffix, ".dpx")

    def test_every_new_knob_has_a_nuke_default(self):
        defaults = SPECS["Write"]["params"]
        self.assertEqual((defaults["tiff_datatype"], defaults["tiff_compression"]), ("8 bit", "Deflate"))
        self.assertEqual((defaults["jpeg_quality"], defaults["jpeg_sub_sampling"]), (0.75, "4:1:1"))
        self.assertEqual((defaults["dpx_datatype"], defaults["dpx_fill"], defaults["dpx_big_endian"], defaults["dpx_transfer"]),
                         ("10 bit", 0, 1, "(auto detect)"))
        self.assertEqual(defaults["exr_dw_compression_level"], 45.0)
        self.assertEqual(core.CHOICES["tiff_datatype"], ["8 bit", "16 bit", "32 bit float"])
        self.assertEqual(core.CHOICES["tiff_compression"], ["none", "PackBits", "LZW", "Deflate"])
        self.assertEqual(core.CHOICES["jpeg_sub_sampling"], ["4:1:1", "4:2:2", "4:4:4"])
        self.assertEqual(core.CHOICES["dpx_datatype"], ["8 bit", "10 bit", "12 bit", "16 bit"])

    def test_properties_show_only_the_knobs_of_the_format_being_written(self):
        exr_only = {"bit_depth", "exr_compression", *writenode.EXR_DEFAULTS}
        for path, file_type, shown in (("a.exr", "Auto", "exr"), ("a.png", "Auto", "png"), ("a.tif", "Auto", "tiff"),
                                       ("", "tiff", "tiff"), ("a.exr", "jpeg", "jpeg"), ("a.dpx", "Auto", "dpx"),
                                       ("", "Auto", "exr"), ("a.xyz", "Auto", "exr")):
            with self.subTest(path=path, file_type=file_type):
                hidden = writenode.hidden_params(params(path=path, file_type=file_type))
                self.assertEqual(writenode.panel_format(params(path=path, file_type=file_type)), shown)
                self.assertEqual(exr_only <= hidden, shown != "exr")
                for name, keys in writeformats.PARAMS.items():
                    self.assertEqual(set(keys) <= hidden, name != shown, name)
                    self.assertFalse(set(keys) & hidden and name == shown)


class TiffTests(Folder):
    def test_eight_bit_tiff_matches_the_png_of_the_same_picture(self):
        raster = make_raster()
        tif, png = self.path("a.tif"), self.path("a.png")
        writenode.write_raster(raster, tif, "tiff", "half", params())
        write_png(png, raster.to_display())
        spec = spec_of(tif)
        self.assertEqual((spec.width, spec.height, spec.nchannels, str(spec.format)), (W, H, 4, "uint8"))
        np.testing.assert_allclose(read_float(tif), read_float(png), atol=1 / 255 + 1e-6)

    def test_sixteen_bit_tiff_holds_the_value_to_one_code(self):
        raster = make_raster()
        target = self.path("b.tif")
        writenode.write_raster(raster, target, "tiff", "half", params(tiff_datatype="16 bit"))
        spec = spec_of(target)
        self.assertEqual((spec.nchannels, str(spec.format)), (4, "uint16"))
        np.testing.assert_allclose(read_float(target), sRGB_expected(raster, 4), atol=1 / 65535 + 1e-6)

    def test_float_tiff_keeps_working_values_unclamped(self):
        raster = make_raster(alpha=1.0)
        raster.pixels[0, 0, :3] = (4.5, -0.25, 0.5)
        target = self.path("c.tif")
        writenode.write_raster(raster, target, "tiff", "half", params(tiff_datatype="32 bit float"))
        spec = spec_of(target)
        self.assertEqual((spec.nchannels, str(spec.format)), (4, "float"))
        back = read_float(target)
        np.testing.assert_allclose(back[..., :3], raster.pixels[..., :3], atol=1e-6)      # the working space itself
        self.assertAlmostEqual(float(back[0, 0, 0]), 4.5, places=5)
        self.assertAlmostEqual(float(back[0, 0, 1]), -0.25, places=5)

    def test_integer_tiff_clamps_before_it_quantises(self):
        raster = make_raster(alpha=1.0)
        raster.pixels[0, 0, :3] = (4.5, -0.25, 0.5)
        for datatype, top in (("8 bit", 255), ("16 bit", 65535)):
            with self.subTest(datatype=datatype):
                target = self.path(f"d{top}.tif")
                writenode.write_raster(raster, target, "tiff", "half", params(tiff_datatype=datatype))
                import OpenImageIO as oiio
                source = oiio.ImageInput.open(target)
                raw = source.read_image(0, 0, 0, 4, oiio.UINT16 if top == 65535 else oiio.UINT8)
                source.close()
                self.assertEqual(int(raw[0, 0, 0]), top)             # overbright clamps to full, it does not wrap
                self.assertEqual(int(raw[0, 0, 1]), 0)               # negative clamps to zero

    def test_compression_reaches_the_file_header(self):
        raster = make_raster()
        found = {}
        for choice, expected in (("none", "none"), ("PackBits", "packbits"), ("LZW", "lzw"), ("Deflate", "zip")):
            target = self.path(f"e_{expected}.tif")
            writenode.write_raster(raster, target, "tiff", "half", params(tiff_compression=choice))
            self.assertEqual(spec_of(target).get_string_attribute("compression"), expected)
            found[choice] = Path(target).stat().st_size
            np.testing.assert_allclose(read_float(target), read_float(self.path("e_none.tif")), atol=1e-6)
        self.assertGreater(found["none"], found["Deflate"])

    def test_output_transform_and_channels_apply(self):
        raster = make_raster()
        target = self.path("f.tif")
        writenode.write_raster(raster, target, "tiff", "half", params(tiff_datatype="16 bit", colorspace="ACEScg", channels="rgb"))
        self.assertEqual(spec_of(target).nchannels, 3)
        straight = raster.pixels[..., :3] / raster.pixels[..., 3:4]
        np.testing.assert_allclose(read_float(target), np.clip(straight, 0, 1), atol=2e-5)
        writenode.write_raster(raster, target, "tiff", "half", params(tiff_datatype="16 bit", raw=1, channels="red alpha"))
        back = read_float(target)
        self.assertEqual(back.shape[2], 4)
        self.assertTrue(np.all(back[..., 1:3] == 0))                    # unselected colour reads zero, as PNG does
        np.testing.assert_allclose(back[..., 3], 0.5, atol=2e-5)

    def test_the_file_reads_back_through_read(self):
        raster = make_raster(alpha=1.0)
        target = self.path("g.tif")
        writenode.write_raster(raster, target, "tiff", "half", params(tiff_datatype="16 bit"))
        back = read_image_raster(target, frame=1)
        np.testing.assert_allclose(back.pixels[..., :3], raster.pixels[..., :3], atol=2e-3)


class JpegTests(Folder):
    def test_jpeg_has_the_size_three_channels_and_the_picture(self):
        raster = make_raster(alpha=1.0)
        target = self.path("a.jpg")
        writenode.write_raster(raster, target, "jpeg", "half", params(jpeg_quality=0.95))
        spec = spec_of(target)
        self.assertEqual((spec.width, spec.height, spec.nchannels, str(spec.format)), (W, H, 3, "uint8"))
        np.testing.assert_allclose(read_float(target), sRGB_expected(raster, 3), atol=12 / 255)

    def test_alpha_is_dropped_and_colour_lands_over_black(self):
        raster = make_raster(alpha=0.5)
        target = self.path("b.jpg")
        writenode.write_raster(raster, target, "jpeg", "half", params(jpeg_quality=1.0, jpeg_sub_sampling="4:4:4"))
        self.assertEqual(spec_of(target).nchannels, 3)
        colour, alpha = writenode.straight_output(raster.to_display(), "sRGB", False)
        np.testing.assert_allclose(read_float(target), np.clip(colour * alpha, 0, 1), atol=6 / 255)

    def test_quality_changes_the_file(self):
        rng = np.random.default_rng(2)
        raster = make_raster(alpha=1.0)
        raster.pixels[..., :3] = rng.random((H, W, 3)).astype(np.float32)         # detail the encoder has to spend bits on
        sizes, errors = {}, {}
        reference = sRGB_expected(raster, 3)
        for quality in (0.1, 0.5, 0.95):
            target = self.path(f"q{int(quality * 100)}.jpg")
            writenode.write_raster(raster, target, "jpeg", "half", params(jpeg_quality=quality, jpeg_sub_sampling="4:4:4"))
            sizes[quality] = Path(target).stat().st_size
            errors[quality] = float(np.abs(read_float(target) - reference).mean())
        self.assertLess(sizes[0.1], sizes[0.5])
        self.assertLess(sizes[0.5], sizes[0.95])
        self.assertGreater(errors[0.1], errors[0.95])

    def test_sub_sampling_reaches_the_frame_header(self):
        raster = make_raster(alpha=1.0)
        seen = {}
        for choice in ("4:1:1", "4:2:2", "4:4:4"):
            target = self.path(f"s{choice[2]}{choice[4]}.jpg")
            writenode.write_raster(raster, target, "jpeg", "half", params(jpeg_sub_sampling=choice))
            seen[choice] = jpeg_sampling(target)
        self.assertEqual(seen["4:4:4"][0], (1, 1))
        self.assertEqual(seen["4:2:2"][0], (2, 1))
        self.assertEqual(seen["4:1:1"][0], (4, 1))
        self.assertEqual(len({tuple(value) for value in seen.values()}), 3)

    def test_the_file_reads_back_through_read(self):
        raster = make_raster(alpha=1.0)
        target = self.path("c.jpeg")
        writenode.write_raster(raster, target, "jpeg", "half", params(jpeg_quality=1.0, jpeg_sub_sampling="4:4:4"))
        back = read_image_raster(target, frame=1)
        np.testing.assert_allclose(back.pixels[..., :3], raster.pixels[..., :3], atol=0.03)


class DpxTests(Folder):
    def test_bit_depths_reach_the_file_and_values_hold_to_one_code(self):
        raster = make_raster(alpha=1.0)
        expected = sRGB_expected(raster, 4)
        for datatype, bits in (("8 bit", 8), ("10 bit", 10), ("12 bit", 12), ("16 bit", 16)):
            with self.subTest(datatype=datatype):
                target = self.path(f"a{bits}.dpx")
                writenode.write_raster(raster, target, "dpx", "half", params(dpx_datatype=datatype, channels="rgb"))
                spec = spec_of(target)
                self.assertEqual((spec.width, spec.height, spec.nchannels), (W, H, 3))
                self.assertEqual(spec.get_int_attribute("oiio:BitsPerSample"), bits)
                np.testing.assert_allclose(read_float(target), expected[..., :3], atol=1 / (2 ** bits - 1) + 1e-6)

    def test_default_is_ten_bit_with_alpha_for_all_channels(self):
        raster = make_raster()
        target = self.path("b.dpx")
        writenode.write_raster(raster, target, "dpx", "half", params())
        spec = spec_of(target)
        self.assertEqual((spec.width, spec.height, spec.nchannels), (W, H, 4))
        self.assertEqual(spec.get_int_attribute("oiio:BitsPerSample"), 10)
        self.assertEqual(spec.get_string_attribute("dpx:ImageDescriptor"), "RGBA")
        samples = dpx_ten_bit_samples(target, 4)
        self.assertLessEqual(int(samples.max()), 1023)
        np.testing.assert_allclose(samples / 1023.0, sRGB_expected(raster, 4), atol=1.5 / 1023)

    def test_big_endian_reaches_the_header(self):
        raster = make_raster(alpha=1.0)
        big, little = self.path("big.dpx"), self.path("little.dpx")
        writenode.write_raster(raster, big, "dpx", "half", params(dpx_big_endian=1))
        writenode.write_raster(raster, little, "dpx", "half", params(dpx_big_endian=0))
        self.assertEqual(Path(big).read_bytes()[:4], b"SDPX")
        self.assertEqual(Path(little).read_bytes()[:4], b"XPDS")
        np.testing.assert_allclose(read_float(big), read_float(little), atol=0)             # same picture, other byte order

    def test_fill_chooses_the_packing_and_keeps_the_picture(self):
        raster = make_raster(alpha=1.0)
        filled, packed = self.path("filled.dpx"), self.path("packed.dpx")
        writenode.write_raster(raster, filled, "dpx", "half", params(dpx_fill=0, channels="rgb"))
        writenode.write_raster(raster, packed, "dpx", "half", params(dpx_fill=1, channels="rgb"))
        self.assertEqual(spec_of(filled).get_string_attribute("dpx:Packing"), "Filled, method A")
        self.assertEqual(spec_of(packed).get_string_attribute("dpx:Packing"), "Packed")
        np.testing.assert_allclose(read_float(filled), read_float(packed), atol=0)

    def test_transfer_header_follows_the_output_transform_or_the_choice(self):
        raster = make_raster(alpha=1.0)
        cases = (("sRGB", "(auto detect)", "Unspecified video"), ("ACEScg", "(auto detect)", "Linear"),
                 ("Linear Rec.709", "(auto detect)", "Linear"), ("Raw", "(auto detect)", "User defined"),
                 ("sRGB", "Printing density", "Printing density"), ("sRGB", "ADX", "ADX"), ("sRGB", "Z depth linear", "Z depth linear"))
        for space, choice, expected in cases:
            with self.subTest(space=space, choice=choice):
                target = self.path(f"t{abs(hash((space, choice)))}.dpx")
                writenode.write_raster(raster, target, "dpx", "half", params(colorspace="Auto" if space == "Raw" else space,
                                                                             raw=int(space == "Raw"), dpx_transfer=choice))
                self.assertEqual(spec_of(target).get_string_attribute("dpx:Transfer"), expected)

    def test_every_transfer_choice_is_a_name_the_writer_keeps(self):
        raster = make_raster(alpha=1.0)
        for choice in writeformats.DPX_TRANSFERS[1:]:
            with self.subTest(choice=choice):
                target = self.path(f"n{abs(hash(choice))}.dpx")
                writenode.write_raster(raster, target, "dpx", "half", params(dpx_transfer=choice))
                self.assertEqual(spec_of(target).get_string_attribute("dpx:Transfer"), choice)


class ExrAutocropTests(Folder):
    def box_raster(self):
        raster = make_raster(alpha=0.0)
        raster.pixels[...] = 0.0
        raster.pixels[5:15, 8:30] = (0.2, 0.3, 0.4, 0.5)
        raster.pixels[10, 12] = (0.1, 0.1, 0.1, 1.0)
        return raster

    def test_data_window_is_the_box_of_non_zero_alpha(self):
        raster = self.box_raster()
        full, cropped = self.path("full.exr"), self.path("crop.exr")
        writenode.write_raster(raster, full, "exr", "float", params())
        writenode.write_raster(raster, cropped, "exr", "float", params(exr_autocrop=1))
        whole, box = spec_of(full), spec_of(cropped)
        self.assertEqual((whole.x, whole.y, whole.width, whole.height), (0, 0, W, H))
        self.assertEqual((box.x, box.y, box.width, box.height), (8, 5, 22, 10))
        self.assertEqual((box.full_x, box.full_y, box.full_width, box.full_height), (0, 0, W, H))      # the frame stays
        np.testing.assert_allclose(read_float(cropped, unassociated=False), raster.pixels[5:15, 8:30], atol=1e-6)

    def test_read_puts_the_cropped_picture_back_in_the_frame(self):
        raster = self.box_raster()
        cropped = self.path("crop.exr")
        writenode.write_raster(raster, cropped, "exr", "float", params(exr_autocrop=1))
        back = read_image_raster(cropped, frame=1)
        self.assertEqual((back.display.width, back.display.height), (W, H))
        self.assertEqual((back.data.x, back.data.y, back.data.width, back.data.height), (8, 5, 22, 10))
        np.testing.assert_allclose(back.fit(back.display), raster.fit(raster.display), atol=1e-5)

    def test_layers_are_cropped_with_the_beauty(self):
        raster = self.box_raster()
        raster.layers = {"depth": Raster(np.ones((H, W, 4), np.float32))}
        cropped = self.path("layers.exr")
        writenode.write_raster(raster, cropped, "exr", "float", params(exr_autocrop=1))
        spec = spec_of(cropped)
        self.assertEqual((spec.width, spec.height), (22, 10))
        self.assertIn("depth.Z", spec.channelnames)

    def test_an_image_with_no_alpha_above_zero_is_written_whole(self):
        raster = make_raster(alpha=0.0)
        raster.pixels[...] = 0.0
        target = self.path("empty.exr")
        writenode.write_raster(raster, target, "exr", "half", params(exr_autocrop=1))
        self.assertEqual((spec_of(target).width, spec_of(target).height), (W, H))

    def test_autocrop_off_is_the_old_file(self):
        raster = self.box_raster()
        a, b = self.path("a.exr"), self.path("b.exr")
        writenode.write_raster(raster, a, "exr", "half", params())
        write_exr(b, raster.to_display(), bits="half", layers=raster_layer_arrays(raster))
        same_file(self, a, b)


def same_file(case, first, second):
    """Two EXRs hold the same pixels, channel names and header attributes (the DateTime stamp aside)."""
    one, two = spec_of(first), spec_of(second)
    case.assertEqual(list(one.channelnames), list(two.channelnames))
    names = lambda spec: sorted((e.name, str(e.value)) for e in spec.extra_attribs if e.name != "DateTime")  # noqa: E731
    case.assertEqual(names(one), names(two))
    np.testing.assert_array_equal(read_float(first, unassociated=False), read_float(second, unassociated=False))


META = {"input/filename": "/shots/plate.0001.exr", "input/width": "40", "exr/lookName": "studio", "show/shot": "010",
        "timecode": "01:00:00:05", "cryptomatte/abc123/name": "crypto_object"}


class ExrMetadataTests(Folder):
    def written(self, choice):
        raster = make_raster(meta=dict(META))
        target = self.path(f"m{abs(hash(choice))}.exr")
        writenode.write_raster(raster, target, "exr", "half", params(exr_metadata=choice))
        spec = spec_of(target)
        return {e.name: e.value for e in spec.extra_attribs}

    def test_choices_change_the_attributes_written(self):
        none = self.written("no metadata")
        default = self.written("default metadata")
        with_exr = self.written("default metadata and exr/*")
        except_input = self.written("all metadata except input/*")
        everything = self.written("all metadata")
        self.assertNotIn("lookName", none)
        self.assertNotIn("show/shot", none)
        self.assertNotIn("smpte:TimeCode", none)
        self.assertEqual(none.get("cryptomatte/abc123/name"), "crypto_object")      # the layers' manifest is not metadata to drop
        self.assertEqual(default["show/shot"], "010")
        self.assertIn("smpte:TimeCode", default)
        self.assertNotIn("lookName", default)                                          # exr/* is a choice of its own
        self.assertEqual(with_exr["lookName"], "studio")
        self.assertEqual(except_input["lookName"], "studio")
        self.assertNotIn("input/filename", except_input)
        self.assertEqual(everything["input/filename"], "/shots/plate.0001.exr")
        self.assertEqual(everything["input/width"], "40")
        self.assertEqual(everything["lookName"], "studio")

    def test_the_default_writes_what_write_always_wrote(self):
        raster = make_raster(meta=dict(META))
        a, b = self.path("a.exr"), self.path("b.exr")
        writenode.write_raster(raster, a, "exr", "half", params())
        write_exr(b, raster.to_display(), bits="half", layers=raster_layer_arrays(raster), metadata=raster.meta)
        same_file(self, a, b)
        self.assertEqual(SPECS["Write"]["params"]["exr_metadata"], "all metadata except input/*")

    def test_an_unknown_choice_is_refused(self):
        with self.assertRaises(ValueError):
            writenode.write_raster(make_raster(meta=dict(META)), self.path("x.exr"), "exr", "half",
                                   params(exr_metadata="everything", exr_autocrop=1))


class ExrInterleaveTests(Folder):
    def test_one_view_makes_channels_and_layers_a_single_part(self):
        raster = make_raster(with_layers=True)
        default, single = self.path("default.exr"), self.path("single.exr")
        writenode.write_raster(raster, default, "exr", "half", params())
        writenode.write_raster(raster, single, "exr", "half", params(exr_interleave="channels and layers"))
        self.assertEqual(len(parts_of(single)), 1)
        same_file(self, default, single)

    def test_channels_writes_one_part_per_layer_sorted_alphabetically(self):
        raster = make_raster(with_layers=True)
        target = self.path("parts.exr")
        writenode.write_raster(raster, target, "exr", "half", params(exr_interleave="channels"))
        self.assertEqual(parts_of(target), [("depth", ["Z"]), ("normals", ["X", "Y", "Z"]), ("rgba", ["R", "G", "B", "A"])])
        np.testing.assert_allclose(read_float(target, 2, unassociated=False), raster.to_display(), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(read_float(target, 0, unassociated=False)[..., 0],
                                   raster_layer_arrays(raster)["depth"][..., 0], rtol=1e-3, atol=1e-3)

    def test_first_part_leads_and_the_rest_stay_alphabetical(self):
        raster = make_raster(with_layers=True)
        target = self.path("first.exr")
        writenode.write_raster(raster, target, "exr", "half", params(exr_interleave="channels", exr_first_part="rgba"))
        self.assertEqual([name for name, _channels in parts_of(target)], ["rgba", "depth", "normals"])
        writenode.write_raster(raster, target, "exr", "half", params(exr_interleave="channels", exr_first_part="normals"))
        self.assertEqual([name for name, _channels in parts_of(target)], ["normals", "depth", "rgba"])

    def test_first_part_is_ignored_unless_channels_are_all_and_interleave_is_channels(self):
        raster = make_raster(with_layers=True)
        target = self.path("ignored.exr")
        writenode.write_raster(raster, target, "exr", "half", params(exr_first_part="rgba"))
        self.assertEqual(len(parts_of(target)), 1)
        writenode.write_raster(raster, target, "exr", "half",
                               params(exr_interleave="channels", exr_first_part="normals", channels="rgba depth normals"))
        self.assertEqual([name for name, _channels in parts_of(target)], ["depth", "normals", "rgba"])

    def test_full_layer_names_put_the_layer_in_every_channel_name(self):
        raster = make_raster(with_layers=True)
        legacy, full = self.path("legacy.exr"), self.path("full.exr")
        writenode.write_raster(raster, legacy, "exr", "half", params(exr_interleave="channels"))
        writenode.write_raster(raster, full, "exr", "half", params(exr_interleave="channels", exr_write_full_layer_names=1))
        self.assertEqual(parts_of(legacy)[0], ("depth", ["Z"]))
        self.assertEqual(parts_of(full)[0], ("depth", ["depth.Z"]))
        self.assertEqual(parts_of(full)[1], ("normals", ["normals.X", "normals.Y", "normals.Z"]))

    def test_standard_layer_name_format_changes_no_byte_with_one_view(self):
        raster = make_raster(with_layers=True)
        a, b = self.path("a.exr"), self.path("b.exr")
        writenode.write_raster(raster, a, "exr", "half", params())
        writenode.write_raster(raster, b, "exr", "half", params(exr_standard_layer_name_format=1))
        same_file(self, a, b)


class ExrNamesAndCompressionTests(Folder):
    def long_layer(self):
        name = "a" * 40
        raster = make_raster()
        raster.layers = {name: Raster(np.full((H, W, 4), 0.25, np.float32))}
        return raster, name

    def test_truncate_channel_names_cuts_to_thirty_one_characters(self):
        raster, name = self.long_layer()
        loose, cut = self.path("loose.exr"), self.path("cut.exr")
        writenode.write_raster(raster, loose, "exr", "half", params())
        writenode.write_raster(raster, cut, "exr", "half", params(exr_truncate_channel_names=1))
        self.assertTrue(any(len(channel) > 31 for channel in spec_of(loose).channelnames))
        self.assertTrue(all(len(channel) <= 31 for channel in spec_of(cut).channelnames))
        self.assertIn("R", spec_of(cut).channelnames)                         # short names are left alone
        self.assertTrue(spec_of(cut).channelnames[-1].endswith(".B"))        # the channel letter survives

    def test_names_that_collide_after_truncating_are_refused(self):
        raster = make_raster()
        raster.layers = {"b" * 40: Raster(np.zeros((H, W, 4), np.float32)), "b" * 41: Raster(np.zeros((H, W, 4), np.float32))}
        with self.assertRaises(ValueError):
            writenode.write_raster(raster, self.path("clash.exr"), "exr", "half", params(exr_truncate_channel_names=1))

    def test_dwa_level_reaches_the_file_and_the_size(self):
        rng = np.random.default_rng(5)
        raster = make_raster(alpha=1.0)
        raster.pixels[..., :3] = rng.random((H, W, 3)).astype(np.float32)
        sizes = {}
        for compression in ("dwaa", "dwab"):
            for level in (10.0, 45.0, 200.0):
                target = self.path(f"{compression}{int(level)}.exr")
                writenode.write_raster(raster, target, "exr", "half",
                                       params(exr_compression=compression, exr_dw_compression_level=level))
                self.assertEqual(spec_of(target).get_string_attribute("compression"), compression)
                self.assertEqual(attribute(target, "openexr:dwaCompressionLevel"), level)
                sizes[(compression, level)] = Path(target).stat().st_size
        for compression in ("dwaa", "dwab"):
            self.assertGreater(sizes[(compression, 10.0)], sizes[(compression, 200.0)])

    def test_the_level_means_nothing_for_other_compressions(self):
        raster = make_raster()
        a, b = self.path("a.exr"), self.path("b.exr")
        writenode.write_raster(raster, a, "exr", "half", params(exr_compression="piz"))
        writenode.write_raster(raster, b, "exr", "half", params(exr_compression="piz", exr_dw_compression_level=99.0))
        self.assertIsNone(attribute(b, "openexr:dwaCompressionLevel"))
        same_file(self, a, b)

    def test_a_dwa_file_reads_back_close_to_the_picture(self):
        raster = make_raster(alpha=1.0)
        target = self.path("dwa.exr")
        writenode.write_raster(raster, target, "exr", "half", params(exr_compression="dwaa", exr_dw_compression_level=10.0))
        np.testing.assert_allclose(read_float(target, unassociated=False), raster.pixels, atol=0.02)


class OldDocumentTests(Folder):
    def test_a_write_saved_before_these_knobs_loads_with_the_defaults_and_writes_the_same_file(self):
        document = Dispatcher(empty_document())
        document.execute({"op": "create", "id": "w", "type": "Write", "params": {"path": self.path("out.exr")}})
        old = copy.deepcopy(document.document)
        for key in (*writeformats.DEFAULTS, *writenode.EXR_DEFAULTS):
            old["nodes"]["w"]["params"].pop(key, None)
        loaded = upgrade_document(old)
        validate(loaded)
        self.assertEqual(loaded["schema"] if "schema" in loaded else None, old.get("schema"))      # no schema bump
        for key, default in {**writeformats.DEFAULTS, **writenode.EXR_DEFAULTS}.items():
            self.assertEqual(loaded["nodes"]["w"]["params"][key], default)
        raster = make_raster(with_layers=True, meta=dict(META))
        node = loaded["nodes"]["w"]["params"]
        writenode.write_raster(raster, node["path"], "exr", "half", node)
        reference = self.path("reference.exr")
        write_exr(reference, raster.to_display(), bits="half", layers=raster_layer_arrays(raster), metadata=raster.meta)
        same_file(self, node["path"], reference)

    def test_old_png_output_is_the_same_bytes(self):
        raster = make_raster(with_layers=True)
        a, b = self.path("a.png"), self.path("b.png")
        writenode.write_raster(raster, a, "png", "half", params(exr_autocrop=1, tiff_compression="LZW", dpx_fill=1))
        write_png(b, raster.to_display())
        self.assertEqual(Path(a).read_bytes(), Path(b).read_bytes())        # another format's knobs do not touch a PNG
        self.assertFalse(writenode.active(SPECS["Write"]["params"]))


class WindowTests(unittest.TestCase):
    """The window's Write: Properties shows the knobs of the format being written, and Render writes the new formats."""

    def setUp(self):
        from tests.test_2d_parity_feedback_y3 import close_window, make_window
        self.window = make_window(self)
        self.addCleanup(close_window, self)
        self.dir = Path(tempfile.mkdtemp(prefix="nb-m-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.dir, ignore_errors=True))
        self.window.set_time(first=1, last=3, current=2)

    def set(self, **knobs):
        for name, value in knobs.items():
            self.window.command({"op": "set", "id": "writer", "param": name, "value": value})

    def labels(self):
        from PySide6.QtWidgets import QFormLayout, QLabel
        panel = self.window.build_node_panel("writer")
        found = set()
        for form in panel.findChildren(QFormLayout):
            for row in range(form.rowCount()):
                item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
                if item is not None and isinstance(item.widget(), QLabel):
                    found.add(item.widget().text())
        return found

    EXR = {"Compression level", "Autocrop", "Metadata", "Interleave", "First part", "Standard layer name format",
           "Write full layer names", "Truncate channel names", "EXR compression"}
    TIFF = {"TIFF data type", "TIFF compression"}
    JPEG = {"JPEG quality", "JPEG sub-sampling"}
    DPX = {"DPX data type", "DPX fill", "DPX big endian", "DPX transfer"}

    def test_properties_list_only_the_active_formats_knobs(self):
        everything = self.EXR | self.TIFF | self.JPEG | self.DPX
        for path, expected in (("a.exr", self.EXR), ("a.tif", self.TIFF), ("a.jpg", self.JPEG), ("a.dpx", self.DPX),
                               ("a.png", set())):
            with self.subTest(path=path):
                self.set(path=str(self.dir / path), file_type="Auto")
                shown = self.labels()
                self.assertEqual(shown & everything, expected)
        self.set(path=str(self.dir / "a.exr"), file_type="tiff")
        self.assertEqual(self.labels() & everything, self.TIFF)

    def test_the_float_knobs_are_sliders_when_their_format_is_chosen(self):
        from nodebased.app import FloatSliderControl
        from tests.test_desktop import APP
        for file_type, name in (("jpeg", "jpeg_quality"), ("exr", "exr_dw_compression_level")):
            with self.subTest(file_type=file_type):
                self.set(file_type=file_type)
                panel = self.window.build_node_panel("writer")
                self.window.set_properties_widget(panel)
                APP.processEvents()
                pairs = [pair for pair in panel.findChildren(FloatSliderControl) if pair.spin.objectName() == f"{name}-field"]
                self.assertEqual(len(pairs), 1)
                self.assertFalse(pairs[0].isHidden())
        self.set(file_type="tiff")
        self.assertEqual([pair for pair in self.window.build_node_panel("writer").findChildren(FloatSliderControl)
                          if pair.spin.objectName() == "jpeg_quality-field"], [])

    def test_render_writes_each_new_format_and_names_it_in_the_status_bar(self):
        for name, knobs, check in (
                ("t.%04d.tif", {"tiff_datatype": "16 bit", "tiff_compression": "LZW"},
                 lambda path: (spec_of(path).nchannels, str(spec_of(path).format), spec_of(path).get_string_attribute("compression"))),
                ("j.%04d.jpg", {"jpeg_quality": 0.5}, lambda path: (spec_of(path).nchannels, str(spec_of(path).format), None)),
                ("d.%04d.dpx", {"dpx_big_endian": 0, "dpx_datatype": "12 bit"},
                 lambda path: (spec_of(path).nchannels, spec_of(path).get_int_attribute("oiio:BitsPerSample"),
                               Path(path).read_bytes()[:4]))):
            with self.subTest(name=name):
                self.set(path=str(self.dir / name), **knobs)
                self.window.render_write("writer", single=False)
                files = sorted(p.name for p in self.dir.iterdir() if p.name.startswith(name[0] + "."))
                self.assertEqual(files, [name % n for n in (1, 2, 3)])
                result = check(str(self.dir / files[0]))
                expected = {"t": (4, "uint16", "lzw"), "j": (3, "uint8", None), "d": (4, 12, b"XPDS")}[name[0]]
                self.assertEqual(result, expected)
                self.assertIn("Rendered 3 frames", self.window.statusBar().currentMessage())
        self.assertIn(" · ", self.window.statusBar().currentMessage())

    def test_render_writes_an_autocropped_exr(self):
        target = self.dir / "crop.exr"
        self.set(path=str(target), exr_autocrop=1, exr_interleave="channels")
        self.window.render_write("writer", single=True)
        self.assertEqual(len(parts_of(target)), 1)
        self.assertEqual(spec_of(target).nchannels, 4)


if __name__ == "__main__":
    unittest.main()
