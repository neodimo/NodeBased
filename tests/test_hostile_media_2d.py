"""M1 gate, part 2: hostile media (docs/M1_GATE.md).

Truncated, zero-byte, wrong-extension and header-corrupt files across EXR/PNG/JPEG/TIFF; a
sequence with a missing frame and one with a differently sized frame; a 1x1 image; a file whose
header claims 100,000 x 100,000; and files holding NaN/Inf pixels. Every case must end in a
named, catchable error or a documented fallback (never a crash), inside 5 seconds, and must not
grow resident memory by more than the 1 GB budget even when the header lies about the image size.
"""
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator, read_image_raster
from nodebased.media import read_media, write_exr
from nodebased.procmem import peak_rss_kb

TIME_BUDGET_SECONDS = 5.0
MEMORY_BUDGET_KB = 1024 * 1024  # 1 GB; ru_maxrss is reported in KiB on Linux.
EXTENSIONS = (".exr", ".png", ".jpg", ".tiff")


def write_still(path, width=4, height=4):
    """A tiny, valid image in whatever format `path`'s extension names."""
    pixels = np.zeros((height, width, 4), np.float32)
    pixels[..., 3] = 1.0
    if Path(path).suffix.lower() == ".exr":
        write_exr(path, pixels)
        return
    spec = oiio.ImageSpec(width, height, 4, oiio.UINT8)
    output = oiio.ImageOutput.create(str(path))
    assert output and output.open(str(path), spec), oiio.geterror()
    assert output.write_image((pixels * 255).astype(np.uint8))
    assert output.close()


def write_exr_raw(path, pixels, channelnames, x=0, y=0, full=None, attributes=()):
    """Author an EXR directly, bypassing `write_exr`'s own validation, so the reader is tested
    against a file whose header can disagree with its pixel data."""
    height, width = pixels.shape[:2]
    spec = oiio.ImageSpec(width, height, pixels.shape[2], oiio.FLOAT)
    spec.channelnames = list(channelnames)
    spec.x, spec.y = x, y
    spec.full_x, spec.full_y = 0, 0
    spec.full_width, spec.full_height = full or (width, height)
    for name, value in attributes:
        spec.attribute(name, value)
    output = oiio.ImageOutput.create(str(path))
    assert output and output.open(str(path), spec), oiio.geterror()
    assert output.write_image(np.ascontiguousarray(pixels, np.float32)), output.geterror()
    assert output.close()


class Budgeted(unittest.TestCase):
    """Runs a read under a wall-clock and memory budget, requiring it to fail cleanly."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def assert_catchable(self, path, action=None, pattern=None):
        """`action()` (default: `read_media(str(path))`) must raise, fast and without blowing
        up memory. Returns the raised exception."""
        action = action or (lambda: read_media(str(path)))
        before = peak_rss_kb()
        start = time.monotonic()
        with self.assertRaises(Exception) as ctx:
            action()
        elapsed = time.monotonic() - start
        after = peak_rss_kb()
        self.assertLess(elapsed, TIME_BUDGET_SECONDS,
                        f"{path}: took {elapsed:.2f}s, over the {TIME_BUDGET_SECONDS}s budget")
        self.assertLess(after - before, MEMORY_BUDGET_KB,
                        f"{path}: RSS grew {(after - before) / 1024:.1f} MB, "
                        f"over the {MEMORY_BUDGET_KB / 1024:.0f} MB budget")
        self.assertNotIsInstance(ctx.exception, (SystemError, MemoryError),
                                 f"{path}: raised {type(ctx.exception).__name__}, not a named, "
                                 "catchable ValueError")
        if pattern:
            self.assertRegex(str(ctx.exception), pattern)
        return ctx.exception


class TruncatedFileTests(Budgeted):
    def test_truncated_files_raise_a_catchable_error(self):
        for ext in EXTENSIONS:
            with self.subTest(ext=ext):
                path = self.dir / f"truncated{ext}"
                write_still(path)
                data = path.read_bytes()
                path.write_bytes(data[: max(1, len(data) // 4)])
                self.assert_catchable(path)


class ZeroByteFileTests(Budgeted):
    def test_zero_byte_files_raise_a_catchable_error(self):
        for ext in EXTENSIONS:
            with self.subTest(ext=ext):
                path = self.dir / f"empty{ext}"
                path.write_bytes(b"")
                self.assert_catchable(path)


class WrongExtensionFileTests(Budgeted):
    def test_non_image_bytes_under_an_image_extension_raise_a_catchable_error(self):
        # Content that is not any known image format at all, saved under each of the four
        # supported extensions: a wrong-extension file is only "hostile" if OIIO cannot also
        # recover it by sniffing the content (see the docstring test below for when it can).
        garbage = b"this is a plain text file pretending to be an image\n" * 4
        for ext in EXTENSIONS:
            with self.subTest(ext=ext):
                path = self.dir / f"notreally{ext}"
                path.write_bytes(garbage)
                self.assert_catchable(path)

    def test_real_bytes_under_a_mismatched_extension_are_a_documented_oiio_fallback(self):
        # OpenImageIO's `ImageInput.open` is documented to fall back from the extension's own
        # plugin to content (magic-number) sniffing when the named plugin cannot parse the
        # file, so real EXR bytes saved as .png can come back as a successful, correct read
        # rather than an error -- the "documented fallback" half of this gate's contract, not
        # a crash and not silent corruption: the decoded pixels are still the source image's.
        write_still(self.dir / "real.exr")
        source = read_media(str(self.dir / "real.exr"))
        mismatched = self.dir / "real_exr_bytes.png"
        mismatched.write_bytes((self.dir / "real.exr").read_bytes())
        start = time.monotonic()
        try:
            recovered = read_media(str(mismatched))
        except Exception:
            return  # also acceptable: a clean, catchable rejection instead of the fallback
        self.assertLess(time.monotonic() - start, TIME_BUDGET_SECONDS)
        np.testing.assert_array_equal(recovered, source)

    def test_unsupported_extension_names_the_supported_set(self):
        path = self.dir / "clip.mov"
        write_still(self.dir / "real.exr")
        path.write_bytes((self.dir / "real.exr").read_bytes())
        self.assert_catchable(path, pattern="EXR, PNG, JPEG and TIFF")


class HeaderCorruptFileTests(Budgeted):
    def test_corrupted_header_bytes_raise_a_catchable_error(self):
        for ext in EXTENSIONS:
            with self.subTest(ext=ext):
                path = self.dir / f"corrupt{ext}"
                write_still(path)
                data = bytearray(path.read_bytes())
                # Flip the first sixteen bytes (every one of these formats' magic/header lives
                # there); leave the rest of the file, including its length, intact.
                for i in range(min(16, len(data))):
                    data[i] ^= 0xFF
                path.write_bytes(bytes(data))
                self.assert_catchable(path)


class ExtremeDimensionTests(Budgeted):
    def test_1x1_image_reads_without_crashing(self):
        for ext in EXTENSIONS:
            with self.subTest(ext=ext):
                path = self.dir / f"tiny{ext}"
                write_still(path, width=1, height=1)
                result = read_media(str(path))
                self.assertEqual(result.shape, (1, 1, 4))

    def test_header_claiming_100000_by_100000_is_rejected_before_decode(self):
        # The actual pixel data stays tiny; only the EXR display-window header claims a huge
        # format. A correct reader must reject this from the header alone, never attempt to
        # allocate a 100000x100000 buffer.
        path = self.dir / "huge_header.exr"
        write_exr_raw(path, np.zeros((4, 4, 4), np.float32), ["R", "G", "B", "A"],
                      full=(100_000, 100_000))
        self.assert_catchable(path, pattern="between 1 and 8192")


class MalformedSequenceTests(Budgeted):
    def test_missing_frame_raises_a_named_error_by_default(self):
        pattern = str(self.dir / "seq.%04d.exr")
        write_exr(self.dir / "seq.0001.exr", np.zeros((2, 2, 4), np.float32))
        write_exr(self.dir / "seq.0003.exr", np.zeros((2, 2, 4), np.float32))
        # Frame 2 is missing on disk. Default policy ("error") must name the missing frame,
        # not crash the graph.
        self.assert_catchable(
            pattern, action=lambda: read_image_raster(pattern, frame=2), pattern="Missing frame 2")

    def test_missing_frame_hold_and_black_are_documented_fallbacks(self):
        pattern = str(self.dir / "seq2.%04d.exr")
        frame1 = np.full((2, 2, 4), 0.5, np.float32)
        write_exr(self.dir / "seq2.0001.exr", frame1)
        write_exr(self.dir / "seq2.0003.exr", np.full((2, 2, 4), 0.75, np.float32))
        held = read_image_raster(pattern, missing="hold", frame=2)
        np.testing.assert_array_equal(held.to_display(), frame1)  # nearest earlier frame
        blacked = read_image_raster(pattern, missing="black", frame=2)
        self.assertTrue(np.all(blacked.to_display() == 0))

    def test_a_frame_of_a_different_size_reads_on_its_own_without_crashing(self):
        pattern = str(self.dir / "seq3.%04d.exr")
        write_exr(self.dir / "seq3.0001.exr", np.zeros((6, 8, 4), np.float32))
        write_exr(self.dir / "seq3.0002.exr", np.zeros((9, 12, 4), np.float32))
        first = read_image_raster(pattern, frame=1)
        second = read_image_raster(pattern, frame=2)
        self.assertEqual(first.to_display().shape, (6, 8, 4))
        self.assertEqual(second.to_display().shape, (9, 12, 4))

    def test_merging_two_differently_sized_sequence_frames_raises_a_named_error(self):
        d = Dispatcher()
        path1 = self.dir / "a.exr"
        path2 = self.dir / "b.exr"
        write_exr(path1, np.zeros((6, 8, 4), np.float32))
        write_exr(path2, np.zeros((9, 12, 4), np.float32))
        d.execute(dict(op="create", id="a", type="Read", params=dict(path=str(path1))))
        d.execute(dict(op="create", id="b", type="Read", params=dict(path=str(path2))))
        d.execute(dict(op="create", id="node", type="Merge", params={}))
        d.execute(dict(op="connect", id="node", input="A", source="a"))
        d.execute(dict(op="connect", id="node", input="B", source="b"))
        self.assert_catchable(
            "a merge of mismatched frames",
            action=lambda: Evaluator().evaluate_raster(dict(d.document, view="node")),
            pattern="matching formats")


class NanInfPixelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_nan_and_inf_pixels_survive_a_read_without_crashing(self):
        path = self.dir / "extreme.exr"
        pixels = np.array([[[np.nan, np.inf, -np.inf, 1.0], [0.0, 0.0, 0.0, 1.0]]], np.float32)
        write_exr(path, pixels, bits="float")
        start = time.monotonic()
        result = read_media(str(path))
        self.assertLess(time.monotonic() - start, TIME_BUDGET_SECONDS)
        self.assertTrue(np.isnan(result[0, 0, 0]))
        self.assertTrue(np.isposinf(result[0, 0, 1]))
        self.assertTrue(np.isneginf(result[0, 0, 2]))

    def test_nan_pixels_flow_through_a_grade_node_without_crashing(self):
        # NaN/Inf source pixels must not crash or hang a node that processes them; they are not
        # required to keep their exact numeric identity (an unwired mask/mix blend computes
        # `source * (1 - gate)`, and `inf * 0.0` is `nan` even at `gate == 1.0`, a documented,
        # pre-existing limitation of that blend, not new hostile-media breakage).
        path = self.dir / "extreme_graph.exr"
        pixels = np.array([[[np.nan, np.inf, 0.2, 1.0]]], np.float32)
        write_exr(path, pixels, bits="float")
        d = Dispatcher()
        d.execute(dict(op="create", id="src", type="Read", params=dict(path=str(path))))
        d.execute(dict(op="create", id="node", type="Grade", params=dict(multiply=2.0)))
        d.execute(dict(op="connect", id="node", input="image", source="src"))
        start = time.monotonic()
        raster = Evaluator().evaluate_raster(dict(d.document, view="node"))
        self.assertLess(time.monotonic() - start, TIME_BUDGET_SECONDS)
        self.assertEqual(raster.to_display().shape, (1, 1, 4))


if __name__ == "__main__":
    unittest.main()
