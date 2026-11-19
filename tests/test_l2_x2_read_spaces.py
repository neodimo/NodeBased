"""Lane 2 step X2 part 2: the project's default colour spaces by file type (Root's int8LUT, int16LUT, logLUT, floatLUT).

A Read at Auto takes the project's space for its file's kind: 8-bit, 16-bit, log (Cineon, DPX) and float (EXR). Each setting changes
only its own kind of file; the tile path agrees with the evaluator; a document saved before the fields existed keeps its pixels.
"""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

import tests.isolation  # noqa: F401
from nodebased import color, projectsettings, readspaces
from nodebased.color import cineon_to_linear, to_working
from nodebased.core import SCHEMA_VERSION, Dispatcher, empty_document, upgrade_document
from nodebased.imaging import Evaluator
from nodebased.media import resolve_space
from nodebased.tileexec import TileExecutor

RAMP = np.linspace(0.05, 0.95, 16 * 12 * 3, dtype=np.float32).reshape(12, 16, 3)


def write(path, pixels, fmt):
    spec = oiio.ImageSpec(pixels.shape[1], pixels.shape[0], pixels.shape[2], fmt)
    out = oiio.ImageOutput.create(str(path))
    assert out and out.open(str(path), spec), oiio.geterror()
    assert out.write_image(np.ascontiguousarray(pixels, np.float32)), out.geterror()
    assert out.close()


class ReadSpaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        root = Path(cls.dir.name)
        cls.files = {"int8": root / "a.png", "int16": root / "b.png", "float": root / "c.exr", "log": root / "d.dpx"}
        write(cls.files["int8"], RAMP, oiio.UINT8)
        write(cls.files["int16"], RAMP, oiio.UINT16)
        write(cls.files["float"], RAMP, oiio.FLOAT)
        write(cls.files["log"], RAMP, oiio.UINT16)      # 10-bit DPX data is written from the float picture

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def read(self, kind, spaces=None, colorspace="Auto", legacy=False, tiles=False):
        d = Dispatcher()
        d.execute({"op": "create", "id": "r", "type": "Read", "params": {"path": str(self.files[kind]), "colorspace": colorspace}})
        d.execute({"op": "view", "id": "r"})
        if spaces:
            d.execute({"op": "settings", "settings": {"project": spaces}})
        document = copy.deepcopy(d.document)
        if legacy:
            document["settings"]["project"] = {"log_lut": "sRGB"} if "project" not in document["settings"] else {**document["settings"]["project"], "log_lut": "sRGB"}
        if tiles:
            executor = TileExecutor(evaluator=Evaluator(), tile_edge=7)
            self.assertTrue(executor.supports_tiled(document, "r"))
            return executor.compose(document, "r", frame=1, tier=1).pixels
        return Evaluator().evaluate_raster(document, "r").pixels

    # ---- the new defaults and the settings
    def test_a_new_project_starts_with_the_guides_defaults(self):
        fields = projectsettings.project(empty_document())
        self.assertEqual([fields[n] for n in readspaces.FIELDS], ["sRGB", "sRGB", "Cineon", "Linear Rec.709"])

    def test_a_document_saved_before_keeps_reading_a_log_file_as_srgb(self):
        old = empty_document()
        old["version"] = 35
        upgraded = upgrade_document(old)
        self.assertEqual(upgraded["version"], SCHEMA_VERSION)
        fields = projectsettings.project(upgraded)
        self.assertEqual([fields[n] for n in readspaces.FIELDS], list(readspaces.LEGACY))
        self.assertEqual(readspaces.token(upgraded), "Auto")        # the string an old document always sent: its cache keys do not change
        self.assertNotEqual(readspaces.token(empty_document()), "Auto")

    def test_the_settings_reject_an_unknown_space_and_round_trip_through_the_operation(self):
        d = Dispatcher()
        with self.assertRaises(ValueError):
            d.execute({"op": "settings", "settings": {"project": {"log_lut": "Kodak 5219"}}})
        d.execute({"op": "settings", "settings": {"project": {"int8_lut": "ACEScg"}}})
        self.assertEqual(projectsettings.project(d.document)["int8_lut"], "ACEScg")
        d.execute({"op": "undo"})
        self.assertEqual(projectsettings.project(d.document)["int8_lut"], "sRGB")

    # ---- the Cineon transform
    def test_cineon_anchors(self):
        code = np.array([95, 470, 685, 1023], np.float32) / 1023.0
        linear = cineon_to_linear(code)
        self.assertAlmostEqual(float(linear[0]), 0.0, places=6)                     # the black point
        self.assertAlmostEqual(float(linear[1]), 0.18, delta=0.005)                 # code 470 is 18% grey
        self.assertAlmostEqual(float(linear[2]), 1.0, places=5)                     # reference white
        self.assertGreater(float(linear[3]), 10.0)                                  # 1023 holds highlights over reference white
        self.assertEqual(float(cineon_to_linear(np.array([0.0], np.float32))[0]), 0.0)   # below black reads as 0

    def test_a_log_file_resolves_through_the_projects_log_space(self):
        class Spec:
            format = "uint16"
            def get_string_attribute(self, name):
                return ""
        spaces = readspaces.parse(readspaces.token(empty_document()))
        self.assertEqual(resolve_space("Auto|sRGB|sRGB|Cineon|Linear Rec.709", ".dpx", Spec()), "Cineon")
        self.assertEqual(resolve_space("Auto|sRGB|sRGB|Cineon|Linear Rec.709", ".cin", Spec()), "Cineon")
        self.assertEqual(resolve_space("Auto|sRGB|ACEScg|Cineon|Linear Rec.709", ".png", Spec()), "ACEScg")
        self.assertEqual(resolve_space("Auto|ACES2065-1|sRGB|Cineon|Linear Rec.709", ".png", type("S", (Spec,), {"format": "uint8"})()), "ACES2065-1")
        self.assertEqual(resolve_space("Auto|sRGB|sRGB|Cineon|ACEScg", ".exr", Spec()), "ACEScg")
        self.assertEqual(resolve_space("Raw", ".exr", Spec()), "Raw")
        self.assertIsNone(spaces and readspaces.parse("Auto"))

    # ---- each setting changes its own kind of file, and only that
    def test_each_default_changes_only_its_own_kind_of_file(self):
        picks = {"int8": ("int8_lut", "Linear Rec.709"), "int16": ("int16_lut", "Linear Rec.709"),
                 "float": ("float_lut", "sRGB")}
        for kind, (field, space) in picks.items():
            before = self.read(kind)
            after = self.read(kind, {field: space})
            with self.subTest(kind=kind):
                self.assertFalse(np.allclose(before, after, atol=1e-4), "the setting must change its own files")
                explicit = self.read(kind, colorspace=space)                      # the Read's own colorspace: the same pixels
                np.testing.assert_allclose(after, explicit, atol=1e-6)
            for other in picks:
                if other != kind:
                    with self.subTest(kind=kind, other=other):
                        np.testing.assert_array_equal(self.read(other), self.read(other, {field: space}))

    def test_the_log_default_decodes_a_dpx_as_cineon_and_a_legacy_document_as_srgb(self):
        raw = np.asarray(oiio.ImageBuf(str(self.files["log"])).get_pixels(oiio.FLOAT), np.float32)[..., :3]
        opaque = np.ones((12, 16, 1), np.float32)
        new = self.read("log")
        np.testing.assert_allclose(new[..., :3], to_working(np.concatenate([cineon_to_linear(raw), opaque], axis=2), "Linear Rec.709", False)[..., :3], atol=1e-5)
        old = self.read("log", legacy=True)
        np.testing.assert_allclose(old, self.read("log", colorspace="sRGB", legacy=True), atol=1e-6)
        np.testing.assert_allclose(old[..., :3], to_working(np.concatenate([raw, opaque], axis=2), "sRGB", False)[..., :3], atol=1e-6)
        self.assertFalse(np.allclose(new[..., :3], old[..., :3], atol=1e-3))
        # and the project's setting moves it: log files as sRGB again
        np.testing.assert_allclose(self.read("log", {"log_lut": "sRGB"}), old, atol=1e-6)

    def test_a_read_with_its_own_colorspace_ignores_the_defaults(self):
        for kind in ("int8", "float"):
            np.testing.assert_array_equal(self.read(kind, colorspace="ACEScg"),
                                          self.read(kind, {"int8_lut": "Raw", "float_lut": "Raw"}, colorspace="ACEScg"))

    def test_the_tile_path_agrees_with_the_evaluator(self):
        for kind, spaces in (("int8", {"int8_lut": "ACEScg"}), ("float", {"float_lut": "sRGB"}), ("log", {}), ("int16", {"int16_lut": "Raw"})):
            with self.subTest(kind=kind):
                np.testing.assert_array_equal(self.read(kind, spaces, tiles=True), self.read(kind, spaces))

    def test_a_project_that_keeps_the_old_spaces_reads_as_before(self):
        d = Dispatcher()
        d.execute({"op": "settings", "settings": {"project": {"log_lut": "sRGB"}}})
        self.assertEqual(readspaces.token(d.document), "Auto")


if __name__ == "__main__":
    unittest.main()
