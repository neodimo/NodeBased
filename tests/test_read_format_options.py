"""Read's format-specific panel tabs and the EXR options the local decoder can honour."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
import unittest
import uuid
import copy
from pathlib import Path

import numpy as np
import OpenImageIO as oiio
from PySide6.QtWidgets import QApplication, QCheckBox, QLineEdit, QTabWidget, QWidget

from nodebased.app import Window
from nodebased.core import Dispatcher, SCHEMA_VERSION, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.media import read_media_raster
from nodebased.metadata import header_entry
from nodebased import readknobs

APP = QApplication.instance() or QApplication([])


def write_fixture(path, pixels, *, x=0, y=0, full_x=0, full_y=0, full=None, attributes=()):
    h, w, c = pixels.shape
    spec = oiio.ImageSpec(w, h, c, oiio.FLOAT)
    spec.channelnames = ["R", "G", "B", "A"][:c]
    spec.x, spec.y = x, y
    spec.full_x, spec.full_y = full_x, full_y
    spec.full_width, spec.full_height = full or (w, h)
    for name, value in attributes:
        spec.attribute(name, value)
    output = oiio.ImageOutput.create(str(path))
    assert output and output.open(str(path), spec), oiio.geterror()
    assert output.write_image(np.ascontiguousarray(pixels, np.float32)), output.geterror()
    assert output.close()


class ReadFormatOptionsTests(unittest.TestCase):
    def test_exr_offset_moves_picture_by_negative_display_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "negative.exr"
            source = np.zeros((2, 2, 4), np.float32)
            source[..., 0] = 0.75
            source[..., 3] = 1
            write_fixture(path, source, x=-1, full_x=-2, full=(4, 2))
            before = read_media_raster(path, edge_pixels="black").to_display()
            after = read_media_raster(path, edge_pixels="black",
                                      offset_negative_display_window=True).to_display()
            self.assertEqual(before.shape, after.shape)
            np.testing.assert_array_equal(np.flatnonzero(before[0, :, 0]), [1, 2])
            np.testing.assert_array_equal(np.flatnonzero(after[0, :, 0]), [3])

    def test_edge_pixels_only_repeats_edges_named_by_the_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "crop.exr"
            source = np.zeros((2, 2, 4), np.float32)
            source[..., :3] = 0.5
            source[..., 3] = 1
            write_fixture(path, source, full=(4, 4))
            plate = read_media_raster(path, edge_pixels="plate detect").to_display()
            edge = read_media_raster(path, edge_pixels="edge detect").to_display()
            repeat = read_media_raster(path, edge_pixels="repeat").to_display()
            black = read_media_raster(path, edge_pixels="black").to_display()
            self.assertEqual(float(plate[3, 3, 0]), 0.0)
            self.assertEqual(float(edge[0, 0, 0]), 0.5)
            self.assertEqual(float(edge[3, 3, 0]), 0.0)
            self.assertEqual(float(repeat[3, 3, 0]), 0.5)
            np.testing.assert_array_equal(black, np.pad(source, ((0, 2), (0, 2), (0, 0))))

    def test_noprefix_changes_exr_metadata_key_only(self):
        self.assertEqual(header_entry("artistName", "artist", noprefix=False), ("exr/artistName", "artist"))
        self.assertEqual(header_entry("artistName", "artist", noprefix=True), ("artistName", "artist"))
        self.assertEqual(header_entry("layer/owner", "artist", noprefix=True), ("layer/owner", "artist"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.exr"
            write_fixture(path, np.ones((1, 1, 4), np.float32), attributes=(("artistName", "artist"),))
            prefixed = read_media_raster(path).meta
            plain = read_media_raster(path, noprefix=True).meta
            self.assertEqual(prefixed["exr/artistName"], "artist")
            self.assertEqual(plain["artistName"], "artist")

    def test_graph_read_honours_edge_option_and_v29_load_keeps_legacy_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "crop.exr"
            source = np.zeros((2, 2, 4), np.float32)
            source[..., :3] = 0.5
            source[..., 3] = 1
            write_fixture(path, source, full=(4, 4))
            dispatcher = Dispatcher()
            dispatcher.execute({"op": "create", "id": "read", "type": "Read", "params": {"path": str(path)}})
            evaluator = Evaluator()
            plate = evaluator.evaluate_raster(dispatcher.document, "read").to_display()
            dispatcher.execute({"op": "set", "id": "read", "param": "edge_pixels", "value": "repeat"})
            repeated = evaluator.evaluate_raster(dispatcher.document, "read").to_display()
            self.assertNotEqual(float(plate[3, 3, 0]), float(repeated[3, 3, 0]))

            saved = copy.deepcopy(dispatcher.document)
            saved["version"] = SCHEMA_VERSION - 1
            for name in readknobs.FORMAT_DEFAULTS:
                saved["nodes"]["read"]["params"].pop(name, None)
            restored = upgrade_document(saved)
            validate(restored)
            self.assertEqual(restored["version"], SCHEMA_VERSION)
            self.assertEqual(restored["nodes"]["read"]["params"]["edge_pixels"], "black")
            self.assertFalse(restored["nodes"]["read"]["params"]["offset_negative_display_window"])
            legacy = Evaluator().evaluate_raster(restored, "read").to_display()
            np.testing.assert_array_equal(legacy, plate)

    def test_panel_shows_only_the_path_format_tab(self):
        window = Window(agent_name="read-format-" + uuid.uuid4().hex)
        window.resize(900, 900)
        window.show()
        window.command({"op": "create", "id": "read-format", "type": "Read", "pos": [0, 0]})
        window.command({"op": "set", "id": "read-format", "param": "path", "value": "plate.exr"})
        window.inspect("read-format")
        APP.processEvents()
        panel = window.properties.widget()
        tabs = panel.findChild(QTabWidget, "node-tabs")
        self.assertIn("EXR Options", [tabs.tabText(i) for i in range(tabs.count())])
        self.assertNotIn("MOV Options", [tabs.tabText(i) for i in range(tabs.count())])
        self.assertIsNotNone(panel.findChild(QWidget, "edge_pixels-choice"))
        exr_tab = next(tabs.widget(i) for i in range(tabs.count()) if tabs.tabText(i) == "EXR Options")
        self.assertIsNotNone(exr_tab.findChild(QWidget, "edge_pixels-choice"))
        window.command({"op": "set", "id": "read-format", "param": "path", "value": "plate.png"})
        window.inspect("read-format")
        APP.processEvents()
        panel = window.properties.widget()
        tabs = panel.findChild(QTabWidget, "node-tabs")
        self.assertFalse(any(tabs.tabText(i) in {"EXR Options", "MOV Options", "PSD Options"}
                             for i in range(tabs.count())))
        for suffix, expected in (("mov", "MOV Options"), ("psd", "PSD Options")):
            window.command({"op": "set", "id": "read-format", "param": "path", "value": f"plate.{suffix}"})
            window.inspect("read-format")
            APP.processEvents()
            tabs = window.properties.widget().findChild(QTabWidget, "node-tabs")
            self.assertIn(expected, [tabs.tabText(i) for i in range(tabs.count())])
            if suffix == "mov":
                self.assertFalse(window.properties.widget().findChild(QCheckBox, "mov64_first_track_only-switch").isEnabled())
                self.assertTrue(window.properties.widget().findChild(QLineEdit, "mov64_ycbcr_matrix-field").isReadOnly())
            else:
                self.assertTrue(window.properties.widget().findChild(QLineEdit, "breakout-field").isReadOnly())
        window.saved_document = window.dispatcher.document
        window.close(); window.deleteLater(); APP.processEvents()


if __name__ == "__main__":
    unittest.main()
