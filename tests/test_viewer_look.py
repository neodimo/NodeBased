"""Viewer gain, gamma, clipping warning and display choice (plan step V2), offscreen Qt."""
import json
import os
import time
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from nodebased import color
from nodebased.app import STYLE, Window
from nodebased.core import Dispatcher, VIEWER_LOOK_DEFAULT, empty_document, upgrade_document, validate, viewer_look
from nodebased.imaging import ZEBRA_HIGH, ZEBRA_LOW, to_qimage

APP = QApplication.instance() or QApplication([])
APP.setStyle("Fusion")
APP.setStyleSheet(STYLE)


def wait_until(condition, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        APP.processEvents()
        if condition():
            return True
        QTest.qWait(10)
    return False


def buffer(image):
    """The displayable 8-bit buffer of a QImage as an (h, w, 3) array."""
    width, height = image.width(), image.height()
    raw = np.frombuffer(image.constBits(), np.uint8, image.bytesPerLine() * height)
    return raw.reshape(height, image.bytesPerLine())[:, :width * 3].reshape(height, width, 3).copy()


def reload(document):
    """What opening the saved file does: upgrade, then validate."""
    loaded = upgrade_document(json.loads(json.dumps(document)))
    validate(loaded)
    return loaded


def frame_of(value, size=(16, 16)):
    frame = np.zeros((*size, 4), np.float32)
    frame[..., :3] = value
    frame[..., 3] = 1.0
    return frame


class DisplayBufferTests(unittest.TestCase):
    def test_gain_plus_one_stop_doubles_mid_grey_before_the_transform(self):
        frame = frame_of(0.1)
        base = buffer(to_qimage(frame, 0.0, view="Linear"))[0, 0, 0]
        gained = buffer(to_qimage(frame, 1.0, view="Linear"))[0, 0, 0]
        self.assertEqual((base, gained), (round(0.1 * 255), round(0.2 * 255)))
        # Under a real view the doubled value is what enters the transform.
        expect = np.clip(color.display_rgb(np.full((1, 1, 3), 0.2, np.float32), "sRGB"), 0, 1)[0, 0, 0]
        self.assertEqual(buffer(to_qimage(frame, 1.0, view="sRGB"))[0, 0, 0], round(expect * 255))

    def test_gamma_two_applies_the_reciprocal_power(self):
        image = buffer(to_qimage(frame_of(0.25), 0.0, view="Linear", look={"gamma": 2.0}))
        self.assertEqual(image[0, 0, 0], round(0.5 * 255))
        self.assertEqual(buffer(to_qimage(frame_of(0.25), 0.0, view="Linear", look={"gamma": 1.0}))[0, 0, 0],
                         round(0.25 * 255))

    def test_gain_then_gamma_order(self):
        image = buffer(to_qimage(frame_of(0.125), 1.0, view="Linear", look={"gamma": 2.0}))
        self.assertEqual(image[0, 0, 0], round(np.sqrt(0.25) * 255))

    def test_zebra_marks_over_range_and_leaves_in_range(self):
        frame = frame_of(0.5, (24, 24))
        frame[:12, :, :3] = 1.5
        plain = buffer(to_qimage(frame, 0.0, view="Linear"))
        zebra = buffer(to_qimage(frame, 0.0, view="Linear", look={"zebra": True}))
        self.assertTrue(np.array_equal(zebra[12:], plain[12:]), "in-range pixels must be untouched")
        top = zebra[:12]
        marked = (top == (255, 0, 0)).all(axis=2)
        self.assertTrue(marked.any() and not marked.all(), "stripes, not solid, on the clipped area")
        self.assertTrue(np.array_equal(zebra[0, 0], (255, 0, 0)))

    def test_zebra_marks_negative_pixels_blue(self):
        frame = frame_of(0.5, (12, 12))
        frame[:, :, 0] = -0.2
        zebra = buffer(to_qimage(frame, 0.0, view="Linear", look={"zebra": True}))
        self.assertTrue(((zebra == (0, 90, 255)).all(axis=2)).any())
        self.assertEqual((ZEBRA_HIGH, ZEBRA_LOW), (1.0, 0.0))

    def test_zebra_tests_the_gained_value(self):
        frame = frame_of(0.75)
        self.assertFalse((buffer(to_qimage(frame, 0.0, view="Linear", look={"zebra": True})) == (255, 0, 0)).all(axis=2).any())
        self.assertTrue((buffer(to_qimage(frame, 1.0, view="Linear", look={"zebra": True})) == (255, 0, 0)).all(axis=2).any())

    def test_raw_display_shows_the_linear_value(self):
        raw = buffer(to_qimage(frame_of(0.1), 0.0, view="Raw"))[0, 0, 0]
        self.assertEqual(raw, round(0.1 * 255))
        self.assertNotEqual(buffer(to_qimage(frame_of(0.1), 0.0, view="sRGB"))[0, 0, 0], raw)

    def test_display_choices_come_from_the_config(self):
        views = color.viewer_displays()
        self.assertEqual(views, tuple(color.config().getViews("sRGB - Display")))
        self.assertIn("Raw", views)
        self.assertGreater(len(views), 1)
        for view in views:
            self.assertEqual(color.display_rgb(np.full((2, 2, 3), 0.18, np.float32), view).shape, (2, 2, 3))


class LookStateTests(unittest.TestCase):
    def test_default_is_absent_and_old_documents_load_with_defaults(self):
        document = empty_document()
        self.assertNotIn("look", document["settings"]["viewer"])
        self.assertEqual(viewer_look(document), VIEWER_LOOK_DEFAULT)
        loaded = reload(document)
        self.assertEqual(viewer_look(loaded), VIEWER_LOOK_DEFAULT)

    def test_command_round_trips_and_default_removes_the_key(self):
        dispatcher = Dispatcher(empty_document())
        dispatcher.execute({"op": "viewer_look", "gain": 1.5, "gamma": 2.2, "zebra": True, "display": "Raw"})
        document = dispatcher.document
        self.assertEqual(document["settings"]["viewer"]["look"],
                         {"gain": 1.5, "gamma": 2.2, "zebra": True, "display": "Raw"})
        reloaded = reload(document)
        self.assertEqual(viewer_look(reloaded), viewer_look(document))
        dispatcher.execute({"op": "viewer_look", "gain": 0.0, "gamma": 1.0, "zebra": False,
                            "display": "Project view"})
        self.assertNotIn("look", dispatcher.document["settings"]["viewer"])
        dispatcher.execute({"op": "undo"})
        self.assertEqual(viewer_look(dispatcher.document)["gain"], 1.5)

    def test_bad_values_are_rejected(self):
        dispatcher = Dispatcher(empty_document())
        for bad in ({"gain": 99}, {"gamma": 0}, {"zebra": 1}, {"display": "Nope"}, {"gain": "1"}):
            with self.assertRaises(ValueError, msg=str(bad)):
                dispatcher.execute({"op": "viewer_look", **bad})
        self.assertNotIn("look", dispatcher.document["settings"]["viewer"])


class ViewerLookWindowTests(unittest.TestCase):
    def setUp(self):
        dispatcher = Dispatcher(empty_document())
        dispatcher.execute({"op": "create", "id": "c", "type": "Constant", "pos": [0, 0],
                            "params": {"width": 64, "height": 32, "red": 0.18, "green": 0.18,
                                       "blue": 0.18, "alpha": 1.0}})
        dispatcher.execute({"op": "view", "id": "c"})
        self.window = Window(dispatcher.document)
        self.window.resize(900, 700)
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None
                                   and self.window.viewer.format_rect is not None))

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def settle(self):
        self.assertTrue(wait_until(lambda: not self.window.busy and not len(self.window.preview_queue)
                                   and self.window.frame_generation == self.window.generation
                                   and self.window.rendered_identity == self.window.render_identity()))
        APP.processEvents()

    def readout(self):
        viewer = self.window.viewer
        rect = viewer.format_rect
        point = viewer.mapFromScene(QPointF(rect.center()))
        event = QMouseEvent(QEvent.Type.MouseMove, QPointF(point), Qt.MouseButton.NoButton,
                            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
        viewer._update_pixel_readout(event)
        return viewer.pixel_readout.label.text()

    def test_controls_store_state_and_readout_stays_scene_linear(self):
        before_text = self.readout()
        before_frame = np.array(self.window.frame)
        document_before = json.dumps(self.window.dispatcher.document["nodes"], sort_keys=True)
        self.window.exposure.setValue(2.0)
        self.window.gamma.setValue(2.0)
        self.window.zebra.setChecked(True)
        self.window.viewer_display.setCurrentText("Raw")
        self.window.set_viewer_look(display="Raw")
        self.settle()
        look = viewer_look(self.window.dispatcher.document)
        self.assertEqual((look["gain"], look["gamma"], look["zebra"], look["display"]),
                         (2.0, 2.0, True, "Raw"))
        self.assertEqual(self.readout(), before_text)
        self.assertTrue(np.array_equal(self.window.frame, before_frame))
        self.assertEqual(json.dumps(self.window.dispatcher.document["nodes"], sort_keys=True), document_before)
        self.assertEqual(self.window.effective_view(), "Raw")

    def test_reset_buttons_and_undo_follow_the_document(self):
        self.window.exposure.setValue(3.0)
        self.settle()
        self.window.command({"op": "undo"})
        self.settle()
        self.assertEqual(self.window.exposure.value(), 0.0)
        self.window.gamma.setValue(1.6)
        self.settle()
        self.window.set_viewer_look(gamma=1.0)
        self.assertEqual(self.window.gamma.value(), 1.0)
        self.assertNotIn("look", self.window.dispatcher.document["settings"]["viewer"])

    def test_picture_on_screen_follows_gain(self):
        self.window.set_viewer_look(display="Raw")
        self.settle()
        dim = self.window.viewer.viewport().grab().toImage()
        point = self.window.viewer.mapFromScene(QPointF(self.window.viewer.format_rect.center()))
        low = dim.pixelColor(point).red()
        self.window.set_viewer_look(gain=1.0)
        self.settle()
        high = self.window.viewer.viewport().grab().toImage().pixelColor(point).red()
        self.assertEqual((low, high), (round(0.18 * 255), round(0.36 * 255)))


if __name__ == "__main__":
    unittest.main()
