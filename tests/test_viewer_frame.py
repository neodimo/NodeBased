"""Viewer region of interest, proxy toggle and format masks (plan step V3), offscreen Qt."""
import json
import os
import time
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from nodebased import viewframe
from nodebased.app import STYLE, Window
from nodebased.core import (Dispatcher, VIEWER_MASKS_DEFAULT, VIEWER_ROI_DEFAULT, empty_document,
                            upgrade_document, validate, viewer_masks, viewer_proxy, viewer_roi)
from nodebased.tiers import PIXEL_UNIT_PARAMS, PROXY_TIERS, scale_params

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


def reload(document):
    loaded = upgrade_document(json.loads(json.dumps(document)))
    validate(loaded)
    return loaded


def blur_document(width=1024, height=768, radius=16.0):
    dispatcher = Dispatcher(empty_document())
    dispatcher.execute({"op": "create", "id": "c", "type": "Constant", "pos": [0, 0],
                        "params": {"width": width, "height": height, "red": 0.5, "green": 0.25,
                                   "blue": 0.125, "alpha": 1.0}})
    dispatcher.execute({"op": "create", "id": "b", "type": "Blur", "pos": [200, 0],
                        "params": {"radius": radius}})
    dispatcher.execute({"op": "connect", "id": "b", "input": "image", "source": "c"})
    dispatcher.execute({"op": "view", "id": "b"})
    return dispatcher.document


class GeometryTests(unittest.TestCase):
    def test_roi_pixels_round_outward_and_stay_inside(self):
        self.assertEqual(viewframe.roi_pixels([0, 0, 0.5, 0.5], 0, 0, 1024, 768), (0, 0, 512, 384))
        self.assertEqual(viewframe.roi_pixels([0.1, 0.1, 0.3, 0.3], 0, 0, 100, 100), (10, 10, 30, 30))
        self.assertEqual(viewframe.roi_pixels([0.104, 0.5, 0.306, 0.9], 0, 0, 100, 100), (10, 50, 31, 90))
        # A box thinner than a pixel is still one pixel, and an origin offset is honoured.
        self.assertEqual(viewframe.roi_pixels([0.5, 0.5, 0.501, 0.501], 10, 20, 100, 100), (60, 70, 61, 71))

    def test_235_mask_on_1080p_darkens_the_expected_rows(self):
        bars = viewframe.mask_bars("2.35", 1920, 1080)
        kept = 1920 / 2.35
        top = round((1080 - kept) / 2)
        bottom = round((1080 + kept) / 2)
        self.assertEqual(bars, [(0, 0, 1920, top), (0, bottom, 1920, 1080)])
        self.assertEqual((top, 1080 - bottom), (131, 131))
        self.assertEqual(bottom - top, 818)
        self.assertEqual(viewframe.mask_bars("format", 1920, 1080), [])
        # 1.78 is a hair narrower than 16:9, so it trims one row top and bottom.
        self.assertEqual(viewframe.mask_bars("1.78", 1920, 1080), [(0, 0, 1920, 1), (0, 1079, 1920, 1080)])

    def test_narrower_mask_bars_the_sides(self):
        bars = viewframe.mask_bars("1.33", 1920, 1080)
        kept_w = 1080 * 1.33
        left = round((1920 - kept_w) / 2)
        self.assertEqual(bars, [(0, 0, left, 1080), (round((1920 + kept_w) / 2), 0, 1920, 1080)])


class StateTests(unittest.TestCase):
    def test_old_documents_load_with_defaults(self):
        document = empty_document()
        for key in ("roi", "proxy", "masks"):
            self.assertNotIn(key, document["settings"]["viewer"])
        loaded = reload(document)
        self.assertEqual(viewer_roi(loaded), VIEWER_ROI_DEFAULT)
        self.assertEqual(viewer_proxy(loaded), 1)
        self.assertEqual(viewer_masks(loaded), VIEWER_MASKS_DEFAULT)

    def test_commands_round_trip_and_defaults_remove_the_keys(self):
        dispatcher = Dispatcher(empty_document())
        dispatcher.execute({"op": "viewer_roi", "on": True, "rect": [0.1, 0.2, 0.6, 0.7]})
        dispatcher.execute({"op": "viewer_proxy", "tier": 8})
        dispatcher.execute({"op": "viewer_mask", "mask": "2.40", "mode": "half"})
        viewer = dispatcher.document["settings"]["viewer"]
        self.assertEqual(viewer["roi"], {"on": True, "rect": [0.1, 0.2, 0.6, 0.7]})
        self.assertEqual((viewer["proxy"], viewer["masks"]), (8, {"mask": "2.40", "mode": "half"}))
        reloaded = reload(dispatcher.document)
        self.assertEqual(viewer_masks(reloaded)["mode"], "half")
        dispatcher.execute({"op": "viewer_roi", "on": False, "rect": [0, 0, 1, 1]})
        dispatcher.execute({"op": "viewer_proxy", "tier": 1})
        dispatcher.execute({"op": "viewer_mask", "mask": "format", "mode": "none"})
        self.assertEqual(set(dispatcher.document["settings"]["viewer"]), {"background"})

    def test_bad_values_are_rejected(self):
        dispatcher = Dispatcher(empty_document())
        for command in ({"op": "viewer_roi", "rect": [0.5, 0.5, 0.2, 0.9]},
                        {"op": "viewer_roi", "rect": [0, 0, 2, 1]},
                        {"op": "viewer_proxy", "tier": 3},
                        {"op": "viewer_mask", "mask": "9.99"},
                        {"op": "viewer_mask", "mode": "purple"}):
            with self.assertRaises(ValueError, msg=str(command)):
                dispatcher.execute(command)
        self.assertEqual(set(dispatcher.document["settings"]["viewer"]), {"background"})


class ProxyTierTableTests(unittest.TestCase):
    def test_eighth_tier_is_in_the_table_and_scales_blur(self):
        self.assertIn(8, PROXY_TIERS)
        self.assertIn("radius", PIXEL_UNIT_PARAMS["Blur"])
        for tier, expected in ((1, 16.0), (2, 8.0), (4, 4.0), (8, 2.0)):
            self.assertEqual(scale_params("Blur", {"radius": 16.0}, tier)["radius"], expected)


class ViewerFrameWindowTests(unittest.TestCase):
    def setUp(self):
        self.window = Window(blur_document())
        self.window.resize(1000, 700)
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None
                                   and self.window.viewer.format_rect is not None))
        self.composed = []
        executor = self.window.tile_executor
        original = executor.compose_region

        def spy(document, target, region, *args, **kwargs):
            result = original(document, target, region, *args, **kwargs)
            self.composed.append({"region": (region.x, region.y, region.width, region.height),
                                  "tiles": result.tile_hits + result.tile_misses, "misses": result.tile_misses,
                                  "tier": kwargs.get("tier", 1), "shape": result.pixels.shape})
            return result

        executor.compose_region = spy
        self.addCleanup(lambda: setattr(executor, "compose_region", original))

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def settle(self):
        self.assertTrue(wait_until(lambda: not self.window.busy and not len(self.window.preview_queue)
                                   and self.window.frame_generation == self.window.generation
                                   and self.window.rendered_identity == self.window.render_identity()))
        APP.processEvents()

    def displayed(self):
        return [entry for entry in self.composed if entry["tiles"]]

    def cold(self):
        """Forget every cached tile and display frame, so a render's misses are its evaluated tiles."""
        self.window.tile_executor.cache.clear()
        self.window.display_cache.clear()
        self.composed.clear()

    def test_roi_clips_the_tile_request_and_leaves_the_rest_unevaluated(self):
        self.settle()
        self.cold()
        self.window.set_viewer_roi(on=True, rect=[0.0, 0.0, 0.5, 0.5])
        self.settle()
        last = self.displayed()[-1]
        self.assertEqual(last["region"], (0, 0, 512, 384))
        self.assertEqual(self.window.frame.shape[:2], (384, 512))
        roi_evaluated = last["misses"]
        self.cold()
        self.window.set_viewer_roi(on=False)
        self.settle()
        full = self.displayed()[-1]
        self.assertEqual(full["region"], (0, 0, 1024, 768))
        self.assertEqual(self.window.frame.shape[:2], (768, 1024))
        # The 4 x 3 tile grid of the canvas against the 2 x 2 tiles the ROI touches (the Blur's
        # halo reaches into neighbours, which is why the count is not exactly a third).
        self.assertGreater(roi_evaluated, 0)
        self.assertLessEqual(roi_evaluated, full["misses"] * 2 // 3)

    def test_outside_the_roi_shows_the_last_full_image_dimmed(self):
        viewer = self.window.viewer
        self.window.set_viewer_roi(on=True, rect=[0.0, 0.0, 0.5, 0.5])
        self.settle()
        grab = viewer.viewport().grab().toImage()
        inside = grab.pixelColor(viewer.mapFromScene(QPointF(100, 100)))
        outside = grab.pixelColor(viewer.mapFromScene(QPointF(900, 700)))
        background = grab.pixelColor(2, 2)
        self.assertGreater(outside.red(), background.red(), "the last full image stays behind the ROI")
        self.assertLess(outside.red(), inside.red(), "and it is dimmed")
        # 35 per cent of the picture over the viewer background.
        expected = 0.35 * inside.red() + 0.65 * background.red()
        self.assertAlmostEqual(outside.red(), expected, delta=2)

    def test_roi_box_is_draggable_and_saved_on_release(self):
        viewer = self.window.viewer
        self.window.set_viewer_roi(on=True, rect=[0.25, 0.25, 0.75, 0.75])
        self.settle()
        start = viewer.mapFromScene(QPointF(512, 384))
        end = viewer.mapFromScene(QPointF(512 + 102.4, 384 + 76.8))
        QTest.mousePress(viewer.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
        QTest.mouseMove(viewer.viewport(), end)
        QTest.mouseRelease(viewer.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end)
        self.settle()
        rect = viewer_roi(self.window.dispatcher.document)["rect"]
        self.assertEqual([round(value, 2) for value in rect], [0.35, 0.35, 0.85, 0.85])
        x, y, width, height = self.displayed()[-1]["region"]
        self.assertTrue(357 <= x <= 361 and 267 <= y <= 271, (x, y))
        self.assertTrue(511 <= width <= 514 and 383 <= height <= 386, (width, height))

    def test_proxy_half_halves_the_requested_tier_and_scales_the_blur(self):
        self.settle()
        self.composed.clear()
        self.window.proxy.setCurrentIndex(self.window.proxy.findData(2))
        self.window.proxy.activated.emit(self.window.proxy.currentIndex())
        self.settle()
        last = self.displayed()[-1]
        self.assertEqual(last["tier"], 2)
        self.assertEqual(last["shape"][:2], (384, 512))
        self.assertEqual(viewer_proxy(self.window.dispatcher.document), 2)
        self.assertEqual(self.window.viewer.proxy_badge_text(), "proxy 1/2")
        # The Blur ran with its size halved: the tier table is what the tile path reads.
        self.assertEqual(scale_params("Blur", {"radius": 16.0}, 2)["radius"], 8.0)
        self.assertEqual(self.window.frame.shape[:2], (384, 512))

    def test_proxy_persists_in_the_document_and_playback_keeps_working(self):
        self.window.proxy.setCurrentIndex(self.window.proxy.findData(4))
        self.window.proxy.activated.emit(self.window.proxy.currentIndex())
        self.settle()
        self.assertEqual(self.window.dispatcher.document["settings"]["viewer"]["proxy"], 4)
        self.window.command({"op": "time", "first": 1, "last": 24})
        self.settle()
        self.window.toggle_playback(True)
        self.assertTrue(wait_until(lambda: self.window.playback_frames_rendered >= 2, 15.0))
        self.window.toggle_playback(False)
        self.assertEqual(self.window.proxy.currentData(), 4)
        self.assertEqual(self.window.viewer.proxy_badge_text(), "proxy 1/4")
        self.window.command({"op": "undo"})  # the time range
        self.window.command({"op": "undo"})  # the proxy pick
        self.settle()
        self.assertEqual(self.window.proxy.currentData(), 1)
        self.assertEqual(self.window.viewer.proxy_badge_text(), "")

    def test_ctrl_p_toggles_the_proxy(self):
        self.settle()
        QTest.keyClick(self.window.viewer, Qt.Key.Key_P, Qt.KeyboardModifier.ControlModifier)
        self.settle()
        self.assertEqual(self.window.proxy.currentData(), 2)
        QTest.keyClick(self.window.viewer, Qt.Key.Key_P, Qt.KeyboardModifier.ControlModifier)
        self.settle()
        self.assertEqual(self.window.proxy.currentData(), 1)

    def test_235_mask_paints_the_expected_rows_and_leaves_the_frame_alone(self):
        viewer = self.window.viewer
        frame_before = np.array(self.window.frame)
        for mode, alpha in (("full", 255), ("half", 128)):
            image = QImage(1920, 1080, QImage.Format.Format_ARGB32)
            image.fill(QColor(0, 0, 0, 0))
            painter = QPainter(image)
            viewer._paint_mask(painter, QRectF(0, 0, 1920, 1080), "2.35", mode)
            painter.end()
            alphas = np.array([[image.pixelColor(x, y).alpha() for x in (0, 960, 1919)]
                               for y in range(1080)])
            dark_rows = np.flatnonzero(alphas[:, 1])
            self.assertEqual(list(dark_rows), list(range(0, 131)) + list(range(949, 1080)))
            self.assertTrue((alphas[dark_rows] == alpha).all())
            self.assertTrue((alphas[131:949] == 0).all())
        lines = QImage(1920, 1080, QImage.Format.Format_ARGB32)
        lines.fill(QColor(0, 0, 0, 0))
        painter = QPainter(lines)
        viewer._paint_mask(painter, QRectF(0, 0, 1920, 1080), "2.35", "lines")
        painter.end()
        marked = [y for y in range(1080) if lines.pixelColor(960, y).alpha()]
        self.assertEqual(len(marked), 2)
        self.assertIn(sorted(marked)[0], (130, 131))
        self.window.set_viewer_mask(mask="2.35", mode="full")
        self.settle()
        self.assertTrue(np.array_equal(self.window.frame, frame_before), "masks are display only")
        self.assertEqual(self.window.mask_mode.currentText(), "full")
        self.window.command({"op": "undo"})
        self.assertEqual(self.window.mask_mode.currentText(), "none")


if __name__ == "__main__":
    unittest.main()
