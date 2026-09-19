"""Graph and viewer navigation: Alt+left-drag pan, Alt+scroll zoom, and unfiltered viewer pixels."""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import Qt, QPoint, QPointF
from PySide6.QtGui import QImage, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from nodebased.app import Window

APP = QApplication.instance() or QApplication([])
ALT = Qt.KeyboardModifier.AltModifier
NONE = Qt.KeyboardModifier.NoModifier
LEFT = Qt.MouseButton.LeftButton


def wait_until(condition, timeout=20.0):
    waited = 0
    while waited < timeout * 1000:
        if condition():
            return True
        QTest.qWait(10)
        waited += 10
    return False


def wheel(view, angle, modifiers=NONE):
    pos = QPointF(view.viewport().rect().center())
    event = QWheelEvent(pos, QPointF(view.viewport().mapToGlobal(pos.toPoint())), QPoint(0, 0), angle,
                        Qt.MouseButton.NoButton, modifiers, Qt.ScrollPhase.NoScrollPhase, False)
    view.wheelEvent(event)


class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.window = Window()
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def roomy(self, view):
        """Zoom in so the scene is larger than the viewport and the scroll bars have range."""
        view.resetTransform()
        view.scale(6, 6)
        APP.processEvents()
        return view.viewport().rect().center()

    def scroll(self, view):
        return view.horizontalScrollBar().value(), view.verticalScrollBar().value()

    def alt_drag(self, view, start, offset):
        QTest.mousePress(view.viewport(), LEFT, ALT, start)
        QTest.mouseMove(view.viewport(), start + offset, 20)
        QTest.mouseRelease(view.viewport(), LEFT, ALT, start + offset)

    def test_alt_left_drag_pans_the_graph_even_over_a_node(self):
        graph = self.window.graph
        self.roomy(graph)
        key, item = next((k, i) for k, i in graph.items_by_id.items() if not i.is_dot)
        graph.centerOn(item)
        APP.processEvents()
        start = graph.mapFromScene(item.sceneBoundingRect().center())
        before_scroll = self.scroll(graph)
        before_pos = list(self.window.dispatcher.document["nodes"][key]["pos"])
        before_selection = [i for i in graph.scene().selectedItems()]
        self.alt_drag(graph, start, QPoint(-60, -40))
        after = self.scroll(graph)
        self.assertEqual((after[0] - before_scroll[0], after[1] - before_scroll[1]), (60, 40))
        self.assertEqual(self.window.dispatcher.document["nodes"][key]["pos"], before_pos)
        self.assertEqual(graph.scene().selectedItems(), before_selection)
        self.assertIsNone(graph.pan)

    def test_pan_survives_releasing_alt_before_the_button(self):
        graph = self.window.graph
        start = self.roomy(graph)
        before = self.scroll(graph)
        QTest.mousePress(graph.viewport(), LEFT, ALT, start)
        QTest.mouseMove(graph.viewport(), start + QPoint(-30, 0), 20)
        QTest.mouseRelease(graph.viewport(), LEFT, NONE, start + QPoint(-30, 0))
        self.assertEqual(self.scroll(graph)[0] - before[0], 30)
        self.assertIsNone(graph.pan)
        # The next plain drag is a normal one again, not a pan.
        after = self.scroll(graph)
        QTest.mouseMove(graph.viewport(), start + QPoint(-90, 0), 20)
        self.assertEqual(self.scroll(graph), after)

    def test_plain_left_drag_still_does_not_pan(self):
        graph = self.window.graph
        start = self.roomy(graph)
        empty = graph.mapFromScene(graph.sceneRect().topLeft() + QPointF(5, 5))
        before = self.scroll(graph)
        QTest.mousePress(graph.viewport(), LEFT, NONE, empty)
        QTest.mouseMove(graph.viewport(), empty + QPoint(40, 40), 20)
        QTest.mouseRelease(graph.viewport(), LEFT, NONE, empty + QPoint(40, 40))
        self.assertEqual(self.scroll(graph), before)
        self.assertIsNone(graph.pan)

    def test_alt_left_drag_pans_the_viewer_and_places_no_roto_point(self):
        viewer = self.window.viewer
        start = self.roomy(viewer)
        viewer.roto_drawing = True
        try:
            before = self.scroll(viewer)
            self.alt_drag(viewer, start, QPoint(-50, -20))
            after = self.scroll(viewer)
            self.assertEqual((after[0] - before[0], after[1] - before[1]), (50, 20))
            self.assertEqual(viewer.roto_draw_points, [])
            self.assertIsNone(viewer.pan)
        finally:
            viewer.roto_drawing = False

    def test_alt_scroll_zooms_both_ways(self):
        # With Alt held, X11 and Windows deliver the wheel on the horizontal axis. Reading only
        # the vertical axis made every Alt+scroll a zoom out.
        for view in (self.window.graph, self.window.viewer):
            view.resetTransform()
            wheel(view, QPoint(120, 0), ALT)
            self.assertAlmostEqual(view.transform().m11(), 1.15, places=6)
            wheel(view, QPoint(-120, 0), ALT)
            self.assertAlmostEqual(view.transform().m11(), 1.0, places=6)
            # Platforms that leave the axis alone (Wayland, macOS) zoom the same way.
            wheel(view, QPoint(0, 120), ALT)
            self.assertAlmostEqual(view.transform().m11(), 1.15, places=6)

    def test_touchpad_deltas_zoom_in_proportion(self):
        graph = self.window.graph
        graph.resetTransform()
        for _ in range(8):
            wheel(graph, QPoint(0, 15))  # eight small touchpad deltas add up to one notch
        self.assertAlmostEqual(graph.transform().m11(), 1.15, places=6)
        wheel(graph, QPoint(0, 0))
        self.assertAlmostEqual(graph.transform().m11(), 1.15, places=6)

    def test_viewer_shows_unfiltered_pixels_at_any_zoom(self):
        viewer = self.window.viewer
        size = 64
        pixels = np.zeros((size, size, 3), np.uint8)
        pixels[::2, ::2] = 255
        pixels[1::2, 1::2] = 255
        image = QImage(pixels.data, size, size, pixels.strides[0], QImage.Format.Format_RGB888).copy()
        self.window._show_image(image, 1, None)
        for zoom in (1.0, 4.0, 2.5, 0.5, 0.37):
            viewer.resetTransform()
            viewer.scale(zoom, zoom)
            viewer.centerOn(size / 2, size / 2)
            APP.processEvents()
            top_left = viewer.mapFromScene(QPointF(0, 0))
            bottom_right = viewer.mapFromScene(QPointF(size, size))
            # One pixel inside the picture's edge, clear of the format overlay's outline.
            grabbed = viewer.viewport().grab().toImage().convertToFormat(QImage.Format.Format_RGB888)
            data = np.frombuffer(grabbed.constBits(), np.uint8, grabbed.sizeInBytes())
            data = data.reshape(grabbed.height(), grabbed.bytesPerLine())[:, :grabbed.width() * 3]
            data = data.reshape(grabbed.height(), grabbed.width(), 3)
            inset = max(2, int(np.ceil(zoom)) + 1)
            crop = data[top_left.y() + inset:bottom_right.y() - inset,
                        top_left.x() + inset:bottom_right.x() - inset]
            self.assertGreater(crop.size, 0)
            values = set(np.unique(crop).tolist())
            self.assertTrue(values <= {0, 255}, f"zoom {zoom}: interpolated values {sorted(values)}")


if __name__ == "__main__":
    unittest.main()
