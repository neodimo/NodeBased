"""Offscreen interaction tests for 3D viewport selection (lane L1, step 2)."""
import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from nodebased.app import STYLE, Window
from nodebased.core import Dispatcher

APP = QApplication.instance() or QApplication([])
APP.setStyle("Fusion")
APP.setStyleSheet(STYLE)


def wait_until(condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        APP.processEvents()
        if condition():
            return True
        QTest.qWait(10)
    return False


def two_cards_document():
    """'near' (tz=2) and 'far' (tz=-2) Card3D nodes, grouped by a Scene3D, into a Render3D."""
    d = Dispatcher()
    for key, kind in (("near", "Card3D"), ("far", "Card3D"), ("scene", "Scene3D"),
                      ("cam", "Camera3D"), ("render", "Render3D")):
        d.execute({"op": "create", "id": key, "type": kind})
    d.execute({"op": "set", "id": "near", "param": "tz", "value": 2.0})
    d.execute({"op": "set", "id": "far", "param": "tz", "value": -2.0})
    # Off the shared Z axis so its own screen-space marker (lane L1 step 4) does not sit on
    # top of the pixel these tests click to pick the cards -- 'cam' is only wired up here to
    # complete the Render3D graph and is never itself the target of these assertions.
    d.execute({"op": "set", "id": "cam", "param": "tx", "value": 6.0})
    d.execute({"op": "connect", "id": "scene", "input": "object0", "source": "near"})
    d.execute({"op": "connect", "id": "scene", "input": "object1", "source": "far"})
    d.execute({"op": "connect", "id": "render", "input": "scene", "source": "scene"})
    d.execute({"op": "connect", "id": "render", "input": "camera", "source": "cam"})
    return d.document


class Viewport3DPickingTests(unittest.TestCase):
    def setUp(self):
        self.window = Window(two_cards_document())
        self.window.show()
        self.window.viewport_dock.show()
        self.window.resize(1000, 800)
        self.viewport = self.window.viewport
        self.viewport.resize(640, 360)
        # A simple front-on view: camera at (0, 0, 10) looking at the origin, matching both
        # cards' xy so 'near' (z=2) sits directly in front of 'far' (z=-2) on screen.
        self.viewport.azimuth, self.viewport.elevation, self.viewport.distance = 0.0, 0.0, 10.0
        APP.processEvents()
        self.center = QPointF(self.viewport.width() / 2, self.viewport.height() / 2)

    def tearDown(self):
        # A mutating command (the delete test) leaves the document dirty; sync saved_document
        # first so close() does not raise a blocking "save changes?" dialog in offscreen mode.
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def test_click_on_a_card_picks_it_and_opens_its_properties(self):
        QTest.mouseClick(self.viewport, Qt.MouseButton.LeftButton, pos=self.center.toPoint())
        self.assertEqual(self.viewport.selected_key, "near")
        self.assertEqual(self.window.graph.selected_id(), "near")

    def test_nearer_of_two_overlapping_cards_wins(self):
        # Both cards are centred on the same screen point; 'near' (z=2) must win over 'far'.
        QTest.mouseClick(self.viewport, Qt.MouseButton.LeftButton, pos=self.center.toPoint())
        self.assertEqual(self.viewport.selected_key, "near")

    def test_click_on_empty_space_clears_selection(self):
        QTest.mouseClick(self.viewport, Qt.MouseButton.LeftButton, pos=self.center.toPoint())
        self.assertIsNotNone(self.viewport.selected_key)
        corner = QPointF(4, 4).toPoint()
        QTest.mouseClick(self.viewport, Qt.MouseButton.LeftButton, pos=corner)
        self.assertIsNone(self.viewport.selected_key)
        self.assertIsNone(self.window.graph.selected_id())

    def test_a_drag_does_not_pick(self):
        target = QPointF(self.center.x() + 120, self.center.y() + 40).toPoint()
        QTest.mousePress(self.viewport, Qt.MouseButton.LeftButton, pos=self.center.toPoint())
        QTest.mouseMove(self.viewport, target)
        QTest.mouseRelease(self.viewport, Qt.MouseButton.LeftButton, pos=target)
        self.assertIsNone(self.viewport.selected_key)
        # Orbiting is the drag's real job: the camera actually moved.
        self.assertNotEqual((self.viewport.azimuth, self.viewport.elevation), (0.0, 0.0))

    def test_orbit_still_works_after_a_click_picks_something(self):
        QTest.mouseClick(self.viewport, Qt.MouseButton.LeftButton, pos=self.center.toPoint())
        self.assertEqual(self.viewport.selected_key, "near")
        azimuth_before = self.viewport.azimuth
        start = self.center.toPoint()
        end = QPointF(self.center.x() + 60, self.center.y()).toPoint()
        QTest.mousePress(self.viewport, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(self.viewport, end)
        QTest.mouseRelease(self.viewport, Qt.MouseButton.LeftButton, pos=end)
        self.assertNotEqual(self.viewport.azimuth, azimuth_before)
        # A drag never picks (see test_a_drag_does_not_pick), so the earlier click's
        # selection is left exactly as it was, not cleared or replaced.
        self.assertEqual(self.viewport.selected_key, "near")

    def test_selection_outline_bounds_are_drawn_for_the_picked_card(self):
        QTest.mouseClick(self.viewport, Qt.MouseButton.LeftButton, pos=self.center.toPoint())
        bounds = self.viewport._selected_bounds()
        self.assertIsNotNone(bounds)
        low, high = bounds
        # The card sits 1x1 (default Card3D size) centred at (0, 0, 2): its bounds must
        # straddle that centre, which is what `_draw_selection` projects into the frame.
        self.assertLess(low[0], 0.0)
        self.assertGreater(high[0], 0.0)
        self.assertAlmostEqual((low[2] + high[2]) / 2, 2.0, delta=1e-4)

    def test_disabling_the_selected_node_clears_selection_on_the_next_document_swap(self):
        QTest.mouseClick(self.viewport, Qt.MouseButton.LeftButton, pos=self.center.toPoint())
        self.assertEqual(self.viewport.selected_key, "near")
        self.window.command({"op": "delete", "id": "near"})
        self.assertIsNone(self.viewport.selected_key)


if __name__ == "__main__":
    unittest.main()
