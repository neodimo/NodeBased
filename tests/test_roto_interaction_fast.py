"""Roto interaction stays quick: a drag ends in one undo step and the document the old route produced, its picture requests
carry the document the release will commit, handles repaint without waiting for the picture, and the heavy panel rebuilds
once after a burst instead of inside every click.

No test asserts elapsed time (the suite runs many interpreters at once); they assert the work that was slow: copies, undo
entries, picture requests and panel rebuilds.
"""
import copy
import math
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, QRectF, QSettings, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from nodebased.app import STYLE, Window
from tests.test_roto_ui import roto_document, triangle, point_value
from tests.waiting import wait_until

APP = QApplication.instance() or QApplication([])
APP.setStyle("Fusion")
APP.setStyleSheet(STYLE)


def blob(index, count=12):
    cx, cy = 20 + (index % 5) * 15, 20 + (index // 5) * 20
    points = []
    for i in range(count):
        a = 2 * math.pi * i / count
        points.append({"x": cx + 5 * math.cos(a), "y": cy + 5 * math.sin(a), "in_x": 0.0, "in_y": 0.0,
                       "out_x": 0.0, "out_y": 0.0})
    return {"name": f"blob{index}", "mode": "union", "opacity": 1.0, "feather": 0.0, "points": points}


class FastRotoTests(unittest.TestCase):
    def open(self, shapes):
        QSettings("NodeBased", "NodeBased").clear()
        self.window = Window(roto_document(shapes))
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))
        self.window.graph.items_by_id["r"].setSelected(True)
        APP.processEvents()
        self.window.timer.stop()
        return self.window.viewer

    def close(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def tearDown(self):
        if getattr(self, "window", None) is not None:
            self.close()
            self.window = None

    def at(self, x, y):
        return self.window.viewer.mapFromScene(QPointF(x, y))

    def drag_point(self, viewer):
        start, end = self.at(10, 10), self.at(40, 35)
        QTest.mousePress(viewer.viewport(), Qt.MouseButton.LeftButton, pos=start)
        for step in range(1, 7):
            QTest.mouseMove(viewer.viewport(), self.at(10 + 5 * step, 10 + 25 * step / 6))
        QTest.mouseMove(viewer.viewport(), end)
        mid = (len(self.window.dispatcher.undo_stack), copy.deepcopy(self.window.dispatcher.document["node_data"]))
        QTest.mouseRelease(viewer.viewport(), Qt.MouseButton.LeftButton, pos=end)
        return mid, copy.deepcopy(self.window.dispatcher.document["node_data"]), len(self.window.dispatcher.undo_stack)

    def test_drag_ends_in_one_undo_step_and_the_document_a_full_copy_edit_produces(self):
        viewer = self.open([triangle()])
        document_before = copy.deepcopy(self.window.dispatcher.document)
        (mid_undo, mid_data), committed, undo_steps = self.drag_point(viewer)
        self.assertEqual(mid_undo, 0)                                   # nothing is committed mid-drag
        self.assertEqual(mid_data, document_before["node_data"])
        self.assertEqual(undo_steps, 1)
        self.window.command({"op": "undo"})
        self.assertEqual(self.window.dispatcher.document["node_data"], document_before["node_data"])
        self.close()
        # the same drag with every shape copied before the edit (how the commit worked before it copied only what moves)
        viewer = self.open([triangle()])
        viewer._shapes_for_edit = lambda payload, indices: copy.deepcopy(payload.get("shapes", []))
        _, reference, _ = self.drag_point(viewer)
        self.assertEqual(committed, reference)
        moved = committed["r"]["shapes"][0]["points"]
        self.assertNotEqual(moved[0], document_before["node_data"]["r"]["shapes"][0]["points"][0])
        self.assertEqual(moved[1], document_before["node_data"]["r"]["shapes"][0]["points"][1])

    def test_the_picture_requested_during_a_drag_carries_the_document_the_release_commits(self):
        shapes = [blob(i) for i in range(10)]
        viewer = self.open(shapes)
        target = (shapes[3]["points"][2]["x"], shapes[3]["points"][2]["y"])
        end = self.at(target[0] + 6, target[1] + 4)
        QTest.mousePress(viewer.viewport(), Qt.MouseButton.LeftButton, pos=self.at(*target))
        QTest.mouseMove(viewer.viewport(), end)
        self.window.timer.stop()
        request, _cancel = self.window.preview_queue.take()
        self.assertEqual(request.tier, 8)
        live = self.window.dispatcher.document["node_data"]["r"]["shapes"]
        previewed = request.document["node_data"]["r"]["shapes"]
        for index, shape in enumerate(previewed):
            if index != 3:
                self.assertIs(shape, live[index], "an untouched shape is shared, not copied")
        self.assertNotEqual(previewed[3], live[3])
        QTest.mouseRelease(viewer.viewport(), Qt.MouseButton.LeftButton, pos=end)
        committed = self.window.dispatcher.document["node_data"]["r"]["shapes"]
        self.assertEqual(previewed, committed)
        self.assertEqual(len(self.window.dispatcher.undo_stack), 1)

    def test_moves_inside_the_preview_interval_make_one_request(self):
        viewer = self.open([triangle()])
        requests = []
        original = self.window.request_preview
        self.window.request_preview = lambda *a, **k: requests.append(k.get("document_override")) or original(*a, **k)
        viewer.DRAG_PREVIEW_INTERVAL = 3600.0           # no second request can be due while the test runs
        QTest.mousePress(viewer.viewport(), Qt.MouseButton.LeftButton, pos=self.at(10, 10))
        for step in range(1, 12):
            QTest.mouseMove(viewer.viewport(), self.at(10 + 3 * step, 10 + 2 * step))
        self.assertEqual(len(requests), 1)
        viewer._drag_preview_timer.stop()
        viewer._flush_drag_preview()                    # the interval is up: the newest position is requested once
        self.assertEqual(len(requests), 2)
        position = requests[1]["node_data"]["r"]["shapes"][0]["points"][0]
        self.assertAlmostEqual(point_value(position["x"]), 10 + 33, delta=2.0)
        QTest.mouseRelease(viewer.viewport(), Qt.MouseButton.LeftButton, pos=self.at(43, 32))
        self.assertFalse(viewer._drag_preview_timer.isActive())

    def test_resolved_shapes_are_kept_until_the_document_frame_or_node_changes(self):
        viewer = self.open([triangle()])
        context = viewer._roto_context()
        first = viewer._roto_resolved(context)
        self.assertIs(viewer._roto_resolved(context), first)
        self.window.command({"op": "set_shapes", "id": "r", "shapes": [triangle(), blob(1)]})
        second = viewer._roto_resolved(viewer._roto_context())
        self.assertIsNot(second, first)
        self.assertEqual(len(second), 2)
        self.window.command({"op": "time", "current": 2})
        self.assertIsNot(viewer._roto_resolved(viewer._roto_context()), second)

    def test_painting_a_drag_never_changes_the_cached_resolved_points(self):
        viewer = self.open([triangle()])
        context = viewer._roto_context()
        resolved = copy.deepcopy(viewer._roto_resolved(context))
        QTest.mousePress(viewer.viewport(), Qt.MouseButton.LeftButton, pos=self.at(10, 10))
        QTest.mouseMove(viewer.viewport(), self.at(40, 40))
        image = QImage(200, 200, QImage.Format.Format_ARGB32)
        painter = QPainter(image)
        viewer.paint_overlays(painter, QRectF(0, 0, 100, 100))      # draws the point at the pointer
        painter.end()
        self.assertEqual(viewer._roto_resolved(context), resolved)
        QTest.mouseRelease(viewer.viewport(), Qt.MouseButton.LeftButton, pos=self.at(40, 40))

    def test_a_marquee_draws_its_rectangle_without_error(self):
        viewer = self.open([triangle()])
        viewer.select_roto_tool("Select", "Select Points")
        QTest.mousePress(viewer.viewport(), Qt.MouseButton.LeftButton, pos=self.at(2, 2))
        QTest.mouseMove(viewer.viewport(), self.at(60, 60))
        self.assertEqual(viewer.roto_drag["kind"], "marquee")
        image = QImage(300, 300, QImage.Format.Format_ARGB32)
        image.fill(0)
        painter = QPainter(image)
        viewer.paint_overlays(painter, QRectF(0, 0, 100, 100))      # a marquee has no point: this raised KeyError
        painter.end()
        QTest.mouseRelease(viewer.viewport(), Qt.MouseButton.LeftButton, pos=self.at(60, 60))

    def test_a_heavy_roto_panel_rebuilds_once_after_a_burst_and_a_light_one_at_once(self):
        self.open([triangle()])
        builds = []
        original = self.window.build_node_panel
        self.window.build_node_panel = lambda key: builds.append(key) or original(key)
        self.window.command({"op": "set_shapes", "id": "r", "shapes": [triangle(), blob(1)]})
        self.assertIn("r", builds)                                  # a light panel is rebuilt inside the command
        self.close()
        self.open([blob(i) for i in range(self.window.HEAVY_PANEL_ITEMS)])
        builds = []
        original = self.window.build_node_panel
        self.window.build_node_panel = lambda key: builds.append(key) or original(key)
        shapes = [blob(i) for i in range(self.window.HEAVY_PANEL_ITEMS)]
        for count in (1, 2, 3, 4):
            self.window.command({"op": "set_shapes", "id": "r", "shapes": shapes[:-1] + [blob(count)]})
        self.assertEqual(builds, [])                                # nothing rebuilt inside the burst
        self.assertTrue(self.window._panel_rebuild_timer.isActive())
        self.window._panel_rebuild_timer.stop()
        self.window._run_panel_rebuild()
        self.assertEqual(builds, ["r"])                             # one rebuild for four edits
        self.window._run_panel_rebuild()
        self.assertEqual(builds, ["r"])


if __name__ == "__main__":
    unittest.main()
