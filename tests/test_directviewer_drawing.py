"""Roto and RotoPaint drawing on the direct picture (Lane 2, 10/22).

The same sequence of presses, moves, releases and keys is played once through the Viewer's viewport and once
through the direct picture's widget, from the same document, at device scale 1.0 and 1.75 and at several
zooms. The shape or stroke data each leaves in the document must be identical, with one undo entry per shape or
stroke, and the in-progress marks (new points, the rubber band, the brush line) must reach the direct picture's
overlay while the drawing is under way.

Like `test_directviewer_hit_order`, the direct widget is the real one with its GPU presenter left out.
"""
import copy
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest

import tests.isolation  # noqa: F401
from tests.test_directviewer_hit_order import (ALT, APP, CTRL, SHIFT, HAVE_CANVAS, NO_MOD, SCALES, WIDTH, ZOOMS,  # noqa: F401
                                                HitOrderBase, numbers_close, scene)

YELLOW = QColor("#f4ce63")
RED = QColor("#ff5c48")


def shapes_of(window, key="r"):
    return copy.deepcopy(window.dispatcher.document.get("node_data", {}).get(key, {}).get("shapes", []))


def items_of(window, key="p"):
    return copy.deepcopy(window.dispatcher.document.get("node_data", {}).get(key, {}).get("items", []))


class Step:
    """One pointer or key action: ("click", x, y), ("drag", x, y, dx, dy), ("hover", x, y), ("key", Qt.Key, mods)."""
    def __init__(self, kind, *args):
        self.kind, self.args = kind, args


def click(x, y, mods=NO_MOD):
    return Step("click", x, y, mods)


def drag(x, y, dx, dy, mods=NO_MOD):
    return Step("drag", x, y, dx, dy, mods)


def hover(x, y):
    return Step("hover", x, y)


def key(code, mods=NO_MOD):
    return Step("key", code, mods)


# A shape in the empty right-hand part of the picture (clear of the blob the document already holds).
OPEN_SHAPE = (hover(430, 90), click(430, 90), hover(520, 100), drag(540, 130, 30, 20), hover(560, 220),
              click(560, 230), hover(470, 250), key(Qt.Key.Key_Return))
CLOSED_SHAPE = (click(430, 90), drag(540, 110, 24, 12), click(560, 230), click(450, 250), hover(433, 93),
                click(430, 90))
CANCELLED_SHAPE = (click(430, 90), click(520, 130), hover(500, 200), key(Qt.Key.Key_Escape))
STROKE = ("stroke", (30, 30), (50, 44), (80, 60), (95, 52))


class DrawingBase(HitOrderBase):
    def play(self, surface, steps, zoom, scale):
        """Play `steps` through `surface`. Scene coordinates are the picture's own pixels."""
        target = surface.viewport() if surface is self.viewer else surface
        for step in steps:
            if step.kind == "key":
                code, mods = step.args
                QTest.keyClick(target if surface is not self.viewer else self.viewer, code, mods)
                APP.processEvents()
                continue
            x, y = step.args[:2]
            start = self.pixel(x, y)
            if step.kind == "hover":
                QTest.mouseMove(target, start)
            elif step.kind == "click":
                mods = step.args[2]
                QTest.mousePress(target, Qt.MouseButton.LeftButton, mods, start)
                QTest.mouseRelease(target, Qt.MouseButton.LeftButton, mods, start)
            elif step.kind == "drag":
                dx, dy, mods = step.args[2:]
                end = self.pixel(x + dx, y + dy)
                QTest.mousePress(target, Qt.MouseButton.LeftButton, mods, start)
                QTest.mouseMove(target, QPoint((start.x() + end.x()) // 2, (start.y() + end.y()) // 2))
                QTest.mouseMove(target, end)
                QTest.mouseRelease(target, Qt.MouseButton.LeftButton, mods, end)
            APP.processEvents()

    def pixel(self, x, y):
        return self.viewer.mapFromScene(scene(self.window, x, y))

    def lay_out(self, zoom, scale):
        self.place_surfaces(scene(self.window, WIDTH / 2.0, 180.0), zoom, scale)


class RotoDrawingTests(DrawingBase):
    def setUp(self):
        super().setUp()
        self.show_node("r")

    def one_run(self, surface, steps, zoom, scale):
        self.lay_out(zoom, scale)
        self.viewer.cancel_roto_edit()
        self.base_undo = len(self.window.dispatcher.undo_stack)
        before = shapes_of(self.window)
        self.play(surface, steps, zoom, scale)
        result = (before, shapes_of(self.window), len(self.window.dispatcher.undo_stack) - self.base_undo,
                  self.viewer.roto_drawing)
        self.viewer.cancel_roto_edit()
        self.rewind()
        self.assertEqual(shapes_of(self.window), before)
        return result

    def compare(self, steps, label, expect_new=True):
        for scale in SCALES:
            for zoom in (0.5, 1.0, 2.0):
                where = f"{label}, scale {scale}, zoom {zoom}"
                with self.subTest(where):
                    before, via_viewer, undo_viewer, drawing_viewer = self.one_run(self.viewer, steps, zoom, scale)
                    _, via_direct, undo_direct, drawing_direct = self.one_run(self.widget, steps, zoom, scale)
                    self.assertIsNone(numbers_close(via_viewer, via_direct), where)
                    self.assertEqual(undo_viewer, undo_direct, where)
                    self.assertEqual(drawing_viewer, drawing_direct, where)
                    if expect_new:
                        self.assertEqual(len(via_viewer), len(before) + 1, f"{where}: the Viewer drew a shape")
                        self.assertEqual(undo_viewer, 1, where)
                    else:
                        self.assertEqual(via_viewer, before, where)

    def test_an_open_shape_with_a_dragged_tangent_ends_on_enter_identically(self):
        self.compare(OPEN_SHAPE, "open shape")

    def test_a_shape_closes_on_its_first_point_identically(self):
        self.compare(CLOSED_SHAPE, "closed shape")
        before, drawn, _, _ = self.one_run(self.viewer, CLOSED_SHAPE, 1.0, 1.0)
        self.assertTrue(drawn[-1]["closed"])
        self.assertEqual(len(drawn[-1]["points"]), 4)
        self.assertTrue(any(abs(p["out_x"]) > 1.0 for p in drawn[-1]["points"]), "a dragged tangent was kept")

    def test_escape_cancels_the_drawing_on_both_surfaces(self):
        self.compare(CANCELLED_SHAPE, "cancelled", expect_new=False)

    def test_the_marks_of_a_shape_under_way_are_on_the_direct_overlay(self):
        """After every action the direct picture's overlay is repainted and shows the new points and the rubber band."""
        self.lay_out(1.0, 1.0)
        widget, viewer = self.widget, self.viewer
        size = self.widget._physical()

        def yellow_near(x, y, radius=6):
            image = self.window._direct_overlay(widget.view, size, 1.0, force=True)
            cx, cy, zoom = widget.view
            px = int(round((scene(self.window, x, y).x() - cx) * zoom + size[0] / 2.0))
            py = int(round((scene(self.window, x, y).y() - cy) * zoom + size[1] / 2.0))
            for yy in range(py - radius, py + radius + 1):
                for xx in range(px - radius, px + radius + 1):
                    if 0 <= xx < image.width() and 0 <= yy < image.height():
                        color = image.pixelColor(xx, yy)
                        if color.alpha() > 128 and abs(color.red() - YELLOW.red()) < 40 and abs(color.green() - YELLOW.green()) < 40:
                            return True
            return False

        serial = widget.overlay_serial
        self.play(widget, (click(430, 90),), 1.0, 1.0)
        self.assertTrue(viewer.roto_drawing)
        self.assertGreater(widget.overlay_serial, serial, "the press and release repaint the overlay")
        self.assertTrue(yellow_near(430, 90), "the first point is drawn")
        serial = widget.overlay_serial
        self.play(widget, (hover(520, 150),), 1.0, 1.0)
        self.assertGreater(widget.overlay_serial, serial, "the rubber band follows the pointer")
        self.assertTrue(yellow_near(520, 150), "the rubber band ends at the pointer")
        self.play(widget, (click(520, 150),), 1.0, 1.0)
        self.assertTrue(yellow_near(520, 150), "the second point is drawn")
        self.assertTrue(yellow_near(475, 120, 8), "the line between the points is drawn")
        self.play(widget, (key(Qt.Key.Key_Escape),), 1.0, 1.0)
        self.assertFalse(viewer.roto_drawing)
        self.assertFalse(yellow_near(430, 90), "cancelling takes the marks away")

    def test_delete_and_the_tangent_keys_reach_a_selected_point_from_the_direct_picture(self):
        """Shift+Z (cusp) and Delete need the Viewer's focus on the Viewer; the direct picture has the focus
        itself and must get the same result."""
        results = []
        for surface in (self.viewer, self.widget):
            self.lay_out(1.0, 1.0)
            self.viewer.cancel_roto_edit()
            self.base_undo = len(self.window.dispatcher.undo_stack)
            before = shapes_of(self.window)
            self.viewer.roto_selected_shape_index = 0
            self.viewer.roto_selected_points = {(0, 1)}
            surface.setFocus()
            APP.processEvents()
            QTest.keyClick(surface if surface is not self.viewer else self.viewer, Qt.Key.Key_Z, SHIFT)
            APP.processEvents()
            after_z = shapes_of(self.window)
            QTest.keyClick(surface if surface is not self.viewer else self.viewer, Qt.Key.Key_Delete)
            APP.processEvents()
            after_delete = shapes_of(self.window)
            results.append((after_z, after_delete))
            self.assertNotEqual(after_z, before, "Shift+Z made the point a cusp")
            self.assertEqual(len(after_delete[0]["points"]), len(before[0]["points"]) - 1, "Delete removed the point")
            self.rewind()
            self.assertEqual(shapes_of(self.window), before)
        self.assertIsNone(numbers_close(results[0], results[1]))


class RotoPaintDrawingTests(DrawingBase):
    def setUp(self):
        super().setUp()
        for command in ({"op": "create", "id": "p", "type": "RotoPaint", "pos": [900, 0]},
                        {"op": "connect", "id": "p", "input": "image", "source": "c"}):
            self.window.command(command)
        self.show_node("p")
        self.viewer.paint_tool = "paint"

    def stroke(self, surface, zoom, scale, points=STROKE[1:]):
        self.lay_out(zoom, scale)
        self.base_undo = len(self.window.dispatcher.undo_stack)
        target = surface.viewport() if surface is self.viewer else surface
        start = self.pixel(*points[0])
        QTest.mousePress(target, Qt.MouseButton.LeftButton, NO_MOD, start)
        for point in points[1:]:
            QTest.mouseMove(target, self.pixel(*point))
            APP.processEvents()
        QTest.mouseRelease(target, Qt.MouseButton.LeftButton, NO_MOD, self.pixel(*points[-1]))
        APP.processEvents()
        result = (items_of(self.window), len(self.window.dispatcher.undo_stack) - self.base_undo)
        self.rewind()
        self.assertEqual(items_of(self.window), [])
        return result

    def test_a_paint_stroke_is_the_same_item_on_both_surfaces(self):
        for scale in SCALES:
            for zoom in (0.5, 1.0, 2.0):
                where = f"scale {scale}, zoom {zoom}"
                with self.subTest(where):
                    via_viewer, undo_viewer = self.stroke(self.viewer, zoom, scale)
                    via_direct, undo_direct = self.stroke(self.widget, zoom, scale)
                    self.assertEqual(len(via_viewer), 1, where)
                    self.assertGreaterEqual(len(via_viewer[0]["points"]), 4, where)
                    self.assertIsNone(numbers_close(via_viewer, via_direct), where)
                    self.assertEqual((undo_viewer, undo_direct), (1, 1), where)

    def test_each_brush_tool_key_works_on_the_direct_picture(self):
        for code, tool in ((Qt.Key.Key_E, "eraser"), (Qt.Key.Key_C, "clone"), (Qt.Key.Key_B, "paint")):
            QTest.keyClick(self.widget, code)
            self.assertEqual(self.viewer.paint_tool, tool)
            via_direct, _ = self.stroke(self.widget, 1.0, 1.0)
            self.assertEqual([item["tool"] for item in via_direct], [tool])
            self.assertEqual(self.window.channels.currentText(), "RGB")

    def test_the_brush_line_under_way_is_on_the_direct_overlay(self):
        self.lay_out(1.0, 1.0)
        widget, size = self.widget, self.widget._physical()
        start = self.pixel(30, 30)
        serial = widget.overlay_serial
        QTest.mousePress(widget, Qt.MouseButton.LeftButton, NO_MOD, start)
        for point in STROKE[2:]:
            QTest.mouseMove(widget, self.pixel(*point))
        self.assertGreater(widget.overlay_serial, serial, "each sample repaints the overlay")
        image = self.window._direct_overlay(widget.view, size, 1.0, force=True)
        cx, cy, zoom = widget.view
        mid = scene(self.window, 80, 60)
        px = int(round((mid.x() - cx) * zoom + size[0] / 2.0))
        py = int(round((mid.y() - cy) * zoom + size[1] / 2.0))
        color = image.pixelColor(px, py)
        self.assertGreater(color.alpha(), 128, "the stroke so far is drawn under the pointer")
        QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, NO_MOD, self.pixel(*STROKE[-1]))
        self.rewind()


if __name__ == "__main__":
    unittest.main()
