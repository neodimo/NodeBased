"""Disconnecting a wire is obvious and takes one gesture (DiMo, 10/18): hover and click a wire to select it,
Delete or Backspace disconnects it, a knife line cuts every wire it crosses, a hovered wire shows an "x"
handle. Everything goes through real mouse and key events on the graph's viewport; one undo restores."""
import tests.isolation  # noqa: F401  (settings of the test run, not the developer's)

import unittest

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest

from nodebased.app import CUT_HANDLE_RADIUS, WIRE_PICK
from tests import test_wire_feel as wire_feel
from tests.test_wire_feel import LEFT, NOTHING, ZOOMS, P
from tests.test_desktop import wait_until

CTRL_ALT = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier


class WireDisconnectTests(wire_feel.WireFeelTests):
    """Reuses the wire-feel scene: Read -> Grade -> Blur, with Merge, a second Grade and a Constant free."""

    def record(self, destination, slot="image"):
        return next(r for r in self.graph.edges if (r[2], r[3]) == (destination, slot))

    def on_wire(self, destination, fraction=0.3, slot="image", dx=0, dy=0):
        path = self.record(destination, slot)[0].path()
        return self.at(path.pointAtPercent(fraction), dx, dy)

    def click(self, point, modifiers=NOTHING):
        QTest.mousePress(self.graph.viewport(), LEFT, modifiers, point)
        QTest.mouseRelease(self.graph.viewport(), LEFT, modifiers, point)
        self.settle()

    def key(self, key):
        QTest.keyClick(self.graph, key)
        self.settle(20)

    def undo(self):
        self.window.command({"op": "undo"})
        self.settle(20)

    # ---- hover
    def test_hovering_a_wire_highlights_it_and_leaving_clears_it(self):
        for zoom in ZOOMS:
            with self.subTest(zoom=zoom):
                self.show(zoom)
                edge = self.record("gr")[0]
                self.hover(self.on_wire("gr"))
                self.assertEqual(self.graph.hover_wire, ("gr", "image"))
                self.assertTrue(edge.hovered)
                self.assertFalse(self.record("bl")[0].hovered)
                self.hover(self.at(P(1500, -150)))
                self.assertIsNone(self.graph.hover_wire)
                self.assertFalse(edge.hovered)

    def test_a_wire_is_hit_within_a_few_pixels_at_any_zoom_and_not_beyond(self):
        for zoom in ZOOMS:
            with self.subTest(zoom=zoom):
                self.show(zoom)
                self.hover(self.on_wire("gr", dx=WIRE_PICK - 2))
                self.assertEqual(self.graph.hover_wire, ("gr", "image"))
                self.hover(self.on_wire("gr", dx=WIRE_PICK + 6))
                self.assertIsNone(self.graph.hover_wire)

    # ---- select and delete
    def test_clicking_a_wire_selects_it_with_a_distinct_look(self):
        self.show(0.4)
        self.graph.centerOn(P(1300, -100))
        self.settle()
        self.assertTrue(self.graph.viewport().rect().contains(self.on_wire("gr")))
        edge, other = self.record("gr")[0], self.record("bl")[0]
        self.click(self.on_wire("gr"))
        self.assertEqual(self.graph.selected_wires, {("gr", "image")})
        self.assertTrue(edge.selected and not other.selected)
        self.assertEqual(self.source("gr", "image"), "rd")
        self.graph.select_wires(())
        before = self.graph.grab().toImage()
        self.graph.select_wires({("gr", "image")})
        self.assertNotEqual(before, self.graph.grab().toImage())

    def test_clicking_a_node_or_empty_canvas_lets_go_of_the_wire(self):
        self.show(1.0)
        self.click(self.on_wire("gr"))
        self.click(self.at(P(1500, -150)))
        self.assertEqual(self.graph.selected_wires, set())
        self.click(self.on_wire("gr"))
        self.graph.items_by_id["mg"].setSelected(True)
        self.assertEqual(self.graph.selected_wires, set())

    def test_delete_and_backspace_disconnect_the_selected_wire_and_undo_restores_it(self):
        for zoom in ZOOMS:
            for key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
                with self.subTest(zoom=zoom, key=key):
                    self.command({"op": "connect", "id": "gr", "input": "image", "source": "rd"})
                    self.show(zoom)
                    self.click(self.on_wire("gr"))
                    self.key(key)
                    self.assertIsNone(self.source("gr", "image"))
                    self.assertEqual(self.source("bl", "image"), "gr")
                    self.assertIn("gr", self.nodes)
                    self.undo()
                    self.assertEqual(self.source("gr", "image"), "rd")

    def test_shift_click_selects_several_and_one_delete_is_one_undo_step(self):
        self.show(1.0)
        self.click(self.on_wire("gr"))
        self.click(self.on_wire("bl"), Qt.KeyboardModifier.ShiftModifier)
        self.assertEqual(self.graph.selected_wires, {("gr", "image"), ("bl", "image")})
        self.key(Qt.Key.Key_Delete)
        self.assertIsNone(self.source("gr", "image"))
        self.assertIsNone(self.source("bl", "image"))
        self.undo()
        self.assertEqual((self.source("gr", "image"), self.source("bl", "image")), ("rd", "gr"))

    def test_delete_with_a_node_selected_still_deletes_the_node(self):
        self.show(1.0)
        self.graph.items_by_id["cn"].setSelected(True)
        self.key(Qt.Key.Key_Delete)
        self.assertNotIn("cn", self.nodes)

    # ---- the cut handle
    def test_the_hovered_wire_shows_an_x_handle_that_disconnects_on_click(self):
        for zoom in ZOOMS:
            with self.subTest(zoom=zoom):
                self.command({"op": "connect", "id": "gr", "input": "image", "source": "rd"})
                self.show(zoom)
                handle = self.at(self.record("gr")[0].handle)
                self.hover(handle)
                self.assertEqual(self.graph.hover_wire, ("gr", "image"))
                self.click(handle + QPoint(CUT_HANDLE_RADIUS - 3, 0))
                self.assertIsNone(self.source("gr", "image"))
                self.undo()
                self.assertEqual(self.source("gr", "image"), "rd")


for _name in dir(wire_feel.WireFeelTests):   # the scene and helpers are inherited; its own tests run in their module
    if _name.startswith("test_"):
        setattr(WireDisconnectTests, _name, None)


if __name__ == "__main__":
    unittest.main()
