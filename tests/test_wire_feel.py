"""New look, wiring feel: connecting a noodle is natural, never fussy.

A dragged wire connects to the nearest compatible socket within 40 screen pixels, or to the socket of the
card under the pointer; a press near a socket starts a wire; the same gestures give the same result at any
zoom. Everything goes through real mouse events on the graph's viewport."""
import tests.isolation  # noqa: F401  (settings of the test run, not the developer's)

import unittest

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest

from nodebased import graphlook
from nodebased.app import GRAB_REACH, OUTPUT_BAND, WIRE_REACH, Preferences, Window
from tests.test_desktop import APP, wait_until

P = QPointF
LEFT, NOTHING = Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
ZOOMS = (0.4, 1.0, 1.5)


class WireFeelTests(unittest.TestCase):
    def setUp(self):
        Preferences().set_wire_mode(graphlook.CURVED)
        self.window = Window()
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))
        self.graph = self.window.graph
        # Read feeds Grade; Merge and a second Grade wait unconnected; Blur hangs below Grade (a cycle trap).
        self.command(
            {"op": "create", "id": "rd", "type": "Read", "pos": [1200, -300]},
            {"op": "create", "id": "gr", "type": "Grade", "pos": [1200, 0]},
            {"op": "connect", "id": "gr", "input": "image", "source": "rd"},
            {"op": "create", "id": "bl", "type": "Blur", "pos": [1200, 300]},
            {"op": "connect", "id": "bl", "input": "image", "source": "gr"},
            {"op": "create", "id": "mg", "type": "Merge", "pos": [1800, 0]},
            {"op": "create", "id": "g2", "type": "Grade", "pos": [2300, 0]},
            {"op": "create", "id": "cn", "type": "Constant", "pos": [1800, -300]})

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()
        Preferences().set_wire_mode(graphlook.CURVED)

    # ---- helpers
    @property
    def nodes(self):
        return self.window.dispatcher.document["nodes"]

    def settle(self, rounds=10):
        for _ in range(rounds):
            APP.processEvents()

    def command(self, *commands):
        self.window.command({"op": "batch", "commands": list(commands)}, render=False)
        self.settle()

    def show(self, zoom):
        self.graph.zoom_to(zoom)
        self.graph.centerOn(P(1800, 0))
        self.settle()

    def at(self, scene_point, dx=0, dy=0):
        """The viewport point of a scene point, moved by whole screen pixels."""
        return self.graph.mapFromScene(scene_point) + QPoint(dx, dy)

    def output(self, key):
        return self.graph.items_by_id[key].output.scenePos()

    def input(self, key, slot="image"):
        return self.graph.items_by_id[key].inputs[slot].scenePos()

    def card_centre(self, key):
        item = self.graph.items_by_id[key]
        return item.mapToScene(item.rect().center())

    def press(self, point):
        QTest.mousePress(self.graph.viewport(), LEFT, NOTHING, point)

    def move(self, start, end, steps=5):
        for step in range(1, steps + 1):
            delta = end - start
            QTest.mouseMove(self.graph.viewport(), start + QPoint(delta.x() * step // steps, delta.y() * step // steps))
        self.settle(3)

    def hover(self, point):
        """A pointer move with no button down (QTest.mouseMove does not reach the graph without one)."""
        view = self.graph.viewport()
        event = QMouseEvent(QEvent.Type.MouseMove, QPointF(point), QPointF(view.mapToGlobal(point)),
                            Qt.MouseButton.NoButton, Qt.MouseButton.NoButton, NOTHING)
        QApplication.sendEvent(view, event)
        self.settle(3)

    def release(self, point):
        QTest.mouseRelease(self.graph.viewport(), LEFT, NOTHING, point)
        self.settle()

    def drag(self, start, end):
        self.press(start)
        self.move(start, end)
        self.release(end)

    def source(self, key, slot):
        return self.nodes[key]["inputs"].get(slot)

    # ---- magnetic drop
    def test_a_drop_thirty_screen_pixels_off_a_socket_connects(self):
        for zoom in ZOOMS:
            for dx, dy in ((0, -30), (14, -26), (-14, -26)):
                with self.subTest(zoom=zoom, offset=(dx, dy)):
                    self.command({"op": "connect", "id": "mg", "input": "B", "source": None})
                    self.show(zoom)
                    self.drag(self.at(self.output("cn")), self.at(self.input("mg", "B"), dx, dy))
                    self.assertEqual(self.source("mg", "B"), "cn")

    def test_a_drop_beyond_the_reach_is_dropped(self):
        for zoom in ZOOMS:
            with self.subTest(zoom=zoom):
                self.show(zoom)
                self.drag(self.at(self.output("cn")), self.at(self.input("mg", "B"), 0, -(WIRE_REACH + 25)))
                self.assertIsNone(self.source("mg", "B"))
                self.assertIsNone(self.graph.pending_edge)

    def bottom_middle(self, key, zoom):
        """The middle of the bottom edge of the card of `key`: 50 screen pixels or more from every socket at 100 percent and up."""
        item = self.graph.items_by_id[key]
        return item.mapToScene(item.rect().bottomLeft()) + P(item.rect().width() / 2, -3 / zoom)

    def test_a_drop_on_the_body_of_a_node_takes_its_first_free_input(self):
        for zoom in (1.0, 1.5):
            with self.subTest(zoom=zoom):
                self.command({"op": "connect", "id": "mg", "input": "A", "source": None},
                             {"op": "connect", "id": "mg", "input": "B", "source": None})
                self.show(zoom)
                self.drag(self.at(self.output("cn")), self.at(self.bottom_middle("mg", zoom)))
                self.assertEqual(self.source("mg", "A"), "cn")
                self.assertIsNone(self.source("mg", "B"))
                # The next wire dropped on the same body finds A taken and goes to B.
                self.drag(self.at(self.output("g2")), self.at(self.bottom_middle("mg", zoom)))
                self.assertEqual(self.source("mg", "B"), "g2")

    def test_zoomed_far_out_the_same_drop_still_connects_to_the_node(self):
        self.command({"op": "connect", "id": "mg", "input": "A", "source": None},
                     {"op": "connect", "id": "mg", "input": "B", "source": None})
        self.show(0.4)
        self.drag(self.at(self.output("cn")), self.at(self.bottom_middle("mg", 0.4)))
        self.assertIn("cn", (self.source("mg", "A"), self.source("mg", "B")))

    def test_a_drop_on_a_node_with_no_inputs_is_dropped(self):
        for zoom in ZOOMS:
            with self.subTest(zoom=zoom):
                self.show(zoom)
                self.drag(self.at(self.output("cn")), self.at(self.card_centre("rd")))   # a Read takes no wire
                self.assertIsNone(self.graph.pending_edge)
                self.assertEqual(self.source("gr", "image"), "rd")
                self.assertNotIn("cn", self.nodes["rd"]["inputs"].values())

    def test_a_dragged_input_dropped_on_a_body_takes_that_nodes_output(self):
        for zoom in ZOOMS:
            with self.subTest(zoom=zoom):
                self.command({"op": "connect", "id": "mg", "input": "A", "source": None})
                self.show(zoom)
                self.drag(self.at(self.input("mg", "A")), self.at(self.card_centre("cn")))
                self.assertEqual(self.source("mg", "A"), "cn")

    def test_between_two_nodes_the_nearer_compatible_socket_wins(self):
        for zoom in ZOOMS:
            with self.subTest(zoom=zoom):
                # A second Grade 60 screen pixels above the Merge: its input is 60 px above Merge.B.
                self.command({"op": "connect", "id": "mg", "input": "B", "source": None},
                             {"op": "create", "id": "g3", "type": "Grade", "pos": [1800, -60 / zoom]})
                self.show(zoom)
                nearer_merge = self.at(self.input("mg", "B"), 0, -25)
                nearer_grade = self.at(self.input("mg", "B"), 0, -35)
                self.drag(self.at(self.output("cn")), nearer_merge)
                self.assertEqual((self.source("mg", "B"), self.source("g3", "image")), ("cn", None))
                self.command({"op": "connect", "id": "mg", "input": "B", "source": None})
                self.drag(self.at(self.output("cn")), nearer_grade)
                self.assertEqual((self.source("mg", "B"), self.source("g3", "image")), (None, "cn"))
                self.command({"op": "delete", "id": "g3"})

    # ---- what never snaps
    def test_a_socket_that_would_close_a_cycle_never_takes_the_wire(self):
        self.show(1.0)
        # Blur feeds from Grade; Blur's output onto Grade's image would loop.
        self.press(self.at(self.output("bl")))
        self.move(self.at(self.output("bl")), self.at(self.input("gr")))
        self.assertIsNone(self.graph.wire_snap)
        self.release(self.at(self.input("gr")))
        self.assertEqual(self.source("gr", "image"), "rd")

    def test_a_node_never_snaps_to_itself(self):
        self.show(1.0)
        self.press(self.at(self.output("gr")))
        self.move(self.at(self.output("gr")), self.at(self.input("gr", "mask")))
        self.assertIsNone(self.graph.wire_snap)
        self.release(self.at(self.input("gr", "mask")))
        self.assertIsNone(self.source("gr", "mask"))

    def test_a_dragged_input_never_snaps_to_a_node_downstream_of_it(self):
        self.show(1.0)
        start = self.at(self.input("gr"))
        self.press(start)
        self.move(start, self.at(self.output("bl")))   # Blur sits downstream of Grade
        self.assertIsNone(self.graph.wire_snap)
        self.release(self.at(self.output("bl")))
        self.assertEqual(self.source("gr", "image"), "rd")

    # ---- the snap tells you
    def test_the_wire_snaps_lights_the_target_and_names_it(self):
        for zoom in ZOOMS:
            with self.subTest(zoom=zoom):
                self.show(zoom)
                start = self.at(self.output("cn"))
                self.press(start)
                self.move(start, self.at(self.input("mg", "B"), 0, -30))
                port = self.graph.items_by_id["mg"].inputs["B"]
                self.assertIs(self.graph.wire_snap, port)
                self.assertTrue(self.graph.pending_edge.highlighted)
                self.assertTrue(port._lit)
                self.assertTrue(self.graph.items_by_id["mg"].wire_lit)
                self.assertIn("to " + self.nodes["mg"]["name"] + ".B", self.window.statusBar().currentMessage())
                self.wait_for_snap()
                end = self.graph.pending_edge.path().currentPosition()
                self.assertLess((end - port.scenePos()).manhattanLength(), 1.5)
                # Leaving the reach lets the wire follow the pointer again and puts the lights out.
                self.move(self.at(self.input("mg", "B"), 0, -30), self.at(self.input("mg", "B"), 0, -WIRE_REACH - 80))
                self.assertIsNone(self.graph.wire_snap)
                self.assertFalse(self.graph.pending_edge.highlighted)
                self.assertFalse(port._lit)
                self.assertFalse(self.graph.items_by_id["mg"].wire_lit)
                self.release(self.at(self.input("mg", "B"), 0, -WIRE_REACH - 80))

    def wait_for_snap(self):
        self.assertTrue(wait_until(lambda: self.graph.snap_progress >= 1.0, 2000))

    def test_the_snap_is_quick(self):
        self.assertLessEqual(self.graph.snap_animation.duration(), 80)

    # ---- starting a wire
    def test_a_press_near_a_socket_off_the_card_starts_a_wire(self):
        for zoom in ZOOMS:
            with self.subTest(zoom=zoom):
                self.show(zoom)
                self.press(self.at(self.output("cn"), 0, GRAB_REACH - 4))   # below the card, inside the reach
                self.assertEqual(self.graph.wire_source, "cn")
                self.release(self.at(self.output("cn"), 0, GRAB_REACH + 4))
                self.graph.cancel_wire()

    def test_a_press_in_the_output_band_starts_an_output_wire(self):
        for zoom in ZOOMS:
            with self.subTest(zoom=zoom):
                self.show(zoom)
                item = self.graph.items_by_id["cn"]
                bottom_left = item.mapToScene(item.rect().bottomLeft()) + P(10 / zoom, 0)
                start = self.at(bottom_left, 0, -OUTPUT_BAND // 2)
                self.drag(start, self.at(self.input("mg", "B")))
                self.assertEqual(self.source("mg", "B"), "cn")
                self.command({"op": "connect", "id": "mg", "input": "B", "source": None})

    def test_a_press_deep_inside_a_card_still_drags_the_node(self):
        for zoom in ZOOMS:
            with self.subTest(zoom=zoom):
                self.show(zoom)
                before = list(self.nodes["g2"]["pos"])
                centre = self.at(self.card_centre("g2"))
                self.drag(centre, centre + QPoint(0, 40))
                self.assertIsNone(self.graph.pending_edge)
                self.assertNotEqual(self.nodes["g2"]["pos"], before)
                self.command({"op": "move", "id": "g2", "pos": before})

    def test_a_connected_input_is_still_picked_up_and_moved(self):
        for zoom in ZOOMS:
            with self.subTest(zoom=zoom):
                self.command({"op": "connect", "id": "mg", "input": "A", "source": "cn"})
                self.show(zoom)
                self.drag(self.at(self.input("mg", "A")), self.at(self.output("g2")))
                self.assertEqual(self.source("mg", "A"), "g2")

    def test_a_socket_near_the_pointer_draws_larger(self):
        self.show(1.0)
        port = self.graph.items_by_id["cn"].output
        self.hover(self.at(port.scenePos(), 0, 12))
        self.assertTrue(port._hot)
        self.assertGreater(port._visible.scale(), 1.0)
        self.hover(self.at(port.scenePos(), 300, 300))
        self.assertFalse(port._hot)

    # ---- cancelling
    def test_escape_cancels_the_wire_and_clears_the_lights(self):
        self.show(1.0)
        start = self.at(self.output("cn"))
        self.press(start)
        self.move(start, self.at(self.input("mg", "B"), 0, -10))
        self.assertIsNotNone(self.graph.wire_snap)
        QTest.keyClick(self.graph, Qt.Key.Key_Escape)
        self.settle()
        self.assertIsNone(self.graph.pending_edge)
        self.assertIsNone(self.graph.wire_snap)
        self.assertFalse(self.graph.items_by_id["mg"].wire_lit)
        self.release(self.at(self.input("mg", "B"), 0, -10))
        self.assertIsNone(self.source("mg", "B"))

    def test_a_picked_up_input_dragged_off_and_dropped_on_empty_space_disconnects(self):
        """Nuke's gesture (10/18): drag the input end of a wire away and let go on empty canvas."""
        for zoom in ZOOMS:
            with self.subTest(zoom=zoom):
                self.command({"op": "connect", "id": "gr", "input": "image", "source": "rd"})
                self.show(zoom)
                self.drag(self.at(self.input("gr")), self.at(P(1500, -150)))
                self.assertTrue(wait_until(lambda: self.source("gr", "image") is None))
                self.assertIsNone(self.graph.pending_edge)

    def test_a_click_or_tiny_drag_on_a_connected_input_changes_nothing(self):
        for zoom in ZOOMS:
            with self.subTest(zoom=zoom):
                self.command({"op": "connect", "id": "gr", "input": "image", "source": "rd"})
                self.show(zoom)
                start = self.at(self.input("gr"))
                self.drag(start, start + QPoint(2, 2))
                self.settle(20)
                self.assertEqual(self.source("gr", "image"), "rd")

    def test_escape_while_dragging_a_picked_up_input_keeps_its_connection(self):
        self.show(1.0)
        start = self.at(self.input("gr"))
        far = self.at(P(1500, -150))
        self.press(start)
        self.move(start, far)
        QTest.keyClick(self.graph, Qt.Key.Key_Escape)
        self.release(far)
        self.settle(20)
        self.assertEqual(self.source("gr", "image"), "rd")

if __name__ == "__main__":
    unittest.main()
