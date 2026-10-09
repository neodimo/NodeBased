"""New look, step 3: dragging a free node onto a wire inserts it into the stream (Lane 2, NL3).

A node with no connections that is dropped on a wire is spliced in: the wire's source feeds the node's
main input and the node's output feeds the wire's old destination input (a mask input included). A
node that has any connection just moves, and so does a node with no input or no output. One undo step
puts the wire and the node's position back. The wire lights up while the node hovers over it."""
import tests.isolation  # noqa: F401  (settings of the test run, not the developer's)

import unittest

from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest

from nodebased import graphlook
from nodebased.app import Preferences, Window
from tests.test_desktop import APP, wait_until

P = QPointF
LEFT, NOTHING = Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier


class WireInsertTests(unittest.TestCase):
    def setUp(self):
        Preferences().set_wire_mode(graphlook.CURVED)
        self.window = Window()
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))
        self.graph = self.window.graph
        # A corner of the graph of its own: Read -> Grade, and a mask wire from a Constant into the Grade.
        self.command(
            {"op": "create", "id": "rd", "type": "Read", "pos": [1200, -300]},
            {"op": "create", "id": "gr", "type": "Grade", "pos": [1200, -100]},
            {"op": "connect", "id": "gr", "input": "image", "source": "rd"},
            {"op": "create", "id": "ms", "type": "Constant", "pos": [1500, -300]},
            {"op": "connect", "id": "gr", "input": "mask", "source": "ms"},
            {"op": "create", "id": "bl", "type": "Blur", "pos": [900, 150]})

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()
        Preferences().set_wire_mode(graphlook.CURVED)

    # ---- helpers
    @property
    def nodes(self):
        return self.window.dispatcher.document["nodes"]

    def settle(self):
        for _ in range(10):
            APP.processEvents()

    def command(self, *commands):
        self.window.command({"op": "batch", "commands": list(commands)}, render=False)
        self.settle()

    def edge_record(self, source, target, slot):
        return next(record for record in self.graph.edges if record[1:4] == (source, target, slot))

    def centre(self, key):
        item = self.graph.items_by_id[key]
        return item.mapToScene(item.rect().center())

    def viewport_point(self, scene_point):
        return self.graph.mapFromScene(scene_point)

    def show(self, *scene_points, zoom=1.0):
        self.graph.zoom_to(zoom)
        self.graph.centerOn(sum((p for p in scene_points), P(0, 0)) / len(scene_points))
        self.settle()

    def grab(self, key):
        """Press on the card of `key`; the viewport is left with the button held."""
        QTest.mousePress(self.graph.viewport(), LEFT, NOTHING, self.viewport_point(self.centre(key)))

    def carry(self, key, to, steps=4):
        """Move the held card so that its centre ends at the scene point `to`."""
        start = self.centre(key)
        for step in range(1, steps + 1):
            QTest.mouseMove(self.graph.viewport(), self.viewport_point(start + (to - start) * step / steps))
        self.settle()

    def drop(self, key, to):
        self.grab(key)
        self.carry(key, to)
        QTest.mouseRelease(self.graph.viewport(), LEFT, NOTHING, self.viewport_point(to))
        self.settle()

    def wire_middle(self, source, target, slot):
        return self.edge_record(source, target, slot)[0].handle

    def lit_wires(self):
        return [(source, target, slot) for edge, source, target, slot, _out in self.graph.edges if edge.highlighted]

    # ---- the insert
    def test_a_free_blur_dropped_on_read_to_grade_goes_between_them(self):
        middle = self.wire_middle("rd", "gr", "image")
        self.show(middle, self.centre("bl"))
        self.drop("bl", middle)
        self.assertEqual(self.nodes["bl"]["inputs"]["image"], "rd")
        self.assertEqual(self.nodes["gr"]["inputs"]["image"], "bl")
        self.assertEqual(self.nodes["gr"]["inputs"]["mask"], "ms")   # the mask wire is untouched
        self.assertIsNone(self.nodes["bl"]["inputs"]["mask"])

    def test_dropped_on_a_mask_wire_the_mask_input_is_kept(self):
        middle = self.wire_middle("ms", "gr", "mask")
        self.show(middle, self.centre("bl"))
        self.drop("bl", middle)
        self.assertEqual(self.nodes["bl"]["inputs"]["image"], "ms")
        self.assertEqual(self.nodes["gr"]["inputs"]["mask"], "bl")
        self.assertEqual(self.nodes["gr"]["inputs"]["image"], "rd")

    def test_the_card_stays_where_it_was_dropped(self):
        middle = self.wire_middle("rd", "gr", "image")
        self.show(middle, self.centre("bl"))
        self.drop("bl", middle)
        item = self.graph.items_by_id["bl"]
        self.assertLess((item.mapToScene(item.rect().center()) - middle).manhattanLength(), 2)

    # ---- what does not insert
    def test_a_connected_node_dragged_over_a_wire_only_moves(self):
        self.command({"op": "connect", "id": "bl", "input": "image", "source": "ms"})
        middle = self.wire_middle("rd", "gr", "image")
        self.show(middle, self.centre("bl"))
        self.grab("bl")
        self.carry("bl", middle)
        self.assertEqual(self.lit_wires(), [])
        QTest.mouseRelease(self.graph.viewport(), LEFT, NOTHING, self.viewport_point(middle))
        self.settle()
        self.assertEqual(self.nodes["gr"]["inputs"]["image"], "rd")
        self.assertEqual(self.nodes["bl"]["inputs"]["image"], "ms")
        self.assertNotEqual(self.nodes["bl"]["pos"], [900, 150])   # it did move

    def test_a_node_that_feeds_something_is_connected_too(self):
        self.command({"op": "create", "id": "g2", "type": "Grade", "pos": [900, 400]},
                     {"op": "connect", "id": "g2", "input": "image", "source": "bl"})
        middle = self.wire_middle("rd", "gr", "image")
        self.show(middle, self.centre("bl"))
        self.drop("bl", middle)
        self.assertEqual(self.nodes["gr"]["inputs"]["image"], "rd")
        self.assertIsNone(self.nodes["bl"]["inputs"]["image"])

    def test_a_read_a_viewer_and_a_write_do_not_insert(self):
        self.command({"op": "create", "id": "r2", "type": "Read", "pos": [900, 150 + 100]},
                     {"op": "create", "id": "vw", "type": "Viewer", "pos": [900, 150 + 200]},
                     {"op": "create", "id": "wr", "type": "Write", "pos": [900, 150 + 300]})
        middle = self.wire_middle("rd", "gr", "image")
        for key in ("r2", "vw", "wr"):
            self.show(middle, self.centre(key))
            self.grab(key)
            self.carry(key, middle)
            self.assertEqual(self.lit_wires(), [], key)
            QTest.mouseRelease(self.graph.viewport(), LEFT, NOTHING, self.viewport_point(middle))
            self.settle()
            self.assertEqual(self.nodes["gr"]["inputs"]["image"], "rd", key)
            self.assertNotIn(key, [n for node in self.nodes.values() for n in node["inputs"].values()], key)

    def test_a_node_resting_on_a_wire_is_not_inserted_by_a_click(self):
        middle = self.wire_middle("rd", "gr", "image")
        self.command({"op": "move", "id": "bl", "pos": [middle.x() - 95, middle.y() - 26]})
        self.show(middle)
        self.grab("bl")
        QTest.mouseRelease(self.graph.viewport(), LEFT, NOTHING, self.viewport_point(self.centre("bl")))
        self.settle()
        self.assertEqual(self.nodes["gr"]["inputs"]["image"], "rd")

    # ---- undo
    def test_one_undo_restores_the_wire_and_the_position(self):
        before = self.window.dispatcher.revision
        middle = self.wire_middle("rd", "gr", "image")
        self.show(middle, self.centre("bl"))
        self.drop("bl", middle)
        self.assertEqual(self.nodes["gr"]["inputs"]["image"], "bl")
        self.window.command({"op": "undo"}, render=False)
        self.settle()
        self.assertEqual(self.nodes["gr"]["inputs"]["image"], "rd")
        self.assertIsNone(self.nodes["bl"]["inputs"]["image"])
        self.assertEqual(self.nodes["bl"]["pos"], [900, 150])
        self.assertEqual(self.graph.items_by_id["bl"].pos(), P(900, 150))
        self.assertIsNotNone(before)

    # ---- the highlight
    def test_the_wire_lights_up_under_the_card_and_clears_when_it_leaves(self):
        middle = self.wire_middle("rd", "gr", "image")
        self.show(middle, self.centre("bl"))
        self.assertEqual(self.lit_wires(), [])
        self.grab("bl")
        self.carry("bl", self.centre("bl"))                       # still far from every wire
        self.assertEqual(self.lit_wires(), [])
        self.carry("bl", middle)
        self.assertEqual(self.lit_wires(), [("rd", "gr", "image")])
        self.carry("bl", middle + P(-300, 120))
        self.assertEqual(self.lit_wires(), [])
        self.carry("bl", middle)
        self.assertEqual(self.lit_wires(), [("rd", "gr", "image")])
        QTest.mouseRelease(self.graph.viewport(), LEFT, NOTHING, self.viewport_point(middle))
        self.settle()
        self.assertEqual(self.lit_wires(), [])                    # and the rebuilt graph has none lit

    def test_the_wire_nearest_the_card_wins_when_two_run_under_it(self):
        middle = self.wire_middle("rd", "gr", "image")
        # The mask wire runs past to the right of the main wire; a card on the main wire lights only that one.
        self.show(middle, self.centre("bl"))
        self.grab("bl")
        self.carry("bl", middle)
        self.assertEqual(len(self.lit_wires()), 1)
        QTest.mouseRelease(self.graph.viewport(), LEFT, NOTHING, self.viewport_point(middle))
        self.settle()

    def test_a_lit_wire_looks_different_from_a_plain_one(self):
        from PySide6.QtGui import QImage, QPainter
        from PySide6.QtWidgets import QStyleOptionGraphicsItem
        edge = self.edge_record("rd", "gr", "image")[0]

        def paint():
            rect = edge.boundingRect()
            image = QImage(int(rect.width()) + 2, int(rect.height()) + 2, QImage.Format.Format_ARGB32_Premultiplied)
            image.fill(0)
            painter = QPainter(image)
            painter.translate(-rect.left() + 1, -rect.top() + 1)
            edge.paint(painter, QStyleOptionGraphicsItem())
            painter.end()
            return image
        plain = paint()
        edge.set_highlight(True)
        lit = paint()
        edge.set_highlight(False)
        self.assertNotEqual(plain, lit)
        self.assertEqual(plain, paint())

    # ---- both wire modes, any zoom
    def test_it_works_in_both_wire_modes_at_any_zoom(self):
        for mode in graphlook.WIRE_MODES:
            for zoom in (0.4, 1.0, 2.0):
                with self.subTest(mode=mode, zoom=zoom):
                    self.command({"op": "move", "id": "bl", "pos": [900, 150]},
                                 {"op": "connect", "id": "gr", "input": "image", "source": "rd"})
                    self.graph.set_wire_mode(mode, save=False)
                    middle = self.wire_middle("rd", "gr", "image")
                    self.show(middle, self.centre("bl"), zoom=zoom)
                    self.grab("bl")
                    self.carry("bl", middle)
                    self.assertEqual(self.lit_wires(), [("rd", "gr", "image")])
                    QTest.mouseRelease(self.graph.viewport(), LEFT, NOTHING, self.viewport_point(middle))
                    self.settle()
                    self.assertEqual(self.nodes["bl"]["inputs"]["image"], "rd")
                    self.assertEqual(self.nodes["gr"]["inputs"]["image"], "bl")
                    self.window.command({"op": "undo"}, render=False)
                    self.settle()
                    self.assertEqual(self.nodes["gr"]["inputs"]["image"], "rd")


if __name__ == "__main__":
    unittest.main()
