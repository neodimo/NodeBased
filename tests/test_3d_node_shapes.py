"""3D nodes in the graph: rounded cards, and full circles for scene, light and camera."""
import math
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF
from PySide6.QtGui import QImage, QPainter
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from nodebased.app import (CIRCLE_DIAMETER, CIRCLE_TYPES, NODE_HEIGHT, NODE_WIDTH, Window,
                           circle_port_positions, node_form)
from nodebased.core import SPECS

APP = QApplication.instance() or QApplication([])


def wait_until(condition, timeout=20.0):
    waited = 0
    while waited < timeout * 1000:
        if condition():
            return True
        QTest.qWait(10)
        waited += 10
    return False


class NodeFormTests(unittest.TestCase):
    def test_every_3d_type_is_rounded_or_circular_and_2d_types_stay_cards(self):
        for kind in SPECS:
            form = node_form({"type": kind})
            if kind == "Dot":
                self.assertEqual(form, "dot")
            elif kind in CIRCLE_TYPES:
                self.assertEqual(form, "circle", kind)
            elif kind.endswith("3D"):
                self.assertEqual(form, "round", kind)
            else:
                self.assertEqual(form, "card", kind)
        for kind in ("Scene3D", "Light3D", "Camera3D"):
            self.assertIn(kind, CIRCLE_TYPES)

    def test_rim_sockets_sit_on_the_circle_and_do_not_crowd(self):
        radius = CIRCLE_DIAMETER / 2
        points = circle_port_positions(8)
        for x, y, _ in points:
            self.assertAlmostEqual(math.hypot(x - radius, y - radius), radius, places=6)
            self.assertLessEqual(y, radius)  # inputs stay on the upper half, the output owns the bottom
        for (ax, ay, _), (bx, by, _) in zip(points, points[1:]):
            self.assertGreaterEqual(math.hypot(bx - ax, by - ay), 20)  # a 12 px socket plus clear air
        self.assertAlmostEqual(circle_port_positions(1)[0][0], radius)
        self.assertAlmostEqual(circle_port_positions(1)[0][1], 0.0)


class NodeShapeTests(unittest.TestCase):
    def setUp(self):
        self.window = Window()
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def create(self, kind):
        before = set(self.window.graph.items_by_id)
        self.window.graph.scene().clearSelection()
        self.window.add_node(kind, position=QPointF(2000 + 400 * len(before), 2000))
        APP.processEvents()
        (key,) = set(self.window.graph.items_by_id) - before
        return key

    def item(self, kind):
        # The graph rebuilds its items on every document change, so look the item up last.
        key = self.create(kind)
        return self.window.graph.items_by_id[key]

    def painted(self, item):
        """Render one node alone on a transparent canvas and return its alpha as a lookup."""
        rect = item.boundingRect()
        image = QImage(int(rect.width()) + 2, int(rect.height()) + 2, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(0)
        painter = QPainter(image)
        painter.translate(-rect.left() + 1, -rect.top() + 1)
        from PySide6.QtWidgets import QStyleOptionGraphicsItem
        item.paint(painter, QStyleOptionGraphicsItem())
        painter.end()
        return lambda x, y: image.pixelColor(int(x - rect.left() + 1), int(y - rect.top() + 1)).alpha()

    def test_scene_light_and_camera_are_full_circles(self):
        for kind in ("Scene3D", "Light3D", "Camera3D"):
            item = self.item(kind)
            rect = item.rect()
            self.assertEqual((rect.width(), rect.height()), (CIRCLE_DIAMETER, CIRCLE_DIAMETER), kind)
            alpha = self.painted(item)
            self.assertGreater(alpha(rect.center().x(), rect.center().y()), 200, kind)
            for corner in ((3, 3), (rect.width() - 3, 3), (3, rect.height() - 3), (rect.width() - 3, rect.height() - 3)):
                self.assertEqual(alpha(*corner), 0, f"{kind} paints into the corner {corner}")
                self.assertFalse(item.shape().contains(QPointF(*corner)), f"{kind} corner {corner} grabs clicks")
            self.assertTrue(item.shape().contains(rect.center()))

    def test_scene_sockets_are_on_the_rim_with_the_output_at_the_bottom(self):
        item = self.item("Scene3D")
        radius = CIRCLE_DIAMETER / 2
        self.assertEqual(len(item.inputs), 8)
        for port in item.inputs.values():
            self.assertAlmostEqual(math.hypot(port.pos().x() - radius, port.pos().y() - radius), radius, places=4)
        self.assertEqual((item.output.pos().x(), item.output.pos().y()), (radius, CIRCLE_DIAMETER))

    def test_title_fits_inside_the_circle(self):
        item = self.item("ReadAlembicCamera3D")  # the longest name that gets a circle
        from PySide6.QtWidgets import QGraphicsSimpleTextItem
        title = next(child for child in item.childItems() if isinstance(child, QGraphicsSimpleTextItem))
        bounds = title.mapRectToParent(title.boundingRect())
        self.assertGreaterEqual(bounds.left(), 0)
        self.assertLessEqual(bounds.right(), CIRCLE_DIAMETER)

    def test_other_3d_nodes_are_rounded_and_2d_nodes_keep_square_corners(self):
        sphere_key, grade_key = self.create("Sphere3D"), self.create("Grade")
        sphere, grade = self.window.graph.items_by_id[sphere_key], self.window.graph.items_by_id[grade_key]
        self.assertEqual((sphere.rect().width(), sphere.rect().height()), (NODE_WIDTH, NODE_HEIGHT))
        round_alpha, card_alpha = self.painted(sphere), self.painted(grade)
        self.assertEqual(round_alpha(2, 2), 0)
        self.assertFalse(sphere.shape().contains(QPointF(2, 2)))
        self.assertGreater(round_alpha(NODE_WIDTH / 2, NODE_HEIGHT / 2), 200)
        self.assertGreater(card_alpha(2, 2), 200)
        self.assertTrue(grade.shape().contains(QPointF(2, 2)))


if __name__ == "__main__":
    unittest.main()
