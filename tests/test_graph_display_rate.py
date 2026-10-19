"""The node graph stays at display rate with hundreds of nodes: a move re-routes only the wires on
the moved nodes, and node bodies are painted from a cache that is redrawn only when their look changes."""
import math
import os
import unittest
import uuid

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
import tests.isolation  # noqa: F401
from nodebased.app import Window
from tests.waiting import wait_until

APP = QApplication.instance() or QApplication([])


class GraphDisplayRateTests(unittest.TestCase):
    def setUp(self):
        self.window = Window(agent_name='nodebased-test-' + uuid.uuid4().hex)
        self.window.show()
        wait_until(lambda: self.window.frame is not None)
        APP.processEvents()
        # A chain a -> b -> c plus a second branch a -> d, away from the demo nodes.
        cmds = [{'op': 'create', 'type': 'Checker', 'id': 'ra', 'pos': [3000, 0]}]
        for key, src, y in (('rb', 'ra', 120), ('rc', 'rb', 240), ('rd', 'ra', 360)):
            cmds.append({'op': 'create', 'type': 'Grade', 'id': key, 'pos': [3000, y]})
            cmds.append({'op': 'connect', 'id': key, 'input': 'image', 'source': src})
        self.window.agent_command({'op': 'batch', 'commands': cmds})
        APP.processEvents()
        self.graph = self.window.graph

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def wire(self, source, destination):
        return next(edge for edge, src, dst, *_ in self.graph.edges if (src, dst) == (source, destination))

    def test_a_move_reroutes_only_the_wires_on_the_moved_node(self):
        graph = self.graph
        before = {id(edge): edge.routes for edge, *_ in graph.edges}
        item = graph.items_by_id['rb']
        item.setPos(item.pos().x() + 40, item.pos().y() + 10)
        changed = {id(edge) for edge, *_ in graph.edges if edge.routes != before[id(edge)]}
        attached = {id(self.wire('ra', 'rb')), id(self.wire('rb', 'rc'))}
        self.assertEqual(changed, attached)
        self.assertEqual(self.wire('ra', 'rb').routes, before[id(self.wire('ra', 'rb'))] + 1)
        # The re-routed wire ends on the moved node's input socket.
        port = item.inputs['image'].scenePos()
        end = self.wire('ra', 'rb').path().pointAtPercent(1.0)
        self.assertAlmostEqual(end.x(), port.x(), places=3)
        self.assertAlmostEqual(end.y(), port.y(), places=3)

    def test_a_full_update_keeps_paths_whose_ends_did_not_move(self):
        graph = self.graph
        before = {id(edge): edge.routes for edge, *_ in graph.edges}
        graph.update_edges()
        self.assertEqual({id(edge): edge.routes for edge, *_ in graph.edges}, before)
        # A wire mode change rebuilds every wire.
        other = 'right_angle' if graph.wire_mode != 'right_angle' else 'curved'
        graph.set_wire_mode(other, save=False)
        self.assertTrue(all(edge.routes == before[id(edge)] + 1 for edge, *_ in graph.edges))

    def frame_on(self, keys):
        graph = self.graph
        graph.resetTransform()
        graph.centerOn(graph.items_by_id[keys[0]].sceneBoundingRect().center())
        APP.processEvents()

    def test_a_cached_node_body_is_redrawn_only_when_its_look_changes(self):
        graph = self.graph
        NodeItem = type(graph.items_by_id['rb'])
        self.frame_on(['rb'])
        NodeItem._body_cache.clear()
        graph.viewport().repaint()
        first = NodeItem.body_renders
        self.assertGreater(len(NodeItem._body_cache), 0)
        graph.viewport().repaint()
        graph.viewport().repaint()
        self.assertEqual(NodeItem.body_renders, first, 'an unchanged graph redrew node bodies')
        graph.scene().clearSelection()
        graph.items_by_id['rb'].setSelected(True)
        graph.viewport().repaint()
        self.assertEqual(NodeItem.body_renders, first + 1, 'selecting one node should draw one new look')
        graph.items_by_id['rb'].setSelected(False)
        graph.viewport().repaint()
        self.assertEqual(NodeItem.body_renders, first + 1, 'the unselected look was still cached')

    def test_the_cached_body_matches_the_direct_drawing(self):
        graph = self.graph
        NodeItem = type(graph.items_by_id['rb'])
        self.frame_on(['rb'])
        cached = graph.viewport().grab().toImage()
        real = NodeItem.paint_family_body
        NodeItem.paint_family_body = lambda item, painter, option, widget=None: item.draw_family_body(
            painter, False, True)
        try:
            graph.viewport().repaint()
            direct = graph.viewport().grab().toImage()
        finally:
            NodeItem.paint_family_body = real
        item = graph.items_by_id['rb']
        box = graph.mapFromScene(item.sceneBoundingRect()).boundingRect().adjusted(2, 2, -2, -2)
        worst = 0
        for y in range(box.top(), box.bottom(), 2):
            for x in range(box.left(), box.right(), 2):
                a, b = cached.pixelColor(x, y), direct.pixelColor(x, y)
                worst = max(worst, abs(a.red() - b.red()), abs(a.green() - b.green()), abs(a.blue() - b.blue()))
        self.assertGreater(box.width() * box.height(), 1000)
        self.assertLessEqual(worst, 8, 'the cached node looks different from the drawn one')

    def test_cached_wires_match_the_drawn_ones(self):
        from PySide6.QtWidgets import QGraphicsItem
        graph = self.graph
        self.frame_on(['rb'])
        graph.viewport().repaint()
        cached = graph.viewport().grab().toImage()
        edges = [edge for edge, *_ in graph.edges]
        self.assertTrue(all(edge.cacheMode() == QGraphicsItem.CacheMode.DeviceCoordinateCache for edge in edges))
        for edge in edges:
            edge.setCacheMode(QGraphicsItem.CacheMode.NoCache)
        graph.viewport().repaint()
        direct = graph.viewport().grab().toImage()
        self.assertEqual(cached.size(), direct.size())
        wire = self.wire('ra', 'rb')
        box = graph.mapFromScene(wire.sceneBoundingRect()).boundingRect()
        ratio = cached.devicePixelRatio()
        worst = 0
        for y in range(int(box.top() * ratio), int(box.bottom() * ratio)):
            for x in range(int(box.left() * ratio), int(box.right() * ratio)):
                if 0 <= x < cached.width() and 0 <= y < cached.height():
                    a, b = cached.pixelColor(x, y), direct.pixelColor(x, y)
                    worst = max(worst, abs(a.red() - b.red()), abs(a.green() - b.green()), abs(a.blue() - b.blue()))
        self.assertGreater(box.width() * box.height(), 100)
        self.assertLessEqual(worst, 8, 'a cached wire looks different from a drawn one')

    def test_the_body_is_cached_at_full_device_resolution(self):
        item = self.graph.items_by_id['rb']
        pixmap = item.body_pixmap(False, True, 1.0, 1.75)
        margin = item.BODY_MARGIN
        self.assertEqual(pixmap.devicePixelRatio(), 1.75)
        self.assertEqual(pixmap.width(), math.ceil((item.rect().width() + 2 * margin) * 1.75))
        self.assertEqual(pixmap.height(), math.ceil((item.rect().height() + 2 * margin) * 1.75))

    def send_mouse(self, kind, point, buttons, button=None):
        from PySide6.QtCore import QEvent, QPointF, Qt
        from PySide6.QtGui import QMouseEvent
        viewport = self.graph.viewport()
        local = QPointF(point)
        event = QMouseEvent({'press': QEvent.Type.MouseButtonPress, 'move': QEvent.Type.MouseMove,
                             'release': QEvent.Type.MouseButtonRelease}[kind], local,
                            QPointF(viewport.mapToGlobal(point)), button or Qt.MouseButton.NoButton, buttons,
                            Qt.KeyboardModifier.NoModifier)
        APP.sendEvent(viewport, event)

    def test_wires_are_plain_only_while_a_node_is_dragged(self):
        from PySide6.QtCore import QPoint, Qt
        graph = self.graph
        self.frame_on(['rb'])
        item = graph.items_by_id['rb']
        start = graph.mapFromScene(item.mapToScene(item.rect().center()))
        left = Qt.MouseButton.LeftButton
        self.assertFalse(getattr(graph.scene(), 'fast_wires', False))
        self.send_mouse('press', start, left, left)
        for step in range(1, 4):
            self.send_mouse('move', start + QPoint(6 * step, 0), left)
        self.assertTrue(graph.scene().fast_wires, 'a node drag should draw wires without smoothing')
        self.send_mouse('release', start + QPoint(18, 0), Qt.MouseButton.NoButton, left)
        self.assertFalse(graph.scene().fast_wires, 'letting go should bring the smooth wires back')

    def test_zoomed_out_nodes_are_flat_cards_without_text(self):
        from PySide6.QtGui import QImage, QPainter, QColor
        from PySide6.QtWidgets import QStyleOptionGraphicsItem
        import nodebased.app as appmodule
        item = self.graph.items_by_id['rb']
        title = next(child for child in item.childItems() if isinstance(child, appmodule.NodeText))

        def ink(zoom):
            image = QImage(400, 200, QImage.Format.Format_ARGB32)
            image.fill(QColor(0, 0, 0, 0))
            painter = QPainter(image)
            painter.scale(zoom, zoom)
            option = QStyleOptionGraphicsItem()
            title.paint(painter, option, None)
            painter.end()
            return sum(1 for y in range(0, 200, 2) for x in range(0, 400, 2) if image.pixelColor(x, y).alpha())
        self.assertGreater(ink(1.0), 0)
        self.assertEqual(ink(appmodule.GRAPH_DETAIL_ZOOM * 0.9), 0)
        flat, full = [], []
        real_flat, real_full = appmodule.NodeItem.draw_flat_body, appmodule.NodeItem.draw_family_body
        appmodule.NodeItem.draw_flat_body = lambda node, *a: (flat.append(node.key), real_flat(node, *a))
        appmodule.NodeItem.draw_family_body = lambda node, *a: (full.append(node.key), real_full(node, *a))
        try:
            appmodule.NodeItem._body_cache.clear()
            self.graph.resetTransform()
            self.graph.scale(0.3, 0.3)
            self.graph.centerOn(item)
            self.graph.viewport().repaint()
        finally:
            appmodule.NodeItem.draw_flat_body, appmodule.NodeItem.draw_family_body = real_flat, real_full
        self.assertIn('rb', flat)
        self.assertEqual(full, [])


if __name__ == '__main__':
    unittest.main()
