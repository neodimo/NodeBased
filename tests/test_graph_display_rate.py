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

    def test_the_body_is_cached_at_full_device_resolution(self):
        item = self.graph.items_by_id['rb']
        pixmap = item.body_pixmap(False, True, 1.0, 1.75)
        margin = item.BODY_MARGIN
        self.assertEqual(pixmap.devicePixelRatio(), 1.75)
        self.assertEqual(pixmap.width(), math.ceil((item.rect().width() + 2 * margin) * 1.75))
        self.assertEqual(pixmap.height(), math.ceil((item.rect().height() + 2 * margin) * 1.75))


if __name__ == '__main__':
    unittest.main()
