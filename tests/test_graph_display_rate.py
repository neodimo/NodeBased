"""The node graph stays at display rate with hundreds of nodes: a move re-routes only the wires on
the moved nodes, and node bodies are painted from a cache that is redrawn only when their look changes."""
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


if __name__ == '__main__':
    unittest.main()
