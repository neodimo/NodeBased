"""A click on a node highlights it before the Properties panel is built, and a move does not rebuild it."""
import os
import unittest
import uuid

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
import tests.isolation  # noqa: F401
from nodebased.app import Window
from tests.waiting import wait_until

APP = QApplication.instance() or QApplication([])


class GraphInputLatencyTests(unittest.TestCase):
    def setUp(self):
        self.window = Window(agent_name='nodebased-test-' + uuid.uuid4().hex)
        self.window.show()
        wait_until(lambda: self.window.frame is not None)
        APP.processEvents()
        self.keys = [k for k, n in self.window.graph_nodes().items() if n['type'] not in ('Backdrop',)]
        self.calls = []
        real = self.window.build_node_panel

        def counted(key):
            self.calls.append(key)
            return real(key)
        self.window.build_node_panel = counted

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def test_a_press_does_not_build_the_panel_synchronously(self):
        w = self.window
        key = self.keys[0]
        self.calls.clear()
        w.graph.items_by_id[key].setSelected(True)
        self.assertTrue(w.graph.items_by_id[key].isSelected())
        self.assertEqual(self.calls, [], 'the panel was built inside the selection change')
        APP.processEvents()
        self.assertEqual(self.calls, [key])

    def test_reselecting_a_node_reuses_its_panel(self):
        w = self.window
        a, b = self.keys[0], self.keys[1]
        for key in (a, b, a):
            w.graph.scene().clearSelection()
            w.graph.items_by_id[key].setSelected(True)
            APP.processEvents()
        self.assertEqual(self.calls, [a, b])
        self.assertIsNotNone(w.properties.widget())

    def test_an_edit_invalidates_the_cached_panel(self):
        w = self.window
        a, b = self.keys[0], self.keys[1]
        for key in (a, b):
            w.graph.scene().clearSelection()
            w.graph.items_by_id[key].setSelected(True)
            APP.processEvents()
        node = w.graph_nodes()[a]
        param = next(iter(node['params']))
        w.command({'op': 'set', 'id': a, 'param': param, 'value': node['params'][param]})
        w.command({'op': 'rename', 'id': a, 'name': 'Renamed'})
        self.calls.clear()
        w.graph.scene().clearSelection()
        w.graph.items_by_id[a].setSelected(True)
        APP.processEvents()
        self.assertEqual(self.calls, [a])

    def test_a_move_does_not_rebuild_the_panel(self):
        w = self.window
        key = self.keys[0]
        w.graph.items_by_id[key].setSelected(True)
        APP.processEvents()
        self.calls.clear()
        x, y = w.graph_nodes()[key]['pos']
        w.command({'op': 'move', 'id': key, 'pos': [x + 40, y + 20]})
        w.command({'op': 'batch', 'commands': [{'op': 'move', 'id': key, 'pos': [x + 80, y]}]})
        APP.processEvents()
        self.assertEqual(self.calls, [])
        self.assertEqual(w.graph_nodes()[key]['pos'], [x + 80, y])
        # A param edit on the selected node still rebuilds it.
        param = next(iter(w.graph_nodes()[key]['params']), None)
        if param is not None:
            w.command({'op': 'rename', 'id': key, 'name': 'Moved'})
            self.assertEqual(self.calls, [key])

    def test_a_drag_release_does_not_rebuild_the_graph_but_an_agent_move_does(self):
        w = self.window
        key = self.keys[0]
        rebuilds = []
        real = w.graph.rebuild
        w.graph.rebuild = lambda: (rebuilds.append(1), real())[1]
        item = w.graph.items_by_id[key]
        item.setPos(item.pos().x() + 30, item.pos().y() + 10)   # what the drag did
        w.graph.commit_moves([{'op': 'move', 'id': key, 'pos': [item.pos().x(), item.pos().y()]}], [])
        self.assertEqual(rebuilds, [])
        w.command({'op': 'move', 'id': key, 'pos': [500, 500]})
        self.assertEqual(len(rebuilds), 1)
        self.assertEqual((w.graph.items_by_id[key].pos().x(), w.graph.items_by_id[key].pos().y()), (500, 500))


if __name__ == '__main__':
    unittest.main()
