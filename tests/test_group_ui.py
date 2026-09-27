"""The Group UI in the node graph (step S2 of the script-structure plan): Ctrl+G, Ctrl+Shift+G,
entering a group by double-click, the Root > Group breadcrumb bar and the group's properties."""
import copy
import os
import unittest
import uuid

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import Qt, QEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLineEdit, QPushButton, QLabel

from nodebased.imaging import Evaluator
from nodebased.app import LabelEdit
from tests.test_desktop import APP, Window, wait_until

CTRL = Qt.KeyboardModifier.ControlModifier
SHIFT = Qt.KeyboardModifier.ShiftModifier


class GroupUiTests(unittest.TestCase):
    """One window for the whole class; each test names its nodes with a prefix of its own."""

    @classmethod
    def setUpClass(cls):
        cls.window = Window(agent_name='nodebased-test-' + uuid.uuid4().hex)
        cls.window.show()
        assert wait_until(lambda: cls.window.frame is not None)

    @classmethod
    def tearDownClass(cls):
        cls.window.saved_document = cls.window.dispatcher.document
        cls.window.close()
        APP.processEvents()
        cls.window.deleteLater()
        APP.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        APP.processEvents()
        del cls.window

    def tearDown(self):
        self.window.go_to_depth(0)
        APP.processEvents()

    # -- helpers ------------------------------------------------------------------------------

    def command(self, *commands):
        result = self.window.command({'op': 'batch', 'commands': list(commands)})
        self.assertIsNotNone(result, self.window.last_command_error)
        return result

    def doc(self):
        return self.window.dispatcher.document

    def select(self, *keys):
        graph = self.window.graph
        graph.scene().clearSelection()
        for key in keys:
            graph.items_by_id[key].setSelected(True)
        APP.processEvents()
        graph.setFocus()

    def keys(self, key, modifiers):
        QTest.keyClick(self.window.graph, key, modifiers)
        APP.processEvents()

    def crumbs(self):
        return [b.text() for b in self.window.breadcrumbs.findChildren(QPushButton)]

    def double_click(self, key):
        graph = self.window.graph
        center = graph.items_by_id[key].sceneBoundingRect().center()
        graph.centerOn(center)
        APP.processEvents()
        QTest.mouseDClick(graph.viewport(), Qt.MouseButton.LeftButton, pos=graph.mapFromScene(center))
        APP.processEvents()

    def chain(self, prefix, x=2000, y=2000):
        self.command(
            {'op': 'create', 'id': prefix + 'a', 'type': 'Constant', 'pos': [x, y]},
            {'op': 'create', 'id': prefix + 'b', 'type': 'Grade', 'pos': [x, y + 150]},
            {'op': 'connect', 'id': prefix + 'b', 'input': 'image', 'source': prefix + 'a'},
            {'op': 'create', 'id': prefix + 'c', 'type': 'Blur', 'pos': [x, y + 300]},
            {'op': 'connect', 'id': prefix + 'c', 'input': 'image', 'source': prefix + 'b'})

    def group_of(self, *members):
        groups = [k for k, n in self.doc()['nodes'].items() if n['type'] == 'Group']
        return next(k for k in groups if set(self.doc()['nodes'][k]['graph']['nodes']) >= set(members))

    # -- Ctrl+G and Ctrl+Shift+G --------------------------------------------------------------

    def test_ctrl_g_groups_the_selection_and_ctrl_shift_g_puts_the_layout_back(self):
        self.chain('g1')
        before = copy.deepcopy({k: self.doc()['nodes'][k] for k in ('g1a', 'g1b', 'g1c')})
        self.select('g1b', 'g1c')
        self.keys(Qt.Key.Key_G, CTRL)
        group = self.group_of('g1b', 'g1c')
        nodes = self.doc()['nodes']
        self.assertNotIn('g1b', nodes)
        self.assertNotIn('g1c', nodes)
        self.assertEqual(nodes[group]['inputs'], {'in1': 'g1a'})
        self.assertTrue(nodes[group]['name'].startswith('Group'))
        self.assertIn(group, self.window.graph.items_by_id)
        self.assertNotIn('g1b', self.window.graph.items_by_id)
        self.assertEqual(self.window.graph.selected_id(), group)
        self.assertEqual(self.window.graph_path, [])

        self.keys(Qt.Key.Key_G, CTRL | SHIFT)
        after = {k: self.doc()['nodes'][k] for k in ('g1a', 'g1b', 'g1c')}
        self.assertEqual(after, before)
        self.assertNotIn(group, self.doc()['nodes'])
        self.assertEqual(self.window.graph.items_by_id['g1c'].pos().x(), before['g1c']['pos'][0])
        self.assertEqual(self.window.graph.items_by_id['g1c'].pos().y(), before['g1c']['pos'][1])

    def test_grouping_nothing_and_ungrouping_a_plain_node_only_report(self):
        self.chain('g2')
        self.select()
        revision = self.window.dispatcher.revision
        self.keys(Qt.Key.Key_G, CTRL)
        self.select('g2a')
        self.keys(Qt.Key.Key_G, CTRL | SHIFT)
        self.assertEqual(self.window.dispatcher.revision, revision)

    def test_a_new_group_is_named_after_the_groups_already_there(self):
        self.chain('g3', 2400)
        self.chain('g4', 2800)
        self.select('g3b')
        self.keys(Qt.Key.Key_G, CTRL)
        first = self.group_of('g3b')
        self.select('g4b')
        self.keys(Qt.Key.Key_G, CTRL)
        second = self.group_of('g4b')
        names = {self.doc()['nodes'][first]['name'], self.doc()['nodes'][second]['name']}
        self.assertEqual(len(names), 2)

    # -- entering, editing, leaving -----------------------------------------------------------

    def test_enter_a_group_edit_a_knob_inside_go_back_and_the_viewer_shows_it(self):
        w = self.window
        self.assertTrue(wait_until(lambda: w.frame is not None))
        self.select('plate', 'grade')
        self.keys(Qt.Key.Key_G, CTRL)
        group = self.group_of('plate', 'grade')
        name = self.doc()['nodes'][group]['name']
        self.assertTrue(wait_until(lambda: np.array_equal(w.frame, Evaluator().evaluate(self.doc()))))
        unchanged = w.frame.copy()

        self.double_click(group)
        self.assertEqual(w.graph_path, [group])
        self.assertEqual(self.crumbs(), ['Root', name])
        inside = w.graph.items_by_id
        self.assertEqual(set(inside), set(self.doc()['nodes'][group]['graph']['nodes']))
        self.assertTrue({'plate', 'grade'} <= set(inside))
        self.assertNotIn('merge', inside)
        kinds = {self.doc()['nodes'][group]['graph']['nodes'][k]['type'] for k in inside}
        self.assertTrue({'Input', 'Output'} & kinds >= {'Output'})

        self.select('grade')
        self.assertTrue(w.properties.findChildren(QLineEdit))
        w.commit_param('grade', 'exposure', 2.0)
        self.assertTrue(wait_until(
            lambda: self.doc()['nodes'][group]['graph']['nodes']['grade']['params']['exposure'] == 2.0))
        self.assertNotIn('grade', self.doc()['nodes'])

        back = w.breadcrumbs.findChildren(QPushButton)[0]
        self.assertTrue(back.isEnabled())
        back.click()
        APP.processEvents()
        self.assertEqual(w.graph_path, [])
        self.assertEqual(self.crumbs(), ['Root'])
        self.assertIn('merge', w.graph.items_by_id)
        self.assertTrue(wait_until(lambda: np.array_equal(w.frame, Evaluator().evaluate(self.doc()))
                                   and not np.array_equal(w.frame, unchanged)))

    def test_groups_nest_and_each_level_has_a_crumb(self):
        w = self.window
        self.chain('n1', 3200)
        self.select('n1b', 'n1c')
        self.keys(Qt.Key.Key_G, CTRL)
        outer = self.group_of('n1b', 'n1c')
        self.double_click(outer)
        self.select('n1b')
        self.keys(Qt.Key.Key_G, CTRL)
        self.assertEqual(w.graph_path, [outer])
        inner = next(k for k, n in self.doc()['nodes'][outer]['graph']['nodes'].items() if n['type'] == 'Group')
        self.double_click(inner)
        self.assertEqual(w.graph_path, [outer, inner])
        self.assertEqual(len(self.crumbs()), 3)
        w.breadcrumbs.findChildren(QPushButton)[1].click()
        APP.processEvents()
        self.assertEqual(w.graph_path, [outer])
        self.assertEqual(len(self.crumbs()), 2)
        # Ctrl+Shift+G inside a group ungroups the inner one and leaves the outer alone.
        self.select(inner)
        self.keys(Qt.Key.Key_G, CTRL | SHIFT)
        self.assertNotIn(inner, self.doc()['nodes'][outer]['graph']['nodes'])
        self.assertIn('n1b', self.doc()['nodes'][outer]['graph']['nodes'])

    def test_one_undo_takes_the_group_back_and_an_edit_inside_is_one_undo_step(self):
        w = self.window
        self.chain('u1', 3600)
        self.select('u1b')
        self.keys(Qt.Key.Key_G, CTRL)
        group = self.group_of('u1b')
        self.double_click(group)
        revision = w.dispatcher.revision
        self.command({'op': 'set', 'id': 'u1b', 'param': 'exposure', 'value': 1.5})
        self.assertEqual(w.dispatcher.revision, revision + 1)
        w.command({'op': 'undo'})
        self.assertEqual(self.doc()['nodes'][group]['graph']['nodes']['u1b']['params']['exposure'],
                         self.window.dispatcher.execute({'op': 'describe'})['nodes']['Grade']['params']['exposure'])
        self.assertEqual(w.graph_path, [group])
        # Undoing the grouping itself removes the group the graph is showing: back to the top level.
        w.command({'op': 'undo'})
        self.assertEqual(w.graph_path, [])
        self.assertIn('u1b', self.doc()['nodes'])
        self.assertIn('u1b', w.graph.items_by_id)
        self.assertEqual(self.crumbs(), ['Root'])

    # -- the group's own panel ----------------------------------------------------------------

    def test_the_group_panel_shows_its_name_and_a_note(self):
        w = self.window
        self.chain('p1', 4000)
        self.select('p1b', 'p1c')
        self.keys(Qt.Key.Key_G, CTRL)
        group = self.group_of('p1b', 'p1c')
        self.select(group)
        self.assertTrue(wait_until(lambda: w.properties.findChild(QLabel, 'group-summary') is not None))
        summary = w.properties.findChild(QLabel, 'group-summary')
        self.assertIn('2 nodes inside', summary.text())
        self.assertIn('1 input', summary.text())
        names = [e.text() for e in w.properties.findChildren(QLineEdit)]
        self.assertIn(self.doc()['nodes'][group]['name'], names)
        note = w.properties.findChild(LabelEdit, 'group-note')
        note.setPlainText('Sky cleanup')
        note.finished.emit()
        self.assertTrue(wait_until(lambda: self.doc()['nodes'][group].get('label') == 'Sky cleanup'))
        # The note draws on the node, under its name.
        self.assertTrue(wait_until(lambda: any(
            'Sky cleanup' in getattr(child, 'text', lambda: '')()
            for child in self.window.graph.items_by_id[group].childItems())))
        enter = w.properties.findChild(QPushButton, 'enter-group')
        enter.click()
        APP.processEvents()
        self.assertEqual(w.graph_path, [group])


if __name__ == '__main__':
    unittest.main()
