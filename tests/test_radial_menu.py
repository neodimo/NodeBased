"""The graph's context-sensitive radial menu (plan "Radial menu", DiMo 9/27): the command model
and rules table are pure functions (`radialrules.py`), tested directly; the ring gesture itself
(hold Q, flick or tap) is tested through the real `Graph` widget."""
import math
import os
import unittest
import uuid

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtTest import QTest

from nodebased.core import Dispatcher
from nodebased.radialrules import CONTEXTS, SLOT_COUNT, commands_for, context_for_selection
from tests.test_desktop import APP, Window, wait_until


def build_nodes(*commands):
    """Real, fully-registered node dicts (correct "inputs" shape for every kind) for the rules
    tests, the same way test_groups.py's `build` helper does for the engine."""
    dispatcher = Dispatcher()
    dispatcher.execute({"op": "batch", "commands": list(commands)})
    return dispatcher.document["nodes"]


class RadialRulesTests(unittest.TestCase):
    """Pure-function tests: no Qt widget needed for the context and slot logic."""

    def setUp(self):
        self.nodes = build_nodes(
            {"op": "create", "id": "src", "type": "Checker"},
            {"op": "create", "id": "grade", "type": "Grade"},
            {"op": "create", "id": "grade2", "type": "Grade"},
            {"op": "create", "id": "key", "type": "Keyer"},
            {"op": "create", "id": "scene", "type": "Scene3D"},
        )

    def test_six_contexts(self):
        n = self.nodes
        self.assertEqual(context_for_selection(n, []), "empty")
        self.assertEqual(context_for_selection(n, ["grade"]), "one_image")
        self.assertEqual(context_for_selection(n, ["key"]), "keyer")
        self.assertEqual(context_for_selection(n, ["grade", "grade2"]), "two_nodes")
        self.assertEqual(context_for_selection(n, ["scene"]), "3d")
        self.assertEqual(context_for_selection(n, ["grade", "scene"]), "3d")
        self.assertEqual(context_for_selection(n, ["src", "grade", "key"]), "several")

    def test_every_context_has_exactly_eight_slots(self):
        for name, table in CONTEXTS.items():
            self.assertEqual(len(table), SLOT_COUNT, name)

    def test_one_image_context_shows_the_expected_commands_in_the_expected_slots(self):
        ids = [c.id if c else None for c in commands_for(self.nodes, ["grade"])]
        self.assertEqual(ids, ["add_grade", "add_merge", "add_blur", "add_transform",
                               "view", "bypass", "backdrop", "group"])

    def test_when_false_leaves_a_gap_without_moving_its_neighbours(self):
        # Checker is a pure generator (core.bypass_slot returns None for it), so its Bypass slice
        # must not show -- but the slots on either side of it keep their own positions.
        ids = [c.id if c else None for c in commands_for(self.nodes, ["src"])]
        self.assertEqual(ids[4], "view")
        self.assertIsNone(ids[5])
        self.assertEqual(ids[6], "backdrop")

    def test_keyer_context_is_picked_over_one_image_for_a_keyer_node(self):
        ids = [c.id if c else None for c in commands_for(self.nodes, ["key"])]
        self.assertEqual(ids[0], "add_premult")

    def test_a_3d_node_wins_the_3d_context_even_alongside_2d_nodes(self):
        ids = [c.id if c else None for c in commands_for(self.nodes, ["grade", "scene"])]
        self.assertEqual(ids[0], "add_scene3d")

    def test_two_nodes_context_offers_align_and_both_view_slots(self):
        ids = [c.id if c else None for c in commands_for(self.nodes, ["grade", "grade2"])]
        self.assertIn("align", ids)
        self.assertIn("view_a", ids)
        self.assertIn("view_b", ids)

    def test_several_context_hides_ungroup_with_no_group_selected(self):
        ids = [c.id if c else None for c in commands_for(self.nodes, ["src", "grade", "key"])]
        self.assertEqual(ids[0], "group")
        self.assertIsNone(ids[1])   # "ungroup"'s own slot: no Group node in this selection

    def test_several_context_shows_ungroup_once_a_group_is_in_the_selection(self):
        nodes = build_nodes(
            {"op": "create", "id": "src", "type": "Checker"},
            {"op": "create", "id": "grade", "type": "Grade"},
            {"op": "create", "id": "grp", "type": "Group"})
        ids = [c.id if c else None for c in commands_for(nodes, ["src", "grade", "grp"])]
        self.assertEqual(ids[1], "ungroup")


def _slot_offset(index, distance=60):
    """The (dx, dy) a flick toward slot `index`'s centre direction looks like, matching
    radialmenu._slot_center_angle: slot 0 is straight up, slots run clockwise from there."""
    angle = math.radians(-90 + index * (360.0 / SLOT_COUNT))
    return distance * math.cos(angle), distance * math.sin(angle)


class RadialMenuGestureTests(unittest.TestCase):
    """One window for the whole class; each test uses its own node id prefix."""

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
        graph = self.window.graph
        if graph.radial_menu.is_open():
            graph.radial_menu.close_menu()

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

    def open_menu_at(self, scene_pos):
        graph = self.window.graph
        graph.setFocus()
        QTest.mouseMove(graph.viewport(), graph.mapFromScene(scene_pos))
        APP.processEvents()
        QTest.keyPress(graph, Qt.Key.Key_Q)
        APP.processEvents()
        self.assertTrue(graph.radial_menu.is_open())

    def flick_and_release(self, dx, dy):
        graph = self.window.graph
        center = graph.radial_menu.center
        QTest.mouseMove(graph.viewport(), QPoint(int(center.x() + dx), int(center.y() + dy)))
        APP.processEvents()
        QTest.keyRelease(graph, Qt.Key.Key_Q)
        APP.processEvents()

    def tap_release(self):
        QTest.keyRelease(self.window.graph, Qt.Key.Key_Q)
        APP.processEvents()

    def click_menu_at(self, dx, dy):
        graph = self.window.graph
        center = graph.radial_menu.center
        QTest.mouseClick(graph.viewport(), Qt.MouseButton.LeftButton,
                         pos=QPoint(int(center.x() + dx), int(center.y() + dy)))
        APP.processEvents()

    # -- flick: direction only, runs on release ------------------------------------------------

    def test_flick_up_with_nothing_selected_adds_a_checker_node(self):
        # Slot 0 (up) in the empty context is "Add Read", but Read alone then opens a real file
        # browser (Window.browse_read) -- exactly as R does today outside this menu -- so this
        # flicks to slot 1 (Checker) instead, a plain node create with no dialog behind it.
        self.select()
        before = set(self.doc()['nodes'])
        revision = self.window.dispatcher.revision
        self.open_menu_at(self.window.graph_center() + QPointF(1200, 1200))
        dx, dy = _slot_offset(1)
        self.flick_and_release(dx, dy)
        self.assertFalse(self.window.graph.radial_menu.is_open())
        added = set(self.doc()['nodes']) - before
        self.assertEqual(len(added), 1)
        self.assertEqual(self.doc()['nodes'][next(iter(added))]['type'], 'Checker')
        # One undo step puts the graph back exactly as it was.
        self.assertEqual(self.window.dispatcher.revision, revision + 1)
        self.window.command({'op': 'undo'})
        self.assertEqual(set(self.doc()['nodes']), before)

    def test_flick_right_with_a_grade_selected_adds_grade_wired_below_it(self):
        self.command({'op': 'create', 'id': 'rg1', 'type': 'Grade', 'pos': [3000, 3000]})
        self.select('rg1')
        before = set(self.doc()['nodes'])
        self.open_menu_at(self.window.graph.items_by_id['rg1'].sceneBoundingRect().center())
        dx, dy = _slot_offset(0)   # slot 0: "Add Grade" in the one_image context
        self.flick_and_release(dx, dy)
        added = list(set(self.doc()['nodes']) - before)
        self.assertEqual(len(added), 1)
        new_id = added[0]
        self.assertEqual(self.doc()['nodes'][new_id]['type'], 'Grade')
        self.assertEqual(self.doc()['nodes'][new_id]['inputs']['image'], 'rg1')

    def test_releasing_inside_the_dead_zone_with_no_prior_flick_pins_the_menu_open(self):
        self.select()
        self.open_menu_at(self.window.graph_center() + QPointF(-1200, 1200))
        before = set(self.doc()['nodes'])
        self.tap_release()
        graph = self.window.graph
        self.assertTrue(graph.radial_menu.is_open())
        self.assertTrue(graph.radial_menu.sustained)
        self.assertEqual(set(self.doc()['nodes']), before)

    def test_a_pinned_menu_runs_the_slice_clicked(self):
        self.select()
        self.open_menu_at(self.window.graph_center() + QPointF(-1200, -1200))
        self.tap_release()
        before = set(self.doc()['nodes'])
        dx, dy = _slot_offset(2)   # "Add Constant" in the empty context
        self.click_menu_at(dx, dy)
        self.assertFalse(self.window.graph.radial_menu.is_open())
        added = set(self.doc()['nodes']) - before
        self.assertEqual(len(added), 1)
        self.assertEqual(self.doc()['nodes'][next(iter(added))]['type'], 'Constant')

    def test_a_pinned_menu_clicked_in_the_dead_zone_cancels(self):
        self.select()
        self.open_menu_at(self.window.graph_center() + QPointF(1200, -1200))
        self.tap_release()
        before = set(self.doc()['nodes'])
        self.click_menu_at(0, 0)
        self.assertFalse(self.window.graph.radial_menu.is_open())
        self.assertEqual(set(self.doc()['nodes']), before)

    def test_escape_cancels_a_pinned_menu(self):
        self.select()
        self.open_menu_at(self.window.graph_center() + QPointF(600, -1600))
        self.tap_release()
        self.assertTrue(self.window.graph.radial_menu.is_open())
        QTest.keyClick(self.window.graph, Qt.Key.Key_Escape)
        APP.processEvents()
        self.assertFalse(self.window.graph.radial_menu.is_open())

    def test_flicking_toward_a_slot_whose_when_is_false_runs_nothing(self):
        # A source node alone has no Bypass slot (core.bypass_slot is None for it): flicking
        # toward slot 5 must cancel, not raise or silently run a neighbour's command.
        self.command({'op': 'create', 'id': 'rg2', 'type': 'Checker', 'pos': [3600, 3000]})
        self.select('rg2')
        before = set(self.doc()['nodes'])
        self.open_menu_at(self.window.graph.items_by_id['rg2'].sceneBoundingRect().center())
        dx, dy = _slot_offset(5)
        self.flick_and_release(dx, dy)
        self.assertFalse(self.window.graph.radial_menu.is_open())
        self.assertEqual(set(self.doc()['nodes']), before)

    def test_align_selection_is_one_undo_step(self):
        self.command(
            {'op': 'create', 'id': 'al1', 'type': 'Constant', 'pos': [4000, 3000]},
            {'op': 'create', 'id': 'al2', 'type': 'Grade', 'pos': [4200, 3200]})
        self.select('al1', 'al2')
        revision = self.window.dispatcher.revision
        before = {k: list(self.doc()['nodes'][k]['pos']) for k in ('al1', 'al2')}
        self.window.graph.align_selection()
        self.assertEqual(self.window.dispatcher.revision, revision + 1)
        self.assertEqual(self.doc()['nodes']['al2']['pos'][0], self.doc()['nodes']['al1']['pos'][0])
        self.window.command({'op': 'undo'})
        after = {k: list(self.doc()['nodes'][k]['pos']) for k in ('al1', 'al2')}
        self.assertEqual(after, before)
