"""The graph's context-sensitive radial menu (plan "Radial menu", DiMo 9/27): the command model
and rules table are pure functions (`radialrules.py`), tested directly; the ring gesture itself
(hold Q, flick or tap) is tested through the real `Graph` widget. Local-usage learning and
pinning (deliverable R3) live in `app.Preferences`, exercised directly, and layered onto
`commands_for` -- see `radialrules._apply_slot_overrides`."""
import json
import math
import os
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QEvent, QPoint, QPointF, QSettings, Qt
from PySide6.QtTest import QTest

from nodebased import radialcommands, radialrules
from nodebased.app import Preferences
from nodebased.core import Dispatcher
from nodebased.radialrules import CONTEXTS, SLOT_COUNT, commands_for, context_for_selection
from tests.test_desktop import APP, Window
from tests.waiting import wait_until


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


class FakeRadialPreferences:
    """Just enough of `app.Preferences`' radial-learning surface for a pure-function test of
    `commands_for`'s override layering, with no QSettings involved."""

    def __init__(self, learned=None, pins=None):
        self._learned = learned or {}
        self._pins = pins or {}

    def radial_bucket(self, context_kind, node_type):
        return f"{context_kind}|{node_type or ''}"

    def radial_learned(self):
        return self._learned

    def radial_pins(self):
        return self._pins


class RadialSlotOverrideTests(unittest.TestCase):
    """`radialrules.builtin_ids_for_context` and `_apply_slot_overrides`, the pure machinery
    behind learned and pinned slots (deliverable R3), plus `commands_for`'s own layering order."""

    def setUp(self):
        self.nodes = build_nodes(
            {"op": "create", "id": "src", "type": "Checker"},
            {"op": "create", "id": "grade", "type": "Grade"},
            {"op": "create", "id": "grade2", "type": "Grade"},
            {"op": "create", "id": "key", "type": "Keyer"})

    def test_builtin_ids_for_context_lists_every_id_regardless_of_when(self):
        self.assertEqual(radialrules.builtin_ids_for_context("one_image"),
                         {"add_grade", "add_merge", "add_blur", "add_transform",
                          "view", "bypass", "backdrop", "group"})

    def test_apply_slot_overrides_replaces_only_the_named_slot(self):
        table = commands_for(self.nodes, ["grade", "grade2"])
        self.assertEqual(table[2].id, "group")
        selection = [self.nodes["grade"], self.nodes["grade2"]]
        overridden = radialrules._apply_slot_overrides(table, {"2": "add_dissolve"}, "two_nodes",
                                                        selection, [])
        ids = [c.id if c else None for c in overridden]
        self.assertEqual(ids[2], "add_dissolve")
        self.assertEqual(ids[0], "add_merge")     # untouched neighbour
        self.assertEqual(ids[1], "add_dissolve")  # its own, separate slot: also untouched

    def test_apply_slot_overrides_ignores_an_id_not_available_in_this_context(self):
        table = list(commands_for(self.nodes, ["src"]))
        overridden = radialrules._apply_slot_overrides(table, {"5": "add_scene3d"}, "one_image",
                                                        [self.nodes["src"]], [])
        self.assertIsNone(overridden[5])   # "add_scene3d" only exists in the 3d table

    def test_apply_slot_overrides_skips_a_candidate_whose_own_when_is_false(self):
        table = [None] * SLOT_COUNT
        selection = [self.nodes["src"], self.nodes["grade"], self.nodes["key"]]
        overridden = radialrules._apply_slot_overrides(table, {"1": "ungroup"}, "several",
                                                        selection, [])
        self.assertIsNone(overridden[1])   # no Group node selected: "ungroup"'s own when says no

    def test_commands_for_applies_a_pin_after_a_learned_slot_so_the_pin_wins(self):
        prefs = FakeRadialPreferences(learned={"two_nodes|": {"2": "add_dissolve"}},
                                      pins={"two_nodes|": {"2": "align"}})
        ids = [c.id if c else None for c in commands_for(self.nodes, ["grade", "grade2"], prefs)]
        self.assertEqual(ids[2], "align")

    def test_commands_for_with_no_preferences_behaves_exactly_as_before(self):
        self.assertEqual([c.id if c else None for c in commands_for(self.nodes, ["grade"])],
                         [c.id if c else None for c in commands_for(self.nodes, ["grade"], None)])


class RadialUsageLearningTests(unittest.TestCase):
    """`app.Preferences`' local-usage learning engine (deliverable R3, part 1): counting picks per
    context+type bucket, promoting a favourite once it clears the threshold, and resetting."""

    PREF_KEYS = ('interface/radial_usage', 'interface/radial_learned_slots',
                'interface/radial_pinned_slots', 'interface/radial_learning_enabled')

    def setUp(self):
        for key in self.PREF_KEYS:
            QSettings('NodeBased', 'NodeBased').remove(key)
        self.prefs = Preferences()

    def tearDown(self):
        for key in self.PREF_KEYS:
            QSettings('NodeBased', 'NodeBased').remove(key)

    def test_repeated_picks_promote_a_command_into_the_lowest_priority_slot(self):
        for _ in range(Preferences.RADIAL_LEARN_THRESHOLD):
            self.prefs.record_radial_usage("several", None, "user:add_note")
        self.assertEqual(self.prefs.radial_learned()["several|"], {"7": "user:add_note"})

    def test_a_second_favourite_takes_the_next_slot_without_moving_the_first(self):
        for _ in range(Preferences.RADIAL_LEARN_THRESHOLD):
            self.prefs.record_radial_usage("several", None, "user:add_note")
        for _ in range(Preferences.RADIAL_LEARN_THRESHOLD):
            self.prefs.record_radial_usage("several", None, "user:second_favourite")
        self.assertEqual(self.prefs.radial_learned()["several|"],
                         {"7": "user:add_note", "6": "user:second_favourite"})

    def test_below_threshold_promotes_nothing_yet(self):
        for _ in range(Preferences.RADIAL_LEARN_THRESHOLD - 1):
            self.prefs.record_radial_usage("several", None, "user:add_note")
        self.assertEqual(self.prefs.radial_learned(), {})

    def test_a_command_already_guaranteed_by_the_rules_is_never_promoted(self):
        for _ in range(20):
            self.prefs.record_radial_usage("one_image", "Grade", "add_grade")
        self.assertEqual(self.prefs.radial_learned(), {})

    def test_a_pinned_slot_is_never_claimed_by_a_new_promotion(self):
        self.prefs.toggle_radial_pin("several", None, 7, "existing_pin")
        for _ in range(Preferences.RADIAL_LEARN_THRESHOLD):
            self.prefs.record_radial_usage("several", None, "user:add_note")
        self.assertEqual(self.prefs.radial_learned()["several|"], {"6": "user:add_note"})

    def test_disabling_learning_stops_new_promotions(self):
        self.prefs.set_radial_learning_enabled(False)
        for _ in range(Preferences.RADIAL_LEARN_THRESHOLD):
            self.prefs.record_radial_usage("several", None, "user:add_note")
        self.assertEqual(self.prefs.radial_learned(), {})

    def test_toggle_pin_twice_unpins(self):
        self.assertTrue(self.prefs.toggle_radial_pin("one_image", "Grade", 0, "add_grade"))
        self.assertEqual(self.prefs.radial_pins()["one_image|Grade"], {"0": "add_grade"})
        self.assertFalse(self.prefs.toggle_radial_pin("one_image", "Grade", 0, "add_grade"))
        self.assertEqual(self.prefs.radial_pins(), {})

    def test_reset_clears_usage_and_learned_slots_but_not_pins(self):
        self.prefs.toggle_radial_pin("several", None, 7, "pinned_cmd")
        for _ in range(Preferences.RADIAL_LEARN_THRESHOLD):
            self.prefs.record_radial_usage("several", None, "user:add_note")
        self.prefs.reset_radial_learning()
        self.assertEqual(self.prefs.radial_learned(), {})
        self.assertEqual(self.prefs.radial_usage(), {})
        self.assertEqual(self.prefs.radial_pins(), {"several|": {"7": "pinned_cmd"}})


class RadialLearningRenderTests(unittest.TestCase):
    """Local-usage learning applied through `commands_for` end to end: a user command that keeps
    losing the ordinary free-slot race gets a fixed slot of its own once it is picked enough,
    without disturbing the commands already sitting in the other free slots."""

    PREF_KEYS = RadialUsageLearningTests.PREF_KEYS

    def setUp(self):
        for key in self.PREF_KEYS:
            QSettings('NodeBased', 'NodeBased').remove(key)
        self.prefs = Preferences()
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        # A Group in the selection shows "ungroup" (slot 1); a bypassable Grade shows "bypass
        # all" (slot 4) -- between the always-shown group/backdrop/align (0, 2, 3) that leaves
        # exactly three free slots (5, 6, 7) for the fixture below to compete over.
        self.nodes = build_nodes(
            {"op": "create", "id": "n1", "type": "Checker"},
            {"op": "create", "id": "n2", "type": "Grade"},
            {"op": "create", "id": "n3", "type": "Group"})
        self.ids = ["n1", "n2", "n3"]

    def tearDown(self):
        for key in self.PREF_KEYS:
            QSettings('NodeBased', 'NodeBased').remove(key)

    def _write_always_on_command(self, name):
        (self.tmp / f"{name}.json").write_text(json.dumps(
            {"label": name, "when": {}, "slot": None, "enabled": True, "ops": []}))

    def test_a_command_that_loses_the_free_slot_race_gets_promoted_and_evicts_the_current_occupant(self):
        for name in ("a", "b", "c", "d"):
            self._write_always_on_command(name)
        with mock.patch.object(radialcommands, "user_commands_directory", return_value=self.tmp):
            # Before learning: only three free slots exist ("several"'s 5, 6, 7), so alphabetical
            # file order fills them a, b, c -- "d" never gets a slot at all.
            before = [c.id if c else None for c in commands_for(self.nodes, self.ids, self.prefs)]
            self.assertEqual(before[5:8], ["user:a", "user:b", "user:c"])

            for _ in range(Preferences.RADIAL_LEARN_THRESHOLD):
                self.prefs.record_radial_usage("several", None, "user:d")

            after = [c.id if c else None for c in commands_for(self.nodes, self.ids, self.prefs)]
            self.assertEqual(after[5], "user:a")   # untouched
            self.assertEqual(after[6], "user:b")   # untouched
            self.assertEqual(after[7], "user:d")   # "d" claimed the lowest-priority slot from "c"

    def test_a_pinned_slice_is_never_displaced_by_a_new_promotion(self):
        self.prefs.toggle_radial_pin("several", None, 7, "align")
        for _ in range(Preferences.RADIAL_LEARN_THRESHOLD):
            self.prefs.record_radial_usage("several", None, "user:mystery")
        with mock.patch.object(radialcommands, "user_commands_directory", return_value=self.tmp):
            ids = [c.id if c else None for c in commands_for(self.nodes, self.ids, self.prefs)]
        self.assertEqual(ids[7], "align")


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
        # A scene-space offset of 1,200 units lands outside the docked graph's visible viewport.
        # QTest then can't deliver the flick move there (the small dock layout used by the
        # desktop suite made this ordering-dependent). Start in a visible, empty corner instead.
        viewport = self.window.graph.viewport()
        start = QPoint(viewport.width() - 80, viewport.height() - 35)
        self.open_menu_at(self.window.graph.mapToScene(start))
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

    # -- deliverable R3: pinning and the trigger-key setting ------------------------------------

    def test_right_click_pins_a_sustained_slice_and_a_second_right_click_unpins_it(self):
        self.addCleanup(lambda: QSettings('NodeBased', 'NodeBased').remove('interface/radial_pinned_slots'))
        self.select()
        self.open_menu_at(self.window.graph_center() + QPointF(-1800, 1800))
        self.tap_release()
        graph = self.window.graph
        center = graph.radial_menu.center
        dx, dy = _slot_offset(2)   # "Add Constant" in the empty context
        pos = QPoint(int(center.x() + dx), int(center.y() + dy))
        QTest.mouseClick(graph.viewport(), Qt.MouseButton.RightButton, pos=pos)
        APP.processEvents()
        self.assertTrue(graph.radial_menu.is_open())   # a right-click never closes the ring
        self.assertEqual(self.window.preferences.radial_pins().get("empty|", {}).get("2"), "add_constant")
        QTest.mouseClick(graph.viewport(), Qt.MouseButton.RightButton, pos=pos)
        APP.processEvents()
        self.assertNotIn("empty|", self.window.preferences.radial_pins())
        graph.radial_menu.close_menu()

    def test_changing_the_trigger_key_takes_effect_immediately(self):
        prefs = self.window.preferences
        self.addCleanup(prefs.set_radial_trigger_key, Qt.Key.Key_Q)
        prefs.set_radial_trigger_key(Qt.Key.Key_W)
        graph = self.window.graph
        self.select()
        graph.setFocus()
        QTest.keyPress(graph, Qt.Key.Key_Q)
        APP.processEvents()
        self.assertFalse(graph.radial_menu.is_open())   # Q no longer opens it
        QTest.keyPress(graph, Qt.Key.Key_W)
        APP.processEvents()
        self.assertTrue(graph.radial_menu.is_open())
        graph.radial_menu.close_menu()
        QTest.keyRelease(graph, Qt.Key.Key_W)
        APP.processEvents()
