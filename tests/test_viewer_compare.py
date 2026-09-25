"""Viewer inputs 1-9, A/B buffers, compare modes and the wipe (offscreen Qt)."""
import copy
import os
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from nodebased import compare as compare_model
from nodebased.app import SHORTCUT_SECTIONS, STYLE, Window
from nodebased.core import COMPARE_MODES, Dispatcher, empty_document, load_document, viewer_state

APP = QApplication.instance() or QApplication([])
APP.setStyle("Fusion")
APP.setStyleSheet(STYLE)

RED, GREEN, BLUE = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)


def wait_until(condition, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        APP.processEvents()
        if condition():
            return True
        QTest.qWait(10)
    return False


def constant_document(colours=(("a", RED), ("b", BLUE), ("c", GREEN)), view="a"):
    dispatcher = Dispatcher(empty_document())
    for index, (key, (r, g, b)) in enumerate(colours):
        dispatcher.execute({"op": "create", "id": key, "type": "Constant", "pos": [index * 120, 0],
                            "params": {"width": 160, "height": 80, "red": r, "green": g, "blue": b,
                                       "alpha": 1.0}})
    dispatcher.execute({"op": "view", "id": view})
    return dispatcher.document


class ViewerCompareBase(unittest.TestCase):
    document = staticmethod(constant_document)

    def setUp(self):
        self.window = Window(self.document())
        self.window.resize(900, 700)
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None
                                   and self.window.viewer.format_rect is not None))
        self.viewer = self.window.viewer

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def settle(self):
        self.assertTrue(wait_until(lambda: not self.window.busy and not len(self.window.preview_queue)
                                   and self.window.frame_generation == self.window.generation
                                   and self.window.rendered_identity == self.window.render_identity()))
        APP.processEvents()

    def press_graph_key(self, node, key):
        graph = self.window.graph
        graph.scene().clearSelection()
        graph.items_by_id[node].setSelected(True)
        QTest.keyClick(graph, key)
        self.settle()

    def compare(self, b, mode):
        self.window.command({"op": "viewer_input", "slot": b, "id": {1: "a", 2: "b", 3: "c"}[b],
                             "activate": False})
        self.window.command({"op": "viewer_compare", "b": b, "mode": mode})
        self.settle()

    def screen(self, fx, fy):
        """Viewport pixel colour at a fraction of the format rectangle."""
        rect = self.viewer.format_rect
        point = self.viewer.mapFromScene(QPointF(rect.left() + fx * rect.width(),
                                                 rect.top() + fy * rect.height()))
        image = self.viewer.viewport().grab().toImage()
        colour = image.pixelColor(point)
        return colour.red(), colour.green(), colour.blue()

    def scene_point(self, fx, fy):
        rect = self.viewer.format_rect
        return self.viewer.mapFromScene(QPointF(rect.left() + fx * rect.width(),
                                                rect.top() + fy * rect.height()))


class InputTests(ViewerCompareBase):
    def test_keys_one_to_three_assign_inputs_and_strip_reflects_it(self):
        for node, key in (("a", Qt.Key.Key_1), ("b", Qt.Key.Key_2), ("c", Qt.Key.Key_3)):
            self.press_graph_key(node, key)
        state = viewer_state(self.window.dispatcher.document)
        self.assertEqual(state["inputs"][:4], ["a", "b", "c", None])
        self.assertEqual(state["active"], 3)
        self.assertEqual(self.window.dispatcher.document["view"], "c")
        self.assertEqual(self.viewer.input_strip.states(),
                         ["wired", "wired", "active"] + ["empty"] * 6)

    def test_switching_inputs_shows_the_other_nodes_pixels(self):
        self.press_graph_key("b", Qt.Key.Key_2)
        self.assertEqual(tuple(self.window.frame[0, 0][:3]), BLUE)
        QTest.keyClick(self.viewer, Qt.Key.Key_1)
        self.settle()
        self.assertEqual(tuple(self.window.frame[0, 0][:3]), RED)
        self.assertEqual(self.viewer.input_strip.states()[:2], ["active", "wired"])
        # An empty input is ignored, and clicking a wired button in the strip switches too.
        QTest.keyClick(self.viewer, Qt.Key.Key_7)
        self.assertEqual(self.window.dispatcher.document["view"], "a")
        QTest.mouseClick(self.viewer.input_strip.buttons[1], Qt.MouseButton.LeftButton)
        self.settle()
        self.assertEqual(tuple(self.window.frame[0, 0][:3]), BLUE)

    def test_key_one_still_views_the_selected_node(self):
        self.press_graph_key("c", Qt.Key.Key_1)
        self.assertEqual(self.window.dispatcher.document["view"], "c")
        self.assertEqual(viewer_state(self.window.dispatcher.document)["inputs"][0], "c")

    def test_alt_number_sets_b_and_turns_the_compare_on(self):
        self.press_graph_key("b", Qt.Key.Key_2)
        QTest.keyClick(self.viewer, Qt.Key.Key_1)
        QTest.keyClick(self.viewer, Qt.Key.Key_2, Qt.KeyboardModifier.AltModifier)
        self.settle()
        state = viewer_state(self.window.dispatcher.document)
        self.assertEqual((state["b"], state["compare"]), (2, "wipe"))
        self.assertEqual(self.viewer.input_strip.states()[:2], ["active", "wired"])
        self.assertIn(" b", self.viewer.input_strip._states[1])
        self.assertIsNotNone(self.window.frame_b)
        QTest.keyClick(self.viewer, Qt.Key.Key_0, Qt.KeyboardModifier.AltModifier)
        self.settle()
        self.assertIsNone(viewer_state(self.window.dispatcher.document)["b"])
        self.assertIsNone(self.window.frame_b)


class CompareModeTests(ViewerCompareBase):
    def test_wipe_at_fifty_percent_shows_a_left_and_b_right(self):
        only_a = (self.screen(0.25, 0.5), self.screen(0.75, 0.5))
        self.assertEqual(only_a[0], only_a[1])
        self.compare(2, "B only")
        only_b = self.screen(0.25, 0.5)
        self.assertNotEqual(only_b, only_a[0])
        self.compare(2, "wipe")
        self.assertIsNotNone(self.viewer.wipe_pixmap)
        self.assertEqual(self.screen(0.25, 0.5), only_a[0])
        self.assertEqual(self.screen(0.75, 0.5), only_b)

    def test_wipe_drag_does_not_evaluate_the_graph(self):
        self.compare(2, "wipe")
        generation, undo_depth = self.window.generation, len(self.window.dispatcher.undo_stack)
        start = self.scene_point(0.5, 0.5)
        target = self.scene_point(0.8, 0.5)
        QTest.mousePress(self.viewer.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
        QTest.mouseMove(self.viewer.viewport(), target, 20)
        QTest.mouseRelease(self.viewer.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, target)
        APP.processEvents()
        self.assertAlmostEqual(self.viewer.wipe["x"], 0.8, delta=0.02)
        self.assertEqual(self.window.generation, generation)
        self.assertEqual(len(self.window.dispatcher.undo_stack), undo_depth)
        # The split moved: 0.7 is now on A's side.
        self.compare(2, "B only")
        only_b = self.screen(0.7, 0.5)
        self.compare(2, "wipe")
        self.assertNotEqual(self.screen(0.7, 0.5), only_b)

    def test_rotating_the_wipe_moves_the_split(self):
        self.compare(2, "wipe")
        before = (self.screen(0.4, 0.25), self.screen(0.4, 0.75))
        self.assertEqual(before[0], before[1])
        centre, _, _, handle = self.viewer._wipe_scene_geometry()
        start = self.viewer.mapFromScene(handle)
        # Drag the rotation handle a quarter turn: from straight up to straight right.
        target = self.viewer.mapFromScene(QPointF(centre.x() + 80, centre.y()))
        QTest.mousePress(self.viewer.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
        QTest.mouseMove(self.viewer.viewport(), target, 20)
        QTest.mouseRelease(self.viewer.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, target)
        APP.processEvents()
        self.assertAlmostEqual(self.viewer.wipe["angle"], 90.0, delta=1.0)
        after = (self.screen(0.4, 0.25), self.screen(0.4, 0.75))
        self.assertNotEqual(after[0], after[1])
        self.assertEqual(after[0], before[0])

    def test_ctrl_click_and_shift_w_reset_the_wipe(self):
        self.compare(2, "wipe")
        self.viewer.wipe.update(x=0.3, y=0.6, angle=40.0)
        centre = self.viewer.mapFromScene(self.viewer._wipe_scene_geometry()[0])
        QTest.mouseClick(self.viewer.viewport(), Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.ControlModifier, centre)
        self.assertEqual(self.viewer.wipe, compare_model.WIPE_DEFAULT)
        self.viewer.wipe.update(x=0.9, angle=10.0)
        QTest.keyClick(self.viewer, Qt.Key.Key_W, Qt.KeyboardModifier.ShiftModifier)
        self.assertEqual(self.viewer.wipe, compare_model.WIPE_DEFAULT)

    def test_pixel_readout_shows_the_buffer_under_the_pointer(self):
        self.compare(2, "wipe")
        viewport = self.viewer.viewport()
        QTest.mouseMove(viewport, self.scene_point(0.25, 0.5))
        APP.processEvents()
        self.assertEqual(self.viewer.readout_buffer, "A")
        self.assertIn("1.00000 0.00000 0.00000", self.viewer.pixel_readout.label.text())
        QTest.mouseMove(viewport, self.scene_point(0.75, 0.5))
        APP.processEvents()
        self.assertEqual(self.viewer.readout_buffer, "B")
        self.assertIn("0.00000 0.00000 1.00000", self.viewer.pixel_readout.label.text())
        self.assertEqual(self.viewer.pixel_readout.buffer_tag.text(), "B")

    def test_over_under_minus_difference_pixels(self):
        a = np.array([[[0.5, 0.2, 0.1, 0.5]]], dtype=np.float32)
        b = np.array([[[0.2, 0.4, 0.6, 1.0]]], dtype=np.float32)
        over = compare_model.combine("over", a, b)[0, 0]
        np.testing.assert_allclose(over, [0.5 + 0.2 * 0.5, 0.2 + 0.4 * 0.5, 0.1 + 0.6 * 0.5, 1.0], atol=1e-6)
        under = compare_model.combine("under", a, b)[0, 0]
        np.testing.assert_allclose(under, [0.2, 0.4, 0.6, 1.0], atol=1e-6)  # B is opaque
        minus = compare_model.combine("minus", a, b)[0, 0]
        np.testing.assert_allclose(minus[:3], [0.3, -0.2, -0.5], atol=1e-6)
        difference = compare_model.combine("difference", a, b)[0, 0]
        np.testing.assert_allclose(difference[:3], [0.3, 0.2, 0.5], atol=1e-6)

    def test_minus_of_identical_inputs_is_black(self):
        self.tearDown()
        self.window = Window(constant_document((("a", RED), ("b", RED))))
        self.window.resize(900, 700)
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None
                                   and self.window.viewer.format_rect is not None))
        self.viewer = self.window.viewer
        self.compare(2, "minus")
        self.assertEqual(self.screen(0.5, 0.5), (0, 0, 0))
        self.assertEqual(float(np.abs(self.window.frame_b - self.window.frame).max()), 0.0)
        self.compare(2, "difference")
        self.assertEqual(self.screen(0.5, 0.5), (0, 0, 0))

    def test_minus_of_different_inputs_is_not_black(self):
        self.compare(2, "difference")
        self.assertNotEqual(self.screen(0.5, 0.5), (0, 0, 0))

    def test_over_and_under_show_the_expected_layer(self):
        # Both constants are opaque: A over B is A, and A under B is B.
        only_a = self.screen(0.5, 0.5)
        self.compare(2, "B only")
        only_b = self.screen(0.5, 0.5)
        self.compare(2, "over")
        self.assertEqual(self.screen(0.5, 0.5), only_a)
        self.compare(2, "under")
        self.assertEqual(self.screen(0.5, 0.5), only_b)


class PlaybackTests(ViewerCompareBase):
    @staticmethod
    def document():
        dispatcher = Dispatcher(constant_document((("a", RED), ("b", BLUE))))
        dispatcher.execute({"op": "time", "first": 1, "last": 6, "current": 1})
        for frame in (1, 6):
            dispatcher.execute({"op": "set_key", "id": "a", "param": "red", "frame": frame,
                                "value": 0.1 * frame})
            dispatcher.execute({"op": "set_key", "id": "b", "param": "red", "frame": frame,
                                "value": 1.0 - 0.1 * frame})
        return dispatcher.document

    def test_both_buffers_are_rendered_at_the_same_frame(self):
        self.compare(2, "wipe")
        for current in (2, 4, 5):
            self.window.set_time(current=current)
            self.settle()
            self.assertAlmostEqual(float(self.window.frame[0, 0, 0]), 0.1 * current, places=4)
            self.assertAlmostEqual(float(self.window.frame_b[0, 0, 0]), 1.0 - 0.1 * current, places=4)

    def test_playback_keeps_a_and_b_on_the_same_frame(self):
        self.compare(2, "wipe")
        seen = []

        def sample():
            if self.window.frame is not None and self.window.frame_b is not None:
                seen.append((float(self.window.frame[0, 0, 0]), float(self.window.frame_b[0, 0, 0])))
            return len({round(a, 3) for a, _ in seen}) >= 3
        self.window.toggle_playback(True)
        self.assertTrue(wait_until(sample, timeout=60))
        self.window.toggle_playback(False)
        for a, b in seen:
            self.assertAlmostEqual(a + b, 1.0, places=2)  # the display cache stores half floats


class DocumentTests(unittest.TestCase):
    def test_old_document_loads_with_input_one_only(self):
        document = constant_document()
        self.assertNotIn("inputs", document["settings"]["viewer"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.nbcomp"
            dispatcher = Dispatcher(copy.deepcopy(document))
            dispatcher.execute({"op": "save", "path": str(path)})
            loaded = Dispatcher(load_document(str(path))).document
        state = viewer_state(loaded)
        self.assertEqual(state["inputs"], ["a"] + [None] * 8)
        self.assertEqual((state["active"], state["b"], state["compare"]), (1, None, "A only"))
        self.assertEqual(loaded["settings"]["viewer"], {"background": "black"})

    def test_state_round_trips_undoes_and_survives_a_delete(self):
        dispatcher = Dispatcher(constant_document())
        dispatcher.execute({"op": "viewer_input", "slot": 3, "id": "c"})
        dispatcher.execute({"op": "viewer_input", "slot": 2, "id": "b", "activate": False})
        dispatcher.execute({"op": "viewer_compare", "b": 2, "mode": "minus"})
        state = viewer_state(dispatcher.document)
        self.assertEqual((state["active"], state["b"], state["compare"]), (3, 2, "minus"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.nbcomp"
            dispatcher.execute({"op": "save", "path": str(path)})
            self.assertEqual(viewer_state(load_document(str(path))), state)
        dispatcher.execute({"op": "delete", "id": "b"})
        self.assertEqual(viewer_state(dispatcher.document)["inputs"][:3], ["a", None, "c"])
        dispatcher.execute({"op": "undo"})
        self.assertEqual(viewer_state(dispatcher.document)["inputs"][:3], ["a", "b", "c"])

    def test_the_view_command_fills_the_active_input_and_the_default_stays_unstored(self):
        dispatcher = Dispatcher(constant_document())
        dispatcher.execute({"op": "viewer_input", "slot": 2, "id": "b"})
        dispatcher.execute({"op": "view", "id": "c"})
        state = viewer_state(dispatcher.document)
        self.assertEqual((state["inputs"][:3], state["active"]), (["a", "c", None], 2))
        dispatcher.execute({"op": "viewer_input", "slot": 1, "id": "a"})
        dispatcher.execute({"op": "viewer_input", "slot": 2, "id": None, "activate": False})
        self.assertEqual(dispatcher.document["settings"]["viewer"], {"background": "black"})

    def test_bad_commands_are_rejected(self):
        dispatcher = Dispatcher(constant_document())
        for command in ({"op": "viewer_input", "slot": 0, "id": "a"}, {"op": "viewer_input", "slot": 10, "id": "a"},
                        {"op": "viewer_input", "slot": 2, "id": "nope"}, {"op": "viewer_compare", "mode": "blend"},
                        {"op": "viewer_compare", "b": 12}, {"op": "viewer_input", "slot": True, "id": "a"}):
            with self.assertRaises(ValueError, msg=str(command)):
                dispatcher.execute(command)
        self.assertEqual(dispatcher.document["settings"]["viewer"], {"background": "black"})

    def test_every_compare_mode_is_accepted(self):
        dispatcher = Dispatcher(constant_document())
        for mode in COMPARE_MODES:
            dispatcher.execute({"op": "viewer_compare", "b": 2, "mode": mode})
            self.assertEqual(viewer_state(dispatcher.document)["compare"], mode)


class ShortcutTableTests(unittest.TestCase):
    def test_the_table_lists_every_new_key(self):
        sections = dict(SHORTCUT_SECTIONS)
        viewer_keys = [key for key, _ in sections["Viewer"]]
        for key in ("1-9", "Alt+1-9", "Shift+W"):
            self.assertIn(key, viewer_keys)
        self.assertTrue(any("wipe" in key.lower() for key in viewer_keys))
        self.assertIn("2-9", [key for key, _ in sections["Node graph"]])


class TransformPriorityTests(ViewerCompareBase):
    @staticmethod
    def document():
        dispatcher = Dispatcher(constant_document((("a", RED), ("b", BLUE))))
        dispatcher.execute({"op": "create", "id": "t", "type": "Transform", "pos": [0, 150]})
        dispatcher.execute({"op": "connect", "id": "t", "input": "image", "source": "a"})
        dispatcher.execute({"op": "view", "id": "t"})
        return dispatcher.document

    def test_a_transform_handle_keeps_its_click_while_the_wipe_is_on(self):
        self.compare(2, "wipe")
        self.window.graph.scene().clearSelection()
        self.window.graph.items_by_id["t"].setSelected(True)
        APP.processEvents()
        self.assertIsNotNone(self.viewer._transform_context())
        centre = self.scene_point(0.5, 0.5)  # the wipe centre lies inside the Transform box
        QTest.mousePress(self.viewer.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, centre)
        self.assertIsNotNone(self.viewer.transform_drag)
        self.assertIsNone(self.viewer.wipe_drag)
        QTest.keyClick(self.viewer, Qt.Key.Key_Escape)
        QTest.mouseRelease(self.viewer.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, centre)
        self.assertEqual(self.viewer.wipe, compare_model.WIPE_DEFAULT)


if __name__ == "__main__":
    unittest.main()
