"""Lane 2 step Y3: rendering and animation that say what they are doing (FINDINGS.md 3-5 of the
10/6 real-display QA). A Write render shows frame N of M with elapsed time in the status bar, can be
cancelled there, ends with a message and a "Show in folder" button, and a failing frame stops the
render with its number."""
import tests.isolation  # noqa: F401  (keeps the window's layout out of the real Qt settings)

import tempfile
import unittest
import unittest.mock
from pathlib import Path

from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtGui import QHelpEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDoubleSpinBox, QLabel, QLineEdit, QPushButton

from nodebased.app import Window
from tests.test_desktop import APP, release_window
from tests.waiting import wait_until


def make_window(test):
    window = test.window = Window()
    window.show()
    test.assertTrue(wait_until(lambda: window.frame is not None))
    source = next(k for k, n in window.dispatcher.document["nodes"].items() if n["type"] == "Constant")
    window.command({"op": "batch", "commands": [
        {"op": "set", "id": source, "param": "width", "value": 16},
        {"op": "set", "id": source, "param": "height", "value": 16},
        {"op": "create", "id": "writer", "type": "Write"},
        {"op": "connect", "id": "writer", "input": "image", "source": source}]})
    return window


def close_window(test):
    test.addCleanup(release_window, test)
    test.window.saved_document = test.window.dispatcher.document
    test.window.close()
    APP.processEvents()


class WriteRenderFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.window = make_window(self)
        self.addCleanup(close_window, self)
        self.dir = Path(tempfile.mkdtemp(prefix="nb-y3-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.dir, ignore_errors=True))
        self.pattern = str(self.dir / "render.%04d.exr")
        self.window.command({"op": "set", "id": "writer", "param": "path", "value": self.pattern})
        self.window.set_time(first=1, last=24, current=1)

    def files(self):
        return sorted(p.name for p in self.dir.iterdir())

    def test_a_24_frame_range_reports_progress_24_times_and_ends_with_a_message(self):
        w = self.window
        seen, bar = [], []

        def on_progress(position, total, frame, elapsed):
            seen.append((position, total, frame))
            bar.append((w.render_progress.isVisible(), w.render_progress.format(), w.render_cancel.isVisible()))
        w.write_render_progress.connect(on_progress)
        w.render_write("writer", single=False)
        self.assertEqual(seen, [(n, 24, n) for n in range(1, 25)])
        self.assertTrue(all(visible and cancel for visible, _text, cancel in bar))
        self.assertEqual(bar[4][1].split(" · ")[0], "Frame 5 of 24")
        self.assertEqual(len(self.files()), 24)
        message = w.statusBar().currentMessage()
        self.assertIn("Rendered 24 frames", message)
        self.assertIn(self.pattern, message)
        self.assertRegex(message, r" in \d+\.\d s")
        self.assertFalse(w.render_progress.isVisible())
        self.assertFalse(w.render_cancel.isVisible())
        self.assertFalse(w._write_job)

    def test_show_in_folder_appears_when_done_and_opens_the_output_folder(self):
        w = self.window
        self.assertTrue(w.show_in_folder.isHidden())
        w.set_time(first=1, last=3, current=1)
        w.render_write("writer", single=False)
        self.assertFalse(w.show_in_folder.isHidden())
        with unittest.mock.patch("nodebased.app.QDesktopServices.openUrl") as opened:
            w.show_in_folder.click()
        self.assertEqual(opened.call_args.args[0].toLocalFile(), str(self.dir))
        # The next render clears the button until it finishes again.
        w.render_write("writer", single=True)
        self.assertEqual(w.statusBar().currentMessage().split(" to ")[0], "Rendered 1 frame")

    def test_cancel_at_frame_5_leaves_exactly_frames_1_to_4(self):
        """Each EXR is written to a temporary file and renamed, so a frame is either whole or
        absent; cancel at the start of frame 5 leaves 1-4 and no partial file."""
        w = self.window
        w.write_render_progress.connect(lambda position, *_: w.render_cancel.click() if position == 5 else None)
        w.render_write("writer", single=False)
        self.assertEqual(self.files(), [f"render.{n:04d}.exr" for n in range(1, 5)])
        self.assertIn("cancelled after 4 of 24 frames", w.statusBar().currentMessage())
        self.assertFalse(w.render_cancel.isVisible())
        self.assertFalse(w.show_in_folder.isHidden())

    def test_a_failing_frame_stops_the_render_and_marks_the_write(self):
        w = self.window
        real = w.evaluator.evaluate_raster

        def failing(document, target=None, **kwargs):
            if kwargs.get("frame") == 3:
                raise ValueError("disk on fire")
            return real(document, target, **kwargs)
        with unittest.mock.patch.object(w.evaluator, "evaluate_raster", failing):
            w.render_write("writer", single=False)
        self.assertEqual(self.files(), ["render.0001.exr", "render.0002.exr"])
        message = w.statusBar().currentMessage()
        self.assertIn("Write failed at frame 3", message)
        self.assertIn("disk on fire", message)
        self.assertIn("disk on fire", w.node_errors["writer"])
        item = w.graph.items_by_id["writer"]
        self.assertIn("frame 3", item.error)
        self.assertIn("frame 3", item.toolTip())
        self.assertTrue(w.show_in_folder.isHidden())
        self.assertFalse(w.render_progress.isVisible())
        # The next render starts clean.
        w.render_write("writer", single=True)
        self.assertNotIn("writer", w.node_errors)
        self.assertIsNone(w.graph.items_by_id["writer"].error)

    def test_the_failing_write_shows_its_error_on_the_properties_panel(self):
        w = self.window
        w.set_node_error("writer", "Write failed at frame 7: nope")
        w.graph.items_by_id["writer"].setSelected(True)
        panel = w.build_node_panel("writer")
        label = panel.findChild(QLabel, "write-error")
        self.assertIsNotNone(label)
        self.assertIn("frame 7", label.text())

    def test_a_second_render_is_refused_while_one_runs(self):
        w = self.window
        w.set_time(first=1, last=3, current=1)
        calls = []

        def reenter(position, *_):
            if position == 2:
                w.render_write("writer", single=True)
                calls.append(w.statusBar().currentMessage())
        w.write_render_progress.connect(reenter)
        w.render_write("writer", single=False)
        self.assertEqual(self.files(), ["render.0001.exr", "render.0002.exr", "render.0003.exr"])
        self.assertIn("already running", calls[0])


class AnimatedKnobTests(unittest.TestCase):
    """Finding 4 and 5: an animated knob is tinted, a key frame shows a diamond in the field, the key
    button is a drawn diamond with a tooltip, and the right-click menu reaches every curve edit."""

    def setUp(self):
        self.window = Window()
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))
        self.addCleanup(close_window, self)
        self.window.set_time(first=1, last=20, current=1)
        self.select("grade")

    def select(self, node_id):
        self.window.graph.items_by_id[node_id].setSelected(True)
        APP.processEvents()

    def spin(self, index=0):
        """The Grade panel's numeric fields in order: index 0 is Exposure, 1 the next knob."""
        return self.window.properties.findChildren(QDoubleSpinBox)[index]

    def button(self, param="exposure"):
        return self.window.properties.findChild(QPushButton, "key-button")

    def curve(self):
        return ((self.window.dispatcher.document.get("animation") or {})
                .get("curves", {}).get("grade", {}).get("exposure"))

    def keys(self):
        return [(k["frame"], k["value"]) for k in self.curve()["keys"]]

    def key(self, frame, value):
        self.window.command({"op": "set_key", "id": "grade", "param": "exposure", "frame": frame,
                             "value": value}, render=False)
        self.select("grade")

    def goto(self, frame):
        self.window.set_time(current=frame)
        self.assertTrue(wait_until(lambda: self.spin().property("animated") is not None))
        APP.processEvents()

    @staticmethod
    def glyphs(spin):
        return [a for a in spin.lineEdit().actions() if a.objectName() == "key-glyph"]

    @staticmethod
    def corner_pixel(spin):
        image = spin.grab().toImage()
        return image.pixelColor(4, 4)

    def test_a_static_knob_is_not_tinted_and_has_no_glyph(self):
        spin = self.spin()
        self.assertFalse(spin.property("animated"))
        self.assertEqual(spin.styleSheet(), "")
        self.assertEqual(self.glyphs(spin), [])
        self.assertEqual(self.button().property("keyState"), "none")

    def test_an_animated_knob_is_tinted_between_keys_and_shows_a_diamond_on_a_key_frame(self):
        self.key(3, 0.5)
        self.key(10, 2.0)
        static_pixel = self.corner_pixel(self.spin(1))
        self.goto(6)   # between the keys
        spin = self.spin()
        self.assertTrue(spin.property("animated"))
        self.assertFalse(spin.property("keyedHere"))
        self.assertEqual(self.glyphs(spin), [])
        self.assertEqual(self.button().property("keyState"), "between")
        between = self.corner_pixel(spin)
        self.assertGreater(between.blue() - between.red(), static_pixel.blue() - static_pixel.red() + 15,
                           "the animated field is not visibly tinted blue against a static one")
        # The tint is the same on a key frame, which adds the diamond and a filled button.
        self.goto(10)
        spin = self.spin()
        self.assertTrue(spin.property("keyedHere"))
        self.assertEqual(len(self.glyphs(spin)), 1)
        self.assertEqual(self.button().property("keyState"), "keyed")
        self.assertEqual(self.corner_pixel(spin), between)
        # The static knob next to it is untouched.
        self.assertFalse(self.spin(1).property("animated"))
        self.assertEqual(self.glyphs(self.spin(1)), [])

    def test_the_key_button_icon_is_outlined_unkeyed_and_filled_when_keyed(self):
        self.key(3, 0.5)
        self.goto(6)
        outline = self.button().icon().pixmap(14, 14).toImage()
        self.goto(3)
        filled = self.button().icon().pixmap(14, 14).toImage()
        centre = lambda image: image.pixelColor(image.width() // 2, image.height() // 2).alpha()
        self.assertEqual(centre(outline), 0)
        self.assertGreater(centre(filled), 200)
        self.assertEqual(self.button().text(), "")

    def test_the_key_button_tooltip_names_the_action_and_the_shortcut(self):
        self.assertIn("Set the first key at frame 1", self.button().toolTip())
        self.assertIn("I with the field focused", self.button().toolTip())
        self.key(1, 0.5)
        self.assertIn("Delete the key at frame 1", self.button().toolTip())
        self.goto(4)
        self.assertIn("Set a key at frame 4", self.button().toolTip())
        self.assertIn("I with the field focused", self.button().toolTip())

    def test_i_with_the_field_focused_sets_a_key_at_the_playhead(self):
        self.goto(1)
        self.window.set_time(current=5)
        APP.processEvents()
        spin = self.spin()
        spin.setValue(1.25)
        spin.setFocus()
        QTest.keyClick(spin, Qt.Key.Key_I)
        self.assertTrue(wait_until(lambda: self.curve() is not None))
        self.assertEqual(self.keys(), [(5, 1.25)])

    def menu_actions(self):
        menu = self.window.build_curve_menu("grade", "exposure", self.spin())
        return {a.text().split(" (")[0]: a for a in menu.actions() if a.text()}

    def test_the_knob_menu_offers_set_delete_previous_next_and_clear(self):
        self.key(3, 0.5)
        self.key(10, 2.0)
        self.goto(6)
        actions = self.menu_actions()
        for name in ("Set key at frame 6", "Delete key at frame 6", "Previous key", "Next key", "Clear animation"):
            self.assertIn(name, actions)
        self.assertFalse(actions["Delete key at frame 6"].isEnabled())
        self.assertIn("frame 3", [a.text() for a in actions.values() if a.text().startswith("Previous")][0])
        self.assertIn("frame 10", [a.text() for a in actions.values() if a.text().startswith("Next")][0])

    def test_previous_and_next_key_move_the_playhead_between_keys(self):
        self.key(3, 0.5)
        self.key(10, 2.0)
        self.goto(6)
        self.menu_actions()["Next key"].trigger()
        self.assertEqual(self.window.dispatcher.document["time"]["current"], 10)
        self.goto(10)
        self.assertFalse(self.menu_actions()["Next key"].isEnabled())
        self.menu_actions()["Previous key"].trigger()
        self.assertEqual(self.window.dispatcher.document["time"]["current"], 3)
        self.goto(3)
        self.assertFalse(self.menu_actions()["Previous key"].isEnabled())

    def test_set_delete_and_clear_edit_the_curve_and_each_undoes_in_one_step(self):
        w = self.window
        self.key(3, 0.5)
        self.key(10, 2.0)
        self.goto(6)
        self.spin().setValue(1.0)
        self.menu_actions()["Set key at frame 6"].trigger()
        self.assertTrue(wait_until(lambda: len(self.curve()["keys"]) == 3))
        self.assertEqual(self.keys(), [(3, 0.5), (6, 1.0), (10, 2.0)])
        w.command({"op": "undo"})
        self.assertEqual(self.keys(), [(3, 0.5), (10, 2.0)])
        self.goto(10)
        self.menu_actions()["Delete key at frame 10"].trigger()
        self.assertTrue(wait_until(lambda: len(self.curve()["keys"]) == 1))
        w.command({"op": "undo"})
        self.assertEqual(self.keys(), [(3, 0.5), (10, 2.0)])
        self.menu_actions()["Clear animation"].trigger()
        self.assertTrue(wait_until(lambda: self.curve() is None))
        w.command({"op": "undo"})
        self.assertEqual(self.keys(), [(3, 0.5), (10, 2.0)])


class TimelineKeyMarkTests(unittest.TestCase):
    """The key marks under the timeline show the selected node's keys and say what they are."""

    def setUp(self):
        self.window = Window()
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))
        self.addCleanup(close_window, self)
        self.window.set_time(first=1, last=20, current=1)
        for param, frame in (("exposure", 3), ("exposure", 12), ("offset", 12)):
            self.window.command({"op": "set_key", "id": "grade", "param": param, "frame": frame,
                                 "value": 0.5}, render=False)
        self.select("grade")

    def select(self, node_id):
        self.window.graph.scene().clearSelection()
        self.window.graph.items_by_id[node_id].setSelected(True)
        APP.processEvents()

    def test_the_marks_follow_the_selected_node(self):
        w = self.window
        self.assertTrue(wait_until(lambda: w.frame_slider.key_frames == {3, 12}))
        self.select("wash")
        self.assertTrue(wait_until(lambda: w.frame_slider.key_frames == set()))
        self.select("grade")
        self.assertTrue(wait_until(lambda: w.frame_slider.key_frames == {3, 12}))

    def test_the_tooltip_over_a_key_mark_names_the_node_and_knobs(self):
        w = self.window
        self.assertTrue(wait_until(lambda: w.frame_slider.key_frames == {3, 12}))
        bar = w.frame_slider
        self.assertEqual(bar.key_tooltip(3), "Frame 3: key on Grade · exposure")
        self.assertEqual(bar.key_tooltip(12), "Frame 12: key on Grade · exposure, Grade · offset")
        self.assertIsNone(bar.key_tooltip(5))
        # The hover itself: a tooltip event over frame 12's cell shows that text; over an
        # unkeyed frame the bar's general tooltip applies instead.
        with unittest.mock.patch("nodebased.timeline.QToolTip.showText") as shown:
            x = int(bar.frame_x(12) + bar.frame_width() / 2)
            handled = bar.event(QHelpEvent(QEvent.Type.ToolTip, QPoint(x, 10), bar.mapToGlobal(QPoint(x, 10))))
        self.assertTrue(handled)
        self.assertIn("Frame 12: key on Grade", shown.call_args.args[1])
        with unittest.mock.patch("nodebased.timeline.QToolTip.showText") as shown:
            x = int(bar.frame_x(5) + bar.frame_width() / 2)
            bar.event(QHelpEvent(QEvent.Type.ToolTip, QPoint(x, 10), bar.mapToGlobal(QPoint(x, 10))))
        self.assertNotIn("key on", str(shown.call_args))


if __name__ == "__main__":
    unittest.main()
