"""Lane 2 step Y3: rendering and animation that say what they are doing (FINDINGS.md 3-5 of the
10/6 real-display QA). A Write render shows frame N of M with elapsed time in the status bar, can be
cancelled there, ends with a message and a "Show in folder" button, and a failing frame stops the
render with its number."""
import tests.isolation  # noqa: F401  (keeps the window's layout out of the real Qt settings)

import tempfile
import unittest
import unittest.mock
from pathlib import Path

from PySide6.QtWidgets import QLabel

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


if __name__ == "__main__":
    unittest.main()
