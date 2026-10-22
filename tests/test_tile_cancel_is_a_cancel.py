"""A tile render that is cancelled is a cancel, not a NodeBased bug (Lane 2, 10/22).

`overlay_play`'s measuring tool found it: stopping playback while a tile was rendering reached the preview worker as
a bare `CancelledTile`, which the worker reported as "Internal error while evaluating (CancelledTile) ... This is a
NodeBased bug" and which cleared the Viewer's picture and its format rectangle."""
import os
import threading
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

import tests.isolation  # noqa: F401
from tests.waiting import wait_until

from nodebased.cancellation import Cancelled
from nodebased.core import demo_document
from nodebased.imaging import Cancelled as ImagingCancelled
from nodebased.tileexec import CancelledTile, TileExecutor


class CancelledTileTests(unittest.TestCase):
    def test_it_is_the_cancel_the_preview_worker_already_answers(self):
        self.assertTrue(issubclass(CancelledTile, Cancelled))
        self.assertIs(ImagingCancelled, Cancelled)

    def test_a_compose_with_the_cancel_set_raises_it_as_a_cancel(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(Cancelled):
            TileExecutor().compose(demo_document(), "viewer", frame=1, tier=1, cancel=cancel)

    def test_the_worker_names_no_bug_for_it(self):
        import inspect
        from nodebased import app
        source = inspect.getsource(app.Window.start_preview)
        self.assertIn("except Cancelled:", source)
        self.assertLess(source.index("except Cancelled:"), source.index("Internal error while evaluating"))


class AbandonedRequestTests(unittest.TestCase):
    def test_a_cancelled_request_with_no_frame_leaves_the_picture_alone(self):
        from nodebased.app import STYLE, Window
        from nodebased.core import Dispatcher, empty_document
        app = QApplication.instance() or QApplication([])
        app.setStyleSheet(STYLE)
        d = Dispatcher(empty_document())
        d.document["nodes"] = {}
        d.document["view"] = None
        d.execute({"op": "create", "id": "c", "type": "Constant", "pos": [0, 0],
                   "params": {"width": 64, "height": 48, "red": 0.4, "green": 0.4, "blue": 0.4, "alpha": 1.0}})
        d.document["view"] = "c"
        window = Window(d.document, agent_name="nodebased-test-abandoned")
        seen = []
        real = window.preview_ready

        def spy(payload, frame, image, status, render_region=None):
            if frame is not None:
                seen.append(payload)
            return real(payload, frame, image, status, render_region)
        window.signals.finished.disconnect()
        window.signals.finished.connect(spy)
        window.show()
        try:
            self.assertTrue(wait_until(lambda: window.frame is not None and window.viewer.format_rect is not None and seen))
            request, _ = seen[-1]
            stopped = threading.Event()
            stopped.set()
            shown = window.viewer.format_rect
            # the newest request, cancelled before it had a frame: playback stopped while it was rendering
            window.preview_ready((request, stopped), None, None, "Cancelled", None)
            self.assertIsNotNone(window.frame, "the picture on screen is still the frame")
            self.assertEqual(window.viewer.format_rect, shown, "the format rectangle and the handles stay")
            self.assertNotEqual(window.viewer_info.text(), "Evaluation error")
            # an uncancelled failure still says so
            window.preview_ready((request, threading.Event()), None, None, "Missing input", None)
            self.assertEqual(window.viewer_info.text(), "Evaluation error")
        finally:
            window.saved_document = window.dispatcher.document
            window.close()
            app.processEvents()


if __name__ == "__main__":
    unittest.main()
