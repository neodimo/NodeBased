"""Rendering 13 step L3: the viewer's "view undistorted / view plate" switch for a graph with an Undistort node."""
import unittest

import tests.isolation  # noqa: F401  (Qt settings of the test run stay out of the user's real ones)
from nodebased import lensview


class ViewerToggleTests(unittest.TestCase):
    """The viewer's "view undistorted / view plate" switch for a graph with an Undistort node."""

    @staticmethod
    def document(with_undistort=True, view="plate"):
        from nodebased.core import Dispatcher, empty_document
        d = Dispatcher(empty_document())
        commands = [{"op": "create", "id": "plate", "type": "Checker", "params": {"width": 192, "height": 108, "size": 12}}]
        if with_undistort:
            commands += [{"op": "create", "id": "cam", "type": "Camera3D", "params": {"lens_k1": -0.03}},
                         {"op": "create", "id": "undistort", "type": "Undistort"},
                         {"op": "connect", "id": "undistort", "input": "image", "source": "plate"},
                         {"op": "connect", "id": "undistort", "input": "camera", "source": "cam"}]
        d.execute({"op": "batch", "commands": commands})
        d.execute({"op": "view", "id": view})
        return d.document

    def test_the_switch_goes_between_the_plate_and_the_undistorted_frame(self):
        document = self.document()
        self.assertEqual(lensview.state(document), lensview.PLATE)
        self.assertEqual(lensview.next_view(document), (lensview.UNDISTORTED, "undistort"))
        document["view"] = "undistort"
        self.assertEqual(lensview.state(document), lensview.UNDISTORTED)
        self.assertEqual(lensview.next_view(document), (lensview.PLATE, "plate"))
        document["view"] = "cam"
        self.assertIsNone(lensview.state(document))
        self.assertEqual(lensview.next_view(document)[0], lensview.UNDISTORTED)       # from elsewhere: to the straightened frame

    def test_a_graph_without_an_undistort_node_has_no_switch(self):
        document = self.document(with_undistort=False)
        self.assertIsNone(lensview.next_view(document))
        self.assertIsNone(lensview.state(document))
        self.assertFalse(lensview.shows_overscan(document, "plate"))
        self.assertTrue(lensview.shows_overscan(self.document(), "undistort"))

    def test_the_strip_button_shows_for_the_lens_graph_and_switches_the_viewer(self):
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import QSettings
        from nodebased.app import STYLE, Window
        from tests.waiting import wait_until
        app = QApplication.instance() or QApplication([])
        app.setStyle("Fusion")
        app.setStyleSheet(STYLE)
        QSettings("NodeBased", "NodeBased").clear()
        window = Window(self.document())
        window.resize(1440, 900)
        window.graph_dock.hide()
        window.nodes_dock.hide()
        window._default_split_pending = False
        window.show()
        try:
            self.assertTrue(wait_until(lambda: window.frame is not None and window.viewer.format_rect is not None))
            strip = window.viewer_strip
            # On a small screen (the Windows CI runner's) the window cannot reach 1440 px, the strip goes compact
            # and the button moves into the "more" menu; drive the same switch through the viewer then.
            self.assertEqual(strip.lens_button.isHidden(), strip.compact)
            self.assertEqual(strip.lens_button.text(), lensview.LABELS[lensview.UNDISTORTED])
            self.assertEqual(window.viewer.sceneRect().size().toTuple(), (192.0, 108.0))
            toggle = window.viewer.toggle_lens_view if strip.compact else strip.lens_button.click
            toggle()
            self.assertEqual(window.dispatcher.document["view"], "undistort")
            self.assertEqual(strip.lens_button.text(), lensview.LABELS[lensview.PLATE])
            # the whole grown frame is shown, the overscan beyond the 192 x 108 format
            self.assertTrue(wait_until(lambda: window.viewer.sceneRect().width() > 192.0))
            self.assertTrue(wait_until(lambda: window.viewer.sceneRect().height() > 108.0))
            self.assertLess(window.viewer.sceneRect().x(), 0.0)
            # An edit shows a 1/8 then 1/4 picture first, whose guide is rounded to proxy pixels; read the finished one.
            self.assertTrue(wait_until(lambda: not window.busy and not len(window.preview_queue)
                                       and window.frame_generation == window.generation))
            # the format guide stays on the plate's format inside the grown frame
            self.assertEqual(window.viewer.format_rect.getRect(), (0.0, 0.0, 192.0, 108.0))
            self.assertFalse(window.viewer.viewport().grab().isNull())          # the dashed format outline paints
            toggle()
            self.assertEqual(window.dispatcher.document["view"], "plate")
            self.assertTrue(wait_until(lambda: window.viewer.sceneRect().size().toTuple() == (192.0, 108.0)))
            # the "more" menu offers the same switch when the strip has no room for the button
            strip.fill_more_menu()
            self.assertIn(lensview.LABELS[lensview.UNDISTORTED], [a.text() for a in strip.more_menu.actions()])
        finally:
            window.saved_document = window.dispatcher.document
            window.close()
            app.processEvents()

    def test_a_graph_without_undistort_hides_the_button(self):
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import QSettings
        from nodebased.app import STYLE, Window
        from tests.waiting import wait_until
        app = QApplication.instance() or QApplication([])
        QSettings("NodeBased", "NodeBased").clear()
        window = Window(self.document(with_undistort=False))
        window.resize(1440, 900)
        window.show()
        try:
            self.assertTrue(wait_until(lambda: window.frame is not None))
            self.assertTrue(window.viewer_strip.lens_button.isHidden())
        finally:
            window.saved_document = window.dispatcher.document
            window.close()
            app.processEvents()


if __name__ == "__main__":
    unittest.main()
