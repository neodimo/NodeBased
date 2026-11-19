"""Lane 10 step T9, part 0: an on-screen box drag keeps the box and frame it grabbed when a new frame lands mid-drag.

The ColorTransfer default ROI (all four knobs zero) is drawn over the whole frame, so its size comes from the Viewer's
format; before this fix the release re-read that format, and a frame shown between the press and the release (another
format size or origin, as under a loaded test run) moved or resized the box the artist had grabbed."""

import unittest

import tests.isolation  # noqa: F401


class BoxDragSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tools.nuke_viewer_controls import Bench
        cls.bench = Bench()

    @classmethod
    def tearDownClass(cls):
        cls.bench.close()

    def _roi(self):
        p = self.bench.params("ct")
        return tuple(float(p[f"roi_{k}"]) for k in "xyrt")

    def test_default_roi_drag_survives_a_frame_change(self):
        from PySide6.QtCore import QRectF, Qt
        bench = self.bench
        bench.focus("ct", roi_x=0.0, roi_y=0.0, roi_r=0.0, roi_t=0.0)
        bench.idle()
        viewer = bench.window.viewer
        shown = QRectF(viewer.format_rect)
        width, height = shown.width(), shown.height()
        point = viewer._transform_scene_point
        start, end = point(0, height / 2), point(50, height / 2)
        none = Qt.KeyboardModifier.NoModifier
        self.assertTrue(viewer.hit_press(start, Qt.MouseButton.LeftButton, none))
        try:
            viewer.draw_format_overlay(QRectF(37.0, 11.0, width * 2, height * 2))     # a frame of another format lands
            viewer.hit_move(end, none)
            viewer.hit_release(end, Qt.MouseButton.LeftButton, none)
        finally:
            viewer.draw_format_overlay(shown)
        bench.settle()
        self.assertEqual(self._roi(), (50.0, 0.0, float(width), float(height)))
        self.assertEqual(bench.undo_steps(), 1)
        bench.focus("ct", roi_x=0.0, roi_y=0.0, roi_r=0.0, roi_t=0.0)


if __name__ == "__main__":
    unittest.main()
