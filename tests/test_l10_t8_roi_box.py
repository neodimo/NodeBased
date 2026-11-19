"""ColorTransfer's ROI box in the Viewer (lane 10, step T8, 2026-11-18). Nuke 17.0 Reference Guide, ColorTransfer: the ROI
x, y, r, t count from the bottom left of input A, "By default, it is set to cover the entire input", and "You can also
adjust the ROI in the Viewer by dragging its edges." The live drags are probed in tests/test_viewer_controls.py; this module
checks the box geometry the Viewer draws and writes back."""
import unittest

from nodebased import handles2d


class RoiBoxGeometryTests(unittest.TestCase):
    def test_the_roi_knobs_draw_a_box_counted_from_the_bottom_left(self):
        box = handles2d.box_from_knobs("ColorTransfer", {"roi_x": 100, "roi_y": 100, "roi_r": 500, "roi_t": 400}, 960, 540)
        self.assertEqual(box, (100.0, 140.0, 400.0, 300.0))

    def test_the_default_box_covers_the_whole_input(self):
        zero = {"roi_x": 0.0, "roi_y": 0.0, "roi_r": 0.0, "roi_t": 0.0}
        self.assertEqual(handles2d.box_from_knobs("ColorTransfer", zero, 960, 540), (0.0, 0.0, 960.0, 540.0))

    def test_a_dragged_box_writes_the_knobs_back(self):
        start = handles2d.box_from_knobs("ColorTransfer", {"roi_x": 100, "roi_y": 100, "roi_r": 500, "roi_t": 400}, 960, 540)
        moved = handles2d.box_drag(start, "ne", 50, -20)          # top right corner: r out, t up
        self.assertEqual(handles2d.knobs_from_box("ColorTransfer", moved, 540),
                         {"roi_x": 100.0, "roi_y": 100.0, "roi_r": 550.0, "roi_t": 420.0})

    def test_crop_and_text_boxes_are_unchanged(self):
        values = {"x": 10, "y": 20, "width": 30, "height": 40}
        self.assertEqual(handles2d.box_from_knobs("Crop", values, 960, 540), (10.0, 20.0, 30.0, 40.0))
        self.assertEqual(handles2d.knobs_from_box("Crop", (1, 2, 3, 4), 540), {"x": 1, "y": 2, "width": 3, "height": 4})


if __name__ == "__main__":
    unittest.main()
