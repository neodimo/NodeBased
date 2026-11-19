"""Lane 10 step T9, part 1: on-screen handles for Rectangle's and Radial's area, Ramp's two ends and CornerPin's to-pins.

Each probe presses, drags and releases through the Viewer's hit path and checks the right knobs changed, one undo step per
drag, a key on an animated knob, and that the handle is gone once another node is selected."""

import unittest

import tests.isolation  # noqa: F401

from tools import nuke_viewer_controls as controls


class PointAndAreaHandleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bench = controls.Bench()

    @classmethod
    def tearDownClass(cls):
        cls.bench.close()

    def _run(self, control):
        passed, detail = controls.PROBES[control](self.bench)
        self.assertTrue(passed, detail)

    def test_rectangle_area(self):
        self._run("Rectangle.area")

    def test_radial_area(self):
        self._run("Radial.area")

    def test_ramp_points(self):
        self._run("Ramp.points")

    def test_reformat_box(self):
        self._run("Reformat.box")

    def test_cornerpin_pins(self):
        self._run("CornerPin.pins")


if __name__ == "__main__":
    unittest.main()
