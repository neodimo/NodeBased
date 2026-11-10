"""Lane 2 step I6: ripple edit reaches Fit to edges; Track points and Fill in-betweens say they do not ripple."""
import copy
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from nodebased import rotoedges
from tests import test_roto_edges_i1 as edges
from tests.test_roto_edges_i1 import APP, circle_shape, disc_plate_document

KEY_FRAMES = (1, 5, 9)


def animate(shape, frames=KEY_FRAMES):
    """The same shape with every x and y keyed on `frames` at the static value."""
    shape = copy.deepcopy(shape)
    for point in shape["points"]:
        for name in ("x", "y"):
            value = point[name]
            point[name] = {"value": value, "curve": {"interpolation": "linear",
                                                     "keys": [{"frame": f, "value": value} for f in frames]}}
    return shape


def key_table(shape, name="x"):
    return [{k["frame"]: k["value"] for k in p[name]["curve"]["keys"]} for p in shape["points"]]


class RippleFitTests(unittest.TestCase):
    setUp = edges.ViewerEdgeTests.setUp
    open = edges.ViewerEdgeTests.open
    tearDown = edges.ViewerEdgeTests.tearDown
    shapes = edges.ViewerEdgeTests.shapes

    def start(self, ripple, mode="all", autokey=True, **extra):
        radial = np.random.default_rng(11).uniform(-3.5, 3.5, 16)
        shape = animate(circle_shape(radial, handles=False))
        document = disc_plate_document([shape])
        document["time"].update(first=1, last=9, current=5)
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()
        rotoedges.CACHE.clear()
        self.open(document)
        self.viewer.select_roto_tool("Select", "Select Points")
        toolbar = self.viewer.roto_toolbar
        toolbar.set_setting("r", "Select Points", "autokey", autokey)
        if ripple:
            toolbar.set_setting("r", "Select Points", "ripple_edit", True)
            toolbar.set_setting("r", "Select Points", "ripple_mode", mode)
            for name, value in extra.items():
                toolbar.set_setting("r", "Select Points", name, value)
        self.viewer.roto_selected_shape_index = 0
        self.before = key_table(copy.deepcopy(self.shapes()[0]))
        self.before_y = key_table(copy.deepcopy(self.shapes()[0]), "y")

    def fitted_offsets(self):
        after = key_table(self.shapes()[0])
        return [a[5] - b[5] for a, b in zip(after, self.before)]

    def test_ripple_off_changes_only_the_current_key(self):
        self.start(ripple=False)
        self.assertTrue(self.viewer.fit_roto_to_edges())
        after = key_table(self.shapes()[0])
        moved = 0
        for a, b in zip(after, self.before):
            self.assertEqual((a[1], a[9]), (b[1], b[9]))
            moved += abs(a[5] - b[5]) > 1e-6
        self.assertGreater(moved, 4)

    def test_ripple_all_moves_every_key_by_the_fitted_offset(self):
        self.start(ripple=True)
        self.assertTrue(self.viewer.fit_roto_to_edges())
        after = key_table(self.shapes()[0])
        after_y = key_table(self.shapes()[0], "y")
        offsets = self.fitted_offsets()
        self.assertGreater(max(abs(o) for o in offsets), 0.3)
        for a, b, o in zip(after, self.before, offsets):
            for frame in (1, 9):
                self.assertAlmostEqual(a[frame], b[frame] + o, places=6)
        for a, b, bx, ax in zip(after_y, self.before_y, self.before, after):
            offset_y = a[5] - b[5]
            for frame in (1, 9):
                self.assertAlmostEqual(a[frame], b[frame] + offset_y, places=6)

    def test_ripple_to_end_leaves_earlier_keys(self):
        self.start(ripple=True, mode="to end")
        self.assertTrue(self.viewer.fit_roto_to_edges())
        after = key_table(self.shapes()[0])
        for a, b, o in zip(after, self.before, self.fitted_offsets()):
            self.assertEqual(a[1], b[1])
            self.assertAlmostEqual(a[9], b[9] + o, places=6)

    def test_ripple_range_leaves_keys_outside_it(self):
        self.start(ripple=True, mode="range", ripple_from=5.0, ripple_to=8.0)
        self.assertTrue(self.viewer.fit_roto_to_edges())
        after = key_table(self.shapes()[0])
        for a, b in zip(after, self.before):
            self.assertEqual((a[1], a[9]), (b[1], b[9]))

    def test_one_undo_restores_every_key(self):
        self.start(ripple=True)
        before = copy.deepcopy(self.shapes())
        depth = len(self.window.dispatcher.undo_stack)
        self.assertTrue(self.viewer.fit_roto_to_edges())
        self.assertEqual(len(self.window.dispatcher.undo_stack), depth + 1)
        self.window.command({"op": "undo"})
        self.assertEqual(self.shapes(), before)

    def test_ripple_on_a_static_shape_just_fits_it(self):
        self.start(ripple=True)
        static = circle_shape(np.random.default_rng(11).uniform(-3.5, 3.5, 16))
        self.window.command({"op": "set_shapes", "id": "r", "shapes": [static]})
        self.assertTrue(self.viewer.fit_roto_to_edges())
        for point in self.shapes()[0]["points"]:
            self.assertIsInstance(point["x"], float)

    def test_fill_in_betweens_says_ripple_does_not_apply(self):
        self.assertEqual(self.viewer_for(ripple=True)._ripple_skipped_suffix("Fill in-betweens"),
                         " · ripple edit does not apply: Fill in-betweens writes its own keys")
        self.assertEqual(self.viewer_for(ripple=False)._ripple_skipped_suffix("Fill in-betweens"), "")

    def viewer_for(self, ripple):
        self.start(ripple=ripple)
        return self.viewer


if __name__ == "__main__":
    unittest.main()
