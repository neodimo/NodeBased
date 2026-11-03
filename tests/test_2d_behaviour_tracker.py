"""2D behaviour, lane 9 step I: the Tracker's Transform tab acts on the picture, shown on a plate moved by a known offset.

* reference frame: the frame the transform is the identity on; the picture at any other frame moves by the track's displacement from it.
* filter: Impulse hands every output pixel a value the plate already holds; Cubic and the others blend, so a half-pixel move gives new values.
* clamp: with a sharpening filter the output stays inside the plate's range once it is on.
* black outside: pixels the move uncovers are black when it is on and the edge pixel held when it is off.
* jitter period: the number of frames averaged to get the stable position, so the picture's move changes with it.
Step G (tests/test_2d_fidelity_g_tracker.py) covers the solving; these read the pixels.
"""
import copy
import unittest

import numpy as np

from nodebased import tracker
from nodebased import trackerknobs
from nodebased.core import SPECS, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor
from tests.test_2d_fidelity_g_tracker import H, W, track
from tests.test_2d_fidelity_g_tracker import build as legacy_build
from tests.test_2d_fidelity_g_tracker import at_frame as raster_at

SLIDING = [track("a", {1: (20.0, 20.0), 3: (28.0, 24.0)})]
SHAKING = [track("a", {f: (20.0 + (4.0 if f % 2 else 0.0), 20.0) for f in range(1, 13)})]


def build(tracks, **params):
    """A Tracker as a new one is made: black outside on, the edges and clamp read as the guide does."""
    return legacy_build(tracks, **{"legacy_transform_edges": 0, "black_outside": 1, **params})


def at_frame(g, frame, target="node"):
    """The picture on the format's rectangle: the node's data window pasted onto black (a moved picture carries its window with it)."""
    made = raster_at(g, frame, target)
    display, data = made.display, made.data
    canvas = np.zeros((display.height, display.width, made.pixels.shape[2]), np.float32)
    shown = display.intersect(data)
    if not shown.is_empty:
        canvas[shown.y - display.y:shown.bottom - display.y, shown.x - display.x:shown.right - display.x] = \
            made.pixels[shown.y - data.y:shown.bottom - data.y, shown.x - data.x:shown.right - data.x]
    return canvas


def plate(g):
    return at_frame(g, 1, "plate")


def interior(a, margin=12):
    return a[margin:-margin, margin:-margin]


class ReferenceFrameTests(unittest.TestCase):
    def test_the_picture_moves_by_the_displacement_from_the_reference_frame(self):
        g = build(SLIDING, reference_frame=1)
        base = plate(g)
        np.testing.assert_array_equal(at_frame(g, 1), base)                      # the reference frame is the identity
        moved = at_frame(g, 3)                                                    # 8 right and 4 down
        np.testing.assert_array_equal(interior(moved, 14)[:, :], interior(np.roll(np.roll(base, 4, axis=0), 8, axis=1), 14))

    def test_another_reference_frame_makes_that_frame_the_identity(self):
        g = build(SLIDING, reference_frame=3)
        base = plate(g)
        np.testing.assert_array_equal(at_frame(g, 3), base)
        back = at_frame(g, 1)                                                     # 8 left and 4 up
        np.testing.assert_array_equal(interior(back, 14), interior(np.roll(np.roll(base, -4, axis=0), -8, axis=1), 14))

    def test_changing_the_reference_frame_changes_the_pixels(self):
        self.assertFalse(np.array_equal(at_frame(build(SLIDING, reference_frame=1), 3),
                                        at_frame(build(SLIDING, reference_frame=2), 3)))


class FilterTests(unittest.TestCase):
    half = [track("a", {1: (20.0, 20.0), 2: (20.5, 20.0)})]

    def output(self, **params):
        return at_frame(build(self.half, **{"filter": "cubic", "mode": "match_move", **params}), 2)

    def test_impulse_carries_the_plates_own_values(self):
        values = set(np.unique(plate(build(self.half))[..., 0]).tolist())
        got = set(np.unique(interior(self.output(filter="nearest"))[..., 0]).tolist())
        self.assertTrue(got <= values)

    def test_a_smoothing_filter_makes_new_values_on_a_half_pixel_move(self):
        values = set(np.unique(plate(build(self.half))[..., 0]).tolist())
        for name in ("bilinear", "cubic"):
            with self.subTest(filter=name):
                got = set(np.unique(np.round(interior(self.output(filter=name))[..., 0], 5)).tolist())
                self.assertFalse(got <= values)

    def test_the_filters_differ_from_each_other(self):
        impulse, linear, cubic = (self.output(filter=n) for n in ("nearest", "bilinear", "cubic"))
        self.assertFalse(np.array_equal(impulse, linear))
        self.assertFalse(np.array_equal(linear, cubic))

    def test_clamp_keeps_a_sharpening_filter_inside_the_plates_range(self):
        base = plate(build(self.half))[..., :3]
        free = interior(self.output(filter="cubic", clamp=0))[..., :3]
        held = interior(self.output(filter="cubic", clamp=1))[..., :3]
        self.assertGreaterEqual(float(held.min()), float(base.min()) - 1e-6)
        self.assertLessEqual(float(held.max()), float(base.max()) + 1e-6)
        self.assertFalse(np.array_equal(free, held))


class BlackOutsideTests(unittest.TestCase):
    shove = [track("a", {1: (20.0, 20.0), 2: (30.0, 20.0)})]

    def test_black_outside_blacks_what_the_move_uncovers_and_off_holds_the_edge(self):
        base = plate(build(self.shove))
        black = at_frame(build(self.shove, black_outside=1), 2)
        held = at_frame(build(self.shove, black_outside=0), 2)
        np.testing.assert_array_equal(black[:, :10, :3], 0.0)                            # the 10 columns the move uncovers
        np.testing.assert_array_equal(held[:, :10], np.repeat(base[:, :1], 10, axis=1))   # the plate's first column held
        np.testing.assert_array_equal(black[:, 10:], held[:, 10:])                       # the rest is the same plate, moved 10
        np.testing.assert_array_equal(black[:, 10:], base[:, :-10])


class EdgesAndTilesTests(unittest.TestCase):
    shove = BlackOutsideTests.shove

    def test_a_new_tracker_has_black_outside_on_and_the_guides_edges(self):
        params = SPECS["Tracker"]["params"]
        self.assertEqual((params["black_outside"], params["legacy_transform_edges"]), (1, 0))

    def test_an_older_document_keeps_its_edges_bit_for_bit(self):
        old = legacy_build(self.shove)                      # the older reading: black outside 0, edges flagged legacy
        before = raster_at(old, 2).pixels
        document = copy.deepcopy(old.doc)
        for name in ("legacy_transform_edges", "black_outside", "clamp"):
            document["nodes"]["node"]["params"].pop(name, None)
        loaded = upgrade_document(document)
        validate(loaded)
        params = loaded["nodes"]["node"]["params"]
        self.assertEqual((params["legacy_transform_edges"], params["black_outside"]), (1, 0))
        np.testing.assert_array_equal(Evaluator().evaluate_raster(dict(loaded, view="node"), target="node", frame=2).pixels, before)
        self.assertEqual(trackerknobs.OLD["legacy_transform_edges"], 1)

    def test_the_tile_path_equals_the_evaluator_where_it_is_used(self):
        for params in (dict(), dict(filter="bilinear"), dict(mode="none", translate_x=3.5, filter="bilinear")):
            with self.subTest(params=params):
                g = build(SLIDING, **params)
                document = dict(g.doc, view="node")
                executor = TileExecutor(evaluator=Evaluator())
                self.assertTrue(executor.supports_tiled(document, "node"))          # black outside on and no clamp: a plain zero fill
                tiled = executor.compose(document, "node", frame=3, tier=1)
                self.assertTrue(tiled.tiled)
                np.testing.assert_allclose(tiled.pixels, Evaluator().evaluate(document, "node", frame=3), atol=1e-6)

    def test_black_outside_off_and_clamp_run_on_the_full_frame(self):
        for params in (dict(black_outside=0), dict(clamp=1), dict(motionblur=1)):
            with self.subTest(params=params):
                g = build(SLIDING, **params)
                self.assertFalse(TileExecutor(evaluator=Evaluator()).supports_tiled(dict(g.doc, view="node"), "node"))


class JitterPeriodTests(unittest.TestCase):
    def test_the_period_is_the_number_of_frames_averaged(self):
        picture = {n: at_frame(build(SHAKING, mode="remove_jitter", jitter_period=n), 6) for n in (2, 6)}
        self.assertFalse(np.array_equal(picture[2], picture[6]))

    def test_a_shaking_track_is_steadied_by_removing_the_jitter(self):
        # Frame 6 sits at x = 20 (an even frame sits low, an odd one 4 higher). The mean over the period about it is where the picture goes:
        # period 6 covers frames 3 to 9 (four at 24, three at 20: 22.2857), period 2 covers frames 5 to 7 (24, 20, 24: 22.6667).
        for period, mean in ((6, 22.0 + 2.0 / 7.0), (2, 22.0 + 2.0 / 3.0)):
            with self.subTest(period=period):
                params = dict(SPECS["Tracker"]["params"], mode="remove_jitter", jitter_period=period)
                solved = tracker.solve({"tracks": SHAKING}, 6, params)
                self.assertAlmostEqual(solved["translate_x"], mean - 20.0, places=4)
                self.assertAlmostEqual(solved["translate_y"], 0.0, places=6)

    def test_adding_the_jitter_moves_the_picture_the_other_way(self):
        params = dict(SPECS["Tracker"]["params"], mode="add_jitter", jitter_period=6)
        solved = tracker.solve({"tracks": SHAKING}, 6, params)
        self.assertAlmostEqual(solved["translate_x"], -(2.0 / 7.0 + 2.0), places=4)


if __name__ == "__main__":
    unittest.main()
