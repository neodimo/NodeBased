"""The faster Roto and RotoPaint matte drawer (nodebased/rotofast.py) against the drawer it replaced.

The legacy drawer stays selectable (`NODEBASED_ROTO_DRAWER=legacy`, `roto.set_drawer("legacy")`) for one release, so
every case here draws the same resolved shapes with both and compares: the coverage kernels agree to float rounding,
tiles equal the full frame, the cache keys do not depend on the drawer, cancellation and old documents behave as before.
"""
import copy
import math
import os
import threading
import unittest
from unittest import mock

import numpy as np

from nodebased import roto, rotofast, shapes
from nodebased.core import Dispatcher, empty_document
from nodebased.imaging import Cancelled, Evaluator
from nodebased.tileexec import TileExecutor

TOL = 1e-6          # far below one 8-bit step (0.0039); the kernels differ by about 6e-8


def point(x, y, **extra):
    return {"x": x, "y": y, "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0, **extra}


def smooth(x, y, handle, angle, **extra):
    hx, hy = handle * math.cos(angle), handle * math.sin(angle)
    return {"x": x, "y": y, "in_x": -hx, "in_y": -hy, "out_x": hx, "out_y": hy, **extra}


def shape(points, **extra):
    return {"name": "s", "mode": "union", "opacity": 1.0, "feather": 0.0, "points": points, **extra}


def blob(cx, cy, radius, count=12, **extra):
    return [smooth(cx + radius * math.cos(2 * math.pi * i / count) * (1 + 0.2 * math.sin(3 * i)),
                   cy + radius * math.sin(2 * math.pi * i / count), radius * 0.5,
                   2 * math.pi * i / count + math.pi / 2, **extra) for i in range(count)]


def draw(shape_list, width, height, drawer, frame=1, params=None, **kwargs):
    resolved = shapes.resolve_shapes({"shapes": copy.deepcopy(shape_list)}, frame, params)
    roto.set_drawer(drawer)
    try:
        return roto.rasterise(resolved, width, height, **kwargs)
    finally:
        roto.set_drawer("fast")


def both(shape_list, width=96, height=80, **kwargs):
    return draw(shape_list, width, height, "fast", **kwargs), draw(shape_list, width, height, "legacy", **kwargs)


class DrawerSettingTests(unittest.TestCase):
    def tearDown(self):
        roto.set_drawer("fast")

    def test_the_fast_drawer_is_the_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(roto.DRAWER_ENV, None)
            self.assertEqual(roto.drawer(), "fast")

    def test_set_drawer_chooses_and_rejects(self):
        self.assertEqual(roto.set_drawer("legacy"), "legacy")
        self.assertEqual(roto.set_drawer("fast"), "fast")
        with self.assertRaises(ValueError):
            roto.set_drawer("turbo")

    def test_the_environment_wins_over_the_saved_choice(self):
        roto.set_drawer("fast")
        with mock.patch.dict(os.environ, {roto.DRAWER_ENV: "legacy"}):
            self.assertEqual(roto.drawer(), "legacy")
        roto.set_drawer("legacy")
        with mock.patch.dict(os.environ, {roto.DRAWER_ENV: "fast"}):
            self.assertEqual(roto.drawer(), "fast")
        with mock.patch.dict(os.environ, {roto.DRAWER_ENV: "nonsense"}):
            self.assertEqual(roto.drawer(), "legacy")                  # an unknown value is ignored

    def test_the_legacy_drawer_never_enters_the_fast_kernels(self):
        boom = mock.Mock(side_effect=AssertionError("fast kernel used"))
        line = shape([point(5, 5), point(60, 8), point(30, 50)], feather=4.0)
        wire = shape([point(5, 5), point(60, 40)], closed=False, stroke_width=6.0)
        edge = shape(blob(40, 40, 20), feather=6.0, feather_model="edge")
        with mock.patch.object(rotofast, "coverage", boom), mock.patch.object(rotofast, "stroke_coverage", boom), \
                mock.patch.object(rotofast, "edge_feather", boom), mock.patch.object(rotofast, "stroke_polyline", boom):
            draw([line, wire, edge], 96, 80, "legacy")

    def test_the_fast_drawer_never_enters_the_legacy_kernels(self):
        boom = mock.Mock(side_effect=AssertionError("legacy kernel used"))
        line = shape([point(5, 5), point(60, 8), point(30, 50)], feather=4.0)
        wire = shape([point(5, 5), point(60, 40)], closed=False, stroke_width=6.0)
        with mock.patch.object(roto, "coverage_legacy", boom):
            draw([line, wire], 96, 80, "fast")


class CoverageAgreementTests(unittest.TestCase):
    """The scanline kernels on their own: random polygons and curves, clipped, offset and degenerate."""

    def compare(self, points, width, height, origin=(0, 0)):
        polyline = roto.flatten_points(points)
        fast = rotofast.coverage(polyline, width, height, origin)
        slow = roto.coverage_legacy(points, width, height, origin)
        self.assertEqual(fast.dtype, np.float32)
        self.assertEqual(fast.shape, (height, width))
        np.testing.assert_allclose(fast, slow, atol=TOL, rtol=0)
        return fast

    def test_random_polygons_including_self_intersecting_ones(self):
        rng = np.random.default_rng(7)
        for _ in range(40):
            count = int(rng.integers(3, 14))
            pts = [point(float(x), float(y)) for x, y in rng.uniform(-20, 100, size=(count, 2))]
            self.compare(pts, 80, 64, origin=(int(rng.integers(0, 30)), int(rng.integers(0, 30))))

    def test_random_curves_with_handles(self):
        rng = np.random.default_rng(11)
        for _ in range(25):
            count = int(rng.integers(3, 9))
            pts = [smooth(float(x), float(y), float(rng.uniform(0, 30)), float(rng.uniform(0, 6.3)))
                   for x, y in rng.uniform(0, 90, size=(count, 2))]
            self.compare(pts, 96, 80)

    def test_a_hole_wound_the_other_way_stays_empty(self):
        outer = [(5, 5), (60, 5), (60, 50), (5, 50)]
        inner = [(20, 20), (20, 35), (45, 35), (45, 20)]                  # the opposite winding
        pts = [point(*p) for p in outer + [outer[0]] + inner + [inner[0]]]
        cover = self.compare(pts, 70, 60)
        self.assertEqual(float(cover[28, 30]), 0.0)
        self.assertEqual(float(cover[10, 10]), 1.0)

    def test_exact_values_for_an_axis_aligned_box(self):
        cover = self.compare([point(10.25, 8.5), point(30.75, 8.5), point(30.75, 24.5), point(10.25, 24.5)], 40, 32)
        self.assertAlmostEqual(float(cover[10, 10]), 0.75, places=5)      # a quarter pixel in from the left edge
        self.assertAlmostEqual(float(cover[10, 30]), 0.75, places=5)
        self.assertAlmostEqual(float(cover[8, 15]), 0.5, places=5)        # half a row in at the top
        self.assertAlmostEqual(float(cover.sum()), 20.5 * 16, places=3)

    def test_clipped_off_canvas_and_empty_shapes(self):
        self.compare([point(-50, -50), point(200, -20), point(60, 300)], 64, 48)
        empty = self.compare([point(300, 300), point(400, 300), point(350, 380)], 64, 48)
        self.assertEqual(float(empty.max()), 0.0)
        self.compare([point(5, 5), point(5, 5), point(5, 5)], 32, 32)               # no area at all
        self.compare([point(5, 5), point(30, 5), point(60, 5)], 64, 32)             # every edge horizontal

    def test_a_point_that_is_not_a_number_behaves_as_it_always_did(self):
        # validation keeps these out of a document and the shape bounds refuse them; the kernels still answer as before
        def outcome(call):
            with np.errstate(all="ignore"):
                try:
                    return call()
                except (ValueError, OverflowError) as error:
                    return type(error)

        quad = [point(5, 5), point(40, 8), point(30, 30), point(10, 40)]
        wire = [point(5, 5), point(30, 30), point(50, 40)]
        for bad in (float("nan"), float("inf")):
            closed = [dict(p) for p in quad]
            closed[2]["x"] = bad
            open_ = [dict(p) for p in wire]
            open_[1]["x"] = bad
            calls = {"coverage": lambda: roto.coverage(closed, 64, 48),
                     "stroke": lambda: roto.stroke_coverage(open_, 64, 48, 5.0),
                     "edge": lambda: roto.edge_feather(closed, np.ones((48, 64), np.float32), 3.0)}
            for name, call in calls.items():
                results = []
                for drawer in roto.DRAWERS:
                    roto.set_drawer(drawer)
                    try:
                        results.append(outcome(call))
                    finally:
                        roto.set_drawer("fast")
                a, b = results
                if isinstance(a, np.ndarray):
                    np.testing.assert_array_equal(a, b, err_msg=name)
                else:
                    self.assertIs(a, b, name)

    def test_a_tiny_shape_inside_one_pixel(self):
        self.compare([point(10.2, 10.2), point(10.6, 10.3), point(10.4, 10.7)], 24, 24)

    def test_vertices_on_sub_scanline_boundaries(self):
        # y = 1/16 steps are exactly where the old drawer's sub-scanline centres sit
        pts = [point(4, 2.0625), point(20, 2.1875), point(20, 9.0625), point(4, 9.4375)]
        self.compare(pts, 32, 16)

    def test_bands_give_the_same_picture_as_one_pass(self):
        rng = np.random.default_rng(3)
        pts = [point(float(x), float(y)) for x, y in rng.uniform(0, 200, size=(9, 2))]
        polyline = roto.flatten_points(pts)
        whole = rotofast.coverage(polyline, 200, 200, (0, 0))
        with mock.patch.object(rotofast, "BAND_CELLS", 200 * 8):
            banded = rotofast.coverage(polyline, 200, 200, (0, 0))
        np.testing.assert_array_equal(whole, banded)


class RasteriseAgreementTests(unittest.TestCase):
    """Whole mattes: every shape kind, blend, feather and motion-blur setting the Roto node offers."""

    def check(self, shape_list, **kwargs):
        fast, slow = both(shape_list, **kwargs)
        self.assertEqual(fast.shape, slow.shape)
        self.assertEqual(fast.dtype, np.float32)
        self.assertLess(float(np.abs(fast - slow).max()), TOL)
        return fast

    def test_plain_polygon_and_bezier(self):
        self.check([shape([point(10, 10), point(70, 14), point(40, 60)])])
        self.check([shape(blob(48, 40, 28))])

    def test_box_feather_with_the_falloff_curves(self):
        for falloff in ("linear", "smooth", "gaussian"):
            self.check([shape(blob(48, 40, 24), feather=7.0, feather_falloff=falloff)])
        self.check([shape(blob(48, 40, 24), feather=7.0, falloff_power=0.5)])
        self.check([shape(blob(48, 40, 24), feather=3.0, falloff_power=2.0)])

    def test_signed_distance_feather_with_per_point_offsets(self):
        offsets = [dict(feather_x=float(2 + i % 3), feather_y=0.0) for i in range(12)]
        pts = [dict(p, **o) for p, o in zip(blob(48, 40, 24), offsets)]
        for falloff in ("linear", "smooth", "gaussian"):
            self.check([shape(pts, feather=5.0, feather_model="edge", feather_falloff=falloff)])
        self.check([shape(blob(48, 40, 24), feather=-4.0, feather_model="edge")])
        self.check([shape(blob(48, 40, 24), feather=0.0, feather_model="edge")])

    def test_inverted_shapes_and_an_inverted_node(self):
        self.check([shape(blob(48, 40, 24), feather=4.0, invert=True)])
        self.check([shape(blob(48, 40, 24), feather=4.0)], invert=True)

    def test_blend_modes_opacity_colour_and_subtract(self):
        base = shape(blob(40, 40, 26), opacity=0.8, color=[0.9, 0.5, 0.2, 0.7])
        for blend in ("over", "plus", "minus", "multiply"):
            top = shape(blob(56, 42, 22), opacity=0.6, blend_mode=blend, color=[0.2, 0.8, 0.4, 1.0], feather=3.0)
            self.check([base, top])
        cut = shape(blob(52, 40, 14), mode="subtract", feather=2.0)
        self.check([base, cut])
        self.check([cut, base])                                               # subtract before anything is painted
        self.check([shape(blob(40, 40, 20), visible=False), base])

    def test_open_strokes_with_every_cap_and_feather(self):
        pts = [smooth(10, 40, 10, 0.4), smooth(40, 20, 12, 0.0), smooth(70, 50, 12, -0.5), point(86, 30)]
        for caps in (("rounded", "rounded"), ("butt", "butt"), ("square", "square"), ("butt", "square"), ("rounded", "butt")):
            self.check([shape(pts, closed=False, stroke_width=9.0, feather=3.0,
                              cap_start=caps[0], cap_end=caps[1])])
        for falloff in ("smooth", "gaussian"):
            self.check([shape(pts, closed=False, stroke_width=6.0, feather=4.0, feather_falloff=falloff)])
        offset = [dict(p, feather_x=3.0 + i, feather_y=1.0) for i, p in enumerate(pts)]
        self.check([shape(offset, closed=False, stroke_width=7.0, feather=2.0)])
        self.check([shape([point(10, 10), point(10, 10)], closed=False, stroke_width=6.0)])      # a dot

    def test_motion_blur_with_a_shutter_and_the_legacy_three_sample_blur(self):
        def keyed(points, move):
            return [dict(p, x={"value": p["x"], "curve": {"interpolation": "linear", "keys": [
                {"frame": 1, "value": p["x"]}, {"frame": 6, "value": p["x"] + move}]}}) for p in points]
        moving = shape(keyed(blob(40, 40, 20), 30.0), feather=5.0, motion_blur=True, motion_blur_samples=4,
                       motion_blur_shutter=0.75)
        self.check([moving], frame=3)
        self.check([dict(moving, feather_model="edge")], frame=3)
        self.check([shape(keyed(blob(40, 40, 20), 30.0), feather=5.0, motion_blur=True)], frame=3)       # legacy plan
        self.check([shape(keyed(blob(40, 40, 20), 30.0), feather=5.0)], frame=3,
                   params={"global_motionblur_on": 1, "global_motionblur": 3, "global_motionblur_shutter": 0.5})

    def test_shape_transforms_reach_the_drawer(self):
        self.check([shape(blob(48, 40, 20), feather=3.0,
                          transform={"translate_x": 9.0, "translate_y": -4.0, "rotate": 25.0, "scale_x": 1.3})])
        self.check([shape(blob(48, 40, 20), extra_matrix=[1.0, 0.3, 4.0, 0.0, 1.0, -3.0, 0.0, 0.0, 1.0])])


class TileAgreementTests(unittest.TestCase):
    """A region of the matte equals the same pixels of the whole frame, with either drawer."""

    def shapes(self):
        return [shape(blob(60, 50, 34), feather=9.0, color=[1.0, 0.7, 0.4, 1.0], opacity=0.9),
                shape(blob(90, 70, 16), feather=2.0, feather_model="edge", blend_mode="plus"),
                shape([smooth(10, 90, 8, 0.2), point(70, 20), point(120, 100)], closed=False, stroke_width=7.0, feather=3.0),
                shape(blob(60, 50, 20), mode="subtract", feather=3.0, invert=False)]

    def test_regions_equal_the_full_frame(self):
        for drawer in roto.DRAWERS:
            full = draw(self.shapes(), 140, 110, drawer)
            for region in ((0, 0, 140, 110), (13, 7, 50, 40), (60, 50, 80, 60), (0, 0, 16, 16), (130, 100, 10, 10), (33, 0, 1, 110)):
                part = draw(self.shapes(), 140, 110, drawer, region=region)
                x, y, w, h = region
                np.testing.assert_allclose(part, full[y:y + h, x:x + w], atol=1e-6, err_msg=f"{drawer} {region}")

    def test_the_tile_executor_equals_the_evaluator(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "r", "type": "Roto", "params": {"width": 64, "height": 48}})
        d.execute({"op": "set_shapes", "id": "r", "shapes": [shape(blob(30, 24, 18), feather=6.0),
                                                              shape(blob(40, 20, 9), blend_mode="plus", feather=2.0)]})
        d.execute({"op": "view", "id": "r"})
        for drawer in roto.DRAWERS:
            roto.set_drawer(drawer)
            try:
                whole = Evaluator().evaluate_raster(d.document, target="r")
                executor = TileExecutor(evaluator=Evaluator(), tile_edge=16)
                region = executor.canvas_region(d.document, "r", frame=1, tier=1)
                tiled = executor.compose_region(d.document, "r", region, frame=1, tier=1).pixels
            finally:
                roto.set_drawer("fast")
            np.testing.assert_allclose(tiled, whole.pixels, atol=1e-6, err_msg=drawer)

    def test_both_drawers_give_the_same_picture_through_the_evaluator(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "r", "type": "Roto", "params": {"width": 64, "height": 48}})
        d.execute({"op": "set_shapes", "id": "r", "shapes": [shape(blob(30, 24, 18), feather=6.0)]})
        d.execute({"op": "view", "id": "r"})
        pictures = {}
        for drawer in roto.DRAWERS:
            roto.set_drawer(drawer)
            try:
                pictures[drawer] = Evaluator().evaluate_raster(d.document, target="r").pixels
            finally:
                roto.set_drawer("fast")
        np.testing.assert_allclose(pictures["fast"], pictures["legacy"], atol=TOL)


class CacheAndCancelTests(unittest.TestCase):
    def document(self, shape_list=None):
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "r", "type": "Roto", "params": {"width": 64, "height": 48}})
        d.execute({"op": "set_shapes", "id": "r", "shapes": shape_list or [shape(blob(30, 24, 18), feather=6.0)]})
        d.execute({"op": "view", "id": "r"})
        return d.document

    def test_the_cache_key_does_not_depend_on_the_drawer(self):
        doc = self.document()
        digests = {}
        for drawer in roto.DRAWERS:
            roto.set_drawer(drawer)
            try:
                _, digests[drawer] = Evaluator().evaluate_raster(doc, "r", typed=True, return_digest=True)
            finally:
                roto.set_drawer("fast")
        self.assertEqual(digests["fast"], digests["legacy"])

    def test_the_matte_cache_is_hit_by_a_second_frame_with_the_same_shapes(self):
        evaluator = Evaluator()
        doc = self.document()
        evaluator.evaluate_raster(doc, "r", frame=1)
        misses = evaluator.roto_matte_misses
        evaluator.evaluate_raster(doc, "r", frame=2)                 # nothing animated: the same digest, the same matte
        self.assertEqual(evaluator.roto_matte_misses, misses)

    def test_a_set_cancel_stops_the_roto(self):
        doc = self.document()
        for drawer in roto.DRAWERS:
            roto.set_drawer(drawer)
            try:
                cancel = threading.Event()
                cancel.set()
                with self.assertRaises(Cancelled):
                    Evaluator().evaluate_raster(doc, "r", cancel=cancel)
            finally:
                roto.set_drawer("fast")

    def test_a_set_cancel_stops_before_any_matte_is_drawn_or_stored(self):
        doc = self.document([shape(blob(30, 24, 18)), shape(blob(40, 24, 10))])
        evaluator = Evaluator()
        cancel = threading.Event()
        cancel.set()
        with mock.patch.object(roto, "rasterise", side_effect=AssertionError("drew a matte")), self.assertRaises(Cancelled):
            evaluator.evaluate_raster(doc, "r", cancel=cancel)
        self.assertEqual(len(evaluator.cache), 0)
        self.assertEqual(evaluator.roto_matte_misses, 0)

    def test_a_cancel_that_arrives_later_leaves_the_evaluator_working(self):
        doc = self.document()
        evaluator = Evaluator()
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(Cancelled):
            evaluator.evaluate_raster(doc, "r", cancel=cancel)
        self.assertGreater(float(evaluator.evaluate_raster(doc, "r").pixels[..., 3].max()), 0.99)


class OldDocumentTests(unittest.TestCase):
    def test_a_shape_with_only_the_original_fields_draws_the_same(self):
        old = {"name": "old", "mode": "union", "opacity": 1.0, "feather": 5.0,
               "points": [point(10, 10), point(60, 12), point(50, 55), point(14, 50)]}
        fast, slow = both([old])
        np.testing.assert_allclose(fast, slow, atol=TOL)
        self.assertGreater(float(fast[..., 3].max()), 0.99)

    def test_the_realistic_comps_roto(self):
        import json
        import pathlib
        fixture = pathlib.Path(__file__).parent / "fixtures" / "playback_comp.nbcomp"
        payload = json.loads(fixture.read_text())["node_data"]["roto"]
        for frame in (1, 12, 48):
            resolved = shapes.resolve_shapes(payload, frame, {})
            outputs = []
            for drawer in roto.DRAWERS:
                roto.set_drawer(drawer)
                try:
                    outputs.append(roto.rasterise(resolved, 1920, 1080))
                finally:
                    roto.set_drawer("fast")
            self.assertLess(float(np.abs(outputs[0] - outputs[1]).max()), TOL)

    def test_a_rotopaint_shape_item_draws_the_same(self):
        from nodebased import paint
        base = np.zeros((40, 48, 4), dtype=np.float32)
        base[..., 2] = 1.0
        base[..., 3] = 1.0
        item = {"kind": "shape", "name": "p", "mode": "union", "opacity": 0.8, "feather": 4.0, "visible": True,
                "blend": "over", "points": [point(8, 8), point(40, 10), point(24, 34)]}
        items = [item, dict(item, blend="add", opacity=0.5, points=blob(24, 20, 12))]
        results = []
        for drawer in roto.DRAWERS:
            roto.set_drawer(drawer)
            try:
                results.append(paint.rasterise(base.copy(), copy.deepcopy(items), 1))
            finally:
                roto.set_drawer("fast")
        self.assertGreater(float(np.abs(results[0] - base).max()), 0.3)       # the shapes really painted
        np.testing.assert_allclose(results[0], results[1], atol=TOL)

    def test_a_rotopaint_shape_item_in_a_region_equals_the_whole_frame(self):
        from nodebased import paint
        base = np.zeros((40, 48, 4), dtype=np.float32)
        item = {"kind": "shape", "name": "p", "mode": "union", "opacity": 1.0, "feather": 3.0, "visible": True,
                "blend": "over", "points": blob(24, 20, 14)}
        whole = paint.rasterise(base.copy(), [item], 1)
        crop = paint.rasterise(base[8:30, 10:40].copy(), [item], 1, origin=(10, 8), canvas_size=(48, 40))
        np.testing.assert_allclose(crop, whole[8:30, 10:40], atol=1e-6)


class ThreadSafetyTests(unittest.TestCase):
    def test_workers_drawing_together_get_the_picture_a_single_draw_gets(self):
        shape_list = [shape(blob(60, 50, 34), feather=9.0), shape(blob(90, 70, 16), feather=2.0, feather_model="edge")]
        resolved = shapes.resolve_shapes({"shapes": shape_list}, 1)
        expected = roto.rasterise(resolved, 140, 110)
        results, errors = [], []

        def work():
            try:
                for _ in range(6):
                    results.append(roto.rasterise(shapes.resolve_shapes({"shapes": shape_list}, 1), 140, 110))
            except Exception as error:               # noqa: BLE001 - reported below
                errors.append(error)
        threads = [threading.Thread(target=work) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 24)
        for out in results:
            np.testing.assert_array_equal(out, expected)


if __name__ == "__main__":
    unittest.main()
