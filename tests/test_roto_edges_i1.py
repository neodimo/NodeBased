"""Roto edge snap and fit-to-edges: the edge field (lane 9, step I1, group 1)."""
import copy
import math
import os
import tempfile
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QPointF, QSettings, Qt
from PySide6.QtTest import QTest

from nodebased import rotoedges, roto
from nodebased.app import Window
from nodebased.core import Dispatcher, atomic_save, empty_document, load_document
from nodebased.rotolist import ShapeListTree
from nodebased.imaging import Evaluator
from tests.test_roto_ui import APP
from tests.waiting import wait_until

CENTRE = (100.0, 90.0)
RADIUS = 40.0


def disc_image(width=200, height=180, centre=CENTRE, radius=RADIUS):
    """A white disc on black, anti-aliased by exact-ish coverage; pixel centres at +0.5."""
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    distance = np.hypot(xs + 0.5 - centre[0], ys + 0.5 - centre[1])
    coverage = np.clip(radius - distance + 0.5, 0.0, 1.0).astype(np.float32)
    image = np.zeros((height, width, 4), np.float32)
    image[..., :3] = coverage[..., None]
    image[..., 3] = 1.0
    return image


def full_field(image, sigma=rotoedges.SIGMA):
    height, width = image.shape[:2]
    return rotoedges.field_from_image(image, (0, 0, width, height), sigma)


def circle_points(count=16, radius=RADIUS, centre=CENTRE, radial=None, handles=True):
    points = []
    k = 4.0 / 3.0 * math.tan(math.pi / (2 * count))
    for i in range(count):
        a = 2 * math.pi * i / count
        r = radius + (radial[i] if radial is not None else 0.0)
        tx, ty = -math.sin(a), math.cos(a)
        handle = k * radius if handles else 0.0
        points.append({name: float(value) for name, value in
                       (("x", centre[0] + r * math.cos(a)), ("y", centre[1] + r * math.sin(a)),
                        ("in_x", -tx * handle), ("in_y", -ty * handle),
                        ("out_x", tx * handle), ("out_y", ty * handle))})
    return points


def rms_to_circle(points, closed=True):
    outline = roto.flatten_points(points, closed=closed)
    distances = [math.hypot(x - CENTRE[0], y - CENTRE[1]) - RADIUS for x, y in outline]
    return math.sqrt(sum(d * d for d in distances) / len(distances))


class SnapTests(unittest.TestCase):
    def setUp(self):
        self.field = full_field(disc_image())

    def test_snap_from_five_pixels_outside_lands_on_the_edge(self):
        for angle in (0.0, 0.7, 2.2, 4.0, 5.5):
            point = (CENTRE[0] + (RADIUS + 5) * math.cos(angle), CENTRE[1] + (RADIUS + 5) * math.sin(angle))
            x, y = self.field.snap(point, 8)
            error = abs(math.hypot(x - CENTRE[0], y - CENTRE[1]) - RADIUS)
            self.assertLess(error, 0.5, (angle, error))
        # inside too
        point = (CENTRE[0] + (RADIUS - 4), CENTRE[1])
        x, y = self.field.snap(point, 8)
        self.assertLess(abs(math.hypot(x - CENTRE[0], y - CENTRE[1]) - RADIUS), 0.5)

    def test_snap_slides_toward_the_edge_not_along_it(self):
        point = (CENTRE[0] + RADIUS + 5, CENTRE[1])
        x, y = self.field.snap(point, 8)
        self.assertLess(abs(y - CENTRE[1]), 1.0)

    def test_point_with_no_edge_nearby_is_unchanged(self):
        point = (CENTRE[0] + RADIUS + 30, CENTRE[1])
        self.assertEqual(self.field.snap(point, 8), point)
        flat = full_field(np.zeros((64, 64, 4), np.float32))
        self.assertEqual(flat.snap((20.0, 20.0), 8), (20.0, 20.0))

    def test_snap_ignores_edges_beyond_the_radius(self):
        point = (CENTRE[0] + RADIUS + 12, CENTRE[1])
        self.assertEqual(self.field.snap(point, 8), point)

    def test_a_coloured_edge_with_equal_luma_is_found(self):
        image = np.zeros((60, 60, 4), np.float32)
        image[..., 3] = 1.0
        image[:, :30, 0] = 0.5   # red half
        image[:, 30:, 1] = 0.5   # green half: the max channel is equal on both sides? no: both 0.5
        image[:, :30, 0] = 0.9
        field = full_field(image)
        x, y = field.snap((26.0, 30.0), 8)
        self.assertAlmostEqual(x, 30.0, delta=0.5)


class FieldTests(unittest.TestCase):
    def test_region_of_interest_is_bounded(self):
        image = disc_image(1920, 1080, centre=(960.0, 540.0), radius=300.0)
        box = rotoedges.aligned_box(950 - 12, 530 - 12, 950 + 12, 530 + 12, 1920, 1080)
        field = rotoedges.field_from_image(image, box)
        self.assertEqual(field.mag.shape, (box[3] - box[1], box[2] - box[0]))
        self.assertLessEqual(field.mag.size, 64 * 64)
        self.assertLess(field.mag.size, 1920 * 1080 / 400)

    def test_cache_hits_on_the_same_frame_and_misses_on_an_edit(self):
        cache = rotoedges.EdgeCache()
        calls = []
        image = disc_image()

        def provider():
            calls.append(1)
            return image

        box = (0, 0, 64, 64)
        first = cache.field("roto1", 5, "digestA", box, provider)
        second = cache.field("roto1", 5, "digestA", box, provider)
        self.assertIs(first, second)
        self.assertEqual((cache.hits, cache.misses, len(calls)), (1, 1, 1))
        cache.field("roto1", 6, "digestA", box, provider)
        cache.field("roto1", 5, "digestB", box, provider)
        cache.field("roto1", 5, "digestA", (0, 0, 64, 96), provider)
        self.assertEqual(cache.misses, 4)

    def test_window_field_matches_full_frame_field(self):
        image = disc_image()
        full = full_field(image)
        box = (64, 32, 128, 96)
        part = rotoedges.field_from_image(image, box)
        np.testing.assert_allclose(part.mag, full.mag[32:96, 64:128], atol=1e-6)


class FitTests(unittest.TestCase):
    def setUp(self):
        self.field = full_field(disc_image())

    def test_perturbed_circle_fits_the_disc_edge(self):
        rng = np.random.default_rng(7)
        radial = rng.uniform(-4.0, 4.0, 16)
        shape = {"closed": True, "points": circle_points(radial=radial)}
        before = rms_to_circle(shape["points"])
        self.assertGreater(before, 1.5)
        fitted = self.field.fit(shape, band=10, iterations=12)
        after = rms_to_circle(fitted)
        self.assertLess(after, 0.75, (before, after))
        self.assertEqual(len(fitted), 16)
        # the input shape is not modified
        self.assertAlmostEqual(rms_to_circle(shape["points"]), before)
        self.last_circle_rms = after

    def test_fit_keeps_feather_offsets_relative(self):
        points = circle_points(radial=np.full(16, 3.0))
        for p in points:
            p["feather_x"], p["feather_y"] = 2.0, -1.0
        fitted = self.field.fit({"closed": True, "points": points}, band=10, iterations=8)
        self.assertTrue(all(p["feather_x"] == 2.0 and p["feather_y"] == -1.0 for p in fitted))

    def test_square_fits_to_its_corners(self):
        image = np.zeros((160, 160, 4), np.float32)
        image[..., 3] = 1.0
        image[40:120, 40:120, :3] = 1.0
        field = full_field(image)
        corners = [(40.0, 40.0), (120.0, 40.0), (120.0, 120.0), (40.0, 120.0)]
        offsets = [(3.0, -2.0), (-2.5, 3.0), (2.0, 2.5), (-3.0, -2.0)]
        points = [{"x": x + dx, "y": y + dy, "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0}
                  for (x, y), (dx, dy) in zip(corners, offsets)]
        fitted = field.fit({"closed": True, "points": points}, band=8, iterations=12)
        for (x, y), p in zip(corners, fitted):
            self.assertLess(math.hypot(p["x"] - x, p["y"] - y), 1.0, (x, y, p["x"], p["y"]))

    def test_open_spline_fits_along_a_line_edge(self):
        image = np.zeros((120, 200, 4), np.float32)
        image[..., 3] = 1.0
        image[60:, :, :3] = 1.0   # horizontal edge at y = 60
        field = full_field(image)
        rng = np.random.default_rng(3)
        points = [{"x": 20.0 + 20.0 * i, "y": 60.0 + float(rng.uniform(-4, 4)),
                   "in_x": -6.0, "in_y": 0.0, "out_x": 6.0, "out_y": 0.0} for i in range(8)]
        fitted = field.fit({"closed": False, "points": points}, band=8, iterations=12)
        for p in fitted:
            self.assertAlmostEqual(p["y"], 60.0, delta=0.5)
        outline = roto.flatten_points(fitted, closed=False)
        self.assertLess(max(abs(y - 60.0) for _, y in outline), 1.0)

    def test_points_with_no_edge_in_the_band_stay_put(self):
        shape = {"closed": True, "points": circle_points(radius=15.0)}
        fitted = self.field.fit(shape, band=6, iterations=6)
        for a, b in zip(shape["points"], fitted):
            self.assertAlmostEqual(a["x"], b["x"], places=3)
            self.assertAlmostEqual(a["y"], b["y"], places=3)

    def test_fit_time_on_a_1080p_plate(self):
        image = disc_image(1920, 1080, centre=(960.0, 540.0), radius=300.0)
        points = circle_points(24, 300.0, (960.0, 540.0), radial=np.linspace(-3, 3, 24))
        xs = [p["x"] for p in points]; ys = [p["y"] for p in points]
        band = 12
        box = rotoedges.aligned_box(min(xs) - band - 8, min(ys) - band - 8,
                                    max(xs) + band + 8, max(ys) + band + 8, 1920, 1080)
        started = time.perf_counter()
        field = rotoedges.field_from_image(image, box)
        mid = time.perf_counter()
        fitted = field.fit({"closed": True, "points": points}, band=band, iterations=12)
        done = time.perf_counter()
        self.assertEqual(len(fitted), 24)
        print(f"\nfit 24 points on 1080p: field {1000 * (mid - started):.1f} ms, "
              f"fit {1000 * (done - mid):.1f} ms, region {box}")
        self.assertLess(done - started, 5.0)


# -- the viewer: snap to edges and fit to edges (groups 2 and 3) -------------------------------

DISC_CENTRE = (50.0, 50.0)
DISC_RADIUS = 25.08   # area-equivalent radius of the plate's hard-edged Radial disc (box 25..75)


def disc_plate_document(shapes=None, layers=None):
    """A 100 x 100 Roto whose background is a white disc on black (a Radial with no softness)."""
    document = empty_document()
    document["nodes"] = {}
    document["view"] = None
    document["time"].update(first=1, last=3, current=2)
    dispatcher = Dispatcher(document)
    dispatcher.execute({"op": "create", "id": "plate", "type": "Radial"})
    for name, value in dict(width=100, height=100, box_x=25, box_y=25, box_width=50, box_height=50,
                            softness=0.0).items():
        dispatcher.execute({"op": "set", "id": "plate", "param": name, "value": value})
    dispatcher.execute({"op": "create", "id": "r", "type": "Roto", "pos": [0, 0]})
    dispatcher.execute({"op": "set", "id": "r", "param": "width", "value": 100})
    dispatcher.execute({"op": "set", "id": "r", "param": "height", "value": 100})
    dispatcher.execute({"op": "connect", "id": "r", "input": "bg", "source": "plate"})
    dispatcher.execute({"op": "view", "id": "r"})
    if shapes:
        command = {"op": "set_shapes", "id": "r", "shapes": shapes}
        if layers:
            command["layers"] = layers
        dispatcher.execute(command)
    return dispatcher.document


def circle_shape(radial=None, count=16, name="circle", handles=True, **extra):
    shape = {"name": name, "mode": "union", "opacity": 1.0, "feather": 0.0, "closed": True,
             "points": circle_points(count, DISC_RADIUS, DISC_CENTRE, radial, handles)}
    shape.update(extra)
    return shape


def static_points(shape):
    return [{k: float(p[k]) for k in rotoedges.POINT_FIELDS} for p in shape["points"]]


def edge_error(x, y):
    return abs(math.hypot(x - DISC_CENTRE[0], y - DISC_CENTRE[1]) - DISC_RADIUS)


class ViewerEdgeTests(unittest.TestCase):
    def setUp(self):
        QSettings("NodeBased", "NodeBased").clear()
        rotoedges.CACHE.clear()
        self.open(disc_plate_document())

    def open(self, document):
        self.window = Window(document)
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))
        self.window.graph.items_by_id["r"].setSelected(True)
        APP.processEvents()
        self.viewer = self.window.viewer

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def reopen(self, shapes, autokey=None, layers=None):
        """Start a fresh window whose document already holds `shapes` (as the existing gesture tests do)."""
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()
        rotoedges.CACHE.clear()
        self.open(disc_plate_document(shapes, layers))
        if autokey is not None:
            self.viewer.roto_toolbar.set_setting("r", "Select Points", "autokey", autokey)
        self.viewer.select_roto_tool("Select", "Select Points")

    def scene_pos(self, x, y):
        return self.viewer.mapFromScene(QPointF(x, y))

    def tolerance(self):
        return 1.0 / max(abs(self.viewer.transform().m11()), 0.05) + 0.05

    def shapes(self):
        return self.window.dispatcher.document["node_data"]["r"]["shapes"]

    def set_snap(self, on=True, radius=8.0):
        self.viewer.roto_toolbar.set_setting("r", "Select Points", "snap_edges", on)
        self.viewer.roto_toolbar.set_setting("r", "Select Points", "snap_radius", radius)

    def click_ring(self, offset=4.0, count=8, modifiers=Qt.KeyboardModifier.NoModifier):
        clicks = []
        self.assertTrue(self.window.begin_roto_draw("r"))
        for i in range(count):
            angle = 2 * math.pi * (i + 0.3) / count
            x = DISC_CENTRE[0] + (DISC_RADIUS + offset) * math.cos(angle)
            y = DISC_CENTRE[1] + (DISC_RADIUS + offset) * math.sin(angle)
            clicks.append((x, y))
            QTest.mouseClick(self.viewer.viewport(), Qt.MouseButton.LeftButton, modifiers, pos=self.scene_pos(x, y))
        self.undo_before_finish = len(self.window.dispatcher.undo_stack)
        self.assertTrue(self.viewer.finish_roto_draw(closed=True))
        return clicks

    # -- group 2 -----------------------------------------------------------------------------
    def test_snap_is_off_by_default_and_points_land_where_clicked(self):
        self.viewer.select_roto_tool("Shapes", "Bezier")
        self.assertEqual(self.viewer._roto_snap_config("r")[0], False)
        clicks = self.click_ring()
        for (x, y), p in zip(clicks, self.shapes()[-1]["points"]):
            self.assertAlmostEqual(p["x"], x, delta=self.tolerance())
            self.assertAlmostEqual(p["y"], y, delta=self.tolerance())
        self.assertGreater(min(edge_error(p["x"], p["y"]) for p in self.shapes()[-1]["points"]), 3.0)

    def test_bezier_drawn_near_the_edge_snaps_every_point_onto_it(self):
        self.viewer.select_roto_tool("Shapes", "Bezier")
        self.set_snap(True, 8.0)
        self.click_ring(offset=4.0)
        points = self.shapes()[-1]["points"]
        self.assertEqual(len(points), 8)
        errors = [edge_error(p["x"], p["y"]) for p in points]
        self.assertLess(max(errors), 0.5, errors)
        self.assertEqual(len(self.window.dispatcher.undo_stack), self.undo_before_finish + 1)   # the clicks add none; finishing adds one

    def test_alt_shift_disables_snapping_for_the_gesture(self):
        self.viewer.select_roto_tool("Shapes", "Bezier")
        self.set_snap(True, 8.0)
        clicks = self.click_ring(modifiers=(Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.ShiftModifier))
        for (x, y), p in zip(clicks, self.shapes()[-1]["points"]):
            self.assertAlmostEqual(p["x"], x, delta=self.tolerance())
            self.assertAlmostEqual(p["y"], y, delta=self.tolerance())

    def test_select_tool_point_drag_snaps_and_undoes_once(self):
        self.reopen([circle_shape(handles=False)], autokey=False)
        self.set_snap(True, 8.0)
        before = copy.deepcopy(self.shapes())
        x0 = DISC_CENTRE[0] + DISC_RADIUS
        start, target = self.scene_pos(x0, DISC_CENTRE[1]), self.scene_pos(x0 + 5.0, DISC_CENTRE[1] + 1.0)
        undo = len(self.window.dispatcher.undo_stack)
        QTest.mousePress(self.viewer.viewport(), Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(self.viewer.viewport(), target, 20)
        QTest.mouseRelease(self.viewer.viewport(), Qt.MouseButton.LeftButton, pos=target)
        point = self.shapes()[0]["points"][0]
        self.assertLess(edge_error(point["x"], point["y"]), 0.5)
        self.assertEqual(len(self.window.dispatcher.undo_stack), undo + 1)
        self.window.command({"op": "undo"})
        self.assertEqual(self.shapes(), before)

    def test_drag_without_snap_lands_where_released_and_alt_shift_overrides_snap(self):
        for snap_on, modifiers in ((False, Qt.KeyboardModifier.NoModifier), (True, (Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.ShiftModifier))):
            self.reopen([circle_shape(handles=False)], autokey=False)
            self.set_snap(snap_on, 8.0)
            x0 = DISC_CENTRE[0] + DISC_RADIUS
            start, target = self.scene_pos(x0, DISC_CENTRE[1]), self.scene_pos(x0 + 4.0, DISC_CENTRE[1] + 3.0)
            QTest.mousePress(self.viewer.viewport(), Qt.MouseButton.LeftButton, modifiers, pos=start)
            QTest.mouseMove(self.viewer.viewport(), target, 20)
            QTest.mouseRelease(self.viewer.viewport(), Qt.MouseButton.LeftButton, modifiers, pos=target)
            point = self.shapes()[0]["points"][0]
            self.assertAlmostEqual(point["x"], x0 + 4.0, delta=self.tolerance() + 0.3)
            self.assertAlmostEqual(point["y"], DISC_CENTRE[1] + 3.0, delta=self.tolerance() + 0.3)

    def test_settings_save_and_reload_and_old_documents_open_with_snap_off(self):
        self.assertEqual(self.viewer._roto_snap_config("r"), (False, 8.0))
        self.assertNotIn("tool_settings", self.window.dispatcher.document["node_data"].get("r", {}))
        self.set_snap(True, 11.0)
        self.viewer.roto_toolbar.set_setting("r", "Select Points", "fit_band", 15.0)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "snap.json")
            atomic_save(path, self.window.dispatcher.document)
            reloaded = load_document(path)
        shared = reloaded["node_data"]["r"]["tool_settings"]["__global__"]
        self.assertEqual((shared["snap_edges"], shared["snap_radius"], shared["fit_band"]), (True, 11.0, 15.0))
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        self.open(reloaded)
        self.assertEqual(self.viewer._roto_snap_config("r"), (True, 11.0))
        self.set_snap(False, 8.0)
        self.assertEqual(self.viewer._roto_snap_config("r"), (False, 8.0))
        shared = self.window.dispatcher.document["node_data"]["r"]["tool_settings"]["__global__"]
        self.assertNotIn("snap_edges", shared)
        self.assertNotIn("snap_radius", shared)

    def test_strip_shows_snap_and_fit_controls_for_select_and_shape_tools(self):
        strip = self.viewer.roto_toolbar
        strip.choose("Select", "Select Points")
        self.assertTrue({"snap_edges", "snap_radius", "fit_band", "fit_to_edges"} <= set(strip.settings_widgets))
        strip.choose("Shapes", "Bezier")
        self.assertTrue({"snap_edges", "snap_radius"} <= set(strip.settings_widgets))
        self.assertNotIn("fit_to_edges", strip.settings_widgets)
        strip.choose("Brushes", "Brush") if "Brushes" in strip.buttons and strip.buttons["Brushes"].isVisible() else None

    # -- group 3 -----------------------------------------------------------------------------
    def fit_setup(self, shapes=None, autokey=False):
        radial = np.random.default_rng(11).uniform(-3.5, 3.5, 16)
        self.reopen(shapes or [circle_shape(radial)], autokey=autokey)
        self.viewer.roto_selected_shape_index = 0

    def test_fit_to_edges_through_the_strip_button_matches_the_edge(self):
        self.fit_setup()
        before = rms_to_circle_about(static_points(self.shapes()[0]), DISC_CENTRE, DISC_RADIUS)
        self.assertGreater(before, 1.2)
        self.viewer.roto_toolbar.choose("Select", "Select Points")
        self.viewer.roto_toolbar.settings_widgets["fit_to_edges"].click()
        APP.processEvents()
        after = rms_to_circle_about(static_points(self.shapes()[0]), DISC_CENTRE, DISC_RADIUS)
        self.assertLess(after, 0.75, (before, after))
        self.assertEqual(self.shapes()[0]["points"][0].get("feather_x", 0.0), 0.0)

    def test_one_undo_restores_the_exact_previous_shape(self):
        self.fit_setup([circle_shape(np.linspace(-3, 3, 16), name="a"),
                        circle_shape(np.linspace(3, -3, 16), name="b")])
        before = copy.deepcopy(self.shapes())
        undo = len(self.window.dispatcher.undo_stack)
        self.assertTrue(self.viewer.fit_roto_to_edges([0, 1]))
        self.assertEqual(len(self.window.dispatcher.undo_stack), undo + 1)
        self.assertNotEqual(self.shapes(), before)
        self.window.command({"op": "undo"})
        self.assertEqual(self.shapes(), before)

    def test_autokey_on_writes_a_key_at_the_current_frame_only(self):
        self.fit_setup(autokey=True)
        self.assertTrue(self.viewer.fit_roto_to_edges())
        frame = int(self.window.dispatcher.document["time"]["current"])
        for point in self.shapes()[0]["points"]:
            for name in ("x", "y"):
                value = point[name]
                self.assertIsInstance(value, dict)
                self.assertEqual([k["frame"] for k in value["curve"]["keys"]], [frame])

    def test_autokey_off_changes_the_static_shape(self):
        self.fit_setup(autokey=False)
        self.assertTrue(self.viewer.fit_roto_to_edges())
        for point in self.shapes()[0]["points"]:
            self.assertIsInstance(point["x"], float)

    def test_locked_shapes_and_shapes_in_locked_layers_are_skipped(self):
        locked = circle_shape(np.linspace(-3, 3, 16), name="locked", locked=True)
        layered = circle_shape(np.linspace(-3, 3, 16), name="layered", layer="L")
        free = circle_shape(np.linspace(-3, 3, 16), name="free")
        self.reopen([locked, layered, free], autokey=False,
                    layers=[{"name": "L", "visible": True, "locked": True}])
        before = copy.deepcopy(self.shapes())
        self.assertTrue(self.viewer.fit_roto_to_edges([0, 1, 2]))
        after = self.shapes()
        self.assertEqual(after[0], before[0])
        self.assertEqual(after[1], before[1])
        self.assertNotEqual(after[2], before[2])
        self.viewer.window.command({"op": "undo"})
        self.assertFalse(self.viewer.fit_roto_to_edges([0, 1]))

    def test_rotopaint_shapes_fit_too(self):
        document = disc_plate_document()
        dispatcher = Dispatcher(document)
        dispatcher.execute({"op": "create", "id": "p", "type": "RotoPaint", "pos": [0, 0]})
        dispatcher.execute({"op": "connect", "id": "p", "input": "image", "source": "plate"})
        shape = circle_shape(np.linspace(-3, 3, 16))
        item = {"kind": "shape", **{k: shape[k] for k in ("name", "mode", "opacity", "feather", "points", "closed")},
                "blend": "over", "visible": True}
        dispatcher.execute({"op": "set_paint_items", "id": "p", "items": [item]})
        dispatcher.execute({"op": "view", "id": "p"})
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()
        self.open(dispatcher.document)
        self.window.graph.scene().clearSelection()
        self.window.graph.items_by_id["p"].setSelected(True)
        APP.processEvents()
        self.viewer.roto_selected_shape_index = 0
        self.viewer.roto_toolbar.set_setting("p", "Select Points", "autokey", False)

        def points():
            return self.window.dispatcher.document["node_data"]["p"]["items"][0]["points"]

        before = rms_to_circle_about(static_points({"points": points()}), DISC_CENTRE, DISC_RADIUS)
        self.assertTrue(self.viewer.fit_roto_to_edges())
        after = rms_to_circle_about(static_points({"points": points()}), DISC_CENTRE, DISC_RADIUS)
        self.assertGreater(before, 1.2)
        self.assertLess(after, 0.75)

    def test_list_menu_has_fit_to_edges(self):
        self.fit_setup()
        self.window.rebuild_properties_dock()
        APP.processEvents()
        lists = [w for w in self.window.findChildren(ShapeListTree)]
        if not lists:
            self.skipTest("no Roto list in this layout")
        menu = lists[0].build_menu()
        self.assertIn("roto-list-fit-to-edges", [a.objectName() for a in menu.actions()
                                                  if a.objectName()] + [a.objectName() for sub in menu.findChildren(type(menu)) for a in sub.actions()])


def rms_to_circle_about(points, centre, radius):
    outline = roto.flatten_points(points, closed=True)
    distances = [math.hypot(x - centre[0], y - centre[1]) - radius for x, y in outline]
    return math.sqrt(sum(d * d for d in distances) / len(distances))


if __name__ == "__main__":
    unittest.main()
