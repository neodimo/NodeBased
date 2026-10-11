"""Rendering 13 step L1: the lens model on the camera, and the Undistort and Redistort nodes.

The reference for the distortion maths is written out again here from the Brown-Conrady equations (the same ones
OpenCV uses, with the focal length set to half the format height), with a plain fixed-point inverse, so the module's
own Newton solver is checked against separate code."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodebased import lensmodel, scene3d, track3d
from nodebased.core import (Dispatcher, LIMITS, SCHEMA_VERSION, SPECS, bypass_slot, empty_document,
                            upgrade_document, validate)
from nodebased.imaging import Evaluator, Raster
from nodebased.tileexec import SUPPORTED_TILED_KINDS, TileExecutor
from nodebased import tiers

LENS = dict(lens_k1=-0.03, lens_k2=0.004, lens_k3=0.0, lens_p1=0.001, lens_p2=-0.0008,
            lens_cx=0.01, lens_cy=-0.005)


def reference_distort(x, y, k1, k2, k3, p1, p2, cx, cy, squeeze=1.0):
    dx, dy = x / squeeze - cx, y - cy
    r2 = dx * dx + dy * dy
    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    return ((dx * radial + 2 * p1 * dx * dy + p2 * (r2 + 2 * dx * dx) + cx) * squeeze,
            dy * radial + p1 * (r2 + 2 * dy * dy) + 2 * p2 * dx * dy + cy)


def reference_undistort(x, y, *coefficients, squeeze=1.0, iterations=200):
    """Fixed-point inverse: keep correcting the guess by the error of the forward model."""
    gx, gy = np.array(x, float), np.array(y, float)
    for _ in range(iterations):
        fx, fy = reference_distort(gx, gy, *coefficients, squeeze=squeeze)
        gx, gy = gx - (fx - x), gy - (fy - y)
    return gx, gy


def lens_from(values, squeeze=1.0):
    return lensmodel.from_params(values, squeeze)


def make_camera(**extra):
    return scene3d.Camera(**{**LENS, **extra})


def graph(**lens):
    d = Dispatcher()
    d.execute(dict(op="create", id="plate", type="Checker", params=dict(width=480, height=272, size=48)))
    d.execute(dict(op="create", id="cam", type="Camera3D", params=lens or dict(LENS)))
    d.execute(dict(op="create", id="undistort", type="Undistort", params={}))
    d.execute(dict(op="connect", id="undistort", input="image", source="plate"))
    d.execute(dict(op="connect", id="undistort", input="camera", source="cam"))
    d.execute(dict(op="create", id="redistort", type="Redistort", params={}))
    d.execute(dict(op="connect", id="redistort", input="image", source="undistort"))
    d.execute(dict(op="connect", id="redistort", input="camera", source="cam"))
    return d


def raster(document, target, frame=1, tier=1):
    return Evaluator().evaluate_raster(dict(document, view=target), target, frame=frame, tier=tier)


class LensModelTests(unittest.TestCase):
    def test_forward_model_matches_the_brown_conrady_equations(self):
        x, y = np.meshgrid(np.linspace(-1.7, 1.7, 15), np.linspace(-1, 1, 9))
        lens = lens_from(LENS, 1.0)
        got = lensmodel.distort(lens, x, y)
        want = reference_distort(x, y, *(LENS[name] for name in lensmodel.KNOBS))
        np.testing.assert_allclose(got[0], want[0], atol=1e-12)
        np.testing.assert_allclose(got[1], want[1], atol=1e-12)

    def test_inverse_agrees_with_an_independent_solver_and_round_trips(self):
        x, y = np.meshgrid(np.linspace(-1.7, 1.7, 15), np.linspace(-1, 1, 9))
        for squeeze in (1.0, 2.0):
            lens = lens_from(LENS, squeeze)
            xd, yd = lensmodel.distort(lens, x, y)
            ux, uy = lensmodel.undistort(lens, xd, yd)
            np.testing.assert_allclose(ux, x, atol=1e-9)
            np.testing.assert_allclose(uy, y, atol=1e-9)
            rx, ry = reference_undistort(xd, yd, *(LENS[name] for name in lensmodel.KNOBS), squeeze=squeeze)
            np.testing.assert_allclose(ux, rx, atol=1e-6)
            np.testing.assert_allclose(uy, ry, atol=1e-6)

    def test_zero_coefficients_are_the_identity_and_barrel_pulls_the_corner_in(self):
        x, y = np.array([1.5, 0.0, -0.3]), np.array([0.8, 0.0, 0.2])
        np.testing.assert_array_equal(lensmodel.distort(lensmodel.Lens(), x, y), (x, y))
        xd, yd = lensmodel.distort(lensmodel.Lens(k1=-0.03), x, y)
        self.assertLess(np.hypot(xd[0], yd[0]), np.hypot(x[0], y[0]))

    def test_squeeze_stretches_only_the_horizontal_axis(self):
        lens = lensmodel.Lens(k1=-0.03, squeeze=2.0)
        xd, yd = lensmodel.distort(lens, np.array([1.0]), np.array([0.0]))
        # the squeezed image has x = 0.5 on the sensor, so the radial term is smaller than at x = 1
        plain = lensmodel.distort(lensmodel.Lens(k1=-0.03), np.array([1.0]), np.array([0.0]))[0]
        self.assertGreater(xd[0], plain[0])
        self.assertAlmostEqual(float(yd[0]), 0.0)

    def test_pixel_conversion_round_trips_and_centres_on_the_format(self):
        x, y = lensmodel.pixels_to_units(480.0, 270.0, 960, 540)
        self.assertEqual((float(x), float(y)), (0.0, 0.0))
        px, py = lensmodel.units_to_pixels(*lensmodel.pixels_to_units(12.5, 400.25, 960, 540), 960, 540)
        self.assertAlmostEqual(float(px), 12.5)
        self.assertAlmostEqual(float(py), 400.25)

    def test_presets_name_known_lenses(self):
        self.assertEqual(set(lensmodel.PRESETS), {"none", "mild barrel", "strong barrel", "pincushion",
                                                  "wide angle", "fisheye"})
        self.assertLess(lensmodel.PRESETS["fisheye"]["lens_k1"], lensmodel.PRESETS["mild barrel"]["lens_k1"])


class CameraProjectionTests(unittest.TestCase):
    POINTS = np.array([(0.5, 0.3, 0.0), (-1.2, 0.7, 0.5), (1.4, -0.6, -0.4), (0.0, 0.0, 0.0)], np.float64)

    def test_a_camera_without_a_lens_projects_exactly_as_before(self):
        camera = scene3d.Camera()
        np.testing.assert_array_equal(scene3d.project_lens(camera, 640, 360, self.POINTS)[0],
                                      scene3d.project(camera, 640, 360, self.POINTS)[0])

    def test_projection_goes_through_the_lens(self):
        camera = make_camera()
        ideal, _ = scene3d.project(camera, 640, 360, self.POINTS)
        lensed, depth = scene3d.project_lens(camera, 640, 360, self.POINTS)
        self.assertGreater(float(np.abs(lensed - ideal).max()), 1.0)
        want = lensmodel.distort_pixels(lensmodel.from_camera(camera), ideal[:, 0], ideal[:, 1], 640, 360)
        np.testing.assert_allclose(lensed, np.column_stack(want), atol=1e-3)
        np.testing.assert_allclose(depth, scene3d.project(camera, 640, 360, self.POINTS)[1])

    def test_a_point_projected_through_the_lens_and_back_is_the_ideal_projection(self):
        camera = make_camera(lens_k1=-0.05, anamorphic_squeeze=1.5)
        ideal, _ = scene3d.project(camera, 640, 360, self.POINTS)
        lensed, _ = scene3d.project_lens(camera, 640, 360, self.POINTS)
        np.testing.assert_allclose(scene3d.unproject_lens(camera, 640, 360, lensed), ideal, atol=2e-3)

    def test_a_tracked_point_triangulates_through_the_lens(self):
        cameras = [make_camera(transform=scene3d.Transform3D(position=scene3d.Vec3(x, 0.4, 5.0)))
                   for x in (-1.5, 0.0, 1.5)]
        world = np.array((0.4, 0.2, 0.3))
        plate_pixels = [scene3d.project_lens(c, 640, 360, world[None, :])[0][0] for c in cameras]
        xyz, residuals = track3d.triangulate(cameras, plate_pixels, 640, 360)
        np.testing.assert_allclose(xyz, world, atol=1e-2)
        self.assertLess(max(residuals), 0.05)
        # the same pixels read as an ideal camera's give a visibly wrong point
        flat = [scene3d.replace(c, lens_k1=0.0, lens_k2=0.0, lens_p1=0.0, lens_p2=0.0) for c in cameras]
        wrong, _ = track3d.triangulate(flat, plate_pixels, 640, 360)
        self.assertGreater(float(np.linalg.norm(wrong - world)), 5 * float(np.linalg.norm(xyz - world)) + 1e-3)

    def test_reconcile_reports_the_plate_pixel(self):
        camera = make_camera()
        found = track3d.reconcile(camera, (0.5, 0.3, 0.0), 640, 360)
        want = scene3d.project_lens(camera, 640, 360, np.array([[0.5, 0.3, 0.0]]))[0][0]
        self.assertAlmostEqual(found["x"], float(want[0]), places=3)
        self.assertAlmostEqual(found["y"], float(want[1]), places=3)

    def test_project3d_samples_the_plate_where_the_lens_put_the_point(self):
        camera = make_camera(lens_k1=-0.08)
        texture = np.zeros((108, 192, 4), np.float32)
        projection = scene3d.Projection(camera=camera, texture=texture)
        uv, _ = scene3d._projection_uv(projection, self.POINTS)
        lensed, _ = scene3d.project_lens(camera, 192, 108, self.POINTS)
        np.testing.assert_allclose(uv[:, 0], lensed[:, 0] / 192, atol=1e-5)
        np.testing.assert_allclose(uv[:, 1], 1 - lensed[:, 1] / 108, atol=1e-5)
        flat_uv, _ = scene3d._projection_uv(scene3d.Projection(camera=scene3d.replace(
            camera, lens_k1=0.0, lens_k2=0.0, lens_p1=0.0, lens_p2=0.0), texture=texture), self.POINTS)
        self.assertGreater(float(np.abs(uv - flat_uv).max()), 0.002)

    def test_camera_node_carries_the_lens_and_imported_cameras_can_too(self):
        d = Dispatcher()
        d.execute(dict(op="create", id="cam", type="Camera3D", params=dict(LENS, anamorphic_squeeze=1.3)))
        camera = scene3d.camera_from_node(d.document["nodes"]["cam"])
        self.assertEqual(lensmodel.from_camera(camera), lens_from(LENS, 1.3))
        for kind in ("ReadAlembicCamera3D", "ReadUSDCamera3D"):
            self.assertTrue(set(lensmodel.KNOBS) <= set(SPECS[kind]["params"]))
        applied = scene3d.with_node_lens(scene3d.Camera(), LENS)
        self.assertEqual(applied.lens_k1, LENS["lens_k1"])
        self.assertEqual(scene3d.with_node_lens(scene3d.Camera(), {}), scene3d.Camera())

    def test_lens_knobs_are_animatable_and_limited(self):
        for name in lensmodel.KNOBS:
            self.assertIn(name, LIMITS)
            self.assertIn(name, SPECS["Camera3D"]["params"])
        d = Dispatcher()
        d.execute(dict(op="create", id="cam", type="Camera3D", params={}))
        d.execute(dict(op="set_key", id="cam", param="lens_k1", frame=1, value=0.0))
        d.execute(dict(op="set_key", id="cam", param="lens_k1", frame=11, value=-0.04))
        values = []
        from nodebased.animation import resolve_params
        node = d.document["nodes"]["cam"]
        curves = d.document["animation"]["curves"]["cam"]
        for frame in (1, 6, 11):
            values.append(resolve_params(node, curves, frame, SPECS["Camera3D"]["params"], LIMITS)["lens_k1"])
        self.assertAlmostEqual(values[1], -0.02, places=6)
        self.assertEqual((values[0], values[2]), (0.0, -0.04))


class SchemaTests(unittest.TestCase):
    def test_an_older_document_gains_a_lens_that_changes_nothing(self):
        d = Dispatcher()
        d.execute(dict(op="create", id="cam", type="Camera3D", params=dict(tx=1.0)))
        old = copy.deepcopy(d.document)
        old["version"] = 20
        for name in lensmodel.KNOBS:
            del old["nodes"]["cam"]["params"][name]
        upgraded = upgrade_document(copy.deepcopy(old))
        validate(upgraded)
        self.assertEqual(upgraded["version"], SCHEMA_VERSION)
        params = upgraded["nodes"]["cam"]["params"]
        for name in lensmodel.KNOBS:
            self.assertEqual(params[name], 0.0)
        self.assertEqual(params["tx"], 1.0)
        camera = scene3d.camera_from_node(upgraded["nodes"]["cam"])
        self.assertFalse(lensmodel.active(camera))
        points = np.array([[0.2, 0.1, 0.0]])
        np.testing.assert_array_equal(scene3d.project_lens(camera, 320, 180, points)[0],
                                      scene3d.project(camera, 320, 180, points)[0])

    def test_a_current_document_keeps_the_lens_it_saved(self):
        d = Dispatcher()
        d.execute(dict(op="create", id="cam", type="Camera3D", params=dict(LENS)))
        again = upgrade_document(json.loads(json.dumps(d.document)))
        self.assertEqual(again["nodes"]["cam"]["params"]["lens_k1"], LENS["lens_k1"])

    def test_new_nodes_are_specified(self):
        for kind in ("Undistort", "Redistort"):
            self.assertEqual(SPECS[kind]["inputs"], ["image"])
            self.assertIn("camera", SPECS[kind]["optional_inputs"])


class WarpTests(unittest.TestCase):
    def test_undistort_then_redistort_a_checker_is_the_identity_away_from_the_border(self):
        d = graph()
        plate = raster(d.document, "plate")
        undistorted = raster(d.document, "undistort")
        back = raster(d.document, "redistort")
        self.assertEqual(back.data, plate.data)
        self.assertEqual(back.display, plate.display)
        border = 30
        inside = (slice(border, -border), slice(border, -border))
        error = np.abs(back.pixels[inside] - plate.pixels[inside])
        self.assertLess(float(error.mean()), 0.03)
        # the straightened plate is a different picture
        self.assertGreater(float(np.abs(undistorted.fit(plate.display) - plate.pixels).mean()), 0.01)

    def test_undistort_grows_the_data_window_and_records_the_overscan(self):
        d = graph(lens_k1=-0.03)
        undistorted = raster(d.document, "undistort")
        plate = raster(d.document, "plate")
        self.assertEqual(undistorted.display, plate.display)
        self.assertLess(undistorted.data.x, 0)
        self.assertLess(undistorted.data.y, 0)
        self.assertGreater(undistorted.data.right, plate.display.right)
        self.assertGreater(undistorted.data.bottom, plate.display.bottom)
        self.assertEqual(undistorted.pixels.shape[:2], (undistorted.data.height, undistorted.data.width))
        # nothing is cropped: the plate's corners land inside the grown window
        for corner in ((0, 0), (480, 0), (0, 272), (480, 272)):
            ux, uy = lensmodel.undistort_pixels(lens_from({"lens_k1": -0.03}), *corner, 480, 272)
            self.assertGreaterEqual(float(ux), undistorted.data.x)
            self.assertLessEqual(float(ux), undistorted.data.right)
            self.assertGreaterEqual(float(uy), undistorted.data.y)
            self.assertLessEqual(float(uy), undistorted.data.bottom)
        # and the corner of the plate is really there (not black) in the overscan
        self.assertGreater(float(undistorted.pixels[..., 3].min()), -1)
        self.assertGreater(float(undistorted.pixels[..., 3].max()), 0.99)

    def test_a_synthetic_distortion_of_a_render_is_undone(self):
        width, height = 192, 108
        checks = (np.indices((16, 16)).sum(axis=0) % 2).astype(np.float32)
        texture = np.repeat(checks[..., None], 4, axis=2) * 0.8 + 0.1
        texture[..., 3] = 1.0
        card = scene3d._card(4.0, 2.25, (1, 1, 1, 1), scene3d.Transform3D(), texture=texture)
        ideal_camera = scene3d.Camera(transform=scene3d.Transform3D(position=scene3d.Vec3(0, 0, 3.2)))
        truth = scene3d.render(scene3d.Scene((card,)), ideal_camera, width, height, samples=1)
        # the plate: every plate pixel reads the ideal render where the independent inverse says it came from
        coefficients = tuple(LENS[name] for name in lensmodel.KNOBS)
        gx, gy = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
        ux, uy = lensmodel.pixels_to_units(gx, gy, width, height)
        rx, ry = reference_undistort(ux, uy, *coefficients)
        qx, qy = lensmodel.units_to_pixels(rx, ry, width, height)
        plate = Evaluator._resample(truth, (qx - 0.5).astype(np.float32), (qy - 0.5).astype(np.float32), "bilinear")
        camera = make_camera()
        warped = Evaluator._lens_warp("Undistort", {"lens_preset": "custom", "filter": "bilinear"},
                                      [Raster.of(plate), camera])
        got = warped.fit(Raster.of(plate).display)
        border = 14
        inside = (slice(border, -border), slice(border, -border))
        error = np.abs(got[inside] - truth[inside])
        self.assertLess(float(error.mean()), 0.03)
        self.assertLess(float((error.max(axis=2) > 0.3).mean()), 0.04)
        # leaving the plate alone is much worse
        self.assertGreater(float(np.abs(plate[inside] - truth[inside]).mean()), 2 * float(error.mean()))

    def test_no_lens_and_no_camera_passes_the_image_through(self):
        d = Dispatcher()
        d.execute(dict(op="create", id="plate", type="Checker", params=dict(width=96, height=54, size=8)))
        d.execute(dict(op="create", id="u", type="Undistort", params={}))
        d.execute(dict(op="connect", id="u", input="image", source="plate"))
        np.testing.assert_array_equal(raster(d.document, "u").pixels, raster(d.document, "plate").pixels)

    def test_a_preset_chosen_in_the_node_stands_in_for_a_camera(self):
        d = Dispatcher()
        d.execute(dict(op="create", id="plate", type="Checker", params=dict(width=192, height=108, size=16)))
        d.execute(dict(op="create", id="u", type="Undistort", params=dict(lens_preset="mild barrel")))
        d.execute(dict(op="connect", id="u", input="image", source="plate"))
        d.execute(dict(op="create", id="cam", type="Camera3D", params=dict(lens_k1=-0.02)))
        d.execute(dict(op="create", id="uc", type="Undistort", params={}))
        d.execute(dict(op="connect", id="uc", input="image", source="plate"))
        d.execute(dict(op="connect", id="uc", input="camera", source="cam"))
        a, b = raster(d.document, "u"), raster(d.document, "uc")
        self.assertEqual(a.data, b.data)
        np.testing.assert_allclose(a.pixels, b.pixels, atol=1e-6)
        self.assertNotEqual(a.data, raster(d.document, "plate").data)

    def test_the_wired_camera_wins_over_the_node_knobs(self):
        d = Dispatcher()
        d.execute(dict(op="create", id="plate", type="Checker", params=dict(width=192, height=108, size=16)))
        d.execute(dict(op="create", id="cam", type="Camera3D", params={}))
        d.execute(dict(op="create", id="u", type="Undistort", params=dict(lens_k1=-0.05)))
        d.execute(dict(op="connect", id="u", input="image", source="plate"))
        d.execute(dict(op="connect", id="u", input="camera", source="cam"))
        np.testing.assert_array_equal(raster(d.document, "u").pixels, raster(d.document, "plate").pixels)

    def test_filters_are_those_of_transform(self):
        from nodebased.core import CHOICES, TRANSFORM_FILTERS
        self.assertEqual(CHOICES["filter"], list(TRANSFORM_FILTERS))
        outputs = {}
        for filt in TRANSFORM_FILTERS:
            d = graph()
            d.execute(dict(op="set", id="undistort", param="filter", value=filt))
            outputs[filt] = raster(d.document, "undistort").pixels
        self.assertGreater(float(np.abs(outputs["nearest"] - outputs["bilinear"]).mean()), 0)
        self.assertGreater(float(np.abs(outputs["cubic"] - outputs["bilinear"]).mean()), 0)

    def test_bypass_passes_the_image(self):
        d = graph()
        d.execute(dict(op="disable", id="undistort", value=True))
        self.assertEqual(bypass_slot(d.document["nodes"]["undistort"]), "image")
        np.testing.assert_array_equal(raster(d.document, "undistort").pixels, raster(d.document, "plate").pixels)

    def test_an_unwired_image_says_so(self):
        d = Dispatcher()
        d.execute(dict(op="create", id="u", type="Undistort", params=dict(lens_k1=-0.03)))
        with self.assertRaises(ValueError):
            raster(d.document, "u")


class RoiAndProxyTests(unittest.TestCase):
    def test_the_warp_asks_for_the_whole_image_and_is_not_tiled(self):
        for kind in ("Undistort", "Redistort"):
            regions = tiers.input_regions(kind, SPECS[kind]["params"], tiers.Region(10, 10, 32, 32), 2)
            self.assertIsNone(regions[0])
            self.assertNotIn(kind, SUPPORTED_TILED_KINDS)

    def test_the_tile_executor_falls_back_to_the_full_frame_result(self):
        d = graph()
        for target in ("undistort", "redistort"):
            document = dict(d.document, view=target)
            executor = TileExecutor(evaluator=Evaluator())
            self.assertFalse(executor.supports_tiled(document, target))
            tiled = executor.compose(document, target, frame=1, tier=1).pixels
            full = Evaluator().evaluate(document, target, frame=1)
            np.testing.assert_array_equal(tiled, full)

    def test_a_proxy_tier_agrees_with_the_full_resolution_result(self):
        d = graph()
        full = raster(d.document, "redistort", tier=1)
        for tier in (2, 4):
            small = raster(d.document, "redistort", tier=tier)
            self.assertEqual(small.pixels.shape[0], full.pixels.shape[0] // tier)
            reduced = full.pixels.reshape(full.pixels.shape[0] // tier, tier,
                                          full.pixels.shape[1] // tier, tier, 4).mean(axis=(1, 3))
            margin = 40 // tier
            inside = (slice(margin, -margin), slice(margin, -margin))
            self.assertLess(float(np.abs(small.pixels[inside] - reduced[inside]).mean()), 0.06)
        # the overscan scales with the tier too
        undistorted_full = raster(d.document, "undistort", tier=1)
        undistorted_half = raster(d.document, "undistort", tier=2)
        self.assertAlmostEqual(undistorted_half.data.width, undistorted_full.data.width / 2, delta=2)
        self.assertAlmostEqual(undistorted_half.data.x, undistorted_full.data.x / 2, delta=2)

    def test_lens_knobs_are_not_pixel_units(self):
        for kind in ("Undistort", "Redistort"):
            self.assertNotIn(kind, tiers.PIXEL_UNIT_PARAMS)


class SceneStateTests(unittest.TestCase):
    def test_the_export_writes_the_lens(self):
        from nodebased import scene_state
        camera = make_camera(anamorphic_squeeze=1.25)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "shot.json"
            scene_state.write_scene_state(path, {1: {"scene": scene3d.Scene(), "camera": camera}},
                                          resolution=(640, 360))
            written = json.loads(path.read_text())
            lens = written["frames"][0]["camera_data"]["lens"]
            self.assertEqual(lens["model"], "brown_conrady")
            self.assertEqual(lens["k1"], LENS["lens_k1"])
            self.assertEqual(lens["p2"], LENS["lens_p2"])
            self.assertEqual(lens["principal_point"], [LENS["lens_cx"], LENS["lens_cy"]])
            self.assertEqual(lens["anamorphic_squeeze"], 1.25)
            self.assertTrue(lens["distorted"])
            read = scene_state.read_scene_state(path)
            self.assertEqual(read["frames"][0]["camera"].lens_k1, LENS["lens_k1"])
            path2 = Path(folder) / "flat.json"
            scene_state.write_scene_state(path2, {1: {"scene": scene3d.Scene(), "camera": scene3d.Camera()}})
            self.assertFalse(json.loads(path2.read_text())["frames"][0]["camera_data"]["lens"]["distorted"])


class LensPanelTests(unittest.TestCase):
    def setUp(self):
        from tests.test_desktop import APP, Window
        self.app, self.window = APP, Window()
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        from tests.test_desktop import release_window
        self.addCleanup(release_window, self)
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        self.app.processEvents()

    def panel_texts(self, kind):
        from PySide6.QtWidgets import QLabel
        self.window.dispatcher.execute({"op": "create", "id": "n1", "type": kind, "pos": [0, 0]})
        panel = self.window.build_node_panel("n1")
        return [label.text() for label in panel.findChildren(QLabel)], panel

    def test_camera_properties_show_a_lens_section_with_a_sentence_for_each_knob(self):
        from nodebased.knobs import LENS_HELP
        texts, panel = self.panel_texts("Camera3D")
        self.assertIn("LENS", texts)
        for name in lensmodel.KNOBS + ("anamorphic_squeeze",):
            self.assertIn(LENS_HELP[name], texts, name)
        self.assertIn("Radial k1", texts)
        self.assertIn("Principal point x", texts)
        self.assertLess(texts.index("LENS"), texts.index("Radial k1"))

    def test_the_warp_nodes_show_the_same_lens_section(self):
        for kind in ("Undistort", "Redistort"):
            texts, _ = self.panel_texts(kind)
            self.assertIn("Lens preset", texts)
            self.assertIn("LENS", texts)
            self.assertIn("Tangential p1", texts)
            self.window.dispatcher.execute({"op": "delete", "id": "n1"})


if __name__ == "__main__":
    unittest.main()
