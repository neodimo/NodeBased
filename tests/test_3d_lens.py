"""Depth of field (lane L4 step R5, part 1): the thin lens of Camera3D in the path tracer and the ray-traced mode.

docs/3D_FOUNDATION.md "Depth of field". Every image is tiny so the file runs in seconds. The blur circle is
measured from the second moment of the intensity (a uniform disc of diameter c has variance c^2 / 16 per axis),
after subtracting the same point rendered sharp, and compared with the textbook thin-lens circle of confusion.
"""
import dataclasses
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodebased import core, lens, pathtrace as pt, scene3d as s
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from tests.test_3d_pathtrace import card, gpu_ready, gtrace, sphere

FOCAL = 200.0
FOV = math.degrees(2 * math.atan(18.672 / (2 * FOCAL)))   # a 200 mm lens on the default film back
SIZE = 96


def camera(**fields):
    return s.Camera(s.Transform3D(position=s.Vec3(0, 0, 0)), s.Vec3(0, 0, -1), FOV, **fields)


def lens_camera(**fields):
    return camera(fstop=1.0, focus_distance=10.0, **fields)


def dot_scene(depth):
    """One emissive dot about a pixel wide, `depth` in front of the camera."""
    return s.Scene((sphere(0.01, (1, 1, 1, 1), (0, 0, -depth), emission=1.0),))


def render_dot(depth, cam, samples=256, size=SIZE, backend="cpu"):
    settings = pt.PathSettings(samples=samples, max_bounces=0)
    if backend == "gpu":
        from nodebased import gpupathtrace
        return gpupathtrace.render(dot_scene(depth), cam, size, size, (0, 0, 0, 0), 0.0, "rgba", settings)
    return pt.render(dot_scene(depth), cam, size, size, (0, 0, 0, 0), 0.0, "rgba", settings)


def moments(image):
    """Variances along x and y and the third moment along x of the intensity, plus its total."""
    a = image[..., 0].astype(np.float64)
    h, w = a.shape
    xs, ys = np.arange(w) + 0.5, np.arange(h) + 0.5
    total = a.sum()
    mx, my = (a.sum(0) * xs).sum() / total, (a.sum(1) * ys).sum() / total
    vx, vy = (a.sum(0) * (xs - mx) ** 2).sum() / total, (a.sum(1) * (ys - my) ** 2).sum() / total
    skew = (a.sum(0) * (xs - mx) ** 3).sum() / total
    return vx, vy, skew, total


def measured_circle(depth, cam, **kwargs):
    """The blur circle diameter in pixels of the dot at `depth`, from the extra variance over the sharp render."""
    blurred = moments(render_dot(depth, cam, **kwargs))
    sharp = moments(render_dot(depth, dataclasses.replace(cam, fstop=0.0), **kwargs))
    return 4.0 * math.sqrt(max(blurred[0] - sharp[0], 0.0))


class ThinLensFormulaTests(unittest.TestCase):
    def test_circle_of_confusion_is_the_textbook_formula(self):
        cam = lens_camera()
        f, S, A = FOCAL * 1e-3, 10.0, FOCAL * 1e-3 / 1.0
        for depth in (2.0, 5.0, 10.0, 20.0, 1000.0):
            on_sensor = A * f * abs(S - depth) / (depth * (S - f))          # metres
            expected = on_sensor * 1000.0 / cam.vaperture * SIZE
            self.assertAlmostEqual(float(lens.circle_of_confusion_px(cam, depth, SIZE)), expected, places=9)
        self.assertEqual(float(lens.circle_of_confusion_px(cam, S, SIZE)), 0.0)

    def test_pinhole_is_off_and_lens_fields_are_ignored_without_an_f_stop(self):
        self.assertFalse(lens.active(s.Camera()))
        plain = render_dot(20.0, camera(), samples=8)
        fancy = render_dot(20.0, camera(focus_distance=3.0, aperture_blades=5, blade_rotation=30.0,
                                        anamorphic_squeeze=2.0), samples=8)
        np.testing.assert_array_equal(plain, fancy)

    def test_aperture_samples_fill_the_shape(self):
        u = np.random.default_rng(1).random((2, 20000))
        x, y = lens.aperture_points(u[0], u[1])
        self.assertLessEqual(float(np.hypot(x, y).max()), 1.0 + 1e-9)
        self.assertAlmostEqual(float((x * x + y * y).mean()), 0.5, delta=0.01)        # a uniform disc
        for blades in (3, 5, 8):
            x, y = lens.aperture_points(u[0], u[1], blades, 0.0)
            corners = lens.blade_vertices(blades, 0.0)
            # every sample lies inside the polygon: on the inner side of every edge
            for a, b in zip(corners, np.roll(corners, -1, axis=0)):
                edge = b - a
                side = edge[0] * (y - a[1]) - edge[1] * (x - a[0])
                self.assertTrue(np.all(side >= -1e-9), blades)
        self.assertIsNone(lens.blade_vertices(2, 0.0))


class PathTracerDepthOfFieldTests(unittest.TestCase):
    def test_in_focus_plane_is_sharp(self):
        cam = lens_camera()
        blurred, sharp = (moments(render_dot(10.0, c)) for c in (cam, dataclasses.replace(cam, fstop=0.0)))
        self.assertAlmostEqual(blurred[0], sharp[0], delta=0.02)
        self.assertAlmostEqual(blurred[1], sharp[1], delta=0.02)
        self.assertAlmostEqual(blurred[3], sharp[3], delta=0.02 * sharp[3])          # and it keeps its energy

    def test_blur_circle_matches_thin_lens_in_pixels(self):
        cam = lens_camera()
        for depth in (5.0, 20.0):                       # in front of and behind the focus plane
            with self.subTest(depth=depth):
                expected = float(lens.circle_of_confusion_px(cam, depth, SIZE))
                self.assertGreater(expected, 8.0)
                self.assertAlmostEqual(measured_circle(depth, cam), expected, delta=0.06 * expected)

    def test_a_wider_aperture_makes_a_bigger_circle(self):
        wide, narrow = (measured_circle(20.0, dataclasses.replace(lens_camera(), fstop=f), samples=128)
                        for f in (1.0, 2.0))
        self.assertAlmostEqual(wide / narrow, 2.0, delta=0.15)

    def test_blade_count_shapes_the_bokeh(self):
        """A regular polygon of n blades has E[r^2] = (2 + cos(2 pi / n)) / 6 against the disc's 1 / 2, so the
        second moment of its bokeh is sqrt((2 + cos(2 pi / n)) / 3) of the round one (0.71 for three blades)."""
        def spread(blades):
            return 4.0 * math.sqrt(moments(render_dot(5.0, lens_camera(aperture_blades=blades), samples=384))[0])
        round_ = spread(0)
        for blades in (3, 4, 6):
            with self.subTest(blades=blades):
                expected = math.sqrt((2 + math.cos(2 * math.pi / blades)) / 3)
                self.assertAlmostEqual(spread(blades) / round_, expected, delta=0.045)

    def test_blade_rotation_turns_the_polygon(self):
        # Nearer than the focus plane the aperture's image is turned around, so the corner that points right
        # at rotation 0 shows on the left; rotating the blades by 180 degrees swaps them back.
        skew = {r: moments(render_dot(5.0, lens_camera(aperture_blades=3, blade_rotation=r), samples=384))[2]
                for r in (0.0, 180.0)}
        self.assertLess(skew[0.0], 0.0)
        self.assertGreater(skew[180.0], 0.0)
        self.assertAlmostEqual(skew[0.0], -skew[180.0], delta=0.3 * abs(skew[0.0]))

    def test_anamorphic_squeeze_makes_the_bokeh_taller(self):
        vx, vy, _, _ = moments(render_dot(5.0, lens_camera(anamorphic_squeeze=2.0), samples=256))
        self.assertAlmostEqual(math.sqrt(vy / vx), 2.0, delta=0.12)

    def test_seeded_and_reproducible(self):
        cam = lens_camera()
        np.testing.assert_array_equal(render_dot(20.0, cam, samples=16), render_dot(20.0, cam, samples=16))

    def test_data_passes_stay_pinhole(self):
        scene = dot_scene(20.0)
        sharp = pt.render(scene, camera(), 32, 32, (0, 0, 0, 0), 0.0, "depth")
        lensed = pt.render(scene, lens_camera(), 32, 32, (0, 0, 0, 0), 0.0, "depth")
        np.testing.assert_array_equal(sharp, lensed)


class RayTracedDepthOfFieldTests(unittest.TestCase):
    @staticmethod
    def line_spread(cam, depth):
        """Variance of the derivative of a blurred vertical edge of a white wall at `depth` (its edge at x = 0)."""
        wall = s.Scene((card(40, 40, (1, 1, 1, 1), (20, 0, -depth)),))
        row = s.render(wall, cam, SIZE, SIZE, mode="raytrace", ambient=1.0)[..., 0].astype(np.float64).mean(0)
        d = np.diff(row)
        xs = np.arange(len(d)) + 1.0
        mean = (d * xs).sum() / d.sum()
        return float((d * (xs - mean) ** 2).sum() / d.sum())

    def test_blur_circle_matches_thin_lens_in_pixels(self):
        cam = lens_camera()
        for depth in (5.0, 20.0):
            with self.subTest(depth=depth):
                expected = float(lens.circle_of_confusion_px(cam, depth, SIZE))
                blurred = self.line_spread(cam, depth) - self.line_spread(camera(), depth)
                self.assertAlmostEqual(4.0 * math.sqrt(blurred), expected, delta=0.08 * expected)

    def test_in_focus_wall_is_sharp_and_pinhole_is_unchanged(self):
        self.assertAlmostEqual(self.line_spread(lens_camera(), 10.0), self.line_spread(camera(), 10.0), delta=0.05)
        wall = s.Scene((card(40, 40, (1, 1, 1, 1), (20, 0, -20)),))
        a = s.render(wall, camera(), 32, 32, mode="raytrace", ambient=1.0)
        b = s.render(wall, camera(focus_distance=2.0, aperture_blades=4), 32, 32, mode="raytrace", ambient=1.0)
        np.testing.assert_array_equal(a, b)


class ImportedCameraTests(unittest.TestCase):
    def test_alembic_camera_carries_f_stop_focus_and_squeeze(self):
        from nodebased.alembicio import ObjectReader, Property as Prop, camera_to_scene3d
        from tests.test_3d_alembic import SampleProperty, synthetic_xform
        core_values = [50, 3.6, 0, 2, 0, 2.0, 0, 0, 0, 0, 2.8, 7.5, 0, .02, .1, 100]
        cam = ObjectReader('c', '/p/c', {'schema': 'AbcGeom_Camera_v1'}, Prop('', {}, properties={
            '.geom': Prop('.geom', {}, properties={'.core': SampleProperty(core_values)})}))
        parent = synthetic_xform('/p', [np.eye(4)])
        parent.children['c'] = cam
        camera_value = camera_to_scene3d(parent, cam, 0)
        self.assertAlmostEqual(camera_value.fstop, 2.8)
        self.assertAlmostEqual(camera_value.focus_distance, 7.5)
        self.assertAlmostEqual(camera_value.anamorphic_squeeze, 2.0)
        core_values[10] = 0.0
        pinhole = ObjectReader('c', '/p/c', {'schema': 'AbcGeom_Camera_v1'}, Prop('', {}, properties={
            '.geom': Prop('.geom', {}, properties={'.core': SampleProperty(core_values)})}))
        parent.children['c'] = pinhole
        self.assertEqual(camera_to_scene3d(parent, pinhole, 0).fstop, 0.0)

    def test_usd_camera_carries_f_stop_and_focus_in_metres(self):
        from nodebased import usdio
        if not usdio.available():
            self.skipTest("usd-core not installed")
        from pxr import Usd, UsdGeom
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cam.usda"
            stage = Usd.Stage.CreateNew(str(path))
            camera_prim = UsdGeom.Camera.Define(stage, "/cam")
            camera_prim.CreateFocalLengthAttr(35)
            camera_prim.CreateVerticalApertureAttr(24)
            camera_prim.CreateFStopAttr(4.0)
            camera_prim.CreateFocusDistanceAttr(6.0)
            stage.GetRootLayer().Save()
            imported = usdio.load_camera(path, 1)
            self.assertAlmostEqual(imported.fstop, 4.0)
            self.assertAlmostEqual(imported.focus_distance, 6.0)
            camera_prim.CreateFStopAttr(0.0)
            stage.GetRootLayer().Save()
            self.assertEqual(usdio.load_camera(path, 1).fstop, 0.0)


class NodeTests(unittest.TestCase):
    def graph(self, **camera_params):
        d = Dispatcher()
        for key, kind, params in (("dot", "Sphere3D", dict(sphere_radius=0.01, tz=-20.0, emission=1.0)),
                                  ("camera", "Camera3D", dict(tz=0.0, focal=FOCAL, target_z=-1.0, **camera_params)),
                                  ("scene", "Scene3D", {}),
                                  ("render", "Render3D", dict(width=64, height=64, render_mode="pathtrace",
                                                              pt_samples=64, max_bounces=0))):
            d.execute(dict(op="create", id=key, type=kind, params=params))
        d.execute(dict(op="connect", id="scene", input="object0", source="dot"))
        d.execute(dict(op="connect", id="render", input="scene", source="scene"))
        d.execute(dict(op="connect", id="render", input="camera", source="camera"))
        return d

    def test_camera3d_defaults_are_a_pinhole_and_old_documents_load(self):
        params = core.SPECS["Camera3D"]["params"]
        self.assertEqual((params["fstop"], params["aperture_blades"], params["anamorphic_squeeze"]), (0.0, 0, 1.0))
        for name in ("fstop", "focus_distance", "aperture_blades", "blade_rotation", "anamorphic_squeeze"):
            self.assertIn(name, core.LIMITS)
        d = Dispatcher()
        d.execute(dict(op="create", id="camera", type="Camera3D", params={}))
        doc = d.document
        for name in ("fstop", "focus_distance", "aperture_blades", "blade_rotation", "anamorphic_squeeze"):
            doc["nodes"]["camera"]["params"].pop(name)
        upgraded = core.upgrade_document(doc)
        self.assertEqual(upgraded["nodes"]["camera"]["params"]["fstop"], 0.0)
        self.assertEqual(upgraded["nodes"]["camera"]["params"]["focus_distance"], 5.0)
        self.assertEqual(s.camera_from_node({"params": {k: v for k, v in upgraded["nodes"]["camera"]["params"].items()}}).fstop, 0.0)

    def test_render3d_shows_the_lens_of_its_camera(self):
        sharp = moments(Evaluator().evaluate(self.graph().document, "render"))
        blurred = moments(Evaluator().evaluate(self.graph(fstop=1.0, focus_distance=10.0).document, "render"))
        self.assertGreater(blurred[0], 10 * sharp[0])


class PickFocusTests(unittest.TestCase):
    def test_pick_focus_reads_the_view_depth_under_the_click(self):
        scene = s.Scene((card(0.2, 0.2, (1, 1, 1, 1), (0, 0, -7.0)),))
        cam = lens_camera()
        self.assertAlmostEqual(lens.pick_focus_distance(cam, scene, 64, 64, 32, 32), 7.0, places=4)
        self.assertIsNone(lens.pick_focus_distance(cam, scene, 64, 64, 0, 0))


@unittest.skipUnless(gpu_ready(), "wgpu adapter unavailable for the path tracer")
class GpuDepthOfFieldTests(unittest.TestCase):
    def test_gpu_blur_circle_matches_thin_lens_and_the_cpu(self):
        cam = lens_camera()
        for depth in (5.0, 20.0):
            with self.subTest(depth=depth):
                expected = float(lens.circle_of_confusion_px(cam, depth, SIZE))
                self.assertAlmostEqual(measured_circle(depth, cam, backend="gpu"), expected, delta=0.08 * expected)

    def test_gpu_blades_and_squeeze_shape_the_bokeh(self):
        triangle = lens_camera(aperture_blades=3, blade_rotation=0.0)
        gpu = render_dot(5.0, triangle, samples=384, backend="gpu")
        self.assertLess(moments(gpu)[2], 0.0)            # the same inverted corner as the CPU reference
        cpu = render_dot(5.0, triangle, samples=384)
        # both draw from the same random streams, so the two spots agree pixel for pixel bar edge cases
        self.assertGreater(float(np.mean(np.abs(gpu - cpu)[..., 0] < 2e-3)), 0.98)
        vx, vy, _, _ = moments(render_dot(5.0, lens_camera(anamorphic_squeeze=2.0), samples=256, backend="gpu"))
        self.assertAlmostEqual(math.sqrt(vy / vx), 2.0, delta=0.15)

    def test_gpu_pinhole_matches_the_cpu_still(self):
        cpu = moments(render_dot(10.0, camera(), samples=16))
        gpu = moments(render_dot(10.0, camera(), samples=16, backend="gpu"))
        self.assertAlmostEqual(cpu[3], gpu[3], delta=0.03 * cpu[3])


if __name__ == "__main__":
    unittest.main()
