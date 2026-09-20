"""Gaussian splats in the editor viewport: GPU discs, CPU fallback points, framing.

The viewport shows a layout proxy, never the Render3D look. GPU cases skip without an adapter.
"""
import os
import unittest
from dataclasses import replace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from nodebased import gpu3d, scene3d as s, splatraster, splats, splatshade, viewportgpu
from nodebased.viewport3d import Viewport3D

BLACK = (0, 0, 0, 1)
CAMERA = s.Camera(s.Transform3D(s.Vec3(0, 0, 5)), s.Vec3(), 45.0, 0.1, 100.0)
IDENTITY = (1, 0, 0, 0)


def cloud(positions, colours, scales=0.2, opacity=0.9, rotations=None):
    """Degree-0 cloud with linear-light colours."""
    positions = np.atleast_2d(np.asarray(positions, np.float32))
    count = len(positions)
    sh = ((np.broadcast_to(np.asarray(colours, np.float32), (count, 3)) - 0.5) / splats.C0)[:, None, :]
    return splats.SplatCloud(
        positions, np.broadcast_to(np.asarray(scales, np.float32), (count, 3)),
        np.broadcast_to(np.asarray(IDENTITY if rotations is None else rotations, np.float32), (count, 4)),
        np.broadcast_to(np.float32(opacity), (count,)), sh, 0, colorspace="linear")


def srgb8(linear):
    linear = np.clip(np.asarray(linear, np.float64), 0, 1)
    return np.round(255 * np.where(linear <= 0.0031308, linear * 12.92,
                                   1.055 * np.power(np.maximum(linear, 1e-9), 1 / 2.4) - 0.055))


def pixel(camera, width, height, point):
    xy, _z = s.project(camera, width, height, np.asarray(point, np.float32)[None])
    return int(xy[0, 1]), int(xy[0, 0])


class Proxy(unittest.TestCase):
    def test_rows_hold_local_position_middle_axis_radius_colour_normal_and_confidence(self):
        flat = cloud(((1, 2, 3),), (0.2, 0.4, 0.6), scales=(0.3, 0.01, 0.2), opacity=0.7)
        rows, stride = viewportgpu.splat_proxy(flat)
        self.assertEqual((rows.shape, rows.dtype, stride), ((1, 12), np.float32, 1))
        np.testing.assert_allclose(rows[0, :3], (1, 2, 3))
        self.assertAlmostEqual(rows[0, 3], 0.2 * viewportgpu.SPLAT_SIGMA, places=6)
        np.testing.assert_allclose(rows[0, 4:7], (0.2, 0.4, 0.6), atol=1e-6)
        self.assertAlmostEqual(rows[0, 7], 0.7, places=6)
        np.testing.assert_allclose(rows[0, 8:11], (0, 1, 0), atol=1e-6)   # shortest axis
        np.testing.assert_allclose(rows[0, 8:11], flat.normals()[0], atol=1e-6)
        self.assertAlmostEqual(rows[0, 11], splatshade.normal_confidence(flat.scales)[0], places=6)

    def test_large_clouds_are_strided_evenly_and_never_exceed_the_limit(self):
        big = cloud(np.arange(3000, dtype=np.float32).reshape(1000, 3), (0.5, 0.5, 0.5))
        for limit, stride in ((1000, 1), (999, 2), (100, 10), (7, 143)):
            rows, used = viewportgpu.splat_proxy(big, limit)
            self.assertEqual(used, stride)
            self.assertLessEqual(len(rows), limit)
            np.testing.assert_array_equal(rows[:, :3], big.positions[::stride])

    def test_strided_discs_grow_to_keep_surfaces_closed_up_to_a_cap(self):
        instance = s.SplatInstance(cloud(((0, 0, 0),), (1, 1, 1)), scale_scale=2.0,
                                   matrix=np.diag((3.0, 3.0, 3.0, 1.0)))
        self.assertAlmostEqual(viewportgpu.splat_radius_scale(instance, 1), 6.0)
        self.assertAlmostEqual(viewportgpu.splat_radius_scale(instance, 4), 12.0)
        self.assertAlmostEqual(viewportgpu.splat_radius_scale(instance, 400), 24.0)


@unittest.skipUnless(gpu3d.available(), "no wgpu adapter")
class GPUSplats(unittest.TestCase):
    def setUp(self):
        self.gpu = viewportgpu.renderer()
        self.assertIsNotNone(self.gpu, viewportgpu.failure())

    def frame(self, scene, camera=CAMERA, size=(128, 128), **kwargs):
        return self.gpu.render(scene, camera, *size, BLACK, headlight=not scene.lights, **kwargs)

    def test_disc_lands_on_the_projected_centre_in_its_dc_colour(self):
        points = ((-1, 0.5, 0), (1, -0.5, 1))
        colours = ((0.8, 0.2, 0.1), (0.1, 0.3, 0.9))
        scene = s.Scene(splats=(s.SplatInstance(cloud(points, colours, scales=0.05)),))
        frame = self.frame(scene)
        for point, colour in zip(points, colours):
            y, x = pixel(CAMERA, 128, 128, point)
            np.testing.assert_allclose(frame[y, x, :3], srgb8(colour), atol=1)
        self.assertGreater((frame[..., :3].max(axis=2) == 0).mean(), 0.9)   # the rest stays background

    def test_instance_transform_moves_discs_without_a_new_upload(self):
        shared = cloud(((0, 0, 0),), (0, 1, 0), scales=0.05)
        self.frame(s.Scene(splats=(s.SplatInstance(shared),)))
        uploads = self.gpu.uploads
        matrix = np.eye(4)
        matrix[:3, 3] = (1, 1, 0)
        frame = self.frame(s.Scene(splats=(s.SplatInstance(shared, matrix=matrix),)))
        self.assertEqual(self.gpu.uploads, uploads)
        y, x = pixel(CAMERA, 128, 128, (1, 1, 0))
        np.testing.assert_allclose(frame[y, x, :3], (0, 255, 0), atol=1)
        np.testing.assert_array_equal(frame[64, 64, :3], (0, 0, 0))
        self.frame(s.Scene())
        self.assertEqual(self.gpu._splats, {})   # evicted with its cloud

    def test_discs_and_meshes_occlude_each_other_by_depth(self):
        wall = s._card(1, 1, (1, 0, 0, 1), s.Transform3D())
        for z, expected in ((-1.0, (255, 0, 0)), (1.0, (0, 0, 255))):
            scene = s.Scene((wall,), (), (s.SplatInstance(cloud(((0, 0, z),), (0, 0, 1), scales=0.1)),))
            frame = self.gpu.render(scene, CAMERA, 128, 128, BLACK, headlight=False, ambient=1.0)
            np.testing.assert_allclose(frame[64, 64, :3], expected, atol=1)

    def test_faint_splats_are_hidden_and_the_opacity_scale_counts(self):
        one = cloud(((0, 0, 0),), (1, 1, 1), scales=0.1, opacity=0.5)
        shown = self.frame(s.Scene(splats=(s.SplatInstance(one),)))
        hidden = self.frame(s.Scene(splats=(s.SplatInstance(one, opacity_scale=0.05),)))
        self.assertEqual(shown[64, 64, 0], 255)
        self.assertEqual(int(hidden[..., :3].max()), 0)

    def test_disc_size_is_clamped_between_one_pixel_and_the_maximum(self):
        largest = viewportgpu.SPLAT_MAX_PIXELS
        for scale, low, high in ((1e-6, 1, 9), (50.0, 1.5 * largest ** 2, (2 * largest + 2) ** 2)):
            lit = self.frame(s.Scene(splats=(s.SplatInstance(cloud(((0, 0, 0),), (1, 1, 1), scales=scale)),)))
            covered = int((lit[..., 0] > 30).sum())   # multisampled edges count
            self.assertGreaterEqual(covered, low, scale)
            self.assertLessEqual(covered, high, scale)

    def test_relight_follows_shade_splats_and_zero_keeps_the_baked_colour(self):
        # Flat splats (confident normals): one faces the light, one is turned a quarter away.
        turned = (np.cos(np.pi / 8), 0, np.sin(np.pi / 8), 0)   # 45 degrees about Y
        flats = cloud(((-1, 0, 0), (1, 0, 0)), (0.6, 0.5, 0.4), scales=(0.05, 0.05, 0.0005),
                      rotations=(IDENTITY, turned))
        light = s.Light(intensity=0.7, position=s.Vec3(0, 0, 5))   # travels along -Z, onto the first splat
        baked = s.SplatInstance(flats)
        for instance in (baked, replace(baked, relight=1.0), replace(baked, relight=0.5)):
            scene = s.Scene((), (light,), (instance,))
            frame = self.gpu.render(scene, CAMERA, 128, 128, BLACK, headlight=False, ambient=0.15)
            eye = s._view_basis(CAMERA)[0]
            expected = splatshade.instance_colors(instance, eye, scene.lights, 0.15)
            for point, colour in zip(flats.positions, expected):
                y, x = pixel(CAMERA, 128, 128, point)
                np.testing.assert_allclose(frame[y, x, :3], srgb8(colour), atol=2, err_msg=str(instance.relight))
        self.assertGreater(expected[0].sum(), 0)

    def test_a_million_discs_stay_interactive_and_report_their_stride(self):
        import time
        rng = np.random.default_rng(7)
        count = viewportgpu.MAX_SPLATS * 2 + 1
        many = splats.SplatCloud(
            rng.uniform(-2, 2, (count, 3)).astype(np.float32), np.full((count, 3), 0.01, np.float32),
            np.broadcast_to(np.float32(IDENTITY), (count, 4)), np.full(count, 0.9, np.float32),
            np.zeros((count, 1, 3), np.float32), 0)
        scene = s.Scene(splats=(s.SplatInstance(many),))
        self.frame(scene, size=(640, 360))   # upload
        self.assertEqual(self.gpu.splat_stride, 3)
        start = time.perf_counter()
        for _ in range(5):
            self.frame(scene, size=(640, 360))
        self.assertLess((time.perf_counter() - start) / 5, 0.1)


class Widget(unittest.TestCase):
    def scene(self, z):
        # Everything sits off the world axes: the editor draws those on top of the picture.
        wall = s._card(1, 1, (1, 0, 0, 1), s.Transform3D(s.Vec3(1, 1, 0)))
        return s.Scene((wall,), (), (s.SplatInstance(cloud(((1, 1, z), (-2, 1, 0)), (0, 0, 1), scales=5.0)),))

    def paint(self, widget, scene, renderer=None):
        widget._scene_cache = ((widget._frame(), None), (scene, CAMERA))
        widget.look_through = True
        with patch.object(viewportgpu, "renderer", return_value=renderer), \
                patch.object(splatraster, "prepare_splats", side_effect=AssertionError("splat rasterizer used")):
            image = widget.grab().toImage()
        return image

    def test_cpu_fallback_marks_depth_tested_centres_and_never_rasterizes_splats(self):
        app = QApplication.instance() or QApplication([])
        widget = Viewport3D()
        widget.resize(320, 220)
        widget.document = {"nodes": {}, "view": None}
        for z, blue in ((1.0, True), (-1.0, False)):
            image = self.paint(widget, self.scene(z))
            self.assertEqual(widget.status, "")
            y, x = pixel(CAMERA, 320, 220, (1, 1, z))
            centre = image.pixelColor(x, y)
            # Hidden behind the wall, the headlit red card shows instead.
            self.assertEqual((centre.blue() > 200, centre.red() > centre.blue() + 50), (blue, not blue))
            y, x = pixel(CAMERA, 320, 220, (-2, 1, 0))
            self.assertGreater(image.pixelColor(x, y).blue(), 200)   # beside the wall: always seen
        self.assertEqual(widget.splat_note, "splats: 2 shown as points (layout proxy, not the render)")
        widget.close()

    def test_cpu_fallback_strides_large_clouds_and_says_so(self):
        app = QApplication.instance() or QApplication([])
        widget = Viewport3D()
        widget.resize(160, 120)
        widget.document = {"nodes": {}, "view": None}
        count = 10
        line = cloud(np.column_stack((np.linspace(-1, 1, count), np.zeros(count), np.zeros(count))), (1, 1, 1))
        with patch("nodebased.viewport3d.CPU_SPLATS", 5):
            self.paint(widget, s.Scene(splats=(s.SplatInstance(line),)))
        self.assertEqual(widget.splat_note, "splats: 1 in 2 of 10 shown as points (layout proxy, not the render)")
        self.paint(widget, s.Scene())
        self.assertEqual((widget.splat_note, widget._splat_points), ("", {}))
        widget.close()

    @unittest.skipUnless(gpu3d.available(), "no wgpu adapter")
    def test_gpu_widget_reports_discs(self):
        app = QApplication.instance() or QApplication([])
        widget = Viewport3D()
        widget.resize(320, 220)
        widget.document = {"nodes": {}, "view": None}
        with patch.object(s, "render", side_effect=AssertionError("CPU renderer used")):
            self.paint(widget, self.scene(1.0), viewportgpu.renderer())
        self.assertEqual(widget.status, "")
        self.assertEqual(widget.splat_note, "splats: 2 shown as discs (layout proxy, not the render)")
        widget.close()

    def test_framing_covers_the_bulk_of_a_cloud_and_ignores_stray_splats(self):
        app = QApplication.instance() or QApplication([])
        rng = np.random.default_rng(3)
        positions = rng.uniform(-1, 1, (2000, 3)).astype(np.float32)
        positions[:5] = 5000   # strays far outside the capture
        matrix = np.eye(4)
        matrix[:3, 3] = (10, 0, 0)
        widget = Viewport3D()
        widget.document = {"nodes": {}, "view": None}
        widget._scene_cache = ((widget._frame(), None),
                               (s.Scene(splats=(s.SplatInstance(cloud(positions, (1, 1, 1)), matrix=matrix),)), None))
        widget.frame_scene()
        np.testing.assert_allclose(widget.center, (10, 0, 0), atol=0.15)
        self.assertLess(widget.distance, 10)
        widget.close()


if __name__ == "__main__":
    unittest.main()
