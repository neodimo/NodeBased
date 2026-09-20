"""The interactive wgpu viewport: agreement with the CPU reference, caching, and fallback.

GPU cases skip without an adapter; the fallback cases never need one.
"""
import os
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from nodebased import gpu3d, scene3d as s, viewportgpu
from nodebased.core import Dispatcher
from nodebased.viewport3d import Viewport3D, _line_vertices

BACKGROUND = (0.025, 0.025, 0.03, 1.0)
CAMERA = s.Camera(s.Transform3D(s.Vec3(4, 3, 6)), s.Vec3(), 45.0, 0.07, 1000.0)


def srgb(linear):
    linear = np.clip(linear, 0, 1)
    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.power(np.maximum(linear, 1e-9), 1 / 2.4) - 0.055)


def quadrants():
    texture = np.zeros((64, 64, 4), np.float32)
    texture[..., 3] = 1
    texture[:32, :32, 0] = texture[:32, 32:, 1] = texture[32:, :32, 2] = 1
    texture[32:, 32:, :3] = 1
    return texture


def test_scene(lights=True):
    floor = s._card(4, 4, (1, 1, 1, 1), s.Transform3D(s.Vec3(0, -1, 0), s.Vec3(-90, 0, 0)), quadrants())
    glass = s._cube(1, (1, 0.3, 0.3, 0.5), s.Transform3D(s.Vec3(2, 0, 0)))
    ball = replace(s._sphere(1, 32, (0.8, 0.8, 0.9, 1), s.Transform3D()), specular=0.8, shininess=40.0)
    return s.Scene((ball, floor, glass), (s.Light(intensity=0.8),) if lights else ())


def reference(scene, width, height, headlight):
    image = s.render(scene, CAMERA, width, height, BACKGROUND, shade=headlight, ambient=0.15,
                     shadows=False, samples=2)
    return (srgb(image[..., :3]) * 255).round().astype(int)


def graph():
    d = Dispatcher()
    for key, kind in (("ball", "Sphere3D"), ("cube", "Cube3D"), ("cam", "Camera3D"),
                      ("scene", "Scene3D"), ("render", "Render3D")):
        d.execute({"op": "create", "id": key, "type": kind})
    d.execute({"op": "connect", "id": "scene", "input": "object0", "source": "ball"})
    d.execute({"op": "connect", "id": "scene", "input": "object1", "source": "cube"})
    d.execute({"op": "connect", "id": "render", "input": "scene", "source": "scene"})
    d.execute({"op": "connect", "id": "render", "input": "camera", "source": "cam"})
    d.execute({"op": "view", "id": "render"})
    return d


class ProjectionMatrix(unittest.TestCase):
    def test_clip_space_matches_scene3d_project(self):
        points = np.array(((0, 0, 0), (1, 2, -3), (-2, 0.5, 1), (0.3, -1, 4)), np.float32)
        width, height = 640, 360
        _eye, matrix = viewportgpu.view_projection(CAMERA, width, height)
        clip = (matrix @ np.column_stack((points, np.ones(len(points)))).T).T
        ndc = clip[:, :3] / clip[:, 3:4]
        pixels = np.column_stack(((ndc[:, 0] * 0.5 + 0.5) * width, (1 - (ndc[:, 1] * 0.5 + 0.5)) * height))
        expected, depth = s.project(CAMERA, width, height, points)
        np.testing.assert_allclose(pixels, expected, atol=1e-3)
        np.testing.assert_allclose(clip[:, 3], depth, rtol=1e-5)   # w is view depth
        self.assertTrue(((ndc[:, 2] > 0) & (ndc[:, 2] < 1)).all())

    def test_flat_meshes_get_face_normals_and_smooth_meshes_keep_theirs(self):
        cube = s._cube(2, (1, 1, 1, 1), s.Transform3D())
        soup = viewportgpu._soup(cube)
        self.assertEqual(soup.shape, (36, 8))
        normals = soup[:, 3:6] / np.linalg.norm(soup[:, 3:6], axis=1, keepdims=True)
        np.testing.assert_allclose(np.abs(normals).max(axis=1), 1.0, atol=1e-6)   # axis aligned
        ball = s._sphere(1, 8, (1, 1, 1, 1), s.Transform3D())
        soup = viewportgpu._soup(ball)
        np.testing.assert_allclose(soup[:, 3:6], soup[:, :3], atol=1e-6)          # unit sphere


@unittest.skipUnless(gpu3d.available(), "no wgpu adapter")
class GPUViewport(unittest.TestCase):
    def setUp(self):
        self.gpu = viewportgpu.renderer()
        self.assertIsNotNone(self.gpu, viewportgpu.failure())

    def compare(self, scene, headlight):
        width, height = 320, 200
        frame = self.gpu.render(scene, CAMERA, width, height, BACKGROUND, headlight=headlight, ambient=0.15)
        self.assertEqual(frame.shape, (height, width, 4))
        difference = np.abs(frame[..., :3].astype(int) - reference(scene, width, height, headlight))
        # Interiors agree; silhouettes differ by antialiasing pattern (4x MSAA against 2x2 supersampling).
        self.assertLess(difference.mean(), 0.6)
        self.assertLess((difference.max(axis=2) > 12).mean(), 0.03)
        return frame

    def test_lit_scene_matches_the_cpu_reference(self):
        self.compare(test_scene(), headlight=False)

    def test_headlight_scene_matches_the_cpu_reference(self):
        self.compare(test_scene(lights=False), headlight=True)

    def test_texture_is_upright(self):
        card = s._card(2, 2, (1, 1, 1, 1), s.Transform3D(), quadrants())
        frame = self.gpu.render(s.Scene((card,)), s.Camera(), 64, 64, (0, 0, 0, 1), headlight=False, ambient=1.0)
        for (y, x), colour in (((24, 24), (255, 0, 0)), ((24, 40), (0, 255, 0)),
                               ((40, 24), (0, 0, 255)), ((40, 40), (255, 255, 255))):
            np.testing.assert_array_equal(frame[y, x, :3], colour)

    def test_projection_matches_the_cpu_reference(self):
        projector = s.Camera(s.Transform3D(s.Vec3(0, 0, 4)), s.Vec3(), 30.0, 0.1, 100.0)
        card = s._card(4, 4, (1, 1, 1, 1), s.Transform3D())
        card = replace(card, projection=s.Projection(projector, quadrants()))
        self.compare(s.Scene((card,)), headlight=True)

    def test_editor_lines_are_depth_tested(self):
        wall = s._card(2, 2, (1, 1, 1, 1), s.Transform3D())
        camera = s.Camera()
        behind = _line_vertices([((-1, 0, -2), (1, 0, -2), (1, 0, 0, 1))])
        before = _line_vertices([((-1, 0, 2), (1, 0, 2), (1, 0, 0, 1))])
        hidden = self.gpu.render(s.Scene((wall,)), camera, 64, 64, (0, 0, 0, 1), lines=behind, headlight=True)
        shown = self.gpu.render(s.Scene((wall,)), camera, 64, 64, (0, 0, 0, 1), lines=before, headlight=True)
        red = lambda frame: ((frame[..., 0] > 150) & (frame[..., 1] < 80)).sum()
        self.assertEqual(red(hidden), 0)
        self.assertGreater(red(shown), 10)

    def test_frame_is_contiguous_at_widths_off_the_readback_stride(self):
        # 333 * 4 bytes is not a multiple of the 256-byte readback stride, as on a real dock.
        from PySide6.QtGui import QImage
        scene = test_scene()
        for width, height in ((333, 187), (65, 33), (64, 64)):
            frame = self.gpu.render(scene, CAMERA, width, height, BACKGROUND)
            self.assertEqual(frame.shape, (height, width, 4))
            self.assertTrue(frame.flags.c_contiguous, (width, height))
            image = QImage(frame.data, width, height, frame.strides[0], QImage.Format.Format_RGBA8888).copy()
            self.assertEqual((image.width(), image.height()), (width, height))
        wide = self.gpu.render(scene, CAMERA, 320, 187, BACKGROUND)
        odd = self.gpu.render(scene, CAMERA, 333, 187, BACKGROUND)
        # Same picture either way: the centre pixel of both frames sees the same surface.
        np.testing.assert_allclose(odd[93, 166].astype(int), wide[93, 160].astype(int), atol=12)

    def test_orbiting_and_moving_objects_upload_nothing(self):
        scene = test_scene()
        self.gpu.render(scene, CAMERA, 64, 64, BACKGROUND)
        uploads = self.gpu.uploads
        other = s.Camera(s.Transform3D(s.Vec3(-3, 2, 5)), s.Vec3(), 45.0, 0.07, 1000.0)
        self.gpu.render(scene, other, 64, 64, BACKGROUND)
        moved = s.Scene(tuple(replace(g, transform=s.Transform3D(s.Vec3(0, 1, 0))) for g in scene.geometries),
                        scene.lights)
        self.gpu.render(moved, other, 64, 64, BACKGROUND)
        self.assertEqual(self.gpu.uploads, uploads)

    def test_unused_meshes_are_evicted(self):
        self.gpu.render(test_scene(), CAMERA, 64, 64, BACKGROUND)
        self.gpu.render(s.Scene(), CAMERA, 64, 64, BACKGROUND)
        self.assertEqual((len(self.gpu._meshes), len(self.gpu._textures)), (0, 0))

    def test_dense_mesh_stays_interactive(self):
        ball = s._sphere(1, 256, (1, 1, 1, 1), s.Transform3D())      # 65,536 triangles
        scene = s.Scene((ball,), (s.Light(),))
        self.gpu.render(scene, CAMERA, 960, 600, BACKGROUND)
        start = time.perf_counter()
        for _ in range(10):
            self.gpu.render(scene, CAMERA, 960, 600, BACKGROUND)
        # Measured 1.3 ms on an RTX 3080 Ti; the CPU reference needs seconds for this mesh.
        self.assertLess((time.perf_counter() - start) / 10, 0.05)

    def test_busy_device_returns_no_frame(self):
        with patch.object(viewportgpu, "_LOCK_TIMEOUT", 0.0):
            import threading
            held, release = threading.Event(), threading.Event()

            def hold():
                with gpu3d._lock:
                    held.set()
                    release.wait(5)
            thread = threading.Thread(target=hold)
            thread.start()
            held.wait(5)
            try:
                self.assertIsNone(self.gpu.render(s.Scene(), CAMERA, 32, 32, BACKGROUND))
            finally:
                release.set()
                thread.join()

    def test_widget_paints_with_the_gpu_and_reports_it(self):
        app = QApplication.instance() or QApplication([])
        widget = Viewport3D()
        widget.resize(320, 220)
        widget.set_document(graph().document)
        with patch.object(s, "render", side_effect=AssertionError("CPU renderer used")):
            image = widget.grab().toImage()
        self.assertFalse(image.isNull())
        self.assertEqual(widget.status, "")
        self.assertIsNotNone(widget._last_frame)
        widget.close()


class Fallback(unittest.TestCase):
    def test_widget_uses_the_cpu_renderer_without_an_adapter(self):
        app = QApplication.instance() or QApplication([])
        widget = Viewport3D()
        widget.resize(160, 120)
        widget.set_document(graph().document)
        with patch.object(viewportgpu, "renderer", return_value=None), \
                patch.object(s, "render", wraps=s.render) as render:
            self.assertFalse(widget.grab().toImage().isNull())
        self.assertTrue(render.called)
        widget.close()

    def test_gpu_fault_falls_back_and_says_so(self):
        app = QApplication.instance() or QApplication([])

        class Broken:
            def render(self, *args, **kwargs):
                raise RuntimeError("device lost")
        widget = Viewport3D()
        widget.resize(160, 120)
        widget.set_document(graph().document)
        with patch.object(viewportgpu, "renderer", return_value=Broken()):
            self.assertFalse(widget.grab().toImage().isNull())
        self.assertEqual(widget.backend, "cpu")
        self.assertIn("device lost", widget.status)
        widget.close()


if __name__ == "__main__":
    unittest.main()
