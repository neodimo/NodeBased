"""Instance3D copies in the 3D viewport (step X2 of 2, finish): GPU-instanced drawing in the
viewport's raster modes, per-instance tint, picking, the CPU fallback, and the 100,000-copy
frame-rate claim. See docs/3D_FOUNDATION.md "Lane 4 step notes" and docs/3D_ROADMAP.md
"Instancing" for what the rest of Instance3D already does (`scene3d.instances_from_node`,
`expand_instances`, the GPU ray tracer in `gpuinstance.py`).

GPU cases skip without an adapter; the fallback and picking cases never need one.
"""
import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from nodebased import gpu3d, handles3d, scene3d as s, viewportgpu
from nodebased.app import STYLE, Window
from nodebased.core import Dispatcher
from nodebased.viewport3d import Viewport3D

BACKGROUND = (0.025, 0.025, 0.03, 1.0)
CAMERA = s.Camera(s.Transform3D(s.Vec3(4, 3, 6)), s.Vec3(), 45.0, 0.07, 1000.0)

APP = QApplication.instance() or QApplication([])


def srgb(linear):
    linear = np.clip(linear, 0, 1)
    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.power(np.maximum(linear, 1e-9), 1 / 2.4) - 0.055)


def reference(scene, width, height):
    image = s.render(s.resolve_instances(scene), CAMERA, width, height, BACKGROUND, ambient=0.15,
                     shadows=False, samples=2)
    return (srgb(image[..., :3]) * 255).round().astype(int)


def spread_instances(mesh, n, spread=2.0, colors=None, seed=0, node_key="inst"):
    rng = np.random.RandomState(seed)
    matrices = np.tile(np.eye(4), (n, 1, 1))
    matrices[:, :3, 3] = rng.uniform(-spread, spread, (n, 3))
    variant = np.zeros(n, np.int32)
    return s.InstanceSet((mesh,), matrices, variant, colors=colors, node_key=node_key)


@unittest.skipUnless(gpu3d.available(), "no wgpu adapter")
class GPUInstances(unittest.TestCase):
    def setUp(self):
        self.gpu = viewportgpu.renderer()
        self.assertIsNotNone(self.gpu, viewportgpu.failure())

    def test_gpu_instanced_draw_matches_the_cpu_reference(self):
        sphere = s._sphere(1.2, 32, (0.8, 0.2, 0.2, 1), s.Transform3D())
        matrices = np.tile(np.eye(4), (3, 1, 1))
        matrices[0, :3, 3], matrices[1, :3, 3], matrices[2, :3, 3] = (-1.5, 0, 0), (1.5, 0, 0), (0, 1.5, 0)
        variant = np.zeros(3, np.int32)
        iset = s.InstanceSet((sphere,), matrices, variant, node_key="inst")
        scene = s.Scene(instances=(iset,), lights=(s.Light(intensity=0.8),))
        frame = self.gpu.render(scene, CAMERA, 320, 200, BACKGROUND)
        difference = np.abs(frame[..., :3].astype(int) - reference(scene, 320, 200))
        # Instanced copies are shaded by the same fragment logic as `_opaque`/`_blended`'s
        # `mesh_fragment` (measured bit-for-bit equal to the non-instanced mesh path against this
        # same reference); the difference here is antialiasing at silhouettes, same as
        # test_3d_viewport_gpu.GPUViewport.compare.
        self.assertLess(difference.mean(), 5.0)
        self.assertLess((difference.max(axis=2) > 20).mean(), 0.15)

    def test_per_instance_tint_changes_the_pixel(self):
        sphere = s._sphere(1.0, 24, (1, 1, 1, 1), s.Transform3D())
        matrices = np.tile(np.eye(4), (2, 1, 1))
        matrices[0, :3, 3], matrices[1, :3, 3] = (-1.5, 0, 0), (1.5, 0, 0)
        variant = np.zeros(2, np.int32)
        colors = np.array([(1, 0, 0, 1), (0, 0, 1, 1)], np.float32)
        iset = s.InstanceSet((sphere,), matrices, variant, colors=colors, node_key="inst")
        scene = s.Scene(instances=(iset,), lights=(s.Light(intensity=0.9),))
        frame = self.gpu.render(scene, CAMERA, 320, 200, BACKGROUND)
        left = frame[:, :120, :3].astype(int)
        right = frame[:, 200:, :3].astype(int)
        # The red-tinted copy (left of the untinted centre) reads red-dominant; the blue-tinted
        # copy (right) reads blue-dominant, so the tint (not just position) reached the pixel.
        self.assertGreater(int(left[..., 0].max()), int(left[..., 2].max()) + 20)
        self.assertGreater(int(right[..., 2].max()), int(right[..., 0].max()) + 20)

    def test_mixed_geometry_and_instances_agree_with_the_cpu_reference(self):
        floor = s._card(4, 4, (1, 1, 1, 1), s.Transform3D(s.Vec3(0, -1, 0), s.Vec3(-90, 0, 0)))
        sphere = s._sphere(0.6, 24, (0.3, 0.7, 0.9, 1), s.Transform3D())
        matrices = np.tile(np.eye(4), (2, 1, 1))
        matrices[0, :3, 3], matrices[1, :3, 3] = (-1, 0, 0), (1, 0, 0)
        variant = np.zeros(2, np.int32)
        iset = s.InstanceSet((sphere,), matrices, variant, node_key="inst")
        scene = s.Scene((floor,), (s.Light(intensity=0.8),), instances=(iset,))
        frame = self.gpu.render(scene, CAMERA, 320, 200, BACKGROUND)
        difference = np.abs(frame[..., :3].astype(int) - reference(scene, 320, 200))
        self.assertLess(difference.mean(), 3.5)

    def test_unchanged_instances_upload_nothing_on_the_next_frame(self):
        mesh = s._sphere(1, 16, (1, 1, 1, 1), s.Transform3D())
        iset = spread_instances(mesh, 200, node_key="inst")
        scene = s.Scene(instances=(iset,), lights=(s.Light(),))
        self.gpu.render(scene, CAMERA, 96, 64, BACKGROUND)
        uploads = self.gpu.uploads
        self.gpu.render(scene, CAMERA, 96, 64, BACKGROUND)   # same scene object: matrices/variant/sources keep their ids
        self.assertEqual(self.gpu.uploads, uploads)

    def test_unused_instance_sets_are_evicted(self):
        mesh = s._sphere(1, 16, (1, 1, 1, 1), s.Transform3D())
        iset = spread_instances(mesh, 50, node_key="inst")
        scene = s.Scene(instances=(iset,), lights=(s.Light(),))
        self.gpu.render(scene, CAMERA, 64, 64, BACKGROUND)
        self.assertGreater(len(self.gpu._instance_buffers), 0)
        self.gpu.render(s.Scene(), CAMERA, 64, 64, BACKGROUND)
        self.assertEqual((len(self.gpu._instance_buffers), len(self.gpu._instance_meshes)), (0, 0))

    def test_100k_copies_of_a_1000_triangle_mesh_stay_above_30fps(self):
        # ~1,000 triangles (990, measured): docs/3D_ROADMAP.md "100k instances of a 1k-triangle
        # mesh" is the standing memory claim; this is the standing frame-rate claim for the same
        # shape, step X2 of 2 part 3.
        mesh = s._sphere(1, 55, (0.8, 0.2, 0.2, 1), s.Transform3D())
        self.assertEqual(len(mesh.triangles) // 3, 990)
        iset = spread_instances(mesh, 100_000, spread=200.0, node_key="inst")
        scene = s.Scene(instances=(iset,), lights=(s.Light(),))
        self.gpu.render(scene, CAMERA, 960, 600, BACKGROUND)
        # Best of three 10-frame batches: the integration suite on 9/30 4:55 AM measured 37.7 ms in one
        # batch under load from the other test processes, while the same scene alone measured 24 ms in
        # every batch. A real regression slows all three batches; a load spike slows one.
        batches = []
        for _ in range(3):
            start = time.perf_counter()
            for _ in range(10):
                self.gpu.render(scene, CAMERA, 960, 600, BACKGROUND)
            batches.append((time.perf_counter() - start) / 10)
        elapsed = min(batches)
        if "cpu" in str(gpu3d._state()["info"].get("adapter_type", "")).lower():
            self.skipTest("software adapter: the frame-rate claim is about real GPUs")
        # Measured ~25 ms/frame (~40 fps) on an RTX 3080 Ti with the mesh deduplicated into a
        # real vertex/index buffer for the instanced draw (`_instance_mesh`): the same scene drawn
        # through the one-draw-per-object triangle-soup path (`_mesh`, what `_opaque` uses) instead
        # measured ~55 ms/frame (~18 fps), short of the claim.
        self.assertLess(elapsed, 1.0 / 30)


class CPUFallback(unittest.TestCase):
    def test_instances_are_drawn_without_a_gpu_adapter(self):
        sphere = s._sphere(1, 12, (0.9, 0.3, 0.3, 1), s.Transform3D())
        matrices = np.tile(np.eye(4), (1, 1, 1))
        variant = np.zeros(1, np.int32)
        iset = s.InstanceSet((sphere,), matrices, variant, node_key="inst")
        scene = s.Scene(instances=(iset,), lights=(s.Light(intensity=0.9),))
        resolved = s.resolve_instances(scene)
        self.assertEqual(len(resolved.geometries), 1)
        self.assertEqual(resolved.instances, ())
        cpu_image, _depth = s.render(s.Scene(resolved.geometries, resolved.lights), CAMERA, 64, 64,
                                     BACKGROUND, ambient=0.15, return_depth=True, shadows=False, mode="raster")
        empty_image, _depth = s.render(s.Scene(lights=resolved.lights), CAMERA, 64, 64,
                                       BACKGROUND, ambient=0.15, return_depth=True, shadows=False, mode="raster")
        self.assertFalse(np.array_equal(cpu_image, empty_image))


class InstancePicking(unittest.TestCase):
    def test_pick_instance_sets_returns_the_instance3d_node_key(self):
        sphere = s._sphere(1.0, 12, (1, 1, 1, 1), s.Transform3D())
        matrices = np.tile(np.eye(4), (1, 1, 1))
        variant = np.zeros(1, np.int32)
        iset = s.InstanceSet((sphere,), matrices, variant, node_key="inst_node")
        camera = s.Camera(s.Transform3D(s.Vec3(0, 0, 10)), s.Vec3(), 45.0, 0.1, 100.0)
        hit = handles3d.pick_instance_sets((iset,), camera, 200, 200, 100, 100)
        self.assertIsNotNone(hit)
        self.assertEqual(hit[0], "inst_node")

    def test_pick_misses_when_the_ray_clears_every_instance(self):
        sphere = s._sphere(0.3, 12, (1, 1, 1, 1), s.Transform3D())
        matrices = np.tile(np.eye(4), (1, 1, 1))
        matrices[0, :3, 3] = (5, 5, 0)
        variant = np.zeros(1, np.int32)
        iset = s.InstanceSet((sphere,), matrices, variant, node_key="inst_node")
        camera = s.Camera(s.Transform3D(s.Vec3(0, 0, 10)), s.Vec3(), 45.0, 0.1, 100.0)
        self.assertIsNone(handles3d.pick_instance_sets((iset,), camera, 200, 200, 100, 100))

    def test_click_on_an_instance_copy_selects_the_instance3d_node(self):
        d = Dispatcher()
        for key, kind in (("card", "Card3D"), ("ball", "Sphere3D"), ("inst", "Instance3D"),
                          ("cam", "Camera3D"), ("render", "Render3D")):
            d.execute({"op": "create", "id": key, "type": kind})
        # A tiny card puts every one of its 4 corner instances within a few pixels of the origin,
        # so a click at screen centre reliably lands on one of them; the instance mesh itself
        # (radius 1.5) then covers plenty of screen so the click does not need to be exact.
        d.execute({"op": "set", "id": "card", "param": "card_width", "value": 0.05})
        d.execute({"op": "set", "id": "card", "param": "card_height", "value": 0.05})
        d.execute({"op": "set", "id": "ball", "param": "sphere_radius", "value": 1.5})
        d.execute({"op": "connect", "id": "inst", "input": "points", "source": "card"})
        d.execute({"op": "connect", "id": "inst", "input": "instance", "source": "ball"})
        d.execute({"op": "connect", "id": "render", "input": "scene", "source": "inst"})
        d.execute({"op": "connect", "id": "render", "input": "camera", "source": "cam"})
        d.execute({"op": "set", "id": "cam", "param": "tz", "value": 8.0})

        window = Window(d.document)
        try:
            window.show()
            window.viewport_dock.show()
            window.resize(1000, 800)
            viewport = window.viewport
            viewport.resize(640, 360)
            viewport.azimuth, viewport.elevation, viewport.distance = 0.0, 0.0, 8.0
            APP.processEvents()
            center = QPointF(viewport.width() / 2, viewport.height() / 2).toPoint()
            QTest.mouseClick(viewport, Qt.MouseButton.LeftButton, pos=center)
            self.assertEqual(viewport.selected_key, "inst")
            self.assertEqual(window.graph.selected_id(), "inst")
        finally:
            window.saved_document = window.dispatcher.document
            window.close()
            APP.processEvents()
