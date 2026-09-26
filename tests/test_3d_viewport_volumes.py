"""Lane L4 plan 3, step A part 2: volumes in the editor viewport.

GPU cases skip without an adapter; the carry-through and CPU-fallback cases never need one.
"""
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from nodebased import gpu3d, gpuvolume, scene3d as s, viewportgpu
from nodebased.core import Dispatcher
from nodebased.viewport3d import Viewport3D

APP = QApplication.instance() or QApplication([])
BACKGROUND = (0.025, 0.025, 0.03, 1.0)
CAMERA = s.Camera(s.Transform3D(s.Vec3(0.3, 0.6, 3.2)), s.Vec3(0, 0.5, 0), 45.0, 0.07, 1000.0)
SUN = s.Light("Directional", position=s.Vec3(4, 5, 3), target=s.Vec3())
W, H = 320, 240


def plume_scene(**kwargs):
    return s.Scene(volumes=(s.analytic_plume(40),), lights=(SUN,), **kwargs)


def frame(renderer, scene, width=W, height=H):
    image = renderer.render(scene, CAMERA, width, height, BACKGROUND, ambient=0.15)
    assert image is not None
    return image.astype(int)


class CarryThrough(unittest.TestCase):
    """Every place viewport3d.py rebuilds a Scene keeps the volumes."""

    def widget(self):
        widget = Viewport3D()
        d = Dispatcher()
        d.execute({"op": "create", "id": "k", "type": "Card3D"})
        widget.document = d.document
        self.addCleanup(widget.close)
        return widget

    def test_gizmo_drag_rebuilds_keep_the_volumes(self):
        widget = self.widget()
        card = s._card(1, 1, (1, 1, 1, 1), s.Transform3D())
        scene = s.Scene((card,), (), (), (), (s.analytic_plume(8),))
        with patch.object(widget, "_pick_candidates", return_value=[("k", card)]), \
                patch.object(widget, "_gizmo_world_shift", return_value=np.array((1.0, 0, 0))):
            widget._gizmo_drag = {"kind": "axis", "key": "k"}
            moved = widget._dragged_scene(scene)
        self.assertIsNot(moved, scene)
        self.assertIs(moved.volumes[0], scene.volumes[0])
        with patch.object(widget, "_pick_candidates", return_value=[("k", card)]), \
                patch.object(widget, "_gizmo_values", return_value={"rx": 20.0}):
            widget._gizmo_drag = {"kind": "ring", "key": "k"}
            turned = widget._dragged_scene(scene)
        self.assertIsNot(turned, scene)
        self.assertIs(turned.volumes[0], scene.volumes[0])

    def test_the_cpu_fallback_says_volumes_need_the_gpu_viewport(self):
        widget = self.widget()
        widget.resize(160, 120)
        widget._scene_cache = ((widget._frame(), None), (plume_scene(), CAMERA))
        widget.look_through = True
        with patch.object(viewportgpu, "renderer", return_value=None):
            self.assertFalse(widget.grab().toImage().isNull())
        self.assertIn("GPU viewport", widget.volume_note)


@unittest.skipUnless(gpu3d.available(), "no wgpu adapter")
class GPUViewport(unittest.TestCase):
    def setUp(self):
        self.renderer = viewportgpu.renderer()
        self.assertIsNotNone(self.renderer, viewportgpu.failure())
        self.renderer.volume_quality = False
        self.addCleanup(setattr, self.renderer, "volume_quality", False)

    def test_pixels_where_the_plume_is_differ_from_the_background(self):
        empty = frame(self.renderer, s.Scene(lights=(SUN,)))
        smoke = frame(self.renderer, plume_scene())
        changed = np.abs(smoke - empty).max(axis=2) > 8
        self.assertGreater(int(changed.sum()), 1500)
        # The plume rises over the middle of the frame and leaves the corners alone.
        ys, xs = np.nonzero(changed)
        self.assertLess(abs(float(xs.mean()) - W / 2), W * 0.2)
        self.assertFalse(changed[:10, :10].any() or changed[-10:, -10:].any())
        self.assertEqual(self.renderer.volume_note, "")
        self.assertGreater(self.renderer.volume_steps, 10)

    def test_a_card_inside_the_plume_is_partially_covered(self):
        card = s._card(1.2, 1.2, (0.8, 0.3, 0.2, 1.0), s.Transform3D(s.Vec3(0, 0.5, 0.0)))
        volume = plume_scene()
        bare = frame(self.renderer, s.Scene((card,), (SUN,)))
        smoke = frame(self.renderer, volume)
        both = frame(self.renderer, s.Scene((card,), (SUN,), volumes=volume.volumes))
        on_card = np.abs(bare - frame(self.renderer, s.Scene(lights=(SUN,)))).max(axis=2) > 20
        self.assertGreater(int(on_card.sum()), 500)
        front = np.abs(both - bare).max(axis=2) > 6           # the smoke in front of the card shows on it
        self.assertGreater(int((front & on_card).sum()), 100)
        # Smoke behind the card is hidden: where the card is, the result is below card plus a full plume.
        full_stack = np.abs(both - bare) < np.abs(smoke - frame(self.renderer, s.Scene(lights=(SUN,)))) + 6
        self.assertGreater(float(full_stack[on_card].mean()), 0.95)

    def test_the_quality_toggle_uses_finer_steps_and_the_key_flips_it(self):
        scene = plume_scene()
        frame(self.renderer, scene)
        fast = self.renderer.volume_steps
        self.renderer.volume_quality = True
        frame(self.renderer, scene)
        self.assertGreater(self.renderer.volume_steps, fast * 2)
        widget = Viewport3D()
        self.addCleanup(widget.close)
        self.renderer.volume_quality = False
        widget.keyPressEvent(QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_V, Qt.KeyboardModifier.NoModifier))
        self.assertTrue(self.renderer.volume_quality)
        self.assertIn("quality", widget._volume_note(scene, self.renderer))

    def test_orbiting_uploads_the_density_once(self):
        scene = plume_scene()
        frame(self.renderer, scene)
        state = gpu3d._state()
        before = gpuvolume.upload_count(state)
        for x in (0.5, 1.0, 1.5):
            camera = s.Camera(s.Transform3D(s.Vec3(x, 0.6, 3.0)), s.Vec3(0, 0.5, 0), 45.0, 0.07, 1000.0)
            self.assertIsNotNone(self.renderer.render(scene, camera, W, H, BACKGROUND, ambient=0.15))
        self.assertEqual(gpuvolume.upload_count(state), before)

    def test_an_oversized_grid_is_hidden_with_a_note_and_the_rest_still_draws(self):
        card = s._card(1.2, 1.2, (0.8, 0.3, 0.2, 1.0), s.Transform3D(s.Vec3(0, 0.5, 0.0)))
        wide = s.Volume(np.ones((1 << 20, 1, 1), np.float32), 1e-4)
        image = frame(self.renderer, s.Scene((card,), (SUN,), volumes=(wide,)))
        self.assertIn("volumes hidden", self.renderer.volume_note)
        np.testing.assert_array_equal(image, frame(self.renderer, s.Scene((card,), (SUN,))))

    def test_the_widget_paints_volumes_with_the_gpu(self):
        widget = Viewport3D()
        widget.resize(W, H)
        self.addCleanup(widget.close)
        widget._scene_cache = ((widget._frame(), None), (plume_scene(), CAMERA))
        widget.look_through = True
        with patch.object(s, "render", side_effect=AssertionError("CPU renderer used")):
            self.assertFalse(widget.grab().toImage().isNull())
        self.assertEqual(widget.status, "")
        self.assertIn("volumes:", widget.volume_note)


if __name__ == "__main__":
    unittest.main()
