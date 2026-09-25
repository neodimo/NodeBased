"""Lane L4 step E, parts 2 and 3: particles and Spot lights in the editor viewport.

GPU cases skip without an adapter; the CPU-fallback and carry-through cases never need one.
"""
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from nodebased import gpu3d, scene3d as s, viewportgpu
from nodebased.core import Dispatcher
from nodebased.viewport3d import Viewport3D

APP = QApplication.instance() or QApplication([])
BLACK = (0, 0, 0, 1)
CAMERA = s.Camera(s.Transform3D(s.Vec3(0, 0, 5)), s.Vec3(), 45.0, 0.1, 100.0)


def particle_set(kind="points", position=(0, 0, 0), size=1.0, colour=(1, 0, 0, 1), n=1, texture=None):
    return s.ParticleInstance(np.tile(np.array(position, 'f4'), (n, 1)), np.full(n, size, 'f4'),
                              np.tile(np.array(colour, 'f4'), (n, 1)), render_as=kind, texture=texture)


def pixel(width, height, point=(0, 0, 0)):
    xy, _z = s.project(CAMERA, width, height, np.asarray(point, np.float32)[None])
    return int(xy[0, 1]), int(xy[0, 0])


def spot_card():
    """A white 6 x 6 card at the origin under a narrow Spot 4 units in front of it."""
    card = s._card(6, 6, (1, 1, 1, 1), s.Transform3D())
    spot = s.Light("Spot", intensity=1.0, position=s.Vec3(0, 0, 4), target=s.Vec3(),
                   cone_angle=30.0, cone_penumbra_angle=10.0)
    return s.Scene((card,), (spot,))


class CarryThrough(unittest.TestCase):
    """The three places viewport3d.py rebuilds a Scene keep the particles."""

    def widget(self):
        widget = Viewport3D()
        d = Dispatcher()
        d.execute({"op": "create", "id": "k", "type": "Card3D"})
        widget.document = d.document
        self.addCleanup(widget.close)
        return widget

    def test_gizmo_drag_rebuilds_keep_the_particles(self):
        widget = self.widget()
        card = s._card(1, 1, (1, 1, 1, 1), s.Transform3D())
        scene = s.Scene((card,), (), (), (particle_set(),))
        with patch.object(widget, "_pick_candidates", return_value=[("k", card)]), \
                patch.object(widget, "_gizmo_world_shift", return_value=np.array((1.0, 0, 0))):
            widget._gizmo_drag = {"kind": "axis", "key": "k"}
            moved = widget._dragged_scene(scene)
        self.assertIsNot(moved, scene)
        self.assertEqual(moved.particles, scene.particles)
        with patch.object(widget, "_pick_candidates", return_value=[("k", card)]), \
                patch.object(widget, "_gizmo_values", return_value={"rx": 20.0}):
            widget._gizmo_drag = {"kind": "ring", "key": "k"}
            turned = widget._dragged_scene(scene)
        self.assertIsNot(turned, scene)
        self.assertEqual(turned.particles, scene.particles)

    def paint(self, widget, scene, renderer=None):
        widget._scene_cache = ((widget._frame(), None), (scene, CAMERA))
        widget.look_through = True
        with patch.object(viewportgpu, "renderer", return_value=renderer):
            return widget.grab().toImage()

    def test_the_cpu_fallback_draws_particles(self):
        widget = self.widget()
        widget.resize(320, 220)
        # Off the grid axes, which the editor draws over the render.
        scene = s.Scene(particles=(particle_set(position=(0.6, 0.6, 0), size=0.6),))
        image = self.paint(widget, scene)
        y, x = pixel(320, 220, (0.6, 0.6, 0))
        colour = image.pixelColor(x, y)
        self.assertGreater(colour.red(), 200)
        self.assertLess(colour.green() + colour.blue(), 60)
        self.assertLess(image.pixelColor(300, 200).red(), 90)
        self.assertLess(image.pixelColor(x, y).green(), image.pixelColor(x, y).red())
        empty = self.paint(widget, s.Scene())
        self.assertLess(empty.pixelColor(x, y).red(), 90)


@unittest.skipUnless(gpu3d.available(), "no wgpu adapter")
class GPUViewportParticles(unittest.TestCase):
    def setUp(self):
        self.gpu = viewportgpu.renderer()
        self.assertIsNotNone(self.gpu, viewportgpu.failure())

    def frame(self, scene, size=(200, 160), **kwargs):
        return self.gpu.render(scene, CAMERA, *size, BLACK, headlight=kwargs.pop("headlight", True), **kwargs)

    def test_particles_draw_non_background_pixels_where_they_are(self):
        for kind in ("points", "spheres", "cards"):
            frame = self.frame(s.Scene(particles=(particle_set(kind, (0.5, 0, 0)),)))
            covered = (frame[..., :3].max(axis=2) > 30)
            self.assertGreater(int(covered.sum()), 300, kind)
            y, x = pixel(200, 160, (0.5, 0, 0))
            self.assertGreater(int(frame[y, x, 0]), 100, kind)
            ys, xs = np.nonzero(covered)
            self.assertLess(abs(ys.mean() - y), 2.0, kind)
            self.assertLess(abs(xs.mean() - x), 2.0, kind)
            self.assertFalse(covered[:8].any() or covered[:, :8].any(), kind)   # far from the particle: background

    def test_particles_match_render3d_and_are_depth_tested_against_meshes(self):
        wall = s._card(2, 2, (0.2, 0.5, 0.9, 1), s.Transform3D())
        behind = particle_set("cards", (0, 0, -1.0), 1.0, (1, 0, 0, 1))
        front = particle_set("cards", (0.5, 0.5, 1.0), 0.2, (0, 1, 0, 1))
        scene = s.Scene((wall,), (), (), (behind, front))
        frame = self.frame(scene, headlight=False, ambient=1.0)
        y, x = pixel(200, 160, (0.5, 0.5, 1.0))
        self.assertGreater(int(frame[y, x, 1]), 200)                       # the near card covers the wall
        y, x = pixel(200, 160, (-0.5, -0.5, 0.0))
        bare = self.frame(s.Scene((wall,)), headlight=False, ambient=1.0)
        np.testing.assert_array_equal(frame[y, x], bare[y, x])             # the far one is hidden by it

    def test_particles_alone_count_thousands_and_a_huge_set_is_strided(self):
        n = 3000
        rng = np.random.default_rng(2)
        cloud = s.ParticleInstance(rng.uniform(-1.5, 1.5, (n, 3)).astype("f4"), np.full(n, 0.1, "f4"),
                                   np.full((n, 4), 1.0, "f4"), render_as="spheres")
        self.assertGreater(int((self.frame(s.Scene(particles=(cloud,)))[..., :3].max(axis=2) > 30).sum()), 500)
        self.assertEqual(self.gpu.particle_stride, 1)
        with patch.object(viewportgpu, "MAX_PARTICLES", 1000):
            self.frame(s.Scene(particles=(cloud,)))
        self.assertEqual(self.gpu.particle_stride, 3)
        self.frame(s.Scene())
        self.assertEqual(self.gpu.particle_stride, 1)

    def test_textured_cards_show_the_sprite(self):
        texture = np.zeros((2, 2, 4), "f4")
        texture[..., 3] = 1
        texture[0, 0, 0] = texture[0, 1, 1] = texture[1, 0, 2] = 1      # top left red, top right green, bottom left blue
        frame = self.frame(s.Scene(particles=(particle_set("cards", size=2.0, colour=(1, 1, 1, 1), texture=texture),)))
        y, x = pixel(200, 160)
        top_left, top_right, bottom_left = frame[y - 20, x - 20], frame[y - 20, x + 20], frame[y + 20, x - 20]
        self.assertGreater(int(top_left[0]), 200)
        self.assertGreater(int(top_right[1]), 200)
        self.assertGreater(int(bottom_left[2]), 200)


@unittest.skipUnless(gpu3d.available(), "no wgpu adapter")
class GPUViewportSpot(unittest.TestCase):
    def setUp(self):
        self.gpu = viewportgpu.renderer()
        self.assertIsNotNone(self.gpu, viewportgpu.failure())

    def test_a_spot_lit_card_is_dark_outside_the_cone_and_lit_inside(self):
        scene = spot_card()
        frame = self.gpu.render(scene, CAMERA, 200, 160, BLACK, headlight=False, ambient=0.0)
        centre, corner = frame[80, 100, :3], frame[6, 6, :3]
        self.assertGreater(int(centre.max()), 200)
        self.assertLess(int(corner.max()), 8)
        # The cone hard-edge sits at the outer half-angle (20 degrees at 4 units: 1.46 units off axis).
        y, x = pixel(200, 160, (1.2, 0, 0))
        self.assertGreater(int(frame[y, x, :3].max()), 30)
        y, x = pixel(200, 160, (2.0, 0, 0))
        self.assertLess(int(frame[y, x, :3].max()), 8)

    def test_the_viewport_agrees_with_the_cpu_reference_on_the_spot(self):
        scene = spot_card()
        frame = self.gpu.render(scene, CAMERA, 200, 160, BLACK, headlight=False, ambient=0.0)
        image = s.render(scene, CAMERA, 200, 160, BLACK, shade=False, ambient=0.0, shadows=False, samples=2)
        encoded = np.clip(image[..., :3], 0, 1)
        encoded = np.where(encoded <= 0.0031308, encoded * 12.92,
                           1.055 * np.power(np.maximum(encoded, 1e-9), 1 / 2.4) - 0.055)
        difference = np.abs(frame[..., :3].astype(int) - (encoded * 255).round().astype(int))
        self.assertLess(float(difference.mean()), 1.5)
        self.assertLess(float((difference.max(axis=2) > 25).mean()), 0.02)

    def test_point_falloff_and_a_directional_light_are_unchanged_by_the_new_layout(self):
        card = s._card(6, 6, (1, 1, 1, 1), s.Transform3D())
        point = s.Light("Point", position=s.Vec3(0, 0, 4), falloff_type="Quadratic", intensity=30.0)
        direct = s.Light("Directional", position=s.Vec3(0, 0, 4), target=s.Vec3())
        for light in (point, direct):
            scene = s.Scene((card,), (light,))
            frame = self.gpu.render(scene, CAMERA, 200, 160, BLACK, headlight=False, ambient=0.0)
            image = s.render(scene, CAMERA, 200, 160, BLACK, shade=False, ambient=0.0, shadows=False, samples=2)
            encoded = np.clip(image[..., :3], 0, 1)
            encoded = np.where(encoded <= 0.0031308, encoded * 12.92,
                               1.055 * np.power(np.maximum(encoded, 1e-9), 1 / 2.4) - 0.055)
            self.assertLess(float(np.abs(frame[..., :3].astype(int) - (encoded * 255).round()).mean()), 1.5, light.kind)


if __name__ == "__main__":
    unittest.main()
