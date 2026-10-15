"""Rendering 13, step G2 part 1: the live 3D viewport looks through the camera's lens.

The checks that mean something are about where things land: a small sphere seen through a barrel lens sits
where `scene3d.project_lens` (the tracker's and Project3D's projection) says, not where the ideal camera puts it; a
camera with no lens draws exactly what it drew before; the strip's switch and the L key flip it. The warp under it
(`viewportlens.redistort`) is held to the Redistort node's own sample coordinates on every adapter this machine can
name; an adapter that cannot run it skips with the reason.
"""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

import tests.isolation  # noqa: F401  (Qt settings of the test run stay out of the user's real ones)
from nodebased import gpu3d, gpulens, lensmodel, scene3d, viewportlens
from nodebased.core import Dispatcher
from nodebased.viewport3d import Viewport3D

APP = QApplication.instance() or QApplication([])

SIZE = (320, 220)      # the widget's own minimum size
BARREL = dict(lens_k1=-0.06, lens_k2=0.004)
BALL = (2.0, -0.83, -4.0)       # in the lower right quarter of the frame, well clear of the header text


def graph(**lens):
    d = Dispatcher()
    d.execute(dict(op="create", id="ball", type="Sphere3D", params=dict(sphere_radius=0.12, tx=BALL[0], ty=BALL[1], tz=BALL[2])))
    d.execute(dict(op="create", id="cam", type="Camera3D", params=dict(tz=0.0, target_z=-1.0, **lens)))
    d.execute(dict(op="create", id="scene", type="Scene3D", params={}))
    d.execute(dict(op="create", id="render", type="Render3D", params={}))
    d.execute(dict(op="connect", id="scene", input="object0", source="ball"))
    d.execute(dict(op="connect", id="render", input="scene", source="scene"))
    d.execute(dict(op="connect", id="render", input="camera", source="cam"))
    d.execute({"op": "view", "id": "render"})
    return d


def viewport(d, backend="auto"):
    widget = Viewport3D()
    widget.backend = backend
    widget.resize(*SIZE)
    widget.set_document(d.document)
    widget.look_through = True
    return widget


def picture(widget):
    image = widget.grab().toImage().convertToFormat(widget.grab().toImage().Format.Format_RGBA8888)
    ptr = image.constBits()
    return np.frombuffer(ptr, np.uint8).reshape(image.height(), image.bytesPerLine() // 4, 4)[:, :image.width(), :3].copy()


def centroid(rgb):
    """Brightness-weighted centre of what is lit in the lower right of the frame: clear of the header text, and
    below the horizon line the viewport's grid draws through the middle and above the footer note."""
    height, width = rgb.shape[:2]
    top, left, bottom = int(height * 0.62), width // 2, height - 32
    luma = rgb.astype(np.float64).sum(axis=2)[top:bottom, left:]
    luma = np.where(luma > 60, luma, 0.0)
    yy, xx = np.mgrid[top:bottom, left:width]
    total = luma.sum()
    return float((luma * xx).sum() / total + 0.5), float((luma * yy).sum() / total + 0.5)


class ProjectionMath(unittest.TestCase):
    def test_a_camera_without_a_lens_needs_no_overscan(self):
        self.assertIsNone(viewportlens.frame_for(scene3d.Camera(), 320, 180))

    def test_a_pincushion_lens_pulls_the_picture_inwards_and_needs_none(self):
        frame = viewportlens.frame_for(scene3d.Camera(lens_k1=0.03), 320, 180)
        self.assertEqual((frame.margin_x, frame.margin_y), (0, 0))

    def test_a_barrel_lens_needs_overscan_and_the_margin_covers_every_read(self):
        camera = scene3d.Camera(**BARREL)
        frame = viewportlens.frame_for(camera, 320, 180)
        self.assertGreater(frame.margin_x, 10)
        self.assertGreater(frame.margin_y, 5)
        sx, sy = lensmodel.sample_coordinates(frame.lens, frame.display, frame.display, frame.source, "redistort")
        self.assertGreaterEqual(float(sx.min()), -1.0)
        self.assertLessEqual(float(sx.max()), frame.render_width)
        self.assertGreaterEqual(float(sy.min()), -1.0)
        self.assertLessEqual(float(sy.max()), frame.render_height)

    def test_the_overscan_camera_keeps_the_pixel_scale_and_the_centre(self):
        """A point projects to the same pixel through the grown camera, offset by the margin."""
        camera = scene3d.Camera(scene3d.Transform3D(scene3d.Vec3(0, 0, 3)), scene3d.Vec3(0, 0, -1), 50.0, **BARREL)
        frame = viewportlens.frame_for(camera, 320, 180)
        points = np.array([(0.3, -0.2, -2.0), (-1.2, 0.5, -5.0), (0.0, 0.0, -1.0)])
        base, _ = scene3d.project(camera, 320, 180, points)
        grown, _ = scene3d.project(frame.camera(camera), frame.render_width, frame.render_height, points)
        np.testing.assert_allclose(grown - np.array((frame.margin_x, frame.margin_y)), base, atol=1e-3)

    def test_the_margin_is_capped_for_a_lens_that_throws_a_corner_far_out(self):
        frame = viewportlens.frame_for(scene3d.Camera(lens_k1=-0.5), 320, 180)
        self.assertLessEqual(frame.margin_x, int(viewportlens.MAX_MARGIN * 320))
        self.assertLessEqual(frame.margin_y, int(viewportlens.MAX_MARGIN * 180))


def smooth(width, height, offset=(0, 0)):
    """An 8-bit picture that is a smooth function of its (ideal, overscan) pixel position."""
    yy, xx = np.mgrid[:height, :width].astype(np.float64)
    xx, yy = xx + offset[0], yy + offset[1]
    return np.stack([127.5 + 100 * np.sin(xx / 61.0), 127.5 + 100 * np.cos(yy / 47.0),
                     127.5 + 80 * np.sin((xx + yy) / 83.0), np.full_like(xx, 255.0)], axis=-1)


class RedistortAgreement:
    """`viewportlens.redistort` on one adapter, against the value the lens model gives analytically."""
    choice = "default"

    def setUp(self):
        try:
            gpu3d._state(self.choice)
        except Exception as exc:
            self.skipTest(f"{self.choice} adapter not usable here: {exc}")

    def test_the_8_bit_warp_reads_the_ideal_picture_where_the_lens_says(self):
        camera = scene3d.Camera(**BARREL, lens_cx=0.01, lens_cy=-0.01)
        frame = viewportlens.frame_for(camera, 320, 180)
        ideal = smooth(frame.render_width, frame.render_height, (-frame.margin_x, -frame.margin_y))
        try:
            out = viewportlens.redistort(frame, np.round(ideal).astype(np.uint8), choice=self.choice, backend="gpu")
        except gpulens.Unsupported as exc:
            self.skipTest(f"{self.choice} adapter cannot run the viewport lens warp: {exc}")
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, (180, 320, 4))
        gx, gy = np.meshgrid(np.arange(320) + 0.5, np.arange(180) + 0.5)
        ux, uy = lensmodel.undistort_pixels(frame.lens, gx, gy, 320, 180)       # ideal pixel in format coordinates
        yy, xx = uy, ux
        want = np.stack([127.5 + 100 * np.sin((xx - 0.5) / 61.0), 127.5 + 100 * np.cos((yy - 0.5) / 47.0),
                         127.5 + 80 * np.sin(((xx - 0.5) + (yy - 0.5)) / 83.0)], axis=-1)
        error = np.abs(out[..., :3].astype(np.float64) - want)
        self.assertLessEqual(float(error.max()), 2.0, f"{self.choice}: the warp is off by {error.max():.2f} levels")
        self.assertTrue((out[..., 3] == 255).all())

    def test_the_card_agrees_with_the_cpu_reference_to_one_level(self):
        camera = scene3d.Camera(**BARREL)
        frame = viewportlens.frame_for(camera, 320, 180)
        picture_in = np.round(smooth(frame.render_width, frame.render_height)).astype(np.uint8)
        try:
            card = viewportlens.redistort(frame, picture_in, choice=self.choice, backend="gpu")
        except gpulens.Unsupported as exc:
            self.skipTest(f"{self.choice} adapter cannot run the viewport lens warp: {exc}")
        reference = viewportlens.redistort_cpu(frame, picture_in)
        self.assertLessEqual(int(np.abs(card.astype(int) - reference.astype(int)).max()), 1)


class DefaultAdapter(RedistortAgreement, unittest.TestCase):
    choice = "default"


class IntegratedAdapter(RedistortAgreement, unittest.TestCase):
    choice = "integrated"


class SoftwareAdapter(RedistortAgreement, unittest.TestCase):
    choice = "cpu"


class ViewportThroughTheLens(unittest.TestCase):
    def lit_centre(self, d, on, backend="auto"):
        widget = viewport(d, backend)
        self.addCleanup(widget.close)
        widget.lens_view = on
        return centroid(picture(widget)), widget

    def expected(self, d, lens_on):
        camera = scene3d.camera_from_node(d.document["nodes"]["cam"])
        project = scene3d.project_lens if lens_on else scene3d.project
        xy, _z = project(camera, *SIZE, np.array([BALL], np.float64))
        return float(xy[0, 0]), float(xy[0, 1])

    def test_the_sphere_lands_where_the_lens_puts_it_not_where_the_ideal_camera_does(self):
        d = graph(**BARREL)
        bent, ideal = self.expected(d, True), self.expected(d, False)
        self.assertGreater(np.hypot(bent[0] - ideal[0], bent[1] - ideal[1]), 5.0, "the test lens moves nothing")
        for backend in ("auto", "cpu"):
            with self.subTest(backend=backend):
                (x, y), widget = self.lit_centre(d, None, backend)
                self.assertTrue(widget._lens_drawn)
                self.assertLess(np.hypot(x - bent[0], y - bent[1]), 1.5)
                (x, y), widget = self.lit_centre(d, False, backend)
                self.assertFalse(widget._lens_drawn)
                self.assertLess(np.hypot(x - ideal[0], y - ideal[1]), 1.5)

    def test_the_progressive_render_mode_draws_through_the_lens_too(self):
        from nodebased import pathtrace
        try:
            pathtrace.render(scene3d.Scene(), scene3d.Camera(scene3d.Transform3D(scene3d.Vec3(0, 0, 5))), 4, 3, (0, 0, 0, 1),
                             output="rgba", settings=pathtrace.PathSettings(samples=1), backend="auto")
        except Exception as error:     # pragma: no cover - a machine with no path tracer backend at all
            self.skipTest(f"no working path tracer backend: {error}")
        d = graph(**BARREL)
        widget = viewport(d)
        self.addCleanup(widget.close)
        widget.render_mode = True
        rgb = picture(widget)
        self.assertTrue(widget._lens_drawn)
        frame = viewportlens.frame_for(scene3d.camera_from_node(d.document["nodes"]["cam"]), *SIZE)
        self.assertEqual(widget._progressive_state.image.shape[:2], (frame.render_height, frame.render_width))
        x, y = centroid(rgb)
        bent, ideal = self.expected(d, True), self.expected(d, False)
        self.assertLess(np.hypot(x - bent[0], y - bent[1]), 4.0)
        self.assertGreater(np.hypot(x - ideal[0], y - ideal[1]), 5.0)

    def test_a_camera_with_no_lens_draws_the_same_picture_with_the_switch_on_or_off(self):
        d = graph()
        widgets = []
        for on in (None, True, False):
            widget = viewport(d)
            self.addCleanup(widget.close)
            widget.lens_view = on
            widgets.append(picture(widget))
            self.assertFalse(widget._lens_drawn)
            self.assertFalse(widget.lens_available())
        self.assertTrue((widgets[0] == widgets[1]).all() and (widgets[0] == widgets[2]).all())

    def test_the_lens_shows_only_while_looking_through_the_camera(self):
        d = graph(**BARREL)
        widget = viewport(d)
        self.addCleanup(widget.close)
        widget.look_through = False
        widget.grab()
        self.assertFalse(widget._lens_drawn)
        self.assertFalse(widget.lens_available())

    def test_the_l_key_and_the_switch_flip_it(self):
        d = graph(**BARREL)
        widget = viewport(d)
        self.addCleanup(widget.close)
        self.assertTrue(widget.lens_view_on())
        widget.keyPressEvent(QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_L, Qt.KeyboardModifier.NoModifier))
        self.assertFalse(widget.lens_view_on())
        self.assertIs(widget.lens_view, False)
        widget.set_lens_view(True)
        self.assertTrue(widget.lens_view_on())
        self.assertTrue(widget.toggle_lens_view())
        self.assertFalse(widget.lens_view_on())

    def test_the_switch_does_nothing_for_a_camera_without_a_lens(self):
        widget = viewport(graph())
        self.addCleanup(widget.close)
        self.assertFalse(widget.toggle_lens_view())
        self.assertIsNone(widget.lens_view)

    def test_the_editor_overlays_follow_the_lens(self):
        """The selection box of the sphere is drawn through the lens: its pixels land where project_lens puts the box."""
        d = graph(**BARREL)
        widget = viewport(d)
        self.addCleanup(widget.close)
        widget.selected_key = "ball"
        widget.grab()
        camera = scene3d.camera_from_node(d.document["nodes"]["cam"])
        xy_bent, _ = widget._project(camera, *SIZE, np.array([BALL], np.float64))
        widget._lens_drawn = False
        xy_ideal, _ = widget._project(camera, *SIZE, np.array([BALL], np.float64))
        self.assertGreater(float(np.abs(xy_bent - xy_ideal).max()), 5.0)
        widget._lens_drawn = True
        want, _ = scene3d.project_lens(camera, *SIZE, np.array([BALL], np.float64))
        np.testing.assert_allclose(xy_bent, want)


class StripSwitch(unittest.TestCase):
    def window(self, **lens):
        from nodebased.app import STYLE, Window
        APP.setStyle("Fusion")
        APP.setStyleSheet(STYLE)
        window = Window(graph(**lens).document)
        window.resize(1500, 900)
        window.show()
        self.addCleanup(window.close)
        return window

    def test_the_strip_offers_the_switch_in_3d_for_a_camera_with_a_lens(self):
        window = self.window(**BARREL)
        strip = window.viewer_strip
        window.set_viewer_mode("2d")
        APP.processEvents()
        self.assertFalse(strip.lens3d_present)
        window.set_viewer_mode("3d")
        window.viewport.look_through = True
        strip.sync_lens_3d()
        APP.processEvents()
        self.assertTrue(strip.lens3d_present)
        self.assertTrue(strip.lens3d_button.isChecked())
        strip.lens3d_button.click()
        APP.processEvents()
        self.assertIs(window.viewport.lens_view, False)
        self.assertFalse(strip.lens3d_button.isChecked())

    def test_the_strip_has_no_switch_for_a_camera_without_a_lens(self):
        window = self.window()
        window.set_viewer_mode("3d")
        window.viewport.look_through = True
        window.viewer_strip.sync_lens_3d()
        self.assertFalse(window.viewer_strip.lens3d_present)

    def test_the_more_menu_carries_the_switch(self):
        window = self.window(**BARREL)
        window.set_viewer_mode("3d")
        window.viewport.look_through = True
        strip = window.viewer_strip
        strip.sync_lens_3d()
        strip.fill_more_menu()
        action = next(a for a in strip.more_menu.actions() if a.text() == "Lens distortion")
        self.assertTrue(action.isChecked())
        action.setChecked(False)
        self.assertIs(window.viewport.lens_view, False)


if __name__ == "__main__":
    unittest.main()
