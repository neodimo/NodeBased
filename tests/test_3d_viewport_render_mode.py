"""R6 (docs/SPLAT_RELIGHTING.md, "Production look"): the 3D viewport's `P` progressive "Render"
mode -- the widget-level wiring around `progressiverender` (reset on camera movement, the sample
count shown, drawing the traced image). GPU-accelerated; not gated on `gpu3d.available()` the way
the rasterised viewport is, since the path tracer's own backend selection handles that.
"""
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from nodebased.core import Dispatcher
from nodebased.viewport3d import Viewport3D
from nodebased import pathtrace, scene3d as s

APP = QApplication.instance() or QApplication([])


def _skip_reason():
    try:
        pathtrace.render(s.Scene(), s.Camera(s.Transform3D(s.Vec3(0, 0, 5))), 4, 3, (0, 0, 0, 1),
                         output="rgba", settings=pathtrace.PathSettings(samples=1), backend="auto")
    except Exception as error:  # pragma: no cover - environment without any path tracer backend
        return str(error)
    return None


_SKIP = _skip_reason()


def graph():
    d = Dispatcher()
    for key, kind in (("ball", "Sphere3D"), ("light", "Light3D"), ("cam", "Camera3D"),
                      ("scene", "Scene3D"), ("render", "Render3D")):
        d.execute({"op": "create", "id": key, "type": kind})
    d.execute({"op": "connect", "id": "scene", "input": "object0", "source": "ball"})
    d.execute({"op": "connect", "id": "scene", "input": "object1", "source": "light"})
    d.execute({"op": "connect", "id": "render", "input": "scene", "source": "scene"})
    d.execute({"op": "connect", "id": "render", "input": "camera", "source": "cam"})
    d.execute({"op": "view", "id": "render"})
    return d


@unittest.skipIf(_SKIP, f"no working path tracer backend: {_SKIP}")
class RenderMode(unittest.TestCase):
    def _widget(self):
        widget = Viewport3D()
        self.addCleanup(widget.close)
        widget.resize(160, 120)
        widget.set_document(graph().document)
        return widget

    def test_the_p_key_turns_it_on_and_off(self):
        widget = self._widget()
        self.assertFalse(widget.render_mode)
        widget.keyPressEvent(QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_P, Qt.KeyboardModifier.NoModifier))
        self.assertTrue(widget.render_mode)
        widget.keyPressEvent(QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_P, Qt.KeyboardModifier.NoModifier))
        self.assertFalse(widget.render_mode)
        self.assertEqual(widget.render_note, "")

    def test_a_paint_shows_the_progressive_image_and_sample_count(self):
        widget = self._widget()
        widget.render_mode = True
        image = widget.grab().toImage()
        self.assertFalse(image.isNull())
        self.assertIsNotNone(widget._progressive_state)
        self.assertIn("RENDER", widget.render_note)
        self.assertIn("sample", widget.render_note)

    def test_the_note_shows_the_pass_and_then_the_converged_share(self):
        widget = self._widget()
        widget.render_mode = True
        widget.grab()
        self.assertIn("pass 1", widget.render_note)
        self.assertNotIn("converged", widget.render_note.replace("converging", ""))     # the low-res reset step measures nothing
        for _ in range(4):
            widget.grab()
        self.assertIn("pass 5", widget.render_note)
        self.assertRegex(widget.render_note, r"\d+% converged")

    def test_repeated_paints_advance_the_sample_count_while_the_camera_is_still(self):
        widget = self._widget()
        widget.render_mode = True
        widget.grab()
        first = widget._progressive_state.samples
        for _ in range(4):
            widget.grab()
        self.assertGreater(widget._progressive_state.samples, first)

    def test_orbiting_resets_the_sample_count(self):
        widget = self._widget()
        widget.render_mode = True
        for _ in range(3):
            widget.grab()
        self.assertGreater(widget._progressive_state.samples, 1)
        widget.azimuth += 30.0   # the same effect orbiting has, without driving real mouse events
        widget.grab()
        self.assertEqual(widget._progressive_state.samples, 1)
        self.assertTrue(widget._progressive_state.low_res)

    def test_turning_it_off_falls_back_to_the_interactive_viewport(self):
        widget = self._widget()
        widget.render_mode = True
        widget.grab()
        widget.keyPressEvent(QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_P, Qt.KeyboardModifier.NoModifier))
        image = widget.grab().toImage()
        self.assertFalse(image.isNull())
        self.assertEqual(widget.render_note, "")

    def test_depth_of_field_shows_when_looking_through_a_lens_camera(self):
        """Step X2: the viewed camera's own fstop/focus (Camera3D, look-through) reaches the
        progressive trace with no extra wiring, since `pathtrace.camera_rays` already samples the
        lens on every ray `pathtrace.render` casts; a pinhole (fstop 0) stays sharp for comparison."""
        d = graph()
        d.execute({"op": "set", "id": "cam", "param": "fstop", "value": 2.8})
        d.execute({"op": "set", "id": "cam", "param": "focus_distance", "value": 3.0})
        widget = Viewport3D()
        self.addCleanup(widget.close)
        widget.resize(64, 48)
        widget.set_document(d.document)
        widget.look_through = True
        widget.render_mode = True
        for _ in range(6):
            widget.grab()
        blurred = widget._progressive_state.image.copy()

        pinhole_doc = graph()
        widget2 = Viewport3D()
        self.addCleanup(widget2.close)
        widget2.resize(64, 48)
        widget2.set_document(pinhole_doc.document)
        widget2.look_through = True
        widget2.render_mode = True
        for _ in range(6):
            widget2.grab()
        sharp = widget2._progressive_state.image.copy()

        self.assertFalse(np.allclose(blurred, sharp, atol=0.03))

    def test_motion_blur_knob_reaches_the_progressive_render(self):
        """Step X2: Render3D's own `motion_blur`/`shutter` knobs, read off the upstream Render3D node
        by `_motion_moments`, blur the progressive image the same way a moving camera does in
        `progressiverender`'s own tests."""
        still_doc = graph()
        d = graph()
        d.execute({"op": "set", "id": "render", "param": "motion_blur", "value": 1})
        d.execute({"op": "set", "id": "render", "param": "shutter", "value": 1.0})
        d.execute({"op": "set", "id": "render", "param": "motion_samples", "value": 3})
        # animate the camera across the shutter so there is something to blur
        d.execute({"op": "set_key", "id": "cam", "param": "tx", "frame": 1, "value": -2.0})
        d.execute({"op": "set_key", "id": "cam", "param": "tx", "frame": 2, "value": 2.0})

        widget = Viewport3D()
        self.addCleanup(widget.close)
        widget.resize(64, 48)
        widget.set_document(d.document)
        widget.look_through = True
        widget.render_mode = True
        for _ in range(6):
            widget.grab()
        blurred = widget._progressive_state.image.copy()

        widget2 = Viewport3D()
        self.addCleanup(widget2.close)
        widget2.resize(64, 48)
        widget2.set_document(still_doc.document)
        widget2.look_through = True
        widget2.render_mode = True
        for _ in range(6):
            widget2.grab()
        still = widget2._progressive_state.image.copy()

        self.assertFalse(np.allclose(blurred, still, atol=0.03))

    def test_a_scene_the_path_tracer_refuses_falls_back_without_crashing(self):
        widget = self._widget()
        widget.render_mode = True
        with patch("nodebased.progressiverender.pathtrace.render", side_effect=ValueError("no particles")):
            image = widget.grab().toImage()
        self.assertFalse(image.isNull())
        self.assertFalse(widget.render_mode)
        self.assertIn("Render mode failed", widget.status)
