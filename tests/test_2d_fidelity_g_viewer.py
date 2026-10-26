"""Fidelity step G, part 4: the Viewer takes Nuke's knobs (frame range and lock, frame increment, rgb only, ignore pixel aspect,
zoom lock, show overscan and overscan, full frame processing, safe zone, format centre, centre f-stop, shutter angle, use GPU for
viewer). Each knob is driven through the real Viewer window (offscreen Qt) and checked on the played frames or the displayed
pixels; the Direct picture honours it or hands the picture back to the Viewer."""
import copy
import json
import os
import time
import unittest
import unittest.mock

import numpy as np

import tests.isolation  # noqa: F401  (Qt settings stay out of the real user file)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QPointF, QRectF, QSettings, Qt
from PySide6.QtGui import QAction, QColor, QImage, QPainter
from PySide6.QtWidgets import QCheckBox, QComboBox, QLineEdit, QSpinBox

from nodebased import carddisplay, viewerknobs
from nodebased.color import display_rgb
from nodebased.core import (SCHEMA_VERSION, Dispatcher, empty_document, upgrade_document, validate, viewer_knobs)
from nodebased.imaging import to_qimage
from tests.test_directviewer_overlays import OverlayParityBase, as_array
from tests.test_viewer_compare import APP, ViewerCompareBase
from tests.waiting import wait_until


def reload(document):
    loaded = upgrade_document(json.loads(json.dumps(document)))
    validate(loaded)
    return loaded


def timed_document(first=1, last=20, current=1, fps=24):
    dispatcher = Dispatcher(empty_document())
    dispatcher.execute({"op": "create", "id": "c", "type": "Constant", "pos": [0, 0],
                        "params": {"width": 160, "height": 80, "red": 0.5, "green": 0.25, "blue": 0.125, "alpha": 0.25}})
    dispatcher.execute({"op": "view", "id": "c"})
    dispatcher.execute({"op": "time", "first": first, "last": last, "current": current, "fps": fps})
    return dispatcher.document


def anamorphic_document(aspect=2.0):
    dispatcher = Dispatcher(empty_document())
    dispatcher.execute({"op": "create", "id": "c", "type": "Constant", "pos": [0, 0],
                        "params": {"width": 160, "height": 80, "red": 0.5, "green": 0.25, "blue": 0.125, "alpha": 1.0}})
    dispatcher.execute({"op": "create", "id": "r", "type": "Reformat", "pos": [200, 0],
                        "params": {"reformat_type": "to_box", "width": 160, "height": 80, "pixel_aspect": aspect}})
    dispatcher.execute({"op": "connect", "id": "r", "input": "image", "source": "c"})
    dispatcher.execute({"op": "view", "id": "r"})
    return dispatcher.document


class KnobDocumentTests(unittest.TestCase):
    def test_defaults_are_not_stored_and_an_old_document_reads_them(self):
        d = Dispatcher(empty_document())
        self.assertNotIn("knobs", d.document["settings"]["viewer"])
        self.assertEqual(viewer_knobs(d.document), viewerknobs.DEFAULT)
        d.execute({"op": "viewer_knobs", "frame_increment": 5})
        self.assertEqual(d.document["settings"]["viewer"]["knobs"]["frame_increment"], 5)
        d.execute({"op": "viewer_knobs", "frame_increment": 10})
        self.assertNotIn("knobs", d.document["settings"]["viewer"], "back at the default, the key goes")

    def test_a_document_with_knobs_survives_a_save_and_load(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "viewer_knobs", "frame_range": "3-7", "safe_zone": "action safe", "overscan": 40,
                   "center_fstop": 5.6, "rgb_only": False})
        loaded = reload(d.document)
        self.assertEqual(viewer_knobs(loaded)["frame_range"], "3-7")
        self.assertEqual(viewer_knobs(loaded)["safe_zone"], "action safe")
        self.assertFalse(viewer_knobs(loaded)["rgb_only"])
        self.assertEqual(viewer_knobs(loaded)["center_fstop"], 5.6)

    def test_bad_values_are_refused_and_change_nothing(self):
        d = Dispatcher(empty_document())
        for bad in ({"frame_range": "ten"}, {"frame_range": "1-2-3"}, {"frame_increment": 0}, {"frame_increment": 2.5},
                    {"safe_zone": "wide"}, {"rgb_only": 1}, {"overscan": -1}, {"center_fstop": 0},
                    {"shutter_angle": 1000}, {"nonsense": 1}):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    d.execute({"op": "viewer_knobs", **bad})
        self.assertNotIn("knobs", d.document["settings"]["viewer"])

    def test_one_undo_step_per_edit(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "viewer_knobs", "zoom_lock": True})
        d.execute({"op": "undo"})
        self.assertFalse(viewer_knobs(d.document)["zoom_lock"])


class RangeTests(unittest.TestCase):
    def test_parse_range(self):
        self.assertEqual(viewerknobs.parse_range("10-20"), (10, 20))
        self.assertEqual(viewerknobs.parse_range("10 20"), (10, 20))
        self.assertEqual(viewerknobs.parse_range("20,10"), (10, 20))
        self.assertEqual(viewerknobs.parse_range("12"), (12, 12))
        self.assertEqual(viewerknobs.parse_range("-5-3"), (-5, 3))
        for text in ("", "a-b", "1-2-3", None):
            self.assertIsNone(viewerknobs.parse_range(text))

    def test_the_range_is_cut_down_to_the_project_and_the_lock_puts_it_in_force(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "time", "first": 5, "last": 30, "current": 5})
        self.assertEqual(viewerknobs.playback_range(d.document), (5, 30))
        d.execute({"op": "viewer_knobs", "frame_range": "10-20"})
        self.assertEqual(viewerknobs.playback_range(d.document), (10, 20))
        d.execute({"op": "viewer_knobs", "frame_range": "1-12"})
        self.assertEqual(viewerknobs.playback_range(d.document), (5, 12))
        d.execute({"op": "viewer_knobs", "frame_range": "100-200"})
        self.assertEqual(viewerknobs.playback_range(d.document), (5, 30), "a range wholly outside the project is not in force")
        d.execute({"op": "viewer_knobs", "frame_range": "10-20", "frame_range_lock": False})
        self.assertEqual(viewerknobs.playback_range(d.document), (5, 30), "unlocked, the viewer follows the project")

    def test_the_increment_steps_stop_at_the_edge_of_the_range(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "time", "first": 1, "last": 100, "current": 50})
        d.execute({"op": "viewer_knobs", "frame_increment": 7, "frame_range": "40-60"})
        self.assertEqual(viewerknobs.step_target(d.document, 1), 57)
        self.assertEqual(viewerknobs.step_target(d.document, -1), 43)
        d.execute({"op": "time", "current": 58})
        self.assertEqual(viewerknobs.step_target(d.document, 1), 60)
        d.execute({"op": "time", "current": 41})
        self.assertEqual(viewerknobs.step_target(d.document, -1), 40)


class GeometryTests(unittest.TestCase):
    def test_safe_rectangles_are_centred_fractions_of_the_format(self):
        self.assertEqual(viewerknobs.safe_rects(200, 100, "no guides"), [])
        self.assertEqual(viewerknobs.safe_rects(200, 100, "action safe"), [(10.0, 5.0, 190.0, 95.0)])
        self.assertEqual(viewerknobs.safe_rects(200, 100, "title safe"), [(20.0, 10.0, 180.0, 90.0)])
        self.assertEqual(len(viewerknobs.safe_rects(200, 100, "title and action safe")), 2)
        self.assertEqual(viewerknobs.overscan_rect(200, 100, 30), (-30, -30, 230, 130))

    def test_the_gain_reads_as_an_f_number_and_the_shutter_as_a_time(self):
        knobs = dict(viewerknobs.DEFAULT)
        self.assertEqual(viewerknobs.exposure_label(0.0, knobs, 24), "f/8.0 · 180° · 1/48 s")
        self.assertEqual(viewerknobs.exposure_label(2.0, knobs, 24), "f/4.0 · 180° · 1/48 s")     # two stops of gain, two stops of aperture
        self.assertIn("f/5.6", viewerknobs.exposure_label(0.0, dict(knobs, center_fstop=5.6), 24))
        self.assertIn("90° · 1/96 s", viewerknobs.exposure_label(0.0, dict(knobs, shutter_angle=90.0), 24))

    def test_the_pixel_aspect_is_the_nearest_reformats(self):
        document = anamorphic_document(2.0)
        self.assertEqual(viewerknobs.pixel_aspect(document, "r"), 2.0)
        self.assertEqual(viewerknobs.pixel_aspect(document, "c"), 1.0)
        self.assertEqual(viewerknobs.stretch(document), 1.0, "ignored by default: NodeBased has always drawn square pixels")
        d = Dispatcher(document)
        d.execute({"op": "viewer_knobs", "ignore_pixel_aspect": False})
        self.assertEqual(viewerknobs.stretch(d.document), 2.0)
        d.execute({"op": "set", "id": "r", "param": "pixel_aspect", "value": 1.0})
        self.assertEqual(viewerknobs.stretch(d.document), 1.0)


class RgbOnlyPictureTests(unittest.TestCase):
    """to_qimage: with the Viewer's `apply LUT to colour channels only` off, gain, gamma and the view act on alpha too."""

    def frame(self, alpha=0.25):
        pixels = np.zeros((8, 8, 4), np.float32)
        pixels[..., :3] = 0.1
        pixels[..., 3] = alpha
        return pixels

    def grey(self, image):
        return image.pixelColor(3, 3).red()

    def test_alpha_is_shown_as_it_is_by_default(self):
        plain = self.grey(to_qimage(self.frame(), 1.0, "A", view="Linear", look={"gamma": 1.0, "zebra": False}))
        self.assertEqual(plain, round(0.25 * 255), "the old behaviour: gain does not touch the alpha")
        also = to_qimage(self.frame(), 1.0, "A", view="Linear", look={"gamma": 2.0, "zebra": False, "rgb_only": True})
        self.assertEqual(self.grey(also), plain)

    def test_off_applies_gain_and_gamma_to_alpha(self):
        look = {"gamma": 1.0, "zebra": False, "rgb_only": False}
        self.assertAlmostEqual(self.grey(to_qimage(self.frame(), 1.0, "A", view="Linear", look=look)), 0.5 * 255, delta=1)
        look = {"gamma": 2.0, "zebra": False, "rgb_only": False}
        self.assertAlmostEqual(self.grey(to_qimage(self.frame(), 0.0, "A", view="Linear", look=look)), (0.25 ** 0.5) * 255, delta=1)

    def test_off_runs_alpha_through_the_view_transform(self):
        look = {"gamma": 1.0, "zebra": False, "rgb_only": False}
        expected = float(np.clip(display_rgb(np.full((1, 1, 3), 0.25, np.float32), "sRGB"), 0, 1)[0, 0, 0]) * 255
        self.assertAlmostEqual(self.grey(to_qimage(self.frame(), 0.0, "A", view="sRGB", look=look)), expected, delta=1)

    def test_the_colour_channels_are_unchanged_by_the_knob(self):
        on = to_qimage(self.frame(1.0), 1.0, "RGB", view="sRGB", look={"gamma": 1.0, "zebra": False})
        off = to_qimage(self.frame(1.0), 1.0, "RGB", view="sRGB", look={"gamma": 1.0, "zebra": False, "rgb_only": False})
        self.assertEqual(on, off)

    def test_use_gpu_for_viewer_off_gives_the_same_picture_without_the_card(self):
        frame = self.frame(1.0)
        with unittest.mock.patch("nodebased.imaging.gpupicture.take", return_value=None) as take, \
                unittest.mock.patch("nodebased.imaging.CARD_PICTURE", True):
            cpu = to_qimage(frame, 0.0, "RGB", view="sRGB", look={"gamma": 1.0, "zebra": False, "gpu": False})
            self.assertEqual(take.call_count, 0, "the card is not asked")
            on_card = to_qimage(frame, 0.0, "RGB", view="sRGB", look={"gamma": 1.0, "zebra": False})
            self.assertEqual(take.call_count, 1)
        self.assertEqual(cpu, on_card)


class WindowBase(ViewerCompareBase):
    document = staticmethod(timed_document)

    def current(self):
        return self.window.dispatcher.document["time"]["current"]


class PlaybackRangeTests(WindowBase):
    def test_play_stays_inside_the_range_and_wraps_at_its_end(self):
        w = self.window
        w.command({"op": "viewer_knobs", "frame_range": "5-8"})
        w.set_time(current=6)
        self.settle()
        w.toggle_playback(True)
        visited = []
        for step in range(1, 7):
            w.playback_origin_time = time.monotonic() - (step + 0.1) / 24.0
            w.playback_frames_rendered = 1000
            w.playback_tick()
            visited.append(self.current())
        w.toggle_playback(False)
        self.assertEqual(visited, [7, 8, 5, 6, 7, 8])

    def test_unlocking_the_range_plays_the_project(self):
        w = self.window
        w.command({"op": "viewer_knobs", "frame_range": "5-8", "frame_range_lock": False})
        w.set_time(current=19)
        w.toggle_playback(True)
        w.playback_origin_time = time.monotonic() - 1.1 / 24.0
        w.playback_frames_rendered = 1000
        w.playback_tick()
        self.assertEqual(self.current(), 20)
        w.toggle_playback(False)

    def test_read_ahead_and_the_jump_buttons_follow_the_range(self):
        w = self.window
        w.command({"op": "viewer_knobs", "frame_range": "5-8"})
        self.assertEqual(w.future_frames(7, count=3), [8, 5, 6])
        w.go_to_start()
        self.assertEqual(self.current(), 5)
        w.go_to_end()
        self.assertEqual(self.current(), 8)

    def test_playing_backwards_once_from_the_start_of_the_range_starts_again_from_its_end(self):
        w = self.window
        w.command({"op": "viewer_knobs", "frame_range": "5-8"})
        w.set_loop_mode("once")
        w.set_time(current=5)
        w.play(-1)
        self.assertEqual(self.current(), 8)
        w.toggle_playback(False)
        w.set_loop_mode("loop")

    def test_the_increment_steps_use_the_knob(self):
        w = self.window
        w.command({"op": "viewer_knobs", "frame_increment": 6})
        w.set_time(current=3)
        w.step_increment(1)
        self.assertEqual(self.current(), 9)
        w.step_increment(1)
        w.step_increment(1)
        self.assertEqual(self.current(), 20, "held at the last frame")
        w.step_increment(-1)
        self.assertEqual(self.current(), 14)
        w.set_time(current=3)
        w.findChild(QAction, "step-increment-forward").trigger()
        self.assertEqual(self.current(), 9)
        w.findChild(QAction, "step-increment-back").trigger()
        self.assertEqual(self.current(), 3)


class SettingsWindowTests(WindowBase):
    def test_the_controls_write_the_document_and_follow_undo(self):
        w = self.window
        dialog = w.open_viewer_settings()
        field = dialog.controls["frame_range"]
        field.setText("4-9")
        field.editingFinished.emit()
        self.assertEqual(viewer_knobs(w.dispatcher.document)["frame_range"], "4-9")
        dialog.controls["safe_zone"].setCurrentText("title safe")
        dialog.controls["safe_zone"].activated.emit(dialog.controls["safe_zone"].currentIndex())
        self.assertEqual(viewer_knobs(w.dispatcher.document)["safe_zone"], "title safe")
        dialog.controls["zoom_lock"].setChecked(True)
        self.assertTrue(viewer_knobs(w.dispatcher.document)["zoom_lock"])
        dialog.controls["frame_increment"].setValue(3)
        self.assertEqual(viewer_knobs(w.dispatcher.document)["frame_increment"], 3)
        dialog.controls["center_fstop"].setValue(5.6)
        self.assertEqual(viewer_knobs(w.dispatcher.document)["center_fstop"], 5.6)
        w.command({"op": "undo"})
        self.assertEqual(dialog.controls["center_fstop"].value(), 8.0)
        w.command({"op": "undo"})
        self.assertEqual(dialog.controls["frame_increment"].value(), 10)
        w.command({"op": "undo"})
        self.assertFalse(dialog.controls["zoom_lock"].isChecked())
        self.assertIn("frames 4 to 9", dialog.note.text())

    def test_every_knob_has_a_control_and_a_label(self):
        dialog = self.window.open_viewer_settings()
        self.assertEqual(set(dialog.controls), set(viewerknobs.DEFAULT))
        for name in viewerknobs.DEFAULT:
            self.assertIn(name, dialog.LABELS)
            self.assertTrue(dialog.controls[name].toolTip())
        types = {name: type(dialog.controls[name]) for name in dialog.controls}
        self.assertIs(types["frame_range"], QLineEdit)
        self.assertIs(types["safe_zone"], QComboBox)
        self.assertIs(types["frame_increment"], QSpinBox)
        self.assertIs(types["rgb_only"], QCheckBox)

    def test_a_badly_typed_range_is_refused_and_the_field_restored(self):
        w = self.window
        dialog = w.open_viewer_settings()
        w.command({"op": "viewer_knobs", "frame_range": "5-8"})
        field = dialog.controls["frame_range"]
        field.setText("soon")
        field.editingFinished.emit()
        self.assertEqual(viewer_knobs(w.dispatcher.document)["frame_range"], "5-8")
        self.assertEqual(field.text(), "5-8")

    def test_the_gain_control_is_labelled_with_the_exposure(self):
        w = self.window
        w.command({"op": "viewer_knobs", "center_fstop": 5.6, "shutter_angle": 90.0})
        self.assertIn("f/5.6 · 90° · 1/96 s", w.exposure.toolTip())
        w.set_viewer_look(gain=2.0)
        self.assertIn("f/2.8", w.exposure.toolTip())


class PictureTests(WindowBase):
    def alpha_value(self):
        self.settle()
        return self.window.display_image().pixelColor(5, 5).red()

    def test_rgb_only_decides_whether_gain_reaches_the_alpha_channel(self):
        w = self.window
        w.channels.setCurrentText("A")
        w.set_viewer_look(gain=1.0)
        self.settle()
        shown = self.alpha_value()
        self.assertEqual(shown, round(0.25 * 255), "on (the default), alpha is shown as it is")
        w.set_viewer_knobs(rgb_only=False)
        expected = float(np.clip(display_rgb(np.full((1, 1, 3), 0.5, np.float32), w.effective_view()), 0, 1)[0, 0, 0]) * 255
        self.assertAlmostEqual(self.alpha_value(), expected, delta=1)
        w.command({"op": "undo"})
        self.assertEqual(self.alpha_value(), shown)

    def test_the_direct_picture_hands_alpha_through_the_view_back_to_the_viewer(self):
        w = self.window
        request = unittest.mock.Mock(document=w.dispatcher.document)
        w._direct_linear_on = True
        with unittest.mock.patch.object(carddisplay, "supported", return_value=True):
            self.assertTrue(w._direct_linear_for(request, "sRGB", None, "RGB"))
            self.assertTrue(w._direct_linear_for(request, "sRGB", None, "A"), "alpha as it is: the card can show it")
            document = copy.deepcopy(w.dispatcher.document)
            document["settings"]["viewer"]["knobs"] = {**viewerknobs.DEFAULT, "rgb_only": False}
            request = unittest.mock.Mock(document=document)
            self.assertTrue(w._direct_linear_for(request, "sRGB", None, "RGB"))
            self.assertFalse(w._direct_linear_for(request, "sRGB", None, "A"))
            document["settings"]["viewer"]["knobs"] = {**viewerknobs.DEFAULT, "use_gpu_for_viewer": False}
            self.assertFalse(w._direct_linear_for(unittest.mock.Mock(document=document), "sRGB", None, "RGB"))

    def test_the_look_the_conversion_receives_only_changes_when_a_knob_is_off(self):
        w = self.window
        self.assertEqual(set(w.look_args()), {"gamma", "zebra"})
        w.command({"op": "viewer_knobs", "rgb_only": False, "use_gpu_for_viewer": False})
        self.assertEqual(w.look_args(), {"gamma": 1.0, "zebra": False, "rgb_only": False, "gpu": False})

    def test_the_picture_is_the_same_without_the_card(self):
        w = self.window
        self.settle()
        before = w.display_image().copy()
        w.set_viewer_knobs(use_gpu_for_viewer=False)
        self.settle()
        self.assertEqual(w.display_image(), before)


def blur_document():
    dispatcher = Dispatcher(empty_document())
    dispatcher.execute({"op": "create", "id": "c", "type": "Constant", "pos": [0, 0],
                        "params": {"width": 160, "height": 80, "red": 0.5, "green": 0.25, "blue": 0.125, "alpha": 1.0}})
    dispatcher.execute({"op": "create", "id": "b", "type": "Blur", "pos": [200, 0], "params": {"radius": 4.0}})
    dispatcher.execute({"op": "connect", "id": "b", "input": "image", "source": "c"})
    dispatcher.execute({"op": "view", "id": "b"})
    return dispatcher.document


class FullFrameTests(WindowBase):
    document = staticmethod(blur_document)

    def spy(self):
        regions = []
        executor = self.window.tile_executor
        original = executor.compose_region

        def record(document, target, region, *args, **kwargs):
            regions.append((region.x, region.y, region.width, region.height))
            return original(document, target, region, *args, **kwargs)

        executor.compose_region = record
        self.addCleanup(lambda: setattr(executor, "compose_region", original))
        return regions

    def zoom_in(self):
        viewer = self.window.viewer
        viewer.auto_fit = False
        viewer.resetTransform()
        viewer.scale(6.0, 6.0)
        viewer.centerOn(viewer.format_rect.center())
        APP.processEvents()

    def test_a_zoomed_view_evaluates_its_visible_part_unless_full_frame_processing_is_on(self):
        w = self.window
        self.settle()
        self.zoom_in()
        regions = self.spy()
        w.tile_executor.cache.clear()
        w.display_cache.clear()
        w.request_preview()
        self.settle()
        visible = regions[-1]
        self.assertLess(visible[2] * visible[3], 160 * 80, "only the part of the frame the view shows")
        w.set_viewer_knobs(full_frame_processing=True)
        self.settle()
        self.zoom_in()
        regions.clear()
        w.tile_executor.cache.clear()
        w.display_cache.clear()
        w.request_preview()
        self.settle()
        self.assertEqual(regions[-1], (0, 0, 160, 80), "the whole frame")


class PixelAspectTests(WindowBase):
    document = staticmethod(lambda: anamorphic_document(2.0))

    def screen_shape(self):
        self.settle()
        viewer = self.window.viewer
        rect = viewer.mapFromScene(viewer.format_rect).boundingRect()
        return rect.width() / rect.height()

    def test_square_pixels_until_ignore_pixel_aspect_is_turned_off(self):
        w = self.window
        self.assertAlmostEqual(self.screen_shape(), 2.0, delta=0.05)          # 160 x 80
        w.set_viewer_knobs(ignore_pixel_aspect=False)
        self.assertAlmostEqual(self.screen_shape(), 4.0, delta=0.1)           # pixels twice as wide
        self.assertEqual(w.viewer.pixel_stretch, 2.0)
        w.command({"op": "undo"})
        self.assertAlmostEqual(self.screen_shape(), 2.0, delta=0.05)

    def test_the_picture_still_fits_the_view(self):
        w = self.window
        w.set_viewer_knobs(ignore_pixel_aspect=False)
        self.settle()
        viewer = w.viewer
        shown = viewer.mapFromScene(viewer.format_rect).boundingRect()
        self.assertLessEqual(shown.width(), viewer.viewport().width())
        self.assertLessEqual(shown.height(), viewer.viewport().height())
        self.assertGreater(shown.width(), viewer.viewport().width() * 0.6)

    def test_scene_points_still_land_on_image_pixels(self):
        w = self.window
        w.set_viewer_knobs(ignore_pixel_aspect=False)
        self.settle()
        viewer = w.viewer
        point = viewer.mapToScene(viewer.mapFromScene(QPointF(40.0, 30.0)))
        self.assertAlmostEqual(point.x(), 40.0, delta=0.6)
        self.assertAlmostEqual(point.y(), 30.0, delta=0.6)

    def test_the_direct_picture_hands_a_stretched_picture_back_to_the_viewer(self):
        w = self.window
        w.set_viewer_knobs(ignore_pixel_aspect=False)
        self.settle()
        request = unittest.mock.Mock(document=w.dispatcher.document)
        w._direct_linear_on = True
        with unittest.mock.patch.object(carddisplay, "supported", return_value=True):
            self.assertFalse(w._direct_linear_for(request, "sRGB", None, "RGB"))
        widget = unittest.mock.Mock()
        image = QImage(160, 80, QImage.Format.Format_RGB888)
        with unittest.mock.patch.object(type(w), "_direct_shown", lambda self: True):
            w.direct_widget = widget
            try:
                w._offer_direct(image, 1, None, QRectF(0, 0, 160, 80), 1)
            finally:
                w.direct_widget = None
        widget.submit.assert_not_called()
        self.assertIn("pixel aspect", w.statusBar().currentMessage())

    def test_an_ordinary_format_goes_to_the_direct_picture_as_before(self):
        w = self.window
        w.set_viewer_knobs(ignore_pixel_aspect=False)
        w.command({"op": "set", "id": "r", "param": "pixel_aspect", "value": 1.0})
        self.settle()
        widget = unittest.mock.Mock()
        widget.submit.return_value = True
        image = QImage(160, 80, QImage.Format.Format_RGB888)
        with unittest.mock.patch.object(type(w), "_direct_shown", lambda self: True), \
                unittest.mock.patch.object(type(w), "_direct_handed_over", lambda self, accepted: None):
            w.direct_widget = widget
            try:
                w._offer_direct(image, 1, None, QRectF(0, 0, 160, 80), 1)
            finally:
                w.direct_widget = None
        widget.submit.assert_called_once()


class ZoomLockTests(WindowBase):
    def resize_picture(self, width):
        self.window.command({"op": "set", "id": "c", "param": "width", "value": width})
        self.settle()

    def test_a_picture_of_another_size_refits_the_view_unless_zoom_is_locked(self):
        w = self.window
        self.settle()
        viewer = w.viewer
        viewer.resetTransform()
        viewer.scale(3.0, 3.0)
        viewer.auto_fit = False
        self.resize_picture(480)
        refit = viewer.transform().m11()
        self.assertNotAlmostEqual(refit, 3.0, delta=0.01, msg="the view was framed again")
        w.set_viewer_knobs(zoom_lock=True)
        viewer.resetTransform()
        viewer.scale(3.0, 3.0)
        viewer.auto_fit = False
        self.resize_picture(160)
        self.assertAlmostEqual(viewer.transform().m11(), 3.0, delta=1e-6)
        w.set_viewer_knobs(zoom_lock=False)
        self.resize_picture(480)
        self.assertNotAlmostEqual(viewer.transform().m11(), 3.0, delta=0.01)


class GuideTests(OverlayParityBase):
    document = staticmethod(timed_document)

    def paint(self, width=200, height=100):
        viewer = self.viewer
        image = QImage(width + 40, height + 40, QImage.Format.Format_RGBA8888_Premultiplied)
        image.fill(0)
        painter = QPainter(image)
        painter.translate(20, 20)
        saved = viewer.format_rect
        viewer.format_rect = QRectF(0, 0, width, height)
        try:
            viewer._draw_viewer_guides(painter)
        finally:
            viewer.format_rect = saved
            painter.end()
        return as_array(image)[..., 3]

    def test_nothing_is_painted_by_default(self):
        self.assertEqual(int(self.paint().max()), 0)

    def test_safe_zone_outlines_sit_at_the_conventional_insets(self):
        self.window.command({"op": "viewer_knobs", "safe_zone": "action safe"})
        alpha = self.paint()
        columns = np.flatnonzero(alpha[70])
        rows = np.flatnonzero(alpha[:, 120])
        # 90 % action safe of 200 x 100 is a rectangle 10 px and 5 px in from the format's edges (the image is offset by 20).
        self.assertTrue(any(abs(int(c) - 30) <= 1 for c in columns) and any(abs(int(c) - 210) <= 1 for c in columns), columns)
        self.assertTrue(any(abs(int(r) - 25) <= 1 for r in rows) and any(abs(int(r) - 115) <= 1 for r in rows), rows)
        self.window.command({"op": "viewer_knobs", "safe_zone": "title and action safe"})
        self.assertGreater(int((self.paint()[70] > 0).sum()), int((alpha[70] > 0).sum()))

    def test_the_format_centre_is_a_cross_in_the_middle(self):
        self.window.command({"op": "viewer_knobs", "format_center": True})
        alpha = self.paint()
        self.assertGreater(int(alpha[70, 110:130].max()), 0)
        self.assertGreater(int(alpha[60:80, 120].max()), 0)
        self.assertEqual(int(alpha[:, 20:60].max()), 0)

    def test_show_overscan_outlines_the_format_and_the_margin(self):
        self.window.command({"op": "viewer_knobs", "show_overscan": True, "overscan": 10})
        alpha = self.paint()
        columns = np.flatnonzero(alpha[70])
        self.assertTrue(any(abs(int(c) - 20) <= 1 for c in columns), "the format's edge")
        self.assertTrue(any(abs(int(c) - 10) <= 1 for c in columns), "the margin, 10 px outside it")

    def test_the_direct_overlay_carries_the_same_guides(self):
        self.window.command({"op": "viewer_knobs", "safe_zone": "title and action safe", "format_center": True,
                             "show_overscan": True, "overscan": 20})
        self.settle()
        for ratio in (1.0, 1.75):
            with self.subTest(ratio=ratio):
                self.assert_card_matches_viewer(ratio)

    def test_a_guide_makes_the_direct_overlay_non_empty_and_none_makes_it_empty(self):
        from nodebased import directviewer as dv
        view, physical, exact = self.pin_view(1.0)
        self.assertIs(self.viewer.render_overlay(view, physical, exact, probe=True), dv.EMPTY_OVERLAY)
        self.window.command({"op": "viewer_knobs", "safe_zone": "title safe"})
        self.settle()
        view, physical, exact = self.pin_view(1.0)
        self.assertIsNot(self.viewer.render_overlay(view, physical, exact, probe=True), dv.EMPTY_OVERLAY)


class OverscanFitTests(WindowBase):
    def test_the_first_fit_leaves_room_for_the_overscan_margin(self):
        w = self.window
        self.settle()
        viewer = w.viewer
        viewer.fit()
        plain = viewer.transform().m11()
        w.set_viewer_knobs(show_overscan=True, overscan=400)
        self.settle()
        self.assertLess(viewer.transform().m11(), plain)
        w.set_viewer_knobs(show_overscan=False)
        self.settle()
        self.assertAlmostEqual(viewer.transform().m11(), plain, delta=1e-6)


class OldDocumentTests(WindowBase):
    def test_an_old_document_draws_as_it_did(self):
        w = self.window
        self.settle()
        self.assertNotIn("knobs", w.dispatcher.document["settings"]["viewer"])
        image = w.display_image().copy()
        w.set_viewer_knobs(safe_zone="action safe")
        w.set_viewer_knobs(frame_range="2-3")
        w.command({"op": "undo"})
        w.command({"op": "undo"})
        self.settle()
        self.assertEqual(viewer_knobs(w.dispatcher.document), viewerknobs.DEFAULT)
        self.assertEqual(w.display_image(), image)


if __name__ == "__main__":
    unittest.main()
