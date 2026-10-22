"""The direct picture follows the frame: handles that move with time, the pixel readout under a still pointer, and the
overlay painted off the Qt thread while a picture plays.

`OverlayRingTests` run the card's overlay ring on a stand-in surface (the inline worker for the pixel checks, the real
thread for the ordering ones); `FollowWindowTests` run the real Window with the dock open on an animated Roto."""
import os
import unittest

import numpy as np

import tests.isolation  # noqa: F401  (Qt settings stay out of the real user file)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPicture

from nodebased import directviewer as dv
from nodebased.app import STYLE
from tests.test_directviewer import APP, checker_image, direct_dock_skip, make_presenter, needs_wgpu
from tests.test_directviewer_overlays import as_array
from tests.test_viewer_compare import ViewerCompareBase
from tests.waiting import wait_until

APP.setStyle("Fusion")
APP.setStyleSheet(STYLE)

SIZE = (160, 120)


def recording(*marks, text=None):
    """A QPicture of red squares at (x, y) (logical pixels) and, with `text`, a line of it: commands only."""
    picture = QPicture()
    painter = QPainter(picture)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    for x, y in marks:
        painter.fillRect(x, y, 4, 3, QColor(255, 0, 0))
    pen = QPen(QColor(0, 255, 0), 1)
    pen.setCosmetic(True)
    painter.setPen(pen)
    painter.drawLine(QPointF(5, 100), QPointF(150, 70.5))
    if text:
        painter.drawText(QPointF(10, 20), text)
    painter.end()
    return picture


def direct_paint(picture, ratio=1.0):
    """The same commands painted straight into an image, as the Viewer's own `render_overlay` does."""
    image = QImage(*SIZE, QImage.Format.Format_RGBA8888_Premultiplied)
    image.setDevicePixelRatio(ratio)
    image.fill(0)
    painter = QPainter(image)
    picture.play(painter)
    painter.end()
    return image


@needs_wgpu
class OverlayRingTests(unittest.TestCase):
    """The overlay recorded on the Qt thread, replayed, compared and staged by the ring, copied inside a draw."""
    threaded = False

    def setUp(self):
        self.presenter, self.context = make_presenter(fill_thread=self.threaded)
        self.addCleanup(self.presenter.lose_device)
        self.assertTrue(self.presenter.offer(checker_image(), (0, 0, 64, 48), (0, 0, 64, 48), 1))
        self.view = (32.0, 24.0, 2.0)
        self.hints = QPainter.RenderHint.Antialiasing
        self.presenter.ring.wait()                      # the picture is on the card before an overlay goes over it
        self.presenter.draw(self.view, self.context.size)

    def deferred(self, picture, key):
        self.presenter.paint_deferred(dv.DeferredOverlay(picture, self.hints), key, SIZE, 1.0)

    def settle(self):
        self.presenter.overlay_ring.wait()

    def shown(self):
        self.settle()
        self.presenter.draw(self.view, self.context.size)
        return self.context.read()

    def plain(self):
        self.presenter.draw(self.view, self.context.size)
        return self.context.read().copy()

    def test_a_replayed_recording_lands_on_the_same_pixels_as_a_direct_paint(self):
        picture = recording((30, 40), (100, 60), text="x 12, y 7")
        self.deferred(picture, "a")
        self.settle()
        replayed = as_array(self.presenter.overlay_ring.previous[0])
        np.testing.assert_array_equal(replayed, as_array(direct_paint(picture)))
        self.assertGreater(int((replayed[..., 3] > 0).sum()), 100)

    def test_the_mark_lands_on_its_pixel_and_nothing_else_changes(self):
        before = self.plain()
        self.deferred(recording((30, 40)), "a")
        after = self.shown()
        self.assertEqual(tuple(after[41, 31, :3]), (255, 0, 0))
        changed = np.argwhere((after != before).any(axis=2))
        self.assertTrue(all(y in range(40, 43) and x in range(30, 34) or y in range(69, 101) for y, x in changed))

    def test_a_second_overlay_uploads_only_what_moved(self):
        self.deferred(recording((30, 40)), "a")
        self.shown()
        rows = self.presenter.counters["paint_upload_rows"]
        self.assertEqual(rows, SIZE[1])                  # the first goes up whole
        self.deferred(recording((30, 41)), "b")        # one row down; the line is the same
        after = self.shown()
        self.assertEqual(self.presenter.counters["paint_upload_rows"] - rows, 4)
        self.assertEqual(tuple(after[43, 31, :3]), (255, 0, 0))
        self.assertNotEqual(tuple(after[40, 31, :3]), (255, 0, 0))

    def test_an_identical_overlay_uploads_nothing(self):
        self.deferred(recording((30, 40)), "a")
        self.shown()
        uploads = self.presenter.counters["paint_uploads"]
        self.deferred(recording((30, 40)), "b")
        self.shown()
        self.assertEqual(self.presenter.counters["paint_uploads"], uploads)
        self.assertEqual(self.presenter.counters["paint_unchanged"], 1)

    def test_an_empty_overlay_clears_the_pass_in_order(self):
        before = self.plain()
        self.deferred(recording((30, 40)), "a")
        self.presenter.paint_empty_deferred("b")
        for _ in range(2):          # the empty state waits behind the overlay before it: each takes a draw
            after = self.shown()
        np.testing.assert_array_equal(after, before)
        self.assertIsNone(self.presenter.paint_size)
        self.deferred(recording((30, 40)), "c")        # and it comes back, painted against what the card still holds
        self.assertEqual(tuple(self.shown()[41, 31, :3]), (255, 0, 0))

    def test_an_empty_overlay_with_nothing_on_the_way_costs_no_job(self):
        self.presenter.paint_empty_deferred("a")
        self.assertEqual(self.presenter.counters["overlay_deferred_empty"], 0)
        self.assertFalse(self.presenter.overlay_busy())
        self.assertEqual(self.presenter.paint_key, "a")

    def test_a_sync_paint_after_the_ring_goes_up_whole_and_is_right(self):
        self.deferred(recording((30, 40)), "a")
        self.shown()
        image = direct_paint(recording((70, 50)))
        self.presenter.set_paint(image, "s", nonempty=True)
        after = self.shown()
        self.assertEqual(tuple(after[51, 71, :3]), (255, 0, 0))
        self.assertNotEqual(tuple(after[41, 31, :3]), (255, 0, 0))

    def test_the_ring_after_a_sync_paint_is_a_whole_image_compared_with_what_the_card_holds(self):
        self.presenter.set_paint(direct_paint(recording((70, 50))), "s", nonempty=True)
        self.shown()
        self.deferred(recording((30, 40)), "a")
        after = self.shown()
        self.assertEqual(tuple(after[41, 31, :3]), (255, 0, 0))
        self.assertNotEqual(tuple(after[51, 71, :3]), (255, 0, 0))

    def test_an_overlay_in_flight_when_the_texture_is_written_another_way_is_dropped_and_asked_for_again(self):
        self.deferred(recording((30, 40)), "a")
        self.settle()
        self.presenter.set_paint(direct_paint(recording((70, 50))), "s", nonempty=True)
        after = self.shown()
        self.assertEqual(tuple(after[51, 71, :3]), (255, 0, 0))            # the newer write is what is on the card
        self.assertNotEqual(tuple(after[41, 31, :3]), (255, 0, 0))
        self.assertTrue(self.presenter.overlay_lost())
        self.assertFalse(self.presenter.overlay_lost())                     # said once
        self.assertIsNone(self.presenter.paint_key)

    def test_a_surface_of_another_size_is_painted_whole_and_the_old_size_is_not_drawn(self):
        self.deferred(recording((30, 40)), "a")
        self.shown()
        bigger = (200, 150)
        self.presenter.paint_deferred(dv.DeferredOverlay(recording((30, 40)), self.hints), "b", bigger, 1.0)
        self.settle()
        rows = self.presenter.counters["paint_upload_rows"]
        self.presenter.draw(self.view, (160, 120))
        self.assertEqual(self.presenter.paint_size, bigger)
        self.assertEqual(self.presenter.counters["paint_upload_rows"] - rows, bigger[1])      # whole

    def test_recovery_forgets_what_the_ring_had_and_paints_again(self):
        self.deferred(recording((30, 40)), "a")
        self.shown()
        self.presenter.recover("test")
        self.assertIsNone(self.presenter.paint_key)
        self.assertIsNone(self.presenter.paint_size)
        self.assertFalse(self.presenter.overlay_busy())
        self.deferred(recording((30, 40)), "b")
        self.assertEqual(tuple(self.shown()[41, 31, :3]), (255, 0, 0))

    def test_a_newer_overlay_replaces_one_that_never_started(self):
        ring = self.presenter.overlay_ring
        with ring.lock:
            ring.running = object()          # the worker is held: nothing starts
        self.deferred(recording((30, 40)), "a")
        self.deferred(recording((70, 50)), "b")
        self.assertEqual(self.presenter.counters["overlay_replaced"], 1)
        with ring.lock:
            ring.running = None
        ring.pump()
        after = self.shown()
        self.assertEqual(tuple(after[51, 71, :3]), (255, 0, 0))
        self.assertNotEqual(tuple(after[41, 31, :3]), (255, 0, 0))


@needs_wgpu
class ThreadedOverlayRingTests(OverlayRingTests):
    """The same checks with the worker on its own thread, so a draw can find the overlay not done yet."""
    threaded = True

    def test_a_draw_that_finds_the_overlay_unfinished_shows_it_at_the_next_draw(self):
        self.deferred(recording((30, 40)), "a")
        self.settle()
        self.presenter.draw(self.view, self.context.size)
        self.assertEqual(tuple(self.context.read()[41, 31, :3]), (255, 0, 0))
        for index in range(10):
            self.deferred(recording((30 + index, 40)), f"k{index}")
        for _ in range(3):          # the one that was running is copied first; the newest follows at a later draw
            self.shown()
        self.assertEqual(tuple(self.context.read()[41, 40, :3]), (255, 0, 0))       # the newest, in order, nothing skipped over it
        self.assertNotEqual(tuple(self.context.read()[41, 31, :3]), (255, 0, 0))


class OverlayFollowsFrameTests(ViewerCompareBase):
    """With an animated Roto selected the handles are read again for every new frame."""

    @staticmethod
    def document():
        from tests.test_roto import keyed, square
        from tests.test_roto_ui import roto_document
        shape = square(10, 10, 60, 60)
        for point in shape["points"]:
            point["x"] = keyed(point["x"], [(1, point["x"]), (2, point["x"] + 20.0)])
        return roto_document([shape])

    def setUp(self):
        super().setUp()
        self.window.graph.items_by_id["r"].setSelected(True)
        APP.processEvents()

    def test_a_selected_roto_follows_the_frame_and_a_wipe_does_not(self):
        self.assertTrue(self.viewer.overlay_follows_frame())
        self.window.graph.scene().clearSelection()
        APP.processEvents()
        self.assertFalse(self.viewer.overlay_follows_frame())

    def test_the_overlay_painted_for_frame_two_is_not_the_one_for_frame_one(self):
        view = (50.0, 50.0, 4.0)
        size = (400, 300)
        self.window.set_time(current=1)
        first = as_array(self.viewer.render_overlay(view, size, 1.0))
        self.window.set_time(current=2)
        second = as_array(self.viewer.render_overlay(view, size, 1.0))
        self.assertGreater(int((first[..., 3] > 0).sum()), 50)
        self.assertFalse(np.array_equal(first, second))

    def test_a_recording_replayed_off_the_thread_is_the_overlay_the_viewer_paints(self):
        view, size = (50.0, 50.0, 4.0), (400, 300)
        deferred = self.viewer.render_overlay(view, size, 1.0, probe=True, deferred=True)
        self.assertIsInstance(deferred, dv.DeferredOverlay)
        again = self.viewer.render_overlay(view, size, 1.0, probe=True, deferred=True)
        self.assertIs(again, dv.SAME_OVERLAY)            # the memo knows the recording
        image = QImage(*size, QImage.Format.Format_RGBA8888_Premultiplied)
        image.fill(0)
        painter = QPainter(image)
        painter.setRenderHints(deferred.hints)
        deferred.picture.play(painter)
        painter.end()
        direct = as_array(self.viewer.render_overlay(view, size, 1.0))
        replayed = as_array(image)
        # the same shapes in the same places: a pixel of one has the other's colour within a step of it
        self.assertEqual(replayed.shape, direct.shape)
        self.assertEqual(int(((replayed[..., 3] > 0) != (direct[..., 3] > 0)).sum()), 0)
        self.assertLessEqual(int(np.abs(replayed.astype(int) - direct.astype(int)).max()), 2)


class FollowWindowTests(OverlayFollowsFrameTests):
    """The real Window with the dock open: a new frame bumps the overlay and the readout under a still pointer."""

    def setUp(self):
        reason = direct_dock_skip()
        if reason:
            self.skipTest(reason)
        super().setUp()
        self.window.direct_action.setChecked(True)
        self.widget = self.window.direct_widget
        self.assertIsNotNone(self.widget)
        self.assertTrue(wait_until(lambda: self.widget.presenter is not None and self.widget.presenter.has_picture()))
        self.addCleanup(lambda: self.window.direct_action.setChecked(False))

    def test_a_new_frame_asks_for_a_new_overlay_only_when_the_handles_move_with_it(self):
        self.settle()
        serial = self.widget.overlay_serial
        self.window.set_time(current=2)
        self.assertGreater(self.widget.overlay_serial, serial)
        self.window.graph.scene().clearSelection()
        APP.processEvents()
        serial = self.widget.overlay_serial
        self.window.set_time(current=1)
        self.assertEqual(self.widget.overlay_serial, serial)

    def test_playing_turns_the_overlay_ring_on_and_stopping_turns_it_off(self):
        self.assertFalse(self.widget.overlay_async)
        self.window.toggle_playback(True)
        self.assertTrue(self.widget.overlay_async)
        self.window.toggle_playback(False)
        self.assertFalse(self.widget.overlay_async)
        self.window.toggle_playback(True)
        self.window.direct_action.setChecked(False)
        self.window.direct_action.setChecked(True)
        self.assertTrue(self.window.direct_widget.overlay_async)       # a dock opened while playing starts the same way
        self.window.toggle_playback(False)

    def test_the_readout_under_a_still_pointer_follows_the_picture(self):
        self.settle()
        self.assertTrue(wait_until(lambda: self.widget.rect is not None))
        width, height = self.widget._physical()
        self.widget.view = dv.fit_view((width, height), self.widget.rect)
        point = self.widget.scene_point(width / 2.0, height / 2.0)
        self.window._direct_hover(point)
        first = dict(self.window._direct_readout)
        self.window.command({"op": "set", "id": "r", "param": "width", "value": 100})     # any change that makes a new picture
        self.window.frame_b = None
        self.window._direct_frame_moved()
        self.assertEqual(self.window._direct_pointer, point)
        self.assertEqual(self.window._direct_readout["text"], first["text"])              # the same pixel, read again
        # a picture with other values: the readout changes without the pointer moving
        self.window.frame = self.window.frame.copy()
        self.window.frame[..., 0] += 0.25
        self.window._direct_frame_moved()
        self.assertNotEqual(self.window._direct_readout["text"], first["text"])
        self.assertAlmostEqual(self.window._direct_readout["rgba"][0], first["rgba"][0] + 0.25, places=5)
        self.window._direct_hover(None)
        self.window._direct_frame_moved()
        self.assertIsNone(self.window._direct_readout)

    def test_a_view_that_moved_under_a_still_pointer_reads_the_point_now_under_it(self):
        self.settle()
        self.assertTrue(wait_until(lambda: self.widget.rect is not None))
        calls = []
        self.widget.hover = lambda scene: calls.append(scene)
        width, height = self.widget._physical()
        self.widget.view = dv.fit_view((width, height), self.widget.rect)
        self.widget.auto_fit = False
        self.widget._pointer = (width * 0.25, height * 0.5)
        self.widget._hover_view = self.widget.view
        # a pan (the view moved under a still pointer) is read again at the next draw, with the pointer where it was
        cx, cy, zoom = self.widget.view
        self.widget.view = (cx + 10.0 / zoom, cy, zoom)
        draws = self.widget.presenter.counters["draws"]
        self.widget._request()
        self.assertTrue(wait_until(lambda: self.widget.presenter.counters["draws"] > draws))
        self.assertTrue(calls)
        self.assertEqual(calls[-1], self.widget.scene_point(width * 0.25, height * 0.5))
        self.assertEqual(self.widget._hover_view, self.widget.view)


if __name__ == "__main__":
    unittest.main()


class DirectMainWindowTests(ViewerCompareBase):
    """The real Window and the real card with the switch on: the picture in the Viewer's place with the strip and the
    timeline around it, the corner readout on the picture, and the hand-back to the Viewer."""

    def setUp(self):
        reason = direct_dock_skip()
        if reason:
            self.skipTest(reason)
        from PySide6.QtCore import QSettings
        from nodebased.app import Preferences
        QSettings("NodeBased", "NodeBased").remove(Preferences.DIRECT_MAIN)
        self.addCleanup(lambda: QSettings("NodeBased", "NodeBased").remove(Preferences.DIRECT_MAIN))
        super().setUp()
        self.addCleanup(lambda: self.window.direct_main_action.setChecked(False))

    def switch_on(self):
        self.window.direct_main_action.setChecked(True)
        self.assertTrue(wait_until(lambda: self.window.direct_widget is not None
                                   and self.window.direct_widget.presenter is not None
                                   and self.window.direct_widget.presenter.has_picture()))
        self.widget = self.window.direct_widget
        self.assertTrue(wait_until(lambda: self.window._dock_front["direct"] and not self.window._dock_front["viewer"]))
        APP.processEvents()

    def test_the_switch_puts_the_card_in_the_viewers_place_with_the_controls_around_it(self):
        self.switch_on()
        window = self.window
        self.assertTrue(wait_until(lambda: window.viewer_strip.docked))
        self.assertTrue(window.direct_panel.isAncestorOf(window.timeline_panel))
        self.assertTrue(window.direct_panel.isAncestorOf(window.viewer_strip))
        self.assertTrue(window.direct_panel.isAncestorOf(self.widget))
        window.direct_main_action.setChecked(False)
        self.assertTrue(wait_until(lambda: not window.viewer_strip.docked and window.viewer_panel.isAncestorOf(window.timeline_panel)))
        self.assertTrue(wait_until(lambda: window._dock_front["viewer"] and not window._dock_front["direct"]))

    def test_the_picture_size_and_the_zoom_are_drawn_on_the_card(self):
        self.switch_on()
        presenter = self.widget.presenter
        self.assertTrue(wait_until(lambda: presenter.hud_shown is not None))
        first = presenter.hud_shown
        self.assertIn("%", first[1])
        self.assertIn("×", first[1])
        before = presenter.counters["badge_uploads"]
        draws = presenter.counters["draws"]
        cx, cy, zoom = self.widget.view
        self.widget.view = (cx, cy, zoom * 1.5)
        self.widget.auto_fit = False
        self.widget._request()
        self.assertTrue(wait_until(lambda: presenter.counters["draws"] > draws and presenter.hud_shown != first))
        self.assertGreater(presenter.counters["badge_uploads"], before)       # a new zoom is a new label
        self.widget.view = (cx, cy, zoom)
        draws = presenter.counters["draws"]
        self.widget._request()
        self.assertTrue(wait_until(lambda: presenter.counters["draws"] > draws and presenter.hud_shown == first))

    def test_a_card_that_cannot_start_hands_back_to_the_viewer_with_a_message(self):
        import unittest.mock
        with unittest.mock.patch.object(dv, "make_widget", side_effect=RuntimeError("no surface here")):
            self.window.direct_main_action.setChecked(True)
            self.assertTrue(wait_until(lambda: "unavailable" in self.window.statusBar().currentMessage()))
        self.assertIsNone(self.window.direct_widget)
        self.assertTrue(wait_until(lambda: self.window._dock_front["viewer"]))
        self.assertFalse(self.window.viewer_strip.docked)
        self.assertTrue(self.window.viewer_panel.isAncestorOf(self.window.timeline_panel))

    def test_a_picture_the_card_refuses_brings_the_viewer_forward(self):
        self.switch_on()
        window = self.window
        refusal = "Direct picture: this picture is larger than the card can hold; the Viewer shows it."
        original = self.widget.submit_linear

        def refuse(*args, **kwargs):
            self.widget.refusal = refusal
            return False
        self.widget.submit_linear = refuse
        self.addCleanup(lambda: setattr(self.widget, "submit_linear", original))
        messages = []
        window.statusBar().messageChanged.connect(messages.append)
        window.command({"op": "set", "id": "a", "param": "red", "value": 0.7})
        self.settle()
        self.assertTrue(wait_until(lambda: window._dock_front["viewer"] and not window._dock_front["direct"]))
        self.assertTrue(any("Showing the Viewer" in message for message in messages), messages)
        self.assertTrue(window.viewer_panel.isAncestorOf(window.timeline_panel))

    def test_fitting_the_viewer_fits_the_card_and_keys_go_to_the_viewer(self):
        import unittest.mock
        from PySide6.QtCore import QEvent, Qt
        from PySide6.QtGui import QKeyEvent
        self.switch_on()
        self.widget.auto_fit = False
        self.window.viewer.fit()
        self.assertTrue(self.widget.auto_fit)
        self.widget.auto_fit = False
        with unittest.mock.patch.object(type(self.window.viewer), "keyPressEvent") as handler:
            self.widget.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_F, Qt.KeyboardModifier.NoModifier, "f"))
        handler.assert_called_once()
