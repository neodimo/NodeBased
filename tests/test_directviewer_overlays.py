"""VP2b: overlays and the pixel readout on the direct picture.

The overlays are the Viewer's own (`Viewer.paint_overlays`), painted for the direct view's transform into a
premultiplied image that the card draws one texel per surface pixel. The pixel tests here compare that image to
what the Viewer's own widget paints, at scale 1.0 and 1.75."""
import os
import unittest
import unittest.mock
import uuid

import numpy as np

import tests.isolation  # noqa: F401  (Qt settings stay out of the real user file)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QPoint, QPointF, QRectF, QSettings
from PySide6.QtGui import QImage, QPainter

from nodebased import directviewer as dv
from nodebased.app import STYLE, Window
from tests.test_directviewer import APP, HAVE_WGPU, direct_dock_skip, make_presenter, needs_wgpu
from tests.test_viewer_compare import ViewerCompareBase
from tests.waiting import wait_until

APP.setStyle("Fusion")
APP.setStyleSheet(STYLE)


def as_array(image):
    """Premultiplied RGBA8 bytes of a QImage as (h, w, 4)."""
    image = image.convertToFormat(QImage.Format.Format_RGBA8888_Premultiplied)
    raw = np.frombuffer(image.constBits(), np.uint8)
    return raw.reshape(image.height(), image.bytesPerLine())[:, :image.width() * 4].reshape(image.height(), image.width(), 4).copy()


class OverlayParityBase(ViewerCompareBase):
    """Window with the Viewer pinned to a known view, and a way to compare its paint to the card's overlay."""

    def pin_view(self, ratio):
        """Fix the Viewer's view and return the direct view (centre, physical zoom) and size that match it."""
        viewer = self.viewer
        viewer.auto_fit = False
        viewer.resetTransform()
        viewer.scale(0.9, 0.9)
        viewer.centerOn(self.viewer.format_rect.center())
        APP.processEvents()
        size = viewer.viewport().size()
        # The exact ratio that makes the physical image an integer size, as a real display's does.
        physical = (round(size.width() * ratio), round(size.height() * ratio))
        exact = physical[0] / size.width()
        # The Viewer's own scene-to-viewport map (its translation is a whole number of logical pixels). The direct
        # picture centres on half its physical size, which on the short axis is not quite half the logical size
        # times the ratio (307 * 1.75 = 537.25 rounds to 537), so the centre is solved on the physical pixels:
        # a thin slanted line then lands on the same sub-pixel phase as the Viewer's and antialiases identically.
        t = viewer.viewportTransform()
        zoom = t.m11() * exact
        centre = QPointF((physical[0] / 2.0 - t.dx() * exact) / zoom, (physical[1] / 2.0 - t.dy() * exact) / zoom)
        return (centre.x(), centre.y(), zoom), physical, exact

    def viewer_paint(self, physical, exact, with_overlays):
        """The Viewer widget's own paint, rendered at `exact` device pixel ratio."""
        image = QImage(*physical, QImage.Format.Format_RGBA8888_Premultiplied)
        image.setDevicePixelRatio(exact)
        image.fill(0)
        painter = QPainter(image)
        try:
            if with_overlays:
                self.viewer.viewport().render(painter, QPoint())
            else:
                with unittest.mock.patch.object(type(self.viewer), "paint_overlays", lambda *a, **k: None):
                    self.viewer.viewport().render(painter, QPoint())
        finally:
            painter.end()
        return image

    def assert_card_matches_viewer(self, ratio, expect_overlay=True, tolerance=16):
        view, physical, exact = self.pin_view(ratio)
        base = as_array(self.viewer_paint(physical, exact, False)).astype(int)
        full = as_array(self.viewer_paint(physical, exact, True)).astype(int)
        mine = as_array(self.viewer.render_overlay(view, physical, exact)).astype(int)
        # the card draws the overlay with premultiplied "over" onto the picture
        composite = mine + base * (255 - mine[..., 3:4]) // 255
        changed = np.abs(full - base).max(axis=2) > 0
        # The corner labels ("ROI", "proxy 1/2") are text: Qt antialiases it differently on an opaque widget
        # than on a transparent image, so the label strip is compared for ink, not for pixels.
        label = (slice(0, int(26 * exact)), slice(physical[0] - int(130 * exact), physical[0]))
        self.assertEqual(bool(changed[label].any()), bool((mine[label][..., 3] > 0).any()), "label drawn on one side only")
        composite[label] = full[label]
        changed[label] = False
        if expect_overlay:
            self.assertGreater(int(changed.sum()), 40, "the Viewer drew no overlay here")
        # "The same place" means to within one pixel's rounding: the Viewer's scroll offset is a whole number of
        # logical pixels and a rectangle edge on a pixel boundary can round either way. A pixel agrees when some
        # pixel of the Viewer's paint within one step of it has the card's colour.
        diff = np.full(full.shape[:2], 255)
        padded = np.pad(full[..., :3], ((1, 1), (1, 1), (0, 0)), mode="edge")
        for dy in range(3):
            for dx in range(3):
                near = padded[dy:dy + full.shape[0], dx:dx + full.shape[1]]
                diff = np.minimum(diff, np.abs(composite[..., :3] - near).max(axis=2))
        bad = int((diff > tolerance).sum())
        worst = int(diff.max())
        self.assertLessEqual(bad, max(8, int(changed.sum()) // 150),
                             f"{bad} pixels differ from the Viewer by more than {tolerance}/255 even allowing one "
                             f"pixel of shift (worst {worst}), {int(changed.sum())} overlay pixels")
        return int(changed.sum())


class WipeParityTests(OverlayParityBase):
    def test_wipe_line_split_and_handles_match_the_viewer(self):
        self.compare(2, "wipe")
        self.assertTrue(self.viewer._wipe_active())
        for ratio in (1.0, 1.75):
            with self.subTest(ratio=ratio):
                self.assert_card_matches_viewer(ratio)

    def test_a_moved_and_turned_wipe_matches_too(self):
        self.compare(2, "wipe")
        self.viewer.wipe.update(x=0.3, y=0.6, angle=35.0)
        self.assert_card_matches_viewer(1.75)

    def test_the_b_half_is_in_the_overlay(self):
        self.compare(2, "wipe")
        view, physical, exact = self.pin_view(1.0)
        mine = as_array(self.viewer.render_overlay(view, physical, exact))
        self.assertGreater(int((mine[..., 3] == 255).sum()), 1000)   # B's opaque half of the picture


class RegionAndMaskParityTests(OverlayParityBase):
    def test_region_of_interest_box_matches_the_viewer(self):
        self.window.set_viewer_roi(on=True, rect=[0.2, 0.25, 0.7, 0.8])
        self.settle()
        for ratio in (1.0, 1.75):
            with self.subTest(ratio=ratio):
                self.assert_card_matches_viewer(ratio)

    def test_format_masks_match_the_viewer(self):
        for mode in ("half", "full", "lines"):
            with self.subTest(mode=mode):
                self.window.command({"op": "viewer_mask", "mask": "2.35", "mode": mode})
                self.settle()
                self.assert_card_matches_viewer(1.0)
                self.assert_card_matches_viewer(1.75)

    def test_nothing_to_draw_gives_an_empty_overlay(self):
        self.window.command({"op": "viewer_mask", "mask": "format", "mode": "none"})
        self.window.set_viewer_roi(on=False)
        self.settle()
        view, physical, exact = self.pin_view(1.0)
        mine = as_array(self.viewer.render_overlay(view, physical, exact))
        self.assertEqual(int(mine[..., 3].max()), 0)


class HandleParityTests(unittest.TestCase):
    """Roto and Transform handles, painted read-only in the same place."""

    def make(self, document, select):
        QSettings("NodeBased", "NodeBased").clear()
        self.window = Window(document, agent_name="nodebased-test-" + uuid.uuid4().hex)
        self.window.resize(900, 700)
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None and self.window.viewer.format_rect is not None))
        self.window.graph.items_by_id[select].setSelected(True)
        APP.processEvents()
        self.addCleanup(self.close)
        base = OverlayParityBase("assert_card_matches_viewer")
        base.window, base.viewer = self.window, self.window.viewer
        return base

    def close(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def test_roto_shapes_and_points_match_the_viewer(self):
        from tests.test_roto_ui import roto_document, triangle
        base = self.make(roto_document([triangle()]), "r")
        self.assertIsNotNone(self.window.viewer._roto_context())
        for ratio in (1.0, 1.75):
            with self.subTest(ratio=ratio):
                base.assert_card_matches_viewer(ratio)

    def test_transform_box_and_pivot_match_the_viewer(self):
        from tests.test_transform_handle_ui import transform_document
        base = self.make(transform_document(), "t")
        self.assertIsNotNone(self.window.viewer._transform_context())
        self.window.command({"op": "set", "id": "t", "param": "rotate", "value": 20.0})
        APP.processEvents()
        for ratio in (1.0, 1.75):
            with self.subTest(ratio=ratio):
                base.assert_card_matches_viewer(ratio)


class ReadoutTests(OverlayParityBase):
    def hdr_window(self):
        self.window.command({"op": "set", "id": "a", "param": "red", "value": 2.5})
        self.settle()

    def test_the_readout_reads_the_float_picture_not_the_display_bytes(self):
        self.hdr_window()
        rect = self.viewer.format_rect
        info = self.viewer.pixel_readout_info(QPointF(rect.left() + 10.5, rect.top() + 7.5))
        self.assertIsNotNone(info)
        self.assertEqual(info["rgba"][0], 2.5)                 # a display byte would be 255 (1.0)
        self.assertIn("2.50000", info["text"])
        self.assertTrue(info["text"].startswith("10, "))

    def test_the_card_readout_reports_what_the_viewer_readout_reports(self):
        self.hdr_window()
        rect = self.viewer.format_rect
        for fx, fy in ((12.2, 9.7), (80.5, 40.5), (150.0, 70.9)):
            scene = QPointF(rect.left() + fx, rect.top() + fy)
            point = self.viewer.mapFromScene(scene)
            self.viewer._update_pixel_readout(unittest.mock.Mock(**{"position.return_value": QPointF(point)}))
            info = self.viewer.pixel_readout_info(self.viewer.mapToScene(point))
            readout = self.viewer.pixel_readout
            self.assertEqual(readout.position.text(), info["position"])
            self.assertEqual(readout.values.text(), info["values"])
            self.assertEqual(info["values"].split()[0], "2.500")      # float value above 1, never a display byte

    def test_a_played_frame_is_read_with_its_own_size_and_place(self):
        import types
        rect = self.viewer.format_rect
        frame = np.zeros((20, 30, 4), np.float32)
        frame[5, 7] = (4.0, 0.5, 0.25, 1.0)
        # the card shows a 30 x 20 crop whose corner is at (100, 40) of a 160 x 80 format; the held picture is another size
        region = types.SimpleNamespace(x=100, y=40, full_x=0, full_y=0, full_width=160, full_height=80)
        self.window.frame = frame
        self.viewer.readout_geometry = (1, region, (30, 20))
        info = self.viewer.pixel_readout_info(QPointF(rect.left() + 107.5, rect.top() + 45.5))
        self.assertEqual(info["rgba"], (4.0, 0.5, 0.25, 1.0))
        self.assertTrue(info["text"].startswith("107, "))
        self.assertIsNone(self.viewer.pixel_readout_info(QPointF(rect.left() + 20, rect.top() + 10)))   # outside the crop
        self.viewer.readout_geometry = None

    def test_off_the_picture_there_is_no_readout(self):
        rect = self.viewer.format_rect
        self.assertIsNone(self.viewer.pixel_readout_info(QPointF(rect.right() + 50, rect.bottom() + 50)))

    def test_the_readout_is_painted_on_the_card_overlay(self):
        self.hdr_window()
        rect = self.viewer.format_rect
        info = self.viewer.pixel_readout_info(QPointF(rect.left() + 10.5, rect.top() + 7.5))
        view, physical, exact = self.pin_view(1.0)
        plain = as_array(self.viewer.render_overlay(view, physical, exact))
        shown = as_array(self.viewer.render_overlay(view, physical, exact, readout=info))
        self.assertEqual(int(plain[..., 3].max()), 0)
        self.assertGreater(int((shown[..., 3] > 0).sum()), 200)
        rows = np.nonzero(shown[..., 3].max(axis=1))[0]
        self.assertGreater(rows.min(), physical[1] * 0.7)       # bottom left of the surface


class DirectCoordinateTests(unittest.TestCase):
    def test_scene_point_is_the_inverse_of_the_view_the_shader_draws(self):
        needs_display = os.name != "nt" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if needs_display or not HAVE_WGPU:
            self.skipTest("the direct widget needs a window and the card library")
        widget = dv.make_widget()
        self.addCleanup(widget.close)
        widget.resize(400, 300)
        widget.show()
        APP.processEvents()
        widget.view = (50.0, 40.0, 2.0)
        width, height = widget._physical()
        x, y = widget.scene_point(width / 2.0, height / 2.0)
        self.assertEqual((x, y), (50.0, 40.0))
        x, y = widget.scene_point(width / 2.0 + 100.0, height / 2.0 - 60.0)
        self.assertEqual((x, y), (100.0, 10.0))


@needs_wgpu
class WindowOverlayTests(ViewerCompareBase):
    """The real Window with the direct dock open: overlays reach the card, the pointer reads the float picture,
    and clicks on the direct surface still only navigate."""

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

    def test_a_wipe_reaches_the_card_and_follows_the_viewer(self):
        self.compare(2, "wipe")
        self.assertTrue(wait_until(lambda: self.widget.presenter.counters["paint_uploads"] >= 1))
        uploads = self.widget.presenter.counters["paint_uploads"]
        self.viewer.wipe.update(x=0.25)
        self.viewer.viewport().update()
        self.assertTrue(wait_until(lambda: self.widget.presenter.counters["paint_uploads"] > uploads))

    def test_hovering_the_direct_picture_reads_the_float_picture(self):
        self.window.command({"op": "set", "id": "a", "param": "red", "value": 2.5})
        self.settle()
        self.assertTrue(wait_until(lambda: self.widget.rect is not None))
        width, height = self.widget._physical()
        self.widget.view = dv.fit_view((width, height), self.widget.rect)
        self.widget.hover(self.widget.scene_point(width / 2.0, height / 2.0))
        info = self.window._direct_readout
        self.assertIsNotNone(info)
        self.assertEqual(info["rgba"][0], 2.5)
        self.widget.hover(None)
        self.assertIsNone(self.window._direct_readout)

    def test_clicks_on_the_direct_surface_change_nothing_in_the_viewer(self):
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest
        self.compare(2, "wipe")
        wipe, roi = dict(self.viewer.wipe), self.window.dispatcher.document["settings"]["viewer"].get("roi")
        centre = QPoint(self.widget.width() // 2, self.widget.height() // 2)
        QTest.mouseClick(self.widget, Qt.MouseButton.LeftButton, pos=centre)
        QTest.mousePress(self.widget, Qt.MouseButton.LeftButton, pos=centre)
        QTest.mouseMove(self.widget, centre + QPoint(30, 10))
        QTest.mouseRelease(self.widget, Qt.MouseButton.LeftButton, pos=centre + QPoint(30, 10))
        self.assertEqual(self.viewer.wipe, wipe)
        self.assertEqual(self.window.dispatcher.document["settings"]["viewer"].get("roi"), roi)


@needs_wgpu
class CardPaintTests(unittest.TestCase):
    """The painted image goes on the card one texel per pixel, over the picture."""

    def setUp(self):
        self.presenter, self.context = make_presenter()
        self.addCleanup(self.presenter.lose_device)
        from tests.test_directviewer import checker_image
        self.assertTrue(self.presenter.offer(checker_image(), (0, 0, 64, 48), (0, 0, 64, 48), 1))
        self.view = (32.0, 24.0, 2.0)

    def paint(self, fill):
        image = QImage(160, 120, QImage.Format.Format_RGBA8888_Premultiplied)
        image.fill(0)
        fill(image)
        return image

    def test_an_opaque_texel_lands_on_the_same_pixel_and_the_rest_is_untouched(self):
        def fill(image):
            painter = QPainter(image)
            painter.fillRect(30, 40, 4, 3, 0xFFFF0000)
            painter.end()
        self.presenter.draw(self.view, self.context.size)
        before = self.context.read().copy()
        self.presenter.set_paint(self.paint(fill), "k")
        self.presenter.draw(self.view, self.context.size)
        after = self.context.read()
        self.assertEqual(tuple(after[41, 31, :3]), (255, 0, 0))
        changed = np.argwhere((after != before).any(axis=2))
        self.assertEqual({tuple(p) for p in changed}, {(y, x) for y in range(40, 43) for x in range(30, 34)})

    def test_a_half_transparent_texel_blends_with_the_picture(self):
        def fill(image):
            painter = QPainter(image)
            painter.setOpacity(0.5)
            painter.fillRect(30, 40, 4, 3, 0xFFFFFFFF)
            painter.end()
        self.presenter.draw(self.view, self.context.size)
        before = self.context.read()[41, 31, :3].astype(int)
        self.presenter.set_paint(self.paint(fill), "k")
        self.presenter.draw(self.view, self.context.size)
        after = self.context.read()[41, 31, :3].astype(int)
        np.testing.assert_allclose(after, (before + 255) / 2, atol=3)

    def test_clearing_the_paint_removes_it_and_a_wrong_size_is_not_drawn(self):
        def fill(image):
            painter = QPainter(image)
            painter.fillRect(0, 0, 160, 120, 0xFFFF0000)
            painter.end()
        self.presenter.set_paint(self.paint(fill), "k")
        self.presenter.draw(self.view, self.context.size)
        self.assertEqual(tuple(self.context.read()[5, 5, :3]), (255, 0, 0))
        self.presenter.draw(self.view, (160, 120))
        self.presenter.set_paint(None, None)
        self.presenter.draw(self.view, self.context.size)
        self.assertNotEqual(tuple(self.context.read()[5, 5, :3]), (255, 0, 0))
        self.presenter.set_paint(self.paint(fill), "k")
        self.presenter.paint_size = (100, 100)       # a stale texture from before a resize
        self.presenter.draw(self.view, self.context.size)
        self.assertNotEqual(tuple(self.context.read()[5, 5, :3]), (255, 0, 0))

    @staticmethod
    def mark(x, y):
        def fill(image):
            painter = QPainter(image)
            painter.fillRect(x, y, 4, 3, 0xFFFF0000)
            painter.end()
        return fill

    def test_an_unchanged_overlay_is_uploaded_once(self):
        self.presenter.set_paint(self.paint(self.mark(30, 40)), "k")
        for _ in range(5):
            self.presenter.draw(self.view, self.context.size)
        self.assertEqual(self.presenter.counters["paint_uploads"], 1)
        self.presenter.set_paint(self.paint(self.mark(30, 40)), "k2")      # painted again, same pixels
        self.presenter.draw(self.view, self.context.size)
        self.assertEqual(self.presenter.counters["paint_uploads"], 1)
        self.assertEqual(self.presenter.counters["paint_unchanged"], 1)

    def test_an_empty_overlay_is_not_uploaded_and_its_pass_is_skipped(self):
        self.presenter.draw(self.view, self.context.size)
        before = self.context.read().copy()
        self.presenter.set_paint(self.paint(lambda i: None), "k")
        self.presenter.draw(self.view, self.context.size)
        self.assertEqual(self.presenter.counters["paint_uploads"], 0)
        self.assertEqual(self.presenter.counters["paint_empty"], 1)
        self.assertIsNone(self.presenter.paint_size)
        np.testing.assert_array_equal(self.context.read(), before)

    def test_only_the_changed_rows_go_up_and_the_picture_is_right(self):
        self.presenter.draw(self.view, self.context.size)
        plain = self.context.read().copy()
        self.presenter.set_paint(self.paint(self.mark(30, 40)), "a")
        self.presenter.draw(self.view, self.context.size)
        self.assertEqual(tuple(self.context.read()[41, 31, :3]), (255, 0, 0))
        full = self.presenter.counters["paint_upload_rows"]
        self.presenter.set_paint(self.paint(self.mark(30, 41)), "b")        # moved one row down
        self.presenter.draw(self.view, self.context.size)
        self.assertEqual(self.presenter.counters["paint_upload_rows"] - full, 4)    # rows 40 and 43 changed: 40..43
        again = self.context.read()
        self.assertEqual(tuple(again[40, 31, :3]), tuple(plain[40, 31, :3]))        # the old top row is clear
        self.assertEqual(tuple(again[43, 31, :3]), (255, 0, 0))
        self.presenter.set_paint(self.paint(lambda i: None), "c")
        self.presenter.set_paint(self.paint(self.mark(30, 40)), "d")        # empty, then back: a full upload
        self.presenter.draw(self.view, self.context.size)
        self.assertEqual(tuple(self.context.read()[41, 31, :3]), (255, 0, 0))

    def test_recovery_forgets_the_paint_so_it_is_painted_again(self):
        self.presenter.set_paint(self.paint(lambda i: None), "k")
        self.presenter.recover("test")
        self.assertIsNone(self.presenter.paint_key)
        self.assertIsNone(self.presenter.paint_size)


if __name__ == "__main__":
    unittest.main()
