import os
import unittest
import unittest.mock
import uuid

import numpy as np

import tests.isolation  # noqa: F401  (Qt settings stay out of the real user file)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from nodebased import directviewer as dv
from tests.waiting import wait_until

try:
    import wgpu  # noqa: F401
    HAVE_WGPU = True
except ImportError:   # CI machines without the card library (10/18 red Desktop conformance)
    HAVE_WGPU = False
needs_wgpu = unittest.skipUnless(HAVE_WGPU, "wgpu unavailable")

APP = QApplication.instance() or QApplication([])


def direct_dock_skip():
    """Why a real direct dock (a wgpu surface on a window) cannot open here, or "".

    rendercanvas needs a real window-system id: GitHub's Linux packaging job has wgpu but no display (10/19,
    v0.39.0), and offscreen Qt has no native windows on any OS. On Windows its made-up window id reached D3D12
    and wgpu aborted the process (10/21, v0.41.0 package job); the product now refuses it, so the dock fails."""
    from PySide6.QtGui import QGuiApplication
    platform = QGuiApplication.platformName().lower()
    if platform in dv.NO_NATIVE_WINDOW:
        return f"Qt platform {platform}: no native window for the direct surface"
    if os.name != "nt" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return "no display server: the direct dock needs a real window"
    return ""


def stub_surface():
    """For dock tests with a stub presenter: the widget asks for no window-system surface at all, so they run on
    offscreen Qt and without a display server."""
    import contextlib
    from rendercanvas.qt import QRenderWidget
    stack = contextlib.ExitStack()
    stack.enter_context(unittest.mock.patch.object(dv, "surface_problem", return_value=("", False)))
    stack.enter_context(unittest.mock.patch.object(dv, "native_window_alive", return_value=True))
    stack.enter_context(unittest.mock.patch.object(QRenderWidget, "get_wgpu_context", return_value=None))
    return stack


def checker_image(width=64, height=48, cell=8, shift=0):
    y, x = np.indices((height, width))
    rgb = np.zeros((height, width, 3), np.uint8)
    on = ((x // cell + y // cell + shift) % 2).astype(bool)
    rgb[on] = (200, 120, 40)
    rgb[~on] = (20, 60, 180)
    return QImage(rgb.data, width, height, width * 3, QImage.Format.Format_RGB888).copy()


class FrameLedgerTests(unittest.TestCase):
    def test_only_the_newest_ready_picture_survives(self):
        ledger = dv.FrameLedger()
        self.assertTrue(ledger.offer("a", 1, 0))
        self.assertTrue(ledger.offer("b", 1, 0))
        sequence, frame = ledger.take()
        self.assertEqual(frame, "b")
        self.assertEqual(ledger.stats["dropped_pending"], 1)
        self.assertIsNone(ledger.take())

    def test_stale_generation_and_old_epoch_are_rejected(self):
        ledger = dv.FrameLedger()
        self.assertTrue(ledger.offer("new", 5, 0))
        self.assertFalse(ledger.offer("old", 4, 0))
        self.assertEqual(ledger.stats["rejected_generation"], 1)
        self.assertEqual(ledger.take()[1], "new")
        ledger.new_epoch()
        self.assertFalse(ledger.offer("from the lost device", 6, 0))
        self.assertEqual(ledger.stats["rejected_epoch"], 1)
        self.assertTrue(ledger.offer("after", 1, 1))   # generations restart with the epoch

    def test_equal_generation_is_accepted_for_playback(self):
        ledger = dv.FrameLedger()
        self.assertTrue(ledger.offer("frame 1", 3, 0))
        self.assertTrue(ledger.offer("frame 2", 3, 0))

    def test_retired_resources_are_released_after_the_next_submit(self):
        ledger = dv.FrameLedger()
        ledger.retire("texture")
        self.assertEqual(ledger.note_present(), ["texture"])
        self.assertEqual(ledger.note_present(), [])

    def test_new_epoch_returns_everything_still_held(self):
        ledger = dv.FrameLedger()
        ledger.retire("a")
        ledger.retire("b")
        self.assertEqual(sorted(ledger.new_epoch()), ["a", "b"])
        self.assertEqual(ledger.epoch, 1)


class ViewMathTests(unittest.TestCase):
    def test_fit_shows_the_whole_rectangle_centred(self):
        cx, cy, zoom = dv.fit_view((800, 400), (0, 0, 1600, 400))
        self.assertAlmostEqual(zoom, 0.5)
        self.assertAlmostEqual(cx, 800)
        self.assertAlmostEqual(cy, 200)

    def test_zoom_keeps_the_point_under_the_cursor(self):
        view = (500.0, 300.0, 1.0)
        size = (800, 600)
        cursor = (650, 100)
        before = (view[0] + (cursor[0] - 400) / view[2], view[1] + (cursor[1] - 300) / view[2])
        after_view = dv.zoom_about(view, cursor, size, 2.0)
        after = (after_view[0] + (cursor[0] - 400) / after_view[2], after_view[1] + (cursor[1] - 300) / after_view[2])
        self.assertAlmostEqual(before[0], after[0])
        self.assertAlmostEqual(before[1], after[1])
        self.assertAlmostEqual(after_view[2], 2.0)


class FakeContext:
    """Stands in for a screen surface: an offscreen texture we can read back after every draw."""

    def __init__(self, size, fmt="rgba8unorm"):
        self.size, self.fmt, self.device, self.texture, self.acquired = size, fmt, None, None, 0

    def get_preferred_format(self, adapter):
        return self.fmt

    def configure(self, device, format, alpha_mode):
        import wgpu
        self.device = device
        self.texture = device.create_texture(
            size=(*self.size, 1), format=format,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)

    def get_current_texture(self):
        self.acquired += 1
        return self.texture

    def read(self):
        data = self.device.queue.read_texture({"texture": self.texture}, {"bytes_per_row": self.size[0] * 4},
                                              (*self.size, 1))
        return np.frombuffer(data, np.uint8).reshape(self.size[1], self.size[0], 4)


def make_presenter(size=(160, 120), adapter=None, fill_thread=False):
    import wgpu
    from nodebased import gpu3d
    adapters = gpu3d.rank_adapters(list(wgpu.gpu.enumerate_adapters_sync()))
    kind = {"discrete": "discretegpu", "integrated": "integratedgpu", "cpu": "cpu"}.get(os.environ.get("NB_TEST_ADAPTER", ""))
    if kind:   # run the pixel tests on one named adapter: NB_TEST_ADAPTER=integrated|cpu|discrete
        adapters = [a for a in adapters if gpu3d._kind_of(a.info) == kind] or adapters
    context = FakeContext(size)
    # The pixel tests fill the staging buffer inline so a draw right after an offer shows it;
    # StagingRingTests run the filler thread.
    presenter = dv.RetainedPresenter(context, adapter=adapter or adapters[0], fill_thread=fill_thread)
    return presenter, context


@needs_wgpu
class RetainedPresenterTests(unittest.TestCase):
    def setUp(self):
        self.presenter, self.context = make_presenter()
        self.addCleanup(self.presenter.lose_device)

    def draw(self, view, image=None, rect=(0, 0, 64, 48), generation=1):
        if image is not None:
            self.assertTrue(self.presenter.offer(image, rect, rect, generation))
        self.presenter.draw(view, self.context.size)
        return self.context.read()

    def test_picture_lands_where_the_view_says(self):
        image = checker_image()
        pixels = self.draw((32.0, 24.0, 2.0), image)
        # Scene (0,0)..(64,48) at zoom 2 centred on (32,24): 128x96 physical px centred in 160x120.
        # Physical (20,12) is scene (0,0): the top-left cell is on-colour (shift 0 -> cell parity 0 -> blue).
        self.assertEqual(tuple(pixels[12 + 1, 20 + 1, :3]), (20, 60, 180))
        self.assertEqual(tuple(pixels[12 + 1, 20 + 17, :3]), (200, 120, 40))   # next cell, 8 scene px = 16 physical
        self.assertEqual(tuple(pixels[60, 5, :3]), (0, 0, 0))                  # surround

    def test_pan_and_zoom_do_not_convert_or_upload_again(self):
        image = checker_image()
        self.draw((32.0, 24.0, 2.0), image)
        for step in range(10):
            self.draw((32.0 + step, 24.0 - step, 2.0 + step * 0.1))
        self.assertEqual(self.presenter.counters["conversions"], 1)
        self.assertEqual(self.presenter.counters["uploads"], 1)
        self.assertEqual(self.presenter.counters["draws"], 11)
        self.assertEqual(self.context.acquired, 11)

    def test_pan_moves_the_picture(self):
        image = checker_image()
        base = self.draw((32.0, 24.0, 2.0), image).copy()
        moved = self.draw((32.0 - 8.0, 24.0, 2.0))   # view centre left by 8 scene px = picture right by 16 px
        self.assertEqual(tuple(moved[13, 20 + 16 + 1, :3]), tuple(base[13, 20 + 1, :3]))

    def test_new_picture_same_size_reuses_the_texture(self):
        self.draw((32.0, 24.0, 2.0), checker_image(shift=0))
        first = self.presenter.texture
        pixels = self.draw((32.0, 24.0, 2.0), checker_image(shift=1))
        self.assertIs(self.presenter.texture, first)
        self.assertEqual(self.presenter.counters["texture_creates"], 1)
        self.assertEqual(self.presenter.counters["uploads"], 2)
        self.assertEqual(tuple(pixels[13, 21, :3]), (200, 120, 40))   # parity flipped

    def test_a_smaller_picture_reuses_the_texture_and_a_larger_one_grows_it(self):
        self.draw((32.0, 24.0, 2.0), checker_image())
        old = self.presenter.texture
        pixels = self.draw((16.0, 12.0, 4.0), checker_image(width=32, height=24, shift=1), rect=(0, 0, 32, 24), generation=2)
        self.assertIs(self.presenter.texture, old)                    # the crop fits: no new texture
        self.assertEqual(self.presenter.texture_size, (32, 24))
        self.assertEqual(tuple(pixels[13, 21, :3]), (200, 120, 40))   # and the shader reads only the crop
        self.assertEqual(tuple(pixels[13, 142, :3]), (20, 60, 180))  # last column of the crop, not the old picture
        destroyed = []
        original = self.presenter._destroy
        self.presenter._destroy = lambda resource: (destroyed.append(resource), original(resource))
        self.draw((48.0, 36.0, 1.0), checker_image(width=96, height=72), rect=(0, 0, 96, 72), generation=3)
        self.assertIsNot(self.presenter.texture, old)
        self.assertEqual(self.presenter.texture_capacity, (96, 72))
        self.assertEqual(destroyed, [old])
        self.assertEqual(self.presenter.ledger._retired, [])
        self.assertEqual(self.presenter.counters["texture_creates"], 2)

    def test_resize_and_close_while_work_is_queued(self):
        # Many submits in flight, then a size change mid-stream: the picture after it is right
        # (no use of the destroyed texture), and closing waits for the queue rather than racing it.
        for shift in range(12):
            self.presenter.offer(checker_image(shift=shift), (0, 0, 64, 48), (0, 0, 64, 48), 1)
            self.presenter.draw((32.0, 24.0, 2.0), self.context.size)
        pixels = self.draw((16.0, 12.0, 4.0), checker_image(width=32, height=24, shift=0), rect=(0, 0, 32, 24))
        self.assertEqual(tuple(pixels[13, 21, :3]), (20, 60, 180))
        for _ in range(12):
            self.presenter.draw((16.0, 12.0, 4.0), self.context.size)
        self.presenter.lose_device()
        self.assertEqual(self.presenter.counters["idle_waits"], 1)
        self.assertIsNone(self.presenter.device)

    def test_dropped_pending_pictures_are_never_converted(self):
        # The first picture goes into a staging slot at once; while it waits for a draw, newer offers
        # replace each other in the pending slot and only the newest of them is ever converted.
        for shift in range(5):
            self.presenter.offer(checker_image(shift=shift), (0, 0, 64, 48), (0, 0, 64, 48), 1)
        self.presenter.draw((32.0, 24.0, 2.0), self.context.size)
        pixels = self.draw((32.0, 24.0, 2.0))
        self.assertEqual(self.presenter.counters["conversions"], 2)
        self.assertEqual(self.presenter.counters["uploads"], 2)
        self.assertEqual(self.presenter.ledger.stats["dropped_pending"], 3)
        self.assertEqual(tuple(pixels[13, 21, :3]), (20, 60, 180))   # shift 4, the newest

    def test_overlay_marks_the_corner_and_the_format_edge(self):
        pixels = self.draw((32.0, 24.0, 2.0), checker_image())
        r, g, b = (int(v) for v in pixels[2, 2, :3])
        self.assertGreater(r, 200)
        self.assertLess(b, 60)                      # orange experimental mark
        edge = pixels[12, 80, :3].astype(int)       # top edge of the format outline at physical y=12
        self.assertGreater(int(edge[0]) + int(edge[1]), 300)

    def test_lost_device_drops_pending_and_rejects_old_epoch_offers(self):
        self.presenter.offer(checker_image(), (0, 0, 64, 48), (0, 0, 64, 48), 1)
        epoch = self.presenter.ledger.epoch
        self.presenter.lose_device()
        self.assertIsNone(self.presenter.ledger.pending)
        self.assertEqual(self.presenter.ledger.epoch, epoch + 1)
        self.assertFalse(self.presenter.ledger.offer("late", 9, epoch))


@needs_wgpu
class DeviceLossTests(unittest.TestCase):
    """A real loss: the device is destroyed under the presenter (wgpu reports 'Parent device is lost')."""

    def setUp(self):
        self.presenter, self.context = make_presenter()
        self.addCleanup(self.presenter.lose_device)

    def draw(self, view=(32.0, 24.0, 2.0), image=None, generation=1):
        if image is not None:
            self.assertTrue(self.presenter.offer(image, (0, 0, 64, 48), (0, 0, 64, 48), generation))
        self.presenter.draw(view, self.context.size)
        return self.context.read()

    def test_destroyed_device_makes_the_next_draw_raise(self):
        self.draw(image=checker_image())
        self.presenter.device.destroy()
        with self.assertRaises(Exception) as caught:
            self.draw()
        self.assertIn("lost", str(caught.exception).lower())

    def test_recover_rebuilds_the_picture_from_the_retained_cpu_image(self):
        before = self.draw(image=checker_image()).copy()
        old_device, old_texture = self.presenter.device, self.presenter.texture
        epoch = self.presenter.ledger.epoch
        old_device.destroy()
        self.presenter.recover("test loss")
        self.assertIsNot(self.presenter.device, old_device)
        self.assertEqual(self.presenter.ledger.epoch, epoch + 1)
        self.assertIsNone(self.presenter.texture)
        self.assertEqual(self.presenter.counters["recoveries"], 1)
        after = self.draw()                                   # no new offer: the retained picture comes back
        self.assertTrue((before == after).all())
        self.assertEqual(self.presenter.counters["uploads"], 2)
        self.assertEqual(self.presenter.counters["conversions"], 2)
        self.assertIsNot(self.presenter.texture, old_texture)

    def test_offer_from_the_lost_epoch_is_refused_and_a_newer_pending_picture_survives(self):
        self.draw(image=checker_image(shift=0))
        epoch = self.presenter.ledger.epoch
        self.presenter.offer(checker_image(shift=1), (0, 0, 64, 48), (0, 0, 64, 48), 2)   # waiting for the next draw
        self.presenter.device.destroy()
        self.presenter.recover("test loss")
        self.assertFalse(self.presenter.ledger.offer("from the lost device", 9, epoch))
        pixels = self.draw()
        self.assertEqual(tuple(pixels[13, 21, :3]), (200, 120, 40))   # shift 1, not the older shift 0 picture

    def test_a_logged_device_loss_is_noticed_at_the_top_of_the_next_draw(self):
        import logging
        self.draw(image=checker_image())
        old_device = self.presenter.device
        logging.getLogger("wgpu").error("The WGPU device was lost (Unknown):\nsimulated by the test")
        self.assertTrue(self.presenter.device_lost())
        self.draw()
        self.assertIsNot(self.presenter.device, old_device)
        self.assertFalse(self.presenter.device_lost())
        self.assertEqual(self.presenter.counters["recoveries"], 1)

    def test_recoveries_that_come_too_fast_give_up(self):
        self.draw(image=checker_image())
        limit = dv.RECOVERY_LIMIT[0]
        for _ in range(limit):
            self.presenter.recover("again")
        with self.assertRaises(RuntimeError):
            self.presenter.recover("once too often")


@needs_wgpu
class WidgetDeviceLossTests(unittest.TestCase):
    """The Qt surface repeats a draw once on a replaced device and gives up (dock hides) when that fails too."""

    def test_widget_recovers_once_from_a_destroyed_device(self):
        widget = dv.make_widget()
        self.addCleanup(widget.close)
        presenter, context = make_presenter(size=(64, 48))
        self.addCleanup(presenter.lose_device)
        widget.presenter = presenter
        widget.rect = widget.format_rect = (0, 0, 64, 48)
        widget._physical = lambda: (64, 48)
        self.assertTrue(presenter.offer(checker_image(), widget.rect, widget.rect, 1))
        widget._draw()
        presenter.device.destroy()
        widget._draw()
        self.assertIsNone(widget.error)
        self.assertEqual(widget.recoveries, 1)
        self.assertEqual(presenter.counters["recoveries"], 1)

    def test_widget_fails_over_when_recovery_cannot_help(self):
        widget = dv.make_widget()
        self.addCleanup(widget.close)
        presenter, context = make_presenter(size=(64, 48))
        widget.presenter = presenter
        widget.rect = widget.format_rect = (0, 0, 64, 48)
        widget._physical = lambda: (64, 48)
        failures = []
        widget.failed.connect(failures.append)
        presenter.offer(checker_image(), widget.rect, widget.rect, 1)
        with unittest.mock.patch.object(presenter, "recover", side_effect=RuntimeError("gone for good")), \
                unittest.mock.patch.object(presenter, "draw", side_effect=RuntimeError("Parent device is lost")):
            widget._draw()
        self.assertIsNotNone(widget.error)
        self.assertTrue(failures and "draw failed" in failures[0])
        self.assertIsNone(widget.presenter)


class PlatformGuardTests(unittest.TestCase):
    def test_native_wayland_is_refused_before_any_surface_is_made(self):
        self.assertIn("QT_QPA_PLATFORM=xcb", dv.unsupported_platform("wayland"))
        self.assertIn("QT_QPA_PLATFORM=xcb", dv.unsupported_platform("wayland-egl"))
        for name in ("xcb", "offscreen", "windows", "cocoa"):
            self.assertEqual(dv.unsupported_platform(name), "")

    def test_make_widget_raises_on_wayland_and_the_window_switch_falls_back(self):
        with unittest.mock.patch.object(dv, "unsupported_platform", return_value="native Wayland is not supported"):
            with self.assertRaises(RuntimeError):
                dv.make_widget()


class SurfaceGuardTests(unittest.TestCase):
    """wgpu-native aborts the process on a surface it cannot configure, so the widget checks the window first."""

    def test_platforms_without_native_windows_are_refused_for_good(self):
        for name in ("offscreen", "minimal", "Offscreen"):
            reason, permanent = dv.surface_problem(name, 7, (640, 360))
            self.assertIn("no native window", reason)
            self.assertTrue(permanent)

    def test_a_window_that_is_not_ready_waits(self):
        self.assertEqual(dv.surface_problem("windows", 0, (640, 360))[1], False)
        self.assertIn("no native id", dv.surface_problem("xcb", 0, (640, 360))[0])
        self.assertIn("no area", dv.surface_problem("windows", 1234, (0, 360))[0])
        self.assertIn("no area", dv.surface_problem("cocoa", 1234, (640, 0))[0])
        reason, permanent = dv.surface_problem("windows", 1234, (640, 360), window_alive=lambda wid: False)
        self.assertIn("not a live window", reason)
        self.assertFalse(permanent)

    def test_a_live_window_with_area_may_have_a_surface(self):
        self.assertEqual(dv.surface_problem("windows", 1234, (640, 360), window_alive=lambda wid: True), ("", False))
        self.assertEqual(dv.surface_problem("xcb", 1234, (1, 1)), ("", False))

    @unittest.skipUnless(os.name == "nt", "Windows window handles")
    def test_a_made_up_window_handle_is_not_alive(self):
        self.assertFalse(dv.native_window_alive(7))

    @needs_wgpu
    def test_offscreen_qt_fails_the_dock_before_any_surface_is_made(self):
        widget = dv.make_widget()
        self.addCleanup(widget.close)
        failures = []
        widget.failed.connect(failures.append)
        made = unittest.mock.Mock(side_effect=AssertionError("no surface may be made"))
        with unittest.mock.patch("PySide6.QtGui.QGuiApplication.platformName", return_value="offscreen"), \
                unittest.mock.patch.object(widget, "get_wgpu_context", made), \
                unittest.mock.patch.object(dv, "RetainedPresenter", made):
            self.assertFalse(widget.submit(checker_image(), (0, 0, 64, 48)))
        made.assert_not_called()
        self.assertIsNone(widget.presenter)
        self.assertTrue(failures and "offscreen" in failures[0])

    @needs_wgpu
    def test_a_window_without_area_waits_and_starts_once_it_has_one(self):
        widget = dv.make_widget()
        self.addCleanup(widget.close)
        problem = ["the window has no area (minimised or hidden)", False]
        with unittest.mock.patch.object(dv, "surface_problem", side_effect=lambda *a: tuple(problem)), \
                unittest.mock.patch.object(dv, "native_window_alive", return_value=True), \
                unittest.mock.patch.object(widget, "get_wgpu_context", return_value=None), \
                unittest.mock.patch.object(dv, "RetainedPresenter", StubPresenter):
            self.assertFalse(widget.submit(checker_image(), (0, 0, 64, 48), generation=3))
            self.assertIsNone(widget.presenter)
            self.assertIsNone(widget.error)
            self.assertIn("no area", widget.waiting)
            problem[:] = ["", False]
            widget.resize(200, 120)
            widget.show()
            self.assertTrue(wait_until(lambda: widget.presenter is not None))
        self.assertEqual(widget.presenter.offers, [3])
        self.assertEqual(widget.waiting, "")
        self.assertEqual(widget.rect, (0, 0, 64, 48))


class StubPresenter:
    """Takes pictures and draws nothing: these tests are about which surface gets which input."""

    def __init__(self, context, adapter=None, preference=None, **kwargs):
        self.offers = []
        self.ledger = dv.FrameLedger()
        self.counters = {}
        self.paint_size = self.paint_key = None

    def set_paint(self, image, key):
        self.paint_key = key

    def offer(self, qimage, rect, format_rect, generation=None):
        self.offers.append(generation)
        return True

    def draw(self, view, size):
        pass

    def lose_device(self):
        pass

    def retained_bytes(self):
        return 0


@needs_wgpu
class HitOrderTests(unittest.TestCase):
    """While the direct dock is open the Viewer keeps every overlay and every hit; the direct surface only navigates."""

    def setUp(self):
        from tests.test_roto_ui import roto_document
        from nodebased.app import Window
        from PySide6.QtCore import QSettings
        QSettings("NodeBased", "NodeBased").clear()
        self.window = Window(roto_document(), agent_name="nodebased-test-" + uuid.uuid4().hex)
        # A skip below leaves setUp early and tearDown never runs; close the window then too.
        self.addCleanup(self.close_if_skipped)
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))
        self.window.graph.items_by_id["r"].setSelected(True)
        APP.processEvents()
        patcher = unittest.mock.patch.object(dv, "RetainedPresenter", StubPresenter)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.enterContext(stub_surface())
        self.window.direct_action.setChecked(True)
        APP.processEvents()
        self.assertIsNotNone(self.window.direct_widget)

    def tearDown(self):
        from tests.test_desktop import release_window
        self.addCleanup(release_window, self)
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def close_if_skipped(self):
        window = self.__dict__.get("window")
        if window is not None and window.isVisible():
            window.saved_document = window.dispatcher.document
            window.close()
            APP.processEvents()

    def test_roto_click_order_is_unchanged_and_the_direct_surface_never_takes_a_roto_click(self):
        from PySide6.QtCore import QPoint, QPointF, Qt
        from PySide6.QtTest import QTest
        viewer, widget = self.window.viewer, self.window.direct_widget
        self.assertTrue(viewer.begin_roto_draw("r"))
        transform = viewer.transform()
        QTest.mouseClick(viewer.viewport(), Qt.MouseButton.LeftButton, pos=viewer.mapFromScene(QPointF(30, 30)))
        self.assertEqual(len(viewer.roto_draw_points), 1)
        # A plain click on the direct surface is swallowed there: no Roto point, no pan, no view change.
        view = widget.view
        QTest.mouseClick(widget, Qt.MouseButton.LeftButton, pos=QPoint(widget.width() // 2, widget.height() // 2))
        self.assertEqual(len(viewer.roto_draw_points), 1)
        self.assertEqual(widget.view, view)
        self.assertTrue(widget.auto_fit)
        # Wheel and Alt-drag on the direct surface move only the direct surface.
        widget.rect = (0, 0, 100, 100)
        QTest.mousePress(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.AltModifier, QPoint(10, 10))
        QTest.mouseMove(widget, QPoint(30, 25))
        QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.AltModifier, QPoint(30, 25))
        self.assertFalse(widget.auto_fit)
        self.assertEqual(viewer.transform(), transform)
        self.assertEqual(len(viewer.roto_draw_points), 1)
        # The Viewer still takes the next Roto point, in the same place in the order.
        QTest.mouseClick(viewer.viewport(), Qt.MouseButton.LeftButton, pos=viewer.mapFromScene(QPointF(70, 30)))
        self.assertEqual(len(viewer.roto_draw_points), 2)

    def test_viewer_wipe_and_roi_state_are_not_read_or_changed_by_the_direct_surface(self):
        viewer, widget = self.window.viewer, self.window.direct_widget
        wipe, mode = dict(self.window.viewer.wipe), viewer.compare_mode()
        widget.fit()
        APP.processEvents()
        self.assertEqual(viewer.wipe, wipe)
        self.assertEqual(viewer.compare_mode(), mode)
        self.assertIn("view only", self.window.direct_dock.windowTitle())


@needs_wgpu
class SrgbSurfaceTests(unittest.TestCase):
    def test_srgb_surface_shows_the_same_bytes_the_cpu_image_holds(self):
        import wgpu
        from nodebased import gpu3d
        adapter = gpu3d.rank_adapters(list(wgpu.gpu.enumerate_adapters_sync()))[0]
        context = FakeContext((160, 120), fmt="rgba8unorm-srgb")
        presenter = dv.RetainedPresenter(context, adapter=adapter, fill_thread=False)
        self.addCleanup(presenter.lose_device)
        presenter.offer(checker_image(), (0, 0, 64, 48), (0, 0, 64, 48), 1)
        presenter.draw((32.0, 24.0, 2.0), (160, 120))
        # The surface re-encodes on write, so the stored bytes equal the display bytes the CPU image holds.
        got = context.read()[13, 21, :3].astype(int)
        for have, want in zip(got, (20, 60, 180)):
            self.assertLessEqual(abs(int(have) - want), 1)


@needs_wgpu
class AdapterChoiceTests(unittest.TestCase):
    """The display adapter comes from wgpu's own surface filter, never from get_preferred_format alone."""

    class Adapter:
        def __init__(self, name, kind):
            self.info = {"device": name, "adapter_type": kind}

    class Context:
        def get_preferred_format(self, adapter):
            return "bgra8unorm"

    def request(self, answers):
        calls = []

        def fake(canvas=None, **kwargs):
            calls.append(kwargs)
            answer = answers.get(tuple(sorted(kwargs.items())), RuntimeError("not offered"))
            if isinstance(answer, Exception):
                raise answer
            return answer
        return fake, calls

    def test_request_is_filtered_by_the_surface_and_follows_the_wish(self):
        import wgpu
        amd, nv = self.Adapter("amd", "IntegratedGPU"), self.Adapter("nv", "DiscreteGPU")
        fake, calls = self.request({(("power_preference", "low-power"),): amd,
                                    (("power_preference", "high-performance"),): nv})
        with unittest.mock.patch.object(wgpu.gpu, "request_adapter_sync", fake):
            # backend "auto": on Windows the default asks D3D12 first, and the note would then name the backend
            # (the 0.41.0 Windows package job, 10/21). This test is about the power preference only.
            adapter, fmt, note = dv.pick_display_adapter(self.Context(), preference="integrated", backend="auto")
        self.assertIs(adapter, amd)
        self.assertEqual((fmt, note), ("bgra8unorm", ""))
        self.assertEqual(calls, [{"power_preference": "low-power"}])

    def test_wish_that_the_window_cannot_satisfy_says_so(self):
        import wgpu
        nv = self.Adapter("nv", "DiscreteGPU")
        fake, _ = self.request({(("power_preference", "low-power"),): RuntimeError("no surface support"),
                                (("power_preference", "high-performance"),): nv})
        with unittest.mock.patch.object(wgpu.gpu, "request_adapter_sync", fake):
            adapter, _, note = dv.pick_display_adapter(self.Context(), preference="integrated")
        self.assertIs(adapter, nv)
        self.assertIn("integrated was asked for", note)
        self.assertNotIn("only", note)    # what was selected, never a claim about adapters not asked for

    def test_windows_asks_d3d12_first_and_vulkan_is_the_switch(self):
        self.assertEqual(dv.backend_order("", platform="win32"), ["D3D12", None])
        self.assertEqual(dv.backend_order("vulkan", platform="win32"), ["Vulkan", None])
        self.assertEqual(dv.backend_order("", platform="linux"), [None])
        self.assertEqual(dv.backend_order("auto", platform="win32"), [None])

    def test_selection_records_the_backend_asked_for_and_what_failed(self):
        import wgpu
        nv = self.Adapter("nv", "DiscreteGPU")
        nv.info["backend_type"] = "Vulkan"
        seen = []

        def fake(canvas=None, **kwargs):
            backend = os.environ.get("WGPU_BACKEND_TYPE")
            seen.append(backend)
            if backend == "D3D12":
                raise RuntimeError("no D3D12 here")
            return nv
        with unittest.mock.patch.object(wgpu.gpu, "request_adapter_sync", fake):
            adapter, selection = dv.choose_display_adapter(self.Context(), preference="", backend="d3d12")
        self.assertIs(adapter, nv)
        self.assertEqual(seen[:4], ["D3D12"] * 4)
        self.assertIsNone(seen[4])
        self.assertNotIn("WGPU_BACKEND_TYPE", os.environ)
        self.assertEqual(selection["requested_backends"], ["D3D12", "auto"])
        self.assertEqual((selection["selected_device"], selection["selected_backend"]), ("nv", "Vulkan"))
        self.assertIn("D3D12 was asked for; selected Vulkan", selection["note"])
        self.assertEqual(sum("error" in a for a in selection["attempts"]), 4)

    def test_no_presentable_adapter_is_an_error(self):
        import wgpu
        fake, _ = self.request({})
        with unittest.mock.patch.object(wgpu.gpu, "request_adapter_sync", lambda canvas=None, **k: (_ for _ in ()).throw(RuntimeError("no"))):
            with self.assertRaises(RuntimeError):
                dv.pick_display_adapter(self.Context(), preference="")


class WindowSwitchTests(unittest.TestCase):
    def setUp(self):
        from nodebased.app import Window
        self.window = Window(agent_name="nodebased-test-" + uuid.uuid4().hex)
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))

    def tearDown(self):
        from tests.test_desktop import release_window
        self.addCleanup(release_window, self)
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def test_default_is_off_and_the_viewer_paints(self):
        w = self.window
        self.assertFalse(w.direct_action.isChecked())
        self.assertIsNone(w.direct_dock)
        self.assertIsNotNone(w.viewer.last_display_image)
        self.assertEqual(w.viewer_mode(), "2d")

    def test_wayland_refusal_leaves_the_default_viewer_in_charge(self):
        w = self.window
        with unittest.mock.patch.object(dv, "unsupported_platform", return_value="native Wayland is not supported"):
            w.direct_action.setChecked(True)
        self.assertFalse(w.direct_action.isChecked())
        self.assertIsNone(w.direct_dock)
        self.assertIn("Wayland", w.statusBar().currentMessage())

    def test_switch_failure_leaves_the_default_viewer_in_charge(self):
        w = self.window
        with unittest.mock.patch.object(dv, "make_widget", side_effect=RuntimeError("no surface")):
            w.direct_action.setChecked(True)
        self.assertFalse(w.direct_action.isChecked())
        self.assertIsNone(w.direct_dock)
        self.assertIn("unavailable", w.statusBar().currentMessage())
        self.assertIsNotNone(w.viewer.last_display_image)


if __name__ == "__main__":
    unittest.main()
