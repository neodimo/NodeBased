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

APP = QApplication.instance() or QApplication([])


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

    def test_retired_resources_wait_for_enough_presents(self):
        ledger = dv.FrameLedger()
        ledger.retire("texture")
        for _ in range(dv.RETIRE_PRESENTS - 1):
            self.assertEqual(ledger.note_present(), [])
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


def make_presenter(size=(160, 120), adapter=None):
    import wgpu
    from nodebased import gpu3d
    adapters = gpu3d.rank_adapters(list(wgpu.gpu.enumerate_adapters_sync()))
    context = FakeContext(size)
    presenter = dv.RetainedPresenter(context, adapter=adapter or adapters[0])
    return presenter, context


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

    def test_size_change_retires_the_old_texture_until_presents_pass(self):
        self.draw((32.0, 24.0, 2.0), checker_image())
        old = self.presenter.texture
        self.draw((32.0, 24.0, 2.0), checker_image(width=32, height=24), rect=(0, 0, 32, 24), generation=2)
        self.assertIsNot(self.presenter.texture, old)
        self.assertEqual(len(self.presenter.ledger._retired), 1)
        for _ in range(dv.RETIRE_PRESENTS):
            self.draw((16.0, 12.0, 2.0))
        self.assertEqual(self.presenter.ledger._retired, [])

    def test_dropped_pending_pictures_are_never_converted(self):
        for shift in range(5):
            self.presenter.offer(checker_image(shift=shift), (0, 0, 64, 48), (0, 0, 64, 48), 1)
        self.presenter.draw((32.0, 24.0, 2.0), self.context.size)
        self.assertEqual(self.presenter.counters["conversions"], 1)
        self.assertEqual(self.presenter.ledger.stats["dropped_pending"], 4)

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


class AdapterChoiceTests(unittest.TestCase):
    def test_first_adapter_the_surface_accepts_wins(self):
        import wgpu
        adapters = list(wgpu.gpu.enumerate_adapters_sync())

        class Picky:
            def __init__(self, ok):
                self.ok = ok

            def get_preferred_format(self, adapter):
                if dict(adapter.info) != dict(self.ok.info):
                    raise RuntimeError("surface cannot present here")
                return "bgra8unorm"

        chosen = adapters[-1]
        adapter, fmt = dv.pick_display_adapter(Picky(chosen), preference="")
        self.assertEqual(dict(adapter.info), dict(chosen.info))
        self.assertEqual(fmt, "bgra8unorm")

    def test_no_presentable_adapter_is_an_error(self):
        class Never:
            def get_preferred_format(self, adapter):
                raise RuntimeError("no")

        with self.assertRaises(RuntimeError):
            dv.pick_display_adapter(Never(), preference="")


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
