"""VP2a: the direct picture during playback (staging ring, filler thread, 4K pictures, too-large refusal,
retained-picture memory). The pixel tests share tests/test_directviewer.py's offscreen target."""
import os
import threading
import types
import unittest
import unittest.mock
import uuid

import numpy as np

from nodebased import directviewer as dv
from tests.test_directviewer import (APP, FakeContext, StubPresenter, checker_image, make_presenter, needs_wgpu,
                                     stub_surface)
from tests.waiting import wait_until

class PlainRulesTests(unittest.TestCase):
    """No card needed: row padding, the size rule and the copy into a mapped buffer."""

    def test_rows_are_padded_to_the_copy_alignment(self):
        self.assertEqual(dv.padded_row(64, 4), 256)
        self.assertEqual(dv.padded_row(33, 4), 256)
        self.assertEqual(dv.padded_row(3840, 16), 61440)        # 4K float rows need no padding
        self.assertEqual(dv.padded_row(1001, 16), 16128)

    def test_a_picture_wider_than_the_card_is_refused_by_name(self):
        limits = {"max-texture-dimension-2d": 8192, "max-buffer-size": 1 << 30}
        self.assertEqual(dv.too_large(limits, 3840, 2160, 16, "RTX"), "")
        reason = dv.too_large(limits, 10000, 2000, 16, "RTX")
        self.assertIn("10000 x 2000", reason)
        self.assertIn("RTX takes at most 8192 pixels on a side", reason)
        self.assertIn("the Viewer shows it", reason)
        self.assertEqual(dv.too_large({}, 99999, 99999, 16), "")   # no presenter yet: nothing is refused

    def test_a_picture_larger_than_one_upload_buffer_is_refused_by_name(self):
        limits = {"max-texture-dimension-2d": 16384, "max-buffer-size": 256 << 20}
        self.assertEqual(dv.too_large(limits, 3840, 2160, 16), "")                       # 127 MB
        reason = dv.too_large(limits, 7680, 4320, 16)                                    # 506 MB
        self.assertIn("7680 x 4320", reason)
        self.assertIn("MB at once", reason)

    def test_fill_copies_a_cropped_float_frame_into_padded_rows(self):
        full = np.random.default_rng(1).random((40, 50, 4)).astype(np.float32)
        crop = full[5:25, 7:40]                              # a non-contiguous view, like a cache crop
        width, height, fmt, texel = dv.picture_layout("linear", crop)
        self.assertEqual((width, height, fmt, texel), (33, 20, "rgba32float", 16))
        row = dv.padded_row(width, texel)
        memory = bytearray(row * height)
        converted = dv.fill_rows(memoryview(memory), "linear", crop, width, height, row)
        self.assertFalse(converted)
        got = np.frombuffer(memory, np.float32).reshape(height, row // 4)[:, :width * 4].reshape(height, width, 4)
        np.testing.assert_array_equal(got, crop)

    def test_4k_float_frames_go_up_as_half_floats_and_1080p_stays_float(self):
        self.assertEqual(dv.picture_layout("linear", np.zeros((2160, 3840, 4), np.float32))[2:], ("rgba16float", 8))
        self.assertEqual(dv.picture_layout("linear", np.zeros((1080, 1920, 4), np.float32))[2:], ("rgba32float", 16))

    def test_a_large_frame_is_cast_in_bands_exactly_like_one_cast(self):
        full = (np.random.default_rng(2).random((1300, 1100, 4)).astype(np.float32) - 0.2) * 300.0
        crop = full[3:1203, 5:1005]                          # 1000 x 1200: over the band threshold
        width, height = 1000, 1200
        row = dv.padded_row(width, 8)
        memory = bytearray(row * height)
        self.assertTrue(dv.fill_rows(memoryview(memory), "linear", crop, width, height, row, "rgba16float"))
        got = np.frombuffer(memory, np.float16).reshape(height, row // 2)[:, :width * 4].reshape(height, width, 4)
        np.testing.assert_array_equal(got, crop.astype(np.float16))
        self.assertEqual(sum(b - a for a, b in dv._bands(height)), height)

    def test_fill_keeps_half_floats_as_they_are(self):
        frame = (np.arange(6 * 4 * 4, dtype=np.float32).reshape(6, 4, 4) / 7).astype(np.float16)
        width, height, fmt, texel = dv.picture_layout("linear", frame)
        self.assertEqual((fmt, texel), ("rgba16float", 8))
        row = dv.padded_row(width, texel)
        memory = bytearray(row * height)
        dv.fill_rows(memoryview(memory), "linear", frame, width, height, row)
        got = np.frombuffer(memory, np.float16).reshape(height, row // 2)[:, :width * 4].reshape(height, width, 4)
        np.testing.assert_array_equal(got, frame)


@needs_wgpu
class StagingRingTests(unittest.TestCase):
    """The filler thread copies pictures into reused buffers; the draw only records the GPU copy."""

    def setUp(self):
        self.presenter, self.context = make_presenter(fill_thread=True)
        self.addCleanup(self.presenter.lose_device)

    def settle_and_draw(self, view=(32.0, 24.0, 2.0)):
        self.presenter.ring.wait()
        self.presenter.draw(view, self.context.size)
        return self.context.read()

    def test_the_copy_into_the_staging_buffer_runs_off_the_qt_thread(self):
        threads = []
        original = dv.fill_rows

        def noting(*args):
            threads.append(threading.get_ident())
            return original(*args)
        with unittest.mock.patch.object(dv, "fill_rows", noting):
            self.presenter.offer(checker_image(), (0, 0, 64, 48), (0, 0, 64, 48), 1)
            pixels = self.settle_and_draw()
        self.assertEqual(len(threads), 1)
        self.assertNotEqual(threads[0], threading.get_ident())
        self.assertEqual(tuple(pixels[13, 21, :3]), (20, 60, 180))
        self.assertEqual(self.presenter.counters["fills"], 1)

    def test_buffers_and_texture_are_reused_across_a_played_sequence(self):
        for shift in range(12):
            self.presenter.offer(checker_image(shift=shift), (0, 0, 64, 48), (0, 0, 64, 48), 1)
            pixels = self.settle_and_draw()
            want = (20, 60, 180) if shift % 2 == 0 else (200, 120, 40)
            self.assertEqual(tuple(pixels[13, 21, :3]), want, f"picture {shift}")
        self.assertEqual(self.presenter.counters["uploads"], 12)
        self.assertLessEqual(self.presenter.counters["buffer_creates"], dv.STAGING_SLOTS)
        self.assertEqual(self.presenter.counters["texture_creates"], 1)

    def test_offers_during_a_fill_keep_only_the_newest(self):
        gate = threading.Event()
        original = dv.fill_rows

        def held(*args):
            gate.wait(5)
            return original(*args)
        with unittest.mock.patch.object(dv, "fill_rows", held):
            for shift in range(6):
                self.presenter.offer(checker_image(shift=shift), (0, 0, 64, 48), (0, 0, 64, 48), 1)
            self.assertTrue(self.presenter.ring.waiting())     # the first picture is still being filled
            gate.set()
            self.settle_and_draw()                             # picture 0; picture 5 goes to the filler
            pixels = self.settle_and_draw()
        self.assertEqual(self.presenter.counters["fills"], 2)
        self.assertEqual(self.presenter.ledger.stats["dropped_pending"], 4)
        self.assertEqual(tuple(pixels[13, 21, :3]), (200, 120, 40))   # shift 5, the newest

    def test_a_filled_picture_wakes_the_widget(self):
        woken = threading.Event()
        presenter, context = make_presenter(fill_thread=True)
        self.addCleanup(presenter.lose_device)
        presenter.ring.notify = woken.set
        presenter.offer(checker_image(), (0, 0, 64, 48), (0, 0, 64, 48), 1)
        self.assertTrue(woken.wait(5))

    def test_float_frames_play_through_the_ring_into_one_float_texture(self):
        from nodebased import carddisplay
        settings = carddisplay.settings_from(0.0, "RGB", "black", "Linear", None)
        for value in (0.25, 0.5, 0.75):
            frame = np.full((48, 64, 4), value, np.float32)
            frame[..., 3] = 1.0
            self.assertTrue(self.presenter.offer_linear(frame, settings, (0, 0, 64, 48), (0, 0, 64, 48), 1))
            pixels = self.settle_and_draw()
            self.assertLessEqual(abs(int(pixels[60, 80, 0]) - round(value * 255)), 1)
        self.assertEqual(self.presenter.counters["linear_uploads"], 3)
        self.assertEqual(self.presenter.counters["texture_creates"], 1)

    def test_a_large_frame_and_a_crop_alternating_keep_one_texture_each(self):
        # During playback the 4K frame (half floats) and the Viewer's visible crop (float32) alternate.
        from nodebased import carddisplay
        settings = carddisplay.settings_from(0.0, "RGB", "black", "Linear", None)
        with unittest.mock.patch.object(dv, "HALF_ABOVE", 2000):
            for step in range(6):
                size = (64, 48) if step % 2 == 0 else (32, 24)
                value = 0.25 if step % 2 == 0 else 0.75
                frame = np.full((size[1], size[0], 4), value, np.float32)
                frame[..., 3] = 1.0
                rect = (0, 0, *size)
                self.presenter.offer_linear(frame, settings, rect, rect, step)
                pixels = self.settle_and_draw((size[0] / 2, size[1] / 2, 2.0))
                self.assertLessEqual(abs(int(pixels[60, 80, 0]) - round(value * 255)), 1, f"step {step}")
        self.assertEqual(sorted(self.presenter.linear_by_format), ["rgba16float", "rgba32float"])
        self.assertEqual(self.presenter.counters["texture_creates"], 2)

    def test_close_waits_for_a_fill_in_flight(self):
        gate = threading.Event()
        original = dv.fill_rows

        def held(*args):
            gate.wait(5)
            return original(*args)
        with unittest.mock.patch.object(dv, "fill_rows", held):
            self.presenter.offer(checker_image(), (0, 0, 64, 48), (0, 0, 64, 48), 1)
            threading.Timer(0.2, gate.set).start()
            self.presenter.lose_device()                        # must not unmap under the filler
        self.assertIsNone(self.presenter.device)
        self.assertIsNone(self.presenter.ring)


@needs_wgpu
class FourKTests(unittest.TestCase):
    def test_a_4k_float_picture_goes_up_through_one_reused_buffer(self):
        presenter, context = make_presenter(size=(320, 180))
        self.addCleanup(presenter.lose_device)
        from nodebased import carddisplay
        settings = carddisplay.settings_from(0.0, "RGB", "black", "Linear", None)
        frame = np.zeros((2160, 3840, 4), np.float32)
        frame[:1080, :1920] = (1.0, 0.0, 0.0, 1.0)              # top-left quarter red, the rest black
        frame[1080:, 1920:] = (0.0, 0.0, 1.0, 1.0)              # bottom-right quarter blue
        for _ in range(2):
            self.assertTrue(presenter.offer_linear(frame, settings, (0, 0, 3840, 2160), (0, 0, 3840, 2160), 1))
            presenter.draw(dv.fit_view(context.size, (0, 0, 3840, 2160)), context.size)
        pixels = context.read()
        self.assertGreater(int(pixels[40, 80, 0]), 250)
        self.assertGreater(int(pixels[140, 240, 2]), 250)
        self.assertEqual(presenter.counters["linear_uploads"], 2)
        self.assertEqual(presenter.counters["texture_creates"], 1)
        self.assertLessEqual(presenter.counters["buffer_creates"], 2)
        # 4K float32 goes up as half floats (HALF_ABOVE): one 66 MB buffer instead of 133 MB.
        self.assertEqual(presenter.linear_texture.format, "rgba16float")
        self.assertEqual(presenter.staging_bytes() // presenter.counters["buffer_creates"], 3840 * 8 * 2160)

    def test_a_too_large_picture_is_refused_and_the_last_picture_stays(self):
        presenter, context = make_presenter()
        self.addCleanup(presenter.lose_device)
        self.assertTrue(presenter.offer(checker_image(), (0, 0, 64, 48), (0, 0, 64, 48), 1))
        presenter.draw((32.0, 24.0, 2.0), context.size)
        presenter.limits = dict(presenter.limits, **{"max-texture-dimension-2d": 64})
        with self.assertRaises(dv.TooLarge) as caught:
            presenter.offer(checker_image(width=96, height=72), (0, 0, 96, 72), (0, 0, 96, 72), 2)
        self.assertIn("96 x 72", str(caught.exception))
        self.assertIn("64 pixels on a side", str(caught.exception))
        self.assertEqual(presenter.refusal, str(caught.exception))
        self.assertIsNone(presenter.ledger.pending)
        presenter.draw((32.0, 24.0, 2.0), context.size)
        self.assertEqual(tuple(context.read()[13, 21, :3]), (20, 60, 180))   # the earlier picture, still shown
        self.assertEqual(presenter.counters["uploads"], 1)

    def test_the_widget_reports_a_refusal_and_stays_up(self):
        widget = dv.make_widget()
        self.addCleanup(widget.close)
        presenter, context = make_presenter(size=(64, 48))
        self.addCleanup(presenter.lose_device)
        widget.presenter = presenter
        presenter.limits = dict(presenter.limits, **{"max-texture-dimension-2d": 64})
        self.assertFalse(widget.submit(checker_image(width=96, height=72), (0, 0, 96, 72)))
        self.assertIn("96 x 72", widget.refusal)
        self.assertIsNone(widget.error)
        self.assertTrue(widget.submit(checker_image(), (0, 0, 64, 48)))
        self.assertEqual(widget.refusal, "")


@needs_wgpu
class RetainedMemoryTests(unittest.TestCase):
    def test_only_the_newest_picture_is_retained_and_closing_releases_it(self):
        widget = dv.make_widget()
        self.addCleanup(widget.close)
        presenter, context = make_presenter(size=(64, 48))
        widget.presenter = presenter
        widget._physical = lambda: (64, 48)
        for width in (64, 32):
            image = checker_image(width=width, height=48)
            widget.submit(image, (0, 0, width, 48))
            widget._draw()
        self.assertEqual(presenter.retained_bytes(), checker_image(width=32, height=48).sizeInBytes())
        self.assertGreater(presenter.staging_bytes(), 0)
        widget.release()
        self.assertIsNone(widget.presenter)
        self.assertEqual(widget.retained_bytes(), 0)
        self.assertIsNone(presenter.last_frame)
        self.assertIsNone(presenter.device)


class StubWithMemory(StubPresenter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.limits = {"max-texture-dimension-2d": 64, "max-buffer-size": 1 << 30}
        self.adapter_info = {"device": "Stub card"}
        self.released = 0

    def retained_bytes(self):
        return 3 * 1048576

    def staging_bytes(self):
        return 6 * 1048576

    def b_texture_bytes(self):
        return 2 * 1048576

    def lose_device(self, wait=True):
        self.released += 1


class WindowPlaybackTests(unittest.TestCase):
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

    def test_a_frame_too_large_for_the_card_is_converted_on_the_worker(self):
        w = self.window
        request = types.SimpleNamespace(playing=True, document=w.dispatcher.document)
        w._direct_linear_on = True
        w._direct_limits = {"max-texture-dimension-2d": 4096, "max-buffer-size": 1 << 30, "device": "Stub card"}
        fits = np.zeros((2160, 3840, 4), np.float16)
        wide = np.zeros((10, 5000, 4), np.float16)
        self.assertTrue(w._direct_linear_for(request, "sRGB", fits))
        self.assertFalse(w._direct_linear_for(request, "sRGB", wide))
        w._direct_limits = None
        self.assertTrue(w._direct_linear_for(request, "sRGB", wide))   # no card yet: the dock decides

    @needs_wgpu
    def test_closing_the_dock_releases_the_picture_and_the_inspector_says_so(self):
        w = self.window
        with unittest.mock.patch.object(dv, "RetainedPresenter", StubWithMemory), stub_surface():
            w.direct_action.setChecked(True)
            APP.processEvents()
            self.assertIsNotNone(w.direct_widget)
            w.direct_widget.presenter = presenter = StubWithMemory(None)
            w._refresh_direct_memory()
            self.assertIn("retained picture 3.0 MB", w.cache_inspector.direct_memory.text())
            self.assertIn("upload buffers 6.0 MB", w.cache_inspector.direct_memory.text())
            self.assertIn("B texture 2.0 MB", w.cache_inspector.direct_memory.text())
            w.direct_action.setChecked(False)
            APP.processEvents()
        self.assertEqual(presenter.released, 1)
        self.assertIsNone(w.direct_widget.presenter)
        self.assertIn("closed", w.cache_inspector.direct_memory.text())

    @needs_wgpu
    def test_a_refused_picture_is_named_in_the_status_bar(self):
        w = self.window
        with unittest.mock.patch.object(dv, "RetainedPresenter", StubWithMemory), stub_surface():
            w.direct_action.setChecked(True)
            APP.processEvents()
            widget = w.direct_widget
            widget.presenter = StubWithMemory(None)

            def refuse(*args, **kwargs):
                raise dv.TooLarge("Direct picture: the picture is 96 x 72 pixels and Stub card takes at most 64")
            widget.presenter.offer = refuse
            w._offer_direct(checker_image(width=96, height=72), 1, None, w.viewer.sceneRect(), None)
        self.assertIn("96 x 72", w.statusBar().currentMessage())
        self.assertEqual(w._direct_limits["device"], "Stub card")


if __name__ == "__main__":
    unittest.main()
