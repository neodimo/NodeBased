"""The direct viewer's display transform on the card agrees with `to_qimage` within 1/255.

Run on each adapter: the default (discrete), then through force-adapter.py with integrated and cpu
(the card is the adapter `gpu3d._state()` reports, so forcing the app's adapter forces this test's).
"""
import unittest

import numpy as np

import tests.isolation  # noqa: F401  (Qt settings stay out of the real user file)

from PySide6.QtWidgets import QApplication

from nodebased import carddisplay
from nodebased import directviewer as dv
from nodebased.imaging import to_qimage
from tests.test_directviewer import FakeContext, needs_wgpu


def test_frames(width=96, height=64):
    """Premultiplied frames: linear ramps, HDR highlights above 1, alpha edges, random colour."""
    rng = np.random.default_rng(7)
    ramps = np.zeros((height, width, 4), np.float32)
    ramps[..., 0] = np.linspace(0.0, 1.0, width)[None]
    ramps[..., 1] = np.linspace(0.0, 1.0, width)[None] ** 2
    ramps[..., 2] = np.linspace(1.0, 0.0, width)[None]
    ramps[..., 3] = 1.0
    hdr = np.ones((height, width, 4), np.float32)
    hdr[..., :3] = np.linspace(0.0, 64.0, width, dtype=np.float32)[None, :, None] ** np.float32(1.3)
    hdr[: height // 2, :, 1] *= 0.2   # saturated highlights, the gamut-compression case
    hdr[height // 2:, ::3, 2] = -0.05  # a little below zero, as grades leave it
    edges = np.zeros((height, width, 4), np.float32)
    edges[..., 3] = np.clip(np.linspace(-0.25, 1.25, height, dtype=np.float32), 0, 1)[:, None]
    edges[..., :3] = rng.random((height, width, 3), np.float32) * 3.0
    edges[..., :3] *= edges[..., 3:4]
    noise = rng.random((height, width, 4), np.float32) * np.float32([6, 6, 6, 1])
    noise[..., :3] *= noise[..., 3:4]
    return {"ramps": ramps, "hdr": hdr, "alpha edges": edges, "noise": noise}


def reference(frame, settings):
    exposure, channel, background, view, gamma, zebra = settings
    image = to_qimage(frame, exposure, channel, background=background, view=view,
                      look={"gamma": gamma, "zebra": zebra})
    height, width = frame.shape[:2]
    rows = np.frombuffer(image.constBits(), np.uint8, count=image.sizeInBytes()).reshape(height, image.bytesPerLine())
    return rows[:, :width * 3].reshape(height, width, 3).copy()   # the QImage owns the bytes


def card_adapter():
    """The adapter the app's renderer uses (`gpu3d._state`, which force-adapter.py redirects)."""
    import wgpu
    from nodebased import gpu3d
    try:
        name = gpu3d._state()["info"].get("device")
    except Exception:
        return None
    adapters = list(wgpu.gpu.enumerate_adapters_sync())
    return next((a for a in adapters if a.info.get("device") == name), None)


@needs_wgpu
class CardDisplayAgreementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        adapter = card_adapter()
        if adapter is None:
            raise unittest.SkipTest("no wgpu adapter")
        cls.frames = test_frames()
        cls.size = (96, 64)
        cls.context = FakeContext(cls.size)
        cls.presenter = dv.RetainedPresenter(cls.context, adapter=adapter, overlay=False, fill_thread=False)

    @classmethod
    def tearDownClass(cls):
        cls.presenter.lose_device()

    def card(self, frame, settings):
        width, height = self.size
        self.assertTrue(self.presenter.offer_linear(frame, settings, (0, 0, width, height), (0, 0, width, height)))
        self.presenter.draw((width / 2.0, height / 2.0, 1.0), self.size)
        return self.context.read()[..., :3]

    def check(self, views, channels, backgrounds, looks, exposures=(0.0,)):
        for name, frame in self.frames.items():
            for view in views:
                for channel in channels:
                    for background in backgrounds:
                        for gamma, zebra in looks:
                            for exposure in exposures:
                                settings = carddisplay.settings_from(exposure, channel, background, view,
                                                                     {"gamma": gamma, "zebra": zebra})
                                with self.subTest(frame=name, settings=settings):
                                    diff = np.abs(self.card(frame, settings).astype(int) - reference(frame, settings).astype(int))
                                    self.assertLessEqual(int(diff.max()), 1)

    def test_every_channel_mode_through_each_view(self):
        self.check(("ACES 2.0", "sRGB", "Linear"), carddisplay.CHANNELS, ("black",), ((1.0, False),))

    def test_checker_exposure_gamma_and_clipping_warning(self):
        self.check(("ACES 2.0", "sRGB"), ("RGB", "A"), ("checker", "black"), ((1.0, False), (2.2, True)),
                   exposures=(-1.5, 2.0))

    def test_one_upload_per_frame_and_no_cpu_conversion(self):
        before = dict(self.presenter.counters)
        settings = carddisplay.settings_from(0.0, "RGB", "black", "ACES 2.0")
        self.card(self.frames["ramps"], settings)
        self.presenter.draw((48.0, 32.0, 2.0), self.size)       # a view change alone uploads nothing
        self.assertEqual(self.presenter.counters["linear_uploads"], before.get("linear_uploads", 0) + 1)
        self.assertEqual(self.presenter.counters["conversions"], before.get("conversions", 0))


def compare_reference(a, b, mode, settings, wipe, size):
    """What the Viewer builds on the processor for a compare mode: B alone, A or B on either side of the wipe
    line (decided at each pixel's centre), or the composite of over, under, minus and difference."""
    from nodebased import compare
    a, b = a.astype(np.float32), b.astype(np.float32)   # the card computes in float32; so does the Viewer for B
    if mode == "B only":
        return reference(b, settings)
    if mode == "wipe":
        cx, cy, nx, ny = wipe
        ys, xs = np.mgrid[0:size[1], 0:size[0]]
        on_b = ((xs + 0.5 - cx) * nx + (ys + 0.5 - cy) * ny) > 0
        return np.where(on_b[..., None], reference(b, settings), reference(a, settings))
    return reference(compare.combine(mode, a.astype(np.float32), b.astype(np.float32)), settings)


def wipe_line(cx, cy, angle):
    from nodebased import compare
    return (cx, cy, *compare.wipe_normal(angle))


COMPARE_MODES = ("B only", "wipe", "over", "under", "minus", "difference")


def pair_frames():
    """(A, B) pairs: different content, different alpha, one pair with a half-float B as a cache frame comes."""
    frames = test_frames()
    return {"ramps/noise": (frames["ramps"], frames["noise"]),
            "edges/hdr": (frames["alpha edges"], frames["hdr"]),
            "noise/edges half": (frames["noise"], frames["alpha edges"].astype(np.float16))}


@needs_wgpu
class CardCompareAgreementTests(unittest.TestCase):
    """A/B compare on the card agrees with the processor path within 1/255 for every compare mode."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        adapter = card_adapter()
        if adapter is None:
            raise unittest.SkipTest("no wgpu adapter")
        cls.pairs = pair_frames()
        cls.size = (96, 64)
        cls.context = FakeContext(cls.size)
        cls.presenter = dv.RetainedPresenter(cls.context, adapter=adapter, overlay=False, fill_thread=False)

    @classmethod
    def tearDownClass(cls):
        cls.presenter.lose_device()

    def card(self, a, b, mode, settings, wipe):
        width, height = self.size
        rect = (0, 0, width, height)
        self.assertTrue(self.presenter.offer_linear(a, settings, rect, rect, None, b, mode))
        self.presenter.draw((width / 2.0, height / 2.0, 1.0), self.size, wipe)
        return self.context.read()[..., :3]

    def check(self, modes, views, channels, backgrounds, looks, exposures=(0.0,), wipes=((40.3, 27.1, 0.0),)):
        for name, (a, b) in self.pairs.items():
            for mode in modes:
                for view in views:
                    for channel in channels:
                        for background in backgrounds:
                            for gamma, zebra in looks:
                                for exposure in exposures:
                                    for cx, cy, angle in (wipes if mode == "wipe" else wipes[:1]):
                                        wipe = wipe_line(cx, cy, angle)
                                        settings = carddisplay.settings_from(
                                            exposure, channel, background, view, {"gamma": gamma, "zebra": zebra})
                                        with self.subTest(pair=name, mode=mode, settings=settings, wipe=wipe):
                                            got = self.card(a, b, mode, settings, wipe).astype(int)
                                            want = compare_reference(a, b, mode, settings, wipe, self.size).astype(int)
                                            self.assertLessEqual(int(np.abs(got - want).max()), 1)

    def test_every_compare_mode_through_each_view(self):
        self.check(COMPARE_MODES, ("sRGB", "ACES 2.0", "Linear"), ("RGB",), ("black",), ((1.0, False),))

    def test_the_wipe_at_several_positions_and_angles(self):
        # 0 degrees is a vertical line (A left, B right); 90 a horizontal split; the others are slanted.
        wipes = ((40.3, 27.1, 0.0), (10.7, 50.2, 33.0), (70.4, 20.6, 90.0), (48.2, 32.9, 200.0), (48.2, 32.9, 315.0))
        self.check(("wipe",), ("sRGB", "ACES 2.0"), ("RGB",), ("black",), ((1.0, False),), wipes=wipes)

    def test_checker_exposure_gamma_clipping_warning_and_channels(self):
        self.check(COMPARE_MODES, ("ACES 2.0", "sRGB"), ("RGB", "A", "G"), ("checker", "black"),
                   ((1.0, False), (2.2, True)), exposures=(-1.0, 1.5))

    def test_a_wipe_drag_changes_the_uniform_and_uploads_nothing(self):
        a, b = self.pairs["ramps/noise"]
        settings = carddisplay.settings_from(0.0, "RGB", "black", "sRGB")
        self.card(a, b, "wipe", settings, wipe_line(40.3, 27.1, 0.0))
        before = dict(self.presenter.counters)
        for cx in (20.3, 60.7, 80.1):
            wipe = wipe_line(cx, 27.1, 0.0)
            self.presenter.draw((48.0, 32.0, 1.0), self.size, wipe)
            diff = np.abs(self.context.read()[..., :3].astype(int)
                          - compare_reference(a, b, "wipe", settings, wipe, self.size).astype(int))
            self.assertLessEqual(int(diff.max()), 1)
        for key in ("linear_uploads", "b_uploads", "upload_bytes", "conversions"):
            self.assertEqual(self.presenter.counters[key], before.get(key, 0), key)

    def test_a_picture_without_b_after_one_with_b_shows_a_alone(self):
        a, b = self.pairs["ramps/noise"]
        settings = carddisplay.settings_from(0.0, "RGB", "black", "sRGB")
        self.card(a, b, "difference", settings, None)
        alone = self.card(a, None, "A only", settings, None).astype(int)
        self.assertLessEqual(int(np.abs(alone - reference(a, settings).astype(int)).max()), 1)
        self.assertEqual(self.presenter.compare, "A only")

    def test_the_b_texture_is_counted_and_reused(self):
        a, b = self.pairs["ramps/noise"]
        settings = carddisplay.settings_from(0.0, "RGB", "black", "sRGB")
        self.card(a, b, "over", settings, None)
        made = self.presenter.counters["b_texture_creates"]
        self.assertGreater(self.presenter.b_texture_bytes(), 0)
        self.card(a, b, "under", settings, None)
        self.assertEqual(self.presenter.counters["b_texture_creates"], made)
        width, height = self.size
        self.assertGreaterEqual(self.presenter.b_texture_bytes(), width * height * 16)

    def test_a_pair_that_does_not_fit_the_card_is_refused_by_name(self):
        a, b = self.pairs["ramps/noise"]
        settings = carddisplay.settings_from(0.0, "RGB", "black", "sRGB")
        rect = (0, 0, 96, 64)
        saved = dict(self.presenter.limits)
        try:
            one = dv.padded_row(96, 16) * 64
            self.presenter.limits["max-buffer-size"] = one + one // 2     # A fits alone, A and B together do not
            self.assertTrue(self.presenter.offer_linear(a, settings, rect, rect))
            self.presenter.draw((48.0, 32.0, 1.0), self.size)
            with self.assertRaises(dv.TooLarge) as caught:
                self.presenter.offer_linear(a, settings, rect, rect, None, b, "wipe")
            self.assertIn("A and B", str(caught.exception))
            self.assertIn("MB", str(caught.exception))
            self.presenter.limits["max-buffer-size"] = saved["max-buffer-size"]
            with self.assertRaises(dv.TooLarge) as caught:
                self.presenter.offer_linear(a, settings, rect, rect, None, b[:32], "wipe")
            self.assertIn("B picture", str(caught.exception))
        finally:
            self.presenter.limits.update(saved)


class ShaderSourceTests(unittest.TestCase):
    def test_aces_tables_move_to_the_storage_buffer(self):
        # Constant arrays in the shader cost 17 ms a draw (naga copies them per dynamic index).
        source = carddisplay.fragment_source("ACES 2.0")
        self.assertNotIn("sampler1D", source)
        self.assertNotRegex(source, r"const\s+float\s+\w+\s*\[")
        self.assertEqual(carddisplay.view_tables("ACES 2.0").shape, (363 * 3, 4))

    def test_unknown_view_and_channel_are_unsupported(self):
        with self.assertRaises(carddisplay.Unsupported):
            carddisplay.view_source("No such view")
        with self.assertRaises(carddisplay.Unsupported):
            carddisplay.settings_from(channel="Z")

    def test_a_3d_lookup_view_falls_back(self):
        class Desc:
            def get3DTextures(self):
                return [object()]
        with self.assertRaises(carddisplay.Unsupported):
            carddisplay._move_tables("", Desc())


def fake_widget_class():
    from PySide6.QtCore import Signal
    from PySide6.QtWidgets import QWidget

    class FakeDirectWidget(QWidget):
        """The direct picture without a screen surface: records what the window hands it."""
        failed = Signal(str)
        accept_linear = True

        def __init__(self, parent=None):
            super().__init__(parent)
            self.error = None
            self.linear, self.images, self.pairs = [], [], []
            self.compare_provider = None

        def submit(self, image, rect, format_rect=None, generation=None):
            self.images.append(image)
            return True

        def submit_linear(self, frame, settings, rect, format_rect=None, generation=None, frame_b=None,
                          compare="A only"):
            if not self.accept_linear:
                return False
            self.linear.append((frame, settings, tuple(rect)))
            self.pairs.append((frame_b, compare))
            return True

        def overlay_dirty(self):
            self.dirtied = getattr(self, "dirtied", 0) + 1

        def release(self):
            pass

    return FakeDirectWidget


class DirectWindowCase(unittest.TestCase):
    """A real Window with the direct dock open on a fake widget that records what the window hands it."""

    document = staticmethod(lambda: None)

    def setUp(self):
        import uuid
        import unittest.mock
        from nodebased.app import Window
        from tests.waiting import wait_until
        QApplication.instance() or QApplication([])
        self.window = Window(self.document(), agent_name="nodebased-test-" + uuid.uuid4().hex)
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))
        with unittest.mock.patch.object(dv, "make_widget", fake_widget_class()):
            self.window.direct_action.setChecked(True)
        self.widget = self.window.direct_widget
        self.assertIsNotNone(self.widget)

    def tearDown(self):
        from tests.test_desktop import release_window
        self.addCleanup(release_window, self)
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        QApplication.instance().processEvents()

    def request(self, playing=True):
        import copy
        import types
        return types.SimpleNamespace(playing=playing, document=copy.deepcopy(self.window.dispatcher.document))


class PlaybackRoutingTests(DirectWindowCase):
    """With the direct picture up, playback and paused frames go to the card as floats and skip `to_qimage`."""

    def test_worker_routes_playing_and_paused_frames_with_or_without_a_b_buffer(self):
        w = self.window
        self.assertTrue(w._direct_linear_for(self.request(), "ACES 2.0"))
        self.assertTrue(w._direct_linear_for(self.request(playing=False), "ACES 2.0"))
        w.command({"op": "viewer_input", "slot": 2, "id": next(iter(w.dispatcher.document["nodes"])), "activate": False})
        w.command({"op": "viewer_compare", "b": 2, "mode": "wipe"})
        self.assertTrue(w._direct_linear_for(self.request(playing=False), "ACES 2.0"))   # B rides along on the card
        self.assertFalse(w._direct_linear_for(self.request(), "No such view"))
        w.direct_action.setChecked(False)
        self.assertFalse(w._direct_linear_for(self.request(), "ACES 2.0"))
        self.assertFalse(w._direct_linear_for(self.request(playing=False), "ACES 2.0"))

    def test_lazy_picture_goes_to_the_card_and_is_never_converted(self):
        w = self.window
        before = w.viewer.last_display_image
        frame = test_frames()["ramps"]
        picture = carddisplay.LazyPicture(frame, 0.0, "RGB", "black", "ACES 2.0", None)
        w._show_image(picture, 1, None, generation=99)
        self.assertEqual(len(self.widget.linear), 1)
        self.assertIs(self.widget.linear[0][0], frame)
        self.assertEqual(self.widget.linear[0][2], (0.0, 0.0, 96.0, 64.0))
        self.assertFalse(picture.built())
        self.assertIs(w.viewer.last_display_image, before)     # the Viewer keeps its last picture
        self.assertIs(w.direct_lazy, picture)

    def test_a_view_the_card_refuses_is_built_on_the_processor(self):
        w = self.window
        self.widget.accept_linear = False
        picture = carddisplay.LazyPicture(test_frames()["ramps"], 0.0, "RGB", "black", "ACES 2.0", None)
        w._show_image(picture, 1, None, generation=99)
        self.assertTrue(picture.built())
        self.assertIs(w.viewer.last_display_image, picture.image())
        self.assertIs(self.widget.images[-1], picture.image())   # and the dock gets the 8-bit picture


class PausedPictureTests(DirectWindowCase):
    """Outside playback the card shows the float frame and the Viewer builds no 8-bit picture."""

    def lazy(self, name="ramps", tier=1):
        return carddisplay.LazyPicture(test_frames()[name], 0.0, "RGB", "black", "ACES 2.0", None)

    def test_a_paused_picture_parks_the_viewer_and_builds_nothing(self):
        w = self.window
        self.assertFalse(w.playing)
        picture = self.lazy()
        w._show_image(picture, 1, None, generation=99)
        self.assertFalse(picture.built())
        self.assertEqual(len(self.widget.linear), 1)
        self.assertEqual(w.viewer.last_frame_size, (96, 64))
        self.assertEqual((w.viewer.sceneRect().width(), w.viewer.sceneRect().height()), (96.0, 64.0))
        self.assertEqual(w.viewer.format_rect.size().toTuple(), (96.0, 64.0))
        self.assertEqual(len(w.viewer.scene().items()), 1)           # one black rectangle, no picture
        self.assertIsNone(w.viewer.readout_geometry)
        # Something asks for 8 bits: the output preview, once, from the same frame.
        image = w.display_image()
        self.assertIs(image, picture.image())
        self.assertTrue(picture.built())

    def test_progressive_tiers_each_reach_the_card_as_floats_and_replace_each_other(self):
        w = self.window
        pictures = [self.lazy() for _ in range(3)]
        for tier, picture in zip((8, 4, 1), pictures):
            w._show_image(picture, tier, None, generation=100 + tier)
        self.assertEqual(len(self.widget.linear), 3)
        self.assertEqual([entry[2][2] for entry in self.widget.linear], [96.0 * 8, 96.0 * 4, 96.0])
        self.assertFalse(any(picture.built() for picture in pictures))
        self.assertIs(w.direct_lazy, pictures[-1])

    def test_closing_the_dock_repaints_the_viewer_from_the_float_frame(self):
        w = self.window
        picture = self.lazy()
        w._show_image(picture, 1, None, generation=99)
        self.assertFalse(picture.built())
        w.direct_action.setChecked(False)
        QApplication.instance().processEvents()
        self.assertTrue(picture.built())
        self.assertIs(w.viewer.last_display_image, picture.image())
        self.assertTrue(any(hasattr(item, "pixmap") for item in w.viewer.scene().items()))

    def test_an_edit_while_paused_reaches_the_card_without_to_qimage(self):
        import unittest.mock
        from tests.waiting import wait_until
        import nodebased.app as app_module
        w = self.window
        calls = []
        real = app_module.to_qimage

        def counting(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)
        before = len(self.widget.linear)
        with unittest.mock.patch.object(app_module, "to_qimage", counting):
            w.request_preview()
            self.assertTrue(wait_until(lambda: len(self.widget.linear) > before, timeout=20))
        self.assertEqual(calls, [])
        self.assertFalse(w.direct_lazy.built())


class CompareOnTheCardTests(DirectWindowCase):
    """With a B buffer chosen, the window hands the card A and B as floats and builds neither as 8 bits."""

    document = staticmethod(lambda: __import__("tests.test_viewer_compare", fromlist=["x"]).constant_document())

    def settle(self, mode):
        """Until the card has been handed a picture with this compare mode (earlier renders may still land first)."""
        from tests.waiting import wait_until
        self.assertTrue(wait_until(lambda: bool(self.widget.pairs) and self.widget.pairs[-1][1] == mode, timeout=30))
        QApplication.instance().processEvents()

    def choose(self, mode):
        w = self.window
        w.command({"op": "viewer_input", "slot": 2, "id": "b", "activate": False})
        w.command({"op": "viewer_compare", "b": 2, "mode": mode})

    def count_conversions(self, action):
        import unittest.mock
        import nodebased.app as app_module
        calls, real = [], app_module.to_qimage

        def counting(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)
        with unittest.mock.patch.object(app_module, "to_qimage", counting):
            action()
        return len(calls)

    def test_every_compare_mode_reaches_the_card_with_b_and_builds_nothing(self):
        w = self.window
        for mode in ("wipe", "B only", "over", "under", "minus", "difference"):
            self.choose(mode)
            conversions = self.count_conversions(lambda: self.settle(mode))
            frame_b, compare = self.widget.pairs[-1]
            with self.subTest(mode=mode):
                self.assertEqual(compare, mode)
                self.assertIsNotNone(frame_b)
                self.assertEqual(frame_b.shape, self.widget.linear[-1][0].shape)
                self.assertEqual(conversions, 0)
                self.assertFalse(w.direct_lazy.built())
                self.assertIsNotNone(w.frame_b)                  # the readout still reads B from the float frame
        self.assertEqual(self.window.viewer.card_wipe, False)       # the last mode was not a wipe

    def test_the_wipe_line_and_handles_stay_on_the_viewer_and_the_card_draws_b(self):
        w = self.window
        self.choose("wipe")
        self.settle("wipe")
        viewer = w.viewer
        self.assertTrue(viewer.card_wipe)
        self.assertIsNone(viewer.wipe_pixmap)
        self.assertTrue(viewer._wipe_active())
        cx, cy, nx, ny = viewer.wipe_line()
        rect = viewer.format_rect
        self.assertAlmostEqual(cx, rect.left() + rect.width() / 2)
        self.assertAlmostEqual(cy, rect.top() + rect.height() / 2)
        self.assertEqual((round(nx, 6), round(ny, 6)), (1.0, 0.0))
        viewer.wipe["angle"] = 90.0
        self.assertEqual(tuple(round(v, 6) for v in viewer.wipe_line()[2:]), (0.0, 1.0))

    def test_a_pair_too_big_for_the_card_is_named_and_the_viewer_shows_it(self):
        from tests.waiting import wait_until
        w = self.window
        w._direct_limits = {"max-texture-dimension-2d": 8192, "max-buffer-size": 160 * 80 * 16, "device": "Stub card"}
        self.choose("wipe")
        self.assertTrue(wait_until(lambda: "compare on the Viewer" in w.viewer_info.text(), timeout=30))
        text = w.viewer_info.text()
        self.assertIn("A and B", text)
        self.assertIn("Stub card", text)
        self.assertIsNotNone(w.viewer.wipe_pixmap)         # the Viewer's own wipe, built on the processor
        self.assertFalse(w.viewer.card_wipe)

    def test_a_card_that_refuses_the_pair_falls_back_to_the_viewer_wipe(self):
        from tests.waiting import wait_until
        w = self.window
        self.widget.accept_linear = False
        self.choose("wipe")
        self.assertTrue(wait_until(lambda: w.viewer.wipe_pixmap is not None, timeout=30))
        self.assertFalse(w.viewer.card_wipe)

    def test_dragging_the_wipe_centre_on_the_direct_picture_moves_the_cards_line(self):
        from PySide6.QtCore import Qt
        w = self.window
        self.choose("wipe")
        self.settle("wipe")
        viewer = w.viewer
        start = viewer.wipe_line()
        centre = viewer._wipe_scene_geometry()[0]
        none = Qt.KeyboardModifier.NoModifier
        undo = len(w.dispatcher.undo_stack)
        self.assertTrue(w._direct_edit("press", (centre.x(), centre.y()), Qt.MouseButton.LeftButton, none, 1.0))
        self.assertTrue(w._direct_edit("move", (centre.x() + 24, centre.y() - 10), None, none, 1.0))
        moved = viewer.wipe_line()
        self.assertAlmostEqual(moved[0], start[0] + 24, places=3)
        self.assertAlmostEqual(moved[1], start[1] - 10, places=3)
        self.assertEqual(moved[2:], start[2:])
        self.assertGreater(getattr(self.widget, "dirtied", 0), 0)          # the overlay (the line) is painted again
        w._direct_edit("release", (centre.x() + 24, centre.y() - 10), Qt.MouseButton.LeftButton, none, 1.0)
        self.assertEqual(len(w.dispatcher.undo_stack), undo)               # the wipe is view state: no undo step
        self.assertEqual(len(self.widget.linear), len(self.widget.pairs))  # and no new picture was needed

    def test_closing_the_dock_gives_the_viewer_its_wipe_back(self):
        w = self.window
        self.choose("wipe")
        self.settle("wipe")
        self.assertIsNone(w.viewer.wipe_pixmap)
        w.direct_action.setChecked(False)          # synchronous: a render still in flight may land afterwards
        self.assertIsNotNone(w.viewer.wipe_pixmap)
        self.assertFalse(w.viewer.card_wipe)


if __name__ == "__main__":
    unittest.main()
